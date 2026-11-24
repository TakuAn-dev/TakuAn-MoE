# SwiftLM (SharpAI) — Expert Streaming, TurboQuant KV, Memory Management

Analyzed: 2026-11-22, shallow clone at `refs/SwiftLM`. All line refs are to this clone.
Repo = SwiftLM CLI/server (Swift, Hummingbird HTTP, OpenAI-compatible) + SwiftBuddy app +
MLXInferenceCore + DFlash speculative decoding. It is a Swift wrapper over **MLX** via
**custom forks** of mlx-swift / mlx-swift-lm (and ultimately `SharpAI/mlx`, `SharpAI/mlx-c`).

---

## 0. Overview + submodule status

- `.gitmodules`: `mlx-swift` → `SharpAI/mlx-swift` @`5d78578`, `mlx-swift-lm` →
  `SharpAI/mlx-swift-lm` @`69346d0`. **Both submodules are uninitialized/empty in this
  clone** (`git submodule status` shows `-`; directories empty). There is also a top-level
  `Packages/mlx-swift-lm` directory (empty). Everything below the wrapper API surface —
  `ExpertStreamerManager`, `streamedGatherMM`, `moe_stream_op.cpp`, `ssd_streamer.mm`,
  `fence.air`, `fast_turbo.cpp`, `KVCacheSimple.turboQuantEnabled`, `MLX.Memory.*`,
  `MLXFast.setPrefetchEnabled` — lives in the missing submodules and **cannot be read here**.
  This doc therefore documents the wrapper-side mechanism (activation, budgets, flags,
  orchestration) plus their own architecture docs (`docs/moe_ssd_streaming_architecture.md`,
  `docs/turboquant_hybrid_architecture.md`, `docs/PAPPS_Research_Review.md`, `README.md`),
  which describe the kernel/C++ side. Items from those docs are marked *[their docs]*.
- README "Why We Forked Apple MLX" (README.md:549-557): custom forks add
  **out-of-core memory-mapped execution**, "streaming tensor blocks directly from the SSD
  (NVMe) to the GPU via custom Metal kernels (`ssd_streamer.mm` and `fence.air`)".
- Grep over all of `Sources/` for `mmap|madvise|F_NOCACHE|dispatch_io|purgeable|memoryScope|
  setPurgeable|vm_|jetsam`: **no direct Darwin-VM calls anywhere**. Memory work is done via
  MLX allocator knobs (`Memory.cacheLimit`, `Memory.memoryLimit`), `task_info` telemetry,
  and `DispatchSource` memory pressure — plus fork-side pread machinery referenced only in
  comments. There is **no `madvise`/`F_NOCACHE`/purgeable-memory trick in this repo**.

---

## 1. `--stream-experts` — the headline mechanism

### 1.1 What it is

CLI flag (`Sources/SwiftLM/Server.swift:656-657`): *"Enable SSD expert streaming for MoE
models (Flash-MoE style memory-mapping)"*. Companion flag `--ssd-prefetch`
(Server.swift:659-660): *"16-worker background SSD thread pool queue (PAPPS). Requires
--stream-experts."*

Only MoE models qualify: Server.swift:816-829 profiles `config.json` and **disables the
flag** if `!profile.isMoE` (prevents wrong memory limits). Dense models never stream
(StreamingDirectory.swift:198-205 hands partial dense copies back to the normal loader).

### 1.2 Exact activation sequence (Server.swift `runInner`)

1. **Open-file limit raise**: `setrlimit(RLIMIT_NOFILE, 4096)` (Server.swift:720-728) —
   182-shard models + draft + metallib exceed the macOS default 256 FDs.
2. **Watchdog mitigation env**: `setenv("MLX_MAX_OPS_PER_BUFFER","50",1)` *before MLX
   initializes* (Server.swift:730-737): "prevent the 5-second Apple GPU Watchdog from
   killing processes under swap pressure… splits large computation graphs into ~1-layer
   chunks so macOS can page in weights incrementally". Same env re-set at line 898.
3. **Single-concurrency lock** (Server.swift:743-748): "SSD expert streaming keeps **one
   persistent pread buffer set per layer**, so concurrent requests overwrite each other's
   experts and corrupt the heap" → `--parallel` forced to 1.
4. **Directory must be complete**: `resolveStreamingDirectory`
   (`Sources/SwiftLM/StreamingDirectory.swift:163-228`) verifies local copy against the
   Hub listing (`getFilenames`), shard completeness (`stem-0000I-of-0000N.safetensors`,
   `shardNumber()` L69-82; index-driven via `model.safetensors.index.json` weight_map,
   L123-130), follows HF symlink-to-blob layout (L60-66). Incomplete → full download with
   a `.swiftlm-download-in-progress` sentinel (L169, 214, 226).
5. **Byte-range streaming activation** (Server.swift:887-899):
   ```swift
   setenv("EXPERIMENTAL_SSD_STREAM", modelDir.path, 1)
   ExpertStreamingConfig.shared.activate(
       modelDirectory: modelDir,
       useDirectIO: true  // macOS: 5 GB/s pread() via moe_stream_op.cpp
   )
   ```
   Comment (L889-892): activation lets fork-side `Load.swift` "1. Initialize
   **ExpertStreamerManager (shard index for getFile())** 2. Assign **tensorName** on every
   **QuantizedSwitchLinear** after quantize(). Without this, the **streamedGatherMM
   direct-NVMe path** never fires." → per-shard file handles + tensor-name→byte-range
   index + a gather-MatMul kernel that pulls expert slices by `pread` (impl in submodule).
6. **Lazy weights**: `modelConfig.lazyLoad = true` (Server.swift:840-842; mirrors in
   InferenceEngine.swift:434). InferenceEngine.swift:394-395 comment: "**lazyLoad=true means
   weights are mmap'd and not paged into RAM at load time — only active expert pages touch
   RAM during inference**". Draft and MTP-assistant configs also get `lazyLoad`
   (Server.swift:1195-1197, 1233).
7. **MLX allocator overrides** (Server.swift:901-956, §3 below).
8. **Forces all layers to GPU**: with `--stream-experts`, auto CPU layer partitioning is
   bypassed (Server.swift:1029-1041: "Bypassing CPU auto-partitioning (forcing all layers
   to GPU)"); `--gpu-layers` CPU-split MoE is warned as ~0.4 tok/s (Server.swift:1326-1331).
9. **Runtime switch**: `await container.setStreamExperts(true)` (Server.swift:1340-1351);
   if `--ssd-prefetch`: `MLXFast.setPrefetchEnabled(true)` ("PAPPS 16-Worker Thread Pool
   prefetcher") (Server.swift:1344-1347).
10. **Wisdom auto-calibration bypassed** for streaming (Server.swift:1354-1372).

The app-side engine (`Sources/MLXInferenceCore/InferenceEngine.swift:392-449`) does the
same activation before load, default **ON for MoE** (`effectiveStreamExperts(defaultingTo:
isMoE)`, L432; `GenerationConfig.swift:116-117`: persisted user toggle wins once saved),
`useDirectIO: true` on macOS / false on iOS (L438-444), and `deactivate()` on load failure
and unload (L525, 567; unload also zeroes `MLX.Memory.cacheLimit`, L568).

### 1.3 How they avoid the swap death spiral

The trick is a *combination*, all in the wrapper except one kernel-side piece:

- **Nothing is eagerly materialized**: mmap'd lazy load means weights are file-backed pages
  that only touch RAM when routed. The OS maps safetensors reads into the "Inactive"
  pool *[their docs, moe_ssd_streaming_architecture.md §3]*.
- **Routed experts are pulled by explicit `pread` into persistent per-layer buffers**
  (`moe_stream_op.cpp` / `mlx_fast_pread_into` at 4–9 GB/s *[PAPPS doc §1]*), not by
  faulting mmap'd pages on the critical path. Buffers are **allocated once per SwitchGLU
  layer and reused across tokens** — zero per-token allocation *[README:485, "Persistent
  Metal Buffers"]*. So expert bytes never enter the app's anonymous (swappable) heap.
- **Tell MLX its memory budget is effectively infinite** so its allocator neither errors nor
  spin-waits while pages page in: `Memory.memoryLimit = 200 GB sentinel` "to bypass MLX
  eval_impl's **spin-wait loop**" (Server.swift:901-904, 948, 955; ModelProfiler.swift:568).
  Safe "because only one model's expert pages are demanded at a time" (L903-904).
- **Cap the MLX cache allocator below physical RAM** so MLX evicts its own buffers instead
  of letting macOS swap: `Memory.cacheLimit = computeSSDMemoryBudget(...)`
  = `RAM×0.85 − 4GB − draftWeights`, floored at 2GB (Server.swift:1794-1808; on 64 GB:
  ≈50.6 GB). If main+draft weights exceed **70% of RAM**, they instead *lower*
  `Memory.memoryLimit` to `RAM×1.1` "so MLX will hit its hard limit sooner and begin
  **evicting old expert pages more aggressively instead of extending into swap**"
  (Server.swift:906-944).
- **Expert cache = the kernel page cache. Deliberately.** Key engineering finding
  [README:492]: "**Don't cache experts in application memory** — An LRU expert cache
  *stole* from the OS page cache and regressed performance (4.84 → 4.01 tok/s). Let the
  kernel manage it." Their measured steady state: "**OS page cache serves ~90% of expert
  reads from RAM**; GPU compute is ~190ms of ~200ms per token" [README:490-491].
  Streaming page eviction is "clean" because the pages are file-backed: "expert weights are
  mmap'd from SSD via the OS page cache. No swap involved — the page cache evicts stale
  expert pages cleanly" (Server.swift:982-983).
- **GPU watchdog survival**: explicit per-MoE-layer CPU drain `MLX.eval(expertOutput);
  Stream.gpu.synchronize()` "GPU Release Lock" *[their docs,
  moe_ssd_streaming_architecture.md §1]* + `MLX_MAX_OPS_PER_BUFFER=50` graph chunking.
  Consequence noted honestly: the sync blocks same-thread `pread`, motivating PAPPS.
- **Prefetch**: reactive streaming stalls; `--ssd-prefetch` moves I/O to a **16-worker C++
  thread pool** issuing async `mmap/pread()` behind the Swift boundary
  *[moe_ssd_streaming_architecture.md §2]*. The cheaper default trick (no flag needed):
  **AsyncEval speculative pread** — "uses **previous-token routing** to speculatively
  pre-load experts for the next token during the GPU async window (~70% hit rate). Only
  missed experts (~30%) require on-demand pread after routing sync" [README:483]. PAPPS
  phase-2 N-1 heuristic claims ">65% accurate for long-context sequential generation"
  *[PAPPS doc §4]*.
- **NVMe queue depth**: "Concurrent NVMe pread (**QD=24**): Replaces sequential pread with
  `DispatchQueue.concurrentPerform`… (8 experts × 3 projections = 24 parallel reads)"
  [README:482].
- **Fewer evals**: "Cross-Projection Batching: collapses ~1,400 per-expert `eval()` calls
  down to ~48 per token by orchestrating gate/up/down projections together in `SwitchGLU`"
  [README:481].
- **Quality knob**: `SWIFTLM_TOP_K` env reduces active experts/token at runtime (top-k 8→2;
  0.58→6.52 tok/s on Qwen3.5-122B M1 Ultra 64GB, "~10.6 GB resident, no swap" across all
  modes) [README:457-487].

### 1.4 Throughput model they state

`moe_ssd_streaming_architecture.md` §3: Qwen3.5-122B payload **1.84 GB/token**;
M-series random-read OS bandwidth **3.1–3.5 GB/s** → pipeline ceiling **1.69–1.84 tok/s**
when cold. Baseline background GPU allocations (>12 GB on 64 GB) guarantee swap overlap;
once swap writes start, "NVMe pread() latency spikes" — i.e. they explicitly identify the
swap spiral and prevent it via the resident-budget rules above. Also recommends `sudo purge`
for measurement isolation, and hardcoding `maxEntries = 8192` to stop backend cache growth
into swap (which cache this refers to is **ambiguous** — see §7).

### 1.5 Draft-model interaction (I/O fan-out law)

Verify pass sends N+1 positions each routing to *different* experts → SSD I/O scales with
the **union** of expert sets. At 4 draft tokens = 5× fan-out → net regression. Auto-cap
`--num-draft-tokens` to 1 when combined with `--stream-experts` (net positive if acceptance
≥50%) (Server.swift:844-870; README:470-477). MTP on streamed MoE: same fan-out, "MTP on
35B+ MoE will be **slower** than baseline" (README:535-545).

---

## 2. TurboQuant KV cache (`--turbo-kv`)

Two **separate** KV-quant systems (README.md:695-710 comparison table):
- `kv_bits` per-request (JSON body; only **4 or 8**): MLX-native group-quantized
  `QuantizedKVCache`, active **from token 0**, ~2–4× vs FP16
  (Server.swift:5027-5030; `GenerationConfig.kvBits/kvGroupSize`;
  InferenceEngine.swift:704-705).
- `--turbo-kv` server-wide (Server.swift:662-663): *"3-bit PolarQuant+QJL. Compresses KV
  history > 8192 tokens to ~3.5 bits/token — recommended for 100k+ context."*

### 2.1 Mechanism (wrapper side)

Per request, after `model.newCache`: flip a flag on every simple cache layer
(Server.swift:2792-2798; identical loop InferenceEngine.swift:749-760):
```swift
if let simple = layer as? KVCacheSimple { simple.turboQuantEnabled = true }
```
`KVCacheSimple` exposes `turboQuantEnabled` and `compressedOffset` to the wrapper
(Server.swift:3003). **Only `KVCacheSimple` is compressed**; with `--ctx-size` the model
gets `RotatingKVCache` and `--turbo-kv` is a **silent no-op** (warning at
Server.swift:1496-1497, 2803-2807 — a real benchmark footgun: their own README tables had
bogus turbo-kv columns for exactly this reason, README:151).

### 2.2 Algorithm (all from their docs; kernels in submodule)

`docs/turboquant_hybrid_architecture.md` + README.md:425-448:
- Encode (C++ `fast_turbo.cpp`): L2-normalize → **in-place Fast Walsh–Hadamard rotation**
  `O(d log d)` → **3-bit Lloyd-Max (non-linear, Gaussian) centroids** via binary search on
  `BOUNDARIES_3BIT`, packed `uint8_t` ("PolarQuant").
- **Asymmetric K vs V**: K gets 3-bit PolarQuant **+ 1-bit QJL sign residual** (additive
  correction only — the paper's "spend 1 payload bit on QJL" variant failed on Apple
  Silicon because softmax amplifies the resolution drop) → **4.25 bits/dim**; V gets plain
  3-bit → **3.125 bits/dim**. Average ≈ **3.6 bits/dim ≈ 3.5× vs FP16**.
- Dequant is **fused into the SDPA Metal kernel**: 16-byte packed structs passed to
  bggml-metal SDPA; kernel unpacks 3-bit indices, substitutes from a
  `CENTROIDS_3BIT` constant buffer, accumulates QJL signs in the hot loop. No separate
  dequant pass.
- **Hot-window eviction** (production design, doc lines 47-57): last **256 tokens stay
  fp16**; older tokens compressed in **step=256 chunks**; SDPA sees
  `[decoded_prior_history | fp16_hot_window]`; fp16 cold pages are *evicted*, decoded
  on demand (rejecting the reference's always-compress + shadow fp16 buffer as RAM-negative).
  Short prompts (<256) get zero compression.
- Activation threshold: README table says active "after **2048** tokens"
  (`turboMinActivationTokens` named at Server.swift:2999; constant in submodule).

### 2.3 Measured impact (README benchmarks)

- 100K context: Vanilla+TurboQuant **66.9 tok/s vs 27.5 vanilla (2.43×)**; OS RAM at 40K:
  **48.7 → 18.2 GB (−63%)**; GPU alloc 54.8 → 23.9 GB @40K (README:243-248, 194-216).
- MTP+TurboQuant: 66.2 avg tok/s vs 36.6 baseline (1.81×) on dense 27B (README:194-216);
  neutral/negative on 4-bit MoE (bandwidth-bound → FFN-bound flip, README:248-249).
- SSD-stream + TurboQuant at 32K/100K **collapses** (2.5 / 1.6 tok/s) — decode/quant churn
  on the I/O-bound path (README:240-241).
- Long-range recall bug fixed (SharpAI/mlx-swift-lm#65): attention now also covers
  compressed history; cost: 97 s vs 72 s on an 11.8K prompt (README:166).
- Telemetry: 10 s log line `[⚡️ SSD Stream] 8977 MB/s | … 🗜 TurboKV 4.3x (5MB saved)` via
  C atomics (doc lines 59-65).

### 2.4 Prompt-cache interplay (memory trap)

Reading `cache.state` **decodes all polar buffers back to fp16** — "~37 GB allocation" at
100K that "completely negates the compression savings". So prompt-cache save is **skipped**
when `turboQuantEnabled && compressedOffset > 0` (Server.swift:2994-3019), and `--turbo-kv`
disables the hybrid/rotating/boundary prompt-cache paths wholesale
(Server.swift:2847, 2857-2858, 2874-2875). Quantized-KV (`kv_bits`) requests also never
share the prompt cache (format mismatch, Server.swift:2822-2824, 3020-3023).

---

## 3. Memory management

### 3.1 Telemetry — `Sources/SwiftLM/MemoryUtils.swift`

- `getOSPhysFootprintGB`: `task_info(mach_task_self_, TASK_VM_INFO)` →
  `phys_footprint` (L9-24) — "capped by available physical RAM — when this hits the
  ceiling, macOS begins swapping" (L6-8).
- `getTotalMemoryDemandGB` (L30-51): `phys_footprint + compressed` — the **true demand**
  including compressed+swapped pages; used to compare Vanilla vs TurboQuant at extreme
  contexts.
- `getGPUActiveMemoryGB`: MLX allocator `Memory.activeMemory` (L54-56).

### 3.2 Pressure handling

- Server: `DispatchSource.makeMemoryPressureSource([.warning, .critical])`
  (Server.swift:1575-1590) → on pressure drop the in-memory **prompt cache**: warning keeps
  most-recent entry, critical empties it (L2122; flag help text L617, README:674).
- App engine (iOS-first pattern, worth copying): `didReceiveMemoryWarning` → unload +
  error state (never mid-stream, L200-207); `didEnterBackground` → stop+unload, foreground
  reload with **30 s grace period** (L209-267); thermal `.critical` → immediate
  `stopGeneration()`, throttled states block load (L270-295, 305-308). `unload()` →
  `ExpertStreamingConfig.deactivate()` + `Memory.cacheLimit = 0` (L560-569).
- **No** `jetsam` / `dispatch_source` memory-pressure on macOS beyond the one Server source
  above; no purgeable regions, no VM tags anywhere.

### 3.3 Profiler + limits — `Sources/SwiftLM/ModelProfiler.swift`

- `profile(modelDirectory:)` parses `config.json` (+`text_config` fallback), quant bits
  from `quantization_config` or model-id suffix (L440-450), weight bytes by dir walk
  (HF symlink-resolved, L454-480). **MoE detection** `findExpertCounts` (L375-414):
  BFS ≤3 deep over nested configs for `num_local_experts|num_experts|n_routed_experts`
  with language-container priority list and non-language exclusions (vision/audio towers) —
  because single-key detection silently misclassified MoE as dense → full materialization →
  OOM-kill (issue #112, L290-306; false-positive direction deliberately biased toward
  streaming, L425-437).
- KV estimate honors `layer_types` (full/sliding/linear) + per-layout KV heads/head-dim;
  naive per-layer estimate over-counted Gemma 4 ~10× (L64-79).
- `systemProfile()` (L485-500): **`GPU.deviceInfo().memorySize` is Apple's artificially
  capped Metal working set (~22–23 GB on a 64 GB machine), not RAM** — they take
  `max(0.85×RAM − 4GB, deviceInfo.memorySize)`. Directly relevant to our M4 Max budget.
- Strategy thresholds (L518-536): `fullGPU` ≤0.85–1× available; `swapAssisted` ≤2×;
  `layerPartitioned` ≤4×; `tooLarge` >4×. Per-strategy limits (L552-576):
  - fullGPU: `memoryLimit = 1.5× workingSet`; `cacheLimit = min(workingSet,
    (RAM − weights − draft − KV − 8GB)/2)` — comment: MLX's default whole-working-set cache
    "pushed macOS into swap by ~8K tokens" on 32 GB (L558-564).
  - swapAssisted: 200 GB sentinel + **2 MB cacheLimit** ("let OS manage caching").
  - With `--stream-experts` the Server overrides swapAssisted/layerPartitioned with the
    0.85×RAM streaming budget and relabels the strategy "SSD STREAMING (page-cache managed,
    no swap)" (Server.swift:976-1003).
- `computeSSDMemoryBudget` (Server.swift:1794-1808): `RAM×0.85 − 4GB − draftWeights`,
  floor 2 GB (issue #72: draft pages must be reserved or the page cache overcommits).
- `/health` exposes the plan (`healthInfo`, ModelProfiler.swift:162-175).

---

## 4. SSD streaming failure recovery — `Sources/MLXInferenceCore/SSDStreamingRecovery.swift`

- `SSDStreamingError` (L9-19): wraps errors from corrupted/truncated/incomplete safetensors
  detected during pread I/O → user message "re-download the model".
- **Error latch** `SSDStreamingErrorLatch` (L28-56): global singleton, NSLock, **first-error-
  wins** `set()`, `consume()`, `throwIfSet()`. Pattern (documented L21-27): reads inside
  **non-throwing `callAsFunction`** paths (SwitchGLU expert streaming inside MLX eval) post
  to the latch instead of `fatalError`; the generation loop calls `throwIfSet()` after each
  token; `InferenceEngine.generate()` catches and surfaces.
- Consumption site: InferenceEngine.swift:853-877 — catch `SSDStreamingError` → yield an
  error token + `markModelCorrupted` (UI offers "Delete & Re-download"); after the loop, a
  final `latch.consume()` check. Heuristic `isModelCorruptionError` string-matches "pread"
  (L599). **Recovery is coarse: re-download. No mid-generation retry, re-seek, or skip.**

---

## 5. Speculative decoding (brief)

- Three paths: dual-model draft (mlx-lm style, `--draft-model`, default 4 tokens/round),
  **MTP** (native heads, `MLXLMCommon.generateMTP`, default 3 tokens; Leviathan-style
  probabilistic rejection sampling claimed exact-parity, README:522-531), and **DFlash**.
- **DFlash** (arXiv:2602.06036; `Sources/DFlash/`): block-diffusion draft model
  (`DFlashDraftModel`, SwiGLU MLP, `mask_token`, `block_size`, consumes hidden states
  captured from selected target layers via `DFlashTargetModel.dflashForwardWithCapture`,
  DFlashRuntime.swift:16-55). Greedy acceptance = `matchAcceptanceLength`:
  `cumprod(drafted == posterior).sum()` (DFlashRuntime.swift:133-142).
- Rollback engines (`DFlashEngine.swift`): `FullAttentionEngine` (trim = decrement offset)
  vs `HybridGDNEngine` for GatedDeltaNet models; recurrent state can't be trimmed →
  `RecurrentRollbackCache` records an **innovation tape** and replays accepted steps
  (accurate, ~30% slower) or `MambaSnapshotCache` whole-state checkpoints
  (DFlashRuntime.swift:43-53, 146-190; `RecurrentRollbackCache.swift`).
- Custom Metal kernels (`DFlashKernels.swift`, 843 lines + 603 optimized): tape-replay,
  GatedDelta-forward-with-tape, batched 2-pass SDPA numerically aligned with stock MLX —
  registered as a global provider (Server.swift:1296-1298).
- `Gemma4MTPBench/main.swift`: real-model MTP bench with safety caps (maxKVSize=512,
  maxTokens=50, numDraft=2).

---

## 6. What Mac64-MoE should steal

1. **Two-knob allocator policy is their core swap-avoidance**: (a) `memoryLimit` sentinel
   high enough (200 GB) that the *allocator* never spins/errors while file-backed pages
   demand in, and (b) a hard **resident cache budget `0.85×RAM − OS reserve − other-model
   weights`** on everything it caches. Our native equivalent: no allocator cap on mmap'd
   file-backed expert regions (they're clean, kernel-evictable), but a strict cap on
   **anonymous/Dirty** memory (KV, arena, activations) below ~85% of RAM — Dirty pages are
   what macOS swaps; file-clean pages are not. Their `1.1×RAM` "tight limit" fallback when
   two models (main+draft) coexist is the same idea under co-residency pressure.
2. **`pread` into persistent per-layer buffers, not fresh allocations**: their "one
   persistent pread buffer set per layer" + "allocated once per SwitchGLU layer, reused
   across tokens" = our expert-aligned resident arena, zero per-token allocation. Note the
   side effect they hit: persistent buffers are per-layer and **not reentrant** → hard
   single-request constraint; our design should make arena slots request-scoped or per-slot.
3. **Do not build an app-level LRU *on top of* the page cache — but do build one *instead
   of it***. Their data: an app LRU competing with the OS cache regressed 4.84→4.01 tok/s;
   letting the kernel manage the file cache hit ~90% RAM-served reads. If we use
   `F_NOCACHE`/direct reads into a private arena (our plan), the kernel cannot help and we
   **must** provide the LRU/eviction PAPPS proposes (Pinned / Speculative / Eviction tiers)
   — their conclusion "let the kernel do it" applies only to their mmap+buffered-pread
   hybrid. Worth A/B on M4 Max: buffered pread+page-cache (their way, free LRU) vs
   F_NOCACHE+arena (deterministic, no Inactive-pool churn, no jetsam surprise).
4. **Previous-token-routing speculative prefetch during the attention window**: cheapest
   big win, ~70% hit without any predictor model; overlap with the O(N²) SDPA GPU window
   and with `gpu.synchronize()` waits via a worker pool (QD≈24 concurrent reads saturates
   the NVMe). N-1 heuristic alone beats a learned pre-attention router (zero-shot SVD of
   gate weights called useless; offline-trained linear classifiers only if needed, PAPPS
   doc §3).
5. **GPU watchdog discipline**: chunk command buffers (~1 layer/buffer, their
   `MLX_MAX_OPS_PER_BUFFER=50`) and drain per MoE layer; under swap the 5 s watchdog
   otherwise kills the process.
6. **Draft/speculate × streaming fan-out law**: verify of N+1 positions fans expert I/O by
   the **union** across positions. Auto-cap drafts to 1 token when streaming (net-positive
   iff acceptance ≥50%); MTP on streamed MoE is a known negative. Our residency-set design
   should budget SSD bytes/token as `|∪ experts over verify batch| × expert_size`.
7. **TurboQuant KV**: asymmetric K/V bit budget (QJL correction only on K: 4.25 vs 3.125
   b/d) — QJL helps only where dot-products happen; WHT+Lloyd-Max fused dequant inside the
   SDPA kernel (no dequant pass, no fp16 shadow buffer); **256-token fp16 hot window +
   256-token compression chunks**; measured 2.4–3.2× decode speedup at 100K and −63% OS
   RAM at 40K. Also steal: never snapshot/serialize compressed KV (37 GB fp16 blow-up);
   and beware the "flag silently no-op'd because the cache class was different" failure —
   fail loudly (they now warn at startup).
8. **Memory telemetry pair**: `phys_footprint` (swap begins at ceiling) vs
   `phys_footprint + compressed` (true demand) from one `task_info(TASK_VM_INFO)` call —
   exactly the resident-vs-demand split our benchmarks need. Plus
   `DispatchSource.makeMemoryPressureSource` to shed caches on `.warning` before jetsam.
9. **Don't trust `GPU.deviceInfo().memorySize`** (Metal recommendedWorkingSet is Apple-
   capped ≈22–23 GB on 64 GB machines); budget from `ProcessInfo.physicalMemory×0.85 − 4 GB`
   (ModelProfiler.swift:485-500).
10. **Error latch for non-throwing kernel paths**: NSLock first-wins global latch checked
    after each token, surfaced as "model corrupted → re-download". Crude but prevents
    `fatalError` in `callAsFunction`.
11. **RLIMIT_NOFILE → 4096** up front for sharded checkpoints.
12. **Cold-stream ceiling honesty**: 1.84 GB/token payload ÷ 3.1–3.5 GB/s random-read ≈
    1.7–1.8 tok/s when the page cache is cold — set expectations for our 100B+ targets;
    expert locality/pinning (hot expert arena) is what buys past it, since their ~90%
    page-cache hit rate *is* the locality win.

---

## 7. Open questions / unverifiable

- **`useDirectIO: true` semantics**: comment says "5 GB/s pread() via moe_stream_op.cpp"
  but README credits the page cache with ~90% of reads — so is the streamed path buffered
  `pread` (page-cache-backed) despite the name? Impl is in the missing submodule. Marked
  unverifiable.
- **ExpertStreamerManager data structures** (shard index layout, tensor→(file, offset, len)
  table), `streamedGatherMM` kernel, `QuantizedSwitchLinear` internals, `ssd_streamer.mm`,
  `fence.air`, `fast_turbo.cpp`: **all in empty submodules** — none verified.
- **`maxEntries = 8192`** in moe_ssd_streaming_architecture.md §3: which cache (prompt
  cache? expert-chunk index?) unspecified.
- **PAPPS shipping state**: flag + `MLXFast.setPrefetchEnabled` exist; the 16-worker pool
  and N-1 predictor are submodule-internal; claimed hit rates (70%, 65%) are their numbers,
  not independently checkable.
- **TurboQuant exact constants**: `turboMinActivationTokens` value (README says 2048, flag
  help says >8192 compressed), centroid tables, QJL matrix construction — in submodule.
  The "8192" vs "2048" discrepancy between README.md:662 help text and README:695 table is
  unresolved from the wrapper alone.
- **SWIFTLM_TOP_K quality deltas** (top-k 6/4/2 perplexity) not quantified in-repo.
- README benchmark figures (M6 32 GB, M1 Ultra 64 GB) taken on their profiling harness
  (`scripts/profiling/`), plausible but not re-run here; note their own history of retracted
  turbo-kv benchmark columns (README:151) — treat numbers as directional.
- Whether they ever measured swap-death directly (they describe the mechanism and avoid it
  structurally; no swap-trace artifacts found in repo).
