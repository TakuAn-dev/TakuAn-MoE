# Mac64-MoE Engine — Architecture Design (Phase 0)

Status: Phase 0 deliverable. 2026-11-22. Grounded in: `docs/refs/{strata,basert,swiftlm,mlx-od-moe}.md`
(code-level analyses of `refs/*` clones) + live probes on the target machine. No engine code exists yet.

Machine facts (live-probed 2026-11-22, `scratch/m4_probe.mm`, `diskutil`, `dd`):
- Apple M4 Max, 64 GB (16 CPU / 40 GPU), macOS 27.2.1 (26B131), Apple clang 21, arm64.
- `MTLDevice.recommendedMaxWorkingSetSize = 55.66 GB`, `maxBufferLength = 41.75 GB`
  (→ arenas > 41.75 GB must be multi-buffer; total GPU footprint ceiling is ~0.81×RAM, NOT the
  ~22 GB cap SwiftLM saw on older macOS — re-verified here, `ModelProfiler.swift:485-500` is stale for macOS 27).
- `supportsFamily(MTLGPUFamilyMetal4) = 1` → Metal 4 / MetalPerformancePrimitives `matmul2d`,
  cooperative tensors, `MTLTensor` available.
- `newIOCommandQueueWithDescriptor:` returns OK (MTLIO family available, macOS 13+ headers in CLT SDK).
- CLT SDK ships `MTLResidencySet.h`, `MTLIOCommandQueue.h`, `MTLTensor.h`, `MetalPerformancePrimitives.framework`.
  **Xcode is not installed** — CMake/Ninja not installed either (Phase 1 DoD installs them via brew).
- Storage: internal NVMe = Apple Fabric (Strata measured 2.97 GB/s @1.75 MB reads on an M4 mini).
  External **nvme500 (PCIe NVMe, where all model blobs live): measured 2.21 GB/s sustained write**
  (`dd` 2 GiB); cold-read sustained TBD by benchmark harness (expect ≥2.5 GB/s; design assumes ≥2.2).
- Local blobs: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ3_XXS ≈ **87.0 GB** (4 shards; blobs in
  `~/Library/Application Support/Splash/internal-cache/blobs`, symlinked from nvme500 HF cache) and
  UD-IQ4_XS variant. Splash 1.7.0 serves them with `--max-memory=45G`, PLE→SSD, experts ~58–60% resident.

Reference decisions carried into this design are cited as `[strata §N]`, `[basert §N]`, `[swiftlm §N]`,
`[mlx-od-moe §N]` against the four docs in `docs/refs/`.

---

## 0.1 Phase 1 amendments (approved 2026-11-22, supersede conflicting text below)

**A1 — Build system & Metal compilation (zero external installs).**
`brew`, `cmake`, `ninja` and `xcrun metal` (full Xcode) are **not installed** on this machine and
Phase 1 installs nothing system-wide. The engine builds with a POSIX **`Makefile` + `/usr/bin/clang++`
(`-std=c++20 -fobjc-arc`)**; Metal kernels ship as MSL source embedded in the binary and are compiled
at first load via `MTLDevice newLibraryWithSource:options:error:`, then cached on disk with
`MTLBinaryArchive` (keyed by source hash + device) so steady-state startup skips compilation.
Section 8 step 0 ("brew install cmake ninja") is **cancelled** for the engine. (An independent
`llama.cpp` checkout used *only* as the Phase 1 parity oracle may be built with a venv-local `pip
install cmake ninja`; that toolchain never touches the engine build.)

**A2 — Phase 1 verification model: `thinkingmachines/Inkling-2-Mini` at 8-bit.**
Replaces Qwen3-30B-A3B (§1.1) as the Phase 1 end-to-end target:
`ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf` (tensor dtypes are actually **Q8_1**+F32,
704 tensors, 24 physical layers expanded to 48 KV slots by the loop). All blobs and packs live on
**`<MODEL_DIR>/`** — the internal SSD stays clean.
Forward semantics are pinned to upstream `llama.cpp` `src/models/inkling2{,-loop}.cpp`
(arch `inkling2_loop`): no RoPE; learned relative-bias table + per-head attention sink; log-scaled
positions `tau = 1 + α·ln(max(1,(pos+1)/floor))` on Q/R of the 4 global layers; short causal convs
(k=4, residual, dynamic taps on global layers); depth-attention residual mixing (`mix`, block size 6);
loop core 6..17 ×3 with per-slot KV/conv state and rank-64 gated message passing
(`out = cur·(1−tanh f) + tanh(g)·msg`); Sigmoid+bias router over 128 experts (top-8, weights from the
*bias-free* sigmoid probs, `w/Σ(w+1e-20)·8`); fused gate|up experts (gate first half, up second half);
`SITU(g,u) = 4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)`; final logits `30·tanh(z/30)`.
Phase 1 remains **resident-only** (no SSD expert streaming, no arena eviction).

---

## 1. Target models

### 1.1 Phase 1 verification MoE (numerical parity)

**Qwen3-30B-A3B** (`Qwen/Qwen3-30B-A3B`, HF safetensors; MLX 4-bit and GGUF Q4_K_M both exist):
- 48 layers, hidden 2048, 128 routed experts, top-8, `moe_intermediate 768`, GQA 32Q/4KV,
  head_dim 128, rope_theta 1e6, vocab 151,936. ~30.1 GB at 4-bit → runs **fully resident** on
  this machine: streaming can be off while kernels are validated.
- Why this model: standard GQA + softmax-top-k MoE (identical routing math family as Flash-Next,
  minus bias/sigmoid variants), and **two independent references are free**: llama.cpp (GGUF,
  same quant codes we will pack) and MLX (mlx-lm 4-bit, the family the user's Splash/omlx stack
  already runs). AGENTS.md parity rule: per-tensor dequant exactness + per-layer cosine ≥ 0.999.
- What it does NOT cover: GDN linear attention, QSA sparse attention, HyperConnections, n-gram
  PLE, MTP. Those get a dedicated **reduced-layer Flash-Next slice** (below), per AGENTS.md
  ("small MoE or reduced-layer slice").

**Flash-Next reduced slice** (Phase 2): convert only layers {0..3} of a BF16 Flash-Next checkpoint
(one full GDN→MoE×3 + one QSA→MoE block = the full 4-layer period `12 × (3×(GDN→MoE) → 1×(QSA→MoE))`
plus the MTP layer) into f16 + one quant tier. Parity references per kernel:
- GDN delta rule: torch/CPU reference in the reduced-layer harness (same equations as `gdn.metal`,
  `[strata §4]`), cross-checked against mlx-lm's Qwen3-Next GDN layer (mlx-lm ≥0.26 supports
  Qwen3-Next; it is installed on this machine under `omlx_env`/splash).
- QSA top-k indexer + sparse attention: CPU reference (scores→top-k→flash) + llama.cpp's
  Qwen3-Next path if present at build time.
- Router/HC/norm: exact fp32 torch reference.
Slice stays f16 → fits any RAM; streams nothing. This isolates kernel correctness from I/O.

### 1.2 100B+ target: Qwen3.8-Flash-Next 125B-A6B

Confirmed geometry (HF `schwyzquants/Qwen3.8-Flash-Next` model card, 2026-11-22, identical to
Strata's compiled geometry `refs/Strata/engine/include/strata/core/geometry.hpp:46-54`,
`engine/metal/src/engine/geo.h:1-33`):

| field | value |
|---|---|
| total / active params | 125B / 6B (+ **51B n-gram embedding table (PLE)**, off-accumulator, 26.8 GiB IQ4_NL rows `[strata §3]`) |
| layers / layout | 48; `12 × (3×(GDN→MoE) → 1×(QSA→MoE))` = 36 GDN + 12 QSA |
| hidden | 2560 |
| MoE | **512 routed experts/layer, top-10 routed + 1 shared**, expert intermediate **640** |
| GDN | 16 QK-heads / 48 V-heads, state 128×128, conv 4-tap, conv channels 10240 |
| QSA | 24 Q / 2 KV heads, head_dim 256, rotary dim 64; indexer 4×128 MQA + 1 shared key head; budget 2048+3 cells |
| HyperConnection | 4 streams, bottleneck rank 320 (gated residuals) |
| vocab / lm output | 248,320 padded |
| MTP | 1 layer, multi-step trained (~4B params incl. projections) |
| context | 262,144 native (→1M extensible) |

Per-expert bytes: `3 × 640×2560 = 4.915M params` → UD-IQ3_XXS ≈ **1.96 MB** (3.19 bpw effective),
UD-IQ4_XS ≈ 2.63 MB (4.28 bpw). 24,576 routed experts → **≈48.2 GB (IQ3_XXS) / 64.6 GB (IQ4_XS)**.
Decode payload/token: 10×48×1.96 MB ≈ **0.94 GB/token routed** + shared 0.09 GB @Q8 + dense ≈ **≈1.1 GB/token (IQ3_XXS)**
— vs 2.2–3 GB/s SSD this is a **cold-stream floor of ~2–3 tok/s**; arena hit-rate is what buys past
it (SwiftLM's page-cache ~90% hit ≈ 4.8 tok/s on a slower path `[swiftlm §1.3]`; Strata gets
7.4–8.6 tok/s on a 32 GB M4 mini with the same class of model `[strata §0]`).

Blob choice for bring-up: the **existing 87 GB UD-IQ3_XXS GGUF shards on this machine** — Phase 2
packs them with a GGUF-passthrough quantizer (no re-quantization → parity is by-construction on
expert bytes), Phase 3 adds a native HF-safetensors→m64pack quantizer for Q4_K/IQ tiers.
Phase 4 target model stays Flash-Next (matches the user's daily Splash setup, so A/B against
Splash's own tok/s is one command away).

Fallback 100B+ model if Flash-Next weights prove unavailable for HF-side conversion:
`moonshotai/Kimi-K2.5` (384×28 layers, top-8 — the mlx-od-moe target, `[mlx-od-moe §1]`) or
`Qwen/Qwen3-Next-80B-A3B` (same architecture family, 80B → lighter).

---

## 2. Pack format: `.m64pack` v1

One file per model (+ optional sidecars for the PLE table and drafter, see 2.5). Design rules:
(a) every GPU-addressed byte-range is page-aligned so it can be `newBufferWithBytesNoCopy`'d or
`pread`'d with zero staging `[basert §1]` `[strata §1]`; (b) **no per-forward transformation** —
QKV fusion, gate|up fusion, expert grouping, layout tiling all happen at pack time `[basert §1]`;
(c) a routed expert is exactly **one contiguous aligned range** so a miss is exactly **one pread**
`[strata §1]`.

### 2.1 Byte layout

```
offset        size  field
0x0000        4     magic = "M64P"
0x0004        4     format_version = 1
0x0008        8     header_len (u64)
0x0010        N     header JSON (canonical: sorted keys, UTF-8, no whitespace)
…               —   zero padding to next 64 KiB
ALIGN(64KiB)        weights_blob (tensor bytes, per-tensor alignment below)
after blob          extension slots (see 2.6)
EOF
```

All integers little-endian. Blob starts at a 64 KiB boundary so that 16 KiB-aligned tensor offsets
are 16 KiB-aligned in the **absolute file** — the precondition both Strata (`general.alignment=16384`
+ mmap+NoCopy, `[strata §1]`) and baseRT (blob at 64 KiB, Apple gpu-region 16 KiB, `[basert §1]`)
independently landed on. 16 KiB = Apple GPU page (Strata's `PAGE=16384`, `stream.cpp:26`); the
arena slot allocator rounds every expert to this.

Per-tensor alignment from header `alignment_log2` (default): `region= gpu→14 (16 KiB)`,
`expert→14`, `cpu→6 (64 B)`. Tensor record `offset` values are **absolute file offsets**
(so the blob starts at `blob_start = 64 KiB` and every record's `offset` is checked directly).
Load-time assertion (baseRT's `tensor_is_zero_copy_eligible`
pattern, `[basert §1]`) checks `(tensor_offset) % 16384 == 0` for every
gpu/expert-region tensor; hard failure, not a warning.

### 2.2 Header JSON schema

```jsonc
{
  "magic": "M64P", "version": 1, "tool": "m64pack 0.1", "created": "…ISO8601…",
  "model": { "arch": "qwen3_8_flash_next" | "qwen3_moe", "name": "…",
             "n_layers":48, "n_embd":2560, "n_ff_expert":640,
             "n_expert":512, "n_expert_used":10, "n_expert_shared":1,
             "n_head":24, "n_head_kv":2, "head_dim":256, "rotary_dim":64,
             "vocab":248320, "ctx_train":262144, "rope_theta":…, "norm_eps":…,
             "layer_types": ["gdn","gdn","gdn","qsa", …],   // 48 entries
             "gdn": {"hk":16,"hv":48,"state":128,"conv_kernel":4,"conv_ch":10240},
             "qsa": {"idx_q_heads":4,"idx_key_dim":128,"idx_top_k":2048},
             "hc": {"streams":4,"rank":320},
             "mtp": {"present":true,"kind":"mtp"},
             "ple": {"present":true,"rows":320001536,"row_bytes":90,"fmt":"iq4_nl",
                      "sidecar":"model.ple.m64ple"} },
  "quant": { "profile_sha256":"…", "tier_table":[ {"pattern":"…","dtype":"…","group":…,
             "scale":"bf16"} , … ] },          // the pack-time rules used (§2.7)
  "residency": { "total_weights_bytes":…, "hot_weights_bytes":…, "expert_bytes_each":…,
                 "max_expert_bytes_per_token":…, "recommended_min_device_mb":… },
  "tensors": [ { "name":"…", "dtype":"…", "shape":[…], "region":"gpu|expert|cpu",
                 "offset":…, "length":…,
                 "scale_off":…, "scale_len":…, "group_size":…,
                 "layout":"rowmajor|tile_8x8_mlx|qkv_fused|gateup_fused",
                 "residency":"hot|warm|cold", "xxh64":"…" } ],
  "expert_index": { "stride_mode":"per_layer",                     // all layers same size in v1
                    "layer_base":[48 × u64], "expert_stride":…, "order":"gate|up|down" },
  "checksum": { "blob_xxh64":"…", "sig":"…(ed25519, optional)" }
}
```

Tensor serialization order = **forward order** (embed → per layer norm/qkv/o/norm/router/shared/
experts → final_norm → lm_head → mtp), i.e. baseRT's readahead convention `[basert §1]`, so a
sequential warm-up or the page cache naturally pipelines layers.

### 2.3 Tensor records

- `dtype` enum mirrors GGML (`q4_0, q4_k, q5_k, q6_k, q8_0, iq3_xxs, iq4_xs, iq4_nl, f16, bf16, f32`)
  — **GGML block layouts verbatim** for quantized tensors so (a) existing GGUF blobs can be
  byte-copied in (zero re-quant, parity by construction), (b) llama.cpp remains a valid oracle for
  dequant parity. We do NOT inherit GGUF's header/KV encoding — that stays a converter detail
  `[strata §1]` `[basert §6]`.
- Scales/mins: GGML blocks carry their own scales inline (that's the block format). The baseRT
  "separate scale regions" rule `[basert §1]` applies only to **our own** formats added later;
  GGML-compat wins for Phase 1-3.
- `layout`: `rowmajor` (native), `qkv_fused` (Q|K|V rows concatenated in one tensor),
  `gateup_fused` (gate rows followed by up rows inside the expert slice — expert order becomes
  `gate|up|down`, one contiguous range, matching Strata's STRX and `moe_gu` access order
  `[strata §1]` `[strata §4]`), `tile_8x8_mlx` (reserved for an MPP-tiled repack in Phase 4).
- Per-tensor `xxh64` over exactly `length` bytes; blob-wide `blob_xxh64`; optional ed25519 over
  (canonical header + blob hash) — structure adopted now, signing enabled later `[basert §1]`.

### 2.4 Expert index and residency

- All 512 experts of a layer are equal-sized → `expert_index` stores one `layer_base` per layer +
  `expert_stride`; expert *e*'s bytes = `[layer_base[L] + e*stride, +stride)` with stride rounded
  **up to 16 KiB** `[strata §2]`. This is the entire random-access structure — kernels and I/O
  threads both address experts by `(layer, e)` arithmetic, no per-expert table in RAM.
- `residency` hints: routers/shared/norms/embed/lm_head = `hot`; current-layer attention = `warm`;
  routed experts = `cold` `[basert §1]`. The `residency` block is **computed at pack time** and
  re-verified by the engine at open before any allocation (fail fast if the machine can't fit the
  documented hot set + minimum arena).

### 2.5 PLE (n-gram embedding) sidecar

`.m64ple`: raw IQ4_NL row blob, 320,001,536 × 90 B, row-aligned to 4 KiB; never mapped, never in
the arena. Engine does token-triggered 4 KiB `pread`s (16 rows/page) + a bounded row cache
(~1M rows ≈ 90 MB, clock eviction) — Strata's measured design, 20–34% intra-prompt hit
`[strata §3]`. Keep-alive pings prevent SSD sleep.

### 2.6 Extension slots

After the blob, sequence of `{u16 kind, u16 flags, u64 len, u64 xxh64, payload}` (baseRT's
framing `[basert §1]`): `kind 1=MTP drafter sub-pack` (its own header-JSON-in-payload + blob),
`kind 2=KvWarmup`, `3=RopeTables`, `4=PackStats`. Unknown kinds skipped unless `REQUIRED` set.

### 2.7 Quant profile (pack-time policy)

JSON rule list `[{pattern(glob), dtype, group, scale, algo}]` applied in order, last match wins —
baseRT profile format adopted nearly verbatim `[basert §2]`. v1 default tiers for Flash-Next:
- norms/router/**shared expert**/HC/gdn_conv → `f16` or `q8_0`;
- GDN & QSA projections → `q6_k`; attention o/q/k/v → `q6_k`;
- embed/lm_head → `q8_0` (vocab is the hot 0.64 GB); MTP drafter → `q8_0` `[strata §5]`;
- routed experts → `q4_k` / `iq3_xxs` / `iq4_xs` (CLI-selected; per-expert `gate|up|down` fused);
- PLE rows → `iq4_nl` (fixed).
Group sizes for our native `q*` formats (if we add non-GGML formats later): 2/3-bit→32,
4/5/6→64, 8→128, scale bf16 `[basert §2]`. Converter refuses quant-from-quant unless
`--allow-quant-from-quant`; GGUF-passthrough mode for the local IQ3_XXS blobs `[basert §2]`.

### 2.8 Converter (`tools/m64pack.py`)

Inputs: HF safetensors dir **or** GGUF (+ multi-shard index) → one `.m64pack` (+ `.m64ple`,
sidecar slots). Per-tensor output hashes printed; `--verify` re-reads and xxh64-compares
(Strata's `--verify` discipline `[strata §1]`). Output written with `fcntl(F_NOCACHE)` so packing
doesn't dirty the page cache `[strata §1]`. Phase-1 build of this tool doubles as the parity
fixture generator (dumps golden dequant vectors per tensor).

---

## 3. Memory budget (≤ 48 GB resident on 64 GB)

Machine ceiling: Metal working-set cap 55.66 GB (probed). Budget keeps **≥ 16 GB off-resident**
for macOS + page cache headroom + spikes. Strata's guard rail (`keep = max(7 GB, 15% RAM)`,
process anon footprint ≤ 60% RAM, available-memory counted as free+purgeable+**file-backed**
inactive only, arena grown in 1.5 GB steps re-measuring free each step, `[strata §2]`) is adopted
as the runtime allocator policy — our 48 GB cap is stricter than their 60% (38.4 GB) footprint
cap on this machine, so we inherit their measured margin.

Resident plan for Flash-Next @IQ3_XXS (targets; all re-computed from pack header at load):

| component | class | budget | notes |
|---|---|---|---|
| Attention (36 GDN + 12 QSA projections) | hot, mlocked | **2.4 GB** | q6_k; GDN 58M p/layer×36, QSA 35M×12 |
| Routers (48×512×2560) | hot, mlocked | **0.13 GB** | f16 (router precision is quality-critical `[basert §2]`) |
| Shared experts (48) | hot, mlocked | **0.25 GB** | q8_0 |
| Norms + HyperConnections (HC4/rank320) | hot, mlocked | **0.25 GB** | f16 |
| Embeddings + lm_head | hot, mlocked | **1.3 GB** | q8_0, 2×636M params |
| **Hot Expert Arena** | hot, mlocked, anonymous mmap ×≤4 buffers | **30 GB** | ≈ 15,200 of 24,576 expert slots (≈62%) @1.96 MB; size-classed LRU |
| KV cache (QSA, INT8, 128K ctx, 1 seq) | hot | **0.85 GB** | 12 layers × 2 × 256 × 2B = 12.3 KB/tok fp16 → 1.61 GB @128K fp16 → **0.85 GB INT8 (codes+fp16 scale/64)** @128K (`[strata §4]`-style storage-only quant) |
| GDN recurrent state (1 seq) | hot | **0.11 GB** | 36×48×128×128 fp32 = 113 MB fp32 / 57 MB fp16; fp32 state for parity first |
| PLE row cache | hot | **0.2 GB** | ~1.5M rows |
| Activations / scratch / command buffers | hot | **1.5 GB** | 8-token windows × 48 layer segments + MTP chain |
| MTP drafter (resident only while `--spec>0`) | warm, mlocked on demand | **4.3 GB** | q8_0; pageable out when spec off |
| **Total** | | **≈ 37.0 GB** (+4.3 spec) = **≤ 41.3 GB** | headroom ≈ 7 GB under the 48 GB cap |

Non-resident: routed experts not in arena + PLE rows → nvme500; the pack file itself is
**never mapped** — file-backed GPU buffers would put 87 GB of file pages into the wired/Inactive
dance we must avoid (Strata's "37.6 GB NoCopy buffer cannot be made resident in 32 GB",
`[strata §2]`).

Runtime guard (all measured, not assumed):
- `task_info(TASK_VM_INFO).phys_footprint` and `+compressed` logged per 10 s
  `[swiftlm §3.1]`; swapout counter (`pageins`, `vm.swapusage`, `task_vm_info.swapouts` if
  present) must stay flat — hard assertion in soak tests (§8 Phase 2 DoD).
- `DispatchSource` memory-pressure: `.warning` → shrink arena to 80%, drop PLE row cache, shed
  MTP; `.critical` → arena to 50% + stop accepting new prompts `[swiftlm §3.2]`.
- Arena grown in 1.5 GB steps, re-measuring `vm_page_free_count × page_size + purgeable` before
  each step; stop early rather than reach jetsam `[strata §2]`.

**Zero-swap mechanism decision**: anonymous `MAP_PRIVATE` arena + `mlock()` +
`newBufferWithBytesNoCopy` (zero staging), **not** `MTLResidencySet` — two independent measured
comments in production code say requestResidency "did not hold the allocations"
(`mtl_rt.mm:49`, `stream.cpp:549`, `[strata §2]`). We re-verify both on macOS 27 in Phase 2 (the
claims are pre-27); design defaults to mlock either way. Arena chunking: ≤4 ×
min(maxBufferLength−slack, arena/4) buffers, slot ids `(chunk<<28)|page16k`, exactly Strata's
schema (`stream.cpp:423-470`, `[strata §2]`) — probed `maxBufferLength`=41.75 GB confirms one
30 GB buffer is legal here, but we keep chunking for >41 GB headroom (IQ4_XS packs).

---

## 4. NVMe SSD async streaming pipeline

### 4.1 Mechanism comparison (decision table)

| | pread + F_NOCACHE + thread pool | MTLIOCommandQueue | file-backed mmap + madvise |
|---|---|---|---|
| What refs use it | **Strata production** (6 threads, ≤8-way split reads, 2.97 GB/s M4 mini `[strata §3]`); SwiftLM fork-side pread "4–9 GB/s, QD=24" `[swiftlm §1.3]` | none of the 4 refs (grep-verified in Strata `[strata §3]`) | SwiftLM uses mmap only for *lazy resident* loads, not expert misses `[swiftlm §1.2]` |
| Destination | our mlocked arena (GPU-visible zero-copy) | directly into a GPU `MTLBuffer` (NoCopy-from-file must be block-size aligned; scratch for compressed) | page cache → copy into arena (double cache) |
| Bypasses page cache? | yes (`F_NOCACHE`), deterministic | yes (I/O via ANS/IOKit path) | no — and Inactive-pool churn is exactly what we must not pay |
| Deterministic eviction | yes (our LRU) | yes (buffer lifetime) | no (kernel decides; SwiftLM measured this *as a feature* at 90% hit but lost single-request determinism and had to force `parallel=1`) |
| Overlap with GPU | independent threads; sync via our own queues | encode into IO command buffer, `MTLSharedEvent` completion — GPU can wait natively | fault-driven (uncontrollable) — **rejected** |
| Cost | we own latency hiding | extra API surface, `MTLBuffer` must be NoCopy-of-mmap anyway (we'd mmap the *arena*, not the file); Apple docs thin | zero code, unbounded risk |
| Verdict | **primary** | **phase-2 prototype A/B** (`newIOCommandQueue` probed OK here) | resident-path lazy load only, experts never |

Decision: **`open(O_RDONLY)` + `fcntl(F_NOCACHE,1)` + `fcntl(F_RDAHEAD,0)` + `pread` into the
mlocked arena, 6–8 worker threads, per-miss split into ≤8 chunks** — the exact proven Strata
configuration `[strata §3]` — because both working reference engines independently converged on
pread and nothing in the corpus demonstrates MTLIO beating it on Apple silicon. MTLIO is built as
a swappable `ExpertSource` backend (interface: `submit(slot, file_off, len) → future`), A/B'd in
Phase 2 with the same bench; if it matches ≥2.5 GB/s at QD≥8 we adopt it for prefetch reads only
(keeps urgent-path latency predictable).

### 4.2 Per-decode-step data flow

```
host scheduler (one thread/seq)                I/O pool (6–8)            GPU (1 cmd buffer, 48 segments)
─────────────────────────────                 ────────────────          ──────────────────────────────
at_layer(l) fires on shared-event cut ──►  urgent: read misses(l)   ◄─ segment l computes
  pin slots hit this step                     promote wanted prefetched
  cancel unwanted prefetch, promote wanted    read_split ≤8 threads/expert
  issue prefetch(l+1|l+2 guessed)   ◄──── router-GEMV predictor results
  wait done_cv_ (only THIS layer)                                         segment l+1 released by event bump
```

- GPU↔host sync: **one command buffer per decode step**, 48 segments with
  `encodeSignalEvent(ev, 2s+1)` / `encodeWaitForEvent(ev, 2s+2)` on one `MTLSharedEvent`, host
  spins `signaledValue` then bumps; 20 ms completion spin before `waitUntilCompleted`
  (+5% tok/s) — adopted verbatim `[strata §2]` (`mtl_rt.mm:517-558`).
- Prefetch: layer *l*'s FFN input is GEMV'd against layer l+1/l+2 **real router weights**
  (10 MB GEMVs, sub-ms on 40 cores) → top-K(≤32) guessed experts queued at LRU front (small
  experts) / tail (big) `[strata §2]`. No learned shadow model in v1: SwiftLM measured previous-
  token routing ≈70% hit for free and their own learned router-predictor "useless"
  `[swiftlm §6.4]`; mlx-od-moe's 4.4M-MLP accuracy is unmeasured `[mlx-od-moe §3]`. Shadow MLP
  stays a Phase-4 experiment behind the same interface.
- Prompt path: prefill chunk issues **whole-chunk expert union** reads, offset-sorted, deduped by
  slot, before the chunk's layer-0 GEMM; arena min_slots raised to full-layer width (Strata's
  64/576 size-class rule `[strata §2]`).
- Double buffering: no separate "double buffer" — the arena IS the persistent staging area
  (slots reused across tokens, zero per-token allocation `[swiftlm §6.2]`). Slot states:
  `free → queued → reading → ready(pinned@stamp)`; a read into a ready slot is coalesced (refcount).
- Failure path: I/O error or hash mismatch on first read of a slot → global error latch
  (first-wins, checked per token) → generation aborts with "pack corrupted, re-pack"
  `[swiftlm §4]`. No mid-generation silent retry.

---

## 5. Metal 4 kernel plan

Quant formats v1: GGML-block-compat decode in MSL for `{q4_0, q4_k, q5_k, q6_k, q8_0, iq3_xxs,
iq4_xs, iq4_nl, f16, bf16}` — codebooks (i-quant) loaded to **threadgroup memory** once per
threadgroup, never constant memory ("constant-memory divergent lookups serialise on Apple GPUs",
`[strata §4]`). Parity: each format's MSL decode is checked against a C++ reference transcribed
from `ggml-quants.c` with **exact integer/bit rules** (max_abs_diff = 0 on codes, ≤1e-6 f16 on
outputs) — mirrors Strata's `ref_quant.cpp` harness discipline `[strata §4]`.

Kernel map (decode, T ≤ 8 tokens):
- `gemv.<fmt>.<LPR,NR,SG,KS>` — shared dequant+dot tiles, KS deterministic split-K, activations
  **pre-scaled per format (xprep planes + S16 sums)** so the inner loop is
  `s·Σq·xs − m·xsum` `[strata §4]`. Tile picks = measured table per shape (built by bench harness,
  not hand-guessed).
- `moe.route` — fused router GEMV tail (softmax→top-k→normalize, llama.cpp-bit-faithful tie
  rules) + deterministic expert plan (no atomics) `[strata §4]`.
- `moe.gu` / `moe.down` — per-expert grouped GEMV over the token-plan; expert base addressed via
  `smap[layer*512+e]` → arena slot or resident offset (resident/arena unified in one kernel,
  `kArena` function constant) `[strata §4]`. Reads each row once for all ≤8 window tokens.
- `gdn.recurrent` — fused conv4+SiLU+L2+delta-rule, 128×128 state register-resident per
  threadgroup-per-v-head, NTOK function constant so spec-verify folds accepted tokens in-place
  `[strata §4]`.
- `qsa.*` — fused rmsnorm+rope+**int8-KV append** (fp16 scale per 64 `[strata §4]`), indexer
  scores → `topk(2051)` → split-K flash attention over selected cells.
- `hc.*`, norms, rope, embedding-gather, `sample` (GPU-side argmax/top-p) as in every ref.

Prompt (T > 8): grouped GEMM with **Metal 4 `matmul2d` (MetalPerformancePrimitives)** — dequant to
fp16 threadgroup tiles, fp32 accumulate, 3-pass deterministic plan publishing **indirect dispatch
grids** from the plan kernel itself `[strata §4]` (`moe_mm.metal` design). This is the main place
Metal 4 (vs Metal 3) matters; probed available here (`supportsFamily(Metal4)=1`). M4-applicability
of MPP matmul2d is stated by Strata (M5 tiles exist, M4 runs v2 tiles) — Phase 2 gates it with
`MPPTensorOpsMatmul2d` availability check and a `gemm.metal` fp16 fallback.
Cooperative tensors / `MTLTensor`: not needed for v1 GEMV; revisit for prompt GEMM + residency
integration in Phase 4 (design hook only, no dependency).

Kernel residency: kernels live in the metallib (resident), arena is the only dynamic GPU memory;
no runtime shader compilation in the hot path (PSO cache warmed at model load).

---

## 6. C API (binding surface)

BaseRT-shaped C ABI `[basert §3]`: opaque `m64_ctx`, by-value config struct + `sizeof` handshake
(`m64_model_config_sizeof()`), `m64_model_config(ctx)`, `m64_device_memory_budget()/m64_set_budget()`,
`m64_pack_probe(path, budget_gb) → struct {hot_bytes, arena_max, fits:bool, reasons[]}` (runs the
§3 accounting before allocating), `m64_load(path, opts)`, `m64_generate(prompt_ids, …, m64_stream_fn,
m64_stats*)`, `m64_step/accept` for server use, `m64_stats(ctx, M64Stats*)` (cache hit rate, SSD
GB/s, phys_footprint, swapouts — the AGENTS.md tracked set), `m64_set_spec(ctx, k)`, error model =
`m64_err_t` + `m64_last_error`. C++ internals expose the same via `engine/include/m64/`.

---

## 7. Swap-death analysis (why this design structurally can't spiral)

1. Resident set ≤41.3 GB ≤ Metal ceiling 55.7 GB, with mlock on all anonymous resident bytes —
   nothing the engine holds is ever a swap candidate by construction (`[strata §2]`,
   `[swiftlm §6.1]`: only *Dirty* anonymous pages get swapped/compressed).
2. Expert misses never enter the app heap (arena pread destination) and never touch the page
   cache (`F_NOCACHE`).
3. File-backed resident data (pack file, metallib) is clean → kernel evicts it freely if ever
   needed, it cannot swap.
4. No allocator with hidden retention beyond arena/KV budgets; arena growth re-measures free
   memory stepwise (§3).
5. The known spiral source in every ref: **over-budget anonymous allocations + draft-token fan-out**
   — spec bytes/token computed as `|∪experts over verify batch| × 1.96 MB` and auto-capped at
   `--spec 2` under 40 GB RAM `[swiftlm §1.5]`; Strata's counterpoint (verify tokens ride
   already-paid reads, `--spec 3` gave 28→50.6 tok/s streaming `[strata §5]`) means the cap is a
   measured curve, not a constant — Phase 4 output is the tok/s-vs-spec-k sweep on this box.

---

## 8. Roadmap (Phases 1–4), each with automatable DoD

Build system fixed now: **CMake ≥3.30 + Ninja, Apple clang 21, `-std=c++20 -fobjc-arc`,
deployment target macOS 26** (Metal4 headers present; MTLIO macOS 13+). CI-equivalent = local
script `scripts/check.sh` (configure, build, `ctest`, `scripts/swap_check.sh <bench>`).
Everything below is runnable unattended — AGENTS.md wants numbers, so every phase ends with a
`docs/BENCHMARKS.md` entry written by scripts, not prose.

**Phase 1 — pack format + resident forward + parity (no streaming).**
Build `.m64pack` v1 (spec §2) + `m64pack.py` (GGUF multi-shard in / HF safetensors in), GGML
dequant C++ reference, MSL `gemv`+dense forward, run **Qwen3-30B-A3B Q4_K_M fully resident**.
DoD:
- `cmake -B build -G Ninja && ninja -C build && ctest --test-dir build` green on this box.
- `m64pack.py --verify`: 100% per-tensor xxh64 vs source; pack probe matches header residency block.
- Dequant parity: for every quantized tensor in a 3-layer slice, MSL dequant **bit-exact** vs
  C++ reference (diff = 0 on codes; f16 out ≤1e-6); golden vectors dumped by converter.
- End-to-end parity vs llama.cpp (same quant, greedy 256 tokens, 3 prompts): per-token top-1
  agreement ≥ 99%, final logits cosine ≥ 0.999 (AGENTS.md threshold), perplexity Δ < 1%.
- `scripts/swap_check.sh`: `sysctl vm.swapusage` Δ = 0 and `pageins` Δ = 0 over the parity run.
- Bench line in BENCHMARKS.md: resident decode/prefill tok/s, RSS, swap=0 (expected order: llama.cpp).

**Phase 2 — arena + SSD streaming + per-layer pipeline (still Qwen3-30B-A3B).**
Arena (mlock, chunked NoCopy), expert index, urgent/prefetch queues, `at_layer` shared-event
segments, router-GEMV 2-layer prefetch, F_NOCACHE pool; Flash-Next **reduced-layer slice** kernels
(GDN/QSA/HC) land in parallel with CPU-reference parity harnesses.
DoD:
- Same machine, pack on nvme500, 12 GB arena forced: 256-token greedy generation completes with
  **swap Δ = 0, swapouts Δ = 0**, `vm.swapusage` flat, arena never exceeds budget (stats log),
  30-min soak: no tok/s decay > 10% (Strata's soak bar `[strata §3]`).
- Cache-hit ≥ 60% at 12 GB forced on prose prompt (≥85% at 30 GB); effective SSD GB/s in stats
  matches `iostat` within 20%.
- GDN/QSA slice: cosine ≥ 0.999 per layer vs torch reference (f16), spec-verify fold == 1-token×m
  loop (exactness on accepted sequences).
- Prefetch: measured hit rate of the 2-layer router-GEMV predictor ≥ 70% (gate: if <60%, revisit
  policy before Phase 3).
- MTLIO A/B harness (`m64bench --io=pread|mtlio`): both report GB/s + p50/p95 miss latency
  (decision doc in BENCHMARKS.md, whichever wins becomes default).

**Phase 3 — Flash-Next full model.**
Full 48-layer stream: PLE sidecar reader + row cache, INT8 KV (QSA), HC streams, embed/lm_head q8,
30 GB arena, 128K ctx; `m64pack.py` native quantizer (q6_k/q4_k/q8_0 tiers) replacing
passthrough where better.
DoD:
- End-to-end greedy: 5 prose + 5 code prompts, top-1 agreement ≥ 97% vs llama.cpp/Splash serving
  the *same* UD-IQ3_XXS GGUF on the same box (Splash is the oracle — 1 command, `splash curl`).
- Steady-state decode ≥ **3.0 tok/s** at forced 0% initial arena warmth, ≥ **5.5 tok/s** warm
  (1000-token repeated-window), prompt ≥ 40 tok/s @2K chunk, KV 128K fits budget, **swap Δ = 0**
  across the whole benchmark, RSS(phys_footprint) ≤ 48 GB.
- BENCHMARKS.md full matrix: tok/s vs arena size {10,20,30,38 GB} × ctx {8K,128K} × spec {0,2},
  hit-rate/GB/s columns; machine/flags/versions header (Strata doc discipline `[strata §0]`).
- `m64_stats` exposed via C API + one JSON `/stats` HTTP shim (single endpoint, no server scope).

**Phase 4 — MTP/spec + last-mile perf.**
MTP drafter slot (§2.6) GPU-resident chain (one command buffer draft round `[strata §5]`), spec
controller maximizing committed-tokens/time `[strata §5]`, INT8 KV validated vs fp16 (KL of
sampled tokens < 0.01), MPP tile repack for prompt GEMM (`tile_8x8_mlx` layout) if it beats v2,
optional shadow-MLP predictor behind the existing interface, optional TurboQuant-style 3-bit
history compression `[swiftlm §2]`.
DoD: `--spec 2` on Flash-Next: ≥ +40% decode tok/s over Phase 3 at acceptance rate ≥ 55%,
swap Δ = 0; distribution parity: greedy 256-token outputs byte-identical to `--spec 0`
(exactness rule from SwiftLM's rejection sampling claim); one-page `docs/MAC.md`-style results
table for the box.

---

## 9. Open trade-offs for orchestrator review

1. **IQ3_XXS passthrough vs native q4_k pack for Flash-Next**: passthrough (existing 87 GB blobs,
   oracle = Splash itself) trades ~0.3 tok/s quality/tier balance. Proposal: Phase 2/3 on
   passthrough, Phase 3 also packs q4_k (≈107 GB pack) and the bench matrix picks the default.
2. **Arena at 30 GB vs a larger one**: measured routing locality decides; `--arena-gb` stays
   manual until Phase 2 hit-rate curves exist. IQ4_XS packs (arena ≤ 24 GB then) fit the same
   budget table with smaller resident fraction.
3. **MTLIO**: zero evidence it beats pread on M-series (no ref uses it) but the API is present
   here; Phase 2 spends ≤ 2 days on the A/B and we move on either way.
4. **Single-request scope for v1** (concurrent batching deferred: SwiftLM had to force parallel=1
   `[swiftlm §1.2]`; Strata's `--parallel` numbers are M5-128 only `[strata §7.6]`). C API is
   already request-shaped so multi-seq is additive later.
5. **KV quant depth**: INT8 per-64 storage-only (Strata, low risk, kernels untouched) is the plan;
   TurboQuant 3-bit PolarQuant+QJL is tempting at 1M ctx but was only verified by SwiftLM docs +
   submodule they couldn't open `[swiftlm §7]` — Phase 4 experiment, gated on the KL check.
6. **Packs live on external nvme500** (measured 2.21 GB/s write; read TBD). Internal-disk packs
   would gain maybe +20–30% cold-stream tok/s; disk is only 51 GB after existing blobs — moving
   the IQ3_XXS blobs onto internal (87 GB) and keeping packs external is the likely squeeze;
   Phase 2 bench runs both placements explicitly.
7. **cmake/ninja absent** — Phase 1 step 0 is `brew install cmake ninja` (needs master's brew
   password prompt once).
