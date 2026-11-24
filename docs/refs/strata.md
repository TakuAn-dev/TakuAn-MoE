# Strata (refs/Strata) — analysis for the Mac64-MoE Engine

Analyzed 2026-11-22 at `refs/Strata` (github.com/Niko1221/Strata, shallow clone). All claims below were read
from source or docs in that clone; anything not verified in source is marked *(documented, source not
inspected)*.

Model in the repo's own examples: **Qwen3.8-Flash-Next** ("Flash-Next") geometry (`engine/metal/src/engine/geo.h`
reads it from `strata::core::kCompiledGeometry`): 48 layers, hidden **2560**, routed-expert width **640**,
**512 experts/layer, top-k 10**, 1 shared expert, 4 hyper-connection streams (`HC=4`, low-rank `LR`),
**36 GDN (gated delta-net) layers + 12 full-attention layers** (every 4th, `qsa_interval=4`), attention 24 Q
heads / 2 KV heads / head width 256, a learned **indexer** (4 heads × dim 128) selecting attention cells
(`idx_top_k=2048`, budget **2051 cells** = top_k + tail block's 3), GDN state 128×128 with 16 k-heads / 48
v-heads, conv channels 10240, vocab ≤ 248320 (`geo.h` lines 20–45; `window.cpp:127`).

---

## 0. Repo overview and scale

* ~706k lines of C/C++/ObjC++/MSL under `engine/` (whole engine incl. CUDA, ROCm/HIP, SYCL/Intel, Metal).
* The **Metal ("mac-kern") engine** is `engine/metal/`: kernels in `engine/metal/kernels/*.metal` (~9.5k MSL
  lines incl. `engine/metal/src/engine/msl/`), host runtime `engine/metal/src/` (`window.cpp` 3805 lines,
  `stream.cpp` 827, `mtl_rt.mm` 635, `gguf_lite.cpp` small GGUF reader, `moe_tiles.h`, `ref_quant.h/.cpp`),
  plus `engine/metal/tools/` (Python prep: `metal_repack.py`, `stream_experts.py`), `engine/metal/tests/`
  (40 parity/bench harnesses: `gemv_parity.cpp`, `expert_parity.cpp`, `route_parity.cpp`, `gdn_parity.cpp`,
  `qsa_parity.cpp`, `moe_mm_parity.cpp`, `mma_parity.cpp`, `arena_budget_test.cpp`, `gyro_parity.sh`), and
  `engine/metal/setup_mac.py`.
* The **host/decode loop above the GPU layer is shared with the PC engine** (`engine/src/core/` — verify,
  sessions, spec controller, expert cache; `engine/include/strata/`). `docs/MAC.md:3`: on a Mac "the NVIDIA
  kernels are replaced by Metal ones, and the host code and the server are the PC's" — the Mac runs the engine
  process and exposes HTTP (`docs/DETAILS.md:972` `[model_providers.strata]`, upstream URLs
  `http://my-server:8040/upstream/strata/v1/status`) that a PC server proxies to *(topology documented, exact
  HTTP server source not inspected)*.
* Mac-relevant docs: `docs/MAC.md` (350 lines, all Mac numbers), `docs/DETAILS.md`, `docs/INSTALL.md`,
  `docs/QUANTS.md`, `docs/EXPERTDEC.md`, `docs/MMVQ_IL_TABLE.md`, `docs/HOW_IT_WORKS.md`. Docs
  `DISJOINT_EXPERT_CACHE.md` and `KV_PREFETCH.md` are **CUDA/Linux dual-3080 features, not Mac** (verified —
  validation sections name RTX 3080 / PCIe; only the expert-cache *policy* ideas transfer).
* Measured targets in `docs/MAC.md`: M5 Max 128 GB resident: ~96–100 tok/s decode (IQ3_S+MTP, `--spec 2`),
  1000+ prompt tok/s. **M4 Mac mini 32 GB streaming from SSD**: Q2_0 12.2 tok/s, IQ3_XXS ~8.1, Gyro-S 9.14
  story / 109 tok/s prompt; SSD ≈ **2.97 GB/s for one 1.75 MB read** (line ~327). llama.cpp Q2_0 on the same
  box: 56.6 tok/s — Strata trades single-user speed for running 100B MoE at all.

## 1. Model container ("pack") format — **there is no `.strata` file**

The engine consumes **standard GGUF v2/v3 tensors** (types = GGML's, `engine/metal/src/ref_quant.h:13–17`
mirrors the GGML type enum) with two side artifacts:

1. **`-metal.gguf` repack** (`engine/metal/tools/metal_repack.py`, docstring lines 1–30):
   * Multi-shard GGUF → **one** GGUF, because the engine maps a single file into one GPU buffer
     (`newBufferWithBytesNoCopy` over an `mmap` of the file: `mtl_rt.mm:336/363/386`, `gguf_lite.cpp:67`).
   * Layout: shard-1 metadata (minus `split.*`), then **all dense tensors first, then per layer the routed
     experts grouped: `[gate rows | up rows | down rows]` per expert** — i.e. expert-contiguous inside the
     GGUF. `per_layer_token_embd.weight` (the n-gram PLE table) is deliberately **left out** (stays in its
     shard, read row-wise from SSD via `--ple-gguf`).
   * `general.alignment = 16384` — **Apple's 16 KiB page** (line 28: `ALIGN = 16384`), so every tensor is page
     aligned; writes use `F_NOCACHE` (`nocache()`, lines 36–47, measured with `mincore`: 100%→0% cached).
   * `TYPE_SIZE` table (lines 31–35) covers GGML types incl. Unsloth "Gyro" trellis type 42 and types
     143–146; `--verify` SHA-256-hashes every output tensor against the source shard.
2. **`<gguf>.experts` sidecar ("STRX")** (`engine/metal/tools/stream_experts.py` docstring; consumer
   `engine/metal/src/engine/stream.cpp:84–121`): every expert's gate+up+down slices byte-for-byte from the
   GGUF, laid contiguously per expert, so **a miss = one pread** instead of three (M4 SSD: 2.97 GB/s vs
   2.30 GB/s → +17% decode, +30% prompt, MAC.md line ~69 and the measured table at lines 317–327:
   7.37→8.62 tok/s, 64.7→84.1 tok/s prompt). Header: `"STRX"`, u32 version=1, n_layers, n_experts, u64 source
   GGUF size, then per layer `{file_off, gate_slice, up_slice, down_slice}` as 4×u64. At open, the engine
   validates header sizes **and spot-checks first/last expert bytes against the GGUF** before trusting it
   (stream.cpp:89–110), else falls back to the GGUF.

Profiles: **STRP** `profile.bin` = expert ranking (`include/strata/core/expert_cache.hpp:32–45`: `STRP`,
`version,n_layers,n_expert,slots,n_ranked`, then ranked `(layer,expert)` pairs by routing frequency, then a
frequency table), written by `tools/make_profile.py`; the Mac arena fill consumes an `expert-profile.bin`
(stream.h line 18). A `STRSESS` session-state format exists in the core *(referenced, not re-inspected)*.
`~/.strata` (or `STRATA_HOME`) is just the app's home dir (`docs/INSTALL.md:134`).

Difference vs GGUF: GGUF-compatible header/tensor types + (a) 16 KiB page alignment, (b) expert-contiguous
tensor order, (c) single-file, (d) optional expert sidecar and frequency profile.

## 2. Hot/cold tiers and the Hot Expert Arena (the core of the Mac path)

All in `engine/metal/src/engine/stream.{h,cpp}` + `arena_budget.h`; header doc `stream.h:1–20`.

**Mode selection** — `ExpertStream::wanted()`: streaming is on when `STRATA_METAL_STREAM=1`, or auto when the
GGUF's expert bytes + 12 GiB of working memory don't fit Metal's residency share (~3/4 of RAM, MAC.md
line 38); `=0` forces the fully-resident path (used on the 128 GB M5 Max).

**Hot tier (always resident):**
* Dense weights (everything except routed experts — attention/GDN projections, router `ffn_gate_inp`,
  **shared expert**, embeddings, norms) read **once** into one shared `MTLBuffer` via `load_dense()`
  (stream.h:44–46), which rewrites each tensor's file offset to its buffer offset. The GGUF is *never*
  wrapped whole in stream mode ("a 37.6 GB NoCopy buffer cannot be made resident in 32 GB").
* KV/GDN state, PLE row cache, prompt buffers stay resident.
* In the **non-streaming** path the whole file is wrapped `NoCopy` from its `mmap` (`mtl_rt.mm:336/363`),
  with a clever exclusion: in Gyro files the 28–54 GB PLE table region is *skipped* and only the ranges
  around it are wrapped (`window.cpp:745–760`) — otherwise those file pages would be pulled into wired GPU
  residency.
* `docs/MAC.md:34` says residency uses a "residency set" (macOS 15+); the code comments say the opposite —
  "a residency set alone (requestResidency) did not hold the allocations" (`mtl_rt.mm:49`), "a Metal residency
  set did not hold it either, measured" (`stream.cpp:549`). They wire memory with **`mlock()`** instead
  (stream.cpp:548–570, `STRATA_METAL_WIRE_ARENA=1` default): an mlocked arena "macOS cannot compress while
  the engine idles". **This directly answers the zero-swap question: wire via mlock, size via the budget
  formula below, never rely on the file cache.**

**The arena (cold-expert cache), `make_arena()` stream.cpp:378–540:**
* Slots sized in **16 KiB pages** (`PAGE = 16384`, stream.cpp:26) — one slot = one expert's complete GGUF
  bytes `[gate|up|down]`, unchanged ("the answers are those of the resident engine", stream.h:19).
* **Size classes:** the 48 layers' slot sizes (9 distinct in the IQ3_XXS mix) merged greedily so no member
  wastes >15% (`STRATA_METAL_CLASS_WASTE`) of its slot; every class keeps ≥ `min_slots` slots — **64 normally,
  576 (= a whole layer) when prompt chunks run** (stream.h:48–49, stream.cpp:130–158, 418). One LRU list per
  class.
* Arena = up to **4 anonymous `MAP_PRIVATE` `mmap` regions** wrapped zero-copy
  (`cx.wrap()` → `newBufferWithBytesNoCopy`), each ≤ min(`maxBufferLength`, ~8 GiB or total/4+4·biggest)
  (stream.cpp:423–470; "the M4 refused one 20.6 GB buffer" line 424–426). Kernel-side addressing:
  `smap[layer*512+expert] = (chunk << 28) | pageInChunk`; `arena_slot()` maps chunk→buffer 0/7/8/9 and
  returns `base + page*16384` (`moe.metal:55–67`, buffer(7..9) params at `moe_gu`). `0xFF`-filled smap = not
  resident.
* **Budget** (`arena_budget.h`, fully commented + unit-tested in `tests/arena_budget_test.cpp`):
  `keep = max(7 GB, 15% RAM)` for macOS/desktop (M5 Max froze with 89 GB arena + 7 GB kept beside a 50 GB
  table); whole-process anonymous footprint capped at 60% of RAM on big Macs; "available" counts
  free+purgeable+**file-backed** inactive only (other programs' inactive anon excluded — measured a 93 MB
  swapout otherwise); arena filled in 1.5 GB steps, re-asking free memory before each step and stopping
  (`arena_grow`). `STRATA_METAL_ARENA_GB` overrides.
* **Seeding:** ranked STRP pairs first (best-first, each step in file order, on the I/O threads), then
  layer-interleaved pairs.

**Policy per decode step** (`at_layer()` stream.cpp:616–700+, called at every layer cut):
* stamp per step; slots used this step are pinned (`pin = stamp`) so nothing evicts a just-hit expert.
* Wanted = distinct experts this window routed to; pass 1 pins resident ones, pass 2 reads misses into LRU
  **victims of the same size class** (never evicts a pinned slot; per-class LRU).
* Unwanted, not-yet-started read-aheads are **cancelled** (`cancel_slot`) and wanted-but-queued ones
  **promoted** to the urgent queue front (`promote_slot`).
* **Predictive prefetch:** while layer *l* computes, a tiny extra GEMV of layer *l*'s FFN input against layer
  **l+1's and l+2's** router weights produces guessed logits (`window.cpp:1294–1301`, buffers `predL_/predL2_`
  48×8×512); `ahead()` (stream.cpp:577–614) takes top-K(≤32) guessed experts per token, dedups by best logit,
  and streams them in LRU front (`pre_mru_`) or tail — tuned per expert size: top-10 front for small experts
  (Q2_0 1.38 MB: 12.2 vs 11.3 tok/s), top-6 tail for big ones (IQ3_XXS mix: 8.06 vs 7.43) (stream.cpp:161–171,
  measured in `phaseA-tests/notes-mac-ssd.md`).
* Host blocks only inside the current layer's cut until that layer's misses land (`done_cv_`), so I/O wait is
  bounded to one layer's experts.

**GPU↔host handshake per layer** (`mtl_rt.mm:517–558`, `run_graph_seg`): the whole 48-layer decode window is
**one command buffer split into 48 segments**; between segments `[cb encodeSignalEvent:ev value:2s+1]` /
`[cb encodeWaitForEvent:ev value:2s+2]` on a single `MTLSharedEvent`; host spins on `ev.signaledValue`, runs
`at_cut` (= stream `at_layer`, updates `smap`, issues I/O), then bumps the value to release the next segment —
no buffer round-trip, no new commits. `MTLDispatchTypeConcurrent` encoders + explicit
`memoryBarrierWithScope(MTLBarrierScopeBuffers)` between dependent dispatches. Host completion uses a 20 ms
status **spin** before `waitUntilCompleted` (+5% tok/s measured, `mtl_rt.mm:460–479`). A persistent-kernel
grid-barrier variant was probed and rejected (Metal doesn't guarantee co-resident threadgroups —
`kernels/gridbar.metal` docstring).

**Old PC design** (`include/strata/core/expert_cache.hpp`, CUDA engine): VRAM expert tier with the same
residency table idea — 4105 slots sized from a **10-fold leave-one-out hit-rate 0.6447**, "h_layer 0.0456 →
a per-layer all-resident kernel is impossible, the hit/miss split is mandatory". Useful design rationale,
not Mac code.

## 3. SSD I/O pipeline

* **APIs: plain POSIX.** `open(O_RDONLY)` + `fcntl(F_NOCACHE,1)` + `fcntl(F_RDAHEAD,0)` (stream.cpp:62–65),
  `pread` loops capped at 64 MiB (stream.cpp:170–179). **No** `MTLIOCommandQueue`, no `dispatch_io`, no
  `aio`, no `DISPATCH_USE_IO_PARALLELISM` anywhere in the Metal engine (grep-verified); zero-copy is only
  the `mmap`+`newBufferWithBytesNoCopy` wrap for resident data. Reads land in the mlocked anonymous arena
  bytes that the GPU reads directly (host pointer = wrapped buffer pointer, `host_[s] = chunks_[c].ptr +
  page*16384`, stream.cpp:466–469) — **one copy, no staging**.
* **Parallelism:** `STRATA_METAL_IO_THREADS=6` default worker threads; `STRATA_METAL_READ_SPLIT` (1..8)
  slices **one expert across several threads** "so a single miss is read with several requests in flight
  (its latency, not only the SSD's throughput, is what a layer waits for)" (`read_part` stream.cpp:277–308;
  slices split on 4 KiB floors). Urgent queue preempts prefetch queue (`io_main` stream.cpp:311–336).
* **Read sizes:** one contiguous ~1.4–2.4 MB read per expert miss (STRX, split per thread); dense load is one
  big sequential pass; PLE rows are **4 KiB direct reads** (below).
* **Overlap:** three kinds — (1) predictive next-layer/next-next-layer prefetch during compute (above);
  (2) in-flight reads for *other* experts while the host waits on the current layer; (3) PLE: issue reads the
  moment the token id is known, collect before layer 1 (`ngram/ple_reader.hpp:8–14`), and for prefill
  issue an entire chunk's rows, deduped by page and offset-sorted, capped `max_inflight`, "so a chunk's reads
  can run while the previous chunk computes".
* **n-gram PLE table:** 320,001,536 rows × 90 B (IQ4_NL, 26.8 GiB) or 160 B (FP8 E4M3, 51.2 GB), never in
  RAM; 16 rows/token on 16 pages; bounded **row cache** (1M rows ≈ 95 MB, clock eviction; serves 20–34% of
  reads within one long prompt, up to 82% on a frozen corpus — `ple_reader.hpp` header +
  `bench/results/2026-09-23-ngram-io` ref). Mac default = row-wise F_NOCACHE reads (`--ple-io ram` to map
  whole; budget reserves 0.5 GB for the cache, `arena_budget.h` "ple"). Keep-alive reads exist to stop the
  SSD sleeping (`ReaderStats.keepalive_reads`).
* **Fill-time discipline:** arena fill goes ranked-first in steps with re-measured free memory (stream.cpp:
  485+), prints GB/s, and `docs/MAC.md:325+` documents soak tests (34-min server run, no slowdown, no
  swapouts) as the acceptance bar.

## 4. Metal kernels and quant formats

**Format abstraction** (`engine/metal/kernels/quant.metal`, `units.metal`): every weight format is a
`fmt<F>` with `(QK, BB, UNIT, TAB)` (quant.metal:58–88): Q4_0/Q5_0/Q5_1/Q8_0 (QK 32), Q3_K/Q4_K/Q5_K/Q6_K,
Q2_0 (64-block Strata contract), IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS, BF16/F32, and
agentionai "Gyro" trellis TQ2_T/TQK6/TQK7/TQP5-7 (QK 128, 2048-entry threadgroup LUTs). One **unit** = UNIT
consecutive weights one lane decodes per step (16 or 32); codebook formats (i-quants, trellis) load their
tables **to threadgroup memory once per threadgroup** because "constant-memory lookups with divergent
indices serialise on Apple GPUs" (quant.metal:1–13). Decode rules transcribed from
`ggml-quants.c @3cf03257`; blocks are 2-byte-aligned so fields are read as ushorts.
* **Prepared activations (xprep)** (`units.metal:31–70`, `glue.metal: swiglu_prep`): activations are
  pre-scaled per format into planes (raw; `x·4^-i` for Q2_0's mask trick; `x·2^-8` at odd i for K-quants) +
  per-16 sums `S16` for the offset/min terms, so the dot is `s·Σ q·xs − m·xsum` with **no per-row x math**;
  a half-precision variant rides function constant 25 `STRATA_METAL_F16` (`quant.metal:17–30`, auto-on for
  resident Q2_0/IQ3_S decode — `window.cpp:739`).
* **Decode GEMV** (`gemv.metal`): template `<F, LPR, NR, SG, KS>` — SG simdgroups per threadgroup, KS split
  K (deterministic fixed-order partial adds), LPR lanes share a row and each lane holds NR rows; one decoded
  weight unit is applied to **all ≤8 window tokens**. NTOK is a function constant so token loops specialize.
  Tile picks are a measured table in `window.cpp:1029–1046` (e.g. M4 Q5_K 2560→10240: `<2,1,4,1>` −20% vs
  `<16,1,4,1>`; kernel name `gemv_<type>_<LPR>_<NR>_<SG>_<KS>`).
* **Decode MoE** (`moe.metal`, ≤8 tokens): `moe_plan` (one 512-thread group, deterministic, no atomics) groups
  (token,k) entries by expert into a plan buffer; `moe_route` **fuses the router tail** (softmax over 512,
  top-10, normalized weights — bit-faithful to llama.cpp `topk-moe`, incl. tie→lower index and the
  `max(sum, 2^-14)` clamp) + plan in one dispatch; `moe_gu` / `moe_down` read each expert row **once** for
  all entries of its group (NE specialized 1..4), expert bytes via either resident addressing (`W + e*stride`)
  or the arena (`kArena` fc 20 + `smap`/buffers 7–9). `moe_combine` + shared-expert glue in `glue.metal`.
* **Prompt MoE GEMM** (`moe_mm.metal`, 1496 lines): grouped GEMM via **Metal 4 MetalPerformancePrimitives
  `matmul2d`** (`#include <MetalPerformancePrimitives/...>`, `mpp::tensor_ops`); deterministic 3-pass plan
  (`mm_hist`/`mm_scan`/`mm_place`, 1024-entry blocks, padding to TM rows) that **publishes the indirect
  dispatch grids** (`meta[2..7]`) so kernel and grid can't drift (postmortem in `moe_tiles.h:8–14`); tiles
  dequant TN×TK weights to fp16 + convert activations to fp16 into threadgroup memory, matmul2d with fp32
  accumulation; per-format tile table in `moe_tiles.h` (K-quants TK 128, Q8_0/IQ4_XS TK 32; v3 M5 tiles use
  the M5 "right-input layout" — on M4 the v2 tiles run). Prompt runs in chunks via
  `prompt_chain.inc`; `STRATA_METAL_PROMPT=window` is the slow exact fallback (~190 tok/s vs 1000+).
* **GDN (linear attention)** `gdn.metal`: whole per-head recurrence fused — conv(4 taps)+SiLU, L2 norms,
  `gate=exp(softplus(...))`, delta rule, register-resident 128×128 state (one threadgroup per v-head = 16
  simdgroups each holding 32×32 state), ≤8 tokens without commit ("a verify window … the fold of m accepted
  tokens is the same kernel with GDN_NTOK=m"), ping-pong conv history; `GDN_ROWS` fc 31 batches slots; M4
  gets a tuned prompt-scan kernel (`gdn_scanr_4_8`, `window.cpp:75–87`).
* **QSA (sparse full-attention)** `qsa.metal`: fused rmsnorm+RoPE+**int8 KV append** (per 64 values:
  `scale=fp16(amax/127)`, clamp rint — `kernels/kv_q8.hpp:8–14`; 12,672 B/token over 12 layers vs 24,576 fp16:
  3.3 GB @262K); indexer scores → `qsa_topk` (all cells ≤2051, else learned top-k + tail); split-K flash
  attention `qsa_attn` (12 heads/CG, `s = q·k/16`). Dequant happens into the fp16 scratch the attention
  kernels already read — only storage is quantized (`kv_q8.hpp:12–15`).
* **Misc:** GPU sampling (`sample.metal`, `samp_pick`), fp32 accum throughout, `expert_parity`/`gemv_parity`/
  `route_parity` etc. validate kernels against `ref_quant.cpp` CPU reference (AGENTS-style parity culture);
  `STRATA_METAL_STREAM_FAKE` mode answers wrong but real-timed for pure-IO benches.

## 5. MTP / speculative decoding

* Decode windows carry **1..8 tokens** (`MAXT = 8`, `window.cpp:127`); speculative control lives in shared
  core code with a cost model (`include/strata/spec/controller.hpp`): each step maximize
  `E[tokens committed]/T(step)` over {none, suffix lookup, MTP≤8}, EMA per-position acceptance, keep k=0
  unless gain >5% (`K_MAX=8`, measured `CostModel` defaults lines 26–36: dense ratios, distinct-expert ratios,
  `sync_ms=2.4`, `mtp_draft_ms=1.2`). Drafters: **MTP head** and **suffix drafter** (`spec/suffix_drafter.hpp`,
  `draft_policy.hpp`).
* On Mac the MTP draft layer runs **on the GPU**: "the MTP draft layer (mtp.hpp's math on mac-kern's kernels;
  the MTP GGUF from `tools/mtp_pack.py`)" (`window.cpp:292–296`); its BF16 matrices are stored **Q8_0** —
  half the bytes a draft step reads, "drafts only" (`window.cpp:621`). A **whole draft round is one command
  buffer**: `mtp_chain.metal` = `mtp_pick` (draft token + its softmax prob via the draft vocabulary `vmap`)
  + `mtp_embed_F` (next step's input = picked token's embedding row dequantized on GPU, previously host work)
  — "the two steps that keep a whole MTP draft round on the GPU" (`msl/mtp_chain.metal:1–16`).
* A second drafter, **DFlash2** (`msl/dflash2.metal`, 852 lines + `src/kernels/cuda/dflash2.cu` parity):
  a small transformer (rmsnorm, 2-tap dynamic conv, rope, ring-KV windowed attention, top-16 heads, candidate
  selection walk) with Q8_0 GEMVs over a "DF2P pack"; tuned Mac config uses `--spec 6 --draft-conf 0.7`
  (MAC.md:100).
* `--spec N` Mac results (`docs/MAC.md`): default `--spec 2`; M5 Max IQ3_S story 106 (spec2) vs 92 (spec3),
  prompt1 111 vs 116 (308–313); on the **streaming** M4 spec decode shines because extra tokens ride already
 -paid SSD reads: Q5 story 28 → 41.8 (`--spec 2`) → 50.6 (`--spec 3`) tok/s (line 200). Streaming Macs <96 GB
  skip MTP by default (line 97) but keep suffix; ≥96 GB writes `--spec 3`.
* Verify = the ≤8-token window kernels themselves; accepted-token folding of GDN state is the same GDN kernel
  with `GDN_NTOK=m` (gdn.metal:13); batch slots share the window (`--parallel N`: M5 Max 1→4 req = 85→119
  aggregate tok/s, 8.3→16.0 GB; MAC.md:243).

## 6. What Mac64-MoE should steal (with refs)

* **16 KiB page = the universal alignment** across pack, arena slots, `smap` addressing
  (`metal_repack.py:28`, `stream.cpp:26`, `moe.metal` `arena_slot`) — and `F_NOCACHE` on both write (repack)
  and read paths, with the measured "dirty-page floor" rationale (`metal_repack.py:36–47`) and
  `F_RDAHEAD 0` for random expert reads.
* **Expert-contiguous layout + STRX sidecar**: one pread per miss, +17% decode / +30% prompt measured
  (`stream_experts.py` docstring; `stream.cpp:84–121`) — plus the paranoid open-time byte validation of the
  sidecar against the GGUF before use.
* **Arena as anonymous `mmap` + `newBufferWithBytesNoCopy` + `mlock`**: zero staging copies, and **mlock, not
  MTLResidencySet** (two measured comments: `mtl_rt.mm:49`, `stream.cpp:549`); ≤4 chunks ≤ maxBufferLength
  with `(chunk<<28)|page` slot codes because the M4 rejected a single 20.6 GB buffer (`stream.cpp:423–426`).
* **The memory budget formula** (`arena_budget.h`): keep `max(7 GB, 15% RAM)`; count file-cache-inactive only
  if file-backed; stepwise fill re-measuring free memory; whole-process footprint cap — all pure arithmetic,
  unit-tested without a Mac (`tests/arena_budget_test.cpp`).
* **Per-layer GPU↔host handshake with one command buffer + MTLSharedEvent signal/wait pairs**
  (`mtl_rt.mm:517–558`) — 48 cut points, host spins on `signaledValue`, updates `smap`, releases next segment;
  no per-layer commit overhead.
* **Router-logit-predicted prefetch one and two layers ahead** from a cheap router GEMV on the current FFN
  input (`window.cpp:1294–1301` + `stream.cpp ahead()`), with the measured placement nuance (prefetch at LRU
  front for small experts, tail for big; top-6 vs top-10, `stream.cpp:161–171`) and **cancel/promote** of
  speculative jobs (`cancel_slot`/`promote_slot`).
* **Size-class LRUs with per-class minimums** (64 / 576-for-prompt) so a prompt chunk can pin a whole layer
  (`stream.cpp:130–158`, `stream.h:48–49`) and greedy class merging capped at 15% waste.
* **Kernel design**: single fused decode-MoE with `moe_route` fusing the router tail + plan (deterministic,
  no atomics, `moe.metal:119–186`); expert rows read once for all window entries (`gu_body` NE
  specialization); the **xprep planes + S16 sums** trick so mask-decoded codes multiply pre-scaled x
  exactly (`units.metal:31–70`); threadgroup-memory LUTs for i-quant/trellis decode ("constant-memory
  divergent lookups serialise on Apple GPUs", `quant.metal:10–12`); measured per-shape tile tables
  (`window.cpp:1029–1046`, `moe_tiles.h`) — with the drift postmortem that indirect grids and tiles must be
  returned together (`moe_tiles.h:8–14`).
* **Prompt path**: MetalPerformancePrimitives `matmul2d` grouped GEMM with GPU-published **indirect
  dispatch** args (`moe_mm.metal`), T>8 → GEMM path and the same streaming arena feeding it (min_slots 576).
* **INT8 KV** (fp16 scale per 64, storage-only change so attention kernels stay untouched,
  `kernels/kv_q8.hpp` + `qsa.metal` fused append) — halves the hot tier; 2051-cell top-k sparse attention to
  bound KV work (geo.h `BUD`).
* **Speculative economics**: controller maximizing committed-tokens/time with EMAs (`spec/controller.hpp`);
  Q8_0 draft weights; keeping whole draft rounds in one command buffer (`msl/mtp_chain.metal`); the M4
  insight that **streaming models want big spec windows** because verification tokens share already-paid I/O
  (MAC.md:200).
* **PLE/n-gram table**: never map it, issue-token-triggered 4 KiB reads + collect-after-layer-0 API, bounded
  row cache with measured hit rates (`ngram/ple_reader.hpp:1–20`).
* **Discipline**: every kernel has a parity harness vs a CPU reference (`engine/metal/tests/*_parity.*`),
  benchmarks are medians of ≥5 interleaved runs with named configs, and docs carry the machine/flags/versions
  per number (`docs/MAC.md` "Measured on" blocks).

## 7. Open questions

1. `MTLIOCommandQueue`/`dispatch_io` are used **nowhere** — is plain `pread`+F_NOCACHE+6 threads simply at
   the M4 NVMe ceiling (2.97 GB/s at 1.75 MB reads suggests near-SATA-ish or QoS-capped single-stream;
   multi-stream scaling never quantified in the docs), or does Apple's I/O kit path add real headroom we
   should prototype first?
2. The M5-128GB "resident" mode wraps the file NoCopy but relies on Metal's ~3/4-RAM residency heuristic;
   on a 64 GB M4 Max a 40–48 GB pack leaves only ~16–24 GB for arena+KV+macOS — what does
   `arena_budget`/`keep=max(7 GB,15%)` actually yield there (7–9.75 GB kept, 60% cap = 38.4 GB footprint:
   plausible but untested at 64 GB in any doc number; all Mac numbers are 32/34/128 GB)?
3. `moe_gu`'s arena path indexes `smap[ex]` per threadgroup and reads expert bytes from a *shared* (CPU
   coherent) buffer — do they ever benchmarked `MTLResidencySet` for just the arena chunks *while* using
   mlock for the rest, or measured purgeable/voluntary-ramp effects? (Code only shows the negative result.)
4. The layer-cut spin serializes: per-layer GPU stall = router-GEMV→host→I/O→release. What is the actual
   cut-overhead share at 12 tok/s (11 ms/layer budget) on M4 vs the pure-read time (`host_ms_`/`done` stats
   exist; no published split)?
5. Gyro trellis formats (types 42/143–149, TQP5–7) — encode-side quality math not in this clone
   (`tools/gen_tq_lut.py`, `docs/UNSLOTH_Q4.md` partially); relevance for our packs?
6. Multi-request on a *streaming* Mac: min_slots 576/class + per-sequence KV — is concurrent `--parallel`
   tested there (MAC.md numbers are M5 Max only)?
7. The exact Mac↔PC serving topology (HTTP upstream + `strata-gguf-preflight` Rust helper contract
   `terminal/crates/strata-gguf-preflight/src/main.rs`) is documented but the server glue (`terminal/`)
   wasn't fully read — needed only if we mimic the split-host design.
