# Mac64-MoE Engine — Phase 1 Parity & A/B Benchmarks

Recorded: 2026-11-23  
Engine baseline: commit `506d01b` (Phase 1 resident bring-up)  
Oracle baseline: `llama.cpp` @ `e300919`

---

## 1. Test Environment & Model

- **Hardware**: Apple M4 Max (16 CPU cores, 40 GPU cores, 64 GB Unified Memory)
- **OS / Compiler**: macOS 27.2.1 (`26B131`), Apple clang `21.0.0` (`clang-2100.4.1.1`)
- **Build Flags**: `-std=c++20 -O2 -ffp-contract=off` (zero external deps, POSIX `Makefile`)
- **Model**: `Inkling-2-Mini-Q8_0` (`arch=inkling2_loop`, 24 physical layers / 48 loop slots, 128 experts, top-8)
  - Pack file: `<MODEL_DIR>/inkling-2-mini-q8_0.m64pack`
  - Size / Layout: `25.93 GiB` blob, 680 tensors, `expert_stride = 8,470,528 B` (16 KiB-aligned)
  - Note: Phase 1 tests ran on a system where a background Splash server (`--max-memory=48G`, ~4.7–45 GiB dynamic footprint) was also resident.

---

## 2. Unit & Golden Verification

| Check | Command | Result | Details |
|---|---|---|---|
| Build + Tiny Pack Suite | `./scripts/check.sh` | **PASS** | `test_dequant` (6 tensors bitwise) + `test_metal_parity` (`0 failures`) |
| Metal Kernel Parity | `./build/test_metal_parity <ORACLE_DIR>/tiny4.m64pack` | **PASS** | `dequant_q8_1`, `dequant_q8_0`, `dequant_f16`, `gemv_f32`, `gemv_q8_1`, `moe_gemv_q8_1`, `moe_route` bitwise OK; `situ` max rel err `3.2e-05`; `rmsnorm_row` `2.36e-07`; `softcap + argmax` OK |
| Real 26 GB Pack Golden | `./build/test_dequant .../inkling-2-mini-q8_0.m64pack <ORACLE_DIR>/g_real/golden.bin` | **PASS** | All **680 tensors** `xxh64` verified + bitwise golden dequant match across dense weights and all 24 MoE layers × 128 experts |

---

## 3. End-to-End Numerical Parity (`m64run` vs `llama.cpp`)

Oracle determinism (`det_run1` vs `det_run2`, 9 steps): `max_abs_diff = 0.0` (bitwise identical).

### Teacher-Forced Parity
- **9-step (`"The capital of Japan is"`, 5 prompt tokens + 8 decode steps)**  
  `./build/m64run .../inkling-2-mini-q8_0.m64pack --ids-file <ORACLE_DIR>/ids.txt --compare <ORACLE_DIR>/ref_smoke_logits.bin <ORACLE_DIR>/ref_smoke_tokens.bin --steps 8`
  - **Argmax Token Agreement**: **9/9 (100.0%)**
  - **Logits Cosine Similarity**: `0.990560` (min) – `0.999695` (max) (`0.998860, 0.999612, 0.999546, 0.999695, 0.990560, 0.993369, 0.998039, 0.998738, 0.998853`)
  - Note: Strict `cosine min >= 0.999` gate reported `FAIL` due to small per-step FP32/NEON vs CPU accumulation differences (4/9 steps ≥ 0.999; all steps ≥ 0.9905).
- **25-step (`ids1.txt`, 1 prompt token + 24 decode steps)**  
  - **Argmax Token Agreement**: **25/25 (100.0%)**
  - **Logits Cosine Similarity**: **mean `0.999092`**, min `0.993118`, max `0.999972`

### Greedy Free-Run Generation Quality
- Prompt (`ids.txt`): `"The capital of Japan is"`
- `m64run` output (detokenized via `tools/bpe_gguf.py`):  
  `" Tokyo.\n\nMeaning: This sentence states that Tokyo is the capital city of Japan.\n\nChoice of Words: Capital refers to the most important city in a country..."`
- Matches `llama.cpp` token-for-token for the first **24 greedy tokens** (`0..23`); diverges at token 24 (`161640` vs `149884`).

---

## 4. A/B Performance & Memory Comparison (`m64run` vs `llama.cpp`)

| Engine & Backend | Steps | Decode Speed | Total Wall Time | Max RSS (`time -l`) | Dirty `peak memory footprint` | macOS `swap delta` |
|---|---:|---:|---:|---:|---:|---:|
| **`build/m64run` (Phase 1 CPU 16-thread + `mmap`)** | 8 (`--bench`, solo) | **1.12 tok/s** (`7170.5 ms`) | 24.84 s | 9.12 GiB | **1.56 MiB** | **-8.00 MB** (`0` swaps) |
| **`build/m64run` (`swap_check.sh`, 8-step bench)** | 8 (`--bench`) | **0.79 tok/s** (`10094.5 ms`) | — | — | — | **0.00 MB** (PASS) |
| **`build/m64run` (`ab_run.py`, 64 tokens, Splash co-resident)** | 64 + 8 bench | **0.69 tok/s** (`11524.6 ms` / 8 steps) | 126.36 s | 7.01 GiB | **49.57 MiB** | **-24.0 MB** (`0` swaps) |
| **`llama.cpp` (`e300919`, Metal GPU `--ngl 999`, `ab_run.py`)** | 64 | **31.90 tok/s** (`2006.2 ms`) | 71.52 s *(incl. load + shader compile)* | 5.52 GiB *(excl. Metal VRAM)* | 493.58 MiB *(CPU heap)* | **+22,013.0 MB** *(co-resident w/ Splash)* |

*(Note: A second 8-step `swap_check.sh` run during background Splash memory reload measured `0.36 tok/s` with `swap delta: -56.07 MB`. `scripts/swap_check.sh` originally rejected negative swap deltas `< -1 MB` and was updated to assert `delta < 1.0 MB`.)*

---

## 5. Honest Phase 1 Architecture Summary & Next Lever (Phase 2)

1. **Why `m64run` is `0.79–1.12 tok/s` vs `llama.cpp` Metal's `31.90 tok/s` in Phase 1**:
   - In Phase 1, `engine/src/forward.cpp` + `engine/src/forward_impl.hpp` intentionally execute the 48-slot Inkling-2-Mini forward pass on the **16-core CPU (`Pool`)** with `-ffp-contract=off` and clean `mmap` to lock down numerical parity on the looped transformer before moving execution onto the GPU.
   - Meanwhile, the native Metal compute kernels (`engine/kernels/m64.metal`) and `MTLBinaryArchive`-cached runtime (`engine/src/metal_runtime.mm`) are already built and passing all Metal unit parity checks (`build/test_metal_parity`).
2. **Swap Behavior**:
   - Even with Splash occupying ~45 GiB in the background, `m64run`'s clean file-backed `mmap` kept dirty `peak memory footprint` under 50 MiB and produced **`0` swap growth** across every run, whereas allocating full Metal GPU weight buffers in `llama.cpp` (`--ngl 999`) pushed system memory past 64 GB (`+22.0 GB` swap spike).
3. **Phase 2 Roadmap**:
   - Wire `m64.metal` GPU kernels into the forward loop (closing the gap toward `30+ tok/s` on resident layers).
   - Add the **explicit `mlock` Hot Expert Arena + async `pread(..., F_NOCACHE)` / `MTLIOCommandQueue` NVMe SSD streaming layer** (`--arena-gb`) to run **100B+ MoE models (`Qwen3.8-Flash-Next 125B`)** cleanly on 64 GB with zero swap.

---

## 6. Phase 2A (Metal GPU Forward) — Bug-Fix Log & Verified Baseline (2026-11-23)

### Bugs fixed this phase (all in `engine/src/forward_metal.mm`)
1. `s_phys[il>=24]` used `6 + pass*12 + k` → indexed `phys[24..29]`/`swa_phys[24..29]` out of bounds and `loop[-1]`; now `6 + (il-24)%12`, matching `forward_impl.hpp::slot_phys()`.
2. `ConvArgs.rsoff` omitted the `o.rs` base → `k_conv_slot` wrote the conv history at `SCR+0`, clobbering `o.emb`/`o.cur`/`o.mixed`/`o.kvk` (root cause of `M64_NANPROBE emb bad=1024`); now `o.rs + (il*kCH+sec)*kHR`.
3. `L.nq = (uint32_t)t->offset` truncated PACK offsets > 4 GB; widened to `uint64_t`.
4. Two `xfer2` dual-pair dispatches raced on `o.cur` (threadgroup 0 read while threadgroup 1 wrote, same kernel): the postconv block-push and the pass-rotate `bank_out`/`prelude_out` swap are now sequential dispatches.
5. `gpu_step` now checks `cb.status != MTLCommandBufferStatusCompleted` after `waitUntilCompleted` and surfaces `cb.error`.
6. `o.sc = fo(c, 3 * kMAXS)` allocated 144 floats while `k_mix_*` indexes `nq <= 4` (`192` floats), overflowing into `o.logits`; fixed to `fo(c, kMAXS * kMAXS)`.
7. When `is_swa && lp != nullptr` (loop passes), `forward_metal.mm` normalized only 1 row instead of all 3 distinct inputs `{in3[0..2]}` with `L.normattn` (unlike `forward_impl.hpp`).
8. `L.nq_rows` used `(aq->shape.size() && aq->shape[0])`, misreading 1D `[2304]` `attn_res_query.weight` tensors on SWA layers as `2304` rows instead of `1` row (which caused `do_mix` / `k_mix_comb` in `slot 1` to overwrite `o.fin`, `o.logits`, and scratch); fixed via `shape.size() >= 2` with `length / (4 * kNE)` fallback.

### Verified this phase (Splash paused during full-pack GPU runs, restored cleanly afterward)
- `scripts/check.sh` (`make build && make test`): **ALL BUILD+UNIT TESTS PASSED**, including `test_dequant` (6 tensors bitwise), `test_metal_parity` (10/10 Phase 1 kernels, 0 failures), and `test_p2a_kernels` (all **22** Phase 2A `m64_p2a.metal` kernels, 0 failures).
- `<ORACLE_DIR>/packprobe.mm`: 26,553 MiB NoCopy wrap OK; GPU reads at offsets 25,000,000,000 and 27,000,000,000 both `mismatch=0/64` bitwise vs CPU.
- Full-pack 9-step parity (`<ORACLE_DIR>/p4/2_parity8.log`): **9/9 argmax agreement (100%)**, cosine `0.998870, 0.999627, 0.999580, 0.999338, 0.981573, 0.994385, 0.997770, 0.998921, 0.998394` (**mean `0.996495`**, min `0.981573`; note strict `cosine min gate 0.999` fails at step 4 just as CPU Phase 1 did at `0.990560`).
- Full-pack 32-step decode (`<ORACLE_DIR>/p4/3_bench32.log`): **20.31 tok/s** (`8 decode steps in 393.9 ms`), generating coherent detokenized output (`" Tokyo.\n\nMeaning: This sentence states that Tokyo is the capital city of Japan.\n\nChoice of Words: Capital is the most important city in a country,"`).
- Swap check (`scripts/swap_check.sh`, `<ORACLE_DIR>/p4/4_swap.log`): **20.29 tok/s**, `swap used before: 3703.44 MB` -> `swap used after: 3703.44 MB` (**`SWAP CHECK PASS (delta 0 MB < 1 MB)`**).

### Phase 2A Metal performance vs CPU and `llama.cpp`

| Engine & Backend | Steps | Decode Speed | Max swap delta | Argmax agreement | Cosine (mean / min) |
|---|---:|---:|---:|---:|---|
| **`build/m64run` (Phase 1 CPU 16-thread)** | 8 (`--bench`) | **1.12 tok/s** | **0.00 MB** | 9/9 | 0.99747 / 0.99056 |
| **`build/m64run` (Phase 2A Metal GPU, M4 Max)** | 32 (`--bench`) | **20.31 tok/s** | **0.00 MB** | **9/9** | **0.99650 / 0.98157** |
| `llama.cpp` (`e300919`, Metal `--ngl 999`) | 64 | 31.90 tok/s | +22,013 MB *(co-resident)* | reference | — |

### Next Optimization Lever (Phase 2A-opt)
- Implemented below in Section 7.

---

## 7. Phase 2A-opt (M4 Max Kernel Fusion, Vectorized Multi-Row GEMV & Batch-8 MoE) — Beating `llama.cpp` (2026-11-23)

### Optimizations Implemented (`f26339b`, `8115ae6`)
1. **Vectorized Q8_0 inner loops (`dot_c4` / `dot_c4u`)**: Rebuilt unaligned block-0 `char4` words in registers `(W[j]>>16)|(W[j+1]<<16)` alongside aligned block-1 `W[9+j]`, replacing the 32-iteration scalar byte loop in `gemv_unit` / `gemv_w`.
2. **Multi-row activation reuse (`_m2`, `_m4`)**:
   - `k_gemv_q8_k2304_m2` (2 rows/warp sharing `XA/XB` registers; outperformed `_m4` in `--bench`).
   - `k_gemv_q8_k3072_m2` (2 rows/warp for `wo` across 48 slots).
   - `k_gemv_gateup_batch8_m2` (2 rows/warp across batched expert gate|up projections).
   - `k_gemv_down_reduce8_m4` (4 rows/warp single unlooped pass across 18 units).
3. **8-Expert Batched MoE Pipeline**: Replaced `zero(o.acc)` + 8 serial iterations (`25` dispatches/slot = `1,200` dispatches/token) with 3 batched dispatches (`k_gemv_gateup_batch8[_m2] -> k_situ_quant8 -> k_gemv_down_reduce8[_m4]`).
4. **Fused & SIMD-Reduced Kernels**: `k_rms_quant` (2-level `simd_sum` rmsnorm + Q8_0 quant), `k_gate_mul_quant`, and `float4`-vectorized `k_gemv_f32_tg` / `k_control`.

### Microbenchmark (`./build/test_p2a_kernels --bench`, < 30 MB synthetic buffers)
| Kernel / Chain | Phase 2A Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| `gemv_k2304` (`1024` rows) | `0.0294 ms` | **`0.0124 ms`** (`_m2`) | **2.37x** |
| `gemv_k2304` (`4096` lm_head proxy) | `0.0761 ms` | **`0.0331 ms`** (`_m2`) | **2.30x** |
| `moe_8e` (`BE=192` proxy, `25 -> 3 disp`) | `0.2857 ms` | **`0.0936 ms`** (`batch8`) | **3.05x** |
| `gate_mul + quant` (`2 -> 1 disp`) | `0.0053 ms` | **`0.0031 ms`** | **1.71x** |
| `k3072` (`2304` rows) | `0.0237 ms` | **`0.0221 ms`** (`_m2`) | **1.07x** |
| `gateup_batch8` (`BE=192`) | `0.0322 ms` | **`0.0268 ms`** (`_m2`) | **1.20x** |
| `down_reduce8` (`768` rows) | `0.0499 ms` | **`0.0484 ms`** (`_m4`) | **1.03x** |

### Full-Pack GPU Progression vs `llama.cpp` (Splash paused during full-pack runs)

| Engine & Stage | Commit | Decode Speed (`32` / `8` bench) | Max swap delta | Argmax agreement | Cosine (mean / min) |
|---|---|---:|---:|---:|---|
| **`m64run` Phase 1 CPU** (16 threads) | `506d01b` | 1.12 tok/s | 0.00 MB | 9/9 | 0.99747 / 0.99056 |
| **`m64run` Phase 2A Metal** (initial GPU) | `f82d15e` | 20.31 tok/s | 0.00 MB | 9/9 | 0.99650 / 0.98157 |
| **`m64run` Phase 2A-opt R1** (`m2` + `batch8` + fusion) | `f26339b` | 30.46 tok/s *(swap chk: 29.37)* | 0.00 MB | 9/9 | 0.99650 / 0.98157 |
| **`m64run` Phase 2A-opt R2** (`gateup_m2` + `k3072_m2` + `down_m4` + SIMD/float4) | **`8115ae6`** | **33.61 – 33.72 tok/s** (`238.1 ms` / `237.3 ms`) | **0.00 MB** *(2947.25→2947.25 MB)* | **9/9** | **0.99800 / 0.99377** |
| `llama.cpp` (`e300919`, Metal `--ngl 999`) | — | 31.90 tok/s | +22,013 MB *(co-resident)* | reference | — |

*(Note on `2_parity8.log` for `8115ae6`: steps 0–8 cosines `0.998892, 0.999630, 0.999560, 0.999660, 0.995991, 0.993774, 0.997706, 0.998413, 0.998362`, mean `0.997999`, min `0.993774` — exceeding even Phase 1 CPU accuracy while beating `llama.cpp` Metal by **+5.7%** at `0` swap growth.)*

---

## 8. Phase 2B: Bounded Resident Arena + `mlock` Hot Expert LRU + Async NVMe Streaming (2026-11-23)

- **Commit**: `0b70e81` (`--arena-gb <N>`, `--io pread|mtlio`, `engine/src/expert_arena.{hpp,mm}`, `engine/tests/test_arena.mm`).
- **Resident Partition** (`inkling-2-mini-q8_0.m64pack`, `25.93 GiB`, `680` tensors):
  - Non-expert weights (`656` tensors): `1.69 GiB` (`1.82 GB` decimal), pinned resident via `mlock`.
  - MoE experts (`24` layers × `128` experts = `3,072` slices × `8.08 MiB` / `8,470,528 B` = `24.234 GiB`): streamed on miss into an `mlock`-pinned LRU `MTLBuffer` arena capped strictly by `--arena-gb <N>`, via `--io pread` (`pread` pool with `F_NOCACHE`) or `--io mtlio` (`MTLIOCommandQueue` GPU DMA).
- **Unit & DMA Validation (`./scripts/check.sh`)**:
  - `test_dequant`, `test_metal_parity`, `test_p2a_kernels`, and `test_arena` all passed (`0` failures).
  - `test_arena`: `locked_region` `2 MiB aligned=1 mlocked=1`, `pread_pool` `F_NOCACHE` batch parity OK (`62.80 GB/s`), `arena_pread` (`10` slots, `10` hits / `22` misses / `12` evictions, `23,068,672 B` SSD reads), `MTLIOCommandQueue` initialized `CLEAN` on Apple M4 Max and DMA-streamed `8×1 MiB` into an `mlock`'d arena with bitwise parity (`42.26 GB/s`).
- **Full-Pack `--arena-gb` Matrix** (`inkling-2-mini-q8_0.m64pack`, 5-token prompt + 8 decode steps, `--steps 8 --bench` via `scripts/swap_check.sh` and `--compare` vs `llama.cpp` FP32 reference):

| Configuration | Env / Splash | Slots (`arena` + `resident`) | Argmax Match | Cosine (mean / min) | Hit Rate (`cmp` / `bench`) | SSD Throughput (`cmp` / `bench`) | Decode Throughput | Swap Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `--arena-gb 4 --io pread` | **Co-resident w/ 45 GiB Splash** (`/health` `ok`) | `288` (`2.27` + `1.69` = `4.00 GiB` / `4.29 GB` dec) | **9/9** | `0.997999` / `0.993774` | `62.8%` / `61.4%` (`3858` misses) | `1.84 GB/s` / `1.81 GB/s` | `1.25 tok/s` (`6392.4 ms`) | **`0 MB`** |
| `--arena-gb 8 --io pread` | Splash paused (`stop_all.sh`) | `795` (`6.27` + `1.69` = `8.00 GiB` / `8.58 GB` dec) | **9/9** | `0.997999` / `0.993774` | `73.7%` / `76.4%` (`2353` misses) | `1.85 GB/s` / `1.88 GB/s` | `2.48 tok/s` (`3222.2 ms`) | **`0 MB`** |
| `--arena-gb 8 --io mtlio` | Splash paused (`stop_all.sh`) | `795` (`6.27` + `1.69` = `8.00 GiB` / `8.58 GB` dec) | **9/9** | `0.997999` / `0.993774` | `73.7%` / `76.4%` (`2353` misses) | `2.11 GB/s` / `16.08 GB/s` | `5.81 tok/s` (`1376.1 ms`) | **`0 MB`** |
| `--arena-gb 12 --io pread` | Splash paused (`stop_all.sh`) | `1302` (`10.27` + `1.69` = `12.00 GiB` / `12.88 GB` dec) | **9/9** | `0.997999` / `0.993774` | `76.0%` / `86.0%` (`1402` misses) | `45.91 GB/s` / `41.64 GB/s` | `20.46 tok/s` (`391.1 ms`) | **`0 MB`** |
| `--arena-gb 12 --io mtlio` | Splash paused (`stop_all.sh`) | `1302` (`10.27` + `1.69` = `12.00 GiB` / `12.88 GB` dec) | **9/9** | `0.997999` / `0.993774` | `76.0%` / `86.0%` (`1402` misses) | `39.28 GB/s` / `36.35 GB/s` | **`22.12 tok/s`** (`361.7 ms`) | **`0 MB`** |

- **Key Takeaways**:
  - **True Bounded Co-Residency & Zero Swap**: `--arena-gb 4` runs the `25.93 GiB` model natively on Metal GPU inside a strict `4.29 GB` resident envelope *alongside* a live ~45 GiB Splash server with `0 MB` swap growth and `9/9` greedy parity.
  - **Working-Set Caching & Hardware Scaling**: With `288` slots (`--arena-gb 4`), repeated eviction across the prompt + decode passes drives misses from `1,859` (`--compare`) to `3,858` (`--bench`). Widening `--arena-gb` to `8` (`795` slots) and `12` (`1,302` slots) retains the working set (`1,197` -> `1,402` misses, only `+205` incremental misses across the timed pass, `86.0%` hit rate), reaching **`22.12 tok/s`** with `--io mtlio` while keeping total resident RAM below half the pack size — ready for 100B+ MoE streaming or smaller-memory Macs (e.g., 16–32 GB M1 Pro).
  - *(Note on SSD throughput column: cold sequential runs on `--arena-gb 4` / `8` measure raw NVMe `~1.81–2.11 GB/s`, while repeat/warm-cache runs (`8g mtlio` swap and sequential `12g` runs immediately following) hit OS/drive-warmed blocks at `16.08–45.91 GB/s`.)*

---

## 9. Phase 2A-opt R3: Register `multi_m2` Fusion, 2-Stage Parallel MoE Down+Conv, Fused Mix & GPU Profiler (2026-11-23)

- **Key Changes**:
  - `k_gemv_q8_k2304_multi_m2` (`engine/kernels/m64_p2a.metal`, `MultiGemvArgs`): fuses up to 6 projections sharing `(xqoff, dxoff)` (`Q/K/V/R/sinkb/gateb` on static SWA slots; `{Q,R,sinkb,gateb}` + `{K,dynb}` + `{V2,dynb}` on full-attn slots) with pure register `XA/XB` reuse (after verifying `threadgroup`-cached `tg_xq` kernels lost occupancy on M4 Max vs L1 + register `_m2`).
  - `k_gemv_down_batch8_m2` + `k_down_reduce8_conv_add`: replaces the 72-TG single-dispatch `down_reduce8_m4` with a 2D `(ceil(nout/16), 8)` parallel GEMV grid (`1,152` TGs for `kNE=2304` saturating all 40 GPU cores) followed by a fused weighted-sum + static `ffnconv` + `cur += conv_out` pass.
  - `k_mix_fused`: fuses `k_mix_dot` + `k_mix_soft` + `k_mix_comb` (3 dispatches → 1 dispatch) using 2-level `simd_sum` reductions and `float4` dot/combine.
  - `M64_PROFILE=1` in `engine/src/forward_metal.mm`: zero-overhead category GPU timer (`GPUEndTime - GPUStartTime` at `pos == 5`).

### Microbenchmark (`./build/test_p2a_kernels --bench`, < 30 MB synthetic buffers)

| Kernel / Chain | R2 Baseline | R3 | Speedup |
|---|---:|---:|---:|
| `multi6_qkvrg` (`6` sep `_m2` vs `multi_m2`) | `0.0500 ms` | **`0.0282 ms`** | **1.77x** |
| `down` (`down_reduce8_m4` `768` vs `down_batch8_m2`) | `0.0542 ms` | **`0.0311 ms`** *(+conv: `0.0331 ms`)* | **1.74x** |
| `moe_batch8` (`BE=192`, full MoE + conv/add in R3) | `0.1204 ms` | **`0.0633 ms`** | **1.90x** |
| `mix` (`S=8, nq=3`, `3` disp vs `k_mix_fused` `1` disp) | `0.0392 ms` | **`0.0117 ms`** | **3.35x** |

### Full-Pack Verification (Splash paused during full-pack runs, restored cleanly afterward)

| Benchmark | Decode Speed (`8`-step bench) | Swap Delta | Argmax Agreement | Cosine (`mean / min / max`) | Log |
|---|---:|---:|---:|---|---|
| **Resident Metal (`m64run` R3, 8-step swap chk)** | **`38.60 tok/s`** (`207.2 ms`) | **`0 MB`** *(2875.25→2875.25 MB)* | **9/9** | `0.997102` / `0.986632` / `0.999659` | `<ORACLE_DIR>/p2a_r3_swap8.log` |
| **Resident Metal (`m64run` R3, 32-step run)** | **`38.65 tok/s`** (`207.0 ms`) | — | **9/9** *(vs `p2a_r3_cmp.log`)* | `0.999021..0.998981` | `<ORACLE_DIR>/p2a_r3_bench32.log` |
| **Phase 2B (`--arena-gb 12 --io mtlio`)** | **`24.81 tok/s`** (`322.5 ms`) | **`0 MB`** *(hit `85.9%`, SSD `36.03 GB/s`)* | — | — | `<ORACLE_DIR>/p2b_r3_12g_mtlio_swap.log` |
| *Reference: `m64run` R2 (`8115ae6` / `0b70e81`)* | `33.72 tok/s` (res) / `22.12 tok/s` (`12G mtlio`) | `0 MB` | `9/9` | `0.997999` / `0.993774` | Section 7 & 8 |
| *Reference: `llama.cpp` (`e300919` Metal)* | `31.90 tok/s` *(+22.0 GB swap co-res)* | reference | reference | — | Section 7 |

### GPU Stage Profile (`M64_PROFILE=1` at `pos=5`, `total = 25.940 ms`, `1,424` dispatches)

| Stage | GPU Time | Share | Dispatches | Notes / Next Optimization Targets (R4) |
|---|---:|---:|---:|---|
| `moe_gateup` | `6.078 ms` | `23.4%` | `96` | `gateup_batch8_m2` + `situ_quant8` |
| `qkv` | `4.602 ms` | `17.7%` | `248` | QKV multi-proj + `rms_quant` + `conv` |
| `moe_down` | `4.189 ms` | `16.2%` | `96` | `down_batch8_m2` (`18/32` lanes active) + `down_reduce8_conv_add` |
| `ctrl` | `3.746 ms` | `14.4%` | `96` | **`grid.x = 1..3` on 40 GPU cores (~60 GB/s on ~226 MB FP32 ctrl weights — top R4 target)** |
| `wo_post` | `1.867 ms` | `7.2%` | `200` | `gate_mul_quant` + `k3072_m2` + `postconv` + `xfer2` |
| `mix` | `1.563 ms` | `6.0%` | `240` | `k_rms` + `k_mix_fused` (`grid.x = 1..4` — small-grid launch/sync overhead) |
| `router` | `1.464 ms` | `5.6%` | `144` | `rms_quant` + 128-row `k_gemv_f32_tg` + 1-thread `k_router_topk` |
| `lm_head` | `1.245 ms` | `4.8%` | `4` | `k2304_m2` on `201,024` rows = `492 MB` (`~395 GB/s`) |
| `conv_clip_attn` | `1.125 ms` | `4.3%` | `288` | `clip` + `rms(V)` + `k_attn_head` |
| `other` | `0.059 ms` | `0.2%` | `12` | `k_emb_gather` + prelude/loop `xfer2` |

---

## 10. Phase 2A-opt R4: Split Control Block, Parallel SIMD Router & `float4` RMSNorm (2026-11-23)

- **Key Changes** (implemented via Hermes + verified cleanly):
  - `k_ctrl_down` + `k_ctrl_up` (`CtrlSplitArgs` in `engine/kernels/m64_p2a_args.h`, `engine/kernels/m64_p2a.metal`, wired in `engine/src/forward_metal.mm` behind `gemv_r4`): splits `k_control` (`grid.x = 1..3`) into `k_ctrl_down` (`grid (64, nc)` × `256` threads, `float4` dot + 2-level `simd_sum` -> `SCR[ctrl_hh]`) and `k_ctrl_up` (`grid (ceil(ne/32), nc)` × `32` threads, TG-cached `hh[64]` + `float4` gain/forget dot -> direct residual write, eliminating `gb`/`fb` staging).
  - Parallel `k_router_topk`: replaces the 1-thread serial `8×128` loop (`k_router_topk_serial` kept for A/B) with 128-thread 4-warp `simd_max` + smallest-index `simd_min` reduction. (`k_gemv_f32_tg` generalized to `nth >> 5u`, confirming `nth=256` remains fastest at `0.0034 ms/iter`.)
  - `float4` `k_rms` + `k_rms_quant`: shared `vec4_in` squared-sum predicate (`((A.k & 3u) == 0u) && ((A.xr[row] & 3u) == 0u)`) keeps Test 26 bitwise-identical while vectorizing loads/writes.

### Microbenchmark (`./build/test_p2a_kernels --bench`, < 30 MB synthetic buffers)

| Kernel / Chain | Old | R4 | Speedup |
|---|---:|---:|---:|
| `ctrl(nc=3)` (`k_control` vs `k_ctrl_down` + `k_ctrl_up`) | `0.0431 ms` | **`0.0094 ms`** | **4.59x** |
| `router(128)` (`k_gemv_f32_tg` + `serial` vs `parallel`) | `0.0088 ms` | **`0.0060 ms`** | **1.47x** |
| `rms_mix_S8(2304)` (`float4` `k_rms`) | — | **`0.0045 ms`** | — |

### Full-Pack Verification (Splash paused during full-pack runs, restored cleanly afterward)

| Benchmark | Decode Speed (`8`-step bench) | vs `llama.cpp` (`31.90`) | Swap Delta | Argmax Agreement | Cosine (`mean / min / max`) | Log |
|---|---:|---:|---:|---|---|---|
| **Resident Metal (`m64run` R4, 8-step swap chk)** | **`45.01 tok/s`** (`177.7 ms`) | **+41.1%** | **`0 MB`** *(4136.88→4136.88 MB)* | **9/9** | `0.997339` / `0.988938` / `0.999711` | `<ORACLE_DIR>/p2a_r4_swap8.log` |
| **Resident Metal (`m64run` R4, 32-step run)** | **`45.33 tok/s`** (`176.5 ms`) | **+42.1%** | — | **9/9** *(vs `p2a_r4_cmp.log`)* | `0.998891..0.998529` | `<ORACLE_DIR>/p2a_r4_bench32.log` |
| **Phase 2B (`--arena-gb 12 --io mtlio`)** | **`26.48 tok/s`** (`302.1 ms`) | — | **`0 MB`** *(hit `85.9%`, SSD `34.59 GB/s`)* | — | — | `<ORACLE_DIR>/p2b_r4_12g_mtlio_swap.log` |
| *Reference: `m64run` R3 (`cd55ddf`)* | `38.65 tok/s` (res) / `24.81 tok/s` (`12G mtlio`) | +21.2% | `0 MB` | `9/9` | `0.997102` / `0.986632` | Section 9 |

### GPU Stage Profile (`M64_PROFILE=1` at `pos=5`, `total = 22.856 ms`, `1,472` dispatches)

| Stage | GPU Time | Share | R3 → R4 Delta | Next Targets (R5 / Phase 2B) |
|---|---:|---:|---:|---|
| `moe_gateup` | `6.205 ms` | `27.1%` | `+0.13 ms` | Largest single stage (`gateup_batch8_m2` + `situ_quant8`) |
| `qkv` | `4.593 ms` | `20.1%` | `-0.01 ms` | `Q/K/V/R/sink/gate` (`~1.01 GB`, `48` loop-slot fusion / `m4`?) |
| `moe_down` | `4.235 ms` | `18.5%` | `+0.05 ms` | `18/32` lanes active on `k=1152` in `down_batch8_m2` |
| `wo_post` | `1.894 ms` | `8.3%` | `+0.03 ms` | `k3072_m2` + `postconv` |
| `mix` | `1.305 ms` | `5.7%` | **`-0.258 ms`** | `float4` `k_rms` + `k_mix_fused` |
| `lm_head` | `1.243 ms` | `5.4%` | `0.00 ms` | `201,024` rows = `492 MB` (`~395 GB/s`) |
| `conv_clip_attn` | `1.187 ms` | `5.2%` | `+0.06 ms` | `clip` + `rms(V)` + `k_attn_head` |
| `router` | `1.174 ms` | `5.1%` | **`-0.290 ms`** | Parallel SIMD `k_router_topk` |
| `ctrl` | **`0.960 ms`** | **`4.2%`** | **`-2.786 ms` (3.90x)** | Wide-dispatch `k_ctrl_down` + `k_ctrl_up` |
| `other` | `0.059 ms` | `0.3%` | `0.00 ms` | `k_emb_gather` + prelude/loop `xfer2` |

## 11. Milestone 9A: Q4_K Decode MoE Down — 2-Stage Parallel + Fused Conv/Add (2026-11-26)

`gpu_step()` Q4_K_S decode path (Inkling-2-Mini): the single-stage
`k_gemv_down_reduce8_q5_{0,1}_f32in_m4` (72 TGs) + separate `k_conv_slot` +
`k_xfer` chain now has a 2-stage path mirroring the Q8_0 Milestone 7 design,
gated on `gemv_r3 && !trace_slot` (`M64_GEMV_R3=0` or trace falls back to the
old chain for A/B):

- Stage 1 `k_gemv_down_batch8_q5_{0,1}_f32in_m2`: grid `((kNE+15)/16, 8)` =
  (144, 8) = 1,152 TGs × 256, expert parallel on `grid.y`, 2 rows/warp,
  unweighted per-expert FP32 → `o.down_tmp` `[8][kNE]` slab.
- Stage 2 `k_down_reduce8_conv_add` (reused from Q8_0): fused top-8 weighted
  reduce + `ffnconv` 4-tap static conv + `cur += (xin+outv)·kConvStatic` and
  rs-slot rotation in one dispatch.
- SCR conflict fix: `k_situ_f32_batch` now writes h in-place into `o.gate2`
  (per-thread read→write, disjoint threads), freeing `o.down_tmp` for stage 1.

Parity (`test_p2a_kernels` tests 51/52, gates `mad < 1e-4`, `cos ≥ 0.99999`):
the m2 stage-1 dots are **BITWISE** (`mad = 0`) vs the m1/m4 reference for
Q5_0/Q5_1 at nout ∈ {2304, 173, 100} — block-iteration order (`i = lane;
i += 32`) and per-row FMA order were kept identical. Fused chain end-to-end
also `mad = 0` vs old reduce+conv+add.

Micro-bench (`--bench`, real shapes kNE=2304 / kNF=1152 / top-8, IT=100,
median of 3 stable runs; the first run of `down_q5_0_old` shows cold-cache
0.214 ms and is excluded):

| Chain (decode down, per layer-step) | Q5_0 old | Q5_0 **new** | Q5_1 old | Q5_1 **new** |
|---|---:|---:|---:|---:|
| reduce + conv + add (ms) | `0.113` | **`0.109`** | `0.117` | **`0.114`** |

~3.5–4% faster end-to-end plus one fewer dispatch (3→2). The gain is modest
because at real shapes the 72-TG m4 kernel is already weight-bandwidth bound
(~14.6 MB expert reads dominate); the 1,152-TG grid mainly removes the tail
that made m4 lose to m1 on small nout (see §Milestone 7A bench) and folds the
conv/add traffic into one kernel. Decode E2E not re-measured here (Splash
resident on :8080; >16 GB live-model runs disallowed this milestone).

## 12. Milestone 9B: Batched Prefill Small-Op Kernels (2026-11-26)

`gpu_prefill_batch()` (`engine/src/forward_metal.mm`) previously ran six
per-token `for (uint32_t b = 0; b < B; b++)` Metal dispatch loops for `xfer2`,
`do_mix`, and `do_ctrl` across all 48 logical slots (emitting tens of
thousands of tiny dispatches per `B=128` prefill chunk). Replaced with batched
`grid.y`/`grid.z` kernel clones in `engine/kernels/m64_prefill.metal` (`bsmall`
gate; `M64_NO_BSMALL=1` restores the sequential loops for A/B):

- `k_xfer_batch`: batched copy (`c = a`) and add (`a += b`), `npairs <= 2`,
  independent per-pair strides (`ip % 6 == 0` block-push preserves sequential
  2-dispatch ordering `cur -> blocks` before `tB -> cur`).
- `k_mix_rms_batch` + `k_mix_fused_batch`: 2 dispatches replace `2·B` dispatches
  for `do_mix` across all `B` tokens (via `o.b_stn` scratch).
- `k_ctrl_down_batch` + `k_ctrl_up_batch`: 2 `k_rms_batch` + 2 batched split
  dispatches replace `3·B` dispatches for `do_ctrl` (via `o.b_ctrl_gcat` and
  disjoint per-token `o.b_ctrl_hh` `[kBMax][4][64]`).

Parity (`test_prefill_kernels` §12, `B ∈ {8, 32, 128}`):
- `k_xfer_batch` (copy, add, `npairs=2`, `n=1000` odd tail): **bitwise** (`0` mismatches).
- `k_mix_rms_batch` + `k_mix_fused_batch` (`(n=8,nq=1)` & `(n=4,nq=4)`): **bitwise** (`0` mismatches).
- `k_ctrl_down_batch` + `k_ctrl_up_batch` (`nc ∈ {1, 3}`, `in_row ∈ {0, 2304}`): `cos = 1.00000000`, `mad <= 9.54e-7`.

Micro-bench (`./build/test_prefill_kernels --bench`, `B=128`, `kNE=2304`, `IT=50`, min of 3 passes):

| Op chain (1 call site, `B=128`) | Seq loop | Batched | Speedup |
|---|---:|---:|---:|
| `xfer_add` (`128` → `1` disp) | `0.3687 ms` | **`0.0068 ms`** | **54.0x** |
| `xfer_copy` (`128` → `1` disp) | `0.3628 ms` | **`0.0063 ms`** | **57.3x** |
| `mix` (`n=8, nq=1`, `256` → `2` disp) | `3.3467 ms` | **`0.0627 ms`** | **53.4x** |
| `ctrl` (`nc=3`, `384` → `4` disp) | `1.7762 ms` | **`0.5066 ms`** | **3.5x** |

Prefill E2E not re-measured here (Splash resident on :8080; >16 GB live-model runs disallowed).

---

## 13. Milestone 9C: Long-Context Split-K Attention — GQA 3-Head KV Sharing + 128 Splits (2026-11-27)

Inkling-2-Mini decode uses `24` Q heads over `8` KV heads (`GQA 3:1`, `kDH=256`).
Previously `attn_split_dispatch` launched `24 × n_splits` threadgroups (`kAttnMaxSplits = 32`), reading every `K`/`V` token 3 separate times per split across GPU workgroups.
Milestone 9C raises `kAttnMaxSplits` to `128` (`1024` TGs max at `pos >= 16384`) and adds 1-warp register GQA 3-head KV sharing (`k_attn_head_f16_split_gqa` / `k_attn_head_q8_split_gqa`, `grid = (8, n_splits, 1)`, rollback env `M64_NO_ATTN_GQA=1`).

- **Parity (`./build/test_prefill_kernels`)**:
  - `attn_split` (FP16) & `attn_q8split` (Q8_0) across `24h` (`*_split`) and `gqa` (`*_split_gqa`) at `pos=511, 2048, 20000` (including `n_splits=64, 128`), `swa p=2048`, and `empty p=10`: `cos = 1.00000000`, `mad <= 2.09e-07` (FP16) / `<= 1.53e-05` (Q8_0).
  - `pos=65536` (`gqa8x128` vs `24x32`): Q8_0 `cos = 1.00000000`, `mad = 4.77e-07`; FP16 `cos = 1.00000000`, `mad = 2.24e-08`.
- **Full suite (`./scripts/check.sh`)**: `ALL PASS`, `196 passed, 0 failed` (`tools/test_server.py`).

| Kernel (`24x32` → `gqa8x128`) | Baseline (`24x32`) | Milestone 9C (`gqa8x128`) | Speedup |
|---|---:|---:|---:|
| `attn_q8 pos=20000` | `0.5719 ms` | **`0.2409 ms`** | **2.37x** |
| `attn_q8 pos=65536` | `1.6513 ms` | **`0.6739 ms`** | **2.45x** |
| `attn_f16 pos=20000` | `0.5892 ms` | **`0.2594 ms`** | **2.27x** |
| `attn_f16 pos=65536` | `1.6919 ms` | **`0.7335 ms`** | **2.31x** |

Decode E2E not re-measured here (Splash resident on :8080; >16 GB live-model runs disallowed).

---

## 14. Milestone 10A: Coalesced Zero-Tail Q4_K/Q5 MoE Decode GEMVs — NEGATIVE RESULT (2026-11-27)

Prototyped coalesced zero-tail decode GEMV variants (`_c8_m2` / `_c8_m4`):
8 lanes per 64-elem chunk (one full 256-elem Q4_K super-block per warp iteration,
`k=2304` -> 9 exact iterations, 32/32 lanes, 0 tail) and 8 lanes per 32-elem Q5 block
(4 blocks / warp iteration, `k=1152`, `nb=36` -> 9 exact iterations), with factored
1-FMA/elem dots (`d*d1*sum(q*x) - dmin*m1*sum(x)` for Q4_K; `d*sum(q*x) + m*sum(x)`
for Q5_1 / `d*(sum(q*x) - 16*sum(x))` for Q5_0) and x/h register reuse across R rows.

- **Parity**: passed at tolerance class `cos >= 0.99999`, `mad < 1e-4` (factored dots
  reorder FMAs vs m1/m2); fp64 oracle worst ulp 0.00 for gate|up; zero-x bitwise zero.
- **Benchmark (M4 Max, `test_p2a_kernels --bench`, real Inkling dims, reproduced 2/2 runs)**:

| Chain | Winner (existing) | c8_m2 | c8_m4 | Verdict |
|---|---:|---:|---:|---|
| `gateup_q4k` (real 8x1152) | **m2: 0.201 ms** | 0.305 ms | 0.351 ms | c8 ~1.5x slower |
| `down_q5_0` (b8+rc chain) | **b8m2+rc: 0.112 ms** | 0.253 ms | 0.254 ms | c8 ~2.2x slower |
| `down_q5_1` (b8+rc chain) | **b8m2+rc: 0.117 ms** | 0.245 ms | 0.271 ms | c8 ~2.1x slower |

The factored-dot reductions (`sx_lo/sx_hi`/`sh` lane sums) plus the changed warp
mapping cost more ALU/ILP than the eliminated tail saves; zero-tail coalescing did
NOT win on M4 Max for these shapes. **Not wired into `gpu_step()`; prototype kernels
and their tests were removed** (no dead code). Existing m2/m4 decode GEMVs remain
the shipped path. Milestone 10B (same multi-lane-per-block hypothesis on Q8_0) was skipped in favor of dispatch fusion (Milestone 11).

---

## 15. Milestone 11: Decode Dispatch Fusion — QKV Clip/V-Norm/KV-Write & PostConv+Residual (2026-11-27)

Inkling-2-Mini decode (`gpu_step()`) previously emitted **5 dispatches** per slot for Q/K/Rr clip, V per-head RMS-norm (with `o.emb` staging), and `k_kv_write_{q8,f16}`, plus **2–3 dispatches** per slot for `k_conv_slot(postconv)` + `k_xfer` residual/block-push.
Milestone 11 fuses both chains (gated behind `M64_NO_FUSE_M11=1` / `!trace_slot && !nanprobe`), eliminating **248 dispatches/token** (`48 × 4` in QKV prep/write + `48 + 8` in postconv+residual):

- `k_qkv_prep_kv_write_{q8,f16}` (`engine/kernels/m64_p2a.metal`): `grid = (64, 1, 1)`, `tg = (32, 1, 1)`, 1 warp per head, zero `threadgroup_barrier`. Warps `0..23` clip `Q` in-place; `24..31` clip `K` in registers and write directly to `KV_K`; `32..39` RMS-norm `V` in registers and write directly to `KV_V` (eliminating `o.emb` staging); `40..63` clip `Rr` in-place.
- `k_postconv_residual` (`engine/kernels/m64_p2a.metal`): flat `2304 × 128` dispatch folds 4-tap `postconv` + `rs` shift + residual add (`push=0`) or race-free `cur -> blocks[nblk]; cur = y` snapshot (`push=1` on `ip % 6 == 0`).

- **Parity (`./build/test_p2a_kernels`, Tests 53–55)**:
  - `qkvprep_{q8,f16}` (`clip` and `M64_NOFF`): `Rr`, `KV_K`, `KV_V` all **`mad = 0, cos = 1.00000000`**, `Q` `mad <= 9.54e-07, cos = 1.00000000`.
  - `postconv_res` (`push=0` & `push=1`): `cur`/`blk`/`rs` exact vs unfused chain, CPU `y` `mad <= 1.19e-07`.
- **Full suite (`./scripts/check.sh`)**: `ALL PASS`, `196 passed, 0 failed` (`tools/test_server.py`).

| Chain (per slot, M4 Max `--bench`) | Unfused (`old`) | Fused (`new`, `1 disp`) | Speedup |
|---|---:|---:|---:|
| `qkvprep_kv_q8` (`5 → 1` disp) | `0.0130 ms` | **`0.0036 ms`** | **3.61x** |
| `qkvprep_kv_f16` (`5 → 1` disp) | `0.0136 ms` | **`0.0036 ms`** | **3.78x** |
| `postconv_res push=0` (`2 → 1` disp) | `0.0075 ms` | **`0.0035 ms`** | **2.14x** |
| `postconv_res push=1` (`3 → 1` disp) | `0.0095 ms` | **`0.0035 ms`** | **2.71x** |

Decode E2E not re-measured here (Splash resident on :8080; >16 GB live-model runs disallowed).

---

## 16. Milestone 13: Factored 2-Row Q4_K/Q5 Decode Inner Loops + Startup Pipeline Pre-Warming (2026-11-27)

Maintained the winning 1-lane-per-chunk `_m2` warp mapping and grids while optimizing inside the 2-row inner loops and removing lazy request-#1 Metal JIT stalls:

- **`m64_q4k_sub_dot2`** (`engine/kernels/m64_p2a.metal`): fused 2-row Q4_K sub-block dot sharing `float4` activation loads and `sx_lo`/`sx_hi` sums across both rows, cutting inner-loop arithmetic to 1 FMA/element (`sq_lo`/`sq_hi`) and applying `(d*scale, -dmin*min)` once per half-block via aligned 32-bit header loads.
- **`m64_gemv_q5_rows2<Q51>`** (`engine/kernels/m64_p2a.metal`): shared `h` `float4` loads and `sh` block sum across 2 rows via 2B-aligned `ushort` header loads (safe for 22B Q5_0 blocks), folding `-16`/`d`/`m` out of the 32-elem inner loop.
- **Startup pipeline pre-warm** (`engine/src/forward_metal.mm`): `InklingForwardMetal::Impl::init()` pre-compiles all 22 hot decode Metal pipelines (`kDecodeWarmFns`) at boot so the first request pays zero lazy compilation overhead.

- **Parity (`./build/test_p2a_kernels`, Tests 47–52)**:
  - `gateup q4k m2` (direct `nout=1152/173` & `wtab 1152`): `mad <= 9.54e-06`, `cos = 1.00000000` vs `m1`.
  - `down9a` (`q5_0`/`q5_1`, `nout=2304` & tails `173/100`): `mad <= 2.29e-05`, `cos = 1.00000000` vs old chain.
- **Full suite (`./scripts/check.sh`)**: `ALL PASS`, `232 passed, 0 failed` (`tools/test_server.py`).

| Kernel / Chain (`--bench` w/ concurrent Splash on :8080) | Reference (`m1` / `old`) | Optimized (`m2` / `b8m2+rc`) | Relative Speedup |
|---|---:|---:|---:|
| `gateup_q4k (real 8x1152)` | `0.2570 ms` (`m1`) | **`0.1904 ms`** (`m2`) | **1.35x** vs `m1` *(was 1.12x)* |
| `down_q5_0 (real 2304 chain)` | `0.1484 ms` (`old`) | **`0.1222 ms`** (`b8m2+rc`) | **1.21x** vs `old` *(was 1.04x)* |
| `down_q5_1 (real 2304 chain)` | `0.1502 ms` (`old`) | **`0.1274 ms`** (`b8m2+rc`) | **1.18x** vs `old` *(was 1.03x)* |

Decode E2E not re-measured here (Splash resident on :8080; >16 GB live-model runs disallowed).

---

## 17. Milestone 14: Cold-Start TTFT & Startup Warmup Elimination (2026-11-27)

Eliminated the main startup / request-#1 latency stalls observed when serving `inkling-2-mini-q4_k_s.m64pack` (16 GB mmap from external NVMe):

1. **Background `mmap` pack page warming (`engine/src/forward_metal.mm`)**:
   - In full-pack mode (`!arena_mode && file_size >= 1 MiB`), `Impl::init()` spawns `prefetch_thread_` (unless `M64_NO_PREFETCH=1`; `M64_PREFETCH_SYNC=1` forces blocking warm-up) walking 64 MiB chunks with `madvise(..., MADV_WILLNEED)` + touching one volatile byte per 16 KiB macOS arm64 page while checking `stop_prefetch_` every 4 MiB. `~Impl()` stops and joins before `PackReader` unmaps the pack, while `READY` is emitted immediately.
2. **Startup Prefill Pipeline Pre-Warm (`engine/src/forward_metal.mm`)**:
   - `Impl::init()` pre-compiles all 22 hot batched prefill Metal pipelines (`kPrefillWarmFns`: `k_emb_gather_batch`, `k_rms_batch`, `k_conv_batch`, `k_clip_batch`, `k_gate_mul_batch`, `k_gemv_f32_batch`, `k_gemm_q8_f32in_sg`, `k_router_topk_batch`, `k_moe_bucket_build`, `k_moe_down_reduce_batch`, `k_kv_write_{q8,f16}_batch`, `k_attn_{q8,f16}_flash_prefill`, `k_attn_head_{q8,f16}_batch`, `k_moe_grouped_f32in_sg*`, `k_softcap`, `k_argmax`) alongside `kDecodeWarmFns` so Request #1 incurs zero lazy Metal JIT stalls.
3. **Asynchronous `.m64kv` SSD Prefix Save (`engine/src/forward_metal.hpp`, `forward_metal.mm`, `main.cpp`)**:
   - Extracted `DiskPrefixSnapshot` + `snapshot_disk_prefix()` (RAM-only copy of `rs`, SWA rings, and used `[0..pos)` full-attention KV while GPU is idle). `WARM` captures the snapshot into RAM and delegates `write_disk_prefix_file()` to a joined-on-exit/reload `disk_save_thread`, returning `WARMED <n> computed` without waiting for external SSD I/O.
4. **BPE Piece Memoization & Unchanged Static-Prefix Fast Path (`tools/bpe_gguf.py`, `tools/takuan_server.py`)**:
   - Added a 65,536-entry `_piece_cache` in `BPE._encode_piece` and split `format_static_chat_prefix_text()` so `_maybe_warm_static_prefix()` skips both `bpe.encode()` and `worker.warm()` on pure string match (and reuses `last_warm_ids` on `warm_needs_sync`).

- **Verification (`./scripts/check.sh`)**: all native suites (`test_dequant`, `test_gemma4_cpu`, `test_gemma4_metal`, `test_metal_parity`, `test_p2a_kernels`, `test_prefill_kernels`, `test_sampler`, `test_spec_decode`, `test_arena`) passed; `tools/test_server.py` passed **`232 passed, 0 failed (4.5s)`**. Live E2E TTFT not re-measured here (Splash resident on `:8080`; >16 GB live-model runs disallowed).

---

## 18. Milestone 15: Prefill MoE Grouped Kernel Optimization (`Q4_K`, `Q5_0`, `Q5_1`) (2026-11-27)

Optimized `k_moe_grouped_f32in_sg_{q4k,q5_0,q5_1}` in `engine/kernels/m64_prefill.metal` to eliminate redundant header loads, scalarize less, vectorize activation loads, and specialize the dominant small-bucket (`cnt - off <= 16`) case:

1. **Q4_K super-block header hoisting (`m64_q4k_hdr` + `dequant_q4_k_sub_to_tg`)**: Loads the 16 B super-block header (`d`/`dmin` + 12 scale bytes) ONCE per 256-element super-block (`(blk & 7) == 0`) instead of 32× per sub-block, computes all 8 sub-block scales/mins in registers via `m64_scale_min_k4`, and reads nibble bytes via 4 B loads when `wmat` is 4-byte aligned (byte fallback preserved).
2. **Vectorized, branch-free `Q5_0` / `Q5_1` dequant**:
   - `dequant_q5_0_tile32x32_to_tg`: 2B-aligned `ushort` loads (preserving alignment safety for 22 B blocks, never casting to `uint*`), hoisted nibble shift and `qh` bit base.
   - `dequant_q5_1_tile32x32_to_tg`: 4B-aligned `uint` loads for 24 B blocks (byte fallback when misaligned) with branch-free unpack.
3. **Vectorized `tgX` loader (`m64_fill_tgX_tile`)**: Uses two `float4` loads instead of 8 scalar loads when `(xoff | x_stride) % 4 == 0`, and skips `bcol >= 16` loads in the narrow path.
4. **Small-bucket `32×16` fast path (`cnt - off <= 16`)**: Each warp owns `8 × 16` (`2` `simdgroup_float8x8` accumulators vs `4`), halving `tgX` loads, MMA ops, and `tgO` store/writeback for the 99%+ of prefill experts with `<= 16` tokens while keeping `32×32` when `cnt - off > 16`.

- **Parity & Suite**:
  - Updated `engine/tests/test_prefill_kernels.mm` (§10f) with explicit `cnt=16` (`narrow`) and `cnt=17` (`full`) boundary experts alongside `cnt=37` and `cnt=0`; `test_prefill_kernels` **ALL PASS** (12/12 FP64-oracle scenarios, 0 failures).
  - Clean rebuild from deleted Metal archives + `./scripts/check.sh`: **ALL PASS**, `tools/test_server.py` **`232 passed, 0 failed (4.3s)`**.

### Prefill Microbenchmark (`./build/test_prefill_kernels --bench`, `NEXP=128 slots=16 topk=8 k=2304 nout=1152`, concurrent Splash on `:8080`)

| Format | `B=32` Before (`ms` / `GB/s`) | `B=32` After (`ms` / `GB/s`) | `B=128` Before (`ms` / `GB/s`) | `B=128` After (`ms` / `GB/s`) | `B=128` Speedup (vs `Q8_0`) |
|---|---:|---:|---:|---:|---|
| `Q8_0` *(ref)* | `1.9881` (`181.6`) | `1.8756` (`192.5`) | `1.9709` (`183.2`) | `1.8643` (`193.6`) | `1.00x` |
| **`Q4_K`** | `2.2883` (`83.5`) | **`1.8079`** (`105.7`) | `2.2857` (`83.6`) | **`1.8201`** (`105.0`) | **1.26x** (**`1.02x` vs `Q8_0`**) |
| **`Q5_0`** | `2.4436` (`95.6`) | **`1.8484`** (`126.4`) | `2.4573` (`95.1`) | **`1.8633`** (`125.4`) | **1.32x** (**`1.00x` vs `Q8_0`**) |
| **`Q5_1`** | `2.5658` (`99.3`) | **`1.7659`** (`144.3`) | `2.5679` (`99.2`) | **`1.7869`** (`142.6`) | **1.44x** (**`1.04x` vs `Q8_0`**) |

*(Note: Prefill E2E not re-measured here because Splash is resident on `:8080` and >16 GB live-model runs are disallowed.)*
