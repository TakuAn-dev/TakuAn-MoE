[日本語版 README (README.ja.md)](./README.ja.md)

# TakuAn-MoE

Zero-dependency C++20 / Objective-C++ + Metal 4 (MSL) local inference engine
and OpenAI-compatible backend server for running **massive MoE LLMs on Apple
Silicon** — models whose total weights far exceed unified memory. Built for
the M4 Max (16 CPU / 40 GPU cores, 64 GB) with a plain POSIX `Makefile`,
Apple `clang++`, Python 3 stdlib HTTP/BPE tooling, and no Homebrew, CMake, or
external server dependencies. MSL kernels are compiled at load time via
`newLibraryWithSource:` and cached with `MTLBinaryArchive`.

Repository: https://github.com/TakuAn-dev/TakuAn-MoE

## Highlights

- **Apple Silicon (M4 Max, 64 GiB)** — verified on M4 Max 64 GiB.
- **Zero external dependencies** — C++20 / Objective-C++ / Metal + Python 3
  stdlib only.
- **Inkling-2-Mini-Q8_0** (25.93 GiB / 680 tensors / 48 logical slots + MoE
  24x128 top-8) — full resident (`--arena-gb 0`) and bounded-memory NVMe
  streaming (`--arena-gb` + `pread`/`mtlio`).
- **Inkling-2-Mini-Q4_K_S** (`15.71 GiB` `.m64pack`) — `Q4_K` / `Q5_0` /
  `Q5_1` pack converter, dequantization, and Metal MoE kernels
  (`k_gemv_gateup_batch8_q4k_f32in`, `k_gemv_down_reduce8_q5_{0,1}_f32in`,
  `k_moe_grouped_f32in_sg_{q4k,q5_0,q5_1}`), with `F32` dense support
  (`attn_sink_proj`, `attn_{k,v}_conv_proj`) and per-layer `ExpertArena`
  stride.
- **Up to 128K context** (`--ctx 131072`).
- **Milestone 3C/3D optimizations** (measured):
  - Tensor-Core Grouped MoE (`B=128`) + FP32 SiTU — batch prefill
    **222–246 tok/s**.
  - Split-K FlashDecoding — long-context decode **24.1x** (kernel-only,
    `pos=20000`).
  - SSD persistent prefix cache (`.m64kv`) — **17.5k tokens** restored in
    ~**0.1 s**.
  - OpenAI-compatible HTTP/SSE server — tool calling, continuous `: keepalive`
    heartbeats; verified with Hermes Agent (**45k+ tokens**, multi-turn
    continuous tool execution).
- **Milestone 4A/4B/4C optimizations** (measured):
  - Tiled FlashAttention prefill (`k_attn_f16_flash_prefill`,
    `engine/kernels/m64_prefill.metal`) for `B >= 8` on both full (`is_swa=0`)
    and SWA (`is_swa=1`) attention layers — up to **7.63x** faster attention
    prefill @ `p=20000` (`B=128`; `5.93 ms` vs `45.25 ms/iter`), **3.44x** @
    `B=32`, and **4.93x** @ `B=128` on SWA layers.
  - Full-pack `--prefix-cache` auto-default raised from `2` to **4**
    checkpoints (`--prefix-cache` still overrides).
  - Server streaming & disconnect stability — live `delta.reasoning_content`
    SSE streaming during the thinking state (no client idle / stale-stream
    timeouts during extended reasoning), and clean `SIGUSR1` in-flight `GEN`
    cancellation (`DONE ... cancelled`) with `stdout` pipe draining on client
    disconnect/stop (no worker pipe deadlock).

## Verified benchmarks (Apple M4 Max, 64 GB)

Model: `Inkling-2-Mini-Q8_0` (`arch=inkling2_loop`, 24 physical layers / 48
loop slots, 128 experts, top-8), pack size **25.93 GiB**, 680 tensors.
Reference oracle: `llama.cpp` (Metal). Full data in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### Resident Metal (Phase 2A-opt R4)

| Metric | Result |
| :--- | :--- |
| Decode throughput (32-step) | **45.33 tok/s** — **+42.1%** vs `llama.cpp` Metal (`31.90 tok/s`) |
| Greedy argmax agreement vs oracle | **9/9** |
| Swap growth | **0 MB** |

### Q4_K_S quantization (`Inkling-2-Mini-Q4_K_S`, 15.71 GiB pack)

| Metric | Result |
| :--- | :--- |
| Decode throughput, full resident Metal | **50.17 tok/s** — **+14.5%** vs `Q8_0` (`43.80 tok/s`) |
| `--arena-gb 4` (`ctx=256`) | **23.29 tok/s** at **~4.29 GB** resident |

### Bounded streaming (`--arena-gb`)

| Configuration | Resident | Decode | Swap growth | Notes |
| :--- | ---: | ---: | ---: | :--- |
| `--arena-gb 4 --io pread` | **4.29 GB** | 1.25 tok/s | **0 MB** | Co-resident with a live ~45 GiB Splash server, 9/9 argmax |
| `--arena-gb 8 --io mtlio` | 8.58 GB | **5.81 tok/s** | **0 MB** | 795 expert slots, 76.4% hit rate |
| `--arena-gb 12 --io mtlio` | 12.88 GB | **26.48 tok/s** | **0 MB** | 1,302 slots, 85.9% hit, 34.59 GB/s SSD, 9/9 argmax |

## Build

```sh
make
```

## Self-test

```sh
./scripts/check.sh
```

`make test` (invoked by `scripts/check.sh`) generates a tiny synthetic pack +
golden file in `build/fixture/` with `tools/make_test_fixture.py` (Python 3
stdlib only) and runs `test_dequant`, `test_metal_parity`, `test_p2a_kernels`,
`test_prefill_kernels`, `test_arena`, `test_prefix_disk` and the server test
suite.

## Server

```sh
python3 tools/takuan_server.py --pack <MODEL_DIR>/inkling-2-mini-q8_0.m64pack \
  --gguf <MODEL_DIR>/Inkling-2-Mini-Q8_0.gguf \
  --port 8080 --ctx 131072 --cache-dir <CACHE_DIR>
```

Bounded-RAM streaming server:

```sh
python3 tools/takuan_server.py --pack <MODEL_DIR>/inkling-2-mini-q8_0.m64pack \
  --gguf <MODEL_DIR>/Inkling-2-Mini-Q8_0.gguf \
  --port 8081 --ctx 4096 --arena-gb 12 --io mtlio
```

Connect any OpenAI SDK client with:
- **Base URL:** `http://127.0.0.1:8080/v1`
- **Model:** `takuan-inkling-2-mini`
- **API Key:** any non-empty string

## Layout

```
engine/src      C++20 core + ObjC++ Metal host
engine/kernels  MSL kernels (compiled at load, cached via MTLBinaryArchive)
engine/tests    parity/unit tests
tools/          m64pack.py, bpe_gguf.py, takuan_server.py, test_server.py,
                make_test_fixture.py
scripts/        check.sh, swap_check.sh, gpu_validation_suite.sh
docs/           DESIGN.md, BENCHMARKS.md
```

## License

MIT — see [LICENSE](LICENSE).

## References & Acknowledgments

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — numerical verification oracle.
- **strata** / **[mlx-od-moe](https://github.com/kqb/mlx-od-moe)** / **[baseRT](https://github.com/basecompute/baseRT)** / **swiftlm** — pioneering insights on on-demand MoE and Apple Silicon inference.
- **[FlashDecoding](https://arxiv.org/abs/2307.09179)** — basis of the Split-K Attention technique.
- **Hermes Agent** — for assistance with development, testing, and the release process.
