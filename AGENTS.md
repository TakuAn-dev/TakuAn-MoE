# Project Rules: Mac64-MoE Engine

## Target Hardware
- Apple M4 Max (16 CPU cores, 40 GPU cores, 546 GB/s unified memory bandwidth, Metal 4)
- 64 GB Unified Memory (Target engine resident budget: <= 48 GB; zero/minimal macOS swap)
- macOS 27.2.1 / Internal NVMe SSD

## Mission
Build a native Apple Silicon inference engine capable of running massive MoE LLMs (100B+ total parameters, exceeding 64 GB RAM) via:
1. Custom expert-aligned binary pack format
2. Strict Resident Hot Tier (Attention, Router, Shared Experts, Embeddings, KV Cache, Hot Expert Arena)
3. On-demand Cold Tier (Async NVMe SSD streaming via `MTLIOCommandQueue` / zero-swap I/O)
4. Hand-tuned Metal 4 compute kernels (fused dequantization + GEMV/GEMM, Residency Sets)

## Tech Stack
- Core runtime & Metal host: C++20 / Objective-C++ (`clang++ -std=c++20 -fobjc-arc`)
- GPU Kernels: Metal Shading Language (MSL / Metal 4), compiled at load time via `newLibraryWithSource:options:error:` and cached with `MTLBinaryArchive` (full Xcode / `xcrun metal` is NOT installed on this machine)
- Build system: POSIX `Makefile` + `/usr/bin/make` + `/usr/bin/clang++` (no Homebrew / CMake / Ninja required)
- Validation & Conversion scripts only: Python (MLX / PyTorch / safetensors / GGUF) — NEVER use Python/MLX in the inference runtime hot path.
- Model storage: Download/store large model blobs & `.m64pack` files under `<MODEL_DIR>/` (do NOT fill the internal SSD).

## Hard Constraints
- No silent fallbacks, dummy/stub outputs, or mocked benchmarks.
- Every numerical/kernel implementation MUST pass parity tests against a reference implementation (max abs diff < 1e-3 for FP16; cosine sim > 0.999 for quantized outputs).
- Always validate on a small MoE (or reduced-layer slice) for exact numerical parity before scaling to 100B+ models.
- Track RSS, wired memory, macOS swap/pageins, prefill tok/s, decode tok/s, cache hit rate, and effective SSD GB/s in benchmarks.
- Keep `docs/DESIGN.md` and `docs/BENCHMARKS.md` up to date. Never save unverified assumptions to persistent memory.
