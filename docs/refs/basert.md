# baseRT — Reference Analysis

Repo: `refs/baseRT` (cloned 2026-11-22, https://github.com/BaseCompute/baseRT)

## 0. Overview and open/closed split

**Open**: the `.base` file format spec + Rust reference implementation
(`base-convert/crates/base-format/`), the quantization converter
(`base-convert/crates/base-quant/`, `base-arch/`), the public C API headers
(`include/baseRT/*.h`), docs (`docs/reference/`), benchmarks scripts/CSVs,
language bindings (Python/Node/Rust/Swift).

**Closed source**: the GPU engine itself. No `.metal`/`.cpp`/`.mm` engine
source exists in the repo (verified with `find`). The engine is downloaded as
a prebuilt binary per `docs/reference/engine-releases.md`. The "Superfluid"
server (continuous batching, prefix caching, speculative decoding) is
documented but not implemented here. Everything below marked *engine behavior*
comes from docs/headers, not from engine source — treat as advertised API,
verified only at the interface level.

Scale: `base-convert/` is a Rust workspace (~15 crates); FORMAT.md 610 lines;
CANONICAL_QUANT_SPEC.md 258 lines.

## 1. `.base` format byte layout (`base-convert/FORMAT.md`, `base-format/src/{header,reader,slots}.rs`)

File layout (all LE):
```
0x0000  4   magic = b"BASE"
0x0004  4   format_version (u32) = 1
0x0008  8   header_len (u64)
0x0010  N   header_json (canonical JSON: sorted keys, no whitespace)
      pad to next 64 KiB boundary
0x???   M   weights_blob (raw tensor data, per-tensor alignment)
      extension slots after blob
EOF
```

Key decisions (FORMAT.md §"File layout", §"Compute regions"):
- **Weights blob starts at a 64 KiB boundary** (max page size across targets)
  so tensors aligned to 16 KiB (Apple page) land page-aligned at *absolute
  file offsets* — explicitly stated as "a precondition for zero-copy buffer
  creation via `MTLBuffer.makeBufferWithBytesNoCopy`".
- Tensor alignment is driven by a per-tensor `compute_region` field:
  `accelerator` (64–128 B, MLP/MoE stacks), `gpu` (16 KiB Apple / 64 KiB
  NVIDIA, embeddings/norms/shader path), `cpu` (64 B, RoPE tables etc.).
  Alignment read from header `alignment` field (log2 bytes), not hardcoded.
- Zero-copy invariant checked at load (`BaseReader::tensor_is_zero_copy_eligible`).
- Header JSON carries `tensors[]` entries: `name, dtype, shape, offset,
  length, scale_offset/length, bias_offset/length, awq_scale_offset/length,
  group_size, layout ("tile_8x8_mlx" | "rowmajor"), residency ("hot"|"warm"|"cold"),
  checksum_xxh64`.
- **Scales/biases in separate contiguous regions, not interleaved with weight
  bytes**, "so GEMM kernels can prefetch them into registers once per group".

Design invariants worth copying verbatim (FORMAT.md §Design invariants):
- **"No per-forward transformation"** — transpose, QKV fusion, gate+up
  SwiGLU fusion, AWQ scale folding, tile interleaving all happen at convert
  time; runtime does zero repacking per forward.
- Cross-backend repack is **cached** (`~/.cache/baseRT/repacked/<sha>.<target>.base`),
  paid once per (model, backend) pair.
- **Tensor ordering convention**: embed → layer0(attn,ffn) → … → final_norm →
  lm_head, "OS page-cache readahead naturally prefetches layer N+1 while layer
  N runs. MoE expert tensors cluster at the end of each layer's block."
- MoE expert stacks stored as **3D `[num_experts, out, in]` tensors**; header
  `config.moe_layout` declares `grouped` (all up-projs contiguous, then gate,
  then down — "matches the current Metal MoE kernel's access pattern", the
  default) vs `interleaved` (per-expert tuple contiguous).
- Residency hints per tensor: `hot` (embeddings, norms, routers, lm_head),
  `warm` (resident while layer active), `cold` (MoE inactive experts).
  "Hints are priors, not constraints. Runtime measures access and adjusts."
- Header `residency` budget block: `total_weights_bytes`, `hot_weights_bytes`,
  `max_expert_bytes_per_token`, `recommended_min_device_mb`; runtime budgets
  `MTLResidencySet` from it (engine-side, docs-level).
- **MTP/speculator sub-bundle**: `speculator` in header (`kind: "mtp"`,
  `mtp.layers.0.*` canonicalized tensor names), always f16; DFlash/DSpark
  drafters as sidecar bundles. Directly relevant to our MTP plan.
- Extension slots after blob: `u16 kind, u16 flags, u64 len, u64 xxh64, payload`
  (KvWarmup, ComputeGraph/MPSGraph archive, RopeTables, Custom). Unknown kinds
  skipped; `REQUIRED` flag = hard failure.
- Integrity: per-tensor xxh64 + optional ed25519 over canonical JSON header +
  sha256(blob).
- Layer map: typed `layers[]` array (`attention_gqa`, `attention_moe`, `ssm`,
  `moe_ffn`, …) + `config.layer_types` for hybrid (Qwen3.5 GDN: `"linear_attention" |
  "full_attention"`), plus GDN geometry fields (`gdn_num_k_heads`,
  `gdn_key_head_dim`, `gdn_conv_kernel`…) and DeepSeek router extras
  (`expert_gating` softmax/sigmoid, `e_score_correction_bias`).
- Auto-convert cache keyed on sha256 of input (`~/.cache/baseRT/converted/`).

## 2. Quantization (`base-convert/CANONICAL_QUANT_SPEC.md`, `base-quant/src/`, `profiles/*.json`)

Canonical affine family (RTN default, optional AWQ with calibration sidecar):

| bits | group_size | default scale | notes |
|------|-----------|---------------|-------|
| q2/q3 | **32** | bf16 | more scale resolution needed at low bits; e8m0 scale = 3.1% overhead, negligible |
| q4/q5/q6 | 64 | bf16 | matches MLX affine default |
| q8 | **128** | bf16 | 256 levels/group; scale storage 0.8% of bytes |

- Scale dtypes: bf16 (default), f16 (legacy), e8m0 (1-byte power-of-2,
  shift-only dequant on integer paths).
- Weights must start from fp16/bf16/fp32 source; re-quantizing quantized
  input refused unless `--allow-quant-from-quant`; `--kquant-passthrough` /
  `--mlx-passthrough` keep GGUF/MLX bytes (`passthrough_gguf` dtype,
  `layout: gguf_super`).
- Blob layouts: `base_q4`: `[packed_i4: gs/2 B][scale f16 region][bias f16 region]`
  (asymmetric); `base_q8`: `[i8: gs B][scale f16]` (symmetric);
  `mxfp4`: E2M1 + E8M0 u8 scale gs=32; `nvfp4`: E2M1 + E4M3 gs=16.
- **Mixed-precision policy is a JSON profile of name-glob rules**
  (`profiles/qwen3-moe-q6.json`, `moe-q4mix.json`, `qwen35-moe-cuda-q4mix.json`):
  norms/q_norm/k_norm → f16; router (`ffn_gate_inp`, `mlp.gate`) → **q8 gs=128**;
  **shared expert → q8 gs=128** while routed experts → q4/q6 gs=64; embed →
  q6 or f16; lm_head often kept higher. This tiering (router/shared high
  precision, routed experts aggressive) is exactly our split too.
- `base_qn.rs` = N-bit general path; `profile.rs` applies rules;
  `rtn.rs` = round-to-nearest; `base-awq` crate = AWQ calibration pipeline.

## 3. C API surface (`include/baseRT/types.h`, `baseRT.h`)

- `baseRT_model_config_t` returned **by value** with an explicit
  `baseRT_model_config_sizeof()` ABI check ("mismatches are detectable rather
  than silent") — cheap, good ABI hygiene pattern we should copy.
- Config exposes per-layer arrays (`ffn_dims[128]`, `n_kv_heads_per_layer[128]`),
  hybrid switches (`full_attention_interval` — "every Nth layer is full
  attention (rest are GDN)"; `partial_rotary_factor` "Qwen3.5 = 0.25"), GDN
  geometry, SSM geometry, MoE block (`n_experts`, `n_experts_used`,
  `n_experts_shared`, `expert_ffn_dim`, `expert_weights_scale`), PLE per-layer
  embedding (`n_embd_per_layer`, `n_layer_kv_from_start`), logits softcap.
- Memory planning API: `baseRT_device_memory_budget()` (a fraction of RAM the
  OS recommends), `baseRT_suggest_max_context()` / `baseRT_context_window_fits()`
  (auto context sizing within the budget, incl. co-resident multi-model
  variants), `BaseRTMemoryStats {capacity, used…}` with a KV allocation
  strategy enum; KV precision switch: "bits = 16 → force F16 K and V slabs"
  (i.e. quantized KV is the default path).
- Weights described as "mmap-backed Metal buffers independent of the kernel
  page cache" (baseRT.h:161) — mmap + no-copy buffers is their resident model;
  streaming/expert paging internals are engine-closed (no CLI flag for it in
  headers/docs beyond the residency hints).
- Decode surface is token-step oriented: `baseRT_decode_token*`,
  `baseRT_batch_step_fused*` (fused forward incl. logits/sample variants),
  `baseRT_decode_stream_*`, drafter hooks (`baseRT_drafter_step`,
  `baseRT_drafter_confidence`, `drafter_markov_argmax` — Markov head
  speculation), chain decode.

## 4. MoE arch/tensor conventions (`base-arch/src/{qwen,gpt_oss,glm,muse_glimmer}.rs`)

- Canonical naming keeps HF names with `**` globs in profiles:
  `model.layers.N.mlp.experts.{e}.{gate,up,down}_proj.weight`, shared expert
  under `mlp.shared_expert.*` / `mlp.shared_experts.*` / Gemma-style dense
  `mlp.*` as shared (`n_experts_shared=1`), router `ffn_gate_inp.weight`
  (Mixtral-style) or `mlp.gate.weight`.
- Expert stacks collapsed to 3D tensors at convert (see §1), so on-disk an
  expert = a fixed stride inside one big tensor — byte-range extraction of
  expert e is `base + e * stride`.
- Router variants: softmax vs sigmoid gating + DeepSeek score-correction bias
  tensor (format §Layer map). Relevant if we ever target DeepSeek-style MoE.

## 5. Advertised engine behaviors + benchmarks

- `docs/reference/engine-releases.md` and README advertise: hand-written GPU
  kernels, mmap weights, continuous batching, prefix caching, speculative
  decoding, OpenAI/Anthropic/Ollama-compatible server. No public detail on
  expert streaming — baseRT targets models that fit RAM with mmap, unlike
  Strata/SwiftLM. (No `--mmap-experts`-style flag found; grep-verified.)
- `benchmarks/`: M3/M4 Pro CSVs + shell harness comparing baseRT vs MLX
  (`baseRT_mlx_bench.sh`, `fetch_3way_models.sh`) — prefill/decode tok/s
  matrix; methodology reusable (fixed prompt sets, clean-run script).

## 6. What Mac64-MoE should steal

1. **Blob start at 64 KiB, GPU tensors at 16 KiB absolute alignment + a
   load-time zero-copy eligibility check** — same precondition we need for
   `newBufferWithBytesNoCopy` (Strata's stream.cpp reaches the same rule
   independently with 16 KiB pages).
2. **"No per-forward transformation"** as a hard format invariant; convert-time
   tile interleave keyed by a per-tensor `layout` string; repack cache for
   alt backends.
3. **Scales/biases as separate contiguous regions** (not interleaved) so a
   GEMV/GEMM kernel bulk-loads scales once per tile.
4. **Layer-order tensor serialization with experts clustered at each layer's
   end** — gives us page-cache-friendly next-layer readahead for free even in
   the pread path; also makes the hot-tier extraction pass a linear scan.
5. **Per-tensor `residency: hot|warm|cold` hints + header `residency` budget
   block** (`hot_weights_bytes`, `max_expert_bytes_per_token`,
   `recommended_min_device_mb`) — we should make these *computed at pack time*
   and verified at load before we allocate anything.
6. **Quant profile JSON** (name-glob rules → dtype/group/scale-dtype) as the
   pack-time spec for our Q4/Q6/Q8 tiering incl. router/shared-expert q8 rule.
7. **`grouped` expert layout** (all experts of one projection contiguous per
   layer) chosen to match the kernel access pattern; per-expert byte range =
   `base + e*stride` — simplest possible random access for our arena slots.
8. **Extension slots** (typed, length-prefixed, xxh64, unknown-skipped) for
   MTP drafter weights, KV warmup, precompiled graphs.
9. **ABI-sized struct returns + sizeof handshake** for our C API.
10. **Config struct completeness** (GDN/hybrid/MoE/PLE fields, per-layer
    arrays) as a checklist for our own header JSON.

## 7. Open questions / limitations

- Engine internals (residency-set usage, streaming policy, KV quant kernels)
  are closed; we cannot verify their claimed behaviors from source.
- baseRT has no demonstrated >RAM MoE streaming path — format supports the
  hints, engine may not use them aggressively. Not a reference for SSD
  pipeline design (Strata/SwiftLM are).
- `passthrough_gguf` escape hatch conflicts with our "one loader" goal; we
  should skip it and quantize everything at pack time.
- ed25519 signing: adopt the *structure* later, not Phase 1.
