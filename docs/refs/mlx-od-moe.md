# mlx-od-moe — Reference Analysis

Repo: `refs/mlx-od-moe` (cloned 2026-11-22, https://github.com/kqb/mlx-od-moe)
Python/MLX reference design for on-demand expert paging, sized around
Kimi-K2.5-class models (28 MoE layers × 384 experts, top-8, hidden 4096,
FFN 14336). Read whole-repo (~31 files); ~2.4k LOC core + tests.

**Maturity caveat (verified)**: this is a *design-forward* repo. ARCHITECTURE.md
describes the ideal; the code implements a simpler version; performance
numbers in docs (70 tok/s target, >90% top-8 predictor accuracy) are targets,
not measured results — there is no benchmark output committed and **no parity
test against a reference engine** (tests use synthetic weights and sanity
assertions). Treat it as a clean taxonomy of the moving parts, not as a
worked example of throughput.

## 1. On-disk expert store (`convert/gguf_to_od_moe.py`, `mlx_od_moe/expert_store.py`)

- Output layout: `base_model/*.safetensors` (embeddings, attention, norms,
  lm_head = always-resident tier) + `experts/layer_LL_expert_EEE.safetensors`,
  **one file per (layer, expert)**, ~30 MB each, 10,752 files for Kimi-K2.5
  (384×28). Keys `w1` (gate/up per SwiGLU naming in layer code: w1=up-ish
  (hidden→ffn), w3=gate, w2=down), stored as plain arrays.
- GGUF quant handling: `parse_gguf_metadata()` reads the GGUF header + tensor
  info (`dtype` recorded as source `tensor_type`); extraction dequantizes —
  it is a convert tool, not a runtime path. No byte-level quant passthrough.
- Registry (`expert_store._initialize_registry`): walks the expected key space
  `layer_{L:02d}_expert_{E:03d}`, stores `{path, size, last_accessed}` in a
  dict — metadata only, ~10 KB, no weights touched.
- **File-per-expert vs Strata's single sidecar**: 10k+ small files means
  open()/stat() syscall churn, per-file safetensors JSON header overhead,
  dentry cache pressure, and RLIMIT_NOFILE issues at scale (SwiftLM hit
  exactly that at 182 shards). A packed contiguous blob with an offset index
  (what we will do) strictly dominates for our use.

## 2. LRU cache / hot tier (`expert_store.py: UnifiedMemoryExpertStore`)

- Structure: `OrderedDict[str, dict-of-mx.arrays]` + `current_cache_bytes`
  vs `cache_size` (default 48 GB), single `threading.RLock`. Plain LRU —
  no segmented/hot-protected split, no per-layer fairness: one LRU line
  across all layers, so a chatty layer can starve others (we should consider
  per-layer or class-partitioned LRU; Strata uses per-size-class LRU).
- `fetch()`: hit → `move_to_end` + return; miss → `mx.load(safetensors)`
  (full load; no madvise/memmap in the hot path despite ARCHITECTURE.md's
  "np.memmap Tier 3" claim — the memmap story is not in the runtime code),
  `_add_to_cache` evicts from the LRU front (`popitem(last=False)`) until it
  fits, freeing relies on Python GC for the MLX arrays — **non-deterministic
  eviction → memory spike risk**, exactly what a C++ arena must not do.
- `prefetch(layer, experts)`: ThreadPoolExecutor(4 workers), `prefetch_futures`
  dedup dict, skips already-cached; a worker just calls `fetch()`. Comment
  admits MLX isn't thread-safe; safety comes only from the lock around the
  cache, the actual GPU work is deferred by MLX laziness.
- Telemetry (directly liftable): `cache_hits/misses/prefetch_hits`,
  `bytes_loaded`, `fetch_times` → p50/p95/p99 + hit_rate +
  `working_set_experts` in `get_stats()`; `/health` endpoint exposes it.
- Pinned resident (never in the LRU): embeddings, lm_head, all attention,
  norms, per-layer router `gate` (4096×384 ≈ 1.5 MB, `od_moe_layer.py:46`),
  and the shadow model. Shared-expert handling: absent (Kimi assumed no
  shared expert) — our hot tier adds shared experts here.
- Per-layer `active_experts` dict is rebuilt every call and pruned to the
  current top-k after use (`load_experts()` deletes non-top-k) — so the layer
  holds no residency itself; the store is the only cache. Fine design
  separation: mechanism (store) vs policy (layer).

## 3. Shadow predictor (`shadow_model.py`, `training/`)

- `ExpertPredictor` (per the code, not ARCHITECTURE.md's fancier figure):
  mean-pool hidden_states over sequence → `Linear(4096→1024)+LayerNorm+SiLU`
  → **4 independent `Linear(1024→384)` heads** (one per lookahead layer
  t+1..t+4) → top-8 per head. ~4.4M params; target <1 ms.
- `ShadowRunner.predict_async(...)` runs the predictor off-thread; layers call
  `get_predictions_for_layer(self.layer_idx + 1)` at the *end* of their
  forward (`od_moe_layer.py:268-272`) and hand the IDs to
  `expert_store.prefetch()`. Note the mismatch: heads exist for 4 layers of
  lookahead but consumption is strictly N+1 (heads t+2..t+4 unused at
  runtime) — with NVMe at ~5 ms/30 MB, 1-layer-ahead is marginal to hide a
  miss; Strata's 2-layer-ahead router GEMV prefetch is the stronger design.
- No fallback cost control: a wrong prediction is just a wasted prefetch; a
  missed expert blocks the layer (`fetch()` on the critical path).
- Training loop (`training/collect_training_data.py` → `train_shadow.py`):
  labels = actual top-k router selections per future layer, cross-entropy;
  examples in `examples/train_shadow_example.py`. No published accuracy —
  unlike Strata (empirical router reuse statistics) or SwiftLM's PAPPS doc
  (previous-token routing ≈70% hit, N-1 heuristic >65%, learned SVD-router
  "useless"). Consensus across refs: a learned global predictor is the
  weakest option; use activation-pattern statistics + previous-token routing.

## 4. MoE-layer forward call sequence (`od_moe_layer.py:__call__`)

```
x (B,S,H) → x_flat
1. router_logits = gate(x_flat)                 # resident GEMV
2. argsort → top_k_indices; softmax → renormalized top_k_weights
3. aux load-balancing loss (training only, O(T*k) python loops — slow)
4. unique_experts = np.unique(indices.tolist())  # GPU→CPU sync, GIL
5. load_experts(): store.fetch() each (blocks on SSD miss)
6. for e in unique: apply_expert over ALL tokens, mask-weighted add
7. shadow predictions for layer+1 → store.prefetch(layer+1, preds)
```
Step 6 is O(unique_experts × all_tokens): every expert processes every token
and the weight masks it out — 48 GEMVs for 8/48-token batches in decode; and
step 4 forces a device→host sync per layer. Our kernel plan (route tokens to
per-expert buckets, gather only its tokens) plus router-on-GPU with a single
async readback is what this repo proves should *not* be done the Python way.

## 5. Parity / testing (`tests/`)

- `create_toy_model.py` builds tiny MoE fixtures; tests assert shapes, cache
  hit/miss counts, prefetch call args (e.g. `prefetch_layer == 6`), aux-loss
  bounds, weight-normalization `atol=1e-5`. **No comparison against a
  reference forward** — parity vs llama.cpp/MLX is left as an exercise.
- For us this validates the need for the AGENTS.md mandate: numeric parity
  harness first (per-tensor dequant parity, per-layer output cosine > 0.999),
  because on-demand loading has many places to silently use the wrong
  expert bytes (wrong stride/layer key).

## 6. What Mac64-MoE should steal / must do differently

Steal:
- Resident/paged split exactly as: {embed, lm_head, attention, norms, router,
  (ours: + shared experts, KV)} resident; routed experts paged. Their numbers
  for Kimi-class: base ≈35 GB, working set ≈11 GB (8×28×50 MB).
- Registry/manifest indexed at open time with byte sizes per expert, so the
  LRU is byte-budgeted (not entry-budgeted).
- Telemetry schema: hits/misses/prefetch-hits/bytes + p50/p95/p99 fetch
  latency + hit_rate — lift directly into our stats module (AGENTS.md already
  mandates cache hit rate + effective GB/s).
- Store(mechanism)/layer(policy) separation; prefetch dedup futures map.

Do differently (each backed by a defect seen above):
- One packed `.m64pack` blob, experts 16 KiB-aligned contiguous + offset
  index — no 10k-file tree, no per-file header overhead.
- Deterministic C++ arena with explicit slot free (no GC-dependent eviction).
- Prefetch ≥2 layers ahead, driven by cheap statistical router (previous-token
  routing + Strata-style top-k LRU heuristics), not a 4.4M-param MLP whose
  accuracy is unproven — and keep the shadow model idea only as an optional
  Phase 4 experiment.
- Router→GEMM pipelining without GPU→CPU syncs per layer (Strata's
  shared-event segmented command buffers), and token-bucketed expert GEMV
  instead of mask-all-tokens.
- Threaded pread pool (they got 4 workers; SwiftLM 24 concurrent preads;
  Strata 6–8 split reads) — we'll do an io_uring-style thread pool with
  F_NOCACHE.

## 7. Open questions

- Does any real deployment of this repo exist with measured tok/s? (No
  results committed; README targets unverified.)
- Shadow model accuracy on real Kimi routing — unmeasured here; if we ever
  try a learned predictor, measure against previous-token-routing baseline
  first (SwiftLM's data suggests the gap is small).
- MLA KV compression claim (58 GB → 7 GB at 256K) is Kimi-architecture
  general knowledge, not implemented in this repo (`KVCache` in model.py is
  plain dense MLX cache).
