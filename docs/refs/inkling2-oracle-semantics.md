# inkling2_loop oracle semantics (llama.cpp @ e300919, CPU backend, `--ngl 0`)

Source root: `<ORACLE_DIR>/llama.cpp` (HEAD `e300919cc704ddde3a8a8346a8714f57eeeee71b`).
All line numbers refer to that tree. This digest is the implementation contract for the mac64 CPU engine
(logits cosine >= 0.999 vs oracle).

## 0. ggml conventions used throughout (verified against ggml.c)

- ggml tensors are reverse-column-major: flat index of element (i0,i1,i2,i3) = `i0 + ne0*(i1 + ne1*(i2 + ne2*i3))`.
  A "matrix" W with `ne = {n_in, n_out}` stored contiguously is exactly the C array `W[n_out][n_in]` (row-major rows of length n_in).
- `ggml_mul_mat(a, b)` (ggml.c:3366): requires `a->ne[0]==b->ne[0]`; result `ne = {a->ne[1], b->ne[1], b->ne[2], b->ne[3]}`, always F32;
  `res[j][t] = Σ_i a_flat[j*n_in + i] * b_flat[t*n_in + i]` (plain row-major dot; CPU accumulates f32; `ggml_prec_set_acc` F32 only forces F32 on GPU, CPU is F32 regardless).
- `ggml_transpose(a) == ggml_permute(a,1,0,2,3)`; `ggml_permute(a, x,y,z,w)` builds a view with `result->ne = {a->ne[x], a->ne[y], a->ne[z], a->ne[w]}` (verified in ggml.c:3890 ff.).
- `ggml_get_rows(a, b)` (ggml.c:3966): asserts `a->ne[2]==b->ne[1]`, `b->ne[3]==1`, b I32; result `ne = {a->ne[0], b->ne[0], b->ne[1], b->ne[2]}`;
  gathering is over a's **ne1 axis**, iterating b's flat elements in order: `res_flat[k + ne0*m] = a[k + ne0*idx_m]`, `idx_m = b_flat[m]`.
- `ggml_pad(a, p0l,p0r,p1l,p1r,...)` zero-appends: result row `dst[i1] = [0]*p1l ++ src_row ++ [0]*p1r` along each axis; contiguous output; fill value 0 (ggml_compute_forward_pad_f32, ops.cpp:8433).
- `ggml_concat(a, b, dim)`: result has `a` first along `dim`, then `b`.
- Elementwise binary ops broadcast by equal-or-1 dims along ne0..ne3 (per dim, in order).
- CPU softmax (`ggml_compute_forward_soft_max_f32`, ops.cpp:5828, verbatim excerpt):
  ```
  5890| ggml_vec_cpy_f32(ne00, wp, sp);
  5891| ggml_vec_scale_f32(ne00, wp, scale);        // scale applied FIRST
  5894|   for (i) wp[i] += slope*mp_f32[i];          // mask (bias+mask) added SECOND; slope=1 (alibi off)
  5911| float max = -INFINITY; ggml_vec_max_f32(ne00, &max, wp);
  5915| if (sk) max = MAX(max, sk[i02]);              // sink correction (sinks here = 0 → max>=0)
  5919| ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);   // dp[i]=exp(wp[i]-max), sum accumulated in DOUBLE
  5922| if (sk) sum += (ggml_float) expf(sk[i02] - max);
  5926| sum = 1.0/sum;
  5927| ggml_vec_scale_f32(ne00, dp, sum);
  ```
  Softmax reduces along **ne0**. Scale is applied to the raw values, mask added after scaling.
- q8_1 GEMV path: activations are quantized to q8_1 in blocks of 32 (round-to-nearest, block scale = amax/127), dot = per-block float accumulation of `d_x*d_w*Σ qx*qw`. For cosine>=0.999 the engine may keep activations in fp32; expected deviation ~1e-4 relative on large GEMVs.

```
IMPLEMENTATION NOTE (conventions)
W_ggml[n_in, n_out] == C row-major W[n_out][n_in];  out_row[o][t] = dot(&W[o*n_in], &x[t*n_in]).
All reductions over a ggml ne0 axis = contiguous C loop; softmax denominators in double.
```

## 1. mix() — attn-res / ffn-res / message mixing (inkling2.cpp:179-192, verbatim)

```
179| ggml_tensor * mix(blocks, cur, query) {
180|   auto states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
181|   for (const auto & block : blocks) {
182|     states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
183|   }
184|   states = ggml_rms_norm(ctx0, states, 1e-6f);
185|   ggml_tensor * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));
187|   ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
188|   scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);
189|   return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
```

- Concat order: each block is put **FIRST** on dim1 → after pushing blocks `b0..bn-1` (vector order) the dim1 layout is
  `[b_{n-1}, b_{n-2}, ..., b_0, cur]` — newest block at index 0, `cur` at the LAST index `nb`.
- `rms_norm(states, 1e-6)` normalizes along ne0 (=n_embd) independently for every (slot i, token t). NOTE: no weight tensor here; plain RMS with eps=1e-6, `y = x * rsqrt(mean(x^2)+1e-6)`.
- `queries`: `query [n_embd, q]` repeated over tokens → `[n_embd, q, T]`, `query[c][j]` shared over t.
- `scores = mul_mat(states [n_embd, S, T], queries [n_embd, q, T])` → `[S, q, T]`,
  `scores[i][j][t] = Σ_e states[e,i,t]*query[e,j]`.
- `soft_max_ext(scale=1/sqrt(2304)=1/48, mask=null)`: softmax over **ne0 (i = the S state axis)** per (j,t):
  `w[i][j][t] = exp((scores[i][j][t]-max_i)/48) / Σ_{i'} exp((scores[i'][j][t]-max_i')/48)` (scale applied to scores before max/exp; double sum; then multiply by 1/sum).
- Final `mul_mat(transpose(cont(states)) [S, n_embd, T], scores [S, q, T])` → `[n_embd, q, T]`:
  `out[e][j][t] = Σ_{i=0..S-1} states_rms[e,i,t] * w[i][j][t]`.

```
IMPLEMENTATION NOTE (mix)
S = blocks.size()+1. Build array of pointers st[S]: st[i] = blocks[S-2-i] for i<S-1 (newest first), st[S-1]=cur.
rms_norm each st[i] row-wise (eps 1e-6) → rs[i][e].
for j in 0..q-1: for t: for i: s[i] = dot(rs[i][*t], query[:, j]); softmax over i with scale 1/48;
  out[t][e][j] = Σ_i w[i]*rs[i][e].
Output layout [n_embd, q, T]; callers slice with offset j*n_embd*T (row-major per token block) or nb[1]-stride equivalent.
```

## 2. control() + message views (inkling2.cpp:241-249, 287-289; models.h:2511)

```
241| ggml_tensor * control(cur, message, loop, ctrl) {
242|   ggml_tensor * gate = ggml_concat(ctx0,
243|       ggml_rms_norm(ctx0, cur, 1e-6f), ggml_rms_norm(ctx0, message, 1e-6f), 0);   // [2*n_embd, T], cur first
244|   gate = build_lora_mm(ctx0, loop.ctrl_down[ctrl], gate);                          // [64, T]
245|   ggml_tensor * gain = ggml_tanh(ggml_mul_mat(ctx0, loop.ctrl_gain[ctrl], gate));  // [2304, T]
246|   ggml_tensor * forget = ggml_tanh(ggml_mul_mat(ctx0, loop.ctrl_forget[ctrl], gate));
248|   return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)),
249|                              ggml_mul(ctx0, gain, message));
287| auto message = [&](int ctrl) {
288|   return ggml_cont(ctx0, ggml_view_2d(ctx0, messages, n_embd, n_tokens, messages->nb[2], ctrl*messages->nb[1]));
```
`enum { LOOP_CTRL_Q, LOOP_CTRL_K, LOOP_CTRL_V, LOOP_CTRL_FFN, LOOP_CTRL_COUNT }` (models.h:2511) = 0,1,2,3,4.

- `messages` is the mix output `[n_embd, 4, T]`; dim1 IS the ctrl axis; `message(ctrl)` = flat slice `[ctrl*n_embd*T, ...)` viewed `[n_embd, T]`.
- Formula: `g = rms16(cur) || rms16(message)` (concat along ne0: cur occupies rows 0..2303, message 2304..4607; ctrl_down ggml `[4608,64]` = C `[64][4608]`),
  `h = tanh(G64 @ g)` where G64 = ctrl_down (row r: dot over 4608), `gain = tanh(A64 @ h)`, `forget = tanh(B64 @ h)` (ctrl_gain/forget ggml `[64,2304]` = C `[2304][64]`),
  `out = (cur - forget*cur) + gain*message` — three separate elementwise ops, not fused.

```
IMPLEMENTATION NOTE (control)
for t: h64 = tanh_vec(ctrl_down_c * [rms(cur[t]); rms(msg_c[t])]);  gain = tanh(ctrl_gain_c*h64); forget = tanh(ctrl_forget_c*h64);
out[t][e] = cur[t][e] - forget[e]*cur[t][e] + gain[e]*msg[e];   // keep the 3-step order for bit-similarity
```

## 3. build_layer order (inkling2.cpp:274-418)

Exact per-layer order (slot layer `il`, physical `ip`):
1. `state = recr->mctx->get_r_l(il)` (per-slot conv state), `history = build_rs(...)` reshaped `[n_conv-1=3, n_sconv_channels, n_seqs]` (:291-293).
2. `inpSA = cur` (:295).
3. attention inputs mix (:298-303): `inputs[0..2] = cur`; if attn_res_query: `mixed = mix(blocks, cur, attn_res_query)` `[n_embd, q, T]` (q=3 full, 1 swa);
   `inputs[j] = cont(view_2d(mixed, n_embd, T, mixed->nb[2], (is_swa?0:j)*mixed->nb[1]))` — flat slice per query j.
4. messages control (:306-311): `inputs[j] = control(inputs[j], message(LOOP_CTRL_Q+j), loop, LOOP_CTRL_Q+j)` for j=0,1,2 (Q,K,V).
5. norms (:313-324):
   - `is_swa && messages`: all three → `rms(inputs[j], attn_norm, 1e-6)`.
   - `is_swa` only: `inputs[0] = rms(inputs[0], attn_norm)`; `inputs[1] = inputs[2] = inputs[0]` (same tensor!).
   - full attention: Q→`attn_q_inp_norm`, K→`attn_k_inp_norm`, V→`attn_v_inp_norm`.
   `build_norm` (llama-graph.cpp:1943-1976) = `ggml_rms_norm(cur, f_norm_rms_eps)` **then separate** `ggml_mul(cur, mw)` (weight `[n_embd]`) — eps here = 1e-6 (`LLM_KV_ATTENTION_LAYERNORM_RMS_EPS`).
6. GEMVs (:328-333): `Qcur = wk @ inputs[0]` `[3072,T]`; `Kcur = convolve(wk' @ inputs[1], attn_k_conv, dyn=wk_conv_proj@inputs[1], state, history, recr, rs_off=0)`; `Vcur = convolve(wv @ inputs[2], attn_v_conv, dyn=wv_conv_proj@inputs[2], ..., rs_off=n_embd_k_gqa=1024)`; `Rcur = attn_rel @ inputs[0]` `[n_rel*n_head? no: [384,T]]` — see below; then reshaped `[16, 24, T]` at :338. (attn_rel ggml `[2304, 16*24=384]`; column block for head h starts at h*16.)
7. clip (:335-338, clip body :165-176):
   - `Qcur3 = reshape3 [128, 24, T]`; scale = `attn_q_clip` ggml `[128, 24]` → elementwise multiply (per dim-offset AND head, broadcast over tokens) — NOT per-head scalar;
   - `norm[i1][i2] = sqrt(Σ_{d<128} cur[d,i1,i2]^2)`; `norm = norm*1 + 1e-6` (`scale_bias(1.0f, 1e-6f)`); `factor = clamp(qk_clip/norm, 0, 1)`; `cur *= factor`.  qk_clip=22.627416997969522.
   - K: same with scale=nullptr; V: `ggml_rms_norm(Vcur3, 1e-6)` (no weight); R: reshape `[16, 24, T]`, scale `attn_rel_clip` `[16, 24]`, threshold `relative_clip=8.0`.
8. tau (:340-343): if `!is_swa`: `Qcur *= tau; Rcur *= tau`, tau `[1,1,T]` broadcast. `tau[t] = 1 + 0.1*log(max((pos[t]+1)/8192, 1))` (build_inputs :260-262). For positions < 8191 this is exactly 1.0.
9. sinks GEMV (:355): `sinks = attn_sink_proj @ inputs[0]` → `[24, T]` (per head, per token).
10. rel-bias table (:358-366):
    - `bias_pad = ggml_pad(attn_rel_bias, 0, 1, 0, 0)`: ggml `[16, 1024]` → `[16, 1025]`; row (distance col) 1024 is all zeros.
    - `table = mul_mat(bias_pad, Rcur3)` → `[1025, 24, T]`: `table[bucket][h][t] = Σ_{r<16} bias[r][bucket]*Rcur[r,h,t]` (bias_flat = bias[bucket*16 + r]).
    - `table -= reshape_3d(sinks, 1, 24, T)` → `table[b][h][t] -= sinks[h][t]` (sink folded in; comment :360: "exp(s-t)/(1+sum(exp(s-t))) = exp(s)/(exp(t)+sum(exp(s))), with a zero sink").
    - `table = cont(permute(table, 1, 0, 2, 3))` → `[24, 1025, T]` contiguous (h slowest→ actually ne0=24: flat `h + 24*(bucket + 1025*t)`).
    - `kq_b = get_rows(table, rel_pos)` — gathers along the bucket axis: `[24, n_kv, T]`, flat `kq_b[h + 24*(m)]`, `m = rel_pos_flat[token i * n_kv + kv j]` (rel_pos data is written `data[i*n_kv + j]`, :2015 kv-cache).
    - `reshape_4d [24, n_kv, T/ns, ns]` → `cont(permute(2, 0, 1, 3))` → `[T/ns, 24, n_kv, ns]` = the per-head dense bias in the mask layout.
    - `sinks = ggml_fill(new f32[n_head], 0.0f)` — a ZERO sink vector is passed to build_attn (:369).
11. attention (:371-373): `cur = build_attn(inp, nullptr,nullptr,nullptr, Qcur, Kcur, Vcur, kq_b, sinks(zeros), kq_scale=1/sqrtf(128), il)` — full call chain in §5/§6.
12. gate/output (:375-376): `cur = cur * sigmoid(wqkv_gate @ inputs[0])` (`[3072,T]` — per head-dim gate); `cur = convolve(wo @ cur, attn_post_conv, nullptr, state, history, recr, rs_off=2*n_embd_k_gqa=2048)`.
13. residual (:381-385): `if (ip % 6 == 0) blocks.push_back(inpSA); else cur = inpSA + cur;` — ip is the PHYSICAL index (0..23); push happens AFTER the attn mix and BEFORE the ffn mix.
14. `ffn_inp = cur` (:387).
15. ffn mix (:391): `cur = reshape2d(mix(blocks, cur, ffn_res_query [n_embd,1]))`.
16. if messages: `cur = control(cur, message(LOOP_CTRL_FFN), loop, LOOP_CTRL_FFN)` (:392-394).
17. `cur = rms(cur, ffn_norm, 1e-6)` (:395).
18. MoE (:397-410) — see §9.
19. `cur = convolve(cur, ffn_post_conv, nullptr, state, history, recr, rs_off=2*n_embd_k_gqa + n_embd = 4352)` (:412).
20. `cur = ffn_inp + cur` (:415).

```
IMPLEMENTATION NOTE (build_layer order)
norm → (control before norms for Q/K/V; control AFTER mix for FFN) → GEMV(Q,K,V,R) → clip → tau → sinks GEMV
→ bias table+gather → attn → sigmoid gate → wo → post-conv(rs 2048) → block push/add
→ ffn mix (blocks AFTER push) → control → ffn_norm → MoE → post-conv(rs 4352) → += ffn_inp.
```

## 4. rel_pos inputs (llama-kv-cache.cpp:2008-2022; llama-graph.cpp:1415-1434)

```
2008| void llama_kv_cache::set_input_relative_pos(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
2009|     GGML_ASSERT(dst->type == GGML_TYPE_I32);
2010|     const int64_t n_kv = dst->ne[0];
2012|     for (i < ubatch->n_tokens) {
2013|         const auto & cells = v_cells.at(seq_to_stream.at(ubatch->seq_id[i][0]));
2014|         for (int64_t j = 0; j < n_kv; ++j) {
2015|             const int64_t distance = cells.is_empty(j) ? -1 : int64_t(ubatch->pos[i]) - cells.pos_get(j);
2016|             data[i*n_kv + j] = distance >= 0 && distance < extent ? distance : extent;
```
- `self_rel_pos` I32 `[n_kv_base, n_tokens]`, set with `extent = hparams.n_relative_extent = 1024` (base cache).
- `self_rel_pos_swa` I32 `[n_kv_swa, n_tokens]`, set by the SWA cache with `extent_swa = hparams.n_swa = 1024` (llama-graph.cpp:1433: `set_input_relative_pos(self_rel_pos_swa, ubatch, extent_swa)`).
- **Head-independent** (single gathered bucket index per (query token, kv cell); the per-head dependence comes from Rcur·bias after the gather).
- Bucket `extent` (=1024) ⇒ zero column of the padded table for: empty cells, future/invalid cells (distance<0), and distance >= extent.
- The current ubatch is already inserted into the cell list before the mask/rel_pos fill (positions reserved via apply_ubatch), so the query's own cell yields distance 0.

```
IMPLEMENTATION NOTE (rel_pos)
For engine-side per-slot cache of size nkv: relpos[i][j] = (cell j occupied && 0 <= p_i - pos_j < 1024) ? p_i-pos_j : 1024;
bias_col[1024] = zeros; kq_b[i][j][h] = table[h][relpos[i][j]] (table = bias_pad^T·R - sink).
```

## 5. build_attn for this model (llama-graph.cpp:4388-4422, 3679-3781, 3106-3253)

Rel-pos overload (verbatim, :4400-4421):
```
4400| const bool is_swa = hparams.is_swa(il);                       // il = SLOT index; slot inherits physical swa
4404| const auto & attn_mask = is_swa ? inp_attn->self_attn_mask_swa : inp_attn->self_attn_mask;
4408| GGML_ASSERT(attn_mask.rows == nullptr);
4411| ggml_tensor * kq_mask = attn_mask.raw;                          // [n_kv, n_tokens/ns, 1, ns], F32 or F16
4412| if (kq_mask->type != GGML_TYPE_F32) kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
4415| kq_mask = ggml_add(ctx0, kq_b, kq_mask);                        // kq_b [T/ns, h, n_kv, ns] broadcasts mask over heads
4416| if (cparams.flash_attn) kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
4421| return build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur,
                                 nullptr /*kq_b*/, kq_mask, nullptr /*kv_rows*/, sinks, nullptr /*v_mla*/, kq_scale, il);
```
- **The kq bias is folded INTO the mask**; `build_attn_mha` receives `kq_b == nullptr`, so `use_flash_attn = cparams.flash_attn` (llama-graph.cpp:3147 `const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;`).
- Mask values (llama-kv-cache.cpp:1718-1719): `mask_keep = 0.0f`, `mask_drop = -INFINITY`; causal drop when `p0 > p1` (kv pos > query pos); SWA mask additionally drops when `llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)` (standard type: drop if `p0 < p1 - n_swa`); cross-sequence/empty cells dropped. alibi OFF (`f_max_alibi_bias = 0` passed → slope=1 in softmax).
- FA auto-resolution (llama-context.cpp:708-711): `if (cparams.auto_fa) resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);` — the probe enables FA unless the scheduler sees a device mismatch; the CPU backend supports FLASH_ATTN_EXT (`ggml_backend_cpu_device_supports_op`, ggml-cpu.cpp:424-488, `default: return true`). With default CLI flags `--ngl 0` ⇒ **FA is very likely ENABLED in the oracle** (q,k,v F32→F16 cast, mask F16, `sinks` handled inside the kernel). If the oracle was run with `--flash-attn off` the dense path below (F32 mask) is exact. For cosine>=0.999 both agree within ~1e-3; reproduce FA's F16 casts of Q/K/V if you want bit-level closeness.
- No attn softcap here: `hparams.attn_soft_cap` false → SOFTCAP=0 (dense branch :3203 skipped; FA: `logit_softcap != 0.0f ? scale /= softcap` skipped).
- `kq_scale = 1.0f/sqrtf(128)` applied in the softmax (`wp = raw*scale + mask`).

**Dense path** (llama-graph.cpp:3183-3240, verbatim):
```
3185| ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);                  // k [128, n_kv, h_kv, ns], q [128, tq, h, ns] (stream-permuted)
3188| ggml_prec_set_acc(kq, GGML_PREC_BF16);                        // no effect on CPU
3212| if (kq_b) kq = ggml_add(ctx0, kq, kq_b);                      // NOT used here (kq_b==nullptr)
3217| kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
3218| ggml_soft_max_add_sinks(kq, sinks);                            // sinks = zeros [24]
3221| if (!v_trans) v = ggml_cont(ctx0, ggml_transpose(ctx0, v));    // v^T [n_kv, 128, h_kv, ns]
3227| ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);                 // [128, tq, h, ns]
3236| cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);                     // [128, h, tq, ns]
3239| cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);  // [3072, T]
```
GQA: q/k were `permute(0,2,1,3)`-ed to `[d, tq, h, ns]` / `[d, nkv, h_kv, ns]`; `mul_mat` broadcasts heads with FLOOR mapping (kv = h/3, Appendix A): `kq[jkv][t][h][s]` (pre-scale) `= Σ_{d<128} k[d + 128*(jkv + nkv*((h/3) + h_kv*s))] * q[d + 128*(t + tq*(h + h*s))]` (h_kv=8, r2=3; kq then permuted to `[jkv, tq, h, ns]` by the stream permute).
FA path: same semantics with online-block softmax; FA out reshape yields the same logical `[3072, T]` (`o = h*128 + d` rows), and adds sink `s=0` as `M=max(M,0); S+=exp(0-M)` inside (ops.cpp:8858-9058).
Sink handling (both paths, dense kernel excerpt §0): denominator gets `+exp(0 - max)`; numerators contain no extra term; the per-query sink t was already SUBTRACTED from all gathered biases (§3.10), realizing `exp(s-t)/(1+Σexp(s-t))` as `exp(s)/(exp(t)+Σexp(s))` with sink logit `t` moved to the zero-sink form.

```
IMPLEMENTATION NOTE (attention, dense-equivalent)
For query token t, head h (kv head = h / (n_head/n_head_kv) = h/3, floor — see Appendix A):
  s[j] = (dot(Q[t,h,*], K[j,kv,*]) / sqrt(128)) + bias[t][j][h] + (allowed(j,t) ? 0 : -INF)
  M = max(max_j s[j], 0.0f)                       // zero sink
  dsum = Σ_j exp(s[j]-M)  + exp(0-M)              // +1 sink term, double
  out[t, h*128+d] = Σ_j exp(s[j]-M)/dsum * V[j, kv, d]
Then cur = reshape to [T][3072] with row = h*128+d. (FA oracle additionally rounds Q,K,V and bias+mask to f16 first.)
```

## 6. Attention output + blocks bookkeeping (inkling2.cpp:375-385, 391; inkling2-loop.cpp:76-115)

Output path: `cur[t][o] *= sigmoid(dot(wqkv_gate_row_o, inputs0[t]))`; `tmp = wo @ cur` (`wo [3072,2304]` C rows=2304); `cur = convolve(tmp, attn_post_conv, nullptr, ..., rs_off=2048)` (§7).

Blocks bookkeeping (`ip` = PHYSICAL index ∈ 0..23; push when `ip % 6 == 0`, i.e. ip ∈ {0,6,12,18}; push happens after the attn mix / before ffn mix):

Graph driver (inkling2-loop.cpp:81-115, verbatim above in §12):
- **Prelude** (ip 0..5, slots 0..5): ip0: attn mix sees `blocks=[]` (mix over just `cur` → S=1, softmax of 1 element = 1.0, i.e. mix is `rms(cur)` alone); after attn: push `B0 = inpSA(layer0)`. ip1..5: additive residual `cur += inpSA`.
  After prelude: `blocks = [B0]`; `prelude = cur`; `prelude_blocks = [B0]`.
- **Pass 0** (ip 6..17, slots 6..17): ip6: attn mix blocks=[B0]; push B6. ip7-11 additive. ip12: attn mix [B0,B6]; push B12; ffn mix [B0,B6,B12]. ip13-17 additive.
  Within a pass, layer ip's attn mix uses blocks accumulated in THIS pass up to ip (e.g. ffn mix at ip=12 includes B12 pushed moments earlier at the same layer).
- **Pass p>0**: `bank = blocks.begin() + prelude_blocks.size() + 1 .. end` = core blocks EXCLUDING the first one (`[B12^p-1]` here, the pass's own last core block); `bank_out = cur` (that pass's output); then `cur = prelude; blocks = prelude_blocks` (=[B0] restored). `messages = mix(bank, bank_out, msg_query [2304,4])` computed ONCE per layer (constant through the pass; dim1 of mix output = `[bank_out, B12]`). Slots: `il = n_phys + (pass-1)*12 + (ip-6)` (24..47).
  Pass 1 ends with blocks=[B0, B6^1, B12^1]; pass 2: bank=[B12^1], bank_out=pass1_out; ends blocks=[B0, B6^2, B12^2].
- **Coda** (ip 18..23, slots 18..23, running on the pass-2 blocks): ip18: attn mix [B0,B6^2,B12^2]; push B18; ffn mix [B0,B6^2,B12^2,B18]; ip19-23 additive.
- **Output mix**: `blocks = [B0, B6^2, B12^2, B18]`; mix dim1 order = `[B18, B12^2, B6^2, B0, cur]` (§1 concat order).

```
IMPLEMENTATION NOTE (blocks)
Maintain blocks as vector<string tensor refs>. push iff ip%6==0 (ip physical). mix() consumes a SNAPSHOT of blocks at the call site:
attn mix = before push; ffn mix = after push. Between passes: bank = blocks[prelude_sz+1 .. end]; bank_out = last cur;
reset blocks = prelude_blocks, cur = prelude_out.
```

## 7. convolve() (inkling2.cpp:196-238) + ggml_ssm_conv CPU kernel (ops.cpp:9942)

```
196| convolve(cur, weight, dynamic, state, history, recr, offset):
197|   n_conv = hparams.n_shortconv_l_cache;                        // 4
198|   n_channels = cur->ne[0];
200|   past = ggml_view_3d(ctx0, history, n_conv-1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);
202|   x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));   // [n_seq_tokens, n_channels, n_seqs]
203|   x = ggml_concat(ctx0, past, x, 0);                            // [3 + n_seq_tokens, n_channels, n_seqs]; OLDEST first
204|   build_conv_state_store(inp, state, x, n_conv-1, offset*(n_conv-1)*ggml_element_size(state));
206|   out = ggml_reshape_2d(ssm_conv(x, weight), n_channels, n_tokens);
208|   if (dynamic) {   // K/V of full-attention layers only
209|     n_heads = dynamic->ne[0]/n_conv;                           // 1024/4 = 256
210|     dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);
213|     for (lag in 0..n_conv-1) {
214|       tap = view_3d(dynamic, 1, n_heads, n_tokens, nb1, nb2, lag*nb[0]);
216|       window = view_3d(x, n_seq_tokens, n_channels, n_seqs, nb[1], nb[2], (n_conv-1-lag)*nb[0]);  // start col = 3-lag
217|       window = cont(transpose(window));                        // [n_channels, n_seq_tokens]
218|       window = reshape_3d(window, n_channels/n_heads, n_heads, n_tokens);   // [4, 256, T]
220|       term = reshape_2d(ggml_mul(window, tap), n_channels, n_tokens);
221|       out = add(out, term);
224|   return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
```
`ggml_compute_forward_ssm_conv_f32` (ops.cpp ~9942, verbatim core):
```
sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];   // "not using ggml_vec_dot_f32, because its sum is in double precision"
```
i0 = tap 0..nc-1, i1 = channel, `ncs = ne0(x) = n_conv-1+n_t`, n-row stride of s = 1, of c(w) = nc. Pointer `s` starts at window row `i2` (the output token index):

- **`out[c][t] = Σ_{k=0..3} w[k][c] * x[t + k][c]`** where x columns `0..2` = history (oldest first: col0 = oldest lag), cols `3..3+T-1` = current tokens in order. So `w[3]` weights the CURRENT token, `w[0]` the most distant; float sequential accumulation.
- state store (llama-graph.cpp:4184-4210): `last = view_3d(x, 3, channels, n_seqs, ..., offset_cols = n_seq_tokens)` = **the LAST n_conv-1=3 columns of x** (for prefill T=5: current tokens 2,3,4); copied contiguously into the slot cell `[head*row_size + offs]` of size `3*channels`, layout flat = `lag + 3*channel` (lag fastest), matching the past-view read `history[ch*(n_conv-1) + lag]`.
- Final: `out = (cur + ssm + Σ_lags taps) * scale`, `scale = 1/sqrt(2*n_conv+1) = 1/3` (dynamic K/V) or `1/sqrt(n_conv+1) = 1/sqrt5` (post-convs, dynamic=nullptr).
- Dynamic taps: `out[c][t] += tap[head(c)][lag] * x[3 - lag + t][c]` with `head(c) = c / 4` and within-head group index (window reshape `[4,256,T]` = c-major: element (g, h, t) = channel g + 4*h — so tap index = head = c/4, same tap for all 4 channels of a head).

```
IMPLEMENTATION NOTE (convolve)
hist[c][lag] per slot; x[t+k] indexing: k=3→token t itself.
out = (cur + Σ_k w[k]·x[t+k] + Σ_lag tap[c/4][lag]·x[t+3-lag]) * (dynamic?1/3:1/sqrt(5)).
After the pass, hist[c][lag] = x[t=T-3+lag] i.e. the conv inputs (PRE-scaling cur values? NO: x is concat(hist, reshape(cur)) — cur here is the raw GEMV output, before the residual/scale).
```
Caution: `x` stores the **raw input `cur` of convolve** (Kcur/Vcur/wo-out/MoE-out pre-scale), not the scaled output.

## 8. Recurrent state layout (llama-hparams.cpp:208-217; inkling2.cpp:63-79)

```
llama-hparams.cpp:214| if (n_sconv_channels != 0) { 216|   return n_sconv_channels * (n_shortconv_l_cache - 1); }
inkling2.cpp:73| n_channels_max = max over layers of (2*n_embd_k_gqa + 2*n_embd);   // 2*1024 + 2*2304 = 6656
inkling2.cpp:79| hparams.n_sconv_channels = n_channels_max;
```
- `n_embd_r() = 6656 * 3 = 19968` f32 per **cache slot**; there are 48 slots (`get_r_l(il)`, il 0..47) — conv state is PER SLOT (loop slots keep their own), same as KV.
- Channel offsets used by inkling2 (in units of channels within the 6656-channel state; `offset` args in code):
  - K conv history: `[0, 1024)`  (width n_embd_k_gqa = 8*128 = 1024)
  - V conv history: `[1024, 2048)`
  - attn-post-conv history: `[2048, 4352)` (width n_embd = 2304)
  - ffn-post-conv history: `[4352, 6656)`
- Element flat index inside one slot's state: `((channel_offset + c)*(n_conv-1) + lag)`; slot state = rows `[head*row_size...]` in the rs buffer (build_rs loads rows for the active seqs via `s_copy`; §7).

```
IMPLEMENTATION NOTE (rs)
state[slot][4][6656][3] f32, init 0. Index: state[slot][sec][1024..2047 etc][lag]; sec order K,V,attn,ffn; lag 0=oldest.
```

## 9. MoE FFN (build_moe_ffn, llama-graph.cpp:2329-2735) + SITU

Router (verbatim excerpts):
```
2361| logits = build_lora_mm(gate_inp, cur);          // ffn_gate_inp f32 [2304,128] → [128, T]  (gate_inp_b==nullptr for inkling2)
2378| case SIGMOID: probs = ggml_sigmoid(logits);       // sigmoid, not softmax
2398| if (exp_probs_b) selection_probs = ggml_add(probs, exp_probs_b);   // bias [128] broadcast; weights later read from UNBIASED probs
2442| selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);   // [8, T]
2456| weights = ggml_get_rows(ctx0, probs, selected_experts);          // gather UNBIASED probs
2467| if (norm_w) { weights_sum = ggml_sum_rows(weights);              // [1, T] over the 8
2476|   if (hparams.expert_weights_norm_eps > 0) weights_sum = scale_bias(weights_sum, 1, eps);   // inkling2: eps=1e-20
2484|   weights = ggml_div(weights, weights_sum); }
2489| if (w_scale != 0 && != 1) weights = ggml_scale(weights, w_scale);              // ×8.0
2422/2429| (group top-k skipped: n_expert_groups == 1)
```
- `ggml_argsort_top_k` (ggml.c:5358) = `ggml_argsort(a, GGML_SORT_ORDER_DESC)` then view of the first 8 index slots. CPU (`ggml-cpu/ops.cpp:8733-8775`): full `std::sort` DESC with comparator `src[a] > src[b]` on **unstable** std::sort → exact tie-break is not defined by the source; assume no exact f32 ties (cosine tolerance absorbs).
- Fused gate/up (llama-graph.cpp:2534-2546): `gate_up = mul_mat_id(ffn_gate_up_exps [2304, 2304, 128], cur, ids)` → `[2304, 8, T]`;
  ```
  2542| const int64_t n_ff = gate_up->ne[0] / 2;                                   // 1152
  2543| cur(gate) = view_3d(gate_up, n_ff, 8, T, nb[1], nb[2], 0);                 // rows 0..1151  == GATE (first half)
  2545| up        = view_3d(gate_up, n_ff, 8, T, nb[1], nb[2], n_ff*nb[0]);        // rows 1152..2303 == UP
  ```
  ⇒ in the pack's contiguous gate|up expert blob, **gate occupies the first 1152 rows** of 2304.
- SITU (llama-graph.cpp:2603-2616, verbatim):
  ```
  2605| // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
  2610| act = ggml_scale(tanh(ggml_scale(cur, 1/beta)), beta);       // beta = 4.0
  2611| act = ggml_mul(act, ggml_sigmoid(cur));
  2612| if (lb > 0) up = ggml_scale(tanh(ggml_scale(up, 1/lb)), lb); // lb = 25.0
  2615| cur = ggml_mul(act, up);
  ```
  i.e. `f(x) = β·tanh(x/β)·sigmoid(x) · [lb·tanh(u/lb)]`. No separate ggml_situ op; betas from GGUF keys
  `%s.activation.situ_beta` / `%s.activation.situ_linear_beta` (llama-arch.cpp:232-233), read in load_arch_hparams (inkling2.cpp:24-25) → `hparams.situ_beta`, `hparams.situ_linear_beta`.
- down + weighting (2666-2729): `experts = mul_mat_id(ffn_down_exps [1152, 2304, 128], situ_out, ids)` → `[2304, 8, T]`; `experts = ggml_mul(experts, weights)` ([1,8,T] broadcast); aggregate **in slot order**:
  `moe_out = ((view(i=0)+view(i=1)) + view(i=2)) + ... + view(i=7)` (sequential adds, views at `i*experts->nb[1]`).

```
IMPLEMENTATION NOTE (MoE, per token t)
p[e] = sigmoid(dot(gate_inp[e,*], x[t])); sel = top-8 argsort-desc of (p[e]+b[e]); w = p[sel]/(sum(p[sel])+1e-20) * 8;
for k in 0..7 (ascending rank): g = Wgu[sel[k]][0:1152]@x ; u = Wgu[sel[k]][1152:2304]@x ;
  a = 4*tanh(g/4)*sigmoid(g); h = (a*u_lin) with u_lin = 25*tanh(u/25); out += w[k] * (Wdown[sel[k]] @ h);
accumulate k sequentially.
```

## 10. build_output (inkling2.cpp:421-446, verbatim incl. tail)

```
421| void build_output(cur, blocks) {
422|   cur = ggml_reshape_2d(mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
423|   cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);   // rms(eps=1e-6) * weight
425|   if (inp_out_ids) cur = ggml_get_rows(cur, inp_out_ids);                // ONLY out_ids rows BEFORE lm_head
433|   cur = build_lora_mm(model.output, cur);                                 // [201024, n_out] — output has prec F32 acc set (create_tensor time)
435|   if (hparams.f_final_logit_softcapping != 0.0f) {
438|     cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap); }   // logit = 30*tanh(x/30)
442|   cur = ggml_cont(ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));  // keep rows 0..200057
445|   res->t_logits = cur;
```
- Softcap is applied **after get_rows** (only to the returned rows) and **before** the padded-vocab trim.
- `llama_get_logits` returns the last out_ids row; each row is the first `n_vocab_unpadded = 200058` logits of the 201024 lm_head rows (rows 200058..201023 of `output.weight` participate in the GEMV only if they were also in out_ids — they are NOT returned; for a returned row the GEMV still computes all 201024 rows then trims — the engine must compute 201024 dot products and keep 0..200057).

```
IMPLEMENTATION NOTE (tail)
mix(blocks,cur,out_res_query) → rms*w → for each out row t: z[v] = 30*tanh(dot(output_row_v, h[t])/30) for v<201024; logits = z[0:200058].
```

## 11. build_inputs (inkling2.cpp:251-272)

```
252| inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);   // rel extents 1024/1024
256| ggml_tensor * inp_pos = build_inp_pos();               // I32 [n_tokens] = ubatch positions (0-based seq positions)
257| inp_out_ids = build_inp_out_ids();                     // I32 idxs of ubatch tokens with output=1 (last token for single prompt)
264| cur = build_inp_embd(model.tok_embd);                  // gather tok_embd rows for token ids
267| if (ubatch.token) cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
```
- **`tok_norm` IS a required tensor** — inkling2.cpp:96 verbatim: `tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);` (last arg 0 = not optional → llama.cpp throws at model load if `token_embd_norm.weight` is absent from the GGUF). The oracle therefore ran WITH it; if the .m64pack conversion missed it, that is a pack bug (do NOT default to ones — use GGUF key `token_embd_norm.weight`, `{n_embd}` f32). `build_norm` (llama-graph.cpp:1943-1962) is NOT identity for nullptr mw: it always runs `ggml_rms_norm(cur, f_norm_rms_eps)` and applies `ggml_mul` only `if (mw || mb)`. So embeddings = `tok_norm_w * h_raw * rsqrt(mean(h_raw^2)+1e-6)`, gated by `if (ubatch.token)` (text tokens only).
- Positions: `inp_pos[i] = ubatch.pos[i]` — plain sequence position; every cache slot of the loop stores K/V at the SAME position p for token p (per-slot KV via `get_k/get_v` of slot il — build_attn stores into the slot's own cache view selected by `il`), so all 48 slots cache token p at pos p from their own forward pass.
- tau (§3.8) uses `cast_f32(inp_pos)+1`.

```
IMPLEMENTATION NOTE (inputs)
h[t] = w_toknorm * h_raw[t] * rsqrt(mean(h_raw^2)+1e-6); h_raw = emb[token[t]]; pos[t] = t for a fresh prompt.
```

## 12. Loop graph driver (inkling2-loop.cpp:76-115) — verbatim

```
 81| auto * cur = build_inputs();  std::vector<ggml_tensor *> blocks;
 83| for (il < start) cur = build_layer(il, il, cur, blocks);              // prelude 0..5
 88| auto * prelude = cur;  const auto prelude_blocks = blocks;            // [B0]
 90| for (pass = 0; pass < n_passes; ++pass) {
 94|   if (pass > 0) {
 95|     bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
 96|     bank_out = cur;  cur = prelude;  blocks = prelude_blocks; }
100|   for (ip = start; ip < end; ++ip) {
101|     const int il = pass == 0 ? ip : m.n_phys + (pass-1)*(end-start) + (ip-start);
102|     if (pass == 0) { cur = build_layer(il, ip, cur, blocks); continue; }
106|     const auto & loop = m.loop_layers[ip - start];
107|     cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, loop.msg_query), &loop); }
111| for (il = end; il < n_phys) cur = build_layer(il, il, cur, blocks);    // coda 18..23
114| build_output(cur, blocks);
```
- `mix(bank, bank_out, msg_query)` → `[n_embd, 4, T]` (4 queries); dim1 = ctrl axis (§2); mix dim1 states = `[bank_out, B12^prev]` (bank_out LAST = newest).
- `loop_layers[ip-start]` per physical core layer: `msg_query {n_embd, LOOP_CTRL_COUNT=4}`, `ctrl_down {2*n_embd, rank=64}`, `ctrl_gain/ctrl_forget {64, n_embd}` (inkling2-loop.cpp:56-61). Names: `blk.N.loop_{q,k,v,ffn}_ctrl_down.weight` etc.
- Loop slots share PHYSICAL weights (`layers[il] = layers[layer_phys(il)]`, :65-70) but have their own KV + rs slot.
- messages for Q/K/V: `message(LOOP_CTRL_Q+j)`, FFN: `message(LOOP_CTRL_FFN)` (enum 0..3 = q,k,v,ffn).

## 13. `varying`

`LLM_KV_LOOP_VARYING` false in this GGUF ⇒ `n_passes = 3` fixed for every token; validation requires `n_passes >= (varying ? 1 : 2)` and `core_start%block == core_end%block == 0`, `core_start >= 1`, `core_end <= n_phys`. Nothing dynamic.

## 14. KV per slot; 5-token example

- `is_swa(il)`: loop slots inherit (hparams arrays slot-expanded, e.g. `is_swa[24+...] = is_swa[phys]`). Physical swa pattern: layers 20-23 (coda) only → **no loop slot and no core slot is swa**; prelude 0-5 and core 6-17 slots are FULL attention (rel extent 1024, no window mask). Coda slots 18-23 are SWA (window mask 1024, extent_swa 1024).
- Causal mask keep 0 / drop -inf per cell; empty cells dropped (so full-attention slots with a padded n_kv simply contribute -inf columns; their rel bucket 1024 = zero bias column, irrelevant).
- Sequence length 5, evaluating token at seq pos 4 (kv cells hold positions 0..4):
  (a) full-attn slot: rel_pos row = `[4,3,2,1,0, 1024×(n_kv-5)]` for kv columns 0..4 (positions 0..4) — distances 4,3,2,1,0; remaining columns empty→bucket 1024 + -inf mask.
  (b) swa slot (hypothetically the same cache state; coda behavior): identical row (window 1024 > 4 masks nothing); only difference would be for distances > 1024.

```
IMPLEMENTATION NOTE (engine mapping)
Test seq of 5: slots 0-17 (prelude+core, all full) and 24-47 (loop core, full): mask = causal only; rel buckets {4,3,2,1,0} + 1024 for pad.
Coda slots 18-23 (swa): causal + window mask (no effect at len<=1024); same rel buckets via self_rel_pos_swa (extent_swa=1024).
```

## Appendix A — attention head→kv-head mapping (VERIFIED)

`build_attn_mha` permutes q to `[d, tq, h=24, ns]`, k/v to `[d, n_kv, h_kv=8, ns]` and relies on `ggml_mul_mat` head-broadcast. CPU mapping (ggml-cpu.c:1341 + :1256): `r2 = ne12/ne02 = 3`, `kv_head = result_head / r2` (integer FLOOR division). So query head `h` uses kv head `h / 3` (standard GQA: heads 0-2→0, 3-5→1, …, 21-23→7), NOT `h % 8`.

## Appendix B — FA-vs-dense fidelity knobs

Default CLI (`--ngl 0`, no --flash-attn flag): `flash_attn=true, auto_fa=true` → probe → enabled (CPU "supports" the op). Oracle numbers then include: Q/K/V cast F32→F16, bias+mask F16 (`-inf` and 0 unaffected; bias values rounded to f16 ~1e-3 rel), online-block softmax in F32 with sink=0 correction. Engine may implement the dense form of §5; expected cosine ~0.9999. If parity < 0.9995, add the F16 casts of Q/K/V/bias first.
