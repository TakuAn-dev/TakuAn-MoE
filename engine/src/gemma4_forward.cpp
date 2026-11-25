// gemma4_forward.cpp — Gemma 4 (arch "gemma4") resident CPU forward oracle.
//
// Semantics pinned 1:1 to the llama.cpp oracle build at e300919cc704ddde3a8a8346a8714f57eeeee71b:
//   src/models/gemma4.cpp   per-layer sliding_window_pattern, per-layer head_count_kv,
//                           full-attention tied-V layers (V = K BEFORE K-norm/RoPE),
//                           per-head Q/K RMSNorm (K-norm weighted, V-norm unweighted),
//                           NEOX RoPE (full layers divide angles by rope_freqs),
//                           f_attention_scale = 1.0 (NO 1/sqrt(d)), router input
//                           rms_norm * (1/sqrt(n_embd)) * ffn_gate_inp.scale,
//                           parallel shared-MLP + routed MoE, post_ffw_norm_1/2 +
//                           post_ffw_norm, layer_output_scale, tied LM head fallback,
//                           final softcap cap * tanh(z / cap).
//   llama-arch / ggml       softmax denominators accumulate in double, f32xf32
//                           dots accumulate in double (ggml_vec_dot_f32), quantized
//                           dots dequant-then-dot; window keeps dist < sliding_window.
//
// Numerics: f32 everywhere with -ffp-contract=off; fixed sequential reduction
// orders only (single-threaded oracle), so results are bitwise deterministic
// across runs and between step()/prefill() replays.
#include "gemma4_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

#include "dequant.hpp"
#include "pack_reader.hpp"

namespace m64 {

namespace {
inline float gelu_tanh(float x) {
    // ggml GELU_EXACT-tanh form: 0.5x(1 + tanh(c(x + 0.044715 x^3)))
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}
}  // namespace

struct Gemma4Forward::Impl {
    PackReader * pack_ = nullptr;  // set by Gemma4Forward::init before Impl::init
    // ---- static config (from pack hparams / tensor shapes) ----
    int64_t n_layer = 0, n_embd = 0, n_ff = 0, n_ff_exp = 0;
    int64_t n_expert = 0, top_k = 0, n_head = 0;
    int64_t key_len = 0, val_len = 0, key_len_swa = 0, val_len_swa = 0;
    int64_t rope_dim = 0, rope_dim_swa = 0;
    int64_t n_swa = 0;
    float freq_base_full = 0, freq_base_swa = 0, eps = 0, cap = 0;
    int ctx = 0;
    std::vector<bool> is_swa;                 // per layer
    std::vector<int64_t> n_kv;                // per layer
    std::vector<int64_t> hd_k, hd_v, n_rot;   // per layer
    int64_t vocab_rows = 0, vocab_eff = 0;
    int64_t pos = 0;
    size_t mapped_bytes = 0;

    // ---- pack tensors ----
    const TensorInfo * tok_embd = nullptr, * output = nullptr, * output_norm = nullptr;
    const TensorInfo * rope_freqs_root = nullptr;
    struct LT {
        const TensorInfo *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
        const TensorInfo *q_norm = nullptr, *k_norm = nullptr, *rope_freqs = nullptr;
        const TensorInfo *attn_norm = nullptr, *post_attn_norm = nullptr;
        const TensorInfo *ffn_norm = nullptr, *post_ffw_norm_1 = nullptr;
        const TensorInfo *pre_ffw_norm_2 = nullptr, *post_ffw_norm_2 = nullptr;
        const TensorInfo *post_ffw_norm = nullptr, *layer_scale = nullptr;
        const TensorInfo *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
        const TensorInfo *router_w = nullptr, *router_scale = nullptr;
        const TensorInfo *down_scale = nullptr, * experts = nullptr;
    };
    std::vector<LT> lt;

    // ---- KV caches (full-ctx store per layer; masking does the windowing) ----
    std::vector<std::vector<float>> kv_k, kv_v;  // [layer][pos * n_kv*hd + kv*hd + d]

    // ---- activation scratch (all preallocated; no allocs in forward) ----
    std::vector<float> inp, cur, normed, attn_out, mlp, moe, expert_acc;
    std::vector<float> q, k, v, attn, scores, work_a, work_b, router, scratch;
    std::vector<float> logits;
    std::vector<int64_t> sel;
    std::vector<float> sel_w;

    bool init(PackReader & pack, int context_len, std::string & err) {
        if (pack.header().arch != "gemma4") {
            err = "gemma4_forward: arch '" + pack.header().arch + "' is not 'gemma4'";
            return false;
        }
        if (context_len <= 0) { err = "gemma4_forward: context_len must be > 0"; return false; }
        try {
            n_layer = pack.hparam_i64("gemma4.block_count");
            n_embd = pack.hparam_i64("gemma4.embedding_length");
            n_ff = pack.hparam_i64("gemma4.feed_forward_length");
            n_ff_exp = pack.hparam_i64("gemma4.expert_feed_forward_length");
            n_expert = pack.hparam_i64("gemma4.expert_count");
            top_k = pack.hparam_i64("gemma4.expert_used_count");
            n_head = pack.hparam_i64("gemma4.attention.head_count");
            key_len = pack.hparam_i64("gemma4.attention.key_length");
            val_len = pack.hparam_i64("gemma4.attention.value_length");
            key_len_swa = pack.hparam_i64("gemma4.attention.key_length_swa", key_len);
            val_len_swa = pack.hparam_i64("gemma4.attention.value_length_swa", val_len);
            rope_dim = pack.hparam_i64("gemma4.rope.dimension_count", key_len);
            rope_dim_swa = pack.hparam_i64("gemma4.rope.dimension_count_swa", rope_dim);
            freq_base_full = (float)pack.hparam_f64("gemma4.rope.freq_base", 1e6);
            freq_base_swa = (float)pack.hparam_f64("gemma4.rope.freq_base_swa", 1e4);
            n_swa = pack.hparam_i64("gemma4.attention.sliding_window", 0);
            eps = (float)pack.hparam_f64("gemma4.attention.layer_norm_rms_epsilon", 1e-6);
            cap = (float)pack.hparam_f64("gemma4.final_logit_softcapping", 0.0);
            if (n_layer <= 0 || n_embd <= 0 || n_head <= 0) {
                err = "gemma4_forward: bad core hparams"; return false;
            }
            std::vector<bool> pat = pack.hparam_bool_arr("gemma4.attention.sliding_window_pattern");
            if ((int64_t)pat.size() != n_layer) {
                err = "gemma4_forward: sliding_window_pattern size != block_count"; return false;
            }
            std::vector<int64_t> kv = pack.hparam_i64_arr("gemma4.attention.head_count_kv");
            if (kv.empty()) kv.assign(n_layer, pack.hparam_i64("gemma4.attention.head_count_kv", 1));
            if ((int64_t)kv.size() == 1) kv.resize(n_layer, kv[0]);
            if ((int64_t)kv.size() != n_layer) {
                err = "gemma4_forward: head_count_kv size != block_count"; return false;
            }
            is_swa = pat; n_kv = kv;
            hd_k.resize(n_layer); hd_v.resize(n_layer); n_rot.resize(n_layer);
            for (int64_t il = 0; il < n_layer; il++) {
                hd_k[il] = is_swa[il] ? key_len_swa : key_len;
                hd_v[il] = is_swa[il] ? val_len_swa : val_len;
                n_rot[il] = is_swa[il] ? rope_dim_swa : rope_dim;
                if (n_kv[il] <= 0 || n_head % n_kv[il] != 0) {
                    err = "gemma4_forward: bad head_count_kv on layer " + std::to_string(il); return false;
                }
            }
        } catch (const std::exception & e) { err = std::string("gemma4_forward: ") + e.what(); return false; }

        tok_embd = pack.find("token_embd.weight");
        output = pack.find("output.weight");  // optional: tied LM head when absent
        output_norm = pack.find("output_norm.weight");
        rope_freqs_root = pack.find("rope_freqs.weight");
        if (!tok_embd || !output_norm) { err = "gemma4_forward: missing token_embd/output_norm"; return false; }
        lt.assign(n_layer, LT());
        std::string miss;
        auto req = [&](const TensorInfo *& dst, const std::string & name) {
            dst = pack.find(name);
            if (!dst && miss.empty()) miss = name;
        };
        for (int64_t il = 0; il < n_layer; il++) {
            LT & t = lt[il];
            const std::string b = "blk." + std::to_string(il) + ".";
            req(t.wq, b + "attn_q.weight");
            req(t.wk, b + "attn_k.weight");
            t.wv = pack.find(b + "attn_v.weight");  // optional: tied-V layers (V = K)
            req(t.wo, b + "attn_output.weight");
            req(t.q_norm, b + "attn_q_norm.weight");
            req(t.k_norm, b + "attn_k_norm.weight");
            t.rope_freqs = pack.find(b + "rope_freqs.weight");
            req(t.attn_norm, b + "attn_norm.weight");
            req(t.post_attn_norm, b + "post_attention_norm.weight");
            req(t.ffn_norm, b + "ffn_norm.weight");
            req(t.post_ffw_norm_1, b + "post_ffw_norm_1.weight");
            req(t.pre_ffw_norm_2, b + "pre_ffw_norm_2.weight");
            req(t.post_ffw_norm_2, b + "post_ffw_norm_2.weight");
            req(t.post_ffw_norm, b + "post_ffw_norm.weight");
            t.layer_scale = pack.find(b + "layer_output_scale.weight");  // optional
            req(t.ffn_gate, b + "ffn_gate.weight");
            req(t.ffn_up, b + "ffn_up.weight");
            req(t.ffn_down, b + "ffn_down.weight");
            t.router_w = pack.find(b + "ffn_gate_inp.weight");  // optional: dense-only layer
            t.router_scale = pack.find(b + "ffn_gate_inp.scale");
            t.down_scale = pack.find(b + "ffn_down_exps.scale");
            t.experts = pack.find(b + "ffn_experts.weight");
            if (t.router_w && (!t.experts || n_expert <= 0 || top_k <= 0 || top_k > n_expert || n_ff_exp <= 0)) {
                err = "gemma4_forward: router present but MoE hparams/tensors missing on layer " + std::to_string(il);
                return false;
            }
        }
        if (!miss.empty()) { err = "gemma4_forward: missing tensor '" + miss + "'"; return false; }

        vocab_rows = (output ? output : tok_embd)->shape[0];
        vocab_eff = pack.hparam_i64("gemma4.vocab_size_unpadded", vocab_rows);
        if (vocab_eff <= 0 || vocab_eff > vocab_rows) {
            err = "gemma4_forward: bad vocab sizes"; return false;
        }
        // sanity: embedding rows must cover the effective vocab
        if ((int64_t)tok_embd->shape.size() < 2 || tok_embd->shape[1] != n_embd ||
            tok_embd->shape[0] < vocab_eff) {
            err = "gemma4_forward: token_embd shape mismatch"; return false;
        }
        for (int64_t il = 0; il < n_layer; il++) {
            const LT & t = lt[il];
            if (t.wq->shape[0] != n_head * hd_k[il] || t.wq->shape[1] != n_embd ||
                t.wk->shape[0] != n_kv[il] * hd_k[il] ||
                (t.wv && t.wv->shape[0] != n_kv[il] * hd_v[il]) ||
                t.wo->shape[0] != n_embd || t.wo->shape[1] != n_head * hd_v[il]) {
                err = "gemma4_forward: attention shape mismatch on layer " + std::to_string(il);
                return false;
            }
        }

        ctx = context_len;
        // dequant scratch row: widest GEMV input row across all ops
        int64_t wide = std::max<int64_t>({n_embd, n_ff, n_ff_exp, n_head * val_len, n_head * val_len_swa});
        scratch.resize(wide);
        inp.assign(n_embd, 0.f); cur.assign(n_embd, 0.f); normed.assign(n_embd, 0.f);
        attn_out.assign(n_embd, 0.f); mlp.assign(n_embd, 0.f); moe.assign(n_embd, 0.f);
        expert_acc.assign(n_embd, 0.f);
        q.assign(n_head * std::max(key_len, key_len_swa), 0.f);
        {
            int64_t maxkv = *std::max_element(n_kv.begin(), n_kv.end());
            int64_t maxhk = std::max(key_len, key_len_swa), maxhv = std::max(val_len, val_len_swa);
            k.assign(maxkv * maxhk, 0.f);
            v.assign(maxkv * maxhv, 0.f);
            attn.assign(n_head * maxhv, 0.f);
            work_a.assign(std::max(n_ff, n_ff_exp), 0.f);
            work_b.assign(std::max(n_ff, n_ff_exp), 0.f);
        }
        scores.assign(ctx, 0.f);
        router.assign(std::max(n_expert, (int64_t)1), 0.f);
        sel.assign(std::max(top_k, (int64_t)1), 0);
        sel_w.assign(std::max(top_k, (int64_t)1), 0.f);
        logits.assign(vocab_rows, 0.f);
        kv_k.resize(n_layer); kv_v.resize(n_layer);
        for (int64_t il = 0; il < n_layer; il++) {
            kv_k[il].assign((size_t)ctx * n_kv[il] * hd_k[il], 0.f);
            kv_v[il].assign((size_t)ctx * n_kv[il] * hd_v[il], 0.f);
        }
        pos = 0;
        return true;
    }

    void reset() {
        for (auto & b : kv_k) std::fill(b.begin(), b.end(), 0.f);
        for (auto & b : kv_v) std::fill(b.begin(), b.end(), 0.f);
        pos = 0;
    }

    size_t resident_bytes() const {
        size_t bytes = mapped_bytes;
        auto acc = [&bytes](const std::vector<float> & b) { bytes += b.size() * sizeof(float); };
        for (const auto * b : {&inp, &cur, &normed, &attn_out, &mlp, &moe, &expert_acc,
                               &q, &k, &v, &attn, &work_a, &work_b, &router, &scratch, &logits}) acc(*b);
        for (auto & b : kv_k) bytes += b.size() * sizeof(float);
        for (auto & b : kv_v) bytes += b.size() * sizeof(float);
        return bytes;
    }

    // ---- tensor dot helpers (weights dequantized per row into scratch) ----
    float dot_row(DType dt, const uint8_t * row, int64_t nin, const float * x) {
        if (dt == DType::F32) {
            const float * w = (const float *)row;
            double acc = 0.0;
            for (int64_t j = 0; j < nin; j++) acc += (double)w[j] * (double)x[j];
            return (float)acc;
        }
        dequant_row(dt, row, scratch.data(), nin);
        float acc = 0.f;
        for (int64_t j = 0; j < nin; j++) acc = fmaf(scratch[j], x[j], acc);
        return acc;
    }
    void gemv(const TensorInfo * t, int64_t nout, int64_t nin, const float * x, float * y) {
        int be = dtype_block_elems(t->dtype), ts = dtype_type_size(t->dtype);
        const uint8_t * base = (const uint8_t *)pack_->ptr(*t);
        int64_t rbytes = nin / be * (int64_t)ts;
        for (int64_t r = 0; r < nout; r++) y[r] = dot_row(t->dtype, base + (int64_t)r * rbytes, nin, x);
    }
    // expert part GEMV: rows of one (gate|up|down) part inside an expert record
    void gemv_part(DType dt, const uint8_t * part, int64_t nout, int64_t nin, const float * x, float * y) {
        int be = dtype_block_elems(dt), ts = dtype_type_size(dt);
        int64_t rbytes = nin / be * (int64_t)ts;
        for (int64_t r = 0; r < nout; r++) y[r] = dot_row(dt, part + (int64_t)r * rbytes, nin, x);
    }

    // ---- norms / rope / softmax (fixed sequential order) ----
    void rms_norm(const float * x, const float * w, float * dst, int64_t n) {
        double ssq = 0.0;
        for (int64_t i = 0; i < n; i++) ssq += (double)x[i] * (double)x[i];
        float scale = 1.0f / sqrtf((float)(ssq / (double)n) + eps);
        if (w) for (int64_t i = 0; i < n; i++) dst[i] = w[i] * (x[i] * scale);
        else   for (int64_t i = 0; i < n; i++) dst[i] = x[i] * scale;
    }
    // NEOX RoPE: pairs (j, j + n_rot/2); full layers divide by rope_freqs[j]
    void rope(float * h, int64_t n_rot_, float freq_base, const float * ff) {
        int64_t half = n_rot_ / 2;
        for (int64_t j = 0; j < half; j++) {
            float theta = (float)pos * powf(freq_base, -2.0f * (float)j / (float)n_rot_);
            float angle;
            if (ff) {
                float f = ff[j];
                if (!std::isfinite(f) || f == 0.0f) continue;  // guard: no rotation
                angle = theta / f;
            } else {
                angle = theta;
            }
            float c = cosf(angle), s = sinf(angle);
            float x0 = h[j], x1 = h[j + half];
            h[j] = x0 * c - x1 * s;
            h[j + half] = x0 * s + x1 * c;
        }
    }
    void softmax(float * y, int64_t n) {
        float m = y[0];
        for (int64_t i = 1; i < n; i++) if (y[i] > m) m = y[i];
        double sum = 0.0;
        for (int64_t i = 0; i < n; i++) { y[i] = expf(y[i] - m); sum += (double)y[i]; }
        float inv = (float)(1.0 / sum);
        for (int64_t i = 0; i < n; i++) y[i] *= inv;
    }

    // ---- one token through the whole stack ----
    StepResult forward_token(int32_t token, bool want_logits) {
        StepResult r{nullptr, 0, -1};
        if (token < 0 || (int64_t)token >= vocab_eff) return r;
        if (pos >= ctx) reset();

        // input embedding scaled by sqrt(n_embd)
        gemv_row(tok_embd, token, n_embd, inp.data());
        { float s = sqrtf((float)n_embd); for (int64_t i = 0; i < n_embd; i++) inp[i] *= s; }

        for (int64_t il = 0; il < n_layer; il++) layer(il);

        rms_norm(inp.data(), f32_row(output_norm), cur.data(), n_embd);
        if (want_logits) {
            gemv(output ? output : tok_embd, vocab_rows, n_embd, cur.data(), logits.data());
            if (cap > 0.0f)
                for (int64_t i = 0; i < vocab_rows; i++) logits[i] = tanhf(logits[i] / cap) * cap;
            r.logits = logits.data();
            r.n_logits = vocab_eff;
            int32_t best = 0;
            for (int64_t i = 1; i < vocab_eff; i++) if (logits[i] > logits[best]) best = (int32_t)i;
            r.argmax = best;
        }
        pos++;
        return r;
    }

private:
    // f32 weight vector (norms/scales); dequantize everything else per row.
    const float * f32_row(const TensorInfo * t) {
        return (const float *)pack_->ptr(*t);
    }
    void gemv_row(const TensorInfo * t, int64_t row, int64_t nin, float * dst) {
        int be = dtype_block_elems(t->dtype), ts = dtype_type_size(t->dtype);
        const uint8_t * base = (const uint8_t *)pack_->ptr(*t) + (row * nin / be) * (int64_t)ts;
        if (t->dtype == DType::F32) {
            const float * w = (const float *)base;
            for (int64_t j = 0; j < nin; j++) dst[j] = w[j];
            return;
        }
        dequant_row(t->dtype, base, dst, nin);
    }

    void layer(int64_t il) {
        const LT & t = lt[il];
        const int64_t hk = hd_k[il], hv = hd_v[il], nk = n_kv[il], nrot = n_rot[il];
        const int64_t gqa = n_head / nk;
        const float * qn = f32_row(t.q_norm), * kn = f32_row(t.k_norm);

        // ----- attention -----
        rms_norm(inp.data(), f32_row(t.attn_norm), normed.data(), n_embd);
        gemv(t.wq, n_head * hk, n_embd, normed.data(), q.data());
        for (int64_t h = 0; h < n_head; h++) rms_norm(q.data() + h * hk, qn, q.data() + h * hk, hk);
        const float * ff = nullptr;
        if (!is_swa[il]) {
            const TensorInfo * rft = t.rope_freqs ? t.rope_freqs : rope_freqs_root;
            if (rft) ff = f32_row(rft);  // full layers: angles divided by rope_freqs
        }
        float freq_base = is_swa[il] ? freq_base_swa : freq_base_full;
        for (int64_t h = 0; h < n_head; h++) rope(q.data() + h * hk, nrot, freq_base, ff);
        gemv(t.wk, nk * hk, n_embd, normed.data(), k.data());
        if (t.wv) gemv(t.wv, nk * hv, n_embd, normed.data(), v.data());
        else std::memcpy(v.data(), k.data(), (size_t)nk * hv * sizeof(float));  // tied V = K (pre-norm!)
        for (int64_t kv = 0; kv < nk; kv++) {
            rms_norm(k.data() + kv * hk, kn, k.data() + kv * hk, hk);  // K: weighted norm
            rope(k.data() + kv * hk, nrot, freq_base, ff);
            rms_norm(v.data() + kv * hv, nullptr, v.data() + kv * hv, hv);  // V: UNWEIGHTED norm
        }
        float * kbase = kv_k[il].data();
        float * vbase = kv_v[il].data();
        std::memcpy(kbase + (size_t)pos * nk * hk, k.data(), (size_t)nk * hk * sizeof(float));
        std::memcpy(vbase + (size_t)pos * nk * hv, v.data(), (size_t)nk * hv * sizeof(float));

        const int64_t j0 = is_swa[il] && n_swa > 0 ? std::max((int64_t)0, pos - n_swa + 1) : 0;
        for (int64_t h = 0; h < n_head; h++) {
            const float * qh = q.data() + h * hk;
            const int64_t kv = h / gqa;
            for (int64_t j = j0; j <= pos; j++) {
                const float * kj = kbase + (size_t)j * nk * hk + kv * hk;
                float s = 0.f;
                for (int64_t d = 0; d < hk; d++) s = fmaf(qh[d], kj[d], s);
                // f_attention_scale = 1.0: NO 1/sqrt(dk) scaling
                scores[j - j0] = s;
            }
            softmax(scores.data(), pos - j0 + 1);
            for (int64_t d = 0; d < hv; d++) {
                float o = 0.f;
                for (int64_t j = j0; j <= pos; j++) o = fmaf(scores[j - j0], vbase[(size_t)j * nk * hv + kv * hv + d], o);
                attn[h * hv + d] = o;
            }
        }
        gemv(t.wo, n_embd, n_head * hv, attn.data(), cur.data());
        rms_norm(cur.data(), f32_row(t.post_attn_norm), cur.data(), n_embd);
        for (int64_t i = 0; i < n_embd; i++) attn_out[i] = cur[i] + inp[i];

        // ----- parallel shared MLP + routed MoE FFN -----
        rms_norm(attn_out.data(), f32_row(t.ffn_norm), normed.data(), n_embd);
        gemv(t.ffn_gate, n_ff, n_embd, normed.data(), work_a.data());
        gemv(t.ffn_up, n_ff, n_embd, normed.data(), work_b.data());
        for (int64_t i = 0; i < n_ff; i++) work_a[i] = gelu_tanh(work_a[i]) * work_b[i];
        gemv(t.ffn_down, n_embd, n_ff, work_a.data(), mlp.data());
        rms_norm(mlp.data(), f32_row(t.post_ffw_norm_1), mlp.data(), n_embd);

        if (t.router_w) {
            // router on attn_out: unweighted RMSNorm * 1/sqrt(n_embd) * ffn_gate_inp.scale
            rms_norm(attn_out.data(), nullptr, normed.data(), n_embd);
            const float inv_sqrt = 1.0f / sqrtf((float)n_embd);
            const float * rs = t.router_scale ? f32_row(t.router_scale) : nullptr;
            for (int64_t i = 0; i < n_embd; i++) normed[i] = (normed[i] * inv_sqrt) * (rs ? rs[i] : 1.0f);
            gemv(t.router_w, n_expert, n_embd, normed.data(), router.data());
            softmax(router.data(), n_expert);
            // top-k selection (deterministic sequential argmax), renormalize to sum 1
            double wsum = 0.0;
            for (int64_t e = 0; e < top_k; e++) {
                int64_t best = -1; float bv = -INFINITY;
                for (int64_t i = 0; i < n_expert; i++) {
                    bool taken = false;
                    for (int64_t qe = 0; qe < e; qe++) if (sel[qe] == i) { taken = true; break; }
                    if (!taken && router[i] > bv) { bv = router[i]; best = i; }
                }
                sel[e] = best; sel_w[e] = router[best]; wsum += (double)router[best];
            }
            float inv = (float)(1.0 / (wsum + 1e-20));
            const float * ds = t.down_scale ? f32_row(t.down_scale) : nullptr;
            for (int64_t e = 0; e < top_k; e++) sel_w[e] = sel_w[e] * inv * (ds ? ds[sel[e]] : 1.0f);

            rms_norm(attn_out.data(), f32_row(t.pre_ffw_norm_2), normed.data(), n_embd);
            std::fill(expert_acc.begin(), expert_acc.end(), 0.f);
            const TensorInfo * ex = t.experts;
            int64_t pg = ex->part_bytes((n_ff_exp * n_embd / 256) * 144, 0);
            int64_t pu = ex->part_bytes((n_ff_exp * n_embd / 256) * 144, 1);
            DType dg = ex->part_dtype(ex->dtype, 0), du = ex->part_dtype(ex->dtype, 1), dd = ex->part_dtype(ex->dtype, 2);
            const uint8_t * exbase = (const uint8_t *)pack_->ptr(*ex);
            for (int64_t e = 0; e < top_k; e++) {
                const uint8_t * eb = exbase + sel[e] * ex->expert_stride;
                gemv_part(dg, eb, n_ff_exp, n_embd, normed.data(), work_a.data());
                gemv_part(du, eb + pg, n_ff_exp, n_embd, normed.data(), work_b.data());
                for (int64_t i = 0; i < n_ff_exp; i++) work_a[i] = gelu_tanh(work_a[i]) * work_b[i];
                gemv_part(dd, eb + pg + pu, n_embd, n_ff_exp, work_a.data(), cur.data());
                float w = sel_w[e];
                for (int64_t i = 0; i < n_embd; i++) expert_acc[i] = fmaf(w, cur[i], expert_acc[i]);
            }
            rms_norm(expert_acc.data(), f32_row(t.post_ffw_norm_2), moe.data(), n_embd);
            for (int64_t i = 0; i < n_embd; i++) cur[i] = mlp[i] + moe[i];
        } else {
            for (int64_t i = 0; i < n_embd; i++) cur[i] = mlp[i];
        }
        rms_norm(cur.data(), f32_row(t.post_ffw_norm), cur.data(), n_embd);
        for (int64_t i = 0; i < n_embd; i++) inp[i] = cur[i] + attn_out[i];
        if (t.layer_scale) { float s = f32_row(t.layer_scale)[0]; for (int64_t i = 0; i < n_embd; i++) inp[i] *= s; }
    }
};

Gemma4Forward::Gemma4Forward() = default;
Gemma4Forward::~Gemma4Forward() = default;

bool Gemma4Forward::init(PackReader & pack, int context_len, std::string & err) {
    impl_ = std::make_unique<Impl>();
    impl_->pack_ = &pack;
    impl_->mapped_bytes = (size_t)pack.file_size();
    return impl_->init(pack, context_len, err);
}

StepResult Gemma4Forward::step(int32_t token) { return impl_->forward_token(token, true); }

StepResult Gemma4Forward::prefill(const std::vector<int32_t> & prompt) {
    StepResult r{nullptr, 0, -1};
    for (size_t i = 0; i + 1 < prompt.size(); i++) r = impl_->forward_token(prompt[i], false);
    r = impl_->forward_token(prompt.back(), true);
    return r;
}

void Gemma4Forward::reset() { if (impl_) impl_->reset(); }
int64_t Gemma4Forward::vocab_effective() const { return impl_->vocab_eff; }
int64_t Gemma4Forward::vocab_logit_rows() const { return impl_->vocab_rows; }
size_t Gemma4Forward::resident_bytes() const { return impl_ ? impl_->resident_bytes() : 0; }

}  // namespace m64
