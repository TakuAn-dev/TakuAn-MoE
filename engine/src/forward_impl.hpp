// forward_impl.hpp — InklingForward::Impl pimpl body.
//
// The body must be visible in EVERY translation unit that includes forward.hpp:
// Apple clang 21 + libc++ instantiate ~unique_ptr<Incomplete> eagerly at the
// closing of class InklingForward unless the point of instantiation (end of TU)
// sees a complete Impl. So main.cpp includes this header too.
#pragma once
#include "forward.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "dequant.hpp"

namespace m64 {
namespace {

// fixed constants, all cross-checked against pack hparams in Impl::init()
constexpr int NE    = 2304;   // embedding_length
constexpr int NH    = 24;     // attention.head_count
constexpr int NKV   = 8;      // attention.head_count_kv
constexpr int DH    = 128;    // key_length == value_length
constexpr int KVD   = 1024;   // n_embd_k_gqa = NKV*DH
constexpr int NREL  = 16;     // attention.relative_length
constexpr int EXT   = 1024;   // attention.relative_extent (== sliding_window)
constexpr int NEXP  = 128;    // expert_count
constexpr int TOPK  = 8;      // expert_used_count
constexpr int NF    = 1152;   // expert_feed_forward_length
constexpr int NCONV = 4;      // shortconv.l_cache
constexpr int HR    = NCONV - 1;
constexpr int NPHYS = 24;     // block_count
constexpr int NSLOT = 48;     // n_phys + (loop.count-1)*(loop.end-loop.start)
constexpr int MAXS  = NSLOT;  // max mix states
constexpr int CH    = 2 * KVD + 2 * NE;  // 6656 sconv channels
constexpr int PART_BYTES = (NF * NE / 32) * 34;  // bytes per expert gate|up|down part (q8_0)

inline int slot_phys(int il) {  // inkling2-loop.cpp layer_phys()
    return il < NPHYS ? il : 6 + (il - NPHYS) % 12;
}

inline float f16r(float x) {  // GGML_F32_TO_F16 round-trip used by the FA path
    __fp16 h = (__fp16)x;
    return (float)h;
}

inline float ksigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// persistent fixed-partition row pool. Each output row is computed exactly
// once; partition boundaries are deterministic and rows are independent.
// ---------------------------------------------------------------------------
struct Pool {
    int nt = 1;
    std::vector<std::thread> th;
    std::mutex m;
    std::condition_variable cv, cvd;
    std::function<void(int)> part_fn;
    uint64_t gen = 0;
    std::atomic<int> done{0};
    bool quit = false;

    void start() {
        nt = (int)std::thread::hardware_concurrency();
        if (nt < 1) nt = 1;
        if (nt > 16) nt = 16;
        for (int i = 1; i < nt; i++) th.emplace_back([this, i] { worker(i); });
    }
    void stop() {
        {
            std::lock_guard<std::mutex> g(m);
            quit = true;
        }
        cv.notify_all();
        for (auto & t : th) t.join();
        th.clear();
        nt = 1;
    }
    void run(int rows, const std::function<void(int, int)> & fn) {
        if (rows <= 0) return;
        if (nt <= 1) { fn(0, rows); return; }
        {
            std::lock_guard<std::mutex> g(m);
            part_fn = [&](int p) {
                long long lo = (long long)rows * p / nt, hi = (long long)rows * (p + 1) / nt;
                if (lo < hi) fn((int)lo, (int)hi);
            };
            done.store(0, std::memory_order_relaxed);
            ++gen;
        }
        cv.notify_all();
        part_fn(0);  // caller owns partition 0
        if (done.fetch_add(1, std::memory_order_acq_rel) + 1 == nt) {
            std::lock_guard<std::mutex> g2(m);
            cvd.notify_all();
        }
        std::unique_lock<std::mutex> g(m);
        cvd.wait(g, [&] { return done.load(std::memory_order_acquire) == nt; });
    }

private:
    void worker(int w) {
        uint64_t mygen = 0;
        for (;;) {
            std::unique_lock<std::mutex> g(m);
            cv.wait(g, [&] { return quit || gen != mygen; });
            if (quit) return;
            mygen = gen;
            g.unlock();  // must not hold m during work or at the cvd notify
            part_fn(w);
            if (done.fetch_add(1, std::memory_order_acq_rel) + 1 == nt) {
                std::lock_guard<std::mutex> g2(m);
                cvd.notify_all();
            }
        }
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct InklingForward::Impl {
    const PackReader * pack = nullptr;
    int ctx = 0;
    int64_t mapped_bytes = 0, heap_bytes = 0;

    // hparams (validated against pack)
    float f_eps = 0, f_qk_clip = 0, f_rel_clip = 0, situ_beta = 0, situ_lb = 0,
          w_scale = 0, cap = 0, alpha = 0;
    int64_t vocab_rows = 0, vocab_eff = 0, n_swa = 1024, gate_rank = 64;
    int64_t exp_stride = 0;  // per-expert blob stride from pack expert_index
    float tau_v = 1.0f;      // log-scale for full-attention layers (per token)
    bool fa16 = true;  // emulate the oracle's Flash-Attention F16 cast path

    struct LW {
        const uint8_t *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr,
                      *wgate = nullptr, *wsink = nullptr, *wrel = nullptr,
                      *wkcovp = nullptr, *wvcovp = nullptr;
        const float *qclip = nullptr, *relclip = nullptr, *relbias = nullptr;
        const float *kconv = nullptr, *vconv = nullptr, *postconv = nullptr, *ffnconv = nullptr;
        const float *norm_attn = nullptr, *norm_q = nullptr, *norm_k = nullptr, *norm_v = nullptr;
        const float *router = nullptr, *probs_b = nullptr, *ffn_norm = nullptr;
        const float *aquery = nullptr;  int nq = 0;   // attn_res_query rows: 3 (swa) / 0 (absent)
        const float *ffn_query = nullptr;
        const uint8_t * experts = nullptr;  // base of 128 experts for this layer
    } phys[NPHYS];
    struct LL {
        const float * msgq = nullptr;             // [4][NE] ctrl order Q,K,V,FFN
        const float * down[4] = {}, *gain[4] = {}, *forget[4] = {};
    } loop[12];                                   // physical layers 6..17
    int s_phys[NSLOT]; bool s_swa[NSLOT]; int s_loop[NSLOT];  // slot tables

    // caches (allocated in init)
    std::vector<float> kvk, kvv;  // [NSLOT][ctx][KVD]
    std::vector<float> rs;        // [NSLOT][CH][HR]
    int64_t pos = 0;

    // scratch — everything preallocated, capacity ctx
    std::vector<float> emb, cur, inpSA, stn, mixed, ctrlout, gcat, gainb, forgetb,
        Q, K, V, Rr, sinkb, dynb, attnb, gateb, tA, tB, ffinp, moein, acc, probs, gbuf, ubuf, hbuf, dbuf,
        gate2, finb, prelude_out, prelude_blk,
        blocks, bank, bank_out, msgs, bias_t, sbuf, pbuf, logits;
    std::vector<int32_t> sel;
    // q8_0 activation quantization scratch (max q8_0 GEMV nin = NH*DH = 3072)
    std::vector<int8_t> xq_;
    std::vector<uint16_t> xd16_;
    std::vector<float> dx32_;
    Pool pool;

    static size_t bytes(const std::vector<float> & v) { return v.capacity() * sizeof(float); }

    // ---------------- GEMV kernels ----------------
    // Oracle (ggml-cpu ARM, traits.cpp: q8_0 .vec_dot_type == GGML_TYPE_Q8_0,
    // .from_float == quantize_row_q8_0): mul_mat of f32 activations vs q8_0
    // weights quantizes the ACTIVATION ROW to Q8_0 first (arch/arm/quants.c
    // quantize_row_q8_0: amax over 32-block, d = amax/127 (f32 div), d stored
    // as f16, q = vcvtnq = RNE(x * (1.0f/d_f32))), then the dot is
    // int32-exact per 32-block, scaled by (d_x * d_w) and accumulated in f32
    // (quants.c ggml_vec_dot_q8_0_q8_0_generic). Reproduced exactly; the old
    // dequant-on-the-fly path was ~10x more precise than the oracle itself.
    static void quant_q8_0(const float * x, int n, int8_t * q, uint16_t * dh, float * d32) {
        for (int b = 0, nb = n / 32; b < nb; b++) {
            const float * xb = x + b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; i++) { float v = xb[i]; v = v < 0 ? -v : v; if (v > amax) amax = v; }
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            __fp16 h = (__fp16)d;
            std::memcpy(&dh[b], &h, 2);
            d32[b] = f16_to_f32(dh[b]);
            for (int i = 0; i < 32; i++) {
                float v = xb[i] * id;
                int r = (int)nearbyintf(v);  // RNE == vcvtnq_s32_f32
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                q[b * 32 + i] = (int8_t)r;
            }
        }
    }
    // PATCH #8c: bit-exact emulation of ggml_gemv_q8_0_4x8_q8_0 NEON+DOTPROD path
    // (arch/arm/repack.cpp:1915+, M4 Max has dotprod, no SVE -> this is the oracle kernel).
    // Per 32-wide block b, per output row (lane j):
    //   ret_lane = sum of vdotq over {qs[8k..8k+7]} pairs -> exact int sum of ALL 32 products
    //              (vpaddq merges the two 4-lane accumulators; lane j = plain sequential int sum)
    //   scale    = (float)ad * (float)bd_j              (f16->f32 cvt, then mul, NO fma)
    //   acc_lane = fmaf((float)ret_lane, scale, acc_lane)   // vfmaq: FUSED
    // Column order = plain row order (vpaddq interleaves ret0/ret1 back to j=0..3), so a
    // row-major engine buffer is arithmetically identical to the 4x8 repacked layout.
    static float gemv_q8_0_row_4x8(const uint8_t * w, const int8_t * xq, const float * dx32, int nb) {
        float acc = 0.0f;
        for (int l = 0; l < nb; l++) {
            const uint8_t * wb = w + (size_t)l * 34;
            uint16_t dw16;
            std::memcpy(&dw16, wb, 2);
            const int8_t * qw = (const int8_t *)(wb + 2);
            const int8_t * qx = xq + (size_t)l * 32;
            int sumi = 0;
            for (int i = 0; i < 32; i++) sumi += (int)qw[i] * (int)qx[i];
            float scale = dx32[l] * f16_to_f32(dw16);
            acc = fmaf((float)sumi, scale, acc);
        }
        return acc;
    }
    void gemv_q8_0(const uint8_t * w, int nout, int nin, const float * x, float * y) {
        int nb = nin / 32;
        quant_q8_0(x, nin, xq_.data(), xd16_.data(), dx32_.data());
        auto fn = [&](int lo, int hi) {
            const size_t rbytes = (size_t)nb * 34;
            lo &= ~3;  // col-group-aligned (4-col interleave), each row keeps its fold order
            for (int o = lo; o < hi; o++)
                y[o] = gemv_q8_0_row_4x8(w + (size_t)o * rbytes, xq_.data(), dx32_.data(), nb);
        };
        if ((size_t)nout * nin >= 200000) pool.run(nout, fn); else fn(0, nout);
    }
    // f32 weights: f32 FMA chain (ggml_vec_dot_f32 NEON path = float acc + vfma)
    void gemv_f32(const float * w, int nout, int nin, const float * x, float * y) {
        auto fn = [&](int lo, int hi) {
            for (int o = lo; o < hi; o++) {
                const float * wr = w + (size_t)o * nin;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;  // NEON-style 4 accumulators
                int i = 0;
                for (; i + 3 < nin; i += 4) {
                    a0 = fmaf(wr[i], x[i], a0); a1 = fmaf(wr[i+1], x[i+1], a1);
                    a2 = fmaf(wr[i+2], x[i+2], a2); a3 = fmaf(wr[i+3], x[i+3], a3);
                }
                float acc = (a0 + a1) + (a2 + a3);
                for (; i < nin; i++) acc = fmaf(wr[i], x[i], acc);
                y[o] = acc;
            }
        };
        if ((size_t)nout * nin >= 200000) pool.run(nout, fn); else fn(0, nout);
    }
    void dequant_row_q8_0(const uint8_t * w, float * y, int n) {  // n multiple of 32
        int nb = n / 32;
        for (int b = 0; b < nb; b++) {
            uint16_t d16;
            std::memcpy(&d16, w, 2);
            const int8_t * qs = (const int8_t *)(w + 2);
            float d = f16_to_f32(d16);
            for (int i = 0; i < 32; i++) y[b * 32 + i] = d * (float)qs[i];
            w += 34;
        }
    }

    // ---------------- elementwise building blocks ----------------
    // build_norm / rms_norm: y = x * 1/sqrt(mean(x^2)+eps) [* w]
    static void rmsnorm(const float * x, const float * w, float eps, int n, float * y) {
        double sumsq = 0.0;  // ggml rms_norm accumulates the sum in float; the
        for (int i = 0; i < n; i++) sumsq += (double)x[i] * x[i];  // f64 vs f32 gap is ~1e-7, far under parity noise
        float scale = 1.0f / sqrtf((float)(sumsq / n) + eps);
        if (w) for (int i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
        else   for (int i = 0; i < n; i++) y[i] = x[i] * scale;
    }

    // mix(): depth-attention over [blocks(rev), cur] with learned queries.
    // blocks[] are in PUSH order; states dim1 becomes [b_{n-1},...,b_0, cur].
    void mix(const float * blocks_buf, int nblk, const float * cur_p,
             const float * query, int nq, float * out) {
        int S = nblk + 1;
        for (int i = 0; i < S - 1; i++)
            rmsnorm(blocks_buf + (size_t)(S - 2 - i) * NE, nullptr, 1e-6f, NE, stn.data() + (size_t)i * NE);
        rmsnorm(cur_p, nullptr, 1e-6f, NE, stn.data() + (size_t)(S - 1) * NE);
        const float scale = 1.0f / sqrtf((float)NE);
        for (int j = 0; j < nq; j++) {
            float sc[MAXS], e[MAXS];
            float maxv = -std::numeric_limits<float>::infinity();
            const float * qj = query + (size_t)j * NE;
            for (int i = 0; i < S; i++) {
                const float * st = stn.data() + (size_t)i * NE;
                float d = 0.0f;  // mul_mat f32: ggml_vec_dot_f32 = float acc + vfma
                for (int k = 0; k < NE; k++) d = fmaf(st[k], qj[k], d);
                sc[i] = d * scale;
            }
            for (int i = 0; i < S; i++) maxv = std::max(maxv, sc[i]);
            double sum = 0.0;  // soft_max sum is ggml_float (double) — keep
            for (int i = 0; i < S; i++) { e[i] = expf(sc[i] - maxv); sum += (double)e[i]; }
            float recip = (float)(1.0 / sum);
            for (int i = 0; i < S; i++) e[i] = e[i] * recip;
            float * oj = out + (size_t)j * NE;
            for (int k = 0; k < NE; k++) {
                float accd = 0.0f;  // mul_mat f32: float acc + vfma
                for (int i = 0; i < S; i++) accd = fmaf(e[i], stn[(size_t)i * NE + k], accd);
                oj[k] = accd;
            }
        }
    }

    // control(): (1-forget)*cur + gain*message with tanh gates on rms(cur)||rms(msg)
    void control(const float * c, const float * msg, const LL & lp, int ctrl, float * out) {
        rmsnorm(c, nullptr, 1e-6f, NE, gcat.data());
        rmsnorm(msg, nullptr, 1e-6f, NE, gcat.data() + NE);
        gemv_f32(lp.down[ctrl], (int)gate_rank, 2 * NE, gcat.data(), hbuf.data());
        for (int i = 0; i < (int)gate_rank; i++) hbuf[i] = tanhf(hbuf[i]);
        gemv_f32(lp.gain[ctrl], NE, (int)gate_rank, hbuf.data(), gainb.data());
        gemv_f32(lp.forget[ctrl], NE, (int)gate_rank, hbuf.data(), forgetb.data());
        for (int i = 0; i < NE; i++) {
            float g = tanhf(gainb[i]);
            float fg = tanhf(forgetb[i]);
            // graph order: add(sub(cur, mul(forget,cur)), mul(gain,msg))
            out[i] = (c[i] - fg * c[i]) + g * msg[i];
        }
    }

    // convolve(): causal short conv with residual over per-slot recurrent history.
    // rs_sec points at rs[slot][sec_offset][0]; state rolling happens in place.
    // dynamic taps: dyn flat index lag + 4*head, head = c / group. scale = conv scale.
    void conv_slot(float * rs_sec, int C, const float * xin, const float * wconv,
                   const float * dyn, int group, float scale, float * y) {
        const float sc_dyn = scale;
        if (dyn) {
            for (int c = 0; c < C; c++) {
                const float * w = wconv + c * NCONV;
                float * st = rs_sec + (size_t)c * HR;
                float outv = 0.0f;
                // ssm_conv part: window = {lag0,lag1,lag2,xin}
                for (int k = 0; k < NCONV; k++)
                    outv += (k < HR ? st[k] : xin[c]) * w[k];
                // dynamic taps: lag reads window col (NCONV-1-lag)
                const int head = c / group;
                for (int lag = 0; lag < NCONV; lag++) {
                    const float tap = dyn[lag + NCONV * head];
                    outv += tap * (lag > 0 ? st[HR - lag] : xin[c]);
                }
                y[c] = (xin[c] + outv) * sc_dyn;
            }
        } else {
            for (int c = 0; c < C; c++) {
                const float * w = wconv + c * NCONV;
                float * st = rs_sec + (size_t)c * HR;
                float outv = 0.0f;
                for (int k = 0; k < NCONV; k++)
                    outv += (k < HR ? st[k] : xin[c]) * w[k];
                y[c] = (xin[c] + outv) * scale;
            }
        }
        // build_conv_state_store: last HR columns of x = {lag1, lag2, xin}
        for (int c = 0; c < C; c++) {
            float * st = rs_sec + (size_t)c * HR;
            st[0] = st[1]; st[1] = st[2]; st[2] = xin[c];
        }
    }

    // clip(): scale channels, L2-clip per group (norm=sqrt(sum sq)+1e-6, factor clamped)
    static void clip_heads(float * c, const float * scale, int groups, int per, float thresh) {
        for (int g = 0; g < groups; g++) {
            float * base = c + (size_t)g * per;
            if (scale) for (int d = 0; d < per; d++) base[d] *= scale[(size_t)g * per + d];
            double sumsq = 0.0;
            for (int d = 0; d < per; d++) sumsq += (double)base[d] * (double)base[d];
            float norm = sqrtf((float)sumsq) + 1e-6f;
            float factor = thresh / norm;
            factor = std::min(std::max(factor, 0.0f), 1.0f);
            for (int d = 0; d < per; d++) base[d] *= factor;
        }
    }

    // ---------------- MoE ----------------
    void moe(const LW & L, const float * x, float * out) {
        gemv_f32(L.router, NEXP, NE, x, probs.data());
        float sel_prob[NEXP];
        for (int e = 0; e < NEXP; e++) {
            probs[e] = ksigmoid(probs[e]);
            sel_prob[e] = probs[e] + L.probs_b[e];
        }
        for (int i = 0; i < TOPK; i++) sel[i] = i;
        std::sort(sel.begin(), sel.begin() + TOPK, [&](int a, int b) {
            if (sel_prob[a] != sel_prob[b]) return sel_prob[a] > sel_prob[b];
            return a < b;
        });
        // full 128-descending sort then first 8 == full sorted-descending head:
        // do the complete sort to match ggml's std::sort over all 128 (tie order may
        // differ from ggml's unstable sort; identical values make that irrelevant).
        std::vector<int32_t> all(NEXP);
        for (int e = 0; e < NEXP; e++) all[e] = e;
        std::sort(all.begin(), all.end(), [&](int a, int b) {
            if (sel_prob[a] != sel_prob[b]) return sel_prob[a] > sel_prob[b];
            return a < b;
        });
        for (int i = 0; i < TOPK; i++) sel[i] = all[i];
        if (getenv("M64_DEBUG_ROUTER")) {
            static long dbg_n = 0;
            float m8 = sel_prob[all[7]], m9 = sel_prob[all[8]];
            fprintf(stderr, "DBG_ROUTER %ld p8=%.7f p9=%.7f margin=%.2e sel=%d,%d,%d,%d,%d,%d,%d,%d\n",
                    dbg_n++, m8, m9, m8 - m9, sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7]);
        }
        float wsum = 0.0f;
        for (int i = 0; i < TOPK; i++) wsum += probs[sel[i]];
        float divisor = wsum + 1e-20f;  // scale_bias(1, expert_weights_norm_eps)
        float wgt[TOPK];
        for (int i = 0; i < TOPK; i++) wgt[i] = (probs[sel[i]] / divisor) * w_scale;

        for (int i = 0; i < NE; i++) acc[i] = 0.0f;
        for (int i = 0; i < TOPK; i++) {
            // expert blob stride from the pack header (= 3*part + tail padding)
            const uint8_t * eb = L.experts + (int64_t)sel[i] * exp_stride;
            gemv_q8_0(eb, NF, NE, x, gate2.data());
            gemv_q8_0(eb + PART_BYTES, NF, NE, x, ubuf.data());
            // situ = [beta*tanh(x*(1/beta))] * sigmoid(x) * [lb*tanh(u*(1/lb))]
            // oracle graph: ggml_scale(x, 1.0f/beta) -> tanh -> ggml_scale(beta); sigmoid = 1/(1+expf(-x))
            const float inv_beta = 1.0f / situ_beta, inv_lb = 1.0f / situ_lb;
            for (int f = 0; f < NF; f++) {
                float g = gate2[f], u = ubuf[f];
                float act = situ_beta * tanhf(g * inv_beta);
                act = act * (1.0f / (1.0f + expf(-g)));
                float ulin = situ_lb * tanhf(u * inv_lb);
                hbuf[f] = act * ulin;
            }
            gemv_q8_0(eb + 2 * PART_BYTES, NE, NF, hbuf.data(), dbuf.data());
            float w = wgt[i];
            for (int e = 0; e < NE; e++) acc[e] = acc[e] + w * dbuf[e];
        }
        for (int i = 0; i < NE; i++) out[i] = acc[i];
    }

    // ---------------- attention ----------------
    // K/V already clipped/normed and written into cache slot il at position pos.
    void attention(int il, bool is_swa, float tau_v,
                   float * out /*NE of attn rows: NH*DH*/) {
        int nkv = (int)pos + 1;
        const float scale = 1.0f / sqrtf((float)DH);
        const float * kbase = kvk.data() + (size_t)il * ctx * KVD;
        const float * vbase = kvv.data() + (size_t)il * ctx * KVD;
        // bias gather: bias_t[h][j] = table[h][bucket(dist)] - sink, table[h][b]=sum_r bias[b][r]*R[h][r]
        for (int h = 0; h < NH; h++) {
            const float * rh = Rr.data() + h * NREL;
            for (int j = 0; j < nkv; j++) {
                int dist = (int)pos - j;
                int bucket = dist < EXT ? dist : EXT;
                float v = 0.0f;  // rel bias table via mul_mat f32: float acc + vfma
                if (bucket < EXT) {
                    const float * brow = phys[slot_phys(il)].relbias + (size_t)bucket * NREL;
                    for (int r = 0; r < NREL; r++) v = fmaf(brow[r], rh[r], v);
                }
                bias_t[(size_t)h * ctx + j] = (float)v - sinkb[h];
            }
        }
        for (int h = 0; h < NH; h++) {
            int kv = h / 3;  // GQA: floor div (Appendix A, VERIFIED)
            const float * q = Q.data() + h * DH;
            float * oh = out + h * DH;
            // ggml_compute_forward_flash_attn_ext_f16_one_chunk emulation (online softmax)
            for (int t = 0; t < DH; t++) oh[t] = 0.0f;  // VKQ32 memset
            float M = -std::numeric_limits<float>::infinity();
            double S = 0.0;  // S accumulation is ggml_float
            for (int j = 0; j < nkv; j++) {
                if (is_swa && pos - j > n_swa) continue;  // -inf mask row skipped
                const float * kj = kbase + (size_t)j * KVD + kv * DH;
                float d = 0.0f;
                if (fa16) {
                    for (int t = 0; t < DH; t++) d = fmaf(f16r(q[t]), f16r(kj[t]), d);
                } else {
                    for (int t = 0; t < DH; t++) d = fmaf(q[t], kj[t], d);
                }
                float bv = bias_t[(size_t)h * ctx + j];
                float s = d * scale + (fa16 ? f16r(bv) : bv);
                float ms = 1.0f, vs;
                if (s > M) {  // new max: rescale V and sum
                    ms = expf(M - s);
                    for (int t = 0; t < DH; t++) oh[t] *= ms;
                    M = s; vs = 1.0f;
                } else {
                    vs = expf(s - M);
                }
                const float * vj = vbase + (size_t)j * KVD + kv * DH;
                for (int t = 0; t < DH; t++)
                    oh[t] = fmaf(vs, (fa16 ? f16r(vj[t]) : vj[t]), oh[t]);  // vec_mad = vfma
                S = S * (double)ms + (double)vs;
            }
            // zero-sink branch (sinks tensor is filled with 0.0f)
            {
                float ms = 1.0f, vs;
                if (0.0f > M) { ms = expf(M - 0.0f); for (int t = 0; t < DH; t++) oh[t] *= ms; M = 0.0f; vs = 1.0f; }
                else vs = expf(0.0f - M);
                S = S * (double)ms + (double)vs;
            }
            float S_inv = (float)(1.0 / S);  // ggml_vec_scale_f32(VKQ32, S_inv)
            for (int t = 0; t < DH; t++) oh[t] = oh[t] * S_inv;
        }
        (void)tau_v;
    }

    // ---------------- one layer slot ----------------
    // lp: loop layer for this slot (messages/control); nullptr on pass 0, prelude, coda
    void build_layer(int il, int ip, int & nblk, const LL * lp) {
        int p = s_phys[il];
        const LW & L = phys[p];
        bool is_swa = s_swa[il];

        // attn inputs
        if (L.aquery) {
            mix(blocks.data(), nblk, cur.data(), L.aquery, L.nq, mixed.data());
        } else {
            std::memcpy(mixed.data(), cur.data(), NE * sizeof(float));
        }
        float * inputs[3] = {mixed.data(), mixed.data(), mixed.data()};
        if (is_swa) {
            for (int j = 0; j < 3; j++) inputs[j] = mixed.data();
        } else {
            for (int j = 0; j < 3; j++) inputs[j] = mixed.data() + (size_t)j * NE;
        }
        if (lp) {
            for (int j = 0; j < 3; j++)
                control(inputs[j], msgs.data() + (size_t)j * NE, *lp, j, ctrlout.data() + (size_t)j * NE);
            for (int j = 0; j < 3; j++) inputs[j] = ctrlout.data() + (size_t)j * NE;
        }
        // norms (inkling2.cpp:313-324): swa+messages -> each input normed with
        // shared attn_norm; swa-only -> inputs[1]=inputs[2]=inputs[0]
        float * qsrc = gbuf.data();
        float * ksrc = gbuf.data() + NE;
        float * vsrc = tB.data();
        if (is_swa) {
            rmsnorm(inputs[0], L.norm_attn, f_eps, NE, qsrc);
            if (lp) {
                rmsnorm(inputs[1], L.norm_attn, f_eps, NE, ksrc);
                rmsnorm(inputs[2], L.norm_attn, f_eps, NE, vsrc);
            } else {
                ksrc = vsrc = qsrc;
            }
        } else {
            rmsnorm(inputs[0], L.norm_q, f_eps, NE, qsrc);
            rmsnorm(inputs[1], L.norm_k, f_eps, NE, ksrc);
            rmsnorm(inputs[2], L.norm_v, f_eps, NE, vsrc);
        }
        std::memcpy(inpSA.data(), cur.data(), NE * sizeof(float));  // (kept for push; set by caller too)

        // Q,K,V,R
        gemv_q8_0(L.wq, NH * DH, NE, qsrc, Q.data());
        gemv_q8_0(L.wk, NKV * DH, NE, ksrc, K.data());
        gemv_q8_0(L.wv, NKV * DH, NE, vsrc, V.data());
        gemv_q8_0(L.wrel, NH * NREL, NE, qsrc, Rr.data());

        // convolutions (K: channels 0, V: channels 1024 of the rs state)
        float * rs_sec = rs.data() + ((size_t)il * CH) * HR;
        const float conv_sc = fa_conv_scale_dyn;
        if (L.wkcovp) {
            gemv_q8_0(L.wkcovp, 32, NE, ksrc, dynb.data());
            conv_slot(rs_sec + (size_t)0 * HR, KVD, K.data(), L.kconv, dynb.data(), KVD / 8, conv_sc, tA.data());
        } else {
            conv_slot(rs_sec + (size_t)0 * HR, KVD, K.data(), L.kconv, nullptr, KVD, 1.0f / sqrtf(5.0f), tA.data());
        }
        std::memcpy(K.data(), tA.data(), KVD * sizeof(float));
        if (L.wvcovp) {
            gemv_q8_0(L.wvcovp, 32, NE, vsrc, dynb.data());
            conv_slot(rs_sec + (size_t)KVD * HR, KVD, V.data(), L.vconv, dynb.data(), KVD / 8, conv_sc, tA.data());
        } else {
            conv_slot(rs_sec + (size_t)KVD * HR, KVD, V.data(), L.vconv, nullptr, KVD, 1.0f / sqrtf(5.0f), tA.data());
        }
        std::memcpy(V.data(), tA.data(), KVD * sizeof(float));

        clip_heads(Q.data(), L.qclip, NH, DH, f_qk_clip);
        clip_heads(K.data(), nullptr, NKV, DH, f_qk_clip);
        // V: ggml_rms_norm(Vcur3 [128, 8, T]) -> PER-HEAD norm over dim 128
        for (int h = 0; h < NKV; h++)
            rmsnorm(V.data() + (size_t)h * DH, nullptr, f_eps, DH, tA.data() + (size_t)h * DH);
        std::memcpy(V.data(), tA.data(), KVD * sizeof(float));
        clip_heads(Rr.data(), L.relclip, NH, NREL, f_rel_clip);

        if (!is_swa && tau_v != 1.0f) {
            for (int i = 0; i < NH * DH; i++) Q[i] *= tau_v;
            for (int i = 0; i < NH * NREL; i++) Rr[i] *= tau_v;
        }

        gemv_q8_0(L.wsink, NH, NE, qsrc, sinkb.data());

        // write K,V into cache slot (store_kv writes the clipped/normed tensors)
        {
            float * kb = kvk.data() + ((size_t)il * ctx + pos) * KVD;
            float * vb = kvv.data() + ((size_t)il * ctx + pos) * KVD;
            std::memcpy(kb, K.data(), KVD * sizeof(float));
            std::memcpy(vb, V.data(), KVD * sizeof(float));
        }

        attention(il, is_swa, tau_v, attnb.data());

        // gated output
        gemv_q8_0(L.wgate, NH * DH, NE, qsrc, gateb.data());
        for (int i = 0; i < NH * DH; i++) gateb[i] = ksigmoid(gateb[i]);
        for (int i = 0; i < NH * DH; i++) attnb[i] *= gateb[i];
        gemv_q8_0(L.wo, NE, NH * DH, attnb.data(), tA.data());
        conv_slot(rs_sec + (size_t)(2 * KVD) * HR, NE, tA.data(), L.postconv, nullptr, NE,
                  1.0f / sqrtf(5.0f), tB.data());

        if (ip % 6 == 0) {
            std::memcpy(blocks.data() + (size_t)nblk * NE, inpSA.data(), NE * sizeof(float));
            nblk++;
            std::memcpy(cur.data(), tB.data(), NE * sizeof(float));
        } else {
            for (int i = 0; i < NE; i++) cur[i] = inpSA[i] + tB[i];
        }

        std::memcpy(ffinp.data(), cur.data(), NE * sizeof(float));

        // ffn
        mix(blocks.data(), nblk, cur.data(), L.ffn_query, 1, mixed.data());
        const float * fin;
        if (lp) {
            control(mixed.data(), msgs.data() + (size_t)3 * NE, *lp, 3, ctrlout.data());
            rmsnorm(ctrlout.data(), L.ffn_norm, f_eps, NE, finb.data());
            fin = finb.data();
        } else {
            rmsnorm(mixed.data(), L.ffn_norm, f_eps, NE, finb.data());
            fin = finb.data();
        }
        moe(L, fin, moein.data());
        conv_slot(rs_sec + (size_t)(2 * KVD + NE) * HR, NE, moein.data(), L.ffnconv, nullptr, NE,
                  1.0f / sqrtf(5.0f), tA.data());
        for (int i = 0; i < NE; i++) cur[i] = ffinp[i] + tA[i];
    }

    float fa_conv_scale_dyn = 1.0f / 3.0f;

    // ---------------- full token forward ----------------
    StepResult forward_token(int32_t token, bool want_logits) {
        StepResult r{logits.data(), vocab_eff, 0};
        // embeddings + tok norm
        dequant_row_q8_0(tok_embd_w + (size_t)token * ((size_t)NE / 32 * 34), emb.data(), NE);
        rmsnorm(emb.data(), tok_norm_w, f_eps, NE, cur.data());
        int nblk = 0;

        // tau for full-attention layers (pos < 8191 => exactly 1.0f)
        {
            float t = ((float)pos + 1.0f) * (1.0f / 8192.0f);
            t = t < 1.0f ? 1.0f : t;
            tau_v = logf(t) * alpha + 1.0f;
        }

        // prelude 0..5
        for (int ip = 0; ip < 6; ip++) {
            std::memcpy(inpSA.data(), cur.data(), NE * sizeof(float));
            build_layer(ip, ip, nblk, nullptr);
        }
        // snapshot prelude state (blocks=[B0]); pass machinery per inkling2-loop.cpp:76-110
        const int npre = nblk;  // == 1 (B0)
        std::memcpy(prelude_out.data(), cur.data(), NE * sizeof(float));
        std::memcpy(prelude_blk.data(), blocks.data(), (size_t)npre * NE * sizeof(float));

        for (int pass = 0; pass < 3; pass++) {
            int nbank = 0;
            if (pass > 0) {
                // bank = previous pass's core blocks AFTER the first one (B12^{p-1});
                // bank_out = that pass's output; each pass restarts from prelude output
                nbank = nblk - npre - 1;
                std::memcpy(bank.data(), blocks.data() + (size_t)(npre + 1) * NE,
                            (size_t)nbank * NE * sizeof(float));
                std::memcpy(bank_out.data(), cur.data(), NE * sizeof(float));
                std::memcpy(cur.data(), prelude_out.data(), NE * sizeof(float));
                std::memcpy(blocks.data(), prelude_blk.data(), (size_t)npre * NE * sizeof(float));
                nblk = npre;
            }
            for (int ip = 6; ip < 18; ip++) {
                int il = pass == 0 ? ip : NPHYS + (pass - 1) * 12 + (ip - 6);
                const LL * lp = pass > 0 ? &loop[ip - 6] : nullptr;
                if (lp) mix(bank.data(), nbank, bank_out.data(), lp->msgq, 4, msgs.data());
                std::memcpy(inpSA.data(), cur.data(), NE * sizeof(float));
                build_layer(il, ip, nblk, lp);
            }
        }
        // coda 18..23
        for (int ip = 18; ip < NPHYS; ip++) {
            std::memcpy(inpSA.data(), cur.data(), NE * sizeof(float));
            build_layer(ip, ip, nblk, nullptr);
        }

        // build_output
        mix(blocks.data(), nblk, cur.data(), output_res_query_w, 1, mixed.data());
        rmsnorm(mixed.data(), output_norm_w, f_eps, NE, gbuf.data());

        if (want_logits) {
            gemv_q8_0(output_w, (int)vocab_rows, NE, gbuf.data(), logits.data());
            for (int64_t i = 0; i < vocab_rows; i++) {
                float t = tanhf(logits[i] * (1.0f / cap));
                logits[i] = t * cap;
            }
            int32_t best = 0;
            float bv = logits[0];
            for (int64_t i = 1; i < vocab_eff; i++)
                if (logits[i] > bv) { bv = logits[i]; best = (int32_t)i; }
            r.argmax = best;
        }
        pos++;
        return r;
    }

    // tensor pointers
    const uint8_t * tok_embd_w = nullptr;
    const float * tok_norm_w = nullptr, * output_norm_w = nullptr, * output_res_query_w = nullptr;
    const uint8_t * output_w = nullptr;

    // ---------------- build ----------------
    static const float * f32p(const PackReader & pk, const std::string & n) {
        return (const float *)pk.ptr(pk.require(n));
    }
    static const uint8_t * q8p(const PackReader & pk, const std::string & n) {
        return (const uint8_t *)pk.ptr(pk.require(n));
    }

    bool init(PackReader & pk, int context_len, std::string & err) {
        pack = &pk;
        if (pk.header().arch != "inkling2_loop") { err = "init: arch '" + pk.header().arch + "' != inkling2_loop"; return false; }
        auto hp = [&](const char *k, double f) { return pk.hparam_f64(std::string("inkling2_loop.") + k, f); };
        auto hi = [&](const char *k, int64_t f) { return pk.hparam_i64(std::string("inkling2_loop.") + k, f); };

        // hard hparam validation against the pinned constants
        auto chk = [&](const char * key, int64_t want) {
            int64_t got = hi(key, -1);
            if (got != want) { err = std::string("init: hparam ") + key + " = " + std::to_string(got) +
                                     ", expected " + std::to_string(want); return false; }
            return true;
        };
        if (!chk("block_count", NPHYS) || !chk("embedding_length", NE) ||
            !chk("attention.head_count", NH) || !chk("attention.head_count_kv", NKV) ||
            !chk("attention.key_length", DH) || !chk("attention.value_length", DH) ||
            !chk("attention.relative_length", NREL) || !chk("attention.relative_extent", EXT) ||
            !chk("attention.sliding_window", EXT) || !chk("attn_res.block_size", 6) ||
            !chk("expert_count", NEXP) || !chk("expert_used_count", TOPK) ||
            !chk("expert_feed_forward_length", NF) || !chk("shortconv.l_cache", NCONV) ||
            !chk("loop.count", 3) || !chk("loop.start", 6) || !chk("loop.end", 18) ||
            !chk("loop.gate_rank", 64) || !chk("attention.log_scale_floor", 8192))
            return false;

        f_eps = (float)hp("attention.layer_norm_rms_epsilon", 1e-6);
        f_qk_clip = (float)hp("attention.qk_clip", 22.627416997969522);
        f_rel_clip = (float)hp("attention.relative_clip", 8.0);
        alpha = (float)hp("attention.log_scale_alpha", 0.1);
        situ_beta = (float)hp("activation.situ_beta", 4.0);
        situ_lb = (float)hp("activation.situ_linear_beta", 25.0);
        w_scale = (float)hp("expert_weights_scale", 8.0);
        cap = (float)hp("final_logit_softcapping", 30.0);
        vocab_rows = hi("vocab_size", 201024);
        vocab_eff = hi("vocab_size_unpadded", 200058);
        n_swa = (int64_t)hi("attention.sliding_window", 1024);
        gate_rank = hi("loop.gate_rank", 64);

        if (const char * e = getenv("M64_NO_FA16")) { (void)e; fa16 = false; }

        // swa pattern: 1 -> sliding window, 0 -> full
        std::vector<int64_t> pattern;
        {
            auto it = pk.header().hparams.find("inkling2_loop.attention.sliding_window_pattern");
            if (it == pk.header().hparams.end() || it->second->type != JValue::Arr) {
                err = "init: missing sliding_window_pattern"; return false;
            }
            for (auto & v : it->second->arr) pattern.push_back(v->as_int());
            if ((int)pattern.size() != NPHYS) { err = "init: pattern size != block_count"; return false; }
        }
        bool swa_phys[NPHYS];
        for (int l = 0; l < NPHYS; l++) swa_phys[l] = pattern[l] != 0;
        for (int il = 0; il < NSLOT; il++) {
            s_phys[il] = slot_phys(il);
            s_swa[il] = swa_phys[s_phys[il]];
            int p = s_phys[il];
            s_loop[il] = (p >= 6 && p < 18) ? p - 6 : -1;
        }

        // resolve tensors
        tok_embd_w = q8p(pk, "token_embd.weight");
        tok_norm_w = f32p(pk, "token_embd_norm.weight");
        output_w = q8p(pk, "output.weight");
        output_norm_w = f32p(pk, "output_norm.weight");
        output_res_query_w = f32p(pk, "output_res_query.weight");

        const auto & eidx = pk.header().expert_index;
        if (!eidx.present || (int)eidx.layer_base.size() != NPHYS || eidx.expert_stride <= 0) {
            err = "init: expert index missing/broken"; return false;
        }

        for (int p = 0; p < NPHYS; p++) {
            LW & L = phys[p];
            std::string b = "blk." + std::to_string(p) + ".";
            L.wq = q8p(pk, b + "attn_q.weight");
            L.wk = q8p(pk, b + "attn_k.weight");
            L.wv = q8p(pk, b + "attn_v.weight");
            L.wo = q8p(pk, b + "attn_output.weight");
            L.wgate = q8p(pk, b + "attn_gate.weight");
            L.wsink = q8p(pk, b + "attn_sink_proj.weight");
            L.wrel = q8p(pk, b + "attn_rel.weight");
            L.qclip = f32p(pk, b + "attn_q_clip.weight");
            L.relclip = f32p(pk, b + "attn_rel_clip.weight");
            L.relbias = f32p(pk, b + "attn_rel_bias.weight");
            L.kconv = f32p(pk, b + "attn_k_conv.weight");
            L.vconv = f32p(pk, b + "attn_v_conv.weight");
            L.postconv = f32p(pk, b + "attn_post_conv.weight");
            L.ffnconv = f32p(pk, b + "ffn_post_conv.weight");
            L.router = f32p(pk, b + "ffn_gate_inp.weight");
            L.probs_b = f32p(pk, b + "exp_probs_b.bias");
            L.ffn_norm = f32p(pk, b + "ffn_norm.weight");
            L.ffn_query = f32p(pk, b + "ffn_res_query.weight");
            const TensorInfo * einfo = pk.find(b + "ffn_experts.weight");
            if (!einfo || !einfo->is_expert) { err = "init: missing " + b + "ffn_experts.weight"; return false; }
            if (einfo->expert_stride != eidx.expert_stride) { err = "init: expert stride mismatch"; return false; }
            if (!einfo->part_gate.empty()) {
                int64_t gsz = einfo->part_gate[0] * einfo->part_gate[1];
                if (gsz / 32 * 34 != PART_BYTES) { err = "init: expert part size mismatch"; return false; }
            }
            L.experts = (const uint8_t *)pk.data() + eidx.layer_base[p];
            exp_stride = eidx.expert_stride;
            bool full = !swa_phys[p];
            L.norm_attn = pk.find(b + "attn_norm.weight") ? f32p(pk, b + "attn_norm.weight") : nullptr;
            L.nq = 0;
            if (const TensorInfo * aq = pk.find(b + "attn_res_query.weight")) {
                L.aquery = (const float *)pk.ptr(*aq);
                if (aq->length % (4 * (int64_t)NE) != 0) { err = "init: attn_res_query rows not multiple of n_embd"; return false; }
                L.nq = (int)(aq->length / (4 * (int64_t)NE));
            }
            if (full) {
                L.norm_q = f32p(pk, b + "attn_q_inp_norm.weight");
                L.norm_k = f32p(pk, b + "attn_k_inp_norm.weight");
                L.norm_v = f32p(pk, b + "attn_v_inp_norm.weight");
                L.wkcovp = q8p(pk, b + "attn_k_conv_proj.weight");
                L.wvcovp = q8p(pk, b + "attn_v_conv_proj.weight");
                if (L.nq != 3) { err = "init: full layer " + std::to_string(p) + " attn_res_query must have 3 queries"; return false; }
            } else {
                if (!L.norm_attn) { err = "init: swa layer " + std::to_string(p) + " missing attn_norm"; return false; }
                // layer 0 has no earlier blocks to mix -> no attn_res_query; other swa layers: 1 query
                if (p == 0 ? (L.nq != 0) : (L.nq != 1)) { err = "init: swa layer " + std::to_string(p) + " attn_res_query query count wrong"; return false; }
            }
        }
        for (int li = 0; li < 12; li++) {
            std::string b = "blk." + std::to_string(li + 6) + ".";
            LL & lpre = loop[li];
            lpre.msgq = f32p(pk, b + "loop_msg_query.weight");
            const char * nm[4] = {"q", "k", "v", "ffn"};
            for (int c = 0; c < 4; c++) {
                std::string s = b + "loop_" + nm[c] + "_ctrl_";
                lpre.down[c] = f32p(pk, s + "down.weight");
                lpre.gain[c] = f32p(pk, s + "gain.weight");
                lpre.forget[c] = f32p(pk, s + "forget.weight");
            }
        }

        if (context_len < 4) context_len = 4;
        if (context_len > 4096) context_len = 4096;
        ctx = context_len;

        // ---- allocate everything up front ----
        auto alloc = [&](std::vector<float> & v, size_t n) { v.assign(n, 0.0f); };
        alloc(kvk, (size_t)NSLOT * ctx * KVD);
        alloc(kvv, (size_t)NSLOT * ctx * KVD);
        alloc(rs, (size_t)NSLOT * CH * HR);
        alloc(emb, NE); alloc(cur, NE); alloc(inpSA, NE);
        alloc(stn, (size_t)MAXS * NE);
        alloc(mixed, 3 * NE); alloc(ctrlout, 3 * NE);
        alloc(gcat, 2 * NE); alloc(gainb, NE); alloc(forgetb, NE);
        alloc(Q, NH * DH); alloc(K, KVD); alloc(V, KVD); alloc(Rr, NH * NREL);
        alloc(sinkb, NH); alloc(dynb, 64); alloc(attnb, NH * DH); alloc(gateb, NH * DH);
        alloc(tA, (size_t)4 * NE); alloc(tB, (size_t)4 * NE);
        alloc(ffinp, NE); alloc(moein, NE); alloc(acc, NE);
        alloc(probs, NEXP); alloc(gbuf, (size_t)2 * NE); alloc(ubuf, NF); alloc(hbuf, NF); alloc(dbuf, NE);
        alloc(blocks, (size_t)MAXS * NE); alloc(bank, (size_t)MAXS * NE); alloc(bank_out, NE);
        alloc(gate2, NF); alloc(finb, NE); alloc(prelude_out, NE); alloc(prelude_blk, (size_t)MAXS * NE);
        alloc(msgs, 4 * NE);
        alloc(bias_t, (size_t)NH * ctx); alloc(sbuf, ctx); alloc(pbuf, ctx);
        alloc(logits, (size_t)vocab_rows);
        sel.resize(TOPK);
        xq_.assign(3072, 0); xd16_.assign(96, 0); dx32_.assign(96, 0.0f);

        pos = 0;
        pool.start();
        return true;
    }

    size_t resident_bytes() const {
        size_t h = bytes(kvk) + bytes(kvv) + bytes(rs) + bytes(emb) + bytes(cur) + bytes(inpSA) +
                   bytes(stn) + bytes(mixed) + bytes(ctrlout) + bytes(gcat) + bytes(gainb) +
                   bytes(forgetb) + bytes(Q) + bytes(K) + bytes(V) + bytes(Rr) + bytes(sinkb) +
                   bytes(dynb) + bytes(attnb) + bytes(gateb) + bytes(tA) + bytes(tB) + bytes(ffinp) +
                   bytes(moein) + bytes(acc) + bytes(probs) + bytes(gbuf) + bytes(ubuf) + bytes(hbuf) +
                   bytes(dbuf) + bytes(gate2) + bytes(finb) + bytes(prelude_out) + bytes(prelude_blk) +
                   bytes(blocks) + bytes(bank) + bytes(bank_out) + bytes(msgs) +
                   bytes(bias_t) + bytes(sbuf) + bytes(pbuf) + bytes(logits);
        return h + mapped_bytes;
    }

    void reset() {
        std::fill(kvk.begin(), kvk.end(), 0.0f);
        std::fill(kvv.begin(), kvv.end(), 0.0f);
        std::fill(rs.begin(), rs.end(), 0.0f);
        pos = 0;
    }
};
}  // namespace m64
