// gemma4_forward_metal.mm — Gemma 4 resident Metal forward (Milestone 6B).
//
// Orchestrates the g4_* kernels in engine/kernels/m64_gemma4.metal to replay
// gemma4_forward.cpp exactly: one command buffer + one compute encoder per
// token; the whole .m64pack is wrapped zero-copy as PACK (mmap is page-
// aligned, so rounding file_size up to the page boundary satisfies
// newBufferWithBytesNoCopy), activations live in one shared SCR float buffer
// at non-overlapping FLOAT offsets, and K/V caches are per-layer fp16 or
// Q8_0 ring buffers (SWA layers ring at n_swa+1 slots, full layers at ctx).
//
// The pack tensor byte offset is derived from the reader itself:
//   woff = (uint8_t *)pack.ptr(*t) - (uint8_t *)pack.data()
// so no header-offset guessing happens anywhere.
#import <Foundation/Foundation.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

#include "gemma4_forward_metal.hpp"
#include "metal_runtime.hpp"
#include "pack_reader.hpp"
#include "../kernels/m64_gemma4_args.h"

#ifndef M64_KERNEL_DIR
#define M64_KERNEL_DIR "engine/kernels"
#endif
#ifndef M64_ARCHIVE_PATH
#define M64_ARCHIVE_PATH "build/m64_p2a.ar"
#endif

namespace m64 {

namespace {
constexpr uint32_t kTG = 256u;  // standard vector-kernel threadgroup

const char * dense_gemv_kernel(DType dt) {
    switch (dt) {
        case DType::F32:  return "g4_gemv_f32";
        case DType::Q8_0: return "g4_gemv_q8_0";
        case DType::Q4_K: return "g4_gemv_q4_k";
        case DType::Q5_0: return "g4_gemv_q5_0";
        default:          return nullptr;
    }
}
const char * moe_gemv_kernel(DType dt) {
    switch (dt) {
        case DType::Q8_0: return "g4_moe_gemv_q8_0";
        case DType::Q4_K: return "g4_moe_gemv_q4_k";
        case DType::Q5_0: return "g4_moe_gemv_q5_0";
        default:          return nullptr;
    }
}
int g4_dtype_code(DType dt) {
    switch (dt) {
        case DType::F32:  return G4_DT_F32;
        case DType::F16:  return G4_DT_F16;
        case DType::Q8_0: return G4_DT_Q8_0;
        case DType::Q4_K: return G4_DT_Q4_K;
        case DType::Q5_0: return G4_DT_Q5_0;
        default:          return -1;
    }
}
}  // namespace

struct Gemma4ForwardMetal::Impl {
    PackReader * pack_ = nullptr;

    // ---- static config (from pack hparams / tensor shapes) ----
    int64_t n_layer = 0, n_embd = 0, n_ff = 0, n_ff_exp = 0;
    int64_t n_expert = 0, top_k = 0, n_head = 0;
    int64_t key_len = 0, val_len = 0, key_len_swa = 0, val_len_swa = 0;
    int64_t rope_dim = 0, rope_dim_swa = 0;
    int64_t n_swa = 0;
    float freq_base_full = 0, freq_base_swa = 0, eps = 0, cap = 0;
    int ctx = 0;
    bool kv_q8 = false;
    std::vector<bool> is_swa;
    std::vector<int64_t> n_kv;
    std::vector<int64_t> hd_k, hd_v, n_rot;
    int64_t vocab_rows = 0, vocab_eff = 0;
    int64_t pos_ = 0;
    size_t mapped_bytes = 0, scr_bytes = 0, kv_bytes = 0;

    // ---- pack tensors (names/dtypes per gemma4_forward.cpp) ----
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

    // ---- Metal state ----
    MetalRuntime rt_;
    id<MTLBuffer> PACK_ = nil, SCR_ = nil;
    std::vector<id<MTLBuffer>> kv_k_, kv_v_;   // per layer
    std::vector<uint32_t> kv_cap_;             // ring capacity per layer (tokens)

    // ---- SCR float offsets (non-overlapping) ----
    uint32_t off_inp = 0, off_cur = 0, off_normed = 0, off_attn_out = 0;
    uint32_t off_mlp = 0, off_moe = 0, off_expert_acc = 0;
    uint32_t off_q = 0, off_k = 0, off_v = 0, off_attn = 0;
    uint32_t off_work_a = 0, off_work_b = 0;
    uint32_t off_router = 0, off_sel_ids = 0, off_sel_w = 0;
    uint32_t off_logits = 0, off_argmax_arena = 0, off_argmax_out = 0;
    uint32_t off_zero = 0;  // SCR slot that stays all-zero (GPU-side memset source)

    // ---- Milestone 8A: batched-prefill slots ([G4_BMAX][stride] row-major) ----
    int64_t chunk_ = 0;  // rows per chunk (0 => G4_BMAX); set_prefill_chunk clamps
    uint32_t b_inp = 0, b_cur = 0, b_normed = 0, b_attn_out = 0;
    uint32_t b_mlp = 0, b_moe = 0, b_eacc = 0, b_zero = 0;
    uint32_t b_q = 0, b_k = 0, b_v = 0, b_attn = 0;
    uint32_t b_wa = 0, b_wb = 0, b_ea = 0, b_eb = 0;
    uint32_t b_router = 0, b_sel_ids = 0, b_sel_w = 0;
    id<MTLBuffer> stg_k_ = nil, stg_v_ = nil;  // in-chunk KV staging (ring layout)

    std::vector<float> logits_;  // host copy backing StepResult::logits

    uint64_t woff(const TensorInfo * t) const {
        return (uint64_t)((const uint8_t *)pack_->ptr(*t) - (const uint8_t *)pack_->data());
    }

    // ============================== init ==============================
    bool init(PackReader & pack, int context_len, std::string & err,
              double arena_gb, const std::string & io_mode) {
        (void)io_mode;  // resident-only in Milestone 6B
        if (pack.header().arch != "gemma4") {
            err = "gemma4_forward_metal: arch '" + pack.header().arch + "' is not 'gemma4'";
            return false;
        }
        if (context_len <= 0) { err = "gemma4_forward_metal: context_len must be > 0"; return false; }
        if (arena_gb > 0.0) {
            err = "gemma4_forward_metal: arena streaming not yet implemented";
            return false;
        }
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
                err = "gemma4_forward_metal: bad core hparams"; return false;
            }
            std::vector<bool> pat = pack.hparam_bool_arr("gemma4.attention.sliding_window_pattern");
            if ((int64_t)pat.size() != n_layer) {
                err = "gemma4_forward_metal: sliding_window_pattern size != block_count"; return false;
            }
            std::vector<int64_t> kv = pack.hparam_i64_arr("gemma4.attention.head_count_kv");
            if (kv.empty()) kv.assign(n_layer, pack.hparam_i64("gemma4.attention.head_count_kv", 1));
            if ((int64_t)kv.size() == 1) kv.resize(n_layer, kv[0]);
            if ((int64_t)kv.size() != n_layer) {
                err = "gemma4_forward_metal: head_count_kv size != block_count"; return false;
            }
            is_swa = pat; n_kv = kv;
            hd_k.resize(n_layer); hd_v.resize(n_layer); n_rot.resize(n_layer);
            for (int64_t il = 0; il < n_layer; il++) {
                hd_k[il] = is_swa[il] ? key_len_swa : key_len;
                hd_v[il] = is_swa[il] ? val_len_swa : val_len;
                n_rot[il] = is_swa[il] ? rope_dim_swa : rope_dim;
                if (n_kv[il] <= 0 || n_head % n_kv[il] != 0) {
                    err = "gemma4_forward_metal: bad head_count_kv on layer " + std::to_string(il); return false;
                }
                if (hd_k[il] != hd_v[il]) {
                    err = "gemma4_forward_metal: hd_k != hd_v on layer " + std::to_string(il) +
                          " (single-hd attention kernel)"; return false;
                }
                if (hd_k[il] > 512 || hd_k[il] <= 0) {
                    err = "gemma4_forward_metal: hd must be in (0, 512] on layer " + std::to_string(il); return false;
                }
                if (kv_q8 && hd_k[il] % 32 != 0) {
                    err = "gemma4_forward_metal: kv_q8 needs hd % 32 == 0 on layer " + std::to_string(il); return false;
                }
            }
        } catch (const std::exception & e) {
            err = std::string("gemma4_forward_metal: ") + e.what(); return false;
        }

        tok_embd = pack.find("token_embd.weight");
        output = pack.find("output.weight");  // optional: tied LM head when absent
        output_norm = pack.find("output_norm.weight");
        rope_freqs_root = pack.find("rope_freqs.weight");
        if (!tok_embd || !output_norm) { err = "gemma4_forward_metal: missing token_embd/output_norm"; return false; }
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
                err = "gemma4_forward_metal: router present but MoE hparams/tensors missing on layer " +
                      std::to_string(il);
                return false;
            }
        }
        if (!miss.empty()) { err = "gemma4_forward_metal: missing tensor '" + miss + "'"; return false; }

        vocab_rows = (output ? output : tok_embd)->shape[0];
        vocab_eff = pack.hparam_i64("gemma4.vocab_size_unpadded", vocab_rows);
        if (vocab_eff <= 0 || vocab_eff > vocab_rows) {
            err = "gemma4_forward_metal: bad vocab sizes"; return false;
        }
        if ((int64_t)tok_embd->shape.size() < 2 || tok_embd->shape[1] != n_embd ||
            tok_embd->shape[0] < vocab_eff) {
            err = "gemma4_forward_metal: token_embd shape mismatch"; return false;
        }
        for (int64_t il = 0; il < n_layer; il++) {
            const LT & t = lt[il];
            if (t.wq->shape[0] != n_head * hd_k[il] || t.wq->shape[1] != n_embd ||
                t.wk->shape[0] != n_kv[il] * hd_k[il] ||
                (t.wv && t.wv->shape[0] != n_kv[il] * hd_v[il]) ||
                t.wo->shape[0] != n_embd || t.wo->shape[1] != n_head * hd_v[il]) {
                err = "gemma4_forward_metal: attention shape mismatch on layer " + std::to_string(il);
                return false;
            }
        }

        // ---- dtype gating: every tensor must map cleanly onto an existing kernel ----
        const int emb_dt = g4_dtype_code(tok_embd->dtype);
        if (emb_dt != G4_DT_Q8_0 && emb_dt != G4_DT_F32) {
            err = "gemma4_forward_metal: token_embd dtype must be q8_0 or f32 for the gather kernel";
            return false;
        }
        auto need_f32 = [&](const TensorInfo * t, const char * what, int64_t il) -> bool {
            if (t && t->dtype != DType::F32) {
                err = std::string("gemma4_forward_metal: ") + what + " must be f32 (layer " +
                      std::to_string(il) + ", got " + dtype_name(t->dtype) + ")";
                return false;
            }
            return true;
        };
        if (!need_f32(output_norm, "output_norm", -1)) return false;
        if (!need_f32(rope_freqs_root, "rope_freqs", -1)) return false;
        for (int64_t il = 0; il < n_layer; il++) {
            const LT & t = lt[il];
            if (!need_f32(t.q_norm, "attn_q_norm", il) || !need_f32(t.k_norm, "attn_k_norm", il) ||
                !need_f32(t.rope_freqs, "rope_freqs", il) || !need_f32(t.attn_norm, "attn_norm", il) ||
                !need_f32(t.post_attn_norm, "post_attention_norm", il) ||
                !need_f32(t.ffn_norm, "ffn_norm", il) ||
                !need_f32(t.post_ffw_norm_1, "post_ffw_norm_1", il) ||
                !need_f32(t.pre_ffw_norm_2, "pre_ffw_norm_2", il) ||
                !need_f32(t.post_ffw_norm_2, "post_ffw_norm_2", il) ||
                !need_f32(t.post_ffw_norm, "post_ffw_norm", il) ||
                !need_f32(t.layer_scale, "layer_output_scale", il) ||
                !need_f32(t.router_scale, "ffn_gate_inp.scale", il) ||
                !need_f32(t.down_scale, "ffn_down_exps.scale", il)) return false;
            const TensorInfo * gm[] = {t.wq, t.wk, t.wv, t.wo, t.ffn_gate, t.ffn_up, t.ffn_down, t.router_w};
            const char * gmn[] = {"attn_q", "attn_k", "attn_v", "attn_output",
                                  "ffn_gate", "ffn_up", "ffn_down", "ffn_gate_inp"};
            for (int g = 0; g < 8; g++) {
                if (gm[g] && !dense_gemv_kernel(gm[g]->dtype)) {
                    err = std::string("gemma4_forward_metal: unsupported dense GEMV dtype (") +
                          dtype_name(gm[g]->dtype) + ") for " + gmn[g] + " on layer " + std::to_string(il);
                    return false;
                }
            }
            // GEMV nin divisibility by the dtype block size
            auto nin_ok = [&](const TensorInfo * t, int64_t nin, const char * what) -> bool {
                const int64_t be = dtype_block_elems(t->dtype);
                if (be <= 0 || nin % be != 0) {
                    err = std::string("gemma4_forward_metal: nin ") + std::to_string(nin) +
                          " not divisible by " + dtype_name(t->dtype) + " block (" + what +
                          ", layer " + std::to_string(il) + ")";
                    return false;
                }
                return true;
            };
            if (!nin_ok(t.wq, n_embd, "attn_q") || !nin_ok(t.wk, n_embd, "attn_k") ||
                (t.wv && !nin_ok(t.wv, n_embd, "attn_v")) || !nin_ok(t.wo, n_head * hd_v[il], "attn_output") ||
                !nin_ok(t.ffn_gate, n_embd, "ffn_gate") || !nin_ok(t.ffn_up, n_embd, "ffn_up") ||
                !nin_ok(t.ffn_down, n_ff, "ffn_down") ||
                (t.router_w && !nin_ok(t.router_w, n_embd, "ffn_gate_inp"))) return false;
            if (t.router_w) {
                const TensorInfo * ex = t.experts;
                DType dg = ex->part_dtype(ex->dtype, 0), du = ex->part_dtype(ex->dtype, 1),
                      dd = ex->part_dtype(ex->dtype, 2);
                if (!moe_gemv_kernel(dg) || !moe_gemv_kernel(du) || !moe_gemv_kernel(dd)) {
                    err = "gemma4_forward_metal: unsupported expert part dtype on layer " +
                          std::to_string(il);
                    return false;
                }
                if (n_embd % dtype_block_elems(dg) || n_embd % dtype_block_elems(du) ||
                    n_ff_exp % dtype_block_elems(dd)) {
                    err = "gemma4_forward_metal: expert nin not divisible by block size on layer " +
                          std::to_string(il);
                    return false;
                }
            }
        }
        if (output && !dense_gemv_kernel(output->dtype)) {
            err = std::string("gemma4_forward_metal: unsupported LM-head dtype (") +
                  dtype_name(output->dtype) + ")";
            return false;
        }

        ctx = context_len;

        // ===================== Metal resources =====================
        if (!rt_.init(M64_KERNEL_DIR, M64_ARCHIVE_PATH, err)) {
            err = "gemma4_forward_metal: " + err;
            return false;
        }
        const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
        const size_t page_up = ((size_t)pack.file_size() + pg - 1) & ~(pg - 1);
        PACK_ = rt_.wrap_external(const_cast<void *>((const void *)pack.data()), page_up, @"gemma4_pack");
        if (!PACK_) { err = "gemma4_forward_metal: failed to wrap pack mmap as GPU buffer"; return false; }
        mapped_bytes = (size_t)pack.file_size();

        // SCR float-offset layout (non-overlapping slots)
        auto slot = [&](int64_t nfloats) {
            const uint32_t o = (uint32_t)(scr_bytes / 4u);  // FLOAT offset into SCR
            scr_bytes += (size_t)nfloats * 4u;
            return o;
        };
        off_inp        = slot(n_embd);
        off_cur        = slot(n_embd);
        off_normed     = slot(n_embd);
        off_attn_out   = slot(n_embd);
        off_mlp        = slot(n_embd);
        off_moe        = slot(n_embd);
        off_expert_acc = slot(n_embd);
        off_zero       = slot(n_embd);  // never written: alloc_shared zeros it once
        const int64_t maxhk = std::max(key_len, key_len_swa), maxhv = std::max(val_len, val_len_swa);
        const int64_t maxnk = *std::max_element(n_kv.begin(), n_kv.end());
        off_q       = slot(n_head * maxhk);
        off_k       = slot(maxnk * maxhk);
        off_v       = slot(maxnk * maxhv);
        off_attn    = slot(n_head * maxhv);
        const int64_t wide = std::max(n_ff, n_ff_exp);
        off_work_a  = slot(wide);
        off_work_b  = slot(wide);
        off_router  = slot(std::max(n_expert, (int64_t)1));
        off_sel_ids = slot(std::max(top_k, (int64_t)1));
        off_sel_w   = slot(std::max(top_k, (int64_t)1));
        off_logits  = slot(vocab_rows);
        off_argmax_arena = slot(64);   // legacy argmax arena slot (unused)
        off_argmax_out   = slot(4);    // int32 result (+ padding)

        // ---- Milestone 8A batched-prefill slots (row-major [rows][stride]) ----
        // Strides equal the single-token slot widths (n_embd, n_head*hd, ...) so
        // g4_rms_norm / g4_add / g4_copy / g4_scale_scalar_buf — which address
        // rows as base + row * k or run flat over n = rows * stride — replay on
        // the batch slots with no stride-aware variants.
        const int64_t ne = n_embd, B = G4_BMAX;
        b_inp      = slot(ne * B);
        b_cur      = slot(ne * B);
        b_normed   = slot(ne * B);
        b_attn_out = slot(ne * B);
        b_mlp      = slot(ne * B);
        b_moe      = slot(ne * B);
        b_eacc     = slot(ne * B);
        b_q       = slot(n_head * maxhk * B);
        b_k       = slot(maxnk * maxhk * B);
        b_v       = slot(maxnk * maxhv * B);
        b_attn    = slot(n_head * maxhv * B);
        b_wa      = slot(n_ff * B);
        b_wb      = slot(n_ff * B);
        b_ea      = slot(n_ff_exp * B);
        b_eb      = slot(n_ff_exp * B);
        b_router  = slot(std::max(n_expert, (int64_t)1) * B);
        b_sel_ids = slot(std::max(top_k, (int64_t)1) * B);  // ints, float-sized
        b_sel_w   = slot(std::max(top_k, (int64_t)1) * B);
        b_zero    = slot(ne * B);  // never written: alloc_shared zeros it once
        SCR_ = rt_.alloc_shared(scr_bytes, @"gemma4_scr");
        if (!SCR_) { err = "gemma4_forward_metal: SCR allocation failed"; return false; }

        // In-chunk KV staging: same byte layout as a ring slice, G4_BMAX slots,
        // sized for the widest layer (shared across layers, serialized use).
        const size_t stg_bytes = kv_q8
            ? (size_t)G4_BMAX * maxnk * (maxhk / 32u) * 34u
            : (size_t)G4_BMAX * maxnk * maxhk * sizeof(uint16_t);
        stg_k_ = rt_.alloc_shared(stg_bytes, @"gemma4_stg_k");
        stg_v_ = rt_.alloc_shared(stg_bytes, @"gemma4_stg_v");
        if (!stg_k_ || !stg_v_) {
            err = "gemma4_forward_metal: staging KV allocation failed"; return false;
        }
        kv_bytes += 2 * stg_bytes;

        // Per-layer KV rings (fp16 or Q8_0)
        kv_k_.assign(n_layer, nil);
        kv_v_.assign(n_layer, nil);
        kv_cap_.assign(n_layer, 0);
        for (int64_t il = 0; il < n_layer; il++) {
            const uint32_t cap_l = (uint32_t)((is_swa[il] && n_swa > 0)
                                                  ? std::min<int64_t>(ctx, n_swa + 1) : ctx);
            kv_cap_[il] = cap_l;
            const size_t bytes = kv_q8
                ? (size_t)cap_l * n_kv[il] * (hd_k[il] / 32) * 34u
                : (size_t)cap_l * n_kv[il] * hd_k[il] * sizeof(uint16_t);
            kv_k_[il] = rt_.alloc_shared(bytes, @"gemma4_kv_k");
            kv_v_[il] = rt_.alloc_shared(bytes, @"gemma4_kv_v");
            if (!kv_k_[il] || !kv_v_[il]) {
                err = "gemma4_forward_metal: KV allocation failed on layer " + std::to_string(il);
                return false;
            }
            kv_bytes += 2 * bytes;
        }

        // ---- pre-validate every pipeline this forward will dispatch ----
        std::vector<const char *> fns = {
            emb_dt == G4_DT_Q8_0 ? "g4_emb_gather_q8_0" : "g4_emb_gather_f32",
            "g4_rms_norm", "g4_rope_neox",
            kv_q8 ? "g4_kv_write_q8" : "g4_kv_write_f16",
            kv_q8 ? "g4_attn_q8" : "g4_attn_f16",
            "g4_geglu", "g4_add", "g4_copy", "g4_argmax",
        };
        for (int64_t il = 0; il < n_layer; il++) {
            const LT & t = lt[il];
            const TensorInfo * gm[] = {t.wq, t.wk, t.wv, t.wo, t.ffn_gate, t.ffn_up, t.ffn_down, t.router_w};
            for (const TensorInfo * g : gm)
                if (g) fns.push_back(dense_gemv_kernel(g->dtype));
            if (t.layer_scale) fns.push_back("g4_scale_scalar_buf");
            if (t.router_w) {
                fns.push_back("g4_router_topk");
                const TensorInfo * ex = t.experts;
                fns.push_back(moe_gemv_kernel(ex->part_dtype(ex->dtype, 0)));
                fns.push_back(moe_gemv_kernel(ex->part_dtype(ex->dtype, 1)));
                fns.push_back(moe_gemv_kernel(ex->part_dtype(ex->dtype, 2)));
            }
        }
        if (output) fns.push_back(dense_gemv_kernel(output->dtype));
        else fns.push_back(dense_gemv_kernel(tok_embd->dtype));  // tied LM head
        if (cap > 0.0f) fns.push_back("g4_softcap");
        // Milestone 8A: batched prefill kernels (same set, *_b variants).
        std::vector<std::string> bnames;  // owned strings; validated in a separate
                                          // loop below (never c_str() into fns here)
        auto need_b = [&](const char * base) {
            bnames.push_back(std::string(base) + "_b");
        };
        need_b(emb_dt == G4_DT_Q8_0 ? "g4_emb_gather_q8_0" : "g4_emb_gather_f32");
        need_b("g4_rope_neox");
        need_b(kv_q8 ? "g4_kv_write_q8" : "g4_kv_write_f16");
        need_b(kv_q8 ? "g4_attn_q8" : "g4_attn_f16");
        need_b("g4_geglu");
        for (int64_t il = 0; il < n_layer; il++) {
            const LT & t = lt[il];
            const TensorInfo * gm[] = {t.wq, t.wk, t.wv, t.wo, t.ffn_gate, t.ffn_up, t.ffn_down, t.router_w};
            for (const TensorInfo * g : gm)
                if (g) need_b(dense_gemv_kernel(g->dtype));
            if (t.router_w) {
                need_b("g4_router_topk");
                const TensorInfo * ex = t.experts;
                need_b(moe_gemv_kernel(ex->part_dtype(ex->dtype, 0)));
                need_b(moe_gemv_kernel(ex->part_dtype(ex->dtype, 1)));
                need_b(moe_gemv_kernel(ex->part_dtype(ex->dtype, 2)));
            }
        }
        need_b(output ? dense_gemv_kernel(output->dtype) : dense_gemv_kernel(tok_embd->dtype));
        for (const char * fn : fns) {
            if (!fn || !rt_.pipeline(fn)) {
                err = std::string("gemma4_forward_metal: kernel '") + (fn ? fn : "(null)") +
                      "' missing from Metal library";
                return false;
            }
        }
        for (const std::string & bn : bnames) {
            if (!rt_.pipeline(bn.c_str())) {
                err = "gemma4_forward_metal: kernel '" + bn + "' missing from Metal library";
                return false;
            }
        }
        logits_.assign(vocab_rows, 0.f);
        pos_ = 0;
        return true;
    }

    void reset() {
        for (id<MTLBuffer> b : kv_k_) memset(b.contents, 0, b.length);
        for (id<MTLBuffer> b : kv_v_) memset(b.contents, 0, b.length);
        pos_ = 0;
    }

    size_t resident_bytes() const { return mapped_bytes + scr_bytes + kv_bytes; }

    // ============================ dispatch ============================
    void enc_base(id<MTLComputeCommandEncoder> enc, const char * fn) {
        [enc setComputePipelineState:rt_.pipeline(fn)];
        [enc setBuffer:SCR_ offset:0 atIndex:0];
        [enc setBuffer:PACK_ offset:0 atIndex:1];
    }
    static void set_args(id<MTLComputeCommandEncoder> enc, const void * a, size_t len) {
        [enc setBytes:a length:len atIndex:2];
    }
    static void set_pos(id<MTLComputeCommandEncoder> enc, int pos) {
        [enc setBytes:&pos length:sizeof(pos) atIndex:3];
    }
    static MTLSize tg_ceil(uint64_t n, uint64_t per_tg) {
        return MTLSizeMake((n + per_tg - 1) / per_tg, 1, 1);
    }
    void dispatch_gemv(id<MTLComputeCommandEncoder> enc, const TensorInfo * t,
                       uint32_t x_off, uint32_t y_off, uint64_t nout, uint64_t nin) {
        G4GemvArgs A;
        A.woff = woff(t);
        A.x_off = x_off; A.y_off = y_off;
        A.nout = (unsigned int)nout; A.nin = (unsigned int)nin;
        A.rbytes = (unsigned int)(nin / (uint64_t)dtype_block_elems(t->dtype) *
                                  (uint64_t)dtype_type_size(t->dtype));
        enc_base(enc, dense_gemv_kernel(t->dtype));
        set_args(enc, &A, sizeof(A));
        [enc dispatchThreadgroups:tg_ceil(nout, 4u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    void dispatch_rms(id<MTLComputeCommandEncoder> enc, uint32_t src_off, uint32_t dst_off,
                      uint64_t w_off, uint32_t k, uint32_t rows, bool has_w, float post_scale) {
        G4RmsArgs A;
        A.src_off = src_off; A.dst_off = dst_off;
        A.woff = w_off;
        A.k = k; A.rows = rows;
        A.has_w = has_w ? 1u : 0u;
        A.eps = eps;
        A.post_scale = post_scale;
        enc_base(enc, "g4_rms_norm");
        set_args(enc, &A, sizeof(A));
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    }
    void dispatch_rope(id<MTLComputeCommandEncoder> enc, uint32_t off, uint32_t heads,
                       uint32_t head_dim, uint32_t nrot, uint64_t freq_off,
                       float freq_base, bool has_freqs, int pos) {
        if (heads == 0u || nrot < 2u) return;  // nothing to rotate
        G4RopeArgs A;
        A.off = off; A.n_heads = heads; A.head_dim = head_dim; A.n_rot = nrot;
        A.freq_off = freq_off; A.freq_base = freq_base; A.has_freqs = has_freqs ? 1 : 0;
        enc_base(enc, "g4_rope_neox");
        set_args(enc, &A, sizeof(A));
        set_pos(enc, pos);
        const uint64_t pairs = nrot / 2;
        [enc dispatchThreadgroups:MTLSizeMake((pairs + 7) / 8, (heads + 7) / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    }

    // ===================== Milestone 8A batch dispatchers =====================
    // All _b kernels take token positions as pos0 + row (never set_pos) and
    // rows-major SCR batch slots. Grids mirror the single-token grids with the
    // extra rows dimension; see engine/kernels/m64_gemma4_args.h for the PODs.
    static std::string bname(const char * base) { return std::string(base) + "_b"; }
    void enc_b(id<MTLComputeCommandEncoder> enc, const std::string & fn) {
        [enc setComputePipelineState:rt_.pipeline(fn.c_str())];
        [enc setBuffer:SCR_ offset:0 atIndex:0];
        [enc setBuffer:PACK_ offset:0 atIndex:1];
    }
    static uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }
    void dispatch_emb_b(id<MTLComputeCommandEncoder> enc, const int32_t * toks, uint32_t rows) {
        G4EmbBArgs A;
        A.woff = woff(tok_embd);
        A.dst_off = b_inp; A.dst_stride = (unsigned int)n_embd;
        A.n_embd = (unsigned int)n_embd;
        A.row_bytes = (unsigned int)(n_embd / (int64_t)dtype_block_elems(tok_embd->dtype) *
                                     (int64_t)dtype_type_size(tok_embd->dtype));
        A.dtype = (unsigned int)g4_dtype_code(tok_embd->dtype);
        A.scale = sqrtf((float)n_embd);
        A.rows = rows; A.pad0 = 0;
        enc_b(enc, bname(A.dtype == (unsigned int)G4_DT_Q8_0 ? "g4_emb_gather_q8_0"
                                                              : "g4_emb_gather_f32"));
        set_args(enc, &A, sizeof(A));
        [enc setBytes:toks length:rows * sizeof(int32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(n_embd, kTG), rows, 1)
            threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    }
    void dispatch_gemv_b(id<MTLComputeCommandEncoder> enc, const TensorInfo * t,
                         uint32_t x_off, uint32_t x_stride, uint32_t y_off, uint32_t y_stride,
                         uint64_t nout, uint64_t nin, uint32_t rows) {
        G4GemvBArgs A;
        A.woff = woff(t);
        A.x_off = x_off; A.y_off = y_off;
        A.x_stride = x_stride; A.y_stride = y_stride;
        A.nout = (unsigned int)nout; A.nin = (unsigned int)nin;
        A.rbytes = (unsigned int)(nin / (uint64_t)dtype_block_elems(t->dtype) *
                                  (uint64_t)dtype_type_size(t->dtype));
        A.rows = rows;
        enc_b(enc, bname(dense_gemv_kernel(t->dtype)));
        set_args(enc, &A, sizeof(A));
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(nout, 4u), ceil_div(rows, G4_RTILE), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    void dispatch_rope_b(id<MTLComputeCommandEncoder> enc, uint32_t off, uint32_t stride,
                         uint32_t heads, uint32_t head_dim, uint32_t nrot, uint64_t freq_off,
                         float freq_base, bool has_freqs, int pos0, uint32_t rows) {
        if (heads == 0u || nrot < 2u) return;  // nothing to rotate
        G4RopeBArgs A;
        A.off = off; A.stride = stride;
        A.n_heads = heads; A.head_dim = head_dim; A.n_rot = nrot;
        A.freq_off = freq_off; A.freq_base = freq_base; A.has_freqs = has_freqs ? 1 : 0;
        A.pos0 = pos0; A.rows = rows;
        enc_b(enc, bname("g4_rope_neox"));
        set_args(enc, &A, sizeof(A));
        const uint64_t pairs = nrot / 2;
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(pairs, 8u), ceil_div(heads, 8u), rows)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    }
    // KV write to either the layer ring (swa/pos0 from the layer) or the chunk
    // staging (swa=false, pos0=0) which mirrors the ring byte layout.
    void dispatch_kv_write_b(id<MTLComputeCommandEncoder> enc, uint32_t k_off, uint32_t v_off,
                             uint32_t k_stride, uint32_t v_stride, int64_t il,
                             bool swa, int pos0, uint32_t rows, bool to_staging) {
        const uint32_t hk = (uint32_t)hd_k[il], nk = (uint32_t)n_kv[il];
        G4KvBArgs A;
        A.k_off = k_off; A.v_off = v_off;
        A.k_stride = k_stride; A.v_stride = v_stride;
        A.n_kv = nk; A.hd = hk;
        A.ring_cap = kv_cap_[il]; A.rows = rows;
        A.is_swa = swa ? 1 : 0; A.pos0 = pos0;
        enc_b(enc, bname(kv_q8 ? "g4_kv_write_q8" : "g4_kv_write_f16"));
        set_args(enc, &A, sizeof(A));
        [enc setBuffer:(to_staging ? stg_k_ : kv_k_[il]) offset:0 atIndex:4];
        [enc setBuffer:(to_staging ? stg_v_ : kv_v_[il]) offset:0 atIndex:5];
        const uint64_t total = kv_q8 ? (uint64_t)nk * hk / 32u : (uint64_t)nk * hk;
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(total, kTG), rows, 1)
            threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    }
    void dispatch_attn_b(id<MTLComputeCommandEncoder> enc, int64_t il, int pos0, uint32_t rows) {
        const uint32_t hk = (uint32_t)hd_k[il], nk = (uint32_t)n_kv[il];
        G4AttnBArgs A;
        A.q_off = b_q; A.q_stride = (unsigned int)(n_head * hk);
        A.out_off = b_attn; A.out_stride = (unsigned int)(n_head * hk);
        A.n_head = (unsigned int)n_head; A.n_kv = nk;
        A.hd = hk; A.gqa_group = (unsigned int)(n_head / nk);
        A.ring_cap = kv_cap_[il]; A.rows = rows;
        A.is_swa = is_swa[il] ? 1 : 0; A.n_swa = (int)n_swa; A.pos0 = pos0;
        enc_b(enc, bname(kv_q8 ? "g4_attn_q8" : "g4_attn_f16"));
        set_args(enc, &A, sizeof(A));
        [enc setBuffer:kv_k_[il] offset:0 atIndex:4];
        [enc setBuffer:kv_v_[il] offset:0 atIndex:5];
        [enc setBuffer:stg_k_ offset:0 atIndex:6];
        [enc setBuffer:stg_v_ offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((uint64_t)n_head, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    }
    void dispatch_geglu_b(id<MTLComputeCommandEncoder> enc, uint32_t g_off, uint32_t u_off,
                          uint32_t dst_off, uint32_t n, uint32_t rows) {
        G4GeGLUBArgs A;
        A.g_off = g_off; A.u_off = u_off; A.dst_off = dst_off;
        A.g_stride = n; A.u_stride = n; A.dst_stride = n;  // slots are [rows][n]
        A.n = n; A.rows = rows;
        enc_b(enc, bname("g4_geglu"));
        set_args(enc, &A, sizeof(A));
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(n, kTG), rows, 1)
            threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    }
    void dispatch_moe_gemv_b(id<MTLComputeCommandEncoder> enc, DType dt, uint64_t layer_base,
                             uint64_t estr, uint64_t part_off, uint32_t x_off, uint32_t x_stride,
                             uint32_t y_off, uint32_t y_stride, uint32_t nout, uint32_t nin,
                             uint32_t rbytes, uint32_t k_slot, uint32_t rows, uint32_t accum) {
        G4MoeGemvBArgs A;
        A.layer_base = layer_base; A.expert_stride = estr; A.part_off = part_off;
        A.x_off = x_off; A.y_off = y_off;
        A.x_stride = x_stride; A.y_stride = y_stride;
        A.sel_ids_off = b_sel_ids; A.sel_w_off = b_sel_w;
        A.k_slot = k_slot; A.k_slots = (unsigned int)top_k;
        A.nout = nout; A.nin = nin; A.rbytes = rbytes;
        A.rows = rows; A.accum = accum;
        enc_b(enc, bname(moe_gemv_kernel(dt)));
        set_args(enc, &A, sizeof(A));
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(nout, 4u), ceil_div(rows, G4_MTILE), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    // rms/add/copy/scale replay on batch slots by growing rows: g4_rms_norm
    // addresses row r as base + r * k (batch stride == k) and the flat kernels
    // just take n = rows * stride.
    void dispatch_rms_rows(id<MTLComputeCommandEncoder> enc, uint32_t src_off, uint32_t dst_off,
                           uint64_t w_off, uint32_t k, uint32_t rows, bool has_w, float post_scale) {
        dispatch_rms(enc, src_off, dst_off, w_off, k, rows, has_w, post_scale);
    }
    void dispatch_flat(id<MTLComputeCommandEncoder> enc, const char * fn, uint32_t a_off,
                       uint32_t b_off, uint32_t c_off, uint64_t n) {
        // a_off/b_off/c_off used per-kernel: add = (a,b)->c, copy = a->b.
        if (strcmp(fn, "g4_copy") == 0) {
            G4CopyArgs A{a_off, b_off, (unsigned int)n};
            enc_base(enc, fn);
            set_args(enc, &A, sizeof(A));
        } else {
            G4AddArgs A{a_off, b_off, c_off, (unsigned int)n};
            enc_base(enc, fn);
            set_args(enc, &A, sizeof(A));
        }
        [enc dispatchThreadgroups:tg_ceil(n, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    }


    // ===================== one token, one command buffer =====================
    StepResult forward_token(int32_t token, bool want_logits) {
        StepResult r{nullptr, 0, -1};
        if (token < 0 || (int64_t)token >= vocab_eff) return r;
        if (pos_ >= ctx) reset();
        const int pos = (int)pos_;

        id<MTLCommandBuffer> cmd = [rt_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // input embedding * sqrt(n_embd)
        {
            G4EmbArgs A;
            A.woff = woff(tok_embd);
            A.dst_off = off_inp;
            A.n_embd = (unsigned int)n_embd;
            A.row_bytes = (unsigned int)(n_embd / (int64_t)dtype_block_elems(tok_embd->dtype) *
                                         (int64_t)dtype_type_size(tok_embd->dtype));
            A.dtype = (unsigned int)g4_dtype_code(tok_embd->dtype);
            A.scale = sqrtf((float)n_embd);
            enc_base(enc, A.dtype == (unsigned int)G4_DT_Q8_0 ? "g4_emb_gather_q8_0" : "g4_emb_gather_f32");
            set_args(enc, &A, sizeof(A));
            [enc setBytes:&token length:sizeof(token) atIndex:3];
            [enc dispatchThreadgroups:tg_ceil((uint64_t)n_embd, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }

        for (int64_t il = 0; il < n_layer; il++) layer(enc, il, pos);

        // final norm
        dispatch_rms(enc, off_inp, off_cur, woff(output_norm), (uint32_t)n_embd, 1u, true, 1.0f);

        if (want_logits) {
            const TensorInfo * lm = output ? output : tok_embd;
            dispatch_gemv(enc, lm, off_cur, off_logits, (uint64_t)vocab_rows, (uint64_t)n_embd);
            if (cap > 0.0f) {
                G4SoftcapArgs A;
                A.src_off = off_logits; A.dst_off = off_logits;
                A.n = (unsigned int)vocab_rows; A.cap = cap;
                enc_base(enc, "g4_softcap");
                set_args(enc, &A, sizeof(A));
                [enc dispatchThreadgroups:tg_ceil((uint64_t)vocab_rows, kTG)
                    threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
            }
            G4ArgmaxArgs A;
            A.src_off = off_logits; A.out_off = off_argmax_out;
            A.arena_off = off_argmax_arena; A.n = (unsigned int)vocab_eff;
            enc_base(enc, "g4_argmax");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }

        [enc endEncoding];
        rt_.wait(cmd);

        if (want_logits) {
            const float * s = (const float *)SCR_.contents;
            logits_.assign(s + off_logits, s + off_logits + vocab_rows);
            r.logits = logits_.data();
            r.n_logits = vocab_eff;
            r.argmax = ((const int32_t *)(s + off_argmax_out))[0];
        }
        pos_++;
        return r;
    }

private:
    // one transformer layer, mirroring Gemma4Forward::Impl::layer 1:1
    void layer(id<MTLComputeCommandEncoder> enc, int64_t il, int pos) {
        const LT & t = lt[il];
        const uint32_t hk = (uint32_t)hd_k[il], nk = (uint32_t)n_kv[il], nrot = (uint32_t)n_rot[il];
        const bool swa = is_swa[il];

        // ----- attention -----
        dispatch_rms(enc, off_inp, off_normed, woff(t.attn_norm), (uint32_t)n_embd, 1u, true, 1.0f);
        dispatch_gemv(enc, t.wq, off_normed, off_q, (uint64_t)n_head * hk, (uint64_t)n_embd);
        dispatch_rms(enc, off_q, off_q, woff(t.q_norm), hk, (uint32_t)n_head, true, 1.0f);
        const TensorInfo * rft = nullptr;
        if (!swa) rft = t.rope_freqs ? t.rope_freqs : rope_freqs_root;
        const float freq_base = swa ? freq_base_swa : freq_base_full;
        const uint64_t freq_off = rft ? woff(rft) : 0;
        dispatch_rope(enc, off_q, (uint32_t)n_head, hk, nrot, freq_off, freq_base, rft != nullptr, pos);
        dispatch_gemv(enc, t.wk, off_normed, off_k, (uint64_t)nk * hk, (uint64_t)n_embd);
        if (t.wv) {
            dispatch_gemv(enc, t.wv, off_normed, off_v, (uint64_t)nk * hk, (uint64_t)n_embd);
        } else {  // tied V = K BEFORE norm/RoPE
            G4CopyArgs A{off_k, off_v, (unsigned int)(nk * hk)};
            enc_base(enc, "g4_copy");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:tg_ceil(nk * (uint64_t)hk, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }
        dispatch_rms(enc, off_k, off_k, woff(t.k_norm), hk, nk, true, 1.0f);
        dispatch_rope(enc, off_k, nk, hk, nrot, freq_off, freq_base, rft != nullptr, pos);
        dispatch_rms(enc, off_v, off_v, 0, hk, nk, false, 1.0f);  // V: UNWEIGHTED norm

        // KV write into the per-layer ring
        {
            G4KvArgs A;
            A.k_off = off_k; A.v_off = off_v;
            A.n_kv = nk; A.hd = hk;
            A.ring_cap = kv_cap_[il];
            A.is_swa = swa ? 1 : 0;
            enc_base(enc, kv_q8 ? "g4_kv_write_q8" : "g4_kv_write_f16");
            set_args(enc, &A, sizeof(A));
            set_pos(enc, pos);
            [enc setBuffer:kv_k_[il] offset:0 atIndex:4];
            [enc setBuffer:kv_v_[il] offset:0 atIndex:5];
            const uint64_t total = kv_q8 ? (uint64_t)nk * hk / 32u : (uint64_t)nk * hk;
            [enc dispatchThreadgroups:tg_ceil(total, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }
        {
            G4AttnArgs A;
            A.q_off = off_q; A.out_off = off_attn;
            A.n_head = (unsigned int)n_head; A.n_kv = nk;
            A.hd = hk; A.gqa_group = (unsigned int)(n_head / nk);
            A.ring_cap = kv_cap_[il];
            A.is_swa = swa ? 1 : 0;
            A.n_swa = (int)n_swa;
            enc_base(enc, kv_q8 ? "g4_attn_q8" : "g4_attn_f16");
            set_args(enc, &A, sizeof(A));
            set_pos(enc, pos);
            [enc setBuffer:kv_k_[il] offset:0 atIndex:4];
            [enc setBuffer:kv_v_[il] offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((uint64_t)n_head, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }
        dispatch_gemv(enc, t.wo, off_attn, off_cur, (uint64_t)n_embd, (uint64_t)n_head * hk);
        dispatch_rms(enc, off_cur, off_cur, woff(t.post_attn_norm), (uint32_t)n_embd, 1u, true, 1.0f);
        {
            G4AddArgs A{off_cur, off_inp, off_attn_out, (unsigned int)n_embd};
            enc_base(enc, "g4_add");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:tg_ceil((uint64_t)n_embd, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }

        // ----- parallel shared MLP + routed MoE FFN -----
        dispatch_rms(enc, off_attn_out, off_normed, woff(t.ffn_norm), (uint32_t)n_embd, 1u, true, 1.0f);
        dispatch_gemv(enc, t.ffn_gate, off_normed, off_work_a, (uint64_t)n_ff, (uint64_t)n_embd);
        dispatch_gemv(enc, t.ffn_up, off_normed, off_work_b, (uint64_t)n_ff, (uint64_t)n_embd);
        {
            G4GeGLUArgs A{off_work_a, off_work_b, off_work_a, (unsigned int)n_ff};
            enc_base(enc, "g4_geglu");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:tg_ceil((uint64_t)n_ff, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }
        dispatch_gemv(enc, t.ffn_down, off_work_a, off_mlp, (uint64_t)n_embd, (uint64_t)n_ff);
        dispatch_rms(enc, off_mlp, off_mlp, woff(t.post_ffw_norm_1), (uint32_t)n_embd, 1u, true, 1.0f);

        if (t.router_w) {
            // router input: unweighted RMSNorm * 1/sqrt(n_embd) * ffn_gate_inp.scale
            // — ONE g4_rms_norm dispatch (post_scale + w)
            dispatch_rms(enc, off_attn_out, off_normed,
                         t.router_scale ? woff(t.router_scale) : 0,
                         (uint32_t)n_embd, 1u, t.router_scale != nullptr,
                         1.0f / sqrtf((float)n_embd));
            dispatch_gemv(enc, t.router_w, off_normed, off_router, (uint64_t)n_expert, (uint64_t)n_embd);
            {
                G4RouterArgs A;
                A.logits_off = off_router;
                A.sel_ids_off = off_sel_ids; A.sel_w_off = off_sel_w;
                A.n_expert = (unsigned int)n_expert; A.top_k = (unsigned int)top_k;
                A.down_scale_off = t.down_scale ? woff(t.down_scale) : 0;
                A.has_down_scale = t.down_scale ? 1 : 0;
                enc_base(enc, "g4_router_topk");
                set_args(enc, &A, sizeof(A));
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            dispatch_rms(enc, off_attn_out, off_normed, woff(t.pre_ffw_norm_2), (uint32_t)n_embd, 1u, true, 1.0f);
            // GPU-side zeroing of the accumulation buffer (a host memset here
            // would race / no-op against the in-flight encoder commands).
            {
                G4CopyArgs Z{off_zero, off_expert_acc, (unsigned int)n_embd};
                enc_base(enc, "g4_copy");
                set_args(enc, &Z, sizeof(Z));
                [enc dispatchThreadgroups:tg_ceil((uint64_t)n_embd, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
            }
            const TensorInfo * ex = t.experts;
            const int64_t pg = ex->part_bytes((n_ff_exp * n_embd / 256) * 144, 0);
            const int64_t pu = ex->part_bytes((n_ff_exp * n_embd / 256) * 144, 1);
            const DType dg = ex->part_dtype(ex->dtype, 0), du = ex->part_dtype(ex->dtype, 1),
                        dd = ex->part_dtype(ex->dtype, 2);
            const unsigned int be_g = (unsigned int)dtype_block_elems(dg), ts_g = (unsigned int)dtype_type_size(dg);
            const unsigned int be_u = (unsigned int)dtype_block_elems(du), ts_u = (unsigned int)dtype_type_size(du);
            const unsigned int be_d = (unsigned int)dtype_block_elems(dd), ts_d = (unsigned int)dtype_type_size(dd);
            const uint64_t exb = woff(ex), estr = (uint64_t)ex->expert_stride;
            // Per selected expert, sequentially: gate -> up -> geglu -> down(accum).
            // Reusing work_a/work_b across ALL of top_k before geglu would make
            // down read the last expert's raw gate output instead of the fused
            // activation, so the whole chain must be e-major, one expert at a time.
            for (uint32_t e = 0; e < (uint32_t)top_k; e++) {
                G4MoeGemvArgs A;
                A.layer_base = exb; A.expert_stride = estr;
                A.sel_ids_off = off_sel_ids; A.sel_w_off = off_sel_w;
                A.k_slot = e;
                A.x_off = off_normed; A.y_off = off_work_a;
                A.part_off = 0;
                A.nout = (uint32_t)n_ff_exp; A.nin = (uint32_t)n_embd;
                A.rbytes = (unsigned int)((uint64_t)n_embd / be_g * ts_g); A.accum = 0u;
                enc_base(enc, moe_gemv_kernel(dg));
                set_args(enc, &A, sizeof(A));
                [enc dispatchThreadgroups:tg_ceil((uint32_t)n_ff_exp, 4u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                A.part_off = (unsigned long)pg;
                A.y_off = off_work_b;
                A.rbytes = (unsigned int)((uint64_t)n_embd / be_u * ts_u);
                enc_base(enc, moe_gemv_kernel(du));
                set_args(enc, &A, sizeof(A));
                [enc dispatchThreadgroups:tg_ceil((uint32_t)n_ff_exp, 4u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                {
                    G4GeGLUArgs G{off_work_a, off_work_b, off_work_a, (unsigned int)n_ff_exp};
                    enc_base(enc, "g4_geglu");
                    set_args(enc, &G, sizeof(G));
                    [enc dispatchThreadgroups:tg_ceil((uint32_t)n_ff_exp, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
                }
                A.part_off = (unsigned long)(pg + pu);
                A.x_off = off_work_a; A.y_off = off_expert_acc;
                A.nout = (uint32_t)n_embd; A.nin = (uint32_t)n_ff_exp;
                A.rbytes = (unsigned int)((uint64_t)n_ff_exp / be_d * ts_d); A.accum = 1u;
                enc_base(enc, moe_gemv_kernel(dd));
                set_args(enc, &A, sizeof(A));
                [enc dispatchThreadgroups:tg_ceil((uint32_t)n_embd, 4u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            dispatch_rms(enc, off_expert_acc, off_moe, woff(t.post_ffw_norm_2), (uint32_t)n_embd, 1u, true, 1.0f);
            {
                G4AddArgs A{off_mlp, off_moe, off_cur, (unsigned int)n_embd};
                enc_base(enc, "g4_add");
                set_args(enc, &A, sizeof(A));
                [enc dispatchThreadgroups:tg_ceil((uint64_t)n_embd, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
            }
        } else {
            G4CopyArgs A{off_mlp, off_cur, (unsigned int)n_embd};
            enc_base(enc, "g4_copy");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:tg_ceil((uint64_t)n_embd, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }
        dispatch_rms(enc, off_cur, off_cur, woff(t.post_ffw_norm), (uint32_t)n_embd, 1u, true, 1.0f);
        {
            G4AddArgs A{off_cur, off_attn_out, off_inp, (unsigned int)n_embd};
            enc_base(enc, "g4_add");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:tg_ceil((uint64_t)n_embd, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }
        if (t.layer_scale) {
            G4ScaleArgs A;
            A.dst_off = off_inp; A.n = (unsigned int)n_embd; A.woff = woff(t.layer_scale);
            enc_base(enc, "g4_scale_scalar_buf");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:tg_ceil((uint64_t)n_embd, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }
    }

    // ===================== Milestone 8A: one batched chunk =====================
    // (public within Impl: Gemma4ForwardMetal::prefill drives it directly)
public:
    // Mirrors layer() row-for-row. Every batch slot is [rows][stride] with the
    // stride equal to the layer's vector width, so rms/add/copy/scale run flat
    // over rows * stride and each row replays the single-token arithmetic.
    void layer_b(id<MTLComputeCommandEncoder> enc, int64_t il, int pos0, uint32_t rows) {
        const LT & t = lt[il];
        const uint32_t hk = (uint32_t)hd_k[il], nk = (uint32_t)n_kv[il], nrot = (uint32_t)n_rot[il];
        const bool swa = is_swa[il];
        const uint64_t ne = (uint64_t)n_embd;

        // ----- attention -----
        dispatch_rms(enc, b_inp, b_normed, woff(t.attn_norm), (uint32_t)n_embd, rows, true, 1.0f);
        dispatch_gemv_b(enc, t.wq, b_normed, n_embd, b_q, n_head * hk, n_head * hk, ne, rows);
        dispatch_rms(enc, b_q, b_q, woff(t.q_norm), hk, rows * n_head, true, 1.0f);
        const TensorInfo * rft = nullptr;
        if (!swa) rft = t.rope_freqs ? t.rope_freqs : rope_freqs_root;
        const float freq_base = swa ? freq_base_swa : freq_base_full;
        const uint64_t freq_off = rft ? woff(rft) : 0;
        dispatch_rope_b(enc, b_q, n_head * hk, n_head, hk, nrot, freq_off, freq_base,
                        rft != nullptr, pos0, rows);
        dispatch_gemv_b(enc, t.wk, b_normed, n_embd, b_k, nk * hk, nk * hk, ne, rows);
        if (t.wv) {
            dispatch_gemv_b(enc, t.wv, b_normed, n_embd, b_v, nk * hk, nk * hk, ne, rows);
        } else {  // tied V = K BEFORE norm/RoPE
            dispatch_flat(enc, "g4_copy", b_k, b_v, 0, rows * (uint64_t)nk * hk);
        }
        dispatch_rms(enc, b_k, b_k, woff(t.k_norm), hk, rows * nk, true, 1.0f);
        dispatch_rope_b(enc, b_k, nk * hk, nk, hk, nrot, freq_off, freq_base,
                        rft != nullptr, pos0, rows);
        dispatch_rms(enc, b_v, b_v, 0, hk, rows * nk, false, 1.0f);  // V: UNWEIGHTED norm

        // Staging first (visible to this chunk's attention), then attention
        // (ring j < pos0 | staging pos0 <= j), and only AFTER attention the
        // rows land in the ring — a tight SWA ring (cap = n_swa + 1) must not
        // be stomped mid-chunk.
        dispatch_kv_write_b(enc, b_k, b_v, nk * hk, nk * hk, il, false, 0, rows, true);
        dispatch_attn_b(enc, il, pos0, rows);
        // Ring commit: only the last ring_cap rows for SWA layers. Earlier rows
        // share slots with later ones (pos % cap); writing both would be a race
        // with undefined winner, and no future row can reference them anyway.
        // Within the kept range slot = pos % cap is unique, so writes are safe.
        if (!swa) {
            dispatch_kv_write_b(enc, b_k, b_v, nk * hk, nk * hk, il, swa, pos0, rows, false);
        } else {
            const uint32_t cap = kv_cap_[il];
            const uint32_t keep = rows < cap ? rows : cap;
            const uint32_t skip = rows - keep;   // staging rows not committed
            dispatch_kv_write_b(enc, b_k + skip * nk * hk, b_v + skip * nk * hk,
                                nk * hk, nk * hk, il, swa, pos0 + (int)skip, keep, false);
        }

        dispatch_gemv_b(enc, t.wo, b_attn, n_head * hk, b_cur, n_embd, ne, n_head * hk, rows);
        dispatch_rms(enc, b_cur, b_cur, woff(t.post_attn_norm), (uint32_t)n_embd, rows, true, 1.0f);
        dispatch_flat(enc, "g4_add", b_cur, b_inp, b_attn_out, rows * ne);

        // ----- parallel shared MLP + routed MoE FFN -----
        dispatch_rms(enc, b_attn_out, b_normed, woff(t.ffn_norm), (uint32_t)n_embd, rows, true, 1.0f);
        dispatch_gemv_b(enc, t.ffn_gate, b_normed, n_embd, b_wa, n_ff, n_ff, ne, rows);
        dispatch_gemv_b(enc, t.ffn_up, b_normed, n_embd, b_wb, n_ff, n_ff, ne, rows);
        dispatch_geglu_b(enc, b_wa, b_wb, b_wa, n_ff, rows);
        dispatch_gemv_b(enc, t.ffn_down, b_wa, n_ff, b_mlp, n_embd, ne, n_ff, rows);
        dispatch_rms(enc, b_mlp, b_mlp, woff(t.post_ffw_norm_1), (uint32_t)n_embd, rows, true, 1.0f);

        if (t.router_w) {
            dispatch_rms(enc, b_attn_out, b_normed,
                         t.router_scale ? woff(t.router_scale) : 0,
                         (uint32_t)n_embd, rows, t.router_scale != nullptr,
                         1.0f / sqrtf((float)n_embd));
            dispatch_gemv_b(enc, t.router_w, b_normed, n_embd, b_router, n_expert,
                            n_expert, ne, rows);
            {
                G4RouterBArgs A;
                A.logits_off = b_router; A.logits_stride = (unsigned int)n_expert;
                A.sel_ids_off = b_sel_ids; A.sel_w_off = b_sel_w;
                A.n_expert = (unsigned int)n_expert; A.top_k = (unsigned int)top_k;
                A.down_scale_off = t.down_scale ? woff(t.down_scale) : 0;
                A.has_down_scale = t.down_scale ? 1 : 0;
                A.rows = rows;
                enc_b(enc, bname("g4_router_topk"));
                set_args(enc, &A, sizeof(A));
                [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            dispatch_rms(enc, b_attn_out, b_normed, woff(t.pre_ffw_norm_2),
                         (uint32_t)n_embd, rows, true, 1.0f);
            dispatch_flat(enc, "g4_copy", b_zero, b_eacc, 0, rows * ne);
            const TensorInfo * ex = t.experts;
            const int64_t pg = ex->part_bytes((n_ff_exp * n_embd / 256) * 144, 0);
            const int64_t pu = ex->part_bytes((n_ff_exp * n_embd / 256) * 144, 1);
            const DType dg = ex->part_dtype(ex->dtype, 0), du = ex->part_dtype(ex->dtype, 1),
                        dd = ex->part_dtype(ex->dtype, 2);
            const uint64_t exb = woff(ex), estr = (uint64_t)ex->expert_stride;
            const unsigned int rb_g = (unsigned int)((uint64_t)n_embd / dtype_block_elems(dg) * dtype_type_size(dg));
            const unsigned int rb_u = (unsigned int)((uint64_t)n_embd / dtype_block_elems(du) * dtype_type_size(du));
            const unsigned int rb_d = (unsigned int)((uint64_t)n_ff_exp / dtype_block_elems(dd) * dtype_type_size(dd));
            for (uint32_t e = 0; e < (uint32_t)top_k; e++) {
                dispatch_moe_gemv_b(enc, dg, exb, estr, 0,
                                    b_normed, n_embd, b_ea, n_ff_exp,
                                    n_ff_exp, n_embd, rb_g, e, rows, 0u);
                dispatch_moe_gemv_b(enc, du, exb, estr, (unsigned long)pg,
                                    b_normed, n_embd, b_eb, n_ff_exp,
                                    n_ff_exp, n_embd, rb_u, e, rows, 0u);
                dispatch_geglu_b(enc, b_ea, b_eb, b_ea, n_ff_exp, rows);
                dispatch_moe_gemv_b(enc, dd, exb, estr, (unsigned long)(pg + pu),
                                    b_ea, n_ff_exp, b_eacc, n_embd,
                                    n_embd, n_ff_exp, rb_d, e, rows, 1u);
            }
            dispatch_rms(enc, b_eacc, b_moe, woff(t.post_ffw_norm_2),
                         (uint32_t)n_embd, rows, true, 1.0f);
            dispatch_flat(enc, "g4_add", b_mlp, b_moe, b_cur, rows * ne);
        } else {
            dispatch_flat(enc, "g4_copy", b_mlp, b_cur, 0, rows * ne);
        }
        dispatch_rms(enc, b_cur, b_cur, woff(t.post_ffw_norm), (uint32_t)n_embd, rows, true, 1.0f);
        dispatch_flat(enc, "g4_add", b_cur, b_attn_out, b_inp, rows * ne);
        if (t.layer_scale) {
            G4ScaleArgs A;
            A.dst_off = b_inp; A.n = (unsigned int)(rows * ne); A.woff = woff(t.layer_scale);
            enc_base(enc, "g4_scale_scalar_buf");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:tg_ceil(rows * ne, kTG) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }
    }

    StepResult forward_chunk(const int32_t * toks, uint32_t rows, bool want_logits) {
        StepResult r{nullptr, 0, -1};
        if (rows == 0u) return r;
        if (pos_ >= ctx) reset();
        const int pos0 = (int)pos_;

        id<MTLCommandBuffer> cmd = [rt_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        dispatch_emb_b(enc, toks, rows);
        for (int64_t il = 0; il < n_layer; il++) layer_b(enc, il, pos0, rows);

        if (want_logits) {
            // final norm + LM head on the LAST chunk row only (into the
            // single-token slots so softcap/argmax/readback are byte-identical)
            dispatch_rms(enc, b_inp + (rows - 1) * n_embd, off_cur,
                         woff(output_norm), (uint32_t)n_embd, 1u, true, 1.0f);
            const TensorInfo * lm = output ? output : tok_embd;
            dispatch_gemv(enc, lm, off_cur, off_logits, (uint64_t)vocab_rows, (uint64_t)n_embd);
            if (cap > 0.0f) {
                G4SoftcapArgs A;
                A.src_off = off_logits; A.dst_off = off_logits;
                A.n = (unsigned int)vocab_rows; A.cap = cap;
                enc_base(enc, "g4_softcap");
                set_args(enc, &A, sizeof(A));
                [enc dispatchThreadgroups:tg_ceil((uint64_t)vocab_rows, kTG)
                    threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
            }
            G4ArgmaxArgs A;
            A.src_off = off_logits; A.out_off = off_argmax_out;
            A.arena_off = off_argmax_arena; A.n = (unsigned int)vocab_eff;
            enc_base(enc, "g4_argmax");
            set_args(enc, &A, sizeof(A));
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        }

        [enc endEncoding];
        rt_.wait(cmd);

        if (want_logits) {
            const float * s = (const float *)SCR_.contents;
            logits_.assign(s + off_logits, s + off_logits + vocab_rows);
            r.logits = logits_.data();
            r.n_logits = vocab_eff;
            r.argmax = ((const int32_t *)(s + off_argmax_out))[0];
        }
        pos_ += rows;
        return r;
    }
};

Gemma4ForwardMetal::Gemma4ForwardMetal() = default;
Gemma4ForwardMetal::~Gemma4ForwardMetal() = default;

bool Gemma4ForwardMetal::init(PackReader & pack, int context_len, std::string & err,
                              double arena_gb, const std::string & io_mode, bool kv_q8) {
    impl_ = std::make_unique<Impl>();
    impl_->pack_ = &pack;
    impl_->kv_q8 = kv_q8;
    return impl_->init(pack, context_len, err, arena_gb, io_mode);
}

StepResult Gemma4ForwardMetal::step(int32_t token) { return impl_->forward_token(token, true); }

StepResult Gemma4ForwardMetal::prefill(const std::vector<int32_t> & prompt) {
    StepResult r{nullptr, 0, -1};
    if (prompt.empty()) return r;
    if (prompt.size() == 1) return impl_->forward_token(prompt[0], true);
    // Any out-of-range token keeps the legacy per-token semantics (error step,
    // tokens consumed until then), so bail to the old loop.
    for (int32_t t : prompt)
        if (t < 0 || (int64_t)t >= impl_->vocab_eff) {
            for (size_t i = 0; i + 1 < prompt.size(); i++) r = impl_->forward_token(prompt[i], false);
            return impl_->forward_token(prompt.back(), true);
        }
    const uint32_t chunk = impl_->chunk_ > 0
        ? (uint32_t)std::min<int64_t>(impl_->chunk_, (int64_t)G4_BMAX) : (uint32_t)G4_BMAX;
    size_t i = 0;
    while (i < prompt.size()) {
        if (impl_->pos_ >= impl_->ctx) impl_->reset();
        uint32_t rows = (uint32_t)std::min<uint64_t>(chunk, prompt.size() - i);
        const int64_t room = impl_->ctx - impl_->pos_;
        if ((int64_t)rows > room) rows = (uint32_t)room;  // next round trips the reset
        r = impl_->forward_chunk(prompt.data() + i, rows, i + rows >= prompt.size());
        i += rows;
    }
    return r;
}

void Gemma4ForwardMetal::set_prefill_chunk(int n) {
    if (!impl_) return;
    impl_->chunk_ = n > 0 ? std::min<int64_t>(n, (int64_t)G4_BMAX) : 0;  // 0 => G4_BMAX
}

void Gemma4ForwardMetal::reset() { if (impl_) impl_->reset(); }
int64_t Gemma4ForwardMetal::current_pos() const { return impl_ ? impl_->pos_ : 0; }
int64_t Gemma4ForwardMetal::vocab_effective() const { return impl_ ? impl_->vocab_eff : 0; }
int64_t Gemma4ForwardMetal::vocab_logit_rows() const { return impl_ ? impl_->vocab_rows : 0; }
size_t Gemma4ForwardMetal::resident_bytes() const { return impl_ ? impl_->resident_bytes() : 0; }
std::string Gemma4ForwardMetal::arena_report() const { return ""; }

}  // namespace m64
