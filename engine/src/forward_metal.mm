// forward_metal.mm — GPU scheduler for the Inkling-2-Mini resident forward pass.
//
// One MTLCommandBuffer + one open compute encoder PER TOKEN (commit/wait exactly
// once per step). All weights are addressed zero-copy inside a single
// wrap_external() buffer over the mmap'd .m64pack (NoCopy file mapping — never
// mlock'd, splash-safe); all activations live in one preallocated scratch buffer
// referenced by float-index offsets in the POD arg structs (m64_p2a_args.h).
// Dispatch order reproduces forward_impl.hpp (CPU oracle) op-for-op.
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "forward_metal.hpp"
#include "metal_runtime.hpp"
#include "expert_arena.hpp"
#include "prefix_disk.hpp"
#include "../kernels/m64_p2a_args.h"

#include <cerrno>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <unordered_map>
#include <sys/mman.h>

#ifndef M64_KERNEL_DIR
#define M64_KERNEL_DIR "engine/kernels"
#endif
#ifndef M64_ARCHIVE_PATH
#define M64_ARCHIVE_PATH "build/m64_p2a.ar"
#endif

namespace m64 {

static_assert(sizeof(KVWriteF16Args) == 32, "KVWriteF16Args");
static_assert(sizeof(AttnF16Args) == 64, "AttnF16Args");

namespace {
constexpr int kNE = 2304, kNH = 24, kDH = 128, kKVD = 1024, kNREL = 16,
              kNEXP = 128, kTOPK = 8, kNF = 1152, kNPHYS = 24, kNSLOT = 48,
              kCH = 6656, kMAXS = 48, kHR = 3, kBMax = 128, kBBlkMax = 8,
              kAttnMaxSplits = 128;
constexpr uint32_t kBSpec = 9;  // Milestone 7B: max verify batch (anchor + 8 drafts)
constexpr uint32_t kPartBytes = 2820096u;  // (1152*2304/32)*34: gate|up|down parts
constexpr float kConvStatic = 0.447213595499957939281834733746255247f;  // 1/sqrt(5)
// Milestone 9C A/B switch: M64_NO_ATTN_GQA=1 restores the legacy 24-head
// split-K attention (3x duplicate KV loads) for speed comparison.
const bool g_no_attn_gqa = getenv("M64_NO_ATTN_GQA") != nullptr;
}  // namespace

struct InklingForwardMetal::Impl {
    MetalRuntime rt;
    PackReader * pk = nullptr;
    id<MTLBuffer> packBuf = nil, scr = nil;
    // FP16 KV cache (separate buffer; scratch stays ctx-independent).
    // SWA slots use a ring of n_swa+1 positions, full slots keep all ctx.
    id<MTLBuffer> kvBuf = nil;
    size_t kv_bytes = 0;
    // Q8_0 KV cache (Milestone 5B): 544 uint16 units (1088 B) per token slot
    // vs kKVD (1024 u16 = 2048 B) for FP16. kv_u16_per_pos is the kvBuf /
    // PrefixState::swa_kv / .m64kv storage stride in uint16 units; kKVD stays
    // the SCR activation vector width (kw.n, kw.k_stride, ...).
    bool kv_q8 = false;
    uint64_t kv_u16_per_pos = kKVD;
    uint32_t kv_cap[kNSLOT] = {};
    uint64_t kv_k_off[kNSLOT] = {};
    uint64_t kv_v_off[kNSLOT] = {};
    int ctx = 64;
    int64_t pos = 0;

    float f_eps = 1e-6f, f_qk_clip = 22.6f, f_rel_clip = 8.0f;
    float situ_beta = 4.0f, situ_lb = 25.0f, w_scale = 8.0f, cap = 30.0f;
    int64_t vocab_rows = 201024, vocab_eff = 200058;
    int64_t n_swa = 1024;
    float log_scale_floor = 8192.0f;   // attention.log_scale_floor (long-context tau)
    float log_scale_alpha = 0.1f;      // attention.log_scale_alpha
    bool fa16 = true;
    uint64_t exp_stride = 0;
    uint64_t pack_fp = 0;  // pack+hparam fingerprint for the SSD prefix cache

    struct LW { uint64_t wq = 0, wk = 0, wv = 0, wo = 0, wgate = 0, wsink = 0, wrel = 0,
                         qclip = 0, relclip = 0, relbias = 0, kconv = 0, vconv = 0,
                         kcovp = 0, vcovp = 0, postconv = 0, ffnconv = 0, router = 0,
                         probsb = 0, ffnnorm = 0, ffnq = 0, aquery = 0, experts = 0,
                         normattn = 0, nq = 0, nk = 0, nv = 0; int nq_rows = 0;
                         // Milestone 4B: per-layer MoE part layout from the pack. Defaults are the
                         // legacy uniform-Q8_0 layout (gate@0 | up@kPartBytes | down@2*kPartBytes);
                         // init() overwrites them from TensorInfo::part_dtype/part_bytes/expert_stride.
                         // moe_dt parts: 0=gate, 1=up, 2=down. moe_stride = per-layer expert stride.
                         DType moe_dt[3] = {DType::Q8_0, DType::Q8_0, DType::Q8_0};
                         uint64_t moe_off[3] = {0ull, (uint64_t)kPartBytes, 2ull * kPartBytes};
                         uint64_t moe_stride = 0;  // 0 until init()
                         bool moe_q4k = false;     // gate/up Q4_K + down Q5_0/Q5_1 dispatch flag
                         // Milestone 4B fix: Q4_K_S packs store the small dense attn projections
                         // (attn_sink_proj / attn_k_conv_proj / attn_v_conv_proj) as f32 instead
                         // of q8_0. false keeps the legacy uniform-Q8_0 dispatch 100% intact.
                         bool wsink_f32 = false, kcovp_f32 = false, vcovp_f32 = false;
                         } phys[kNPHYS];
    struct LL { uint64_t msgq = 0, down[4] = {0}, gain[4] = {0}, forg[4] = {0}; } loop[12];
    int s_phys[kNSLOT]; bool s_swa[kNSLOT]; int s_loop[kNSLOT];

    uint64_t tok_embd = 0, tok_norm = 0, outw = 0, out_norm = 0, out_q = 0;

    struct Off {
        uint32_t emb, cur, mixed, ctrlout, stn, gcat, msgs, bank_out, prelude_out;
        uint32_t blocks, bank, prelude_blk, gb, fb, ctrl_hh;
        uint32_t Q, K, V2, Rr, sinkb, gateb, attnb, dynb, tA, tB;
        uint32_t qsrc, ksrc, vsrc, fin;
        uint32_t probs, sel, wgt, gate2, ubuf, acc, down_tmp;
        uint32_t sc, logits, amax;
        uint32_t xqA, dxA, xqB, dxB, xqC, dxC, xqD, dxD, xqE, dxE, hq, dh;
        uint32_t rs, arena_sel, attn_part_o, attn_part_ms;
        // Milestone 3C batched prefill scratch (kBMax=128, kBBlkMax=8, 4-float
        // aligned, placed cleanly after single-token layout + o.rs).
        uint32_t b_tok, b_emb, b_cur, b_mixed, b_ctrlout, b_msgs, b_bank_out, b_prelude_out;
        uint32_t b_blocks, b_bank, b_prelude_blk, b_n3;
        uint32_t b_Q, b_K, b_V2, b_Vn, b_Rr, b_sinkb, b_gateb, b_attnb, b_dynb, b_tA, b_tB;
        uint32_t b_fin, b_probs, b_sel, b_wgt;
        uint32_t b_gate2, b_ubuf, b_situ, b_down_tmp, b_acc;
        uint32_t b_cnt, b_off, b_list, b_wtab;
        uint32_t b_lgx, b_amax;  // Milestone 7B: per-row verify logits / argmax
        uint32_t b_stn, b_ctrl_gcat, b_ctrl_hh;  // Milestone 9B: batched mix/ctrl scratch
        uint32_t total = 0;
    } o;
    int prefill_chunk = kBMax;

    static uint32_t fo(uint32_t & c, uint32_t n) { uint32_t r = c; c += n; return r; }

    // ---- Milestone 14: background mmap pack page warming -------------------
    // ~Impl MUST stop+join the prefetch thread BEFORE the PackReader unmaps
    // the pack: in main.cpp `pk` is declared before `fwd`, so fwd (and this
    // Impl) is always destructed first while the mapping is still alive.
    std::atomic<bool> stop_prefetch_{false};
    std::thread prefetch_thread_;
    ~Impl() {
        stop_prefetch_.store(true, std::memory_order_relaxed);
        if (prefetch_thread_.joinable()) prefetch_thread_.join();
    }

    // ---- Phase 2B arena state (arena_mode only) ----
    bool arena_mode = false;
    double arena_gb = 0.0;
    std::string io_mode = "pread";
    LockedRegion resident_;                    // compact dense weights (mlock'd)
    std::unique_ptr<ExpertArena> arena;        // Hot Expert LRU Arena (mlock'd)
    id<MTLBuffer> residentBuf = nil, expertBuf = nil;
    size_t arena_resident_bytes = 0;

    bool init(PackReader & pack, int context_len, std::string & err,
              double arena_gb_in = 0.0, const std::string & io_mode_in = "pread",
              bool kv_q8_in = false) {
        arena_gb = arena_gb_in;
        io_mode = io_mode_in;
        arena_mode = arena_gb > 0.0;
        kv_q8 = kv_q8_in;
        kv_u16_per_pos = kv_q8 ? 544ull : (uint64_t)kKVD;
        pk = &pack;
        if (pack.header().arch != "inkling2_loop") { err = "metal: arch != inkling2_loop"; return false; }
        auto hi = [&](const char * k, int64_t f) { return pack.hparam_i64(std::string("inkling2_loop.") + k, f); };
        auto hf = [&](const char * k, double f) { return pack.hparam_f64(std::string("inkling2_loop.") + k, f); };
        if (hi("embedding_length", 0) != kNE || hi("block_count", 0) != kNPHYS ||
            hi("expert_count", 0) != kNEXP || hi("expert_used_count", 0) != kTOPK) {
            err = "metal: unexpected hparams"; return false;
        }
        f_eps = (float)hf("attention.layer_norm_rms_epsilon", 1e-6);
        f_qk_clip = (float)hf("attention.qk_clip", 22.627416997969522);
        f_rel_clip = (float)hf("attention.relative_clip", 8.0);
        situ_beta = (float)hf("activation.situ_beta", 4.0);
        situ_lb = (float)hf("activation.situ_linear_beta", 25.0);
        w_scale = (float)hf("expert_weights_scale", 8.0);
        cap = (float)hf("final_logit_softcapping", 30.0);
        vocab_rows = hi("vocab_size", 201024);
        vocab_eff = hi("vocab_size_unpadded", 200058);
        n_swa = hi("attention.sliding_window", 1024);
        log_scale_floor = (float)hf("attention.log_scale_floor", 8192.0);
        log_scale_alpha = (float)hf("attention.log_scale_alpha", 0.1);
        if (log_scale_floor < 1.0f) log_scale_floor = 8192.0f;
        // Pack fingerprint for the SSD prefix cache: identifies the pack bytes
        // and every hparam that can change KV/ring values. Deliberately NOT
        // hashing `ctx`, so a prefix cached at --ctx 131072 still loads after a
        // restart at --ctx 32768 (the n_tokens < max_ctx check guards capacity).
        {
            Xxh64 fh;
            auto feed = [&fh](uint64_t v) { fh.update(&v, sizeof(v)); };
            feed((uint64_t)pack.file_size());
            feed((uint64_t)pack.n_tensors());
            feed(pack.header().blob_xxh64);
            feed((uint64_t)vocab_eff);
            feed((uint64_t)n_swa);
            uint32_t bits = 0;
            memcpy(&bits, &log_scale_floor, sizeof(bits));
            feed(bits);
            memcpy(&bits, &log_scale_alpha, sizeof(bits));
            feed(bits);
            // Fold the KV storage dtype in ONLY for Q8_0: existing FP16
            // .m64kv fingerprints stay byte-identical, while FP16 and Q8_0
            // files can never collide (Q8 slots are 544 u16 units per pos).
            if (kv_q8) feed(544ull);
            pack_fp = fh.digest();
        }
        if (getenv("M64_NO_FA16")) fa16 = false;
        if (const char * pc = getenv("M64_PREFILL_CHUNK")) {
            int v = atoi(pc);
            prefill_chunk = v < 1 ? 1 : (v > kBMax ? kBMax : v);
        }

        std::vector<int64_t> pattern;
        auto it = pack.header().hparams.find("inkling2_loop.attention.sliding_window_pattern");
        if (it == pack.header().hparams.end() || it->second->type != JValue::Arr) {
            err = "metal: missing sliding_window_pattern"; return false;
        }
        for (auto & v : it->second->arr) pattern.push_back(v->as_int());
        bool swa_phys[kNPHYS];
        for (int l = 0; l < kNPHYS; l++) swa_phys[l] = pattern[l] != 0;
        for (int il = 0; il < kNSLOT; il++) {
            int p;
            p = (il < kNPHYS) ? il : (6 + (il - kNPHYS) % 12);  // == forward_impl.hpp slot_phys()
            s_phys[il] = p; s_swa[il] = swa_phys[p]; s_loop[il] = (p >= 6 && p < 18) ? p - 6 : -1;
        }

        auto off = [&](const char * n) -> uint64_t {
            const TensorInfo * t = pk->find(n);
            if (!t) { static std::string miss; miss = std::string("metal: missing tensor ") + n; err = miss; throw std::runtime_error(miss); }
            return (uint64_t)t->offset;
        };
        // Milestone 4B fix: classify a small dense projection tensor: 1=f32, 0=q8_0, -1=bad.
        auto proj_dt = [&](const std::string & n) -> int {
            const TensorInfo * t = pk->find(n);
            if (!t) return -1;
            if (t->dtype == DType::F32) return 1;
            if (t->dtype == DType::Q8_0) return 0;
            return -1;
        };
        try {
            tok_embd = off("token_embd.weight"); tok_norm = off("token_embd_norm.weight");
            outw = off("output.weight"); out_norm = off("output_norm.weight");
            out_q = off("output_res_query.weight");
            for (int p = 0; p < kNPHYS; p++) {
                LW & L = phys[p];
                std::string b = "blk." + std::to_string(p) + ".";
                L.wq = off((b + "attn_q.weight").c_str());
                L.wk = off((b + "attn_k.weight").c_str());
                L.wv = off((b + "attn_v.weight").c_str());
                L.wo = off((b + "attn_output.weight").c_str());
                L.wgate = off((b + "attn_gate.weight").c_str());
                L.wsink = off((b + "attn_sink_proj.weight").c_str());
                {  // f32-vs-q8_0 dispatch flag (Q4_K_S packs store this as f32).
                    const int dt = proj_dt(b + "attn_sink_proj.weight");
                    if (dt < 0) { err = "metal: attn_sink_proj.weight missing/bad dtype (need f32 or q8_0) in " + b; return false; }
                    L.wsink_f32 = dt == 1;
                }
                L.wrel = off((b + "attn_rel.weight").c_str());
                L.qclip = off((b + "attn_q_clip.weight").c_str());
                L.relclip = off((b + "attn_rel_clip.weight").c_str());
                L.relbias = off((b + "attn_rel_bias.weight").c_str());
                L.kconv = off((b + "attn_k_conv.weight").c_str());
                L.vconv = off((b + "attn_v_conv.weight").c_str());
                L.postconv = off((b + "attn_post_conv.weight").c_str());
                L.ffnconv = off((b + "ffn_post_conv.weight").c_str());
                L.router = off((b + "ffn_gate_inp.weight").c_str());
                L.probsb = off((b + "exp_probs_b.bias").c_str());
                L.ffnnorm = off((b + "ffn_norm.weight").c_str());
                L.ffnq = off((b + "ffn_res_query.weight").c_str());
                if (const TensorInfo * aq = pack.find(b + "attn_res_query.weight")) {
                    L.aquery = (uint64_t)aq->offset;
                    // SWA blocks store this 1D as shape=[kNE]: shape[0] is the row
                    // WIDTH, not a row count. Only 2D shapes carry rows; for 1D,
                    // derive rows from the f32 byte length (length/kNE).
                    L.nq_rows = (aq->shape.size() >= 2 && aq->shape[0] > 0)
                        ? (int)aq->shape[0] : (int)(aq->length / (4 * (int64_t)kNE));
                    const int want = swa_phys[p] ? 1 : 3;
                    if (L.nq_rows != want || L.nq_rows > 4) {
                        err = "metal: blk." + std::to_string(p) + " attn_res_query rows=" +
                              std::to_string(L.nq_rows) + " expected " + std::to_string(want);
                        throw std::runtime_error(err);
                    }
                }
                if (const TensorInfo * t = pack.find(b + "attn_norm.weight")) L.normattn = (uint64_t)t->offset;
                if (const TensorInfo * t = pack.find(b + "attn_q_inp_norm.weight")) {
                    L.nq = (uint64_t)t->offset;
                    L.nk = off((b + "attn_k_inp_norm.weight").c_str());
                    L.nv = off((b + "attn_v_inp_norm.weight").c_str());
                    L.kcovp = off((b + "attn_k_conv_proj.weight").c_str());
                    L.vcovp = off((b + "attn_v_conv_proj.weight").c_str());
                    {  // f32-vs-q8_0 dispatch flags (Q4_K_S packs store these as f32).
                        const int dk = proj_dt(b + "attn_k_conv_proj.weight");
                        const int dv = proj_dt(b + "attn_v_conv_proj.weight");
                        if (dk < 0 || dv < 0) { err = "metal: attn _conv_proj.weight missing/bad dtype (need f32 or q8_0) in " + b; return false; }
                        L.kcovp_f32 = dk == 1;
                        L.vcovp_f32 = dv == 1;
                    }
                }
                const TensorInfo * einfo = pack.find(b + "ffn_experts.weight");
                if (!einfo || !einfo->is_expert) { err = "metal: missing experts " + b; return false; }
                {  // Milestone 4B: per-part dtypes / byte offsets / expert stride.
                    const DType bdt = einfo->dtype;
                    DType dts[3];
                    for (int part = 0; part < 3; part++) dts[part] = einfo->part_dtype(bdt, part);
                    const bool all_q8 = dts[0] == DType::Q8_0 && dts[1] == DType::Q8_0 && dts[2] == DType::Q8_0;
                    const bool q4k_combo = dts[0] == DType::Q4_K && dts[1] == DType::Q4_K &&
                                           (dts[2] == DType::Q5_0 || dts[2] == DType::Q5_1);
                    if (!all_q8 && !q4k_combo) {
                        err = "metal: unsupported expert part dtype combo in " + b +
                              " (need all q8_0 or gate/up q4_k + down q5_0/q5_1)"; return false;
                    }
                    for (int part = 0; part < 3; part++) L.moe_dt[part] = dts[part];
                    if (einfo->part_bytes_gate > 0 && einfo->part_bytes_up > 0 && einfo->part_bytes_down > 0) {
                        L.moe_off[0] = 0;
                        L.moe_off[1] = (uint64_t)einfo->part_bytes_gate;
                        L.moe_off[2] = L.moe_off[1] + (uint64_t)einfo->part_bytes_up;
                        if (L.moe_off[2] + (uint64_t)einfo->part_bytes_down > (uint64_t)einfo->expert_stride) {
                            err = "metal: expert part bytes exceed expert stride in " + b; return false;
                        }
                    }
                    if (q4k_combo && (L.moe_off[1] % 4u || L.moe_off[2] % 4u || einfo->expert_stride % 4)) {
                        err = "metal: q4_k/q5 expert layout needs 4-byte aligned offsets in " + b; return false;
                    }
                    L.moe_stride = (uint64_t)einfo->expert_stride;
                    L.moe_q4k = q4k_combo;
                    if (L.moe_stride > (uint64_t)exp_stride) exp_stride = L.moe_stride;  // arena slot stride = max
                }
            }
            for (int li = 0; li < 12; li++) {
                std::string b = "blk." + std::to_string(li + 6) + ".";
                LL & lp = loop[li];
                lp.msgq = off((b + "loop_msg_query.weight").c_str());
                const char * nm[4] = {"q", "k", "v", "ffn"};
                for (int c = 0; c < 4; c++) {
                    std::string s = b + "loop_" + nm[c] + "_ctrl_";
                    lp.down[c] = off((s + "down.weight").c_str());
                    lp.gain[c] = off((s + "gain.weight").c_str());
                    lp.forg[c] = off((s + "forget.weight").c_str());
                }
            }
        } catch (const std::exception &) { return false; }

        const auto & eidx = pack.header().expert_index;
        if (!eidx.present || (int)eidx.layer_base.size() != kNPHYS) { err = "metal: expert index"; return false; }
        // exp_stride is the arena SLOT stride: max over the header value and every
        // per-layer stride collected in the init loop above (uniform Q8_0 packs: equal).
        if ((uint64_t)eidx.expert_stride > exp_stride) exp_stride = (uint64_t)eidx.expert_stride;
        for (int p = 0; p < kNPHYS; p++) phys[p].experts = (uint64_t)eidx.layer_base[p];

        // ---- Phase 2B: compact resident weights (arena_mode) ----
        // Pack all NON-expert tensors into one mlock'd region, loaded through a
        // clean O_RDONLY|F_NOCACHE fd (zero OS page-cache bloat; the mmap'd
        // pack view is never touched for cold experts). Dense tensors are then
        // remapped to residentBuf-local offsets. L.experts keeps the raw FILE
        // byte offset for SSD streaming.
        if (arena_mode) {
            auto al64 = [](int64_t n) { return (uint64_t)((n + 63) & ~63LL); };
            uint64_t total = 64;  // 64B reserve so no real tensor maps to offset 0 (null sentinel)
            std::vector<const TensorInfo *> dense;
            for (size_t i = 0; i < pack.n_tensors(); i++) {
                const TensorInfo & t = pack.tensor(i);
                if (t.is_expert) continue;
                dense.push_back(&t);
                total += al64(t.length);
            }
            if (!resident_.alloc((size_t)total, err)) return false;
            int rfd = ::open(pack.path().c_str(), O_RDONLY);
            if (rfd < 0) { err = "metal: arena open " + pack.path(); return false; }
            if (fcntl(rfd, F_NOCACHE, 1) != 0)
                fprintf(stderr, "m64metal: WARNING fcntl(F_NOCACHE) failed: %s\n", strerror(errno));
            std::unordered_map<uint64_t, uint64_t> resmap;
            uint64_t ro = 64;
            bool rdfail = false;
            for (const TensorInfo * t : dense) {
                uint8_t * dst = (uint8_t *)resident_.ptr() + ro;
                int64_t left = t->length, done = 0;
                while (left > 0) {
                    ssize_t n = ::pread(rfd, dst + done, (size_t)left, (off_t)(t->offset + done));
                    if (n > 0) { done += n; left -= n; continue; }
                    if (n < 0 && errno == EINTR) continue;
                    rdfail = true;
                    break;
                }
                if (rdfail) { err = "metal: resident pread failed for " + t->name; break; }
                resmap[(uint64_t)t->offset] = ro;
                ro += al64(t->length);
            }
            ::close(rfd);
            if (rdfail) return false;
            arena_resident_bytes = (size_t)total;
            auto remap = [&](uint64_t & x) {
                if (x == 0) return;
                auto it = resmap.find(x);
                if (it == resmap.end()) { err = "metal: dense offset missing from resident map"; return; }
                x = it->second;
            };
            std::vector<uint64_t *> fields = {&tok_embd, &tok_norm, &outw, &out_norm, &out_q};
            for (int p = 0; p < kNPHYS; p++) {
                LW & L = phys[p];
                uint64_t * f[] = {&L.wq, &L.wk, &L.wv, &L.wo, &L.wgate, &L.wsink, &L.wrel,
                                  &L.qclip, &L.relclip, &L.relbias, &L.kconv, &L.vconv,
                                  &L.kcovp, &L.vcovp, &L.postconv, &L.ffnconv, &L.router,
                                  &L.probsb, &L.ffnnorm, &L.ffnq, &L.aquery, &L.normattn,
                                  &L.nq, &L.nk, &L.nv};
                for (uint64_t * q : f) fields.push_back(q);
            }
            for (int li = 0; li < 12; li++) {
                LL & lp = loop[li];
                fields.push_back(&lp.msgq);
                for (int c = 0; c < 4; c++) {
                    fields.push_back(&lp.down[c]);
                    fields.push_back(&lp.gain[c]);
                    fields.push_back(&lp.forg[c]);
                }
            }
            for (uint64_t * q : fields) { remap(*q); if (!err.empty()) return false; }
        }

        if (context_len < 4) context_len = 4;
        if (context_len > 131072) context_len = 131072;
        ctx = context_len;

        uint32_t c = 0;
        o.emb = fo(c, kNE); o.cur = fo(c, kNE); o.mixed = fo(c, 3 * kNE);
        o.ctrlout = fo(c, 3 * kNE); o.stn = fo(c, kMAXS * kNE); o.gcat = fo(c, 8 * kNE);
        o.msgs = fo(c, 4 * kNE); o.bank_out = fo(c, kNE); o.prelude_out = fo(c, kNE);
        o.blocks = fo(c, kMAXS * kNE); o.bank = fo(c, kMAXS * kNE); o.prelude_blk = fo(c, kMAXS * kNE);
        o.gb = fo(c, 4 * kNE); o.fb = fo(c, 4 * kNE); o.ctrl_hh = fo(c, 4 * 64);
        o.Q = fo(c, kNH * kDH); o.K = fo(c, kKVD); o.V2 = fo(c, kKVD); o.Rr = fo(c, kNH * kNREL);
        o.sinkb = fo(c, kNH); o.gateb = fo(c, kNH * kDH); o.attnb = fo(c, kNH * kDH);
        o.dynb = fo(c, 64); o.tA = fo(c, kNH * kDH); o.tB = fo(c, kNE);
        o.qsrc = fo(c, kNE); o.ksrc = fo(c, kNE); o.vsrc = fo(c, kNE); o.fin = fo(c, kNE);
        o.probs = fo(c, kNEXP); o.sel = fo(c, kTOPK); o.wgt = fo(c, kTOPK);
        o.gate2 = fo(c, kTOPK * kNF); o.ubuf = fo(c, kTOPK * kNF); o.acc = fo(c, kNE);
        o.down_tmp = fo(c, kTOPK * kNE);
        o.sc = fo(c, kMAXS * kMAXS); o.logits = fo(c, 201024); o.amax = fo(c, 4);
        o.xqA = fo(c, 576); o.dxA = fo(c, 72); o.xqB = fo(c, 576); o.dxB = fo(c, 72);
        o.xqC = fo(c, 576); o.dxC = fo(c, 72); o.xqD = fo(c, 768); o.dxD = fo(c, 96);
        o.xqE = fo(c, 576); o.dxE = fo(c, 72); o.hq = fo(c, kTOPK * 288); o.dh = fo(c, kTOPK * 36);
        o.rs = fo(c, kNSLOT * kCH * kHR);
        o.arena_sel = fo(c, kTOPK);
        auto fa = [&](uint32_t n) { c = (c + 3u) & ~3u; uint32_t r = c; c += n; return r; };
        o.attn_part_o = fa(kNH * kAttnMaxSplits * kDH);
        o.attn_part_ms = fa(kNH * kAttnMaxSplits * 2);
        o.b_tok = fa(kBMax);
        o.b_emb = fa(kBMax * kNE);
        o.b_cur = fa(kBMax * kNE);
        o.b_mixed = fa(kBMax * 3 * kNE);
        o.b_ctrlout = fa(kBMax * 3 * kNE);
        o.b_msgs = fa(kBMax * 4 * kNE);
        o.b_bank_out = fa(kBMax * kNE);
        o.b_prelude_out = fa(kBMax * kNE);
        o.b_blocks = fa(kBMax * kBBlkMax * kNE);
        o.b_bank = fa(kBMax * kBBlkMax * kNE);
        o.b_prelude_blk = fa(kBMax * kNE);
        o.b_n3 = fa(kBMax * 3 * kNE);
        // Milestone 9B: per-token mix stn rows + per-token ctrl gcat/hh so
        // all B tokens run in one dispatch without SCR collisions.
        o.b_stn = fa(kBMax * (kBBlkMax + 1) * kNE);
        o.b_ctrl_gcat = fa(kBMax * 6 * kNE);
        o.b_ctrl_hh = fa(kBMax * 4 * 64u);
        o.b_Q = fa(kBMax * kNH * kDH);
        o.b_K = fa(kBMax * kKVD);
        o.b_V2 = fa(kBMax * kKVD);
        o.b_Vn = fa(kBMax * kKVD);
        o.b_Rr = fa(kBMax * kNH * kNREL);
        o.b_sinkb = fa(kBMax * kNH);
        o.b_gateb = fa(kBMax * kNH * kDH);
        o.b_attnb = fa(kBMax * kNH * kDH);
        o.b_dynb = fa(kBMax * 32);
        o.b_tA = fa(kBMax * kNH * kDH);
        o.b_tB = fa(kBMax * kNE);
        o.b_fin = fa(kBMax * kNE);
        o.b_probs = fa(kBMax * kNEXP);
        o.b_sel = fa(kBMax * kTOPK);
        o.b_wgt = fa(kBMax * kTOPK);
        o.b_gate2 = fa(kBMax * kTOPK * kNF);
        o.b_ubuf = fa(kBMax * kTOPK * kNF);
        o.b_situ = fa(kBMax * kTOPK * kNF);
        o.b_down_tmp = fa(kBMax * kTOPK * kNE);
        o.b_acc = fa(kBMax * kNE);
        o.b_cnt = fa(kNEXP);
        o.b_off = fa(kNEXP);
        o.b_list = fa(kBMax * kTOPK);
        o.b_wtab = fa(kNEXP);
        // Milestone 7B: full-row logits + argmax for speculative verify.
        o.b_lgx = fa(kBSpec * 201024u);
        o.b_amax = fa(kBSpec);
        o.total = c;

        // ---- KV cache layout (FP16 or Q8_0; separate buffer, scratch ctx-free) ----
        // SWA slots keep a ring of n_swa+1 positions (at pos, distances 0..n_swa
        // are attended -> n_swa+1 distinct j); full-attention slots keep all ctx.
        // At ctx=131072: 40 SWA rings + 8 full slots ~= 4.16 GiB K+V in FP16
        // (46.9% of that with the Q8_0 layout).
        {
            uint64_t kv_elems = 0;
            for (int il = 0; il < kNSLOT; il++) {
                const int64_t cap = s_swa[il]
                    ? ((int64_t)ctx < n_swa + 1 ? (int64_t)ctx : n_swa + 1)
                    : (int64_t)ctx;
                kv_cap[il] = (uint32_t)cap;
                kv_k_off[il] = kv_elems; kv_elems += (uint64_t)cap * kv_u16_per_pos;
                kv_v_off[il] = kv_elems; kv_elems += (uint64_t)cap * kv_u16_per_pos;
            }
            kv_bytes = (size_t)kv_elems * sizeof(uint16_t);
        }

        std::string kerr;
        if (!rt.init(M64_KERNEL_DIR, M64_ARCHIVE_PATH, kerr)) { err = "metal runtime: " + kerr; return false; }
        if (arena_mode) {
            const size_t scratch_bytes = (size_t)o.total * 4 + kv_bytes;
            const size_t cap_bytes = (size_t)(arena_gb * 1073741824.0);
            if (cap_bytes <= resident_.bytes() + scratch_bytes) {
                err = "metal: --arena-gb " + std::to_string(arena_gb) +
                      " GiB too small for resident weights + scratch/KV (need > " +
                      std::to_string((resident_.bytes() + scratch_bytes) / 1e9) + " GB)";
                return false;
            }
            const size_t budget = cap_bytes - resident_.bytes() - scratch_bytes;
            ExpertArenaCfg acfg;
            acfg.pack_path = pack.path();
            for (int p = 0; p < kNPHYS; p++)
                acfg.layer_base.push_back((uint64_t)pack.header().expert_index.layer_base[p]);
            acfg.exp_stride = exp_stride;  // slot spacing = max pack stride over layers
            for (int p = 0; p < kNPHYS; p++)
                acfg.layer_stride.push_back(phys[p].moe_stride);  // per-layer pack stride
            acfg.n_experts = kNEXP;
            size_t slots = budget / (size_t)exp_stride;
            const size_t total_experts = (size_t)kNPHYS * kNEXP;
            if (slots > total_experts) slots = total_experts;  // arena covers whole model
            acfg.num_slots = slots;
            acfg.io_mode = io_mode;
            arena = std::make_unique<ExpertArena>();
            if (!arena->create(acfg, err)) { err = "arena: " + err; return false; }
            void * mbuf = arena->mtl_buffer();
            if (mbuf) {
                expertBuf = (__bridge id<MTLBuffer>)mbuf;  // owned by the arena (mtlio DMA target)
            } else {
                expertBuf = rt.wrap_external(arena->base(), arena->total_bytes(), @"m64-arena");
            }
            if (!expertBuf) { err = "metal: wrap arena failed"; return false; }
            residentBuf = rt.wrap_external(resident_.ptr(), resident_.bytes(), @"m64-resident");
            if (!residentBuf) { err = "metal: wrap resident failed"; return false; }
            fprintf(stderr,
                    "arena: %zu slots x %.2f MB, expert budget %.2f GB, resident weights %.2f GB "
                    "(mlock pinned), io=%s\n",
                    arena->num_slots(), exp_stride / 1048576.0,
                    arena->total_bytes() / 1e9, resident_.bytes() / 1e9,
                    arena->mtlio_active() ? "mtlio" : "pread");
        } else {
            packBuf = rt.wrap_external((void *)pack.data(), (size_t)pack.file_size(), @"m64pack");
            if (!packBuf) { err = "metal: wrap_external pack failed"; return false; }
            // Milestone 14: stream the whole pack into RAM in the background
            // (madvise WILLNEED + a 16 KiB-page touch per macOS arm64 page) so
            // the first real touches don't serialize on minor faults. READY is
            // emitted immediately; M64_PREFETCH_SYNC=1 waits instead, and
            // M64_NO_PREFETCH=1 disables the thread entirely.
            if (!arena_mode && !getenv("M64_NO_PREFETCH") &&
                pack.file_size() >= 1048576) {
                const uint8_t * base = pack.data();
                const size_t len = (size_t)pack.file_size();
                auto warm_pages = [this, base, len]() {
                    const size_t kChunk = (size_t)64ull * 1024 * 1024;
                    const size_t kPage = 16384;  // macOS arm64 page size
                    for (size_t off = 0; off < len; off += kChunk) {
                        if (stop_prefetch_.load(std::memory_order_relaxed)) return;
                        size_t n = len - off;
                        if (n > kChunk) n = kChunk;
                        madvise((void *)(base + off), n, MADV_WILLNEED);
                        for (size_t i = 0; i < n; i += kPage) {
                            // Stop within <=4 MiB (256 pages) of a shutdown request.
                            if (i && (i % (4ull * 1024 * 1024)) == 0 &&
                                stop_prefetch_.load(std::memory_order_relaxed))
                                return;
                            (void)*(volatile const uint8_t *)(base + off + i);
                        }
                    }
                };
                if (getenv("M64_PREFETCH_SYNC")) {
                    warm_pages();
                } else {
                    prefetch_thread_ = std::thread(warm_pages);
                }
            }
        }
        scr = rt.alloc_shared((size_t)o.total * 4, @"m64-scratch");
        if (!scr) { err = "metal: scratch alloc failed"; return false; }
        kvBuf = rt.alloc_shared(kv_bytes, @"m64-kv");
        if (!kvBuf) {
            err = "metal: KV buffer alloc failed (" + std::to_string(kv_bytes) + " B)";
            return false;
        }
        fprintf(stderr, "kv cache: %.2f MiB (%s, ctx=%d, swa_ring=%lld)\n",
                kv_bytes / 1048576.0, kv_q8 ? "Q8_0" : "FP16", ctx, (long long)n_swa + 1);
        // Milestone 9B: pre-compile the batched small-op kernels so a missing
        // or uncompilable kernel fails at init() instead of mid-prefill.
        if (prefill_chunk > 1) {
            static const char * const kBatchFns[] = {
                "k_xfer_batch", "k_mix_rms_batch", "k_mix_fused_batch",
                "k_ctrl_down_batch", "k_ctrl_up_batch"};
            for (const char * fn : kBatchFns) {
                if (!rt.pipeline(fn)) {
                    err = std::string("metal: batch prefill pipeline missing: ") + fn;
                    return false;
                }
            }
        }
        // Milestone 13: pre-warm the hot decode pipelines so request #1 after
        // startup pays zero lazy pipeline-compilation stalls.
        static const char * const kDecodeWarmFns[] = {
            "k_gemv_gateup_batch8_q4k_f32in_m2",
            "k_gemv_down_batch8_q5_0_f32in_m2",
            "k_gemv_down_batch8_q5_1_f32in_m2",
            "k_down_reduce8_conv_add",
            "k_qkv_prep_kv_write_q8",
            "k_qkv_prep_kv_write_f16",
            "k_postconv_residual",
            "k_attn_head_q8_split_gqa",
            "k_attn_head_f16_split_gqa",
            "k_attn_head_f16_reduce",
            "k_gemv_q8_k2304_m2",
            "k_gemv_q8_k2304_multi_m2",
            "k_gemv_q8_k3072_m2",
            "k_gate_mul_quant",
            "k_rms_quant",
            "k_mix_fused",
            "k_ctrl_down",
            "k_ctrl_up",
            "k_router_topk",
            "k_gemv_f32_tg",
            "k_situ_f32_batch",
            "k_situ_quant8"};
        for (const char * fn : kDecodeWarmFns) {
            if (!rt.pipeline(fn)) {
                err = std::string("metal: decode pre-warm pipeline missing: ") + fn;
                return false;
            }
        }
        // Milestone 14: pre-compile all hot prefill pipelines too, so request
        // #1 pays zero lazy JIT stalls on the batched path.
        if (prefill_chunk > 1) {
            static const char * const kPrefillWarmFns[] = {
                "k_emb_gather_batch", "k_rms_batch", "k_conv_batch", "k_clip_batch",
                "k_gate_mul_batch", "k_gemv_f32_batch", "k_gemm_q8_f32in_sg",
                "k_router_topk_batch", "k_moe_bucket_build", "k_moe_down_reduce_batch",
                "k_kv_write_q8_batch", "k_kv_write_f16_batch",
                "k_attn_q8_flash_prefill", "k_attn_f16_flash_prefill",
                "k_attn_head_q8_batch", "k_attn_head_f16_batch",
                "k_moe_grouped_f32in_sg", "k_moe_grouped_f32in_sg_q4k",
                "k_moe_grouped_f32in_sg_q5_0", "k_moe_grouped_f32in_sg_q5_1",
                "k_softcap", "k_argmax"
            };
            for (const char * fn : kPrefillWarmFns) {
                if (!rt.pipeline(fn)) {
                    err = std::string("metal: prefill pre-warm pipeline missing: ") + fn;
                    return false;
                }
            }
        }
        reset();
        return true;
    }

    void reset() {
        pos = 0;
        // O(1): only the recurrent-state scratch is cleared. kvBuf is NOT
        // re-zeroed per prompt — k_attn_head_f16 only reads slots j <= pos,
        // and every slot is written before it is ever read again.
        if (scr) {
            float * base = (float *)scr.contents;
            memset(base + o.rs, 0, (size_t)(kNSLOT * kCH * kHR) * 4);
        }
    }

    // ---- Bitwise-exact prefix caching --------------------------------------
    // Total uint16 elements of the SWA ring K/V slots (the only KV regions a
    // future position >= pos can overwrite; full slots stay linear 0..pos-1).
    size_t swa_kv_elems() const {
        size_t n = 0;
        for (int il = 0; il < kNSLOT; il++)
            if (s_swa[il]) n += (size_t)2 * (size_t)kv_cap[il] * (size_t)kv_u16_per_pos;
        return n;
    }

    // Safe between prefill/step calls: every gpu_step ends with
    // [cb waitUntilCompleted], so scr/kvBuf are GPU-idle here.
    bool save_prefix_state(PrefixState & out, const std::vector<int32_t> & tokens) const {
        if (!scr || !kvBuf) return false;
        if (pos <= 0 || pos >= ctx) return false;
        if (tokens.size() != (size_t)pos) return false;
        out.pos = pos;
        out.tokens = tokens;
        const size_t rs_n = (size_t)kNSLOT * kCH * kHR;
        out.rs.resize(rs_n);
        memcpy(out.rs.data(), (const float *)scr.contents + o.rs, rs_n * sizeof(float));
        const uint16_t * kv = (const uint16_t *)kvBuf.contents;
        out.swa_kv.clear();
        out.swa_kv.reserve(swa_kv_elems());
        for (int il = 0; il < kNSLOT; il++) {
            if (!s_swa[il]) continue;
            const size_t n = (size_t)kv_cap[il] * kv_u16_per_pos;
            out.swa_kv.insert(out.swa_kv.end(), kv + kv_k_off[il], kv + kv_k_off[il] + n);
            out.swa_kv.insert(out.swa_kv.end(), kv + kv_v_off[il], kv + kv_v_off[il] + n);
        }
        return out.swa_kv.size() == swa_kv_elems();
    }

    bool restore_prefix_state(const PrefixState & in) {
        if (!scr || !kvBuf) return false;
        if (in.pos <= 0 || in.pos >= ctx) return false;
        if (in.tokens.size() != (size_t)in.pos) return false;
        if (in.rs.size() != (size_t)kNSLOT * kCH * kHR) return false;
        if (in.swa_kv.size() != swa_kv_elems()) return false;
        // Self-contained stash: full-attention KV must cover [0..in.pos).
        if (!in.full_kv.empty() && in.full_kv.size() != full_kv_elems(in.pos))
            return false;
        pos = in.pos;
        memcpy((float *)scr.contents + o.rs, in.rs.data(), in.rs.size() * sizeof(float));
        uint16_t * kv = (uint16_t *)kvBuf.contents;
        size_t c = 0;
        for (int il = 0; il < kNSLOT; il++) {
            if (!s_swa[il]) continue;
            const size_t n = (size_t)kv_cap[il] * kv_u16_per_pos;
            memcpy(kv + kv_k_off[il], in.swa_kv.data() + c, n * sizeof(uint16_t));
            c += n;
            memcpy(kv + kv_v_off[il], in.swa_kv.data() + c, n * sizeof(uint16_t));
            c += n;
        }
        if (!in.full_kv.empty()) {
            c = 0;
            for (int il = 0; il < kNSLOT; il++) {
                if (s_swa[il]) continue;
                const size_t n = (size_t)in.pos * kv_u16_per_pos;
                memcpy(kv + kv_k_off[il], in.full_kv.data() + c, n * sizeof(uint16_t));
                c += n;
                memcpy(kv + kv_v_off[il], in.full_kv.data() + c, n * sizeof(uint16_t));
                c += n;
            }
        }
        return true;
    }

    // ---- SSD-persistent prefix cache (.m64kv) ------------------------------
    // SWA ring capacity actually used by kvBuf: min(ctx, n_swa + 1). All SWA
    // slots share it, so a --ctx 256 test run and a --ctx 131072 production
    // run never validate/restore against mismatched ring geometry.
    uint32_t swa_ring_cap() const {
        return (uint32_t)((int64_t)ctx < n_swa + 1 ? (int64_t)ctx : n_swa + 1);
    }
    size_t n_full_slots() const {
        size_t n = 0;
        for (int il = 0; il < kNSLOT; il++)
            if (!s_swa[il]) n++;
        return n;
    }

    // ---- Self-contained RAM stash (PrefixState::full_kv) -------------------
    // uint16 elements needed to snapshot [0..n_pos) of every full-attention
    // slot: 2 (K+V) * n_pos * per-token units * number of full slots.
    size_t full_kv_elems(int64_t n_pos) const {
        if (n_pos <= 0) return 0;
        return (size_t)2 * (size_t)n_pos * (size_t)kv_u16_per_pos * n_full_slots();
    }

    // Copy the USED [0..st.pos) K/V of every full-attention slot out of the
    // live kvBuf so the checkpoint survives fwd.reset() + full re-prefills.
    // Same per-slot K-then-V layout as save_disk_prefix's `full` array, so a
    // stash and a .m64kv agree bit-for-bit. GPU-idle between prefill/step.
    void capture_full_kv(PrefixState & st) const {
        st.full_kv.clear();
        if (!kvBuf) return;
        if (st.pos <= 0 || st.pos > pos) return;
        for (int il = 0; il < kNSLOT; il++)
            if (!s_swa[il] && kv_cap[il] < (uint32_t)st.pos) return;  // corrupt st
        st.full_kv.reserve(full_kv_elems(st.pos));
        const uint16_t * kv = (const uint16_t *)kvBuf.contents;
        for (int il = 0; il < kNSLOT; il++) {
            if (s_swa[il]) continue;
            const size_t n = (size_t)st.pos * kv_u16_per_pos;
            st.full_kv.insert(st.full_kv.end(), kv + kv_k_off[il], kv + kv_k_off[il] + n);
            st.full_kv.insert(st.full_kv.end(), kv + kv_v_off[il], kv + kv_v_off[il] + n);
        }
    }

    // Snapshot rs + SWA rings + the USED 0..pos-1 of every full-attention
    // slot (never the unused tail up to ctx) into a RAM DiskPrefixSnapshot.
    // GPU-idle between prefill/step calls, so raw buffer reads are safe.
    // Milestone 14: zero disk I/O here — the caller may hand the snapshot to
    // a background writer thread (write_disk_prefix_file).
    bool snapshot_disk_prefix(const std::string & path, const std::vector<int32_t> & tokens,
                              DiskPrefixSnapshot & out, std::string & err) const {
        auto fail = [&](const std::string & m) { err = "prefix disk: " + m; return false; };
        if (!scr || !kvBuf) return fail("engine not initialized");
        if (pos <= 0 || pos >= ctx) return fail("pos " + std::to_string(pos) +
                                                " outside (0," + std::to_string(ctx) + ")");
        if (tokens.size() != (size_t)pos) return fail("tokens.size() != pos");
        const uint16_t * kv = (const uint16_t *)kvBuf.contents;
        out.path = path;
        out.tokens = tokens;
        const size_t rs_n = (size_t)kNSLOT * kCH * kHR;
        out.rs.resize(rs_n);
        memcpy(out.rs.data(), (const float *)scr.contents + o.rs, rs_n * sizeof(float));
        out.swa.clear();
        out.swa.reserve(swa_kv_elems());
        for (int il = 0; il < kNSLOT; il++) {
            if (!s_swa[il]) continue;
            const size_t n = (size_t)kv_cap[il] * kv_u16_per_pos;
            out.swa.insert(out.swa.end(), kv + kv_k_off[il], kv + kv_k_off[il] + n);
            out.swa.insert(out.swa.end(), kv + kv_v_off[il], kv + kv_v_off[il] + n);
        }
        out.full.clear();
        for (int il = 0; il < kNSLOT; il++) {
            if (s_swa[il]) continue;
            if (kv_cap[il] < (uint32_t)pos)
                return fail("full slot " + std::to_string(il) + " cap < pos");
            const size_t n = (size_t)pos * kv_u16_per_pos;
            out.full.insert(out.full.end(), kv + kv_k_off[il], kv + kv_k_off[il] + n);
            out.full.insert(out.full.end(), kv + kv_v_off[il], kv + kv_v_off[il] + n);
        }
        DiskPrefixHeader & hdr = out.hdr;
        hdr = DiskPrefixHeader{};
        memcpy(hdr.magic, "M64K", 4);
        hdr.version = 1;
        hdr.n_tokens = (uint32_t)pos;
        hdr.n_swa_ring = swa_ring_cap();
        hdr.pack_fp = pack_fp;
        hdr.rs_floats = (uint64_t)kNSLOT * kCH * kHR;
        hdr.swa_u16 = (uint64_t)out.swa.size();
        hdr.full_u16 = (uint64_t)out.full.size();
        err.clear();
        return true;
    }

    // One-shot snapshot + .m64kv write (sync path; main.cpp's WARM handler
    // uses snapshot_disk_prefix + a background writer thread instead).
    bool save_disk_prefix(const std::string & path, const std::vector<int32_t> & tokens,
                          std::string & err) const {
        DiskPrefixSnapshot snap;
        if (!snapshot_disk_prefix(path, tokens, snap, err)) return false;
        if (!write_disk_prefix_file(snap.path, snap.hdr, snap.tokens.data(), snap.rs.data(),
                                    snap.swa.data(), snap.full.data(), err))
            return false;
        err.clear();
        return true;
    }

    // Validate the .m64kv against the LIVE engine geometry, then restore
    // pos + o.rs + SWA rings + full-slot 0..pos-1 bit-for-bit, and hand back
    // the same state as a RAM PrefixState (warm_state in main.cpp stays live).
    bool load_disk_prefix(const std::string & path, PrefixState & out_ram_state,
                          std::string & err) {
        if (!scr || !kvBuf) { err = "prefix disk: engine not initialized"; return false; }
        std::vector<int32_t> tokens;
        std::vector<float> rs;
        std::vector<uint16_t> swa, full;
        if (!read_disk_prefix_file(path, pack_fp, (uint32_t)ctx, swa_ring_cap(),
                                   (uint64_t)kNSLOT * kCH * kHR,
                                   (uint64_t)swa_kv_elems(),
                                   (uint32_t)(2 * (size_t)kv_u16_per_pos * n_full_slots()),
                                   tokens, rs, swa, full, err))
            return false;
        const int64_t p = (int64_t)tokens.size();
        if (p <= 0 || p >= ctx) { err = "prefix disk: restored pos out of range"; return false; }
        pos = p;
        memcpy((float *)scr.contents + o.rs, rs.data(), rs.size() * sizeof(float));
        uint16_t * kv = (uint16_t *)kvBuf.contents;
        size_t c = 0;
        for (int il = 0; il < kNSLOT; il++) {
            if (!s_swa[il]) continue;
            const size_t n = (size_t)kv_cap[il] * kv_u16_per_pos;
            memcpy(kv + kv_k_off[il], swa.data() + c, n * sizeof(uint16_t));
            c += n;
            memcpy(kv + kv_v_off[il], swa.data() + c, n * sizeof(uint16_t));
            c += n;
        }
        c = 0;
        for (int il = 0; il < kNSLOT; il++) {
            if (s_swa[il]) continue;
            const size_t n = (size_t)p * kv_u16_per_pos;
            memcpy(kv + kv_k_off[il], full.data() + c, n * sizeof(uint16_t));
            c += n;
            memcpy(kv + kv_v_off[il], full.data() + c, n * sizeof(uint16_t));
            c += n;
        }
        out_ram_state.pos = p;
        out_ram_state.tokens = std::move(tokens);
        out_ram_state.rs = std::move(rs);
        out_ram_state.swa_kv = std::move(swa);
        err.clear();
        return true;
    }

    // =============================== one token ===============================
    // need_logits=false skips the LM head (rms_quant + vocab GEMV + softcap +
    // argmax): used by prefill for all tokens except the last. State (pos,
    // KV, conv rs, block buffers) still advances exactly like a full step.
    StepResult gpu_step(int32_t token, std::string & err, bool need_logits = true) {
        StepResult r{(const float *)scr.contents + o.logits, vocab_eff, 0};
        if (pos >= ctx) { err = "metal: pos >= ctx (context/KV full)"; return r; }
        // Long-context log-scale (identical formula to forward_impl.hpp):
        // tau_v = log(max(1, (pos+1)/floor)) * alpha + 1
        float tsc = ((float)pos + 1.0f) / log_scale_floor;
        if (tsc < 1.0f) tsc = 1.0f;
        const float tau_v = logf(tsc) * log_scale_alpha + 1.0f;
        id<MTLCommandBuffer> cb = [rt.queue() commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        const bool trace_slot = (getenv("M64_TRACE_SLOT") != nullptr);
        const bool nanprobe = (getenv("M64_NANPROBE") != nullptr);
        const bool gemv_m2 = (getenv("M64_NO_M2") == nullptr);
        const bool gemv_r3 = !(getenv("M64_GEMV_R3") && strcmp(getenv("M64_GEMV_R3"), "0") == 0);
        const bool gemv_r4 = !(getenv("M64_GEMV_R4") && strcmp(getenv("M64_GEMV_R4"), "0") == 0);
        // Milestone 11: fuse clip(Q)+clip(K)+rms(V)+clip(Rr)+kv_write (5 -> 1)
        // and postconv+residual (2 -> 1). trace_slot/nanprobe keep the
        // unfused chain so per-dispatch fault attribution still works.
        const bool fuse_m11 = (getenv("M64_NO_FUSE_M11") == nullptr) && !trace_slot && !nanprobe;
        const bool profile = (getenv("M64_PROFILE") != nullptr) && !arena_mode && pos == 5;
        enum ProfCat { P_OTHER = 0, P_MIX, P_CTRL, P_QKV, P_ATTN, P_WO, P_ROUTER, P_GATEUP, P_DOWN, P_LMHEAD, P_NCAT };
        double prof_ms[P_NCAT] = {};
        uint32_t prof_disp[P_NCAT] = {};
        uint32_t cur_disp = 0;
        auto prof_flush = [&](int cat) {
            if (!profile || !err.empty()) return;
            [en endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                NSString * msg = cb.error.localizedDescription ?: @"unknown MTLCommandBufferError";
                err = std::string("metal: profile status=") +
                      std::to_string((NSInteger)cb.status) + ": " + msg.UTF8String;
            } else {
                prof_ms[cat] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                prof_disp[cat] += cur_disp;
            }
            cur_disp = 0;
            cb = [rt.queue() commandBuffer];
            en = [cb computeCommandEncoder];
        };
        bool trace_aborted = false;
        auto trace_flush = [&](const char * tag) {
            if (!trace_slot || trace_aborted || !err.empty()) return;
            [en endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                NSString * msg = cb.error.localizedDescription ?: @"unknown MTLCommandBufferError";
                err = std::string("metal: command buffer status=") +
                      std::to_string((NSInteger)cb.status) + ": " + msg.UTF8String;
                fprintf(stderr, "TRACE %-14s GPU_FAULT %s\n", tag, err.c_str());
                trace_aborted = true;
            } else {
                const float * S = (const float *)scr.contents;
                auto chk = [&](uint32_t off, uint32_t n, float & mx) -> uint32_t {
                    uint32_t b = 0; mx = 0.0f;
                    for (uint32_t i = 0; i < n; i++) {
                        float v = S[off + i];
                        if (!isfinite(v)) b++;
                        else { float a = fabsf(v); if (a > mx) mx = a; }
                    }
                    return b;
                };
                float me = 0, mc = 0, mq = 0, mk = 0, mv = 0, ma = 0, mm = 0, mf = 0, ml = 0;
                uint32_t be = chk(o.emb, kNE, me), bc = chk(o.cur, kNE, mc);
                uint32_t bq = chk(o.Q, kNH * kDH, mq), bk = chk(o.K, kKVD, mk);
                uint32_t bv = chk(o.V2, kKVD, mv), ba = chk(o.attnb, kNH * kDH, ma);
                uint32_t bm = chk(o.mixed, kNE, mm), bf = chk(o.fin, kNE, mf);
                uint32_t bl = chk(o.logits, (uint32_t)vocab_rows, ml);
                printf("TRACE pos=%lld %-12s bad(e=%u c=%u Q=%u K=%u V=%u a=%u m=%u f=%u l=%u) "
                       "max(e=%.4g c=%.4g Q=%.4g K=%.4g a=%.4g m=%.4g f=%.4g l=%.4g)\n",
                       (long long)pos, tag, be, bc, bq, bk, bv, ba, bm, bf, bl, me, mc, mq, mk, ma, mm, mf, ml);
                fflush(stdout);
                if (be || bc || bq || bk || bv || ba || bm || bf || bl) trace_aborted = true;
            }
            cb = [rt.queue() commandBuffer];
            en = [cb computeCommandEncoder];
        };

        // Current PACK-role MTLBuffer for d_tg/d_th: the full pack (Phase 2A)
        // or residentBuf; switched to expertBuf only around the 2 MoE GEMVs.
        id<MTLBuffer> curPack = arena_mode ? residentBuf : packBuf;
        // d_tg: threadgroup-indexed kernels (kernel reads threadgroup_position_in_grid).
        auto d_tg = [&](const char * fn, uint tgx, uint tgy, uint nth, const void * a, size_t sz) {
            if (err.empty()) {
                id<MTLComputePipelineState> p = rt.pipeline(fn);
                if (!p) { err = std::string("pipeline ") + fn; return; }
                [en setComputePipelineState:p];
                [en setBuffer:scr offset:0 atIndex:0];
                [en setBytes:a length:sz atIndex:1];
                [en setBuffer:curPack offset:0 atIndex:2];
                [en dispatchThreadgroups:MTLSizeMake(tgx, tgy, 1)
                    threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                cur_disp++;
            }
        };
        // d_th: flat 1D thread-indexed kernels (kernel reads thread_position_in_grid).
        auto d_th = [&](const char * fn, uint total, uint nth, const void * a, size_t sz) {
            if (err.empty()) {
                id<MTLComputePipelineState> p = rt.pipeline(fn);
                if (!p) { err = std::string("pipeline ") + fn; return; }
                [en setComputePipelineState:p];
                [en setBuffer:scr offset:0 atIndex:0];
                [en setBytes:a length:sz atIndex:1];
                [en setBuffer:curPack offset:0 atIndex:2];
                [en dispatchThreads:MTLSizeMake(total, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                cur_disp++;
            }
        };
        // d_th2: flat 2D thread-indexed kernels (kernel reads (x, y) of
        // thread_position_in_grid, e.g. k_situ_f32_batch's (i, pair) grid).
        auto d_th2 = [&](const char * fn, uint nx, uint ny, uint nth, const void * a, size_t sz) {
            if (err.empty()) {
                id<MTLComputePipelineState> p = rt.pipeline(fn);
                if (!p) { err = std::string("pipeline ") + fn; return; }
                [en setComputePipelineState:p];
                [en setBuffer:scr offset:0 atIndex:0];
                [en setBytes:a length:sz atIndex:1];
                [en setBuffer:curPack offset:0 atIndex:2];
                [en dispatchThreads:MTLSizeMake(nx, ny, 1)
                    threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                cur_disp++;
            }
        };
        auto rms = [&](const uint32_t * xr, const uint64_t * wr, const uint32_t * yr,
                       uint32_t k, uint32_t rows, uint32_t hasw, float eps) {
            RMSArgs A{}; A.k = k; A.rows = rows; A.hasw = hasw; A.eps = eps;
            for (uint32_t i = 0; i < rows; i++) { A.xr[i] = xr[i]; A.wr[i] = wr ? wr[i] : 0; A.yr[i] = yr[i]; }
            d_tg("k_rms", rows, 1, 256, &A, sizeof(A));
        };
        auto quant = [&](uint32_t x, uint32_t q, uint32_t dd, uint32_t nb) {
            QuantArgs A{x, q, dd, nb}; d_th("k_quant", nb, 32, &A, sizeof(A));
        };
        auto gemv2304 = [&](uint32_t y, uint32_t nout, uint32_t xq, uint32_t dx, uint64_t w,
                            uint32_t idx = 0, uint32_t stride = 0) {
            GemvArgs A{w, 0, xq, dx, y, 0, nout, idx, stride, 0};
            if (gemv_m2) d_tg("k_gemv_q8_k2304_m2", (nout + 15) / 16, 1, 256, &A, sizeof(A));
            else d_tg("k_gemv_q8_k2304", (nout + 7) / 8, 1, 256, &A, sizeof(A));
        };
        auto gemv2304_multi = [&](uint32_t xq, uint32_t dx, uint32_t count,
                                  const uint64_t * w, const uint32_t * y, const uint32_t * nout) {
            MultiGemvArgs A{};
            A.xqoff = xq; A.dxoff = dx; A.count = count;
            uint32_t total = 0;
            for (uint32_t i = 0; i < count; i++) {
                A.woff[i] = w[i]; A.yoff[i] = y[i]; A.nout[i] = nout[i];
                total += nout[i];
                A.cum[i] = total;
            }
            A.total = total;
            d_tg("k_gemv_q8_k2304_multi_m2", (total + 15) / 16, 1, 256, &A, sizeof(A));
        };
        auto gemv3072 = [&](uint32_t y, uint32_t nout, uint32_t xq, uint32_t dx, uint64_t w) {
            GemvArgs A{w, 0, xq, dx, y, 0, nout, 0, 0, 0};
            if (gemv_m2) d_tg("k_gemv_q8_k3072_m2", (nout + 15) / 16, 1, 256, &A, sizeof(A));
            else d_tg("k_gemv_q8_k3072", (nout + 7) / 8, 1, 256, &A, sizeof(A));
        };
        // rmsnorm fused with Q8_0 quant (yr[i] = M64_NOFFU skips the FP32 write).
        auto rms_quant = [&](const uint32_t * xr, const uint64_t * wr, const uint32_t * yr,
                             const uint32_t * qr, const uint32_t * dr, uint32_t k, uint32_t rows, float eps) {
            RmsQuantArgs A{}; A.k = k; A.rows = rows; A.hasw = 1; A.eps = eps;
            for (uint32_t i = 0; i < rows; i++) {
                A.xr[i] = xr[i]; A.wr[i] = wr[i]; A.yr[i] = yr[i]; A.q[i] = qr[i]; A.d[i] = dr[i];
            }
            d_tg("k_rms_quant", rows, 1, 256, &A, sizeof(A));
        };
        auto gemvf32 = [&](uint32_t y, uint32_t nout, uint32_t k, uint32_t x, uint64_t w) {
            GemvF32Args A{w, x, y, nout, k}; d_tg("k_gemv_f32_tg", nout, 1, 256, &A, sizeof(A));
        };
        auto conv = [&](uint32_t il, uint32_t sec_chan, uint32_t C, uint32_t x, uint32_t y,
                        uint64_t wpack, uint32_t dyn, uint32_t group, float scale) {
            ConvArgs A{o.rs + (il * (uint32_t)kCH + sec_chan) * kHR, x, y, wpack, dyn, C, group, scale, dyn ? 1u : 0u};
            d_th("k_conv_slot", C, 128, &A, sizeof(A));
        };
        auto clip = [&](uint32_t coff, uint64_t soff, uint32_t groups, uint32_t per, float thresh) {
            ClipArgs A{coff, soff, groups, per, thresh}; d_tg("k_clip", groups, 1, 128, &A, sizeof(A));
        };
        auto xfer2 = [&](uint32_t a0, uint32_t b0, uint32_t c0, uint32_t n0,
                         uint32_t a1, uint32_t b1, uint32_t c1, uint32_t n1, uint32_t mode) {
            XferArgs A{{a0, a1}, {b0, b1}, {c0, c1}, {n0, n1}, mode};
            uint32_t n = mode == 0 ? 1 : (n1 ? 2 : 1);
            d_tg("k_xfer", n, 1, 256, &A, sizeof(A));
        };

        auto do_mix = [&](uint32_t states, int n, uint32_t extra, uint64_t qpack, int nq, uint32_t out) {
            uint32_t xr[kMAXS + 1], yr[kMAXS + 1];
            int S = n + 1;
            for (int i = 0; i < S - 1; i++) xr[i] = states + (uint32_t)(n - 1 - i) * kNE;
            xr[S - 1] = extra;
            for (int i = 0; i < S; i++) yr[i] = o.stn + (uint32_t)i * kNE;
            rms(xr, nullptr, yr, kNE, (uint32_t)S, 0, 1e-6f);
            if (gemv_r3) {
                MixFusedArgs ma{qpack, o.stn, out, (uint32_t)S, (uint32_t)nq, 1.0f / sqrtf((float)kNE), 0};
                d_tg("k_mix_fused", (uint32_t)nq, 1, 256, &ma, sizeof(ma));
            } else {
                MixDotArgs da{o.stn, o.sc, qpack, (uint32_t)S, 0, 1.0f / sqrtf((float)kNE)};
                d_tg("k_mix_dot", (uint32_t)nq, (uint32_t)((S + 7) / 8), 256, &da, sizeof(da));
                MixSoftArgs sa{o.sc, (uint32_t)S};
                d_tg("k_mix_soft", (uint32_t)nq, 1, 64, &sa, sizeof(sa));
                MixCombArgs ca{o.stn, o.sc, out, (uint32_t)S, (uint32_t)nq};
                d_tg("k_mix_comb", (uint32_t)nq, 1, 256, &ca, sizeof(ca));
            }
            prof_flush(P_MIX);
        };
        auto do_ctrl = [&](const uint32_t * in, const uint32_t * msg, const uint32_t * out,
                           const uint64_t * wd, const uint64_t * wg, const uint64_t * wf, int nc) {
            uint32_t xr[8], yr[8];
            uint64_t wr[8];
            for (int j = 0; j < nc; j++) {
                xr[2 * j] = in[j];   yr[2 * j] = o.gcat + (uint32_t)(2 * j) * kNE;
                xr[2 * j + 1] = msg[j]; yr[2 * j + 1] = o.gcat + (uint32_t)(2 * j + 1) * kNE;
                wr[2 * j] = wr[2 * j + 1] = 0;
            }
            rms(xr, wr, yr, kNE, (uint32_t)(2 * nc), 0, 1e-6f);
            if (gemv_r4) {
                // R4: split control into two wide-dispatch kernels. No gb/fb
                // staging: k_ctrl_down writes hh[nc][64], k_ctrl_up writes out.
                // prof_flush(P_CTRL) stays once at the end (no flush in between).
                CtrlSplitArgs A{};
                for (int j = 0; j < nc; j++) {
                    A.in[j] = in[j]; A.msg[j] = msg[j]; A.out[j] = out[j];
                    A.wdown[j] = wd[j]; A.wgain[j] = wg[j]; A.wforg[j] = wf[j];
                }
                A.gcat = o.gcat; A.hh = o.ctrl_hh; A.eps = 1e-6f;
                A.grank = 64; A.ne = kNE; A.nc = (uint32_t)nc;
                d_tg("k_ctrl_down", 64, (uint32_t)nc, 256, &A, sizeof(A));
                d_tg("k_ctrl_up", (kNE + 31) / 32, (uint32_t)nc, 32, &A, sizeof(A));
            } else {
                CtrlArgs A{};
                for (int j = 0; j < nc; j++) {
                    A.in[j] = in[j]; A.msg[j] = msg[j]; A.out[j] = out[j];
                    A.wdown[j] = wd[j]; A.wgain[j] = wg[j]; A.wforg[j] = wf[j];
                    A.gb[j] = o.gb + (uint32_t)j * kNE; A.fb[j] = o.fb + (uint32_t)j * kNE;
                }
                A.gcat = o.gcat; A.eps = 1e-6f; A.grank = 64; A.ne = kNE;
                d_tg("k_control", (uint32_t)nc, 1, 1024, &A, sizeof(A));
            }
            prof_flush(P_CTRL);
        };

        struct { int nblk; } st = {0};
        auto slot = [&](int il, int ip, const LL * lp) {
            const LW & L = phys[s_phys[il]];
            const bool is_swa = s_swa[il];
            // --- attn inputs ---
            if (L.aquery) do_mix(o.blocks, st.nblk, o.cur, L.aquery, L.nq_rows, o.mixed);
            else { xfer2(o.cur, 0, o.mixed, kNE, 0, 0, 0, 0, 1); prof_flush(P_OTHER); }
            uint32_t in3[3];
            for (int j = 0; j < 3; j++) in3[j] = o.mixed + (is_swa ? 0u : (uint32_t)j * kNE);
            if (lp) {
                uint32_t msgp[3] = {o.msgs, o.msgs + kNE, o.msgs + 2 * kNE};
                uint32_t outp[3] = {o.ctrlout, o.ctrlout + kNE, o.ctrlout + 2 * kNE};
                do_ctrl(in3, msgp, outp, lp->down, lp->gain, lp->forg, 3);
                for (int j = 0; j < 3; j++) in3[j] = outp[j];
            }
            // --- fused input norms + Q8_0 quant ---
            // *_f32 projections read the normalized FP32 directly (o.qsrc/ksrc/vsrc
            // stay dead scratch when the flags are false, so Q8_0 packs are unchanged).
            if (is_swa && !lp) {
                uint32_t y1 = L.wsink_f32 ? o.qsrc : M64_NOFFU, q1 = o.xqA, d1 = o.dxA;
                rms_quant(in3, &L.normattn, &y1, &q1, &d1, kNE, 1, f_eps);
            } else {
                uint64_t wr3[3] = {
                    is_swa ? L.normattn : L.nq,
                    is_swa ? L.normattn : L.nk,
                    is_swa ? L.normattn : L.nv
                };
                uint32_t yr3[3] = {
                    L.wsink_f32 ? o.qsrc : M64_NOFFU,
                    L.kcovp_f32 ? o.ksrc : M64_NOFFU,
                    L.vcovp_f32 ? o.vsrc : M64_NOFFU
                };
                uint32_t qr3[3] = {o.xqA, o.xqB, o.xqC};
                uint32_t dr3[3] = {o.dxA, o.dxB, o.dxC};
                rms_quant(in3, wr3, yr3, qr3, dr3, kNE, 3, f_eps);
            }
            const uint32_t xqB = (is_swa && !lp) ? o.xqA : o.xqB;
            const uint32_t dxb = (is_swa && !lp) ? o.dxA : o.dxB;
            const uint32_t xqC = (is_swa && !lp) ? o.xqA : o.xqC;
            const uint32_t dxc = (is_swa && !lp) ? o.dxA : o.dxC;
            // --- Q K V R (+ sink/gate when R3 fused) projections & short convolutions ---
            if (gemv_r3) {
                if (is_swa && !lp) {
                    uint64_t ws[6]; uint32_t ys[6], ns[6], nn = 0;
                    ws[nn] = L.wq;   ys[nn] = o.Q;    ns[nn] = kNH * kDH;   nn++;
                    ws[nn] = L.wk;   ys[nn] = o.K;    ns[nn] = kKVD;        nn++;
                    ws[nn] = L.wv;   ys[nn] = o.V2;   ns[nn] = kKVD;        nn++;
                    ws[nn] = L.wrel; ys[nn] = o.Rr;   ns[nn] = kNH * kNREL; nn++;
                    if (!L.wsink_f32) { ws[nn] = L.wsink; ys[nn] = o.sinkb; ns[nn] = kNH; nn++; }
                    ws[nn] = L.wgate; ys[nn] = o.gateb; ns[nn] = kNH * kDH; nn++;
                    gemv2304_multi(o.xqA, o.dxA, nn, ws, ys, ns);
                    if (L.wsink_f32) gemvf32(o.sinkb, kNH, kNE, o.qsrc, L.wsink);
                    conv(il, 0, kKVD, o.K, o.K, L.kconv, 0, 128, kConvStatic);
                    conv(il, kKVD, kKVD, o.V2, o.V2, L.vconv, 0, 128, kConvStatic);
                } else {
                    uint64_t ws[4]; uint32_t ys[4], ns[4], nn = 0;
                    ws[nn] = L.wq;   ys[nn] = o.Q;    ns[nn] = kNH * kDH;   nn++;
                    ws[nn] = L.wrel; ys[nn] = o.Rr;   ns[nn] = kNH * kNREL; nn++;
                    if (!L.wsink_f32) { ws[nn] = L.wsink; ys[nn] = o.sinkb; ns[nn] = kNH; nn++; }
                    ws[nn] = L.wgate; ys[nn] = o.gateb; ns[nn] = kNH * kDH; nn++;
                    gemv2304_multi(o.xqA, o.dxA, nn, ws, ys, ns);
                    if (L.wsink_f32) gemvf32(o.sinkb, kNH, kNE, o.qsrc, L.wsink);
                    if (!is_swa) {
                        if (L.kcovp_f32) {
                            gemv2304(o.K, kKVD, o.xqB, o.dxB, L.wk);
                            gemvf32(o.dynb, 32, kNE, o.ksrc, L.kcovp);
                        } else {
                            const uint64_t wks[2] = {L.wk, L.kcovp};
                            const uint32_t yks[2] = {o.K, o.dynb};
                            const uint32_t nks[2] = {kKVD, 32};
                            gemv2304_multi(o.xqB, o.dxB, 2, wks, yks, nks);
                        }
                        conv(il, 0, kKVD, o.K, o.K, L.kconv, o.dynb, 128, 1.0f / 3.0f);
                        if (L.vcovp_f32) {
                            gemv2304(o.V2, kKVD, xqC, dxc, L.wv);
                            gemvf32(o.dynb, 32, kNE, o.vsrc, L.vcovp);
                        } else {
                            const uint64_t wvs[2] = {L.wv, L.vcovp};
                            const uint32_t yvs[2] = {o.V2, o.dynb};
                            const uint32_t nvs[2] = {kKVD, 32};
                            gemv2304_multi(xqC, dxc, 2, wvs, yvs, nvs);
                        }
                        conv(il, kKVD, kKVD, o.V2, o.V2, L.vconv, o.dynb, 128, 1.0f / 3.0f);
                    } else {
                        gemv2304(o.K, kKVD, xqB, dxb, L.wk);
                        gemv2304(o.V2, kKVD, xqC, dxc, L.wv);
                        conv(il, 0, kKVD, o.K, o.K, L.kconv, 0, 128, kConvStatic);
                        conv(il, kKVD, kKVD, o.V2, o.V2, L.vconv, 0, 128, kConvStatic);
                    }
                }
            } else {
                gemv2304(o.Q, kNH * kDH, o.xqA, o.dxA, L.wq);
                gemv2304(o.K, kKVD, xqB, dxb, L.wk);
                gemv2304(o.V2, kKVD, xqC, dxc, L.wv);
                gemv2304(o.Rr, kNH * kNREL, o.xqA, o.dxA, L.wrel);
                if (!is_swa) {
                    if (L.kcovp_f32) gemvf32(o.dynb, 32, kNE, o.ksrc, L.kcovp);
                    else gemv2304(o.dynb, 32, o.xqB, o.dxB, L.kcovp);
                    conv(il, 0, kKVD, o.K, o.K, L.kconv, o.dynb, 128, 1.0f / 3.0f);
                    if (L.vcovp_f32) gemvf32(o.dynb, 32, kNE, o.vsrc, L.vcovp);
                    else gemv2304(o.dynb, 32, xqC, dxc, L.vcovp);
                    conv(il, kKVD, kKVD, o.V2, o.V2, L.vconv, o.dynb, 128, 1.0f / 3.0f);
                } else {
                    conv(il, 0, kKVD, o.K, o.K, L.kconv, 0, 128, kConvStatic);
                    conv(il, kKVD, kKVD, o.V2, o.V2, L.vconv, 0, 128, kConvStatic);
                }
            }
            // --- clips / head norms ---
            prof_flush(P_QKV);
            const uint32_t ring_idx = is_swa ? ((uint32_t)pos % kv_cap[il]) : (uint32_t)pos;
            const uint64_t dstk = kv_k_off[il] + (uint64_t)ring_idx * kv_u16_per_pos;
            const uint64_t dstv = kv_v_off[il] + (uint64_t)ring_idx * kv_u16_per_pos;
            if (fuse_m11) {
                // Milestone 11: ONE dispatch replaces clip(Q)+clip(K)+
                // rms(V)+clip(Rr)+k_kv_write_{q8,f16}. K clip uses M64_NOFF
                // scale; V rms writes straight into KV (no o.emb staging).
                QkvPrepKvArgs qa{};
                qa.qclip_off = L.qclip; qa.relclip_off = L.relclip;
                qa.dst_k = dstk; qa.dst_v = dstv;
                qa.qoff = o.Q; qa.koff = o.K; qa.voff = o.V2; qa.roff = o.Rr;
                qa.qk_thresh = f_qk_clip; qa.rel_thresh = f_rel_clip; qa.eps = f_eps;
                qa.pad0 = 0u;
                const char * fn = kv_q8 ? "k_qkv_prep_kv_write_q8"
                                        : "k_qkv_prep_kv_write_f16";
                id<MTLComputePipelineState> p = rt.pipeline(fn);
                if (!p) { err = std::string("pipeline ") + fn; }
                else {
                    [en setComputePipelineState:p];
                    [en setBuffer:scr offset:0 atIndex:0];
                    [en setBytes:&qa length:sizeof(qa) atIndex:1];
                    [en setBuffer:curPack offset:0 atIndex:2];
                    [en setBuffer:kvBuf offset:0 atIndex:3];
                    [en dispatchThreadgroups:MTLSizeMake(64, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    cur_disp++;
                }
                if (!gemv_r3) {
                    if (L.wsink_f32) gemvf32(o.sinkb, kNH, kNE, o.qsrc, L.wsink);
                    else gemv2304(o.sinkb, kNH, o.xqA, o.dxA, L.wsink);
                }
            } else {
            clip(o.Q, L.qclip, kNH, kDH, f_qk_clip);
            clip(o.K, M64_NOFF, kKVD / kDH, kDH, f_qk_clip);
            // V per-head rmsnorm (in-place unsafe on GPU): stage via o.emb (consumed long ago)
            {
                uint32_t xr8[8], yr8[8];
                for (int hh = 0; hh < 8; hh++) { xr8[hh] = o.V2 + hh * kDH; yr8[hh] = o.emb + hh * kDH; }
                rms(xr8, nullptr, yr8, kDH, 8, 0, f_eps);
                // stage normed V rows into kvv slot via emb below (pair 2 of KV write)
            }
            clip(o.Rr, L.relclip, kNH, kNREL, f_rel_clip);
            // --- sink bias, KV cache write ---
            if (!gemv_r3) {
                if (L.wsink_f32) gemvf32(o.sinkb, kNH, kNE, o.qsrc, L.wsink);
                else gemv2304(o.sinkb, kNH, o.xqA, o.dxA, L.wsink);
            }
            if (err.empty()) {
                KVWriteF16Args kw{
                    o.K, o.emb, kKVD, 0u,
                    dstk, dstv,
                };
                // Milestone 5B: Q8_0 writes 8 KV heads x (32 lanes x char4).
                const bool wq8 = kv_q8;
                const char * wn = wq8 ? "k_kv_write_q8" : "k_kv_write_f16";
                id<MTLComputePipelineState> p = rt.pipeline(wn);
                if (!p) { err = std::string("pipeline ") + wn; }
                else {
                    [en setComputePipelineState:p];
                    [en setBuffer:scr offset:0 atIndex:0];
                    [en setBytes:&kw length:sizeof(kw) atIndex:1];
                    [en setBuffer:kvBuf offset:0 atIndex:2];
                    if (wq8)
                        [en dispatchThreadgroups:MTLSizeMake(8, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    else
                        [en dispatchThreads:MTLSizeMake(kKVD, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    cur_disp++;
                }
            }
            }  // !fuse_m11 (unfused clip/rms/kv-write chain)
            // --- attention ---
            if (err.empty()) {
                const uint32_t n_tok = is_swa
                    ? (uint32_t)std::min<int64_t>(pos + 1, (int64_t)n_swa + 1)
                    : (uint32_t)(pos + 1);
                if (n_tok >= 512u) {
                    // Milestone 9C: GQA 3-head KV-shared split-K dispatches
                    // grid (8, n_splits, 1); M64_NO_ATTN_GQA=1 falls back to
                    // the legacy 24-head kernels. 128-token chunks scale the
                    // split count smoothly to the 128-split cap at 16k tok.
                    const bool attn_gqa = !g_no_attn_gqa;
                    const uint32_t n_splits = std::min<uint32_t>(
                        (uint32_t)kAttnMaxSplits, (n_tok + 127u) / 128u);
                    AttnF16SplitArgs sa{};
                    sa.relbiaspack = L.relbias;
                    sa.kvkoff = kv_k_off[il];
                    sa.kvvoff = kv_v_off[il];
                    sa.qoff = o.Q; sa.rout = o.Rr; sa.sinkoff = o.sinkb; sa.outoff = o.attnb;
                    sa.part_o_off = o.attn_part_o; sa.part_ms_off = o.attn_part_ms;
                    sa.pos = (uint32_t)pos;
                    sa.is_swa = is_swa ? 1u : 0u;
                    sa.n_swa = (uint32_t)n_swa;
                    sa.ring_cap = kv_cap[il];
                    sa.n_splits = n_splits;
                    sa.pad0 = 0u;
                    sa.scale = 1.0f / sqrtf((float)kDH);
                    sa.tau = is_swa ? 1.0f : tau_v;
                    const char * sn = attn_gqa
                        ? (kv_q8 ? "k_attn_head_q8_split_gqa"
                                 : "k_attn_head_f16_split_gqa")
                        : (kv_q8 ? "k_attn_head_q8_split" : "k_attn_head_f16_split");
                    id<MTLComputePipelineState> ps = rt.pipeline(sn);
                    id<MTLComputePipelineState> pr = rt.pipeline("k_attn_head_f16_reduce");
                    if (!ps || !pr) {
                        err = !ps ? std::string("pipeline ") + sn
                                  : std::string("pipeline k_attn_head_f16_reduce");
                    } else {
                        [en setComputePipelineState:ps];
                        [en setBuffer:scr offset:0 atIndex:0];
                        [en setBytes:&sa length:sizeof(sa) atIndex:1];
                        [en setBuffer:curPack offset:0 atIndex:2];
                        [en setBuffer:kvBuf offset:0 atIndex:3];
                        [en dispatchThreadgroups:
                             MTLSizeMake(attn_gqa ? kKVD / kDH : kNH, n_splits, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        [en setComputePipelineState:pr];
                        [en setBuffer:scr offset:0 atIndex:0];
                        [en setBytes:&sa length:sizeof(sa) atIndex:1];
                        [en dispatchThreadgroups:MTLSizeMake(kNH, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        cur_disp += 2;
                    }
                } else {
                    AttnF16Args aa{};
                    aa.relbiaspack = L.relbias;
                    aa.kvkoff = kv_k_off[il];
                    aa.kvvoff = kv_v_off[il];
                    aa.qoff = o.Q; aa.rout = o.Rr; aa.sinkoff = o.sinkb; aa.outoff = o.attnb;
                    aa.pos = (uint32_t)pos;
                    aa.is_swa = is_swa ? 1u : 0u;
                    aa.n_swa = (uint32_t)n_swa;
                    aa.ring_cap = kv_cap[il];
                    aa.scale = 1.0f / sqrtf((float)kDH);
                    aa.tau = is_swa ? 1.0f : tau_v;
                    const char * an = kv_q8 ? "k_attn_head_q8" : "k_attn_head_f16";
                    id<MTLComputePipelineState> p = rt.pipeline(an);
                    if (!p) { err = std::string("pipeline ") + an; }
                    else {
                        [en setComputePipelineState:p];
                        [en setBuffer:scr offset:0 atIndex:0];
                        [en setBytes:&aa length:sizeof(aa) atIndex:1];
                        [en setBuffer:curPack offset:0 atIndex:2];
                        [en setBuffer:kvBuf offset:0 atIndex:3];
                        [en dispatchThreadgroups:MTLSizeMake(kNH, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        cur_disp++;
                    }
                }
            }
            prof_flush(P_ATTN);
            if (trace_slot) { char tb[32]; snprintf(tb, sizeof(tb), "s%d_p%d_attn", il, ip); trace_flush(tb); }
            // --- gate, output projection, postconv ---
            if (!gemv_r3) gemv2304(o.gateb, kNH * kDH, o.xqA, o.dxA, L.wgate);
            if (!trace_slot && !nanprobe) {
                GateMulQuantArgs gq{o.attnb, o.gateb, o.xqD, o.dxD, 96};
                d_th("k_gate_mul_quant", 96 * 32, 128, &gq, sizeof(gq));
            } else {
                GateMulArgs gma{o.attnb, o.gateb, kNH * kDH};
                d_th("k_gate_mul", kNH * kDH, 128, &gma, sizeof(gma));
                quant(o.attnb, o.xqD, o.dxD, 96);
            }
            gemv3072(o.tA, kNE, o.xqD, o.dxD, L.wo);
            if (fuse_m11) {
                // Milestone 11: k_postconv_residual folds k_conv_slot +
                // k_xfer into ONE dispatch (per-channel ownership makes the
                // cur read/modify/write race-free).
                PostConvResArgs pa{
                    L.postconv,
                    o.rs + ((uint32_t)il * (uint32_t)kCH + 2u * kKVD) * kHR,
                    o.tA, o.cur,
                    o.blocks + (uint32_t)st.nblk * kNE,
                    kNE, (uint32_t)((ip % 6 == 0) ? 1 : 0), kConvStatic, 0u};
                d_th("k_postconv_residual", kNE, 128, &pa, sizeof(pa));
                if (ip % 6 == 0) st.nblk++;
            } else {
            conv(il, 2 * kKVD, kNE, o.tA, o.tB, L.postconv, 0, kNE, kConvStatic);
            if (ip % 6 == 0) {
                // Two sequential dispatches: reading o.cur while threadgroup 1
                // writes o.cur in one parallel dispatch is a data race.
                xfer2(o.cur, 0, o.blocks + (uint32_t)st.nblk * kNE, kNE, 0, 0, 0, 0, 1);
                xfer2(o.tB, 0, o.cur, kNE, 0, 0, 0, 0, 1);
                st.nblk++;
            } else {
                xfer2(o.cur, o.tB, 0, kNE, 0, 0, 0, 0, 0);
            }
            }  // !fuse_m11 (unfused postconv+residual chain)
            prof_flush(P_WO);
            // --- FFN ---
            do_mix(o.blocks, st.nblk, o.cur, L.ffnq, 1, o.mixed);
            {
                uint32_t fsrc = o.mixed;
                if (lp) {
                    uint32_t in1 = o.mixed, msg1 = o.msgs + 3 * kNE, out1 = o.ctrlout;
                    do_ctrl(&in1, &msg1, &out1, lp->down + 3, lp->gain + 3, lp->forg + 3, 1);
                    fsrc = out1;
                }
                // o.fin is read in FP32 by router (gemvf32); xqE/dxE feed MoE gate|up.
                rms_quant(&fsrc, &L.ffnnorm, &o.fin, &o.xqE, &o.dxE, kNE, 1, f_eps);
            }
            // --- MoE (3-dispatch batched top-8) ---
            gemvf32(o.probs, kNEXP, kNE, o.fin, L.router);
            RouterArgs ra{o.probs, L.probsb, o.sel, o.wgt, w_scale};
            d_tg("k_router_topk", 1, 1, 128, &ra, sizeof(ra));
            prof_flush(P_ROUTER);
            // Phase 2B: flush router -> resolve top-8 into the mlock'd Hot
            // Expert Arena (NVMe streaming on miss) -> rebind expertBuf for
            // the two batched MoE GEMVs, then residentBuf for the rest.
            const uint32_t asel = o.arena_sel;
            if (arena_mode && err.empty()) {
                [en endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    NSString * msg = cb.error.localizedDescription ?: @"unknown MTLCommandBufferError";
                    err = std::string("metal: arena sync status=") +
                          std::to_string((NSInteger)cb.status) + ": " + msg.UTF8String;
                    fprintf(stderr, "m64metal: %s\n", err.c_str());
                } else {
                    float * S = (float *)scr.contents;
                    uint32_t sel8[kTOPK];
                    for (int i = 0; i < kTOPK; i++) sel8[i] = ((const uint32_t *)S)[o.sel + i];
                    uint32_t slots8[kTOPK];
                    if (!arena->resolve_top8(s_phys[il], sel8, slots8, err))
                        fprintf(stderr, "m64metal: arena resolve failed: %s\n", err.c_str());
                    for (int i = 0; i < kTOPK; i++) ((uint32_t *)S)[asel + i] = slots8[i];
                }
                cb = [rt.queue() commandBuffer];
                en = [cb computeCommandEncoder];
            }
            {
                const uint32_t seloff = arena_mode ? asel : o.sel;
                const uint64_t mst = arena_mode ? exp_stride : L.moe_stride;  // arena slot stride vs per-layer pack stride
                if (L.moe_q4k) {
                    // Milestone 4B FP32-input pipeline: Q4_K gate/up -> f32 SITU
                    // -> Q5_0/Q5_1 down+reduce. sel holds direct blob indexes
                    // (expert ids on pack, arena slots in arena mode) so wtab=0.
                    MoeGateUpQ4KF32Args g4{arena_mode ? 0ul : (unsigned long)(L.experts + L.moe_off[0]),
                                           arena_mode ? (unsigned long)L.moe_off[1] : (unsigned long)(L.experts + L.moe_off[1]),
                                           o.fin, o.gate2, o.ubuf, seloff, 0u,
                                           kNF, kNE, (unsigned int)mst};
                    if (arena_mode) curPack = expertBuf;
                    if (gemv_m2) d_tg("k_gemv_gateup_batch8_q4k_f32in_m2", (kTOPK * kNF + 7) / 8, 1, 256, &g4, sizeof(g4));
                    else d_tg("k_gemv_gateup_batch8_q4k_f32in", (kTOPK * 2 * kNF + 7) / 8, 1, 256, &g4, sizeof(g4));
                    // Milestone 9A: SITU writes h back into o.gate2 IN-PLACE
                    // (k_situ_f32_batch reads (goff,p,i)/(uoff,p,i) then writes
                    // (hoff,p,i) from the same thread, so hoff==goff with equal
                    // strides is element-safe). This frees o.down_tmp
                    // ([8][kNE]) for the 2-stage parallel down stage 1.
                    SituF32BatchArgs sf{o.gate2, o.ubuf, o.gate2, kNF, kNF, kNF, kNF, kTOPK,
                                        situ_beta, situ_lb, 1.0f / situ_beta, 1.0f / situ_lb};
                    d_th2("k_situ_f32_batch", kNF, kTOPK, 128, &sf, sizeof(sf));
                    prof_flush(P_GATEUP);
                    MoeDownQ5F32Args md{arena_mode ? (unsigned long)L.moe_off[2] : (unsigned long)(L.experts + L.moe_off[2]),
                                        o.gate2, o.acc, o.wgt, seloff, 0u, 0u,
                                        kNE, kNF, (unsigned int)mst, 0u};
                    if (gemv_r3 && !trace_slot) {
                        // 2-stage: (144,8)=1,152 TGs of 256 unweighted per-expert
                        // GEMV into down_tmp, then fused weighted reduce + ffnconv
                        // + add (mirrors the Q8_0 k_gemv_down_batch8_m2 path).
                        md.accoff = o.down_tmp;
                        d_tg(L.moe_dt[2] == DType::Q5_1 ? "k_gemv_down_batch8_q5_1_f32in_m2"
                                                        : "k_gemv_down_batch8_q5_0_f32in_m2",
                             (kNE + 15) / 16, kTOPK, 256, &md, sizeof(md));
                        if (arena_mode) curPack = residentBuf;   // ffnconv lives in residentBuf
                        DownReduceConvArgs drc{o.down_tmp, o.wgt,
                                               o.rs + ((uint32_t)il * (uint32_t)kCH + (uint32_t)(2 * kKVD + kNE)) * kHR,
                                               o.cur, L.ffnconv, kNE, kConvStatic};
                        d_th("k_down_reduce8_conv_add", kNE, 128, &drc, sizeof(drc));
                    } else {
                        // A/B fallback: single-stage 72-TG reduce + separate conv/add.
                        if (gemv_m2)
                            d_tg(L.moe_dt[2] == DType::Q5_1 ? "k_gemv_down_reduce8_q5_1_f32in_m4"
                                                            : "k_gemv_down_reduce8_q5_0_f32in_m4",
                                 (kNE + 31) / 32, 1, 256, &md, sizeof(md));
                        else
                            d_tg(L.moe_dt[2] == DType::Q5_1 ? "k_gemv_down_reduce8_q5_1_f32in"
                                                            : "k_gemv_down_reduce8_q5_0_f32in",
                                 (kNE + 7) / 8, 1, 256, &md, sizeof(md));
                        if (arena_mode) curPack = residentBuf;
                        conv(il, 2 * kKVD + kNE, kNE, o.acc, o.tA, L.ffnconv, 0, kNE, kConvStatic);
                        xfer2(o.cur, o.tA, 0, kNE, 0, 0, 0, 0, 0);
                    }
                    prof_flush(P_DOWN);
                } else {
                GemvArgs gu{arena_mode ? 0ul : L.experts,
                            arena_mode ? (unsigned long)kPartBytes : L.experts + kPartBytes,
                            o.xqE, o.dxE, o.gate2, o.ubuf,
                            kNF, 0, (uint32_t)exp_stride, seloff};
                if (arena_mode) curPack = expertBuf;
                if (gemv_m2) d_tg("k_gemv_gateup_batch8_m2", (kTOPK * kNF + 7) / 8, 1, 256, &gu, sizeof(gu));
                else d_tg("k_gemv_gateup_batch8", (kTOPK * 2 * kNF + 7) / 8, 1, 256, &gu, sizeof(gu));
                SituArgs sa{o.gate2, o.ubuf, o.hq, o.dh, kTOPK * (kNF / 32), situ_beta, situ_lb,
                            1.0f / situ_beta, 1.0f / situ_lb};
                d_th("k_situ_quant8", kTOPK * kNF, 128, &sa, sizeof(sa));
                prof_flush(P_GATEUP);
                if (gemv_r3 && !trace_slot) {
                    MoeDownArgs md{arena_mode ? (unsigned long)(2 * kPartBytes) : L.experts + 2 * kPartBytes,
                                   o.hq, o.dh, o.down_tmp, o.wgt, 0, kNE, (uint32_t)exp_stride, seloff};
                    d_tg("k_gemv_down_batch8_m2", (kNE + 15) / 16, kTOPK, 256, &md, sizeof(md));
                    if (arena_mode) curPack = residentBuf;
                    DownReduceConvArgs drc{o.down_tmp, o.wgt,
                                           o.rs + ((uint32_t)il * (uint32_t)kCH + (uint32_t)(2 * kKVD + kNE)) * kHR,
                                           o.cur, L.ffnconv, kNE, kConvStatic};
                    d_th("k_down_reduce8_conv_add", kNE, 128, &drc, sizeof(drc));
                } else {
                    MoeDownArgs md{arena_mode ? (unsigned long)(2 * kPartBytes) : L.experts + 2 * kPartBytes,
                                   o.hq, o.dh, o.acc, o.wgt, 0, kNE, (uint32_t)exp_stride, seloff};
                    if (gemv_m2) d_tg("k_gemv_down_reduce8_m4", (kNE + 31) / 32, 1, 256, &md, sizeof(md));
                    else d_tg("k_gemv_down_reduce8", (kNE + 15) / 16, 1, 256, &md, sizeof(md));
                    if (arena_mode) curPack = residentBuf;
                    conv(il, 2 * kKVD + kNE, kNE, o.acc, o.tA, L.ffnconv, 0, kNE, kConvStatic);
                    xfer2(o.cur, o.tA, 0, kNE, 0, 0, 0, 0, 0);
                }
                prof_flush(P_DOWN);
                }
            }
            if (trace_slot) { char tb[32]; snprintf(tb, sizeof(tb), "s%d_p%d_end", il, ip); trace_flush(tb); }
        };

        // ---------------- token body ----------------
        {   // embedding gather + first norm
            uint32_t toku = (uint32_t)(uint32_t)token;
            EmbArgs ea{tok_embd, o.emb, 72};
            [en setBuffer:scr offset:0 atIndex:0];
            [en setBytes:&ea length:sizeof(ea) atIndex:1];
            [en setBuffer:curPack offset:0 atIndex:2];
            [en setBytes:&toku length:4 atIndex:3];
            d_th("k_emb_gather", 72, 32, &ea, sizeof(ea));
            rms(&o.emb, &tok_norm, &o.cur, kNE, 1, 1, f_eps);
            prof_flush(P_OTHER);
            if (trace_slot) trace_flush("tok_norm");
        }
        for (int ip = 0; ip < 6; ip++) slot(ip, ip, nullptr);          // prelude
        xfer2(o.cur, 0, o.prelude_out, kNE, o.blocks, 0, o.prelude_blk, kNE, 1);
        prof_flush(P_OTHER);
        int npre = st.nblk;
        for (int pass = 0; pass < 3; pass++) {                          // core loop
            int nbank = 0;
            if (pass > 0) {
                nbank = st.nblk - 2;
                if (nbank > 0) xfer2(o.blocks + 2 * kNE, 0, o.bank, (uint32_t)nbank * kNE, 0, 0, 0, 0, 1);
                // Sequential: pair (cur->bank_out) must finish before (prelude_out->cur).
                xfer2(o.cur, 0, o.bank_out, kNE, 0, 0, 0, 0, 1);
                xfer2(o.prelude_out, 0, o.cur, kNE, 0, 0, 0, 0, 1);
                xfer2(o.prelude_blk, 0, o.blocks, kNE, 0, 0, 0, 0, 1);
                prof_flush(P_OTHER);
                st.nblk = npre;
            }
            for (int ip = 6; ip < 18; ip++) {
                int il = pass == 0 ? ip : 24 + (pass - 1) * 12 + (ip - 6);
                if (pass > 0) {
                    const LL & lp = loop[s_loop[il]];
                    do_mix(o.bank, nbank, o.bank_out, lp.msgq, 4, o.msgs);
                }
                slot(il, ip, pass > 0 ? &loop[s_loop[il]] : nullptr);
            }
        }
        for (int ip = 18; ip < 24; ip++) slot(ip, ip, nullptr);         // coda
        // ---------------- output (LM head; skipped when !need_logits) -------
        if (need_logits) {
            do_mix(o.blocks, st.nblk, o.cur, out_q, 1, o.mixed);
            {
                uint32_t oy = (trace_slot || nanprobe) ? o.fin : M64_NOFFU;
                rms_quant(&o.mixed, &out_norm, &oy, &o.xqE, &o.dxE, kNE, 1, f_eps);
            }
            gemv2304(o.logits, (uint32_t)vocab_rows, o.xqE, o.dxE, outw);
            SoftcapArgs sca{o.logits, (uint32_t)vocab_rows, cap};
            d_th("k_softcap", (uint32_t)vocab_rows, 256, &sca, sizeof(sca));
            ArgmaxArgs ara{o.logits, o.amax, (uint32_t)vocab_eff};
            d_tg("k_argmax", 1, 1, 1024, &ara, sizeof(ara));
        }

        [en endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            NSString * msg = cb.error.localizedDescription ?: @"unknown MTLCommandBufferError";
            err = std::string("metal: command buffer status=") +
                  std::to_string((NSInteger)cb.status) + ": " + msg.UTF8String;
            fprintf(stderr, "m64metal: GPU fault: %s\n", err.c_str());
        } else if (profile && need_logits) {
            prof_ms[P_LMHEAD] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            prof_disp[P_LMHEAD] += cur_disp;
            const char * pnames[P_NCAT] = {
                "other", "mix", "ctrl", "qkv", "conv_clip_attn",
                "wo_post", "router", "moe_gateup", "moe_down", "lm_head"
            };
            double tot_ms = 0.0; uint32_t tot_d = 0;
            for (int i = 0; i < P_NCAT; i++) { tot_ms += prof_ms[i]; tot_d += prof_disp[i]; }
            fprintf(stderr, "M64_PROFILE pos=%lld total=%.3f ms dispatches=%u\n", (long long)pos, tot_ms, tot_d);
            for (int i = 0; i < P_NCAT; i++) {
                fprintf(stderr, "  %-16s %7.3f ms (%5.1f%%)  disp=%4u\n",
                        pnames[i], prof_ms[i], tot_ms > 0 ? (100.0 * prof_ms[i] / tot_ms) : 0.0, prof_disp[i]);
            }
        }
        if (getenv("M64_NANPROBE")) {
            const float * S = (const float *)scr.contents;
            auto probe = [&](const char * nm, uint32_t off, uint32_t n) {
                uint32_t bad = 0; float mv = 0;
                for (uint32_t i = 0; i < n; i++) { float v = S[off + i];
                    if (!isfinite(v)) bad++; float a = fabsf(v); if (a > mv) mv = a; }
                printf("probe %-8s off=%7u bad=%7u max=%.4g\n", nm, off, bad, mv);
            };
            probe("emb", o.emb, kNE); probe("cur", o.cur, kNE); probe("Q", o.Q, kNH * kDH);
            probe("K", o.K, kKVD); probe("attnb", o.attnb, kNH * kDH); probe("probs", o.probs, kNEXP);
            probe("fin", o.fin, kNE); probe("mixed", o.mixed, kNE);
            if (need_logits) {
                uint32_t badl = 0; float maxl = 0;
                for (int i = 0; i < vocab_rows; i++) { float v = S[o.logits + i];
                    if (!isfinite(v)) badl++; float a = fabsf(v); if (a > maxl) maxl = a; }
                printf("probe logits bad=%u max=%.4g\n", badl, maxl);
            }
        }
        if (err.empty()) {
            if (need_logits) {
                const float * sbase = (const float *)scr.contents;
                r.argmax = (int32_t)*(const uint32_t *)(sbase + o.amax);
            }
            pos++;
        }
        return r;
    }

    // ========================= chunked batched prefill =========================
    // Runs B tokens [pos .. pos+B) in lockstep inside one MTLCommandBuffer (or
    // per-MoE-layer commit+resolve in arena_mode). Keeps o.rs and kvBuf layouts
    // identical to gpu_step so RAM/SSD prefix caching and subsequent decode stay
    // 100% compatible.
    StepResult gpu_prefill_batch(const int32_t * toks, uint32_t B, std::string & err, bool need_logits,
                                 uint32_t lm_rows = 0) {
        StepResult r{(const float *)scr.contents + o.logits, vocab_eff, 0};
        if (B == 0u) return r;
        if (B > (uint32_t)kBMax || pos + (int64_t)B > ctx) {
            err = "metal: batch bounds exceed kBMax or ctx";
            return r;
        }
        const bool gemv_m2 = (getenv("M64_NO_M2") == nullptr);
        const bool gemv_r3 = !(getenv("M64_GEMV_R3") && strcmp(getenv("M64_GEMV_R3"), "0") == 0);
        const bool gemv_r4 = !(getenv("M64_GEMV_R4") && strcmp(getenv("M64_GEMV_R4"), "0") == 0);
        const bool prof_batch = (getenv("M64_PROFILE_BATCH") != nullptr) && !arena_mode &&
                                (pos == (int64_t)prefill_chunk || pos == 32);
        enum BProfCat { BP_OTHER = 0, BP_MIX, BP_CTRL, BP_QKV, BP_ATTN, BP_WO, BP_ROUTER, BP_GATEUP, BP_DOWN, BP_LMHEAD, BP_NCAT };
        double bprof_ms[BP_NCAT] = {};
        uint32_t bprof_disp[BP_NCAT] = {};
        uint32_t bcur_disp = 0;
        const uint32_t kBlkStride = (uint32_t)kBBlkMax * kNE;

        uint32_t * tok_dst = (uint32_t *)((float *)scr.contents + o.b_tok);
        for (uint32_t b = 0; b < B; b++) tok_dst[b] = (uint32_t)toks[b];

        id<MTLCommandBuffer> cb = [rt.queue() commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        id<MTLBuffer> curPack = arena_mode ? residentBuf : packBuf;
        auto bprof_flush = [&](int cat) {
            if (!prof_batch || !err.empty()) return;
            [en endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                NSString * msg = cb.error.localizedDescription ?: @"unknown MTLCommandBufferError";
                err = std::string("metal: batch profile status=") +
                      std::to_string((NSInteger)cb.status) + ": " + msg.UTF8String;
            } else {
                bprof_ms[cat] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                bprof_disp[cat] += bcur_disp;
            }
            bcur_disp = 0;
            cb = [rt.queue() commandBuffer];
            en = [cb computeCommandEncoder];
        };

        auto disp = [&](const char * fn, bool tg, MTLSize grid, MTLSize nth,
                        const void * a, size_t sz, id<MTLBuffer> b2, id<MTLBuffer> b3 = nil) {
            if (!err.empty()) return;
            id<MTLComputePipelineState> p = rt.pipeline(fn);
            if (!p) { err = std::string("pipeline ") + fn; return; }
            [en setComputePipelineState:p];
            [en setBuffer:scr offset:0 atIndex:0];
            [en setBytes:a length:sz atIndex:1];
            [en setBuffer:b2 offset:0 atIndex:2];
            if (b3) [en setBuffer:b3 offset:0 atIndex:3];
            if (tg) [en dispatchThreadgroups:grid threadsPerThreadgroup:nth];
            else    [en dispatchThreads:grid threadsPerThreadgroup:nth];
            bcur_disp++;
        };
        auto d_tg = [&](const char * fn, uint tgx, uint tgy, uint nth, const void * a, size_t sz) {
            disp(fn, true, MTLSizeMake(tgx, tgy, 1), MTLSizeMake(nth, 1, 1), a, sz, curPack);
        };
        auto d_th = [&](const char * fn, uint total, uint nth, const void * a, size_t sz) {
            disp(fn, false, MTLSizeMake(total, 1, 1), MTLSizeMake(nth, 1, 1), a, sz, curPack);
        };
        auto rms1 = [&](const uint32_t * xr, const uint64_t * wr, const uint32_t * yr,
                        uint32_t k, uint32_t rows, uint32_t hasw, float eps) {
            RMSArgs A{}; A.k = k; A.rows = rows; A.hasw = hasw; A.eps = eps;
            for (uint32_t i = 0; i < rows; i++) { A.xr[i] = xr[i]; A.wr[i] = wr ? wr[i] : 0; A.yr[i] = yr[i]; }
            d_tg("k_rms", rows, 1, 256, &A, sizeof(A));
        };
        auto xfer2 = [&](uint32_t a0, uint32_t b0, uint32_t c0, uint32_t n0,
                         uint32_t a1, uint32_t b1, uint32_t c1, uint32_t n1, uint32_t mode) {
            XferArgs A{{a0, a1}, {b0, b1}, {c0, c1}, {n0, n1}, mode};
            uint32_t n = mode == 0 ? 1 : (n1 ? 2 : 1);
            d_tg("k_xfer", n, 1, 256, &A, sizeof(A));
        };
        auto do_mix = [&](uint32_t states, int n, uint32_t extra, uint64_t qpack, int nq, uint32_t out) {
            uint32_t xr[kMAXS + 1], yr[kMAXS + 1];
            int S = n + 1;
            for (int i = 0; i < S - 1; i++) xr[i] = states + (uint32_t)(n - 1 - i) * kNE;
            xr[S - 1] = extra;
            for (int i = 0; i < S; i++) yr[i] = o.stn + (uint32_t)i * kNE;
            rms1(xr, nullptr, yr, kNE, (uint32_t)S, 0, 1e-6f);
            if (gemv_r3) {
                MixFusedArgs ma{qpack, o.stn, out, (uint32_t)S, (uint32_t)nq, 1.0f / sqrtf((float)kNE), 0};
                d_tg("k_mix_fused", (uint32_t)nq, 1, 256, &ma, sizeof(ma));
            } else {
                MixDotArgs da{o.stn, o.sc, qpack, (uint32_t)S, 0, 1.0f / sqrtf((float)kNE)};
                d_tg("k_mix_dot", (uint32_t)nq, (uint32_t)((S + 7) / 8), 256, &da, sizeof(da));
                MixSoftArgs sa{o.sc, (uint32_t)S};
                d_tg("k_mix_soft", (uint32_t)nq, 1, 64, &sa, sizeof(sa));
                MixCombArgs ca{o.stn, o.sc, out, (uint32_t)S, (uint32_t)nq};
                d_tg("k_mix_comb", (uint32_t)nq, 1, 256, &ca, sizeof(ca));
            }
        };
        auto do_ctrl = [&](const uint32_t * in, const uint32_t * msg, const uint32_t * out,
                           const uint64_t * wd, const uint64_t * wg, const uint64_t * wf, int nc) {
            uint32_t xr[8], yr[8];
            uint64_t wr[8];
            for (int j = 0; j < nc; j++) {
                xr[2 * j] = in[j];   yr[2 * j] = o.gcat + (uint32_t)(2 * j) * kNE;
                xr[2 * j + 1] = msg[j]; yr[2 * j + 1] = o.gcat + (uint32_t)(2 * j + 1) * kNE;
                wr[2 * j] = wr[2 * j + 1] = 0;
            }
            rms1(xr, wr, yr, kNE, (uint32_t)(2 * nc), 0, 1e-6f);
            if (gemv_r4) {
                CtrlSplitArgs A{};
                for (int j = 0; j < nc; j++) {
                    A.in[j] = in[j]; A.msg[j] = msg[j]; A.out[j] = out[j];
                    A.wdown[j] = wd[j]; A.wgain[j] = wg[j]; A.wforg[j] = wf[j];
                }
                A.gcat = o.gcat; A.hh = o.ctrl_hh; A.eps = 1e-6f;
                A.grank = 64; A.ne = kNE; A.nc = (uint32_t)nc;
                d_tg("k_ctrl_down", 64, (uint32_t)nc, 256, &A, sizeof(A));
                d_tg("k_ctrl_up", (kNE + 31) / 32, (uint32_t)nc, 32, &A, sizeof(A));
            } else {
                CtrlArgs A{};
                for (int j = 0; j < nc; j++) {
                    A.in[j] = in[j]; A.msg[j] = msg[j]; A.out[j] = out[j];
                    A.wdown[j] = wd[j]; A.wgain[j] = wg[j]; A.wforg[j] = wf[j];
                    A.gb[j] = o.gb + (uint32_t)j * kNE; A.fb[j] = o.fb + (uint32_t)j * kNE;
                }
                A.gcat = o.gcat; A.eps = 1e-6f; A.grank = 64; A.ne = kNE;
                d_tg("k_control", (uint32_t)nc, 1, 1024, &A, sizeof(A));
            }
        };
        auto gemm_sg = [&](uint64_t w, uint32_t x, uint32_t xs, uint32_t y, uint32_t ys,
                           uint32_t k, uint32_t nout) {
            GemmF32SgArgs A{w, x, xs, y, ys, k, nout, B, 0u, 0u};
            disp("k_gemm_q8_f32in_sg", true,
                 MTLSizeMake((nout + 31) / 32, (B + 31) / 32, 1),
                 MTLSizeMake(32, 4, 1), &A, sizeof(A), curPack);
        };
        // Milestone 4B fix: f32 small dense projections (wsink/kcovp/vcovp) in Q4_K_S packs.
        // b_n3 already holds the normalized FP32 rows (stride 3*kNE), so no extra staging.
        auto gemv_f32_b = [&](uint64_t w, uint32_t x, uint32_t y, uint32_t ys, uint32_t nout) {
            GemvF32BatchArgs A{w, x, y, 3 * kNE, ys, nout, kNE, B, 0u};
            d_tg("k_gemv_f32_batch", nout, B, 256, &A, sizeof(A));
        };
        auto rms_b = [&](uint64_t wp, uint32_t x, uint32_t xs, uint32_t xr,
                         uint32_t y, uint32_t ys, uint32_t yr,
                         uint32_t q, uint32_t d, uint32_t qs, uint32_t ds,
                         uint32_t k, uint32_t rows, uint32_t hasw, uint32_t q8, float eps) {
            RmsBatchArgs A{wp, x, xs, xr, y, ys, yr, q, d, qs, ds, k, B, rows, hasw, q8, 0u, eps};
            d_tg("k_rms_batch", rows, B, 256, &A, sizeof(A));
        };
        auto conv_b = [&](uint32_t il, uint32_t sec_chan, uint32_t C, uint32_t x, uint32_t xs,
                          uint32_t y, uint32_t ys, uint64_t wp, uint32_t dyn, uint32_t ds,
                          uint32_t grp, float scale) {
            ConvBatchArgs A{wp, x, y, o.rs + (il * (uint32_t)kCH + sec_chan) * kHR,
                            dyn, C, B, xs, ys, ds, grp, dyn ? 1u : 0u, 0u, scale, 0.0f};
            d_th("k_conv_batch", C, 128, &A, sizeof(A));
        };
        auto clip_b = [&](uint32_t coff, uint32_t cs, uint64_t soff, uint32_t groups,
                          uint32_t per, float thresh) {
            ClipBatchArgs A{soff, coff, cs, groups, per, B, 0u, thresh};
            d_tg("k_clip_batch", groups, B, 128, &A, sizeof(A));
        };
        // ---- Milestone 9B: batched xfer / mix / ctrl (replace per-token for(b)) ----
        // The batch kernels mirror k_mix_fused (r3) and the k_ctrl split path (r4),
        // so keep the sequential loops when those fallbacks are active.
        const bool bsmall = !getenv("M64_NO_BSMALL") && !getenv("M64_NO_GEMV_R3") &&
                            !getenv("M64_NO_GEMV_R4");
        auto d_tg3 = [&](const char * fn, uint tgx, uint tgy, uint tgz, uint nth,
                         const void * a, size_t sz) {
            disp(fn, true, MTLSizeMake(tgx, tgy, tgz), MTLSizeMake(nth, 1, 1), a, sz, curPack);
        };
        // is_copy=1: SCR[c + b*c_stride] = SCR[a + b*a_stride]; else a += b (k_xfer shape).
        auto xfer_b = [&](bool is_copy, uint32_t a0, uint32_t a_s, uint32_t b0, uint32_t b_s,
                          uint32_t c0, uint32_t c_s, uint32_t n) {
            XferBatchArgs A{};
            A.a[0] = a0; A.b[0] = b0; A.c[0] = c0; A.n[0] = n;
            A.a_stride[0] = a_s; A.b_stride[0] = b_s; A.c_stride[0] = c_s;
            A.B = B; A.npairs = 1u; A.is_copy = is_copy ? 1u : 0u;
            d_tg3("k_xfer_batch", (n + 255u) / 256u, B, 1u, 256, &A, sizeof(A));
        };
        // Batched do_mix: rows r<n read newest-first blocks, row n = extra.
        auto do_mix_b = [&](uint32_t states, uint32_t st_stride, int n, uint32_t extra,
                            uint32_t ex_stride, uint64_t qpack, int nq,
                            uint32_t out, uint32_t out_stride) {
            const uint32_t stn_stride = (uint32_t)(kBBlkMax + 1) * kNE;
            MixRmsBatchArgs ra{states, st_stride, extra, ex_stride,
                               o.b_stn, stn_stride, (uint32_t)n, B, kNE, 0u};
            d_tg("k_mix_rms_batch", (uint32_t)(n + 1), B, 256, &ra, sizeof(ra));
            MixFusedBatchArgs ma{};
            ma.qpack = qpack; ma.stnoff = o.b_stn; ma.stn_stride = stn_stride;
            ma.outoff = out; ma.out_stride = out_stride;
            ma.S = (uint32_t)(n + 1); ma.nq = (uint32_t)nq; ma.B = B;
            ma.scale = 1.0f / sqrtf((float)kNE);
            d_tg("k_mix_fused_batch", (uint32_t)nq, B, 256, &ma, sizeof(ma));
        };
        // Batched do_ctrl: two rms passes write per-token gcat pairs, then
        // down/up read a per-token hh region (never the shared o.ctrl_hh).
        auto do_ctrl_b = [&](uint32_t in0, uint32_t in_stride, uint32_t in_row,
                             uint32_t msg0, uint32_t msg_stride, uint32_t msg_row,
                             uint32_t out0, uint32_t out_stride, uint32_t out_row,
                             const uint64_t * wd, const uint64_t * wg, const uint64_t * wf, int nc) {
            const uint32_t gcstr = (uint32_t)(2u * (uint32_t)nc * kNE);
            rms_b(0ull, in0, in_stride, in_row, o.b_ctrl_gcat, gcstr, 2u * kNE,
                  M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, (uint32_t)nc, 0u, 0u, 1e-6f);
            rms_b(0ull, msg0, msg_stride, msg_row, o.b_ctrl_gcat + kNE, gcstr, 2u * kNE,
                  M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, (uint32_t)nc, 0u, 0u, 1e-6f);
            CtrlBatchSplitArgs A{};
            for (int j = 0; j < nc; j++) {
                A.in[j] = in0 + (uint32_t)j * in_row;
                A.msg[j] = msg0 + (uint32_t)j * msg_row;
                A.out[j] = out0 + (uint32_t)j * out_row;
                A.wdown[j] = wd[j]; A.wgain[j] = wg[j]; A.wforg[j] = wf[j];
            }
            A.in_stride = in_stride; A.msg_stride = msg_stride; A.out_stride = out_stride;
            A.gcat = o.b_ctrl_gcat; A.gcat_stride = gcstr;
            A.hh = o.b_ctrl_hh; A.hh_stride = 4u * 64u;
            A.grank = 64u; A.ne = kNE; A.nc = (uint32_t)nc; A.B = B;
            A.eps = 1e-6f;
            d_tg3("k_ctrl_down_batch", 64u, (uint32_t)nc, B, 256, &A, sizeof(A));
            d_tg3("k_ctrl_up_batch", (kNE + 31u) / 32u, (uint32_t)nc, B, 32, &A, sizeof(A));
        };

        struct { int nblk; } st = {0};
        auto batch_slot = [&](int il, int ip, const LL * lp) {
            const LW & L = phys[s_phys[il]];
            const bool is_swa = s_swa[il];
            // --- attn inputs ---
            if (L.aquery) {
                if (bsmall)
                    do_mix_b(o.b_blocks, kBlkStride, st.nblk, o.b_cur, kNE,
                             L.aquery, L.nq_rows, o.b_mixed, 3 * kNE);
                else
                    for (uint32_t b = 0; b < B; b++)
                        do_mix(o.b_blocks + b * kBlkStride, st.nblk, o.b_cur + b * kNE,
                               L.aquery, L.nq_rows, o.b_mixed + b * (3 * kNE));
                bprof_flush(BP_MIX);
            } else {
                if (bsmall)
                    xfer_b(true, o.b_cur, kNE, 0u, 0u, o.b_mixed, 3 * kNE, kNE);
                else
                    for (uint32_t b = 0; b < B; b++)
                        xfer2(o.b_cur + b * kNE, 0, o.b_mixed + b * (3 * kNE), kNE, 0, 0, 0, 0, 1);
                bprof_flush(BP_OTHER);
            }
            if (lp) {
                if (bsmall) {
                    do_ctrl_b(o.b_mixed, 3 * kNE, is_swa ? 0u : kNE,
                              o.b_msgs, 4 * kNE, kNE,
                              o.b_ctrlout, 3 * kNE, kNE, lp->down, lp->gain, lp->forg, 3);
                } else {
                    for (uint32_t b = 0; b < B; b++) {
                        uint32_t in3[3], msgp[3], outp[3];
                        for (int j = 0; j < 3; j++) {
                            in3[j]  = o.b_mixed + b * (3 * kNE) + (is_swa ? 0u : (uint32_t)j * kNE);
                            msgp[j] = o.b_msgs + b * (4 * kNE) + (uint32_t)j * kNE;
                            outp[j] = o.b_ctrlout + b * (3 * kNE) + (uint32_t)j * kNE;
                        }
                        do_ctrl(in3, msgp, outp, lp->down, lp->gain, lp->forg, 3);
                    }
                }
                bprof_flush(BP_CTRL);
            }
            // --- input rmsnorms (FP32 output feeds k_gemm_q8_f32in_sg directly) ---
            const uint32_t src_att = lp ? o.b_ctrlout : o.b_mixed;
            if (is_swa && !lp) {
                rms_b(L.normattn, src_att, 3 * kNE, kNE, o.b_n3, 3 * kNE, kNE,
                      M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, 1u, 1u, 0u, f_eps);
            } else if (is_swa) {
                rms_b(L.normattn, o.b_ctrlout, 3 * kNE, kNE, o.b_n3, 3 * kNE, kNE,
                      M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, 3u, 1u, 0u, f_eps);
            } else {
                rms_b(L.nq, src_att,           3 * kNE, kNE, o.b_n3,           3 * kNE, kNE,
                      M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, 1u, 1u, 0u, f_eps);
                rms_b(L.nk, src_att + kNE,     3 * kNE, kNE, o.b_n3 + kNE,     3 * kNE, kNE,
                      M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, 1u, 1u, 0u, f_eps);
                rms_b(L.nv, src_att + 2 * kNE, 3 * kNE, kNE, o.b_n3 + 2 * kNE, 3 * kNE, kNE,
                      M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, 1u, 1u, 0u, f_eps);
            }
            const uint32_t nA = o.b_n3;
            const uint32_t nB = (is_swa && !lp) ? o.b_n3 : (o.b_n3 + kNE);
            const uint32_t nC = (is_swa && !lp) ? o.b_n3 : (o.b_n3 + 2 * kNE);

            // --- Q K V R sink gate (+ dyn conv proj) via tensor-core Q8_0 GEMM ---
            gemm_sg(L.wq,    nA, 3 * kNE, o.b_Q,     kNH * kDH,   kNE, kNH * kDH);
            gemm_sg(L.wk,    nB, 3 * kNE, o.b_K,     kKVD,        kNE, kKVD);
            gemm_sg(L.wv,    nC, 3 * kNE, o.b_V2,    kKVD,        kNE, kKVD);
            gemm_sg(L.wrel,  nA, 3 * kNE, o.b_Rr,    kNH * kNREL, kNE, kNH * kNREL);
            if (L.wsink_f32) gemv_f32_b(L.wsink, nA, o.b_sinkb, kNH, kNH);
            else gemm_sg(L.wsink, nA, 3 * kNE, o.b_sinkb, kNH, kNE, kNH);
            gemm_sg(L.wgate, nA, 3 * kNE, o.b_gateb, kNH * kDH, kNE, kNH * kDH);
            if (!is_swa) {
                if (L.kcovp_f32) gemv_f32_b(L.kcovp, nB, o.b_dynb, 32u, 32u);
                else gemm_sg(L.kcovp, nB, 3 * kNE, o.b_dynb, 32u, kNE, 32u);
                conv_b((uint32_t)il, 0u,   kKVD, o.b_K,  kKVD, o.b_K,  kKVD, L.kconv, o.b_dynb, 32u, 128u, 1.0f / 3.0f);
                if (L.vcovp_f32) gemv_f32_b(L.vcovp, nC, o.b_dynb, 32u, 32u);
                else gemm_sg(L.vcovp, nC, 3 * kNE, o.b_dynb, 32u, kNE, 32u);
                conv_b((uint32_t)il, kKVD, kKVD, o.b_V2, kKVD, o.b_V2, kKVD, L.vconv, o.b_dynb, 32u, 128u, 1.0f / 3.0f);
            } else {
                conv_b((uint32_t)il, 0u,   kKVD, o.b_K,  kKVD, o.b_K,  kKVD, L.kconv, 0u, 0u, 128u, kConvStatic);
                conv_b((uint32_t)il, kKVD, kKVD, o.b_V2, kKVD, o.b_V2, kKVD, L.vconv, 0u, 0u, 128u, kConvStatic);
            }
            bprof_flush(BP_QKV);
            // --- clips, V per-head rmsnorm, KV write, batched causal attention ---
            clip_b(o.b_Q,  kNH * kDH,   L.qclip,   kNH,        kDH,   f_qk_clip);
            clip_b(o.b_K,  kKVD,        M64_NOFF,  kKVD / kDH, kDH,   f_qk_clip);
            rms_b(0ull, o.b_V2, kKVD, kDH, o.b_Vn, kKVD, kDH,
                  M64_NOFFU, M64_NOFFU, 0u, 0u, kDH, 8u, 0u, 0u, f_eps);
            clip_b(o.b_Rr, kNH * kNREL, L.relclip, kNH,        kNREL, f_rel_clip);
            {
                KVWriteBatchF16Args kw{
                    kv_k_off[il], kv_v_off[il], o.b_K, o.b_Vn, kKVD, kKVD, kKVD,
                    B, (uint32_t)pos, is_swa ? 1u : 0u, kv_cap[il], 0u
                };
                if (kv_q8)
                    // Q8 grid = (8 KV heads, B) threadgroups x 32 lanes (5A parity test).
                    disp("k_kv_write_q8_batch", true,
                         MTLSizeMake(8, B, 1), MTLSizeMake(32, 1, 1), &kw, sizeof(kw), kvBuf);
                else
                    disp("k_kv_write_f16_batch", false,
                         MTLSizeMake(kKVD, B, 1), MTLSizeMake(128, 1, 1), &kw, sizeof(kw), kvBuf);
                AttnBatchF16Args aa{
                    L.relbias, kv_k_off[il], kv_v_off[il],
                    o.b_Q, o.b_Rr, o.b_sinkb, o.b_attnb,
                    kNH * kDH, kNH * kNREL, kNH, kNH * kDH,
                    (uint32_t)pos, B, is_swa ? 1u : 0u, (uint32_t)n_swa, kv_cap[il], 0u,
                    1.0f / sqrtf((float)kDH), log_scale_floor, log_scale_alpha, 0.0f
                };
                // Milestone 4C: tiled FlashAttention prefill (tensor-core PV,
                // 32-query-row threadgroups) wins from B >= 8 for both full
                // and SWA attention; the scalar per-token kernel still wins
                // for tiny batches.
                if (B >= 8u)
                    disp(kv_q8 ? "k_attn_q8_flash_prefill" : "k_attn_f16_flash_prefill", true,
                         MTLSizeMake(kNH, (B + 31u) / 32u, 1), MTLSizeMake(32, 4, 1),
                         &aa, sizeof(aa), curPack, kvBuf);
                else
                    disp(kv_q8 ? "k_attn_head_q8_batch" : "k_attn_head_f16_batch", true,
                         MTLSizeMake(kNH, B, 1), MTLSizeMake(32, 1, 1),
                         &aa, sizeof(aa), curPack, kvBuf);
            }
            bprof_flush(BP_ATTN);
            // --- gate mul, wo (sg GEMM), postconv, residual ---
            {
                GateMulBatchArgs gma{o.b_attnb, o.b_gateb, kNH * kDH, kNH * kDH, kNH * kDH, B};
                disp("k_gate_mul_batch", false,
                     MTLSizeMake(kNH * kDH, B, 1), MTLSizeMake(128, 1, 1), &gma, sizeof(gma), curPack);
            }
            gemm_sg(L.wo, o.b_attnb, kNH * kDH, o.b_tA, kNE, kNH * kDH, kNE);
            conv_b((uint32_t)il, 2 * kKVD, kNE, o.b_tA, kNE, o.b_tB, kNE, L.postconv, 0u, 0u, kNE, kConvStatic);
            if (ip % 6 == 0) {
                if (st.nblk >= kBBlkMax) {
                    err = "metal: batch block stack overflow";
                } else {
                    if (bsmall) {
                        // Order preserved by separate dispatches: cur -> blocks
                        // must finish before tB -> cur overwrites cur.
                        xfer_b(true, o.b_cur, kNE, 0u, 0u,
                               o.b_blocks + (uint32_t)st.nblk * kNE, kBlkStride, kNE);
                        xfer_b(true, o.b_tB, kNE, 0u, 0u, o.b_cur, kNE, kNE);
                    } else {
                        for (uint32_t b = 0; b < B; b++) {
                            xfer2(o.b_cur + b * kNE, 0, o.b_blocks + b * kBlkStride + (uint32_t)st.nblk * kNE,
                                  kNE, 0, 0, 0, 0, 1);
                            xfer2(o.b_tB + b * kNE, 0, o.b_cur + b * kNE, kNE, 0, 0, 0, 0, 1);
                        }
                    }
                    st.nblk++;
                }
            } else {
                if (bsmall)
                    xfer_b(false, o.b_cur, kNE, o.b_tB, kNE, o.b_cur, kNE, kNE);
                else
                    for (uint32_t b = 0; b < B; b++)
                        xfer2(o.b_cur + b * kNE, o.b_tB + b * kNE, 0, kNE, 0, 0, 0, 0, 0);
            }
            bprof_flush(BP_WO);

            // --- FFN ---
            if (bsmall)
                do_mix_b(o.b_blocks, kBlkStride, st.nblk, o.b_cur, kNE, L.ffnq, 1, o.b_mixed, 3 * kNE);
            else
                for (uint32_t b = 0; b < B; b++)
                    do_mix(o.b_blocks + b * kBlkStride, st.nblk, o.b_cur + b * kNE,
                           L.ffnq, 1, o.b_mixed + b * (3 * kNE));
            bprof_flush(BP_MIX);
            if (lp) {
                if (bsmall) {
                    do_ctrl_b(o.b_mixed, 3 * kNE, 0u, o.b_msgs + 3 * kNE, 4 * kNE, 0u,
                              o.b_ctrlout, 3 * kNE, 0u,
                              lp->down + 3, lp->gain + 3, lp->forg + 3, 1);
                } else {
                    for (uint32_t b = 0; b < B; b++) {
                        uint32_t in1 = o.b_mixed + b * (3 * kNE);
                        uint32_t msg1 = o.b_msgs + b * (4 * kNE) + 3 * kNE;
                        uint32_t out1 = o.b_ctrlout + b * (3 * kNE);
                        do_ctrl(&in1, &msg1, &out1, lp->down + 3, lp->gain + 3, lp->forg + 3, 1);
                    }
                }
                bprof_flush(BP_CTRL);
            }
            rms_b(L.ffnnorm, lp ? o.b_ctrlout : o.b_mixed, 3 * kNE, kNE,
                  o.b_fin, kNE, kNE, M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, 1u, 1u, 0u, f_eps);

            // --- Router + Bucket ---
            {
                GemvF32BatchArgs rga{L.router, o.b_fin, o.b_probs, kNE, kNEXP, kNEXP, kNE, B, 0u};
                d_tg("k_gemv_f32_batch", kNEXP, B, 256, &rga, sizeof(rga));
                RouterBatchArgs rba{L.probsb, o.b_probs, o.b_sel, o.b_wgt, kNEXP, kTOPK, kTOPK, B, 0u, w_scale};
                d_tg("k_router_topk_batch", B, 1, 128, &rba, sizeof(rba));
                MoeBucketArgs mba{o.b_sel, o.b_cnt, o.b_off, o.b_list, kNEXP, B * kTOPK, 0u, 0u};
                d_tg("k_moe_bucket_build", 1, 1, 256, &mba, sizeof(mba));
            }
            bprof_flush(BP_ROUTER);

            // --- MoE Grouped (resident OR arena-resolved batch) ---
            bool used_per_tok_fallback = false;
            uint32_t wtab_off = 0u;
            uint64_t moe_wbase = L.experts;
            if (arena_mode && err.empty()) {
                [en endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    NSString * msg = cb.error.localizedDescription ?: @"unknown MTLCommandBufferError";
                    err = std::string("metal: batch arena sync status=") +
                          std::to_string((NSInteger)cb.status) + ": " + msg.UTF8String;
                } else {
                    float * S = (float *)scr.contents;
                    const uint32_t * cnts = (const uint32_t *)S + o.b_cnt;
                    std::vector<uint32_t> exps;
                    exps.reserve(kNEXP);
                    for (uint32_t e = 0; e < kNEXP; e++) if (cnts[e] > 0u) exps.push_back(e);
                    cb = [rt.queue() commandBuffer];
                    en = [cb computeCommandEncoder];
                    if (exps.size() <= arena->num_slots()) {
                        std::vector<uint32_t> slots(exps.size());
                        if (!arena->resolve_batch(s_phys[il], exps.data(), (int)exps.size(),
                                                  slots.data(), err)) {
                            fprintf(stderr, "m64metal: arena resolve_batch failed: %s\n", err.c_str());
                        } else {
                            uint32_t * wtab = (uint32_t *)S + o.b_wtab;
                            memset(wtab, 0, kNEXP * sizeof(uint32_t));
                            for (size_t i = 0; i < exps.size(); i++) wtab[exps[i]] = slots[i];
                            wtab_off = o.b_wtab;
                            moe_wbase = 0ull;
                            curPack = expertBuf;
                        }
                    } else {
                        // Tiny arena fallback: stream & execute per token cleanly.
                        used_per_tok_fallback = true;
                        for (uint32_t b = 0; b < B && err.empty(); b++) {
                            uint32_t sel8[kTOPK], slots8[kTOPK];
                            for (int i = 0; i < kTOPK; i++)
                                sel8[i] = ((const uint32_t *)S)[o.b_sel + b * kTOPK + i];
                            if (!arena->resolve_top8(s_phys[il], sel8, slots8, err)) break;
                            for (int i = 0; i < kTOPK; i++) ((uint32_t *)S)[o.b_wtab + i] = slots8[i];
                            // Note: synchronize per token in tiny-arena mode because shared
                            // o.b_wtab / o.xqE / o.gate2 are overwritten across iterations.
                            if (L.moe_q4k) {
                                // Milestone 4B FP32 pipeline over arena slots (slots are
                                // the direct blob indexes in o.b_wtab[0..7]).
                                const uint32_t msel = o.b_wtab;
                                MoeGateUpQ4KF32Args g4{0ul, (unsigned long)L.moe_off[1],
                                                       o.b_fin + b * kNE, o.gate2, o.ubuf, msel, 0u,
                                                       kNF, kNE, (unsigned int)exp_stride};
                                curPack = expertBuf;
                                if (gemv_m2) d_tg("k_gemv_gateup_batch8_q4k_f32in_m2", (kTOPK * kNF + 7) / 8, 1, 256, &g4, sizeof(g4));
                                else d_tg("k_gemv_gateup_batch8_q4k_f32in", (kTOPK * 2 * kNF + 7) / 8, 1, 256, &g4, sizeof(g4));
                                SituF32BatchArgs sf4{o.gate2, o.ubuf, o.down_tmp, kNF, kNF, kNF, kNF, kTOPK,
                                                     situ_beta, situ_lb, 1.0f / situ_beta, 1.0f / situ_lb};
                                disp("k_situ_f32_batch", false, MTLSizeMake(kNF, kTOPK, 1),
                                     MTLSizeMake(128, 1, 1), &sf4, sizeof(sf4), curPack);
                                MoeDownQ5F32Args md4{(unsigned long)L.moe_off[2], o.down_tmp,
                                                     o.b_acc + b * kNE, o.b_wgt + b * kTOPK, msel, 0u, 0u,
                                                     kNE, kNF, (unsigned int)exp_stride, 0u};
                                if (gemv_m2)
                                    d_tg(L.moe_dt[2] == DType::Q5_1 ? "k_gemv_down_reduce8_q5_1_f32in_m4"
                                                                    : "k_gemv_down_reduce8_q5_0_f32in_m4",
                                         (kNE + 31) / 32, 1, 256, &md4, sizeof(md4));
                                else
                                    d_tg(L.moe_dt[2] == DType::Q5_1 ? "k_gemv_down_reduce8_q5_1_f32in"
                                                                    : "k_gemv_down_reduce8_q5_0_f32in",
                                         (kNE + 7) / 8, 1, 256, &md4, sizeof(md4));
                                curPack = residentBuf;
                            } else {
                            QuantArgs qa{o.b_fin + b * kNE, o.xqE, o.dxE, kNE / 32u};
                            d_th("k_quant", kNE / 32u, 32, &qa, sizeof(qa));
                            GemvArgs gu{0ull, (unsigned long)kPartBytes,
                                        o.xqE, o.dxE, o.gate2, o.ubuf,
                                        kNF, 0, (uint32_t)exp_stride, o.b_wtab};
                            curPack = expertBuf;
                            if (gemv_m2) d_tg("k_gemv_gateup_batch8_m2", (kTOPK * kNF + 7) / 8, 1, 256, &gu, sizeof(gu));
                            else d_tg("k_gemv_gateup_batch8", (kTOPK * 2 * kNF + 7) / 8, 1, 256, &gu, sizeof(gu));
                            SituArgs sa{o.gate2, o.ubuf, o.hq, o.dh, kTOPK * (kNF / 32), situ_beta, situ_lb,
                                        1.0f / situ_beta, 1.0f / situ_lb};
                            d_th("k_situ_quant8", kTOPK * kNF, 128, &sa, sizeof(sa));
                            MoeDownArgs md{(unsigned long)(2 * kPartBytes),
                                           o.hq, o.dh, o.b_acc + b * kNE, o.b_wgt + b * kTOPK, 0, kNE,
                                           (uint32_t)exp_stride, o.b_wtab};
                            if (gemv_m2) d_tg("k_gemv_down_reduce8_m4", (kNE + 31) / 32, 1, 256, &md, sizeof(md));
                            else d_tg("k_gemv_down_reduce8", (kNE + 15) / 16, 1, 256, &md, sizeof(md));
                            curPack = residentBuf;
                            }
                            if (b + 1 < B) {
                                [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
                                if (cb.status != MTLCommandBufferStatusCompleted) { err = "metal: tiny arena fault"; break; }
                                cb = [rt.queue() commandBuffer];
                                en = [cb computeCommandEncoder];
                            }
                        }
                    }
                }
            }
            if (!used_per_tok_fallback && err.empty()) {
                // Milestone 4B: per-part grouped kernel picked from the layer's
                // part dtypes (Q8_0 layers dispatch the exact same kernels as before).
                auto moe_gk = [&](int part) {
                    const DType dt = L.moe_dt[part];
                    return dt == DType::Q4_K ? "k_moe_grouped_f32in_sg_q4k"
                         : dt == DType::Q5_0 ? "k_moe_grouped_f32in_sg_q5_0"
                         : dt == DType::Q5_1 ? "k_moe_grouped_f32in_sg_q5_1"
                                            : "k_moe_grouped_f32in_sg";
                };
                const uint64_t mst = arena_mode ? exp_stride : L.moe_stride;  // slot vs per-layer pack stride
                MoeGroupedF32SgArgs gup{moe_wbase, mst, (uint64_t)L.moe_off[0],
                                        o.b_fin, o.b_gate2, o.b_list, o.b_cnt, o.b_off,
                                        wtab_off, kNF, kNE, kNE, kNF, kNEXP, kTOPK, 0u, 0u};
                disp(moe_gk(0), true, MTLSizeMake((kNF + 31) / 32, kNEXP, 1),
                     MTLSizeMake(32, 4, 1), &gup, sizeof(gup), curPack);
                gup.part_off = (uint64_t)L.moe_off[1];
                gup.yoff = o.b_ubuf;
                disp(moe_gk(1), true, MTLSizeMake((kNF + 31) / 32, kNEXP, 1),
                     MTLSizeMake(32, 4, 1), &gup, sizeof(gup), curPack);
                SituF32BatchArgs sba{o.b_gate2, o.b_ubuf, o.b_situ,
                                     kNF, kNF, kNF, kNF, B * kTOPK,
                                     situ_beta, situ_lb, 1.0f / situ_beta, 1.0f / situ_lb};
                disp("k_situ_f32_batch", false, MTLSizeMake(kNF, B * kTOPK, 1),
                     MTLSizeMake(128, 1, 1), &sba, sizeof(sba), curPack);
                bprof_flush(BP_GATEUP);
                MoeGroupedF32SgArgs dn{moe_wbase, mst, (uint64_t)L.moe_off[2],
                                       o.b_situ, o.b_down_tmp, o.b_list, o.b_cnt, o.b_off,
                                       wtab_off, kNE, kNF, kNF, kNE, kNEXP, kTOPK, 1u, 0u};
                disp(moe_gk(2), true, MTLSizeMake((kNE + 31) / 32, kNEXP, 1),
                     MTLSizeMake(32, 4, 1), &dn, sizeof(dn), curPack);
                if (arena_mode) curPack = residentBuf;
                MoeReduceBatchArgs mra{o.b_down_tmp, o.b_wgt, o.b_acc,
                                       kNE, kTOPK, kNE, kNE, kTOPK, B, 0u};
                disp("k_moe_down_reduce_batch", false, MTLSizeMake(kNE, B, 1),
                     MTLSizeMake(128, 1, 1), &mra, sizeof(mra), curPack);
            }
            conv_b((uint32_t)il, 2 * kKVD + kNE, kNE, o.b_acc, kNE, o.b_tA, kNE,
                   L.ffnconv, 0u, 0u, kNE, kConvStatic);
            if (bsmall)
                xfer_b(false, o.b_cur, kNE, o.b_tA, kNE, o.b_cur, kNE, kNE);
            else
                for (uint32_t b = 0; b < B; b++)
                    xfer2(o.b_cur + b * kNE, o.b_tA + b * kNE, 0, kNE, 0, 0, 0, 0, 0);
            bprof_flush(BP_DOWN);
        };

        // --- Embedding + tok_norm ---
        {
            EmbBatchArgs ea{tok_embd, o.b_tok, o.b_emb, kNE, 72u, B, 0u, 0u};
            disp("k_emb_gather_batch", false, MTLSizeMake(72, B, 1), MTLSizeMake(32, 1, 1),
                 &ea, sizeof(ea), curPack);
            rms_b(tok_norm, o.b_emb, kNE, kNE, o.b_cur, kNE, kNE,
                  M64_NOFFU, M64_NOFFU, 0u, 0u, kNE, 1u, 1u, 0u, f_eps);
            bprof_flush(BP_OTHER);
        }
        // --- Prelude (0..5), Core loop (3 x 6..17), Coda (18..23) ---
        for (int ip = 0; ip < 6; ip++) batch_slot(ip, ip, nullptr);
        if (st.nblk != 1 && err.empty()) err = "metal: unexpected batch prelude block count";
        if (bsmall) {
            xfer_b(true, o.b_cur, kNE, 0u, 0u, o.b_prelude_out, kNE, kNE);
            xfer_b(true, o.b_blocks, kBlkStride, 0u, 0u, o.b_prelude_blk, kNE, kNE);
        } else {
            for (uint32_t b = 0; b < B; b++)
                xfer2(o.b_cur + b * kNE, 0, o.b_prelude_out + b * kNE, kNE,
                      o.b_blocks + b * kBlkStride, 0, o.b_prelude_blk + b * kNE, kNE, 1);
        }
        bprof_flush(BP_OTHER);
        int npre = st.nblk;
        for (int pass = 0; pass < 3; pass++) {
            int nbank = 0;
            if (pass > 0) {
                nbank = st.nblk - 2;
                if ((nbank < 0 || nbank > kBBlkMax) && err.empty()) err = "metal: batch bank stack bounds";
                if (bsmall) {
                    if (nbank > 0)
                        xfer_b(true, o.b_blocks + 2 * kNE, kBlkStride, 0u, 0u,
                               o.b_bank, kBlkStride, (uint32_t)nbank * kNE);
                    xfer_b(true, o.b_cur, kNE, 0u, 0u, o.b_bank_out, kNE, kNE);
                    xfer_b(true, o.b_prelude_out, kNE, 0u, 0u, o.b_cur, kNE, kNE);
                    xfer_b(true, o.b_prelude_blk, kNE, 0u, 0u, o.b_blocks, kBlkStride, kNE);
                } else {
                    for (uint32_t b = 0; b < B; b++) {
                        if (nbank > 0)
                            xfer2(o.b_blocks + b * kBlkStride + 2 * kNE, 0,
                                  o.b_bank + b * kBlkStride, (uint32_t)nbank * kNE, 0, 0, 0, 0, 1);
                        xfer2(o.b_cur + b * kNE, 0, o.b_bank_out + b * kNE, kNE, 0, 0, 0, 0, 1);
                        xfer2(o.b_prelude_out + b * kNE, 0, o.b_cur + b * kNE, kNE, 0, 0, 0, 0, 1);
                        xfer2(o.b_prelude_blk + b * kNE, 0, o.b_blocks + b * kBlkStride, kNE, 0, 0, 0, 0, 1);
                    }
                }
                st.nblk = npre;
                bprof_flush(BP_OTHER);
            }
            for (int ip = 6; ip < 18; ip++) {
                int il = pass == 0 ? ip : 24 + (pass - 1) * 12 + (ip - 6);
                if (pass > 0) {
                    const LL & lp = loop[s_loop[il]];
                    if (bsmall)
                        do_mix_b(o.b_bank, kBlkStride, nbank, o.b_bank_out, kNE,
                                 lp.msgq, 4, o.b_msgs, 4 * kNE);
                    else
                        for (uint32_t b = 0; b < B; b++)
                            do_mix(o.b_bank + b * kBlkStride, nbank, o.b_bank_out + b * kNE,
                                   lp.msgq, 4, o.b_msgs + b * (4 * kNE));
                    bprof_flush(BP_MIX);
                }
                batch_slot(il, ip, pass > 0 ? &loop[s_loop[il]] : nullptr);
            }
        }
        for (int ip = 18; ip < 24; ip++) batch_slot(ip, ip, nullptr);

        // --- Final LM head on LAST token in chunk only (when need_logits) ---
        if (lm_rows > 0u && err.empty()) {
            // Milestone 7B verify: per-row LM head for every batch row, using
            // the EXACT same kernel/arg sequence as the single-row tail below
            // (staged through o.mixed / o.xqE one row at a time) so row b's
            // logits are bitwise-equal to the plain path's logits for token b.
            for (uint32_t b = 0; b < B; b++) {
                do_mix(o.b_blocks + b * kBlkStride, st.nblk, o.b_cur + b * kNE,
                       out_q, 1, o.b_mixed + b * (3 * kNE));
                const uint32_t lg = o.b_lgx + b * 201024u;
                RmsQuantArgs RQA{};
                RQA.k = kNE; RQA.rows = 1; RQA.hasw = 1; RQA.eps = f_eps;
                RQA.xr[0] = o.b_mixed + b * (3 * kNE); RQA.wr[0] = out_norm; RQA.yr[0] = M64_NOFFU;
                RQA.q[0] = o.xqE; RQA.d[0] = o.dxE;
                d_tg("k_rms_quant", 1, 1, 256, &RQA, sizeof(RQA));
                GemvArgs GA{outw, 0, o.xqE, o.dxE, lg, 0, (uint32_t)vocab_rows, 0, 0, 0};
                if (gemv_m2) d_tg("k_gemv_q8_k2304_m2", ((uint32_t)vocab_rows + 15) / 16, 1, 256, &GA, sizeof(GA));
                else         d_tg("k_gemv_q8_k2304",    ((uint32_t)vocab_rows + 7)  / 8,  1, 256, &GA, sizeof(GA));
                SoftcapArgs sca{lg, (uint32_t)vocab_rows, cap};
                d_th("k_softcap", (uint32_t)vocab_rows, 256, &sca, sizeof(sca));
                ArgmaxArgs ara{lg, o.b_amax + b, (uint32_t)vocab_eff};
                d_tg("k_argmax", 1, 1, 1024, &ara, sizeof(ara));
            }
            bprof_flush(BP_LMHEAD);
        } else if (need_logits && err.empty()) {
            const uint32_t blast = B - 1u;
            do_mix(o.b_blocks + blast * kBlkStride, st.nblk, o.b_cur + blast * kNE, out_q, 1, o.mixed);
            RmsQuantArgs RQA{};
            RQA.k = kNE; RQA.rows = 1; RQA.hasw = 1; RQA.eps = f_eps;
            RQA.xr[0] = o.mixed; RQA.wr[0] = out_norm; RQA.yr[0] = M64_NOFFU;
            RQA.q[0] = o.xqE; RQA.d[0] = o.dxE;
            d_tg("k_rms_quant", 1, 1, 256, &RQA, sizeof(RQA));
            GemvArgs GA{outw, 0, o.xqE, o.dxE, o.logits, 0, (uint32_t)vocab_rows, 0, 0, 0};
            if (gemv_m2) d_tg("k_gemv_q8_k2304_m2", ((uint32_t)vocab_rows + 15) / 16, 1, 256, &GA, sizeof(GA));
            else         d_tg("k_gemv_q8_k2304",    ((uint32_t)vocab_rows + 7)  / 8,  1, 256, &GA, sizeof(GA));
            SoftcapArgs sca{o.logits, (uint32_t)vocab_rows, cap};
            d_th("k_softcap", (uint32_t)vocab_rows, 256, &sca, sizeof(sca));
            ArgmaxArgs ara{o.logits, o.amax, (uint32_t)vocab_eff};
            d_tg("k_argmax", 1, 1, 1024, &ara, sizeof(ara));
        }

        [en endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            NSString * msg = cb.error.localizedDescription ?: @"unknown MTLCommandBufferError";
            err = std::string("metal: batch command buffer status=") +
                  std::to_string((NSInteger)cb.status) + ": " + msg.UTF8String;
            fprintf(stderr, "m64metal: GPU fault: %s\n", err.c_str());
        } else if (prof_batch) {
            bprof_ms[BP_LMHEAD] += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            bprof_disp[BP_LMHEAD] += bcur_disp;
            const char * bpnames[BP_NCAT] = {
                "other", "mix", "ctrl", "qkv_sg", "conv_clip_attn",
                "wo_post", "router", "moe_gateup", "moe_down", "lm_head"
            };
            double tot_ms = 0.0; uint32_t tot_d = 0;
            for (int i = 0; i < BP_NCAT; i++) { tot_ms += bprof_ms[i]; tot_d += bprof_disp[i]; }
            fprintf(stderr, "M64_PROFILE_BATCH pos=%lld B=%u total=%.3f ms (%.2f tok/s) dispatches=%u\n",
                    (long long)pos, B, tot_ms, tot_ms > 0 ? (B * 1000.0 / tot_ms) : 0.0, tot_d);
            for (int i = 0; i < BP_NCAT; i++) {
                fprintf(stderr, "  %-16s %7.3f ms (%5.1f%%)  disp=%5u\n",
                        bpnames[i], bprof_ms[i], tot_ms > 0 ? (100.0 * bprof_ms[i] / tot_ms) : 0.0, bprof_disp[i]);
            }
        }
        if (err.empty()) {
            if (need_logits) {
                const float * sbase = (const float *)scr.contents;
                r.argmax = (int32_t)*(const uint32_t *)(sbase + o.amax);
            }
            pos += (int64_t)B;
        }
        return r;
    }

    // Milestone 7B: one batched GPU pass over up to kBSpec draft/verify rows;
    // returns the per-row argmax (amax[i] = model argmax after toks[0..i]) and
    // the last row's logits so the next round's anchor needs no extra pass.
    bool gpu_verify(const std::vector<int32_t> &toks, std::vector<int32_t> &amax,
                    std::vector<float> &last_logits, std::string &err) {
        amax.clear(); last_logits.clear();
        if (toks.empty()) return true;
        if (toks.size() > (size_t)kBSpec) { err = "metal: verify batch exceeds kBSpec"; return false; }
        gpu_prefill_batch(toks.data(), (uint32_t)toks.size(), err, false, 1u);
        if (!err.empty()) return false;
        const float * S = (const float *)scr.contents;
        for (uint32_t b = 0; b < (uint32_t)toks.size(); b++)
            amax.push_back((int32_t)*(const uint32_t *)(S + o.b_amax + b));
        const float * lrow = S + o.b_lgx + ((uint32_t)toks.size() - 1u) * 201024u;
        last_logits.assign(lrow, lrow + vocab_eff);
        return true;
    }

    // Milestone 7C: state-only replay (Branch fast path). lm_rows=0 and
    // need_logits=false skip EVERY LM head row (rms_quant + vocab GEMV +
    // softcap + argmax); pos/KV/conv rs/SWA still advance exactly like the
    // verify pass, so the engine lands bitwise where a verify2 would put it.
    bool gpu_advance_no_logits(const std::vector<int32_t> &toks, std::string &err) {
        if (toks.empty()) return true;
        if (toks.size() == 1u) {
            (void)gpu_step(toks[0], err, /*need_logits=*/false);
            return err.empty();
        }
        if (toks.size() > (size_t)kBSpec) { err = "metal: advance batch exceeds kBSpec"; return false; }
        (void)gpu_prefill_batch(toks.data(), (uint32_t)toks.size(), err,
                                /*need_logits=*/false, /*lm_rows=*/0u);
        return err.empty();
    }

};

InklingForwardMetal::InklingForwardMetal() = default;
InklingForwardMetal::~InklingForwardMetal() = default;

bool InklingForwardMetal::init(PackReader & p, int c, std::string & e,
                               double arena_gb, const std::string & io_mode,
                               bool kv_q8) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    return impl_->init(p, c, e, arena_gb, io_mode, kv_q8);
}
StepResult InklingForwardMetal::step(int32_t t) {
    std::string err;
    StepResult r = impl_->gpu_step(t, err);
    if (!err.empty()) fprintf(stderr, "m64metal: %s\n", err.c_str());
    return r;
}
bool InklingForwardMetal::verify_batch(const std::vector<int32_t> &toks,
                                       std::vector<int32_t> &amax,
                                       std::vector<float> &last_logits,
                                       std::string &err) {
    if (!impl_) { err = "metal: engine not initialized"; return false; }
    return impl_->gpu_verify(toks, amax, last_logits, err);
}
bool InklingForwardMetal::advance_no_logits(const std::vector<int32_t> &toks,
                                            std::string &err) {
    if (!impl_) { err = "metal: engine not initialized"; return false; }
    return impl_->gpu_advance_no_logits(toks, err);
}
StepResult InklingForwardMetal::prefill(const std::vector<int32_t> & p) {
    StepResult r{};
    if (p.empty()) return r;
    const bool allow_batch = (impl_->prefill_chunk > 1) &&
                             !getenv("M64_TRACE_SLOT") && !getenv("M64_NANPROBE");
    const size_t n = p.size();
    for (size_t i = 0; i < n;) {
        size_t rem = n - i;
        size_t chunk = allow_batch ? std::min<size_t>((size_t)impl_->prefill_chunk, rem) : 1u;
        bool is_last = (i + chunk == n);
        std::string err;
        if (chunk <= 1u) {
            r = impl_->gpu_step(p[i], err, /*need_logits=*/is_last);
        } else {
            r = impl_->gpu_prefill_batch(p.data() + i, (uint32_t)chunk, err, /*need_logits=*/is_last);
        }
        if (!err.empty()) { fprintf(stderr, "m64metal: %s\n", err.c_str()); return r; }
        i += chunk;
    }
    return r;
}
void InklingForwardMetal::reset() { if (impl_) impl_->reset(); }
int64_t InklingForwardMetal::current_pos() const { return impl_ ? impl_->pos : 0; }
bool InklingForwardMetal::save_prefix_state(PrefixState & out,
                                            const std::vector<int32_t> & tokens) const {
    return impl_ ? impl_->save_prefix_state(out, tokens) : false;
}
bool InklingForwardMetal::restore_prefix_state(const PrefixState & in) {
    return impl_ ? impl_->restore_prefix_state(in) : false;
}
size_t InklingForwardMetal::full_kv_elems(int64_t n_pos) const {
    return impl_ ? impl_->full_kv_elems(n_pos) : 0;
}
void InklingForwardMetal::capture_full_kv(PrefixState & st) const {
    if (impl_) impl_->capture_full_kv(st);
}
uint64_t InklingForwardMetal::pack_fingerprint() const {
    return impl_ ? impl_->pack_fp : 0;
}
bool InklingForwardMetal::save_disk_prefix(const std::string & path,
                                           const std::vector<int32_t> & tokens,
                                           std::string & err) const {
    return impl_ ? impl_->save_disk_prefix(path, tokens, err) : false;
}
bool InklingForwardMetal::snapshot_disk_prefix(const std::string & path,
                                               const std::vector<int32_t> & tokens,
                                               DiskPrefixSnapshot & out,
                                               std::string & err) const {
    return impl_ ? impl_->snapshot_disk_prefix(path, tokens, out, err) : false;
}
bool InklingForwardMetal::load_disk_prefix(const std::string & path,
                                           PrefixState & out_ram_state,
                                           std::string & err) {
    return impl_ ? impl_->load_disk_prefix(path, out_ram_state, err) : false;
}
int64_t InklingForwardMetal::vocab_effective() const { return impl_ ? impl_->vocab_eff : 0; }
int64_t InklingForwardMetal::vocab_logit_rows() const { return impl_ ? impl_->vocab_rows : 0; }
size_t InklingForwardMetal::resident_bytes() const {
    if (impl_ && impl_->arena_mode && impl_->arena)
        return impl_->arena_resident_bytes + impl_->arena->total_bytes() +
               (size_t)impl_->o.total * 4 + impl_->kv_bytes;
    return impl_ ? ((size_t)impl_->o.total * 4 + impl_->kv_bytes) : 0;  // scratch + KV (weights mmap'd)
}
std::string InklingForwardMetal::arena_report() const {
    if (!impl_ || !impl_->arena_mode || !impl_->arena) return {};
    const ArenaTelemetry & t = impl_->arena->telemetry();
    double total_sel = (double)(t.hits + t.misses);
    double hit_pct = total_sel > 0 ? 100.0 * t.hits / total_sel : 0.0;
    double ssd_ms = t.ssd_ns / 1e6;
    double ssd_gbs = t.ssd_ns > 0 ? (t.ssd_read_bytes / 1e9) / (t.ssd_ns / 1e9) : 0.0;
    char buf[512];
    snprintf(buf, sizeof buf,
             "arena: budget=%.2f GiB resident=%.2f GiB arena=%.2f GiB slots=%zu "
             "hits=%llu misses=%llu (hit=%.1f%%) ssd=%.3f GB in %.1f ms (%.2f GB/s)",
             impl_->arena_gb,
             impl_->arena_resident_bytes / 1073741824.0,
             impl_->arena->total_bytes() / 1073741824.0,
             impl_->arena->num_slots(),
             (unsigned long long)t.hits, (unsigned long long)t.misses, hit_pct,
             t.ssd_read_bytes / 1e9, ssd_ms, ssd_gbs);
    return std::string(buf);
}

}  // namespace m64
#endif  // __OBJC__
