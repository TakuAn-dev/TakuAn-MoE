// test_prefill_kernels.mm — Milestone-3A/3B-1 batched-prefill kernel parity.
// Covers m64_prefill.metal: k_gemm_q8_batch, k_conv_batch, k_conv_slot
// bitwise chains, k_kv_write_f16_batch, k_attn_head_f16_batch (full +
// SWA ring wrap + tau log-scale + out-of-extent relbias sentinel) and
// k_moe_grouped_gemm_q8 (is_down 0/1, 0-token / 1-token / >8-token experts).
// Milestone-3B-1 adds: k_gemm_q8_f32in_sg (simdgroup_matrix, FP32 in,
// FP64-oracle gate + bitwise add_to_out/tail guards + cross-check vs the
// 3A quant+gemv chain) and the 7 batched helpers k_emb_gather_batch,
// k_rms_batch, k_clip_batch, k_gate_mul_batch, k_situ_quant_batch,
// k_moe_down_reduce_batch, k_router_topk_batch (bitwise vs sequential
// m64_p2a dispatches / host fp32 mirrors).
// Oracle policy: Q8 dots are int32-exact per 32-block so FP64 accumulation
// is the reference (gates: rel < 1e-4, cos > 0.99999 for GEMM-class;
// rel < 1e-3, cos > 0.99999 for attention); short-conv chains are BITWISE.
// All buffers < 40 MB; never touches real packs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "metal_runtime.hpp"
#include "../../engine/kernels/m64_p2a_args.h"

// C++-side layout guards (mirror the static_asserts in m64_prefill.metal).
static_assert(sizeof(GemmBatchArgs) == 48, "GemmBatchArgs");
static_assert(sizeof(ConvBatchArgs) == 64, "ConvBatchArgs");
static_assert(sizeof(KVWriteBatchF16Args) == 56, "KVWriteBatchF16Args");
static_assert(sizeof(AttnBatchF16Args) == 96, "AttnBatchF16Args");
static_assert(sizeof(MoeGroupedGemmArgs) == 88, "MoeGroupedGemmArgs");
static_assert(sizeof(GemmF32SgArgs) == 48, "GemmF32SgArgs");
static_assert(sizeof(EmbBatchArgs) == 40, "EmbBatchArgs");
static_assert(sizeof(RmsBatchArgs) == 80, "RmsBatchArgs");
static_assert(sizeof(ClipBatchArgs) == 40, "ClipBatchArgs");
static_assert(sizeof(GateMulBatchArgs) == 24, "GateMulBatchArgs");
static_assert(sizeof(SituBatchArgs) == 56, "SituBatchArgs");
static_assert(sizeof(MoeReduceBatchArgs) == 40, "MoeReduceBatchArgs");
static_assert(sizeof(RouterBatchArgs) == 48, "RouterBatchArgs");
static_assert(sizeof(GemvF32BatchArgs) == 40, "GemvF32BatchArgs");
static_assert(sizeof(MoeBucketArgs) == 32, "MoeBucketArgs");
static_assert(sizeof(MoeGroupedF32SgArgs) == 80, "MoeGroupedF32SgArgs");
static_assert(sizeof(SituF32BatchArgs) == 48, "SituF32BatchArgs");
static_assert(sizeof(AttnF16SplitArgs) == 80, "AttnF16SplitArgs");

using namespace m64;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static constexpr uint32_t kNH = 24, kDH = 128, kKVD = 1024, kNREL = 16;

struct Rng {
    std::mt19937 g{0x9E3779B9u};
    uint32_t u32() { return g(); }
    float f(float a = -2.f, float b = 2.f) { return std::uniform_real_distribution<float>(a, b)(g); }
};

static inline float f16le(const uint8_t * p) { return (float)std::bit_cast<__fp16>((uint16_t)(p[0] | (p[1] << 8))); }
static inline float r16(float x) { return (float)(__fp16)x; }
static inline uint16_t h16bits(float x) { const __fp16 h = (const __fp16)x; uint16_t u; memcpy(&u, &h, 2); return u; }
static inline double h16d(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (double)h; }
static inline int sb(uint8_t b) { return (int)(int8_t)b; }

static double max_rel(const float * a, const float * b, size_t n) {
    double worst = 0;
    for (size_t i = 0; i < n; i++) {
        double d = std::fabs((double)a[i] - b[i]);
        worst = std::max(worst, d / std::max(1e-6, std::fabs((double)a[i])));
    }
    return worst;
}
static double cosine(const float * a, const float * b, size_t n) {
    double sa = 0, sb2 = 0, sab = 0;
    for (size_t i = 0; i < n; i++) { sa += (double)a[i] * a[i]; sb2 += (double)b[i] * b[i]; sab += (double)a[i] * b[i]; }
    return sab / (std::sqrt(sa) * std::sqrt(sb2) + 1e-300);
}

// Gate policy: primary gates are rel < 1e-4 (GEMM) / 1e-3 (attention) with
// cos > 0.99999 plus a noise-floor escape matching test_p2a_kernels' ok():
// GEMM mad/max|ref| < 1e-7 (~ulp*k f32 accumulation), attention mad < 1e-5
// (softmax outputs are O(0.02), so an absolute floor is the right metric).
// GATE_SG: k_gemm_q8_f32in_sg vs the FP64 oracle — inputs become half for
// the tensor-core product, so per-element ~5e-4 relative noise accumulates.
// GATE_XQ: same kernel vs the Q8_0-activation k_gemv chain — the delta is
// dominated by Q8_0 ACTIVATION rounding, so rel is loose and cos is the
// real anchor.
enum GateMode { GATE_GEMM = 0, GATE_ATTN = 1, GATE_SG = 2, GATE_XQ = 3 };
static void gate(const char * name, const float * ref, const float * got, size_t n, int mode) {
    double rel = max_rel(ref, got, n), cosv = cosine(ref, got, n), mad = 0, maxabs = 0;
    for (size_t i = 0; i < n; i++) {
        mad = std::max(mad, std::fabs((double)ref[i] - got[i]));
        maxabs = std::max(maxabs, std::fabs((double)ref[i]));
    }
    const double relscale = mad / std::max(1e-30, maxabs);
    const bool pass =
        mode == GATE_GEMM ? ((rel < 1e-4 || relscale < 1e-7) && cosv > 0.99999) :
        mode == GATE_ATTN ? ((rel < 1e-3 || mad < 1e-5) && cosv > 0.99999) :
        mode == GATE_SG   ? ((rel < 1e-2 || relscale < 2e-3) && cosv > 0.9995) :
                            ((rel < 5e-2 || relscale < 1e-2) && cosv > 0.999);
    CHECK(pass, "%s rel=%.3g relscale=%.3g cos=%.8f mad=%.3g", name, rel, relscale, cosv, mad);
    printf("%-26s rel=%.3g relscale=%.3g cos=%.8f mad=%.3g %s\n",
           name, rel, relscale, cosv, mad, pass ? "OK" : "FAIL");
}

// Host Q8_0 quantizer, exactly mirroring k_quant (f16-rounded stored d,
// codes quantized with the pre-rounding d, rint ties-to-even, clamp +-127).
// k must be a multiple of 32; codes holds k bytes, d holds k/32 floats.
static void quantize(const float * x, int k, uint8_t * codes, float * d) {
    const int nb = k / 32;
    for (int b = 0; b < nb; b++) {
        float amax = 0;
        for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[(size_t)b * 32 + i]));
        const float dr = amax / 127.0f;
        const float id = dr != 0.0f ? 1.0f / dr : 0.0f;
        d[(size_t)b] = r16(dr);
        for (int i = 0; i < 32; i++) {
            int q = (int)std::rintf(x[(size_t)b * 32 + i] * id);
            q = q > 127 ? 127 : (q < -127 ? -127 : q);
            codes[(size_t)b * 32 + i] = (uint8_t)(int8_t)q;
        }
    }
}
static void quantize(const std::vector<float> & x, int k,
                     std::vector<uint8_t> & codes, std::vector<float> & d) {
    codes.resize((size_t)k); d.resize((size_t)(k / 32));
    quantize(x.data(), k, codes.data(), d.data());
}

// FP64 oracle over a Q8_0 row (nb = k/32 blocks of 34 B; int-exact block dot).
static double gemm_oracle(const uint8_t * wrow, const int8_t * xc, const float * dx, int k) {
    const int nb = k / 32;
    double a = 0;
    for (int b = 0; b < nb; b++) {
        const uint8_t * wb = wrow + (size_t)b * 34;
        int s = 0;
        for (int i = 0; i < 32; i++) s += sb(wb[2 + i]) * xc[(size_t)b * 32 + i];
        a += (double)s * ((double)dx[b] * (double)f16le(wb));
    }
    return a;
}

static void fill_q8(uint8_t * at, size_t rows, int nb, Rng & R) {
    // row stride is nb*34 B; every 34 B block stores its f16 scale big-endian
    // at bytes 0..1, so mask byte 1 (high byte) for ALL blocks incl. odd tail.
    const size_t rstride = (size_t)nb * 34;
    for (size_t i = 0; i < rows * rstride; i++) at[i] = (uint8_t)R.u32();
    for (size_t r = 0; r < rows; r++) for (int b = 0; b < nb; b++)
        at[r * rstride + (size_t)b * 34 + 1] &= 0x3B;   // finite f16 scales
}

// ---- Milestone 4B host writers (q4_K / q5_0 / Q5_1), bit layouts per
// engine/src/dequant.hpp — mirrors the verified writers in test_p2a_kernels.
static inline void w16le(uint8_t * at, uint16_t hv) { at[0] = (uint8_t)hv; at[1] = (uint8_t)(hv >> 8); }
static inline void w32le(uint8_t * at, uint32_t v) {
    at[0] = (uint8_t)v; at[1] = (uint8_t)(v >> 8); at[2] = (uint8_t)(v >> 16); at[3] = (uint8_t)(v >> 24);
}

// q4_K: 256 values -> one 144 B block. scales/min clamped <=15 so they fit the
// 4-bit slots the word-packed decoders read (same restriction as test_p2a_kernels).
static void fill_q4k(uint8_t * at, const float * v, float dmval, Rng & R) {
    float bmn[8], bmx[8];
    for (int j = 0; j < 8; j++) {
        bmn[j] = v[j * 32]; bmx[j] = v[j * 32];
        for (int i = 1; i < 32; i++) { float x = v[j * 32 + i]; if (x < bmn[j]) bmn[j] = x; if (x > bmx[j]) bmx[j] = x; }
    }
    float dmax = 0.0f;
    for (int j = 0; j < 8; j++) dmax = std::max(dmax, (bmx[j] - bmn[j]) / 15.0f);
    const float dval = dmax > 0.0f ? dmax : 1.0f;
    w16le(at, h16bits(dval)); w16le(at + 2, h16bits(dmval));
    int sc8[8], mn8[8];
    for (int j = 0; j < 8; j++) {
        double s = dmval != 0.0f ? std::rint((double)(bmn[j] / dmval)) : 0.0;
        sc8[j] = (int)std::max(0.0, std::min(15.0, std::rint((double)((bmx[j] - bmn[j]) / 15.0f / dval))));
        mn8[j] = (int)std::max(0.0, std::min(15.0, s));   // <16 keeps the q[j-4]-quirk decoders bit-agreeing
    }
    uint8_t scb[12];
    memset(scb, 0, 12);
    for (int j = 0; j < 4; j++) { scb[j] = (uint8_t)sc8[j]; scb[j + 4] = (uint8_t)mn8[j]; }
    for (int j = 4; j < 8; j++) scb[j + 4] = (uint8_t)((sc8[j] & 0xF) | ((mn8[j] & 0xF) << 4));
    uint8_t code[256];
    for (int j = 0; j < 8; j++) {
        float step = dval * (float)sc8[j];
        for (int i = 0; i < 32; i++) {
            double c = step != 0.0f ? std::rint((double)((v[j * 32 + i] - dmval * (float)mn8[j]) / step)) : 0.0;
            code[j * 32 + i] = (uint8_t)std::max(0.0, std::min(15.0, c));
        }
    }
    for (int j = 0; j < 12; j++) at[4 + (size_t)j] = scb[j];
    for (int c = 0; c < 4; c++) for (int l = 0; l < 32; l++)
        at[16 + (size_t)c * 32 + l] = (uint8_t)(code[c * 64 + l] | (code[c * 64 + 32 + l] << 4));
}

// q5_0 (22 B) / q5_1 (24 B): 32 values -> one block; qh bit j = MSB of elem j,
// qh bit j+16 = MSB of elem 16+j (dequant.hpp:110-149).
static void fill_q5(uint8_t * at, const float * v, bool q51) {
    float mn = v[0], mx = v[0];
    for (int i = 1; i < 32; i++) { if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i]; }
    float d = (mx - mn) / 31.0f, m = q51 ? mn : 0.0f;
    if (d == 0.0f) d = 1.0f;
    w16le(at, h16bits(d));
    uint32_t qh = 0;
    uint8_t qs[16], code[32];
    for (int i = 0; i < 32; i++) {
        double c = std::rint((double)((v[i] - m) / d));
        code[i] = (uint8_t)std::max(0.0, std::min(31.0, c));
    }
    for (int i = 0; i < 16; i++) {
        qs[i] = (uint8_t)((code[i] & 0xF) | ((code[16 + i] & 0xF) << 4));
        qh |= (uint32_t)((code[i] >> 4) & 1) << (uint32_t)i;
        qh |= (uint32_t)((code[16 + i] >> 4) & 1) << (uint32_t)(16 + i);
    }
    if (q51) w16le(at + 2, h16bits(m));
    w32le(at + (q51 ? 4 : 2), qh);
    for (int i = 0; i < 16; i++) at[(q51 ? 8 : 6) + (size_t)i] = qs[i];
}

// FP64 oracle for the sg kernels: dequant bit-exactly per dequant.hpp, model the
// kernel's fp32 dequant + f16 rounding of weights (threadgroup staging) and the
// FP32->f16 activation cast, then accumulate in FP64. The only residual delta is
// the tensor-core fp32 accumulation order (gate: 1024-ulp fp32 bound, test 23 policy).
struct DotS { double a, sab; };
static void sm4k(const uint8_t * q, int j, double & D, double & M) {
    if (j < 4) { D = q[j] & 63; M = q[j + 4] & 63; }
    else { D = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); M = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
// fmt: 1 = Q4_K (row = (nb/8) * 144 B), 2 = Q5_0 (nb * 22 B), 3 = Q5_1 (nb * 24 B).
static DotS dot_sg_q(int fmt, const uint8_t * wrow, uint32_t nb, const float * x) {
    DotS r{0, 0};
    double & a = r.a, & sa = r.sab;
    for (uint32_t blk = 0; blk < nb; blk++) {
        const float * xb = x + (size_t)blk * 32;
        if (fmt == 1) {
            const uint8_t * bp = wrow + (size_t)(blk >> 3) * 144;
            const int is = (int)(blk & 7u);
            double D, M; sm4k(bp + 4, is, D, M);
            const float vd = f16le(bp) * (float)D;      // exact: f16 x small int
            const float vm = f16le(bp + 2) * (float)M;  // (<= 15 bits mantissa)
            const uint8_t * qb = bp + 16 + (is >> 1) * 32;
            for (int i = 0; i < 32; i++) {
                const uint8_t n = (is & 1) ? (qb[i] >> 4) : (qb[i] & 0xFu);
                const double w = (double)r16(vd * (float)n - vm);
                const double xh = (double)r16(xb[i]);
                a += w * xh; sa += std::fabs(w) * std::fabs(xh);
            }
        } else {
            const uint32_t bs = fmt == 2 ? 22u : 24u;
            const uint8_t * bp = wrow + (size_t)blk * bs;
            const float d = f16le(bp);
            const float m = fmt == 3 ? f16le(bp + 2) : 0.0f;
            const uint32_t qh = (uint32_t)bp[fmt == 3 ? 4 : 2] | ((uint32_t)bp[fmt == 3 ? 5 : 3] << 8) |
                                ((uint32_t)bp[fmt == 3 ? 6 : 4] << 16) | ((uint32_t)bp[fmt == 3 ? 7 : 5] << 24);
            for (int j = 0; j < 16; j++) {
                const uint8_t qv = bp[(fmt == 3 ? 8 : 6) + j];
                const int x0 = (qv & 0xF) | (int)(((qh >> j) << 4) & 0x10u);
                const int x1 = (qv >> 4) | (int)((qh >> (j + 12)) & 0x10u);
                const double w0 = (double)r16(fmt == 2 ? d * (float)(x0 - 16) : d * (float)x0 + m);
                const double w1 = (double)r16(fmt == 2 ? d * (float)(x1 - 16) : d * (float)x1 + m);
                const double xh0 = (double)r16(xb[j]), xh1 = (double)r16(xb[16 + j]);
                a += w0 * xh0 + w1 * xh1;
                sa += std::fabs(w0) * std::fabs(xh0) + std::fabs(w1) * std::fabs(xh1);
            }
        }
    }
    return r;
}

int main(int argc, char ** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--bench")) do_bench = true;
    @autoreleasepool {
        Rng R;
        MetalRuntime rt;
        std::string err;
        if (!rt.init("engine/kernels", "build/m64_prefill_test.arch", err)) {
            printf("FAIL: MetalRuntime init: %s\n", err.c_str()); return 1;
        }

        // ---------------- host mirrors + GPU buffers ----------------
        const uint32_t SCR_F = 8192u << 10;           // 32 MB of floats
        const size_t PK_B = 64u * 1024 * 1024;         // 64 MB fake pack
        std::vector<float> scr(SCR_F, 0.f);
        std::vector<uint8_t> pk(PK_B, 0);
        float * pkf = (float *)pk.data();              // f32 view of PACK payload
        id<MTLBuffer> bScr = [rt.device() newBufferWithLength:SCR_F * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bPk = [rt.device() newBufferWithLength:PK_B options:MTLResourceStorageModeShared];
        CHECK(bScr != nil && bPk != nil, "scr/pk buffer alloc");
        uint32_t cur = 16;
        auto seg = [&](uint32_t n) { uint32_t r = cur; CHECK((size_t)cur + n + 15u < SCR_F, "SCR overflow"); cur += n + 15u; return r; };
        size_t pcur = 0;
        auto psegs = [&](size_t n) { pcur = (pcur + 63) & ~(size_t)63; size_t r = pcur; CHECK(pcur + n <= PK_B, "PACK overflow"); pcur += n; return r; };

        auto run = [&](const char * fn, bool tg, uint tx, uint ty, uint nthx, uint nthy,
                       const void * a, size_t sz, id<MTLBuffer> bKv = nil) {
            memcpy(bScr.contents, scr.data(), SCR_F * 4);
            memcpy(bPk.contents, pk.data(), PK_B);
            id<MTLComputePipelineState> p = rt.pipeline(fn);
            CHECK(p != nil, "pipeline %s missing", fn);
            if (!p) return;
            auto cmd = [rt.queue() commandBuffer];
            auto en = [cmd computeCommandEncoder];
            [en setComputePipelineState:p];
            [en setBuffer:bScr offset:0 atIndex:0];
            [en setBytes:a length:sz atIndex:1];
            [en setBuffer:bPk offset:0 atIndex:2];
            if (bKv) [en setBuffer:bKv offset:0 atIndex:3];
            if (tg) [en dispatchThreadgroups:MTLSizeMake(tx, ty, 1)
                       threadsPerThreadgroup:MTLSizeMake(nthx, nthy, 1)];
            else    [en dispatchThreads:MTLSizeMake(tx, ty, 1)
                       threadsPerThreadgroup:MTLSizeMake(nthx, nthy, 1)];
            [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
            CHECK(cmd.status == MTLCommandBufferStatusCompleted, "%s cmdbuf status=%ld err=%s", fn,
                  (long)cmd.status, cmd.error.localizedDescription.UTF8String);
            memcpy(scr.data(), bScr.contents, SCR_F * 4);
        };

        // ================= 1. k_gemm_q8_batch =================
        {
            struct GG { uint32_t k, nout, B; } ggs[] = {
                {2304, 64, 1}, {2304, 64, 7}, {2304, 64, 16}, {2304, 64, 32},
                {1152, 50, 13}, {96, 9, 5},   // k=96: odd nb=3 -> unaligned rows + 34B tail
            };
            const uint32_t BMAX = 32;
            const uint32_t sxq = ((kDH * 2304) / 4) / 4;   // 0 placeholder, per-k below
            (void)sxq;
            const uint32_t KMAX = 2304;
            const uint32_t qstr = (KMAX / 4 + 15u) & ~15u;         // 576 floats / token
            const uint32_t dstr = (KMAX / 32 + 3u) & ~3u;          // 72 floats / token
            const uint32_t ystr = 4096;
            uint32_t sXq = seg(BMAX * qstr + 64), sDx = seg(BMAX * dstr + 64), sY = seg(BMAX * ystr + 64);
            for (auto & gg : ggs) {
                const uint32_t k = gg.k, nout = gg.nout, B = gg.B, nb = k / 32;
                size_t w = psegs((size_t)nout * nb * 34);
                fill_q8(pk.data() + w, nout, (int)nb, R);
                // quantize B activations into SCR (codes as bytes, dx as floats)
                for (uint32_t t = 0; t < B; t++) {
                    std::vector<float> x(k);
                    for (auto & v : x) v = R.f(-6, 6);
                    std::vector<uint8_t> codes; std::vector<float> d;
                    quantize(x, (int)k, codes, d);
                    memcpy(scr.data() + sXq + t * qstr, codes.data(), k);
                    for (uint32_t b = 0; b < nb; b++) scr[sDx + t * dstr + b] = d[b];
                }
                GemmBatchArgs A{w, sXq, sDx, sY, nout, k, B, qstr, dstr, ystr, 0u};
                for (size_t i = 0; i < (size_t)B * ystr; i++) scr[sY + i] = 12345.0f;  // poison stale rows
                run("k_gemm_q8_batch", true, (nout + 3) / 4, (B + 7) / 8, 32, 4, &A, sizeof(A));
                std::vector<float> ref((size_t)B * nout), got((size_t)B * nout);
                for (uint32_t t = 0; t < B; t++) {
                    const int8_t * xc = (const int8_t *)(scr.data() + sXq + t * qstr);
                    const float * dx = scr.data() + sDx + t * dstr;
                    for (uint32_t r = 0; r < nout; r++) {
                        ref[(size_t)t * nout + r] =
                            (float)gemm_oracle(pk.data() + w + (size_t)r * nb * 34, xc, dx, (int)k);
                        got[(size_t)t * nout + r] = scr[sY + t * ystr + r];
                    }
                }
                char nm[64]; snprintf(nm, sizeof(nm), "gemm_batch k=%u B=%u", k, B);
                if (getenv("PK_DEBUG"))
                    for (uint32_t t = 0; t < B; t++) for (uint32_t r = 0; r < nout; r++) {
                        double a = ref[(size_t)t * nout + r], b = got[(size_t)t * nout + r];
                        if (std::fabs(a - b) > 1e-3 * (1 + std::fabs(a)))
                            printf("  dbg t=%u r=%u ref=%.6f got=%.6f\n", t, r, a, b);
                    }
                gate(nm, ref.data(), got.data(), (size_t)B * nout, GATE_GEMM);
            }
        }

        // ================= 2/3. k_conv_batch vs k_conv_slot chains (BITWISE) =================
        for (uint32_t hasdyn : {0u, 1u}) {
            const uint32_t C = hasdyn ? 256 : 512, group = 128, B = 16;
            const float cscale = 0.25f;
            size_t wpk = psegs((size_t)C * 4 * 4);
            for (uint32_t i = 0; i < C * 4; i++) ((float *)(pk.data() + wpk))[i] = R.f(-1, 1);
            uint32_t x0 = seg(B * C), y0 = seg(B * C), rs0 = seg(C * 3), dyn = seg(4 * 4);
            for (uint32_t i = 0; i < B * C; i++) scr[x0 + i] = R.f(-3, 3);
            for (uint32_t i = 0; i < C * 3; i++) scr[rs0 + i] = R.f(-2, 2);
            for (uint32_t i = 0; i < 16; i++) scr[dyn + i] = R.f(-1, 1);
            const std::vector<float> rs_init(scr.begin() + rs0, scr.begin() + rs0 + C * 3);
            std::vector<float> y_seq((size_t)B * C);
            // --- sequential k_conv_slot dispatches (x stride = C, y stride = C) ---
            if (hasdyn) {
                // per-token dyn rows are only expressible via dyn_stride; the
                // sequential chain uses one shared dyn row (dyn_stride = 0).
            }
            std::copy(rs_init.begin(), rs_init.end(), scr.begin() + rs0);
            for (uint32_t b = 0; b < B; b++) {
                ConvArgs ca{rs0, x0 + b * C, y0 + b * C, wpk, dyn, C, group, cscale, hasdyn};
                run("k_conv_slot", false, C, 1, 128, 1, &ca, sizeof(ca));
                memcpy(y_seq.data() + (size_t)b * C, scr.data() + y0 + b * C, C * 4);
            }
            std::vector<float> rs_seq(scr.begin() + rs0, scr.begin() + rs0 + C * 3);
            // --- batched ---
            std::copy(rs_init.begin(), rs_init.end(), scr.begin() + rs0);
            for (uint32_t i = 0; i < B * C; i++) scr[y0 + i] = 12345.0f;   // poison
            ConvBatchArgs cb{wpk, x0, y0, rs0, dyn, C, B, C, C, 0u, group, hasdyn, 0u, cscale, 0u};
            run("k_conv_batch", false, C, 1, 128, 1, &cb, sizeof(cb));
            int bad_y = 0, bad_rs = 0;
            for (uint32_t i = 0; i < B * C; i++) if (scr[y0 + i] != y_seq[i]) bad_y++;
            for (uint32_t i = 0; i < C * 3; i++) if (scr[rs0 + i] != rs_seq[i]) bad_rs++;
            CHECK(bad_y == 0 && bad_rs == 0, "k_conv_batch dyn=%u: y=%d rs=%d bitwise mismatches",
                  hasdyn, bad_y, bad_rs);
            printf("conv_batch dyn=%u          bitwise y=%d rs=%d mismatches (%u x %u) %s\n",
                   hasdyn, bad_y, bad_rs, B, C, (bad_y || bad_rs) ? "FAIL" : "OK");
            // absolute-value sanity vs host scalar chain (fp-contract off)
            std::vector<float> ref((size_t)B * C);
            {
                const float * w4 = (const float *)(pk.data() + wpk);
                for (uint32_t c = 0; c < C; c++) {
                    float s0 = rs_init[c * 3], s1 = rs_init[c * 3 + 1], s2 = rs_init[c * 3 + 2];
                    for (uint32_t b = 0; b < B; b++) {
                        const float xin = scr[x0 + b * C + c];
                        float outv = s0 * w4[c * 4] + s1 * w4[c * 4 + 1] + s2 * w4[c * 4 + 2] + xin * w4[c * 4 + 3];
                        if (hasdyn) {
                            const float * dnr = scr.data() + dyn + (c / group) * 4u;
                            outv += dnr[0] * xin + dnr[1] * s2 + dnr[2] * s1 + dnr[3] * s0;
                        }
                        ref[(size_t)b * C + c] = (xin + outv) * cscale;
                        s0 = s1; s1 = s2; s2 = xin;
                    }
                }
            }
            char nm[48]; snprintf(nm, sizeof(nm), "conv_batch host dyn=%u", hasdyn);
            gate(nm, ref.data(), scr.data() + y0, (size_t)B * C, GATE_GEMM);
        }

        // ================= 4/5. k_kv_write_f16_batch + k_attn_head_f16_batch =================
        {
            const uint32_t B = 16;
            const float scale = 1.0f / sqrtf((float)kDH);
            // relbias [1200][16]; entries with d >= 1024 are sentinels that MUST
            // never be read (dist >= M64_EXT guard).
            size_t rb = psegs(1200u * kNREL * sizeof(float));
            for (uint32_t d = 0; d < 1200u * kNREL; d++)
                ((float *)(pk.data() + rb))[d] = (d < 1024u * kNREL) ? R.f(-0.5f, 0.5f) : 100.0f;

            uint32_t sQb = seg(B * kNH * kDH), sRb = seg(B * kNH * kNREL),
                     sSb = seg(B * kNH + 4), sOb = seg(B * kNH * kDH),
                     sKv = seg(B * 2u * kKVD);
            for (uint32_t i = 0; i < B * kNH * kDH; i++) scr[sQb + i] = R.f(-1.5f, 1.5f);
            for (uint32_t i = 0; i < B * kNH * kNREL; i++) scr[sRb + i] = R.f(-0.5f, 0.5f);
            for (uint32_t i = 0; i < B * kNH; i++) scr[sSb + i] = R.f(-1.5f, 1.5f);
            for (uint32_t i = 0; i < B * 2u * kKVD; i++) scr[sKv + i] = R.f(-2.0f, 2.0f);

            // generic oracle: KV contents come from host half-mirror (K then V)
            auto ref_attn = [&](uint32_t pos0, uint32_t b_cnt, uint32_t is_swa,
                                uint32_t n_swa, uint32_t ring_cap,
                                const std::vector<uint16_t> & mk, const std::vector<uint16_t> & mv,
                                std::vector<float> & ref) {
                for (uint32_t b = 0; b < b_cnt; b++) {
                    const uint32_t pos = pos0 + b;
                    const float tau = is_swa ? 1.0f
                        : (float)(std::log(std::max(1.0, ((double)pos + 1.0) / 8192.0)) * 0.25 + 1.0);
                    const uint32_t j_start = (is_swa && pos > n_swa) ? (pos - n_swa) : 0u;
                    for (uint32_t h = 0; h < kNH; h++) {
                        const float * q = scr.data() + sQb + (size_t)b * kNH * kDH + h * kDH;
                        const float * ro = scr.data() + sRb + (size_t)b * kNH * kNREL + h * kNREL;
                        const double snk = scr[sSb + b * kNH + h];
                        const uint32_t kvh = h / 3u;
                        double oh[kDH] = {0}, M = -INFINITY, S = 0.0;
                        for (uint32_t j = j_start; j <= pos; j++) {
                            const uint32_t slot = is_swa ? (j % ring_cap) : j;
                            const uint16_t * kj = mk.data() + (size_t)slot * kKVD + kvh * kDH;
                            const uint16_t * vj = mv.data() + (size_t)slot * kKVD + kvh * kDH;
                            double d = 0.0;
                            for (int i = 0; i < (int)kDH; i++)
                                d += (double)r16(q[i] * tau) * h16d(kj[i]);
                            uint32_t dist = pos - j;
                            double bv = 0.0;
                            if (dist < 1024u) {
                                const float * br = ((const float *)(pk.data() + rb)) + dist * 16u;
                                for (int i = 0; i < 16; i++) bv += (double)br[i] * ((double)ro[i] * (double)tau);
                            }
                            double s = d * (double)scale + (double)r16((float)(bv - snk));
                            double ms = 1.0, vs = 1.0;
                            if (s > M) {
                                ms = std::exp(M - s);
                                for (int i = 0; i < (int)kDH; i++) oh[i] *= ms;
                                M = s;
                            } else {
                                vs = std::exp(s - M);
                            }
                            for (int i = 0; i < (int)kDH; i++)
                                oh[i] += vs * h16d(vj[i]);
                            S = S * ms + vs;
                        }
                        {
                            double ms = 1.0, vs = 1.0;
                            if (0.0 > M) { ms = std::exp(M); for (int i = 0; i < (int)kDH; i++) oh[i] *= ms; M = 0.0; }
                            else { vs = std::exp(0.0 - M); }
                            S = S * ms + vs;
                        }
                        const double inv = 1.0 / S;
                        for (int i = 0; i < (int)kDH; i++)
                            ref[((size_t)b * kNH + h) * kDH + i] = (float)(oh[i] * inv);
                    }
                }
            };

            // batch KV write + bitwise verify against host r16 expectation
            auto kv_write = [&](id<MTLBuffer> bKv, uint32_t pos0, uint32_t is_swa,
                                uint32_t ring_cap, std::vector<uint16_t> & mk,
                                std::vector<uint16_t> & mv, size_t kv_total_half) {
                KVWriteBatchF16Args kw{};
                kw.kvk_base = 0; kw.kvv_base = kv_total_half / 2;
                kw.koff = sKv; kw.voff = sKv + kKVD;
                kw.k_stride = 2 * kKVD; kw.v_stride = 2 * kKVD;
                kw.n = kKVD; kw.B = B; kw.pos0 = pos0; kw.is_swa = is_swa;
                kw.ring_cap = ring_cap; kw.pad0 = 0;
                memcpy(bScr.contents, scr.data(), SCR_F * 4);
                id<MTLComputePipelineState> p = rt.pipeline("k_kv_write_f16_batch");
                CHECK(p != nil, "pipeline k_kv_write_f16_batch missing");
                if (!p) return;
                auto cmd = [rt.queue() commandBuffer];
                auto en = [cmd computeCommandEncoder];
                [en setComputePipelineState:p];
                [en setBuffer:bScr offset:0 atIndex:0];
                [en setBytes:&kw length:sizeof(kw) atIndex:1];
                [en setBuffer:bKv offset:0 atIndex:2];
                [en dispatchThreads:MTLSizeMake(kKVD, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                CHECK(cmd.status == MTLCommandBufferStatusCompleted, "k_kv_write_f16_batch status=%ld", (long)cmd.status);
                int bad = 0;
                const uint16_t * dev = (const uint16_t *)bKv.contents;
                const size_t vbase = kv_total_half / 2;
                for (uint32_t b = 0; b < B; b++) {
                    const uint32_t pos = pos0 + b;
                    const uint32_t slot = is_swa ? (pos % ring_cap) : pos;
                    // full-row bitwise verify + host mirror sync (oracle reads mirror)
                    for (uint32_t c = 0; c < kKVD; c++) {
                        const uint16_t ek = h16bits(scr[sKv + b * 2 * kKVD + c]);
                        const uint16_t ev = h16bits(scr[sKv + b * 2 * kKVD + kKVD + c]);
                        if (dev[slot * kKVD + c] != ek || dev[vbase + slot * kKVD + c] != ev) bad++;
                        mk[slot * kKVD + c] = ek;
                        mv[slot * kKVD + c] = ev;
                    }
                }
                CHECK(bad == 0, "k_kv_write_f16_batch %d bitwise mismatches", bad);
                printf("kv_write_f16_batch p0=%u  bitwise %d mismatches (full rows) %s\n",
                       pos0, bad, bad ? "FAIL" : "OK");
            };

            auto attn_run = [&](id<MTLBuffer> bKv, const AttnBatchF16Args & A) {
                run("k_attn_head_f16_batch", true, kNH, A.B, 32, 1, &A, sizeof(A), bKv);
            };

            // ---- 4. FULL attention: slots 0..8184 pre-populated, batch pos0=8185, B=16 ----
            {
                const uint32_t NPRE = 8185, POSMAX = 8201;
                const size_t half_per = (size_t)POSMAX * kKVD;
                std::vector<uint16_t> mk(half_per), mv(half_per);
                for (size_t i = 0; i < half_per; i++) {
                    mk[i] = h16bits(R.f(-2, 2));
                    mv[i] = h16bits(R.f(-2, 2));
                }
                id<MTLBuffer> bKv = [rt.device() newBufferWithLength:half_per * 2 * sizeof(uint16_t)
                                                            options:MTLResourceStorageModeShared];
                CHECK(bKv != nil, "alloc full-attention KV buffer");
                if (!bKv) return 1;
                // upload pre-populated KV (K region then V region)
                {
                    uint16_t * dev = (uint16_t *)bKv.contents;
                    memcpy(dev, mk.data(), half_per * 2);
                    memcpy(dev + half_per, mv.data(), half_per * 2);
                }
                kv_write(bKv, NPRE, 0u, 1u, mk, mv, half_per * 2);
                attn_run(bKv, [&] { AttnBatchF16Args A{};
                    A.relbiaspack = rb; A.kvkoff = 0; A.kvvoff = half_per;
                    A.qoff = sQb; A.rout = sRb; A.sinkoff = sSb; A.outoff = sOb;
                    A.q_stride = kNH * kDH; A.rout_stride = kNH * kNREL;
                    A.sink_stride = kNH; A.out_stride = kNH * kDH;
                    A.pos0 = NPRE; A.B = B; A.is_swa = 0; A.n_swa = 1024;
                    A.ring_cap = POSMAX; A.pad0 = 0;
                    A.scale = scale; A.log_scale_floor = 8192.0f; A.log_scale_alpha = 0.25f; A.pad1 = 0;
                    return A; }());
                std::vector<float> ref((size_t)B * kNH * kDH);
                ref_attn(NPRE, B, 0u, 1024u, POSMAX, mk, mv, ref);
                gate("attn_f16_batch full", ref.data(), scr.data() + sOb, ref.size(), GATE_ATTN);
                bKv = nil;   // release 33 MB before SWA block
            }

            // ---- 5. SWA ring wrap: slots 0..1024 pre-populated, batch pos0=1020, B=16 ----
            {
                const uint32_t RING = 1025, NPRE = 1020;
                const size_t half_per = (size_t)RING * kKVD;
                std::vector<uint16_t> mk(half_per), mv(half_per);
                for (size_t i = 0; i < half_per; i++) {
                    mk[i] = h16bits(R.f(-2, 2));
                    mv[i] = h16bits(R.f(-2, 2));
                }
                id<MTLBuffer> bKv = [rt.device() newBufferWithLength:half_per * 2 * sizeof(uint16_t)
                                                            options:MTLResourceStorageModeShared];
                CHECK(bKv != nil, "alloc SWA KV buffer");
                if (!bKv) return 1;
                {
                    uint16_t * dev = (uint16_t *)bKv.contents;
                    memcpy(dev, mk.data(), half_per * 2);
                    memcpy(dev + half_per, mv.data(), half_per * 2);
                }
                kv_write(bKv, NPRE, 1u, RING, mk, mv, half_per * 2);   // pos 1020..1035 wrap
                attn_run(bKv, [&] { AttnBatchF16Args A{};
                    A.relbiaspack = rb; A.kvkoff = 0; A.kvvoff = half_per;
                    A.qoff = sQb; A.rout = sRb; A.sinkoff = sSb; A.outoff = sOb;
                    A.q_stride = kNH * kDH; A.rout_stride = kNH * kNREL;
                    A.sink_stride = kNH; A.out_stride = kNH * kDH;
                    A.pos0 = NPRE; A.B = B; A.is_swa = 1; A.n_swa = 1024;
                    A.ring_cap = RING; A.pad0 = 0;
                    A.scale = scale; A.log_scale_floor = 8192.0f; A.log_scale_alpha = 0.25f; A.pad1 = 0;
                    return A; }());
                std::vector<float> ref((size_t)B * kNH * kDH);
                ref_attn(NPRE, B, 1u, 1024u, RING, mk, mv, ref);
                gate("attn_f16_batch swa", ref.data(), scr.data() + sOb, ref.size(), GATE_ATTN);
                bKv = nil;
            }
        }

        // ================= 5b. k_attn_f16_flash_prefill parity (Milestone 4C)
        // Flash-2 tiled kernel vs the scalar batch kernel across B / pos0,
        // SWA ring wrap (ring_cap 1025 crossing at j=1025), relbias-extent
        // boundary (dist 1023 vs 1024, sentinel 100.0 beyond), causal +
        // SWA-window masking and OOB query tails (B % 32). P is f16-rounded
        // for the tensor-core PV product, so the batch->flash gate is
        // GATE_SG (half-input accumulation policy); selected small configs
        // additionally gate the FP64 host oracle (also GATE_SG: the flash PV
        // mma sees f16-rounded P, so fp64-oracle parity is tolerance-based).
        {
            const float scale = 1.0f / sqrtf((float)kDH);
            size_t rbF = psegs(1200u * kNREL * sizeof(float));
            for (uint32_t d = 0; d < 1200u * kNREL; d++)
                ((float *)(pk.data() + rbF))[d] = (d < 1024u * kNREL) ? R.f(-0.5f, 0.5f) : 100.0f;

            const uint32_t BMAXF = 128;
            uint32_t sQf = seg(BMAXF * kNH * kDH), sRf = seg(BMAXF * kNH * kNREL),
                     sSf = seg(BMAXF * kNH + 4), sOf = seg(BMAXF * kNH * kDH);
            for (uint32_t i = 0; i < BMAXF * kNH * kDH; i++) scr[sQf + i] = R.f(-1.5f, 1.5f);
            for (uint32_t i = 0; i < BMAXF * kNH * kNREL; i++) scr[sRf + i] = R.f(-0.5f, 0.5f);
            for (uint32_t i = 0; i < BMAXF * kNH; i++) scr[sSf + i] = R.f(-1.5f, 1.5f);

            auto ref_attn_f = [&](uint32_t pos0, uint32_t b_cnt, uint32_t is_swa,
                                  uint32_t n_swa, uint32_t ring_cap,
                                  const std::vector<uint16_t> & mk, const std::vector<uint16_t> & mv,
                                  std::vector<float> & ref) {
                for (uint32_t b = 0; b < b_cnt; b++) {
                    const uint32_t pos = pos0 + b;
                    const float tau = is_swa ? 1.0f
                        : (float)(std::log(std::max(1.0, ((double)pos + 1.0) / 8192.0)) * 0.25 + 1.0);
                    const uint32_t j_start = (is_swa && pos > n_swa) ? (pos - n_swa) : 0u;
                    for (uint32_t h = 0; h < kNH; h++) {
                        const float * q = scr.data() + sQf + (size_t)b * kNH * kDH + h * kDH;
                        const float * ro = scr.data() + sRf + (size_t)b * kNH * kNREL + h * kNREL;
                        const double snk = scr[sSf + b * kNH + h];
                        const uint32_t kvh2 = h / 3u;
                        double oh[kDH] = {0}, M = -INFINITY, S = 0.0;
                        for (uint32_t j = j_start; j <= pos; j++) {
                            const uint32_t slot = is_swa ? (j % ring_cap) : j;
                            const uint16_t * kj = mk.data() + (size_t)slot * kKVD + kvh2 * kDH;
                            const uint16_t * vj = mv.data() + (size_t)slot * kKVD + kvh2 * kDH;
                            double d = 0.0;
                            for (int i = 0; i < (int)kDH; i++)
                                d += (double)r16(q[i] * tau) * h16d(kj[i]);
                            uint32_t dist = pos - j;
                            double bv = 0.0;
                            if (dist < 1024u) {
                                const float * br = ((const float *)(pk.data() + rbF)) + dist * 16u;
                                for (int i = 0; i < 16; i++) bv += (double)br[i] * ((double)ro[i] * (double)tau);
                            }
                            double s = d * (double)scale + (double)r16((float)(bv - snk));
                            double ms = 1.0, vs = 1.0;
                            if (s > M) {
                                ms = std::exp(M - s);
                                for (int i = 0; i < (int)kDH; i++) oh[i] *= ms;
                                M = s;
                            } else {
                                vs = std::exp(s - M);
                            }
                            for (int i = 0; i < (int)kDH; i++)
                                oh[i] += vs * h16d(vj[i]);
                            S = S * ms + vs;
                        }
                        {
                            double ms = 1.0, vs = 1.0;
                            if (0.0 > M) { ms = std::exp(M); for (int i = 0; i < (int)kDH; i++) oh[i] *= ms; M = 0.0; }
                            else { vs = std::exp(0.0 - M); }
                            S = S * ms + vs;
                        }
                        const double inv = 1.0 / S;
                        for (int i = 0; i < (int)kDH; i++)
                            ref[((size_t)b * kNH + h) * kDH + i] = (float)(oh[i] * inv);
                    }
                }
            };

            auto attn_args_f = [&](uint32_t pos0, uint32_t Bb, uint32_t is_swa, uint32_t n_swa,
                                   uint32_t ring_cap, uint64_t kvv_off) {
                AttnBatchF16Args A{};
                A.relbiaspack = rbF; A.kvkoff = 0; A.kvvoff = kvv_off;
                A.qoff = sQf; A.rout = sRf; A.sinkoff = sSf; A.outoff = sOf;
                A.q_stride = kNH * kDH; A.rout_stride = kNH * kNREL;
                A.sink_stride = kNH; A.out_stride = kNH * kDH;
                A.pos0 = pos0; A.B = Bb; A.is_swa = is_swa; A.n_swa = n_swa;
                A.ring_cap = ring_cap; A.pad0 = 0;
                A.scale = scale; A.log_scale_floor = 8192.0f; A.log_scale_alpha = 0.25f; A.pad1 = 0;
                return A;
            };

            struct FCfg { const char * nm; uint32_t pos0, B, oracle; };
            // full attention: relbias-extent boundary at p=1020/B=37 spans
            // dist 1023/1024; p=20000/B=128 exercises the tile fast path + tail
            const FCfg full_cfgs[] = {
                {"full b1 p0",       0u,     1u, 1u},
                {"full b32 p0",      0u,    32u, 1u},
                {"full b7 p7",       7u,     7u, 0u},
                {"full b8 p512",     512u,   8u, 1u},
                {"full b37 p1020",   1020u, 37u, 0u},
                {"full b128 p20000", 20000u,128u, 0u},
            };
            // SWA ring_cap 1025: batch p1020..1051 wraps j=1025 -> slot 0;
            // window mask j < pos-1024; p20000 wraps the ring repeatedly.
            const FCfg swa_cfgs[] = {
                {"swa b8 p0",        0u,     8u, 1u},
                {"swa b32 p1020",    1020u, 32u, 1u},
                {"swa b37 p2040",    2040u, 37u, 0u},
                {"swa b128 p20000",  20000u,128u, 0u},
            };

            const uint32_t FPMAX = 20040u;
            const size_t fhalf = (size_t)FPMAX * kKVD;
            std::vector<uint16_t> fmk(fhalf), fmv(fhalf);
            for (size_t i = 0; i < fhalf; i++) { fmk[i] = h16bits(R.f(-2, 2)); fmv[i] = h16bits(R.f(-2, 2)); }
            id<MTLBuffer> bKvF = [rt.device() newBufferWithLength:fhalf * 2 * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
            CHECK(bKvF != nil, "flash full KV alloc");
            if (bKvF) {
                uint16_t * dev = (uint16_t *)bKvF.contents;
                memcpy(dev, fmk.data(), fhalf * 2);
                memcpy(dev + fhalf, fmv.data(), fhalf * 2);
            }
            const uint32_t SWR = 1025u;
            const size_t shalf = (size_t)SWR * kKVD;
            std::vector<uint16_t> smk(shalf), smv(shalf);
            for (size_t i = 0; i < shalf; i++) { smk[i] = h16bits(R.f(-2, 2)); smv[i] = h16bits(R.f(-2, 2)); }
            id<MTLBuffer> bKvS = [rt.device() newBufferWithLength:shalf * 2 * sizeof(uint16_t)
                                                          options:MTLResourceStorageModeShared];
            CHECK(bKvS != nil, "flash SWA KV alloc");
            if (bKvS) {
                uint16_t * dev = (uint16_t *)bKvS.contents;
                memcpy(dev, smk.data(), shalf * 2);
                memcpy(dev + shalf, smv.data(), shalf * 2);
            }
            CHECK(bKvF && bKvS, "flash parity KV buffers");

            auto flash_case = [&](const FCfg & c, uint32_t is_swa, uint32_t ring, size_t halfs,
                                  id<MTLBuffer> bKv, const std::vector<uint16_t> & mk,
                                  const std::vector<uint16_t> & mv) {
                if (!bKv) return;
                const auto A = attn_args_f(c.pos0, c.B, is_swa, 1024u, ring, halfs);
                const size_t n = (size_t)c.B * kNH * kDH;
                std::vector<float> refb(n);
                std::fill(scr.begin() + sOf, scr.begin() + sOf + n, -9999.0f);
                run("k_attn_head_f16_batch", true, kNH, c.B, 32, 1, &A, sizeof(A), bKv);
                memcpy(refb.data(), scr.data() + sOf, n * 4);
                std::fill(scr.begin() + sOf, scr.begin() + sOf + n, -9999.0f);
                run("k_attn_f16_flash_prefill", true, kNH, (c.B + 31u) / 32u, 32, 4, &A, sizeof(A), bKv);
                char nm1[48];
                snprintf(nm1, sizeof(nm1), "flash vs batch %s", c.nm);
                gate(nm1, refb.data(), scr.data() + sOf, n, GATE_SG);
                if (c.oracle) {
                    std::vector<float> refh(n);
                    ref_attn_f(c.pos0, c.B, is_swa, 1024u, ring, mk, mv, refh);
                    char nm2[48];
                    snprintf(nm2, sizeof(nm2), "flash vs fp64 %s", c.nm);
                    // GATE_SG, not GATE_ATTN: the flash PV product runs on
                    // tensor cores with f16-rounded P (mma operands MUST be
                    // half), so its fp64-oracle error is set by the p->half
                    // rounding (~1e-3 rel), not by accumulation order.
                    gate(nm2, refh.data(), scr.data() + sOf, n, GATE_SG);
                }
            };
            for (const auto & c : full_cfgs)
                flash_case(c, 0u, FPMAX, fhalf, bKvF, fmk, fmv);
            for (const auto & c : swa_cfgs)
                flash_case(c, 1u, SWR, shalf, bKvS, smk, smv);
            bKvF = nil; bKvS = nil;
        }

        // ================= 6/7. k_moe_grouped_gemm_q8 =================
        {
            const uint32_t B = 16, TOPK = 4, NEXP = 16, NPAIR = B * TOPK;
            const uint32_t qstr = 32, dstr = 8;    // k=128: 32 floats codes, 4 floats dx
            uint32_t sXq = seg(B * qstr + 16), sDx = seg(B * dstr + 16);
            for (uint32_t t = 0; t < B; t++) {
                const uint32_t k = (t == 0) ? 128 : 0;   // placeholder, quantized below
                (void)k;
            }
            // quantize 16 tokens at k=128 and k=64 into two regions
            const uint32_t qstr64 = 16, dstr64 = 4;
            uint32_t sXq64 = seg(B * qstr64 + 16), sDx64 = seg(B * dstr64 + 16);
            std::vector<std::vector<float>> dx128(B), dx64(B);
            uint32_t sList = seg(NPAIR + 16), sCount = seg(NEXP + 4), sOff = seg(NEXP + 4);
            uint32_t sY = seg(NPAIR * 128 + 64);
            for (uint32_t t = 0; t < B; t++) {
                std::vector<float> x128(128), x64(64);
                for (auto & v : x128) v = R.f(-6, 6);
                for (auto & v : x64) v = R.f(-6, 6);
                std::vector<uint8_t> c128, c64;
                quantize(x128, 128, c128, dx128[t]);
                quantize(x64, 64, c64, dx64[t]);
                memcpy(scr.data() + sXq + t * qstr, c128.data(), 128);
                memcpy(scr.data() + sXq64 + t * qstr64, c64.data(), 64);
                for (uint32_t b = 0; b < 4; b++) scr[sDx + t * dstr + b] = dx128[t][b];
                for (uint32_t b = 0; b < 2; b++) scr[sDx64 + t * dstr64 + b] = dx64[t][b];
            }

            // ---- 6. is_down=0 (gate/up): 64 pair outputs, k=128, nout=64 ----
            {
                const uint32_t counts[] = {0, 1, 13, 4, 2, 9, 16, 3, 4, 4, 0, 2, 0, 1, 3, 2};
                uint32_t sum = 0; for (uint32_t e = 0; e < NEXP; e++) sum += counts[e];
                CHECK(sum == NPAIR, "grouped counts sum=%u != %u", sum, NPAIR);
                std::vector<uint32_t> ids(NPAIR);
                for (uint32_t i = 0; i < NPAIR; i++) ids[i] = i;
                std::shuffle(ids.begin(), ids.end(), R.g);
                std::vector<uint32_t> owner(NPAIR, 0xFFFFFFFFu);
                uint32_t run_off = 0;
                for (uint32_t e = 0; e < NEXP; e++) {
                    ((uint32_t *)scr.data())[sCount + e] = counts[e];
                    ((uint32_t *)scr.data())[sOff + e] = run_off;
                    for (uint32_t i = 0; i < counts[e]; i++) {
                        ((uint32_t *)scr.data())[sList + run_off + i] = ids[run_off + i];
                        owner[ids[run_off + i]] = e;
                    }
                    run_off += counts[e];
                }
                const uint32_t nout = 64, k = 128, nb = 4;
                const uint64_t part = 4096;   // exercise part_off
                const uint64_t exp_stride = (uint64_t)nout * nb * 34;
                size_t wbase = psegs((size_t)part + (size_t)NEXP * exp_stride);
                for (uint32_t e = 0; e < NEXP; e++)
                    fill_q8(pk.data() + wbase + part + (size_t)e * exp_stride, nout, (int)nb, R);
                MoeGroupedGemmArgs A{wbase, exp_stride, part, sXq, sDx, sY, sList,
                                     sCount, sOff, nout, k, qstr, dstr, nout, NEXP, TOPK, 0u, 0u, 0u};
                CHECK(sY + NPAIR * nout < SCR_F, "grouped y overflow");
                for (uint32_t i = 0; i < NPAIR * nout; i++) scr[sY + i] = -9999.0f;   // poison
                run("k_moe_grouped_gemm_q8", true, (nout + 3) / 4, NEXP, 32, 4, &A, sizeof(A));
                std::vector<float> ref((size_t)NPAIR * nout);
                for (uint32_t p = 0; p < NPAIR; p++) {
                    const uint32_t e = owner[p], tok = p / TOPK;
                    const int8_t * xc = (const int8_t *)(scr.data() + sXq + (size_t)tok * qstr);
                    const uint8_t * wrow = pk.data() + wbase + part + (size_t)e * exp_stride;
                    for (uint32_t r = 0; r < nout; r++)
                        ref[(size_t)p * nout + r] = (float)gemm_oracle(wrow + (size_t)r * nb * 34,
                                                                        xc, dx128[tok].data(), (int)k);
                }
                gate("moe_grouped is_down=0", ref.data(), scr.data() + sY, ref.size(), GATE_GEMM);
            }

            // ---- 7. is_down=1: unique token ids, nout=128, k=64 ----
            {
                const uint32_t counts[] = {2, 0, 1, 1, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                uint32_t tot = 0; for (uint32_t e = 0; e < NEXP; e++) tot += counts[e];
                CHECK(tot == B, "down counts sum=%u != %u", tot, B);
                std::vector<uint32_t> owner(B, 0xFFFFFFFFu);
                uint32_t run_off = 0, nxt = 0;
                for (uint32_t e = 0; e < NEXP; e++) {
                    ((uint32_t *)scr.data())[sCount + e] = counts[e];
                    ((uint32_t *)scr.data())[sOff + e] = run_off;
                    for (uint32_t i = 0; i < counts[e]; i++) {
                        ((uint32_t *)scr.data())[sList + run_off + i] = nxt;
                        owner[nxt] = e; nxt++;
                    }
                    run_off += counts[e];
                }
                const uint32_t nout = 128, k = 64, nb = 2;
                const uint64_t part = 0;
                const uint64_t exp_stride = (uint64_t)nout * nb * 34;
                size_t wbase = psegs((size_t)NEXP * exp_stride);
                for (uint32_t e = 0; e < NEXP; e++)
                    fill_q8(pk.data() + wbase + (size_t)e * exp_stride, nout, (int)nb, R);
                MoeGroupedGemmArgs A{wbase, exp_stride, part, sXq64, sDx64, sY, sList,
                                     sCount, sOff, nout, k, qstr64, dstr64, nout, NEXP, TOPK, 1u, 0u, 0u};
                CHECK(sY + B * nout < SCR_F, "grouped down y overflow");
                for (uint32_t i = 0; i < B * nout; i++) scr[sY + i] = -9999.0f;
                run("k_moe_grouped_gemm_q8", true, (nout + 3) / 4, NEXP, 32, 4, &A, sizeof(A));
                std::vector<float> ref((size_t)B * nout);
                for (uint32_t t = 0; t < B; t++) {
                    const uint32_t e = owner[t];
                    const int8_t * xc = (const int8_t *)(scr.data() + sXq64 + (size_t)t * qstr64);
                    const uint8_t * wrow = pk.data() + wbase + (size_t)e * exp_stride;
                    for (uint32_t r = 0; r < nout; r++)
                        ref[(size_t)t * nout + r] = (float)gemm_oracle(wrow + (size_t)r * nb * 34,
                                                                        xc, dx64[t].data(), (int)k);
                }
                gate("moe_grouped is_down=1", ref.data(), scr.data() + sY, ref.size(), GATE_GEMM);
            }
        }

        // ================= 8. k_gemm_q8_f32in_sg (simdgroup_matrix, FP32 in) =================
        {
            struct SGCase { uint32_t k, nout, B; } cases[] = {
                {96, 37, 13},        // nb=3 odd (unaligned Q8_0 rows) + both tile tails
                {1152, 128, 13},
                {2304, 256, 1},
                {2304, 256, 7},
                {2304, 256, 16},
                {2304, 256, 32},
            };
            const uint32_t KMAX = 2304, NMAX = 256, BMAX = 32;
            uint32_t sXf = seg(BMAX * KMAX + 64), sYg = seg(BMAX * NMAX + 64);
            for (auto & cs : cases) {
                const uint32_t k = cs.k, nout = cs.nout, B = cs.B, nb = k / 32;
                size_t w = psegs((size_t)nout * nb * 34);
                fill_q8(pk.data() + w, nout, (int)nb, R);
                std::vector<std::vector<float>> X(B, std::vector<float>(k));
                for (uint32_t t = 0; t < B; t++) {
                    for (auto & v : X[t]) v = R.f(-6, 6);
                    memcpy(scr.data() + sXf + (size_t)t * KMAX, X[t].data(), k * 4);
                }
                GemmF32SgArgs A{w, sXf, KMAX, sYg, NMAX, k, nout, B, 0u, 0u};
                for (uint32_t t = 0; t < B; t++)
                    for (uint32_t r = 0; r < NMAX; r++) scr[sYg + t * NMAX + r] = 12345.f;
                run("k_gemm_q8_f32in_sg", true, (nout + 31) / 32, (B + 31) / 32, 32, 4, &A, sizeof(A));
                // FP64 oracle: exact Q8_0 dequant weights x raw fp32 X
                std::vector<float> ref((size_t)B * nout), got((size_t)B * nout);
                for (uint32_t t = 0; t < B; t++)
                    for (uint32_t r = 0; r < nout; r++) {
                        const uint8_t * wr = pk.data() + w + (size_t)r * nb * 34;
                        double a = 0;
                        for (uint32_t bi = 0; bi < nb; bi++) {
                            const uint8_t * wb = wr + (size_t)bi * 34;
                            const double d = h16d((uint16_t)(wb[0] | (wb[1] << 8)));
                            for (int i = 0; i < 32; i++)
                                a += d * (double)sb(wb[2 + i]) * (double)X[t][(size_t)bi * 32 + i];
                        }
                        ref[(size_t)t * nout + r] = (float)a;
                        got[(size_t)t * nout + r] = scr[sYg + (size_t)t * NMAX + r];
                    }
                char nm[64]; snprintf(nm, sizeof(nm), "gemm_f32in_sg k=%u n=%u B=%u", k, nout, B);
                gate(nm, ref.data(), got.data(), ref.size(), GATE_SG);
                // add_to_out=1: Y pre-filled with y0, expect y0 + W*X, tails untouched
                std::vector<float> y0((size_t)B * NMAX, 0.f);
                for (uint32_t t = 0; t < B; t++)
                    for (uint32_t r = 0; r < nout; r++) {
                        y0[(size_t)t * nout + r] = R.f(-1, 1);
                        scr[sYg + t * NMAX + r] = y0[(size_t)t * nout + r];
                    }
                for (uint32_t t = 0; t < B; t++)
                    for (uint32_t r = nout; r < NMAX; r++) scr[sYg + t * NMAX + r] = 12345.f;
                GemmF32SgArgs Aadd = A; Aadd.add_to_out = 1u;
                run("k_gemm_q8_f32in_sg", true, (nout + 31) / 32, (B + 31) / 32, 32, 4, &Aadd, sizeof(Aadd));
                std::vector<float> ref2(ref.size());
                for (uint32_t t = 0; t < B; t++)
                    for (uint32_t r = 0; r < nout; r++) {
                        ref2[(size_t)t * nout + r] =
                            (float)((double)y0[(size_t)t * nout + r] + (double)ref[(size_t)t * nout + r]);
                        got[(size_t)t * nout + r] = scr[sYg + (size_t)t * NMAX + r];
                    }
                snprintf(nm, sizeof(nm), "gemm_f32in_sg add k=%u B=%u", k, B);
                gate(nm, ref2.data(), got.data(), ref2.size(), GATE_SG);
                int tail = 0;
                for (uint32_t t = 0; t < B; t++)
                    for (uint32_t r = nout; r < NMAX; r++)
                        if (scr[sYg + t * NMAX + r] != 12345.f) tail++;
                CHECK(tail == 0, "gemm_f32in_sg add k=%u B=%u touched %d tail slots", k, B, tail);
            }
            // cross-check vs the 3A pipeline on one token (k_quant + k_gemv_q8_k2304_m2):
            // difference is Q8_0 activation rounding -> relaxed gate, cos still ~1.
            {
                const uint32_t k = 2304, nout = 2304, nb = k / 32;
                size_t w = psegs((size_t)nout * nb * 34);
                fill_q8(pk.data() + w, nout, (int)nb, R);
                std::vector<float> x(k);
                for (auto & v : x) v = R.f(-4, 4);
                memcpy(scr.data() + sXf, x.data(), k * 4);
                uint32_t sXq1 = seg(k / 4 + 15), sDx1 = seg(nb + 15), sY1 = seg(nout + 32);
                std::vector<uint8_t> codes(k);
                std::vector<float> d(nb);
                quantize(x.data(), (int)k, codes.data(), d.data());
                memcpy(scr.data() + sXq1, codes.data(), k);
                for (uint32_t b = 0; b < nb; b++) scr[sDx1 + b] = d[b];
                GemmF32SgArgs A{w, sXf, k, sYg, nout, k, nout, 1, 0u, 0u};
                run("k_gemm_q8_f32in_sg", true, (nout + 31) / 32, 1, 32, 4, &A, sizeof(A));
                GemvArgs GA{w, 0, sXq1, sDx1, sY1, 0, nout, 0, 0, 0};
                run("k_gemv_q8_k2304_m2", true, (nout + 15) / 16, 1, 256, 1, &GA, sizeof(GA));
                gate("gemm_f32in_sg vs quant+gemv", scr.data() + sY1,
                     scr.data() + sYg, nout, GATE_XQ);
            }
        }

        // ================= 9. batched helper kernels (3B-1) =================
        const uint32_t cur9 = cur;
        const size_t pcur9 = pcur;
        // ---- 9a. k_emb_gather_batch ----
        {
            const uint32_t B = 16, TBLR = 512, RB = 72;   // 72 blocks = 2304 elem/row
            size_t ew = psegs((size_t)TBLR * RB * 34);
            fill_q8(pk.data() + ew, TBLR, (int)RB, R);
            uint32_t sTok = seg(B + 4), sEmb = seg(B * 2304 + 64),
                     sEmbRef = seg(B * 2304 + 64), sEmb1 = seg(2304 + 16);
            std::vector<uint32_t> toks(B);
            for (uint32_t b = 0; b < B; b++) {
                toks[b] = R.u32() % TBLR;
                ((uint32_t *)scr.data())[sTok + b] = toks[b];
            }
            EmbBatchArgs EA{ew, sTok, sEmb, 2304u, RB, B, 0u, 0u};
            for (uint32_t i = 0; i < B * 2304; i++) scr[sEmb + i] = -9.f;
            run("k_emb_gather_batch", false, RB, B, 32, 1, &EA, sizeof(EA));
            int bad = 0;
            for (uint32_t b = 0; b < B; b++) {
                const uint8_t * row = pk.data() + ew + (size_t)toks[b] * RB * 34;
                for (uint32_t blk = 0; blk < RB; blk++) {
                    const uint8_t * wp = row + (size_t)blk * 34;
                    const float dv = h16d((uint16_t)(wp[0] | (wp[1] << 8)));
                    for (int i = 0; i < 32; i++) {
                        const float e = dv * (float)sb(wp[2 + i]);
                        if (scr[sEmb + b * 2304 + blk * 32 + i] != e) bad++;
                    }
                }
            }
            printf("emb_gather_batch B=%-3u      bitwise %d mismatches (host oracle) %s\n",
                   B, bad, bad ? "FAIL" : "OK");
            if (bad) g_fail = true;
            // batched vs single-token kernel rows: bitwise
            id<MTLBuffer> bTok = [rt.device() newBufferWithLength:4 options:MTLResourceStorageModeShared];
            CHECK(bTok != nil, "alloc tok buffer");
            for (uint32_t b = 0; b < B; b++) {
                memcpy(bTok.contents, &toks[b], 4);
                EmbArgs E{ew, sEmb1, 2304u};
                run("k_emb_gather", false, RB, 1, 32, 1, &E, sizeof(E), bTok);
                memcpy(scr.data() + sEmbRef + b * 2304, scr.data() + sEmb1, 2304 * 4);
            }
            int sbad = 0;
            for (uint32_t b = 0; b < B; b++)
                for (uint32_t i = 0; i < 2304; i++)
                    if (scr[sEmb + b * 2304 + i] != scr[sEmbRef + b * 2304 + i]) sbad++;
            printf("emb_gather_batch vs single   bitwise %d mismatches %s\n",
                   sbad, sbad ? "FAIL" : "OK");
            if (sbad) g_fail = true;
            bTok = nil;
        }
        // ---- 9b. k_rms_batch ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t rows = 2, k = 2304, nbk = k / 32;
            struct RC { uint32_t B, hasw, q8, yskip, offodd; } rcs[] = {
                {8, 0, 0, 0, 0},    // no weight, f32 out
                {16, 1, 0, 0, 0},   // weight, f32 out
                {8, 1, 1, 0, 0},    // weight + Q8_0 out
                {16, 0, 1, 1, 1},   // yoff skipped, odd x/y offsets (scalar branches)
            };
            for (auto & rc : rcs) {
                const uint32_t B = rc.B;
                const float eps = 1e-6f;
                uint32_t xb = seg(B * rows * k + 8) + (rc.offodd ? 1u : 0u);
                uint32_t yb = seg(B * rows * k + 8) + (rc.offodd ? 1u : 0u);
                uint32_t qf = seg(B * rows * k / 4 + 16);
                uint32_t df = seg(B * rows * nbk + 16);
                size_t wpk = psegs((size_t)k * 4);
                for (uint32_t i = 0; i < B * rows * k; i++) scr[xb + i] = R.f(-4, 4);
                for (uint32_t i = 0; i < k; i++) pkf[wpk / 4 + i] = R.f(-2, 2);
                // sequential single-token k_rms reference, token by token
                std::vector<float> yseq((size_t)B * rows * k);
                for (uint32_t b = 0; b < B; b++) {
                    RMSArgs ra{};
                    for (uint32_t r = 0; r < rows; r++) {
                        ra.xr[r] = xb + b * rows * k + r * k;
                        ra.yr[r] = yb + b * rows * k + r * k;
                        ra.wr[r] = wpk;
                    }
                    ra.rows = rows; ra.k = k; ra.hasw = rc.hasw; ra.eps = eps;
                    run("k_rms", true, rows, 1, 256, 1, &ra, sizeof(ra));
                    memcpy(yseq.data() + (size_t)b * rows * k,
                           scr.data() + yb + (size_t)b * rows * k, (size_t)rows * k * 4);
                }
                RmsBatchArgs A{wpk, xb, rows * k, k, yb, rows * k, k,
                               rc.q8 ? qf : M64_NOFFU, rc.q8 ? df : M64_NOFFU,
                               rows * k / 4, rows * nbk, k, B, rows,
                               rc.hasw, rc.q8, 0u, eps};
                if (rc.yskip) A.yoff = M64_NOFFU;
                for (uint32_t i = 0; i < B * rows * k; i++) scr[yb + i] = 12345.f;
                run("k_rms_batch", true, rows, B, 256, 1, &A, sizeof(A));
                int bad = 0;
                if (!rc.yskip) {
                    for (uint32_t i = 0; i < B * rows * k; i++)
                        if (scr[yb + i] != yseq[i]) bad++;
                } else {
                    for (uint32_t i = 0; i < B * rows * k; i++)
                        if (scr[yb + i] != 12345.f) bad++;   // must not write
                }
                int qbad = 0;
                if (rc.q8) {
                    std::vector<uint8_t> codes(k);
                    std::vector<float> dd(nbk);
                    for (uint32_t b = 0; b < B; b++)
                        for (uint32_t r = 0; r < rows; r++) {
                            quantize(yseq.data() + (size_t)(b * rows + r) * k, (int)k,
                                     codes.data(), dd.data());
                            const uint8_t * q = (const uint8_t *)scr.data() + qf * 4 +
                                (size_t)(b * rows + r) * k;
                            for (uint32_t j = 0; j < nbk; j++) {
                                if (scr[df + b * rows * nbk + r * nbk + j] != dd[j]) qbad++;
                                for (int i = 0; i < 32; i++)
                                    if (q[j * 32 + i] != codes[j * 32 + i]) qbad++;
                            }
                        }
                }
                printf("rms_batch B=%-3u hasw=%u q8=%u yskip=%u odd=%u: f32 %d, q8 %d mismatches %s\n",
                       B, rc.hasw, rc.q8, rc.yskip, rc.offodd, bad, qbad, (bad || qbad) ? "FAIL" : "OK");
                if (bad || qbad) g_fail = true;
            }
        }
        // ---- 9c. k_clip_batch ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t groups = 3;
            for (uint32_t per : {128u, 16u})
                for (uint32_t B : {8u, 16u})
                    for (uint32_t usesoff : {0u, 1u}) {
                        uint32_t cb = seg(B * groups * per + 16);
                        // shared scales row lives in PACK (M64_NOFF = absent)
                        size_t sb2 = usesoff ? psegs((size_t)groups * per * 4) : M64_NOFF;
                        for (uint32_t i = 0; i < B * groups * per; i++) scr[cb + i] = R.f(-8, 8);
                        if (usesoff)
                            for (uint32_t i = 0; i < groups * per; i++) pkf[sb2 / 4 + i] = R.f(0.5f, 1.5f);
                        std::vector<float> orig((size_t)B * groups * per);
                        memcpy(orig.data(), scr.data() + cb, orig.size() * 4);
                        std::vector<float> ref;
                        for (uint32_t b = 0; b < B; b++) {
                            ClipArgs ca{cb + b * groups * per, sb2, groups, per, 0.5f};
                            run("k_clip", true, groups, 1, 128, 1, &ca, sizeof(ca));
                            const size_t o = ref.size();
                            ref.resize(o + (size_t)groups * per);
                            memcpy(ref.data() + o, scr.data() + cb + (size_t)b * groups * per,
                                   (size_t)groups * per * 4);
                        }
                        memcpy(scr.data() + cb, orig.data(), orig.size() * 4);
                        ClipBatchArgs A{sb2, cb, groups * per, groups, per, B, 0u, 0.5f};
                        run("k_clip_batch", true, groups, B, 128, 1, &A, sizeof(A));
                        int bad = 0;
                        for (uint32_t b = 0; b < B; b++)
                            for (uint32_t i = 0; i < groups * per; i++)
                                if (scr[cb + b * groups * per + i] != ref[(size_t)b * groups * per + i]) bad++;
                        printf("clip_batch B=%-3u groups=%u per=%-3u soff=%u: bitwise %d mismatches %s\n",
                               B, groups, per, usesoff, bad, bad ? "FAIL" : "OK");
                        if (bad) g_fail = true;
                    }
        }
        // ---- 9d. k_gate_mul_batch ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t n = 1000;   // not a multiple of 128: tail guard
            for (uint32_t B : {8u, 16u}) {
                uint32_t ab = seg(B * n + 16), gb = seg(B * n + 16);
                std::vector<float> a0((size_t)B * n), g0((size_t)B * n);
                for (uint32_t i = 0; i < B * n; i++) { a0[i] = R.f(-4, 4); g0[i] = R.f(-4, 4); }
                std::vector<float> ref((size_t)B * n);
                // sequential reference: k_gate_mul mutates a IN PLACE, so the
                // inputs must be restored before every dispatch.
                for (uint32_t b = 0; b < B; b++) {
                    memcpy(scr.data() + ab + b * n, a0.data() + (size_t)b * n, n * 4);
                    memcpy(scr.data() + gb + b * n, g0.data() + (size_t)b * n, n * 4);
                    GateMulArgs ga{ab + b * n, gb + b * n, n};
                    run("k_gate_mul", false, n, 1, 128, 1, &ga, sizeof(ga));
                    memcpy(ref.data() + (size_t)b * n, scr.data() + ab + b * n, n * 4);
                }
                for (uint32_t i = 0; i < B * n; i++) {
                    scr[ab + i] = a0[i]; scr[gb + i] = g0[i];
                }
                GateMulBatchArgs A{ab, gb, n, n, n, B};
                run("k_gate_mul_batch", false, n, B, 128, 1, &A, sizeof(A));
                int bad = 0;
                for (uint32_t b = 0; b < B; b++)
                    for (uint32_t i = 0; i < n; i++)
                        if (scr[ab + b * n + i] != ref[(size_t)b * n + i]) bad++;
                printf("gate_mul_batch B=%-3u n=%u: bitwise %d mismatches vs sequential %s\n",
                       B, n, bad, bad ? "FAIL" : "OK");
                if (bad) g_fail = true;
                std::vector<float> host((size_t)B * n);
                for (uint32_t i = 0; i < B * n; i++)
                    host[i] = a0[i] * (1.f / (1.f + expf(-g0[i])));
                gate("gate_mul_batch vs host fp32", host.data(), scr.data() + ab, host.size(), GATE_ATTN);
            }
        }
        // ---- 9e. k_situ_quant_batch ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t nbk = 72;   // k = 2304
            const float beta = 2.0f, lb = 4.0f, ib = 1.f / beta, il = 1.f / lb;
            for (uint32_t B : {8u, 16u}) {
                const uint32_t npairs = B * 8;
                uint32_t g1 = seg(npairs * 2304 + 16), u1 = seg(npairs * 2304 + 16);
                uint32_t qa = seg(npairs * 576 + 16), da = seg(npairs * 72 + 16);   // sequential
                uint32_t qb = seg(npairs * 576 + 16), db = seg(npairs * 72 + 16);   // batched
                for (uint32_t i = 0; i < npairs * 2304; i++) { scr[g1 + i] = R.f(-6, 6); scr[u1 + i] = R.f(-6, 6); }
                for (uint32_t p = 0; p < npairs; p++) {
                    SituArgs sa{g1 + p * 2304, u1 + p * 2304, qa + p * 576, da + p * 72, nbk,
                                beta, lb, ib, il};
                    run("k_situ_quant", false, nbk, 1, 32, 1, &sa, sizeof(sa));
                }
                SituBatchArgs A{g1, u1, qb, db, 2304u, 2304u, 576u, 72u, nbk, npairs,
                                beta, lb, ib, il};
                run("k_situ_quant_batch", false, nbk, npairs, 32, 1, &A, sizeof(A));
                int bad = 0;
                for (uint32_t p = 0; p < npairs; p++) {
                    const uint8_t * q1 = (const uint8_t *)scr.data() + qa * 4 + (size_t)p * 2304;
                    const uint8_t * q2 = (const uint8_t *)scr.data() + qb * 4 + (size_t)p * 2304;
                    for (uint32_t j = 0; j < nbk; j++) {
                        if (scr[da + p * 72 + j] != scr[db + p * 72 + j]) bad++;
                        for (int i = 0; i < 32; i++)
                            if (q1[j * 32 + i] != q2[j * 32 + i]) bad++;
                    }
                }
                printf("situ_quant_batch B=%-3u pairs=%-4u: bitwise %d mismatches vs sequential %s\n",
                       B, npairs, bad, bad ? "FAIL" : "OK");
                if (bad) g_fail = true;
            }
        }
        // ---- 9f. k_moe_down_reduce_batch ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t nout = 2304;
            for (uint32_t B : {8u, 16u})
                for (uint32_t topk : {8u, 4u}) {
                    uint32_t ib2 = seg((size_t)B * topk * nout + 16), wb2 = seg(B * 8 + 8),
                             ob = seg(B * nout + 16);
                    for (uint32_t i = 0; i < B * topk * nout; i++) scr[ib2 + i] = R.f(-3, 3);
                    for (uint32_t i = 0; i < B * 8; i++) scr[wb2 + i] = R.f(0.f, 1.f);
                    MoeReduceBatchArgs A{ib2, wb2, ob, nout /*in_stride*/, 8u /*w_stride*/,
                                         nout /*out_stride*/, nout, topk, B, 0u};
                    run("k_moe_down_reduce_batch", false, nout, B, 128, 1, &A, sizeof(A));
                    // host fp32 mirror with identical accumulation order -> bitwise
                    int bad = 0;
                    for (uint32_t b = 0; b < B; b++)
                        for (uint32_t c = 0; c < nout; c++) {
                            float xin = 0.f;
                            for (uint32_t e = 0; e < topk; e++)
                                xin = xin + scr[wb2 + b * 8 + e] * scr[ib2 + ((size_t)b * topk + e) * nout + c];
                            if (scr[ob + b * nout + c] != xin) bad++;
                        }
                    printf("moe_down_reduce_batch B=%-3u topk=%u: bitwise %d mismatches vs host fp32 %s\n",
                           B, topk, bad, bad ? "FAIL" : "OK");
                    if (bad) g_fail = true;
                }
        }
        // ---- 9g. k_router_topk_batch ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t NE = 128;
            struct RTC { uint32_t B; int tie; };
            for (RTC rc : {RTC{8, 0}, RTC{16, 0}, RTC{2, 1}}) {
                const uint32_t B = rc.B;
                uint32_t pb = seg(B * NE + 16), sb3 = seg(B * 8 + 8), wb3 = seg(B * 8 + 8);
                size_t biasp = psegs(NE * 4);
                for (uint32_t i = 0; i < B * NE; i++) scr[pb + i] = R.f(-4, 4);
                for (uint32_t i = 0; i < NE; i++) pkf[biasp / 4 + i] = R.f(-0.1f, 0.1f);
                if (rc.tie) {
                    // force exact ties on token 0: experts 5/20 tied (largest),
                    // 40/41/42 tied next
                    for (uint32_t e = 0; e < NE; e++) scr[pb + e] = R.f(-3, 1);
                    scr[pb + 5] = 6.0f; scr[pb + 20] = 6.0f;
                    scr[pb + 40] = 5.5f; scr[pb + 41] = 5.5f; scr[pb + 42] = 5.5f;
                    pkf[biasp / 4 + 5] = 0.25f; pkf[biasp / 4 + 20] = 0.25f;
                    pkf[biasp / 4 + 40] = 0.2f; pkf[biasp / 4 + 41] = 0.2f; pkf[biasp / 4 + 42] = 0.2f;
                }
                for (uint32_t b = 0; b < B; b++) {
                    RouterArgs ra{pb + b * NE, biasp, sb3 + b * 8, wb3 + b * 8, 1.5f};
                    run("k_router_topk", true, 1, 1, 128, 1, &ra, sizeof(ra));
                }
                std::vector<int> selref((size_t)B * 8);
                std::vector<float> wref((size_t)B * 8);
                for (uint32_t b = 0; b < B; b++) {
                    memcpy(selref.data() + b * 8, scr.data() + sb3 + b * 8, 8 * 4);
                    memcpy(wref.data() + b * 8, scr.data() + wb3 + b * 8, 8 * 4);
                }
                RouterBatchArgs A{biasp, pb, sb3, wb3, NE /*p_stride*/, 8 /*sel_stride*/,
                                  8 /*w_stride*/, B, 0u, 1.5f};
                run("k_router_topk_batch", true, B, 1, 128, 1, &A, sizeof(A));
                int bad = 0;
                for (uint32_t b = 0; b < B * 8; b++) {
                    if (((const int *)scr.data())[sb3 + b] != selref[b]) bad++;
                    if (scr[wb3 + b] != wref[b]) bad++;
                }
                printf("router_topk_batch B=%-3u tie=%d: bitwise %d mismatches vs sequential %s\n",
                       B, rc.tie, bad, bad ? "FAIL" : "OK");
                if (bad) g_fail = true;
                if (rc.tie) {
                    const int want[5] = {5, 20, 40, 41, 42};
                    int tbad = 0;
                    for (int t = 0; t < 5; t++)
                        if (((const int *)scr.data())[sb3 + t] != want[t]) tbad++;
                    printf("router_topk_batch tie-break order 5,20,40,41,42: %s\n",
                           tbad ? "FAIL" : "OK");
                    if (tbad) g_fail = true;
                }
            }
        }

        // ================= 10. Milestone 3B-2 dispatch helpers =================
        // ---- 10a. k_gemv_f32_batch ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t nout = 128, k = 2304;
            for (uint32_t B : {8u, 16u}) {
                uint32_t xb = seg(B * k + 16), yb = seg(B * nout + 16), y1 = seg(nout + 16);
                size_t wpk = psegs((size_t)nout * k * 4);
                for (uint32_t i = 0; i < B * k; i++) scr[xb + i] = R.f(-3, 3);
                for (size_t i = 0; i < (size_t)nout * k; i++) pkf[wpk / 4 + i] = R.f(-1, 1);
                std::vector<float> yseq((size_t)B * nout);
                for (uint32_t b = 0; b < B; b++) {
                    GemvF32Args ga{wpk, xb + b * k, y1, nout, k};
                    run("k_gemv_f32_tg", true, nout, 1, 256, 1, &ga, sizeof(ga));
                    memcpy(yseq.data() + (size_t)b * nout, scr.data() + y1, nout * 4);
                }
                GemvF32BatchArgs A{wpk, xb, yb, k, nout, nout, k, B, 0u};
                run("k_gemv_f32_batch", true, nout, B, 256, 1, &A, sizeof(A));
                int bad = 0;
                for (uint32_t i = 0; i < B * nout; i++)
                    if (scr[yb + i] != yseq[i]) bad++;
                printf("gemv_f32_batch B=%-3u       bitwise %d mismatches vs sequential %s\n",
                       B, bad, bad ? "FAIL" : "OK");
                if (bad) g_fail = true;
            }
        }
        // ---- 10b. k_moe_bucket_build ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t B = 32, TOPK = 8, NE = 128, npairs = B * TOPK;
            uint32_t sSel = seg(npairs + 16), sCnt = seg(NE + 16),
                     sOff = seg(NE + 16), sList = seg(npairs + 16);
            std::vector<uint32_t> sel(npairs);
            // Force low expert IDs (0, 1, 2 -> subnormal uint bits if loaded as float!),
            // middle/max IDs (127), and untouched experts (counts == 0).
            for (uint32_t p = 0; p < npairs; p++) {
                uint32_t e = (p < 12) ? (p % 3) : ((p < 20) ? 127u : ((R.u32() % 40u) * 3u));
                if (e >= NE) e = NE - 1;
                sel[p] = e;
                ((uint32_t *)scr.data())[sSel + p] = e;
            }
            for (uint32_t i = 0; i < NE; i++) {
                ((uint32_t *)scr.data())[sCnt + i] = 0xDEADBEEFu;
                ((uint32_t *)scr.data())[sOff + i] = 0xDEADBEEFu;
            }
            for (uint32_t i = 0; i < npairs + 8; i++)
                ((uint32_t *)scr.data())[sList + i] = 0xDEADBEEFu;
            MoeBucketArgs A{sSel, sCnt, sOff, sList, NE, npairs, 0u, 0u};
            run("k_moe_bucket_build", true, 1, 1, 256, 1, &A, sizeof(A));
            std::vector<uint32_t> ref_cnt(NE, 0);
            for (uint32_t p = 0; p < npairs; p++) ref_cnt[sel[p]]++;
            int bad = 0;
            uint32_t acc = 0;
            std::vector<int> seen(npairs, 0);
            const uint32_t * cnt = (const uint32_t *)scr.data() + sCnt;
            const uint32_t * off = (const uint32_t *)scr.data() + sOff;
            const uint32_t * lst = (const uint32_t *)scr.data() + sList;
            for (uint32_t e = 0; e < NE; e++) {
                if (cnt[e] != ref_cnt[e]) bad++;
                if (off[e] != acc) bad++;
                for (uint32_t j = 0; j < cnt[e]; j++) {
                    uint32_t p = lst[acc + j];
                    if (p >= npairs || sel[p] != e) bad++;
                    else seen[p]++;
                }
                acc += ref_cnt[e];
            }
            if (acc != npairs) bad++;
            for (uint32_t p = 0; p < npairs; p++) if (seen[p] != 1) bad++;
            for (uint32_t p = npairs; p < npairs + 8; p++) if (lst[p] != 0xDEADBEEFu) bad++;
            printf("moe_bucket_build B=%u       %d structural/FTZ errors %s\n",
                   B, bad, bad ? "FAIL" : "OK");
            if (bad) g_fail = true;
        }
        // ---- 10c. k_moe_grouped_gemm_q8 arena wtab_off != 0 ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t NEXP = 16, NSLOTS = 8, TOPK = 8, B = 6, k = 128, nout = 48, nb = k / 32;
            const uint32_t npairs = B * TOPK;
            const uint64_t exp_stride = (uint64_t)nout * nb * 34;
            // Only slots 0..NSLOTS-1 hold weights in PACK (permuted slot assignment,
            // testing slot=0, slot=7, and active experts 1,2,3,5,9,15 < FTZ threshold).
            size_t arenaBase = psegs((size_t)NSLOTS * exp_stride);
            for (uint32_t s = 0; s < NSLOTS; s++)
                fill_q8(pk.data() + arenaBase + (size_t)s * exp_stride, nout, (int)nb, R);
            const uint32_t active[8] = {1u, 2u, 3u, 5u, 7u, 9u, 11u, 15u};
            const uint32_t slots[8]  = {5u, 0u, 7u, 2u, 6u, 1u, 4u, 3u};
            uint32_t qstr = k / 4 + 4, dstr = nb + 4;
            uint32_t sXq = seg(B * qstr + 16), sDx = seg(B * dstr + 16), sY = seg(npairs * nout + 16);
            uint32_t sSel = seg(npairs + 16), sCnt = seg(NEXP + 16), sOff = seg(NEXP + 16),
                     sList = seg(npairs + 16), sWtab = seg(NEXP + 16);
            for (uint32_t t = 0; t < B; t++) {
                std::vector<float> x(k);
                for (auto & v : x) v = R.f(-4, 4);
                std::vector<uint8_t> codes; std::vector<float> d;
                quantize(x, (int)k, codes, d);
                memcpy(scr.data() + sXq + t * qstr, codes.data(), k);
                for (uint32_t b = 0; b < nb; b++) scr[sDx + t * dstr + b] = d[b];
                for (uint32_t k0 = 0; k0 < TOPK; k0++)
                    ((uint32_t *)scr.data())[sSel + t * TOPK + k0] = active[(k0 + t) % TOPK];
            }
            std::vector<uint32_t> exp2slot(NEXP, 0);
            for (uint32_t i = 0; i < 8; i++) {
                exp2slot[active[i]] = slots[i];
                ((uint32_t *)scr.data())[sWtab + active[i]] = slots[i];
            }
            MoeBucketArgs BA{sSel, sCnt, sOff, sList, NEXP, npairs, 0u, 0u};
            run("k_moe_bucket_build", true, 1, 1, 256, 1, &BA, sizeof(BA));
            MoeGroupedGemmArgs A{arenaBase, exp_stride, 0ull, sXq, sDx, sY, sList,
                                 sCnt, sOff, nout, k, qstr, dstr, nout, NEXP, TOPK, 0u, sWtab, 0u};
            run("k_moe_grouped_gemm_q8", true, (nout + 3) / 4, NEXP, 32, 4, &A, sizeof(A));
            std::vector<float> ref((size_t)npairs * nout);
            for (uint32_t p = 0; p < npairs; p++) {
                uint32_t t = p / TOPK, e = ((const uint32_t *)scr.data())[sSel + p];
                const int8_t * xc = (const int8_t *)(scr.data() + sXq + (size_t)t * qstr);
                const float * dx = scr.data() + sDx + (size_t)t * dstr;
                const uint8_t * wrow = pk.data() + arenaBase + (size_t)exp2slot[e] * exp_stride;
                for (uint32_t r = 0; r < nout; r++)
                    ref[(size_t)p * nout + r] = (float)gemm_oracle(wrow + (size_t)r * nb * 34, xc, dx, (int)k);
            }
            gate("moe_grouped wtab arena", ref.data(), scr.data() + sY, ref.size(), GATE_GEMM);
        }
        // ---- 10d. k_situ_f32_batch (Milestone 3C) ----
        cur = cur9; pcur = pcur9;
        {
            const uint32_t k = 1152;
            const float beta = 4.0f, lb = 25.0f, ib = 1.f / beta, il = 1.f / lb;
            for (uint32_t B : {16u, 128u}) {
                const uint32_t npairs = B * 8;
                uint32_t go = seg(npairs * k + 16), uo = seg(npairs * k + 16), ho = seg(npairs * k + 16);
                for (uint32_t i = 0; i < npairs * k; i++) { scr[go + i] = R.f(-6, 6); scr[uo + i] = R.f(-6, 6); }
                SituF32BatchArgs A{go, uo, ho, k, k, k, k, npairs, beta, lb, ib, il};
                run("k_situ_f32_batch", false, k, npairs, 128, 1, &A, sizeof(A));
                std::vector<float> ref((size_t)npairs * k);
                for (size_t i = 0; i < (size_t)npairs * k; i++) {
                    float x = scr[go + i], u = scr[uo + i];
                    float act = beta * tanhf(x * ib) * (1.f / (1.f + expf(-x)));
                    ref[i] = act * (lb * tanhf(u * il));
                }
                char nm[64]; snprintf(nm, sizeof(nm), "situ_f32_batch B=%u", B);
                gate(nm, ref.data(), scr.data() + ho, ref.size(), GATE_ATTN);
            }
        }
        // ---- 10e. k_moe_grouped_f32in_sg (Milestone 3C, multi-tile + arena) ----
        {
            struct SC { uint32_t B, NEXP, NSLOTS, TOPK, k, nout, is_down, use_wtab; } scs[] = {
                {8u,   16u,  16u, 8u, 96u,   37u,  0u, 0u},   // odd nb=3 + nout tail + cnt=0/small/37?
                {32u,  32u,  12u, 8u, 128u,  64u,  0u, 1u},   // is_down=0 + wtab + cnt > 32
                {128u, 128u, 16u, 8u, 128u,  96u,  0u, 1u},   // full B=128, npairs=1024, cnt crossing 32
                {128u, 128u, 16u, 8u, 96u,   64u,  1u, 1u},   // is_down=1 + odd nb=3 + wtab
            };
            for (auto & sc : scs) {
                cur = cur9; pcur = pcur9;
                const uint32_t B = sc.B, NEXP = sc.NEXP, TOPK = sc.TOPK,
                               k = sc.k, nout = sc.nout, nb = k / 32, npairs = B * TOPK;
                const uint64_t exp_stride = (uint64_t)nout * nb * 34;
                const uint32_t nstore = sc.use_wtab ? sc.NSLOTS : NEXP;
                size_t wbase = psegs((size_t)nstore * exp_stride);
                for (uint32_t s = 0; s < nstore; s++)
                    fill_q8(pk.data() + wbase + (size_t)s * exp_stride, nout, (int)nb, R);
                const uint32_t nrows_x = sc.is_down ? npairs : B;
                uint32_t sX = seg(nrows_x * k + 32), sY = seg(npairs * nout + 32);
                uint32_t sSel = seg(npairs + 16), sCnt = seg(NEXP + 16), sOff = seg(NEXP + 16),
                         sList = seg(npairs + 16), sWtab = seg(NEXP + 16);
                for (uint32_t i = 0; i < nrows_x * k; i++) scr[sX + i] = R.f(-4, 4);
                // Force expert 0 untouched (cnt=0), expert 1 hit 37 times when npairs>=64 (multi-tile >32!),
                // and diverse counts across remaining experts.
                for (uint32_t p = 0; p < npairs; p++) {
                    uint32_t e;
                    if (npairs >= 64 && p < 37) e = 1u;
                    else e = 1u + ((p * 7u + (p / TOPK) * 3u) % (NEXP - 1u));
                    ((uint32_t *)scr.data())[sSel + p] = e;
                }
                std::vector<uint32_t> exp2slot(NEXP, 0);
                for (uint32_t e = 0; e < NEXP; e++) {
                    uint32_t s = sc.use_wtab ? ((e * 5u + 3u) % sc.NSLOTS) : e;
                    exp2slot[e] = s;
                    ((uint32_t *)scr.data())[sWtab + e] = s;
                }
                MoeBucketArgs BA{sSel, sCnt, sOff, sList, NEXP, npairs, 0u, 0u};
                run("k_moe_bucket_build", true, 1, 1, 256, 1, &BA, sizeof(BA));
                MoeGroupedF32SgArgs A{wbase, exp_stride, 0ull, sX, sY, sList, sCnt, sOff,
                                      sc.use_wtab ? sWtab : 0u, nout, k, k, nout, NEXP, TOPK,
                                      sc.is_down, 0u};
                run("k_moe_grouped_f32in_sg", true, (nout + 31) / 32, NEXP, 32, 4, &A, sizeof(A));
                std::vector<float> ref((size_t)npairs * nout);
                for (uint32_t p = 0; p < npairs; p++) {
                    const uint32_t xin = sc.is_down ? p : (p / TOPK);
                    const uint32_t e = ((const uint32_t *)scr.data())[sSel + p];
                    const uint8_t * wr = pk.data() + wbase + (size_t)exp2slot[e] * exp_stride;
                    for (uint32_t r = 0; r < nout; r++) {
                        double a = 0;
                        const uint8_t * rr = wr + (size_t)r * nb * 34;
                        for (uint32_t bi = 0; bi < nb; bi++) {
                            const uint8_t * wb = rr + (size_t)bi * 34;
                            const double d = h16d((uint16_t)(wb[0] | (wb[1] << 8)));
                            for (int i = 0; i < 32; i++)
                                a += d * (double)sb(wb[2 + i]) * (double)scr[sX + (size_t)xin * k + bi * 32 + i];
                        }
                        ref[(size_t)p * nout + r] = (float)a;
                    }
                }
                char nm[64];
                snprintf(nm, sizeof(nm), "moe_grouped_sg B=%-3u d=%u w=%u", B, sc.is_down, sc.use_wtab);
                gate(nm, ref.data(), scr.data() + sY, ref.size(), GATE_SG);
            }
        }
        // ---- 10f. k_moe_grouped_f32in_sg_{q4k,q5_0,q5_1} (Milestone 4B) ----
        // FP64-oracle parity per DotS gate (1024-ulp fp32 bound): q4k needs k%256==0.
        // Expert 0 stays cnt=0 in every scenario; expert 1 gets cnt=37 (> 32-tile).
        {
            const double EPSF = 1.1920928955078125e-7;
            auto gate_ulp = [&](const char * name, const std::vector<DotS> & ref, uint32_t off, size_t n) {
                const float * got = scr.data() + off;
                double worst = 0.0; size_t bad = 0, nonfinite = 0;
                for (size_t i = 0; i < n; i++) {
                    const double g = (double)got[i];
                    if (!std::isfinite(g) || !std::isfinite(ref[i].a)) { nonfinite++; continue; }
                    double bnd = 1e-4 * std::fabs(ref[i].a) + 1024.0 * EPSF * ref[i].sab;
                    double dl = std::fabs(g - ref[i].a);
                    if (dl > bnd) bad++;
                    if (bnd != 0.0 && dl / bnd > worst) worst = dl / bnd;
                }
                bad += nonfinite;   // NaN/inf never compares > bound; count them explicitly
                CHECK(bad == 0, "%s %zu/%zu over 1024-ulp bound (%zu nonfinite, worst %.1f)",
                      name, bad, n, nonfinite, worst);
                printf("%-36s ulp-gated %zu rows %s (worst %.2f bound)\n", name, n, bad ? "FAIL" : "OK", worst);
            };
            const char * fq[3] = {"q4k", "q5_0", "q5_1"};
            const char * fk[3] = {"k_moe_grouped_f32in_sg_q4k", "k_moe_grouped_f32in_sg_q5_0",
                                  "k_moe_grouped_f32in_sg_q5_1"};
            struct SC4 { uint32_t B, NEXP, NSLOTS, TOPK, k, nout, is_down, use_wtab; } scs4[] = {
                {8u,   16u,  16u, 8u, 256u,  37u,  0u, 0u},   // direct + cnt=0/37 + nout tail
                {32u,  32u,  12u, 8u, 512u,  64u,  0u, 1u},   // wtab arena + multi-tile
                {128u,128u,  16u, 8u, 256u,  96u,  0u, 1u},   // B=128 full, npairs=1024
                {128u,128u,  16u, 8u, 512u,  64u,  1u, 1u},   // is_down=1 + wtab
            };
            float v4[256], v5[32];
            for (int f = 0; f < 3; f++) {
                const int fmt = f + 1;
                const uint32_t bb = fmt == 1 ? 18u : (fmt == 2 ? 22u : 24u);
                for (auto & sc : scs4) {
                    cur = cur9; pcur = pcur9;
                    const uint32_t B = sc.B, NEXP = sc.NEXP, TOPK = sc.TOPK,
                                   k = sc.k, nout = sc.nout, nb = k / 32, npairs = B * TOPK;
                    const uint64_t exp_stride = (uint64_t)nout * nb * bb;
                    const uint32_t nstore = sc.use_wtab ? sc.NSLOTS : NEXP;
                    size_t wbase = psegs((size_t)nstore * exp_stride);
                    for (uint32_t s = 0; s < nstore; s++) {
                        uint8_t * wp = pk.data() + wbase + (size_t)s * exp_stride;
                        for (uint32_t r = 0; r < nout; r++) {
                            if (fmt == 1)
                                for (uint32_t b = 0; b < nb; b += 8u) {
                                    for (int i = 0; i < 256; i++) v4[i] = R.f(-0.4f, 0.4f);
                                    fill_q4k(wp + (size_t)r * nb * 18 + (size_t)b * 18, v4,
                                             ((r + b) & 1) ? R.f(-0.0003f, 0.0003f) : 0.0f, R);
                                }
                            else
                                for (uint32_t b = 0; b < nb; b++) {
                                    for (int i = 0; i < 32; i++) v5[i] = R.f(-0.4f, 0.4f);
                                    fill_q5(wp + (size_t)r * nb * bb + (size_t)b * bb, v5, fmt == 3);
                                }
                        }
                    }
                    const uint32_t nrows_x = sc.is_down ? npairs : B;
                    uint32_t sX = seg(nrows_x * k + 32), sY = seg(npairs * nout + 32);
                    uint32_t sSel = seg(npairs + 16), sCnt = seg(NEXP + 16), sOff = seg(NEXP + 16),
                             sList = seg(npairs + 16), sWtab = seg(NEXP + 16);
                    for (uint32_t i = 0; i < nrows_x * k; i++) scr[sX + i] = R.f(-4, 4);
                    for (uint32_t p = 0; p < npairs; p++) {
                        uint32_t e;
                        if (npairs >= 64 && p < 37) e = 1u;
                        else e = 1u + ((p * 7u + (p / TOPK) * 3u) % (NEXP - 1u));
                        ((uint32_t *)scr.data())[sSel + p] = e;
                    }
                    std::vector<uint32_t> exp2slot(NEXP, 0);
                    for (uint32_t e = 0; e < NEXP; e++) {
                        uint32_t sl = sc.use_wtab ? ((e * 5u + 3u) % sc.NSLOTS) : e;
                        exp2slot[e] = sl;
                        ((uint32_t *)scr.data())[sWtab + e] = sl;
                    }
                    MoeBucketArgs BA{sSel, sCnt, sOff, sList, NEXP, npairs, 0u, 0u};
                    run("k_moe_bucket_build", true, 1, 1, 256, 1, &BA, sizeof(BA));
                    MoeGroupedF32SgArgs A{wbase, exp_stride, 0ull, sX, sY, sList, sCnt, sOff,
                                          sc.use_wtab ? sWtab : 0u, nout, k, k, nout, NEXP, TOPK,
                                          sc.is_down, 0u};
                    run(fk[f], true, (nout + 31) / 32, NEXP, 32, 4, &A, sizeof(A));
                    std::vector<DotS> ref((size_t)npairs * nout);
                    for (uint32_t p = 0; p < npairs; p++) {
                        const uint32_t xin = sc.is_down ? p : (p / TOPK);
                        const uint32_t e = ((const uint32_t *)scr.data())[sSel + p];
                        const uint8_t * wr = pk.data() + wbase + (size_t)exp2slot[e] * exp_stride;
                        const float * xrow = scr.data() + sX + (size_t)xin * k;
                        for (uint32_t r = 0; r < nout; r++)
                            ref[(size_t)p * nout + r] = dot_sg_q(fmt, wr + (size_t)r * nb * bb, nb, xrow);
                    }
                    char nm[64];
                    snprintf(nm, sizeof(nm), "moe_grouped_sg_%s B=%-3u d=%u w=%u", fq[f], B, sc.is_down, sc.use_wtab);
                    gate_ulp(nm, ref, sY, ref.size());
                }
            }
            // wtab indirection == direct, bitwise (identical slot contents, shuffled mapping)
            for (int f = 0; f < 3; f++) {
                const int fmt = f + 1;
                const uint32_t bb = fmt == 1 ? 18u : (fmt == 2 ? 22u : 24u);
                cur = cur9; pcur = pcur9;
                const uint32_t B = 16u, NEXP = 8u, TOPK = 8u, k = 256u, nout = 64u,
                               nb = k / 32, npairs = B * TOPK;
                const uint64_t exp_stride = (uint64_t)nout * nb * bb;
                size_t wbase = psegs((size_t)NEXP * exp_stride);
                for (uint32_t r = 0; r < nout; r++) {
                    if (fmt == 1)
                        for (uint32_t b = 0; b < nb; b += 8u) {
                            for (int i = 0; i < 256; i++) v4[i] = R.f(-0.4f, 0.4f);
                            fill_q4k(pk.data() + wbase + (size_t)r * nb * 18 + (size_t)b * 18, v4, 0.0f, R);
                        }
                    else
                        for (uint32_t b = 0; b < nb; b++) {
                            for (int i = 0; i < 32; i++) v5[i] = R.f(-0.4f, 0.4f);
                            fill_q5(pk.data() + wbase + (size_t)r * nb * bb + (size_t)b * bb, v5, fmt == 3);
                        }
                }
                for (uint32_t s = 1; s < NEXP; s++)      // all slots carry identical expert bytes
                    memcpy(pk.data() + wbase + (size_t)s * exp_stride, pk.data() + wbase, (size_t)exp_stride);
                uint32_t sX = seg(B * k + 32), sYa = seg(npairs * nout + 32), sYb = seg(npairs * nout + 32);
                uint32_t sSel = seg(npairs + 16), sCnt = seg(NEXP + 16), sOff = seg(NEXP + 16),
                         sList = seg(npairs + 16), sWtab = seg(NEXP + 16);
                for (uint32_t i = 0; i < B * k; i++) scr[sX + i] = R.f(-4, 4);
                for (uint32_t p = 0; p < npairs; p++)
                    ((uint32_t *)scr.data())[sSel + p] = (p * 3u + 1u) % NEXP;
                MoeBucketArgs BA{sSel, sCnt, sOff, sList, NEXP, npairs, 0u, 0u};
                run("k_moe_bucket_build", true, 1, 1, 256, 1, &BA, sizeof(BA));
                MoeGroupedF32SgArgs A{wbase, exp_stride, 0ull, sX, sYa, sList, sCnt, sOff, 0u,
                                       nout, k, k, nout, NEXP, TOPK, 0u, 0u};
                MoeGroupedF32SgArgs B2 = A; B2.yoff = sYb;
                for (uint32_t e = 0; e < NEXP; e++) ((uint32_t *)scr.data())[sWtab + e] = (e * 3u + 1u) % NEXP;
                B2.wtab_off = sWtab;
                run(fk[f], true, (nout + 31) / 32, NEXP, 32, 4, &A, sizeof(A));
                run(fk[f], true, (nout + 31) / 32, NEXP, 32, 4, &B2, sizeof(B2));
                const uint32_t * pa = (const uint32_t *)scr.data();
                size_t bad = 0;
                for (uint32_t i = 0; i < npairs * nout; i++)
                    if (pa[sYa + i] != pa[sYb + i]) bad++;
                CHECK(bad == 0, "moe_grouped_sg_%s wtab vs direct %zu bitwise mismatches", fq[f], bad);
                printf("%-36s wtab == direct bitwise %s\n", fq[f], bad ? "FAIL" : "OK");
            }
        }

        // ================= 11. k_attn_head_f16_split + reduce vs k_attn_head_f16 =================
        id<MTLBuffer> bKvSplit = nil;
        uint32_t splitKvLen = 20001u;
        size_t splitRb = 0;
        {
            const uint32_t MAX_SP = 32u;
            const float scale = 1.0f / sqrtf((float)kDH);
            splitRb = psegs(1024u * kNREL * sizeof(float));
            for (uint32_t d = 0; d < 1024u * kNREL; d++)
                ((float *)(pk.data() + splitRb))[d] = R.f(-0.4f, 0.4f);
            const size_t kv_halfs = (size_t)splitKvLen * 2u * kKVD;
            bKvSplit = [rt.device() newBufferWithLength:kv_halfs * sizeof(uint16_t)
                                               options:MTLResourceStorageModeShared];
            CHECK(bKvSplit != nil, "bKvSplit alloc");
            uint16_t * kvp = (uint16_t *)bKvSplit.contents;
            for (size_t i = 0; i < kv_halfs; i++) kvp[i] = h16bits(R.f(-1.5f, 1.5f));

            const uint32_t sQ = seg(kNH * kDH), sRr = seg(kNH * kNREL), sSnk = seg(kNH);
            const uint32_t sPartO = seg(kNH * MAX_SP * kDH), sPartMS = seg(kNH * MAX_SP * 2u);
            const uint32_t sRef = seg(kNH * kDH), sGot = seg(kNH * kDH);
            for (uint32_t i = 0; i < kNH * kDH; i++) scr[sQ + i] = R.f(-1.2f, 1.2f);
            for (uint32_t i = 0; i < kNH * kNREL; i++) scr[sRr + i] = R.f(-0.5f, 0.5f);
            for (uint32_t i = 0; i < kNH; i++) scr[sSnk + i] = R.f(-1.0f, 1.0f);

            struct SplitCase { uint32_t pos, is_swa, n_swa, ring_cap, n_splits; const char * label; };
            const SplitCase cases[] = {
                {511u,   0u, 0u,    0u,    4u,  "attn_split full p=511"},
                {2048u,  0u, 0u,    0u,    16u, "attn_split full p=2048"},
                {20000u, 0u, 0u,    0u,    32u, "attn_split full p=20000"},
                {2048u,  1u, 1024u, 1025u, 9u,  "attn_split swa  p=2048"},
                {10u,    0u, 0u,    0u,    16u, "attn_split empty p=10"},
            };
            const uint64_t kvv_half_off = (uint64_t)splitKvLen * kKVD;
            for (const auto & tc : cases) {
                const float tau = tc.is_swa ? 1.0f
                    : (float)(std::log(std::max(1.0, ((double)tc.pos + 1.0) / 8192.0)) * 0.25 + 1.0);
                AttnF16Args ref_a{splitRb, 0ull, kvv_half_off, sQ, sRr, sSnk, sRef,
                                  tc.pos, tc.is_swa, tc.n_swa, tc.ring_cap, scale, tau};
                for (uint32_t i = 0; i < kNH * kDH; i++) {
                    scr[sRef + i] = -99.0f;
                    scr[sGot + i] = 99.0f;
                }
                for (uint32_t i = 0; i < kNH * MAX_SP * kDH; i++) scr[sPartO + i] = 77.0f;
                for (uint32_t i = 0; i < kNH * MAX_SP * 2u; i++) scr[sPartMS + i] = 77.0f;
                run("k_attn_head_f16", true, kNH, 1, 32, 1, &ref_a, sizeof(ref_a), bKvSplit);
                const std::vector<float> ref_out(scr.begin() + sRef, scr.begin() + sRef + kNH * kDH);

                AttnF16SplitArgs sp_a{splitRb, 0ull, kvv_half_off, sQ, sRr, sSnk, sGot,
                                      sPartO, sPartMS, tc.pos, tc.is_swa, tc.n_swa, tc.ring_cap,
                                      tc.n_splits, 0u, scale, tau};
                run("k_attn_head_f16_split", true, kNH, tc.n_splits, 32, 1, &sp_a, sizeof(sp_a), bKvSplit);
                run("k_attn_head_f16_reduce", true, kNH, 1, 32, 1, &sp_a, sizeof(sp_a));

                double mad = 0, maxabs = 0;
                const float * got = scr.data() + sGot;
                for (size_t i = 0; i < ref_out.size(); i++) {
                    mad = std::max(mad, std::fabs((double)ref_out[i] - got[i]));
                    maxabs = std::max(maxabs, std::fabs((double)ref_out[i]));
                }
                const double relscale = mad / std::max(1e-30, maxabs);
                const double cosv = cosine(ref_out.data(), got, ref_out.size());
                const bool strict_ok = (cosv >= 0.999999 && relscale < 1e-5);
                CHECK(strict_ok, "%s strict parity cos=%.8f relscale=%.3g", tc.label, cosv, relscale);
                printf("%-26s cos=%.8f relscale=%.3g mad=%.3g %s\n",
                       tc.label, cosv, relscale, mad, strict_ok ? "OK" : "FAIL");
            }
        }

        // ================= optional microbenchmark (--bench) =================
        if (do_bench) {
            cur = cur9; pcur = pcur9;
            const uint32_t k = 2304, nout = 2304, B = 32;
            const uint32_t nb = k / 32, qstr = (k / 4 + 15u) & ~15u, dstr = (nb + 3u) & ~3u;
            uint32_t sXq = seg(B * qstr + 64), sDx = seg(B * dstr + 64), sY = seg(B * nout + 64);
            for (uint32_t t = 0; t < B; t++) {
                std::vector<float> x(k);
                for (auto & v : x) v = R.f(-4, 4);
                std::vector<uint8_t> codes; std::vector<float> d;
                quantize(x, (int)k, codes, d);
                memcpy(scr.data() + sXq + t * qstr, codes.data(), k);
                for (uint32_t b = 0; b < nb; b++) scr[sDx + t * dstr + b] = d[b];
            }
            size_t w = psegs((size_t)nout * nb * 34);
            printf("bench weights: %.1f MB\n", (double)nout * nb * 34 / 1e6);
            fill_q8(pk.data() + w, nout, (int)nb, R);
            memcpy(bScr.contents, scr.data(), SCR_F * 4);
            memcpy(bPk.contents, pk.data(), PK_B);

            id<MTLComputePipelineState> pB = rt.pipeline("k_gemm_q8_batch");
            id<MTLComputePipelineState> pS = rt.pipeline("k_gemv_q8_k2304_m2");
            CHECK(pB != nil && pS != nil, "bench pipelines missing");
            GemmBatchArgs BA{w, sXq, sDx, sY, nout, k, B, qstr, dstr, nout, 0u};
            const int IT = 50;
            double t_batch = 0, t_seq = 0;
            auto timed = [&](const char * nm, const std::function<void(id<MTLComputeCommandEncoder>)> & one) {
                double best = 1e30;
                for (int pass = 0; pass < 3; pass++) {   // min of 3 passes
                    id<MTLCommandBuffer> c0 = [rt.queue() commandBuffer];
                    id<MTLComputeCommandEncoder> e0 = [c0 computeCommandEncoder];
                    one(e0);
                    [e0 endEncoding]; [c0 commit]; [c0 waitUntilCompleted];
                    id<MTLCommandBuffer> cmd = [rt.queue() commandBuffer];
                    id<MTLComputeCommandEncoder> en = [cmd computeCommandEncoder];
                    auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < IT; i++) one(en);
                    [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                    double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count() / IT;
                    CHECK(cmd.status == MTLCommandBufferStatusCompleted, "%s status", nm);
                    best = std::min(best, ms);
                }
                printf("BENCH %-30s %8.4f ms/iter\n", nm, best);
                return best;
            };
            if (pB) t_batch = timed("gemm_q8_batch B=32", [&](id<MTLComputeCommandEncoder> en) {
                [en setComputePipelineState:pB];
                [en setBuffer:bScr offset:0 atIndex:0];
                [en setBytes:&BA length:sizeof(BA) atIndex:1];
                [en setBuffer:bPk offset:0 atIndex:2];
                [en dispatchThreadgroups:MTLSizeMake((nout + 3) / 4, (B + 7) / 8, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            });
            if (pS) t_seq = timed("32x gemv_q8_k2304_m2", [&](id<MTLComputeCommandEncoder> en) {
                for (uint32_t t = 0; t < B; t++) {
                    GemvArgs SA{w, 0, sXq + t * qstr, sDx + t * dstr, sY + t * nout, 0, nout, 0, 0, 0};
                    [en setComputePipelineState:pS];
                    [en setBuffer:bScr offset:0 atIndex:0];
                    [en setBytes:&SA length:sizeof(SA) atIndex:1];
                    [en setBuffer:bPk offset:0 atIndex:2];
                    [en dispatchThreadgroups:MTLSizeMake((nout + 15) / 16, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            });
            // cross-check batch vs sequential numerics (relaxed: same data)
            std::vector<float> seqout((size_t)B * nout);
            memcpy(seqout.data(), (const float *)bScr.contents + sY, (size_t)B * nout * 4);   // seq ran last on bScr
            if (pB) {
                memset(scr.data() + sY, 0, (size_t)B * nout * 4);
                run("k_gemm_q8_batch", true, (nout + 3) / 4, (B + 7) / 8, 32, 4, &BA, sizeof(BA));
                gate("bench batch-vs-seq", seqout.data(), scr.data() + sY, seqout.size(), GATE_GEMM);
            }
            if (t_batch > 0 && t_seq > 0) {
                const double gbps = (double)nout * nb * 34 / (t_batch * 1e-3) / 1e9;
                printf("SPEEDUP gemm_q8_batch vs 32x gemv_m2: %.2fx  (batch %.1f GB/s weight stream)\n",
                       t_seq / t_batch, gbps);
            }
            // ---- k_gemm_q8_f32in_sg (FP32-in tensor-core gemm) vs the gemv path ----
            {
                id<MTLComputePipelineState> pSG = rt.pipeline("k_gemm_q8_f32in_sg");
                CHECK(pSG != nil, "bench sg pipeline missing");
                const uint32_t KMAX = 2304, NMAX = nout;
                uint32_t sXf = seg(B * KMAX + 64), sYg = seg(B * NMAX + 64);
                std::vector<float> xf((size_t)B * KMAX);
                for (auto & v : xf) v = R.f(-4, 4);
                memcpy(scr.data() + sXf, xf.data(), (size_t)B * KMAX * 4);
                memcpy(bScr.contents, scr.data(), SCR_F * 4);
                auto bench_sg = [&](uint32_t Bb) -> double {
                    GemmF32SgArgs A{w, sXf, KMAX, sYg, NMAX, k, nout, Bb, 0u, 0u};
                    return timed("k_gemm_q8_f32in_sg", [&](id<MTLComputeCommandEncoder> en) {
                        [en setComputePipelineState:pSG];
                        [en setBuffer:bScr offset:0 atIndex:0];
                        [en setBytes:&A length:sizeof(A) atIndex:1];
                        [en setBuffer:bPk offset:0 atIndex:2];
                        [en dispatchThreadgroups:MTLSizeMake((nout + 31) / 32, (Bb + 31) / 32, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                    });
                };
                const float * gout = (const float *)bScr.contents + sYg;   // timed() left B=16 last
                for (uint32_t Bb : {32u, 16u}) {
                    const double ms = bench_sg(Bb);
                    const double mvb = Bb * (double)nout * k * 2 / 1e6;
                    const double sg_vs_gemv = (t_seq * ((double)Bb / B)) / ms;
                    printf("B=%-3u k_gemm_q8_f32in_sg: %8.4f ms/iter  %8.1f Gflop/s "
                           "(vs %ux gemv_m2: %.2fx, vs gemm_q8_batch@32: %.2fx)\n",
                           Bb, ms, mvb / ms, Bb, sg_vs_gemv,
                           Bb == 32 ? t_batch / ms : 0.0);
                    // parity vs FP64 oracle on the buffer bench left (B=16 run)
                    if (Bb == 16) {
                        std::vector<float> ref, got;
                        for (uint32_t t = 0; t < 4; t++)
                            for (uint32_t r = 0; r < 256; r++) {
                                const uint8_t * wr = pk.data() + w + (size_t)r * nb * 34;
                                double a = 0;
                                for (uint32_t bi = 0; bi < nb; bi++) {
                                    const uint8_t * wb = wr + (size_t)bi * 34;
                                    const double d = h16d((uint16_t)(wb[0] | (wb[1] << 8)));
                                    for (int i = 0; i < 32; i++)
                                        a += d * (double)sb(wb[2 + i]) *
                                             (double)xf[(size_t)t * KMAX + (size_t)bi * 32 + i];
                                }
                                ref.push_back((float)a);
                                got.push_back(gout[(size_t)t * NMAX + r]);
                            }
                        gate("bench sg-vs-fp64 oracle", ref.data(), got.data(), ref.size(), GATE_SG);
                    }
                }
            }
            // ---- k_moe_grouped_f32in_sg: Q8_0 (3C) vs Q4_K/Q5_0/Q5_1 (4B) + vs k_moe_grouped_gemm_q8 ----
            {
                const uint32_t NEXP = 128, NSLOTS = 16, TOPK = 8, k = 2304, nout = 1152, nb = k / 32;
                struct QFmt { const char * tag; const char * fn; uint32_t bb; } fmts[] = {
                    {"Q8_0", "k_moe_grouped_f32in_sg",      34u},
                    {"Q4_K", "k_moe_grouped_f32in_sg_q4k",  18u},
                    {"Q5_0", "k_moe_grouped_f32in_sg_q5_0", 22u},
                    {"Q5_1", "k_moe_grouped_f32in_sg_q5_1", 24u},
                };
                float v4[256], v5[32];
                double ms_sg[4][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
                double t_mg[2] = {0.0, 0.0};
                for (int fi = 0; fi < 4; fi++) {
                    cur = cur9; pcur = pcur9;
                    const uint64_t exp_stride = (uint64_t)nout * nb * fmts[fi].bb;
                    size_t wbase = psegs((size_t)NSLOTS * exp_stride);
                    for (uint32_t s = 0; s < NSLOTS; s++) {
                        uint8_t * wp = pk.data() + wbase + (size_t)s * exp_stride;
                        if (fi == 0) {
                            fill_q8(wp, nout, (int)nb, R);
                        } else if (fi == 1) {
                            for (uint32_t r = 0; r < nout; r++)
                                for (uint32_t b = 0; b < nb; b += 8u) {
                                    for (int i = 0; i < 256; i++) v4[i] = R.f(-0.4f, 0.4f);
                                    fill_q4k(wp + (size_t)r * nb * 18 + (size_t)b * 18, v4,
                                             ((r + b) & 1) ? R.f(-0.0003f, 0.0003f) : 0.0f, R);
                                }
                        } else {
                            for (uint32_t r = 0; r < nout; r++)
                                for (uint32_t b = 0; b < nb; b++) {
                                    for (int i = 0; i < 32; i++) v5[i] = R.f(-0.4f, 0.4f);
                                    fill_q5(wp + (size_t)r * nb * fmts[fi].bb + (size_t)b * fmts[fi].bb,
                                            v5, fi == 3);
                                }
                        }
                    }
                    id<MTLComputePipelineState> pSG = rt.pipeline(fmts[fi].fn);
                    id<MTLComputePipelineState> pMG = fi == 0 ? rt.pipeline("k_moe_grouped_gemm_q8") : nil;
                    CHECK(pSG != nil && (fi != 0 || pMG != nil), "bench moe pipelines missing");
                    int bi = 0;
                    for (uint32_t Bm : {32u, 128u}) {
                        cur = (uint32_t)(cur9 + 0);   // rewind SCR per Bm
                        const uint32_t npairs = Bm * TOPK, qstr = k / 4 + 16, dstr = nb + 4;
                        uint32_t sXq = seg(Bm * qstr + 32), sDx = seg(Bm * dstr + 32), sXf = seg(Bm * k + 32);
                        uint32_t sY = seg(npairs * nout + 32);
                        uint32_t sSel = seg(npairs + 16), sCnt = seg(NEXP + 16), sOff = seg(NEXP + 16),
                                 sList = seg(npairs + 16), sWtab = seg(NEXP + 16);
                        for (uint32_t t = 0; t < Bm; t++) {
                            std::vector<float> x(k);
                            for (auto & v : x) v = R.f(-4, 4);
                            memcpy(scr.data() + sXf + (size_t)t * k, x.data(), k * 4);
                            std::vector<uint8_t> codes; std::vector<float> d;
                            quantize(x, (int)k, codes, d);
                            memcpy(scr.data() + sXq + (size_t)t * qstr, codes.data(), k);
                            for (uint32_t b = 0; b < nb; b++) scr[sDx + (size_t)t * dstr + b] = d[b];
                            for (uint32_t k0 = 0; k0 < TOPK; k0++)
                                ((uint32_t *)scr.data())[sSel + t * TOPK + k0] = (k0 * 13u + t * 5u) % NEXP;
                        }
                        for (uint32_t e = 0; e < NEXP; e++)
                            ((uint32_t *)scr.data())[sWtab + e] = e % NSLOTS;
                        MoeBucketArgs BA{sSel, sCnt, sOff, sList, NEXP, npairs, 0u, 0u};
                        run("k_moe_bucket_build", true, 1, 1, 256, 1, &BA, sizeof(BA));
                        memcpy(bScr.contents, scr.data(), SCR_F * 4);
                        memcpy(bPk.contents, pk.data(), PK_B);

                        if (fi == 0) {
                            MoeGroupedGemmArgs MGA{wbase, exp_stride, 0ull, sXq, sDx, sY, sList, sCnt, sOff,
                                                   nout, k, qstr, dstr, nout, NEXP, TOPK, 0u, sWtab, 0u};
                            char nm1[48];
                            snprintf(nm1, sizeof(nm1), "moe_grouped_q8 B=%u", Bm);
                            t_mg[bi] = timed(nm1, [&](id<MTLComputeCommandEncoder> en) {
                                [en setComputePipelineState:pMG];
                                [en setBuffer:bScr offset:0 atIndex:0];
                                [en setBytes:&MGA length:sizeof(MGA) atIndex:1];
                                [en setBuffer:bPk offset:0 atIndex:2];
                                [en dispatchThreadgroups:MTLSizeMake((nout + 3) / 4, NEXP, 1)
                                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                            });
                        }
                        MoeGroupedF32SgArgs SGA{wbase, exp_stride, 0ull, sXf, sY, sList, sCnt, sOff, sWtab,
                                                nout, k, k, nout, NEXP, TOPK, 0u, 0u};
                        char nm2[64];
                        snprintf(nm2, sizeof(nm2), "moe_grouped_f32_sg %s B=%u", fmts[fi].tag, Bm);
                        ms_sg[fi][bi] = timed(nm2, [&](id<MTLComputeCommandEncoder> en) {
                            [en setComputePipelineState:pSG];
                            [en setBuffer:bScr offset:0 atIndex:0];
                            [en setBytes:&SGA length:sizeof(SGA) atIndex:1];
                            [en setBuffer:bPk offset:0 atIndex:2];
                            [en dispatchThreadgroups:MTLSizeMake((nout + 31) / 32, NEXP, 1)
                                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                        });
                        bi++;
                    }
                }
                for (int bi = 0; bi < 2; bi++) {
                    const uint32_t Bm = bi ? 128u : 32u;
                    printf("MOE-SG weight-format bench B=%u (NEXP=128 slots=16 topk=8 k=2304 nout=1152):\n", Bm);
                    for (int fi = 0; fi < 4; fi++) {
                        const double gbs = (double)NEXP * nout * nb * fmts[fi].bb / (ms_sg[fi][bi] * 1e-3) / 1e9;
                        printf("  %-5s %8.4f ms/iter  %6.2fx vs Q8_0  logical wstream %6.1f GB/s\n",
                               fmts[fi].tag, ms_sg[fi][bi], ms_sg[0][bi] / ms_sg[fi][bi], gbs);
                    }
                    printf("  Q8_0 per-token vs k_moe_grouped_gemm_q8: %.2fx (%.3f vs %.3f ms/tok)\n",
                           t_mg[bi] / ms_sg[0][bi], ms_sg[0][bi] / Bm, t_mg[bi] / Bm);
                }
            }
            // ---- k_attn_head_f16_split + reduce vs k_attn_head_f16 @ pos=20000 ----
            {
                cur = cur9; pcur = pcur9;
                const uint32_t MAX_SP = 32u, pos_b = 20000u;
                const float scale = 1.0f / sqrtf((float)kDH);
                const float tau = (float)(std::log(((double)pos_b + 1.0) / 8192.0) * 0.25 + 1.0);
                const size_t rbBench = psegs(1024u * kNREL * sizeof(float));
                for (uint32_t d = 0; d < 1024u * kNREL; d++)
                    ((float *)(pk.data() + rbBench))[d] = R.f(-0.4f, 0.4f);
                const uint32_t sQ = seg(kNH * kDH), sRr = seg(kNH * kNREL), sSnk = seg(kNH);
                const uint32_t sPartO = seg(kNH * MAX_SP * kDH), sPartMS = seg(kNH * MAX_SP * 2u);
                const uint32_t sOut = seg(kNH * kDH);
                for (uint32_t i = 0; i < kNH * kDH; i++) scr[sQ + i] = R.f(-1.2f, 1.2f);
                for (uint32_t i = 0; i < kNH * kNREL; i++) scr[sRr + i] = R.f(-0.5f, 0.5f);
                for (uint32_t i = 0; i < kNH; i++) scr[sSnk + i] = R.f(-1.0f, 1.0f);
                memcpy(bScr.contents, scr.data(), SCR_F * 4);
                memcpy(bPk.contents, pk.data(), PK_B);

                id<MTLComputePipelineState> pA1 = rt.pipeline("k_attn_head_f16");
                id<MTLComputePipelineState> pAS = rt.pipeline("k_attn_head_f16_split");
                id<MTLComputePipelineState> pAR = rt.pipeline("k_attn_head_f16_reduce");
                CHECK(pA1 != nil && pAS != nil && pAR != nil, "bench attn split pipelines missing");
                const uint64_t kvv_half_off = (uint64_t)splitKvLen * kKVD;
                AttnF16Args a1{rbBench, 0ull, kvv_half_off, sQ, sRr, sSnk, sOut,
                               pos_b, 0u, 0u, 0u, scale, tau};
                AttnF16SplitArgs as{rbBench, 0ull, kvv_half_off, sQ, sRr, sSnk, sOut,
                                    sPartO, sPartMS, pos_b, 0u, 0u, 0u, MAX_SP, 0u, scale, tau};
                double t_serial = timed("attn_f16 pos=20000", [&](id<MTLComputeCommandEncoder> en) {
                    [en setComputePipelineState:pA1];
                    [en setBuffer:bScr offset:0 atIndex:0];
                    [en setBytes:&a1 length:sizeof(a1) atIndex:1];
                    [en setBuffer:bPk offset:0 atIndex:2];
                    [en setBuffer:bKvSplit offset:0 atIndex:3];
                    [en dispatchThreadgroups:MTLSizeMake(kNH, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                });
                double t_split = timed("attn_f16_split32 pos=20000", [&](id<MTLComputeCommandEncoder> en) {
                    [en setComputePipelineState:pAS];
                    [en setBuffer:bScr offset:0 atIndex:0];
                    [en setBytes:&as length:sizeof(as) atIndex:1];
                    [en setBuffer:bPk offset:0 atIndex:2];
                    [en setBuffer:bKvSplit offset:0 atIndex:3];
                    [en dispatchThreadgroups:MTLSizeMake(kNH, MAX_SP, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [en setComputePipelineState:pAR];
                    [en setBuffer:bScr offset:0 atIndex:0];
                    [en setBytes:&as length:sizeof(as) atIndex:1];
                    [en dispatchThreadgroups:MTLSizeMake(kNH, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                });
                printf("SPEEDUP attn_f16_split32@20000: %.2fx (%.4f vs %.4f ms/iter)\n",
                       t_serial / t_split, t_split, t_serial);
            }
            // ---- k_attn_f16_flash_prefill vs k_attn_head_f16_batch @ pos=20000 ----
            {
                cur = cur9; pcur = pcur9;
                const uint32_t POSB = 20000u, KB = 128u;
                const float scale = 1.0f / sqrtf((float)kDH);
                const size_t rbB = psegs(1024u * kNREL * sizeof(float));
                for (uint32_t d = 0; d < 1024u * kNREL; d++)
                    ((float *)(pk.data() + rbB))[d] = R.f(-0.4f, 0.4f);
                const uint32_t sQ = seg(KB * kNH * kDH), sRr = seg(KB * kNH * kNREL),
                                 sSnk = seg(KB * kNH), sOut = seg(KB * kNH * kDH);
                for (uint32_t i = 0; i < KB * kNH * kDH; i++) scr[sQ + i] = R.f(-1.2f, 1.2f);
                for (uint32_t i = 0; i < KB * kNH * kNREL; i++) scr[sRr + i] = R.f(-0.5f, 0.5f);
                for (uint32_t i = 0; i < KB * kNH; i++) scr[sSnk + i] = R.f(-1.0f, 1.0f);
                const size_t kvh = (size_t)(POSB + KB + 8u) * kKVD;   // 20136 slots
                id<MTLBuffer> bKvB = [rt.device() newBufferWithLength:kvh * 2 * sizeof(uint16_t)
                                                              options:MTLResourceStorageModeShared];
                CHECK(bKvB != nil, "bench flash KV alloc");
                if (bKvB) {
                    uint16_t * dev = (uint16_t *)bKvB.contents;
                    for (size_t i = 0; i < kvh * 2u; i++) dev[i] = h16bits(R.f(-1.5f, 1.5f));
                }
                memcpy(bScr.contents, scr.data(), SCR_F * 4);
                memcpy(bPk.contents, pk.data(), PK_B);
                id<MTLComputePipelineState> pFB = rt.pipeline("k_attn_head_f16_batch");
                id<MTLComputePipelineState> pFF = rt.pipeline("k_attn_f16_flash_prefill");
                CHECK(pFB != nil && pFF != nil && bKvB != nil, "bench flash pipelines missing");
                if (pFB && pFF && bKvB) {
                    double t_b[2][2] = {{0, 0}, {0, 0}}, t_f[2][2] = {{0, 0}, {0, 0}};  // [swa][B]
                    int si = 0;
                    for (uint32_t is_swa : {0u, 1u}) {
                        int bi = 0;
                        for (uint32_t Bb : {128u, 32u}) {
                            AttnBatchF16Args A{};
                            A.relbiaspack = rbB; A.kvkoff = 0; A.kvvoff = (uint64_t)kvh;
                            A.qoff = sQ; A.rout = sRr; A.sinkoff = sSnk; A.outoff = sOut;
                            A.q_stride = kNH * kDH; A.rout_stride = kNH * kNREL;
                            A.sink_stride = kNH; A.out_stride = kNH * kDH;
                            A.pos0 = POSB; A.B = Bb; A.is_swa = is_swa; A.n_swa = 1024u;
                            A.ring_cap = 1025u; A.pad0 = 0;
                            A.scale = scale; A.log_scale_floor = 8192.0f;
                            A.log_scale_alpha = 0.25f; A.pad1 = 0;
                            char nm[64];
                            snprintf(nm, sizeof(nm), "attn_batch p=%u B=%u swa=%u", POSB, Bb, is_swa);
                            t_b[si][bi] = timed(nm, [&](id<MTLComputeCommandEncoder> en) {
                                [en setComputePipelineState:pFB];
                                [en setBuffer:bScr offset:0 atIndex:0];
                                [en setBytes:&A length:sizeof(A) atIndex:1];
                                [en setBuffer:bPk offset:0 atIndex:2];
                                [en setBuffer:bKvB offset:0 atIndex:3];
                                [en dispatchThreadgroups:MTLSizeMake(kNH, Bb, 1)
                                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                            });
                            snprintf(nm, sizeof(nm), "attn_flash p=%u B=%u swa=%u", POSB, Bb, is_swa);
                            t_f[si][bi] = timed(nm, [&](id<MTLComputeCommandEncoder> en) {
                                [en setComputePipelineState:pFF];
                                [en setBuffer:bScr offset:0 atIndex:0];
                                [en setBytes:&A length:sizeof(A) atIndex:1];
                                [en setBuffer:bPk offset:0 atIndex:2];
                                [en setBuffer:bKvB offset:0 atIndex:3];
                                [en dispatchThreadgroups:MTLSizeMake(kNH, (Bb + 31u) / 32u, 1)
                                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                            });
                            // post-timing parity cross-check of the two kernels
                            const size_t n = (size_t)Bb * kNH * kDH;
                            std::vector<float> refb(n);
                            run("k_attn_head_f16_batch", true, kNH, Bb, 32, 1, &A, sizeof(A), bKvB);
                            memcpy(refb.data(), scr.data() + sOut, n * 4);
                            std::fill(scr.begin() + sOut, scr.begin() + sOut + n, -9999.0f);
                            run("k_attn_f16_flash_prefill", true, kNH, (Bb + 31u) / 32u, 32, 4,
                                &A, sizeof(A), bKvB);
                            char gn[64];
                            snprintf(gn, sizeof(gn), "bench flash B=%u swa=%u", Bb, is_swa);
                            gate(gn, refb.data(), scr.data() + sOut, n, GATE_SG);
                            bi++;
                        }
                        si++;
                    }
                    for (int s2 = 0; s2 < 2; s2++)
                        for (int b2 = 0; b2 < 2; b2++)
                            printf("SPEEDUP attn_flash vs batch p=%u B=%u is_swa=%u: %.2fx (%.4f vs %.4f ms/iter)\n",
                                   POSB, b2 ? 32u : 128u, (uint32_t)s2,
                                   t_b[s2][b2] / t_f[s2][b2], t_f[s2][b2], t_b[s2][b2]);
                }
                bKvB = nil;
            }
        }

        printf("test_prefill_kernels: %s\n", g_fail ? "FAILED" : "ALL PASS");
        return g_fail ? 1 : 0;
    }
}
