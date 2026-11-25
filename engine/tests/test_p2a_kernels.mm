// test_p2a_kernels.mm — Phase-2A forward kernel CPU-vs-Metal parity.
// Mirrors forward_metal.mm launch configs exactly (SCR float offsets + PACK
// byte offsets; d_tg threadgroup dispatch / d_th flat dispatch).
// Exactness policy: integer-exact paths (gather/xfer/scale/quant codes,
// conv arithmetic with fp-contract off) are BITWISE; reduction/activation
// kernels use rel-tol + cosine gates (AGENTS.md: cos > 0.999 / |diff| < 1e-3).
// All buffers < 10 MB; runs in ~2 s alongside other GPU workloads.
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
#include <tuple>
#include <vector>

#include "metal_runtime.hpp"
#include "../../engine/kernels/m64_p2a_args.h"

using namespace m64;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static constexpr uint32_t kNE = 2304, kNH = 24, kDH = 128, kKVD = 1024, kNREL = 16;

struct Rng {
    std::mt19937 g{0x5A2C1137u};
    uint32_t u32() { return g(); }
    float f(float a = -2.f, float b = 2.f) { return std::uniform_real_distribution<float>(a, b)(g); }
};

static inline float f16le(const uint8_t * p) { return (float)std::bit_cast<__fp16>((uint16_t)(p[0] | (p[1] << 8))); }
static inline float r16(float x) { return (float)(__fp16)x; }
static inline int sb(uint8_t b) { return (int)(int8_t)b; }
static inline float ksig(float x) { return 1.0f / (1.0f + expf(-x)); }

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

// q8 gemv oracle: int32-exact per 32-block dots, ascending-unit fmaf (matches
// gemv_unit() in m64_p2a.metal; cross-lane simd tree differs by ulps only).
static float gemv_ref(const uint8_t * wr, const int8_t * xq, const float * dx, int nu) {
    float a = 0.0f;
    for (int u = 0; u < nu; u++) {
        const uint8_t * wb = wr + u * 68;
        int s0 = 0, s1 = 0;
        for (int k = 0; k < 32; k++) { s0 += sb(wb[2 + k]) * xq[u * 64 + k]; s1 += sb(wb[36 + k]) * xq[u * 64 + 32 + k]; }
        a = std::fma((float)s0, dx[2 * u] * f16le(wb), a);
        a = std::fma((float)s1, dx[2 * u + 1] * f16le(wb + 34), a);
    }
    return a;
}

// f64 GEMV oracle + accumulation magnitude (ulp-based gate for garbage Q8 data:
// f32 simd-tree vs scalar order noise is ~eps * sum|terms| when a row cancels).
static std::pair<double, double> gemv_ref64(const uint8_t * wr, const int8_t * xq, const float * dx, int nu) {
    double a = 0.0, sumabs = 0.0;
    for (int u = 0; u < nu; u++) {
        const uint8_t * wb = wr + u * 68;
        int s0 = 0, s1 = 0;
        for (int k = 0; k < 32; k++) { s0 += sb(wb[2 + k]) * xq[u * 64 + k]; s1 += sb(wb[36 + k]) * xq[u * 64 + 32 + k]; }
        double t0 = (double)s0 * ((double)dx[2 * u] * (double)f16le(wb));
        double t1 = (double)s1 * ((double)dx[2 * u + 1] * (double)f16le(wb + 34));
        a += t0; sumabs += std::fabs(t0); a += t1; sumabs += std::fabs(t1);
    }
    return {a, sumabs};
}

// ---------------- Phase-2A-opt microbenchmark (--bench) ----------------
// Synthetic small-buffer decode-loop pieces (total GPU buffers < 30 MB, safe
// alongside Splash). Measures per-iteration GPU time of old vs optimized
// kernel chains. NOT a parity gate — parity is the tests above.
struct BenchCtx {
    uint32_t sX, sD, sW1, sY, sW2, sW2b;
    uint32_t sG, sU, sHQ, sDH, sAcc, sSel, sWt;
    uint32_t sQ, sGate2;
};

static void run_bench(m64::MetalRuntime & rt) {
    using namespace m64;
    const uint32_t BE = 192;                    // bench expert nout
    // 100 MiB: 26 MiB legacy Q8 pack + 27-82 MiB Milestone 7A f32-in regions.
    // Still trivial next to Splash's resident pool on the shared 64 GB host.
    const size_t BPK = (size_t)100 * 1024 * 1024;
    std::mt19937 g2(0xBEEF1234u);
    std::uniform_real_distribution<float> uf(-0.02f, 0.02f);
    std::vector<uint8_t> bp(BPK, 0);
    for (size_t i = 0; i < BPK; i += 4) *(uint32_t *)(bp.data() + i) = g2();
    for (size_t o = 0; o + 68 <= BPK; o += 68) { bp[o + 1] &= 0x3B; bp[o + 35] &= 0x3B; }
    std::vector<float> bs(1u << 20, 0.f);
    for (auto & v : bs) v = uf(g2);
    id<MTLBuffer> sBuf = [rt.device() newBufferWithLength:bs.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> pBuf = [rt.device() newBufferWithLength:BPK options:MTLResourceStorageModeShared];
    memcpy(sBuf.contents, bs.data(), bs.size() * 4);
    memcpy(pBuf.contents, bp.data(), BPK);

    // scratch float layout
    uint32_t c = 16;
    auto segf = [&](uint32_t n) { c = (c + 3u) & ~3u; uint32_t r = c; c += n + 4u; return r; };
    BenchCtx B;
    B.sX = segf(2304); B.sD = segf(128);
    B.sY = segf(4096); B.sQ = segf(4096); B.sGate2 = segf(8 * BE);
    B.sU = segf(8 * BE); B.sHQ = segf(8 * 288); B.sDH = segf(512);
    B.sAcc = segf(2304); B.sSel = segf(16); B.sWt = segf(16);
    uint32_t sStn = segf(8 * 2304), sMixOut = segf(4 * 2304), sMixSc = segf(4 * M64_MAXS);
    ((uint32_t *)bs.data())[B.sSel + 0] = 0; ((uint32_t *)bs.data())[B.sSel + 1] = 1;
    ((uint32_t *)bs.data())[B.sSel + 2] = 2; ((uint32_t *)bs.data())[B.sSel + 3] = 3;
    ((uint32_t *)bs.data())[B.sSel + 4] = 4; ((uint32_t *)bs.data())[B.sSel + 5] = 5;
    ((uint32_t *)bs.data())[B.sSel + 6] = 6; ((uint32_t *)bs.data())[B.sSel + 7] = 7;
    // pack layout (all offsets multiples of 68B)
    auto PO = [&](uint64_t units) { return units * 68ull; };
    const uint64_t R2448 = PO(36), R1224 = PO(18);
    const uint64_t pW1 = PO(60000);                                   // 1024 rows k2304
    const uint64_t pW2 = pW1 + 1024 * R2448;                          // 4096 rows k2304 (lm_head-ish)
    const uint64_t pEX = pW2 + 4096 * R2448;                          // 8 experts
    const uint64_t EXPS = 2u * BE * R2448 + BE * R1224;
    const uint64_t pNormW = pEX + 8 * EXPS + 64;
    uint64_t used = pNormW + (size_t)2304 * 4;
    printf("bench pack used: %.1f MB\n", used / 1e6);
    for (size_t i = 0; i < 2304; i++) ((float *)(bp.data() + pNormW))[i] = 1.0f;

    auto setl = [&](id<MTLComputeCommandEncoder> en, const char * fn, const void * a, size_t sz) {
        id<MTLComputePipelineState> p = rt.pipeline(fn);
        if (!p) { fprintf(stderr, "bench: missing kernel %s\n", fn); return; }
        [en setComputePipelineState:p];
        [en setBuffer:sBuf offset:0 atIndex:0];
        [en setBytes:a length:sz atIndex:1];
        [en setBuffer:pBuf offset:0 atIndex:2];
    };
    auto bench = [&](const char * name, int iters,
                     const std::function<void(id<MTLComputeCommandEncoder>)> & one) {
        // warmup
        {
            id<MTLCommandBuffer> c0 = [rt.queue() commandBuffer];
            id<MTLComputeCommandEncoder> e0 = [c0 computeCommandEncoder];
            one(e0); one(e0);
            [e0 endEncoding]; [c0 commit]; [c0 waitUntilCompleted];
        }
        id<MTLCommandBuffer> cmd = [rt.queue() commandBuffer];
        id<MTLComputeCommandEncoder> en = [cmd computeCommandEncoder];
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) one(en);
        [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("BENCH %-34s %8.4f ms/iter %s\n", name, ms / iters,
               cmd.status == MTLCommandBufferStatusCompleted ? "" : "GPU-FAULT");
        if (cmd.status != MTLCommandBufferStatusCompleted)
            printf("  %s\n", cmd.error.localizedDescription.UTF8String);
    };
    const int IT = 100;

    // ---- 1. plain warp-per-row k2304 GEMVs ----
    {
        GemvArgs A{pW1, 0, B.sX, B.sD, B.sY, 0, 1024, 0, 0, 0};
        bench("gemv_k2304_1024", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k2304", &A, sizeof(A));
            [en dispatchThreadgroups:MTLSizeMake((1024 + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        GemvArgs A2{pW2, 0, B.sX, B.sD, B.sY, 0, 4096, 0, 0, 0};
        bench("gemv_k2304_4096_lmhead", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k2304", &A2, sizeof(A2));
            [en dispatchThreadgroups:MTLSizeMake((4096 + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
    }

    // ---- 2. MoE: unfused 25-dispatch chain (zero + 8 x 3) ----
    {
        ScaleArgs za{B.sAcc, 2304, 0.0f};
        bench("moe_unfused_8e(25disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_scale", &za, sizeof(za));
            [en dispatchThreads:MTLSizeMake(2304, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            for (uint32_t e = 0; e < 8; e++) {
                uint64_t base = pEX + e * EXPS;
                GemvArgs gu{base, base + BE * R2448, B.sX, B.sD, B.sGate2, B.sU, BE, e, (uint32_t)EXPS, B.sSel};
                setl(en, "k_gemv_gateup", &gu, sizeof(gu));
                [en dispatchThreadgroups:MTLSizeMake((2 * BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                SituArgs sa{B.sGate2, B.sU, B.sHQ, B.sDH, BE / 32, 4.0f, 25.0f, 0.25f, 0.04f};
                setl(en, "k_situ_quant", &sa, sizeof(sa));
                [en dispatchThreads:MTLSizeMake(BE / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                MoeDownArgs md{base + 2 * BE * R2448, B.sHQ, B.sDH, B.sAcc, B.sWt, e, BE, (uint32_t)EXPS, B.sSel};
                setl(en, "k_gemv_downacc", &md, sizeof(md));
                [en dispatchThreadgroups:MTLSizeMake((BE + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        });
    }

    // ---- 3. rms + quant chains ----
    {
        RMSArgs A{}; A.k = 2304; A.rows = 1; A.hasw = 1; A.eps = 1e-6f;
        A.xr[0] = B.sX; A.wr[0] = pNormW; A.yr[0] = B.sQ;
        QuantArgs Q{B.sQ, B.sGate2, B.sDH, 72};
        bench("rms1+quant(2disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_rms", &A, sizeof(A));
            [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            setl(en, "k_quant", &Q, sizeof(Q));
            [en dispatchThreads:MTLSizeMake(72, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        });
        GateMulArgs G{B.sX, B.sQ, 3072};
        QuantArgs Q2{B.sX, B.sGate2, B.sDH, 96};
        bench("gate_mul+quant(2disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gate_mul", &G, sizeof(G));
            [en dispatchThreads:MTLSizeMake(3072, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            setl(en, "k_quant", &Q2, sizeof(Q2));
            [en dispatchThreads:MTLSizeMake(96, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        });
    }

    // ---- 4. multi-row k2304 GEMV (m2 / m4) vs warp-per-row ----
    {
        GemvArgs A1{pW1, 0, B.sX, B.sD, B.sY, 0, 1024, 0, 0, 0};
        GemvArgs A2{pW2, 0, B.sX, B.sD, B.sY, 0, 4096, 0, 0, 0};
        bench("gemv_k2304_m2_1024", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k2304_m2", &A1, sizeof(A1));
            [en dispatchThreadgroups:MTLSizeMake((1024 + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("gemv_k2304_m4_1024", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k2304_m4", &A1, sizeof(A1));
            [en dispatchThreadgroups:MTLSizeMake((1024 + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("gemv_k2304_m2_4096_lmhead", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k2304_m2", &A2, sizeof(A2));
            [en dispatchThreadgroups:MTLSizeMake((4096 + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("gemv_k2304_m4_4096_lmhead", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k2304_m4", &A2, sizeof(A2));
            [en dispatchThreadgroups:MTLSizeMake((4096 + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
    }

    // ---- 5. MoE batch-8: 3-dispatch chain vs 25-dispatch unfused ----
    {
        GemvArgs gu{pEX, pEX + BE * R2448, B.sX, B.sD, B.sGate2, B.sU, BE, 0, (uint32_t)EXPS, B.sSel};
        SituArgs sa{B.sGate2, B.sU, B.sHQ, B.sDH, 8 * BE / 32, 4.0f, 25.0f, 0.25f, 0.04f};
        MoeDownArgs md{pEX + 2 * BE * R2448, B.sHQ, B.sDH, B.sAcc, B.sWt, 0, BE, (uint32_t)EXPS, B.sSel};
        bench("moe_batch8(3disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8", &gu, sizeof(gu));
            [en dispatchThreadgroups:MTLSizeMake((8 * 2 * BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            setl(en, "k_situ_quant8", &sa, sizeof(sa));
            [en dispatchThreads:MTLSizeMake(8 * BE, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            setl(en, "k_gemv_down_reduce8", &md, sizeof(md));
            [en dispatchThreadgroups:MTLSizeMake((BE + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
    }

    // ---- 6. fused rms_quant / gate_mul_quant (1 disp vs 2) ----
    {
        RmsQuantArgs RQ{}; RQ.k = 2304; RQ.rows = 1; RQ.hasw = 1; RQ.eps = 1e-6f;
        RQ.xr[0] = B.sX; RQ.wr[0] = pNormW; RQ.yr[0] = B.sQ; RQ.q[0] = B.sGate2; RQ.d[0] = B.sDH;
        bench("rms_quant(1disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_rms_quant", &RQ, sizeof(RQ));
            [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        GateMulQuantArgs GQ{B.sX, B.sQ, B.sGate2, B.sDH, 96};
        bench("gate_mul_quant(1disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gate_mul_quant", &GQ, sizeof(GQ));
            [en dispatchThreads:MTLSizeMake(96 * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });
    }

    // ---- 7. Round-2 A/B: k3072_m2 vs m1, gateup_batch8_m2 vs m1, down_reduce8_m4 vs m2 ----
    {
        GemvArgs K3{pW2, 0, B.sX, B.sD, B.sY, 0, 2304, 0, 0, 0};
        bench("gemv_k3072_m1_2304", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k3072", &K3, sizeof(K3));
            [en dispatchThreadgroups:MTLSizeMake((2304 + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("gemv_k3072_m2_2304", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k3072_m2", &K3, sizeof(K3));
            [en dispatchThreadgroups:MTLSizeMake((2304 + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });

        GemvArgs gu{pEX, pEX + BE * R2448, B.sX, B.sD, B.sGate2, B.sU, BE, 0, (uint32_t)EXPS, B.sSel};
        bench("gateup_batch8_m1(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8", &gu, sizeof(gu));
            [en dispatchThreadgroups:MTLSizeMake((8 * 2 * BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("gateup_batch8_m2(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8_m2", &gu, sizeof(gu));
            [en dispatchThreadgroups:MTLSizeMake((8 * BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });

        MoeDownArgs md{pW2, B.sHQ, B.sDH, B.sAcc, B.sWt, 0, 768, (uint32_t)(768 * R1224), B.sSel};
        bench("down_reduce8_m2(768)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8", &md, sizeof(md));
            [en dispatchThreadgroups:MTLSizeMake((768 + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("down_reduce8_m4(768)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_m4", &md, sizeof(md));
            [en dispatchThreadgroups:MTLSizeMake((768 + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });

        SituArgs sa{B.sGate2, B.sU, B.sHQ, B.sDH, 8 * BE / 32, 4.0f, 25.0f, 0.25f, 0.04f};
        MoeDownArgs md_e{pEX + 2 * BE * R2448, B.sHQ, B.sDH, B.sAcc, B.sWt, 0, BE, (uint32_t)EXPS, B.sSel};
        bench("moe_batch8_r2(m2+m4)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8_m2", &gu, sizeof(gu));
            [en dispatchThreadgroups:MTLSizeMake((8 * BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            setl(en, "k_situ_quant8", &sa, sizeof(sa));
            [en dispatchThreads:MTLSizeMake(8 * BE, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            setl(en, "k_gemv_down_reduce8_m4", &md_e, sizeof(md_e));
            [en dispatchThreadgroups:MTLSizeMake((BE + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });

        // ---- 8. Round-3 A/B: multi_m2 fusion, 2-stage parallel MoE down+conv, fused mix ----
        bench("down_batch8_m2(768)", IT, [&](id<MTLComputeCommandEncoder> en) {
            MoeDownArgs md2{pW2, B.sHQ, B.sDH, B.sY, B.sWt, 0, 768, (uint32_t)(768 * R1224), B.sSel};
            setl(en, "k_gemv_down_batch8_m2", &md2, sizeof(md2));
            [en dispatchThreadgroups:MTLSizeMake((768 + 15) / 16, 8, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("down_batch8_m2+conv(768)", IT, [&](id<MTLComputeCommandEncoder> en) {
            MoeDownArgs md2{pW2, B.sHQ, B.sDH, B.sY, B.sWt, 0, 768, (uint32_t)(768 * R1224), B.sSel};
            setl(en, "k_gemv_down_batch8_m2", &md2, sizeof(md2));
            [en dispatchThreadgroups:MTLSizeMake((768 + 15) / 16, 8, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            DownReduceConvArgs drc{B.sY, B.sWt, sMixOut, B.sAcc, pNormW & ~15ull, 768, 0.4472136f};
            setl(en, "k_down_reduce8_conv_add", &drc, sizeof(drc));
            [en dispatchThreads:MTLSizeMake(768, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });
        bench("moe_batch8_r3(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8_m2", &gu, sizeof(gu));
            [en dispatchThreadgroups:MTLSizeMake((8 * BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            setl(en, "k_situ_quant8", &sa, sizeof(sa));
            [en dispatchThreads:MTLSizeMake(8 * BE, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            MoeDownArgs md3{pEX + 2 * BE * R2448, B.sHQ, B.sDH, B.sY, B.sWt, 0, BE, (uint32_t)EXPS, B.sSel};
            setl(en, "k_gemv_down_batch8_m2", &md3, sizeof(md3));
            [en dispatchThreadgroups:MTLSizeMake((BE + 15) / 16, 8, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            DownReduceConvArgs drc{B.sY, B.sWt, sMixOut, B.sAcc, pNormW & ~15ull, BE, 0.4472136f};
            setl(en, "k_down_reduce8_conv_add", &drc, sizeof(drc));
            [en dispatchThreads:MTLSizeMake(BE, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });
        const uint32_t mns[6] = {1024, 512, 512, 384, 24, 1024};
        MultiGemvArgs MG{};
        MG.xqoff = B.sX; MG.dxoff = B.sD; MG.count = 6;
        uint32_t mtot = 0;
        for (uint32_t i = 0; i < 6; i++) {
            MG.woff[i] = pW2 + (uint64_t)mtot * R2448;
            MG.yoff[i] = B.sY + mtot;
            MG.nout[i] = mns[i];
            mtot += mns[i];
            MG.cum[i] = mtot;
        }
        MG.total = mtot;
        bench("multi6_qkvrg_6disp_r2", IT, [&](id<MTLComputeCommandEncoder> en) {
            for (int i = 0; i < 6; i++) {
                GemvArgs gi{MG.woff[i], 0, B.sX, B.sD, MG.yoff[i], 0, MG.nout[i], 0, 0, 0};
                setl(en, "k_gemv_q8_k2304_m2", &gi, sizeof(gi));
                [en dispatchThreadgroups:MTLSizeMake((MG.nout[i] + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        });
        bench("multi6_qkvrg_1disp_m2", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_q8_k2304_multi_m2", &MG, sizeof(MG));
            [en dispatchThreadgroups:MTLSizeMake((MG.total + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        const uint64_t qmixOff = (pW1 + 15ull) & ~15ull;
        MixDotArgs md_mix{sStn, sMixSc, qmixOff, 8, 0, 1.0f / sqrtf(2304.0f)};
        MixSoftArgs ms_mix{sMixSc, 8};
        MixCombArgs mc_mix{sStn, sMixSc, sMixOut, 8, 3};
        bench("mix_unfused(S=8,nq=3,3disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_mix_dot", &md_mix, sizeof(md_mix));
            [en dispatchThreadgroups:MTLSizeMake(3, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            setl(en, "k_mix_soft", &ms_mix, sizeof(ms_mix));
            [en dispatchThreadgroups:MTLSizeMake(3, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            setl(en, "k_mix_comb", &mc_mix, sizeof(mc_mix));
            [en dispatchThreadgroups:MTLSizeMake(3, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        MixFusedArgs mf_mix{qmixOff, sStn, sMixOut, 8, 3, 1.0f / sqrtf(2304.0f), 0};
        bench("mix_fused(S=8,nq=3,1disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_mix_fused", &mf_mix, sizeof(mf_mix));
            [en dispatchThreadgroups:MTLSizeMake(3, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
    }

    // ---- 9. Round-4 A/B: split ctrl vs k_control, gemv_f32 nth sweep, router par vs serial, rms S8 ----
    {
        // Control weights reuse the 9.4 MB expert pack region (timing only;
        // offsets stay 16B-aligned so both kernels take their float4 paths).
        const uint64_t cWD = pEX;                    // 3 x 64*4608*4 = 3.54 MB
        const uint64_t cWG = pEX + 3538944ull;       // 3 x 2304*64*4  = 1.77 MB
        const uint64_t cWF = pEX + 5308416ull;       // ends 7.08 MB  < 9.40 MB region
        uint32_t sGcat = segf(3 * 2 * 2304), sHh = segf(3 * 64);
        uint32_t sCin = segf(3 * 2304), sCmsg = segf(3 * 2304), sCout = segf(3 * 2304);
        uint32_t sGb = segf(3 * 2304), sFb = segf(3 * 2304);
        CtrlArgs CA{};
        for (uint32_t j = 0; j < 3; j++) {
            CA.in[j] = sCin + j * 2304; CA.msg[j] = sCmsg + j * 2304; CA.out[j] = sCout + j * 2304;
            CA.wdown[j] = cWD + j * 1179648ull; CA.wgain[j] = cWG + j * 589824ull; CA.wforg[j] = cWF + j * 589824ull;
            CA.gb[j] = sGb + j * 2304; CA.fb[j] = sFb + j * 2304;
        }
        CA.gcat = sGcat; CA.eps = 1e-6f; CA.grank = 64; CA.ne = 2304;
        bench("ctrl_old(nc=3,k_control)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_control", &CA, sizeof(CA));
            [en dispatchThreadgroups:MTLSizeMake(3, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        });
        CtrlSplitArgs SA{};
        for (uint32_t j = 0; j < 3; j++) {
            SA.in[j] = sCin + j * 2304; SA.msg[j] = sCmsg + j * 2304; SA.out[j] = sCout + j * 2304;
            SA.wdown[j] = cWD + j * 1179648ull; SA.wgain[j] = cWG + j * 589824ull; SA.wforg[j] = cWF + j * 589824ull;
        }
        SA.gcat = sGcat; SA.hh = sHh; SA.eps = 1e-6f; SA.grank = 64; SA.ne = 2304; SA.nc = 3;
        bench("ctrl_r4_split(nc=3,2disp)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_ctrl_down", &SA, sizeof(SA));
            [en dispatchThreadgroups:MTLSizeMake(64, 3, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            setl(en, "k_ctrl_up", &SA, sizeof(SA));
            [en dispatchThreadgroups:MTLSizeMake(72, 3, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        });
        // gemv_f32 nth sweep (router GEMV shape: nout=128, k=2304). Appended
        // f32 block: 26.02 MB used + 1.18 MB = 27.20 MB <= BPK 27.26 MB.
        const uint64_t pF32 = (pNormW + (size_t)2304 * 4 + 15ull) & ~(size_t)15;
        if (pF32 + 128ull * 2304 * 4 > BPK) { fprintf(stderr, "bench: pF32 exceeds 26 MB pack\n"); return; }
        {
            float * w32 = (float *)((uint8_t *)pBuf.contents + pF32);
            for (uint32_t i = 0; i < 128u * 2304u; i++) w32[i] = uf(g2);
        }
        GemvF32Args FA32{pF32, B.sX, B.sY, 128, 2304};
        for (uint nth : {256u, 128u, 64u}) {
            char nm[48]; snprintf(nm, sizeof(nm), "gemv_f32_r128(k2304,nth=%u)", nth);
            bench(nm, IT, [&](id<MTLComputeCommandEncoder> en) {
                setl(en, "k_gemv_f32_tg", &FA32, sizeof(FA32));
                [en dispatchThreadgroups:MTLSizeMake(128, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            });
        }
        // Router: serial vs R4 parallel SIMD. Both kernels index e=0..127
        // (threadgroup arrays are [128]) so 128th is the only legal config;
        // a 256th dispatch would write pp/sp out of bounds.
        uint32_t sPr = segf(128), sSel2 = segf(8), sWt2 = segf(8);
        RouterArgs RA4{sPr, pNormW, sSel2, sWt2, 8.0f};   // biaspack = ones region (timing only)
        bench("router_old(128th+serial)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_router_topk_serial", &RA4, sizeof(RA4));
            [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });
        bench("router_r4(128th+parallel)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_router_topk", &RA4, sizeof(RA4));
            [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        });
        // k_rms on 8 x 2304 rows (do_mix S=8 prelude shape).
        uint32_t sYr8 = segf(8 * 2304);
        RMSArgs RM8{}; RM8.k = 2304; RM8.rows = 8; RM8.hasw = 1; RM8.eps = 1e-6f;
        for (uint32_t r = 0; r < 8; r++) { RM8.xr[r] = sStn + r * 2304; RM8.yr[r] = sYr8 + r * 2304; RM8.wr[r] = pNormW; }
        bench("rms_mix_S8(2304)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_rms", &RM8, sizeof(RM8));
            [en dispatchThreadgroups:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
    }

    // ---- 10. Milestone 7A: Q4_K gateup m1/m2 + Q5_0/Q5_1 down m1/m4 ----
    // The Q8 lm_head span [pW2, pEX) (~10 MB, ~9.1 MB needed) is dead after
    // the multi6/mix benches, so seed it with Q4_K / Q5_0 / Q5_1 expert blobs
    // for timing only (parity is gated by tests 47-50, not the bench).
    {
        const uint64_t RQ4 = 9ull * 144ull;                    // Q4_K row bytes (k=2304)
        const uint64_t R50 = 36ull * 22ull, R51 = 36ull * 24ull;  // Q5_0 / Q5_1 rows (k=1152)
        auto al4k = [](uint64_t v) { return (v + 4095ull) & ~4095ull; };
        const uint64_t pQ4G = pW2;
        const uint64_t pQ4U = al4k(pQ4G + 8ull * BE * RQ4);
        const uint64_t pQ50 = al4k(pQ4U + 8ull * BE * RQ4);
        const uint64_t pQ51 = al4k(pQ50 + 8ull * BE * R50);
        if (pQ51 + 8ull * BE * R51 > pEX) { fprintf(stderr, "bench: 7A blobs exceed pW2 span\n"); return; }
        uint8_t * pp = (uint8_t *)pBuf.contents;
        // Random bytes; mask f16 scale exponent bytes so no inf/nan/denormals.
        auto seedq = [&](uint64_t off, uint64_t bytes, uint64_t blk, int f16n) {
            for (uint64_t i = 0; i + 4 <= bytes; i += 4) { uint32_t r = g2(); memcpy(pp + off + i, &r, 4); }
            for (uint64_t o = 0; o + blk <= bytes; o += blk) {
                pp[off + o + 1] &= 0x3B;
                if (f16n > 1) pp[off + o + 3] &= 0x3B;
            }
        };
        seedq(pQ4G, 8ull * BE * RQ4, 144, 2);
        seedq(pQ4U, 8ull * BE * RQ4, 144, 2);
        seedq(pQ50, 8ull * BE * R50, 22, 1);
        seedq(pQ51, 8ull * BE * R51, 24, 2);
        MoeGateUpQ4KF32Args GQ{pQ4G, pQ4U, B.sX, B.sGate2, B.sU, B.sSel, 0u, BE, 2304u, (uint32_t)(BE * RQ4)};
        bench("gateup_q4k_f32in_m1(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8_q4k_f32in", &GQ, sizeof(GQ));
            [en dispatchThreadgroups:MTLSizeMake((8 * 2 * BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("gateup_q4k_f32in_m2(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8_q4k_f32in_m2", &GQ, sizeof(GQ));
            [en dispatchThreadgroups:MTLSizeMake((8 * BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        MoeDownQ5F32Args DQ0{pQ50, sStn, B.sAcc, B.sWt, B.sSel, 0u, 0u, BE, 1152u, (uint32_t)(BE * R50), 0u};
        MoeDownQ5F32Args DQ1{pQ51, sStn, B.sAcc, B.sWt, B.sSel, 0u, 0u, BE, 1152u, (uint32_t)(BE * R51), 0u};
        bench("down_q5_0_f32in_m1(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_q5_0_f32in", &DQ0, sizeof(DQ0));
            [en dispatchThreadgroups:MTLSizeMake((BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("down_q5_0_f32in_m4(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_q5_0_f32in_m4", &DQ0, sizeof(DQ0));
            [en dispatchThreadgroups:MTLSizeMake((BE + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("down_q5_1_f32in_m1(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_q5_1_f32in", &DQ1, sizeof(DQ1));
            [en dispatchThreadgroups:MTLSizeMake((BE + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("down_q5_1_f32in_m4(BE=192)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_q5_1_f32in_m4", &DQ1, sizeof(DQ1));
            [en dispatchThreadgroups:MTLSizeMake((BE + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });

        // Real Inkling-2-Mini decode shapes (gate/up nout=1152 k=2304, down
        // nout=2304 k=1152, 8 experts). At BE=192 the m4 grid starves the GPU
        // (6 TGs); the parity-gated real shapes decide whether m2/m4 wins.
        const uint32_t NR4 = 1152u, NR5 = 2304u;
        const uint64_t pRG = (size_t)34ull * 1024 * 1024;         // past all legacy regions
        const uint64_t pRU = al4k(pRG + 8ull * NR4 * RQ4);
        const uint64_t pR0 = al4k(pRU + 8ull * NR4 * RQ4);
        const uint64_t pR1 = al4k(pR0 + 8ull * NR5 * R50);
        if (pR1 + 8ull * NR5 * R51 > BPK) { fprintf(stderr, "bench: 7A real-shape blobs exceed pack\n"); return; }
        seedq(pRG, 8ull * NR4 * RQ4, 144, 2);
        seedq(pRU, 8ull * NR4 * RQ4, 144, 2);
        seedq(pR0, 8ull * NR5 * R50, 22, 1);
        seedq(pR1, 8ull * NR5 * R51, 24, 2);
        uint32_t sG2R = segf(8 * NR4), sUR = segf(8 * NR4), sHR = segf(8 * 1152);
        if (c > (1u << 20)) { fprintf(stderr, "bench: scratch exhausted\n"); return; }
        MoeGateUpQ4KF32Args GR{pRG, pRU, B.sX, sG2R, sUR, B.sSel, 0u, NR4, 2304u, (uint32_t)(NR4 * RQ4)};
        bench("gateup_q4k_m1(real 8x1152)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8_q4k_f32in", &GR, sizeof(GR));
            [en dispatchThreadgroups:MTLSizeMake(8 * 2 * NR4 / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("gateup_q4k_m2(real 8x1152)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_gateup_batch8_q4k_f32in_m2", &GR, sizeof(GR));
            [en dispatchThreadgroups:MTLSizeMake(8 * NR4 / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        MoeDownQ5F32Args DR0{pR0, sHR, B.sAcc, B.sWt, B.sSel, 0u, 0u, NR5, 1152u, (uint32_t)(NR5 * R50), 0u};
        MoeDownQ5F32Args DR1{pR1, sHR, B.sAcc, B.sWt, B.sSel, 0u, 0u, NR5, 1152u, (uint32_t)(NR5 * R51), 0u};
        bench("down_q5_0_m1(real 2304)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_q5_0_f32in", &DR0, sizeof(DR0));
            [en dispatchThreadgroups:MTLSizeMake((NR5 + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("down_q5_0_m4(real 2304)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_q5_0_f32in_m4", &DR0, sizeof(DR0));
            [en dispatchThreadgroups:MTLSizeMake((NR5 + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("down_q5_1_m1(real 2304)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_q5_1_f32in", &DR1, sizeof(DR1));
            [en dispatchThreadgroups:MTLSizeMake((NR5 + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
        bench("down_q5_1_m4(real 2304)", IT, [&](id<MTLComputeCommandEncoder> en) {
            setl(en, "k_gemv_down_reduce8_q5_1_f32in_m4", &DR1, sizeof(DR1));
            [en dispatchThreadgroups:MTLSizeMake((NR5 + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });
    }
}

int main(int argc, char ** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--bench")) do_bench = true;
    @autoreleasepool {
        Rng R;
        MetalRuntime rt;
        std::string err;
        if (!rt.init("engine/kernels", "build/m64_p2a_test.arch", err)) {
            printf("FAIL: MetalRuntime init: %s\n", err.c_str()); return 1;
        }

        // ---------------- host mirrors + GPU buffers ----------------
        const uint32_t SCR_F = 1024u << 10;            // 4 MB of floats (33 legacy tests + R4 need >2 MB)
        const size_t PK_B = 96u * 1024 * 1024;         // 96 MB fake pack (legacy Q8 blobs + M4B Q4_K/Q5 arenas)
        std::vector<float> scr(SCR_F, 0.f);
        std::vector<uint8_t> pk(PK_B, 0);
        id<MTLBuffer> bScr = [rt.device() newBufferWithLength:SCR_F * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bPk = [rt.device() newBufferWithLength:PK_B options:MTLResourceStorageModeShared];
        uint8_t * SCR_B = (uint8_t *)scr.data();
        uint32_t cur = 16;
        auto seg = [&](uint32_t n) { uint32_t r = cur; CHECK((size_t)cur + n + 15u < SCR_F, "SCR overflow"); cur += n + 15u; return r; };   // padded float offsets
        size_t pcur = 0;
        auto psegs = [&](size_t n) { pcur = (pcur + 63) & ~(size_t)63; size_t r = pcur; CHECK(pcur + n <= PK_B, "PACK overflow"); pcur += n; return r; };

        auto run = [&](const char * fn, bool tg, uint tx, uint ty, uint nth, const void * a, size_t sz, int tok = -1) {
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
            if (tok >= 0) { uint32_t t = (uint32_t)tok; [en setBytes:&t length:4 atIndex:3]; }
            if (tg) [en dispatchThreadgroups:MTLSizeMake(tx, ty, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            else    [en dispatchThreads:MTLSizeMake(tx, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
            [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
            CHECK(cmd.status == MTLCommandBufferStatusCompleted, "%s cmdbuf status=%ld err=%s", fn,
                  (long)cmd.status, cmd.error.localizedDescription.UTF8String);
            memcpy(scr.data(), bScr.contents, SCR_F * 4);
        };
        auto ok = [&](const char * name, const float * ref, uint32_t off, size_t n, double tol) {
            const float * got = scr.data() + off;
            double rel = max_rel(ref, got, n), cosv = cosine(ref, got, n), mad = 0;
            for (size_t i = 0; i < n; i++) mad = std::max(mad, std::fabs((double)ref[i] - got[i]));
            const bool pass = ((rel < tol || mad < 1e-5) && cosv > 0.999);
            CHECK(pass, "%s rel=%.3g cos=%.6f mad=%.3g", name, rel, cosv, mad);
            printf("%-24s rel=%.3g cos=%.8f mad=%.3g %s\n", name, rel, cosv, mad, pass ? "OK" : "FAIL");
        };
        auto fill_q8 = [&](size_t off, size_t rows, int nu) {
            for (size_t i = 0; i < rows * nu * 68; i++) pk[off + i] = (uint8_t)R.u32();
            for (size_t r = 0; r < rows; r++) for (int u = 0; u < nu; u++) {
                pk[off + r * nu * 68 + u * 68 + 1] &= 0x3B;        // finite f16 scales
                pk[off + r * nu * 68 + u * 68 + 35] &= 0x3B;
            }
        };

        // ---- 1. k_emb_gather (BITWISE) ----
        {
            uint32_t sE = seg(kNE);
            size_t epk = psegs(8 * 2448);
            for (size_t i = 0; i < 8 * 2448; i++) pk[epk + i] = (uint8_t)R.u32();
            for (int i = 0; i < 8; i++) for (int b = 0; b < 72; b++) pk[epk + i * 2448 + b * 34 + 1] &= 0x3B;
            int tok = 5;
            EmbArgs A{epk, sE, 72};
            run("k_emb_gather", false, 72, 1, 32, &A, sizeof(A), tok);
            int bad = 0;
            for (int b = 0; b < 72; b++) {
                const uint8_t * wp = pk.data() + epk + tok * 2448 + b * 34;
                float d = f16le(wp);
                for (int i = 0; i < 32; i++) if (scr[sE + b * 32 + i] != d * (float)sb(wp[2 + i])) bad++;
            }
            CHECK(bad == 0, "k_emb_gather %d mismatches", bad);
            printf("k_emb_gather             bitwise %d elems %s\n", 72 * 32, bad ? "FAIL" : "OK");
        }

        // ---- 2. k_rms (hasw 1/0; rows 1/3/8 — prod launch: tg=rows nth=256) ----
        {
            size_t wpk = psegs(kNE * 4);
            for (uint32_t i = 0; i < kNE; i++) ((float *)(pk.data() + wpk))[i] = 1.0f + R.f(-0.1f, 0.1f);
            float eps = 1e-6f;
            for (auto [rows, hasw] : std::vector<std::pair<uint32_t, uint32_t>>{{1, 1}, {3, 1}, {8, 0}}) {
                uint32_t xs[8], ys[8];
                for (uint32_t r = 0; r < rows; r++) {
                    xs[r] = seg(kNE); ys[r] = seg(kNE);
                    for (uint32_t i = 0; i < kNE; i++) scr[xs[r] + i] = R.f(-4, 4);
                }
                RMSArgs A{}; A.k = kNE; A.rows = rows; A.hasw = hasw; A.eps = eps;
                for (uint32_t r = 0; r < rows; r++) { A.xr[r] = xs[r]; A.yr[r] = ys[r]; A.wr[r] = hasw ? wpk : 0; }
                run("k_rms", true, rows, 1, 256, &A, sizeof(A));
                for (uint32_t r = 0; r < rows; r++) {
                    double ss = 0;
                    for (uint32_t i = 0; i < kNE; i++) ss += (double)scr[xs[r] + i] * scr[xs[r] + i];
                    float inv = 1.0f / sqrtf((float)(ss / kNE) + eps);
                    std::vector<float> ref(kNE);
                    for (uint32_t i = 0; i < kNE; i++)
                        ref[i] = hasw ? scr[xs[r] + i] * inv * ((const float *)(pk.data() + wpk))[i] : scr[xs[r] + i] * inv;
                    char nm[48]; snprintf(nm, sizeof(nm), "k_rms r=%u w=%u", rows, hasw);
                    ok(nm, ref.data(), ys[r], kNE, 2e-5);
                }
            }
        }

        // ---- 3. k_quant (int8 codes + f16-rounded d: BITWISE) ----
        uint32_t sX = seg(kNE), sQ = seg(kNE), sD = seg(256);
        for (uint32_t i = 0; i < kNE; i++) scr[sX + i] = R.f(-8, 8);
        QuantArgs QA{sX, sQ, sD, kNE / 32};
        run("k_quant", false, kNE / 32, 1, 32, &QA, sizeof(QA));
        {
            std::vector<int8_t> rq(kNE); std::vector<float> rd(kNE / 32);
            int bad = 0;
            for (uint32_t b = 0; b < kNE / 32; b++) {
                float amax = 0;
                for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(scr[sX + b * 32 + i]));
                float dq = amax / 127.0f, id = dq != 0.0f ? 1.0f / dq : 0.0f;
                rd[b] = r16(dq);
                if (memcmp(&scr[sD + b], &rd[b], 4) != 0) bad++;
                for (int i = 0; i < 32; i++) {
                    int r = (int)rintf(scr[sX + b * 32 + i] * id);
                    rq[b * 32 + i] = (int8_t)(r > 127 ? 127 : (r < -127 ? -127 : r));
                    if ((int8_t)SCR_B[sQ * 4 + b * 32 + i] != rq[b * 32 + i]) bad++;
                }
            }
            CHECK(bad == 0, "k_quant %d mismatches", bad);
            printf("k_quant                  codes+f16d %d elems %s\n", kNE + kNE / 32, bad ? "FAIL" : "OK");
        }

        // ---- 4-6. k_gemv_q8_{k2304,k1152,k3072} ----
        {
            struct GM { uint32_t nout; int nu; const char * fn; } gms[] = {
                {64, 36, "k_gemv_q8_k2304"}, {64, 18, "k_gemv_q8_k1152"}, {32, 48, "k_gemv_q8_k3072"}};
            const int8_t * xq = (const int8_t *)(SCR_B + sQ * 4);
            for (auto & gm : gms) {
                size_t w = psegs((size_t)gm.nout * gm.nu * 68);
                fill_q8(w, gm.nout, gm.nu);
                uint32_t y = seg(gm.nout);
                GemvArgs A{w, 0, sQ, sD, y, 0, gm.nout, 0, 0, 0};
                uint32_t rpg = gm.nu == 18 ? 16 : 8;                    // warp-per-row vs warp-per-2-rows
                run(gm.fn, true, (gm.nout + rpg - 1) / rpg, 1, 256, &A, sizeof(A));
                std::vector<float> ref(gm.nout);
                for (uint32_t r = 0; r < gm.nout; r++)
                    ref[r] = gemv_ref(pk.data() + w + (size_t)r * gm.nu * 68, xq, scr.data() + sD, gm.nu);
                ok(gm.fn, ref.data(), y, gm.nout, 1e-4);
            }
            // ---- 7. k_gemv_gateup: expert blob via sel[idx] (idx=1 -> expert 2) ----
            uint32_t nout = 32;
            size_t gate0 = psegs(3 * nout * 36 * 68), up0 = psegs(3 * nout * 36 * 68);
            for (int e = 0; e < 3; e++) { fill_q8(gate0 + e * nout * 36 * 68, nout, 36); fill_q8(up0 + e * nout * 36 * 68, nout, 36); }
            uint32_t sel = seg(8);
            ((uint32_t *)scr.data())[sel] = 5; ((uint32_t *)scr.data())[sel + 1] = 2;
            uint32_t yg = seg(64), yu = seg(64);
            GemvArgs A{gate0, up0, sQ, sD, yg, yu, nout, 1, nout * 36 * 68, sel};
            run("k_gemv_gateup", true, (2 * nout + 7) / 8, 1, 256, &A, sizeof(A));
            std::vector<float> ref(nout);
            for (uint32_t r = 0; r < nout; r++)
                ref[r] = gemv_ref(pk.data() + gate0 + 2 * nout * 36 * 68 + (size_t)r * 36 * 68, xq, scr.data() + sD, 36);
            ok("k_gemv_gateup.gate", ref.data(), yg, nout, 1e-4);
            for (uint32_t r = 0; r < nout; r++)
                ref[r] = gemv_ref(pk.data() + up0 + 2 * nout * 36 * 68 + (size_t)r * 36 * 68, xq, scr.data() + sD, 36);
            ok("k_gemv_gateup.up", ref.data(), yu, nout, 1e-4);
            // ---- 8. k_gemv_downacc: acc += w*dot (plain mul+add both sides) ----
            nout = 64;
            ref.resize(nout);
            size_t est = psegs(2 * (size_t)nout * 18 * 68);
            fill_q8(est, 2 * nout, 18);
            uint32_t sel2 = seg(8), wt = seg(8), acc = seg(nout);
            ((uint32_t *)scr.data())[sel2] = 0; ((uint32_t *)scr.data())[sel2 + 1] = 1;
            scr[wt + 1] = 0.37f;
            for (uint32_t i = 0; i < nout; i++) scr[acc + i] = R.f(-3, 3);
            std::vector<float> pre(scr.begin() + acc, scr.begin() + acc + nout);
            MoeDownArgs DA{est, sQ, sD, acc, wt, 1, nout, nout * 18 * 68, sel2};
            run("k_gemv_downacc", true, (nout + 15) / 16, 1, 256, &DA, sizeof(DA));
            for (uint32_t r = 0; r < nout; r++)
                ref[r] = pre[r] + 0.37f * gemv_ref(pk.data() + est + (size_t)(nout + r) * 18 * 68, xq, scr.data() + sD, 18);
            ok("k_gemv_downacc", ref.data(), acc, nout, 1e-4);
        }

        // ---- 9. k_gemv_f32_tg ----
        {
            uint32_t nout = 64, k = 512;
            size_t wpk = psegs((size_t)nout * k * 4);
            for (uint32_t i = 0; i < nout * k; i++) ((float *)(pk.data() + wpk))[i] = R.f(-1, 1);
            uint32_t x = seg(k), y = seg(nout);
            for (uint32_t i = 0; i < k; i++) scr[x + i] = R.f(-2, 2);
            GemvF32Args A{wpk, x, y, nout, k};
            run("k_gemv_f32_tg", true, nout, 1, 256, &A, sizeof(A));
            std::vector<float> ref(nout);
            const float * W = (const float *)(pk.data() + wpk);
            for (uint32_t r = 0; r < nout; r++) {
                double a = 0;
                for (uint32_t i = 0; i < k; i++) a += (double)W[r * k + i] * scr[x + i];
                ref[r] = (float)a;
            }
            ok("k_gemv_f32_tg", ref.data(), y, nout, 1e-5);
            // Round-2: force 16B-aligned x so the float4 path is exercised.
            uint32_t xa = (seg(k) + 3u) & ~3u, y2 = seg(nout);
            for (uint32_t i = 0; i < k; i++) scr[xa + i] = R.f(-2, 2);
            GemvF32Args AV{wpk, xa, y2, nout, k};
            run("k_gemv_f32_tg", true, nout, 1, 256, &AV, sizeof(AV));
            for (uint32_t r = 0; r < nout; r++) {
                double a = 0;
                for (uint32_t i = 0; i < k; i++) a += (double)W[r * k + i] * scr[xa + i];
                ref[r] = (float)a;
            }
            ok("k_gemv_f32_tg_vec4", ref.data(), y2, nout, 1e-5);
        }

        // ---- 10. k_situ_quant (codes + d: exact expected; dq cosine reported) ----
        {
            uint32_t nb = kNE / 32, g = seg(kNE), u = seg(kNE), q = seg(kNE / 4), d = seg(nb);
            for (uint32_t i = 0; i < kNE; i++) { scr[g + i] = R.f(-6, 6); scr[u + i] = R.f(-20, 20); }
            float beta = 4.0f, lb = 25.0f;
            SituArgs A{g, u, q, d, nb, beta, lb, 1.0f / beta, 1.0f / lb};
            run("k_situ_quant", false, nb, 1, 32, &A, sizeof(A));
            std::vector<float> hs_ref(kNE), hs_got(kNE);
            int bad = 0;
            for (uint32_t b = 0; b < nb; b++) {
                float hs[32], amax = 0;
                for (int i = 0; i < 32; i++) {
                    float x = scr[g + b * 32 + i];
                    float act = beta * tanhf(x * (1.0f / beta));
                    act = act * (1.0f / (1.0f + expf(-x)));
                    float ulin = lb * tanhf(scr[u + b * 32 + i] * (1.0f / lb));
                    hs[i] = act * ulin; amax = std::max(amax, std::fabs(hs[i]));
                    hs_ref[b * 32 + i] = hs[i];
                }
                float dq = amax / 127.0f, dr = r16(dq), id = dq != 0.0f ? 1.0f / dq : 0.0f;
                if (memcmp(&scr[d + b], &dr, 4) != 0) bad++;             // f16(d) must match bitwise
                for (int i = 0; i < 32; i++) {
                    int r = (int)rintf(hs[i] * id);
                    r = r > 127 ? 127 : (r < -127 ? -127 : r);
                    int gq = sb(SCR_B[q * 4 + b * 32 + i]);
                    if (gq != r) bad++;
                    hs_got[b * 32 + i] = scr[d + b] * (float)gq;         // GPU dequantized activation
                }
            }
            double rel = max_rel(hs_ref.data(), hs_got.data(), kNE), cosv = cosine(hs_ref.data(), hs_got.data(), kNE);
            CHECK(bad == 0, "k_situ_quant %d code/d mismatches", bad);
            CHECK(cosv > 0.999, "k_situ_quant dq rel=%.3g cos=%.6f", rel, cosv);
            printf("k_situ_quant             mismatches=%d dq rel=%.3g cos=%.8f %s\n", bad, rel, cosv, bad ? "FAIL" : "OK");
        }

        // ---- 11. k_conv_slot (BITWISE: fp-contract off, identical order) ----
        for (uint32_t hasdyn : {0u, 1u}) {
            uint32_t C = hasdyn ? 256u : 512u, group = 128;
            uint32_t rs = seg(C * 3), x = seg(C), y = seg(C), dyn = seg(64);
            size_t w = psegs(C * 16);
            for (uint32_t i = 0; i < C * 3; i++) scr[rs + i] = R.f(-3, 3);
            for (uint32_t i = 0; i < C; i++) { scr[x + i] = R.f(-3, 3); for (int c = 0; c < 4; c++) ((float *)(pk.data() + w))[i * 4 + c] = R.f(-1, 1); }
            for (uint32_t i = 0; i < (C / group) * 4; i++) scr[dyn + i] = R.f(-1, 1);
            float scale = hasdyn ? 1.0f / 3.0f : 1.0f / sqrtf(5.0f);
            std::vector<float> st0(scr.begin() + rs, scr.begin() + rs + C * 3), xin(scr.begin() + x, scr.begin() + x + C);
            ConvArgs A{rs, x, y, w, dyn, C, group, scale, hasdyn};
            run("k_conv_slot", false, C, 1, 128, &A, sizeof(A));
            int bad = 0;
            for (uint32_t c = 0; c < C; c++) {
                const float * wv = ((const float *)(pk.data() + w)) + c * 4;
                float s0 = st0[c * 3], s1 = st0[c * 3 + 1], s2 = st0[c * 3 + 2], xi = xin[c];
                float outv = s0 * wv[0] + s1 * wv[1] + s2 * wv[2] + xi * wv[3];
                if (hasdyn) { const float * dn = scr.data() + dyn + (c / group) * 4; outv += dn[0] * xi + dn[1] * s2 + dn[2] * s1 + dn[3] * s0; }
                if (scr[y + c] != (xi + outv) * scale) bad++;
                if (scr[rs + c * 3] != s1 || scr[rs + c * 3 + 1] != s2 || scr[rs + c * 3 + 2] != xi) bad++;
            }
            char nm[48]; snprintf(nm, sizeof(nm), "k_conv_slot dyn=%u", hasdyn);
            CHECK(bad == 0, "%s %d mismatches", nm, bad);
            printf("%-24s bitwise C=%u %s\n", nm, C, bad ? "FAIL" : "OK");
        }

        // ---- 12. k_clip (scale tensor + M64_NOFF) ----
        for (auto [groups, per, usescale] : std::vector<std::tuple<uint32_t, uint32_t, int>>{{24, 128, 1}, {64, 16, 0}}) {
            uint32_t c = seg(groups * per);
            uint64_t soff = M64_NOFF;
            if (usescale) { soff = psegs((size_t)groups * per * 4); for (uint32_t i = 0; i < groups * per; i++) ((float *)(pk.data() + soff))[i] = 1.0f + R.f(-0.2f, 0.2f); }
            std::vector<float> in0(groups * per);
            for (uint32_t i = 0; i < groups * per; i++) in0[i] = scr[c + i] = R.f(-5, 5);
            float thresh = per == 128 ? 22.6f : 8.0f;
            ClipArgs A{c, soff, groups, per, thresh};
            run("k_clip", true, groups, 1, 128, &A, sizeof(A));
            std::vector<float> ref(groups * per);
            for (uint32_t g = 0; g < groups; g++) {
                double ss = 0;
                for (uint32_t i = 0; i < per; i++) {
                    float v = in0[g * per + i];
                    if (usescale) v *= ((const float *)(pk.data() + soff))[g * per + i];
                    ref[g * per + i] = v; ss += (double)v * v;
                }
                float f = thresh / (sqrtf((float)ss) + 1e-6f);
                f = fminf(fmaxf(f, 0.0f), 1.0f);
                for (uint32_t i = 0; i < per; i++) ref[g * per + i] *= f;
            }
            char nm[48]; snprintf(nm, sizeof(nm), "k_clip g=%u p=%u s=%d", groups, per, usescale);
            ok(nm, ref.data(), c, groups * per, 1e-5);
        }

        // ---- 13. k_attn_head (is_swa 0/1, pos 0 & >0, rout/sink/relbias, fa16 0/1) ----
        {
            uint32_t sQh = seg(kNH * kDH), sRout = seg(kNH * kNREL), sSink = seg(kNH);
            uint32_t sKvk = seg(10 * kKVD), sKvv = seg(10 * kKVD), sOut = seg(kNH * kDH);
            size_t rb = psegs(64 * kNREL * 4);
            for (uint32_t i = 0; i < kNH * kDH; i++) scr[sQh + i] = R.f(-3, 3);
            for (uint32_t i = 0; i < kNH * kNREL; i++) scr[sRout + i] = R.f(-1, 1);
            for (uint32_t i = 0; i < kNH; i++) scr[sSink + i] = R.f(-2, 2);
            for (uint32_t i = 0; i < 10 * kKVD; i++) { scr[sKvk + i] = R.f(-3, 3); scr[sKvv + i] = R.f(-3, 3); }
            for (uint32_t i = 0; i < 64 * kNREL; i++) ((float *)(pk.data() + rb))[i] = R.f(-1, 1);
            float scale = 1.0f / sqrtf((float)kDH);
            auto ref_attn = [&](const AttnArgs & A, int nkv, int pos, std::vector<float> & ref) {
                for (uint32_t h = 0; h < kNH; h++) {
                    const float * q = scr.data() + sQh + h * kDH, * ro = scr.data() + sRout + h * kNREL;
                    float snk = scr[sSink + h];
                    uint32_t kvh = h / 3u;
                    float oh[128] = {0}, M = -INFINITY, S = 0;
                    for (int j = 0; j < nkv; j++) {
                        if (A.is_swa && (uint32_t)(pos - j) > A.n_swa) continue;
                        const float * kj = scr.data() + sKvk + j * kKVD + kvh * kDH;
                        const float * vj = scr.data() + sKvv + j * kKVD + kvh * kDH;
                        float d = 0;
                        for (int i = 0; i < 128; i++)
                            d = A.fa16 ? std::fma(r16(q[i]), r16(kj[i]), d) : std::fma(q[i], kj[i], d);
                        uint32_t dist = pos - j;
                        float bv = 0;
                        if (dist < 1024) { const float * br = ((const float *)(pk.data() + rb)) + dist * 16; for (int i = 0; i < 16; i++) bv += br[i] * ro[i]; }
                        float s = d * scale + (A.fa16 ? r16(bv - snk) : (bv - snk));
                        float ms = 1;
                        if (s > M) { ms = expf(M - s); for (int i = 0; i < 128; i++) oh[i] *= ms; M = s; }
                        float vs = s > 0 && ms != 1.0f ? 1.0f : expf(s - M);
                        for (int i = 0; i < 128; i++)
                            oh[i] = A.fa16 ? std::fma(vs, r16(vj[i]), oh[i]) : std::fma(vs, vj[i], oh[i]);
                        S = S * ms + vs;
                    }
                    {   // zero-sink term
                        float ms = 1;
                        if (0.0f > M) { ms = expf(M); for (int i = 0; i < 128; i++) oh[i] *= ms; M = 0; }
                        float vs = ms != 1.0f ? 1.0f : expf(0.0f - M);
                        S = S * ms + vs;
                    }
                    for (int i = 0; i < 128; i++) ref[h * kDH + i] = oh[i] * (1.0f / S);
                }
            };
            struct AC { uint32_t is_swa, fa16; int nkv, pos; uint32_t n_swa; } cfgs[] = {
                {1, 1, 1, 0, 1024}, {1, 1, 10, 9, 4}, {1, 0, 10, 9, 4}, {0, 1, 10, 9, 1024}, {0, 0, 10, 9, 1024}};
            for (auto & cf : cfgs) {
                AttnArgs A{};
                A.qoff = sQh; A.rout = sRout; A.sinkoff = sSink; A.relbiaspack = rb;
                A.kvkoff = sKvk; A.kvvoff = sKvv; A.outoff = sOut;
                A.nkv = cf.nkv; A.pos = cf.pos; A.is_swa = cf.is_swa; A.fa16 = cf.fa16; A.n_swa = cf.n_swa;
                A.scale = scale;
                run("k_attn_head", true, kNH, 1, 32, &A, sizeof(A));
                std::vector<float> ref(kNH * kDH);
                ref_attn(A, cf.nkv, cf.pos, ref);
                char nm[48]; snprintf(nm, sizeof(nm), "k_attn_head s=%u f=%u p=%d", cf.is_swa, cf.fa16, cf.pos);
                ok(nm, ref.data(), sOut, kNH * kDH, 1e-3);
            }
        }

        // ---- 14. k_gate_mul / 15. k_router_topk ----
        {
            uint32_t a = seg(kNE), gg = seg(kNE);
            std::vector<float> a0(kNE);
            for (uint32_t i = 0; i < kNE; i++) { a0[i] = scr[a + i] = R.f(-5, 5); scr[gg + i] = R.f(-6, 6); }
            GateMulArgs A{a, gg, kNE};
            run("k_gate_mul", false, kNE, 1, 128, &A, sizeof(A));
            std::vector<float> ref(kNE);
            for (uint32_t i = 0; i < kNE; i++) ref[i] = a0[i] * ksig(scr[gg + i]);
            ok("k_gate_mul", ref.data(), a, kNE, 1e-5);

            uint32_t pr = seg(128), sel = seg(8), wt = seg(8);
            size_t bias = psegs(128 * 4);
            for (uint32_t i = 0; i < 128; i++) { scr[pr + i] = R.f(-8, 8); ((float *)(pk.data() + bias))[i] = R.f(-0.1f, 0.1f); }
            RouterArgs RA{pr, bias, sel, wt, 8.0f};
            run("k_router_topk", true, 1, 1, 128, &RA, sizeof(RA));
            {
                float sp[128], pp[128], wsum = 0; int selr[8]; float psl[8];
                for (int i = 0; i < 128; i++) { pp[i] = ksig(scr[pr + i]); sp[i] = pp[i] + ((const float *)(pk.data() + bias))[i]; }
                for (int t = 0; t < 8; t++) {
                    int bi = 0; float best = -INFINITY;
                    for (int i = 0; i < 128; i++) if (sp[i] > best) { best = sp[i]; bi = i; }
                    selr[t] = bi; psl[t] = pp[bi]; sp[bi] = -INFINITY; wsum += pp[bi];
                }
                int * seld = (int *)scr.data();
                int rbad = 0;
                for (int t = 0; t < 8; t++) {
                    if (seld[sel + t] != selr[t]) rbad++;
                    CHECK(seld[sel + t] == selr[t], "router sel[%d] %d vs %d", t, seld[sel + t], selr[t]);
                    float rw = (psl[t] / (wsum + 1e-20f)) * 8.0f;
                    if (!(std::fabs(scr[wt + t] - rw) < 1e-4f * rw + 1e-9f)) rbad++;
                    CHECK(std::fabs(scr[wt + t] - rw) < 1e-4f * rw + 1e-9f, "router w[%d] %.7g vs %.7g", t, scr[wt + t], rw);
                }
                printf("k_router_topk            top8 sel+w %s\n", rbad ? "FAIL" : "OK");
            }
        }

        // ---- 16. k_mix_dot + k_mix_soft + k_mix_comb (S=12, nq=1/3) ----
        for (uint32_t nq : {1u, 3u}) {
            uint32_t S = 12;
            uint32_t stn = seg(S * kNE), sc = seg(nq * M64_MAXS), out = seg(nq * kNE);
            size_t qpk = psegs(nq * kNE * 4);
            for (uint32_t i = 0; i < S * kNE; i++) scr[stn + i] = R.f(-2, 2);
            for (uint32_t i = 0; i < nq * kNE; i++) ((float *)(pk.data() + qpk))[i] = R.f(-1, 1);
            float scale = 1.0f / sqrtf((float)kNE);
            MixDotArgs DA{stn, sc, qpk, S, 0, scale};
            run("k_mix_dot", true, nq, (S + 7) / 8, 256, &DA, sizeof(DA));
            MixSoftArgs SA{sc, S};
            run("k_mix_soft", true, nq, 1, 64, &SA, sizeof(SA));
            MixCombArgs CA{stn, sc, out, S, nq};
            run("k_mix_comb", true, nq, 1, 256, &CA, sizeof(CA));
            std::vector<float> ref(nq * kNE);
            const float * qw = (const float *)(pk.data() + qpk);
            for (uint32_t j = 0; j < nq; j++) {
                double scv[64], mx = -1e300;
                for (uint32_t i = 0; i < S; i++) {
                    double dot = 0;
                    for (uint32_t k = 0; k < kNE; k++) dot += (double)scr[stn + i * kNE + k] * qw[j * kNE + k];
                    scv[i] = dot * scale; mx = std::max(mx, scv[i]);
                }
                double sum = 0;
                for (uint32_t i = 0; i < S; i++) { scv[i] = exp(scv[i] - mx); sum += scv[i]; }
                for (uint32_t k = 0; k < kNE; k++) {
                    float a = 0;
                    for (uint32_t i = 0; i < S; i++) a = std::fma((float)(scv[i] / sum), scr[stn + i * kNE + k], a);
                    ref[j * kNE + k] = a;
                }
            }
            char nm[48]; snprintf(nm, sizeof(nm), "k_mix chain nq=%u", nq);
            ok(nm, ref.data(), out, nq * kNE, 1e-4);
        }

        // ---- 17. k_control (nc=2 groups x 1024 threads) ----
        {
            uint32_t nc = 2;
            uint32_t gcat = seg(nc * 2 * kNE); gcat = (gcat + 3u) & ~3u;   // 16B-align: exercise float4 path
            uint32_t gb = seg(nc * kNE), fb = seg(nc * kNE);
            uint32_t cin = seg(nc * kNE), cmsg = seg(nc * kNE), cout = seg(nc * kNE);
            size_t wd[2], wg[2], wf[2];
            for (uint32_t c = 0; c < nc; c++) {
                wd[c] = psegs((size_t)64 * 2 * kNE * 4); wg[c] = psegs((size_t)kNE * 64 * 4); wf[c] = psegs((size_t)kNE * 64 * 4);
                for (size_t i = 0; i < 64 * 2 * kNE; i++) ((float *)(pk.data() + wd[c]))[i] = R.f(-0.021f, 0.021f);
                for (size_t i = 0; i < (size_t)kNE * 64; i++) { ((float *)(pk.data() + wg[c]))[i] = R.f(-0.12f, 0.12f); ((float *)(pk.data() + wf[c]))[i] = R.f(-0.12f, 0.12f); }
            }
            for (uint32_t i = 0; i < nc * 2 * kNE; i++) scr[gcat + i] = R.f(-1, 1);
            for (uint32_t i = 0; i < nc * kNE; i++) { scr[cin + i] = R.f(-3, 3); scr[cmsg + i] = R.f(-3, 3); }
            CtrlArgs A{};
            for (uint32_t c = 0; c < nc; c++) {
                A.in[c] = cin + c * kNE; A.msg[c] = cmsg + c * kNE; A.out[c] = cout + c * kNE;
                A.wdown[c] = wd[c]; A.wgain[c] = wg[c]; A.wforg[c] = wf[c];
                A.gb[c] = gb + c * kNE; A.fb[c] = fb + c * kNE;
            }
            A.gcat = gcat; A.eps = 1e-6f; A.grank = 64; A.ne = kNE;
            run("k_control", true, nc, 1, 1024, &A, sizeof(A));
            std::vector<float> ref(nc * kNE);
            for (uint32_t c = 0; c < nc; c++) {
                const float * gc = scr.data() + gcat + c * 2 * kNE;
                const float * wmd = (const float *)(pk.data() + wd[c]);
                float hh[64];
                for (int t = 0; t < 64; t++) {
                    double a = 0;
                    for (uint32_t k = 0; k < 2 * kNE; k++) a += (double)wmd[t * 2 * kNE + k] * gc[k];
                    hh[t] = tanhf((float)a);
                }
                const float * wmg = (const float *)(pk.data() + wg[c]), * wmf = (const float *)(pk.data() + wf[c]);
                for (uint32_t row = 0; row < kNE; row++) {
                    float g = 0, f = 0;
                    for (int t = 0; t < 64; t++) { g = std::fma(wmg[row * 64 + t], hh[t], g); f = std::fma(wmf[row * 64 + t], hh[t], f); }
                    float gbt = tanhf(g), fbt = tanhf(f);
                    float gb_got = scr[gb + c * kNE + row], fb_got = scr[fb + c * kNE + row];
                    if (std::fabs((double)gbt - gb_got) > 1e-5 && max_rel(&gbt, &gb_got, 1) > 1e-3) g_fail++;
                    if (std::fabs((double)fbt - fb_got) > 1e-5 && max_rel(&fbt, &fb_got, 1) > 1e-3) g_fail++;
                    ref[c * kNE + row] = (scr[A.in[c] + row] - fbt * scr[A.in[c] + row]) + gbt * scr[A.msg[c] + row];
                }
            }
            ok("k_control (gb/fb+out)", ref.data(), cout, nc * kNE, 1e-3);
        }

        // ---- 18. k_xfer (mode 0 a+=b, mode 1 pair copies: BITWISE) ----
        {
            uint32_t a0s = seg(kNE), b0s = seg(kNE), a1s = seg(kNE), c1s = seg(kNE), d1s = seg(kNE);
            for (uint32_t i = 0; i < kNE; i++) { scr[a0s + i] = R.f(); scr[b0s + i] = R.f(); scr[a1s + i] = R.f(); }
            std::vector<float> a0(scr.begin() + a0s, scr.begin() + a0s + kNE), b0(scr.begin() + b0s, scr.begin() + b0s + kNE);
            XferArgs A{{a0s, a1s}, {b0s, 0}, {0, c1s}, {kNE, 0}, 0u};
            A.n[0] = kNE;
            run("k_xfer", true, 1, 1, 256, &A, sizeof(A));
            int bad = 0;
            for (uint32_t i = 0; i < kNE; i++) if (scr[a0s + i] != a0[i] + b0[i]) bad++;
            CHECK(bad == 0, "k_xfer mode0 %d mismatches", bad);
            XferArgs B{{a0s, a1s}, {0, 0}, {b0s, d1s}, {kNE, kNE}, 1u};
            run("k_xfer", true, 2, 1, 256, &B, sizeof(B));
            for (uint32_t i = 0; i < kNE; i++) if (scr[b0s + i] != a0[i] + b0[i] || scr[d1s + i] != scr[a1s + i]) bad++;
            printf("k_xfer mode0+mode1       bitwise %d elems %s\n", 3 * kNE, bad ? "FAIL" : "OK");
            CHECK(bad == 0, "k_xfer mode1 %d mismatches", bad);
        }

        // ---- 19. k_scale (BITWISE) ----
        {
            uint32_t v = seg(kNE);
            std::vector<float> v0(scr.begin() + v, scr.begin() + v + kNE);
            for (uint32_t i = 0; i < kNE; i++) v0[i] = scr[v + i] = R.f();
            ScaleArgs A{v, kNE, 0.25f};
            run("k_scale", false, kNE, 1, 256, &A, sizeof(A));
            int bad = 0;
            for (uint32_t i = 0; i < kNE; i++) if (scr[v + i] != v0[i] * 0.25f) bad++;
            CHECK(bad == 0, "k_scale %d mismatches", bad);
            printf("k_scale                  bitwise %d elems %s\n", kNE, bad ? "FAIL" : "OK");
        }

        // ---- 20. k_softcap + 21. k_argmax ----
        {
            uint32_t lg = seg(4096);
            std::vector<float> ref(4096);
            for (uint32_t i = 0; i < 4096; i++) { float z = R.f(-90, 90); ref[i] = 30.0f * tanhf(z * (1.0f / 30.0f)); scr[lg + i] = z; }
            SoftcapArgs A{lg, 4096, 30.0f};
            run("k_softcap", false, 4096, 1, 256, &A, sizeof(A));
            ok("k_softcap", ref.data(), lg, 4096, 1e-5);
            uint32_t am = seg(4);
            ((float *)(scr.data() + lg))[1234] = 12345.0f;            // unique max
            ArgmaxArgs B{lg, am, 4096};
            run("k_argmax", true, 1, 1, 1024, &B, sizeof(B));
            int got = ((int *)scr.data())[am];
            CHECK(got == 1234, "k_argmax %d vs 1234", got);
            printf("k_argmax                 -> %d %s\n", got, got == 1234 ? "OK" : "FAIL");
        }

        // ---- 22. k_gemv_q8_k2304_m2 / _m4 (tail clamp: nout 64/52/50) ----
        {
            const int8_t * xq = (const int8_t *)(SCR_B + sQ * 4);
            for (uint32_t nout : {64u, 52u, 50u}) {
                size_t w = psegs((size_t)nout * 36 * 68);
                fill_q8(w, nout, 36);
                uint32_t y = seg(64);
                std::vector<float> ref(nout);
                for (uint32_t r = 0; r < nout; r++)
                    ref[r] = gemv_ref(pk.data() + w + (size_t)r * 36 * 68, xq, scr.data() + sD, 36);
                for (auto gm : {std::make_pair("k_gemv_q8_k2304_m2", 16u), std::make_pair("k_gemv_q8_k2304_m4", 32u)}) {
                    GemvArgs A{w, 0, sQ, sD, y, 0, nout, 0, 0, 0};
                    run(gm.first, true, (nout + gm.second - 1) / gm.second, 1, 256, &A, sizeof(A));
                    char nm[48]; snprintf(nm, sizeof(nm), "%s n=%u", gm.first, nout);
                    ok(nm, ref.data(), y, nout, 1e-4);
                }
            }
        }

        // ---- 23. k_gemv_gateup_batch8 (8 experts, sel[e] = 7 - e) ----
        {
            uint32_t nout = 64;
            const size_t EXPB = (size_t)2 * nout * 36 * 68;
            size_t gbase = psegs(8 * EXPB);
            for (int e = 0; e < 8; e++) {
                fill_q8(gbase + e * EXPB, nout, 36);                       // gate rows
                fill_q8(gbase + e * EXPB + (size_t)nout * 36 * 68, nout, 36); // up rows
            }
            uint32_t selb = seg(8);
            for (uint32_t e = 0; e < 8; e++) ((uint32_t *)scr.data())[selb + e] = 7 - e;
            uint32_t g2 = seg(8 * nout), u2 = seg(8 * nout);
            GemvArgs A{gbase, gbase + (size_t)nout * 36 * 68, sQ, sD, g2, u2,
                       nout, 0, (uint32_t)EXPB, selb};
            run("k_gemv_gateup_batch8", true, (8 * 2 * nout + 7) / 8, 1, 256, &A, sizeof(A));
            const int8_t * xq = (const int8_t *)(SCR_B + sQ * 4);
            int bad = 0; double worst = 0;
            for (int part = 0; part < 2; part++) {
                uint32_t yoff = part ? u2 : g2;
                for (uint32_t e = 0; e < 8; e++) {
                    uint32_t ex = 7 - e;
                    for (uint32_t r = 0; r < nout; r++) {
                        const uint8_t * eb = pk.data() + gbase + ex * EXPB + (size_t)(part ? nout + r : r) * 36 * 68;
                        auto [a, sumabs] = gemv_ref64(eb, xq, scr.data() + sD, 36);
                        double bound = 1e-4 * std::fabs(a) + 1024.0 * 1.1920928955078125e-7 * sumabs;
                        double dev = std::fabs(a - (double)scr[yoff + e * nout + r]) / (bound > 0 ? bound : 1.0);
                        if (dev > 1.0) bad++;
                        if (dev > worst) worst = dev;
                    }
                }
            }
            CHECK(bad == 0, "gateup_batch8 %d rows outside 1024-ulp bound (worst=%.1f)", bad, worst);
            printf("gateup_batch8 (g+u)      ulp-gated %d/%d rows %s (worst %.1f bound)\n",
                   8 * nout * 2 - bad, 8 * nout * 2, bad ? "FAIL" : "OK", worst);
        }

        // ---- 24. k_situ_quant8 (8 experts x 36 blocks) vs CPU, bitwise ----
        {
            uint32_t nb = 8 * (kNE / 32), g = seg(8 * kNE), u = seg(8 * kNE), q = seg(2 * kNE), d = seg(nb);
            for (uint32_t i = 0; i < 8 * kNE; i++) { scr[g + i] = R.f(-6, 6); scr[u + i] = R.f(-20, 20); }
            float beta = 4.0f, lb = 25.0f;
            SituArgs A{g, u, q, d, nb, beta, lb, 1.0f / beta, 1.0f / lb};
            run("k_situ_quant8", false, nb * 32, 1, 32, &A, sizeof(A));
            int bad = 0;
            for (uint32_t b = 0; b < nb; b++) {
                float amax = 0; float hs[32];
                for (int i = 0; i < 32; i++) {
                    float x = scr[g + b * 32 + i];
                    float act = beta * tanhf(x * (1.0f / beta));
                    act = act * (1.0f / (1.0f + expf(-x)));
                    float ulin = lb * tanhf(scr[u + b * 32 + i] * (1.0f / lb));
                    hs[i] = act * ulin; float a = hs[i] < 0 ? -hs[i] : hs[i]; if (a > amax) amax = a;
                }
                float dq = amax / 127.0f, dr = r16(dq), id = dq != 0.0f ? 1.0f / dq : 0.0f;
                if (memcmp(&scr[d + b], &dr, 4) != 0) bad++;
                for (int i = 0; i < 32; i++) {
                    int r = (int)rintf(hs[i] * id);
                    r = r > 127 ? 127 : (r < -127 ? -127 : r);
                    if (sb(SCR_B[q * 4 + b * 32 + i]) != r) bad++;
                }
            }
            CHECK(bad == 0, "k_situ_quant8 %d mismatches", bad);
            printf("k_situ_quant8            bitwise %d elems %s\n", nb * 32 + nb, bad ? "FAIL" : "OK");
        }

        // ---- 25. k_gemv_down_reduce8: acc = sum_e w_e * dot_e (CPU order) ----
        {
            uint32_t nout = 64;
            const size_t DEXP = (size_t)nout * 18 * 68;
            size_t dbase = psegs(8 * DEXP);
            for (int e = 0; e < 8; e++) fill_q8(dbase + e * DEXP, nout, 18);
            uint32_t seld = seg(8), wtd = seg(8), accd = seg(nout);
            uint32_t hqb = seg(2304), dhb = seg(288);   // batch layout: 8*1152 codes, 8*36 d
            for (uint32_t i = 0; i < 2304; i++) ((uint32_t *)scr.data())[hqb + i] = R.u32();
            for (uint32_t i = 0; i < 288; i++) scr[dhb + i] = r16(R.f(-0.02f, 0.02f));
            std::vector<float> wts(8);
            for (uint32_t e = 0; e < 8; e++) {
                ((uint32_t *)scr.data())[seld + e] = R.u32() % 8;
                wts[e] = R.f(-1.5f, 1.5f); scr[wtd + e] = wts[e];
            }
            MoeDownArgs A{dbase, hqb, dhb, accd, wtd, 0, nout, (uint32_t)DEXP, seld};
            run("k_gemv_down_reduce8", true, (nout + 15) / 16, 1, 256, &A, sizeof(A));
            const int8_t * hq8 = (const int8_t *)(SCR_B + hqb * 4);
            std::vector<float> ref(nout);
            for (uint32_t r = 0; r < nout; r++) {
                float accv = 0.0f;
                for (uint32_t e = 0; e < 8; e++) {
                    uint32_t ex = ((uint32_t *)scr.data())[seld + e];
                    accv = accv + wts[e] * gemv_ref(pk.data() + dbase + ex * DEXP + (size_t)r * 18 * 68,
                                                    hq8 + (size_t)e * 1152, scr.data() + dhb + (size_t)e * 36, 18);
                }
                ref[r] = accv;
            }
            ok("k_gemv_down_reduce8", ref.data(), accd, nout, 1e-4);
        }

        // ---- 26. k_rms_quant vs k_rms + k_quant (codes/d/y BITWISE) ----
        {
            size_t wpk2 = psegs(kNE * 4);
            for (uint32_t i = 0; i < kNE; i++) ((float *)(pk.data() + wpk2))[i] = 1.0f + R.f(-0.1f, 0.1f);
            float eps = 1e-6f;
            for (uint32_t rows : {1u, 3u}) for (int skipy : {0, 1}) {
                uint32_t xr[4], yr[4], yr2[4], q1[4], q2[4], d1[4], d2[4];
                for (uint32_t r = 0; r < rows; r++) {
                    xr[r] = seg(kNE); for (uint32_t i = 0; i < kNE; i++) scr[xr[r] + i] = R.f(-4, 4);
                    yr[r] = seg(kNE); yr2[r] = seg(kNE);
                    q1[r] = seg(kNE); q2[r] = seg(kNE); d1[r] = seg(72); d2[r] = seg(72);
                }
                RMSArgs RA{}; RA.k = kNE; RA.rows = rows; RA.hasw = 1; RA.eps = eps;
                RmsQuantArgs QA{}; QA.k = kNE; QA.rows = rows; QA.hasw = 1; QA.eps = eps;
                for (uint32_t r = 0; r < rows; r++) {
                    RA.xr[r] = xr[r]; RA.wr[r] = wpk2; RA.yr[r] = yr[r];
                    QA.xr[r] = xr[r]; QA.wr[r] = wpk2; QA.q[r] = q2[r]; QA.d[r] = d2[r];
                    QA.yr[r] = skipy ? M64_NOFFU : yr2[r];
                }
                run("k_rms", true, rows, 1, 256, &RA, sizeof(RA));
                for (uint32_t r = 0; r < rows; r++) {
                    QuantArgs Q{yr[r], q1[r], d1[r], kNE / 32};
                    run("k_quant", false, kNE / 32, 1, 32, &Q, sizeof(Q));
                }
                run("k_rms_quant", true, rows, 1, 256, &QA, sizeof(QA));
                int bad = 0;
                for (uint32_t r = 0; r < rows; r++) {
                    const uint8_t * c1 = SCR_B + q1[r] * 4, * c2 = SCR_B + q2[r] * 4;
                    for (uint32_t i = 0; i < kNE; i++) if (c1[i] != c2[i]) bad++;
                    for (uint32_t i = 0; i < kNE / 32; i++) if (memcmp(&scr[d1[r] + i], &scr[d2[r] + i], 4) != 0) bad++;
                    if (!skipy) { for (uint32_t i = 0; i < kNE; i++) if (scr[yr[r] + i] != scr[yr2[r] + i]) bad++; }
                }
                CHECK(bad == 0, "k_rms_quant rows=%u skipy=%d: %d mismatches", rows, skipy, bad);
                printf("k_rms_quant r=%u skipy=%d   bitwise %d mismatches %s\n", rows, skipy, bad, bad ? "FAIL" : "OK");
            }
        }

        // ---- 27. k_gate_mul_quant vs k_gate_mul + k_quant (BITWISE) ----
        {
            uint32_t n = 3072, a = seg(n), gg = seg(n), q1 = seg(n / 4), q2 = seg(n / 4), d1 = seg(n / 32), d2 = seg(n / 32);
            for (uint32_t i = 0; i < n; i++) { scr[a + i] = R.f(-5, 5); scr[gg + i] = R.f(-6, 6); }
            GateMulQuantArgs FA{a, gg, q2, d2, n / 32};
            run("k_gate_mul_quant", false, n, 1, 128, &FA, sizeof(FA));   // fused first (a stays raw)
            GateMulArgs GA{a, gg, n};
            run("k_gate_mul", false, n, 1, 128, &GA, sizeof(GA));
            QuantArgs QA{a, q1, d1, n / 32};
            run("k_quant", false, n / 32, 1, 32, &QA, sizeof(QA));
            int bad = 0;
            const uint8_t * c1 = SCR_B + q1 * 4, * c2 = SCR_B + q2 * 4;
            for (uint32_t i = 0; i < n; i++) if (c1[i] != c2[i]) bad++;
            for (uint32_t i = 0; i < n / 32; i++) if (memcmp(&scr[d1 + i], &scr[d2 + i], 4) != 0) bad++;
            CHECK(bad == 0, "k_gate_mul_quant %d mismatches", bad);
            printf("k_gate_mul_quant         bitwise %d elems %s\n", n + n / 32, bad ? "FAIL" : "OK");
        }

        // ---- 28. k_gemv_q8_k3072_m2 (ROUND-2, 2 rows/warp, tail clamp) ----
        {
            const int8_t * xq = (const int8_t *)(SCR_B + sQ * 4);
            for (uint32_t nout : {64u, 52u}) {
                size_t w = psegs((size_t)nout * 48 * 68);
                fill_q8(w, nout, 48);
                uint32_t y1 = seg(64), y2 = seg(64);
                std::vector<float> ref(nout);
                for (uint32_t r = 0; r < nout; r++)
                    ref[r] = gemv_ref(pk.data() + w + (size_t)r * 48 * 68, xq, scr.data() + sD, 48);
                GemvArgs A{w, 0, sQ, sD, y1, 0, nout, 0, 0, 0};
                run("k_gemv_q8_k3072", true, (nout + 7) / 8, 1, 256, &A, sizeof(A));
                GemvArgs B2{w, 0, sQ, sD, y2, 0, nout, 0, 0, 0};
                run("k_gemv_q8_k3072_m2", true, (nout + 15) / 16, 1, 256, &B2, sizeof(B2));
                int bad = 0;
                for (uint32_t r = 0; r < nout; r++) if (scr[y1 + r] != scr[y2 + r]) bad++;
                CHECK(bad == 0, "k3072_m2 n=%u: %d bitwise mismatches vs k3072", nout, bad);
                char nm[48]; snprintf(nm, sizeof(nm), "k3072_m2 n=%u", nout);
                ok(nm, ref.data(), y2, nout, 1e-4);
            }
        }

        // ---- 29. k_gemv_gateup_batch8_m2 vs k_gemv_gateup_batch8 (BITWISE) ----
        {
            for (uint32_t nout : {64u, 6u}) {   // even nout pairs cleanly (kNF = 1152)
                const size_t EXPB = (size_t)2 * nout * 36 * 68;
                size_t gbase = psegs(8 * EXPB);
                for (int e = 0; e < 8; e++) {
                    fill_q8(gbase + e * EXPB, nout, 36);
                    fill_q8(gbase + e * EXPB + (size_t)nout * 36 * 68, nout, 36);
                }
                uint32_t selb = seg(8);
                for (uint32_t e = 0; e < 8; e++) ((uint32_t *)scr.data())[selb + e] = (e * 3 + 1) % 8;
                uint32_t g1 = seg(8 * nout), u1 = seg(8 * nout), g2 = seg(8 * nout), u2 = seg(8 * nout);
                GemvArgs A{gbase, gbase + (size_t)nout * 36 * 68, sQ, sD, g1, u1, nout, 0, (uint32_t)EXPB, selb};
                run("k_gemv_gateup_batch8", true, (8 * 2 * nout + 7) / 8, 1, 256, &A, sizeof(A));
                GemvArgs B2 = A; B2.yoff = g2; B2.yoff2 = u2;
                run("k_gemv_gateup_batch8_m2", true, (8 * nout + 7) / 8, 1, 256, &B2, sizeof(B2));
                int bad = 0;
                for (uint32_t i = 0; i < 8 * nout; i++) {
                    if (scr[g1 + i] != scr[g2 + i]) bad++;
                    if (scr[u1 + i] != scr[u2 + i]) bad++;
                }
                CHECK(bad == 0, "gateup_batch8_m2 n=%u: %d bitwise mismatches vs batch8", nout, bad);
                printf("gateup_batch8_m2 n=%-4u bitwise %d elems %s\n", nout, 16 * nout, bad ? "FAIL" : "OK");
            }
        }

        // ---- 30. k_gemv_down_reduce8_m4 (ROUND-2, 4 rows/warp, tail clamp) ----
        {
            for (uint32_t nout : {64u, 50u}) {
                const size_t DEXP = (size_t)nout * 18 * 68;
                size_t dbase = psegs(8 * DEXP);
                for (int e = 0; e < 8; e++) fill_q8(dbase + e * DEXP, nout, 18);
                uint32_t seld = seg(8), wtd = seg(8), accd = seg(nout);
                uint32_t hqb = (seg(2304) + 3u) & ~3u, dhb = seg(288);   // xq needs 16B
                for (uint32_t i = 0; i < 2304; i++) ((uint32_t *)scr.data())[hqb + i] = R.u32();
                for (uint32_t i = 0; i < 288; i++) scr[dhb + i] = r16(R.f(-0.02f, 0.02f));
                std::vector<float> wts(8);
                for (uint32_t e = 0; e < 8; e++) {
                    ((uint32_t *)scr.data())[seld + e] = R.u32() % 8;
                    wts[e] = R.f(-1.5f, 1.5f); scr[wtd + e] = wts[e];
                }
                MoeDownArgs A{dbase, hqb, dhb, accd, wtd, 0, nout, (uint32_t)DEXP, seld};
                run("k_gemv_down_reduce8_m4", true, (nout + 31) / 32, 1, 256, &A, sizeof(A));
                std::vector<float> ref(nout);
                const int8_t * hq8 = (const int8_t *)(SCR_B + hqb * 4);
                for (uint32_t r = 0; r < nout; r++) {           // CPU oracle order
                    float accv = 0.0f;
                    for (uint32_t e = 0; e < 8; e++) {
                        uint32_t ex = ((uint32_t *)scr.data())[seld + e];
                        accv = accv + wts[e] * gemv_ref(pk.data() + dbase + (size_t)ex * DEXP + (size_t)r * 18 * 68,
                                                        hq8 + (size_t)e * 1152, scr.data() + dhb + (size_t)e * 36, 18);
                    }
                    ref[r] = accv;
                }
                char nm[48]; snprintf(nm, sizeof(nm), "down_reduce8_m4 n=%u", nout);
                ok(nm, ref.data(), accd, nout, 1e-4);
            }
        }

        // ---- 31. k_gemv_q8_k2304_multi_m2 (count=6 and count=2: bitwise vs m2 + ULP-gated vs f64) ----
        {
            const int8_t * xq = (const int8_t *)(SCR_B + sQ * 4);
            auto test_multi = [&](const std::vector<uint32_t> & counts, const char * tag) {
                MultiGemvArgs M{}; M.xqoff = sQ; M.dxoff = sD; M.count = (uint32_t)counts.size();
                uint32_t tot = 0;
                std::vector<uint32_t> singles(counts.size());
                for (size_t i = 0; i < counts.size(); i++) {
                    uint32_t no = counts[i];
                    size_t w = psegs((size_t)no * 36 * 68);
                    fill_q8(w, no, 36);
                    M.woff[i] = w;
                    M.yoff[i] = seg(no);
                    singles[i] = seg(no);
                    M.nout[i] = no;
                    tot += no;
                    M.cum[i] = tot;
                }
                M.total = tot;
                run("k_gemv_q8_k2304_multi_m2", true, (tot + 15) / 16, 1, 256, &M, sizeof(M));
                std::vector<std::vector<float>> multi_out(counts.size());
                for (size_t i = 0; i < counts.size(); i++)
                    multi_out[i].assign(scr.data() + M.yoff[i], scr.data() + M.yoff[i] + counts[i]);
                for (size_t i = 0; i < counts.size(); i++) {
                    GemvArgs SA{M.woff[i], 0, sQ, sD, singles[i], 0, counts[i], 0, 0, 0};
                    run("k_gemv_q8_k2304_m2", true, (counts[i] + 15) / 16, 1, 256, &SA, sizeof(SA));
                    int bit_bad = 0, ulp_bad = 0;
                    double worst = 0;
                    for (uint32_t r = 0; r < counts[i]; r++) {
                        float got = multi_out[i][r];
                        if (got != scr[singles[i] + r]) bit_bad++;
                        auto [a, sumabs] = gemv_ref64(pk.data() + M.woff[i] + (size_t)r * 36 * 68, xq, scr.data() + sD, 36);
                        double bound = 1e-4 * std::fabs(a) + 1024.0 * 1.1920928955078125e-7 * sumabs;
                        double dev = std::fabs(a - (double)got) / (bound > 0 ? bound : 1.0);
                        if (dev > 1.0) ulp_bad++;
                        if (dev > worst) worst = dev;
                    }
                    CHECK(bit_bad == 0 && ulp_bad == 0, "%s[%zu] n=%u: bit_bad=%d ulp_bad=%d worst=%.2f",
                          tag, i, counts[i], bit_bad, ulp_bad, worst);
                    printf("%-12s[%zu] n=%-3u bitwise=%d ulp-bad=%d (worst %.2f bound) %s\n",
                           tag, i, counts[i], bit_bad, ulp_bad, worst, (bit_bad || ulp_bad) ? "FAIL" : "OK");
                }
            };
            test_multi({50u, 24u, 17u, 64u, 1u, 52u}, "multi6_m2");
            test_multi({64u, 32u}, "multi2_m2");
        }

        // ---- 32. k_gemv_down_batch8_m2 + k_down_reduce8_conv_add (nout 64/50) ----
        {
            for (uint32_t nout : {64u, 50u}) {
                const size_t DEXP = (size_t)nout * 18 * 68;
                size_t dbase = psegs(8 * DEXP);
                for (int e = 0; e < 8; e++) fill_q8(dbase + e * DEXP, nout, 18);
                uint32_t seld = seg(8), wtd = seg(8), tmpo = seg(8 * nout);
                uint32_t rso = seg(nout * 3), curo = seg(nout);
                size_t wco = psegs((size_t)nout * 16);
                uint32_t hqb = (seg(2304) + 3u) & ~3u, dhb = seg(288);
                for (uint32_t i = 0; i < 2304; i++) ((uint32_t *)scr.data())[hqb + i] = R.u32();
                for (uint32_t i = 0; i < 288; i++) scr[dhb + i] = r16(R.f(-0.02f, 0.02f));
                std::vector<float> wts(8), rs0(nout * 3), cur0(nout);
                for (uint32_t e = 0; e < 8; e++) {
                    ((uint32_t *)scr.data())[seld + e] = R.u32() % 8;
                    wts[e] = R.f(-1.5f, 1.5f); scr[wtd + e] = wts[e];
                }
                for (uint32_t i = 0; i < nout * 3; i++) rs0[i] = scr[rso + i] = R.f(-2, 2);
                for (uint32_t i = 0; i < nout; i++) cur0[i] = scr[curo + i] = R.f(-2, 2);
                for (uint32_t i = 0; i < nout * 4; i++) ((float *)(pk.data() + wco))[i] = R.f(-1, 1);

                MoeDownArgs MD{dbase, hqb, dhb, tmpo, wtd, 0, nout, (uint32_t)DEXP, seld};
                run("k_gemv_down_batch8_m2", true, (nout + 15) / 16, 8, 256, &MD, sizeof(MD));
                const int8_t * hq8 = (const int8_t *)(SCR_B + hqb * 4);
                std::vector<float> ref_xin(nout, 0.0f);
                for (uint32_t e = 0; e < 8; e++) {
                    std::vector<float> ref_e(nout);
                    uint32_t ex = ((uint32_t *)scr.data())[seld + e];
                    for (uint32_t r = 0; r < nout; r++) {
                        ref_e[r] = gemv_ref(pk.data() + dbase + (size_t)ex * DEXP + (size_t)r * 18 * 68,
                                            hq8 + (size_t)e * 1152, scr.data() + dhb + (size_t)e * 36, 18);
                        ref_xin[r] = ref_xin[r] + wts[e] * ref_e[r];
                    }
                    if (e == 0) {
                        char nm[48]; snprintf(nm, sizeof(nm), "down_batch8_m2[e0] n=%u", nout);
                        ok(nm, ref_e.data(), tmpo, nout, 1e-4);
                    }
                }
                const float sc_static = 0.4472135955f;
                DownReduceConvArgs DRC{tmpo, wtd, rso, curo, wco, nout, sc_static};
                run("k_down_reduce8_conv_add", false, nout, 1, 128, &DRC, sizeof(DRC));
                std::vector<float> ref_cur(nout), ref_rs(nout * 3);
                const float * wc = (const float *)(pk.data() + wco);
                for (uint32_t c2 = 0; c2 < nout; c2++) {
                    float s0 = rs0[c2 * 3], s1 = rs0[c2 * 3 + 1], s2 = rs0[c2 * 3 + 2];
                    float xin = ref_xin[c2];
                    float outv = s0 * wc[c2 * 4] + s1 * wc[c2 * 4 + 1] + s2 * wc[c2 * 4 + 2] + xin * wc[c2 * 4 + 3];
                    ref_cur[c2] = cur0[c2] + (xin + outv) * sc_static;
                    ref_rs[c2 * 3] = s1; ref_rs[c2 * 3 + 1] = s2; ref_rs[c2 * 3 + 2] = xin;
                }
                char nm1[48], nm2[48];
                snprintf(nm1, sizeof(nm1), "down_reduce8_conv n=%u", nout);
                snprintf(nm2, sizeof(nm2), "down_conv_rs n=%u", nout);
                ok(nm1, ref_cur.data(), curo, nout, 1e-4);
                ok(nm2, ref_rs.data(), rso, nout * 3, 1e-4);
            }
        }

        // ---- 33. k_mix_fused (nq=1/3, S=1/8, aligned float4 + unaligned scalar fallback) ----
        {
            for (int unalign : {0, 1}) for (uint32_t S : {1u, 8u}) for (uint32_t nq : {1u, 3u}) {
                uint32_t stn = (seg(S * kNE + 4) + 3u) & ~3u;
                uint32_t out = (seg(nq * kNE + 4) + 3u) & ~3u;
                size_t qpk = (psegs(nq * kNE * 4 + 16) + 15) & ~(size_t)15;
                if (unalign) { stn += 1u; out += 1u; qpk += 4u; }
                for (uint32_t i = 0; i < S * kNE; i++) scr[stn + i] = R.f(-2, 2);
                for (uint32_t i = 0; i < nq * kNE; i++) ((float *)(pk.data() + qpk))[i] = R.f(-1, 1);
                float scale = 1.0f / sqrtf((float)kNE);
                MixFusedArgs MFA{qpk, stn, out, S, nq, scale, 0};
                run("k_mix_fused", true, nq, 1, 256, &MFA, sizeof(MFA));
                std::vector<float> ref(nq * kNE);
                const float * qw = (const float *)(pk.data() + qpk);
                for (uint32_t j = 0; j < nq; j++) {
                    std::vector<double> scd(S);
                    double mx = -INFINITY, sum = 0;
                    for (uint32_t i = 0; i < S; i++) {
                        double a = 0;
                        for (uint32_t k = 0; k < kNE; k++) a += (double)scr[stn + i * kNE + k] * qw[j * kNE + k];
                        scd[i] = a * scale; mx = std::max(mx, scd[i]);
                    }
                    for (uint32_t i = 0; i < S; i++) { scd[i] = std::exp(scd[i] - mx); sum += scd[i]; }
                    for (uint32_t i = 0; i < S; i++) scd[i] /= sum;
                    for (uint32_t k = 0; k < kNE; k++) {
                        double a = 0;
                        for (uint32_t i = 0; i < S; i++) a += scd[i] * scr[stn + i * kNE + k];
                        ref[j * kNE + k] = (float)a;
                    }
                }
                char nm[48]; snprintf(nm, sizeof(nm), "mix_fused nq=%u S=%u al=%d", nq, S, unalign);
                ok(nm, ref.data(), out, nq * kNE, 1e-4);
            }
        }

        // ---- 34. R4 split control: k_ctrl_down + k_ctrl_up vs k_control (nc=1/3, aligned) ----
        for (uint32_t nc : {1u, 3u}) {
            uint32_t gcat = seg(nc * 2 * kNE); gcat = (gcat + 3u) & ~3u;   // 16B-align: float4 path
            uint32_t hh = (seg(4 * 64u) + 3u) & ~3u;
            uint32_t gb = seg(nc * kNE), fb = seg(nc * kNE);
            uint32_t cin = seg(nc * kNE), cmsg = seg(nc * kNE);
            uint32_t cout = seg(nc * kNE), cout2 = seg(nc * kNE);
            size_t wd[4], wg[4], wf[4];
            for (uint32_t c = 0; c < nc; c++) {
                wd[c] = psegs((size_t)64 * 2 * kNE * 4); wg[c] = psegs((size_t)kNE * 64 * 4); wf[c] = psegs((size_t)kNE * 64 * 4);
                for (size_t i = 0; i < 64 * 2 * kNE; i++) ((float *)(pk.data() + wd[c]))[i] = R.f(-0.021f, 0.021f);
                for (size_t i = 0; i < (size_t)kNE * 64; i++) { ((float *)(pk.data() + wg[c]))[i] = R.f(-0.12f, 0.12f); ((float *)(pk.data() + wf[c]))[i] = R.f(-0.12f, 0.12f); }
            }
            for (uint32_t i = 0; i < nc * 2 * kNE; i++) scr[gcat + i] = R.f(-1, 1);
            for (uint32_t i = 0; i < nc * kNE; i++) { scr[cin + i] = R.f(-3, 3); scr[cmsg + i] = R.f(-3, 3); }
            CtrlArgs A{};
            for (uint32_t c = 0; c < nc; c++) {
                A.in[c] = cin + c * kNE; A.msg[c] = cmsg + c * kNE; A.out[c] = cout + c * kNE;
                A.wdown[c] = wd[c]; A.wgain[c] = wg[c]; A.wforg[c] = wf[c];
                A.gb[c] = gb + c * kNE; A.fb[c] = fb + c * kNE;
            }
            A.gcat = gcat; A.eps = 1e-6f; A.grank = 64; A.ne = kNE;
            run("k_control", true, nc, 1, 1024, &A, sizeof(A));
            CtrlSplitArgs S{};
            for (uint32_t c = 0; c < nc; c++) {
                S.in[c] = cin + c * kNE; S.msg[c] = cmsg + c * kNE; S.out[c] = cout2 + c * kNE;
                S.wdown[c] = wd[c]; S.wgain[c] = wg[c]; S.wforg[c] = wf[c];
            }
            S.gcat = gcat; S.hh = hh; S.eps = 1e-6f; S.grank = 64; S.ne = kNE; S.nc = nc;
            run("k_ctrl_down", true, 64, nc, 256, &S, sizeof(S));
            run("k_ctrl_up", true, (kNE + 31) / 32, nc, 32, &S, sizeof(S));
            // Exact FP64 CPU reference (identical formula to Test 17).
            std::vector<float> ref(nc * kNE);
            for (uint32_t c = 0; c < nc; c++) {
                const float * gc = scr.data() + gcat + c * 2 * kNE;
                const float * wmd = (const float *)(pk.data() + wd[c]);
                float hhv[64];
                for (int t = 0; t < 64; t++) {
                    double a = 0;
                    for (uint32_t k = 0; k < 2 * kNE; k++) a += (double)wmd[t * 2 * kNE + k] * gc[k];
                    hhv[t] = tanhf((float)a);
                }
                const float * wmg = (const float *)(pk.data() + wg[c]), * wmf = (const float *)(pk.data() + wf[c]);
                for (uint32_t row = 0; row < kNE; row++) {
                    float g = 0, f = 0;
                    for (int t = 0; t < 64; t++) { g = std::fma(wmg[row * 64 + t], hhv[t], g); f = std::fma(wmf[row * 64 + t], hhv[t], f); }
                    float gbt = tanhf(g), fbt = tanhf(f);
                    ref[c * kNE + row] = (scr[cin + c * kNE + row] - fbt * scr[cin + c * kNE + row]) + gbt * scr[cmsg + c * kNE + row];
                }
            }
            // Both k_control and k_ctrl_down+k_ctrl_up vs the FP64 reference (Test 17 tolerances).
            const float * ctl_o = scr.data() + cout, * spl_o = scr.data() + cout2;
            double rel_c = max_rel(ref.data(), ctl_o, (size_t)nc * kNE), cos_c = cosine(ref.data(), ctl_o, (size_t)nc * kNE);
            double rel_s = max_rel(ref.data(), spl_o, (size_t)nc * kNE), cos_s = cosine(ref.data(), spl_o, (size_t)nc * kNE);
            const bool pass = rel_c < 1e-3 && cos_c > 0.9999 && rel_s < 1e-3 && cos_s > 0.9999;
            // Diagnostic only: GPU-vs-GPU delta is float4 accumulation-order noise (~1e-5..2e-4) with cos == 1.0.
            double mad = 0;
            for (size_t i = 0; i < (size_t)nc * kNE; i++) mad = std::max(mad, std::fabs((double)ctl_o[i] - spl_o[i]));
            double rel_d = max_rel(ctl_o, spl_o, (size_t)nc * kNE), cos_d = cosine(ctl_o, spl_o, (size_t)nc * kNE);
            CHECK(pass, "ctrl_split nc=%u vs_fp64 ctl rel=%.3g cos=%.8f split rel=%.3g cos=%.8f",
                  nc, rel_c, cos_c, rel_s, cos_s);
            printf("ctrl_split nc=%u    vs_fp64 ctl rel=%.3g cos=%.8f | split rel=%.3g cos=%.8f | mad(split-ctl)=%.3g rel=%.3g cos=%.8f %s\n",
                   nc, rel_c, cos_c, rel_s, cos_s, mad, rel_d, cos_d, pass ? "OK" : "FAIL");
        }

        // ---- 35. R4 router: k_router_topk (parallel SIMD) vs k_router_topk_serial ----
        {
            uint32_t pr = seg(128), selA = seg(8), wtA = seg(8), selB = seg(8), wtB = seg(8);
            size_t bias = psegs(128 * 4);
            int * seld = (int *)scr.data();
            for (int tie : {0, 1}) {
                for (uint32_t i = 0; i < 128; i++) {
                    scr[pr + i] = tie ? 0.75f : R.f(-8, 8);
                    ((float *)(pk.data() + bias))[i] = tie ? 0.0f : R.f(-0.1f, 0.1f);
                }
                RouterArgs RA{pr, bias, selA, wtA, 8.0f};
                run("k_router_topk_serial", true, 1, 1, 128, &RA, sizeof(RA));
                int refsel[8]; float refw[8];
                for (int t = 0; t < 8; t++) { refsel[t] = seld[selA + t]; refw[t] = scr[wtA + t]; }
                if (tie) for (int t = 0; t < 8; t++)
                    CHECK(refsel[t] == t, "router tie serial sel[%d]=%d (want %d)", t, refsel[t], t);
                RouterArgs RB{pr, bias, selB, wtB, 8.0f};
                run("k_router_topk", true, 1, 1, 128, &RB, sizeof(RB));
                int rbad = 0;
                for (int t = 0; t < 8; t++) {
                    if (seld[selB + t] != refsel[t]) rbad++;
                    if (!(std::fabs(scr[wtB + t] - refw[t]) < 1e-6f)) rbad++;
                    CHECK(seld[selB + t] == refsel[t], "router par sel[%d] %d vs %d", t, seld[selB + t], refsel[t]);
                    CHECK(std::fabs(scr[wtB + t] - refw[t]) < 1e-6f,
                          "router par w[%d] %.9g vs %.9g", t, scr[wtB + t], refw[t]);
                    if (tie) CHECK(seld[selB + t] == t, "router tie parallel sel[%d]=%d (want %d)", t, seld[selB + t], t);
                }
                printf("router_par tie=%d            sel+w match %s\n", tie, rbad ? "FAIL" : "OK");
            }
        }

        // ---- 36. R4: k_rms vs k_rms_quant, xr explicitly 4-aligned (vec4_in fast path, BITWISE) ----
        {
            size_t wpk4 = psegs(kNE * 4);
            for (uint32_t i = 0; i < kNE; i++) ((float *)(pk.data() + wpk4))[i] = 1.0f + R.f(-0.1f, 0.1f);
            float eps = 1e-6f;
            for (uint32_t rows : {1u, 3u}) {
                uint32_t xr[4], yr[4], yr2[4], q2[4], d2[4];
                for (uint32_t r = 0; r < rows; r++) {
                    xr[r] = (seg(kNE) + 3u) & ~3u;
                    yr[r] = (seg(kNE) + 3u) & ~3u;
                    yr2[r] = (seg(kNE) + 3u) & ~3u;
                    q2[r] = seg(kNE / 4); d2[r] = seg(72);   // q is uchar[k]: kNE bytes = kNE/4 floats
                    CHECK((xr[r] & 3u) == 0u && (yr[r] & 3u) == 0u, "rms_vec4 r=%u not 4-aligned", r);
                    for (uint32_t i = 0; i < kNE; i++) scr[xr[r] + i] = R.f(-4, 4);
                }
                RMSArgs RA{}; RA.k = kNE; RA.rows = rows; RA.hasw = 1; RA.eps = eps;
                RmsQuantArgs QA{}; QA.k = kNE; QA.rows = rows; QA.hasw = 1; QA.eps = eps;
                for (uint32_t r = 0; r < rows; r++) {
                    RA.xr[r] = xr[r]; RA.wr[r] = wpk4; RA.yr[r] = yr[r];
                    QA.xr[r] = xr[r]; QA.wr[r] = wpk4; QA.q[r] = q2[r]; QA.d[r] = d2[r]; QA.yr[r] = yr2[r];
                }
                run("k_rms", true, rows, 1, 256, &RA, sizeof(RA));          // vec4_in + vec4 out
                run("k_rms_quant", true, rows, 1, 256, &QA, sizeof(QA));    // vec4_in + scalar y/xn
                int bad = 0;
                for (uint32_t r = 0; r < rows; r++)
                    for (uint32_t i = 0; i < kNE; i++) if (scr[yr[r] + i] != scr[yr2[r] + i]) bad++;
                CHECK(bad == 0, "k_rms vs k_rms_quant vec4 r=%u: %d bitwise mismatches", rows, bad);
                printf("k_rms_vec4 r=%u            bitwise %d mismatches %s\n", rows, bad, bad ? "FAIL" : "OK");
            }
        }

        // ---- 37 & 38. FP16 KV cache (k_kv_write_f16 + k_attn_head_f16) ----
        // Test 37: full-attention path at tau=1.0 and tau=1.25 (pos=19).
        // Test 38: SWA ring wrap-around over 1100 positions (pos=0..1099,
        //          ring_cap=1025, n_swa=1024 -> attends j=75..1099 only, and
        //          dist==1024 must NOT read relbias beyond M64_EXT=1024).
        {
            const uint32_t RING_CAP = 1025u;
            const uint32_t N_WRITE = 1100u;
            const size_t kv_half_elems = (size_t)2u * RING_CAP * kKVD;
            id<MTLBuffer> bKv = [rt.device() newBufferWithLength:kv_half_elems * sizeof(uint16_t)
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> bStg = [rt.device() newBufferWithLength:(size_t)N_WRITE * 2u * kKVD * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            CHECK(bKv != nil && bStg != nil, "alloc kv/stage buffers");

            uint32_t sQh = seg(kNH * kDH), sRout = seg(kNH * kNREL), sSink = seg(kNH), sOut = seg(kNH * kDH);
            size_t rb = psegs(1200u * kNREL * sizeof(float));
            for (uint32_t i = 0; i < kNH * kDH; i++) scr[sQh + i] = R.f(-1.5f, 1.5f);
            for (uint32_t i = 0; i < kNH * kNREL; i++) scr[sRout + i] = R.f(-0.5f, 0.5f);
            for (uint32_t i = 0; i < kNH; i++) scr[sSink + i] = R.f(-1.5f, 1.5f);
            for (uint32_t d = 0; d < 1200u * kNREL; d++) {
                // Sentinel values beyond d >= 1024 catches out-of-extent relbias reads.
                ((float *)(pk.data() + rb))[d] = (d < 1024u * kNREL) ? R.f(-0.5f, 0.5f) : 100.0f;
            }
            float * stg = (float *)bStg.contents;
            for (uint32_t j = 0; j < N_WRITE * 2u * kKVD; j++) stg[j] = R.f(-2.0f, 2.0f);

            const float scale = 1.0f / sqrtf((float)kDH);
            id<MTLComputePipelineState> pWrite = rt.pipeline("k_kv_write_f16");
            id<MTLComputePipelineState> pAttn = rt.pipeline("k_attn_head_f16");
            CHECK(pWrite != nil && pAttn != nil, "fp16 kv pipelines missing");

            auto run_write_range = [&](uint32_t count, uint32_t ring_cap) {
                id<MTLCommandBuffer> cmd = [rt.queue() commandBuffer];
                id<MTLComputeCommandEncoder> en = [cmd computeCommandEncoder];
                [en setComputePipelineState:pWrite];
                for (uint32_t j = 0; j < count; j++) {
                    uint32_t slot = j % ring_cap;
                    KVWriteF16Args kw{
                        j * 2u * kKVD, j * 2u * kKVD + kKVD, kKVD, 0u,
                        (uint64_t)slot * kKVD,
                        (uint64_t)ring_cap * kKVD + (uint64_t)slot * kKVD,
                    };
                    [en setBuffer:bStg offset:0 atIndex:0];
                    [en setBytes:&kw length:sizeof(kw) atIndex:1];
                    [en setBuffer:bKv offset:0 atIndex:2];
                    [en dispatchThreads:MTLSizeMake(kKVD, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                CHECK(cmd.status == MTLCommandBufferStatusCompleted, "k_kv_write_f16 status=%ld", (long)cmd.status);
            };

            auto run_attn_f16 = [&](const AttnF16Args & A) {
                memcpy(bScr.contents, scr.data(), SCR_F * 4);
                memcpy(bPk.contents, pk.data(), PK_B);
                id<MTLCommandBuffer> cmd = [rt.queue() commandBuffer];
                id<MTLComputeCommandEncoder> en = [cmd computeCommandEncoder];
                [en setComputePipelineState:pAttn];
                [en setBuffer:bScr offset:0 atIndex:0];
                [en setBytes:&A length:sizeof(A) atIndex:1];
                [en setBuffer:bPk offset:0 atIndex:2];
                [en setBuffer:bKv offset:0 atIndex:3];
                [en dispatchThreadgroups:MTLSizeMake(kNH, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                CHECK(cmd.status == MTLCommandBufferStatusCompleted, "k_attn_head_f16 status=%ld", (long)cmd.status);
                memcpy(scr.data(), bScr.contents, SCR_F * 4);
            };

            auto ref_attn_f16 = [&](uint32_t pos, uint32_t is_swa, uint32_t n_swa, float tau,
                                    std::vector<float> & ref) {
                const uint32_t j_start = (is_swa && pos > n_swa) ? (pos - n_swa) : 0u;
                for (uint32_t h = 0; h < kNH; h++) {
                    const float * q = scr.data() + sQh + h * kDH;
                    const float * ro = scr.data() + sRout + h * kNREL;
                    const float snk = scr[sSink + h];
                    const uint32_t kvh = h / 3u;
                    double oh[128] = {0}, M = -INFINITY, S = 0.0;
                    for (uint32_t j = j_start; j <= pos; j++) {
                        const float * kj = stg + j * 2u * kKVD + kvh * kDH;
                        const float * vj = stg + j * 2u * kKVD + kKVD + kvh * kDH;
                        double d = 0.0;
                        for (int i = 0; i < 128; i++) {
                            double qv = (double)r16(q[i] * tau);
                            double kv = (double)r16(kj[i]);
                            d += qv * kv;
                        }
                        uint32_t dist = pos - j;
                        double bv = 0.0;
                        if (dist < 1024u) {
                            const float * br = ((const float *)(pk.data() + rb)) + dist * 16u;
                            for (int i = 0; i < 16; i++) bv += (double)br[i] * ((double)ro[i] * (double)tau);
                        }
                        double s = d * (double)scale + (double)r16((float)(bv - (double)snk));
                        double ms = 1.0, vs = 1.0;
                        if (s > M) {
                            ms = std::exp(M - s);
                            for (int i = 0; i < 128; i++) oh[i] *= ms;
                            M = s;
                        } else {
                            vs = std::exp(s - M);
                        }
                        for (int i = 0; i < 128; i++) oh[i] += vs * (double)r16(vj[i]);
                        S = S * ms + vs;
                    }
                    {
                        double ms = 1.0, vs = 1.0;
                        if (0.0 > M) { ms = std::exp(M); for (int i = 0; i < 128; i++) oh[i] *= ms; M = 0.0; }
                        else { vs = std::exp(0.0 - M); }
                        S = S * ms + vs;
                    }
                    double inv = 1.0 / S;
                    for (int i = 0; i < 128; i++) ref[h * kDH + i] = (float)(oh[i] * inv);
                }
            };

            // Test 37: full attention (pos=19, tau=1.0 and tau=1.25)
            run_write_range(20u, 20u);
            for (float tau : {1.0f, 1.25f}) {
                AttnF16Args aa{};
                aa.relbiaspack = rb;
                aa.kvkoff = 0u; aa.kvvoff = 20u * kKVD;
                aa.qoff = sQh; aa.rout = sRout; aa.sinkoff = sSink; aa.outoff = sOut;
                aa.pos = 19u; aa.is_swa = 0u; aa.n_swa = 1024u; aa.ring_cap = 20u;
                aa.scale = scale; aa.tau = tau;
                run_attn_f16(aa);
                std::vector<float> ref(kNH * kDH);
                ref_attn_f16(19u, 0u, 1024u, tau, ref);
                const float * got = scr.data() + sOut;
                double rel = max_rel(ref.data(), got, kNH * kDH), cosv = cosine(ref.data(), got, kNH * kDH);
                CHECK(rel < 1e-3 && cosv > 0.99999, "attn_f16 full tau=%.2f rel=%.3g cos=%.8f", tau, rel, cosv);
                printf("attn_f16 full tau=%.2f  rel=%.3g cos=%.8f %s\n",
                       tau, rel, cosv, (rel < 1e-3 && cosv > 0.99999) ? "OK" : "FAIL");
            }

            // Test 38: SWA ring wrap-around (1100 writes into ring_cap=1025, n_swa=1024)
            run_write_range(N_WRITE, RING_CAP);
            {
                AttnF16Args aa{};
                aa.relbiaspack = rb;
                aa.kvkoff = 0u; aa.kvvoff = (uint64_t)RING_CAP * kKVD;
                aa.qoff = sQh; aa.rout = sRout; aa.sinkoff = sSink; aa.outoff = sOut;
                aa.pos = N_WRITE - 1u; aa.is_swa = 1u; aa.n_swa = 1024u; aa.ring_cap = RING_CAP;
                aa.scale = scale; aa.tau = 1.0f;
                run_attn_f16(aa);
                std::vector<float> ref(kNH * kDH);
                ref_attn_f16(N_WRITE - 1u, 1u, 1024u, 1.0f, ref);
                const float * got = scr.data() + sOut;
                double rel = max_rel(ref.data(), got, kNH * kDH), cosv = cosine(ref.data(), got, kNH * kDH), mad = 0;
                for (uint32_t i = 0; i < kNH * kDH; i++) mad = std::max(mad, std::fabs((double)ref[i] - got[i]));
                const bool pass = ((rel < 1e-3 || mad < 1e-5) && cosv > 0.99999);
                CHECK(pass, "attn_f16 swa_ring rel=%.3g mad=%.3g cos=%.8f", rel, mad, cosv);
                printf("attn_f16 swa_ring p=1099 rel=%.3g mad=%.3g cos=%.8f %s\n",
                       rel, mad, cosv, pass ? "OK" : "FAIL");
            }

            // ---- 39. Q8_0 KV cache (Milestone 5A): k_kv_write_q8 byte-exact
            // 136 B/head encoding vs a host Q8 mirror (id = 127/amax, codes
            // rint(x*id) clamp +-127, stored d = half(amax*(1/127)) — the
            // kernel's EXACT arithmetic, which differs by 1 f32-ulp from
            // amax/127), then k_attn_head_q8 vs the FP16 oracle: Q8 rounding
            // noise dominates so the gate is isfinite + cos >= 0.999.
            {
                id<MTLBuffer> bKvQ8 = [rt.device() newBufferWithLength:(size_t)2u * RING_CAP * 1088u
                                                               options:MTLResourceStorageModeShared];
                CHECK(bKvQ8 != nil, "q8 kv buffer alloc");
                const uint64_t kvv_u16_full = 20u * 544u;             // half-unit V base, ring_cap=20
                const uint64_t kvv_u16_swa  = (uint64_t)RING_CAP * 544u;

                auto q8_mirror = [&](const float * x128, uint8_t * at136) {
                    for (uint32_t blk = 0; blk < 4u; blk++) {
                        const float * xb = x128 + blk * 32u;
                        float amax = 0.0f;
                        for (uint32_t i = 0; i < 32u; i++) amax = std::max(amax, std::fabs(xb[i]));
                        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                        for (uint32_t i = 0; i < 32u; i++) {
                            int v = (int)std::rintf(xb[i] * id);
                            v = v < -127 ? -127 : (v > 127 ? 127 : v);
                            at136[blk * 32u + i] = (uint8_t)(int8_t)v;
                        }
                        const __fp16 hd = (__fp16)(amax * (1.0f / 127.0f));
                        uint16_t hb; memcpy(&hb, &hd, 2);
                        at136[128u + blk * 2u] = (uint8_t)hb;
                        at136[129u + blk * 2u] = (uint8_t)(hb >> 8);
                    }
                };

                id<MTLComputePipelineState> pWriteQ8 = rt.pipeline("k_kv_write_q8");
                CHECK(pWriteQ8 != nil, "k_kv_write_q8 pipeline missing");

                auto run_write_q8_range = [&](uint32_t count, uint32_t ring_cap) {
                    if (!pWriteQ8 || !bKvQ8) { CHECK(false, "kv_write_q8 prerequisites"); return; }
                    id<MTLCommandBuffer> cmd = [rt.queue() commandBuffer];
                    id<MTLComputeCommandEncoder> en = [cmd computeCommandEncoder];
                    [en setComputePipelineState:pWriteQ8];
                    for (uint32_t j = 0; j < count; j++) {
                        uint32_t slot = j % ring_cap;
                        KVWriteF16Args kw{
                            j * 2u * kKVD, j * 2u * kKVD + kKVD, kKVD, 0u,
                            (uint64_t)slot * 544u,
                            (uint64_t)ring_cap * 544u + (uint64_t)slot * 544u,
                        };
                        [en setBuffer:bStg offset:0 atIndex:0];
                        [en setBytes:&kw length:sizeof(kw) atIndex:1];
                        [en setBuffer:bKvQ8 offset:0 atIndex:2];
                        [en dispatchThreadgroups:MTLSizeMake(8, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }
                    [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                    CHECK(cmd.status == MTLCommandBufferStatusCompleted,
                          "k_kv_write_q8 status=%ld", (long)cmd.status);
                    int bad = 0;
                    const uint8_t * dev = (const uint8_t *)bKvQ8.contents;
                    const uint64_t vbase_u16 = (uint64_t)ring_cap * 544u;
                    uint8_t want[136];
                    for (uint32_t j = 0; j < count; j++) {
                        const uint32_t slot = j % ring_cap;
                        if (j + ring_cap < count) continue; // overwritten by a later writer
                        for (int s = 0; s < 2; s++) {
                            const uint64_t byte_base =
                                ((uint64_t)(s ? vbase_u16 : 0u) + (uint64_t)slot * 544u) * 2u;
                            for (uint32_t kvh = 0; kvh < 8u; kvh++) {
                                q8_mirror(stg + j * 2u * kKVD + (uint32_t)(s ? kKVD : 0) + kvh * kDH, want);
                                if (memcmp(dev + byte_base + kvh * 136u, want, 136u) != 0) bad++;
                            }
                        }
                    }
                    CHECK(bad == 0, "k_kv_write_q8 rc=%u: %d head encodings vs host mirror", count, bad);
                    printf("kv_write_q8 rc=%-5u       %d head mismatches (136B byte-exact) %s\n",
                           ring_cap, bad, bad ? "FAIL" : "OK");
                };

                id<MTLComputePipelineState> pAttnQ8 = rt.pipeline("k_attn_head_q8");
                CHECK(pAttnQ8 != nil, "k_attn_head_q8 pipeline missing");
                auto run_attn_q8 = [&](const AttnF16Args & A) {
                    if (!pAttnQ8 || !bKvQ8) { CHECK(false, "attn_q8 prerequisites"); return; }
                    memcpy(bScr.contents, scr.data(), SCR_F * 4);
                    memcpy(bPk.contents, pk.data(), PK_B);
                    id<MTLCommandBuffer> cmd = [rt.queue() commandBuffer];
                    id<MTLComputeCommandEncoder> en = [cmd computeCommandEncoder];
                    [en setComputePipelineState:pAttnQ8];
                    [en setBuffer:bScr offset:0 atIndex:0];
                    [en setBytes:&A length:sizeof(A) atIndex:1];
                    [en setBuffer:bPk offset:0 atIndex:2];
                    [en setBuffer:bKvQ8 offset:0 atIndex:3];
                    [en dispatchThreadgroups:MTLSizeMake(kNH, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [en endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                    CHECK(cmd.status == MTLCommandBufferStatusCompleted,
                          "k_attn_head_q8 status=%ld", (long)cmd.status);
                    memcpy(scr.data(), bScr.contents, SCR_F * 4);
                };

                auto q8_gate = [&](const char * nm, const std::vector<float> & ref) {
                    const float * got = scr.data() + sOut;
                    bool fin = true;
                    for (uint32_t i = 0; i < kNH * kDH; i++) if (!std::isfinite(got[i])) fin = false;
                    double rel = max_rel(ref.data(), got, kNH * kDH), cosv = cosine(ref.data(), got, kNH * kDH);
                    const bool pass = fin && cosv >= 0.999; // rel printed for visibility only
                    CHECK(pass, "%s finite=%d rel=%.3g cos=%.8f", nm, (int)fin, rel, cosv);
                    printf("%-26s finite=%d rel=%.3g cos=%.8f %s\n",
                           nm, (int)fin, rel, cosv, pass ? "OK" : "FAIL");
                };

                // 39a: full attention (pos=19) at tau=1.0 / 1.25
                run_write_q8_range(20u, 20u);
                for (float tau : {1.0f, 1.25f}) {
                    AttnF16Args aa{};
                    aa.relbiaspack = rb;
                    aa.kvkoff = 0u; aa.kvvoff = kvv_u16_full;
                    aa.qoff = sQh; aa.rout = sRout; aa.sinkoff = sSink; aa.outoff = sOut;
                    aa.pos = 19u; aa.is_swa = 0u; aa.n_swa = 1024u; aa.ring_cap = 20u;
                    aa.scale = scale; aa.tau = tau;
                    run_attn_q8(aa);
                    std::vector<float> ref(kNH * kDH);
                    ref_attn_f16(19u, 0u, 1024u, tau, ref);
                    char nm[48];
                    snprintf(nm, sizeof(nm), "attn_q8 full tau=%.2f", tau);
                    q8_gate(nm, ref);
                }
                // 39b: SWA ring wrap-around (1100 writes into ring_cap=1025,
                // pos=1099 -> attends j=75..1099 through the wrapped slots)
                run_write_q8_range(N_WRITE, RING_CAP);
                {
                    AttnF16Args aa{};
                    aa.relbiaspack = rb;
                    aa.kvkoff = 0u; aa.kvvoff = kvv_u16_swa;
                    aa.qoff = sQh; aa.rout = sRout; aa.sinkoff = sSink; aa.outoff = sOut;
                    aa.pos = N_WRITE - 1u; aa.is_swa = 1u; aa.n_swa = 1024u; aa.ring_cap = RING_CAP;
                    aa.scale = scale; aa.tau = 1.0f;
                    run_attn_q8(aa);
                    std::vector<float> ref(kNH * kDH);
                    ref_attn_f16(N_WRITE - 1u, 1u, 1024u, 1.0f, ref);
                    q8_gate("attn_q8 swa_ring p=1099", ref);
                }
                bKvQ8 = nil;
            }
        }

        // ============ Milestone 4B: decode Q4_K / Q5_0 / Q5_1 (FP32-input MoE) ============
        // Host oracles dequantize bit-exactly per engine/src/dequant.hpp (q4_K get_scale_min_k4
        // + interleaved qs; q5_0/qh, q5_1/m) and accumulate in FP64. The kernels reduce over a
        // 32-lane simd_sum tree, so the gate is the 1024-ulp fp64 bound (same policy as test 23).
        {
            auto hbits = [](float x) { __fp16 h = (__fp16)x; uint16_t v; memcpy(&v, &h, 2); return v; };
            auto w16 = [&](size_t off, uint16_t hv) { pk[off] = (uint8_t)hv; pk[off + 1] = (uint8_t)(hv >> 8); };
            auto w32 = [&](size_t off, uint32_t v) { pk[off] = (uint8_t)v; pk[off + 1] = (uint8_t)(v >> 8); pk[off + 2] = (uint8_t)(v >> 16); pk[off + 3] = (uint8_t)(v >> 24); };
            const uint32_t NB4 = 9u, NO4 = 1152u;                    // gateup: 9 Q4_K blocks/row, Inkling nout
            const uint32_t NB5 = 36u, NO5 = 2304u;                   // down:   36 Q5 blocks/row, Inkling nout

            // ---- q4_K writer: 256 values -> one 144 B block (d/dmin/scales/qs per dequant.hpp layout) ----
            auto fill_q4k = [&](size_t off, const float * v, int tag) {
                float bmn[8], bmx[8];
                for (int j = 0; j < 8; j++) {
                    bmn[j] = v[j * 32]; bmx[j] = v[j * 32];
                    for (int i = 1; i < 32; i++) { float x = v[j * 32 + i]; if (x < bmn[j]) bmn[j] = x; if (x > bmx[j]) bmx[j] = x; }
                }
                float dmax = 0.0f;
                for (int j = 0; j < 8; j++) dmax = std::max(dmax, (bmx[j] - bmn[j]) / 15.0f);
                float dval = dmax > 0.0f ? dmax : 1.0f;
                float dmval = (tag & 1) ? R.f(-0.0003f, 0.0003f) : 0.0f;
                w16(off, hbits(dval)); w16(off + 2, hbits(dmval));
                int sc8[8], mn8[8];
                for (int j = 0; j < 8; j++) {
                    double s = dmval != 0.0f ? std::rint((double)(bmn[j] / dmval)) : 0.0;
                    // sc8 <=15: scales fit the 4-bit word slot the decoders read (realistic q4_K)
                    sc8[j] = (int)std::max(0.0, std::min(15.0, std::rint((double)((bmx[j] - bmn[j]) / 15.0f / dval))));
                    mn8[j] = (int)std::max(0.0, std::min(15.0, s));   // <16 keeps the q[j-0]-quirk decoders bit-agreeing (see sm4k)
                }
                uint8_t scb[12];
                memset(scb, 0, 12);
                // Writer <-> dequant.hpp:80-87 decoder, valid because sc8/mn8 <= 15 here:
                //   j<4:  d = scb[j]&63, m = scb[j+4]&63
                //   j>=4: d = scb[j+4]&0xF (hi2 bits of scb[j-4] = 0), m = scb[j+4]>>4 (hi2 bits of scb[j] = 0)
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
                for (int j = 0; j < 12; j++) pk[off + 4 + (size_t)j] = scb[j];
                for (int c = 0; c < 4; c++) for (int l = 0; l < 32; l++)
                    pk[off + 16 + (size_t)c * 32 + l] = (uint8_t)(code[c * 64 + l] | (code[c * 64 + 32 + l] << 4));
            };
            // ---- q5 writer: 32 values -> one 22 B (Q5_0) / 24 B (Q5_1) block ----
            auto fill_q5 = [&](size_t off, const float * v, bool q51) {
                float mn = v[0], mx = v[0];
                for (int i = 1; i < 32; i++) { if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i]; }
                float d = (mx - mn) / 31.0f, m = q51 ? mn : 0.0f;
                if (d == 0.0f) d = 1.0f;
                w16(off, hbits(d));
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
                if (q51) w16(off + 2, hbits(m));
                w32(off + (q51 ? 4 : 2), qh);
                for (int i = 0; i < 16; i++) pk[off + (q51 ? 8 : 6) + (size_t)i] = qs[i];
            };
            // ---- FP64 oracles (dequant math mirrors dequant.hpp references line-for-line) ----
            auto sm4k = [](const uint8_t * q, int j, double & D, double & M) {
                if (j < 4) { D = q[j] & 63; M = q[j + 4] & 63; }
                else { D = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); M = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
            };
            struct Dot { double a, sab; };
            auto dot_q4k = [&](size_t row, const float * x) {
                double a = 0, sa = 0;
                for (int s = 0; s < 36; s++) {
                    const uint8_t * p = pk.data() + row + (size_t)(s >> 2) * 144;
                    double d = f16le(p), dm = f16le(p + 2), d1, m1, d2, m2;
                    sm4k(p + 4, 2 * (s & 3), d1, m1); sm4k(p + 4, 2 * (s & 3) + 1, d2, m2);
                    d1 *= d; m1 *= dm; d2 *= d; m2 *= dm;
                    for (int l = 0; l < 32; l++) {
                        uint8_t qv = p[16 + (s & 3) * 32 + l];
                        double w0 = d1 * (qv & 0xF) - m1, w1 = d2 * (qv >> 4) - m2;
                        a += w0 * (double)x[s * 64 + l]; sa += std::fabs(w0) * std::fabs(x[s * 64 + l]);
                        a += w1 * (double)x[s * 64 + 32 + l]; sa += std::fabs(w1) * std::fabs(x[s * 64 + 32 + l]);
                    }
                }
                return Dot{a, sa};
            };
            auto dot_q5 = [&](size_t row, const float * x, bool q51) {
                double a = 0, sa = 0;
                for (int b = 0; b < 36; b++) {
                    const uint8_t * p = pk.data() + row + (size_t)b * (q51 ? 24 : 22);
                    double d = f16le(p), m = q51 ? f16le(p + 2) : 0.0;
                    uint32_t qh = (uint32_t)p[q51 ? 4 : 2] | ((uint32_t)p[q51 ? 5 : 3] << 8) | ((uint32_t)p[q51 ? 6 : 4] << 16) | ((uint32_t)p[q51 ? 7 : 5] << 24);
                    for (int j = 0; j < 16; j++) {
                        uint8_t qv = p[(q51 ? 8 : 6) + j];
                        int x0 = (qv & 0xF) | (((qh >> j) << 4) & 0x10), x1 = (qv >> 4) | ((qh >> (j + 12)) & 0x10);
                        double w0, w1;
                        if (q51) { w0 = x0 * d + m; w1 = x1 * d + m; }
                        else { w0 = (x0 - 16) * d; w1 = (x1 - 16) * d; }
                        a += w0 * (double)x[b * 32 + j]; sa += std::fabs(w0) * std::fabs(x[b * 32 + j]);
                        a += w1 * (double)x[b * 32 + 16 + j]; sa += std::fabs(w1) * std::fabs(x[b * 32 + 16 + j]);
                    }
                }
                return Dot{a, sa};
            };
            const double EPSF = 1.1920928955078125e-7;
            auto gate = [&](const char * name, const std::vector<Dot> & ref, uint32_t off, size_t n) {
                const float * got = scr.data() + off;
                double worst = 0.0; size_t widx = 0; size_t bad = 0;
                for (size_t i = 0; i < n; i++) {
                    double bnd = 1e-4 * std::fabs(ref[i].a) + 1024.0 * EPSF * ref[i].sab;
                    double dl = std::fabs((double)got[i] - ref[i].a);
                    if (bnd == 0.0) { if (got[i] != 0.0f) bad++; continue; }
                    if (dl / bnd > worst) { worst = dl / bnd; widx = i; }
                    if (dl > bnd) bad++;
                }
                CHECK(bad == 0, "%s %zu/%zu over 1024-ulp bound (worst %.1f @ %zu)", name, bad, n, worst, widx);
                if (bad && getenv("M64_DUMP")) {
                    for (size_t i = 0; i < n; i++) {
                        double bnd = 1e-4 * std::fabs(ref[i].a) + 1024.0 * EPSF * ref[i].sab;
                        if (std::fabs((double)got[i] - ref[i].a) > bnd) {
                            printf("DUMP %s i=%zu ref=%.9g got=%.9g\n", name, i, ref[i].a, got[i]);
                            fflush(stdout);
                            break;
                        }
                    }
                }
                printf("%-38s ulp-gated %zu rows %s (worst %.2f bound)\n", name, n, bad ? "FAIL" : "OK", worst);
            };

            uint32_t ssel = seg(8u);
            uint32_t * sel = (uint32_t *)(scr.data() + ssel);
            const uint32_t expids[8] = {2300u, 0u, 7u, 123u, 8u, 255u, 1400u, 1401u};
            const uint32_t modsel8[8] = {4u, 0u, 7u, 3u, 0u, 7u, 0u, 1u};   // expids[e] % 8
            for (uint32_t e = 0; e < 8u; e++) if (expids[e] % 8u != modsel8[e]) { fprintf(stderr, "modsel8 mismatch\n"); return 2; }
            for (uint32_t e = 0; e < 8u; e++) sel[e] = e;   // direct mode: blob index = sel (only 8 blobs filled)
            uint32_t swtab = seg(1024u);
            uint32_t * wt = (uint32_t *)(scr.data() + swtab);
            auto set_sel = [&](const uint32_t * v) { for (uint32_t e = 0; e < 8u; e++) sel[e] = v[e]; };

            uint32_t sx = seg(2304u), sh = seg(8u * 1152u);
            float * x = scr.data() + sx, * h = scr.data() + sh;
            for (uint32_t i = 0; i < 2304u; i++) x[i] = R.f(-3.0f, 3.0f);
            // per-expert h rows (k_gemv_down_reduce8_q5_* consumes SCR[hoff + e*k])
            for (uint32_t i = 0; i < 8u * 1152u; i++) h[i] = R.f(-2.0f, 2.0f);

            size_t wgt = psegs((size_t)8 * NO4 * NB4 * 144), wup = psegs((size_t)8 * NO4 * NB4 * 144);
            float v4[256], v5[32];
            for (uint32_t e = 0; e < 8u; e++) for (uint32_t r = 0; r < NO4; r++) for (uint32_t b = 0; b < NB4; b++) {
                for (int i = 0; i < 256; i++) v4[i] = R.f(-0.4f, 0.4f);
                fill_q4k(wgt + ((size_t)e * NO4 + r) * NB4 * 144 + (size_t)b * 144, v4, (int)((e + r + b) & 1));
                for (int i = 0; i < 256; i++) v4[i] = R.f(-0.4f, 0.4f);
                fill_q4k(wup + ((size_t)e * NO4 + r) * NB4 * 144 + (size_t)b * 144, v4, (int)(r & 1));
            }
            uint32_t sgy = seg(8u * NO4), suy = seg(8u * NO4);

            // Test 39: k_gemv_gateup_batch8_q4k_f32in, Inkling dims (k=2304, nout=1152), direct
            {
                MoeGateUpQ4KF32Args A{};
                A.woff = wgt; A.woff2 = wup; A.xoff = sx; A.yoff = sgy; A.yoff2 = suy;
                A.seloff = ssel; A.wtab_off = 0u; A.nout = NO4; A.k = 2304u; A.stride = NO4 * NB4 * 144u;
                run("k_gemv_gateup_batch8_q4k_f32in", true, 8u * 2u * NO4 / 8u, 1, 256, &A, sizeof(A));
                std::vector<Dot> rg(8 * NO4), ru(8 * NO4);
                for (uint32_t e = 0; e < 8u; e++) for (uint32_t r = 0; r < NO4; r++) {
                    rg[e * NO4 + r] = dot_q4k(wgt + (size_t)e * (NO4 * NB4 * 144) + (size_t)r * NB4 * 144, x);
                    ru[e * NO4 + r] = dot_q4k(wup + (size_t)e * (NO4 * NB4 * 144) + (size_t)r * NB4 * 144, x);
                }
                gate("gateup_batch8_q4k_f32in gate", rg, sgy, 8 * NO4);
                gate("gateup_batch8_q4k_f32in up", ru, suy, 8 * NO4);
            }

            // Test 40: same output via wtab arena indirection -> must be BITWISE equal (same
            // blobs reached through expert-id -> slot mapping with out-of-order slots)
            {
                uint32_t modsel[8];
                for (uint32_t e = 0; e < 8u; e++) { wt[e] = e; modsel[e] = expids[e] % 8u; }
                for (uint32_t s = 8u; s < 1024u; s++) wt[s] = 0u;
                for (uint32_t e = 0; e < 8u; e++) wt[expids[e]] = modsel[e];
                set_sel(expids);
                MoeGateUpQ4KF32Args A{};
                A.woff = wgt; A.woff2 = wup; A.xoff = sx; A.yoff = sgy; A.yoff2 = suy;
                A.seloff = ssel; A.wtab_off = swtab; A.nout = NO4; A.k = 2304u; A.stride = NO4 * NB4 * 144u;
                run("k_gemv_gateup_batch8_q4k_f32in", true, 8u * 2u * NO4 / 8u, 1, 256, &A, sizeof(A));
                set_sel(modsel);
                MoeGateUpQ4KF32Args B = A; B.wtab_off = 0u;
                uint32_t sgy2 = seg(8u * NO4), suy2 = seg(8u * NO4);
                B.yoff = sgy2; B.yoff2 = suy2;
                run("k_gemv_gateup_batch8_q4k_f32in", true, 8u * 2u * NO4 / 8u, 1, 256, &B, sizeof(B));
                int bad = 0;
                for (uint32_t e = 0; e < 8u; e++) for (uint32_t r = 0; r < NO4; r++) {
                    if (scr[sgy + e * NO4 + r] != scr[sgy2 + e * NO4 + r]) bad++;
                    if (scr[suy + e * NO4 + r] != scr[suy2 + e * NO4 + r]) bad++;
                }
                CHECK(bad == 0, "gateup q4k wtab vs direct %d bitwise mismatches", bad);
                printf("%-38s wtab == direct bitwise %s\n", "gateup_batch8_q4k_f32in wtab", bad ? "FAIL" : "OK");
            }

            // Test 41: edge nout 55/173 with wtab (small 16-blob arena: slot count bounded by pool)
            {
                const uint32_t NE = 16u;
                size_t eg = psegs((size_t)NE * 64 * NB4 * 144), eu = psegs((size_t)NE * 64 * NB4 * 144);
                for (uint32_t s = 0; s < NE; s++) for (uint32_t r = 0; r < 64; r++) for (uint32_t b = 0; b < NB4; b++) {
                    for (int i = 0; i < 256; i++) v4[i] = R.f(-0.4f, 0.4f);
                    fill_q4k(eg + ((size_t)s * 64 + r) * NB4 * 144 + (size_t)b * 144, v4, (int)(r & 1));
                    for (int i = 0; i < 256; i++) v4[i] = R.f(-0.4f, 0.4f);
                    fill_q4k(eu + ((size_t)s * 64 + r) * NB4 * 144 + (size_t)b * 144, v4, 0);
                }
                uint32_t slots[8] = {15u, 0u, 3u, 15u, 1u, 7u, 7u, 14u};
                for (uint32_t s = 0; s < 1024u; s++) wt[s] = 0u;
                for (uint32_t e = 0; e < 8u; e++) wt[expids[e]] = slots[e];
                set_sel(expids);   // wtab path: exp_id -> slot via wt[]
                for (uint32_t nout : {55u, 173u}) {
                    uint32_t ey = seg(8u * nout), euY = seg(8u * nout);
                    MoeGateUpQ4KF32Args A{};
                    A.woff = eg; A.woff2 = eu; A.xoff = sx; A.yoff = ey; A.yoff2 = euY;
                    A.seloff = ssel; A.wtab_off = swtab; A.nout = nout; A.k = 2304u; A.stride = 64 * NB4 * 144u;
                    run("k_gemv_gateup_batch8_q4k_f32in", true, (8u * 2u * nout + 7u) / 8u, 1, 256, &A, sizeof(A));
                    std::vector<Dot> rg(8 * nout), ru(8 * nout);
                    for (uint32_t e = 0; e < 8u; e++) for (uint32_t r = 0; r < nout; r++) {
                        rg[e * nout + r] = dot_q4k(eg + (size_t)slots[e] * (64 * NB4 * 144) + (size_t)r * NB4 * 144, x);
                        ru[e * nout + r] = dot_q4k(eu + (size_t)slots[e] * (64 * NB4 * 144) + (size_t)r * NB4 * 144, x);
                    }
                    char nm[64];
                    snprintf(nm, sizeof(nm), "gateup q4k edge nout=%u wtab g", nout); gate(nm, rg, ey, 8 * nout);
                    snprintf(nm, sizeof(nm), "gateup q4k edge nout=%u wtab u", nout); gate(nm, ru, euY, 8 * nout);
                }
            }

            // ---- Q5 blobs: one arena reused by Q5_0 then Q5_1 data ----
            size_t wdn = psegs((size_t)8 * NO5 * NB5 * 24);
            uint32_t sacc = seg(NO5);
            uint32_t swt = seg(8u);
            float * wts = scr.data() + swt;
            auto run_down = [&](bool q51, uint32_t nout, uint32_t wtab_on, const uint32_t * slots) {
                for (uint32_t e = 0; e < 8u; e++)
                    sel[e] = wtab_on ? expids[e] : e;   // direct: blob index = sel
                for (uint32_t e = 0; e < 8u; e++) wts[e] = R.f(-1.5f, 1.5f);
                if (slots) wts[2] = 0.0f;                          // zero-weight item must drop out
                MoeDownQ5F32Args A{};
                A.woff = wdn; A.hoff = sh; A.accoff = sacc; A.wtoff = swt; A.seloff = ssel;
                A.wtab_off = wtab_on ? swtab : 0u; A.nout = nout; A.k = 1152u;
                A.stride = (size_t)NO5 * NB5 * (q51 ? 24 : 22);
                run(q51 ? "k_gemv_down_reduce8_q5_1_f32in" : "k_gemv_down_reduce8_q5_0_f32in",
                    true, (nout + 7u) / 8u, 1, 256, &A, sizeof(A));
                std::vector<Dot> ref(nout);
                for (uint32_t r = 0; r < nout; r++) {
                    double a = 0, sa = 0;
                    for (uint32_t e = 0; e < 8u; e++) {
                        uint32_t slot = slots ? slots[e] : e;
                        Dot d = dot_q5(wdn + (size_t)slot * (NO5 * NB5 * (q51 ? 24 : 22)) + (size_t)r * NB5 * (q51 ? 24 : 22), h + (size_t)e * 1152u, q51);
                        a += (double)wts[e] * d.a; sa += std::fabs(wts[e]) * d.sab;   // oracle: acc + w*dbuf, e-ascending
                    }
                    ref[r] = Dot{a, sa};
                }
                char nm[64];
                snprintf(nm, sizeof(nm), "down %s nout=%u %s", q51 ? "q5_1" : "q5_0", nout, wtab_on ? "wtab" : "direct");
                gate(nm, ref, sacc, nout);
            };
            auto fill_q5_blobs = [&](bool q51) {
                const uint32_t nexp = 8u;
                for (uint32_t s = 0; s < nexp; s++) for (uint32_t r = 0; r < NO5; r++) {
                    for (uint32_t b = 0; b < NB5; b++) {
                        for (int i = 0; i < 32; i++) v5[i] = R.f(-0.3f, 0.3f);
                        fill_q5(wdn + (size_t)s * (NO5 * NB5 * (q51 ? 24 : 22)) + (size_t)r * NB5 * (q51 ? 24 : 22) + (size_t)b * (q51 ? 24 : 22), v5, q51);
                    }
                }
            };

            // Test 42: down Q5_0, Inkling dims (k=1152, nout=2304), direct
            fill_q5_blobs(false);
            run_down(false, NO5, 0, nullptr);

            // Test 43: down Q5_0 edge nout=173, direct (odd tail rows)
            run_down(false, 173u, 0, nullptr);

            // Test 44: down Q5_1, Inkling dims direct
            fill_q5_blobs(true);
            run_down(true, NO5, 0, nullptr);

            // Test 45: down Q5_1 edge nout=100 + wtab + zero-weight experts (slots inside the 8-blob arena)
            {
                const uint32_t slots[8] = {7u, 0u, 3u, 7u, 1u, 5u, 6u, 4u};
                for (uint32_t s = 0; s < 1024u; s++) wt[s] = 0u;
                for (uint32_t e = 0; e < 8u; e++) wt[expids[e]] = slots[e];
                set_sel(expids);
                run_down(true, 100u, 1, slots);
                set_sel(modsel8);
            }

            // Test 46: zero activations -> exact zero outputs (guards the bnd==0 path)
            {
                std::vector<float> save(8 * 1152);
                memcpy(save.data(), h, 8u * 1152u * 4);
                memset(h, 0, 8u * 1152u * 4);
                MoeDownQ5F32Args A{};
                A.woff = wdn; A.hoff = sh; A.accoff = sacc; A.wtoff = swt; A.seloff = ssel;
                A.wtab_off = 0u; A.nout = NO5; A.k = 1152u; A.stride = (size_t)NO5 * NB5 * 24;
                run("k_gemv_down_reduce8_q5_1_f32in", true, NO5 / 8u, 1, 256, &A, sizeof(A));
                int bad = 0;
                for (uint32_t r = 0; r < NO5; r++) if (scr[sacc + r] != 0.0f) bad++;
                CHECK(bad == 0, "down q5_1 zero-x %d nonzero outputs", bad);
                printf("%-38s zero-x bitwise %s\n", "down_reduce8_q5_1_f32in zero-x", bad ? "FAIL" : "OK");
                memcpy(h, save.data(), 8u * 1152u * 4);
            }

            // ============ Milestone 7A: _m2 / _m4 bitwise parity vs m1 ============
            // The optimized kernels keep per-row FMA order, lane block iteration
            // and simd_sum trees identical, so parity is exact-bit (stronger than
            // the fp64 1024-ulp gate; poison-filled outputs catch unwritten slots).
            {
                auto cmp_bits = [&](const char * nm, const std::vector<float> & ref,
                                    uint32_t off, size_t n) {
                    const uint32_t * pa = (const uint32_t *)ref.data();
                    const uint32_t * pb = (const uint32_t *)(scr.data() + off);
                    size_t bad = 0, widx = 0;
                    for (size_t i = 0; i < n; i++) if (pa[i] != pb[i]) { if (!bad) widx = i; bad++; }
                    CHECK(bad == 0, "%s %zu bitwise mismatches (first @ %zu)", nm, bad, widx);
                    printf("%-38s bitwise vs m1 %s\n", nm, bad ? "FAIL" : "OK");
                };
                auto cmp_gateup = [&](uint32_t nout, uint32_t wtab_on) {
                    for (uint32_t e = 0; e < 8u; e++) sel[e] = wtab_on ? expids[e] : e;
                    MoeGateUpQ4KF32Args A{};
                    A.woff = wgt; A.woff2 = wup; A.xoff = sx; A.yoff = sgy; A.yoff2 = suy;
                    A.seloff = ssel; A.wtab_off = wtab_on ? swtab : 0u; A.nout = nout; A.k = 2304u;
                    A.stride = (size_t)NO4 * NB4 * 144;
                    run("k_gemv_gateup_batch8_q4k_f32in", true, 2u * nout, 1, 256, &A, sizeof(A));
                    std::vector<float> rg(scr.data() + sgy, scr.data() + sgy + 8u * nout);
                    std::vector<float> ru(scr.data() + suy, scr.data() + suy + 8u * nout);
                    memset(scr.data() + sgy, 0x7F, 8u * nout * 4);
                    memset(scr.data() + suy, 0x7F, 8u * nout * 4);
                    run("k_gemv_gateup_batch8_q4k_f32in_m2", true, 2u * ((nout + 1u) >> 1u), 1, 256, &A, sizeof(A));
                    char nm[64];
                    snprintf(nm, sizeof(nm), "gateup q4k m2 nout=%u %s gate", nout, wtab_on ? "wtab" : "direct");
                    cmp_bits(nm, rg, sgy, 8u * nout);
                    snprintf(nm, sizeof(nm), "gateup q4k m2 nout=%u %s up", nout, wtab_on ? "wtab" : "direct");
                    cmp_bits(nm, ru, suy, 8u * nout);
                };
                auto cmp_down = [&](bool q51, uint32_t nout, uint32_t wtab_on, const uint32_t * slots) {
                    for (uint32_t e = 0; e < 8u; e++) sel[e] = wtab_on ? expids[e] : e;
                    for (uint32_t e = 0; e < 8u; e++) wts[e] = R.f(-1.5f, 1.5f);
                    if (slots) wts[2] = 0.0f;   // zero-weight expert must drop out
                    MoeDownQ5F32Args A{};
                    A.woff = wdn; A.hoff = sh; A.accoff = sacc; A.wtoff = swt; A.seloff = ssel;
                    A.wtab_off = wtab_on ? swtab : 0u; A.nout = nout; A.k = 1152u;
                    A.stride = (size_t)NO5 * NB5 * (q51 ? 24 : 22);
                    run(q51 ? "k_gemv_down_reduce8_q5_1_f32in" : "k_gemv_down_reduce8_q5_0_f32in",
                        true, (nout + 7u) / 8u, 1, 256, &A, sizeof(A));
                    std::vector<float> ref(scr.data() + sacc, scr.data() + sacc + nout);
                    memset(scr.data() + sacc, 0x7F, nout * 4);
                    run(q51 ? "k_gemv_down_reduce8_q5_1_f32in_m4" : "k_gemv_down_reduce8_q5_0_f32in_m4",
                        true, (nout + 31u) / 32u, 1, 256, &A, sizeof(A));
                    char nm[64];
                    snprintf(nm, sizeof(nm), "down %s m4 nout=%u %s", q51 ? "q5_1" : "q5_0", nout,
                             wtab_on ? "wtab" : "direct");
                    cmp_bits(nm, ref, sacc, nout);
                };

                // Test 47: gateup q4_k m2 == m1, Inkling dims (nout=1152) direct
                cmp_gateup(NO4, 0);

                // Test 48: gateup q4_k m2 == m1, odd tail nout=173 + wtab indirection
                cmp_gateup(173u, 0);
                {
                    const uint32_t gslots[8] = {5u, 2u, 7u, 0u, 3u, 6u, 1u, 4u};
                    for (uint32_t s = 0; s < 1024u; s++) wt[s] = 0u;
                    for (uint32_t e = 0; e < 8u; e++) wt[expids[e]] = gslots[e];
                    cmp_gateup(NO4, 1);
                }

                // Test 49: down q5_0 m4 == m1, full nout=2304 + odd tail 173
                fill_q5_blobs(false);
                cmp_down(false, NO5, 0, nullptr);
                cmp_down(false, 173u, 0, nullptr);

                // Test 50: down q5_1 m4 == m1, full + wtab slots + zero-weight expert
                fill_q5_blobs(true);
                cmp_down(true, NO5, 0, nullptr);
                {
                    const uint32_t slots[8] = {7u, 0u, 3u, 7u, 1u, 5u, 6u, 4u};
                    for (uint32_t s = 0; s < 1024u; s++) wt[s] = 0u;
                    for (uint32_t e = 0; e < 8u; e++) wt[expids[e]] = slots[e];
                    cmp_down(true, 100u, 1, slots);
                }
                set_sel(modsel8);
            }
        }

        if (do_bench) run_bench(rt);
        printf("\n%s (%d failures)\n", g_fail ? "TESTS FAILED" : "ALL P2A KERNEL PARITY TESTS PASSED", g_fail);
        return g_fail ? 1 : 0;
    }
}
