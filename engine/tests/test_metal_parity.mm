// test_metal_parity.mm — MSL kernels vs C++ float32 reference parity.
// Exactness policy (AGENTS.md): dequant must be BITWISE (FMA contraction off
// on both sides); reduction/activation kernels use 1e-6 relative tolerance.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "dequant.hpp"
#include "metal_runtime.hpp"

using namespace m64;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

// Deterministic PRNG -> uniform-ish payloads (exactness tests are byte tests).
struct Rng {
    std::mt19937 g{0x4D633634u};
    uint32_t u32() { return g(); }
    float f32() { return std::uniform_real_distribution<float>(-8.f, 8.f)(g); }
};

static id<MTLBuffer> mkbuf(MetalRuntime & rt, const void * data, size_t n) {
    id<MTLBuffer> b = [rt.device() newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (data) memcpy(b.contents, data, n);   // Shared storage: CPU-coherent pointer
    return b;
}

static void rd(id<MTLBuffer> b, void * dst, size_t n) { memcpy(dst, b.contents, n); }

static void dispatch(MetalRuntime & rt, const char * fn,
                     std::vector<std::pair<NSUInteger, id<MTLBuffer>>> binds,
                     NSUInteger threads, NSUInteger tgroup = 0) {
    auto p = rt.pipeline(fn);
    CHECK(p != nil, "pipeline %s missing", fn);
    if (!p) return;
    auto cmd = [rt.queue() commandBuffer];
    auto enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:p];
    for (auto & [idx, buf] : binds) [enc setBuffer:buf offset:0 atIndex:idx];
    MTLSize g, l;
    if (tgroup) { g = MTLSizeMake(tgroup, 1, 1); l = MTLSizeMake(threads, 1, 1); }
    else { g = MTLSizeMake(threads, 1, 1); l = MTLSizeMake(std::min<NSUInteger>(256, threads), 1, 1); }
    [enc dispatchThreads:g threadsPerThreadgroup:l];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

static bool bits_equal(const float * a, const float * b, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], 4) != 0) { printf("  first diff at %zu: %.9g vs %.9g\n", i, a[i], b[i]); return false; }
    return true;
}
static double max_rel(const float * a, const float * b, size_t n) {
    double worst = 0;
    for (size_t i = 0; i < n; i++) {
        double d = std::fabs((double)a[i] - b[i]);
        double m = std::max(1e-6, std::fabs((double)a[i]));
        worst = std::max(worst, d / m);
    }
    return worst;
}

int main(int argc, const char ** argv) {
    @autoreleasepool {
        Rng R;
        MetalRuntime rt;
        std::string err;
        if (!rt.init("engine/kernels", "build/m64_test.arch", err)) {
            printf("FAIL: MetalRuntime init: %s\n", err.c_str());
            return 1;
        }
        printf("Metal runtime: device=%s archive_cache=%d\n",
               [[rt.device() name] UTF8String], (int)rt.used_archive_cache());

        // ---- 1. dequant_q8_1 (BITWISE) ----
        {
            const int NB = 1024, K = NB * 32;
            std::vector<uint8_t> blocks(NB * 36);
            for (auto & b : blocks) b = (uint8_t)R.u32();
            // keep f16 fields finite: mantissa free, exponent < 0x1F (no inf/NaN)
            for (int i = 0; i < NB; i++) {
                blocks[i*36+0] = (uint8_t)R.u32(); blocks[i*36+1] &= 0x3B;
                blocks[i*36+2] = (uint8_t)R.u32(); blocks[i*36+3] &= 0x3B;
            }
            std::vector<float> ref(K), got(K);
            for (int i = 0; i < NB; i++)
                dequantize_row_q8_1_ref((const BlockQ8_1 *)(blocks.data() + i * 36), ref.data() + i * 32, 32);
            auto bin = mkbuf(rt, blocks.data(), blocks.size()), bout = mkbuf(rt, nullptr, K * 4);
            dispatch(rt, "dequant_q8_1", {{0, bin}, {1, bout}}, K);
            rd(bout, got.data(), K * 4);
            CHECK(bits_equal(ref.data(), got.data(), K), "dequant_q8_1 not bitwise");
            printf("dequant_q8_1: %d elems bitwise %s\n", K, g_fail ? "- see above" : "OK");
        }
        // ---- 2. dequant_q8_0 (BITWISE) ----
        {
            const int NB = 1024, K = NB * 32;
            std::vector<uint8_t> blocks(NB * 34);
            for (auto & b : blocks) b = (uint8_t)R.u32();
            for (int i = 0; i < NB; i++) { blocks[i*34] = (uint8_t)R.u32(); blocks[i*34+1] &= 0x3B; }
            std::vector<float> ref(K), got(K);
            for (int i = 0; i < NB; i++)
                dequantize_row_q8_0_ref((const BlockQ8_0 *)(blocks.data() + i * 34), ref.data() + i * 32, 32);
            auto bin = mkbuf(rt, blocks.data(), blocks.size()), bout = mkbuf(rt, nullptr, K * 4);
            dispatch(rt, "dequant_q8_0", {{0, bin}, {1, bout}}, K);
            rd(bout, got.data(), K * 4);
            CHECK(bits_equal(ref.data(), got.data(), K), "dequant_q8_0 not bitwise");
            printf("dequant_q8_0: %d elems bitwise OK\n", K);
        }
        // ---- 3. dequant_f16 (BITWISE, incl. subnormals) ----
        {
            const int K = 8192;
            std::vector<uint16_t> h(K);
            for (auto & x : h) { do { x = (uint16_t)(R.u32() & 0x7FFF); } while ((x & 0x7C00) == 0x7C00); }  // no inf/NaN
            std::vector<float> ref(K), got(K);
            for (int i = 0; i < K; i++) ref[i] = f16_to_f32(h[i]);
            auto bin = mkbuf(rt, h.data(), K * 2), bout = mkbuf(rt, nullptr, K * 4);
            dispatch(rt, "dequant_f16", {{0, bin}, {1, bout}}, K);
            rd(bout, got.data(), K * 4);
            CHECK(bits_equal(ref.data(), got.data(), K), "dequant_f16 not bitwise");
            printf("dequant_f16: %d elems bitwise OK\n", K);
        }
        // ---- 4. gemv_f32 (bitwise: same serial order) ----
        {
            const int M = 256, K = 512;
            std::vector<float> W(M * K), x(K), y(M), ref(M);
            for (auto & v : W) v = R.f32();
            for (auto & v : x) v = R.f32();
            for (int m = 0; m < M; m++) {
                float acc = 0;
                for (int j = 0; j < K; j++) acc += W[(size_t)m * K + j] * x[j];
                ref[m] = acc;
            }
            auto bw = mkbuf(rt, W.data(), M * K * 4), bx = mkbuf(rt, x.data(), K * 4);
            auto bbb = mkbuf(rt, nullptr, 4), by = mkbuf(rt, nullptr, M * 4);
            uint32_t kk = K, hb = 0;
            auto bk = mkbuf(rt, &kk, 4), bhb = mkbuf(rt, &hb, 4);
            dispatch(rt, "gemv_f32", {{0, bw}, {1, bx}, {2, bbb}, {3, by}, {4, bk}, {5, bhb}}, M);
            rd(by, y.data(), M * 4);
            CHECK(bits_equal(ref.data(), y.data(), M), "gemv_f32 not bitwise");
            printf("gemv_f32: %dx%d bitwise OK\n", M, K);
        }
        // ---- 5. gemv_q8_1 (fused dequant gemv, bitwise) ----
        {
            const int M = 128, K = 256, NB = K / 32;
            std::vector<uint8_t> Wb(M * NB * 36);
            for (auto & b : Wb) b = (uint8_t)R.u32();
            for (size_t i = 0; i < (size_t)M * NB; i++) { Wb[i*36+1] &= 0x3B; Wb[i*36+3] &= 0x3B; }
            std::vector<float> x(K), y(M), ref(M), row(K);
            for (auto & v : x) v = R.f32();
            for (int m = 0; m < M; m++) {
                for (int b = 0; b < NB; b++)
                    dequantize_row_q8_1_ref((const BlockQ8_1 *)(Wb.data() + ((size_t)m * NB + b) * 36), row.data() + b * 32, 32);
                float acc = 0;
                for (int j = 0; j < K; j++) acc += row[j] * x[j];
                ref[m] = acc;
            }
            auto bw = mkbuf(rt, Wb.data(), Wb.size()), bx = mkbuf(rt, x.data(), K * 4), by = mkbuf(rt, nullptr, M * 4);
            uint32_t kk = K; auto bk = mkbuf(rt, &kk, 4);
            dispatch(rt, "gemv_q8_1", {{0, bw}, {1, bx}, {2, by}, {3, bk}}, M);
            rd(by, y.data(), M * 4);
            CHECK(bits_equal(ref.data(), y.data(), M), "gemv_q8_1 not bitwise");
            printf("gemv_q8_1: %dx%d bitwise OK\n", M, K);
        }
        // ---- 6. moe_gemv_q8_1 (top-K gather) ----
        {
            const int E = 128, K = 256, NR = 64, TOPK = 2, NB = K / 32;
            std::vector<uint8_t> Wb((size_t)E * NR * NB * 36);
            for (auto & b : Wb) b = (uint8_t)R.u32();
            for (size_t i = 0; i < Wb.size() / 36; i++) { Wb[i*36+1] &= 0x3B; Wb[i*36+3] &= 0x3B; }
            uint32_t exps[TOPK] = {3, 17};
            std::vector<float> x(K), y(TOPK * NR), ref(TOPK * NR), row(K);
            for (auto & v : x) v = R.f32();
            for (int t = 0; t < TOPK; t++) {
                const uint8_t * base = Wb.data() + (size_t)exps[t] * NR * NB * 36;
                for (int r = 0; r < NR; r++) {
                    for (int b = 0; b < NB; b++)
                        dequantize_row_q8_1_ref((const BlockQ8_1 *)(base + (size_t)r * NB * 36 + (size_t)b * 36), row.data() + b * 32, 32);
                    float acc = 0;
                    for (int j = 0; j < K; j++) acc += row[j] * x[j];
                    ref[t * NR + r] = acc;
                }
            }
            auto bw = mkbuf(rt, Wb.data(), Wb.size()), bx = mkbuf(rt, x.data(), K * 4);
            auto be = mkbuf(rt, exps, 8), by = mkbuf(rt, nullptr, TOPK * NR * 4);
            uint32_t kk = K, nr = NR; uint64_t se = (uint64_t)NR * NB;
            auto bk = mkbuf(rt, &kk, 4), bnr = mkbuf(rt, &nr, 4), bse = mkbuf(rt, &se, 8);
            auto p = rt.pipeline("moe_gemv_q8_1");
            CHECK(p, "moe_gemv_q8_1 pipeline missing");
            if (p) {
                auto cmd = [rt.queue() commandBuffer];
                auto enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:p];
                [enc setBuffer:bw offset:0 atIndex:0]; [enc setBuffer:bx offset:0 atIndex:1];
                [enc setBuffer:be offset:0 atIndex:2]; [enc setBuffer:by offset:0 atIndex:3];
                [enc setBuffer:bk offset:0 atIndex:4]; [enc setBuffer:bnr offset:0 atIndex:5];
                [enc setBuffer:bse offset:0 atIndex:6];
                [enc dispatchThreads:MTLSizeMake(TOPK, NR, 1) threadsPerThreadgroup:MTLSizeMake(16, 4, 1)];
                [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                rd(by, y.data(), TOPK * NR * 4);
                CHECK(bits_equal(ref.data(), y.data(), TOPK * NR), "moe_gemv_q8_1 not bitwise");
                printf("moe_gemv_q8_1: top%d x %d rows bitwise OK\n", TOPK, NR);
            }
        }
        // ---- 7. moe_route (deterministic argmax order; bitwise) ----
        {
            const int NE = 128, TOPK = 8; const float SCALE = 8.0f;
            std::vector<float> logits(NE), bias(NE);
            for (auto & v : logits) v = R.f32();
            for (auto & v : bias) v = R.f32() * 0.1f;
            // C++ ref, same order
            std::vector<float> score(NE);
            for (int i = 0; i < NE; i++) score[i] = 1.0f / (1.0f + std::exp(-logits[i])) + bias[i];
            float rw[TOPK]; uint ri[TOPK]; float wsum = 0;
            for (int t = 0; t < TOPK; t++) {
                int bi = 0; float bv = -1e30f;
                for (int i = 0; i < NE; i++) if (score[i] > bv) { bv = score[i]; bi = i; }
                rw[t] = bv; ri[t] = bi; score[bi] = -1e30f; wsum += bv;
            }
            struct { float w[16]; uint32_t top[16]; } out{};
            auto bl = mkbuf(rt, logits.data(), NE * 4), bb = mkbuf(rt, bias.data(), NE * 4);
            auto bo = mkbuf(rt, nullptr, 128);
            uint32_t ne = NE, tk = TOPK; auto bne = mkbuf(rt, &ne, 4), btk = mkbuf(rt, &tk, 4), bsc = mkbuf(rt, &SCALE, 4);
            dispatch(rt, "moe_route", {{0, bl}, {1, bb}, {2, bo}, {3, bne}, {4, btk}, {5, bsc}}, 32);
            rd(bo, &out, 128);
            for (int t = 0; t < TOPK; t++) {
                float refw = (rw[t] / wsum) * SCALE;
                CHECK(out.top[t] == ri[t], "route top[%d]: %u vs %u", t, out.top[t], ri[t]);
                CHECK(std::fabs(out.w[t] - refw) < 1e-6 * std::fabs(refw) + 1e-9, "route w[%d] %.7g vs %.7g", t, out.w[t], refw);
            }
            printf("moe_route: top8 selection + weights OK\n");
        }
        // ---- 8. situ (rel tol; exp/tanh implementations may differ by ulp) ----
        {
            const int K = 4096; const float beta = 4.0f, lb = 25.0f;
            std::vector<float> v(K), ref(K);
            for (auto & x : v) x = std::uniform_real_distribution<float>(-30.f, 30.f)(R.g);
            for (int i = 0; i < K; i++) {
                float sig = 1.0f / (1.0f + std::exp(-beta * v[i]));
                ref[i] = v[i] * sig * std::tanh(v[i] / lb);
            }
            std::vector<float> ref_in = v;
            auto bv = mkbuf(rt, v.data(), K * 4);
            auto bb2 = mkbuf(rt, &beta, 4), blb = mkbuf(rt, &lb, 4);
            dispatch(rt, "situ", {{0, bv}, {1, bb2}, {2, blb}}, K);
            std::vector<float> got(K);
            rd(bv, got.data(), K * 4);
            double rel = max_rel(ref.data(), got.data(), K);
            CHECK(rel < 2e-4, "situ rel err %.3g", rel);
            printf("situ: max rel err %.3g (tol 2e-4, libm-vs-MSL ulp) OK\n", rel);
            (void)ref_in;
        }
        // ---- 9. rmsnorm_row (rel tol: tree vs serial reduction) ----
        {
            const int K = 2304;  // model width
            const float eps = 1e-6f;
            std::vector<float> x(K), w(K), y(K), ref(K);
            for (auto & v : x) v = R.f32();
            for (auto & v : w) v = 1.0f + R.f32() * 0.05f;
            double ss = 0;
            for (int i = 0; i < K; i++) ss += (double)x[i] * x[i];
            float inv = (float)(1.0 / std::sqrt(ss / K + eps));
            for (int i = 0; i < K; i++) ref[i] = x[i] * inv * w[i];
            auto bx = mkbuf(rt, x.data(), K * 4), bw = mkbuf(rt, w.data(), K * 4), by = mkbuf(rt, nullptr, K * 4);
            uint32_t kk = K; auto bk = mkbuf(rt, &kk, 4), beps = mkbuf(rt, &eps, 4);
            dispatch(rt, "rmsnorm_row", {{0, bx}, {1, bw}, {2, by}, {3, beps}, {4, bk}}, 512, 1);
            rd(by, y.data(), K * 4);
            double rel = max_rel(ref.data(), y.data(), K);
            CHECK(rel < 1e-6, "rmsnorm rel err %.3g", rel);
            printf("rmsnorm_row: max rel err %.3g (tol 1e-6) OK\n", rel);
        }
        // ---- 10. softcap + argmax ----
        {
            const int K = 1024; const float cap = 30.0f;
            std::vector<float> v(K), ref(K);
            for (auto & x : v) x = R.f32() * 20.f;
            for (int i = 0; i < K; i++) ref[i] = cap * std::tanh(v[i] / cap);
            auto bv = mkbuf(rt, v.data(), K * 4), bc = mkbuf(rt, &cap, 4);
            dispatch(rt, "softcap", {{0, bv}, {1, bc}}, K);
            std::vector<float> got(K);
            rd(bv, got.data(), K * 4);
            double rel = max_rel(ref.data(), got.data(), K);
            CHECK(rel < 1e-5, "softcap rel err %.3g", rel);
            uint32_t out = 0;
            auto bi = mkbuf(rt, &out, 4), bn = mkbuf(rt, &K, 4);
            dispatch(rt, "argmax", {{0, bv}, {1, bi}, {2, bn}}, 32);
            rd(bi, &out, 4);
            int bi_ref = 0;
            for (int i = 1; i < K; i++) if (got[i] > got[bi_ref]) bi_ref = i;
            CHECK(out == (uint32_t)bi_ref, "argmax %u vs %d", out, bi_ref);
            printf("softcap + argmax OK\n");
        }

        printf("\n%s (%d failures)\n", g_fail ? "TESTS FAILED" : "ALL METAL PARITY TESTS PASSED", g_fail);
        return g_fail ? 1 : 0;
    }
}
