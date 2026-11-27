// test_gemma4_metal.mm — Milestone 6B GPU-vs-CPU-oracle parity for Gemma 4.
//
// usage: test_gemma4_metal <tiny_gemma4.m64pack>
//
// 1. init() error rejection: arena_gb > 0, context_len <= 0, non-gemma4 pack.
// 2. FP16 KV parity vs Gemma4Forward CPU oracle over a 10-step walk that
//    crosses the sliding window (window = 4 on tiny_gemma4):
//    all logits finite in [-30.01, 30.01], cos >= 0.9999, max_abs < 5e-2,
//    10/10 greedy argmax match.
// 3. Q8_0 KV parity over the same walk: cos >= 0.999, 10/10 argmax.
// 4. Prefill parity (Metal vs Metal): reset + prefill(tokens[0..8]) +
//    step(tokens[9]) == stepwise Metal FP16 (max_abs < 1e-4, argmax equal).
// 5. KV history sensitivity: [1,5] vs [7,5] differ on BOTH Metal FP16 and
//    Metal Q8_0 (max_abs > 1e-4) — proves the Q8_0 rings are actually read.
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gemma4_forward.hpp"
#include "gemma4_forward_metal.hpp"
#include "pack_reader.hpp"

using namespace m64;

static int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s\n", (msg)); g_fail++;                    \
        }                                                                    \
    } while (0)

static double cosine(const float * a, const float * b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return na > 0 && nb > 0 ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
}

// Runs a 10-step walk on both engines; returns worst cos / max abs / argmax hits.
struct WalkMetrics {
    double min_cos = 1e30;
    float max_abs = 0.f;
    int argmax_hits = 0;
};

static WalkMetrics walk_parity(Gemma4Forward & cpu, Gemma4ForwardMetal & gpu, int64_t vocab) {
    cpu.reset();
    gpu.reset();
    WalkMetrics m;
    for (int i = 0; i < 10; i++) {
        const int32_t tok = (int32_t)((i * 37 + 5) % vocab);
        StepResult rc = cpu.step(tok);
        StepResult rg = gpu.step(tok);
        CHECK(rc.argmax >= 0 && rg.argmax >= 0, "both engines produce an argmax");
        if (rc.argmax < 0 || rg.argmax < 0) return m;
        bool finite = true;
        float mx = 0.f;
        for (int64_t v = 0; v < rc.n_logits; v++) {
            if (!std::isfinite(rg.logits[v])) finite = false;
            float a = std::fabs(rg.logits[v]);
            if (a > mx) mx = a;
            float d = std::fabs(rg.logits[v] - rc.logits[v]);
            if (d > m.max_abs) m.max_abs = d;
        }
        CHECK(finite, "all GPU logits finite");
        CHECK(mx <= 30.01f, "logits within the softcap +-30.01 bound");
        const double cos = cosine(rc.logits, rg.logits, rc.n_logits);
        if (cos < m.min_cos) m.min_cos = cos;
        if (rc.argmax == rg.argmax) m.argmax_hits++;
    }
    return m;
}

// One walk on the Metal engine alone; returns max |a - b| for two prompt histories
// (caller resets between).
static float metal_delta(Gemma4ForwardMetal & gpu, const std::vector<int32_t> & a,
                         const std::vector<int32_t> & b, int64_t vocab) {
    gpu.reset();
    StepResult ra{nullptr, 0, -1}, rb{nullptr, 0, -1};
    for (size_t i = 0; i + 1 < a.size(); i++) gpu.step(a[i]);
    ra = gpu.step(a.back());
    std::vector<float> la(ra.logits, ra.logits + vocab);  // internal buffer is reused
    gpu.reset();
    for (size_t i = 0; i + 1 < b.size(); i++) gpu.step(b[i]);
    rb = gpu.step(b.back());
    float mx = 0.f;
    for (int64_t v = 0; v < vocab; v++) {
        const float d = std::fabs(la[v] - rb.logits[v]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: test_gemma4_metal <tiny_gemma4.m64pack>\n"); return 2; }
    const std::string pack_path = argv[1];

    PackReader pack;
    try {
        pack.open(pack_path, true);  // full SHA-256 verification
    } catch (const std::exception & e) {
        std::printf("FAIL: pack open: %s\n", e.what()); return 2;
    }

    std::printf("== 1. init() error rejection ==\n");
    {
        std::string err;
        Gemma4ForwardMetal f;
        CHECK(!f.init(pack, 32, err, 1.0, "pread", false) && !err.empty(),
              "arena_gb > 0 rejected with a clear error");
        Gemma4ForwardMetal g;
        err.clear();
        CHECK(!g.init(pack, 0, err, 0.0, "pread", false) && !err.empty(), "context_len <= 0 rejected");
        Gemma4ForwardMetal h;
        err.clear();
        CHECK(!h.init(pack, -5, err, 0.0, "pread", false) && !err.empty(), "negative context_len rejected");
        // Non-gemma4 pack: tiny.m64pack (Milestone-1 fixture) is built alongside.
        std::string alt;
        size_t slash = pack_path.find_last_of('/');
        std::string other = (slash == std::string::npos ? std::string(".") : pack_path.substr(0, slash)) + "/tiny.m64pack";
        PackReader p2;
        try {
            p2.open(other, false);
            Gemma4ForwardMetal k;
            err.clear();
            CHECK(!k.init(p2, 32, err, 0.0, "pread", false) && !err.empty(),
                  "non-gemma4 arch rejected with a clear error");
        } catch (const std::exception &) {
            std::printf("  SKIP: no co-located non-gemma4 pack (%s)\n", other.c_str());
        }
        (void)alt;
    }

    Gemma4Forward cpu;
    {
        std::string err;
        if (!cpu.init(pack, 32, err)) { std::printf("FAIL: CPU oracle init: %s\n", err.c_str()); return 2; }
    }
    const int64_t vocab = cpu.vocab_effective();
    std::printf("vocab=%lld rows=%lld\n", (long long)vocab, (long long)cpu.vocab_logit_rows());

    std::printf("== 2. FP16 KV parity vs CPU oracle (10-step walk, crosses window=4) ==\n");
    Gemma4ForwardMetal gpu_f16;
    {
        std::string err;
        if (!gpu_f16.init(pack, 32, err, 0.0, "pread", false)) {
            std::printf("FAIL: Metal FP16 init: %s\n", err.c_str()); return 2;
        }
    }
    {
        WalkMetrics m = walk_parity(cpu, gpu_f16, vocab);
        std::printf("  min_cos=%.7f max_abs_diff=%.3e argmax %d/10\n",
                    m.min_cos, (double)m.max_abs, m.argmax_hits);
        CHECK(m.min_cos >= 0.9999, "FP16 parity: worst cosine >= 0.9999");
        CHECK(m.max_abs < 5e-2f, "FP16 parity: worst max_abs_diff < 5e-2");
        CHECK(m.argmax_hits == 10, "FP16 parity: 10/10 greedy argmax match");
    }

    std::printf("== 3. Q8_0 KV parity vs CPU oracle ==\n");
    Gemma4ForwardMetal gpu_q8;
    {
        std::string err;
        if (!gpu_q8.init(pack, 32, err, 0.0, "pread", true)) {
            std::printf("FAIL: Metal Q8_0-KV init: %s\n", err.c_str()); return 2;
        }
    }
    {
        WalkMetrics m = walk_parity(cpu, gpu_q8, vocab);
        std::printf("  min_cos=%.7f max_abs_diff=%.3e argmax %d/10\n",
                    m.min_cos, (double)m.max_abs, m.argmax_hits);
        CHECK(m.min_cos >= 0.999, "Q8_0 KV parity: worst cosine >= 0.999");
        CHECK(m.argmax_hits == 10, "Q8_0 KV parity: 10/10 greedy argmax match");
    }

    std::printf("== 4. prefill parity (Metal prefill+step vs Metal stepwise) ==\n");
    {
        std::vector<int32_t> tokens(17);
        for (size_t i = 0; i < tokens.size(); i++) tokens[i] = (int32_t)((i * 37 + 5) % vocab);
        // Parity sweep: batched prefill (default + several chunk sizes) vs
        // stepwise, with prompt lengths crossing chunk boundaries AND the
        // sliding-window boundary, on FP16 and Q8_0 KV. Also verifies that
        // steps continuing after a batched prefill stay identical to stepwise
        // (KV ring, SWA slots and pos_ consistency).
        auto sweep = [&](Gemma4ForwardMetal & g, const char * tag) {
            for (int k : {2, 5, 9, 17}) {
                g.reset();
                std::vector<std::vector<float>> sl;
                std::vector<int> sa;
                for (int i = 0; i < k; i++) {
                    StepResult st = g.step(tokens[i]);
                    sl.emplace_back(st.logits, st.logits + st.n_logits);
                    sa.push_back(st.argmax);
                }
                for (int c : {0, 1, 2, 4, 5}) {
                    g.reset();
                    if (c != 0) g.set_prefill_chunk(c);
                    StepResult pk = g.prefill(std::vector<int32_t>(tokens.begin(), tokens.begin() + k));
                    if (c != 0) g.set_prefill_chunk(0);
                    double dot = 0.0, na = 0.0, nb = 0.0;
                    float mxk = 0.f;
                    for (size_t v = 0; v < sl[k - 1].size(); v++) {
                        const float d = std::fabs(sl[k - 1][v] - pk.logits[v]);
                        if (d > mxk) mxk = d;
                        dot += sl[k - 1][v] * pk.logits[v];
                        na += (double)sl[k - 1][v] * sl[k - 1][v];
                        nb += (double)pk.logits[v] * pk.logits[v];
                    }
                    const double cos = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 1.0;
                    std::printf("  [%s k=%d chunk=%d] max_abs=%.3e cos=%.7f argmax %d vs %d\n",
                                tag, k, c, (double)mxk, cos, sa[k - 1], pk.argmax);
                    CHECK(cos >= 0.9999, "prefill parity cosine >= 0.9999");
                    CHECK(sa[k - 1] == pk.argmax, "prefill parity argmax exact");
                }
                // continuation: prefill(k) then 3 more steps vs fully stepwise
                for (int st = 0; st < 3; st++) {
                    const int32_t nt = (int32_t)(((k + st) * 37 + 5) % vocab);
                    g.reset();
                    g.prefill(std::vector<int32_t>(tokens.begin(), tokens.begin() + k));
                    StepResult sb{nullptr, 0, -1};
                    for (int u = 0; u < st; u++) sb = g.step((int32_t)(((k + u) * 37 + 5) % vocab));
                    sb = g.step(nt);
                    std::vector<float> lb(sb.logits, sb.logits + sb.n_logits);
                    const int ab = sb.argmax;
                    g.reset();
                    StepResult sw{nullptr, 0, -1};
                    for (int i = 0; i < k; i++) sw = g.step(tokens[i]);
                    for (int u = 0; u <= st; u++) sw = g.step(u == st ? nt : (int32_t)(((k + u) * 37 + 5) % vocab));
                    float mxd = 0.f;
                    for (size_t v = 0; v < lb.size(); v++) {
                        const float d = std::fabs(sw.logits[v] - lb[v]);
                        if (d > mxd) mxd = d;
                    }
                    CHECK(mxd < 1e-4f, "step-after-prefill matches stepwise");
                    CHECK(ab == sw.argmax, "step-after-prefill argmax exact");
                }
            }
        };
        sweep(gpu_f16, "f16");
        sweep(gpu_q8, "q8");
        // full-length prompt: prefill(16)+step vs fully stepwise over 17 tokens
        gpu_f16.reset();
        StepResult a{nullptr, 0, -1};
        for (int i = 0; i < 17; i++) a = gpu_f16.step(tokens[i]);
        std::vector<float> la(a.logits, a.logits + a.n_logits);
        const int a_argmax = a.argmax;
        gpu_f16.reset();
        gpu_f16.prefill(std::vector<int32_t>(tokens.begin(), tokens.begin() + 16));
        StepResult b = gpu_f16.step(tokens[16]);
        CHECK(b.argmax >= 0, "both paths produce logits");
        float mx = 0.f;
        for (size_t v = 0; v < la.size(); v++) {
            const float d = std::fabs(la[v] - b.logits[v]);
            if (d > mx) mx = d;
        }
        std::printf("  max_abs_diff=%.3e argmax %d vs %d\n", (double)mx, a_argmax, b.argmax);
        CHECK(mx < 1e-4f, "prefill path bit-identical to stepwise (< 1e-4)");
        CHECK(a_argmax == b.argmax, "prefill argmax identical");
    }

    std::printf("== 5. KV history sensitivity ([1,5] vs [7,5]) on FP16 AND Q8_0 ==\n");
    {
        const std::vector<int32_t> a{1, 5}, b{7, 5};
        const float d_f16 = metal_delta(gpu_f16, a, b, vocab);
        const float d_q8 = metal_delta(gpu_q8, a, b, vocab);
        std::printf("  f16 max_abs_diff=%.6e   q8 max_abs_diff=%.6e\n", (double)d_f16, (double)d_q8);
        CHECK(d_f16 > 1e-4f, "FP16 KV: different histories change the logits");
        CHECK(d_q8 > 1e-4f, "Q8_0 KV: different histories change the logits (rings really read)");
    }

    if (g_fail == 0) {
        std::printf("PASS: Gemma 4 Metal GPU forward matches CPU oracle (resident)\n");
        return 0;
    }
    std::printf("FAILED (%d checks)\n", g_fail);
    return 1;
}
