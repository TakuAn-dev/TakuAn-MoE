// test_gemma4_cpu.cpp — Gemma 4 CPU oracle smoke + determinism on tiny pack.
// Usage: test_gemma4_cpu <tiny_gemma4.m64pack>
// Checks: hash-verified pack open, arch gating, 10-token greedy walk across the
// sliding-window boundary (all logits finite + within softcap), step-by-step vs
// prefill replay bitwise equality, and cross-reset replay determinism.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gemma4_forward.hpp"
#include "pack_reader.hpp"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <pack>\n", argv[0]); return 2; }
    m64::PackReader pack;
    try {
        pack.open(argv[1], /*verify_hashes=*/true);
        printf("pack opened: %zu tensors, arch=%s, hashes verified\n", pack.n_tensors(), pack.header().arch.c_str());
    } catch (const std::exception & e) {
        printf("FAIL: pack open: %s\n", e.what());
        return 1;
    }

    // --- bad context rejected cleanly ---
    {
        m64::Gemma4Forward bad;
        std::string err;
        CHECK(!bad.init(pack, 0, err) && !err.empty(), "init(ctx=0) should fail with message");
    }

    // --- bad-arch pack rejected cleanly (tiny.m64pack if co-located) ---
    {
        std::string ink = std::string(argv[1]) + ".nope";  // missing file: still clean
        ink = ink.substr(0, ink.rfind('/') + 1) + "tiny.m64pack";
        m64::PackReader p2;
        try {
            p2.open(ink.c_str(), false);
            if (p2.header().arch != "gemma4") {
                m64::Gemma4Forward bad;
                std::string err;
                CHECK(!bad.init(p2, 8, err), "init(arch=%s) should reject non-gemma4", p2.header().arch.c_str());
                CHECK(!err.empty(), "arch rejection message non-empty");
            }
        } catch (...) { /* fixture colocated only in make test; skip when absent */ }
    }

    // --- init + 10-token greedy walk ---
    m64::Gemma4Forward fwd;
    std::string err;
    if (!fwd.init(pack, 32, err)) { printf("FAIL: init: %s\n", err.c_str()); return 1; }
    printf("init ok: vocab %lld/%lld, resident %.1f MiB\n",
           (long long)fwd.vocab_effective(), (long long)fwd.vocab_logit_rows(),
           fwd.resident_bytes() / 1048576.0);
    CHECK(fwd.vocab_effective() == 32 && fwd.vocab_logit_rows() == 32,
          "tied-vocab rows expected 32/32, got %lld/%lld",
          (long long)fwd.vocab_effective(), (long long)fwd.vocab_logit_rows());

    std::vector<int32_t> tokens;
    for (int i = 0; i < 10; i++) tokens.push_back((i * 37 + 5) % (int32_t)fwd.vocab_effective());

    const float cap = 30.0f;
    std::vector<std::vector<float>> step_logits;
    std::vector<int32_t> step_tokens;
    for (int i = 0; i < (int)tokens.size(); i++) {
        m64::StepResult r = fwd.step(tokens[i]);
        CHECK(r.logits && r.n_logits == fwd.vocab_effective(), "step %d shape", i);
        for (int64_t j = 0; j < r.n_logits; j++) {
            if (!std::isfinite(r.logits[j])) { CHECK(false, "step %d logits[%lld] non-finite", i, (long long)j); break; }
            if (r.logits[j] < -cap - 1e-4f || r.logits[j] > cap + 1e-4f) {
                CHECK(false, "step %d logits[%lld]=%.3f outside softcap [-30,30]", i, (long long)j, r.logits[j]); break;
            }
        }
        float mn = r.logits[0], mx = r.logits[0];
        for (int64_t j = 1; j < r.n_logits; j++) { if (r.logits[j] < mn) mn = r.logits[j]; if (r.logits[j] > mx) mx = r.logits[j]; }
        CHECK(mx - mn > 1e-4f, "step %d logits degenerate (range %.2e)", i, mx - mn);
        CHECK(r.argmax >= 0 && (int64_t)r.argmax < r.n_logits && r.logits[r.argmax] == mx, "step %d argmax", i);
        step_logits.emplace_back(r.logits, r.logits + r.n_logits);
        step_tokens.push_back(r.argmax);
    }
    printf("ok  10-step walk (window=4) finite/in-cap/argmax, tokens:");
    for (int32_t t : step_tokens) printf(" %d", t);
    printf("\n");

    // --- reset + prefill(first 9) + step(10th) must match step-by-step bitwise ---
    fwd.reset();
    std::vector<int32_t> nine(tokens.begin(), tokens.end() - 1);
    m64::StepResult r = fwd.prefill(nine);
    CHECK(r.logits && memcmp(r.logits, step_logits[8].data(), sizeof(float) * step_logits[8].size()) == 0,
          "prefill logits[8] != stepwise logits[8]");
    CHECK(r.argmax == step_tokens[8], "prefill argmax[8] mismatch");
    r = fwd.step(tokens[9]);
    CHECK(memcmp(r.logits, step_logits[9].data(), sizeof(float) * step_logits[9].size()) == 0,
          "post-prefill step logits[9] != stepwise");
    CHECK(r.argmax == step_tokens[9], "post-prefill argmax[9] mismatch");
    printf("ok  prefill+step == stepwise (bitwise)\n");

    // --- full replay from a second reset must be bitwise identical ---
    fwd.reset();
    for (int i = 0; i < (int)tokens.size(); i++) {
        m64::StepResult r2 = fwd.step(tokens[i]);
        CHECK(memcmp(r2.logits, step_logits[i].data(), sizeof(float) * step_logits[i].size()) == 0,
              "replay step %d bitwise mismatch", i);
        CHECK(r2.argmax == step_tokens[i], "replay step %d argmax", i);
    }
    printf("ok  full replay bitwise deterministic\n");

    // --- KV history sensitivity: step(5) after different histories MUST differ ---
    // Catches attention that reads pos+j (future/zero slots) instead of cached 0..pos.
    fwd.reset(); fwd.step(1);
    m64::StepResult r_a = fwd.step(5);
    std::vector<float> logits_a(r_a.logits, r_a.logits + r_a.n_logits);
    fwd.reset(); fwd.step(7);
    m64::StepResult r_b = fwd.step(5);
    float kv_maxdiff = 0.0f;
    for (int64_t i = 0; i < r_b.n_logits; i++) {
        float d = std::fabs(logits_a[i] - r_b.logits[i]);
        if (d > kv_maxdiff) kv_maxdiff = d;
    }
    CHECK(kv_maxdiff > 1e-4f,
          "KV history sensitivity: logits differ only %.3e — attention may ignore cached KV", kv_maxdiff);
    printf("ok  KV history sensitivity (max |diff| = %.3e)\n", kv_maxdiff);

    printf(g_fail ? "\nFAILURES: %d\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
