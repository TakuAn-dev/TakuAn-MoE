// test_sampler.cpp — unit tests for src/sampler.hpp (temperature softmax +
// top-p nucleus sampling with the 1e-6 candidate pre-filter).
//
// Hard contract: for every logit row and (temp, top_p) pair, the pre-filtered
// sampler emits EXACTLY the same token sequence as the old brute-force
// full-vocabulary-sort sampler given the same RNG seed (both consume one
// uniform draw per token). Rows whose sub-floor tail carries mass the nucleus
// still needs must take the documented full-sort fallback, and even there the
// tokens must match.
//
// Plus a decode-cost guard: at the real vocabulary size (201,024) with a
// peaked (chat-like) row, the filtered sampler must beat the reference by a
// wide margin — that is the whole point of the pre-filter.
#include "sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        g_fail++;
    }
}

// The reference: sample_token() exactly as it behaved before the pre-filter
// (full-vocabulary std::sort per token), kept here so parity is provable.
static int32_t sample_token_ref(const float * logits, int64_t nv, float temp,
                                float top_p, int32_t argmax,
                                std::mt19937_64 & rng) {
    if (temp <= 0.0f) return argmax;
    if (nv <= 0) return argmax;
    float mx = logits[0];
    for (int64_t i = 1; i < nv; i++) mx = std::max(mx, logits[i]);
    std::vector<double> prob((size_t)nv);
    double sum = 0.0;
    const double inv_t = 1.0 / (double)temp;
    for (int64_t i = 0; i < nv; i++) {
        double p = std::exp(((double)logits[i] - (double)mx) * inv_t);
        prob[(size_t)i] = p;
        sum += p;
    }
    if (!(sum > 0.0)) return argmax;
    std::vector<size_t> keep;
    if (top_p < 1.0f) {
        std::vector<size_t> idx((size_t)nv);
        std::iota(idx.begin(), idx.end(), (size_t)0);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return prob[a] > prob[b]; });
        double cum = 0.0;
        for (size_t i : idx) {
            keep.push_back(i);
            cum += prob[i] / sum;
            if (cum >= top_p) break;
        }
    } else {
        keep.resize((size_t)nv);
        std::iota(keep.begin(), keep.end(), (size_t)0);
    }
    double ksum = 0.0;
    for (size_t i : keep) ksum += prob[i];
    std::uniform_real_distribution<double> uni(0.0, ksum);
    const double target = uni(rng);
    double acc = 0.0;
    for (size_t i : keep) {
        acc += prob[i];
        if (acc >= target) return (int32_t)i;
    }
    return (int32_t)keep.back();
}

static int32_t argmax_of(const std::vector<float> & v) {
    int32_t best = 0;
    for (size_t i = 1; i < v.size(); i++)
        if (v[i] > v[best]) best = (int32_t)i;
    return best;
}

// Logit rows spanning sharp -> flat, plus one row that MUST force the
// full-sort fallback (a long sub-floor tail carrying mass top_p still needs).
// Every row keeps strictly distinct probs so std::sort tie-breaking (which is
// unspecified for equal keys) can never make the two samplers disagree.
static std::vector<float> make_row(const std::string & kind, int64_t nv,
                                   uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<float> l((size_t)nv);
    for (int64_t i = 0; i < nv; i++) {
        double logit = 0.0;
        if (kind == "peaked") {          // typical chat row: ~700 above 1e-6
            logit = -0.02 * (double)i + 0.01 * uni(rng);
        } else if (kind == "zipf") {     // heavy tail: mixed above/below floor
            logit = -1.2 * std::log((double)i + 1.0) + 0.02 * uni(rng);
        } else if (kind == "sloppy") {   // wide nucleus, nothing below floor
            logit = 1.5 * uni(rng) - (double)i / (double)nv;
        } else if (kind == "uniform") {  // near-flat, nothing below floor
            logit = 1e-4 * (double)i + 0.5 * uni(rng);
        } else {                         // "tail": fallback is mandatory
            // w[0] = 1, 99 tokens at 1e-3, the rest at 5e-7 (< the 1e-6
            // floor). Candidate mass ~1.099 of total ~1.101, so the nucleus
            // tops out near 0.998: any top_p above that must widen to nv.
            const double w = (i == 0) ? 1.0 : (i < 100 ? 1e-3 : 5e-7);
            logit = std::log(w * (1.0 + 0.05 * uni(rng)));
        }
        l[(size_t)i] = (float)logit;
    }
    // Shuffle: a real logit row has no index/probability correlation, and a
    // pre-sorted row would let std::sort run cache-friendly and predicted,
    // hiding exactly the cost the pre-filter removes.
    std::shuffle(l.begin(), l.end(), std::mt19937_64(seed ^ 0xBEEF0000u));
    return l;
}

// Sample the same row with both samplers from the same seed: token-for-token
// identical, or the pre-filter changed the distribution.
static bool parity_case(const char * kind, int64_t nv, float temp, float top_p,
                        int draws) {
    const std::vector<float> row = make_row(kind, nv, 0xC0FFEEu + (uint64_t)nv);
    const int32_t am = argmax_of(row);
    std::mt19937_64 a(20261125u), b(20261125u);
    bool same = true;
    for (int d = 0; d < draws; d++) {
        const int32_t got = sample_token(row.data(), nv, temp, top_p, am, a);
        const int32_t want = sample_token_ref(row.data(), nv, temp, top_p, am, b);
        if (got != want) same = false;
    }
    char label[192];
    std::snprintf(label, sizeof(label), "parity %s nv=%lld temp=%.2f top_p=%.6g",
                  kind, (long long)nv, (double)temp, (double)top_p);
    check(same, label);
    return sampler_uses_full_sort(row.data(), nv, temp, top_p);
}

int main() {
    // ---- degenerate paths are untouched --------------------------------
    {
        std::vector<float> v = {1.0f, 5.0f, 2.0f};
        std::mt19937_64 rng(1);
        check(sample_token(v.data(), 3, 0.0f, 0.9f, 1, rng) == 1,
              "temp <= 0 is greedy (precomputed argmax)");
        check(sample_token(v.data(), 0, 0.7f, 0.9f, 7, rng) == 7, "nv <= 0 returns argmax");
        std::vector<float> one = {-1e30f, 0.0f, -1e30f};
        std::mt19937_64 r2(3), r3(3);
        bool onehot = true;
        for (int d = 0; d < 64; d++) {
            const int32_t g = sample_token(one.data(), 3, 0.7f, 0.9f, 1, r2);
            const int32_t w = sample_token_ref(one.data(), 3, 0.7f, 0.9f, 1, r3);
            if (g != 1 || g != w) onehot = false;
        }
        check(onehot, "one-hot row keeps the max token (and matches reference)");
        std::printf("ok: greedy / nv<=0 / one-hot guards\n");
    }

    // ---- parity sweeps --------------------------------------------------
    static const float temps[] = {0.01f, 0.3f, 0.7f, 1.0f, 2.0f, 5.0f};
    static const float tops[] = {0.0f, 0.5f, 0.8f, 0.9f, 0.99f, 0.9999f, 1.0f};
    static const char * kinds[] = {"peaked", "zipf", "sloppy", "uniform", "tail"};
    int cases = 0, fallback_cases = 0, filtered_cases = 0;
    for (const char * kind : kinds) {
        for (int64_t nv : {512, 4096}) {
            for (float t : temps) {
                for (float p : tops) {
                    if (parity_case(kind, nv, t, p, 32)) fallback_cases++;
                    else filtered_cases++;
                    cases++;
                }
            }
        }
    }
    std::printf("info: %d parity cases (%d pre-filtered, %d full-sort fallback)\n",
                cases, filtered_cases, fallback_cases);
    check(fallback_cases > 0, "the sweep exercises the full-sort fallback path");
    check(filtered_cases > 0, "the sweep exercises the pre-filtered path");
    std::printf("ok: tokens identical to the full-sort sampler in every case\n");

    // ---- fallback arms exactly where it must ---------------------------
    {
        const std::vector<float> row = make_row("tail", 4096, 0xC0FFEEu + 4096);
        check(sampler_uses_full_sort(row.data(), 4096, 1.0f, 0.9999f),
              "flat-tail row at top_p=0.9999 takes the full-sort fallback");
        check(!sampler_uses_full_sort(row.data(), 4096, 1.0f, 0.99f),
              "the same row at top_p=0.99 stays pre-filtered");
        check(!sampler_uses_full_sort(row.data(), 4096, 1.0f, 1.0f),
              "top_p = 1 never sorts at all");
        std::printf("ok: full-sort fallback arms exactly where it must\n");
    }

    // ---- decode-cost guard at the real vocabulary size ------------------
    {
        const int64_t nv = 201024;  // Inkling-2-Mini vocab
        const std::vector<float> row = make_row("peaked", nv, 0xC0FFEEu + nv);
        const int32_t am = argmax_of(row);
        const int iters = 16;
        // The token sink keeps the compiler from deleting the calls (the
        // samplers are side-effect free, so an unused result is dead code).
        long long sink = 0;
        auto bench = [&](bool filtered) {
            std::mt19937_64 rng(7);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++)
                sink += filtered
                            ? sample_token(row.data(), nv, 0.7f, 0.9f, am, rng)
                            : sample_token_ref(row.data(), nv, 0.7f, 0.9f, am, rng);
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        };
        bench(false); bench(true);  // warm up caches / branch predictor
        const double ref_ms = bench(false);
        const double new_ms = bench(true);
        std::printf("info: sampler cost at nv=201024, temp=0.7 top_p=0.9: "
                    "full-sort %.2f ms/tok -> pre-filtered %.3f ms/tok (%.1fx, sink=%lld)\n",
                    ref_ms, new_ms, ref_ms / (new_ms > 0 ? new_ms : 1e-9), sink);
        check(new_ms * 2.0 < ref_ms, "pre-filtered sampler is >= 2x faster at nv=201k");
    }

    if (g_fail) {
        std::printf("test_sampler: %d FAILURES\n", g_fail);
        return 1;
    }
    std::printf("test_sampler: all checks passed\n");
    return 0;
}
