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

    // ============ Milestone 8B: sampling penalties ========================
    {
        // ---- disabled penalties are a pure pass-through (bitwise) ---------
        bool bitwise = true;
        const std::vector<int32_t> hist = {0, 1, 2, 1, 300, 7, 7, 7};
        for (const char * kind : {"peaked", "zipf"}) {
            for (int64_t nv : {512, 4096}) {
                const std::vector<float> row = make_row(kind, nv, 0xA5A5u + nv);
                const int32_t am = argmax_of(row);
                for (float t : {0.0f, 0.3f, 0.7f, 2.0f}) {
                    for (float p : {0.5f, 0.9f}) {
                        std::mt19937_64 r1(77), r2(77);
                        for (int d = 0; d < 8; d++) {
                            const int32_t a = sample_token(row.data(), nv, t, p, am, r1);
                            const int32_t b = sample_token_penalized(
                                row.data(), nv, t, p, am, r2,
                                hist.data(), (int64_t)hist.size(),
                                SamplerPenalties{});
                            if (a != b) bitwise = false;
                        }
                    }
                }
            }
        }
        check(bitwise, "disabled penalties stay bitwise-identical to sample_token()");
        check(!SamplerPenalties{}.active(), "default penalties are inactive");

        // ---- penalty arithmetic is exact ----------------------------------
        {
            std::vector<float> l = {4.0f, -4.0f, 0.0f, 5.0f, -2.0f};
            const std::vector<int32_t> h = {1, 1, 1, 3};  // token1 x3, token3 x1
            SamplerPenalties pen;
            pen.presence = 1.5f; pen.frequency = 0.5f; pen.repetition = 2.0f;
            apply_sampling_penalties(l.data(), 5, h.data(), (int64_t)h.size(), pen);
            // token1: repetition first (-4 <= 0 -> *2 = -8), then -1.5 - 0.5*3
            check(l[1] == -8.0f - 1.5f - 1.5f, "frequency/repetition arithmetic");
            // token3: 5 > 0 -> /2 = 2.5, then -1.5 - 0.5
            check(l[3] == 2.5f - 1.5f - 0.5f, "positive logit division");
            check(l[0] == 4.0f && l[2] == 0.0f && l[4] == -2.0f,
                  "tokens unseen in the window stay untouched");
        }
        {
            std::vector<float> l = {0.0f};
            const std::vector<int32_t> h = {0, 0, 0};
            SamplerPenalties pen;
            pen.repetition = 2.0f;
            apply_sampling_penalties(l.data(), 1, h.data(), 3, pen);
            check(l[0] == 0.0f, "zero logit follows the <= 0 branch (stays 0)");
            SamplerPenalties p2;
            p2.presence = 2.0f;
            std::vector<float> l2 = {3.0f, 3.0f};
            const std::vector<int32_t> h3 = {0, 0, 0};
            apply_sampling_penalties(l2.data(), 2, h3.data(), 3, p2);
            check(l2[0] == 1.0f, "presence fires once regardless of count");
        }
        {   // window: only the last penalty_last_n ids count
            std::vector<float> l = {9.0f, 9.0f};
            const std::vector<int32_t> h = {0, 0, 1, 1, 1};
            SamplerPenalties pen;
            pen.presence = 1.0f;
            pen.last_n = 3;
            apply_sampling_penalties(l.data(), 2, h.data(), 5, pen);
            check(l[0] == 9.0f && l[1] == 8.0f,
                  "penalty window covers only the last last_n history ids");
        }
        {   // out-of-vocab ids ignored; last_n = 0 never penalizes
            std::vector<float> l = {1.0f};
            const std::vector<int32_t> h = {42, -3};
            SamplerPenalties pen;
            pen.presence = 1.0f;
            apply_sampling_penalties(l.data(), 1, h.data(), 2, pen);
            check(l[0] == 1.0f, "out-of-range history ids ignored");
            SamplerPenalties p0;
            p0.presence = 1.0f; p0.last_n = 0;
            apply_sampling_penalties(l.data(), 1, h.data(), 2, p0);
            check(l[0] == 1.0f, "last_n = 0 never penalizes");
        }

        // ---- greedy picks the POST-penalty argmax --------------------------
        {
            std::vector<float> l = {10.0f, 9.0f, 8.0f};
            const std::vector<int32_t> h = {0, 0, 0};
            std::mt19937_64 rng(1);
            SamplerPenalties pen;
            pen.presence = 2.0f;  // token0 -> 8, argmax moves to token 1
            check(sample_token_penalized(l.data(), 3, 0.0f, 0.9f, 0, rng,
                                         h.data(), 3, pen) == 1,
                  "temp <= 0 with penalties picks the post-penalty argmax");
            SamplerPenalties rep;
            rep.repetition = 2.0f;  // token0 -> 5, still below 9
            check(sample_token_penalized(l.data(), 3, 0.0f, 0.9f, 0, rng,
                                         h.data(), 3, rep) == 1,
                  "repetition_penalty alone steers greedy away from repeats");
        }

        // ---- loop suppression: the milestone's whole reason to exist -------
        {
            // Attractor row: greedy WITHOUT penalties emits token 0 forever.
            std::vector<float> l(16, 1.0f);
            l[0] = 3.0f;
            SamplerPenalties pen;
            pen.presence = 0.5f;
            pen.frequency = 1.0f;
            std::vector<int32_t> hist = {0};
            std::vector<int32_t> emitted;
            std::mt19937_64 rng(9);
            for (int step = 0; step < 10; step++) {
                const int32_t t = sample_token_penalized(
                    l.data(), 16, 0.0f, 1.0f, 0, rng,
                    hist.data(), (int64_t)hist.size(), pen);
                emitted.push_back(t);
                hist.push_back(t);
            }
            int max_rep = 0;
            for (int tok = 0; tok < 16; tok++)
                max_rep = std::max(max_rep,
                                   (int)std::count(emitted.begin(), emitted.end(), tok));
            check(max_rep <= 3, "penalties break a greedy repeat loop (max repeat <= 3)");
        }

        // ---- pre-filter stays intact under penalties -----------------------
        // Sampling the ORIGINAL row with the penalty sampler must equal the
        // full-sort reference on the manually penalized row (same seed):
        // copy+penalize cannot change the distribution nor RNG consumption.
        {
            bool pen_parity = true;
            const std::vector<int32_t> phist = {5, 9, 9, 777, 9, 5, 12, 300};
            for (const char * kind : {"peaked", "zipf", "tail"}) {
                const int64_t nv = 4096;
                const std::vector<float> row = make_row(kind, nv, 0xBEEF01u + nv);
                std::vector<float> adj = row;
                SamplerPenalties pen;
                pen.presence = 0.8f; pen.frequency = 0.4f; pen.repetition = 1.15f;
                apply_sampling_penalties(adj.data(), nv, phist.data(),
                                         (int64_t)phist.size(), pen);
                const int32_t am_adj = argmax_of(adj);
                const int32_t am_row = argmax_of(row);
                for (float p : {0.5f, 0.9f, 0.99f}) {
                    std::mt19937_64 r1(404), r2(404);
                    for (int d = 0; d < 16; d++) {
                        const int32_t got = sample_token_penalized(
                            row.data(), nv, 0.7f, p, am_row, r1,
                            phist.data(), (int64_t)phist.size(), pen);
                        const int32_t want = sample_token_ref(adj.data(), nv, 0.7f, p,
                                                              am_adj, r2);
                        if (got != want) pen_parity = false;
                    }
                }
            }
            check(pen_parity,
                  "penalized sampler == full-sort reference on the penalized row");
        }

        // ---- GEN penalty-options parser ------------------------------------
        {
            SamplerPenalties pen;
            check(sampler_parse_penalty_opt("presence_penalty=1.5", pen) == 1 &&
                      pen.presence == 1.5f, "kv: presence_penalty parsed");
            check(sampler_parse_penalty_opt("frequency_penalty=-0.25", pen) == 1 &&
                      pen.frequency == -0.25f, "kv: frequency_penalty parsed");
            check(sampler_parse_penalty_opt("repetition_penalty=1.1", pen) == 1 &&
                      pen.repetition == 1.1f, "kv: repetition_penalty parsed");
            check(sampler_parse_penalty_opt("penalty_last_n=64", pen) == 1 &&
                      pen.last_n == 64, "kv: penalty_last_n parsed");
            check(sampler_parse_penalty_opt("--", pen) == 0, "-- is not an option");
            check(sampler_parse_penalty_opt("42", pen) == 0, "bare number is not an option");
            check(sampler_parse_penalty_opt("foo=1", pen) == 0, "unknown key -> 0");
            check(sampler_parse_penalty_opt("presence_penalty=abc", pen) == -1,
                  "garbage float rejected");
            check(sampler_parse_penalty_opt("presence_penalty=nan", pen) == -1,
                  "NaN rejected");
            check(sampler_parse_penalty_opt("frequency_penalty=1x", pen) == -1,
                  "trailing junk rejected");
            check(sampler_parse_penalty_opt("repetition_penalty=0", pen) == -1,
                  "repetition_penalty <= 0 rejected");
            check(sampler_parse_penalty_opt("repetition_penalty=-2", pen) == -1,
                  "negative repetition_penalty rejected");
            check(sampler_parse_penalty_opt("penalty_last_n=-5", pen) == -1,
                  "negative last_n rejected");
            check(pen.active(), "parser marks the struct active after a real value");
        }
        std::printf("ok: penalties (bitwise-off / arithmetic / window / greedy / "
                    "loop suppression / pre-filter parity / kv parser)\n");
    }

    if (g_fail) {
        std::printf("test_sampler: %d FAILURES\n", g_fail);
        return 1;
    }
    std::printf("test_sampler: all checks passed\n");
    return 0;
}
