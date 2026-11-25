// sampler.hpp — temperature softmax + top-p (nucleus) sampling for decode.
//
// Pure C++20 stdlib (extracted from src/main.cpp so it is unit-testable:
// engine/tests/test_sampler.cpp). temp <= 0 is greedy (returns the
// caller's precomputed argmax). Stable softmax: p_i ∝ exp((logit_i - max)/temp).
// When top_p < 1 the probs are sorted descending and truncated at the nucleus
// whose cumulative mass >= top_p; the kept mass is renormalized during a
// uniform [0, sum) scan.
//
// Candidate pre-filter (the reason this exists): sorting the WHOLE vocabulary
// costs ~9 ms/token at nv = 201k, which alone cut decode from 38.8 to
// 28.8 tok/s. Only tokens with prob[i] >= 1e-6 * prob[max_i] are collected
// first (usually a few dozen..few hundred entries, so the sort is < 0.1 ms),
// and the sort runs on those alone.
//
// Exactness: every dropped token holds less than 1e-6 of the max token's
// mass, so at most nv * 1e-6 of the total mass sits below the floor and the
// dropped tokens sort LAST. If the candidates alone already carry
// cumulative mass >= top_p, the nucleus is reached before any sub-floor token
// is touched, i.e. the nucleus is a subset of the candidate set and the
// sampled token is BITWISE identical to the old full-vocabulary sort. Only
// when the candidates cannot reach top_p (a rare flat / very-high-temperature
// distribution) does the code widen to all nv indices, which is exactly the
// old path.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

// Temperature-scaled softmax + top-p nucleus sampling over logits[0..nv).
// temp <= 0 returns argmax. rng is the caller's engine RNG (one uniform draw
// per sampled token, so identical seeds replay identical token sequences).
inline int32_t sample_token(const float * logits, int64_t nv, float temp,
                            float top_p, int32_t argmax,
                            std::mt19937_64 & rng) {
    if (temp <= 0.0f) return argmax;
    if (nv <= 0) return argmax;
    float mx = logits[0];
    int64_t mx_i = 0;
    for (int64_t i = 1; i < nv; i++)
        if (logits[i] > mx) { mx = logits[i]; mx_i = i; }
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
        // Pre-filter: the argmax token is always kept (a NaN-free floor test
        // already guarantees it, the explicit i == mx_i is belt-and-braces).
        std::vector<size_t> cand;
        cand.reserve(512);
        const double floor_p = 1e-6 * prob[(size_t)mx_i];
        double csum = 0.0;
        for (int64_t i = 0; i < nv; i++) {
            const double p = prob[(size_t)i];
            if (i == mx_i || p >= floor_p) {
                cand.push_back((size_t)i);
                csum += p;
            }
        }
        // Candidates too flat to reach top_p: fall back to the exact full sort.
        if (csum < (double)top_p * sum) {
            cand.resize((size_t)nv);
            std::iota(cand.begin(), cand.end(), (size_t)0);
        }
        std::sort(cand.begin(), cand.end(),
                  [&](size_t a, size_t b) { return prob[a] > prob[b]; });
        double cum = 0.0;
        for (size_t i : cand) {
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

// True when sample_token() would take the full-vocabulary fallback for this
// logit row (i.e. the 1e-6 pre-filter cannot reach top_p on its own). Mirrors
// the guard inside sample_token() so tests can prove the fallback is covered.
inline bool sampler_uses_full_sort(const float * logits, int64_t nv,
                                   float temp, float top_p) {
    if (temp <= 0.0f || nv <= 0 || top_p >= 1.0f) return false;
    double mx = logits[0];
    for (int64_t i = 1; i < nv; i++) mx = std::max(mx, (double)logits[i]);
    const double inv_t = 1.0 / (double)temp;
    double sum = 0.0, csum = 0.0;
    for (int64_t i = 0; i < nv; i++) {
        const double p = std::exp(((double)logits[i] - mx) * inv_t);
        sum += p;
        if (p >= 1e-6) csum += p;  // prob[max] == exp(0) == 1, so floor == 1e-6
    }
    return csum < (double)top_p * sum;
}
