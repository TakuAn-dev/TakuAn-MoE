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
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Sampling penalties (Milestone 8B). All disabled by default; when disabled
// the sampler MUST stay bitwise-identical to the original fast path.
struct SamplerPenalties {
    float presence = 0.0f;   // OpenAI: -presence on tokens with count > 0
    float frequency = 0.0f;  // OpenAI: -frequency * count on tokens with count > 0
    float repetition = 1.0f; // != 1: logit/rep if > 0 else logit*rep (once per token)
    int last_n = 256;        // penalty window: last `last_n` history tokens

    bool active() const {
        return presence != 0.0f || frequency != 0.0f || repetition != 1.0f;
    }
};

// Penalize logits[0..nv) in place from a generation history window (only the
// last `pen.last_n` entries count). Out-of-vocabulary history ids are ignored.
// Per unique token: repetition first (sign-aware, once), then the additive
// presence/frequency penalties computed in double.
inline void apply_sampling_penalties(float * logits, int64_t nv,
                                     const int32_t * history, int64_t n_hist,
                                     const SamplerPenalties & pen) {
    if (!pen.active() || nv <= 0 || n_hist <= 0) return;
    if (pen.last_n <= 0) return;
    if (n_hist > pen.last_n) {
        history += (n_hist - pen.last_n);
        n_hist = pen.last_n;
    }
    std::unordered_map<int32_t, int> counts;
    for (int64_t i = 0; i < n_hist; i++) {
        const int32_t t = history[i];
        if (t >= 0 && t < (int32_t)nv) counts[t]++;
    }
    for (const auto & kv : counts) {
        double v = logits[kv.first];
        if (pen.repetition != 1.0f)
            v = v > 0.0 ? v / pen.repetition : v * pen.repetition;
        v -= pen.presence;                 // presence: once when count > 0
        v -= (double)pen.frequency * kv.second;
        logits[kv.first] = (float)v;
    }
}

// Parse one optional "key=value" penalty token from a GEN worker line (the
// tokens between <eos_csv> and "--"). Recognized keys: presence_penalty,
// frequency_penalty, repetition_penalty (must be > 0), penalty_last_n (>= 0).
// Returns: 1 = recognized and stored; 0 = not a penalty option (missing '=',
// unknown key); -1 = recognized key with malformed/invalid value.
inline int sampler_parse_penalty_opt(const std::string & tok,
                                     SamplerPenalties & pen) {
    const size_t eq = tok.find('=');
    if (eq == std::string::npos) return 0;
    const std::string key = tok.substr(0, eq);
    const std::string val = tok.substr(eq + 1);
    auto parse_float = [](const std::string & s, float & out) {
        try {
            size_t pos = 0;
            const float f = std::stof(s, &pos);
            if (pos != s.size() || !std::isfinite(f)) return false;
            out = f;
            return true;
        } catch (...) { return false; }
    };
    if (key == "presence_penalty")
        return parse_float(val, pen.presence) ? 1 : -1;
    if (key == "frequency_penalty")
        return parse_float(val, pen.frequency) ? 1 : -1;
    if (key == "repetition_penalty") {
        float f = 1.0f;
        if (!parse_float(val, f) || f <= 0.0f) return -1;
        pen.repetition = f;
        return 1;
    }
    if (key == "penalty_last_n") {
        try {
            size_t pos = 0;
            const int n = std::stoi(val, &pos);
            if (pos != val.size() || n < 0) return -1;
            pen.last_n = n;
            return 1;
        } catch (...) { return -1; }
    }
    return 0;
}

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

// Penalty-aware sampler. With all penalties disabled this dispatches to the
// exact original sample_token() (bitwise identical). With penalties active the
// row is copied, penalized, its argmax recomputed (so temp <= 0 picks the
// post-penalty argmax), and the ORIGINAL core sampler runs on the penalized
// row — the 1e-6 candidate pre-filter and RNG consumption stay intact.
inline int32_t sample_token_penalized(const float * logits, int64_t nv,
                                      float temp, float top_p, int32_t argmax,
                                      std::mt19937_64 & rng,
                                      const int32_t * history, int64_t n_hist,
                                      const SamplerPenalties & pen) {
    if (!pen.active() || nv <= 0) return sample_token(logits, nv, temp, top_p, argmax, rng);
    std::vector<float> adj((size_t)nv);
    std::copy(logits, logits + nv, adj.begin());
    apply_sampling_penalties(adj.data(), nv, history, n_hist, pen);
    int32_t am = 0;
    if (nv > 0) {
        for (int64_t i = 1; i < nv; i++)
            if (adj[(size_t)i] > adj[(size_t)am]) am = (int32_t)i;
    }
    return sample_token(adj.data(), nv, temp, top_p, am, rng);
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
