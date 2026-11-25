#pragma once
// Milestone 7B — Prompt-Lookup (N-gram) speculative decoding, greedy only.
// Milestone 7C — Branch fast path (no re-verify), adaptive cooldown (auto-throttle).
//
// Pure logic, zero engine dependencies: the accept/rollback state machine is
// expressed against a tiny hooks interface so it is unit-testable with a
// mocked verifier (engine/tests/test_spec_decode.cpp). At temp == 0 the
// emitted token sequence is provably identical to plain step-by-step greedy
// decoding: every token is still produced by a model argmax (the anchor or a
// verified row), drafts only decide WHICH model positions get computed, never
// WHICH token is emitted. Adaptive cooldown only decides WHETHER a round runs
// at all, so it can never alter the greedy token sequence.
//
// Round protocol (caller owns engine state and snapshots):
//   1. snapshot engine (PrefixState) BEFORE calling spec_round,
//   2. hooks.verify() advances the engine through the whole batch and must
//      return the per-row argmax (argmax[i] = model prediction after
//      verify[0..i]),
//   3. on a mismatch the driver restores the snapshot and replays ONLY the
//      accepted prefix [anchor, d0..da-1] with hooks.advance_no_logits() —
//      ZERO LM head rows. The true branch token (amax[a]) is already known
//      from step 2, so it is handed back as next_argmax instead of being
//      forwarded/re-verified (Milestone 7C: a+2 LM rows -> 0),
//   4. the engine is always left exactly at round.emitted.size() tokens past
//      the snapshot.
//
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace m64 {

// Longest-suffix prompt lookup: try n = nmax..nmin, find the LAST occurrence
// of the trailing n-gram strictly before the suffix itself, return up to
// max_k observed tokens that followed it. Empty on no match.
inline std::vector<int32_t> lookup_draft(const std::vector<int32_t> &hist,
                                         int nmax, int nmin, size_t max_k) {
    const long len = (long)hist.size();
    for (int n = nmax; n >= nmin; n--) {
        if (n < 1 || len <= n) continue;
        for (long i = len - n - 1; i >= 0; i--) {
            long j = 0;
            for (; j < n; j++)
                if (hist[(size_t)i + (size_t)j] != hist[(size_t)(len - n) + (size_t)j]) break;
            if (j == n) {
                size_t k = std::min(max_k, (size_t)(len - (i + n)));
                return std::vector<int32_t>(hist.begin() + i + n,
                                            hist.begin() + i + n + (long)k);
            }
        }
    }
    return {};
}

struct SpecHooks {
    // Advance the engine through toks (in order); fill amax with one argmax
    // per token (amax[i] = model argmax after toks[0..i]). Returns false on
    // any engine failure (state then unspecified until restore()).
    std::function<bool(const std::vector<int32_t> &toks, std::vector<int32_t> &amax)> verify;
    // Rewind engine to the caller's round-start snapshot.
    std::function<void()> restore;
    // Milestone 7C: advance the engine through toks with ZERO LM head rows
    // (state/KV/conv only). Used after restore() on a Branch to replay only
    // the accepted prefix — the branch token itself never needs a forward.
    // Returns false on engine failure (state unspecified until restore()).
    std::function<bool(const std::vector<int32_t> &toks)> advance_no_logits;
};

struct SpecStats {
    unsigned long long drafted = 0;    // draft tokens offered
    unsigned long long accepted = 0;   // draft tokens confirmed
    unsigned long long rounds = 0;     // verify rounds run (Accepted or Branch)
    unsigned long long rollbacks = 0;  // rounds that hit a mismatch + restore
    unsigned long long branch_fast = 0; // Branches resolved with 0 re-verify rows (7C)
    unsigned long long cooldowns = 0;  // auto-throttle activations (7C)
    double accept_rate() const { return drafted ? (double)accepted / (double)drafted : 0.0; }
};

// Milestone 7C adaptive cooldown (auto-throttle). Prose with little n-gram
// repetition keeps missing the draft and each failed round still pays verify +
// restore + replay; after `zero_streak` consecutive Branch rounds with a==0
// (zero drafts accepted) the driver skips speculative rounds for
// cooldown_len plain steps (8 -> 16 -> 32, capped at 32). Any accepted draft
// clears the streak and re-enables speculation immediately. Skipping only
// removes acceleration — the greedy token sequence is untouched.
struct SpecController {
    static constexpr int kZeroStreakTrigger = 2;   // consecutive a==0 Branches
    static constexpr int kCooldownBase = 8;        // first cooldown length
    static constexpr int kCooldownMax = 32;        // backoff cap
    int zero_streak = 0;      // consecutive Branch rounds that accepted 0 drafts
    int cooldown_left = 0;    // speculative rounds still to skip
    int cooldown_len = kCooldownBase;  // length of the next cooldown
};

// Driver-side gate: true while cooling down (consumes one plain step of the
// cooldown budget). Callers skip snapshot+spec_round when it returns true.
inline bool spec_cooling_tick(SpecController &ctrl) {
    if (ctrl.cooldown_left > 0) { ctrl.cooldown_left--; return true; }
    return false;
}

enum class SpecOutcome {
    NoDraft,  // no n-gram match: caller should fall back to a plain step
    Plain,    // drafts hit ctx/budget guards: caller should fall back
    Cooling,  // auto-throttle active: caller should run plain steps
    Accepted, // whole draft verified; engine advanced by 1+drafts; next_argmax free
    Branch,   // mismatch at row a; restored + replayed [anchor, d0..da-1] with
              // NO LM head; next_argmax = true branch token (7C fast path)
    Failed,   // verify()/advance_no_logits() reported an engine failure; caller must restore()
};

struct SpecRound {
    std::vector<int32_t> emitted;  // anchor first; emit in order (eos-checked by caller)
    int32_t next_argmax = -1;      // model argmax after all emitted (bonus, no extra pass)
};

// hist = every token already in the engine state. anchor = current greedy
// argmax (known from the previous round's logits, NOT yet forwarded).
// draft_cap = max draft tokens; caller must guarantee
// pos + (draft_cap + 1) <= ctx so the worst-case verify batch fits.
inline SpecOutcome spec_round(const std::vector<int32_t> &hist, int32_t anchor,
                              size_t draft_cap, int nmax, int nmin,
                              SpecHooks &hooks, SpecRound &out, SpecStats &stats,
                              SpecController &ctrl) {
    out.emitted.clear();
    out.emitted.push_back(anchor);
    out.next_argmax = -1;

    if (ctrl.cooldown_left > 0) {  // auto-throttle: plain steps for now
        ctrl.cooldown_left--;
        return SpecOutcome::Cooling;
    }

    std::vector<int32_t> tail = hist;
    tail.push_back(anchor);
    const std::vector<int32_t> drafts = lookup_draft(tail, nmax, nmin, draft_cap);
    if (drafts.empty() || draft_cap == 0u) return SpecOutcome::NoDraft;

    std::vector<int32_t> verify;
    verify.reserve(drafts.size() + 1);
    verify.push_back(anchor);
    verify.insert(verify.end(), drafts.begin(), drafts.end());
    std::vector<int32_t> amax;
    if (!hooks.verify(verify, amax) || amax.size() != verify.size())
        return SpecOutcome::Failed;

    stats.rounds++;
    stats.drafted += drafts.size();
    size_t a = 0;
    while (a < drafts.size() && amax[a] == drafts[a]) a++;
    stats.accepted += a;

    if (a == drafts.size()) {  // all drafts confirmed; bonus argmax is row B-1
        out.emitted.insert(out.emitted.end(), drafts.begin(), drafts.end());
        out.next_argmax = amax.back();
        ctrl.zero_streak = 0;              // speculation paying off
        ctrl.cooldown_len = SpecController::kCooldownBase;
        return SpecOutcome::Accepted;
    }

    // Milestone 7C Branch fast path: amax[a] = branch is ALREADY the model
    // argmax after [anchor, d0..da-1], i.e. the true next token. No verify2,
    // no branch forward, no LM head at all — just roll back and replay the
    // accepted prefix as state-only steps.
    stats.rollbacks++;
    const int32_t branch = amax[a];
    hooks.restore();
    std::vector<int32_t> replay;
    replay.reserve(a + 1);
    replay.push_back(anchor);
    replay.insert(replay.end(), drafts.begin(), drafts.begin() + (long)a);
    if (!hooks.advance_no_logits || !hooks.advance_no_logits(replay))
        return SpecOutcome::Failed;
    out.emitted.insert(out.emitted.end(), drafts.begin(), drafts.begin() + (long)a);
    out.next_argmax = branch;
    stats.branch_fast++;
    if (a == 0u) {  // draft fully wrong: consecutive zero-accept bookkeeping
        if (++ctrl.zero_streak >= SpecController::kZeroStreakTrigger &&
            ctrl.cooldown_left == 0) {
            ctrl.cooldown_left = ctrl.cooldown_len;
            ctrl.cooldown_len = std::min(ctrl.cooldown_len * 2,
                                         SpecController::kCooldownMax);
            stats.cooldowns++;
        }
    } else {  // at least one draft confirmed: back to full speed
        ctrl.zero_streak = 0;
        ctrl.cooldown_len = SpecController::kCooldownBase;
    }
    return SpecOutcome::Branch;
}

}  // namespace m64
