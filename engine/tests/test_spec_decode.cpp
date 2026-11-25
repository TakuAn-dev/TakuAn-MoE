// test_spec_decode.cpp — Milestone 7B/7C unit tests for the pure prompt-lookup
// speculative decode state machine (src/spec_decode.hpp) against a
// deterministic MOCK model. Verifies the hard contract:
//   temp==0 greedy + speculative  ==  plain step-by-step greedy,
//   token-for-token identical, for every acceptance pattern (all-match,
//   branch/rollback, mid-round eos truncation, tight token limits, sustained
//   misses + adaptive cooldown), and the engine-state invariant
//   engine_tokens == hist at every round boundary.
// 7C adds: Branch runs verify() exactly ONCE (no verify2) and replays the
// accepted prefix via advance_no_logits() (zero LM rows); the cooldown
// auto-throttle arms after 2 consecutive zero-accept Branches (8 -> 16 -> 32
// skipped rounds), recovers the moment any draft is accepted, and never
// changes the emitted token sequence.
#include "spec_decode.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

namespace {

int failures = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) { std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
    } while (0)

// Deterministic mock engine. verify() advances state through EVERY row like
// the real gpu_prefill_batch pass; advance_no_logits() does the same WITHOUT
// counting an LM head row (7C); rollback is snapshot/restore like the real
// PrefixState path.
struct MockEngine {
    std::vector<int32_t> toks;
    int (*next_fn)(const std::vector<int32_t> &);
    unsigned rounds = 0, restore_calls = 0, verify_rows = 0;
    unsigned advance_calls = 0, advance_rows = 0;

    explicit MockEngine(std::vector<int32_t> seed, int (*fn)(const std::vector<int32_t> &))
        : toks(std::move(seed)), next_fn(fn) {}

    int32_t argmax() const { return next_fn(toks); }
    int32_t step(int32_t t) { toks.push_back(t); return next_fn(toks); }
    bool verify(const std::vector<int32_t> & t, std::vector<int32_t> & amax) {
        rounds++;
        amax.clear();
        for (int32_t x : t) { toks.push_back(x); amax.push_back(next_fn(toks)); }
        verify_rows += (unsigned)t.size();
        return true;
    }
    // 7C fast path: state-only advance — the mock's "argmax" for the next
    // anchor is recomputed from state, mirroring a plain gpu_step replay.
    bool advance_no_logits(const std::vector<int32_t> & t) {
        advance_calls++;
        for (int32_t x : t) toks.push_back(x);
        advance_rows += (unsigned)t.size();
        return true;
    }
};

int32_t cycle_next(const std::vector<int32_t> & ctx) {
    const int32_t last = ctx.empty() ? 0 : ctx.back();
    return last == 0 ? 17 : last == 17 ? 5 : 0;  // 3-cycle: prompt-lookup drafts always match
}

// Successor independent of history repeats: prompt-lookup finds no usable
// n-gram continuation -> sustained NoDraft fallback (pure plain steps).
int32_t successor_next(const std::vector<int32_t> & ctx) {
    const int32_t last = ctx.empty() ? 0 : ctx.back();
    return (last + 1) % 97 + 50;
}

// Scripted branch generator for the end-to-end loop: prompt-lookup offers
// drafts [8,1,2,3] after anchor 3 (from history [1,2,3,8,..]) but the model
// says 9 -> a==0 Branch with the 7C zero-LM-row replay, then the loop keeps
// exactness through the NoDraft plain fallback (3/9 alternating).
int32_t three_next(const std::vector<int32_t> & ctx) {
    const int32_t last = ctx.empty() ? 0 : ctx.back();
    return last == 2 ? 3 : last == 3 ? 9 : 3;
}

std::vector<int32_t> plain_greedy(const std::vector<int32_t> & prompt,
                                  int (*fn)(const std::vector<int32_t> &),
                                  int limit, const std::set<int32_t> & eos) {
    MockEngine eng(prompt, fn);
    int32_t am = eng.argmax();
    std::vector<int32_t> out;
    for (int k = 0; k < limit; k++) {
        if (eos.count(am)) break;
        out.push_back(am);
        if (k + 1 < limit) am = eng.step(am);
    }
    return out;
}

struct SpecRunResult {
    std::vector<int32_t> out;
    m64::SpecStats stats;
    m64::SpecController ctrl;
    bool engine_ok = false;  // engine == prompt + out[0..k), k in {out-1, out}
    bool invariant_ok = true;  // engine == hist at every round boundary
    unsigned cooling_skips = 0;  // rounds skipped via the driver-side tick
};

// Mirrors the worker loop in main.cpp (same cap guards, early anchor stream,
// cooldown tick, same rewind/replay).
SpecRunResult spec_greedy(const std::vector<int32_t> & prompt,
                          int (*fn)(const std::vector<int32_t> &),
                          int limit, int K, const std::set<int32_t> & eos) {
    SpecRunResult res;
    MockEngine eng(prompt, fn);
    int32_t am = eng.argmax();
    std::vector<int32_t> hist = prompt;
    int n_gen = 0;
    while (n_gen < limit) {
        // 7C early streaming: the anchor is confirmed before the round runs.
        if (eos.count(am)) break;
        res.out.push_back(am);
        n_gen++;

        size_t cap = (size_t)K;
        const size_t slots = (size_t)(limit - n_gen);
        if (cap > slots) cap = slots;

        m64::SpecHooks hooks;
        std::vector<int32_t> snap;
        const size_t base = hist.size();
        hooks.restore = [&]() { eng.toks = snap; eng.restore_calls++; };
        hooks.verify = [&](const std::vector<int32_t> & t, std::vector<int32_t> & amax) {
            return eng.verify(t, amax);
        };
        hooks.advance_no_logits = [&](const std::vector<int32_t> & t) {
            return eng.advance_no_logits(t);
        };
        m64::SpecRound rd;
        m64::SpecOutcome oc = m64::SpecOutcome::NoDraft;
        const bool cooling = m64::spec_cooling_tick(res.ctrl);
        if (cooling) res.cooling_skips++;
        if (cap > 0u && !cooling) {
            snap = eng.toks;  // save_prefix_state(snap, hist) equivalent
            if (eng.toks != hist) res.invariant_ok = false;
            oc = m64::spec_round(hist, am, cap, 4, 3, hooks, rd, res.stats, res.ctrl);
        }
        if (oc == m64::SpecOutcome::Accepted || oc == m64::SpecOutcome::Branch) {
            bool cut = false;
            for (size_t j = 0; j < rd.emitted.size(); j++) {
                if (j > 0u) {
                    if (n_gen >= limit) { cut = true; break; }
                    if (eos.count(rd.emitted[j])) { cut = true; break; }
                    res.out.push_back(rd.emitted[j]);
                    n_gen++;
                }
                hist.push_back(rd.emitted[j]);
            }
            if (cut) {  // engine holds every emitted row: rewind + replay
                eng.toks = snap;
                eng.restore_calls++;
                for (size_t j = base; j < hist.size(); j++) eng.step(hist[j]);
                break;
            }
            am = rd.next_argmax;
        } else {
            if (oc == m64::SpecOutcome::Failed) { eng.toks = snap; }
            // anchor already streamed above; just advance state (7C loop)
            if (n_gen < limit) {
                const int32_t t = am;
                am = eng.step(t);
                hist.push_back(t);
            }
        }
    }
    // Engine must hold prompt + every emitted token, minus at most the final
    // pending token (the plain tail never steps the last emission — same as
    // the real worker's `processed` list).
    const size_t n = res.out.size();
    if (eng.toks.size() != prompt.size() + n && eng.toks.size() != prompt.size() + n - 1u)
        res.engine_ok = false;
    else {
        res.engine_ok = true;
        for (size_t i = 0; i < eng.toks.size() - prompt.size(); i++)
            if (eng.toks[prompt.size() + i] != res.out[i]) res.engine_ok = false;
    }
    return res;
}

void run_case(const char * name, const std::vector<int32_t> & prompt,
              int (*fn)(const std::vector<int32_t> &), int limit, int K,
              const std::set<int32_t> & eos, bool expect_accepts, bool expect_rollbacks) {
    const std::vector<int32_t> base = plain_greedy(prompt, fn, limit, eos);
    const SpecRunResult sp = spec_greedy(prompt, fn, limit, K, eos);
    CHECK(sp.out == base, name);
    CHECK(sp.engine_ok, name);
    CHECK(sp.invariant_ok, name);
    if (expect_accepts) CHECK(sp.stats.accepted > 0, name);
    if (expect_rollbacks) CHECK(sp.stats.rollbacks > 0, name);
    std::printf("ok: %-46s tokens=%zu rounds=%llu accepted=%llu/%llu rollbacks=%llu "
                "branch_fast=%llu cooldowns=%llu cooling_skips=%u\n",
                name, sp.out.size(), sp.stats.rounds, sp.stats.accepted,
                sp.stats.drafted, sp.stats.rollbacks, sp.stats.branch_fast,
                sp.stats.cooldowns, sp.cooling_skips);
}

// Scripted verifier drives the spec_round state machine directly: forces the
// Branch (mid-draft mismatch), Failed, and deep-Accept paths — the patterns a
// small deterministic mock model cannot produce (they are always fully
// predictable by prompt-lookup, hence always Accepted).
void test_scripted_rounds() {
    // hist + anchor=3 gives tail {9,1,2,3,8,1,2,3,7,1,2,3}; the last 3-gram
    // [1,2,3] matches the occurrence at index 5 -> drafts [7,1,2,3] (cap=4).
    const std::vector<int32_t> hist = {9, 1, 2, 3, 8, 1, 2, 3, 7, 1, 2};
    const int32_t anchor = 3;

    // --- deep Accept: amax matches every draft, next = amax.back() ---
    {
        std::vector<std::vector<int32_t>> script = {{3, 7, 1, 2, 3}};  // rows
        std::vector<std::vector<int32_t>> answers = {{7, 1, 2, 3, 42}};  // all drafts match + bonus
        size_t calls = 0;
        int restores = 0, advances = 0;
        m64::SpecHooks hooks;
        hooks.restore = [&]() { restores++; };
        hooks.verify = [&](const std::vector<int32_t> & t, std::vector<int32_t> & amax) {
            CHECK(calls < script.size() && t == script[calls], "accept: verify rows = anchor+drafts");
            amax = answers[calls++];
            return true;
        };
        hooks.advance_no_logits = [&](const std::vector<int32_t> &) { advances++; return true; };
        m64::SpecRound rd;
        m64::SpecStats st;
        m64::SpecController ctrl;
        const m64::SpecOutcome oc = m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl);
        CHECK(oc == m64::SpecOutcome::Accepted, "accept: outcome");
        CHECK(rd.emitted.size() == 5 && rd.emitted[0] == 3 && rd.emitted[1] == 7, "accept: emitted = anchor+drafts");
        CHECK(rd.next_argmax == 42, "accept: next = bonus argmax");
        CHECK(st.accepted == 4 && st.drafted == 4 && st.rollbacks == 0 && st.rounds == 1, "accept: stats");
        CHECK(restores == 0 && advances == 0, "accept: no restore/advance");
        CHECK(ctrl.zero_streak == 0 && ctrl.cooldown_left == 0, "accept: controller reset");
        std::printf("ok: scripted deep-accept round\n");
    }

    // --- 7C Branch: mismatch at draft[1] -> ONE verify only (no verify2);
    //     restore + state-only replay of [anchor, accepted prefix]; the true
    //     branch token comes back as next_argmax with zero extra forwards. ---
    {
        std::vector<std::vector<int32_t>> script = {{3, 7, 1, 2, 3}};
        std::vector<std::vector<int32_t>> answers = {{7, 99, 55, 66, 77}};
        size_t calls = 0;
        int restores = 0;
        std::vector<std::vector<int32_t>> replayed;
        m64::SpecHooks hooks;
        hooks.restore = [&]() { restores++; };
        hooks.verify = [&](const std::vector<int32_t> & t, std::vector<int32_t> & amax) {
            CHECK(calls < script.size() && t == script[calls], "branch: verify rows");
            amax = answers[calls++];
            return true;
        };
        hooks.advance_no_logits = [&](const std::vector<int32_t> & t) {
            replayed.push_back(t);
            return true;
        };
        m64::SpecRound rd;
        m64::SpecStats st;
        m64::SpecController ctrl;
        const m64::SpecOutcome oc = m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl);
        CHECK(oc == m64::SpecOutcome::Branch, "branch: outcome");
        CHECK(rd.emitted.size() == 2 && rd.emitted[0] == 3 && rd.emitted[1] == 7,
              "branch: emitted = anchor + accepted prefix (branch NOT emitted)");
        CHECK(rd.next_argmax == 99, "branch: next_argmax = true branch token");
        CHECK(restores == 1, "branch: exactly one restore");
        CHECK(calls == 1, "branch: verify called exactly ONCE (no verify2)");
        CHECK(replayed.size() == 1 && replayed[0].size() == 2 && replayed[0][0] == 3 && replayed[0][1] == 7,
              "branch: advance_no_logits replays [anchor, accepted prefix]");
        CHECK(st.accepted == 1 && st.rollbacks == 1 && st.rounds == 1 && st.branch_fast == 1,
              "branch: stats");
        CHECK(ctrl.zero_streak == 0, "branch: a>=1 clears zero_streak");
        std::printf("ok: scripted 7C branch fast path (1 verify, 0 LM rows)\n");
    }

    // --- 7C Branch with a==0: replay is the anchor ALONE (1 state-only step) ---
    {
        std::vector<std::vector<int32_t>> script = {{3, 7, 1, 2, 3}};
        std::vector<std::vector<int32_t>> answers = {{99, 55, 66, 77, 88}};
        size_t calls = 0;
        std::vector<std::vector<int32_t>> replayed;
        m64::SpecHooks hooks;
        hooks.restore = [&]() {};
        hooks.verify = [&](const std::vector<int32_t> & t, std::vector<int32_t> & amax) {
            amax = answers[calls++];
            return true;
        };
        hooks.advance_no_logits = [&](const std::vector<int32_t> & t) {
            replayed.push_back(t);
            return true;
        };
        m64::SpecRound rd;
        m64::SpecStats st;
        m64::SpecController ctrl;
        const m64::SpecOutcome oc = m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl);
        CHECK(oc == m64::SpecOutcome::Branch, "branch0: outcome");
        CHECK(rd.emitted.size() == 1 && rd.emitted[0] == 3, "branch0: emitted = anchor only");
        CHECK(rd.next_argmax == 99, "branch0: next_argmax = branch");
        CHECK(calls == 1 && replayed.size() == 1 && replayed[0].size() == 1 && replayed[0][0] == 3,
              "branch0: anchor-only state step, no verify2");
        CHECK(ctrl.zero_streak == 1, "branch0: zero_streak=1");
        std::printf("ok: scripted a==0 branch (anchor-only replay)\n");
    }

    // --- 7C adaptive cooldown: 2 consecutive a==0 Branches arm an 8-round
    //     cooldown (Cooling skips verify entirely); backoff doubles on the
    //     next arm; any accepted draft resets it. ---
    {
        std::vector<std::vector<int32_t>> script = {{3, 7, 1, 2, 3}};
        size_t calls = 0;
        int advances = 0;
        bool all_bad = true;  // amax[0] never matches draft[0]=7
        m64::SpecHooks hooks;
        hooks.restore = [&]() {};
        hooks.verify = [&](const std::vector<int32_t> &, std::vector<int32_t> & amax) {
            calls++;
            amax = all_bad ? std::vector<int32_t>{99, 1, 2, 3, 4}
                           : std::vector<int32_t>{7, 1, 2, 3, 4};
            return true;
        };
        hooks.advance_no_logits = [&](const std::vector<int32_t> &) { advances++; return true; };
        m64::SpecStats st;
        m64::SpecController ctrl;
        m64::SpecRound rd;
        // Round 1 + 2: a==0 Branches; second one arms the cooldown.
        for (int i = 0; i < 2; i++)
            CHECK(m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl) ==
                      m64::SpecOutcome::Branch, "cooldown: zero-accept branch round");
        CHECK(st.cooldowns == 1 && ctrl.cooldown_left == 8 && ctrl.cooldown_len == 16,
              "cooldown: armed 8 steps after 2 zero-accepts, next backoff 16");
        // Rounds 3..10: Cooling, no verify, no advance, no engine touch.
        for (int i = 0; i < 8; i++) {
            CHECK(m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl) ==
                      m64::SpecOutcome::Cooling, "cooldown: skip");
            CHECK(rd.emitted.size() == 1 && rd.emitted[0] == anchor, "cooldown: no emissions");
        }
        CHECK(calls == 2 && advances == 2, "cooldown: verify frozen during cooldown");
        CHECK(ctrl.cooldown_left == 0, "cooldown: budget spent");
        // Recovery: round 11 runs again (still a==0; streak 3 re-arms at 16).
        CHECK(m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl) ==
                  m64::SpecOutcome::Branch, "cooldown: round resumes after cooldown");
        CHECK(ctrl.cooldown_left == 16 && ctrl.cooldown_len == 32, "cooldown: backoff doubled to 16");
        // Accept a draft mid-16-cooldown window: full recovery, streak cleared.
        for (int i = 0; i < 4; i++)
            CHECK(m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl) ==
                      m64::SpecOutcome::Cooling, "cooldown: draining 4 of 16");
        all_bad = false;
        // Drain the remaining 12 manually, then one accepted round resets state.
        while (ctrl.cooldown_left > 0)
            CHECK(m64::spec_cooling_tick(ctrl), "cooldown: driver tick drains");
        CHECK(m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl) ==
                  m64::SpecOutcome::Accepted, "cooldown: accept after recovery");
        CHECK(ctrl.zero_streak == 0 && ctrl.cooldown_left == 0 &&
                  ctrl.cooldown_len == m64::SpecController::kCooldownBase,
              "cooldown: accept resets streak + backoff");
        std::printf("ok: scripted adaptive cooldown (arm/skip/backoff/recover)\n");
    }

    // --- Failed: verify error -> caller must restore, no stats change ---
    {
        int restores = 0;
        m64::SpecHooks hooks;
        hooks.restore = [&]() { restores++; };
        hooks.verify = [&](const std::vector<int32_t> &, std::vector<int32_t> & amax) {
            amax.clear();
            return false;
        };
        m64::SpecRound rd;
        m64::SpecStats st;
        m64::SpecController ctrl;
        const m64::SpecOutcome oc = m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl);
        CHECK(oc == m64::SpecOutcome::Failed, "failed: outcome");
        CHECK(st.rounds == 0 && st.drafted == 0, "failed: stats untouched");
        std::printf("ok: scripted verify-failure round\n");
    }

    // --- Failed via advance_no_logits: branch replay error -> Failed (caller restores) ---
    {
        m64::SpecHooks hooks;
        hooks.restore = [&]() {};
        hooks.verify = [&](const std::vector<int32_t> &, std::vector<int32_t> & amax) {
            amax = {99, 99, 99, 99, 99};
            return true;
        };
        hooks.advance_no_logits = [&](const std::vector<int32_t> &) { return false; };
        m64::SpecRound rd;
        m64::SpecStats st;
        m64::SpecController ctrl;
        CHECK(m64::spec_round(hist, anchor, 4, 4, 3, hooks, rd, st, ctrl) ==
                  m64::SpecOutcome::Failed, "advance-fail: outcome");
        std::printf("ok: scripted advance_no_logits failure round\n");
    }

    // --- NoDraft when cap is 0 ---
    {
        m64::SpecHooks hooks;
        hooks.verify = [&](const std::vector<int32_t> &, std::vector<int32_t> &) { return true; };
        m64::SpecRound rd;
        m64::SpecStats st;
        m64::SpecController ctrl;
        CHECK(m64::spec_round(hist, anchor, 0, 4, 3, hooks, rd, st, ctrl) == m64::SpecOutcome::NoDraft,
              "cap=0 -> NoDraft");
        std::printf("ok: cap=0 NoDraft\n");
    }
}

void test_lookup_draft() {
    using m64::lookup_draft;
    const std::vector<int32_t> h = {1, 2, 3, 4, 5};
    CHECK(lookup_draft(h, 4, 3, 8).empty(), "lookup: no match -> empty");
    const std::vector<int32_t> h2 = {1, 2, 3, 4, 1, 2, 3};
    const std::vector<int32_t> d = lookup_draft(h2, 4, 3, 8);
    CHECK(d.size() == 4 && d[0] == 4 && d[1] == 1 && d[2] == 2 && d[3] == 3,
          "lookup: 3-gram match, longest-first, last occurrence");
    const std::vector<int32_t> d2 = lookup_draft(h2, 4, 3, 2);
    CHECK(d2.size() == 2 && d2[0] == 4 && d2[1] == 1, "lookup: max_k caps depth");
    const std::vector<int32_t> h3 = {1, 2, 1, 2, 1};
    const std::vector<int32_t> d3 = lookup_draft(h3, 2, 2, 1);
    CHECK(d3.size() == 1 && d3[0] == 2, "lookup: n=2 last-occurrence lookahead");
}

}  // namespace

int main() {
    test_lookup_draft();
    test_scripted_rounds();

    // Periodic 3-cycle prompt: drafts match -> deep acceptance.
    std::vector<int32_t> prompt = {3, 9, 21, 8, 13};
    for (int i = 0; i < 24; i++) prompt.push_back(cycle_next(prompt));
    run_case("cycle K=4 exact vs plain", prompt, cycle_next, 32, 4, {}, true, false);
    run_case("cycle K=8 exact vs plain", prompt, cycle_next, 32, 8, {}, true, false);
    run_case("cycle K=1 exact vs plain", prompt, cycle_next, 32, 1, {}, true, false);
    // Tight limits exercise the slot cap / plain-fallback guard.
    run_case("cycle limit=1 guard", prompt, cycle_next, 1, 4, {}, false, false);
    run_case("cycle limit=2 guard", prompt, cycle_next, 2, 4, {}, false, false);
    run_case("cycle limit=3 guard", prompt, cycle_next, 3, 4, {}, false, false);
    // Mid-round eos truncation: engine must be rewound+replayed exactly.
    run_case("cycle eos mid-round truncation", prompt, cycle_next, 32, 4, {5}, false, false);

    // End-to-end Branch: prompt-lookup offers drafts that the model rejects
    // (a==0) -> 7C fast-path rollback (one verify + state-only replay), then
    // plain fallback keeps the sequence token-for-token equal to greedy.
    run_case("three K=4 branch exact vs plain", {4, 1, 2, 3, 8, 1, 2}, three_next,
             40, 4, {}, false, true);
    {
        const SpecRunResult sp = spec_greedy({4, 1, 2, 3, 8, 1, 2}, three_next, 40, 4, {});
        CHECK(sp.stats.branch_fast >= 1, "7C fast path used on the scripted branch");
        CHECK(sp.stats.rollbacks >= 1, "scripted branch rolled back");
        std::printf("ok: end-to-end a==0 branch uses the 7C fast path\n");
    }
    // Sustained NoDraft prompt (no usable n-gram): pure plain fallback.
    run_case("successor K=4 exact vs plain (no drafts)", {7, 1, 2, 3, 1, 2, 3, 1, 2},
             successor_next, 40, 4, {}, false, false);
    {
        const SpecRunResult sp = spec_greedy({7, 1, 2, 3, 1, 2, 3, 1, 2}, successor_next, 40, 4, {});
        CHECK(sp.stats.rounds == 0 && sp.stats.drafted == 0, "no drafts drafted");
        std::printf("ok: unmatched history falls back to plain steps\n");
    }
    // Cooldown must never arm when drafts pay off: deep-accept prompt.
    {
        const SpecRunResult sp = spec_greedy(prompt, cycle_next, 40, 4, {});
        CHECK(sp.stats.cooldowns == 0, "cooldown never arms when drafts pay off");
        std::printf("ok: no cooldown while accept rate is high\n");
    }

    if (failures) {
        std::printf("SPEC DECODE: %d FAILURES\n", failures);
        return 1;
    }
    std::printf("SPEC DECODE: ALL TESTS PASSED (0 failures)\n");
    return 0;
}
