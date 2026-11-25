// test_prefix_stash.cpp — unit tests for the RAM prefix-stash policy
// (prefix_stash.hpp) plus a mock-engine replay of the production sequence:
// long conversation -> interrupting short prompt -> second interrupt ->
// resume. Mirrors the mock-engine convention of test_spec_decode.cpp: the
// tiny test fixture cannot run the real Metal engine (hparams gate), so the
// fake engine below reproduces the *state bookkeeping* the worker loop
// relies on: in-place full-KV slots, reset+re-prefill clobbering them,
// capture_full_kv / restore_prefix_state bit copy, invalidate_after(0),
// keep-the-longer-stash across two interrupts, warm-prefix survival, and
// stash release after resume. Bit-exactness of the stash vs a full
// re-prefill is checked byte-for-byte on the fake kvBuf; the real-model
// bitwise gate lives in scripts/gpu_validation_suite.sh step 5.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "prefix_stash.hpp"

namespace {

int g_fail = 0;
void check(bool ok, const char * what) {
    if (ok) {
        std::printf("ok: %s\n", what);
    } else {
        std::printf("FAIL: %s\n", what);
        g_fail++;
    }
}

// ---- policy: select_stash --------------------------------------------------
void test_select_stash() {
    using namespace m64;
    check(select_stash({}, {}) == -1, "select: empty -> none");
    check(select_stash({10, 200}, {1024, 1024}) == -1, "select: all < min -> none");
    check(select_stash({255}, {1 << 20}) == -1, "select: 255 tokens -> none");
    check(select_stash({256}, {1 << 20}) == 0, "select: exactly 256 qualifies");
    check(select_stash({300, 900, 640}, {1, 1, 1}) == 1, "select: longest wins");
    check(select_stash({900, 300}, {1, 1}) == 0, "select: order irrelevant");
    check(select_stash({900, 640}, {1, 1}) == 0, "select: tie-free stability");
    check(select_stash({500, 500}, {1, 1}) == 0, "select: equal length -> first");
    // 1 token per element * 2 bytes: exactly at the cap is stashable.
    const size_t cap_tok = (size_t)(k_stash_max_kv_bytes / sizeof(uint16_t));
    check(select_stash({cap_tok}, {k_stash_max_kv_bytes}) == 0, "select: cap bytes OK");
    check(select_stash({cap_tok}, {k_stash_max_kv_bytes + 1}) == -1, "select: over cap -> none");
    // Over-cap long entry must yield to a shorter stashable one.
    check(select_stash({10000, 400}, {k_stash_max_kv_bytes * 2, 400 * 2}) == 1,
          "select: over-cap longest skipped, shorter taken");
    // Short-but-over-cap entry is skipped by min tokens first.
    check(select_stash({100}, {k_stash_max_kv_bytes * 2}) == -1, "select: short over-cap skipped");

    // keep-the-longer: strictly-longer candidate replaces, shorter/equal keeps.
    check(stash_should_replace(1000, 1001), "keep: longer candidate replaces");
    check(!stash_should_replace(1000, 1000), "keep: equal keeps existing");
    check(!stash_should_replace(1000, 999), "keep: shorter keeps existing");
    check(!stash_should_replace(0, 0), "keep: 0 vs 0 keeps");

    // warm-prefix survival predicate.
    std::vector<int32_t> a{1, 2, 3}, b{1, 2, 3, 4, 5}, c{1, 2, 9, 4}, d{1, 2};
    check(tokens_are_prefix(a, b), "prefix: strict prefix true");
    check(tokens_are_prefix(a, a), "prefix: equal true");
    check(!tokens_are_prefix(c, b), "prefix: divergence false");
    check(!tokens_are_prefix(b, a), "prefix: longer head false");
    check(tokens_are_prefix({}, b), "prefix: empty head true");
}

// ---- mock engine: geometry mirrors Impl's kvBuf layout ---------------------
// 3 full-attention slots (linear 0..ctx) + 2 SWA ring slots (ring wrap).
constexpr int kFull = 3, kSwa = 2, kPerPos = 8, kRing = 32, kCtx = 4096;

struct FakeState {
    int64_t pos = 0;
    std::vector<int32_t> tokens;
    std::vector<float> rs;
    std::vector<uint16_t> swa_kv;
    std::vector<uint16_t> full_kv;  // self-contained stash payload only
};

struct FakeEngine {
    std::vector<uint16_t> kv;  // full: [full][K|V][pos][per]; swa: [swa][K|V][ring][per]
    std::vector<float> rs;
    int64_t pos = 0;
    FakeEngine() : kv(kv_elems()), rs(7, 0.0f) {}
    // uint16 element count of a self-contained [0..n) full-attention copy
    size_t full_elems(int64_t n) const { return (size_t)2 * n * kPerPos * kFull; }
    size_t kv_elems() const {
        return (size_t)2 * kCtx * kPerPos * kFull + (size_t)2 * kRing * kPerPos * kSwa;
    }
    // Deterministic KV payload: every (slot, kind, position, unit) byte pair
    // depends only on the token sequence processed up to that position, so a
    // full re-prefill and a stash restore must agree bit-for-bit.
    void prefill(const std::vector<int32_t> & ids) {
        for (size_t j = 0; j < ids.size(); j++) {
            const int64_t p = pos + (int64_t)j;
            for (int il = 0; il < kFull; il++)
                for (int kv = 0; kv < 2; kv++)
                    for (int u = 0; u < kPerPos; u++)
                        kv_at_full(il, kv, p, u) = mix(ids[j], il, kv, p, u);
            for (int il = 0; il < kSwa; il++)
                for (int kv = 0; kv < 2; kv++)
                    for (int u = 0; u < kPerPos; u++)
                        kv_at_swa(il, kv, p % kRing, u) = mix(ids[j], 100 + il, kv, p, u);
            rs[(size_t)(j % 7)] = (float)(ids[j] ^ (int32_t)(pos + (int64_t)j));
        }
        pos += (int64_t)ids.size();
    }
    void reset() { pos = 0; std::memset(rs.data(), 0, rs.size() * sizeof(float)); }
    uint16_t & kv_at_full(int il, int kind, int64_t p, int u) {
        return kv[(((size_t)il * 2 + kind) * kCtx + (size_t)p) * kPerPos + u];
    }
    uint16_t & kv_at_swa(int il, int kind, int64_t r, int u) {
        const size_t base = (size_t)2 * kCtx * kPerPos * kFull;
        return kv[base + (((size_t)il * 2 + kind) * kRing + (size_t)r) * kPerPos + u];
    }
    static uint16_t mix(int32_t tok, int il, int kv, int64_t p, int u) {
        uint32_t h = (uint32_t)tok * 2654435761u;
        h ^= (uint32_t)(il * 31 + kv * 7 + u) * 2246822519u;
        h ^= (uint32_t)p * 3266489917u;
        h ^= h >> 13;
        return (uint16_t)(h ^ (h >> 16));
    }
    // mirrors save_prefix_state + capture_full_kv
    FakeState save_checkpoint(bool with_full) {
        FakeState st;
        st.pos = pos;
        st.tokens.assign(conv.begin(), conv.begin() + (size_t)pos);
        st.rs = rs;
        for (int il = 0; il < kSwa; il++)
            for (int kv = 0; kv < 2; kv++)
                for (int r = 0; r < kRing; r++)
                    for (int u = 0; u < kPerPos; u++)
                        st.swa_kv.push_back(kv_at_swa(il, kv, r, u));
        if (with_full) capture(st);
        return st;
    }
    // mirrors capture_full_kv (guard: 0 < st.pos <= pos)
    void capture(FakeState & st) {
        st.full_kv.clear();
        if (st.pos <= 0 || st.pos > pos) return;
        for (int il = 0; il < kFull; il++)
            for (int kv = 0; kv < 2; kv++)
                for (int p = 0; p < st.pos; p++)
                    for (int u = 0; u < kPerPos; u++)
                        st.full_kv.push_back(kv_at_full(il, kv, p, u));
    }
    // mirrors restore_prefix_state (validates full_kv geometry)
    bool restore(const FakeState & in) {
        if (in.pos <= 0 || in.pos >= kCtx) return false;
        if ((int64_t)in.tokens.size() != in.pos) return false;
        if (!in.full_kv.empty() && in.full_kv.size() != full_elems(in.pos))
            return false;
        pos = in.pos;
        rs = in.rs;
        size_t c = 0;
        for (int il = 0; il < kSwa; il++)
            for (int kv = 0; kv < 2; kv++)
                for (int r = 0; r < kRing; r++)
                    for (int u = 0; u < kPerPos; u++)
                        kv_at_swa(il, kv, r, u) = in.swa_kv[c++];
        if (!in.full_kv.empty()) {
            c = 0;
            for (int il = 0; il < kFull; il++)
                for (int kv = 0; kv < 2; kv++)
                    for (int p = 0; p < in.pos; p++)
                        for (int u = 0; u < kPerPos; u++)
                            kv_at_full(il, kv, p, u) = in.full_kv[c++];
        }
        return true;
    }
    std::vector<int32_t> conv;  // full processed sequence (for save snapshots)
};

void run_conversation(FakeEngine & eng, std::vector<int32_t> & conv,
                      const std::vector<int32_t> & add, int gen_tokens) {
    conv.insert(conv.end(), add.begin(), add.end());
    eng.conv = conv;
    eng.prefill(add);
    for (int i = 0; i < gen_tokens; i++) {  // decode: one fresh token each step
        const int32_t t = (int32_t)(eng.pos * 3 + 1000);
        conv.push_back(t);
        eng.conv = conv;
        eng.prefill({t});
    }
}

}  // namespace

int main() {
    using namespace m64;
    test_select_stash();

    // ---- mock replay: long conversation -> 2 interrupts -> resume ----------
    // Production worker loop state.
    FakeEngine eng, ref;
    bool has_stash = false;
    FakeState warm, stashed, lru0, lru1;
    warm.pos = 8;
    warm.tokens = {7, 7, 7, 7, 7, 7, 7, 7};
    std::vector<int32_t> conv, ref_conv;

    // Turn 1: long conversation turn (>=256 tokens) + decode.
    std::vector<int32_t> t1(600);
    for (size_t i = 0; i < t1.size(); i++) t1[i] = (int32_t)(300 + i);
    run_conversation(eng, conv, t1, 4);
    run_conversation(ref, ref_conv, t1, 4);
    // save_lru_checkpoint(processed) at end of GEN: two live checkpoints.
    lru0 = eng.save_checkpoint(false);                       // processed
    lru1 = eng.save_checkpoint(false); lru1.pos -= 1;        // prompt-only-ish
    lru1.tokens.resize((size_t)lru1.pos);

    // Turn 2: continue the same conversation (normal LRU hit path).
    run_conversation(eng, conv, {5001, 5002, 5003}, 2);
    run_conversation(ref, ref_conv, {5001, 5002, 5003}, 2);
    lru0 = eng.save_checkpoint(false);
    lru1 = eng.save_checkpoint(false); lru1.pos -= 1;
    lru1.tokens.resize((size_t)lru1.pos);

    // Interrupt 1: unrelated short prompt, prefix_hit == 0.
    {
        std::vector<size_t> ln{lru0.tokens.size(), lru1.tokens.size()};
        std::vector<uint64_t> by{eng.full_elems((int64_t)lru0.pos) * 2,
                                 eng.full_elems((int64_t)lru1.pos) * 2};
        const long sel = select_stash(ln, by);
        check(sel == 0, "interrupt 1: longest checkpoint selected");
        stashed = lru0;
        eng.capture(stashed);
        check(!stashed.full_kv.empty(), "interrupt 1: full KV captured");
        check(stashed.full_kv.size() == eng.full_elems(stashed.pos),
              "interrupt 1: capture covers [0..pos) exactly");
        has_stash = true;
    }
    const std::vector<uint16_t> kv_head_saved(stashed.full_kv);
    eng.reset();
    lru0 = {}; lru1 = {};  // invalidate_after(0)
    eng.prefill({90001, 90002, 90003});  // title gen clobbers kvBuf [0..3)

    // Interrupt 2: another short prompt. Its own checkpoint is <256 tokens,
    // so select_stash finds nothing and the existing stash must survive.
    {
        FakeState small_cp = eng.save_checkpoint(false);  // 3 tokens
        std::vector<size_t> ln{small_cp.tokens.size()};
        std::vector<uint64_t> by{eng.full_elems(small_cp.pos) * 2};
        const long sel = select_stash(ln, by);
        check(sel == -1, "interrupt 2: too-short checkpoint not stashable");
        check(has_stash && stashed.full_kv == kv_head_saved,
              "interrupt 2: existing stash untouched");
    }
    eng.reset();
    eng.prefill({91001, 91002});

    // A hypothetical LONGER candidate replaces (keep-the-longer rule).
    check(stash_should_replace(stashed.tokens.size(), stashed.tokens.size() + 1),
          "replacement rule: longer candidate would replace");
    check(!stash_should_replace(stashed.tokens.size(), stashed.tokens.size() - 1),
          "replacement rule: shorter candidate keeps");

    // Resume: the next real turn finds the stash and restores it whole.
    FakeState resume_cp = stashed;  // what find_best_prefix -> hit would give
    resume_cp.full_kv.pop_back();
    check(!eng.restore(resume_cp), "resume: wrong-size full_kv rejected");
    check(eng.restore(stashed), "resume: stash restored");
    // warm survival predicate: warm head IS a prefix of the conversation.
    warm.tokens = {t1[0], t1[1], t1[2]};
    const bool warm_kept = tokens_are_prefix(warm.tokens, stashed.tokens);
    check(warm_kept, "resume: prefix-aligned warm kept");
    std::vector<int32_t> other_warm = {t1[0], t1[1], 12345};
    check(!tokens_are_prefix(other_warm, stashed.tokens),
          "resume: diverging warm dropped");
    // stash released after restore (main.cpp frees it here).
    has_stash = false; stashed = FakeState{};
    // tail prefill + decoded tokens of the resumed turn (same as reference).
    std::vector<int32_t> tail = {5004, 5005};
    run_conversation(eng, conv, tail, 3);
    run_conversation(ref, ref_conv, tail, 3);

    // Bit-exactness vs the reference engine that never left the conversation.
    check(conv == ref_conv, "resume: token streams identical");
    check(eng.pos == ref.pos, "resume: engine pos identical");
    check(std::memcmp(eng.rs.data(), ref.rs.data(), eng.rs.size() * sizeof(float)) == 0,
          "resume: recurrent state bitwise identical");
    bool kv_identical = true;
    for (int64_t p = 0; p < ref.pos && kv_identical; p++)
        for (int il = 0; il < kFull && kv_identical; il++)
            for (int kv = 0; kv < 2 && kv_identical; kv++)
                for (int u = 0; u < kPerPos && kv_identical; u++)
                    kv_identical = eng.kv_at_full(il, kv, p, u) ==
                                   ref.kv_at_full(il, kv, p, u);
    check(kv_identical, "resume: full-attention KV bitwise identical to full re-prefill");

    if (g_fail) {
        std::printf("test_prefix_stash: %d FAILURES\n", g_fail);
        return 1;
    }
    std::printf("test_prefix_stash: all checks passed\n");
    return 0;
}
