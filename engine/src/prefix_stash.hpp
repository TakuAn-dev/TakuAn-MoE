// prefix_stash.hpp — policy for the worker's self-contained RAM stash
// (main.cpp --worker loop). A short interrupting request (title gen, memory
// op) arrives with prefix_hit == 0, so the worker resets the engine and
// invalidate_after(0) wipes every in-place LRU checkpoint; the next long
// conversation turn then re-prefills everything but the SSD-warmed head.
// Policy here decides WHICH checkpoint is worth copying its full-attention
// KV out of kvBuf for (PrefixState::full_kv), when an existing stash must
// NOT be replaced by a shorter one, and whether a pinned warm prefix
// survives a stash restore. Pure C++ so engine/tests can pin the decisions.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace m64 {

// Checkpoints shorter than this re-prefill fast enough to not be worth the
// capture copy + the resident RAM.
inline constexpr size_t k_stash_min_tokens = 256;

// Hard ceiling for one stash's full-attention KV footprint (bytes).
inline constexpr uint64_t k_stash_max_kv_bytes = 2ull << 30;  // 2 GiB

// Index of the longest stashable checkpoint among parallel arrays `lens`
// (token counts) / `bytes` (that checkpoint's full_kv size in bytes):
// lens[i] >= k_stash_min_tokens AND bytes[i] <= k_stash_max_kv_bytes.
// Ties go to the earlier index. -1 when nothing qualifies.
inline long select_stash(const std::vector<size_t> & lens,
                         const std::vector<uint64_t> & bytes) {
    long best = -1;
    for (size_t i = 0; i < lens.size() && i < bytes.size(); i++) {
        if (lens[i] < k_stash_min_tokens) continue;
        if (bytes[i] > k_stash_max_kv_bytes) continue;
        if (best < 0 || lens[i] > lens[(size_t)best]) best = (long)i;
    }
    return best;
}

// Two interrupts in a row: the first stash already holds the long
// conversation's KV; the second interrupt's own (shorter) checkpoints must
// NOT clobber it. Replace strictly only when the candidate is LONGER.
inline bool stash_should_replace(size_t stashed_tokens, size_t candidate_tokens) {
    return candidate_tokens > stashed_tokens;
}

// Element-wise "head is a prefix of whole" (equal counts as a prefix).
// A pinned warm prefix only survives a stash restore — which rewrites
// kvBuf [0, stash_pos) — when its tokens are a prefix of the stash.
inline bool tokens_are_prefix(const std::vector<int32_t> & head,
                              const std::vector<int32_t> & whole) {
    return head.size() <= whole.size() &&
           std::equal(head.begin(), head.end(), whole.begin());
}

}  // namespace m64
