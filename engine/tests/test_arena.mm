// test_arena.mm — Phase-2B unit test: LockedRegion / PreadPool / ExpertArena.
// Builds a deterministic 16 MiB pseudo pack in /tmp (2 layers x 8 experts x
// 1 MiB, 16 KiB aligned), then exercises:
//   1) LockedRegion  — 16 KiB alignment + mlock pin
//   2) PreadPool     — 4 non-contiguous 1 MiB F_NOCACHE batch reads, bitwise
//   3) ExpertArena   — LRU top-8 resolution over 10 slots / 16 experts:
//                      slot distinctness (pin guarantee), bitwise content
//                      parity for every resolved expert, telemetry invariants
//   4) MTLIO backend — mtlio_active() init report + bitwise streamed parity
// Peak RAM < 32 MB (one 10 MiB arena alive at a time + 1 MiB scratch).
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "expert_arena.hpp"

using namespace m64;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static constexpr int      kNLAYERS = 2;
static constexpr int      kNEXP    = 8;
static constexpr size_t   kSTRIDE  = 1u << 20;               // 1 MiB (16 KiB aligned)
// Fixture path: $TMPDIR (fallback /tmp) + pid so parallel runs never collide.
// The file is unlinked (via ::remove) on every exit path below.
static const char *arena_tmp_path() {
    static char path[1024];
    const char *tmp = std::getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = "/tmp";
    std::snprintf(path, sizeof(path), "%s/m64_arena_unit_%d.bin", tmp, static_cast<int>(getpid()));
    return path;
}
static const char *       kFile    = arena_tmp_path();

// splitmix64 finalizer: stateless deterministic word stream, regenerated
// on demand so we never hold more than one expert blob in RAM.
static uint64_t smix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline uint64_t expert_word(uint32_t expert, uint64_t w) {
    return smix(((uint64_t)expert << 40) | w);
}
static void expert_fill(uint32_t expert, uint8_t * dst) {
    for (uint64_t w = 0; w < kSTRIDE / 8; w++) {
        uint64_t v = expert_word(expert, w);
        memcpy(dst + w * 8, &v, 8);
    }
}
// Bitwise verify one expert-sized block (host ground truth regenerated into scratch).
static bool expert_ok(const uint8_t * got, uint32_t expert, uint8_t * scratch) {
    expert_fill(expert, scratch);
    return memcmp(got, scratch, kSTRIDE) == 0;
}

// ---- fixture: layers are contiguous, expert e of layer l lives at
// (l * kNEXP + e) * kSTRIDE, i.e. global expert id * 1 MiB.
static bool write_pack(std::string & err) {
    int fd = ::open(kFile, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { err = std::string("open ") + kFile + ": " + strerror(errno); return false; }
    std::vector<uint8_t> buf(kSTRIDE);
    for (uint32_t g = 0; g < (uint32_t)(kNLAYERS * kNEXP); g++) {
        expert_fill(g, buf.data());
        ssize_t n = ::pwrite(fd, buf.data(), kSTRIDE, (off_t)g * kSTRIDE);
        if (n != (ssize_t)kSTRIDE) { err = "pwrite"; ::close(fd); return false; }
    }
    fsync(fd);
    ::close(fd);
    return true;
}

int main() {
    printf("== Phase-2B arena unit test (%d layers x %d experts x 1 MiB) ==\n",
           kNLAYERS, kNEXP);

    // ---------- 1) LockedRegion: alignment + mlock ----------
    {
        LockedRegion lr;
        std::string err;
        CHECK(lr.alloc(2u << 20, err), "LockedRegion alloc: %s", err.c_str());
        CHECK(lr.ptr() != nullptr, "LockedRegion null ptr");
        CHECK((reinterpret_cast<uintptr_t>(lr.ptr()) & 16383u) == 0,
              "LockedRegion not 16 KiB aligned");
        CHECK(lr.bytes() >= (2u << 20), "LockedRegion bytes too small");
        CHECK(lr.mlocked(), "LockedRegion not mlocked");
        // touch every page (mlock must keep it resident)
        memset(lr.ptr(), 0x5A, lr.bytes());
        printf("locked_region: 2 MiB aligned=%d mlocked=%d\n",
               (int)((reinterpret_cast<uintptr_t>(lr.ptr()) & 16383u) == 0),
               (int)lr.mlocked());
        lr.release();
    }

    // ---------- fixture ----------
    std::string err;
    if (!write_pack(err)) { printf("FAIL: pack: %s\n", err.c_str()); ::remove(kFile); return 1; }

    // ---------- 2) PreadPool: F_NOCACHE parallel batch reads ----------
    {
        PreadPool pool;
        if (!pool.open(kFile, 4, err)) { printf("FAIL: pool open: %s\n", err.c_str()); ::remove(kFile); return 1; }
        CHECK(pool.ok(), "pool not ok after open");
        const uint32_t ids[4] = {1, 6, 9, 14};  // non-contiguous experts
        std::vector<uint8_t> dst(4 * kSTRIDE);
        std::vector<PreadPool::Job> jobs;
        for (int i = 0; i < 4; i++)
            jobs.push_back({dst.data() + (size_t)i * kSTRIDE, (uint64_t)ids[i] * kSTRIDE});
        uint64_t ns = 0;
        CHECK(pool.read_batch(jobs, kSTRIDE, &ns, err), "read_batch: %s", err.c_str());
        std::vector<uint8_t> scratch(kSTRIDE);
        for (int i = 0; i < 4; i++)
            CHECK(expert_ok(dst.data() + (size_t)i * kSTRIDE, ids[i], scratch.data()),
                  "pread bitwise mismatch for expert %u", ids[i]);
        double gbs = ns > 0 ? (4.0 * kSTRIDE / 1e9) / (ns / 1e9) : 0.0;
        printf("pread_pool: 4x1 MiB F_NOCACHE batch parity OK (%.1f ms, %.2f GB/s)\n",
               ns / 1e6, gbs);
        pool.close();
    }

    ExpertArenaCfg base_cfg;
    for (int l = 0; l < kNLAYERS; l++)
        base_cfg.layer_base.push_back((uint64_t)l * kNEXP * kSTRIDE);
    base_cfg.exp_stride = kSTRIDE;
    base_cfg.n_experts  = kNEXP;
    base_cfg.num_slots  = 10;   // tight: 10 slots across 2*8 = 16 experts
    base_cfg.nworkers   = 4;

    // ---------- 3) ExpertArena (pread): LRU + pin guarantee + parity ----------
    {
        ExpertArenaCfg cfg = base_cfg;
        cfg.pack_path = kFile;
        cfg.io_mode   = "pread";
        ExpertArena arena;
        CHECK(arena.create(cfg, err), "arena create: %s", err.c_str());
        CHECK(arena.mlocked(), "arena region not mlocked");
        CHECK(!arena.mtlio_active(), "pread mode must not activate mtlio");

        auto resolve = [&](int layer, uint32_t slot_out[8]) -> bool {
            uint32_t sel[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            return arena.resolve_top8(layer, sel, slot_out, err);
        };
        std::vector<uint8_t> scratch(kSTRIDE);
        const uint8_t * abase = (const uint8_t *)arena.base();

        // A: fill (all miss). B: full-capacity cold call -> pin guarantee:
        // all 8 slot_out must be distinct (an expert picked earlier in the
        // same call is never evicted by a later miss).
        uint32_t sa[8], sb[8], sc[8], sd[8];
        auto abort_if = [&](const char * what) -> bool {
            if (!g_fail) return false;
            ::remove(kFile);
            printf("test_arena: abort (%s): %s\n", what, err.c_str());
            return true;
        };
        CHECK(resolve(0, sa), "resolve A: %s", err.c_str());
        CHECK(resolve(1, sb), "resolve B: %s", err.c_str());
        if (abort_if("resolve")) return 1;
        for (int i = 0; i < 8; i++) {
            CHECK(sb[i] < arena.num_slots(), "B slot %u out of range", sb[i]);
            for (int j = 0; j < i; j++)
                CHECK(sb[i] != sb[j], "B slots collide (%u==%u) — pin broken", sb[i], sb[j]);
            CHECK(expert_ok(abase + (size_t)sb[i] * kSTRIDE, (uint32_t)(kNEXP + i), scratch.data()),
                  "B content mismatch expert L1E%d slot %u", i, sb[i]);
        }
        // C: 2 hits (L0E6/L0E7 survived B) + 6 streamed misses.
        CHECK(resolve(0, sc), "resolve C: %s", err.c_str());
        for (int i = 0; i < 8; i++)
            CHECK(expert_ok(abase + (size_t)sc[i] * kSTRIDE, (uint32_t)i, scratch.data()),
                  "C content mismatch expert L0E%d slot %u", i, sc[i]);
        // D: warm repeat (10 slots >= 8 hot experts) -> full hit set.
        CHECK(resolve(0, sd), "resolve D: %s", err.c_str());
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < i; j++)
                CHECK(sd[i] != sd[j], "D slots collide — pin broken");

        const ArenaTelemetry & t = arena.telemetry();
        printf("arena_pread: slots=%zu hits=%llu misses=%llu evictions=%llu ssd=%llu B\n",
               t.arena_slots, (unsigned long long)t.hits, (unsigned long long)t.misses,
               (unsigned long long)t.evictions, (unsigned long long)t.ssd_read_bytes);
        CHECK(t.hits > 0, "expected hits > 0");
        CHECK(t.misses > 0, "expected misses > 0");
        CHECK(t.evictions > 0, "expected evictions > 0");
        CHECK(t.ssd_read_bytes == t.misses * kSTRIDE,
              "ssd_read_bytes (%llu) != misses*stride (%llu)",
              (unsigned long long)t.ssd_read_bytes,
              (unsigned long long)(t.misses * kSTRIDE));
        CHECK(t.arena_bytes >= 10 * kSTRIDE, "arena_bytes too small: %zu", t.arena_bytes);
    }  // arena (and its mlock'd 10 MiB) released before the mtlio arena

    // ---------- 4) MTLIO backend: init report + streamed parity ----------
    {
        ExpertArenaCfg cfg = base_cfg;
        cfg.pack_path = kFile;
        cfg.io_mode   = "mtlio";
        ExpertArena arena;
        CHECK(arena.create(cfg, err), "mtlio arena create: %s", err.c_str());
        const bool mtlio = arena.mtlio_active();
        printf("mtlio: initialized=%s on this device\n", mtlio ? "CLEAN" : "NO (fell back to pread)");
        CHECK(mtlio, "MTLIOCommandQueue did not initialize cleanly on M4 Max");
        CHECK(arena.telemetry().mtlio == mtlio, "telemetry.mtlio disagrees with mtlio_active()");
        CHECK(arena.mtl_buffer() != nullptr, "mtlio arena exposes no MTLBuffer");

        uint32_t sel[8] = {2, 3, 4, 5, 6, 7, 0, 1};
        uint32_t so[8];
        CHECK(arena.resolve_top8(0, sel, so, err), "mtlio resolve: %s", err.c_str());
        std::vector<uint8_t> scratch(kSTRIDE);
        const uint8_t * abase = (const uint8_t *)arena.base();
        for (int i = 0; i < 8; i++) {
            CHECK(so[i] < arena.num_slots(), "mtlio slot %u out of range", so[i]);
            for (int j = 0; j < i; j++)
                CHECK(so[i] != so[j], "mtlio slots collide (%u==%u)", so[i], so[j]);
            CHECK(expert_ok(abase + (size_t)so[i] * kSTRIDE, sel[i], scratch.data()),
                  "mtlio content mismatch expert L0E%d slot %u", sel[i], so[i]);
        }
        const ArenaTelemetry & t = arena.telemetry();
        CHECK(t.misses == 8 && t.ssd_read_bytes == 8 * kSTRIDE,
              "mtlio first-call telemetry wrong: misses=%llu ssd=%llu",
              (unsigned long long)t.misses, (unsigned long long)t.ssd_read_bytes);
        printf("mtlio: 8x1 MiB DMA-streamed into mlock'd arena, bitwise parity OK "
               "(%.1f ms, %.2f GB/s)\n",
               t.ssd_ns / 1e6,
               t.ssd_ns > 0 ? (t.ssd_read_bytes / 1e9) / (t.ssd_ns / 1e9) : 0.0);
    }

    ::remove(kFile);  // clean up the 16 MiB fixture
    printf(g_fail ? "test_arena: %d FAILURES\n" : "test_arena: ALL CHECKS PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
