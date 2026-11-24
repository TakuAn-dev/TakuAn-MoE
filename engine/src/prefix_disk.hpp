// prefix_disk.hpp — SSD-persistent prefix cache (.m64kv) for the
// bitwise-exact RAM PrefixState. One self-contained file per warm prefix:
//
//   DiskPrefixHeader (48 B, packed layout: magic/version/counts + fingerprints)
//   tokens   : n_tokens  x int32_t   (exact token sequence, len == pos)
//   rs       : rs_floats x float     (ShortConv recurrent state, o.rs)
//   swa_kv   : swa_u16   x uint16_t  (FP16 K+V, all 40 SWA ring slots)
//   full_kv  : full_u16  x uint16_t  (per full-attention slot i:
//              K[0..pos*kKVD) then V[0..pos*kKVD) — ONLY 0..pos-1, never the
//              unused tail up to ctx)
//   trailer  : uint64_t xxh64 digest of every preceding byte
//
// Any wrong-pack / truncated / bit-flipped file fails the header or trailer
// check and the caller cleanly falls back to computing the prefill.
// Pure C++20 stdlib + xxh64.hpp: zero Metal / zero external dependencies.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "xxh64.hpp"

namespace m64 {

struct DiskPrefixHeader {
    char magic[4];           // "M64K"
    uint32_t version;        // 1
    uint32_t n_tokens;       // == pos
    uint32_t n_swa_ring;     // SWA ring capacity per slot (e.g. 1025) -> invalidates
                             // a load whose kvBuf ring geometry differs
    uint64_t pack_fp;        // pack fingerprint (invalidates on pack/hparam change)
    uint64_t rs_floats;      // expected == kNSLOT * kCH * kHR
    uint64_t swa_u16;        // expected == swa_kv_elems()
    uint64_t full_u16;       // expected == n_full_slots * n_tokens * kKVD * 2
};
static_assert(sizeof(DiskPrefixHeader) == 48, "DiskPrefixHeader must be 48 bytes");

// Cache key: pack fingerprint + exact token ids. Same pack + same prefix
// tokens -> same key; any difference -> different file.
inline uint64_t prefix_cache_key(uint64_t pack_fp, const int32_t * tokens,
                                 size_t n_tokens) {
    Xxh64 h;
    h.update(&pack_fp, sizeof(pack_fp));
    if (n_tokens) h.update(tokens, n_tokens * sizeof(int32_t));
    return h.digest();
}

// <cache_dir>/prefix_<16-hex-key>.m64kv (pack_fp is inside the key and
// re-verified on load, so a key collision from another pack still rejects).
inline std::string prefix_cache_path(const std::string & cache_dir, uint64_t pack_fp,
                                     const std::vector<int32_t> & tokens) {
    const uint64_t key = prefix_cache_key(pack_fp, tokens.data(), tokens.size());
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long)key);
    std::filesystem::path p(cache_dir);
    p /= std::string("prefix_") + hex + ".m64kv";
    return p.string();
}

// Streams header + payload into <path>.tmp while hashing every byte, appends
// the 8-byte trailer digest, then atomically renames into place. Any I/O
// failure removes the .tmp and fills `err`.
inline bool write_disk_prefix_file(const std::string & path, const DiskPrefixHeader & hdr,
                                   const int32_t * tokens, const float * rs,
                                   const uint16_t * swa_kv, const uint16_t * full_kv,
                                   std::string & err) {
    auto fail = [&](const std::string & m) { err = "prefix disk: " + m; return false; };
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path() && !p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) return fail("mkdir " + p.parent_path().string() + ": " + ec.message());
    }
    const std::string tmp = path + ".tmp";
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) return fail("cannot open " + tmp + " for write");
    Xxh64 h;
    bool ok = true;
    auto put = [&](const void * d, size_t n) {
        if (!n) return;
        if (std::fwrite(d, 1, n, f) != n) { ok = false; return; }
        h.update(d, n);
    };
    put(&hdr, sizeof(hdr));
    if (hdr.n_tokens) put(tokens, (size_t)hdr.n_tokens * sizeof(int32_t));
    if (ok && hdr.rs_floats) put(rs, (size_t)hdr.rs_floats * sizeof(float));
    if (ok && hdr.swa_u16) put(swa_kv, (size_t)hdr.swa_u16 * sizeof(uint16_t));
    if (ok && hdr.full_u16) put(full_kv, (size_t)hdr.full_u16 * sizeof(uint16_t));
    const uint64_t digest = h.digest();  // covers everything above, not itself
    if (ok) ok = std::fwrite(&digest, 1, sizeof(digest), f) == sizeof(digest);
    if (ok && std::fflush(f) != 0) ok = false;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) {
        std::filesystem::remove(tmp, ec);
        return fail("write error on " + tmp);
    }
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return fail("rename " + tmp + " -> " + path + ": " + ec.message());
    }
    return true;
}

// Reads + fully validates a .m64kv against the LIVE engine geometry before
// returning true: magic/version, pack_fp, 0 < n_tokens < max_ctx, SWA ring
// capacity, rs/swa element counts, and the trailing xxh64 of every payload
// byte. Corrupted, truncated or wrong-pack files return false with `err`.
inline bool read_disk_prefix_file(const std::string & path, uint64_t expect_pack_fp,
                                  uint32_t max_ctx, uint32_t expect_swa_ring,
                                  uint64_t expect_rs_floats, uint64_t expect_swa_u16,
                                  uint32_t expect_n_full_kv_per_pos_u16,
                                  std::vector<int32_t> & out_tokens,
                                  std::vector<float> & out_rs,
                                  std::vector<uint16_t> & out_swa_kv,
                                  std::vector<uint16_t> & out_full_kv,
                                  std::string & err) {
    auto fail = [&](const std::string & m) { err = "prefix disk: " + m + " (" + path + ")"; return false; };
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open");
    bool ok = true;
    auto get = [&](void * d, size_t n) {
        if (!n) return;
        if (std::fread(d, 1, n, f) != n) ok = false;
    };
    auto get_hashed = [&](void * d, size_t n, Xxh64 & h) {
        if (!n) return;
        if (ok) {
            if (std::fread(d, 1, n, f) != n) { ok = false; return; }
            h.update(d, n);
        }
    };
    DiskPrefixHeader hdr{};
    get(&hdr, sizeof(hdr));
    if (!ok) { std::fclose(f); return fail("truncated header"); }
    Xxh64 h;
    h.update(&hdr, sizeof(hdr));
    if (std::memcmp(hdr.magic, "M64K", 4) != 0) { std::fclose(f); return fail("bad magic"); }
    if (hdr.version != 1) { std::fclose(f); return fail("unsupported version " + std::to_string(hdr.version)); }
    if (hdr.pack_fp != expect_pack_fp) { std::fclose(f); return fail("pack fingerprint mismatch"); }
    if (hdr.n_tokens == 0 || hdr.n_tokens >= max_ctx) {
        std::fclose(f); return fail("n_tokens " + std::to_string(hdr.n_tokens) + " outside (0," +
                                    std::to_string(max_ctx) + ")");
    }
    if (hdr.n_swa_ring != expect_swa_ring) {
        std::fclose(f); return fail("swa ring " + std::to_string(hdr.n_swa_ring) +
                                    " != " + std::to_string(expect_swa_ring));
    }
    if (hdr.rs_floats != expect_rs_floats) { std::fclose(f); return fail("rs size mismatch"); }
    if (hdr.swa_u16 != expect_swa_u16) { std::fclose(f); return fail("swa_kv size mismatch"); }
    if (hdr.full_u16 != (uint64_t)hdr.n_tokens * (uint64_t)expect_n_full_kv_per_pos_u16) {
        std::fclose(f); return fail("full_kv size mismatch");
    }
    out_tokens.assign(hdr.n_tokens, 0);
    out_rs.assign((size_t)hdr.rs_floats, 0.0f);
    out_swa_kv.assign((size_t)hdr.swa_u16, 0);
    out_full_kv.assign((size_t)hdr.full_u16, 0);
    get_hashed(out_tokens.data(), out_tokens.size() * sizeof(int32_t), h);
    get_hashed(out_rs.data(), out_rs.size() * sizeof(float), h);
    get_hashed(out_swa_kv.data(), out_swa_kv.size() * sizeof(uint16_t), h);
    get_hashed(out_full_kv.data(), out_full_kv.size() * sizeof(uint16_t), h);
    uint64_t trailer = 0;
    if (ok) get(&trailer, sizeof(trailer));
    const uint64_t want = h.digest();
    std::fclose(f);
    if (!ok) return fail("truncated payload");
    if (trailer != want) return fail("xxh64 trailer mismatch (corrupted file)");
    return true;
}

}  // namespace m64
