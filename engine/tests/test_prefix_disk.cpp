// test_prefix_disk.cpp — unit tests for the SSD persistent prefix cache
// (.m64kv) format in engine/src/prefix_disk.hpp. Zero Metal dependency:
// pure C++20 stdlib + xxh64. Checks bit-exact round-trip of every array,
// and that wrong pack_fp / wrong ring / n_tokens >= max_ctx / 1-byte bit
// flips / truncation all reject cleanly.
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#include "prefix_disk.hpp"

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

// Deterministic pseudo-random payloads (bit patterns, not "nice" values) so
// any silent rescaling or reordering in write/read shows up as a bit diff.
std::vector<int32_t> make_tokens(size_t n) {
    std::vector<int32_t> t(n);
    for (size_t i = 0; i < n; i++) t[i] = (int32_t)(i * 1000003u) - 500;
    return t;
}
std::vector<float> make_rs(size_t n) {
    std::vector<float> r(n);
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)i * 2654435761u;
        std::memcpy(&r[i], &bits, sizeof(float));
    }
    return r;
}
std::vector<uint16_t> make_u16(size_t n, uint32_t mul, uint16_t add) {
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint16_t)(i * mul + add);
    return v;
}

bool bits_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
    using namespace m64;
    static_assert(sizeof(DiskPrefixHeader) == 48, "header size 48");

    // Synthetic geometry: 7 tokens, 12 rs floats, one ring of 5 slots' worth
    // of SWA u16 (40 elems), 6 full-attention u16 per token position.
    const uint32_t n_tokens = 7, swa_ring = 5, per_pos_u16 = 6;
    const uint64_t rs_floats = 12, swa_u16 = 40, pack_fp = 0x0123456789abcdefULL;
    const auto tokens = make_tokens(n_tokens);
    const auto rs = make_rs((size_t)rs_floats);
    const auto swa = make_u16((size_t)swa_u16, 7919u, 1);
    const auto full = make_u16((size_t)n_tokens * per_pos_u16, 104729u, 7);

    const auto root = std::filesystem::temp_directory_path() /
                      ("m64_prefix_disk_test_" + std::to_string((long)getpid()));
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    const std::string path = (root / "nested" / "prefix_store" / "p0.m64kv").string();

    DiskPrefixHeader hdr{};
    std::memcpy(hdr.magic, "M64K", 4);
    hdr.version = 1;
    hdr.n_tokens = n_tokens;
    hdr.n_swa_ring = swa_ring;
    hdr.pack_fp = pack_fp;
    hdr.rs_floats = rs_floats;
    hdr.swa_u16 = swa_u16;
    hdr.full_u16 = (uint64_t)n_tokens * per_pos_u16;

    std::string err;
    check(write_disk_prefix_file(path, hdr, tokens.data(), rs.data(),
                                 swa.data(), full.data(), err),
          "write_disk_prefix_file (creates nested parent dirs)");

    // ---- bit-exact round trip ----
    std::vector<int32_t> rt_tok;
    std::vector<float> rt_rs;
    std::vector<uint16_t> rt_swa, rt_full;
    bool ok = read_disk_prefix_file(path, pack_fp, 4096, swa_ring, rs_floats,
                                    swa_u16, per_pos_u16, rt_tok, rt_rs,
                                    rt_swa, rt_full, err);
    check(ok, "read_disk_prefix_file accepts a valid file");
    check(rt_tok == tokens, "tokens round-trip bit-for-bit");
    check(bits_equal(rt_rs, rs), "rs round-trip bit-for-bit");
    check(rt_swa == swa, "swa_kv round-trip bit-for-bit");
    check(rt_full == full, "full_kv round-trip bit-for-bit");

    // ---- header validation rejects ----
    ok = read_disk_prefix_file(path, pack_fp ^ 1, 4096, swa_ring, rs_floats,
                               swa_u16, per_pos_u16, rt_tok, rt_rs, rt_swa,
                               rt_full, err);
    check(!ok, "wrong pack_fp rejected");
    ok = read_disk_prefix_file(path, pack_fp, 4096, swa_ring + 1, rs_floats,
                               swa_u16, per_pos_u16, rt_tok, rt_rs, rt_swa,
                               rt_full, err);
    check(!ok, "wrong swa_ring rejected");
    ok = read_disk_prefix_file(path, pack_fp, n_tokens, swa_ring, rs_floats,
                               swa_u16, per_pos_u16, rt_tok, rt_rs, rt_swa,
                               rt_full, err);
    check(!ok, "n_tokens >= max_ctx rejected");
    ok = read_disk_prefix_file(path, pack_fp, 4096, swa_ring, rs_floats + 1,
                               swa_u16, per_pos_u16, rt_tok, rt_rs, rt_swa,
                               rt_full, err);
    check(!ok, "wrong rs size rejected");
    ok = read_disk_prefix_file(path, pack_fp, 4096, swa_ring, rs_floats,
                               swa_u16 + 2, per_pos_u16, rt_tok, rt_rs,
                               rt_swa, rt_full, err);
    check(!ok, "wrong swa size rejected");
    ok = read_disk_prefix_file(path, pack_fp, 4096, swa_ring, rs_floats,
                               swa_u16, per_pos_u16 + 1, rt_tok, rt_rs,
                               rt_swa, rt_full, err);
    check(!ok, "wrong full_kv per-pos stride rejected");

    // ---- integrity: 1-byte payload bit flip must fail the xxh64 trailer ----
    {
        std::FILE * f = std::fopen(path.c_str(), "r+b");
        if (!f) {
            check(false, "reopen for bit flip");
        } else {
            std::fseek(f, (long)(sizeof(DiskPrefixHeader) + 2 * sizeof(int32_t)),
                       SEEK_SET);  // 3rd token
            int c = std::fgetc(f);
            std::fseek(f, -1, SEEK_CUR);
            std::fputc(c ^ 0x80, f);
            std::fclose(f);
        }
        ok = read_disk_prefix_file(path, pack_fp, 4096, swa_ring, rs_floats,
                                   swa_u16, per_pos_u16, rt_tok, rt_rs, rt_swa,
                                   rt_full, err);
        check(!ok, "1-byte payload bit flip rejected (xxh64 trailer)");
    }

    // ---- integrity: truncated file rejects ----
    {
        const auto sz = (uint64_t)std::filesystem::file_size(path, ec);
        std::filesystem::resize_file(path, sz - 16, ec);
        ok = read_disk_prefix_file(path, pack_fp, 4096, swa_ring, rs_floats,
                                   swa_u16, per_pos_u16, rt_tok, rt_rs, rt_swa,
                                   rt_full, err);
        check(!ok, "truncated file rejected");
    }

    // ---- nonexistent file rejects ----
    ok = read_disk_prefix_file((root / "nope.m64kv").string(), pack_fp, 4096,
                               swa_ring, rs_floats, swa_u16, per_pos_u16,
                               rt_tok, rt_rs, rt_swa, rt_full, err);
    check(!ok, "missing file rejected");

    // ---- cache key / path ----
    const uint64_t k1 = prefix_cache_key(pack_fp, tokens.data(), tokens.size());
    const uint64_t k2 = prefix_cache_key(pack_fp, tokens.data(), tokens.size());
    const uint64_t k3 = prefix_cache_key(pack_fp ^ 7, tokens.data(), tokens.size());
    const auto tokens2 = make_tokens(n_tokens - 1);
    const uint64_t k4 = prefix_cache_key(pack_fp, tokens2.data(), tokens2.size());
    check(k1 == k2, "prefix_cache_key deterministic");
    check(k1 != k3 && k1 != k4, "prefix_cache_key sensitive to pack_fp and tokens");
    const std::string p1 = prefix_cache_path(root.string(), pack_fp, tokens);
    const std::string p2 = prefix_cache_path(root.string(), pack_fp, tokens);
    const std::string p3 = prefix_cache_path(root.string(), pack_fp, tokens2);
    check(p1 == p2 && p1 != p3, "prefix_cache_path stable + token-sensitive");
    const std::filesystem::path pp(p1);
    const std::string name = pp.filename().string();
    bool shaped = name.size() == 7 + 16 + 6 && name.rfind("prefix_", 0) == 0 &&
                  name.size() > 6 && name.compare(name.size() - 6, 6, ".m64kv") == 0;
    for (size_t i = 7; i < 7 + 16 && shaped; i++)
        shaped = std::isxdigit((unsigned char)name[i]) != 0;
    check(shaped, "path is <cache_dir>/prefix_<16-hex>.m64kv");

    std::filesystem::remove_all(root, ec);
    if (g_fail) {
        std::printf("test_prefix_disk: %d FAILURES\n", g_fail);
        return 1;
    }
    std::printf("test_prefix_disk: all checks passed\n");
    return 0;
}
