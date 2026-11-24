// test_dequant.cpp — unit parity: pack_reader + dequant reference vs golden.bin.
// Also checks XXH64 (C++) == xxh64 field values in the pack header.
// Usage: test_dequant <tiny.m64pack> <golden.bin>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "dequant.hpp"
#include "pack_reader.hpp"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static std::vector<uint8_t> slurp(const char *p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <pack> <golden.bin>\n", argv[0]); return 2; }
    m64::PackReader pack;
    try {
        pack.open(argv[1], /*verify_hashes=*/true);
        printf("pack opened: %zu tensors, arch=%s, hashes verified\n", pack.n_tensors(), pack.header().arch.c_str());
    } catch (const std::exception &e) {
        printf("FAIL: pack open: %s\n", e.what());
        return 1;
    }
    CHECK(pack.header().version == 1, "version");
    CHECK(pack.header().blob_start % 65536 == 0, "blob_start=%lld", (long long)pack.header().blob_start);

    // --- golden.bin self-contained record stream ---
    auto gb = slurp(argv[2]);
    CHECK(gb.size() > 8 && memcmp(gb.data(), "M64G", 4) == 0, "golden magic");
    size_t p = 4;
    uint32_t count; memcpy(&count, gb.data() + p, 4); p += 4;
    int checked = 0;
    for (uint32_t ri = 0; ri < count; ri++) {
        uint16_t nlen; memcpy(&nlen, gb.data() + p, 2); p += 2;
        std::string name((char *)gb.data() + p, nlen); p += nlen;
        uint32_t dt, nblk; memcpy(&dt, gb.data() + p, 4); memcpy(&nblk, gb.data() + p + 4, 4); p += 8;
        uint64_t rlen; memcpy(&rlen, gb.data() + p, 8); p += 8;
        const uint8_t *raw = gb.data() + p; p += rlen;
        uint64_t nvals; memcpy(&nvals, gb.data() + p, 8); p += 8;
        const float *stored = (const float *)(gb.data() + p); p += nvals * 4;

        // Independent dequant via the C++ reference dispatcher.
        m64::DType D = (m64::DType)0;
        switch (dt) {
            case 0: D = m64::DType::F32; break;
            case 1: D = m64::DType::F16; break;
            case 6: D = m64::DType::Q5_0; break;
            case 7: D = m64::DType::Q5_1; break;
            case 8: D = m64::DType::Q8_0; break;
            case 9: D = m64::DType::Q8_1; break;
            case 12: D = m64::DType::Q4_K; break;
            case 30: D = m64::DType::BF16; break;
            default: CHECK(false, "golden dtype %u", dt); continue;
        }
        std::vector<float> out(nvals);
        m64::dequant_row(D, raw, out.data(), (int64_t)nvals);
        int bad = 0;
        float maxdiff = 0;
        for (uint64_t i = 0; i < nvals; i++) {
            if (memcmp(&out[i], &stored[i], 4) != 0) bad++;
            float d = std::abs(out[i] - stored[i]);
            if (d > maxdiff) maxdiff = d;
        }
        CHECK(bad == 0, "golden '%s': %d/%llu bitwise mismatches (maxdiff %.3e)",
              name.c_str(), bad, (unsigned long long)nvals, maxdiff);
        if (!bad) printf("  ok  %-35s %llu vals bitwise\n", name.c_str(), (unsigned long long)nvals);
        checked++;
    }
    CHECK(checked == (int)count, "records checked %d != %u", checked, count);

    // --- dequant on real pack tensors: row lengths must be block-multiple ---
    for (size_t i = 0; i < pack.n_tensors(); i++) {
        const auto &t = pack.tensor(i);
        if (t.is_expert) continue;  // fused layout: validated via parts below
        int64_t n = m64::numel(t);
        int be = m64::dtype_block_elems(t.dtype);
        CHECK(be && n % be == 0, "tensor %s n=%lld not multiple of block %d", t.name.c_str(), (long long)n, be);
        // dequant every tensor once (exercises dispatch; tiny sizes)
        std::vector<float> out(n);
        try {
            m64::dequant_row(t.dtype, pack.ptr(t), out.data(), n);
        } catch (const std::exception &e) {
            CHECK(false, "dequant %s: %s", t.name.c_str(), e.what());
        }
    }
    printf("tested %zu pack tensors\n", pack.n_tensors());

    // --- expert layout sanity: gate|up|down parts fill stride region ---
    for (size_t i = 0; i < pack.n_tensors(); i++) {
        const auto &t = pack.tensor(i);
        if (!t.is_expert) continue;
        // per-part dtype: manifests record part_dtypes/part_bytes for mixed-dtype
        // experts (e.g. q4_k gate|up + q5_0/q5_1 down). Legacy packs omit both
        // fields and fall back to the tensor-level dtype.
        auto bytes = [&](int part) {
            const std::vector<int64_t> &sh = part == 0 ? t.part_gate : part == 1 ? t.part_up : t.part_down;
            int64_t n = 1; for (auto d : sh) n *= d;
            m64::DType pd = t.part_dtype(t.dtype, part);
            int be = m64::dtype_block_elems(pd), ts = m64::dtype_type_size(pd);
            CHECK(be && ts && n % be == 0, "%s part %d dtype %s: n=%lld not multiple of block %d",
                  t.name.c_str(), part, m64::dtype_name(pd), (long long)n, be);
            int64_t computed = n / be * ts;
            int64_t manifest = t.part_bytes(computed, part);
            CHECK(manifest == computed, "%s part %d manifest bytes %lld != computed %lld",
                  t.name.c_str(), part, (long long)manifest, (long long)computed);
            return computed;
        };
        int64_t pg = bytes(0), pu = bytes(1), pdown = bytes(2);
        int64_t per = pg + pu + pdown;
        CHECK(per <= t.expert_stride, "expert parts %lld > stride %lld", (long long)per, (long long)t.expert_stride);
        // dispatch a few blocks of each part of expert 0 through dequant_row
        // (exercises q4_k/q5_0/q5_1 on real pack bytes; golden checks exactness)
        {
            const uint8_t *base = (const uint8_t *)pack.ptr(t);
            int64_t off = 0;
            int64_t plens[3] = {pg, pu, pdown};
            for (int part = 0; part < 3; part++) {
                m64::DType pd = t.part_dtype(t.dtype, part);
                int64_t n = (int64_t)m64::dtype_block_elems(pd) * 4;
                if (n <= plens[part]) {
                    std::vector<float> out(n);
                    try { m64::dequant_row(pd, base + off, out.data(), n); }
                    catch (const std::exception &e) { CHECK(false, "expert part dequant %s/%d: %s", t.name.c_str(), part, e.what()); }
                }
                off += plens[part];
            }
        }
        printf("  ok  %-35s expert_stride=%lld parts=%lld+%lld+%lld (%s|%s|%s)\n", t.name.c_str(),
               (long long)t.expert_stride, (long long)pg, (long long)pu, (long long)pdown,
               m64::dtype_name(t.part_dtype(t.dtype, 0)), m64::dtype_name(t.part_dtype(t.dtype, 1)),
               m64::dtype_name(t.part_dtype(t.dtype, 2)));
    }

    printf(g_fail ? "\nFAILURES: %d\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
