// xxh64.hpp — XXH64 (matches xxhash >= 0.8 reference, streaming-capable).
// Namespace-scoped copy for engine use; verified against Python xxhash in tests.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace m64 {

namespace xxh64_detail {
constexpr uint64_t P1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t P2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t P3 = 0x165667B19E3779F9ULL;
constexpr uint64_t P4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t P5 = 0x27D4EB2F165667C5ULL;

inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t round_acc(uint64_t acc, uint64_t inp) {
    acc += inp * P2;
    acc = rotl(acc, 31);
    return acc * P1;
}
inline uint64_t merge_acc(uint64_t acc, uint64_t val) {
    val = round_acc(0, val);
    acc ^= val;
    return acc * P1 + P4;
}
inline uint64_t get64(const uint8_t *p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v; // little-endian hosts (arm64/x86) only
}
inline uint32_t get32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
} // namespace xxh64_detail

class Xxh64 {
public:
    explicit Xxh64(uint64_t seed = 0)
        : acc1_(seed + xxh64_detail::P1 + xxh64_detail::P2), acc2_(seed + xxh64_detail::P2),
          acc3_(seed), acc4_(seed - xxh64_detail::P1), total_(0), seed_(seed) {}

    void update(const void *data, size_t len) {
        const uint8_t *p = (const uint8_t *)data;
        using namespace xxh64_detail;
        total_ += len;
        if (buf_len_ + len < 32) { std::memcpy(buf_ + buf_len_, p, len); buf_len_ += len; return; }
        if (buf_len_) {
            size_t fill = 32 - buf_len_;
            std::memcpy(buf_ + buf_len_, p, fill);
            process_block(buf_);
            p += fill; len -= fill; buf_len_ = 0;
        }
        while (len >= 32) { process_block(p); p += 32; len -= 32; }
        if (len) { std::memcpy(buf_, p, len); buf_len_ = len; }
    }

    uint64_t digest() const {
        using namespace xxh64_detail;
        uint64_t h;
        if (total_ >= 32) {
            h = rotl(acc1_, 1) + rotl(acc2_, 7) + rotl(acc3_, 12) + rotl(acc4_, 18);
            h = merge_acc(h, acc1_);
            h = merge_acc(h, acc2_);
            h = merge_acc(h, acc3_);
            h = merge_acc(h, acc4_);
        } else {
            h = seed_ + P5;
        }
        h += total_;
        const uint8_t *p = buf_;
        size_t l = buf_len_;
        while (l >= 8) { h ^= round_acc(0, get64(p)); h = rotl(h, 27) * P1 + P4; p += 8; l -= 8; }
        if (l >= 4) { h ^= (uint64_t)get32(p) * P1; h = rotl(h, 23) * P2 + P3; p += 4; l -= 4; }
        while (l--) { h ^= (*p++) * P5; h = rotl(h, 11) * P1; }
        h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
        return h;
    }

private:
    void process_block(const uint8_t *p) {
        using namespace xxh64_detail;
        acc1_ = round_acc(acc1_, get64(p));
        acc2_ = round_acc(acc2_, get64(p + 8));
        acc3_ = round_acc(acc3_, get64(p + 16));
        acc4_ = round_acc(acc4_, get64(p + 24));
    }
    uint64_t acc1_, acc2_, acc3_, acc4_, total_, seed_;
    uint8_t buf_[32];
    size_t buf_len_ = 0;
};

inline uint64_t xxh64(const void *data, size_t len, uint64_t seed = 0) {
    Xxh64 h(seed);
    h.update(data, len);
    return h.digest();
}

} // namespace m64
