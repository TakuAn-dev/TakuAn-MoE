// dequant.hpp — bit-exact GGML dequantization reference (float32 outputs).
// Phase 1 exactness scope: f32 / f16 / bf16 / q8_0 / q8_1.
// Q4_K / Q6_K are reference-complete (port of ggml dequantize_row_*_ref).
// IQ3_XXS is scaffolding: requires ggml's iq3_tables (not vendored; Phase 1
// model does not use it — calling it throws so no silent fallback exists).
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "pack_reader.hpp"

namespace m64 {

union F16Bits { uint16_t u; };

inline float f16_to_f32(uint16_t h) {
#if defined(__aarch64__) || defined(__ARM_NEON)
    __fp16 f;
    std::memcpy(&f, &h, 2);
    return (float)f;
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) out = sign;
        else { // subnormal
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            out = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) out = sign | 0x7F800000u | (man << 13);
    else out = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &out, 4);
    return f;
#endif
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };          // 34 B
struct BlockQ8_1 { uint16_t d, s; int8_t qs[32]; };       // 36 B
struct BlockQ4_0 { uint16_t d; uint8_t qs[16]; };          // 18 B
struct BlockQ4_K { uint8_t scales[12]; uint16_t d, dmin; uint8_t qs[128]; }; // 144 B
struct BlockQ6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; }; // 210 B

#define QK_K 256
#define N_ADD_BLOCKS_4_K 8

inline void dequantize_row_q8_0_ref(const BlockQ8_0 *x, float *y, int64_t k) {
    const float d = f16_to_f32(x->d);
    for (int64_t j = 0; j < 32; j++) y[j] = d * (float)x->qs[j];
    (void)k;
}

// ggml block_q8_1 stores s = f16(sum(qs)*scale) as a shared offset:
// x_i = d*qs_i - s/32  (pinned from quantize_row_q8_1_ref, ggml-quants.c:299).
inline void dequantize_row_q8_1_ref(const BlockQ8_1 *x, float *y, int64_t k) {
    const float d = f16_to_f32(x->d);
    const float m = f16_to_f32(x->s) / 32.0f;
    for (int64_t j = 0; j < 32; j++) y[j] = d * (float)x->qs[j] - m;
    (void)k;
}

// Q4_K / Q6_K: byte-identical ports of dequantize_row_q4_K / dequantize_row_q6_K
// from llama.cpp ggml/src/ggml-quants.c (this tree).
inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

inline void dequantize_row_q4_K_ref(const BlockQ4_K *x, float *y, int64_t k) {
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d = f16_to_f32(x[i].d);
        const float min = f16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

inline void dequantize_row_q6_K_ref(const BlockQ6_K *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

// IQ3_XXS scaffolding: needs ggml iq3_tables (256x8 ternary nibble table).
// Not vendored — Phase 1 model is Q8_1+F32, so this must never be reached.
inline void dequantize_row_iq3_xxs_ref(const void *, float *, int64_t) {
    throw std::runtime_error("dequant: IQ3_XXS reference is scaffolding (iq3_tables not vendored); "
                             "Phase 1 parity uses Q8_0/Q8_1/F32 only — no silent fallback.");
}

// Dispatch: one row of k float32 outputs.
inline void dequant_row(DType dt, const void *blocks, float *y, int64_t n_elems) {
    switch (dt) {
        case DType::F32: std::memcpy(y, blocks, (size_t)n_elems * 4); break;
        case DType::F16: {
            const uint16_t *h = (const uint16_t *)blocks;
            for (int64_t i = 0; i < n_elems; i++) y[i] = f16_to_f32(h[i]);
            break;
        }
        case DType::BF16: {
            const uint16_t *h = (const uint16_t *)blocks;
            for (int64_t i = 0; i < n_elems; i++) y[i] = bf16_to_f32(h[i]);
            break;
        }
        case DType::Q8_0:
            for (int64_t b = 0, nb = n_elems / 32; b < nb; b++)
                dequantize_row_q8_0_ref((const BlockQ8_0 *)blocks + b, y + b * 32, 32);
            break;
        case DType::Q8_1:
            for (int64_t b = 0, nb = n_elems / 32; b < nb; b++)
                dequantize_row_q8_1_ref((const BlockQ8_1 *)blocks + b, y + b * 32, 32);
            break;
        case DType::Q4_K:
            for (int64_t b = 0, nb = n_elems / QK_K; b < nb; b++)
                dequantize_row_q4_K_ref((const BlockQ4_K *)blocks + b, y + b * QK_K, QK_K);
            break;
        case DType::Q6_K:
            for (int64_t b = 0, nb = n_elems / QK_K; b < nb; b++)
                dequantize_row_q6_K_ref((const BlockQ6_K *)blocks + b, y + b * QK_K, QK_K);
            break;
        case DType::IQ3_XXS:
            for (int64_t b = 0, nb = n_elems / QK_K; b < nb; b++)
                dequantize_row_iq3_xxs_ref((const uint8_t *)blocks + b * 110, y + b * QK_K, QK_K);
            break;
        default: throw std::runtime_error("dequant: unsupported dtype");
    }
}

// n_elems from shape (row-major) and dtype-block rounding.
inline int64_t numel(const TensorInfo &t) {
    int64_t n = 1;
    for (auto d : t.shape) n *= d;
    return n;
}

} // namespace m64
