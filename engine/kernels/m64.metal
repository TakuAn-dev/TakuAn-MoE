// m64.metal — Phase 1 MSL kernels. Compiled at load via newLibraryWithSource,
// archived with MTLBinaryArchive (no offline metal CLI; Makefile amendment A1).
//
// Bit-exactness policy: fast math is OFF at compile time and every reduction
// here is a strict in-order serial loop, so results are bitwise identical to
// the C++ float32 references in dequant.hpp / forward.cpp given identical
// inputs. Kernels needing tolerance-based parity (rmsnorm) are marked.

#include <metal_stdlib>
using namespace metal;

// Determinism: forbid implicit FMA contraction on both sides (C++ built with
// -ffp-contract=off). Same expression order => bitwise-identical results.
#pragma clang fp contract(off)

// ---- ggml block layouts (byte-identical) ---------------------------------
struct block_q8_0 { half d; char qs[32]; };            // 34 B
struct block_q8_1 { half d; half s; char qs[32]; };    // 36 B

// ---- dequant: bitwise vs dequantize_row_q8_*_ref --------------------------
kernel void dequant_q8_1(const device block_q8_1 * in  [[buffer(0)]],
                         device float * out             [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint32_t b = gid >> 5, j = gid & 31u;
    const float d = float(in[b].d);
    const float m = float(in[b].s) / 32.0f;
    out[gid] = d * float(in[b].qs[j]) - m;
}

kernel void dequant_q8_0(const device block_q8_0 * in  [[buffer(0)]],
                         device float * out             [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    const uint32_t b = gid >> 5, j = gid & 31u;
    const float d = float(in[b].d);
    out[gid] = d * float(in[b].qs[j]);
}

kernel void dequant_f16(const device ushort * in        [[buffer(0)]],
                        device float * out              [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    out[gid] = float(as_type<half>(in[gid]));
}

// ---- gemv: y[m] = W[m,:] . x + bias[m]  (W row-major m x k, f32) ---------
// Per-thread serial k-ascending accumulation => bitwise vs scalar C++ ref.
kernel void gemv_f32(const device float * W             [[buffer(0)]],
                     const device float * x             [[buffer(1)]],
                     const device float * bias          [[buffer(2)]],  // may be dummy
                     device float * y                   [[buffer(3)]],
                     constant uint & k                  [[buffer(4)]],
                     constant uint & has_bias           [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    const device float * w = W + (size_t)gid * k;
    float acc = 0.0f;
    for (uint j = 0; j < k; j++) acc += w[j] * x[j];
    if (has_bias) acc += bias[gid];
    y[gid] = acc;
}

// ---- fused dequant+gemv for q8_1 rows -------------------------------------
kernel void gemv_q8_1(const device block_q8_1 * W       [[buffer(0)]],
                      const device float * x            [[buffer(1)]],
                      device float * y                  [[buffer(2)]],
                      constant uint & k                 [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    const device block_q8_1 * w = W + (size_t)gid * (k / 32u);
    const uint nb = k / 32u;
    float acc = 0.0f;
    uint xi = 0;
    for (uint b = 0; b < nb; b++) {
        const float d = float(w[b].d);
        const float m = float(w[b].s) / 32.0f;
        for (uint j = 0; j < 32u; j++)
            acc += (d * float(w[b].qs[j]) - m) * x[xi + j];
        xi += 32u;
    }
    y[gid] = acc;
}

// ---- rmsnorm: y = x * rsqrt(mean(x^2)+eps) * w_eff  (tolerance class) ----
// w_eff[i] must be precomputed (1+w) by the host if required by the arch.
kernel void rmsnorm_row(const device float * x          [[buffer(0)]],
                        const device float * w          [[buffer(1)]],
                        device float * y                [[buffer(2)]],
                        constant float & eps            [[buffer(3)]],
                        constant uint & k               [[buffer(4)]],
                        uint row                        [[threadgroup_position_in_grid]],
                        uint tid                        [[thread_position_in_threadgroup]],
                        uint nth                        [[threads_per_threadgroup]]) {
    threadgroup float red[1024];
    const device float * xr = x + (size_t)row * k;
    float acc = 0.0f;
    for (uint i = tid; i < k; i += nth) acc += xr[i] * xr[i];  // serial per-thread
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nth >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = red[0] / (float)k;
    const float inv = rsqrt(mean + eps);
    for (uint i = tid; i < k; i += nth) y[(size_t)row * k + i] = xr[i] * inv * w[i];
}

// ---- SITU activation (Inkling MoE): y = x * sigmoid(beta*x) * tanh(x/lb) --
// Exact op order pinned against llama.cpp build_moe_ffn SITU path.
kernel void situ(device float * v                       [[buffer(0)]],
                 constant float & beta                  [[buffer(1)]],
                 constant float & lin_beta              [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    const float x = v[gid];
    const float sig = 1.0f / (1.0f + exp(-beta * x));
    v[gid] = x * sig * tanh(x / lin_beta);
}

// ---- swiglu-moe gather-GEMV (small, correct-first): -----------------------
// For token x: y[e][i] = dot(W_e[i,:], x) per selected expert e (top-K),
// where W_e rows are q8_1. gate/up fused tensor: row i of expert e at
// base + e*stride + i. down GEMV uses the same kernel with its own base/stride.
kernel void moe_gemv_q8_1(const device block_q8_1 * W   [[buffer(0)]],
                          const device float * x        [[buffer(1)]],
                          const device uint * experts   [[buffer(2)]],  // topK ids
                          device float * y              [[buffer(3)]],  // [topK][n_rows]
                          constant uint & k             [[buffer(4)]],
                          constant uint & n_rows        [[buffer(5)]],
                          constant uint64_t & stride_e  [[buffer(6)]],  // blocks per expert
                          uint2 gid [[thread_position_in_grid]]) {      // x=expert slot, y=row
    if (gid.y >= n_rows) return;
    const device block_q8_1 * w = W + (size_t)experts[gid.x] * stride_e
                                     + (size_t)gid.y * (k / 32u);
    const uint nb = k / 32u;
    float acc = 0.0f;
    uint xi = 0;
    for (uint b = 0; b < nb; b++) {
        const float d = float(w[b].d);
        const float m = float(w[b].s) / 32.0f;
        for (uint j = 0; j < 32u; j++)
            acc += (d * float(w[b].qs[j]) - m) * x[xi + j];
        xi += 32u;
    }
    y[(size_t)gid.x * n_rows + gid.y] = acc;
}

// ---- router: sigmoid(logits)+exp_probs_b, top-K by score, softmax over top-K
// then *scale. tid==0 serial => deterministic, bitwise vs C++ ref.
struct RouteOut { float w[16]; uint top[16]; };
kernel void moe_route(const device float * logits       [[buffer(0)]],  // [n_experts]
                      const device float * bias         [[buffer(1)]],  // exp_probs_b
                      device RouteOut & out             [[buffer(2)]],
                      constant uint & n_experts         [[buffer(3)]],
                      constant uint & top_k             [[buffer(4)]],
                      constant float & scale            [[buffer(5)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float score[256];
    for (uint i = 0; i < n_experts; i++)
        score[i] = 1.0f / (1.0f + exp(-logits[i])) + bias[i];
    float wsum = 0.0f;
    float sel_w[16]; uint sel_i[16];
    for (uint t = 0; t < top_k; t++) {
        uint bi = 0; float bv = -1e30f;
        for (uint i = 0; i < n_experts; i++)
            if (score[i] > bv) { bv = score[i]; bi = i; }
        sel_w[t] = bv; sel_i[t] = bi; score[bi] = -1e30f;
        wsum += bv;
    }
    for (uint t = 0; t < top_k; t++) {
        out.w[t] = (sel_w[t] / wsum) * scale;
        out.top[t] = sel_i[t];
    }
    for (uint t = top_k; t < 16; t++) { out.w[t] = 0.0f; out.top[t] = 0xFFFFFFFFu; }
}

// ---- elementwise helpers ---------------------------------------------------
kernel void add_inplace(device float * a, const device float * b,
                        uint gid [[thread_position_in_grid]]) { a[gid] += b[gid]; }
kernel void mul_inplace(device float * a, const device float * b,
                        uint gid [[thread_position_in_grid]]) { a[gid] *= b[gid]; }
kernel void scale_inplace(device float * a, constant float & s,
                          uint gid [[thread_position_in_grid]]) { a[gid] *= s; }
// softcap: x = cap * tanh(x / cap)
kernel void softcap(device float * a, constant float & cap,
                    uint gid [[thread_position_in_grid]]) { a[gid] = cap * tanh(a[gid] / cap); }
// argmax (greedy sample): tid0 serial
kernel void argmax(const device float * v, device uint * out,
                   constant uint & n, uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint bi = 0; float bv = -1e30f;
    for (uint i = 0; i < n; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
    *out = bi;
}
