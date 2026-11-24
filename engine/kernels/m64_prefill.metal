// m64_prefill.metal — Milestone-3A batched-prefill kernels.
// ASSUMES the runtime concatenates m64_p2a_args.h, then m64_p2a.metal, then
// this file into ONE translation unit (MetalRuntime::init order): helpers
// f16b / f16r / dot_c4u / M64_EXT are reused from m64_p2a.metal and MUST NOT
// be redefined here.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#pragma clang fp contract(off)
#ifndef M64_EXT
#define M64_EXT 1024u
#endif

static_assert(sizeof(GemmBatchArgs) == 48, "GemmBatchArgs");
static_assert(sizeof(ConvBatchArgs) == 64, "ConvBatchArgs");
static_assert(sizeof(KVWriteBatchF16Args) == 56, "KVWriteBatchF16Args");
static_assert(sizeof(AttnBatchF16Args) == 96, "AttnBatchF16Args");
static_assert(sizeof(MoeGroupedGemmArgs) == 88, "MoeGroupedGemmArgs");
static_assert(sizeof(GemvF32BatchArgs) == 40, "GemvF32BatchArgs");
static_assert(sizeof(MoeBucketArgs) == 32, "MoeBucketArgs");
static_assert(sizeof(MoeGroupedF32SgArgs) == 80, "MoeGroupedF32SgArgs");
static_assert(sizeof(SituF32BatchArgs) == 48, "SituF32BatchArgs");
static_assert(sizeof(GemmF32SgArgs) == 48, "GemmF32SgArgs");
static_assert(sizeof(EmbBatchArgs) == 40, "EmbBatchArgs");
static_assert(sizeof(RmsBatchArgs) == 80, "RmsBatchArgs");
static_assert(sizeof(ClipBatchArgs) == 40, "ClipBatchArgs");
static_assert(sizeof(GateMulBatchArgs) == 24, "GateMulBatchArgs");
static_assert(sizeof(SituBatchArgs) == 56, "SituBatchArgs");
static_assert(sizeof(MoeReduceBatchArgs) == 40, "MoeReduceBatchArgs");
static_assert(sizeof(RouterBatchArgs) == 48, "RouterBatchArgs");

// ---------------- Q8_0 weight-decode helpers ----------------
// One Q8_0 "unit" = 2 consecutive 32-elem blocks = 68 B = 17 register words:
//   words 0..8:   block0 f16 scale (W[0] low half) + codes bytes 2..33,
//   words 9..16:  block1 codes bytes 36..67 with the f16 scale in W[8] high.
// Integer dots are exact (char products promote to int), so the float result
// differs from the scalar oracle only by the cross-lane reduction order.
// Future Q4_0 / Q4_K batched kernels swap ONLY these two helpers.
static inline void q8_0_pair_scales(const uint W[17], float & d0, float & d1) {
    d0 = f16b((unsigned short)(W[0] & 0xFFFFu));
    d1 = f16b((unsigned short)(W[8] >> 16));
}

static inline void q8_0_pair_dot8(const uint W[17], const device uchar4 * X,
                                  float dx0, float dx1, float & a) {
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint v0 = (W[j] >> 16) | (W[j + 1] << 16);
        s0 += dot_c4u(as_type<char4>(v0), X[j]);
        s1 += dot_c4u(as_type<char4>(W[9 + j]), X[8 + j]);
    }
    float d0, d1;
    q8_0_pair_scales(W, d0, d1);
    a = fma(float(s0), dx0 * d0, a);
    a = fma(float(s1), dx1 * d1, a);
}

// Alignment-safe 68 B unit fetch (odd nb = k/32 makes row bases 2 mod 4).
static inline void q8_0_load_unit(const device uchar * wu, bool aligned4, uint * W) {
    if (aligned4) {
        const device uint * w4 = (const device uint *)wu;
        #pragma unroll
        for (int j = 0; j < 17; j++) W[j] = w4[j];
    } else {
        #pragma unroll
        for (int j = 0; j < 17; j++)
            W[j] = (uint)wu[4 * j] | ((uint)wu[4 * j + 1] << 8) |
                   ((uint)wu[4 * j + 2] << 16) | ((uint)wu[4 * j + 3] << 24);
    }
}

// Odd-tail single 32-block (34 B, scale f16 + 32 codes): byte-wise loads.
static inline void q8_0_single_dot8(const device uchar * wb, const device uchar4 * X,
                                    float dxv, float & a) {
    int s0 = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uchar4 wj = uchar4(wb[2 + 4 * j], wb[3 + 4 * j], wb[4 + 4 * j], wb[5 + 4 * j]);
        s0 += dot_c4u(as_type<char4>(wj), X[j]);
    }
    const float d = f16b((unsigned short)(((uint)wb[1] << 8) | (uint)wb[0]));
    a = fma(float(s0), dxv * d, a);
}

// --------------------------------------------- Q8_0 batched GEMM (prefill)
// grid = (ceil(nout/4), ceil(B/8)), threadgroup = (32, 4): warp = 1 output
// row x up to 8 tokens; each weight unit is loaded ONCE per warp and reused
// against all 8 register accumulators. Works for ANY k that is a multiple
// of 32 (full 68 B pair units + optional 34 B odd tail).
kernel void k_gemm_q8_batch(device float * SCR [[buffer(0)]],
                            constant GemmBatchArgs & A [[buffer(1)]],
                            const device uchar * PACK [[buffer(2)]],
                            uint3 M64_TITG [[thread_position_in_threadgroup]],
                            uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint row = M64_TGPG.x * 4u + M64_TITG.y;
    const uint lane = M64_TITG.x & 31u;
    if (row >= A.nout) return;                    // warp-uniform
    const uint t0 = M64_TGPG.y * 8u;
    const uint nb = A.k / 32u, nu = nb >> 1u, tail = nb & 1u;
    const size_t wrow0 = (size_t)A.woff + (size_t)row * ((size_t)nb * 34ull);
    const device uchar * wrow = PACK + wrow0;
    const bool w4 = (wrow0 & 3ul) == 0ul;
    bool valid[8];
    float acc[8];
    #pragma unroll
    for (int m = 0; m < 8; m++) { acc[m] = 0.0f; valid[m] = (t0 + (uint)m) < A.B; }
    for (uint u = lane; u < nu; u += 32u) {
        const device uchar * wu = wrow + (size_t)u * 68ull;
        uint W[17];
        q8_0_load_unit(wu, w4, W);
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            if (!valid[m]) continue;
            const uint t = t0 + (uint)m;
            const device uchar4 * xu = (const device uchar4 *)(SCR + A.xqoff +
                                           (size_t)t * A.xq_stride) + (size_t)u * 16u;
            const device float * dxp = SCR + A.dxoff + (size_t)t * A.dx_stride;
            q8_0_pair_dot8(W, xu, dxp[2 * u], dxp[2 * u + 1u], acc[m]);
        }
    }
    if (tail && lane == 0u) {                     // rare path: k/32 odd
        const device uchar * wt = wrow + (size_t)nu * 68ull;
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            if (!valid[m]) continue;
            const uint t = t0 + (uint)m;
            const device uchar4 * xu = (const device uchar4 *)(SCR + A.xqoff +
                                           (size_t)t * A.xq_stride) + (size_t)nu * 16u;
            q8_0_single_dot8(wt, xu, SCR[A.dxoff + (size_t)t * A.dx_stride + 2u * nu], acc[m]);
        }
    }
    #pragma unroll
    for (int m = 0; m < 8; m++) {
        const float a = simd_sum(acc[m]);
        if (lane == 0u && valid[m])
            SCR[A.yoff + (size_t)(t0 + (uint)m) * A.y_stride + row] = a;
    }
}

// ------------------------------------------- batched depthwise short conv
// Bitwise identical to B sequential k_conv_slot dispatches with the same
// rs state: the per-channel state (s0, s1, s2) lives in registers across the
// B tokens and is read back / written back exactly once. dyn row per token b
// sits at SCR + dynoff + b * dyn_stride + (c/group)*4 (dyn_stride=0 shares).
kernel void k_conv_batch(device float * SCR [[buffer(0)]],
                         constant ConvBatchArgs & A [[buffer(1)]],
                         const device uchar * PACK [[buffer(2)]],
                         uint3 M64_TIG [[thread_position_in_grid]]) {
    const uint c = M64_TIG.x;
    if (c >= A.C) return;
    device float * st = SCR + A.rsoff + c * 3u;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const device float * w = ((const device float *)(PACK + A.woff)) + c * 4u;
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    for (uint b = 0; b < A.B; b++) {
        const float xin = SCR[A.xoff + b * A.x_stride + c];
        float outv = s0 * w0 + s1 * w1 + s2 * w2 + xin * w3;
        if (A.hasdyn) {
            const device float * dyn = SCR + A.dynoff + b * A.dyn_stride + (c / A.group) * 4u;
            outv += dyn[0] * xin + dyn[1] * s2 + dyn[2] * s1 + dyn[3] * s0;
        }
        SCR[A.yoff + b * A.y_stride + c] = (xin + outv) * A.scale;
        s0 = s1; s1 = s2; s2 = xin;
    }
    st[0] = s0; st[1] = s1; st[2] = s2;
}

// --------------------------------------------- batched FP16 KV write
// grid = dispatchThreads (n, B): (c, b) converts one K and one V element of
// token b at position pos0+b into KV slot (is_swa ? (pos0+b)%ring_cap : pos0+b).
kernel void k_kv_write_f16_batch(const device float * SCR [[buffer(0)]],
                                 constant KVWriteBatchF16Args & A [[buffer(1)]],
                                 device half * KV [[buffer(2)]],
                                 uint3 M64_TIG [[thread_position_in_grid]]) {
    const uint c = M64_TIG.x, b = M64_TIG.y;
    if (c >= A.n || b >= A.B) return;
    const uint pos = A.pos0 + b;
    const uint slot = A.is_swa ? (pos % A.ring_cap) : pos;
    KV[A.kvk_base + (size_t)slot * A.n + c] = half(SCR[A.koff + b * A.k_stride + c]);
    KV[A.kvv_base + (size_t)slot * A.n + c] = half(SCR[A.voff + b * A.v_stride + c]);
}

// --------------------------------------- batched causal attention over FP16 KV
// Numerics replicate k_attn_head_f16 exactly (same fma order, fused relbias
// dot on lanes 0..15, online softmax, zero-sink term, (float)(1.0 / S)); the
// only addition is the per-token tau computed inside the kernel with the same
// formula forward_metal.mm applies on the host (tau == 1 for SWA layers).
kernel void k_attn_head_f16_batch(device float * SCR [[buffer(0)]],
                                  constant AttnBatchF16Args & A [[buffer(1)]],
                                  const device uchar * PACK [[buffer(2)]],
                                  const device half * KV [[buffer(3)]],
                                  uint3 M64_TITG [[thread_position_in_threadgroup]],
                                  uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint h = M64_TGPG.x, b = M64_TGPG.y, lane = M64_TITG.x;
    const uint pos = A.pos0 + b;
    const float tau = A.is_swa ? 1.0f
        : (metal::log(fmax(1.0f, ((float)pos + 1.0f) / A.log_scale_floor)) *
               A.log_scale_alpha + 1.0f);
    const uint t = lane * 4u;
    const device float * q = SCR + A.qoff + (size_t)b * A.q_stride + h * 128u;
    half qh[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { float qf = q[t + i] * tau; qh[i] = half(qf); }
    const device float * rout = SCR + A.rout + (size_t)b * A.rout_stride + h * 16u;
    const float sink = SCR[A.sinkoff + (size_t)b * A.sink_stride + h];
    const uint kvoff = (h / 3u) * 128u;
    float oh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M = -INFINITY;
    float S = 0.0;
    const uint j_start = (A.is_swa && pos > A.n_swa) ? (pos - A.n_swa) : 0u;
    for (uint j = j_start; j <= pos; j++) {
        const uint slot_idx = A.is_swa ? (j % A.ring_cap) : j;
        const device half * kj = KV + A.kvkoff + (size_t)slot_idx * 1024u + kvoff + t;
        float d = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) d = fma(float(qh[i]), float(kj[i]), d);
        d = simd_sum(d);
        uint dist = pos - j;
        float bv = 0.0f;
        if (dist < M64_EXT) {
            const device float * brow = ((const device float *)(PACK + A.relbiaspack)) +
                                        (size_t)dist * 16u;
            // split dot over 16 channels across lanes 0..15 (else simd_sum = 32x dot)
            float v = (lane < 16u) ? (brow[lane] * (rout[lane] * tau)) : 0.0f;
            bv = simd_sum(v);
        }
        float s = d * A.scale + f16r(bv - sink);
        float ms = 1.0f, vs;
        if (s > M) {
            ms = metal::exp(M - s);
            oh[0] *= ms; oh[1] *= ms; oh[2] *= ms; oh[3] *= ms;
            M = s; vs = 1.0f;
        } else {
            vs = metal::exp(s - M);
        }
        const device half * vj = KV + A.kvvoff + (size_t)slot_idx * 1024u + kvoff + t;
        #pragma unroll
        for (int i = 0; i < 4; i++) oh[i] = fma(vs, float(vj[i]), oh[i]);
        S = S * (float)ms + (float)vs;
    }
    // zero-sink term (sinks tensor == 0)
    {
        float ms = 1.0f, vs;
        if (0.0f > M) {
            ms = metal::exp(M - 0.0f);
            oh[0] *= ms; oh[1] *= ms; oh[2] *= ms; oh[3] *= ms;
            M = 0.0f; vs = 1.0f;
        } else vs = metal::exp(0.0f - M);
        S = S * (float)ms + (float)vs;
    }
    const float inv = (float)(1.0 / S);
    device float * out = SCR + A.outoff + (size_t)b * A.out_stride + h * 128u;
    #pragma unroll
    for (int i = 0; i < 4; i++) out[t + i] = oh[i] * inv;
}

// ---------------- FlashAttention-2 tiled prefill over FP16 KV (Milestone 4C)
// Same AttnBatchF16Args / KV layout / exact Inkling-2-Mini semantics as
// k_attn_head_f16_batch above: q*tau f16-rounded, GQA kvoff (h/3)*128,
// tau, causal mask, SWA window + ring slot wrap (j % ring_cap), fused
// relbias bv = sum_c relbias[dist][c]*(rout[c]*tau) for dist < M64_EXT
// (bv = 0 beyond, but the f16r(bv - sink) bias term NEVER drops -sink),
// online softmax with row-wise M/S and O rescaling, then the final
// zero-sink term before out = O * (float)(1.0 / S).
//
// Tiling: BQ = 32 query tokens x BK = 32 KV tokens.
//   grid = (kNH, ceil(B/32), 1); threads = (32, 4) = 4 simdgroups.
//   Each simdgroup owns 8 query rows and keeps the 8x128 O block in
//   16 simdgroup_float8x8 registers. Tensor-core mmas:
//   S = Qh @ K^T (b loaded with transpose from tgKV) and O += P @ V.
// Threadgroup memory = 24576 B (24.0 KiB <= 32 KiB cap):
//   tgBig  half [2*32*136]  17408 B  rows [0..31] K tile then V tile (ld 136),
//          rows [4352..]    Qh tile (hoisted to registers before the loop);
//          the tgKV region (free between QK^T and the V load, and again in
//          the epilogue) doubles as a float32 [32][64] staging tile (ld 64)
//          for rescaling/writing-out accO: Apple Silicon's simdgroup_matrix
//          element mapping is quad/Morton, so thread_elements() rows are
//          NOT lane>>2 — all row-wise scaling round-trips through it.
//   tgS    float [32][36]    4608 B  S scores, then P halfs in-place (ld 72h).
//   tgRT   float [32][16]    2048 B  rout[c]*tau per query row.
//   tgSink/tgM/tgSum/tgAlpha        512 B row-wise scalars.
// Thread mappings (both cover all 32 query rows): loader/softmax/staging pass
// uses qrow = tid>>2, seg = tid&3 (one 32-lane simdgroup holds 8 rows as
// 4-thread quads -> butterfly max/sum with simd_shuffle_xor 1,2); the mma
// tiles use the native simdgroup_matrix row = sgid*8 + (lane>>2) mapping
// only via simdgroup_load/store, never through thread_elements().
// P is f16-rounded for the PV product (tensor cores), so parity with the
// scalar batch kernel is tolerance-based (GATE_SG in the test), not bitwise.
// The relbias DRAM read has a tile-level fast path when every dist in the
// tile is >= M64_EXT.
#define M64_FQ_LD 136u          // padded threadgroup ld (halfs) for Q/K/V/O staging
#define M64_FS_LD  36u          // threadgroup ld for tgS (floats) / tgP (halfs: 2x)

kernel void k_attn_f16_flash_prefill(device float * SCR [[buffer(0)]],
                                     constant AttnBatchF16Args & A [[buffer(1)]],
                                     const device uchar * PACK [[buffer(2)]],
                                     const device half * KV [[buffer(3)]],
                                     uint3 M64_TITG [[thread_position_in_threadgroup]],
                                     uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint h = M64_TGPG.x, b0 = M64_TGPG.y * 32u;
    const uint tid = M64_TITG.y * 32u + M64_TITG.x;      // 0..127
    const uint sgid = tid >> 5u;
    const uint qrow = tid >> 2u, seg = tid & 3u;         // loader/softmax mapping

    threadgroup half tgBig[2u * 32u * M64_FQ_LD];        // K/V tiles + Qh tile + O float staging
    threadgroup half * tgQ = tgBig + 32u * M64_FQ_LD;
    threadgroup half * tgKV = tgBig;
    threadgroup float tgS[32u * M64_FS_LD];
    threadgroup float tgRT[32u * 16u];
    threadgroup float tgSink[32u];
    threadgroup float tgAlpha[32u];
    threadgroup float tgML[2u * 32u];                    // [0..31]=M, [32..63]=S
    threadgroup float * tgM = tgML;
    threadgroup float * tgSum = tgML + 32u;

    const uint kvoff = (h / 3u) * 128u;
    const uint b_hi = min(b0 + 31u, A.B - 1u);           // grid.y = ceil(B/32) => b0 < A.B
    const uint pmin = A.pos0 + b0, pmax = A.pos0 + b_hi;
    const uint js0 = (A.is_swa && pmin > A.n_swa) ? (pmin - A.n_swa) : 0u;

    // ---- load Qh = half(q * tau) and per-row scalars (rout*tau, sink)
    {
        const uint b = b0 + qrow;
        const bool v = b < A.B;
        float tau = 1.0f;
        if (v && !A.is_swa)
            tau = metal::log(fmax(1.0f, ((float)(A.pos0 + b) + 1.0f) / A.log_scale_floor)) *
                      A.log_scale_alpha + 1.0f;
        threadgroup half * dq = tgQ + qrow * M64_FQ_LD + seg * 32u;
        if (v) {
            const device float * q = SCR + A.qoff + (size_t)b * A.q_stride + h * 128u;
            #pragma unroll
            for (uint i = 0; i < 32u; i++) dq[i] = half(q[seg * 32u + i] * tau);
            if (seg == 0u) {
                tgSink[qrow] = SCR[A.sinkoff + (size_t)b * A.sink_stride + h];
                const device float * rt = SCR + A.rout + (size_t)b * A.rout_stride + h * 16u;
                #pragma unroll
                for (uint c = 0; c < 16u; c++) tgRT[qrow * 16u + c] = rt[c] * tau;
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 32u; i++) dq[i] = half(0.0f);
            if (seg == 0u) {
                tgSink[qrow] = 0.0f;
                #pragma unroll
                for (uint c = 0; c < 16u; c++) tgRT[qrow * 16u + c] = 0.0f;
            }
        }
        if (seg == 0u) { tgM[qrow] = -INFINITY; tgSum[qrow] = 0.0f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 aQ[16];
    #pragma unroll
    for (uint d = 0u; d < 16u; d++)
        simdgroup_load(aQ[d], tgQ + (sgid * 8u) * M64_FQ_LD + d * 8u, M64_FQ_LD);

    simdgroup_float8x8 accO[16];
    #pragma unroll
    for (uint c = 0u; c < 16u; c++)
        accO[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const float sink = tgSink[qrow];
    for (uint j0 = js0; j0 <= pmax; j0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- load K tile [32][128] (j > pmax -> zeros)
        {
            const uint jj = j0 + qrow;
            threadgroup half * dK = tgKV + qrow * M64_FQ_LD + seg * 32u;
            if (jj <= pmax) {
                const uint slot = A.is_swa ? (jj % A.ring_cap) : jj;
                const device half * src = KV + A.kvkoff + (size_t)slot * 1024u + kvoff + seg * 32u;
                #pragma unroll
                for (uint i = 0; i < 32u; i++) dK[i] = src[i];
            } else {
                #pragma unroll
                for (uint i = 0; i < 32u; i++) dK[i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- S = Qh @ K^T -> tgS (raw dot; scale applied in softmax pass)
        {
            simdgroup_float8x8 accS[4];
            #pragma unroll
            for (uint k = 0u; k < 4u; k++)
                accS[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma unroll
            for (uint d = 0u; d < 16u; d++)
                #pragma unroll
                for (uint k = 0u; k < 4u; k++) {
                    simdgroup_half8x8 bT;
                    simdgroup_load(bT, tgKV + (k * 8u) * M64_FQ_LD + d * 8u, (ulong)M64_FQ_LD,
                                           ulong2(0u, 0u), true);
                    simdgroup_multiply_accumulate(accS[k], aQ[d], bT, accS[k]);
                }
            #pragma unroll
            for (uint k = 0u; k < 4u; k++)
                simdgroup_store(accS[k], tgS + (sgid * 8u) * M64_FS_LD + k * 8u, M64_FS_LD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- relbias + mask + online softmax: one 4-thread quad per row
        {
            const uint pos_q = A.pos0 + b0 + qrow;
            const uint jstart_q = (A.is_swa && pos_q > A.n_swa) ? (pos_q - A.n_swa) : 0u;
            const bool vq = (b0 + qrow) < A.B;
            // fast path: EVERY valid entry in this tile has dist >= M64_EXT
            const bool skip_rb = ((int64_t)pmin - (int64_t)(j0 + 31u)) >= (int64_t)M64_EXT;
            float sv[8];
            const threadgroup float * srow = tgS + qrow * M64_FS_LD + seg * 8u;
            float m = -INFINITY;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                const uint j = j0 + seg * 8u + i;
                float s = -INFINITY;
                if (vq && j <= pos_q && j >= jstart_q) {
                    s = srow[i] * A.scale;
                    const int dist = (int)pos_q - (int)j;
                    if (skip_rb) {
                        s += f16r(0.0f - sink);       // bv = 0, -sink MUST stay (see batch kernel)
                    } else {
                        float bv = 0.0f;
                        if (dist < (int)M64_EXT) {    // out-of-extent relbias rows are 0
                            const device float * brow =
                                ((const device float *)(PACK + A.relbiaspack)) + (size_t)dist * 16u;
                            const threadgroup float * rt = tgRT + qrow * 16u;
                            #pragma unroll
                            for (uint c = 0; c < 16u; c++) bv = fma(brow[c], rt[c], bv);
                        }
                        s += f16r(bv - sink);
                    }
                }
                sv[i] = s;
                m = fmax(m, s);
            }
            m = fmax(m, simd_shuffle_xor(m, 1u));
            m = fmax(m, simd_shuffle_xor(m, 2u));
            const float M_old = tgM[qrow];
            const float M_new = fmax(M_old, m);
            const float alpha = (M_new == -INFINITY) ? 1.0f : metal::exp(M_old - M_new);
            float rs = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                const float p = (sv[i] == -INFINITY) ? 0.0f : metal::exp(sv[i] - M_new);
                rs += p;
                ((threadgroup half *)tgS)[qrow * (M64_FS_LD * 2u) + seg * 8u + i] = half(p);
            }
            rs += simd_shuffle_xor(rs, 1u);
            rs += simd_shuffle_xor(rs, 2u);
            if (seg == 0u) {
                tgAlpha[qrow] = alpha;
                tgM[qrow] = M_new;
                tgSum[qrow] = fma(tgSum[qrow], alpha, rs);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- rescale accO rows to the running max via tgKV float staging.
        // Apple Silicon simdgroup_matrix element mapping is quad/Morton, NOT
        // row = lane>>2, so thread_elements() must not be touched: round-trip
        // the tiles through (threadgroup float *)tgKV (free between QK^T and
        // the V load), scale by row, and load back. j0 == js0: accO is all
        // zeros, skipping the pass is exact and saves work.
        if (j0 > js0) {
            threadgroup float * tgF = (threadgroup float *)tgKV;   // 32*64 floats
            #pragma unroll
            for (uint hb = 0u; hb < 16u; hb += 8u) {
                #pragma unroll
                for (uint c = 0u; c < 8u; c++)
                    simdgroup_store(accO[hb + c], tgF + (sgid * 8u) * 64u + c * 8u, 64u);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float aq = tgAlpha[qrow];
                threadgroup float * frow = tgF + qrow * 64u + seg * 16u;
                #pragma unroll
                for (uint i = 0u; i < 16u; i++) frow[i] *= aq;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma unroll
                for (uint c = 0u; c < 8u; c++)
                    simdgroup_load(accO[hb + c], tgF + (sgid * 8u) * 64u + c * 8u, 64u);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        // ---- load V tile [32][128]
        {
            const uint jj = j0 + qrow;
            threadgroup half * dV = tgKV + qrow * M64_FQ_LD + seg * 32u;
            if (jj <= pmax) {
                const uint slot = A.is_swa ? (jj % A.ring_cap) : jj;
                const device half * src = KV + A.kvvoff + (size_t)slot * 1024u + kvoff + seg * 32u;
                #pragma unroll
                for (uint i = 0; i < 32u; i++) dV[i] = src[i];
            } else {
                #pragma unroll
                for (uint i = 0; i < 32u; i++) dV[i] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- O += P @ V (P read back f16-rounded, exactly what the mma sees)
        #pragma unroll
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pT;
            simdgroup_load(pT, (const threadgroup half *)tgS + (sgid * 8u) * (M64_FS_LD * 2u) + kk,
                           M64_FS_LD * 2u);
            #pragma unroll
            for (uint c = 0u; c < 16u; c++) {
                simdgroup_half8x8 vT;
                simdgroup_load(vT, tgKV + kk * M64_FQ_LD + c * 8u, M64_FQ_LD);
                simdgroup_multiply_accumulate(accO[c], pT, vT, accO[c]);
            }
        }
    }
    // ---- final zero-sink term (sinks tensor == 0), same as
    // k_attn_head_f16_batch, then write O through the tgKV float staging
    // (2 x 64-column passes): never touch thread_elements() (see above).
    {
        threadgroup float * tgF = (threadgroup float *)tgKV;   // 32*64 floats
        const float M = tgM[qrow];
        const float ms = (0.0f > M) ? ((M == -INFINITY) ? 0.0f : metal::exp(M)) : 1.0f;
        const float sum_final = (0.0f > M) ? fma(tgSum[qrow], metal::exp(M), 1.0f)
                                           : tgSum[qrow] + metal::exp(0.0f - M);
        const float scale_out = ms * (float)(1.0 / sum_final);
        const uint b = b0 + qrow;
        device float * out = SCR + A.outoff + (size_t)b * A.out_stride + h * 128u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint hb = 0u; hb < 16u; hb += 8u) {
            #pragma unroll
            for (uint c = 0u; c < 8u; c++)
                simdgroup_store(accO[hb + c], tgF + (sgid * 8u) * 64u + c * 8u, 64u);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (b < A.B) {
                const threadgroup float * srcrow = tgF + qrow * 64u + seg * 16u;
                #pragma unroll
                for (uint i = 0u; i < 16u; i++)
                    out[hb * 8u + seg * 16u + i] = srcrow[i] * scale_out;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// --------------------------------------------- grouped MoE Q8_0 GEMM (prefill)
// grid = (ceil(nout/4), n_experts), threadgroup = (32, 4): warp = one output
// row of one expert; walks the expert's token list in chunks of 8, loading
// each weight unit ONCE per chunk and reusing it against 8 accumulators.
kernel void k_moe_grouped_gemm_q8(device float * SCR [[buffer(0)]],
                                  constant MoeGroupedGemmArgs & A [[buffer(1)]],
                                  const device uchar * PACK [[buffer(2)]],
                                  uint3 M64_TITG [[thread_position_in_threadgroup]],
                                  uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint row = M64_TGPG.x * 4u + M64_TITG.y;
    const uint exp_id = M64_TGPG.y;
    const uint lane = M64_TITG.x & 31u;
    if (row >= A.nout || exp_id >= A.n_experts) return;   // warp-uniform
    const device uint * counts = (const device uint *)(SCR + A.count_off);
    const device uint * offsets = (const device uint *)(SCR + A.offset_off);
    const device uint * list = (const device uint *)(SCR + A.list_off);
    const uint cnt = counts[exp_id];
    if (cnt == 0u) return;                                // warp-uniform
    const uint base = offsets[exp_id];
    const uint nb = A.k / 32u, nu = nb >> 1u, tail = nb & 1u;
    // Milestone 3B-2 arena mode: wtab_off != 0 -> arena slot index comes
    // from the host-built slot table (uint32 as bits in SCR).
    const uint slot = A.wtab_off != 0u
        ? ((const device uint *)SCR)[A.wtab_off + exp_id]
        : exp_id;
    const size_t wrow0 = (size_t)A.wbase + (size_t)slot * A.exp_stride +
                         (size_t)A.part_off + (size_t)row * ((size_t)nb * 34ull);
    const device uchar * wrow = PACK + wrow0;
    const bool w4 = (wrow0 & 3ul) == 0ul;
    for (uint off = 0u; off < cnt; off += 8u) {
        uint in_idx[8], out_idx[8];
        bool valid[8];
        float acc[8];
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            acc[m] = 0.0f;
            valid[m] = (off + (uint)m) < cnt;
            in_idx[m] = 0u;
            out_idx[m] = 0u;
            if (valid[m]) {
                const uint p = list[base + off + (uint)m];
                in_idx[m] = A.is_down ? p : (p / A.topk);
                out_idx[m] = p;
            }
        }
        for (uint u = lane; u < nu; u += 32u) {
            const device uchar * wu = wrow + (size_t)u * 68ull;
            uint W[17];
            q8_0_load_unit(wu, w4, W);
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                if (!valid[m]) continue;
                const device uchar4 * xu = (const device uchar4 *)(SCR + A.xqoff +
                                               (size_t)in_idx[m] * A.xq_stride) + (size_t)u * 16u;
                const device float * dxp = SCR + A.dxoff + (size_t)in_idx[m] * A.dx_stride;
                q8_0_pair_dot8(W, xu, dxp[2 * u], dxp[2 * u + 1u], acc[m]);
            }
        }
        if (tail && lane == 0u) {
            const device uchar * wt = wrow + (size_t)nu * 68ull;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                if (!valid[m]) continue;
                const device uchar4 * xu = (const device uchar4 *)(SCR + A.xqoff +
                                               (size_t)in_idx[m] * A.xq_stride) + (size_t)nu * 16u;
                q8_0_single_dot8(wt, xu, SCR[A.dxoff + (size_t)in_idx[m] * A.dx_stride + 2u * nu], acc[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            const float a = simd_sum(acc[m]);
            if (lane == 0u && valid[m])
                SCR[A.yoff + (size_t)out_idx[m] * A.y_stride + row] = a;
        }
    }
}

// ==================================================================
// Milestone-3B-1 kernels
// ==================================================================

// ---- swappable Q8_0 weight-tile dequant helper (future Q4_0/Q4_K kernels
// replace ONLY this function) ----
// Dequant tile rows [m0, m0+32) x blocks [blk] (32 codes) of a Q8_0 matrix
// into threadgroup memory as half. Byte-wise scale read: with odd nb the row
// base is 2 mod 4, so NEVER cast bp to uint16/uint pointers here.
// 128 threads: thread (row = tid>>2, seg = tid&3) writes 8 codes of 1 row.
static inline void dequant_q8_0_tile32x32_to_tg(const device uchar * wmat,
                                                uint m0, uint nout, uint blk,
                                                uint nb, uint tid,
                                                threadgroup half * tg, uint ld) {
    const uint row = tid >> 2u, seg = tid & 3u;
    threadgroup half * dst = tg + row * ld + seg * 8u;
    if (m0 + row < nout) {
        const device uchar * bp = wmat +
            ((size_t)(m0 + row) * (size_t)nb + (size_t)blk) * 34ull;
        const float d = f16b((unsigned short)(((uint)bp[1] << 8) | (uint)bp[0]));
        #pragma unroll
        for (uint i = 0; i < 8u; i++)
            dst[i] = half(d * float(s8(bp[2 + seg * 8u + i])));
    } else {
        #pragma unroll
        for (uint i = 0; i < 8u; i++) dst[i] = half(0.0f);
    }
}

// ---------------- simdgroup_matrix Q8_0 GEMM with FP32 activations ----------------
// y[b][m] = (add_to_out ? y[b][m] : 0) + dot(W_m, x_b); k multiple of 32.
// grid = (ceil(nout/32), ceil(B/32)), threadgroup = (32, 4) = 4 simdgroups,
// each owning a 16x16 quadrant (acc[2][2] of simdgroup_float8x8) of the
// 32x32 output tile. Weight dequant goes through dequant_q8_0_tile32x32_to_tg
// so future quant formats swap only that helper. Inputs are converted to
// half for the tensor-core product and accumulated in float, so numerics
// are tolerance-parity (NOT bitwise) with the int-dot k_gemv/k_gemm path.
#define M64_SG_LD 40u      // padded threadgroup leading dim (halfs)

kernel void k_gemm_q8_f32in_sg(device float * SCR [[buffer(0)]],
                               constant GemmF32SgArgs & A [[buffer(1)]],
                               const device uchar * PACK [[buffer(2)]],
                               uint3 M64_TITG [[thread_position_in_threadgroup]],
                               uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint tid = M64_TITG.y * 32u + M64_TITG.x;      // 0..127
    const uint m0 = M64_TGPG.x * 32u, b0 = M64_TGPG.y * 32u;
    const uint sg_row = ((tid >> 5u) >> 1u) * 16u;       // simdgroup quadrant
    const uint sg_col = ((tid >> 5u) & 1u) * 16u;
    threadgroup half tgW[32u * M64_SG_LD];
    threadgroup half tgX[32u * M64_SG_LD];               // [k_local][b_local]
    threadgroup float tgO[32u * M64_SG_LD];
    const device uchar * wmat = PACK + A.woff;

    simdgroup_float8x8 acc[2][2];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < 2; c++)
            acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint nb = A.k / 32u;
    // X slice loader: thread (bcol = tid>>2, kseg = tid&3) covers 8 contiguous
    // k values of one token -> tgX[k_local][b_local].
    const uint bcol = tid >> 2u, kseg = tid & 3u;
    const bool xvalid = (b0 + bcol) < A.B;
    const device float * xb0 = SCR + A.xoff + (size_t)(b0 + bcol) * A.x_stride;

    for (uint blk = 0u; blk < nb; blk++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dequant_q8_0_tile32x32_to_tg(wmat, m0, A.nout, blk, nb, tid, tgW, M64_SG_LD);
        {
            const uint k0 = blk * 32u;
            threadgroup half * dx = tgX + kseg * 8u * M64_SG_LD + bcol;
            if (xvalid) {
                const device float * xb = xb0 + k0;
                #pragma unroll
                for (uint i = 0; i < 8u; i++) dx[i * M64_SG_LD] = half(xb[kseg * 8u + i]);
            } else {
                #pragma unroll
                for (uint i = 0; i < 8u; i++) dx[i * M64_SG_LD] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0m, b1m;
            simdgroup_load(a0, tgW + (sg_row + 0u) * M64_SG_LD + kk, M64_SG_LD);
            simdgroup_load(a1, tgW + (sg_row + 8u) * M64_SG_LD + kk, M64_SG_LD);
            simdgroup_load(b0m, tgX + kk * M64_SG_LD + sg_col, M64_SG_LD);
            simdgroup_load(b1m, tgX + kk * M64_SG_LD + sg_col + 8u, M64_SG_LD);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0m, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1m, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0m, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1m, acc[1][1]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < 2; c++)
            simdgroup_store(acc[r][c], tgO + (sg_row + (uint)r * 8u) * M64_SG_LD +
                                     sg_col + (uint)c * 8u, M64_SG_LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint m = m0 + (tid >> 2u);
    if (m < A.nout) {
        const uint cbase = (tid & 3u) * 8u;
        const threadgroup float * src = tgO + (tid >> 2u) * M64_SG_LD + cbase;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            const uint b = b0 + cbase + i;
            if (b < A.B) {
                device float * y = SCR + A.yoff + (size_t)b * A.y_stride + m;
                const float v = src[i];
                *y = A.add_to_out ? (*y + v) : v;
            }
        }
    }
}

// ---------------- batched embedding row gather ----------------
// grid = (row_blocks, B) threads of 32; thread M64_TIG.x is one 34 B Q8_0
// block (identical inner loop to k_emb_gather). Token ids are uint32 stored
// in SCR float slots (same convention as router sel[]).
kernel void k_emb_gather_batch(device float * SCR [[buffer(0)]],
                               constant EmbBatchArgs & A [[buffer(1)]],
                               const device uchar * PACK [[buffer(2)]],
                               uint3 M64_TIG [[thread_position_in_grid]]) {
    const uint blk = M64_TIG.x, b = M64_TIG.y;
    if (blk >= A.row_blocks || b >= A.B) return;
    const device uint * tok = (const device uint *)(SCR + A.tok_off);
    const device uchar * wp = PACK + A.woff +
        (size_t)tok[b] * ((size_t)A.row_blocks * 34ull) + (size_t)blk * 34ull;
    const float d = f16b((unsigned short)(((uint)wp[1] << 8) | (uint)wp[0]));
    device float * y = SCR + A.boff + (size_t)b * A.boff_stride + blk * 32u;
    #pragma unroll
    for (int i = 0; i < 32; i++) y[i] = d * float(s8(wp[2 + i]));
}

// ---------------- batched rmsnorm ----------------
// grid = (rows, B) threadgroups of 256. Reduction code shape is kept
// identical to k_rms (float4 accumulate + warp simd_sum tree) so each
// (token, row) result is BITWISE equal to the matching single-token k_rms
// dispatch at equal offsets. Optional Q8_0 output re-quantizes the OUTPUT
// row with the exact k_quant block math (per-block single thread, 32 lanes).
kernel void k_rms_batch(device float * SCR [[buffer(0)]],
                        constant RmsBatchArgs & A [[buffer(1)]],
                        const device uchar * PACK [[buffer(2)]],
                        uint3 M64_TGPG [[threadgroup_position_in_grid]],
                        uint3 M64_TITG [[thread_position_in_threadgroup]],
                        uint3 M64_NPTG [[threads_per_threadgroup]]) {
    const uint r = M64_TGPG.x, b = M64_TGPG.y;
    if (r >= A.rows || b >= A.B) return;
    const uint xcur = A.xoff + b * A.x_stride + r * A.x_row;
    const device float * x = SCR + xcur;
    const device float * w = A.hasw ? ((const device float *)(PACK + A.wpack)) : nullptr;
    const bool write_y = (A.yoff != M64_NOFFU);
    const uint ycur = write_y ? (A.yoff + b * A.y_stride + r * A.y_row) : 0u;
    device float * y = write_y ? (SCR + ycur) : nullptr;
    const uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    threadgroup float red[256];
    const bool vec4_in = ((A.k & 3u) == 0u) && ((xcur & 3u) == 0u);
    float s;
    if (vec4_in) {
        const device float4 * x4 = (const device float4 *)x;
        float4 av = float4(0.0f);
        for (uint i = tid; i < (A.k >> 2); i += nth) { float4 v = x4[i]; av += v * v; }
        s = (av.x + av.y) + (av.z + av.w);
    } else {
        s = 0.0f;
        for (uint i = tid; i < A.k; i += nth) { float v = x[i]; s += v * v; }
    }
    s = simd_sum(s);
    uint lane0 = tid & 31u;
    if (lane0 == 0u) red[tid >> 5u] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float t = (lane0 < 8u) ? red[lane0] : 0.0f;
        t = simd_sum(t);
        if (tid == 0u) red[0] = 1.0f / (float)(sqrt((float)(t / float(A.k)) + A.eps));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = (float)red[0];
    if (write_y && vec4_in && (!A.hasw || (A.wpack & 15ul) == 0ul) && ((ycur & 3u) == 0u)) {
        const device float4 * x4 = (const device float4 *)x;
        const device float4 * w4 = w ? (const device float4 *)w : nullptr;
        device float4 * y4 = (device float4 *)y;
        const float4 inv4 = float4(inv);
        for (uint i = tid; i < (A.k >> 2); i += nth) {
            float4 v = x4[i] * inv4;                    // same (x*inv)*w order as k_rms
            y4[i] = w ? (v * w4[i]) : v;
        }
    } else if (write_y) {
        for (uint i = tid; i < A.k; i += nth) y[i] = w ? (x[i] * inv * w[i]) : (x[i] * inv);
    }
    if (A.q8) {
        // normalize the SAME value the y-branch would store; recompute per
        // element (register-live inv) and quantize like k_quant: one thread
        // per 32-block, amax -> d (f16-stored) -> rint codes.
        const uint nbk = A.k / 32u;
        const uint qcur = A.qoff + b * A.q_stride + (r * A.k) / 4u;
        const uint dcur = A.doff + b * A.d_stride + r * nbk;
        for (uint blk = tid; blk < nbk; blk += nth) {
            const device float * xb = x + blk * 32u;
            float amax = 0.0f;
            float xs[32];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float v = w ? (xb[i] * inv * w[blk * 32u + i]) : (xb[i] * inv);
                xs[i] = v;
                float a = v < 0.0f ? -v : v;
                amax = a > amax ? a : amax;
            }
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            SCR[dcur + blk] = f16b(as_type<unsigned short>(half(d)));
            device uchar * q = (device uchar *)(SCR + qcur) + blk * 32u;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int rr = int(rint(xs[i] * id));
                rr = rr > 127 ? 127 : (rr < -127 ? -127 : rr);
                q[i] = uchar(char(rr));
            }
        }
    }
}

// ---------------- batched head clip ----------------
// grid = (groups, B) threadgroups of 128; identical tree reduction / clamp
// math to k_clip, per token via c_stride, in place.
kernel void k_clip_batch(device float * SCR [[buffer(0)]],
                         constant ClipBatchArgs & A [[buffer(1)]],
                         const device uchar * PACK [[buffer(2)]],
                         uint3 M64_TGPG [[threadgroup_position_in_grid]],
                         uint3 M64_TITG [[thread_position_in_threadgroup]],
                         uint3 M64_NPTG [[threads_per_threadgroup]]) {
    const uint g = M64_TGPG.x, b = M64_TGPG.y;
    if (g >= A.groups || b >= A.B) return;
    const uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    device float * base = SCR + A.coff + b * A.c_stride + g * A.per;
    const device float * sc = A.soff != M64_NOFF
        ? ((const device float *)(PACK + A.soff)) + g * A.per : nullptr;
    threadgroup float red[128];
    float s = 0.0;
    for (uint i = tid; i < A.per; i += nth) {
        float v = sc ? base[i] * sc[i] : base[i];
        base[i] = v; s += (float)v * (float)v;
    }
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = nth / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float norm = sqrt((float)red[0]) + 1e-6f;
        float f = A.thresh / norm;
        f = fmin(fmax(f, 0.0f), 1.0f);
        red[0] = (float)f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float f = (float)red[0];
    for (uint i = tid; i < A.per; i += nth) base[i] *= f;
}

// ---------------- batched gate * sigmoid(gate) ----------------
// dispatchThreads (ceil? no: (n, B)); identical per-element math to k_gate_mul.
kernel void k_gate_mul_batch(device float * SCR [[buffer(0)]],
                             constant GateMulBatchArgs & A [[buffer(1)]],
                             uint3 M64_TIG [[thread_position_in_grid]]) {
    const uint i = M64_TIG.x, b = M64_TIG.y;
    if (i >= A.n || b >= A.B) return;
    SCR[A.aoff + b * A.a_stride + i] *= ksig(SCR[A.goff + b * A.g_stride + i]);
}

// ---------------- batched SITU activation + Q8_0 quant ----------------
// dispatchThreads (nb, npairs): thread (j, p) is one 32-block of one
// (token, expert) pair; identical math to k_situ_quant.
kernel void k_situ_quant_batch(device float * SCR [[buffer(0)]],
                               constant SituBatchArgs & A [[buffer(1)]],
                               uint3 M64_TIG [[thread_position_in_grid]]) {
    const uint j = M64_TIG.x, p = M64_TIG.y;
    if (j >= A.nb || p >= A.npairs) return;
    const device float * g = SCR + A.goff + p * A.g_stride + j * 32u;
    const device float * u = SCR + A.uoff + p * A.u_stride + j * 32u;
    float hs[32], amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float x = g[i];
        float act = A.beta * tanh(x * A.inv_beta);
        act = act * (1.0f / (1.0f + metal::exp(-x)));
        float ulin = A.lb * tanh(u[i] * A.inv_lb);
        float v = act * ulin; hs[i] = v;
        float a = v < 0.0f ? -v : v; amax = a > amax ? a : amax;
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    SCR[A.doff + p * A.d_stride + j] = f16b(as_type<unsigned short>(half(d)));
    device uchar * q = (device uchar *)(SCR + A.qoff + (size_t)p * A.q_stride) + j * 32u;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int r = int(rint(hs[i] * id));
        r = r > 127 ? 127 : (r < -127 ? -127 : r);
        q[i] = uchar(char(r));
    }
}

// ---------------- batched MoE down reduce ----------------
// dispatchThreads (ceil(nout/1), B) style flat: out[b][c] =
// sum_{e<topk} w[b][e] * in[b*topk + e][c], e ascending, plain mul+add —
// the exact reduce math of k_down_reduce8_conv_add / k_gemv_down_reduce8.
kernel void k_moe_down_reduce_batch(device float * SCR [[buffer(0)]],
                                    constant MoeReduceBatchArgs & A [[buffer(1)]],
                                    uint3 M64_TIG [[thread_position_in_grid]]) {
    const uint c = M64_TIG.x, b = M64_TIG.y;
    if (c >= A.nout || b >= A.B) return;
    const device float * in = SCR + A.inoff + (size_t)b * A.topk * A.in_stride + c;
    const device float * w = SCR + A.wtoff + b * A.w_stride;
    float xin = 0.0f;
    #pragma unroll 8
    for (uint e = 0u; e < A.topk; e++)
        xin = xin + w[e] * in[(size_t)e * A.in_stride];
    SCR[A.outoff + (size_t)b * A.out_stride + c] = xin;
}

// ---------------- batched router top-8 ----------------
// grid = (B, 1) threadgroups of 128; per-token algorithm identical to
// k_router_topk (4-warp simd_max + simd_min tie -> smallest index,
// renormalized weights * scale).
kernel void k_router_topk_batch(device float * SCR [[buffer(0)]],
                                constant RouterBatchArgs & A [[buffer(1)]],
                                const device uchar * PACK [[buffer(2)]],
                                uint3 M64_TITG [[thread_position_in_threadgroup]],
                                uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint b = M64_TGPG.x;
    if (b >= A.B) return;
    const uint e = M64_TITG.x;       // 0..127 expert id
    const uint lane = e & 31u;
    threadgroup float pp[128], sp[128];
    threadgroup float warp_v[4];
    threadgroup uint warp_i[4];
    float p = ksig(SCR[A.probs + b * A.p_stride + e]);
    pp[e] = p;
    sp[e] = p + ((const device float *)(PACK + A.biaspack))[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sel[8]; float ps[8];
    float wsum = 0.0f;
    for (uint t = 0; t < 8u; t++) {
        float val = sp[e];
        float wv = simd_max(val);
        uint wi = simd_min(wv == val ? e : 128u);
        if (lane == 0u) { warp_v[e >> 5u] = wv; warp_i[e >> 5u] = wi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0u) {
            float bv = warp_v[0]; uint bi = warp_i[0];
            #pragma unroll
            for (uint q = 1u; q < 4u; q++)
                if (warp_v[q] > bv || (warp_v[q] == bv && warp_i[q] < bi)) { bv = warp_v[q]; bi = warp_i[q]; }
            sel[t] = bi; ps[t] = pp[bi];
            wsum += pp[bi];
            sp[bi] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e == 0u) {
        float div = wsum + 1e-20f;
        device int * seld = (device int *)(SCR + A.seloff + b * A.sel_stride);
        for (uint t = 0; t < 8u; t++) {
            seld[t] = (int)sel[t];
            SCR[A.wtoff + b * A.w_stride + t] = (ps[t] / div) * A.scale;
        }
    }
}

// ---------------------------------------------------------------------------
// k_gemv_f32_batch: batched FP32 GEMV (router: y = router.weight @ x_b, no
// quantization). One threadgroup per (row, token). Reduction is byte-for-byte
// the k_gemv_f32_tg structure (float4 dot when aligned + 2-level simd reduce),
// so results are bitwise-identical to running k_gemv_f32_tg once per token.
// Grid (nout, B) threadgroups x 256 threads.
kernel void k_gemv_f32_batch(device float * SCR [[buffer(0)]],
                             constant GemvF32BatchArgs & A [[buffer(1)]],
                             const device uchar * PACK [[buffer(2)]],
                             uint3 M64_TGPG [[threadgroup_position_in_grid]],
                             uint3 M64_TITG [[thread_position_in_threadgroup]],
                             uint3 M64_NPTG [[threads_per_threadgroup]]) {
    uint row = M64_TGPG.x;
    uint b = M64_TGPG.y;
    if (row >= A.nout || b >= A.B) return;
    const device float * w = ((const device float *)(PACK + A.woff)) + (size_t)row * A.k;
    const uint xoff = A.xoff + b * A.x_stride;
    const device float * x = SCR + xoff;
    uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    threadgroup float red[256];
    float a = 0.0f;
    if ((A.k & 3u) == 0u && (xoff & 3u) == 0u && (A.woff & 15u) == 0ul) {
        const device float4 * w4 = (const device float4 *)w;
        const device float4 * x4 = (const device float4 *)x;
        const uint kq = A.k >> 2;
        float4 av = float4(0.0f);
        for (uint i = tid; i < kq; i += nth) av += w4[i] * x4[i];
        a = (av.x + av.y) + (av.z + av.w);
    } else {
        for (uint i = tid; i < A.k; i += nth) a = fma(w[i], x[i], a);
    }
    a = simd_sum(a);
    uint lane0 = tid & 31u;
    if (lane0 == 0u) red[tid >> 5u] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float t = (lane0 < (nth >> 5u)) ? red[lane0] : 0.0f;
        t = simd_sum(t);
        if (tid == 0u) SCR[A.yoff + (size_t)b * A.y_stride + row] = t;
    }
}

// ---------------------------------------------------------------------------
// k_moe_bucket_build: build the CSR (counts/offsets/list) consumed by
// k_moe_grouped_gemm_q8 / k_moe_grouped_f32in_sg from sel[] (pair -> expert).
// Single threadgroup of 256; n_experts <= 128; arbitrary npairs = B*topk (e.g.
// up to 1024 at B=128). counts/offsets are written for ALL n_experts (zeroed for
// untouched experts); list entries are pair ids in nondeterministic atomic order.
kernel void k_moe_bucket_build(device float * SCR [[buffer(0)]],
                               constant MoeBucketArgs & A [[buffer(1)]],
                               const device uchar * PACK [[buffer(2)]],
                               uint3 M64_TGPG [[threadgroup_position_in_grid]],
                               uint3 M64_TITG [[thread_position_in_threadgroup]],
                               uint3 M64_NPTG [[threads_per_threadgroup]]) {
    (void)M64_TGPG; (void)PACK;
    if (A.n_experts > 128u) return;
    uint t = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    threadgroup uint cnt[128];
    threadgroup uint off[128];
    threadgroup uint cur[128];
    if (t < A.n_experts) { cnt[t] = 0u; off[t] = 0u; cur[t] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uint * selu = (const device uint *)(SCR + A.sel_off);
    for (uint p = t; p < A.npairs; p += nth) {
        uint e = selu[p];
        if (e < A.n_experts)
            atomic_fetch_add_explicit((threadgroup atomic_uint *)&cnt[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
        uint acc = 0u;
        for (uint e = 0u; e < A.n_experts; e++) { off[e] = acc; acc += cnt[e]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint * counts = (device uint *)(SCR + A.count_off);
    device uint * offsets = (device uint *)(SCR + A.offset_off);
    device uint * list = (device uint *)(SCR + A.list_off);
    if (t < A.n_experts) { counts[t] = cnt[t]; offsets[t] = off[t]; cur[t] = off[t]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = t; p < A.npairs; p += nth) {
        uint e = selu[p];
        if (e < A.n_experts) {
            uint i = atomic_fetch_add_explicit((threadgroup atomic_uint *)&cur[e], 1u, memory_order_relaxed);
            list[i] = p;
        }
    }
}

// ============================================================================
// Milestone 3C: tensor-core grouped MoE + FP32 SITU
// ============================================================================

// Pure FP32 batched SITU activation over npairs (token, expert) rows (matches
// pre-quant math of k_situ_quant_batch bit-for-bit). dispatchThreads (k, npairs)
// with threadsPerThreadgroup (128, 1, 1).
kernel void k_situ_f32_batch(device float * SCR [[buffer(0)]],
                             constant SituF32BatchArgs & A [[buffer(1)]],
                             uint3 M64_TIG [[thread_position_in_grid]]) {
    const uint i = M64_TIG.x, p = M64_TIG.y;
    if (i >= A.k || p >= A.npairs) return;
    const float x = SCR[A.goff + (size_t)p * A.g_stride + i];
    const float u = SCR[A.uoff + (size_t)p * A.u_stride + i];
    float act = A.beta * tanh(x * A.inv_beta);
    act = act * (1.0f / (1.0f + metal::exp(-x)));
    const float ulin = A.lb * tanh(u * A.inv_lb);
    SCR[A.hoff + (size_t)p * A.h_stride + i] = act * ulin;
}

// Grouped (per-expert) tensor-core Q8_0 GEMM taking FP32 activations directly.
// grid = (ceil(nout/32), n_experts, 1), threadgroup = (32, 4, 1) = 4 warps.
// Processes each active expert's pair list in 32-token tiles using
// dequant_q8_0_tile32x32_to_tg (reusable for future Q4 quant types) +
// simdgroup_half8x8 x simdgroup_half8x8 -> simdgroup_float8x8.
kernel void k_moe_grouped_f32in_sg(device float * SCR [[buffer(0)]],
                                   constant MoeGroupedF32SgArgs & A [[buffer(1)]],
                                   const device uchar * PACK [[buffer(2)]],
                                   uint3 M64_TITG [[thread_position_in_threadgroup]],
                                   uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint m0 = M64_TGPG.x * 32u;
    const uint exp_id = M64_TGPG.y;
    if (m0 >= A.nout || exp_id >= A.n_experts) return;     // threadgroup-uniform
    const device uint * counts = (const device uint *)(SCR + A.count_off);
    const uint cnt = counts[exp_id];
    if (cnt == 0u) return;                                 // threadgroup-uniform
    const uint base = ((const device uint *)(SCR + A.offset_off))[exp_id];
    const device uint * list = (const device uint *)(SCR + A.list_off);

    const uint tid = M64_TITG.y * 32u + M64_TITG.x;        // 0..127
    const uint sg_row = ((tid >> 5u) >> 1u) * 16u;
    const uint sg_col = ((tid >> 5u) & 1u) * 16u;
    const uint bcol = tid >> 2u, kseg = tid & 3u;

    const uint slot = A.wtab_off != 0u
        ? ((const device uint *)SCR)[A.wtab_off + exp_id]
        : exp_id;
    const device uchar * wmat = PACK + (size_t)A.wbase + (size_t)slot * A.exp_stride +
                                (size_t)A.part_off;
    const uint nb = A.k / 32u;

    threadgroup half tgW[32u * M64_SG_LD];
    threadgroup half tgX[32u * M64_SG_LD];
    threadgroup float tgO[32u * M64_SG_LD];
    threadgroup uint tgIn[32];
    threadgroup uint tgOutP[32];

    for (uint off = 0u; off < cnt; off += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {
            const bool valid = (off + tid) < cnt;
            const uint p = valid ? list[base + off + tid] : 0u;
            tgIn[tid] = A.is_down ? p : (p / A.topk);
            tgOutP[tid] = valid ? p : 0xFFFFFFFFu;
        }
        simdgroup_float8x8 acc[2][2];
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int c = 0; c < 2; c++)
                acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint blk = 0u; blk < nb; blk++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dequant_q8_0_tile32x32_to_tg(wmat, m0, A.nout, blk, nb, tid, tgW, M64_SG_LD);
            {
                const uint p_out = tgOutP[bcol];
                threadgroup half * dx = tgX + kseg * 8u * M64_SG_LD + bcol;
                if (p_out != 0xFFFFFFFFu) {
                    const device float * xb = SCR + A.xoff +
                        (size_t)tgIn[bcol] * A.x_stride + (size_t)blk * 32u + kseg * 8u;
                    #pragma unroll
                    for (uint i = 0; i < 8u; i++) dx[i * M64_SG_LD] = half(xb[i]);
                } else {
                    #pragma unroll
                    for (uint i = 0; i < 8u; i++) dx[i * M64_SG_LD] = half(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 a0, a1, b0m, b1m;
                simdgroup_load(a0, tgW + (sg_row + 0u) * M64_SG_LD + kk, M64_SG_LD);
                simdgroup_load(a1, tgW + (sg_row + 8u) * M64_SG_LD + kk, M64_SG_LD);
                simdgroup_load(b0m, tgX + kk * M64_SG_LD + sg_col, M64_SG_LD);
                simdgroup_load(b1m, tgX + kk * M64_SG_LD + sg_col + 8u, M64_SG_LD);
                simdgroup_multiply_accumulate(acc[0][0], a0, b0m, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], a0, b1m, acc[0][1]);
                simdgroup_multiply_accumulate(acc[1][0], a1, b0m, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], a1, b1m, acc[1][1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int c = 0; c < 2; c++)
                simdgroup_store(acc[r][c], tgO + (sg_row + (uint)r * 8u) * M64_SG_LD +
                                         sg_col + (uint)c * 8u, M64_SG_LD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint m = m0 + (tid >> 2u);
        if (m < A.nout) {
            const uint cbase = (tid & 3u) * 8u;
            const threadgroup float * src = tgO + (tid >> 2u) * M64_SG_LD + cbase;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                const uint outp = tgOutP[cbase + i];
                if (outp != 0xFFFFFFFFu)
                    SCR[A.yoff + (size_t)outp * A.y_stride + m] = src[i];
            }
        }
    }
}

// ============================================================================
// Milestone 4B: tensor-core grouped MoE with Q4_K / Q5_0 / Q5_1 weights
// ============================================================================
// Same tile pipeline as k_moe_grouped_f32in_sg; ONLY the weight loader swaps.
// Bit layouts byte-for-byte per engine/src/dequant.hpp refs (see m64_p2a.metal
// Milestone 4B header for the derivation). All weight loads are byte-wise
// (m64_ld16 / m64_ld32) so Q5_0's 22 B blocks need no alignment at all.

// Q4_K: one 32-col tile = one sub-block (blk & 7) of block (blk >> 3).
// row bytes = nb/8 * 144 == nb * 18 (nb = k/32, k % 256 == 0).
static inline void dequant_q4_k_tile32x32_to_tg(const device uchar * wmat,
                                                uint m0, uint nout, uint blk,
                                                uint nb, uint tid,
                                                threadgroup half * tg, uint ld) {
    const uint row = tid >> 2u, seg = tid & 3u;
    const uint b = blk >> 3u, is = blk & 7u;
    const size_t rowb = (size_t)nb * 18ull;
    threadgroup half * dst = tg + row * ld + seg * 8u;
    if (m0 + row < nout) {
        const device uchar * bp = wmat + (size_t)(m0 + row) * rowb + (size_t)b * 144ull;
        const uint q = m64_ld32(bp + 4u), q2 = m64_ld32(bp + 8u), q3 = m64_ld32(bp + 12u);
        float sd, sm;
        m64_scale_min_k4(is, q, q2, q3, sd, sm);
        const float vd = f16b(m64_ld16(bp)) * sd;
        const float vm = f16b(m64_ld16(bp + 2u)) * sm;
        const device uchar * qb = bp + 16u + (is >> 1u) * 32u + seg * 8u;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            const uint byte = qb[i];
            const uint n = (is & 1u) ? (byte >> 4) : (byte & 0xFu);
            dst[i] = half(vd * float(n) - vm);
        }
    } else {
        #pragma unroll
        for (uint i = 0; i < 8u; i++) dst[i] = half(0.0f);
    }
}

// Q5_0: one 32-col tile = one 22 B block. row bytes = nb * 22.
// elem j (0..15): (qs[j] & 0xF) | bit j of qh at 0x10, minus 16, times d;
// elem 16+j: (qs[j] >> 4) | bit (j + 12) of qh at 0x10 (dequant.hpp:117-124).
static inline void dequant_q5_0_tile32x32_to_tg(const device uchar * wmat,
                                                uint m0, uint nout, uint blk,
                                                uint nb, uint tid,
                                                threadgroup half * tg, uint ld) {
    const uint row = tid >> 2u, seg = tid & 3u;
    const size_t rowb = (size_t)nb * 22ull;
    threadgroup half * dst = tg + row * ld + seg * 8u;
    if (m0 + row < nout) {
        const device uchar * bp = wmat + (size_t)(m0 + row) * rowb + (size_t)blk * 22ull;
        const float d = f16b(m64_ld16(bp));
        const uint qh = m64_ld32(bp + 2u);
        const device uchar * qs = bp + 6u;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            const uint l = seg * 8u + i, j = l & 15u;
            const uint lo = (l < 16u);
            const uint byte = qs[j];
            const uint nib = lo ? (byte & 0xFu) : (byte >> 4);
            const uint bit = lo ? (((qh >> j) << 4) & 0x10u)
                                : ((qh >> (j + 12u)) & 0x10u);
            dst[i] = half(d * (float(nib | bit) - 16.0f));   // signed: 16u would underflow to inf
        }
    } else {
        #pragma unroll
        for (uint i = 0; i < 8u; i++) dst[i] = half(0.0f);
    }
}

// Q5_1: 24 B block {d f16 @0, m f16 @2, qh @4, qs[16] @8}, y = d*x + m.
static inline void dequant_q5_1_tile32x32_to_tg(const device uchar * wmat,
                                                uint m0, uint nout, uint blk,
                                                uint nb, uint tid,
                                                threadgroup half * tg, uint ld) {
    const uint row = tid >> 2u, seg = tid & 3u;
    const size_t rowb = (size_t)nb * 24ull;
    threadgroup half * dst = tg + row * ld + seg * 8u;
    if (m0 + row < nout) {
        const device uchar * bp = wmat + (size_t)(m0 + row) * rowb + (size_t)blk * 24ull;
        const float d = f16b(m64_ld16(bp));
        const float mm = f16b(m64_ld16(bp + 2u));
        const uint qh = m64_ld32(bp + 4u);
        const device uchar * qs = bp + 8u;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            const uint l = seg * 8u + i, j = l & 15u;
            const uint lo = (l < 16u);
            const uint byte = qs[j];
            const uint nib = lo ? (byte & 0xFu) : (byte >> 4);
            const uint bit = lo ? (((qh >> j) << 4) & 0x10u)
                                : ((qh >> (j + 12u)) & 0x10u);
            dst[i] = half(d * float(nib | bit) + mm);
        }
    } else {
        #pragma unroll
        for (uint i = 0; i < 8u; i++) dst[i] = half(0.0f);
    }
}

// QT: 1 = Q4_K, 2 = Q5_0, 3 = Q5_1. Identical control flow to
// k_moe_grouped_f32in_sg; only the weight loader call switches (dead branches
// fold at compile time since QT is a template constant). MSL forbids
// threadgroup-declared arrays in non-kernel functions, so the buffers are
// declared by each kernel wrapper and passed in.
template <int QT>
static inline void m64_moe_grouped_sg_q_impl(
        device float * SCR, constant MoeGroupedF32SgArgs & A,
        const device uchar * PACK, uint3 TITG, uint3 TGPG,
        threadgroup half * tgW, threadgroup half * tgX, threadgroup float * tgO,
        threadgroup uint * tgIn, threadgroup uint * tgOutP) {
    const uint m0 = TGPG.x * 32u;
    const uint exp_id = TGPG.y;
    if (m0 >= A.nout || exp_id >= A.n_experts) return;     // threadgroup-uniform
    const device uint * counts = (const device uint *)(SCR + A.count_off);
    const uint cnt = counts[exp_id];
    if (cnt == 0u) return;                                 // threadgroup-uniform
    const uint base = ((const device uint *)(SCR + A.offset_off))[exp_id];
    const device uint * list = (const device uint *)(SCR + A.list_off);

    const uint tid = TITG.y * 32u + TITG.x;                // 0..127
    const uint sg_row = ((tid >> 5u) >> 1u) * 16u;
    const uint sg_col = ((tid >> 5u) & 1u) * 16u;
    const uint bcol = tid >> 2u, kseg = tid & 3u;

    const uint slot = A.wtab_off != 0u
        ? ((const device uint *)SCR)[A.wtab_off + exp_id]
        : exp_id;
    const device uchar * wmat = PACK + (size_t)A.wbase + (size_t)slot * A.exp_stride +
                                (size_t)A.part_off;
    const uint nb = A.k / 32u;

    for (uint off = 0u; off < cnt; off += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {
            const bool valid = (off + tid) < cnt;
            const uint p = valid ? list[base + off + tid] : 0u;
            tgIn[tid] = A.is_down ? p : (p / A.topk);
            tgOutP[tid] = valid ? p : 0xFFFFFFFFu;
        }
        simdgroup_float8x8 acc[2][2];
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int c = 0; c < 2; c++)
                acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint blk = 0u; blk < nb; blk++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (QT == 1)
                dequant_q4_k_tile32x32_to_tg(wmat, m0, A.nout, blk, nb, tid, tgW, M64_SG_LD);
            else if (QT == 2)
                dequant_q5_0_tile32x32_to_tg(wmat, m0, A.nout, blk, nb, tid, tgW, M64_SG_LD);
            else
                dequant_q5_1_tile32x32_to_tg(wmat, m0, A.nout, blk, nb, tid, tgW, M64_SG_LD);
            {
                const uint p_out = tgOutP[bcol];
                threadgroup half * dx = tgX + kseg * 8u * M64_SG_LD + bcol;
                if (p_out != 0xFFFFFFFFu) {
                    const device float * xb = SCR + A.xoff +
                        (size_t)tgIn[bcol] * A.x_stride + (size_t)blk * 32u + kseg * 8u;
                    #pragma unroll
                    for (uint i = 0; i < 8u; i++) dx[i * M64_SG_LD] = half(xb[i]);
                } else {
                    #pragma unroll
                    for (uint i = 0; i < 8u; i++) dx[i * M64_SG_LD] = half(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 a0, a1, b0m, b1m;
                simdgroup_load(a0, tgW + (sg_row + 0u) * M64_SG_LD + kk, M64_SG_LD);
                simdgroup_load(a1, tgW + (sg_row + 8u) * M64_SG_LD + kk, M64_SG_LD);
                simdgroup_load(b0m, tgX + kk * M64_SG_LD + sg_col, M64_SG_LD);
                simdgroup_load(b1m, tgX + kk * M64_SG_LD + sg_col + 8u, M64_SG_LD);
                simdgroup_multiply_accumulate(acc[0][0], a0, b0m, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], a0, b1m, acc[0][1]);
                simdgroup_multiply_accumulate(acc[1][0], a1, b0m, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], a1, b1m, acc[1][1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int c = 0; c < 2; c++)
                simdgroup_store(acc[r][c], tgO + (sg_row + (uint)r * 8u) * M64_SG_LD +
                                         sg_col + (uint)c * 8u, M64_SG_LD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint m = m0 + (tid >> 2u);
        if (m < A.nout) {
            const uint cbase = (tid & 3u) * 8u;
            const threadgroup float * src = tgO + (tid >> 2u) * M64_SG_LD + cbase;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                const uint outp = tgOutP[cbase + i];
                if (outp != 0xFFFFFFFFu)
                    SCR[A.yoff + (size_t)outp * A.y_stride + m] = src[i];
            }
        }
    }
}

// grid = (ceil(nout/32), n_experts), threadgroup = (32, 4); Q4_K needs
// k % 256 == 0, Q5_* need k % 32 == 0 (same as the Q8_0 kernel).
kernel void k_moe_grouped_f32in_sg_q4k(device float * SCR [[buffer(0)]],
                                       constant MoeGroupedF32SgArgs & A [[buffer(1)]],
                                       const device uchar * PACK [[buffer(2)]],
                                       uint3 M64_TITG [[thread_position_in_threadgroup]],
                                       uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    threadgroup half tgW[32u * M64_SG_LD];
    threadgroup half tgX[32u * M64_SG_LD];
    threadgroup float tgO[32u * M64_SG_LD];
    threadgroup uint tgIn[32];
    threadgroup uint tgOutP[32];
    m64_moe_grouped_sg_q_impl<1>(SCR, A, PACK, M64_TITG, M64_TGPG,
                                    tgW, tgX, tgO, tgIn, tgOutP);
}

kernel void k_moe_grouped_f32in_sg_q5_0(device float * SCR [[buffer(0)]],
                                        constant MoeGroupedF32SgArgs & A [[buffer(1)]],
                                        const device uchar * PACK [[buffer(2)]],
                                        uint3 M64_TITG [[thread_position_in_threadgroup]],
                                        uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    threadgroup half tgW[32u * M64_SG_LD];
    threadgroup half tgX[32u * M64_SG_LD];
    threadgroup float tgO[32u * M64_SG_LD];
    threadgroup uint tgIn[32];
    threadgroup uint tgOutP[32];
    m64_moe_grouped_sg_q_impl<2>(SCR, A, PACK, M64_TITG, M64_TGPG,
                                    tgW, tgX, tgO, tgIn, tgOutP);
}

kernel void k_moe_grouped_f32in_sg_q5_1(device float * SCR [[buffer(0)]],
                                        constant MoeGroupedF32SgArgs & A [[buffer(1)]],
                                        const device uchar * PACK [[buffer(2)]],
                                        uint3 M64_TITG [[thread_position_in_threadgroup]],
                                        uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    threadgroup half tgW[32u * M64_SG_LD];
    threadgroup half tgX[32u * M64_SG_LD];
    threadgroup float tgO[32u * M64_SG_LD];
    threadgroup uint tgIn[32];
    threadgroup uint tgOutP[32];
    m64_moe_grouped_sg_q_impl<3>(SCR, A, PACK, M64_TITG, M64_TGPG,
                                    tgW, tgX, tgO, tgIn, tgOutP);
}
