// m64_p2a.metal — Phase 2A native GPU forward kernels for inkling2_loop.
// One command buffer + one compute encoder per token; dispatches are serial.
// Numerics: q8_0 GEMV reproduces the C++ oracle emulation int-exactly per
// 32-block with ascending-block fmaf accumulation; cross-lane tree reductions
// (rmsnorm, softmax, split dots) are tolerance-class (~1e-7 rel), far below
// the 1e-3 parity gate. m64_p2a_args.h is force-included via headerFileNames.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
static_assert(sizeof(unsigned long) == 8, "msl long64");
static_assert(sizeof(RMSArgs) == 784, "RMSArgs");
static_assert(sizeof(GemvArgs) == 48, "GemvArgs");
static_assert(sizeof(MoeDownArgs) == 40, "MoeDownArgs");
static_assert(sizeof(EmbArgs) == 16, "EmbArgs");
static_assert(sizeof(AttnArgs) == 64, "AttnArgs");
static_assert(sizeof(KVWriteF16Args) == 32, "KVWriteF16Args");
static_assert(sizeof(AttnF16Args) == 64, "AttnF16Args");
static_assert(sizeof(AttnF16SplitArgs) == 80, "AttnF16SplitArgs");
static_assert(sizeof(ConvArgs) == 48, "ConvArgs");
static_assert(sizeof(CtrlArgs) == 192, "CtrlArgs");
static_assert(sizeof(CtrlSplitArgs) == 168, "CtrlSplitArgs");
static_assert(sizeof(RmsQuantArgs) == 1168, "RmsQuantArgs");
static_assert(sizeof(GateMulQuantArgs) == 20, "GateMulQuantArgs");
static_assert(sizeof(MultiGemvArgs) == 136, "MultiGemvArgs");
static_assert(sizeof(DownReduceConvArgs) == 32, "DownReduceConvArgs");
static_assert(sizeof(MixFusedArgs) == 32, "MixFusedArgs");
static_assert(sizeof(MoeGateUpQ4KF32Args) == 48, "MoeGateUpQ4KF32Args");
static_assert(sizeof(MoeDownQ5F32Args) == 48, "MoeDownQ5F32Args");
static_assert(sizeof(QkvPrepKvArgs) == 64, "QkvPrepKvArgs");
static_assert(sizeof(PostConvResArgs) == 40, "PostConvResArgs");

#define M64_NE    2304u
#define M64_EXT   1024u


static inline float f16b(unsigned short u) { return float(as_type<half>(u)); }
static inline float f16r(float x) { return float(half(x)); }
static inline float ksig(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
static inline int s8(uchar b) { return int(as_type<char>(b)); }

// ------------------------------------------------------- embedding row gather
kernel void k_emb_gather(device float * SCR [[buffer(0)]],
                         constant EmbArgs & A [[buffer(1)]],
                         const device uchar * PACK [[buffer(2)]],
                         constant uint & tok [[buffer(3)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint b = M64_TIG.x;              // 72 blocks
    if (b >= A.n) return;
    const device uchar * wp = PACK + A.off + (size_t)tok * 2448u + b * 34u;
    const float d = f16b((uint(wp[1]) << 8) | wp[0]);
    device float * y = SCR + A.boff + b * 32u;
    #pragma unroll
    for (int i = 0; i < 32; i++) y[i] = d * float(s8(wp[2 + i]));
}

// ------------------------------------------------------------------ rmsnorm
kernel void k_rms(device float * SCR [[buffer(0)]],
                  constant RMSArgs & A [[buffer(1)]],
                  const device uchar * PACK [[buffer(2)]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_NPTG [[threads_per_threadgroup]]) {
    uint row = M64_TGPG.x;   // one threadgroup per row (dispatchThreadgroups)
    if (row >= A.rows) return;
    const device float * x = SCR + A.xr[row];
    const device float * w = A.hasw ? ((const device float *)(PACK + A.wr[row])) : nullptr;
    device float * y = SCR + A.yr[row];
    uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    threadgroup float red[256];
    // R4: float4 squared-sum accumulate. vec4_in depends ONLY on k and
    // xr[row], so k_rms and k_rms_quant ALWAYS take the identical reduction
    // branch (and thus get a bitwise-identical s) for identical inputs.
    const bool vec4_in = ((A.k & 3u) == 0u) && ((A.xr[row] & 3u) == 0u);
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
    // Round-2 SIMD reduction (256 threads = 8 warps): warp simd_sum -> 8 warp
    // sums -> warp-0 simd_sum. k_rms and k_rms_quant MUST keep this exact
    // code shape so their results stay bitwise-identical to each other.
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
    if (vec4_in && (!A.hasw || (A.wr[row] & 15ul) == 0ul) && (y != nullptr && (A.yr[row] & 3u) == 0u)) {
        const device float4 * x4 = (const device float4 *)x;
        const device float4 * w4 = w ? (const device float4 *)w : nullptr;
        device float4 * y4 = (device float4 *)y;
        const float4 inv4 = float4(inv);
        for (uint i = tid; i < (A.k >> 2); i += nth) {
            float4 v = x4[i] * inv4;                    // same (x*inv)*w order as scalar
            y4[i] = w ? (v * w4[i]) : v;
        }
    } else {
        for (uint i = tid; i < A.k; i += nth) y[i] = w ? (x[i] * inv * w[i]) : (x[i] * inv);
    }
}

// ----------------------------------------------------------------- quant q8_0
kernel void k_quant(device float * SCR [[buffer(0)]],
                    constant QuantArgs & A [[buffer(1)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint b = M64_TIG.x;
    if (b >= A.nb) return;
    const device float * xb = SCR + A.xoff + b * 32u;
    float amax = 0.0f;
    float xs[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float v = xb[i]; xs[i] = v;
        float a = v < 0.0f ? -v : v;
        amax = a > amax ? a : amax;
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    SCR[A.doff + b] = f16b(as_type<unsigned short>(half(d)));
    device uchar * q = (device uchar *)(SCR + A.qoff) + b * 32u;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int r = int(rint(xs[i] * id));
        r = r > 127 ? 127 : (r < -127 ? -127 : r);
        q[i] = uchar(char(r));
    }
}

// --- gemv core: unit = 2 q8_0 blocks (68 B row slice), int32-exact block dots
// scalar reference (kept for A/B benchmark)
static inline void gemv_unit_scalar(const device uint * wu, const device uchar4 * xu,
                                    const device float * dx, uint u, float & a) {
    uint W[17];
    #pragma unroll
    for (int j = 0; j < 17; j++) W[j] = wu[j];
    uchar4 V[17];
    #pragma unroll
    for (int j = 0; j < 17; j++) V[j] = as_type<uchar4>(W[j]);
    uchar4 XA[8], XB[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) XA[j] = xu[j];
    #pragma unroll
    for (int j = 0; j < 8; j++) XB[j] = xu[8 + j];
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        s0 += s8(V[(k + 2) >> 2][(k + 2) & 3]) * s8(XA[k >> 2][k & 3]);
        s1 += s8(V[(k + 36) >> 2][(k + 36) & 3]) * s8(XB[k >> 2][k & 3]);
    }
    a = fma(float(s0), dx[2 * u] * f16b((unsigned short)(W[0] & 0xFFFFu)), a);
    a = fma(float(s1), dx[2 * u + 1] * f16b((unsigned short)(W[8] >> 16)), a);
}

// MSL has no dot() overload for integer vectors; char products promote to
// int so the sum is exact (identical to the hardware DP4A result).
static inline int dot_c4(char4 a, char4 b) {
    return (int)a.x * (int)b.x + (int)a.y * (int)b.y +
           (int)a.z * (int)b.z + (int)a.w * (int)b.w;
}
static inline int dot_c4u(char4 a, uchar4 b) {
    char4 c = as_type<char4>(b);
    return (int)a.x * (int)c.x + (int)a.y * (int)c.y +
           (int)a.z * (int)c.z + (int)a.w * (int)c.w;
}

// vectorized unit: hardware char4 dots. Block-0 weights live at bytes 2..33
// (misaligned): rebuild 8 aligned words (W[j]>>16)|(W[j+1]<<16). Block-1
// weights are bytes 36..67, already 4B aligned at W[9+j]. Integer dot sums
// are exact, so s0/s1 are bitwise identical to the scalar loop above.
static inline void gemv_unit(const device uint * wu, const device uchar4 * xu,
                             const device float * dx, uint u, float & a) {
    uint W[17];
    #pragma unroll
    for (int j = 0; j < 17; j++) W[j] = wu[j];
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint v0 = (W[j] >> 16) | (W[j + 1] << 16);
        s0 += dot_c4(as_type<char4>(v0), as_type<char4>(xu[j]));
        s1 += dot_c4(as_type<char4>(W[9 + j]), as_type<char4>(xu[8 + j]));
    }
    a = fma(float(s0), dx[2 * u] * f16b((unsigned short)(W[0] & 0xFFFFu)), a);
    a = fma(float(s1), dx[2 * u + 1] * f16b((unsigned short)(W[8] >> 16)), a);
}

// warp-per-row, 36 units (k = 2304)
kernel void k_gemv_q8_k2304(device float * SCR [[buffer(0)]],
                            constant GemvArgs & A [[buffer(1)]],
                            const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint row = M64_TGPG.x * 8u + (tid >> 5);
    if (row >= A.nout) return;
    uint lane = tid & 31u;
    const device uint * wr = (const device uint *)(PACK + A.woff + (size_t)A.idx * A.stride) + row * 612u;
    float a = 0.0f;
    for (uint u = lane; u < 36u; u += 32u)
        gemv_unit(wr + u * 17u, (const device uchar4 *)(SCR + A.xqoff) + u * 16u,
                  SCR + A.dxoff, u, a);
    a = simd_sum(a);
    if (lane == 0) SCR[A.yoff + row] = a;
}

// warp-per-2-rows, 18 units (k = 1152)
kernel void k_gemv_q8_k1152(device float * SCR [[buffer(0)]],
                            constant GemvArgs & A [[buffer(1)]],
                            const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint rbase = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (rbase >= A.nout) return;
    uint lane = tid & 31u;
    uint rb = lane & 1u;
    const device uint * wr = (const device uint *)(PACK + A.woff + (size_t)A.idx * A.stride) + (rbase + rb) * 306u;
    float fa = 0.0f, fb = 0.0f;
    float & a = rb ? fb : fa;
    for (uint u = (lane >> 1); u < 18u; u += 16u)
        gemv_unit(wr + u * 17u, (const device uchar4 *)(SCR + A.xqoff) + u * 16u,
                  SCR + A.dxoff, u, a);
    fa = simd_sum(fa); fb = simd_sum(fb);
    if (lane == 0) SCR[A.yoff + rbase] = fa;
    if (lane == 1 && rbase + 1u < A.nout) SCR[A.yoff + rbase + 1u] = fb;
}

// warp-per-row, 48 units (k = 3072)
kernel void k_gemv_q8_k3072(device float * SCR [[buffer(0)]],
                            constant GemvArgs & A [[buffer(1)]],
                            const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint row = M64_TGPG.x * 8u + (tid >> 5);
    if (row >= A.nout) return;
    uint lane = tid & 31u;
    const device uint * wr = (const device uint *)(PACK + A.woff + (size_t)A.idx * A.stride) + row * 816u;
    float a = 0.0f;
    for (uint u = lane; u < 48u; u += 32u)
        gemv_unit(wr + u * 17u, (const device uchar4 *)(SCR + A.xqoff) + u * 16u,
                  SCR + A.dxoff, u, a);
    a = simd_sum(a);
    if (lane == 0) SCR[A.yoff + row] = a;
}

// fused gate|up (two row-halves, same activation)
kernel void k_gemv_gateup(device float * SCR [[buffer(0)]],
                          constant GemvArgs & A [[buffer(1)]],
                          const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint row = M64_TGPG.x * 8u + (tid >> 5);
    if (row >= 2u * A.nout) return;
    uint lane = tid & 31u;
    bool up = row >= A.nout;
    uint r = up ? row - A.nout : row;
    // A.idx is the top-k SLOT; the expert blob comes from the router sel[] array
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[A.idx];
    const device uint * wr = (const device uint *)(PACK + (up ? A.woff2 : A.woff) + (size_t)exp_id * A.stride) + r * 612u;
    float a = 0.0f;
    for (uint u = lane; u < 36u; u += 32u)
        gemv_unit(wr + u * 17u, (const device uchar4 *)(SCR + A.xqoff) + u * 16u,
                  SCR + A.dxoff, u, a);
    a = simd_sum(a);
    if (lane == 0) SCR[(up ? A.yoff2 : A.yoff) + r] = a;
}

// fused down + weighted MoE accumulation: acc[o] = fma(w_i, dot, acc[o])
kernel void k_gemv_downacc(device float * SCR [[buffer(0)]],
                           constant MoeDownArgs & A [[buffer(1)]],
                           const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint rbase = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (rbase >= A.nout) return;
    uint lane = tid & 31u;
    uint rb = lane & 1u;
    const float w = SCR[A.wtoff + A.idx];
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[A.idx];
    const device uint * wr = (const device uint *)(PACK + A.woff + (size_t)exp_id * A.stride) + (rbase + rb) * 306u;
    float fa = 0.0f, fb = 0.0f;
    float & a = rb ? fb : fa;
    for (uint u = (lane >> 1); u < 18u; u += 16u)
        gemv_unit(wr + u * 17u, (const device uchar4 *)(SCR + A.xqoff) + u * 16u,
                  SCR + A.dxoff, u, a);
    fa = simd_sum(fa); fb = simd_sum(fb);
    // CPU oracle: acc = acc + w*dbuf (plain mul+add, contract off on both sides)
    if (lane == 0) SCR[A.accoff + rbase] = SCR[A.accoff + rbase] + w * fa;
    if (lane == 1 && rbase + 1u < A.nout)
        SCR[A.accoff + rbase + 1u] = SCR[A.accoff + rbase + 1u] + w * fb;
}

// ------------------------------------------------------------------ gemv f32
kernel void k_gemv_f32_tg(device float * SCR [[buffer(0)]],
                          constant GemvF32Args & A [[buffer(1)]],
                          const device uchar * PACK [[buffer(2)]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_NPTG [[threads_per_threadgroup]]) {
    uint row = M64_TGPG.x;   // one threadgroup per row (dispatchThreadgroups)
    if (row >= A.nout) return;
    const device float * w = ((const device float *)(PACK + A.woff)) + (size_t)row * A.k;
    const device float * x = SCR + A.xoff;
    uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    threadgroup float red[256];
    float a = 0.0f;
    // Round-2: float4-vectorized dot (production k = 2304 is /4; SCR float
    // offsets and PACK byte offsets must be 16B-aligned — scalar otherwise).
    if ((A.k & 3u) == 0u && (A.xoff & 3u) == 0u && (A.woff & 15u) == 0ul) {
        const device float4 * w4 = (const device float4 *)w;
        const device float4 * x4 = (const device float4 *)x;
        const uint kq = A.k >> 2;
        float4 av = float4(0.0f);
        for (uint i = tid; i < kq; i += nth) av += w4[i] * x4[i];
        a = (av.x + av.y) + (av.z + av.w);
    } else {
        for (uint i = tid; i < A.k; i += nth) a = fma(w[i], x[i], a);
    }
    // 2-level SIMD reduction (nth/32 warps -> warp-0 simd_sum). R4: warp
    // count derives from nth so 64/128/256-thread groups all reduce cleanly.
    a = simd_sum(a);
    uint lane0 = tid & 31u;
    if (lane0 == 0u) red[tid >> 5u] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float t = (lane0 < (nth >> 5u)) ? red[lane0] : 0.0f;
        t = simd_sum(t);
        if (tid == 0u) SCR[A.yoff + row] = t;
    }
}

// ---------------------------------------------------------------- conv slot
kernel void k_conv_slot(device float * SCR [[buffer(0)]],
                        constant ConvArgs & A [[buffer(1)]],
                        const device uchar * PACK [[buffer(2)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint c = M64_TIG.x;
    if (c >= A.C) return;
    device float * st = SCR + A.rsoff + c * 3u;
    const float s0 = st[0], s1 = st[1], s2 = st[2];
    const float xin = SCR[A.xoff + c];
    const device float * w = ((const device float *)(PACK + A.woffpack)) + c * 4u;
    float outv = s0 * w[0] + s1 * w[1] + s2 * w[2] + xin * w[3];
    if (A.hasdyn) {
        const device float * dyn = SCR + A.dynoff + (c / A.group) * 4u;
        outv += dyn[0] * xin + dyn[1] * s2 + dyn[2] * s1 + dyn[3] * s0;
    }
    SCR[A.yoff + c] = (xin + outv) * A.scale;
    st[0] = s1; st[1] = s2; st[2] = xin;
}

// ---------------------------------------------------------------- clip heads
kernel void k_clip(device float * SCR [[buffer(0)]],
                   constant ClipArgs & A [[buffer(1)]],
                   const device uchar * PACK [[buffer(2)]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_NPTG [[threads_per_threadgroup]]) {
    uint g = M64_TGPG.x;   // one threadgroup per clip group (dispatchThreadgroups)
    if (g >= A.groups) return;
    uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    device float * base = SCR + A.coff + g * A.per;
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

// ------------------------------------------------------------------- attention
// threadgroup = one query head (32 lanes); grid.x = NH. Relative-bias is fused
// (dot over 16 rel channels + sink subtract) to avoid per-slot bias tables.
kernel void k_attn_head(device float * SCR [[buffer(0)]],
                        constant AttnArgs & A [[buffer(1)]],
                        const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint h = M64_TGPG.x;
    uint lane = M64_TITG.x;
    const uint t = lane * 4u;
    const device float * q = SCR + A.qoff + h * 128u;
    half qh[4]; float qf[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { qf[i] = q[t + i]; qh[i] = half(q[t + i]); }
    const device float * rout = SCR + A.rout + h * 16u;
    const float sink = SCR[A.sinkoff + h];
    const device float * kb = SCR + A.kvkoff;
    const device float * vb = SCR + A.kvvoff;
    const uint kvoff = (h / 3u) * 128u;
    float oh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M = -INFINITY;
    float S = 0.0;
    for (uint j = 0; j < A.nkv; j++) {
        if (A.is_swa && (A.pos - j) > A.n_swa) continue;
        const device float * kj = kb + (size_t)j * 1024u + kvoff + t;
        float d = 0.0f;
        if (A.fa16) {
            #pragma unroll
            for (int i = 0; i < 4; i++) d = fma(float(qh[i]), f16r(kj[i]), d);
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) d = fma(qf[i], kj[i], d);
        }
        d = simd_sum(d);
        uint dist = A.pos - j;
        float bv = 0.0f;
        if (dist < M64_EXT) {
            const device float * brow = ((const device float *)(PACK + A.relbiaspack)) + (size_t)dist * 16u;
            // split dot over 16 channels across lanes 0..15 (else simd_sum = 32x dot)
            float v = (lane < 16u) ? (brow[lane] * rout[lane]) : 0.0f;
            bv = simd_sum(v);
        }
        float s = d * A.scale + (A.fa16 ? f16r(bv - sink) : (bv - sink));
        float ms = 1.0f, vs;
        if (s > M) {
            ms = metal::exp(M - s);
            oh[0] *= ms; oh[1] *= ms; oh[2] *= ms; oh[3] *= ms;
            M = s; vs = 1.0f;
        } else {
            vs = metal::exp(s - M);
        }
        const device float * vj = vb + (size_t)j * 1024u + kvoff + t;
        if (A.fa16) {
            #pragma unroll
            for (int i = 0; i < 4; i++) oh[i] = fma(vs, f16r(vj[i]), oh[i]);
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) oh[i] = fma(vs, vj[i], oh[i]);
        }
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
    device float * out = SCR + A.outoff + h * 128u;
    #pragma unroll
    for (int i = 0; i < 4; i++) out[t + i] = oh[i] * inv;
}

// ------------------------------------------- FP16 KV cache write (128K/SWA)
// Converts one token's fp32 K/V rows to FP16 into the shared half KV buffer.
// dst_k/dst_v are HALF-element offsets; the host pre-computes the ring slot.
kernel void k_kv_write_f16(const device float * SCR [[buffer(0)]],
                           constant KVWriteF16Args & A [[buffer(1)]],
                           device half * KV [[buffer(2)]],
                           uint3 M64_TIG [[thread_position_in_grid]]) {
    if (M64_TIG.x < A.n) {
        KV[A.dst_k + M64_TIG.x] = half(SCR[A.koff + M64_TIG.x]);
        KV[A.dst_v + M64_TIG.x] = half(SCR[A.voff + M64_TIG.x]);
    }
}

// ------------------------------------------------- attention over FP16 KV cache
// Numerics identical to k_attn_head with fa16=1 (1 threadgroup of 32 lanes per
// query head, fused relative-bias dot, online softmax, zero-sink term), reading
// K/V from the FP16 cache with SWA ring-slot indexing. tau folds in the
// long-context log-scale at the same points as forward_impl.hpp (Q *= tau_v on
// q BEFORE the half cast, Rr *= tau_v in float before the bias dot); at
// tau == 1.0f this is bitwise-equal to k_attn_head fa16=1.
static inline void m64_attn_f16_range(
    const device float * q, const device float * rout, float sink,
    unsigned long relbiaspack, unsigned long kvkoff, unsigned long kvvoff,
    uint pos, uint is_swa, uint ring_cap, float scale, float tau,
    uint h, uint lane, uint j0, uint j1,
    const device uchar * PACK, const device half * KV,
    thread float * oh, thread float & M, thread float & S) {
    const uint t = lane * 4u;
    half qh[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { float qf = q[t + i] * tau; qh[i] = half(qf); }
    const uint kvoff = (h / 3u) * 128u;
    for (uint j = j0; j < j1; j++) {
        const uint slot_idx = is_swa ? (j % ring_cap) : j;
        const device half * kj = KV + kvkoff + (size_t)slot_idx * 1024u + kvoff + t;
        float d = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) d = fma(float(qh[i]), float(kj[i]), d);
        d = simd_sum(d);
        uint dist = pos - j;
        float bv = 0.0f;
        if (dist < M64_EXT) {
            const device float * brow = ((const device float *)(PACK + relbiaspack)) + (size_t)dist * 16u;
            // split dot over 16 channels across lanes 0..15 (else simd_sum = 32x dot)
            float v = (lane < 16u) ? (brow[lane] * (rout[lane] * tau)) : 0.0f;
            bv = simd_sum(v);
        }
        float s = d * scale + f16r(bv - sink);
        float ms = 1.0f, vs;
        if (s > M) {
            ms = metal::exp(M - s);
            oh[0] *= ms; oh[1] *= ms; oh[2] *= ms; oh[3] *= ms;
            M = s; vs = 1.0f;
        } else {
            vs = metal::exp(s - M);
        }
        const device half * vj = KV + kvvoff + (size_t)slot_idx * 1024u + kvoff + t;
        #pragma unroll
        for (int i = 0; i < 4; i++) oh[i] = fma(vs, float(vj[i]), oh[i]);
        S = S * (float)ms + (float)vs;
    }
}

static inline void m64_attn_f16_finish(device float * out, uint t,
                                       thread float * oh, thread float & M, thread float & S) {
    // zero-sink term (sinks tensor == 0)
    float ms = 1.0f, vs;
    if (0.0f > M) {
        ms = (S > 0.0f) ? metal::exp(M - 0.0f) : 0.0f;
        oh[0] *= ms; oh[1] *= ms; oh[2] *= ms; oh[3] *= ms;
        M = 0.0f; vs = 1.0f;
    } else {
        vs = metal::exp(0.0f - M);
    }
    S = S * ms + vs;
    const float inv = (float)(1.0 / S);
    #pragma unroll
    for (int i = 0; i < 4; i++) out[t + i] = oh[i] * inv;
}

kernel void k_attn_head_f16(device float * SCR [[buffer(0)]],
                            constant AttnF16Args & A [[buffer(1)]],
                            const device uchar * PACK [[buffer(2)]],
                            const device half * KV [[buffer(3)]],
                            uint3 M64_TITG [[thread_position_in_threadgroup]],
                            uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint h = M64_TGPG.x;
    uint lane = M64_TITG.x;
    const uint t = lane * 4u;
    const device float * q = SCR + A.qoff + h * 128u;
    const device float * rout = SCR + A.rout + h * 16u;
    const float sink = SCR[A.sinkoff + h];
    float oh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M = -INFINITY;
    float S = 0.0f;
    const uint j_start = (A.is_swa && A.pos > A.n_swa) ? (A.pos - A.n_swa) : 0u;
    m64_attn_f16_range(q, rout, sink, A.relbiaspack, A.kvkoff, A.kvvoff,
                       A.pos, A.is_swa, A.ring_cap, A.scale, A.tau,
                       h, lane, j_start, A.pos + 1u, PACK, KV, oh, M, S);
    m64_attn_f16_finish(SCR + A.outoff + h * 128u, t, oh, M, S);
}

// Split-K FlashDecoding partial pass: grid = (kNH, n_splits, 1), tg = (32, 1, 1).
kernel void k_attn_head_f16_split(device float * SCR [[buffer(0)]],
                                  constant AttnF16SplitArgs & A [[buffer(1)]],
                                  const device uchar * PACK [[buffer(2)]],
                                  const device half * KV [[buffer(3)]],
                                  uint3 M64_TITG [[thread_position_in_threadgroup]],
                                  uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint h = M64_TGPG.x;
    const uint sp = M64_TGPG.y;
    const uint lane = M64_TITG.x;
    if (h >= 24u || sp >= A.n_splits) return;
    const uint t = lane * 4u;
    const uint j_start = (A.is_swa && A.pos > A.n_swa) ? (A.pos - A.n_swa) : 0u;
    const uint total = A.pos + 1u - j_start;
    const uint chunk = (total + A.n_splits - 1u) / A.n_splits;
    const uint j0 = j_start + sp * chunk;
    const uint j1 = (j0 < A.pos + 1u) ? min(j0 + chunk, A.pos + 1u) : j0;
    float oh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M = -INFINITY;
    float S = 0.0f;
    if (j0 < j1) {
        const device float * q = SCR + A.qoff + h * 128u;
        const device float * rout = SCR + A.rout + h * 16u;
        const float sink = SCR[A.sinkoff + h];
        m64_attn_f16_range(q, rout, sink, A.relbiaspack, A.kvkoff, A.kvvoff,
                           A.pos, A.is_swa, A.ring_cap, A.scale, A.tau,
                           h, lane, j0, j1, PACK, KV, oh, M, S);
    }
    const uint pidx = h * A.n_splits + sp;
    device float * po = SCR + A.part_o_off + pidx * 128u;
    #pragma unroll
    for (int i = 0; i < 4; i++) po[t + i] = oh[i];
    if (lane == 0u) {
        SCR[A.part_ms_off + pidx * 2u] = M;
        SCR[A.part_ms_off + pidx * 2u + 1u] = S;
    }
}

// Split-K FlashDecoding reduction: grid = (kNH, 1, 1), tg = (32, 1, 1).
// Each lane holds output channels t = lane*4 .. t+3, so every lane folds all
// splits sp in [0, n_splits) in ascending order (works for any n_splits <= 128).
kernel void k_attn_head_f16_reduce(device float * SCR [[buffer(0)]],
                                   constant AttnF16SplitArgs & A [[buffer(1)]],
                                   uint3 M64_TITG [[thread_position_in_threadgroup]],
                                   uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint h = M64_TGPG.x;
    if (h >= 24u) return;
    const uint lane = M64_TITG.x;
    const uint t = lane * 4u;
    float oh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M = -INFINITY;
    float S = 0.0f;
    for (uint sp = 0; sp < A.n_splits; sp++) {
        const uint pidx = h * A.n_splits + sp;
        const float Ms = SCR[A.part_ms_off + pidx * 2u];
        const float Ss = SCR[A.part_ms_off + pidx * 2u + 1u];
        if (Ss <= 0.0f) continue;
        const device float * po = SCR + A.part_o_off + pidx * 128u;
        const float M_new = (Ms > M) ? Ms : M;
        const float alpha = (S > 0.0f) ? metal::exp(M - M_new) : 0.0f;
        const float beta = metal::exp(Ms - M_new);
        #pragma unroll
        for (int i = 0; i < 4; i++) oh[i] = oh[i] * alpha + po[t + i] * beta;
        S = S * alpha + Ss * beta;
        M = M_new;
    }
    m64_attn_f16_finish(SCR + A.outoff + h * 128u, t, oh, M, S);
}

// ------------------------------------------------- Milestone 9C: GQA split-K
// k_attn_head_f16_split / k_attn_head_q8_split loaded each KV block 3x (once
// per Q head of the GQA group). These variants dispatch ONE warp per KV head
// (grid = (8, n_splits, 1), tg = (32, 1, 1)) that keeps the 3 Q heads'
// q/oh/M/S in registers (qh 12 half + oh 12 float per lane: no barriers, no
// shared memory) and loads every kh/vh exactly once. Per-head operation order
// matches m64_attn_f16_range / m64_attn_q8_range line-for-line, so the
// partials are numerically identical to the 24-head kernels; the partial
// layout part_o[h * n_splits + sp] / part_ms[h * n_splits + sp] is unchanged,
// so k_attn_head_f16_reduce reduces both variants identically.
static inline void m64_attn_f16_acc(float s, float v0, float v1, float v2, float v3,
                                    thread float * oh, thread float & M, thread float & S) {
    float ms = 1.0f, vs;
    if (s > M) {
        ms = metal::exp(M - s);
        oh[0] *= ms; oh[1] *= ms; oh[2] *= ms; oh[3] *= ms;
        M = s; vs = 1.0f;
    } else {
        vs = metal::exp(s - M);
    }
    oh[0] = fma(vs, v0, oh[0]);
    oh[1] = fma(vs, v1, oh[1]);
    oh[2] = fma(vs, v2, oh[2]);
    oh[3] = fma(vs, v3, oh[3]);
    S = S * (float)ms + (float)vs;
}

static inline void m64_attn_q8_acc(float s, float vd, char4 vq,
                                   thread float * oh, thread float & M, thread float & S) {
    float ms = 1.0f, vs;
    if (s > M) {
        ms = metal::exp(M - s);
        oh[0] *= ms; oh[1] *= ms; oh[2] *= ms; oh[3] *= ms;
        M = s; vs = 1.0f;
    } else {
        vs = metal::exp(s - M);
    }
    const float vscale = vs * vd;
    oh[0] = fma(vscale, float(vq.x), oh[0]);
    oh[1] = fma(vscale, float(vq.y), oh[1]);
    oh[2] = fma(vscale, float(vq.z), oh[2]);
    oh[3] = fma(vscale, float(vq.w), oh[3]);
    S = S * (float)ms + (float)vs;
}

// Relative-bias dot for the 3 GQA heads (same order/rounding as the 1-head
// kernels: (lane < 16) ? brow[lane] * (rout[lane] * tau) : 0, then simd_sum).
static inline void m64_attn_gqa_bias(const device uchar * PACK, unsigned long relbiaspack,
                                     uint dist, const device float * rout0, float tau,
                                     uint lane, thread float & bv0, thread float & bv1,
                                     thread float & bv2) {
    bv0 = 0.0f; bv1 = 0.0f; bv2 = 0.0f;
    if (dist >= M64_EXT) return;
    const device float * brow = ((const device float *)(PACK + relbiaspack)) + (size_t)dist * 16u;
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f;
    if (lane < 16u) {
        v0 = brow[lane] * (rout0[lane] * tau);
        v1 = brow[lane] * (rout0[lane + 16u] * tau);
        v2 = brow[lane] * (rout0[lane + 32u] * tau);
    }
    bv0 = simd_sum(v0);
    bv1 = simd_sum(v1);
    bv2 = simd_sum(v2);
}

kernel void k_attn_head_f16_split_gqa(device float * SCR [[buffer(0)]],
                                      constant AttnF16SplitArgs & A [[buffer(1)]],
                                      const device uchar * PACK [[buffer(2)]],
                                      const device half * KV [[buffer(3)]],
                                      uint3 M64_TITG [[thread_position_in_threadgroup]],
                                      uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint kvh = M64_TGPG.x;
    const uint sp = M64_TGPG.y;
    const uint lane = M64_TITG.x;
    if (kvh >= 8u || sp >= A.n_splits) return;
    const uint t = lane * 4u;
    const uint j_start = (A.is_swa && A.pos > A.n_swa) ? (A.pos - A.n_swa) : 0u;
    const uint total = A.pos + 1u - j_start;
    const uint chunk = (total + A.n_splits - 1u) / A.n_splits;
    const uint j0 = j_start + sp * chunk;
    const uint j1 = (j0 < A.pos + 1u) ? min(j0 + chunk, A.pos + 1u) : j0;
    const uint h0 = kvh * 3u;
    float oh0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float oh1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float oh2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M0 = -INFINITY, M1 = -INFINITY, M2 = -INFINITY;
    float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;
    if (j0 < j1) {
        const device float * q = SCR + A.qoff + h0 * 128u;
        half qa0[4], qa1[4], qa2[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            qa0[i] = half(q[t + i] * A.tau);
            qa1[i] = half(q[t + i + 128u] * A.tau);
            qa2[i] = half(q[t + i + 256u] * A.tau);
        }
        const device float * rout0 = SCR + A.rout + h0 * 16u;
        const float snk0 = SCR[A.sinkoff + h0];
        const float snk1 = SCR[A.sinkoff + h0 + 1u];
        const float snk2 = SCR[A.sinkoff + h0 + 2u];
        const device half * kb = KV + A.kvkoff + kvh * 128u + t;
        const device half * vb = KV + A.kvvoff + kvh * 128u + t;
        for (uint j = j0; j < j1; j++) {
            const uint slot_idx = A.is_swa ? (j % A.ring_cap) : j;
            const device half * kj = kb + (size_t)slot_idx * 1024u;
            const float k0 = float(kj[0]), k1 = float(kj[1]);
            const float k2 = float(kj[2]), k3 = float(kj[3]);
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const float kv = (i == 0) ? k0 : ((i == 1) ? k1 : ((i == 2) ? k2 : k3));
                d0 = fma(float(qa0[i]), kv, d0);
                d1 = fma(float(qa1[i]), kv, d1);
                d2 = fma(float(qa2[i]), kv, d2);
            }
            d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2);
            float bv0, bv1, bv2;
            m64_attn_gqa_bias(PACK, A.relbiaspack, A.pos - j, rout0, A.tau, lane, bv0, bv1, bv2);
            const float s0 = d0 * A.scale + f16r(bv0 - snk0);
            const float s1 = d1 * A.scale + f16r(bv1 - snk1);
            const float s2 = d2 * A.scale + f16r(bv2 - snk2);
            const device half * vj = vb + (size_t)slot_idx * 1024u;
            const float v0 = float(vj[0]), v1 = float(vj[1]);
            const float v2 = float(vj[2]), v3 = float(vj[3]);
            m64_attn_f16_acc(s0, v0, v1, v2, v3, oh0, M0, S0);
            m64_attn_f16_acc(s1, v0, v1, v2, v3, oh1, M1, S1);
            m64_attn_f16_acc(s2, v0, v1, v2, v3, oh2, M2, S2);
        }
    }
    const uint p0 = h0 * A.n_splits + sp;
    const uint p1 = (h0 + 1u) * A.n_splits + sp;
    const uint p2 = (h0 + 2u) * A.n_splits + sp;
    device float * po0 = SCR + A.part_o_off + p0 * 128u;
    device float * po1 = SCR + A.part_o_off + p1 * 128u;
    device float * po2 = SCR + A.part_o_off + p2 * 128u;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        po0[t + i] = oh0[i];
        po1[t + i] = oh1[i];
        po2[t + i] = oh2[i];
    }
    if (lane == 0u) {
        SCR[A.part_ms_off + p0 * 2u] = M0;
        SCR[A.part_ms_off + p0 * 2u + 1u] = S0;
        SCR[A.part_ms_off + p1 * 2u] = M1;
        SCR[A.part_ms_off + p1 * 2u + 1u] = S1;
        SCR[A.part_ms_off + p2 * 2u] = M2;
        SCR[A.part_ms_off + p2 * 2u + 1u] = S2;
    }
}

// ---------------- Milestone 5A: Q8_0 KV cache (write + attention + split-K) --
// Per KV head 136 B (8 B aligned): int8 qs[128] = 4 contiguous Q8_0 blocks of
// 32 (d[blk] = amax_blk / 127 for elements [blk*32, blk*32+32)), then 4 half
// scales d[4]. Per token slot 8 KV heads x 136 B = 1088 B (46.9% of FP16).
// A.dst_k / A.dst_v / A.kvkoff / A.kvvoff KEEP their HALF-unit units ->
// byte offsets via *2. Head base at KV + off*2 + slot*1088 + kvh*136.
// Quant math mirrors k_quant/k_kv_write_q8_batch: d pre-rounding (codes use
// the full-float id = 127/amax), stored d is f16-rounded, rint ties-to-even,
// clamp +-127. dst bases must be multiples of 4 half units (8 B) for the
// char4 stores.
static inline char q8_c8(float x, float id) {
    int v = (int)metal::rint(x * id);
    v = (v < -127) ? -127 : ((v > 127) ? 127 : v);
    return (char)v;
}

// grid = (8, 1, 1), threadgroup = (32, 1, 1): kvh from TGPG.x. Lanes 0..7,
// 8..15, 16..23, 24..31 are the independent 8-lane quant groups of the 4
// 32-element blocks (xor masks 1/2/4 never cross a group).
kernel void k_kv_write_q8(const device float * SCR [[buffer(0)]],
                          constant KVWriteF16Args & A [[buffer(1)]],
                          device uchar * KV [[buffer(2)]],
                          uint3 M64_TITG [[thread_position_in_threadgroup]],
                          uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint kvh = M64_TGPG.x, lane = M64_TITG.x;
    const uint t = lane * 4u;
    const uint blk = lane >> 3u;
    const device float * src[2] = {SCR + A.koff + kvh * 128u + t,
                                   SCR + A.voff + kvh * 128u + t};
    device uchar * dst[2] = {KV + (size_t)A.dst_k * 2ul + (size_t)kvh * 136ul,
                             KV + (size_t)A.dst_v * 2ul + (size_t)kvh * 136ul};
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        const device float * x = src[s];
        const float x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];
        float amax = fmax(fmax(fabs(x0), fabs(x1)), fmax(fabs(x2), fabs(x3)));
        amax = fmax(amax, simd_shuffle_xor(amax, 1u));
        amax = fmax(amax, simd_shuffle_xor(amax, 2u));
        amax = fmax(amax, simd_shuffle_xor(amax, 4u));
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        char4 q;
        q.x = q8_c8(x0, id); q.y = q8_c8(x1, id);
        q.z = q8_c8(x2, id); q.w = q8_c8(x3, id);
        *(device char4 *)(dst[s] + t) = q;
        if ((lane & 7u) == 0u)
            ((device half *)(dst[s] + 128u))[blk] = half(amax * (1.0f / 127.0f));
    }
}

// ---------------- Milestone 11: fused decode QKV prep + KV write (5 -> 1) ---
// grid = (64, 1, 1), tg = (32, 1, 1): one warp per head, ZERO threadgroup
// barriers. wg 0..23 Q-head clip (k_clip per=128 semantics, qclip scale,
// in-place SCR); 24..31 K-head clip -> direct KV write (K clip uses
// M64_NOFF scale, mirrors k_clip + k_kv_write_*); 32..39 V-head per-head
// rmsnorm (k_rms hasw=0 semantics: inv = 1/sqrt(s/k + eps)) -> direct KV
// write, o.emb staging eliminated; 40..63 Rr-head clip (k_clip per=16
// semantics, relclip scale, in-place SCR).
template <typename KVt, bool Q8>
static inline void m64_qkv_prep_kv_write(
    device float * SCR, constant QkvPrepKvArgs & A,
    const device uchar * PACK, device KVt * KV, uint wg, uint lane) {
    if (wg < 24u) {                                   // Q heads 0..23
        const uint h = wg, t = lane * 4u;
        device float * base = SCR + A.qoff + h * 128u;
        const device float * sc = A.qclip_off != M64_NOFF
            ? ((const device float *)(PACK + A.qclip_off)) + h * 128u : nullptr;
        float v0 = base[t], v1 = base[t + 1u], v2 = base[t + 2u], v3 = base[t + 3u];
        if (sc) {
            const device float4 * s4 = (const device float4 *)(sc + t);
            const float4 sm = *s4;
            v0 *= sm.x; v1 *= sm.y; v2 *= sm.z; v3 *= sm.w;
        }
        const float s = simd_sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
        float f = A.qk_thresh / (sqrt(s) + 1e-6f);
        f = fmin(fmax(f, 0.0f), 1.0f);
        base[t] = v0 * f; base[t + 1u] = v1 * f;
        base[t + 2u] = v2 * f; base[t + 3u] = v3 * f;
    } else if (wg < 32u) {                            // KV heads (K) 0..7
        const uint kvh = wg - 24u, t = lane * 4u;
        const device float * x = SCR + A.koff + kvh * 128u;
        float k0 = x[t], k1 = x[t + 1u], k2 = x[t + 2u], k3 = x[t + 3u];
        const float s = simd_sum(k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3);
        float f = A.qk_thresh / (sqrt(s) + 1e-6f);
        f = fmin(fmax(f, 0.0f), 1.0f);
        k0 *= f; k1 *= f; k2 *= f; k3 *= f;
        if constexpr (Q8) {
            device uchar * dst = KV + (size_t)A.dst_k * 2ul + (size_t)kvh * 136ul;
            float amax = fmax(fmax(fabs(k0), fabs(k1)), fmax(fabs(k2), fabs(k3)));
            amax = fmax(amax, simd_shuffle_xor(amax, 1u));
            amax = fmax(amax, simd_shuffle_xor(amax, 2u));
            amax = fmax(amax, simd_shuffle_xor(amax, 4u));
            const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            char4 q;
            q.x = q8_c8(k0, id); q.y = q8_c8(k1, id);
            q.z = q8_c8(k2, id); q.w = q8_c8(k3, id);
            *(device char4 *)(dst + t) = q;
            if ((lane & 7u) == 0u)
                ((device half *)(dst + 128u))[lane >> 3u] = half(amax * (1.0f / 127.0f));
        } else {
            device half * dst = (device half *)KV + (size_t)A.dst_k + (size_t)kvh * 128ul;
            dst[t] = half(k0); dst[t + 1u] = half(k1);
            dst[t + 2u] = half(k2); dst[t + 3u] = half(k3);
        }
    } else if (wg < 40u) {                            // KV heads (V) 0..7
        const uint kvh = wg - 32u, t = lane * 4u;
        const device float * x = SCR + A.voff + kvh * 128u;
        const float x0 = x[t], x1 = x[t + 1u], x2 = x[t + 2u], x3 = x[t + 3u];
        const float s = simd_sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
        const float inv = 1.0f / sqrt(s / 128.0f + A.eps);
        const float v0 = x0 * inv, v1 = x1 * inv, v2 = x2 * inv, v3 = x3 * inv;
        if constexpr (Q8) {
            device uchar * dst = KV + (size_t)A.dst_v * 2ul + (size_t)kvh * 136ul;
            float amax = fmax(fmax(fabs(v0), fabs(v1)), fmax(fabs(v2), fabs(v3)));
            amax = fmax(amax, simd_shuffle_xor(amax, 1u));
            amax = fmax(amax, simd_shuffle_xor(amax, 2u));
            amax = fmax(amax, simd_shuffle_xor(amax, 4u));
            const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            char4 q;
            q.x = q8_c8(v0, id); q.y = q8_c8(v1, id);
            q.z = q8_c8(v2, id); q.w = q8_c8(v3, id);
            *(device char4 *)(dst + t) = q;
            if ((lane & 7u) == 0u)
                ((device half *)(dst + 128u))[lane >> 3u] = half(amax * (1.0f / 127.0f));
        } else {
            device half * dst = (device half *)KV + (size_t)A.dst_v + (size_t)kvh * 128ul;
            dst[t] = half(v0); dst[t + 1u] = half(v1);
            dst[t + 2u] = half(v2); dst[t + 3u] = half(v3);
        }
    } else if (wg < 64u) {                            // Rr heads 0..23
        const uint h = wg - 40u;
        const device float * src = SCR + A.roff + h * 16u;
        const device float * sc = A.relclip_off != M64_NOFF
            ? ((const device float *)(PACK + A.relclip_off)) + h * 16u : nullptr;
        float v = (lane < 16u) ? src[lane] : 0.0f;
        if (sc && lane < 16u) v *= sc[lane];
        const float s = simd_sum(v * v);
        float f = A.rel_thresh / (sqrt(s) + 1e-6f);
        f = fmin(fmax(f, 0.0f), 1.0f);
        if (lane < 16u) SCR[A.roff + h * 16u + lane] = v * f;
    }
}

kernel void k_qkv_prep_kv_write_q8(device float * SCR [[buffer(0)]],
                                   constant QkvPrepKvArgs & A [[buffer(1)]],
                                   const device uchar * PACK [[buffer(2)]],
                                   device uchar * KV [[buffer(3)]],
                                   uint3 M64_TITG [[thread_position_in_threadgroup]],
                                   uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    m64_qkv_prep_kv_write<uchar, true>(SCR, A, PACK, KV, M64_TGPG.x, M64_TITG.x);
}

kernel void k_qkv_prep_kv_write_f16(device float * SCR [[buffer(0)]],
                                    constant QkvPrepKvArgs & A [[buffer(1)]],
                                    const device uchar * PACK [[buffer(2)]],
                                    device half * KV [[buffer(3)]],
                                    uint3 M64_TITG [[thread_position_in_threadgroup]],
                                    uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    m64_qkv_prep_kv_write<half, false>(SCR, A, PACK, KV, M64_TGPG.x, M64_TITG.x);
}

// ---------------- Milestone 11: fused postconv + residual (2 -> 1) ----------
// k_conv_slot (postconv, hasdyn=0) + k_xfer fold. Flat (A.n)x128 dispatch;
// every channel c is owned by ONE thread, so the read/modify/write of
// SCR[curoff+c] (push=0 add) and the cur->blkoff snapshot (push=1) are both
// race-free within this single dispatch.
kernel void k_postconv_residual(device float * SCR [[buffer(0)]],
                                constant PostConvResArgs & A [[buffer(1)]],
                                const device uchar * PACK [[buffer(2)]],
                                uint3 M64_TIG [[thread_position_in_grid]]) {
    const uint c = M64_TIG.x;
    if (c >= A.n) return;
    device float * st = SCR + A.rsoff + c * 3u;
    const float s0 = st[0], s1 = st[1], s2 = st[2];
    const float xin = SCR[A.xoff + c];
    const device float4 * w4 = ((const device float4 *)(PACK + A.wconv)) + c;
    const float4 w = *w4;
    const float outv = s0 * w.x + s1 * w.y + s2 * w.z + xin * w.w;
    st[0] = s1; st[1] = s2; st[2] = xin;
    const float y = (xin + outv) * A.scale;
    if (A.push == 0u) SCR[A.curoff + c] = SCR[A.curoff + c] + y;
    else { SCR[A.blkoff + c] = SCR[A.curoff + c]; SCR[A.curoff + c] = y; }
}

// Attention over the Q8_0 KV buffer. Numerics mirror m64_attn_f16_range with
// K/V dequantized on the fly: the 4-lane-per-block layout puts ONE Q8_0
// block under each 8-lane group, so the block scale folds into kq per lane
// BEFORE simd_sum (kd/vscale are lane-dependent: blk = lane >> 3).
static inline void m64_attn_q8_range(
    const device float * q, const device float * rout, float sink,
    unsigned long relbiaspack, unsigned long kvkoff, unsigned long kvvoff,
    uint pos, uint is_swa, uint ring_cap, float scale, float tau,
    uint h, uint lane, uint j0, uint j1,
    const device uchar * PACK, const device uchar * KV,
    thread float * oh, thread float & M, thread float & S) {
    const uint t = lane * 4u;
    const uint kvh = h / 3u;
    const uint blk = lane >> 3u;
    half qh[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { float qf = q[t + i] * tau; qh[i] = half(qf); }
    const device uchar * kbase = KV + (size_t)kvkoff * 2ul + (size_t)kvh * 136ul;
    const device uchar * vbase = KV + (size_t)kvvoff * 2ul + (size_t)kvh * 136ul;
    for (uint j = j0; j < j1; j++) {
        const uint slot_idx = is_swa ? (j % ring_cap) : j;
        const device uchar * kh = kbase + (size_t)slot_idx * 1088ul;
        const float kd = float(((const device half *)(kh + 128u))[blk]);
        const char4 kq = *(const device char4 *)(kh + t);
        float d = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) d = fma(float(qh[i]), float(kq[i]) * kd, d);
        d = simd_sum(d);
        uint dist = pos - j;
        float bv = 0.0f;
        if (dist < M64_EXT) {
            const device float * brow = ((const device float *)(PACK + relbiaspack)) + (size_t)dist * 16u;
            // split dot over 16 channels across lanes 0..15 (else simd_sum = 32x dot)
            float v = (lane < 16u) ? (brow[lane] * (rout[lane] * tau)) : 0.0f;
            bv = simd_sum(v);
        }
        float s = d * scale + f16r(bv - sink);
        float ms = 1.0f, vs;
        if (s > M) {
            ms = metal::exp(M - s);
            oh[0] *= ms; oh[1] *= ms; oh[2] *= ms; oh[3] *= ms;
            M = s; vs = 1.0f;
        } else {
            vs = metal::exp(s - M);
        }
        const device uchar * vh = vbase + (size_t)slot_idx * 1088ul;
        const float vscale = vs * float(((const device half *)(vh + 128u))[blk]);
        const char4 vq = *(const device char4 *)(vh + t);
        #pragma unroll
        for (int i = 0; i < 4; i++) oh[i] = fma(vscale, float(vq[i]), oh[i]);
        S = S * (float)ms + (float)vs;
    }
}

kernel void k_attn_head_q8(device float * SCR [[buffer(0)]],
                           constant AttnF16Args & A [[buffer(1)]],
                           const device uchar * PACK [[buffer(2)]],
                           const device uchar * KV [[buffer(3)]],
                           uint3 M64_TITG [[thread_position_in_threadgroup]],
                           uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint h = M64_TGPG.x;
    uint lane = M64_TITG.x;
    const uint t = lane * 4u;
    const device float * q = SCR + A.qoff + h * 128u;
    const device float * rout = SCR + A.rout + h * 16u;
    const float sink = SCR[A.sinkoff + h];
    float oh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M = -INFINITY;
    float S = 0.0f;
    const uint j_start = (A.is_swa && A.pos > A.n_swa) ? (A.pos - A.n_swa) : 0u;
    m64_attn_q8_range(q, rout, sink, A.relbiaspack, A.kvkoff, A.kvvoff,
                      A.pos, A.is_swa, A.ring_cap, A.scale, A.tau,
                      h, lane, j_start, A.pos + 1u, PACK, KV, oh, M, S);
    m64_attn_f16_finish(SCR + A.outoff + h * 128u, t, oh, M, S);
}

// Split-K FlashDecoding over the Q8_0 KV cache: grid = (kNH, n_splits, 1),
// tg = (32, 1, 1). k_attn_head_f16_reduce (SCR-only) reduces the partials.
kernel void k_attn_head_q8_split(device float * SCR [[buffer(0)]],
                                 constant AttnF16SplitArgs & A [[buffer(1)]],
                                 const device uchar * PACK [[buffer(2)]],
                                 const device uchar * KV [[buffer(3)]],
                                 uint3 M64_TITG [[thread_position_in_threadgroup]],
                                 uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint h = M64_TGPG.x;
    const uint sp = M64_TGPG.y;
    const uint lane = M64_TITG.x;
    if (h >= 24u || sp >= A.n_splits) return;
    const uint t = lane * 4u;
    const uint j_start = (A.is_swa && A.pos > A.n_swa) ? (A.pos - A.n_swa) : 0u;
    const uint total = A.pos + 1u - j_start;
    const uint chunk = (total + A.n_splits - 1u) / A.n_splits;
    const uint j0 = j_start + sp * chunk;
    const uint j1 = (j0 < A.pos + 1u) ? min(j0 + chunk, A.pos + 1u) : j0;
    float oh[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M = -INFINITY;
    float S = 0.0f;
    if (j0 < j1) {
        const device float * q = SCR + A.qoff + h * 128u;
        const device float * rout = SCR + A.rout + h * 16u;
        const float sink = SCR[A.sinkoff + h];
        m64_attn_q8_range(q, rout, sink, A.relbiaspack, A.kvkoff, A.kvvoff,
                          A.pos, A.is_swa, A.ring_cap, A.scale, A.tau,
                          h, lane, j0, j1, PACK, KV, oh, M, S);
    }
    const uint pidx = h * A.n_splits + sp;
    device float * po = SCR + A.part_o_off + pidx * 128u;
    #pragma unroll
    for (int i = 0; i < 4; i++) po[t + i] = oh[i];
    if (lane == 0u) {
        SCR[A.part_ms_off + pidx * 2u] = M;
        SCR[A.part_ms_off + pidx * 2u + 1u] = S;
    }
}

// Split-K FlashDecoding over the Q8_0 KV cache with GQA 3-head KV sharing:
// grid = (8, n_splits, 1), tg = (32, 1, 1). Same partial layout / numerics as
// k_attn_head_q8_split (kh/vh dequantized once for the 3 Q heads of kvh).
kernel void k_attn_head_q8_split_gqa(device float * SCR [[buffer(0)]],
                                     constant AttnF16SplitArgs & A [[buffer(1)]],
                                     const device uchar * PACK [[buffer(2)]],
                                     const device uchar * KV [[buffer(3)]],
                                     uint3 M64_TITG [[thread_position_in_threadgroup]],
                                     uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    const uint kvh = M64_TGPG.x;
    const uint sp = M64_TGPG.y;
    const uint lane = M64_TITG.x;
    if (kvh >= 8u || sp >= A.n_splits) return;
    const uint t = lane * 4u;
    const uint j_start = (A.is_swa && A.pos > A.n_swa) ? (A.pos - A.n_swa) : 0u;
    const uint total = A.pos + 1u - j_start;
    const uint chunk = (total + A.n_splits - 1u) / A.n_splits;
    const uint j0 = j_start + sp * chunk;
    const uint j1 = (j0 < A.pos + 1u) ? min(j0 + chunk, A.pos + 1u) : j0;
    const uint h0 = kvh * 3u;
    float oh0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float oh1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float oh2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float M0 = -INFINITY, M1 = -INFINITY, M2 = -INFINITY;
    float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;
    if (j0 < j1) {
        const device float * q = SCR + A.qoff + h0 * 128u;
        half qa0[4], qa1[4], qa2[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            qa0[i] = half(q[t + i] * A.tau);
            qa1[i] = half(q[t + i + 128u] * A.tau);
            qa2[i] = half(q[t + i + 256u] * A.tau);
        }
        const device float * rout0 = SCR + A.rout + h0 * 16u;
        const float snk0 = SCR[A.sinkoff + h0];
        const float snk1 = SCR[A.sinkoff + h0 + 1u];
        const float snk2 = SCR[A.sinkoff + h0 + 2u];
        const uint blk = lane >> 3u;
        const device uchar * kbase = KV + (size_t)A.kvkoff * 2ul + (size_t)kvh * 136ul;
        const device uchar * vbase = KV + (size_t)A.kvvoff * 2ul + (size_t)kvh * 136ul;
        for (uint j = j0; j < j1; j++) {
            const uint slot_idx = A.is_swa ? (j % A.ring_cap) : j;
            const device uchar * kh = kbase + (size_t)slot_idx * 1088ul;
            const float kd = float(((const device half *)(kh + 128u))[blk]);
            const char4 kq = *(const device char4 *)(kh + t);
            const float kk0 = float(kq.x) * kd, kk1 = float(kq.y) * kd;
            const float kk2 = float(kq.z) * kd, kk3 = float(kq.w) * kd;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const float kv = (i == 0) ? kk0 : ((i == 1) ? kk1 : ((i == 2) ? kk2 : kk3));
                d0 = fma(float(qa0[i]), kv, d0);
                d1 = fma(float(qa1[i]), kv, d1);
                d2 = fma(float(qa2[i]), kv, d2);
            }
            d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2);
            float bv0, bv1, bv2;
            m64_attn_gqa_bias(PACK, A.relbiaspack, A.pos - j, rout0, A.tau, lane, bv0, bv1, bv2);
            const float s0 = d0 * A.scale + f16r(bv0 - snk0);
            const float s1 = d1 * A.scale + f16r(bv1 - snk1);
            const float s2 = d2 * A.scale + f16r(bv2 - snk2);
            const device uchar * vh = vbase + (size_t)slot_idx * 1088ul;
            const float vd = float(((const device half *)(vh + 128u))[blk]);
            const char4 vq = *(const device char4 *)(vh + t);
            m64_attn_q8_acc(s0, vd, vq, oh0, M0, S0);
            m64_attn_q8_acc(s1, vd, vq, oh1, M1, S1);
            m64_attn_q8_acc(s2, vd, vq, oh2, M2, S2);
        }
    }
    const uint p0 = h0 * A.n_splits + sp;
    const uint p1 = (h0 + 1u) * A.n_splits + sp;
    const uint p2 = (h0 + 2u) * A.n_splits + sp;
    device float * po0 = SCR + A.part_o_off + p0 * 128u;
    device float * po1 = SCR + A.part_o_off + p1 * 128u;
    device float * po2 = SCR + A.part_o_off + p2 * 128u;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        po0[t + i] = oh0[i];
        po1[t + i] = oh1[i];
        po2[t + i] = oh2[i];
    }
    if (lane == 0u) {
        SCR[A.part_ms_off + p0 * 2u] = M0;
        SCR[A.part_ms_off + p0 * 2u + 1u] = S0;
        SCR[A.part_ms_off + p1 * 2u] = M1;
        SCR[A.part_ms_off + p1 * 2u + 1u] = S1;
        SCR[A.part_ms_off + p2 * 2u] = M2;
        SCR[A.part_ms_off + p2 * 2u + 1u] = S2;
    }
}

// ------------------------------------------------------------ gate sigmoid mul
kernel void k_gate_mul(device float * SCR [[buffer(0)]],
                       constant GateMulArgs & A [[buffer(1)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint i = M64_TIG.x;
    if (i >= A.n) return;
    SCR[A.aoff + i] *= ksig(SCR[A.goff + i]);
}

// ------------------------------------------------------- SITU activation+quant
kernel void k_situ_quant(device float * SCR [[buffer(0)]],
                         constant SituArgs & A [[buffer(1)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint b = M64_TIG.x;
    if (b >= A.nb) return;
    const device float * g = SCR + A.goff + b * 32u;
    const device float * u = SCR + A.uoff + b * 32u;
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
    SCR[A.doff + b] = f16b(as_type<unsigned short>(half(d)));
    device uchar * q = (device uchar *)(SCR + A.qoff) + b * 32u;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int r = int(rint(hs[i] * id));
        r = r > 127 ? 127 : (r < -127 ? -127 : r);
        q[i] = uchar(char(r));
    }
}

// ------------------------------------------------------------------ router topk
// Serial reference kept for --bench A/B (dispatch name k_router_topk_serial).
kernel void k_router_topk_serial(device float * SCR [[buffer(0)]],
                                 constant RouterArgs & A [[buffer(1)]],
                                 const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]]) {
    uint e = M64_TITG.x;       // 128 threads
    threadgroup float pp[128], sp[128];
    float p = ksig(SCR[A.probs + e]);
    pp[e] = p;
    sp[e] = p + ((const device float *)(PACK + A.biaspack))[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e != 0) return;
    int sel[8]; float ps[8];
    float wsum = 0.0f;
    for (int t = 0; t < 8; t++) {
        float best = -INFINITY; int bi = 0;
        for (int i = 0; i < 128; i++)
            if (sp[i] > best) { best = sp[i]; bi = i; }
        sel[t] = bi; ps[t] = pp[bi]; sp[bi] = -INFINITY;
        wsum += pp[bi];
    }
    float div = wsum + 1e-20f;
    device int * seld = (device int *)(SCR + A.seloff);
    for (int t = 0; t < 8; t++) {
        seld[t] = sel[t];
        SCR[A.wtoff + t] = (ps[t] / div) * A.scale;
    }
}

// R4: same selection as k_router_topk_serial (ties -> smallest index) but the
// argmax scan runs in parallel on all 128 threads (4 warps of 32 lanes):
// per-warp (simd_max, simd_min-of-tie-index) -> warp 0 merges the 4 candidates.
kernel void k_router_topk(device float * SCR [[buffer(0)]],
                          constant RouterArgs & A [[buffer(1)]],
                          const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]]) {
    uint e = M64_TITG.x;       // 128 threads
    uint lane = e & 31u;
    threadgroup float pp[128], sp[128];
    threadgroup float warp_v[4];
    threadgroup uint warp_i[4];
    float p = ksig(SCR[A.probs + e]);
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
        device int * seld = (device int *)(SCR + A.seloff);
        for (uint t = 0; t < 8u; t++) {
            seld[t] = (int)sel[t];
            SCR[A.wtoff + t] = (ps[t] / div) * A.scale;
        }
    }
}

// ---------------------------------------------------------------- mix kernels
// sc[j][i] = dot(stn_i, query_j) * scale          grid (nq, ceil(S/8))
kernel void k_mix_dot(device float * SCR [[buffer(0)]],
                      constant MixDotArgs & A [[buffer(1)]],
                      const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_NPTG [[threads_per_threadgroup]]) {
    uint j = M64_TGPG.x;
    uint i0 = M64_TGPG.y * 8u;
    uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    const device float * q = ((const device float *)(PACK + A.qpack)) + (size_t)j * M64_NE;
    float acc[8];
    #pragma unroll
    for (int u = 0; u < 8; u++) acc[u] = 0.0f;
    uint hi = min(i0 + 8u, A.S);
    for (uint k = tid; k < M64_NE; k += nth) {
        float qv = q[k];
        #pragma unroll
        for (uint u = 0; u < 8u; u++) {
            if (i0 + u < hi)
                acc[u] = fma(SCR[A.stnoff + (size_t)(i0 + u) * M64_NE + k], qv, acc[u]);
        }
    }
    threadgroup float red[256];
    #pragma unroll
    for (uint u = 0; u < 8u; u++) {
        if (i0 + u >= hi) break;
        red[tid] = acc[u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = nth / 2; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) SCR[A.scoff + j * M64_MAXS + (i0 + u)] = red[0] * A.scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// softmax over S states                                      grid.x = nq, tg 64
kernel void k_mix_soft(device float * SCR [[buffer(0)]],
                       constant MixSoftArgs & A [[buffer(1)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint j = M64_TGPG.x;
    uint tid = M64_TITG.x;
    threadgroup float wt[64];
    if (tid == 0) {
        const device float * sc = SCR + A.scoff + j * M64_MAXS;
        float mx = -INFINITY;
        for (uint i = 0; i < A.S; i++) mx = sc[i] > mx ? sc[i] : mx;
        float sum = 0.0;
        for (uint i = 0; i < A.S; i++) { wt[i] = metal::exp(sc[i] - mx); sum += (float)wt[i]; }
        float recip = (float)(1.0 / sum);
        for (uint i = 0; i < A.S; i++) wt[i] = wt[i] * recip;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < A.S) SCR[A.scoff + j * M64_MAXS + tid] = wt[tid];
}

// out_j = Σ_i w_i · stn_i                                    grid.x = nq, tg 256
kernel void k_mix_comb(device float * SCR [[buffer(0)]],
                       constant MixCombArgs & A [[buffer(1)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_NPTG [[threads_per_threadgroup]]) {
    uint j = M64_TGPG.x;
    uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    threadgroup float wt[64];
    if (tid < A.S) wt[tid] = SCR[A.scoff + j * M64_MAXS + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * out = SCR + A.outoff + (size_t)j * M64_NE;
    for (uint k = tid; k < M64_NE; k += nth) {
        float a = 0.0f;
        for (uint i = 0; i < A.S; i++)
            a = fma(wt[i], SCR[A.stnoff + (size_t)i * M64_NE + k], a);
        out[k] = a;
    }
}

// ------------------------------------------------------------------- control
kernel void k_control(device float * SCR [[buffer(0)]],
                      constant CtrlArgs & A [[buffer(1)]],
                      const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint cg = M64_TGPG.x;
    uint tid = M64_TITG.x;      // 1024
    const device float * gc = SCR + A.gcat + cg * 2u * M64_NE;
    const device float * wm = (const device float *)(PACK + A.wdown[cg]);
    threadgroup float hh[64];
    threadgroup float red[1024];
    uint r = tid >> 4, l = tid & 15u;
    {
        const device float * wr = wm + (size_t)r * (2u * M64_NE);
        float a = 0.0f;
        // Round-2: float4 dot over wdown row (4608 f32 = 1152 float4, 16
        // lanes -> 72 float4 each). PACK rows are 16B-aligned; guard on the
        // SCR gcat offset (scalar fallback keeps misaligned tests safe).
        if ((A.gcat & 3u) == 0u) {
            const device float4 * w4 = (const device float4 *)wr;
            const device float4 * g4 = (const device float4 *)gc;
            float4 av = float4(0.0f);
            for (uint k = l; k < (2u * M64_NE) / 4u; k += 16u) av += w4[k] * g4[k];
            a = (av.x + av.y) + (av.z + av.w);
        } else {
            for (uint k = l; k < 2u * M64_NE; k += 16u) a = fma(wr[k], gc[k], a);
        }
        red[tid] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = 8u; st > 0; st >>= 1) {
            if (l < st) red[tid] += red[(r << 4) + (l + st)];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (l == 0) hh[r] = tanh(red[r << 4]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    {
        const device float * wg = ((const device float *)(PACK + A.wgain[cg]));
        const device float * wf = ((const device float *)(PACK + A.wforg[cg]));
        for (uint row = tid; row < M64_NE; row += 1024u) {
            float g = 0.0f, f = 0.0f;
            #pragma unroll
            for (int t = 0; t < 64; t++) { float hv = hh[t]; g = fma(wg[row * 64u + t], hv, g); f = fma(wf[row * 64u + t], hv, f); }
            SCR[A.gb[cg] + row] = tanh(g);
            SCR[A.fb[cg] + row] = tanh(f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < M64_NE; i += 1024u) {
        float c = SCR[A.in[cg] + i];
        float g = SCR[A.gb[cg] + i], fg = SCR[A.fb[cg] + i];
        float m = SCR[A.msg[cg] + i];
        SCR[A.out[cg] + i] = (c - fg * c) + g * m;
    }
}

// -------------------------------------------------------- transfers & output
kernel void k_xfer(device float * SCR [[buffer(0)]],
                   constant XferArgs & A [[buffer(1)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_NPTG [[threads_per_threadgroup]]) {
    uint g = M64_TGPG.x;      // pair index
    uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    uint n = A.n[g];
    if (A.mode == 0u) {                           // in-place a += b
        device float * a = SCR + A.a[g];
        const device float * b = SCR + A.b[g];
        for (uint i = tid; i < n; i += nth) a[i] = a[i] + b[i];
    } else {                                      // copy a -> c
        const device float * a = SCR + A.a[g];
        device float * c = SCR + A.c[g];
        for (uint i = tid; i < n; i += nth) c[i] = a[i];
    }
}

kernel void k_scale(device float * SCR [[buffer(0)]],
                    constant ScaleArgs & A [[buffer(1)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint i = M64_TIG.x;
    if (i >= A.n) return;
    SCR[A.off + i] *= A.v;
}

kernel void k_softcap(device float * SCR [[buffer(0)]],
                      constant SoftcapArgs & A [[buffer(1)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint i = M64_TIG.x;
    if (i >= A.n) return;
    float z = SCR[A.loff + i];
    SCR[A.loff + i] = A.cap * tanh(z * (1.0f / A.cap));
}

kernel void k_argmax(device float * SCR [[buffer(0)]],
                     constant ArgmaxArgs & A [[buffer(1)]], uint3 M64_TITG [[thread_position_in_threadgroup]]) {
    uint tid = M64_TITG.x;  // 1024
    float bv = -INFINITY; uint bi = 0;
    for (uint i = tid; i < A.n; i += 1024u) {
        float v = SCR[A.loff + i];
        if (v > bv) { bv = v; bi = i; }
    }
    threadgroup float rv[1024];
    threadgroup uint ri[1024];
    rv[tid] = bv; ri[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = 512u; st > 0; st >>= 1) {
        if (tid < st) {
            float v2 = rv[tid + st]; uint i2 = ri[tid + st];
            if (v2 > rv[tid] || (v2 == rv[tid] && i2 < ri[tid])) { rv[tid] = v2; ri[tid] = i2; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) ((device int *)(SCR + A.aoff))[0] = (int)ri[0];
}

// =============================== Phase-2A-opt ===============================

// legacy scalar-unpack GEMV, kept only for A/B benchmarking
kernel void k_gemv_q8_k2304_scalar(device float * SCR [[buffer(0)]],
                                   constant GemvArgs & A [[buffer(1)]],
                                   const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint row = M64_TGPG.x * 8u + (tid >> 5);
    if (row >= A.nout) return;
    uint lane = tid & 31u;
    const device uint * wr = (const device uint *)(PACK + A.woff + (size_t)A.idx * A.stride) + row * 612u;
    float a = 0.0f;
    for (uint u = lane; u < 36u; u += 32u)
        gemv_unit_scalar(wr + u * 17u, (const device uchar4 *)(SCR + A.xqoff) + u * 16u,
                         SCR + A.dxoff, u, a);
    a = simd_sum(a);
    if (lane == 0) SCR[A.yoff + row] = a;
}

// helper: one 68B unit weight block vs register-resident activations
static inline float gemv_w(const device uint * W, const uchar4 * XA, const uchar4 * XB,
                           float dx0, float dx1, float & a) {
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint v0 = (W[j] >> 16) | (W[j + 1] << 16);
        s0 += dot_c4u(as_type<char4>(v0), XA[j]);
        s1 += dot_c4u(as_type<char4>(W[9 + j]), XB[j]);
    }
    a = fma(float(s0), dx0 * f16b((unsigned short)(W[0] & 0xFFFFu)), a);
    a = fma(float(s1), dx1 * f16b((unsigned short)(W[8] >> 16)), a);
    return a;
}

// multi-row (2 rows/warp, k = 2304): activations loaded once per unit and
// reused across the row block; grid.x = ceil(nout / 16) threadgroups of 256.
kernel void k_gemv_q8_k2304_m2(device float * SCR [[buffer(0)]],
                               constant GemvArgs & A [[buffer(1)]],
                               const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint row0 = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (row0 >= A.nout) return;
    uint lane = tid & 31u;
    const device uint * base = (const device uint *)(PACK + A.woff + (size_t)A.idx * A.stride);
    float a0 = 0.0f, a1 = 0.0f;
    for (uint u = lane; u < 36u; u += 32u) {
        const device uchar4 * xup = (const device uchar4 *)(SCR + A.xqoff) + u * 16u;
        uchar4 XA[8], XB[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) { XA[j] = xup[j]; XB[j] = xup[8 + j]; }
        float dx0 = SCR[A.dxoff + 2 * u], dx1 = SCR[A.dxoff + 2 * u + 1u];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            uint row = row0 + (uint)r; if (row >= A.nout) row = A.nout - 1u;
            float * acc = r ? &a1 : &a0;
            gemv_w(base + (size_t)row * 612u + u * 17u, XA, XB, dx0, dx1, *acc);
        }
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0) SCR[A.yoff + row0] = a0;
    if (lane == 1 && row0 + 1u < A.nout) SCR[A.yoff + row0 + 1u] = a1;
}

// multi-row (4 rows/warp, k = 2304); grid.x = ceil(nout / 32) of 256.
kernel void k_gemv_q8_k2304_m4(device float * SCR [[buffer(0)]],
                               constant GemvArgs & A [[buffer(1)]],
                               const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint row0 = (M64_TGPG.x * 8u + (tid >> 5)) * 4u;
    if (row0 >= A.nout) return;
    uint lane = tid & 31u;
    const device uint * base = (const device uint *)(PACK + A.woff + (size_t)A.idx * A.stride);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint u = lane; u < 36u; u += 32u) {
        const device uchar4 * xup = (const device uchar4 *)(SCR + A.xqoff) + u * 16u;
        uchar4 XA[8], XB[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) { XA[j] = xup[j]; XB[j] = xup[8 + j]; }
        float dx0 = SCR[A.dxoff + 2 * u], dx1 = SCR[A.dxoff + 2 * u + 1u];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            uint row = row0 + (uint)r; if (row >= A.nout) row = A.nout - 1u;
            float * acc = r == 0 ? &a0 : (r == 1 ? &a1 : (r == 2 ? &a2 : &a3));
            gemv_w(base + (size_t)row * 612u + u * 17u, XA, XB, dx0, dx1, *acc);
        }
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0) SCR[A.yoff + row0] = a0;
    if (lane == 1 && row0 + 1u < A.nout) SCR[A.yoff + row0 + 1u] = a1;
    if (lane == 2 && row0 + 2u < A.nout) SCR[A.yoff + row0 + 2u] = a2;
    if (lane == 3 && row0 + 3u < A.nout) SCR[A.yoff + row0 + 3u] = a3;
}

// Batch-8 MoE gate|up: one dispatch for all 8 selected experts sharing the
// quantized activation. A.nout = kNF; grid.x = ceil(8 * 2 * nout / 8) of 256.
// Outputs laid out gate2[e*nout + r] / yoff2[e*nout + r].
kernel void k_gemv_gateup_batch8(device float * SCR [[buffer(0)]],
                                 constant GemvArgs & A [[buffer(1)]],
                                 const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint rr = M64_TGPG.x * 8u + (tid >> 5);
    if (rr >= 8u * 2u * A.nout) return;
    uint lane = tid & 31u;
    uint e = rr / (2u * A.nout);
    uint r2 = rr - e * (2u * A.nout);
    bool up = r2 >= A.nout;
    uint r = up ? r2 - A.nout : r2;
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
    const device uint * wr = (const device uint *)(PACK + (up ? A.woff2 : A.woff) + (size_t)exp_id * A.stride) + r * 612u;
    float a = 0.0f;
    for (uint u = lane; u < 36u; u += 32u)
        gemv_unit(wr + u * 17u, (const device uchar4 *)(SCR + A.xqoff) + u * 16u,
                  SCR + A.dxoff, u, a);
    a = simd_sum(a);
    if (lane == 0) SCR[(up ? A.yoff2 : A.yoff) + e * A.nout + r] = a;
}

// SIMD-group situ+quant over up to 8 experts of contiguous gate2/ubuf -> hq/dh
// (32 lanes per 32-element block, simd_max for amax). grid = nb*32 threads.
kernel void k_situ_quant8(device float * SCR [[buffer(0)]],
                          constant SituArgs & A [[buffer(1)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint tid = M64_TIG.x;
    if (tid >= A.nb * 32u) return;
    uint b = tid >> 5, lane = tid & 31u;
    float x = SCR[A.goff + tid];
    float act = A.beta * tanh(x * A.inv_beta);
    act = act * (1.0f / (1.0f + metal::exp(-x)));
    float ulin = A.lb * tanh(SCR[A.uoff + tid] * A.inv_lb);
    float v = act * ulin;
    const float amax = simd_max(fabs(v));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int r0 = int(rint(v * id));
    r0 = r0 > 127 ? 127 : (r0 < -127 ? -127 : r0);
    ((device uchar *)(SCR + A.qoff))[tid] = uchar(char(r0));
    if (lane == 0u) SCR[A.doff + b] = f16b(as_type<unsigned short>(half(d)));
}

// Batch-8 MoE down-proj + weighted sum -> acc[] directly (no zero kernel).
// warp-per-2-rows over 18 units, expert loop in-order (CPU oracle order).
// A.nout = kNE outputs; grid.x = ceil(nout / 16) of 256.
kernel void k_gemv_down_reduce8(device float * SCR [[buffer(0)]],
                                constant MoeDownArgs & A [[buffer(1)]],
                                const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint rbase = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (rbase >= A.nout) return;
    uint lane = tid & 31u;
    uint rb = lane & 1u;
    uint r = rbase + rb;
    const device uchar4 * xq = (const device uchar4 *)(SCR + A.xqoff);
    const device float * dhb = SCR + A.dxoff;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint e = 0u; e < 8u; e++) {
        const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
        const float w = SCR[A.wtoff + e];
        const device uint * wr = (const device uint *)(PACK + A.woff + (size_t)exp_id * A.stride) + (size_t)r * 306u;
        float ea = 0.0f, eb = 0.0f;
        float & eacc = rb ? eb : ea;   // even lanes: row rbase, odd lanes: rbase+1
        for (uint u = (lane >> 1); u < 18u; u += 16u)
            gemv_unit(wr + u * 17u, xq + e * 288u + u * 16u, dhb + e * 36u, u, eacc);
        ea = simd_sum(ea); eb = simd_sum(eb);
        if (lane == 0u) acc0 = acc0 + w * ea;   // oracle: acc = acc + w*dbuf (contract off)
        if (lane == 1u) acc1 = acc1 + w * eb;
    }
    if (lane == 0u) SCR[A.accoff + rbase] = acc0;
    if (lane == 1u && rbase + 1u < A.nout) SCR[A.accoff + rbase + 1u] = acc1;
}

// rmsnorm (tree reduction, bitwise-identical to k_rms) fused with Q8_0
// quant. xn[] staging holds the normalized row (k <= 2304 = M64_NE).
kernel void k_rms_quant(device float * SCR [[buffer(0)]],
                        constant RmsQuantArgs & A [[buffer(1)]],
                        const device uchar * PACK [[buffer(2)]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_NPTG [[threads_per_threadgroup]]) {
    uint row = M64_TGPG.x;
    if (row >= A.rows) return;
    const device float * x = SCR + A.xr[row];
    const device float * w = A.hasw ? ((const device float *)(PACK + A.wr[row])) : nullptr;
    device float * y = A.yr[row] != M64_NOFFU ? (SCR + A.yr[row]) : nullptr;
    uint tid = M64_TITG.x;
    const uint nth = M64_NPTG.x;
    threadgroup float red[256];
    threadgroup float xn[2304];
    // R4: must mirror k_rms EXACTLY (same vec4_in predicate and float4
    // accumulate shape) so s / inv stay bitwise-identical between the two
    // kernels — vec4_in depends only on k and xr[row].
    const bool vec4_in = ((A.k & 3u) == 0u) && ((A.xr[row] & 3u) == 0u);
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
    // SIMD reduction — keep EXACTLY the same code shape as k_rms so the two
    // kernels stay bitwise-identical (parity test 26 checks bitwise).
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
    for (uint i = tid; i < A.k; i += nth) {
        float v = w ? (x[i] * inv * w[i]) : (x[i] * inv);
        xn[i] = v;
        if (y) y[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (A.q[row] == M64_NOFFU) return;
    const uint nb = A.k >> 5;
    uint sg = tid >> 5, lane = tid & 31u;
    const uint nsg = nth >> 5;
    device uchar * q = (device uchar *)(SCR + A.q[row]);
    for (uint b = sg; b < nb; b += nsg) {
        float v = xn[b * 32u + lane];
        const float amax = simd_max(fabs(v));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        int r0 = int(rint(v * id));
        r0 = r0 > 127 ? 127 : (r0 < -127 ? -127 : r0);
        q[b * 32u + lane] = uchar(char(r0));
        if (lane == 0u) SCR[A.d[row] + b] = f16b(as_type<unsigned short>(half(d)));
    }
}

// gate *= sigmoid(gate) fused with Q8_0 quant (3072 = 96 blocks).
kernel void k_gate_mul_quant(device float * SCR [[buffer(0)]],
                             constant GateMulQuantArgs & A [[buffer(1)]], uint3 M64_TIG [[thread_position_in_grid]]) {
    uint tid = M64_TIG.x;
    if (tid >= A.nb * 32u) return;
    uint b = tid >> 5, lane = tid & 31u;
    float v = SCR[A.aoff + tid] * ksig(SCR[A.goff + tid]);
    const float amax = simd_max(fabs(v));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int r0 = int(rint(v * id));
    r0 = r0 > 127 ? 127 : (r0 < -127 ? -127 : r0);
    ((device uchar *)(SCR + A.qoff))[tid] = uchar(char(r0));
    if (lane == 0u) SCR[A.doff + b] = f16b(as_type<unsigned short>(half(d)));
}

// ---------------------------------------------------------------------------
// Phase-2A-opt ROUND 2
// ---------------------------------------------------------------------------

// 2 rows/warp (k = 3072), XA/XB register-reuse via gemv_w — bitwise-identical
// accumulation order to k_gemv_q8_k3072. grid.x = ceil(nout / 16) of 256.
kernel void k_gemv_q8_k3072_m2(device float * SCR [[buffer(0)]],
                               constant GemvArgs & A [[buffer(1)]],
                               const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint row0 = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (row0 >= A.nout) return;
    uint lane = tid & 31u;
    const device uint * base = (const device uint *)(PACK + A.woff + (size_t)A.idx * A.stride);
    uint row1 = row0 + 1u < A.nout ? row0 + 1u : row0;   // clamp tail, guard write
    float a0 = 0.0f, a1 = 0.0f;
    for (uint u = lane; u < 48u; u += 32u) {
        const device uchar4 * xup = (const device uchar4 *)(SCR + A.xqoff) + u * 16u;
        uchar4 XA[8], XB[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) { XA[j] = xup[j]; XB[j] = xup[8 + j]; }
        float dx0 = SCR[A.dxoff + 2 * u], dx1 = SCR[A.dxoff + 2 * u + 1u];
        gemv_w(base + (size_t)row0 * 816u + u * 17u, XA, XB, dx0, dx1, a0);
        gemv_w(base + (size_t)row1 * 816u + u * 17u, XA, XB, dx0, dx1, a1);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0) SCR[A.yoff + row0] = a0;
    if (lane == 1 && row0 + 1u < A.nout) SCR[A.yoff + row0 + 1u] = a1;
}

// Batch-8 MoE gate|up with 2 rows/warp: adjacent rows of one (expert, half)
// share XA/XB registers. Pair index rr in [0, 8*nout):
//   e   = rr / nout                       (expert slot 0..7)
//   rem = rr - e * nout                   (0..nout-1; nout/2 pairs per half)
//   up  = rem >= nout/2;  r0 = 2*(rem - (up ? nout/2 : 0))
// nout even (kNF = 1152) pairs cleanly; odd tails clamp reads and guard the
// r0+1 write. grid.x = ceil(8 * nout / 8) of 256 (nout=1152 -> 1152 TGs).
kernel void k_gemv_gateup_batch8_m2(device float * SCR [[buffer(0)]],
                                    constant GemvArgs & A [[buffer(1)]],
                                    const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint rr = M64_TGPG.x * 8u + (tid >> 5);
    if (rr >= 8u * A.nout) return;
    uint lane = tid & 31u;
    uint nhalf = A.nout >> 1;
    uint e = rr / A.nout;
    uint rem = rr - e * A.nout;
    bool up = rem >= nhalf;
    uint r0 = (up ? rem - nhalf : rem) * 2u;
    uint r1 = r0 + 1u < A.nout ? r0 + 1u : r0;
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
    const device uint * base = (const device uint *)(PACK + (up ? A.woff2 : A.woff) + (size_t)exp_id * A.stride);
    float a0 = 0.0f, a1 = 0.0f;
    for (uint u = lane; u < 36u; u += 32u) {
        const device uchar4 * xup = (const device uchar4 *)(SCR + A.xqoff) + u * 16u;
        uchar4 XA[8], XB[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) { XA[j] = xup[j]; XB[j] = xup[8 + j]; }
        float dx0 = SCR[A.dxoff + 2 * u], dx1 = SCR[A.dxoff + 2 * u + 1u];
        gemv_w(base + (size_t)r0 * 612u + u * 17u, XA, XB, dx0, dx1, a0);
        gemv_w(base + (size_t)r1 * 612u + u * 17u, XA, XB, dx0, dx1, a1);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    device float * y = SCR + (up ? A.yoff2 : A.yoff) + e * A.nout;
    if (lane == 0) y[r0] = a0;
    if (lane == 1 && r0 + 1u < A.nout) y[r0 + 1u] = a1;
}

// Batch-8 MoE down-proj, 4 rows/warp, single 18-unit pass (vs 2x16-lane passes
// in k_gemv_down_reduce8): lanes 0..17 hold unit u=lane activations in
// registers and run gemv_w over rows rbase..rbase+3 for each expert; lanes
// 0..3 own the row accumulators (CPU oracle order e = 0..7, plain mul+add).
// grid.x = ceil(nout / 32) of 256.
kernel void k_gemv_down_reduce8_m4(device float * SCR [[buffer(0)]],
                                   constant MoeDownArgs & A [[buffer(1)]],
                                   const device uchar * PACK [[buffer(2)]], uint3 M64_TITG [[thread_position_in_threadgroup]], uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint rbase = (M64_TGPG.x * 8u + (tid >> 5)) * 4u;
    if (rbase >= A.nout) return;
    uint lane = tid & 31u;
    const device uchar4 * xq = (const device uchar4 *)(SCR + A.xqoff);
    const device float * dhb = SCR + A.dxoff;
    uint r1 = rbase + 1u < A.nout ? rbase + 1u : rbase;
    uint r2 = rbase + 2u < A.nout ? rbase + 2u : rbase;
    uint r3 = rbase + 3u < A.nout ? rbase + 3u : rbase;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint e = 0u; e < 8u; e++) {
        const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
        const float w = SCR[A.wtoff + e];
        const device uint * eb = (const device uint *)(PACK + A.woff + (size_t)exp_id * A.stride);
        float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
        if (lane < 18u) {
            const device uchar4 * xup = xq + e * 288u + lane * 16u;
            uchar4 XA[8], XB[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) { XA[j] = xup[j]; XB[j] = xup[8 + j]; }
            float dx0 = dhb[e * 36u + 2 * lane], dx1 = dhb[e * 36u + 2 * lane + 1u];
            const device uint * wu = eb + lane * 17u;
            gemv_w(wu + (size_t)rbase * 306u, XA, XB, dx0, dx1, e0);
            gemv_w(wu + (size_t)r1 * 306u, XA, XB, dx0, dx1, e1);
            gemv_w(wu + (size_t)r2 * 306u, XA, XB, dx0, dx1, e2);
            gemv_w(wu + (size_t)r3 * 306u, XA, XB, dx0, dx1, e3);
        }
        e0 = simd_sum(e0); e1 = simd_sum(e1); e2 = simd_sum(e2); e3 = simd_sum(e3);
        if (lane == 0u) a0 = a0 + w * e0;   // oracle: acc = acc + w*dbuf
        if (lane == 1u) a1 = a1 + w * e1;
        if (lane == 2u) a2 = a2 + w * e2;
        if (lane == 3u) a3 = a3 + w * e3;
    }
    if (lane == 0u) SCR[A.accoff + rbase] = a0;
    if (lane == 1u && rbase + 1u < A.nout) SCR[A.accoff + rbase + 1u] = a1;
    if (lane == 2u && rbase + 2u < A.nout) SCR[A.accoff + rbase + 2u] = a2;
    if (lane == 3u && rbase + 3u < A.nout) SCR[A.accoff + rbase + 3u] = a3;
}

// ---------------------------------------------------------------------------
// Phase-2A-opt R3: register multi_m2 fusion, 2-stage MoE down+conv, fused mix
// ---------------------------------------------------------------------------

// Fused 1..6 Q8_0 k=2304 GEMVs sharing (xqoff, dxoff), zero TG SRAM (_m2
// register reuse of XA/XB across pairs of rows).
// grid.x = ceil(total / 16), threadsPerThreadgroup = 256
kernel void k_gemv_q8_k2304_multi_m2(device float * SCR [[buffer(0)]],
                                     constant MultiGemvArgs & A [[buffer(1)]],
                                     const device uchar * PACK [[buffer(2)]],
                                     uint3 M64_TITG [[thread_position_in_threadgroup]],
                                     uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint g0 = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (g0 >= A.total) return;
    uint lane = tid & 31u;
    uint g1 = (g0 + 1u < A.total) ? (g0 + 1u) : g0;
    uint p0 = 0u, p1 = 0u;
    for (uint i = 0u; i + 1u < A.count; i++) {
        if (g0 >= A.cum[i]) p0 = i + 1u;
        if (g1 >= A.cum[i]) p1 = i + 1u;
    }
    uint r0 = g0 - (p0 ? A.cum[p0 - 1u] : 0u);
    uint r1 = g1 - (p1 ? A.cum[p1 - 1u] : 0u);
    const device uint * base0 = (const device uint *)(PACK + A.woff[p0]) + (size_t)r0 * 612u;
    const device uint * base1 = (const device uint *)(PACK + A.woff[p1]) + (size_t)r1 * 612u;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint u = lane; u < 36u; u += 32u) {
        const device uchar4 * xup = (const device uchar4 *)(SCR + A.xqoff) + u * 16u;
        uchar4 XA[8], XB[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) { XA[j] = xup[j]; XB[j] = xup[8 + j]; }
        float dx0 = SCR[A.dxoff + 2u * u], dx1 = SCR[A.dxoff + 2u * u + 1u];
        gemv_w(base0 + u * 17u, XA, XB, dx0, dx1, a0);
        gemv_w(base1 + u * 17u, XA, XB, dx0, dx1, a1);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) SCR[A.yoff[p0] + r0] = a0;
    if (lane == 1u && g0 + 1u < A.total) SCR[A.yoff[p1] + r1] = a1;
}

// Stage 1 of MoE down-proj: 2D grid (ceil(nout/16), 8) -> 1,152 TGs of 256 for
// kNE=2304, saturating all 40 cores with 2-row register reuse. Writes per-expert
// outputs into SCR[A.accoff + e * A.nout + row].
kernel void k_gemv_down_batch8_m2(device float * SCR [[buffer(0)]],
                                   constant MoeDownArgs & A [[buffer(1)]],
                                   const device uchar * PACK [[buffer(2)]],
                                   uint3 M64_TITG [[thread_position_in_threadgroup]],
                                   uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint e = M64_TGPG.y;
    uint tid = M64_TITG.x;
    uint r0 = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (r0 >= A.nout) return;
    uint lane = tid & 31u;
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
    uint r1 = (r0 + 1u < A.nout) ? (r0 + 1u) : r0;
    const device uint * base = (const device uint *)(PACK + A.woff + (size_t)exp_id * A.stride);
    const device uchar4 * xq = (const device uchar4 *)(SCR + A.xqoff) + e * 288u;
    const device float * dh = SCR + A.dxoff + e * 36u;
    float a0 = 0.0f, a1 = 0.0f;
    if (lane < 18u) {
        const device uchar4 * xup = xq + lane * 16u;
        uchar4 XA[8], XB[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) { XA[j] = xup[j]; XB[j] = xup[8 + j]; }
        float dx0 = dh[2u * lane], dx1 = dh[2u * lane + 1u];
        gemv_w(base + (size_t)r0 * 306u + lane * 17u, XA, XB, dx0, dx1, a0);
        gemv_w(base + (size_t)r1 * 306u + lane * 17u, XA, XB, dx0, dx1, a1);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) SCR[A.accoff + e * A.nout + r0] = a0;
    if (lane == 1u && r0 + 1u < A.nout) SCR[A.accoff + e * A.nout + r1] = a1;
}

// Stage 2 of MoE down-proj: weighted sum of 8 experts + static ffnconv + cur += conv_out.
// Flat thread dispatch: total = nout (2304), threadsPerThreadgroup = 128.
kernel void k_down_reduce8_conv_add(device float * SCR [[buffer(0)]],
                                    constant DownReduceConvArgs & A [[buffer(1)]],
                                    const device uchar * PACK [[buffer(2)]],
                                    uint3 M64_TIG [[thread_position_in_grid]]) {
    uint c = M64_TIG.x;
    if (c >= A.nout) return;
    float xin = 0.0f;
    #pragma unroll
    for (uint e = 0u; e < 8u; e++)
        xin = xin + SCR[A.wtoff + e] * SCR[A.tmpoff + e * A.nout + c];
    device float * st = SCR + A.rsoff + c * 3u;
    const float s0 = st[0], s1 = st[1], s2 = st[2];
    const device float * w = ((const device float *)(PACK + A.wconv)) + c * 4u;
    float outv = s0 * w[0] + s1 * w[1] + s2 * w[2] + xin * w[3];
    st[0] = s1; st[1] = s2; st[2] = xin;
    SCR[A.curoff + c] = SCR[A.curoff + c] + (xin + outv) * A.scale;
}

// Fused state-mixer: dot(stn_i, q_j)*scale -> softmax -> out_j = sum_i wt_i * stn_i.
// Replaces k_mix_dot + k_mix_soft + k_mix_comb (3 dispatches -> 1 dispatch).
// grid.x = nq, threadsPerThreadgroup = 256.
kernel void k_mix_fused(device float * SCR [[buffer(0)]],
                        constant MixFusedArgs & A [[buffer(1)]],
                        const device uchar * PACK [[buffer(2)]],
                        uint3 M64_TITG [[thread_position_in_threadgroup]],
                        uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint j = M64_TGPG.x;
    uint tid = M64_TITG.x;
    uint lane = tid & 31u;
    uint sg = tid >> 5;
    threadgroup float red[8];
    threadgroup float wt[M64_MAXS];
    const bool vec4 = ((A.stnoff & 3u) == 0u) && ((A.outoff & 3u) == 0u) && ((A.qpack & 15ul) == 0ul);
    const device float * qp = ((const device float *)(PACK + A.qpack)) + (size_t)j * M64_NE;
    const device float4 * q4 = (const device float4 *)qp;
    for (uint i = 0u; i < A.S; i++) {
        float a = 0.0f;
        if (vec4) {
            const device float4 * s4 = (const device float4 *)(SCR + A.stnoff + (size_t)i * M64_NE);
            float4 av = float4(0.0f);
            for (uint k = tid; k < M64_NE / 4u; k += 256u) av += s4[k] * q4[k];
            a = (av.x + av.y) + (av.z + av.w);
        } else {
            const device float * sp = SCR + A.stnoff + (size_t)i * M64_NE;
            for (uint k = tid; k < M64_NE; k += 256u) a = fma(sp[k], qp[k], a);
        }
        a = simd_sum(a);
        if (lane == 0u) red[sg] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {
            float t = (lane < 8u) ? red[lane] : 0.0f;
            t = simd_sum(t);
            if (tid == 0u) wt[i] = t * A.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float mx = -INFINITY;
        for (uint i = 0u; i < A.S; i++) mx = wt[i] > mx ? wt[i] : mx;
        float sum = 0.0f;
        for (uint i = 0u; i < A.S; i++) { wt[i] = metal::exp(wt[i] - mx); sum += wt[i]; }
        float recip = (float)(1.0 / sum);
        for (uint i = 0u; i < A.S; i++) wt[i] = wt[i] * recip;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (vec4) {
        device float4 * out4 = (device float4 *)(SCR + A.outoff + (size_t)j * M64_NE);
        for (uint k = tid; k < M64_NE / 4u; k += 256u) {
            float4 a = float4(0.0f);
            for (uint i = 0u; i < A.S; i++)
                a = fma(float4(wt[i]), ((const device float4 *)(SCR + A.stnoff + (size_t)i * M64_NE))[k], a);
            out4[k] = a;
        }
    } else {
        device float * out = SCR + A.outoff + (size_t)j * M64_NE;
        for (uint k = tid; k < M64_NE; k += 256u) {
            float a = 0.0f;
            for (uint i = 0u; i < A.S; i++)
                a = fma(wt[i], SCR[A.stnoff + (size_t)i * M64_NE + k], a);
            out[k] = a;
        }
    }
}

// --------------------------------------------------- R4 split control block
// k_ctrl_down: grid (64, nc) x 256 threads. One wdown row r per threadgroup;
// writes hh[cg][r] = tanh(dot(wdown_row, rms'd gate pair cat)).
kernel void k_ctrl_down(device float * SCR [[buffer(0)]],
                        constant CtrlSplitArgs & A [[buffer(1)]],
                        const device uchar * PACK [[buffer(2)]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_TITG [[thread_position_in_threadgroup]]) {
    uint r = M64_TGPG.x;
    uint cg = M64_TGPG.y;
    uint tid = M64_TITG.x;
    threadgroup float red[8];
    const uint nk = 2u * A.ne;
    const device float * wr = ((const device float *)(PACK + A.wdown[cg])) + (size_t)r * nk;
    const device float * gc = SCR + A.gcat + (size_t)cg * nk;
    float a;
    if ((A.ne & 1u) == 0u && (A.gcat & 3u) == 0u && (A.wdown[cg] & 15ul) == 0ul) {
        const device float4 * w4 = (const device float4 *)wr;
        const device float4 * g4 = (const device float4 *)gc;
        float4 av = float4(0.0f);
        for (uint k = tid; k < nk / 4u; k += 256u) av += w4[k] * g4[k];
        a = (av.x + av.y) + (av.z + av.w);
    } else {
        a = 0.0f;
        for (uint k = tid; k < nk; k += 256u) a = fma(wr[k], gc[k], a);
    }
    // 256 threads = 8 warps: simd_sum -> red[8] -> warp-0 simd_sum.
    a = simd_sum(a);
    if ((tid & 31u) == 0u) red[tid >> 5u] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float t = ((tid & 31u) < 8u) ? red[tid & 31u] : 0.0f;
        t = simd_sum(t);
        if (tid == 0u) SCR[A.hh + cg * 64u + r] = tanh(t);
    }
}

// k_ctrl_up: grid (ceil(ne/32), nc) x 32 threads (1 warp/TG). One hh[64]
// broadcast via threadgroup memory; each lane takes one output row.
kernel void k_ctrl_up(device float * SCR [[buffer(0)]],
                      constant CtrlSplitArgs & A [[buffer(1)]],
                      const device uchar * PACK [[buffer(2)]], uint3 M64_TGPG [[threadgroup_position_in_grid]], uint3 M64_TITG [[thread_position_in_threadgroup]]) {
    uint row = M64_TGPG.x * 32u + M64_TITG.x;
    uint cg = M64_TGPG.y;
    uint tid = M64_TITG.x;
    threadgroup float hh[64];
    hh[tid] = SCR[A.hh + cg * 64u + tid];
    hh[tid + 32u] = SCR[A.hh + cg * 64u + tid + 32u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= A.ne) return;
    const device float * wg = ((const device float *)(PACK + A.wgain[cg])) + (size_t)row * 64u;
    const device float * wf = ((const device float *)(PACK + A.wforg[cg])) + (size_t)row * 64u;
    float gs, fs;
    if ((A.wgain[cg] & 15ul) == 0ul && (A.wforg[cg] & 15ul) == 0ul) {
        const device float4 * wg4 = (const device float4 *)wg;
        const device float4 * wf4 = (const device float4 *)wf;
        float4 gv = float4(0.0f), fv = float4(0.0f);
        #pragma unroll
        for (uint t = 0u; t < 16u; t++) {
            float4 hv = float4(hh[4u * t], hh[4u * t + 1u], hh[4u * t + 2u], hh[4u * t + 3u]);
            gv += wg4[t] * hv;
            fv += wf4[t] * hv;
        }
        gs = (gv.x + gv.y) + (gv.z + gv.w);
        fs = (fv.x + fv.y) + (fv.z + fv.w);
    } else {
        gs = 0.0f; fs = 0.0f;
        #pragma unroll
        for (uint t = 0u; t < 64u; t++) { float hv = hh[t]; gs = fma(wg[t], hv, gs); fs = fma(wf[t], hv, fs); }
    }
    float g = tanh(gs), fg = tanh(fs);
    float c = SCR[A.in[cg] + row];
    float m = SCR[A.msg[cg] + row];
    SCR[A.out[cg] + row] = (c - fg * c) + g * m;
}

// ============================ Milestone 4B: decode Q4_K / Q5_0 / Q5_1 ============================
// FP32-input MoE decode GEMVs (no k_quant pass). Bit layouts verified against
// engine/src/dequant.hpp dequantize_row_q4_K_ref / q5_0_ref / q5_1_ref:
//  * BlockQ4_K (144 B): d f16 @0, dmin f16 @2, scales[12] @4, qs[128] @16.
//    get_scale_min_k4(j in 0..7) over the scales; 64-elem chunk c uses
//    (2c, 2c+1): elems 0..31 = low nibbles of qs bytes [c*32 .. c*32+31],
//    elems 32..63 = high nibbles of the SAME bytes (dequant.hpp:102-104).
//  * BlockQ5_0 (22 B): d f16 @0, qh[4] @2, qs[16] @6. elem j (0..15):
//    (qs[j]&0xF | bit j of qh <<4) - 16; elem 16+j: (qs[j]>>4 | bit 16+j <<4)
//    - 16 (dequant.hpp:117-124). BlockQ5_1 (24 B): d @0, m f16 @2, qh @4,
//    qs @8; x in 0..31, y = x*d + m (dequant.hpp:137-144).
// All Q5 reads are byte / ushort little-endian loads: Q5_0 blocks are 22 B so
// their 4-byte members are NOT reliably aligned — never cast them to uint.
// Q4_K qs words load as device uint: block offsets 16+32c+4t stay 4-byte
// aligned iff (woff | stride) % 4 == 0 (guaranteed by pack 4 KiB alignment).

static inline ushort m64_ld16(const device uchar * p) { return ushort(p[0]) | (ushort(p[1]) << 8); }
static inline uint   m64_ld32(const device uchar * p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

// ggml get_scale_min_k4 word-packed: q = scales[0..3] (lo byte = scales[0]),
// q2 = scales[4..7], q3 = scales[8..11]; j in 0..7. Byte-for-byte port of
// dequant.hpp:80-87:
//   j < 4 : d = scales[j] & 63        ; m = scales[j + 4] & 63
//   j >= 4: d = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//           m = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
static inline void m64_scale_min_k4(uint j, uint q, uint q2, uint q3, float & d, float & m) {
    if (j < 4u) {
        d = float((q  >> (8u * j)) & 63u);
        m = float((q2 >> (8u * j)) & 63u);
    } else {
        uint b_jm4 = (q  >> (8u * (j - 4u))) & 255u;   // scales[j - 4]
        uint b_j   = (q2 >> (8u * (j - 4u))) & 255u;   // scales[j]
        uint b_jp4 = (q3 >> (8u * (j - 4u))) & 255u;   // scales[j + 4]
        d = float((b_jp4 & 0xFu) | ((b_jm4 >> 6) << 4));
        m = float((b_jp4 >> 4)   | ((b_j   >> 6) << 4));
    }
}

// Batch-8 MoE gate|up over FP32 activations, Q4_K weights. Warp per row,
// warp-w iteration order identical to k_gemv_gateup_batch8 (bitwise layout
// swap-compatible); inside a row the 36 64-elem sub-blocks map lane->sub-block
// (x float4 loads coalesce warp-wide, weights read once). Cross-lane simd_sum
// tree: tolerance class (1024-ulp fp64-gated test), not bitwise vs scalar.
kernel void k_gemv_gateup_batch8_q4k_f32in(device float * SCR [[buffer(0)]],
                                           constant MoeGateUpQ4KF32Args & A [[buffer(1)]],
                                           const device uchar * PACK [[buffer(2)]],
                                           uint3 M64_TITG [[thread_position_in_threadgroup]],
                                           uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    uint rr = M64_TGPG.x * 8u + (tid >> 5);
    if (rr >= 8u * 2u * A.nout) return;
    uint lane = tid & 31u;
    uint e = rr / (2u * A.nout);
    uint r2 = rr - e * (2u * A.nout);
    bool up = r2 >= A.nout;
    uint r = up ? r2 - A.nout : r2;
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
    const uint slot = A.wtab_off != 0u ? ((const device uint *)(SCR + A.wtab_off))[exp_id] : exp_id;
    const device uchar * row = PACK + (up ? A.woff2 : A.woff) + (size_t)slot * A.stride
                             + (size_t)r * ((A.k >> 8) * 144u);
    const device float * x = SCR + A.xoff;
    float a = 0.0f;
    for (uint u = lane; u < A.k / 64u; u += 32u) {   // 64-elem sub-chunks (k = 2304 -> 36)
        const device uchar * b = row + (u >> 2) * 144u;
        const uint q = m64_ld32(b + 4u), q2 = m64_ld32(b + 8u), q3 = m64_ld32(b + 12u);
        float d1, m1, d2, m2;
        m64_scale_min_k4(2u * (u & 3u), q, q2, q3, d1, m1);
        m64_scale_min_k4(2u * (u & 3u) + 1u, q, q2, q3, d2, m2);
        const float vd1 = f16b(m64_ld16(b)) * d1, vm1 = f16b(m64_ld16(b + 2u)) * m1;
        const float vd2 = f16b(m64_ld16(b)) * d2, vm2 = f16b(m64_ld16(b + 2u)) * m2;
        const device uint * qs = (const device uint *)(b + 16u + (u & 3u) * 32u);
        const device float4 * xv = (const device float4 *)(x + u * 64u);
        // uint t holds qs bytes 4t..4t+3 (LE): byte u low nibble = elem 4t+u
        // -> xv[t]; byte u high nibble = elem 32+4t+u -> xv[8+t] (dequant.hpp:102-104).
        // Element value w = vd*q - vm, dot term fma(w, x): two chained fmas keep
        // every product multiplied by x (no d*q term without x — that was wrong).
        #pragma unroll
        for (uint t = 0u; t < 8u; t++) {
            const uint v = qs[t];
            const float4 xlo = xv[t], xhi = xv[8u + t];
            a = fma(fma(vd1, float(v & 0xFu), -vm1), xlo.x, a);
            a = fma(fma(vd1, float((v >> 8) & 0xFu), -vm1), xlo.y, a);
            a = fma(fma(vd1, float((v >> 16) & 0xFu), -vm1), xlo.z, a);
            a = fma(fma(vd1, float((v >> 24) & 0xFu), -vm1), xlo.w, a);
            a = fma(fma(vd2, float((v >> 4) & 0xFu), -vm2), xhi.x, a);
            a = fma(fma(vd2, float((v >> 12) & 0xFu), -vm2), xhi.y, a);
            a = fma(fma(vd2, float((v >> 20) & 0xFu), -vm2), xhi.z, a);
            a = fma(fma(vd2, float(v >> 28), -vm2), xhi.w, a);
        }
    }
    a = simd_sum(a);
    if (lane == 0u) SCR[(up ? A.yoff2 : A.yoff) + e * A.nout + r] = a;
}

// Shared inner: one warp-GEMV of Q5_0/Q5_1 row at wp (nb blocks) vs h.
// Lane owns blocks i = lane, lane+32, ...; 4 (u32-paired) ushort qs loads plus
// byte-wise d/qh per block (Q5_0's 22 B blocks are only 2-byte aligned by
// construction). `lane` must equal thread_index_in_simdgroup() of the caller.
template <bool Q51>
static inline float m64_gemv_q5_row(const device uchar * wp, const device float * h,
                                    uint nb, uint lane) {
    float a = 0.0f;
    for (uint i = lane; i < nb; i += 32u) {
        const device uchar * b = wp + (size_t)i * (Q51 ? 24u : 22u);
        const uint qh = Q51 ? m64_ld32(b + 4u) : m64_ld32(b + 2u);
        const float d = f16b(m64_ld16(b));
        const float m = Q51 ? f16b(m64_ld16(b + 2u)) : 0.0f;
        const ushort * qs = (const ushort *)(b + (Q51 ? 8u : 6u));   // even offsets: aligned
        const device float4 * hlo = (const device float4 *)(h + i * 32u);
        const device float4 * hhi = hlo + 4u;
        // uint t = qs bytes 4t..4t+3 (LE ushort pair): byte jj=4t+u low nibble
        // = elem jj (qh bit jj) -> hlo[t].comp[u]; high nibble = elem 16+jj
        // (qh bit jj+16) -> hhi[t].comp[u] (dequant.hpp:117-124).
        #pragma unroll
        for (uint t = 0u; t < 4u; t++) {
            const uint v = uint(qs[2u * t]) | (uint(qs[2u * t + 1u]) << 16);
            const float4 vlo = hlo[t], vhi = hhi[t];
            #pragma unroll
            for (uint u = 0u; u < 4u; u++) {
                const uint jj = 4u * t + u;
                float x0 = float(((v >> (8u * u)) & 0xFu) | (((qh >> jj) << 4) & 0x10u));
                // high nibble element 16+jj takes qh bit jj+16 == (qh >> (jj+12)) & 0x10 (dequant.hpp:119)
                float x1 = float(((v >> (8u * u + 4u)) & 0xFu) | ((qh >> (jj + 12u)) & 0x10u));
                if (!Q51) { x0 -= 16.0f; x1 -= 16.0f; }
                float s0 = d * x0, s1 = d * x1;
                if (Q51) { s0 += m; s1 += m; }
                if (u == 0u) { a = fma(s0, vlo.x, a); a = fma(s1, vhi.x, a); }
                if (u == 1u) { a = fma(s0, vlo.y, a); a = fma(s1, vhi.y, a); }
                if (u == 2u) { a = fma(s0, vlo.z, a); a = fma(s1, vhi.z, a); }
                if (u == 3u) { a = fma(s0, vlo.w, a); a = fma(s1, vhi.w, a); }
            }
        }
    }
    return simd_sum(a);
}

// Batch-8 MoE down-proj + weighted reduce, Q5_0 weights, FP32 activations.
// Warp per row; the FULL warp cooperates on each expert dot (m64_gemv_q5_row
// is a warp collective), expert loop e-ascending plain mul+add
// (k_gemv_down_reduce8 oracle order). Activations are PER-EXPERT rows: expert
// e reads SCR[hoff + e*k .. hoff + (e+1)*k) (k_situ_f32_batch emits a packed
// [8][k] h slab with row stride k). grid.x = ceil(nout / 8) tgs of 256.
kernel void k_gemv_down_reduce8_q5_0_f32in(device float * SCR [[buffer(0)]],
                                           constant MoeDownQ5F32Args & A [[buffer(1)]],
                                           const device uchar * PACK [[buffer(2)]],
                                           uint3 M64_TITG [[thread_position_in_threadgroup]],
                                           uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint r = M64_TGPG.x * 8u + (M64_TITG.x >> 5);
    if (r >= A.nout) return;
    uint lane = M64_TITG.x & 31u;
    const device float * h = SCR + A.hoff;
    const uint nb = A.k >> 5;
    const size_t rowb = (size_t)nb * 22ull;
    float acc = 0.0f;
    for (uint e = 0u; e < 8u; e++) {
        const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
        const uint slot = A.wtab_off != 0u ? ((const device uint *)(SCR + A.wtab_off))[exp_id] : exp_id;
        const device uchar * wp = PACK + A.woff + (size_t)slot * A.stride + (size_t)r * rowb;
        const float ea = m64_gemv_q5_row<false>(wp, h + (size_t)e * A.k, nb, lane);
        acc = acc + SCR[A.wtoff + e] * ea;              // oracle: acc + w*dbuf
    }
    if (lane == 0u) SCR[A.accoff + r] = acc;
}

// Same with Q5_1 blocks (24 B, shared min-offset m; no -16 bias).
kernel void k_gemv_down_reduce8_q5_1_f32in(device float * SCR [[buffer(0)]],
                                           constant MoeDownQ5F32Args & A [[buffer(1)]],
                                           const device uchar * PACK [[buffer(2)]],
                                           uint3 M64_TITG [[thread_position_in_threadgroup]],
                                           uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint r = M64_TGPG.x * 8u + (M64_TITG.x >> 5);
    if (r >= A.nout) return;
    uint lane = M64_TITG.x & 31u;
    const device float * h = SCR + A.hoff;
    const uint nb = A.k >> 5;
    const size_t rowb = (size_t)nb * 24ull;
    float acc = 0.0f;
    for (uint e = 0u; e < 8u; e++) {
        const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
        const uint slot = A.wtab_off != 0u ? ((const device uint *)(SCR + A.wtab_off))[exp_id] : exp_id;
        const device uchar * wp = PACK + A.woff + (size_t)slot * A.stride + (size_t)r * rowb;
        const float ea = m64_gemv_q5_row<true>(wp, h + (size_t)e * A.k, nb, lane);
        acc = acc + SCR[A.wtoff + e] * ea;
    }
    if (lane == 0u) SCR[A.accoff + r] = acc;
}

// ---------------------------------------------------------------------------
// Milestone 7A: FP32-input MoE decode kernels, Q8 _m2/_m4 activation-sharing
// pattern ported to Q4_K / Q5_0 / Q5_1. Every warp of the warp-per-row
// kernels re-reads the same fp32 activations from SCR; pairing 2 gate/up rows
// per warp halves that traffic and 4 down rows per warp quarters it. m4
// kernels keep the exact m1 FMA order (bitwise, tests 49/50); the m2 kernels
// use Milestone 13 factored 2-row inner loops (tolerance-class, tests 47/48).
// ---------------------------------------------------------------------------

// Milestone 13: factored 2-row Q4_K sub-block dot. Row 0 (and row 1 when
// has_r1) share the x float4 loads and the x row-sums; raw nibble dots are
// accumulated per half and the (d*scale, dm*min) factors are applied once.
// Reorders FMAs vs m64_q4k_sub_dot (tolerance-class vs m1, test 47/48).
// 32-bit header loads are aligned: Q4_K blocks are 144 B and all pack
// offsets/strides are 4 B aligned (pack is 4 KiB aligned).
static inline void m64_q4k_sub_dot2(const device uchar * b0, const device uchar * b1,
                                    bool has_r1, uint u, const device float4 * xv,
                                    float & a0, float & a1) {
    const device uint * u0 = (const device uint *)b0;
    const uint h0 = u0[0];
    const float d0 = f16b(ushort(h0 & 0xFFFFu)), dm0 = f16b(ushort(h0 >> 16));
    float q1d, q1m, q2d, q2m;
    m64_scale_min_k4(2u * (u & 3u), u0[1], u0[2], u0[3], q1d, q1m);
    m64_scale_min_k4(2u * (u & 3u) + 1u, u0[1], u0[2], u0[3], q2d, q2m);
    const float vd1_0 = d0 * q1d, vm1_0 = dm0 * q1m;
    const float vd2_0 = d0 * q2d, vm2_0 = dm0 * q2m;
    float vd1_1 = 0.0f, vm1_1 = 0.0f, vd2_1 = 0.0f, vm2_1 = 0.0f;
    if (has_r1) {
        const device uint * u1 = (const device uint *)b1;
        const uint h1 = u1[0];
        const float d1 = f16b(ushort(h1 & 0xFFFFu)), dm1 = f16b(ushort(h1 >> 16));
        m64_scale_min_k4(2u * (u & 3u), u1[1], u1[2], u1[3], q1d, q1m);
        m64_scale_min_k4(2u * (u & 3u) + 1u, u1[1], u1[2], u1[3], q2d, q2m);
        vd1_1 = d1 * q1d; vm1_1 = dm1 * q1m;
        vd2_1 = d1 * q2d; vm2_1 = dm1 * q2m;
    }
    const device uint * qs0 = (const device uint *)(b0 + 16u + (u & 3u) * 32u);
    const device uint * qs1 = (const device uint *)(b1 + 16u + (u & 3u) * 32u);
    float sq_lo0 = 0.0f, sq_hi0 = 0.0f, sq_lo1 = 0.0f, sq_hi1 = 0.0f;
    float sx_lo = 0.0f, sx_hi = 0.0f;
    #pragma unroll
    for (uint t = 0u; t < 8u; t++) {
        const float4 xlo = xv[t], xhi = xv[8u + t];
        sx_lo += (xlo.x + xlo.y) + (xlo.z + xlo.w);
        sx_hi += (xhi.x + xhi.y) + (xhi.z + xhi.w);
        const uint v0 = qs0[t];
        sq_lo0 = fma(float(v0 & 0xFu), xlo.x, sq_lo0);
        sq_lo0 = fma(float((v0 >> 8) & 0xFu), xlo.y, sq_lo0);
        sq_lo0 = fma(float((v0 >> 16) & 0xFu), xlo.z, sq_lo0);
        sq_lo0 = fma(float((v0 >> 24) & 0xFu), xlo.w, sq_lo0);
        sq_hi0 = fma(float((v0 >> 4) & 0xFu), xhi.x, sq_hi0);
        sq_hi0 = fma(float((v0 >> 12) & 0xFu), xhi.y, sq_hi0);
        sq_hi0 = fma(float((v0 >> 20) & 0xFu), xhi.z, sq_hi0);
        sq_hi0 = fma(float(v0 >> 28), xhi.w, sq_hi0);
        if (has_r1) {
            const uint v1 = qs1[t];
            sq_lo1 = fma(float(v1 & 0xFu), xlo.x, sq_lo1);
            sq_lo1 = fma(float((v1 >> 8) & 0xFu), xlo.y, sq_lo1);
            sq_lo1 = fma(float((v1 >> 16) & 0xFu), xlo.z, sq_lo1);
            sq_lo1 = fma(float((v1 >> 24) & 0xFu), xlo.w, sq_lo1);
            sq_hi1 = fma(float((v1 >> 4) & 0xFu), xhi.x, sq_hi1);
            sq_hi1 = fma(float((v1 >> 12) & 0xFu), xhi.y, sq_hi1);
            sq_hi1 = fma(float((v1 >> 20) & 0xFu), xhi.z, sq_hi1);
            sq_hi1 = fma(float(v1 >> 28), xhi.w, sq_hi1);
        }
    }
    a0 = fma(vd1_0, sq_lo0, fma(-vm1_0, sx_lo, a0));
    a0 = fma(vd2_0, sq_hi0, fma(-vm2_0, sx_hi, a0));
    if (has_r1) {
        a1 = fma(vd1_1, sq_lo1, fma(-vm1_1, sx_lo, a1));
        a1 = fma(vd2_1, sq_hi1, fma(-vm2_1, sx_hi, a1));
    }
}

// Batch-8 Q4_K gate|up, 2 rows/warp, shared x float4s. Pair index rr in
// [0, 8*2*pp): pp = ceil(nout/2) pairs per (expert, half);
//   e = rr / (2*pp); rem = rr - e*2*pp; up = rem >= pp; r0 = 2*(rem - pp*up).
// Odd tails clamp r1 = r0 (duplicate weight reads, guarded write).
// grid.x = ceil(16*pp / 8) of 256 (nout even -> 8*nout/8 = nout TGs... nout=1152).
kernel void k_gemv_gateup_batch8_q4k_f32in_m2(device float * SCR [[buffer(0)]],
                                              constant MoeGateUpQ4KF32Args & A [[buffer(1)]],
                                              const device uchar * PACK [[buffer(2)]],
                                              uint3 M64_TITG [[thread_position_in_threadgroup]],
                                              uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint tid = M64_TITG.x;
    const uint pp = (A.nout + 1u) >> 1u;
    uint rr = M64_TGPG.x * 8u + (tid >> 5);
    if (rr >= 16u * pp) return;
    uint lane = tid & 31u;
    uint e = rr / (2u * pp);
    uint rem = rr - e * (2u * pp);
    bool up = rem >= pp;
    uint r0 = (up ? rem - pp : rem) * 2u;
    uint r1 = r0 + 1u < A.nout ? r0 + 1u : r0;
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
    const uint slot = A.wtab_off != 0u ? ((const device uint *)(SCR + A.wtab_off))[exp_id] : exp_id;
    const size_t rstride = (size_t)(A.k >> 8) * 144u;
    const device uchar * wb = PACK + (up ? A.woff2 : A.woff) + (size_t)slot * A.stride;
    const device float * x = SCR + A.xoff;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint u = lane; u < A.k / 64u; u += 32u) {
        const device uchar * b0 = wb + (size_t)r0 * rstride + (u >> 2) * 144u;
        const device uchar * b1 = wb + (size_t)r1 * rstride + (u >> 2) * 144u;
        const device float4 * xv = (const device float4 *)(x + u * 64u);
        m64_q4k_sub_dot2(b0, b1, (r1 != r0), u, xv, a0, a1);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    device float * y = SCR + (up ? A.yoff2 : A.yoff) + e * A.nout;
    if (lane == 0u) y[r0] = a0;
    if (lane == 1u && r0 + 1u < A.nout) y[r1] = a1;
}

// One 32-elem Q5_0/Q5_1 block dot into `a`, verbatim order from
// m64_gemv_q5_row (x passed as vlo/vhi float4 pair for index t).
template <bool Q51>
static inline void m64_q5_block_dot(uint v, uint qh, float d, float m, uint t,
                                    const float4 & vlo, const float4 & vhi, float & a) {
    #pragma unroll
    for (uint u = 0u; u < 4u; u++) {
        const uint jj = 4u * t + u;
        float x0 = float(((v >> (8u * u)) & 0xFu) | (((qh >> jj) << 4) & 0x10u));
        float x1 = float(((v >> (8u * u + 4u)) & 0xFu) | ((qh >> (jj + 12u)) & 0x10u));
        if (!Q51) { x0 -= 16.0f; x1 -= 16.0f; }
        float s0 = d * x0, s1 = d * x1;
        if (Q51) { s0 += m; s1 += m; }
        if (u == 0u) { a = fma(s0, vlo.x, a); a = fma(s1, vhi.x, a); }
        if (u == 1u) { a = fma(s0, vlo.y, a); a = fma(s1, vhi.y, a); }
        if (u == 2u) { a = fma(s0, vlo.z, a); a = fma(s1, vhi.z, a); }
        if (u == 3u) { a = fma(s0, vlo.w, a); a = fma(s1, vhi.w, a); }
    }
}

// Four Q5_0/Q5_1 rows vs one h slab, h float4s loaded once per block.
// Lane block order (i = lane, lane+32, ...) and per-row FMA order match
// m64_gemv_q5_row exactly; simd_sum stays a per-row warp collective in the
// caller, so each ea is bitwise-equal to the m1 kernel's.
template <bool Q51>
static inline void m64_gemv_q5_rows4(const device uchar * w0, const device uchar * w1,
                                     const device uchar * w2, const device uchar * w3,
                                     const device float * h, uint nb, uint lane,
                                     float & a0, float & a1, float & a2, float & a3) {
    const uint blk = Q51 ? 24u : 22u;
    for (uint i = lane; i < nb; i += 32u) {
        const size_t bo = (size_t)i * blk;
        const device float4 * hlo = (const device float4 *)(h + (size_t)i * 32u);
        const device uchar * b0 = w0 + bo, * b1 = w1 + bo, * b2 = w2 + bo, * b3 = w3 + bo;
        const uint qh0 = Q51 ? m64_ld32(b0 + 4u) : m64_ld32(b0 + 2u);
        const uint qh1 = Q51 ? m64_ld32(b1 + 4u) : m64_ld32(b1 + 2u);
        const uint qh2 = Q51 ? m64_ld32(b2 + 4u) : m64_ld32(b2 + 2u);
        const uint qh3 = Q51 ? m64_ld32(b3 + 4u) : m64_ld32(b3 + 2u);
        const float d0 = f16b(m64_ld16(b0)), d1 = f16b(m64_ld16(b1));
        const float d2 = f16b(m64_ld16(b2)), d3 = f16b(m64_ld16(b3));
        const float m0 = Q51 ? f16b(m64_ld16(b0 + 2u)) : 0.0f;
        const float m1 = Q51 ? f16b(m64_ld16(b1 + 2u)) : 0.0f;
        const float m2 = Q51 ? f16b(m64_ld16(b2 + 2u)) : 0.0f;
        const float m3 = Q51 ? f16b(m64_ld16(b3 + 2u)) : 0.0f;
        const ushort * q0 = (const ushort *)(b0 + (Q51 ? 8u : 6u));
        const ushort * q1 = (const ushort *)(b1 + (Q51 ? 8u : 6u));
        const ushort * q2 = (const ushort *)(b2 + (Q51 ? 8u : 6u));
        const ushort * q3 = (const ushort *)(b3 + (Q51 ? 8u : 6u));
        #pragma unroll
        for (uint t = 0u; t < 4u; t++) {
            const float4 vlo = hlo[t], vhi = hlo[4u + t];   // shared across rows
            m64_q5_block_dot<Q51>(uint(q0[2u * t]) | (uint(q0[2u * t + 1u]) << 16), qh0, d0, m0, t, vlo, vhi, a0);
            m64_q5_block_dot<Q51>(uint(q1[2u * t]) | (uint(q1[2u * t + 1u]) << 16), qh1, d1, m1, t, vlo, vhi, a1);
            m64_q5_block_dot<Q51>(uint(q2[2u * t]) | (uint(q2[2u * t + 1u]) << 16), qh2, d2, m2, t, vlo, vhi, a2);
            m64_q5_block_dot<Q51>(uint(q3[2u * t]) | (uint(q3[2u * t + 1u]) << 16), qh3, d3, m3, t, vlo, vhi, a3);
        }
    }
}

// Batch-8 MoE down-proj + weighted reduce over Q5_0, 4 rows/warp (h rows in
// registers shared by the 4 output rows). grid.x = ceil(nout / 32) of 256;
// lanes 0..3 own rows rbase..rbase+3 (clamped tail reads, guarded writes).
// Expert loop / accumulator order identical to k_gemv_down_reduce8_q5_0_f32in.
kernel void k_gemv_down_reduce8_q5_0_f32in_m4(device float * SCR [[buffer(0)]],
                                              constant MoeDownQ5F32Args & A [[buffer(1)]],
                                              const device uchar * PACK [[buffer(2)]],
                                              uint3 M64_TITG [[thread_position_in_threadgroup]],
                                              uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint rbase = (M64_TGPG.x * 8u + (M64_TITG.x >> 5)) * 4u;
    if (rbase >= A.nout) return;
    uint lane = M64_TITG.x & 31u;
    const device float * h = SCR + A.hoff;
    const uint nb = A.k >> 5;
    const size_t rowb = (size_t)nb * 22ull;
    uint r1 = rbase + 1u < A.nout ? rbase + 1u : rbase;
    uint r2 = rbase + 2u < A.nout ? rbase + 2u : rbase;
    uint r3 = rbase + 3u < A.nout ? rbase + 3u : rbase;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint e = 0u; e < 8u; e++) {
        const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
        const uint slot = A.wtab_off != 0u ? ((const device uint *)(SCR + A.wtab_off))[exp_id] : exp_id;
        const device uchar * wp = PACK + A.woff + (size_t)slot * A.stride;
        float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
        m64_gemv_q5_rows4<false>(wp + (size_t)rbase * rowb, wp + (size_t)r1 * rowb,
                                 wp + (size_t)r2 * rowb, wp + (size_t)r3 * rowb,
                                 h + (size_t)e * A.k, nb, lane, e0, e1, e2, e3);
        e0 = simd_sum(e0); e1 = simd_sum(e1); e2 = simd_sum(e2); e3 = simd_sum(e3);
        const float w = SCR[A.wtoff + e];
        acc0 = acc0 + w * e0;   // oracle: acc = acc + w*dbuf
        acc1 = acc1 + w * e1;
        acc2 = acc2 + w * e2;
        acc3 = acc3 + w * e3;
    }
    if (lane == 0u) SCR[A.accoff + rbase] = acc0;
    if (lane == 1u && rbase + 1u < A.nout) SCR[A.accoff + rbase + 1u] = acc1;
    if (lane == 2u && rbase + 2u < A.nout) SCR[A.accoff + rbase + 2u] = acc2;
    if (lane == 3u && rbase + 3u < A.nout) SCR[A.accoff + rbase + 3u] = acc3;
}

// Same with Q5_1 blocks (24 B, shared min-offset m; no -16 bias).
kernel void k_gemv_down_reduce8_q5_1_f32in_m4(device float * SCR [[buffer(0)]],
                                              constant MoeDownQ5F32Args & A [[buffer(1)]],
                                              const device uchar * PACK [[buffer(2)]],
                                              uint3 M64_TITG [[thread_position_in_threadgroup]],
                                              uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint rbase = (M64_TGPG.x * 8u + (M64_TITG.x >> 5)) * 4u;
    if (rbase >= A.nout) return;
    uint lane = M64_TITG.x & 31u;
    const device float * h = SCR + A.hoff;
    const uint nb = A.k >> 5;
    const size_t rowb = (size_t)nb * 24ull;
    uint r1 = rbase + 1u < A.nout ? rbase + 1u : rbase;
    uint r2 = rbase + 2u < A.nout ? rbase + 2u : rbase;
    uint r3 = rbase + 3u < A.nout ? rbase + 3u : rbase;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint e = 0u; e < 8u; e++) {
        const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
        const uint slot = A.wtab_off != 0u ? ((const device uint *)(SCR + A.wtab_off))[exp_id] : exp_id;
        const device uchar * wp = PACK + A.woff + (size_t)slot * A.stride;
        float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
        m64_gemv_q5_rows4<true>(wp + (size_t)rbase * rowb, wp + (size_t)r1 * rowb,
                                wp + (size_t)r2 * rowb, wp + (size_t)r3 * rowb,
                                h + (size_t)e * A.k, nb, lane, e0, e1, e2, e3);
        e0 = simd_sum(e0); e1 = simd_sum(e1); e2 = simd_sum(e2); e3 = simd_sum(e3);
        const float w = SCR[A.wtoff + e];
        acc0 = acc0 + w * e0;
        acc1 = acc1 + w * e1;
        acc2 = acc2 + w * e2;
        acc3 = acc3 + w * e3;
    }
    if (lane == 0u) SCR[A.accoff + rbase] = acc0;
    if (lane == 1u && rbase + 1u < A.nout) SCR[A.accoff + rbase + 1u] = acc1;
    if (lane == 2u && rbase + 2u < A.nout) SCR[A.accoff + rbase + 2u] = acc2;
    if (lane == 3u && rbase + 3u < A.nout) SCR[A.accoff + rbase + 3u] = acc3;
}

// ---------------------------------------------------------------------------
// Milestone 9A: 2-stage parallel MoE down for the Q4_K (Q5_0/Q5_1 weights)
// decode path, mirroring k_gemv_down_batch8_m2 + k_down_reduce8_conv_add of
// the Q8_0 path. Stage 1: 2D grid (ceil(nout/16), 8) -> 1,152 TGs of 256 for
// kNE=2304 (vs the 72-TG single-stage m4), one expert per grid.y, 2 rows per
// warp sharing the h float4s. Unweighted per-expert FP32 results land in
// SCR[A.accoff + e * A.nout + row] (the [8][kNE] down_tmp slab) so the fused
// weighted-reduce + ffnconv + add kernel can consume them directly.
// Lane block iteration (i = lane, lane+32, ...) and the per-row simd_sum are
// identical to m64_gemv_q5_row; Milestone 13 factored the d/m/-16 scaling out
// of the inner loop, so each ea is tolerance-class (mad < 1e-4,
// cos >= 0.99999) vs the m1/m4 kernels' expert dot (tests 51/52).
// ---------------------------------------------------------------------------

// Milestone 13: factored 5-bit raw accumulator for the rows2 2-row loop:
// sq accumulates q(0..31)·x directly; d / m / -16 are applied per block in
// the caller. Reorders FMAs vs m64_q5_block_dot (tolerance-class parity,
// tests 51/52).
static inline void m64_q5_block_acc(uint v, uint qh, uint t,
                                    const float4 & vlo, const float4 & vhi, float & sq) {
    #pragma unroll
    for (uint u = 0u; u < 4u; u++) {
        const uint jj = 4u * t + u;
        const float x0 = float(((v >> (8u * u)) & 0xFu) | (((qh >> jj) << 4) & 0x10u));
        const float x1 = float(((v >> (8u * u + 4u)) & 0xFu) | ((qh >> (jj + 12u)) & 0x10u));
        if (u == 0u) { sq = fma(x0, vlo.x, sq); sq = fma(x1, vhi.x, sq); }
        if (u == 1u) { sq = fma(x0, vlo.y, sq); sq = fma(x1, vhi.y, sq); }
        if (u == 2u) { sq = fma(x0, vlo.z, sq); sq = fma(x1, vhi.z, sq); }
        if (u == 3u) { sq = fma(x0, vlo.w, sq); sq = fma(x1, vhi.w, sq); }
    }
}

// Two Q5_0/Q5_1 rows vs one h slab (Milestone 13 factored): h float4s and the
// 32-elem x sum are loaded/computed ONCE per block for both rows; raw 5-bit
// dots are scaled once per block after the t loop. Headers use 2 B aligned
// ushort loads (Q5_0 blocks are 22 B — NEVER uint-cast them). Lane block
// iteration (i = lane, lane+32, ...) is unchanged; per-block d/m application
// keeps magnitude parity with m64_gemv_q5_row (tolerance-class, tests 51/52).
template <bool Q51>
static inline void m64_gemv_q5_rows2(const device uchar * w0, const device uchar * w1,
                                     const device float * h, uint nb, uint lane,
                                     float & a0, float & a1) {
    const uint blk = Q51 ? 24u : 22u;
    for (uint i = lane; i < nb; i += 32u) {
        const size_t bo = (size_t)i * blk;
        const device float4 * hlo = (const device float4 *)(h + (size_t)i * 32u);
        const device ushort * u0 = (const device ushort *)(w0 + bo);
        const device ushort * u1 = (const device ushort *)(w1 + bo);
        const float d0 = f16b(u0[0]), d1 = f16b(u1[0]);
        const float m0 = Q51 ? f16b(u0[1]) : 0.0f;
        const float m1 = Q51 ? f16b(u1[1]) : 0.0f;
        const uint qh0 = Q51 ? (uint(u0[2]) | (uint(u0[3]) << 16))
                             : (uint(u0[1]) | (uint(u0[2]) << 16));
        const uint qh1 = Q51 ? (uint(u1[2]) | (uint(u1[3]) << 16))
                             : (uint(u1[1]) | (uint(u1[2]) << 16));
        const device ushort * q0 = u0 + (Q51 ? 4u : 3u);
        const device ushort * q1 = u1 + (Q51 ? 4u : 3u);
        float sh = 0.0f, sq0 = 0.0f, sq1 = 0.0f;
        #pragma unroll
        for (uint t = 0u; t < 4u; t++) {
            const float4 vlo = hlo[t], vhi = hlo[4u + t];   // shared across rows
            sh += ((vlo.x + vlo.y) + (vlo.z + vlo.w)) + ((vhi.x + vhi.y) + (vhi.z + vhi.w));
            m64_q5_block_acc(uint(q0[2u * t]) | (uint(q0[2u * t + 1u]) << 16), qh0, t, vlo, vhi, sq0);
            m64_q5_block_acc(uint(q1[2u * t]) | (uint(q1[2u * t + 1u]) << 16), qh1, t, vlo, vhi, sq1);
        }
        if (!Q51) {
            a0 = fma(d0, fma(-16.0f, sh, sq0), a0);
            a1 = fma(d1, fma(-16.0f, sh, sq1), a1);
        } else {
            a0 = fma(d0, sq0, fma(m0, sh, a0));
            a1 = fma(d1, sq1, fma(m1, sh, a1));
        }
    }
}

// Stage 1 (Q5_0): expert e = grid.y, warp w = tid>>5 -> rows r0 = (tgx*8+w)*2,
// r1 = r0+1 (tail clamps reads, guards writes).
kernel void k_gemv_down_batch8_q5_0_f32in_m2(device float * SCR [[buffer(0)]],
                                             constant MoeDownQ5F32Args & A [[buffer(1)]],
                                             const device uchar * PACK [[buffer(2)]],
                                             uint3 M64_TITG [[thread_position_in_threadgroup]],
                                             uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint e = M64_TGPG.y;
    uint tid = M64_TITG.x;
    uint r0 = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (r0 >= A.nout) return;
    uint lane = tid & 31u;
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
    const uint slot = A.wtab_off != 0u ? ((const device uint *)(SCR + A.wtab_off))[exp_id] : exp_id;
    uint r1 = (r0 + 1u < A.nout) ? (r0 + 1u) : r0;
    const device uchar * wp = PACK + A.woff + (size_t)slot * A.stride;
    const device float * h = SCR + A.hoff + (size_t)e * A.k;
    const uint nb = A.k >> 5;
    const size_t rowb = (size_t)nb * 22ull;
    float a0 = 0.0f, a1 = 0.0f;
    m64_gemv_q5_rows2<false>(wp + (size_t)r0 * rowb, wp + (size_t)r1 * rowb, h, nb, lane, a0, a1);
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) SCR[A.accoff + e * A.nout + r0] = a0;
    if (lane == 1u && r0 + 1u < A.nout) SCR[A.accoff + e * A.nout + r1] = a1;
}

// Same with Q5_1 blocks (24 B, shared min-offset m; no -16 bias).
kernel void k_gemv_down_batch8_q5_1_f32in_m2(device float * SCR [[buffer(0)]],
                                             constant MoeDownQ5F32Args & A [[buffer(1)]],
                                             const device uchar * PACK [[buffer(2)]],
                                             uint3 M64_TITG [[thread_position_in_threadgroup]],
                                             uint3 M64_TGPG [[threadgroup_position_in_grid]]) {
    uint e = M64_TGPG.y;
    uint tid = M64_TITG.x;
    uint r0 = (M64_TGPG.x * 8u + (tid >> 5)) * 2u;
    if (r0 >= A.nout) return;
    uint lane = tid & 31u;
    const uint exp_id = ((const device uint *)(SCR + A.seloff))[e];
    const uint slot = A.wtab_off != 0u ? ((const device uint *)(SCR + A.wtab_off))[exp_id] : exp_id;
    uint r1 = (r0 + 1u < A.nout) ? (r0 + 1u) : r0;
    const device uchar * wp = PACK + A.woff + (size_t)slot * A.stride;
    const device float * h = SCR + A.hoff + (size_t)e * A.k;
    const uint nb = A.k >> 5;
    const size_t rowb = (size_t)nb * 24ull;
    float a0 = 0.0f, a1 = 0.0f;
    m64_gemv_q5_rows2<true>(wp + (size_t)r0 * rowb, wp + (size_t)r1 * rowb, h, nb, lane, a0, a1);
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) SCR[A.accoff + e * A.nout + r0] = a0;
    if (lane == 1u && r0 + 1u < A.nout) SCR[A.accoff + e * A.nout + r1] = a1;
}
