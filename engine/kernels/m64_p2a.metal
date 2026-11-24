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
