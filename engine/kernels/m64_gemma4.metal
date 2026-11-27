// m64_gemma4.metal — Milestone 6B-1: Gemma 4 (tiny fixture n_embd=256 and
// real 26B-A4B n_embd=2816 share the SAME kernels; every dimension arrives
// through the POD args below — nothing is hardcoded).
//
// Symbol discipline: MetalRuntime::init() concatenates ALL engine/kernels/
// *.metal files (m64.metal, m64_gemma4.metal, m64_p2a.metal, m64_prefill.metal)
// into ONE MSL compilation unit right after m64_p2a_args.h, so EVERY struct,
// helper and kernel here is prefixed G4 / g4_ to guarantee zero collisions.
// <metal_stdlib> + `using namespace metal;` come from m64.metal at the top of
// the unit; we only re-pin deterministic FP contraction for our region.
//
// m64_gemma4_args.h is NOT #include'd here (newLibraryWithSource cannot
// resolve local includes); the struct block below is a byte-for-byte mirror of
// it and static_asserts the shared layout.
#pragma clang fp contract(off)

// ==== mirror of engine/kernels/m64_gemma4_args.h (keep in sync) =============
struct G4EmbArgs {
    unsigned long woff;
    unsigned int  dst_off;
    unsigned int  n_embd;
    unsigned int  row_bytes;
    unsigned int  dtype;
    float         scale;
};
static_assert(sizeof(G4EmbArgs) == 32, "G4EmbArgs layout");

#define G4_DT_F32  0u
#define G4_DT_F16  1u
#define G4_DT_Q8_0 2u
#define G4_DT_Q4_K 3u
#define G4_DT_Q5_0 4u

struct G4RmsArgs {
    unsigned int  src_off, dst_off;
    unsigned long woff;
    unsigned int  k;
    unsigned int  rows;
    unsigned int  has_w;
    float         eps;
    float         post_scale;
};
static_assert(sizeof(G4RmsArgs) == 40, "G4RmsArgs layout");

struct G4RopeArgs {
    unsigned int  off;
    unsigned int  n_heads;
    unsigned int  head_dim;
    unsigned int  n_rot;
    unsigned long freq_off;
    float         freq_base;
    int           has_freqs;
};
static_assert(sizeof(G4RopeArgs) == 32, "G4RopeArgs layout");

struct G4GemvArgs {
    unsigned long woff;
    unsigned int  x_off, y_off;
    unsigned int  nout, nin;
    unsigned int  rbytes;
};
static_assert(sizeof(G4GemvArgs) == 32, "G4GemvArgs layout");

struct G4MoeGemvArgs {
    unsigned long layer_base;
    unsigned long expert_stride;
    unsigned long part_off;
    unsigned int  x_off, y_off;
    unsigned int  sel_ids_off;
    unsigned int  sel_w_off;
    unsigned int  k_slot;
    unsigned int  nout, nin;
    unsigned int  rbytes;
    unsigned int  accum;
};
static_assert(sizeof(G4MoeGemvArgs) == 64, "G4MoeGemvArgs layout");

struct G4KvArgs {
    unsigned int  k_off, v_off;
    unsigned int  n_kv, hd;
    unsigned int  ring_cap;
    int           is_swa;
};
static_assert(sizeof(G4KvArgs) == 24, "G4KvArgs layout");

struct G4AttnArgs {
    unsigned int  q_off;
    unsigned int  out_off;
    unsigned int  n_head;
    unsigned int  n_kv;
    unsigned int  hd;
    unsigned int  gqa_group;
    unsigned int  ring_cap;
    int           is_swa;
    int           n_swa;
};
static_assert(sizeof(G4AttnArgs) == 36, "G4AttnArgs layout");

struct G4GeGLUArgs {
    unsigned int  g_off, u_off, dst_off;
    unsigned int  n;
};
static_assert(sizeof(G4GeGLUArgs) == 16, "G4GeGLUArgs layout");

struct G4RouterArgs {
    unsigned int  logits_off;
    unsigned int  sel_ids_off;
    unsigned int  sel_w_off;
    unsigned int  n_expert, top_k;
    unsigned long down_scale_off;
    int           has_down_scale;
};
static_assert(sizeof(G4RouterArgs) == 40, "G4RouterArgs layout");

struct G4AddArgs {
    unsigned int  a_off, b_off, dst_off;
    unsigned int  n;
};
static_assert(sizeof(G4AddArgs) == 16, "G4AddArgs layout");

struct G4CopyArgs {
    unsigned int  src_off, dst_off;
    unsigned int  n;
};
static_assert(sizeof(G4CopyArgs) == 12, "G4CopyArgs layout");

struct G4ScaleArgs {
    unsigned int  dst_off;
    unsigned int  n;
    unsigned long woff;
};
static_assert(sizeof(G4ScaleArgs) == 16, "G4ScaleArgs layout");

struct G4SoftcapArgs {
    unsigned int  src_off, dst_off;
    unsigned int  n;
    float         cap;
};
static_assert(sizeof(G4SoftcapArgs) == 16, "G4SoftcapArgs layout");

#define G4_ARGMAX_CAND 32
struct G4ArgmaxArgs {
    unsigned int  src_off;
    unsigned int  out_off;
    unsigned int  arena_off;
    unsigned int  n;
};
static_assert(sizeof(G4ArgmaxArgs) == 16, "G4ArgmaxArgs layout");

// ==== Milestone 8A: chunked batched prefill (mirror of m64_gemma4_args.h) ====
#define G4_BMAX 64u
#define G4_RTILE 8u
#define G4_MTILE 4u

struct G4EmbBArgs {
    unsigned long woff;
    unsigned int  dst_off;
    unsigned int  dst_stride;
    unsigned int  n_embd, row_bytes, dtype;
    float         scale;
    unsigned int  rows;
    unsigned int  pad0;
};
static_assert(sizeof(G4EmbBArgs) == 40, "G4EmbBArgs layout");

struct G4GemvBArgs {
    unsigned long woff;
    unsigned int  x_off, y_off;
    unsigned int  x_stride, y_stride;
    unsigned int  nout, nin;
    unsigned int  rbytes;
    unsigned int  rows;
};
static_assert(sizeof(G4GemvBArgs) == 40, "G4GemvBArgs layout");

struct G4RopeBArgs {
    unsigned long freq_off;
    unsigned int  off, stride;
    unsigned int  n_heads, head_dim;
    unsigned int  n_rot;
    float         freq_base;
    int           has_freqs;
    int           pos0;
    unsigned int  rows;
};
static_assert(sizeof(G4RopeBArgs) == 48, "G4RopeBArgs layout");

struct G4KvBArgs {
    unsigned int  k_off, v_off;
    unsigned int  k_stride, v_stride;
    unsigned int  n_kv, hd;
    unsigned int  ring_cap, rows;
    int           is_swa;
    int           pos0;
};
static_assert(sizeof(G4KvBArgs) == 40, "G4KvBArgs layout");

struct G4AttnBArgs {
    unsigned int  q_off, q_stride;
    unsigned int  out_off, out_stride;
    unsigned int  n_head, n_kv;
    unsigned int  hd, gqa_group;
    unsigned int  ring_cap;
    unsigned int  rows;
    int           is_swa;
    int           n_swa;
    int           pos0;
};
static_assert(sizeof(G4AttnBArgs) == 52, "G4AttnBArgs layout");

struct G4RouterBArgs {
    unsigned long down_scale_off;
    unsigned int  logits_off, logits_stride;
    unsigned int  sel_ids_off, sel_w_off;
    unsigned int  n_expert, top_k;
    unsigned int  rows;
    int           has_down_scale;
};
static_assert(sizeof(G4RouterBArgs) == 40, "G4RouterBArgs layout");

struct G4GeGLUBArgs {
    unsigned int  g_off, u_off, dst_off;
    unsigned int  g_stride, u_stride, dst_stride;
    unsigned int  n, rows;
};
static_assert(sizeof(G4GeGLUBArgs) == 32, "G4GeGLUBArgs layout");

struct G4MoeGemvBArgs {
    unsigned long layer_base;
    unsigned long expert_stride;
    unsigned long part_off;
    unsigned int  x_off, y_off;
    unsigned int  x_stride, y_stride;
    unsigned int  sel_ids_off, sel_w_off;
    unsigned int  k_slot, k_slots;
    unsigned int  nout, nin;
    unsigned int  rbytes;
    unsigned int  rows;
    unsigned int  accum;
};
static_assert(sizeof(G4MoeGemvBArgs) == 80, "G4MoeGemvBArgs layout");

// ==== helpers (all g4_ prefixed) =============================================
// PACK carries arbitrary byte offsets: Q5_0 blocks are 22 B (odd blocks are
// only 2-byte aligned), Q8_0 34 B, Q4_K 144 B. NEVER reinterpret_cast to
// short/int on PACK — always assemble half/uint bytewise (portable + exact).
static inline float g4_ld_f16(const device uchar * p) {
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return float(as_type<half>(u));
}
static inline ushort g4_f16_bits(float f) {
    return as_type<ushort>(half(f));
}
static inline uint g4_ld_u32(const device uchar * p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

// Q4_K sub-scale unpack (exact port of ggml get_scale_min_k4).
static inline void g4_get_scale_min_k4(uint j, const device uchar * q, uint * d, uint * m) {
    if (j < 4) {
        *d = q[j] & 63u; *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---- 1. Embedding gather -----------------------------------------------------
kernel void g4_emb_gather_q8_0(device float * SCR            [[buffer(0)]],
                               const device uchar * PACK      [[buffer(1)]],
                               constant G4EmbArgs & A         [[buffer(2)]],
                               constant int & tok             [[buffer(3)]],
                               uint gid                       [[thread_position_in_grid]]) {
    if (gid >= A.n_embd) return;
    const device uchar * row = PACK + A.woff + (unsigned long)(uint)tok * A.row_bytes;
    const uint b = gid >> 5u, j = gid & 31u;
    const float d = g4_ld_f16(row + b * 34u);              // half, bytewise
    SCR[A.dst_off + gid] = d * float((char)row[b * 34u + 2u + j]) * A.scale;
}

kernel void g4_emb_gather_f32(device float * SCR            [[buffer(0)]],
                              const device uchar * PACK      [[buffer(1)]],
                              constant G4EmbArgs & A         [[buffer(2)]],
                              constant int & tok             [[buffer(3)]],
                              uint gid                       [[thread_position_in_grid]]) {
    if (gid >= A.n_embd) return;
    const device float * row = (const device float *)(PACK + A.woff +
                                                      (unsigned long)(uint)tok * A.row_bytes);
    SCR[A.dst_off + gid] = row[gid] * A.scale;
}

// ---- 2. RMSNorm (in-place safe: barrier between reduce and write) -----------
kernel void g4_rms_norm(device float * SCR              [[buffer(0)]],
                        const device uchar * PACK        [[buffer(1)]],
                        constant G4RmsArgs & A           [[buffer(2)]],
                        uint tg_id  [[threadgroup_position_in_grid]],
                        uint tid    [[thread_position_in_threadgroup]],
                        uint tgs    [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    device float * x = SCR + A.src_off + tg_id * A.k;
    float acc = 0.0f;
    for (uint i = tid; i < A.k; i += tgs) {
        const float v = x[i];
        acc += v * v;
    }
    sh[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = metal::rsqrt(sh[0] / float(A.k) + A.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device float * w = (const device float *)(PACK + A.woff);
    device float * y = SCR + A.dst_off + tg_id * A.k;
    for (uint i = tid; i < A.k; i += tgs) {
        const float wf = A.has_w != 0u ? w[i] : 1.0f;
        y[i] = wf * (x[i] * inv) * A.post_scale;
    }
}

// ---- 3. NEOX RoPE ------------------------------------------------------------
kernel void g4_rope_neox(device float * SCR              [[buffer(0)]],
                         const device uchar * PACK        [[buffer(1)]],
                         constant G4RopeArgs & A          [[buffer(2)]],
                         constant int & pos               [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x, h = gid.y;
    if (j >= A.n_rot / 2u || h >= A.n_heads) return;
    device float * p = SCR + A.off + h * A.head_dim;
    const float theta = float(pos) * metal::pow(A.freq_base, -2.0f * float(j) / float(A.n_rot));
    float angle;
    if (A.has_freqs != 0) {
        const float ff = ((const device float *)(PACK + A.freq_off))[j];
        if (!metal::isfinite(ff) || ff == 0.0f) return;   // no rotation
        angle = theta / ff;
    } else {
        angle = theta;
    }
    const float c = metal::cos(angle), s = metal::sin(angle);
    const uint half_ = A.n_rot / 2u;
    const float x0 = p[j], x1 = p[j + half_];
    p[j]           = x0 * c - x1 * s;
    p[j + half_]   = x0 * s + x1 * c;
}

// ---- 4. Dense GEMV: 1 simdgroup (32 threads) per output row -----------------
// Dispatch: threadgroups = ceil(nout/4) with (128,1,1) threads (4 simdgroups),
// or (32,1,1) with nout threadgroups — both work via the guards below.
kernel void g4_gemv_q8_0(device float * SCR              [[buffer(0)]],
                         const device uchar * PACK        [[buffer(1)]],
                         constant G4GemvArgs & A          [[buffer(2)]],
                         uint tg_id  [[threadgroup_position_in_grid]],
                         uint sg_id  [[simdgroup_index_in_threadgroup]],
                         uint lane   [[thread_index_in_simdgroup]],
                         uint sgs    [[simdgroups_per_threadgroup]]) {
    const uint row = tg_id * sgs + sg_id;
    if (row >= A.nout) return;
    const device uchar * rb = PACK + A.woff + (unsigned long)row * A.rbytes;
    const device float * x = SCR + A.x_off;
    const uint nb = A.nin / 32u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        const device uchar * blk = rb + b * 34u;
        const float d = g4_ld_f16(blk);
        const uint qs = b * 32u + lane;
        acc = metal::fma(d * float((char)blk[2u + lane]), x[qs], acc);
    }
    acc = metal::simd_sum(acc);
    if (lane == 0u) SCR[A.y_off + row] = acc;
}

kernel void g4_gemv_f32(device float * SCR              [[buffer(0)]],
                        const device uchar * PACK        [[buffer(1)]],
                        constant G4GemvArgs & A          [[buffer(2)]],
                        uint tg_id  [[threadgroup_position_in_grid]],
                        uint sg_id  [[simdgroup_index_in_threadgroup]],
                        uint lane   [[thread_index_in_simdgroup]],
                        uint sgs    [[simdgroups_per_threadgroup]]) {
    const uint row = tg_id * sgs + sg_id;
    if (row >= A.nout) return;
    const device float * w = (const device float *)(PACK + A.woff) + (unsigned long)row * A.nin;
    const device float * x = SCR + A.x_off;
    float acc = 0.0f;
    for (uint i = lane; i < A.nin; i += 32u)
        acc = metal::fma(w[i], x[i], acc);
    acc = metal::simd_sum(acc);
    if (lane == 0u) SCR[A.y_off + row] = acc;
}

kernel void g4_gemv_q4_k(device float * SCR              [[buffer(0)]],
                         const device uchar * PACK        [[buffer(1)]],
                         constant G4GemvArgs & A          [[buffer(2)]],
                         uint tg_id  [[threadgroup_position_in_grid]],
                         uint sg_id  [[simdgroup_index_in_threadgroup]],
                         uint lane   [[thread_index_in_simdgroup]],
                         uint sgs    [[simdgroups_per_threadgroup]]) {
    const uint row = tg_id * sgs + sg_id;
    if (row >= A.nout) return;
    const device uchar * rb = PACK + A.woff + (unsigned long)row * A.rbytes;
    const device float * x = SCR + A.x_off;
    const uint nb = A.nin / 256u;
    float acc = 0.0f;
    for (uint i = 0; i < nb; i++) {
        const device uchar * blk = rb + i * 144u;
        const float d = g4_ld_f16(blk);
        const float dm = g4_ld_f16(blk + 2u);
        // GGML Q4_K: qs is 128 bytes for 256 elems; byte (c*32 + lane) holds
        // elems [c*64 + lane] (low nibble) and [c*64 + 32 + lane] (high nibble),
        // sub-scales 2c / 2c+1 from scales[12] at blk+4.
        for (uint c = 0; c < 4u; c++) {
            uint sc0, m0, sc1, m1;
            g4_get_scale_min_k4(2u * c,      blk + 4u, &sc0, &m0);
            g4_get_scale_min_k4(2u * c + 1u, blk + 4u, &sc1, &m1);
            const uchar byte = blk[16u + c * 32u + lane];
            const uint base = i * 256u + c * 64u;
            acc = metal::fma(d * float(sc0) * float(byte & 0xFu) - dm * float(m0), x[base + lane],       acc);
            acc = metal::fma(d * float(sc1) * float(byte >> 4)   - dm * float(m1), x[base + 32u + lane], acc);
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0u) SCR[A.y_off + row] = acc;
}

kernel void g4_gemv_q5_0(device float * SCR              [[buffer(0)]],
                         const device uchar * PACK        [[buffer(1)]],
                         constant G4GemvArgs & A          [[buffer(2)]],
                         uint tg_id  [[threadgroup_position_in_grid]],
                         uint sg_id  [[simdgroup_index_in_threadgroup]],
                         uint lane   [[thread_index_in_simdgroup]],
                         uint sgs    [[simdgroups_per_threadgroup]]) {
    const uint row = tg_id * sgs + sg_id;
    if (row >= A.nout) return;
    const device uchar * rb = PACK + A.woff + (unsigned long)row * A.rbytes;
    const device float * x = SCR + A.x_off;
    const uint nb = A.nin / 32u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        const device uchar * blk = rb + b * 22u;     // odd blocks: 2B aligned only
        const float d = g4_ld_f16(blk);
        const uint qh = g4_ld_u32(blk + 2u);
        const uint jj = lane >> 4u;                  // 0/1
        const uint c  = lane & 15u;                  // 0..15
        // 5th bit: refs use ((qh >> (j+0)) << 4) & 0x10 for the low nibbles
        // and ((qh >> (j+12)) & 0x10) for the high ones — i.e. qh bits j and
        // j+16. (qh >> (c + 16*jj)) & 1 << 4 reproduces both exactly.
        const uint xh = ((qh >> (c + jj * 16u)) & 1u) << 4;
        const uint q  = blk[6u + c];
        const uint n  = jj ? ((q >> 4) | xh) : ((q & 0xFu) | xh);
        acc = metal::fma(float(int(n) - 16) * d, x[b * 32u + jj * 16u + c], acc);
    }
    acc = metal::simd_sum(acc);
    if (lane == 0u) SCR[A.y_off + row] = acc;
}

// ---- 5. MoE expert GEMV (weights live in expert records; SCR carries sel) ---
kernel void g4_moe_gemv_q4_k(device float * SCR              [[buffer(0)]],
                             const device uchar * PACK        [[buffer(1)]],
                             constant G4MoeGemvArgs & A       [[buffer(2)]],
                             uint tg_id  [[threadgroup_position_in_grid]],
                             uint sg_id  [[simdgroup_index_in_threadgroup]],
                             uint lane   [[thread_index_in_simdgroup]],
                             uint sgs    [[simdgroups_per_threadgroup]]) {
    const uint row = tg_id * sgs + sg_id;
    if (row >= A.nout) return;
    const device int * sel = (const device int *)(SCR + A.sel_ids_off);
    const unsigned long base = A.layer_base
        + (unsigned long)sel[A.k_slot] * A.expert_stride + A.part_off;
    const device uchar * rb = PACK + base + (unsigned long)row * A.rbytes;
    const device float * x = SCR + A.x_off;
    const uint nb = A.nin / 256u;
    float acc = 0.0f;
    for (uint i = 0; i < nb; i++) {
        const device uchar * blk = rb + i * 144u;
        const float d = g4_ld_f16(blk);
        const float dm = g4_ld_f16(blk + 2u);
        // GGML Q4_K: qs is 128 bytes for 256 elems; byte (c*32 + lane) holds
        // elems [c*64 + lane] (low nibble) and [c*64 + 32 + lane] (high nibble),
        // sub-scales 2c / 2c+1 from scales[12] at blk+4.
        for (uint c = 0; c < 4u; c++) {
            uint sc0, m0, sc1, m1;
            g4_get_scale_min_k4(2u * c,      blk + 4u, &sc0, &m0);
            g4_get_scale_min_k4(2u * c + 1u, blk + 4u, &sc1, &m1);
            const uchar byte = blk[16u + c * 32u + lane];
            const uint base = i * 256u + c * 64u;
            acc = metal::fma(d * float(sc0) * float(byte & 0xFu) - dm * float(m0), x[base + lane],       acc);
            acc = metal::fma(d * float(sc1) * float(byte >> 4)   - dm * float(m1), x[base + 32u + lane], acc);
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0u) {
        if (A.accum != 0u) {
            const float wgt = SCR[A.sel_w_off + A.k_slot];
            SCR[A.y_off + row] += wgt * acc;
        } else {
            SCR[A.y_off + row] = acc;
        }
    }
}

kernel void g4_moe_gemv_q5_0(device float * SCR              [[buffer(0)]],
                             const device uchar * PACK        [[buffer(1)]],
                             constant G4MoeGemvArgs & A       [[buffer(2)]],
                             uint tg_id  [[threadgroup_position_in_grid]],
                             uint sg_id  [[simdgroup_index_in_threadgroup]],
                             uint lane   [[thread_index_in_simdgroup]],
                             uint sgs    [[simdgroups_per_threadgroup]]) {
    const uint row = tg_id * sgs + sg_id;
    if (row >= A.nout) return;
    const device int * sel = (const device int *)(SCR + A.sel_ids_off);
    const unsigned long base = A.layer_base
        + (unsigned long)sel[A.k_slot] * A.expert_stride + A.part_off;
    const device uchar * rb = PACK + base + (unsigned long)row * A.rbytes;
    const device float * x = SCR + A.x_off;
    const uint nb = A.nin / 32u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        const device uchar * blk = rb + b * 22u;
        const float d = g4_ld_f16(blk);
        const uint qh = g4_ld_u32(blk + 2u);
        const uint jj = lane >> 4u;
        const uint c  = lane & 15u;
        // 5th bit: refs use ((qh >> (j+0)) << 4) & 0x10 for the low nibbles
        // and ((qh >> (j+12)) & 0x10) for the high ones — i.e. qh bits j and
        // j+16. (qh >> (c + 16*jj)) & 1 << 4 reproduces both exactly.
        const uint xh = ((qh >> (c + jj * 16u)) & 1u) << 4;
        const uint q  = blk[6u + c];
        const uint n  = jj ? ((q >> 4) | xh) : ((q & 0xFu) | xh);
        acc = metal::fma(float(int(n) - 16) * d, x[b * 32u + jj * 16u + c], acc);
    }
    acc = metal::simd_sum(acc);
    if (lane == 0u) {
        if (A.accum != 0u) {
            const float wgt = SCR[A.sel_w_off + A.k_slot];
            SCR[A.y_off + row] += wgt * acc;
        } else {
            SCR[A.y_off + row] = acc;
        }
    }
}

kernel void g4_moe_gemv_q8_0(device float * SCR              [[buffer(0)]],
                             const device uchar * PACK        [[buffer(1)]],
                             constant G4MoeGemvArgs & A       [[buffer(2)]],
                             uint tg_id  [[threadgroup_position_in_grid]],
                             uint sg_id  [[simdgroup_index_in_threadgroup]],
                             uint lane   [[thread_index_in_simdgroup]],
                             uint sgs    [[simdgroups_per_threadgroup]]) {
    const uint row = tg_id * sgs + sg_id;
    if (row >= A.nout) return;
    const device int * sel = (const device int *)(SCR + A.sel_ids_off);
    const unsigned long base = A.layer_base
        + (unsigned long)sel[A.k_slot] * A.expert_stride + A.part_off;
    const device uchar * rb = PACK + base + (unsigned long)row * A.rbytes;
    const device float * x = SCR + A.x_off;
    const uint nb = A.nin / 32u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        const device uchar * blk = rb + b * 34u;
        const float d = g4_ld_f16(blk);
        const uint qs = b * 32u + lane;
        acc = metal::fma(d * float((char)blk[2u + lane]), x[qs], acc);
    }
    acc = metal::simd_sum(acc);
    if (lane == 0u) {
        if (A.accum != 0u) {
            const float wgt = SCR[A.sel_w_off + A.k_slot];
            SCR[A.y_off + row] += wgt * acc;
        } else {
            SCR[A.y_off + row] = acc;
        }
    }
}

// ---- 6a. KV cache write ------------------------------------------------------
// FP16: 1 thread per (head, dim) element of K and V for this token.
kernel void g4_kv_write_f16(device float * SCR              [[buffer(0)]],
                            const device uchar * PACK        [[buffer(1)]],
                            constant G4KvArgs & A            [[buffer(2)]],
                            constant int & pos               [[buffer(3)]],
                            device uchar * KBUF              [[buffer(4)]],
                            device uchar * VBUF              [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint total = A.n_kv * A.hd;
    if (gid >= total) return;
    const uint slot = A.is_swa ? ((uint)pos % A.ring_cap) : (uint)pos;
    const unsigned long o = ((unsigned long)slot * total + gid) * 2u;
    const ushort kb = g4_f16_bits(SCR[A.k_off + gid]);
    const ushort vb = g4_f16_bits(SCR[A.v_off + gid]);
    KBUF[o] = (uchar)(kb & 0xFFu); KBUF[o + 1u] = (uchar)(kb >> 8);
    VBUF[o] = (uchar)(vb & 0xFFu); VBUF[o + 1u] = (uchar)(vb >> 8);
}

// Q8_0: 32-elem blocks along hd; one thread per (head, block) pair for K and V
// (hd/32 pairs per head; serial block scan keeps quantization bit-reproducible).
kernel void g4_kv_write_q8(device float * SCR              [[buffer(0)]],
                           const device uchar * PACK        [[buffer(1)]],
                           constant G4KvArgs & A            [[buffer(2)]],
                           constant int & pos               [[buffer(3)]],
                           device uchar * KBUF              [[buffer(4)]],
                           device uchar * VBUF              [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint nb_head = A.hd / 32u;
    const uint total = A.n_kv * nb_head;
    if (gid >= total) return;
    const uint h = gid / nb_head, b = gid % nb_head;
    const uint slot = A.is_swa ? ((uint)pos % A.ring_cap) : (uint)pos;
    const device float * src[2] = { SCR + A.k_off, SCR + A.v_off };
    device uchar * dst[2] = { KBUF, VBUF };
    for (int t = 0; t < 2; t++) {
        const device float * x = src[t] + h * A.hd + b * 32u;
        float amax = 0.0f;
        for (uint i = 0; i < 32u; i++) amax = metal::max(amax, metal::fabs(x[i]));
        const float d = amax / 127.0f;
        const ushort db = g4_f16_bits(d);
        const float id = float(as_type<half>(db)) != 0.0f
                       ? 1.0f / float(as_type<half>(db)) : 0.0f;
        const unsigned long o = (((unsigned long)slot * A.n_kv + h) * nb_head + b) * 34u;
        device uchar * blk = dst[t] + o;
        blk[0] = (uchar)(db & 0xFFu); blk[1] = (uchar)(db >> 8);
        for (uint i = 0; i < 32u; i++) {
            int qv = int(metal::rint(x[i] * id));
            qv = qv < -127 ? -127 : (qv > 127 ? 127 : qv);
            blk[2u + i] = (uchar)(uchar)qv;
        }
    }
}

// ---- 6b. Causal GQA decode attention (online softmax, scale == 1.0f) --------
kernel void g4_attn_f16(device float * SCR              [[buffer(0)]],
                        const device uchar * PACK        [[buffer(1)]],
                        constant G4AttnArgs & A          [[buffer(2)]],
                        constant int & pos               [[buffer(3)]],
                        const device uchar * KBUF        [[buffer(4)]],
                        const device uchar * VBUF        [[buffer(5)]],
                        uint tg_id [[threadgroup_position_in_grid]],
                        uint tid   [[thread_position_in_threadgroup]]) {
    threadgroup float red[256];
    threadgroup float ao[512];
    const uint h = tg_id;
    if (h >= A.n_head) return;
    const uint kv = h / A.gqa_group;
    const uint hd = A.hd;
    device float * q = SCR + A.q_off + h * hd;
    const uint i0 = tid, i1 = tid + 256u;
    const bool has0 = i0 < hd, has1 = i1 < hd;
    const float q0 = has0 ? q[i0] : 0.0f;
    const float q1 = has1 ? q[i1] : 0.0f;
    float o0 = 0.0f, o1 = 0.0f;
    float m = -metal::numeric_limits<float>::max(), dsum = 0.0f;
    const int j0 = (A.is_swa && A.n_swa > 0 && pos >= (int)A.n_swa)
                 ? pos - (int)A.n_swa + 1 : 0;
    for (int j = j0; j <= pos; j++) {
        const uint slot = A.is_swa ? ((uint)j % A.ring_cap) : (uint)j;
        float k0 = 0.0f;
        if (has0) {
            const device uchar * k = KBUF + (((unsigned long)slot * A.n_kv + kv) * hd + i0) * 2u;
            k0 = float(as_type<half>(ushort((ushort)k[0] | ((ushort)k[1] << 8))));
        }
        float k1 = 0.0f;
        if (has1) {
            const device uchar * k1p = KBUF + (((unsigned long)slot * A.n_kv + kv) * hd + i1) * 2u;
            k1 = float(as_type<half>(ushort((ushort)k1p[0] | ((ushort)k1p[1] << 8))));
        }
        float dot = q0 * k0 + q1 * k1;
        red[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // all threads must read final red[0]
        threadgroup_barrier(mem_flags::mem_threadgroup);  // all threads must read final red[0]
        const float score = red[0];   // scale == 1.0f (Gemma fused attention)
        const float m_new = metal::max(m, score);
        const float a = metal::exp(m - m_new);
        const float b = metal::exp(score - m_new);
        dsum = dsum * a + b;
        o0 = o0 * a; o1 = o1 * a;
        if (has0) {
            const device uchar * v = VBUF + (((unsigned long)slot * A.n_kv + kv) * hd + i0) * 2u;
            o0 = metal::fma(b, float(as_type<half>(ushort((ushort)v[0] | ((ushort)v[1] << 8)))), o0);
        }
        if (has1) {
            const device uchar * v1p = VBUF + (((unsigned long)slot * A.n_kv + kv) * hd + i1) * 2u;
            o1 = metal::fma(b, float(as_type<half>(ushort((ushort)v1p[0] | ((ushort)v1p[1] << 8)))), o1);
        }
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // reuse red next iter
    }
    const float inv = 1.0f / dsum;
    if (has0) ao[i0] = o0 * inv;
    if (has1) ao[i1] = o1 * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);  // ao fully written before readback
    device float * out = SCR + A.out_off + h * hd;
    for (uint i = tid; i < hd; i += 256u) out[i] = ao[i];
}

kernel void g4_attn_q8(device float * SCR              [[buffer(0)]],
                       const device uchar * PACK        [[buffer(1)]],
                       constant G4AttnArgs & A          [[buffer(2)]],
                       constant int & pos               [[buffer(3)]],
                       const device uchar * KBUF        [[buffer(4)]],
                       const device uchar * VBUF        [[buffer(5)]],
                       uint tg_id [[threadgroup_position_in_grid]],
                       uint tid   [[thread_position_in_threadgroup]]) {
    threadgroup float red[256];
    threadgroup float ao[512];
    const uint h = tg_id;
    if (h >= A.n_head) return;
    const uint kv = h / A.gqa_group;
    const uint hd = A.hd, nb_head = hd / 32u;
    device float * q = SCR + A.q_off + h * hd;
    const uint i0 = tid, i1 = tid + 256u;
    const bool has0 = i0 < hd, has1 = i1 < hd;
    const float q0 = has0 ? q[i0] : 0.0f;
    const float q1 = has1 ? q[i1] : 0.0f;
    const uint blk0 = i0 >> 5u, blk1 = i1 >> 5u;
    const uint in0 = i0 & 31u, in1 = i1 & 31u;
    float o0 = 0.0f, o1 = 0.0f;
    float m = -metal::numeric_limits<float>::max(), dsum = 0.0f;
    const int j0 = (A.is_swa && A.n_swa > 0 && pos >= (int)A.n_swa)
                 ? pos - (int)A.n_swa + 1 : 0;
    for (int j = j0; j <= pos; j++) {
        const uint slot = A.is_swa ? ((uint)j % A.ring_cap) : (uint)j;
        const unsigned long kbase = ((unsigned long)slot * A.n_kv + kv) * nb_head * 34u;
        float k0 = 0.0f;
        if (has0) {
            const device uchar * kb0 = KBUF + kbase + blk0 * 34u;
            k0 = g4_ld_f16(kb0) * float((char)kb0[2u + in0]);
        }
        float k1 = 0.0f;
        if (has1) {
            const device uchar * kb1 = KBUF + kbase + blk1 * 34u;
            k1 = g4_ld_f16(kb1) * float((char)kb1[2u + in1]);
        }
        float dot = q0 * k0 + q1 * k1;
        red[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // all threads must read final red[0]
        threadgroup_barrier(mem_flags::mem_threadgroup);  // all threads must read final red[0]
        const float score = red[0];   // scale == 1.0f (Gemma fused attention)
        const float m_new = metal::max(m, score);
        const float a = metal::exp(m - m_new);
        const float b = metal::exp(score - m_new);
        dsum = dsum * a + b;
        o0 = o0 * a; o1 = o1 * a;
        const unsigned long vbase = ((unsigned long)slot * A.n_kv + kv) * nb_head * 34u;
        if (has0) {
            const device uchar * vb0 = VBUF + vbase + blk0 * 34u;
            o0 = metal::fma(b, g4_ld_f16(vb0) * float((char)vb0[2u + in0]), o0);
        }
        if (has1) {
            const device uchar * vb1 = VBUF + vbase + blk1 * 34u;
            o1 = metal::fma(b, g4_ld_f16(vb1) * float((char)vb1[2u + in1]), o1);
        }
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / dsum;
    if (has0) ao[i0] = o0 * inv;
    if (has1) ao[i1] = o1 * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);  // ao fully written before readback
    device float * out = SCR + A.out_off + h * hd;
    for (uint i = tid; i < hd; i += 256u) out[i] = ao[i];
}

// ---- 7a. GeGLU: 0.5 * g * (1 + tanh(0.79788456*(g + 0.044715 g^3))) * u -----
kernel void g4_geglu(device float * SCR          [[buffer(0)]],
                     const device uchar * PACK     [[buffer(1)]],
                     constant G4GeGLUArgs & A      [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= A.n) return;
    const float g = SCR[A.g_off + gid];
    const float u = SCR[A.u_off + gid];
    const float arg = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    const float t = arg > 20.0f ? 1.0f : (arg < -20.0f ? -1.0f : metal::tanh(arg));
    SCR[A.dst_off + gid] = 0.5f * g * (1.0f + t) * u;
}

// ---- 7b. Router softmax + deterministic greedy top-k (1 thread) -------------
kernel void g4_router_topk(device float * SCR              [[buffer(0)]],
                           const device uchar * PACK        [[buffer(1)]],
                           constant G4RouterArgs & A        [[buffer(2)]],
                           uint tid [[thread_position_in_threadgroup]]) {
    if (tid != 0u) return;
    device float * lg = SCR + A.logits_off;
    const uint n = A.n_expert, k = A.top_k;
    // softmax (same op order as gemma4_forward.cpp; scalar accumulator keeps
    // the double-precision denominator even on GPU)
    float mx = lg[0];
    for (uint i = 1u; i < n; i++) mx = lg[i] > mx ? lg[i] : mx;
    float sum = 0.0f;   // MSL has no double: f32 accumulator (router logits
                        // are softmax-normalized, so the drift is negligible)
    for (uint i = 0u; i < n; i++) { lg[i] = metal::exp(lg[i] - mx); sum += lg[i]; }
    const float inv_sm = 1.0f / sum;
    for (uint i = 0u; i < n; i++) lg[i] *= inv_sm;
    // greedy top-k: strict >, first (smallest) index wins ties
    device int * sel = (device int *)(SCR + A.sel_ids_off);
    device float * sw = SCR + A.sel_w_off;
    float wsum = 0.0f;
    for (uint e = 0; e < k; e++) {
        int best = -1;
        float bv = -metal::numeric_limits<float>::max();
        for (uint i = 0; i < n; i++) {
            bool taken = false;
            for (uint q = 0; q < e; q++) if (sel[q] == (int)i) { taken = true; break; }
            if (!taken && lg[i] > bv) { bv = lg[i]; best = (int)i; }
        }
        sel[e] = best;
        sw[e] = lg[(uint)best];
        wsum += lg[(uint)best];
    }
    const float inv = 1.0f / (wsum + 1e-20f);
    const device float * ds = A.has_down_scale != 0
                            ? (const device float *)(PACK + A.down_scale_off) : nullptr;
    for (uint e = 0; e < k; e++)
        sw[e] = sw[e] * inv * (ds ? ds[(uint)sel[e]] : 1.0f);
}

// ---- 7c. Vector ops -----------------------------------------------------------
kernel void g4_add(device float * SCR          [[buffer(0)]],
                   const device uchar * PACK     [[buffer(1)]],
                   constant G4AddArgs & A        [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= A.n) return;
    SCR[A.dst_off + gid] = SCR[A.a_off + gid] + SCR[A.b_off + gid];
}

kernel void g4_copy(device float * SCR          [[buffer(0)]],
                    const device uchar * PACK     [[buffer(1)]],
                    constant G4CopyArgs & A       [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= A.n) return;
    SCR[A.dst_off + gid] = SCR[A.src_off + gid];
}

kernel void g4_scale_scalar_buf(device float * SCR          [[buffer(0)]],
                                const device uchar * PACK     [[buffer(1)]],
                                constant G4ScaleArgs & A      [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= A.n) return;
    const float s = ((const device float *)(PACK + A.woff))[0];
    SCR[A.dst_off + gid] *= s;
}

kernel void g4_softcap(device float * SCR              [[buffer(0)]],
                       const device uchar * PACK        [[buffer(1)]],
                       constant G4SoftcapArgs & A       [[buffer(2)]],
                       uint gid       [[thread_position_in_grid]],
                       uint gr        [[threadgroups_per_grid]]) {
    const uint stride = gr * 256u;   // launched with (ceil(n/256),1,1)x(256,1,1)
    for (uint i = gid; i < A.n; i += stride) {
        const float x = SCR[A.src_off + i];
        SCR[A.dst_off + i] = A.cap * metal::tanh(x / A.cap);
    }
}

// Single-dispatch EXACT argmax: one threadgroup (grid (1,1,1) x (256,1,1)).
// Each thread keeps a local (best value, index) pair over its strided slice,
// then a 256-entry tree reduction with strict `>` and smallest-index
// tie-breaking (identical to the CPU oracle's sequential first-max scan)
// writes the int32 winner to SCR + out_off. Works for ANY finite (including
// negative) float values and any n in a single dispatch — no sortable-key
// encoding, so 18-bit indices (> 65535) are exact too. arena_off is unused
// (legacy two-pass scratch; the SCR slot is kept for layout stability).
kernel void g4_argmax(device float * SCR              [[buffer(0)]],
                      const device uchar * PACK        [[buffer(1)]],
                      constant G4ArgmaxArgs & A        [[buffer(2)]],
                      uint tid       [[thread_position_in_threadgroup]]) {
    threadgroup float sh_v[256];
    threadgroup int   sh_i[256];
    float bv = -metal::numeric_limits<float>::max();
    int bi = 0;
    for (uint i = tid; i < A.n; i += 256u) {
        const float v = SCR[A.src_off + i];
        const int idx = (int)i;
        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
    }
    sh_v[tid] = bv;
    sh_i[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float v2 = sh_v[tid + s];
            const int i2 = sh_i[tid + s];
            if (v2 > sh_v[tid] || (v2 == sh_v[tid] && i2 < sh_i[tid])) {
                sh_v[tid] = v2;
                sh_i[tid] = i2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) ((device int *)(SCR + A.out_off))[0] = sh_i[0];
}

// ==== Milestone 8A: chunked batched prefill kernels ==========================
// Bit-for-bit the same arithmetic per batch row as the single-token kernels
// above (identical fma order, identical block iteration, identical reduction
// pattern). The batch win: a dense weight row is streamed once per G4_RTILE
// rows (MoE: once per row, but all rows x nout run in parallel instead of a
// sequential top_k chain), so weight bandwidth — the prefill bottleneck — is
// shared across the chunk.

// ---- embedding gather (token ids in buffer(3), rows = grid.y) ----
kernel void g4_emb_gather_q8_0_b(device float * SCR, const device uchar * PACK,
    constant G4EmbBArgs & A [[buffer(2)]], constant int * TOK [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= A.n_embd) return;
    const device uchar * row = PACK + A.woff + (unsigned long)(uint)TOK[gid.y] * A.row_bytes;
    const uint b = gid.x >> 5u, j = gid.x & 31u;
    const float d = g4_ld_f16(row + b * 34u);
    SCR[A.dst_off + gid.y * A.dst_stride + gid.x] =
        d * float((char)row[b * 34u + 2u + j]) * A.scale;
}

kernel void g4_emb_gather_f32_b(device float * SCR, const device uchar * PACK,
    constant G4EmbBArgs & A [[buffer(2)]], constant int * TOK [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= A.n_embd) return;
    const device float * w = (const device float *)(PACK + A.woff) +
                             (unsigned long)(uint)TOK[gid.y] * A.n_embd;
    SCR[A.dst_off + gid.y * A.dst_stride + gid.x] = w[gid.x] * A.scale;
}

// ---- dense GEMV: 1 threadgroup per (nout/sgs, ceil(rows/G4_RTILE)) ----
// Row lanes read clamped batch indices (min(r0+u, rows-1)) so the inner loop
// stays branch-free; only in-bounds lanes store, and in-bounds lanes map to
// distinct rows, so no write races and no SCR overrun (m0 >= r0 >= 0).
kernel void g4_gemv_f32_b(device float * SCR, const device uchar * PACK, constant G4GemvBArgs & A,
    uint3 tg_id3 [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sgs   [[simdgroups_per_threadgroup]]) {
    const uint2 tg_id2 = uint2(tg_id3.x, tg_id3.y);
    const uint row = tg_id2.x * sgs + sg_id;
    if (row >= A.nout) return;
    const uint r0 = tg_id2.y * G4_RTILE;
    const uint m1 = min(r0 + 1u, A.rows - 1u), m2 = min(r0 + 2u, A.rows - 1u),
               m3 = min(r0 + 3u, A.rows - 1u), m4 = min(r0 + 4u, A.rows - 1u),
               m5 = min(r0 + 5u, A.rows - 1u), m6 = min(r0 + 6u, A.rows - 1u),
               m7 = min(r0 + 7u, A.rows - 1u);
    const device float * w = (const device float *)(PACK + A.woff) + (unsigned long)row * A.nin;
    const device float * x0 = SCR + A.x_off + (unsigned long)r0 * A.x_stride;
    const device float * x1 = SCR + A.x_off + (unsigned long)m1 * A.x_stride;
    const device float * x2 = SCR + A.x_off + (unsigned long)m2 * A.x_stride;
    const device float * x3 = SCR + A.x_off + (unsigned long)m3 * A.x_stride;
    const device float * x4 = SCR + A.x_off + (unsigned long)m4 * A.x_stride;
    const device float * x5 = SCR + A.x_off + (unsigned long)m5 * A.x_stride;
    const device float * x6 = SCR + A.x_off + (unsigned long)m6 * A.x_stride;
    const device float * x7 = SCR + A.x_off + (unsigned long)m7 * A.x_stride;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f,
          a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    for (uint i = lane; i < A.nin; i += 32u) {
        const float wv = w[i];
        a0 = metal::fma(wv, x0[i], a0); a1 = metal::fma(wv, x1[i], a1);
        a2 = metal::fma(wv, x2[i], a2); a3 = metal::fma(wv, x3[i], a3);
        a4 = metal::fma(wv, x4[i], a4); a5 = metal::fma(wv, x5[i], a5);
        a6 = metal::fma(wv, x6[i], a6); a7 = metal::fma(wv, x7[i], a7);
    }
    a0 = metal::simd_sum(a0); a1 = metal::simd_sum(a1);
    a2 = metal::simd_sum(a2); a3 = metal::simd_sum(a3);
    a4 = metal::simd_sum(a4); a5 = metal::simd_sum(a5);
    a6 = metal::simd_sum(a6); a7 = metal::simd_sum(a7);
    const unsigned long ys = A.y_stride;
    if (lane == 0u) {  // single writer; all 32 lanes hold the same reduced values
        if (r0 + 0u < A.rows) SCR[A.y_off + (unsigned long)r0 * ys + row] = a0;
        if (r0 + 1u < A.rows) SCR[A.y_off + (unsigned long)m1 * ys + row] = a1;
        if (r0 + 2u < A.rows) SCR[A.y_off + (unsigned long)m2 * ys + row] = a2;
        if (r0 + 3u < A.rows) SCR[A.y_off + (unsigned long)m3 * ys + row] = a3;
        if (r0 + 4u < A.rows) SCR[A.y_off + (unsigned long)m4 * ys + row] = a4;
        if (r0 + 5u < A.rows) SCR[A.y_off + (unsigned long)m5 * ys + row] = a5;
        if (r0 + 6u < A.rows) SCR[A.y_off + (unsigned long)m6 * ys + row] = a6;
        if (r0 + 7u < A.rows) SCR[A.y_off + (unsigned long)m7 * ys + row] = a7;
    }
}

kernel void g4_gemv_q8_0_b(device float * SCR, const device uchar * PACK, constant G4GemvBArgs & A,
    uint3 tg_id3 [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sgs   [[simdgroups_per_threadgroup]]) {
    const uint2 tg_id2 = uint2(tg_id3.x, tg_id3.y);
    const uint row = tg_id2.x * sgs + sg_id;
    if (row >= A.nout) return;
    const uint r0 = tg_id2.y * G4_RTILE;
    const uint m1 = min(r0 + 1u, A.rows - 1u), m2 = min(r0 + 2u, A.rows - 1u),
               m3 = min(r0 + 3u, A.rows - 1u), m4 = min(r0 + 4u, A.rows - 1u),
               m5 = min(r0 + 5u, A.rows - 1u), m6 = min(r0 + 6u, A.rows - 1u),
               m7 = min(r0 + 7u, A.rows - 1u);
    const device uchar * rb = PACK + A.woff + (unsigned long)row * A.rbytes;
    const device float * x0 = SCR + A.x_off + (unsigned long)r0 * A.x_stride;
    const device float * x1 = SCR + A.x_off + (unsigned long)m1 * A.x_stride;
    const device float * x2 = SCR + A.x_off + (unsigned long)m2 * A.x_stride;
    const device float * x3 = SCR + A.x_off + (unsigned long)m3 * A.x_stride;
    const device float * x4 = SCR + A.x_off + (unsigned long)m4 * A.x_stride;
    const device float * x5 = SCR + A.x_off + (unsigned long)m5 * A.x_stride;
    const device float * x6 = SCR + A.x_off + (unsigned long)m6 * A.x_stride;
    const device float * x7 = SCR + A.x_off + (unsigned long)m7 * A.x_stride;
    const uint nblocks = A.nin / 32u;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f,
          a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    for (uint b = 0; b < nblocks; b++) {
        const device uchar * blk = rb + b * 34u;
        const float d = g4_ld_f16(blk);
        const float v = d * float((char)blk[2u + lane]);
        const uint qs = b * 32u + lane;
        a0 = metal::fma(v, x0[qs], a0); a1 = metal::fma(v, x1[qs], a1);
        a2 = metal::fma(v, x2[qs], a2); a3 = metal::fma(v, x3[qs], a3);
        a4 = metal::fma(v, x4[qs], a4); a5 = metal::fma(v, x5[qs], a5);
        a6 = metal::fma(v, x6[qs], a6); a7 = metal::fma(v, x7[qs], a7);
    }
    a0 = metal::simd_sum(a0); a1 = metal::simd_sum(a1);
    a2 = metal::simd_sum(a2); a3 = metal::simd_sum(a3);
    a4 = metal::simd_sum(a4); a5 = metal::simd_sum(a5);
    a6 = metal::simd_sum(a6); a7 = metal::simd_sum(a7);
    const unsigned long ys = A.y_stride;
    if (lane == 0u) {  // single writer; all 32 lanes hold the same reduced values
        if (r0 + 0u < A.rows) SCR[A.y_off + (unsigned long)r0 * ys + row] = a0;
        if (r0 + 1u < A.rows) SCR[A.y_off + (unsigned long)m1 * ys + row] = a1;
        if (r0 + 2u < A.rows) SCR[A.y_off + (unsigned long)m2 * ys + row] = a2;
        if (r0 + 3u < A.rows) SCR[A.y_off + (unsigned long)m3 * ys + row] = a3;
        if (r0 + 4u < A.rows) SCR[A.y_off + (unsigned long)m4 * ys + row] = a4;
        if (r0 + 5u < A.rows) SCR[A.y_off + (unsigned long)m5 * ys + row] = a5;
        if (r0 + 6u < A.rows) SCR[A.y_off + (unsigned long)m6 * ys + row] = a6;
        if (r0 + 7u < A.rows) SCR[A.y_off + (unsigned long)m7 * ys + row] = a7;
    }
}

kernel void g4_gemv_q4_k_b(device float * SCR, const device uchar * PACK, constant G4GemvBArgs & A,
    uint3 tg_id3 [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sgs   [[simdgroups_per_threadgroup]]) {
    const uint2 tg_id2 = uint2(tg_id3.x, tg_id3.y);
    const uint row = tg_id2.x * sgs + sg_id;
    if (row >= A.nout) return;
    const uint r0 = tg_id2.y * G4_RTILE;
    const uint m1 = min(r0 + 1u, A.rows - 1u), m2 = min(r0 + 2u, A.rows - 1u),
               m3 = min(r0 + 3u, A.rows - 1u), m4 = min(r0 + 4u, A.rows - 1u),
               m5 = min(r0 + 5u, A.rows - 1u), m6 = min(r0 + 6u, A.rows - 1u),
               m7 = min(r0 + 7u, A.rows - 1u);
    const device uchar * rb = PACK + A.woff + (unsigned long)row * A.rbytes;
    const device float * x0 = SCR + A.x_off + (unsigned long)r0 * A.x_stride;
    const device float * x1 = SCR + A.x_off + (unsigned long)m1 * A.x_stride;
    const device float * x2 = SCR + A.x_off + (unsigned long)m2 * A.x_stride;
    const device float * x3 = SCR + A.x_off + (unsigned long)m3 * A.x_stride;
    const device float * x4 = SCR + A.x_off + (unsigned long)m4 * A.x_stride;
    const device float * x5 = SCR + A.x_off + (unsigned long)m5 * A.x_stride;
    const device float * x6 = SCR + A.x_off + (unsigned long)m6 * A.x_stride;
    const device float * x7 = SCR + A.x_off + (unsigned long)m7 * A.x_stride;
    const uint nblocks = A.nin / 256u;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f,
          a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    for (uint i = 0; i < nblocks; i++) {
        const device uchar * blk = rb + i * 144u;
        const float d = g4_ld_f16(blk), dm = g4_ld_f16(blk + 2u);
        for (uint c = 0; c < 4u; c++) {
            uint sc_lo, sc_hi, mn_lo, mn_hi;
            g4_get_scale_min_k4(2 * c, blk + 4u, &sc_lo, &mn_lo);
            g4_get_scale_min_k4(2 * c + 1u, blk + 4u, &sc_hi, &mn_hi);
            const uchar byte = blk[16u + c * 32u + lane];
            const uint base = i * 256u + c * 64u;
            const float w0 = d * float(sc_lo) * float(byte & 0xFu) - dm * float(mn_lo);
            const float w1 = d * float(sc_hi) * float(byte >> 4) - dm * float(mn_hi);
            a0 = metal::fma(w0, x0[base + lane], a0);
            a1 = metal::fma(w0, x1[base + lane], a1);
            a2 = metal::fma(w0, x2[base + lane], a2);
            a3 = metal::fma(w0, x3[base + lane], a3);
            a4 = metal::fma(w0, x4[base + lane], a4);
            a5 = metal::fma(w0, x5[base + lane], a5);
            a6 = metal::fma(w0, x6[base + lane], a6);
            a7 = metal::fma(w0, x7[base + lane], a7);
            a0 = metal::fma(w1, x0[base + 32u + lane], a0);
            a1 = metal::fma(w1, x1[base + 32u + lane], a1);
            a2 = metal::fma(w1, x2[base + 32u + lane], a2);
            a3 = metal::fma(w1, x3[base + 32u + lane], a3);
            a4 = metal::fma(w1, x4[base + 32u + lane], a4);
            a5 = metal::fma(w1, x5[base + 32u + lane], a5);
            a6 = metal::fma(w1, x6[base + 32u + lane], a6);
            a7 = metal::fma(w1, x7[base + 32u + lane], a7);
        }
    }
    a0 = metal::simd_sum(a0); a1 = metal::simd_sum(a1);
    a2 = metal::simd_sum(a2); a3 = metal::simd_sum(a3);
    a4 = metal::simd_sum(a4); a5 = metal::simd_sum(a5);
    a6 = metal::simd_sum(a6); a7 = metal::simd_sum(a7);
    const unsigned long ys = A.y_stride;
    if (lane == 0u) {  // single writer; all 32 lanes hold the same reduced values
        if (r0 + 0u < A.rows) SCR[A.y_off + (unsigned long)r0 * ys + row] = a0;
        if (r0 + 1u < A.rows) SCR[A.y_off + (unsigned long)m1 * ys + row] = a1;
        if (r0 + 2u < A.rows) SCR[A.y_off + (unsigned long)m2 * ys + row] = a2;
        if (r0 + 3u < A.rows) SCR[A.y_off + (unsigned long)m3 * ys + row] = a3;
        if (r0 + 4u < A.rows) SCR[A.y_off + (unsigned long)m4 * ys + row] = a4;
        if (r0 + 5u < A.rows) SCR[A.y_off + (unsigned long)m5 * ys + row] = a5;
        if (r0 + 6u < A.rows) SCR[A.y_off + (unsigned long)m6 * ys + row] = a6;
        if (r0 + 7u < A.rows) SCR[A.y_off + (unsigned long)m7 * ys + row] = a7;
    }
}

kernel void g4_gemv_q5_0_b(device float * SCR, const device uchar * PACK, constant G4GemvBArgs & A,
    uint3 tg_id3 [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sgs   [[simdgroups_per_threadgroup]]) {
    const uint2 tg_id2 = uint2(tg_id3.x, tg_id3.y);
    const uint row = tg_id2.x * sgs + sg_id;
    if (row >= A.nout) return;
    const uint r0 = tg_id2.y * G4_RTILE;
    const uint m1 = min(r0 + 1u, A.rows - 1u), m2 = min(r0 + 2u, A.rows - 1u),
               m3 = min(r0 + 3u, A.rows - 1u), m4 = min(r0 + 4u, A.rows - 1u),
               m5 = min(r0 + 5u, A.rows - 1u), m6 = min(r0 + 6u, A.rows - 1u),
               m7 = min(r0 + 7u, A.rows - 1u);
    const device uchar * rb = PACK + A.woff + (unsigned long)row * A.rbytes;
    const device float * x0 = SCR + A.x_off + (unsigned long)r0 * A.x_stride;
    const device float * x1 = SCR + A.x_off + (unsigned long)m1 * A.x_stride;
    const device float * x2 = SCR + A.x_off + (unsigned long)m2 * A.x_stride;
    const device float * x3 = SCR + A.x_off + (unsigned long)m3 * A.x_stride;
    const device float * x4 = SCR + A.x_off + (unsigned long)m4 * A.x_stride;
    const device float * x5 = SCR + A.x_off + (unsigned long)m5 * A.x_stride;
    const device float * x6 = SCR + A.x_off + (unsigned long)m6 * A.x_stride;
    const device float * x7 = SCR + A.x_off + (unsigned long)m7 * A.x_stride;
    const uint nblocks = A.nin / 32u;
    const uint jj = lane >> 4u;
    const uint c  = lane & 15u;
    const uint qs = jj * 16u + c;      // same per-lane element as single kernel
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f,
          a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    for (uint b = 0; b < nblocks; b++) {
        const device uchar * blk = rb + b * 22u;
        const float d = g4_ld_f16(blk);
        const uint qh = g4_ld_u32(blk + 2u);
        const uint xh = ((qh >> (c + jj * 16u)) & 1u) << 4;
        const uint q  = blk[6u + c];
        const uint n  = jj ? ((q >> 4) | xh) : ((q & 0xFu) | xh);
        const float w = float(int(n) - 16) * d;
        const uint qi = b * 32u + qs;
        a0 = metal::fma(w, x0[qi], a0); a1 = metal::fma(w, x1[qi], a1);
        a2 = metal::fma(w, x2[qi], a2); a3 = metal::fma(w, x3[qi], a3);
        a4 = metal::fma(w, x4[qi], a4); a5 = metal::fma(w, x5[qi], a5);
        a6 = metal::fma(w, x6[qi], a6); a7 = metal::fma(w, x7[qi], a7);
    }
    a0 = metal::simd_sum(a0); a1 = metal::simd_sum(a1);
    a2 = metal::simd_sum(a2); a3 = metal::simd_sum(a3);
    a4 = metal::simd_sum(a4); a5 = metal::simd_sum(a5);
    a6 = metal::simd_sum(a6); a7 = metal::simd_sum(a7);
    const unsigned long ys = A.y_stride;
    if (lane == 0u) {  // single writer; all 32 lanes hold the same reduced values
        if (r0 + 0u < A.rows) SCR[A.y_off + (unsigned long)r0 * ys + row] = a0;
        if (r0 + 1u < A.rows) SCR[A.y_off + (unsigned long)m1 * ys + row] = a1;
        if (r0 + 2u < A.rows) SCR[A.y_off + (unsigned long)m2 * ys + row] = a2;
        if (r0 + 3u < A.rows) SCR[A.y_off + (unsigned long)m3 * ys + row] = a3;
        if (r0 + 4u < A.rows) SCR[A.y_off + (unsigned long)m4 * ys + row] = a4;
        if (r0 + 5u < A.rows) SCR[A.y_off + (unsigned long)m5 * ys + row] = a5;
        if (r0 + 6u < A.rows) SCR[A.y_off + (unsigned long)m6 * ys + row] = a6;
        if (r0 + 7u < A.rows) SCR[A.y_off + (unsigned long)m7 * ys + row] = a7;
    }
}

// ---- NEOX RoPE over a chunk: grid ((pairs+7)/8, (heads+7)/8, rows) x (8,8,1) ----
kernel void g4_rope_neox_b(device float * SCR, const device uchar * PACK, constant G4RopeBArgs & A,
    uint3 gid [[thread_position_in_grid]]) {
    const uint j = gid.x, h = gid.y, r = gid.z;
    if (j >= A.n_rot / 2u || h >= A.n_heads || r >= A.rows) return;
    device float * p = SCR + A.off + (unsigned long)r * A.stride + h * A.head_dim;
    const float theta = float((int)A.pos0 + (int)r) *
                        metal::pow(A.freq_base, -2.0f * float(j) / float(A.n_rot));
    float angle;
    if (A.has_freqs != 0) {
        const float ff = ((const device float *)(PACK + A.freq_off))[j];
        if (!metal::isfinite(ff) || ff == 0.0f) return;   // no rotation
        angle = theta / ff;
    } else {
        angle = theta;
    }
    const float c = metal::cos(angle), s = metal::sin(angle);
    const uint half_ = A.n_rot / 2u;
    const float x0 = p[j], x1 = p[j + half_];
    p[j]           = x0 * c - x1 * s;
    p[j + half_]   = x0 * s + x1 * c;
}

// ---- batch KV write (ring OR staging; staging = is_swa 0, pos0 0) ----
kernel void g4_kv_write_f16_b(device float * SCR, const device uchar * PACK, constant G4KvBArgs & A,
    device uchar * KBUF [[buffer(4)]], device uchar * VBUF [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint total = A.n_kv * A.hd;
    if (gid.x >= total) return;
    const uint r = gid.y;
    const uint posr = (uint)((int)A.pos0 + (int)r);
    const uint slot = A.is_swa ? (posr % A.ring_cap) : posr;
    const unsigned long o = ((unsigned long)slot * total + gid.x) * 2u;
    const ushort kb = g4_f16_bits(SCR[A.k_off + r * A.k_stride + gid.x]);
    const ushort vb = g4_f16_bits(SCR[A.v_off + r * A.v_stride + gid.x]);
    KBUF[o] = (uchar)(kb & 0xFFu); KBUF[o + 1u] = (uchar)(kb >> 8);
    VBUF[o] = (uchar)(vb & 0xFFu); VBUF[o + 1u] = (uchar)(vb >> 8);
}

kernel void g4_kv_write_q8_b(device float * SCR, const device uchar * PACK, constant G4KvBArgs & A,
    device uchar * KBUF [[buffer(4)]], device uchar * VBUF [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint nb_head = A.hd / 32u;
    const uint total = A.n_kv * nb_head;
    if (gid.x >= total) return;
    const uint r = gid.y;
    const uint h = gid.x / nb_head, b = gid.x % nb_head;
    const uint posr = (uint)((int)A.pos0 + (int)r);
    const uint slot = A.is_swa ? (posr % A.ring_cap) : posr;
    const device float * src[2] = { SCR + A.k_off + r * A.k_stride,
                                    SCR + A.v_off + r * A.v_stride };
    device uchar * dst[2] = { KBUF, VBUF };
    for (int t = 0; t < 2; t++) {
        const device float * x = src[t] + h * A.hd + b * 32u;
        float amax = 0.0f;
        for (uint i = 0; i < 32u; i++) amax = metal::max(amax, metal::fabs(x[i]));
        const float d = amax / 127.0f;
        const ushort db = g4_f16_bits(d);
        const float id = float(as_type<half>(db)) != 0.0f
                       ? 1.0f / float(as_type<half>(db)) : 0.0f;
        const unsigned long o = (((unsigned long)slot * A.n_kv + h) * nb_head + b) * 34u;
        device uchar * blk = dst[t] + o;
        blk[0] = (uchar)(db & 0xFFu); blk[1] = (uchar)(db >> 8);
        for (uint i = 0; i < 32u; i++) {
            int qv = int(metal::rint(x[i] * id));
            qv = qv < -127 ? -127 : (qv > 127 ? 127 : qv);
            blk[2u + i] = (uchar)(uchar)qv;
        }
    }
}

// ---- batch causal attention: threadgroup (x = head, y = batch row) x 256 ----
// j < pos0 -> ring (KBUF/VBUF); pos0 <= j -> in-chunk staging (SBUF/SVBUF).
kernel void g4_attn_f16_b(device float * SCR, const device uchar * PACK, constant G4AttnBArgs & A,
    const device uchar * KBUF [[buffer(4)]], const device uchar * VBUF [[buffer(5)]],
    const device uchar * SBUF [[buffer(6)]], const device uchar * SVBUF [[buffer(7)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
    const uint tid = tid3.x;
    threadgroup float red[256];
    threadgroup float ao[512];
    const uint h = tg_id.x, r = tg_id.y;
    if (h >= A.n_head || r >= A.rows) return;
    const uint kv = h / A.gqa_group;
    const uint hd = A.hd;
    device float * q = SCR + A.q_off + r * A.q_stride + h * hd;
    const uint i0 = tid, i1 = tid + 256u;
    const bool has0 = i0 < hd, has1 = i1 < hd;
    const float q0 = has0 ? q[i0] : 0.0f;
    const float q1 = has1 ? q[i1] : 0.0f;
    float o0 = 0.0f, o1 = 0.0f;
    float m = -metal::numeric_limits<float>::max(), dsum = 0.0f;
    const int pos = A.pos0 + (int)r;
    const int j0 = (A.is_swa && A.n_swa > 0 && pos >= (int)A.n_swa)
                 ? pos - (int)A.n_swa + 1 : 0;
    for (int j = j0; j <= pos; j++) {
        const device uchar * kb;
        const device uchar * vb;
        uint slot;
        if (j < A.pos0) {
            slot = A.is_swa ? ((uint)j % A.ring_cap) : (uint)j;
            kb = KBUF; vb = VBUF;
        } else {
            slot = (uint)(j - A.pos0);
            kb = SBUF; vb = SVBUF;
        }
        float k0 = 0.0f;
        if (has0) {
            const device uchar * k = kb + (((unsigned long)slot * A.n_kv + kv) * hd + i0) * 2u;
            k0 = float(as_type<half>(ushort((ushort)k[0] | ((ushort)k[1] << 8))));
        }
        float k1 = 0.0f;
        if (has1) {
            const device uchar * k1p = kb + (((unsigned long)slot * A.n_kv + kv) * hd + i1) * 2u;
            k1 = float(as_type<half>(ushort((ushort)k1p[0] | ((ushort)k1p[1] << 8))));
        }
        float dot = q0 * k0 + q1 * k1;
        red[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // all threads must read final red[0]
        const float score = red[0];   // scale == 1.0f (Gemma fused attention)
        const float m_new = metal::max(m, score);
        const float a = metal::exp(m - m_new);
        const float b = metal::exp(score - m_new);
        dsum = dsum * a + b;
        o0 = o0 * a; o1 = o1 * a;
        if (has0) {
            const device uchar * v = vb + (((unsigned long)slot * A.n_kv + kv) * hd + i0) * 2u;
            o0 = metal::fma(b, float(as_type<half>(ushort((ushort)v[0] | ((ushort)v[1] << 8)))), o0);
        }
        if (has1) {
            const device uchar * v1p = vb + (((unsigned long)slot * A.n_kv + kv) * hd + i1) * 2u;
            o1 = metal::fma(b, float(as_type<half>(ushort((ushort)v1p[0] | ((ushort)v1p[1] << 8)))), o1);
        }
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // reuse red next iter
    }
    const float inv = 1.0f / dsum;
    if (has0) ao[i0] = o0 * inv;
    if (has1) ao[i1] = o1 * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);  // ao fully written before readback
    device float * out = SCR + A.out_off + r * A.out_stride + h * hd;
    for (uint i = tid; i < hd; i += 256u) out[i] = ao[i];
}

kernel void g4_attn_q8_b(device float * SCR, const device uchar * PACK, constant G4AttnBArgs & A,
    const device uchar * KBUF [[buffer(4)]], const device uchar * VBUF [[buffer(5)]],
    const device uchar * SBUF [[buffer(6)]], const device uchar * SVBUF [[buffer(7)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
    const uint tid = tid3.x;
    threadgroup float red[256];
    threadgroup float ao[512];
    const uint h = tg_id.x, r = tg_id.y;
    if (h >= A.n_head || r >= A.rows) return;
    const uint kv = h / A.gqa_group;
    const uint hd = A.hd, nb_head = hd / 32u;
    device float * q = SCR + A.q_off + r * A.q_stride + h * hd;
    const uint i0 = tid, i1 = tid + 256u;
    const bool has0 = i0 < hd, has1 = i1 < hd;
    const float q0 = has0 ? q[i0] : 0.0f;
    const float q1 = has1 ? q[i1] : 0.0f;
    const uint blk0 = i0 >> 5u, blk1 = i1 >> 5u;
    const uint in0 = i0 & 31u, in1 = i1 & 31u;
    float o0 = 0.0f, o1 = 0.0f;
    float m = -metal::numeric_limits<float>::max(), dsum = 0.0f;
    const int pos = A.pos0 + (int)r;
    const int j0 = (A.is_swa && A.n_swa > 0 && pos >= (int)A.n_swa)
                 ? pos - (int)A.n_swa + 1 : 0;
    for (int j = j0; j <= pos; j++) {
        const device uchar * kb;
        const device uchar * vb;
        uint slot;
        if (j < A.pos0) {
            slot = A.is_swa ? ((uint)j % A.ring_cap) : (uint)j;
            kb = KBUF; vb = VBUF;
        } else {
            slot = (uint)(j - A.pos0);
            kb = SBUF; vb = SVBUF;
        }
        const unsigned long kbase = ((unsigned long)slot * A.n_kv + kv) * nb_head * 34u;
        float k0 = 0.0f;
        if (has0) {
            const device uchar * kb0 = kb + kbase + blk0 * 34u;
            k0 = g4_ld_f16(kb0) * float((char)kb0[2u + in0]);
        }
        float k1 = 0.0f;
        if (has1) {
            const device uchar * kb1 = kb + kbase + blk1 * 34u;
            k1 = g4_ld_f16(kb1) * float((char)kb1[2u + in1]);
        }
        float dot = q0 * k0 + q1 * k1;
        red[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s >>= 1u) {
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // all threads must read final red[0]
        const float score = red[0];   // scale == 1.0f (Gemma fused attention)
        const float m_new = metal::max(m, score);
        const float a = metal::exp(m - m_new);
        const float b = metal::exp(score - m_new);
        dsum = dsum * a + b;
        o0 = o0 * a; o1 = o1 * a;
        const unsigned long vbase = ((unsigned long)slot * A.n_kv + kv) * nb_head * 34u;
        if (has0) {
            const device uchar * vb0 = vb + vbase + blk0 * 34u;
            o0 = metal::fma(b, g4_ld_f16(vb0) * float((char)vb0[2u + in0]), o0);
        }
        if (has1) {
            const device uchar * vb1 = vb + vbase + blk1 * 34u;
            o1 = metal::fma(b, g4_ld_f16(vb1) * float((char)vb1[2u + in1]), o1);
        }
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / dsum;
    if (has0) ao[i0] = o0 * inv;
    if (has1) ao[i1] = o1 * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);  // ao fully written before readback
    device float * out = SCR + A.out_off + r * A.out_stride + h * hd;
    for (uint i = tid; i < hd; i += 256u) out[i] = ao[i];
}

// ---- GeGLU with row strides: grid (ceil(n/256), rows) x (256,1,1) ----
kernel void g4_geglu_b(device float * SCR, const device uchar * PACK, constant G4GeGLUBArgs & A,
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= A.n) return;
    const uint r = gid.y;
    const float g = SCR[A.g_off + r * A.g_stride + gid.x];
    const float u = SCR[A.u_off + r * A.u_stride + gid.x];
    const float arg = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    const float t = arg > 20.0f ? 1.0f : (arg < -20.0f ? -1.0f : metal::tanh(arg));
    SCR[A.dst_off + r * A.dst_stride + gid.x] = 0.5f * g * (1.0f + t) * u;
}

// ---- router softmax + top-k, one threadgroup per batch row ----
kernel void g4_router_topk_b(device float * SCR, const device uchar * PACK, constant G4RouterBArgs & A,
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
    const uint tid = tid3.x;
    if (tid != 0u || tg_id.x >= A.rows) return;
    device float * lg = SCR + A.logits_off + tg_id.x * A.logits_stride;
    const uint n = A.n_expert, k = A.top_k;
    float mx = lg[0];
    for (uint i = 1u; i < n; i++) mx = lg[i] > mx ? lg[i] : mx;
    float sum = 0.0f;
    for (uint i = 0u; i < n; i++) { lg[i] = metal::exp(lg[i] - mx); sum += lg[i]; }
    const float inv_sm = 1.0f / sum;
    for (uint i = 0u; i < n; i++) lg[i] *= inv_sm;
    device int * sel = (device int *)(SCR + A.sel_ids_off) + tg_id.x * A.top_k;
    device float * sw = SCR + A.sel_w_off + tg_id.x * A.top_k;
    float wsum = 0.0f;
    for (uint e = 0; e < k; e++) {
        int best = -1;
        float bv = -metal::numeric_limits<float>::max();
        for (uint i = 0; i < n; i++) {
            bool taken = false;
            for (uint q = 0; q < e; q++) if (sel[q] == (int)i) { taken = true; break; }
            if (!taken && lg[i] > bv) { bv = lg[i]; best = (int)i; }
        }
        sel[e] = best;
        sw[e] = lg[(uint)best];
        wsum += lg[(uint)best];
    }
    const float inv = 1.0f / (wsum + 1e-20f);
    const device float * ds = A.has_down_scale != 0
                            ? (const device float *)(PACK + A.down_scale_off) : nullptr;
    for (uint e = 0; e < k; e++)
        sw[e] = sw[e] * inv * (ds ? ds[(uint)sel[e]] : 1.0f);
}

// ---- batch MoE expert GEMV: grid (ceil(nout/4), ceil(rows/G4_MTILE)) x 128 ----
// Each of the G4_MTILE tile rows may pick a different expert; the tile is
// unrolled with static indices. Row lanes clamp (min) so loads stay in SCR
// and only in-bounds lanes store. Arithmetic per row == single kernel.
kernel void g4_moe_gemv_q4_k_b(device float * SCR, const device uchar * PACK, constant G4MoeGemvBArgs & A,
    uint3 tg_id3 [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sgs   [[simdgroups_per_threadgroup]]) {
    const uint2 tg_id2 = uint2(tg_id3.x, tg_id3.y);
    const uint row = tg_id2.x * sgs + sg_id;
    if (row >= A.nout) return;
    const uint r0 = tg_id2.y * G4_MTILE;
    const uint m1 = min(r0 + 1u, A.rows - 1u), m2 = min(r0 + 2u, A.rows - 1u),
               m3 = min(r0 + 3u, A.rows - 1u);
    const device int * sel = (const device int *)(SCR + A.sel_ids_off);
    const unsigned long eb0 = A.layer_base + (unsigned long)sel[r0 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb1 = A.layer_base + (unsigned long)sel[m1 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb2 = A.layer_base + (unsigned long)sel[m2 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb3 = A.layer_base + (unsigned long)sel[m3 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const device uchar * rb0 = PACK + eb0 + (unsigned long)row * A.rbytes;
    const device uchar * rb1 = PACK + eb1 + (unsigned long)row * A.rbytes;
    const device uchar * rb2 = PACK + eb2 + (unsigned long)row * A.rbytes;
    const device uchar * rb3 = PACK + eb3 + (unsigned long)row * A.rbytes;
    const device float * x0 = SCR + A.x_off + (unsigned long)r0 * A.x_stride;
    const device float * x1 = SCR + A.x_off + (unsigned long)m1 * A.x_stride;
    const device float * x2 = SCR + A.x_off + (unsigned long)m2 * A.x_stride;
    const device float * x3 = SCR + A.x_off + (unsigned long)m3 * A.x_stride;
    const uint nblocks = A.nin / 256u;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (uint i = 0; i < nblocks; i++) {
        {   const device uchar * blk = rb0 + i * 144u;
            const float d = g4_ld_f16(blk), dm = g4_ld_f16(blk + 2u);
            for (uint c = 0; c < 4u; c++) {
                uint sc0, m0, sc1, m1v;
                g4_get_scale_min_k4(2u * c,      blk + 4u, &sc0, &m0);
                g4_get_scale_min_k4(2u * c + 1u, blk + 4u, &sc1, &m1v);
                const uchar byte = blk[16u + c * 32u + lane];
                const uint base = i * 256u + c * 64u;
                a0 = metal::fma(d * float(sc0) * float(byte & 0xFu) - dm * float(m0), x0[base + lane], a0);
                a0 = metal::fma(d * float(sc1) * float(byte >> 4)   - dm * float(m1v), x0[base + 32u + lane], a0);
            } }
        {   const device uchar * blk = rb1 + i * 144u;
            const float d = g4_ld_f16(blk), dm = g4_ld_f16(blk + 2u);
            for (uint c = 0; c < 4u; c++) {
                uint sc0, m0, sc1, m1v;
                g4_get_scale_min_k4(2u * c,      blk + 4u, &sc0, &m0);
                g4_get_scale_min_k4(2u * c + 1u, blk + 4u, &sc1, &m1v);
                const uchar byte = blk[16u + c * 32u + lane];
                const uint base = i * 256u + c * 64u;
                a1 = metal::fma(d * float(sc0) * float(byte & 0xFu) - dm * float(m0), x1[base + lane], a1);
                a1 = metal::fma(d * float(sc1) * float(byte >> 4)   - dm * float(m1v), x1[base + 32u + lane], a1);
            } }
        {   const device uchar * blk = rb2 + i * 144u;
            const float d = g4_ld_f16(blk), dm = g4_ld_f16(blk + 2u);
            for (uint c = 0; c < 4u; c++) {
                uint sc0, m0, sc1, m1v;
                g4_get_scale_min_k4(2u * c,      blk + 4u, &sc0, &m0);
                g4_get_scale_min_k4(2u * c + 1u, blk + 4u, &sc1, &m1v);
                const uchar byte = blk[16u + c * 32u + lane];
                const uint base = i * 256u + c * 64u;
                a2 = metal::fma(d * float(sc0) * float(byte & 0xFu) - dm * float(m0), x2[base + lane], a2);
                a2 = metal::fma(d * float(sc1) * float(byte >> 4)   - dm * float(m1v), x2[base + 32u + lane], a2);
            } }
        {   const device uchar * blk = rb3 + i * 144u;
            const float d = g4_ld_f16(blk), dm = g4_ld_f16(blk + 2u);
            for (uint c = 0; c < 4u; c++) {
                uint sc0, m0, sc1, m1v;
                g4_get_scale_min_k4(2u * c,      blk + 4u, &sc0, &m0);
                g4_get_scale_min_k4(2u * c + 1u, blk + 4u, &sc1, &m1v);
                const uchar byte = blk[16u + c * 32u + lane];
                const uint base = i * 256u + c * 64u;
                a3 = metal::fma(d * float(sc0) * float(byte & 0xFu) - dm * float(m0), x3[base + lane], a3);
                a3 = metal::fma(d * float(sc1) * float(byte >> 4)   - dm * float(m1v), x3[base + 32u + lane], a3);
            } }
    }
    a0 = metal::simd_sum(a0); a1 = metal::simd_sum(a1);
    a2 = metal::simd_sum(a2); a3 = metal::simd_sum(a3);
    if (lane == 0u) {
        if (r0 < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)r0 * A.y_stride + row] += SCR[A.sel_w_off + r0 * A.k_slots + A.k_slot] * a0;
            else               SCR[A.y_off + (unsigned long)r0 * A.y_stride + row]  = a0;
        }
        if (r0 + 1u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m1 * A.y_stride + row] += SCR[A.sel_w_off + m1 * A.k_slots + A.k_slot] * a1;
            else               SCR[A.y_off + (unsigned long)m1 * A.y_stride + row]  = a1;
        }
        if (r0 + 2u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m2 * A.y_stride + row] += SCR[A.sel_w_off + m2 * A.k_slots + A.k_slot] * a2;
            else               SCR[A.y_off + (unsigned long)m2 * A.y_stride + row]  = a2;
        }
        if (r0 + 3u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m3 * A.y_stride + row] += SCR[A.sel_w_off + m3 * A.k_slots + A.k_slot] * a3;
            else               SCR[A.y_off + (unsigned long)m3 * A.y_stride + row]  = a3;
        }
    }
}

kernel void g4_moe_gemv_q5_0_b(device float * SCR, const device uchar * PACK, constant G4MoeGemvBArgs & A,
    uint3 tg_id3 [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sgs   [[simdgroups_per_threadgroup]]) {
    const uint2 tg_id2 = uint2(tg_id3.x, tg_id3.y);
    const uint row = tg_id2.x * sgs + sg_id;
    if (row >= A.nout) return;
    const uint r0 = tg_id2.y * G4_MTILE;
    const uint m1 = min(r0 + 1u, A.rows - 1u), m2 = min(r0 + 2u, A.rows - 1u),
               m3 = min(r0 + 3u, A.rows - 1u);
    const device int * sel = (const device int *)(SCR + A.sel_ids_off);
    const unsigned long eb0 = A.layer_base + (unsigned long)sel[r0 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb1 = A.layer_base + (unsigned long)sel[m1 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb2 = A.layer_base + (unsigned long)sel[m2 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb3 = A.layer_base + (unsigned long)sel[m3 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const device uchar * rb0 = PACK + eb0 + (unsigned long)row * A.rbytes;
    const device uchar * rb1 = PACK + eb1 + (unsigned long)row * A.rbytes;
    const device uchar * rb2 = PACK + eb2 + (unsigned long)row * A.rbytes;
    const device uchar * rb3 = PACK + eb3 + (unsigned long)row * A.rbytes;
    const device float * x0 = SCR + A.x_off + (unsigned long)r0 * A.x_stride;
    const device float * x1 = SCR + A.x_off + (unsigned long)m1 * A.x_stride;
    const device float * x2 = SCR + A.x_off + (unsigned long)m2 * A.x_stride;
    const device float * x3 = SCR + A.x_off + (unsigned long)m3 * A.x_stride;
    const uint nblocks = A.nin / 32u;
    const uint jj = lane >> 4u;
    const uint c  = lane & 15u;
    const uint qs = jj * 16u + c;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (uint b = 0; b < nblocks; b++) {
        {   const device uchar * blk = rb0 + b * 22u;
            const float d = g4_ld_f16(blk);
            const uint qh = g4_ld_u32(blk + 2u);
            const uint xh = ((qh >> (c + jj * 16u)) & 1u) << 4;
            const uint q  = blk[6u + c];
            const uint n  = jj ? ((q >> 4) | xh) : ((q & 0xFu) | xh);
            a0 = metal::fma(float(int(n) - 16) * d, x0[b * 32u + qs], a0); }
        {   const device uchar * blk = rb1 + b * 22u;
            const float d = g4_ld_f16(blk);
            const uint qh = g4_ld_u32(blk + 2u);
            const uint xh = ((qh >> (c + jj * 16u)) & 1u) << 4;
            const uint q  = blk[6u + c];
            const uint n  = jj ? ((q >> 4) | xh) : ((q & 0xFu) | xh);
            a1 = metal::fma(float(int(n) - 16) * d, x1[b * 32u + qs], a1); }
        {   const device uchar * blk = rb2 + b * 22u;
            const float d = g4_ld_f16(blk);
            const uint qh = g4_ld_u32(blk + 2u);
            const uint xh = ((qh >> (c + jj * 16u)) & 1u) << 4;
            const uint q  = blk[6u + c];
            const uint n  = jj ? ((q >> 4) | xh) : ((q & 0xFu) | xh);
            a2 = metal::fma(float(int(n) - 16) * d, x2[b * 32u + qs], a2); }
        {   const device uchar * blk = rb3 + b * 22u;
            const float d = g4_ld_f16(blk);
            const uint qh = g4_ld_u32(blk + 2u);
            const uint xh = ((qh >> (c + jj * 16u)) & 1u) << 4;
            const uint q  = blk[6u + c];
            const uint n  = jj ? ((q >> 4) | xh) : ((q & 0xFu) | xh);
            a3 = metal::fma(float(int(n) - 16) * d, x3[b * 32u + qs], a3); }
    }
    a0 = metal::simd_sum(a0); a1 = metal::simd_sum(a1);
    a2 = metal::simd_sum(a2); a3 = metal::simd_sum(a3);
    if (lane == 0u) {
        if (r0 < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)r0 * A.y_stride + row] += SCR[A.sel_w_off + r0 * A.k_slots + A.k_slot] * a0;
            else               SCR[A.y_off + (unsigned long)r0 * A.y_stride + row]  = a0;
        }
        if (r0 + 1u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m1 * A.y_stride + row] += SCR[A.sel_w_off + m1 * A.k_slots + A.k_slot] * a1;
            else               SCR[A.y_off + (unsigned long)m1 * A.y_stride + row]  = a1;
        }
        if (r0 + 2u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m2 * A.y_stride + row] += SCR[A.sel_w_off + m2 * A.k_slots + A.k_slot] * a2;
            else               SCR[A.y_off + (unsigned long)m2 * A.y_stride + row]  = a2;
        }
        if (r0 + 3u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m3 * A.y_stride + row] += SCR[A.sel_w_off + m3 * A.k_slots + A.k_slot] * a3;
            else               SCR[A.y_off + (unsigned long)m3 * A.y_stride + row]  = a3;
        }
    }
}

kernel void g4_moe_gemv_q8_0_b(device float * SCR, const device uchar * PACK, constant G4MoeGemvBArgs & A,
    uint3 tg_id3 [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sgs   [[simdgroups_per_threadgroup]]) {
    const uint2 tg_id2 = uint2(tg_id3.x, tg_id3.y);
    const uint row = tg_id2.x * sgs + sg_id;
    if (row >= A.nout) return;
    const uint r0 = tg_id2.y * G4_MTILE;
    const uint m1 = min(r0 + 1u, A.rows - 1u), m2 = min(r0 + 2u, A.rows - 1u),
               m3 = min(r0 + 3u, A.rows - 1u);
    const device int * sel = (const device int *)(SCR + A.sel_ids_off);
    const unsigned long eb0 = A.layer_base + (unsigned long)sel[r0 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb1 = A.layer_base + (unsigned long)sel[m1 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb2 = A.layer_base + (unsigned long)sel[m2 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const unsigned long eb3 = A.layer_base + (unsigned long)sel[m3 * A.k_slots + A.k_slot] * A.expert_stride + A.part_off;
    const device uchar * rb0 = PACK + eb0 + (unsigned long)row * A.rbytes;
    const device uchar * rb1 = PACK + eb1 + (unsigned long)row * A.rbytes;
    const device uchar * rb2 = PACK + eb2 + (unsigned long)row * A.rbytes;
    const device uchar * rb3 = PACK + eb3 + (unsigned long)row * A.rbytes;
    const device float * x0 = SCR + A.x_off + (unsigned long)r0 * A.x_stride;
    const device float * x1 = SCR + A.x_off + (unsigned long)m1 * A.x_stride;
    const device float * x2 = SCR + A.x_off + (unsigned long)m2 * A.x_stride;
    const device float * x3 = SCR + A.x_off + (unsigned long)m3 * A.x_stride;
    const uint nblocks = A.nin / 32u;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (uint b = 0; b < nblocks; b++) {
        {   const device uchar * blk = rb0 + b * 34u;
            const float d = g4_ld_f16(blk);
            const uint qs = b * 32u + lane;
            a0 = metal::fma(d * float((char)blk[2u + lane]), x0[qs], a0); }
        {   const device uchar * blk = rb1 + b * 34u;
            const float d = g4_ld_f16(blk);
            const uint qs = b * 32u + lane;
            a1 = metal::fma(d * float((char)blk[2u + lane]), x1[qs], a1); }
        {   const device uchar * blk = rb2 + b * 34u;
            const float d = g4_ld_f16(blk);
            const uint qs = b * 32u + lane;
            a2 = metal::fma(d * float((char)blk[2u + lane]), x2[qs], a2); }
        {   const device uchar * blk = rb3 + b * 34u;
            const float d = g4_ld_f16(blk);
            const uint qs = b * 32u + lane;
            a3 = metal::fma(d * float((char)blk[2u + lane]), x3[qs], a3); }
    }
    a0 = metal::simd_sum(a0); a1 = metal::simd_sum(a1);
    a2 = metal::simd_sum(a2); a3 = metal::simd_sum(a3);
    if (lane == 0u) {
        if (r0 < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)r0 * A.y_stride + row] += SCR[A.sel_w_off + r0 * A.k_slots + A.k_slot] * a0;
            else               SCR[A.y_off + (unsigned long)r0 * A.y_stride + row]  = a0;
        }
        if (r0 + 1u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m1 * A.y_stride + row] += SCR[A.sel_w_off + m1 * A.k_slots + A.k_slot] * a1;
            else               SCR[A.y_off + (unsigned long)m1 * A.y_stride + row]  = a1;
        }
        if (r0 + 2u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m2 * A.y_stride + row] += SCR[A.sel_w_off + m2 * A.k_slots + A.k_slot] * a2;
            else               SCR[A.y_off + (unsigned long)m2 * A.y_stride + row]  = a2;
        }
        if (r0 + 3u < A.rows) {
            if (A.accum != 0u) SCR[A.y_off + (unsigned long)m3 * A.y_stride + row] += SCR[A.sel_w_off + m3 * A.k_slots + A.k_slot] * a3;
            else               SCR[A.y_off + (unsigned long)m3 * A.y_stride + row]  = a3;
        }
    }
}
