// m64_gemma4_args.h — Milestone 6B-1: plain-POD kernel argument structs shared
// between engine/src/forward_metal_gemma4.mm (C++/ObjC++) and
// engine/kernels/m64_gemma4.metal.
//
// This header is #include'd ONLY by C++/ObjC++ translation units. The Metal
// runtime concatenates engine/kernels/*.metal into ONE compilation unit, and
// m64_gemma4.metal mirrors the exact same struct definitions inline (MSL
// cannot resolve local #include from newLibraryWithSource: string sources).
// Keep the two copies byte-for-byte identical.
//
// Layout policy (same as m64_p2a_args.h): only unsigned int / unsigned long /
// int / float fields with natural alignment, so C++ (LP64) and MSL agree
// byte-for-byte. *off / x_off / *_off fields are FLOAT offsets into the SCR
// scratch buffer (32-bit safe); woff / layer_base / freq_off / woff fields on
// PACK are BYTE offsets into the .m64pack blob (26B model > 4 GB, so 64-bit).
//
// NOTE: <cstdint> is included for the C++ side only; <cstdint> is NOT
// reachable in the concatenated MSL unit, so do NOT use uint32_t/int64_t etc.
// inside the structs below.
#pragma once
#include <cstdint>

// ---- 1. Embedding gather ----------------------------------------------------
struct G4EmbArgs {
    unsigned long woff;        // PACK: embedding table row 0
    unsigned int  dst_off;     // SCR float offset of destination row
    unsigned int  n_embd;      // row length (elements)
    unsigned int  row_bytes;   // PACK bytes per row (dtype-block sized)
    unsigned int  dtype;       // G4EmbDtype
    float         scale;       // post-multiply (sqrt(n_embd) at runtime)
};
static_assert(sizeof(G4EmbArgs) == 32, "G4EmbArgs layout");

enum G4EmbDtype {
    G4_DT_F32  = 0,
    G4_DT_F16  = 1,
    G4_DT_Q8_0 = 2,
    G4_DT_Q4_K = 3,
    G4_DT_Q5_0 = 4,
};

// ---- 2. RMSNorm --------------------------------------------------------------
// rows threadgroups x 256 threads. src_off == dst_off (in-place) is race-free:
// a threadgroup barrier sits between the ssq reduction and the write-back.
struct G4RmsArgs {
    unsigned int  src_off, dst_off;   // SCR float offsets
    unsigned long woff;               // PACK f32 weights (used when has_w)
    unsigned int  k;                  // row length (elements)
    unsigned int  rows;
    unsigned int  has_w;              // 1: multiply by (1 + w) form weight
    float         eps;
    float         post_scale;         // 1.0f normally; 1/sqrt(n_embd) for router
};
static_assert(sizeof(G4RmsArgs) == 40, "G4RmsArgs layout");

// ---- 3. NEOX RoPE ------------------------------------------------------------
// grid x = j in [0, n_rot/2), y = head in [0, n_heads).
struct G4RopeArgs {
    unsigned int  off;           // SCR float offset of head h=0, token row
    unsigned int  n_heads;       // Q heads for Q-tensor, KV heads for K-tensor
    unsigned int  head_dim;      // full head stride in elements
    unsigned int  n_rot;         // rotary dim (pairs are (j, j + n_rot/2))
    unsigned long freq_off;      // PACK f32 rope_freqs[j] (valid when has_freqs)
    float         freq_base;     // rope_theta
    int           has_freqs;
};
static_assert(sizeof(G4RopeArgs) == 32, "G4RopeArgs layout");

// ---- 4. Dense GEMV (1 row per simdgroup, 4 simdgroups / 128 threads per TG) -
// rbytes = row bytes for this dtype: f32 = nin*4, q8_0 = (nin/32)*34,
// q5_0 = (nin/32)*22, q4_k = (nin/256)*144. nin must be a multiple of the
// dtype block size.
struct G4GemvArgs {
    unsigned long woff;          // PACK weight row 0
    unsigned int  x_off, y_off;  // SCR: input[nin] -> output[nout]
    unsigned int  nout, nin;
    unsigned int  rbytes;
};
static_assert(sizeof(G4GemvArgs) == 32, "G4GemvArgs layout");

// ---- 5. MoE expert GEMV ------------------------------------------------------
// base = PACK + layer_base + sel_ids[k_slot] * expert_stride + part_off
struct G4MoeGemvArgs {
    unsigned long layer_base;    // PACK: expert 0 record start
    unsigned long expert_stride; // bytes per expert record
    unsigned long part_off;      // bytes into the record (gate=0, up=pg, down=pg+pu)
    unsigned int  x_off, y_off;
    unsigned int  sel_ids_off;   // SCR int32 offset (router output)
    unsigned int  sel_w_off;     // SCR float offset
    unsigned int  k_slot;
    unsigned int  nout, nin;
    unsigned int  rbytes;
    unsigned int  accum;         // 1: y[row] += sel_w[k_slot] * dot
};
static_assert(sizeof(G4MoeGemvArgs) == 64, "G4MoeGemvArgs layout");

// ---- 6. KV cache write + decode attention -----------------------------------
// K rows at SCR + k_off (n_kv * hd), V rows at SCR + v_off. Token slot:
//   slot = is_swa ? (pos % ring_cap) : pos
// FP16 buffers: 2 B/elem at (slot*n_kv*hd + head*hd + i) * 2.
// Q8_0 buffers: 32-elem blocks along hd, 34 B/block (half d; char qs[32]),
//   block index ((slot*n_kv + head) * nb_head + b), nb_head = hd/32.
struct G4KvArgs {
    unsigned int  k_off, v_off;  // SCR float offsets
    unsigned int  n_kv, hd;
    unsigned int  ring_cap;      // resident KV ring capacity (tokens)
    int           is_swa;
};
static_assert(sizeof(G4KvArgs) == 24, "G4KvArgs layout");

// 1 threadgroup (256 threads) per Q head. scale is EXACTLY 1.0f (Gemma fused
// attention scale — no 1/sqrt(hd)).
struct G4AttnArgs {
    unsigned int  q_off;         // SCR float offset of rope'd Q head 0
    unsigned int  out_off;       // SCR float offset of attention output head 0
    unsigned int  n_head;        // Q heads
    unsigned int  n_kv;
    unsigned int  hd;            // head dim (<= 512 for the 256-thread kernel)
    unsigned int  gqa_group;     // n_head / n_kv
    unsigned int  ring_cap;
    int           is_swa;
    int           n_swa;         // sliding-window tokens; <=0 disables the clamp
};
static_assert(sizeof(G4AttnArgs) == 36, "G4AttnArgs layout");

// ---- 7. Activations / router / vector ops ------------------------------------
// dst[i] = 0.5 * g * (1 + tanh(0.79788456*(g + 0.044715 g^3))) * u
struct G4GeGLUArgs {
    unsigned int  g_off, u_off, dst_off;
    unsigned int  n;
};
static_assert(sizeof(G4GeGLUArgs) == 16, "G4GeGLUArgs layout");

// 1 threadgroup, single executing thread (tid==0): softmax over router logits
// then deterministic greedy top-k (strict >, smallest index wins ties), then
// renormalize / (wsum + 1e-20) and optional down_scale[sel_id]. Matches
// gemma4_forward.cpp bit-for-bit for n_expert <= 1024.
struct G4RouterArgs {
    unsigned int  logits_off;
    unsigned int  sel_ids_off;   // int32[n] at SCR
    unsigned int  sel_w_off;     // float[n] at SCR
    unsigned int  n_expert, top_k;
    unsigned long down_scale_off;  // PACK f32[n_expert] (valid when has_down_scale)
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

// dst[i] *= ((const device float *)(PACK + woff))[0]
struct G4ScaleArgs {
    unsigned int  dst_off;
    unsigned int  n;
    unsigned long woff;          // PACK: single f32 scalar
};
static_assert(sizeof(G4ScaleArgs) == 16, "G4ScaleArgs layout");

// out[i] = cap * tanh(x[i] / cap), n up to vocab (262144), grid-stride.
struct G4SoftcapArgs {
    unsigned int  src_off, dst_off;
    unsigned int  n;
    float         cap;
};
static_assert(sizeof(G4SoftcapArgs) == 16, "G4SoftcapArgs layout");

// Single-dispatch EXACT argmax: one threadgroup ((1,1,1) x (256,1,1)).
// Each thread keeps a local (best value, smallest winning index) pair over a
// strided slice of the source, then a 256-entry tree reduction with strict >
// and smallest-index tie-breaking (matching the CPU oracle's sequential
// first-max scan) writes the int32 winner to SCR + out_off. Exact for any
// finite (including negative) float values and any index width (no
// sortable-key encoding). G4_ARGMAX_CAND / arena_off are legacy from the
// removed two-pass scheme: the arena slot stays reserved in the SCR layout.
#define G4_ARGMAX_CAND 32
struct G4ArgmaxArgs {
    unsigned int  src_off;
    unsigned int  out_off;       // int32 result at SCR
    unsigned int  arena_off;     // legacy 32-float SCR slot (unused)
    unsigned int  n;
};
static_assert(sizeof(G4ArgmaxArgs) == 16, "G4ArgmaxArgs layout");
