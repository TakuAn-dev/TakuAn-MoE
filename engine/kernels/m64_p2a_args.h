// m64_p2a_args.h — plain-POD kernel argument structs shared between
// engine/kernels/m64_p2a.metal and engine/src/forward_metal.mm.
// uint fields are FLOAT offsets into the scratch buffer (SCR, ~30 MB, 32-bit
// safe); fields named *pack/*off-on-PACK are BYTE offsets into the wrapped
// .m64pack file — that file is 27.8 GB, so every PACK offset is 64-bit.
// Only uint / unsigned long / float fields (natural alignment,
// identical layout both sides — same clang C ABI).
#pragma once

#define M64_MAXS 48
// PACK offset sentinel meaning "no scale tensor" (pack offsets are >=4KB
// aligned, so 0xFFFFFFFF can never collide).
#define M64_NOFF 0xFFFFFFFFul

struct RMSArgs {
    unsigned int xr[M64_MAXS], yr[M64_MAXS];      // SCRATCH float offsets
    unsigned long wr[M64_MAXS];              // PACK offsets (f32 weights)
    unsigned int k;       // row length in elements
    unsigned int rows;
    unsigned int hasw;    // 1: multiply by w (PACK)
    float eps;
};

struct QuantArgs {
    unsigned int xoff, qoff, doff;  // SCRATCH: f32 in, int8 out, f32 d32
    unsigned int nb;
};

struct GemvArgs {
    unsigned long woff, woff2; // PACK (woff2 unused unless gateup variant)
    unsigned int xqoff, dxoff;      // SCRATCH
    unsigned int yoff, yoff2;       // SCRATCH
    unsigned int nout;
    unsigned int idx, stride;       // expert slot (index into seloff) & blob stride bytes
    unsigned int seloff;            // SCRATCH: router sel[] array (MoE variants)
};

struct MoeDownArgs {
    unsigned long woff;        // PACK
    unsigned int xqoff, dxoff, accoff, wtoff;  // SCRATCH
    unsigned int idx;               // expert slot index into seloff/wtoff
    unsigned int nout;
    unsigned int stride;            // expert blob stride bytes
    unsigned int seloff;            // SCRATCH: router sel[] array
};

struct GemvF32Args {
    unsigned long woff;        // PACK f32
    unsigned int xoff, yoff;
    unsigned int nout, k;
};

struct ConvArgs {
    unsigned int rsoff;             // SCRATCH: rs[slot][sec][0]
    unsigned int xoff, yoff;
    unsigned long woffpack;    // PACK wconv [C][4]
    unsigned int dynoff;            // SCRATCH (dynb)
    unsigned int C, group;
    float scale;
    unsigned int hasdyn;
};

struct ClipArgs {
    unsigned int coff;              // SCRATCH [groups*per]
    unsigned long soff;        // PACK [groups*per] or M64_NOFF
    unsigned int groups;
    unsigned int per;               // 128 or 16
    float thresh;
};

struct BiasArgs {
    unsigned long relbiaspack; // PACK [EXT][16]
    unsigned int sinkoff, rout, btoff;
    unsigned int nkv, pos, ctx;
};

struct EmbArgs { unsigned long off; unsigned int boff, n; };

struct AttnArgs {
    unsigned int qoff, rout, sinkoff;
    unsigned long relbiaspack;
    unsigned int kvkoff, kvvoff, outoff;
    unsigned int nkv, pos, is_swa, fa16, n_swa;
    float scale;
};

// ---------------- FP16 KV cache + SWA ring buffer + 128K context ----------------
// KV write: converts one token's fp32 K/V rows (SCR) into the shared FP16 KV
// buffer. dst_k/dst_v are HALF-element offsets (host pre-computes the ring slot).
struct KVWriteF16Args {
    unsigned int koff, voff, n, pad;
    unsigned long dst_k, dst_v;  // half-element offsets in KV buffer
};

// Attention over the FP16 KV buffer. tau folds the long-context log-scale into
// q and rout at exactly the points forward_impl.hpp applies Q *= tau_v /
// Rr *= tau_v (tau == 1.0f on SWA layers). Ring slot = j % ring_cap for SWA.
struct AttnF16Args {
    unsigned long relbiaspack;   // byte offset in PACK
    unsigned long kvkoff;        // half-element offset in KV
    unsigned long kvvoff;        // half-element offset in KV
    unsigned int qoff, rout, sinkoff, outoff;
    unsigned int pos, is_swa, n_swa, ring_cap;
    float scale, tau;
};

// Split-K FlashDecoding over the FP16 KV buffer (`k_attn_head_f16_split` +
// `k_attn_head_f16_reduce`). Parallelizes long-context KV scans across
// `n_splits` threadgroups per head, then reduces partial (M, S, O).
struct AttnF16SplitArgs {
    unsigned long relbiaspack;   // byte offset in PACK
    unsigned long kvkoff;        // half-element offset in KV
    unsigned long kvvoff;        // half-element offset in KV
    unsigned int qoff, rout, sinkoff, outoff;
    unsigned int part_o_off, part_ms_off;
    unsigned int pos, is_swa, n_swa, ring_cap;
    unsigned int n_splits, pad0;
    float scale, tau;
};

struct GateMulArgs { unsigned int aoff, goff, n; };

struct SituArgs {
    unsigned int goff, uoff, qoff, doff;
    unsigned int nb;
    float beta, lb, inv_beta, inv_lb;
};

struct RouterArgs {
    unsigned int probs;
    unsigned long biaspack;
    unsigned int seloff, wtoff;
    float scale;
};

struct SoftcapArgs { unsigned int loff, n; float cap; };
struct ArgmaxArgs  { unsigned int loff, aoff, n; };

struct XferArgs {                  // mode 0: c=a+b in place at a; mode 1: pairs copy
    unsigned int a[2], b[2], c[2], n[2];
    unsigned int mode;
};

struct ScaleArgs { unsigned int off, n; float v; };

struct CtrlArgs {                  // grid.x = 1..4 threadgroups of 1024
    unsigned int in[4], msg[4], out[4];           // SCRATCH (in = rms pair row0 src)
    unsigned long wdown[4], wgain[4], wforg[4];  // PACK
    unsigned int gcat;                            // SCRATCH [4][2][NE] rms pairs
    unsigned int gb[4], fb[4];                    // SCRATCH gain/forget rows
    float eps;
    unsigned int grank, ne;
};

// Phase-2A-opt R4: control block split into two wide-dispatch kernels
// (k_ctrl_down grid (64, nc) x 256th; k_ctrl_up grid (ceil(ne/32), nc) x 32th).
// Natural alignment: 3*4*u32 = 48B (u64-aligned), total 168B.
struct CtrlSplitArgs {
    unsigned int in[4], msg[4], out[4];           // SCRATCH token rows
    unsigned long wdown[4], wgain[4], wforg[4];   // PACK
    unsigned int gcat;                            // SCRATCH [nc][2][NE] rms pairs
    unsigned int hh;                              // SCRATCH [nc][64] tanh(down)
    float eps;
    unsigned int grank, ne, nc;
};

struct MixDotArgs {
    unsigned int stnoff, scoff;
    unsigned long qpack;
    unsigned int S, scale_dummy;
    float scale;
};

struct MixSoftArgs { unsigned int scoff; unsigned int S; };
struct MixCombArgs { unsigned int stnoff, scoff, outoff, S, nq; };

// ---------------- Phase-2A-opt fused / batched kernels ----------------
#define M64_NOFFU 0xFFFFFFFFu

// rmsnorm (1..M64_MAXS rows, tree-reduced like k_rms) fused with Q8_0
// quantization. yr[r]==M64_NOFFU: skip float output; q[r]==M64_NOFFU: skip
// quant output. grid.x = rows threadgroups of 256.
struct RmsQuantArgs {
    unsigned int xr[M64_MAXS], yr[M64_MAXS], q[M64_MAXS], d[M64_MAXS];
    unsigned long wr[M64_MAXS];
    unsigned int k, rows, hasw;
    float eps;
};

// gate*sigmoid(gate) fused with Q8_0 quant. n = nb*32 elements.
// grid = nb*32 threads (32/lane per block, simd_max reduction).
struct GateMulQuantArgs { unsigned int aoff, goff, qoff, doff, nb; };

// ---------------- Phase-2A-opt R3 multi-projection GEMV & fused ops ----------------
// Fuse 1..6 Q8_0 k=2304 GEMVs sharing (xqoff, dxoff) into one dispatch.
struct MultiGemvArgs {
    unsigned long woff[6];
    unsigned int yoff[6];
    unsigned int nout[6];
    unsigned int cum[6];      // prefix sums: cum[i] = sum_{j<=i} nout[j]
    unsigned int xqoff, dxoff;
    unsigned int count;       // 1..6
    unsigned int total;       // cum[count - 1]
};

struct DownReduceConvArgs {
    unsigned int tmpoff, wtoff, rsoff, curoff;  // SCR: down_tmp[8*kNE], wgt[8], rs, cur[kNE]
    unsigned long wconv;                        // PACK: ffnconv [kNE][4]
    unsigned int nout;                          // kNE = 2304
    float scale;                                // kConvStatic
};

struct MixFusedArgs {
    unsigned long qpack;                        // PACK: query [nq][kNE]
    unsigned int stnoff, outoff, S, nq;         // SCR
    float scale;
    unsigned int _pad;
};

// ---------------- Milestone-3A batched-prefill kernels (m64_prefill.metal) ----------------
// Q8_0 batched GEMM: y[t][row] = dot(w_row, xq_t). Weights are packed Q8_0
// rows: nb = k/32 blocks of (f16 d, 32 int8) = 34 B, so row stride is
// nb*34 B (units of 2 blocks = 68 B, odd tail block = 34 B). xq holds one
// token's int8 codes (byte view of SCR floats), dx the per-32 f32 scales.
// All *stride fields are SCR float-element strides. sizeof == 48.
struct GemmBatchArgs {
    unsigned long woff;                         // PACK: Q8_0 weight rows
    unsigned int xqoff, dxoff, yoff, nout;
    unsigned int k, B, xq_stride, dx_stride;
    unsigned int y_stride, pad0;
};

// Batched depthwise k_conv_slot: sliding window over B tokens per channel,
// conv state rs[c][3] loaded/stored once. dyn row for token b:
// SCR + dynoff + b*dyn_stride + (c/group)*4. sizeof == 64.
struct ConvBatchArgs {
    unsigned long woff;                         // PACK: static wconv [C][4] f32
    unsigned int xoff, yoff, rsoff, dynoff;
    unsigned int C, B, x_stride, y_stride;
    unsigned int dyn_stride, group, hasdyn, pad0;
    float scale, pad1;
};

// Batched FP16 KV write: token b (pos0+b) -> ring slot (is_swa ?
// (pos0+b) % ring_cap : pos0+b). kvk_base/kvv_base are HALF-element offsets.
// sizeof == 56.
struct KVWriteBatchF16Args {
    unsigned long kvk_base, kvv_base;           // half-element base offsets in KV
    unsigned int koff, voff, k_stride, v_stride;
    unsigned int n, B, pos0, is_swa;
    unsigned int ring_cap, pad0;
};

// Batched causal attention over FP16 KV for B consecutive tokens
// (pos = pos0 + b). tau (long-context log-scale) is computed per token
// inside the kernel: is_swa layers use tau == 1, otherwise
// log(max(1,(pos+1)/log_scale_floor))*log_scale_alpha + 1.
// Numerics match k_attn_head_f16 1:1. sizeof == 96.
struct AttnBatchF16Args {
    unsigned long relbiaspack, kvkoff, kvvoff;
    unsigned int qoff, rout, sinkoff, outoff;
    unsigned int q_stride, rout_stride, sink_stride, out_stride;
    unsigned int pos0, B, is_swa, n_swa;
    unsigned int ring_cap, pad0;
    float scale, log_scale_floor, log_scale_alpha, pad1;
};

// Grouped (per-expert) Q8_0 GEMM for MoE gate/up/down during prefill.
// Expert exp owns SCR tokens via list[offset[exp] .. +count[exp]-1] of
// "pair ids" p: is_down=0 (gate/up) input token = p/topk, output row-index
// = p (topk pairs per token); is_down=1 (down) list entries are already
// unique input==output token indices. part_off selects gate|up|down part
// inside the expert blob (PACK + wbase + exp*exp_stride + part_off).
// sizeof == 80.
struct MoeGroupedGemmArgs {
    unsigned long wbase, exp_stride, part_off;  // PACK
    unsigned int xqoff, dxoff, yoff, list_off;
    unsigned int count_off, offset_off, nout, k;
    unsigned int xq_stride, dx_stride, y_stride, n_experts;
    unsigned int topk, is_down;
    // Milestone 3B-2 arena mode: when wtab_off != 0, expert exp's arena slot
    // index comes from ((const uint *)(SCR + wtab_off))[exp] and is scaled by
    // exp_stride in 64-bit instead of exp. 0 = disabled (resident full-pack).
    unsigned int wtab_off, pad2;
};

// ---------------- Milestone-3B-1 kernels (m64_prefill.metal) ----------------
// simdgroup_matrix Q8_0 GEMM with FP32 activations (no k_quant pass):
// y[b][m] = (add_to_out ? y[b][m] : 0) + dot(W_m, x_b), k a multiple of 32.
// W = PACK Q8_0 rows (row stride (k/32)*34 B); X/Y = SCR float element offsets,
// x_stride / y_stride float strides between token rows. Tile 32m x 32b x 32k,
// grid = (ceil(nout/32), ceil(B/32)), threadgroup = (32, 4) = 4 simdgroups.
// sizeof == 48.
struct GemmF32SgArgs {
    unsigned long woff;                         // PACK: Q8_0 weight rows
    unsigned int xoff, x_stride;                // SCR fp32 activations
    unsigned int yoff, y_stride;                // SCR fp32 output
    unsigned int k, nout, B;
    unsigned int add_to_out;                    // 1: accumulate into Y
    unsigned int pad0;
};

// Batched embedding row gather (matches k_emb_gather per token): token id
// tok[b] (uint32 stored in a SCR float slot) selects the PACK row at
// woff + tok[b] * row_blocks * 34; writes f32 rows to SCR. sizeof == 40.
struct EmbBatchArgs {
    unsigned long woff;                         // PACK: embedding table base
    unsigned int tok_off;                       // SCR: uint32 token ids
    unsigned int boff, boff_stride;             // SCR: f32 row base / token stride
    unsigned int row_blocks, B, pad0, pad1;
};

// Batched rmsnorm (bitwise-matches k_rms per (token, row) with nth=256).
// Row (b, r) input lives at xoff + b*x_stride + r*x_row; f32 output at
// yoff + b*y_stride + r*y_row (yoff == M64_NOFFU: skip f32 output). Optional
// Q8_0 output mirrors k_quant block math: codes as bytes starting at
// (uchar*)SCR + 4*(qoff + b*q_stride + r*k/4) (q_stride float units >=
// rows*k/4), d32 at doff + b*d_stride + r*nb (d_stride >= rows*nb).
// grid = (rows, B) threadgroups of 256. sizeof == 80.
struct RmsBatchArgs {
    unsigned long wpack;                        // PACK: shared f32 scale row (hasw=1)
    unsigned int xoff, x_stride, x_row;
    unsigned int yoff, y_stride, y_row;
    unsigned int qoff, doff, q_stride, d_stride;
    unsigned int k, B, rows;
    unsigned int hasw, q8, pad0;
    float eps;
};

// Batched head clip (matches k_clip per token; nth=128): group g of token b
// at coff + b*c_stride + g*per, in place. Shared PACK scales row across
// tokens, or M64_NOFF. grid = (groups, B) threadgroups of 128. sizeof == 40.
struct ClipBatchArgs {
    unsigned long soff;                         // PACK [groups*per] or M64_NOFF
    unsigned int coff, c_stride;                // SCR, in place
    unsigned int groups, per, B, pad0;
    float thresh;
};

// Batched a *= sigmoid(g) (matches k_gate_mul per token). sizeof == 24.
struct GateMulBatchArgs {
    unsigned int aoff, goff, a_stride, g_stride, n, B;
};

// Batched SITU activation + Q8_0 quant over npairs (token, expert) rows
// (matches k_situ_quant per block): block (p, j) reads g/u at
// goff/uoff + p*stride + j*32, writes q bytes / d32 like SituArgs.
// q_stride is SCR FLOAT units per pair (>= nb*8, multiple of 4).
// grid = (nb, npairs) threads of 32. sizeof == 56.
struct SituBatchArgs {
    unsigned int goff, uoff, qoff, doff;
    unsigned int g_stride, u_stride, q_stride, d_stride;
    unsigned int nb, npairs;
    float beta, lb, inv_beta, inv_lb;
};

// Batched MoE down-projection weighted reduce (reduce math identical to
// k_down_reduce8_conv_add / k_gemv_down_reduce8: e ascending, plain mul+add):
// out[b][c] = sum_{e<topk} w[b][e] * in[b*topk + e][c].
// grid = (ceil(nout/128), B) threads of 128. sizeof == 40.
struct MoeReduceBatchArgs {
    unsigned int inoff, wtoff, outoff;          // SCR
    unsigned int in_stride, w_stride, out_stride;
    unsigned int nout, topk, B, pad0;
};

// Batched router top-8 (matches k_router_topk per token; 128 threads/token,
// 4-warp simd_max argmax, ties -> smallest index). grid = (B, 1) groups of
// 128. sizeof == 48.
struct RouterBatchArgs {
    unsigned long biaspack;                     // PACK router bias [128] f32
    unsigned int probs, seloff, wtoff;          // SCR bases (token 0)
    unsigned int p_stride, sel_stride, w_stride;
    unsigned int B, pad0;
    float scale;
};

// Batched FP32 GEMV (router): y[b][row] = dot(w_row fp32, x_b fp32).
// Reduction identical to k_gemv_f32_tg (float4 dot + 2-level simd reduce),
// so bitwise equals running k_gemv_f32_tg once per token.
// Grid (ceil(nout), B) threadgroups x 256 threads. sizeof == 40.
struct GemvF32BatchArgs {
    unsigned long woff;                         // PACK: fp32 row-major [nout][k]
    unsigned int xoff, yoff, x_stride, y_stride;
    unsigned int nout, k, B, pad0;
};

// MoE bucket builder: one threadgroup builds the CSR consumed by
// k_moe_grouped_gemm_q8 from sel[] (topk pair -> expert id, uint32 stored
// as float bits). counts/offsets hold expert counts + exclusive-sum offsets
// (uint32 as float bits, all n_experts entries always written); list holds
// pair ids p in per-expert order. n_experts <= 128. sizeof == 32.
struct MoeBucketArgs {
    unsigned int sel_off, count_off, offset_off, list_off;
    unsigned int n_experts, npairs, pad0, pad1;
};

// ---- Milestone 3C: tensor-core grouped MoE with FP32 activations ----
// k_moe_grouped_f32in_sg: grid = ((nout + 31)/32, n_experts, 1),
// threadgroup = (32, 4, 1). When wtab_off != 0 the expert's slot index is
// read from ((const uint *)(SCR + wtab_off))[exp] (64-bit exp_stride math,
// never a uint->float cast); 0 = resident full-pack identity mapping.
struct MoeGroupedF32SgArgs {
    unsigned long wbase, exp_stride, part_off;  // PACK (or arena expertBuf)
    unsigned int xoff, yoff, list_off, count_off;
    unsigned int offset_off, wtab_off, nout, k;
    unsigned int x_stride, y_stride, n_experts, topk;
    unsigned int is_down, pad0;
};

// k_situ_f32_batch: grid = (k, npairs), thread = (i, pair). Pure FP32 SITU
// activation (identical math to k_situ_quant_batch without the Q8 store).
struct SituF32BatchArgs {
    unsigned int goff, uoff, hoff;
    unsigned int g_stride, u_stride, h_stride;
    unsigned int k, npairs;
    float beta, lb, inv_beta, inv_lb;
};

// ---------------- Milestone 4B: FP32-input decode MoE, Q4_K / Q5_0 / Q5_1 ----------------
// Batch-8 MoE gate|up GEMV over FP32 activations (no k_quant pass). Weights are
// PACK Q4_K rows: row bytes = (k/QK_K)*144 (Inkling gate/up: k=2304 -> 9 blocks,
// 1296 B). Expert exp of slot e: exp_id = sel[e] (SCR uint32), arena slot index
// slot = wtab_off ? ((const uint *)(SCR + wtab_off))[exp_id] : exp_id, row base
// = PACK + (up ? woff2 : woff) + (u64)slot * stride + r * row_bytes. Same layout
// as k_gemv_gateup_batch8: gate rows at woff, up rows at woff2 (row r in both).
// grid.x = ceil(8*2*nout / 8) threadgroups of 256 (warp per row).
// Requires woff % 4 == woff2 % 4 == stride % 4 == 0 (uint-4B loads inside the
// 144 B block; f16 scales are byte-wise). sizeof == 48.
struct MoeGateUpQ4KF32Args {
    unsigned long woff, woff2;                  // PACK: Q4_K gate / up rows
    unsigned int xoff;                          // SCR fp32 activations [k]
    unsigned int yoff, yoff2;                   // SCR fp32 out [8][nout] (gate/up)
    unsigned int seloff, wtab_off;              // SCR uint32: router sel / arena slot table (0 = direct)
    unsigned int nout, k, stride;               // k % 256 == 0; stride = expert blob bytes
};

// Batch-8 MoE down-proj + weighted reduce over FP32 activations, Q5_0 / Q5_1
// weights: acc[o] = sum_{e<8} wgt[e] * dot(row o of expert sel[e], h). Row
// bytes = (k/32)*22 (Q5_0) or (k/32)*24 (Q5_1) (Inkling down: k=1152 -> 36
// blocks). Arena indirection identical to MoeGateUpQ4KF32Args. Expert loop is
// e-ascending with plain mul+add (k_gemv_down_reduce8 oracle order); all-8
// per-row dots are FP32-simd-tree sum-reduce class. grid.x = ceil(nout / 8)
// threadgroups of 256 (warp per row; all k/32 blocks per row are ushort/byte
// loads, so Q5_0's 22 B block needs only even PACK offsets). sizeof == 48.
struct MoeDownQ5F32Args {
    unsigned long woff;                         // PACK: Q5_0 / Q5_1 down rows
    unsigned int hoff, accoff;                  // SCR fp32 in [8][k] (expert e row at hoff+e*k) / out [nout]
    unsigned int wtoff, seloff;                 // SCR fp32 weights [8] / uint32 sel [8]
    unsigned int wtab_off, pad0;                // SCR uint32 arena slot table (0 = direct)
    unsigned int nout, k, stride, pad1;         // k % 32 == 0; stride = expert blob bytes
};
// ===================== Milestone 9B: batched small prefill helpers =====================
// Batched (grid.y/.z = token b in [0,B)) clones of the tiny per-token prefill
// kernels k_xfer / k_rms+k_mix_fused / k_ctrl_down+k_ctrl_up. Element shapes
// are copied verbatim from the single-token kernels so batch results are
// bitwise-identical to the old for(b=0..B) dispatch loops.

// grid = (ceil(nmax/256), B, npairs) threadgroups of 256.
// is_copy=1: c[pair] = a[pair];  is_copy=0: a[pair] += b[pair] (same
// `a = a + b` shape as k_xfer). Strides are float offsets per token.
// sizeof == 72.
struct XferBatchArgs {
    unsigned int a[2], b[2], c[2], n[2];
    unsigned int a_stride[2], b_stride[2], c_stride[2];
    unsigned int B, npairs, is_copy, pad0;
};

// Batched RMS rows feeding k_mix_fused_batch. grid = (n+1, B) of 256.
// Rows r < n read states + b*states_stride + (n-1-r)*k (slot 0 = NEWEST
// block, matching do_mix's xr ordering); row n reads extra + b*extra_stride.
// Reduction is k_rms' verbatim (hasw=0 path) -> bitwise vs the per-token
// k_rms dispatches. sizeof == 40.
struct MixRmsBatchArgs {
    unsigned int states, states_stride;
    unsigned int extra, extra_stride;
    unsigned int stn, stn_stride;               // y = stn + b*stn_stride + r*k
    unsigned int n, B, k, pad0;
};

// Batched k_mix_fused. grid = (nq, B) of 256; qpack rows shared across b.
// sizeof == 48.
struct MixFusedBatchArgs {
    unsigned long qpack;                        // PACK q rows [j][M64_NE] fp32
    unsigned int stnoff, stn_stride, outoff, out_stride;
    unsigned int S, nq, B, pad0;
    float scale;                                // 1/sqrt(kNE)
};

// Batched k_ctrl_down (grid (64, nc, B) x 256) / k_ctrl_up (grid
// (ceil(ne/32), nc, B) x 32). in/msg/out hold TOKEN-0 offsets; per-token
// stride comes from *_stride. hh points at a B*4*64 SCR region (NOT the
// shared single-token o.ctrl_hh) so all B tokens run concurrently.
// gcat rows per token = 2*nc*kNE (rms pairs at +2j*kNE / +kNE+2j*kNE).
// sizeof == 192.
struct CtrlBatchSplitArgs {
    unsigned long wdown[4], wgain[4], wforg[4];
    unsigned int in[4], msg[4], out[4];
    unsigned int in_stride, msg_stride, out_stride;
    unsigned int gcat, gcat_stride;
    unsigned int hh, hh_stride;
    unsigned int grank, ne, nc, B;
    float eps;                                  // documented-only (split path)
};

// ---------------- Milestone 11: decode dispatch fusion ----------------
// k_qkv_prep_kv_write_{q8,f16}: one 64-threadgroup dispatch (grid=(64,1,1),
// tg=(32,1,1)) replaces clip(Q)+clip(K)+rms(V)+clip(Rr)+k_kv_write_{q8,f16}.
// wg 0..23: Q heads clip (qclip PACK scale or M64_NOFF, in-place SCR);
// wg 24..31: K heads clip -> direct KV write; wg 32..39: V heads per-head
// rmsnorm -> direct KV write (o.emb staging eliminated); wg 40..63: Rr heads
// clip (relclip PACK scale or M64_NOFF, in-place SCR). dst_k/dst_v keep the
// KVWriteF16Args HALF-element units (*2 for the Q8 byte base).
// sizeof == 64.
struct QkvPrepKvArgs {
    unsigned long qclip_off, relclip_off, dst_k, dst_v;   // PACK / half-units
    unsigned int qoff, koff, voff, roff;                  // SCR float offsets
    float qk_thresh, rel_thresh, eps;
    unsigned int pad0;
};

// k_postconv_residual: flat (n=kNE)x128 dispatch replaces
// k_conv_slot + k_xfer (2 dispatches). push=1 snapshots cur -> blkoff then
// cur = y (ip%6==0 block push); push=0 folds cur += y (thread-local c, so
// the read/write of the same element is race-free).
// sizeof == 40.
struct PostConvResArgs {
    unsigned long wconv;                        // PACK wconv [kNE][4] f32
    unsigned int rsoff, xoff, curoff, blkoff, n, push;
    float scale;
    unsigned int pad0;
};

