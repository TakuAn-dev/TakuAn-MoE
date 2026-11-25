#!/usr/bin/env python3
"""make_gemma4_fixture.py — tiny Gemma 4 (.m64pack v1) fixture generator.

Stdlib-only (no numpy, no xxhash): ships its own XXH64 (seed 0) and float16
helpers, mirroring tools/make_test_fixture.py; cross-validated against the C++
engine dequant/xxh64 via the test_gemma4_cpu / PackReader hash checks.

Emits a self-describing pack with arch = "gemma4" whose hparams/tensor names
follow real Gemma 4 26B-A4B GGUF conventions (per-layer sliding-window pattern,
per-layer head_count_kv, swa/full rope parameter split, tied embeddings with NO
output.weight, fused expert records with per-part dtypes), at tiny sizes so the
CPU oracle test runs instantly:

    n_layer=2 (layer0 SWA hd=64 kv=2, layer1 Full hd=128 kv=1 tied-V)
    n_embd=256  n_ff=128  n_ff_exp=64  n_expert=8  top_k=2
    q8_0 attn/dense MLP, q4_k gate|up + q5_0/q8_0 down experts,
    f32 norms/scales/router.

Usage: python3 make_gemma4_fixture.py <out_dir>
"""
import json
import os
import random
import struct
import sys

# ------------------------------------------------------------------ XXH64
# Pure-Python mirror of engine/src/xxh64.hpp (XXH64, seed 0, xxhash >= 0.8).
M64 = (1 << 64) - 1
P1, P2, P3, P4, P5 = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                      0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5)


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & M64


def _round_acc(acc, inp):
    return _rotl((acc + inp * P2) & M64, 31) * P1 & M64


def _merge_acc(acc, val):
    return (((acc ^ _round_acc(0, val)) * P1) + P4) & M64


class Xxh64:
    def __init__(self, seed=0):
        self.a1 = (seed + P1 + P2) & M64
        self.a2 = (seed + P2) & M64
        self.a3 = seed & M64
        self.a4 = (seed - P1) & M64
        self.total = 0
        self.seed = seed
        self.buf = b""

    def update(self, data):
        self.total += len(data)
        data = self.buf + data
        nblocks = len(data) // 32
        for b in range(nblocks):
            p = data[32 * b:32 * b + 32]
            self.a1 = _round_acc(self.a1, int.from_bytes(p[0:8], "little"))
            self.a2 = _round_acc(self.a2, int.from_bytes(p[8:16], "little"))
            self.a3 = _round_acc(self.a3, int.from_bytes(p[16:24], "little"))
            self.a4 = _round_acc(self.a4, int.from_bytes(p[24:32], "little"))
        self.buf = data[nblocks * 32:]

    def digest(self):
        if self.total >= 32:
            h = (_rotl(self.a1, 1) + _rotl(self.a2, 7) + _rotl(self.a3, 12) + _rotl(self.a4, 18)) & M64
            h = _merge_acc(h, self.a1)
            h = _merge_acc(h, self.a2)
            h = _merge_acc(h, self.a3)
            h = _merge_acc(h, self.a4)
        else:
            h = (self.seed + P5) & M64
        h = (h + self.total) & M64
        p, l = self.buf, len(self.buf)
        i = 0
        while l - i >= 8:
            h ^= _round_acc(0, int.from_bytes(p[i:i + 8], "little"))
            h = (_rotl(h, 27) * P1 + P4) & M64
            i += 8
        if l - i >= 4:
            h ^= (int.from_bytes(p[i:i + 4], "little") * P1) & M64
            h = (_rotl(h, 23) * P2 + P3) & M64
            i += 4
        while i < l:
            h ^= (p[i] * P5) & M64
            h = _rotl(h, 11) * P1 & M64
            i += 1
        h ^= h >> 33
        h = h * P2 & M64
        h ^= h >> 29
        h = h * P3 & M64
        h ^= h >> 32
        return h


def xxh64_hex(data):
    h = Xxh64(0)
    h.update(data)
    return f"{h.digest():016x}"


def xxh64(data: bytes, seed: int = 0) -> int:
    h = Xxh64(seed)
    h.update(data)
    return h.digest()


def f32(vals):
    return struct.pack("<%df" % len(vals), *vals)


def f16_bits(x):
    """round-to-nearest-even float32 -> IEEE half bits."""
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f16_round(x):
    """float32 -> value of the nearest float16 (saturating like GGML on overflow)."""
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    sign = b >> 31
    e = (b >> 23) & 0xFF
    frac = b & 0x7FFFFF
    if e == 0xFF:
        return float("-inf") if sign else float("inf")
    exp = e - 127 + 15
    if exp <= 0:
        if exp < -10:
            return 0.0
        f9 = 0x800000 | frac
        shift = -exp + 14
        q = f9 >> shift
        rem = f9 & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
        bits = (sign << 15) | q
        val = struct.unpack("<e", struct.pack("<H", bits & 0x7FFF))[0]
        return -val if sign else val
    if exp >= 31:  # saturate to max finite f16 (65504), not inf
        return -65504.0 if sign else 65504.0
    h = (exp << 10) | (frac >> 13)
    rem = frac & 0x1FFF
    if rem > 0x1000 or (rem == 0x1000 and (h & 1)):
        h += 1
        if h >= 0x7C00:
            return -65504.0 if sign else 65504.0
    bits = (sign << 15) | h
    val = struct.unpack("<e", struct.pack("<H", bits & 0x7FFF))[0]
    return -val if sign else val


# ------------------------------------------------------------- block packers
def q8_0(vals):  # mirrors ggml quantize_row_q8_0_ref value semantics
    out = bytearray()
    for b in range(len(vals) // 32):
        blk = vals[b * 32:(b + 1) * 32]
        amax = max(max(abs(v) for v in blk), 1e-30)
        d = max(f16_round(amax / 127.0), 1e-30)
        ids = 1.0 / d
        qs = bytearray()
        for v in blk:
            q = int(v * ids + (0.5 if v >= 0 else -0.5))
            qs.append(max(-127, min(127, q)) & 0xFF)
        out += struct.pack("<H", f16_bits(d))  # ggml BlockQ8_0: d first
        out += qs
    return bytes(out)


def q5_0(vals):  # mirrors ggml quantize_row_q5_0_ref: d = amax/31, q = round(x/d)+16
    out = bytearray()
    for b in range(len(vals) // 32):
        blk = vals[b * 32:(b + 1) * 32]
        amax = max(max(abs(v) for v in blk), 1e-30)
        d = max(f16_round(amax / 31.0), 1e-30)
        ids = 1.0 / d
        q = []
        for v in blk:
            k = int(v * ids + (0.5 if v >= 0 else -0.5)) + 16
            q.append(max(0, min(31, k)))
        qs = bytearray(16)
        qh = 0
        for j in range(16):
            lo, hi = q[j], q[j + 16]
            qs[j] = (lo & 0xF) | ((hi & 0xF) << 4)
            qh |= ((lo >> 4) & 1) << j
            qh |= ((hi >> 4) & 1) << (j + 12)
        # ggml BlockQ5_0: d(2) + qh(4) + qs(16)
        out += struct.pack("<H", f16_bits(d))
        out += struct.pack("<I", qh & 0xFFFFFFFF)
        out += qs
    return bytes(out)


def q4_k(vals):
    """Valid q4_K blocks (256 elts: 128B qs + 12B scales + 2x f16 d/dmin).

    Not bit-identical to ggml's numpy optimiser (unavailable here); this picks
    format-valid scales/quantums, so the C++ reference dequant (dequant.hpp,
    golden-verified against GGML) is the definition of the stored value.
    Layout mirrors get_scale_min_k4 + dequantize_row_q4_K_ref: sub-block s
    covers elements [32s, 32s+32); byte 32jj+c low nibble -> element 64jj+c,
    high nibble -> element 64jj+32+c; all 8 scale/min values are 6-bit.
    """
    assert len(vals) % 256 == 0
    out = bytearray()
    for b in range(len(vals) // 256):
        blk = vals[b * 256:(b + 1) * 256]
        subs = [blk[32 * s:32 * s + 32] for s in range(8)]
        amax = max(1e-30, max(abs(v) for sb in subs for v in sb))
        d = max(f16_round(amax / 15.0), 1e-30)
        q = [0] * 256
        sc = [0] * 8
        mn = [0] * 8
        shift_t = [0.0] * 8
        for s in range(8):
            sb = subs[s]
            bmin = min(sb)
            span = max(sb) - bmin
            sc[s] = max(1, min(63, int((span / 15.0) / d + 0.5)))
            step = d * sc[s]
            shift_t[s] = -bmin  # target offset: y = step*q - dmin*mn, y(q=0) = bmin
            for i, v in enumerate(sb):
                k = int((v - bmin) / step + 0.5)
                q[32 * s + i] = max(0, min(15, k))
        dmin = 0.0
        for s in range(8):
            if shift_t[s] > 0:
                m = shift_t[s] / (d * sc[s])
                if m > 63:
                    m = 63.0
                dmin = max(dmin, m * d)
        dmin = min(f16_round(dmin), 63.0 * d)
        for s in range(8):
            if dmin > 0:
                mn[s] = max(0, min(63, int(shift_t[s] / dmin + 0.5)))
        scales = [0] * 12
        for s in range(4):
            # get_scale_min_k4: j<4 -> low 6 bits of sc[j]/mn[j] in scales[j]/scales[j+4];
            # j>=4 -> low nibble of sc[j+4]/mn[j+4] in scales[j+8] (lo/hi nibble),
            # high 2 bits stashed at bits 6-7 of scales[j]/scales[j+4].
            scales[s] = sc[s] | ((sc[s + 4] >> 4) << 6)
            scales[s + 4] = mn[s] | ((mn[s + 4] >> 4) << 6)
            scales[s + 8] = (sc[s + 4] & 0xF) | ((mn[s + 4] & 0xF) << 4)
        qs = bytearray(128)
        for jj in range(4):
            for c in range(32):
                qs[32 * jj + c] = q[64 * jj + c] | (q[64 * jj + 32 + c] << 4)
        # ggml BlockQ4_K: d(2) + dmin(2) + scales(12) + qs(128)
        out += struct.pack("<H", f16_bits(d))
        out += struct.pack("<H", f16_bits(dmin))
        out += bytes(scales)
        out += qs
    return bytes(out)


# --------------------------------------------------------------- pack writer
# M64P v1 layout (mirrors make_test_fixture.write_pack / PackReader::open):
#   [0, HEADER_RESERVE)          header: "M64P" + u32 version + u64 header_len + JSON
#   [HEADER_RESERVE, EOF)        blob (offsets in records are ABSOLUTE file offsets)
MAGIC = b"M64P"
FMT_VERSION = 1
HEADER_RESERVE = 1 << 20
GPU_ALIGN = 1 << 14
EXP_ALIGN = 1 << 14
GROUP_SIZE = {"f32": 1, "f16": 1, "bf16": 1, "q8_0": 32, "q8_1": 32, "q5_0": 32,
              "q5_1": 32, "q4_k": 256, "q6_k": 256, "iq3_xxs": 256}


class M64P:
    def __init__(self, path):
        self.path = path
        self.buf = bytearray()               # blob bytes, starting at blob_start
        self.pos = HEADER_RESERVE            # absolute file offset of next byte
        self.entries = []
        self.total_weights = self.hot_weights = self.cold_experts = 0

    def tensor(self, name, dt, rows, cols, blob, region="gpu", residency="warm",
               shape=None, **extra):
        align = GPU_ALIGN if region != "expert" else EXP_ALIGN
        pad = (align - self.pos % align) % align
        if pad:
            self.buf += b"\0" * pad
            self.pos += pad
        e = {"name": name, "dtype": dt, "shape": shape or [rows, cols],
             "region": region, "layout": "rowmajor", "offset": self.pos,
             "length": len(blob), "group_size": GROUP_SIZE[dt],
             "residency": residency, "xxh64": "%016x" % xxh64(blob)}
        e.update(extra)
        self.entries.append(e)
        self.buf += blob
        self.pos += len(blob)
        self.total_weights += len(blob)
        if residency == "cold":
            self.cold_experts += len(blob)
        elif residency == "hot":
            self.hot_weights += len(blob)

    def close(self, arch, hparams, expert_index=None):
        header = {
            "magic": "M64P", "version": FMT_VERSION, "tool": "tools/make_gemma4_fixture.py",
            "created": "2026-01-01T00:00:00Z",
            "source": {"kind": "synthetic", "file": "tiny_gemma4.gguf", "sha256_partial": ""},
            "model": {"arch": arch, "name": "tiny_gemma4", "hparams": hparams},
            "quant": {"profile_sha256": "", "tier_table": [
                {"pattern": "*", "dtype": "passthrough", "group": "source",
                 "scale": "inline", "algo": "ggml_block_verbatim"}]},
            "residency": {"total_weights_bytes": self.total_weights,
                          "hot_weights_bytes": self.hot_weights,
                          "cold_expert_bytes": self.cold_experts, "max_gpu_bytes": 0},
            "alignment_log2": {"gpu": 14, "expert": 14, "cpu": 6},
            "blob_start": HEADER_RESERVE, "blob_bytes": len(self.buf),
            "tensors": self.entries,
            "checksum": {"blob_xxh64": "%016x" % xxh64(bytes(self.buf))},
        }
        if expert_index:
            header["expert_index"] = expert_index
        hj = json.dumps(header, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True).encode()
        if 16 + len(hj) > HEADER_RESERVE:
            raise RuntimeError("header JSON exceeds HEADER_RESERVE")
        with open(self.path, "wb") as f:
            f.write(b"\0" * HEADER_RESERVE)
            f.write(self.buf)
            f.seek(0)
            f.write(MAGIC + struct.pack("<I", FMT_VERSION) + struct.pack("<Q", len(hj)) + hj)


# ------------------------------------------------------------ tiny Gemma 4
NE = 256          # embedding_length (mult of 256 -> q4_k gate/up rows clean)
NF = 128          # feed_forward_length (dense shared MLP)
NFE = 64          # expert_feed_forward_length (mult 32, NOT mult 256: q5_0/q8_0 down)
NEXP = 8
TOPK = 2
VL = 32           # vocab rows — NO output.weight: tied LM head fallback is tested
CTX = 64
N_LAYER = 2
SWA_PATTERN = [1, 0]   # layer 0 sliding-window (win 4), layer 1 full attention
KVH = [2, 1]           # head_count_kv per layer (layer 1 also ties V = K)
DOWN_DT = {0: "q5_0", 1: "q8_0"}  # exercise both down-part dtypes across layers


def vals(seed, n):
    r = random.Random(seed)
    return [round(r.gauss(0.0, 0.08), 4) for _ in range(n)]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "build/test_fixture"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tiny_gemma4.m64pack")
    p = M64P(path)

    hparams = {
        "general.architecture": "gemma4",
        "gemma4.block_count": N_LAYER,
        "gemma4.attention.sliding_window_pattern": SWA_PATTERN,  # ints, like real GGUF
        "gemma4.attention.sliding_window": 4,
        "gemma4.attention.key_length": 128,
        "gemma4.attention.value_length": 128,
        "gemma4.attention.key_length_swa": 64,
        "gemma4.attention.value_length_swa": 64,
        "gemma4.rope.dimension_count": 128,
        "gemma4.rope.freq_base": 1000000.0,
        "gemma4.rope.dimension_count_swa": 64,
        "gemma4.rope.freq_base_swa": 10000.0,
        "gemma4.context_length": CTX,
        "gemma4.embedding_length": NE,
        "gemma4.feed_forward_length": NF,
        "gemma4.expert_feed_forward_length": NFE,
        "gemma4.expert_count": NEXP,
        "gemma4.expert_used_count": TOPK,
        "gemma4.attention.head_count": 4,
        "gemma4.attention.head_count_kv": KVH,
        "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
        "gemma4.final_logit_softcapping": 30.0,
    }

    # Global tensors. NO output.weight -> tied LM head from token_embd.weight.
    p.tensor("token_embd.weight", "q8_0", VL, NE, q8_0(vals(900, VL * NE)), residency="hot")
    p.tensor("output_norm.weight", "f32", NE, 1, f32(vals(901, NE)), residency="hot")
    # rope_freqs: first half exact 1.0 (normal angles), second half 1e30 (angles
    # vanish) — matches the sentinel pattern real packs store for non-rotated dims.
    p.tensor("rope_freqs.weight", "f32", 64, 1, f32([1.0] * 32 + [1e30] * 32), residency="hot")

    seed = 1000
    expert_bases = []
    expert_stride = 0
    for il in range(N_LAYER):
        hd = 64 if SWA_PATTERN[il] else 128
        nkv = KVH[il]
        p.tensor(f"blk.{il}.attn_q.weight", "q8_0", 4 * hd, NE, q8_0(vals(seed, 4 * hd * NE)), residency="warm"); seed += 1
        p.tensor(f"blk.{il}.attn_k.weight", "q8_0", nkv * hd, NE, q8_0(vals(seed, nkv * hd * NE)), residency="warm"); seed += 1
        if il == 0:  # layer 1 ties V = K -> emit NO attn_v there
            p.tensor(f"blk.{il}.attn_v.weight", "q8_0", nkv * hd, NE, q8_0(vals(seed, nkv * hd * NE)), residency="warm"); seed += 1
        p.tensor(f"blk.{il}.attn_output.weight", "q8_0", NE, 4 * hd, q8_0(vals(seed, NE * 4 * hd)), residency="warm"); seed += 1
        p.tensor(f"blk.{il}.attn_q_norm.weight", "f32", hd, 1, f32(vals(seed, hd)), residency="hot"); seed += 1
        p.tensor(f"blk.{il}.attn_k_norm.weight", "f32", hd, 1, f32(vals(seed, hd)), residency="hot"); seed += 1
        for nm in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm_1",
                   "pre_ffw_norm_2", "post_ffw_norm_2", "post_ffw_norm"):
            p.tensor(f"blk.{il}.{nm}.weight", "f32", NE, 1, f32(vals(seed, NE)), residency="hot"); seed += 1
        # layer_output_scale != 1.0 so the scaling op is actually exercised
        p.tensor(f"blk.{il}.layer_output_scale.weight", "f32", 1, 1,
                 f32([0.85 if il == 0 else 0.9]), residency="hot")
        # dense shared MLP
        p.tensor(f"blk.{il}.ffn_gate.weight", "q8_0", NF, NE, q8_0(vals(seed, NF * NE)), residency="warm"); seed += 1
        p.tensor(f"blk.{il}.ffn_up.weight", "q8_0", NF, NE, q8_0(vals(seed, NF * NE)), residency="warm"); seed += 1
        p.tensor(f"blk.{il}.ffn_down.weight", "q8_0", NE, NF, q8_0(vals(seed, NE * NF)), residency="warm"); seed += 1
        # MoE router: f32 weight + f32 scale, like real Gemma 4 26B-A4B
        p.tensor(f"blk.{il}.ffn_gate_inp.weight", "f32", NEXP, NE, f32(vals(seed, NEXP * NE)), residency="hot"); seed += 1
        p.tensor(f"blk.{il}.ffn_gate_inp.scale", "f32", NE, 1, f32(vals(seed, NE)), residency="hot"); seed += 1
        p.tensor(f"blk.{il}.ffn_down_exps.scale", "f32", NEXP, 1, f32(vals(seed, NEXP)), residency="hot"); seed += 1
        # fused expert record: gate|up q4_k; down q5_0 (il 0) / q8_0 (il 1)
        gate_by = NFE * NE // 256 * 144
        down_by = NE * NFE // 32 * (22 if DOWN_DT[il] == "q5_0" else 34)
        stride = (gate_by + gate_by + down_by + 16383) // 16384 * 16384
        rec = bytearray()
        for e in range(NEXP):
            one = bytearray()
            one += q4_k(vals(seed + 100 * e, NFE * NE))
            one += q4_k(vals(seed + 100000 + 100 * e, NFE * NE))
            dv = vals(seed + 200000 + 100 * e, NE * NFE)
            one += q5_0(dv) if DOWN_DT[il] == "q5_0" else q8_0(dv)
            one += b"\0" * (stride - len(one))
            rec += one
        seed += 3000
        p.tensor(f"blk.{il}.ffn_experts.weight", "q4_k", NEXP, 1, bytes(rec),
                 region="expert", residency="cold", shape=[NEXP],
                 layout="expert_gateupdown",
                 expert_stride=stride,
                 parts={"gate": [NFE, NE], "up": [NFE, NE], "down": [NE, NFE]},
                 part_dtypes={"gate": "q4_k", "up": "q4_k", "down": DOWN_DT[il]},
                 part_bytes={"gate": gate_by, "up": gate_by, "down": down_by})
        expert_bases.append(p.entries[-1]["offset"])
        expert_stride = stride

    p.close("gemma4", hparams, expert_index={
        "stride_mode": "per_layer", "layer_base": expert_bases,
        "expert_stride": expert_stride, "order": "gate|up|down"})
    print("wrote", path, os.path.getsize(path), "bytes,", len(p.entries), "tensors")


if __name__ == "__main__":
    main()