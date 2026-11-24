#!/usr/bin/env python3
"""make_test_fixture.py — generate the self-contained `make test` fixture.

Produces, with Python 3 stdlib ONLY (no numpy / xxhash / pip):
  <out_dir>/tiny.gguf     tiny synthetic GGUF v3 (f32/f16/bf16/q8_0/q8_1 + fused MoE)
  <out_dir>/tiny.m64pack  packed expert-aligned blob (format mirrors tools/m64pack.py v1)
  <out_dir>/golden.bin    bitwise dequant reference records (M64G stream) consumed
                          by engine/tests/test_dequant.cpp

The fixture exercises the exact pack-format invariants the engine asserts at load
time: 1 MiB header reserve, 16 KiB gpu/expert alignment, per-tensor xxh64 (pure-Python
XXH64 mirrored byte-for-byte from engine/src/xxh64.hpp), gate|up|down expert stride
layout, and bit-exact GGML dequant semantics (dequant.hpp q8_0/q8_1/f16/bf16/f32).

Usage:
  python3 tools/make_test_fixture.py <out_dir>
"""
import hashlib
import json
import os
import random
import struct
import sys

MAGIC = b"M64P"
FMT_VERSION = 1
HEADER_RESERVE = 1 << 20          # header reserve / blob_start (64 KiB aligned)
GPU_ALIGN = 1 << 14               # 16 KiB gpu/expert tensor alignment

# ggml dtype ids mirrored from tools/m64pack.py DTYPES
DT_F32, DT_F16, DT_Q8_0, DT_Q8_1, DT_BF16 = 0, 1, 8, 9, 30
DT_Q5_0, DT_Q5_1, DT_Q4_K = 6, 7, 12
DT_NAME = {DT_F32: "f32", DT_F16: "f16", DT_Q8_0: "q8_0", DT_Q8_1: "q8_1", DT_BF16: "bf16",
           DT_Q5_0: "q5_0", DT_Q5_1: "q5_1", DT_Q4_K: "q4_k"}


# ------------------------------------------------------------------ tokenizer
# Byte-level single-token vocabulary written into tiny.gguf (GPT-2 byte-space
# chars, see tools/bpe_gguf.bytes_to_unicode) so the stdlib BPE tool and the
# OpenAI-server integration tests run against the fixture without downloading
# any model. Includes a 3-byte UTF-8 char ('あ' = E3 81 82) split across three
# single-byte tokens to exercise incremental UTF-8 decoding.
def _b2u():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    c = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            c.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(x) for x in c)))


VOCAB_CHARS = "adefghiklnorstuv01279." + "STOP! \n"
TOKENS = [_b2u()[ord(c)] for c in VOCAB_CHARS]
TOKENS += [_b2u()[0xE3], _b2u()[0x81], _b2u()[0x82]]  # 'あ' split across 3 ids
TOKENS += ["<|eos|>"]
TOKEN_TYPES = [1] * (len(TOKENS) - 1) + [3]           # GGUF CONTROL on last id
EOS_TOKEN_ID = len(TOKENS) - 1
DT_BLOCK = {DT_F32: (1, 4), DT_F16: (1, 2), DT_Q8_0: (32, 34), DT_Q8_1: (32, 36), DT_BF16: (1, 2),
            DT_Q5_0: (32, 22), DT_Q5_1: (32, 24), DT_Q4_K: (256, 144)}


def f32(x):
    """Round a double to the nearest IEEE float32 (as the C++ side computes it)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16_bits(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def f16_val(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


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


# ------------------------------------------------------------------ quantizers
# Same scheme as tools/m64pack.py / ggml quantize_row_q8_{0,1}_ref:
# d = amax/127 -> f16, q = round(x/d) clamped to [-127, 127];
# q8_1 additionally stores s = f16(sum(q)*d); dequant x = d*q - s/32.
def q8_0_bytes(vals):
    out = bytearray()
    for i in range(0, len(vals), 32):
        blk = vals[i:i + 32]
        amax = max(abs(v) for v in blk)
        d = f16_val(f16_bits(amax / 127.0))
        q = [0 if not d else max(-127, min(127, int(round(v / d)))) for v in blk]
        out += struct.pack("<H", f16_bits(d)) + bytes((x & 0xFF) for x in q)
    return bytes(out)


def q8_1_bytes(vals):
    out = bytearray()
    for i in range(0, len(vals), 32):
        blk = vals[i:i + 32]
        amax = max(abs(v) for v in blk)
        d = f16_val(f16_bits(amax / 127.0))
        q = [0 if not d else max(-127, min(127, int(round(v / d)))) for v in blk]
        s = f16_val(f16_bits(sum(q) * d))
        out += struct.pack("<H", f16_bits(d)) + struct.pack("<H", f16_bits(s)) + bytes((x & 0xFF) for x in q)
    return bytes(out)


def f16_bytes(vals):
    return b"".join(struct.pack("<e", v) for v in vals)


def bf16_bytes(vals):
    out = bytearray()
    for v in vals:
        u = struct.unpack("<I", struct.pack("<f", v))[0]
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
        out += struct.pack("<H", u >> 16)
    return bytes(out)


# ------------------------------------------------------------------ dequant mirrors of dequant.hpp
def deq_row(dt, raw, nvals):
    """Bit-exact Python mirror of m64::dequant_row (float32 semantics)."""
    be, ts = DT_BLOCK[dt]
    out = []
    if dt == DT_F32:
        out = list(struct.unpack(f"<{nvals}f", raw[: nvals * 4]))
    elif dt == DT_F16:
        out = [f32(v) for v in struct.unpack(f"<{nvals}e", raw[: nvals * 2])]
    elif dt == DT_BF16:
        for i in range(nvals):
            (u,) = struct.unpack_from("<H", raw, 2 * i)
            out.append(struct.unpack("<f", struct.pack("<I", u << 16))[0])
    elif dt == DT_Q8_0:
        for b in range(len(raw) // 34):
            (dh,) = struct.unpack_from("<H", raw, b * 34)
            d = f16_val(dh)
            for j in range(32):
                q = raw[b * 34 + 2 + j]
                q = q - 256 if q >= 128 else q
                out.append(f32(d * q))
    elif dt == DT_Q8_1:
        for b in range(len(raw) // 36):
            (dh, sh) = struct.unpack_from("<HH", raw, b * 36)
            d = f16_val(dh)
            m = f32(f16_val(sh) / 32.0)
            for j in range(32):
                q = raw[b * 36 + 4 + j]
                q = q - 256 if q >= 128 else q
                out.append(f32(d * q - m))
    elif dt == DT_Q4_K:
        for b in range(len(raw) // 144):
            o = b * 144
            d = f16_val(struct.unpack_from("<H", raw, o)[0])
            mn = f16_val(struct.unpack_from("<H", raw, o + 2)[0])
            sc = raw[o + 4:o + 16]
            def gsm(j):  # get_scale_min_k4
                if j < 4:
                    return sc[j] & 63, sc[j + 4] & 63
                return ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
                        (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))
            qo = o + 16
            for jj in range(4):           # 4 x 64 elements
                sc1, m1i = gsm(2 * jj)
                sc2, m2i = gsm(2 * jj + 1)
                d1 = f32(d * sc1); m1 = f32(mn * m1i)
                d2 = f32(d * sc2); m2 = f32(mn * m2i)
                for l in range(32):
                    out.append(f32(f32(d1 * (raw[qo + 32 * jj + l] & 0xF)) - m1))
                for l in range(32):
                    out.append(f32(f32(d2 * (raw[qo + 32 * jj + l] >> 4)) - m2))
    elif dt == DT_Q5_0:
        for b in range(len(raw) // 22):
            o = b * 22
            d = f16_val(struct.unpack_from("<H", raw, o)[0])
            qh = int.from_bytes(raw[o + 2:o + 6], "little")
            lo, hi = [], []
            for j in range(16):
                xh0 = ((qh >> (j + 0)) << 4) & 0x10
                xh1 = (qh >> (j + 12)) & 0x10
                qs = raw[o + 6 + j]
                lo.append(((qs & 0xF) | xh0) - 16)
                hi.append(((qs >> 4) | xh1) - 16)
            for v in lo + hi:
                out.append(f32(v * d))
    elif dt == DT_Q5_1:
        for b in range(len(raw) // 24):
            o = b * 24
            d = f16_val(struct.unpack_from("<H", raw, o)[0])
            m = f16_val(struct.unpack_from("<H", raw, o + 2)[0])
            qh = int.from_bytes(raw[o + 4:o + 8], "little")
            lo, hi = [], []
            for j in range(16):
                xh0 = ((qh >> (j + 0)) << 4) & 0x10
                xh1 = (qh >> (j + 12)) & 0x10
                qs = raw[o + 8 + j]
                lo.append((qs & 0xF) | xh0)
                hi.append((qs >> 4) | xh1)
            for v in lo + hi:
                out.append(f32(f32(v * d) + m))
    else:
        raise ValueError(f"unsupported golden dtype {dt}")
    assert len(out) >= nvals
    return out[:nvals]


# ------------------------------------------------------------------ golden.bin
class GoldenWriter:
    """M64G stream: count u32; per rec: u16 name_len+name, u32 dtype, u32 nblocks,
    u64 raw_len + raw, u64 nvals + f32 dequant (record layout of test_dequant.cpp)."""

    def __init__(self):
        self.count = 0
        self.buf = bytearray(struct.pack("<4sI", b"M64G", 0))

    def emit(self, name, dt, raw, nvals):
        deq = deq_row(dt, raw, nvals)
        self.buf += struct.pack("<H", len(name)) + name.encode()
        self.buf += struct.pack("<II", dt, len(raw) // DT_BLOCK[dt][1])
        self.buf += struct.pack("<Q", len(raw)) + raw
        self.buf += struct.pack("<Q", nvals) + b"".join(struct.pack("<f", v) for v in deq)
        self.count += 1

    def write(self, path):
        self.buf[4:8] = struct.pack("<I", self.count)
        with open(path, "wb") as f:
            f.write(self.buf)


# ------------------------------------------------------------------ tensors
def build_tensors(rng):
    """Returns list of dicts: name, dims (ggml ne order), dt, vals (python floats, f32-rounded)."""
    def gauss(n):
        return [f32(rng.gauss(0.0, 1.0)) for _ in range(n)]

    T = []
    def add(name, dims, dt, vals=None):
        n = 1
        for d in dims:
            n *= d
        T.append({"name": name, "dims": dims, "dt": dt,
                  "vals": gauss(n) if vals is None else vals})

    add("token_embd.weight", [32, 16], DT_F32)
    add("token_embd_norm.weight", [32], DT_F32)
    add("blk.0.attn_norm.weight", [32], DT_F32)
    add("blk.0.ffn_norm.weight", [32], DT_F32)
    add("blk.0.attn_q.weight", [32, 64], DT_Q8_0)     # 2048 el = 64 blocks
    add("blk.0.attn_k.weight", [32, 32], DT_Q8_1)     # 32 blocks
    add("blk.0.attn_v.weight", [32, 32], DT_F16)
    add("blk.0.attn_o.weight", [32, 32], DT_BF16)
    # fused MoE: gate_up ne=[n_embd=32, 2*n_ff=8, E=2]; down ne=[n_embd=32, n_ff=4, E=2]
    add("blk.0.ffn_gate_up_exps.weight", [32, 8, 2], DT_Q8_1)   # 16 blocks, 8/expert
    add("blk.0.ffn_down_exps.weight", [32, 4, 2], DT_Q8_1)      # 8 blocks, 4/expert
    add("output.weight", [32, 16], DT_F32)
    return T


def tensor_bytes(t):
    vals, dt = t["vals"], t["dt"]
    if dt == DT_F32:
        return b"".join(struct.pack("<f", v) for v in vals)
    if dt == DT_F16:
        return f16_bytes(vals)
    if dt == DT_BF16:
        return bf16_bytes(vals)
    if dt == DT_Q8_0:
        return q8_0_bytes(vals)
    if dt == DT_Q8_1:
        return q8_1_bytes(vals)
    raise ValueError(dt)


# ------------------------------------------------------------------ tiny.gguf
def write_gguf(path, tensors):
    """GGUF v3, mirrors the layout tools/m64pack.py GgufInput reads."""
    def wstr(b, s):
        e = s.encode()
        return b + struct.pack("<Q", len(e)) + e

    def kv(b, k, t, payload):
        return wstr(b, k) + struct.pack("<I", t) + payload

    align = 32
    infos = bytearray()
    off = 0
    for t in tensors:
        raw = tensor_bytes(t)
        infos = bytearray(wstr(bytes(infos), t["name"]))
        infos += struct.pack("<I", len(t["dims"])) + struct.pack(f"<{len(t['dims'])}Q", *t["dims"])
        infos += struct.pack("<IQ", t["dt"], off)
        off += len(raw)
    kvs = b""
    kvs = kv(kvs, "general.architecture", 8, struct.pack("<Q", 13) + b"inkling2_loop")
    kvs = kv(kvs, "general.name", 8, struct.pack("<Q", 4) + b"tiny")
    kvs = kv(kvs, "general.block_count", 4, struct.pack("<I", 1))
    kvs = kv(kvs, "general.alignment", 4, struct.pack("<I", align))
    # Minimal tokenizer metadata so tools/bpe_gguf.py can load this fixture
    # (byte-level single-token vocab, no merges, one CONTROL eos token).
    toks_payload = struct.pack("<I", 8) + struct.pack("<Q", len(TOKENS))
    for _t in TOKENS:
        toks_payload = wstr(toks_payload, _t)
    kvs = kv(kvs, "tokenizer.ggml.tokens", 9, toks_payload)
    kvs = kv(kvs, "tokenizer.ggml.merges", 9, struct.pack("<I", 8) + struct.pack("<Q", 0))
    kvs = kv(kvs, "tokenizer.ggml.token_type", 9,
             struct.pack("<I", 4) + struct.pack("<Q", len(TOKEN_TYPES))
             + struct.pack("<%dI" % len(TOKEN_TYPES), *TOKEN_TYPES))
    kvs = kv(kvs, "tokenizer.ggml.eos_token_id", 4, struct.pack("<I", EOS_TOKEN_ID))
    kvs = kv(kvs, "tokenizer.ggml.eot_token_id", 4, struct.pack("<I", EOS_TOKEN_ID))
    nkv = 4 + 5
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), nkv) + kvs
    head = header + bytes(infos)
    align_start = (len(head) + align - 1) // align * align
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\x00" * (align_start - len(head)))
        for t in tensors:
            f.write(tensor_bytes(t))


# ------------------------------------------------------------------ tiny.m64pack
def align_up(x, a):
    return (x + a - 1) // a * a


def sort_key(name):
    if name == "token_embd.weight":
        return (0, name)
    if "token_embd_norm" in name:
        return (1, name)
    if name.startswith("blk."):
        return (int(name.split(".")[1]), name)
    return (1000, name)


def residency_of(name):
    if "norm" in name or "gate_inp" in name or "embd" in name or name == "output.weight":
        return "hot"
    return "warm"


def write_pack(path, tensors, gguf_path):
    raws = {t["name"]: tensor_bytes(t) for t in tensors}
    order = sorted(tensors, key=lambda t: sort_key(t["name"]))
    gu = raws["blk.0.ffn_gate_up_exps.weight"]
    dn = raws["blk.0.ffn_down_exps.weight"]
    E = 2
    gate_bytes = 4 * 32 // 32 * 36        # half_cols(4) * n_col(32) / 32 * 36
    down_bytes = 4 * 32 // 32 * 36        # dn dims[1] * dims[0] / 32 * 36
    per_exp = 2 * gate_bytes + down_bytes
    stride = align_up(per_exp, GPU_ALIGN)

    blobs = []                             # (name, bytes) in blob order
    records = []
    pos = HEADER_RESERVE
    blob_hash = Xxh64(0)

    def place(name, data, region, resident, extra=None):
        nonlocal pos
        pad = (GPU_ALIGN - pos % GPU_ALIGN) % GPU_ALIGN
        if pad:
            blobs.append(b"\x00" * pad)
            blob_hash.update(b"\x00" * pad)
            pos += pad
        h = Xxh64(0)
        h.update(data)
        blob_hash.update(data)
        rec = {"name": name, "dtype": DT_NAME[raw_dt[name]], "shape": shapes[name],
               "region": region, "layout": "rowmajor", "offset": pos, "length": len(data),
               "group_size": DT_BLOCK[raw_dt[name]][0], "residency": resident,
               "xxh64": f"{h.digest():016x}"}
        pos += len(data)
        if extra:
            rec.update(extra)
        records.append(rec)
        blobs.append(data)

    raw_dt = {t["name"]: t["dt"] for t in tensors}
    shapes = {t["name"]: list(reversed(t["dims"])) for t in tensors}

    for t in order:
        n = t["name"]
        if n == "blk.0.ffn_down_exps.weight":
            continue  # folded into the merged expert record
        if n == "blk.0.ffn_gate_up_exps.weight":
            lay_base = align_up(pos, GPU_ALIGN)
            if lay_base != pos:
                pad = lay_base - pos
                blobs.append(b"\x00" * pad)
                blob_hash.update(b"\x00" * pad)
                pos = lay_base
            h = Xxh64(0)
            for e in range(E):
                parts = [gu[e * 2 * gate_bytes: e * 2 * gate_bytes + gate_bytes],
                         gu[e * 2 * gate_bytes + gate_bytes: (e + 1) * 2 * gate_bytes],
                         dn[e * down_bytes: (e + 1) * down_bytes],
                         b"\x00" * (stride - per_exp)]
                data = b"".join(parts)
                h.update(data)
                blob_hash.update(data)
                blobs.append(data)
            pos = lay_base + E * stride
            records.append({"name": "blk.0.ffn_experts.weight", "dtype": "q8_1",
                            "shape": [E], "region": "expert", "layout": "expert_gateupdown",
                            "offset": lay_base, "length": E * stride, "group_size": 32,
                            "residency": "cold", "xxh64": f"{h.digest():016x}",
                            "expert_stride": stride,
                            "parts": {"gate": [4, 32], "up": [4, 32], "down": [4, 32]}})
            continue
        place(n, raws[n], "gpu", residency_of(n))

    blob_bytes = pos - HEADER_RESERVE
    total = sum(r["length"] for r in records)
    hot = sum(r["length"] for r in records if r["residency"] != "cold")
    sha = hashlib.sha256()
    with open(gguf_path, "rb") as f:
        for i, chunk in enumerate(iter(lambda: f.read(1 << 20), b"")):
            if i >= 32:
                break
            sha.update(chunk)
    header = {
        "magic": "M64P", "version": FMT_VERSION, "tool": "tools/make_test_fixture.py",
        "created": "2026-01-01T00:00:00Z",
        "source": {"kind": "GgufInput", "file": os.path.basename(gguf_path),
                   "sha256_partial": sha.hexdigest()[:32]},
        "model": {"arch": "inkling2_loop", "name": "tiny",
                  "hparams": {"general.architecture": "inkling2_loop", "general.block_count": 1,
                              "general.alignment": 32, "m64.n_embd": 32, "m64.n_ff": 4,
                              "m64.expert_count": E, "m64.top_k": 2}},
        "quant": {"profile_sha256": "", "tier_table": [
            {"pattern": "*", "dtype": "passthrough", "group": "source",
             "scale": "inline", "algo": "ggml_block_verbatim"}]},
        "residency": {"total_weights_bytes": total, "hot_weights_bytes": hot,
                      "cold_expert_bytes": total - hot, "max_gpu_bytes": 0},
        "alignment_log2": {"gpu": 14, "expert": 14, "cpu": 6},
        "blob_start": HEADER_RESERVE, "blob_bytes": blob_bytes,
        "tensors": records,
        "checksum": {"blob_xxh64": f"{blob_hash.digest():016x}"},
        "expert_index": {"stride_mode": "per_layer",
                         "layer_base": [r["offset"] for r in records if r["region"] == "expert"],
                         "expert_stride": stride, "order": "gate|up|down"},
    }
    hj = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if 16 + len(hj) > HEADER_RESERVE:
        raise RuntimeError("header JSON exceeds HEADER_RESERVE")
    with open(path, "wb") as f:
        f.write(b"\x00" * HEADER_RESERVE)
        for b in blobs:
            f.write(b)
        f.seek(0)
        f.write(MAGIC + struct.pack("<I", FMT_VERSION) + struct.pack("<Q", len(hj)) + hj)
    return header


# ------------------------------------------------------------------ golden from fixture
def _synth_blocks(rng, ts, f16_offs, nblocks):
    """Random block bytes with finite f16 fields (exponent masked < 0x7C00 so
    no NaN/Inf — the bitwise golden compare must stay deterministic)."""
    out = bytearray()
    for _ in range(nblocks):
        blk = bytearray(rng.getrandbits(8) for _ in range(ts))
        for off in f16_offs:
            bits = (rng.getrandbits(1) << 15) | (rng.randrange(0x1F) << 10) | rng.getrandbits(10)
            blk[off], blk[off + 1] = bits & 0xFF, (bits >> 8) & 0xFF
        out += blk
    return bytes(out)


def write_golden(path, tensors):
    g = GoldenWriter()
    raws = {t["name"]: tensor_bytes(t) for t in tensors}
    nvals = {t["name"]: len(t["vals"]) for t in tensors}
    dts = {t["name"]: t["dt"] for t in tensors}
    for name in ("token_embd.weight", "token_embd_norm.weight", "blk.0.attn_norm.weight",
                 "blk.0.ffn_norm.weight", "output.weight"):
        g.emit(name, DT_F32, raws[name], nvals[name])
    g.emit("blk.0.attn_v.weight", DT_F16, raws["blk.0.attn_v.weight"], nvals["blk.0.attn_v.weight"])
    g.emit("blk.0.attn_o.weight", DT_BF16, raws["blk.0.attn_o.weight"], nvals["blk.0.attn_o.weight"])
    g.emit("blk.0.attn_q.weight", DT_Q8_0, raws["blk.0.attn_q.weight"], nvals["blk.0.attn_q.weight"])
    g.emit("blk.0.attn_k.weight", DT_Q8_1, raws["blk.0.attn_k.weight"], nvals["blk.0.attn_k.weight"])
    gate_bytes = 4 * 32 // 32 * 36
    gu = raws["blk.0.ffn_gate_up_exps.weight"]
    for e in range(2):
        g.emit(f"blk.0.ffn_gate_up_exps.weight#{e}", DT_Q8_1,
               gu[e * 2 * gate_bytes: e * 2 * gate_bytes + gate_bytes], 4 * 32)
    # Synthetic raw-block records for the Milestone 4A dtypes: the raw bytes need
    # not come from real tensors — test_dequant re-dequantizes them with the C++
    # reference and compares bitwise. This covers q4_k/q5_0/q5_1 dispatch under
    # `make test` without changing the m64run-facing tiny.m64pack layout at all.
    rng = random.Random(4604)
    g.emit("synth.q4_k", DT_Q4_K, _synth_blocks(rng, 144, (0, 2), 8), 8 * 256)
    g.emit("synth.q5_0", DT_Q5_0, _synth_blocks(rng, 22, (0,), 8), 8 * 32)
    g.emit("synth.q5_1", DT_Q5_1, _synth_blocks(rng, 24, (0, 2), 8), 8 * 32)
    g.write(path)
    return g.count


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate the tiny make-test fixture "
                                             "(tiny.gguf, tiny.m64pack, golden.bin). stdlib only.")
    ap.add_argument("out_dir", help="directory to write the fixture into (created if missing)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(123)
    tensors = build_tensors(rng)
    gguf = os.path.join(a.out_dir, "tiny.gguf")
    pack = os.path.join(a.out_dir, "tiny.m64pack")
    golden = os.path.join(a.out_dir, "golden.bin")
    write_gguf(gguf, tensors)
    hdr = write_pack(pack, tensors, gguf)
    ngold = write_golden(golden, tensors)
    print(f"wrote {gguf} ({os.path.getsize(gguf)} B), {pack} ({os.path.getsize(pack)} B, "
          f"{len(hdr['tensors'])} tensors, expert_stride={hdr['expert_index']['expert_stride']}), "
          f"{golden} ({ngold} records)")


if __name__ == "__main__":
    main()
