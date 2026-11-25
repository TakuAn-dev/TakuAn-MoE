#!/usr/bin/env python3
"""m64pack v1 converter: GGUF (single/multi-shard) or safetensors -> .m64pack.

Implements docs/DESIGN.md §2: 64 KiB blob start, 16 KiB-aligned gpu/expert tensors,
canonical JSON header, expert-contiguous gate|up|down records, xxh64 per-tensor +
blob checksums, --verify, --dump-golden, F_NOCACHE I/O. Quantized tensors are
GGML block bytes copied verbatim (no requant); MoE experts are repacked so each
(layer, expert) is exactly one contiguous aligned range.

Usage:
  m64pack.py pack   IN.gguf OUT.m64pack [--verify] [--dump-golden DIR]
  m64pack.py verify PACK.m64pack
  m64pack.py info   PACK.m64pack
"""
import argparse, fcntl, hashlib, json, math, os, re, struct, sys, time

try:
    import xxhash
except ImportError:
    sys.exit("pip install xxhash into this interpreter (see tools/README)")

MAGIC = b"M64P"
FMT_VERSION = 1
BLOB_START = 64 * 1024
# The canonical header JSON is written at file start and can exceed 64 KiB
# (Inkling's 680-tensor header is ~150 KiB). Reserve 1 MiB so the final
# seek(0) header write can never stomp blob data. Hard guard below.
HEADER_RESERVE = 1 << 20
GPU_ALIGN = 1 << 14
CPU_ALIGN = 1 << 6

DTYPES = {  # ggml enum (llama.cpp current: q8_0=8, q8_1=9, bf16=30) -> (name, block_elems, type_size)
    0: ("f32", 1, 4), 1: ("f16", 1, 2),
    2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36),
    10: ("q2_k", 256, 84), 11: ("q3_k", 256, 110), 12: ("q4_k", 256, 144),
    13: ("q5_k", 256, 176), 14: ("q6_k", 256, 210), 15: ("q8_k", 256, 144),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136), 24: ("i8", 32, 32),
    25: ("i16", 32, 64), 26: ("i32", 32, 128), 27: ("i64", 32, 256),
    28: ("f64", 1, 8), 29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2),
}
DTYPE_ID = {v[0]: k for k, v in DTYPES.items()}
GGUF_T = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6,
          "bool": 7, "str": 8, "arr": 9, "u64": 10, "i64": 11, "f64": 12}


class Src:
    """Byte-range reader over one or more GGUF shards / safetensors files, with F_NOCACHE."""

    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        for f in self.files:
            try:
                fcntl.fcntl(f, fcntl.F_NOCACHE, 1)
            except OSError:
                pass
        self.cum = []
        acc = 0
        for f in self.files:
            self.cum.append(acc)
            acc += os.fstat(f.fileno()).st_size
        self.total = acc

    def read(self, off, n, partial_ok=False):
        for i in range(len(self.files) - 1, -1, -1):
            if off >= self.cum[i]:
                self.files[i].seek(off - self.cum[i])
                buf = self.files[i].read(n)
                if len(buf) == n or i == len(self.files) - 1 or partial_ok:
                    return buf
                return buf + self.files[i + 1].read(n - len(buf))
        raise EOFError("read out of range")

    def sizes(self):
        return [s - c for c, s in zip(self.cum, self.cum[1:])] + [self.total - self.cum[-1]]


def align_up(v, a):
    return (v + a - 1) // a * a


# ---------------------------------------------------------------- GGUF input
class GgufInput:
    SKIP_KV = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.scores",
               "tokenizer.ggml.token_type", "tokenizer.ggml.added_tokens",
               "tokenizer.ggml.add_bos", "tokenizer.ggml.eos_token_id")

    def __init__(self, paths):
        self.src = Src(paths)
        self.path = paths[0]
        magic = self.src.read(0, 4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")
        # metadata from shard 0
        r0 = _ByteR(self.src.read(0, 64 << 20, partial_ok=True))
        r0.raw(4)  # skip magic
        if r0.u32() != 3:
            raise ValueError("only GGUF v3 supported")
        nt0 = r0.u64()
        nkv = r0.u64()
        self.md = {}
        for _ in range(nkv):
            k = r0.string()
            self.md[k] = _read_kv(r0)
        self.alignment = self.md.get("general.alignment", 32)
        self.tensors = []
        for i, f in enumerate(self.src.files):
            cap = 8 << 20
            while True:
                hdr = _ByteR(self.src.read(self.src.cum[i], cap, partial_ok=True))
                try:
                    parsed = self._parse_shard_header(hdr, i)
                    break
                except EOFError:
                    if cap >= (1 << 28):  # 256 MiB header — beyond any sane GGUF
                        raise
                    cap <<= 1
            base, infos_here = parsed
            for name, dims, dt, off in infos_here:
                self.tensors.append({"name": name, "dims": dims, "dtype": dt,
                                     "offset": off, "_shard": i, "_base": base})
        if len(self.tensors) != nt0 and len(paths) == 1:
            raise ValueError("tensor count mismatch")

    def _parse_shard_header(self, hdr, shard_i):
        hdr.raw(4)  # magic
        if hdr.u32() != 3:
            raise ValueError("only GGUF v3 supported")
        nt = hdr.u64()
        kn = hdr.u64()
        for _ in range(kn):  # skip KVs
            hdr.string()
            _skip_kv(hdr)
        infos_here = []
        for _ in range(nt):
            name = hdr.string()
            dims = [hdr.u64() for _ in range(hdr.u32())]
            dt = hdr.u32()
            off = hdr.u64()
            infos_here.append((name, dims, dt, off))
        base = align_up(hdr.tell(), self.alignment)  # data section start for this shard
        return base, infos_here

    def read_tensor(self, t, start=0, n=None):
        nbytes = tensor_nbytes(t["dims"], t["dtype"])
        n = nbytes if n is None else min(n, nbytes - start)
        return self.src.read(self.src.cum[t["_shard"]] + t["_base"] + t["offset"] + start, n)


def _skip_kv(r):
    t = r.u32()
    import math as _m
    stack = [(t, 1)]
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    while stack:
        tt, cnt = stack.pop()
        if tt == 8:  # string
            for _ in range(cnt):
                n = r.u64()
                r.p += n
        elif tt == 9:  # array
            at = r.u32()
            n = r.u64()
            if n > (1 << 22):
                r.p = len(r.b)
                return
            stack.append((at, n * cnt))
        else:
            r.p += cnt * sizes[tt]


class _ByteR:
    def __init__(self, buf, off=0):
        self.b = buf; self.p = off

    def raw(self, n):
        if self.p + n > len(self.b):
            raise EOFError
        v = self.b[self.p:self.p + n]; self.p += n
        return v

    def tell(self):
        return self.p

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        n = self.u64()
        return self.raw(n).decode("utf-8")


def _read_kv(r):
    t = r.u32()
    if t == GGUF_T["bool"]:
        return r.u8() != 0 if hasattr(r, "u8") else struct.unpack("<B", r.raw(1))[0] != 0
    if t == GGUF_T["u8"]:
        return struct.unpack("<B", r.raw(1))[0]
    if t == GGUF_T["i8"]:
        return struct.unpack("<b", r.raw(1))[0]
    if t == GGUF_T["u16"]:
        return struct.unpack("<H", r.raw(2))[0]
    if t == GGUF_T["i16"]:
        return struct.unpack("<h", r.raw(2))[0]
    if t == GGUF_T["u32"]:
        return struct.unpack("<I", r.raw(4))[0]
    if t == GGUF_T["i32"]:
        return struct.unpack("<i", r.raw(4))[0]
    if t == GGUF_T["f32"]:
        return struct.unpack("<f", r.raw(4))[0]
    if t == GGUF_T["f64"]:
        return struct.unpack("<d", r.raw(8))[0]
    if t == GGUF_T["str"]:
        return r.string()
    if t == GGUF_T["u64"]:
        return r.u64()
    if t == GGUF_T["i64"]:
        return r.i64() if hasattr(r, "i64") else struct.unpack("<q", r.raw(8))[0]
    if t == GGUF_T["arr"]:
        at = r.u32()
        n = r.u64()
        if n > (1 << 22):  # huge tokenizer arrays: skip contents, record marker
            r.p = len(r.b)
            return f"<skipped-array x{n}>"
        if at == GGUF_T["arr"]:
            return [_read_kv(r) for _ in range(n)]
        out = []
        for _ in range(n):
            # element values follow with the array's element type (not per-element type tags)
            if at == GGUF_T["str"]:
                out.append(r.string())
            elif at == GGUF_T["u8"] or at == GGUF_T["bool"]:
                out.append(struct.unpack("<B", r.raw(1))[0])
            elif at == GGUF_T["i8"]:
                out.append(struct.unpack("<b", r.raw(1))[0])
            elif at == GGUF_T["u16"]:
                out.append(struct.unpack("<H", r.raw(2))[0])
            elif at == GGUF_T["i16"]:
                out.append(struct.unpack("<h", r.raw(2))[0])
            elif at == GGUF_T["u32"]:
                out.append(struct.unpack("<I", r.raw(4))[0])
            elif at == GGUF_T["i32"]:
                out.append(struct.unpack("<i", r.raw(4))[0])
            elif at == GGUF_T["f32"]:
                out.append(struct.unpack("<f", r.raw(4))[0])
            elif at == GGUF_T["f64"]:
                out.append(struct.unpack("<d", r.raw(8))[0])
            elif at == GGUF_T["u64"]:
                out.append(r.u64())
            elif at == GGUF_T["i64"]:
                out.append(struct.unpack("<q", r.raw(8))[0])
            else:
                raise ValueError(f"gguf array elem type {at}")
        return out
    raise ValueError(f"gguf type {t}")


def tensor_nbytes(dims, dt):
    name, bs, ts = DTYPES[dt]
    ne = 1
    for d in dims:
        ne *= d
    return (ne // bs) * ts


# ---------------------------------------------------------------- safetensors input
class SfInput:
    DT = {"F32": 0, "F16": 1, "BF16": 2, "F64": None, "I8": 28, "I32": None, "I64": None, "U8": None}

    def __init__(self, paths):
        import glob
        if len(paths) == 1 and os.path.isdir(paths[0]):
            paths = sorted(glob.glob(os.path.join(paths[0], "*.safetensors")))
        self.src = Src(paths)
        self.path = paths[0]
        self.base0 = {}
        self.map = {}
        self.tensors = []
        for i, p in enumerate(paths):
            n = struct.unpack("<Q", self.src.read(self.src.cum[i], 8))[0]
            js = json.loads(self.src.read(self.src.cum[i] + 8, n))
            base = align_up(8 + n, 8)
            self.base0[i] = self.src.cum[i] + base
            for name, m in js.items():
                if name == "__metadata__":
                    continue
                if m["dtype"] not in self.DT or self.DT[m["dtype"]] is None:
                    raise ValueError(f"{name}: dtype {m['dtype']} unsupported (needs conversion first)")
                self.tensors.append({"name": name, "dims": m["shape"], "dtype": self.DT[m["dtype"]],
                                     "offset": m["data_offsets"][0], "_shard": i})
                self.map[name] = (i, m["data_offsets"][0], m["data_offsets"][1])
        self.md = {"general.architecture": "safetensors"}
        self.base = None

    def read_tensor(self, t, start=0, n=None):
        i, a, b = self.map[t["name"]]
        a = self.base0[i] + a + start
        b = self.base0[i] + min(b, a - self.base0[i] - start + (n if n is not None else math.inf))
        return self.src.read(a, int(b - a))


# ---------------------------------------------------------------- dequant reference (golden)
import numpy as np

def dequant_blocks(dt, raw):
    """Bit-exact ggml dequant for f32/f16/bf16/q8_0/q8_1/q4_k/q5_0/q5_1 blocks.

    All arithmetic stays in float32 with one IEEE op per C++ expression
    (engine/src/dequant.hpp is compiled with -ffp-contract=off), so results
    are bitwise identical to the C++ reference for the same block bytes."""
    name, bs, ts = DTYPES[dt]
    if name == "f32":
        return np.frombuffer(raw, dtype="<f4")
    if name == "f16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if name == "bf16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32)
    nb = len(raw) // ts
    if name == "q8_0":
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, ts)
        d = blk[:, 0:2].copy().view(np.float16).astype(np.float32)
        q = blk[:, 2:].astype(np.int8).astype(np.float32)
        return (d * q).reshape(-1)
    if name == "q8_1":
        # ggml: block = {f16 d; f16 s = d*sum(qs); i8 qs[32]}; x = d*(qs - s/(32d))
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, ts)
        d = blk[:, 0:2].copy().view(np.float16).astype(np.float32)
        s = blk[:, 2:4].copy().view(np.float16).astype(np.float32)
        q = blk[:, 4:].astype(np.int8).astype(np.float32)
        return (q * d - s * (1.0 / 32.0)).reshape(-1)
    if name == "q4_k":
        # block_q4_K (144 B / 256 el): {half d; half dmin; u8 scales[12]; u8 qs[128]}
        # 8 sub-blocks of 32; get_scale_min_k4 unpacks the 6-bit scale/min pairs.
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 144)
        d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        dmin = blk[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        sc = blk[:, 4:16]
        sci = sc.astype(np.int16)
        s32 = (sci[:, :4] & 63).astype(np.float32)
        m32 = (sci[:, 4:8] & 63).astype(np.float32)
        s47 = ((sci[:, 8:12] & 15) + (sci[:, 0:4] >> 6) * 16).astype(np.float32)
        m47 = ((sci[:, 8:12] >> 4) + (sci[:, 4:8] >> 6) * 16).astype(np.float32)
        dsub = np.hstack([s32, s47])   # (nb,8) per-sub-block scale multipliers
        msub = np.hstack([m32, m47])
        qs = blk[:, 16:144].astype(np.uint16)  # (nb,128) bytes; 4 chunks of 32
        nib = np.empty((nb, 4, 2, 32), dtype=np.float32)
        nib[:, :, 0, :] = (qs[:, 0:128:1] & 15).reshape(nb, 4, 32).astype(np.float32)   # low nibbles -> elems 64c+0..31
        nib[:, :, 1, :] = (qs >> 4).reshape(nb, 4, 32).astype(np.float32)                # high nibbles -> elems 64c+32..63
        d1 = (d * dsub).reshape(nb, 4, 2, 1)   # float32: d*sc per sub-block
        m1 = (dmin * msub).reshape(nb, 4, 2, 1)
        out = (d1 * nib - m1).reshape(nb, 256)
        return out.reshape(-1)
    if name == "q5_0":
        # block_q5_0 (22 B / 32 el): {half d; u8 qh[4]; u8 qs[16]}
        # elem j:    ((qs[j]&0xF) | (((qh>>j)<<4)&0x10)) - 16, times d
        # elem j+16: ((qs[j]>>4)  | ((qh>>(j+12))&0x10)) - 16, times d
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 22)
        d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        qh = blk[:, 2:6].copy().view(np.uint32)[:, 0]
        qs = blk[:, 6:22]
        j = np.arange(16, dtype=np.uint32)[None, :]
        lo = ((qs & 0xF).astype(np.uint32) | (((qh[:, None] >> j) << 4) & 0x10)).astype(np.int32) - 16
        hi = ((qs >> 4).astype(np.uint32) | ((qh[:, None] >> (j + 12)) & 0x10)).astype(np.int32) - 16
        v = np.empty((nb, 32), dtype=np.float32)
        v[:, :16] = lo.astype(np.float32)
        v[:, 16:] = hi.astype(np.float32)
        return (v * d).reshape(-1)
    if name == "q5_1":
        # block_q5_1 (24 B / 32 el): {half d; half m; u8 qh[4]; u8 qs[16]}; y = q*d + m
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 24)
        d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        m = blk[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        qh = blk[:, 4:8].copy().view(np.uint32)[:, 0]
        qs = blk[:, 8:24]
        j = np.arange(16, dtype=np.uint32)[None, :]
        lo = ((qs & 0xF).astype(np.uint32) | (((qh[:, None] >> j) << 4) & 0x10)).astype(np.uint32)
        hi = ((qs >> 4).astype(np.uint32) | ((qh[:, None] >> (j + 12)) & 0x10)).astype(np.uint32)
        v = np.empty((nb, 32), dtype=np.float32)
        v[:, :16] = lo.astype(np.float32)
        v[:, 16:] = hi.astype(np.float32)
        return (v * d + m).reshape(-1)
    raise ValueError(f"golden dequant unsupported dtype {name}")


# ---------------------------------------------------------------- packer
class Packer:
    def __init__(self, src, out_path):
        self.src = src
        self.out = out_path
        self.src_name = os.path.basename(src.path)

    def _expert_key(self, n):
        m = re.match(r"^blk\.(\d+)\.ffn_gate_up_exps\.weight$", n)
        return m.group(1) if m else None

    def order(self, tensors):
        def key(t):
            n = t["name"]
            if n == "token_embd.weight":
                cls = 0
            elif n == "token_embd_norm.weight":
                cls = 1
            elif n.startswith("blk."):
                cls = 2 + int(n.split(".")[1])
            else:
                cls = 1000
            return (cls, n)
        return sorted(tensors, key=key)

    def run(self, golden_dir=None, golden_blocks=512, seed=7):
        md = self.src.md
        arch = md.get("general.architecture", "unknown")
        nl = md.get("general.block_count", md.get("general.attention_layers"))
        tensors = self.order(self.src.tensors)
        # group expert gate_up/down pairs per layer so they can be merged
        exp = {}
        for t in tensors:
            m = re.match(r"^blk\.(\d+)\.ffn_(gate_up|down)_exps\.weight$", t["name"])
            if m:
                exp.setdefault(m.group(1), {})[m.group(2)] = t

        out = open(self.out, "wb")
        try:
            fcntl.fcntl(out, fcntl.F_NOCACHE, 1)
        except OSError:
            pass
        out.write(b"\x00" * HEADER_RESERVE)
        out.seek(HEADER_RESERVE)
        rng = np.random.default_rng(seed)
        records, expert_index, layer_bases = [], [], {}
        blob_hash = xxhash.xxh64()
        pos = HEADER_RESERVE
        golden = GoldenWriter(golden_dir) if golden_dir else None
        for t in tensors:
            n = t["name"]
            lay = self._expert_key(n)
            if lay is not None and exp[lay].get("gate_up") is t:  # merged expert record
                lay_base, rec_len, rec_hash, stride = self._write_experts(out, exp[lay], blob_hash, pos, golden, rng, golden_blocks)
                layer_bases[int(lay)] = lay_base
                pos = lay_base + rec_len
                gu, dn = exp[lay]["gate_up"], exp[lay]["down"]
                gname, dname = DTYPES[gu["dtype"]][0], DTYPES[dn["dtype"]][0]
                _, gbs, gts = DTYPES[gu["dtype"]]
                _, dbs, dts = DTYPES[dn["dtype"]]
                n_col = int(gu["dims"][0]); half_cols = int(gu["dims"][1] // 2)
                gate_len = half_cols * n_col // gbs * gts
                down_len = int(dn["dims"][1]) * int(dn["dims"][0]) // dbs * dts
                records.append({"name": n.replace("gate_up_exps", "experts"), "dtype": gname,
                                "shape": [int(gu["dims"][2])],
                                "parts": {"gate": [int(gu["dims"][1] // 2), int(gu["dims"][0])],
                                          "up": [int(gu["dims"][1] // 2), int(gu["dims"][0])],
                                          "down": [int(dn["dims"][1]), int(dn["dims"][0])]},
                                "part_dtypes": {"gate": gname, "up": gname, "down": dname},
                                "part_bytes": {"gate": gate_len, "up": gate_len, "down": down_len},
                                "region": "expert", "layout": "expert_gateupdown",
                                "offset": lay_base, "length": rec_len, "group_size": gbs,
                                "residency": "cold", "xxh64": rec_hash.hexdigest(), "expert_stride": stride})
                continue
            if n.startswith("blk.") and n.endswith(".ffn_down_exps.weight"):
                continue  # merged into ffn_experts.weight; keep sibling tensors (e.g. .scale)
            align = GPU_ALIGN
            pad = (align - pos % align) % align
            if pad:
                out.write(b"\x00" * pad); blob_hash.update(b"\x00" * pad)
                pos += pad
            nbytes = tensor_nbytes(t["dims"], t["dtype"])
            h = xxhash.xxh64()
            CHUNK = 1 << 24
            done = 0
            while done < nbytes:
                buf = self.src.read_tensor(t, done, min(CHUNK, nbytes - done))
                out.write(buf); h.update(buf); blob_hash.update(buf)
                if golden is not None:
                    golden.maybe(n, t["dtype"], buf, done, rng, golden_blocks)
                done += len(buf)
            if nbytes == 0:
                pass
            pos += nbytes
            records.append({"name": n, "dtype": DTYPES[t["dtype"]][0],
                            "shape": [int(d) for d in reversed(t["dims"])],
                            "region": "gpu", "layout": "rowmajor", "offset": pos - nbytes,
                            "length": nbytes, "group_size": DTYPES[t["dtype"]][1],
                            "residency": "hot" if ("norm" in n or "gate_inp" in n or "embd" in n
                                           or "scale" in n or n == "rope_freqs.weight"
                                           or n == "output.weight") else "warm",
                            "xxh64": h.hexdigest()})
        # blob pad to alignment then header
        blob_end = pos
        pad = (-blob_end) % CPU_ALIGN
        if pad:
            out.write(b"\x00" * pad)
        out.flush()
        os.fsync(out.fileno())
        if layer_bases:
            stride = next(r["expert_stride"] for r in records if "expert_stride" in r)
            expert_index = {"stride_mode": "per_layer", "layer_base": [layer_bases[i] for i in sorted(layer_bases)],
                            "expert_stride": stride, "order": "gate|up|down"}
        total = sum(r["length"] for r in records)
        hot = sum(r["length"] for r in records if r["residency"] != "cold")
        cold = total - hot
        hp = {k: v for k, v in md.items() if not any(k.startswith(s) or k == s for s in GgufInput.SKIP_KV)
              and not k.startswith("general.sf")}
        header = {
            "magic": "M64P", "version": 1, "tool": "m64pack 0.1",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": {"kind": type(self.src).__name__, "file": self.src_name,
                       "sha256_partial": ""},
            "model": {"arch": arch, "hparams": _jsonable(hp), "name": md.get("general.name", self.src_name)},
            "quant": {"profile_sha256": "", "tier_table": [
                {"pattern": "*", "dtype": "passthrough", "group": "source", "scale": "inline",
                 "algo": "ggml_block_verbatim"}]},
            "residency": {"total_weights_bytes": total, "hot_weights_bytes": hot,
                          "cold_expert_bytes": cold, "max_gpu_bytes": int(md.get("general.max_gpu_bytes", 0))},
            "alignment_log2": {"gpu": 14, "expert": 14, "cpu": 6},
            "blob_start": HEADER_RESERVE, "blob_bytes": blob_end - HEADER_RESERVE,
            "tensors": records,
            "checksum": {"blob_xxh64": blob_hash.hexdigest()},
        }
        if expert_index:
            header["expert_index"] = expert_index
        hj = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        if 16 + len(hj) > HEADER_RESERVE:
            raise RuntimeError(f"header JSON {len(hj)} B exceeds HEADER_RESERVE {HEADER_RESERVE} B")
        out.seek(0)
        out.write(MAGIC + struct.pack("<I", FMT_VERSION) + struct.pack("<Q", len(hj)))
        out.write(hj)
        out.flush()
        os.fsync(out.fileno())
        out.close()
        if golden is not None:
            golden.close()
        return header


def _jsonable(x):
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


class PackerHelpers:
    pass


# methods attached for readability
def _write_experts(self, out, pair, blob_hash, pos, golden, rng, golden_blocks):
    gu, dn = pair["gate_up"], pair["down"]
    E = gu["dims"][2]
    n_col = gu["dims"][0]            # ne0 = n_embd (contiguous per column)
    half_cols = gu["dims"][1] // 2   # n_ff per gate/up half
    d_col = dn["dims"][0]
    bs_g, ts_g = DTYPES[gu["dtype"]][1], DTYPES[gu["dtype"]][2]
    bs_d, ts_d = DTYPES[dn["dtype"]][1], DTYPES[dn["dtype"]][2]
    gate_bytes = half_cols * n_col // bs_g * ts_g
    down_bytes = dn["dims"][1] * d_col // bs_d * ts_d
    per_exp = 2 * gate_bytes + down_bytes
    stride = align_up(per_exp, GPU_ALIGN)
    layer_base = align_up(pos, GPU_ALIGN)
    if layer_base != pos:
        out.write(b"\x00" * (layer_base - pos)); blob_hash.update(b"\x00" * (layer_base - pos))
    h = xxhash.xxh64()
    CHUNK = 1 << 24
    parts = [
        (gu, 0, gate_bytes, DTYPES[gu["dtype"]][0] + ".gate"),
        (gu, gate_bytes, gate_bytes, DTYPES[gu["dtype"]][0] + ".up"),
        (dn, 0, down_bytes, "down"),
    ]
    for e in range(E):
        for part, (src_t, part_off, ln, suffix) in enumerate(parts):
            base = e * (2 * gate_bytes if src_t is gu else down_bytes) + part_off
            done = 0
            while done < ln:
                buf = self.src.read_tensor(src_t, base + done, min(CHUNK, ln - done))
                out.write(buf); h.update(buf); blob_hash.update(buf)
                # Golden volume control: fused experts keep per-(layer,expert) records
                # only for expert 0 and each part, so mixed gate/up/down dtypes
                # (e.g. q4_k gate|up + q5_0/q5_1 down) are all covered without
                # dumping one record per expert on 128-expert models.
                if golden is not None and e == 0:
                    golden.maybe(f"{src_t['name']}#0.{suffix}", src_t["dtype"], buf, done, rng, golden_blocks)
                done += len(buf)
        pad = stride - per_exp
        if pad:
            out.write(b"\x00" * pad); h.update(b"\x00" * pad); blob_hash.update(b"\x00" * pad)
    return layer_base, E * stride, h, stride


Packer._write_experts = _write_experts


class GoldenWriter:
    """golden.bin: M64G, count u32; per rec: u16 name_len+name, u32 dtype, u32 nblocks,
    u64 raw_len + raw, u64 nvals + f32 little-endian dequant."""

    def __init__(self, d):
        os.makedirs(d, exist_ok=True)
        self.f = open(os.path.join(d, "golden.bin"), "wb")
        self.count = 0
        self.buf = bytearray(struct.pack("<4sI", b"M64G", 0))

    def maybe(self, name, dt, raw, part_off, rng, max_blocks):
        if DTYPES[dt][0] in ("f32", "f16", "bf16"):
            if part_off != 0:
                return
            deq = dequant_blocks(dt, raw[: 1 << 20])
            blocks = len(deq)
            self._emit(name, dt, raw[: 1 << 20], deq)
        elif DTYPES[dt][0] in ("q8_0", "q8_1", "q4_k", "q5_0", "q5_1"):
            ts = DTYPES[dt][2]
            nb = len(raw) // ts
            k = min(max_blocks, nb)
            idx = np.sort(rng.choice(nb, size=k, replace=False))
            raw2 = b"".join(raw[i * ts:(i + 1) * ts] for i in idx)
            deq = dequant_blocks(dt, raw2)
            self._emit(name, dt, raw2, deq)

    def _emit(self, name, dt, raw, deq):
        nb = struct.pack
        self.buf += struct.pack("<H", len(name)) + name.encode() + struct.pack("<II", dt, len(raw) // DTYPES[dt][2])
        self.buf += struct.pack("<Q", len(raw)) + raw
        self.buf += struct.pack("<Q", len(deq)) + deq.astype("<f4").tobytes()
        self.count += 1

    def close(self):
        self.buf[4:8] = struct.pack("<I", self.count)
        self.f.write(self.buf)
        self.f.close()
        print(f"golden: {self.count} records -> {self.f.name}")


# ---------------------------------------------------------------- verify
def _read_pack_header(f):
    f.seek(0)
    if f.read(4) != MAGIC:
        raise ValueError("bad magic")
    ver = struct.unpack("<I", f.read(4))[0]
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hlen))
    assert header["magic"] == "M64P" and header["version"] == 1, "format mismatch"
    return header


def verify(path):
    f = open(path, "rb")
    fcntl.fcntl(f, fcntl.F_NOCACHE, 1)
    try:
        header = _read_pack_header(f)
    except ValueError as e:
        print("VERIFY FAIL:", e); return 1
    ok = bad = 0
    blobh = xxhash.xxh64()
    blob_end = header["blob_start"] + header["blob_bytes"]
    blob_done = 0
    f.seek(header["blob_start"])
    while blob_done < min(1 << 24, blob_end - header["blob_start"]):
        b = f.read(min(1 << 24, blob_end - header["blob_start"] - blob_done))
        blobh.update(b); blob_done += len(b)
    # full blob hash
    f.seek(header["blob_start"] + blob_done)
    rem = blob_end - header["blob_start"] - blob_done
    while rem:
        b = f.read(min(1 << 24, rem)); blobh.update(b); rem -= len(b)
    if blobh.hexdigest() != header["checksum"]["blob_xxh64"]:
        print("VERIFY FAIL: blob_xxh64 mismatch"); return 1
    for r in header["tensors"]:
        if r["region"] in ("gpu", "expert") and r["length"] > 0:
            al = 1 << header["alignment_log2"][r["region"]]
            if r["offset"] % al != 0:
                print(f"VERIFY FAIL: {r['name']} offset {r['offset']} not {al}-aligned"); return 1
        h = xxhash.xxh64()
        f.seek(r["offset"])
        rem = r["length"]
        while rem:
            b = f.read(min(1 << 24, rem)); h.update(b); rem -= len(b)
        if h.hexdigest() != r["xxh64"]:
            print(f"VERIFY FAIL: {r['name']} xxh64"); bad += 1
        else:
            ok += 1
    print(f"verify: {ok} tensors ok, {bad} bad; blob hash ok; size={os.path.getsize(path)}")
    return 0 if bad == 0 else 1


def info(path):
    with open(path, "rb") as f:
        h = _read_pack_header(f)
    m = h["model"]
    print(f"arch={m['arch']} tensors={len(h['tensors'])} weights={h['residency']['total_weights_bytes']/2**30:.2f} GiB "
          f"blob={h['blob_bytes']/2**30:.2f} GiB")
    if "expert_index" in h:
        print(f"expert_index: {len(h['expert_index']['layer_base'])} layers stride={h['expert_index']['expert_stride']}")
    for r in h["tensors"][:6]:
        print(" ", r["name"], r["dtype"], r["shape"], r["offset"], r["length"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pack", "verify", "info"])
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dump-golden")
    ap.add_argument("--golden-blocks", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.cmd in ("verify", "info"):
        sys.exit(verify(a.src) if a.cmd == "verify" else info(a.src))
    paths = [a.src]
    if a.src.endswith(".safetensors") and os.path.isdir(a.src):
        paths = [a.src]
    src = GgufInput(paths) if a.src.endswith(".gguf") else SfInput(paths)
    if not a.dst:
        sys.exit("pack needs OUT.m64pack")
    t0 = time.time()
    hdr = Packer(src, a.dst).run(golden_dir=a.dump_golden, golden_blocks=a.golden_blocks, seed=a.seed)
    print(f"packed {len(hdr['tensors'])} tensors, {hdr['residency']['total_weights_bytes']/2**30:.2f} GiB "
          f"in {time.time()-t0:.1f}s -> {a.dst}")
    if a.verify:
        sys.exit(verify(a.dst))


if __name__ == "__main__":
    main()
