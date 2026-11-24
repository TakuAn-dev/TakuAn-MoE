#!/usr/bin/env python3
"""Scan golden.bin for NaN/Inf dequant records and report which tensors/dtypes are affected."""
import struct, sys
import numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: golden_nan_scan.py GOLDEN.bin")
path = sys.argv[1]
f = open(path, "rb")
magic, count = struct.unpack("<4sI", f.read(8))
assert magic == b"M64G", magic
bad = {}
total = 0
for i in range(count):
    (nl,) = struct.unpack("<H", f.read(2))
    name = f.read(nl).decode()
    dt, nblocks = struct.unpack("<II", f.read(8))
    (rlen,) = struct.unpack("<Q", f.read(8))
    raw = f.read(rlen)
    (nvals,) = struct.unpack("<Q", f.read(8))
    deq = np.frombuffer(f.read(nvals * 4), dtype="<f4")
    total += 1
    nan = int(np.isnan(deq).sum())
    inf = int(np.isinf(deq).sum())
    if nan or inf:
        bad[name] = (dt, nblocks, nan, inf)
f.close()
print(f"records={total} bad={len(bad)}")
for name, (dt, nb, nan, inf) in sorted(bad.items()):
    print(f"dt={dt} blocks={nb} nan={nan} inf={inf}  {name}")
