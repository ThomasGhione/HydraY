#!/usr/bin/env python3
"""Cuts a 1024-wide deep net (no threats) to the 768-wide layout of this branch.

Not a trained net: it keeps the pairwise pairs (j, j + 512) for j < 384 of the
source and drops the other 128, so the file is valid for the engine (and for
sanity_deep) and good for tests, the embedded placeholder and timing. Its
strength is far below the source's: train a real one with colab_v10_threats.

The columns of l0 (l0w rows, l0b) and of l1 that survive: source 0..383 become
0..383 and source 512..895 become 384..767, so the new pair (j, j + 384) is the
old pair (j, j + 512).

Usage: cut_width.py <in.nnue> <out.nnue>      (in: 8 or 16 output buckets, no threats)
"""
import struct
import sys

SRC, DST = 1024, 768
INPUTS, L1 = 4 * 768, 16
KEEP = list(range(0, DST // 2)) + list(range(SRC // 2, SRC // 2 + DST // 2))


def layout(size):
    for ob in (16, 8):
        pay = INPUTS * SRC * 2 + SRC * 2 + ob * L1 * SRC + ob * L1 * 4 * 2 + ob * 4
        if size == (pay + 63) // 64 * 64:
            return ob, pay
    sys.exit(f"{size} bytes: not a {SRC}-wide net without threats")


def main():
    src, dst = sys.argv[1:3]
    data = open(src, "rb").read()
    ob, pay = layout(len(data))
    off = 0

    def take(n):
        nonlocal off
        chunk = data[off:off + n]
        off += n
        return chunk

    l0w = take(INPUTS * SRC * 2)
    l0b = take(SRC * 2)
    l1w = take(ob * L1 * SRC)
    tail = take(pay - off)  # l1b, l2w, l2b: float, width-free

    out = bytearray()
    for f in range(INPUTS):
        row = struct.unpack_from(f"<{SRC}h", l0w, f * SRC * 2)
        out += struct.pack(f"<{DST}h", *(row[i] for i in KEEP))
    out += struct.pack(f"<{DST}h", *(struct.unpack(f"<{SRC}h", l0b)[i] for i in KEEP))
    for r in range(ob * L1):
        row = l1w[r * SRC:(r + 1) * SRC]
        out += bytes(row[i] for i in KEEP)
    out += tail
    pad = (-len(out)) % 64
    out += (b"bullet" * 12)[:pad]
    open(dst, "wb").write(out)
    print(f"{dst}: {len(out)} bytes, {ob} output buckets, {SRC} -> {DST}")


if __name__ == "__main__":
    main()
