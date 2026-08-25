#!/usr/bin/env python3
"""Compare our static eval against the Stockfish oracle, per material signature.

  ./coverage_report.py <oracle.tsv> <ours.tsv> [frequenze.tsv]

The scale problem: the two evaluations are in different "centipawns" (SCALE=400
on our side, Stockfish's own normalisation on theirs). Comparing them raw would
measure the calibration difference, not the holes. So the global map ours ~ a*sf
is estimated ONCE, and the residuals are read PER SIGNATURE: a region whose
residual is systematically negative is a bucket-0, i.e. a part of the input
space the net has never seen.

Signatures are ranked by |bias| (systematic error), not by absolute error: noise
on individual positions is not the defect we are looking for.
"""
import statistics
import sys
from collections import defaultdict

CLAMP = 2000


def load(path, value_index):
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            fen = parts[-1]
            try:
                out[fen] = (parts[0], int(parts[value_index]))
            except ValueError:
                continue
    return out


def main():
    oracle = load(sys.argv[1], 1)          # firma, cp, fen
    ours_raw = {}
    with open(sys.argv[2]) as f:           # eval, bucket, fen
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] == "NA":
                continue
            ours_raw[parts[-1]] = int(parts[0])

    freq = {}
    if len(sys.argv) > 3:
        with open(sys.argv[3]) as f:
            for line in f:
                sig, n = line.split()
                freq[sig] = int(n)

    common = [(sig, sf, ours_raw[fen]) for fen, (sig, sf) in oracle.items()
              if fen in ours_raw]
    if not common:
        sys.exit("no positions in common")

    # Global map ours ~ a*sf, from the median of the ratios (robust to tails).
    ratios = [o / s for _, s, o in common if abs(s) > 100]
    a = statistics.median(ratios) if ratios else 1.0

    # WARNING: clamping BEFORE rescaling manufactures a huge bias on crushing
    # positions: with a~3, a Stockfish score clamped to CLAMP becomes 3*CLAMP
    # against our CLAMP, and every overwhelming material edge looks
    # underestimated by thousands of centipawns. Bring both to the SAME scale
    # first, clamp afterwards.
    clip = lambda v: max(-CLAMP, min(CLAMP, v))
    per = defaultdict(list)
    for sig, s, o in common:
        per[sig].append(clip(o) - clip(a * s))

    rows = []
    for sig, res in per.items():
        if len(res) < 20:
            continue
        rows.append((sig, len(res), statistics.mean(res),
                     statistics.mean(abs(r) for r in res), freq.get(sig, 0)))

    print(f"positions compared: {len(common)}   estimated scale factor a = {a:.3f}")
    print(f"signatures with >=20 positions: {len(rows)}\n")

    def show(title, key, n=18):
        print(title)
        print(f"  {'signature':<16}{'n':>5}{'bias':>9}{'|err|':>9}{'in games':>12}")
        for sig, cnt, bias, mae, fr in sorted(rows, key=key)[:n]:
            print(f"  {sig:<16}{cnt:>5}{bias:>9.0f}{mae:>9.0f}{fr:>12}")
        print()

    show("UNDERESTIMATED -- the net sees less edge than the truth (the bucket-0 case):",
         lambda r: r[2])
    show("OVERESTIMATED -- the net sees more edge than the truth:",
         lambda r: -r[2])
    if freq:
        show("WEIGHTED BY IN-GAME FREQUENCY (|bias| x occurrences):",
             lambda r: -abs(r[2]) * r[4])


if __name__ == "__main__":
    main()
