#!/usr/bin/env python3
"""Confronta la nostra eval statica con l'oracolo Stockfish, per firma di materiale.

  ./coverage_report.py <oracle.tsv> <ours.tsv> [frequenze.tsv]

Il problema di scala: le due eval sono in "centipedine" diverse (SCALE=400 da
noi, la normalizzazione di SF da loro). Confrontarle grezze misurerebbe la
differenza di taratura, non i buchi. Quindi si stima UNA volta sola la mappa
globale ours ~ a*sf, e si guardano i residui PER FIRMA: una regione dove il
residuo e' sistematicamente negativo e' un bucket-0, cioe' una zona dello spazio
d'ingresso che la rete non ha mai visto.

Le firme sono ordinate per |bias| (errore sistematico), non per errore assoluto:
il rumore su singole posizioni non e' il difetto che cerchiamo.
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
        sys.exit("nessuna posizione in comune")

    # Mappa globale ours ~ a*sf, dalla mediana dei rapporti (robusta alle code).
    ratios = [o / s for _, s, o in common if abs(s) > 100]
    a = statistics.median(ratios) if ratios else 1.0

    # ⚠️ Ritagliare PRIMA di riscalare fabbrica un bias enorme sulle posizioni
    # stravinte: con a~3, un SF ritagliato a CLAMP diventa 3*CLAMP contro il
    # nostro CLAMP, e ogni materiale schiacciante sembra sottovalutato di
    # migliaia di centipedine. Si porta tutto nella STESSA scala e si ritaglia
    # dopo.
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

    print(f"posizioni confrontate: {len(common)}   fattore di scala stimato a = {a:.3f}")
    print(f"firme con >=20 posizioni: {len(rows)}\n")

    def show(title, key, n=18):
        print(title)
        print(f"  {'firma':<16}{'n':>5}{'bias':>9}{'|err|':>9}{'in partita':>12}")
        for sig, cnt, bias, mae, fr in sorted(rows, key=key)[:n]:
            print(f"  {sig:<16}{cnt:>5}{bias:>9.0f}{mae:>9.0f}{fr:>12}")
        print()

    show("SOTTOVALUTATE — la rete vede meno vantaggio del vero (il caso bucket 0):",
         lambda r: r[2])
    show("SOPRAVVALUTATE — la rete vede piu' vantaggio del vero:",
         lambda r: -r[2])
    if freq:
        show("PESATE PER FREQUENZA IN PARTITA (|bias| x occorrenze):",
             lambda r: -abs(r[2]) * r[4])


if __name__ == "__main__":
    main()
