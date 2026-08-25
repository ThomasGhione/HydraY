#!/usr/bin/env python3
"""Oracolo Stockfish per l'audit di copertura.

Legge "firma<TAB>fen" da stdin, scrive "firma<TAB>cp<TAB>fen" su stdout, con cp
dal punto di vista del lato al tratto (come la nostra eval). I punteggi di matto
diventano +-MATE_CP: per l'audit conta "quanto in vantaggio", non il numero di
mosse.

  ./sf_oracle.py <sf_bin> [nodi|static] [processi] < positions.tsv > oracle.tsv

Con "static" si usa il comando `eval` di Stockfish invece della ricerca. Serve
per confrontare mele con mele: la nostra e' una valutazione STATICA, e una
ricerca da 200k nodi trova tattiche che nessuna statica puo' vedere -- il
confronto misurerebbe "statica contro ricerca", non i buchi di copertura.
"""
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

MATE_CP = 8000


def run_chunk(args):
    sf_bin, nodes, rows = args
    p = subprocess.Popen([sf_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1)
    p.stdin.write("uci\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.startswith("uciok"):
            break
    p.stdin.write("setoption name Threads value 1\nsetoption name Hash value 16\nisready\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.startswith("readyok"):
            break

    out = []
    if nodes == "static":
        for sig, fen in rows:
            p.stdin.write(f"position fen {fen}\neval\n")
            p.stdin.flush()
            cp = None
            for line in p.stdout:
                if line.startswith("Final evaluation"):
                    tok = line.split()
                    try:
                        cp = int(round(float(tok[2]) * 100))
                    except (ValueError, IndexError):
                        cp = None
                    break
            if cp is not None:
                # `eval` riporta dal punto di vista del BIANCO; noi vogliamo
                # dal lato al tratto, come la nostra eval.
                if fen.split()[1] == "b":
                    cp = -cp
                out.append(f"{sig}\t{cp}\t{fen}")
        p.stdin.write("quit\n")
        p.stdin.flush()
        p.wait()
        return out

    for sig, fen in rows:
        p.stdin.write(f"position fen {fen}\ngo nodes {nodes}\n")
        p.stdin.flush()
        cp = None
        for line in p.stdout:
            if line.startswith("info ") and " score " in line:
                tok = line.split()
                i = tok.index("score")
                if tok[i + 1] == "cp":
                    cp = int(tok[i + 2])
                elif tok[i + 1] == "mate":
                    m = int(tok[i + 2])
                    cp = MATE_CP if m > 0 else -MATE_CP
            elif line.startswith("bestmove"):
                break
        if cp is not None:
            out.append(f"{sig}\t{cp}\t{fen}")
    p.stdin.write("quit\n")
    p.stdin.flush()
    p.wait()
    return out


def main():
    sf_bin = sys.argv[1]
    nodes = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] == "static" \
        else (int(sys.argv[2]) if len(sys.argv) > 2 else 200_000)
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, (os.cpu_count() or 4) - 1)

    rows = [tuple(l.rstrip("\n").split("\t")) for l in sys.stdin if l.strip()]
    # Molti chunk piccoli invece di uno per processo: i risultati escono via via
    # (con uno per processo il file resta vuoto fino alla fine) e un chunk lento
    # non tiene fermo un core.
    size = max(50, len(rows) // (procs * 20))
    chunks = [(sf_bin, nodes, rows[i:i + size]) for i in range(0, len(rows), size)]
    print(f"# {len(rows)} posizioni, {procs} processi, {nodes} nodi, "
          f"{len(chunks)} chunk da {size}", file=sys.stderr)
    done = 0
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for res in ex.map(run_chunk, chunks):
            for line in res:
                print(line)
            sys.stdout.flush()
            done += 1
            if done % 20 == 0:
                print(f"# {done}/{len(chunks)} chunk", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
