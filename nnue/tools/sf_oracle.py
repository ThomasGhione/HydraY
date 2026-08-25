#!/usr/bin/env python3
"""Stockfish oracle for the coverage audit.

Reads "signature<TAB>fen" on stdin, writes "signature<TAB>cp<TAB>fen" on stdout,
with cp from the side-to-move's point of view (like our own eval). Mate scores
become +-MATE_CP: what matters for the audit is "how far ahead", not the number
of moves.

  ./sf_oracle.py <sf_bin> [nodi|static] [processi] < positions.tsv > oracle.tsv

With "static" it uses Stockfish's `eval` command instead of a search. That is
needed to compare like with like: ours is a STATIC evaluation, and a 200k-node
search finds tactics no static eval can see -- the comparison would measure
"static versus search", not coverage holes.
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
                # `eval` reports from WHITE's point of view; we want it from
                # the side to move, like our own eval.
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
    # Many small chunks instead of one per process: results come out as they
    # land (with one per process the file stays empty until the end) and a slow
    # chunk does not idle a core.
    size = max(50, len(rows) // (procs * 20))
    chunks = [(sf_bin, nodes, rows[i:i + size]) for i in range(0, len(rows), size)]
    print(f"# {len(rows)} positions, {procs} processes, {nodes} nodes, "
          f"{len(chunks)} chunks of {size}", file=sys.stderr)
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
