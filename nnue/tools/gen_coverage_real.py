#!/usr/bin/env python3
"""Seconda passata dell'audit: posizioni REALI, per i bucket d'uscita densi.

La prima passata (gen_coverage.py) enumera firme di materiale e piazza i pezzi a
caso. Va bene fino a ~8 pezzi, dove ogni disposizione e' una posizione
plausibile. Sopra non funziona: trenta pezzi a caso non somigliano agli scacchi,
la rete non li ha mai visti a ragion veduta, e l'audit produrrebbe falsi
positivi.

Quindi per il mediogioco le posizioni vengono dalle partite, e la stratificazione
e' sul BUCKET D'USCITA — che e' come la rete stessa partiziona il materiale:
bucket = (popcount - 2) // 4.

  ./gen_coverage_real.py <file.pgn> [per_bucket] > positions_real.tsv
  formato: b<bucket><TAB>fen

Le posizioni sono prese a intervalli lungo la partita e deduplicate, per non
riempire il campione con la stessa posizione ripetuta a ogni semimossa.
"""
import random
import sys

import chess
import chess.pgn

TARGETS = range(2, 8)          # i bucket che la prima passata non tocca


def main():
    pgn_path = sys.argv[1]
    per_bucket = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    rng = random.Random(20260825)

    want = {b: per_bucket for b in TARGETS}
    seen = set()
    got = {b: 0 for b in TARGETS}
    games = 0

    with open(pgn_path) as f:
        while any(got[b] < want[b] for b in TARGETS):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games += 1
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply < 8 or rng.random() > 0.25:
                    continue
                if board.is_game_over(claim_draw=False):
                    continue
                bucket = (chess.popcount(board.occupied) - 2) // 4
                if bucket not in want or got[bucket] >= want[bucket]:
                    continue
                key = board.board_fen() + (" w" if board.turn else " b")
                if key in seen:
                    continue
                seen.add(key)
                got[bucket] += 1
                print(f"b{bucket}\t{board.fen()}")

    print(f"# {games} partite, per bucket: "
          + ", ".join(f"{b}:{got[b]}" for b in TARGETS), file=sys.stderr)


if __name__ == "__main__":
    main()
