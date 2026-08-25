#!/usr/bin/env python3
"""Second audit pass: REAL positions, for the dense output buckets.

The first pass (gen_coverage.py) enumerates material signatures and places the
pieces at random. That works up to ~8 pieces, where every arrangement is a
plausible position. Above that it does not: thirty pieces placed at random do
not resemble chess, the net has legitimately never seen them, and the audit
would produce false positives.

So for the middlegame the positions come from games, and the stratification is
on the OUTPUT BUCKET -- which is how the net itself partitions material:
bucket = (popcount - 2) // 4.

  ./gen_coverage_real.py <file.pgn> [per_bucket] > positions_real.tsv
  format: b<bucket><TAB>fen

Positions are taken at intervals along the game and deduplicated, so the sample
is not filled with the same position repeated at every ply.
"""
import random
import sys

import chess
import chess.pgn

TARGETS = range(2, 8)          # the buckets the first pass does not reach


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

    print(f"# {games} games, per bucket: "
          + ", ".join(f"{b}:{got[b]}" for b in TARGETS), file=sys.stderr)


if __name__ == "__main__":
    main()
