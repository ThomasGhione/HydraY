#!/usr/bin/env python3
"""Generate random legal positions per material signature, for the coverage audit.

Why enumerate rather than sample from our own games: bucket 0 was empty
precisely because datagen never produced those positions (adjudication cut at
<=5 pieces). Sampling what we play would miss the same hole for the same reason.
Here coverage is decided up front by the list of signatures.

  ./gen_coverage.py [positions_per_signature] > positions.tsv
  format: signature<TAB>fen
"""
import itertools
import random
import sys

import chess

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
SYM = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
       chess.ROOK: "R", chess.QUEEN: "Q"}


def signatures(max_per_side=3):
    """All combinations with <= max_per_side non-king pieces per side.

    Only half the pairs (white >= black in canonical order): the evaluation is
    antisymmetric, so KQvKR and KRvKQ are the same question.
    """
    combos = []
    for n in range(0, max_per_side + 1):
        combos += [tuple(sorted(c)) for c in itertools.combinations_with_replacement(PIECES, n)]
    out = []
    for w, b in itertools.product(combos, repeat=2):
        if len(w) + len(b) == 0:
            continue                      # K vs K: drawn by material
        if (len(w), w) < (len(b), b):
            continue                      # keep only one of the two directions
        out.append((w, b))
    return out


def name(w, b):
    return "K" + "".join(SYM[p] for p in reversed(w)) + "vK" + "".join(SYM[p] for p in reversed(b))


def random_position(w, b, rng, tries=400):
    for _ in range(tries):
        board = chess.Board(None)
        squares = rng.sample(range(64), 2)
        wk, bk = squares
        if chess.square_distance(wk, bk) <= 1:
            continue
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        free = [s for s in range(64) if board.piece_at(s) is None]
        rng.shuffle(free)
        ok = True
        for piece, colour in [(p, chess.WHITE) for p in w] + [(p, chess.BLACK) for p in b]:
            # Pawns do not sit on rank 1 or 8.
            pool = free if piece != chess.PAWN else [s for s in free if 8 <= s < 56]
            if not pool:
                ok = False
                break
            sq = pool[0] if piece != chess.PAWN else rng.choice(pool)
            free.remove(sq)
            board.set_piece_at(sq, chess.Piece(piece, colour))
        if not ok:
            continue
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.clear_stack()
        # It has to be a playable position: legal, not already over, and the
        # opponent's king not in check (that would be the wrong side to move).
        if not board.is_valid():
            continue
        if board.is_game_over(claim_draw=False):
            continue
        # No checks. Choosing the side to move at random and then filtering
        # with is_valid() discards the positions where the STRONGER side gives
        # check only when it is that side's turn: the sample fills up with
        # positions where the side to move is in check (measured: 33.5%) and
        # every signature looks underestimated. Excluding them keeps the sample
        # symmetric.
        if board.is_check():
            continue
        return board.fen()
    return None


def main():
    per_sig = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rng = random.Random(20260825)
    sigs = signatures()
    print(f"# {len(sigs)} signatures x {per_sig} positions", file=sys.stderr)
    emitted = 0
    for w, b in sigs:
        label = name(w, b)
        seen = set()
        for _ in range(per_sig * 3):
            if len(seen) >= per_sig:
                break
            fen = random_position(w, b, rng)
            if fen and fen not in seen:
                seen.add(fen)
                print(f"{label}\t{fen}")
                emitted += 1
    print(f"# {emitted} positions emitted", file=sys.stderr)


if __name__ == "__main__":
    main()
