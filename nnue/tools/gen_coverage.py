#!/usr/bin/env python3
"""Genera posizioni legali casuali per firma di materiale, per l'audit di copertura.

Perche' enumerare invece di pescare dalle nostre partite: il bucket 0 era vuoto
proprio perche' datagen non produceva mai quelle posizioni (l'adjudication
chiudeva a <=5 pezzi). Campionare da cio' che giochiamo rimancherebbe il buco
per la stessa ragione. Qui la copertura e' decisa a priori dalla lista di firme.

  ./gen_coverage.py [posizioni_per_firma] > positions.tsv
  formato: firma<TAB>fen
"""
import itertools
import random
import sys

import chess

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
SYM = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
       chess.ROOK: "R", chess.QUEEN: "Q"}


def signatures(max_per_side=3):
    """Tutte le combinazioni con <= max_per_side pezzi non-re per lato.

    Solo meta' delle coppie (white >= black in ordine canonico): la valutazione
    e' antisimmetrica, quindi KQvKR e KRvKQ sono la stessa domanda.
    """
    combos = []
    for n in range(0, max_per_side + 1):
        combos += [tuple(sorted(c)) for c in itertools.combinations_with_replacement(PIECES, n)]
    out = []
    for w, b in itertools.product(combos, repeat=2):
        if len(w) + len(b) == 0:
            continue                      # K vs K: patta per materiale
        if (len(w), w) < (len(b), b):
            continue                      # tenuta una sola delle due direzioni
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
            # I pedoni non stanno su traversa 1 o 8.
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
        # Deve essere una posizione giocabile: legale, non gia' finita, e il re
        # avversario non sotto scacco (sarebbe il turno sbagliato).
        if not board.is_valid():
            continue
        if board.is_game_over(claim_draw=False):
            continue
        # Niente scacchi. Scegliendo il tratto a caso e filtrando poi con
        # is_valid(), le posizioni in cui il lato FORTE da' scacco vengono
        # scartate solo quando tocca a lui: il campione si riempie di posizioni
        # in cui il lato al tratto e' sotto scacco (misurato: 33,5%) e ogni
        # firma sembra sottovalutata. Escluderle rende il campione simmetrico.
        if board.is_check():
            continue
        return board.fen()
    return None


def main():
    per_sig = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rng = random.Random(20260825)
    sigs = signatures()
    print(f"# {len(sigs)} firme x {per_sig} posizioni", file=sys.stderr)
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
    print(f"# {emitted} posizioni emesse", file=sys.stderr)


if __name__ == "__main__":
    main()
