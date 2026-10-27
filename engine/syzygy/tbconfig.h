/*
 * tbconfig.h - Pyrrhic configuration for HydraY.
 *
 * Pyrrhic numbers the squares from a1 = 0 to h8 = 63. Its attacks come from
 * the engine's tables, through syzygy.cpp.
 */

#pragma once

#include <stdint.h>

#define PYRRHIC_POPCOUNT(x) (__builtin_popcountll(x))
#define PYRRHIC_LSB(x)      (__builtin_ctzll(x))

/* Clears the lowest set bit of *b and returns its index: tbprobe.c uses the
 * result as a square. */
static inline unsigned pyrrhic_poplsb(uint64_t *b) {
    unsigned s = (unsigned)__builtin_ctzll(*b);
    *b &= *b - 1;
    return s;
}
#define PYRRHIC_POPLSB(p) pyrrhic_poplsb(p)

uint64_t hydray_tb_pawn_attacks(int sq, int white);
uint64_t hydray_tb_knight_attacks(int sq);
uint64_t hydray_tb_bishop_attacks(int sq, uint64_t occ);
uint64_t hydray_tb_rook_attacks(int sq, uint64_t occ);
uint64_t hydray_tb_king_attacks(int sq);

#define PYRRHIC_PAWN_ATTACKS(sq, c)     hydray_tb_pawn_attacks((sq), (c))
#define PYRRHIC_KNIGHT_ATTACKS(sq)      hydray_tb_knight_attacks(sq)
#define PYRRHIC_BISHOP_ATTACKS(sq, occ) hydray_tb_bishop_attacks((sq), (occ))
#define PYRRHIC_ROOK_ATTACKS(sq, occ)   hydray_tb_rook_attacks((sq), (occ))
#define PYRRHIC_QUEEN_ATTACKS(sq, occ)  (hydray_tb_bishop_attacks((sq), (occ)) | hydray_tb_rook_attacks((sq), (occ)))
#define PYRRHIC_KING_ATTACKS(sq)        hydray_tb_king_attacks(sq)
