#pragma once

#include <cstdint>

// Piece values, the last survivors of the handcrafted evaluator's constant
// set (removed in the NNUE era). They feed SEE, MVV capture ordering and the
// qsearch delta-pruning margins — search bookkeeping, not evaluation.

namespace engine {

// ===================================================
// PIECE BASE VALUES (scalar — used for SEE, MVV, qsearch delta margins)
// ===================================================
inline constexpr int32_t PAWN_VALUE   =    100;
inline constexpr int32_t KNIGHT_VALUE =    344;
inline constexpr int32_t BISHOP_VALUE =    359;
inline constexpr int32_t ROOK_VALUE   =    502;
inline constexpr int32_t QUEEN_VALUE  =    960;
inline constexpr int32_t KING_VALUE   = 20'000;

// Indexed by piece type (0=EMPTY, 1=PAWN, 2=KNIGHT, 3=BISHOP, 4=ROOK, 5=QUEEN, 6=KING, 7=unused)
inline constexpr int32_t PIECE_VALUES[8] = {
    0, PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE, 0
};

// ===================================================
// MVV (Most Valuable Victim) table for capture ordering
// ===================================================
inline constexpr int32_t MVV_TABLE[7] = {
    0,                  // EMPTY
    PAWN_VALUE * 10,    // PAWN
    KNIGHT_VALUE * 10,  // KNIGHT
    BISHOP_VALUE * 10,  // BISHOP
    ROOK_VALUE * 10,    // ROOK
    QUEEN_VALUE * 10,   // QUEEN
    KING_VALUE * 10     // KING
};

} // namespace engine
