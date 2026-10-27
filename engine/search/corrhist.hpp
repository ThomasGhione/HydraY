#pragma once

// Correction history: how far the search score has been landing from the
// static eval, keyed by sub-structures of the position and added back to the
// eval before the pruning gates read it. A cell is a fixed-point gravity
// statistic in [-CORR_HIST_LIMIT, CORR_HIST_LIMIT]: an update adds
// residual * depth / 8 units (/ 10 after a raised alpha), and a full cell reads
// as CORR_HIST_LIMIT * weight / CORR_HIST_SCALE eval units.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../../board/board.hpp"
#include "search_constants.hpp"

namespace engine {

struct CorrectionHistory {
    // [side to move][key]; nonPawn adds the colour whose pieces form the key.
    int16_t pawn[2][CORR_HIST_SIZE] {};
    int16_t minor[2][CORR_HIST_SIZE] {};
    int16_t nonPawn[2][2][CORR_HIST_SIZE] {};

    struct Cells {
        int16_t* pawn;
        int16_t* minor;
        int16_t* nonPawnWhite;
        int16_t* nonPawnBlack;
    };

    Cells cellsFor(const chess::Board& b) noexcept {
        const int stm = chess::Board::colorToIndex(b.getActiveColor());
        return {&pawn[stm][pawnIndex(b)], &minor[stm][minorIndex(b)],
                &nonPawn[stm][0][nonPawnIndex(b, 0)], &nonPawn[stm][1][nonPawnIndex(b, 1)]};
    }

    // Correction to add to the raw static eval (side-to-move relative).
    int32_t correction(const chess::Board& b) noexcept {
        const Cells c = cellsFor(b);
        const int64_t sum = int64_t{CORR_WEIGHT_PAWN} * *c.pawn
                          + int64_t{CORR_WEIGHT_MINOR} * *c.minor
                          + int64_t{CORR_WEIGHT_NON_PAWN} * (*c.nonPawnWhite + *c.nonPawnBlack);
        return static_cast<int32_t>(sum * CORR_HIST_WEIGHT_PCT / (int64_t{CORR_HIST_SCALE} * 100));
    }

    void clear() noexcept {
        std::memset(pawn, 0, sizeof(pawn));
        std::memset(minor, 0, sizeof(minor));
        std::memset(nonPawn, 0, sizeof(nonPawn));
    }

private:
    // Multiply-shift: the top bits of each product depend on every bit of its
    // bitboard, and distinct odd multipliers keep piece types from aliasing.
    static constexpr int SHIFT = 64 - CORR_HIST_BITS;

    static size_t pawnIndex(const chess::Board& b) noexcept {
        return static_cast<size_t>((b.bb[chess::Board::PAWN][0] * 0x9E3779B97F4A7C15ULL
                                  + b.bb[chess::Board::PAWN][1] * 0xD6E8FEB86659FD93ULL) >> SHIFT);
    }
    static size_t minorIndex(const chess::Board& b) noexcept {
        return static_cast<size_t>((b.bb[chess::Board::KNIGHT][0] * 0xA0761D6478BD642FULL
                                  + b.bb[chess::Board::KNIGHT][1] * 0xE7037ED1A0B428DBULL
                                  + b.bb[chess::Board::BISHOP][0] * 0x8EBC6AF09C88C6E3ULL
                                  + b.bb[chess::Board::BISHOP][1] * 0x589965CC75374CC3ULL) >> SHIFT);
    }
    static size_t nonPawnIndex(const chess::Board& b, int c) noexcept {
        return static_cast<size_t>((b.bb[chess::Board::KNIGHT][c] * 0x1D8E4E27C47D124FULL
                                  + b.bb[chess::Board::BISHOP][c] * 0xBF58476D1CE4E5B9ULL
                                  + b.bb[chess::Board::ROOK][c]   * 0x94D049BB133111EBULL
                                  + b.bb[chess::Board::QUEEN][c]  * 0xC2B2AE3D27D4EB4FULL
                                  + b.bb[chess::Board::KING][c]   * 0x165667B19E3779F9ULL) >> SHIFT);
    }
};

} // namespace engine
