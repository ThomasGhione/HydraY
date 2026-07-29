#pragma once

// Correction history: exponentially-smoothed (search - static eval) residuals
// keyed by hashes of board sub-structures, blended back into the static eval
// before it feeds pruning decisions (RFP, futility, improving, qsearch
// stand-pat). Pawn structure plus minor (N+B) and major (R+Q) skeletons give
// three semi-independent signals; each contributes residual/CORR_HIST_DIVISOR
// and the sum is capped at CORR_TOTAL_CAP.
//
// Tables are per-SearchRuntime (one per Lazy SMP thread) and persist across
// searches of a game; softResetHistory intentionally leaves them alone.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../../board/board.hpp"
#include "search_constants.hpp"

namespace engine {

struct CorrectionHistory {
    int16_t pawn[2][CORR_HISTORY_SIZE] {};
    int16_t minor[2][CORR_HISTORY_SIZE] {};
    int16_t major[2][CORR_HISTORY_SIZE] {};

    // Correction to add to the raw static eval (side-to-move-relative).
    [[nodiscard]] int32_t correction(const chess::Board& b) const noexcept {
        const int side = chess::Board::colorToIndex(b.getActiveColor());
        const int32_t c = pawn[side][pawnIndex(b)]   / CORR_HIST_DIVISOR
                        + minor[side][minorIndex(b)] / CORR_HIST_DIVISOR
                        + major[side][majorIndex(b)] / CORR_HIST_DIVISOR;
        return std::clamp(c, -CORR_TOTAL_CAP, CORR_TOTAL_CAP);
    }

    // Blend one node's residual into every table. The caller enforces the
    // trustworthiness gates (depth, quiet best move, non-mate, bound checks);
    // deeper nodes weigh more, shallow noise is suppressed.
    void update(const chess::Board& b, int32_t searchScore, int32_t staticEval,
                int depth) noexcept {
        const int side = chess::Board::colorToIndex(b.getActiveColor());
        const int32_t residual =
            std::clamp(searchScore - staticEval, -CORR_HIST_LIMIT, CORR_HIST_LIMIT);
        const int w = std::min(depth, CORR_HIST_MAX_W);
        const auto blend = [&](int16_t& cell) noexcept {
            cell = static_cast<int16_t>(
                (cell * (CORR_HIST_BLEND - w) + residual * w) / CORR_HIST_BLEND);
        };
        blend(pawn[side][pawnIndex(b)]);
        blend(minor[side][minorIndex(b)]);
        blend(major[side][majorIndex(b)]);
    }

private:
    static size_t indexOf(uint64_t a, uint64_t c) noexcept {
        const uint64_t k = a * 0x9E3779B97F4A7C15ULL + c * 0xD6E8FEB86659FD93ULL;
        return static_cast<size_t>(k >> 48) & (CORR_HISTORY_SIZE - 1);
    }
    static size_t pawnIndex(const chess::Board& b) noexcept {
        return indexOf(b.pawns_bb[0], b.pawns_bb[1]);
    }
    static size_t minorIndex(const chess::Board& b) noexcept {
        return indexOf(b.knights_bb[0] | b.bishops_bb[0],
                       b.knights_bb[1] | b.bishops_bb[1]);
    }
    static size_t majorIndex(const chess::Board& b) noexcept {
        return indexOf(b.rooks_bb[0] | b.queens_bb[0],
                       b.rooks_bb[1] | b.queens_bb[1]);
    }
};

} // namespace engine
