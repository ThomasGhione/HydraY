#pragma once

#include <cstdint>

#include "../../board/board.hpp"
#include "../movelist.hpp"
#include "sorter.hpp"

namespace engine::MoveGenerator {

// `checkers` is Board::checkersTo for the side to move (0 = not in check).
MoveList generateLegalMoves(const chess::Board& b, uint64_t checkers) noexcept;
inline MoveList generateLegalMoves(const chess::Board& b) noexcept {
    return generateLegalMoves(b, b.checkersTo(b.getActiveColor()));
}

// Legal captures (en passant included) and promotions.
MoveList generateTacticalMoves(const chess::Board& b) noexcept;

MovePicker generateQSearchEvasions(const chess::Board& b, uint64_t checkers) noexcept;
MovePicker generateQSearchTacticalMoves(const chess::Board& b, int32_t standPat,
                                        int32_t alpha, int ply) noexcept;

} // namespace engine::MoveGenerator
