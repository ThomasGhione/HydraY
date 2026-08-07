#include "evaluator.hpp"

#include "../nnue/nnue.hpp"
#include "search/search_constants.hpp"

namespace engine {

int32_t Evaluator::evaluate(const chess::Board& board) noexcept {
    // A missing king is a decisive, not an infinite, score: it must stay inside
    // the int16_t range every other search score lives in.
    const bool whiteToMove = (board.getActiveColor() == chess::Board::WHITE);
    if (board.kings_bb[0] == 0 || board.kings_bb[1] == 0) [[unlikely]] {
        if (board.kings_bb[0] == 0) return whiteToMove ? -MATE_VALUE : MATE_VALUE;
        return whiteToMove ? MATE_VALUE : -MATE_VALUE;
    }

    return NNUE::evaluate(board);
}

} // namespace engine
