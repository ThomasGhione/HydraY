#pragma once

#include <cstdint>

#include "../../board/board.hpp"
#include "../../tt/tt.hpp"
#include "../movelist.hpp"
#include "../search/search_constants.hpp"
#include "movepicker.hpp"

namespace engine {

struct SearchRuntime;

// What the search and the sorter need to know about a move before playing it.
struct MoveTraits {
    int  fromType;
    int  victimType;   // PAWN for en passant, EMPTY for a non-capture
    bool isCapture;
    bool isPromotion;
};

inline MoveTraits moveTraits(const chess::Board& b, const chess::Move& m,
                             chess::Square enPassant, int promotionRank) noexcept {
    using chess::Board;
    const int fromType = b.get(m.from) & Board::MASK_PIECE_TYPE;
    const int toType   = b.get(m.to)   & Board::MASK_PIECE_TYPE;
    const bool ep = m.to == enPassant && fromType == Board::PAWN && toType == Board::EMPTY
                 && chess::file(m.from) != chess::file(m.to);
    return {fromType, ep ? static_cast<int>(Board::PAWN) : toType, toType != Board::EMPTY || ep,
            fromType == Board::PAWN && chess::rank(m.to) == promotionRank};
}

namespace Sorter {

// encodedHashMove is the TT move from the caller's node probe (0 = none).
MovePicker sortLegalMoves(MoveList moves, int ply, const chess::Board& b, const SearchRuntime& runtime,
                          uint16_t encodedHashMove = 0, const chess::Move* previousMove = nullptr,
                          bool* outHashMoveIsLegal = nullptr, const int16_t* contHistEntry = nullptr,
                          const int16_t* const* contX = nullptr) noexcept;
// The same for the moves already in picker; threats = b.attackedBy(opponent).
void scoreLegalMoves(MovePicker& picker, int ply, const chess::Board& b, const SearchRuntime& runtime,
                     uint16_t encodedHashMove, const chess::Move* previousMove,
                     bool* outHashMoveIsLegal, const int16_t* contHistEntry, uint64_t threats,
                     const int16_t* const* contX = nullptr) noexcept;

MovePicker sortTacticalMoves(const MoveList& tacticalMoves, const chess::Board& b,
                             int32_t standPat, int32_t alpha, int ply) noexcept;

// Captures, en passant and promotions first, both groups in generation order.
void sortEvasionsForcingFirst(MoveList& evasions, const chess::Board& b) noexcept;

bool givesCheckAfterQuietMoveFast(const chess::Board& b, const chess::Move& m,
                                  int fromPieceType, int oppKingSq, uint64_t occ) noexcept;

} // namespace Sorter

} // namespace engine
