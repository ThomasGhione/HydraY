#pragma once

#include <cstdint>
#include <cstring>

#include "../../board/board.hpp"
#include "../../tt/tt.hpp"
#include "../eval_constants.hpp"
#include "../movelist.hpp"
#include "../search/search_constants.hpp"
#include "movepicker.hpp"

namespace engine {

// Full definition lives in searcher.hpp; sorter.cpp only needs the members.
struct SearchRuntime;

class Sorter final {
public:
    Sorter() = delete;
    ~Sorter() = delete;

    // All scoring constants live in search_constants.hpp (namespace engine).
    // MovePicker (lazy selection-sort picker + deferred SEE) lives in movepicker.hpp.

    // Move::promotionType is already the Board piece type; 0 (not a promotion /
    // garbage sentinel) contributes no delta instead of silently acting as QUEEN.
    static inline constexpr int32_t getPromotionValueDelta(uint8_t promotionType) noexcept {
        return (promotionType == chess::Board::EMPTY) ? 0
                : PIECE_VALUES[promotionType] - PIECE_VALUES[chess::Board::PAWN];
    }

    // encodedHashMove is the TT move from the caller's node probe (0 = none);
    // the sorter itself never touches the TT.
    static MovePicker sortLegalMoves(
        MoveList moves,
        int ply,
        const chess::Board& b,
        const SearchRuntime& runtime,
        uint16_t encodedHashMove = 0,
        const chess::Move* previousMove = nullptr,
        bool* outHashMoveIsLegal = nullptr,
        const int16_t* contHistEntry = nullptr) noexcept;

    struct CaptureInfo {
        bool isCapture;
        int  victimType;
    };

    static CaptureInfo classifyCapture(
        const chess::Move& m,
        int fromPieceType,
        int toPieceType,
        const chess::Square& enPassant) noexcept;

    static int32_t staticExchangeEvaluation(
        const chess::Board& b, 
        const chess::Move& m) noexcept;

    static bool givesCheckAfterQuietMoveFast(
        const chess::Board& b,
        const chess::Move& m,
        int fromPieceType,
        int oppKingSq,
        uint64_t occ) noexcept;

    static MovePicker sortTacticalMoves(
        const MoveList& tacticalMoves,
        const chess::Board& b,
        int32_t standPat,
        int32_t alpha,
        int ply) noexcept;

    static MoveList sortEvasionsForcingFirst(
        MoveList evasions,
        const chess::Board& b) noexcept;

private:
    struct MoveOrderingContext {
        const chess::Move* previousMove;
        const SearchRuntime& runtime;
        const int16_t* contHistEntry;
        int ply;
        int usSide;
    };

    // square == 64 means "no attacker" (type then unspecified).
    struct LeastValuableAttacker {
        int square;
        int type;
    };

    static int32_t scoreMoveOrderingPriorityInline(
        const MoveOrderingContext& ctx,
        const chess::Move& m,
        bool isCapture,
        int victimType,
        bool isPromotionCandidate,
        bool isHashMove,
        int fromPieceType) noexcept;

    static LeastValuableAttacker getLeastValuableAttackerTo(
        const chess::Board& b,
        int sq,
        uint64_t occLocal,
        int sideLocal) noexcept;

    static bool isForcingEvasion(
        const chess::Board& b,
        const chess::Move& m,
        const chess::Square& enPassant) noexcept;
};

} // namespace engine
