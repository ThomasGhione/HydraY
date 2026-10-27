#include "sorter.hpp"

#include <algorithm>
#include <bit>

#include "../engine.hpp"

namespace engine::Sorter {

namespace {

using chess::Board;

// The scorer's node-invariant inputs, resolved once per node.
struct MoveOrderingContext {
    const SearchRuntime& runtime;
    const int16_t* contHistEntry;
    int usSide;
    chess::Move killer0;
    chess::Move killer1;
    uint16_t counterMove;   // 0 = none
};

// Move::promotionType is already the piece type; 0 (no promotion) adds nothing.
constexpr int32_t promotionGain(uint8_t promotionType) noexcept {
    return (promotionType == Board::EMPTY) ? 0 : PIECE_VALUES[promotionType] - PIECE_VALUES[Board::PAWN];
}

int32_t scoreMove(const MoveOrderingContext& ctx, const chess::Move& m, const MoveTraits& t,
                  bool isHashMove) noexcept {
    if (isHashMove) return HASH_MOVE_SCORE;

    if (t.isCapture) {
        // Provisionally a good capture: finalizeSee demotes a losing one later.
        const int16_t* history = ctx.runtime.captureHistory[ctx.usSide][m.to][t.victimType];
        return CAPTURE_BASE_SCORE + PIECE_VALUES[t.victimType] * 10
             + (t.isPromotion ? promotionGain(m.promotionType) : 0)
             + std::min<int32_t>(500, (history[0] + (history[1] >> 1)) / 20);
    }

    if (m.sameFromTo(ctx.killer0)) return KILLER_1_SCORE;
    if (m.sameFromTo(ctx.killer1)) return KILLER_2_SCORE;
    if (ctx.counterMove != 0 && ctx.counterMove == TT::Entry::encodeMove(m)) return COUNTER_MOVE_SCORE;
    if (t.isPromotion) return PROMOTION_BASE_SCORE + PIECE_VALUES[m.promotionType];

    int32_t score = std::clamp<int32_t>(ctx.runtime.history[ctx.usSide][m.from][m.to],
                                        HISTORY_SCORE_MIN, HISTORY_SCORE_MAX);
    if (ctx.contHistEntry != nullptr) {
        score += std::clamp<int32_t>(ctx.contHistEntry[contHistIndex(t.fromType, m.to)],
                                     HISTORY_SCORE_MIN, HISTORY_SCORE_MAX) / 2;
    }
    return score;
}

// Square and type of `side`'s least valuable attacker of sq (square 64: none).
struct Attacker {
    int square;
    int type;
};

Attacker leastValuableAttacker(const Board& b, int sq, uint64_t occ, int side) noexcept {
    uint64_t mask = b.bb[Board::PAWN][side] & occ & pieces::PAWN_ATTACKS[side ^ 1][sq];
    if (mask) return {std::countr_zero(mask), Board::PAWN};
    mask = b.bb[Board::KNIGHT][side] & occ & pieces::KNIGHT_ATTACKS[sq];
    if (mask) return {std::countr_zero(mask), Board::KNIGHT};
    // Slider rays shared by bishop/queen and rook/queen.
    const uint64_t bishopRays = pieces::getBishopAttacks(sq, occ);
    const uint64_t rookRays   = pieces::getRookAttacks(sq, occ);
    const uint64_t rays[4] = {bishopRays, rookRays, bishopRays | rookRays, pieces::KING_ATTACKS[sq]};
    for (int type = Board::BISHOP; type <= Board::KING; ++type) {
        mask = b.bb[type][side] & occ & rays[type - Board::BISHOP];
        if (mask) return {std::countr_zero(mask), type};
    }
    return {64, 0};
}

} // namespace

int32_t staticExchangeEvaluation(const Board& b, const chess::Move& m) noexcept {
    const int toSq = m.to;
    const int fromSq = m.from;
    int capturedType = b.get(toSq) & Board::MASK_PIECE_TYPE;
    int onTarget = b.get(fromSq) & Board::MASK_PIECE_TYPE;   // the piece left standing on toSq
    const bool isEp = (onTarget == Board::PAWN) && (m.to == b.getEnPassant());
    if (isEp) capturedType = Board::PAWN;

    constexpr int MAX_SEE_DEPTH = 16;
    int32_t gain[MAX_SEE_DEPTH];
    gain[0] = PIECE_VALUES[capturedType];
    // Only N..Q are real promotions; anything else must not index PIECE_VALUES.
    if (m.promotionType >= Board::KNIGHT && m.promotionType <= Board::QUEEN) {
        gain[0] += PIECE_VALUES[m.promotionType] - PIECE_VALUES[Board::PAWN];
        onTarget = m.promotionType;
    }

    uint64_t occ = b.getPiecesBitMap() ^ (1ULL << fromSq);
    if (isEp) occ ^= 1ULL << (chess::rank(fromSq) * 8 + chess::file(toSq));

    int depth = 1;
    for (int side = Board::colorToIndex(b.getActiveColor()) ^ 1; depth < MAX_SEE_DEPTH; side ^= 1, ++depth) {
        const auto [square, type] = leastValuableAttacker(b, toSq, occ, side);
        if (square == 64) break;
        gain[depth] = PIECE_VALUES[onTarget] - gain[depth - 1];
        occ ^= 1ULL << square;
        onTarget = type;
    }
    while (--depth > 0) {
        gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
    }
    return gain[0];
}

bool givesCheckAfterQuietMoveFast(const Board& b, const chess::Move& m,
                                  int fromPieceType, int oppKingSq, uint64_t occ) noexcept {
    const int usSide = Board::colorToIndex(b.getActiveColor());
    const uint64_t fromBit = 1ULL << m.from;
    const uint64_t toBit   = 1ULL << m.to;
    const uint64_t occAfter = (occ & ~fromBit) | toBit;
    // Our pieces of a type after the move.
    const auto mine = [&](int type) {
        const uint64_t bits = b.bb[type][usSide];
        return type == fromPieceType ? (bits & ~fromBit) | toBit : bits;
    };

    if (pieces::PAWN_ATTACKS[usSide ^ 1][oppKingSq] & mine(Board::PAWN)) return true;
    if (pieces::KNIGHT_ATTACKS[oppKingSq] & mine(Board::KNIGHT))         return true;

    const uint64_t queens       = mine(Board::QUEEN);
    const uint64_t rookQueens   = mine(Board::ROOK)   | queens;
    const uint64_t bishopQueens = mine(Board::BISHOP) | queens;
    if (rookQueens   && (pieces::getRookAttacks(oppKingSq, occAfter)   & rookQueens))   return true;
    return bishopQueens && (pieces::getBishopAttacks(oppKingSq, occAfter) & bishopQueens);
}

MovePicker sortLegalMoves(MoveList moves, int ply, const Board& b, const SearchRuntime& runtime,
                          uint16_t encodedHashMove, const chess::Move* previousMove,
                          bool* outHashMoveIsLegal, const int16_t* contHistEntry) noexcept {
    MovePicker picker;
    picker.moves = std::move(moves);
    picker.board = &b;

    const int promotionRank = Board::promotionRank(b.getActiveColor() == Board::WHITE);
    const chess::Square enPassant = b.getEnPassant();
    const MoveOrderingContext ctx{
        runtime, contHistEntry, Board::colorToIndex(b.getActiveColor()),
        runtime.killerMoves[ply][0], runtime.killerMoves[ply][1],
        (previousMove != nullptr) ? runtime.counterMoves[previousMove->from][previousMove->to] : uint16_t{0}
    };
    const chess::Move hashMove = TT::Entry::decodeMove(encodedHashMove);
    bool hashMoveFound = false;

    for (int i = 0; i < picker.moves.size; ++i) {
        const chess::Move& m = picker.moves[i];
        const MoveTraits t = moveTraits(b, m, enPassant, promotionRank);
        const bool isHashMove = (encodedHashMove != 0) && m == hashMove;
        hashMoveFound |= isHashMove;
        // SEE is deferred to the picker, so a move no cutoff reaches never pays
        // for it. Quiet pawn and king moves skip the hanging-quiet demotion: for
        // them it does not repay its SEE.
        const SeePending pending = isHashMove ? SeePending::Final
            : t.isCapture ? SeePending::Capture
            : (!t.isPromotion && t.fromType != Board::KING && t.fromType != Board::PAWN) ? SeePending::Quiet
            : SeePending::Final;
        picker.setScore(i, scoreMove(ctx, m, t, isHashMove));
        picker.seePending[i] = pending;
    }

    if (outHashMoveIsLegal != nullptr) *outHashMoveIsLegal = hashMoveFound;
    return picker;
}

MovePicker sortTacticalMoves(const MoveList& tacticalMoves, const Board& b,
                             int32_t standPat, int32_t alpha, int ply) noexcept {
    MovePicker picker;
    const int promotionRank = Board::promotionRank(b.getActiveColor() == Board::WHITE);
    const chess::Square enPassant = b.getEnPassant();
    const int32_t seeThreshold = (ply < 10) ? SEE_THRESHOLD_SHALLOW
                               : (ply < 20) ? SEE_THRESHOLD_MID
                                            : SEE_THRESHOLD_DEEP;
    int32_t npm = 0;
    for (int type = Board::KNIGHT; type <= Board::QUEEN; ++type) {
        npm += std::popcount(b.bb[type][0] | b.bb[type][1]) * PIECE_VALUES[type];
    }
    const int32_t evalPct = (npm > 5000) ? MATERIAL_TO_EVAL_PCT
                          : (npm > 3000) ? MATERIAL_TO_EVAL_PCT_LATE_MG
                                         : MATERIAL_TO_EVAL_PCT_ENDGAME;
    int n = 0;
    for (const chess::Move& m : tacticalMoves) {
        const MoveTraits t = moveTraits(b, m, enPassant, promotionRank);
        int32_t score = TACTICAL_PROMOTION_SCORE;
        if (t.isCapture) {
            const int32_t captured = PIECE_VALUES[t.victimType] + (t.isPromotion ? promotionGain(m.promotionType) : 0);
            // Delta pruning: not even the capture lifts the stand pat to alpha.
            if (standPat + materialToEval(captured, evalPct) + FUTILITY_MARGIN <= alpha) continue;
            const int32_t see = staticExchangeEvaluation(b, m);
            if (see < seeThreshold || standPat + materialToEval(see, evalPct) + MOVE_DELTA_MARGIN <= alpha) continue;
            score = CAPTURE_BASE_SCORE + see + PIECE_VALUES[t.victimType] * 10;
        } else if (!t.isPromotion) {
            continue;
        }
        picker.moves[n] = m;
        picker.setScore(n, score);
        picker.seePending[n++] = SeePending::Final;   // SEE already applied
    }
    picker.moves.size = n;
    return picker;
}

MoveList sortEvasionsForcingFirst(MoveList evasions, const Board& b) noexcept {
    const chess::Square enPassant = b.getEnPassant();
    const int promotionRank = Board::promotionRank(b.getActiveColor() == Board::WHITE);
    int write = 0;
    for (int i = 0; i < evasions.size; ++i) {
        const chess::Move& m = evasions[i];
        const bool forcing = b.get(m.to) != Board::EMPTY
            || ((b.get(m.from) & Board::MASK_PIECE_TYPE) == Board::PAWN
                && (chess::rank(m.to) == promotionRank || m.to == enPassant));
        // Rotating it down to `write` keeps both groups in generation order.
        if (forcing) std::rotate(&evasions[write++], &evasions[i], &evasions[i] + 1);
    }
    return evasions;
}

} // namespace engine::Sorter
