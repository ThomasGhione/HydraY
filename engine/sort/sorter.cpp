#include "sorter.hpp"

#include <algorithm>
#include <bit>

#include "../engine.hpp"

namespace engine::Sorter {

namespace {

using chess::Board;

// Pack (from, to) into a single 16-bit key so killer / counter / hash move
// checks compile to a single 16-bit compare instead of two byte loads+branches.
__attribute__((always_inline))
constexpr uint16_t packFromTo(uint8_t from, uint8_t to) noexcept {
    return static_cast<uint16_t>(static_cast<uint16_t>(from) | (static_cast<uint16_t>(to) << 8));
}

__attribute__((always_inline))
inline int32_t clampHist(int16_t raw) noexcept {
    int32_t v = raw;
    v = (v < HISTORY_SCORE_MIN) ? HISTORY_SCORE_MIN : v;
    v = (v > HISTORY_SCORE_MAX) ? HISTORY_SCORE_MAX : v;
    return v;
}

// The scorer's node-invariant inputs, resolved once per node.
struct MoveOrderingContext {
    const int16_t (*history)[2][64][64];
    uint64_t threats;
    const int16_t (*captureHistory)[7][CAPTURE_HISTORY_SLOTS];
    const int16_t* contHistEntry;
    uint16_t killer0Ft;
    uint16_t killer1Ft;
    uint16_t counterFt;      // 0xFFFF = none (never matches any legal move)
    uint8_t  counterPromo;
};

// Move::promotionType is already the piece type; 0 (no promotion) adds nothing.
constexpr int32_t promotionGain(uint8_t promotionType) noexcept {
    return (promotionType == Board::EMPTY) ? 0 : PIECE_VALUES[promotionType] - PIECE_VALUES[Board::PAWN];
}

template<bool HasContHist>
__attribute__((always_inline))
inline int32_t scoreMove(const MoveOrderingContext& ctx, const chess::Move& m, uint16_t ft,
                         const MoveTraits& t) noexcept {
    if (t.isCapture) {
        // Provisionally a good capture: finalizeSee demotes a losing one later.
        const int16_t* history = ctx.captureHistory[m.to][t.victimType];
        return CAPTURE_BASE_SCORE + PIECE_VALUES[t.victimType] * 10
             + (t.isPromotion ? promotionGain(m.promotionType) : 0)
             + std::min<int32_t>(500, (history[0] + (history[1] >> 1)) / 20);
    }

    if (ft == ctx.killer0Ft) return KILLER_1_SCORE;
    if (ft == ctx.killer1Ft) return KILLER_2_SCORE;
    if (ft == ctx.counterFt && m.promotionType == ctx.counterPromo) return COUNTER_MOVE_SCORE;
    if (t.isPromotion) return PROMOTION_BASE_SCORE + PIECE_VALUES[m.promotionType];

    int32_t score = clampHist(ctx.history[(ctx.threats >> m.from) & 1][(ctx.threats >> m.to) & 1][m.from][m.to]);
    if constexpr (HasContHist) {
        score += clampHist(ctx.contHistEntry[contHistIndex(t.fromType, m.to)]) / 2;
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
    const uint64_t bishopRays = pieces::Bishop::attacks(sq, occ);
    const uint64_t rookRays   = pieces::Rook::attacks(sq, occ);
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
    // Our pieces as they stand after the move.
    const auto after = [&](int type) {
        const uint64_t bits = b.bb[type][usSide];
        return type == fromPieceType ? (bits & ~fromBit) | toBit : bits;
    };
    return Board::attackersOf(static_cast<uint8_t>(oppKingSq), usSide, (occ & ~fromBit) | toBit, after) != 0;
}

MovePicker sortLegalMoves(MoveList moves, int ply, const Board& b, const SearchRuntime& runtime,
                          uint16_t encodedHashMove, const chess::Move* previousMove,
                          bool* outHashMoveIsLegal, const int16_t* contHistEntry) noexcept {
    MovePicker picker;
    picker.moves = std::move(moves);
    scoreLegalMoves(picker, ply, b, runtime, encodedHashMove, previousMove, outHashMoveIsLegal, contHistEntry,
                    b.attackedBy(Board::colorToIndex(b.getActiveColor()) ^ 1));
    return picker;
}

template<bool HasContHist>
static void scoreLegalMovesImpl(MovePicker& picker, const Board& b, const MoveOrderingContext& ctx,
                                uint16_t hashFt, uint8_t hashPromo, int promotionRank,
                                chess::Square enPassant, bool* outHashMoveIsLegal) noexcept {
    bool hashMoveFound = false;
    const int n = picker.moves.size;

    for (int i = 0; i < n; ++i) {
        const chess::Move& m = picker.moves[i];
        const uint16_t ft = packFromTo(m.from, m.to);
        if (ft == hashFt && m.promotionType == hashPromo) [[unlikely]] {
            hashMoveFound = true;
            picker.setScore(i, HASH_MOVE_SCORE);
            picker.seePending[i] = SeePending::Final;
            continue;
        }
        const MoveTraits t = moveTraits(b, m, enPassant, promotionRank);
        // SEE is deferred to the picker, so a move no cutoff reaches never pays
        // for it. Quiet pawn and king moves skip the hanging-quiet demotion: for
        // them it does not repay its SEE.
        const SeePending pending = t.isCapture ? SeePending::Capture
            : (!t.isPromotion && t.fromType != Board::KING && t.fromType != Board::PAWN) ? SeePending::Quiet
            : SeePending::Final;
        picker.setScore(i, scoreMove<HasContHist>(ctx, m, ft, t));
        picker.seePending[i] = pending;
    }

    if (outHashMoveIsLegal != nullptr) *outHashMoveIsLegal = hashMoveFound;
}

void scoreLegalMoves(MovePicker& picker, int ply, const Board& b, const SearchRuntime& runtime,
                     uint16_t encodedHashMove, const chess::Move* previousMove,
                     bool* outHashMoveIsLegal, const int16_t* contHistEntry, uint64_t threats) noexcept {
    picker.board = &b;

    const int usSide = Board::colorToIndex(b.getActiveColor());
    const int promotionRank = Board::promotionRank(b.getActiveColor() == Board::WHITE);
    const chess::Square enPassant = b.getEnPassant();

    const uint16_t rawCounter = (previousMove != nullptr)
        ? runtime.counterMoves[previousMove->from][previousMove->to] : uint16_t{0};
    uint16_t counterFt = 0xFFFFu;
    uint8_t counterPromo = 0;
    if (rawCounter != 0) {
        const chess::Move cm = TT::Entry::decodeMove(rawCounter);
        counterFt = packFromTo(cm.from, cm.to);
        counterPromo = cm.promotionType;
    }

    const chess::Move& k0 = runtime.killerMoves[ply][0];
    const chess::Move& k1 = runtime.killerMoves[ply][1];
    const MoveOrderingContext ctx{
        runtime.history[usSide],
        threats,
        runtime.captureHistory[usSide],
        contHistEntry,
        packFromTo(k0.from, k0.to),
        packFromTo(k1.from, k1.to),
        counterFt,
        counterPromo
    };

    uint16_t hashFt = 0xFFFFu;
    uint8_t hashPromo = 0;
    if (encodedHashMove != 0) {
        const chess::Move hm = TT::Entry::decodeMove(encodedHashMove);
        hashFt = packFromTo(hm.from, hm.to);
        hashPromo = hm.promotionType;
    }

    if (contHistEntry != nullptr) {
        scoreLegalMovesImpl<true>(picker, b, ctx, hashFt, hashPromo, promotionRank, enPassant, outHashMoveIsLegal);
    } else {
        scoreLegalMovesImpl<false>(picker, b, ctx, hashFt, hashPromo, promotionRank, enPassant, outHashMoveIsLegal);
    }
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
    const int32_t evalPct = (npm > 5000) ? QS_EVAL_PCT_MG
                          : (npm > 3000) ? QS_EVAL_PCT_LATE_MG
                                         : QS_EVAL_PCT_ENDGAME;
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

void sortEvasionsForcingFirst(MoveList& evasions, const Board& b) noexcept {
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
}

} // namespace engine::Sorter
