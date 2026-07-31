#include "move_generator.hpp"

#include "../inl/bitboard_helpers.inl"

namespace engine {

namespace {

__attribute__((always_inline))
inline void appendPromotionSetByIndex(MoveList& moves, int from, int to) noexcept {
    // Queen first: keeps the pre-sort promo ordering the sorter relies on.
    for (const uint8_t p : {chess::Board::QUEEN, chess::Board::ROOK,
                            chess::Board::BISHOP, chess::Board::KNIGHT}) {
        moves.emplace_back(from, to, p);
    }
}

constexpr uint64_t computeBetweenExclusiveConstexpr(int from, int to) noexcept {
    if (from == to) return 0ULL;

    const int fromFile = chess::file(from);
    const int fromRank = chess::rank(from);
    const int toFile = chess::file(to);
    const int toRank = chess::rank(to);
    const int df = toFile - fromFile;
    const int dr = toRank - fromRank;

    int stepFile = 0;
    int stepRank = 0;
    if (df == 0) {
        stepRank = (dr > 0) ? 1 : -1;
    } else if (dr == 0) {
        stepFile = (df > 0) ? 1 : -1;
    } else if (std::abs(df) == std::abs(dr)) {
        stepFile = (df > 0) ? 1 : -1;
        stepRank = (dr > 0) ? 1 : -1;
    } else {
        return 0ULL;
    }

    uint64_t mask = 0ULL;
    int f = fromFile + stepFile;
    int r = fromRank + stepRank;
    while (f != toFile || r != toRank) {
        const int sq = (r << 3) | f;
        mask |= chess::Board::BIT_MASKS[sq];
        f += stepFile;
        r += stepRank;
    }

    return mask & ~chess::Board::BIT_MASKS[to];
}

inline constexpr std::array<std::array<uint64_t, 64>, 64> BETWEEN_EXCLUSIVE_LUT = [] {
    std::array<std::array<uint64_t, 64>, 64> lut{};
    for (int from = 0; from < 64; ++from) {
        for (int to = 0; to < 64; ++to) {
            lut[from][to] = computeBetweenExclusiveConstexpr(from, to);
        }
    }
    return lut;
}();

template<bool HasPins, uint8_t PieceType>
__attribute__((always_inline))
inline void appendNonPawnTacticalNoChecks(
    MoveList& moves,
    uint64_t pieceBB,
    uint64_t occ,
    uint64_t oppOcc,
    [[maybe_unused]] uint64_t pinnedMask,
    [[maybe_unused]] const uint64_t pinRayBySquare[64]) noexcept {

    while (pieceBB) {
        const int from = engine::popLSB(pieceBB);

        uint64_t attacks = pieces::generateMovesByType<PieceType>(from, occ) & oppOcc;
        if constexpr (HasPins) {
            const uint64_t fromBit = chess::Board::BIT_MASKS[from];
            if (pinnedMask & fromBit) {
                attacks &= pinRayBySquare[from];
            }
        }

        while (attacks) {
            moves.emplace_back(from, engine::popLSB(attacks));
        }
    }
}

template<bool IsWhite>
__attribute__((always_inline))
inline void appendPawnTacticalNoChecks(
    const chess::Board& b,
    MoveList& moves,
    uint64_t pawnBB,
    uint64_t occ,
    uint64_t oppOcc,
    uint64_t enPassantBit,
    int enPassantIndex,
    uint64_t pinnedMask,
    const uint64_t pinRayBySquare[64],
    uint8_t pawnPiece) noexcept {
    constexpr int Side = IsWhite ? 0 : 1;
    constexpr int promotionRank = IsWhite ? 0 : 7;
    constexpr int prePromotionRank = IsWhite ? 1 : 6;
    constexpr int pawnPushDelta = IsWhite ? -8 : 8;

    while (pawnBB) {
        const int from = engine::popLSB(pawnBB);
        uint64_t attacks = pieces::PAWN_ATTACKS[Side][from] & (oppOcc | enPassantBit);

        // Add forward push moves for promotion
        if (chess::rank(from) == prePromotionRank) {
            const int frontSq = from + pawnPushDelta;
            if ((chess::Board::BIT_MASKS[frontSq] & occ) == 0) {
                attacks |= chess::Board::BIT_MASKS[frontSq];
            }
        }

        if (pinnedMask & chess::Board::BIT_MASKS[from]) {
            attacks &= pinRayBySquare[from];
        }

        while (attacks) {
            const int to = engine::popLSB(attacks);
            const bool isEnPassant = enPassantBit != 0ULL && (to == enPassantIndex);

            if (isEnPassant && !b.isLegalPseudoMove(from, to, pawnPiece)) continue;

            if (chess::rank(to) == promotionRank) {
                appendPromotionSetByIndex(moves, from, to);
            } else {
                moves.emplace_back(from, to);
            }
        }
    }
}

// Side-to-move view gathered once: the six piece bitboards, own/enemy
// occupancy and the en-passant square/bit. Shared by the legal and tactical
// generators, which used to open with the same ~15-line preamble.
template<bool IsWhite>
struct SideView {
    static constexpr uint8_t color = IsWhite ? chess::Board::WHITE : chess::Board::BLACK;
    static constexpr int side = IsWhite ? 0 : 1;
    uint64_t occ, pawns, knights, bishops, rooks, queens, kings, ownOcc, oppOcc;
    chess::Square enPassant;
    bool hasEnPassant;
    uint64_t enPassantBit;

    explicit SideView(const chess::Board& b) noexcept
        : occ(b.getPiecesBitMap()),
          pawns(b.pawns_bb[side]), knights(b.knights_bb[side]), bishops(b.bishops_bb[side]),
          rooks(b.rooks_bb[side]), queens(b.queens_bb[side]), kings(b.kings_bb[side]),
          ownOcc(pawns | knights | bishops | rooks | queens | kings),
          oppOcc(occ & ~ownOcc),
          enPassant(b.getEnPassant()),
          hasEnPassant(chess::isValidSquare(enPassant)),
          enPassantBit(hasEnPassant ? chess::Board::BIT_MASKS[enPassant] : 0ULL) {}
};

} // namespace

MoveList MoveGenerator::generateLegalMoves(const chess::Board& b,
                                                               bool knownNotInCheck) noexcept {
    CheckContext check;
    check.known = knownNotInCheck; // known not-in-check (checkers 0); otherwise scanned
    return (b.getActiveColor() == chess::Board::WHITE)
        ? generateLegalMovesFor<true>(b, check)
        : generateLegalMovesFor<false>(b, check);
}

MoveList MoveGenerator::generateLegalEvasions(
    const chess::Board& b,
    uint64_t checkers) noexcept {
    const CheckContext check{true, checkers};
    return (b.getActiveColor() == chess::Board::WHITE)
        ? generateLegalMovesFor<true>(b, check)
        : generateLegalMovesFor<false>(b, check);
}

template<bool IsWhite>
MoveList MoveGenerator::generateLegalMovesFor(const chess::Board& b, CheckContext check) noexcept {
    // Macro-step 1: gather the side-to-move view (bitboards, occupancy, EP).
    MoveList moves;
    const SideView<IsWhite> v(b);

    // Callers that already computed the checkers bitboard skip the scan.
    const uint64_t checkers = check.known ? check.checkers : b.checkersTo(v.color);
    const bool inCheck = (checkers != 0ULL);
    const bool inDoubleCheck = (checkers & (checkers - 1)) != 0ULL;
    const bool singleCheck = inCheck && !inDoubleCheck;
    const uint8_t kingPiece = chess::Board::KING | v.color;

    const int kingFrom = std::countr_zero(v.kings);
    const chess::Square kingFromC = static_cast<uint8_t>(kingFrom);

    // Macro-step 2: single-check evasion targets — capture the checker or
    // block the ray (BETWEEN is empty for contact and knight checks).
    const uint64_t evasionMask = singleCheck
        ? (checkers | BETWEEN_EXCLUSIVE_LUT[kingFrom][std::countr_zero(checkers)])
        : ~0ULL;

    uint64_t mask = pieces::KING_ATTACKS[kingFrom] & ~v.ownOcc;
    while (mask) {
        const int to = engine::popLSB(mask);
        if (b.isLegalPseudoMove(kingFrom, to, kingPiece)) {
            moves.emplace_back(kingFrom, to);
        }
    }

    if (!inCheck) { // castling: illegal when in check.
        const int f = chess::file(kingFrom);
        if (f <= 5 && b.isLegalPseudoMove(kingFrom, kingFrom + 2, kingPiece)) {
            moves.emplace_back(kingFrom, kingFrom + 2);
        }
        if (f >= 2 && b.isLegalPseudoMove(kingFrom, kingFrom - 2, kingPiece)) {
            moves.emplace_back(kingFrom, kingFrom - 2);
        }
    }
    // In double-check only king moves are legal.
    if (inDoubleCheck) return moves;

    // Macro-step 4: pin rays restrict non-king mobility. Not zero-initialised
    // for performance — only read where pinnedMask has a bit set.
    std::array<uint64_t, 64> pinRayBySquare;
    const uint64_t pinnedMask = (v.pawns | v.knights | v.bishops | v.rooks | v.queens)
        ? computePinRays<IsWhite>(b, kingFromC, pinRayBySquare.data()) : 0ULL;

    appendPawnPseudoLegalMoves<IsWhite>(
        b, moves, v.pawns, v.occ, v.oppOcc, v.enPassantBit, v.enPassant, evasionMask, pinnedMask, pinRayBySquare.data());

    if (singleCheck) {
        if (pinnedMask)
            emitAllNonPawnLegal<true, true>(moves, v.knights, v.bishops, v.rooks, v.queens, v.occ, v.ownOcc, evasionMask, pinnedMask, pinRayBySquare.data());
        else
            emitAllNonPawnLegal<false, true>(moves, v.knights, v.bishops, v.rooks, v.queens, v.occ, v.ownOcc, evasionMask, pinnedMask, pinRayBySquare.data());
    } else {
        if (pinnedMask)
            emitAllNonPawnLegal<true, false>(moves, v.knights, v.bishops, v.rooks, v.queens, v.occ, v.ownOcc, 0ULL, pinnedMask, pinRayBySquare.data());
        else
            emitAllNonPawnLegal<false, false>(moves, v.knights, v.bishops, v.rooks, v.queens, v.occ, v.ownOcc, 0ULL, pinnedMask, pinRayBySquare.data());
    }

    return moves;
}

MoveList MoveGenerator::generateTacticalMoves(const chess::Board& b) noexcept {
    return (b.getActiveColor() == chess::Board::WHITE)
        ? generateTacticalMovesFor<true>(b)
        : generateTacticalMovesFor<false>(b);
}

template<bool IsWhite>
MoveList MoveGenerator::generateTacticalMovesFor(const chess::Board& b) noexcept {
    MoveList moves;
    const SideView<IsWhite> v(b);

    const uint8_t pawnPiece = chess::Board::PAWN | v.color;
    const int kingIndex = std::countr_zero(v.kings);

    // NOTE: per performance, don't zero-initialize this array.
    std::array<uint64_t, 64> pinRayBySquare;
    const uint64_t pinnedMask = (v.pawns | v.knights | v.bishops | v.rooks | v.queens)
        ? computePinRays<IsWhite>(b, static_cast<uint8_t>(kingIndex), pinRayBySquare.data()) : 0ULL;

    const int enPassantIndex = v.hasEnPassant ? v.enPassant : 0;
    appendPawnTacticalNoChecks<IsWhite>(
        b, moves, v.pawns, v.occ, v.oppOcc, v.enPassantBit, enPassantIndex,
        pinnedMask, pinRayBySquare.data(), pawnPiece);

    if (pinnedMask)
        emitAllNonPawnTactical<true>(moves, v.knights, v.bishops, v.rooks, v.queens, v.occ, v.oppOcc, pinnedMask, pinRayBySquare.data());
    else
        emitAllNonPawnTactical<false>(moves, v.knights, v.bishops, v.rooks, v.queens, v.occ, v.oppOcc, pinnedMask, pinRayBySquare.data());

    // King captures must be verified for legality. The non-king tactical paths
    // above already filter pinned movers via pinRayBySquare, but king moves
    // cannot be pinned — they are illegal when stepping into a square attacked
    // by another enemy piece. Without this filter Probcut would happily search
    // illegal "king captures own attacker" positions and contaminate the TT
    // with mate-like scores derived from the opponent's immediate king
    // capture. The cost is paid only for enemy pieces adjacent to our king
    // (KING_ATTACKS[ksq] & oppOcc, typically 0–1 squares).
    uint64_t kingAttacks = pieces::KING_ATTACKS[kingIndex] & v.oppOcc;
    if (kingAttacks) {
        constexpr uint8_t kingPiece = chess::Board::KING | v.color;
        do {
            const int to = engine::popLSB(kingAttacks);
            if (b.isLegalPseudoMove(kingIndex, to, kingPiece)) {
                moves.emplace_back(kingIndex, to);
            }
        } while (kingAttacks);
    }
    return moves;
}

engine::MovePicker MoveGenerator::generateQSearchEvasions(
    const chess::Board& b,
    uint64_t checkers) noexcept {
    MoveList evasions = generateLegalEvasions(b, checkers);
    if (evasions.is_empty()) return engine::MovePicker{};

    engine::MovePicker data;
    data.moves = engine::Sorter::sortEvasionsForcingFirst(std::move(evasions), b);
    data.currentIndex = 0;
    // Evasions keep their forcing-first partition order: equal scores so
    // nextMove() is order-preserving, and no deferred SEE. MovePicker no longer
    // zero-inits, so fill the live prefix explicitly.
    for (int i = 0; i < data.moves.size; ++i) {
        data.scores[i]     = 0;
        data.seePending[i] = engine::SeePending::Final;
    }
    return data;
}

engine::MovePicker MoveGenerator::generateQSearchTacticalMoves(
    const chess::Board& b,
    int32_t standPat,
    int32_t alpha,
    int ply) noexcept {
    MoveList tacticalMoves = generateTacticalMoves(b);
    if (tacticalMoves.is_empty()) return engine::MovePicker{};
    return engine::Sorter::sortTacticalMoves(tacticalMoves, b, standPat, alpha, ply);
}

template<bool IsWhite>
__attribute__((always_inline))
inline void MoveGenerator::appendPawnPseudoLegalMoves(
    const chess::Board& b, MoveList& moves, uint64_t pawns,
    uint64_t occ, uint64_t oppOcc, uint64_t enPassantBit, chess::Square enPassant,
    uint64_t evasionMask, uint64_t pinnedMask, const uint64_t pinRayBySquare[64]) noexcept {
    constexpr int side = IsWhite ? 0 : 1;

    while (pawns) {
        const int from = engine::popLSB(pawns);
        uint64_t mask = pieces::getPawnForwardPushes(from, IsWhite, occ);
        const uint64_t epCandidate = (pieces::PAWN_ATTACKS[side][from] & enPassantBit) ? enPassantBit : 0ULL;
        mask |= (pieces::PAWN_ATTACKS[side][from] & oppOcc) | epCandidate;
        mask &= evasionMask; // ~0ULL outside single check, so a no-op there
        if (pinnedMask & chess::Board::BIT_MASKS[from]) mask &= pinRayBySquare[from];
        // Keep EP candidate for legality check because EP changes occupancy on two squares.
        mask |= epCandidate;
        addPawnMovesFromMask<IsWhite>(b, moves, from, mask, enPassant);
    }
}

template<bool IsWhite>
void MoveGenerator::addPawnMovesFromMask(const chess::Board& b, MoveList& moves,
                                         int from, uint64_t mask,
                                         chess::Square enPassant) noexcept {
    if (!mask) [[unlikely]] return;

    constexpr uint8_t pawnPiece = chess::Board::PAWN | (IsWhite ? chess::Board::WHITE : chess::Board::BLACK);
    constexpr int promotionRank = IsWhite ? 0 : 7;
    const int fromFile = chess::file(from);

    while (mask) {
        const int to = engine::popLSB(mask);
        const bool isEnPassant = chess::isValidSquare(enPassant)
            && (to == enPassant)
            && (chess::file(to) != fromFile);

        if (isEnPassant && !b.isLegalPseudoMove(from, to, pawnPiece)) {
            continue;
        }

        if (chess::rank(to) == promotionRank) {
            appendPromotionSetByIndex(moves, from, to);
        } else {
            moves.emplace_back(from, to);
        }
    }
}

template<bool IsWhite>
uint64_t MoveGenerator::computePinRays(const chess::Board& b, chess::Square kingPos,
                                       uint64_t pinRays[64]) noexcept {
    constexpr int us = IsWhite ? 0 : 1;
    constexpr int them = us ^ 1;
    const int kingSq = kingPos;

    const uint64_t ownOcc = b.pawns_bb[us] | b.knights_bb[us] | b.bishops_bb[us] | b.rooks_bb[us] | b.queens_bb[us];

    const uint64_t rookLikeEnemy = b.rooks_bb[them] | b.queens_bb[them];
    const uint64_t bishopLikeEnemy = b.bishops_bb[them] | b.queens_bb[them];
    if ((rookLikeEnemy | bishopLikeEnemy) == 0ULL) {
        return 0ULL;
    }

    // Geometric fast-path: if no enemy slider is geometrically aligned with our king
    // (even with an empty board), we don't need to compute real occupancy rays.
    if (!(pieces::getRookAttacks(kingSq, 0ULL) & rookLikeEnemy) &&
        !(pieces::getBishopAttacks(kingSq, 0ULL) & bishopLikeEnemy)) {
        return 0ULL;
    }

    const uint64_t occWithoutOwn = b.getPiecesBitMap() ^ ownOcc;
    uint64_t rookPinners = pieces::getRookAttacks(kingSq, occWithoutOwn) & rookLikeEnemy;
    uint64_t bishopPinners = pieces::getBishopAttacks(kingSq, occWithoutOwn) & bishopLikeEnemy;

    uint64_t pinnedMask = 0ULL;
    const auto processPinners = [&](uint64_t pinners) noexcept {
        while (pinners) {
            const int pinnerSq = engine::popLSB(pinners);
            const uint64_t between = BETWEEN_EXCLUSIVE_LUT[kingSq][pinnerSq];
            const uint64_t blockers = between & ownOcc;
            if (blockers && ((blockers & (blockers - 1)) == 0ULL)) {
                const int pinnedSq = std::countr_zero(blockers);
                pinnedMask |= blockers;
                pinRays[pinnedSq] = between | chess::Board::BIT_MASKS[pinnerSq];
            }
        }
    };
    processPinners(rookPinners);
    processPinners(bishopPinners);
    return pinnedMask;
}

template<bool HasPins, bool InCheck, uint8_t PieceType>
void MoveGenerator::generateNonPawnLegalMoves(
    MoveList& moves,
    uint64_t bb,
    uint64_t occ,
    uint64_t ownOcc,
    [[maybe_unused]] uint64_t evasionMask,
    [[maybe_unused]] uint64_t pinnedMask,
    [[maybe_unused]] const uint64_t pinRayBySquare[64]) noexcept {

    while (bb) {
        const int from = engine::popLSB(bb);
        uint64_t mask = pieces::generateMovesByType<PieceType>(from, occ) & ~ownOcc;
        if constexpr (InCheck) {
            mask &= evasionMask;
        }
        if constexpr (HasPins) {
            if (pinnedMask & chess::Board::BIT_MASKS[from]) mask &= pinRayBySquare[from];
        }

        while (mask) {
            moves.emplace_back(from, engine::popLSB(mask));
        }
    }
}

template<bool HasPins, bool InCheck>
void MoveGenerator::emitAllNonPawnLegal(
    MoveList& moves,
    uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
    uint64_t occ, uint64_t ownOcc, uint64_t evasionMask,
    uint64_t pinnedMask, const uint64_t pinRayBySquare[64]) noexcept {
    generateNonPawnLegalMoves<HasPins, InCheck, chess::Board::KNIGHT>(moves, knights, occ, ownOcc, evasionMask, pinnedMask, pinRayBySquare);
    generateNonPawnLegalMoves<HasPins, InCheck, chess::Board::BISHOP>(moves, bishops, occ, ownOcc, evasionMask, pinnedMask, pinRayBySquare);
    generateNonPawnLegalMoves<HasPins, InCheck, chess::Board::ROOK>(moves, rooks, occ, ownOcc, evasionMask, pinnedMask, pinRayBySquare);
    generateNonPawnLegalMoves<HasPins, InCheck, chess::Board::QUEEN>(moves, queens, occ, ownOcc, evasionMask, pinnedMask, pinRayBySquare);
}

template<bool HasPins>
void MoveGenerator::emitAllNonPawnTactical(
    MoveList& moves,
    uint64_t knights, uint64_t bishops, uint64_t rooks, uint64_t queens,
    uint64_t occ, uint64_t oppOcc,
    uint64_t pinnedMask, const uint64_t pinRayBySquare[64]) noexcept {
    appendNonPawnTacticalNoChecks<HasPins, chess::Board::KNIGHT>(moves, knights, occ, oppOcc, pinnedMask, pinRayBySquare);
    appendNonPawnTacticalNoChecks<HasPins, chess::Board::BISHOP>(moves, bishops, occ, oppOcc, pinnedMask, pinRayBySquare);
    appendNonPawnTacticalNoChecks<HasPins, chess::Board::ROOK>(moves, rooks, occ, oppOcc, pinnedMask, pinRayBySquare);
    appendNonPawnTacticalNoChecks<HasPins, chess::Board::QUEEN>(moves, queens, occ, oppOcc, pinnedMask, pinRayBySquare);
}

} // namespace engine
