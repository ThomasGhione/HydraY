#include "move_generator.hpp"

namespace engine::MoveGenerator {

namespace {

using chess::Board;

// Squares strictly between two aligned squares, 0 when they are not aligned.
constexpr std::array<std::array<uint64_t, 64>, 64> BETWEEN = [] {
    std::array<std::array<uint64_t, 64>, 64> lut{};
    for (int a = 0; a < 64; ++a)
        for (int b = 0; b < 64; ++b)
            for (const auto dirs : {pieces::ROOK_DIRS, pieces::BISHOP_DIRS})
                if (pieces::slidingAttacks(a, 0ULL, dirs) & (1ULL << b))
                    lut[a][b] = pieces::slidingAttacks(a, 1ULL << b, dirs) & pieces::slidingAttacks(b, 1ULL << a, dirs);
    return lut;
}();

// The side to move's bitboards, occupancy and en-passant square, gathered once.
template<bool IsWhite>
struct SideView {
    static constexpr uint8_t color = IsWhite ? Board::WHITE : Board::BLACK;
    static constexpr int side = IsWhite ? 0 : 1;
    uint64_t occ, pawns, knights, bishops, rooks, queens, kings, ownOcc, oppOcc;
    chess::Square enPassant;
    uint64_t enPassantBit;

    explicit SideView(const Board& b) noexcept
        : occ(b.getPiecesBitMap()),
          pawns(b.bb[Board::PAWN][side]), knights(b.bb[Board::KNIGHT][side]), bishops(b.bb[Board::BISHOP][side]),
          rooks(b.bb[Board::ROOK][side]), queens(b.bb[Board::QUEEN][side]), kings(b.bb[Board::KING][side]),
          ownOcc(b.sideOcc(side)), oppOcc(occ & ~ownOcc),
          enPassant(b.getEnPassant()),
          enPassantBit(chess::isValidSquare(enPassant) ? 1ULL << enPassant : 0ULL) {}
};

__attribute__((always_inline))
inline void appendPromotions(MoveList& moves, int from, int to) noexcept {
    // Queen first: keeps the pre-sort promo ordering the sorter relies on.
    for (const uint8_t p : {Board::QUEEN, Board::ROOK, Board::BISHOP, Board::KNIGHT}) moves.emplace_back(from, to, p);
}

// Pawn moves to `targets`. En passant takes two pieces off their squares, so it
// gets a full legality check instead of relying on the pin ray.
template<bool IsWhite>
inline void appendPawnTargets(const Board& b, MoveList& moves, int from, uint64_t targets,
                              chess::Square enPassant) noexcept {
    constexpr uint8_t pawnPiece = Board::PAWN | (IsWhite ? Board::WHITE : Board::BLACK);
    while (targets) {
        const int to = pieces::popLSB(targets);
        if (chess::isValidSquare(enPassant) && to == enPassant && chess::file(to) != chess::file(from)
            && !b.isLegalPseudoMove(from, to, pawnPiece)) continue;
        if (chess::rank(to) == Board::promotionRank(IsWhite)) appendPromotions(moves, from, to);
        else moves.emplace_back(from, to);
    }
}

// Pushes and captures; `evasionMask` is ~0 outside single check.
template<bool IsWhite>
inline void appendPawnMoves(const Board& b, MoveList& moves, const SideView<IsWhite>& v,
                            uint64_t evasionMask, uint64_t pinnedMask, const uint64_t* pinRays) noexcept {
    uint64_t pawns = v.pawns;
    while (pawns) {
        const int from = pieces::popLSB(pawns);
        const uint64_t ep = (pieces::PAWN_ATTACKS[v.side][from] & v.enPassantBit) ? v.enPassantBit : 0ULL;
        uint64_t mask = (pieces::getPawnForwardPushes(from, IsWhite, v.occ)
                         | (pieces::PAWN_ATTACKS[v.side][from] & v.oppOcc) | ep) & evasionMask;
        if (pinnedMask & (1ULL << from)) mask &= pinRays[from];
        appendPawnTargets<IsWhite>(b, moves, from, mask | ep, v.enPassant);
    }
}

// Captures, en passant and the pushes that promote.
template<bool IsWhite>
inline void appendPawnTactical(const Board& b, MoveList& moves, const SideView<IsWhite>& v,
                               uint64_t pinnedMask, const uint64_t* pinRays) noexcept {
    constexpr int push = IsWhite ? -8 : 8;
    uint64_t pawns = v.pawns;
    while (pawns) {
        const int from = pieces::popLSB(pawns);
        uint64_t attacks = pieces::PAWN_ATTACKS[v.side][from] & (v.oppOcc | v.enPassantBit);
        if (chess::rank(from) == (IsWhite ? 1 : 6) && !(v.occ & (1ULL << (from + push))))
            attacks |= 1ULL << (from + push);
        if (pinnedMask & (1ULL << from)) attacks &= pinRays[from];
        appendPawnTargets<IsWhite>(b, moves, from, attacks, v.enPassant);
    }
}

// Moves of one piece type to `targets`, a pinned piece kept on its pin ray.
template<bool HasPins, uint8_t PieceType>
__attribute__((always_inline))
inline void appendPieceMoves(MoveList& moves, uint64_t movers, uint64_t occ, uint64_t targets,
                             [[maybe_unused]] uint64_t pinnedMask,
                             [[maybe_unused]] const uint64_t* pinRays) noexcept {
    while (movers) {
        const int from = pieces::popLSB(movers);
        uint64_t mask = pieces::generateMovesByType<PieceType>(from, occ) & targets;
        if constexpr (HasPins) {
            if (pinnedMask & (1ULL << from)) mask &= pinRays[from];
        }
        while (mask) moves.emplace_back(from, pieces::popLSB(mask));
    }
}

template<bool IsWhite>
inline void appendAllPieceMoves(MoveList& moves, const SideView<IsWhite>& v, uint64_t targets,
                                uint64_t pinnedMask, const uint64_t* pinRays) noexcept {
    const auto all = [&]<bool HasPins>() {
        appendPieceMoves<HasPins, Board::KNIGHT>(moves, v.knights, v.occ, targets, pinnedMask, pinRays);
        appendPieceMoves<HasPins, Board::BISHOP>(moves, v.bishops, v.occ, targets, pinnedMask, pinRays);
        appendPieceMoves<HasPins, Board::ROOK>(moves, v.rooks, v.occ, targets, pinnedMask, pinRays);
        appendPieceMoves<HasPins, Board::QUEEN>(moves, v.queens, v.occ, targets, pinnedMask, pinRays);
    };
    if (pinnedMask) all.template operator()<true>();
    else            all.template operator()<false>();
}

// Every square the opponent attacks with `occ` as blockers. Built with our king
// taken off the board, so a slider's ray runs through the king's square exactly
// as Board::isSquareAttacked(to, them, kingSq) sees it for each king step.
template<bool IsWhite>
inline uint64_t enemyAttacks(const Board& b, uint64_t occ) noexcept {
    constexpr int them = IsWhite ? 1 : 0;
    constexpr uint64_t FILE_A = 0x0101010101010101ULL;
    constexpr uint64_t FILE_H = FILE_A << 7;
    // a8 = 0: a white pawn on sq attacks sq - 9 and sq - 7, a black one sq + 7 and sq + 9.
    const uint64_t p = b.bb[Board::PAWN][them];
    uint64_t att = IsWhite ? (((p << 7) & ~FILE_H) | ((p << 9) & ~FILE_A))
                           : (((p >> 9) & ~FILE_H) | ((p >> 7) & ~FILE_A));
    uint64_t knights = b.bb[Board::KNIGHT][them];
    while (knights) att |= pieces::KNIGHT_ATTACKS[pieces::popLSB(knights)];
    uint64_t diag = b.bb[Board::BISHOP][them] | b.bb[Board::QUEEN][them];
    while (diag) att |= pieces::getBishopAttacks(static_cast<uint8_t>(pieces::popLSB(diag)), occ);
    uint64_t line = b.bb[Board::ROOK][them] | b.bb[Board::QUEEN][them];
    while (line) att |= pieces::getRookAttacks(static_cast<uint8_t>(pieces::popLSB(line)), occ);
    if (b.bb[Board::KING][them]) att |= pieces::KING_ATTACKS[std::countr_zero(b.bb[Board::KING][them])];
    return att;
}

// Fills pinRays for our pinned pieces and returns their mask.
template<bool IsWhite>
uint64_t computePinRays(const Board& b, int kingSq, uint64_t pinRays[64]) noexcept {
    constexpr int us = IsWhite ? 0 : 1;
    constexpr int them = us ^ 1;
    const uint64_t ownOcc = b.sideOcc(us) ^ b.bb[Board::KING][us];
    const uint64_t rookLikeEnemy = b.bb[Board::ROOK][them] | b.bb[Board::QUEEN][them];
    const uint64_t bishopLikeEnemy = b.bb[Board::BISHOP][them] | b.bb[Board::QUEEN][them];
    // No enemy slider on a line through the king, even on an empty board: no pins.
    if (!(pieces::getRookAttacks(kingSq, 0ULL) & rookLikeEnemy) &&
        !(pieces::getBishopAttacks(kingSq, 0ULL) & bishopLikeEnemy)) {
        return 0ULL;
    }

    const uint64_t occWithoutOwn = b.getPiecesBitMap() ^ ownOcc;
    uint64_t pinnedMask = 0ULL;
    const auto addPins = [&](uint64_t pinners) noexcept {
        while (pinners) {
            const int pinnerSq = pieces::popLSB(pinners);
            const uint64_t blockers = BETWEEN[kingSq][pinnerSq] & ownOcc;
            if (blockers && !(blockers & (blockers - 1))) {
                pinnedMask |= blockers;
                pinRays[std::countr_zero(blockers)] = BETWEEN[kingSq][pinnerSq] | (1ULL << pinnerSq);
            }
        }
    };
    addPins(pieces::getRookAttacks(kingSq, occWithoutOwn) & rookLikeEnemy);
    addPins(pieces::getBishopAttacks(kingSq, occWithoutOwn) & bishopLikeEnemy);
    return pinnedMask;
}

template<bool IsWhite>
MoveList generateLegalMovesFor(const Board& b, uint64_t checkers) noexcept {
    MoveList moves;
    const SideView<IsWhite> v(b);
    const bool inCheck = (checkers != 0ULL);
    const bool inDoubleCheck = (checkers & (checkers - 1)) != 0ULL;
    const uint8_t kingPiece = Board::KING | v.color;
    const int kingFrom = std::countr_zero(v.kings);

    // Single check: capture the checker or block the ray (empty for contact and knight checks).
    const uint64_t evasionMask = (inCheck && !inDoubleCheck)
        ? (checkers | BETWEEN[kingFrom][std::countr_zero(checkers)])
        : ~0ULL;

    uint64_t mask = pieces::KING_ATTACKS[kingFrom] & ~v.ownOcc;
    if (mask & (mask - 1)) {
        // Two or more king steps: one attack map is cheaper than a scan per step.
        mask &= ~enemyAttacks<IsWhite>(b, v.occ ^ v.kings);
        while (mask) moves.emplace_back(kingFrom, pieces::popLSB(mask));
    } else if (mask) {
        const int to = std::countr_zero(mask);
        if (b.isLegalPseudoMove(kingFrom, to, kingPiece)) moves.emplace_back(kingFrom, to);
    }

    if (!inCheck) { // castling: illegal when in check.
        const int f = chess::file(kingFrom);
        if (f <= 5 && b.isLegalPseudoMove(kingFrom, kingFrom + 2, kingPiece)) moves.emplace_back(kingFrom, kingFrom + 2);
        if (f >= 2 && b.isLegalPseudoMove(kingFrom, kingFrom - 2, kingPiece)) moves.emplace_back(kingFrom, kingFrom - 2);
    }
    if (inDoubleCheck) return moves; // only the king can move

    // Read only where pinnedMask has a bit, so left uninitialised.
    uint64_t pinRays[64];
    const uint64_t pinnedMask = (v.ownOcc ^ v.kings) ? computePinRays<IsWhite>(b, kingFrom, pinRays) : 0ULL;
    appendPawnMoves<IsWhite>(b, moves, v, evasionMask, pinnedMask, pinRays);
    appendAllPieceMoves<IsWhite>(moves, v, ~v.ownOcc & evasionMask, pinnedMask, pinRays);
    return moves;
}

template<bool IsWhite>
MoveList generateTacticalMovesFor(const Board& b) noexcept {
    MoveList moves;
    const SideView<IsWhite> v(b);
    const int kingSq = std::countr_zero(v.kings);

    uint64_t pinRays[64];
    const uint64_t pinnedMask = (v.ownOcc ^ v.kings) ? computePinRays<IsWhite>(b, kingSq, pinRays) : 0ULL;
    appendPawnTactical<IsWhite>(b, moves, v, pinnedMask, pinRays);
    appendAllPieceMoves<IsWhite>(moves, v, v.oppOcc, pinnedMask, pinRays);

    // A king cannot be pinned, but it may not capture a defended piece.
    uint64_t kingCaptures = pieces::KING_ATTACKS[kingSq] & v.oppOcc;
    while (kingCaptures) {
        const int to = pieces::popLSB(kingCaptures);
        if (b.isLegalPseudoMove(kingSq, to, Board::KING | v.color)) moves.emplace_back(kingSq, to);
    }
    return moves;
}

} // namespace

MoveList generateLegalMoves(const Board& b, uint64_t checkers) noexcept {
    return (b.getActiveColor() == Board::WHITE) ? generateLegalMovesFor<true>(b, checkers)
                                                : generateLegalMovesFor<false>(b, checkers);
}

MoveList generateTacticalMoves(const Board& b) noexcept {
    return (b.getActiveColor() == Board::WHITE) ? generateTacticalMovesFor<true>(b)
                                                : generateTacticalMovesFor<false>(b);
}

MovePicker generateQSearchEvasions(const Board& b, uint64_t checkers) noexcept {
    MovePicker picker;
    picker.moves = Sorter::sortEvasionsForcingFirst(generateLegalMoves(b, checkers), b);
    // Equal scores keep the forcing-first order through nextMove(); no deferred SEE.
    for (int i = 0; i < picker.moves.size; ++i) {
        picker.setScore(i, 0);
        picker.seePending[i] = SeePending::Final;
    }
    return picker;
}

MovePicker generateQSearchTacticalMoves(const Board& b, int32_t standPat, int32_t alpha, int ply) noexcept {
    return Sorter::sortTacticalMoves(generateTacticalMoves(b), b, standPat, alpha, ply);
}

} // namespace engine::MoveGenerator
