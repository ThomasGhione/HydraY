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
// gets a full legality check instead of relying on the pin ray. CanPromote is
// false for the ~98% of pawns not on the 7th rank, keeping promo setup out of
// their hot loop.
template<bool IsWhite, bool CanPromote>
__attribute__((always_inline))
inline void appendPawnTargets(const Board& b, MoveList& moves, int from, uint64_t targets,
                              chess::Square enPassant) noexcept {
    constexpr uint8_t pawnPiece = Board::PAWN | (IsWhite ? Board::WHITE : Board::BLACK);
    while (targets) {
        const int to = pieces::popLSB(targets);
        if constexpr (!CanPromote) {
            if (chess::isValidSquare(enPassant) && to == enPassant && chess::file(to) != chess::file(from)
                && !b.isLegalPseudoMove(from, to, pawnPiece)) continue;
            moves.emplace_back(from, to);
        } else {
            appendPromotions(moves, from, to);
        }
    }
}

// Pushes and captures, or with Tactical only the captures and the pushes that
// promote; `evasionMask` is ~0 outside single check. En passant skips the pin
// ray: appendPawnTargets checks it in full.
template<bool IsWhite, bool Tactical>
inline void appendPawnMoves(const Board& b, MoveList& moves, const SideView<IsWhite>& v,
                            uint64_t evasionMask, uint64_t pinnedMask, const uint64_t* pinRays) noexcept {
    constexpr int push = IsWhite ? -8 : 8;
    constexpr int prePromoRank = IsWhite ? 1 : 6;
    uint64_t pawns = v.pawns;
    while (pawns) {
        const int from = pieces::popLSB(pawns);
        const uint64_t attacks = pieces::PAWN_ATTACKS[v.side][from];
        uint64_t mask = attacks & v.oppOcc;
        const bool isPrePromo = chess::rank(from) == prePromoRank;
        if constexpr (Tactical) {
            if (isPrePromo && !(v.occ & (1ULL << (from + push))))
                mask |= 1ULL << (from + push);
        } else {
            mask = (mask | pieces::getPawnForwardPushes(from, IsWhite, v.occ)) & evasionMask;
        }
        if (pinnedMask & (1ULL << from)) mask &= pinRays[from];
        const uint64_t targets = mask | (attacks & v.enPassantBit);
        if (!targets) continue;
        if (isPrePromo) [[unlikely]] appendPawnTargets<IsWhite, true>(b, moves, from, targets, v.enPassant);
        else                         appendPawnTargets<IsWhite, false>(b, moves, from, targets, v.enPassant);
    }
}

// Moves of one piece type to `targets`, a pinned piece kept on its pin ray.
// Writes through a local pointer cursor so moves.size is updated once per call.
template<bool HasPins, uint8_t PieceType>
__attribute__((always_inline))
inline void appendPieceMoves(chess::Move*& out, uint64_t movers, uint64_t occ, uint64_t targets,
                             [[maybe_unused]] uint64_t pinnedMask,
                             [[maybe_unused]] const uint64_t* pinRays) noexcept {
    while (movers) {
        const int from = pieces::popLSB(movers);
        uint64_t mask = pieces::generateMovesByType<PieceType>(from, occ) & targets;
        if constexpr (HasPins) {
            if (pinnedMask & (1ULL << from)) mask &= pinRays[from];
        }
        while (mask) {
            *out++ = chess::Move{static_cast<chess::Square>(from),
                                 static_cast<chess::Square>(pieces::popLSB(mask)), 0};
        }
    }
}

template<bool IsWhite>
inline void appendAllPieceMoves(MoveList& moves, const SideView<IsWhite>& v, uint64_t targets,
                                uint64_t pinnedMask, const uint64_t* pinRays) noexcept {
    chess::Move* out = moves.begin() + moves.size;
    const auto all = [&]<bool HasPins>() {
        appendPieceMoves<HasPins, Board::KNIGHT>(out, v.knights, v.occ, targets, pinnedMask, pinRays);
        appendPieceMoves<HasPins, Board::BISHOP>(out, v.bishops, v.occ, targets, pinnedMask, pinRays);
        appendPieceMoves<HasPins, Board::ROOK>(out, v.rooks, v.occ, targets, pinnedMask, pinRays);
        appendPieceMoves<HasPins, Board::QUEEN>(out, v.queens, v.occ, targets, pinnedMask, pinRays);
    };
    if (pinnedMask) all.template operator()<true>();
    else            all.template operator()<false>();
    moves.size = static_cast<int>(out - moves.begin());
}

// Fills pinRays for our pinned pieces and returns their mask.
template<bool IsWhite>
uint64_t computePinRays(const Board& b, int kingSq, uint64_t ownNonKing, uint64_t occ, uint64_t pinRays[64]) noexcept {
    constexpr int them = IsWhite ? 1 : 0;
    const uint64_t rookLikeEnemy = b.bb[Board::ROOK][them] | b.bb[Board::QUEEN][them];
    const uint64_t bishopLikeEnemy = b.bb[Board::BISHOP][them] | b.bb[Board::QUEEN][them];
    // No enemy slider on a line through the king, even on an empty board: no pins.
    if (!(pieces::Rook::EMPTY_ATTACKS[kingSq] & rookLikeEnemy) &&
        !(pieces::Bishop::EMPTY_ATTACKS[kingSq] & bishopLikeEnemy)) {
        return 0ULL;
    }

    const uint64_t occWithoutOwn = occ ^ ownNonKing;
    uint64_t pinnedMask = 0ULL;
    const auto addPins = [&](uint64_t pinners) noexcept {
        while (pinners) {
            const int pinnerSq = pieces::popLSB(pinners);
            const uint64_t blockers = BETWEEN[kingSq][pinnerSq] & ownNonKing;
            if (blockers && !(blockers & (blockers - 1))) {
                pinnedMask |= blockers;
                pinRays[std::countr_zero(blockers)] = BETWEEN[kingSq][pinnerSq] | (1ULL << pinnerSq);
            }
        }
    };
    addPins(pieces::Rook::attacks(static_cast<uint8_t>(kingSq), occWithoutOwn) & rookLikeEnemy);
    addPins(pieces::Bishop::attacks(static_cast<uint8_t>(kingSq), occWithoutOwn) & bishopLikeEnemy);
    return pinnedMask;
}

// All legal moves, or with Tactical (never in check) the captures and promotions.
template<bool IsWhite, bool Tactical>
void generateMovesFor(const Board& b, uint64_t checkers, MoveList& moves) noexcept {
    const SideView<IsWhite> v(b);
    const bool inCheck = (checkers != 0ULL);
    const bool inDoubleCheck = (checkers & (checkers - 1)) != 0ULL;
    const uint8_t kingPiece = Board::KING | v.color;
    const int kingFrom = std::countr_zero(v.kings);

    // Single check: capture the checker or block the ray (empty for contact and knight checks).
    const uint64_t evasionMask = (inCheck && !inDoubleCheck)
        ? (checkers | BETWEEN[kingFrom][std::countr_zero(checkers)])
        : ~0ULL;

    if constexpr (!Tactical) {
        uint64_t mask = pieces::KING_ATTACKS[kingFrom] & ~v.ownOcc;
        if (mask & (mask - 1)) {
            // Two or more king steps: one attack map is cheaper than a scan per
            // step. Our king is off the board, so a slider's ray runs on past it.
            mask &= ~b.attackedBy(v.side ^ 1, v.occ ^ v.kings);
            while (mask) moves.emplace_back(kingFrom, pieces::popLSB(mask));
        } else if (mask) {
            const int to = std::countr_zero(mask);
            if (b.isLegalPseudoMove(kingFrom, to, kingPiece)) moves.emplace_back(kingFrom, to);
        }

        if (!inCheck) { // castling: illegal when in check.
            if (const uint8_t cr = b.castlingRights() & static_cast<uint8_t>(3u << (2 * v.side))) {
                for (const int right : {2 * v.side, 2 * v.side + 1})
                    if ((cr & (1u << right)) && b.canCastle(right))
                        moves.emplace_back(Board::CASTLINGS[right].king, Board::CASTLINGS[right].kingTo);
            }
        }
        if (inDoubleCheck) return; // only the king can move
    }

    // Read only where pinnedMask has a bit, so left uninitialised.
    uint64_t pinRays[64];
    const uint64_t ownNonKing = v.ownOcc ^ v.kings;
    const uint64_t pinnedMask = ownNonKing ? computePinRays<IsWhite>(b, kingFrom, ownNonKing, v.occ, pinRays) : 0ULL;
    appendPawnMoves<IsWhite, Tactical>(b, moves, v, evasionMask, pinnedMask, pinRays);
    appendAllPieceMoves<IsWhite>(moves, v, (Tactical ? v.oppOcc : ~v.ownOcc) & evasionMask, pinnedMask, pinRays);

    if constexpr (Tactical) {
        // A king cannot be pinned, but it may not capture a defended piece.
        uint64_t kingCaptures = pieces::KING_ATTACKS[kingFrom] & v.oppOcc;
        while (kingCaptures) {
            const int to = pieces::popLSB(kingCaptures);
            if (b.isLegalPseudoMove(kingFrom, to, kingPiece)) moves.emplace_back(kingFrom, to);
        }
    }
}

} // namespace

void generateLegalMoves(const Board& b, uint64_t checkers, MoveList& moves) noexcept {
    if (b.getActiveColor() == Board::WHITE) generateMovesFor<true, false>(b, checkers, moves);
    else                                    generateMovesFor<false, false>(b, checkers, moves);
}

MoveList generateLegalMoves(const Board& b, uint64_t checkers) noexcept {
    MoveList moves;
    generateLegalMoves(b, checkers, moves);
    return moves;
}

MoveList generateTacticalMoves(const Board& b) noexcept {
    MoveList moves;
    if (b.getActiveColor() == Board::WHITE) generateMovesFor<true, true>(b, 0, moves);
    else                                    generateMovesFor<false, true>(b, 0, moves);
    return moves;
}

MovePicker generateQSearchEvasions(const Board& b, uint64_t checkers) noexcept {
    MovePicker picker;
    generateLegalMoves(b, checkers, picker.moves);
    Sorter::sortEvasionsForcingFirst(picker.moves, b);
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
