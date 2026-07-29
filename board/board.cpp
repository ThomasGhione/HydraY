#include <algorithm>
#include "board.hpp"
#include "../tt/zobrist.hpp"

namespace chess {


bool Board::move(Move move) noexcept {
    const uint8_t moving = get(move.from);
    if (moving == EMPTY || (moving & MASK_COLOR) != activeColor) [[unlikely]]
        return false;
    if (!isLegalPseudoMove(move.from, move.to, moving)) [[unlikely]]
        return false;
    MoveState st{};
    doMove(move, st);
    return true;
}

bool Board::isLegalPseudoMove(uint8_t fromIndex, uint8_t toIndex, uint8_t fromPiece) const noexcept {
    const uint8_t fromType = fromPiece & MASK_PIECE_TYPE;
    const uint8_t movingColor = fromPiece & MASK_COLOR;

    const uint8_t destPiece = get(toIndex);

    if (destPiece != EMPTY && (destPiece & MASK_COLOR) == movingColor) [[unlikely]]
        return false;

    const uint64_t toBit = Board::BIT_MASKS[toIndex];
    switch (fromType) {
        case PAWN: {
            const bool isWhite = (movingColor == WHITE);

            // Diagonal move (capture or en-passant)?
            if (pieces::PAWN_ATTACKS[colorToIndex(movingColor)][fromIndex] & toBit) {
                // En-passant: diagonal to empty square that matches EP target
                if (destPiece == EMPTY) {
                    if (isValidSquare(enPassant) && toIndex == enPassant) {
                        const int8_t epDir = isWhite ? 8 : -8;
                        const uint8_t capturedPawnIdx = toIndex + epDir;
                        return isKingSafeAfterMove(movingColor, fromIndex, toIndex, BIT_MASKS[capturedPawnIdx]);
                    }
                    return false; // diagonal to empty square but not en-passant
                }
                // Normal capture (destPiece is enemy)
                return isKingSafeAfterMove(movingColor, fromIndex, toIndex, toBit);
            }

            // Forward push: must land on valid push square and destination must be empty
            if (destPiece != EMPTY) [[unlikely]] return false;
            if (!(pieces::getPawnForwardPushes(fromIndex, isWhite, occupancy) & toBit)) return false;
            return isKingSafeAfterMove(movingColor, fromIndex, toIndex, 0ULL);
        }
        case KNIGHT: return pseudoMoveLegalByType<KNIGHT>(fromIndex, toIndex, toBit, movingColor, destPiece);
        case BISHOP: return pseudoMoveLegalByType<BISHOP>(fromIndex, toIndex, toBit, movingColor, destPiece);
        case ROOK:   return pseudoMoveLegalByType<ROOK>(fromIndex, toIndex, toBit, movingColor, destPiece);
        case QUEEN:  return pseudoMoveLegalByType<QUEEN>(fromIndex, toIndex, toBit, movingColor, destPiece);
        case KING: return isKingMoveLegal(fromIndex, toIndex, toBit, movingColor);
        default: return false;
    }
}

template<uint8_t PieceType>
inline bool Board::pseudoMoveLegalByType(uint8_t fromIndex, uint8_t toIndex, uint64_t toBit, uint8_t movingColor, uint8_t destPiece) const noexcept {
    if ((pieces::generateMovesByType<PieceType>(fromIndex, occupancy) & toBit) == 0ULL) [[unlikely]] return false;
    return isKingSafeAfterMove(movingColor, fromIndex, toIndex, (destPiece != EMPTY) ? toBit : 0ULL);
}

// ============================================
// HELPER FUNCTIONS FOR isLegalPseudoMove
// ============================================

[[nodiscard]] uint64_t Board::checkersTo(uint8_t color) const noexcept {
    const uint8_t side = colorToIndex(color);
    const uint64_t kingBB = kings_bb[side];
    if (!kingBB) [[unlikely]] return 0ULL;
    const uint8_t kingIndex = std::countr_zero(kingBB);
    const uint8_t oppSide = side ^ 1;

    uint64_t checkers = (pieces::PAWN_ATTACKERS_TO[oppSide][kingIndex] & pawns_bb[oppSide])
                      | (pieces::KNIGHT_ATTACKS[kingIndex] & knights_bb[oppSide])
                      | (pieces::KING_ATTACKS[kingIndex] & kings_bb[oppSide]);

    const uint64_t rookLike = rooks_bb[oppSide] | queens_bb[oppSide];
    if (rookLike) {
        checkers |= (pieces::getRookAttacks(kingIndex, occupancy) & rookLike);
    }

    const uint64_t bishopLike = bishops_bb[oppSide] | queens_bb[oppSide];
    if (bishopLike) {
        checkers |= (pieces::getBishopAttacks(kingIndex, occupancy) & bishopLike);
    }

    return checkers;
}

[[nodiscard]] bool Board::isKingMoveLegal(
    uint8_t fromIndex,
    uint8_t toIndex,
    uint64_t toBit,
    uint8_t movingColor
) const noexcept {
    const uint8_t oppColor = oppositeColor(movingColor);
    const int diff = static_cast<int>(toIndex) - fromIndex;

    // Castling moves are uniquely identified by a destination offset of +2 or -2.
    // Normal king moves have offsets of +/-1, +/-7, +/-8, +/-9, so they cannot clash.
    if (diff == 2 || diff == -2) [[unlikely]] {
        if (chess::file(fromIndex) != 4) return false;
        const bool isWhite = (movingColor == WHITE);
        const uint8_t expectedRank = isWhite ? 7 : 0;
        if (chess::rank(fromIndex) != expectedRank) return false;
        return canCastleGeneric(isWhite, fromIndex, diff == 2);
    }

    // Normal king move: one-step king attack and destination not attacked
    const uint64_t attacks = pieces::KING_ATTACKS[fromIndex];
    if ((attacks & toBit) == 0ULL) return false;
    if (isSquareAttacked(toIndex, oppColor, fromIndex)) return false;

    return true;
}

[[nodiscard]] bool Board::canCastleGeneric(
    bool isWhite,
    uint8_t fromIndex,
    bool isKingside
) const noexcept {
    const uint8_t side = isWhite ^ 1; // 0 for White, 1 for Black
    const uint8_t oppColor = isWhite ? BLACK : WHITE;
    
    // Check castling rights
    const uint8_t rightBit = (!isWhite << 1) | !isKingside;
    
    if ((castle & (1u << rightBit)) == 0u) return false;
    
    // Setup indices based on direction
    const int8_t direction = isKingside ? 1 : -1;
    const uint8_t sq1 = fromIndex + direction;
    const uint8_t sq2 = fromIndex + 2 * direction;
    const uint8_t rookIdx = isKingside ? (fromIndex + 3) : (fromIndex - 4);
    
    // Check empty squares (always check 2, for queenside check 3rd)
    if (get(sq1) != EMPTY || get(sq2) != EMPTY)
        return false;
    
    if (!isKingside && get(fromIndex - 3) != EMPTY)
        return false;
    
    // Check rook presence
    if ((rooks_bb[side] & Board::BIT_MASKS[rookIdx]) == 0ULL)
        return false;
    
    if (isSquareAttacked(fromIndex, oppColor)) return false;
    if (isSquareAttacked(sq1, oppColor)) return false;
    if (isSquareAttacked(sq2, oppColor)) return false;

    return true;
}

// ------------------------------------------------------------
// CHECK / CHECKMATE / STALEMATE UTILITIES
// ------------------------------------------------------------
// Returns true if square 'targetIndex' is attacked by 'byColor'
// Version that excludes a square from occupancy - useful for king moves
bool Board::isSquareAttacked(uint8_t targetIndex, uint8_t byColor, uint8_t excludeSquare) const noexcept {
    const uint64_t occMinus = excludeSquare < 64 ? (occupancy & ~Board::BIT_MASKS[excludeSquare]) : occupancy;
    const uint8_t side = colorToIndex(byColor);
    return isKingAttackedCustom(targetIndex, side, occMinus,
                                pawns_bb[side], knights_bb[side], bishops_bb[side],
                                rooks_bb[side], queens_bb[side], kings_bb[side]);
}

// Helper: check if king at kingSq is attacked using custom bitboards
// Used internally to avoid code duplication when simulating moves
bool Board::isKingAttackedCustom(uint8_t kingSq, uint8_t bySide, uint64_t occ,
                                 uint64_t pawns, uint64_t knights, uint64_t bishops,
                                 uint64_t rooks, uint64_t queens, uint64_t kings) noexcept {
    if (pieces::PAWN_ATTACKERS_TO[bySide][kingSq] & pawns) return true;
    if (pieces::KNIGHT_ATTACKS[kingSq] & knights) return true;
    if (pieces::KING_ATTACKS[kingSq] & kings) return true;
    
    const uint64_t rookLike = rooks | queens;
    if (rookLike && (pieces::getRookAttacks(kingSq, occ) & rookLike)) return true;

    const uint64_t bishopLike = bishops | queens;
    if (bishopLike && (pieces::getBishopAttacks(kingSq, occ) & bishopLike)) return true;
    
    return false;
}

__attribute__((hot))
bool Board::inCheck(uint8_t color) const noexcept {
    const uint8_t side = colorToIndex(color);
    const uint64_t kingBB = kings_bb[side];

    if (!kingBB) [[unlikely]] return false;
    const uint8_t kingSq = std::countr_zero(kingBB);
    const uint8_t bySide = side ^ 1;
    return isKingAttackedCustom(kingSq, bySide, occupancy,
                                pawns_bb[bySide], knights_bb[bySide], bishops_bb[bySide],
                                rooks_bb[bySide], queens_bb[bySide], kings_bb[bySide]);
}


template<uint8_t PieceType>
bool Board::hasLegalMovesForPieceType(
    uint64_t pieceBB,
    uint64_t ownOcc,
    uint64_t enemyOcc,
    uint8_t movingColor
) const noexcept {
    while (pieceBB) {
        const uint8_t from = std::countr_zero(pieceBB);
        pieceBB &= pieceBB - 1;

        uint64_t movesMask = pieces::generateMovesByType<PieceType>(from, occupancy) & ~ownOcc;
        while (movesMask) {
            const uint64_t toBit = movesMask & -movesMask;
            movesMask ^= toBit;
            const uint8_t to = std::countr_zero(toBit);
            const uint64_t capturedMask = toBit & enemyOcc;
            if (isKingSafeAfterMove(movingColor, from, to, capturedMask)) return true;
        }
    }
    return false;
}


bool Board::hasAnyLegalMove(uint8_t color) const noexcept {
    const int side = colorToIndex(color);
    const int oppSide = side ^ 1;
    
    const uint64_t ownOcc = pawns_bb[side] | knights_bb[side] | bishops_bb[side] |
                             rooks_bb[side] | queens_bb[side]  | kings_bb[side];
    const uint64_t enemyOcc = pawns_bb[oppSide] | knights_bb[oppSide] | bishops_bb[oppSide] |
                               rooks_bb[oppSide] | queens_bb[oppSide]  | kings_bb[oppSide];

    // --- KING MOVES ---
    const uint8_t king = std::countr_zero(kings_bb[side]);
    uint64_t moves = pieces::KING_ATTACKS[king] & ~ownOcc;
    while (moves) {
        const uint8_t to = std::countr_zero(moves);
        moves &= moves - 1;
        if (!isSquareAttacked(to, oppositeColor(color), king)) return true;
    }

    const uint64_t checkers = checkersTo(color);
    if (checkers) {
        // Double check: only king moves are legal, all rejected above.
        if ((checkers & (checkers - 1)) != 0ULL) return false;
    } else {
        constexpr uint8_t WHITE_KING_START = 60;  // e1
        constexpr uint8_t BLACK_KING_START = 4;   // e8
        const uint8_t eIndex = (side == 0) ? WHITE_KING_START : BLACK_KING_START;
        if (king == eIndex) {
            if (canCastleGeneric(side == 0, eIndex, true)) return true;
            if (canCastleGeneric(side == 0, eIndex, false)) return true;
        }
    }

    // --- NON-KING PIECES: skip isLegalPseudoMove, call isKingSafeAfterMove directly ---

    if (hasLegalMovesForPieceType<KNIGHT>(knights_bb[side], ownOcc, enemyOcc, color))
        return true;

    const bool isWhite = (side == 0);
    uint64_t pawns = pawns_bb[side];
    while (pawns) {
        const uint8_t from = std::countr_zero(pawns);
        pawns &= pawns - 1;

        uint64_t push = pieces::getPawnForwardPushes(from, isWhite, occupancy);
        while (push) {
            const uint8_t to = std::countr_zero(push);
            push &= push - 1;
            if (isKingSafeAfterMove(color, from, to, 0ULL)) return true;
        }

        uint64_t caps = pieces::PAWN_ATTACKS[side][from] & enemyOcc;
        while (caps) {
            const uint64_t capBit = caps & -caps;
            caps ^= capBit;
            const uint8_t to = std::countr_zero(capBit);
            if (isKingSafeAfterMove(color, from, to, capBit)) return true;
        }
    }

    if (hasLegalMovesForPieceType<BISHOP>(bishops_bb[side], ownOcc, enemyOcc, color))
        return true;

    if (hasLegalMovesForPieceType<ROOK>(rooks_bb[side], ownOcc, enemyOcc, color))
        return true;

    if (hasLegalMovesForPieceType<QUEEN>(queens_bb[side], ownOcc, enemyOcc, color))
        return true;

    return false;
}

void Board::recomputeHashAndEp() noexcept {
    currentHash = zobrist::computeHashKey(*this);
    epHashFile = 0xFF;
    if (zobrist::hasPseudoLegalEnPassantCapture(*this, getEnPassant()))
        epHashFile = chess::file(getEnPassant());
}

void Board::rebuildRepetitionHistory() noexcept {
    recomputeHashAndEp();
    historySize = 0;
    repetitionHistory[historySize++] = currentHash;
}

void Board::updateRepetitionAfterMove(bool resetHistory, MoveState& st) noexcept {
    if (resetHistory)
        historySize = 0;

    if (historySize >= repetitionHistory.size()) {
        std::memmove(repetitionHistory.data(), repetitionHistory.data() + 1, (repetitionHistory.size() - 1) * sizeof(uint64_t));
        historySize = static_cast<uint8_t>(repetitionHistory.size() - 1);
    }
    // Remember the slot we are about to overwrite so undoMove can restore it
    // (writeIndex == historySize here; on undo historySize-1 recovers it). This
    // is what keeps a reset-to-0 on an irreversible move from corrupting earlier
    // entries in the parent line.
    st.prevHistorySlotValue = repetitionHistory[historySize];
    repetitionHistory[historySize++] = currentHash;
}

int Board::countRepetitions() const noexcept {
    const uint64_t* const begin = repetitionHistory.data();
    return static_cast<int>(std::count(begin, begin + historySize, currentHash));
}

bool Board::hasInsufficientMaterialDraw() const noexcept {
    if (pawns_bb[0] || pawns_bb[1] || rooks_bb[0] || rooks_bb[1] || queens_bb[0] || queens_bb[1]) return false;
    const uint64_t wMinors = knights_bb[0] | bishops_bb[0];
    const uint64_t bMinors = knights_bb[1] | bishops_bb[1];
    return (std::popcount(wMinors) <= 1 && bMinors == 0ULL)
        || (std::popcount(bMinors) <= 1 && wMinors == 0ULL);
}

} // namespace chess
