#include <algorithm>
#include "board.hpp"
#include "../tt/zobrist.hpp"

namespace chess {

// Slider attack tables, filled once before main.
[[maybe_unused]] static const bool sliderTablesReady = (pieces::initMagicBitboards(), true);

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

    const uint64_t toBit = (1ULL << toIndex);
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
                        return isKingSafeAfterMove(movingColor, fromIndex, toIndex, (1ULL << capturedPawnIdx));
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


[[nodiscard]] uint64_t Board::checkersTo(uint8_t color) const noexcept {
    const uint8_t side = colorToIndex(color);
    const uint64_t kingBB = bb[KING][side];
    if (!kingBB) [[unlikely]] return 0ULL;
    const uint8_t kingIndex = std::countr_zero(kingBB);
    const uint8_t oppSide = side ^ 1;

    uint64_t checkers = (pieces::PAWN_ATTACKS[side][kingIndex] & bb[PAWN][oppSide])
                      | (pieces::KNIGHT_ATTACKS[kingIndex] & bb[KNIGHT][oppSide])
                      | (pieces::KING_ATTACKS[kingIndex] & bb[KING][oppSide]);

    const uint64_t rookLike = bb[ROOK][oppSide] | bb[QUEEN][oppSide];
    if (rookLike) {
        checkers |= (pieces::getRookAttacks(kingIndex, occupancy) & rookLike);
    }

    const uint64_t bishopLike = bb[BISHOP][oppSide] | bb[QUEEN][oppSide];
    if (bishopLike) {
        checkers |= (pieces::getBishopAttacks(kingIndex, occupancy) & bishopLike);
    }

    return checkers;
}

[[nodiscard]] bool Board::isKingMoveLegal(uint8_t fromIndex, uint8_t toIndex, uint64_t toBit,
                                          uint8_t movingColor) const noexcept {
    const uint8_t oppColor = oppositeColor(movingColor);
    const int diff = static_cast<int>(toIndex) - fromIndex;
    // Only castling moves the king two files: a one-step offset is 1, 7, 8 or 9.
    if (diff == 2 || diff == -2) [[unlikely]] {
        const bool isWhite = (movingColor == WHITE), kingside = (diff == 2);
        const int dir = kingside ? 1 : -1;
        const uint8_t sq1 = fromIndex + dir, sq2 = fromIndex + 2 * dir;
        const uint8_t rookIdx = kingside ? fromIndex + 3 : fromIndex - 4;
        return fromIndex == (isWhite ? 60 : 4)
            && ((castle >> ((!isWhite << 1) | !kingside)) & 1)
            && get(sq1) == EMPTY && get(sq2) == EMPTY && (kingside || get(fromIndex - 3) == EMPTY)
            && (bb[ROOK][!isWhite] & (1ULL << rookIdx))
            && !isSquareAttacked(fromIndex, oppColor) && !isSquareAttacked(sq1, oppColor)
            && !isSquareAttacked(sq2, oppColor);
    }
    return (pieces::KING_ATTACKS[fromIndex] & toBit) && !isSquareAttacked(toIndex, oppColor, fromIndex);
}

// Whether byColor attacks targetIndex, with excludeSquare (if < 64) lifted off
// the board, as a moving king needs.
bool Board::isSquareAttacked(uint8_t targetIndex, uint8_t byColor, uint8_t excludeSquare) const noexcept {
    const uint64_t occMinus = excludeSquare < 64 ? (occupancy & ~(1ULL << excludeSquare)) : occupancy;
    return attackedBy(targetIndex, colorToIndex(byColor), occMinus);
}

bool Board::attackedBy(uint8_t sq, uint8_t bySide, uint64_t occ, uint64_t keep) const noexcept {
    if (pieces::PAWN_ATTACKS[bySide ^ 1][sq] & bb[PAWN][bySide] & keep) return true;
    if (pieces::KNIGHT_ATTACKS[sq] & bb[KNIGHT][bySide] & keep) return true;
    if (pieces::KING_ATTACKS[sq] & bb[KING][bySide] & keep) return true;

    const uint64_t rookLike = (bb[ROOK][bySide] | bb[QUEEN][bySide]) & keep;
    if (rookLike && (pieces::getRookAttacks(sq, occ) & rookLike)) return true;

    const uint64_t bishopLike = (bb[BISHOP][bySide] | bb[QUEEN][bySide]) & keep;
    return bishopLike && (pieces::getBishopAttacks(sq, occ) & bishopLike);
}

__attribute__((hot))
bool Board::inCheck(uint8_t color) const noexcept {
    const uint8_t side = colorToIndex(color);
    if (!bb[KING][side]) [[unlikely]] return false;
    return attackedBy(std::countr_zero(bb[KING][side]), side ^ 1, occupancy);
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
    pliesFromNull = 255;
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

// Occurrences of the current position, itself included. Only the same side to
// move can match, and every ply (null moves too) takes one slot, so the scan
// steps by two. It stops at the last null move: no game can repeat a position
// across one, the same bound Stockfish applies (pliesFromNull).
int Board::countRepetitions() const noexcept {
    const int last = static_cast<int>(historySize) - 1;
    const int stop = std::max(0, last - static_cast<int>(pliesFromNull));
    int count = 0;
    for (int i = last; i >= stop; i -= 2) {
        count += (repetitionHistory[static_cast<size_t>(i)] == currentHash);
    }
    return count;
}

bool Board::hasInsufficientMaterialDraw() const noexcept {
    if (bb[PAWN][0] | bb[PAWN][1] | bb[ROOK][0] | bb[ROOK][1] | bb[QUEEN][0] | bb[QUEEN][1]) return false;
    const uint64_t wMinors = bb[KNIGHT][0] | bb[BISHOP][0];
    const uint64_t bMinors = bb[KNIGHT][1] | bb[BISHOP][1];
    return (std::popcount(wMinors) <= 1 && bMinors == 0ULL)
        || (std::popcount(bMinors) <= 1 && wMinors == 0ULL);
}

} // namespace chess
