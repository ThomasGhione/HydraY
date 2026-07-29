#include "board.hpp"
#include "../tt/zobrist.hpp"


namespace chess {
// ------------------------------------------------------------
// INCREMENTAL DO/UNDO MOVE (NO LEGALITY CHECKS)
// ------------------------------------------------------------
__attribute__((hot))
void Board::doMove(const Move& m, MoveState& st) noexcept {
    const uint8_t fromIndex = m.from;
    const uint8_t toIndex   = m.to;

    const uint8_t moving      = get(fromIndex);
    const uint8_t movingType  = moving & MASK_PIECE_TYPE;
    const uint8_t movingColor = moving & MASK_COLOR;
    const uint8_t destBefore  = get(toIndex);

    prepareMoveState(st, moving, destBefore);
    st.moveKind = classifyMoveKind(movingType, movingColor, fromIndex, toIndex, destBefore, st.prevEnPassant);

    uint64_t newHash = currentHash;
    if (st.prevEpHashFile < 8) {
        newHash ^= zobrist::TABLES.enPassant[st.prevEpHashFile];
    }
    newHash ^= zobrist::TABLES.pieces[moving][fromIndex];

    enPassant = NO_SQUARE;

    const MoveKind kind = st.moveKind;
    switch (kind) {
        case MoveKind::Capture:
            doMoveByKind<MoveKind::Capture>(st, moving, movingType, movingColor, destBefore,
                                            fromIndex, toIndex, m.promotionType);
            if (destBefore != EMPTY) {
                newHash ^= zobrist::TABLES.pieces[destBefore][toIndex];
            }
            break;
        case MoveKind::DoublePawnPush:
            doMoveByKind<MoveKind::DoublePawnPush>(st, moving, movingType, movingColor, destBefore,
                                                   fromIndex, toIndex, m.promotionType);
            break;
        case MoveKind::EnPassant:
            doMoveByKind<MoveKind::EnPassant>(st, moving, movingType, movingColor, destBefore,
                                              fromIndex, toIndex, m.promotionType);
            newHash ^= zobrist::TABLES.pieces[st.capturedPiece][st.enPassantCapturedIndex];
            break;
        case MoveKind::Castling:
            doMoveByKind<MoveKind::Castling>(st, moving, movingType, movingColor, destBefore,
                                             fromIndex, toIndex, m.promotionType);
            {
                const uint8_t rook = get(st.rookToIndex);
                newHash ^= zobrist::TABLES.pieces[rook][st.rookFromIndex];
                newHash ^= zobrist::TABLES.pieces[rook][st.rookToIndex];
            }
            break;
        case MoveKind::PromotionQuiet:
            doMoveByKind<MoveKind::PromotionQuiet>(st, moving, movingType, movingColor, destBefore,
                                                   fromIndex, toIndex, m.promotionType);
            {
                const uint8_t promotedPiece = static_cast<uint8_t>(st.promotionPieceType | movingColor);
                newHash ^= zobrist::TABLES.pieces[promotedPiece][toIndex];
            }
            break;
        case MoveKind::PromotionCapture:
            doMoveByKind<MoveKind::PromotionCapture>(st, moving, movingType, movingColor, destBefore,
                                                     fromIndex, toIndex, m.promotionType);
            newHash ^= zobrist::TABLES.pieces[destBefore][toIndex];
            {
                const uint8_t promotedPiece = static_cast<uint8_t>(st.promotionPieceType | movingColor);
                newHash ^= zobrist::TABLES.pieces[promotedPiece][toIndex];
            }
            break;
        case MoveKind::Quiet:
        default:
            doMoveByKind<MoveKind::Quiet>(st, moving, movingType, movingColor, destBefore,
                                          fromIndex, toIndex, m.promotionType);
            break;
    }
    if (!isPromotionKind(kind)) {
        newHash ^= zobrist::TABLES.pieces[moving][toIndex];
    }

    const bool resetHistory = (movingType == PAWN || destBefore != EMPTY);
    if (resetHistory) {
        halfMoveClock = 0;
    } else if (halfMoveClock < 255) {
        ++halfMoveClock;
    }
    if (activeColor == BLACK && fullMoveClock < 255) {
        ++fullMoveClock;
    }
    activeColor = oppositeColor(activeColor);
    newHash ^= zobrist::TABLES.sideToMove;

    if (castle != st.prevCastle) {
        newHash ^= zobrist::TABLES.castling[st.prevCastle];
        newHash ^= zobrist::TABLES.castling[castle];
    }
    uint8_t newEpHashFile = 0xFF;
    if (isValidSquare(enPassant) && zobrist::hasPseudoLegalEnPassantCapture(*this, enPassant)) {
        newEpHashFile = chess::file(enPassant);
        newHash ^= zobrist::TABLES.enPassant[newEpHashFile];
    }
    epHashFile = newEpHashFile;
    currentHash = newHash;

    updateRepetitionAfterMove(resetHistory, st);
}

__attribute__((hot))
void Board::undoMove(const Move& m, const MoveState& st) noexcept {
    const uint8_t fromIndex = m.from;
    const uint8_t toIndex   = m.to;

    uint8_t pieceOnTo = get(toIndex);

    switch (st.moveKind) {
        case MoveKind::Capture:
            undoMoveByKind<MoveKind::Capture>(st, pieceOnTo, fromIndex, toIndex);
            break;
        case MoveKind::DoublePawnPush:
            undoMoveByKind<MoveKind::DoublePawnPush>(st, pieceOnTo, fromIndex, toIndex);
            break;
        case MoveKind::EnPassant:
            undoMoveByKind<MoveKind::EnPassant>(st, pieceOnTo, fromIndex, toIndex);
            break;
        case MoveKind::Castling:
            undoMoveByKind<MoveKind::Castling>(st, pieceOnTo, fromIndex, toIndex);
            break;
        case MoveKind::PromotionQuiet:
            undoMoveByKind<MoveKind::PromotionQuiet>(st, pieceOnTo, fromIndex, toIndex);
            break;
        case MoveKind::PromotionCapture:
            undoMoveByKind<MoveKind::PromotionCapture>(st, pieceOnTo, fromIndex, toIndex);
            break;
        case MoveKind::Quiet:
        default:
            undoMoveByKind<MoveKind::Quiet>(st, pieceOnTo, fromIndex, toIndex);
            break;
    }

    restoreState(st);
}

__attribute__((hot))
void Board::doNullMove(MoveState& st) noexcept {
    prepareNullMoveState(st);
    uint64_t newHash = currentHash;
    if (st.prevEpHashFile < 8) {
        newHash ^= zobrist::TABLES.enPassant[st.prevEpHashFile];
    }

    enPassant = NO_SQUARE;
    epHashFile = 0xFF;

    if (halfMoveClock < 255) {
        ++halfMoveClock;
    }
    if (activeColor == BLACK && fullMoveClock < 255) {
        ++fullMoveClock;
    }
    activeColor = oppositeColor(activeColor);
    newHash ^= zobrist::TABLES.sideToMove;
    currentHash = newHash;
    // Null move is a search artifact and must not enter repetition history.
    // Otherwise threefold detection in subtree can be polluted.
}

__attribute__((hot))
void Board::undoNullMove(const MoveState& st) noexcept {
    restoreState(st);
}
} // namespace chess
