// --------------------------
// API Helpers (engineapi)
// --------------------------

__attribute__((always_inline))
inline constexpr bool Board::isCaptureKind(MoveKind kind) noexcept {
    return kind == MoveKind::Capture || kind == MoveKind::PromotionCapture;
}

__attribute__((always_inline))
inline constexpr bool Board::isPromotionKind(MoveKind kind) noexcept {
    return kind == MoveKind::PromotionQuiet || kind == MoveKind::PromotionCapture;
}

inline Board::MoveKind Board::classifyMoveKind(
    uint8_t movingType,
    uint8_t movingColor,
    uint8_t fromIndex,
    uint8_t toIndex,
    uint8_t destBefore,
    const Square& prevEnPassant
) noexcept {
    if (movingType == KING) {
        if (chess::rank(fromIndex) == chess::rank(toIndex)) {
            const int df = chess::file(toIndex) - chess::file(fromIndex);
            if (df == 2 || df == -2) {
                return MoveKind::Castling;
            }
        }
        return (destBefore != EMPTY) ? MoveKind::Capture : MoveKind::Quiet;
    }

    if (movingType != PAWN) {
        return (destBefore != EMPTY) ? MoveKind::Capture : MoveKind::Quiet;
    }

    if (chess::file(fromIndex) != chess::file(toIndex)
        && destBefore == EMPTY
        && isValidSquare(prevEnPassant)
        && toIndex == prevEnPassant) {
        return MoveKind::EnPassant;
    }

    const uint8_t toRank = chess::rank(toIndex);
    if (toRank == promotionRank(movingColor == WHITE)) {
        return (destBefore != EMPTY) ? MoveKind::PromotionCapture : MoveKind::PromotionQuiet;
    }

    const int dr = toRank - chess::rank(fromIndex);
    if (dr == 2 || dr == -2) {
        return MoveKind::DoublePawnPush;
    }

    return (destBefore != EMPTY) ? MoveKind::Capture : MoveKind::Quiet;
}

// Missing/garbage promotion type defaults to queen (same fallback the old
// char-based normalization applied to unrecognized input).
__attribute__((always_inline))
inline uint8_t Board::normalizePromotionType(uint8_t promoType) noexcept {
    if (promoType >= KNIGHT && promoType <= QUEEN) [[likely]]
        return promoType;
    return QUEEN;
}

// This function assumes the caller has already validated that the piece being promoted is a pawn
// and that `promo` is a normalized promotion piece type.
__attribute__((always_inline))
inline void Board::promoteUnchecked(uint8_t atIndex, uint8_t pawnPiece, uint8_t promo) noexcept {
    const uint8_t newPiece = static_cast<uint8_t>(promo | (pawnPiece & MASK_COLOR));
    removePieceFromBB(pawnPiece, atIndex);
    addPieceToBB(newPiece, atIndex);
    set(atIndex, static_cast<piece_id>(newPiece));
}

__attribute__((always_inline))
inline void Board::snapshotState(MoveState& st) const noexcept {
    st.prevHalfMoveClock = halfMoveClock;
    st.prevFullMoveClock = fullMoveClock;
    st.prevEnPassant     = enPassant;
    st.prevEpHashFile    = epHashFile;
    st.prevCastle        = castle;
    st.prevHasMoved      = hasMoved;
    st.prevHistorySize   = historySize;
    st.prevHistoryHead   = currentHash;
    // Default that keeps undo a no-op for moves that do not push a history slot
    // (null moves); doMove overwrites this via updateRepetitionAfterMove with the
    // value of the exact slot it clobbers. historySize >= 1 always.
    st.prevHistorySlotValue = repetitionHistory[historySize - 1];
}

__attribute__((always_inline))
inline void Board::prepareMoveState(MoveState& st, uint8_t moving, uint8_t destBefore) const noexcept {
    snapshotState(st);
    st.capturedPiece = destBefore;
    st.fromPiece = moving;
    st.promotionPieceType = 0;
    st.enPassantCapturedIndex = 0;
    st.rookFromIndex = 0;
    st.rookToIndex = 0;
}

__attribute__((always_inline))
inline void Board::prepareNullMoveState(MoveState& st) const noexcept {
    snapshotState(st);
    st.moveKind = MoveKind::Quiet;
    st.capturedPiece = EMPTY;
    st.fromPiece = EMPTY;
    st.promotionPieceType = 0;
    st.enPassantCapturedIndex = 0;
    st.rookFromIndex = 0;
    st.rookToIndex = 0;
}

__attribute__((always_inline))
inline void Board::restoreState(const MoveState& st) noexcept {
    activeColor   = oppositeColor(activeColor);
    halfMoveClock = st.prevHalfMoveClock;
    fullMoveClock = st.prevFullMoveClock;
    enPassant     = st.prevEnPassant;
    epHashFile    = st.prevEpHashFile;
    castle        = st.prevCastle;
    hasMoved      = st.prevHasMoved;
    // Restore the one repetitionHistory slot doMove overwrote before shrinking
    // historySize back (writeIndex == current historySize - 1).
    repetitionHistory[historySize - 1] = st.prevHistorySlotValue;
    historySize   = st.prevHistorySize;
    currentHash   = st.prevHistoryHead;
}

__attribute__((always_inline))
inline uint8_t Board::rookStartSlot(uint8_t index) noexcept {
    switch (index) {
        case WHITE_ROOK_A_START: return 0;
        case WHITE_ROOK_H_START: return 1;
        case BLACK_ROOK_A_START: return 2;
        case BLACK_ROOK_H_START: return 3;
        default: return 0xFF;
    }
}

inline void Board::clearCastlingByRookStart(uint8_t rookStartIndex, bool setHasMovedBit) noexcept {
    static constexpr std::array<uint8_t, 4> ROOK_CASTLE_CLEAR_MASKS = {
        1u << WHITE_QUEENSIDE,
        1u << WHITE_KINGSIDE,
        1u << BLACK_QUEENSIDE,
        1u << BLACK_KINGSIDE
    };
    static constexpr std::array<uint8_t, 4> ROOK_HAS_MOVED_BITS = {
        1u << 1,
        1u << 2,
        1u << 4,
        1u << 5
    };

    const uint8_t slot = rookStartSlot(rookStartIndex);
    if (slot == 0xFF) return;

    castle &= static_cast<uint8_t>(~ROOK_CASTLE_CLEAR_MASKS[slot]);
    if (setHasMovedBit) {
        hasMoved |= ROOK_HAS_MOVED_BITS[slot];
    }
}

inline void Board::updateCastlingRightsOnPieceMove(uint8_t movingType, uint8_t movingColor, uint8_t fromIndex) noexcept {
    if (movingType == KING) {
        const uint8_t kingBit = (movingColor == WHITE) ? 0x01 : 0x08;
        const uint8_t castleMask = (movingColor == WHITE) ? 0x03 : 0x0C;
        castle &= static_cast<uint8_t>(~castleMask);
        hasMoved |= kingBit;
        return;
    }

    if (movingType == ROOK) {
        clearCastlingByRookStart(fromIndex, true);
    }
}

inline void Board::updateCastlingRightsOnRookCapture(uint8_t capturedPiece, uint8_t toIndex) noexcept {
    if ((capturedPiece & MASK_PIECE_TYPE) != ROOK) return;

    const uint8_t slot = rookStartSlot(toIndex);
    if (slot == 0xFF) return;

    const uint8_t capturedColor = capturedPiece & MASK_COLOR;
    if ((capturedColor == WHITE && slot >= 2) || (capturedColor == BLACK && slot < 2)) return;

    clearCastlingByRookStart(toIndex, false);
}

template<Board::MoveKind Kind>
inline void Board::doMoveByKind(
    MoveState& st,
    uint8_t moving,
    uint8_t movingType,
    uint8_t movingColor,
    uint8_t destBefore,
    uint8_t fromIndex,
    uint8_t toIndex,
    uint8_t promotionType
) noexcept {
    if constexpr (Kind == MoveKind::EnPassant) {
        // Remove the captured pawn from board storage and bitboards before moving.
        const int8_t captureOffset = (movingColor == WHITE) ? 8 : -8;
        const uint8_t capIndex = static_cast<uint8_t>(toIndex + captureOffset);
        const uint8_t capturedPiece = get(capIndex);
        const uint64_t capBit = BIT_MASKS[capIndex];
        st.capturedPiece = capturedPiece;
        st.enPassantCapturedIndex = capIndex;

        set(capIndex, EMPTY);
        occupancy &= ~capBit;
        removePieceFromBB(capturedPiece, capIndex);
    }

    if constexpr (isCaptureKind(Kind)) {
        removePieceFromBB(destBefore, toIndex);
    }

    // Apply the moving piece update using index-based helpers already available on Board.
    set(toIndex, static_cast<piece_id>(moving));
    set(fromIndex, EMPTY);
    fastUpdateOccupancyBB(fromIndex, toIndex);
    removePieceFromBB(moving, fromIndex);
    addPieceToBB(moving, toIndex);

    if constexpr (Kind == MoveKind::Castling) {
        // Move the rook with the same index-based fast path used for the king.
        const uint8_t fromFile = chess::file(fromIndex);
        const uint8_t toFile = chess::file(toIndex);
        const uint8_t rankBase = chess::rank(toIndex) << 3;
        const uint8_t rookFromFile = (toFile > fromFile) ? 7 : 0;
        const uint8_t rookToFile   = (toFile > fromFile) ? 5 : 3;
        const uint8_t rookFromIndex = rankBase | rookFromFile;
        const uint8_t rookToIndex   = rankBase | rookToFile;
        st.rookFromIndex = rookFromIndex;
        st.rookToIndex   = rookToIndex;

        const uint8_t rook = get(rookFromIndex);
        set(rookToIndex, static_cast<piece_id>(rook));
        set(rookFromIndex, EMPTY);
        fastUpdateOccupancyBB(rookFromIndex, rookToIndex);
        removePieceFromBB(rook, rookFromIndex);
        addPieceToBB(rook, rookToIndex);
    }

    updateCastlingRightsOnPieceMove(movingType, movingColor, fromIndex);

    if constexpr (isCaptureKind(Kind)) {
        updateCastlingRightsOnRookCapture(destBefore, toIndex);
    }

    if constexpr (Kind == MoveKind::DoublePawnPush) {
        // Cache the en-passant square directly from the midpoint index.
        const uint8_t enPassantIndex = (fromIndex + toIndex) >> 1;
        enPassant = enPassantIndex;
    }

    if constexpr (isPromotionKind(Kind)) {
        // Finalize the promotion with the dedicated unchecked helper used by the hot path.
        const uint8_t promo = normalizePromotionType(promotionType);
        st.promotionPieceType = promo;
        promoteUnchecked(toIndex, moving, promo);
    }
}

template<Board::MoveKind Kind>
inline void Board::undoMoveByKind(
    const MoveState& st,
    uint8_t& pieceOnTo,
    uint8_t fromIndex,
    uint8_t toIndex
) noexcept {
    if constexpr (isPromotionKind(Kind)) {
        // Rebuild the pawn on the destination square before rewinding the move.
        const uint8_t color = pieceOnTo & MASK_COLOR;
        const uint8_t pawnPiece = (PAWN | color);
        removePieceFromBB(pieceOnTo, toIndex);
        addPieceToBB(pawnPiece, toIndex);
        set(toIndex, static_cast<piece_id>(pawnPiece));
        pieceOnTo = pawnPiece;
    }

    // Restore the moving piece back to its source square using index-based writes.
    set(fromIndex, static_cast<piece_id>(pieceOnTo));
    set(toIndex, EMPTY);
    fastUpdateOccupancyBB(toIndex, fromIndex);
    removePieceFromBB(pieceOnTo, toIndex);
    addPieceToBB(pieceOnTo, fromIndex);

    if constexpr (Kind == MoveKind::EnPassant) {
        // Put back the pawn captured en-passant on its original square.
        const uint8_t capIndex = st.enPassantCapturedIndex;
        const uint64_t capBit = BIT_MASKS[capIndex];
        set(capIndex, static_cast<piece_id>(st.capturedPiece));
        occupancy |= capBit;
        addPieceToBB(st.capturedPiece, capIndex);
    } else if constexpr (isCaptureKind(Kind)) {
        if (st.capturedPiece != EMPTY) {
            // Restore the captured piece on the destination square.
            const uint64_t toBit = BIT_MASKS[toIndex];
            set(toIndex, static_cast<piece_id>(st.capturedPiece));
            occupancy |= toBit;
            addPieceToBB(st.capturedPiece, toIndex);
        }
    }

    if constexpr (Kind == MoveKind::Castling) {
        // Rewind the rook to its original square after the king is restored.
        const uint8_t rookFromIndex = st.rookFromIndex;
        const uint8_t rookToIndex   = st.rookToIndex;
        const uint8_t rook = get(rookToIndex);
        set(rookFromIndex, static_cast<piece_id>(rook));
        set(rookToIndex, EMPTY);
        fastUpdateOccupancyBB(rookToIndex, rookFromIndex);
        removePieceFromBB(rook, rookToIndex);
        addPieceToBB(rook, rookFromIndex);
    }
}
