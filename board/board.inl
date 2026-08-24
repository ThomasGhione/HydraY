// ==============================
// Constructors
// ==============================
inline Board::Board() noexcept {
    fenToBoard(STARTING_FEN);
}

inline Board::Board(const std::string& fen) {
    fenToBoard(fen);
}

inline Board::Board(const Board& other) noexcept {
    copyFromBoard(other);
}

inline Board& Board::operator=(const Board& other) noexcept {
    if (this != &other) {
        copyFromBoard(other);
    }
    return *this;
}

inline void Board::copyFromBoard(const Board& other) noexcept {
    pawns_bb = other.pawns_bb;
    knights_bb = other.knights_bb;
    bishops_bb = other.bishops_bb;
    rooks_bb = other.rooks_bb;
    queens_bb = other.queens_bb;
    kings_bb = other.kings_bb;

    chessboard = other.chessboard;
    currentHash = other.currentHash;

    historySize = other.historySize;
    if (historySize > 0) {
        std::memcpy(repetitionHistory.data(), other.repetitionHistory.data(), historySize * sizeof(uint64_t));
    }

    occupancy = other.occupancy;
    nnueAccumulator = other.nnueAccumulator;
    halfMoveClock = other.halfMoveClock;
    fullMoveClock = other.fullMoveClock;
    castle = other.castle;
    hasMoved = other.hasMoved;
    enPassant = other.enPassant;
    epHashFile = other.epHashFile;
    activeColor = other.activeColor;
}

// ==============================
// Public Board API
// ==============================
__attribute__((hot, always_inline))
inline void Board::set(uint8_t index, piece_id value) noexcept {
    const uint8_t internal_row = 7 - (index >> 3);
    const uint8_t shift = (index & 7) << 2; // file * 4
    chessboard[internal_row] = (chessboard[internal_row] & ~(MASK_PIECE << shift)) | ((value & MASK_PIECE) << shift);
}


// ==============================
// Board Internals
// ==============================

inline void Board::rebuildBitboardsFromSquares() noexcept {
    // Reset all bitboards
    occupancy       = 0ULL;
    pawns_bb[0]     = pawns_bb[1]     = 0ULL;
    knights_bb[0]   = knights_bb[1]   = 0ULL;
    bishops_bb[0]   = bishops_bb[1]   = 0ULL;
    rooks_bb[0]     = rooks_bb[1]     = 0ULL;
    queens_bb[0]    = queens_bb[1]    = 0ULL;
    kings_bb[0]     = kings_bb[1]     = 0ULL;

    // Single loop: iterate all 64 squares directly
    // index = rank * 8 + file, where rank 0 = row 8, rank 7 = row 1
    for (uint8_t index = 0; index < 64; ++index) {
        const uint8_t piece = get(index);
        
        if (piece == EMPTY) continue;
        
        const uint64_t bit = BIT_MASKS[index];
        const uint8_t color = colorToIndex(piece);

        occupancy |= bit;
        dispatchPieceBBUpdate<true>(piece & MASK_PIECE_TYPE, color, bit);
    }

    refreshNnueAccumulator();
}

__attribute__((always_inline))
inline void Board::fastUpdateOccupancyBB(uint8_t fromIndex, uint8_t toIndex) noexcept {
    occupancy |= BIT_MASKS[toIndex];  // Set the bit at 'to' position    
    occupancy &= ~BIT_MASKS[fromIndex]; // Clear the bit at 'from' position
}

inline bool Board::isKingSafeAfterMove(
    uint8_t movingColor,
    uint8_t fromIndex,
    uint8_t toIndex,
    uint64_t capturedMask
) const noexcept {
    const uint8_t side = colorToIndex(movingColor);
    const uint64_t kingBB = kings_bb[side];
    if (!kingBB) [[unlikely]] return false;

    const uint8_t oppSide = side ^ 1;
    const uint8_t kingSq = std::countr_zero(kingBB);

    const uint64_t occNew =
        (occupancy & ~BIT_MASKS[fromIndex] & ~capturedMask) | BIT_MASKS[toIndex];
    const uint64_t keep = ~capturedMask;

    return !isKingAttackedCustom(kingSq, oppSide, occNew,
                                 pawns_bb[oppSide]   & keep,
                                 knights_bb[oppSide] & keep,
                                 bishops_bb[oppSide] & keep,
                                 rooks_bb[oppSide]   & keep,
                                 queens_bb[oppSide]  & keep,
                                 kings_bb[oppSide]   & keep);
}

template<uint8_t PieceType, bool Add>
inline void Board::updatePieceTypeBB(uint8_t color, uint64_t bit) noexcept {
    if constexpr (PieceType == PAWN) {
        if constexpr (Add) pawns_bb[color] |= bit;
        else pawns_bb[color] &= ~bit;
    } else if constexpr (PieceType == KNIGHT) {
        if constexpr (Add) knights_bb[color] |= bit;
        else knights_bb[color] &= ~bit;
    } else if constexpr (PieceType == BISHOP) {
        if constexpr (Add) bishops_bb[color] |= bit;
        else bishops_bb[color] &= ~bit;
    } else if constexpr (PieceType == ROOK) {
        if constexpr (Add) rooks_bb[color] |= bit;
        else rooks_bb[color] &= ~bit;
    } else if constexpr (PieceType == QUEEN) {
        if constexpr (Add) queens_bb[color] |= bit;
        else queens_bb[color] &= ~bit;
    } else if constexpr (PieceType == KING) {
        if constexpr (Add) kings_bb[color] |= bit;
        else kings_bb[color] &= ~bit;
    }
}

template<bool Add>
inline void Board::dispatchPieceBBUpdate(uint8_t pieceType, uint8_t color, uint64_t bit) noexcept {
    switch (pieceType) {
        case PAWN:   updatePieceTypeBB<PAWN, Add>(color, bit); break;
        case KNIGHT: updatePieceTypeBB<KNIGHT, Add>(color, bit); break;
        case BISHOP: updatePieceTypeBB<BISHOP, Add>(color, bit); break;
        case ROOK:   updatePieceTypeBB<ROOK, Add>(color, bit); break;
        case QUEEN:  updatePieceTypeBB<QUEEN, Add>(color, bit); break;
        case KING:   updatePieceTypeBB<KING, Add>(color, bit); break;
        default: break;
    }
}

__attribute__((always_inline))
inline void Board::addPieceToBB(uint8_t piece, uint8_t index) noexcept {
    dispatchPieceBBUpdate<true>(piece & MASK_PIECE_TYPE, colorToIndex(piece), BIT_MASKS[index]);
    if (NNUE::activeNetwork != nullptr) [[likely]] {
        nnueAccumulator.update<true>(piece, index);
    }
}

__attribute__((always_inline))
inline void Board::removePieceFromBB(uint8_t piece, uint8_t index) noexcept {
    dispatchPieceBBUpdate<false>(piece & MASK_PIECE_TYPE, colorToIndex(piece), BIT_MASKS[index]);
    if (NNUE::activeNetwork != nullptr) [[likely]] {
        nnueAccumulator.update<false>(piece, index);
    }
}

__attribute__((always_inline))
inline void Board::movePieceOnBB(uint8_t piece, uint8_t fromIndex, uint8_t toIndex) noexcept {
    const uint8_t type = piece & MASK_PIECE_TYPE;
    const uint8_t color = colorToIndex(piece);
    dispatchPieceBBUpdate<false>(type, color, BIT_MASKS[fromIndex]);
    dispatchPieceBBUpdate<true>(type, color, BIT_MASKS[toIndex]);
    if (NNUE::activeNetwork != nullptr) [[likely]] {
        nnueAccumulator.updateMove(piece, fromIndex, toIndex);
    }
}

// From-scratch accumulator rebuild. Note rebuildBitboardsFromSquares goes
// through dispatchPieceBBUpdate directly (NOT addPieceToBB), so bulk rebuilds
// never double-count: they land here once at the end instead.
inline void Board::refreshNnueAccumulator() noexcept {
    if (NNUE::activeNetwork == nullptr) return;
    // No kings (unit-test fragments, mid-load states): leave the accumulator
    // as-is; evaluation is guarded upstream by the kings-missing check.
    if (kings_bb[0] == 0 || kings_bb[1] == 0) return;
    const int wKingLerf      = std::countr_zero(kings_bb[0]) ^ 56;
    const int bKingLerfView  = std::countr_zero(kings_bb[1]); // lerf ^ 56
    nnueAccumulator.resetWithKings(wKingLerf, bKingLerfView);
    for (uint8_t index = 0; index < 64; ++index) {
        const uint8_t piece = get(index);
        if (piece != EMPTY) nnueAccumulator.update<true>(piece, index);
    }
}

// HalfKA lazy refresh: rebuild only the perspectives whose own king crossed
// bucket/flip since the last clean state, as a Finny-table diff against the
// cached accumulator for the target (bucket, flip). Called from
// consistent-board points only (NNUE::evaluate, selftest) — never mid-doMove.
inline void Board::ensureNnueAccumulatorClean() const noexcept {
    if (NNUE::activeNetwork == nullptr) return;
    if (!(nnueAccumulator.dirty[0] || nnueAccumulator.dirty[1])) [[likely]] return;
    if (kings_bb[0] == 0 || kings_bb[1] == 0) return;

    // Per-thread cache (Lazy SMP: each helper searches its own Board on its
    // own thread). Stale entries from other positions are still correct diff
    // bases — no invalidation needed, ever.
    static thread_local NNUE::FinnyTable finny;
    finny.ensureInitialised();

    const uint64_t cur[2][6] = {
        {pawns_bb[0], knights_bb[0], bishops_bb[0], rooks_bb[0], queens_bb[0], kings_bb[0]},
        {pawns_bb[1], knights_bb[1], bishops_bb[1], rooks_bb[1], queens_bb[1], kings_bb[1]},
    };

    for (int p = 0; p < 2; ++p) {
        if (!nnueAccumulator.dirty[p]) continue;
        // Own king square from this perspective's view: lerf for white,
        // lerf ^ 56 for black — which folds back to the raw engine index.
        const int engineKing = std::countr_zero(kings_bb[p]);
        const int ownKingView = (p == 1) ? engineKing : (engineKing ^ 56);
        const int rowBase = NNUE::kingFeatureBase(ownKingView);
        const int rowFlip = NNUE::kingFlip(ownKingView);
        NNUE::FinnyEntry& e =
            finny.entry[p][NNUE::KING_BUCKET_MAP[ownKingView]][rowFlip ? 1 : 0];

        for (int c = 0; c < 2; ++c) {
            for (int t = 0; t < 6; ++t) {
                const uint8_t piece = static_cast<uint8_t>((c == 0 ? 0x8 : 0) | (t + 1));
                uint64_t added   = cur[c][t] & ~e.bb[c][t];
                uint64_t removed = e.bb[c][t] & ~cur[c][t];
                while (added) {
                    const int idx = std::countr_zero(added);
                    added &= added - 1;
                    NNUE::Accumulator::updateRow<true>(e.v, rowBase, rowFlip, p, piece,
                                                       static_cast<uint8_t>(idx));
                }
                while (removed) {
                    const int idx = std::countr_zero(removed);
                    removed &= removed - 1;
                    NNUE::Accumulator::updateRow<false>(e.v, rowBase, rowFlip, p, piece,
                                                        static_cast<uint8_t>(idx));
                }
                e.bb[c][t] = cur[c][t];
            }
        }

        std::memcpy(nnueAccumulator.v[p], e.v, sizeof(e.v));
        nnueAccumulator.base[p] = rowBase;
        nnueAccumulator.flip[p] = static_cast<uint8_t>(rowFlip);
        nnueAccumulator.dirty[p] = false;
    }
}