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
    bb = other.bb;

    chessboard = other.chessboard;
    currentHash = other.currentHash;

    historySize = other.historySize;
    if (historySize > 0) {
        std::memcpy(repetitionHistory.data(), other.repetitionHistory.data(), historySize * sizeof(uint64_t));
    }

    occupancy = other.occupancy;
    // Only the current position's frame: this board's line starts there.
    other.ensureNnueAccumulatorClean();
    accTop = accFloor = other.accTop;
    frame(accTop) = other.frame(other.accTop);
    halfMoveClock = other.halfMoveClock;
    fullMoveClock = other.fullMoveClock;
    castle = other.castle;
    enPassant = other.enPassant;
    epHashFile = other.epHashFile;
    activeColor = other.activeColor;
    pliesFromNull = other.pliesFromNull;
}

inline void Board::rebuildBitboardsFromSquares() noexcept {
    occupancy = 0ULL;
    bb = {};
    for (uint8_t index = 0; index < 64; ++index) {
        const uint8_t piece = get(index);
        if (piece == EMPTY) continue;
        occupancy |= 1ULL << index;
        bb[piece & MASK_PIECE_TYPE][colorToIndex(piece)] |= 1ULL << index;
    }
    refreshNnueAccumulator();
}

inline bool Board::isKingSafeAfterMove(
    uint8_t movingColor,
    uint8_t fromIndex,
    uint8_t toIndex,
    uint64_t capturedMask
) const noexcept {
    const uint8_t side = colorToIndex(movingColor);
    if (!bb[KING][side]) [[unlikely]] return false;
    const uint64_t occNew =
        (occupancy & ~(1ULL << fromIndex) & ~capturedMask) | (1ULL << toIndex);
    return !attackedBy(std::countr_zero(bb[KING][side]), side ^ 1, occNew, ~capturedMask);
}

// From-scratch rebuild of the current frame, which becomes the oldest one of
// the line. rebuildBitboardsFromSquares sets the bitboards directly (NOT
// through addPieceToBB), so bulk rebuilds land here once at the end.
inline void Board::refreshNnueAccumulator() noexcept {
    if (NNUE::activeNetwork == nullptr) return;
    accFloor = accTop;
    NNUE::Accumulator& acc = frame(accTop);
    acc.ndelta = 0;
    // No kings (unit-test fragments, mid-load states): no values; evaluation
    // is guarded upstream by the kings-missing check.
    acc.dirty[0] = acc.dirty[1] = true;
    if (bb[KING][0] == 0 || bb[KING][1] == 0) return;
    const int wKingLerf      = std::countr_zero(bb[KING][0]) ^ 56;
    const int bKingLerfView  = std::countr_zero(bb[KING][1]); // lerf ^ 56
    acc.resetWithKings(wKingLerf, bKingLerfView);
    uint16_t add[64];   // piece << 8 | square
    int na = 0;
    for (uint8_t index = 0; index < 64; ++index) {
        const uint8_t piece = get(index);
        if (piece != EMPTY) add[na++] = static_cast<uint16_t>(piece << 8 | index);
    }
    NNUE::Accumulator::forEachPass(0, na, [&]<int S, int A>(int, int ai) {
        const int16_t* subRow[2][2] = {};
        const int16_t* addRow[2][2];
        for (int p = 0; p < 2; ++p) {
            for (int k = 0; k < A; ++k) {
                addRow[p][k] = acc.featureRow(p, static_cast<uint8_t>(add[ai + k] >> 8),
                                              static_cast<uint8_t>(add[ai + k]));
            }
        }
        NNUE::Accumulator::fusePair<S, A>(acc.v[0], acc.v[1], subRow, addRow);
    });
}

// Computes the current frame. Its nearest computed ancestor, at most
// ACC_MAX_WALK moves back, is a valid start for a perspective whose king still
// stands on the same basis: the changes logged since sum to the feature
// difference whatever the king did in between. Rows that appear on both sides
// cancel, and int16 is modular, so the rest applies in any order. A
// perspective without such a start is rebuilt as a diff against the Finny
// table. Called from consistent-board points only (NNUE::evaluate, a copy,
// the selftest), never mid-doMove.
inline void Board::ensureNnueAccumulatorClean() const noexcept {
    if (NNUE::activeNetwork == nullptr) return;
    NNUE::Accumulator& top = frame(accTop);
    if (!top.dirty[0]) [[likely]] return;   // a frame is computed for both perspectives at once
    if (bb[KING][0] == 0 || bb[KING][1] == 0) return;

    int kingView[2];
    int rowBase[2];
    int rowFlip[2];
    for (int p = 0; p < 2; ++p) {
        // Own king square from this perspective's view: lerf for white,
        // lerf ^ 56 for black - which folds back to the raw engine index.
        const int engineKing = std::countr_zero(bb[KING][p]);
        kingView[p] = (p == 1) ? engineKing : (engineKing ^ 56);
        rowBase[p] = NNUE::kingFeatureBase(kingView[p]);
        rowFlip[p] = NNUE::kingFlip(kingView[p]);
    }
    int from = -1;
    for (int i = accTop - 1; i >= std::max(accFloor, accTop - NNUE::ACC_MAX_WALK); --i) {
        if (!frame(i).dirty[0]) {
            from = i;
            break;
        }
    }
    bool usable[2] = {false, false};
    if (from >= 0) {
        const NNUE::Accumulator& src = frame(from);
        for (int p = 0; p < 2; ++p) usable[p] = src.base[p] == rowBase[p] && src.flip[p] == rowFlip[p];
    }

    // Frame `target` from frame `start` for the usable perspectives, through
    // the changes logged in the frames between.
    const auto incremental = [&](int start, int target) {
        constexpr int CAP = NNUE::ACC_MAX_WALK * NNUE::ACC_MAX_DELTAS;
        uint16_t sub[CAP];   // piece << 8 | square
        uint16_t add[CAP];
        int ns = 0;
        int na = 0;
        for (int i = start + 1; i <= target; ++i) {
            const NNUE::Accumulator& f = frame(i);
            for (int k = 0; k < f.ndelta; ++k) {
                const NNUE::AccDelta& d = f.delta[k];
                const uint16_t at = static_cast<uint16_t>(d.piece << 8 | d.from);
                if (d.kind == NNUE::AccDelta::Add) {
                    add[na++] = at;
                } else {
                    sub[ns++] = at;
                    if (d.kind == NNUE::AccDelta::Move) add[na++] = static_cast<uint16_t>(d.piece << 8 | d.to);
                }
            }
        }
        for (int i = 0; i < ns; ) {
            int j = 0;
            while (j < na && add[j] != sub[i]) ++j;
            if (j == na) { ++i; continue; }
            sub[i] = sub[--ns];   // swap-remove: order is irrelevant
            add[j] = add[--na];
        }
        const NNUE::Accumulator& src = frame(start);
        NNUE::Accumulator& dst = frame(target);
        const auto row = [&](int p, uint16_t c) {
            return NNUE::Accumulator::rowFor(rowBase[p], rowFlip[p], p, static_cast<uint8_t>(c >> 8),
                                             static_cast<uint8_t>(c));
        };
        if (usable[0] && usable[1]) {
            if (ns == 0 && na == 0) {
                std::memcpy(dst.v, src.v, sizeof(dst.v));
            } else {
                bool first = true;
                NNUE::Accumulator::forEachPass(ns, na, [&]<int S, int A>(int si, int ai) {
                    const int16_t* subRow[2][2];
                    const int16_t* addRow[2][2];
                    for (int p = 0; p < 2; ++p) {
                        for (int k = 0; k < S; ++k) subRow[p][k] = row(p, sub[si + k]);
                        for (int k = 0; k < A; ++k) addRow[p][k] = row(p, add[ai + k]);
                    }
                    if (first) {
                        NNUE::Accumulator::fusePairFrom<S, A>(src.v[0], src.v[1], dst.v[0], dst.v[1], subRow, addRow);
                        first = false;
                    } else {
                        NNUE::Accumulator::fusePair<S, A>(dst.v[0], dst.v[1], subRow, addRow);
                    }
                });
            }
        } else {
            const int p = usable[0] ? 0 : 1;
            const int16_t* subRow[CAP];
            const int16_t* addRow[CAP];
            for (int k = 0; k < ns; ++k) subRow[k] = row(p, sub[k]);
            for (int k = 0; k < na; ++k) addRow[k] = row(p, add[k]);
            NNUE::Accumulator::applyRows(dst.v[p], subRow, ns, addRow, na, src.v[p]);
        }
    };

    if (usable[0] || usable[1]) {
        // Every frame on the way is computed too, one move at a time, back to
        // the last king move (an older frame stands on another basis): the
        // siblings and cousins that follow start one move away.
        int first = accTop;
        if (usable[0] && usable[1]) {
            while (first - 1 > from) {
                bool kingMoved = false;
                const NNUE::Accumulator& f = frame(first);
                for (int k = 0; k < f.ndelta; ++k) kingMoved |= (f.delta[k].piece & MASK_PIECE_TYPE) == KING;
                if (kingMoved) break;
                --first;
            }
        }
        for (int i = first; i <= accTop; ++i) {
            incremental(from, i);
            NNUE::Accumulator& f = frame(i);
            for (int p = 0; p < 2; ++p) {
                f.base[p] = rowBase[p];
                f.flip[p] = static_cast<uint8_t>(rowFlip[p]);
                f.dirty[p] = false;
            }
            from = i;
        }
    }

    // Per-thread cache (Lazy SMP: each helper searches its own Board on its
    // own thread). Stale entries from other positions are still correct diff
    // bases - no invalidation needed, ever.
    static thread_local NNUE::FinnyTable finny;
    for (int p = 0; p < 2; ++p) {
        top.base[p] = rowBase[p];
        top.flip[p] = static_cast<uint8_t>(rowFlip[p]);
        top.dirty[p] = false;
        if (usable[p]) continue;
        finny.ensureInitialised();
        NNUE::FinnyEntry& e = finny.entry[p][NNUE::KING_BUCKET_MAP[kingView[p]]][rowFlip[p] ? 1 : 0];

        // Collect the diff first and apply it in chunks: a board holds at most
        // 64 pieces, so neither side of the diff can exceed that.
        const int16_t* sub[64];
        const int16_t* add[64];
        int ns = 0;
        int na = 0;
        for (int c = 0; c < 2; ++c) {
            for (int t = 0; t < 6; ++t) {
                const uint8_t piece = static_cast<uint8_t>((c == 0 ? 0x8 : 0) | (t + 1));
                uint64_t added   = bb[t + 1][c] & ~e.bb[c][t];
                uint64_t removed = e.bb[c][t] & ~bb[t + 1][c];
                while (added)
                    add[na++] = NNUE::Accumulator::rowFor(rowBase[p], rowFlip[p], p, piece,
                                                          static_cast<uint8_t>(pieces::popLSB(added)));
                while (removed)
                    sub[ns++] = NNUE::Accumulator::rowFor(rowBase[p], rowFlip[p], p, piece,
                                                          static_cast<uint8_t>(pieces::popLSB(removed)));
                e.bb[c][t] = bb[t + 1][c];
            }
        }
        NNUE::Accumulator::applyRows(e.v, sub, ns, add, na);
        std::memcpy(top.v[p], e.v, sizeof(e.v));
    }
}
