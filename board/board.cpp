#include <algorithm>
#include "board.hpp"
#include "../tt/zobrist.hpp"

namespace chess {

// Slider attack tables, filled once before main.
[[maybe_unused]] static const bool sliderTablesReady = (pieces::initMagicBitboards(), true);

void Board::copyFromBoard(const Board& other) noexcept {
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

void Board::rebuildBitboardsFromSquares() noexcept {
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

inline bool Board::isKingSafeAfterMove(uint8_t movingColor, uint8_t fromIndex, uint8_t toIndex,
                                       uint64_t capturedMask) const noexcept {
    const uint8_t side = colorToIndex(movingColor);
    if (!bb[KING][side]) [[unlikely]] return false;
    const uint64_t occNew =
        (occupancy & ~(1ULL << fromIndex) & ~capturedMask) | (1ULL << toIndex);
    return !attackedBy(std::countr_zero(bb[KING][side]), side ^ 1, occNew, ~capturedMask);
}

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

// From-scratch rebuild of the current frame, which becomes the oldest one of
// the line. rebuildBitboardsFromSquares sets the bitboards directly (NOT
// through addPieceToBB), so bulk rebuilds land here once at the end.
void Board::refreshNnueAccumulator() noexcept {
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
        NNUE::Accumulator::fusePair<S, A>(acc.v[0], acc.v[1], nullptr, nullptr, subRow, addRow);
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
__attribute__((hot))
void Board::ensureNnueAccumulatorClean() const noexcept {
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
    const auto incremental = [&](int start, int target) __attribute__((hot)) {
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
                        NNUE::Accumulator::fusePair<S, A, true>(dst.v[0], dst.v[1], src.v[0], src.v[1], subRow, addRow);
                        first = false;
                    } else {
                        NNUE::Accumulator::fusePair<S, A>(dst.v[0], dst.v[1], nullptr, nullptr, subRow, addRow);
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
    // Return before the thread_local: mingw emulates TLS with a call per access.
    if (usable[0] && usable[1]) return;

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

} // namespace chess
