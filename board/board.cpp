#include <algorithm>
#include "board.hpp"
#include "../tt/zobrist.hpp"

namespace chess {

// Slider attack tables, filled once before main.
[[maybe_unused]] static const bool sliderTablesReady = (pieces::Rook::init(), pieces::Bishop::init(), true);

void Board::copyFromBoard(const Board& other) noexcept {
    bb = other.bb;

    chessboard = other.chessboard;
    state = other.state;
    if (state.historySize > 0) {
        std::memcpy(repetitionHistory.data(), other.repetitionHistory.data(), state.historySize * sizeof(uint64_t));
    }

    occupancy = other.occupancy;
    // Only the current position's frame: this board's line starts there.
    other.ensureNnueAccumulatorClean();
    accTop = accFloor = other.accTop;
    frame(accTop) = other.frame(other.accTop);
    activeColor = other.activeColor;
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
    return (attackersOf(std::countr_zero(bb[KING][side]), side ^ 1, occNew) & ~capturedMask) == 0;
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
                    if (isValidSquare(state.enPassant) && toIndex == state.enPassant)
                        return isKingSafeAfterMove(movingColor, fromIndex, toIndex, 1ULL << enPassantVictim(fromPiece, toIndex));
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


uint64_t Board::checkersTo(uint8_t color) const noexcept {
    const uint8_t side = colorToIndex(color);
    if (!bb[KING][side]) [[unlikely]] return 0ULL;
    return attackersOf(std::countr_zero(bb[KING][side]), side ^ 1, occupancy);
}

[[nodiscard]] bool Board::isKingMoveLegal(uint8_t fromIndex, uint8_t toIndex, uint64_t toBit,
                                          uint8_t movingColor) const noexcept {
    const int diff = static_cast<int>(toIndex) - fromIndex;
    // Only castling moves the king two files: a one-step offset is 1, 7, 8 or 9.
    if (diff == 2 || diff == -2) [[unlikely]] {
        const int right = 2 * colorToIndex(movingColor) + (diff < 0);
        return fromIndex == CASTLINGS[right].king && canCastle(right);
    }
    return (pieces::KING_ATTACKS[fromIndex] & toBit) && !isSquareAttacked(toIndex, oppositeColor(movingColor), fromIndex);
}

bool Board::canCastle(int right) const noexcept {
    const Castling& c = CASTLINGS[right];
    const int side = right >> 1;
    if (!((state.castle >> right) & 1) || !((bb[KING][side] >> c.king) & 1) || !((bb[ROOK][side] >> c.rook) & 1)
        || (occupancy & c.between)) return false;
    for (uint64_t path = c.kingPath; path; )
        if (attackersOf(static_cast<uint8_t>(pieces::popLSB(path)), side ^ 1, occupancy)) return false;
    return true;
}

// Whether byColor attacks targetIndex, with excludeSquare (if < 64) lifted off
// the board, as a moving king needs.
bool Board::isSquareAttacked(uint8_t targetIndex, uint8_t byColor, uint8_t excludeSquare) const noexcept {
    const uint64_t occMinus = excludeSquare < 64 ? (occupancy & ~(1ULL << excludeSquare)) : occupancy;
    return attackersOf(targetIndex, colorToIndex(byColor), occMinus) != 0;
}

void Board::recomputeHashAndEp() noexcept {
    state.hash = zobrist::computeHashKey(*this);
    state.epHashFile = 0xFF;
    if (zobrist::hasPseudoLegalEnPassantCapture(*this, getEnPassant()))
        state.epHashFile = chess::file(getEnPassant());
}

void Board::rebuildRepetitionHistory() noexcept {
    recomputeHashAndEp();
    state.historySize = 0;
    state.pliesFromNull = 255;
    repetitionHistory[state.historySize++] = state.hash;
}

void Board::updateRepetitionAfterMove(bool resetHistory, MoveState& st) noexcept {
    if (resetHistory)
        state.historySize = 0;

    if (state.historySize >= repetitionHistory.size()) {
        std::memmove(repetitionHistory.data(), repetitionHistory.data() + 1, (repetitionHistory.size() - 1) * sizeof(uint64_t));
        state.historySize = static_cast<uint8_t>(repetitionHistory.size() - 1);
    }
    // Remember the slot we are about to overwrite so undoMove can restore it
    // (writeIndex == historySize here; on undo historySize-1 recovers it). This
    // is what keeps a reset-to-0 on an irreversible move from corrupting earlier
    // entries in the parent line.
    st.prevHistorySlotValue = repetitionHistory[state.historySize];
    repetitionHistory[state.historySize++] = state.hash;
}

// Occurrences of the current position, itself included. Only the same side to
// move can match, and every ply (null moves too) takes one slot, so the scan
// steps by two. It stops at the last null move: no game can repeat a position
// across one, the same bound Stockfish applies (pliesFromNull).
int Board::countRepetitions() const noexcept {
    const int last = static_cast<int>(state.historySize) - 1;
    const int stop = std::max(0, last - static_cast<int>(state.pliesFromNull));
    int count = 0;
    for (int i = last; i >= stop; i -= 2) {
        count += (repetitionHistory[static_cast<size_t>(i)] == state.hash);
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

namespace {

// Perspective p's own king square from its own view: lerf for white, lerf ^ 56
// for black, which folds back to the engine index.
int ownKingView(const Board& b, int p) noexcept {
    const int engineKing = std::countr_zero(b.bb[Board::KING][p]);
    return (p == 1) ? engineKing : (engineKing ^ 56);
}

// Brings row, perspective p's values for the pieces in `cached` (by colour and
// type - 1), to the pieces of b, and cached with it. The diff is collected
// first and applied in chunks: a board holds at most 64 pieces, so neither
// side of it can exceed that.
__attribute__((always_inline))
inline void syncRow(const Board& b, int16_t* row, uint64_t (&cached)[2][6], int p, int rowBase, int rowFlip) noexcept {
    const int16_t* sub[64];
    const int16_t* add[64];
    int ns = 0;
    int na = 0;
    for (int c = 0; c < 2; ++c) {
        for (int t = 0; t < 6; ++t) {
            const uint8_t piece = static_cast<uint8_t>((c == 0 ? Board::WHITE : Board::BLACK) | (t + 1));
            uint64_t added   = b.bb[t + 1][c] & ~cached[c][t];
            uint64_t removed = cached[c][t] & ~b.bb[t + 1][c];
            while (added)
                add[na++] = NNUE::Accumulator::rowFor(rowBase, rowFlip, p, piece, static_cast<uint8_t>(pieces::popLSB(added)));
            while (removed)
                sub[ns++] = NNUE::Accumulator::rowFor(rowBase, rowFlip, p, piece, static_cast<uint8_t>(pieces::popLSB(removed)));
            cached[c][t] = b.bb[t + 1][c];
        }
    }
    NNUE::Accumulator::applyRows(row, sub, ns, add, na);
}

// dst = src - sub + add, a row in both lists cancelled first; add needs 8 spare
// slots past na. Out of line: every caller shares one copy.
__attribute__((noinline, hot))
void cancelAndApplyThreats(int16_t* __restrict dst, const int16_t* __restrict src,
                           const int8_t** sub, int ns, const int8_t** add, int na) noexcept {
    if (ns > 0 && na > 0) {
#if defined(__AVX2__)
        // A fixed-size store: clang turns a variable-length one into a memset call.
        std::memset(add + na, 0, 8 * sizeof(const int8_t*));
        const auto matchMask8 = [](const __m256i key, const __m256i a0, const __m256i a1) noexcept {
            const int m0 = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(key, a0)));
            const int m1 = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(key, a1)));
            return m0 | (m1 << 4);
        };
        const auto swapRemove = [&](int i, int j) noexcept {
            sub[i] = sub[--ns];
            add[j] = add[--na];
            add[na] = nullptr;
        };
        if (na <= 8) {
            __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add));
            __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add + 4));
            for (int i = 0; i < ns; ) {
                const int mask = matchMask8(_mm256_set1_epi64x(reinterpret_cast<int64_t>(sub[i])), a0, a1);
                if (mask == 0) { ++i; continue; }
                swapRemove(i, std::countr_zero(static_cast<unsigned>(mask)));
                if (na == 0) break;
                a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add));
                a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add + 4));
            }
        } else {
            for (int i = 0; i < ns; ) {
                const __m256i key = _mm256_set1_epi64x(reinterpret_cast<int64_t>(sub[i]));
                int hit = -1;
                for (int base = 0; base < na; base += 8) {
                    const int mask = matchMask8(
                        key,
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add + base)),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add + base + 4)));
                    if (mask != 0) {
                        hit = base + std::countr_zero(static_cast<unsigned>(mask));
                        break;
                    }
                }
                if (hit < 0) { ++i; continue; }
                swapRemove(i, hit);
                if (na == 0) break;
            }
        }
#else
        for (int i = 0; i < ns; ) {
            int j = 0;
            while (j < na && add[j] != sub[i]) ++j;
            if (j == na) { ++i; continue; }
            sub[i] = sub[--ns];
            add[j] = add[--na];
        }
#endif
    }
    if (ns == 0 && na == 0) {
        std::memcpy(dst, src, NNUE::HIDDEN * sizeof(int16_t));
        return;
    }
    NNUE::Accumulator::applyThreatRows(dst, src, sub, ns, add, na);
}

// Both perspectives' threats of a frame from the one before it.
__attribute__((noinline, hot))
void applyThreatPair1(NNUE::Accumulator& dst, const NNUE::Accumulator& src, int flip0, int flip1) noexcept {
    const int8_t* sub[2][NNUE::ACC_MAX_THREATS];
    const int8_t* add[2][NNUE::ACC_MAX_THREATS + 8];
    int ns[2] = {0, 0};
    int na[2] = {0, 0};
    const int toView0 = 56 ^ flip0;
    for (int k = 0; k < dst.nthreat; ++k) {
        const NNUE::ThreatDelta& d = dst.threatDelta[k];
        int f0 = 0, f1 = 0;
        NNUE::threatFeaturePair(toView0, flip1, d.attacker, d.from, d.victim, d.to(), f0, f1);
        const bool isAdd = d.add();
        const auto pushRow = [&](int p, int feat) noexcept {
            if (feat < 0) return;
            const int8_t* row = NNUE::Accumulator::threatRow(static_cast<uint32_t>(feat));
            (isAdd ? add[p][na[p]++] : sub[p][ns[p]++]) = row;
#if defined(__AVX2__)
            _mm_prefetch(reinterpret_cast<const char*>(row), _MM_HINT_T0);
#endif
        };
        pushRow(0, f0);
        pushRow(1, f1);
    }
    for (int p = 0; p < 2; ++p)
        cancelAndApplyThreats(dst.threat[p], src.threat[p], sub[p], ns[p], add[p], na[p]);
}

// A piece diff of more than one pass, out of line to keep incremental() small.
__attribute__((noinline))
void fusePairFallback(NNUE::Accumulator& dst, const NNUE::Accumulator& src,
                      const int rowBase[2], const int rowFlip[2],
                      const uint16_t* sub, int ns, const uint16_t* add, int na) noexcept {
    const auto row = [&](int p, uint16_t c) {
        return NNUE::Accumulator::rowFor(rowBase[p], rowFlip[p], p, static_cast<uint8_t>(c >> 8),
                                         static_cast<uint8_t>(c));
    };
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

} // namespace

void Board::computeThreats(int16_t* row, int p, int flip) const noexcept {
    const int8_t* add[NNUE::MAX_ACTIVE_THREATS];
    int na = 0;
    const NNUE::ThreatPerspective tp(p, flip);
    const uint64_t bishops = bb[BISHOP][0] | bb[BISHOP][1];
    const uint64_t queens  = bb[QUEEN][0]  | bb[QUEEN][1];
    const uint64_t kings   = bb[KING][0]   | bb[KING][1];
    const uint64_t nonQueenVictims = occupancy ^ queens;
    const uint64_t pawnVictims     = occupancy ^ (bishops | queens | kings);
    const uint64_t kingVictims     = occupancy ^ (queens | kings);

    const auto emit = [&](int relA, uint8_t from, uint64_t victims) {
        const int fromView = from ^ tp.toView;
        uint32_t feature = 0;
        while (victims) {
            const auto to = static_cast<uint8_t>(pieces::popLSB(victims));
            if (tp.tryFeature(relA, fromView, get(to), to, feature))
                add[na++] = NNUE::Accumulator::threatRow(feature);
        }
    };

    for (int side = 0; side < 2; ++side) {
        const int baseA = (side ^ p) * 6;
        for (uint64_t pcs = bb[PAWN][side]; pcs; ) {
            const auto from = static_cast<uint8_t>(pieces::popLSB(pcs));
            emit(baseA + (PAWN - 1), from, pieces::PAWN_ATTACKS[side][from] & pawnVictims);
        }
        for (uint64_t pcs = bb[KNIGHT][side]; pcs; ) {
            const auto from = static_cast<uint8_t>(pieces::popLSB(pcs));
            emit(baseA + (KNIGHT - 1), from, pieces::KNIGHT_ATTACKS[from] & occupancy);
        }
        for (uint64_t pcs = bb[BISHOP][side]; pcs; ) {
            const auto from = static_cast<uint8_t>(pieces::popLSB(pcs));
            emit(baseA + (BISHOP - 1), from, pieces::Bishop::attacks(from, occupancy) & nonQueenVictims);
        }
        for (uint64_t pcs = bb[ROOK][side]; pcs; ) {
            const auto from = static_cast<uint8_t>(pieces::popLSB(pcs));
            emit(baseA + (ROOK - 1), from, pieces::Rook::attacks(from, occupancy) & nonQueenVictims);
        }
        for (uint64_t pcs = bb[QUEEN][side]; pcs; ) {
            const auto from = static_cast<uint8_t>(pieces::popLSB(pcs));
            const uint64_t atks = pieces::Rook::attacks(from, occupancy) | pieces::Bishop::attacks(from, occupancy);
            emit(baseA + (QUEEN - 1), from, atks & occupancy);
        }
        if (bb[KING][side]) {
            const auto from = static_cast<uint8_t>(std::countr_zero(bb[KING][side]));
            emit(baseA + (KING - 1), from, pieces::KING_ATTACKS[from] & kingVictims);
        }
    }
    NNUE::Accumulator::applyThreatRows<false, false>(row, nullptr, nullptr, 0, add, na);
}

// From-scratch rebuild of the current frame, which becomes the oldest one of
// the line: a diff from the empty board, whose values are the bias.
// rebuildBitboardsFromSquares sets the bitboards directly (NOT through
// addPieceToBB), so bulk rebuilds land here once at the end.
void Board::refreshNnueAccumulator() noexcept {
    if (NNUE::activeNetwork == nullptr) return;
    accFloor = accTop;
    NNUE::Accumulator& acc = frame(accTop);
    acc.ndelta = 0;
    acc.nthreat = 0;
    // No kings (unit-test fragments, mid-load states): no values; evaluation
    // is guarded upstream by the kings-missing check.
    acc.dirty[0] = acc.dirty[1] = true;
    if (bb[KING][0] == 0 || bb[KING][1] == 0) return;
    for (int p = 0; p < 2; ++p) {
        const int view = ownKingView(*this, p);
        acc.base[p] = NNUE::kingFeatureBase(view);
        acc.flip[p] = static_cast<uint8_t>(NNUE::kingFlip(view));
        acc.dirty[p] = false;
        uint64_t empty[2][6] = {};
        std::memcpy(acc.v[p], NNUE::activeNetwork->l0b, sizeof(acc.v[p]));
        syncRow(*this, acc.v[p], empty, p, acc.base[p], acc.flip[p]);
        computeThreats(acc.threat[p], p, acc.flip[p]);
    }
}

// Computes the current frame. Its nearest computed ancestor, at most
// ACC_MAX_WALK moves back, is a valid start for a perspective whose king still
// stands on the same basis: the changes logged since sum to the feature
// difference whatever the king did in between. Rows that appear on both sides
// cancel, and int16 is modular, so the rest applies in any order. The same
// holds for the threats under the same flip alone, unless a move in between
// logged too many. A perspective without such a start is rebuilt as a diff
// against the Finny table, and its threats from nothing. Called from
// consistent-board points only (NNUE::evaluate, a copy, the selftest), never
// mid-doMove.
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
        kingView[p] = ownKingView(*this, p);
        rowBase[p] = NNUE::kingFeatureBase(kingView[p]);
        rowFlip[p] = NNUE::kingFlip(kingView[p]);
    }
    int from = -1;
    bool threatsLogged = top.nthreat != NNUE::ACC_THREATS_OVERFLOW;
    const int minFrom = std::max(accFloor, accTop - NNUE::ACC_MAX_WALK);
    for (int i = accTop - 1; i >= minFrom; --i) {
        const NNUE::Accumulator& f = frame(i);
        if (!f.dirty[0]) {
            from = i;
            break;
        }
        // Here, not in a loop of its own: clang vectorised that one.
        threatsLogged &= f.nthreat != NNUE::ACC_THREATS_OVERFLOW;
    }
    const int origin = from;
    bool usable[2] = {false, false};
    bool threatsUsable[2] = {false, false};
    if (from >= 0) {
        const NNUE::Accumulator& src = frame(from);
        for (int p = 0; p < 2; ++p) {
            threatsUsable[p] = threatsLogged && src.flip[p] == rowFlip[p];
            usable[p] = threatsUsable[p] && src.base[p] == rowBase[p];
        }
    }

    // Perspective p's threats of frame `target` from those of frame `start`.
    // A row added and removed on the way is 1 KiB not read twice.
    const auto applyThreatRange = [&]<int Cap>(int p, int start, int target) __attribute__((hot)) {
        const int8_t* sub[Cap];
        const int8_t* add[Cap + 8];
        int ns = 0;
        int na = 0;
        const NNUE::ThreatPerspective tp(p, rowFlip[p]);
        for (int i = start + 1; i <= target; ++i) {
            const NNUE::Accumulator& f = frame(i);
            for (int k = 0; k < f.nthreat; ++k) {
                const NNUE::ThreatDelta& d = f.threatDelta[k];
                uint32_t feature = 0;
                if (tp.tryFeature(d.attacker, d.from, d.victim, d.to(), feature)) {
                    const int8_t* row = NNUE::Accumulator::threatRow(feature);
                    (d.add() ? add[na++] : sub[ns++]) = row;
#if defined(__AVX2__)
                    // The row loads while the others are collected and cancelled.
                    _mm_prefetch(reinterpret_cast<const char*>(row), _MM_HINT_T0);
#endif
                }
            }
        }
        cancelAndApplyThreats(frame(target).threat[p], frame(start).threat[p], sub, ns, add, na);
    };
    const auto incrementalThreats = [&](int p, int start, int target) noexcept {
        if (target == start + 1) applyThreatRange.template operator()<NNUE::ACC_MAX_THREATS>(p, start, target);
        else applyThreatRange.template operator()<NNUE::ACC_MAX_WALK * NNUE::ACC_MAX_THREATS>(p, start, target);
    };

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
            const auto singlePass = [&]<int S, int A>() {
                const int16_t* subRow[2][2];
                const int16_t* addRow[2][2];
                for (int p = 0; p < 2; ++p) {
                    for (int k = 0; k < S; ++k) subRow[p][k] = row(p, sub[k]);
                    for (int k = 0; k < A; ++k) addRow[p][k] = row(p, add[k]);
                }
                NNUE::Accumulator::fusePair<S, A, true>(dst.v[0], dst.v[1], src.v[0], src.v[1], subRow, addRow);
            };
            if (ns == 0 && na == 0) {
                std::memcpy(dst.v, src.v, sizeof(dst.v));
            } else if (ns == 1 && na == 1) {
                singlePass.template operator()<1, 1>();
            } else if (ns == 2 && na == 1) {
                singlePass.template operator()<2, 1>();
            } else {
                fusePairFallback(dst, src, rowBase, rowFlip, sub, ns, add, na);
            }
        } else {
            const int p = usable[0] ? 0 : 1;
            const int16_t* subRow[CAP];
            const int16_t* addRow[CAP];
            for (int k = 0; k < ns; ++k) subRow[k] = row(p, sub[k]);
            for (int k = 0; k < na; ++k) addRow[k] = row(p, add[k]);
            NNUE::Accumulator::applyRows(dst.v[p], subRow, ns, addRow, na, src.v[p]);
        }
        if (usable[0] && usable[1] && target == start + 1) {
            applyThreatPair1(dst, src, rowFlip[0], rowFlip[1]);
        } else {
            for (int p = 0; p < 2; ++p)
                if (usable[p]) incrementalThreats(p, start, target);
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
        syncRow(*this, e.v, e.bb, p, rowBase[p], rowFlip[p]);
        std::memcpy(top.v[p], e.v, sizeof(e.v));
        if (threatsUsable[p]) incrementalThreats(p, origin, accTop);
        else computeThreats(top.threat[p], p, rowFlip[p]);
    }
}

} // namespace chess
