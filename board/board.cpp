#include <algorithm>
#include <type_traits>

#include "board.hpp"
#include "../tt/zobrist.hpp"

#if defined(__x86_64__) && defined(__GNUC__)
#include <cpuid.h>
#endif

namespace chess {

bool Board::onEfficiencyCore() noexcept {
#if defined(__x86_64__) && defined(__GNUC__)
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_max(0, nullptr) < 0x1A) return false;
    __cpuid_count(7, 0, a, b, c, d);
    if ((d & (1U << 15)) == 0) return false;   // not a hybrid CPU
    __cpuid_count(0x1A, 0, a, b, c, d);
    return (a >> 24) == 0x20;                  // Intel Atom core type
#else
    return false;
#endif
}

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

// The destination is in reach, and our king (on its new square if it is the
// mover) stands unattacked once the mover and its capture are lifted.
bool Board::isLegalPseudoMove(uint8_t fromIndex, uint8_t toIndex, uint8_t fromPiece) const noexcept {
    const uint8_t color = fromPiece & MASK_COLOR;
    const uint8_t dest = get(toIndex);
    if (dest != EMPTY && (dest & MASK_COLOR) == color) [[unlikely]] return false;
    const int side = colorToIndex(color);
    const uint64_t toBit = 1ULL << toIndex;
    uint64_t captured = (dest != EMPTY) ? toBit : 0ULL;
    uint64_t reach = 0ULL;
    switch (fromPiece & MASK_PIECE_TYPE) {
        case PAWN:
            reach = pieces::PAWN_ATTACKS[side][fromIndex];
            if (!(reach & toBit)) {
                reach = (dest == EMPTY) ? pieces::getPawnForwardPushes(fromIndex, color == WHITE, occupancy) : 0ULL;
            } else if (dest == EMPTY) {
                // A diagonal step to an empty square is en passant or nothing.
                if (toIndex != state.enPassant) return false;
                captured = 1ULL << enPassantVictim(fromPiece, toIndex);
            }
            break;
        case KNIGHT: reach = pieces::KNIGHT_ATTACKS[fromIndex]; break;
        case BISHOP: reach = pieces::Bishop::attacks(fromIndex, occupancy); break;
        case ROOK:   reach = pieces::Rook::attacks(fromIndex, occupancy); break;
        case QUEEN:  reach = pieces::Rook::attacks(fromIndex, occupancy) | pieces::Bishop::attacks(fromIndex, occupancy); break;
        case KING: {
            // Only castling moves the king two files: a one-step offset is 1, 7, 8 or 9.
            const int diff = static_cast<int>(toIndex) - fromIndex;
            if (diff == 2 || diff == -2) [[unlikely]] {
                const int right = 2 * side + (diff < 0);
                return fromIndex == CASTLINGS[right].king && canCastle(right);
            }
            reach = pieces::KING_ATTACKS[fromIndex];
            break;
        }
        default: return false;
    }
    if (!(reach & toBit) || !bb[KING][side]) [[unlikely]] return false;
    const uint64_t occNew = (occupancy & ~(1ULL << fromIndex) & ~captured) | toBit;
    const auto king = static_cast<uint8_t>((fromPiece & MASK_PIECE_TYPE) == KING ? toIndex : std::countr_zero(bb[KING][side]));
    return (attackersOf(king, side ^ 1, occNew) & ~captured) == 0;
}

uint64_t Board::checkersTo(uint8_t color) const noexcept {
    const uint8_t side = colorToIndex(color);
    if (!bb[KING][side]) [[unlikely]] return 0ULL;
    return attackersOf(std::countr_zero(bb[KING][side]), side ^ 1, occupancy);
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
inline void syncRow(const Board& b, int16_t* row, uint64_t (&cached)[2][6], int p, int rowBase, int rowFlip,
                    bool prefetch) noexcept {
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
    if (prefetch) NNUE::Accumulator::applyRows<true>(row, sub, ns, add, na);
    else          NNUE::Accumulator::applyRows<false>(row, sub, ns, add, na);
}

// Drops the entries that appear in both lists: they cancel, in any order.
template<typename T>
inline void cancelPairs(T* sub, int& ns, T* add, int& na) noexcept {
    for (int i = 0; i < ns; ) {
        int j = 0;
        while (j < na && add[j] != sub[i]) ++j;
        if (j == na) { ++i; continue; }
        sub[i] = sub[--ns];
        add[j] = add[--na];
    }
}

// The same for threat rows, eight at a time; add needs 8 spare slots past na.
// Out of line: every caller shares one copy.
__attribute__((noinline, hot))
void cancelThreats(const int8_t** sub, int& ns, const int8_t** add, int& na) noexcept {
#if defined(__AVX2__)
    if (ns == 0 || na == 0) return;
    // A fixed-size store: clang turns a variable-length one into a memset call.
    std::memset(add + na, 0, 8 * sizeof(const int8_t*));
    for (int i = 0; i < ns; ) {
        const __m256i key = _mm256_set1_epi64x(reinterpret_cast<int64_t>(sub[i]));
        const auto matches = [&](const int8_t** at) noexcept {
            return _mm256_movemask_pd(_mm256_castsi256_pd(
                _mm256_cmpeq_epi64(key, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(at)))));
        };
        int hit = -1;
        for (int base = 0; base < na && hit < 0; base += 8)
            if (const int mask = matches(add + base) | (matches(add + base + 4) << 4))
                hit = base + std::countr_zero(static_cast<unsigned>(mask));
        if (hit < 0) { ++i; continue; }
        sub[i] = sub[--ns];
        add[hit] = add[--na];
        add[na] = nullptr;
        if (na == 0) break;
    }
#else
    cancelPairs(sub, ns, add, na);
#endif
}

template<bool Prefetch>
__attribute__((always_inline))
inline void applyMixedShape(int16_t* __restrict dst, const int16_t* __restrict src,
                            const int16_t* const* psub, int pns, const int16_t* const* padd, int pna,
                            const int8_t* const* tsub, int tns, const int8_t* const* tadd, int tna) noexcept {
    using Acc = NNUE::Accumulator;
    switch (pns * 4 + pna) {
        case 1 * 4 + 1: Acc::applyMixedRows<1, 1, Prefetch>(dst, src, psub, padd, tsub, tns, tadd, tna); return;
        case 2 * 4 + 1: Acc::applyMixedRows<2, 1, Prefetch>(dst, src, psub, padd, tsub, tns, tadd, tna); return;
        case 2 * 4 + 2: Acc::applyMixedRows<2, 2, Prefetch>(dst, src, psub, padd, tsub, tns, tadd, tna); return;
        case 0:         Acc::applyMixedRows<0, 0, Prefetch>(dst, src, psub, padd, tsub, tns, tadd, tna); return;
        default:
            Acc::applyMixedRows<0, 0, Prefetch>(dst, src, psub, padd, tsub, tns, tadd, tna);
            Acc::applyRows<Prefetch>(dst, psub, pns, padd, pna);
    }
}

// dst = src - psub + padd - tsub + tadd (piece rows i16, threat rows i8) in
// one pass when the piece rows fit the kernel's shapes. Out of line: one copy
// of the kernels per prefetch mode.
__attribute__((noinline, hot))
void applyMixed(int16_t* __restrict dst, const int16_t* __restrict src,
                const int16_t* const* psub, int pns, const int16_t* const* padd, int pna,
                const int8_t* const* tsub, int tns, const int8_t* const* tadd, int tna, bool prefetch) noexcept {
    if (pns + pna + tns + tna == 0) {
        std::memcpy(dst, src, NNUE::HIDDEN * sizeof(int16_t));
        return;
    }
    if (prefetch) applyMixedShape<true>(dst, src, psub, pns, padd, pna, tsub, tns, tadd, tna);
    else          applyMixedShape<false>(dst, src, psub, pns, padd, pna, tsub, tns, tadd, tna);
}

// Both perspectives of dst from src, one move apart: the move logged in
// `moved` leads from src to dst, or from dst to src with invert. Its piece
// rows are given. One move rarely adds and removes the same threat, so
// nothing is cancelled.
template<bool Invert>
__attribute__((noinline, hot))
void applyFramePair1(NNUE::Accumulator& dst, const NNUE::Accumulator& src, const NNUE::Accumulator& moved,
                         int flip0, int flip1,
                         const int16_t* const (&psub)[2][NNUE::ACC_MAX_DELTAS], int pns,
                         const int16_t* const (&padd)[2][NNUE::ACC_MAX_DELTAS], int pna, bool prefetch) noexcept {
    const int8_t* sub0Buf[NNUE::ACC_MAX_THREATS];
    const int8_t* add0Buf[NNUE::ACC_MAX_THREATS];
    const int8_t* sub1Buf[NNUE::ACC_MAX_THREATS];
    const int8_t* add1Buf[NNUE::ACC_MAX_THREATS];
    const int8_t** sub0 = sub0Buf;
    const int8_t** add0 = add0Buf;
    const int8_t** sub1 = sub1Buf;
    const int8_t** add1 = add1Buf;

    using Acc = NNUE::Accumulator;
    // Prefetch the piece weight rows (6 MiB l0w) while the threat rows are resolved.
    for (int k = 0; k < pns; ++k) Acc::prefetchLines<1>(psub[0][k]), Acc::prefetchLines<1>(psub[1][k]);
    for (int k = 0; k < pna; ++k) Acc::prefetchLines<1>(padd[0][k]), Acc::prefetchLines<1>(padd[1][k]);

    // Hoist l0t base so stores of const int8_t* do not force clang to reload activeNetwork->l0t.
    const int8_t* const l0tBase = NNUE::activeNetwork->l0t.get();
    const auto push = [&](const int8_t**& out, uint32_t entry, int to) noexcept {
        uint32_t feat;
        if (!NNUE::threats::tryIndexBase(entry, to, feat)) return;
        const int8_t* row = l0tBase + static_cast<size_t>(feat) * NNUE::HIDDEN;
        *out++ = row;
        Acc::prefetchThreatRow(row, prefetch);
    };
    const int toView0 = 56 ^ flip0;
    const int nth = moved.nthreat;
    for (int k = 0; k < nth; ++k) {
        const NNUE::ThreatDelta d = moved.threatDelta[k];
        const uint32_t bases = NNUE::BOTH_PAIR_BASE[d.attacker & 0xf][d.victim & 0xf];
        const bool isAdd = Invert ? !d.add() : d.add();
        push(isAdd ? add0 : sub0, (bases & 0xffffu) + (d.from ^ toView0), d.to() ^ toView0);
        push(isAdd ? add1 : sub1, (bases >> 16) + (d.from ^ flip1), d.to() ^ flip1);
    }
    applyMixed(dst.v[0], src.v[0], psub[0], pns, padd[0], pna,
               sub0Buf, static_cast<int>(sub0 - sub0Buf), add0Buf, static_cast<int>(add0 - add0Buf), prefetch);
    applyMixed(dst.v[1], src.v[1], psub[1], pns, padd[1], pna,
               sub1Buf, static_cast<int>(sub1 - sub1Buf), add1Buf, static_cast<int>(add1 - add1Buf), prefetch);
}

// The king's move among a frame's deltas (a move makes at most one), or null.
const NNUE::AccDelta* kingDelta(const NNUE::Accumulator& f) noexcept {
    for (int k = 0; k < f.ndelta; ++k)
        if ((f.delta[k].piece & Board::MASK_PIECE_TYPE) == Board::KING) return &f.delta[k];
    return nullptr;
}

} // namespace

void Board::computeThreats(int16_t* row, const int16_t* src, int p, int flip) const noexcept {
    const int8_t* add[NNUE::MAX_ACTIVE_THREATS];
    int na = 0;
    const NNUE::ThreatPerspective tp(p, flip);
    const uint64_t queens = bb[QUEEN][0] | bb[QUEEN][1];
    const uint64_t kings  = bb[KING][0] | bb[KING][1];
    const int8_t* const l0tBase = NNUE::activeNetwork->l0t.get();
    // Every piece of `side` and Type, against the victims its type can have.
    const auto emit = [&]<uint8_t Type>(int side, uint64_t victims) {
        for (uint64_t pcs = bb[Type][side]; pcs; ) {
            const auto from = static_cast<uint8_t>(pieces::popLSB(pcs));
            const int fromView = from ^ tp.toView;
            uint64_t targets = victims & (Type == PAWN ? pieces::PAWN_ATTACKS[side][from]
                                                       : pieces::generateMovesByType<Type>(from, occupancy));
            uint32_t feature = 0;
            while (targets) {
                const auto to = static_cast<uint8_t>(pieces::popLSB(targets));
                if (tp.tryFeature((side ^ p) * 6 + Type - 1, fromView, get(to), to, feature)) {
                    add[na] = l0tBase + static_cast<size_t>(feature) * NNUE::HIDDEN;
                    if (weightPrefetch) NNUE::Accumulator::prefetchLines<2>(add[na]);
                    ++na;
                }
            }
        }
    };
    for (int side = 0; side < 2; ++side) {
        const uint64_t validOcc = occupancy ^ bb[KING][side ^ 1];
        emit.template operator()<PAWN>(side, occupancy ^ (bb[BISHOP][0] | bb[BISHOP][1] | queens | kings));
        emit.template operator()<KNIGHT>(side, validOcc);
        emit.template operator()<BISHOP>(side, validOcc ^ queens);
        emit.template operator()<ROOK>(side, validOcc ^ queens);
        emit.template operator()<QUEEN>(side, validOcc);
        emit.template operator()<KING>(side, occupancy ^ (queens | kings));
    }
    applyMixed(row, src, nullptr, 0, nullptr, 0, nullptr, 0, add, na, weightPrefetch);
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
        alignas(64) int16_t pieceRow[NNUE::HIDDEN];
        std::memcpy(pieceRow, NNUE::activeNetwork->l0b, sizeof(pieceRow));
        syncRow(*this, pieceRow, empty, p, acc.base[p], acc.flip[p], weightPrefetch);
        computeThreats(acc.v[p], pieceRow, p, acc.flip[p]);
    }
}

// Computes the current frame. Its nearest computed ancestor, at most
// ACC_MAX_WALK moves back, is a valid start for a perspective whose king still
// stands on the same basis, unless a move in between logged too many threats:
// the changes logged since sum to the feature difference whatever the king did
// in between. Rows that appear on both sides cancel, and int16 is modular, so
// the rest applies in any order. A perspective without such a start is
// rebuilt from the Finny table and its threats from nothing (cheaper than
// carrying them over a change of bucket: the old basis's Finny entry is
// rarely close to the ancestor's pieces). Called from consistent-board points
// only (NNUE::evaluate, a copy, the selftest), never mid-doMove.
__attribute__((hot, noinline))
void Board::ensureNnueAccumulatorCleanSlow() const noexcept {
    NNUE::Accumulator& top = frame(accTop);
    if (bb[KING][0] == 0 || bb[KING][1] == 0) [[unlikely]] return;

    int kingView[2];
    int rowBase[2];
    int rowFlip[2];
    for (int p = 0; p < 2; ++p) {
        kingView[p] = ownKingView(*this, p);
        rowBase[p] = NNUE::kingFeatureBase(kingView[p]);
        rowFlip[p] = NNUE::kingFlip(kingView[p]);
    }
    // Marks both perspectives of a frame computed on this basis.
    const auto stamp = [&](NNUE::Accumulator& f) noexcept {
        for (int p = 0; p < 2; ++p) {
            f.base[p] = rowBase[p];
            f.flip[p] = static_cast<uint8_t>(rowFlip[p]);
            f.dirty[p] = false;
        }
    };
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
    bool usable[2] = {false, false};
    if (from >= 0 && threatsLogged) {
        const NNUE::Accumulator& src = frame(from);
        for (int p = 0; p < 2; ++p) usable[p] = src.flip[p] == rowFlip[p] && src.base[p] == rowBase[p];
    }

    // Frame dst from frame src one move away, both perspectives on this basis;
    // `moved` holds the move (dst's own, or src's with invert). No single move
    // logs cancelling piece deltas (promotions emit Remove+Add directly), so
    // piece rows map straight from moved.delta without a cancel.
    const auto pair1FromDeltas = [&](NNUE::Accumulator& dst, const NNUE::Accumulator& src,
                                     const NNUE::Accumulator& moved, auto invert) noexcept {
        const int16_t* subRow[2][NNUE::ACC_MAX_DELTAS];
        const int16_t* addRow[2][NNUE::ACC_MAX_DELTAS];
        int ns = 0;
        int na = 0;
        const auto pushPiece = [&](bool isAdd, uint8_t piece, uint8_t sq) noexcept {
            const int idx = isAdd ? na++ : ns++;
            auto& arr = isAdd ? addRow : subRow;
            arr[0][idx] = NNUE::Accumulator::rowFor(rowBase[0], rowFlip[0], 0, piece, sq);
            arr[1][idx] = NNUE::Accumulator::rowFor(rowBase[1], rowFlip[1], 1, piece, sq);
        };
        const int nd = moved.ndelta;
#if defined(__clang__)
        #pragma clang loop vectorize(disable) unroll(disable)
#endif
        for (int k = 0; k < nd; ++k) moved.delta[k].forEachChange(invert, pushPiece);
        applyFramePair1<decltype(invert)::value>(dst, src, moved, rowFlip[0], rowFlip[1], subRow, ns, addRow, na,
                                                 weightPrefetch);
        stamp(dst);
    };

    // Multi-move or single-perspective incremental update: kept out of line so
    // its ~9 KiB threat-row scratch arrays are never allocated on the 1-move fast path.
    const auto incrementalMulti = [&](int start, int target) __attribute__((noinline)) {
        constexpr int CAP = NNUE::ACC_MAX_WALK * NNUE::ACC_MAX_DELTAS;
        uint16_t sub[CAP];
        uint16_t add[CAP];
        int ns = 0;
        int na = 0;
        for (int i = start + 1; i <= target; ++i) {
            const NNUE::Accumulator& f = frame(i);
            for (int k = 0; k < f.ndelta; ++k)
                f.delta[k].forEachChange(false, [&](bool isAdd, uint8_t piece, uint8_t sq) noexcept {
                    (isAdd ? add[na++] : sub[ns++]) = static_cast<uint16_t>(piece << 8 | sq);
                });
        }
        cancelPairs(sub, ns, add, na);
        const NNUE::Accumulator& src = frame(start);
        NNUE::Accumulator& dst = frame(target);
        for (int p = 0; p < 2; ++p) {
            if (!usable[p]) continue;
            const auto rows = [&](const uint16_t* keys, int n, const int16_t** out) noexcept {
#if defined(__clang__)
                #pragma clang loop vectorize(disable)
#endif
                for (int k = 0; k < n; ++k)
                    out[k] = NNUE::Accumulator::rowFor(rowBase[p], rowFlip[p], p, static_cast<uint8_t>(keys[k] >> 8),
                                                       static_cast<uint8_t>(keys[k]));
            };
            const int16_t* subRow[CAP];
            const int16_t* addRow[CAP];
            rows(sub, ns, subRow);
            rows(add, na, addRow);
            const int8_t* tsub[NNUE::ACC_MAX_WALK * NNUE::ACC_MAX_THREATS];
            const int8_t* tadd[NNUE::ACC_MAX_WALK * NNUE::ACC_MAX_THREATS + 8];
            int tns = 0;
            int tna = 0;
            const int8_t* const l0tBase = NNUE::activeNetwork->l0t.get();
            const NNUE::ThreatPerspective tp(p, rowFlip[p]);
            for (int i = start + 1; i <= target; ++i) {
                const NNUE::Accumulator& f = frame(i);
                for (int k = 0; k < f.nthreat; ++k) {
                    const NNUE::ThreatDelta& d = f.threatDelta[k];
                    uint32_t feature = 0;
                    if (tp.tryFeature(d.attacker, d.from, d.victim, d.to(), feature)) {
                        const int8_t* r = l0tBase + static_cast<size_t>(feature) * NNUE::HIDDEN;
                        (d.add() ? tadd[tna++] : tsub[tns++]) = r;
                        NNUE::Accumulator::prefetchThreatRow(r, weightPrefetch);
                    }
                }
            }
            cancelThreats(tsub, tns, tadd, tna);
            applyMixed(dst.v[p], src.v[p], subRow, ns, addRow, na, tsub, tns, tadd, tna, weightPrefetch);
        }
    };

    if (usable[0] || usable[1]) {
        // Every frame on the way is computed too, one move at a time, back to
        // the last king move (an older frame stands on another basis): the
        // siblings and cousins that follow start one move away.
        int first = accTop;
        if (usable[0] && usable[1])
            while (first - 1 > from && kingDelta(frame(first)) == nullptr) --first;
        for (int i = first; i <= accTop; ++i) {
            if (usable[0] && usable[1] && i == from + 1) {
                pair1FromDeltas(frame(i), frame(from), frame(i), std::false_type{});
            } else {
                incrementalMulti(from, i);
                stamp(frame(i));
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
    stamp(top);
    for (int p = 0; p < 2; ++p) {
        if (usable[p]) continue;
        finny.ensureInitialised();
        NNUE::FinnyEntry& e = finny.entry[p][NNUE::KING_BUCKET_MAP[kingView[p]]][rowFlip[p] ? 1 : 0];
        syncRow(*this, e.v, e.bb, p, rowBase[p], rowFlip[p], weightPrefetch);
        computeThreats(top.v[p], e.v, p, rowFlip[p]);
    }

    // The ancestors of a frame refreshed here stay uncomputed, and every later
    // sibling or cousin would refresh too: compute them back from this frame
    // while the moves kept both bases.
    for (int i = accTop; i - 1 >= accFloor && i > accTop - NNUE::ACC_MAX_WALK; --i) {
        NNUE::Accumulator& child = frame(i);
        if (!frame(i - 1).dirty[0] || child.nthreat == NNUE::ACC_THREATS_OVERFLOW) return;
        if (const NNUE::AccDelta* king = kingDelta(child)) {
            const int p = colorToIndex(king->piece);
            const int view = (p == 1) ? king->from : (king->from ^ 56);
            if (NNUE::kingFeatureBase(view) != rowBase[p] || NNUE::kingFlip(view) != rowFlip[p]) return;
        }
        pair1FromDeltas(frame(i - 1), child, child, std::true_type{});
    }
}

} // namespace chess
