#include "board.hpp"
#include "../tt/zobrist.hpp"

namespace chess {

namespace {

// The castling rights a move from or to each square keeps: moving the king or
// a rook off its home square, or capturing on a rook's home square, gives the
// matching rights up.
constexpr std::array<uint8_t, 64> CASTLING_KEEP = [] {
    std::array<uint8_t, 64> keep{};
    keep.fill(Board::CASTLING_RIGHTS_ALL);
    for (int right = 0; right < 4; ++right) {
        keep[Board::CASTLINGS[right].king] &= static_cast<uint8_t>(~(1 << right));
        keep[Board::CASTLINGS[right].rook] &= static_cast<uint8_t>(~(1 << right));
    }
    return keep;
}();

// BEYOND_BY_SQ[b][a]: the squares past b on the line from a through b, 0 when
// the two are not aligned. Indexed by b first so emitThreatDeltas walks a
// single 512-byte slice for fixed sq instead of striding 512 bytes per slider.
constexpr std::array<std::array<uint64_t, 64>, 64> BEYOND_BY_SQ = [] {
    std::array<std::array<uint64_t, 64>, 64> lut{};
    for (int a = 0; a < 64; ++a)
        for (int b = 0; b < 64; ++b) {
            const int df = chess::file(b) - chess::file(a), dr = chess::rank(b) - chess::rank(a);
            if (a == b || (df != 0 && dr != 0 && df != dr && df != -dr)) continue;
            const int sf = (df > 0) - (df < 0), sr = (dr > 0) - (dr < 0);
            for (int f = chess::file(b) + sf, r = chess::rank(b) + sr; f >= 0 && f < 8 && r >= 0 && r < 8; f += sf, r += sr)
                lut[b][a] |= 1ULL << (r * 8 + f);
        }
    return lut;
}();

} // namespace

__attribute__((always_inline))
inline void Board::set(uint8_t index, piece_id value) noexcept {
    chessboard[index] = static_cast<uint8_t>(value & MASK_PIECE);
}

// A new frame for the position doMove is about to make: computed later, from
// the changes the hooks log into it. A long game wraps the ring; only frames
// still inside it stay valid starts.
__attribute__((always_inline))
inline NNUE::Accumulator& Board::pushAccFrame() noexcept {
    NNUE::Accumulator& f = frame(++accTop);
    f.ndelta = 0;
    f.nthreat = 0;
    f.dirty[0] = f.dirty[1] = true;
    if (accTop - accFloor >= NNUE::ACC_FRAMES) accFloor = accTop - NNUE::ACC_FRAMES + 1;
    return f;
}

__attribute__((always_inline))
inline void Board::recordAcc(NNUE::Accumulator& f, NNUE::AccDelta delta) noexcept {
    f.delta[f.ndelta++] = delta;
}

// Its attacks, the attacks on it, and the lines of the sliders that see sq,
// which reach past sq exactly when the piece is not there. The masks leave out
// the pairs that are never a feature.
namespace {

struct ThreatMasks {
    uint64_t victims;
    uint64_t attackers;
};

// What any piece on sq shares, computed once for both pieces of a replacement.
struct SquareThreatInputs {
    uint64_t occupancy;
    const std::array<std::array<uint64_t, 2>, 7>& bb;
    uint8_t sq;
    uint64_t rookOcc;
    uint64_t bishopOcc;
    uint64_t queens;
    uint64_t kings;
    uint64_t sliders;
    uint64_t knightAtks;
    uint64_t slidersAndKnights;

    __attribute__((always_inline))
    static SquareThreatInputs collect(uint8_t sq, uint64_t occupancy,
                                      const std::array<std::array<uint64_t, 2>, 7>& bb) noexcept {
        const uint64_t rookOcc    = pieces::Rook::attacks(sq, occupancy) & occupancy;
        const uint64_t bishopOcc  = pieces::Bishop::attacks(sq, occupancy) & occupancy;
        const uint64_t queens     = bb[Board::QUEEN][0] | bb[Board::QUEEN][1];
        const uint64_t kings      = bb[Board::KING][0]  | bb[Board::KING][1];
        const uint64_t sliders    = (rookOcc   & (bb[Board::ROOK][0]   | bb[Board::ROOK][1]   | queens))
                                  | (bishopOcc & (bb[Board::BISHOP][0] | bb[Board::BISHOP][1] | queens));
        const uint64_t knightAtks = pieces::KNIGHT_ATTACKS[sq] & (bb[Board::KNIGHT][0] | bb[Board::KNIGHT][1]);
        return {occupancy, bb, sq, rookOcc, bishopOcc, queens, kings, sliders,
                knightAtks, sliders | knightAtks};
    }

    __attribute__((always_inline))
    uint64_t pawnAtks() const noexcept {
        return (pieces::PAWN_ATTACKS[1][sq] & bb[Board::PAWN][0])
             | (pieces::PAWN_ATTACKS[0][sq] & bb[Board::PAWN][1]);
    }

    __attribute__((always_inline))
    uint64_t kingAtks() const noexcept {
        return pieces::KING_ATTACKS[sq] & kings;
    }

    __attribute__((always_inline))
    ThreatMasks masksFor(uint8_t piece) const noexcept {
        // Opponent king bitboard: (WHITE & MASK_COLOR) >> 3 == 1 (BLACK), and vice versa.
        const auto oppKing = [&]() noexcept { return bb[Board::KING][(piece & Board::MASK_COLOR) >> 3]; };
        switch (piece & Board::MASK_PIECE_TYPE) {
            case Board::PAWN: {
                const uint64_t nonTargets = bb[Board::BISHOP][0] | bb[Board::BISHOP][1] | queens | kings;
                return {pieces::PAWN_ATTACKS[Board::colorToIndex(piece)][sq] & (occupancy ^ nonTargets),
                        slidersAndKnights | pawnAtks() | kingAtks()};
            }
            case Board::KNIGHT:
                return {pieces::KNIGHT_ATTACKS[sq] & (occupancy ^ oppKing()),
                        slidersAndKnights | pawnAtks() | kingAtks()};
            case Board::BISHOP:
                return {bishopOcc & ~(queens | oppKing()),
                        slidersAndKnights | kingAtks()};
            case Board::ROOK:
                return {rookOcc & ~(queens | oppKing()),
                        slidersAndKnights | pawnAtks() | kingAtks()};
            case Board::QUEEN: {
                const uint64_t occRays = rookOcc | bishopOcc;
                return {occRays & ~oppKing(), (occRays & queens) | knightAtks};
            }
            default: { // KING
                // Checks (* -> opp K) are excluded, so only friendly sliders/knights defending own K remain.
                const int c = Board::colorToIndex(piece);
                const uint64_t ownMinorMajor = bb[Board::KNIGHT][c] | bb[Board::BISHOP][c]
                                             | bb[Board::ROOK][c]   | bb[Board::QUEEN][c];
                return {pieces::KING_ATTACKS[sq] & (occupancy ^ (queens | kings)),
                        slidersAndKnights & ownMinorMajor};
            }
        }
    }
};

} // namespace

template<bool Add, bool Checked>
inline void Board::emitThreatDeltas(NNUE::Accumulator& f, uint8_t startN, uint8_t piece, uint8_t sq,
                                    uint64_t victims, uint64_t attackers, uint64_t sliders,
                                    uint64_t sliderOcc) const noexcept {
    uint8_t n = startN;
    const auto push = [&f, &n](uint8_t attacker, uint8_t from, uint8_t victim, uint8_t to, bool add) {
        if constexpr (Checked) {
            if (n >= NNUE::ACC_MAX_THREATS) { n = NNUE::ACC_THREATS_OVERFLOW; return; }
        }
        f.threatDelta[n++] = NNUE::ThreatDelta::make(attacker, from, victim, to, add);
    };
    while (victims) {
        const auto to = static_cast<uint8_t>(pieces::popLSB(victims));
        push(piece, sq, get(to), to, Add);
    }
    while (attackers) {
        const auto from = static_cast<uint8_t>(pieces::popLSB(attackers));
        push(get(from), from, piece, sq, Add);
    }
    // A line past sq meets rookOcc only if orthogonal, bishopOcc only if
    // diagonal: one mask serves both.
    const uint64_t* const beyondSq = BEYOND_BY_SQ[sq].data();
    while (sliders) {
        const auto from = static_cast<uint8_t>(pieces::popLSB(sliders));
        if (const uint64_t past = sliderOcc & beyondSq[from]) {
            const auto to = static_cast<uint8_t>(std::countr_zero(past));
            const uint8_t att = get(from);
            const uint8_t vic = get(to);
            if (NNUE::threatPairIsFeature(att, vic)) push(att, from, vic, to, !Add);
        }
    }
    f.nthreat = n;
}

template<bool Checked>
__attribute__((always_inline))
inline void Board::emitReplacedThreatDeltas(NNUE::Accumulator& f, uint8_t startN, uint8_t oldPiece,
                                            uint8_t newPiece, uint8_t sq, uint64_t oldVictims,
                                            uint64_t newVictims, uint64_t oldAttackers,
                                            uint64_t newAttackers) const noexcept {
    uint8_t n = startN;
    const auto push = [&f, &n](uint8_t attacker, uint8_t from, uint8_t victim, uint8_t to, bool add) {
        if constexpr (Checked) {
            if (n >= NNUE::ACC_MAX_THREATS) { n = NNUE::ACC_THREATS_OVERFLOW; return; }
        }
        f.threatDelta[n++] = NNUE::ThreatDelta::make(attacker, from, victim, to, add);
    };
    const auto emitVictims = [&](uint64_t mask, uint8_t att, bool add) {
        while (mask) {
            const auto to = static_cast<uint8_t>(pieces::popLSB(mask));
            push(att, sq, get(to), to, add);
        }
    };
    const auto emitAttackers = [&](uint64_t mask, uint8_t vic, bool add) {
        while (mask) {
            const auto from = static_cast<uint8_t>(pieces::popLSB(mask));
            push(get(from), from, vic, sq, add);
        }
    };
    emitVictims(oldVictims, oldPiece, false);
    emitVictims(newVictims, newPiece, true);
    uint64_t shared = oldAttackers & newAttackers;
    emitAttackers(oldAttackers ^ shared, oldPiece, false);
    emitAttackers(newAttackers ^ shared, newPiece, true);
    while (shared) {
        const auto from = static_cast<uint8_t>(pieces::popLSB(shared));
        const uint8_t att = get(from);
        push(att, from, oldPiece, sq, false);
        push(att, from, newPiece, sq, true);
    }
    f.nthreat = n;
}

template<bool Add>
__attribute__((noinline))
void Board::recordThreats(NNUE::Accumulator& f, uint8_t piece, uint8_t sq) noexcept {
    const uint8_t startN = f.nthreat;

    const SquareThreatInputs in = SquareThreatInputs::collect(sq, occupancy, bb);
    const auto [victims, attackers] = in.masksFor(piece);
    const uint64_t sliderOcc = in.rookOcc | in.bishopOcc;

    // At most 8 victims, 19 attackers and 8 lines: 35 deltas.
    if (startN <= NNUE::ACC_MAX_THREATS - 35) [[likely]] {
        emitThreatDeltas<Add, false>(f, startN, piece, sq, victims, attackers, in.sliders, sliderOcc);
        return;
    }
    const int worstCase = static_cast<int>(startN)
        + std::popcount(victims)
        + std::popcount(attackers)
        + std::popcount(in.sliders);
    if (worstCase <= NNUE::ACC_MAX_THREATS) {
        emitThreatDeltas<Add, false>(f, startN, piece, sq, victims, attackers, in.sliders, sliderOcc);
    } else if (startN != NNUE::ACC_THREATS_OVERFLOW) [[unlikely]] {
        emitThreatDeltas<Add, true>(f, startN, piece, sq, victims, attackers, in.sliders, sliderOcc);
    }
}

__attribute__((noinline))
void Board::recordReplacedThreats(NNUE::Accumulator& f, uint8_t oldPiece, uint8_t newPiece, uint8_t sq) noexcept {
    const uint8_t startN = f.nthreat;

    const SquareThreatInputs in = SquareThreatInputs::collect(sq, occupancy, bb);
    const auto [oldVictims, oldAttackers] = in.masksFor(oldPiece);
    const auto [newVictims, newAttackers] = in.masksFor(newPiece);

    // At most 2 * (8 + 19) = 54 deltas.
    if (startN <= NNUE::ACC_MAX_THREATS - 54) [[likely]] {
        emitReplacedThreatDeltas<false>(f, startN, oldPiece, newPiece, sq,
                                        oldVictims, newVictims, oldAttackers, newAttackers);
        return;
    }
    const int worstCase = static_cast<int>(startN)
        + std::popcount(oldVictims) + std::popcount(newVictims)
        + std::popcount(oldAttackers) + std::popcount(newAttackers);
    if (worstCase <= NNUE::ACC_MAX_THREATS) {
        emitReplacedThreatDeltas<false>(f, startN, oldPiece, newPiece, sq,
                                        oldVictims, newVictims, oldAttackers, newAttackers);
    } else if (startN != NNUE::ACC_THREATS_OVERFLOW) [[unlikely]] {
        emitReplacedThreatDeltas<true>(f, startN, oldPiece, newPiece, sq,
                                       oldVictims, newVictims, oldAttackers, newAttackers);
    }
}

template<bool Record>
__attribute__((always_inline))
inline void Board::addPieceToBB(uint8_t piece, uint8_t index, NNUE::Accumulator* f) noexcept {
    bb[piece & MASK_PIECE_TYPE][colorToIndex(piece)] |= 1ULL << index;
    if constexpr (Record) recordAcc(*f, {NNUE::AccDelta::Add, piece, index, 0});
}

template<bool Record>
__attribute__((always_inline))
inline void Board::removePieceFromBB(uint8_t piece, uint8_t index, NNUE::Accumulator* f) noexcept {
    bb[piece & MASK_PIECE_TYPE][colorToIndex(piece)] &= ~(1ULL << index);
    if constexpr (Record) recordAcc(*f, {NNUE::AccDelta::Remove, piece, index, 0});
}

// Moves a piece between squares: mailbox, occupancy, its bitboard (which never
// holds the destination, so one XOR does both ends) and, in doMove, the frame.
template<bool Record>
__attribute__((always_inline))
inline void Board::relocate(uint8_t piece, uint8_t fromIndex, uint8_t toIndex, NNUE::Accumulator* f) noexcept {
    set(toIndex, static_cast<piece_id>(piece));
    set(fromIndex, EMPTY);
    occupancy = (occupancy | (1ULL << toIndex)) & ~(1ULL << fromIndex);
    bb[piece & MASK_PIECE_TYPE][colorToIndex(piece)] ^= (1ULL << fromIndex) | (1ULL << toIndex);
    if constexpr (Record) recordAcc(*f, {NNUE::AccDelta::Move, piece, fromIndex, toIndex});
}

inline Board::MoveKind Board::classifyMoveKind(uint8_t movingType, uint8_t movingColor, uint8_t fromIndex,
                                               uint8_t toIndex, uint8_t destBefore, const Square& prevEnPassant) noexcept {
    const MoveKind plain = (destBefore != EMPTY) ? MoveKind::Capture : MoveKind::Quiet;
    if (movingType == KING) {
        const int df = chess::file(toIndex) - chess::file(fromIndex);
        return (chess::rank(fromIndex) == chess::rank(toIndex) && (df == 2 || df == -2)) ? MoveKind::Castling : plain;
    }
    if (movingType != PAWN) return plain;
    if (chess::file(fromIndex) != chess::file(toIndex) && destBefore == EMPTY
        && isValidSquare(prevEnPassant) && toIndex == prevEnPassant) {
        return MoveKind::EnPassant;
    }
    const uint8_t toRank = chess::rank(toIndex);
    if (toRank == promotionRank(movingColor == WHITE)) {
        return (destBefore != EMPTY) ? MoveKind::PromotionCapture : MoveKind::PromotionQuiet;
    }
    const int dr = toRank - chess::rank(fromIndex);
    return (dr == 2 || dr == -2) ? MoveKind::DoublePawnPush : plain;
}

// Runs f.template operator()<Kind>() for the move kind known at runtime.
template<typename F>
__attribute__((always_inline))
inline void Board::visitKind(MoveKind kind, F&& f) noexcept {
    switch (kind) {
        case MoveKind::Capture:          f.template operator()<MoveKind::Capture>(); break;
        case MoveKind::DoublePawnPush:   f.template operator()<MoveKind::DoublePawnPush>(); break;
        case MoveKind::EnPassant:        f.template operator()<MoveKind::EnPassant>(); break;
        case MoveKind::Castling:         f.template operator()<MoveKind::Castling>(); break;
        case MoveKind::PromotionQuiet:   f.template operator()<MoveKind::PromotionQuiet>(); break;
        case MoveKind::PromotionCapture: f.template operator()<MoveKind::PromotionCapture>(); break;
        default:                         f.template operator()<MoveKind::Quiet>(); break;
    }
}

__attribute__((always_inline))
inline void Board::prepareMoveState(MoveState& st, uint8_t destBefore) const noexcept {
    st.prev = state;
    st.capturedPiece = destBefore;
}

__attribute__((always_inline))
inline void Board::restoreState(const MoveState& st) noexcept {
    activeColor = oppositeColor(activeColor);
    // The slot doMove overwrote is historySize - 1 before shrinking back.
    repetitionHistory[state.historySize - 1] = st.prevHistorySlotValue;
    state = st.prev;
}

__attribute__((always_inline))
inline void Board::passTurn(bool resetHalfMoves) noexcept {
    if (resetHalfMoves) state.halfMoveClock = 0;
    else if (state.halfMoveClock < 255) ++state.halfMoveClock;
    if (activeColor == BLACK && state.fullMoveClock < 255) ++state.fullMoveClock;
    activeColor = oppositeColor(activeColor);
}

template<Board::MoveKind Kind, bool WithPrefetch>
inline void Board::doMoveByKind(NNUE::Accumulator& f, MoveState& st, uint8_t moving, uint8_t destBefore, uint8_t fromIndex,
                                uint8_t toIndex, uint8_t promotionType, uint64_t& key, const HashPrefetch& tt) noexcept {
    // The child's key first, so that its TT bucket loads while the threats
    // are recorded.
    uint8_t capIndex = 0;
    uint8_t rook = 0;
    uint8_t placed = moving;
    if constexpr (isPromotionKind(Kind)) {
        // A missing or invalid promotion piece means a queen.
        const uint8_t promo = (promotionType >= KNIGHT && promotionType <= QUEEN) ? promotionType : uint8_t{QUEEN};
        placed = static_cast<uint8_t>(promo | (moving & MASK_COLOR));
    }
    key ^= zobrist::TABLES.pieces[placed][toIndex];
    if constexpr (Kind == MoveKind::EnPassant) {
        capIndex = enPassantVictim(moving, toIndex);
        st.capturedPiece = get(capIndex);
        key ^= zobrist::TABLES.pieces[st.capturedPiece][capIndex];
    } else if constexpr (isCaptureKind(Kind)) {
        key ^= zobrist::TABLES.pieces[destBefore][toIndex];
    } else if constexpr (Kind == MoveKind::Castling) {
        const Castling& c = castlingTo(toIndex);
        rook = get(c.rook);
        key ^= zobrist::TABLES.pieces[rook][c.rook] ^ zobrist::TABLES.pieces[rook][c.rookTo];
    } else if constexpr (Kind == MoveKind::DoublePawnPush) {
        state.enPassant = static_cast<Square>((fromIndex + toIndex) >> 1);
        const int us = colorToIndex(moving);
        if ((pieces::PAWN_ATTACKS[us][state.enPassant] & bb[PAWN][us ^ 1]) != 0ULL) {
            state.epHashFile = chess::file(state.enPassant);
            key ^= zobrist::TABLES.enPassant[state.epHashFile];
        }
    }
    if constexpr (WithPrefetch) tt.prefetch(key);

    // Every recordThreats sees the board between two of these steps, the piece
    // it records standing on its square.
    if constexpr (Kind == MoveKind::EnPassant) {
        recordThreats<false>(f, st.capturedPiece, capIndex);
        set(capIndex, EMPTY);
        occupancy &= ~(1ULL << capIndex);
        removePieceFromBB<true>(st.capturedPiece, capIndex, &f);
    }
    recordThreats<false>(f, moving, fromIndex);
    if constexpr (isCaptureKind(Kind)) {
        removePieceFromBB<true>(destBefore, toIndex, &f);
    }
    relocate<true>(moving, fromIndex, toIndex, &f);
    if constexpr (Kind == MoveKind::Castling) {
        const Castling& c = castlingTo(toIndex);
        recordThreats<true>(f, moving, toIndex);
        recordThreats<false>(f, rook, c.rook);
        relocate<true>(rook, c.rook, c.rookTo, &f);
        recordThreats<true>(f, rook, c.rookTo);
    }
    if constexpr (isPromotionKind(Kind)) {
        removePieceFromBB<true>(moving, toIndex, &f);
        addPieceToBB<true>(placed, toIndex, &f);
        set(toIndex, static_cast<piece_id>(placed));
    }
    if constexpr (isCaptureKind(Kind)) {
        recordReplacedThreats(f, destBefore, placed, toIndex);
    } else if constexpr (Kind != MoveKind::Castling) {
        recordThreats<true>(f, placed, toIndex);
    }
}

template<Board::MoveKind Kind>
inline void Board::undoMoveByKind(const MoveState& st, uint8_t pieceOnTo, uint8_t fromIndex, uint8_t toIndex) noexcept {
    if constexpr (isPromotionKind(Kind)) {
        const uint8_t pawn = PAWN | (pieceOnTo & MASK_COLOR);
        removePieceFromBB<false>(pieceOnTo, toIndex);
        addPieceToBB<false>(pawn, toIndex);
        set(toIndex, static_cast<piece_id>(pawn));
        pieceOnTo = pawn;
    }
    relocate<false>(pieceOnTo, toIndex, fromIndex);
    if constexpr (Kind == MoveKind::EnPassant || isCaptureKind(Kind)) {
        const uint8_t sq = (Kind == MoveKind::EnPassant) ? enPassantVictim(pieceOnTo, toIndex) : toIndex;
        set(sq, static_cast<piece_id>(st.capturedPiece));
        occupancy |= 1ULL << sq;
        addPieceToBB<false>(st.capturedPiece, sq);
    }
    if constexpr (Kind == MoveKind::Castling) {
        const Castling& c = castlingTo(toIndex);
        relocate<false>(get(c.rookTo), c.rookTo, c.rook);
    }
}

// --- Incremental do/undo, no legality checks ---
template<bool WithPrefetch>
inline void Board::doMoveImpl(const Move& m, MoveState& st, const HashPrefetch& tt) noexcept {
    NNUE::Accumulator& f = pushAccFrame();
    const uint8_t fromIndex = m.from;
    const uint8_t toIndex = m.to;
    const uint8_t promo = m.promotionType;
    const uint8_t moving     = get(fromIndex);
    const uint8_t destBefore = get(toIndex);
    prepareMoveState(st, destBefore);
    st.moveKind = classifyMoveKind(moving & MASK_PIECE_TYPE, moving & MASK_COLOR, fromIndex, toIndex,
                                   destBefore, st.prev.enPassant);

    uint64_t key = state.hash ^ zobrist::TABLES.pieces[moving][fromIndex] ^ zobrist::TABLES.sideToMove;
    if (st.prev.epHashFile < 8) key ^= zobrist::TABLES.enPassant[st.prev.epHashFile];
    const uint8_t newCastle = state.castle & CASTLING_KEEP[fromIndex] & CASTLING_KEEP[toIndex];
    if (newCastle != st.prev.castle) key ^= zobrist::TABLES.castling[st.prev.castle] ^ zobrist::TABLES.castling[newCastle];
    state.castle     = newCastle;
    state.enPassant  = NO_SQUARE;
    state.epHashFile = 0xFF;

    visitKind(st.moveKind, [&]<MoveKind K>() {
        doMoveByKind<K, WithPrefetch>(f, st, moving, destBefore, fromIndex, toIndex, promo, key, tt);
    });

    const bool resetHistory = (moving & MASK_PIECE_TYPE) == PAWN || destBefore != EMPTY;
    passTurn(resetHistory);
    state.hash = key;
    if (state.pliesFromNull < 255) ++state.pliesFromNull;
    updateRepetitionAfterMove(resetHistory, st);
}

__attribute__((hot))
void Board::doMove(const Move& m, MoveState& st) noexcept {
    static constexpr HashPrefetch kNoPrefetch{};
    doMoveImpl<false>(m, st, kNoPrefetch);
}

__attribute__((hot))
void Board::doMove(const Move& m, MoveState& st, const HashPrefetch& tt) noexcept {
    doMoveImpl<true>(m, st, tt);
}

__attribute__((hot))
void Board::undoMove(const Move& m, const MoveState& st) noexcept {
    const uint8_t pieceOnTo = get(m.to);
    visitKind(st.moveKind, [&]<MoveKind K>() { undoMoveByKind<K>(st, pieceOnTo, m.from, m.to); });
    restoreState(st);
    --accTop;
}

__attribute__((hot))
void Board::doNullMove(MoveState& st) noexcept {
    prepareMoveState(st, EMPTY);
    uint64_t key = state.hash ^ zobrist::TABLES.sideToMove;
    if (st.prev.epHashFile < 8) key ^= zobrist::TABLES.enPassant[st.prev.epHashFile];
    state.enPassant = NO_SQUARE;
    state.epHashFile = 0xFF;
    passTurn(false);
    state.hash = key;
    // The post-null position takes a history slot like any ply, so the same
    // side to move stays on every second slot; countRepetitions never looks
    // further back than this entry.
    updateRepetitionAfterMove(false, st);
    state.pliesFromNull = 0;
}

__attribute__((hot))
void Board::doNullMove(MoveState& st, const HashPrefetch& tt) noexcept {
    prepareMoveState(st, EMPTY);
    uint64_t key = state.hash ^ zobrist::TABLES.sideToMove;
    if (st.prev.epHashFile < 8) key ^= zobrist::TABLES.enPassant[st.prev.epHashFile];
    tt.prefetch(key);
    state.enPassant = NO_SQUARE;
    state.epHashFile = 0xFF;
    passTurn(false);
    state.hash = key;
    updateRepetitionAfterMove(false, st);
    state.pliesFromNull = 0;
}

__attribute__((hot))
void Board::undoNullMove(const MoveState& st) noexcept {
    restoreState(st);
}

} // namespace chess
