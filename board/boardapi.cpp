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

// BEYOND[a][b]: the squares past b on the line from a through b, 0 when the two
// are not aligned.
constexpr std::array<std::array<uint64_t, 64>, 64> BEYOND = [] {
    std::array<std::array<uint64_t, 64>, 64> lut{};
    for (int a = 0; a < 64; ++a)
        for (int b = 0; b < 64; ++b) {
            const int df = chess::file(b) - chess::file(a), dr = chess::rank(b) - chess::rank(a);
            if (a == b || (df != 0 && dr != 0 && df != dr && df != -dr)) continue;
            const int sf = (df > 0) - (df < 0), sr = (dr > 0) - (dr < 0);
            for (int f = chess::file(b) + sf, r = chess::rank(b) + sr; f >= 0 && f < 8 && r >= 0 && r < 8; f += sf, r += sr)
                lut[a][b] |= 1ULL << (r * 8 + f);
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
inline void Board::pushAccFrame() noexcept {
    NNUE::Accumulator& f = frame(++accTop);
    f.ndelta = 0;
    f.nthreat = 0;
    f.dirty[0] = f.dirty[1] = true;
    if (accTop - accFloor >= NNUE::ACC_FRAMES) accFloor = accTop - NNUE::ACC_FRAMES + 1;
}

__attribute__((always_inline))
inline void Board::recordAcc(NNUE::AccDelta delta) noexcept {
    NNUE::Accumulator& f = frame(accTop);
    f.delta[f.ndelta++] = delta;
}

// Its attacks, the attacks on it, and the lines of the sliders that see sq,
// which reach past sq exactly when the piece is not there.
template<bool Add>
__attribute__((always_inline))
inline void Board::recordThreats(uint8_t piece, uint8_t sq) noexcept {
    NNUE::Accumulator& f = frame(accTop);
    const auto record = [&f](uint8_t attacker, uint8_t from, uint8_t victim, uint8_t to, bool add) {
        if (!NNUE::threatPairIsFeature(attacker, victim)) return;
        if (f.nthreat < NNUE::ACC_MAX_THREATS) f.threatDelta[f.nthreat++] = {attacker, from, victim, to, add};
        else f.nthreat = NNUE::ACC_THREATS_OVERFLOW;
    };
    const uint64_t rookRays = pieces::Rook::attacks(sq, occupancy);
    const uint64_t bishopRays = pieces::Bishop::attacks(sq, occupancy);
    const uint64_t sliders = (rookRays & (bb[ROOK][0] | bb[ROOK][1] | bb[QUEEN][0] | bb[QUEEN][1]))
                           | (bishopRays & (bb[BISHOP][0] | bb[BISHOP][1] | bb[QUEEN][0] | bb[QUEEN][1]));
    uint64_t attackers = sliders
                       | (pieces::PAWN_ATTACKS[1][sq] & bb[PAWN][0]) | (pieces::PAWN_ATTACKS[0][sq] & bb[PAWN][1])
                       | (pieces::KNIGHT_ATTACKS[sq] & (bb[KNIGHT][0] | bb[KNIGHT][1]))
                       | (pieces::KING_ATTACKS[sq] & (bb[KING][0] | bb[KING][1]));
    for (uint64_t victims = attacksFrom(piece, sq, occupancy) & occupancy; victims; ) {
        const auto to = static_cast<uint8_t>(pieces::popLSB(victims));
        record(piece, sq, get(to), to, Add);
    }
    while (attackers) {
        const auto from = static_cast<uint8_t>(pieces::popLSB(attackers));
        record(get(from), from, piece, sq, Add);
    }
    for (uint64_t s = sliders; s; ) {
        const auto from = static_cast<uint8_t>(pieces::popLSB(s));
        if (const uint64_t past = (rookRays | bishopRays) & BEYOND[from][sq] & occupancy) {
            const auto to = static_cast<uint8_t>(std::countr_zero(past));
            record(get(from), from, get(to), to, !Add);
        }
    }
}

template<bool Record>
__attribute__((always_inline))
inline void Board::addPieceToBB(uint8_t piece, uint8_t index) noexcept {
    bb[piece & MASK_PIECE_TYPE][colorToIndex(piece)] |= 1ULL << index;
    if constexpr (Record) recordAcc({NNUE::AccDelta::Add, piece, index, 0});
}

template<bool Record>
__attribute__((always_inline))
inline void Board::removePieceFromBB(uint8_t piece, uint8_t index) noexcept {
    bb[piece & MASK_PIECE_TYPE][colorToIndex(piece)] &= ~(1ULL << index);
    if constexpr (Record) recordAcc({NNUE::AccDelta::Remove, piece, index, 0});
}

// Moves a piece between squares: mailbox, occupancy, its bitboard (which never
// holds the destination, so one XOR does both ends) and, in doMove, the frame.
template<bool Record>
__attribute__((always_inline))
inline void Board::relocate(uint8_t piece, uint8_t fromIndex, uint8_t toIndex) noexcept {
    set(toIndex, static_cast<piece_id>(piece));
    set(fromIndex, EMPTY);
    occupancy = (occupancy | (1ULL << toIndex)) & ~(1ULL << fromIndex);
    bb[piece & MASK_PIECE_TYPE][colorToIndex(piece)] ^= (1ULL << fromIndex) | (1ULL << toIndex);
    if constexpr (Record) recordAcc({NNUE::AccDelta::Move, piece, fromIndex, toIndex});
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

template<Board::MoveKind Kind>
inline void Board::doMoveByKind(MoveState& st, uint8_t moving, uint8_t destBefore, uint8_t fromIndex,
                                uint8_t toIndex, uint8_t promotionType, uint64_t& key) noexcept {
    // Every recordThreats sees the board between two of these steps, the piece
    // it records standing on its square.
    if constexpr (Kind == MoveKind::EnPassant) {
        const uint8_t capIndex = enPassantVictim(moving, toIndex);
        st.capturedPiece = get(capIndex);
        recordThreats<false>(st.capturedPiece, capIndex);
        set(capIndex, EMPTY);
        occupancy &= ~(1ULL << capIndex);
        removePieceFromBB<true>(st.capturedPiece, capIndex);
        key ^= zobrist::TABLES.pieces[st.capturedPiece][capIndex];
    }
    if constexpr (isCaptureKind(Kind)) {
        recordThreats<false>(destBefore, toIndex);
        removePieceFromBB<true>(destBefore, toIndex);
        occupancy &= ~(1ULL << toIndex);
        key ^= zobrist::TABLES.pieces[destBefore][toIndex];
    }
    recordThreats<false>(moving, fromIndex);
    relocate<true>(moving, fromIndex, toIndex);
    if constexpr (Kind == MoveKind::Castling) {
        const Castling& c = castlingTo(toIndex);
        const uint8_t rook = get(c.rook);
        recordThreats<true>(moving, toIndex);
        recordThreats<false>(rook, c.rook);
        relocate<true>(rook, c.rook, c.rookTo);
        recordThreats<true>(rook, c.rookTo);
        key ^= zobrist::TABLES.pieces[rook][c.rook] ^ zobrist::TABLES.pieces[rook][c.rookTo];
    }
    state.castle &= CASTLING_KEEP[fromIndex] & CASTLING_KEEP[toIndex];
    if constexpr (Kind == MoveKind::DoublePawnPush) state.enPassant = (fromIndex + toIndex) >> 1;
    if constexpr (isPromotionKind(Kind)) {
        // A missing or invalid promotion piece means a queen.
        const uint8_t promo = (promotionType >= KNIGHT && promotionType <= QUEEN) ? promotionType : uint8_t{QUEEN};
        const uint8_t promoted = static_cast<uint8_t>(promo | (moving & MASK_COLOR));
        removePieceFromBB<true>(moving, toIndex);
        addPieceToBB<true>(promoted, toIndex);
        set(toIndex, static_cast<piece_id>(promoted));
        recordThreats<true>(promoted, toIndex);
        key ^= zobrist::TABLES.pieces[promoted][toIndex];
    } else {
        if constexpr (Kind != MoveKind::Castling) recordThreats<true>(moving, toIndex);
        key ^= zobrist::TABLES.pieces[moving][toIndex];
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
__attribute__((hot))
void Board::doMove(const Move& m, MoveState& st) noexcept {
    pushAccFrame();
    const uint8_t moving     = get(m.from);
    const uint8_t destBefore = get(m.to);
    prepareMoveState(st, destBefore);
    st.moveKind = classifyMoveKind(moving & MASK_PIECE_TYPE, moving & MASK_COLOR, m.from, m.to,
                                   destBefore, st.prev.enPassant);

    uint64_t key = state.hash ^ zobrist::TABLES.pieces[moving][m.from] ^ zobrist::TABLES.sideToMove;
    if (st.prev.epHashFile < 8) key ^= zobrist::TABLES.enPassant[st.prev.epHashFile];
    state.enPassant = NO_SQUARE;
    visitKind(st.moveKind, [&]<MoveKind K>() {
        doMoveByKind<K>(st, moving, destBefore, m.from, m.to, m.promotionType, key);
    });

    const bool resetHistory = (moving & MASK_PIECE_TYPE) == PAWN || destBefore != EMPTY;
    passTurn(resetHistory);
    if (state.castle != st.prev.castle) key ^= zobrist::TABLES.castling[st.prev.castle] ^ zobrist::TABLES.castling[state.castle];
    state.epHashFile = 0xFF;
    if (isValidSquare(state.enPassant) && zobrist::hasPseudoLegalEnPassantCapture(*this, state.enPassant)) {
        state.epHashFile = chess::file(state.enPassant);
        key ^= zobrist::TABLES.enPassant[state.epHashFile];
    }
    state.hash = key;
    if (state.pliesFromNull < 255) ++state.pliesFromNull;
    updateRepetitionAfterMove(resetHistory, st);
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
void Board::undoNullMove(const MoveState& st) noexcept {
    restoreState(st);
}

} // namespace chess
