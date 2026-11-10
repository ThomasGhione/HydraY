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
    f.dirty[0] = f.dirty[1] = true;
    if (accTop - accFloor >= NNUE::ACC_FRAMES) accFloor = accTop - NNUE::ACC_FRAMES + 1;
}

__attribute__((always_inline))
inline void Board::recordAcc(NNUE::AccDelta delta) noexcept {
    NNUE::Accumulator& f = frame(accTop);
    f.delta[f.ndelta++] = delta;
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
    if constexpr (Kind == MoveKind::EnPassant) {
        const uint8_t capIndex = enPassantVictim(moving, toIndex);
        st.capturedPiece = get(capIndex);
        set(capIndex, EMPTY);
        occupancy &= ~(1ULL << capIndex);
        removePieceFromBB<true>(st.capturedPiece, capIndex);
        key ^= zobrist::TABLES.pieces[st.capturedPiece][capIndex];
    }
    if constexpr (isCaptureKind(Kind)) {
        removePieceFromBB<true>(destBefore, toIndex);
        key ^= zobrist::TABLES.pieces[destBefore][toIndex];
    }
    relocate<true>(moving, fromIndex, toIndex);
    if constexpr (Kind == MoveKind::Castling) {
        const Castling& c = castlingTo(toIndex);
        const uint8_t rook = get(c.rook);
        relocate<true>(rook, c.rook, c.rookTo);
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
        key ^= zobrist::TABLES.pieces[promoted][toIndex];
    } else {
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
