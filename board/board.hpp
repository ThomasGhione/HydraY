#pragma once

#include <algorithm>
#include <bit>
#include <string>
#include <array>
#include <cstdint>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <memory>

#include "../nnue/accumulator.hpp"
#include "./coords.hpp"
#include "./piece.hpp"

namespace chess {

// The FEN letter of each piece nibble (type | 0x8 for white), '.' for none.
inline constexpr std::string_view PIECE_LETTERS = ".pnbrqk..PNBRQK.";

struct Move {
    Square from = NO_SQUARE;
    Square to   = NO_SQUARE;
    // Board::piece_id type of the promotion piece (KNIGHT=2 .. QUEEN=5); 0 = not
    // a promotion. Char conversion happens only at the UCI/user-input boundary.
    uint8_t promotionType = 0;

    constexpr bool operator==(const Move&) const noexcept = default;
    constexpr bool sameFromTo(const Move& other) const noexcept { return from == other.from && to == other.to; };

    // KNIGHT..QUEEN are 2..5 (Board is declared below; static_asserts there pin it).
    static constexpr uint8_t promotionTypeFromChar(char c) noexcept {
        const size_t type = PIECE_LETTERS.find(ascii::toLower(c));
        return (type >= 2 && type <= 5) ? static_cast<uint8_t>(type) : 0;
    }
    constexpr char promotionChar() const noexcept { return PIECE_LETTERS[promotionType & 7]; }

    std::string toUCIString() const noexcept { return squareToString(from) + squareToString(to) + (promotionType ? std::string(1, promotionChar()) : std::string{}); }
};

class Board {
public:
    static constexpr const char* STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    static constexpr const char* EMPTY_FEN = "8/8/8/8/8/8/8/8 w - - 0 1";

    // --- Enums ---

    enum piece_id : uint8_t {
        EMPTY  = 0x0,
        PAWN   = 0x1,
        KNIGHT = 0x2,
        BISHOP = 0x3,
        ROOK   = 0x4,
        QUEEN  = 0x5,
        KING   = 0x6,
        BLACK  = 0x0,
        WHITE  = 0x8,
    };

    enum class MoveKind : uint8_t {
        Quiet = 0,
        Capture,
        DoublePawnPush,
        EnPassant,
        Castling,
        PromotionQuiet,
        PromotionCapture
    };

    // --- Public structs ---

    // What a move changes and undoMove cannot work back out: doMove saves it
    // whole in the MoveState and undoMove puts it back whole.
    struct PlyState {
        uint64_t hash;
        uint8_t  halfMoveClock;
        uint8_t  fullMoveClock;
        uint8_t  castle;
        Square   enPassant;
        uint8_t  epHashFile;     // the en passant file in the hash, 0xFF = none
        uint8_t  historySize;
        uint8_t  pliesFromNull;  // since the last null move, saturating; 255 = none within reach
    };

    // Members are intentionally NOT default-initialised: skips a full zero-fill per node.
    // Value-init sites (`MoveState st{}`) still zero everything.
    struct MoveState {
        PlyState prev;
        // The repetitionHistory slot this move overwrites: an irreversible move
        // restarts the history at slot 0, and undo must give the entry back.
        uint64_t prevHistorySlotValue;
        uint8_t  capturedPiece;
        MoveKind moveKind;
    };

    static_assert(sizeof(MoveState) <= 96, "MoveState layout regressed; keep it compact for search stack usage.");

    // --- Static constants ---
    static constexpr uint8_t  MASK_PIECE              = 0x0F;
    static constexpr uint8_t  MASK_COLOR              = 0x08;
    static constexpr uint8_t  MASK_PIECE_TYPE         = 0x07;
    static constexpr uint8_t  CASTLING_RIGHTS_ALL     = 0x0F;
    static constexpr uint16_t REPETITION_HISTORY_CAPACITY = 255; // must be 255 instead of 100 due to the 8-bit PlyState::historySize

    // Castling by right, KQkq = bits 0..3 of the rights: the king's and the
    // rook's squares before and after, the squares between them, which must be
    // empty, and the king's path, which no enemy piece may attack.
    struct Castling {
        Square king, kingTo, rook, rookTo;
        uint64_t between, kingPath;
    };
    static constexpr std::array<Castling, 4> CASTLINGS = [] {
        std::array<Castling, 4> all{};
        for (int right = 0; right < 4; ++right) {
            const int king = right < 2 ? 60 : 4, dir = (right & 1) ? -1 : 1;   // e1 or e8; queenside is west
            const int rook = dir > 0 ? king + 3 : king - 4;
            Castling& c = all[right];
            c = {Square(king), Square(king + 2 * dir), Square(rook), Square(king + dir), 0, 0};
            for (int sq = std::min(king, rook) + 1; sq < std::max(king, rook); ++sq) c.between |= 1ULL << sq;
            for (int step = 0; step <= 2; ++step) c.kingPath |= 1ULL << (king + step * dir);
        }
        return all;
    }();
    // The castling whose king lands on kingTo: g1, c1, g8 or c8.
    static constexpr const Castling& castlingTo(Square kingTo) noexcept {
        return CASTLINGS[2 * (rank(kingTo) == 0) + (file(kingTo) == 2)];
    }

    // --- Constructors ---
    Board() noexcept { fenToBoard(STARTING_FEN); }
    // A malformed FEN gives the empty board (no kings: the search answers
    // `bestmove 0000`), never a half-written one.
    explicit Board(const std::string& fen) { if (!fenToBoard(fen)) fenToBoard(EMPTY_FEN); }
    // A copy takes the current position's accumulator only, computed first.
    Board(const Board& other) noexcept { copyFromBoard(other); }
    Board& operator=(const Board& other) noexcept {
        if (this != &other) copyFromBoard(other);
        return *this;
    }
    Board(Board&& other) noexcept = default;
    Board& operator=(Board&& other) noexcept = default;

    // --- Static utilities ---
    static constexpr uint8_t  oppositeColor(uint8_t color) noexcept { return color ^ 0x8; };
    static constexpr uint8_t  colorToIndex(uint8_t color) noexcept { return ((color & MASK_COLOR) >> 3) ^ 0x1u; };
    static constexpr uint8_t  promotionRank(bool isWhite) noexcept { return isWhite ? 0 : 7; };

    // --- Board access ---
    __attribute__((hot, always_inline)) constexpr uint8_t get(uint8_t index) const noexcept { return chessboard[index]; }

    // --- Move execution ---
    void doMove(const Move& m, MoveState& st) noexcept;
    void undoMove(const Move& m, const MoveState& st) noexcept;
    void doNullMove(MoveState& st) noexcept;
    void undoNullMove(const MoveState& st) noexcept;
    bool move(Move move) noexcept;

    // --- Legality & attack queries ---
    bool isLegalPseudoMove(uint8_t fromIndex, uint8_t toIndex, uint8_t fromPiece) const noexcept;
    bool isSquareAttacked(uint8_t targetIndex, uint8_t byColor, uint8_t excludeSquare = 64) const noexcept;
    // Whether the side of `right` may castle that way now (CASTLINGS[right]).
    bool canCastle(int right) const noexcept;
    bool inCheck(uint8_t color) const noexcept { return checkersTo(color) != 0; }
    // The pieces of `side` that attack sq with `occ` as the blockers, where
    // piecesOf(type) is that side's bitboard of a piece type.
    template<typename PiecesOf>
    static uint64_t attackersOf(uint8_t sq, int side, uint64_t occ, PiecesOf piecesOf) noexcept {
        uint64_t attackers = (pieces::PAWN_ATTACKS[side ^ 1][sq] & piecesOf(PAWN))
                           | (pieces::KNIGHT_ATTACKS[sq] & piecesOf(KNIGHT))
                           | (pieces::KING_ATTACKS[sq] & piecesOf(KING));
        if (const uint64_t rookLike = piecesOf(ROOK) | piecesOf(QUEEN)) attackers |= pieces::Rook::attacks(sq, occ) & rookLike;
        if (const uint64_t bishopLike = piecesOf(BISHOP) | piecesOf(QUEEN)) attackers |= pieces::Bishop::attacks(sq, occ) & bishopLike;
        return attackers;
    }
    // Bitboard of enemy pieces giving check to `color`'s king (0 = no check).
    // One scan answers inCheck (!=0), double check (>1 bit) and, via the
    // checker square, the evasion mask - callers should reuse it.
    uint64_t checkersTo(uint8_t color) const noexcept;

    // --- Game state queries ---
    bool isFiftyMoveRule() const noexcept { return state.halfMoveClock >= 100; }
    bool isThreefoldRepetition() const noexcept { return countRepetitions() >= 3; }
    int  countRepetitions() const noexcept;
    bool hasInsufficientMaterialDraw() const noexcept;

    // --- State accessors ---
    constexpr uint8_t  getActiveColor() const noexcept   { return activeColor; }
    constexpr uint8_t  castlingRights() const noexcept   { return state.castle; }
    constexpr uint16_t getFullMoveClock() const noexcept { return state.fullMoveClock; }
    constexpr uint8_t  getHalfMoveClock() const noexcept { return state.halfMoveClock; }
    Square             getEnPassant() const noexcept     { return state.enPassant; }
    constexpr uint64_t getHash() const noexcept          { return state.hash; }
    uint64_t           getPiecesBitMap() const noexcept  { return occupancy; }
    void               rebuildBitboardsFromSquares() noexcept;
    // From-scratch NNUE accumulator recompute (no-op when no net is loaded).
    // Incremental maintenance happens in addPieceToBB/removePieceFromBB.
    void               refreshNnueAccumulator() noexcept;
    // Computes the current position's accumulator if it is not yet, from the
    // nearest computed ancestor or the Finny table. Must be called with the
    // Board in a consistent state (NNUE::evaluate and the selftest do); const
    // because the accumulator is a cache of board-derived state.
    void               ensureNnueAccumulatorClean() const noexcept;
    // The current position's accumulator, valid after ensureNnueAccumulatorClean.
    const NNUE::Accumulator& accumulator() const noexcept { return frame(accTop); }
    // The squares `piece` on sq attacks with `occ` as the blockers.
    static uint64_t attacksFrom(uint8_t piece, uint8_t sq, uint64_t occ) noexcept {
        switch (piece & MASK_PIECE_TYPE) {
            case PAWN:   return pieces::PAWN_ATTACKS[colorToIndex(piece)][sq];
            case KNIGHT: return pieces::KNIGHT_ATTACKS[sq];
            case BISHOP: return pieces::Bishop::attacks(sq, occ);
            case ROOK:   return pieces::Rook::attacks(sq, occ);
            case QUEEN:  return pieces::Rook::attacks(sq, occ) | pieces::Bishop::attacks(sq, occ);
            default:     return pieces::KING_ATTACKS[sq];
        }
    }

    // --- FEN ---
    // False on malformed input, and the board is then left as it was.
    bool        fenToBoard(const std::string& fen);
    std::string boardToFen() const;

    // --- Public bitboards (direct access for eval/search hot paths) ---
    // bb[type][side]: type PAWN..KING (EMPTY unused), side 0 = white.
    std::array<std::array<uint64_t, 2>, 7> bb{};
    uint64_t sideOcc(int side) const noexcept {
        return bb[PAWN][side] | bb[KNIGHT][side] | bb[BISHOP][side] | bb[ROOK][side] | bb[QUEEN][side] | bb[KING][side];
    }

private:
    NNUE::Accumulator& frame(int i) const noexcept { return accFrames[static_cast<size_t>(i & (NNUE::ACC_FRAMES - 1))]; }
    inline void pushAccFrame() noexcept;
    inline void recordAcc(NNUE::AccDelta delta) noexcept;
    // Logs the threats that `piece` on sq takes part in as appearing (Add) or
    // vanishing, with the board as it stands and the piece on sq.
    template<bool Add> void recordThreats(uint8_t piece, uint8_t sq) noexcept;
    // Logs threat changes when oldPiece on sq is replaced in place by newPiece
    // (board stands after the move; since sq remains occupied, no rays unblock through it).
    void recordReplacedThreats(uint8_t oldPiece, uint8_t newPiece, uint8_t sq) noexcept;
    template<bool Add, bool Checked>
    inline void emitThreatDeltas(NNUE::Accumulator& f, uint8_t startN, uint8_t piece, uint8_t sq,
                                 uint64_t victims, uint64_t attackers, uint64_t sliders,
                                 uint64_t sliderOcc) const noexcept;
    template<bool Checked>
    inline void emitReplacedThreatDeltas(NNUE::Accumulator& f, uint8_t startN, uint8_t oldPiece,
                                         uint8_t newPiece, uint8_t sq, uint64_t oldVictims,
                                         uint64_t newVictims, uint64_t oldAttackers,
                                         uint64_t newAttackers) const noexcept;
    // Perspective p's threat values of the current position, from nothing.
    void computeThreats(int16_t* row, int p, int flip) const noexcept;

    // --- Private helpers: move execution ---
    // Record: doMove logs the change in the new frame; undoMove only pops it.
    inline void set(uint8_t index, piece_id value) noexcept;
    template<bool Record> inline void addPieceToBB(uint8_t piece, uint8_t index) noexcept;
    template<bool Record> inline void removePieceFromBB(uint8_t piece, uint8_t index) noexcept;
    template<bool Record> inline void relocate(uint8_t piece, uint8_t fromIndex, uint8_t toIndex) noexcept;
    inline void prepareMoveState(MoveState& st, uint8_t destBefore) const noexcept;
    inline void restoreState(const MoveState& st) noexcept;
    inline void passTurn(bool resetHalfMoves) noexcept;
    template<typename F>
    static inline void visitKind(MoveKind kind, F&& f) noexcept;
    template<MoveKind Kind>
    inline void doMoveByKind(MoveState& st, uint8_t moving, uint8_t destBefore, uint8_t fromIndex,
                             uint8_t toIndex, uint8_t promotionType, uint64_t& key) noexcept;
    template<MoveKind Kind>
    inline void undoMoveByKind(const MoveState& st, uint8_t pieceOnTo, uint8_t fromIndex, uint8_t toIndex) noexcept;

    // --- Private helpers: legality ---
    [[nodiscard]] inline bool isKingSafeAfterMove(uint8_t movingColor, uint8_t fromIndex,
                                                   uint8_t toIndex, uint64_t capturedMask) const noexcept;
    [[nodiscard]] inline bool isKingMoveLegal(uint8_t fromIndex, uint8_t toIndex,
                                               uint64_t toBit, uint8_t movingColor) const noexcept;
    template<uint8_t PieceType>
    [[nodiscard]] inline bool pseudoMoveLegalByType(uint8_t fromIndex, uint8_t toIndex, uint64_t toBit,
                                                    uint8_t movingColor, uint8_t destPiece) const noexcept;
    // The pieces of `side` on this board that attack sq with `occ` as the blockers.
    uint64_t attackersOf(uint8_t sq, uint8_t side, uint64_t occ) const noexcept {
        return attackersOf(sq, side, occ, [&](int type) { return bb[type][side]; });
    }

    // --- Private helpers: move classification ---
    static constexpr bool isCaptureKind(MoveKind k) noexcept { return k == MoveKind::Capture || k == MoveKind::PromotionCapture; }
    static constexpr bool isPromotionKind(MoveKind k) noexcept { return k == MoveKind::PromotionQuiet || k == MoveKind::PromotionCapture; }
    // The square of the pawn that `pawn` takes en passant on toIndex.
    static constexpr uint8_t enPassantVictim(uint8_t pawn, uint8_t toIndex) noexcept {
        return static_cast<uint8_t>(toIndex + ((pawn & MASK_COLOR) == WHITE ? 8 : -8));
    }
    [[nodiscard]] static inline MoveKind    classifyMoveKind(uint8_t movingType, uint8_t movingColor,
                                                              uint8_t fromIndex, uint8_t toIndex,
                                                              uint8_t destBefore, const Square& prevEnPassant) noexcept;

    // --- Private helpers: FEN & hash ---
    static uint8_t safeParseInt(const std::string& section, int min, int max, int defaultValue);
    void           recomputeHashAndEp() noexcept;
    void           rebuildRepetitionHistory() noexcept;
    void           updateRepetitionAfterMove(bool resetHistory, MoveState& st) noexcept;
    void           copyFromBoard(const Board& other) noexcept;

    // --- Private data ---
    // One byte per square, engine index (a8 = 0, h1 = 63): a read is a single
    // load, where nibble packing cost a shift and a mask on every access.
    std::array<uint8_t, 64> chessboard;
    PlyState state{0ULL, 0, 1, CASTLING_RIGHTS_ALL, NO_SQUARE, 0xFF, 0, 255};
    std::array<uint64_t, REPETITION_HISTORY_CAPACITY> repetitionHistory{};
    uint64_t occupancy   = 0ULL;
    uint8_t  activeColor = WHITE;

    // Accumulator stack, a ring of frames: frame(accTop) is the current
    // position, each frame one move after the previous one. doMove pushes the
    // move's feature changes and undoMove pops, so a sibling starts again from
    // its parent's values instead of undoing the previous sibling's move. The
    // values are computed only when read: a node cut before it evaluates
    // pays nothing. Garbage until the first refreshNnueAccumulator(); a cache
    // that the const evaluation paths settle. On the heap: half a MiB.
    int accTop   = 0;   // grows with every move, frame() wraps it
    int accFloor = 0;   // oldest frame of this line: a load, a copy or a refresh
    std::unique_ptr<NNUE::Accumulator[]> accFrames{new NNUE::Accumulator[NNUE::ACC_FRAMES]};
};

static_assert(Board::CASTLINGS[0].rookTo == 61 && Board::CASTLINGS[1].between == 0x0E00000000000000ULL
           && Board::CASTLINGS[2].kingPath == 0x70ULL && Board::CASTLINGS[3].rook == 0, "castling geometry");
static_assert(PIECE_LETTERS[Board::PAWN] == 'p' && PIECE_LETTERS[Board::KING | Board::WHITE] == 'K'
           && Move::promotionTypeFromChar('N') == Board::KNIGHT && Move::promotionTypeFromChar('q') == Board::QUEEN
           && Move::promotionTypeFromChar('k') == 0, "piece letters and promotion types follow Board::piece_id");

// FEN piece placement of 64 squares (0 = a8), each a piece letter or '.'.
std::string placementFen(const std::array<char, 64>& squares);

} // namespace chess
