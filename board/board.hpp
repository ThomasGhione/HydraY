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

struct Move {
    Square from = NO_SQUARE;
    Square to   = NO_SQUARE;
    // Board::piece_id type of the promotion piece (KNIGHT=2 .. QUEEN=5); 0 = not
    // a promotion. Char conversion happens only at the UCI/user-input boundary.
    uint8_t promotionType = 0;

    constexpr bool operator==(const Move&) const noexcept = default;
    constexpr bool sameFromTo(const Move& other) const noexcept { return from == other.from && to == other.to; };

    // Literal values mirror Board::piece_id (declared below Move); the
    // static_asserts after the Board definition pin the coupling.
    static constexpr uint8_t promotionTypeFromChar(char c) noexcept {
        switch (c) {
            case 'q': case 'Q': return 5; // Board::QUEEN
            case 'r': case 'R': return 4; // Board::ROOK
            case 'b': case 'B': return 3; // Board::BISHOP
            case 'n': case 'N': return 2; // Board::KNIGHT
            default:            return 0;
        }
    }

    constexpr char promotionChar() const noexcept {
        constexpr char CHARS[8] = {'\0', '\0', 'n', 'b', 'r', 'q', '\0', '\0'};
        return CHARS[promotionType & 7];
    }

    std::string toUCIString() const noexcept { return squareToString(from) + squareToString(to) + (promotionType ? std::string(1, promotionChar()) : std::string{}); }
};

class Board {
public:
    static constexpr const char* STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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

    // Members are intentionally NOT default-initialised: skips a full zero-fill per node.
    // Value-init sites (`MoveState st{}`) still zero everything.
    struct MoveState {
        uint64_t prevHistoryHead;
        // The repetitionHistory slot this move overwrites: an irreversible move
        // restarts the history at slot 0, and undo must give the entry back.
        uint64_t prevHistorySlotValue;

        uint8_t prevHalfMoveClock;
        uint8_t prevFullMoveClock;
        uint8_t prevHistorySize;

        Square  prevEnPassant;
        uint8_t prevEpHashFile;
        uint8_t prevCastle;

        uint8_t capturedPiece;
        uint8_t enPassantCapturedIndex;
        uint8_t rookFromIndex;
        uint8_t rookToIndex;
        uint8_t prevPliesFromNull;
        MoveKind moveKind;
    };

    static_assert(sizeof(MoveState) <= 96, "MoveState layout regressed; keep it compact for search stack usage.");

    // --- Static constants ---
    static constexpr uint8_t  MASK_PIECE              = 0x0F;
    static constexpr uint8_t  MASK_COLOR              = 0x08;
    static constexpr uint8_t  MASK_PIECE_TYPE         = 0x07;
    static constexpr uint8_t  CASTLING_RIGHTS_ALL     = 0x0F;
    static constexpr uint16_t REPETITION_HISTORY_CAPACITY = 255; // must be 255 instead of 100 due to the 8-bit prevHistorySize in MoveState

    // --- Constructors ---
    Board() noexcept;
    explicit Board(const std::string& fen);
    // A copy takes the current position's accumulator only, computed first.
    Board(const Board& other) noexcept;
    Board& operator=(const Board& other) noexcept;
    Board(Board&& other) noexcept = default;
    Board& operator=(Board&& other) noexcept = default;

    // --- Static utilities ---
    static constexpr uint8_t  oppositeColor(uint8_t color) noexcept { return color ^ 0x8; };
    static constexpr uint8_t  colorToIndex(uint8_t color) noexcept { return ((color & MASK_COLOR) >> 3) ^ 0x1u; };
    static constexpr uint8_t  promotionRank(bool isWhite) noexcept { return isWhite ? 0 : 7; };

    // --- Board access ---
    __attribute__((hot, always_inline)) constexpr uint8_t get(uint8_t index) const noexcept { return chessboard[index]; }

    // --- Move execution ---
    void doMove(const Move& m, MoveState& state) noexcept;
    void undoMove(const Move& m, const MoveState& state) noexcept;
    void doNullMove(MoveState& state) noexcept;
    void undoNullMove(const MoveState& state) noexcept;
    bool move(Move move) noexcept;

    // --- Legality & attack queries ---
    bool isLegalPseudoMove(uint8_t fromIndex, uint8_t toIndex, uint8_t fromPiece) const noexcept;
    bool isSquareAttacked(uint8_t targetIndex, uint8_t byColor, uint8_t excludeSquare = 64) const noexcept;
    bool inCheck(uint8_t color) const noexcept;
    // Bitboard of enemy pieces giving check to `color`'s king (0 = no check).
    // One scan answers inCheck (!=0), double check (>1 bit) and, via the
    // checker square, the evasion mask - callers should reuse it.
    uint64_t checkersTo(uint8_t color) const noexcept;
    [[nodiscard]] inline bool isKingSafeAfterMove(uint8_t movingColor, uint8_t fromIndex,
                                                   uint8_t toIndex, uint64_t capturedMask) const noexcept;

    // --- Game state queries ---
    bool isFiftyMoveRule() const noexcept { return halfMoveClock >= 100; }
    bool isThreefoldRepetition() const noexcept { return countRepetitions() >= 3; }
    int  countRepetitions() const noexcept;
    bool hasInsufficientMaterialDraw() const noexcept;

    // --- State accessors ---
    constexpr uint8_t  getActiveColor() const noexcept   { return activeColor; }
    constexpr uint8_t  castlingRights() const noexcept   { return castle; }
    constexpr uint16_t getFullMoveClock() const noexcept { return fullMoveClock; }
    constexpr uint8_t  getHalfMoveClock() const noexcept { return halfMoveClock; }
    Square             getEnPassant() const noexcept     { return enPassant; }
    constexpr uint64_t getHash() const noexcept          { return currentHash; }
    uint64_t           getPiecesBitMap() const noexcept  { return occupancy; }
    void               rebuildBitboardsFromSquares() noexcept;
    // From-scratch NNUE accumulator recompute (no-op when no net is loaded).
    // Incremental maintenance happens in addPieceToBB/removePieceFromBB.
    inline void        refreshNnueAccumulator() noexcept;
    // Computes the current position's accumulator if it is not yet, from the
    // nearest computed ancestor or the Finny table. Must be called with the
    // Board in a consistent state (NNUE::evaluate and the selftest do); const
    // because the accumulator is a cache of board-derived state.
    inline void        ensureNnueAccumulatorClean() const noexcept;
    // The current position's accumulator, valid after ensureNnueAccumulatorClean.
    const NNUE::Accumulator& accumulator() const noexcept { return frame(accTop); }

    // --- FEN ---
    void        fenToBoard(const std::string& fen);
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
    [[nodiscard]] inline bool isKingMoveLegal(uint8_t fromIndex, uint8_t toIndex,
                                               uint64_t toBit, uint8_t movingColor) const noexcept;
    template<uint8_t PieceType>
    [[nodiscard]] inline bool pseudoMoveLegalByType(uint8_t fromIndex, uint8_t toIndex, uint64_t toBit,
                                                    uint8_t movingColor, uint8_t destPiece) const noexcept;
    // Whether bySide attacks sq with `occ` as blockers, counting only the
    // attackers in `keep` (a capture takes one off the board).
    bool attackedBy(uint8_t sq, uint8_t bySide, uint64_t occ, uint64_t keep = ~0ULL) const noexcept;

    // --- Private helpers: move classification ---
    [[nodiscard]] static constexpr bool     isCaptureKind(MoveKind kind) noexcept;
    [[nodiscard]] static constexpr bool     isPromotionKind(MoveKind kind) noexcept;
    [[nodiscard]] static inline MoveKind    classifyMoveKind(uint8_t movingType, uint8_t movingColor,
                                                              uint8_t fromIndex, uint8_t toIndex,
                                                              uint8_t destBefore, const Square& prevEnPassant) noexcept;

    // --- Private helpers: FEN & hash ---
    static uint8_t safeParseInt(const std::string& section, int min, int max, int defaultValue);
    void           recomputeHashAndEp() noexcept;
    void           rebuildRepetitionHistory() noexcept;
    void           updateRepetitionAfterMove(bool resetHistory, MoveState& st) noexcept;
    inline void    copyFromBoard(const Board& other) noexcept;

    // --- Private data ---
    // One byte per square, engine index (a8 = 0, h1 = 63): a read is a single
    // load, where nibble packing cost a shift and a mask on every access.
    std::array<uint8_t, 64> chessboard;
    uint64_t currentHash = 0ULL;
    std::array<uint64_t, REPETITION_HISTORY_CAPACITY> repetitionHistory{};
    uint64_t occupancy   = 0ULL;

    uint8_t  halfMoveClock       = 0;
    uint8_t  fullMoveClock       = 1;
    uint8_t  castle              = CASTLING_RIGHTS_ALL;
    Square   enPassant           = NO_SQUARE;
    uint8_t  epHashFile          = 0xFF;
    uint8_t  activeColor         = WHITE;
    uint8_t  historySize         = 0;
    // Plies since the last null move, saturating; 255 = none within reach.
    uint8_t  pliesFromNull       = 255;

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

static_assert(Move::promotionTypeFromChar('q') == Board::QUEEN
           && Move::promotionTypeFromChar('r') == Board::ROOK
           && Move::promotionTypeFromChar('b') == Board::BISHOP
           && Move::promotionTypeFromChar('n') == Board::KNIGHT
           && Move{0, 0, Board::QUEEN}.promotionChar() == 'q'
           && Move{0, 0, Board::KNIGHT}.promotionChar() == 'n',
    "Move promotion helpers must mirror Board::piece_id");

// FEN piece placement of 64 squares (0 = a8), each a piece letter or '.'.
std::string placementFen(const std::array<char, 64>& squares);

#include "board.inl"

} // namespace chess
