#pragma once

#include <bit>
#include <string>
#include <array>
#include <cstdint>
#include <cctype>
#include <cstddef>
#include <cstring>

#include "../nnue/accumulator.hpp"
#include "./coords.hpp"
#include "./piece.hpp"

namespace chess {

using board = std::array<uint32_t, 8>;

struct Move {
    Square from = NO_SQUARE;
    Square to   = NO_SQUARE;
    // Board::piece_id type of the promotion piece (KNIGHT=2 .. QUEEN=5); 0 = not
    // a promotion. Char conversion happens only at the UCI/user-input boundary.
    uint8_t promotionType = 0;

    constexpr bool operator==(const Move&) const noexcept = default;
    constexpr bool sameFromTo(const Move& other) const noexcept { return from == other.from && to == other.to; };
    constexpr bool sameFromTo(int f, int t) const noexcept { return from == static_cast<uint8_t>(f) && to == static_cast<uint8_t>(t); };

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
    // --- Enums ---

    enum CastlingBits : uint8_t {
        WHITE_KINGSIDE  = 0,
        WHITE_QUEENSIDE = 1,
        BLACK_KINGSIDE  = 2,
        BLACK_QUEENSIDE = 3
    };

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
        // Old value of the single repetitionHistory slot this move overwrites,
        // so undoMove can restore it. Without this, an irreversible move during
        // search (which resets historySize to 0 and rewrites from index 0)
        // permanently clobbers earlier game-history entries even though
        // historySize is restored: silently breaking repetition detection in
        // every sibling line that follows a capture/pawn move.
        uint64_t prevHistorySlotValue;

        uint8_t prevHalfMoveClock;
        uint8_t prevFullMoveClock;
        uint8_t prevHistorySize;

        Square  prevEnPassant;
        uint8_t prevEpHashFile;
        uint8_t prevCastle;
        uint8_t prevHasMoved;

        uint8_t capturedPiece;
        uint8_t fromPiece;
        uint8_t promotionPieceType;
        uint8_t enPassantCapturedIndex;
        uint8_t rookFromIndex;
        uint8_t rookToIndex;
        MoveKind moveKind;
    };

    static_assert(sizeof(MoveState) <= 96, "MoveState layout regressed; keep it compact for search stack usage.");

    // --- Static constants ---
    static constexpr uint8_t  MASK_PIECE              = 0x0F;
    static constexpr uint8_t  MASK_COLOR              = 0x08;
    static constexpr uint8_t  MASK_PIECE_TYPE         = 0x07;
    static constexpr uint8_t  WHITE_ROOK_A_START      = 56;
    static constexpr uint8_t  WHITE_ROOK_H_START      = 63;
    static constexpr uint8_t  BLACK_ROOK_A_START      = 0;
    static constexpr uint8_t  BLACK_ROOK_H_START      = 7;
    static constexpr uint8_t  CASTLING_RIGHTS_ALL     = 0x0F;
    static constexpr uint16_t REPETITION_HISTORY_CAPACITY = 255; // must be 255 instead of 100 due to the 8-bit prevHistorySize in MoveState

    static constexpr std::array<uint64_t, 64> BIT_MASKS = []() constexpr {
        std::array<uint64_t, 64> masks{};
        for (int i = 0; i < 64; ++i) masks[i] = (1ULL << i);
        return masks;
    }();

    static constexpr std::array<uint8_t, 256> CHAR_TO_PIECE_TYPE = []() {
        std::array<uint8_t, 256> table{};
        for (int i = 0; i < 256; ++i) table[i] = EMPTY;
        table['P'] = PAWN | WHITE;   table['p'] = PAWN | BLACK;
        table['N'] = KNIGHT | WHITE; table['n'] = KNIGHT | BLACK;
        table['B'] = BISHOP | WHITE; table['b'] = BISHOP | BLACK;
        table['R'] = ROOK | WHITE;   table['r'] = ROOK | BLACK;
        table['Q'] = QUEEN | WHITE;  table['q'] = QUEEN | BLACK;
        table['K'] = KING | WHITE;   table['k'] = KING | BLACK;
        return table;
    }();

    // --- Constructors ---
    Board() noexcept;
    explicit Board(const std::string& fen);
    Board(const Board& other) noexcept;
    Board& operator=(const Board& other) noexcept;
    Board(Board&& other) noexcept = default;
    Board& operator=(Board&& other) noexcept = default;

    // --- Static utilities ---
    static constexpr uint8_t  oppositeColor(uint8_t color) noexcept { return color ^ 0x8; };
    static constexpr uint8_t  colorToIndex(uint8_t color) noexcept { return ((color & MASK_COLOR) >> 3) ^ 0x1u; };
    static constexpr uint8_t  promotionRank(bool isWhite) noexcept { return isWhite ? 0 : 7; };

    // --- Board access ---
    __attribute__((hot, always_inline)) constexpr uint8_t get(uint8_t index) const noexcept { return (chessboard[7 - (index >> 3)] >> ((index & 7) << 2)) & MASK_PIECE; }
    __attribute__((always_inline))      constexpr uint8_t get(uint8_t row, uint8_t col) const noexcept { return (chessboard[row] >> (col << 2)) & MASK_PIECE; }
    __attribute__((always_inline))      constexpr uint8_t getColor(uint8_t index) const noexcept { return (get(index) & MASK_COLOR) ? WHITE : BLACK; }
    __attribute__((hot, always_inline)) inline void set(uint8_t index, piece_id value) noexcept;

    __attribute__((always_inline)) void fastUpdateOccupancyBB(uint8_t fromIndex, uint8_t toIndex) noexcept;
    __attribute__((always_inline)) void addPieceToBB(uint8_t piece, uint8_t index) noexcept;
    __attribute__((always_inline)) void removePieceFromBB(uint8_t piece, uint8_t index) noexcept;
    // Un pezzo che si sposta: stessa cosa di remove+add, ma l'accumulatore fa
    // una passata invece di due (vedi Accumulator::updateMove).
    __attribute__((always_inline)) void movePieceOnBB(uint8_t piece, uint8_t fromIndex,
                                                      uint8_t toIndex) noexcept;

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
    // checker square, the evasion mask — callers should reuse it.
    uint64_t checkersTo(uint8_t color) const noexcept;
    [[nodiscard]] inline bool isKingSafeAfterMove(uint8_t movingColor, uint8_t fromIndex,
                                                   uint8_t toIndex, uint64_t capturedMask) const noexcept;

    // --- Game state queries ---
    __attribute__((hot)) bool isCheckmate(uint8_t color) const noexcept { return inCheck(color) && !hasAnyLegalMove(color); }
    bool hasAnyLegalMove(uint8_t color) const noexcept;
    bool isStalemate(uint8_t color) const noexcept { return !inCheck(color) && !hasAnyLegalMove(color); }
    bool isFiftyMoveRule() const noexcept { return halfMoveClock >= 100; }
    bool isDraw(uint8_t color) const noexcept { return isStalemate(color) || isFiftyMoveRule() || isThreefoldRepetition() || hasInsufficientMaterialDraw(); }
    bool isThreefoldRepetition() const noexcept { return countRepetitions() >= 3; }
    int  countRepetitions() const noexcept;
    bool hasInsufficientMaterialDraw() const noexcept;

    // --- State accessors ---
    constexpr uint8_t  getActiveColor() const noexcept   { return activeColor; }
    constexpr bool     getCastle(uint8_t index) const noexcept { return (castle & (1u << index)); }
    constexpr uint16_t getFullMoveClock() const noexcept { return fullMoveClock; }
    constexpr uint8_t  getHalfMoveClock() const noexcept { return halfMoveClock; }
    Square             getEnPassant() const noexcept     { return enPassant; }
    constexpr uint64_t getHash() const noexcept          { return currentHash; }
    uint64_t           getPiecesBitMap() const noexcept  { return occupancy; }
    void               rebuildBitboardsFromSquares() noexcept;
    // From-scratch NNUE accumulator recompute (no-op when no net is loaded).
    // Incremental maintenance happens in addPieceToBB/removePieceFromBB.
    inline void        refreshNnueAccumulator() noexcept;
    // HalfKA lazy refresh: rebuilds any perspective marked dirty by an
    // own-king bucket/flip crossing. Must be called with the Board in a
    // consistent state (NNUE::evaluate and the selftest do); const because
    // the accumulator is a cache of board-derived state.
    inline void        ensureNnueAccumulatorClean() const noexcept;

    // --- FEN ---
    void        fenToBoard(const std::string& fen);
    std::string boardToFen() const;

    // --- Public bitboards (direct access for eval/search hot paths) ---
    std::array<uint64_t, 2> pawns_bb   = {0ULL, 0ULL};
    std::array<uint64_t, 2> knights_bb = {0ULL, 0ULL};
    std::array<uint64_t, 2> bishops_bb = {0ULL, 0ULL};
    std::array<uint64_t, 2> rooks_bb   = {0ULL, 0ULL};
    std::array<uint64_t, 2> queens_bb  = {0ULL, 0ULL};
    std::array<uint64_t, 2> kings_bb   = {0ULL, 0ULL};

    // NNUE dual-perspective accumulator; kept in sync with the bitboards by
    // the same add/remove piece functions whenever a network is loaded.
    // Contains garbage until the first refreshNnueAccumulator() after load.
    // mutable: ensureNnueAccumulatorClean() settles the HalfKA lazy state
    // from const evaluation paths — it is a cache, not board state.
    mutable NNUE::Accumulator nnueAccumulator;

private:
    // --- Private helpers: move execution ---
    inline void snapshotState(MoveState& st) const noexcept;
    inline void prepareMoveState(MoveState& st, uint8_t moving, uint8_t destBefore) const noexcept;
    inline void prepareNullMoveState(MoveState& st) const noexcept;
    inline void restoreState(const MoveState& st) noexcept;
    template<MoveKind Kind>
    inline void doMoveByKind(MoveState& st, uint8_t moving, uint8_t movingType,
                              uint8_t movingColor, uint8_t destBefore,
                              uint8_t fromIndex, uint8_t toIndex, uint8_t promotionType) noexcept;
    template<MoveKind Kind>
    inline void undoMoveByKind(const MoveState& st, uint8_t& pieceOnTo,
                                uint8_t fromIndex, uint8_t toIndex) noexcept;
    inline void promoteUnchecked(uint8_t atIndex, uint8_t pawnPiece, uint8_t promo) noexcept;

    // --- Private helpers: castling ---
    [[nodiscard]] inline bool canCastleGeneric(bool isWhite, uint8_t fromIndex, bool isKingside) const noexcept;
    [[nodiscard]] static inline uint8_t rookStartSlot(uint8_t index) noexcept;
    inline void clearCastlingByRookStart(uint8_t rookStartIndex, bool setHasMovedBit) noexcept;
    inline void updateCastlingRightsOnPieceMove(uint8_t movingType, uint8_t movingColor, uint8_t fromIndex) noexcept;
    inline void updateCastlingRightsOnRookCapture(uint8_t capturedPiece, uint8_t toIndex) noexcept;

    // --- Private helpers: legality ---
    [[nodiscard]] inline bool isKingMoveLegal(uint8_t fromIndex, uint8_t toIndex,
                                               uint64_t toBit, uint8_t movingColor) const noexcept;
    template<uint8_t PieceType>
    [[nodiscard]] bool hasLegalMovesForPieceType(uint64_t pieceBB, uint64_t ownOcc,
                                                  uint64_t enemyOcc, uint8_t movingColor) const noexcept;
    template<uint8_t PieceType>
    [[nodiscard]] inline bool pseudoMoveLegalByType(uint8_t fromIndex, uint8_t toIndex, uint64_t toBit,
                                                    uint8_t movingColor, uint8_t destPiece) const noexcept;
    static bool isKingAttackedCustom(uint8_t kingSq, uint8_t bySide, uint64_t occ,
                                      uint64_t pawns, uint64_t knights, uint64_t bishops,
                                      uint64_t rooks, uint64_t queens, uint64_t kings) noexcept;

    // --- Private helpers: bitboards & incremental counters ---
    template<uint8_t PieceType, bool Add>
    inline void updatePieceTypeBB(uint8_t color, uint64_t bit) noexcept;
    template<bool Add>
    inline void dispatchPieceBBUpdate(uint8_t pieceType, uint8_t color, uint64_t bit) noexcept;

    // --- Private helpers: move classification ---
    [[nodiscard]] static constexpr bool     isCaptureKind(MoveKind kind) noexcept;
    [[nodiscard]] static constexpr bool     isPromotionKind(MoveKind kind) noexcept;
    [[nodiscard]] static inline MoveKind    classifyMoveKind(uint8_t movingType, uint8_t movingColor,
                                                              uint8_t fromIndex, uint8_t toIndex,
                                                              uint8_t destBefore, const Square& prevEnPassant) noexcept;
    [[nodiscard]] static inline uint8_t     normalizePromotionType(uint8_t promoType) noexcept;

    // --- Private helpers: FEN & hash ---
    static uint8_t safeParseInt(const std::string& section, int min, int max, int defaultValue);
    void           recomputeHashAndEp() noexcept;
    void           rebuildRepetitionHistory() noexcept;
    void           updateRepetitionAfterMove(bool resetHistory, MoveState& st) noexcept;
    inline void    copyFromBoard(const Board& other) noexcept;

    // --- Private data ---
    board    chessboard;
    uint64_t currentHash = 0ULL;
    std::array<uint64_t, REPETITION_HISTORY_CAPACITY> repetitionHistory{};
    uint64_t occupancy   = 0ULL;

    uint8_t  halfMoveClock       = 0;
    uint8_t  fullMoveClock       = 1;
    uint8_t  castle              = CASTLING_RIGHTS_ALL;
    uint8_t  hasMoved            = 0x00;
    Square   enPassant           = NO_SQUARE;
    uint8_t  epHashFile          = 0xFF;
    uint8_t  activeColor         = WHITE;
    uint8_t  historySize         = 0;
    uint8_t  nullPly             = 0;

    static constexpr const char* STARTING_FEN =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
};

static_assert(Move::promotionTypeFromChar('q') == Board::QUEEN
           && Move::promotionTypeFromChar('r') == Board::ROOK
           && Move::promotionTypeFromChar('b') == Board::BISHOP
           && Move::promotionTypeFromChar('n') == Board::KNIGHT
           && Move{0, 0, Board::QUEEN}.promotionChar() == 'q'
           && Move{0, 0, Board::KNIGHT}.promotionChar() == 'n',
    "Move promotion helpers must mirror Board::piece_id");

#include "board.inl"
#include "boardapi.inl"

} // namespace chess
