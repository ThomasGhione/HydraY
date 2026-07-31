#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "../board/board.hpp"
#include "../tt/tt.hpp"

#include "search/searcher.hpp"
#include "time/time_manager.hpp"
#include "syzygy/syzygy.hpp"

namespace engine {

static inline constexpr int32_t POS_INF = std::numeric_limits<int32_t>::max();
// Negamax-safe: NEG_INF == -POS_INF so it can be negated without UB.
static inline constexpr int32_t NEG_INF = -POS_INF;

class Engine final {
public:
    enum class GameResult : uint8_t {
        ONGOING = 0,
        WHITE_WINS = 1,
        BLACK_WINS = 2,
        DRAW = 3
    };

    // Lifecycle
    Engine();
    ~Engine() noexcept;

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    // Gameplay API
    void reset() noexcept;
    bool movePiece(const chess::Square from, const chess::Square to, const char promotionPiece = '\0') noexcept;

    // Search API
    void search(int depth) noexcept;
    chess::Move searchUCI(const time::Limits& limits) noexcept;
    void stopThinking() noexcept;
    void setSearchApiMutexEnabled(bool enabled) noexcept;
    bool isSearchApiMutexEnabled() const noexcept;

    // Game state
    bool isGameOver() const noexcept { return gameResult != GameResult::ONGOING; }
    bool isMate() const noexcept { return gameResult == GameResult::WHITE_WINS || gameResult == GameResult::BLACK_WINS; }
    bool isStalemate() const noexcept { return gameResult == GameResult::DRAW && board.isStalemate(board.getActiveColor()); }
    bool isDraw() const noexcept { return gameResult == GameResult::DRAW; }
    void updateGameResult() noexcept;

    // Public state kept for compatibility with existing call-sites.
    chess::Move bestMove;
    chess::Board board;
    bool isPlayerWhite = true;

    // Unified runtime state owned by Searcher.
    Searcher::SearchRuntime searchRuntime{};

    // UCI `Threads` override. 0 = auto (use omp_get_max_threads()). Persists
    // across reset()/ucinewgame so a GUI/cutechess setting survives new games.
    int requestedThreads = 0;

    static constexpr int32_t DEFAULTDEPTH = Searcher::DEFAULT_DEPTH;
    static constexpr int32_t MAX_PLY = Searcher::MAX_PLY;

    static constexpr size_t MOVE_HISTORY_MAX_PLIES = 1024;
    static constexpr size_t MOVE_HISTORY_ENTRY_MAX_LEN = 6; // "e7e8q\n"
    static constexpr size_t MOVE_HISTORY_MAX_BYTES = MOVE_HISTORY_MAX_PLIES * MOVE_HISTORY_ENTRY_MAX_LEN;
    std::string moveHistory {};

    // Syzygy tablebases
    syzygy::SyzygyProber syzygyProber;

    // Transposition table (shared by normal search and pondering)
    TT tt;

    // Per-move time budget + hard-deadline watchdog (UCI clock searches).
    time::TimeManager timeManager;

private:

    GameResult gameResult = GameResult::ONGOING;

    // Search/pondering coordination
    std::thread ponderingThread;
    std::mutex ponderingMutex;
    std::condition_variable ponderingCv;
    chess::Board ponderingBoard {};
    bool ponderingWorkReady = false;
    bool ponderingWorkerStopping = false;
    std::atomic<bool> ponderingStopRequested {false};
    std::atomic<bool> ponderingActive {false};
    std::atomic<bool> stopSearchRequested {false};
    std::atomic<bool> searchInterrupted {false};
    std::mutex searchApiMutex;
    std::atomic<bool> searchApiMutexEnabled {true};
    uint64_t ponderRootHash = 0;
    int      ponderResultDepth = 0;
    int32_t ponderResultScore = 0;
    chess::Move ponderResultMove {};
    bool ponderResultReady = false;

    // Shared bitboard init (all Engine instances); only the ctor touches these.
    static inline bool magicTablesInitialized = false;
    static void ensureMagicTablesInitialized() noexcept;

    // Internal helpers
    void bindSearchRuntime() noexcept;
    void appendMoveHistoryEntry(const chess::Square& from, const chess::Square& to, char promotionPiece) noexcept;
    bool playMoveOnBoard(const chess::Move& move) noexcept;
    void clearPonderResult() noexcept;
    void clearSearchStopFlags() noexcept;
    std::optional<chess::Move> tryInstantMove(int targetDepth) noexcept;
    chess::Move commitSearchResult(const chess::Move& candidate) noexcept;
    std::unique_lock<std::mutex> acquireSearchApiLock() noexcept;
    void requestStopPondering() noexcept;
    bool tryUsePonderResult(int targetDepth, chess::Move& outMove) noexcept;
    void startPondering() noexcept;
    void stopPondering() noexcept;
    bool waitForPonderJob(chess::Board& outBoard) noexcept;
    void ponderWorkerLoop() noexcept;
    void ponderLoop(chess::Board&& rootBoard) noexcept;
};

} // namespace engine

#include "inl/bitboard_helpers.inl"
