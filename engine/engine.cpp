
#include "engine.hpp"

#include <iostream>

#include <omp.h>

#include "../debug.hpp"
#include "../nnue/nnue.hpp"

namespace engine {

namespace {

[[nodiscard]] bool resolveSearchApiMutexGuardDefault() noexcept {
    const char* envValue = std::getenv("CHESS_ENGINE_SEARCH_MUTEX_GUARD");
    if (envValue == nullptr) return true;

    // Any value other than an explicit off-switch (incl. empty) enables the guard.
    const std::string_view value(envValue);
    return !(ascii::iequals(value, "0") || ascii::iequals(value, "off") || ascii::iequals(value, "false"));
}

[[nodiscard]] chess::Move getFallbackPonderMove(
    chess::Board& board,
    const Searcher::SearchRuntime& sourceRuntime) noexcept {

    Searcher::SearchRuntime runtime{};
    runtime.maxThreads = sourceRuntime.maxThreads;
    return Searcher::searchBestMove(board, runtime, 1);
}

} // namespace

void Engine::bindSearchRuntime() noexcept {
    searchRuntime.transpositionTable = &tt;
    searchRuntime.stopSearchRequested = &stopSearchRequested;
    searchRuntime.ponderingStopRequested = &ponderingStopRequested;
    searchRuntime.searchInterrupted = &searchInterrupted;
    searchRuntime.syzygyProber = &syzygyProber;
}

void Engine::ensureMagicTablesInitialized() noexcept {
    if (!magicTablesInitialized) {
        pieces::initMagicBitboards();
        magicTablesInitialized = true;
    }
}

Engine::Engine() {

    ensureMagicTablesInitialized();

    // NNUE is the only evaluator: guarantee an active network before any
    // search from this Engine. Test harnesses construct Engine directly and
    // bypass main()'s activation; searching with no network would crash.
    // The board member is built before this body runs, so refresh its
    // accumulator after activating.
    if (!NNUE::networkLoaded() && NNUE::activateEmbedded()) {
        this->board.refreshNnueAccumulator();
    }

    searchApiMutexEnabled.store(resolveSearchApiMutexGuardDefault(), std::memory_order_relaxed);
    searchRuntime.maxThreads = omp_get_max_threads();

    bindSearchRuntime();
    this->tt.clear();
    this->ponderingThread = std::thread([this] { this->ponderWorkerLoop(); });
}

Engine::~Engine() noexcept {
    {
        std::lock_guard<std::mutex> lock(this->ponderingMutex);
        this->ponderingWorkerStopping = true;
        this->ponderingWorkReady = false;
    }
    this->requestStopPondering();
    this->ponderingCv.notify_one();
    if (this->ponderingThread.joinable()) this->ponderingThread.join();
}

void Engine::reset() noexcept {
    // Stop the watchdog first: if reset arrives mid-search, a still-running
    // watchdog would set stopSearchRequested *after* this reset returns and
    // abort whatever search starts next.
    this->timeManager.stop();
    this->stopPondering();
    this->clearPonderResult();
    board = chess::Board();
    bestMove = chess::Move{};
    moveHistory.clear();
    isPlayerWhite = true;
    gameResult = GameResult::ONGOING;

    searchRuntime = Searcher::SearchRuntime{};
    searchRuntime.maxThreads = (requestedThreads > 0) ? requestedThreads : omp_get_max_threads();
    bindSearchRuntime();

    this->tt.clear();
}

void Engine::setSearchApiMutexEnabled(bool enabled) noexcept {
    this->searchApiMutexEnabled.store(enabled, std::memory_order_release);
}

bool Engine::isSearchApiMutexEnabled() const noexcept {
    return this->searchApiMutexEnabled.load(std::memory_order_acquire);
}

// Returns a guard already holding searchApiMutex when the guard is enabled,
// or an unlocked guard otherwise. Lets the search entry points share one line.
std::unique_lock<std::mutex> Engine::acquireSearchApiLock() noexcept {
    std::unique_lock<std::mutex> guard(this->searchApiMutex, std::defer_lock);
    if (this->searchApiMutexEnabled.load(std::memory_order_acquire)) {
        guard.lock();
    }
    return guard;
}

void Engine::clearPonderResult() noexcept {
    this->ponderRootHash = 0;
    this->ponderResultDepth = 0;
    this->ponderResultScore = 0;
    this->ponderResultMove = chess::Move{};
    this->ponderResultReady = false;
}

void Engine::clearSearchStopFlags() noexcept {
    this->stopSearchRequested.store(false, std::memory_order_relaxed);
    this->searchInterrupted.store(false, std::memory_order_relaxed);
}

// A move available without a full search: a usable ponder result, if any.
// nullopt means the caller must actually search.
std::optional<chess::Move> Engine::tryInstantMove(int targetDepth) noexcept {
    chess::Move ponderMove{};
    if (this->tryUsePonderResult(targetDepth, ponderMove)) return ponderMove;
    return std::nullopt;
}

// Stores `candidate` as bestMove (normalising an out-of-bounds search result
// to the empty sentinel) and returns it.
chess::Move Engine::commitSearchResult(const chess::Move& candidate) noexcept {
    const bool playable = chess::isValidSquare(candidate.from)
                       && chess::isValidSquare(candidate.to);
    this->bestMove = playable ? candidate : chess::Move{};
    return this->bestMove;
}

// Plays "move" on the engine's own board, recording it in the move history and
// refreshing the game result. Returns false (board untouched) when the move is
// out of bounds or illegal.
bool Engine::playMoveOnBoard(const chess::Move& move) noexcept {
    if (!chess::isValidSquare(move.from) || !chess::isValidSquare(move.to)) return false;
    if (!this->board.move(move)) return false;

    this->appendMoveHistoryEntry(move.from, move.to, move.promotionChar());
    this->updateGameResult();
    return true;
}

__attribute__((hot))
bool Engine::movePiece(const chess::Square from, const chess::Square to, const char promotionPiece) noexcept {
    this->requestStopPondering();
    return this->playMoveOnBoard(
        chess::Move{from, to, chess::Move::promotionTypeFromChar(promotionPiece)});
}

void Engine::appendMoveHistoryEntry(const chess::Square& from, const chess::Square& to, char promotionPiece) noexcept {
    const size_t appendLen = (promotionPiece == '\0') ? size_t{5} : size_t{6};

    if (moveHistory.size() + appendLen > MOVE_HISTORY_MAX_BYTES) {
        const size_t overflow = (moveHistory.size() + appendLen) - MOVE_HISTORY_MAX_BYTES;
        const size_t firstErase = std::min(overflow, moveHistory.size());
        moveHistory.erase(0, firstErase);

        const size_t lineEnd = moveHistory.find('\n');
        if (lineEnd != std::string::npos) {
            moveHistory.erase(0, lineEnd + 1);
        } else {
            moveHistory.clear();
        }
    }

    moveHistory.reserve(moveHistory.size() + appendLen);
    moveHistory += chess::squareToString(from);
    moveHistory += chess::squareToString(to);
    if (promotionPiece != '\0') {
        moveHistory += promotionPiece;
    }
    moveHistory += '\n';
}

void Engine::updateGameResult() noexcept {
    gameResult = GameResult::ONGOING;
    const uint8_t toMove = board.getActiveColor();
    if (board.kings_bb[0] == 0) {
        gameResult = GameResult::BLACK_WINS;
    } else if (board.kings_bb[1] == 0) {
        gameResult = GameResult::WHITE_WINS;
    } else if (board.isCheckmate(toMove)) {
        gameResult = (toMove == chess::Board::WHITE) ? GameResult::BLACK_WINS : GameResult::WHITE_WINS;
    } else if (board.isDraw(toMove)) {
        gameResult = GameResult::DRAW;
    }
}

void Engine::ponderLoop(chess::Board&& rootBoard) noexcept {
    this->ponderRootHash = rootBoard.getHash();
    this->clearSearchStopFlags();
    this->searchRuntime.nodesSearched = 0;
    this->tt.incrementGeneration();

    const Searcher::IterativeSearchResult ponderResult = Searcher::runIterativeDeepening(
        rootBoard, this->searchRuntime, Engine::DEFAULTDEPTH, Engine::MAX_PLY);

    this->ponderResultMove = ponderResult.bestMove;
    this->ponderResultScore = ponderResult.bestScore;
    this->ponderResultDepth = ponderResult.completedDepth;

    this->ponderResultReady = ponderResult.hasLegalMoves
        && ponderResult.completedAnyDepth
        && chess::isValidSquare(ponderResult.bestMove.from)
        && chess::isValidSquare(ponderResult.bestMove.to);

    this->ponderingActive.store(false, std::memory_order_release);
}

void Engine::requestStopPondering() noexcept {
    this->ponderingStopRequested.store(true, std::memory_order_release);
    this->stopSearchRequested.store(true, std::memory_order_release);
}

bool Engine::tryUsePonderResult(int targetDepth, chess::Move& outMove) noexcept {
    if (!this->ponderResultReady) return false;
    if (this->ponderRootHash != this->board.getHash()) return false;
    if (this->ponderResultDepth < targetDepth) return false;

    outMove = this->ponderResultMove;
    this->searchRuntime.depth = this->ponderResultDepth;
    this->searchRuntime.eval = this->ponderResultScore;
    return true;
}

void Engine::startPondering() noexcept {
    if (this->isGameOver()) return;

    this->stopPondering();
    this->clearPonderResult();

    chess::Board rootBoard = this->board;
    auto ponderMove = this->tt.probeDecodedMove(rootBoard.getHash());
    if (!chess::isValidSquare(ponderMove.from)) { // did the TT fail?
        ponderMove = getFallbackPonderMove(rootBoard, this->searchRuntime);
    }
    if (!chess::isValidSquare(ponderMove.from)) return; // did the fallback fail too?

    if (!rootBoard.move(ponderMove)) {
        rootBoard = this->board;
        ponderMove = getFallbackPonderMove(rootBoard, this->searchRuntime);
        if (!chess::isValidSquare(ponderMove.from)
            || !rootBoard.move(ponderMove)) {
            return;
        }
    }

    this->ponderingStopRequested.store(false, std::memory_order_release);
    this->stopSearchRequested.store(false, std::memory_order_release);
    this->searchInterrupted.store(false, std::memory_order_release);
    this->ponderingActive.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(this->ponderingMutex);
        this->ponderingBoard = std::move(rootBoard);
        this->ponderingWorkReady = true;
    }
    this->ponderingCv.notify_one();
}

void Engine::stopPondering() noexcept {
    this->requestStopPondering();

    {
        std::unique_lock<std::mutex> lock(this->ponderingMutex);
        this->ponderingCv.wait(lock, [this] {
            return !this->ponderingWorkReady && !this->ponderingActive.load(std::memory_order_relaxed);
        });
    }

    this->ponderingActive.store(false, std::memory_order_release);
    this->ponderingStopRequested.store(false, std::memory_order_release);
    this->stopSearchRequested.store(false, std::memory_order_release);
    this->searchInterrupted.store(false, std::memory_order_release);
}

bool Engine::waitForPonderJob(chess::Board& outBoard) noexcept {
    std::unique_lock<std::mutex> lock(this->ponderingMutex);
    this->ponderingCv.wait(lock, [this] {
        return this->ponderingWorkReady || this->ponderingWorkerStopping;
    });
    if (this->ponderingWorkerStopping) return false;
    outBoard = std::move(this->ponderingBoard);
    this->ponderingWorkReady = false;
    return true;
}

void Engine::ponderWorkerLoop() noexcept {
    chess::Board rootBoard;
    while (waitForPonderJob(rootBoard)) {
        this->ponderLoop(std::move(rootBoard));
        this->ponderingCv.notify_all();
    }
    this->ponderingActive.store(false, std::memory_order_release);
    this->ponderingCv.notify_all();
}

void Engine::stopThinking() noexcept {
    this->requestStopPondering();
}

chess::Move Engine::searchUCI(const time::Limits& limits) noexcept {
    auto searchApiGuard = acquireSearchApiLock();

    this->stopPondering();

    const bool sideIsWhite =
        this->board.getActiveColor() == chess::Board::WHITE;
    const int movesPlayed = static_cast<int>(
        this->board.getFullMoveClock() > 0 ? this->board.getFullMoveClock() - 1 : 0);

    this->clearSearchStopFlags();

    this->timeManager.init(limits, sideIsWhite, movesPlayed,
                           &this->stopSearchRequested);

    // Depth-bounded request keeps its cap; a time-managed search deepens
    // until the budget runs out (the watchdog trips the stop flag).
    int targetDepth;
    if (limits.maxDepth > 0) {
        targetDepth = static_cast<int>(limits.maxDepth);
    } else if (this->timeManager.useTimeManagement() || limits.infinite) {
        // Time-managed / infinite: deepen until the watchdog or an external
        // stop interrupts the search.
        targetDepth = Searcher::MAX_PLY;
    } else {
        // Ponder (or a bare `go`): this engine's `ponderhit` does not convert
        // a running search into a timed one, so a ponder search MUST be
        // bounded or it never returns a move on ponderhit and the engine
        // flags. Keep the pre-time-management fixed-depth behaviour.
        targetDepth = Engine::DEFAULTDEPTH;
    }

    if (auto instant = tryInstantMove(targetDepth)) {
        this->bestMove = *instant;
        return this->bestMove;
    }

    this->clearPonderResult();

    this->searchRuntime.timeManager = &this->timeManager;
    this->searchRuntime.maxNodes    = limits.maxNodes;
    this->timeManager.start();

    chess::Board searchBoard = this->board;
    const chess::Move candidate =
        Searcher::searchBestMove(searchBoard, this->searchRuntime, targetDepth);

    this->timeManager.stop();
    this->searchRuntime.timeManager = nullptr;
    this->searchRuntime.maxNodes    = 0;

    return commitSearchResult(candidate);
}

void Engine::search(int requestedDepth) noexcept {
    // Terminal-mode play: compute the move via the one search entry point, then
    // apply it on our board and ponder the reply. searchUCI already set bestMove
    // (a maxDepth-only Limits runs to depth with no time management).
    const chess::Move candidate =
        searchUCI(time::Limits{.maxDepth = requestedDepth});

    if (!this->playMoveOnBoard(candidate)) {
        this->bestMove = chess::Move{};
        this->updateGameResult();
        return;
    }

    this->clearPonderResult();
    this->startPondering(); // no-op when the move just played ended the game

    DBG_ONLY(
        std::string moveStr = chess::squareToString(candidate.from) + chess::squareToString(candidate.to);
        if (candidate.promotionType != 0) {
            moveStr += candidate.promotionChar();
        }
        DBG_LOG_STREAM("Engine plays: " << moveStr << " (score: " << this->searchRuntime.eval << ")\n");
    );
}

} // namespace engine
