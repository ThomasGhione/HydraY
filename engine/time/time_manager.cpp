#include "time_manager.hpp"

#include <algorithm>

namespace engine::time {

TimeManager::~TimeManager() noexcept {
    stop();
}

int TimeManager::estimateMovesToGo(int movesPlayed) noexcept {
    // No movestogo (sudden death / increment): assume a decaying horizon.
    // Errs toward saving time as the game grows long.
    const int est = DEFAULT_MOVESTOGO - std::max(0, movesPlayed);
    return std::max(MIN_MOVESTOGO, est);
}

void TimeManager::init(const Limits& limits, bool sideIsWhite, int movesPlayed,
                       std::atomic<bool>* stopFlag) noexcept {
    stop(); // ensure no stale watchdog from a previous search

    stopFlag_        = stopFlag;
    stabilityFactor_ = 1.0;
    baseMs_          = 0;
    softCapMs_       = 0;
    softMs_          = 0;
    hardMs_          = 0;

    // Modes that bypass time management entirely: the search runs until an
    // external stop, a node cap, or a depth cap. No clock and no movetime =>
    // no budget. This also covers pure depth/node-bound `go` commands, since
    // those carry neither a clock nor a movetime.
    if (limits.infinite || limits.ponder ||
        (!limits.hasClock && limits.movetime <= 0)) {
        useTm_ = false;
        return;
    }

    // Fixed time per move: soft == hard, only the overhead is shaved off.
    if (limits.movetime > 0) {
        const int64_t t = std::max<int64_t>(
            MIN_THINK_MS, limits.movetime - MOVE_OVERHEAD_MS);
        useTm_     = true;
        baseMs_    = t;
        softCapMs_ = t;
        softMs_    = t;
        hardMs_    = t;
        return;
    }

    // Clock-based allocation.
    const int64_t myTime = sideIsWhite ? limits.wtime : limits.btime;
    const int64_t myInc  = std::max<int64_t>(
        0, sideIsWhite ? limits.winc : limits.binc);

    const int64_t timeLeft =
        std::max<int64_t>(1, myTime - MOVE_OVERHEAD_MS);

    const int mtg = (limits.movestogo > 0)
        ? limits.movestogo
        : estimateMovesToGo(movesPlayed);

    // Base target: even share of the clock + most of the increment.
    double base = static_cast<double>(timeLeft) / static_cast<double>(mtg)
                + static_cast<double>(myInc) * INC_FRACTION;

    // Opening ramp: spend less on the first moves. Only applied WITHOUT an
    // increment: with an increment the clock is replenished every move, so
    // there is no reason to under-think the opening. No increment => the
    // clock only ever shrinks, so the early-game discount still protects it.
    if (myInc <= 0 && movesPlayed < OPENING_MOVES) {
        const double r = OPENING_MIN_SCALE +
            (1.0 - OPENING_MIN_SCALE) * (static_cast<double>(movesPlayed) /
                                         static_cast<double>(OPENING_MOVES));
        base *= r;
    }

    softCapMs_ = static_cast<int64_t>(
        static_cast<double>(timeLeft) * OPT_MAX_FRACTION);
    softCapMs_ = std::max<int64_t>(MIN_THINK_MS, softCapMs_);

    baseMs_ = std::clamp<int64_t>(
        static_cast<int64_t>(base), MIN_THINK_MS, softCapMs_);

    // Hard ceiling: never risk the clock. Fixed at init so it does not
    // fluctuate with stability feedback.
    const int64_t hardByFrac = static_cast<int64_t>(
        static_cast<double>(timeLeft) * HARD_MAX_FRACTION);
    const int64_t hardByMult = static_cast<int64_t>(
        static_cast<double>(baseMs_) * HARD_MULT);
    hardMs_ = std::max<int64_t>(baseMs_, std::min(hardByFrac, hardByMult));
    hardMs_ = std::min<int64_t>(hardMs_, timeLeft);

    useTm_ = true;
    recomputeSoft();
}

void TimeManager::recomputeSoft() noexcept {
    const double scaled = static_cast<double>(baseMs_) * stabilityFactor_;
    int64_t s = std::clamp<int64_t>(
        static_cast<int64_t>(scaled), MIN_THINK_MS, softCapMs_);
    s = std::min<int64_t>(s, hardMs_);
    softMs_ = std::max<int64_t>(MIN_THINK_MS, s);
}

void TimeManager::updateStability(bool bestMoveChanged, int32_t score,
                                   int32_t prevScore,
                                   bool hasPrevScore) noexcept {
    if (!useTm_) return;

    double f = stabilityFactor_;

    if (bestMoveChanged) {
        f *= 1.35;                       // root move still unsettled
    } else {
        f *= 0.90;                       // converging on a move
    }

    if (hasPrevScore) {
        const int64_t drop =
            static_cast<int64_t>(prevScore) - static_cast<int64_t>(score);
        if (drop > 30) {
            // Score is falling: things look worse than last iteration.
            const double extra = std::min(1.0,
                static_cast<double>(drop) / 200.0);
            f *= (1.0 + extra);          // panic time, up to +100%
        }
        // Decisive and stable -> the move barely matters, move on.
        const int32_t absScore = score < 0 ? -score : score;
        if (absScore > 1500 && !bestMoveChanged) {
            f *= 0.60;
        }
    }

    stabilityFactor_ = std::clamp(f, STABILITY_MIN, STABILITY_MAX);
    recomputeSoft();
}

void TimeManager::start() noexcept {
    startTp_ = Clock::now();
    if (!useTm_ || hardMs_ <= 0) return;

    {
        std::lock_guard<std::mutex> lock(mtx_);
        cancel_  = false;
        running_ = true;
    }
    watchdog_ = std::thread([this] { watchdogLoop(); });
}

void TimeManager::stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!running_ && !watchdog_.joinable()) return;
        cancel_ = true;
    }
    cv_.notify_all();
    if (watchdog_.joinable()) watchdog_.join();
    std::lock_guard<std::mutex> lock(mtx_);
    running_ = false;
}

void TimeManager::watchdogLoop() noexcept {
    std::unique_lock<std::mutex> lock(mtx_);
    const auto deadline =
        startTp_ + std::chrono::milliseconds(hardMs_);
    // Wake either at the hard deadline or when stop() cancels us.
    cv_.wait_until(lock, deadline, [this] { return cancel_; });
    if (!cancel_ && stopFlag_ != nullptr) {
        stopFlag_->store(true, std::memory_order_release);
    }
}

int64_t TimeManager::elapsedMs() const noexcept {
    const auto now = Clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now - startTp_).count();
}

bool TimeManager::shouldStartNextDepth() const noexcept {
    if (!useTm_) return true;
    const int64_t threshold = static_cast<int64_t>(
        static_cast<double>(softMs_) * START_NEXT_FRACTION);
    return elapsedMs() < std::max<int64_t>(MIN_THINK_MS, threshold);
}

} // namespace engine::time
