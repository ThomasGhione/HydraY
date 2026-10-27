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
    pondering_.store(false, std::memory_order_relaxed);
    stabilityFactor_ = 1.0;
    fixedTime_       = false;
    baseMs_          = 0;
    softCapMs_       = 0;
    softMs_          = 0;
    hardMs_          = 0;

    // No clock and no movetime (depth, nodes, infinite): no budget, the search
    // runs until it is stopped or capped. A `go ponder` with a clock gets its
    // budget now and spends it from ponderhit.
    if (limits.infinite || (!limits.hasClock && limits.movetime <= 0)) {
        useTm_ = false;
        return;
    }
    pondering_.store(limits.ponder, std::memory_order_relaxed);

    // Fixed time per move: soft == hard, only the overhead is shaved off.
    if (limits.movetime > 0) {
        const int64_t t = std::max<int64_t>(
            MIN_THINK_MS, limits.movetime - MOVE_OVERHEAD_MS);
        useTm_     = true;
        fixedTime_ = true;
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

    // Opening ramp, only without an increment: with one the clock refills
    // every move, so there is no reason to under-think the opening.
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
    if (!useTm_ || fixedTime_) return;

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
    startTicks_.store(Clock::now().time_since_epoch().count(),
                      std::memory_order_release);
    const std::lock_guard<std::mutex> lock(mtx_);
    cancel_ = false;
    armed_  = useTm_;
    if (!useTm_ || hardMs_ <= 0) return;
    if (pondering_.load(std::memory_order_relaxed)) {
        if (!ponderhitSeen_) return;
        // The ponderhit beat the search thread here: the clock started then.
        startTicks_.store(ponderhitTicks_, std::memory_order_release);
        pondering_.store(false, std::memory_order_release);
    }
    watchdog_ = std::thread([this] { watchdogLoop(); });
}

void TimeManager::clearPonderhit() noexcept {
    const std::lock_guard<std::mutex> lock(mtx_);
    ponderhitSeen_ = false;
}

void TimeManager::ponderhit() noexcept {
    const std::lock_guard<std::mutex> lock(mtx_);
    ponderhitSeen_  = true;
    ponderhitTicks_ = Clock::now().time_since_epoch().count();
    if (!armed_ || !pondering_.load(std::memory_order_relaxed)) return;
    // The opponent's thinking time was free; our own clock runs from here.
    startTicks_.store(ponderhitTicks_, std::memory_order_release);
    pondering_.store(false, std::memory_order_release);
    if (hardMs_ <= 0) return;
    watchdog_ = std::thread([this] { watchdogLoop(); });
}

void TimeManager::stop() noexcept {
    // The thread object is taken under the lock: ponderhit() may be assigning
    // it from the command thread.
    std::thread watchdog;
    {
        const std::lock_guard<std::mutex> lock(mtx_);
        armed_  = false;
        cancel_ = true;
        watchdog = std::move(watchdog_);
    }
    cv_.notify_all();
    if (watchdog.joinable()) watchdog.join();
}

void TimeManager::watchdogLoop() noexcept {
    std::unique_lock<std::mutex> lock(mtx_);
    const auto deadline =
        Clock::time_point(Clock::duration(startTicks_.load(std::memory_order_acquire)))
        + std::chrono::milliseconds(hardMs_);
    // Wake either at the hard deadline or when stop() cancels us.
    cv_.wait_until(lock, deadline, [this] { return cancel_; });
    if (!cancel_ && stopFlag_ != nullptr) {
        stopFlag_->store(true, std::memory_order_release);
    }
}

int64_t TimeManager::elapsedMs() const noexcept {
    const auto start = Clock::time_point(
        Clock::duration(startTicks_.load(std::memory_order_acquire)));
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now() - start).count();
}

bool TimeManager::shouldStartNextDepth() const noexcept {
    if (!useTm_ || pondering_.load(std::memory_order_acquire)) return true;
    const int64_t threshold = static_cast<int64_t>(
        static_cast<double>(softMs_) * START_NEXT_FRACTION);
    return elapsedMs() < std::max<int64_t>(MIN_THINK_MS, threshold);
}

} // namespace engine::time
