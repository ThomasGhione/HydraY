#pragma once

#include <atomic>

#include "../board/board.hpp"
#include "../tt/tt.hpp"

#include "search/searcher.hpp"
#include "time/time_manager.hpp"
#include "syzygy/syzygy.hpp"

namespace engine {

class Engine final {
public:
    Engine();

    void reset() noexcept;
    // Clears the stop flags and any stale ponderhit. The UCI reader thread calls
    // it before starting the search thread, so a `stop` right after `go` is kept.
    void prepareSearch() noexcept;
    chess::Move search(const time::Limits& limits) noexcept;
    void stop() noexcept { stopSearchRequested.store(true, std::memory_order_release); }

    chess::Board board;
    SearchRuntime searchRuntime{};
    // UCI `Threads` (0 = omp_get_max_threads()); survives reset().
    int requestedThreads = 0;
    syzygy::SyzygyProber syzygyProber;
    TT tt;
    time::TimeManager timeManager;

private:
    std::atomic<bool> stopSearchRequested {false};
    std::atomic<bool> searchInterrupted {false};

    void bindSearchRuntime() noexcept;
};

} // namespace engine
