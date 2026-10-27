#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

#include "../../board/board.hpp"
#include "../../tt/tt.hpp"
#include "../syzygy/syzygy.hpp"
#include "corrhist.hpp"
#include "search_constants.hpp"

namespace engine {

namespace time { class TimeManager; }

struct SearchRuntime {
    // --- Search state ---
    // Bumped once per node by the owning thread only; the UCI info line sums
    // it across threads, so concurrent access goes through relaxed atomic_ref.
    uint64_t nodesSearched = 0;
    int      depth         = DEFAULT_DEPTH;
    int32_t  eval          = 0;
    int      maxThreads    = 1;
    bool     emitUciInfo   = false;
    // Lazy SMP helper thread: skips the Syzygy root probe (tb_probe_root is
    // main-thread-only in Fathom) and never writes to stdout.
    bool     isHelper      = false;
    // Lazy SMP depth diversification: a helper skips depth d when
    // ((d + phase) / size) is odd, so the threads spread over depths instead of
    // moving in lockstep. Size 0 (the main thread) never skips.
    int      depthSkipSize  = 0;
    int      depthSkipPhase = 0;
    // UCI `go nodes N` (0 = unlimited), checked at every node; each Lazy SMP
    // helper bounds itself by it too.
    uint64_t maxNodes      = 0;

    // --- Heuristics ---
    // [ply][slot]: both killers of a ply share a cache line.
    chess::Move killerMoves[MAX_PLY][2] {};
    int16_t  history[2][64][64] {};
    uint16_t counterMoves[64][64] {};
    int16_t  captureHistory[2][64][7][CAPTURE_HISTORY_SLOTS] {};
    // Continuation history (layout at contHistIndex): 576 KiB per thread.
    int16_t  contHist[2][CONT_HIST_PIECE_TYPES][64][CONT_HIST_PIECE_TYPES][64] {};
    // Static-eval correction; never decayed between searches, only cleared.
    CorrectionHistory corrHist {};
    // evalStack is thread_local in searcher.cpp.

    // --- External coordination ---
    TT*       transpositionTable     = nullptr;
    std::atomic<bool>*        stopSearchRequested    = nullptr;
    std::atomic<bool>*        searchInterrupted      = nullptr;
    // Null for the helpers, datagen and the tests.
    time::TimeManager*        timeManager            = nullptr;
    syzygy::SyzygyProber*     syzygyProber           = nullptr;

    [[nodiscard]] inline bool shouldAbort() const noexcept {
        return stopSearchRequested != nullptr
            && stopSearchRequested->load(std::memory_order_acquire);
    }

    inline void markInterrupted() noexcept {
        if (searchInterrupted != nullptr)
            searchInterrupted->store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] inline bool isInterrupted() const noexcept {
        return searchInterrupted != nullptr
            && searchInterrupted->load(std::memory_order_relaxed);
    }

    inline void clearInterrupted() noexcept {
        if (searchInterrupted != nullptr)
            searchInterrupted->store(false, std::memory_order_relaxed);
    }

    // Full wipe. softResetHistory only halves, so a benchmark game would still
    // inherit the previous one; see CHESS_DATAGEN_SEED in datagen.cpp.
    inline void clearHistory() noexcept {
        for (auto& ply : killerMoves) ply[0] = ply[1] = chess::Move{};
        std::memset(history, 0, sizeof(history));
        std::memset(counterMoves, 0, sizeof(counterMoves));
        std::memset(captureHistory, 0, sizeof(captureHistory));
        std::memset(contHist, 0, sizeof(contHist));
        corrHist.clear();
    }

    // Halves every history cell before a search: old knowledge fades.
    inline void softResetHistory() noexcept {
        const auto halve = [](auto& table) {
            int16_t* cells = reinterpret_cast<int16_t*>(&table);
            #pragma omp simd
            for (size_t i = 0; i < sizeof(table) / sizeof(int16_t); ++i) cells[i] >>= 1;
        };
        halve(history);
        halve(contHist);
        halve(captureHistory);
    }
};

} // namespace engine
