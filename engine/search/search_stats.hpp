#pragma once

#include <cstdint>

// Search counters for a measurement build: compile with -DSEARCH_STATS.
// Without it kSearchStats is false and every counting site compiles away.
#ifdef SEARCH_STATS
inline constexpr bool kSearchStats = true;
#else
inline constexpr bool kSearchStats = false;
#endif

struct SearchStats {
    uint64_t mainNodes, qNodes;
    uint64_t ttProbes, ttHits, ttCuts;
    uint64_t cutNodes, cutFirst, cutIndexSum, noCutNodes;
    uint64_t lmrSearches, lmrResearches, lmrReductionSum;
    uint64_t prunedLmp, prunedFutility, prunedHistory, prunedSee;
    uint64_t nmpTries, nmpCuts, rfpCuts, probcutCuts;
};

// One set per search thread; the report covers the main thread only.
inline thread_local SearchStats gSearchStats{};
