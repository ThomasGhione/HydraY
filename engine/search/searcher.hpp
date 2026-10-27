#pragma once

#include <cstdint>

#include "../../board/board.hpp"
#include "../../tt/tt.hpp"
#include "../movelist.hpp"
#include "search_constants.hpp"
#include "searchruntime.hpp"
#include "../sort/sorter.hpp"

namespace engine {

// Drops every cached NNUE static eval. Must be called whenever the active
// network changes: the cached values belong to the previous net.
void clearEvalCache() noexcept;

namespace Searcher {

struct IterativeSearchResult {
    bool    completedAnyDepth {};
    bool    terminalRoot      {};
    // The root move came from the Syzygy DTZ ranking, not from a search:
    // completedDepth stays 0, so the Lazy SMP vote must not compare it.
    bool    tablebaseRoot     {};
    int     completedDepth    {};
    chess::Move bestMove      {};
    int32_t bestScore         {};
};

chess::Move searchBestMove(chess::Board& board, SearchRuntime& runtime,
                           int requestedDepth = DEFAULT_DEPTH) noexcept;

IterativeSearchResult runIterativeDeepening(chess::Board& rootBoard, SearchRuntime& runtime,
                                            int startDepth, int targetDepth) noexcept;

// alreadyEntered: the caller (a depth-0 searchPosition) has counted the node
// and checked its draw rules.
int32_t quiescenceSearch(chess::Board& b, SearchRuntime& runtime, int32_t alpha, int32_t beta,
                         int ply, bool alreadyEntered = false) noexcept;

} // namespace Searcher

} // namespace engine
