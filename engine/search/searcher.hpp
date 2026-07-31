#pragma once

#include <cstdint>
#include <limits>

#include "../../board/board.hpp"
#include "../../tt/tt.hpp"
#include "../eval_constants.hpp"
#include "../movelist.hpp"
#include "search_constants.hpp"
#include "searchruntime.hpp"
#include "../sort/sorter.hpp"

namespace engine {

class Searcher final {
public:
    Searcher() = delete;

    // Backward-compat aliases.
    static constexpr int32_t  POS_INF               = std::numeric_limits<int32_t>::max();
    // Negamax-safe: NEG_INF == -POS_INF so -(NEG_INF) == POS_INF is valid
    // (negating std::numeric_limits<int32_t>::min() is undefined behaviour).
    static constexpr int32_t  NEG_INF               = -POS_INF;
    static constexpr int      DEFAULT_DEPTH         = ::engine::DEFAULT_DEPTH;
    static constexpr int      MAX_PLY               = ::engine::MAX_PLY;
    static constexpr int      CAPTURE_HISTORY_SLOTS = ::engine::CAPTURE_HISTORY_SLOTS;
    using SearchRuntime = ::engine::SearchRuntime;

    // All tunable search parameters live in search_constants.hpp (namespace
    // engine), referenced unqualified from Searcher's methods.

    // --- Result structs ---
    struct SearchMoveResult {
        chess::Move move;
        int32_t     score;
    };

    struct IterativeSearchResult {
        bool    hasLegalMoves     {};
        bool    completedAnyDepth {};
        bool    terminalRoot      {};
        int     completedDepth    {};
        TT::Entry::Flag rootScoreBound = TT::Entry::EXACT;
        chess::Move bestMove      {};
        int32_t bestScore         {};
    };

    // --- Public interface ---
    static chess::Move searchBestMove(
        chess::Board& board,
        SearchRuntime& runtime,
        int requestedDepth = DEFAULT_DEPTH) noexcept;

    static IterativeSearchResult runIterativeDeepening(
        chess::Board& rootBoard,
        SearchRuntime& runtime,
        int startDepth,
        int targetDepth) noexcept;

    static chess::Move getBestMove(
        chess::Board& rootBoard,
        const MoveList& moves,
        SearchRuntime& runtime,
        int32_t alpha = NEG_INF,
        int32_t beta  = POS_INF) noexcept;

    static int32_t searchPosition(
        chess::Board& b,
        SearchRuntime& runtime,
        int depth,
        int32_t alpha,
        int32_t beta,
        int ply,
        bool allowHeuristicUpdates      = true,
        const chess::Move* previousMove = nullptr,
        uint64_t* nodeCounter           = nullptr,
        bool allowNullMove              = true,
        chess::Move excludedMove        = {}) noexcept;

    static int32_t quiescenceSearch(
        chess::Board& b,
        SearchRuntime& runtime,
        int32_t alpha,
        int32_t beta,
        int ply,
        uint64_t* nodeCounter = nullptr) noexcept;

private:

    struct SearchContext {
        const chess::Move* previousMove {};
        uint64_t* nodeCounter           {};
        int16_t* contHistEntry          {};
        int      depth;
        int      ply;
        int32_t  staticEval             {};
        int      singularExtension      {};
        chess::Move excludedMove        {};
        uint8_t activeColor;
        bool     inCheck                {};
        bool     isPVNode               {};
        bool     iirActive              {};
        bool     improving              {};
    };

    struct SearchNodeState {
        uint8_t activeColor = chess::Board::WHITE;
        bool    inCheck     {};
        bool    isPVNode    {};
        int32_t staticEval  {};
    };

    // --- Score utilities ---
    static constexpr bool isBetter(int32_t newScore, int32_t currentBest) noexcept       { return newScore > currentBest; }
    static constexpr bool isBetaCutoff(int32_t score, int32_t beta) noexcept             { return score >= beta; }
    static void           updateBound(int32_t score, int32_t& alpha) noexcept            { if (score > alpha) alpha = score; }
    static constexpr bool shouldDeltaPrune(int32_t standPat, int32_t margin, int32_t alpha) noexcept { return standPat + margin <= alpha; }
    static constexpr bool shouldResearchPVS(int32_t score, int32_t alphaBound) noexcept  { return score > alphaBound; }
    // True when a TT entry's bound proves a cutoff for the [alpha, beta] window
    // (scores are raw TT scores; mate rebasing happens after the cutoff check).
    static constexpr bool ttBoundCutoff(uint8_t flag, int32_t score, int32_t alpha, int32_t beta) noexcept {
        return flag == TT::Entry::EXACT
           || (flag == TT::Entry::LOWERBOUND && score >= beta)
           || (flag == TT::Entry::UPPERBOUND && score <= alpha);
    }
    static constexpr int32_t saturatingAdd32(int32_t lhs, int32_t rhs) noexcept;
    static constexpr int32_t saturatingSub32(int32_t lhs, int32_t rhs) noexcept;
    // Mate scores embed ply-distance; rebase on TT store/load so mate distances
    // stay correct across transpositions.
    static constexpr int32_t scoreToTT(int32_t score, int ply) noexcept;
    static constexpr int32_t scoreFromTT(int32_t score, int ply) noexcept;
    // delta > 0 is a bonus, delta < 0 a malus; decay uses |delta|.
    static void applyHistoryGravity(int16_t& cell, int32_t delta, int32_t maxValue) noexcept;

    // --- Terminal condition checks ---
    static bool checkEarlyTerminalConditions(
        const chess::Board& b, SearchRuntime& runtime, int ply, int32_t& outScore) noexcept;
    // Node-counter bump + node-budget cap + early-terminal checks; the shared
    // node entry for searchPosition/quiescenceSearch. Returns true (outScore
    // set) when the node must return immediately.
    static bool enterNode(
        chess::Board& b, SearchRuntime& runtime, int ply,
        uint64_t* counter, int32_t& outScore) noexcept;
    // atRoot=true restricts terminals to *forced* draws (3rd repetition,
    // 50-move, insufficient material). The "2nd repetition scores as 0"
    // pruning rule is an interior-node heuristic only: at the root it would
    // abandon the search and play a random legal move, so it is suppressed.
    static bool checkDrawTerminalConditions(
        const chess::Board& b, int32_t& outScore, bool atRoot = false) noexcept;

    // --- Root search helpers ---
    static void updateMinMax(int32_t score, int32_t& alpha, int32_t& bestScore,
                             chess::Move& bestMove, const chess::Move& m) noexcept;
    static int32_t searchRootMoveScore(
        chess::Board& b,
        const chess::Move& m,
        SearchRuntime& runtime,
        int32_t alpha, int32_t beta,
        bool allowHeuristicUpdates,
        uint64_t* nodeCounter) noexcept;

    // --- Internal search primitives ---
    static bool tryNullMovePruning(
        chess::Board& b, const SearchNodeState& node, SearchRuntime& runtime,
        int depth, int32_t alpha, int32_t beta, int ply,
        bool allowHeuristicUpdates,
        uint64_t* nodeCounter, int32_t& outScore) noexcept;
    static bool tryReverseFutilityPruning(
        const SearchNodeState& node,
        int depth, int32_t beta, int32_t& outScore) noexcept;
    static SearchMoveResult searchMoves(
        chess::Board& b, MovePicker& movePicker,
        const SearchContext& ctx, int32_t alpha, int32_t beta,
        SearchRuntime& runtime,
        bool allowHeuristicUpdates) noexcept;
    static void updateKillerAndHistoryOnBetaCutoff(
        const chess::Move& m, bool isCapture, int victimType,
        int depth, int ply, int usSide, SearchRuntime& runtime,
        const chess::Move* previousMove,
        int16_t* contHistEntry = nullptr,
        int fromPieceType = 0) noexcept;
};

} // namespace engine
