#include "searcher.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "../../debug.hpp"
#include "../../nnue/nnue.hpp"
#include "../../uci/output.hpp"
#include "../engine.hpp"
#include "../sort/move_generator.hpp"
#include "evalcache.hpp"
#include "search_stats.hpp"

namespace engine {

namespace {

EvalCache evalCache;

// LMR_REDUCTIONS[depth][moveIndex], at least 1: two logs per LMR candidate
// otherwise.
int8_t LMR_REDUCTIONS[LMR_MAX_DEPTH][MAX_MOVES];

} // namespace

void clearEvalCache() noexcept { evalCache.clear(); }

void rebuildSearchDerivedTables() noexcept {
    const double lmrC = static_cast<double>(LMR_C_PERCENT) / 100.0;
    for (int d = 0; d < LMR_MAX_DEPTH; ++d) {
        for (size_t m = 0; m < MAX_MOVES; ++m) {
            const double raw = (d == 0 || m == 0) ? 1.0 : __builtin_log(d) * __builtin_log(static_cast<double>(m)) / lmrC;
            LMR_REDUCTIONS[d][m] = static_cast<int8_t>(std::max(1, static_cast<int>(raw)));
        }
    }
    for (int d = 1; d <= FUTILITY_DEPTH_CAP; ++d) {
        FUTILITY_MARGINS[d] = FUTILITY_MID_STEP * d;
    }
    for (int improving = 0; improving < 2; ++improving) {
        for (int d = 1; d <= LMP_DEPTH_CAP; ++d) {
            LMP_THRESHOLDS[improving][d] = lmpBaseThreshold(improving, d) * LMP_SCALE_PCT[improving] / 100;
        }
    }
}

// Built once at startup; the UCI layer rebuilds after writing a generator spin.
[[maybe_unused]] static const bool searchTablesBuilt = (rebuildSearchDerivedTables(), true);

namespace Searcher {

using chess::Board;
using chess::Move;

namespace {

// Per thread (Lazy SMP): each ancestor writes evalStack[ply] before its
// grandchild two plies down reads it. Slot 0 is the root, which never runs
// through searchPosition, so getBestMove seeds it.
thread_local int32_t evalStack[MAX_PLY] = {};

// A missing king (reachable only from an illegal FEN) is decisive, not an eval.
int32_t evaluate(const Board& b) noexcept {
    if (b.bb[Board::KING][0] == 0 || b.bb[Board::KING][1] == 0) [[unlikely]]
        return ((b.bb[Board::KING][0] == 0) == (b.getActiveColor() == Board::WHITE)) ? -MATE_VALUE : MATE_VALUE;
    return NNUE::evaluate(b);
}

// Raw static eval for a position not carried by its TT entry.
int32_t cachedEval(uint64_t hashKey, Board& b) noexcept {
    int32_t cached = 0;
    if (evalCache.probe(hashKey, cached)) return cached;
    const int32_t value = evaluate(b);
    evalCache.store(hashKey, value);
    return value;
}

// "Level enough that a repetition is acceptable": half a pawn, converted to
// eval units because it is compared with a static eval.
constexpr int32_t REPETITION_DRAW_ADVANTAGE_THRESHOLD = materialToEval(PIECE_VALUES[Board::PAWN] / 2);

// Once LMP prunes one quiet at a node it prunes every later quiet too, so the
// rest of the picker matters only for its tactical moves. If those are all
// finalised losing captures, each ranks below every quiet (a quiet never scores
// under -3000 - the queen value), so the selection would extract all the quiets
// first and then these in score order: jump straight to them. Equal scores keep
// the normal path, because their order depends on the swaps of the quiet picks.
// Returns false, leaving the picker untouched, when that cannot be guaranteed.
bool skipToLosingCaptures(MovePicker& picker, const Board& b,
                          chess::Square enPassant, int promotionRank) noexcept {
    Move tailMoves[MAX_MOVES];
    int32_t tailScores[MAX_MOVES];
    int n = 0;
    for (int i = picker.currentIndex; i < picker.moves.size; ++i) {
        const Move& mv = picker.moves[i];
        const MoveTraits t = moveTraits(b, mv, enPassant, promotionRank);
        if (!t.isPromotion && !t.isCapture) continue;
        const int32_t score = picker.score(i);
        if (picker.seePending[i] != SeePending::Final || score >= -CAPTURE_BASE_SCORE) return false;
        // Insertion by descending score; n stays tiny.
        int j = n++;
        for (; j > 0 && tailScores[j - 1] < score; --j) {
            tailMoves[j] = tailMoves[j - 1];
            tailScores[j] = tailScores[j - 1];
        }
        if (j > 0 && tailScores[j - 1] == score) return false;
        tailMoves[j] = mv;
        tailScores[j] = score;
    }
    const int start = picker.moves.size - n;
    for (int k = 0; k < n; ++k) {
        picker.moves[start + k] = tailMoves[k];
        picker.setScore(start + k, tailScores[k]);
        picker.seePending[start + k] = SeePending::Final;
    }
    picker.currentIndex = start;
    return true;
}

// Mate and TB scores carry their ply distance; rebase them on TT store/load.
// |score| <= MATE_VALUE and ply <= MAX_PLY keep the result inside int16_t.
constexpr int32_t scoreToTT(int32_t score, int ply) noexcept {
    if (score >= TB_WIN_BOUND)  return score + ply;
    if (score <= -TB_WIN_BOUND) return score - ply;
    return score;
}

constexpr int32_t scoreFromTT(int32_t score, int ply) noexcept {
    return scoreToTT(score, -ply);
}

// Whether a bound of this flag proves a cutoff for the [alpha, beta] window.
constexpr bool ttBoundCutoff(uint8_t flag, int32_t score, int32_t alpha, int32_t beta) noexcept {
    return flag == TT::Entry::EXACT
        || (flag == TT::Entry::LOWERBOUND && score >= beta)
        || (flag == TT::Entry::UPPERBOUND && score <= alpha);
}

// delta > 0 is a bonus, delta < 0 a malus; the decay scales with |delta|.
void applyHistoryGravity(int16_t& cell, int32_t delta, int32_t maxValue) noexcept {
    const int32_t value = cell + delta - cell * std::abs(delta) / maxValue;
    cell = static_cast<int16_t>(std::clamp<int32_t>(value, INT16_MIN, INT16_MAX));
}

// Per-node entry shared by searchPosition and quiescenceSearch: count the node,
// enforce the `go nodes` cap, and stop at an abort, the ply limit or a missing
// king. Returns true with outScore set when the node must return at once.
// Forced inline: the atomic counter store otherwise costs a call per node.
__attribute__((always_inline))
inline bool enterNode(Board& b, SearchRuntime& runtime, int ply, int32_t& outScore) noexcept {
    // Only the owner thread writes its counter (the info line reads the
    // helpers' ones), so a plain read and a relaxed store, no locked RMW.
    const uint64_t searched = runtime.nodesSearched + 1;
    std::atomic_ref<uint64_t>(runtime.nodesSearched).store(searched, std::memory_order_relaxed);
    if ((runtime.maxNodes > 0 && searched >= runtime.maxNodes) || runtime.shouldAbort()) [[unlikely]] {
        runtime.markInterrupted();
        outScore = evaluate(b);
        return true;
    }
    if (ply >= MAX_PLY - 1) {
        outScore = evaluate(b);
        return true;
    }
    // A king can only vanish after an illegal FEN: losing ours is a mate,
    // taking theirs a win.
    if (b.bb[Board::KING][0] == 0 || b.bb[Board::KING][1] == 0) {
        const bool ourKingGone = (b.bb[Board::KING][0] == 0) == (b.getActiveColor() == Board::WHITE);
        outScore = ourKingGone ? (-MATE_VALUE + ply) : (MATE_VALUE - ply);
        return true;
    }
    return false;
}

// Lazy SMP helper state, one slot per helper thread, kept across searches so
// its histories keep learning. searchBestMove is single-entrant (UCI joins one
// search before starting the next), so the statics are safe.
struct HelperSlot {
    SearchRuntime runtime; // defaults are helper-correct: one thread, no info lines, no clock
    Board board;
    std::atomic<bool> interrupted{false};
    // This helper's root result for the vote; cleared at every search start.
    IterativeSearchResult result{};

    HelperSlot() noexcept {
        runtime.searchInterrupted = &interrupted;
        runtime.isHelper = true;
    }
};

std::vector<std::unique_ptr<HelperSlot>> helperSlots;
// Helpers running in the current search; slots past it may hold stale counts.
int activeHelperCount = 0;

// This runtime's nodes plus every active helper's, read with relaxed loads.
uint64_t totalNodesSearched(const SearchRuntime& runtime) noexcept {
    const auto load = [](const uint64_t& word) {
        return std::atomic_ref<uint64_t>(const_cast<uint64_t&>(word)).load(std::memory_order_relaxed);
    };
    uint64_t total = load(runtime.nodesSearched);
    for (int i = 0; i < activeHelperCount; ++i) {
        total += load(helperSlots[i]->runtime.nodesSearched);
    }
    return total;
}

// The principal variation read back from the TT. Each move is checked against
// the legal moves (a key collision could store an illegal one) and the walk
// stops at a repetition, so a cycle cannot loop. The board is left as found.
std::string buildPvFromTT(Board& board, const TT& tt, int maxLen) noexcept {
    std::array<Move, MAX_PLY> pvMoves{};
    std::array<Board::MoveState, MAX_PLY> states{};
    int applied = 0;
    std::string pv;
    while (applied < maxLen && applied < MAX_PLY) {
        const Move mv = tt.probeDecodedMove(board.getHash());
        if (!chess::isValidSquare(mv.from)) break;
        const MoveList legal = MoveGenerator::generateLegalMoves(board);
        if (std::find(legal.begin(), legal.end(), mv) == legal.end()) break;
        if (!pv.empty()) pv += ' ';
        pv += mv.toUCIString();
        pvMoves[applied] = mv;
        board.doMove(mv, states[applied++]);
        if (board.countRepetitions() >= 3) break;
    }
    while (applied > 0) {
        --applied;
        board.undoMove(pvMoves[applied], states[applied]);
    }
    return pv;
}

// One `info` line for a completed iteration. With firstMove the PV starts with
// that root move and follows the TT from the position after it.
void emitUciInfoLine(int depth, int32_t score, const SearchRuntime& runtime,
                     Board& rootBoard, const Move* firstMove = nullptr) noexcept {
    const int64_t timeMs = (runtime.timeManager != nullptr) ? runtime.timeManager->elapsedMs() : 0;
    const uint64_t nodes = totalNodesSearched(runtime);
    const int64_t nps = (timeMs > 0)
        ? static_cast<int64_t>(static_cast<long double>(nodes) * 1000.0L / static_cast<long double>(timeMs))
        : 0;

    std::string line = "info depth " + std::to_string(depth) + " score ";
    if (std::abs(score) >= MATE_BOUND) {
        const int32_t mateMoves = (MATE_VALUE - std::abs(score) + 1) / 2;
        line += "mate " + std::to_string(score > 0 ? mateMoves : -mateMoves);
    } else {
        line += "cp " + std::to_string(score);
    }
    line += " nodes " + std::to_string(nodes) + " nps " + std::to_string(nps)
          + " time " + std::to_string(timeMs);

    std::string pv;
    if (firstMove != nullptr) {
        Board::MoveState state;
        rootBoard.doMove(*firstMove, state);
        const std::string rest = buildPvFromTT(rootBoard, *runtime.transpositionTable, 15);
        rootBoard.undoMove(*firstMove, state);
        pv = firstMove->toUCIString() + (rest.empty() ? "" : " " + rest);
    } else {
        pv = buildPvFromTT(rootBoard, *runtime.transpositionTable, 16);
    }
    if (!pv.empty()) line += " pv " + pv;
    uci::emit(line + '\n');
}

} // namespace

struct SearchContext {
    const Move* previousMove {};
    int16_t* contHistEntry   {};
    int      depth;
    int      ply;
    int32_t  staticEval      {};
    int      singularExtension {};
    Move     excludedMove    {};
    uint8_t  activeColor;
    bool     inCheck         {};
    bool     isPVNode        {};
    bool     iirActive       {};
    bool     improving       {};
    // Expected cut node: a null-window child whose parent hopes it fails
    // high. Always false at PV nodes.
    bool     cutNode         {};
};

struct SearchMoveResult {
    Move    move;
    int32_t score;
    // A move was skipped by LMP, futility, history or SEE pruning, so a
    // fail-low score bounds only the searched moves.
    bool    pruned {};
};

int32_t searchPosition(Board& b, SearchRuntime& runtime, int depth, int32_t alpha, int32_t beta,
                       int ply, bool allowHeuristicUpdates = true, const Move* previousMove = nullptr,
                       bool allowNullMove = true, Move excludedMove = {}, bool cutNode = false) noexcept;

// atRoot restricts terminals to forced draws (third repetition, fifty moves,
// insufficient material): scoring the second repetition as 0 is an interior
// heuristic, and at the root it would end the search with an arbitrary move.
// ply only places a mate delivered on the 100th half-move.
bool checkDrawTerminalConditions(const Board& b, int ply, int32_t& outScore, bool atRoot = false) noexcept {
    // A repetition needs at least 4 reversible plies; the guard skips the scan
    // in most of the tree.
    if (b.getHalfMoveClock() >= 4) {
        const int repCount = b.countRepetitions();
        if (repCount >= 3) {
            // A small fixed contempt: the side ahead prefers to play on, but
            // not enough to give up material to avoid the draw.
            const int32_t staticEval = evaluate(b);
            outScore = (std::abs(staticEval) <= REPETITION_DRAW_ADVANTAGE_THRESHOLD)
                ? 0
                : (staticEval > 0 ? -REPETITION_CONTEMPT : REPETITION_CONTEMPT);
            return true;
        }
        if (!atRoot && repCount >= 2) {
            outScore = 0;
            return true;
        }
    }

    if (b.isFiftyMoveRule()) [[unlikely]] {
        // A mate delivered by the 100th half-move stands (FIDE 9.3).
        const uint64_t checkers = b.checkersTo(b.getActiveColor());
        const bool mated = checkers && MoveGenerator::generateLegalMoves(b, checkers).is_empty();
        outScore = mated ? (-MATE_VALUE + ply) : 0;
        return true;
    }

    if (b.hasInsufficientMaterialDraw()) [[unlikely]] {
        outScore = 0;
        return true;
    }
    return false;
}

SearchMoveResult searchMoves(Board& b, MovePicker& movePicker, const SearchContext& ctx,
                             int32_t alpha, int32_t beta, SearchRuntime& runtime,
                             bool allowHeuristicUpdates) noexcept {
    int32_t best = NEG_INF;
    Move bestMove{};
    bool searchedAnyMove = false;

    struct QuietEntry { uint8_t from; uint8_t to; uint8_t pieceType; };
    constexpr int MAX_QUIETS_TRACKED = 64;
    QuietEntry searchedQuiets[MAX_QUIETS_TRACKED];
    int numSearchedQuiets = 0;

    struct CaptureEntry { uint8_t to; uint8_t victimType; };
    constexpr int MAX_CAPTURES_TRACKED = 32;
    CaptureEntry searchedCaptures[MAX_CAPTURES_TRACKED];
    int numSearchedCaptures = 0;

    // Interior non-PV node past the root: the gate for futility, LMP, history and SEE pruning.
    const bool interiorNonPv = !ctx.isPVNode && !ctx.inCheck && ctx.ply > 0;
    const bool canFutilityPrune = interiorNonPv && ctx.depth >= 1 && ctx.depth <= FUTILITY_MAX_DEPTH;
    const int32_t futilityMargin = canFutilityPrune ? FUTILITY_MARGINS[ctx.depth] : 0;
    const bool canLMP = interiorNonPv && ctx.depth >= 1 && ctx.depth <= LMP_MAX_DEPTH;
    const int lmpThreshold = canLMP ? LMP_THRESHOLDS[ctx.improving][ctx.depth] : 999;

    const int usSide = Board::colorToIndex(ctx.activeColor);
    const int oppKingSq = std::countr_zero(b.bb[Board::KING][usSide ^ 1]);
    const chess::Square enPassant = b.getEnPassant();
    const int promotionRank = Board::promotionRank(ctx.activeColor == Board::WHITE);
    const uint8_t oppColor = Board::oppositeColor(ctx.activeColor);
    bool prunedAny = false;
    bool lmpSkipTried = false;

    while (movePicker.hasNext()) {
        const int moveIndex = movePicker.currentIndex;
        const Move m = movePicker.nextMove();
        if (runtime.shouldAbort()) {
            runtime.markInterrupted();
            break;
        }
        if (chess::isValidSquare(ctx.excludedMove.from) && m.sameFromTo(ctx.excludedMove)) continue;

        const bool isFirstMove = (moveIndex == 0);
        const MoveTraits t = moveTraits(b, m, enPassant, promotionRank);
        const bool isQuietMove = !t.isCapture && !t.isPromotion;

        if (canLMP && isQuietMove && moveIndex >= lmpThreshold) {
            if constexpr (kSearchStats) ++gSearchStats.prunedLmp;
            prunedAny = true;
            if (!lmpSkipTried) {
                lmpSkipTried = true;
                skipToLosingCaptures(movePicker, b, enPassant, promotionRank);
            }
            continue;
        }
        // givesCheck is the last term: it costs two magic lookups, and every
        // cheaper term rejects far more often.
        if (canFutilityPrune && isQuietMove && moveIndex > 0 && ctx.staticEval + futilityMargin <= alpha
            && (t.fromType == Board::KING
                || !Sorter::givesCheckAfterQuietMoveFast(b, m, t.fromType, oppKingSq, b.getPiecesBitMap()))) {
            if constexpr (kSearchStats) ++gSearchStats.prunedFutility;
            prunedAny = true;
            continue;
        }
        // Late quiets with a very negative history rarely raise alpha at low depth.
        if (isQuietMove && interiorNonPv && ctx.depth >= 1 && ctx.depth <= 3 && moveIndex > 0
            && runtime.history[usSide][m.from][m.to] < HISTORY_PRUNE_THRESHOLD[ctx.depth]) {
            if constexpr (kSearchStats) ++gSearchStats.prunedHistory;
            prunedAny = true;
            continue;
        }
        // Clearly losing captures at shallow non-PV nodes; move 0 is always searched.
        if (t.isCapture && !t.isPromotion && moveIndex > 0 && interiorNonPv && ctx.depth <= 3
            && Sorter::staticExchangeEvaluation(b, m) < -SEE_CAPTURE_MARGIN * ctx.depth) {
            if constexpr (kSearchStats) ++gSearchStats.prunedSee;
            prunedAny = true;
            continue;
        }

        Board::MoveState state;
        b.doMove(m, state);
        // The child's TT bucket loads while the child checks its node entry.
        runtime.transpositionTable->prefetch(b.getHash());

        const bool forcing = t.isCapture || t.isPromotion || moveIndex < 3;
        const bool givesCheck = ctx.depth >= 2 && ctx.depth <= 4 && forcing && b.inCheck(oppColor);
        const int childDepth = ctx.depth - 1 + givesCheck + (isFirstMove ? ctx.singularExtension : 0);
        // PVS: the full window for the first move, a null window for the rest.
        const int32_t scoutBeta = isFirstMove ? beta : alpha + 1;
        const auto search = [&](int d, int32_t lo, int32_t hi, bool cut) {
            return -searchPosition(b, runtime, d, -hi, -lo, ctx.ply + 1, allowHeuristicUpdates, &m,
                                   true, Move{}, cut);
        };

        int32_t score;
        // Late captures are reduced too: ordering puts the good ones early.
        if (ctx.depth >= 3 && moveIndex >= 4 && !t.isPromotion) {
            int reduction = LMR_REDUCTIONS[std::min(ctx.depth, LMR_MAX_DEPTH - 1)][moveIndex];
            if (ctx.isPVNode) reduction = std::max(1, reduction - 1);
            if (ctx.iirActive) reduction = std::min(reduction + 1, childDepth - 1);
            // More when the eval is falling, and at an expected cut node.
            reduction += !ctx.improving + (ctx.cutNode ? LMR_CUT_BONUS : 0);
            reduction = std::clamp(reduction, 1, childDepth - 1);
            const int reducedDepth = std::max(1, childDepth - reduction);
            if constexpr (kSearchStats) {
                ++gSearchStats.lmrSearches;
                gSearchStats.lmrReductionSum += static_cast<uint64_t>(childDepth - reducedDepth);
            }
            score = search(reducedDepth, alpha, scoutBeta, true);
            if (score > alpha) {
                if constexpr (kSearchStats) ++gSearchStats.lmrResearches;
                score = search(childDepth, alpha, scoutBeta, !ctx.cutNode);
            }
        } else {
            // The first move of a PV node is a PV child; every other first search is a null window.
            score = search(childDepth, alpha, scoutBeta, !(ctx.isPVNode && isFirstMove) && !ctx.cutNode);
        }
        // Only a PV node's full window is wider than the scout's.
        if (ctx.isPVNode && !isFirstMove && score > alpha) score = search(childDepth, alpha, beta, false);

        b.undoMove(m, state);
        searchedAnyMove = true;
        if (runtime.isInterrupted()) break;

        if (isQuietMove && numSearchedQuiets < MAX_QUIETS_TRACKED) {
            searchedQuiets[numSearchedQuiets++] = {m.from, m.to, static_cast<uint8_t>(t.fromType)};
        }
        if (t.isCapture && numSearchedCaptures < MAX_CAPTURES_TRACKED) {
            searchedCaptures[numSearchedCaptures++] = {m.to, static_cast<uint8_t>(t.victimType)};
        }

        if (score > best) {
            best = score;
            bestMove = m;
        }
        alpha = std::max(alpha, score);
        if (best < beta) continue;
        if constexpr (kSearchStats) {
            ++gSearchStats.cutNodes;
            if (moveIndex == 0) ++gSearchStats.cutFirst;
            gSearchStats.cutIndexSum += static_cast<uint64_t>(moveIndex);
        }

        if (allowHeuristicUpdates) {
            const int32_t bonus = (ctx.depth + 1) * (ctx.depth + 1);
            if (t.isCapture) {
                int16_t* cell = runtime.captureHistory[usSide][m.to][t.victimType];
                applyHistoryGravity(cell[0], bonus, MAX_CAPTURE_HISTORY);
                applyHistoryGravity(cell[1], bonus >> 1, MAX_CAPTURE_HISTORY);
                if (cell[1] > cell[0]) std::swap(cell[0], cell[1]);
            } else {
                if (ctx.previousMove != nullptr) {
                    runtime.counterMoves[ctx.previousMove->from][ctx.previousMove->to] = TT::Entry::encodeMove(m);
                }
                Move* killers = runtime.killerMoves[ctx.ply];
                if (!m.sameFromTo(killers[0])) {
                    killers[1] = killers[0];
                    killers[0] = m;
                }
                applyHistoryGravity(runtime.history[usSide][m.from][m.to], bonus, MAX_HISTORY);
                if (ctx.contHistEntry != nullptr) {
                    applyHistoryGravity(ctx.contHistEntry[contHistIndex(t.fromType, m.to)], bonus, MAX_HISTORY);
                }
            }
            // Malus to the moves searched before the cutoff; the cutoff move
            // itself is the last tracked entry of its kind and is spared.
            for (int i = 0; i < numSearchedQuiets - isQuietMove; ++i) {
                applyHistoryGravity(runtime.history[usSide][searchedQuiets[i].from][searchedQuiets[i].to],
                                    -bonus, MAX_HISTORY);
                if (ctx.contHistEntry != nullptr) {
                    applyHistoryGravity(ctx.contHistEntry[contHistIndex(searchedQuiets[i].pieceType, searchedQuiets[i].to)],
                                        -bonus, MAX_HISTORY);
                }
            }
            for (int i = 0; i < numSearchedCaptures - t.isCapture; ++i) {
                int16_t* cell = runtime.captureHistory[usSide][searchedCaptures[i].to][searchedCaptures[i].victimType];
                applyHistoryGravity(cell[0], -bonus, MAX_CAPTURE_HISTORY);
                applyHistoryGravity(cell[1], -bonus, MAX_CAPTURE_HISTORY);
            }
        }
        break;
    }

    if constexpr (kSearchStats) {
        if (searchedAnyMove && best < beta) ++gSearchStats.noCutNodes;
    }
    // An abort before any move was searched leaves only the static eval.
    return {bestMove, (!searchedAnyMove && runtime.isInterrupted()) ? ctx.staticEval : best, prunedAny};
}

int32_t searchPosition(Board& b, SearchRuntime& runtime, int depth, int32_t alpha, int32_t beta,
                       int ply, bool allowHeuristicUpdates, const Move* previousMove,
                       bool allowNullMove, Move excludedMove, bool cutNode) noexcept {
    const bool hasExcludedMove = chess::isValidSquare(excludedMove.from);
    int32_t early = 0;
    if (enterNode(b, runtime, ply, early)) return early;

    const bool isPVNode = beta - alpha > 1;
    // Mate-distance pruning; it never demotes a PV node to a null window.
    if (ply > 0) {
        alpha = std::max(alpha, -MATE_VALUE + ply);
        beta  = std::min(beta,   MATE_VALUE - ply - 1);
        if (alpha >= beta) return alpha;
    }
    // Draws before qsearch, or a horizon node completing a repetition would be
    // scored by its eval.
    if (checkDrawTerminalConditions(b, ply, early)) return early;
    if (depth <= 0) return quiescenceSearch(b, runtime, alpha, beta, ply, /*alreadyEntered=*/true);
    if constexpr (kSearchStats) ++gSearchStats.mainNodes;

    TT& tt = *runtime.transpositionTable;
    const uint64_t hashKey = b.getHash();
    // One TT read per node feeds the cutoff, the static-eval tightening, the
    // singular-extension gate and the hash move. The stored score is rebased to
    // this ply before the window test.
    const TT::ProbeResult tte = tt.probeEntry(hashKey);
    if constexpr (kSearchStats) {
        ++gSearchStats.ttProbes;
        if (tte.hit) ++gSearchStats.ttHits;
    }
    if (!hasExcludedMove && tte.hit && tte.depth >= std::min(depth, static_cast<int>(TT::Entry::MAX_DEPTH))) {
        const int32_t ttScore = scoreFromTT(tte.score, ply);
        if (ttBoundCutoff(tte.flag, ttScore, alpha, beta)) {
            if constexpr (kSearchStats) ++gSearchStats.ttCuts;
            return ttScore;
        }
    }

    // TB WDL probe, after the TT cutoff: most TB-range nodes cut on the cached
    // entry. A draw is exact; a win or loss counts only right after a zeroing
    // move (the fifty-move counter may draw it later) and is a ply-anchored
    // bound like a mate. No probe with castling rights or under an excluded
    // move: the tables cannot answer for a reduced move set.
    if (!hasExcludedMove && runtime.syzygyProber != nullptr
        && depth >= runtime.syzygyProber->probeDepth && b.castlingRights() == 0) {
        if (const auto wdl = runtime.syzygyProber->probeWDL(b)) {
            const int32_t tbScore = syzygy::SyzygyProber::wdlToScore(*wdl, ply);
            const uint8_t flag = (tbScore == 0) ? TT::Entry::EXACT
                               : (tbScore > 0)  ? TT::Entry::LOWERBOUND
                                                : TT::Entry::UPPERBOUND;
            if ((tbScore == 0 || b.getHalfMoveClock() == 0) && ttBoundCutoff(flag, tbScore, alpha, beta)) {
                tt.store(hashKey, static_cast<uint8_t>(depth), scoreToTT(tbScore, ply), flag);
                return tbScore;
            }
        }
    }

    const uint8_t activeColor = b.getActiveColor();
    // One attack scan answers inCheck and gives movegen the evasion mask.
    const uint64_t checkers = b.checkersTo(activeColor);
    const bool inCheck = (checkers != 0ULL);

    // The raw eval goes back into the TT, so a later visit skips the forward
    // pass (NO_EVAL in check). The correction history learns against the
    // corrected eval; the pruning gates read staticEval, tightened by the TT.
    int32_t rawStaticEval = TT::Entry::NO_EVAL;
    int32_t correctedEval = 0;
    int32_t staticEval = 0;
    if (!inCheck) {
        rawStaticEval = (tte.hit && tte.staticEval != TT::Entry::NO_EVAL) ? tte.staticEval
                                                                          : cachedEval(hashKey, b);
        correctedEval = std::clamp(rawStaticEval + runtime.corrHist.correction(b),
                                   -TB_WIN_BOUND + 1, TB_WIN_BOUND - 1);
        staticEval = correctedEval;
        // A TT bound that proves the eval wrong replaces it (an equal one changes nothing).
        if (tte.hit && ttBoundCutoff(tte.flag, scoreFromTT(tte.score, ply), staticEval, staticEval)) {
            staticEval = scoreFromTT(tte.score, ply);
        }
        staticEval = std::clamp(staticEval, -TB_WIN_BOUND + 1, TB_WIN_BOUND - 1);
    }

    // In check there is no static eval: the sentinel keeps a stale value out of
    // the comparison two plies down, where the same side is to move.
    evalStack[ply] = inCheck ? NEG_INF : staticEval;
    const bool improving = !inCheck && ply >= 2
        && evalStack[ply - 2] != NEG_INF && staticEval > evalStack[ply - 2];

    const int side = Board::colorToIndex(activeColor);
    int singularExtension = 0;
    if (!hasExcludedMove && !inCheck && depth >= SE_MIN_DEPTH && ply > 0 && tte.hit
        && tte.depth >= depth - SE_DEPTH_MARGIN
        && (tte.flag == TT::Entry::LOWERBOUND || tte.flag == TT::Entry::EXACT)
        && std::abs(tte.score) < TB_WIN_BOUND && tte.move != 0) {
        const int32_t ttSeScore = scoreFromTT(tte.score, ply);
        const int32_t seBeta = ttSeScore - SE_BETA_MARGIN * depth;
        // Only this node's own store is suppressed (by the excluded move); the
        // probe's descendants store normally.
        const int32_t seScore = searchPosition(b, runtime, depth / 2 - 1, seBeta - 1, seBeta, ply,
                                               allowHeuristicUpdates, previousMove, false,
                                               TT::Entry::decodeMove(tte.move), cutNode);
        if (seScore < seBeta) {
            singularExtension = 1;
        } else if (seScore >= beta) {
            return seScore; // multi-cut: every move beats beta
        } else if (ttSeScore >= beta) {
            singularExtension = -1; // the TT expects a cutoff from some other move
        }
    }

    // Interior non-PV node past the root: the gate for NMP, RFP and ProbCut.
    const bool interiorNonPv = !isPVNode && !inCheck && ply > 0;

    // Null move when the eval is within ~100cp of beta. One non-pawn piece is
    // enough of a zugzwang guard; pawn-only endings skip it.
    if (allowNullMove && interiorNonPv && depth >= 4 && staticEval + 100 >= beta
        && (b.bb[Board::KNIGHT][side] | b.bb[Board::BISHOP][side]
            | b.bb[Board::ROOK][side] | b.bb[Board::QUEEN][side]) != 0ULL) {
        // The further the eval sits above beta, the deeper the reduction.
        if constexpr (kSearchStats) ++gSearchStats.nmpTries;
        const int reduction = NMP_BASE_R + depth / 3 + std::clamp((staticEval - beta) / NMP_EVAL_DIV, 0, NMP_EVAL_MAX);
        Board::MoveState nullState;
        b.doNullMove(nullState);
        const int32_t nullScore = -searchPosition(b, runtime, depth - reduction, -beta, -alpha, ply + 1,
                                                  allowHeuristicUpdates, nullptr, false);
        b.undoNullMove(nullState);
        // Fail-soft, but a mate or TB win found behind a null move is unproven.
        if (nullScore >= beta) {
            if constexpr (kSearchStats) ++gSearchStats.nmpCuts;
            return (nullScore >= TB_WIN_BOUND) ? beta : nullScore;
        }
    }

    // Reverse futility: the eval beats beta by a margin per ply. Halfway between
    // eval and beta, since the margin test proves only beta.
    if (interiorNonPv && depth <= 3 && staticEval - RFP_MARGIN_PER_DEPTH * depth >= beta) {
        if constexpr (kSearchStats) ++gSearchStats.rfpCuts;
        return (staticEval + beta) / 2;
    }

    // ProbCut: a good capture that still beats beta + margin in a reduced search
    // (behind a qsearch filter above depth 4) makes a fail high very likely.
    if (interiorNonPv && depth >= PROBCUT_MIN_DEPTH && std::abs(beta) < POS_INF - 1000) {
        const int32_t probcutBound = beta + PROBCUT_MARGIN;
        for (const Move& mc : MoveGenerator::generateTacticalMoves(b)) {
            // A singular probe asks about the OTHER moves: the excluded one must not answer.
            if (hasExcludedMove && mc.sameFromTo(excludedMove)) continue;
            // Raw SEE on purpose: PROBCUT_MARGIN was tuned as a material threshold here.
            if (Sorter::staticExchangeEvaluation(b, mc) < PROBCUT_MARGIN) continue;
            Board::MoveState pcState;
            b.doMove(mc, pcState);
            int32_t pcScore = probcutBound;
            if (depth > 4) pcScore = -quiescenceSearch(b, runtime, -probcutBound, -probcutBound + 1, ply + 1);
            if (pcScore >= probcutBound) {
                pcScore = -searchPosition(b, runtime, depth - 4, -probcutBound, -probcutBound + 1, ply + 1,
                                          false, &mc, false, Move{}, !cutNode);
            }
            b.undoMove(mc, pcState);
            if (pcScore >= probcutBound) {
                if constexpr (kSearchStats) ++gSearchStats.probcutCuts;
                return beta;
            }
        }
    }

    int16_t* contHistEntry = nullptr;
    if (previousMove != nullptr && previousMove->to < 64) {
        const int prevPiece = b.get(previousMove->to) & Board::MASK_PIECE_TYPE;
        DBG_ONLY(
            if (prevPiece == 0) {
                DBG_LOG_STREAM("contHist: EMPTY square under previousMove->to\n");
            }
        );
        contHistEntry = &runtime.contHist[side ^ 1][prevPiece - 1][previousMove->to][0][0];
    }

    MoveList moves = MoveGenerator::generateLegalMoves(b, checkers);
    if (moves.is_empty()) return inCheck ? (-MATE_VALUE + ply) : 0; // mated or stalemated

    bool hasHashMove = false;
    MovePicker movePicker = Sorter::sortLegalMoves(std::move(moves), ply, b, runtime, tte.move,
                                                   previousMove, &hasHashMove, contHistEntry);
    // Without a verified hash move: no singular extension, and IIR reduces.
    const SearchContext ctx{
        previousMove, contHistEntry, depth, ply, staticEval,
        hasHashMove ? singularExtension : 0, excludedMove, activeColor, inCheck, isPVNode,
        !hasHashMove && depth >= 6 && ply > 0, improving, cutNode && !isPVNode
    };

    const SearchMoveResult result = searchMoves(b, movePicker, ctx, alpha, beta, runtime, allowHeuristicUpdates);
    const int32_t best = result.score;
    if (runtime.isInterrupted()) return evaluate(b);

    // Learn only where the score contradicts the eval in the direction the
    // bound can prove: a raised alpha proves "at least best", a fail low
    // "at most best", or only "at most alpha" when pruned moves never raised
    // best (Stockfish lifts its fail-low best to the futility value instead).
    // A capture best move says more about the capture than about the eval.
    if (!inCheck && !hasExcludedMove && allowHeuristicUpdates && std::abs(best) < TB_WIN_BOUND) {
        const bool raisedAlpha = best > alpha;
        const bool bestIsCapture = raisedAlpha
            && moveTraits(b, result.move, b.getEnPassant(), Board::promotionRank(activeColor == Board::WHITE)).isCapture;
        const int32_t target = (!raisedAlpha && result.pruned) ? alpha : best;
        if (!bestIsCapture && (target > correctedEval) == raisedAlpha) {
            const int32_t bonus = std::clamp((target - correctedEval) * depth / (raisedAlpha ? 10 : 8),
                                             -CORR_BONUS_CAP, CORR_BONUS_CAP);
            const CorrectionHistory::Cells cells = runtime.corrHist.cellsFor(b);
            applyHistoryGravity(*cells.pawn, bonus, CORR_HIST_LIMIT);
            applyHistoryGravity(*cells.minor, bonus * 156 / 128, CORR_HIST_LIMIT);
            applyHistoryGravity(*cells.nonPawnWhite, bonus * 178 / 128, CORR_HIST_LIMIT);
            applyHistoryGravity(*cells.nonPawnBlack, bonus * 178 / 128, CORR_HIST_LIMIT);
        }
    }

    // Under an excluded move the score belongs to a reduced move set: only this
    // node's store is skipped. A fail-low node has no best move, only the least
    // refuted one, so it stores none and keeps the entry's.
    if (!hasExcludedMove) {
        const auto flag = determineFlag(best, alpha, beta);
        tt.store(hashKey, static_cast<uint8_t>(depth), scoreToTT(best, ply), static_cast<uint8_t>(flag),
                 flag == TT::Entry::UPPERBOUND ? 0 : TT::Entry::encodeMove(result.move), rawStaticEval);
    }
    return best;
}

int32_t quiescenceSearch(Board& b, SearchRuntime& runtime, int32_t alpha, int32_t beta,
                         int ply, bool alreadyEntered) noexcept {
    if constexpr (kSearchStats) ++gSearchStats.qNodes;
    int32_t early = 0;
    if (!alreadyEntered && (enterNode(b, runtime, ply, early) || checkDrawTerminalConditions(b, ply, early))) {
        return early;
    }

    TT& tt = *runtime.transpositionTable;
    // Any stored depth qualifies here (qsearch stores depth 0).
    const TT::ProbeResult tte = tt.probeEntry(b.getHash());
    if (tte.hit) {
        const int32_t ttScore = scoreFromTT(tte.score, ply);
        if (ttBoundCutoff(tte.flag, ttScore, alpha, beta)) return ttScore;
    }

    const uint64_t checkers = b.checkersTo(b.getActiveColor());
    const bool inCheck = (checkers != 0ULL);
    if (ply >= MAX_QSEARCH_DEPTH) {
        if (inCheck && MoveGenerator::generateLegalMoves(b, checkers).is_empty()) return -MATE_VALUE + ply;
        return evaluate(b);
    }

    const int32_t alphaOrig = alpha;
    int32_t rawStaticEval = TT::Entry::NO_EVAL;
    int32_t best = NEG_INF;
    if (!inCheck) {
        // The probe above pulled this bucket into cache, so a recorded eval is
        // free here, where 50-80% of all nodes are.
        rawStaticEval = (tte.hit && tte.staticEval != TT::Entry::NO_EVAL) ? tte.staticEval
                                                                          : cachedEval(b.getHash(), b);
        best = std::clamp(rawStaticEval + runtime.corrHist.correction(b), -TB_WIN_BOUND + 1, TB_WIN_BOUND - 1);
        // As in the main search, a TT bound that proves the eval wrong replaces it.
        if (tte.hit && ttBoundCutoff(tte.flag, scoreFromTT(tte.score, ply), best, best)) {
            best = std::clamp(scoreFromTT(tte.score, ply), -TB_WIN_BOUND + 1, TB_WIN_BOUND - 1);
        }
        // Fail-soft stand pat. No store: the eval cache already returns the eval.
        if (best >= beta) return best;
        alpha = std::max(alpha, best);
    }

    MovePicker movePicker = inCheck ? MoveGenerator::generateQSearchEvasions(b, checkers)
                                    : MoveGenerator::generateQSearchTacticalMoves(b, best, alpha, ply);
    if (!movePicker.hasNext()) return inCheck ? (-MATE_VALUE + ply) : best; // mated, or the stand pat

    while (movePicker.hasNext()) {
        const Move m = movePicker.nextMove();
        if (runtime.shouldAbort()) {
            runtime.markInterrupted();
            return evaluate(b);
        }
        Board::MoveState state;
        b.doMove(m, state);
        tt.prefetch(b.getHash());
        const int32_t score = -quiescenceSearch(b, runtime, -beta, -alpha, ply + 1);
        b.undoMove(m, state);
        best = std::max(best, score);
        alpha = std::max(alpha, score);
        if (score >= beta) break; // fail-soft; the store below records best
    }

    if (!inCheck) {
        tt.store(b.getHash(), 0, scoreToTT(best, ply),
                 static_cast<uint8_t>(determineFlag(best, alphaOrig, beta)), /*bestMove=*/0, rawStaticEval);
    }
    return best;
}

Move getBestMove(Board& rootBoard, const MoveList& moves, SearchRuntime& runtime,
                 int32_t alpha = NEG_INF, int32_t beta = POS_INF) noexcept {
    int32_t bestScore = NEG_INF;
    Move bestMove = moves[0];
    bool searchedAnyMove = false;

    MovePicker ordered = Sorter::sortLegalMoves(moves, 0, rootBoard, runtime,
                                                runtime.transpositionTable->probeEntry(rootBoard.getHash()).move);
    ordered.fullSort(); // the root iterates the list directly

    // ply 2 compares against the root, which never writes slot 0 itself;
    // unseeded, `improving` there would mean eval > 0.
    evalStack[0] = rootBoard.checkersTo(rootBoard.getActiveColor()) ? NEG_INF : evaluate(rootBoard);

    const auto scoreMove = [&](const Move& m, int32_t lo, int32_t hi, bool cut) {
        Board::MoveState state;
        rootBoard.doMove(m, state);
        const int32_t score = -searchPosition(rootBoard, runtime, std::max(0, runtime.depth - 1), -hi, -lo, 1,
                                              true, nullptr, true, Move{}, cut);
        rootBoard.undoMove(m, state);
        return score;
    };

    // Plain PVS: the full window for the first move, a null-window scout and a
    // re-search for the rest. Parallelism lives in searchBestMove.
    for (int i = 0; i < ordered.moves.size; ++i) {
        if (runtime.shouldAbort()) {
            runtime.markInterrupted();
            break;
        }
        const Move& m = ordered.moves[i];
        int32_t score = scoreMove(m, alpha, (i == 0) ? beta : alpha + 1, i != 0);
        if (i > 0 && score > alpha) score = scoreMove(m, alpha, beta, false);

        searchedAnyMove = true;
        if (i > 0 && runtime.isInterrupted()) break;
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
        alpha = std::max(alpha, score);
        if (i > 0 && bestScore >= beta) break;
    }

    if (searchedAnyMove) runtime.eval = bestScore;
    return bestMove;
}

IterativeSearchResult runIterativeDeepening(Board& rootBoard, SearchRuntime& runtime,
                                            int startDepth, int targetDepth) noexcept {
    IterativeSearchResult result;
    if constexpr (kSearchStats) gSearchStats = SearchStats{};
    const int firstDepth = std::max(1, startDepth);
    const int maxDepth = std::max(firstDepth, targetDepth);

    if (rootBoard.bb[Board::KING][0] == 0 || rootBoard.bb[Board::KING][1] == 0) {
        const bool ourKingGone = (rootBoard.bb[Board::KING][0] == 0) == (rootBoard.getActiveColor() == Board::WHITE);
        result.terminalRoot = result.completedAnyDepth = true;
        result.bestScore = runtime.eval = ourKingGone ? -MATE_VALUE : MATE_VALUE; // side-to-move relative
        return result;
    }

    int32_t rootDrawScore = 0;
    if (checkDrawTerminalConditions(rootBoard, 0, rootDrawScore, /*atRoot=*/true)) {
        const MoveList drawMoves = MoveGenerator::generateLegalMoves(rootBoard);
        result.terminalRoot = result.completedAnyDepth = true;
        result.bestScore = runtime.eval = rootDrawScore;
        result.bestMove = drawMoves.is_empty() ? Move{} : drawMoves[0];
        return result;
    }

    MoveList moves = MoveGenerator::generateLegalMoves(rootBoard);
    if (moves.is_empty()) {
        // Checkmated (side to move, negamax) or stalemated.
        result.bestScore = runtime.eval = rootBoard.inCheck(rootBoard.getActiveColor()) ? -MATE_VALUE : 0;
        return result;
    }

    // A root in TB range plays the best DTZ move without a search: the tables
    // already order the moves by exact distance to zeroing, which the eval
    // cannot. Main thread only: tb_probe_root is not thread-safe, and a helper's
    // info line would interleave with the main thread's output.
    if (!runtime.isHelper && runtime.syzygyProber != nullptr && rootBoard.castlingRights() == 0) {
        const auto tbMoves = runtime.syzygyProber->probeRoot(rootBoard);
        const auto tbBest = std::ranges::max_element(tbMoves, {}, &syzygy::RootMove::tbRank);
        // Played only if it is in our own legal-move list, an encoding check.
        if (tbBest != tbMoves.end() && std::find(moves.begin(), moves.end(), tbBest->move) != moves.end()) {
            const int32_t tbScore = syzygy::SyzygyProber::wdlToScore(
                tbBest->tbRank > 0 ? syzygy::WDL::Win : tbBest->tbRank < 0 ? syzygy::WDL::Loss : syzygy::WDL::Draw, 0);
            result.completedAnyDepth = result.tablebaseRoot = true;
            result.bestMove = tbBest->move;
            result.bestScore = runtime.eval = tbScore;
            if (runtime.emitUciInfo) {
                uci::emit("info depth 1 score cp " + std::to_string(tbScore) + " tbrank " + std::to_string(tbBest->tbRank)
                          + " pv " + tbBest->move.toUCIString() + "\n");
            }
            return result;
        }
    }

    Move bestMove = moves[0];
    int32_t prevScore = 0;
    int32_t prevPrevScore = 0;
    bool hasPrevPrevScore = false;

    for (int currentDepth = firstDepth; currentDepth <= maxDepth; ++currentDepth) {
        if (runtime.shouldAbort()) break;
        // Lazy SMP depth diversification (helpers only; size 0 on the main thread).
        if (runtime.depthSkipSize > 0
            && (((currentDepth + runtime.depthSkipPhase) / runtime.depthSkipSize) & 1) != 0) {
            continue;
        }
        // Soft limit: with a move in hand, do not open a depth we cannot finish.
        if (result.completedAnyDepth && runtime.timeManager != nullptr
            && !runtime.timeManager->shouldStartNextDepth()) {
            break;
        }

        runtime.depth = currentDepth;
        if (result.completedAnyDepth) {
            const auto it = std::find(moves.begin(), moves.end(), bestMove);
            if (it != moves.end()) std::rotate(moves.begin(), it, it + 1);
        }
        runtime.clearInterrupted();

        int32_t iterationAlpha = NEG_INF;
        int32_t iterationBeta = POS_INF;
        Move candidate;
        if (!result.completedAnyDepth || currentDepth < 5 || std::abs(prevScore) >= TB_WIN_BOUND) {
            candidate = getBestMove(rootBoard, moves, runtime);
        } else {
            // Aspiration window sized by the last score swing, widened on each fail.
            const int32_t swing = std::abs(prevScore - (hasPrevPrevScore ? prevPrevScore : prevScore));
            int32_t delta = std::clamp(15 + swing / 4, 25, 100);
            int32_t center = prevScore;
            int32_t below = delta;
            int32_t above = delta;
            for (int researches = 0;;) {
                iterationAlpha = std::clamp(center - below, NEG_INF, POS_INF);
                iterationBeta  = std::clamp(center + above, NEG_INF, POS_INF);
                candidate = getBestMove(rootBoard, moves, runtime, iterationAlpha, iterationBeta);
                if (runtime.isInterrupted()) break;
                const int32_t score = runtime.eval;
                const bool failLow = (score <= iterationAlpha);
                if (!failLow && score < iterationBeta) break;
                center = failLow ? std::min(center, score) : std::max(center, score);
                delta = std::min(WINDOW_HARD_CAP, delta * 2 + 10);
                if (++researches >= MAX_ASP_RESEARCHES || delta >= WINDOW_HARD_CAP) {
                    iterationAlpha = NEG_INF;
                    iterationBeta = POS_INF;
                    candidate = getBestMove(rootBoard, moves, runtime);
                    break;
                }
                below = failLow ? delta : std::max(25, delta / 2);
                above = failLow ? std::max(25, delta / 2) : delta;
            }
        }
        if (runtime.isInterrupted()) break;

        if (runtime.timeManager != nullptr) {
            runtime.timeManager->updateStability(result.completedAnyDepth && !(bestMove == candidate),
                                                 runtime.eval, prevScore, result.completedAnyDepth);
        }
        hasPrevPrevScore = result.completedAnyDepth;
        prevPrevScore = prevScore;
        prevScore = runtime.eval;
        bestMove = candidate;

        result.completedAnyDepth = true;
        result.completedDepth = currentDepth;
        result.bestMove = bestMove;
        result.bestScore = runtime.eval;
        runtime.transpositionTable->store(rootBoard.getHash(), currentDepth, scoreToTT(runtime.eval, 0),
                                          static_cast<uint8_t>(determineFlag(runtime.eval, iterationAlpha, iterationBeta)),
                                          TT::Entry::encodeMove(bestMove));
        if (runtime.emitUciInfo) emitUciInfoLine(currentDepth, runtime.eval, runtime, rootBoard);
    }

    // A deadline before depth 1 completes (a few ms left, or the thread
    // descheduled past the watchdog) leaves no searched move. Depth 1 costs
    // microseconds: finish it with the stop requests masked.
    if (!result.completedAnyDepth && !runtime.isHelper) {
        auto* const stopRequest = runtime.stopSearchRequested;
        runtime.stopSearchRequested = nullptr;
        runtime.clearInterrupted();
        runtime.depth = 1;
        const Move depthOneMove = getBestMove(rootBoard, moves, runtime);
        runtime.stopSearchRequested = stopRequest;
        if (!runtime.isInterrupted()) {
            result.completedAnyDepth = true;
            result.completedDepth    = 1;
            result.bestMove          = depthOneMove;
            result.bestScore         = runtime.eval;
        }
    }
    // Counters are per thread: only the main thread reports its own.
    if constexpr (kSearchStats) if (!runtime.isHelper) {
        const SearchStats& st = gSearchStats;
        const auto pct = [](uint64_t x, uint64_t y) {
            return y ? std::to_string(static_cast<double>(x) * 100.0 / static_cast<double>(y)).substr(0, 5) : std::string("-");
        };
        const auto avg = [](uint64_t x, uint64_t y) {
            return y ? std::to_string(static_cast<double>(x) / static_cast<double>(y)).substr(0, 5) : std::string("-");
        };
        const uint64_t all = st.mainNodes + st.qNodes;
        uci::emit("info string stats nodes main " + std::to_string(st.mainNodes) + " q " + std::to_string(st.qNodes)
                  + " qshare " + pct(st.qNodes, all) + "%\n"
                  + "info string stats tt probes " + std::to_string(st.ttProbes) + " hit " + pct(st.ttHits, st.ttProbes)
                  + "% cut " + pct(st.ttCuts, st.ttProbes) + "%\n"
                  + "info string stats cut nodes " + std::to_string(st.cutNodes) + " nocut " + std::to_string(st.noCutNodes)
                  + " firstmove " + pct(st.cutFirst, st.cutNodes) + "% avgindex " + avg(st.cutIndexSum, st.cutNodes) + "\n"
                  + "info string stats lmr searches " + std::to_string(st.lmrSearches) + " avgred "
                  + avg(st.lmrReductionSum, st.lmrSearches) + " research " + pct(st.lmrResearches, st.lmrSearches) + "%\n"
                  + "info string stats pruned lmp " + std::to_string(st.prunedLmp) + " fut " + std::to_string(st.prunedFutility)
                  + " hist " + std::to_string(st.prunedHistory) + " see " + std::to_string(st.prunedSee) + "\n"
                  + "info string stats early nmp " + std::to_string(st.nmpCuts) + "/" + std::to_string(st.nmpTries) + " ("
                  + pct(st.nmpCuts, st.nmpTries) + "%) rfp " + std::to_string(st.rfpCuts) + " probcut "
                  + std::to_string(st.probcutCuts) + " of main " + pct(st.nmpCuts + st.rfpCuts + st.probcutCuts, st.mainNodes)
                  + "%\n");
    }
    return result;
}

Move searchBestMove(Board& board, SearchRuntime& runtime, int requestedDepth) noexcept {
    const int targetDepth = (requestedDepth == 0) ? DEFAULT_DEPTH : requestedDepth;
    runtime.clearInterrupted();
    runtime.transpositionTable->incrementGeneration();
    runtime.nodesSearched = 0;
    runtime.softResetHistory();

    // Lazy SMP: helpers run their own iterative deepening on private boards and
    // runtimes, sharing only the TT. The main thread's result stands unless the
    // vote below picks a helper's; helpers stop the moment it returns.
    const int helperCount = std::clamp(runtime.maxThreads - 1, 0, MAX_HELPER_THREADS);
    std::atomic<bool> helperStop{false};
    std::vector<std::thread> helpers;
    helpers.reserve(helperCount);

    for (int i = 0; i < helperCount; ++i) {
        if (std::cmp_greater_equal(i, helperSlots.size())) {
            helperSlots.emplace_back(std::make_unique<HelperSlot>());
        }
        HelperSlot& slot = *helperSlots[i];
        SearchRuntime& hr = slot.runtime;
        hr.transpositionTable  = runtime.transpositionTable;
        hr.syzygyProber        = runtime.syzygyProber;
        hr.stopSearchRequested = &helperStop;
        hr.maxNodes            = runtime.maxNodes;
        hr.nodesSearched       = 0;
        hr.clearInterrupted();
        hr.softResetHistory();
        slot.board = board;
        slot.result = {};

        // Spread the helpers over the depth sequence, the pattern repeating
        // every 20; odd helpers also start one ply deeper.
        static constexpr int SKIP_SIZE[20]  = {1,1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4};
        static constexpr int SKIP_PHASE[20] = {0,1,0,1,2,3,0,1,2,3,4,5,0,1,2,3,4,5,6,7};
        hr.depthSkipSize  = SKIP_SIZE[i % 20];
        hr.depthSkipPhase = SKIP_PHASE[i % 20];

        const int startDepth = 1 + (i & 1);
        helpers.emplace_back([&slot, startDepth, targetDepth] {
            slot.result = runIterativeDeepening(slot.board, slot.runtime, startDepth, targetDepth);
        });
    }
    activeHelperCount = helperCount;

    IterativeSearchResult result = runIterativeDeepening(board, runtime, 1, targetDepth);

    helperStop.store(true, std::memory_order_release);
    for (auto& helper : helpers) helper.join();
    runtime.depth = targetDepth;

    // Thread vote: among the threads that completed the deepest iteration, the
    // highest root score wins, the main thread on ties. Scores are compared at
    // equal depth only.
    if (helperCount > 0 && result.completedAnyDepth && !result.terminalRoot && !result.tablebaseRoot) {
        const IterativeSearchResult* best = &result;
        for (int i = 0; i < helperCount; ++i) {
            const IterativeSearchResult& hres = helperSlots[i]->result;
            if (!hres.completedAnyDepth || hres.bestMove == Move{}) continue;
            if (hres.completedDepth > best->completedDepth
                || (hres.completedDepth == best->completedDepth && hres.bestScore > best->bestScore)) {
                best = &hres;
            }
        }
        // The last info line was the main thread's: a helper's winning move
        // gets its own, or the GUI shows a PV for another move.
        if (best != &result && runtime.emitUciInfo && !(best->bestMove == result.bestMove)) {
            emitUciInfoLine(best->completedDepth, best->bestScore, runtime, board, &best->bestMove);
        }
        result.bestMove  = best->bestMove;
        result.bestScore = best->bestScore;
    }
    // Only now: the vote's info line above still counts the helpers' nodes.
    activeHelperCount = 0;

    if (result.terminalRoot || result.completedAnyDepth) {
        runtime.eval = result.bestScore;
        return result.bestMove;
    }
    // Nothing completed: a deterministic fallback move.
    const MoveList fallbackMoves = MoveGenerator::generateLegalMoves(board);
    runtime.eval = evaluate(board);
    return fallbackMoves.is_empty() ? Move{} : fallbackMoves[0];
}

} // namespace Searcher

} // namespace engine
