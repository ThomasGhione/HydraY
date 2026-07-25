#include "searcher.hpp"

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "../engine.hpp"
#include "../evaluator.hpp"
#include "../sort/move_generator.hpp"

namespace engine {

namespace {

constexpr int32_t REPETITION_DRAW_ADVANTAGE_THRESHOLD = PAWN_VALUE / 2;

// Precomputed LMR reductions: LMR_TABLE[depth][moveIndex], capped at depth-3.
// Avoids two std::log() calls per LMR candidate in the hot search loop.
// Built at startup and rebuilt by rebuildSearchDerivedTables() when the UCI
// layer changes LMR_C_PERCENT (search is stopped first).
struct LMRTable {
    int8_t data[LMR_MAX_DEPTH][LMR_MAX_MOVES]{};
    void rebuild() noexcept {
        const double lmrC = static_cast<double>(LMR_C_PERCENT) / 100.0;
        for (int d = 0; d < LMR_MAX_DEPTH; ++d) {
            for (int m = 0; m < LMR_MAX_MOVES; ++m) {
                if (d == 0 || m == 0) { data[d][m] = 1; continue; }
                const double raw = __builtin_log(static_cast<double>(d))
                                 * __builtin_log(static_cast<double>(m))
                                 / lmrC;
                const int r = static_cast<int>(raw);
                data[d][m] = static_cast<int8_t>(std::max(1, std::min(r, d - 3)));
            }
        }
    }
    LMRTable() noexcept { rebuild(); }
};
static LMRTable LMR_REDUCTION_TABLE;

} // namespace

void rebuildSearchDerivedTables() noexcept {
    LMR_REDUCTION_TABLE.rebuild();
    for (int d = 1; d <= 6; ++d) {
        FUTILITY_MARGINS[d] = FUTILITY_MID_STEP * d;
    }
    for (int improving = 0; improving < 2; ++improving) {
        for (int d = 1; d <= 4; ++d) {
            LMP_THRESHOLDS[improving][d] =
                LMP_BASE_THRESHOLDS[improving][d] * LMP_SCALE_PCT[improving] / 100;
        }
    }
}


constexpr int32_t Searcher::saturatingAdd32(int32_t lhs, int32_t rhs) noexcept {
    const int64_t sum = static_cast<int64_t>(lhs) + static_cast<int64_t>(rhs);
    return static_cast<int32_t>(std::clamp<int64_t>(sum, NEG_INF, POS_INF));
}

constexpr int32_t Searcher::saturatingSub32(int32_t lhs, int32_t rhs) noexcept {
    const int64_t diff = static_cast<int64_t>(lhs) - static_cast<int64_t>(rhs);
    return static_cast<int32_t>(std::clamp<int64_t>(diff, NEG_INF, POS_INF));
}

constexpr int32_t Searcher::scoreToTT(int32_t score, int ply) noexcept {
    if (score >= MATE_BOUND)  return saturatingAdd32(score, ply);
    if (score <= -MATE_BOUND) return saturatingSub32(score, ply);
    return score;
}

constexpr int32_t Searcher::scoreFromTT(int32_t score, int ply) noexcept {
    if (score >= MATE_BOUND)  return saturatingSub32(score, ply);
    if (score <= -MATE_BOUND) return saturatingAdd32(score, ply);
    return score;
}

void Searcher::applyHistoryGravity(int16_t& cell, int32_t delta, int32_t maxValue) noexcept {
    const int32_t magnitude = (delta < 0) ? -delta : delta;
    int32_t value = cell;
    value += delta - value * magnitude / maxValue;
    constexpr int32_t MAX_I16 = 32767; // fits in int16_t after clamping
    constexpr int32_t MIN_I16 = -32768; // fits in int16_t after clamping
    cell = static_cast<int16_t>(std::clamp(value, MIN_I16, MAX_I16));
}


bool Searcher::checkEarlyTerminalConditions(
    const chess::Board& b,
    SearchRuntime& runtime,
    int ply,
    int32_t& outScore) noexcept {
    
    if (runtime.shouldAbort()) {
        runtime.markInterrupted();
        outScore = Evaluator::evaluate(b);
        return true;
    }

    if (ply >= MAX_PLY - 1) {
        outScore = Evaluator::evaluate(b);
        return true;
    }

    // Terminal king-capture states are possible in this codebase's move model.
    // Negamax / side-to-move relative: losing our own king is a mated score
    // (NEG_INF + ply); capturing the opponent's king is a winning score.
    if (b.kings_bb[0] == 0ULL || b.kings_bb[1] == 0ULL) {
        const bool whiteToMove = (b.getActiveColor() == chess::Board::WHITE);
        const bool ourKingGone = (b.kings_bb[0] == 0ULL) ? whiteToMove : !whiteToMove;
        outScore = ourKingGone ? (NEG_INF + ply) : (POS_INF - ply);
        return true;
    }

    return false;
}

// Per-node entry shared by searchPosition and quiescenceSearch: bump the node
// counter, enforce the UCI `go nodes` hard cap, and run the early-terminal
// checks (abort / max-ply / king-capture). Returns true with outScore set when
// the node must return immediately.
bool Searcher::enterNode(
    chess::Board& b, SearchRuntime& runtime, int ply,
    uint64_t* counter, int32_t& outScore) noexcept {
    ++(*counter);
    // runtime.nodesSearched holds prior IDS iterations; *counter is the current
    // iteration/worker. Their sum is the total node budget consumed.
    if (runtime.maxNodes > 0 && runtime.nodesSearched + *counter >= runtime.maxNodes) [[unlikely]] {
        runtime.markInterrupted();
        outScore = Evaluator::evaluate(b);
        return true;
    }
    return checkEarlyTerminalConditions(b, runtime, ply, outScore);
}

bool Searcher::checkDrawTerminalConditions(
    const chess::Board& b,
    int32_t& outScore,
    bool atRoot) noexcept {
    // A repetition needs at least 4 reversible plies (each side out and back;
    // this guard removes the scan from most of the tree.
    if (b.getHalfMoveClock() >= 4) {
        const int repCount = b.countRepetitions();

        // Third repetition: forced draw, apply full contempt penalty.
        if (repCount >= 3) {
            // Modest fixed contempt below: enough that the engine prefers to
            // keep playing when winning, but smaller than any meaningful
            // material amount so it won't trade a piece to avoid the draw.
            const int32_t staticEval = Evaluator::evaluate(b);
            outScore = (std::abs(staticEval) <= REPETITION_DRAW_ADVANTAGE_THRESHOLD)
                ? 0
                : (staticEval > 0 ? -REPETITION_CONTEMPT : REPETITION_CONTEMPT);
            return true;
        }

        // Second repetition: not yet a forced draw, but scores as 0.
        // This prevents the engine from "chasing" draws when winning (alpha > 0
        // won't be improved by a 0 score, so the engine is forced to find real moves).
        //
        // This is an INTERIOR-NODE heuristic only. At the root the current position
        // is not a draw under FIDE rules until it actually occurs a third time, so
        // returning here would abandon the search and play moves[0] - a random legal
        // move that, as observed, can hang the queen. At the root we must search.
        if (!atRoot && repCount >= 2) {
            outScore = 0;
            return true;
        }
    }

    if (b.isFiftyMoveRule()) [[unlikely]] {
        outScore = 0;
        return true;
    }

    if (b.hasInsufficientMaterialDraw()) [[unlikely]] {
        outScore = 0;
        return true;
    }

    return false;
}


void Searcher::updateMinMax(
    int32_t score,
    int32_t& alpha,
    int32_t& bestScore,
    chess::Move& bestMove,
    const chess::Move& m) noexcept {
    const bool better = isBetter(score, bestScore);
    bestScore = better ? score : bestScore;
    bestMove = better ? m : bestMove;

    updateBound(score, alpha);
}

namespace {

// Lazy SMP helper state: one slot per helper thread, persistent across
// searches so history/killers keep learning like the main runtime does.
// searchBestMove is single-entrant (the UCI layer serializes searches),
// so function-scope statics are safe. The Board lives here (not in a local
// vector) so the worker thread's reference is stable by construction.
struct HelperSlot {
    ::engine::SearchRuntime runtime; // defaults are already helper-correct:
                                     // maxThreads=1, emitUciInfo=false, timeManager=nullptr
    chess::Board board;
    std::atomic<bool> interrupted{false};

    HelperSlot() noexcept {
        runtime.searchInterrupted = &interrupted;
        runtime.isHelper = true;
    }
};

std::vector<std::unique_ptr<HelperSlot>> helperSlots;

} // namespace

chess::Move Searcher::searchBestMove(
    chess::Board& board,
    SearchRuntime& runtime,
    int requestedDepth) noexcept {
    const int targetDepth = (requestedDepth == 0) ? DEFAULT_DEPTH : requestedDepth;

    runtime.clearInterrupted();

    if (runtime.transpositionTable != nullptr) {
        runtime.transpositionTable->incrementGeneration();
    }
    runtime.nodesSearched = 0;
    runtime.softResetHistory();

    // Lazy SMP: helpers run their own full iterative deepening on private
    // Boards and runtimes, sharing only the TT, and through it move ordering
    // and cutoff knowledge, with the main thread. Odd helpers start one ply
    // deeper so the pack doesn't lockstep over identical trees. The main
    // thread's result is authoritative; helpers stop the moment it returns.
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

        const int startDepth = 1 + (i & 1);
        helpers.emplace_back([&slot, startDepth, targetDepth] {
            (void)runIterativeDeepening(slot.board, slot.runtime, startDepth, targetDepth);
        });
    }

    IterativeSearchResult result = runIterativeDeepening(board, runtime, 1, targetDepth);

    helperStop.store(true, std::memory_order_release);
    for (auto& helper : helpers) helper.join();

    runtime.depth = targetDepth;

    // Completed/terminal result, else a deterministic fallback move.
    if (result.terminalRoot || result.completedAnyDepth) {
        runtime.eval = result.bestScore;
        return result.bestMove;
    }

    MoveList fallbackMoves = engine::MoveGenerator::generateLegalMoves(board);
    runtime.eval = Evaluator::evaluate(board);
    return fallbackMoves.is_empty() ? chess::Move{} : fallbackMoves[0];
}

int32_t Searcher::searchRootMoveScore(
    chess::Board& b,
    const chess::Move& m,
    SearchRuntime& runtime,
    int32_t alpha,
    int32_t beta,
    bool allowTTWrite,
    bool allowHeuristicUpdates,
    uint64_t* nodeCounter) noexcept {
    chess::Board::MoveState state;
    b.doMove(m, state);
    // Negamax: child is the opponent to move -> negate and swap/negate bounds.
    const int childDepth = std::max(0, runtime.depth - 1);
    const int32_t score = -searchPosition(
        b, runtime, childDepth, -beta, -alpha, 1,
        allowTTWrite, allowHeuristicUpdates, nullptr, nodeCounter);
    b.undoMove(m, state);
    return score;
}

bool Searcher::tryNullMovePruning(
    chess::Board& b,
    const SearchNodeState& node,
    SearchRuntime& runtime,
    int depth,
    int32_t alpha,
    int32_t beta,
    int ply,
    bool allowTTWrite,
    bool allowHeuristicUpdates,
    uint64_t* nodeCounter,
    int32_t& outScore) noexcept {
    // Eval-scaled reduction: the further static eval sits above beta, the more
    // certain the null move fails high, so reduce deeper. Clamped to >= 0 (the
    // NMP gate allows eval up to ~100cp below beta) and capped.
    const int evalReduction = std::clamp((node.staticEval - beta) / NMP_EVAL_DIV, 0, NMP_EVAL_MAX);
    const int reduction = 3 + depth / 3 + evalReduction;

    chess::Board::MoveState nullState;
    b.doNullMove(nullState);

    // Negamax: after the null move it is the opponent to move.
    const int32_t nullScore = -searchPosition(
        b, runtime, depth - reduction, -beta, -alpha, ply + 1,
        allowTTWrite, allowHeuristicUpdates, nullptr, nodeCounter, false);

    b.undoNullMove(nullState);

    if (!isBetaCutoff(nullScore, beta)) {
        return false;
    }

    bool confirmedCutoff = true;
    if (depth >= NULL_MOVE_VERIFICATION_DEPTH) {
        // Verification re-search of THIS node (same side to move): no negation.
        const int32_t verifyScore = searchPosition(
            b, runtime, depth - reduction, alpha, beta, ply,
            allowTTWrite, allowHeuristicUpdates, nullptr, nodeCounter, false);
        confirmedCutoff = isBetaCutoff(verifyScore, beta);
    }

    if (!confirmedCutoff) {
        return false;
    }

    // Fail-soft: return the null-search score (a heuristic lower bound), but
    // never propagate an unproven mate found behind a null move.
    outScore = (nullScore >= MATE_BOUND) ? beta : nullScore;
    return true;
}

bool Searcher::tryReverseFutilityPruning(
    const SearchNodeState& node,
    int depth,
    int32_t beta,
    int32_t& outScore) noexcept {
    // Precondition (guaranteed by the only caller's canReverseFutilityPrune):
    // !isPVNode && !inCheck && ply > 0 && depth <= 3.

    // Negamax: staticEval is side-to-move relative; fail high if it beats
    // beta even after subtracting the margin.
    const int32_t rfpMargin = RFP_MARGIN_PER_DEPTH * depth;
    const int32_t rfpScore = node.staticEval - rfpMargin;
    if (!isBetaCutoff(rfpScore, beta)) {
        return false;
    }

    outScore = node.staticEval;
    return true;
}

void Searcher::updateKillerAndHistoryOnBetaCutoff(
    const chess::Move& m,
    bool isCapture,
    int victimType,
    int depth,
    int ply,
    int usSide,
    SearchRuntime& runtime,
    const chess::Move* previousMove,
    int16_t* contHistEntry,
    int fromPieceType) noexcept {
    const int fromIndex = m.from;
    const int toIndex = m.to;
    const int depthPlusOne = depth + 1;
    const int32_t bonus = depthPlusOne * depthPlusOne;

    // CAPTURE HISTORY: bonus for captures that cause cutoffs.
    if (isCapture) {
        auto& chPrimary = runtime.captureHistory[usSide][toIndex][victimType][0];
        auto& chSecondary = runtime.captureHistory[usSide][toIndex][victimType][1];
        applyHistoryGravity(chPrimary, bonus, MAX_CAPTURE_HISTORY);
        applyHistoryGravity(chSecondary, bonus >> 1, MAX_CAPTURE_HISTORY);
        if (chSecondary > chPrimary) {
            std::swap(chPrimary, chSecondary);
        }

        return;
    }

    // COUNTER-MOVE: best response to previous quiet move.
    if (previousMove != nullptr) {
        runtime.counterMoves[previousMove->from][previousMove->to] = TT::Entry::encodeMove(m);
    }

    // KILLER MOVES: update while avoiding duplicates.
    auto& km1 = runtime.killerMoves[ply][0];
    const bool isAlreadyKm1 = m.sameFromTo(km1);
    if (!isAlreadyKm1) {
        auto& km2 = runtime.killerMoves[ply][1];
        km2 = km1;
        km1 = m;
    }

    // HISTORY HEURISTIC: gravity update.
    applyHistoryGravity(runtime.history[usSide][fromIndex][toIndex], bonus, MAX_HISTORY);

    // CONTINUATION HISTORY: bonus for this move given the previous move.
    if (contHistEntry != nullptr) {
        applyHistoryGravity(contHistEntry[contHistIndex(fromPieceType, toIndex)], bonus, MAX_HISTORY);
    }
}

Searcher::SearchMoveResult Searcher::searchMoves(
    chess::Board& b,
    MovePicker& movePicker,
    const SearchContext& ctx,
    int32_t alpha,
    int32_t beta,
    SearchRuntime& runtime,
    bool allowHeuristicUpdates,
    bool allowTTWrite) noexcept {
    int32_t best = NEG_INF;
    chess::Move bestMove{};
    bool searchedAnyMove = false;

    struct QuietEntry { uint8_t from; uint8_t to; uint8_t pieceType; };
    constexpr int MAX_QUIETS_TRACKED = 64;
    QuietEntry searchedQuiets[MAX_QUIETS_TRACKED];
    int numSearchedQuiets = 0;

    struct CaptureEntry { uint8_t to; uint8_t victimType; };
    constexpr int MAX_CAPTURES_TRACKED = 32;
    CaptureEntry searchedCaptures[MAX_CAPTURES_TRACKED];
    int numSearchedCaptures = 0;

    // Interior non-PV node past the root: shared gate for futility, LMP, history pruning and SEE capture pruning below.
    const bool interiorNonPv = !ctx.isPVNode && !ctx.inCheck && ctx.ply > 0;

    const bool canFutilityPrune = interiorNonPv && ctx.depth >= 1 && ctx.depth <= 6;
    const int32_t futilityMargin = canFutilityPrune ? FUTILITY_MARGINS[ctx.depth] : 0;

    const bool canLMP = interiorNonPv && ctx.depth >= 1 && ctx.depth <= 4;
    const int lmpThreshold = canLMP ? LMP_THRESHOLDS[ctx.improving][ctx.depth] : 999;

    const int usSide = chess::Board::colorToIndex(ctx.activeColor);
    const int oppKingSq = std::countr_zero(b.kings_bb[usSide ^ 1]);
    const chess::Square enPassant = b.getEnPassant();
    const int promotionRank = chess::Board::promotionRank(ctx.activeColor == chess::Board::WHITE);

    const uint8_t oppColor = chess::Board::oppositeColor(ctx.activeColor);

    while (movePicker.hasNext()) {
        const int moveIndex = movePicker.currentIndex;
        const chess::Move m = movePicker.nextMove();
        
        if (runtime.shouldAbort()) {
            runtime.markInterrupted();
            break;
        }

        if (chess::isValidSquare(ctx.excludedMove.from) && m.sameFromTo(ctx.excludedMove)) {
            continue;
        }

        const bool isFirstMove = (moveIndex == 0);

        const int fromPieceType = b.get(m.from) & chess::Board::MASK_PIECE_TYPE;
        const int toPieceType = b.get(m.to) & chess::Board::MASK_PIECE_TYPE;
        const auto cap = Sorter::classifyCapture(m, fromPieceType, toPieceType, enPassant);
        const bool wasCapture = cap.isCapture;
        const int victimType = cap.victimType;
        const bool isPromotionCandidate = (fromPieceType == chess::Board::PAWN) && (chess::rank(m.to) == promotionRank);
        const bool isQuietMove = !wasCapture && !isPromotionCandidate;

        if (canLMP && isQuietMove && moveIndex >= lmpThreshold) {
            continue;
        }

        // Pre-move check detection, computed lazily only on the quiet moves
        // that can actually be futility-pruned (its sole consumer below).
        bool preMoveGivesCheck = false;
        if (canFutilityPrune && isQuietMove && fromPieceType != chess::Board::KING) {
            preMoveGivesCheck = Sorter::givesCheckAfterQuietMoveFast(
                b, m, fromPieceType, oppKingSq, b.getPiecesBitMap());
        }

        if (canFutilityPrune && isQuietMove && !preMoveGivesCheck && moveIndex > 0
            && shouldDeltaPrune(ctx.staticEval, futilityMargin, alpha)) {
            continue;
        }

        // History-based quiet pruning: skip late quiet moves with very negative
        // history at low depth — they reliably fail to improve alpha.
        if (isQuietMove && interiorNonPv && ctx.depth >= 1 && ctx.depth <= 3 && moveIndex > 0) {
            const int32_t histScore = runtime.history[usSide][m.from][m.to];
            if (histScore < HISTORY_PRUNE_THRESHOLD[ctx.depth]) {
                continue;
            }
        }

        // SEE pruning: skip clearly losing captures at shallow non-PV nodes.
        // Move 0 (a winning or hash capture) is always searched; the margin
        // scales with depth so deeper nodes tolerate slightly worse trades.
        if (wasCapture && !isPromotionCandidate && moveIndex > 0 && interiorNonPv && ctx.depth <= 3
            && Sorter::staticExchangeEvaluation(b, m) < -SEE_CAPTURE_MARGIN * ctx.depth) {
            continue;
        }

        chess::Board::MoveState state;
        b.doMove(m, state);
        // Prefetch the child's TT bucket now: the load overlaps the child's
        // node entry, draw checks and static eval before its TT probe.
        if (runtime.transpositionTable != nullptr) {
            runtime.transpositionTable->prefetch(b.getHash());
        }

        // Late captures are reduced too: move ordering ranks good captures early,
        // so a capture reaching this index is almost always a bad/losing one.
        const bool lmrStructuralCandidate = (ctx.depth >= 4)
            && (moveIndex >= 4)
            && !isPromotionCandidate;

        const bool forcingCandidate = (wasCapture || isPromotionCandidate || moveIndex < 3);
        const bool needsCheckInfo = ctx.depth >= 2 && ctx.depth <= 4 && forcingCandidate;
        const bool givesCheck = needsCheckInfo ? b.inCheck(oppColor) : false;

        const bool shouldCheckExtend = givesCheck && forcingCandidate && ctx.depth >= 2 && ctx.depth <= 4;
        const int childDepth = ctx.depth - 1 + (shouldCheckExtend ? 1 : 0)
                             + (isFirstMove ? ctx.singularExtension : 0);
        const bool canReduce = lmrStructuralCandidate;

        // Negamax PVS scout window: full [alpha,beta] for the first move,
        // null window [alpha, alpha+1] for the rest.
        const int32_t scoutAlpha = alpha;
        const int32_t scoutBeta  = isFirstMove ? beta
                                               : saturatingAdd32(alpha, 1);

        int32_t score = 0;
        if (canReduce) {
            const int di = ctx.depth < LMR_MAX_DEPTH ? ctx.depth : LMR_MAX_DEPTH - 1;
            const int mi = moveIndex < LMR_MAX_MOVES ? moveIndex : LMR_MAX_MOVES - 1;
            int reduction = LMR_REDUCTION_TABLE.data[di][mi];

            if (ctx.isPVNode) {
                reduction = std::max(1, reduction - 1);
            }
            if (ctx.iirActive) {
                reduction = std::min(reduction + 1, childDepth - 1);
            }
            // Reduce more when the position is not improving: a late quiet move
            // is less likely to help when our static eval is falling.
            if (!ctx.improving) {
                reduction += 1;
            }
            // History adjustment (quiet moves only — quiet history is meaningless
            // for captures, which the ordering already ranks by SEE/capture history).
            if (!wasCapture) {
                const int32_t histScore = runtime.history[usSide][m.from][m.to];
                reduction -= histScore / 8192;
            }
            reduction = std::clamp(reduction, 1, childDepth - 1);

            const int reducedDepth = std::max(1, childDepth - reduction);
            score = -searchPosition(b, runtime, reducedDepth, -scoutBeta, -scoutAlpha, ctx.ply + 1,
                                    allowTTWrite, allowHeuristicUpdates, &m, ctx.nodeCounter);

            if (shouldResearchPVS(score, scoutAlpha)) {
                score = -searchPosition(b, runtime, childDepth, -scoutBeta, -scoutAlpha, ctx.ply + 1,
                                        allowTTWrite, allowHeuristicUpdates, &m, ctx.nodeCounter);
            }
        } else { // can't reduce, regular PVS search
            score = -searchPosition(b, runtime, childDepth, -scoutBeta, -scoutAlpha, ctx.ply + 1,
                                    allowTTWrite, allowHeuristicUpdates, &m, ctx.nodeCounter);
        }

        // Wide-window PV re-search, shared by both paths above. It only carries
        // information in PV nodes: in a non-PV node beta == alpha+1 == scoutBeta,
        // so this window equals the null-window scout/research and would just
        // re-probe the TT for an identical result. Gate on isPVNode to avoid the
        // duplicate node. (In the reduce path !isFirstMove always holds.)
        if (ctx.isPVNode && !isFirstMove && shouldResearchPVS(score, scoutAlpha)) {
            score = -searchPosition(b, runtime, childDepth, -beta, -alpha, ctx.ply + 1,
                                    allowTTWrite, allowHeuristicUpdates, &m, ctx.nodeCounter);
        }

        b.undoMove(m, state);
        searchedAnyMove = true;

        if (runtime.isInterrupted()) {
            break;
        }

        if (isQuietMove && numSearchedQuiets < MAX_QUIETS_TRACKED) {
            searchedQuiets[numSearchedQuiets++] =
                {m.from, m.to, static_cast<uint8_t>(fromPieceType)};
        }
        if (wasCapture && numSearchedCaptures < MAX_CAPTURES_TRACKED) {
            searchedCaptures[numSearchedCaptures++] = {m.to, static_cast<uint8_t>(victimType)};
        }

        updateMinMax(score, alpha, best, bestMove, m);

        if (isBetaCutoff(best, beta)) {
            if (allowHeuristicUpdates) {
                updateKillerAndHistoryOnBetaCutoff(
                    m, wasCapture, victimType, ctx.depth, ctx.ply, usSide, runtime, ctx.previousMove, ctx.contHistEntry, fromPieceType);

                const int depthPlusOne = ctx.depth + 1;
                const int32_t malus = -(depthPlusOne * depthPlusOne);

                // Malus to quiet moves searched before the cutoff. When the
                // cutoff move is itself quiet it is the last tracked entry and
                // is excluded (it gets the bonus instead); when the cutoff is a
                // capture, every tracked quiet failed and is penalised.
                const int quietMalusEnd = numSearchedQuiets - (isQuietMove ? 1 : 0);
                for (int i = 0; i < quietMalusEnd; ++i) {
                    applyHistoryGravity(runtime.history[usSide][searchedQuiets[i].from][searchedQuiets[i].to],
                                        malus, MAX_HISTORY);
                    if (ctx.contHistEntry != nullptr) {
                        applyHistoryGravity(
                            ctx.contHistEntry[contHistIndex(searchedQuiets[i].pieceType, searchedQuiets[i].to)],
                            malus, MAX_HISTORY);
                    }
                }

                // Capture history malus: penalize captures searched before the cutoff move.
                const int capMalusEnd = wasCapture ? numSearchedCaptures - 1 : numSearchedCaptures;
                for (int i = 0; i < capMalusEnd; ++i) {
                    applyHistoryGravity(runtime.captureHistory[usSide][searchedCaptures[i].to][searchedCaptures[i].victimType][0],
                                        malus, MAX_CAPTURE_HISTORY);
                    applyHistoryGravity(runtime.captureHistory[usSide][searchedCaptures[i].to][searchedCaptures[i].victimType][1],
                                        malus, MAX_CAPTURE_HISTORY);
                }
            }
            break;
        }
    }

    if (!searchedAnyMove && runtime.isInterrupted()) {
        return SearchMoveResult{bestMove, ctx.staticEval};
    }

    return SearchMoveResult{bestMove, best};
}

int32_t Searcher::searchPosition(
    chess::Board& b,
    SearchRuntime& runtime,
    int depth,
    int32_t alpha,
    int32_t beta,
    int ply,
    bool allowTTWrite,
    bool allowHeuristicUpdates,
    const chess::Move* previousMove,
    uint64_t* nodeCounter,
    bool allowNullMove,
    chess::Move excludedMove) noexcept {

    const bool hasExcludedMove = chess::isValidSquare(excludedMove.from);

    uint64_t* counter = (nodeCounter != nullptr) ? nodeCounter : &runtime.nodesSearched;
    int32_t earlyScore = 0;
    if (enterNode(b, runtime, ply, counter, earlyScore)) return earlyScore;

    const bool isPVNode = (static_cast<int64_t>(beta) - static_cast<int64_t>(alpha) > 1);

    if (ply > 0) {
        alpha = std::max(alpha, NEG_INF + ply);
        beta  = std::min(beta,  POS_INF - ply);
        if (alpha >= beta) return alpha;
    }

    // Actual draw terminals must be recognized before qsearch. Otherwise a
    // horizon node that completes a repetition is scored by the static eval
    // instead of as a draw.
    int32_t drawScore = 0;
    if (checkDrawTerminalConditions(b, drawScore)) {
        return drawScore;
    }

    if (depth <= 0) {
        return quiescenceSearch(b, runtime, alpha, beta, ply, counter, allowTTWrite);
    }

    // TB WDL probe (in-search): only return Draw as exact. Returning Win or
    // Loss anywhere — even guarded by alpha/beta — collapses move ordering:
    // every TB-winning subtree at root produces the same TB_WIN_SCORE-ply
    // score because aspiration windows around small eval-derived scores trip
    // the alpha/beta cutoff at every TB-Loss child node, and PVS scout then
    // never finds a "better" move than moves[0]. The king-wanders-instead-of-
    // pushing-pawns behaviour in KPPvK comes from exactly this.
    //
    // Skipping Win/Loss here costs us TB-based pruning inside the search
    // (the search will work harder in TB-known subtrees), but the static
    // evaluator can now differentiate between equally-winning moves, which
    // is what we need when Pyrrhic ranks every guaranteed win identically.
    // The root TB probe still handles the decisive Win/Loss choice.
    const uint64_t hashKey = b.getHash();
    int32_t score = 0;
    const bool canUseTT = (runtime.transpositionTable != nullptr);

    // Single TT read per node: this snapshot feeds the depth-gated cutoff
    // below, the static-eval tightening, the singular-extension gate and the
    // hash move for ordering (previously four separate bucket reads).
    TT::ProbeResult tte{};
    if (canUseTT) tte = runtime.transpositionTable->probeEntry(hashKey);

    if (!hasExcludedMove && tte.hit
        && static_cast<int>(tte.depth) >= std::min(depth, static_cast<int>(TT::Entry::MAX_DEPTH))
        && ttBoundCutoff(tte.flag, tte.score, alpha, beta)) {
        return scoreFromTT(tte.score, ply); // re-base mate scores to this node's ply
    }

    // TB probe sits AFTER the TT cutoff: in TB range most nodes cut on the
    // (cheap, cached) TT entry — often the TB draw stored below — without
    // paying the mmap'd table lookup.
    if (runtime.syzygyProber != nullptr
        && runtime.syzygyProber->isLoaded()
        && depth >= runtime.syzygyProber->probeDepth
        && runtime.syzygyProber->inTBRange(b)) {
        if (const auto wdl = runtime.syzygyProber->probeWDL(b)) {
            const int32_t tbScore = syzygy::SyzygyProber::wdlToScore(*wdl, ply);
            if (tbScore == 0) {
                // Draw: exact terminal — prevents picking a drawn move when
                // a winning one exists (and vice versa).
                if (canUseTT) {
                    runtime.transpositionTable->store(
                        hashKey, static_cast<uint8_t>(depth),
                        tbScore, TT::Entry::EXACT);
                }
                return tbScore;
            }
            // Win/Loss: deliberately fall through (no cutoff, no bound store).
        }
    }

    SearchNodeState node{};
    node.activeColor = b.getActiveColor();
    // One attack scan answers inCheck, double check and (in movegen) the
    // evasion mask — the bitboard is reused at move generation below.
    const uint64_t checkers = b.checkersTo(node.activeColor);
    node.inCheck = (checkers != 0ULL);
    node.isPVNode = isPVNode;

    // Compute static eval at every ply (including the root) so that the
    // `improving` heuristic can compare evalStack[ply-2] vs current eval
    // starting from ply >= 2. Previously the root left evalStack[0] at its
    // zero-initialised value, biasing `improving` to (staticEval > 0) at ply 2.
    if (!node.inCheck) {
        node.staticEval = Evaluator::evaluate(b);
        if (tte.hit) {
            const int32_t ttStaticScore = scoreFromTT(tte.score, ply); // re-base mate scores
            // Negamax (STM-relative): a LOWERBOUND tightens upward, an
            // UPPERBOUND tightens downward, EXACT always replaces.
            if (tte.flag == TT::Entry::EXACT
                || (tte.flag == TT::Entry::LOWERBOUND && ttStaticScore > node.staticEval)
                || (tte.flag == TT::Entry::UPPERBOUND && ttStaticScore < node.staticEval)) {
                node.staticEval = ttStaticScore;
            }
        }
        // Nudge the static eval by the learned correction-history signal.
        node.staticEval = std::clamp(node.staticEval + runtime.corrHist.correction(b),
                                     -MATE_BOUND + 1, MATE_BOUND - 1);
    }

    // Per-thread (Lazy SMP): a shared evalStack would race and corrupt the
    // `improving` prune. Each ancestor writes evalStack[ply] (line below)
    // before its grandchild 2 plies down reads it on the same thread's stack,
    // so a thread_local array (no per-search reset needed) is correct.
    static thread_local int32_t evalStack[MAX_PLY] = {};

    // Store staticEval in ply stack and compute improving flag.
    // In-check nodes have no meaningful static eval, so store a sentinel
    // instead of the stale default 0, which would otherwise corrupt the
    // improving comparison two plies deeper.
    evalStack[ply] = node.inCheck ? NEG_INF : node.staticEval;
    // Negamax: staticEval is side-to-move relative and ply-2 is the same
    // side, so "improving" is simply eval rising versus two plies ago.
    const bool improving = !node.inCheck && ply >= 2
        && evalStack[ply - 2] != NEG_INF
        && (node.staticEval > evalStack[ply - 2]);

    const int side = chess::Board::colorToIndex(node.activeColor);
    const int nonPawnMajors = std::popcount(
        b.knights_bb[side] | b.bishops_bb[side] |
        b.rooks_bb[side]   | b.queens_bb[side]);
    int singularExtension = 0;
    if (!hasExcludedMove && !node.inCheck && depth >= SE_MIN_DEPTH && ply > 0) {
        if (tte.hit
            && static_cast<int>(tte.depth) >= depth - SE_DEPTH_MARGIN
            && (tte.flag == TT::Entry::LOWERBOUND || tte.flag == TT::Entry::EXACT)
            && std::abs(tte.score) < MATE_BOUND
            && tte.move != 0) {

            const auto seExcluded = TT::Entry::decodeMove(tte.move);

            const int32_t ttSeScore = scoreFromTT(tte.score, ply);
            const int32_t seBeta = ttSeScore - SE_BETA_MARGIN * depth;

            // allowTTWrite stays on for the probe's DESCENDANTS (normal
            // positions the main search revisits right after); only this
            // node's own store is suppressed, via hasExcludedMove.
            const int32_t seScore = searchPosition(
                b, runtime, depth / 2 - 1, seBeta - 1, seBeta, ply,
                allowTTWrite, allowHeuristicUpdates,
                previousMove, counter, false, seExcluded);

            if (seScore < seBeta - SE_DOUBLE_MARGIN) {
                singularExtension = 2;
            } else if (seScore < seBeta) {
                singularExtension = 1;
            } else if (seScore >= beta) {
                return seScore; // multi-cut: every move beats beta, prune
            }
        }
    }

    // Interior non-PV node past the root: shared gate for NMP, reverse futility and probcut.
    const bool interiorNonPv = !node.isPVNode && !node.inCheck && ply > 0;

    // Negamax NMP gate: allow a null move when the static eval is within
    // ~100cp of beta (giving the opponent a free move likely still fails high).
    const int32_t nmpEvalGate = node.staticEval + 100;
    const bool canNullMove = allowNullMove && interiorNonPv
        && depth >= 4 && nonPawnMajors >= 2 && isBetaCutoff(nmpEvalGate, beta);

    if (canNullMove
        && tryNullMovePruning(b, node, runtime, depth, alpha, beta, ply,
                              allowTTWrite, allowHeuristicUpdates,
                              counter, score)) {
        return score;
    }

    if (interiorNonPv && depth <= 3
        && tryReverseFutilityPruning(node, depth, beta, score)) {
        return score;
    }

    // Probcut (negamax): if a winning capture, searched shallow with a null
    // window above beta+margin, still beats beta+margin, this node likely
    // fails high. SEE is side-to-move relative so the filter is one-sided.
    if (interiorNonPv && depth >= PROBCUT_MIN_DEPTH
        && std::abs(beta) < POS_INF - 1000) {
        const int32_t probcutBound = saturatingAdd32(beta, PROBCUT_MARGIN);
        const int32_t pcAlpha = probcutBound - 1;
        const int32_t pcBeta  = probcutBound;
        MoveList captures = engine::MoveGenerator::generateTacticalMoves(b);
        for (int i = 0; i < captures.size; ++i) {
            const auto& mc = captures[i];
            const int32_t see = Sorter::staticExchangeEvaluation(b, mc);
            if (see < PROBCUT_MARGIN) continue;
            chess::Board::MoveState pcState;
            b.doMove(mc, pcState);
            // Negamax child: negate result and swap/negate the scout window.
            const int32_t pcScore = -searchPosition(b, runtime, depth - 4, -pcBeta, -pcAlpha,
                ply + 1, allowTTWrite, false, &mc, counter, false);
            b.undoMove(mc, pcState);
            if (pcScore >= probcutBound) return beta;
        }
    }

    int16_t* contHistEntry = nullptr;
    if (previousMove != nullptr && previousMove->to < 64) {
        const int prevPiece = b.get(previousMove->to) & chess::Board::MASK_PIECE_TYPE;
        contHistEntry = &runtime.contHist[side ^ 1][prevPiece][previousMove->to][0][0];
    }

    SearchContext ctx{
        previousMove, counter, contHistEntry,
        depth, ply, node.staticEval, singularExtension, excludedMove,
        node.activeColor, node.inCheck, node.isPVNode, false, improving
    };

    MoveList moves = node.inCheck
        ? engine::MoveGenerator::generateLegalEvasions(b, checkers)
        : engine::MoveGenerator::generateLegalMoves(b, /*knownNotInCheck=*/true);
    if (moves.is_empty()) {
        return node.inCheck
            ? (NEG_INF + ply)  // side to move is checkmated (negamax)
            : 0;               // stalemate
    }

    bool hasHashMove = false;

    MovePicker movePicker = Sorter::sortLegalMoves(
        std::move(moves),
        ply,
        b,
        runtime,
        tte.move,
        ctx.previousMove,
        &hasHashMove,
        ctx.contHistEntry);

    if (!hasHashMove) {
        ctx.singularExtension = 0;  // SE requires a verified TT hash move
        if (depth >= 6 && ply > 0) ctx.iirActive = true;
    }

    SearchMoveResult result = searchMoves(
        b, movePicker, ctx, alpha, beta, runtime, allowHeuristicUpdates, allowTTWrite);
    const int32_t best = result.score;

    if (runtime.isInterrupted()) {
        return Evaluator::evaluate(b);
    }

    // Correction history: learn the (search - corrected static eval) residual,
    // but only from TRUSTWORTHY nodes — deep enough, quiet best move, non-mate, and
    // the search must genuinely contradict the static eval rather than merely confirm
    // a cutoff bound.
    const bool corrLearn = (best > node.staticEval)
                        || (best < node.staticEval && best < beta);
    if (corrLearn && !node.inCheck && !hasExcludedMove && depth >= 3
        && std::abs(best) < MATE_BOUND && chess::isValidSquare(result.move.from)
        && (b.get(result.move.to) & chess::Board::MASK_PIECE_TYPE) == chess::Board::EMPTY) {
        runtime.corrHist.update(b, best, node.staticEval, depth);
    }

    // hasExcludedMove suppresses only THIS node's store (its score reflects a
    // reduced move set under the same key); descendants store normally.
    if (canUseTT && allowTTWrite && !hasExcludedMove) {
        runtime.transpositionTable->store(
            hashKey, static_cast<uint8_t>(ctx.depth),
            scoreToTT(best, ctx.ply),
            static_cast<uint8_t>(determineFlag(best, alpha, beta)),
            TT::Entry::encodeMove(result.move));
    }

    return best;
}

int32_t Searcher::quiescenceSearch(
    chess::Board& b,
    SearchRuntime& runtime,
    int32_t alpha,
    int32_t beta,
    int ply,
    uint64_t* nodeCounter,
    bool allowTTWrite) noexcept {
    uint64_t* counter = (nodeCounter != nullptr) ? nodeCounter : &runtime.nodesSearched;
    int32_t earlyScore = 0;
    if (enterNode(b, runtime, ply, counter, earlyScore)) return earlyScore;

    int32_t drawScore = 0;
    if (checkDrawTerminalConditions(b, drawScore)) {
        return drawScore;
    }

    const bool canUseTT = (runtime.transpositionTable != nullptr);
    if (canUseTT) {
        const auto tte = runtime.transpositionTable->probeEntry(b.getHash());
        // Any depth qualifies at a quiescence node (stored qsearch depth is 0).
        if (tte.hit && ttBoundCutoff(tte.flag, tte.score, alpha, beta)) {
            return scoreFromTT(tte.score, ply);
        }
    }

    const uint8_t activeColor = b.getActiveColor();
    const bool usIsWhite = (activeColor == chess::Board::WHITE);
    const uint64_t checkers = b.checkersTo(activeColor);
    const bool inCheck = (checkers != 0ULL);

    if (ply >= MAX_QSEARCH_DEPTH) {
        if (inCheck) {
            MoveList evasions = engine::MoveGenerator::generateLegalEvasions(b, checkers);
            if (evasions.is_empty()) {
                return NEG_INF + ply; // side to move is checkmated (negamax)
            }
        }
        return Evaluator::evaluate(b);
    }

    MovePicker movePicker;
    int32_t best;
    const int32_t alphaOrig = alpha;

    if (inCheck) {
        movePicker = engine::MoveGenerator::generateQSearchEvasions(b, checkers);
        if (!movePicker.hasNext()) {
            return NEG_INF + ply; // side to move is checkmated (negamax)
        }
        best = NEG_INF;
    } else {
        const int32_t standPat = Evaluator::evaluate(b) + runtime.corrHist.correction(b);
        if (isBetaCutoff(standPat, beta)) {
            // Bound-only store (bestMove 0 preserves any stored move): sibling
            // qsearch nodes can then cut on this stand-pat without re-evaluating.
            if (canUseTT && allowTTWrite) {
                runtime.transpositionTable->store(
                    b.getHash(), 0, scoreToTT(standPat, ply), TT::Entry::LOWERBOUND);
            }
            return standPat; // fail-soft: tighter bound than a flat beta
        }
        updateBound(standPat, alpha);

        const int32_t EARLY_DELTA_MARGIN = QUEEN_VALUE + 50;
        if (shouldDeltaPrune(standPat, EARLY_DELTA_MARGIN, alpha)) {
            return alpha; // negamax delta-prune fail-low
        }

        int32_t deltaMargin = QUEEN_VALUE;
        const int side = chess::Board::colorToIndex(activeColor);
        const uint64_t ourPawns = b.pawns_bb[side];
        const uint64_t nearPromoMask = usIsWhite ? WHITE_NEAR_PROMO_PAWNS : BLACK_NEAR_PROMO_PAWNS;
        if (ourPawns & nearPromoMask) deltaMargin += QSEARCH_PAWN_PROMO_DELTA;

        // standPat is already side-to-move relative (negamax eval).
        if (standPat < QSEARCH_STANDPAT_BAD) {
            deltaMargin += QSEARCH_STANDPAT_BAD_DELTA;
        } else if (standPat < QSEARCH_STANDPAT_WORSE) {
            deltaMargin += QSEARCH_STANDPAT_WORSE_DELTA;
        }

        const int qsearchDepth = std::max(0, ply - runtime.depth);
        if (qsearchDepth > QSEARCH_DEPTH_REDUCTION_THRESHOLD) {
            deltaMargin -= QSEARCH_DEPTH_REDUCTION_PER_5 * ((qsearchDepth - QSEARCH_DEPTH_REDUCTION_THRESHOLD) / 5);
            deltaMargin = std::max(deltaMargin, QSEARCH_DELTAMARGIN_MIN);
        }

        if (shouldDeltaPrune(standPat, deltaMargin, alpha)) {
            return alpha; // negamax delta-prune fail-low
        }

        movePicker = engine::MoveGenerator::generateQSearchTacticalMoves(
            b, standPat, alpha, ply);
        if (!movePicker.hasNext()) {
            return standPat;
        }
        best = standPat;
    }

    while (movePicker.hasNext()) {
        const auto m = movePicker.nextMove();

        if (runtime.shouldAbort()) {
            runtime.markInterrupted();
            return Evaluator::evaluate(b);
        }

        chess::Board::MoveState state;
        b.doMove(m, state);
        // Prefetch the child's TT bucket (see the same hint in searchMoves).
        if (canUseTT) {
            runtime.transpositionTable->prefetch(b.getHash());
        }
        // Negamax: child is opponent to move -> negate + swap/negate window.
        const int32_t score = -quiescenceSearch(b, runtime, -beta, -alpha, ply + 1, counter, allowTTWrite);
        b.undoMove(m, state);

        if (isBetter(score, best)) {
            best = score;
        }

        updateBound(score, alpha);
        if (isBetaCutoff(score, beta)) {
            break; // fail-soft cutoff; the shared TT store below records `best`
        }
    }

    if (!inCheck && canUseTT && allowTTWrite) {
        runtime.transpositionTable->store(
            b.getHash(), 0, scoreToTT(best, ply),
            static_cast<uint8_t>(determineFlag(best, alphaOrig, beta)));
    }

    return best;
}

chess::Move Searcher::getBestMove(
    chess::Board& rootBoard,
    const MoveList& moves,
    SearchRuntime& runtime,
    int32_t alpha,
    int32_t beta) noexcept {
    int32_t bestScore = NEG_INF;
    auto bestMove = moves[0];
    uint64_t localNodes = 0;
    bool searchedAnyMove = false;

    uint16_t rootHashMove = 0;
    if (runtime.transpositionTable != nullptr) {
        rootHashMove = runtime.transpositionTable->probeEntry(rootBoard.getHash()).move;
    }
    MovePicker orderedRootMoves = Sorter::sortLegalMoves(
        moves,
        0,
        rootBoard,
        runtime,
        rootHashMove);

    // Root iterations index the list directly, so resolve the full order upfront.
    orderedRootMoves.fullSort();

    const MoveList& rootMoves = orderedRootMoves.moves;

    const auto scoreMove = [&](const chess::Move& mv, int32_t a, int32_t b) {
        return searchRootMoveScore(rootBoard, mv, runtime, a, b, true, true, &localNodes);
    };

    // Plain PVS root loop: full window for the first move, null-window scout
    // plus research for the rest. Parallelism lives in searchBestMove (Lazy
    // SMP helper threads run their own iterative deepening over a shared TT).
    for (int i = 0; i < rootMoves.size; ++i) {
        if (runtime.shouldAbort()) {
            runtime.markInterrupted();
            break;
        }

        const auto& m = rootMoves[i];
        const bool isFirst = (i == 0);

        int32_t score;
        if (isFirst) {
            score = scoreMove(m, alpha, beta);
        } else {
            score = scoreMove(m, alpha, saturatingAdd32(alpha, 1));
            if (shouldResearchPVS(score, alpha)) score = scoreMove(m, alpha, beta);
        }

        searchedAnyMove = true;
        if (!isFirst && runtime.isInterrupted()) {
            break;
        }

        updateMinMax(score, alpha, bestScore, bestMove, m);
        if (!isFirst && isBetaCutoff(bestScore, beta)) break;
    }

    runtime.nodesSearched += localNodes;
    if (searchedAnyMove) runtime.eval = bestScore;
    return bestMove;
}

namespace {

// Reconstruct a principal variation by following best moves stored in the TT.
// Each candidate is validated against the legal move list (a TT key collision
// could otherwise yield an illegal move), and the walk stops on a repetition so
// a cyclic PV cannot loop forever. The board is left exactly as it was found.
std::string buildPvFromTT(chess::Board& board, const TT* tt, int maxLen) noexcept {
    if (tt == nullptr) return {};

    std::array<chess::Move, MAX_PLY> pvMoves{};
    std::array<chess::Board::MoveState, MAX_PLY> states{};
    int applied = 0;
    std::string pv;

    for (int i = 0; i < maxLen && applied < MAX_PLY; ++i) {
        const auto mv = tt->probeDecodedMove(board.getHash());
        if (!chess::isValidSquare(mv.from)) break;

        const MoveList legal = engine::MoveGenerator::generateLegalMoves(board);
        bool legalMove = false;
        for (int k = 0; k < legal.size; ++k) {
            if (legal[k] == mv) { legalMove = true; break; }
        }
        if (!legalMove) break;

        if (!pv.empty()) pv += ' ';
        pv += mv.toUCIString();

        pvMoves[applied] = mv;
        board.doMove(mv, states[applied]);
        ++applied;

        if (board.countRepetitions() >= 3) break;
    }

    for (int i = applied - 1; i >= 0; --i) {
        board.undoMove(pvMoves[i], states[i]);
    }
    return pv;
}

// Emit one UCI `info` line for a completed iteration: depth, score (cp or
// mate N), node/time/nps counters and a TT-reconstructed PV.
void emitUciInfoLine(int depth, int32_t score, const SearchRuntime& runtime,
                     chess::Board& rootBoard) noexcept {
    const int64_t timeMs = (runtime.timeManager != nullptr)
        ? runtime.timeManager->elapsedMs() : 0;
    const uint64_t nodes = runtime.nodesSearched;
    const int64_t nps = (timeMs > 0)
        ? static_cast<int64_t>(static_cast<long double>(nodes) * 1000.0L / static_cast<long double>(timeMs))
        : 0;

    std::cout << "info depth " << depth << " score ";
    const int32_t absScore = (score < 0) ? -score : score;
    if (absScore >= MATE_BOUND) {
        const int32_t plies = Searcher::POS_INF - absScore;
        const int32_t mateMoves = (plies + 1) / 2;
        std::cout << "mate " << (score > 0 ? mateMoves : -mateMoves);
    } else {
        std::cout << "cp " << score;
    }
    std::cout << " nodes " << nodes << " nps " << nps << " time " << timeMs;

    const std::string pv = buildPvFromTT(rootBoard, runtime.transpositionTable, 16);
    if (!pv.empty()) std::cout << " pv " << pv;
    std::cout << '\n';
    std::cout.flush();
}

} // namespace

Searcher::IterativeSearchResult Searcher::runIterativeDeepening(
    chess::Board& rootBoard,
    SearchRuntime& runtime,
    int startDepth,
    int targetDepth) noexcept {

    IterativeSearchResult result;
    const int firstDepth = std::max(1, startDepth);
    const int maxDepth = std::max(firstDepth, targetDepth);

    if (rootBoard.kings_bb[0] == 0 || rootBoard.kings_bb[1] == 0) {
        const bool rootWhite = (rootBoard.getActiveColor() == chess::Board::WHITE);
        const bool ourKingGone = (rootBoard.kings_bb[0] == 0) ? rootWhite : !rootWhite;
        result.terminalRoot = true;
        result.completedAnyDepth = true;
        result.bestScore = ourKingGone ? NEG_INF : POS_INF; // STM-relative
        runtime.eval = result.bestScore;
        return result;
    }

    int32_t rootDrawScore = 0;
    if (checkDrawTerminalConditions(rootBoard, rootDrawScore, /*atRoot=*/true)) {
        MoveList drawMoves = engine::MoveGenerator::generateLegalMoves(rootBoard);
        result.terminalRoot = true;
        result.completedAnyDepth = true;
        result.bestScore = rootDrawScore;
        result.bestMove = drawMoves.is_empty() ? chess::Move{} : drawMoves[0];
        runtime.eval = rootDrawScore;
        return result;
    }

    MoveList moves = engine::MoveGenerator::generateLegalMoves(rootBoard);
    if (moves.is_empty()) {
        const uint8_t toMove = rootBoard.getActiveColor();
        if (rootBoard.isCheckmate(toMove)) {
            result.bestScore = NEG_INF; // side to move is mated (negamax/STM)
        } else if (rootBoard.isDraw(toMove)) {
            result.bestScore = 0;
        } else {
            result.bestScore = Evaluator::evaluate(rootBoard);
        }
        runtime.eval = result.bestScore;
        return result;
    }

    result.hasLegalMoves = true;

    // TB root probe: probeRoot returns moves with rank derived from actual
    // DTZ — higher rank is faster win / slower loss / preserved draw. The
    // optimal play in any TB-known endgame is just to follow that ranking,
    // so return immediately for all three outcomes without running a search.
    // Searching adds nothing here: TB already proves the result and orders
    // moves by exact distance-to-zeroing, while the heuristic evaluator can
    // (and does) prefer a slower converting move over the optimal one.
    //
    // Sanity check: only play the TB move if it appears in HydraY's own
    // legal-move list. If somehow it doesn't (encoding mismatch, etc.),
    // skip the TB branch and let the normal search take over.
    // Helpers skip the whole block: tb_probe_root is main-thread-only in
    // Fathom, and a helper printing the info line below would interleave
    // with the main thread's output and corrupt the UCI stream (seen live:
    // a mangled `bestmove` hung lichess-bot in a tablebase-won endgame).
    if (!runtime.isHelper
        && runtime.syzygyProber != nullptr && runtime.syzygyProber->isLoaded()
        && runtime.syzygyProber->inTBRange(rootBoard)) {
        const auto tbMoves = runtime.syzygyProber->probeRoot(rootBoard);
        if (!tbMoves.empty()) {
            int32_t bestRank = tbMoves[0].tbRank;
            auto tbBest = tbMoves[0].move;
            for (const auto& tm : tbMoves) {
                if (tm.tbRank > bestRank) { bestRank = tm.tbRank; tbBest = tm.move; }
            }
            bool tbBestIsLegal = false;
            for (int i = 0; i < moves.size; ++i) {
                if (moves[i] == tbBest) { tbBestIsLegal = true; break; }
            }
            if (tbBestIsLegal) {
                const int32_t tbScore = syzygy::SyzygyProber::wdlToScore(
                    bestRank > 0 ? syzygy::WDL::Win  :
                    bestRank < 0 ? syzygy::WDL::Loss :
                                   syzygy::WDL::Draw, 0);
                result.completedAnyDepth = true;
                result.bestMove  = tbBest;
                result.bestScore = tbScore;
                runtime.eval     = tbScore;
                if (runtime.emitUciInfo) {
                    std::cout << "info depth 1 score cp " << tbScore
                              << " tbrank " << bestRank
                              << " pv " << tbBest.toUCIString() << "\n";
                    std::cout.flush();
                }
                return result;
            }
            // Fall through to search if the TB move isn't legal in our move list.
        }
    }

    auto bestMove = moves[0];
    int32_t prevPrevScore = 0;
    int32_t prevScore = 0;
    bool hasPrevScore = false;
    bool hasPrevPrevScore = false;

    for (int currentDepth = firstDepth; currentDepth <= maxDepth; ++currentDepth) {
        if (runtime.shouldAbort()) {
            break;
        }

        // Soft limit: once a move is in hand, do not open a depth we almost
        // certainly cannot finish within the time budget.
        if (result.completedAnyDepth && runtime.timeManager != nullptr
            && !runtime.timeManager->shouldStartNextDepth()) {
            break;
        }

        runtime.depth = currentDepth;

        if (result.completedAnyDepth) {
            for (int i = 0; i < moves.size; ++i) {
                if (moves[i] == bestMove) {
                    moves.rotate(i);
                    break;
                }
            }
        }

        runtime.clearInterrupted();
        bool iterationCompleted = true;
        int32_t iterationAlpha = NEG_INF;
        int32_t iterationBeta = POS_INF;
        auto candidateBestMove = moves[0];

        const bool canUseAspiration =
            hasPrevScore
            && result.completedAnyDepth
            && currentDepth >= 5
            && std::abs(prevScore) < MATE_BOUND;

        if (!canUseAspiration) {
            candidateBestMove = getBestMove(rootBoard, moves, runtime);
            if (runtime.isInterrupted()) {
                iterationCompleted = false;
            }
        } else {
            const int32_t swingBase = hasPrevPrevScore ? prevPrevScore : prevScore;
            const int64_t scoreDiff64 = static_cast<int64_t>(prevScore) - static_cast<int64_t>(swingBase);
            const int64_t scoreSwing64 = (scoreDiff64 >= 0) ? scoreDiff64 : -scoreDiff64;
            const int32_t scoreSwing = std::min<int64_t>(scoreSwing64, POS_INF);
            int32_t windowDelta = std::clamp<int32_t>(15 + (scoreSwing / 4), 25, 100);
            int aspirationResearches = 0;
            int32_t centerScore = prevScore;
            int32_t aspAlpha = saturatingSub32(centerScore, windowDelta);
            int32_t aspBeta = saturatingAdd32(centerScore, windowDelta);

            while (true) {
                iterationAlpha = aspAlpha;
                iterationBeta = aspBeta;
                candidateBestMove = getBestMove(rootBoard, moves, runtime, aspAlpha, aspBeta);
                if (runtime.isInterrupted()) {
                    iterationCompleted = false;
                    break;
                }

                const int32_t score = runtime.eval;
                const bool failLow = (score <= aspAlpha);
                const bool failHigh = (score >= aspBeta);
                if (!failLow && !failHigh) {
                    break;
                }

                ++aspirationResearches;
                if (failLow) {
                    centerScore = std::min(centerScore, score);
                } else {
                    centerScore = std::max(centerScore, score);
                }

                windowDelta = std::min<int32_t>(WINDOW_HARD_CAP, windowDelta * 2 + 10);
                if (aspirationResearches >= MAX_ASP_RESEARCHES || windowDelta >= WINDOW_HARD_CAP) {
                    iterationAlpha = NEG_INF;
                    iterationBeta = POS_INF;
                    candidateBestMove = getBestMove(rootBoard, moves, runtime);
                    if (runtime.isInterrupted()) {
                        iterationCompleted = false;
                    }
                    break;
                }

                if (failLow) {
                    aspAlpha = std::max<int32_t>(NEG_INF, saturatingSub32(centerScore, windowDelta));
                    aspBeta = std::min<int32_t>(POS_INF, saturatingAdd32(centerScore, std::max<int32_t>(25, windowDelta / 2)));
                } else {
                    aspAlpha = std::max<int32_t>(NEG_INF, saturatingSub32(centerScore, std::max<int32_t>(25, windowDelta / 2)));
                    aspBeta = std::min<int32_t>(POS_INF, saturatingAdd32(centerScore, windowDelta));
                }
            }
        }

        if (!iterationCompleted) {
            break;
        }

        const bool hadPrevScore = hasPrevScore;
        const int32_t prevScoreBefore = prevScore;
        const bool bestMoveChanged =
            result.completedAnyDepth && !(bestMove == candidateBestMove);

        if (hasPrevScore) {
            prevPrevScore = prevScore;
            hasPrevPrevScore = true;
        }

        bestMove = candidateBestMove;
        prevScore = runtime.eval;
        hasPrevScore = true;
        result.completedAnyDepth = true;

        if (runtime.timeManager != nullptr) {
            runtime.timeManager->updateStability(
                bestMoveChanged, runtime.eval, prevScoreBefore, hadPrevScore);
        }
        result.completedDepth = currentDepth;
        result.bestMove = bestMove;
        result.bestScore = runtime.eval;
        result.rootScoreBound = determineFlag(result.bestScore, iterationAlpha, iterationBeta);
        if (runtime.transpositionTable != nullptr) {
            runtime.transpositionTable->store(
                rootBoard.getHash(), currentDepth, scoreToTT(runtime.eval, 0),
                static_cast<uint8_t>(result.rootScoreBound), TT::Entry::encodeMove(bestMove));
        }

        if (runtime.emitUciInfo) {
            emitUciInfoLine(currentDepth, runtime.eval, runtime, rootBoard);
        }
    }

    return result;
}

} // namespace engine
