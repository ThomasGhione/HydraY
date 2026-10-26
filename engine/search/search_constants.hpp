#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// Search and move-ordering parameters. The plain (non-constexpr) globals are
// UCI spins for the tuner (kSpinOptions in uci/uci.cpp).

namespace engine {

// --- Core sizes ---
inline constexpr int32_t  MAX_PLY                = 64;
inline constexpr int32_t  CAPTURE_HISTORY_SLOTS  = 2;
inline constexpr int DEFAULT_DEPTH               = 11;

// A mate n plies away scores ±(MATE_VALUE - n): every score fits the TT's
// int16_t. Scores at or beyond MATE_BOUND carry a ply distance and are rebased
// on TT store and load.
inline constexpr int32_t MATE_VALUE = 32000;
inline constexpr int32_t MATE_BOUND = MATE_VALUE - MAX_PLY;

// Tablebase wins sit between every eval and every mate; like mates they carry a
// ply distance, so everything at or beyond TB_WIN_BOUND is decisive.
inline constexpr int32_t TB_WIN_SCORE = MATE_BOUND - 500;
inline constexpr int32_t TB_WIN_BOUND = TB_WIN_SCORE - MAX_PLY;

// Window sentinels, one past any real score; NEG_INF == -POS_INF.
inline constexpr int32_t POS_INF = MATE_VALUE + 1;
inline constexpr int32_t NEG_INF = -POS_INF;

// scoreToTT pushes a mate score away from zero by up to MAX_PLY before storing.
static_assert(MATE_VALUE + MAX_PLY <= std::numeric_limits<int16_t>::max(),
              "rebased mate scores must fit the int16_t TT score field");

// --- Pruning and extensions ---
// Plain globals, not constexpr, so the tuner can drive them as UCI spins; the
// UCI layer stops any search before writing one. The derived tables
// (FUTILITY_MARGINS, LMP_THRESHOLDS, the LMR table in searcher.cpp) are built
// at startup and rebuilt by rebuildSearchDerivedTables() after a write.
// Null-move reduction: NMP_BASE_R + depth / 3, deeper the further eval beats beta.
inline int32_t NMP_BASE_R = 4;
inline int32_t NMP_EVAL_DIV = 265;
inline int32_t NMP_EVAL_MAX = 4;
// Reverse futility pruning margin per remaining ply.
inline int32_t RFP_MARGIN_PER_DEPTH = 70;
// FUTILITY_MARGINS[depth] = FUTILITY_MID_STEP * depth, depth 1..FUTILITY_MAX_DEPTH.
inline constexpr int FUTILITY_DEPTH_CAP = 10;
inline int32_t FUTILITY_MAX_DEPTH = 6;
inline int32_t FUTILITY_MID_STEP = 176;
inline int32_t FUTILITY_MARGINS[FUTILITY_DEPTH_CAP + 1] = {};
// LMP_THRESHOLDS[improving][depth], depth 1..LMP_MAX_DEPTH: the base curve
// scaled by LMP_SCALE_PCT[improving]. Higher = more permissive.
inline constexpr int LMP_DEPTH_CAP = 10;
constexpr int lmpBaseThreshold(int improving, int depth) noexcept {
    return improving ? depth * depth + 7 * depth + 8 : (depth + 2) * (depth + 3);
}
inline int32_t LMP_MAX_DEPTH = 4;
inline int32_t LMP_SCALE_PCT[2] = {50, 50};
inline int LMP_THRESHOLDS[2][LMP_DEPTH_CAP + 1] = {};
// History-based quiet pruning: skip quiet moves with very negative history.
// Indexed by depth (0..3); depth 0 unused.
inline int32_t HISTORY_PRUNE_THRESHOLD[4] = {0, -4096, -6144, -8192};

// SEE capture pruning: skip captures with SEE < -SEE_CAPTURE_MARGIN * depth,
// depth 1..SEE_PRUNE_MAX_DEPTH.
inline int32_t SEE_CAPTURE_MARGIN = 70;
inline int32_t SEE_PRUNE_MAX_DEPTH = 3;
// SEE quiet pruning: skip quiets with SEE < -QUIET_SEE_MARGIN * depth^2,
// depth 1..QUIET_SEE_MAX_DEPTH (0 = off).
inline int32_t QUIET_SEE_MARGIN = 12;
inline int32_t QUIET_SEE_MAX_DEPTH = 0;
// Singular extension.
inline constexpr int SE_MIN_DEPTH     = 6;
inline constexpr int SE_DEPTH_MARGIN  = 3;
inline int32_t SE_BETA_MARGIN   = 3;  // seBeta = ttScore - margin*depth
// ProbCut.
inline int32_t PROBCUT_MARGIN    = 149;
inline constexpr int32_t PROBCUT_MIN_DEPTH = 3;

// Rebuilds the derived tables from the current generator values. Never during
// a live search.
void rebuildSearchDerivedTables() noexcept;

// --- Late move reductions ---
// LMR_C == LMR_C_PERCENT / 100.0 (int spin for the tuner).
inline int32_t LMR_C_PERCENT = 300;
// Extra plies of reduction at an expected cut node.
inline int32_t LMR_CUT_BONUS = 1;
// First reduced move index: LMR_MIN_INDEX_ALL at an expected all node
// (non-PV, not cut), 4 elsewhere.
inline int32_t LMR_MIN_INDEX_ALL = 4;
// 1: below index 4 only quiets that are neither killer nor countermove.
inline int32_t LMR_EARLY_QUIET_ONLY = 0;
// Reduction change for a quiet by its butterfly history sign: -LMR_GOOD_HIST
// above zero, +LMR_BAD_HIST otherwise.
inline int32_t LMR_GOOD_HIST = 1;
inline int32_t LMR_BAD_HIST = 1;
// 0: butterfly sign only; 1: LMR_BAD_HIST - LMR_GOOD_HIST * n, n = positive
// signals among butterfly and continuation history; 2: -LMR_GOOD_HIST if any.
inline int32_t LMR_HIST_MODE = 1;
// Quiet, continuation and capture history bonus: (depth + 1)^2 * HIST_BONUS_PCT / 100.
inline int32_t HIST_BONUS_PCT = 100;
// 0: no hash move adds an LMR ply; 1: at PV and cut nodes it takes a ply of depth.
inline int32_t IIR_MODE = 1;
inline constexpr int LMR_MAX_DEPTH = 20; // deeper nodes use the last row

// --- History bounds ---
inline constexpr int32_t MAX_HISTORY         = 16384;
inline constexpr int32_t MAX_CAPTURE_HISTORY = 10000;

// --- Correction history ---
// Weights, scale and gates follow Stockfish 18's search.cpp (see corrhist.hpp).
inline constexpr int     CORR_HIST_BITS       = 14;
inline constexpr int     CORR_HIST_SIZE       = 1 << CORR_HIST_BITS;
inline constexpr int32_t CORR_HIST_LIMIT      = 1024;
inline constexpr int32_t CORR_BONUS_CAP       = CORR_HIST_LIMIT / 4;
inline constexpr int32_t CORR_HIST_SCALE      = 131072;
inline constexpr int32_t CORR_WEIGHT_PAWN     = 10347;
inline constexpr int32_t CORR_WEIGHT_MINOR    = 8821;
inline constexpr int32_t CORR_WEIGHT_NON_PAWN = 11665;
// Global strength of the applied correction, in percent (UCI spin).
inline int32_t CORR_HIST_WEIGHT_PCT = 100;

// --- Quiescence search ---
inline constexpr uint8_t MAX_QSEARCH_DEPTH = 48;

// Material (PIECE_VALUES, SEE) and the NNUE eval are different units: the eval
// runs hotter, so a material amount compared with an eval converts first.
inline constexpr int32_t MATERIAL_TO_EVAL_PCT = 260;

// Piece values by Board::piece_id type, for SEE, MVV ordering and the qsearch
// delta margins: search bookkeeping, not evaluation.
inline constexpr int32_t PIECE_VALUES[8] = {0, 100, 344, 359, 502, 960, 20'000, 0};

constexpr int32_t materialToEval(int32_t material, int32_t pct = MATERIAL_TO_EVAL_PCT) noexcept {
    return material * pct / 100;
}

// The ratio grows as material comes off (a rook measures 2.2x in the middlegame,
// 3.4x in the endgame). The qsearch delta margins pick it by non-pawn material.
inline constexpr int32_t MATERIAL_TO_EVAL_PCT_LATE_MG = 300;  // 3000 < npm <= 5000
inline constexpr int32_t MATERIAL_TO_EVAL_PCT_ENDGAME = 320;  // npm <= 3000

// --- Aspiration window ---
inline constexpr int32_t WINDOW_HARD_CAP    = 1500;
inline constexpr int     MAX_ASP_RESEARCHES = 6;

// --- Draw scoring ---
inline constexpr int32_t REPETITION_CONTEMPT = 80; // ~0.8 pawn

// --- Lazy SMP ---
// Hard cap on helper threads.
inline constexpr int MAX_HELPER_THREADS = 63;

// --- Continuation history layout ---
// contHist[side][pieceType][toSq] of the previous move holds a block indexed by
// the current move's (pieceType, toSq). Both piece types belong to a piece that
// moved, never EMPTY, so they are stored as 1..6 minus one; a 0 would index out
// of bounds (the debug assert in searchPosition guards the one board read).
inline constexpr int CONT_HIST_PIECE_TYPES   = 6;
inline constexpr int CONT_HIST_PIECE_STRIDE  = 64;
inline constexpr int contHistIndex(int pieceType, int toSq) noexcept {
    return (pieceType - 1) * CONT_HIST_PIECE_STRIDE + toSq;
}

// --- Move ordering scores ---
inline constexpr int32_t HASH_MOVE_SCORE      = 100000;
inline constexpr int32_t CAPTURE_BASE_SCORE   = 10000;
inline constexpr int32_t KILLER_1_SCORE       = 9000;
inline constexpr int32_t KILLER_2_SCORE       = 8500;
inline constexpr int32_t COUNTER_MOVE_SCORE   = 8200;
inline constexpr int32_t PROMOTION_BASE_SCORE = 7000;
inline constexpr int32_t HISTORY_SCORE_MAX    = 7500;
inline constexpr int32_t HISTORY_SCORE_MIN    = -2000;
// Qsearch tactical move scoring.
inline constexpr int32_t TACTICAL_PROMOTION_SCORE = 9000;
inline constexpr int32_t FUTILITY_MARGIN          = 100;
inline constexpr int32_t MOVE_DELTA_MARGIN        = 140;
inline constexpr int32_t SEE_THRESHOLD_SHALLOW    = -24; // ply < 10
inline constexpr int32_t SEE_THRESHOLD_MID        = -12; // 10 <= ply < 20
inline constexpr int32_t SEE_THRESHOLD_DEEP       = -4;  // ply >= 20

} // namespace engine
