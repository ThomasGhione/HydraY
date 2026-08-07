#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// All tunable search + move-ordering parameters, centralized here. Most are
// compile-time constants; the plain (non-constexpr) globals are exposed as
// UCI spins for the tuning campaigns (see kSpinOptions in uci/uci.cpp).
//
// Purely structural constants that are tied to a specific local data structure
// (SEE cache size, history flat-array cell counts, tracked-quiet buffer sizes,
// int16 clamp bounds) deliberately stay next to their declarations.

namespace engine {

// ===================================================
// CORE SEARCH SIZING (array dimensions, depth defaults)
// ===================================================
inline constexpr int32_t  MAX_PLY                = 64;
inline constexpr int32_t  CAPTURE_HISTORY_SLOTS  = 2;
inline constexpr int32_t  CORR_HISTORY_SIZE      = 1 << 14;
inline constexpr int DEFAULT_DEPTH               = 11;

// Score scale. A forced mate `n` plies away scores ±(MATE_VALUE - n), so every
// score the search can produce fits in int16_t — the precondition for packing a
// static eval next to the score in the TT payload. Scores at or beyond
// MATE_BOUND carry a ply distance and are rebased on TT store/load (see
// scoreToTT / scoreFromTT).
inline constexpr int32_t MATE_VALUE = 32000;
inline constexpr int32_t MATE_BOUND = MATE_VALUE - MAX_PLY;

// Window sentinel: one past the best reachable score, so no real score can ever
// equal it. Negamax-safe (NEG_INF == -POS_INF, negating it is well defined).
inline constexpr int32_t POS_INF = MATE_VALUE + 1;
inline constexpr int32_t NEG_INF = -POS_INF;

// scoreToTT pushes a mate score away from zero by up to MAX_PLY before storing.
static_assert(MATE_VALUE + MAX_PLY <= std::numeric_limits<int16_t>::max(),
              "rebased mate scores must fit the int16_t TT score field");

// ===================================================
// PRUNING / EXTENSION PARAMETERS
// ===================================================
// The margins below are plain globals (not constexpr) so the SMAC3 campaign
// (#9) can drive them as UCI spins, same pattern as eval_constants.hpp. The
// UCI layer stops any search before writing them. Derived tables
// (FUTILITY_MARGINS, LMP_THRESHOLDS, the LMR table in searcher.cpp) are
// rebuilt by rebuildSearchDerivedTables(); the defaults reproduce the frozen
// June values exactly.
inline constexpr int     NULL_MOVE_VERIFICATION_DEPTH = 10;
// Null-move reduction eval scaling: deeper reduction the further eval beats beta.
inline int32_t NMP_EVAL_DIV = 265;
inline int32_t NMP_EVAL_MAX = 4;
// Reverse futility pruning margin per remaining ply.
inline int32_t RFP_MARGIN_PER_DEPTH = 70;
// Futility margin generator: FUTILITY_MARGINS[depth] = MID_STEP*d, consumed
// by the move loop (gated to depth 1..6). The HCE-era endgame row (phase split
// on nonPawnMajors) was removed — see HCE_RESIDUE_AUDIT.md section 1b.
// NOTE: rebuildSearchDerivedTables() only runs on a UCI option change, so this
// row is what a plain build actually uses — keep it == MID_STEP * d by hand.
inline int32_t FUTILITY_MID_STEP = 176;
inline int32_t FUTILITY_MARGINS[7] = {0, 176, 352, 528, 704, 880, 1056};
// LMP_THRESHOLDS[improving][depth]: higher = more permissive.
// Gated to depth 1..4. Derived: base table scaled by LMP_SCALE_PCT[improving].
inline constexpr int LMP_BASE_THRESHOLDS[2][5] = {
    {0, 12, 20, 30, 42},
    {0, 16, 26, 38, 52},
};
inline int32_t LMP_SCALE_PCT[2] = {100, 100};
inline int LMP_THRESHOLDS[2][5] = {
    {0, 12, 20, 30, 42},
    {0, 16, 26, 38, 52},
};
// History-based quiet pruning: skip quiet moves with very negative history.
// Indexed by depth (0..3); depth 0 unused.
inline int32_t HISTORY_PRUNE_THRESHOLD[4] = {0, -4096, -6144, -8192};

// SEE capture pruning: skip captures with SEE < -SEE_CAPTURE_MARGIN * depth.
inline int32_t SEE_CAPTURE_MARGIN = 70;
// Singular extension.
inline constexpr int SE_MIN_DEPTH     = 6;
inline constexpr int SE_DEPTH_MARGIN  = 3;
inline int32_t SE_BETA_MARGIN   = 3;  // seBeta = ttScore - margin*depth
inline int32_t SE_DOUBLE_MARGIN = 16; // double-extend when seScore < seBeta - 16
// ProbCut.
inline int32_t PROBCUT_MARGIN    = 149;
inline constexpr int32_t PROBCUT_MIN_DEPTH = 3;

// Rebuilds the derived tables above plus the LMR reduction table
// (searcher.cpp) from the current generator values. Called by the UCI layer
// after writing a generator option; never call during a live search.
void rebuildSearchDerivedTables() noexcept;

// ===================================================
// LMR (late move reductions) table parameters
// ===================================================
// LMR_C == LMR_C_PERCENT / 100.0 (int spin for the tuner).
inline int32_t LMR_C_PERCENT = 300;
inline constexpr int    LMR_MAX_DEPTH = 20;  // engine never exceeds depth 14 in practice
inline constexpr int    LMR_MAX_MOVES = 218; // theoretical maximum legal moves (== MAX_MOVES, movelist.hpp)

// ===================================================
// HISTORY HEURISTIC BOUNDS
// ===================================================
inline constexpr int32_t MAX_HISTORY         = 16384;
inline constexpr int32_t MAX_CAPTURE_HISTORY = 10000;

// ===================================================
// CORRECTION HISTORY (search - static eval residual)
// ===================================================
inline constexpr int32_t CORR_HIST_LIMIT   = 1024; // bound on the smoothed residual (cp)
inline constexpr int32_t CORR_HIST_DIVISOR = 4;    // applied fraction of each residual
inline constexpr int32_t CORR_HIST_BLEND   = 256;  // weighted-average denominator
inline constexpr int32_t CORR_HIST_MAX_W   = 16;   // per-update weight cap (grows with depth)
// Cap on the SUM of the pawn/minor/major corrections, kept at the old pawn-only 256 cp.
inline constexpr int32_t CORR_TOTAL_CAP    = CORR_HIST_LIMIT / CORR_HIST_DIVISOR;

// ===================================================
// QUIESCENCE SEARCH
// ===================================================
inline constexpr uint8_t MAX_QSEARCH_DEPTH = 48;
// Delta pruning: a qsearch node fails low when no capture can lift the stand-pat
// past alpha even allowing for the heaviest piece plus a cushion. One margin,
// deliberately: a second position-scaled one (near-promotion pawns, stand-pat
// thresholds, depth taper) sat behind this and fired 2 times in 4.6M nodes,
// because each of its widenings pushed the margin above this value.
inline constexpr int32_t QSEARCH_DELTA_MARGIN = 1010; // == QUEEN_VALUE + 50

// ===================================================
// ASPIRATION WINDOW
// ===================================================
inline constexpr int32_t WINDOW_HARD_CAP    = 1500;
inline constexpr int     MAX_ASP_RESEARCHES = 6;

// ===================================================
// DRAW / CONTEMPT SCORING
// ===================================================
inline constexpr int32_t REPETITION_CONTEMPT = 80; // ~0.8 pawn

// ===================================================
// PARALLELISM (Lazy SMP)
// ===================================================
// Hard cap on helper threads (Threads option is already clamped to hwmax).
inline constexpr int MAX_HELPER_THREADS = 63;

// ===================================================
// CONTINUATION HISTORY layout
// ===================================================
// contHist is keyed by the previous move's (side, pieceType, toSq); each context
// holds a [PIECE_TYPES][64] PieceTo block indexed by the CURRENT move's
// (pieceType, toSq). Piece types are 0..6 (EMPTY..KING), so the block is 7*64.
inline constexpr int CONT_HIST_PIECE_TYPES   = 7;
inline constexpr int CONT_HIST_PIECE_STRIDE  = 64;
inline constexpr int contHistIndex(int pieceType, int toSq) noexcept {
    return pieceType * CONT_HIST_PIECE_STRIDE + toSq;
}

// ===================================================
// MOVE ORDERING (sorter) — score buckets
// ===================================================
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
