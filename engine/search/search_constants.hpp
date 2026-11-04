#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Search and move-ordering parameters.

namespace engine {

// A tunable search constant. A tuning build registers each one as a UCI spin
// (name, range, value) where it is defined, and the UCI layer writes it, never
// during a search; a release build offers no spins and folds them as constants.
struct SpinOption { const char* name; int32_t* value; int32_t min, max; };

inline std::vector<SpinOption>& spinOptions() noexcept {
    static std::vector<SpinOption> options;
    return options;
}

template<bool Tunable>
struct SpinValue {
    int32_t value;
    constexpr SpinValue(int32_t v, [[maybe_unused]] const char* name, [[maybe_unused]] int32_t min,
                        [[maybe_unused]] int32_t max) : value(v) {
        if constexpr (Tunable) {
            if (name != nullptr) spinOptions().push_back({name, &value, min, max});
        }
    }
    constexpr operator int32_t() const noexcept { return value; }
};

#ifdef HYDRAY_RELEASE
using Spin = const SpinValue<false>;
#else
using Spin = SpinValue<true>;
#endif

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
// The LMR table in searcher.cpp is built at startup and rebuilt by
// rebuildSearchDerivedTables() after a spin write.
// Null-move reduction: NMP_BASE_R + depth / 3, deeper the further eval beats beta.
inline Spin NMP_BASE_R{4, "NMP_BASE_R", 1, 8};
inline Spin NMP_EVAL_DIV{265, "NMP_EVAL_DIV", 50, 400};
inline Spin NMP_EVAL_MAX{4, "NMP_EVAL_MAX", 0, 10};
// Reverse futility pruning margin per remaining ply.
inline Spin RFP_MARGIN_PER_DEPTH{70, "RFP_MARGIN_PER_DEPTH", 30, 250};
// Futility margin FUTILITY_MID_STEP * depth, depth 1..FUTILITY_MAX_DEPTH.
inline constexpr int FUTILITY_DEPTH_CAP = 10;
inline Spin FUTILITY_MAX_DEPTH{6, "FUTILITY_MAX_DEPTH", 0, FUTILITY_DEPTH_CAP};
inline Spin FUTILITY_MID_STEP{176, "FUTILITY_MID_STEP", 80, 600};
// LMP threshold, depth 1..LMP_MAX_DEPTH: the base curve scaled by
// LMP_SCALE_PCT[improving]. Higher = more permissive.
inline constexpr int LMP_DEPTH_CAP = 10;
inline Spin LMP_MAX_DEPTH{4, "LMP_MAX_DEPTH", 0, LMP_DEPTH_CAP};
inline Spin LMP_SCALE_PCT[2] = {{50, "LMP_SCALE_STD_PCT", 10, 250}, {50, "LMP_SCALE_IMP_PCT", 10, 250}};
inline int lmpThreshold(bool improving, int depth) noexcept {
    const int base = improving ? depth * depth + 7 * depth + 8 : (depth + 2) * (depth + 3);
    return base * LMP_SCALE_PCT[improving] / 100;
}
// History-based quiet pruning: skip quiet moves with very negative history.
// Indexed by depth (0..3); depth 0 unused.
inline Spin HISTORY_PRUNE_THRESHOLD[4] = {
    {0, nullptr, 0, 0}, {-4096, "HISTORY_PRUNE_D1", -16384, 0},
    {-6144, "HISTORY_PRUNE_D2", -16384, 0}, {-8192, "HISTORY_PRUNE_D3", -16384, 0}};

// SEE capture pruning: skip captures with SEE < -SEE_CAPTURE_MARGIN * depth,
// depth 1..SEE_PRUNE_MAX_DEPTH.
inline Spin SEE_CAPTURE_MARGIN{70, "SEE_CAPTURE_MARGIN", 20, 200};
inline Spin SEE_PRUNE_MAX_DEPTH{3, "SEE_PRUNE_MAX_DEPTH", 0, 12};
// SEE quiet pruning: skip quiets with SEE < -QUIET_SEE_MARGIN * depth^2,
// depth 1..QUIET_SEE_MAX_DEPTH (0 = off).
inline Spin QUIET_SEE_MARGIN{12, "QUIET_SEE_MARGIN", 0, 200};
inline Spin QUIET_SEE_MAX_DEPTH{0, "QUIET_SEE_MAX_DEPTH", 0, 12};
// Singular extension.
inline constexpr int SE_MIN_DEPTH     = 6;
inline constexpr int SE_DEPTH_MARGIN  = 3;
inline Spin SE_BETA_MARGIN{3, "SE_BETA_MARGIN", 0, 16};  // seBeta = ttScore - margin*depth
// ProbCut.
inline Spin PROBCUT_MARGIN{149, "PROBCUT_MARGIN", 30, 250};
inline constexpr int32_t PROBCUT_MIN_DEPTH = 3;

// Rebuilds the LMR table from LMR_C_PERCENT. Never during a live search.
void rebuildSearchDerivedTables() noexcept;

// --- Late move reductions ---
// LMR_C == LMR_C_PERCENT / 100.0 (int spin for the tuner).
inline Spin LMR_C_PERCENT{300, "LMR_C_PERCENT", 150, 600};
// Extra plies of reduction at an expected cut node.
inline Spin LMR_CUT_BONUS{1, "LMR_CUT_BONUS", 0, 4};
// First reduced move index: LMR_MIN_INDEX_ALL at an expected all node
// (non-PV, not cut), 4 elsewhere.
inline Spin LMR_MIN_INDEX_ALL{4, "LMR_MIN_INDEX_ALL", 1, 8};
// 1: below index 4 only quiets that are neither killer nor countermove.
inline Spin LMR_EARLY_QUIET_ONLY{0, "LMR_EARLY_QUIET_ONLY", 0, 1};
// Reduction change for a quiet by its butterfly history sign: -LMR_GOOD_HIST
// above zero, +LMR_BAD_HIST otherwise.
inline Spin LMR_GOOD_HIST{1, "LMR_GOOD_HIST", 0, 3};
inline Spin LMR_BAD_HIST{1, "LMR_BAD_HIST", 0, 3};
// 0: butterfly sign only; 1: LMR_BAD_HIST - LMR_GOOD_HIST * n, n = positive
// signals among butterfly and continuation history; 2: -LMR_GOOD_HIST if any.
inline Spin LMR_HIST_MODE{1, "LMR_HIST_MODE", 0, 2};
// Quiet, continuation and capture history bonus: (depth + 1)^2 * HIST_BONUS_PCT / 100.
inline Spin HIST_BONUS_PCT{100, "HIST_BONUS_PCT", 25, 1000};
// 0: no hash move adds an LMR ply; 1: at PV and cut nodes it takes a ply of depth.
inline Spin IIR_MODE{1, "IIR_MODE", 0, 1};
inline Spin IIR_MIN_DEPTH{4, "IIR_MIN_DEPTH", 2, 12};
// 1: a TT cutoff by a quiet hash move gives it the cutoff history bonus.
inline Spin TT_CUT_HIST{0, "TT_CUT_HIST", 0, 1};
// Plies less of reduction for a move that gives check (0 = off).
inline Spin LMR_CHECK_LESS{0, "LMR_CHECK_LESS", 0, 3};
// Hindsight depth: +1 after a reduction of 3+ when the opponent is not worse
// off (1 = on); -1 after a reduction of 2+ when both static evals sum above
// HINDSIGHT_MARGIN (0 = off).
inline Spin HINDSIGHT_EXT{0, "HINDSIGHT_EXT", 0, 1};
inline Spin HINDSIGHT_MARGIN{0, "HINDSIGHT_MARGIN", 0, 2000};
inline constexpr int LMR_MAX_DEPTH = 20; // deeper nodes use the last row

// --- History bounds ---
inline constexpr int32_t MAX_HISTORY         = 16384;
inline constexpr int32_t MAX_CAPTURE_HISTORY = 10000;

// --- Correction history ---
// Weights, scale and gates follow Stockfish 18's search.cpp (see corrhist.hpp).
inline constexpr int     CORR_HIST_BITS       = 14;
inline constexpr int     CORR_HIST_SIZE       = 1 << CORR_HIST_BITS;
inline constexpr int32_t CORR_HIST_LIMIT      = 1024;
inline constexpr int32_t CORR_HIST_SCALE      = 131072;
// Update bonus: residual * depth * rate / 1280 (160 = 1/8, 128 = 1/10 after a
// raised alpha), clamped to +-CORR_BONUS_CAP (UCI spins).
inline Spin CORR_BONUS_RATE{160, "CORR_BONUS_RATE", 0, 640};
inline Spin CORR_BONUS_RATE_RAISED{128, "CORR_BONUS_RATE_RAISED", 0, 640};
inline Spin CORR_BONUS_CAP{CORR_HIST_LIMIT / 4, "CORR_BONUS_CAP", 0, 512};
// Weights of each table in the applied correction (UCI spins).
inline Spin CORR_WEIGHT_PAWN{10347, "CORR_WEIGHT_PAWN", 0, 32768};
inline Spin CORR_WEIGHT_MINOR{8821, "CORR_WEIGHT_MINOR", 0, 32768};
inline Spin CORR_WEIGHT_NON_PAWN{11665, "CORR_WEIGHT_NON_PAWN", 0, 32768};
// Update bonus of the minor and non-pawn tables, in 128ths of the pawn one (UCI spins).
inline Spin CORR_UPDATE_MINOR{156, "CORR_UPDATE_MINOR", 0, 512};
inline Spin CORR_UPDATE_NON_PAWN{178, "CORR_UPDATE_NON_PAWN", 0, 512};
// Global strength of the applied correction, in percent (UCI spin).
inline Spin CORR_HIST_WEIGHT_PCT{100, "CORR_HIST_WEIGHT_PCT", 0, 400};

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
// UCI spins.
inline Spin QS_EVAL_PCT_MG{300, "QS_EVAL_PCT_MG", 100, 600};  // npm > 5000
inline Spin QS_EVAL_PCT_LATE_MG{345, "QS_EVAL_PCT_LATE_MG", 100, 600};  // 3000 < npm <= 5000
inline Spin QS_EVAL_PCT_ENDGAME{350, "QS_EVAL_PCT_ENDGAME", 100, 600};  // npm <= 3000

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
