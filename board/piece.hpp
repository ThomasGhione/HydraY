#pragma once

#include <array>
#include <bit>
#include <cstdint>

#include "coords.hpp"

// Sliding-piece attacks use PEXT (BMI2). Every shipped build (-march=native,
// -march=x86-64-v3) provides it; fail loudly on anything that does not.
#if !defined(__BMI2__)
#error "HydraY requires BMI2 (PEXT) for sliding-piece attacks; build with -march=native or -march=x86-64-v3."
#endif
#include <immintrin.h>

namespace pieces {

inline constexpr uint64_t ONE = 1ULL;
inline constexpr int WHITE_SIDE = 0;
inline constexpr int BLACK_SIDE = 1;
inline constexpr int sideIndex(bool isWhite) noexcept { return isWhite ? WHITE_SIDE : BLACK_SIDE; }

inline constexpr int8_t KNIGHT_OFFSET[8][2] = { {1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2} };
inline constexpr int8_t KING_OFFSET[8][2]   = { {1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1} };

// Compile-time LUT factories: fill a [64] (or [2][64]) table from a generator.
template<typename Gen>
constexpr std::array<uint64_t, 64> squareTable(Gen gen) noexcept {
    std::array<uint64_t, 64> t{};
    for (int sq = 0; sq < 64; ++sq) t[sq] = gen(sq);
    return t;
}
template<typename Gen>
constexpr std::array<std::array<uint64_t, 64>, 2> sideSquareTable(Gen gen) noexcept {
    std::array<std::array<uint64_t, 64>, 2> t{};
    for (int sq = 0; sq < 64; ++sq) {
        t[WHITE_SIDE][sq] = gen(sq, true);
        t[BLACK_SIDE][sq] = gen(sq, false);
    }
    return t;
}

// =============================================================================
// SLIDING PIECES (rook / bishop / queen) — PEXT magic bitboards
// =============================================================================

// Relevant-occupancy masks, generated at compile time: rook = its rank+file,
// bishop = its four diagonals, both minus the board edges (edge squares never
// block, so they carry no index bit). static_asserts anchor the retired table.
constexpr uint64_t rookRelevantMask(int sq) noexcept {
    const int f = sq & 7, r = sq >> 3;
    uint64_t m = 0ULL;
    for (int rr = 1; rr <= 6; ++rr) if (rr != r) m |= ONE << (rr * 8 + f);
    for (int ff = 1; ff <= 6; ++ff) if (ff != f) m |= ONE << (r * 8 + ff);
    return m;
}
constexpr uint64_t bishopRelevantMask(int sq) noexcept {
    const int f = sq & 7, r = sq >> 3;
    uint64_t m = 0ULL;
    for (int rr = r + 1, ff = f + 1; rr <= 6 && ff <= 6; ++rr, ++ff) m |= ONE << (rr * 8 + ff);
    for (int rr = r + 1, ff = f - 1; rr <= 6 && ff >= 1; ++rr, --ff) m |= ONE << (rr * 8 + ff);
    for (int rr = r - 1, ff = f + 1; rr >= 1 && ff <= 6; --rr, ++ff) m |= ONE << (rr * 8 + ff);
    for (int rr = r - 1, ff = f - 1; rr >= 1 && ff >= 1; --rr, --ff) m |= ONE << (rr * 8 + ff);
    return m;
}
inline constexpr std::array<uint64_t, 64> ROOK_MASKS   = squareTable(rookRelevantMask);
inline constexpr std::array<uint64_t, 64> BISHOP_MASKS = squareTable(bishopRelevantMask);
static_assert(ROOK_MASKS[0] == 0x000101010101017EULL && ROOK_MASKS[63] == 0x7E80808080808000ULL, "rook mask");
static_assert(BISHOP_MASKS[0] == 0x0040201008040200ULL && BISHOP_MASKS[27] == 0x0040221400142200ULL, "bishop mask");

// Table index via PEXT: with these fancy masks the index range equals
// 2^popcount(mask), exactly the range the attack tables are built against.
struct MagicParams { 
    uint64_t mask; 
    uint32_t offset; 
};

__attribute__((always_inline))
inline uint32_t sliderIndex(uint64_t occ, const MagicParams& p) noexcept {
    return static_cast<uint32_t>(_pext_u64(occ, p.mask));
}

template<typename MaskArray>
constexpr std::array<MagicParams, 64> buildMagicParams(const MaskArray& masks) noexcept {
    std::array<MagicParams, 64> table{};
    uint32_t offset = 0;
    for (int sq = 0; sq < 64; ++sq) {
        table[sq] = {masks[sq], offset};
        offset += (1u << std::popcount(masks[sq]));
    }
    return table;
}
inline constexpr std::array<MagicParams, 64> ROOK_PARAMS   = buildMagicParams(ROOK_MASKS);
inline constexpr std::array<MagicParams, 64> BISHOP_PARAMS = buildMagicParams(BISHOP_MASKS);

constexpr size_t ROOK_LOOKUP_SIZE   = 102400;  // ~800 KB
constexpr size_t BISHOP_LOOKUP_SIZE = 5248;    // ~40 KB
inline std::array<uint64_t, ROOK_LOOKUP_SIZE>   ROOK_ATTACK_LOOKUP;
inline std::array<uint64_t, BISHOP_LOOKUP_SIZE> BISHOP_ATTACK_LOOKUP;

// The i-th occupancy subset of `mask` (i in [0, 2^popcount(mask))).
constexpr uint64_t occupancySubset(int i, uint64_t mask) noexcept {
    uint64_t occ = 0ULL;
    for (int b = 0; mask; ++b) {
        const int sq = std::countr_zero(mask);
        mask &= mask - 1;
        if (i & (1 << b)) occ |= ONE << sq;
    }
    return occ;
}

// Classical ray-walk, compile-time table build only: stop at edge or blocker.
constexpr uint64_t slidingAttacks(int sq, uint64_t occ, const int8_t dirs[4][2]) noexcept {
    uint64_t attacks = 0ULL;
    const int f0 = chess::file(sq), r0 = chess::rank(sq);
    for (int d = 0; d < 4; ++d)
        for (int f = f0 + dirs[d][0], r = r0 + dirs[d][1];
             f >= 0 && f < 8 && r >= 0 && r < 8; f += dirs[d][0], r += dirs[d][1]) {
            attacks |= ONE << (r * 8 + f);
            if (occ & (ONE << (r * 8 + f))) break;
        }
    return attacks;
}
inline constexpr int8_t ROOK_DIRS[4][2]   = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};
inline constexpr int8_t BISHOP_DIRS[4][2] = {{1, -1}, {-1, -1}, {1, 1}, {-1, 1}};

// Fill one square's slice of a slider table across all its occupancy subsets.
template<size_t N>
inline void populateAttackTable(int sq, const MagicParams& p,
                                std::array<uint64_t, N>& lookup, const int8_t dirs[4][2]) noexcept {
    const int patterns = 1 << std::popcount(p.mask);
    for (int i = 0; i < patterns; ++i) {
        const uint64_t occ = occupancySubset(i, p.mask);
        lookup[p.offset + sliderIndex(occ, p)] = slidingAttacks(sq, occ, dirs);
    }
}
inline void initMagicBitboards() noexcept {
    for (int sq = 0; sq < 64; ++sq) {
        populateAttackTable(sq, ROOK_PARAMS[sq],   ROOK_ATTACK_LOOKUP,   ROOK_DIRS);
        populateAttackTable(sq, BISHOP_PARAMS[sq], BISHOP_ATTACK_LOOKUP, BISHOP_DIRS);
    }
}

// Use uint8_t sq (0..63) to avoid sign-extension corner cases.
__attribute__((hot, always_inline))
inline uint64_t getRookAttacks(uint8_t sq, uint64_t occ) noexcept {
    const MagicParams& p = ROOK_PARAMS[sq];
    return ROOK_ATTACK_LOOKUP[p.offset + sliderIndex(occ, p)];
}
__attribute__((hot, always_inline))
inline uint64_t getBishopAttacks(uint8_t sq, uint64_t occ) noexcept {
    const MagicParams& p = BISHOP_PARAMS[sq];
    return BISHOP_ATTACK_LOOKUP[p.offset + sliderIndex(occ, p)];
}
__attribute__((hot, always_inline))
inline uint64_t getQueenAttacks(uint8_t sq, uint64_t occ) noexcept {
    return getRookAttacks(sq, occ) | getBishopAttacks(sq, occ);
}

// =============================================================================
// STEP PIECES (pawn / knight / king) — precomputed attack & push tables
// =============================================================================
// Square convention: rank 0 = row 8 (top), rank 7 = row 1. White pawns move to
// a lower rank number, black to a higher one.

constexpr uint64_t pawnAttacks(int sq, bool isWhite) noexcept {
    const int f = chess::file(sq);
    const int r = chess::rank(sq) + (isWhite ? -1 : 1);
    uint64_t a = 0ULL;
    if (r >= 0 && r < 8) {
        if (f - 1 >= 0) a |= ONE << (r * 8 + f - 1);
        if (f + 1 < 8)  a |= ONE << (r * 8 + f + 1);
    }
    return a;
}
// Pawn squares (of colour isWhite) that attack `sq` — the mirror of pawnAttacks.
constexpr uint64_t pawnAttackersTo(int sq, bool isWhite) noexcept {
    const int f = chess::file(sq);
    const int r = chess::rank(sq) + (isWhite ? 1 : -1);
    uint64_t a = 0ULL;
    if (r >= 0 && r < 8) {
        if (f - 1 >= 0) a |= ONE << (r * 8 + f - 1);
        if (f + 1 < 8)  a |= ONE << (r * 8 + f + 1);
    }
    return a;
}
constexpr uint64_t stepAttacks(int sq, const int8_t offsets[8][2]) noexcept {
    const int f0 = chess::file(sq), r0 = chess::rank(sq);
    uint64_t a = 0ULL;
    for (int i = 0; i < 8; ++i) {
        const int f = f0 + offsets[i][0], r = r0 + offsets[i][1];
        if (f >= 0 && f < 8 && r >= 0 && r < 8) a |= ONE << (r * 8 + f);
    }
    return a;
}
constexpr uint64_t pawnSinglePush(int sq, bool isWhite) noexcept {
    const int f = chess::file(sq), r = chess::rank(sq);
    if (isWhite) return r > 0 ? ONE << ((r - 1) * 8 + f) : 0ULL;
    return r < 7 ? ONE << ((r + 1) * 8 + f) : 0ULL;
}
constexpr uint64_t pawnDoublePush(int sq, bool isWhite) noexcept {
    const int f = chess::file(sq), r = chess::rank(sq);
    if (isWhite) return r == 6 ? ONE << ((r - 2) * 8 + f) : 0ULL;
    return r == 1 ? ONE << ((r + 2) * 8 + f) : 0ULL;
}

inline constexpr auto PAWN_ATTACKS             = sideSquareTable(pawnAttacks);
inline constexpr auto PAWN_ATTACKERS_TO        = sideSquareTable(pawnAttackersTo);
inline constexpr auto PAWN_SINGLE_PUSH_TARGETS = sideSquareTable(pawnSinglePush);
inline constexpr auto PAWN_DOUBLE_PUSH_TARGETS = sideSquareTable(pawnDoublePush);
inline constexpr std::array<uint64_t, 64> KNIGHT_ATTACKS = squareTable([](int sq) { return stepAttacks(sq, KNIGHT_OFFSET); });
inline constexpr std::array<uint64_t, 64> KING_ATTACKS   = squareTable([](int sq) { return stepAttacks(sq, KING_OFFSET); });

// Forward-push result keyed by target occupancy: occBits bit0 = one-step
// destination blocked, bit1 = two-step blocked.
inline constexpr std::array<std::array<std::array<uint64_t, 4>, 64>, 2> PAWN_FORWARD_PUSH_LOOKUP = [] {
    std::array<std::array<std::array<uint64_t, 4>, 64>, 2> table{};
    for (int c = 0; c < 2; ++c)
        for (int sq = 0; sq < 64; ++sq) {
            const uint64_t oneStep = PAWN_SINGLE_PUSH_TARGETS[c][sq];
            const uint64_t twoStep = PAWN_DOUBLE_PUSH_TARGETS[c][sq];
            for (int occBits = 0; occBits < 4; ++occBits) {
                uint64_t result = 0ULL;
                if (oneStep && !(occBits & 0x1)) {
                    result |= oneStep;
                    if (twoStep && !(occBits & 0x2)) result |= twoStep;
                }
                table[c][sq][occBits] = result;
            }
        }
    return table;
}();

__attribute__((hot, always_inline))
inline constexpr uint64_t getPawnForwardPushes(uint8_t sq, bool isWhite, uint64_t occupancy) noexcept {
    const int side = sideIndex(isWhite);
    const unsigned occBits = ((occupancy & PAWN_SINGLE_PUSH_TARGETS[side][sq]) != 0ULL)
        | (((occupancy & PAWN_DOUBLE_PUSH_TARGETS[side][sq]) != 0ULL) << 1);
    return PAWN_FORWARD_PUSH_LOOKUP[side][sq][occBits];
}

// Piece-type move dispatch (compile-time branch on the piece-type code).
template<uint8_t PieceType>
[[nodiscard]] __attribute__((hot, always_inline))
inline constexpr uint64_t generateMovesByType(uint8_t sq, uint64_t occupancy) noexcept {
    if constexpr (PieceType == 0x2) return KNIGHT_ATTACKS[sq];               // KNIGHT
    if constexpr (PieceType == 0x3) return getBishopAttacks(sq, occupancy);  // BISHOP
    if constexpr (PieceType == 0x4) return getRookAttacks(sq, occupancy);    // ROOK
    if constexpr (PieceType == 0x5) return getQueenAttacks(sq, occupancy);   // QUEEN
    if constexpr (PieceType == 0x6) return KING_ATTACKS[sq];                 // KING
    return 0ULL;
}

} // namespace pieces
