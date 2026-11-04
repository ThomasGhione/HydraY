#pragma once

#include <array>
#include <bit>
#include <cstdint>

#include "coords.hpp"

#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace pieces {

// Pops and returns the index of the least-significant set bit.
constexpr int popLSB(uint64_t& bb) noexcept {
    const int idx = std::countr_zero(bb);
    bb &= bb - 1;
    return idx;
}

inline constexpr int8_t KNIGHT_OFFSET[8][2] = { {1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2} };
inline constexpr int8_t KING_OFFSET[8][2]   = { {1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1} };
inline constexpr int8_t ROOK_DIRS[4][2]     = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};
inline constexpr int8_t BISHOP_DIRS[4][2]   = {{1, -1}, {-1, -1}, {1, 1}, {-1, 1}};

// Compile-time LUT factories: fill a [64] or a [side][64] table (side 0 = white).
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
        t[0][sq] = gen(sq, true);
        t[1][sq] = gen(sq, false);
    }
    return t;
}

// Classical ray-walk, compile-time table build only: stop at edge or blocker.
constexpr uint64_t slidingAttacks(int sq, uint64_t occ, const int8_t dirs[4][2]) noexcept {
    uint64_t attacks = 0ULL;
    const int f0 = chess::file(sq), r0 = chess::rank(sq);
    for (int d = 0; d < 4; ++d)
        for (int f = f0 + dirs[d][0], r = r0 + dirs[d][1];
             f >= 0 && f < 8 && r >= 0 && r < 8; f += dirs[d][0], r += dirs[d][1]) {
            attacks |= 1ULL << (r * 8 + f);
            if (occ & (1ULL << (r * 8 + f))) break;
        }
    return attacks;
}

// --- Sliding pieces: PEXT attack tables ---

// A square's mask and the start of its slice: PEXT of the occupancy under the
// mask indexes the 2^popcount(mask) entries of the slice.
struct MagicParams {
    uint64_t mask;
    uint32_t offset;
};

__attribute__((always_inline))
inline uint32_t sliderIndex(uint64_t occ, const MagicParams& p) noexcept {
#if defined(__BMI2__)
    return static_cast<uint32_t>(_pext_u64(occ, p.mask));
#else
    // x86-64-v2 builds have no PEXT: same index, one mask bit at a time.
    uint32_t index = 0;
    uint32_t out = 1;
    for (uint64_t mask = p.mask; mask != 0; mask &= mask - 1, out <<= 1) {
        if ((occ & mask & -mask) != 0) index |= out;
    }
    return index;
#endif
}

// The i-th occupancy subset of `mask` (i in [0, 2^popcount(mask))).
constexpr uint64_t occupancySubset(int i, uint64_t mask) noexcept {
    uint64_t occ = 0ULL;
    for (int b = 0; mask; ++b) {
        const int sq = popLSB(mask);
        if (i & (1 << b)) occ |= 1ULL << sq;
    }
    return occ;
}

// One slider type. A square's mask is its empty-board rays without the edge
// squares off its own rank and file: nothing stands behind them, so they
// never change the attacks. The slices, and so the table, are sized at
// compile time; the table is filled once at startup.
template<bool IsRook>
struct Slider {
    static constexpr auto& DIRS = IsRook ? ROOK_DIRS : BISHOP_DIRS;
    static constexpr std::array<MagicParams, 64> PARAMS = [] {
        std::array<MagicParams, 64> params{};
        uint32_t offset = 0;
        for (int sq = 0; sq < 64; ++sq) {
            const uint64_t edges = (0xFF000000000000FFULL & ~(0xFFULL << (8 * chess::rank(sq))))
                                 | (0x8181818181818181ULL & ~(0x0101010101010101ULL << chess::file(sq)));
            params[sq] = {slidingAttacks(sq, 0ULL, DIRS) & ~edges, offset};
            offset += 1u << std::popcount(params[sq].mask);
        }
        return params;
    }();
    static constexpr size_t SIZE = PARAMS[63].offset + (size_t{1} << std::popcount(PARAMS[63].mask));
    static inline std::array<uint64_t, SIZE> lookup;

    static void init() noexcept {
        for (int sq = 0; sq < 64; ++sq)
            for (int i = 0; i < (1 << std::popcount(PARAMS[sq].mask)); ++i)
                lookup[PARAMS[sq].offset + i] = slidingAttacks(sq, occupancySubset(i, PARAMS[sq].mask), DIRS);
    }

    __attribute__((hot, always_inline))
    static uint64_t attacks(uint8_t sq, uint64_t occ) noexcept {
        const MagicParams& p = PARAMS[sq];
        return lookup[p.offset + sliderIndex(occ, p)];
    }
};
using Rook = Slider<true>;
using Bishop = Slider<false>;
static_assert(Rook::PARAMS[0].mask == 0x000101010101017EULL && Rook::PARAMS[63].mask == 0x7E80808080808000ULL
              && Bishop::PARAMS[0].mask == 0x0040201008040200ULL && Bishop::PARAMS[27].mask == 0x0040221400142200ULL,
              "relevant-occupancy masks");
static_assert(Rook::SIZE == 102400 && Bishop::SIZE == 5248, "800 KiB and 41 KiB of attack tables");

// --- Step pieces: attack tables ---
// Square convention: rank 0 = row 8 (top), rank 7 = row 1. White pawns move to
// a lower rank number, black to a higher one.

// The squares one (file, rank) offset away from sq that are on the board.
template<size_t N>
constexpr uint64_t stepAttacks(int sq, const int8_t (&offsets)[N][2]) noexcept {
    uint64_t a = 0ULL;
    for (const auto& o : offsets) {
        const int f = chess::file(sq) + o[0], r = chess::rank(sq) + o[1];
        if (f >= 0 && f < 8 && r >= 0 && r < 8) a |= 1ULL << (r * 8 + f);
    }
    return a;
}
constexpr uint64_t pawnAttacks(int sq, bool isWhite) noexcept {
    const int8_t dr = isWhite ? -1 : 1;
    return stepAttacks(sq, {{-1, dr}, {1, dr}});
}
constexpr uint64_t pawnSinglePush(int sq, bool isWhite) noexcept {
    return stepAttacks(sq, {{0, static_cast<int8_t>(isWhite ? -1 : 1)}});
}
constexpr uint64_t pawnDoublePush(int sq, bool isWhite) noexcept {
    return chess::rank(sq) == (isWhite ? 6 : 1) ? stepAttacks(sq, {{0, static_cast<int8_t>(isWhite ? -2 : 2)}}) : 0ULL;
}

// PAWN_ATTACKS[side][sq]: squares a pawn of `side` on sq attacks. The pawns of
// `side` that attack sq are the ones on PAWN_ATTACKS[side ^ 1][sq].
inline constexpr auto PAWN_ATTACKS             = sideSquareTable(pawnAttacks);
inline constexpr auto PAWN_SINGLE_PUSH_TARGETS = sideSquareTable(pawnSinglePush);
inline constexpr auto PAWN_DOUBLE_PUSH_TARGETS = sideSquareTable(pawnDoublePush);
inline constexpr std::array<uint64_t, 64> KNIGHT_ATTACKS = squareTable([](int sq) { return stepAttacks(sq, KNIGHT_OFFSET); });
inline constexpr std::array<uint64_t, 64> KING_ATTACKS   = squareTable([](int sq) { return stepAttacks(sq, KING_OFFSET); });

// Forward pushes; the double push only through an empty first square.
__attribute__((hot, always_inline))
inline constexpr uint64_t getPawnForwardPushes(uint8_t sq, bool isWhite, uint64_t occupancy) noexcept {
    const uint64_t one = PAWN_SINGLE_PUSH_TARGETS[!isWhite][sq] & ~occupancy;
    return one ? one | (PAWN_DOUBLE_PUSH_TARGETS[!isWhite][sq] & ~occupancy) : 0ULL;
}

// Piece-type move dispatch (compile-time branch on the piece-type code).
template<uint8_t PieceType>
[[nodiscard]] __attribute__((hot, always_inline))
inline constexpr uint64_t generateMovesByType(uint8_t sq, uint64_t occupancy) noexcept {
    if constexpr (PieceType == 0x2) return KNIGHT_ATTACKS[sq];                                              // KNIGHT
    if constexpr (PieceType == 0x3) return Bishop::attacks(sq, occupancy);                                  // BISHOP
    if constexpr (PieceType == 0x4) return Rook::attacks(sq, occupancy);                                    // ROOK
    if constexpr (PieceType == 0x5) return Rook::attacks(sq, occupancy) | Bishop::attacks(sq, occupancy);   // QUEEN
    if constexpr (PieceType == 0x6) return KING_ATTACKS[sq];                                                // KING
    return 0ULL;
}

} // namespace pieces
