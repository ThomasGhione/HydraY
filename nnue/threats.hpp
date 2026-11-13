#pragma once

// Threat inputs: "piece a on square s attacks piece v on square t". The
// reference is nnue/trainer/src/threats.rs, which describes the feature set;
// this file must index every feature exactly as it does.
//
// Pieces are numbered from a perspective, 0-5 its own P N B R Q K and 6-11 the
// opponent's; squares are in its view, LERF with its own pawns moving up and
// the king's file mirror applied, as for the piece features.

#include <array>
#include <bit>
#include <cstdint>

namespace NNUE {

inline constexpr int THREAT_FEATURES = 70'050;
// Every piece attacks at most 8 others, and at most 32 pieces stand on the board.
inline constexpr int MAX_ACTIVE_THREATS = 256;

namespace threats {

inline constexpr int PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5;
inline constexpr uint64_t BACK_RANKS = 0xFF000000000000FFULL;

template<size_t N>
constexpr uint64_t steps(int sq, const int (&offsets)[N][2]) noexcept {
    uint64_t out = 0;
    for (const auto& o : offsets) {
        const int f = sq % 8 + o[0], r = sq / 8 + o[1];
        if (f >= 0 && f < 8 && r >= 0 && r < 8) out |= 1ULL << (r * 8 + f);
    }
    return out;
}

constexpr uint64_t rays(int sq, const int (&dirs)[4][2]) noexcept {
    uint64_t out = 0;
    for (const auto& d : dirs)
        for (int f = sq % 8 + d[0], r = sq / 8 + d[1]; f >= 0 && f < 8 && r >= 0 && r < 8; f += d[0], r += d[1])
            out |= 1ULL << (r * 8 + f);
    return out;
}

// The squares piece (0-11) on sq attacks on an empty board. No pawn stands on a
// back rank, so none attacks from there.
constexpr uint64_t emptyBoardAttacks(int piece, int sq) noexcept {
    constexpr int ORTHOGONAL[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int DIAGONAL[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int KNIGHT_STEPS[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    constexpr int KING_STEPS[8][2] = {{1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}};
    constexpr int OWN_PAWN[2][2] = {{-1, 1}, {1, 1}};
    constexpr int THEIR_PAWN[2][2] = {{-1, -1}, {1, -1}};
    switch (piece % 6) {
        case PAWN:   return (BACK_RANKS >> sq & 1) ? 0 : steps(sq, piece < 6 ? OWN_PAWN : THEIR_PAWN);
        case KNIGHT: return steps(sq, KNIGHT_STEPS);
        case BISHOP: return rays(sq, DIAGONAL);
        case ROOK:   return rays(sq, ORTHOGONAL);
        case QUEEN:  return rays(sq, DIAGONAL) | rays(sq, ORTHOGONAL);
        default:     return steps(sq, KING_STEPS);
    }
}

inline constexpr auto EMPTY_BOARD_ATTACKS = [] {
    std::array<std::array<uint64_t, 64>, 12> table{};
    for (int piece = 0; piece < 12; ++piece)
        for (int sq = 0; sq < 64; ++sq) table[piece][sq] = emptyBoardAttacks(piece, sq);
    return table;
}();

// King on king, and the pairs where the victim always attacks back.
constexpr bool pairIsFeature(int a, int v) noexcept {
    const int ta = a % 6, tv = v % 6;
    return !(ta == KING && tv == KING) && !(ta == PAWN && (tv == BISHOP || tv == QUEEN || tv == KING))
        && !(tv == QUEEN && (ta == BISHOP || ta == ROOK || ta == KING));
}

constexpr bool mutual(int a, int v) noexcept {
    return a % 6 == v % 6 && (a % 6 != PAWN || a / 6 != v / 6);
}

// The victim squares that index the features of a on `from` against v.
constexpr uint64_t targets(int a, int v, int from) noexcept {
    uint64_t m = EMPTY_BOARD_ATTACKS[a][from];
    if (v % 6 == PAWN) m &= ~BACK_RANKS;
    if (mutual(a, v)) m &= ~((2ULL << from) - 1);
    return m;
}

// First feature of each (attacker, victim, attacker square); -1 when the pair
// is never a feature.
struct Offsets {
    std::array<std::array<std::array<int32_t, 64>, 12>, 12> first{};
    int32_t total = 0;
};
inline constexpr Offsets OFFSETS = [] {
    Offsets o;
    for (int a = 0; a < 12; ++a)
        for (int v = 0; v < 12; ++v)
            for (int from = 0; from < 64; ++from) {
                o.first[a][v][from] = pairIsFeature(a, v) ? o.total : -1;
                if (pairIsFeature(a, v)) o.total += std::popcount(targets(a, v, from));
            }
    return o;
}();
static_assert(OFFSETS.total == THREAT_FEATURES, "threats.rs counts 70,050");

constexpr int index(int a, int from, int v, int to) noexcept {
    const int32_t offset = OFFSETS.first[a][v][from];
    if (offset < 0 || (mutual(a, v) && to < from)) return -1;
    return offset + std::popcount(targets(a, v, from) & ((1ULL << to) - 1));
}

} // namespace threats

// Whether attacker and victim types (Board's 1-6) can form a feature at all.
[[nodiscard]] constexpr bool threatPairIsFeature(uint8_t attacker, uint8_t victim) noexcept {
    return threats::pairIsFeature((attacker & 0x7) - 1, (victim & 0x7) - 1);
}

// The feature of Board pieces and squares (a8 = 0) for perspective p (0 =
// white) whose king puts the file mirror `flip`; -1 when it is not a feature.
[[nodiscard]] inline int threatFeature(int p, int flip, uint8_t attacker, uint8_t from,
                                       uint8_t victim, uint8_t to) noexcept {
    const int toView = (p == 0 ? 56 : 0) ^ flip;
    const auto relative = [p](uint8_t piece) { return (((piece & 0x8) == 0) != (p == 1)) * 6 + (piece & 0x7) - 1; };
    return threats::index(relative(attacker), from ^ toView, relative(victim), to ^ toView);
}

} // namespace NNUE
