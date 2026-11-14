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

// The 116 feature pairs (29 type pairs in 4 colour combinations) get one entry
// per attacker square; every other pair points to an extra empty block, so a
// lookup needs no branch.
inline constexpr int VALID_PAIR_COUNT = 116;
inline constexpr uint16_t INVALID_PAIR_BASE = VALID_PAIR_COUNT * 64;

// The victim squares of a pair and attacker square, and the first feature.
struct Entry {
    uint64_t mask;
    uint32_t first;
};

struct Table {
    std::array<Entry, (VALID_PAIR_COUNT + 1) * 64> entries{};
    std::array<std::array<uint16_t, 12>, 12> pairBase{};
    int32_t total = 0;
};

inline constexpr Table TABLE = [] {
    Table t{};
    int pairIdx = 0;
    for (int a = 0; a < 12; ++a)
        for (int v = 0; v < 12; ++v) {
            if (!pairIsFeature(a, v)) {
                t.pairBase[a][v] = INVALID_PAIR_BASE;
                continue;
            }
            const uint16_t base = static_cast<uint16_t>(pairIdx * 64);
            t.pairBase[a][v] = base;
            for (int from = 0; from < 64; ++from) {
                const uint64_t mask = targets(a, v, from);
                t.entries[base + from] = {mask, static_cast<uint32_t>(t.total)};
                t.total += std::popcount(mask);
            }
            ++pairIdx;
        }
    return t;
}();
static_assert(TABLE.total == THREAT_FEATURES, "threats.rs counts 70,050");

// Bit v of entry a: Board types a and v (1-6) can form a feature.
inline constexpr std::array<uint8_t, 8> PAIR_FEATURE_MASK = [] {
    std::array<uint8_t, 8> m{};
    for (int ta = 0; ta < 6; ++ta)
        for (int tv = 0; tv < 6; ++tv)
            if (pairIsFeature(ta, tv)) m[ta + 1] |= static_cast<uint8_t>(1U << (tv + 1));
    return m;
}();

constexpr bool tryIndexBase(uint32_t entryIdx, int to, uint32_t& out) noexcept {
    const Entry& e = TABLE.entries[entryIdx];
    const uint64_t bit = 1ULL << to;
    if (!(e.mask & bit)) return false;
    out = e.first + static_cast<uint32_t>(std::popcount(e.mask & (bit - 1)));
    return true;
}

constexpr int indexBase(uint32_t entryIdx, int to) noexcept {
    uint32_t out = 0;
    return tryIndexBase(entryIdx, to, out) ? static_cast<int>(out) : -1;
}

constexpr bool tryIndex(int a, int from, int v, int to, uint32_t& out) noexcept {
    return tryIndexBase(TABLE.pairBase[a][v] + from, to, out);
}

} // namespace threats

// Whether attacker and victim types (Board's 1-6) can form a feature at all.
[[nodiscard]] constexpr bool threatPairIsFeature(uint8_t attacker, uint8_t victim) noexcept {
    return (threats::PAIR_FEATURE_MASK[attacker & 0x7] >> (victim & 0x7)) & 1;
}

// [perspective][Board piece] -> relative piece 0-11.
inline constexpr std::array<std::array<uint8_t, 16>, 2> REL_PIECE = [] {
    std::array<std::array<uint8_t, 16>, 2> t{};
    for (int p = 0; p < 2; ++p)
        for (int pc = 0; pc < 16; ++pc)
            if ((pc & 0x7) >= 1 && (pc & 0x7) <= 6)
                t[p][pc] = static_cast<uint8_t>((((pc & 0x8) == 0) != (p == 1)) * 6 + (pc & 0x7) - 1);
    return t;
}();

// Both perspectives' pair bases by Board attacker and victim, perspective 1 in
// the high half.
inline constexpr std::array<std::array<uint32_t, 16>, 16> BOTH_PAIR_BASE = [] {
    std::array<std::array<uint32_t, 16>, 16> b{};
    for (int a = 0; a < 16; ++a)
        for (int v = 0; v < 16; ++v) {
            if ((a & 0x7) < 1 || (a & 0x7) > 6 || (v & 0x7) < 1 || (v & 0x7) > 6) {
                b[a][v] = threats::INVALID_PAIR_BASE | (static_cast<uint32_t>(threats::INVALID_PAIR_BASE) << 16);
                continue;
            }
            const uint32_t b0 = threats::TABLE.pairBase[REL_PIECE[0][a]][REL_PIECE[0][v]];
            const uint32_t b1 = threats::TABLE.pairBase[REL_PIECE[1][a]][REL_PIECE[1][v]];
            b[a][v] = b0 | (b1 << 16);
        }
    return b;
}();

// Perspective p (0 = white) whose king puts the file mirror `flip`, on Board
// pieces and squares (a8 = 0).
struct ThreatPerspective {
    int p;
    int toView;
    explicit constexpr ThreatPerspective(int perspective, int flip) noexcept
        : p(perspective), toView((perspective == 0 ? 56 : 0) ^ flip) {}

    [[nodiscard]] constexpr int relPiece(uint8_t piece) const noexcept {
        return REL_PIECE[p][piece & 0xf];
    }
    [[nodiscard]] constexpr bool tryFeature(int relA, int fromView, uint8_t victim, uint8_t to, uint32_t& out) const noexcept {
        return threats::tryIndex(relA, fromView, relPiece(victim), to ^ toView, out);
    }
    [[nodiscard]] constexpr bool tryFeature(uint8_t attacker, uint8_t from, uint8_t victim, uint8_t to, uint32_t& out) const noexcept {
        return threats::tryIndex(relPiece(attacker), from ^ toView, relPiece(victim), to ^ toView, out);
    }
};

// Both perspectives' features of one threat, -1 where it is none.
inline void threatFeaturePair(int toView0, int flip1, uint8_t attacker, uint8_t from,
                              uint8_t victim, uint8_t to, int& out0, int& out1) noexcept {
    const uint32_t bases = BOTH_PAIR_BASE[attacker & 0xf][victim & 0xf];
    out0 = threats::indexBase((bases & 0xffffu) + (from ^ toView0), to ^ toView0);
    out1 = threats::indexBase((bases >> 16)     + (from ^ flip1),   to ^ flip1);
}

} // namespace NNUE
