#pragma once

// Dual-perspective NNUE accumulator: v[0] = white's view, v[1] = black's view,
// and the same for the threat inputs in threat[], which the forward adds.
//
// Every feature of perspective p depends on the bucket and flip of p's own
// king, its basis (base = 768 * bucket, flip); a threat feature on the flip
// alone. Board keeps one accumulator per
// position along the current line, a frame: doMove pushes a frame holding the
// move's feature changes, undoMove pops it, and the values are computed only
// when something reads them, from the nearest computed ancestor under the
// same basis or from the Finny table (Board::ensureNnueAccumulatorClean).

#include <cstdint>
#include <cstring>

#include "network.hpp"
#include "network_deep.hpp"

namespace NNUE {

// One feature change of a move: a piece added, removed or moved.
struct AccDelta {
    enum Kind : uint8_t { Add, Remove, Move };
    uint8_t kind;
    uint8_t piece;
    uint8_t from;   // Add/Remove: the square
    uint8_t to;     // Move only
};

// A threat that appears or vanishes with a move, in Board's pieces and squares.
struct ThreatDelta {
    uint8_t attacker;
    uint8_t from;
    uint8_t victim;
    uint8_t to;
    bool    add;
};

// A move changes at most four features: a promotion capture.
inline constexpr int ACC_MAX_DELTAS = 4;
// Threats change by the dozen. A move that changes more leaves its frame's
// list at ACC_THREATS_OVERFLOW, and the frames after it start from scratch.
inline constexpr int ACC_MAX_THREATS = 64;
inline constexpr uint8_t ACC_THREATS_OVERFLOW = 255;
// Frames in Board's ring: a power of two, at least MAX_PLY (a search line
// longer than the ring would overwrite its own ancestors).
inline constexpr int ACC_FRAMES = 128;
// A frame is computed from an ancestor at most this many moves back; past
// that the Finny refresh is the cheaper start.
inline constexpr int ACC_MAX_WALK = 8;

struct alignas(64) Accumulator {
    int16_t v[2][HIDDEN];
    int16_t threat[2][HIDDEN];
    int32_t base[2];   // 768 * king bucket, per perspective
    uint8_t flip[2];   // 0 or 7 (file mirror), per perspective
    bool    dirty[2];  // true = v[p] and threat[p] not computed for this position yet
    // The move from the previous frame, in the same cache line as the flags.
    uint8_t  ndelta;
    AccDelta delta[ACC_MAX_DELTAS];
    uint8_t     nthreat;
    ThreatDelta threatDelta[ACC_MAX_THREATS];

    // Both perspectives in one traversal of the rows, writing d0/d1 from
    // s0/s1 (another frame) when From, in place otherwise (s0/s1 unused).
    // int16 is modular, so the order the contributions are summed in does not
    // change the result. The accumulator rows never alias the weight rows:
    // restrict-qualified parameters say so, or GCC versions the loop on
    // runtime alias checks.
    template<int NS, int NA, bool From = false>
    static inline void fusePair(int16_t* __restrict d0, int16_t* __restrict d1,
                                const int16_t* __restrict s0, const int16_t* __restrict s1,
                                const int16_t* const (&sub)[2][2],
                                const int16_t* const (&add)[2][2]) noexcept {
        for (int i = 0; i < HIDDEN; ++i) {
            int x0 = From ? s0[i] : d0[i];
            int x1 = From ? s1[i] : d1[i];
            for (int k = 0; k < NS; ++k) { x0 -= sub[0][k][i]; x1 -= sub[1][k][i]; }
            for (int k = 0; k < NA; ++k) { x0 += add[0][k][i]; x1 += add[1][k][i]; }
            d0[i] = static_cast<int16_t>(x0);
            d1[i] = static_cast<int16_t>(x1);
        }
    }

    // Weight row for one piece feature under an explicit basis: the live
    // perspectives use their own, the Finny cache rows hold a different one.
    [[nodiscard]] static inline const int16_t* rowFor(int rowBase, int rowFlip, int p,
                                                      uint8_t piece, uint8_t index) noexcept {
        const int type = (piece & 0x7) - 1;      // P..K -> 0..5
        const bool black = (piece & 0x8) == 0;   // Board::WHITE = 0x8
        const int lerf = index ^ 56;
        const int sqView = (p == 1) ? (lerf ^ 56) : lerf;
        const bool isOpp = (p == 1) ? !black : black;
        const int feat768 = (isOpp ? 384 : 0) + type * 64 + sqView;
        return activeNetwork->l0w[rowBase + (feat768 ^ rowFlip)];
    }

    [[nodiscard]] static inline const int8_t* threatRow(int feature) noexcept {
        return activeNetwork->l0t.get() + static_cast<size_t>(feature) * HIDDEN;
    }

    // Single-row counterpart of fusePair.
    template<int NS, int NA, bool From = false>
    static inline void fuseRows(int16_t* __restrict row, const int16_t* __restrict src,
                                const int16_t* const* sub, const int16_t* const* add) noexcept {
        for (int i = 0; i < HIDDEN; ++i) {
            int x = From ? src[i] : row[i];
            for (int k = 0; k < NS; ++k) x -= sub[k][i];
            for (int k = 0; k < NA; ++k) x += add[k][i];
            row[i] = static_cast<int16_t>(x);
        }
    }

    // Walks a diff of ns subs and na adds in passes of at most two of each,
    // calling f.template operator()<S, A>(si, ai) on each pass. The counts are
    // template parameters so the kernels unroll: runtime bounds measured 50%
    // slower than replaying the deltas one at a time.
    template<class F>
    static inline void forEachPass(int ns, int na, F&& f) noexcept {
        for (int si = 0, ai = 0; si < ns || ai < na; ) {
            const int s = (ns - si > 2) ? 2 : ns - si;
            const int a = (na - ai > 2) ? 2 : na - ai;
            switch (s * 4 + a) {
                case 2 * 4 + 2: f.template operator()<2, 2>(si, ai); break;
                case 2 * 4 + 1: f.template operator()<2, 1>(si, ai); break;
                case 1 * 4 + 2: f.template operator()<1, 2>(si, ai); break;
                case 1 * 4 + 1: f.template operator()<1, 1>(si, ai); break;
                case 2 * 4 + 0: f.template operator()<2, 0>(si, ai); break;
                case 0 * 4 + 2: f.template operator()<0, 2>(si, ai); break;
                case 1 * 4 + 0: f.template operator()<1, 0>(si, ai); break;
                default:        f.template operator()<0, 1>(si, ai); break;
            }
            si += s;
            ai += a;
        }
    }

    // row = src (zero when null) - sub + add, over threat rows of i8 weights.
    // Any number of rows costs one pass over the destination: each tile of it
    // stays in registers while every row is added in.
    static inline void applyThreatRows(int16_t* __restrict row, const int16_t* __restrict src,
                                       const int8_t* const* sub, int ns,
                                       const int8_t* const* add, int na) noexcept {
        constexpr int TILE = 64;
        static_assert(HIDDEN % TILE == 0);
        for (int b = 0; b < HIDDEN; b += TILE) {
            int16_t x[TILE];
            for (int i = 0; i < TILE; ++i) x[i] = src != nullptr ? src[b + i] : 0;
            for (int k = 0; k < ns; ++k)
                for (int i = 0; i < TILE; ++i) x[i] = static_cast<int16_t>(x[i] - sub[k][b + i]);
            for (int k = 0; k < na; ++k)
                for (int i = 0; i < TILE; ++i) x[i] = static_cast<int16_t>(x[i] + add[k][b + i]);
            for (int i = 0; i < TILE; ++i) row[b + i] = x[i];
        }
    }

    // Applies a whole feature diff to one row, the first pass reading src (a
    // different row) when it is given. A refresh changes 5.8 features on
    // average, so this reads and writes the row about twice instead of six times.
    static inline void applyRows(int16_t* __restrict row,
                                 const int16_t* const* sub, int ns,
                                 const int16_t* const* add, int na,
                                 const int16_t* src = nullptr) noexcept {
        if (src != nullptr && ns == 0 && na == 0) {
            std::memcpy(row, src, HIDDEN * sizeof(int16_t));
            return;
        }
        forEachPass(ns, na, [&]<int S, int A>(int si, int ai) {
            if (src != nullptr) {
                fuseRows<S, A, true>(row, src, sub + si, add + ai);
                src = nullptr;
            } else {
                fuseRows<S, A>(row, nullptr, sub + si, add + ai);
            }
        });
    }
};

// Finny table: one cached accumulator row per (perspective, king bucket, flip)
// with the piece bitboards it was built from, so a refresh is a diff against
// the cache instead of a rebuild. Any entry of the right basis is a valid
// start, even from another game: the table is per thread and never needs
// invalidation (the initial one is the empty board).
struct FinnyEntry {
    alignas(64) int16_t v[HIDDEN];
    uint64_t bb[2][6]; // [colorIndex][type-1], engine bit convention
};

struct FinnyTable {
    FinnyEntry entry[2][INPUT_BUCKETS][2]; // [perspective][bucket][flip?1:0]
    // Cache rows are only valid diff bases for the network they were built
    // with: re-initialise whenever the active network identity changes
    // (EvalFile swap at runtime).
    const Deep::NetworkDeep* builtFor = nullptr;

    inline void ensureInitialised() noexcept {
        if (builtFor == activeNetwork) [[likely]] return;
        const Deep::NetworkDeep& net = *activeNetwork;
        for (auto& perPersp : entry)
            for (auto& perBucket : perPersp)
                for (auto& e : perBucket) {
                    std::memcpy(e.v, net.l0b, sizeof(e.v));
                    std::memset(e.bb, 0, sizeof(e.bb));
                }
        builtFor = activeNetwork;
    }
};

} // namespace NNUE
