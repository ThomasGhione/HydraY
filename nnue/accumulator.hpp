#pragma once

// Dual-perspective NNUE accumulator: v[0] = white's view, v[1] = black's view,
// each the sum of the piece and the threat features in one i16 row.
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

#if defined(__AVX2__)
#include <immintrin.h>
#endif

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

// A threat that appears or vanishes with a move, in Board's pieces and squares;
// the add flag is bit 7 of toAndAdd, so a delta is one 32-bit store.
struct alignas(4) ThreatDelta {
    uint8_t attacker;
    uint8_t from;
    uint8_t victim;
    uint8_t toAndAdd;

    static constexpr ThreatDelta make(uint8_t attacker, uint8_t from, uint8_t victim,
                                      uint8_t to, bool add) noexcept {
        return {attacker, from, victim, static_cast<uint8_t>(to | (add ? 0x80 : 0))};
    }
    [[nodiscard]] constexpr uint8_t to()  const noexcept { return static_cast<uint8_t>(toAndAdd & 0x3f); }
    [[nodiscard]] constexpr bool    add() const noexcept { return (toAndAdd & 0x80) != 0; }
};
static_assert(sizeof(ThreatDelta) == 4);

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
    int32_t base[2];   // 768 * king bucket, per perspective
    uint8_t flip[2];   // 0 or 7 (file mirror), per perspective
    bool    dirty[2];  // true = v[p] not computed for this position yet
    // The move from the previous frame, in the same cache line as the flags.
    uint8_t     ndelta;
    AccDelta    delta[ACC_MAX_DELTAS];
    uint8_t     nthreat;
    ThreatDelta threatDelta[ACC_MAX_THREATS];

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

    [[nodiscard]] static inline const int8_t* threatRow(uint32_t feature) noexcept {
        return activeNetwork->l0t.get() + static_cast<size_t>(feature) * HIDDEN;
    }

    enum class RowInit { Src, InPlace };

#if defined(__AVX2__)
    template<bool Add>
    __attribute__((always_inline))
    static inline void applyWeightList(const int16_t* const* __restrict rows, int n, int b,
                                       __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3,
                                       __m256i& x4, __m256i& x5, __m256i& x6, __m256i& x7) noexcept {
        const auto loadVec = [](const int16_t* p) noexcept { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
        const auto accum = [](__m256i acc, __m256i d) noexcept {
            if constexpr (Add) return _mm256_add_epi16(acc, d);
            else               return _mm256_sub_epi16(acc, d);
        };
        int k = 0;
        for (; k + 1 < n; k += 2) {
            const int16_t* p0 = rows[k]     + b;
            const int16_t* p1 = rows[k + 1] + b;
            auto step2 = [&](__m256i& x, int off) noexcept {
                // Two rows summed first: half the dependency chain on x.
                x = accum(x, _mm256_add_epi16(loadVec(p0 + off), loadVec(p1 + off)));
            };
            step2(x0,   0); step2(x1,  16); step2(x2,  32); step2(x3,  48);
            step2(x4,  64); step2(x5,  80); step2(x6,  96); step2(x7, 112);
        }
        if (k < n) {
            const int16_t* p = rows[k] + b;
            auto step1 = [&](__m256i& x, int off) noexcept { x = accum(x, loadVec(p + off)); };
            step1(x0,   0); step1(x1,  16); step1(x2,  32); step1(x3,  48);
            step1(x4,  64); step1(x5,  80); step1(x6,  96); step1(x7, 112);
        }
    }

    // Threat rows in the layout loadFromMemory gives them: a lane of block j
    // is 256 * w[16 + i] + w[i] + 128 exactly. r sums the lanes and q their
    // high bytes, so no row is widened; applyMixedRows takes w[i] back as
    // r - 256 q - 128 (adds - subs).
    template<bool Add>
    __attribute__((always_inline))
    static inline void applyThreatList(const int8_t* const* __restrict rows, int n, int b,
                                       __m256i (&r)[4], __m256i (&q)[4]) noexcept {
        const auto accum = [](__m256i acc, __m256i d) noexcept {
            if constexpr (Add) return _mm256_add_epi16(acc, d);
            else               return _mm256_sub_epi16(acc, d);
        };
        const auto load = [](const int8_t* p) noexcept { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
        int k = 0;
        for (; k + 1 < n; k += 2)
            for (int j = 0; j < 4; ++j) {
                const __m256i w0 = load(rows[k] + b + 32 * j);
                const __m256i w1 = load(rows[k + 1] + b + 32 * j);
                r[j] = accum(r[j], _mm256_add_epi16(w0, w1));
                q[j] = accum(q[j], _mm256_add_epi16(_mm256_srai_epi16(w0, 8), _mm256_srai_epi16(w1, 8)));
            }
        if (k < n)
            for (int j = 0; j < 4; ++j) {
                const __m256i w = load(rows[k] + b + 32 * j);
                r[j] = accum(r[j], w);
                q[j] = accum(q[j], _mm256_srai_epi16(w, 8));
            }
    }
#endif

    // row = init - sub + add over rows of i16 piece weights. Any number of
    // rows costs one pass over row: each tile of it stays in registers while
    // every row is added in.
    template<RowInit Init>
    static inline void applyWeightRows(int16_t* __restrict row, const int16_t* __restrict src,
                                       const int16_t* const* __restrict sub, int ns,
                                       const int16_t* const* __restrict add, int na) noexcept {
        const int16_t* base = (Init == RowInit::Src) ? src : row;
#if defined(__AVX2__)
        constexpr int TILE = 128;
        static_assert(HIDDEN % TILE == 0);
        for (int b = 0; b < HIDDEN; b += TILE) {
            const auto* s = reinterpret_cast<const __m256i*>(base + b);
            __m256i x0 = _mm256_loadu_si256(s + 0), x1 = _mm256_loadu_si256(s + 1);
            __m256i x2 = _mm256_loadu_si256(s + 2), x3 = _mm256_loadu_si256(s + 3);
            __m256i x4 = _mm256_loadu_si256(s + 4), x5 = _mm256_loadu_si256(s + 5);
            __m256i x6 = _mm256_loadu_si256(s + 6), x7 = _mm256_loadu_si256(s + 7);
            applyWeightList<false>(sub, ns, b, x0, x1, x2, x3, x4, x5, x6, x7);
            applyWeightList<true>(add, na, b, x0, x1, x2, x3, x4, x5, x6, x7);
            auto* d = reinterpret_cast<__m256i*>(row + b);
            _mm256_storeu_si256(d + 0, x0); _mm256_storeu_si256(d + 1, x1);
            _mm256_storeu_si256(d + 2, x2); _mm256_storeu_si256(d + 3, x3);
            _mm256_storeu_si256(d + 4, x4); _mm256_storeu_si256(d + 5, x5);
            _mm256_storeu_si256(d + 6, x6); _mm256_storeu_si256(d + 7, x7);
        }
#else
        constexpr int TILE = 64;
        static_assert(HIDDEN % TILE == 0);
        for (int b = 0; b < HIDDEN; b += TILE) {
            int16_t x[TILE];
            for (int i = 0; i < TILE; ++i) x[i] = base[b + i];
            for (int k = 0; k < ns; ++k)
                for (int i = 0; i < TILE; ++i) x[i] = static_cast<int16_t>(x[i] - sub[k][b + i]);
            for (int k = 0; k < na; ++k)
                for (int i = 0; i < TILE; ++i) x[i] = static_cast<int16_t>(x[i] + add[k][b + i]);
            for (int i = 0; i < TILE; ++i) row[b + i] = x[i];
        }
#endif
    }

    // row = src - psub + padd - tsub + tadd in one pass: S and A piece rows
    // (i16) and any number of threat rows (i8).
    template<int S, int A>
    static inline void applyMixedRows(int16_t* __restrict row, const int16_t* __restrict src,
                                      const int16_t* const* psub, const int16_t* const* padd,
                                      const int8_t* const* __restrict tsub, int tns,
                                      const int8_t* const* __restrict tadd, int tna) noexcept {
#if defined(__AVX2__)
        constexpr int TILE = 128;
        static_assert(HIDDEN % TILE == 0);
        const auto loadVec = [](const int16_t* p) noexcept { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
        const __m256i bias = _mm256_set1_epi16(static_cast<int16_t>(128 * (tna - tns)));
        for (int b = 0; b < HIDDEN; b += TILE) {
            __m256i r[4], q[4];
            for (int j = 0; j < 4; ++j) r[j] = q[j] = _mm256_setzero_si256();
            applyThreatList<false>(tsub, tns, b, r, q);
            applyThreatList<true>(tadd, tna, b, r, q);
            for (int j = 0; j < 4; ++j) {
                const int o = b + 32 * j;
                __m256i lo = _mm256_sub_epi16(_mm256_sub_epi16(r[j], _mm256_slli_epi16(q[j], 8)), bias);
                __m256i hi = q[j];
                lo = _mm256_add_epi16(lo, loadVec(src + o));
                hi = _mm256_add_epi16(hi, loadVec(src + o + 16));
                for (int k = 0; k < S; ++k) {
                    lo = _mm256_sub_epi16(lo, loadVec(psub[k] + o));
                    hi = _mm256_sub_epi16(hi, loadVec(psub[k] + o + 16));
                }
                for (int k = 0; k < A; ++k) {
                    lo = _mm256_add_epi16(lo, loadVec(padd[k] + o));
                    hi = _mm256_add_epi16(hi, loadVec(padd[k] + o + 16));
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(row + o), lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(row + o + 16), hi);
            }
        }
#else
        for (int i = 0; i < HIDDEN; ++i) {
            int x = src[i];
            for (int k = 0; k < S; ++k) x -= psub[k][i];
            for (int k = 0; k < A; ++k) x += padd[k][i];
            for (int k = 0; k < tns; ++k) x -= tsub[k][i];
            for (int k = 0; k < tna; ++k) x += tadd[k][i];
            row[i] = static_cast<int16_t>(x);
        }
#endif
    }

    // Applies a whole piece-feature diff to one row in one pass, from src (a
    // different row) when it is given, in place otherwise.
    static inline void applyRows(int16_t* __restrict row,
                                 const int16_t* const* sub, int ns,
                                 const int16_t* const* add, int na,
                                 const int16_t* src = nullptr) noexcept {
        if (ns == 0 && na == 0) {
            if (src != nullptr) std::memcpy(row, src, HIDDEN * sizeof(int16_t));
            return;
        }
        if (src != nullptr) applyWeightRows<RowInit::Src>(row, src, sub, ns, add, na);
        else                applyWeightRows<RowInit::InPlace>(row, nullptr, sub, ns, add, na);
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
