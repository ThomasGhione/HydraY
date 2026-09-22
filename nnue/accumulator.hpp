#pragma once

// Dual-perspective NNUE accumulator: v[0] = white's view, v[1] = black's view.
//
// HalfKA king buckets: every feature index of perspective X
// depends on the (bucket, flip) of X's OWN king - the "basis". The basis of
// each perspective is cached here (base = 768*bucket, flip), and the class
// keeps this invariant:
//
//   clean[p]  =>  every feature in v[p] was accumulated with the stored
//                 basis, and that basis matches the current own-king square.
//
// Piece events (Board's addPieceToBB/removePieceFromBB) update both
// perspectives incrementally under their stored bases; add/sub are exact
// integer inverses, so undoMove needs no MoveState snapshot. When a king
// ADD lands on a square whose basis differs from the stored one, that
// perspective is marked DIRTY and stops applying updates: its features
// would need the new basis for the whole board, not a delta. The lazy
// refresh happens in Board::ensureNnueAccumulatorClean() (called by
// NNUE::evaluate and the selftest), where the Board is consistent -
// mid-doMove refreshes would read a half-updated position. Undo needs no
// special case: while dirty everything is skipped and the next refresh
// rebuilds from the real board; while clean the inverse deltas are exact.
//
// Board rebuilds (FEN load, table refresh) call refreshNnueAccumulator()
// for a from-scratch recompute of both perspectives.

#include <cstdint>
#include <cstring>

#include "network.hpp"
#include "network_deep.hpp"

namespace NNUE {

// One deferred accumulator mutation. doMove records these instead of touching
// the 1024-wide rows; evaluate replays them in order, which reproduces the
// eager sequence exactly (the dirty/basis transitions depend on that order).
// A do/undo pair that never reaches an evaluate cancels here and costs nothing.
struct AccDelta {
    enum Kind : uint8_t { Add, Remove, Move };
    uint8_t kind;
    uint8_t piece;
    uint8_t from;   // Add/Remove: the square
    uint8_t to;     // Move only
};

// Deepest chain of moves that can be made without an evaluate is bounded by
// MAX_PLY + the qsearch cap, at most three deltas each. The queue overflowing
// only forces an early replay, so correctness never rests on this number.
inline constexpr int MAX_ACC_PENDING = 512;

struct alignas(64) Accumulator {
    int16_t v[2][HIDDEN];
    int32_t base[2];   // 768 * king bucket, per perspective
    uint8_t flip[2];   // 0 or 7 (file mirror), per perspective
    bool    dirty[2];  // true = v[p] stale, skip updates until refresh

    // Wipes both perspectives to the bias and installs the given own-king
    // squares (LERF, already from each perspective's view) as bases.
    inline void resetWithKings(int wKingLerf, int bKingLerfFromBlack) noexcept {
        const Deep::NetworkDeep& net = *activeNetwork;
        std::memcpy(v[0], net.l0b, sizeof(net.l0b));
        std::memcpy(v[1], net.l0b, sizeof(net.l0b));
        base[0] = kingFeatureBase(wKingLerf);
        base[1] = kingFeatureBase(bKingLerfFromBlack);
        flip[0] = static_cast<uint8_t>(kingFlip(wKingLerf));
        flip[1] = static_cast<uint8_t>(kingFlip(bKingLerfFromBlack));
        dirty[0] = dirty[1] = false;
    }

    // A pair of queued deltas folded into one traversal of the rows. The queue
    // at a flush is almost always the previous sibling's move undone followed by
    // this move done, so replaying entry by entry reads and writes both rows
    // twice. The weight rows must be read either way; the accumulator rows need
    // not be. int16 is modular, so the order the contributions are summed in
    // does not change the result, the same argument updateMove relies on.
    //
    // The counts are template parameters so the inner loops unroll: a version
    // with runtime bounds measured 50% SLOWER than replaying one at a time.
    template<int NS, int NA>
    inline void updateFused(const int16_t* const (&sub)[2][2],
                            const int16_t* const (&add)[2][2]) noexcept {
        for (int i = 0; i < HIDDEN; ++i) {
            int s0 = v[0][i];
            int s1 = v[1][i];
            for (int k = 0; k < NS; ++k) { s0 -= sub[0][k][i]; s1 -= sub[1][k][i]; }
            for (int k = 0; k < NA; ++k) { s0 += add[0][k][i]; s1 += add[1][k][i]; }
            v[0][i] = static_cast<int16_t>(s0);
            v[1][i] = static_cast<int16_t>(s1);
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

    [[nodiscard]] inline const int16_t* featureRow(int p, uint8_t piece, uint8_t index) const noexcept {
        return rowFor(base[p], flip[p], p, piece, index);
    }

    // Adds/removes one piece feature on an arbitrary accumulator row under a
    // fixed basis - shared by the live perspectives and the Finny cache rows.
    template<bool Add>
    static inline void updateRow(int16_t* __restrict row, int rowBase, int rowFlip,
                                 int p, uint8_t piece, uint8_t index) noexcept {
        const int16_t* __restrict w = rowFor(rowBase, rowFlip, p, piece, index);
        for (int i = 0; i < HIDDEN; ++i) {
            if constexpr (Add) row[i] = static_cast<int16_t>(row[i] + w[i]);
            else               row[i] = static_cast<int16_t>(row[i] - w[i]);
        }
    }

    // Single-row counterpart of updateFused, for the Finny refresh: several
    // feature rows summed in one traversal instead of one traversal each.
    template<int NS, int NA>
    static inline void fuseRows(int16_t* __restrict row, const int16_t* const* sub,
                                const int16_t* const* add) noexcept {
        for (int i = 0; i < HIDDEN; ++i) {
            int x = row[i];
            for (int k = 0; k < NS; ++k) x -= sub[k][i];
            for (int k = 0; k < NA; ++k) x += add[k][i];
            row[i] = static_cast<int16_t>(x);
        }
    }

    // Applies a whole feature diff, two subs and two adds per traversal. A
    // refresh changes 5.8 features on average, so this reads and writes the row
    // about twice instead of six times, and the weight rows of one chunk miss
    // in parallel rather than one after the other.
    static inline void applyRows(int16_t* __restrict row,
                                 const int16_t* const* sub, int ns,
                                 const int16_t* const* add, int na) noexcept {
        for (int si = 0, ai = 0; si < ns || ai < na; ) {
            const int s = (ns - si > 2) ? 2 : ns - si;
            const int a = (na - ai > 2) ? 2 : na - ai;
            switch (s * 4 + a) {
                case 2 * 4 + 2: fuseRows<2, 2>(row, sub + si, add + ai); break;
                case 2 * 4 + 1: fuseRows<2, 1>(row, sub + si, add + ai); break;
                case 1 * 4 + 2: fuseRows<1, 2>(row, sub + si, add + ai); break;
                case 1 * 4 + 1: fuseRows<1, 1>(row, sub + si, add + ai); break;
                case 2 * 4 + 0: fuseRows<2, 0>(row, sub + si, add + ai); break;
                case 0 * 4 + 2: fuseRows<0, 2>(row, sub + si, add + ai); break;
                case 1 * 4 + 0: fuseRows<1, 0>(row, sub + si, add + ai); break;
                default:        fuseRows<0, 1>(row, sub + si, add + ai); break;
            }
            si += s;
            ai += a;
        }
    }

    template<bool Add>
    inline void updatePerspective(int p, uint8_t piece, uint8_t index) noexcept {
        updateRow<Add>(v[p], base[p], flip[p], p, piece, index);
    }

    // piece = Board nibble (WHITE=0x8 color bit | type 1..6), index = engine
    // square (0 = a8, 63 = h1); LERF square = index ^ 56.
    template<bool Add>
    inline void update(uint8_t piece, uint8_t index) noexcept {
        const int type = (piece & 0x7) - 1;
        const bool black = (piece & 0x8) == 0;
        const int lerf = index ^ 56;

        // A king ADD landing on a square with a different basis invalidates
        // every feature of its own perspective: mark dirty, the next
        // ensureClean rebuilds. (The matching REMOVE is always applied under
        // the stored basis, which by the invariant is the basis the king was
        // added with.)
        if constexpr (Add) {
            if (type == 5) [[unlikely]] {
                const int p = black ? 1 : 0;
                if (!dirty[p]) {
                    const int kView = (p == 1) ? (lerf ^ 56) : lerf;
                    if (kingFeatureBase(kView) != base[p]
                        || kingFlip(kView) != flip[p]) {
                        dirty[p] = true;
                    }
                }
            }
        }

        // Fast path (the overwhelming majority): both perspectives clean -
        // one fused loop updates both rows, same ILP as the pre-HalfKA code.
        if (!dirty[0] && !dirty[1]) [[likely]] {
            const Deep::NetworkDeep& net = *activeNetwork;
            const int featW = (black ? 384 : 0) + type * 64 + lerf;
            const int featB = (black ? 0 : 384) + type * 64 + (lerf ^ 56);
            const int16_t* __restrict w0 = net.l0w[base[0] + (featW ^ flip[0])];
            const int16_t* __restrict w1 = net.l0w[base[1] + (featB ^ flip[1])];
            for (int i = 0; i < HIDDEN; ++i) {
                if constexpr (Add) {
                    v[0][i] = static_cast<int16_t>(v[0][i] + w0[i]);
                    v[1][i] = static_cast<int16_t>(v[1][i] + w1[i]);
                } else {
                    v[0][i] = static_cast<int16_t>(v[0][i] - w0[i]);
                    v[1][i] = static_cast<int16_t>(v[1][i] - w1[i]);
                }
            }
            return;
        }
        if (!dirty[0]) updatePerspective<Add>(0, piece, index);
        if (!dirty[1]) updatePerspective<Add>(1, piece, index);
    }

    // Sposta un pezzo da `from` a `to` con UNA passata sull'accumulatore invece
    // delle due che costano un remove e un add separati. E' la forma piu'
    // comune di gran lunga: una mossa quieta e' esattamente questo, e undoMove
    // ne fa un'altra al contrario.
    //
    // Il guadagno sta negli accessi in memoria, non nelle somme: due passate
    // leggono l'accumulatore due volte e lo riscrivono due volte, questa lo
    // legge e lo riscrive una volta sola. Misurato in isolamento, 145 ns contro
    // 98 per le due separate.
    //
    // L'aritmetica e' IDENTICA, non approssimata: in int16 la somma e'
    // associativa modulo 2^16, quindi (v - s) + a == v - s + a anche quando
    // trabocca. Il collaudo e' nnue-selftest, che confronta accumulatore
    // incrementale e ricostruito da zero.
    inline void updateMove(uint8_t piece, uint8_t fromIndex, uint8_t toIndex) noexcept {
        const int type = (piece & 0x7) - 1;
        // Un re che atterra puo' cambiare king bucket, e allora la sua
        // prospettiva va rifatta da zero: quella logica vive in update<true> e
        // non si fonde. Con una prospettiva gia' sporca, idem.
        if (type == 5 || dirty[0] || dirty[1]) [[unlikely]] {
            update<false>(piece, fromIndex);
            update<true>(piece, toIndex);
            return;
        }

        const Deep::NetworkDeep& net = *activeNetwork;
        const bool black = (piece & 0x8) == 0;
        const int lerfFrom = fromIndex ^ 56;
        const int lerfTo   = toIndex ^ 56;
        const int featW = (black ? 384 : 0) + type * 64;
        const int featB = (black ? 0 : 384) + type * 64;
        const int16_t* __restrict s0 = net.l0w[base[0] + ((featW + lerfFrom) ^ flip[0])];
        const int16_t* __restrict a0 = net.l0w[base[0] + ((featW + lerfTo)   ^ flip[0])];
        const int16_t* __restrict s1 = net.l0w[base[1] + ((featB + fromIndex) ^ flip[1])];
        const int16_t* __restrict a1 = net.l0w[base[1] + ((featB + toIndex)   ^ flip[1])];
        for (int i = 0; i < HIDDEN; ++i) {
            v[0][i] = static_cast<int16_t>(v[0][i] - s0[i] + a0[i]);
            v[1][i] = static_cast<int16_t>(v[1][i] - s1[i] + a1[i]);
        }
    }
};

// Finny table: one cached accumulator row per (perspective, king bucket,
// mirror flip) plus the piece bitboards it was computed from. A lazy refresh
// becomes a DIFF against the cached state (typically a handful of piece
// updates) instead of a ~30-piece rebuild. Any cached content is a correct
// starting point as long as the entry's basis matches - even from another
// game - so the table lives per-thread (Lazy SMP helpers each get their own)
// and never needs invalidation. The bias-initialised entry with empty
// bitboards is itself a valid state ("diff from the empty board").
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
