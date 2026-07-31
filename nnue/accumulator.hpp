#pragma once

// Dual-perspective NNUE accumulator: v[0] = white's view, v[1] = black's view.
//
// HalfKA king buckets (HALFKA_PLAN.md): every feature index of perspective X
// depends on the (bucket, flip) of X's OWN king — the "basis". The basis of
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
// NNUE::evaluate and the selftest), where the Board is consistent —
// mid-doMove refreshes would read a half-updated position. Undo needs no
// special case: while dirty everything is skipped and the next refresh
// rebuilds from the real board; while clean the inverse deltas are exact.
//
// Board rebuilds (FEN load, table refresh) call refreshNnueAccumulator()
// for a from-scratch recompute of both perspectives.

#include <cstdint>
#include <cstring>

#include "network.hpp"

namespace NNUE {

struct alignas(64) Accumulator {
    int16_t v[2][HIDDEN];
    int32_t base[2];   // 768 * king bucket, per perspective
    uint8_t flip[2];   // 0 or 7 (file mirror), per perspective
    bool    dirty[2];  // true = v[p] stale, skip updates until refresh

    // Wipes both perspectives to the bias and installs the given own-king
    // squares (LERF, already from each perspective's view) as bases.
    inline void resetWithKings(int wKingLerf, int bKingLerfFromBlack) noexcept {
        const Network& net = *activeNetwork;
        std::memcpy(v[0], net.featureBias, sizeof(net.featureBias));
        std::memcpy(v[1], net.featureBias, sizeof(net.featureBias));
        base[0] = kingFeatureBase(wKingLerf);
        base[1] = kingFeatureBase(bKingLerfFromBlack);
        flip[0] = static_cast<uint8_t>(kingFlip(wKingLerf));
        flip[1] = static_cast<uint8_t>(kingFlip(bKingLerfFromBlack));
        dirty[0] = dirty[1] = false;
    }

    // Rebuild helpers for a single perspective (lazy refresh path).
    inline void resetPerspective(int p, int ownKingLerfView) noexcept {
        const Network& net = *activeNetwork;
        std::memcpy(v[p], net.featureBias, sizeof(net.featureBias));
        base[p] = kingFeatureBase(ownKingLerfView);
        flip[p] = static_cast<uint8_t>(kingFlip(ownKingLerfView));
        dirty[p] = false;
    }

    // Adds/removes one piece feature on an arbitrary accumulator row under a
    // fixed basis — shared by the live perspectives and the Finny cache rows.
    template<bool Add>
    static inline void updateRow(int16_t* __restrict row, int rowBase, int rowFlip,
                                 int p, uint8_t piece, uint8_t index) noexcept {
        const Network& net = *activeNetwork;
        const int type = (piece & 0x7) - 1;      // P..K -> 0..5
        const bool black = (piece & 0x8) == 0;   // Board::WHITE = 0x8
        const int lerf = index ^ 56;
        const int sqView = (p == 1) ? (lerf ^ 56) : lerf;
        const bool isOpp = (p == 1) ? !black : black;
        const int feat768 = (isOpp ? 384 : 0) + type * 64 + sqView;
        const int16_t* __restrict w = net.featureWeights[rowBase + (feat768 ^ rowFlip)];
        for (int i = 0; i < HIDDEN; ++i) {
            if constexpr (Add) row[i] = static_cast<int16_t>(row[i] + w[i]);
            else               row[i] = static_cast<int16_t>(row[i] - w[i]);
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

        // Fast path (the overwhelming majority): both perspectives clean —
        // one fused loop updates both rows, same ILP as the pre-HalfKA code.
        if (!dirty[0] && !dirty[1]) [[likely]] {
            const Network& net = *activeNetwork;
            const int featW = (black ? 384 : 0) + type * 64 + lerf;
            const int featB = (black ? 0 : 384) + type * 64 + (lerf ^ 56);
            const int16_t* __restrict w0 = net.featureWeights[base[0] + (featW ^ flip[0])];
            const int16_t* __restrict w1 = net.featureWeights[base[1] + (featB ^ flip[1])];
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
};

// Finny table: one cached accumulator row per (perspective, king bucket,
// mirror flip) plus the piece bitboards it was computed from. A lazy refresh
// becomes a DIFF against the cached state (typically a handful of piece
// updates) instead of a ~30-piece rebuild. Any cached content is a correct
// starting point as long as the entry's basis matches — even from another
// game — so the table lives per-thread (Lazy SMP helpers each get their own)
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
    const Network* builtFor = nullptr;

    inline void ensureInitialised() noexcept {
        if (builtFor == activeNetwork) [[likely]] return;
        const Network& net = *activeNetwork;
        for (auto& perPersp : entry)
            for (auto& perBucket : perPersp)
                for (auto& e : perBucket) {
                    std::memcpy(e.v, net.featureBias, sizeof(e.v));
                    std::memset(e.bb, 0, sizeof(e.bb));
                }
        builtFor = activeNetwork;
    }
};

} // namespace NNUE
