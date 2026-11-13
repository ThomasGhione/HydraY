#pragma once

// The quantised network:
//
//   (768x4kb_hm + threats -> 1024)x2 -> pairwise -> 1024 -> 16 -> 1
//
// The feature space and l0 are in network.hpp, all that board.hpp needs. One
// hidden layer: trainer_deep.rs says why not bullet's 16->32->1 example. The
// reference reader is nnue/trainer/src/bin/sanity_deep.rs; when the two
// disagree, sanity_deep is right.
//
// FILE LAYOUT (little-endian, padded to 64 B with "bullet" repeated), OB =
// output buckets:
//
//   l0w [4*768][1024]  i16  QA=255   (factoriser already folded in at save)
//   l0b [1024]         i16  QA
//   l1w [OB][16][1024] i8   QB=64    (transposed: each bucket contiguous)
//   l1b [OB][16]       f32  real scale
//   l2w [OB][1][16]    f32
//   l2b [OB]           f32
//   l0t [70050][1024]  i8   QA=255   (threat inputs, threats.hpp; absent in older nets)
//   OB = 16: payload 6,557,760 B, with threats 78,288,960 B (no padding)
//   OB = 8:  payload 6,425,632 B, with threats 78,156,832 B
//
// All four are loaded. An 8-bucket file gives each of its heads the two
// 16-bucket slots it covers, and a file without threats gets zero threat
// weights: both compute exactly the function they were trained as.
//
// ARITHMETIC (must match sanity_deep.rs line for line):
//
//   acc[i] = pieces[i] + threats[i]              two i16 accumulators, saturating
//   c[i] = clamp(acc[i], 0, QA)                  QA scale, fits in u8
//   p[j] = c[j] * c[j+512] / QA                  TRUNCATING INTEGER division
//   z[o] = SUM(p[i] * l1w[o][i]) / (QA*QB) + l1b[o]   FLOAT division
//   a[o] = clamp(z[o], 0, 1)^2                   SCReLU in float
//   y    = SUM a[o]*l2w[o] + l2b                 linear
//   cp   = round(y * outputScale)              SCALE, normalised by the engine
//
// The division for p is integer and the one for z is floating point: that is
// not an oversight, it is what the oracle does. Making them uniform would
// change the results.

#include <cstddef>
#include <cstdint>

#include "../huge_pages.hpp"
#include "network.hpp"
#include "threats.hpp"

namespace NNUE::Deep {

using NNUE::HIDDEN, NNUE::INPUTS, NNUE::INPUT_BUCKETS; // the tools reach them through Deep
inline constexpr int L1_SIZE       = 16;
inline constexpr int OUTPUT_BUCKETS = 16;
inline constexpr int LEGACY_OUTPUT_BUCKETS = 8;
// bullet's MaterialCount<OUTPUT_BUCKETS>: bucket = (popcount(occ) - 2) / BUCKET_DIVISOR.
inline constexpr int BUCKET_DIVISOR = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS;
// A legacy head covers OUTPUT_BUCKETS / LEGACY_OUTPUT_BUCKETS consecutive buckets
// only if its divisor is that multiple of ours.
static_assert(OUTPUT_BUCKETS % LEGACY_OUTPUT_BUCKETS == 0
              && (32 + LEGACY_OUTPUT_BUCKETS - 1) / LEGACY_OUTPUT_BUCKETS
                     == BUCKET_DIVISOR * (OUTPUT_BUCKETS / LEGACY_OUTPUT_BUCKETS),
              "legacy heads must map onto whole runs of buckets");
inline constexpr int32_t QA    = 255;
inline constexpr int32_t QB    = 64;
inline constexpr int32_t SCALE = 400;

// Pairwise output per perspective; l1's input is HIDDEN (two perspectives
// concatenated), not 2*HIDDEN -- which is why the layer is cheap.
inline constexpr int PAIRWISE_OUT = HIDDEN / 2;
static_assert(2 * PAIRWISE_OUT == HIDDEN, "pairwise needs an even HIDDEN");

struct alignas(64) NetworkDeep {
    int16_t l0w[INPUT_BUCKETS * INPUTS][HIDDEN];
    int16_t l0b[HIDDEN];
    int8_t  l1w[OUTPUT_BUCKETS][L1_SIZE][HIDDEN];
    float   l1b[OUTPUT_BUCKETS][L1_SIZE];
    float   l2w[OUTPUT_BUCKETS][L1_SIZE];
    float   l2b[OUTPUT_BUCKETS];

    // A copy of l1w already widened to i16, filled at load time. Only the path
    // WITHOUT AVX-VNNI needs it, where the dot product works on i16 lanes:
    // converting the weights on every evaluation cost one cvtepi8_epi16 per
    // useful instruction. 512 KiB more for a clean inner loop.
    // NOT in the file: derived, and rebuilt on every load.
    alignas(64) int16_t l1w16[OUTPUT_BUCKETS][L1_SIZE][HIDDEN];

    // TRANSPOSED weights for the sparse path (AVX-VNNI only). 85% of l1's input
    // is zero, but an output-driven loop multiplies by zero anyway. Driving it
    // from the NON-zero inputs instead needs, for a given group of four inputs,
    // the weights towards all 16 outputs: 16*4 = 64 contiguous bytes. The first
    // 32 are outputs 0-7 (four weights each), the second 32 are outputs 8-15 --
    // exactly the two vpdpbusd operands.
    //
    // The row index is the group of four in the order the pairwise pass WRITES,
    // which is not l1w's order: packus works per 128-bit lane and swaps the two
    // middle quarters of every 32-wide block. Absorbing that swap here is free
    // (the table is built either way) and removes one vpermq per 32 outputs
    // from the hot loop.
    //
    // Derived, not read from the file. 256 KiB, plus one zero row per bucket
    // that pads the non-zero list to a multiple of four.
    alignas(64) int8_t l1wT[OUTPUT_BUCKETS][HIDDEN / 4 + 1][L1_SIZE * 4];

    // [THREAT_FEATURES][HIDDEN], read at a few scattered rows per move: on huge
    // pages, and zero when the file has none.
    HugePageArray<int8_t> l0t;

    // How many output buckets the loaded file had (OUTPUT_BUCKETS or
    // LEGACY_OUTPUT_BUCKETS), and whether it had threat weights. Not in the
    // file either.
    int fileBuckets = 0;
    bool fileThreats = false;

    // y -> centipawns. SCALE gives the net's own (training) units; the engine
    // sets it per net so that every net speaks the reference unit (nnue.cpp).
    float outputScale = static_cast<float>(SCALE);
};

// The order packus_epi16 leaves a block's 32 bytes in, relative to the source:
// the two middle quarters come out swapped. Used by l1wT and by the consistency
// check between the two paths.
inline constexpr int PACKUS_MAP[32] = {
     0,  1,  2,  3,  4,  5,  6,  7, 16, 17, 18, 19, 20, 21, 22, 23,
     8,  9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31,
};

inline constexpr size_t THREAT_BYTES = static_cast<size_t>(THREAT_FEATURES) * HIDDEN * sizeof(int8_t);

// Payload size on disk, independent of the struct's padding: this is what the
// loader must read, and it has to be computed from the fields rather than from
// sizeof(NetworkDeep), which alignas can inflate.
[[nodiscard]] constexpr size_t payloadBytes(int buckets, bool threats) noexcept {
    const auto ob = static_cast<size_t>(buckets);
    return static_cast<size_t>(INPUT_BUCKETS) * INPUTS * HIDDEN * sizeof(int16_t)
         + static_cast<size_t>(HIDDEN) * sizeof(int16_t)
         + ob * L1_SIZE * HIDDEN * sizeof(int8_t)
         + ob * L1_SIZE * sizeof(float)
         + ob * L1_SIZE * sizeof(float)
         + ob * sizeof(float)
         + (threats ? THREAT_BYTES : 0);
}
// bullet pads the payload to 64 B, and adds nothing when it is already a multiple.
[[nodiscard]] constexpr size_t fileBytes(int buckets, bool threats) noexcept {
    return (payloadBytes(buckets, threats) + 63) / 64 * 64;
}
// The layout of a file of `size` bytes; buckets is 0 if it is none of them.
struct FileLayout {
    int buckets = 0;
    bool threats = false;
};
[[nodiscard]] constexpr FileLayout layoutForFileSize(size_t size) noexcept {
    for (const int buckets : {OUTPUT_BUCKETS, LEGACY_OUTPUT_BUCKETS})
        for (const bool threats : {true, false})
            if (size == fileBytes(buckets, threats)) return {buckets, threats};
    return {};
}
static_assert(payloadBytes(OUTPUT_BUCKETS, false) == 6'557'760, "layout changed: update sanity_deep.rs");
static_assert(payloadBytes(LEGACY_OUTPUT_BUCKETS, false) == 6'425'632, "legacy layout changed");
static_assert(payloadBytes(LEGACY_OUTPUT_BUCKETS, true) == 78'156'832, "threat layout changed: update sanity_deep.rs");

// Threat accumulator of a net without threat weights, for the tools.
alignas(64) inline constexpr int16_t NO_THREATS[HIDDEN] = {};

// Scalar forward, from the two perspectives' piece and threat accumulators to
// the evaluation in centipawns. The correctness reference for the vectorised
// version.
[[nodiscard]] int32_t forwardScalar(const NetworkDeep& net,
                                    const int16_t* accStm, const int16_t* accNtm,
                                    const int16_t* thrStm, const int16_t* thrNtm,
                                    int outputBucket) noexcept;

// The same function, vectorised. It MUST return the same value as
// forwardScalar for every input: it is an optimisation, not a different net.
[[nodiscard]] int32_t forwardSimd(const NetworkDeep& net,
                                  const int16_t* accStm, const int16_t* accNtm,
                                  const int16_t* thrStm, const int16_t* thrNtm,
                                  int outputBucket) noexcept;

// Whether the binary has AVX-VNNI (vpdpbusd). Without it forwardSimd takes an
// exact i16-lane path that costs twice the instructions in l1.
[[nodiscard]] bool hasVnniPath() noexcept;

// Loads a quantised.bin in any layout above, and builds the derived tables
// (l1w16, l1wT). False if the size or the padding is wrong, or out of memory.
[[nodiscard]] bool loadFromFile(const char* path, NetworkDeep& net) noexcept;
[[nodiscard]] bool loadFromMemory(const unsigned char* data, size_t size,
                                  NetworkDeep& net) noexcept;

} // namespace NNUE::Deep
