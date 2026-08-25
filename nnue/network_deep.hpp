#pragma once

// Quantised network with a hidden layer.
//
//   (768x4kb_hm -> 1024)x2 -> pairwise -> 1024 -> 16 -> 1
//
// ONE hidden layer only: see trainer_deep.rs for why bullet's 16->32->1 example
// was not copied.
//
// The INPUTS are identical to the single-layer net (network.hpp): the same 4
// mirrored king buckets, the same feature formula, the same i16 accumulator.
// Only what happens AFTER the accumulator differs, so accumulator.hpp needs no
// changes.
//
// This header is deliberately SEPARATE from network.hpp so the two formats can
// coexist without #ifdefs scattered through the engine.
//
// The reference reader is nnue/trainer/src/bin/sanity_deep.rs. When the two
// disagree, sanity_deep is right: it is the oracle, it is simpler, and it is
// written against bullet's format.
//
// FILE LAYOUT (little-endian, padded to 64 B with "bullet" repeated):
//
//   l0w [4*768][1024]  i16  QA=255   (factoriser already folded in at save)
//   l0b [1024]         i16  QA
//   l1w [8][16][1024]  i8   QB=64    (transposed: each bucket contiguous)
//   l1b [8][16]        f32  real scale
//   l2w [8][1][16]     f32
//   l2b [8]            f32
//   payload 6,425,632 B, file 6,425,664 B
//
// ARITHMETIC (must match sanity_deep.rs line for line):
//
//   c[i] = clamp(acc[i], 0, QA)                  QA scale, fits in u8
//   p[j] = c[j] * c[j+512] / QA                  TRUNCATING INTEGER division
//   z[o] = SUM(p[i] * l1w[o][i]) / (QA*QB) + l1b[o]   FLOAT division
//   a[o] = clamp(z[o], 0, 1)^2                   SCReLU in float
//   y    = SUM a[o]*l2w[o] + l2b                 linear
//   cp   = round(y * SCALE)
//
// The division for p is integer and the one for z is floating point: that is
// not an oversight, it is what the oracle does. Making them uniform would
// change the results.

#include <cstddef>
#include <cstdint>

namespace NNUE::Deep {

inline constexpr int INPUTS        = 768;
inline constexpr int HIDDEN        = 1024;
inline constexpr int L1_SIZE       = 16;
inline constexpr int INPUT_BUCKETS = 4;
inline constexpr int OUTPUT_BUCKETS = 8;
inline constexpr int32_t QA    = 255;
inline constexpr int32_t QB    = 64;
inline constexpr int32_t SCALE = 400;

// Pairwise output per perspective; l1's input is HIDDEN (two perspectives
// concatenated), not 2*HIDDEN -- which is why the layer is cheap.
inline constexpr int PAIRWISE_OUT = HIDDEN / 2;
static_assert(2 * PAIRWISE_OUT == HIDDEN, "il pairwise richiede HIDDEN pari");

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
    // useful instruction. 256 KiB more for a clean inner loop.
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
    // Derived, not read from the file. 128 KiB.
    alignas(64) int8_t l1wT[OUTPUT_BUCKETS][HIDDEN / 4][L1_SIZE * 4];
};

// The order packus_epi16 leaves a block's 32 bytes in, relative to the source:
// the two middle quarters come out swapped. Used by l1wT and by the consistency
// check between the two paths.
inline constexpr int PACKUS_MAP[32] = {
     0,  1,  2,  3,  4,  5,  6,  7, 16, 17, 18, 19, 20, 21, 22, 23,
     8,  9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31,
};

// Payload size on disk, independent of the struct's padding: this is what the
// loader must read, and it has to be computed from the fields rather than from
// sizeof(NetworkDeep), which alignas can inflate.
inline constexpr size_t PAYLOAD_BYTES =
      static_cast<size_t>(INPUT_BUCKETS) * INPUTS * HIDDEN * sizeof(int16_t)
    + static_cast<size_t>(HIDDEN) * sizeof(int16_t)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L1_SIZE * HIDDEN * sizeof(int8_t)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L1_SIZE * sizeof(float)
    + static_cast<size_t>(OUTPUT_BUCKETS) * L1_SIZE * sizeof(float)
    + static_cast<size_t>(OUTPUT_BUCKETS) * sizeof(float);
static_assert(PAYLOAD_BYTES == 6'425'632, "layout cambiato: aggiorna sanity_deep.rs");

// Scalar forward, from the two perspectives' accumulators to the evaluation in
// centipawns. The correctness reference for the vectorised version.
[[nodiscard]] int32_t forwardScalar(const NetworkDeep& net,
                                    const int16_t* accStm,
                                    const int16_t* accNtm,
                                    int outputBucket) noexcept;

// The same function, vectorised. It MUST return the same value as
// forwardScalar for every input: it is an optimisation, not a different net.
[[nodiscard]] int32_t forwardSimd(const NetworkDeep& net,
                                  const int16_t* accStm,
                                  const int16_t* accNtm,
                                  int outputBucket) noexcept;

// true se il binario e' stato compilato con AVX-VNNI (vpdpbusd), che raddoppia
// le moltiplicazioni per istruzione nel layer l1. Senza, forwardSimd usa un
// percorso a corsie i16 che e' esatto ma costa il doppio delle istruzioni.
[[nodiscard]] bool hasVnniPath() noexcept;

// Carica un file quantised.bin nel formato sopra. Ritorna false (senza
// scrivere in `net`) se taglia o padding non tornano.
[[nodiscard]] bool loadFromFile(const char* path, NetworkDeep& net) noexcept;

// Same, from a blob already in memory: the EMBEDDED net is not a file. It also
// builds the derived tables (l1w16, l1wT), so unlike the single-layer net the
// deep one cannot be used as a plain overlay on the blob.
[[nodiscard]] bool loadFromMemory(const unsigned char* data, size_t size,
                                  NetworkDeep& net) noexcept;

} // namespace NNUE::Deep
