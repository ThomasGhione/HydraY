#pragma once

// Quantised network for NNUE v4 (HALFKA_PLAN.md):
//   (768x4kb_hm -> 512)x2 -> 8, SCReLU, QA=255 QB=64.
//
// Input features are king-bucketed and horizontally mirrored (bullet
// ChessBucketsMirrored semantics — sanity.rs is the reference):
//   for perspective X with own king on ksq_X (LERF, from X's own view):
//     flip_X   = 7 if file(ksq_X) > 3 else 0
//     bucket_X = KING_BUCKET_MAP[ksq_X]
//     feature  = 768*bucket_X + (feat768 ^ flip_X)
//   where feat768 = isOpp*384 + type*64 + sq_from_X and `^ flip` flips the
//   file of the square component only.
// Output: 8 material-count buckets, bucket = (popcount(occ) - 2) / 4.
//
// The struct mirrors bullet's quantised.bin byte-for-byte (little-endian i16;
// l0f factoriser already merged into l0w at save; l1w saved TRANSPOSED so each
// bucket's 2*HIDDEN weights are contiguous). Kept intentionally light: this
// header is included by board.hpp for the accumulator hot-path hooks.

#include <cstddef>
#include <cstdint>

namespace NNUE {

inline constexpr int INPUTS = 768;
inline constexpr int HIDDEN = 512;
inline constexpr int INPUT_BUCKETS = 4;
inline constexpr int OUTPUT_BUCKETS = 8;
inline constexpr int32_t QA = 255;
inline constexpr int32_t QB = 64;
inline constexpr int32_t SCALE = 400;

// Keep in sync with trainer.rs/sanity.rs BUCKET_LAYOUT (32-entry half-board
// map, files a-d per rank starting at rank 1; e-h fold onto d-a). Expanded
// here to 64 LERF squares exactly like bullet's ChessBucketsMirrored::new.
inline constexpr uint8_t KING_BUCKET_MAP[64] = {
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
};

// Feature base (= 768 * bucket) and file-flip mask for a perspective whose
// own king, seen from that perspective, sits on `lerfKsq`.
[[nodiscard]] inline constexpr int kingFeatureBase(int lerfKsq) noexcept {
    return INPUTS * KING_BUCKET_MAP[lerfKsq];
}
[[nodiscard]] inline constexpr int kingFlip(int lerfKsq) noexcept {
    return (lerfKsq & 0x7) > 3 ? 7 : 0;
}

struct alignas(64) Network {
    int16_t featureWeights[INPUT_BUCKETS * INPUTS][HIDDEN]; // l0w (QA, factoriser merged)
    int16_t featureBias[HIDDEN];                            // l0b (QA)
    int16_t outputWeights[OUTPUT_BUCKETS][2][HIDDEN];       // l1w: [bucket][stm|ntm half] (QB)
    int16_t outputBias[OUTPUT_BUCKETS];                     // l1b (QA*QB)
};

inline constexpr size_t NETWORK_PAYLOAD_BYTES =
    sizeof(int16_t) * (INPUT_BUCKETS * INPUTS * HIDDEN + HIDDEN
                       + OUTPUT_BUCKETS * 2 * HIDDEN + OUTPUT_BUCKETS);

static_assert(NETWORK_PAYLOAD_BYTES == 3163152);
static_assert(offsetof(Network, featureBias) == sizeof(int16_t) * INPUT_BUCKETS * INPUTS * HIDDEN);
static_assert(offsetof(Network, outputWeights) == offsetof(Network, featureBias) + sizeof(int16_t) * HIDDEN);
static_assert(offsetof(Network, outputBias) == offsetof(Network, outputWeights) + sizeof(int16_t) * OUTPUT_BUCKETS * 2 * HIDDEN);

// Non-null once a network is loaded (see NNUE::loadNetwork in nnue.hpp). Read
// in Board's piece-update hot path; written only with no search running.
extern const Network* activeNetwork;

} // namespace NNUE
