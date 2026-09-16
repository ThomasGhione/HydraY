#pragma once

// Feature space and l0 prefix shared by every net format:
//   768x4kb_hm -> 1024 per perspective.
//
// What comes AFTER l0 lives in network_deep.hpp; this header stops at the
// accumulator, which is all board.hpp needs for the hot-path hooks.
//
// Input features are king-bucketed and horizontally mirrored (bullet
// ChessBucketsMirrored semantics - sanity.rs is the reference):
//   for perspective X with own king on ksq_X (LERF, from X's own view):
//     flip_X   = 7 if file(ksq_X) > 3 else 0
//     bucket_X = KING_BUCKET_MAP[ksq_X]
//     feature  = 768*bucket_X + (feat768 ^ flip_X)
//   where feat768 = isOpp*384 + type*64 + sq_from_X and `^ flip` flips the
//   file of the square component only.
// Output: 8 material-count buckets, bucket = (popcount(occ) - 2) / 4.
//
// The struct mirrors the leading bytes of bullet's quantised.bin (little-endian
// i16; l0f factoriser already merged into l0w at save).

#include <cstddef>
#include <cstdint>

namespace NNUE {

inline constexpr int INPUTS = 768;
inline constexpr int HIDDEN = 1024;
inline constexpr int INPUT_BUCKETS = 4;

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

namespace Deep { struct NetworkDeep; }

// Non-null once a network is loaded (see NNUE::loadNetwork in nnue.hpp). Read
// in Board's piece-update hot path; written only with no search running. The
// accumulator reads its l0 arrays straight out of it.
extern const Deep::NetworkDeep* activeNetwork;

} // namespace NNUE
