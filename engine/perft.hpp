#pragma once

#include <cstdint>

namespace chess { class Board; }

namespace engine {

// Perft ("performance test"): the exact number of leaf nodes of the legal-move
// tree at fixed depth. The counts for the standard positions are published, so
// a single wrong number proves a defect in move generation, legality, or
// doMove/undoMove — with no evaluation or search involved.
//
// Depth 1 is answered by the move count alone (standard bulk counting). Every
// move is still generated; each one also gets played at an internal node of any
// deeper run, so do/undo coverage is not lost.
[[nodiscard]] uint64_t perft(chess::Board& b, int depth) noexcept;

// Per-root-move breakdown ("e2e4: 8902" lines plus the total). Used to bisect a
// wrong total down to the offending move: descend into the branch that differs
// from a reference engine until the depth-1 mismatch names the bad move.
uint64_t perftDivide(chess::Board& b, int depth);

// Checks the standard positions against their published counts, depths 1 to
// maxDepth. Returns true when every count matches. `verbose` prints every
// position and depth; when false only failures and the summary are printed.
[[nodiscard]] bool runPerftSuite(int maxDepth, bool verbose = true);

// CLI entry for `./chess perft ...`.
int runPerftCommand(int argc, char* argv[]);

} // namespace engine
