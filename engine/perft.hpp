#pragma once

#include <cstdint>

namespace chess { class Board; }

namespace engine {

// Perft ("performance test"): the exact number of leaf nodes of the legal-move
// tree at fixed depth. The counts for the standard positions are published, so
// a single wrong number proves a defect in move generation, legality, or
// doMove/undoMove - with no evaluation or search involved.
//
// Depth 1 is answered by the move count alone (bulk counting). With divide it
// prints the count under each root move ("e2e4: 8902") and the total: descend
// into the branch that differs from a reference engine to find the bad move.
[[nodiscard]] uint64_t perft(chess::Board& b, int depth, bool divide = false) noexcept;

// Checks the standard positions against their published counts, depths 1 to
// maxDepth. Returns true when every count matches. `verbose` prints every
// position and depth; when false only failures and the summary are printed.
[[nodiscard]] bool runPerftSuite(int maxDepth, bool verbose = true);

// CLI entry for `./chess perft ...`.
int runPerftCommand(int argc, char* argv[]);

} // namespace engine
