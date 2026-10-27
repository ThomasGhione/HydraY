#pragma once

// NNUE evaluation. A network must be active before any evaluate(): main()
// activates the embedded one, EvalFile replaces it, datagen and the selftest
// load their own.

#include <cstdint>
#include <string>

namespace chess { class Board; }

namespace NNUE {

// Loads a bullet quantised.bin (validates size and padding signature). Not for
// use mid-search.
[[nodiscard]] bool loadNetwork(const std::string& path);

// Makes the network compiled into the binary active (nnue/net/hydray.nnue via
// nnue/embedded.cpp); false if the blob fails validation.
[[nodiscard]] bool activateEmbedded() noexcept;

// stm-relative centipawns from the board's incrementally-maintained
// accumulator. Requires an active network.
[[nodiscard]] int32_t evaluate(const chess::Board& board) noexcept;

} // namespace NNUE
