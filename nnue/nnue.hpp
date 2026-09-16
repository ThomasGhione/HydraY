#pragma once

// NNUE evaluation - the engine's only evaluator since the HCE removal in
// 2.0.0. A network must be active before any evaluate():
// main() activates the embedded net at startup; EvalFile can override it;
// datagen/selftest load theirs explicitly. The accumulator maintenance on
// Board activates as soon as a network is loaded.

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

[[nodiscard]] bool networkLoaded() noexcept;

// stm-relative centipawns from the board's incrementally-maintained
// accumulator. Requires networkLoaded().
[[nodiscard]] int32_t evaluate(const chess::Board& board) noexcept;

} // namespace NNUE
