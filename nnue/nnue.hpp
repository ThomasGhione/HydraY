#pragma once

// NNUE evaluation (NNUE_PLAN.md, Fase 3) — the engine's only evaluator since
// the HCE removal (Fase 5). A network must be active before any evaluate():
// main() activates the embedded net at startup; EvalFile can override it;
// datagen/selftest load theirs explicitly. The accumulator maintenance on
// Board activates as soon as a network is loaded.

#include <cstdint>
#include <string>

namespace chess { class Board; }

namespace NNUE {

struct Network;

// Loads a bullet quantised.bin (validates size, padding signature and the
// output-weight bound the AVX2 forward relies on). Not for use mid-search.
[[nodiscard]] bool loadNetwork(const std::string& path);

// The network compiled into the binary (nnue/net/hydray.nnue via
// nnue/embedded.cpp); nullptr if the blob fails validation.
[[nodiscard]] const Network* embeddedNetwork() noexcept;

// Makes the embedded network active (UseNNUE with no EvalFile set).
[[nodiscard]] bool activateEmbedded() noexcept;

[[nodiscard]] bool networkLoaded() noexcept;

// True when the active network is a DEEP one (1024 -> 16 -> 1). Selected
// automatically by loadNetwork() from the file size; the embedded net is always
// the single-layer one. Exposed for `info string` diagnostics: loading the
// wrong format silently would be the worst way to lose an SPRT.
[[nodiscard]] bool deepNetworkActive() noexcept;

// stm-relative centipawns from the board's incrementally-maintained
// accumulator. Requires networkLoaded().
[[nodiscard]] int32_t evaluate(const chess::Board& board) noexcept;

} // namespace NNUE
