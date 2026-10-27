#include <iostream>
#include <string_view>

#include "./ascii_utils.hpp"
#include "./engine/engine.hpp"
#include "./engine/perft.hpp"
#include "./uci/uci.hpp"
#include "./nnue/datagen.hpp"
#include "./nnue/nnue.hpp"
#include "./nnue/selftest.hpp"

int main(int argc, char *argv[]) {

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string_view mode = (argc >= 2) ? argv[1] : "uci";
    // These load their own net.
    if (mode == "datagen")       return NNUE::runDatagen(argc, argv);
    if (mode == "datagen-dump")  return NNUE::runDatagenDump(argc, argv);
    if (mode == "nnue-selftest") return NNUE::runSelfTest(argc, argv);

    // Before any Board is built, so its constructor fills the accumulator.
    if (!NNUE::activateEmbedded()) {
        std::cerr << "fatal: embedded NNUE network failed validation - rebuild with a valid nnue/net/hydray.nnue\n";
        return 1;
    }

    if (mode == "perft") return engine::runPerftCommand(argc, argv);

    while (mode.starts_with('-')) mode.remove_prefix(1);
    if (!ascii::iequals(mode, "uci")) {
        std::cerr << "usage: chess [uci | perft | datagen | datagen-dump | nnue-selftest] ...\n";
        return 1;
    }

    engine::Engine engine;
    uci::UCI(engine).mainLoop();
}
