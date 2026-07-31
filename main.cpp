#include <iostream>
#include <string_view>

#include "./engine/engine.hpp"
#include "./engine/perft.hpp"
#include "./driver/driver.hpp"
#include "./uci/uci.hpp"
#include "./nnue/datagen.hpp"
#include "./nnue/nnue.hpp"
#include "./nnue/selftest.hpp"

#ifndef _WIN32
#include <unistd.h> // isatty, STDIN_FILENO
#else
#include <io.h>     // _isatty, _fileno
#endif

using namespace chess;
using namespace engine;
using namespace driver;

// UCI auto-detection: if stdin is a pipe (launched by a GUI / lichess-bot),
// enter UCI mode immediately - this is the standard UCI protocol behaviour.
static bool stdinIsPipe() noexcept {
#ifndef _WIN32
    return !isatty(STDIN_FILENO);
#else
    return !_isatty(_fileno(stdin));
#endif
}

int main(int argc, char *argv[]) {

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Dispatched before Engine is built: datagen needs no TT/book/ponder
    // thread, only the magic tables (initialized inside runDatagen).
    if (argc >= 2) {
        const std::string_view mode = argv[1];
        if (mode == "datagen")       return NNUE::runDatagen(argc, argv);
        if (mode == "datagen-dump")  return NNUE::runDatagenDump(argc, argv);
        if (mode == "nnue-selftest") return NNUE::runSelfTest(argc, argv);
    }

    // NNUE is the only evaluator (HCE removed, NNUE_PLAN.md Fase 5).
    // Activated before Engine is built so the Board constructor already
    // fills the accumulator. An invalid embedded blob means a broken build:
    // fail fast instead of searching with garbage evals.
    if (!NNUE::activateEmbedded()) {
        std::cerr << "fatal: embedded NNUE network failed validation - rebuild with a valid nnue/net/hydray.nnue\n";
        return 1;
    }

    // Perft needs no TT, book or ponder thread, but Board's constructor fills
    // the NNUE accumulator, so it is dispatched after the net is activated.
    if (argc >= 2 && std::string_view(argv[1]) == "perft")
        return engine::runPerftCommand(argc, argv);

    Engine engine;

    if (argc == 1 && stdinIsPipe()) {
        uci::UCI uciInterface(engine);
        uciInterface.mainLoop(); //mainLoop() is [[noreturn]] so this will exit the program when done
    }

    Driver driver(engine);
    driver.startGame(argc, argv);
}
