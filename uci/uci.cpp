#include "uci.hpp"

#include "../ascii_utils.hpp"
#include "../engine/engine.hpp"
#include "../nnue/nnue.hpp"
#include "output.hpp"

#include <omp.h>

#include <iostream>

namespace {
    // "RfpMarginPerDepth", "RFP_MARGIN_PER_DEPTH" and "rfp margin per depth"
    // all name the same option.
    std::string normalizedOptionName(std::string_view optionName) {
        std::string normalized;
        for (const char c : optionName) {
            if (c != ' ' && c != '_' && c != '-') normalized += ascii::toLower(c);
        }
        return normalized;
    }

    // RFP_MARGIN_PER_DEPTH -> RfpMarginPerDepth
    std::string optionDisplayName(std::string_view key) {
        std::string display;
        bool upperNext = true;
        for (const char c : key) {
            if (c == '_') {
                upperNext = true;
                continue;
            }
            display += upperNext ? ascii::toUpper(c) : ascii::toLower(c);
            upperNext = false;
        }
        return display;
    }

    struct SpinOption final {
        const char* key;
        int32_t* value;
        int32_t minValue;
        int32_t maxValue;
        // Rebuild the search-derived tables (LMR/futility/LMP) after writing.
        bool refreshSearchTables = false;
    };

    SpinOption kSpinOptions[] = {
        {"RFP_MARGIN_PER_DEPTH", &engine::RFP_MARGIN_PER_DEPTH,           30,  250},
        {"NMP_EVAL_DIV",         &engine::NMP_EVAL_DIV,                   50,  400},
        {"NMP_EVAL_MAX",         &engine::NMP_EVAL_MAX,                    0,   10},
        {"SEE_CAPTURE_MARGIN",   &engine::SEE_CAPTURE_MARGIN,             20,  200},
        {"SEE_PRUNE_MAX_DEPTH",  &engine::SEE_PRUNE_MAX_DEPTH,             0,   12},
        {"QUIET_SEE_MARGIN",     &engine::QUIET_SEE_MARGIN,                0,  200},
        {"QUIET_SEE_MAX_DEPTH",  &engine::QUIET_SEE_MAX_DEPTH,             0,   12},
        {"PROBCUT_MARGIN",       &engine::PROBCUT_MARGIN,                 30,  250},
        {"SE_BETA_MARGIN",       &engine::SE_BETA_MARGIN,                  0,   16},
        {"HISTORY_PRUNE_D1",     &engine::HISTORY_PRUNE_THRESHOLD[1], -16384,    0},
        {"HISTORY_PRUNE_D2",     &engine::HISTORY_PRUNE_THRESHOLD[2], -16384,    0},
        {"HISTORY_PRUNE_D3",     &engine::HISTORY_PRUNE_THRESHOLD[3], -16384,    0},
        {"FUTILITY_MID_STEP",    &engine::FUTILITY_MID_STEP,              80,  600, true},
        {"FUTILITY_MAX_DEPTH",   &engine::FUTILITY_MAX_DEPTH,              0, engine::FUTILITY_DEPTH_CAP},
        {"LMP_SCALE_STD_PCT",    &engine::LMP_SCALE_PCT[0],               10,  250, true},
        {"LMP_SCALE_IMP_PCT",    &engine::LMP_SCALE_PCT[1],               10,  250, true},
        {"LMP_MAX_DEPTH",        &engine::LMP_MAX_DEPTH,                   0, engine::LMP_DEPTH_CAP},
        {"LMR_C_PERCENT",        &engine::LMR_C_PERCENT,                 150,  600, true},
        {"LMR_CUT_BONUS",        &engine::LMR_CUT_BONUS,                   0,    4},
        {"LMR_MIN_INDEX_ALL",    &engine::LMR_MIN_INDEX_ALL,               1,    8},
        {"LMR_EARLY_QUIET_ONLY", &engine::LMR_EARLY_QUIET_ONLY,            0,    1},
        {"LMR_GOOD_HIST",        &engine::LMR_GOOD_HIST,                   0,    3},
        {"LMR_BAD_HIST",         &engine::LMR_BAD_HIST,                    0,    3},
        {"LMR_HIST_MODE",        &engine::LMR_HIST_MODE,                   0,    2},
        {"HIST_BONUS_PCT",       &engine::HIST_BONUS_PCT,                 25, 1000},
        {"IIR_MODE",             &engine::IIR_MODE,                        0,    1},
        {"IIR_MIN_DEPTH",        &engine::IIR_MIN_DEPTH,                   2,   12},
        {"TT_CUT_HIST",          &engine::TT_CUT_HIST,                     0,    1},
        {"LMR_CHECK_LESS",       &engine::LMR_CHECK_LESS,                  0,    3},
        {"QS_EVAL_PCT_MG",       &engine::QS_EVAL_PCT_MG,                100,  600},
        {"QS_EVAL_PCT_LATE_MG",  &engine::QS_EVAL_PCT_LATE_MG,           100,  600},
        {"QS_EVAL_PCT_ENDGAME",  &engine::QS_EVAL_PCT_ENDGAME,           100,  600},
        {"HINDSIGHT_EXT",        &engine::HINDSIGHT_EXT,                   0,    1},
        {"HINDSIGHT_MARGIN",     &engine::HINDSIGHT_MARGIN,                0, 2000},
        {"NMP_BASE_R",           &engine::NMP_BASE_R,                      1,    8},
        {"CORR_HIST_WEIGHT_PCT", &engine::CORR_HIST_WEIGHT_PCT,            0,  400},
    };

    std::string ponderSuffix(const engine::Engine& engine, const chess::Move& bestMove) noexcept {
        chess::Board board = engine.board;
        if (!board.move(bestMove)) return {};

        const chess::Move ponderMove = engine.tt.probeDecodedMove(board.getHash());
        if (!chess::isValidSquare(ponderMove.from)) return {};

        return board.move(ponderMove)
            ? (" ponder " + ponderMove.toUCIString())
            : std::string{};
    }

}

namespace uci {
    UCI::UCI(engine::Engine& e) : engine(e) {}

    UCI::~UCI() noexcept {
        stopSearch(false);
    }

    void UCI::emitBestMove(std::string_view move) noexcept {
        emit("bestmove " + std::string(move) + '\n');
        searchPrinted = true;
    }

    void UCI::stopSearch(bool printBestMove) noexcept {
        engine.stop();
        if (searchThread.joinable()) searchThread.join();

        std::lock_guard<std::mutex> lock(searchMutex);
        if (printBestMove && !searchPrinted) emitBestMove(searchBestMove);
    }

    [[noreturn]] void UCI::mainLoop() noexcept {
        std::string command;
        while (std::getline(std::cin, command)) parseCommand(command);
        // EOF is an implicit quit, as in Stockfish: exit() destroys the network
        // a running search still reads, so stop and join the search first.
        stopSearch(true);
        std::exit(EXIT_SUCCESS);
    }

    void UCI::parseCommand(const std::string& command) noexcept {
        std::istringstream is(command);
        std::string token;
        is >> token;
        if (token == "quit") {
            stopSearch(true);
            std::exit(EXIT_SUCCESS);
        }
        if (token == "uci") uci();
        else if (token == "isready") emit("readyok\n");
        else if (token == "stop") stopSearch(true);
        else if (token == "ponderhit") ponderhit();
        else if (token == "ucinewgame") {
            stopSearch(false);
            engine.reset();
        }
        else if (token == "setoption") setOption(is);
        else if (token == "position") position(is);
        else if (token == "go") go(is);
    }

    void UCI::uci() noexcept {
        const std::string threads = std::to_string(omp_get_max_threads());
        std::string out =
            "id name HydraY 4.3.0\n"
            "id author Thomas Ghione, Andrea Vaccari, Daniele Ferretti, Simone Tomasella\n"
            "option name SyzygyPath type string default <empty>\n"
            "option name SyzygyProbeDepth type spin default 1 min 1 max 100\n"
            "option name EvalFile type string default <empty>\n"
            "option name Threads type spin default " + threads + " min 1 max " + threads + "\n"
            "option name Hash type spin default " + std::to_string(TT::DEFAULT_HASH_MB)
            + " min " + std::to_string(TT::MIN_HASH_MB) + " max " + std::to_string(TT::MAX_HASH_MB) + "\n";
        for (const auto& option : kSpinOptions) {
            out += "option name " + optionDisplayName(option.key)
                 + " type spin default " + std::to_string(*option.value)
                 + " min " + std::to_string(option.minValue)
                 + " max " + std::to_string(option.maxValue) + "\n";
        }
        emit(out + "uciok\n");
    }

    void UCI::setOption(std::istringstream& is) noexcept {
        // The options are plain globals the search reads: stop it first.
        stopSearch(false);
        std::string token, name, value;
        if (!(is >> token) || token != "name") return;
        while (is >> token && token != "value") name += (name.empty() ? "" : " ") + token;
        while (is >> token) value += (value.empty() ? "" : " ") + token;
        if (name.empty()) return;

        const std::string key = normalizedOptionName(name);
        int v = 0;
        const bool isInt = static_cast<bool>(std::istringstream(value) >> v);
        const auto invalid = [](const std::string& option) { emit("info string invalid value for " + option + "\n"); };

        for (auto& option : kSpinOptions) {
            if (key != normalizedOptionName(option.key)) continue;
            if (!isInt) return invalid(name);
            if (v < option.minValue || v > option.maxValue) {
                return emit("info string value out of range for " + name + " ("
                            + std::to_string(option.minValue) + ".." + std::to_string(option.maxValue) + ")\n");
            }
            *option.value = v;
            if (option.refreshSearchTables) engine::rebuildSearchDerivedTables();
            return emit("info string " + optionDisplayName(option.key) + " set to " + std::to_string(v) + "\n");
        }

        if (key == "evalfile") {
            if (NNUE::loadNetwork(value)) {
                engine::clearEvalCache();
                // The board may already hold a position: bring the accumulator
                // in sync now; later `position` commands refresh via FEN load.
                engine.board.refreshNnueAccumulator();
                emit("info string EvalFile loaded: " + value + "\n");
            }
        } else if (key == "syzygypath") {
            emit(engine.syzygyProber.load(value)
                 ? "info string SyzygyPath loaded: " + value + " (max pieces: "
                   + std::to_string(engine.syzygyProber.maxPieces()) + ")\n"
                 : "info string SyzygyPath error: could not load '" + value + "'\n");
        } else if (key == "syzygyprobedepth") {
            if (!isInt || v < 1 || v > 100) return invalid("SyzygyProbeDepth");
            engine.syzygyProber.probeDepth = v;
            emit("info string SyzygyProbeDepth set to " + std::to_string(v) + "\n");
        } else if (key == "threads") {
            if (!isInt || v < 1) return invalid("Threads");
            engine.requestedThreads = engine.searchRuntime.maxThreads = std::min(v, omp_get_max_threads());
            emit("info string Threads set to " + std::to_string(engine.requestedThreads) + "\n");
        } else if (key == "hash") {
            if (!isInt || v < 1) return invalid("Hash");
            const int mb = std::min(v, static_cast<int>(TT::MAX_HASH_MB));
            emit(engine.tt.resize(static_cast<size_t>(mb))
                 ? "info string Hash set to " + std::to_string(engine.tt.sizeMB()) + " MB\n"
                 : "info string Hash resize to " + std::to_string(mb) + " MB failed (out of memory)\n");
        }
    }

    void UCI::position(std::istringstream& is) noexcept {
        stopSearch(false);
        std::string token, fen;
        is >> token;
        if (token == "startpos") {
            fen = chess::Board::STARTING_FEN;
            is >> token; // "moves"
        } else if (token == "fen") {
            while (is >> token && token != "moves") fen += token + ' ';
        } else {
            return;
        }
        engine.board = chess::Board(fen);
        // Coordinate notation; an illegal move is skipped.
        while (is >> token) {
            if (token.size() < 4) continue;
            const chess::Square from = chess::parseSquare(token.substr(0, 2));
            const chess::Square to = chess::parseSquare(token.substr(2, 2));
            if (!chess::isValidSquare(from) || !chess::isValidSquare(to)) continue;
            const uint8_t promo = (token.size() > 4) ? chess::Move::promotionTypeFromChar(token[4]) : 0;
            engine.board.move({from, to, promo});
        }
    }

    void UCI::go(std::istringstream& is) noexcept {
        stopSearch(false);
        engine::time::Limits limits;
        std::string token;
        while (is >> token) {
            if (token == "ponder") limits.ponder = true;
            else if (token == "infinite") limits.infinite = true;
            else if (token == "depth") is >> limits.maxDepth;
            else if (token == "nodes") is >> limits.maxNodes;
            else if (token == "movestogo") is >> limits.movestogo;
            else if (token == "wtime") limits.hasClock = static_cast<bool>(is >> limits.wtime);
            else if (token == "btime") limits.hasClock = static_cast<bool>(is >> limits.btime);
            else if (token == "winc") is >> limits.winc;
            else if (token == "binc") is >> limits.binc;
            else if (token == "movetime") is >> limits.movetime;
        }

        const bool ponder = limits.ponder;
        {
            std::lock_guard<std::mutex> lock(searchMutex);
            searchBestMove = "0000";
            searchPonder = ponder;
            searchDone = false;
            searchPrinted = false;
        }
        try {
            engine.searchRuntime.emitUciInfo = true; // UCI mode streams "info" lines
            // Clear the stop flags here, on the reader thread: any `stop` is
            // parsed after this point, so its flag survives. Cleared inside the
            // search thread, a stop landing during startup was lost.
            engine.prepareSearch();
            searchThread = std::thread([this, limits] {
                const chess::Move move = engine.search(limits);
                // No legal move (mated or stalemated root): UCI's null move. A ponder
                // search ends in a real move after ponderhit, so it offers a ponder move too.
                const bool hasMove = chess::isValidSquare(move.from) && chess::isValidSquare(move.to);
                std::string bestMove = hasMove ? move.toUCIString() : std::string("0000");
                if (hasMove) bestMove += ponderSuffix(engine, move);
                std::lock_guard<std::mutex> lock(searchMutex);
                searchBestMove = bestMove;
                searchDone = true;
                if (!searchPonder && !searchPrinted) emitBestMove(searchBestMove);
            });
        } catch (...) {
            std::lock_guard<std::mutex> lock(searchMutex);
            searchDone = true;
            if (!ponder) emitBestMove("0000");
        }
    }
    
    void UCI::ponderhit() noexcept {
        std::lock_guard<std::mutex> lock(searchMutex);
        searchPonder = false;
        if (searchDone) {
            if (!searchPrinted) emitBestMove(searchBestMove);
            return;
        }
        // Still searching: from now on the clock is ours.
        engine.timeManager.ponderhit();
    }
}
