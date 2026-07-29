#include "uci.hpp"

#include "../ascii_utils.hpp"
#include "../engine/engine.hpp"
#include "../nnue/nnue.hpp"

#include <omp.h>

#include <charconv>
#include <iostream>

namespace {
    using std::string_view;

    static string_view trimLeft(string_view s) noexcept {
        size_t i = 0;
        while (i < s.size() && ascii::isSpace(s[i])) ++i;
        return s.substr(i);
    }

    static string_view trimRight(string_view s) noexcept {
        while (!s.empty() && ascii::isSpace(s.back())) s.remove_suffix(1);
        return s;
    }

    static std::string normalizedOptionName(string_view optionName) {
        std::string normalized;
        normalized.reserve(optionName.size());
        for (const char c : optionName) {
            if (c == ' ' || c == '_' || c == '-') continue;
            normalized.push_back(ascii::toLower(c));
        }
        return normalized;
    }

    static std::string optionDisplayName(std::string_view rawName) {
        std::string display;
        display.reserve(rawName.size());
        bool upperNext = true;
        for (const char c : rawName) {
            if (c == '_' || c == '-' || c == ' ') {
                upperNext = true;
                continue;
            }
            display.push_back(upperNext ? ascii::toUpper(c)
                                        : ascii::toLower(c));
            upperNext = false;
        }
        return display;
    }

    struct SpinOption final {
        const char* key;
        int32_t* value;
        bool hasRange;
        int32_t minValue;
        int32_t maxValue;
        // Rebuild the search-derived tables (LMR/futility/LMP) after writing.
        bool refreshSearchTables = false;
    };

    // Wide auto-range so the tuner's sampled values land inside the engine's
    // accepted bounds without each option needing an explicit hasRange entry.
    // ±200% of magnitude with a floor of 10 covers the typical
    // chess-tuning-tools parameter ranges used by the JSON groups.
    static std::pair<int32_t, int32_t> defaultRangeFor(int32_t value) noexcept {
        const int64_t absValue = std::llabs(static_cast<int64_t>(value));
        const int64_t delta = std::max<int64_t>(10, absValue * 2);
        const auto toI32 = [](int64_t x) {
            return static_cast<int32_t>(std::clamp<int64_t>(x, INT32_MIN, INT32_MAX));
        };
        return {toI32(value - delta), toI32(value + delta)};
    }

    // Resolved [min, max] range advertised and accepted for a spin option:
    // its explicit bounds when given, otherwise the auto-range.
    static std::pair<int32_t, int32_t> optionRange(const SpinOption& option) noexcept {
        return option.hasRange ? std::pair{option.minValue, option.maxValue}
                               : defaultRangeFor(*option.value);
    }

    static SpinOption kSpinOptions[] = {
        // --- Search constants (SMAC3 campaign, HOTPATH #9). Generators of the
        // derived tables set refreshSearchTables so LMR/futility/LMP rebuild.
        {"RFP_MARGIN_PER_DEPTH", &engine::RFP_MARGIN_PER_DEPTH,        true,     30,  250},
        {"NMP_EVAL_DIV",         &engine::NMP_EVAL_DIV,                true,     50,  400},
        {"NMP_EVAL_MAX",         &engine::NMP_EVAL_MAX,                true,      0,   10},
        {"SEE_CAPTURE_MARGIN",   &engine::SEE_CAPTURE_MARGIN,          true,     20,  200},
        {"PROBCUT_MARGIN",       &engine::PROBCUT_MARGIN,              true,     30,  250},
        {"SE_BETA_MARGIN",       &engine::SE_BETA_MARGIN,              true,      0,   16},
        {"SE_DOUBLE_MARGIN",     &engine::SE_DOUBLE_MARGIN,            true,      4,   64},
        {"HISTORY_PRUNE_D1",     &engine::HISTORY_PRUNE_THRESHOLD[1],  true, -16384,    0},
        {"HISTORY_PRUNE_D2",     &engine::HISTORY_PRUNE_THRESHOLD[2],  true, -16384,    0},
        {"HISTORY_PRUNE_D3",     &engine::HISTORY_PRUNE_THRESHOLD[3],  true, -16384,    0},
        {"FUTILITY_MID_STEP",    &engine::FUTILITY_MID_STEP,           true,     80,  600, true},
        {"LMP_SCALE_STD_PCT",    &engine::LMP_SCALE_PCT[0],            true,     40,  250, true},
        {"LMP_SCALE_IMP_PCT",    &engine::LMP_SCALE_PCT[1],            true,     40,  250, true},
        {"LMR_C_PERCENT",        &engine::LMR_C_PERCENT,               true,    150,  600, true},
    };

    static bool parseCheckValue(string_view rawValue, bool& outValue) noexcept {
        const string_view value = trimLeft(rawValue);
        if (ascii::iequals(value, "true") || value == "1" || ascii::iequals(value, "on")) {
            outValue = true;
            return true;
        }
        if (ascii::iequals(value, "false") || value == "0" || ascii::iequals(value, "off")) {
            outValue = false;
            return true;
        }
        return false;
    }

    static string_view nextToken(string_view& text) noexcept {
        text = trimLeft(text);
        if (text.empty()) return {};
        size_t end = 0;
        while (end < text.size() && !ascii::isSpace(text[end])) ++end;
        const string_view token = text.substr(0, end);
        text.remove_prefix(end);
        return token;
    }

    // Parses a whole signed-integer token (optional leading '+'); the entire
    // token must be consumed. Works for any integer type via std::from_chars.
    template<typename T>
    static bool parseInt(string_view token, T& out) noexcept {
        if (token.starts_with('+')) token.remove_prefix(1);
        if (token.empty()) return false;
        const auto [ptr, err] = std::from_chars(token.data(), token.data() + token.size(), out);
        return err == std::errc{} && ptr == token.data() + token.size();
    }

    // Parses the first whitespace-delimited token of `text` as an integer.
    template<typename T>
    static bool parseFirstInt(string_view text, T& out) noexcept {
        return parseInt(nextToken(text), out);
    }

    static bool splitCommand(string_view command, string_view name, string_view& args) noexcept {
        if (!command.starts_with(name)) return false;
        if (command.size() == name.size()) {
            args = {};
            return true;
        }
        if (!ascii::isSpace(command[name.size()])) return false;
        args = trimLeft(command.substr(name.size()));
        return true;
    }

    static size_t findWord(string_view text, string_view word) noexcept {
        size_t pos = text.find(word);
        while (pos != string_view::npos) {
            const size_t end = pos + word.size();
            const bool leftOk = (pos == 0) || ascii::isSpace(text[pos - 1]);
            const bool rightOk = (end == text.size()) || ascii::isSpace(text[end]);
            if (leftOk && rightOk) return pos;
            pos = text.find(word, pos + 1);
        }
        return string_view::npos;
    }

    static std::string ponderSuffix(const engine::Engine& engine, const chess::Move& bestMove) noexcept {
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
        std::cout << "bestmove " << move << '\n';
        std::cout.flush();
        searchPrinted = true;
    }

    void UCI::stopSearch(bool printBestMove) noexcept {
        engine.stopThinking();
        if (searchThread.joinable()) searchThread.join();

        std::lock_guard<std::mutex> lock(searchMutex);
        if (printBestMove && !searchPrinted) emitBestMove(searchBestMove);
    }

    [[noreturn]] void UCI::mainLoop() noexcept {
        std::string command;
        while (std::getline(std::cin, command)) {
            if (!command.empty() && command.back() == '\r') command.pop_back();
            if (command.empty()) continue;
            parseCommand(command);
        }
        std::exit(EXIT_SUCCESS);
    }

    void UCI::parseCommand(std::string_view command) noexcept {
        string_view args;
        if (command == "quit") {
            stopSearch(true);
            std::exit(EXIT_SUCCESS);
        }
        if (command == "uci") return uci();
        if (command == "ucinewgame") return ucinewgame();
        if (command == "isready") return isready();
        if (command == "stop") return stopSearch(true);
        if (command == "ponderhit") return ponderhit();
        if (splitCommand(command, "setoption", args)) return setOption(args);
        if (splitCommand(command, "position", args)) return position(args);
        if (splitCommand(command, "go", args)) return go(args);
    }


    void UCI::uci() noexcept {
        std::cout
            << "id name HydraY 2.1.0\n"
            << "id author Thomas Ghione, Daniele Ferretti, Simone Tomasella\n"
            << "option name SyzygyPath type string default <empty>\n"
            << "option name SyzygyProbeDepth type spin default 1 min 1 max 100\n"
            << "option name SearchApiMutexGuard type check default true\n"
            << "option name EvalFile type string default <empty>\n";
        const int hwThreads = omp_get_max_threads();
        std::cout << "option name Threads type spin default " << hwThreads
                  << " min 1 max " << hwThreads << "\n";
        std::cout << "option name Hash type spin default " << TT::DEFAULT_HASH_MB
                  << " min " << TT::MIN_HASH_MB
                  << " max " << TT::MAX_HASH_MB << "\n";
        for (const auto& option : kSpinOptions) {
            const auto [minValue, maxValue] = optionRange(option);
            std::cout << "option name " << optionDisplayName(option.key)
                      << " type spin default " << *option.value
                      << " min " << minValue
                      << " max " << maxValue << "\n";
        }
        std::cout << "uciok\n";
        std::cout.flush();
    }

    void UCI::setOption(std::string_view args) noexcept {
        if (args.empty()) return;

        // Stop any in-flight search before mutating shared engine state: the
        // option backing storage is plain `int32_t` (engine::* globals) and
        // writing it while a worker reads is a data race.
        stopSearch(false);

        string_view rest = args;
        if (nextToken(rest) != "name") return;
        rest = trimLeft(rest); // nextToken leaves the leading gap before the name

        const size_t valuePos = findWord(rest, "value");
        const string_view optionName =
            (valuePos == string_view::npos) ? trimRight(rest) : trimRight(rest.substr(0, valuePos));
        const string_view optionValue =
            (valuePos == string_view::npos) ? string_view{} : trimLeft(rest.substr(valuePos + 5));

        if (optionName.empty()) return;
        const std::string normalizedName = normalizedOptionName(optionName);
        if (normalizedName == "evalfile") {
            const std::string path(optionValue);
            if (NNUE::loadNetwork(path)) {
                // The board may already hold a position: bring the accumulator
                // in sync now; later `position` commands refresh via FEN load.
                engine.board.refreshNnueAccumulator();
                std::cout << "info string EvalFile loaded: " << path << "\n";
            }
            return;
        }

        if (normalizedName == "syzygypath") {
            const std::string path(optionValue);
            if (engine.syzygyProber.load(path)) {
                std::cout << "info string SyzygyPath loaded: " << path
                          << " (max pieces: " << engine.syzygyProber.maxPieces() << ")\n";
            } else {
                std::cout << "info string SyzygyPath error: could not load '" << path << "'\n";
            }
            return;
        }

        if (normalizedName == "syzygyprobedepth") {
            int v = 0;
            if (parseFirstInt(optionValue, v) && v >= 1 && v <= 100) {
                engine.syzygyProber.probeDepth = v;
                std::cout << "info string SyzygyProbeDepth set to " << v << "\n";
            } else {
                std::cout << "info string invalid value for SyzygyProbeDepth\n";
            }
            return;
        }

        if (normalizedName == "threads") {
            int v = 0;
            const int maxT = omp_get_max_threads();
            if (parseFirstInt(optionValue, v) && v >= 1) {
                const int clamped = std::clamp(v, 1, maxT);
                engine.requestedThreads = clamped;
                engine.searchRuntime.maxThreads = clamped;
                std::cout << "info string Threads set to " << clamped << "\n";
            } else {
                std::cout << "info string invalid value for Threads\n";
            }
            return;
        }

        if (normalizedName == "hash") {
            int v = 0;
            if (parseFirstInt(optionValue, v) && v >= 1) {
                const int clamped = std::clamp(v,
                    static_cast<int>(TT::MIN_HASH_MB),
                    static_cast<int>(TT::MAX_HASH_MB));
                if (engine.tt.resize(static_cast<size_t>(clamped))) {
                    std::cout << "info string Hash set to " << engine.tt.sizeMB() << " MB\n";
                } else {
                    std::cout << "info string Hash resize to " << clamped << " MB failed (out of memory)\n";
                }
            } else {
                std::cout << "info string invalid value for Hash\n";
            }
            return;
        }

        if (normalizedName != "searchapimutexguard") {
            for (auto& option : kSpinOptions) {
                if (normalizedName != normalizedOptionName(option.key)) continue;
                int parsedValue = 0;
                if (!parseFirstInt(optionValue, parsedValue)) {
                    std::cout << "info string invalid value for " << optionName << "\n";
                    return;
                }
                const auto [minValue, maxValue] = optionRange(option);
                if (parsedValue < minValue || parsedValue > maxValue) {
                    std::cout << "info string value out of range for " << optionName
                              << " (" << minValue << ".." << maxValue << ")\n";
                    return;
                }
                *option.value = parsedValue;
                if (option.refreshSearchTables) {
                    engine::rebuildSearchDerivedTables();
                }
                std::cout << "info string " << optionDisplayName(option.key)
                          << " set to " << *option.value << "\n";
                return;
            }
            return;
        }

        bool enabled = false;
        if (optionValue.empty() || !parseCheckValue(optionValue, enabled)) {
            std::cout << "info string invalid value for " << optionName << ": '" << optionValue
                      << "' (use true/false)\n";
            return;
        }

        engine.setSearchApiMutexEnabled(enabled);
        std::cout << "info string SearchApiMutexGuard "
                  << (engine.isSearchApiMutexEnabled() ? "enabled" : "disabled")
                  << '\n';
    }

    void UCI::position(std::string_view command) noexcept {
        stopSearch(false);
        string_view moves;

        if (command.starts_with("startpos") && (command.size() == 8 || ascii::isSpace(command[8]))) {
            engine.board = chess::Board{};
            const size_t movesPos = findWord(command, "moves");
            if (movesPos != string_view::npos) moves = command.substr(movesPos);
        } else if (command.starts_with("fen") && (command.size() == 3 || ascii::isSpace(command[3]))) {
            string_view fenPayload = trimLeft(command.substr(3));
            const size_t movesPos = findWord(fenPayload, "moves");
            if (movesPos == string_view::npos) {
                parseFEN(fenPayload);
            } else {
                parseFEN(trimRight(fenPayload.substr(0, movesPos)));
                moves = fenPayload.substr(movesPos);
            }
        } else {
            return;
        }

        engine.bestMove = chess::Move{};
        engine.moveHistory.clear();
        
        if (!moves.empty()) parseMoves(moves);
        
        engine.updateGameResult();
    }

    void UCI::ucinewgame() noexcept {
        stopSearch(false);
        engine.reset();
    }

    void UCI::isready() noexcept {
        std::cout << "readyok\n";
        std::cout.flush();
    }
    
    void UCI::go(std::string_view args) noexcept {
        stopSearch(false);
        engine::time::Limits limits;

        while (!args.empty()) {
            const string_view token = nextToken(args);
            if (token.empty()) break;

            if (token == "ponder")   { limits.ponder = true;   continue; }
            if (token == "infinite") { limits.infinite = true; continue; }

            if (token == "depth") {
                int v = 0;
                if (parseInt(nextToken(args), v) && v > 0) limits.maxDepth = v;
                continue;
            }
            if (token == "nodes") {
                int64_t v = 0;
                if (parseInt(nextToken(args), v) && v > 0) {
                    limits.maxNodes = static_cast<uint64_t>(v);
                }
                continue;
            }
            if (token == "movestogo") {
                int v = 0;
                if (parseInt(nextToken(args), v) && v > 0) limits.movestogo = v;
                continue;
            }

            int64_t v = 0;
            if (token == "wtime"    && parseInt(nextToken(args), v)) { limits.wtime = v; limits.hasClock = true; continue; }
            if (token == "btime"    && parseInt(nextToken(args), v)) { limits.btime = v; limits.hasClock = true; continue; }
            if (token == "winc"     && parseInt(nextToken(args), v)) { limits.winc = v;  continue; }
            if (token == "binc"     && parseInt(nextToken(args), v)) { limits.binc = v;  continue; }
            if (token == "movetime" && parseInt(nextToken(args), v)) { limits.movetime = v; continue; }
            if (token == "mate")    { (void)nextToken(args); continue; }
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
            searchThread = std::thread([this, limits, ponder] {
                const chess::Move move = engine.searchUCI(limits);
                std::string bestMove = move.toUCIString();
                if (!ponder) bestMove += ponderSuffix(engine, move);
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
        if (searchDone && !searchPrinted) emitBestMove(searchBestMove);
    }

    void UCI::parseMoves(std::string_view moves) noexcept {
        // Precondition: caller passes the substring beginning at the "moves"
        // keyword (position() locates it via findWord), so just drop it.
        moves = trimLeft(moves);
        moves.remove_prefix(5);

        for (string_view move = nextToken(moves); !move.empty(); move = nextToken(moves)) {
            if (move.size() < 4) continue;

            const chess::Square from = chess::parseSquare(move.substr(0, 2));
            const chess::Square to = chess::parseSquare(move.substr(2, 2));
            if (!chess::isValidSquare(from) || !chess::isValidSquare(to)) continue;

            const char promo = (move.size() > 4) ? ascii::toLower(move[4]) : '\0';
            engine.movePiece(from, to, promo);
        }
    }

    void UCI::parseFEN(std::string_view fen) noexcept {
        engine.board = chess::Board(std::string(trimRight(trimLeft(fen))));
    }
}
