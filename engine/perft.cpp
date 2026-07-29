#include "perft.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "../board/board.hpp"
#include "../board/piece.hpp"
#include "movelist.hpp"
#include "sort/move_generator.hpp"

namespace engine {

namespace {

// Published leaf counts from chessprogramming.org; index i holds depth i+1.
// A 0 marks a depth left unchecked (too deep to be worth the wall time here).
struct PerftCase {
    const char* name;
    const char* fen;
    std::array<uint64_t, 6> counts;
};

constexpr std::array<PerftCase, 7> STANDARD_CASES = {{
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     {20, 400, 8902, 197281, 4865609, 119060324}},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     {48, 2039, 97862, 4085603, 193690690, 0}},
    {"position 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     {14, 191, 2812, 43238, 674624, 11030083}},
    {"position 4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     {6, 264, 9467, 422333, 15833292, 0}},
    // Same position with the colours swapped: identical counts, but it drives
    // the black-to-move side of every pawn-direction and castling path.
    {"position 4 mirrored", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
     {6, 264, 9467, 422333, 15833292, 0}},
    {"position 5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     {44, 1486, 62379, 2103487, 89941194, 0}},
    // Note the bishop colours on g5/g4: white on g5, black on g4. The variant
    // with both pairs unswapped is a different (also legal) position whose
    // counts are 46/2060/88933 — an easy transcription slip to make.
    {"position 6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     {46, 2079, 89890, 3894594, 164075551, 0}},
}};

// A FEN on the command line arrives pre-split on spaces; glue it back together.
std::string joinArgs(int argc, char* argv[], int first) {
    std::string joined;
    for (int i = first; i < argc; ++i) {
        if (!joined.empty()) joined += ' ';
        joined += argv[i];
    }
    return joined;
}

// fenToBoard bails out silently on malformed input, which would leave perft
// counting a position the user never asked for. One king per side is the cheap
// invariant that catches it.
bool boardLooksValid(const chess::Board& b) noexcept {
    return std::popcount(b.kings_bb[0]) == 1 && std::popcount(b.kings_bb[1]) == 1;
}

void printUsage() {
    std::cerr << "usage: ./chess perft <depth> [fen]\n"
                 "       ./chess perft divide <depth> [fen]\n"
                 "       ./chess perft suite [maxdepth]   (default 4)\n";
}

} // namespace

uint64_t perft(chess::Board& b, int depth) noexcept {
    if (depth <= 0) return 1;

    const MoveList moves = MoveGenerator::generateLegalMoves(b);
    if (depth == 1) return static_cast<uint64_t>(moves.size);

    uint64_t nodes = 0;
    chess::Board::MoveState st{};
    for (const chess::Move& m : moves) {
        b.doMove(m, st);
        nodes += perft(b, depth - 1);
        b.undoMove(m, st);
    }
    return nodes;
}

uint64_t perftDivide(chess::Board& b, int depth) {
    if (depth <= 0) return 1;

    const MoveList moves = MoveGenerator::generateLegalMoves(b);
    uint64_t total = 0;
    chess::Board::MoveState st{};
    for (const chess::Move& m : moves) {
        b.doMove(m, st);
        const uint64_t nodes = perft(b, depth - 1);
        b.undoMove(m, st);

        std::cout << m.toUCIString() << ": " << nodes << '\n';
        total += nodes;
    }
    std::cout << "\nmoves: " << moves.size << "\nnodes: " << total << '\n';
    return total;
}

bool runPerftSuite(int maxDepth, bool verbose) {
    using clock = std::chrono::steady_clock;

    const int cappedDepth = std::min(maxDepth, static_cast<int>(STANDARD_CASES[0].counts.size()));
    bool allPassed = true;
    uint64_t totalNodes = 0;
    const auto suiteStart = clock::now();

    for (const PerftCase& tc : STANDARD_CASES) {
        if (verbose) std::cout << tc.name << "\n  " << tc.fen << '\n';

        for (int depth = 1; depth <= cappedDepth; ++depth) {
            const uint64_t expected = tc.counts[static_cast<size_t>(depth) - 1];
            if (expected == 0) continue; // unchecked depth

            chess::Board b(tc.fen);
            if (!boardLooksValid(b)) {
                std::cout << tc.name << ": FAIL — FEN rejected by the parser\n";
                allPassed = false;
                break;
            }

            const auto start = clock::now();
            const uint64_t got = perft(b, depth);
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - start).count();
            totalNodes += got;

            const bool ok = (got == expected);
            allPassed &= ok;
            if (!ok) {
                std::cout << "  FAIL " << tc.name << " depth " << depth << ": " << got
                          << " (expected " << expected << ")\n"
                          << "       " << tc.fen << '\n';
            } else if (verbose) {
                std::cout << "  ok   depth " << depth << ": " << got
                          << "  [" << elapsedMs << " ms]\n";
            }
        }
    }

    const auto suiteMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now() - suiteStart).count();
    std::cout << "\nperft suite " << (allPassed ? "PASSED" : "FAILED") << " — "
              << totalNodes << " nodes in " << suiteMs << " ms\n";
    return allPassed;
}

int runPerftCommand(int argc, char* argv[]) {
    pieces::initMagicBitboards();

    // argv is [0]=./chess [1]=perft [2]=<depth>|divide|suite ...
    if (argc < 3) {
        printUsage();
        return 1;
    }

    const std::string_view sub = argv[2];

    if (sub == "suite") {
        const int maxDepth = (argc >= 4) ? std::atoi(argv[3]) : 4;
        if (maxDepth < 1) {
            printUsage();
            return 1;
        }
        return runPerftSuite(maxDepth) ? 0 : 1;
    }

    const bool divide = (sub == "divide");
    const int depthIndex = divide ? 3 : 2;
    if (argc <= depthIndex) {
        printUsage();
        return 1;
    }

    const int depth = std::atoi(argv[depthIndex]);
    if (depth < 0 || depth > 15) {
        std::cerr << "perft: depth must be in [0, 15]\n";
        return 1;
    }

    const std::string fen = joinArgs(argc, argv, depthIndex + 1);
    chess::Board b = fen.empty() ? chess::Board{} : chess::Board(fen);
    if (!boardLooksValid(b)) {
        std::cerr << "perft: could not parse FEN \"" << fen << "\"\n";
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    const uint64_t nodes = divide ? perftDivide(b, depth) : perft(b, depth);
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    if (!divide) std::cout << "nodes: " << nodes << '\n';
    std::cout << "time: " << elapsedMs << " ms";
    if (elapsedMs > 0) std::cout << "  (" << (nodes / static_cast<uint64_t>(elapsedMs)) << " knps)";
    std::cout << '\n';
    return 0;
}

} // namespace engine
