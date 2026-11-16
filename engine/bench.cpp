#include "bench.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "engine.hpp"

namespace engine {

namespace {

// Same positions, in the same order, as BENCH6_POS in script/engine_driver.sh.
struct BenchPosition {
    const char* name;
    const char* fen;
};

constexpr std::array<BenchPosition, 6> BENCH_POSITIONS = {{
    {"startpos",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
    {"kiwipete",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"},
    {"kp-endgame", "8/2k5/3p4/p2P1p2/P2P1P2/8/8/4K3 w - - 0 1"},
    {"midgame",    "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1"},
    {"tactical",   "r1bqk2r/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 b kq - 0 1"},
    {"open",       "r2qkbnr/ppp2ppp/2np4/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"},
}};

constexpr int DEFAULT_BENCH_DEPTH = 14;

} // namespace

int runBenchCommand(int argc, char* argv[]) {
    // argv is [0]=./chess [1]=bench [2]=<depth>
    const int depth = (argc >= 3) ? std::atoi(argv[2]) : DEFAULT_BENCH_DEPTH;
    if (depth < 1 || depth > MAX_ITERATION_DEPTH) {
        std::fprintf(stderr, "usage: chess bench [depth 1-%d]\n", MAX_ITERATION_DEPTH);
        return 1;
    }

    // Threads stays at 1 and Hash at its default: Lazy SMP is not reproducible.
    Engine engine;
    time::Limits limits;
    limits.maxDepth = depth;

    uint64_t totalNodes = 0;
    std::chrono::steady_clock::duration elapsed{};
    for (const BenchPosition& position : BENCH_POSITIONS) {
        engine.reset();
        engine.board = chess::Board(position.fen);
        engine.prepareSearch();
        const auto start = std::chrono::steady_clock::now();
        engine.search(limits);
        elapsed += std::chrono::steady_clock::now() - start;
        const uint64_t nodes = engine.searchRuntime.nodesSearched;
        totalNodes += nodes;
        std::printf("%-12s %llu nodes\n", position.name, static_cast<unsigned long long>(nodes));
    }

    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    const uint64_t nps = us > 0 ? totalNodes * 1'000'000 / static_cast<uint64_t>(us) : 0;
    std::printf("%llu nodes %llu nps\n", static_cast<unsigned long long>(totalNodes),
                static_cast<unsigned long long>(nps));
    return 0;
}

} // namespace engine
