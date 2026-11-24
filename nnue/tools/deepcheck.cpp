// Checks the C++ forward of the deep net against the Rust oracle
// (nnue/trainer/src/bin/sanity_deep.rs), and the scalar forward against the
// SIMD one. A standalone program, not linked into ./chess.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/deepcheck.cpp nnue/nnue_deep.cpp -o /tmp/deepcheck
//   /tmp/deepcheck <net.bin> [fen]...
//
// The accumulator comes from position.hpp, shared with the other tools.

#include "position.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace NNUE::Deep;
using namespace NNUE::Deep::tools;

namespace {

struct Case { const char* label; const char* fen; };

constexpr Case DEFAULT_CASES[] = {
    {"startpos (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"},
    {"startpos (b)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"},
    {"wN e4 extra (w)", "rnbqkbnr/pppppppp/8/8/4N3/8/PPPPPPPP/RNBQKBNR w - - 0 1"},
    {"bN e5 extra (b) [must equal previous]", "rnbqkbnr/pppppppp/8/4n3/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"},
    {"stm up a queen (w)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"},
    {"stm down a queen (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1"},
    {"middlegame ~24 pieces (w)", "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1"},
    {"KRPvKR (w)", "8/8/4k3/8/2r5/2K5/4P3/3R4 w - - 0 1"},
    {"KQvK (w)", "8/8/8/4k3/8/8/8/KQ6 w - - 0 1"},
    {"both castled short (w)", "r4rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 1"},
    {"white long castle vs e8 king (w)", "r3kb1r/ppp2ppp/2n1bn2/3qp3/8/2NP1N2/PPPBQPPP/2KR3R w kq - 0 1"},
    {"kings on 2nd rank (w)", "8/1k3ppp/1p6/p1p5/P1P5/1P4P1/1K3P1P/8 w - - 0 1"},
    {"active kings endgame (w)", "8/8/3k4/8/2p5/2P1K3/8/8 w - - 0 1"},
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: deepcheck <net.bin> [fen]...\n");
        return 2;
    }
    static NetworkDeep net;
    if (!loadFromFile(argv[1], net) || net.fileThreats) {
        std::fprintf(stderr, "cannot load %s (threat inputs are not supported here)\n", argv[1]);
        return 1;
    }
    std::printf("%s: layout OK (%d output buckets, %zu B payload)\n", argv[1],
                net.fileBuckets, payloadBytes(net.fileBuckets, false));

    std::vector<Case> cases;
    std::vector<std::string> owned;
    if (argc > 2) {
        for (int i = 2; i < argc; ++i) owned.emplace_back(argv[i]);
        for (const auto& f : owned) cases.push_back({f.c_str(), f.c_str()});
    } else {
        for (const auto& c : DEFAULT_CASES) cases.push_back(c);
    }

    alignas(64) int16_t us[HIDDEN], them[HIDDEN];
    NormPos pos;
    for (const auto& c : cases) {
        if (!parseFenNormPos(c.fen, pos)) {
            std::fprintf(stderr, "invalid FEN: %s\n", c.fen);
            return 1;
        }
        accumulatePos(net, nullptr, pos, us, them);
        const int32_t sc = forwardScalar(net, us, them, pos.outBucket);
        const int32_t sd = forwardSimd(net, us, them, pos.outBucket);
        if (sc != sd) {
            std::fprintf(stderr, "scalar/SIMD MISMATCH on '%s': %d vs %d\n", c.label, sc, sd);
            return 1;
        }
        std::printf("%s: %d cp (stm)\n", c.label, sc);
    }
    return 0;
}
