// Bulk static evaluation with the single-layer net: one FEN per line on stdin,
// "eval<TAB>bucket<TAB>fen" on stdout. The coverage audit needs it: measuring
// where the net is wrong takes tens of thousands of positions, and sanity.rs
// takes its FENs from argv, a handful at a time.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/evalfens.cpp -o /tmp/evalfens
//   /tmp/evalfens <net.bin> < fens.txt
//
// The forward is a transcription of NNUE::evaluate (nnue.cpp): SCReLU over the
// two perspectives, sum, /QA, bias, *SCALE/(QA*QB). The accumulator comes from
// fen_accumulator.hpp, already shared with deepcheck and reorder.
//
// The single-layer net is ALSO read as a NetworkDeep for the l0 prefix alone:
// the two formats have identical l0w/l0b by construction, which is exactly what
// the engine relies on (see the static_asserts in nnue.cpp).

#include "fen_accumulator.hpp"
#include "../network.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int32_t forwardHalf(const int16_t* acc, const int16_t* w) {
    int32_t s = 0;
    for (int i = 0; i < NNUE::HIDDEN; ++i) {
        const int32_t t = std::clamp<int32_t>(acc[i], 0, NNUE::QA);
        s += t * t * static_cast<int32_t>(w[i]);
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: evalfens <net.bin> < fens.txt\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "cannot open: %s\n", argv[1]); return 1; }
    const auto size = static_cast<size_t>(in.tellg());
    if (size < NNUE::NETWORK_PAYLOAD_BYTES) {
        std::fprintf(stderr, "size %zu: not a single-layer net (expected >= %zu)\n",
                     size, NNUE::NETWORK_PAYLOAD_BYTES);
        return 1;
    }
    in.seekg(0);

    auto buf = std::unique_ptr<NNUE::Network>(new NNUE::Network);
    if (!in.read(reinterpret_cast<char*>(buf.get()),
                 static_cast<std::streamsize>(NNUE::NETWORK_PAYLOAD_BYTES))) {
        std::fprintf(stderr, "short read\n");
        return 1;
    }
    const NNUE::Network& net = *buf;
    // A view on the l0 prefix alone, which both formats share byte for byte.
    const auto& deepView = *reinterpret_cast<const NNUE::Deep::NetworkDeep*>(buf.get());

    std::vector<int16_t> accUs, accThem;
    std::string fen;
    while (std::getline(std::cin, fen)) {
        if (fen.empty()) continue;
        int bucket = 0;
        if (!NNUE::Deep::tools::accumulate(deepView, fen, accUs, accThem, bucket)) {
            std::printf("NA\tNA\t%s\n", fen.c_str());
            continue;
        }
        int32_t out = forwardHalf(accUs.data(), net.outputWeights[bucket][0])
                    + forwardHalf(accThem.data(), net.outputWeights[bucket][1]);
        out /= NNUE::QA;
        out += net.outputBias[bucket];
        out = out * NNUE::SCALE / (NNUE::QA * NNUE::QB);
        std::printf("%d\t%d\t%s\n", out, bucket, fen.c_str());
    }
    return 0;
}
