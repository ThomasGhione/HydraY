// Valutazione statica in blocco della rete a un layer, una FEN per riga da
// stdin, "eval<TAB>bucket<TAB>fen" su stdout. Serve all'audit di copertura:
// misurare dove la rete sbaglia richiede decine di migliaia di posizioni, e
// sanity.rs prende le FEN da argv, una manciata alla volta.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/evalfens.cpp -o /tmp/evalfens
//   /tmp/evalfens <net.bin> < fens.txt
//
// Il forward e' la trascrizione di NNUE::evaluate (nnue.cpp): SCReLU sulle due
// prospettive, somma, /QA, bias, *SCALE/(QA*QB). L'accumulatore viene da
// fen_accumulator.hpp, gia' condivisa con deepcheck e reorder.
//
// La rete a un layer viene letta ANCHE come NetworkDeep per il solo prefisso
// l0: i due formati hanno l0w/l0b identici per costruzione, ed e' esattamente
// cio' su cui il motore fa affidamento (vedi gli static_assert in nnue.cpp).

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
        std::fprintf(stderr, "uso: evalfens <net.bin> < fens.txt\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
    if (!in) { std::fprintf(stderr, "non apribile: %s\n", argv[1]); return 1; }
    const auto size = static_cast<size_t>(in.tellg());
    if (size < NNUE::NETWORK_PAYLOAD_BYTES) {
        std::fprintf(stderr, "taglia %zu: non e' una rete a un layer (attesi >= %zu)\n",
                     size, NNUE::NETWORK_PAYLOAD_BYTES);
        return 1;
    }
    in.seekg(0);

    auto buf = std::unique_ptr<NNUE::Network>(new NNUE::Network);
    if (!in.read(reinterpret_cast<char*>(buf.get()),
                 static_cast<std::streamsize>(NNUE::NETWORK_PAYLOAD_BYTES))) {
        std::fprintf(stderr, "lettura corta\n");
        return 1;
    }
    const NNUE::Network& net = *buf;
    // Vista sul solo prefisso l0, che i due formati condividono byte per byte.
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
