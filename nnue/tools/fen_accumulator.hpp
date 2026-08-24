#pragma once

// Costruzione dell'accumulatore da una FEN, per gli strumenti fuori dal motore
// (deepcheck, reorder). E' una TRADUZIONE LETTERALE di sanity_deep.rs: nel
// motore quel lavoro lo fa Board tramite accumulator.hpp, che non cambia perche'
// l0 e' identico alla rete a un layer.
//
// Sta in un'intestazione condivisa e non copiata in ogni strumento: erano due
// copie e stavano per diventare tre. Se sbaglia, i confronti con l'oracolo
// falliscono e lo si scopre subito.

#include "../network_deep.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace NNUE::Deep::tools {

constexpr int BUCKET_LAYOUT[32] = {
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
};

int kingBucket(int ksq) {
    static constexpr int FILE_FOLD[8] = {0, 1, 2, 3, 3, 2, 1, 0};
    return BUCKET_LAYOUT[(ksq / 8) * 4 + FILE_FOLD[ksq % 8]];
}

struct Piece { int type; bool black; int sq; };

// Ritorna false su FEN malformata (re mancante).
bool accumulate(const NetworkDeep& net, const std::string& fen,
                std::vector<int16_t>& accUs, std::vector<int16_t>& accThem,
                int& outBucket) {
    const size_t sp = fen.find(' ');
    const std::string board = fen.substr(0, sp);
    const bool blackToMove = sp != std::string::npos && fen[sp + 1] == 'b';

    std::vector<Piece> pieces;
    int wk = -1, bk = -1;
    int rank = 7, file = 0;
    for (const char c : board) {
        if (c == '/') { --rank; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        int type;
        switch (std::tolower(static_cast<unsigned char>(c))) {
            case 'p': type = 0; break; case 'n': type = 1; break;
            case 'b': type = 2; break; case 'r': type = 3; break;
            case 'q': type = 4; break; case 'k': type = 5; break;
            default: return false;
        }
        const bool black = std::islower(static_cast<unsigned char>(c)) != 0;
        const int sq = rank * 8 + file;
        if (type == 5) (black ? bk : wk) = sq;
        pieces.push_back({type, black, sq});
        ++file;
    }
    if (wk < 0 || bk < 0) return false;

    const auto params = [&](bool fromBlackView, int& base, int& flip) {
        const int own = fromBlackView ? (bk ^ 56) : wk;
        flip = (own % 8) > 3 ? 7 : 0;
        base = INPUTS * kingBucket(own);
    };
    int usBase, usFlip, themBase, themFlip;
    params(blackToMove, usBase, usFlip);
    params(!blackToMove, themBase, themFlip);

    accUs.assign(net.l0b, net.l0b + HIDDEN);
    accThem.assign(net.l0b, net.l0b + HIDDEN);

    for (const auto& p : pieces) {
        const struct { std::vector<int16_t>* acc; bool view; int base; int flip; } sides[2] = {
            {&accUs,   blackToMove,  usBase,   usFlip},
            {&accThem, !blackToMove, themBase, themFlip},
        };
        for (const auto& s : sides) {
            const bool isOpp = (p.black != s.view);
            const int sqView = s.view ? (p.sq ^ 56) : p.sq;
            const int feat768 = (isOpp ? 384 : 0) + p.type * 64 + sqView;
            const int feature = s.base + (feat768 ^ s.flip);
            const int16_t* w = net.l0w[feature];
            for (int i = 0; i < HIDDEN; ++i) (*s.acc)[i] = static_cast<int16_t>((*s.acc)[i] + w[i]);
        }
    }

    const int divisor = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS; // div_ceil
    outBucket = (static_cast<int>(pieces.size()) - 2) / divisor;
    return true;
}

} // namespace NNUE::Deep::tools
