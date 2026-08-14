// Confronto fra il forward C++ della rete profonda e l'oracolo Rust
// (nnue/trainer/src/bin/sanity_deep.rs). Programma a se' stante: non fa parte
// del motore e non viene linkato in ./chess.
//
//   g++ -std=c++23 -O2 -march=native nnue/deepcheck.cpp nnue/nnue_deep.cpp -o /tmp/deepcheck
//   /tmp/deepcheck <net.bin> [fen]...
//
// La costruzione dell'accumulatore da FEN qui sotto e' una TRADUZIONE LETTERALE
// di sanity_deep.rs, ed esiste solo per questo test: nel motore quel lavoro lo
// fa Board tramite accumulator.hpp, che non cambia (l0 e' identico alla rete a
// un layer). Duplicarla qui e' accettabile perche' il bersaglio del test e' il
// FORWARD; se sbagliassi anche questa parte, il confronto con l'oracolo
// fallirebbe comunque e lo scoprirei.

#include "network_deep.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace NNUE::Deep;

namespace {

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
    {"active kings endgame (w)", "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1"},
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "uso: deepcheck <net.bin> [fen]...\n");
        return 2;
    }
    static NetworkDeep net;
    if (!loadFromFile(argv[1], net)) {
        std::fprintf(stderr, "caricamento fallito: %s\n", argv[1]);
        return 1;
    }
    std::printf("%s: layout OK (%zu B payload)\n", argv[1], PAYLOAD_BYTES);

    std::vector<Case> cases;
    std::vector<std::string> owned;
    if (argc > 2) {
        for (int i = 2; i < argc; ++i) owned.emplace_back(argv[i]);
        for (const auto& f : owned) cases.push_back({f.c_str(), f.c_str()});
    } else {
        for (const auto& c : DEFAULT_CASES) cases.push_back(c);
    }

    std::vector<int16_t> accUs, accThem;
    for (const auto& c : cases) {
        int bucket = 0;
        if (!accumulate(net, c.fen, accUs, accThem, bucket)) {
            std::fprintf(stderr, "FEN non valida: %s\n", c.fen);
            return 1;
        }
        std::printf("%s: %d cp (stm)\n", c.label,
                    forwardScalar(net, accUs.data(), accThem.data(), bucket));
    }
    return 0;
}
