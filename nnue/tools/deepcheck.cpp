// Confronto fra il forward C++ della rete profonda e l'oracolo Rust
// (nnue/trainer/src/bin/sanity_deep.rs). Programma a se' stante: non fa parte
// del motore e non viene linkato in ./chess.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/deepcheck.cpp nnue/nnue_deep.cpp -o /tmp/deepcheck
//   /tmp/deepcheck <net.bin> [fen]...
//
// La costruzione dell'accumulatore da FEN sta in fen_accumulator.hpp, condivisa
// con gli altri strumenti: e' una traduzione letterale di sanity_deep.rs, e nel
// motore quel lavoro lo fa Board tramite accumulator.hpp.

#include "fen_accumulator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace NNUE::Deep;
using NNUE::Deep::tools::accumulate;

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


// --- misura: costo del forward profondo contro quello a un layer ---------
// Il secondo e' una copia locale dell'aritmetica di nnue.cpp (SCReLU su
// 2*HIDDEN i16), non il codice del motore: serve un confronto fra due cicli
// sugli stessi dati, non un profilo del binario di produzione.
#include <chrono>
#include <immintrin.h>
namespace {

// Copia dell'aritmetica AVX2 di nnue.cpp::forwardHalf (mullo + madd), NON una
// versione scalare: confrontare il forward profondo vettoriale con un
// riferimento scalare gonfierebbe il risultato di parecchio.
int32_t singleLayerForward(const int16_t* accStm, const int16_t* accNtm,
                           const int16_t* w) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(static_cast<int16_t>(QA));
    int64_t out = 0;
    for (int half = 0; half < 2; ++half) {
        const int16_t* a  = half == 0 ? accStm : accNtm;
        const int16_t* ww = w + half * HIDDEN;
        __m256i sum = _mm256_setzero_si256();
        for (int i = 0; i < HIDDEN; i += 16) {
            const __m256i v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            const __m256i t  = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
            const __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ww + i));
            const __m256i pr = _mm256_mullo_epi16(t, wi);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(pr, t));
        }
        const __m128i lo = _mm256_castsi256_si128(sum);
        const __m128i hi = _mm256_extracti128_si256(sum, 1);
        __m128i sN = _mm_add_epi32(lo, hi);
        sN = _mm_add_epi32(sN, _mm_shuffle_epi32(sN, 0x4E));
        sN = _mm_add_epi32(sN, _mm_shuffle_epi32(sN, 0xB1));
        out += _mm_cvtsi128_si32(sN);
    }
    return static_cast<int32_t>(out / (QA * QB));
}

void bench(const NetworkDeep& net, const std::vector<int16_t>& us,
           const std::vector<int16_t>& them, int bucket) {
    constexpr int N = 100000;
    constexpr int ROUNDS = 7;
    std::vector<int16_t> w(2 * HIDDEN);
    for (int i = 0; i < 2 * HIDDEN; ++i) w[i] = static_cast<int16_t>((i * 37) % 127 - 63);

    // MINIMO su piu' giri, non media: su una macchina non quieta il rumore puo'
    // solo RALLENTARE una misura, mai accelerarla, quindi il minimo e' la stima
    // meno contaminata. I due percorsi si alternano dentro lo stesso giro cosi'
    // un rallentamento transitorio li colpisce entrambi.
    volatile int32_t sink = 0;
    double a = 1e18, b = 1e18;
    for (int r = 0; r < ROUNDS; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) sink = singleLayerForward(us.data(), them.data(), w.data());
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) sink = forwardSimd(net, us.data(), them.data(), bucket);
        auto t2 = std::chrono::steady_clock::now();
        a = std::min(a, std::chrono::duration<double, std::nano>(t1 - t0).count() / N);
        b = std::min(b, std::chrono::duration<double, std::nano>(t2 - t1).count() / N);
    }
    (void)sink;
    std::printf("\nforward a un layer (riferimento locale): %6.1f ns/eval\n", a);
    std::printf("forward profondo (SIMD):                %6.1f ns/eval   = %.2fx\n", b, b / a);
    std::printf("percorso VNNI: %s\n", hasVnniPath() ? "attivo" : "NON attivo (fallback i16)");
}

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
        const int32_t sc = forwardScalar(net, accUs.data(), accThem.data(), bucket);
        const int32_t sd = forwardSimd(net, accUs.data(), accThem.data(), bucket);
        if (sc != sd) {
            std::fprintf(stderr, "DIVERGENZA scalare/SIMD su '%s': %d vs %d\n", c.label, sc, sd);
            return 1;
        }
        std::printf("%s: %d cp (stm)\n", c.label, sc);
    }
    if (const char* e = std::getenv("DEEPCHECK_BENCH"); e != nullptr && e[0] == '1') {
        int bucket = 0;
        accumulate(net, DEFAULT_CASES[6].fen, accUs, accThem, bucket);
        bench(net, accUs, accThem, bucket);
    }
    return 0;
}
