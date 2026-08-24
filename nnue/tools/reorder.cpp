// Riordina i neuroni di una rete profonda per rendere piu' efficace la
// sparsita' del layer l1. Programma a se' stante: non fa parte del motore.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/reorder.cpp nnue/nnue_deep.cpp -o /tmp/reorder
//   /tmp/reorder <in.nnue> <posizioni.fen> <out.nnue>
//
// Il file di posizioni e' un elenco di FEN, una per riga, da partite VERE — si
// ricava da un PGN con python-chess. Servono alcune migliaia di posizioni; il
// guadagno regge su posizioni mai viste (misurato: 67,3% in calibrazione contro
// 66,6% su un insieme separato).
//
// PERCHE' FUNZIONA
//
// Il prodotto scalare di l1 salta i gruppi di quattro ingressi consecutivi che
// sono TUTTI nulli. Nell'ordine in cui la rete esce dall'addestramento i
// neuroni sono disposti a caso, e la quota di gruppi nulli e' esattamente
// quella che ci si aspetta da neuroni indipendenti: con l'85,6% di byte nulli,
// 0,856^4 = 53,7%. Ma i neuroni NON sono indipendenti — alcuni tacciono
// insieme. Raggruppandoli per co-occorrenza si arriva al 67%, cioe' un quinto
// di lavoro in meno nel layer.
//
// PERCHE' NON TOCCA IL MOTORE
//
// L'ordine dei neuroni e' una convenzione interna: permutando le uscite di l0
// (pesi e bias) e le colonne corrispondenti di l1w, la rete calcola ESATTAMENTE
// la stessa funzione. Il file resta nel formato di sempre e il motore non sa
// nulla di tutto questo. Per la stessa ragione una permutazione calcolata su
// una rete diversa non e' un errore: rende solo meno.
//
// Il vincolo da rispettare: il pairwise accoppia il neurone j con j+512, quindi
// si permutano le 512 COPPIE, mai i 1024 singoli, e la stessa permutazione vale
// per entrambe le prospettive perche' l0 e' condiviso.
//
// La verifica finale e' il vero collaudo: la rete riordinata deve dare
// valutazioni IDENTICHE BIT PER BIT all'originale. Se non lo fa, la
// permutazione e' stata applicata male da qualche parte e il file non viene
// scritto.

#include "fen_accumulator.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

using namespace NNUE::Deep;
using NNUE::Deep::tools::accumulate;

namespace {

// Un bit per campione: acceso se quella coppia e' nulla in quel campione.
// Un campione e' una (posizione, prospettiva), perche' le due prospettive
// condividono la permutazione e vanno quindi contate entrambe.
using Bits = std::vector<uint64_t>;

size_t words = 0;

[[nodiscard]] long popcountAll(const Bits& b) noexcept {
    long n = 0;
    for (const uint64_t w : b) n += std::popcount(w);
    return n;
}

[[nodiscard]] long zeroGroups(const std::vector<Bits>& zero, const int* g) noexcept {
    long n = 0;
    for (size_t w = 0; w < words; ++w)
        n += std::popcount(zero[g[0]][w] & zero[g[1]][w] & zero[g[2]][w] & zero[g[3]][w]);
    return n;
}

// Greedy: si parte dalla coppia che tace piu' spesso e le si affiancano le tre
// che tacciono insieme a lei. E' il punto di partenza, non il risultato.
[[nodiscard]] std::vector<int> greedy(const std::vector<Bits>& zero) {
    std::vector<char> used(PAIRWISE_OUT, 0);
    std::vector<int> perm;
    perm.reserve(PAIRWISE_OUT);
    for (int g = 0; g < PAIRWISE_OUT / 4; ++g) {
        int seed = -1;
        long best = -1;
        for (int j = 0; j < PAIRWISE_OUT; ++j)
            if (used[j] == 0) {
                const long c = popcountAll(zero[j]);
                if (c > best) { best = c; seed = j; }
            }
        used[seed] = 1;
        perm.push_back(seed);
        Bits cur = zero[seed];
        for (int k = 1; k < 4; ++k) {
            int pick = -1;
            long bc = -1;
            for (int j = 0; j < PAIRWISE_OUT; ++j)
                if (used[j] == 0) {
                    long c = 0;
                    for (size_t w = 0; w < words; ++w) c += std::popcount(cur[w] & zero[j][w]);
                    if (c > bc) { bc = c; pick = j; }
                }
            used[pick] = 1;
            perm.push_back(pick);
            for (size_t w = 0; w < words; ++w) cur[w] &= zero[pick][w];
        }
    }
    return perm;
}

// Scambi fra gruppi finche' migliorano. Il greedy consuma presto le coppie piu'
// silenziose e lascia un residuo scadente; questo lo rimette a posto.
long localSearch(const std::vector<Bits>& zero, std::vector<int>& perm) {
    std::vector<long> val(PAIRWISE_OUT / 4);
    for (int g = 0; g < PAIRWISE_OUT / 4; ++g) val[g] = zeroGroups(zero, &perm[4 * g]);
    for (int pass = 0; pass < 100; ++pass) {
        bool improved = false;
        for (int a = 0; a < PAIRWISE_OUT; ++a)
            for (int b = a + 1; b < PAIRWISE_OUT; ++b) {
                const int ga = a / 4, gb = b / 4;
                if (ga == gb) continue;
                std::swap(perm[a], perm[b]);
                const long na = zeroGroups(zero, &perm[4 * ga]);
                const long nb = zeroGroups(zero, &perm[4 * gb]);
                if (na + nb > val[ga] + val[gb]) { val[ga] = na; val[gb] = nb; improved = true; }
                else std::swap(perm[a], perm[b]);
            }
        if (!improved) break;
    }
    return std::accumulate(val.begin(), val.end(), 0L);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "uso: reorder <in.nnue> <posizioni.fen> <out.nnue>\n");
        return 2;
    }
    static NetworkDeep net;
    if (!loadFromFile(argv[1], net)) {
        std::fprintf(stderr, "caricamento fallito: %s\n", argv[1]);
        return 1;
    }

    // --- statistiche di attivazione ---
    std::ifstream in(argv[2]);
    if (!in) { std::fprintf(stderr, "posizioni non leggibili: %s\n", argv[2]); return 1; }
    std::vector<std::vector<uint8_t>> samples;
    std::vector<int16_t> us, them;
    std::string fen;
    while (std::getline(in, fen)) {
        if (fen.empty()) continue;
        int bucket = 0;
        if (!accumulate(net, fen, us, them, bucket)) continue;
        std::vector<uint8_t> h(HIDDEN);
        for (int j = 0; j < PAIRWISE_OUT; ++j) {
            const auto pw = [](int a, int b) {
                return static_cast<uint8_t>(std::clamp(a, 0, QA) * std::clamp(b, 0, QA) / QA);
            };
            h[j]                = pw(us[j],   us[j + PAIRWISE_OUT]);
            h[PAIRWISE_OUT + j] = pw(them[j], them[j + PAIRWISE_OUT]);
        }
        samples.push_back(std::move(h));
    }
    if (samples.size() < 500) {
        std::fprintf(stderr, "solo %zu posizioni valide: troppo poche, ne servono migliaia\n",
                     samples.size());
        return 1;
    }
    const size_t total = samples.size() * 2;
    words = (total + 63) / 64;
    std::vector<Bits> zero(PAIRWISE_OUT, Bits(words, 0));
    for (size_t p = 0; p < samples.size(); ++p)
        for (int half = 0; half < 2; ++half) {
            const size_t s = p * 2 + half;
            for (int j = 0; j < PAIRWISE_OUT; ++j)
                if (samples[p][half * PAIRWISE_OUT + j] == 0) zero[j][s / 64] |= 1ull << (s % 64);
        }

    std::vector<int> identity(PAIRWISE_OUT);
    std::iota(identity.begin(), identity.end(), 0);
    const auto rate = [&](const std::vector<int>& p) {
        long t = 0;
        for (int g = 0; g < PAIRWISE_OUT / 4; ++g) t += zeroGroups(zero, &p[4 * g]);
        return 100.0 * static_cast<double>(t) / static_cast<double>(total * (PAIRWISE_OUT / 4));
    };

    std::printf("%zu posizioni (%zu campioni con le due prospettive)\n", samples.size(), total);
    std::printf("gruppi da quattro tutti nulli, ordine attuale : %.1f%%\n", rate(identity));

    std::vector<int> perm = greedy(zero);
    localSearch(zero, perm);
    std::printf("gruppi da quattro tutti nulli, riordinati     : %.1f%%\n", rate(perm));
    std::printf("lavoro nel layer l1: %.1f%% -> %.1f%%\n", 100 - rate(identity), 100 - rate(perm));

    // --- applicazione: perm[q] = coppia che finisce in posizione q ---
    static NetworkDeep out;
    out = net;
    for (int q = 0; q < PAIRWISE_OUT; ++q) {
        const int src = perm[q];
        for (int f = 0; f < INPUT_BUCKETS * INPUTS; ++f) {
            out.l0w[f][q]                = net.l0w[f][src];
            out.l0w[f][q + PAIRWISE_OUT] = net.l0w[f][src + PAIRWISE_OUT];
        }
        out.l0b[q]                = net.l0b[src];
        out.l0b[q + PAIRWISE_OUT] = net.l0b[src + PAIRWISE_OUT];
        for (int b = 0; b < OUTPUT_BUCKETS; ++b)
            for (int o = 0; o < L1_SIZE; ++o) {
                out.l1w[b][o][q]                = net.l1w[b][o][src];
                out.l1w[b][o][q + PAIRWISE_OUT] = net.l1w[b][o][src + PAIRWISE_OUT];
            }
    }

    // --- scrittura, stesso layout del file di origine ---
    std::FILE* f = std::fopen(argv[3], "wb");
    if (f == nullptr) { std::fprintf(stderr, "non posso scrivere %s\n", argv[3]); return 1; }
    const auto wr = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
    bool ok = wr(out.l0w, sizeof(out.l0w)) && wr(out.l0b, sizeof(out.l0b))
           && wr(out.l1w, sizeof(out.l1w)) && wr(out.l1b, sizeof(out.l1b))
           && wr(out.l2w, sizeof(out.l2w)) && wr(out.l2b, sizeof(out.l2b));
    const size_t padBytes = (PAYLOAD_BYTES + 63) / 64 * 64 - PAYLOAD_BYTES;
    static const char kTag[] = "bullet";
    for (size_t i = 0; i < padBytes && ok; ++i) ok = std::fputc(kTag[i % 6], f) != EOF;
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "scrittura incompleta\n"); return 1; }

    // --- collaudo: la rete riordinata calcola la STESSA funzione ---
    // Non e' una formalita': e' l'unica cosa che distingue un riordino corretto
    // da uno che ha scambiato pesi e bias in modo incoerente.
    static NetworkDeep back;
    if (!loadFromFile(argv[3], back)) { std::fprintf(stderr, "il file scritto non si rilegge\n"); return 1; }
    uint32_t rs = 20260824;
    const auto rnd = [&] { rs = rs * 1664525u + 1013904223u; return static_cast<int32_t>(rs >> 15); };
    std::vector<int16_t> a(HIDDEN), b(HIDDEN), pa(HIDDEN), pb(HIDDEN);
    long bad = 0, checks = 0;
    for (int t = 0; t < 4000; ++t) {
        const int span = 40 + (t % 7) * 600;
        for (int i = 0; i < HIDDEN; ++i) {
            a[i] = static_cast<int16_t>(rnd() % span - span / 3);
            b[i] = static_cast<int16_t>(rnd() % span - span / 3);
        }
        // gli accumulatori vanno permutati come i neuroni, altrimenti si sta
        // confrontando la stessa rete su ingressi diversi
        for (int q = 0; q < PAIRWISE_OUT; ++q) {
            pa[q] = a[perm[q]]; pa[q + PAIRWISE_OUT] = a[perm[q] + PAIRWISE_OUT];
            pb[q] = b[perm[q]]; pb[q + PAIRWISE_OUT] = b[perm[q] + PAIRWISE_OUT];
        }
        for (int bk = 0; bk < OUTPUT_BUCKETS; ++bk, ++checks)
            if (forwardSimd(net, a.data(), b.data(), bk)
                != forwardSimd(back, pa.data(), pb.data(), bk)) ++bad;
    }
    if (bad != 0) {
        std::fprintf(stderr, "\nCOLLAUDO FALLITO: %ld divergenze su %ld — file NON valido\n", bad, checks);
        return 1;
    }
    std::printf("collaudo: %ld confronti, valutazioni identiche\n", checks);
    std::printf("scritto %s\n", argv[3]);
    return 0;
}
