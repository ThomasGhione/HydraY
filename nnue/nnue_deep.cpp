#include "network_deep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace NNUE::Deep {

namespace {

// SCReLU in virgola mobile, come nel grafo di bullet: clamp a [0,1] e quadrato.
[[nodiscard]] inline float screlu(float x) noexcept {
    const float y = std::clamp(x, 0.0f, 1.0f);
    return y * y;
}

// CReLU + prodotto a coppie: 1024 accumulatori i16 -> 512 valori in [0, QA].
// La divisione per QA e' intera e troncata, come in sanity_deep.rs.
inline void pairwise(const int16_t* acc, int32_t* out) noexcept {
    for (int j = 0; j < PAIRWISE_OUT; ++j) {
        const int32_t a = std::clamp<int32_t>(acc[j], 0, QA);
        const int32_t b = std::clamp<int32_t>(acc[j + PAIRWISE_OUT], 0, QA);
        out[j] = a * b / QA;
    }
}

} // namespace

int32_t forwardScalar(const NetworkDeep& net,
                      const int16_t* accStm,
                      const int16_t* accNtm,
                      int outputBucket) noexcept {
    int32_t hl1[HIDDEN];
    pairwise(accStm, hl1);
    pairwise(accNtm, hl1 + PAIRWISE_OUT);

    float a1[L1_SIZE];
    for (int o = 0; o < L1_SIZE; ++o) {
        const int8_t* row = net.l1w[outputBucket][o];
        int32_t sum = 0;
        for (int i = 0; i < HIDDEN; ++i) {
            sum += hl1[i] * static_cast<int32_t>(row[i]);
        }
        a1[o] = screlu(static_cast<float>(sum) / static_cast<float>(QA * QB)
                       + net.l1b[outputBucket][o]);
    }

    float y = net.l2b[outputBucket];
    for (int o = 0; o < L1_SIZE; ++o) {
        y += a1[o] * net.l2w[outputBucket][o];
    }
    return static_cast<int32_t>(std::lround(y * static_cast<float>(SCALE)));
}

// ---------------------------------------------------------------------------
// Forward vettoriale
//
// Il pairwise resta su corsie a 16 bit per tutto il percorso, e non e' una
// scelta di comodo:
//   - a e b stanno in [0, QA] = [0, 255], quindi a*b <= 65025, che ci sta
//     ESATTAMENTE in un u16. `mullo_epi16` ne restituisce il valore giusto;
//   - la divisione per 255 diventa `mulhi_epu16(x, 32897) >> 7`, che e'
//     floor(x*32897 / 2^23) == x/255 per ogni x in [0, 65025]. Verificato sui
//     due estremi che contano: 254 -> 0 e 65025 -> 255.
//
// Per il prodotto scalare di l1 la strada ovvia sarebbe `maddubs_epi16`, che
// macina 32 valori per istruzione. NON si puo' usare: satura a i16, e qui i
// suoi addendi arrivano a 255*127*2 = 64770, ben oltre 32767. Saturerebbe in
// silenzio, cioe' la rete giocherebbe peggio senza che nessun test se ne
// accorga. Restano due strade esatte:
//   - AVX-VNNI `dpbusd`: accumula in i32, 32 moltiplicazioni per istruzione;
//   - AVX2 `madd_epi16`: accumula in i32, 16 per istruzione.
// La prima e' il doppio piu' veloce ma richiede una CPU che la supporti,
// quindi c'e' il fallback e i due percorsi vanno confrontati fra loro.
// ---------------------------------------------------------------------------

#if defined(__AVX2__)
#include <immintrin.h>

namespace {

// 32 accumulatori i16 -> 32 valori pairwise in [0, QA], su corsie i16.
inline __m256i pairwiseBlock(const int16_t* acc, int j) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(static_cast<int16_t>(QA));
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + j));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + j + PAIRWISE_OUT));
    const __m256i a  = _mm256_min_epi16(_mm256_max_epi16(lo, zero), qa);
    const __m256i b  = _mm256_min_epi16(_mm256_max_epi16(hi, zero), qa);
    const __m256i p  = _mm256_mullo_epi16(a, b);              // <= 65025, esatto in u16
    const __m256i m  = _mm256_mulhi_epu16(p, _mm256_set1_epi16(static_cast<int16_t>(32897)));
    return _mm256_srli_epi16(m, 7);                            // == p / 255
}

// Per ogni maschera a 8 bit, le posizioni dei bit accesi. Traduce in un colpo
// solo il risultato di movemask nell'elenco dei gruppi non nulli.
struct NnzTable {
    uint16_t idx[256][8];
    constexpr NnzTable() : idx{} {
        for (int m = 0; m < 256; ++m) {
            int c = 0;
            for (int b = 0; b < 8; ++b)
                if ((m & (1 << b)) != 0) idx[m][c++] = static_cast<uint16_t>(b);
        }
    }
};
inline constexpr NnzTable NNZ{};

#if !defined(__AVXVNNI__)
// Quattro accumulatori -> quattro interi, in un colpo solo. Serve solo al
// percorso a corsie i16: quello sparso ottiene le uscite gia' separate per
// corsia e non ha niente da ridurre.
// L'ordine delle somme cambia rispetto a una riduzione lineare, ma sono interi:
// esatti e associativi, quindi il risultato e' identico.
inline void reduce4(__m256i a0, __m256i a1, __m256i a2, __m256i a3, int32_t* out) noexcept {
    const __m256i s = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    _mm_store_si128(reinterpret_cast<__m128i*>(out),
                    _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1)));
}
#endif

// SCReLU su tutte le L1_SIZE uscite insieme. Scritto a mano e non lasciato al
// compilatore perche' la versione scalare pagava due cose: `std::clamp` che GCC
// traduce in confronti e SALTI (imprevedibili, dipendono dai dati) e una
// divisione in virgola mobile per uscita — sedici `vdivss` per valutazione.
// Corsia per corsia sono le stesse operazioni IEEE nello stesso ordine, quindi
// il risultato resta identico bit per bit a quello di forwardScalar.
inline void activate(const int32_t* sums, const float* bias, float* out) noexcept {
    const __m256 scale = _mm256_set1_ps(static_cast<float>(QA * QB));
    const __m256 zero  = _mm256_setzero_ps();
    const __m256 one   = _mm256_set1_ps(1.0f);
    for (int o = 0; o < L1_SIZE; o += 8) {
        const __m256i raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(sums + o));
        __m256 z = _mm256_div_ps(_mm256_cvtepi32_ps(raw), scale);
        z = _mm256_add_ps(z, _mm256_loadu_ps(bias + o));
        z = _mm256_min_ps(_mm256_max_ps(z, zero), one);
        _mm256_store_ps(out + o, _mm256_mul_ps(z, z));
    }
}
static_assert(L1_SIZE % 8 == 0, "activate lavora a otto uscite per volta");

} // namespace
#endif // __AVX2__

bool hasVnniPath() noexcept {
#if defined(__AVX2__) && defined(__AVXVNNI__)
    return true;
#else
    return false;
#endif
}

int32_t forwardSimd(const NetworkDeep& net,
                    const int16_t* accStm,
                    const int16_t* accNtm,
                    int outputBucket) noexcept {
#if !defined(__AVX2__)
    return forwardScalar(net, accStm, accNtm, outputBucket);
#else
    alignas(32) int32_t sums[L1_SIZE];

#if defined(__AVXVNNI__)
    // Pairwise: clamp con packus e prodotto su corsie u16.
    //
    // packus_epi16 satura in UNSIGNED, cioe' fa esattamente clamp(x, 0, 255) --
    // il CReLU -- su 32 valori con UNA istruzione, dove min+max ne servivano
    // quattro. Si paga con quattro unpack per riallargare a u16 e moltiplicare,
    // e il saldo resta a favore.
    //
    // L'ordine dei byte che ne esce non e' quello di origine (packus lavora per
    // corsia da 128 bit) e NON viene raddrizzato: se lo si rimettesse a posto
    // servirebbe una vpermq ogni 32 uscite. Lo scambio e' invece assorbito una
    // volta per tutte in l1wT al caricamento.
    //
    // Gli indici dei gruppi NON nulli si raccolgono QUI, non in un secondo giro:
    // il blocco da 32 byte e' gia' in un registro, mentre una passata separata
    // doveva rileggersi tutto h8 e pagare il proprio giro di ciclo.
    //
    // ⚠️ La store degli indici ne scrive OTTO per volta anche quando il blocco
    // ne ha meno. Il conto sta perche' prima dell'emissione k vale count <= 8k,
    // quindi l'ultima finisce esattamente a HIDDEN/4. Cambiare la taglia del
    // blocco senza rifare questo conto sborda l'array.
    alignas(32) uint8_t  h8[HIDDEN];
    alignas(32) uint16_t nnz[HIDDEN / 4];
    int count = 0;
    {
        const __m256i zero  = _mm256_setzero_si256();
        const __m256i magic = _mm256_set1_epi16(static_cast<int16_t>(32897));
        const __m128i eight = _mm_set1_epi16(8);

        const auto block = [&](const int16_t* acc, int j) noexcept {
            // loadu, non load: gli accumulatori arrivano da fuori e la firma non
            // promette allineamento a 32 byte. Su x86 non costa niente quando lo
            // sono davvero, e con `load` bastava un chiamante con un vector per
            // far saltare tutto.
            const auto ld = [](const int16_t* p) noexcept {
                return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            };
            const __m256i a = _mm256_packus_epi16(ld(acc + j), ld(acc + j + 16));
            const __m256i b = _mm256_packus_epi16(ld(acc + j + PAIRWISE_OUT),
                                                 ld(acc + j + PAIRWISE_OUT + 16));
            // unpack per corsia: lo copre le uscite j..j+15, hi le j+16..j+31
            const auto half = [&](__m256i x, __m256i y, bool low) noexcept {
                const __m256i xa = low ? _mm256_unpacklo_epi8(x, zero) : _mm256_unpackhi_epi8(x, zero);
                const __m256i yb = low ? _mm256_unpacklo_epi8(y, zero) : _mm256_unpackhi_epi8(y, zero);
                const __m256i p  = _mm256_mullo_epi16(xa, yb);      // <= 65025, esatto in u16
                return _mm256_srli_epi16(_mm256_mulhi_epu16(p, magic), 7);   // == p / 255
            };
            return _mm256_packus_epi16(half(a, b, true), half(a, b, false));
        };

        const auto emit = [&](__m256i v, uint8_t* dst, __m128i base) noexcept {
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
            const unsigned mask =
                static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(v, zero)))) ^ 0xFFu;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(nnz + count),
                _mm_add_epi16(_mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(NNZ.idx[mask])), base));
            count += __builtin_popcount(mask);
        };

        // Due basi separate perche' le due prospettive si alternano: tenerle in
        // registro e incrementarle costa un'istruzione, ricostruirle da j ne
        // costerebbe due.
        __m128i baseStm = _mm_setzero_si128();
        __m128i baseNtm = _mm_set1_epi16(static_cast<int16_t>(PAIRWISE_OUT / 4));
        for (int j = 0; j < PAIRWISE_OUT; j += 32) {
            emit(block(accStm, j), h8 + j, baseStm);
            emit(block(accNtm, j), h8 + PAIRWISE_OUT + j, baseNtm);
            baseStm = _mm_add_epi16(baseStm, eight);
            baseNtm = _mm_add_epi16(baseNtm, eight);
        }
    }

    // Prodotto scalare guidato dagli ingressi. Replicando i quattro byte su
    // tutte le corsie, la corsia `o` di vpdpbusd accumula gia' il prodotto
    // scalare dell'uscita `o`: niente riduzioni orizzontali alla fine.
    // Quattro gruppi per iterazione perche' con due soli accumulatori la catena
    // dipendente sarebbe lunga quanto tutto il ciclo.
    {
        const int32_t* dw = reinterpret_cast<const int32_t*>(h8);
        const auto (&wT)[HIDDEN / 4][L1_SIZE * 4] = net.l1wT[outputBucket];
        const auto ldw = [](const int8_t* p) noexcept {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        };
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        __m256i a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        int n = 0;
        for (; n + 4 <= count; n += 4) {
            const int j0 = nnz[n], j1 = nnz[n + 1], j2 = nnz[n + 2], j3 = nnz[n + 3];
            const __m256i v0 = _mm256_set1_epi32(dw[j0]), v1 = _mm256_set1_epi32(dw[j1]);
            const __m256i v2 = _mm256_set1_epi32(dw[j2]), v3 = _mm256_set1_epi32(dw[j3]);
            a0 = _mm256_dpbusd_avx_epi32(a0, v0, ldw(wT[j0]));
            a1 = _mm256_dpbusd_avx_epi32(a1, v0, ldw(wT[j0] + 32));
            a2 = _mm256_dpbusd_avx_epi32(a2, v1, ldw(wT[j1]));
            a3 = _mm256_dpbusd_avx_epi32(a3, v1, ldw(wT[j1] + 32));
            a4 = _mm256_dpbusd_avx_epi32(a4, v2, ldw(wT[j2]));
            a5 = _mm256_dpbusd_avx_epi32(a5, v2, ldw(wT[j2] + 32));
            a6 = _mm256_dpbusd_avx_epi32(a6, v3, ldw(wT[j3]));
            a7 = _mm256_dpbusd_avx_epi32(a7, v3, ldw(wT[j3] + 32));
        }
        for (; n < count; ++n) {
            const int j = nnz[n];
            const __m256i v = _mm256_set1_epi32(dw[j]);
            a0 = _mm256_dpbusd_avx_epi32(a0, v, ldw(wT[j]));
            a1 = _mm256_dpbusd_avx_epi32(a1, v, ldw(wT[j] + 32));
        }
        const __m256i lo = _mm256_add_epi32(_mm256_add_epi32(a0, a2), _mm256_add_epi32(a4, a6));
        const __m256i hi = _mm256_add_epi32(_mm256_add_epi32(a1, a3), _mm256_add_epi32(a5, a7));
        _mm256_store_si256(reinterpret_cast<__m256i*>(sums), lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sums + 8), hi);
    }
    static_assert(L1_SIZE == 16, "il percorso sparso usa due accumulatori da otto corsie");
#else
    // Senza VNNI: corsie i16, con i pesi gia' convertiti al caricamento (vedi
    // l1w16 in loadFromFile) per togliere una cvtepi8_epi16 dal ciclo interno.
    // Qui il ciclo resta guidato dalle uscite: senza vpdpbusd non si puo'
    // replicare un gruppo da quattro byte sulle corsie, e la sparsita' a corsie
    // i16 non ripagherebbe il costo di trovare i non-zeri.
    //
    // Otto righe per gruppo, ciascuna col proprio accumulatore: il prodotto
    // scalare ha ~5 cicli di latenza, e con pochi accumulatori le iterazioni
    // formano catene dipendenti in cui si aspetta la latenza invece di
    // sfruttare il throughput. Otto e' il massimo utile: a sedici gli
    // accumulatori piu' `h` non stanno nei sedici registri ymm.
    constexpr int GROUP = 8;
    static_assert(L1_SIZE % GROUP == 0, "il ciclo elabora GROUP uscite per volta");
    alignas(32) int16_t h16[HIDDEN];
    for (int j = 0; j < PAIRWISE_OUT; j += 16) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(h16 + j), pairwiseBlock(accStm, j));
        _mm256_store_si256(reinterpret_cast<__m256i*>(h16 + PAIRWISE_OUT + j), pairwiseBlock(accNtm, j));
    }

    for (int o = 0; o < L1_SIZE; o += GROUP) {
        __m256i a[GROUP];
        for (int g = 0; g < GROUP; ++g) a[g] = _mm256_setzero_si256();
        for (int i = 0; i < HIDDEN; i += 16) {
            const __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(h16 + i));
            for (int g = 0; g < GROUP; ++g) {
                const int16_t* row = net.l1w16[outputBucket][o + g];
                a[g] = _mm256_add_epi32(a[g], _mm256_madd_epi16(h,
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i))));
            }
        }
        for (int g = 0; g < GROUP; g += 4) reduce4(a[g], a[g + 1], a[g + 2], a[g + 3], sums + o + g);
    }
#endif

    alignas(32) float a1[L1_SIZE];
    activate(sums, net.l1b[outputBucket], a1);

    float y = net.l2b[outputBucket];
    for (int o = 0; o < L1_SIZE; ++o) {
        y += a1[o] * net.l2w[outputBucket][o];
    }
    return static_cast<int32_t>(std::lround(y * static_cast<float>(SCALE)));
#endif
}

bool loadFromFile(const char* path, NetworkDeep& net) noexcept {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;

    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return false; }
    const long size = std::ftell(f);
    std::rewind(f);
    if (size < 0 || static_cast<size_t>(size) < PAYLOAD_BYTES || size % 64 != 0) {
        std::fclose(f);
        return false;
    }

    // I campi si leggono in sequenza: la struct ha lo stesso ordine del file,
    // ma NON se ne puo' fare una fread unica, perche' l'allineamento della
    // struct puo' inserire padding fra i blocchi che nel file non esiste.
    const auto rd = [&](void* dst, size_t bytes) noexcept {
        return std::fread(dst, 1, bytes, f) == bytes;
    };
    const bool ok =
           rd(net.l0w, sizeof(net.l0w))
        && rd(net.l0b, sizeof(net.l0b))
        && rd(net.l1w, sizeof(net.l1w))
        && rd(net.l1b, sizeof(net.l1b))
        && rd(net.l2w, sizeof(net.l2w))
        && rd(net.l2b, sizeof(net.l2b));
    if (!ok) { std::fclose(f); return false; }

    // La coda e' "bullet" ripetuto: qualunque altra cosa significa che il
    // layout e' andato alla deriva, ed e' meglio saperlo qui che come una rete
    // che gioca male senza motivo apparente.
    const size_t padBytes = static_cast<size_t>(size) - PAYLOAD_BYTES;
    unsigned char pad[64];
    if (padBytes > sizeof(pad) || std::fread(pad, 1, padBytes, f) != padBytes) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    static const char kTag[] = "bullet";
    for (size_t i = 0; i < padBytes; ++i) {
        if (pad[i] != static_cast<unsigned char>(kTag[i % 6])) return false;
    }

    // Derivate, non lette dal file.
    for (int b = 0; b < OUTPUT_BUCKETS; ++b)
        for (int o = 0; o < L1_SIZE; ++o)
            for (int i = 0; i < HIDDEN; ++i)
                net.l1w16[b][o][i] = net.l1w[b][o][i];

    // l1wT: stessi pesi, indicizzati per GRUPPO DI QUATTRO INGRESSI invece che
    // per uscita, cosi' il ciclo sparso trova in 64 byte contigui tutto cio' che
    // serve a un gruppo. Le due meta' dei 64 byte sono i due operandi di
    // vpdpbusd: uscite 0-7 e uscite 8-15.
    //
    // PACKUS_MAP traduce la posizione nel blocco da 32 in cui il pairwise
    // scrive verso il neurone di origine. Assorbire qui lo scambio di corsie di
    // packus costa zero e toglie una vpermq ogni 32 uscite dal ciclo caldo.
    for (int b = 0; b < OUTPUT_BUCKETS; ++b)
        for (int g = 0; g < HIDDEN / 4; ++g)
            for (int o = 0; o < L1_SIZE; ++o)
                for (int k = 0; k < 4; ++k) {
                    const int pos   = 4 * g + k;                       // posizione in h8
                    const int block = pos & ~31;                       // blocco da 32
                    const int src   = block + PACKUS_MAP[pos - block]; // neurone di origine
                    net.l1wT[b][g][(o & 7) * 4 + k + (o < 8 ? 0 : 32)] = net.l1w[b][o][src];
                }
    return true;
}

} // namespace NNUE::Deep
