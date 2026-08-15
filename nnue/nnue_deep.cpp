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

// Quattro accumulatori -> quattro interi, in un colpo solo. Ridurli uno per uno
// costava sette operazioni ciascuno e finiva comunque in memoria: qui sono
// undici in tutto e il risultato esce gia' come un blocco di quattro int32.
// L'ordine delle somme cambia rispetto a una riduzione lineare, ma sono interi:
// esatti e associativi, quindi il risultato e' identico.
inline void reduce4(__m256i a0, __m256i a1, __m256i a2, __m256i a3, int32_t* out) noexcept {
    const __m256i s = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    _mm_store_si128(reinterpret_cast<__m128i*>(out),
                    _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1)));
}

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
    // Otto righe per gruppo, ciascuna col proprio accumulatore. Non e'
    // srotolamento per gusto: il prodotto scalare ha ~5 cicli di latenza, e con
    // pochi accumulatori le iterazioni formano catene dipendenti in cui si
    // aspetta la latenza invece di sfruttare il throughput. Otto catene la
    // coprono e il caricamento di `h` viene ammortizzato su otto righe di pesi.
    // Otto e' il massimo utile: a sedici gli accumulatori piu' `h` non stanno
    // nei sedici registri ymm e lo spill costa piu' di quanto rendano le catene
    // in piu' (misurato: 165 ns contro 128).
    constexpr int GROUP = 8;
    static_assert(L1_SIZE % GROUP == 0, "il ciclo elabora GROUP uscite per volta");
    alignas(32) int32_t sums[L1_SIZE];

#if defined(__AVXVNNI__)
    // Il pairwise scrive DIRETTAMENTE in u8: i suoi valori stanno in [0, QA] per
    // costruzione, quindi packus non ha niente da saturare, e passare per un
    // buffer i16 intermedio significherebbe scrivere 2 KiB e rileggerli subito.
    // packus lavora per corsia da 128 bit: la permute rimette i 32 byte
    // nell'ordine di origine.
    alignas(32) uint8_t h8[HIDDEN];
    for (int j = 0; j < PAIRWISE_OUT; j += 32) {
        const __m256i s = _mm256_packus_epi16(pairwiseBlock(accStm, j), pairwiseBlock(accStm, j + 16));
        _mm256_store_si256(reinterpret_cast<__m256i*>(h8 + j), _mm256_permute4x64_epi64(s, 0xD8));
        const __m256i n = _mm256_packus_epi16(pairwiseBlock(accNtm, j), pairwiseBlock(accNtm, j + 16));
        _mm256_store_si256(reinterpret_cast<__m256i*>(h8 + PAIRWISE_OUT + j),
                           _mm256_permute4x64_epi64(n, 0xD8));
    }

    for (int o = 0; o < L1_SIZE; o += GROUP) {
        __m256i a[GROUP];
        for (int g = 0; g < GROUP; ++g) a[g] = _mm256_setzero_si256();
        for (int i = 0; i < HIDDEN; i += 32) {
            const __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(h8 + i));
            for (int g = 0; g < GROUP; ++g) {
                const int8_t* row = net.l1w[outputBucket][o + g];
                a[g] = _mm256_dpbusd_avx_epi32(a[g], h,
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i)));
            }
        }
        for (int g = 0; g < GROUP; g += 4) reduce4(a[g], a[g + 1], a[g + 2], a[g + 3], sums + o + g);
    }
#else
    // Senza VNNI: corsie i16, con i pesi gia' convertiti al caricamento (vedi
    // l1w16 in loadFromFile) per togliere una cvtepi8_epi16 dal ciclo interno.
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

    // Derivata, non letta dal file.
    for (int b = 0; b < OUTPUT_BUCKETS; ++b)
        for (int o = 0; o < L1_SIZE; ++o)
            for (int i = 0; i < HIDDEN; ++i)
                net.l1w16[b][o][i] = net.l1w[b][o][i];
    return true;
}

} // namespace NNUE::Deep
