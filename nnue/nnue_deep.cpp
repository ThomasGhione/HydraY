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

inline int32_t hsum(__m256i v) noexcept {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

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
    alignas(32) int16_t hl1[HIDDEN];
    for (int j = 0; j < PAIRWISE_OUT; j += 16) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(hl1 + j), pairwiseBlock(accStm, j));
        _mm256_store_si256(reinterpret_cast<__m256i*>(hl1 + PAIRWISE_OUT + j), pairwiseBlock(accNtm, j));
    }

    float a1[L1_SIZE];

#if defined(__AVXVNNI__)
    // I valori del pairwise stanno in [0, 255]: impacchettarli in u8 costa una
    // packus + una permute ogni 32, e in cambio dpbusd macina 32 prodotti per
    // istruzione invece di 16. packus_epi16 satura, ma qui non c'e' niente da
    // saturare — il pairwise non puo' uscire da [0, QA] per costruzione.
    alignas(32) uint8_t hl1u8[HIDDEN];
    for (int i = 0; i < HIDDEN; i += 32) {
        const __m256i lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(hl1 + i));
        const __m256i hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(hl1 + i + 16));
        const __m256i pk = _mm256_packus_epi16(lo, hi);
        // packus lavora per corsia da 128 bit: la permute rimette i 32 byte
        // nell'ordine di origine.
        _mm256_store_si256(reinterpret_cast<__m256i*>(hl1u8 + i),
                           _mm256_permute4x64_epi64(pk, 0xD8));
    }
    // Quattro uscite per volta, ciascuna con il proprio accumulatore. Non e'
    // srotolamento per gusto: dpbusd ha ~5 cicli di latenza, e con un solo
    // accumulatore le 32 iterazioni formano una catena dipendente in cui si
    // aspetta la latenza invece di sfruttare il throughput. Quattro catene
    // indipendenti la coprono, e in piu' il caricamento di `h` viene
    // ammortizzato su quattro righe di pesi invece di essere rifatto ogni volta.
    static_assert(L1_SIZE % 4 == 0, "il ciclo elabora quattro uscite per volta");
    for (int o = 0; o < L1_SIZE; o += 4) {
        const int8_t* r0 = net.l1w[outputBucket][o + 0];
        const int8_t* r1 = net.l1w[outputBucket][o + 1];
        const int8_t* r2 = net.l1w[outputBucket][o + 2];
        const int8_t* r3 = net.l1w[outputBucket][o + 3];
        __m256i a0 = _mm256_setzero_si256(), a1v = _mm256_setzero_si256();
        __m256i a2v = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
        for (int i = 0; i < HIDDEN; i += 32) {
            const __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(hl1u8 + i));
            a0  = _mm256_dpbusd_avx_epi32(a0,  h, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + i)));
            a1v = _mm256_dpbusd_avx_epi32(a1v, h, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + i)));
            a2v = _mm256_dpbusd_avx_epi32(a2v, h, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + i)));
            a3  = _mm256_dpbusd_avx_epi32(a3,  h, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r3 + i)));
        }
        const int32_t sums[4] = {hsum(a0), hsum(a1v), hsum(a2v), hsum(a3)};
        for (int k = 0; k < 4; ++k) {
            a1[o + k] = screlu(static_cast<float>(sums[k]) / static_cast<float>(QA * QB)
                               + net.l1b[outputBucket][o + k]);
        }
    }
#else
    // Senza VNNI: corsie i16, con i pesi gia' convertiti al caricamento (vedi
    // l1w16 in loadFromFile) per togliere una cvtepi8_epi16 dal ciclo interno.
    for (int o = 0; o < L1_SIZE; ++o) {
        const int16_t* row = net.l1w16[outputBucket][o];
        __m256i acc = _mm256_setzero_si256();
        for (int i = 0; i < HIDDEN; i += 16) {
            const __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(hl1 + i));
            const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(h, w));
        }
        a1[o] = screlu(static_cast<float>(hsum(acc)) / static_cast<float>(QA * QB)
                       + net.l1b[outputBucket][o]);
    }
#endif

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
