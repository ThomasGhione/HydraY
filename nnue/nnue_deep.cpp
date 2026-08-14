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

    float a2[L2_SIZE];
    for (int k = 0; k < L2_SIZE; ++k) {
        float sum = net.l2b[outputBucket][k];
        for (int o = 0; o < L1_SIZE; ++o) {
            sum += a1[o] * net.l2w[outputBucket][k][o];
        }
        a2[k] = screlu(sum);
    }

    float y = net.l3b[outputBucket];
    for (int k = 0; k < L2_SIZE; ++k) {
        y += a2[k] * net.l3w[outputBucket][k];
    }
    return static_cast<int32_t>(std::lround(y * static_cast<float>(SCALE)));
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
        && rd(net.l2b, sizeof(net.l2b))
        && rd(net.l3w, sizeof(net.l3w))
        && rd(net.l3b, sizeof(net.l3b));
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
    return true;
}

} // namespace NNUE::Deep
