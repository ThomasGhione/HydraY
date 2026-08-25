#include "nnue.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../board/board.hpp"
#include "accumulator.hpp"
#include "network.hpp"
#include "network_deep.hpp"

// Symbols from nnue/embedded.cpp (.incbin of nnue/net/hydray.nnue).
extern "C" const unsigned char g_hydrayEmbeddedNetStart[];
extern "C" const unsigned char g_hydrayEmbeddedNetEnd[];

namespace NNUE {

const Network* activeNetwork = nullptr;

namespace {

std::unique_ptr<Network> ownedNetwork;

// --- deep network (network_deep.hpp) ---------------------------------------
// Lives alongside the single-layer net rather than replacing it: either one can
// be the active evaluator.
//
// The two formats share the SAME prefix -- l0w then l0b, same sizes -- and the
// accumulator reads only that. So with a deep net active, `activeNetwork`
// points at the start of the deep struct and the accumulator keeps working
// without knowing any of this. The static_asserts below check that assumption:
// if either layout changes, it stops compiling.
//
// RULE: with a deep net active, `activeNetwork->outputWeights` and
// `->outputBias` alias l1's weights and must NOT be read. The only reader is
// evaluate(), which dispatches on the flag.
std::unique_ptr<Deep::NetworkDeep> ownedDeep;
const Deep::NetworkDeep* activeDeep = nullptr;

static_assert(offsetof(Network, featureWeights) == offsetof(Deep::NetworkDeep, l0w));
static_assert(sizeof(Network::featureWeights) == sizeof(Deep::NetworkDeep::l0w));
static_assert(offsetof(Network, featureBias) == offsetof(Deep::NetworkDeep, l0b));
static_assert(sizeof(Network::featureBias) == sizeof(Deep::NetworkDeep::l0b));

// File size of a deep net, bullet's padding included.
constexpr size_t DEEP_FILE_BYTES = (Deep::PAYLOAD_BYTES + 63) / 64 * 64;

// The AVX2 forward multiplies clamped activations (<= QA) by output weights in
// int16: |w| must stay <= 32767/QA or mullo_epi16 wraps. bullet's AdamW clips
// weights to [-1.98, 1.98] (so |w| <= ~127 at QB=64); reject anything looser.
constexpr int32_t MAX_OUTPUT_WEIGHT = 32767 / QA; // 128

// Shared validation for file and embedded blobs (raw bytes: no alignment
// assumptions). Returns nullptr on success, an error message otherwise.
const char* validateNetworkBlob(const unsigned char* data, size_t size) noexcept {
    if (size < NETWORK_PAYLOAD_BYTES || size % 64 != 0) {
        return "unexpected size";
    }
    // bullet pads quantised.bin to 64 bytes with the repeating ASCII string
    // "bullet"; anything else means the layout drifted.
    static constexpr unsigned char SIG[6] = {'b', 'u', 'l', 'l', 'e', 't'};
    for (size_t i = NETWORK_PAYLOAD_BYTES; i < size; ++i) {
        if (data[i] != SIG[(i - NETWORK_PAYLOAD_BYTES) % 6]) {
            return "bad padding signature (layout drift?)";
        }
    }
    constexpr size_t OW_OFFSET = offsetof(Network, outputWeights);
    for (size_t i = 0; i < static_cast<size_t>(OUTPUT_BUCKETS) * 2 * HIDDEN; ++i) {
        const auto w = static_cast<int16_t>(
            static_cast<uint16_t>(data[OW_OFFSET + 2 * i])
            | (static_cast<uint16_t>(data[OW_OFFSET + 2 * i + 1]) << 8));
        if (std::abs(static_cast<int32_t>(w)) > MAX_OUTPUT_WEIGHT) {
            return "output weight exceeds the AVX2-safe bound";
        }
    }
    return nullptr;
}

// One half of the output layer: sum of screlu(acc[i]) * w[i].
// Scalar version is the reference (identical to trainer/src/bin/sanity.rs).
int32_t forwardHalf(const int16_t* __restrict acc, const int16_t* __restrict w) noexcept {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi16(static_cast<int16_t>(QA));
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < HIDDEN; i += 16) {
        const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i t = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
        const __m256i wi = _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i));
        // (t*w) fits i16 thanks to MAX_OUTPUT_WEIGHT; madd gives (t*w)*t in i32.
        const __m256i p = _mm256_mullo_epi16(t, wi);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p, t));
    }
    const __m128i lo = _mm256_castsi256_si128(sum);
    const __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01001110));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10110001));
    return _mm_cvtsi128_si32(s);
#else
    int32_t s = 0;
    for (int i = 0; i < HIDDEN; ++i) {
        const int32_t t = std::clamp<int32_t>(acc[i], 0, QA);
        s += t * t * static_cast<int32_t>(w[i]);
    }
    return s;
#endif
}

} // namespace

bool loadNetwork(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cout << "info string EvalFile error: cannot open '" << path << "'\n";
        return false;
    }
    const auto fileSize = static_cast<size_t>(in.tellg());
    in.seekg(0);

    // The format is recognised by SIZE: the two architectures have files of
    // different length and bullet's format carries no header.
    if (fileSize == DEEP_FILE_BYTES) {
        auto deep = std::make_unique<Deep::NetworkDeep>();
        if (!Deep::loadFromFile(path.c_str(), *deep)) {
            std::cout << "info string EvalFile error: deep net failed validation\n";
            return false;
        }
        ownedDeep = std::move(deep);
        activeDeep = ownedDeep.get();
        ownedNetwork.reset();
        // The accumulator only reads the l0 prefix, identical in both formats.
        activeNetwork = reinterpret_cast<const Network*>(activeDeep);
        std::cout << "info string EvalFile: deep network (1024 -> "
                  << Deep::L1_SIZE << " -> 1)"
                  << (Deep::hasVnniPath() ? ", VNNI" : ", no VNNI") << "\n";
        return true;
    }

    std::vector<unsigned char> blob(fileSize);
    if (!in.read(reinterpret_cast<char*>(blob.data()),
                 static_cast<std::streamsize>(fileSize))) {
        std::cout << "info string EvalFile error: short read\n";
        return false;
    }
    if (const char* error = validateNetworkBlob(blob.data(), blob.size())) {
        std::cout << "info string EvalFile error: " << error << "\n";
        return false;
    }

    auto net = std::make_unique<Network>();
    std::memcpy(net.get(), blob.data(), NETWORK_PAYLOAD_BYTES);
    ownedNetwork = std::move(net);
    ownedDeep.reset();
    activeDeep = nullptr;
    activeNetwork = ownedNetwork.get();
    return true;
}

size_t embeddedSize() noexcept {
    return static_cast<size_t>(g_hydrayEmbeddedNetEnd - g_hydrayEmbeddedNetStart);
}

const Network* embeddedNetwork() noexcept {
    // Validated once; the blob is 64-byte aligned (.balign in embedded.cpp),
    // so the Network overlay satisfies the AVX2 aligned loads.
    static const Network* const validated = []() -> const Network* {
        if (validateNetworkBlob(g_hydrayEmbeddedNetStart, embeddedSize()) != nullptr) return nullptr;
        return reinterpret_cast<const Network*>(g_hydrayEmbeddedNetStart);
    }();
    return validated;
}

bool activateEmbedded() noexcept {
    // Same size-based format detection as loadNetwork: the two architectures
    // have different file sizes and bullet's format carries no header. The deep
    // net cannot be a plain overlay on the blob -- l1w16 and l1wT are derived,
    // not stored -- so it is copied into an owned NetworkDeep.
    if (embeddedSize() == DEEP_FILE_BYTES) {
        auto deep = std::make_unique<Deep::NetworkDeep>();
        if (!Deep::loadFromMemory(g_hydrayEmbeddedNetStart, embeddedSize(), *deep)) return false;
        ownedDeep = std::move(deep);
        activeDeep = ownedDeep.get();
        ownedNetwork.reset();
        // The accumulator only reads the l0 prefix, identical in both formats.
        activeNetwork = reinterpret_cast<const Network*>(activeDeep);
        return true;
    }

    const Network* net = embeddedNetwork();
    if (net == nullptr) return false;
    ownedNetwork.reset();
    ownedDeep.reset();
    activeDeep = nullptr;
    activeNetwork = net;
    return true;
}

bool deepNetworkActive() noexcept { return activeDeep != nullptr; }

bool networkLoaded() noexcept {
    return activeNetwork != nullptr;
}

int32_t evaluate(const chess::Board& b) noexcept {
    const Network& net = *activeNetwork;
    // HalfKA laziness: settle any perspective dirtied by a king bucket/flip
    // crossing before reading the accumulator (board is consistent here).
    b.ensureNnueAccumulatorClean();
    const NNUE::Accumulator& acc = b.nnueAccumulator;
    const int stm = chess::Board::colorToIndex(b.getActiveColor()); // 0 = white

    // Material-count output bucket; must match bullet's MaterialCount<8>
    // (and sanity.rs): (popcount - 2) / ceil(32/8). Kings are always on the
    // board, so popcount is in [2, 32] and the bucket in [0, 7].
    const int bucket = (std::popcount(b.getPiecesBitMap()) - 2) / 4;

    if (activeDeep != nullptr) [[unlikely]] {
        return Deep::forwardSimd(*activeDeep, acc.v[stm], acc.v[stm ^ 1], bucket);
    }

    int32_t out = forwardHalf(acc.v[stm], net.outputWeights[bucket][0])
                + forwardHalf(acc.v[stm ^ 1], net.outputWeights[bucket][1]);
    out /= QA;
    out += net.outputBias[bucket];
    return out * SCALE / (QA * QB);
}

} // namespace NNUE
