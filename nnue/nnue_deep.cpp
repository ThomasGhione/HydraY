#include "network_deep.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace NNUE::Deep {

namespace {

// Floating-point SCReLU, as in bullet's graph: clamp to [0,1] and square.
[[nodiscard]] inline float screlu(float x) noexcept {
    const float y = std::clamp(x, 0.0f, 1.0f);
    return y * y;
}

// CReLU + pairwise product: 1024 i16 accumulators -> 512 values in [0, QA].
// The division by QA is integer and truncating, as in sanity_deep.rs.
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
// Vectorised forward
//
// The pairwise pass stays on 16-bit lanes throughout, and that is not just
// convenience:
//   - a and b are in [0, QA] = [0, 255], so a*b <= 65025, which fits EXACTLY in
//     a u16. `mullo_epi16` returns the right value;
//   - the division by 255 becomes `mulhi_epu16(x, 32897) >> 7`, which is
//     floor(x*32897 / 2^23) == x/255 for every x in [0, 65025]. Checked at the
//     two endpoints that matter: 254 -> 0 and 65025 -> 255.
//
// For l1's dot product the obvious route would be `maddubs_epi16`, which chews
// through 32 values per instruction. It CANNOT be used: it saturates to i16,
// and here its addends reach 255*127*2 = 64770, well past 32767. It would
// saturate silently -- the net would play worse with no test noticing. Two
// exact routes remain:
//   - AVX-VNNI `dpbusd`: accumulates in i32, 32 multiplications per instruction;
//   - AVX2 `madd_epi16`: accumulates in i32, 16 per instruction.
// The first is twice as fast but needs a CPU that supports it, hence the
// fallback and the requirement that the two paths be checked against each
// other.
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
    const __m256i p  = _mm256_mullo_epi16(a, b);              // <= 65025, exact in u16
    const __m256i m  = _mm256_mulhi_epu16(p, _mm256_set1_epi16(static_cast<int16_t>(32897)));
    return _mm256_srli_epi16(m, 7);                            // == p / 255
}

// For every 8-bit mask, the positions of the set bits. Turns a movemask result
// into the list of non-zero groups in one step.
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
// Four accumulators -> four integers in one go. Only the i16-lane path needs
// it: the sparse path already gets its outputs separated per lane and has
// nothing to reduce.
// The summation order differs from a linear reduction, but these are integers:
// exact and associative, so the result is identical.
inline void reduce4(__m256i a0, __m256i a1, __m256i a2, __m256i a3, int32_t* out) noexcept {
    const __m256i s = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    _mm_store_si128(reinterpret_cast<__m128i*>(out),
                    _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1)));
}
#endif

// SCReLU over all L1_SIZE outputs at once. Written by hand rather than left to
// the compiler because the scalar version paid for two things: `std::clamp`,
// which GCC turns into compares and BRANCHES (unpredictable, data-dependent),
// and one floating-point division per output -- sixteen `vdivss` per
// evaluation. Lane by lane these are the same IEEE operations in the same
// order, so the result stays bit-identical to forwardScalar's.
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
    // Pairwise: clamp via packus, product on u16 lanes.
    //
    // packus_epi16 saturates UNSIGNED, which is exactly clamp(x, 0, 255) -- the
    // CReLU -- over 32 values in ONE instruction, where min+max needed four. It
    // costs four unpacks to widen back to u16 and multiply, and the balance is
    // still favourable.
    //
    // The byte order it produces is not the source order (packus works per
    // 128-bit lane) and is NOT straightened out: putting it back would cost one
    // vpermq per 32 outputs. The swap is absorbed once and for all into l1wT at
    // load time instead.
    //
    // The indices of the NON-zero groups are collected HERE, not in a second
    // pass: the 32-byte block is already in a register, whereas a separate pass
    // had to re-read all of h8 and pay for its own loop.
    //
    // WARNING: the index store writes EIGHT at a time even when the block holds
    // fewer. It is in bounds because before the k-th emission count <= 8k, so
    // the last one ends exactly at HIDDEN/4. Changing the block size without
    // redoing that arithmetic overruns the array.
    alignas(32) uint8_t  h8[HIDDEN];
    alignas(32) uint16_t nnz[HIDDEN / 4];
    int count = 0;
    {
        const __m256i zero  = _mm256_setzero_si256();
        const __m256i magic = _mm256_set1_epi16(static_cast<int16_t>(32897));
        const __m128i eight = _mm_set1_epi16(8);

        const auto block = [&](const int16_t* acc, int j) noexcept {
            // loadu, not load: the accumulators come from outside and the
            // signature promises no 32-byte alignment. On x86 it costs nothing
            // when they are aligned anyway, and with `load` a single caller
            // passing a vector was enough to crash.
            const auto ld = [](const int16_t* p) noexcept {
                return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            };
            const __m256i a = _mm256_packus_epi16(ld(acc + j), ld(acc + j + 16));
            const __m256i b = _mm256_packus_epi16(ld(acc + j + PAIRWISE_OUT),
                                                 ld(acc + j + PAIRWISE_OUT + 16));
            // unpack per lane: lo covers outputs j..j+15, hi covers j+16..j+31
            const auto half = [&](__m256i x, __m256i y, bool low) noexcept {
                const __m256i xa = low ? _mm256_unpacklo_epi8(x, zero) : _mm256_unpackhi_epi8(x, zero);
                const __m256i yb = low ? _mm256_unpacklo_epi8(y, zero) : _mm256_unpackhi_epi8(y, zero);
                const __m256i p  = _mm256_mullo_epi16(xa, yb);      // <= 65025, exact in u16
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

        // Two separate bases because the perspectives alternate: keeping them
        // in registers and incrementing costs one instruction, rebuilding them
        // from j would cost two.
        __m128i baseStm = _mm_setzero_si128();
        __m128i baseNtm = _mm_set1_epi16(static_cast<int16_t>(PAIRWISE_OUT / 4));
        for (int j = 0; j < PAIRWISE_OUT; j += 32) {
            emit(block(accStm, j), h8 + j, baseStm);
            emit(block(accNtm, j), h8 + PAIRWISE_OUT + j, baseNtm);
            baseStm = _mm_add_epi16(baseStm, eight);
            baseNtm = _mm_add_epi16(baseNtm, eight);
        }
    }

    // Input-driven dot product. By broadcasting the four bytes across every
    // lane, vpdpbusd's lane `o` already accumulates output `o`'s dot product:
    // no horizontal reductions at the end.
    // Four groups per iteration because with only two accumulators the
    // dependent chain would be as long as the whole loop.
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
    // Without VNNI: i16 lanes, with the weights already widened at load time
    // (see l1w16 in loadFromMemory) to drop a cvtepi8_epi16 from the inner loop.
    // Here the loop stays output-driven: without vpdpbusd a four-byte group
    // cannot be broadcast across the lanes, and sparsity on i16 lanes would not
    // repay the cost of finding the non-zeros.
    //
    // Eight rows per group, each with its own accumulator: the dot product has
    // ~5 cycles of latency, and with few accumulators the iterations form
    // dependent chains that wait on latency instead of using throughput. Eight
    // is the useful maximum: at sixteen, the accumulators plus `h` no longer fit
    // in the sixteen ymm registers.
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

bool loadFromMemory(const unsigned char* data, size_t size, NetworkDeep& net) noexcept {
    if (size < PAYLOAD_BYTES || size % 64 != 0) return false;

    // Fields are read in order: the struct matches the file's layout, but a
    // single bulk copy is NOT possible, because struct alignment can insert
    // padding between blocks that the file does not have.
    size_t off = 0;
    const auto rd = [&](void* dst, size_t bytes) noexcept {
        std::memcpy(dst, data + off, bytes);
        off += bytes;
    };
    rd(net.l0w, sizeof(net.l0w));
    rd(net.l0b, sizeof(net.l0b));
    rd(net.l1w, sizeof(net.l1w));
    rd(net.l1b, sizeof(net.l1b));
    rd(net.l2w, sizeof(net.l2w));
    rd(net.l2b, sizeof(net.l2b));

    // The tail is "bullet" repeated: anything else means the layout has
    // drifted, and it is better to find out here than as a net that plays badly
    // for no apparent reason.
    static const char kTag[] = "bullet";
    for (size_t i = PAYLOAD_BYTES; i < size; ++i) {
        if (data[i] != static_cast<unsigned char>(kTag[(i - PAYLOAD_BYTES) % 6])) return false;
    }

    // Derived, not read from the file.
    for (int b = 0; b < OUTPUT_BUCKETS; ++b)
        for (int o = 0; o < L1_SIZE; ++o)
            for (int i = 0; i < HIDDEN; ++i)
                net.l1w16[b][o][i] = net.l1w[b][o][i];

    // l1wT: the same weights, indexed by GROUP OF FOUR INPUTS instead of by
    // output, so the sparse loop finds everything a group needs in 64
    // contiguous bytes. The two halves of those 64 bytes are the two vpdpbusd
    // operands: outputs 0-7 and outputs 8-15.
    //
    // PACKUS_MAP maps the position within the 32-wide block that the pairwise
    // pass writes to the originating neuron. Absorbing packus's lane swap here
    // is free and removes one vpermq per 32 outputs from the hot loop.
    for (int b = 0; b < OUTPUT_BUCKETS; ++b)
        for (int g = 0; g < HIDDEN / 4; ++g)
            for (int o = 0; o < L1_SIZE; ++o)
                for (int k = 0; k < 4; ++k) {
                    const int pos   = 4 * g + k;                       // position in h8
                    const int block = pos & ~31;                       // 32-wide block
                    const int src   = block + PACKUS_MAP[pos - block]; // source neuron
                    net.l1wT[b][g][(o & 7) * 4 + k + (o < 8 ? 0 : 32)] = net.l1w[b][o][src];
                }
    return true;
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
    std::vector<unsigned char> blob(static_cast<size_t>(size));
    const bool read = std::fread(blob.data(), 1, blob.size(), f) == blob.size();
    std::fclose(f);
    return read && loadFromMemory(blob.data(), blob.size(), net);
}

} // namespace NNUE::Deep
