// Reorders the neurons of a deep net (with or without threat inputs) so the
// sparse l1 product skips more work. A standalone program, not part of the engine.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/reorder.cpp nnue/nnue_deep.cpp -o /tmp/reorder
//   /tmp/reorder <in.nnue> <positions.fen|data.bin> <out.nnue> [max_positions]
//
// The positions file holds one FEN per line from real games (or bulletformat
// 32-byte records when passing a .bin pool; default cap 20000 for .bin). A few
// thousand are enough, and the gain holds on unseen positions.
//
// Why it works: the l1 dot product skips groups of four consecutive inputs
// that are all zero. Out of training the neurons sit in random order, so the
// zero-group rate is what independent neurons give; grouping neurons that go
// quiet together cuts l1 work sharply.
//
// Why the engine does not change: permuting l0's outputs (l0w, l0t if present,
// l0b) and the matching columns of l1w computes exactly the same function, in
// the same file format. The pairwise product couples neuron j with j + 512, so
// the 512 PAIRS are permuted, never the 1024 neurons, and one permutation
// serves both perspectives because l0 is shared.
//
// The final check reloads the output file and verifies identical evaluations on
// both synthetic vectors (every output bucket) and the real positions (full
// piece + threat accumulation and forwardSimd).

#include "threat_accumulator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace NNUE::Deep;
using namespace NNUE::Deep::tools;

namespace {

// ---------------------------------------------------------------------------
// Pairwise zero bitsets, greedy + local-search permutation
// ---------------------------------------------------------------------------

using Bits = std::vector<uint64_t>;

[[nodiscard]] long popcountAll(const Bits& b) noexcept {
    long n = 0;
    for (const uint64_t w : b) n += std::popcount(w);
    return n;
}

[[nodiscard]] long zeroGroups(const std::vector<Bits>& zero, const int* g, size_t words) noexcept {
    long n = 0;
    const uint64_t* p0 = zero[g[0]].data();
    const uint64_t* p1 = zero[g[1]].data();
    const uint64_t* p2 = zero[g[2]].data();
    const uint64_t* p3 = zero[g[3]].data();
    for (size_t w = 0; w < words; ++w)
        n += std::popcount(p0[w] & p1[w] & p2[w] & p3[w]);
    return n;
}

[[nodiscard]] std::vector<Bits> buildZeroBits(const NetworkDeep& net, const int8_t* rawL0t,
                                              const std::vector<NormPos>& positions, size_t& words) {
    const size_t total = positions.size() * 2;
    words = (total + 63) / 64;
    std::vector<Bits> zero(PAIRWISE_OUT, Bits(words, 0));
    alignas(64) int16_t us[HIDDEN], them[HIDDEN];
    const auto pwZero = [](int a, int b) noexcept {
        return (std::clamp(a, 0, QA) * std::clamp(b, 0, QA) / QA) == 0;
    };
    for (size_t p = 0; p < positions.size(); ++p) {
        accumulatePos(net, rawL0t, positions[p], us, them);
        const size_t s0 = p * 2;
        const size_t s1 = p * 2 + 1;
        for (int j = 0; j < PAIRWISE_OUT; ++j) {
            if (pwZero(us[j],   us[j + PAIRWISE_OUT]))   zero[j][s0 / 64] |= 1ULL << (s0 % 64);
            if (pwZero(them[j], them[j + PAIRWISE_OUT])) zero[j][s1 / 64] |= 1ULL << (s1 % 64);
        }
    }
    return zero;
}

[[nodiscard]] std::vector<int> greedy(const std::vector<Bits>& zero, size_t words) {
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

void localSearch(const std::vector<Bits>& zero, size_t words, std::vector<int>& perm) {
    std::vector<long> val(PAIRWISE_OUT / 4);
    for (int g = 0; g < PAIRWISE_OUT / 4; ++g) val[g] = zeroGroups(zero, &perm[4 * g], words);
    for (int pass = 0; pass < 100; ++pass) {
        bool improved = false;
        for (int a = 0; a < PAIRWISE_OUT; ++a)
            for (int b = a + 1; b < PAIRWISE_OUT; ++b) {
                const int ga = a / 4, gb = b / 4;
                if (ga == gb) continue;
                std::swap(perm[a], perm[b]);
                const long na = zeroGroups(zero, &perm[4 * ga], words);
                const long nb = zeroGroups(zero, &perm[4 * gb], words);
                if (na + nb > val[ga] + val[gb]) {
                    val[ga] = na;
                    val[gb] = nb;
                    improved = true;
                } else {
                    std::swap(perm[a], perm[b]);
                }
            }
        if (!improved) break;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: reorder <in.nnue> <positions.fen|data.bin> <out.nnue> [max_positions]\n");
        return 2;
    }
    if (!verifyAttackTables()) {
        std::fprintf(stderr, "internal error: threat attack table check failed\n");
        return 1;
    }
    uint64_t limit = 0;
    if (argc > 4 && (!parseU64(argv[4], limit) || limit == 0)) {
        std::fprintf(stderr, "invalid max_positions: %s\n", argv[4]);
        return 2;
    }

    auto net = std::make_unique<NetworkDeep>();
    std::vector<int8_t> rawL0t;
    if (!loadNetAndRawThreats(argv[1], *net, rawL0t)) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const int8_t* rawL0tPtr = net->fileThreats ? rawL0t.data() : nullptr;

    // --- activation statistics ---
    bool isBinary = false;
    const std::vector<NormPos> positions = loadPositions(argv[2], limit, isBinary);
    if (positions.size() < 500) {
        std::fprintf(stderr, "only %zu valid positions in %s: thousands are needed\n",
                     positions.size(), argv[2]);
        return 1;
    }
    size_t words = 0;
    const std::vector<Bits> zero = buildZeroBits(*net, rawL0tPtr, positions, words);
    const size_t total = positions.size() * 2;

    std::vector<int> identity(PAIRWISE_OUT);
    std::iota(identity.begin(), identity.end(), 0);
    const auto rate = [&](const std::vector<int>& p) {
        long t = 0;
        for (int g = 0; g < PAIRWISE_OUT / 4; ++g) t += zeroGroups(zero, &p[4 * g], words);
        return 100.0 * static_cast<double>(t) / static_cast<double>(total * (PAIRWISE_OUT / 4));
    };

    std::printf("%zu positions (%zu samples, both perspectives, %s)\n",
                positions.size(), total, isBinary ? "bin" : "fen");
    std::printf("all-zero groups of four, current order: %.1f%%\n", rate(identity));

    std::vector<int> perm = greedy(zero, words);
    localSearch(zero, words, perm);
    std::printf("all-zero groups of four, reordered    : %.1f%%\n", rate(perm));
    std::printf("l1 work: %.1f%% -> %.1f%%\n", 100 - rate(identity), 100 - rate(perm));

    // --- apply: perm[q] is the pair that lands at position q ---
    auto out = std::make_unique<NetworkDeep>();
    std::memcpy(out->l1b, net->l1b, sizeof(out->l1b));
    std::memcpy(out->l2w, net->l2w, sizeof(out->l2w));
    std::memcpy(out->l2b, net->l2b, sizeof(out->l2b));
    for (int q = 0; q < PAIRWISE_OUT; ++q) {
        const int src = perm[q];
        for (int f = 0; f < INPUT_BUCKETS * INPUTS; ++f) {
            out->l0w[f][q]                = net->l0w[f][src];
            out->l0w[f][q + PAIRWISE_OUT] = net->l0w[f][src + PAIRWISE_OUT];
        }
        out->l0b[q]                = net->l0b[src];
        out->l0b[q + PAIRWISE_OUT] = net->l0b[src + PAIRWISE_OUT];
        for (int b = 0; b < OUTPUT_BUCKETS; ++b)
            for (int o = 0; o < L1_SIZE; ++o) {
                out->l1w[b][o][q]                = net->l1w[b][o][src];
                out->l1w[b][o][q + PAIRWISE_OUT] = net->l1w[b][o][src + PAIRWISE_OUT];
            }
    }

    std::vector<int8_t> outL0t;
    if (net->fileThreats) {
        outL0t.resize(THREAT_BYTES);
        for (int tf = 0; tf < NNUE::THREAT_FEATURES; ++tf) {
            const int8_t* srcRow = rawL0t.data() + static_cast<size_t>(tf) * HIDDEN;
            int8_t*       dstRow = outL0t.data() + static_cast<size_t>(tf) * HIDDEN;
            for (int q = 0; q < PAIRWISE_OUT; ++q) {
                const int src = perm[q];
                dstRow[q]                = srcRow[src];
                dstRow[q + PAIRWISE_OUT] = srcRow[src + PAIRWISE_OUT];
            }
        }
    }

    // --- write, in the layout of the source file ---
    std::FILE* f = std::fopen(argv[3], "wb");
    if (f == nullptr) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const auto wr = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
    const int repeat = OUTPUT_BUCKETS / net->fileBuckets;
    const auto heads = [&](const auto& field) {
        bool good = true;
        for (int b = 0; b < net->fileBuckets && good; ++b) good = wr(&field[b * repeat], sizeof(field[0]));
        return good;
    };
    bool ok = wr(out->l0w, sizeof(out->l0w)) && wr(out->l0b, sizeof(out->l0b))
           && heads(out->l1w) && heads(out->l1b) && heads(out->l2w) && heads(out->l2b);
    if (ok && net->fileThreats) ok = wr(outL0t.data(), THREAT_BYTES);
    const size_t padBytes = fileBytes(net->fileBuckets, net->fileThreats)
                          - payloadBytes(net->fileBuckets, net->fileThreats);
    static const char kTag[] = "bullet";
    for (size_t i = 0; i < padBytes && ok; ++i) ok = std::fputc(kTag[i % 6], f) != EOF;
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "incomplete write\n"); return 1; }

    // --- check: the reordered net computes the SAME function ---
    auto back = std::make_unique<NetworkDeep>();
    std::vector<int8_t> backRawL0t;
    if (!loadNetAndRawThreats(argv[3], *back, backRawL0t)
        || back->fileBuckets != net->fileBuckets
        || back->fileThreats != net->fileThreats) {
        std::fprintf(stderr, "the written file does not load back in the source layout\n");
        return 1;
    }
    const int8_t* backRawL0tPtr = back->fileThreats ? backRawL0t.data() : nullptr;

    uint32_t rs = 20260824;
    const auto rnd = [&] { rs = rs * 1664525u + 1013904223u; return static_cast<int32_t>(rs >> 15); };
    std::vector<int16_t> a(HIDDEN), b(HIDDEN), pa(HIDDEN), pb(HIDDEN);
    long badSimd = 0, checksSimd = 0;
    for (int t = 0; t < 4000; ++t) {
        const int span = 40 + (t % 7) * 600;
        for (int i = 0; i < HIDDEN; ++i) {
            a[i] = static_cast<int16_t>(rnd() % span - span / 3);
            b[i] = static_cast<int16_t>(rnd() % span - span / 3);
        }
        // The accumulators are permuted like the neurons, or the same net would
        // be compared on different inputs.
        for (int q = 0; q < PAIRWISE_OUT; ++q) {
            pa[q] = a[perm[q]]; pa[q + PAIRWISE_OUT] = a[perm[q] + PAIRWISE_OUT];
            pb[q] = b[perm[q]]; pb[q + PAIRWISE_OUT] = b[perm[q] + PAIRWISE_OUT];
        }
        for (int bk = 0; bk < OUTPUT_BUCKETS; ++bk, ++checksSimd)
            if (forwardSimd(*net, a.data(), b.data(), bk)
                != forwardSimd(*back, pa.data(), pb.data(), bk)) ++badSimd;
    }

    alignas(64) int16_t us0[HIDDEN], them0[HIDDEN], us1[HIDDEN], them1[HIDDEN];
    long badPos = 0;
    for (const NormPos& pos : positions) {
        accumulatePos(*net,  rawL0tPtr,     pos, us0, them0);
        accumulatePos(*back, backRawL0tPtr, pos, us1, them1);
        if (forwardSimd(*net, us0, them0, pos.outBucket)
            != forwardSimd(*back, us1, them1, pos.outBucket)) ++badPos;
    }

    if (badSimd != 0 || badPos != 0) {
        std::fprintf(stderr, "\nCHECK FAILED: simd=%ld/%ld pos=%ld/%zu - the file is NOT valid\n",
                     badSimd, checksSimd, badPos, positions.size());
        return 1;
    }
    std::printf("check: %ld forward + %zu position comparisons, identical evaluations\n",
                checksSimd, positions.size());
    std::printf("wrote %s\n", argv[3]);
    return 0;
}
