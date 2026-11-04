// Census of the l1 neurons of a quantised net over a bulletformat range: how
// many never fire, per output bucket of the file.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/l1census.cpp nnue/nnue_deep.cpp -o /tmp/l1census
//   /tmp/l1census <net> <data.bin> [offset] [count]
//
// A neuron is dead in a bucket when its SCReLU input is <= 0 on at least 99.9% of
// that bucket's positions: it contributes nothing and gets no gradient, and
// AdamW's decay then zeroes its row (sanity_deep counts those rows without data).
// On 2026-10-28 every deep net measured had about 30% of them, almost all in the
// buckets of 18-32 pieces; the census of a checkpoint shows when they die.
//
// Use positions no net trained on: the first 20,000,400 records of v8_std.t0.bin.

#include "fen_accumulator.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace NNUE::Deep;
using namespace NNUE::Deep::tools;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <net> <data.bin> [offset] [count]\n", argv[0]);
        return 2;
    }
    auto net = std::make_unique<NetworkDeep>();
    if (!loadFromFile(argv[1], *net)) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const uint64_t offset = argc > 3 ? std::stoull(argv[3]) : 1'000'000;
    const uint64_t count = argc > 4 ? std::stoull(argv[4]) : 600'000;
    const std::vector<Record> recs = readRecords(argv[2], offset, count);
    if (recs.empty()) {
        std::fprintf(stderr, "no records at offset %llu of %s\n", (unsigned long long)offset, argv[2]);
        return 1;
    }

    // The engine's slots: a file with fewer buckets fills consecutive slots with
    // the same head, so slot / share is the file's bucket.
    const int share = OUTPUT_BUCKETS / net->fileBuckets;
    std::vector<uint64_t> seen(net->fileBuckets), fired(net->fileBuckets * L1_SIZE);
    std::vector<int16_t> us(HIDDEN), them(HIDDEN);
    int32_t hl1[HIDDEN];
    for (const Record& r : recs) {
        const int slot = accumulateRecord(*net, r, us.data(), them.data());
        for (int j = 0; j < PAIRWISE_OUT; ++j) {
            hl1[j] = std::clamp<int32_t>(us[j], 0, QA) * std::clamp<int32_t>(us[j + PAIRWISE_OUT], 0, QA) / QA;
            hl1[j + PAIRWISE_OUT] =
                std::clamp<int32_t>(them[j], 0, QA) * std::clamp<int32_t>(them[j + PAIRWISE_OUT], 0, QA) / QA;
        }
        const int fb = slot / share;
        ++seen[fb];
        for (int o = 0; o < L1_SIZE; ++o) {
            int32_t sum = 0;
            for (int i = 0; i < HIDDEN; ++i) sum += hl1[i] * static_cast<int32_t>(net->l1w[slot][o][i]);
            fired[fb * L1_SIZE + o] += float(sum) / float(QA * QB) + net->l1b[slot][o] > 0.0f;
        }
    }

    std::printf("%s: %zu positions, dead l1 neurons per output bucket (of %d):", argv[1], recs.size(), L1_SIZE);
    int total = 0;
    for (int b = 0; b < net->fileBuckets; ++b) {
        int dead = 0;
        for (int o = 0; o < L1_SIZE; ++o) dead += seen[b] > 0 && fired[b * L1_SIZE + o] * 1000 < seen[b];
        total += dead;
        std::printf(" %d", dead);
    }
    std::printf("  total %d/%d\n", total, net->fileBuckets * L1_SIZE);
}
