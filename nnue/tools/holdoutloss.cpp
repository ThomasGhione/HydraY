// Training loss of a quantised net over a bulletformat file.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/holdoutloss.cpp nnue/nnue_deep.cpp -o /tmp/holdoutloss
//   /tmp/holdoutloss <net.bin> <data.bin> [--offset N] [--count N]
//                    [--threads N] [--wdl F] [--scale F] [--selfcheck N]
//
// Both net formats are read, recognised by file size the way the engine's
// activateEmbedded does. The single-layer one is dead in the engine but its
// nets are the only ones whose training loss was ever written down, so it is
// what validates this tool against bullet's own number.
//
// The point is to compare the SAME net on data it trained on against data it
// never saw, which is the only way to tell "the dataset is too small" from
// "the budget is spent". bullet's own TestDataset is a no-op in the pinned rev
// (crates/bullet_lib/src/value.rs:153 prints "Validation data not currently
// implemented"), so the measurement has to live here.
//
// The loss is bullet's, transcribed from crates/bullet_lib/src/value.rs:100-116
// at rev cebc78a:
//
//   target = blend * (result / 2) + (1 - blend) * sigmoid(score / scale)
//   loss   = (sigmoid(output) - target)^2,  averaged per sample
//
// where `blend` is the wdl fraction (ConstantWDL 0.3 in trainer_deep.rs) and
// `output` is the net's raw float output. We only have the quantised net, whose
// forward returns round(output * SCALE) centipawns, so sigmoid(cp / SCALE)
// costs one rounding step of 1/SCALE inside the sigmoid. That is deliberate:
// the number we want is the loss of the net that actually plays, not of the
// float checkpoint it came from.
//
// Two consequences worth remembering before quoting a number:
//   - a quantised loss is NOT directly comparable to a running loss printed by
//     bullet during training. Compare quantised against quantised.
//   - the running loss is a mean taken while the net was still moving, so it
//     sits slightly above the final net's loss on the same data.

#include "fen_accumulator.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace NNUE::Deep;

namespace {

// bulletformat::ChessBoard, 32 bytes. Declared here rather than included from
// nnue/bulletformat.hpp, which pulls in board.hpp for the writer side; this
// tool only reads. The field meanings are documented there.
struct Record {
    uint64_t occ;
    uint8_t  pcs[16];
    int16_t  score;
    uint8_t  result;
    uint8_t  ksq;
    uint8_t  oppKsq;
    uint8_t  extra[3];
};
static_assert(sizeof(Record) == 32, "must match bulletformat::ChessBoard");

constexpr int OUT_BUCKET_DIVISOR = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS;

// Single-layer quantised.bin: (768x4kb_hm -> 1024)x2 -> 8, SCReLU. Same l0 as
// the deep format, so the accumulator is shared; only the output layer differs.
// Layout and arithmetic copied from nnue/tools/evalfens.cpp, itself a
// transcription of sanity.rs, which is the oracle for this format.
//
// The bucket count is spelled out rather than taken from network_deep.hpp: no
// single-layer net will ever be trained again, so every file in this format has
// four king buckets. Following the live constant would silently reinterpret
// these nets on a branch that changes it.
constexpr int SHALLOW_INPUT_BUCKETS = 4;

struct alignas(64) ShallowNet {
    int16_t featureWeights[SHALLOW_INPUT_BUCKETS * INPUTS][HIDDEN];
    int16_t featureBias[HIDDEN];
    int16_t outputWeights[OUTPUT_BUCKETS][2][HIDDEN];
    int16_t outputBias[OUTPUT_BUCKETS];
};

constexpr size_t SHALLOW_PAYLOAD_BYTES =
    sizeof(int16_t) * (SHALLOW_INPUT_BUCKETS * INPUTS * HIDDEN + HIDDEN
                       + OUTPUT_BUCKETS * 2 * HIDDEN + OUTPUT_BUCKETS);
static_assert(SHALLOW_PAYLOAD_BYTES == 6'326'288);

int32_t shallowHalf(const int16_t* acc, const int16_t* w) noexcept {
    int32_t s = 0;
    for (int i = 0; i < HIDDEN; ++i) {
        const int32_t t = std::clamp<int32_t>(acc[i], 0, QA);
        s += t * t * static_cast<int32_t>(w[i]);
    }
    return s;
}

// Which forward to run. The accumulator only ever needs l0, which both formats
// share byte for byte, so `l0` is a NetworkDeep in either case.
struct Evaluator {
    const NetworkDeep* l0 = nullptr;
    const NetworkDeep* deep = nullptr;
    const ShallowNet*  shallow = nullptr;

    [[nodiscard]] int32_t eval(const int16_t* us, const int16_t* them, int bucket) const noexcept {
        if (deep != nullptr) return forwardSimd(*deep, us, them, bucket);
        int32_t out = shallowHalf(us, shallow->outputWeights[bucket][0])
                    + shallowHalf(them, shallow->outputWeights[bucket][1]);
        out /= QA;
        out += shallow->outputBias[bucket];
        return out * SCALE / (QA * QB);
    }
};

void addRow(int16_t* acc, const int16_t* w) noexcept {
    for (int i = 0; i < HIDDEN; ++i) acc[i] = static_cast<int16_t>(acc[i] + w[i]);
}

// Same features as tools::accumulate, read straight from the record instead of
// from a FEN. Records are stored in the stm frame (the stm is always "white"
// and its king is ksq; oppKsq is already the opponent king in its own view),
// so the black-to-move branch of the FEN path collapses away here.
// --selfcheck holds the two paths against each other.
int accumulateRecord(const NetworkDeep& net, const Record& r,
                     int16_t* accUs, int16_t* accThem) noexcept {
    const int usFlip   = (r.ksq & 7) > 3 ? 7 : 0;
    const int usBase   = INPUTS * tools::kingBucket(r.ksq);
    const int themFlip = (r.oppKsq & 7) > 3 ? 7 : 0;
    const int themBase = INPUTS * tools::kingBucket(r.oppKsq);

    std::memcpy(accUs, net.l0b, sizeof(net.l0b));
    std::memcpy(accThem, net.l0b, sizeof(net.l0b));

    uint64_t occ = r.occ;
    int idx = 0;
    while (occ != 0) {
        const int sq = std::countr_zero(occ);
        occ &= occ - 1;
        const uint8_t nibble = (r.pcs[idx / 2] >> (4 * (idx & 1))) & 0xF;
        ++idx;
        const bool isOpp = (nibble & 8) != 0;
        const int  type  = nibble & 7;

        addRow(accUs, net.l0w[usBase + (((isOpp ? 384 : 0) + type * 64 + sq) ^ usFlip)]);
        addRow(accThem, net.l0w[themBase + (((isOpp ? 0 : 384) + type * 64 + (sq ^ 56)) ^ themFlip)]);
    }
    return (idx - 2) / OUT_BUCKET_DIVISOR;
}

double sigmoid(double x) noexcept { return 1.0 / (1.0 + std::exp(-x)); }

struct Totals {
    double   loss[OUTPUT_BUCKETS] = {};
    double   absErr[OUTPUT_BUCKETS] = {};
    uint64_t count[OUTPUT_BUCKETS] = {};

    // Label statistics, so a holdout can be shown to match the training set
    // rather than assumed to. A loss difference between two ranges means
    // nothing until these agree.
    double   absScore = 0.0;
    uint64_t result[3] = {};

    void merge(const Totals& o) noexcept {
        for (int b = 0; b < OUTPUT_BUCKETS; ++b) {
            loss[b]  += o.loss[b];
            absErr[b] += o.absErr[b];
            count[b] += o.count[b];
        }
        absScore += o.absScore;
        for (int i = 0; i < 3; ++i) result[i] += o.result[i];
    }
};

struct Options {
    double   blend = 0.3;
    double   scale = 400.0;
    uint64_t offset = 0;
    uint64_t count = 0;      // 0 = to end of file
    int      threads = 0;    // 0 = hardware concurrency
    uint64_t selfcheck = 0;
    // Writes one i16 evaluation per record, in file order, so two nets can be
    // compared position by position. Nets with different king-bucket counts
    // cannot live in one binary (the map is compiled in), so the comparison has
    // to go through two runs and a join on record index. Forces one thread.
    std::string dumpEvals;
};

// One contiguous record range, read in blocks. A globally shuffled file makes
// any contiguous range a uniform sample, which is why there is no --stride.
void scanRange(const Evaluator& ev, const std::string& path, const Options& opt,
               uint64_t first, uint64_t n, Totals& out) {
    constexpr size_t BLOCK = 1 << 15;

    std::ifstream in(path, std::ios::binary);
    if (!in) return;
    in.seekg(static_cast<std::streamoff>(first * sizeof(Record)));

    std::ofstream dump;
    if (!opt.dumpEvals.empty()) {
        dump.open(opt.dumpEvals, std::ios::binary);
        if (!dump) return;
    }

    std::vector<Record> buf(BLOCK);
    std::vector<int16_t> accUs(HIDDEN), accThem(HIDDEN);
    std::vector<int16_t> dumpBuf;

    uint64_t left = n;
    while (left > 0) {
        const size_t want = static_cast<size_t>(std::min<uint64_t>(left, BLOCK));
        in.read(reinterpret_cast<char*>(buf.data()),
                static_cast<std::streamsize>(want * sizeof(Record)));
        const size_t got = static_cast<size_t>(in.gcount()) / sizeof(Record);
        if (got == 0) break;
        left -= got;
        if (dump.is_open()) dumpBuf.clear();

        for (size_t i = 0; i < got; ++i) {
            const Record& r = buf[i];
            const int bucket = accumulateRecord(*ev.l0, r, accUs.data(), accThem.data());
            const int32_t cp = ev.eval(accUs.data(), accThem.data(), bucket);

            const double target = opt.blend * (static_cast<double>(r.result) / 2.0)
                                + (1.0 - opt.blend) * sigmoid(r.score / opt.scale);
            const double diff = sigmoid(cp / opt.scale) - target;

            out.loss[bucket]  += diff * diff;
            out.absErr[bucket] += std::abs(static_cast<double>(cp - r.score));
            out.count[bucket] += 1;
            out.absScore += std::abs(static_cast<double>(r.score));
            if (r.result < 3) out.result[r.result] += 1;
            if (dump.is_open()) {
                dumpBuf.push_back(static_cast<int16_t>(std::clamp(cp, -32000, 32000)));
            }
        }
        if (dump.is_open() && !dumpBuf.empty()) {
            dump.write(reinterpret_cast<const char*>(dumpBuf.data()),
                       static_cast<std::streamsize>(dumpBuf.size() * sizeof(int16_t)));
        }
    }
}

// FEN of the stored position, for --selfcheck and for eyeballing against
// `./chess datagen-dump`. Transcribed from bulletformat.hpp::recordToFen.
std::string recordToFen(const Record& r) {
    static constexpr char PIECE_CHARS[16] = {
        'P', 'N', 'B', 'R', 'Q', 'K', '?', '?',
        'p', 'n', 'b', 'r', 'q', 'k', '?', '?'
    };
    char squares[64] = {};
    uint64_t occ = r.occ;
    int idx = 0;
    while (occ != 0) {
        const int sq = std::countr_zero(occ);
        occ &= occ - 1;
        squares[sq] = PIECE_CHARS[(r.pcs[idx / 2] >> (4 * (idx & 1))) & 0xF];
        ++idx;
    }
    std::string fen;
    for (int rank = 7; rank >= 0; --rank) {
        int run = 0;
        for (int file = 0; file < 8; ++file) {
            const char c = squares[rank * 8 + file];
            if (c == '\0') { ++run; continue; }
            if (run > 0) { fen += static_cast<char>('0' + run); run = 0; }
            fen += c;
        }
        if (run > 0) fen += static_cast<char>('0' + run);
        if (rank > 0) fen += '/';
    }
    fen += " w - - 0 1";
    return fen;
}

// Holds the record decoder against the FEN path already validated by deepcheck
// and reorder, and the SIMD forward against the scalar one. A silent decoder
// bug would move the loss by exactly the amount we are trying to measure.
bool selfcheck(const Evaluator& ev, const std::string& path, uint64_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

    std::vector<int16_t> accUs(HIDDEN), accThem(HIDDEN), fenUs, fenThem;
    Record r{};
    for (uint64_t i = 0; i < n; ++i) {
        in.read(reinterpret_cast<char*>(&r), sizeof(r));
        if (in.gcount() != sizeof(r)) break;

        const int bucket = accumulateRecord(*ev.l0, r, accUs.data(), accThem.data());
        const int32_t simd = ev.eval(accUs.data(), accThem.data(), bucket);
        if (ev.deep != nullptr) {
            const int32_t scal = forwardScalar(*ev.deep, accUs.data(), accThem.data(), bucket);
            if (simd != scal) {
                std::fprintf(stderr, "record %llu: simd %d != scalar %d\n",
                             static_cast<unsigned long long>(i), simd, scal);
                return false;
            }
        }

        const std::string fen = recordToFen(r);
        int fenBucket = 0;
        if (!tools::accumulate(*ev.l0, fen, fenUs, fenThem, fenBucket)) {
            std::fprintf(stderr, "record %llu: FEN path rejected '%s'\n",
                         static_cast<unsigned long long>(i), fen.c_str());
            return false;
        }
        if (fenBucket != bucket || fenUs != accUs || fenThem != accThem) {
            std::fprintf(stderr, "record %llu: record decoder disagrees with the FEN path\n%s\n",
                         static_cast<unsigned long long>(i), fen.c_str());
            return false;
        }
        if (i < 3) {
            std::printf("  %s | score %6d | result %d | bucket %d | eval %6d\n",
                        fen.c_str(), r.score, r.result, bucket, simd);
        }
    }
    std::printf("selfcheck: %llu records, decoder == FEN path%s\n\n",
                static_cast<unsigned long long>(n),
                ev.deep != nullptr ? ", simd == scalar" : "");
    return true;
}

bool parseU64(const char* s, uint64_t& out) {
    const char* end = s + std::strlen(s);
    return std::from_chars(s, end, out).ec == std::errc{};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: holdoutloss <net.bin> <data.bin> [--offset N] [--count N]\n"
            "                   [--threads N] [--wdl F] [--scale F] [--selfcheck N]\n"
            "                   [--dumpevals FILE]\n");
        return 2;
    }
    const std::string netPath = argv[1];
    const std::string dataPath = argv[2];

    Options opt;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        const bool hasValue = i + 1 < argc;
        if (a == "--offset" && hasValue)       { if (!parseU64(argv[++i], opt.offset)) return 2; }
        else if (a == "--count" && hasValue)   { if (!parseU64(argv[++i], opt.count)) return 2; }
        else if (a == "--selfcheck" && hasValue) { if (!parseU64(argv[++i], opt.selfcheck)) return 2; }
        else if (a == "--threads" && hasValue) { opt.threads = std::atoi(argv[++i]); }
        else if (a == "--wdl" && hasValue)     { opt.blend = std::atof(argv[++i]); }
        else if (a == "--dumpevals" && hasValue) { opt.dumpEvals = argv[++i]; }
        else if (a == "--scale" && hasValue)   { opt.scale = std::atof(argv[++i]); }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return 2; }
    }
    if (opt.blend < 0.0 || opt.blend > 1.0 || opt.scale <= 0.0) {
        std::fprintf(stderr, "wdl must be in [0,1] and scale positive\n");
        return 2;
    }

    // Format by file size, exactly as the engine's activateEmbedded does.
    static NetworkDeep net;
    static ShallowNet shallow;
    Evaluator ev;
    {
        std::ifstream nf(netPath, std::ios::binary | std::ios::ate);
        if (!nf) { std::fprintf(stderr, "cannot open net: %s\n", netPath.c_str()); return 1; }
        const auto netBytes = static_cast<size_t>(nf.tellg());

        if (netBytes >= PAYLOAD_BYTES && netBytes < PAYLOAD_BYTES + 64) {
            if (!loadFromFile(netPath.c_str(), net)) {
                std::fprintf(stderr, "cannot load deep net: %s\n", netPath.c_str());
                return 1;
            }
            ev.l0 = &net;
            ev.deep = &net;
            std::printf("format:  deep (1024 -> 16 -> 1)\n");
        } else if (netBytes >= SHALLOW_PAYLOAD_BYTES && netBytes < SHALLOW_PAYLOAD_BYTES + 64) {
            // The king-bucket map is compiled in, so a build whose map has a
            // different number of buckets cannot read these nets at all: it
            // would index rows the file does not contain.
            if constexpr (INPUT_BUCKETS != SHALLOW_INPUT_BUCKETS) {
                std::fprintf(stderr,
                    "%s is a %d-bucket single-layer net, this build has %d king buckets\n",
                    netPath.c_str(), SHALLOW_INPUT_BUCKETS, INPUT_BUCKETS);
                return 1;
            }
            nf.seekg(0);
            if (!nf.read(reinterpret_cast<char*>(&shallow),
                         static_cast<std::streamsize>(SHALLOW_PAYLOAD_BYTES))) {
                std::fprintf(stderr, "short read on %s\n", netPath.c_str());
                return 1;
            }
            // l0w/l0b are identical in the two layouts, so the accumulator can
            // read them out of a NetworkDeep holding just that prefix.
            std::memcpy(net.l0w, shallow.featureWeights, sizeof(shallow.featureWeights));
            std::memcpy(net.l0b, shallow.featureBias, sizeof(shallow.featureBias));
            ev.l0 = &net;
            ev.shallow = &shallow;
            std::printf("format:  single layer (1024 -> 1)\n");
        } else {
            std::fprintf(stderr, "%s: %zu bytes is neither format (%zu deep, %zu single)\n",
                         netPath.c_str(), netBytes, PAYLOAD_BYTES, SHALLOW_PAYLOAD_BYTES);
            return 1;
        }
    }

    std::ifstream probe(dataPath, std::ios::binary | std::ios::ate);
    if (!probe) {
        std::fprintf(stderr, "cannot open data: %s\n", dataPath.c_str());
        return 1;
    }
    const auto bytes = static_cast<uint64_t>(probe.tellg());
    if (bytes % sizeof(Record) != 0) {
        std::fprintf(stderr, "%s: %llu bytes is not a whole number of 32-byte records\n",
                     dataPath.c_str(), static_cast<unsigned long long>(bytes));
        return 1;
    }
    const uint64_t total = bytes / sizeof(Record);
    if (opt.offset >= total) {
        std::fprintf(stderr, "offset %llu is past the end (%llu records)\n",
                     static_cast<unsigned long long>(opt.offset),
                     static_cast<unsigned long long>(total));
        return 1;
    }
    const uint64_t avail = total - opt.offset;
    const uint64_t n = (opt.count == 0) ? avail : std::min(opt.count, avail);

    std::printf("net:     %s\n", netPath.c_str());
    std::printf("data:    %s (%llu records)\n", dataPath.c_str(),
                static_cast<unsigned long long>(total));
    std::printf("range:   %llu records from offset %llu\n",
                static_cast<unsigned long long>(n),
                static_cast<unsigned long long>(opt.offset));
    std::printf("target:  wdl %.2f, scale %.0f\n\n", opt.blend, opt.scale);

    if (opt.selfcheck > 0 && !selfcheck(ev, dataPath, opt.selfcheck)) return 1;

    // Dumping must keep file order, so it runs on one thread.
    int threads = !opt.dumpEvals.empty() ? 1
                : opt.threads > 0 ? opt.threads
                                  : static_cast<int>(std::thread::hardware_concurrency());
    if (threads < 1) threads = 1;
    if (static_cast<uint64_t>(threads) > n) threads = static_cast<int>(n);

    std::vector<Totals> partial(static_cast<size_t>(threads));
    std::vector<std::thread> pool;
    const uint64_t chunk = n / static_cast<uint64_t>(threads);
    for (int t = 0; t < threads; ++t) {
        const uint64_t first = opt.offset + chunk * static_cast<uint64_t>(t);
        const uint64_t len = (t == threads - 1) ? n - chunk * static_cast<uint64_t>(t) : chunk;
        pool.emplace_back([&, t, first, len] {
            scanRange(ev, dataPath, opt, first, len, partial[static_cast<size_t>(t)]);
        });
    }
    for (auto& th : pool) th.join();

    Totals all;
    for (const auto& p : partial) all.merge(p);

    uint64_t seen = 0;
    double lossSum = 0.0, errSum = 0.0;
    for (int b = 0; b < OUTPUT_BUCKETS; ++b) {
        seen += all.count[b];
        lossSum += all.loss[b];
        errSum += all.absErr[b];
    }
    if (seen == 0) {
        std::fprintf(stderr, "no records read\n");
        return 1;
    }

    const auto n1 = static_cast<double>(seen);
    std::printf("loss                 %.7f\n", lossSum / n1);
    std::printf("mean |eval - score|  %.1f cp\n", errSum / n1);
    std::printf("mean |score|         %.1f cp\n", all.absScore / n1);
    std::printf("label L/D/W          %.2f%% / %.2f%% / %.2f%%\n",
                100.0 * static_cast<double>(all.result[0]) / n1,
                100.0 * static_cast<double>(all.result[1]) / n1,
                100.0 * static_cast<double>(all.result[2]) / n1);
    std::printf("records              %llu\n\n", static_cast<unsigned long long>(seen));

    std::printf("bucket     records        loss   |err| cp\n");
    for (int b = 0; b < OUTPUT_BUCKETS; ++b) {
        if (all.count[b] == 0) { std::printf("%6d           0           -          -\n", b); continue; }
        const auto c = static_cast<double>(all.count[b]);
        std::printf("%6d  %10llu   %.7f     %6.1f\n", b,
                    static_cast<unsigned long long>(all.count[b]),
                    all.loss[b] / c, all.absErr[b] / c);
    }
    return 0;
}
