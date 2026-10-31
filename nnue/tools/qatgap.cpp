// Quantisation gap of a deep net: its float checkpoint against the quantised
// file the engine plays, over a bulletformat range.
//
//   g++ -std=c++23 -O2 -march=native nnue/tools/qatgap.cpp nnue/nnue_deep.cpp -o /tmp/qatgap
//   /tmp/qatgap <checkpoint>/optimiser_state/weights.bin <checkpoint>/quantised.bin <data.bin>
//               [--offset N] [--count N]
//
// raw.bin, next to quantised.bin, is NOT the float net: it holds l0w without the
// factoriser. weights.bin has every tensor, as bullet keeps them: "id\n", a u64
// count, the f32 values, column-major. The tool checks that rounding them gives
// quantised.bin before it measures anything.
//
// Four forwards per position, each giving a loss (bullet's, as in holdoutloss.cpp)
// and an evaluation:
//
//   A  float weights, float activations: what bullet trains
//   D  as A, l1w rounded to 1/QB
//   B  every weight rounded (l0 to 1/QA, l1w to 1/QB), float activations
//   C  the engine: B plus integer accumulators and the truncating pairwise
//
// A-D is the cost of l1w's rounding, D-B of l0's, B-C of the truncation: the three
// things HYDRAY_QAT emulates in trainer_deep.rs. loss C - loss A is the most QAT
// can recover. C is checked against the engine's forwardScalar: a handful of
// mismatches is the float summation order landing on the other side of a 0.5 cp
// rounding.
//
// 600k records of v8_std.t0 (holdout), rms cp of each step and loss C - A:
//   antidead-160sb  l1w 14.8, l0 3.9, truncation 2.6, total 15.7   2.14e-5
//   ob8-160sb       l1w 16.9, l0 3.7, truncation 2.5, total 16.9   3.56e-5  (the 4.2.0 run)

#include "fen_accumulator.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace NNUE::Deep;

namespace {

// bulletformat::ChessBoard, as in holdoutloss.cpp.
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

// The float net in the quantised file's order: l0w with the factoriser folded in,
// row-major [feature][hidden]; l1w [bucket][out][in]; l2w [bucket][out].
struct FloatNet {
    int ob = 0;
    std::vector<float> l0w, l0b, l1w, l1b, l2w, l2b;
};

bool loadWeights(const char* path, FloatNet& net) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    const std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> l0w, l0f, l1w, l2w;
    for (size_t p = 0; p < d.size();) {
        const auto nl = std::find(d.begin() + static_cast<std::ptrdiff_t>(p), d.end(), '\n');
        if (nl == d.end()) return false;
        const auto e = static_cast<size_t>(nl - d.begin());
        const std::string id(d.begin() + static_cast<std::ptrdiff_t>(p), nl);
        uint64_t n = 0;
        if (e + 9 > d.size()) return false;
        std::memcpy(&n, &d[e + 1], 8);
        if (e + 9 + n * 4 > d.size()) return false;
        std::vector<float> v(n);
        std::memcpy(v.data(), &d[e + 9], n * 4);
        p = e + 9 + n * 4;
        if (id == "l0w") l0w = std::move(v);
        else if (id == "l0f") l0f = std::move(v);
        else if (id == "l0b") net.l0b = std::move(v);
        else if (id == "l1w") l1w = std::move(v);
        else if (id == "l1b") net.l1b = std::move(v);
        else if (id == "l2w") l2w = std::move(v);
        else if (id == "l2b") net.l2b = std::move(v);
    }
    const size_t rows = static_cast<size_t>(INPUT_BUCKETS) * INPUTS;
    net.ob = static_cast<int>(net.l2b.size());
    const size_t n1 = static_cast<size_t>(net.ob) * L1_SIZE;
    if (l0w.size() != rows * HIDDEN || l0f.size() != size_t(INPUTS) * HIDDEN || net.l0b.size() != HIDDEN
        || l1w.size() != n1 * HIDDEN || net.l1b.size() != n1 || l2w.size() != n1) {
        return false;
    }
    // Column-major (outputs x inputs) stores each input's outputs contiguously:
    // l0w is already [feature][hidden], l1w and l2w need the transpose.
    net.l0w.resize(l0w.size());
    for (size_t r = 0; r < rows; ++r)
        for (int h = 0; h < HIDDEN; ++h)
            net.l0w[r * HIDDEN + h] = l0w[r * HIDDEN + h] + l0f[(r % INPUTS) * HIDDEN + h];
    net.l1w.resize(l1w.size());
    for (size_t o = 0; o < n1; ++o)
        for (int i = 0; i < HIDDEN; ++i) net.l1w[o * HIDDEN + i] = l1w[i * n1 + o];
    net.l2w.resize(n1);
    for (int b = 0; b < net.ob; ++b)
        for (int o = 0; o < L1_SIZE; ++o) net.l2w[b * L1_SIZE + o] = l2w[o * net.ob + b];
    return true;
}

float screlu(float x) noexcept {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x;
}

double sigmoid(double x) noexcept { return 1.0 / (1.0 + std::exp(-x)); }

bool parseU64(const char* s, uint64_t& out) {
    const char* end = s + std::strlen(s);
    return std::from_chars(s, end, out).ec == std::errc{};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: qatgap <weights.bin> <quantised.bin> <data.bin> [--offset N] [--count N]\n");
        return 2;
    }
    uint64_t offset = 0, count = 600'000;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        const bool hasValue = i + 1 < argc;
        if (a == "--offset" && hasValue)     { if (!parseU64(argv[++i], offset)) return 2; }
        else if (a == "--count" && hasValue) { if (!parseU64(argv[++i], count)) return 2; }
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return 2; }
    }

    static FloatNet fl;
    if (!loadWeights(argv[1], fl)) { std::fprintf(stderr, "cannot read %s as bullet weights\n", argv[1]); return 1; }
    static NetworkDeep q;
    if (!loadFromFile(argv[2], q)) { std::fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
    if (fl.ob != q.fileBuckets) {
        std::fprintf(stderr, "%d output buckets in the weights, %d in the net\n", fl.ob, q.fileBuckets);
        return 1;
    }
    // A file bucket covers OUTPUT_BUCKETS / ob consecutive engine buckets.
    const int span = OUTPUT_BUCKETS / fl.ob;

    // The two files must be the same net: quantised == round(float * Q). A tie
    // rounded the other way can differ by one in a few places.
    long bad0 = 0, bad1 = 0;
    for (size_t r = 0; r < size_t(INPUT_BUCKETS) * INPUTS; ++r)
        for (int h = 0; h < HIDDEN; ++h)
            if (std::lround(fl.l0w[r * HIDDEN + h] * QA) != q.l0w[r][h]) ++bad0;
    for (int b = 0; b < fl.ob; ++b)
        for (int o = 0; o < L1_SIZE; ++o)
            for (int i = 0; i < HIDDEN; ++i)
                if (std::lround(fl.l1w[(size_t(b) * L1_SIZE + o) * HIDDEN + i] * QB) != q.l1w[b * span][o][i]) ++bad1;
    std::printf("same net: %ld l0w and %ld l1w values differ from round(float * Q)\n", bad0, bad1);
    if (bad0 > 100 || bad1 > 100) { std::fprintf(stderr, "not the same net\n"); return 1; }

    std::ifstream in(argv[3], std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    in.seekg(static_cast<std::streamoff>(offset * sizeof(Record)));
    std::vector<Record> recs(count);
    in.read(reinterpret_cast<char*>(recs.data()), static_cast<std::streamsize>(count * sizeof(Record)));
    const auto got = static_cast<long>(in.gcount() / static_cast<std::streamsize>(sizeof(Record)));
    if (got == 0) { std::fprintf(stderr, "no records\n"); return 1; }

    enum { A, D, B, C, N };
    double loss[N] = {}, sq[N] = {}, mean[N] = {}, worst = 0.0;
    long mismatches = 0;
    std::vector<float> fUs(HIDDEN), fThem(HIDDEN);
    std::vector<int16_t> iUs(HIDDEN), iThem(HIDDEN);
    std::vector<float> pf(HIDDEN), pq(HIDDEN);
    std::vector<int32_t> pi(HIDDEN);

    for (long n = 0; n < got; ++n) {
        const Record& r = recs[n];
        // Features as holdoutloss.cpp's accumulateRecord, on both nets at once.
        const int usFlip   = (r.ksq & 7) > 3 ? 7 : 0;
        const int usBase   = INPUTS * tools::kingBucket(r.ksq);
        const int themFlip = (r.oppKsq & 7) > 3 ? 7 : 0;
        const int themBase = INPUTS * tools::kingBucket(r.oppKsq);
        for (int h = 0; h < HIDDEN; ++h) {
            fUs[h] = fThem[h] = fl.l0b[h];
            iUs[h] = iThem[h] = q.l0b[h];
        }
        uint64_t occ = r.occ;
        int idx = 0;
        while (occ != 0) {
            const int sq = std::countr_zero(occ);
            occ &= occ - 1;
            const uint8_t nibble = (r.pcs[idx / 2] >> (4 * (idx & 1))) & 0xF;
            ++idx;
            const bool isOpp = (nibble & 8) != 0;
            const int  type  = nibble & 7;
            const size_t us   = usBase + (((isOpp ? 384 : 0) + type * 64 + sq) ^ usFlip);
            const size_t them = themBase + (((isOpp ? 0 : 384) + type * 64 + (sq ^ 56)) ^ themFlip);
            for (int h = 0; h < HIDDEN; ++h) {
                fUs[h] += fl.l0w[us * HIDDEN + h];
                fThem[h] += fl.l0w[them * HIDDEN + h];
                iUs[h] = static_cast<int16_t>(iUs[h] + q.l0w[us][h]);
                iThem[h] = static_cast<int16_t>(iThem[h] + q.l0w[them][h]);
            }
        }
        const int bucket = (idx - 2) / BUCKET_DIVISOR;
        const int fb = bucket / span;

        for (int side = 0; side < 2; ++side) {
            const float*   fa = side == 0 ? fUs.data() : fThem.data();
            const int16_t* ia = side == 0 ? iUs.data() : iThem.data();
            for (int j = 0; j < PAIRWISE_OUT; ++j) {
                const int o = side * PAIRWISE_OUT + j;
                pf[o] = std::clamp(fa[j], 0.0f, 1.0f) * std::clamp(fa[j + PAIRWISE_OUT], 0.0f, 1.0f);
                const int32_t x = std::clamp<int32_t>(ia[j], 0, QA);
                const int32_t y = std::clamp<int32_t>(ia[j + PAIRWISE_OUT], 0, QA);
                pq[o] = (static_cast<float>(x) / QA) * (static_cast<float>(y) / QA);
                pi[o] = x * y / QA;
            }
        }

        double y[N];
        double yA = fl.l2b[fb], yD = fl.l2b[fb], yB = q.l2b[bucket], yC = q.l2b[bucket];
        for (int o = 0; o < L1_SIZE; ++o) {
            const float* wf = &fl.l1w[(size_t(fb) * L1_SIZE + o) * HIDDEN];
            const int8_t* wq = q.l1w[bucket][o];
            float zA = 0.0f, zD = 0.0f, zB = 0.0f;
            int32_t zC = 0;
            for (int i = 0; i < HIDDEN; ++i) {
                const float w = static_cast<float>(wq[i]) / QB;
                zA += pf[i] * wf[i];
                zD += pf[i] * w;
                zB += pq[i] * w;
                zC += pi[i] * wq[i];
            }
            const float l2 = fl.l2w[fb * L1_SIZE + o];
            yA += screlu(zA + fl.l1b[fb * L1_SIZE + o]) * l2;
            yD += screlu(zD + fl.l1b[fb * L1_SIZE + o]) * l2;
            yB += screlu(zB + q.l1b[bucket][o]) * q.l2w[bucket][o];
            yC += screlu(static_cast<float>(zC) / static_cast<float>(QA * QB) + q.l1b[bucket][o]) * q.l2w[bucket][o];
        }
        y[A] = yA; y[D] = yD; y[B] = yB; y[C] = yC;
        if (std::lround(static_cast<float>(yC) * SCALE) != forwardScalar(q, iUs.data(), iThem.data(), bucket)) ++mismatches;

        const double target = 0.3 * (r.result / 2.0) + 0.7 * sigmoid(r.score / 400.0);
        for (int k = 0; k < N; ++k) {
            const double d = sigmoid(y[k]) - target;
            loss[k] += d * d;
        }
        // Steps A->D->B->C, in centipawns of the training unit.
        for (int k = 1; k < N; ++k) {
            const double step = (y[k - 1] - y[k]) * SCALE;
            sq[k] += step * step;
            mean[k] += step;
        }
        const double total = (y[A] - y[C]) * SCALE;
        sq[A] += total * total;
        worst = std::max(worst, std::abs(total));
    }

    std::printf("%ld positions from record %llu; engine forward differs on %ld\n\n", got,
                static_cast<unsigned long long>(offset), mismatches);
    std::printf("loss  A float            %.7f\n", loss[A] / got);
    std::printf("loss  D l1w rounded      %.7f\n", loss[D] / got);
    std::printf("loss  B weights rounded  %.7f\n", loss[B] / got);
    std::printf("loss  C engine           %.7f   (C - A = %.2e, the most QAT can recover)\n\n",
                loss[C] / got, (loss[C] - loss[A]) / got);
    const char* what[N] = {"", "l1w rounding (A-D)", "l0 rounding  (D-B)", "truncation   (B-C)"};
    for (int k = 1; k < N; ++k)
        std::printf("%s  rms %6.2f cp, mean %+6.2f cp\n", what[k], std::sqrt(sq[k] / got), mean[k] / got);
    std::printf("total        (A-C)  rms %6.2f cp, worst %.1f cp\n", std::sqrt(sq[A] / got), worst);
    return 0;
}
