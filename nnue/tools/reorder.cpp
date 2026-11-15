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

#include "fen_accumulator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace NNUE::Deep;
using NNUE::Deep::tools::Record;
using NNUE::Deep::tools::parseU64;
using NNUE::Deep::tools::readRecords;

namespace {

// ---------------------------------------------------------------------------
// Fast LERF (a1 = 0) attack generation matching nnue/trainer/src/threats.rs
// ---------------------------------------------------------------------------

constexpr std::array<uint64_t, 64> makeLineMask(int d1x, int d1y, int d2x, int d2y) noexcept {
    std::array<uint64_t, 64> m{};
    for (int sq = 0; sq < 64; ++sq) {
        uint64_t out = 0;
        for (const auto& [dx, dy] : {std::pair{d1x, d1y}, std::pair{d2x, d2y}}) {
            for (int f = (sq & 7) + dx, r = (sq >> 3) + dy;
                 f >= 0 && f < 8 && r >= 0 && r < 8;
                 f += dx, r += dy) {
                out |= 1ULL << (r * 8 + f);
            }
        }
        m[sq] = out;
    }
    return m;
}

inline constexpr auto FILE_MASK     = makeLineMask(0, 1, 0, -1);
inline constexpr auto DIAG_MASK     = makeLineMask(1, 1, -1, -1);
inline constexpr auto ANTIDIAG_MASK = makeLineMask(1, -1, -1, 1);

inline constexpr auto RANK_ATTACKS = [] {
    std::array<std::array<uint64_t, 64>, 64> tbl{};
    for (int sq = 0; sq < 64; ++sq) {
        const int r  = sq >> 3;
        const int f0 = sq & 7;
        for (int inner = 0; inner < 64; ++inner) {
            const uint64_t occ = static_cast<uint64_t>(inner) << (r * 8 + 1);
            uint64_t out = 0;
            for (int f = f0 + 1; f < 8; ++f) {
                const uint64_t b = 1ULL << (r * 8 + f);
                out |= b;
                if (occ & b) break;
            }
            for (int f = f0 - 1; f >= 0; --f) {
                const uint64_t b = 1ULL << (r * 8 + f);
                out |= b;
                if (occ & b) break;
            }
            tbl[sq][inner] = out;
        }
    }
    return tbl;
}();

[[nodiscard]] inline uint64_t hyperbolaLine(int sq, uint64_t occ, uint64_t lineMask) noexcept {
    const uint64_t sbit = 1ULL << sq;
    uint64_t fwd = occ & lineMask;
    uint64_t rev = std::byteswap(fwd);
    fwd -= 2ULL * sbit;
    rev -= 2ULL * std::byteswap(sbit);
    return (fwd ^ std::byteswap(rev)) & lineMask;
}

[[nodiscard]] inline uint64_t pieceAttacksLerf(int piece, int sq, uint64_t occ) noexcept {
    const int pt = piece % 6;
    if (pt == NNUE::threats::BISHOP) {
        return hyperbolaLine(sq, occ, DIAG_MASK[sq]) | hyperbolaLine(sq, occ, ANTIDIAG_MASK[sq]);
    }
    if (pt == NNUE::threats::ROOK) {
        const uint64_t inner = (occ >> ((sq & 56) + 1)) & 63ULL;
        return RANK_ATTACKS[sq][inner] | hyperbolaLine(sq, occ, FILE_MASK[sq]);
    }
    if (pt == NNUE::threats::QUEEN) {
        const uint64_t inner = (occ >> ((sq & 56) + 1)) & 63ULL;
        return RANK_ATTACKS[sq][inner] | hyperbolaLine(sq, occ, FILE_MASK[sq])
             | hyperbolaLine(sq, occ, DIAG_MASK[sq]) | hyperbolaLine(sq, occ, ANTIDIAG_MASK[sq]);
    }
    return NNUE::threats::EMPTY_BOARD_ATTACKS[piece][sq];
}

[[nodiscard]] uint64_t slowPieceAttacksLerf(int piece, int sq, uint64_t occ) noexcept {
    const int pt = piece % 6;
    if (pt != NNUE::threats::BISHOP && pt != NNUE::threats::ROOK && pt != NNUE::threats::QUEEN)
        return NNUE::threats::EMPTY_BOARD_ATTACKS[piece][sq];
    static constexpr int ORTH[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    static constexpr int DIAG[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    uint64_t out = 0;
    const auto cast = [&](const int (&dirs)[4][2]) {
        for (const auto& d : dirs) {
            for (int f = (sq & 7) + d[0], r = (sq >> 3) + d[1];
                 f >= 0 && f < 8 && r >= 0 && r < 8;
                 f += d[0], r += d[1]) {
                const uint64_t b = 1ULL << (r * 8 + f);
                out |= b;
                if (occ & b) break;
            }
        }
    };
    if (pt == NNUE::threats::BISHOP || pt == NNUE::threats::QUEEN) cast(DIAG);
    if (pt == NNUE::threats::ROOK   || pt == NNUE::threats::QUEEN) cast(ORTH);
    return out;
}

bool verifyAttackTables() noexcept {
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    const auto next64 = [&] {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return rng * 0x2545F4914F6CDD1DULL;
    };
    for (int iter = 0; iter < 5000; ++iter) {
        const uint64_t occ = next64() & next64();
        const int sq = iter % 64;
        for (int p = 0; p < 12; ++p)
            if (pieceAttacksLerf(p, sq, occ) != slowPieceAttacksLerf(p, sq, occ)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Normalised stm-white positions (shared by .fen and .bin inputs)
// ---------------------------------------------------------------------------

struct PieceEntry {
    uint8_t type;
    bool    isOpp;
    uint8_t sq; // LERF in stm view
};

struct NormPos {
    int        usFlip = 0, usBase = 0;
    int        themFlip = 0, themBase = 0;
    int        outBucket = 0;
    int        count = 0;
    PieceEntry pieces[32]{};
};

bool parseFenNormPos(const std::string& fen, NormPos& out) noexcept {
    const size_t sp = fen.find(' ');
    const std::string board = fen.substr(0, sp);
    const bool blackToMove = sp != std::string::npos && sp + 1 < fen.size() && fen[sp + 1] == 'b';

    out = {};
    int wk = -1, bk = -1;
    int rank = 7, file = 0;
    for (const char c : board) {
        if (c == '/') { --rank; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        if (rank < 0 || rank >= 8 || file < 0 || file >= 8 || out.count >= 32) return false;
        int type;
        switch (std::tolower(static_cast<unsigned char>(c))) {
            case 'p': type = 0; break; case 'n': type = 1; break;
            case 'b': type = 2; break; case 'r': type = 3; break;
            case 'q': type = 4; break; case 'k': type = 5; break;
            default: return false;
        }
        const bool isBlack = std::islower(static_cast<unsigned char>(c)) != 0;
        const int sq = rank * 8 + file;
        if (type == 5) (isBlack ? bk : wk) = sq;
        const bool isOpp = isBlack != blackToMove;
        const int  sqUs  = blackToMove ? (sq ^ 56) : sq;
        out.pieces[out.count++] = {static_cast<uint8_t>(type), isOpp, static_cast<uint8_t>(sqUs)};
        ++file;
    }
    if (wk < 0 || bk < 0 || out.count < 2) return false;

    const int ownUs   = blackToMove ? (bk ^ 56) : wk;
    const int ownThem = blackToMove ? wk : (bk ^ 56);
    out.usFlip    = NNUE::kingFlip(ownUs);
    out.usBase    = NNUE::kingFeatureBase(ownUs);
    out.themFlip  = NNUE::kingFlip(ownThem);
    out.themBase  = NNUE::kingFeatureBase(ownThem);
    out.outBucket = (out.count - 2) / BUCKET_DIVISOR;
    return true;
}

NormPos recordToNormPos(const Record& r) noexcept {
    NormPos out;
    out.usFlip   = NNUE::kingFlip(r.ksq);
    out.usBase   = NNUE::kingFeatureBase(r.ksq);
    out.themFlip = NNUE::kingFlip(r.oppKsq);
    out.themBase = NNUE::kingFeatureBase(r.oppKsq);
    uint64_t occ = r.occ;
    while (occ != 0 && out.count < 32) {
        const int sq = std::countr_zero(occ);
        occ &= occ - 1;
        const uint8_t nibble = (r.pcs[out.count / 2] >> (4 * (out.count & 1))) & 0xF;
        out.pieces[out.count++] = {
            static_cast<uint8_t>(nibble & 7),
            (nibble & 8) != 0,
            static_cast<uint8_t>(sq),
        };
    }
    out.outBucket = (out.count - 2) / BUCKET_DIVISOR;
    return out;
}

bool isBinaryPositionsFile(const char* path) {
    const std::string s(path);
    if (s.size() >= 4 && s.compare(s.size() - 4, 4, ".bin") == 0) return true;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const auto sz = in.tellg();
    if (sz <= 0 || static_cast<uint64_t>(sz) % sizeof(Record) != 0) return false;
    in.seekg(0);
    char buf[512];
    in.read(buf, std::min<std::streamsize>(sizeof(buf), sz));
    const auto n = in.gcount();
    for (std::streamsize i = 0; i < n; ++i) {
        const auto ch = static_cast<unsigned char>(buf[i]);
        if (ch == 0 || (ch < 0x20 && ch != '\n' && ch != '\r' && ch != '\t')) return true;
    }
    return false;
}

std::vector<NormPos> loadPositions(const char* path, uint64_t limit, bool& isBinaryOut) {
    std::vector<NormPos> positions;
    isBinaryOut = isBinaryPositionsFile(path);
    if (isBinaryOut) {
        const uint64_t want = limit > 0 ? limit : 20000ULL;
        const std::vector<Record> recs = readRecords(path, 0, want);
        positions.reserve(recs.size());
        for (const Record& r : recs) positions.push_back(recordToNormPos(r));
        return positions;
    }

    std::ifstream in(path);
    if (!in) return positions;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        NormPos pos;
        if (!parseFenNormPos(line, pos)) continue;
        positions.push_back(pos);
        if (limit > 0 && positions.size() >= limit) break;
    }
    return positions;
}

// ---------------------------------------------------------------------------
// Accumulator builder (modular i16 piece + raw disk i8 threat weights)
// ---------------------------------------------------------------------------

inline void addI16Row(int16_t* __restrict dst, const int16_t* __restrict w) noexcept {
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN; i += 16) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi16(a, b));
    }
#else
    for (int i = 0; i < HIDDEN; ++i) dst[i] = static_cast<int16_t>(dst[i] + w[i]);
#endif
}

inline void addI8Row(int16_t* __restrict dst, const int8_t* __restrict w) noexcept {
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN; i += 16) {
        const __m256i a   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m128i b8  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
        const __m256i b16 = _mm256_cvtepi8_epi16(b8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi16(a, b16));
    }
#else
    for (int i = 0; i < HIDDEN; ++i) dst[i] = static_cast<int16_t>(dst[i] + static_cast<int16_t>(w[i]));
#endif
}

inline void addThreatPerspective(const uint8_t* board, uint64_t occ, uint8_t sqView, uint8_t pieceView,
                                 const int8_t* rawL0t, int16_t* __restrict acc) noexcept {
    uint64_t victims = pieceAttacksLerf(pieceView, sqView, occ) & occ;
    const auto& rowPairs = NNUE::threats::TABLE.pairBase[pieceView];
    while (victims != 0) {
        const int to = std::countr_zero(victims);
        victims &= victims - 1;
        uint32_t feat = 0;
        const uint16_t base = rowPairs[board[to]];
        if (base != NNUE::threats::INVALID_PAIR_BASE
            && NNUE::threats::tryIndexBase(base + sqView, to, feat)) {
            addI8Row(acc, rawL0t + static_cast<size_t>(feat) * HIDDEN);
        }
    }
}

void accumulatePos(const NetworkDeep& net, const int8_t* rawL0t, const NormPos& pos,
                   int16_t* __restrict us, int16_t* __restrict them) noexcept {
    std::memcpy(us, net.l0b, sizeof(net.l0b));
    std::memcpy(them, net.l0b, sizeof(net.l0b));

    uint8_t board0[64];
    uint8_t board1[64];
    uint64_t occ0 = 0;
    uint64_t occ1 = 0;
    struct ViewPiece { uint8_t sq0, p0, sq1, p1; };
    ViewPiece viewPieces[32];
    const int toView1 = 56 ^ pos.themFlip;

    for (int i = 0; i < pos.count; ++i) {
        const PieceEntry& pe = pos.pieces[i];
        const int u = pos.usBase   + (((pe.isOpp ? 384 : 0) + pe.type * 64 + pe.sq)        ^ pos.usFlip);
        const int t = pos.themBase + (((pe.isOpp ? 0 : 384) + pe.type * 64 + (pe.sq ^ 56)) ^ pos.themFlip);
        addI16Row(us, net.l0w[u]);
        addI16Row(them, net.l0w[t]);

        if (rawL0t != nullptr) {
            const uint8_t sq0 = static_cast<uint8_t>(pe.sq ^ pos.usFlip);
            const uint8_t p0  = static_cast<uint8_t>((pe.isOpp ? 6 : 0) + pe.type);
            const uint8_t sq1 = static_cast<uint8_t>(pe.sq ^ toView1);
            const uint8_t p1  = static_cast<uint8_t>((pe.isOpp ? 0 : 6) + pe.type);
            board0[sq0] = p0;
            board1[sq1] = p1;
            occ0 |= 1ULL << sq0;
            occ1 |= 1ULL << sq1;
            viewPieces[i] = {sq0, p0, sq1, p1};
        }
    }

    if (rawL0t != nullptr) {
        for (int i = 0; i < pos.count; ++i) {
            addThreatPerspective(board0, occ0, viewPieces[i].sq0, viewPieces[i].p0, rawL0t, us);
            addThreatPerspective(board1, occ1, viewPieces[i].sq1, viewPieces[i].p1, rawL0t, them);
        }
    }
}

// Loads a net via the official loader and also preserves the un-swizzled on-disk
// i8 threat weights (loadFromMemory packs net.l0t into AVX2 lane order).
bool loadNetAndRawThreats(const char* path, NetworkDeep& net, std::vector<int8_t>& rawL0t) {
    std::ifstream nf(path, std::ios::binary | std::ios::ate);
    if (!nf) return false;
    const auto sz = nf.tellg();
    if (sz <= 0) return false;
    std::vector<unsigned char> blob(static_cast<size_t>(sz));
    nf.seekg(0);
    if (!nf.read(reinterpret_cast<char*>(blob.data()), sz)
        || !loadFromMemory(blob.data(), blob.size(), net)) {
        return false;
    }
    rawL0t.clear();
    if (net.fileThreats) {
        const size_t off = payloadBytes(net.fileBuckets, false);
        rawL0t.resize(THREAT_BYTES);
        std::memcpy(rawL0t.data(), blob.data() + off, THREAT_BYTES);
    }
    return true;
}

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
