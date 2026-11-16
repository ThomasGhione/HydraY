#pragma once

// Piece + threat accumulator and un-swizzled threat loader shared by standalone
// tools (reorder.cpp, l1census.cpp) without linking Board.

#include "fen_accumulator.hpp"
#include "../threats.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace NNUE::Deep::tools {

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

[[nodiscard]] inline uint64_t slowPieceAttacksLerf(int piece, int sq, uint64_t occ) noexcept {
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

inline bool verifyAttackTables() noexcept {
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

inline bool parseFenNormPos(const std::string& fen, NormPos& out) noexcept {
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

inline NormPos recordToNormPos(const Record& r) noexcept {
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

inline bool isBinaryPositionsFile(const char* path) {
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

inline std::vector<NormPos> loadPositions(const char* path, uint64_t limit, bool& isBinaryOut) {
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

inline void accumulatePos(const NetworkDeep& net, const int8_t* rawL0t, const NormPos& pos,
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
inline bool loadNetAndRawThreats(const char* path, NetworkDeep& net, std::vector<int8_t>& rawL0t) {
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

} // namespace NNUE::Deep::tools
