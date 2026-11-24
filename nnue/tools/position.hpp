#pragma once

// A position for the offline tools, from a FEN or a bulletformat record, and
// its two accumulators, piece and threat features, without a Board: a literal
// translation of sanity_deep.rs and threats.rs. In the engine, Board does this
// work through accumulator.hpp. If it is wrong, the comparisons with the
// oracle fail at once.

#include "../bulletformat.hpp"
#include "../network_deep.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace NNUE::Deep::tools {

using Record = BulletRecord;

// The engine's slider tables, filled before main as Board does.
inline const bool sliderTablesReady = (pieces::Rook::init(), pieces::Bishop::init(), true);

// The squares piece (0-11) on LERF square sq attacks with occ as the blockers.
inline uint64_t pieceAttacksLerf(int piece, int sq, uint64_t occ) noexcept {
    const auto e = static_cast<uint8_t>(sq ^ 56);
    const uint64_t o = std::byteswap(occ);
    switch (piece % 6) {
        case threats::BISHOP: return std::byteswap(pieces::Bishop::attacks(e, o));
        case threats::ROOK:   return std::byteswap(pieces::Rook::attacks(e, o));
        case threats::QUEEN:  return std::byteswap(pieces::Bishop::attacks(e, o) | pieces::Rook::attacks(e, o));
        default:              return threats::EMPTY_BOARD_ATTACKS[piece][sq];
    }
}

struct PieceEntry {
    uint8_t type;
    bool    isOpp;
    uint8_t sq; // LERF in stm view
};

// The side to move's view: its king's basis, the opponent's, and the pieces.
struct NormPos {
    int        usFlip = 0, usBase = 0;
    int        themFlip = 0, themBase = 0;
    int        outBucket = 0;
    int        count = 0;
    PieceEntry pieces[64]{};
};

// False on a malformed FEN (a missing king).
inline bool parseFenNormPos(const std::string& fen, NormPos& out) noexcept {
    const size_t sp = fen.find(' ');
    const bool blackToMove = sp != std::string::npos && sp + 1 < fen.size() && fen[sp + 1] == 'b';
    out = {};
    int wk = -1, bk = -1;
    int rank = 7, file = 0;
    for (const char c : fen.substr(0, sp)) {
        if (c == '/') { --rank; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        const size_t letter = std::string_view("pnbrqk").find(ascii::toLower(c));
        if (letter == std::string_view::npos || rank < 0 || file < 0 || file >= 8) return false;
        const int type = static_cast<int>(letter);
        const bool isBlack = c >= 'a';
        const int sq = rank * 8 + file++;
        if (type == 5) (isBlack ? bk : wk) = sq;
        out.pieces[out.count++] = {static_cast<uint8_t>(type), isBlack != blackToMove,
                                   static_cast<uint8_t>(blackToMove ? sq ^ 56 : sq)};
    }
    if (wk < 0 || bk < 0) return false;
    const int ownUs = blackToMove ? (bk ^ 56) : wk;
    const int ownThem = blackToMove ? wk : (bk ^ 56);
    out.usFlip    = kingFlip(ownUs);
    out.usBase    = kingFeatureBase(ownUs);
    out.themFlip  = kingFlip(ownThem);
    out.themBase  = kingFeatureBase(ownThem);
    out.outBucket = (out.count - 2) / BUCKET_DIVISOR;
    return true;
}

// Records are stored in the stm frame: the stm's king is ksq, and oppKsq is
// already the opponent king in its own view.
inline NormPos recordToNormPos(const Record& r) noexcept {
    NormPos out;
    out.usFlip   = kingFlip(r.ksq);
    out.usBase   = kingFeatureBase(r.ksq);
    out.themFlip = kingFlip(r.oppKsq);
    out.themBase = kingFeatureBase(r.oppKsq);
    for (uint64_t occ = r.occ; occ != 0 && out.count < 32; ) {
        const int sq = pieces::popLSB(occ);
        const uint8_t nibble = (r.pcs[out.count / 2] >> (4 * (out.count & 1))) & 0xF;
        out.pieces[out.count++] = {static_cast<uint8_t>(nibble & 7), (nibble & 8) != 0, static_cast<uint8_t>(sq)};
    }
    out.outBucket = (out.count - 2) / BUCKET_DIVISOR;
    return out;
}

// Calls f(us, them) with the l0 row of each perspective for every piece.
template<class F>
void forEachFeature(const NormPos& pos, F&& f) {
    for (int i = 0; i < pos.count; ++i) {
        const PieceEntry& pe = pos.pieces[i];
        f(pos.usBase + (((pe.isOpp ? 384 : 0) + pe.type * 64 + pe.sq) ^ pos.usFlip),
          pos.themBase + (((pe.isOpp ? 0 : 384) + pe.type * 64 + (pe.sq ^ 56)) ^ pos.themFlip));
    }
}

// Up to `count` records from record `offset` of a file, fewer at its end.
inline std::vector<Record> readRecords(const char* path, uint64_t offset, uint64_t count) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(offset * sizeof(Record)));
    std::vector<Record> recs(count);
    in.read(reinterpret_cast<char*>(recs.data()), static_cast<std::streamsize>(count * sizeof(Record)));
    recs.resize(static_cast<size_t>(in.gcount()) / sizeof(Record));
    return recs;
}

inline bool isBinaryPositionsFile(const char* path) {
    const std::string s(path);
    if (s.ends_with(".bin")) return true;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const auto sz = in.tellg();
    if (sz <= 0 || static_cast<uint64_t>(sz) % sizeof(Record) != 0) return false;
    in.seekg(0);
    char buf[512];
    in.read(buf, std::min<std::streamsize>(sizeof(buf), sz));
    for (std::streamsize i = 0; i < in.gcount(); ++i) {
        const auto ch = static_cast<unsigned char>(buf[i]);
        if (ch == 0 || (ch < 0x20 && ch != '\n' && ch != '\r' && ch != '\t')) return true;
    }
    return false;
}

// One FEN per line, or bulletformat records (default cap 20,000).
inline std::vector<NormPos> loadPositions(const char* path, uint64_t limit, bool& isBinaryOut) {
    std::vector<NormPos> positions;
    isBinaryOut = isBinaryPositionsFile(path);
    if (isBinaryOut) {
        for (const Record& r : readRecords(path, 0, limit > 0 ? limit : 20000ULL)) positions.push_back(recordToNormPos(r));
        return positions;
    }
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line) && (limit == 0 || positions.size() < limit)) {
        NormPos pos;
        if (!line.empty() && line[0] != '#' && parseFenNormPos(line, pos)) positions.push_back(pos);
    }
    return positions;
}

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
        const __m256i b16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi16(a, b16));
    }
#else
    for (int i = 0; i < HIDDEN; ++i) dst[i] = static_cast<int16_t>(dst[i] + static_cast<int16_t>(w[i]));
#endif
}

// The threats of the piece on sqView, in one perspective's view (board and occ).
inline void addThreatPerspective(const uint8_t* board, uint64_t occ, uint8_t sqView, uint8_t pieceView,
                                 const int8_t* rawL0t, int16_t* __restrict acc) noexcept {
    const auto& rowPairs = threats::TABLE.pairBase[pieceView];
    for (uint64_t victims = pieceAttacksLerf(pieceView, sqView, occ) & occ; victims != 0; ) {
        const int to = pieces::popLSB(victims);
        uint32_t feat = 0;
        const uint16_t base = rowPairs[board[to]];
        if (base != threats::INVALID_PAIR_BASE && threats::tryIndexBase(base + sqView, to, feat))
            addI8Row(acc, rawL0t + static_cast<size_t>(feat) * HIDDEN);
    }
}

// Both accumulators of pos: modular i16 piece rows, plus the threat rows when
// rawL0t (the on-disk i8 weights) is given.
inline void accumulatePos(const NetworkDeep& net, const int8_t* rawL0t, const NormPos& pos,
                          int16_t* __restrict us, int16_t* __restrict them) noexcept {
    std::memcpy(us, net.l0b, sizeof(net.l0b));
    std::memcpy(them, net.l0b, sizeof(net.l0b));
    forEachFeature(pos, [&](int u, int t) {
        addI16Row(us, net.l0w[u]);
        addI16Row(them, net.l0w[t]);
    });
    if (rawL0t == nullptr) return;

    uint8_t board0[64], board1[64];
    uint64_t occ0 = 0, occ1 = 0;
    struct ViewPiece { uint8_t sq0, p0, sq1, p1; } viewPieces[64];
    for (int i = 0; i < pos.count; ++i) {
        const PieceEntry& pe = pos.pieces[i];
        ViewPiece& v = viewPieces[i];
        v = {static_cast<uint8_t>(pe.sq ^ pos.usFlip), static_cast<uint8_t>((pe.isOpp ? 6 : 0) + pe.type),
             static_cast<uint8_t>(pe.sq ^ 56 ^ pos.themFlip), static_cast<uint8_t>((pe.isOpp ? 0 : 6) + pe.type)};
        board0[v.sq0] = v.p0;
        board1[v.sq1] = v.p1;
        occ0 |= 1ULL << v.sq0;
        occ1 |= 1ULL << v.sq1;
    }
    for (int i = 0; i < pos.count; ++i) {
        addThreatPerspective(board0, occ0, viewPieces[i].sq0, viewPieces[i].p0, rawL0t, us);
        addThreatPerspective(board1, occ1, viewPieces[i].sq1, viewPieces[i].p1, rawL0t, them);
    }
}

// Loads a net via the official loader and also keeps the on-disk i8 threat
// weights, which loadFromMemory rearranges into the AVX2 lane order.
inline bool loadNetAndRawThreats(const char* path, NetworkDeep& net, std::vector<int8_t>& rawL0t) {
    std::ifstream nf(path, std::ios::binary | std::ios::ate);
    const auto sz = nf.tellg();
    if (!nf || sz <= 0) return false;
    std::vector<unsigned char> blob(static_cast<size_t>(sz));
    nf.seekg(0);
    if (!nf.read(reinterpret_cast<char*>(blob.data()), sz) || !loadFromMemory(blob.data(), blob.size(), net)) return false;
    const size_t off = payloadBytes(net.fileBuckets, false);
    rawL0t.assign(net.fileThreats ? blob.begin() + static_cast<std::ptrdiff_t>(off) : blob.end(),
                  net.fileThreats ? blob.begin() + static_cast<std::ptrdiff_t>(off + THREAT_BYTES) : blob.end());
    return true;
}

inline double sigmoid(double x) noexcept { return 1.0 / (1.0 + std::exp(-x)); }

inline bool parseU64(const char* s, uint64_t& out) {
    return std::from_chars(s, s + std::strlen(s), out).ec == std::errc{};
}

} // namespace NNUE::Deep::tools
