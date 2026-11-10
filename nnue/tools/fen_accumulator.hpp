#pragma once

// The accumulator of a FEN or of a bulletformat record, for the tools outside
// the engine: a literal translation of sanity_deep.rs. In the engine, Board
// does this work through accumulator.hpp. If it is wrong, the comparisons with
// the oracle fail at once.

#include "../network_deep.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace NNUE::Deep::tools {

struct Piece { int type; bool black; int sq; };

// False on a malformed FEN (a missing king).
bool accumulate(const NetworkDeep& net, const std::string& fen,
                std::vector<int16_t>& accUs, std::vector<int16_t>& accThem,
                int& outBucket) {
    const size_t sp = fen.find(' ');
    const std::string board = fen.substr(0, sp);
    const bool blackToMove = sp != std::string::npos && fen[sp + 1] == 'b';

    std::vector<Piece> pieces;
    int wk = -1, bk = -1;
    int rank = 7, file = 0;
    for (const char c : board) {
        if (c == '/') { --rank; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        int type;
        switch (std::tolower(static_cast<unsigned char>(c))) {
            case 'p': type = 0; break; case 'n': type = 1; break;
            case 'b': type = 2; break; case 'r': type = 3; break;
            case 'q': type = 4; break; case 'k': type = 5; break;
            default: return false;
        }
        const bool black = std::islower(static_cast<unsigned char>(c)) != 0;
        const int sq = rank * 8 + file;
        if (type == 5) (black ? bk : wk) = sq;
        pieces.push_back({type, black, sq});
        ++file;
    }
    if (wk < 0 || bk < 0) return false;

    const auto params = [&](bool fromBlackView, int& base, int& flip) {
        const int own = fromBlackView ? (bk ^ 56) : wk;
        flip = kingFlip(own);
        base = kingFeatureBase(own);
    };
    int usBase, usFlip, themBase, themFlip;
    params(blackToMove, usBase, usFlip);
    params(!blackToMove, themBase, themFlip);

    accUs.assign(net.l0b, net.l0b + HIDDEN);
    accThem.assign(net.l0b, net.l0b + HIDDEN);

    for (const auto& p : pieces) {
        const struct { std::vector<int16_t>* acc; bool view; int base; int flip; } sides[2] = {
            {&accUs,   blackToMove,  usBase,   usFlip},
            {&accThem, !blackToMove, themBase, themFlip},
        };
        for (const auto& s : sides) {
            const bool isOpp = (p.black != s.view);
            const int sqView = s.view ? (p.sq ^ 56) : p.sq;
            const int feat768 = (isOpp ? 384 : 0) + p.type * 64 + sqView;
            const int feature = s.base + (feat768 ^ s.flip);
            const int16_t* w = net.l0w[feature];
            for (int i = 0; i < HIDDEN; ++i) (*s.acc)[i] = static_cast<int16_t>((*s.acc)[i] + w[i]);
        }
    }

    outBucket = (static_cast<int>(pieces.size()) - 2) / BUCKET_DIVISOR;
    return true;
}

// bulletformat::ChessBoard, 32 bytes. Declared here rather than included from
// nnue/bulletformat.hpp, which pulls in board.hpp for the writer side; the
// tools only read. The field meanings are documented there.
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

// Calls f(us, them) with the l0 row of each perspective for every piece of the
// record and returns its output bucket. Records are stored in the stm frame
// (the stm is always "white" and its king is ksq; oppKsq is already the
// opponent king in its own view), so the black-to-move branch of the FEN path
// collapses away. holdoutloss --selfcheck holds the two paths against each other.
template<class F>
int forEachFeature(const Record& r, F&& f) {
    const int usFlip   = kingFlip(r.ksq);
    const int usBase   = kingFeatureBase(r.ksq);
    const int themFlip = kingFlip(r.oppKsq);
    const int themBase = kingFeatureBase(r.oppKsq);
    uint64_t occ = r.occ;
    int idx = 0;
    while (occ != 0) {
        const int sq = std::countr_zero(occ);
        occ &= occ - 1;
        const uint8_t nibble = (r.pcs[idx / 2] >> (4 * (idx & 1))) & 0xF;
        ++idx;
        const bool isOpp = (nibble & 8) != 0;
        const int  type  = nibble & 7;
        f(usBase + (((isOpp ? 384 : 0) + type * 64 + sq) ^ usFlip),
          themBase + (((isOpp ? 0 : 384) + type * 64 + (sq ^ 56)) ^ themFlip));
    }
    return (idx - 2) / BUCKET_DIVISOR;
}

// The engine's two accumulators of a record; returns its output bucket.
inline int accumulateRecord(const NetworkDeep& net, const Record& r, int16_t* us, int16_t* them) noexcept {
    std::memcpy(us, net.l0b, sizeof(net.l0b));
    std::memcpy(them, net.l0b, sizeof(net.l0b));
    return forEachFeature(r, [&](int u, int t) {
        for (int i = 0; i < HIDDEN; ++i) {
            us[i]   = static_cast<int16_t>(us[i] + net.l0w[u][i]);
            them[i] = static_cast<int16_t>(them[i] + net.l0w[t][i]);
        }
    });
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

inline double sigmoid(double x) noexcept { return 1.0 / (1.0 + std::exp(-x)); }

inline bool parseU64(const char* s, uint64_t& out) {
    return std::from_chars(s, s + std::strlen(s), out).ec == std::errc{};
}

} // namespace NNUE::Deep::tools
