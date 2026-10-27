#pragma once

// bulletformat writer/decoder for NNUE training data.
//
// BulletRecord mirrors bulletformat::ChessBoard byte-for-byte (32 bytes); the
// packing below is a literal transcription of ChessBoard::from_raw from
// https://github.com/jw1912/bulletformat/blob/main/src/chess.rs
//
// Stored positions are normalized to the side to move ("stm frame"): when
// black is to move every bitboard is vertically mirrored, colors are swapped
// and score/result flip sign, so the stored "white" is always the stm.
//   pcs    : one nibble per occupied square in ascending LERF order (a1=0),
//            low nibble first; nibble = (isOpponent << 3) | piece (0=P..5=K)
//   score  : stm-relative centipawns
//   result : 0 = stm loss, 1 = draw, 2 = stm win
//   ksq    : stm king square; oppKsq: opponent king square ^ 56 (its own view)
// Castling/en-passant/clocks are not part of the format.

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

#include "../board/board.hpp"

namespace NNUE {

struct BulletRecord {
    uint64_t occ = 0;
    uint8_t  pcs[16] = {};
    int16_t  score = 0;
    uint8_t  result = 0;
    uint8_t  ksq = 0;
    uint8_t  oppKsq = 0;
    uint8_t  extra[3] = {};
};

static_assert(sizeof(BulletRecord) == 32, "must match bulletformat::ChessBoard");
static_assert(std::is_trivially_copyable_v<BulletRecord>);

// A packed position waiting for the game outcome: bulletformat stores the
// result stm-relative, so the stm at record time must survive until game end.
struct PendingRecord {
    BulletRecord record;
    bool blackToMove = false;
};

// whiteScoreCp is white-relative; the result is filled in at game end.
[[nodiscard]] inline PendingRecord packPosition(const chess::Board& b, int16_t whiteScoreCp) noexcept {
    using chess::Board;
    PendingRecord out;
    out.blackToMove = (b.getActiveColor() == Board::BLACK);
    BulletRecord& r = out.record;
    r.score = out.blackToMove ? static_cast<int16_t>(-whiteScoreCp) : whiteScoreCp;
    // LERF square l of the stm frame is engine square l ^ 56, or l itself when
    // black moves: that frame is the board mirrored vertically.
    for (int l = 0, idx = 0; l < 64; ++l) {
        const uint8_t piece = b.get(static_cast<uint8_t>(out.blackToMove ? l : l ^ 56));
        if (piece == Board::EMPTY) continue;
        const bool opp = (piece & Board::MASK_COLOR) != b.getActiveColor();
        const int type = (piece & Board::MASK_PIECE_TYPE) - 1;
        r.occ |= 1ULL << l;
        r.pcs[idx / 2] |= static_cast<uint8_t>(((opp ? 8 : 0) | type) << (4 * (idx & 1)));
        ++idx;
        if (type == 5) (opp ? r.oppKsq : r.ksq) = static_cast<uint8_t>(opp ? l ^ 56 : l);
    }
    return out;
}

// FEN of the stored (stm-frame) position, for eyeball checks in datagen-dump.
// The stm is printed as White; castling/ep are not stored, so "w - -".
[[nodiscard]] inline std::string recordToFen(const BulletRecord& r) {
    std::array<char, 64> squares;
    squares.fill('.');
    uint64_t occ = r.occ;
    for (int idx = 0; occ != 0; ++idx) {
        squares[pieces::popLSB(occ) ^ 56] = "PNBRQK??pnbrqk??"[(r.pcs[idx / 2] >> (4 * (idx & 1))) & 0xF];
    }
    return chess::placementFen(squares) + " w - - 0 1";
}

} // namespace NNUE
