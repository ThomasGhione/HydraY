#include "syzygy.hpp"

extern "C" {
#include "tbprobe.h"
}

#include <bit>

#include "../search/search_constants.hpp"

// Pyrrhic's attacks (tbconfig.h) from the engine's tables. Its squares run from
// a1 and ours from a8: square s is s ^ 56, and a bitboard is byteswapped.
extern "C" {
uint64_t hydray_tb_pawn_attacks(int sq, int white) {
    return std::byteswap(pieces::PAWN_ATTACKS[white ? 0 : 1][sq ^ 56]);
}
uint64_t hydray_tb_knight_attacks(int sq) {
    return std::byteswap(pieces::KNIGHT_ATTACKS[sq ^ 56]);
}
uint64_t hydray_tb_bishop_attacks(int sq, uint64_t occ) {
    return std::byteswap(pieces::getBishopAttacks(static_cast<uint8_t>(sq ^ 56), std::byteswap(occ)));
}
uint64_t hydray_tb_rook_attacks(int sq, uint64_t occ) {
    return std::byteswap(pieces::getRookAttacks(static_cast<uint8_t>(sq ^ 56), std::byteswap(occ)));
}
uint64_t hydray_tb_king_attacks(int sq) {
    return std::byteswap(pieces::KING_ATTACKS[sq ^ 56]);
}
}

namespace syzygy {

namespace {

// Pyrrhic takes LERF bitboards (a1 = 0); ours put a8 at 0, a vertical mirror.
struct PyrrhicPosition {
    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    unsigned ep;
    bool whiteToMove;

    explicit PyrrhicPosition(const chess::Board& b) noexcept
        : white(std::byteswap(b.sideOcc(0))), black(std::byteswap(b.sideOcc(1))),
          kings(both(b, chess::Board::KING)), queens(both(b, chess::Board::QUEEN)),
          rooks(both(b, chess::Board::ROOK)), bishops(both(b, chess::Board::BISHOP)),
          knights(both(b, chess::Board::KNIGHT)), pawns(both(b, chess::Board::PAWN)),
          ep(chess::isValidSquare(b.getEnPassant()) ? 56u ^ b.getEnPassant() : 0u),
          whiteToMove(b.getActiveColor() == chess::Board::WHITE) {}

    static uint64_t both(const chess::Board& b, int type) noexcept {
        return std::byteswap(b.bb[type][0] | b.bb[type][1]);
    }
};

} // namespace

bool SyzygyProber::load(const std::string& path) {
    loaded_ = tb_init(path.c_str());
    return loaded_;
}

int SyzygyProber::maxPieces() const noexcept {
    return TB_LARGEST;
}

bool SyzygyProber::inTBRange(const chess::Board& board) const noexcept {
    return loaded_ && std::popcount(board.getPiecesBitMap()) <= TB_LARGEST;
}

int32_t SyzygyProber::wdlToScore(WDL wdl, int ply) noexcept {
    switch (wdl) {
        case WDL::Win:  return engine::TB_WIN_SCORE - ply;
        case WDL::Loss: return -engine::TB_WIN_SCORE + ply;
        default:        return 0; // the fifty-move rule makes cursed wins and blessed losses draws
    }
}

std::optional<WDL> SyzygyProber::probeWDL(const chess::Board& board) const {
    if (!inTBRange(board)) return std::nullopt;
    const PyrrhicPosition p(board);
    const unsigned result = tb_probe_wdl(p.white, p.black, p.kings, p.queens, p.rooks,
                                         p.bishops, p.knights, p.pawns, p.ep, p.whiteToMove);
    if (result == TB_RESULT_FAILED) return std::nullopt;
    return static_cast<WDL>(static_cast<int>(result) - 2);
}

std::vector<RootMove> SyzygyProber::probeRoot(const chess::Board& board) const {
    if (!inTBRange(board)) return {};
    const PyrrhicPosition p(board);

    // tb_probe_root, not tb_probe_root_dtz: the latter collapses every win with
    // cnt50 + dtz <= 99 to TB_MAX_DTZ, so "promote in 1" and "shuffle for 30
    // plies" would rank the same. results[] carries each move's own DTZ.
    unsigned results[TB_MAX_MOVES + 1] = {0};
    const unsigned r = tb_probe_root(p.white, p.black, p.kings, p.queens, p.rooks, p.bishops,
                                     p.knights, p.pawns, static_cast<unsigned>(board.getHalfMoveClock()),
                                     p.ep, p.whiteToMove, results);
    if (r == TB_RESULT_FAILED || r == TB_RESULT_CHECKMATE || r == TB_RESULT_STALEMATE) return {};

    static constexpr uint8_t PROMOTION[8] = {0, chess::Board::QUEEN, chess::Board::ROOK,
                                             chess::Board::BISHOP, chess::Board::KNIGHT, 0, 0, 0};
    std::vector<RootMove> out;
    for (unsigned i = 0; i < TB_MAX_MOVES + 1 && results[i] != TB_RESULT_FAILED; ++i) {
        const unsigned res = results[i];
        // A faster win and a slower loss rank higher; every draw ranks 0.
        const int32_t dtz = static_cast<int32_t>(TB_GET_DTZ(res));
        const int32_t rank = TB_GET_WDL(res) == TB_WIN  ?  1'000'000'000 - dtz
                           : TB_GET_WDL(res) == TB_LOSS ? -1'000'000'000 + dtz : 0;
        out.push_back({chess::Move{static_cast<uint8_t>(56 ^ TB_GET_FROM(res)),
                                   static_cast<uint8_t>(56 ^ TB_GET_TO(res)),
                                   PROMOTION[TB_GET_PROMOTES(res)]},
                       rank});
    }
    return out;
}

} // namespace syzygy
