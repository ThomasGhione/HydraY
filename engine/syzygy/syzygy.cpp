#include "syzygy.hpp"

extern "C" {
#include "tbprobe.h"
}

#include <vector>

#include "../search/search_constants.hpp"

namespace syzygy {

// TB win/loss scores sit just inside MATE_BOUND so they're treated as
// decisive but ranked below actual checkmates. Blessed/cursed results
// (50-move rule) are scored as 0 (draw).
static constexpr int32_t TB_WIN_SCORE  = engine::MATE_BOUND - 500;
static constexpr int32_t TB_LOSS_SCORE = -TB_WIN_SCORE;

bool SyzygyProber::load(const std::string& path) {
    loaded_ = tb_init(path.c_str());
    return loaded_;
}

int SyzygyProber::maxPieces() const noexcept {
    return TB_LARGEST;
}

uint64_t SyzygyProber::sideOccupancy(const chess::Board& b, int side) noexcept {
    return b.pawns_bb[side] | b.knights_bb[side] | b.bishops_bb[side]
         | b.rooks_bb[side] | b.queens_bb[side] | b.kings_bb[side];
}

bool SyzygyProber::inTBRange(const chess::Board& board) const noexcept {
    if (!loaded_) return false;
    // All pieces regardless of color — popcount of the full occupancy.
    const int pieces = __builtin_popcountll(sideOccupancy(board, 0) | sideOccupancy(board, 1));
    return pieces <= TB_LARGEST;
}

int32_t SyzygyProber::wdlToScore(WDL wdl, int ply) noexcept {
    switch (wdl) {
        case WDL::Win:         return TB_WIN_SCORE  - ply;
        case WDL::Loss:        return TB_LOSS_SCORE + ply;
        case WDL::BlessedLoss: return 0;
        case WDL::CursedWin:   return 0;
        case WDL::Draw:        return 0;
    }
    return 0;
}

uint64_t SyzygyProber::toPyrrhicBB(uint64_t bb) noexcept {
    return __builtin_bswap64(bb);
}

unsigned SyzygyProber::toPyrrhicEP(chess::Square ep) noexcept {
    if (!chess::isValidSquare(ep)) return 0;
    return static_cast<unsigned>(56 ^ ep);
}

// HydraY convention: bb[0] = White, bb[1] = Black (colorToIndex(WHITE) == 0).
std::optional<WDL> SyzygyProber::probeWDL(const chess::Board& board) const {
    if (!loaded_) return std::nullopt;

    const uint64_t allW = toPyrrhicBB(sideOccupancy(board, 0));
    const uint64_t allB = toPyrrhicBB(sideOccupancy(board, 1));

    if (__builtin_popcountll(allW | allB) > TB_LARGEST) return std::nullopt;

    const bool turn = (board.getActiveColor() == chess::Board::WHITE);

    const unsigned result = tb_probe_wdl(
        allW, allB,
        toPyrrhicBB(board.kings_bb[0]   | board.kings_bb[1]),
        toPyrrhicBB(board.queens_bb[0]  | board.queens_bb[1]),
        toPyrrhicBB(board.rooks_bb[0]   | board.rooks_bb[1]),
        toPyrrhicBB(board.bishops_bb[0] | board.bishops_bb[1]),
        toPyrrhicBB(board.knights_bb[0] | board.knights_bb[1]),
        toPyrrhicBB(board.pawns_bb[0]   | board.pawns_bb[1]),
        toPyrrhicEP(board.getEnPassant()),
        turn);

    if (result == TB_RESULT_FAILED) return std::nullopt;
    return static_cast<WDL>(static_cast<int>(result) - 2);
}

std::vector<RootMove> SyzygyProber::probeRoot(const chess::Board& board) const {
    if (!loaded_) return {};

    const uint64_t allW = toPyrrhicBB(sideOccupancy(board, 0));
    const uint64_t allB = toPyrrhicBB(sideOccupancy(board, 1));

    if (__builtin_popcountll(allW | allB) > TB_LARGEST) return {};

    const bool turn = (board.getActiveColor() == chess::Board::WHITE);

    // Use tb_probe_root (not tb_probe_root_dtz) because we need per-move DTZ.
    // tb_probe_root_dtz collapses every "guaranteed win" to TB_MAX_DTZ when
    // cnt50 + dtz <= 99, which is the common case — that erases the
    // distinction between "promote in 1" and "shuffle the king for 30 plies".
    // The probe_root results[] array carries the actual DTZ per move, which
    // lets the search pick the fastest conversion.
    unsigned results[TB_MAX_MOVES + 1] = {0};
    const unsigned r = tb_probe_root(
        allW, allB,
        toPyrrhicBB(board.kings_bb[0]   | board.kings_bb[1]),
        toPyrrhicBB(board.queens_bb[0]  | board.queens_bb[1]),
        toPyrrhicBB(board.rooks_bb[0]   | board.rooks_bb[1]),
        toPyrrhicBB(board.bishops_bb[0] | board.bishops_bb[1]),
        toPyrrhicBB(board.knights_bb[0] | board.knights_bb[1]),
        toPyrrhicBB(board.pawns_bb[0]   | board.pawns_bb[1]),
        static_cast<unsigned>(board.getHalfMoveClock()),
        toPyrrhicEP(board.getEnPassant()),
        turn,
        results);

    if (r == TB_RESULT_FAILED) return {};
    if (r == TB_RESULT_CHECKMATE || r == TB_RESULT_STALEMATE) return {};

    // Rank scale: large enough that Win > 0 > Loss with room for DTZ
    // tiebreaks. INT32 headroom is comfortable.
    constexpr int32_t WIN_BASE  =  1'000'000'000;
    constexpr int32_t LOSS_BASE = -1'000'000'000;

    std::vector<RootMove> out;
    out.reserve(TB_MAX_MOVES);
    for (unsigned i = 0; i < TB_MAX_MOVES + 1 && results[i] != TB_RESULT_FAILED; ++i) {
        const unsigned res  = results[i];
        const unsigned wdl  = TB_GET_WDL(res);
        const unsigned dtz  = TB_GET_DTZ(res);
        const unsigned from = TB_GET_FROM(res);
        const unsigned to   = TB_GET_TO(res);
        const unsigned promotes = TB_GET_PROMOTES(res);

        uint8_t promo = 0;
        switch (promotes) {
            case PYRRHIC_FLAG_QPROMO: promo = chess::Board::QUEEN;  break;
            case PYRRHIC_FLAG_RPROMO: promo = chess::Board::ROOK;   break;
            case PYRRHIC_FLAG_BPROMO: promo = chess::Board::BISHOP; break;
            case PYRRHIC_FLAG_NPROMO: promo = chess::Board::KNIGHT; break;
            default: break;
        }

        // tbRank: higher is better. Smaller DTZ beats larger DTZ on the
        // winning side; larger DTZ (slower loss) beats smaller DTZ on the
        // losing side. Cursed Win (WDL=3) and Blessed Loss (WDL=1) are
        // 50-move-rule draws — score them as draws to avoid trading a
        // real win for one.
        int32_t rank = 0;
        if (wdl == TB_WIN)        rank = WIN_BASE  - static_cast<int32_t>(dtz);
        else if (wdl == TB_LOSS)  rank = LOSS_BASE + static_cast<int32_t>(dtz);
        // TB_DRAW, TB_CURSED_WIN, TB_BLESSED_LOSS → rank 0.

        out.push_back({
            chess::Move{
                static_cast<uint8_t>(56 ^ from),
                static_cast<uint8_t>(56 ^ to),
                promo},
            rank});
    }
    return out;
}

} // namespace syzygy
