#pragma once
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "../../board/board.hpp"

namespace syzygy {

// WDL result from tb_probe_wdl(), side-to-move relative.
enum class WDL : int8_t { Loss = -2, BlessedLoss = -1, Draw = 0, CursedWin = 1, Win = 2 };

struct RootMove {
    chess::Move move;
    int32_t            tbRank; // higher = better; use to order root moves
};

class SyzygyProber {
public:
    bool load(const std::string& path);
    bool isLoaded() const noexcept { return loaded_; }
    int  maxPieces() const noexcept;

    // WDL probe for use inside search (no 50-move rule consideration).
    // Returns nullopt when position has too many pieces or probe fails.
    std::optional<WDL> probeWDL(const chess::Board& board) const;

    // Root probe: returns a ranked list of moves with DTZ info.
    // Returns empty when probe fails or position not in TB.
    std::vector<RootMove> probeRoot(const chess::Board& board) const;

    // Convert a WDL result to a search score (side-to-move relative):
    // ply-anchored TB_WIN_SCORE/TB_LOSS_SCORE for wins/losses, 0 for draws.
    static int32_t wdlToScore(WDL wdl, int ply) noexcept;

    // Returns true if the position's piece count is within TB range.
    bool inTBRange(const chess::Board& board) const noexcept;

    // Minimum search depth for WDL probing (set via UCI SyzygyProbeDepth).
    int probeDepth = 1;

private:
    bool loaded_ = false;

    static uint64_t toPyrrhicBB(uint64_t bb) noexcept;
    static unsigned toPyrrhicEP(chess::Square ep) noexcept;
    // Union of every piece bitboard for one side (0 = White, 1 = Black).
    static uint64_t sideOccupancy(const chess::Board& b, int side) noexcept;
};

} // namespace syzygy
