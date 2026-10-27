#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "../../board/board.hpp"

namespace syzygy {

// WDL result from tb_probe_wdl(), side-to-move relative.
enum class WDL : int8_t { Loss = -2, BlessedLoss = -1, Draw = 0, CursedWin = 1, Win = 2 };

struct RootMove {
    chess::Move move;
    int32_t     tbRank; // higher = better
};

class SyzygyProber {
public:
    bool load(const std::string& path);
    int  maxPieces() const noexcept;
    // Both probes give up (nullopt / empty) without tables or outside their range.
    // The WDL one ignores the fifty-move counter; the root one ranks every move by DTZ.
    std::optional<WDL> probeWDL(const chess::Board& board) const;
    std::vector<RootMove> probeRoot(const chess::Board& board) const;
    // Ply-anchored TB_WIN_SCORE for a win (negated for a loss), 0 for every draw.
    static int32_t wdlToScore(WDL wdl, int ply) noexcept;

    // Minimum search depth for WDL probing (UCI SyzygyProbeDepth).
    int probeDepth = 1;

private:
    bool loaded_ = false;
    bool inTBRange(const chess::Board& board) const noexcept;
};

} // namespace syzygy
