#pragma once

#include <cstdint>
#include <cstring>
#include <utility>  // for std::move

#include "../../board/board.hpp"
#include "../movelist.hpp"
#include "../search/search_constants.hpp"

namespace engine {

// Defined in sorter.cpp; declared here to break the sorter<->movepicker
// circular dependency (finalizeSee needs SEE, which lives in Sorter).
int32_t computeSeeForPicker(const chess::Board& b, const chess::Move& m) noexcept;

// Deferred-SEE state of a move in the picker: its score is either already
// final, or a capture / quiet whose SEE (good-vs-bad split, hanging-quiet
// demotion) is resolved lazily by MovePicker::finalizeSee.
enum class SeePending : uint8_t { Final = 0, Capture = 1, Quiet = 2 };

// Incremental lazy-selection picker. Parallel arrays (moves/scores/seePending)
// are kept in sync by nextMove() and fullSort(); never manipulate them separately.
//
// Like MoveList, only entries [0, size) are ever live: producers MUST fill both
// scores[i] and seePending[i] for every i < size before the picker is read or
// copied. The arrays are intentionally left uninitialised past `size`, and the
// copy/move ops below touch only [0, size) -- so on the hot quiescence path the
// per-node array traffic is O(size), not O(MAX_MOVES).
struct MovePicker {
    MoveList moves;
    int32_t    scores[MAX_MOVES];
    SeePending seePending[MAX_MOVES];
    int  currentIndex = 0;
    const chess::Board* board = nullptr; // SEE source for deferred finalisation

    MovePicker() noexcept = default;
    MovePicker(const MovePicker& o) noexcept { copyFrom(o); }
    MovePicker(MovePicker&& o) noexcept { moveFrom(std::move(o)); }
    MovePicker& operator=(const MovePicker& o) noexcept {
        if (this != &o) copyFrom(o);
        return *this;
    }
    MovePicker& operator=(MovePicker&& o) noexcept {
        if (this != &o) moveFrom(std::move(o));
        return *this;
    }

    inline bool hasNext() const noexcept { return currentIndex < moves.size; }

    // Resolve a deferred capture/quiet into its final score (good capture kept,
    // losing capture or hanging quiet demoted). Returns true if the score dropped
    // (so the caller must re-pick the max).
    inline bool finalizeSee(int idx) noexcept {
        const SeePending pending = seePending[idx];
        if (pending == SeePending::Final) return false;
        seePending[idx] = SeePending::Final;
        const int32_t see = computeSeeForPicker(*board, moves[idx]);
        if (see >= 0) return false;                      // good capture / safe quiet: unchanged
        if (pending == SeePending::Capture) {            // losing capture
            scores[idx] = -CAPTURE_BASE_SCORE + see;
        } else {                                         // quiet that hangs material
            const int32_t cap = KILLER_2_SCORE - 1;
            scores[idx] = (scores[idx] < cap ? scores[idx] : cap) + see;
        }
        return true;
    }

    inline chess::Move nextMove() noexcept {
        while (currentIndex < moves.size) {
            int bestIdx = currentIndex;
            int32_t bestScore = scores[currentIndex];
            for (int i = currentIndex + 1; i < moves.size; ++i) {
                if (scores[i] > bestScore) {
                    bestScore = scores[i];
                    bestIdx = i;
                }
            }

            // Deferred SEE: finalise the top candidate; only when it actually
            // demotes itself do we re-pick (a kept move is still the max).
            if (finalizeSee(bestIdx)) {
                continue;
            }

            if (bestIdx != currentIndex) {
                std::swap(moves[currentIndex],      moves[bestIdx]);
                std::swap(scores[currentIndex],     scores[bestIdx]);
                std::swap(seePending[currentIndex], seePending[bestIdx]);
            }

            return moves[currentIndex++];
        }
        return {};
    }

    // Full descending insertion sort over [0, moves.size). Used by the root search.
    inline void fullSort() noexcept {
        for (int i = 0; i < moves.size; ++i) finalizeSee(i); // resolve deferred SEE before sorting
        for (int i = 1; i < moves.size; ++i) {
            const auto    keyMove  = moves[i];
            const int32_t keyScore = scores[i];
            int j = i - 1;
            while (j >= 0 && scores[j] < keyScore) {
                scores[j + 1] = scores[j];
                moves[j + 1]  = moves[j];
                --j;
            }
            scores[j + 1] = keyScore;
            moves[j + 1]  = keyMove;
        }
    }

private:
    // Copy/move only the live prefix [0, moves.size); scores and seePending past
    // it are indeterminate and must never be read. Mirrors MoveList's
    // size-bounded copyFrom/moveFrom. `moves` must be assigned before this runs
    // (it is the size authority; o.moves.size is already 0 after a move).
    inline void copyLivePrefix(const MovePicker& o) noexcept {
        board        = o.board;
        currentIndex = o.currentIndex;
        const size_t n = static_cast<size_t>(moves.size);
        std::memcpy(scores,     o.scores,     n * sizeof(scores[0]));
        std::memcpy(seePending, o.seePending, n * sizeof(seePending[0]));
    }

    inline void copyFrom(const MovePicker& o) noexcept {
        moves = o.moves;
        copyLivePrefix(o);
    }

    inline void moveFrom(MovePicker&& o) noexcept {
        moves = std::move(o.moves);
        copyLivePrefix(o);
        o.currentIndex = 0;
    }
};

} // namespace engine
