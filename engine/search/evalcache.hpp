#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

#include "../../tt/tt.hpp"

namespace engine {

// Shared cache of raw NNUE static evals, keyed by Zobrist hash. The TT carries
// a static eval only for nodes that store an entry; nodes that exit through
// NMP, RFP or ProbCut store nothing, and this keeps their eval. A hit returns
// exactly what the forward pass would, so only the time changes.
//
// One 64-bit word per slot, direct-mapped: the upper 48 bits of the key, then
// the eval. Relaxed atomic words make a racing writer cost a miss, never a torn
// value. The 48-bit tag makes a false hit, empty slots included, a 2^-48 event.
class EvalCache {
public:
    EvalCache() : table_(allocateHugePages<uint64_t>(SLOTS)) {
        if (!table_) throw std::bad_alloc{};
        clear();
    }

    void clear() noexcept { std::fill_n(table_.get(), SLOTS, 0); }

    [[nodiscard]] bool probe(uint64_t key, int32_t& outEval) const noexcept {
        const uint64_t slot = word(key).load(std::memory_order_relaxed);
        if ((slot >> 16) != (key >> 16)) return false;
        outEval = static_cast<int16_t>(slot);
        return true;
    }

    void store(uint64_t key, int32_t eval) noexcept {
        word(key).store((key & ~0xFFFFULL) | static_cast<uint16_t>(eval), std::memory_order_relaxed);
    }

private:
    static constexpr size_t SLOTS = 1u << 21; // 16 MiB
    HugePageArray<uint64_t> table_;

    [[nodiscard]] std::atomic_ref<uint64_t> word(uint64_t key) const noexcept {
        return std::atomic_ref<uint64_t>(table_[key & (SLOTS - 1)]);
    }
};

} // namespace engine
