#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#endif

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
    // On the heap like the TT, so it can be backed by transparent huge pages:
    // a probe lands on a random slot, and with 4 KiB pages most of them also
    // missed the TLB.
    EvalCache() : table_(static_cast<uint64_t*>(::operator new(BYTES, ALIGNMENT))) {
#if defined(MADV_HUGEPAGE)
        ::madvise(table_.get(), BYTES, MADV_HUGEPAGE);
#endif
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
    static constexpr size_t BYTES = SLOTS * sizeof(uint64_t);
    static constexpr std::align_val_t ALIGNMENT{2 * 1024 * 1024};
    struct Free {
        void operator()(uint64_t* p) const noexcept { ::operator delete(p, ALIGNMENT); }
    };
    std::unique_ptr<uint64_t[], Free> table_;

    [[nodiscard]] std::atomic_ref<uint64_t> word(uint64_t key) const noexcept {
        return std::atomic_ref<uint64_t>(table_[key & (SLOTS - 1)]);
    }
};

} // namespace engine
