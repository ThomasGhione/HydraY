#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "zobrist.hpp"

class alignas(64) TT {
public:
    struct Entry {
        // `key` holds zobristKey ^ payload ("lockless hashing", Hyatt): a reader
        // recovers the identity as key ^ payload, so writes from two threads
        // interleaved on one slot read as a miss. Each word is accessed through
        // atomic_ref, so neither can tear on its own.
        uint64_t key = 0;
        uint64_t payload = 0;

        enum Flag : uint8_t { INVALID = 0, EXACT, LOWERBOUND, UPPERBOUND };

        // "No static eval recorded", outside the reachable eval range.
        static constexpr int32_t NO_EVAL = -32768;
        static constexpr uint8_t MAX_DEPTH = 63;

        // Move::promotionType (0 or KNIGHT..QUEEN) fits the 4 promo bits as-is.
        static constexpr uint16_t encodeMove(const chess::Move& m) noexcept {
            return (static_cast<uint16_t>(m.from) & 0x3F)
                 | ((static_cast<uint16_t>(m.to) & 0x3F) << 6)
                 | ((static_cast<uint16_t>(m.promotionType) & 0xF) << 12);
        }

        static constexpr chess::Move decodeMove(uint16_t encoded) noexcept {
            return chess::Move{
                static_cast<uint8_t>(encoded & 0x3F),
                static_cast<uint8_t>((encoded >> 6) & 0x3F),
                static_cast<uint8_t>((encoded >> 12) & 0xF)
            };
        }
    };

    // The payload word. Both scores are int16_t: every search score fits (see
    // the static_assert on MATE_VALUE + MAX_PLY in search_constants.hpp).
    struct Data {
        uint8_t  depth : 6  = 0;
        uint8_t  flag  : 2  = Entry::INVALID;
        uint8_t  age        = 0;
        uint16_t move       = 0;
        int16_t  score      = 0;
        int16_t  staticEval = Entry::NO_EVAL;
    };

    // A miss leaves the defaults: flag INVALID, move 0, no static eval.
    struct ProbeResult : Data {
        bool hit = false;
    };

    static constexpr size_t ENTRIES_PER_BUCKET = 4;
    static constexpr size_t DEFAULT_HASH_MB = 64;
    static constexpr size_t MIN_HASH_MB = 1;
    static constexpr size_t MAX_HASH_MB = 4096;

    static_assert(sizeof(Entry) == 16 && sizeof(Data) == 8, "4 entries per 64-byte cache line");

    TT() {
        if (!resize(DEFAULT_HASH_MB)) throw std::bad_alloc{};
    }

    void prefetch(uint64_t key) const noexcept { __builtin_prefetch(bucketFor(key), 0, 3); }

    // The one hot-path read: a single bucket snapshot per node feeds the TT
    // cutoff, static-eval tightening, singular-extension gate and hash-move
    // ordering. Bound and depth gating is the caller's job.
    [[nodiscard]] ProbeResult probeEntry(uint64_t key) const noexcept {
        const Entry* bucket = bucketFor(key);
        for (size_t i = 0; i < ENTRIES_PER_BUCKET; ++i) {
            const uint64_t storedKey = atomicWord(bucket[i].key).load(std::memory_order_relaxed);
            const uint64_t payload   = atomicWord(bucket[i].payload).load(std::memory_order_relaxed);
            const Data data = std::bit_cast<Data>(payload);
            if ((storedKey ^ payload) == key && data.flag != Entry::INVALID) return {data, true};
        }
        return {};
    }

    // Cold-path convenience: the stored move for `key`, decoded ({} = none).
    [[nodiscard]] chess::Move probeDecodedMove(uint64_t key) const noexcept {
        const uint16_t move = probeEntry(key).move;
        return move != 0 ? Entry::decodeMove(move) : chess::Move{};
    }

    // bestMove == 0 is a bound-only write and keeps the move of a matching
    // entry; staticEval == NO_EVAL keeps its eval the same way (the eval
    // depends on the position alone).
    void store(uint64_t key, uint8_t depth, int32_t score, uint8_t flag,
               uint16_t bestMove = 0, int32_t staticEval = Entry::NO_EVAL) noexcept;

    // Resizes to `megabytes` MiB, rounded down to a power-of-two bucket count,
    // and clears. Not during a search. On allocation failure the table stays.
    bool resize(size_t megabytes) noexcept {
        const size_t buckets = std::bit_floor(std::clamp(megabytes, MIN_HASH_MB, MAX_HASH_MB)
                                              * 1024 * 1024 / (ENTRIES_PER_BUCKET * sizeof(Entry)));
        if (buckets == bucketMask_ + 1) {
            clear();
            return true;
        }
        auto* fresh = static_cast<Entry*>(::operator new(buckets * ENTRIES_PER_BUCKET * sizeof(Entry),
                                                         TABLE_ALIGNMENT, std::nothrow));
        if (fresh == nullptr) return false;
#if defined(MADV_HUGEPAGE)
        ::madvise(fresh, buckets * ENTRIES_PER_BUCKET * sizeof(Entry), MADV_HUGEPAGE);
#endif
        table_.reset(fresh);
        bucketMask_ = buckets - 1;
        clear();
        return true;
    }

    [[nodiscard]] size_t sizeMB() const noexcept {
        return (bucketMask_ + 1) * ENTRIES_PER_BUCKET * sizeof(Entry) / (1024 * 1024);
    }

    void incrementGeneration() noexcept { ++generation_; }
    void clear() noexcept { std::fill_n(table_.get(), (bucketMask_ + 1) * ENTRIES_PER_BUCKET, Entry{}); }

private:
    // 2 MiB alignment lets Linux back the table with transparent huge pages.
    static constexpr std::align_val_t TABLE_ALIGNMENT{2 * 1024 * 1024};
    struct FreeTable {
        void operator()(Entry* p) const noexcept { ::operator delete(p, TABLE_ALIGNMENT); }
    };

    std::unique_ptr<Entry[], FreeTable> table_;
    size_t bucketMask_ = 0;
    uint8_t generation_ = 0;

    [[nodiscard]] Entry* bucketFor(uint64_t key) const noexcept {
        return &table_[(key & bucketMask_) * ENTRIES_PER_BUCKET];
    }

    [[nodiscard]] static std::atomic_ref<uint64_t> atomicWord(const uint64_t& word) noexcept {
        return std::atomic_ref<uint64_t>(const_cast<uint64_t&>(word));
    }

    static void write(Entry& entry, uint64_t key, Data data) noexcept {
        const uint64_t payload = std::bit_cast<uint64_t>(data);
        atomicWord(entry.payload).store(payload, std::memory_order_relaxed);
        atomicWord(entry.key).store(key ^ payload, std::memory_order_relaxed);
    }
};

inline void TT::store(uint64_t key, uint8_t depth, int32_t score, uint8_t flag,
                      uint16_t bestMove, int32_t staticEval) noexcept {
    Data fresh;
    fresh.depth = std::min(depth, Entry::MAX_DEPTH);
    fresh.flag = flag;
    fresh.age = generation_;
    fresh.move = bestMove;
    fresh.score = static_cast<int16_t>(score);
    fresh.staticEval = static_cast<int16_t>(staticEval);

    Entry* bucket = bucketFor(key);
    Entry* replaceEntry = &bucket[0];
    Entry* emptyEntry = nullptr;
    int bestReplaceScore = INT32_MIN;

    for (size_t i = 0; i < ENTRIES_PER_BUCKET; ++i) {
        Entry& entry = bucket[i];
        const uint64_t entryKey = atomicWord(entry.key).load(std::memory_order_relaxed);
        const uint64_t entryPayload = atomicWord(entry.payload).load(std::memory_order_relaxed);
        const Data old = std::bit_cast<Data>(entryPayload);
        if (old.flag == Entry::INVALID) {
            if (emptyEntry == nullptr) emptyEntry = &entry;
            continue;
        }

        const int ageDiff = static_cast<uint8_t>(generation_ - old.age);
        // A torn entry fails the identity test and is a replacement candidate.
        if ((entryKey ^ entryPayload) == key) {
            // An entry from an earlier search is overwritten even by a shallower
            // result: otherwise it keeps its old age and is the next one evicted.
            if (fresh.depth >= old.depth || flag == Entry::EXACT || ageDiff != 0) {
                if (bestMove == 0) fresh.move = old.move;
                if (staticEval == Entry::NO_EVAL) fresh.staticEval = old.staticEval;
                write(entry, key, fresh);
            } else if (bestMove != 0 && flag != Entry::UPPERBOUND && bestMove != old.move) {
                // The deeper bound stays, but a newer move that raised alpha is
                // still the better ordering hint (a fail-low "best" move is not).
                Data updated = old;
                updated.move = bestMove;
                write(entry, key, updated);
            }
            return;
        }

        // Evict the lowest depth - 8 * age: one generation is worth 8 plies, so a
        // deep entry from the previous search outlives a shallow current one.
        const int replaceScore = (ageDiff << 3) - old.depth;
        if (replaceScore > bestReplaceScore) {
            bestReplaceScore = replaceScore;
            replaceEntry = &entry;
        }
    }
    write(emptyEntry != nullptr ? *emptyEntry : *replaceEntry, key, fresh);
}

constexpr TT::Entry::Flag determineFlag(int32_t score, int32_t alphaOrig, int32_t beta) noexcept {
    if (score <= alphaOrig) return TT::Entry::UPPERBOUND;
    if (score >= beta) return TT::Entry::LOWERBOUND;
    return TT::Entry::EXACT;
}

static_assert(TT::Entry::decodeMove(TT::Entry::encodeMove({12, 28, chess::Board::KNIGHT}))
              == chess::Move{12, 28, chess::Board::KNIGHT}, "move encoding round trip");
static_assert(determineFlag(100, 50, 200) == TT::Entry::EXACT, "determineFlag logic error");
static_assert(determineFlag(40, 50, 200) == TT::Entry::UPPERBOUND, "determineFlag logic error");
static_assert(determineFlag(250, 50, 200) == TT::Entry::LOWERBOUND, "determineFlag logic error");
