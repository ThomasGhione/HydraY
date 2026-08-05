#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <string_view>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "zobrist.hpp"
#include "../ascii_utils.hpp"

class TT {

public:
    enum class HugePageMode : uint8_t {
        Auto = 0,
        On,
        Off
    };

    struct Entry {
        // 16-byte layout:
        // key(8) + payload(8)
        // payload packing:
        //   bits  0..15  -> packed(depth/age/flag)
        //   bits 16..31  -> bestMove
        //   bits 32..47  -> score (int16_t)
        //   bits 48..63  -> staticEval (int16_t)
        // 4 entries = 64 bytes (1 cache line).
        //
        // Both score fields are int16_t, which the caller must respect: every
        // search score fits by construction (see the static_assert on
        // MATE_VALUE + MAX_PLY in search_constants.hpp). A score outside that
        // range would wrap silently here.
        uint64_t key = 0ULL;
        uint64_t payload = 0ULL;

        enum Flag : uint8_t {
            INVALID = 0,
            EXACT,
            LOWERBOUND,
            UPPERBOUND
        };

        // "No static eval recorded". Outside the reachable eval range, so it can
        // never collide with a real one.
        static constexpr int32_t NO_EVAL = -32768;

        static constexpr uint8_t DEPTH_BITS = 6;      // 0..63 plies
        static constexpr uint8_t FLAG_BITS = 2;       // 0..3
        static constexpr uint8_t AGE_BITS = 8;        // 0..255 generations
        static constexpr uint8_t MAX_DEPTH = (1u << DEPTH_BITS) - 1u;

        static constexpr uint16_t DEPTH_MASK = static_cast<uint16_t>((1u << DEPTH_BITS) - 1u);
        static constexpr uint8_t FLAG_SHIFT = DEPTH_BITS;
        static constexpr uint16_t FLAG_MASK = static_cast<uint16_t>((1u << FLAG_BITS) - 1u);
        static constexpr uint8_t AGE_SHIFT = DEPTH_BITS + FLAG_BITS;

        static constexpr uint16_t packedMeta(uint8_t depthValue, uint8_t ageValue, uint8_t flagValue) noexcept {
            const uint16_t d = static_cast<uint16_t>(depthValue);
            const uint16_t a = static_cast<uint16_t>(ageValue);
            const uint16_t f = static_cast<uint16_t>(flagValue & FLAG_MASK);
            return static_cast<uint16_t>(d | (f << FLAG_SHIFT) | (a << AGE_SHIFT));
        }

        static constexpr uint16_t packedFromPayload(uint64_t payloadValue) noexcept {
            return static_cast<uint16_t>(payloadValue & 0xFFFFULL);
        }

        static constexpr uint8_t depthFromPayload(uint64_t payloadValue) noexcept {
            return static_cast<uint8_t>(packedFromPayload(payloadValue) & DEPTH_MASK);
        }

        static constexpr uint8_t flagFromPayload(uint64_t payloadValue) noexcept {
            return static_cast<uint8_t>((packedFromPayload(payloadValue) >> FLAG_SHIFT) & FLAG_MASK);
        }

        static constexpr uint8_t ageFromPayload(uint64_t payloadValue) noexcept {
            return static_cast<uint8_t>(packedFromPayload(payloadValue) >> AGE_SHIFT);
        }

        static constexpr uint16_t bestMoveFromPayload(uint64_t payloadValue) noexcept {
            return static_cast<uint16_t>((payloadValue >> 16) & 0xFFFFULL);
        }

        // Both narrow to int16_t, whose conversion back to int32_t sign-extends.
        static constexpr int32_t scoreFromPayload(uint64_t payloadValue) noexcept {
            return static_cast<int16_t>(static_cast<uint16_t>(payloadValue >> 32));
        }

        static constexpr int32_t staticEvalFromPayload(uint64_t payloadValue) noexcept {
            return static_cast<int16_t>(static_cast<uint16_t>(payloadValue >> 48));
        }

        static constexpr uint64_t encodePayload(
            int32_t scoreValue,
            int32_t staticEvalValue,
            uint16_t bestMoveValue,
            uint8_t depthValue,
            uint8_t ageValue,
            uint8_t flagValue) noexcept {
            return (static_cast<uint64_t>(static_cast<uint16_t>(static_cast<int16_t>(staticEvalValue))) << 48)
                 | (static_cast<uint64_t>(static_cast<uint16_t>(static_cast<int16_t>(scoreValue))) << 32)
                 | (static_cast<uint64_t>(bestMoveValue) << 16)
                 | static_cast<uint64_t>(packedMeta(depthValue, ageValue, flagValue));
        }

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

    static constexpr size_t ENTRIES_PER_BUCKET = 4;
    static constexpr size_t DEFAULT_BUCKET_COUNT = 1u << 20; // 1M buckets = 4M entries = 64 MiB of entries; tests/perf tuned.
    static constexpr size_t MIN_BUCKET_COUNT = 1u << 14;     // 16K buckets ~= 1 MiB of entries (Hash min).
    static constexpr size_t MAX_BUCKET_COUNT = 1u << 26;     // 64M buckets ~= 4 GiB of entries (Hash max).
    static constexpr size_t DEFAULT_HASH_MB = 64;
    static constexpr size_t MIN_HASH_MB = 1;
    static constexpr size_t MAX_HASH_MB = 4096;
    static constexpr size_t HUGE_PAGE_MIN_BYTES = 32u * 1024u * 1024u; // 32 MiB, common huge page size on x86-64 Linux.

    static_assert(sizeof(Entry) == 16, "TT entry must be 16 bytes");
    static_assert((sizeof(Entry) * ENTRIES_PER_BUCKET) == 64, "Each bucket should be exactly one cache line");
    static_assert((DEFAULT_BUCKET_COUNT & (DEFAULT_BUCKET_COUNT - 1)) == 0, "DEFAULT_BUCKET_COUNT must be power of 2");

    // Map a requested Hash size in MiB to a power-of-two bucket count. The MiB
    // figure sizes the entry array (the 64-byte-per-bucket cache lines); the
    // per-bucket seqlock word is a small (~6%) overhead on top. Rounding down to
    // a power of two keeps the index mask (`& (bucketCount - 1)`) valid.
    [[nodiscard]] static constexpr size_t bucketCountForMB(size_t megabytes) noexcept {
        const size_t clampedMB = std::clamp(megabytes, MIN_HASH_MB, MAX_HASH_MB);
        const size_t targetBytes = clampedMB * 1024u * 1024u;
        const size_t rawBuckets = targetBytes / (ENTRIES_PER_BUCKET * sizeof(Entry));
        const size_t pow2 = std::bit_floor(rawBuckets == 0 ? size_t{1} : rawBuckets);
        return std::clamp(pow2, MIN_BUCKET_COUNT, MAX_BUCKET_COUNT);
    }

    explicit TT(HugePageMode mode = HugePageMode::Auto)
        : hugePageMode_(resolveHugePageMode(mode)) {
        if (!allocateInPlace(DEFAULT_BUCKET_COUNT)) {
            throw std::bad_alloc{};
        }
    }

    // Decoded snapshot of a probed entry. `hit == false` leaves the other
    // fields at their defaults (flag INVALID, move 0, no static eval).
    struct ProbeResult {
        int32_t  score      = 0;
        int32_t  staticEval = Entry::NO_EVAL;
        uint16_t move       = 0;
        uint8_t  depth      = 0;
        uint8_t  flag       = Entry::INVALID;
        bool     hit        = false;
    };

    inline void prefetch(uint64_t key) noexcept;
    inline bool probeMove(uint64_t key, uint16_t& outBestMove) const noexcept;
    // Cold-path convenience: the stored move for `key`, decoded ({} = none).
    [[nodiscard]] inline chess::Move probeDecodedMove(uint64_t key) const noexcept {
        uint16_t encoded = 0;
        return probeMove(key, encoded) ? Entry::decodeMove(encoded) : chess::Move{};
    }
    // The one hot-path read: a single bucket snapshot per node feeds the TT
    // cutoff, static-eval tightening, singular-extension gate and hash-move
    // ordering. Bound/depth gating is the caller's job.
    [[nodiscard]] inline ProbeResult probeEntry(uint64_t key) const noexcept;
    // bestMove == 0 means "no move to store" (a bound-only write): the existing
    // move in a matching entry is preserved rather than clobbered. staticEval ==
    // Entry::NO_EVAL is preserved the same way — the static eval is a property
    // of the position alone, so an older one stays valid regardless of the
    // depth or window this write came from.
    inline void store(uint64_t key, uint8_t depth, int32_t score, uint8_t flag,
                      uint16_t bestMove = 0, int32_t staticEval = Entry::NO_EVAL) noexcept;

    // Resize to approximately `megabytes` MiB (rounded down to a power-of-two
    // bucket count) and clear all entries. NOT safe to call during a live
    // search; the UCI layer stops the search before changing options. On
    // allocation failure the existing table is left intact and false is
    // returned.
    inline bool resize(size_t megabytes) noexcept;
    [[nodiscard]] size_t sizeMB() const noexcept {
        return (bucketCount_ * ENTRIES_PER_BUCKET * sizeof(Entry)) / (1024u * 1024u);
    }

    inline void incrementGeneration() noexcept { ++generation_; }
    inline void clear() noexcept;

    TT(const TT&) = delete;
    TT& operator=(const TT&) = delete;
    TT(TT&&) = default;
    TT& operator=(TT&&) = default;

private:
    struct BucketSeq {
        std::atomic<uint32_t> value{0U};
    };

    enum class AllocationKind : uint8_t {
        Heap = 0,
        MMap
    };

    // Total bytes for one contiguous block holding the entry array followed by
    // the per-bucket seqlock words. The entry array's size is a multiple of 64,
    // so the seqlock sub-array starts cache-line aligned.
    [[nodiscard]] static constexpr size_t allocBytesFor(size_t buckets) noexcept {
        return buckets * ENTRIES_PER_BUCKET * sizeof(Entry) + buckets * sizeof(BucketSeq);
    }

    // Owns one raw contiguous block: [Entry[]][BucketSeq[]]. Both sub-arrays are
    // trivially destructible, so the deleter only releases storage.
    struct BlockDeleter {
        AllocationKind kind = AllocationKind::Heap;
        size_t mappedBytes = 0;

        inline void operator()(void* ptr) const noexcept {
            if (ptr == nullptr) return;
#if defined(__linux__)
            if (kind == AllocationKind::MMap) {
                (void)::munmap(ptr, mappedBytes);
                return;
            }
#endif
            ::operator delete(ptr, std::align_val_t(64));
        }
    };

    using BlockPtr = std::unique_ptr<void, BlockDeleter>;

    BlockPtr table_{nullptr, BlockDeleter{}};
    uint8_t generation_ = 0;
    HugePageMode hugePageMode_;
    bool hugePagesBacked_ = false;
    Entry* entries_ = nullptr;
    BucketSeq* bucketSeq_ = nullptr;
    size_t bucketCount_ = 0;
    size_t bucketMask_ = 0;

    inline Entry* data() noexcept { return entries_; }
    inline const Entry* data() const noexcept { return entries_; }
    inline BucketSeq* seqData() noexcept { return bucketSeq_; }
    inline const BucketSeq* seqData() const noexcept { return bucketSeq_; }

    struct EntrySnapshot {
        uint64_t key = 0ULL;
        uint64_t payload = 0ULL;
    };

    [[nodiscard]] static inline std::atomic_ref<uint64_t> atomicWord(const uint64_t& word) noexcept {
        return std::atomic_ref<uint64_t>(const_cast<uint64_t&>(word));
    }

    [[nodiscard]] static inline uint32_t lockBucket(BucketSeq& bucketSeq) noexcept {
        uint32_t expected = bucketSeq.value.load(std::memory_order_relaxed);
        for (;;) {
            while ((expected & 1U) != 0U) {
                expected = bucketSeq.value.load(std::memory_order_acquire);
            }

            const uint32_t desired = expected + 1U; // odd => writer owns lock
            if (bucketSeq.value.compare_exchange_weak(
                    expected,
                    desired,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                return expected;
            }
        }
    }

    static inline void unlockBucket(BucketSeq& bucketSeq, uint32_t lockBase) noexcept {
        bucketSeq.value.store(lockBase + 2U, std::memory_order_release);
    }

    [[nodiscard]] static inline bool readBucketSnapshot(
        const Entry* bucket,
        const BucketSeq& bucketSeq,
        EntrySnapshot (&snapshot)[ENTRIES_PER_BUCKET]) noexcept {
        constexpr int MAX_RETRIES = 8;
        for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
            const uint32_t seqStart = bucketSeq.value.load(std::memory_order_acquire);
            if ((seqStart & 1U) != 0U) {
                continue;
            }

            for (size_t i = 0; i < ENTRIES_PER_BUCKET; ++i) {
                snapshot[i].key = atomicWord(bucket[i].key).load(std::memory_order_relaxed);
                snapshot[i].payload = atomicWord(bucket[i].payload).load(std::memory_order_relaxed);
            }

            const uint32_t seqEnd = bucketSeq.value.load(std::memory_order_acquire);
            if (seqStart == seqEnd && (seqEnd & 1U) == 0U) {
                return true;
            }
        }
        return false;
    }

    // Shared probe scaffolding: snapshot the bucket and return the first valid
    // entry whose key matches (the only entry any probe ever inspects, since
    // they all stop at the first key match). Returns false if the lock-free
    // snapshot read failed or no matching valid entry exists.
    [[nodiscard]] __attribute__((always_inline)) inline bool findEntrySnapshot(
        uint64_t key, EntrySnapshot& out) const noexcept {
        const size_t bucketIndex = static_cast<size_t>(key) & bucketMask_;
        const Entry* bucket = data() + (bucketIndex * ENTRIES_PER_BUCKET);

        const BucketSeq& bucketSeq = seqData()[bucketIndex];
        EntrySnapshot snapshot[ENTRIES_PER_BUCKET];
        if (!readBucketSnapshot(bucket, bucketSeq, snapshot)) return false;
        for (size_t i = 0; i < ENTRIES_PER_BUCKET; ++i) {
            const EntrySnapshot& entry = snapshot[i];
            if (entry.key != key) continue;
            if (Entry::flagFromPayload(entry.payload) == Entry::INVALID) continue;
            out = entry;
            return true;
        }
        return false;
    }

    [[nodiscard]] static inline HugePageMode hugePageModeFromEnv() noexcept {
        const char* envValue = std::getenv("CHESS_TT_HUGEPAGE");
        if (envValue == nullptr || *envValue == '\0') return HugePageMode::Auto;

        const std::string_view value(envValue);
        if (ascii::iequals(value, "on") || ascii::iequals(value, "1") || ascii::iequals(value, "true") || ascii::iequals(value, "force")) {
            return HugePageMode::On;
        }
        if (ascii::iequals(value, "off") || ascii::iequals(value, "0") || ascii::iequals(value, "false")) {
            return HugePageMode::Off;
        }
        return HugePageMode::Auto;
    }

    [[nodiscard]] static inline HugePageMode resolveHugePageMode(HugePageMode requestedMode) noexcept {
        if (requestedMode == HugePageMode::Auto) {
            return hugePageModeFromEnv();
        }
        return requestedMode;
    }

    [[nodiscard]] static constexpr size_t alignUp(size_t value, size_t alignment) noexcept {
        return (value + alignment - 1u) & ~(alignment - 1u);
    }

    [[nodiscard]] static inline BlockPtr allocateHeapBlock(size_t bytes) noexcept {
        void* raw = ::operator new(bytes, std::align_val_t(64), std::nothrow);
        return BlockPtr(raw, BlockDeleter{AllocationKind::Heap, 0});
    }

#if defined(__linux__)
    [[nodiscard]] static inline BlockPtr tryAllocateExplicitHugePages(size_t bytes, bool& outHugePagesBacked) noexcept {
        constexpr size_t HugePageBytes = 2u * 1024u * 1024u;
        const size_t mapBytes = alignUp(bytes, HugePageBytes);
        int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;
#if defined(MAP_HUGE_2MB)
        flags |= MAP_HUGE_2MB;
#endif
        void* mapped = ::mmap(nullptr, mapBytes, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (mapped == MAP_FAILED) {
            return BlockPtr(nullptr, BlockDeleter{AllocationKind::Heap, 0});
        }

        outHugePagesBacked = true;
        return BlockPtr(mapped, BlockDeleter{AllocationKind::MMap, mapBytes});
    }

    [[nodiscard]] static inline BlockPtr tryAllocateTransparentHugePages(size_t bytes, bool& outHugePagesBacked) noexcept {
        const size_t mapBytes = bytes;
        void* mapped = ::mmap(nullptr, mapBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped == MAP_FAILED) {
            return BlockPtr(nullptr, BlockDeleter{AllocationKind::Heap, 0});
        }

#if defined(MADV_HUGEPAGE)
        if (::madvise(mapped, mapBytes, MADV_HUGEPAGE) == 0) {
            outHugePagesBacked = true;
            return BlockPtr(mapped, BlockDeleter{AllocationKind::MMap, mapBytes});
        }
#endif

        (void)::munmap(mapped, mapBytes);
        return BlockPtr(nullptr, BlockDeleter{AllocationKind::Heap, 0});
    }
#endif

    [[nodiscard]] static inline BlockPtr allocateBlock(HugePageMode mode, size_t buckets, bool& outHugePagesBacked) noexcept {
        outHugePagesBacked = false;
        const size_t bytes = allocBytesFor(buckets);

        const bool hugePageAllowed = (mode != HugePageMode::Off) && (bytes >= HUGE_PAGE_MIN_BYTES);
        if (hugePageAllowed) {
#if defined(__linux__)
            BlockPtr explicitHuge = tryAllocateExplicitHugePages(bytes, outHugePagesBacked);
            if (explicitHuge) return explicitHuge;

            BlockPtr transparentHuge = tryAllocateTransparentHugePages(bytes, outHugePagesBacked);
            if (transparentHuge) return transparentHuge;
#endif
        }

        return allocateHeapBlock(bytes);
    }

    // Begin the lifetimes of the Entry[] and BucketSeq[] sub-arrays inside the
    // raw block and hand back typed pointers. Value-initialization zeroes every
    // field, leaving all entries INVALID and all seqlocks unlocked.
    static inline void constructArrays(void* base, size_t buckets, Entry*& outEntries, BucketSeq*& outSeq) noexcept {
        const size_t entryCount = buckets * ENTRIES_PER_BUCKET;
        Entry* entries = reinterpret_cast<Entry*>(base);
        for (size_t i = 0; i < entryCount; ++i) ::new (static_cast<void*>(&entries[i])) Entry{};

        void* seqRaw = reinterpret_cast<std::byte*>(base) + entryCount * sizeof(Entry);
        BucketSeq* seq = reinterpret_cast<BucketSeq*>(seqRaw);
        for (size_t i = 0; i < buckets; ++i) ::new (static_cast<void*>(&seq[i])) BucketSeq{};

        outEntries = entries;
        outSeq = seq;
    }

    // Allocate a block for `buckets` buckets, construct its sub-arrays, and swap
    // it in as the live table (freeing any previous block). Returns false and
    // leaves the current table untouched on allocation failure.
    [[nodiscard]] inline bool allocateInPlace(size_t buckets) noexcept {
        bool huge = false;
        BlockPtr block = allocateBlock(hugePageMode_, buckets, huge);
        if (!block) return false;

        Entry* newEntries = nullptr;
        BucketSeq* newSeq = nullptr;
        constructArrays(block.get(), buckets, newEntries, newSeq);

        table_ = std::move(block); // releases the previous block via its deleter
        entries_ = newEntries;
        bucketSeq_ = newSeq;
        bucketCount_ = buckets;
        bucketMask_ = buckets - 1;
        hugePagesBacked_ = huge;
        return true;
    }

    [[nodiscard]] static constexpr uint8_t clampDepth(uint8_t depth) noexcept {
        return (depth <= Entry::MAX_DEPTH) ? depth : Entry::MAX_DEPTH;
    }

};

inline void TT::prefetch(uint64_t key) noexcept {
    const size_t bucketIndex = static_cast<size_t>(key) & bucketMask_;
    __builtin_prefetch(data() + (bucketIndex * ENTRIES_PER_BUCKET), 0, 3);
    __builtin_prefetch(seqData() + bucketIndex, 0, 3);
}

static_assert(TT::Entry::encodeMove(chess::Move{12, 28, 'q'}) == TT::Entry::encodeMove(chess::Move{12, 28, 'Q'}), "promotion encoding should be case-insensitive");
static_assert(TT::Entry::decodeMove(TT::Entry::encodeMove(chess::Move{12, 28, 'n'})).from == 12, "move decode from mismatch");
static_assert(TT::Entry::decodeMove(TT::Entry::encodeMove(chess::Move{12, 28, 'n'})).to == 28, "move decode to mismatch");
static_assert(TT::Entry::decodeMove(TT::Entry::encodeMove(
    chess::Move{12, 28, chess::Board::KNIGHT})).promotionType == chess::Board::KNIGHT,
    "move decode promotion mismatch");

inline bool TT::probeMove(uint64_t key, uint16_t& outBestMove) const noexcept {
    EntrySnapshot entry;
    if (!findEntrySnapshot(key, entry)) {
        outBestMove = 0;
        return false;
    }
    outBestMove = Entry::bestMoveFromPayload(entry.payload);
    return outBestMove != 0;
}

inline TT::ProbeResult TT::probeEntry(uint64_t key) const noexcept {
    EntrySnapshot entry;
    ProbeResult result;
    if (!findEntrySnapshot(key, entry)) return result;
    result.score      = Entry::scoreFromPayload(entry.payload);
    result.staticEval = Entry::staticEvalFromPayload(entry.payload);
    result.move       = Entry::bestMoveFromPayload(entry.payload);
    result.depth      = Entry::depthFromPayload(entry.payload);
    result.flag       = Entry::flagFromPayload(entry.payload);
    result.hit        = true;
    return result;
}

inline void TT::store(
    uint64_t key,
    uint8_t depth,
    int32_t score,
    uint8_t flag,
    uint16_t bestMove,
    int32_t staticEval) noexcept {
    const uint8_t storedDepth = clampDepth(depth);
    const size_t bucketIndex = static_cast<size_t>(key) & bucketMask_;
    Entry* bucket = data() + (bucketIndex * ENTRIES_PER_BUCKET);

    BucketSeq& bucketSeq = seqData()[bucketIndex];
    const uint32_t lockBase = lockBucket(bucketSeq);

    Entry* replaceEntry = &bucket[0];
    Entry* emptyEntry = nullptr;
    int bestReplaceScore = INT32_MIN;

    for (size_t i = 0; i < ENTRIES_PER_BUCKET; ++i) {
        Entry& entry = bucket[i];
        const uint64_t entryKey = atomicWord(entry.key).load(std::memory_order_relaxed);
        const uint64_t entryPayload = atomicWord(entry.payload).load(std::memory_order_relaxed);
        const uint8_t entryFlag = Entry::flagFromPayload(entryPayload);

        if (entryFlag == Entry::INVALID) {
            if (emptyEntry == nullptr) emptyEntry = &entry;
            continue;
        }

        if (entryKey == key) {
            if (storedDepth >= Entry::depthFromPayload(entryPayload) || flag == Entry::EXACT) {
                // Preserve the stored move on a bound-only write (bestMove == 0)
                // and the stored eval when this write carries none.
                const uint16_t moveToStore = (bestMove != 0) ? bestMove : Entry::bestMoveFromPayload(entryPayload);
                const int32_t evalToStore = (staticEval != Entry::NO_EVAL)
                    ? staticEval
                    : Entry::staticEvalFromPayload(entryPayload);
                const uint64_t newPayload = Entry::encodePayload(score, evalToStore, moveToStore, storedDepth, generation_, flag);
                atomicWord(entry.payload).store(newPayload, std::memory_order_relaxed);
            }
            unlockBucket(bucketSeq, lockBase);
            return;
        }

        const int ageDiff = static_cast<int>(static_cast<uint8_t>(generation_ - Entry::ageFromPayload(entryPayload)));
        const int replaceScore = (ageDiff << 8) - (static_cast<int>(Entry::depthFromPayload(entryPayload)) << 2);
        if (replaceScore > bestReplaceScore) {
            bestReplaceScore = replaceScore;
            replaceEntry = &entry;
        }
    }

    Entry* const target = (emptyEntry != nullptr) ? emptyEntry : replaceEntry;
    const uint64_t newPayload = Entry::encodePayload(score, staticEval, bestMove, storedDepth, generation_, flag);
    atomicWord(target->payload).store(newPayload, std::memory_order_relaxed);
    atomicWord(target->key).store(key, std::memory_order_relaxed);
    unlockBucket(bucketSeq, lockBase);
}

inline void TT::clear() noexcept {
    std::fill_n(data(), bucketCount_ * ENTRIES_PER_BUCKET, Entry{});
    BucketSeq* bucketSeq = seqData();
    for (size_t i = 0; i < bucketCount_; ++i) {
        bucketSeq[i].value.store(0U, std::memory_order_relaxed);
    }
}

inline bool TT::resize(size_t megabytes) noexcept {
    const size_t newBuckets = bucketCountForMB(megabytes);

    // Same bucket count: nothing to reallocate, just drop the contents.
    if (newBuckets == bucketCount_ && entries_ != nullptr) {
        clear();
        return true;
    }

    // Allocate the replacement before releasing the current table, so a failed
    // allocation leaves the existing table fully intact (allocateInPlace only
    // swaps members after a successful allocation). Peak memory briefly holds
    // both tables; resize is a rare, search-idle operation.
    return allocateInPlace(newBuckets);
}

inline constexpr TT::Entry::Flag
determineFlag(int32_t score, int32_t alphaOrig, int32_t beta) noexcept {
    if (score <= alphaOrig) return TT::Entry::UPPERBOUND;
    if (score >= beta) return TT::Entry::LOWERBOUND;
    return TT::Entry::EXACT;
}

static_assert(determineFlag(100, 50, 200) == TT::Entry::EXACT, "determineFlag logic error");
static_assert(determineFlag(40, 50, 200) == TT::Entry::UPPERBOUND, "determineFlag logic error");
static_assert(determineFlag(250, 50, 200) == TT::Entry::LOWERBOUND, "determineFlag logic error");
