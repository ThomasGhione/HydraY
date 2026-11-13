#pragma once

#include <cstddef>
#include <memory>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#endif

// A heap array on 2 MiB boundaries, which lets Linux back it with transparent
// huge pages: the TT, the eval cache and the threat weights are read at random
// places, and with 4 KiB pages most reads also missed the TLB. Null when out
// of memory.
inline constexpr std::align_val_t HUGE_PAGE_ALIGNMENT{2 * 1024 * 1024};
struct HugePageFree {
    void operator()(void* p) const noexcept { ::operator delete(p, HUGE_PAGE_ALIGNMENT); }
};
template<typename T>
using HugePageArray = std::unique_ptr<T[], HugePageFree>;

template<typename T>
[[nodiscard]] HugePageArray<T> allocateHugePages(size_t count) noexcept {
    T* p = static_cast<T*>(::operator new(count * sizeof(T), HUGE_PAGE_ALIGNMENT, std::nothrow));
#if defined(MADV_HUGEPAGE)
    if (p != nullptr) ::madvise(p, count * sizeof(T), MADV_HUGEPAGE);
#endif
    return HugePageArray<T>(p);
}
