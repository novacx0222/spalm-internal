#pragma once

// Memory on 2 MiB transparent huge pages for large buffers that are written
// once and freed soon after: the largest pages of SS's sparse result, MKL's
// own blocks of 2 MiB and more (sp2m's result above all) and the dense
// inputs of MKL dense-dense.
//
// With THP set to "madvise", memory without MADV_HUGEPAGE stays on 4 KiB
// pages: each 4 KiB is a page fault when first written and a page to unmap
// when freed. Freeing a sparse result then took ~50 ms per GB against ~3 ms
// on huge pages (8488C, v23 SuiteSparse run), and writing it paid the faults.
// Like the operator's dense result, a block of at least kHugeBytes is
// therefore a fresh mapping, 2 MiB aligned and marked MADV_HUGEPAGE (the
// kernel hands it out zero-filled). Smaller blocks come from malloc, which
// reuses memory the process already freed. Off Linux, or if mmap fails,
// everything comes from malloc.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <unordered_map>

#if defined(__linux__)
#include <malloc.h>
#include <sys/mman.h>
#endif

namespace spalm_huge {

constexpr size_t kHugeBytes = size_t(2) << 20;

// A mapping as munmap needs it (map == nullptr: none).
struct mapping {
    void *map = nullptr;
    size_t length = 0;
};

// A fresh mapping holding `bytes` from a 2 MiB boundary, marked
// MADV_HUGEPAGE (if madvise fails it is ordinary memory, still usable).
// Returns its start, or nullptr (off Linux, or mmap failed).
inline void *map_huge(size_t bytes, mapping &m) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (bytes == 0 || bytes > SIZE_MAX - 2 * kHugeBytes) return nullptr;
    const size_t span = (bytes + kHugeBytes - 1) / kHugeBytes * kHugeBytes;
    const size_t length = span + kHugeBytes; // room to align the start
    void *map = mmap(
        nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
        -1, 0
    );
    if (map == MAP_FAILED) return nullptr;
    const uintptr_t at = reinterpret_cast<uintptr_t>(map);
    char *start =
        reinterpret_cast<char *>((at + kHugeBytes - 1) / kHugeBytes * kHugeBytes);
    madvise(start, span, MADV_HUGEPAGE);
    m = {map, length};
    return start;
#else
    (void) bytes;
    (void) m;
    return nullptr;
#endif
}

inline void unmap(const mapping &m) {
#if defined(__linux__)
    if (m.map) munmap(m.map, m.length);
#else
    (void) m;
#endif
}

// Arrays this library owns (SS's pages, MKL dense-dense's inputs).
struct deleter {
    mapping m;
    void operator()(void *p) const {
        if (m.map) {
            unmap(m);
        } else {
            std::free(p);
        }
    }
};
template <typename T> using array = std::unique_ptr<T[], deleter>;

// Whether a is a huge-page mapping (not from malloc).
template <typename T> bool is_mapped(const array<T> &a) {
    return a.get_deleter().m.map != nullptr;
}

// Uninitialized storage for `count` values, as make_unique_for_overwrite:
// huge pages from kHugeBytes on, malloc otherwise or if not `huge` (for
// callers whose blocks malloc can serve from freed memory). Throws
// std::bad_alloc.
template <typename T>
array<T> make_for_overwrite(size_t count, bool huge = true) {
    static_assert(
        std::is_trivially_default_constructible_v<T> &&
            std::is_trivially_destructible_v<T>,
        "plain values only"
    );
    if (count > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
    const size_t bytes = count * sizeof(T);
    if (huge && bytes >= kHugeBytes) {
        mapping m;
        if (void *p = map_huge(bytes, m)) {
            return array<T>(static_cast<T *>(p), deleter{m});
        }
    }
    void *p = std::malloc(bytes ? bytes : 1);
    if (!p) throw std::bad_alloc();
    return array<T>(static_cast<T *>(p), deleter{});
}

// malloc / calloc / realloc / free for MKL (its i_malloc.h pointers). A
// block of at least kHugeBytes is a huge mapping, recorded by its start so
// that free and realloc recognize it; every other pointer, including those
// MKL got before these were installed, belongs to the C library.
namespace detail {
struct block {
    mapping m;
    size_t bytes;    // as last requested
    size_t capacity; // the MADV_HUGEPAGE span from the start: whole 2 MiB pages
};
struct registry {
    std::mutex lock;
    std::unordered_map<void *, block> blocks;
    std::atomic<size_t> live{0};
    std::atomic<unsigned long long> allocated_blocks{0}, allocated_bytes{0};
};
// Never destroyed: MKL may free memory after static destructors ran.
inline registry &blocks() {
    static registry *r = new registry;
    return *r;
}
inline void *map_tracked(size_t bytes) {
    mapping m;
    void *p = map_huge(bytes, m);
    if (!p) return nullptr;
    registry &r = blocks();
    try {
        std::lock_guard<std::mutex> guard(r.lock);
        const size_t capacity =
            (bytes + kHugeBytes - 1) / kHugeBytes * kHugeBytes;
        r.blocks.emplace(p, block{m, bytes, capacity});
    } catch (...) {
        unmap(m);
        return nullptr;
    }
    r.live.fetch_add(1, std::memory_order_relaxed);
    r.allocated_blocks.fetch_add(1, std::memory_order_relaxed);
    r.allocated_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return p;
}
// Removes p from the registry if it is a huge block; returns whether it was.
inline bool take(void *p, block &out) {
    registry &r = blocks();
    if (r.live.load(std::memory_order_relaxed) == 0) return false;
    std::lock_guard<std::mutex> guard(r.lock);
    auto it = r.blocks.find(p);
    if (it == r.blocks.end()) return false;
    out = it->second;
    r.blocks.erase(it);
    r.live.fetch_sub(1, std::memory_order_relaxed);
    return true;
}
inline bool find(void *p, block &out) {
    registry &r = blocks();
    if (r.live.load(std::memory_order_relaxed) == 0) return false;
    std::lock_guard<std::mutex> guard(r.lock);
    auto it = r.blocks.find(p);
    if (it == r.blocks.end()) return false;
    out = it->second;
    return true;
}
inline void resize(void *p, size_t bytes) {
    registry &r = blocks();
    std::lock_guard<std::mutex> guard(r.lock);
    auto it = r.blocks.find(p);
    if (it != r.blocks.end()) it->second.bytes = bytes;
}
} // namespace detail

inline void *huge_malloc(size_t bytes) {
    if (bytes >= kHugeBytes) {
        if (void *p = detail::map_tracked(bytes)) return p;
    }
    return std::malloc(bytes);
}

inline void *huge_calloc(size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) return nullptr;
    const size_t bytes = count * size;
    if (bytes >= kHugeBytes) {
        // Fresh mappings are zero-filled.
        if (void *p = detail::map_tracked(bytes)) return p;
    }
    return std::calloc(count, size);
}

inline void huge_free(void *p) {
    if (!p) return;
    detail::block b;
    if (detail::take(p, b)) {
        unmap(b.m);
    } else {
        std::free(p);
    }
}

inline void *huge_realloc(void *p, size_t bytes) {
    if (!p) return huge_malloc(bytes);
    detail::block b;
    if (detail::find(p, b)) {
        if (bytes == 0) {
            huge_free(p);
            return nullptr;
        }
        if (bytes <= b.capacity) {
            detail::resize(p, bytes);
            return p;
        }
        void *q = huge_malloc(bytes);
        if (!q) return nullptr; // p stays valid, as with realloc
        std::memcpy(q, p, b.bytes);
        huge_free(p);
        return q;
    }
#if defined(__linux__)
    // A C library block growing to huge size moves onto huge pages.
    if (bytes >= kHugeBytes) {
        const size_t old = malloc_usable_size(p);
        if (old < bytes) {
            if (void *q = detail::map_tracked(bytes)) {
                std::memcpy(q, p, old);
                std::free(p);
                return q;
            }
        }
    }
#endif
    return std::realloc(p, bytes);
}

// Huge blocks handed out by huge_malloc / huge_calloc / huge_realloc since
// the process started, and their bytes (for the log).
inline unsigned long long allocated_blocks() {
    return detail::blocks().allocated_blocks.load(std::memory_order_relaxed);
}
inline unsigned long long allocated_bytes() {
    return detail::blocks().allocated_bytes.load(std::memory_order_relaxed);
}

} // namespace spalm_huge
