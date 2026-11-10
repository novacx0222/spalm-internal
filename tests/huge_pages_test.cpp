// Huge-page allocations (huge_pages.h) on portable machines.
//
// The arrays this library owns (make_for_overwrite) and MKL's allocator
// (huge_malloc / huge_calloc / huge_realloc / huge_free): malloc semantics,
// pointers of the C library, the 2 MiB threshold, and threads. On Linux,
// blocks of at least 2 MiB must be fresh mappings: 2 MiB aligned, zero
// filled and marked for huge pages ("hg" in /proc/self/smaps); elsewhere
// they come from malloc. Also the packing buffers of the dense-path kernels
// (packing_buffers.h): alignment, one huge mapping shared by the per-worker
// S blocks, disjoint blocks, and the fallbacks.
#include "huge_pages.h"
#include "packing_buffers.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace {

int failures = 0, cases = 0;

void check(bool ok, const char *what) {
    ++cases;
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

constexpr size_t MiB = size_t(1) << 20;

#if defined(__linux__)
constexpr bool kLinux = true;
#else
constexpr bool kLinux = false;
#endif

bool aligned(const void *p) {
    return reinterpret_cast<uintptr_t>(p) % spalm_huge::kHugeBytes == 0;
}

// Whether the mapping holding p is marked MADV_HUGEPAGE ("hg" among its
// VmFlags); false if not found or off Linux.
bool advised(const void *p) {
#if defined(__linux__)
    std::ifstream smaps("/proc/self/smaps");
    const auto at = reinterpret_cast<unsigned long>(p);
    std::string line;
    bool inside = false;
    while (std::getline(smaps, line)) {
        unsigned long lo, hi;
        char sep;
        if (std::sscanf(line.c_str(), "%lx-%lx%c", &lo, &hi, &sep) == 3 &&
            sep == ' ') {
            inside = lo <= at && at < hi;
        } else if (inside && line.rfind("VmFlags:", 0) == 0) {
            return (line + " ").find(" hg ") != std::string::npos;
        }
    }
#else
    (void) p;
#endif
    return false;
}

void fill(void *p, size_t bytes, unsigned seed) {
    auto *b = static_cast<unsigned char *>(p);
    for (size_t i = 0; i < bytes; ++i) b[i] = static_cast<unsigned char>(i * 31 + seed);
}

bool same(const void *p, size_t bytes, unsigned seed) {
    const auto *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < bytes; ++i)
        if (b[i] != static_cast<unsigned char>(i * 31 + seed)) return false;
    return true;
}

bool zero(const void *p, size_t bytes) {
    const auto *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < bytes; ++i)
        if (b[i]) return false;
    return true;
}

size_t live() { return spalm_huge::detail::blocks().live.load(); }

void check_arrays() {
    {
        auto a = spalm_huge::make_for_overwrite<double>(100);
        fill(a.get(), 800, 1);
        check(same(a.get(), 800, 1) && !a.get_deleter().m.map, "small array: malloc");
    }
    {
        // 8 MiB: a mapping on Linux, usable everywhere.
        const size_t n = 1 << 20;
        auto a = spalm_huge::make_for_overwrite<double>(n);
        const bool mapped = a.get_deleter().m.map != nullptr;
        check(mapped == kLinux, "large array: mapping on Linux only");
        if (kLinux) {
            check(aligned(a.get()), "large array: 2 MiB aligned");
            check(zero(a.get(), n * sizeof(double)), "large array: zero filled");
            check(advised(a.get()), "large array: MADV_HUGEPAGE");
        }
        fill(a.get(), n * sizeof(double), 2);
        check(same(a.get(), n * sizeof(double), 2), "large array: usable");
    }
    {
        // The threshold is 2 MiB inclusive.
        auto below = spalm_huge::make_for_overwrite<char>(spalm_huge::kHugeBytes - 1);
        auto at = spalm_huge::make_for_overwrite<char>(spalm_huge::kHugeBytes);
        check(!below.get_deleter().m.map, "array below 2 MiB: malloc");
        check((at.get_deleter().m.map != nullptr) == kLinux, "array of 2 MiB: mapping");
    }
    {
        auto empty = spalm_huge::make_for_overwrite<int>(0);
        check(empty.get() != nullptr, "empty array: valid pointer");
    }
    bool threw = false;
    try {
        spalm_huge::make_for_overwrite<double>(SIZE_MAX / 4);
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    check(threw, "oversized array: bad_alloc");
}

void check_allocator() {
    using namespace spalm_huge;
    const auto blocks_before = allocated_blocks();
    // Small: the C library.
    void *s = huge_malloc(1000);
    fill(s, 1000, 3);
    check(same(s, 1000, 3) && live() == 0, "small malloc: C library");
    huge_free(s);
    huge_free(nullptr);

    // Large: recorded mapping; realloc in place within its 2 MiB pages.
    void *p = huge_malloc(5 * MiB);
    check(p != nullptr, "large malloc");
    if (kLinux) {
        check(aligned(p) && advised(p) && live() == 1, "large malloc: huge mapping");
        check(allocated_blocks() == blocks_before + 1, "large malloc: counted");
    }
    fill(p, 5 * MiB, 4);
    void *q = huge_realloc(p, 3 * MiB);
    check(same(q, 3 * MiB, 4), "large realloc smaller: content kept");
    if (kLinux) check(q == p, "large realloc smaller: in place");
    void *r = huge_realloc(q, 6 * MiB);
    check(same(r, 3 * MiB, 4), "large realloc within pages: content kept");
    if (kLinux) check(r == q, "large realloc within pages: in place");
    fill(r, 6 * MiB, 5);
    void *t = huge_realloc(r, 20 * MiB);
    check(same(t, 6 * MiB, 5), "large realloc beyond pages: content moved");
    if (kLinux) check(aligned(t) && live() == 1, "large realloc beyond pages: one mapping");
    check(huge_realloc(t, 0) == nullptr && live() == 0, "large realloc to 0: freed");

    // A C library block growing past 2 MiB moves onto huge pages.
    void *c = std::malloc(1000);
    fill(c, 1000, 6);
    void *g = huge_realloc(c, 4 * MiB);
    check(same(g, 1000, 6), "C block realloc to large: content kept");
    if (kLinux) check(aligned(g) && live() == 1, "C block realloc to large: mapping");
    huge_free(g);
    // ...and shrinking a C library block stays there.
    void *h = huge_malloc(4000);
    fill(h, 4000, 7);
    void *h2 = huge_realloc(h, 100);
    check(same(h2, 100, 7) && live() == 0, "C block realloc smaller");
    huge_free(h2);
    void *n = huge_realloc(nullptr, 3 * MiB);
    check(n != nullptr && (live() == 1) == kLinux, "realloc of null: malloc");
    huge_free(n);

    // calloc: zero filled at both sizes; overflow refused.
    auto *z1 = static_cast<double *>(huge_calloc(1000, sizeof(double)));
    auto *z2 = static_cast<double *>(huge_calloc(1 << 20, sizeof(double)));
    check(zero(z1, 8000) && zero(z2, 8 * MiB), "calloc: zero filled");
    if (kLinux) check(aligned(z2) && advised(z2), "large calloc: huge mapping");
    huge_free(z1);
    huge_free(z2);
    check(huge_calloc(SIZE_MAX / 2, 4) == nullptr, "calloc overflow: null");
    check(live() == 0, "allocator: nothing left");
}

void check_threads() {
    using namespace spalm_huge;
    int bad = 0;
#pragma omp parallel for reduction(+ : bad) num_threads(8)
    for (int i = 0; i < 64; ++i) {
        const size_t bytes = i % 2 ? 3 * MiB + 17 * i : 1024 + i;
        void *p = huge_malloc(bytes);
        fill(p, bytes, i);
        void *q = huge_realloc(p, bytes * 2);
        bad += !same(q, bytes, i);
        huge_free(q);
    }
    check(bad == 0 && live() == 0, "threads: malloc / realloc / free");
}

bool aligned_to(const void *p, size_t alignment) {
    return reinterpret_cast<uintptr_t>(p) % alignment == 0;
}

// The threshold is a size; a bool must not convert to one.
template <typename B>
concept takes_threshold = requires(B b) { spalm_packing::try_make_buffer<double>(size_t(1), b); };
static_assert(takes_threshold<size_t>);
static_assert(!takes_threshold<bool>);
static_assert(spalm_packing::kRBlockHugeBytes >= spalm_huge::kHugeBytes);

// Packing buffers of the dense-path kernels (packing_buffers.h).
void check_packing() {
    using namespace spalm_packing;
    {
        auto a = try_make_buffer<double>(100);
        check(a && aligned_to(a.get(), kAlignment) && !spalm_huge::is_mapped(a),
              "small packing buffer: posix_memalign, 64-byte aligned");
        fill(a.get(), 800, 11);
        check(same(a.get(), 800, 11), "small packing buffer: usable");
    }
    {
        const size_t n = spalm_huge::kHugeBytes / sizeof(double) + 16;
        auto a = try_make_buffer<double>(n);
        check(a && aligned_to(a.get(), kAlignment), "large packing buffer: aligned");
        check(spalm_huge::is_mapped(a) == kLinux, "large packing buffer: mapping on Linux only");
        if (kLinux) check(aligned(a.get()) && advised(a.get()), "large packing buffer: huge pages");
        fill(a.get(), n * sizeof(double), 12);
        check(same(a.get(), n * sizeof(double), 12), "large packing buffer: usable");
        auto b = try_make_buffer<double>(n, kNoHuge);
        check(b && !spalm_huge::is_mapped(b) && aligned_to(b.get(), kAlignment),
              "large packing buffer without huge pages: posix_memalign");
    }
    {
        // R's block: huge pages only from kRBlockHugeBytes on.
        const size_t below = kRBlockHugeBytes / sizeof(double) - 16;
        const size_t at = kRBlockHugeBytes / sizeof(double);
        auto a = make_buffer<double>(below, kRBlockHugeBytes);
        auto b = make_buffer<double>(at, kRBlockHugeBytes);
        check(!spalm_huge::is_mapped(a) && aligned_to(a.get(), kAlignment),
              "R block below kRBlockHugeBytes: posix_memalign");
        check(spalm_huge::is_mapped(b) == kLinux && aligned_to(b.get(), kAlignment),
              "R block from kRBlockHugeBytes: mapping on Linux only");
        if (kLinux) check(aligned(b.get()) && advised(b.get()), "R block from kRBlockHugeBytes: huge pages");
        fill(a.get(), below * sizeof(double), 13);
        fill(b.get(), at * sizeof(double), 14);
        check(same(a.get(), below * sizeof(double), 13) && same(b.get(), at * sizeof(double), 14),
              "R blocks: usable");
        // A threshold below 2 MiB still maps nothing smaller than 2 MiB.
        auto c = try_make_buffer<double>(1000, size_t(0));
        check(c && !spalm_huge::is_mapped(c), "threshold 0: small buffer still posix_memalign");
    }
    check(!try_make_buffer<double>(0), "empty packing buffer: none");
    check(!try_make_buffer<double>(SIZE_MAX / 4), "oversized packing buffer: none");
    bool threw = false;
    try {
        make_buffer<double>(SIZE_MAX / 4);
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    check(threw, "oversized packing buffer: bad_alloc");

    // Per-worker blocks: the sizes of slice length 2048 and 512 in double
    // (8 slices x length x 16 values, plus the 128-byte pad), and a small one.
    const struct {
        int workers;
        size_t count;
        bool shared;
        const char *name;
    } sizes[] = {
        {30, 8 * 2048 * 16 + 16, true, "2 MiB blocks"},
        {30, 8 * 512 * 16 + 16, true, "512 KiB blocks"},
        {4, 1000, false, "small blocks"},
        {1, 8 * 2048 * 16 + 16, true, "one 2 MiB block"},
    };
    for (const auto &s : sizes) {
        worker_blocks<double> blocks(s.workers, s.count);
        const std::string name = s.name;
        check(blocks.shared() == (s.shared && kLinux), (name + ": shared mapping on Linux only").c_str());
        std::vector<double *> at(s.workers, nullptr);
        int bad = 0;
#pragma omp parallel for reduction(+ : bad) num_threads(8)
        for (int w = 0; w < s.workers; ++w) {
            at[w] = blocks.get(w);
            bad += at[w] == nullptr || !aligned_to(at[w], kAlignment) ||
                   (blocks.shared() && !aligned_to(at[w], kBlockAlignment));
            if (at[w]) fill(at[w], s.count * sizeof(double), 20 + w);
        }
        check(bad == 0, (name + ": one aligned block per worker").c_str());
        for (int w = 0; w < s.workers; ++w) {
            bad += at[w] && !same(at[w], s.count * sizeof(double), 20 + w);
            bad += blocks.get(w) != at[w];
        }
        check(bad == 0, (name + ": blocks disjoint and stable").c_str());
        check(blocks.get(-1) == nullptr && blocks.get(s.workers) == nullptr,
              (name + ": other workers: none").c_str());
    }
    {
        worker_blocks<double> none(0, 1000);
        worker_blocks<double> empty(4, 0);
        worker_blocks<double> huge(2, SIZE_MAX / 4);
        check(none.get(0) == nullptr && empty.get(0) == nullptr && huge.get(0) == nullptr,
              "no workers, empty or oversized blocks: none");
    }
    {
        worker_blocks<float> blocks(30, 8 * 2048 * 32 + 32);
        float *a = blocks.get(29);
        check(a != nullptr && blocks.shared() == kLinux, "float blocks: shared mapping on Linux only");
        if (a) fill(a, (8 * 2048 * 32 + 32) * sizeof(float), 9);
        check(a && same(a, (8 * 2048 * 32 + 32) * sizeof(float), 9), "float blocks: last worker usable");
    }
    check(live() == 0, "packing buffers: no allocator blocks");
}

} // namespace

int main() {
    check_arrays();
    check_allocator();
    check_threads();
    check_packing();
    std::printf("%s: %d huge-page cases (%s), %d failed\n",
                failures ? "FAIL" : "PASS", cases,
                kLinux ? "Linux: mappings checked" : "not Linux: malloc only",
                failures);
    return failures ? 1 : 0;
}
