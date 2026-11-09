// Huge-page allocations (huge_pages.h) on portable machines.
//
// The arrays this library owns (make_for_overwrite) and MKL's allocator
// (huge_malloc / huge_calloc / huge_realloc / huge_free): malloc semantics,
// pointers of the C library, the 2 MiB threshold, and threads. On Linux,
// blocks of at least 2 MiB must be fresh mappings: 2 MiB aligned, zero
// filled and marked for huge pages ("hg" in /proc/self/smaps); elsewhere
// they come from malloc.
#include "huge_pages.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
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

} // namespace

int main() {
    check_arrays();
    check_allocator();
    check_threads();
    std::printf("%s: %d huge-page cases (%s), %d failed\n",
                failures ? "FAIL" : "PASS", cases,
                kLinux ? "Linux: mappings checked" : "not Linux: malloc only",
                failures);
    return failures ? 1 : 0;
}
