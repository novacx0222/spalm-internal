#pragma once

// Packing buffers of the dense-path kernels (SD, DS and DD in
// dense_dense_matmul.cpp): R's panel block, one per call, and one S block
// per worker, all proportional to the slice length.
//
// They were posix_memalign blocks on 4 KiB pages. A block above glibc's
// mmap threshold is a mapping of its own: every 4 KiB is a page fault when
// the kernel first zeroes it, and a page to unmap when the block is freed on
// the calling thread after the multiplication. At slice length 2048 that is
// 2 MiB of S per worker (60 MiB at 30 threads) and up to 38 MB of R. On the
// 8488C the frees alone took 2-7 ms per call, and 0.1-1 ms at 512 (v25
// SuiteSparse run: "[mm-total]" minus "[dense-dense-total-dense-mm-time]"),
// where small SD and DD calls take 1-4 ms in all.
//
// Here the S blocks of all workers share one huge-page mapping (huge_pages.h)
// when they reach 2 MiB together: one fault per 2 MiB and a cheap unmap. It
// is reserved up front, address space only; a worker's pages are backed when
// it first packs. R's block is a huge-page mapping from kRBlockHugeBytes on.
// Smaller blocks, and every block off Linux or when mmap fails, come from
// posix_memalign as before.
//
// Measured on the 8488C (30 threads, SD/DS/DD). Against v25 with R from
// 2 MiB on (v26):
//   - 10 SuiteSparse matrices, 5 interleaved runs: the free after the
//     multiplication dropped to 0.1-0.7 ms, and at 2048 the library call took
//     0.84x in gmean (0.98x in sum).
//   - All 170 SuiteSparse matrices at 512: SD 0.84x, DD 0.99x, DS 1.06x.
//     Wherever R was mapped, the kernel took 0.47-0.59 ms longer whatever
//     its size. R is zeroed and packed by all threads in one loop that ends
//     in a barrier, and one 2 MiB page is zeroed in a single fault, by one
//     thread. The free saved only about 60 us per MiB of R (0.2 ms at
//     2-4 MiB, 0.6 ms at 8-13 MiB), so R pays from about 8 MiB on: at 512,
//     at least 128 R slices in one slice column for DS, 171 for DD (12
//     values per row); a block holds up to 192.
//     The S blocks showed no such cost: workers touch them first in their
//     own tasks, not behind a barrier.
//   - R from kRBlockHugeBytes on, against v26 (12 SuiteSparse matrices, 5
//     interleaved runs, 512): DS calls with R below 8 MiB took 0.87x (kernel
//     -0.48 ms, free +0.31 ms). DD calls there stayed within noise (1.02x;
//     small DD calls vary by up to 1.3x between runs of the same library).
//   - SD at 2048 on femlab_ns3da and snap_cit_hepph took 1.03x. Each
//     worker's S block is then the size of the 2 MiB L2, and contiguous
//     physical pages may map it less favourably (not measured further).

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#include "huge_pages.h"

namespace spalm_packing {

// Cache-line alignment, as before: packed rows of 64 or 128 bytes
// (ds_aligned_panels) then never straddle two lines.
inline constexpr size_t kAlignment = 64;
// Each worker's S block starts on its own 4 KiB page.
inline constexpr size_t kBlockAlignment = 4096;

// R's block is a huge-page mapping from this size on (see the top).
inline constexpr size_t kRBlockHugeBytes = size_t(8) << 20;
// No huge-page mapping at any size.
inline constexpr size_t kNoHuge = SIZE_MAX;

// `count` values, kAlignment-aligned, uninitialized: a huge-page mapping
// from `huge_from` bytes on (never below spalm_huge::kHugeBytes),
// posix_memalign otherwise or if the mapping fails. Empty on failure; never
// throws, so workers may call it inside a parallel region.
template <typename T>
spalm_huge::array<T> try_make_buffer(
    size_t count, size_t huge_from = spalm_huge::kHugeBytes
) {
    if (count == 0 || count > SIZE_MAX / sizeof(T)) return {};
    const size_t bytes = count * sizeof(T);
    if (bytes >= huge_from && bytes >= spalm_huge::kHugeBytes) {
        spalm_huge::mapping m;
        if (void *p = spalm_huge::map_huge(bytes, m)) {
            return spalm_huge::array<T>(
                static_cast<T *>(p), spalm_huge::deleter{m}
            );
        }
    }
    void *p = nullptr;
    if (posix_memalign(&p, kAlignment, bytes) != 0) return {};
    return spalm_huge::array<T>(static_cast<T *>(p), spalm_huge::deleter{});
}
// A bool here would silently become a size of 0 or 1 byte: do not compile.
template <typename T> spalm_huge::array<T> try_make_buffer(size_t, bool) = delete;

template <typename T>
spalm_huge::array<T> make_buffer(
    size_t count, size_t huge_from = spalm_huge::kHugeBytes
) {
    auto buffer = try_make_buffer<T>(count, huge_from);
    if (!buffer) throw std::bad_alloc();
    return buffer;
}

// One block of `count` values per worker, uninitialized. When the blocks
// together reach spalm_huge::kHugeBytes they are slices of one huge-page
// mapping, each starting kBlockAlignment-aligned; otherwise (or if that
// mapping fails) each worker's block is allocated on its first get(), with
// posix_memalign as before.
template <typename T> class worker_blocks {
  public:
    worker_blocks(int workers, size_t count)
        : count_(count), own_(workers > 0 ? workers : 0) {
        if (count_ == 0 || workers <= 0 || count_ > SIZE_MAX / sizeof(T))
            return;
        const size_t bytes = count_ * sizeof(T);
        if (bytes > SIZE_MAX - kBlockAlignment) return;
        const size_t stride =
            (bytes + kBlockAlignment - 1) / kBlockAlignment * kBlockAlignment;
        if (stride > SIZE_MAX / static_cast<size_t>(workers)) return;
        const size_t total = stride * static_cast<size_t>(workers);
        if (total < spalm_huge::kHugeBytes) return;
        spalm_huge::mapping m;
        if (void *p = spalm_huge::map_huge(total, m)) {
            shared_ = spalm_huge::array<T>(
                static_cast<T *>(p), spalm_huge::deleter{m}
            );
            stride_ = stride / sizeof(T);
        }
    }

    // The block of `worker` (0 <= worker < workers), or nullptr if it could
    // not be allocated. Each worker asks only for its own block.
    T *get(int worker) {
        if (count_ == 0 || worker < 0 ||
            static_cast<size_t>(worker) >= own_.size())
            return nullptr;
        if (shared_) return shared_.get() + static_cast<size_t>(worker) * stride_;
        auto &own = own_[worker];
        if (!own) own = try_make_buffer<T>(count_, kNoHuge);
        return own.get();
    }

    // Whether the blocks share one huge-page mapping (for tests and logs).
    bool shared() const { return static_cast<bool>(shared_); }

  private:
    size_t count_ = 0;
    size_t stride_ = 0; // values from one block's start to the next
    spalm_huge::array<T> shared_;
    std::vector<spalm_huge::array<T>> own_;
};

} // namespace spalm_packing
