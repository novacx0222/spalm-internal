#pragma once

// SS path with sparse output (init_method sparse through spalm_sparse()).
// Each thread computes one output row at a time into a dense buffer of N
// values: the same additions, in the same order, as into that row of the
// dense result. It then appends the row's nonzeros to its own pages and
// zeroes the buffer again. Rows go in tasks of 4 consecutive rows,
// dynamically scheduled like the dense loop's schedule(dynamic, 4); a
// task's rows form one block, split where a page fills up. No M x N result
// is allocated. Pages of kHugePage entries and more are on huge pages
// (huge_pages.h), smaller ones come from malloc.
//
// Portable: the SIMD multiply is passed in (ss_sparse_product), so the
// tests run the same loop with a scalar one.

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "huge_pages.h"
#include "spalm_sparse_result.h"

// Rows per task: the dense result loop's schedule(dynamic, 4).
constexpr int SS_SPARSE_TASK_ROWS = 4;

template <typename T> class ss_sparse_output;

// One thread's output: pages of (column, value) entries, and per block the
// first row and the row offsets into its page.
template <typename T> class ss_row_output {
public:
    ss_row_output(size_t fixed_page, int N, size_t huge_from = kHugePage)
        : fixed_page_(fixed_page), huge_from_(huge_from), N_(N) {}

    // The dense buffer of one output row: N values, zero between rows.
    // Allocated and first touched by the thread that uses it.
    T *row_buffer() {
        if (!acc_) acc_ = std::make_unique<T[]>(static_cast<size_t>(N_));
        return acc_.get();
    }

    void begin_block(int row) { open(row); }
    void end_row() {
        row_ptrs_.push_back(static_cast<int>(used_ - base_));
        ++row_;
    }
    void end_block() { close(); }

    void emit(int col, T value) {
        if (used_ == capacity_) next_page();
        cols_[used_] = col;
        vals_[used_] = value;
        ++used_;
    }
    // At least n free entries in the current page, for vector stores
    // (n <= 64, the smallest page).
    void reserve(size_t n) {
        if (capacity_ - used_ < n) next_page();
    }
    int *cols_end() { return cols_ + used_; }
    T *vals_end() { return vals_ + used_; }
    void advance(size_t n) { used_ += n; }

    // Rows emitted by scanning the whole buffer range / by revisiting the
    // listed columns.
    long long scanned_rows = 0, listed_rows = 0;

    // Default pages: kFirstPage entries, doubling up to kMaxPage.
    static constexpr size_t kFirstPage = size_t(1) << 16;
    static constexpr size_t kMaxPage = size_t(1) << 22;
    // Pages of at least this many entries (8 and 16 MiB of columns and
    // doubles) are on huge pages: on 4 KiB pages, freeing them took ~50 ms
    // per GB. Smaller pages come from malloc, which in queries with up to
    // some 20M result entries reuses memory the query already freed; on
    // fresh huge mappings these were slower (x1.15 for 3-20M entries, v23
    // SuiteSparse set on the 8488C). Default of the ss_page_huge_entries key,
    // which calibrate_run.py --page-sweep measures per machine.
    static constexpr size_t kHugePage = size_t(1) << 21;
    // ss_page_huge_entries = -1: no page on huge pages.
    static constexpr size_t kNeverHuge = SIZE_MAX;

private:
    friend class ss_sparse_output<T>;
    struct page {
        spalm_huge::array<int> cols;
        spalm_huge::array<T> vals;
        size_t capacity;
    };
    struct pending {
        int row_begin, rows, page;
        size_t base, ptr;
    };

    void open(int row) {
        row_begin_ = row_ = row;
        base_ = used_;
        ptr_ = row_ptrs_.size();
        row_ptrs_.push_back(0);
    }
    // Ends the current block after the rows whose end offsets are stored;
    // a block without entries is dropped.
    void close() {
        const int rows = static_cast<int>(row_ptrs_.size() - ptr_ - 1);
        if (used_ == base_) {
            row_ptrs_.resize(ptr_);
            return;
        }
        blocks_.push_back(
            {row_begin_, rows, static_cast<int>(pages_.size()) - 1, base_,
             ptr_}
        );
    }
    // The page is full in the middle of row row_: the block ends with that
    // row's entries so far, and a new block in a new page continues it.
    void next_page() {
        row_ptrs_.push_back(static_cast<int>(used_ - base_));
        close();
        add_page();
        open(row_);
    }
    void add_page() {
        const size_t cap =
            fixed_page_ ? fixed_page_
            : pages_.empty()
                ? kFirstPage
                : std::min(kMaxPage, pages_.back().capacity * 2);
        const bool huge = cap >= huge_from_;
        page p{
            spalm_huge::make_for_overwrite<int>(cap, huge),
            spalm_huge::make_for_overwrite<T>(cap, huge), cap
        };
        cols_ = p.cols.get();
        vals_ = p.vals.get();
        capacity_ = cap;
        used_ = 0;
        pages_.push_back(std::move(p));
    }

    size_t fixed_page_;
    size_t huge_from_;
    int N_;
    std::unique_ptr<T[]> acc_;
    std::vector<page> pages_;
    std::vector<int> row_ptrs_;
    std::vector<pending> blocks_;
    int *cols_ = nullptr;
    T *vals_ = nullptr;
    size_t capacity_ = 0, used_ = 0;
    // Current block: first row, current row, page offset, row_ptrs_ index.
    int row_begin_ = 0, row_ = 0;
    size_t base_ = 0, ptr_ = 0;
};

// All threads' outputs; owns the pages the result's blocks point into.
template <typename T> class ss_sparse_output {
public:
    // page_entries: entries per page (0: pages grow from 64 Ki to 4 Mi
    // entries; at least 64). Small fixed pages are for tests. Pages of at
    // least huge_from entries are on huge pages (kNeverHuge: none).
    explicit ss_sparse_output(
        size_t page_entries = 0,
        size_t huge_from = ss_row_output<T>::kHugePage
    )
        : fixed_page_(page_entries ? std::max<size_t>(page_entries, 64) : 0),
          huge_from_(huge_from) {}

    void start(int threads, int N) {
        threads_.clear();
        threads_.reserve(static_cast<size_t>(threads));
        for (int t = 0; t < threads; ++t)
            threads_.emplace_back(fixed_page_, N, huge_from_);
    }
    ss_row_output<T> &local(int thread) {
        return threads_[static_cast<size_t>(thread)];
    }

    // Fills result.blocks and result.entries (the arrays stay owned by this
    // object) and frees the row buffers.
    void finish(spalm_sparse_result &result) {
        size_t count = 0;
        for (auto &t : threads_) count += t.blocks_.size();
        result.blocks.clear();
        result.blocks.reserve(count);
        result.entries = 0;
        for (auto &t : threads_) {
            t.acc_.reset();
            for (const auto &b : t.blocks_) {
                const int *ptr = t.row_ptrs_.data() + b.ptr;
                const auto &p = t.pages_[static_cast<size_t>(b.page)];
                result.blocks.push_back(
                    {b.row_begin, b.rows, ptr, ptr + 1, p.cols.get() + b.base,
                     p.vals.get() + b.base}
                );
                result.entries += ptr[b.rows];
            }
        }
    }

    long long scanned_rows() const {
        long long n = 0;
        for (const auto &t : threads_) n += t.scanned_rows;
        return n;
    }
    long long listed_rows() const {
        long long n = 0;
        for (const auto &t : threads_) n += t.listed_rows;
        return n;
    }
    size_t pages() const {
        size_t n = 0;
        for (const auto &t : threads_) n += t.pages_.size();
        return n;
    }
    size_t page_bytes() const {
        size_t n = 0;
        for (const auto &t : threads_)
            for (const auto &p : t.pages_)
                n += p.capacity * (sizeof(int) + sizeof(T));
        return n;
    }
    // The part of page_bytes() on huge pages.
    size_t huge_page_bytes() const {
        size_t n = 0;
        for (const auto &t : threads_)
            for (const auto &p : t.pages_)
                n += p.capacity *
                     (spalm_huge::is_mapped(p.cols) * sizeof(int) +
                      spalm_huge::is_mapped(p.vals) * sizeof(T));
        return n;
    }

private:
    size_t fixed_page_;
    size_t huge_from_;
    std::vector<ss_row_output<T>> threads_;
};

// Appends acc[j] for the columns listed for this row (the S^T lists
// Sa[first[k] .. last[k]) of its keys; a column can be listed more than
// once) when it is not 0, and zeroes every listed position, so each column
// is appended once and the buffer is all zero afterwards.
template <typename T>
inline void ss_emit_listed(
    ss_row_output<T> &out, T *acc, const int *keys, int len, const int *Sa,
    const int *first, const int *last
) {
    for (int p = 0; p < len; ++p) {
        const int k = keys[p];
        for (int q = first[k]; q < last[k]; ++q) {
            const int j = Sa[q];
            const T v = acc[j];
            acc[j] = T(0);
            if (v != T(0)) out.emit(j, v);
        }
    }
}

// Appends the nonzeros of acc[lo, hi) (NaN included, -0.0 not, as the dense
// output's `!= 0`) in column order and zeroes the range.
template <typename T>
inline void ss_emit_range(ss_row_output<T> &out, T *acc, int lo, int hi) {
    int j = lo;
#if defined(__AVX512F__)
    const __m512i lanes = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    if constexpr (std::is_same_v<T, double>) {
        const __m512d zero = _mm512_setzero_pd();
        for (; j + 8 <= hi; j += 8) {
            const __m512d v = _mm512_loadu_pd(acc + j);
            const __mmask8 nonzero = _mm512_cmp_pd_mask(v, zero, _CMP_NEQ_UQ);
            if (nonzero) {
                out.reserve(8);
                _mm512_mask_compressstoreu_pd(out.vals_end(), nonzero, v);
                _mm512_mask_compressstoreu_epi32(
                    out.cols_end(), static_cast<__mmask16>(nonzero),
                    _mm512_add_epi32(lanes, _mm512_set1_epi32(j))
                );
                out.advance(static_cast<size_t>(__builtin_popcount(nonzero)));
            }
            _mm512_storeu_pd(acc + j, zero);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        const __m512 zero = _mm512_setzero_ps();
        for (; j + 16 <= hi; j += 16) {
            const __m512 v = _mm512_loadu_ps(acc + j);
            const __mmask16 nonzero = _mm512_cmp_ps_mask(v, zero, _CMP_NEQ_UQ);
            if (nonzero) {
                out.reserve(16);
                _mm512_mask_compressstoreu_ps(out.vals_end(), nonzero, v);
                _mm512_mask_compressstoreu_epi32(
                    out.cols_end(), nonzero,
                    _mm512_add_epi32(lanes, _mm512_set1_epi32(j))
                );
                out.advance(static_cast<size_t>(__builtin_popcount(nonzero)));
            }
            _mm512_storeu_ps(acc + j, zero);
        }
    }
#endif
    for (; j < hi; ++j) {
        const T v = acc[j];
        acc[j] = T(0);
        if (v != T(0)) out.emit(j, v);
    }
}

// C = R * S^T on the SS path, into `out`. R: CSR row_ptrs / keys, M rows.
// S^T: key k's output columns are Sa[offsets[k] .. offsets[k + 1]); with
// output-column blocks (block_starts, blocks > 1, as in the dense loop)
// block b of key k is [block_starts[b * K + k], block_starts[(b+1) * K + k])
// and covers columns [b * width, (b + 1) * width). kernel(p0, p1, first,
// last, acc) adds R's entries [p0, p1) times their S^T lists into the row
// buffer acc: the dense loop's kernel call on one result row.
//
// A row is emitted by revisiting its listed columns, or, when it lists at
// least a quarter of the block's columns, by scanning the block's range.
// Allocation failures inside the parallel region are rethrown after it.
template <typename T, typename Kernel>
void ss_sparse_product(
    ss_sparse_output<T> &out, int threads, int M, int N, int K,
    const int *row_ptrs, const int *keys, const int *Sa, const int *offsets,
    const int *block_starts, int blocks, Kernel kernel
) {
    const bool blocked = block_starts && blocks > 1;
    const int column_blocks = blocked ? blocks : 1;
    const int width = blocked ? (N + blocks - 1) / blocks : N;
    const int tasks = (M + SS_SPARSE_TASK_ROWS - 1) / SS_SPARSE_TASK_ROWS;
    std::atomic<bool> failed{false};
    out.start(threads, N);
#pragma omp parallel num_threads(threads)
    {
        ss_row_output<T> &rows = out.local(omp_get_thread_num());
        T *acc = nullptr;
        try {
            acc = rows.row_buffer();
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
        }
        for (int b = 0; b < column_blocks; ++b) {
            const size_t kb = static_cast<size_t>(b) * K;
            const int *first = blocked ? block_starts + kb : offsets;
            const int *last = blocked ? block_starts + kb + K : offsets + 1;
            const int lo = std::min(N, b * width);
            const int hi = std::min(N, lo + width);
#pragma omp for schedule(dynamic, 1)
            for (int t = 0; t < tasks; ++t) {
                if (!acc || failed.load(std::memory_order_relaxed)) continue;
                try {
                    const int r0 = t * SS_SPARSE_TASK_ROWS;
                    const int r1 = std::min(M, r0 + SS_SPARSE_TASK_ROWS);
                    rows.begin_block(r0);
                    for (int r = r0; r < r1; ++r) {
                        const int p0 = row_ptrs[r], p1 = row_ptrs[r + 1];
                        kernel(p0, p1, first, last, acc);
                        long long listed = 0;
                        for (int p = p0; p < p1; ++p)
                            listed += last[keys[p]] - first[keys[p]];
                        if (4 * listed >= hi - lo) {
                            ss_emit_range(rows, acc, lo, hi);
                            ++rows.scanned_rows;
                        } else {
                            ss_emit_listed(
                                rows, acc, keys + p0, p1 - p0, Sa, first, last
                            );
                            ++rows.listed_rows;
                        }
                        rows.end_row();
                    }
                    rows.end_block();
                } catch (...) {
                    failed.store(true, std::memory_order_relaxed);
                }
            }
        }
    }
    if (failed.load()) throw std::bad_alloc();
}
