#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ss_mat.h"

// rows * cols as size_t, rejecting negative or overflowing dimensions.
inline size_t checked_matrix_size(int rows, int cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Negative matrix dimensions");
    }
    const size_t r = static_cast<size_t>(rows), c = static_cast<size_t>(cols);
    if (c && r > std::numeric_limits<size_t>::max() / c) {
        throw std::overflow_error("Matrix element count overflow");
    }
    return r * c;
}

struct csr_value_summary {
    int stored = 0;
    int nonzero = 0;
    bool finite = true;
};

// One parallel pass over the stored values: how many are nonzero and whether
// all are finite (explicit zeros may only be dropped when they are).
template <typename T>
csr_value_summary summarize_csr_values(
    const int *row_ptrs, const T *values, int rows, int threads
) {
    csr_value_summary summary;
    summary.stored = row_ptrs[rows];
    int nonzero = 0;
    bool finite = true;
#pragma omp parallel for num_threads(threads) if (summary.stored >= 32768)     \
    reduction(+ : nonzero) reduction(&& : finite)
    for (int i = 0; i < summary.stored; ++i) {
        const T value = values[i];
        nonzero += value != T(0);
        finite = finite && std::isfinite(value);
    }
    summary.nonzero = nonzero;
    summary.finite = finite;
    return summary;
}

// Borrows the caller's CSR when it has no explicit zeros to remove, and only
// then allocates a compact copy. The caller's arrays are never modified.
template <typename T> struct prepared_csr {
    csr<T> matrix{};
    std::unique_ptr<int[]> rows, columns;
    std::unique_ptr<T[]> values;

    void initialize(
        int *row_ptrs, int *col_idxs, T *vals, int row_count, int col_count,
        const csr_value_summary &summary, bool prune_zeros, int threads
    ) {
        matrix = {row_ptrs, col_idxs, vals, row_count, col_count};
        if (!prune_zeros || summary.nonzero == summary.stored) return;
        rows = std::make_unique_for_overwrite<int[]>(
            static_cast<size_t>(row_count) + 1
        );
        rows[0] = 0;
#pragma omp parallel for num_threads(threads)
        for (int row = 0; row < row_count; ++row) {
            int count = 0;
            for (int i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i)
                count += vals[i] != T(0);
            rows[row + 1] = count;
        }
        for (int row = 0; row < row_count; ++row) rows[row + 1] += rows[row];
        columns = std::make_unique_for_overwrite<int[]>(summary.nonzero);
        values = std::make_unique_for_overwrite<T[]>(summary.nonzero);
#pragma omp parallel for num_threads(threads)
        for (int row = 0; row < row_count; ++row) {
            int out = rows[row];
            for (int i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
                if (vals[i] == T(0)) continue;
                columns[out] = col_idxs[i];
                values[out++] = vals[i];
            }
        }
        matrix = {
            rows.get(), columns.get(), values.get(), row_count, col_count
        };
    }
};

// A transposed (CSC) copy of the right input, as consumed by the SS kernels:
// offsets[k] .. offsets[k + 1] are column k's (row, value) entries, in
// increasing row order. The bucketed variant stores (bucket_count + 1) * K
// offsets instead (see transpose_sparse_buckets).
template <typename T> struct sparse_transpose {
    std::unique_ptr<int[]> offsets, row_idxs;
    std::unique_ptr<T[]> values;
    int entries = 0;
    size_t offset_count = 0;
    size_t workspace_bytes = 0; // temporary histogram, diagnostics only
};

// Full transpose. The K + 1 offsets array is reused as counter, end cursor and
// start; no threads * K cursor array. Slice columns cover disjoint key
// ranges, so threads writing different slice columns never share a counter.
// Narrow inputs (fewer slice columns than threads) keep row parallelism with
// a per-thread histogram, but only when it costs at most 1 MiB in total.
template <typename T>
sparse_transpose<T>
transpose_all_sparse(const ss_matrix<T> &input, int threads) {
    sparse_transpose<T> result;
    const int K = input.csr_mat.cols;
    const int slice_cols = input.sr_csc_mat.cols;
    result.entries = input.csr_mat.row_ptrs[input.csr_mat.rows];
    result.offset_count = static_cast<size_t>(K) + 1;
    result.offsets = std::make_unique<int[]>(result.offset_count);
    result.row_idxs = std::make_unique_for_overwrite<int[]>(result.entries);
    result.values = std::make_unique_for_overwrite<T[]>(result.entries);
    const size_t histogram_slots = checked_matrix_size(K, threads);
    if (slice_cols < threads && result.entries >= 32768 &&
        histogram_slots <= (1U << 20) / sizeof(int)) {
        auto cursors = std::make_unique<int[]>(histogram_slots);
        result.workspace_bytes = histogram_slots * sizeof(int);
        std::vector<int> ranges;
        compute_thread_row_ranges(
            input.csr_mat.row_ptrs, input.csr_mat.rows, threads, ranges
        );
#pragma omp parallel for num_threads(threads)
        for (int t = 0; t < threads; ++t) {
            int *counts = cursors.get() + static_cast<size_t>(t) * K;
            for (int p = input.csr_mat.row_ptrs[ranges[t]];
                 p < input.csr_mat.row_ptrs[ranges[t + 1]]; ++p)
                ++counts[input.csr_mat.col_idxs[p]];
        }
        int end = 0;
        for (int k = 0; k < K; ++k) {
            result.offsets[k] = end;
            for (int t = 0; t < threads; ++t) {
                auto &cursor = cursors[static_cast<size_t>(t) * K + k];
                end += cursor;
                cursor = end;
            }
        }
        result.offsets[K] = end;
        // Fill backwards from each thread's end cursors: rows stay ascending.
#pragma omp parallel for num_threads(threads)
        for (int t = 0; t < threads; ++t) {
            int *counts = cursors.get() + static_cast<size_t>(t) * K;
            for (int row = ranges[t + 1] - 1; row >= ranges[t]; --row) {
                for (int p = input.csr_mat.row_ptrs[row + 1] - 1;
                     p >= input.csr_mat.row_ptrs[row]; --p) {
                    const int out = --counts[input.csr_mat.col_idxs[p]];
                    result.row_idxs[out] = row;
                    result.values[out] = input.csr_mat.vals[p];
                }
            }
        }
        return result;
    }
#pragma omp parallel for num_threads(threads)
    for (int col = 0; col < slice_cols; ++col) {
        for (int i = input.sr_csc_mat.col_ptrs[col];
             i < input.sr_csc_mat.col_ptrs[col + 1]; ++i) {
            const auto &sr = input.sr_csc_mat.vals[i];
            for (int p = sr.start; p < sr.start + sr.len; ++p)
                ++result.offsets
                      [static_cast<size_t>(input.csr_mat.col_idxs[p]) + 1];
        }
    }
    for (int k = 0; k < K; ++k) result.offsets[k + 1] += result.offsets[k];
    // Fill backwards from column ends; the resulting cursors become starts.
#pragma omp parallel for num_threads(threads)
    for (int col = 0; col < slice_cols; ++col) {
        for (int i = input.sr_csc_mat.col_ptrs[col + 1] - 1;
             i >= input.sr_csc_mat.col_ptrs[col]; --i) {
            const auto &sr = input.sr_csc_mat.vals[i];
            for (int p = sr.start + sr.len - 1; p >= sr.start; --p) {
                const int out =
                    --result.offsets
                          [static_cast<size_t>(input.csr_mat.col_idxs[p]) + 1];
                result.row_idxs[out] = input.sr_csc_mat.row_idxs[i];
                result.values[out] = input.csr_mat.vals[p];
            }
        }
    }
    for (int k = 0; k < K; ++k) result.offsets[k] = result.offsets[k + 1];
    result.offsets[K] = result.entries;
    return result;
}

// Transpose of only the entries in slices with bucket < bucket_count, for the
// mixed SS path. offsets[b * K + k] is where bucket b starts in column k, and
// offsets[bucket_count * K + k] is column k's end, so entries of buckets
// below t are offsets[k] .. offsets[t * K + k]. One bucket-major array serves
// as histogram, end cursors and final starts (no second (B + 1) * K array or
// layout copy), and entry arrays hold only sparse-path entries.
template <typename T>
sparse_transpose<T> transpose_sparse_buckets(
    const ss_matrix<T> &input, int bucket_count, int threads
) {
    sparse_transpose<T> result;
    if (bucket_count <= 0) return result;
    const int K = input.csr_mat.cols;
    result.offset_count = checked_matrix_size(K, bucket_count + 1);
    result.offsets = std::make_unique<int[]>(result.offset_count);
    const int slice_cols = input.s_csc_mat.cols;
#pragma omp parallel for num_threads(threads)
    for (int col = 0; col < slice_cols; ++col) {
        for (int i = input.s_csc_mat.col_ptrs[col];
             i < input.s_csc_mat.col_ptrs[col + 1]; ++i) {
            const auto &s = input.s_csc_mat.vals[i];
            if (s.b >= bucket_count) continue;
            for (int j = s.start; j < s.start + s.len; ++j) {
                const auto &sr = input.sr_csc_mat.vals[j];
                for (int p = sr.start; p < sr.start + sr.len; ++p)
                    ++result.offsets
                          [static_cast<size_t>(s.b) * K +
                           input.csr_mat.col_idxs[p]];
            }
        }
    }
    int end = 0;
    for (int k = 0; k < K; ++k) {
        for (int b = 0; b < bucket_count; ++b) {
            auto &cursor = result.offsets[static_cast<size_t>(b) * K + k];
            end += cursor;
            cursor = end;
        }
        result.offsets[static_cast<size_t>(bucket_count) * K + k] = end;
    }
    result.entries = end;
    result.row_idxs = std::make_unique_for_overwrite<int[]>(result.entries);
    result.values = std::make_unique_for_overwrite<T[]>(result.entries);
#pragma omp parallel for num_threads(threads)
    for (int col = 0; col < slice_cols; ++col) {
        for (int i = input.s_csc_mat.col_ptrs[col + 1] - 1;
             i >= input.s_csc_mat.col_ptrs[col]; --i) {
            const auto &s = input.s_csc_mat.vals[i];
            if (s.b >= bucket_count) continue;
            for (int j = s.start + s.len - 1; j >= s.start; --j) {
                const auto &sr = input.sr_csc_mat.vals[j];
                for (int p = sr.start + sr.len - 1; p >= sr.start; --p) {
                    const int out = --result.offsets
                                          [static_cast<size_t>(s.b) * K +
                                           input.csr_mat.col_idxs[p]];
                    result.row_idxs[out] = input.sr_csc_mat.row_idxs[j];
                    result.values[out] = input.csr_mat.vals[p];
                }
            }
        }
    }
    return result;
}
