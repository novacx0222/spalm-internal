#pragma once

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

// Join-key compaction. A key used by only one input matches nothing, so its
// entries add no product. The keys used by both inputs ("common") are
// renumbered by rank, the number of common keys below them: rank is monotone,
// so compacted rows keep their sorted column order, and every kernel and MKL
// backend then works on K' = |common| <= K keys.
struct key_domain {
    int input_keys = 0;  // K
    int common_keys = 0; // K'
    // Both inputs use exactly the common keys: compaction drops no entry.
    bool same_key_sets = false;
    std::vector<uint64_t> common; // bit k: key k is used by both inputs
    std::vector<int> word_rank;   // common keys in the words before word w

    bool contains(int key) const {
        return (common[key >> 6] >> (key & 63)) & 1;
    }
    int rank(int key) const {
        const uint64_t below =
            common[key >> 6] & ((uint64_t(1) << (key & 63)) - 1);
        return word_rank[key >> 6] + __builtin_popcountll(below);
    }
};

// Bitmaps above this size (K above 512M keys) are not built; the inputs are
// then used without key compaction.
inline constexpr size_t KEY_BITMAP_BUDGET_BYTES = size_t(64) << 20;
// Workers mark keys in private bitmaps while all of them together fit in this
// budget (K below ~800K keys at 20 threads), and in one shared bitmap with
// atomic ORs otherwise.
inline constexpr size_t KEY_PRIVATE_BITMAPS_BUDGET_BYTES = size_t(16) << 20;

// Sets bit k of `bits` for the key k of every entry that takes part (a
// nonzero one when zeros are being removed). Returns false if a key is
// outside [0, K); the caller reports it after the parallel regions.
template <typename T>
bool mark_keys(
    const int *row_ptrs, const int *col_idxs, const T *vals, int rows, int K,
    bool skip_zeros, std::vector<uint64_t> &bits, int threads,
    size_t private_budget
) {
    const size_t words = bits.size();
    int bad = 0;
    if (words * sizeof(uint64_t) * threads <= private_budget) {
        // Small K, typically many entries per key: private bitmaps avoid any
        // contention; one OR pass over threads * K / 64 words merges them.
        std::vector<int> ranges;
        compute_thread_row_ranges(row_ptrs, rows, threads, ranges);
        std::vector<uint64_t> local(words * threads, 0);
        // One iteration per range, so a smaller OpenMP team still covers all.
#pragma omp parallel for num_threads(threads) schedule(static)                 \
    reduction(| : bad)
        for (int t = 0; t < threads; ++t) {
            uint64_t *mine = local.data() + words * t;
            for (int i = row_ptrs[ranges[t]]; i < row_ptrs[ranges[t + 1]];
                 ++i) {
                const int key = col_idxs[i];
                if (key < 0 || key >= K) {
                    bad = 1;
                    continue;
                }
                if (skip_zeros && vals[i] == T(0)) continue;
                mine[key >> 6] |= uint64_t(1) << (key & 63);
            }
        }
#pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t w = 0; w < words; ++w) {
            uint64_t merged = 0;
            for (int t = 0; t < threads; ++t) merged |= local[words * t + w];
            bits[w] = merged;
        }
    } else {
        // Wide K, few entries per key: shared bitmap. Testing the bit first
        // keeps repeated keys at a plain load instead of an atomic RMW.
#pragma omp parallel for num_threads(threads) schedule(dynamic, 256)           \
    reduction(| : bad)
        for (int row = 0; row < rows; ++row) {
            for (int i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
                const int key = col_idxs[i];
                if (key < 0 || key >= K) {
                    bad = 1;
                    continue;
                }
                if (skip_zeros && vals[i] == T(0)) continue;
                const uint64_t bit = uint64_t(1) << (key & 63);
                std::atomic_ref<uint64_t> word(bits[key >> 6]);
                if (!(word.load(std::memory_order_relaxed) & bit))
                    word.fetch_or(bit, std::memory_order_relaxed);
            }
        }
    }
    return !bad;
}

// Builds the common key set of R (rows x K) and S (rows x K). Returns false,
// leaving `domain` empty, if the bitmaps would exceed `bitmap_budget`.
// Throws std::invalid_argument for a column index outside [0, K).
template <typename T>
bool build_key_domain(
    key_domain &domain, const int *Ri, const int *Rj, const T *Rv, int M,
    const int *Si, const int *Sj, const T *Sv, int N, int K, bool skip_zeros,
    int threads, size_t bitmap_budget = KEY_BITMAP_BUDGET_BYTES,
    size_t private_budget = KEY_PRIVATE_BITMAPS_BUDGET_BYTES
) {
    const size_t words = (static_cast<size_t>(K) + 63) / 64;
    if (words * sizeof(uint64_t) > bitmap_budget) return false;
    std::vector<uint64_t> r_keys(words, 0), s_keys(words, 0);
    const bool valid =
        mark_keys(
            Ri, Rj, Rv, M, K, skip_zeros, r_keys, threads, private_budget
        ) &&
        mark_keys(
            Si, Sj, Sv, N, K, skip_zeros, s_keys, threads, private_budget
        );
    if (!valid) {
        throw std::invalid_argument("SPALM CSR column index out of [0, K)");
    }
    domain.input_keys = K;
    domain.common.resize(words);
    domain.word_rank.resize(words);
    int different = 0;
#pragma omp parallel for num_threads(threads) schedule(static)                 \
    reduction(| : different)
    for (size_t w = 0; w < words; ++w) {
        domain.common[w] = r_keys[w] & s_keys[w];
        domain.word_rank[w] = __builtin_popcountll(domain.common[w]);
        different |= r_keys[w] != s_keys[w];
    }
    int common = 0;
    for (size_t w = 0; w < words; ++w) {
        const int count = domain.word_rank[w];
        domain.word_rank[w] = common;
        common += count;
    }
    domain.common_keys = common;
    domain.same_key_sets = !different;
    return true;
}

// Borrows the caller's CSR when there are no explicit zeros to remove and no
// key compaction, and only then allocates a compact copy: entries are kept if
// nonzero (when pruning) and, with `keys`, if their key is common, which is
// then renumbered to its rank. The caller's arrays are never modified.
template <typename T> struct prepared_csr {
    csr<T> matrix{};
    std::unique_ptr<int[]> rows, columns;
    std::unique_ptr<T[]> values;

    void initialize(
        int *row_ptrs, int *col_idxs, T *vals, int row_count, int col_count,
        const csr_value_summary &summary, bool prune_zeros, int threads,
        const key_domain *keys = nullptr
    ) {
        matrix = {row_ptrs, col_idxs, vals, row_count, col_count};
        const bool drop_zeros =
            prune_zeros && summary.nonzero != summary.stored;
        if (!drop_zeros && !keys) return;
        auto keep = [&](int i) {
            return (!drop_zeros || vals[i] != T(0)) &&
                   (!keys || keys->contains(col_idxs[i]));
        };
        rows = std::make_unique_for_overwrite<int[]>(
            static_cast<size_t>(row_count) + 1
        );
        rows[0] = 0;
#pragma omp parallel for num_threads(threads)
        for (int row = 0; row < row_count; ++row) {
            int count = 0;
            for (int i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i)
                count += keep(i);
            rows[row + 1] = count;
        }
        for (int row = 0; row < row_count; ++row) rows[row + 1] += rows[row];
        const int entries = rows[row_count];
        columns = std::make_unique_for_overwrite<int[]>(entries);
        values = std::make_unique_for_overwrite<T[]>(entries);
#pragma omp parallel for num_threads(threads)
        for (int row = 0; row < row_count; ++row) {
            int out = rows[row];
            for (int i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
                if (!keep(i)) continue;
                columns[out] = keys ? keys->rank(col_idxs[i]) : col_idxs[i];
                values[out++] = vals[i];
            }
        }
        matrix = {
            rows.get(), columns.get(), values.get(), row_count,
            keys ? keys->common_keys : col_count
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
