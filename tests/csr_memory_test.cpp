#include "csr_preparation.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

// This test executes portable preprocessing, never AVX-512 or MKL kernels.
template <class T> void check_transposes(ss_matrix<T> &matrix, int threads) {
    const auto &csr = matrix.csr_mat;
    auto full = transpose_all_sparse(matrix, threads);
    assert(full.offset_count == static_cast<size_t>(csr.cols) + 1);
    assert(full.entries == csr.row_ptrs[csr.rows]);
    std::vector<std::vector<std::pair<int, T>>> expected(csr.cols);
    for (int row = 0; row < csr.rows; ++row)
        for (int p = csr.row_ptrs[row]; p < csr.row_ptrs[row + 1]; ++p)
            expected[csr.col_idxs[p]].emplace_back(row, csr.vals[p]);
    for (int k = 0; k < csr.cols; ++k) {
        assert(
            full.offsets[k + 1] - full.offsets[k] ==
            static_cast<int>(expected[k].size())
        );
        for (int i = full.offsets[k]; i < full.offsets[k + 1]; ++i) {
            const auto &reference = expected[k][i - full.offsets[k]];
            assert(
                full.row_idxs[i] == reference.first &&
                full.values[i] == reference.second
            );
        }
    }
    for (int buckets : {1, 4, 51, 101}) {
        // Make selection cover high/low bucket slices, including empty selections.
        for (int i = 0; i < matrix.s_csc_mat.col_ptrs[matrix.s_csc_mat.cols];
             ++i)
            matrix.s_csc_mat.vals[i].b = (i * 37 + 7) % 101;
        auto part = transpose_sparse_buckets(matrix, buckets, threads);
        assert(
            part.offset_count == static_cast<size_t>(buckets + 1) * csr.cols
        );
        std::vector<std::vector<std::pair<int, T>>> groups(
            static_cast<size_t>(buckets) * csr.cols
        );
        int total = 0;
        for (int i = 0; i < matrix.s_csc_mat.col_ptrs[matrix.s_csc_mat.cols];
             ++i) {
            const auto &slice = matrix.s_csc_mat.vals[i];
            if (slice.b >= buckets) continue;
            for (int j = slice.start; j < slice.start + slice.len; ++j) {
                const auto &subrow = matrix.sr_csc_mat.vals[j];
                for (int p = subrow.start; p < subrow.start + subrow.len; ++p) {
                    groups
                        [static_cast<size_t>(slice.b) * csr.cols +
                         csr.col_idxs[p]]
                            .emplace_back(
                                matrix.sr_csc_mat.row_idxs[j], csr.vals[p]
                            );
                    ++total;
                }
            }
        }
        assert(part.entries == total);
        for (int k = 0; k < csr.cols; ++k) {
            for (int b = 0; b < buckets; ++b) {
                const auto &reference =
                    groups[static_cast<size_t>(b) * csr.cols + k];
                const int start =
                    part.offsets[static_cast<size_t>(b) * csr.cols + k];
                const int end =
                    part.offsets[static_cast<size_t>(b + 1) * csr.cols + k];
                assert(end - start == static_cast<int>(reference.size()));
                for (int p = start; p < end; ++p) {
                    assert(part.row_idxs[p] == reference[p - start].first);
                    assert(part.values[p] == reference[p - start].second);
                }
            }
        }
    }
}

template <class T> void check_cases() {
    std::mt19937 random(20261020);
    for (int trial = 0; trial < 40; ++trial) {
        const int M = trial % 11, K = 257, threads = 8;
        std::vector<int> rows(M + 1, 0), columns;
        std::vector<T> values;
        for (int row = 0; row < M; ++row) {
            for (int k = 0; k < K; ++k) {
                if (random() % 9) continue;
                columns.push_back(k);
                values.push_back(
                    random() % 3 ? T(int(random() % 17) - 8) : T(0)
                );
            }
            rows[row + 1] = static_cast<int>(values.size());
        }
        const auto original_rows = rows, original_columns = columns;
        const auto original_values = values;
        const auto summary =
            summarize_csr_values(rows.data(), values.data(), M, threads);
        prepared_csr<T> prepared;
        prepared.initialize(
            rows.data(), columns.data(), values.data(), M, K, summary, true,
            threads
        );
        assert(prepared.matrix.row_ptrs[M] == summary.nonzero);
        if (summary.nonzero == summary.stored)
            assert(prepared.matrix.row_ptrs == rows.data());
        for (int p = 0; p < prepared.matrix.row_ptrs[M]; ++p)
            assert(prepared.matrix.vals[p] != T(0));
        assert(
            rows == original_rows && columns == original_columns &&
            values == original_values
        );
        ss_matrix<T> matrix;
        csr_to_ss_matrix(
            prepared.matrix.row_ptrs, prepared.matrix.col_idxs,
            prepared.matrix.vals, M, K, 32, 4, matrix, threads
        );
        for (int row = 0; row < M; ++row) {
            int item = matrix.sr_csr_mat.row_ptrs[row];
            int p = prepared.matrix.row_ptrs[row];
            while (p < prepared.matrix.row_ptrs[row + 1]) {
                const int start = p, block = prepared.matrix.col_idxs[p] / 32;
                while (p < prepared.matrix.row_ptrs[row + 1] &&
                       prepared.matrix.col_idxs[p] / 32 == block)
                    ++p;
                assert(matrix.sr_csr_mat.col_idxs[item] == block);
                assert(
                    matrix.sr_csr_mat.vals[item].start == start &&
                    matrix.sr_csr_mat.vals[item].len == p - start
                );
                ++item;
            }
            assert(item == matrix.sr_csr_mat.row_ptrs[row + 1]);
        }
        check_transposes(matrix, threads);
    }
    int rows[] = {0, 3};
    int columns[] = {0, 1, 2};
    T values[] = {
        T(0), std::numeric_limits<T>::infinity(),
        std::numeric_limits<T>::quiet_NaN()
    };
    const auto summary = summarize_csr_values(rows, values, 1, 2);
    assert(!summary.finite && summary.nonzero == 2);
    prepared_csr<T> borrowed;
    borrowed.initialize(rows, columns, values, 1, 3, summary, false, 2);
    assert(borrowed.matrix.vals == values && !borrowed.values);
}

// Malformed CSR is rejected with std::invalid_argument instead of being
// written out of bounds or silently truncated.
static bool
rejects(std::vector<int> rows, std::vector<int> columns, int K, int length) {
    std::vector<double> values(columns.size(), 1.0);
    ss_matrix<double> matrix;
    try {
        csr_to_ss_matrix(
            rows.data(), columns.data(), values.data(),
            static_cast<int>(rows.size()) - 1, K, length, 4, matrix, 4
        );
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

static void check_invalid_inputs() {
    assert(rejects({0, 3}, {5, 1, 7}, 8, 2)); // unsorted within a row
    assert(rejects({0, 2}, {0, 9}, 8, 4));    // key >= K
    assert(rejects({0, 2}, {-1, 3}, 8, 4));   // negative key
    // Only repeated keys can make a subrow longer than uint16_t can hold.
    assert(rejects({0, 70000}, std::vector<int>(70000, 0), 1, 1));
    assert(!rejects({0, 3}, {0, 0, 1}, 8, 4)); // duplicates of normal length
}

int main() {
    assert(init_decision_thresholds("sparse", 1, "cpu"));
    check_cases<float>();
    check_cases<double>();
    // Dense rows need one tuple per block, not per stored entry.
    const int M = 256, K = 4096, length = 512;
    std::vector<int> rows(M + 1), columns(M * K);
    std::vector<double> values(M * K, 1.0);
    for (int row = 0; row < M; ++row) {
        rows[row] = row * K;
        for (int k = 0; k < K; ++k) columns[row * K + k] = k;
    }
    rows[M] = M * K;
    ss_matrix<double> dense;
    csr_to_ss_matrix(
        rows.data(), columns.data(), values.data(), M, K, length, 12, dense, 8
    );
    assert(
        dense.subrow_workspace_bytes ==
        size_t(M) * (K / length) * sizeof(sr_tuple)
    );
    std::cout << "Dense tuple allocation: " << size_t(M) * K * sizeof(sr_tuple)
              << " -> " << dense.subrow_workspace_bytes << " bytes\n";
    // Large raw dimension: vanilla workspace must not grow with thread count.
    int tiny_rows[] = {0, 1, 2}, tiny_cols[] = {0, (1 << 20) - 1};
    double tiny_vals[] = {2, 3};
    ss_matrix<double> tiny;
    csr_to_ss_matrix(
        tiny_rows, tiny_cols, tiny_vals, 2, 1 << 20, length, 12, tiny, 8
    );
    auto wide = transpose_all_sparse(tiny, 8);
    assert(wide.offset_count == (1 << 20) + 1 && wide.entries == 2);
    assert(wide.workspace_bytes == 0);
    assert(wide.offsets[1] == 1 && wide.offsets[1 << 20] == 2);
    assert(
        wide.row_idxs[0] == 0 && wide.values[0] == 2 && wide.row_idxs[1] == 1 &&
        wide.values[1] == 3
    );
    // Narrow matrices retain row parallelism with an explicitly bounded buffer.
    const int tall_rows = 4096, narrow_cols = 64;
    std::vector<int> narrow_rows(tall_rows + 1),
        narrow_columns(tall_rows * narrow_cols);
    std::vector<double> narrow_values(tall_rows * narrow_cols);
    for (int r = 0; r < tall_rows; ++r) {
        narrow_rows[r] = r * narrow_cols;
        for (int k = 0; k < narrow_cols; ++k) {
            narrow_columns[r * narrow_cols + k] = k;
            narrow_values[r * narrow_cols + k] = r + k + 1;
        }
    }
    narrow_rows[tall_rows] = tall_rows * narrow_cols;
    ss_matrix<double> narrow;
    csr_to_ss_matrix(
        narrow_rows.data(), narrow_columns.data(), narrow_values.data(),
        tall_rows, narrow_cols, 512, 12, narrow, 8
    );
    auto narrow_t = transpose_all_sparse(narrow, 8);
    assert(narrow_t.workspace_bytes == 8 * narrow_cols * sizeof(int));
    assert(narrow_t.entries == tall_rows * narrow_cols);
    for (int k = 0; k < narrow_cols; ++k) {
        assert(narrow_t.offsets[k] == k * tall_rows);
        for (int r = 0; r < tall_rows; ++r) {
            assert(narrow_t.row_idxs[k * tall_rows + r] == r);
            assert(narrow_t.values[k * tall_rows + r] == r + k + 1);
        }
    }
    assert(checked_matrix_size(100000, 100000) == size_t(10000000000ULL));
    bool rejected = false;
    try {
        checked_matrix_size(-1, 2);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    check_invalid_inputs();
    std::cout << "PASS: zero compaction, borrowing/ownership, float/double, "
                 "subrows, exact full/bucketed transposes, bounded workspace, "
                 "raw wide K, 80 randomized cases\n";
}
