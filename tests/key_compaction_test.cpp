// Key compaction (key_domain, build_key_domain, prepared_csr with keys):
// both bitmap paths against a std::set reference, word boundaries, the
// bitmap budget, zero pruning together with compaction, and sorted output.
#include "csr_preparation.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

struct test_csr {
    std::vector<int> rows{0}, cols;
    std::vector<double> vals;
};

// Each row takes sorted distinct keys from `pool`, each with probability p;
// about one value in five is an explicit zero.
static test_csr random_csr(
    std::mt19937 &rng, int row_count, const std::vector<int> &pool, double p
) {
    test_csr m;
    std::bernoulli_distribution take(p), zero(0.2);
    for (int r = 0; r < row_count; ++r) {
        for (int key : pool) {
            if (!take(rng)) continue;
            m.cols.push_back(key);
            m.vals.push_back(zero(rng) ? 0.0 : 1.0 + m.cols.size() % 7);
        }
        m.rows.push_back(static_cast<int>(m.cols.size()));
    }
    return m;
}

static std::set<int> used_keys(const test_csr &m, bool skip_zeros) {
    std::set<int> keys;
    for (size_t i = 0; i < m.cols.size(); ++i)
        if (!skip_zeros || m.vals[i] != 0.0) keys.insert(m.cols[i]);
    return keys;
}

static void check_case(
    std::mt19937 &rng, int K, const std::vector<int> &r_pool,
    const std::vector<int> &s_pool, int threads
) {
    const int M = 37, N = 23;
    test_csr R = random_csr(rng, M, r_pool, 0.6);
    test_csr S = random_csr(rng, N, s_pool, 0.6);
    for (bool skip_zeros : {false, true}) {
        std::set<int> r_keys = used_keys(R, skip_zeros);
        std::set<int> s_keys = used_keys(S, skip_zeros);
        std::vector<int> common;
        for (int key : r_keys)
            if (s_keys.count(key)) common.push_back(key);
        // Private bitmaps (default budget) and the shared atomic bitmap
        // (private budget 0) must give the same domain.
        for (size_t private_budget :
             {KEY_PRIVATE_BITMAPS_BUDGET_BYTES, size_t(0)}) {
            key_domain domain;
            assert(build_key_domain(
                domain, R.rows.data(), R.cols.data(), R.vals.data(), M,
                S.rows.data(), S.cols.data(), S.vals.data(), N, K, skip_zeros,
                threads, KEY_BITMAP_BUDGET_BYTES, private_budget
            ));
            assert(domain.input_keys == K);
            assert(domain.common_keys == static_cast<int>(common.size()));
            assert(domain.same_key_sets == (r_keys == s_keys));
            for (int key = 0; key < K; ++key) {
                const auto it =
                    std::lower_bound(common.begin(), common.end(), key);
                const bool expected = it != common.end() && *it == key;
                assert(domain.contains(key) == expected);
                if (expected) assert(domain.rank(key) == it - common.begin());
            }
            // Compacted copy: kept entries are common (and nonzero when
            // pruning), renumbered by rank, still strictly increasing per row.
            csr_value_summary summary =
                summarize_csr_values(R.rows.data(), R.vals.data(), M, threads);
            prepared_csr<double> prepared;
            prepared.initialize(
                R.rows.data(), R.cols.data(), R.vals.data(), M, K, summary,
                skip_zeros, threads, &domain
            );
            assert(prepared.matrix.cols == domain.common_keys);
            assert(prepared.matrix.row_ptrs != R.rows.data());
            for (int r = 0; r < M; ++r) {
                std::vector<int> expected_cols;
                std::vector<double> expected_vals;
                for (int i = R.rows[r]; i < R.rows[r + 1]; ++i) {
                    if (skip_zeros && R.vals[i] == 0.0) continue;
                    if (!domain.contains(R.cols[i])) continue;
                    expected_cols.push_back(domain.rank(R.cols[i]));
                    expected_vals.push_back(R.vals[i]);
                }
                const int begin = prepared.matrix.row_ptrs[r];
                const int end = prepared.matrix.row_ptrs[r + 1];
                assert(end - begin == static_cast<int>(expected_cols.size()));
                for (int i = begin; i < end; ++i) {
                    assert(
                        prepared.matrix.col_idxs[i] == expected_cols[i - begin]
                    );
                    assert(prepared.matrix.vals[i] == expected_vals[i - begin]);
                    assert(prepared.matrix.col_idxs[i] < domain.common_keys);
                    if (i > begin)
                        assert(
                            prepared.matrix.col_idxs[i - 1] <
                            prepared.matrix.col_idxs[i]
                        );
                }
            }
        }
    }
}

int main() {
    std::mt19937 rng(7);
    for (int threads : {1, 3, 8}) {
        // Word boundaries: 63/64, the last key, K not a multiple of 64.
        check_case(
            rng, 130, {0, 1, 62, 63, 64, 65, 127, 128, 129},
            {1, 63, 64, 100, 129}, threads
        );
        // Disjoint except one key; identical key sets.
        check_case(rng, 1000, {3, 500, 999}, {4, 500, 998}, threads);
        check_case(rng, 200, {5, 70, 199}, {5, 70, 199}, threads);
        // Wide, sparse key range.
        std::vector<int> r_pool, s_pool;
        for (int k = 0; k < 200000; k += 997) r_pool.push_back(k);
        for (int k = 0; k < 200000; k += 1009) s_pool.push_back(k);
        r_pool.push_back(199999);
        s_pool.push_back(199999);
        check_case(rng, 200000, r_pool, s_pool, threads);
    }
    // A key outside [0, K) is an invalid argument, on both bitmap paths.
    int rows[] = {0, 2}, cols[] = {1, 9};
    double vals[] = {1, 2};
    for (size_t private_budget :
         {KEY_PRIVATE_BITMAPS_BUDGET_BYTES, size_t(0)}) {
        key_domain domain;
        bool threw = false;
        try {
            build_key_domain(
                domain, rows, cols, vals, 1, rows, cols, vals, 1, 8, false, 2,
                KEY_BITMAP_BUDGET_BYTES, private_budget
            );
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
    }
    // Bitmaps over budget: not built, inputs used as they are.
    key_domain domain;
    int in_range[] = {1, 5};
    assert(!build_key_domain(
        domain, rows, in_range, vals, 1, rows, in_range, vals, 1, 1 << 20,
        false, 2, 1024
    ));
    assert(domain.common.empty());
    std::cout << "PASS: key domain on private and shared bitmaps, ranks, word "
                 "boundaries, budget, compaction with zero pruning\n";
}
