// Slice length from counts (slice_length_rule.h): the counts against a
// brute-force reference, and the decision on inputs whose best length is
// known from measurements: uniform 1% rows lengthen to 4x (encoder-1025 and
// the 16000 grid), uniform 20% rows, banded rows and thinly spread
// SuiteSparse-like rows stay at the configured length.
#include "slice_length_rule.h"

#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <tuple>
#include <vector>

namespace slr = slice_length_rule;

struct test_csr {
    int rows = 0, cols = 0;
    std::vector<int> ptr{0}, idx;
};

// Each key of each row is present with probability p.
static test_csr uniform(std::mt19937 &rng, int rows, int cols, double p) {
    test_csr m;
    m.rows = rows;
    m.cols = cols;
    std::geometric_distribution<int> gap(p);
    for (int r = 0; r < rows; ++r) {
        for (long long k = gap(rng); k < cols; k += 1 + gap(rng))
            m.idx.push_back(static_cast<int>(k));
        m.ptr.push_back(static_cast<int>(m.idx.size()));
    }
    return m;
}

// Exactly `per_row` distinct random keys per row.
static test_csr spread(std::mt19937 &rng, int rows, int cols, int per_row) {
    test_csr m;
    m.rows = rows;
    m.cols = cols;
    std::uniform_int_distribution<int> key(0, cols - 1);
    for (int r = 0; r < rows; ++r) {
        std::set<int> keys;
        while ((int) keys.size() < per_row) keys.insert(key(rng));
        m.idx.insert(m.idx.end(), keys.begin(), keys.end());
        m.ptr.push_back(static_cast<int>(m.idx.size()));
    }
    return m;
}

// Row i holds keys i - half .. i + half (a band around the diagonal).
static test_csr banded(int n, int half) {
    test_csr m;
    m.rows = m.cols = n;
    for (int r = 0; r < n; ++r) {
        for (int k = std::max(0, r - half); k <= std::min(n - 1, r + half); ++k)
            m.idx.push_back(k);
        m.ptr.push_back(static_cast<int>(m.idx.size()));
    }
    return m;
}

static slr::column_counts counts(
    const test_csr &m, int base, int levels, int height, int threads
) {
    return slr::count_columns(
        m.ptr.data(), m.idx.data(), m.rows, m.cols, base, levels, height,
        threads
    );
}

static void check_counts_against_reference() {
    std::mt19937 rng(7);
    // Odd sizes: K not a multiple of the length, rows not of the height,
    // a length that is not a power of two.
    const test_csr m = uniform(rng, 103, 1700, 0.02);
    const int base = 300, levels = 3, height = 12;
    for (int threads : {1, 3, 8, 200}) {
        const auto c = counts(m, base, levels, height, threads);
        assert(c.levels == levels);
        for (int l = 0; l < levels; ++l) {
            const int len = base << l;
            const int cols = (m.cols + len - 1) / len;
            assert((int) c.entries[l].size() == cols);
            std::vector<long long> e(cols), sr(cols);
            std::vector<std::set<int>> sl(cols);
            for (int r = 0; r < m.rows; ++r) {
                std::set<int> seen;
                for (int k = m.ptr[r]; k < m.ptr[r + 1]; ++k) {
                    const int col = m.idx[k] / len;
                    ++e[col];
                    seen.insert(col);
                    sl[col].insert(r / height);
                }
                for (int col : seen) ++sr[col];
            }
            for (int col = 0; col < cols; ++col) {
                assert(c.entries[l][col] == e[col]);
                assert(c.subrows[l][col] == sr[col]);
                assert(c.slices[l][col] == (long long) sl[col].size());
            }
        }
    }
    // Empty input and a single level.
    test_csr empty;
    empty.rows = 5;
    empty.cols = 900;
    empty.ptr.assign(6, 0);
    const auto c = counts(empty, 512, 3, 12, 4);
    assert(c.entries[0].size() == 2 && c.entries[2].size() == 1);
    assert(c.subrows[0][0] == 0 && c.slices[2][0] == 0);
    std::cout << "counts match the reference" << std::endl;
}

// Chosen level for SD (R sparse), DS (S sparse) and DD (both).
static std::tuple<int, int, int> levels_for(
    const test_csr &r, const test_csr &s, int threads = 4
) {
    const auto rc = counts(r, 512, 3, 12, threads);
    const auto sc = counts(s, 512, 3, 16, threads);
    const auto sd = slr::choose(rc, sc, true, false);
    const auto ds = slr::choose(rc, sc, false, true);
    const auto dd = slr::choose(rc, sc, true, true);
    std::cout << "    SD level " << sd.level << " (cost " << sd.cost[1] << ", "
              << sd.cost[2] << "; per subrow " << sd.per_subrow[0] << ")"
              << "  DS level " << ds.level << " (cost " << ds.cost[1] << ")"
              << "  DD level " << dd.level << " (cost " << dd.cost[1] << ")"
              << std::endl;
    return {sd.level, ds.level, dd.level};
}

static void check_decisions() {
    std::mt19937 rng(11);
    std::cout << "uniform 1%, 16000 keys" << std::endl;
    {
        const auto a = uniform(rng, 2000, 16000, 0.01);
        assert(levels_for(a, a) == std::make_tuple(2, 2, 2));
    }
    std::cout << "uniform 1% R, full S (uniform_1 x uniform_100, SD)"
              << std::endl;
    {
        const auto a = uniform(rng, 2000, 16000, 0.01);
        const auto b = uniform(rng, 400, 16000, 1.0);
        assert(std::get<0>(levels_for(a, b)) == 2);
    }
    std::cout << "uniform 20%, 16000 keys" << std::endl;
    {
        const auto a = uniform(rng, 400, 16000, 0.2);
        assert(levels_for(a, a) == std::make_tuple(0, 0, 0));
    }
    std::cout << "10 random keys per row, 100000 keys" << std::endl;
    {
        const auto a = spread(rng, 4000, 100000, 10);
        assert(levels_for(a, a) == std::make_tuple(0, 0, 0));
    }
    std::cout << "band of 51 keys around the diagonal" << std::endl;
    {
        const auto a = banded(8192, 25);
        assert(levels_for(a, a) == std::make_tuple(0, 0, 0));
    }
    std::cout << "band of 3 keys around the diagonal" << std::endl;
    {
        const auto a = banded(8192, 1);
        assert(levels_for(a, a) == std::make_tuple(0, 0, 0));
    }
    // The team size does not change the decision.
    {
        const auto a = uniform(rng, 1000, 16000, 0.01);
        assert(levels_for(a, a, 1) == levels_for(a, a, 7));
    }
    // Neither side sparse (SS only), or a single level: the configured length.
    {
        const auto a = uniform(rng, 100, 4000, 0.01);
        const auto rc = counts(a, 512, 3, 12, 2);
        const auto sc = counts(a, 512, 3, 16, 2);
        assert(slr::choose(rc, sc, false, false).level == 0);
        const auto r1 = counts(a, 512, 1, 12, 2);
        const auto s1 = counts(a, 512, 1, 16, 2);
        assert(slr::choose(r1, s1, true, true).level == 0);
    }
    std::cout << "decisions as expected" << std::endl;
}

int main() {
    check_counts_against_reference();
    check_decisions();
    return 0;
}
