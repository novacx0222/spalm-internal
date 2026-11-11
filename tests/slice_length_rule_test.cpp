// Slice length from counts (slice_length_rule.h): the counts against a
// brute-force reference, and the decision on inputs whose best length is
// known from measurements: uniform 1% rows lengthen to 4x (encoder-1025 and
// the 16000 grid), uniform 20% rows, banded rows and thinly spread
// SuiteSparse-like rows stay at the configured length. Then the two limits:
// lengths within K, and nothing lengthened below kMinCost (the 16000 grid's
// uniform_1 is above it, the same rows at 2000 x 16000 below), with the
// size bound that skips counting. Last, the packing term (v34).
#include "slice_length_rule.h"

#include <cassert>
#include <cmath>
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
            for (int r = 0; r < m.rows; ++r) {
                std::set<int> seen;
                for (int k = m.ptr[r]; k < m.ptr[r + 1]; ++k) {
                    const int col = m.idx[k] / len;
                    ++e[col];
                    seen.insert(col);
                }
                for (int col : seen) ++sr[col];
            }
            assert(c.height == height && c.base == base);
            for (int col = 0; col < cols; ++col) {
                assert(c.entries[l][col] == e[col]);
                assert(c.subrows[l][col] == sr[col]);
                // The slicing packs a column's subrows height at a time.
                assert(slr::packed_slices(c, l, col) ==
                       (sr[col] + height - 1) / height);
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
    assert(c.subrows[0][0] == 0 && slr::packed_slices(c, 2, 0) == 0);
    std::cout << "counts match the reference" << std::endl;
}

// Chosen level for SD (R sparse), DS (S sparse) and DD (both). min_cost 0:
// the doubling rule alone, whatever the size.
static std::tuple<int, int, int> levels_for(
    const test_csr &r, const test_csr &s, int threads = 4,
    double min_cost = 0
) {
    const auto rc = counts(r, 512, 3, 12, threads);
    const auto sc = counts(s, 512, 3, 16, threads);
    const auto sd = slr::choose(rc, sc, true, false, min_cost);
    const auto ds = slr::choose(rc, sc, false, true, min_cost);
    const auto dd = slr::choose(rc, sc, true, true, min_cost);
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

// The level-0 cost, from the counts and from sizes alone.
static double cost0(const test_csr &r, const test_csr &s, bool rs, bool ss) {
    const auto rc = counts(r, 512, 1, 12, 4);
    const auto sc = counts(s, 512, 1, 16, 4);
    return slr::choose(rc, sc, rs, ss).cost0;
}

static double bound(const test_csr &r, const test_csr &s, bool rs, bool ss) {
    return slr::cost0_bound(
        r.idx.size(), r.rows, 12, s.idx.size(), s.rows, 16, rs, ss
    );
}

static void check_limits() {
    // Lengths within K.
    assert(slr::levels_within(512, 0) == 1);
    assert(slr::levels_within(512, 1023) == 1);
    assert(slr::levels_within(512, 1024) == 2);
    assert(slr::levels_within(512, 2047) == 2);
    assert(slr::levels_within(512, 2048) == 3);
    assert(slr::levels_within(512, 100000) == 3);
    assert(slr::levels_within(300, 1700) == 3);
    assert(slr::levels_within(0, 100000) == 1);

    std::mt19937 rng(23);
    std::cout << "uniform 1%, 16000 x 16000 (the grid's uniform_1)" << std::endl;
    {
        const auto a = uniform(rng, 16000, 16000, 0.01);
        const auto c = cost0(a, a, true, false);
        std::cout << "    SD cost0 " << c << ", bound " << bound(a, a, true, false)
                  << std::endl;
        assert(c >= slr::kMinCost);
        assert(levels_for(a, a, 8, slr::kMinCost) == std::make_tuple(2, 2, 2));
    }
    std::cout << "uniform 1%, 2000 x 16000: below kMinCost" << std::endl;
    {
        const auto a = uniform(rng, 2000, 16000, 0.01);
        const auto c = cost0(a, a, true, true);
        std::cout << "    DD cost0 " << c << std::endl;
        assert(c < slr::kMinCost);
        assert(levels_for(a, a, 4, slr::kMinCost) == std::make_tuple(0, 0, 0));
        assert(levels_for(a, a, 4, 0) == std::make_tuple(2, 2, 2));
        // The decision keeps the costs for the log.
        const auto rc = counts(a, 512, 3, 12, 4);
        const auto sc = counts(a, 512, 3, 16, 4);
        const auto d = slr::choose(rc, sc, true, true);
        assert(d.level == 0 && d.cost0 == c && d.cost[2] < 0.5);
        assert(d.cost0 == slr::side_cost(rc, sc, 0) + slr::side_cost(sc, rc, 0));
    }
    // The size bound is never below the cost: uniform, spread, banded, empty
    // rows, and R and S of different heights.
    {
        const test_csr inputs[] = {
            uniform(rng, 2000, 16000, 0.01), uniform(rng, 400, 16000, 0.2),
            spread(rng, 3000, 50000, 7), banded(4096, 2),
            uniform(rng, 500, 16000, 0.0001), // mostly empty rows
        };
        for (const auto &r : inputs)
            for (const auto &s : inputs) {
                if (r.cols != s.cols) continue;
                for (int sides = 1; sides <= 3; ++sides) {
                    const bool rs = sides & 1, ss = sides & 2;
                    assert(cost0(r, s, rs, ss) <= bound(r, s, rs, ss));
                }
            }
        // Only the sparse sides count.
        const auto &a = inputs[0];
        assert(bound(a, a, true, true) ==
               bound(a, a, true, false) + bound(a, a, false, true));
        assert(bound(a, a, false, false) == 0);
    }
    std::cout << "limits as expected" << std::endl;
}

// The per-call parameters (spalm.conf keys asl_*): the defaults decide
// exactly as the constants, and each parameter moves the decision the way
// it should.
static void check_params() {
    // Default params are the constants.
    const slr::params def;
    assert(def.subrow_write_entries == slr::kSubrowWriteEntries);
    assert(def.min_cost_ratio == slr::kMinCostRatio);
    assert(def.min_cost == slr::kMinCost);
    assert(def.max_levels == slr::kMaxLevels);
    // max_levels caps the levels within K, clamped to [1, kMaxLevels].
    assert(slr::levels_within(512, 100000, 1) == 1);
    assert(slr::levels_within(512, 100000, 2) == 2);
    assert(slr::levels_within(512, 100000, 3) == 3);
    assert(slr::levels_within(512, 100000, 9) == 3);
    assert(slr::levels_within(512, 100000, 0) == 1);
    assert(slr::levels_within(512, 1500, 3) == 2);

    std::mt19937 rng(31);
    const auto a = uniform(rng, 2000, 16000, 0.01);
    const auto b = uniform(rng, 400, 16000, 0.2);
    const auto rc = counts(a, 512, 3, 12, 4), sc = counts(a, 512, 3, 16, 4);
    const auto rb = counts(b, 512, 3, 12, 4), sb = counts(b, 512, 3, 16, 4);
    auto same = [](const slr::decision &x, const slr::decision &y) {
        if (x.level != y.level || x.cost0 != y.cost0) return false;
        for (int l = 0; l < slr::kMaxLevels; ++l)
            if (x.cost[l] != y.cost[l] || x.per_subrow[l] != y.per_subrow[l])
                return false;
        return true;
    };
    for (const auto *pair : {&rc, &rb}) {
        const auto &r = *pair;
        const auto &s = pair == &rc ? sc : sb;
        for (int sides = 1; sides <= 3; ++sides) {
            const bool rs = sides & 1, ss = sides & 2;
            for (double mc : {0.0, slr::kMinCost}) {
                slr::params p;
                p.min_cost = mc;
                // The min_cost overload and the params overload agree.
                assert(same(slr::choose(r, s, rs, ss, mc), slr::choose(r, s, rs, ss, p)));
            }
            assert(same(slr::choose(r, s, rs, ss), slr::choose(r, s, rs, ss, def)));
        }
    }
    slr::params p;
    p.min_cost = 0;
    // uniform 1%: 4x with the defaults and no minimum cost ...
    assert(slr::choose(rc, sc, true, true, p).level == 2);
    // ... 2x at most with max_levels 2, never longer with 1.
    p.max_levels = 2;
    assert(slr::choose(rc, sc, true, true, p).level == 1);
    p.max_levels = 1;
    assert(slr::choose(rc, sc, true, true, p).level == 0);
    p.max_levels = 3;
    // A minimum cost above the level-0 cost: not lengthened.
    p.min_cost = slr::choose(rc, sc, true, true, p).cost0 * 1.01;
    assert(slr::choose(rc, sc, true, true, p).level == 0);
    p.min_cost = 0;
    // A doubling must cut the cost below min_cost_ratio: the uniform 1%
    // cost falls to about 0.63 of level 0 at 4x (0.8 at 2x).
    p.min_cost_ratio = 0.5;
    assert(slr::choose(rc, sc, true, true, p).level == 0);
    p.min_cost_ratio = 0.9;
    // Subrows of about 5 entries: not lengthened when 5 already counts as
    // full (subrow_write_entries 4); the write cost shapes the model.
    p.subrow_write_entries = 4;
    assert(slr::choose(rc, sc, true, true, p).level == 0);
    p.subrow_write_entries = 16;
    // uniform 20% (100 entries per subrow) lengthens once a subrow write
    // counts as more than its entries, and the ratio allows a rise.
    assert(slr::choose(rb, sb, true, true, p).level == 0);
    p.subrow_write_entries = 1000;
    p.min_cost_ratio = 10;
    assert(slr::choose(rb, sb, true, true, p).level == 2);
    // The size bound uses the same write cost.
    assert(slr::cost0_bound(1000, 100, 12, 2000, 200, 16, true, true, 0) <
           slr::cost0_bound(1000, 100, 12, 2000, 200, 16, true, true));
    assert(slr::cost0_bound(1000, 100, 12, 2000, 200, 16, true, true, 16) ==
           slr::cost0_bound(1000, 100, 12, 2000, 200, 16, true, true));
    assert(slr::side_cost(rc, sc, 0, 16) == slr::side_cost(rc, sc, 0));
    std::cout << "params as expected" << std::endl;
}

// Graph-like rows with locality: `per_row` distinct keys from a window of
// `width` keys (wrapping) around a random centre per row.
static test_csr clustered(
    std::mt19937 &rng, int rows, int cols, int per_row, int width
) {
    test_csr m;
    m.rows = rows;
    m.cols = cols;
    std::uniform_int_distribution<int> centre(0, cols - 1), offset(0, width - 1);
    for (int r = 0; r < rows; ++r) {
        std::set<int> keys;
        const int c = centre(rng);
        while ((int) keys.size() < per_row) keys.insert((c + offset(rng)) % cols);
        m.idx.insert(m.idx.end(), keys.begin(), keys.end());
        m.ptr.push_back(static_cast<int>(m.idx.size()));
    }
    return m;
}

// Packing (v34): weight 0 decides as before, with the packed values only
// logged. With a weight, clustered rows whose packed values grow with the
// length (32 keys in windows of 4096: 3.8 entries per subrow at 512, packing
// 1.5x at 2x and 2x at 4x) stop short of 4x, SD at the configured length;
// uniform 1% rows, whose packing is small and the same at every length,
// still go to 4x. The minimum cost applies to the multiplication cost.
// Thin clustered rows (4 keys in windows of 2048, like gnutella05) stay at
// the configured length at any weight: counted as sliced, their cost rises
// with the length.
static void check_packing() {
    std::mt19937 rng(43);
    // A double call on the 8488C: R packed at 16 values per key (DS) or 12
    // (DD), S at 16, tiles 128 / 2304, 30 threads.
    auto schedule = [](bool rs, bool ss) {
        slr::packing p;
        p.pack_s = rs;
        p.pack_r = ss;
        p.r_stride = rs ? 12 : 16;
        p.threads = 30;
        return p;
    };
    {
        const auto a = clustered(rng, 8192, 8192, 4, 2048);
        const auto b = clustered(rng, 8192, 8192, 4, 2048);
        std::cout << "4 keys per row in windows of 2048, 8192 x 8192"
                  << std::endl;
        assert(levels_for(a, b) == std::make_tuple(0, 0, 0));
    }
    struct named {
        const char *name;
        test_csr r, s;
        bool uniform;
    };
    const named inputs[] = {
        {"uniform 1%, 2000 x 16000", uniform(rng, 2000, 16000, 0.01),
         uniform(rng, 2000, 16000, 0.01), true},
        {"32 keys per row in windows of 4096, 8192 x 8192",
         clustered(rng, 8192, 8192, 32, 4096),
         clustered(rng, 8192, 8192, 32, 4096), false},
    };
    for (const auto &in : inputs) {
        const auto rc = counts(in.r, 512, 3, 12, 4);
        const auto sc = counts(in.s, 512, 3, 16, 4);
        std::cout << in.name << std::endl;
        for (int sides = 1; sides <= 3; ++sides) {
            const bool rs = sides & 1, ss = sides & 2;
            const auto pack = schedule(rs, ss);
            slr::params p;
            p.min_cost = 0;
            const auto plain = slr::choose(rc, sc, rs, ss, p);
            const auto zero = slr::choose(rc, sc, rs, ss, p, &pack);
            // Weight 0: the same decision and costs, packed values logged,
            // at least as many at every longer length.
            assert(plain.level == 2 && zero.level == 2);
            assert(zero.cost0 == plain.cost0 && zero.cost0 == zero.mult[0]);
            for (int l = 0; l < 3; ++l) {
                assert(zero.cost[l] == plain.cost[l]);
                assert(zero.mult[l] == plain.mult[l]);
                assert(plain.packed[l] == 0 && zero.packed[l] > 0);
            }
            p.pack_value_entries = 2.0;
            const auto two = slr::choose(rc, sc, rs, ss, p, &pack);
            std::cout << "    " << (sides == 1 ? "SD" : sides == 2 ? "DS" : "DD")
                      << " mult " << zero.mult[0] << "," << zero.mult[1] << ","
                      << zero.mult[2] << " packed " << zero.packed[0] << ","
                      << zero.packed[1] << "," << zero.packed[2] << "  level "
                      << two.level << " at 2 entries per packed value"
                      << std::endl;
            if (in.uniform) {
                assert(two.level == 2);
                for (int l = 1; l < 3; ++l)
                    assert(std::fabs(zero.packed[l] / zero.packed[0] - 1) < 0.02);
            } else {
                assert(two.level < 2 && (sides != 1 || two.level == 0));
                assert(zero.packed[2] > 1.3 * zero.packed[0]);
            }
            assert(two.cost0 == zero.cost0 && two.mult[2] == zero.mult[2]);
            const double w0 = zero.mult[0] + 2.0 * zero.packed[0];
            const double want = (zero.mult[1] + 2.0 * zero.packed[1]) / w0;
            assert(std::fabs(two.cost[1] - want) <= 1e-12 * want);
            // The minimum cost compares the multiplication cost.
            p.min_cost = zero.mult[0];
            assert(slr::choose(rc, sc, rs, ss, p, &pack).level == two.level);
            p.min_cost = zero.mult[0] * 1.01;
            assert(slr::choose(rc, sc, rs, ss, p, &pack).level == 0);
        }
    }
    // Default params leave packing out.
    assert(slr::params{}.pack_value_entries == slr::kPackValueEntries &&
           slr::kPackValueEntries == 0.0);
    std::cout << "packing as expected" << std::endl;
}

int main() {
    check_counts_against_reference();
    check_decisions();
    check_limits();
    check_params();
    check_packing();
    return 0;
}
