// candidate_work.h against brute force: the features, the result entries
// (exact, sampled, and a long row from column tests), the per-table work and slice
// pairs, and the dense path's packing schedule (a copy of the loops of
// dense_dense_matmul()). Every count must be the same for 1 and 3 threads.
// Also the slice-length rule's packed values (slice_length_rule.h), from
// counts of the CSR alone, against that schedule on the slices themselves.
#include "candidate_work.h"
#include "slice_length_rule.h"

#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <set>

namespace cw = candidate_work;

struct csr_data {
    int rows = 0;
    std::vector<int> ptr{0}, idx;
    std::vector<double> val;
};

// Rows in bands of `band` rows; each band has its own density, so slices
// fall in different buckets.
static csr_data random_csr(
    const int rows, const int K, const std::vector<double> &densities,
    const int band, std::mt19937 &rng
) {
    csr_data m;
    m.rows = rows;
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < rows; ++i) {
        const double d = densities[(i / band) % densities.size()];
        for (int k = 0; k < K; ++k) {
            if (u(rng) < d) {
                m.idx.push_back(k);
                m.val.push_back(1);
            }
        }
        m.ptr.push_back((int) m.idx.size());
    }
    return m;
}

// Slices m; its timing lines (printf) go to /dev/null.
template <typename T>
static void build(csr_data &m, const int K, const int sl, const int sh,
                  ss_matrix<T> &out, const int threads) {
    std::fflush(stdout);
    const int saved = dup(STDOUT_FILENO);
    const int null = open("/dev/null", O_WRONLY);
    assert(saved >= 0 && null >= 0);
    dup2(null, STDOUT_FILENO);
    csr_to_ss_matrix(m.ptr.data(), m.idx.data(), m.val.data(), m.rows, K, sl,
                     sh, out, threads, 1);
    std::fflush(stdout);
    dup2(saved, STDOUT_FILENO);
    close(saved);
    close(null);
}

static unsigned long long brute_entries(const csr_data &R, const csr_data &S,
                                        const int K) {
    std::vector<std::vector<int>> by_key(K);
    for (int j = 0; j < S.rows; ++j)
        for (int p = S.ptr[j]; p < S.ptr[j + 1]; ++p) by_key[S.idx[p]].push_back(j);
    unsigned long long total = 0;
    for (int i = 0; i < R.rows; ++i) {
        std::set<int> cols;
        for (int p = R.ptr[i]; p < R.ptr[i + 1]; ++p)
            cols.insert(by_key[R.idx[p]].begin(), by_key[R.idx[p]].end());
        total += cols.size();
    }
    return total;
}

static void check_features(const csr_data &R, const csr_data &S, const int K,
                           const cw::features &f) {
    std::vector<int> count(K, 0);
    for (int k : S.idx) ++count[k];
    assert(f.s_count == count);
    unsigned long long flops = 0, bound = 0;
    for (int i = 0; i < R.rows; ++i) {
        unsigned long long row = 0;
        for (int p = R.ptr[i]; p < R.ptr[i + 1]; ++p) row += count[R.idx[p]];
        flops += row;
        const unsigned long long b = std::min<unsigned long long>(row, S.rows);
        assert(f.row_bounds[i] == (int) b);
        bound += b;
    }
    assert(f.flops == flops && f.row_bound == bound);
}

// The pairs and work per kernel, slice pair by slice pair, under the global
// table.
static void brute_plan(const ss_matrix<double> &R, const ss_matrix<double> &S,
                       cw::plan &out) {
    const int sl = R.sl, mr = R.sh, nr = S.sh;
    auto entries = [](const ss_matrix<double> &m, int si) {
        const slice &s = m.s_csc_mat.vals[si];
        double e = 0;
        for (int r = s.start; r < s.start + s.len; ++r) e += m.sr_csc_mat.vals[r].len;
        return e;
    };
    for (int col = 0; col < R.s_csc_mat.cols; ++col) {
        for (int a = R.s_csc_mat.col_ptrs[col]; a < R.s_csc_mat.col_ptrs[col + 1]; ++a) {
            for (int b = S.s_csc_mat.col_ptrs[col]; b < S.s_csc_mat.col_ptrs[col + 1]; ++b) {
                const double er = entries(R, a), es = entries(S, b);
                switch (get_alg(R.s_csc_mat.vals[a].b, S.s_csc_mat.vals[b].b)) {
                case SPARSE_SPARSE:
                    out.nnz.ss += er * es / sl;
                    ++out.p.ss;
                    break;
                case DENSE_DENSE_LEFT_INNER_KERNEL:
                    out.nnz.sd += er * nr;
                    ++out.p.sd;
                    break;
                case DENSE_DENSE_RIGHT_INNER_KERNEL:
                    out.nnz.ds += es * mr;
                    ++out.p.ds;
                    break;
                case DENSE_DENSE:
                    out.nnz.dd += (double) sl * mr * nr;
                    ++out.p.dd;
                    break;
                }
            }
        }
    }
}

static bool close(double a, double b) {
    return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(b));
}

// The loops of dense_dense_matmul(), counting instead of packing.
static cw::dense_schedule brute_schedule(const ss_matrix<double> &R,
                                         const ss_matrix<double> &S, int l2,
                                         int l3, int threads) {
    cw::dense_schedule d;
    const int cols = R.s_csc_mat.cols, mr = R.sh, nr = S.sh;
    int max_r = 0, max_s = 0;
    for (int col = 0; col < cols; col++) {
        const int r = R.s_csc_mat.col_ptrs[col + 1] - R.s_csc_mat.col_ptrs[col];
        const int s = S.s_csc_mat.col_ptrs[col + 1] - S.s_csc_mat.col_ptrs[col];
        if (!r || !s) continue;
        max_r = std::max(max_r, r);
        max_s = std::max(max_s, s);
    }
    if (!max_r || !max_s) return d;
    auto div_ceil = [](int a, int b) { return (a + b - 1) / b; };
    auto slice_entries = [](const ss_matrix<double> &m, int si) {
        const slice &s = m.s_csc_mat.vals[si];
        unsigned long long e = 0;
        for (int r = s.start; r < s.start + s.len; ++r) e += m.sr_csc_mat.vals[r].len;
        return e;
    };
    const int S_per_tile = std::min(div_ceil(l2, nr), max_s);
    const int R_per_l3 = std::min(div_ceil(l3, mr), max_r);
    for (int outer = 0; outer < max_r; outer += R_per_l3) {
        for (int col = 0; col < cols; col++) {
            const int R0 = R.s_csc_mat.col_ptrs[col] + outer;
            const int R1 = std::min(R0 + R_per_l3, R.s_csc_mat.col_ptrs[col + 1]);
            if (R0 >= R1 || S.s_csc_mat.col_ptrs[col] == S.s_csc_mat.col_ptrs[col + 1])
                continue;
            for (int r = R0; r < R1; r++) {
                ++d.r_packs;
                d.r_pack_entries += slice_entries(R, r);
            }
            const int s0 = S.s_csc_mat.col_ptrs[col], s1 = S.s_csc_mat.col_ptrs[col + 1];
            const int task_num = div_ceil(s1 - s0, S_per_tile);
            const int R_div = div_ceil(threads, task_num);
            const int per_it = div_ceil(R1 - R0, R_div);
            for (int rt = R0; rt < R1; rt += per_it) {
                for (int s = s0; s < s1; s += S_per_tile) {
                    ++d.tasks;
                    for (int x = s; x < std::min(s + S_per_tile, s1); ++x) {
                        ++d.s_packs;
                        d.s_pack_entries += slice_entries(S, x);
                    }
                }
            }
        }
    }
    return d;
}

static void same(const cw::entries_estimate &a, const cw::entries_estimate &b) {
    assert(a.entries == b.entries && a.exact == b.exact && a.draws == b.draws &&
           a.rows == b.rows && a.visits == b.visits && a.long_rows == b.long_rows);
}

// A mixed table: SS, DS, SD and DD all occur.
static void mixed_table() {
    for (int b = 0; b <= num_buckets; ++b) {
        ss_upper[b] = b < 30 ? 101 : (b < 60 ? 20 : 0);
        ri_upper[b] = b < 60 ? 50 : 0;
    }
    left_dense_threshold = 70;
}

static void check_random() {
    std::mt19937 rng(7);
    struct shape {
        int M, N, K, sl;
        std::vector<double> dr, ds;
        int band;
    };
    const std::vector<shape> shapes = {
        {60, 50, 64, 16, {0.02, 0.3, 0.9}, {0.05, 0.6, 1.0}, 12},
        {200, 150, 300, 32, {0.01, 0.05, 0.4}, {0.02, 0.2, 0.8}, 16},
        {90, 40, 130, 16, {0.0, 0.2}, {0.1, 0.0, 0.7}, 12},
        {300, 400, 1000, 64, {0.003, 0.01}, {0.002, 0.02}, 24},
        // One slice column of about 40000 subrows: estimate_entries() splits
        // it into 16 pieces with 1 thread and 20 with 3.
        {100, 40000, 32, 32, {0.1}, {0.2}, 12},
    };
    for (const shape &sh : shapes) {
        csr_data R = random_csr(sh.M, sh.K, sh.dr, sh.band, rng);
        csr_data S = random_csr(sh.N, sh.K, sh.ds, sh.band, rng);
        const unsigned long long truth = brute_entries(R, S, sh.K);
        cw::entries_estimate exact[2], sampled[2];
        cw::dense_schedule sched[2];
        std::vector<cw::plan> plans[2];
        for (int t = 0; t < 2; ++t) {
            const int threads = t ? 3 : 1;
            ss_matrix<double> Rm, Sm;
            build(R, sh.K, sh.sl, 12, Rm, threads);
            build(S, sh.K, sh.sl, 16, Sm, threads);
            const auto f = cw::count_features(R.ptr.data(), R.idx.data(), sh.M,
                                              Sm, sh.N, sh.K, threads);
            check_features(R, S, sh.K, f);
            exact[t] = cw::estimate_entries(R.ptr.data(), R.idx.data(), sh.M,
                                            Sm, sh.N, sh.K, f, threads);
            assert(exact[t].exact && exact[t].long_rows == 0);
            assert(exact[t].entries == (double) truth);
            sampled[t] = cw::estimate_entries(R.ptr.data(), R.idx.data(), sh.M,
                                              Sm, sh.N, sh.K, f, threads, 1);
            if (f.row_bound) {
                assert(!sampled[t].exact && sampled[t].draws == cw::kMinDraws);
                assert(std::fabs(sampled[t].entries - truth) <= 0.3 * truth);
            }
            mixed_table();
            const int nb = num_buckets;
            const std::vector<uint8_t> table(ss_upper.get(), ss_upper.get() + nb + 1);
            const std::vector<uint8_t> all_ss(nb + 1, (uint8_t) (nb + 1));
            const std::vector<uint8_t> all_dense(nb + 1, (uint8_t) 0);
            plans[t] = cw::plan_counts(Rm, Sm,
                                       {table.data(), all_ss.data(), all_dense.data()},
                                       threads);
            unsigned long long rs, ss, pairs;
            cw::slice_counts(Rm, Sm, rs, ss, pairs);
            const std::vector<uint8_t> *tables[] = {&table, &all_ss, &all_dense};
            for (int p = 0; p < 3; ++p) {
                std::copy(tables[p]->begin(), tables[p]->end(), ss_upper.get());
                cw::plan b;
                brute_plan(Rm, Sm, b);
                const cw::plan &c = plans[t][p];
                assert(c.p.ss == b.p.ss && c.p.sd == b.p.sd && c.p.ds == b.p.ds &&
                       c.p.dd == b.p.dd);
                assert(c.p.ss + c.p.sd + c.p.ds + c.p.dd == pairs);
                assert(close(c.nnz.ss, b.nnz.ss) && close(c.nnz.sd, b.nnz.sd) &&
                       close(c.nnz.ds, b.nnz.ds) && close(c.nnz.dd, b.nnz.dd));
            }
            assert(plans[t][1].p.ss == pairs);
            assert(plans[t][2].p.ss == 0);
            std::copy(table.begin(), table.end(), ss_upper.get());
            sched[t] = cw::dense_schedule_counts(Rm, Sm, 128, 2304, threads + 1);
            const auto b = brute_schedule(Rm, Sm, 128, 2304, threads + 1);
            assert(sched[t].r_packs == b.r_packs &&
                   sched[t].r_pack_entries == b.r_pack_entries &&
                   sched[t].s_packs == b.s_packs &&
                   sched[t].s_pack_entries == b.s_pack_entries &&
                   sched[t].tasks == b.tasks);
            // Smaller tiles: several R blocks and S tiles per column.
            const auto small = cw::dense_schedule_counts(Rm, Sm, 16, 24, 5);
            const auto bs = brute_schedule(Rm, Sm, 16, 24, 5);
            assert(small.r_packs == bs.r_packs && small.s_packs == bs.s_packs &&
                   small.s_pack_entries == bs.s_pack_entries &&
                   small.tasks == bs.tasks);
        }
        same(exact[0], exact[1]);
        same(sampled[0], sampled[1]);
        for (int p = 0; p < 3; ++p) {
            assert(plans[0][p].nnz.ss == plans[1][p].nnz.ss);
            assert(plans[0][p].nnz.sd == plans[1][p].nnz.sd);
            assert(plans[0][p].nnz.ds == plans[1][p].nnz.ds);
            assert(plans[0][p].p.dd == plans[1][p].p.dd);
        }
    }
}

// Key k is in S rows k and k + 1 (mod N); R row i has keys a and a + 1. So
// every row has 4 multiply-adds and 3 result columns: the sampled estimate is
// 3/4 of the row bound whatever rows it draws.
static void check_constant_ratio() {
    const int N = 40, K = 40, M = 500;
    csr_data R, S;
    R.rows = M;
    S.rows = N;
    std::vector<std::vector<int>> s_keys(N);
    for (int k = 0; k < K; ++k) {
        s_keys[k].push_back(k);
        s_keys[(k + 1) % N].push_back(k);
    }
    for (int j = 0; j < N; ++j) {
        std::sort(s_keys[j].begin(), s_keys[j].end());
        for (int k : s_keys[j]) {
            S.idx.push_back(k);
            S.val.push_back(1);
        }
        S.ptr.push_back((int) S.idx.size());
    }
    for (int i = 0; i < M; ++i) {
        const int a = (7 * i) % (K - 1);
        R.idx.insert(R.idx.end(), {a, a + 1});
        R.val.insert(R.val.end(), {1, 1});
        R.ptr.push_back((int) R.idx.size());
    }
    ss_matrix<double> Sm;
    build(S, K, 16, 16, Sm, 2);
    const auto f = cw::count_features(R.ptr.data(), R.idx.data(), M, Sm, N, K, 2);
    assert(f.flops == 4ULL * M && f.row_bound == 4ULL * M);
    const auto e = cw::estimate_entries(R.ptr.data(), R.idx.data(), M, Sm, N, K, f, 2, 1);
    assert(!e.exact && e.draws == cw::kMinDraws && e.long_rows == 0);
    assert(e.entries == 3.0 * M);
    assert(brute_entries(R, S, K) == 3ULL * M);
    const auto x = cw::estimate_entries(R.ptr.data(), R.idx.data(), M, Sm, N, K, f, 2);
    assert(x.exact && x.entries == 3.0 * M && x.rows == (unsigned long long) M);
}

// One R row with keys 0..47 and N = 128 S rows: rows 0..63 have keys 0..47,
// rows 64..127 keys 48..63. 3072 multiply-adds > 8 N, so a sampled count
// tests 64 columns, j = floor((t + offset) * 2): the 32 below 64 hit (one
// S entry read each), the 32 above miss (16 entries each). So 64 of 128
// columns, as the exact count has.
static void check_long_row() {
    const int N = 128, K = 64;
    csr_data R, S;
    R.rows = 1;
    S.rows = N;
    for (int k = 0; k < 48; ++k) {
        R.idx.push_back(k);
        R.val.push_back(1);
    }
    R.ptr.push_back(48);
    for (int j = 0; j < N; ++j) {
        for (int k = j < 64 ? 0 : 48; k < (j < 64 ? 48 : K); ++k) {
            S.idx.push_back(k);
            S.val.push_back(1);
        }
        S.ptr.push_back((int) S.idx.size());
    }
    ss_matrix<double> Sm;
    build(S, K, 16, 16, Sm, 2);
    const auto f = cw::count_features(R.ptr.data(), R.idx.data(), 1, Sm, N, K, 2);
    assert(f.flops == 3072 && f.row_bound == 128);
    const auto e = cw::estimate_entries(R.ptr.data(), R.idx.data(), 1, Sm, N, K, f, 2, 1);
    assert(!e.exact && e.draws == cw::kMinDraws && e.rows == 1 && e.long_rows == 1);
    assert(e.visits == 32 + 32 * 16 && e.entries == 64.0);
    const auto x = cw::estimate_entries(R.ptr.data(), R.idx.data(), 1, Sm, N, K, f, 2);
    assert(x.exact && x.long_rows == 0 && x.entries == 64.0);
    assert(brute_entries(R, S, K) == 64);
}

// slice_length_rule::packed_values() at each level (lengths are powers of
// two, as the slicing shifts keys), from the CSR counts,
// equals the packing schedule of the matrices sliced at that length: R
// slices x length x r_stride plus S slice packs x length x s_stride, for
// either side alone and both, several tile sizes and thread counts.
static void check_rule_packing() {
    namespace slr = slice_length_rule;
    std::mt19937 rng(19);
    struct shape {
        int M, N, K, base;
        std::vector<double> dr, ds;
        int band;
    };
    const std::vector<shape> shapes = {
        {300, 200, 1000, 64, {0.003, 0.01}, {0.002, 0.02}, 24},
        {500, 700, 900, 128, {0.004}, {0.01, 0.0}, 16},
        {120, 90, 700, 32, {0.05, 0.3}, {0.2}, 12},
    };
    for (const shape &sh : shapes) {
        csr_data R = random_csr(sh.M, sh.K, sh.dr, sh.band, rng);
        csr_data S = random_csr(sh.N, sh.K, sh.ds, sh.band, rng);
        const int levels = slr::levels_within(sh.base, sh.K);
        assert(levels == 3);
        const auto rc = slr::count_columns(R.ptr.data(), R.idx.data(), R.rows,
                                           sh.K, sh.base, levels, 12, 3);
        const auto sc = slr::count_columns(S.ptr.data(), S.idx.data(), S.rows,
                                           sh.K, sh.base, levels, 16, 3);
        for (int l = 0; l < levels; ++l) {
            const int sl = sh.base << l;
            ss_matrix<double> Rm, Sm;
            build(R, sh.K, sl, 12, Rm, 2);
            build(S, sh.K, sl, 16, Sm, 2);
            for (const auto &tiles : {std::pair{128, 2304}, std::pair{16, 24},
                                      std::pair{40, 60}}) {
                for (int threads : {1, 5, 30}) {
                    const auto d = cw::dense_schedule_counts(
                        Rm, Sm, tiles.first, tiles.second, threads);
                    for (int sides = 1; sides <= 3; ++sides) {
                        slr::packing p;
                        p.pack_r = sides & 1;
                        p.pack_s = sides & 2;
                        p.r_stride = p.pack_s ? 12 : 16;
                        p.tile_size_l2 = tiles.first;
                        p.tile_size_l3 = tiles.second;
                        p.threads = threads;
                        const double want =
                            (double) sl *
                            ((p.pack_r ? (double) d.r_packs * p.r_stride : 0.0) +
                             (p.pack_s ? (double) d.s_packs * p.s_stride : 0.0));
                        assert(slr::packed_values(rc, sc, l, p) == want);
                    }
                }
            }
        }
        // Nothing packed: no values.
        slr::packing none;
        assert(slr::packed_values(rc, sc, 0, none) == 0.0);
    }
}

int main() {
    assert(init_decision_thresholds("sparse", 1, "cpu"));
    check_random();
    check_constant_ratio();
    check_long_row();
    check_rule_packing();
    std::cout << "PASS: candidate_work counts against brute force (features, "
                 "result entries exact and sampled, a long row, work and "
                 "slice pairs per table, dense packing schedule), the same "
                 "for 1 and 3 threads; the slice-length rule's packed values "
                 "match that schedule\n";
}
