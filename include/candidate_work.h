#ifndef CANDIDATE_WORK_H
#define CANDIDATE_WORK_H

// Diagnostics for comparing whole-call alternatives offline (conf key
// log_candidate_work = 1). Nothing here changes a decision or a result.
//
// - count_features(): exact counts from the prepared CSR. flops is the
//   number of multiply-adds of the product (the SS kernel's inner-loop
//   iterations when every pair is SS). row_bound is an upper bound on the
//   result entries: per R row, min(N, that row's multiply-adds).
// - estimate_entries(): the result entries, exact when the product is small
//   and estimated from a sample of R rows otherwise, for a given budget of
//   S entries read.
// - plan_counts(): per decision table, the cost model of mm_check() with
//   each density bucket's actual stored entries in place of its
//   representative count (bucket_to_nnz), and the slice pairs per kernel.
//   For bucket 0 that representative is 1, however many entries the slices
//   hold, so mm_check()'s SS cost can be far below the multiply-adds.
// - dense_schedule_counts(): the slices dense_dense_matmul() packs.
//
// Sums that can round differently are added in a fixed order, so every
// count is the same for any thread count.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "slice_decision.h"
#include "ss_mat.h"

namespace candidate_work {

// Visit budgets of the two result-entry counts: the large one is exact on
// most calls and is the reference; the small one is what a decision could
// afford.
inline constexpr unsigned long long kEntriesBudget = 1ULL << 27;
inline constexpr unsigned long long kEntriesSmallBudget = 1ULL << 20;
// A drawn row with more than kRowVisitsPerColumn * N multiply-adds is
// "long": instead of visiting its S entries, kColumnTests of its result
// columns are tested (see estimate_entries()).
inline constexpr unsigned long long kRowVisitsPerColumn = 8;
inline constexpr int kColumnTests = 64;
inline constexpr int kMinDraws = 16, kMaxDraws = 4096;

struct features {
    unsigned long long flops = 0;
    unsigned long long row_bound = 0;
    // S entries per key.
    std::vector<int> s_count;
    // Per R row: min(N, the row's multiply-adds).
    std::vector<int> row_bounds;
    // Sum over R rows of the row bound times the work of counting that row
    // in a sample (see estimate_entries()): sizes the sample.
    double weighted_visits = 0;
};

// Rows of R (M x K) and S (N x K), both CSR on the join key; S also as
// its slices. S's key histogram goes by slice column, whose keys no other
// column has, so the columns can be counted in parallel.
template <typename T>
features count_features(
    const int *Ri, const int *Rj, const int M, const ss_matrix<T> &S,
    const int N, const int K, const int threads
) {
    features f;
    f.s_count.assign(K > 0 ? K : 0, 0);
    const int *Sj = S.csr_mat.col_idxs;
    int *count = f.s_count.data();
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (int col = 0; col < S.sr_csc_mat.cols; ++col) {
        for (int idx = S.sr_csc_mat.col_ptrs[col];
             idx < S.sr_csc_mat.col_ptrs[col + 1]; ++idx) {
            const subrow &sr = S.sr_csc_mat.vals[idx];
            for (int p = sr.start; p < sr.start + sr.len; ++p) ++count[Sj[p]];
        }
    }
    f.row_bounds.assign(M > 0 ? M : 0, 0);
    const unsigned long long n = N > 0 ? (unsigned long long) N : 0;
    const unsigned long long cap = kRowVisitsPerColumn * n;
    // A long row: its R keys, plus kColumnTests S rows of average length.
    const double tests =
        N > 0 ? (double) kColumnTests * S.csr_mat.row_ptrs[N] / N : 0.0;
    int *rb = f.row_bounds.data();
    // Fixed chunks of rows, added in order (weighted is a double).
    constexpr int kRows = 1024;
    const int chunks = M > 0 ? (M + kRows - 1) / kRows : 0;
    std::vector<unsigned long long> c_flops(chunks), c_bound(chunks);
    std::vector<double> c_weighted(chunks);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int c = 0; c < chunks; ++c) {
        unsigned long long flops = 0, bound = 0;
        double weighted = 0;
        const int end = std::min(M, (c + 1) * kRows);
        for (int i = c * kRows; i < end; ++i) {
            unsigned long long row = 0;
            for (int p = Ri[i]; p < Ri[i + 1]; ++p)
                row += (unsigned long long) count[Rj[p]];
            flops += row;
            const unsigned long long b = std::min(row, n);
            rb[i] = (int) b;
            bound += b;
            const double work =
                row <= cap ? (double) row : tests + (Ri[i + 1] - Ri[i]);
            weighted += (double) b * work;
        }
        c_flops[c] = flops;
        c_bound[c] = bound;
        c_weighted[c] = weighted;
    }
    for (int c = 0; c < chunks; ++c) {
        f.flops += c_flops[c];
        f.row_bound += c_bound[c];
        f.weighted_visits += c_weighted[c];
    }
    return f;
}

// A fixed pseudo-random 32-bit mix: the offset of a long row's column
// tests.
inline uint32_t mix32(const uint32_t v) {
    uint32_t x = v;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

struct entries_estimate {
    double entries = 0;
    bool exact = false;
    // Sample points, distinct R rows they fall on, long rows among them
    // (estimated from column tests), and S entries read.
    unsigned long long draws = 0, rows = 0, long_rows = 0, visits = 0;
};

// Result entries (distinct (i, j) with a common key).
//
// Exact when flops <= budget: every row is counted from its S entries.
// Otherwise R rows are drawn with probability proportional to their row
// bound b_i (systematic sampling over the cumulative bounds, n points at
// (k + 1/2) B / n, where B is row_bound), and
//   estimate = B / n * sum over points of d_i / b_i,
// d_i being the row's distinct result columns. d_i / b_i lies in [0, 1],
// so the error is at most B / (2 sqrt(n)) in standard deviation, whatever
// the row lengths. n = budget * B / weighted_visits (the expected work per
// point), clamped to [kMinDraws, kMaxDraws].
//
// d_i of a drawn row: its S rows per key are visited, and the distinct
// ones counted (stopping once all N are there). A long row (more than
// kRowVisitsPerColumn * N multiply-adds) instead tests kColumnTests
// columns j, systematically spaced from a hashed offset: whether S row j
// has a key of row i (a bitmap of row i's keys). Then d_i = N * hits /
// kColumnTests, an unbiased estimate; its binomial error averages out over
// the points.
//
// S's rows per key are gathered only for the keys the short drawn rows
// use: two passes over the S entries of the slice columns that hold such
// keys, split into pieces of subrows for all threads.
template <typename T>
entries_estimate estimate_entries(
    const int *Ri, const int *Rj, const int M, const ss_matrix<T> &S,
    const int N, const int K, const features &f, const int threads,
    const unsigned long long budget = kEntriesBudget
) {
    entries_estimate e;
    if (f.row_bound == 0 || M <= 0 || N <= 0 || K <= 0) {
        e.exact = true;
        return e;
    }
    e.exact = f.flops <= budget;
    const unsigned long long B = f.row_bound;
    unsigned long long n = 0;
    if (!e.exact) {
        const double per_point =
            f.weighted_visits > 0 ? f.weighted_visits / (double) B : 1.0;
        const double want = (double) budget / std::max(per_point, 1.0);
        n = (unsigned long long) std::clamp(
            want, (double) kMinDraws, (double) kMaxDraws
        );
    }

    // The drawn rows and their point counts, in row order: chunks of rows
    // in parallel, each starting from the bound sum before it.
    const int chunks = std::max(1, std::min(M, threads * 8));
    const int chunk_rows = (M + chunks - 1) / chunks;
    std::vector<unsigned long long> chunk_start(chunks + 1, 0);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int c = 0; c < chunks; ++c) {
        unsigned long long s = 0;
        const int end = std::min(M, (c + 1) * chunk_rows);
        for (int i = c * chunk_rows; i < end; ++i) s += f.row_bounds[i];
        chunk_start[c + 1] = s;
    }
    for (int c = 0; c < chunks; ++c) chunk_start[c + 1] += chunk_start[c];
    // Points k at (2k + 1) B / (2n): those below X / (2n) number
    // floor(ceil(X / B) / 2).
    auto points_below = [&](const unsigned long long cum) {
        const unsigned __int128 x = (unsigned __int128) 2 * n * cum;
        const unsigned __int128 q = (x + B - 1) / B;
        return (unsigned long long) (q / 2);
    };
    std::vector<std::vector<std::pair<int, int>>> picked(chunks);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int c = 0; c < chunks; ++c) {
        unsigned long long cum = chunk_start[c];
        const int end = std::min(M, (c + 1) * chunk_rows);
        for (int i = c * chunk_rows; i < end; ++i) {
            const int b = f.row_bounds[i];
            if (b == 0) continue;
            if (e.exact) {
                picked[c].push_back({i, 1});
            } else {
                const unsigned long long m =
                    points_below(cum + b) - points_below(cum);
                if (m) picked[c].push_back({i, (int) m});
            }
            cum += b;
        }
    }
    std::vector<std::pair<int, int>> rows;
    for (auto &p : picked) rows.insert(rows.end(), p.begin(), p.end());
    picked.clear();
    e.rows = rows.size();
    e.draws = e.exact ? rows.size() : n;

    // Long rows (none in the exact count); the short ones mark their keys.
    const unsigned long long cap =
        e.exact ? ~0ULL : kRowVisitsPerColumn * (unsigned long long) N;
    std::vector<uint8_t> is_long(rows.size(), 0);
    std::vector<uint8_t> key_used(K, 0);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
    for (size_t r = 0; r < rows.size(); ++r) {
        const int i = rows[r].first;
        unsigned long long flops = 0;
        for (int p = Ri[i]; p < Ri[i + 1]; ++p) flops += f.s_count[Rj[p]];
        is_long[r] = flops > cap;
        if (is_long[r]) continue;
        for (int p = Ri[i]; p < Ri[i + 1]; ++p)
            if (f.s_count[Rj[p]])
                std::atomic_ref<uint8_t>(key_used[Rj[p]])
                    .store(1, std::memory_order_relaxed);
    }

    // S rows per used key, in subrow order (the same for any thread count).
    // Slice column col owns keys [col sl, (col + 1) sl). Its subrows are
    // split into pieces so that all threads take part even with few
    // columns. Each piece counts its rows per used key of its column (one
    // compact slot per used key), the counts become write offsets column by
    // column, and each piece writes its rows. A column gets at most one
    // piece per 8 of its S entries per used key, so the counts are at most
    // about an eighth of S's entries. Arrays read only where written are
    // left uninitialized.
    const int *Si = S.csr_mat.row_ptrs, *Sj = S.csr_mat.col_idxs;
    const int sl = S.sl, cols = S.sr_csc_mat.cols;
    const int *cp = S.sr_csc_mat.col_ptrs;
    std::unique_ptr<int[]> start(new int[K]);
    // 1 + the slot of a used key in its column, else 0.
    std::unique_ptr<int[]> slot(new int[K]);
    std::vector<long long> col_rows(cols + 1, 0);
    std::vector<int> col_used(cols, 0), col_pieces(cols + 1, 0);
    const long long piece_subrows = std::max<long long>(
        2048, ((long long) cp[cols] + 16LL * threads - 1) / (16LL * threads)
    );
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int col = 0; col < cols; ++col) {
        long long rows_used = 0, entries = 0;
        int u = 0;
        const int end = std::min(K, (col + 1) * sl);
        for (int k = col * sl; k < end; ++k) {
            entries += f.s_count[k];
            if (key_used[k]) {
                slot[k] = ++u;
                rows_used += f.s_count[k];
            } else {
                slot[k] = 0;
            }
        }
        col_rows[col + 1] = rows_used;
        col_used[col] = u;
        if (u) {
            const long long by_size =
                (cp[col + 1] - cp[col] + piece_subrows - 1) / piece_subrows;
            col_pieces[col + 1] = (int) std::max<long long>(
                1, std::min<long long>(by_size, entries / (8LL * u))
            );
        }
    }
    for (int col = 0; col < cols; ++col) {
        col_rows[col + 1] += col_rows[col];
        col_pieces[col + 1] += col_pieces[col];
    }
    const int pieces = col_pieces[cols];
    std::vector<int> piece_col(pieces);
    std::vector<long long> piece_at(pieces + 1, 0);
    for (int col = 0; col < cols; ++col)
        for (int x = col_pieces[col]; x < col_pieces[col + 1]; ++x) {
            piece_col[x] = col;
            piece_at[x + 1] = piece_at[x] + col_used[col];
        }
    // Per piece and used key of its column: rows, then the write cursor.
    std::unique_ptr<int[]> at(new int[std::max<long long>(1, piece_at[pieces])]);
    auto subrows = [&](const int x, int &a, int &b) {
        const int col = piece_col[x];
        const long long n_sub = cp[col + 1] - cp[col];
        const long long p = col_pieces[col + 1] - col_pieces[col];
        const long long y = x - col_pieces[col];
        a = cp[col] + (int) (y * n_sub / p);
        b = cp[col] + (int) ((y + 1) * n_sub / p);
    };
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int x = 0; x < pieces; ++x) {
        int a, b;
        subrows(x, a, b);
        int *count = at.get() + piece_at[x];
        std::fill(count, count + col_used[piece_col[x]], 0);
        for (int idx = a; idx < b; ++idx) {
            const subrow &sr = S.sr_csc_mat.vals[idx];
            for (int p = sr.start; p < sr.start + sr.len; ++p)
                if (const int s = slot[Sj[p]]) ++count[s - 1];
        }
    }
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int col = 0; col < cols; ++col) {
        if (!col_used[col]) continue;
        long long pos = col_rows[col];
        const int end = std::min(K, (col + 1) * sl);
        for (int k = col * sl; k < end; ++k) {
            const int s = slot[k] - 1;
            if (s < 0) continue;
            start[k] = (int) pos;
            for (int x = col_pieces[col]; x < col_pieces[col + 1]; ++x) {
                int &c = at[piece_at[x] + s];
                const int n_rows = c;
                c = (int) pos;
                pos += n_rows;
            }
        }
    }
    std::unique_ptr<int[]> s_rows(new int[std::max<long long>(1, col_rows[cols])]);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int x = 0; x < pieces; ++x) {
        int a, b;
        subrows(x, a, b);
        int *cursor = at.get() + piece_at[x];
        for (int idx = a; idx < b; ++idx) {
            const subrow &sr = S.sr_csc_mat.vals[idx];
            const int row = S.sr_csc_mat.row_idxs[idx];
            for (int p = sr.start; p < sr.start + sr.len; ++p)
                if (const int s = slot[Sj[p]]) s_rows[cursor[s - 1]++] = row;
        }
    }

    // d_i per drawn row, with bitmaps per thread: result columns (short
    // rows) or R keys (long rows).
    std::vector<double> distinct(rows.size(), 0.0);
    unsigned long long visits = 0;
#pragma omp parallel num_threads(threads) reduction(+ : visits)
    {
        std::vector<uint64_t> seen((N + 63) / 64, 0);
        std::vector<uint64_t> keys;
        std::vector<int> touched;
#pragma omp for schedule(dynamic, 16)
        for (size_t r = 0; r < rows.size(); ++r) {
            const int i = rows[r].first;
            unsigned long long v = 0;
            if (is_long[r]) {
                if (keys.empty()) keys.assign((K + 63) / 64, 0);
                for (int p = Ri[i]; p < Ri[i + 1]; ++p)
                    keys[Rj[p] >> 6] |= 1ULL << (Rj[p] & 63);
                const double offset = mix32((uint32_t) i) / 4294967296.0;
                int hits = 0;
                for (int t = 0; t < kColumnTests; ++t) {
                    const int j = std::min(
                        N - 1, (int) ((t + offset) * N / kColumnTests)
                    );
                    for (int q = Si[j]; q < Si[j + 1]; ++q) {
                        ++v;
                        if (keys[Sj[q] >> 6] >> (Sj[q] & 63) & 1) {
                            ++hits;
                            break;
                        }
                    }
                }
                for (int p = Ri[i]; p < Ri[i + 1]; ++p)
                    keys[Rj[p] >> 6] = 0;
                distinct[r] = (double) N * hits / kColumnTests;
            } else {
                bool full = false;
                for (int p = Ri[i]; p < Ri[i + 1] && !full; ++p) {
                    const int k = Rj[p];
                    for (int q = start[k]; q < start[k] + f.s_count[k]; ++q) {
                        const int j = s_rows[q];
                        ++v;
                        uint64_t &w = seen[j >> 6];
                        const uint64_t bit = 1ULL << (j & 63);
                        if (w & bit) continue;
                        w |= bit;
                        touched.push_back(j);
                        if ((int) touched.size() == N) {
                            full = true;
                            break;
                        }
                    }
                }
                distinct[r] = (double) touched.size();
                for (const int j : touched) seen[j >> 6] &= ~(1ULL << (j & 63));
                touched.clear();
            }
            visits += v;
        }
    }
    e.visits = visits;
    double sum = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        e.long_rows += is_long[r];
        if (e.exact)
            sum += distinct[r];
        else
            sum += rows[r].second * distinct[r] / f.row_bounds[rows[r].first];
    }
    e.entries = e.exact ? sum : sum * (double) B / (double) n;
    return e;
}

struct work {
    double ss = 0, sd = 0, ds = 0, dd = 0;
};

// Slice pairs per kernel.
struct pairs {
    unsigned long long ss = 0, sd = 0, ds = 0, dd = 0;
};

struct plan {
    work nnz;
    pairs p;
};

// Slices and stored entries per density bucket in one slice column.
template <typename T>
void bucket_totals(
    const ss_matrix<T> &m, const int col, std::vector<double> &count,
    std::vector<double> &entries
) {
    std::fill(count.begin(), count.end(), 0.0);
    std::fill(entries.begin(), entries.end(), 0.0);
    for (int si = m.s_csc_mat.col_ptrs[col]; si < m.s_csc_mat.col_ptrs[col + 1];
         ++si) {
        const slice &s = m.s_csc_mat.vals[si];
        long long e = 0;
        for (int r = s.start; r < s.start + s.len; ++r)
            e += m.sr_csc_mat.vals[r].len;
        count[s.b] += 1;
        entries[s.b] += (double) e;
    }
}

// For each table ss_up (with the global ri_upper and left_dense_threshold),
// over all slice columns:
// - nnz: mm_check() with stored entries per bucket: SS = R entries x S
//   entries / slice length per slice pair; SD = R entries x S rows; DS = S
//   entries x R rows; DD as in mm_check() (it counts no entries);
// - p: the slice pairs of each kernel (get_alg() of the two buckets).
// Columns go in fixed chunks, added in order.
template <typename T>
std::vector<plan> plan_counts(
    const ss_matrix<T> &R, const ss_matrix<T> &S,
    const std::vector<const uint8_t *> &tables, const int threads
) {
    const int nb = num_buckets;
    const int sl = R.sl, mr = R.sh, nr = S.sh;
    const int cols = R.s_csc_mat.cols;
    const int nt = (int) tables.size();
    constexpr int kChunk = 64;
    const int chunks = (cols + kChunk - 1) / kChunk;
    std::vector<plan> part((size_t) chunks * nt);
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> rc(nb + 1), re(nb + 1), sc(nb + 1), se(nb + 1);
        // Prefix sums over S buckets: [0, b).
        std::vector<double> sc_pre(nb + 2, 0.0), se_pre(nb + 2, 0.0);
#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < chunks; ++c) {
            const int end = std::min(cols, (c + 1) * kChunk);
            for (int col = c * kChunk; col < end; ++col) {
                if (R.s_csc_mat.col_ptrs[col] == R.s_csc_mat.col_ptrs[col + 1] ||
                    S.s_csc_mat.col_ptrs[col] == S.s_csc_mat.col_ptrs[col + 1])
                    continue;
                bucket_totals(R, col, rc, re);
                bucket_totals(S, col, sc, se);
                for (int b = 0; b <= nb; ++b) {
                    sc_pre[b + 1] = sc_pre[b] + sc[b];
                    se_pre[b + 1] = se_pre[b] + se[b];
                }
                const double s_slices = sc_pre[nb + 1];
                for (int t = 0; t < nt; ++t) {
                    plan &out = part[(size_t) c * nt + t];
                    const uint8_t *ss_up = tables[t];
                    for (int i = 0; i <= nb; ++i) {
                        if (rc[i] == 0) continue;
                        const int up = std::min<int>(ss_up[i], nb + 1);
                        const int ri = std::min<int>(ri_upper[i], nb + 1);
                        out.nnz.ss += re[i] * se_pre[up] / sl;
                        out.p.ss += (unsigned long long) (rc[i] * sc_pre[up]);
                        const double s_dense =
                            s_slices - sc_pre[std::max(up, ri)];
                        if (i < left_dense_threshold) {
                            out.nnz.sd += re[i] * s_dense * nr;
                            out.p.sd += (unsigned long long) (rc[i] * s_dense);
                        } else {
                            out.nnz.dd += rc[i] * (double) sl * mr * s_dense * nr;
                            out.p.dd += (unsigned long long) (rc[i] * s_dense);
                        }
                        if (ri > up) {
                            out.nnz.ds += (se_pre[ri] - se_pre[up]) * mr * rc[i];
                            out.p.ds += (unsigned long long) (rc[i] *
                                                              (sc_pre[ri] -
                                                               sc_pre[up]));
                        }
                    }
                }
            }
        }
    }
    std::vector<plan> total(nt);
    for (int c = 0; c < chunks; ++c) {
        for (int t = 0; t < nt; ++t) {
            const plan &x = part[(size_t) c * nt + t];
            plan &y = total[t];
            y.nnz.ss += x.nnz.ss;
            y.nnz.sd += x.nnz.sd;
            y.nnz.ds += x.nnz.ds;
            y.nnz.dd += x.nnz.dd;
            y.p.ss += x.p.ss;
            y.p.sd += x.p.sd;
            y.p.ds += x.p.ds;
            y.p.dd += x.p.dd;
        }
    }
    return total;
}

// What dense_dense_matmul() would pack under the table ss_up (with the
// global ri_upper and left_dense_threshold): get_dense_packing_policy().
struct packing {
    bool pack_r = true, pack_s = true, bitmaps = true;
    int fixed_alg = -1;
    int r_stride = 0;
};

inline packing packing_for(
    const uint8_t *ss_up, const bool selective, const bool ds_aligned,
    const int mr
) {
    bool used[4] = {};
    for (int left = 0; left <= num_buckets; ++left) {
        for (int right = 0; right <= num_buckets; ++right) {
            int alg;
            if (right < ss_up[left])
                alg = SPARSE_SPARSE;
            else if (right < ri_upper[left])
                alg = DENSE_DENSE_RIGHT_INNER_KERNEL;
            else if (left >= left_dense_threshold)
                alg = DENSE_DENSE;
            else
                alg = DENSE_DENSE_LEFT_INNER_KERNEL;
            used[alg] = true;
        }
    }
    packing p;
    if (used[0] + used[1] + used[2] + used[3] == 1)
        for (int alg = 0; alg < 4; ++alg)
            if (used[alg]) p.fixed_alg = alg;
    if (selective) {
        p.pack_r = used[DENSE_DENSE] || used[DENSE_DENSE_RIGHT_INNER_KERNEL];
        p.pack_s = used[DENSE_DENSE] || used[DENSE_DENSE_LEFT_INNER_KERNEL];
        p.bitmaps = used[DENSE_DENSE];
    }
    p.r_stride = ds_aligned && p.pack_r && !p.bitmaps ? 16 : mr;
    return p;
}

// dense_dense_matmul()'s schedule, whatever the table: per slice column
// with slices on both sides, R slices go in blocks of R_slices_per_l3, each
// block in R tasks, and every (R task, S tile) packs its S tile. So each R
// slice is packed once, and each S slice once per R task of every block.
// tasks counts the (R task, S tile) pairs.
struct dense_schedule {
    unsigned long long r_packs = 0, r_pack_entries = 0;
    unsigned long long s_packs = 0, s_pack_entries = 0;
    unsigned long long tasks = 0;
};

template <typename T>
dense_schedule dense_schedule_counts(
    const ss_matrix<T> &R, const ss_matrix<T> &S, const int tile_size_l2,
    const int tile_size_l3, const int threads
) {
    dense_schedule d;
    const int cols = R.s_csc_mat.cols;
    int max_r = 0, max_s = 0;
    for (int col = 0; col < cols; ++col) {
        const int r = R.s_csc_mat.col_ptrs[col + 1] - R.s_csc_mat.col_ptrs[col];
        const int s = S.s_csc_mat.col_ptrs[col + 1] - S.s_csc_mat.col_ptrs[col];
        if (!r || !s) continue;
        max_r = std::max(max_r, r);
        max_s = std::max(max_s, s);
    }
    if (!max_r || !max_s) return d;
    auto div_up = [](long long a, long long b) { return (a + b - 1) / b; };
    const int s_per_tile =
        (int) std::min<long long>(div_up(tile_size_l2, S.sh), max_s);
    const int r_per_l3 =
        (int) std::min<long long>(div_up(tile_size_l3, R.sh), max_r);
    // Stored entries of the slices [a, b) of one column: their subrows are
    // consecutive in the subrow CSC.
    auto entries = [](const ss_matrix<T> &m, const int a, const int b) {
        unsigned long long e = 0;
        if (a >= b) return e;
        const int first = m.s_csc_mat.vals[a].start;
        const int last = m.s_csc_mat.vals[b - 1].start + m.s_csc_mat.vals[b - 1].len;
        for (int r = first; r < last; ++r) e += m.sr_csc_mat.vals[r].len;
        return e;
    };
    unsigned long long r_packs = 0, r_entries = 0, s_packs = 0, s_entries = 0,
                       tasks = 0;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)             \
    reduction(+ : r_packs, r_entries, s_packs, s_entries, tasks)
    for (int col = 0; col < cols; ++col) {
        const int r0 = R.s_csc_mat.col_ptrs[col], r1 = R.s_csc_mat.col_ptrs[col + 1];
        const int s0 = S.s_csc_mat.col_ptrs[col], s1 = S.s_csc_mat.col_ptrs[col + 1];
        if (r0 == r1 || s0 == s1) continue;
        const long long s_count = s1 - s0;
        const long long task_num = div_up(s_count, s_per_tile);
        const long long r_div = div_up(threads, task_num);
        const unsigned long long s_col_entries = entries(S, s0, s1);
        r_entries += entries(R, r0, r1);
        r_packs += (unsigned long long) (r1 - r0);
        for (int b = r0; b < r1; b += r_per_l3) {
            const long long len = std::min(r1, b + r_per_l3) - b;
            const long long per_it = div_up(len, r_div);
            const unsigned long long r_tasks = (unsigned long long) div_up(len, per_it);
            s_packs += r_tasks * (unsigned long long) s_count;
            s_entries += r_tasks * s_col_entries;
            tasks += r_tasks * (unsigned long long) task_num;
        }
    }
    d.r_packs = r_packs;
    d.r_pack_entries = r_entries;
    d.s_packs = s_packs;
    d.s_pack_entries = s_entries;
    d.tasks = tasks;
    return d;
}

// Slices per side and slice pairs that share a slice column.
template <typename T>
void slice_counts(
    const ss_matrix<T> &R, const ss_matrix<T> &S, unsigned long long &r_slices,
    unsigned long long &s_slices, unsigned long long &pairs
) {
    r_slices = s_slices = pairs = 0;
    for (int col = 0; col < R.s_csc_mat.cols; ++col) {
        const unsigned long long r =
            R.s_csc_mat.col_ptrs[col + 1] - R.s_csc_mat.col_ptrs[col];
        const unsigned long long s =
            S.s_csc_mat.col_ptrs[col + 1] - S.s_csc_mat.col_ptrs[col];
        r_slices += r;
        s_slices += s;
        pairs += r * s;
    }
}

} // namespace candidate_work

#endif
