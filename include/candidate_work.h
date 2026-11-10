#ifndef CANDIDATE_WORK_H
#define CANDIDATE_WORK_H

// Diagnostics for comparing whole-call alternatives offline (conf key
// log_candidate_work = 1). Nothing here changes a decision or a result.
//
// - count_features(): exact counts from the prepared CSR. flops is the
//   number of multiply-adds of the product (the SS kernel's inner-loop
//   iterations when every pair is SS). row_bound is an upper bound on the
//   result entries: per R row, min(N, that row's multiply-adds).
// - nnz_work(): the cost model of mm_check() for a given ss_upper, with each
//   density bucket's actual stored entries in place of its representative
//   count (bucket_to_nnz). For bucket 0 that representative is 1, however
//   many entries the slices hold, so mm_check()'s SS cost can be far below
//   the multiply-adds.

#include <algorithm>
#include <vector>

#include "slice_decision.h"
#include "ss_mat.h"

namespace candidate_work {

struct features {
    unsigned long long flops = 0;
    unsigned long long row_bound = 0;
};

// R is M x K and S is N x K, both CSR on the join key.
inline features count_features(
    const int *Ri, const int *Rj, const int M, const int *Si, const int *Sj,
    const int N, const int K, const int threads
) {
    std::vector<int> s_count(K > 0 ? K : 0, 0);
    const int s_entries = Si[N];
    for (int p = 0; p < s_entries; ++p) ++s_count[Sj[p]];
    unsigned long long flops = 0, bound = 0;
    const unsigned long long n = N > 0 ? (unsigned long long) N : 0;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 256)           \
    reduction(+ : flops, bound)
    for (int i = 0; i < M; ++i) {
        unsigned long long row = 0;
        for (int p = Ri[i]; p < Ri[i + 1]; ++p)
            row += (unsigned long long) s_count[Rj[p]];
        flops += row;
        bound += std::min(row, n);
    }
    return {flops, bound};
}

struct work {
    double ss = 0, sd = 0, ds = 0, dd = 0;
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

// mm_check() over all slice columns for the table (ss_up, ri_upper,
// left_dense_threshold), with stored entries per bucket: SS = R entries x S
// entries / slice length per slice pair; SD = R entries x S rows; DS = S
// entries x R rows; DD as in mm_check() (it counts no entries).
template <typename T>
work nnz_work(
    const ss_matrix<T> &R, const ss_matrix<T> &S, const uint8_t *ss_up
) {
    const int nb = num_buckets;
    const int sl = R.sl, mr = R.sh, nr = S.sh;
    std::vector<double> rc(nb + 1), re(nb + 1), sc(nb + 1), se(nb + 1);
    // Prefix sums over S buckets: [0, b).
    std::vector<double> sc_pre(nb + 2), se_pre(nb + 2);
    work w;
    for (int col = 0; col < R.s_csc_mat.cols; ++col) {
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
        for (int i = 0; i <= nb; ++i) {
            if (rc[i] == 0) continue;
            const int up = std::min<int>(ss_up[i], nb + 1);
            const int ri = std::min<int>(ri_upper[i], nb + 1);
            w.ss += re[i] * se_pre[up] / sl;
            const double s_dense = s_slices - sc_pre[std::max(up, ri)];
            if (i < left_dense_threshold)
                w.sd += re[i] * s_dense * nr;
            else
                w.dd += rc[i] * (double) sl * mr * s_dense * nr;
            if (ri > up) w.ds += (se_pre[ri] - se_pre[up]) * mr * rc[i];
        }
    }
    return w;
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
