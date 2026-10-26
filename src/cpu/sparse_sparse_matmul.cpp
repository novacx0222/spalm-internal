#include "cpu/sparse_sparse_matmul.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include "csr_preparation.h"
#include "slice_decision.h"
#include "util.h"

void sparse_sparse_mm_count_results(
    const int start, const int end, const int *Rj, const int *Sa, const int *Sj,
    const int *Sj_1, const int write_val, const __m512i &write_val_reg,
    int *scratch, int *result_cnt
) {
    for (int r_kit = start; r_kit < end; r_kit++) {
        const int k = Rj[r_kit];

        const int iterations = Sj_1[k] - Sj[k];

        if (!iterations) {
            continue;
        }

        const int left = iterations & 0b1111;
        const int s_ait_start = Sj[k];
        for (int s_ait = 0; s_ait < iterations - left; s_ait += 16) {
            __m512i sagg = _mm512_loadu_si512(&Sa[s_ait + s_ait_start]);
            __m512i scratch_vals = _mm512_i32gather_epi32(sagg, scratch, 4);
            __mmask16 cmp_result =
                _mm512_cmplt_epi32_mask(write_val_reg, scratch_vals);
            (*result_cnt) += _mm_popcnt_u32(cmp_result);
            _mm512_mask_i32scatter_epi32(
                scratch, cmp_result, sagg, write_val_reg, 4
            );
        }

        __mmask16 mask = (__mmask16) ~0 >> (16 - left);
        const int s_ait = iterations - left;
        __m512i sagg = _mm512_maskz_loadu_epi32(mask, &Sa[s_ait + s_ait_start]);
        __m512i scratch_vals =
            _mm512_mask_i32gather_epi32(write_val_reg, mask, sagg, scratch, 4);
        __mmask16 cmp_result =
            _mm512_mask_cmplt_epi32_mask(mask, write_val_reg, scratch_vals);
        (*result_cnt) += _mm_popcnt_u32(cmp_result);
        _mm512_mask_i32scatter_epi32(
            scratch, mask & cmp_result, sagg, write_val_reg, 4
        );

        /*
    for (int s_ait = Sj[k]; s_ait < Sj_1[k]; s_ait++) {
            const int sagg = Sa[s_ait];
            if (write_val < scratch[sagg]) {
                    (*result_cnt)++;
                    scratch[sagg] = write_val;
            }
    }
    */
    }
}

template <typename T>
void sparse_sparse_mm_calculate_results(
    const int start, const int end, const int *Rj, const double *Rv,
    const int *Sa, const int *Sj, const int *Sj_1, const T *Sv, int *scratch,
    int *row_result_cnt, const int offset, csr<T> &result_csr
) {
    for (int r_kit = start; r_kit < end; r_kit++) {
        const int k = Rj[r_kit];

        const int iterations = Sj_1[k] - Sj[k];

        int *Sj_w_offset = &result_csr.join_idxs[offset];
        T *Sv_w_offset = &result_csr.vals[offset];

        if (!iterations) {
            continue;
        }

        for (int s_ait = Sj[k]; s_ait < Sj_1[k]; s_ait++) {
            const int sagg = Sa[s_ait];
            if (scratch[sagg] >= 0) {
                const int write_idx = scratch[sagg];
                Sv_w_offset[write_idx] += Rv[r_kit] * Sv[s_ait];
            } else {
                const int write_idx = (*row_result_cnt)++;
                scratch[sagg] = write_idx;
                Sj_w_offset[write_idx] = sagg;
                Sv_w_offset[write_idx] = Rv[r_kit] * Sv[s_ait];
            }
        }
    }
}

template <typename T>
void csr_csr_matmul_kernel(
    const int start, const int end, const int *Rj, const T *Rv, const int *Sa,
    const int *Sj, const int *Sj_1, const T *Sv, T *ret
) {
    for (int r_kit = start; r_kit < end; r_kit++) {
        const T rv = Rv[r_kit];
        const int k = Rj[r_kit];
        for (int s_ait = Sj[k]; s_ait < Sj_1[k]; s_ait++) {
            ret[Sa[s_ait]] += rv * Sv[s_ait];
        }
    }
}

void csr_csr_matmul_kernel(
    const int start, const int end, const int *Rj, const double *Rv,
    const int *Sa, const int *Sj, const int *Sj_1, const double *Sv, double *ret
) {
    for (int r_kit = start; r_kit < end; r_kit++) {
        const int k = Rj[r_kit];

        const int iterations = Sj_1[k] - Sj[k];

        __m128d rv_128 = _mm_set_sd(Rv[r_kit]);

        if (!iterations) {
            continue;
        } else if (!(iterations & 0xfffffff8)) {
            /*
      for (int s_ait = Sj[k]; s_ait < Sj_1[k]; s_ait++) {
        const int sagg = Sa[s_ait];
        __m128d res = _mm_fmadd_pd(_mm_loadu_pd(&Sv[s_ait]), rv_128,
                                   _mm_loadu_pd(&ret[sagg]));
        ret[sagg] = _mm_cvtsd_f64(res);
      }
      */
            const int s_ait_start = Sj[k];
            __m512d r_val = _mm512_broadcastsd_pd(rv_128);
            __mmask8 mask = (__mmask8) ~0 >> (8 - iterations);
            __m512d sv = _mm512_maskz_loadu_pd(mask, &Sv[s_ait_start]);
            __m512d subres = _mm512_mul_pd(r_val, sv);
            __m256i sagg =
                _mm256_maskz_loadu_epi32(mask, (__m256i *) &Sa[s_ait_start]);
            __m512d res = _mm512_mask_i32gather_pd(sv, mask, sagg, ret, 8);
            res = _mm512_add_pd(subres, res);
            _mm512_mask_i32scatter_pd(ret, mask, sagg, res, 8);
        } else {
            const double rv = Rv[r_kit];
            const int left = iterations & 0b111;
            const int s_ait_start = Sj[k];
            __m512d r_val = _mm512_broadcastsd_pd(rv_128);
            for (int s_ait = 0; s_ait < iterations - left; s_ait += 8) {
                __m512d sv = _mm512_loadu_pd(&Sv[s_ait + s_ait_start]);
                __m512d subres = _mm512_mul_pd(r_val, sv);

                __m512i sagg = _mm512_cvtepu32_epi64(
                    _mm256_loadu_si256((__m256i *) &Sa[s_ait + s_ait_start])
                );
                __m512d res = _mm512_i64gather_pd(sagg, ret, 8);
                res = _mm512_add_pd(subres, res);
                _mm512_i64scatter_pd(ret, sagg, res, 8);
            }
            __mmask8 mask = (__mmask8) ~0 >> (8 - left);
            const int s_ait = iterations - left;
            __m512d sv = _mm512_maskz_loadu_pd(mask, &Sv[s_ait + s_ait_start]);
            __m512d subres = _mm512_mul_pd(r_val, sv);
            __m256i sagg = _mm256_maskz_loadu_epi32(
                mask, (__m256i *) &Sa[s_ait + s_ait_start]
            );
            __m512d res = _mm512_mask_i32gather_pd(sv, mask, sagg, ret, 8);
            res = _mm512_add_pd(subres, res);
            _mm512_mask_i32scatter_pd(ret, mask, sagg, res, 8);
        }
    }
}

template <typename T>
void spgemm_rows_dense_result(
    ss_matrix<T> &R_ssm, int *R_subrow_buckets, int *s_agg_idxs, T *s_vals,
    int *sj_buckets, const int N, T *ret, int threads
) {
    const int M = R_ssm.csr_mat.rows;
    const int K = R_ssm.csr_mat.cols;

#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (int ra = 0; ra < M; ra++) {
        for (int si = R_ssm.sr_csr_mat.row_ptrs[ra];
             si < R_ssm.sr_csr_mat.row_ptrs[ra + 1]; si++) {
            const int bucket_idx = R_subrow_buckets[si];
            if (bucket_idx == 0) {
                continue;
            }
            const int *sj_starts = &sj_buckets[0];
            const int *sj_ends = &sj_buckets[(size_t) bucket_idx * K];
            const subrow &sr = R_ssm.sr_csr_mat.vals[si];
            csr_csr_matmul_kernel(
                sr.start, sr.start + sr.len, R_ssm.csr_mat.col_idxs,
                R_ssm.csr_mat.vals, s_agg_idxs, sj_starts, sj_ends, s_vals,
                &ret[(size_t) ra * N]
            );
        }
    }
}

template <typename T>
void spgemm_rows_dense_result_vanilla(
    const int N, ss_matrix<T> &R_ssm, int *s_col_idxs, int *s_agg_idxs,
    T *s_vals, T *ret, int threads
) {
    const int M = R_ssm.csr_mat.rows;
    csr<T> &R_csr = R_ssm.csr_mat;

#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (int ra = 0; ra < M; ra++) {
        csr_csr_matmul_kernel(
            R_csr.row_ptrs[ra], R_csr.row_ptrs[ra + 1], R_csr.col_idxs,
            R_csr.vals, s_agg_idxs, s_col_idxs, s_col_idxs + 1, s_vals,
            &ret[(size_t) ra * N]
        );
    }
}

template <typename T>
void sparse_sparse_matmul(
    ss_matrix<T> &R_ssm, ss_matrix<T> &S_ssm, T *ret, bool sparse_output,
    const int threads, const bool all_sparse
) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end,
        sparse_matmul_start;
    sparse_matmul_start = std::chrono::high_resolution_clock::now();
    const int M = R_ssm.csr_mat.rows;
    const int N = S_ssm.csr_mat.rows;

    if (all_sparse) {
        auto transposed = transpose_all_sparse(S_ssm, threads);
        end = std::chrono::high_resolution_clock::now();
        print_time(
            "[sparse-sparse-s-transpose-time]", end - sparse_matmul_start
        );
        start = std::chrono::high_resolution_clock::now();
        spgemm_rows_dense_result_vanilla(
            N, R_ssm, transposed.offsets.get(), transposed.row_idxs.get(),
            transposed.values.get(), ret, threads
        );
        end = std::chrono::high_resolution_clock::now();
        print_time("[sparse-sparse-sparse-mm-time]", end - start);
        print_time(
            "[sparse-sparse-total-sparse-mm-time]", end - sparse_matmul_start
        );
        return;
    }

    // Only entries of S slices with bucket < total_buckets can be multiplied
    // here; the dense path handles the rest.
    const int total_buckets = get_total_sparse_buckets();
    if (total_buckets <= 0) return;
    auto transposed = transpose_sparse_buckets(S_ssm, total_buckets, threads);
    end = std::chrono::high_resolution_clock::now();
    print_time("[sparse-sparse-s-transpose-time]", end - sparse_matmul_start);
    if (transposed.entries == 0) return;

    start = std::chrono::high_resolution_clock::now();
    if (!sparse_output) {
        // Label each subrow in R's sr_csr_mat with its sparse threshold bucket.
        // Use sr_csc_to_csr_idx to map from slice CSC subrows back to CSR subrows.
        const int total_sr_csr_subrows = R_ssm.sr_csr_mat.row_ptrs[M];
        auto R_subrow_buckets =
            std::make_unique_for_overwrite<int[]>(total_sr_csr_subrows);
        memset(R_subrow_buckets.get(), 0, sizeof(int) * total_sr_csr_subrows);

        const int R_num_slice_cols = R_ssm.s_csc_mat.cols;
        for (int col = 0; col < R_num_slice_cols; col++) {
            for (int si = R_ssm.s_csc_mat.col_ptrs[col];
                 si < R_ssm.s_csc_mat.col_ptrs[col + 1]; si++) {
                const slice &s = R_ssm.s_csc_mat.vals[si];
                // The transpose only has bucket starts up to total_buckets;
                // clamp so a non-monotone thresholds file cannot index past it.
                const int thres = std::min<int>(ss_upper[s.b], total_buckets);
                // Label each subrow in this slice
                for (int sr_idx = s.start; sr_idx < s.start + s.len; sr_idx++) {
                    // Map CSC subrow index back to CSR subrow index
                    const int csr_idx = R_ssm.sr_csc_to_csr_idx[sr_idx];
                    R_subrow_buckets[csr_idx] = thres;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("[sparse-sparse-r-subrow-labeling-time]", end - start);

        spgemm_rows_dense_result(
            R_ssm, R_subrow_buckets.get(), transposed.row_idxs.get(),
            transposed.values.get(), transposed.offsets.get(), N, ret, threads
        );
    } else {
        // TODO: sparse output path
    }
    end = std::chrono::high_resolution_clock::now();
    print_time("[sparse-sparse-sparse-mm-time]", end - start);
    print_time(
        "[sparse-sparse-total-sparse-mm-time]", end - sparse_matmul_start
    );
}

template void sparse_sparse_matmul<float>(
    ss_matrix<float> &R_ssm, ss_matrix<float> &S_ssm, float *ret,
    bool sparse_output, const int threads, const bool all_sparse
);

template void sparse_sparse_matmul<double>(
    ss_matrix<double> &R_ssm, ss_matrix<double> &S_ssm, double *ret,
    bool sparse_output, const int threads, const bool all_sparse
);
