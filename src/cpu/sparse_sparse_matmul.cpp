#include "cpu/sparse_sparse_matmul.h"

#include <immintrin.h>
#include <omp.h>
#include <sched.h> // For sched_getcpu()

#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

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

// Must be called by every thread of one `omp parallel` region, with
// thread_idx = omp_get_thread_num() and threads = omp_get_num_threads(): it
// synchronizes with barriers. thread_col_offsets is shared by the team
// (it used to be a process-global static, unsafe for concurrent calls).
template <typename T>
inline void transpose_s_matrix_vanilla(
    const int thread_idx, const int threads, ss_matrix<T> &S_ssm,
    std::unique_ptr<int[]> &s_col_idxs_ref,
    std::unique_ptr<int[]> &s_agg_idxs_ref, std::unique_ptr<T[]> &s_vals_ref,
    std::unique_ptr<int[]> &thread_col_offsets
) {
    csr<T> &S_csr = S_ssm.csr_mat;
    const int N = S_csr.rows;
    const int K = S_csr.cols;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    const int rows_per_thread = div_ceil(N, threads);

    if (thread_idx == 0) {
        std::cout << "Vanilla CSR transpose" << std::endl;
        start = std::chrono::high_resolution_clock::now();

        const int nnz = S_csr.row_ptrs[N];
        s_col_idxs_ref = std::make_unique<int[]>(K + 1);
        s_agg_idxs_ref = std::make_unique_for_overwrite<int[]>(nnz);
        s_vals_ref = std::make_unique_for_overwrite<T[]>(nnz);

        thread_col_offsets =
            std::make_unique_for_overwrite<int[]>((size_t) threads * K);

        end = std::chrono::high_resolution_clock::now();
        print_time("Memory allocation time: ", end - start);
    }

#pragma omp barrier

    int *s_col_idxs = s_col_idxs_ref.get();
    int *s_agg_idxs = s_agg_idxs_ref.get();
    int *thread_col_offset = &thread_col_offsets[(size_t) K * thread_idx];
    T *s_vals = s_vals_ref.get();

    if (thread_idx == 0) {
        for (int t = 0; t < threads; t++) {
            for (int s_ait = rows_per_thread * t;
                 s_ait < std::min(rows_per_thread * (t + 1), N); s_ait++) {
                for (int s_kit = S_csr.row_ptrs[s_ait];
                     s_kit < S_csr.row_ptrs[s_ait + 1]; s_kit++) {
                    const int sk = S_csr.col_idxs[s_kit];
                    s_col_idxs[sk + 1]++;
                }
            }
            memcpy(
                &thread_col_offsets[(size_t) t * K], &s_col_idxs[1],
                sizeof(int) * K
            );
        }
    }

#pragma omp barrier

    if (thread_idx == 0) {
        for (int k = 0; k < K; k++) {
            s_col_idxs[k + 1] += s_col_idxs[k];
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("Stuff up to counting: ", end - start);
    }

#pragma omp barrier

    const int row_start = rows_per_thread * thread_idx;
    const int row_end = std::min(rows_per_thread * (thread_idx + 1), N);

    for (int s_ait = row_end - 1; s_ait >= row_start; s_ait--) {
        for (int s_kit = S_csr.row_ptrs[s_ait];
             s_kit < S_csr.row_ptrs[s_ait + 1]; s_kit++) {
            const int sk = S_csr.col_idxs[s_kit];
            const T sval = S_csr.vals[s_kit];
            const int write_idx = s_col_idxs[sk] + thread_col_offset[sk] - 1;
            thread_col_offset[sk]--;
            s_vals[write_idx] = sval;
            s_agg_idxs[write_idx] = s_ait;
        }
    }

#pragma omp barrier

    if (thread_idx == 0) {
        thread_col_offsets.reset();
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
    const int K = R_ssm.csr_mat.cols;
    const int N = S_ssm.csr_mat.rows;

    std::unique_ptr<int[]> S_T_col_idxs;
    std::unique_ptr<int[]> S_T_row_idxs;
    std::unique_ptr<T[]> S_T_vals;

    if (all_sparse) {
        // A parallel region, not a worksharing loop: the transpose uses
        // barriers, which are invalid inside `omp for` and deadlock or skip
        // rows when OpenMP gives fewer threads than requested.
        std::unique_ptr<int[]> thread_col_offsets;
#pragma omp parallel num_threads(threads)
        {
            transpose_s_matrix_vanilla(
                omp_get_thread_num(), omp_get_num_threads(), S_ssm,
                S_T_col_idxs, S_T_row_idxs, S_T_vals, thread_col_offsets
            );
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("S transpose time: ", end - sparse_matmul_start);

        start = std::chrono::high_resolution_clock::now();
        spgemm_rows_dense_result_vanilla(
            N, R_ssm, S_T_col_idxs.get(), S_T_row_idxs.get(), S_T_vals.get(),
            ret, threads
        );
        end = std::chrono::high_resolution_clock::now();
        print_time("Sparse MM time: ", end - start);
        print_time("Total Sparse MM time: ", end - sparse_matmul_start);
        return;
    }

    const int sl = R_ssm.sl;
    const int S_sh = S_ssm.sh;

    auto transpose_start = std::chrono::high_resolution_clock::now();
    const int total_buckets = get_total_sparse_buckets();
    std::vector<int> sj_buckets_tmp((size_t) K * total_buckets + 1);
    S_T_col_idxs =
        std::make_unique_for_overwrite<int[]>((size_t) (total_buckets + 1) * K);

    int *sj_buckets = S_T_col_idxs.get();

    // Count phase: iterate over S's slice CSC to count elements per (col, bucket)
    const int S_num_slice_cols = S_ssm.s_csc_mat.cols;
#pragma omp parallel for num_threads(threads)
    for (int col = 0; col < S_num_slice_cols; col++) {
        for (int si = S_ssm.s_csc_mat.col_ptrs[col];
             si < S_ssm.s_csc_mat.col_ptrs[col + 1]; si++) {
            const slice &s = S_ssm.s_csc_mat.vals[si];
            if (s.b >= total_buckets) {
                continue;
            }
            const int bucket_idx = s.b;
            // Iterate over subrows in this slice
            for (int sr_idx = s.start; sr_idx < s.start + s.len; sr_idx++) {
                const subrow &sr = S_ssm.sr_csc_mat.vals[sr_idx];
                // Iterate over elements in this subrow
                for (int k = sr.start; k < sr.start + sr.len; k++) {
                    sj_buckets_tmp
                        [(size_t) S_ssm.csr_mat.col_idxs[k] * total_buckets +
                         bucket_idx + 1]++;
                }
            }
        }
    }

    start = std::chrono::high_resolution_clock::now();
    int total_nnz = S_ssm.csr_mat.row_ptrs[N];

    for (size_t sji = 0; sji < (size_t) K * total_buckets; sji++) {
        sj_buckets_tmp[sji + 1] += sj_buckets_tmp[sji];
    }

    S_T_row_idxs = std::make_unique_for_overwrite<int[]>(total_nnz);
    S_T_vals = std::make_unique_for_overwrite<T[]>(total_nnz);

    T *s_vals = S_T_vals.get();
    int *s_agg_idxs = S_T_row_idxs.get();
    end = std::chrono::high_resolution_clock::now();
    print_time("S_T count + alloc: ", end - transpose_start);

    // Fill phase: iterate over S's slice CSC to fill transposed data
#pragma omp parallel for num_threads(threads)
    for (int col = 0; col < S_num_slice_cols; col++) {
        for (int si = S_ssm.s_csc_mat.col_ptrs[col];
             si < S_ssm.s_csc_mat.col_ptrs[col + 1]; si++) {
            const slice &s = S_ssm.s_csc_mat.vals[si];
            if (s.b >= total_buckets) {
                continue;
            }
            const int bucket_idx = s.b;
            // Iterate over subrows in this slice
            for (int sr_idx = s.start; sr_idx < s.start + s.len; sr_idx++) {
                const int row = S_ssm.sr_csc_mat.row_idxs[sr_idx];
                const subrow &sr = S_ssm.sr_csc_mat.vals[sr_idx];
                // Iterate over elements in this subrow
                for (int k = sr.start; k < sr.start + sr.len; k++) {
                    const int sk = S_ssm.csr_mat.col_idxs[k];
                    const T sval = S_ssm.csr_mat.vals[k];
                    const int write_idx = sj_buckets_tmp
                        [(size_t) sk * total_buckets + bucket_idx];

                    s_vals[write_idx] = sval;
                    s_agg_idxs[write_idx] = row;
                    sj_buckets_tmp[(size_t) sk * total_buckets + bucket_idx]++;
                }
            }
        }
    }

    start = std::chrono::high_resolution_clock::now();
    // Overlapping shift by one element: memmove, not memcpy (undefined).
    memmove(
        &sj_buckets_tmp[1], &sj_buckets_tmp[0],
        ((size_t) K * total_buckets) * sizeof(int)
    );
    sj_buckets_tmp[0] = 0;

    for (int b = 0; b < total_buckets + 1; b++) {
        for (int k = 0; k < K; k++) {
            sj_buckets[(size_t) b * K + k] =
                sj_buckets_tmp[(size_t) k * total_buckets + b];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    print_time("S_T copy + transpose: ", end - start);

    end = std::chrono::high_resolution_clock::now();
    print_time("S transpose time: ", end - transpose_start);

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
                const int thres = ss_upper[s.b];
                // Label each subrow in this slice
                for (int sr_idx = s.start; sr_idx < s.start + s.len; sr_idx++) {
                    // Map CSC subrow index back to CSR subrow index
                    const int csr_idx = R_ssm.sr_csc_to_csr_idx[sr_idx];
                    R_subrow_buckets[csr_idx] = thres;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("R subrow labeling time: ", end - start);

        spgemm_rows_dense_result(
            R_ssm, R_subrow_buckets.get(), S_T_row_idxs.get(), S_T_vals.get(),
            S_T_col_idxs.get(), N, ret, threads
        );
    } else {
        // TODO: sparse output path
    }
    end = std::chrono::high_resolution_clock::now();
    print_time("Sparse MM time: ", end - start);
    print_time("Total Sparse MM time: ", end - sparse_matmul_start);
}

template void sparse_sparse_matmul<float>(
    ss_matrix<float> &R_ssm, ss_matrix<float> &S_ssm, float *ret,
    bool sparse_output, const int threads, const bool all_sparse
);

template void sparse_sparse_matmul<double>(
    ss_matrix<double> &R_ssm, ss_matrix<double> &S_ssm, double *ret,
    bool sparse_output, const int threads, const bool all_sparse
);
