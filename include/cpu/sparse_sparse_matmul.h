#ifndef SPARSE_SPARSE_MATMUL_H
#define SPARSE_SPARSE_MATMUL_H

#include <immintrin.h>

#include <memory>

#include "cpu/execution_options.h"
#include "ss_mat.h"

void sparse_sparse_mm_count_results(
    const int start, const int end, const int *Rj, const int *Sa, const int *Sj,
    const int *Sj_1, const int write_val, const __m512i &write_val_reg,
    int *scratch, int *result_cnt
);

template <typename T>
void sparse_sparse_mm_calculate_results(
    const int start, const int end, const int *Rj, const double *Rv,
    const int *Sa, const int *Sj, const int *Sj_1, const T *Sv, int *scratch,
    int *row_result_cnt, const int offset, csr<T> &result_csr
);

template <typename T>
void csr_csr_matmul_kernel(
    const int start, const int end, const int *Rj, const T *Rv, const int *Sa,
    const int *Sj, const int *Sj_1, const T *Sv, T *ret
);

void csr_csr_matmul_kernel(
    const int start, const int end, const int *Rj, const double *Rv,
    const int *Sa, const int *Sj, const int *Sj_1, const double *Sv,
    double *ret, bool contiguous = true
);

template <typename T>
void spgemm_rows_dense_result(
    ss_matrix<T> &R_ssm, int *R_subrow_buckets, int *s_agg_idxs, T *s_vals,
    int *sj_buckets, const int N, T *ret, int threads, bool contiguous = true
);

template <typename T>
void spgemm_rows_dense_result_vanilla(
    const int N, ss_matrix<T> &R_ssm, int *s_col_idxs, int *s_agg_idxs,
    T *s_vals, T *ret, int threads, bool contiguous = true,
    const int *block_starts = nullptr, const int blocks = 1
);

template <typename T>
void sparse_sparse_matmul(
    ss_matrix<T> &R_ssm, ss_matrix<T> &S_ssm, T *ret, bool sparse_output,
    const int threads, const bool all_sparse = false,
    const cpu_execution_options &options = cpu_execution_options()
);

#endif
