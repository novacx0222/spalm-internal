#ifndef KERNEL_H
#define KERNEL_H

#include <immintrin.h>

#include "ss_mat.h"

void __attribute__((noinline)) inner_kernel_bm(
    double *__restrict__ R, BIT_DTYPE *rbm, double *__restrict__ S,
    BIT_DTYPE *sbm, const int panel_size, const int R_sh, const int S_sh,
    const slice &R_slice_info, const int *R_row_idxs, const slice &S_slice_info,
    const int *S_row_idxs, const int M, const int N, double *__restrict__ ret,
    bool sparse_result, int *csr_agg_idxs, const int row_offset
);

void __attribute__((noinline)) inner_kernel_bm(
    float *__restrict__ R, BIT_DTYPE *rbm, float *__restrict__ S,
    BIT_DTYPE *sbm, const int panel_size, const int R_sh, const int S_sh,
    const slice &R_slice_info, const int *R_row_idxs, const slice &S_slice_info,
    const int *S_row_idxs, const int M, const int N, float *__restrict__ ret,
    bool sparse_result, int *csr_agg_idxs, const int row_offset
);

#endif
