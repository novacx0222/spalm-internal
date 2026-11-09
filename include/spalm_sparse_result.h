#pragma once

// Result of spalm_sparse() (C ABI): C = R * S^T as row blocks in CSR form,
// instead of the dense M x N result of spalm(). Only init_method sparse
// (SPALM SS) and FORCE_MKL_SPARSE (mkl_sparse_?_sp2m) produce it.
//
// Block b holds rows row_begin .. row_begin + rows - 1 (the row numbers of
// R as passed): row row_begin + r has the entries [starts[r], ends[r]) of
// cols (column numbers of S as passed) and vals (float or double, as the
// call's type). One row can be split over several blocks, but a column
// appears at most once per row in the whole result. Entries can be 0: the
// SS path never stores one, MKL keeps products that cancel to 0. A caller
// that wants the dense result's rows drops them as that output does
// (`value != 0`: -0.0 dropped, NaN kept).
//
// The arrays stay valid until spalm_sparse_result_free().

#define SPALM_SPARSE_RESULT_ABI 1

#ifdef __cplusplus
#include <memory>
#include <vector>
extern "C" {
#endif

typedef struct spalm_csr_block {
    int row_begin;
    int rows;
    const int *starts;
    const int *ends;
    const int *cols;
    const void *vals;
} spalm_csr_block;

#ifdef __cplusplus
}

// Behind the opaque handle of spalm_sparse(): the blocks and whatever owns
// their arrays (the SS pages, or MKL's result matrix).
struct spalm_sparse_result {
    std::vector<spalm_csr_block> blocks;
    // Entries over all blocks (MKL: explicit zeros included).
    long long entries = 0;
    std::shared_ptr<void> storage;
};
#endif
