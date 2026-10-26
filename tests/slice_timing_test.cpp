// Mode flags are reset per call, forced MKL modes need no thresholds file,
// and slicing still borrows the caller's CSR.
#include "ss_mat.h"

#include <cassert>
#include <iostream>

int main() {
    force_mkl_sparse = true;
    no_features = true;
    assert(init_decision_thresholds("sparse", 1, "cpu"));
    assert(!force_mkl_dense && !force_mkl_sparse && !no_features);
    force_mkl_dense = true;
    assert(init_decision_thresholds("dense_bm_inner_kernel", 1, "cpu"));
    assert(!force_mkl_dense && !force_mkl_sparse);
    // Forced MKL modes return before reading decision_thresholds.txt.
    assert(init_decision_thresholds("FORCE_MKL_DENSE", 1, "no-such-platform"));
    assert(force_mkl_dense && !force_mkl_sparse && fallback);
    assert(init_decision_thresholds("FORCE_MKL_SPARSE", 0, "no-such-platform"));
    assert(!force_mkl_dense && force_mkl_sparse && fallback);
    assert(init_decision_thresholds("sparse", 1, "cpu"));
    assert(!force_mkl_dense && !force_mkl_sparse);

    int offsets[] = {0, 2, 3};
    int columns[] = {0, 2, 1};
    double values[] = {1, 2, 3};
    ss_matrix<double> matrix;
    csr_to_ss_matrix(offsets, columns, values, 2, 4, 2, 2, matrix, 2, 1);
    ss_matrix<double> second;
    csr_to_ss_matrix(offsets, columns, values, 2, 4, 2, 2, second, 2, 2);
    assert(matrix.csr_mat.row_ptrs == offsets);
    assert(matrix.csr_mat.col_idxs == columns);
    assert(matrix.csr_mat.rows == 2 && matrix.csr_mat.cols == 4);
    assert(matrix.sr_csr_mat.row_ptrs[2] == 3);
    std::cout << "PASS: per-call mode reset, forced MKL without thresholds, "
                 "slice timings\n";
}
