// Mode flags are reset per call, forced MKL modes need no thresholds file,
// slicing still borrows the caller's CSR, and machine_params.txt parsing.
#include "ss_mat.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>

static bool parse_text(const char *text, machine_params &params) {
    const std::string path = "/tmp/spalm-machine-params-test.txt";
    std::ofstream(path) << text;
    const bool ok = parse_machine_params(path, params);
    std::remove(path.c_str());
    return ok;
}

static void machine_params_tests() {
    machine_params params;
    // A missing file keeps the defaults (the values before the file existed).
    assert(parse_machine_params("/tmp/spalm-no-such-machine-params", params));
    assert(
        params.path.empty() && params.ignored.empty() &&
        params.result_base_offset_sparse == 16 &&
        params.result_base_offset_dense == 0
    );
    // Keys of other library versions are listed, not rejected.
    assert(parse_text(
        "# measured\n\nresult_base_offset_sparse: 0\n"
        "tile_size_l2: 64\nresult_base_offset_dense : 64\n"
        "slice_length: 1024\n",
        params
    ));
    assert(
        !params.path.empty() && params.result_base_offset_sparse == 0 &&
        params.result_base_offset_dense == 64 &&
        params.ignored == "tile_size_l2 slice_length"
    );
    for (const char *bad :
         {"result_base_offset_dense: 12\n", "result_base_offset_sparse: 4096\n",
          "result_base_offset_dense: -8\n", "result_base_offset_dense: 8 x\n",
          "result_base_offset_dense\n", "tile_size_l2: abc\n"}) {
        machine_params p;
        assert(!parse_text(bad, p));
    }
}

int main() {
    machine_params_tests();
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
                 "slice timings, machine parameters\n";
}
