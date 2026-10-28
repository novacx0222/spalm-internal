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
        params.path.empty() && params.slice_length == 0 &&
        params.tile_size_l2 == 0 && params.tile_size_l3 == 0 &&
        params.slice_target_entries == 16 && params.slice_length_max == 2048 &&
        params.result_base_offset_sparse == 16 &&
        params.result_base_offset_dense == 0
    );
    assert(parse_text(
        "# measured\n\nslice_length: 1024\ntile_size_l2: 256\n"
        "tile_size_l3 : 4608\ntile_size_l2_ds: 128\ntile_size_l3_sd: 576\n"
        "slice_target_entries_dd: 4\nslice_length_max_ds: 4096\n"
        "slice_target_entries: 8\nslice_length_max: 4096\n"
        "result_base_offset_sparse: 0\nresult_base_offset_dense: 64\n",
        params
    ));
    assert(
        !params.path.empty() && params.slice_length == 1024 &&
        params.tile_size_l2 == 256 &&
        params.method_tile_size_l2[MACHINE_DS] == 128 &&
        params.method_tile_size_l2[MACHINE_SD] == 0 &&
        params.method_tile_size_l3[MACHINE_SD] == 576 &&
        params.method_slice_target_entries[MACHINE_DD] == 4 &&
        params.method_slice_target_entries[MACHINE_SD] == 0 &&
        params.method_slice_length_max[MACHINE_DS] == 4096 &&
        params.tile_size_l3 == 4608 && params.slice_target_entries == 8 &&
        params.slice_length_max == 4096 &&
        params.result_base_offset_sparse == 0 &&
        params.result_base_offset_dense == 64
    );
    for (const char *bad :
         {"tile_size_l2: 0\n", "slice_length_max: 3000\n",
          "slice_length: 768\n", "slice_length: 0\n", "tile_size_l2_dd: 0\n",
          "tile_size_l2_xx: 64\n", "slice_length_max_sd: 1000\n",
          "slice_target_entries_ds: 0\n", "slice_length_sd: 1024\n",
          "slice_target_entries: 0\n", "result_base_offset_dense: 12\n",
          "result_base_offset_sparse: 4096\n", "slice_length_max: 2048 x\n",
          "unknown_key: 1\n", "tile_size_l3\n", "tile_size_l3: abc\n"}) {
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
