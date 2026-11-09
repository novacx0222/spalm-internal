#include <iostream>
#include <string>
#include <vector>
#include "path_utils.h"

#include "mkl.h"
#include "util.h"

// Writes C = R * S^T into ret, overwriting it: the GEMM and sparse x dense
// calls use beta = 0 and mkl_sparse_?_spmmd computes C := A * B, so ret
// need not be zeroed.
extern "C" void run_baseline(
    int type, const char *baseline_name, int *Ri, int *Rj, void *Rv, void *R,
    int *Si, int *Sj, void *Sv, void *S, void *ret, const int M, const int K,
    const int N, const int threads
) {
    std::cout << baseline_name << std::endl;
    std::cout << M << " " << K << " " << N << std::endl;
    std::cout << "THREADS: " << threads << std::endl;

    mkl_set_num_threads(threads);

    std::chrono::time_point<std::chrono::high_resolution_clock> total_start,
        start, end, e2e_start;
    e2e_start = std::chrono::high_resolution_clock::now();

    std::string baseline_name_str = baseline_name;

    if (baseline_name_str == "MKL sparse-sparse") {
        sparse_matrix_t R_csr;
        sparse_matrix_t S_csr;
        if (type == 1) {
            mkl_sparse_d_create_csr(
                &R_csr, SPARSE_INDEX_BASE_ZERO, M, K, Ri, Ri + 1, Rj,
                (double *) Rv
            );
            mkl_sparse_d_create_csr(
                &S_csr, SPARSE_INDEX_BASE_ZERO, N, K, Si, Si + 1, Sj,
                (double *) Sv
            );
        } else if (type == 0) {
            mkl_sparse_s_create_csr(
                &R_csr, SPARSE_INDEX_BASE_ZERO, M, K, Ri, Ri + 1, Rj,
                (float *) Rv
            );
            mkl_sparse_s_create_csr(
                &S_csr, SPARSE_INDEX_BASE_ZERO, N, K, Si, Si + 1, Sj,
                (float *) Sv
            );
        }

        sparse_matrix_t S_T_csr;
        auto total_start = std::chrono::high_resolution_clock::now();
        mkl_sparse_convert_csr(S_csr, SPARSE_OPERATION_TRANSPOSE, &S_T_csr);
        end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-csr-transpose]", end - total_start);

        start = std::chrono::high_resolution_clock::now();
        if (type == 1) {
            mkl_sparse_d_spmmd(
                SPARSE_OPERATION_NON_TRANSPOSE, R_csr, S_T_csr,
                SPARSE_LAYOUT_ROW_MAJOR, (double *) ret, N
            );
        } else if (type == 0) {
            mkl_sparse_s_spmmd(
                SPARSE_OPERATION_NON_TRANSPOSE, R_csr, S_T_csr,
                SPARSE_LAYOUT_ROW_MAJOR, (float *) ret, N
            );
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-sparse-mm-time]", end - start);
        print_time("[mkl-total-sparse-mm]", end - total_start);

        const auto release_begin = std::chrono::high_resolution_clock::now();
        mkl_sparse_destroy(R_csr);
        mkl_sparse_destroy(S_csr);
        mkl_sparse_destroy(S_T_csr);
        print_time(
            "[mkl-sparse-deallocate]",
            std::chrono::high_resolution_clock::now() - release_begin
        );
    } else if (baseline_name_str == "MKL block sparse-sparse") {
        const int block_size = 8;
        sparse_matrix_t R_csr;
        sparse_matrix_t S_csr;
        if (type == 1) {
            mkl_sparse_d_create_csr(
                &R_csr, SPARSE_INDEX_BASE_ZERO, M, K, Ri, Ri + 1, Rj,
                (double *) Rv
            );
            mkl_sparse_d_create_csr(
                &S_csr, SPARSE_INDEX_BASE_ZERO, N, K, Si, Si + 1, Sj,
                (double *) Sv
            );
        } else if (type == 0) {
            mkl_sparse_s_create_csr(
                &R_csr, SPARSE_INDEX_BASE_ZERO, M, K, Ri, Ri + 1, Rj,
                (float *) Rv
            );
            mkl_sparse_s_create_csr(
                &S_csr, SPARSE_INDEX_BASE_ZERO, N, K, Si, Si + 1, Sj,
                (float *) Sv
            );
        }

        sparse_matrix_t R_bsr;
        sparse_matrix_t S_T_bsr;
        auto total_start = std::chrono::high_resolution_clock::now();
        mkl_sparse_convert_bsr(
            R_csr, block_size, SPARSE_LAYOUT_ROW_MAJOR,
            SPARSE_OPERATION_NON_TRANSPOSE, &R_bsr
        );
        mkl_sparse_convert_bsr(
            S_csr, block_size, SPARSE_LAYOUT_ROW_MAJOR,
            SPARSE_OPERATION_TRANSPOSE, &S_T_bsr
        );
        end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-bsr-creation]", end - total_start);

        /*
    sparse_matrix_t S_T_bsr;
    start = std::chrono::high_resolution_clock::now();
    mkl_sparse_convert_bsr(S_bsr, block_size, SPARSE_LAYOUT_COLUMN_MAJOR,
    SPARSE_OPERATION_TRANSPOSE, &S_T_bsr); end =
    std::chrono::high_resolution_clock::now(); print_time("[mkl-bsr-transpose]", end
    - start);
    */

        start = std::chrono::high_resolution_clock::now();
        if (type == 1) {
            mkl_sparse_d_spmmd(
                SPARSE_OPERATION_NON_TRANSPOSE, R_bsr, S_T_bsr,
                SPARSE_LAYOUT_ROW_MAJOR, (double *) ret, N
            );
            /*
      mkl_sparse_d_spmmd(SPARSE_OPERATION_NON_TRANSPOSE, R_csr, S_csr,
                         SPARSE_LAYOUT_ROW_MAJOR, (double*)ret, N);
      */
        } else if (type == 0) {
            mkl_sparse_s_spmmd(
                SPARSE_OPERATION_NON_TRANSPOSE, R_bsr, S_T_bsr,
                SPARSE_LAYOUT_ROW_MAJOR, (float *) ret, N
            );
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-sparse-mm-time]", end - start);
        print_time("[mkl-total-sparse-mm]", end - total_start);

        const auto release_begin = std::chrono::high_resolution_clock::now();
        mkl_sparse_destroy(R_csr);
        mkl_sparse_destroy(S_csr);
        mkl_sparse_destroy(R_bsr);
        mkl_sparse_destroy(S_T_bsr);
        print_time(
            "[mkl-bsr-deallocate]",
            std::chrono::high_resolution_clock::now() - release_begin
        );
    } else if (baseline_name_str == "MKL sparse-dense") {
        if (type == 1) {
            start = std::chrono::high_resolution_clock::now();
            /*
      double* S_T = new double[K * N];
      for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
          S_T[i * N + j] = ((double*)S)[j * K + i];
        }
      }
      end = std::chrono::high_resolution_clock::now();
      print_time("[mkl-s-transpose]", end - start);
      */

            sparse_matrix_t R_csr;
            mkl_sparse_d_create_csr(
                &R_csr, SPARSE_INDEX_BASE_ZERO, M, K, Ri, Ri + 1, Rj,
                (double *) Rv
            );

            struct matrix_descr R_descr;
            R_descr.type = SPARSE_MATRIX_TYPE_GENERAL;

            start = std::chrono::high_resolution_clock::now();
            mkl_sparse_d_mm(
                SPARSE_OPERATION_NON_TRANSPOSE, 1, R_csr, R_descr,
                SPARSE_LAYOUT_ROW_MAJOR, (double *) S, N, N, 0, (double *) ret,
                N
            );
            end = std::chrono::high_resolution_clock::now();
            print_time("[mkl-sparse-dense-mm-time]", end - start);

            // delete[] S_T;
        } else if (type == 0) {
            start = std::chrono::high_resolution_clock::now();
            /*
      double* S_T = new double[K * N];
      for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
          S_T[i * N + j] = ((double*)S)[j * K + i];
        }
      }
      end = std::chrono::high_resolution_clock::now();
      print_time("[mkl-s-transpose]", end - start);
      */

            sparse_matrix_t R_csr;
            mkl_sparse_s_create_csr(
                &R_csr, SPARSE_INDEX_BASE_ZERO, M, K, Ri, Ri + 1, Rj,
                (float *) Rv
            );

            struct matrix_descr R_descr;
            R_descr.type = SPARSE_MATRIX_TYPE_GENERAL;

            start = std::chrono::high_resolution_clock::now();
            mkl_sparse_s_mm(
                SPARSE_OPERATION_NON_TRANSPOSE, 1, R_csr, R_descr,
                SPARSE_LAYOUT_ROW_MAJOR, (float *) S, N, N, 0, (float *) ret, N
            );
            end = std::chrono::high_resolution_clock::now();
            print_time("[mkl-sparse-dense-mm-time]", end - start);

            // delete[] S_T;
        }
    } else if (baseline_name_str == "OpenBLAS") {
        void *handle = dlopen(
            resolve_path_near_lib(
                reinterpret_cast<void *>(&run_baseline),
                "../OpenBLAS-0.3.30/libopenblas.so",
                "OpenBLAS-0.3.30/libopenblas.so"
            )
                .c_str(),
            RTLD_LAZY
        );
        void (*cblas_dgemm_openblas)(
            int, int, int, int, int, int, double, double *, int, double *, int,
            double, double *, int
        ) =
            (void (*)(
                int, int, int, int, int, int, double, double *, int, double *,
                int, double, double *, int
            )) dlsym(handle, "cblas_dgemm");
        void (*cblas_sgemm_openblas)(
            int, int, int, int, int, int, float, float *, int, float *, int,
            float, float *, int
        ) =
            (void (*)(
                int, int, int, int, int, int, float, float *, int, float *, int,
                float, float *, int
            )) dlsym(handle, "cblas_sgemm");

        void (*openblas_set_threads)(int) =
            (void (*)(int)) dlsym(handle, "openblas_set_num_threads");
        openblas_set_threads(threads);

        start = std::chrono::high_resolution_clock::now();
        if (type == 0) {
            cblas_sgemm_openblas(
                CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                (float *) R, K, (float *) S, K, 0.0, (float *) ret, N
            );
        } else if (type == 1) {
            cblas_dgemm_openblas(
                CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                (double *) R, K, (double *) S, K, 0.0, (double *) ret, N
            );
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-dense-mm-time]", end - start);
    } else if (baseline_name_str == "MKL dense-dense") {
        int mem, buf;
        mem = mkl_mem_stat(&buf);
        std::cout << mem << " " << buf << std::endl;
        start = std::chrono::high_resolution_clock::now();
        if (type == 0) {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                (float *) R, K, (float *) S, K, 0.0, (float *) ret, N
            );
        } else {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                (double *) R, K, (double *) S, K, 0.0, (double *) ret, N
            );
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-dense-mm-time]", end - start);
        mem = mkl_mem_stat(&buf);
        std::cout << mem << " " << buf << std::endl;
        const auto release_begin = std::chrono::high_resolution_clock::now();
        mkl_free_buffers();
        print_time(
            "[mkl-buffers-deallocate]",
            std::chrono::high_resolution_clock::now() - release_begin
        );
    }

    end = std::chrono::high_resolution_clock::now();
    print_time("[mkl-baseline-total]", end - e2e_start);
}

// C = R * S^T as a sparse matrix (MKL sparse-sparse with sparse output):
// the same S transpose as "MKL sparse-sparse", then mkl_sparse_?_sp2m
// instead of mkl_sparse_?_spmmd, so only the output differs. C's CSR arrays
// (zero-based; rows_start / rows_end per row, columns not necessarily
// sorted, products that cancel to 0 kept) are returned and owned by
// *handle until run_baseline_sparse_output_free(*handle). Returns 0, or the
// failing MKL status (nothing to free then).
extern "C" int run_baseline_sparse_output(
    int type, int *Ri, int *Rj, void *Rv, int *Si, int *Sj, void *Sv,
    const int M, const int K, const int N, const int threads, void **handle,
    int **rows_start, int **rows_end, int **cols, void **vals
) {
    static_assert(sizeof(MKL_INT) == sizeof(int), "LP64 MKL expected");
    std::cout << "MKL sparse-sparse (sparse output)" << std::endl;
    std::cout << M << " " << K << " " << N << std::endl;
    std::cout << "THREADS: " << threads << std::endl;
    mkl_set_num_threads(threads);
    *handle = nullptr;
    const auto e2e_start = std::chrono::high_resolution_clock::now();

    sparse_matrix_t R_csr = nullptr, S_csr = nullptr, S_T_csr = nullptr,
                    C = nullptr;
    auto release_inputs = [&] {
        if (R_csr) mkl_sparse_destroy(R_csr);
        if (S_csr) mkl_sparse_destroy(S_csr);
        if (S_T_csr) mkl_sparse_destroy(S_T_csr);
        R_csr = S_csr = S_T_csr = nullptr;
    };
    auto fail = [&](const char *what, sparse_status_t status) {
        std::cerr << "MKL " << what << " failed with status " << (int) status
                  << std::endl;
        release_inputs();
        if (C) mkl_sparse_destroy(C);
        return status != SPARSE_STATUS_SUCCESS ? (int) status : -1;
    };

    sparse_status_t status;
    if (type == 1) {
        status = mkl_sparse_d_create_csr(
            &R_csr, SPARSE_INDEX_BASE_ZERO, M, K, Ri, Ri + 1, Rj,
            (double *) Rv
        );
        if (status != SPARSE_STATUS_SUCCESS) return fail("create R", status);
        status = mkl_sparse_d_create_csr(
            &S_csr, SPARSE_INDEX_BASE_ZERO, N, K, Si, Si + 1, Sj,
            (double *) Sv
        );
    } else {
        status = mkl_sparse_s_create_csr(
            &R_csr, SPARSE_INDEX_BASE_ZERO, M, K, Ri, Ri + 1, Rj, (float *) Rv
        );
        if (status != SPARSE_STATUS_SUCCESS) return fail("create R", status);
        status = mkl_sparse_s_create_csr(
            &S_csr, SPARSE_INDEX_BASE_ZERO, N, K, Si, Si + 1, Sj, (float *) Sv
        );
    }
    if (status != SPARSE_STATUS_SUCCESS) return fail("create S", status);

    const auto total_start = std::chrono::high_resolution_clock::now();
    status =
        mkl_sparse_convert_csr(S_csr, SPARSE_OPERATION_TRANSPOSE, &S_T_csr);
    auto end = std::chrono::high_resolution_clock::now();
    print_time("[mkl-csr-transpose]", end - total_start);
    if (status != SPARSE_STATUS_SUCCESS) return fail("transpose S", status);

    const auto start = std::chrono::high_resolution_clock::now();
    matrix_descr general;
    general.type = SPARSE_MATRIX_TYPE_GENERAL;
    status = mkl_sparse_sp2m(
        SPARSE_OPERATION_NON_TRANSPOSE, general, R_csr,
        SPARSE_OPERATION_NON_TRANSPOSE, general, S_T_csr,
        SPARSE_STAGE_FULL_MULT, &C
    );
    end = std::chrono::high_resolution_clock::now();
    print_time("[mkl-sparse-mm-time]", end - start);
    print_time("[mkl-total-sparse-mm]", end - total_start);
    if (status != SPARSE_STATUS_SUCCESS) return fail("sp2m", status);

    sparse_index_base_t base;
    MKL_INT rows = 0, columns = 0;
    MKL_INT *starts = nullptr, *ends = nullptr, *col_indx = nullptr;
    if (type == 1) {
        double *values = nullptr;
        status = mkl_sparse_d_export_csr(
            C, &base, &rows, &columns, &starts, &ends, &col_indx, &values
        );
        *vals = values;
    } else {
        float *values = nullptr;
        status = mkl_sparse_s_export_csr(
            C, &base, &rows, &columns, &starts, &ends, &col_indx, &values
        );
        *vals = values;
    }
    if (status != SPARSE_STATUS_SUCCESS) return fail("export C", status);
    if (base != SPARSE_INDEX_BASE_ZERO || rows != M || columns != N) {
        std::cerr << "MKL sp2m result is " << rows << " x " << columns
                  << " (base " << (int) base << "), expected " << M << " x "
                  << N << std::endl;
        return fail("check C", SPARSE_STATUS_SUCCESS);
    }

    const auto release_begin = std::chrono::high_resolution_clock::now();
    release_inputs();
    print_time(
        "[mkl-sparse-deallocate]",
        std::chrono::high_resolution_clock::now() - release_begin
    );
    *handle = C;
    *rows_start = starts;
    *rows_end = ends;
    *cols = col_indx;
    print_time(
        "[mkl-baseline-total]",
        std::chrono::high_resolution_clock::now() - e2e_start
    );
    return 0;
}

extern "C" void run_baseline_sparse_output_free(void *handle) {
    if (handle) mkl_sparse_destroy((sparse_matrix_t) handle);
}
