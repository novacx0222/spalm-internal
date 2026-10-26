#include <iostream>
#include <string>
#include <vector>
#include "path_utils.h"

#include "mkl.h"
#include "util.h"

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
                SPARSE_LAYOUT_ROW_MAJOR, (double *) S, N, N, 1, (double *) ret,
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
                SPARSE_LAYOUT_ROW_MAJOR, (float *) S, N, N, 1, (float *) ret, N
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
                (float *) R, K, (float *) S, K, 1.0, (float *) ret, N
            );
        } else if (type == 1) {
            cblas_dgemm_openblas(
                CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                (double *) R, K, (double *) S, K, 1.0, (double *) ret, N
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
                (float *) R, K, (float *) S, K, 1.0, (float *) ret, N
            );
        } else {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                (double *) R, K, (double *) S, K, 1.0, (double *) ret, N
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
