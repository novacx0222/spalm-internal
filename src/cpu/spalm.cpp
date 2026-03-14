#include <algorithm>
#include "path_utils.h"
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "cpu/dense_dense_matmul.h"
#include "mkl.h"
#include "slice_decision.h"
#include "cpu/sparse_sparse_matmul.h"
#include "ss_mat.h"
#include "util.h"

// Used only as a dladdr anchor to locate this .so at runtime.
static void spalm_lib_anchor() {}

template <typename T>
void spalm_matmul(const int type, int* Ri, int* Rj, T* Rv, int* Si, int* Sj,
                   T* Sv, const int M, const int K, const int N,
                   const int slice_length, const int tile_size_l2,
                   const int tile_size_l3, bool sparse_result, T* ret,
                   int* reti, int* retj, T* retv, const int threads) {
  int R_ik_slice_height, S_ik_slice_height;
  if (type == 0) {
    R_ik_slice_height = 12;
    S_ik_slice_height = 32;
  } else if (type == 1) {
    R_ik_slice_height = 12;
    S_ik_slice_height = 16;
  }

  ss_matrix<T> R_ssm;
  ss_matrix<T> S_ssm;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end,
      total_start, e2e_start;
  e2e_start = std::chrono::high_resolution_clock::now();
  csr_to_ss_matrix(Ri, Rj, Rv, M, K, slice_length, R_ik_slice_height, R_ssm,
                   threads);
  csr_to_ss_matrix(Si, Sj, Sv, N, K, slice_length, S_ik_slice_height, S_ssm,
                   threads);

  end = std::chrono::high_resolution_clock::now();
  print_time("Total ss_matrix init time: ", end - e2e_start);

  total_start = std::chrono::high_resolution_clock::now();

  bool do_sparse_mm = false;
  bool do_dense_mm = false;
  unsigned long long sparse_mm_cost = 0;
  unsigned long long sparse_dense_mm_cost = 0;
  unsigned long long dense_sparse_mm_cost = 0;
  unsigned long long dense_dense_mm_cost = 0;
  const int num_slice_cols = R_ssm.s_csc_mat.cols;
  for (int slice_col = 0; slice_col < num_slice_cols; slice_col++) {
    unsigned long long sparse_mm_cost_col = 0;
    unsigned long long sparse_dense_mm_cost_col = 0;
    unsigned long long dense_sparse_mm_cost_col = 0;
    unsigned long long dense_dense_mm_cost_col = 0;
    mm_check(slice_col, R_ssm, S_ssm, sparse_mm_cost_col,
             sparse_dense_mm_cost_col, dense_sparse_mm_cost_col,
             dense_dense_mm_cost_col);
    sparse_mm_cost += sparse_mm_cost_col;
    sparse_dense_mm_cost += sparse_dense_mm_cost_col;
    dense_sparse_mm_cost += dense_sparse_mm_cost_col;
    dense_dense_mm_cost += dense_dense_mm_cost_col;
  }
  unsigned long long total_dense_mm_cost =
      sparse_dense_mm_cost + dense_sparse_mm_cost + dense_dense_mm_cost;
  end = std::chrono::high_resolution_clock::now();
  std::cout << "SPARSE MM cost: " << sparse_mm_cost << std::endl;
  std::cout << "SPARSE-DENSE MM cost: " << sparse_dense_mm_cost << std::endl;
  std::cout << "DENSE-SPARSE MM cost: " << dense_sparse_mm_cost << std::endl;
  std::cout << "DENSE-DENSE MM cost: " << dense_dense_mm_cost << std::endl;
  print_time("MM check time: ", end - total_start);

  if (total_dense_mm_cost) {
    do_dense_mm = true;
  }

  float sparse_ratio_threshold = 0.1;

  if (total_dense_mm_cost < dense_mm_threshold && total_dense_mm_cost) {
    do_dense_mm = false;
    do_sparse_mm = true;
  } else if (sparse_mm_cost < total_dense_mm_cost * sparse_ratio_threshold) {
    do_dense_mm = true;
    do_sparse_mm = false;
    set_to_all_dense();
  } else if (sparse_mm_cost > 0) {
    do_sparse_mm = true;
  }

  if (no_features) {
    std::cout << "No features, do everything without fallback" << std::endl;
    do_dense_mm = true;
    do_sparse_mm = true;
    fallback = false;
  }

  if (total_dense_mm_cost > (unsigned long long)M * K * N * 0.8 && fallback) {
    // Fallback to mkl
    std::cout << "Fallback: MKL dense-dense" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto R = std::make_unique<T[]>(M * K);
    auto S = std::make_unique<T[]>(N * K);
    end = std::chrono::high_resolution_clock::now();
    print_time("Mem allocation time for MKL: ", end - start);

    start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < M; i++) {
      for (int k = Ri[i]; k < Ri[i + 1]; k++) {
        R[i * K + Rj[k]] = Rv[k];
      }
    }

#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < N; i++) {
      for (int k = Si[i]; k < Si[i + 1]; k++) {
        S[i * K + Sj[k]] = Sv[k];
      }
    }

    void* handle = dlopen(
        resolve_path_near_lib(reinterpret_cast<void*>(&spalm_lib_anchor),
                              "run_baseline.so", "./bin/run_baseline.so")
            .c_str(),
        RTLD_LAZY);
    if (!handle) {
      std::cerr << "Cannot open library: " << dlerror() << '\n';
      return;
    }

    void (*run_baseline)(int, const char*, int*, int*, void*, void*, int*, int*,
                         void*, void*, void*, const int, const int, const int,
                         const int) =
        (void (*)(int, const char*, int*, int*, void*, void*, int*, int*, void*,
                  void*, void*, const int, const int, const int,
                  const int))dlsym(handle, "run_baseline");

    end = std::chrono::high_resolution_clock::now();
    print_time("Matrix initialization for MKL: ", end - start);

    start = std::chrono::high_resolution_clock::now();
    /*
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                (double*)R, K, (double*)S, K, 1.0, (double*)ret, N);
    */
    run_baseline(type, "MKL dense-dense", Ri, Rj, (void*)Rv, (void*)R.get(), Si,
                 Sj, (void*)Sv, (void*)S.get(), (void*)ret, M, K, N, threads);
    end = std::chrono::high_resolution_clock::now();
    print_time("Fallback MM: ", end - start);
    mkl_free_buffers();
  } else if (do_sparse_mm && !do_dense_mm && fallback) {
    std::cout << "Fallback: MKL sparse-sparse" << std::endl;
    void* handle = dlopen(
        resolve_path_near_lib(reinterpret_cast<void*>(&spalm_lib_anchor),
                              "run_baseline.so", "./bin/run_baseline.so")
            .c_str(),
        RTLD_LAZY);
    if (!handle) {
      std::cerr << "Cannot open library: " << dlerror() << '\n';
      return;
    }

    void (*run_baseline)(int, const char*, int*, int*, void*, void*, int*, int*,
                         void*, void*, void*, const int, const int, const int,
                         const int) =
        (void (*)(int, const char*, int*, int*, void*, void*, int*, int*, void*,
                  void*, void*, const int, const int, const int,
                  const int))dlsym(handle, "run_baseline");
    start = std::chrono::high_resolution_clock::now();
    run_baseline(type, "MKL sparse-sparse", Ri, Rj, (void*)Rv, NULL, Si, Sj,
                 (void*)Sv, NULL, (void*)ret, M, K, N, threads);
    end = std::chrono::high_resolution_clock::now();
    print_time("Fallback MM: ", end - start);
  } else {
    std::cout << std::endl;
    if (do_sparse_mm) {
      sparse_sparse_matmul(R_ssm, S_ssm, ret, sparse_result, threads,
                           !do_dense_mm);
    }

    if (do_dense_mm) {
      dense_dense_matmul(R_ssm, S_ssm, tile_size_l2, tile_size_l3, ret,
                         sparse_result, threads);
    }
  }

  end = std::chrono::high_resolution_clock::now();
  std::cout << std::endl;
  print_time("Total MM time: ", end - total_start);
  print_time("E2E MM time: ", end - e2e_start);
}

extern "C" void spalm(int type, int* Ri, int* Rj, void* Rv, int* Si,
                             int* Sj, void* Sv, const int M, const int K,
                             const int N, bool sparse_result,
                             void* ret, int* reti, int* retj, void* retv,
                             int n, const char** keys, const char** vals) {
  auto get_str = [&](const char* key, const char* def) -> const char* {
    for (int i = 0; i < n; i++)
      if (strcmp(keys[i], key) == 0) return vals[i];
    return def;
  };
  auto get_int = [&](const char* key, int def) -> int {
    const char* v = get_str(key, nullptr);
    return v ? atoi(v) : def;
  };

  const char* init_method   = get_str("init_method",   "");
  const char* bucket_method = get_str("bucket_method", "");
  int threads               = get_int("threads",        4);

  std::cout << "SPALM MATMUL" << std::endl;
  std::cout << M << " " << K << " " << N << std::endl;
  std::cout << "THREADS: " << threads << std::endl;
  if (type == 0) {
    std::cout << "FP32" << std::endl;
  } else if (type == 1) {
    std::cout << "FP64" << std::endl;
  }

  parse_bucket_method(bucket_method);
  bool read_success = init_decision_thresholds(init_method, type, "cpu");
  if (!read_success) {
    std::cout << "Initialize decision thresholds." << std::endl;
    return;
  }
  int _sh, slice_length, tile_size_l2, tile_size_l3;
  read_success = read_tile_sizes(type, "cpu", _sh, slice_length,
                                 tile_size_l2, tile_size_l3);
  if (!read_success) {
    std::cout << "Initialize tile sizes." << std::endl;
    return;
  }
  slice_length = get_int("slice_length", slice_length);
  tile_size_l2 = get_int("tile_size_l2", tile_size_l2);
  tile_size_l3 = get_int("tile_size_l3", tile_size_l3);

  std::cout << "Slice length: " << slice_length << std::endl;
  /*
  if (slice_length < BITS) {
    std::cout << "Slice length must be larger than " << BITS << std::endl;
    return;
  }
  */
  if ((slice_length & (slice_length - 1)) != 0) {
    std::cout << "Slice length must be power of 2" << std::endl;
    return;
  }
  // std::cout << "Tile size join dim: " << tile_size_join_dim << std::endl;
  std::cout << "Tile size L2: " << tile_size_l2 << std::endl;
  std::cout << "Tile size L3: " << tile_size_l3 << std::endl;
  if (sparse_result) {
    std::cout << "Writing to sparse result" << std::endl;
  } else {
    std::cout << "Writing to dense result" << std::endl;
  }
  std::cout << std::endl;

  if (type == 0) {
    spalm_matmul(type, Ri, Rj, (float*)Rv, Si, Sj, (float*)Sv, M, K, N,
                  slice_length, tile_size_l2, tile_size_l3, sparse_result,
                  (float*)ret, reti, retj, (float*)retv, threads);
  } else if (type == 1) {
    spalm_matmul(type, Ri, Rj, (double*)Rv, Si, Sj, (double*)Sv, M, K, N,
                  slice_length, tile_size_l2, tile_size_l3, sparse_result,
                  (double*)ret, reti, retj, (double*)retv, threads);
  }

  free_bucket_info();
}