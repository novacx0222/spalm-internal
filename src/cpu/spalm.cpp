#include <algorithm>
#include "path_utils.h"
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "cpu/dense_dense_matmul.h"
#include "cpu/execution_options.h"
#include "csr_preparation.h"
#include "method_dispatch.h"
#include "mkl.h"
#include "slice_decision.h"
#include "cpu/sparse_sparse_matmul.h"
#include "ss_mat.h"
#include "util.h"

// Return codes of spalm(). The output is valid only for SPALM_OK; previously
// every failure returned silently and left the caller's output untouched.
enum spalm_status : int {
    SPALM_OK = 0,
    SPALM_INVALID_ARGUMENT = 1, // bad type, dimensions, threads or options
    SPALM_CONFIG_ERROR = 2,     // missing/invalid threshold or tile files
    SPALM_BASELINE_ERROR = 3,   // run_baseline.so could not be loaded
    SPALM_INTERNAL_ERROR = 4,   // exception, e.g. out of memory
};

// Used only as a dladdr anchor to locate this .so at runtime.
static void spalm_lib_anchor() {}

using run_baseline_fn = void (*)(
    int, const char *, int *, int *, void *, void *, int *, int *, void *,
    void *, void *, const int, const int, const int, const int
);

// Returns nullptr (after printing why) if the MKL baseline cannot be loaded.
// The library stays loaded for the process lifetime (unloading MKL corrupts
// its thread pool); a successful lookup is cached, a failure is retried.
// Calls are serialized by spalm(), so the cache needs no further locking.
static run_baseline_fn load_run_baseline() {
    static run_baseline_fn cached = nullptr;
    if (cached) return cached;
    void *handle = dlopen(
        resolve_path_near_lib(
            reinterpret_cast<void *>(&spalm_lib_anchor), "run_baseline.so",
            "./bin/run_baseline.so"
        )
            .c_str(),
        RTLD_LAZY
    );
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << '\n';
        return nullptr;
    }
    cached = (run_baseline_fn) dlsym(handle, "run_baseline");
    if (!cached) std::cerr << "Cannot find run_baseline: " << dlerror() << '\n';
    return cached;
}

// Runs the whole multiplication with an MKL backend (forced or fallback).
template <typename T>
int run_mkl_backend(
    spalm_dispatch::Backend backend, int type, int *Ri, int *Rj, T *Rv, int *Si,
    int *Sj, T *Sv, int M, int K, int N, T *ret, int threads
) {
    const run_baseline_fn baseline = load_run_baseline();
    if (!baseline) return SPALM_BASELINE_ERROR;
    if (backend == spalm_dispatch::Backend::MklDense) {
        std::cout << "Fallback: MKL dense-dense" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        // Implicit zeros require zero-initialized dense inputs.
        auto R = std::make_unique<T[]>(checked_matrix_size(M, K));
        auto S = std::make_unique<T[]>(checked_matrix_size(N, K));
        auto end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-dense-input-allocate]", end - start);
        start = end;
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < M; ++i) {
            for (int p = Ri[i]; p < Ri[i + 1]; ++p)
                R[(size_t) i * K + Rj[p]] = Rv[p];
        }
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; ++i) {
            for (int p = Si[i]; p < Si[i + 1]; ++p)
                S[(size_t) i * K + Sj[p]] = Sv[p];
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-dense-input-fill]", end - start);
        start = end;
        baseline(
            type, "MKL dense-dense", Ri, Rj, (void *) Rv, (void *) R.get(), Si,
            Sj, (void *) Sv, (void *) S.get(), (void *) ret, M, K, N, threads
        );
        end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-call-total]", end - start);
        const auto release_begin = end;
        mkl_free_buffers();
        R.reset();
        S.reset();
        print_time(
            "[mkl-dense-input-deallocate]",
            std::chrono::high_resolution_clock::now() - release_begin
        );
    } else {
        std::cout << "Fallback: MKL sparse-sparse" << std::endl;
        const auto start = std::chrono::high_resolution_clock::now();
        baseline(
            type, "MKL sparse-sparse", Ri, Rj, (void *) Rv, nullptr, Si, Sj,
            (void *) Sv, nullptr, (void *) ret, M, K, N, threads
        );
        print_time(
            "[mkl-call-total]",
            std::chrono::high_resolution_clock::now() - start
        );
    }
    return SPALM_OK;
}

// Accumulates R * S^T into the caller's zero-initialized ret. Early exits
// for zero work leave ret untouched.
template <typename T>
int spalm_matmul(
    const int type, int *Ri, int *Rj, T *Rv, int *Si, int *Sj, T *Sv,
    const int M, const int K_input, const int N, const int slice_length,
    const int tile_size_l2, const int tile_size_l3, bool sparse_result, T *ret,
    int *reti, int *retj, T *retv, const int threads,
    const cpu_execution_options &options
) {
    // The join dimension every later stage uses; key compaction shrinks it.
    int K = K_input;
    // An empty dimension, or no stored entry on a side, contributes nothing:
    // no CSR copies, slices, packing buffers or MKL call are needed.
    if (M == 0 || N == 0 || K == 0 || Ri[M] == 0 || Si[N] == 0) {
        std::cout << "[backend] zero-work (empty input)" << std::endl;
        return SPALM_OK;
    }
    const auto prepare_begin = std::chrono::high_resolution_clock::now();
    csr_value_summary r_summary{Ri[M], Ri[M], true};
    csr_value_summary s_summary{Si[N], Si[N], true};
    if (options.prune_zeros) {
        r_summary = summarize_csr_values(Ri, Rv, M, threads);
        s_summary = summarize_csr_values(Si, Sv, N, threads);
    }
    // Explicit zeros are dropped only when both sides are finite: with NaN or
    // Inf present, 0 * Inf and 0 * NaN must still produce NaN.
    const bool remove_zeros =
        options.prune_zeros && r_summary.finite && s_summary.finite;
    if (remove_zeros && (!r_summary.nonzero || !s_summary.nonzero)) {
        std::cout << "[backend] zero-work (all-zero input)" << std::endl;
        print_time(
            "[csr-zero-prepare]",
            std::chrono::high_resolution_clock::now() - prepare_begin
        );
        return SPALM_OK;
    }
    // Join keys used by only one input match nothing: drop their entries and
    // renumber the common keys densely (key_domain), in the same copy as the
    // zero pruning. Skipped when every key is common, or when no entry would
    // be dropped and the number of slice columns would stay the same.
    key_domain keys;
    const char *key_result = "disabled";
    bool compact = false;
    if (options.compact_keys) {
        const auto domain_begin = std::chrono::high_resolution_clock::now();
        const bool built = build_key_domain(
            keys, Ri, Rj, Rv, M, Si, Sj, Sv, N, K, remove_zeros, threads
        );
        print_time(
            "[csr-key-domain]",
            std::chrono::high_resolution_clock::now() - domain_begin
        );
        if (!built) {
            key_result = "skipped (bitmap budget)";
        } else if (keys.common_keys == 0) {
            std::cout << "[csr-key-stats] K=" << K << "->0 no common keys"
                      << std::endl;
            std::cout << "[backend] zero-work (no common keys)" << std::endl;
            print_time(
                "[csr-zero-prepare]",
                std::chrono::high_resolution_clock::now() - prepare_begin
            );
            return SPALM_OK;
        } else if (keys.common_keys == K) {
            key_result = "unchanged (all keys common)";
        } else if (
            keys.same_key_sets && div_ceil(keys.common_keys, slice_length) ==
                                      div_ceil(K, slice_length)
        ) {
            key_result = "unchanged (nothing to drop, same slice columns)";
        } else {
            key_result = "compacted";
            compact = true;
        }
    }
    prepared_csr<T> r_input, s_input;
    r_input.initialize(
        Ri, Rj, Rv, M, K, r_summary, remove_zeros, threads,
        compact ? &keys : nullptr
    );
    s_input.initialize(
        Si, Sj, Sv, N, K, s_summary, remove_zeros, threads,
        compact ? &keys : nullptr
    );
    Ri = r_input.matrix.row_ptrs;
    Rj = r_input.matrix.col_idxs;
    Rv = r_input.matrix.vals;
    Si = s_input.matrix.row_ptrs;
    Sj = s_input.matrix.col_idxs;
    Sv = s_input.matrix.vals;
    K = r_input.matrix.cols;
    std::cout << "[csr-key-stats] K=" << K_input << "->" << K << " "
              << key_result << std::endl;
    std::cout << "[csr-zero-stats] R=" << r_summary.stored << "->" << Ri[M]
              << " S=" << s_summary.stored << "->" << Si[N] << std::endl;
    print_time(
        "[csr-zero-prepare]",
        std::chrono::high_resolution_clock::now() - prepare_begin
    );

    // Forced MKL backends need no slices or cost estimate.
    const auto forced_backend = spalm_dispatch::select(
        force_mkl_dense, force_mkl_sparse, false, false, false
    );
    if (forced_backend != spalm_dispatch::Backend::Internal) {
        std::cout << "[backend] "
                  << (forced_backend == spalm_dispatch::Backend::MklDense
                          ? "mkl-dense"
                          : "mkl-sparse")
                  << " (forced)" << std::endl;
        const auto start = std::chrono::high_resolution_clock::now();
        const int status = run_mkl_backend(
            forced_backend, type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, ret, threads
        );
        print_time(
            "[forced-backend-total]",
            std::chrono::high_resolution_clock::now() - start
        );
        return status;
    }

    // type is validated by spalm(): 0 = float, 1 = double.
    const int R_ik_slice_height = 12;
    const int S_ik_slice_height = type == 0 ? 32 : 16;

    ss_matrix<T> R_ssm;
    ss_matrix<T> S_ssm;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end,
        total_start, e2e_start;
    e2e_start = std::chrono::high_resolution_clock::now();
    csr_to_ss_matrix(
        Ri, Rj, Rv, M, K, slice_length, R_ik_slice_height, R_ssm, threads, 1
    );
    csr_to_ss_matrix(
        Si, Sj, Sv, N, K, slice_length, S_ik_slice_height, S_ssm, threads, 2
    );

    end = std::chrono::high_resolution_clock::now();
    print_time("[slice-prepare-total]", end - e2e_start);

    total_start = std::chrono::high_resolution_clock::now();

    bool do_sparse_mm = false;
    bool do_dense_mm = false;
    unsigned long long sparse_mm_cost = 0;
    unsigned long long sparse_dense_mm_cost = 0;
    unsigned long long dense_sparse_mm_cost = 0;
    unsigned long long dense_dense_mm_cost = 0;
    const int num_slice_cols = R_ssm.s_csc_mat.cols;
    // Whether any slice column (K range) is non-empty on both sides; if not,
    // no key matches and the product is zero.
    bool common_panel = false;
    for (int slice_col = 0; slice_col < num_slice_cols; slice_col++) {
        if (R_ssm.s_csc_mat.col_ptrs[slice_col] !=
                R_ssm.s_csc_mat.col_ptrs[slice_col + 1] &&
            S_ssm.s_csc_mat.col_ptrs[slice_col] !=
                S_ssm.s_csc_mat.col_ptrs[slice_col + 1]) {
            common_panel = true;
        }
        unsigned long long sparse_mm_cost_col = 0;
        unsigned long long sparse_dense_mm_cost_col = 0;
        unsigned long long dense_sparse_mm_cost_col = 0;
        unsigned long long dense_dense_mm_cost_col = 0;
        mm_check(
            slice_col, R_ssm, S_ssm, sparse_mm_cost_col,
            sparse_dense_mm_cost_col, dense_sparse_mm_cost_col,
            dense_dense_mm_cost_col
        );
        sparse_mm_cost += sparse_mm_cost_col;
        sparse_dense_mm_cost += sparse_dense_mm_cost_col;
        dense_sparse_mm_cost += dense_sparse_mm_cost_col;
        dense_dense_mm_cost += dense_dense_mm_cost_col;
    }
    if (!common_panel) {
        std::cout << "[backend] zero-work (disjoint slice columns)"
                  << std::endl;
        return SPALM_OK;
    }
    unsigned long long total_dense_mm_cost =
        sparse_dense_mm_cost + dense_sparse_mm_cost + dense_dense_mm_cost;
    end = std::chrono::high_resolution_clock::now();
    std::cout << "SPARSE MM cost: " << sparse_mm_cost << std::endl;
    std::cout << "SPARSE-DENSE MM cost: " << sparse_dense_mm_cost << std::endl;
    std::cout << "DENSE-SPARSE MM cost: " << dense_sparse_mm_cost << std::endl;
    std::cout << "DENSE-DENSE MM cost: " << dense_dense_mm_cost << std::endl;
    print_time("[mm-select-total]", end - total_start);

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

    int status = SPALM_OK;
    const auto backend = spalm_dispatch::select(
        force_mkl_dense, force_mkl_sparse, fallback,
        total_dense_mm_cost > (unsigned long long) M * K * N * 0.8,
        do_sparse_mm && !do_dense_mm
    );
    if (backend != spalm_dispatch::Backend::Internal) {
        std::cout << "[backend] "
                  << (backend == spalm_dispatch::Backend::MklDense
                          ? "mkl-dense"
                          : "mkl-sparse")
                  << " (fallback)" << std::endl;
        // Neither MKL path uses the slices: release them before MKL
        // allocates its dense inputs or transpose, instead of holding both.
        const auto release_begin = std::chrono::high_resolution_clock::now();
        R_ssm = ss_matrix<T>{};
        S_ssm = ss_matrix<T>{};
        print_time(
            "[fallback-slice-deallocate]",
            std::chrono::high_resolution_clock::now() - release_begin
        );
        status = run_mkl_backend(
            backend, type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, ret, threads
        );
    } else {
        std::cout << "[backend] spalm-internal sparse=" << do_sparse_mm
                  << " dense=" << do_dense_mm << std::endl;
        std::cout << std::endl;
        if (do_sparse_mm) {
            sparse_sparse_matmul(
                R_ssm, S_ssm, ret, sparse_result, threads, !do_dense_mm
            );
        }

        if (do_dense_mm) {
            dense_dense_matmul(
                R_ssm, S_ssm, tile_size_l2, tile_size_l3, ret, sparse_result,
                threads
            );
        }
    }

    end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    print_time("[mm-total]", end - total_start);
    print_time("[slice-mm-total]", end - e2e_start);
    const auto release_begin = std::chrono::high_resolution_clock::now();
    R_ssm = ss_matrix<T>{};
    S_ssm = ss_matrix<T>{};
    print_time(
        "[slice-deallocate]",
        std::chrono::high_resolution_clock::now() - release_begin
    );
    return status;
}

// Frees the process-global decision arrays on every exit path.
struct bucket_info_guard {
    ~bucket_info_guard() {
        const auto begin = std::chrono::high_resolution_clock::now();
        free_bucket_info();
        print_time(
            "[thresholds-deallocate]",
            std::chrono::high_resolution_clock::now() - begin
        );
    }
};

static int spalm_impl(
    int type, int *Ri, int *Rj, void *Rv, int *Si, int *Sj, void *Sv,
    const int M, const int K, const int N, bool sparse_result, void *ret,
    int *reti, int *retj, void *retv, int n, const char **keys,
    const char **vals
) {
    const auto initialize_begin = std::chrono::high_resolution_clock::now();
    auto invalid = [](const char *what) {
        std::cerr << "spalm: " << what << std::endl;
        return SPALM_INVALID_ARGUMENT;
    };
    if (type != 0 && type != 1) return invalid("type must be 0 or 1");
    if (M < 0 || K < 0 || N < 0) return invalid("negative dimension");
    if (n < 0 || (n > 0 && (!keys || !vals))) {
        return invalid("invalid key/value options");
    }
    // A zero-sized product leaves the caller's (initialized) output as is.
    if (M == 0 || K == 0 || N == 0) return SPALM_OK;
    if (!Ri || !Rj || !Rv || !Si || !Sj || !Sv || !ret) {
        return invalid("null matrix or result pointer");
    }
    // The dense kernels address the result with 32-bit offsets, including
    // gather/scatter indices up to 12 rows and 32 columns past a tile start.
    if ((long long) (M + 16) * (N + 32) > INT_MAX) {
        return invalid("M * N is too large for 32-bit result offsets");
    }

    auto get_str = [&](const char *key, const char *def) -> const char * {
        for (int i = 0; i < n; i++)
            if (strcmp(keys[i], key) == 0) return vals[i];
        return def;
    };
    auto get_int = [&](const char *key, int def) -> int {
        const char *v = get_str(key, nullptr);
        return v ? atoi(v) : def;
    };

    const char *init_method = get_str("init_method", "");
    const char *bucket_method = get_str("bucket_method", "");
    int threads = get_int("threads", 4);
    if (threads < 1) return invalid("threads must be at least 1");
    cpu_execution_options options;
    options.prune_zeros = get_int("prune_zeros", 1) != 0;
    options.compact_keys = get_int("compact_keys", 1) != 0;

    std::cout << "SPALM MATMUL" << std::endl;
    std::cout << M << " " << K << " " << N << std::endl;
    std::cout << "THREADS: " << threads << std::endl;
    if (type == 0) {
        std::cout << "FP32" << std::endl;
    } else if (type == 1) {
        std::cout << "FP64" << std::endl;
    }

    if (!parse_bucket_method(bucket_method)) {
        return invalid(
            "bucket_method must be linear or log with curvature > 0"
        );
    }
    bucket_info_guard bucket_info;
    bool read_success = init_decision_thresholds(init_method, type, "cpu");
    if (!read_success) {
        std::cerr << "Initialize decision thresholds." << std::endl;
        return SPALM_CONFIG_ERROR;
    }
    int _sh = 0, slice_length = 0, tile_size_l2 = 0, tile_size_l3 = 0;
    read_success = read_tile_sizes(
        type, "cpu", _sh, slice_length, tile_size_l2, tile_size_l3
    );
    if (!read_success) {
        std::cerr << "Initialize tile sizes." << std::endl;
        return SPALM_CONFIG_ERROR;
    }
    slice_length = get_int("slice_length", slice_length);
    tile_size_l2 = get_int("tile_size_l2", tile_size_l2);
    tile_size_l3 = get_int("tile_size_l3", tile_size_l3);

    std::cout << "Slice length: " << slice_length << std::endl;
    // Subrow and slice lengths are uint16_t; 0 used to pass the power-of-2
    // test and divide by zero later.
    if (slice_length < 1 || slice_length > 32768 ||
        (slice_length & (slice_length - 1)) != 0) {
        return invalid("slice length must be a power of 2 in [1, 32768]");
    }
    if (tile_size_l2 < 1 || tile_size_l3 < 1) {
        return invalid("tile sizes must be positive");
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
    print_time(
        "[library-initialize]",
        std::chrono::high_resolution_clock::now() - initialize_begin
    );

    if (type == 0) {
        return spalm_matmul(
            type, Ri, Rj, (float *) Rv, Si, Sj, (float *) Sv, M, K, N,
            slice_length, tile_size_l2, tile_size_l3, sparse_result,
            (float *) ret, reti, retj, (float *) retv, threads, options
        );
    }
    return spalm_matmul(
        type, Ri, Rj, (double *) Rv, Si, Sj, (double *) Sv, M, K, N,
        slice_length, tile_size_l2, tile_size_l3, sparse_result, (double *) ret,
        reti, retj, (double *) retv, threads, options
    );
}

// Returns a spalm_status; the result is valid only for 0 (SPALM_OK).
// Callers that still declare this as returning void keep working.
extern "C" int spalm(
    int type, int *Ri, int *Rj, void *Rv, int *Si, int *Sj, void *Sv,
    const int M, const int K, const int N, bool sparse_result, void *ret,
    int *reti, int *retj, void *retv, int n, const char **keys,
    const char **vals
) {
    // Decision thresholds and bucket settings are process-global, reset and
    // freed by every call: serialize calls so concurrent queries in one
    // process cannot overwrite or free each other's state mid-run.
    static std::mutex spalm_mutex;
    std::lock_guard<std::mutex> lock(spalm_mutex);
    // Never let an exception cross the C ABI. Exceptions thrown inside
    // OpenMP regions still terminate the process; this covers the rest
    // (e.g. std::bad_alloc from the slice or MKL input allocations).
    try {
        const auto begin = std::chrono::high_resolution_clock::now();
        const int status = spalm_impl(
            type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, sparse_result, ret, reti,
            retj, retv, n, keys, vals
        );
        // Everything inside the library, after the call lock is taken.
        print_time(
            "[spalm-total]", std::chrono::high_resolution_clock::now() - begin
        );
        return status;
    } catch (const std::invalid_argument &e) {
        // Malformed input detected during preparation (e.g. unsorted or
        // out-of-range CSR column indices).
        std::cerr << "spalm: " << e.what() << std::endl;
        return SPALM_INVALID_ARGUMENT;
    } catch (const std::exception &e) {
        std::cerr << "spalm: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "spalm: unknown exception" << std::endl;
    }
    return SPALM_INTERNAL_ERROR;
}
