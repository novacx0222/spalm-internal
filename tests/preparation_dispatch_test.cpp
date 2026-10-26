// Execute the real wrapper/preprocessing on portable machines. Replace only
// SIMD/MKL computation and dynamic loading with explicit scalar test doubles.
#include "ss_mat.h"
#include "path_utils.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>

// Skip the x86-only low-level declarations; keep the exact high-level signature.
#define SPARSE_SPARSE_MATMUL_H
template <typename T>
void sparse_sparse_matmul(
    ss_matrix<T> &, ss_matrix<T> &, T *, bool, int, bool = false
);

static int backend_calls, seen_r_nnz, seen_s_nnz;
static void *seen_r_values;
static void scalar_baseline(
    int, const char *, int *, int *, void *, void *, int *, int *, void *,
    void *, void *, int, int, int, int
);
static void *test_dlopen(const char *, int) {
    return reinterpret_cast<void *>(1);
}
static void *test_dlsym(void *, const char *) {
    return reinterpret_cast<void *>(&scalar_baseline);
}
#define dlopen test_dlopen
#define dlsym test_dlsym
#include "../src/cpu/spalm.cpp"
#undef dlopen
#undef dlsym

template <class T>
void scalar_csr(
    int *ri, int *rj, T *rv, int *si, int *sj, T *sv, T *output, int m, int n
) {
    ++backend_calls;
    seen_r_nnz = ri[m];
    seen_s_nnz = si[n];
    seen_r_values = rv;
    for (int r = 0; r < m; ++r)
        for (int s = 0; s < n; ++s)
            for (int p = ri[r]; p < ri[r + 1]; ++p)
                for (int q = si[s]; q < si[s + 1]; ++q)
                    if (rj[p] == sj[q])
                        output[size_t(r) * n + s] += rv[p] * sv[q];
}

template <class T>
void dense_dense_matmul(
    ss_matrix<T> &r, ss_matrix<T> &s, int, int, T *output, bool, int
) {
    const auto begin = std::chrono::steady_clock::now();
    scalar_csr(
        r.csr_mat.row_ptrs, r.csr_mat.col_idxs, r.csr_mat.vals,
        s.csr_mat.row_ptrs, s.csr_mat.col_idxs, s.csr_mat.vals, output,
        r.csr_mat.rows, s.csr_mat.rows
    );
#ifdef SPALM_THRESHOLD_TEST_LIBRARY
    // Exercise the production formatter/capture/parser, with scalar computation.
    print_time(
        "[dense-dense-dense-mm-time]", std::chrono::steady_clock::now() - begin
    );
#endif
}
template <class T>
void sparse_sparse_matmul(
    ss_matrix<T> &r, ss_matrix<T> &s, T *output, bool, int, bool
) {
    const auto begin = std::chrono::steady_clock::now();
    scalar_csr(
        r.csr_mat.row_ptrs, r.csr_mat.col_idxs, r.csr_mat.vals,
        s.csr_mat.row_ptrs, s.csr_mat.col_idxs, s.csr_mat.vals, output,
        r.csr_mat.rows, s.csr_mat.rows
    );
#ifdef SPALM_THRESHOLD_TEST_LIBRARY
    print_time(
        "[sparse-sparse-sparse-mm-time]",
        std::chrono::steady_clock::now() - begin
    );
#endif
}
template <class T>
void baseline_typed(
    const char *method, int *ri, int *rj, void *rv, void *rd, int *si, int *sj,
    void *sv, void *sd, void *output, int m, int k, int n
) {
    if (std::strcmp(method, "MKL sparse-sparse") == 0) {
        scalar_csr(
            ri, rj, static_cast<T *>(rv), si, sj, static_cast<T *>(sv),
            static_cast<T *>(output), m, n
        );
        return;
    }
    ++backend_calls;
    seen_r_nnz = ri[m];
    seen_s_nnz = si[n];
    seen_r_values = rv;
    for (int r = 0; r < m; ++r)
        for (int s = 0; s < n; ++s)
            for (int j = 0; j < k; ++j)
                static_cast<T *>(output)[size_t(r) * n + s] +=
                    static_cast<T *>(rd)[size_t(r) * k + j] *
                    static_cast<T *>(sd)[size_t(s) * k + j];
}
static void scalar_baseline(
    int type, const char *method, int *ri, int *rj, void *rv, void *rd, int *si,
    int *sj, void *sv, void *sd, void *output, int m, int k, int n, int
) {
    if (type == 0)
        baseline_typed<float>(
            method, ri, rj, rv, rd, si, sj, sv, sd, output, m, k, n
        );
    else
        baseline_typed<double>(
            method, ri, rj, rv, rd, si, sj, sv, sd, output, m, k, n
        );
}

template <class T> void check_dispatch() {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    int ri[] = {0, 3, 4}, rj[] = {0, 2, 3, 1};
    T rv[] = {2, 0, -1, 3};
    int si[] = {0, 3, 4}, sj[] = {0, 1, 3, 2};
    T sv[] = {4, 0, 5, 7};
    auto invoke = [&](bool prune, int m = 2, int k = 4, int n = 2) {
        T output[4] = {};
        backend_calls = 0;
        std::ostringstream log;
        auto *old = std::cout.rdbuf(log.rdbuf());
        cpu_execution_options options;
        options.prune_zeros = prune;
        const int status = spalm_matmul(
            type, ri, rj, rv, si, sj, sv, m, k, n, 64, 32, 48, false, output,
            nullptr, nullptr, static_cast<T *>(nullptr), 4, options
        );
        assert(status == SPALM_OK);
        std::cout.rdbuf(old);
        if (backend_calls) {
            T reference[4] = {};
            // Independent expected values for this fixture (same for pruned/raw CSR).
            reference[0] = 3;
            reference[1] = 0;
            reference[2] = 0;
            reference[3] = 0;
            for (int i = 0; i < 4; ++i) assert(output[i] == reference[i]);
        } else {
            for (auto value : output) assert(value == T(0));
        }
        return log.str();
    };
    for (const char *mode : {"sparse", "dense_bm_inner_kernel"}) {
        assert(init_decision_thresholds(mode, type, "cpu"));
        invoke(true);
        assert(
            backend_calls == 1 && seen_r_nnz == 3 && seen_s_nnz == 3 &&
            seen_r_values != rv
        );
        invoke(false);
        assert(
            backend_calls == 1 && seen_r_nnz == 4 && seen_s_nnz == 4 &&
            seen_r_values == rv
        );
        // Structural and numeric zeros must not reach any backend.
        const T saved[] = {rv[0], rv[1], rv[2], rv[3]};
        std::fill(rv, rv + 4, T(0));
        invoke(true);
        assert(backend_calls == 0);
        std::copy(saved, saved + 4, rv);
        invoke(true, 0);
        assert(backend_calls == 0);
        invoke(true, 2, 0);
        assert(backend_calls == 0);
        int middle = ri[1], last = ri[2];
        ri[1] = ri[2] = 0;
        invoke(true);
        assert(backend_calls == 0);
        ri[1] = middle;
        ri[2] = last;
    }
    for (bool dense : {false, true}) {
        assert(init_decision_thresholds("sparse", type, "cpu"));
        force_mkl_dense = dense;
        force_mkl_sparse = !dense;
        auto log = invoke(true);
        assert(backend_calls == 1 && seen_r_nnz == 3 && seen_s_nnz == 3);
        assert(log.find("[slice-prepare-total]") == std::string::npos);
        T saved[] = {sv[0], sv[1], sv[2], sv[3]};
        std::fill(sv, sv + 4, T(0));
        invoke(true);
        assert(backend_calls == 0);
        std::copy(saved, saved + 4, sv);
    }
    // Zero-free CSR is borrowed even with pruning enabled.
    rv[1] = 6;
    sv[1] = 8;
    T output[4] = {};
    backend_calls = 0;
    assert(
        spalm_matmul(
            type, ri, rj, rv, si, sj, sv, 2, 4, 2, 64, 32, 48, false, output,
            nullptr, nullptr, static_cast<T *>(nullptr), 4,
            cpu_execution_options{}
        ) == SPALM_OK
    );
    assert(seen_r_values == rv && seen_r_nnz == 4 && seen_s_nnz == 4);
    assert(
        output[0] == 3 && output[1] == 42 && output[2] == 24 && output[3] == 0
    );
    // Explicit zero must survive when another input contains Inf (0*Inf = NaN).
    int row[] = {0, 1}, col[] = {0};
    T zero[] = {0}, infinite[] = {std::numeric_limits<T>::infinity()},
      result[] = {0};
    force_mkl_dense = true;
    force_mkl_sparse = false;
    backend_calls = 0;
    assert(
        spalm_matmul(
            type, row, col, zero, row, col, infinite, 1, 1, 1, 64, 32, 48,
            false, result, nullptr, nullptr, static_cast<T *>(nullptr), 4,
            cpu_execution_options{}
        ) == SPALM_OK
    );
    assert(backend_calls == 1 && seen_r_nnz == 1 && std::isnan(result[0]));
    // Nonempty inputs with entirely disjoint panels need no native backend.
    assert(init_decision_thresholds("sparse", type, "cpu"));
    T one[] = {1};
    int other_col[] = {64};
    result[0] = 0;
    backend_calls = 0;
    assert(
        spalm_matmul(
            type, row, col, one, row, other_col, one, 1, 128, 1, 64, 32, 48,
            false, result, nullptr, nullptr, static_cast<T *>(nullptr), 4,
            cpu_execution_options{}
        ) == SPALM_OK
    );
    assert(backend_calls == 0 && result[0] == 0);
}

// End to end through the exported C API: option parsing, forced modes without
// a thresholds file, malformed input and status codes. Runs from the
// repository root, where parameters/cpu/*/tile_sizes.txt is found.
template <class T> void check_c_api() {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    int ri[] = {0, 3, 4}, rj[] = {0, 2, 3, 1};
    T rv[] = {2, 6, -1, 3};
    int si[] = {0, 3, 4}, sj[] = {0, 1, 3, 2};
    T sv[] = {4, 8, 5, 7};
    auto call = [&](const char *method, const char *prune, int *columns,
                    T *output) {
        const char *keys[] = {"init_method", "threads", "prune_zeros"};
        const char *vals[] = {method, "4", prune};
        std::ostringstream log;
        auto *old = std::cout.rdbuf(log.rdbuf());
        const int status = spalm(
            type, ri, columns, rv, si, sj, sv, 2, 4, 2, false, output, nullptr,
            nullptr, nullptr, 3, keys, vals
        );
        std::cout.rdbuf(old);
        return status;
    };
    for (const char *method :
         {"FORCE_MKL_DENSE", "FORCE_MKL_SPARSE", "sparse"}) {
        T output[4] = {};
        backend_calls = 0;
        assert(call(method, "1", rj, output) == SPALM_OK);
        assert(backend_calls == 1);
        assert(
            output[0] == 3 && output[1] == 42 && output[2] == 24 &&
            output[3] == 0
        );
    }
    // prune_zeros=0 passes the caller's arrays through unchanged.
    T output[4] = {};
    rv[1] = 0;
    assert(call("FORCE_MKL_SPARSE", "0", rj, output) == SPALM_OK);
    assert(seen_r_values == rv && seen_r_nnz == 4);
    assert(call("FORCE_MKL_SPARSE", "1", rj, output) == SPALM_OK);
    assert(seen_r_values != rv && seen_r_nnz == 3);
    rv[1] = 6;
    // Unsorted CSR columns: rejected as an invalid argument, not computed.
    int unsorted[] = {3, 2, 0, 1};
    backend_calls = 0;
    assert(call("sparse", "1", unsorted, output) == SPALM_INVALID_ARGUMENT);
    assert(backend_calls == 0);
    // Unknown bucket method and an invalid option are rejected up front.
    const char *keys[] = {"init_method", "bucket_method"};
    const char *vals[] = {"sparse", "bogus"};
    assert(
        spalm(
            type, ri, rj, rv, si, sj, sv, 2, 4, 2, false, output, nullptr,
            nullptr, nullptr, 2, keys, vals
        ) == SPALM_INVALID_ARGUMENT
    );
}

#ifndef SPALM_THRESHOLD_TEST_LIBRARY
int main() {
    check_dispatch<float>();
    check_dispatch<double>();
    check_c_api<float>();
    check_c_api<double>();
    std::cout
        << "PASS: actual wrapper zero/empty exits, CSR ownership, "
           "scalar-checked dispatch, forced slice bypass, Inf semantics, C API "
           "status codes and options; SIMD/MKL backends are test doubles\n";
}
#endif
