// Forced methods and timing output through the exported C API on portable
// machines. Only the SIMD kernels, MKL and dynamic loading are replaced by
// scalar test doubles. Runs from the repository root, where
// parameters/cpu/*/tile_sizes.txt is found.
#include "ss_mat.h"
#include "path_utils.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <unistd.h>

// Skip the x86-only low-level declarations; keep the exact high-level signature.
#define SPARSE_SPARSE_MATMUL_H
template <typename T>
void sparse_sparse_matmul(
    ss_matrix<T> &, ss_matrix<T> &, T *, bool, int, bool = false
);

static int backend_calls;
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
    scalar_csr(
        r.csr_mat.row_ptrs, r.csr_mat.col_idxs, r.csr_mat.vals,
        s.csr_mat.row_ptrs, s.csr_mat.col_idxs, s.csr_mat.vals, output,
        r.csr_mat.rows, s.csr_mat.rows
    );
}
template <class T>
void sparse_sparse_matmul(
    ss_matrix<T> &r, ss_matrix<T> &s, T *output, bool, int, bool
) {
    scalar_csr(
        r.csr_mat.row_ptrs, r.csr_mat.col_idxs, r.csr_mat.vals,
        s.csr_mat.row_ptrs, s.csr_mat.col_idxs, s.csr_mat.vals, output,
        r.csr_mat.rows, s.csr_mat.rows
    );
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

// Captures everything written to stdout (std::cout and printf) by f().
template <class F> static std::string capture_stdout(F f) {
    std::cout.flush();
    std::fflush(stdout);
    char path[] = "/tmp/spalm-dispatch-test-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    const int saved = dup(STDOUT_FILENO);
    dup2(fd, STDOUT_FILENO);
    f();
    std::cout.flush();
    std::fflush(stdout);
    dup2(saved, STDOUT_FILENO);
    close(saved);
    close(fd);
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    unlink(path);
    return text.str();
}

static int count(const std::string &log, const std::string &needle) {
    int n = 0;
    for (size_t at = log.find(needle); at != std::string::npos;
         at = log.find(needle, at + needle.size()))
        ++n;
    return n;
}

// Every timing line is "[label] <value with 6 decimals> ms"; other
// bracketed lines ([backend], statistics) carry no unit.
static void check_timing_format(const std::string &log) {
    static const std::regex timing(R"(\[[a-z0-9-]+\] [0-9]+\.[0-9]{6} ms)");
    std::istringstream lines(log);
    int timings = 0;
    for (std::string line; std::getline(lines, line);) {
        if (line.size() < 3 || line.compare(line.size() - 3, 3, " ms") != 0)
            continue;
        assert(std::regex_match(line, timing));
        ++timings;
    }
    assert(timings > 0);
}

template <class T> void check_methods() {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    int ri[] = {0, 3, 4}, rj[] = {0, 2, 3, 1};
    T rv[] = {2, 6, -1, 3};
    int si[] = {0, 3, 4}, sj[] = {0, 1, 3, 2};
    T sv[] = {4, 8, 5, 7};
    struct Case {
        const char *method, *backend;
        bool internal;
    };
    const Case cases[] = {
        {"FORCE_MKL_DENSE", "[backend] mkl-dense (forced)", false},
        {"FORCE_MKL_SPARSE", "[backend] mkl-sparse (forced)", false},
        {"sparse", "[backend] spalm-internal", true},
        {"dense_bm_inner_kernel", "[backend] spalm-internal", true},
        {"dense_left_spmm_inner_kernel", "[backend] spalm-internal", true},
        {"dense_right_spmm_inner_kernel", "[backend] spalm-internal", true},
    };
    for (const Case &c : cases) {
        const char *keys[] = {"init_method", "threads"};
        const char *vals[] = {c.method, "4"};
        T output[4] = {};
        int status = -1;
        backend_calls = 0;
        const std::string log = capture_stdout([&] {
            status = spalm(
                type, ri, rj, rv, si, sj, sv, 2, 4, 2, false, output, nullptr,
                nullptr, nullptr, 2, keys, vals
            );
        });
        assert(status == SPALM_OK);
        assert(backend_calls == 1);
        assert(
            output[0] == 3 && output[1] == 42 && output[2] == 24 &&
            output[3] == 0
        );
        assert(count(log, c.backend) == 1 && count(log, "[backend]") == 1);
        check_timing_format(log);
        for (const char *label :
             {"[library-initialize]", "[thresholds-deallocate]",
              "[spalm-total]"})
            assert(count(log, label) == 1);
        if (c.internal) {
            for (const char *label :
                 {"[1-slice-subrow-csr]", "[1-slice-subrow-csc]",
                  "[1-slice-csc]", "[2-slice-subrow-csr]",
                  "[2-slice-subrow-csc]", "[2-slice-csc]",
                  "[slice-prepare-total]", "[mm-select-total]", "[mm-total]",
                  "[slice-mm-total]", "[slice-deallocate]"})
                assert(count(log, label) == 1);
            assert(count(log, "[forced-backend-total]") == 0);
        } else {
            // Forced MKL: no slices and no cost model.
            assert(count(log, "[slice-prepare-total]") == 0);
            assert(count(log, "[mm-select-total]") == 0);
            assert(count(log, "[forced-backend-total]") == 1);
            assert(count(log, "[mkl-call-total]") == 1);
        }
    }
    // No stored entry on one side: nothing reaches a backend.
    int empty[] = {0, 0, 0};
    for (const char *method : {"FORCE_MKL_DENSE", "sparse"}) {
        const char *keys[] = {"init_method"};
        const char *vals[] = {method};
        T output[4] = {};
        backend_calls = 0;
        const std::string log = capture_stdout([&] {
            assert(
                spalm(
                    type, ri, rj, rv, empty, sj, sv, 2, 4, 2, false, output,
                    nullptr, nullptr, nullptr, 1, keys, vals
                ) == SPALM_OK
            );
        });
        assert(backend_calls == 0);
        assert(count(log, "[backend] zero-work") == 1);
    }
    // An unknown bucket method is rejected before any work.
    const char *keys[] = {"init_method", "bucket_method"};
    const char *vals[] = {"sparse", "bogus"};
    T output[4] = {};
    backend_calls = 0;
    capture_stdout([&] {
        assert(
            spalm(
                type, ri, rj, rv, si, sj, sv, 2, 4, 2, false, output, nullptr,
                nullptr, nullptr, 2, keys, vals
            ) == SPALM_INVALID_ARGUMENT
        );
    });
    assert(backend_calls == 0);
}

int main() {
    check_methods<float>();
    check_methods<double>();
    std::cout << "PASS: forced SPALM and MKL methods through the C API, "
                 "backend lines, timing labels and format\n";
}
