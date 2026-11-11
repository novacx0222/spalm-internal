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
#include "cpu/execution_options.h"
#define SPARSE_SPARSE_MATMUL_H
struct spalm_sparse_result;
template <typename T>
void sparse_sparse_matmul(
    ss_matrix<T> &, ss_matrix<T> &, T *, bool, int, bool = false,
    const cpu_execution_options & = cpu_execution_options(),
    spalm_sparse_result * = nullptr
);

static int backend_calls;
// Calls of the SS kernel with sparse output (its row blocks are tested in
// tests/native_kernel_test.cpp and sparse_output_test.cpp).
static int sparse_out_calls;
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
    ss_matrix<T> &r, ss_matrix<T> &s, int, int, T *output, bool, int,
    const cpu_execution_options &
) {
    scalar_csr(
        r.csr_mat.row_ptrs, r.csr_mat.col_idxs, r.csr_mat.vals,
        s.csr_mat.row_ptrs, s.csr_mat.col_idxs, s.csr_mat.vals, output,
        r.csr_mat.rows, s.csr_mat.rows
    );
}
template <class T>
void sparse_sparse_matmul(
    ss_matrix<T> &r, ss_matrix<T> &s, T *output, bool, int, bool,
    const cpu_execution_options &, spalm_sparse_result *sparse_out
) {
    if (sparse_out) {
        ++sparse_out_calls;
        return;
    }
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

// The "[conf]" line of a log (without the newline), or "" if absent.
static std::string conf_line(const std::string &log) {
    const size_t at = log.find("[conf] ");
    if (at == std::string::npos) return "";
    return log.substr(at, log.find('\n', at) - at);
}

// A decision table on the 100 linear buckets: R slices below bucket 50 go
// to SS with every S slice, denser ones to DD (dense_mm_threshold 0, so the
// rule "dense work below it: all SS" is off unless overridden).
static std::string write_thresholds_file() {
    char path[] = "/tmp/spalm-dispatch-thresholds-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    std::ofstream out(path);
    out << "num_buckets: 100\nss_upper:";
    for (int b = 0; b <= 100; ++b) out << ' ' << (b < 50 ? 101 : 0);
    out << "\nri_upper:";
    for (int b = 0; b <= 100; ++b) out << " 0";
    out << "\nleft_dense_threshold: 0\ndense_mm_threshold: 0\n";
    return path;
}

using conf = std::vector<std::pair<const char *, const char *>>;

// Runs one call on a 24 x 16 R (rows 0-11: one entry, rows 12-23: full)
// and a full 32 x 16 S with slice length 16: one SS slice pair and dense
// ones. Returns the log; *status gets the return code.
template <class T> static std::string run_mixed(const conf &c, int *status) {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    const int M = 24, K = 16, N = 32;
    std::vector<int> ri{0}, rj, si{0}, sj;
    std::vector<T> rv, sv;
    for (int r = 0; r < M; ++r) {
        for (int k = 0; k < K; ++k) {
            if (r < 12 && !(r == 0 && k == 0)) continue;
            rj.push_back(k);
            rv.push_back(T(1));
        }
        ri.push_back(int(rj.size()));
    }
    for (int s = 0; s < N; ++s) {
        for (int k = 0; k < K; ++k) {
            sj.push_back(k);
            sv.push_back(T(1));
        }
        si.push_back(int(sj.size()));
    }
    std::vector<const char *> keys{"threads", "slice_length"},
        vals{"2", "16"};
    for (const auto &[k, v] : c) {
        keys.push_back(k);
        vals.push_back(v);
    }
    std::vector<T> output(size_t(M) * N);
    backend_calls = 0;
    return capture_stdout([&] {
        *status = spalm(
            type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
            sv.data(), M, K, N, false, output.data(), nullptr, nullptr,
            nullptr, int(keys.size()), keys.data(), vals.data()
        );
    });
}

// The tunables of execution_options.h: defaults, parsing, validation, the
// "[conf]" lines, and their effect on the automatic dispatch.
template <class T> void check_tunables() {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    int ri[] = {0, 3, 4}, rj[] = {0, 2, 3, 1};
    T rv[] = {2, 6, -1, 3};
    int si[] = {0, 3, 4}, sj[] = {0, 1, 3, 2};
    T sv[] = {4, 8, 5, 7};
    auto run = [&](const conf &c, int *status) {
        std::vector<const char *> keys{"threads"}, vals{"2"};
        for (const auto &[k, v] : c) {
            keys.push_back(k);
            vals.push_back(v);
        }
        T output[4] = {};
        backend_calls = 0;
        return capture_stdout([&] {
            *status = spalm(
                type, ri, rj, rv, si, sj, sv, 2, 4, 2, false, output, nullptr,
                nullptr, nullptr, int(keys.size()), keys.data(), vals.data()
            );
        });
    };
    int status = -1;

    // Defaults: the constants they replace. No unknown key, no weighted
    // costs (all weights 1).
    {
        const std::string log = run({{"init_method", "sparse"}}, &status);
        assert(status == SPALM_OK && backend_calls == 1);
        assert(count(log, "[conf] ") == 1);
        const std::string line = conf_line(log);
        for (const char *want :
             {"[conf] init_method=sparse bucket_method=linear "
              "thresholds_file=default dense_mm_threshold=0 fallback=0 ",
              " auto_sparse_ratio=0.1 auto_mkl_dense_ratio=0.8 "
              "auto_mkl_sparse_fallback=1 cost_weight_ss=1 cost_weight_sd=1 "
              "cost_weight_ds=1 cost_weight_dd=1 ",
              " auto_slice_length=1 asl_min_cost=3e+09 "
              "asl_subrow_write_entries=16 asl_min_cost_ratio=0.9 "
              "asl_max_levels=3 ",
              " r_block_huge_bytes=8388608 s_blocks_huge_bytes=2097152 ",
              " ss_column_blocks=-1 ss_column_block_min_cols=256 "
              "ss_column_block_cache_divisor=3 "})
            assert(line.find(want) != std::string::npos);
        assert(count(log, "[conf-unused]") == 0);
        assert(count(log, "[weighted-cost]") == 0);
        check_timing_format(log);
    }
    // Values are read and printed; 5e8 is an integer; misspelled keys are
    // listed, the operator's own keys are not.
    {
        const std::string log = run(
            {{"init_method", "sparse"},
             {"auto_sparse_ratio", "0.25"},
             {"auto_mkl_dense_ratio", "2"},
             {"auto_mkl_sparse_fallback", "0"},
             {"cost_weight_ss", "30"},
             {"cost_weight_dd", "0.5"},
             {"asl_min_cost", "1e10"},
             {"asl_subrow_write_entries", "8"},
             {"asl_min_cost_ratio", "0.75"},
             {"asl_max_levels", "2"},
             {"r_block_huge_bytes", "-1"},
             {"s_blocks_huge_bytes", "4194304"},
             {"ss_column_block_min_cols", "64"},
             {"ss_column_block_cache_divisor", "4"},
             {"result_prefault", "1"},
             {"asl_min_cots", "1"},
             {"so_dir", "/nowhere"},
             {"dense_mm_treshold", "5"}},
            &status
        );
        assert(status == SPALM_OK && backend_calls == 1);
        const std::string line = conf_line(log);
        for (const char *want :
             {" auto_sparse_ratio=0.25 auto_mkl_dense_ratio=2 "
              "auto_mkl_sparse_fallback=0 cost_weight_ss=30 cost_weight_sd=1 "
              "cost_weight_ds=1 cost_weight_dd=0.5 ",
              " asl_min_cost=1e+10 asl_subrow_write_entries=8 "
              "asl_min_cost_ratio=0.75 asl_max_levels=2 ",
              " r_block_huge_bytes=-1 s_blocks_huge_bytes=4194304 ",
              " ss_column_block_min_cols=64 ss_column_block_cache_divisor=4 "})
            assert(line.find(want) != std::string::npos);
        assert(count(log, "[conf-unused] keys=asl_min_cots,dense_mm_treshold\n") == 1);
        check_timing_format(log);
    }
    // Out of range or not a number: rejected before any work.
    const conf rejected[] = {
        {{"auto_sparse_ratio", "-0.1"}},
        {{"auto_sparse_ratio", "0.1x"}},
        {{"auto_sparse_ratio", ""}},
        {{"auto_mkl_dense_ratio", "nan"}},
        {{"auto_mkl_sparse_fallback", "2"}},
        {{"auto_mkl_sparse_fallback", "yes"}},
        {{"cost_weight_ss", "-1"}},
        {{"cost_weight_dd", "inf"}},
        {{"dense_mm_threshold", "-2"}},
        {{"dense_mm_threshold", "1.5"}},
        {{"dense_mm_threshold", "1e30"}},
        {{"asl_min_cost", "-1"}},
        {{"asl_subrow_write_entries", "-1"}},
        {{"asl_min_cost_ratio", "-0.5"}},
        {{"asl_max_levels", "0"}},
        {{"asl_max_levels", "4"}},
        {{"r_block_huge_bytes", "-2"}},
        {{"s_blocks_huge_bytes", "1 MiB"}},
        {{"ss_column_block_min_cols", "0"}},
        {{"ss_column_block_cache_divisor", "0"}},
        {{"ss_column_block_cache_divisor", "3000000000"}},
    };
    for (const conf &bad : rejected) {
        conf c = bad;
        c.push_back({"init_method", "sparse"});
        run(c, &status);
        assert(status == SPALM_INVALID_ARGUMENT);
        assert(backend_calls == 0);
    }
    // Accepted edge values: trailing blanks, -1, 0 and an integral 5e8.
    for (const conf &ok :
         {conf{{"dense_mm_threshold", "-1"}},
          conf{{"dense_mm_threshold", "5e8 "}},
          conf{{"auto_sparse_ratio", "0"}},
          conf{{"cost_weight_ss", "0"}},
          conf{{"r_block_huge_bytes", "0"}, {"s_blocks_huge_bytes", "-1"}},
          conf{{"asl_max_levels", "1"}}}) {
        conf c = ok;
        c.push_back({"init_method", "sparse"});
        run(c, &status);
        assert(status == SPALM_OK && backend_calls == 1);
    }

    // The automatic dispatch with a decision table from thresholds_file:
    // the mixed input has an SS pair and dense pairs.
    const std::string table = write_thresholds_file();
    auto mixed = [&](const char *method, conf c) {
        c.push_back({"init_method", method});
        c.push_back({"thresholds_file", table.c_str()});
        const std::string log = run_mixed<T>(c, &status);
        assert(status == SPALM_OK);
        assert(count(log, "[backend]") == 1);
        assert(count(log, "Loaded thresholds from: " + table) == 1);
        assert(conf_line(log).find(" thresholds_file=" + table + " ") != std::string::npos);
        return log;
    };
    const char *internal_mixed = "[backend] spalm-internal sparse=1 dense=1";
    const char *internal_dense = "[backend] spalm-internal sparse=0 dense=1";
    const char *internal_sparse = "[backend] spalm-internal sparse=1 dense=0";
    // Raw costs: SS is far below 0.1 x dense, so all pairs go dense.
    std::string log = mixed("no_fallback", {});
    assert(count(log, internal_dense) == 1);
    assert(count(log, "[weighted-cost]") == 0);
    assert(conf_line(log).find(" dense_mm_threshold=0 fallback=0 ") != std::string::npos);
    // Ratio 0: never all dense, so the call stays mixed.
    log = mixed("no_fallback", {{"auto_sparse_ratio", "0"}});
    assert(count(log, internal_mixed) == 1 && backend_calls == 2);
    // An SS weight that lifts SS above 0.1 x dense: mixed too, and the
    // weighted costs are logged.
    log = mixed("no_fallback", {{"cost_weight_ss", "1e6"}});
    assert(count(log, internal_mixed) == 1);
    assert(count(log, "[weighted-cost] sparse=") == 1);
    // Dense weight 0: dense work 0 is below every threshold, so all SS.
    log = mixed("no_fallback", {{"cost_weight_sd", "0"}, {"cost_weight_ds", "0"},
                                {"cost_weight_dd", "0"}, {"dense_mm_threshold", "1"}});
    assert(count(log, internal_sparse) == 1);
    // dense_mm_threshold overrides the file: dense work below it, all SS.
    log = mixed("no_fallback", {{"dense_mm_threshold", "1e15"}});
    assert(count(log, internal_sparse) == 1);
    assert(conf_line(log).find(" dense_mm_threshold=1000000000000000 ") != std::string::npos);
    // ... before the variants: no_threshold still sets it to 0.
    log = mixed("no_threshold", {{"dense_mm_threshold", "1e15"}, {"auto_mkl_dense_ratio", "1e30"}});
    assert(conf_line(log).find(" dense_mm_threshold=0 fallback=1 ") != std::string::npos);
    assert(count(log, internal_dense) == 1);
    // With fallback (auto): all SS falls back to MKL sparse unless
    // auto_mkl_sparse_fallback = 0.
    log = mixed("", {{"dense_mm_threshold", "1e15"}});
    assert(count(log, "[backend] mkl-sparse (fallback)") == 1);
    log = mixed("", {{"dense_mm_threshold", "1e15"}, {"auto_mkl_sparse_fallback", "0"}});
    assert(count(log, internal_sparse) == 1);
    // All dense falls back to MKL dense when the dense cost exceeds
    // auto_mkl_dense_ratio x M x K x N.
    log = mixed("", {{"auto_mkl_dense_ratio", "0"}});
    assert(count(log, "[backend] mkl-dense (fallback)") == 1);
    log = mixed("", {{"auto_mkl_dense_ratio", "1e30"}});
    assert(count(log, internal_dense) == 1);
    unlink(table.c_str());
    // A thresholds file that does not exist: a configuration error.
    run({{"init_method", ""}, {"thresholds_file", "/nonexistent/thresholds.txt"}}, &status);
    assert(status == SPALM_CONFIG_ERROR && backend_calls == 0);
}

// The caller's dense result for spalm_sparse_or_dense(), filled with `fill`
// when requested; null when `fail` is set.
template <class T> struct dense_buffer {
    std::vector<T> data;
    T fill = T(0);
    bool fail = false;
    int calls = 0;
    static void *provide(void *p) {
        auto *b = static_cast<dense_buffer *>(p);
        ++b->calls;
        if (b->fail) return nullptr;
        std::fill(b->data.begin(), b->data.end(), b->fill);
        return b->data.data();
    }
};

// spalm_sparse_or_dense() on the run_mixed() inputs: row blocks (no request)
// exactly when the call stays on SS; otherwise one request after the
// decision, the product in that buffer and no handle.
template <class T> void check_sparse_or_dense() {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    const int M = 24, K = 16, N = 32;
    std::vector<int> ri{0}, rj, si{0}, sj;
    std::vector<T> rv, sv;
    for (int r = 0; r < M; ++r) {
        for (int k = 0; k < K; ++k) {
            if (r < 12 && !(r == 0 && k == 0)) continue;
            rj.push_back(k);
            rv.push_back(T(1));
        }
        ri.push_back(int(rj.size()));
    }
    for (int s = 0; s < N; ++s) {
        for (int k = 0; k < K; ++k) {
            sj.push_back(k);
            sv.push_back(T(1));
        }
        si.push_back(int(sj.size()));
    }
    // Row 0: one key; rows 12-23: all 16; S is full.
    std::vector<T> expected(size_t(M) * N, T(0));
    for (int s = 0; s < N; ++s) expected[s] = T(1);
    for (int r = 12; r < M; ++r)
        for (int s = 0; s < N; ++s) expected[size_t(r) * N + s] = T(K);
    const std::string table = write_thresholds_file();
    auto run = [&](const conf &c, dense_buffer<T> &buf, void **handle,
                   int k = 16) {
        std::vector<const char *> keys{"threads", "slice_length"},
            vals{"2", "16"};
        for (const auto &[key, v] : c) {
            keys.push_back(key);
            vals.push_back(v);
        }
        buf.data.assign(size_t(M) * N, T(-5));
        backend_calls = 0;
        sparse_out_calls = 0;
        int status = -1;
        const std::string log = capture_stdout([&] {
            status = spalm_sparse_or_dense(
                type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
                sv.data(), M, k, N, int(keys.size()), keys.data(),
                vals.data(), &dense_buffer<T>::provide, &buf, handle
            );
        });
        return std::make_pair(status, log);
    };
    const char *t = table.c_str();
    // Mixed calls: both test doubles add the whole product (the exact
    // per-pair split is checked natively), so the buffer holds it twice.
    std::vector<T> twice(expected);
    for (T &x : twice) x += x;
    struct choice_case {
        conf c;
        bool dense;
        const char *line;
        bool mixed = false;
    } cases[] = {
        {{{"init_method", "no_fallback"}, {"thresholds_file", t},
          {"auto_sparse_ratio", "0"}},
         true, "[result-choice] dense (mixed)", true},
        {{{"init_method", "no_fallback"}, {"thresholds_file", t},
          {"dense_mm_threshold", "1000000000000"}},
         false, "[result-choice] sparse (all SS)"},
        {{{"init_method", "no_fallback"}, {"thresholds_file", t},
          {"cost_weight_ss", "0"}},
         true, "[result-choice] dense (all dense)"},
        {{{"init_method", ""}, {"thresholds_file", t},
          {"dense_mm_threshold", "1000000000000"},
          {"auto_mkl_dense_ratio", "1e9"}},
         true, "[result-choice] dense (mkl-sparse fallback)"},
        {{{"init_method", ""}, {"thresholds_file", t},
          {"auto_mkl_dense_ratio", "0"}},
         true, "[result-choice] dense (mkl-dense fallback)"},
        {{{"init_method", "FORCE_MKL_DENSE"}}, true,
         "[result-choice] dense (forced mkl-dense)"},
        {{{"init_method", "dense_bm_inner_kernel"}}, true,
         "[result-choice] dense (all dense)"},
        {{{"init_method", "sparse"}}, false, "[result-choice] sparse (all SS)"},
    };
    for (const auto &c : cases) {
        dense_buffer<T> buf;
        void *handle = reinterpret_cast<void *>(1);
        const auto [status, log] = run(c.c, buf, &handle);
        if (status != SPALM_OK || count(log, "[result-choice] ") != 1 ||
            count(log, c.line) != 1) {
            std::cerr << "want " << c.line << ", status " << status << ":\n"
                      << log;
        }
        assert(status == SPALM_OK);
        assert(count(log, "[result-choice] ") == 1 && count(log, c.line) == 1);
        if (c.dense) {
            assert(buf.calls == 1 && !handle && sparse_out_calls == 0);
            assert(count(log, "[dense-result-request]") == 1);
            assert(buf.data == (c.mixed ? twice : expected));
        } else {
            assert(buf.calls == 0 && handle && sparse_out_calls == 1);
            spalm_sparse_result_free(handle);
        }
    }
    // An uninitialized buffer (result_zeroed = 0) is zeroed before the
    // kernels add to it (mixed: twice the product, see above).
    {
        dense_buffer<T> buf;
        buf.fill = T(777);
        void *handle = nullptr;
        const auto [status, log] = run(
            {{"init_method", "no_fallback"}, {"thresholds_file", t},
             {"auto_sparse_ratio", "0"}, {"result_zeroed", "0"}},
            buf, &handle
        );
        assert(status == SPALM_OK && buf.calls == 1 && !handle);
        assert(count(log, "[result-choice] dense (mixed)") == 1);
        assert(buf.data == twice);
    }
    // No buffer from the caller: an internal error, no handle, no kernel.
    {
        dense_buffer<T> buf;
        buf.fail = true;
        void *handle = reinterpret_cast<void *>(1);
        const auto [status, log] = run(
            {{"init_method", "no_fallback"}, {"thresholds_file", t},
             {"auto_sparse_ratio", "0"}},
            buf, &handle
        );
        assert(status == SPALM_INTERNAL_ERROR && buf.calls == 1 && !handle);
        assert(backend_calls == 0 && count(log, "[result-choice] ") == 0);
    }
    // K = 0: a zero product, no request, a handle without blocks.
    {
        dense_buffer<T> buf;
        void *handle = nullptr;
        std::vector<int> ri0(M + 1, 0), si0(N + 1, 0);
        std::vector<const char *> keys{"init_method"}, vals{"no_fallback"};
        const int status = spalm_sparse_or_dense(
            type, ri0.data(), rj.data(), rv.data(), si0.data(), sj.data(),
            sv.data(), M, 0, N, 1, keys.data(), vals.data(),
            &dense_buffer<T>::provide, &buf, &handle
        );
        assert(status == SPALM_OK && buf.calls == 0 && handle);
        long long blocks = -1;
        spalm_sparse_result_blocks(handle, &blocks);
        assert(blocks == 0);
        spalm_sparse_result_free(handle);
    }
    // No callback, or no handle pointer: invalid arguments.
    {
        void *handle = reinterpret_cast<void *>(1);
        const char *keys[] = {"init_method"}, *vals[] = {"no_fallback"};
        assert(
            spalm_sparse_or_dense(
                type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
                sv.data(), M, K, N, 1, keys, vals, nullptr, nullptr, &handle
            ) == SPALM_INVALID_ARGUMENT &&
            !handle
        );
        dense_buffer<T> buf;
        assert(
            spalm_sparse_or_dense(
                type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
                sv.data(), M, K, N, 1, keys, vals, &dense_buffer<T>::provide,
                &buf, nullptr
            ) == SPALM_INVALID_ARGUMENT &&
            buf.calls == 0
        );
    }
    // spalm_sparse() keeps rejecting the methods without sparse output.
    {
        void *handle = reinterpret_cast<void *>(1);
        const char *keys[] = {"init_method"}, *vals[] = {"no_fallback"};
        int status = -1;
        capture_stdout([&] {
            status = spalm_sparse(
                type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
                sv.data(), M, K, N, 1, keys, vals, &handle
            );
        });
        assert(status == SPALM_INVALID_ARGUMENT && !handle);
    }
    unlink(table.c_str());
}

// log_candidate_work = 1 on a 24 x 256 R (rows 0-11: one entry each, rows
// 12-23: full) and a full 32 x 256 S, slice length 256, with the
// write_thresholds_file() table (R bucket < 50: SS, else DD). R's first
// slice holds 12 entries but is bucket 0, whose representative count is 1:
// so "bucket" and "nnz" differ there. Without the key nothing is printed,
// and the result is the same either way.
template <class T> void check_candidate_work() {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    const int M = 24, K = 256, N = 32;
    std::vector<int> ri{0}, rj, si{0}, sj;
    std::vector<T> rv, sv;
    for (int r = 0; r < M; ++r) {
        for (int k = 0; k < K; ++k) {
            if (r < 12 && k != r) continue;
            rj.push_back(k);
            rv.push_back(T(1));
        }
        ri.push_back(int(rj.size()));
    }
    for (int s = 0; s < N; ++s) {
        for (int k = 0; k < K; ++k) {
            sj.push_back(k);
            sv.push_back(T(1));
        }
        si.push_back(int(sj.size()));
    }
    const std::string table = write_thresholds_file();
    auto run = [&](const char *log_work, std::vector<T> &out) {
        std::vector<const char *> keys{"threads", "slice_length",
                                       "init_method", "thresholds_file",
                                       "auto_sparse_ratio"},
            vals{"2", "256", "no_fallback", table.c_str(), "0"};
        if (log_work) {
            keys.push_back("log_candidate_work");
            vals.push_back(log_work);
        }
        out.assign(size_t(M) * N, T(0));
        int status = -1;
        const std::string log = capture_stdout([&] {
            status = spalm(
                type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
                sv.data(), M, K, N, false, out.data(), nullptr, nullptr,
                nullptr, int(keys.size()), keys.data(), vals.data()
            );
        });
        assert(status == SPALM_OK);
        return log;
    };
    std::vector<T> with, without, off;
    const std::string log = run("1", with);
    const std::string plain = run(nullptr, without);
    const std::string zero = run("0", off);
    assert(with == without && off == without);
    for (const std::string *l : {&plain, &zero}) {
        assert(count(*l, "[candidate-") == 0);
        assert(count(*l, " log_candidate_work=0\n") == 1);
    }
    assert(count(log, " log_candidate_work=1\n") == 1);
    // S has 2 slices of 16 rows in double, 1 of 32 in float.
    const bool fl = type == 0;
    const std::string features =
        std::string("[candidate-features] M=24 N=32 K=256 sl=256 mr=12 nr=") +
        (fl ? "32" : "16") +
        " nnz_r=3084 nnz_s=8192 flops=98688 row_bound=768 r_slices=2 "
        "s_slices=" +
        (fl ? "1" : "2") + " slice_pairs=" + (fl ? "2" : "4") + "\n";
    assert(count(log, features) == 1);
    const char *lines[] = {
        "[candidate-work] table bucket ss=32 sd=0 ds=0 dd=98304\n",
        "[candidate-work] table nnz ss=384 sd=0 ds=0 dd=98304\n",
        "[candidate-work] all_ss bucket ss=98336 sd=0 ds=0 dd=0\n",
        "[candidate-work] all_ss nnz ss=98688 sd=0 ds=0 dd=0\n",
        "[candidate-work] all_dense bucket ss=0 sd=0 ds=0 dd=196608\n",
        "[candidate-work] all_dense nnz ss=0 sd=0 ds=0 dd=196608\n",
    };
    for (const char *l : lines) assert(count(log, l) == 1);
    assert(count(log, "[candidate-work] ") == 6);
    // Both budgets count exactly: every row has all 32 columns after its
    // first key's 32 S rows.
    for (const char *budget : {"134217728", "1048576"}) {
        assert(count(log, std::string("[candidate-entries] budget=") + budget +
                              " estimate=768 exact=1 draws=24 rows=24 "
                              "long_rows=0 visits=768\n") == 1);
    }
    assert(count(log, "[candidate-entries] ") == 2);
    assert(count(log, "[candidate-entries-small-time] ") == 1);
    // Double: R slice 0 (bucket 0) x 2 S slices on SS, R slice 1 x 2 on DD.
    const std::string half = fl ? "1" : "2", all = fl ? "2" : "4";
    const std::string pairs[] = {
        "[candidate-pairs] table ss=" + half + " sd=0 ds=0 dd=" + half +
            " pack_r=1 pack_s=1 bitmaps=1 r_stride=12 fixed_alg=-1\n",
        "[candidate-pairs] all_ss ss=" + all +
            " sd=0 ds=0 dd=0 pack_r=0 pack_s=0 bitmaps=0 r_stride=12 "
            "fixed_alg=0\n",
        "[candidate-pairs] all_dense ss=0 sd=0 ds=0 dd=" + all +
            " pack_r=1 pack_s=1 bitmaps=1 r_stride=12 fixed_alg=3\n",
    };
    for (const std::string &l : pairs) assert(count(log, l) == 1);
    assert(count(log, "[candidate-pairs] ") == 3);
    // One slice column, 2 threads: both R slices in one block of 2 R tasks
    // (one slice each), and every S tile packed once per R task.
    const std::string schedule =
        std::string("[candidate-dense-schedule] tile_size_l2=128 "
                    "tile_size_l3=") +
        (fl ? "4608" : "2304") +
        " threads=2 value_bytes=" + (fl ? "4" : "8") +
        " r_packs=2 r_pack_entries=3084 s_packs=" + all +
        " s_pack_entries=16384 tasks=2\n";
    assert(count(log, schedule) == 1);
    assert(count(log, "[candidate-entries-time] ") == 1);
    // "table bucket" repeats the cost lines.
    assert(count(log, "\nSPARSE MM cost: 32\n") == 1);
    assert(count(log, "\nDENSE-DENSE MM cost: 98304\n") == 1);
    assert(count(log, "[candidate-work-time] ") == 1);
    // The table is restored: the call still runs mixed.
    assert(count(log, "[backend] spalm-internal sparse=1 dense=1") == 1);
    assert(count(plain, "[backend] spalm-internal sparse=1 dense=1") == 1);
    check_timing_format(log);
    unlink(table.c_str());
}

int main() {
    check_methods<float>();
    check_methods<double>();
    check_tunables<float>();
    check_tunables<double>();
    check_sparse_or_dense<float>();
    check_sparse_or_dense<double>();
    check_candidate_work<float>();
    check_candidate_work<double>();
    std::cout << "PASS: forced SPALM and MKL methods through the C API, "
                 "backend lines, timing labels and format; tunables: "
                 "defaults, parsing, validation, [conf] lines, thresholds "
                 "file and their effect on the automatic dispatch; "
                 "spalm_sparse_or_dense: result choice, requests, zeroing "
                 "and failures; log_candidate_work lines\n";
}
