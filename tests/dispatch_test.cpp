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
    assert(!sparse_out);
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

int main() {
    check_methods<float>();
    check_methods<double>();
    check_tunables<float>();
    check_tunables<double>();
    std::cout << "PASS: forced SPALM and MKL methods through the C API, "
                 "backend lines, timing labels and format; tunables: "
                 "defaults, parsing, validation, [conf] lines, thresholds "
                 "file and their effect on the automatic dispatch\n";
}
