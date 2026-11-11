#include <algorithm>
#include "path_utils.h"
#include <chrono>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cpu/dense_dense_matmul.h"
#include "cpu/execution_options.h"
#include "candidate_work.h"
#include "csr_preparation.h"
#include "huge_pages.h"
#include "method_dispatch.h"
#include "mkl.h"
#include "slice_decision.h"
#include "slice_length_rule.h"
#include "cpu/sparse_sparse_matmul.h"
#include "spalm_sparse_result.h"
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

// Zeroes `bytes` bytes of a result the caller left uninitialized
// (result_zeroed = 0), in 256 KiB blocks over the threads.
static void zero_result(void *ret, size_t bytes, int threads) {
    const auto begin = std::chrono::high_resolution_clock::now();
    constexpr size_t block = 256 * 1024;
    const long long blocks = (long long) ((bytes + block - 1) / block);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long long b = 0; b < blocks; ++b) {
        const size_t start = (size_t) b * block;
        std::memset(
            static_cast<char *>(ret) + start, 0, std::min(block, bytes - start)
        );
    }
    print_time(
        "[library-result-zero]",
        std::chrono::high_resolution_clock::now() - begin
    );
}

// Used only as a dladdr anchor to locate this .so at runtime.
static void spalm_lib_anchor() {}

using run_baseline_fn = void (*)(
    int, const char *, int *, int *, void *, void *, int *, int *, void *,
    void *, void *, const int, const int, const int, const int
);

// MKL sparse-sparse with sparse output (run_baseline_sparse_output): C as
// CSR arrays owned by *handle until run_baseline_sparse_output_free.
using run_baseline_sparse_output_fn = int (*)(
    int, int *, int *, void *, int *, int *, void *, const int, const int,
    const int, const int, void **, int **, int **, int **, void **
);
using run_baseline_sparse_output_free_fn = void (*)(void *);

// Returns nullptr (after printing why) if the MKL baseline cannot be loaded.
// The library stays loaded for the process lifetime (unloading MKL corrupts
// its thread pool); a successful lookup is cached, a failure is retried.
// Calls are serialized by spalm(), so the cache needs no further locking.
static void *load_baseline_library() {
    void *handle = dlopen(
        resolve_path_near_lib(
            reinterpret_cast<void *>(&spalm_lib_anchor), "run_baseline.so",
            "./bin/run_baseline.so"
        )
            .c_str(),
        RTLD_LAZY
    );
    if (!handle) std::cerr << "Cannot open library: " << dlerror() << '\n';
    return handle;
}

template <typename F> static F load_baseline_symbol(const char *name) {
    void *handle = load_baseline_library();
    if (!handle) return nullptr;
    F symbol = (F) dlsym(handle, name);
    if (!symbol) {
        std::cerr << "Cannot find " << name << ": " << dlerror() << '\n';
    }
    return symbol;
}

static run_baseline_fn load_run_baseline() {
    static run_baseline_fn cached = nullptr;
    if (!cached) cached = load_baseline_symbol<run_baseline_fn>("run_baseline");
    return cached;
}

struct baseline_sparse_output {
    run_baseline_sparse_output_fn run = nullptr;
    run_baseline_sparse_output_free_fn free = nullptr;
};

static baseline_sparse_output load_baseline_sparse_output() {
    static baseline_sparse_output cached;
    if (!cached.run || !cached.free) {
        cached.run = load_baseline_symbol<run_baseline_sparse_output_fn>(
            "run_baseline_sparse_output"
        );
        cached.free = load_baseline_symbol<run_baseline_sparse_output_free_fn>(
            "run_baseline_sparse_output_free"
        );
    }
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
        // make_unique<T[]> value-initialized both inputs on one thread (about
        // 1.85 s for two 16000 x 16000 doubles). Allocate them uninitialized;
        // the fill loop zeroes each row (implicit zeros) before scattering
        // its entries, so the first touch is spread over the threads too.
        // On huge pages (huge_pages.h): on 4 KiB pages, filling and freeing
        // them took ~31 and ~33 ms per GB (m1 SuiteSparse run).
        auto R = spalm_huge::make_for_overwrite<T>(checked_matrix_size(M, K));
        auto S = spalm_huge::make_for_overwrite<T>(checked_matrix_size(N, K));
        auto end = std::chrono::high_resolution_clock::now();
        print_time("[mkl-dense-input-allocate]", end - start);
        start = end;
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < M; ++i) {
            T *row = &R[(size_t) i * K];
            std::fill(row, row + K, T(0));
            for (int p = Ri[i]; p < Ri[i + 1]; ++p) row[Rj[p]] = Rv[p];
        }
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; ++i) {
            T *row = &S[(size_t) i * K];
            std::fill(row, row + K, T(0));
            for (int p = Si[i]; p < Si[i + 1]; ++p) row[Sj[p]] = Sv[p];
        }
        end = std::chrono::high_resolution_clock::now();
        // Zeroing included since inner-v6.
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

// Forced MKL sparse-sparse with sparse output: the same S transpose as MKL
// sparse-sparse, then mkl_sparse_?_sp2m instead of mkl_sparse_?_spmmd (in
// run_baseline.so). The result keeps MKL's C; its rows are cut into blocks
// of about 64 Ki entries for the caller's threads.
static int run_mkl_sparse_output(
    int type, int *Ri, int *Rj, void *Rv, int *Si, int *Sj, void *Sv, int M,
    int K, int N, int threads, spalm_sparse_result &out
) {
    const baseline_sparse_output baseline = load_baseline_sparse_output();
    if (!baseline.run || !baseline.free) return SPALM_BASELINE_ERROR;
    std::cout << "Fallback: MKL sparse-sparse (sparse output)" << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();
    void *handle = nullptr;
    int *rows_start = nullptr, *rows_end = nullptr, *cols = nullptr;
    void *vals = nullptr;
    const int status = baseline.run(
        type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, threads, &handle, &rows_start,
        &rows_end, &cols, &vals
    );
    print_time(
        "[mkl-call-total]", std::chrono::high_resolution_clock::now() - start
    );
    if (status != 0) {
        std::cerr << "spalm: MKL sparse output failed (status " << status
                  << ")" << std::endl;
        return SPALM_INTERNAL_ERROR;
    }
    out.storage = std::shared_ptr<void>(handle, baseline.free);
    const auto blocks_begin = std::chrono::high_resolution_clock::now();
    constexpr long long target = 1LL << 16;
    out.blocks.clear();
    out.entries = 0;
    for (int begin = 0; begin < M;) {
        int end = begin;
        long long entries = 0;
        while (end < M && entries < target) {
            entries += rows_end[end] - rows_start[end];
            ++end;
        }
        if (entries > 0) {
            out.blocks.push_back(
                {begin, end - begin, rows_start + begin, rows_end + begin, cols,
                 vals}
            );
        }
        out.entries += entries;
        begin = end;
    }
    print_time(
        "[mkl-sparse-output-blocks]",
        std::chrono::high_resolution_clock::now() - blocks_begin
    );
    std::cout << "[mkl-sparse-output] entries=" << out.entries
              << " blocks=" << out.blocks.size() << std::endl;
    return SPALM_OK;
}

// A decision-table cost times its kernel's weight (cost_weight_*), rounded
// and capped at the largest value; weight 1 returns the cost unchanged.
static unsigned long long weighted_cost(
    const unsigned long long cost, const double weight
) {
    if (weight == 1.0) return cost;
    const double x = std::nearbyint(weight * static_cast<double>(cost));
    if (x >= 18446744073709551615.0) return ULLONG_MAX;
    return x > 0 ? static_cast<unsigned long long>(x) : 0;
}

// Sides whose subrows the decision table multiplies with packed slices of
// the other side: R for SD, S for DS, both for DD or mixed tables. Only SS
// slice pairs: neither, and the slice length stays as configured.
static void used_algorithms(bool (&used)[4]) {
    for (bool &u : used) u = false;
    for (int left = 0; left <= num_buckets; ++left)
        for (int right = 0; right <= num_buckets; ++right)
            used[get_alg(left, right)] = true;
}

static void sparse_sides(bool &r_sparse, bool &s_sparse) {
    bool used[4];
    used_algorithms(used);
    r_sparse = used[DENSE_DENSE_LEFT_INNER_KERNEL] || used[DENSE_DENSE];
    s_sparse = used[DENSE_DENSE_RIGHT_INNER_KERNEL] || used[DENSE_DENSE];
}

// What dense_dense_matmul() will pack for this decision table
// (get_dense_packing_policy() there): with selective packing, R for DS and
// DD, S for SD and DD; R at 16 values per key when only DS reads it.
static slice_length_rule::packing dense_packing(
    const cpu_execution_options &options, const int r_height,
    const int s_height, const int tile_size_l2, const int tile_size_l3,
    const int threads
) {
    bool used[4];
    used_algorithms(used);
    const bool selective = options.selective_packing;
    slice_length_rule::packing p;
    p.pack_r = !selective || used[DENSE_DENSE] ||
               used[DENSE_DENSE_RIGHT_INNER_KERNEL];
    p.pack_s = !selective || used[DENSE_DENSE] ||
               used[DENSE_DENSE_LEFT_INNER_KERNEL];
    const bool bitmaps = !selective || used[DENSE_DENSE];
    p.r_stride =
        options.ds_aligned_panels && p.pack_r && !bitmaps ? 16 : r_height;
    p.s_stride = s_height;
    p.tile_size_l2 = tile_size_l2;
    p.tile_size_l3 = tile_size_l3;
    p.threads = threads;
    return p;
}

// auto_slice_length = 1: the slice length from exact counts of subrows and
// slices at 1, 2 and 4 times the configured length (slice_length_rule.h).
static int choose_slice_length_by_counts(
    const int configured, const int *Ri, const int *Rj, const int M,
    const int *Si, const int *Sj, const int N, const int K,
    const int r_height, const int s_height, const int threads,
    const slice_length_rule::params &rule,
    const slice_length_rule::packing &pack
) {
    bool r_sparse = false, s_sparse = false;
    sparse_sides(r_sparse, s_sparse);
    if (!r_sparse && !s_sparse) return configured;
    // No length beyond K: it would only pad a single slice column.
    const int levels =
        slice_length_rule::levels_within(configured, K, rule.max_levels);
    if (levels == 1) return configured;
    const char *sides = r_sparse && s_sparse ? "RS" : r_sparse ? "R" : "S";
    // Below the minimum cost (asl_min_cost, default kMinCost) nothing is
    // lengthened: inputs whose level-0 cost cannot reach it are not even
    // counted.
    const double bound = slice_length_rule::cost0_bound(
        Ri[M], M, r_height, Si[N], N, s_height, r_sparse, s_sparse,
        rule.subrow_write_entries
    );
    if (bound < rule.min_cost) {
        std::cout << "[slice-length] rule=counts configured=" << configured
                  << " chosen=" << configured << " sides=" << sides
                  << " cost0_bound=" << bound << " (not counted)" << std::endl;
        return configured;
    }
    const auto begin = std::chrono::high_resolution_clock::now();
    const auto r = slice_length_rule::count_columns(
        Ri, Rj, M, K, configured, levels, r_height, threads
    );
    const auto s = slice_length_rule::count_columns(
        Si, Sj, N, K, configured, levels, s_height, threads
    );
    const auto d =
        slice_length_rule::choose(r, s, r_sparse, s_sparse, rule, &pack);
    const int length = configured << d.level;
    print_time(
        "[slice-length-count]",
        std::chrono::high_resolution_clock::now() - begin
    );
    // cost: with packing, relative to the configured length; cost0: the
    // multiplication cost there (the minimum cost's measure); mult and
    // packed: per length, what find_thresholds.py replays the rule on for
    // other weights (10 digits, so the replay decides as the library does).
    auto list = [levels](const double *x, const int digits) {
        const auto saved = std::cout.precision(digits);
        for (int l = 0; l < levels; ++l) std::cout << (l ? "," : "") << x[l];
        std::cout.precision(saved);
    };
    std::cout << "[slice-length] rule=counts configured=" << configured
              << " chosen=" << length << " sides=" << sides
              << " entries_per_subrow=";
    list(d.per_subrow, 10);
    std::cout << " cost=";
    list(d.cost, 6);
    std::cout << " cost0=" << d.cost0 << " mult=";
    list(d.mult, 10);
    std::cout << " packed=";
    list(d.packed, 10);
    std::cout << " pack_value_entries=" << rule.pack_value_entries
              << std::endl;
    return length;
}

// auto_slice_length = 2, kept for comparison: the rule of 2026-10-27, from
// global density alone. It estimates the entries per subrow as if each row
// spread its entries evenly over K, and doubles the configured length, up to
// 2048 and at most K, while that estimate stays below 16 (tuned on the
// uniform encoder-1025 inputs, where the estimate is exact). Clustered rows
// look far sparser than they are, and the packed slices of the other side
// are not considered.
static int choose_slice_length_by_density(
    const int configured, const int *Ri, const int M, const int *Si,
    const int N, const int K
) {
    bool r_sparse = false, s_sparse = false;
    sparse_sides(r_sparse, s_sparse);
    if (!r_sparse && !s_sparse) return configured;
    // Stored entries per key of a nonempty row.
    auto density = [K](const int *row_ptrs, const int rows) {
        long long nonempty = 0;
        for (int r = 0; r < rows; ++r)
            nonempty += row_ptrs[r + 1] != row_ptrs[r];
        return nonempty ? double(row_ptrs[rows]) / (double(nonempty) * K) : 1.0;
    };
    double d = 1.0;
    if (r_sparse) d = std::min(d, density(Ri, M));
    if (s_sparse) d = std::min(d, density(Si, N));
    int length = configured;
    while (length < 2048 && length < K && d * std::min(length, K) < 16.0)
        length *= 2;
    std::cout << "[slice-length] rule=density configured=" << configured
              << " chosen=" << length
              << " entries_per_subrow=" << d * std::min(configured, K) << " -> "
              << d * std::min(length, K) << std::endl;
    return length;
}

// spalm_sparse_or_dense(): the caller's dense result, requested only when
// the decision needs one.
struct dense_result_request {
    spalm_dense_result_fn fn = nullptr;
    void *ctx = nullptr;
    bool called = false;
};

// log_candidate_work = 1: the inputs for comparing whole-call alternatives
// offline (candidate_work.h), printed after the MM cost lines and before the
// whole-call rules. Three decision tables: "table" as loaded (the rules have
// not changed it yet), "all_ss" (every slice pair SS) and "all_dense" (what
// set_to_all_dense() makes of it). For each, "bucket" is mm_check() itself,
// so "table bucket" repeats the MM cost lines, "nnz" counts each bucket's
// stored entries, and [candidate-pairs] gives the slice pairs per kernel and
// what the dense path would pack. The global table is restored before
// returning.
template <typename T>
static void log_candidate_work(
    const int *Ri, const int *Rj, const int M, const int N, const int K,
    ss_matrix<T> &R_ssm, ss_matrix<T> &S_ssm, const int tile_size_l2,
    const int tile_size_l3, const int threads,
    const cpu_execution_options &options
) {
    const auto begin = std::chrono::high_resolution_clock::now();
    const auto f =
        candidate_work::count_features(Ri, Rj, M, S_ssm, N, K, threads);
    unsigned long long r_slices, s_slices, pairs;
    candidate_work::slice_counts(R_ssm, S_ssm, r_slices, s_slices, pairs);
    std::cout << "[candidate-features] M=" << M << " N=" << N << " K=" << K
              << " sl=" << R_ssm.sl << " mr=" << R_ssm.sh
              << " nr=" << S_ssm.sh << " nnz_r=" << Ri[M]
              << " nnz_s=" << S_ssm.csr_mat.row_ptrs[N] << " flops=" << f.flops
              << " row_bound=" << f.row_bound << " r_slices=" << r_slices
              << " s_slices=" << s_slices << " slice_pairs=" << pairs
              << std::endl;
    auto round = [](double x) { return (unsigned long long) std::llround(x); };
    // The result entries with the large budget (the reference) and the
    // small one (what a decision could afford), each timed.
    for (const auto &[budget, timer] :
         {std::pair{candidate_work::kEntriesBudget, "[candidate-entries-time]"},
          std::pair{
              candidate_work::kEntriesSmallBudget,
              "[candidate-entries-small-time]"
          }}) {
        const auto entries_begin = std::chrono::high_resolution_clock::now();
        const auto e = candidate_work::estimate_entries(
            Ri, Rj, M, S_ssm, N, K, f, threads, budget
        );
        const auto entries_end = std::chrono::high_resolution_clock::now();
        std::cout << "[candidate-entries] budget=" << budget
                  << " estimate=" << round(e.entries) << " exact=" << e.exact
                  << " draws=" << e.draws << " rows=" << e.rows
                  << " long_rows=" << e.long_rows << " visits=" << e.visits
                  << std::endl;
        print_time(timer, entries_end - entries_begin);
    }
    const int nb = num_buckets;
    const std::vector<uint8_t> table(ss_upper.get(), ss_upper.get() + nb + 1);
    const std::vector<uint8_t> all_ss(nb + 1, (uint8_t) (nb + 1));
    const std::vector<uint8_t> all_dense(nb + 1, (uint8_t) 0);
    const std::pair<const char *, const std::vector<uint8_t> *> tables[] = {
        {"table", &table}, {"all_ss", &all_ss}, {"all_dense", &all_dense}
    };
    const auto plans = candidate_work::plan_counts(
        R_ssm, S_ssm, {table.data(), all_ss.data(), all_dense.data()}, threads
    );
    for (int t = 0; t < 3; ++t) {
        const auto &[name, up] = tables[t];
        std::copy(up->begin(), up->end(), ss_upper.get());
        unsigned long long c[4] = {};
        for (int col = 0; col < R_ssm.s_csc_mat.cols; ++col) {
            unsigned long long ss = 0, sd = 0, ds = 0, dd = 0;
            mm_check(col, R_ssm, S_ssm, ss, sd, ds, dd);
            c[0] += ss;
            c[1] += sd;
            c[2] += ds;
            c[3] += dd;
        }
        const auto &w = plans[t].nnz;
        const auto &p = plans[t].p;
        const auto pack = candidate_work::packing_for(
            up->data(), options.selective_packing, options.ds_aligned_panels,
            R_ssm.sh
        );
        std::cout << "[candidate-work] " << name << " bucket ss=" << c[0]
                  << " sd=" << c[1] << " ds=" << c[2] << " dd=" << c[3]
                  << std::endl;
        std::cout << "[candidate-work] " << name << " nnz ss=" << round(w.ss)
                  << " sd=" << round(w.sd) << " ds=" << round(w.ds)
                  << " dd=" << round(w.dd) << std::endl;
        std::cout << "[candidate-pairs] " << name << " ss=" << p.ss
                  << " sd=" << p.sd << " ds=" << p.ds << " dd=" << p.dd
                  << " pack_r=" << pack.pack_r << " pack_s=" << pack.pack_s
                  << " bitmaps=" << pack.bitmaps
                  << " r_stride=" << pack.r_stride
                  << " fixed_alg=" << pack.fixed_alg << std::endl;
    }
    std::copy(table.begin(), table.end(), ss_upper.get());
    const auto d = candidate_work::dense_schedule_counts(
        R_ssm, S_ssm, tile_size_l2, tile_size_l3, threads
    );
    std::cout << "[candidate-dense-schedule] tile_size_l2=" << tile_size_l2
              << " tile_size_l3=" << tile_size_l3 << " threads=" << threads
              << " value_bytes=" << sizeof(T) << " r_packs=" << d.r_packs
              << " r_pack_entries=" << d.r_pack_entries
              << " s_packs=" << d.s_packs
              << " s_pack_entries=" << d.s_pack_entries
              << " tasks=" << d.tasks << std::endl;
    print_time(
        "[candidate-work-time]",
        std::chrono::high_resolution_clock::now() - begin
    );
}

// Accumulates R * S^T into the caller's zero-initialized ret; with
// options.result_zeroed false, ret holds garbage and every element is written
// (see cpu_execution_options).
// With dense_request (spalm_sparse_or_dense), ret starts null and sparse_out
// is set: a call that runs entirely on the SS path (or FORCE_MKL_SPARSE)
// keeps the row blocks; any other decision gets ret from the caller and
// clears sparse_out.
template <typename T>
int spalm_matmul(
    const int type, int *Ri, int *Rj, T *Rv, int *Si, int *Sj, T *Sv,
    const int M, const int K_input, const int N, const int slice_length,
    const int tile_size_l2, const int tile_size_l3, bool sparse_result, T *ret,
    int *reti, int *retj, T *retv, const int threads,
    const cpu_execution_options &options,
    spalm_sparse_result *sparse_out = nullptr,
    dense_result_request *dense_request = nullptr
) {
    // The join dimension every later stage uses; key compaction shrinks it.
    int K = K_input;
    // An uninitialized result (result_zeroed = 0) is overwritten by the
    // forced MKL backends; every other path accumulates into it, so it is
    // zeroed here, and the zero-work exits zero it for the MKL backends.
    const bool forced_mkl = force_mkl_dense || force_mkl_sparse;
    const size_t result_bytes = (size_t) M * N * sizeof(T);
    // Sparse output (sparse_out): no dense result; every exit before the
    // multiplication leaves it without blocks.
    if (ret && !options.result_zeroed && !forced_mkl) {
        zero_result(ret, result_bytes, threads);
    }
    auto zero_work = [&](const char *why) {
        if (ret && !options.result_zeroed && forced_mkl) {
            zero_result(ret, result_bytes, threads);
        }
        std::cout << "[backend] zero-work (" << why << ")" << std::endl;
    };
    // spalm_sparse_or_dense(): this decision needs the dense result. It is
    // zeroed here when the caller left it uninitialized, as at the top,
    // except for the forced MKL backends, which overwrite it.
    auto request_dense_result = [&](const char *why) {
        const auto begin = std::chrono::high_resolution_clock::now();
        ret = static_cast<T *>(dense_request->fn(dense_request->ctx));
        dense_request->called = true;
        print_time(
            "[dense-result-request]",
            std::chrono::high_resolution_clock::now() - begin
        );
        if (!ret) {
            std::cerr << "spalm: the caller could not provide the dense result"
                      << std::endl;
            return false;
        }
        sparse_out = nullptr;
        if (!options.result_zeroed && !forced_mkl) {
            zero_result(ret, result_bytes, threads);
        }
        std::cout << "[result-choice] dense (" << why << ")" << std::endl;
        return true;
    };
    // An empty dimension, or no stored entry on a side, contributes nothing:
    // no CSR copies, slices, packing buffers or MKL call are needed.
    if (M == 0 || N == 0 || K == 0 || Ri[M] == 0 || Si[N] == 0) {
        zero_work("empty input");
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
        zero_work("all-zero input");
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
            zero_work("no common keys");
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
        int status = SPALM_INVALID_ARGUMENT;
        if (sparse_out && dense_request &&
            forced_backend == spalm_dispatch::Backend::MklDense &&
            !request_dense_result("forced mkl-dense")) {
            return SPALM_INTERNAL_ERROR;
        }
        if (!sparse_out) {
            status = run_mkl_backend(
                forced_backend, type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, ret,
                threads
            );
        } else if (forced_backend == spalm_dispatch::Backend::MklSparse) {
            if (dense_request) {
                std::cout << "[result-choice] sparse (forced mkl-sparse)"
                          << std::endl;
            }
            status = run_mkl_sparse_output(
                type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, threads, *sparse_out
            );
        } else {
            std::cerr << "spalm: MKL dense has no sparse output" << std::endl;
        }
        print_time(
            "[forced-backend-total]",
            std::chrono::high_resolution_clock::now() - start
        );
        return status;
    }

    // type is validated by spalm(): 0 = float, 1 = double.
    const int R_ik_slice_height = 12;
    const int S_ik_slice_height = type == 0 ? 32 : 16;

    int sl = slice_length;
    if (options.auto_slice_length == 1) {
        slice_length_rule::params rule;
        rule.subrow_write_entries = options.asl_subrow_write_entries;
        rule.min_cost_ratio = options.asl_min_cost_ratio;
        rule.min_cost = options.asl_min_cost;
        rule.pack_value_entries = options.asl_pack_value_entries;
        rule.max_levels = options.asl_max_levels;
        sl = choose_slice_length_by_counts(
            slice_length, Ri, Rj, M, Si, Sj, N, K, R_ik_slice_height,
            S_ik_slice_height, threads, rule,
            dense_packing(
                options, R_ik_slice_height, S_ik_slice_height, tile_size_l2,
                tile_size_l3, threads
            )
        );
    } else if (options.auto_slice_length == 2) {
        sl = choose_slice_length_by_density(slice_length, Ri, M, Si, N, K);
    }

    ss_matrix<T> R_ssm;
    ss_matrix<T> S_ssm;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end,
        total_start, e2e_start;
    e2e_start = std::chrono::high_resolution_clock::now();
    csr_to_ss_matrix(
        Ri, Rj, Rv, M, K, sl, R_ik_slice_height, R_ssm, threads, 1
    );
    csr_to_ss_matrix(
        Si, Sj, Sv, N, K, sl, S_ik_slice_height, S_ssm, threads, 2
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
    if (options.log_candidate_work) {
        log_candidate_work(
            Ri, Rj, M, N, K, R_ssm, S_ssm, tile_size_l2, tile_size_l3, threads,
            options
        );
    }

    if (total_dense_mm_cost) {
        do_dense_mm = true;
    }

    // The two rules below compare the costs weighted per kernel
    // (cost_weight_*, default 1: the raw costs, exactly as before).
    const unsigned long long weighted_sparse_mm_cost =
        weighted_cost(sparse_mm_cost, options.cost_weight_ss);
    unsigned long long weighted_dense_mm_cost = 0;
    for (const auto [cost, weight] :
         {std::pair{sparse_dense_mm_cost, options.cost_weight_sd},
          std::pair{dense_sparse_mm_cost, options.cost_weight_ds},
          std::pair{dense_dense_mm_cost, options.cost_weight_dd}}) {
        const unsigned long long w = weighted_cost(cost, weight);
        weighted_dense_mm_cost = w > ULLONG_MAX - weighted_dense_mm_cost
                                     ? ULLONG_MAX
                                     : weighted_dense_mm_cost + w;
    }
    if (options.cost_weight_ss != 1.0 || options.cost_weight_sd != 1.0 ||
        options.cost_weight_ds != 1.0 || options.cost_weight_dd != 1.0) {
        std::cout << "[weighted-cost] sparse=" << weighted_sparse_mm_cost
                  << " dense=" << weighted_dense_mm_cost << std::endl;
    }

    // Float, as before (the product of an integer cost and a float).
    const float sparse_ratio_threshold = (float) options.auto_sparse_ratio;

    if (weighted_dense_mm_cost < dense_mm_threshold && total_dense_mm_cost) {
        do_dense_mm = false;
        do_sparse_mm = true;
    } else if (weighted_sparse_mm_cost <
               weighted_dense_mm_cost * sparse_ratio_threshold) {
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
        total_dense_mm_cost >
            (unsigned long long) M * K * N * options.auto_mkl_dense_ratio,
        do_sparse_mm && !do_dense_mm && options.auto_mkl_sparse_fallback
    );
    // Sparse output exists for the full SS path only (init_method sparse:
    // no dense slice pair, no fallback). spalm_sparse_or_dense() writes any
    // other decision to the caller's dense result instead.
    if (sparse_out &&
        (backend != spalm_dispatch::Backend::Internal || do_dense_mm)) {
        if (!dense_request) {
            std::cerr << "spalm: sparse output needs every slice pair on the "
                         "SS path (init_method sparse)"
                      << std::endl;
            return SPALM_INVALID_ARGUMENT;
        }
        const char *why =
            backend == spalm_dispatch::Backend::MklDense    ? "mkl-dense fallback"
            : backend == spalm_dispatch::Backend::MklSparse ? "mkl-sparse fallback"
            : do_sparse_mm                                  ? "mixed"
                                                            : "all dense";
        if (!request_dense_result(why)) return SPALM_INTERNAL_ERROR;
    } else if (sparse_out && dense_request) {
        std::cout << "[result-choice] sparse (all SS)" << std::endl;
    }
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
                R_ssm, S_ssm, ret, sparse_result, threads, !do_dense_mm,
                options, sparse_out
            );
        }

        if (do_dense_mm) {
            dense_dense_matmul(
                R_ssm, S_ssm, tile_size_l2, tile_size_l3, ret, sparse_result,
                threads, options
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

// Every key spalm() reads, and the keys the DuckDB operator reads itself
// (it forwards all of spalm.conf except so_dir). Anything else is listed on
// a "[conf-unused]" line, so a misspelled key shows in the log instead of
// being ignored silently.
static constexpr const char *kKnownConfKeys[] = {
    // library
    "init_method", "bucket_method", "threads", "prune_zeros", "compact_keys",
    "selective_packing", "ss_contiguous", "sd_ds_full_rows",
    "ss_column_blocks", "ds_aligned_panels", "result_zeroed",
    "auto_slice_length", "ss_output_page_entries", "slice_length",
    "tile_size_l2", "tile_size_l3", "cost_weight_ss", "cost_weight_sd",
    "cost_weight_ds", "cost_weight_dd", "auto_sparse_ratio",
    "auto_mkl_dense_ratio", "auto_mkl_sparse_fallback", "dense_mm_threshold",
    "thresholds_file", "asl_min_cost", "asl_subrow_write_entries",
    "asl_min_cost_ratio", "asl_pack_value_entries", "asl_max_levels",
    "r_block_huge_bytes",
    "s_blocks_huge_bytes", "ss_column_block_min_cols",
    "ss_column_block_cache_divisor", "log_candidate_work",
    "ss_page_huge_entries", "calibration_file",
    // DuckDB operator (duckdb-spalm)
    "result_base_offset", "result_compact_rows", "result_fresh_mapping",
    "result_huge_pages", "result_parallel_zero", "result_prefault",
    "result_mkl_skip_zero", "so_dir",
};

// One line with the effective settings of this call (after the thresholds
// file, the variants of init_method and the tile_sizes.txt defaults), and
// one with the keys nobody reads, if any.
static void print_conf(
    const cpu_execution_options &o, const char *init_method,
    const char *bucket_method, const char *thresholds_file,
    const int slice_length, const int tile_size_l2, const int tile_size_l3,
    const int n, const char **keys
) {
    std::cout << "[conf] init_method=" << (init_method[0] ? init_method : "auto")
              << " bucket_method="
              << (bucket_method[0] ? bucket_method : "linear")
              << " thresholds_file="
              << (thresholds_file[0] ? thresholds_file : "default")
              << " dense_mm_threshold=" << dense_mm_threshold
              << " fallback=" << fallback
              << " auto_sparse_ratio=" << o.auto_sparse_ratio
              << " auto_mkl_dense_ratio=" << o.auto_mkl_dense_ratio
              << " auto_mkl_sparse_fallback=" << o.auto_mkl_sparse_fallback
              << " cost_weight_ss=" << o.cost_weight_ss
              << " cost_weight_sd=" << o.cost_weight_sd
              << " cost_weight_ds=" << o.cost_weight_ds
              << " cost_weight_dd=" << o.cost_weight_dd
              << " slice_length=" << slice_length
              << " auto_slice_length=" << o.auto_slice_length
              << " asl_min_cost=" << o.asl_min_cost
              << " asl_subrow_write_entries=" << o.asl_subrow_write_entries
              << " asl_min_cost_ratio=" << o.asl_min_cost_ratio
              << " asl_pack_value_entries=" << o.asl_pack_value_entries
              << " asl_max_levels=" << o.asl_max_levels
              << " tile_size_l2=" << tile_size_l2
              << " tile_size_l3=" << tile_size_l3
              << " prune_zeros=" << o.prune_zeros
              << " compact_keys=" << o.compact_keys
              << " selective_packing=" << o.selective_packing
              << " sd_ds_full_rows=" << o.sd_ds_full_rows
              << " ds_aligned_panels=" << o.ds_aligned_panels
              << " r_block_huge_bytes=" << o.r_block_huge_bytes
              << " s_blocks_huge_bytes=" << o.s_blocks_huge_bytes
              << " ss_page_huge_entries=" << o.ss_page_huge_entries
              << " ss_contiguous=" << o.ss_contiguous
              << " ss_column_blocks=" << o.ss_column_blocks
              << " ss_column_block_min_cols=" << o.ss_column_block_min_cols
              << " ss_column_block_cache_divisor="
              << o.ss_column_block_cache_divisor
              << " ss_output_page_entries=" << o.ss_output_page_entries
              << " result_zeroed=" << o.result_zeroed
              << " log_candidate_work=" << o.log_candidate_work << std::endl;
    bool any = false;
    for (int i = 0; i < n; i++) {
        bool known = false;
        for (const char *k : kKnownConfKeys)
            if (strcmp(keys[i], k) == 0) known = true;
        if (known) continue;
        std::cout << (any ? "," : "[conf-unused] keys=") << keys[i];
        any = true;
    }
    if (any) std::cout << std::endl;
}

// One line on calibration.txt: the file read (or none), the keys this call
// took from it (those not given to spalm()), and keys it does not use.
static void print_calibration(
    const calibration_values &c, const int n, const char **keys
) {
    std::cout << "[calibration] file=" << (c.path.empty() ? "none" : c.path)
              << " used=";
    bool any = false;
    for (const auto &[key, value] : c.values) {
        bool given = false;
        for (int i = 0; i < n; i++) given |= strcmp(keys[i], key.c_str()) == 0;
        if (given) continue;
        std::cout << (any ? "," : "") << key;
        any = true;
    }
    if (!any) std::cout << "-";
    if (!c.ignored.empty()) {
        std::cout << " ignored=";
        for (size_t i = 0; i < c.ignored.size(); i++)
            std::cout << (i ? "," : "") << c.ignored[i];
    }
    std::cout << std::endl;
}

// sparse_out: the result goes there as row blocks (spalm_sparse()); ret
// is not used. dense_request (spalm_sparse_or_dense()): any init_method, and
// a decision that needs the dense result gets it from the caller.
static int spalm_impl(
    int type, int *Ri, int *Rj, void *Rv, int *Si, int *Sj, void *Sv,
    const int M, const int K, const int N, bool sparse_result, void *ret,
    int *reti, int *retj, void *retv, int n, const char **keys,
    const char **vals, spalm_sparse_result *sparse_out = nullptr,
    dense_result_request *dense_request = nullptr
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
    // Sparse output: SPALM SS (init_method sparse) and MKL sparse-sparse
    // (FORCE_MKL_SPARSE) only, unless the dense result can be requested. The
    // first init_method counts, as below.
    if (sparse_out && !dense_request) {
        const char *method = "";
        for (int i = 0; i < n; i++) {
            if (strcmp(keys[i], "init_method") == 0) {
                method = vals[i];
                break;
            }
        }
        if (strcmp(method, "sparse") != 0 &&
            strcmp(method, "FORCE_MKL_SPARSE") != 0) {
            return invalid(
                "sparse output needs init_method sparse or FORCE_MKL_SPARSE"
            );
        }
    }
    // A product with K = 0 is all zeros: an uninitialized result
    // (result_zeroed = 0) is zeroed, an initialized one left as is; a
    // sparse result has no blocks.
    if (M == 0 || K == 0 || N == 0) {
        if (sparse_out) return SPALM_OK;
        bool zeroed = true;
        for (int i = 0; i < n; i++)
            if (strcmp(keys[i], "result_zeroed") == 0) zeroed = atoi(vals[i]);
        if (!zeroed && M > 0 && N > 0) {
            if (!ret) return invalid("null result pointer");
            zero_result(
                ret,
                (size_t) M * N * (type == 0 ? sizeof(float) : sizeof(double)), 1
            );
        }
        return SPALM_OK;
    }
    if (!Ri || !Rj || !Rv || !Si || !Sj || !Sv || (!ret && !sparse_out)) {
        return invalid("null matrix or result pointer");
    }
    // The dense kernels address the result with 32-bit offsets, including
    // gather/scatter indices up to 12 rows and 32 columns past a tile start.
    // Kept for sparse output too, which is not tested beyond it yet.
    if ((long long) (M + 16) * (N + 32) > INT_MAX) {
        return invalid("M * N is too large for 32-bit result offsets");
    }

    auto get_key = [&](const char *key) -> const char * {
        for (int i = 0; i < n; i++)
            if (strcmp(keys[i], key) == 0) return vals[i];
        return nullptr;
    };
    // Per-machine defaults of the calibrated keys (slice_decision.h): a key
    // given here wins, then calibration.txt, then the built-in default.
    calibration_values calibration;
    {
        const char *file = get_key("calibration_file");
        if (!read_calibration(type, "cpu", file ? file : "", calibration)) {
            return SPALM_CONFIG_ERROR;
        }
    }
    auto get_str = [&](const char *key, const char *def) -> const char * {
        if (const char *v = get_key(key)) return v;
        if (const char *v = calibration.find(key)) return v;
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
    options.selective_packing = get_int("selective_packing", 1) != 0;
    options.ss_contiguous = get_int("ss_contiguous", 1) != 0;
    options.sd_ds_full_rows = get_int("sd_ds_full_rows", 1) != 0;
    options.ss_column_blocks = get_int("ss_column_blocks", -1);
    options.ds_aligned_panels = get_int("ds_aligned_panels", 1) != 0;
    // An explicit slice_length is used as given.
    options.result_zeroed = get_int("result_zeroed", 1) != 0;
    options.log_candidate_work = get_int("log_candidate_work", 0) != 0;
    options.auto_slice_length = get_str("slice_length", nullptr) == nullptr
                                    ? get_int("auto_slice_length", 1)
                                    : 0;
    {
        const char *page = get_str("ss_output_page_entries", "0");
        options.ss_output_page_entries = atoll(page);
    }
    if (options.auto_slice_length < 0 || options.auto_slice_length > 2) {
        return invalid("auto_slice_length must be 0, 1 (counts) or 2 (density)");
    }
    if (options.ss_column_blocks < -1) {
        return invalid("ss_column_blocks must be -1 (auto), 0, 1 or a count");
    }

    // Tunables (execution_options.h), read strictly: the whole value must be
    // a number (an integer where one is needed; 5e8 is accepted), and values
    // out of range are rejected.
    std::string bad_number;
    auto parse_number = [&](const char *key, double &x) -> bool {
        const char *v = get_str(key, nullptr);
        if (!v) return false;
        char *end = nullptr;
        errno = 0;
        x = std::strtod(v, &end);
        while (end && *end && std::isspace(static_cast<unsigned char>(*end)))
            ++end;
        if (end == v || *end != '\0' || errno == ERANGE || !std::isfinite(x)) {
            if (bad_number.empty()) bad_number = key;
            return false;
        }
        return true;
    };
    auto get_double = [&](const char *key, double def) -> double {
        double x = 0;
        return parse_number(key, x) ? x : def;
    };
    auto get_ll = [&](const char *key, long long def) -> long long {
        const char *v = get_str(key, nullptr);
        if (!v) return def;
        // Plain integers exactly; otherwise an integral number such as 5e8.
        char *end = nullptr;
        errno = 0;
        const long long i = std::strtoll(v, &end, 10);
        while (end && *end && std::isspace(static_cast<unsigned char>(*end)))
            ++end;
        if (end != v && *end == '\0' && errno != ERANGE) return i;
        double x = 0;
        if (!parse_number(key, x)) return def;
        if (x != std::floor(x) || std::fabs(x) >= 9.2e18) {
            if (bad_number.empty()) bad_number = key;
            return def;
        }
        return static_cast<long long>(x);
    };
    options.cost_weight_ss = get_double("cost_weight_ss", options.cost_weight_ss);
    options.cost_weight_sd = get_double("cost_weight_sd", options.cost_weight_sd);
    options.cost_weight_ds = get_double("cost_weight_ds", options.cost_weight_ds);
    options.cost_weight_dd = get_double("cost_weight_dd", options.cost_weight_dd);
    options.auto_sparse_ratio =
        get_double("auto_sparse_ratio", options.auto_sparse_ratio);
    options.auto_mkl_dense_ratio =
        get_double("auto_mkl_dense_ratio", options.auto_mkl_dense_ratio);
    const long long mkl_sparse_fallback = get_ll("auto_mkl_sparse_fallback", 1);
    options.auto_mkl_sparse_fallback = mkl_sparse_fallback != 0;
    // -1: the thresholds file's value.
    const long long dense_mm_threshold_override = get_ll("dense_mm_threshold", -1);
    // Empty: parameters/cpu/<type>/<buckets>/decision_thresholds.txt.
    const char *thresholds_file = get_str("thresholds_file", "");
    options.asl_min_cost = get_double("asl_min_cost", options.asl_min_cost);
    options.asl_subrow_write_entries =
        get_double("asl_subrow_write_entries", options.asl_subrow_write_entries);
    options.asl_min_cost_ratio =
        get_double("asl_min_cost_ratio", options.asl_min_cost_ratio);
    options.asl_pack_value_entries =
        get_double("asl_pack_value_entries", options.asl_pack_value_entries);
    const long long asl_max_levels =
        get_ll("asl_max_levels", options.asl_max_levels);
    options.r_block_huge_bytes =
        get_ll("r_block_huge_bytes", options.r_block_huge_bytes);
    options.s_blocks_huge_bytes =
        get_ll("s_blocks_huge_bytes", options.s_blocks_huge_bytes);
    options.ss_page_huge_entries =
        get_ll("ss_page_huge_entries", options.ss_page_huge_entries);
    const long long column_block_min_cols =
        get_ll("ss_column_block_min_cols", options.ss_column_block_min_cols);
    const long long column_block_cache_divisor = get_ll(
        "ss_column_block_cache_divisor", options.ss_column_block_cache_divisor
    );
    if (!bad_number.empty()) {
        std::cerr << "spalm: " << bad_number << " is not a valid number";
        if (!get_key(bad_number.c_str()) && calibration.find(bad_number.c_str()))
            std::cerr << " (in " << calibration.path << ")";
        std::cerr << std::endl;
        return SPALM_INVALID_ARGUMENT;
    }
    if (options.cost_weight_ss < 0 || options.cost_weight_sd < 0 ||
        options.cost_weight_ds < 0 || options.cost_weight_dd < 0) {
        return invalid("cost_weight_ss/sd/ds/dd must be at least 0");
    }
    if (options.auto_sparse_ratio < 0 || options.auto_mkl_dense_ratio < 0) {
        return invalid("auto_sparse_ratio and auto_mkl_dense_ratio must be at least 0");
    }
    if (mkl_sparse_fallback != 0 && mkl_sparse_fallback != 1) {
        return invalid("auto_mkl_sparse_fallback must be 0 or 1");
    }
    if (dense_mm_threshold_override < -1) {
        return invalid("dense_mm_threshold must be -1 (from the thresholds file) or at least 0");
    }
    if (options.asl_min_cost < 0 || options.asl_subrow_write_entries < 0 ||
        options.asl_min_cost_ratio < 0 || options.asl_pack_value_entries < 0) {
        return invalid(
            "asl_min_cost, asl_subrow_write_entries, asl_min_cost_ratio and "
            "asl_pack_value_entries must be at least 0"
        );
    }
    if (asl_max_levels < 1 || asl_max_levels > slice_length_rule::kMaxLevels) {
        return invalid("asl_max_levels must be 1, 2 or 3");
    }
    options.asl_max_levels = static_cast<int>(asl_max_levels);
    if (options.r_block_huge_bytes < -1 || options.s_blocks_huge_bytes < -1 ||
        options.ss_page_huge_entries < -1) {
        return invalid(
            "r_block_huge_bytes, s_blocks_huge_bytes and ss_page_huge_entries "
            "must be -1 (never) or at least 0"
        );
    }
    if (column_block_min_cols < 1 || column_block_min_cols > INT_MAX ||
        column_block_cache_divisor < 1 || column_block_cache_divisor > INT_MAX) {
        return invalid(
            "ss_column_block_min_cols and ss_column_block_cache_divisor must "
            "be at least 1"
        );
    }
    options.ss_column_block_min_cols = static_cast<int>(column_block_min_cols);
    options.ss_column_block_cache_divisor =
        static_cast<int>(column_block_cache_divisor);

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
    bool read_success = init_decision_thresholds(
        init_method, type, "cpu", thresholds_file, dense_mm_threshold_override
    );
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
    print_conf(
        options, init_method, bucket_method, thresholds_file, slice_length,
        tile_size_l2, tile_size_l3, n, keys
    );
    print_calibration(calibration, n, keys);
    // std::cout << "Tile size join dim: " << tile_size_join_dim << std::endl;
    std::cout << "Tile size L2: " << tile_size_l2 << std::endl;
    std::cout << "Tile size L3: " << tile_size_l3 << std::endl;
    if (sparse_out && dense_request) {
        std::cout << "Writing to sparse result (row blocks), or to the "
                     "caller's dense result if the decision needs one"
                  << std::endl;
    } else if (sparse_out) {
        std::cout << "Writing to sparse result (row blocks)" << std::endl;
    } else if (sparse_result) {
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
            (float *) ret, reti, retj, (float *) retv, threads, options,
            sparse_out, dense_request
        );
    }
    return spalm_matmul(
        type, Ri, Rj, (double *) Rv, Si, Sj, (double *) Sv, M, K, N,
        slice_length, tile_size_l2, tile_size_l3, sparse_result, (double *) ret,
        reti, retj, (double *) retv, threads, options, sparse_out,
        dense_request
    );
}

// Decision thresholds and bucket settings are process-global, reset and
// freed by every call: spalm() and spalm_sparse() are serialized so
// concurrent queries in one process cannot overwrite or free each other's
// state mid-run. Never lets an exception cross the C ABI. Exceptions thrown
// inside OpenMP regions still terminate the process; this covers the rest
// (e.g. std::bad_alloc from the slice or MKL input allocations).
// One lock for both entry points (a static inside the template would be
// one per instantiation).
static std::mutex &spalm_call_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename F> static int serialized_call(F call) {
    std::lock_guard<std::mutex> lock(spalm_call_mutex());
    try {
        const auto begin = std::chrono::high_resolution_clock::now();
        const int status = call();
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

// Returns a spalm_status; the result is valid only for 0 (SPALM_OK).
// Callers that still declare this as returning void keep working.
extern "C" int spalm(
    int type, int *Ri, int *Rj, void *Rv, int *Si, int *Sj, void *Sv,
    const int M, const int K, const int N, bool sparse_result, void *ret,
    int *reti, int *retj, void *retv, int n, const char **keys,
    const char **vals
) {
    return serialized_call([&] {
        return spalm_impl(
            type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, sparse_result, ret, reti,
            retj, retv, n, keys, vals
        );
    });
}

// C = R * S^T as row blocks (spalm_sparse_result.h) instead of a dense
// result, for init_method sparse (SPALM SS) and FORCE_MKL_SPARSE (MKL
// sp2m); other methods are rejected. Same inputs and options as spalm().
// On SPALM_OK *result is a handle for spalm_sparse_result_blocks(), to be
// released with spalm_sparse_result_free(); otherwise it is null.
extern "C" int spalm_sparse(
    int type, int *Ri, int *Rj, void *Rv, int *Si, int *Sj, void *Sv,
    const int M, const int K, const int N, int n, const char **keys,
    const char **vals, void **result
) {
    if (!result) {
        std::cerr << "spalm: null result handle pointer" << std::endl;
        return SPALM_INVALID_ARGUMENT;
    }
    *result = nullptr;
    std::unique_ptr<spalm_sparse_result> out;
    const int status = serialized_call([&] {
        out = std::make_unique<spalm_sparse_result>();
        return spalm_impl(
            type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, false, nullptr, nullptr,
            nullptr, nullptr, n, keys, vals, out.get()
        );
    });
    if (status == SPALM_OK) *result = out.release();
    return status;
}

// C = R * S^T as row blocks when the whole call runs on the SS path with
// sparse output (every slice pair SS and no MKL fallback, e.g. init_method
// no_fallback deciding "all SS", or init_method sparse) or on
// FORCE_MKL_SPARSE; otherwise into a dense M x N result that the caller
// provides on request: dense_result(ctx) is called at most once, after the
// dispatch decision (see spalm_dense_result_fn). Any init_method; same
// inputs and options as spalm(). A null from dense_result fails the call
// with SPALM_INTERNAL_ERROR. On SPALM_OK, *result is null if dense_result
// was called (the result is in that buffer), else a handle as from
// spalm_sparse() (no blocks for a zero product); on failure it is null.
extern "C" int spalm_sparse_or_dense(
    int type, int *Ri, int *Rj, void *Rv, int *Si, int *Sj, void *Sv,
    const int M, const int K, const int N, int n, const char **keys,
    const char **vals, spalm_dense_result_fn dense_result, void *ctx,
    void **result
) {
    if (!result || !dense_result) {
        std::cerr << "spalm: null result handle pointer or dense result "
                     "function"
                  << std::endl;
        if (result) *result = nullptr;
        return SPALM_INVALID_ARGUMENT;
    }
    *result = nullptr;
    std::unique_ptr<spalm_sparse_result> out;
    dense_result_request request{dense_result, ctx};
    const int status = serialized_call([&] {
        out = std::make_unique<spalm_sparse_result>();
        return spalm_impl(
            type, Ri, Rj, Rv, Si, Sj, Sv, M, K, N, false, nullptr, nullptr,
            nullptr, nullptr, n, keys, vals, out.get(), &request
        );
    });
    if (status == SPALM_OK && !request.called) *result = out.release();
    return status;
}

// Layout version of spalm_csr_block (SPALM_SPARSE_RESULT_ABI).
extern "C" int spalm_sparse_result_abi() { return SPALM_SPARSE_RESULT_ABI; }

// The blocks of a spalm_sparse() result; *count gets their number.
extern "C" const spalm_csr_block *
spalm_sparse_result_blocks(void *result, long long *count) {
    const auto *r = static_cast<const spalm_sparse_result *>(result);
    if (count) *count = r ? (long long) r->blocks.size() : 0;
    return r && !r->blocks.empty() ? r->blocks.data() : nullptr;
}

// Entries over all blocks (MKL: explicit zeros included).
extern "C" long long spalm_sparse_result_entries(void *result) {
    const auto *r = static_cast<const spalm_sparse_result *>(result);
    return r ? r->entries : 0;
}

extern "C" void spalm_sparse_result_free(void *result) {
    delete static_cast<spalm_sparse_result *>(result);
}

// Present when spalm() accepts result_zeroed = 0 (an uninitialized result,
// every element of which it then writes); the caller looks the symbol up.
extern "C" int spalm_accepts_unzeroed_result() { return 1; }
