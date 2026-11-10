// Runs the built library (AVX-512 kernels) through the C API and compares
// every forced SPALM method with a scalar C = R * S^T reference, for each
// combination of the execution switches. Values are small integers, so all
// sums are exact and results must match bit for bit.
//
// Build/run from the repository root (needs bin/spalm.so and parameters/):
//   make test-kernels-native
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using spalm_fn = int (*)(
    int, int *, int *, void *, int *, int *, void *, int, int, int, bool,
    void *, int *, int *, void *, int, const char **, const char **
);
#include "spalm_sparse_result.h"
using spalm_sparse_fn = int (*)(
    int, int *, int *, void *, int *, int *, void *, int, int, int, int,
    const char **, const char **, void **
);
using spalm_sparse_blocks_fn = const spalm_csr_block *(*) (void *, long long *);
using spalm_sparse_free_fn = void (*)(void *);
struct sparse_api {
    spalm_sparse_fn run;
    spalm_sparse_blocks_fn blocks;
    spalm_sparse_free_fn free;
};
using spalm_sparse_or_dense_fn = int (*)(
    int, int *, int *, void *, int *, int *, void *, int, int, int, int,
    const char **, const char **, spalm_dense_result_fn, void *, void **
);

struct csr {
    int rows = 0;
    std::vector<int> ptr{0}, col;
    std::vector<double> val;
};

// Row r gets keys from `pick(r, k)`; keys are strictly increasing.
template <typename Pick>
static csr make_csr(int rows, int K, Pick pick, std::mt19937 &rng) {
    csr m;
    m.rows = rows;
    for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < K; ++k) {
            if (!pick(r, k)) continue;
            m.col.push_back(k);
            int v = int(rng() % 7) - 3;
            m.val.push_back(v ? v : 4);
        }
        m.ptr.push_back(int(m.col.size()));
    }
    return m;
}

// Rows with the hot keys except the second, and the first one twice: as many
// entries as the hot keys, but not every key (a full-row proof by length
// alone would be wrong).
static csr
make_repeated(int rows, const std::vector<int> &hot, std::mt19937 &rng) {
    csr m;
    m.rows = rows;
    for (int r = 0; r < rows; ++r) {
        for (size_t h = 0; h < hot.size(); ++h) {
            if (h == 1) continue;
            const int copies = h == 0 ? 2 : 1;
            for (int c = 0; c < copies; ++c) {
                m.col.push_back(hot[h]);
                int v = int(rng() % 7) - 3;
                m.val.push_back(v ? v : 4);
            }
        }
        m.ptr.push_back(int(m.col.size()));
    }
    return m;
}

template <typename T>
static std::vector<T> reference(const csr &R, const csr &S, int K) {
    std::vector<T> out(size_t(R.rows) * S.rows, T(0));
    std::vector<double> row(K);
    for (int i = 0; i < R.rows; ++i) {
        std::fill(row.begin(), row.end(), 0.0);
        for (int p = R.ptr[i]; p < R.ptr[i + 1]; ++p) row[R.col[p]] += R.val[p];
        for (int j = 0; j < S.rows; ++j) {
            double acc = 0;
            for (int q = S.ptr[j]; q < S.ptr[j + 1]; ++q)
                acc += row[S.col[q]] * S.val[q];
            out[size_t(i) * S.rows + j] = T(acc);
        }
    }
    return out;
}

// unzeroed: the output starts as NaN and is passed with result_zeroed = 0,
// so every element must be written; otherwise it starts at 1 and the
// kernels must add to it. With `log`, the library's stdout is kept there.
template <typename T>
static bool run_case(
    spalm_fn spalm, const char *label, const csr &R, const csr &S, int K,
    std::vector<std::pair<std::string, std::string>> options,
    bool unzeroed = false, std::string *log = nullptr
) {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    std::vector<T> rv(R.val.begin(), R.val.end()),
        sv(S.val.begin(), S.val.end());
    std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
    // Non-null data() for inputs without entries, as from the operator.
    rv.reserve(1), sv.reserve(1), rj.reserve(1), sj.reserve(1);
    const T base = unzeroed ? T(NAN) : T(1);
    if (unzeroed) options.push_back({"result_zeroed", "0"});
    std::vector<T> out(size_t(R.rows) * S.rows, base);
    std::vector<const char *> keys, vals;
    for (auto &kv : options) {
        keys.push_back(kv.first.c_str());
        vals.push_back(kv.second.c_str());
    }
    fflush(stdout);
    char log_path[] = "/tmp/spalm-native-log-XXXXXX";
    const int saved = dup(1),
              null = log ? mkstemp(log_path) : open("/dev/null", O_WRONLY);
    dup2(null, 1);
    const int status = spalm(
        type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(), sv.data(),
        R.rows, K, S.rows, false, out.data(), nullptr, nullptr, nullptr,
        int(keys.size()), keys.data(), vals.data()
    );
    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    close(null);
    if (log) {
        log->clear();
        if (FILE *f = std::fopen(log_path, "r")) {
            char buf[4096];
            for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
                log->append(buf, n);
            std::fclose(f);
        }
        unlink(log_path);
    }
    const auto expected = reference<T>(R, S, K);
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (!(out[i] == expected[i] + (unzeroed ? T(0) : T(1)))) {
            if (!bad) first = i;
            ++bad;
        }
    }
    if (status != 0 || bad) {
        std::string opts;
        for (auto &kv : options) opts += " " + kv.first + "=" + kv.second;
        std::printf(
            "FAIL %s %s status=%d mismatches=%zu first=%zu got=%g want=%g%s\n",
            label, type ? "double" : "float", status, bad, first,
            bad ? double(out[first]) : 0.0,
            bad ? double(expected[first] + (unzeroed ? T(0) : T(1))) : 0.0,
            opts.c_str()
        );
        return false;
    }
    return true;
}

// spalm_sparse(): the row blocks must hold exactly the reference's nonzeros
// (each (row, column) once; MKL may add entries that are 0), bit for bit.
template <typename T>
static bool run_sparse_case(
    const sparse_api &api, const char *label, const csr &R, const csr &S,
    int K, const std::vector<std::pair<std::string, std::string>> &options
) {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    std::vector<T> rv(R.val.begin(), R.val.end()),
        sv(S.val.begin(), S.val.end());
    std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
    rv.reserve(1), sv.reserve(1), rj.reserve(1), sj.reserve(1);
    std::vector<const char *> keys, vals;
    for (auto &kv : options) {
        keys.push_back(kv.first.c_str());
        vals.push_back(kv.second.c_str());
    }
    fflush(stdout);
    const int saved = dup(1), null = open("/dev/null", O_WRONLY);
    dup2(null, 1);
    void *handle = nullptr;
    const int status = api.run(
        type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(), sv.data(),
        R.rows, K, S.rows, int(keys.size()), keys.data(), vals.data(), &handle
    );
    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    close(null);
    const auto expected = reference<T>(R, S, K);
    std::vector<char> seen(expected.size(), 0);
    size_t bad = 0;
    if (status == 0 && handle) {
        long long count = 0;
        const spalm_csr_block *blocks = api.blocks(handle, &count);
        for (long long b = 0; b < count; ++b) {
            const spalm_csr_block &blk = blocks[b];
            const T *v = static_cast<const T *>(blk.vals);
            for (int r = 0; r < blk.rows; ++r) {
                const int i = blk.row_begin + r;
                for (int p = blk.starts[r]; p < blk.ends[r]; ++p) {
                    const int j = blk.cols[p];
                    if (i < 0 || i >= R.rows || j < 0 || j >= S.rows) {
                        ++bad;
                        continue;
                    }
                    const size_t e = size_t(i) * S.rows + j;
                    if (seen[e]++ || !(v[p] == expected[e])) ++bad;
                }
            }
        }
        for (size_t e = 0; e < expected.size(); ++e)
            if (!seen[e] && expected[e] != T(0)) ++bad;
        api.free(handle);
    }
    if (status != 0 || !handle || bad) {
        std::string opts;
        for (auto &kv : options) opts += " " + kv.first + "=" + kv.second;
        std::printf(
            "FAIL sparse %s %s status=%d mismatches=%zu%s\n", label,
            type ? "double" : "float", status, bad, opts.c_str()
        );
        return false;
    }
    return true;
}

// Both types, default pages and 64-entry pages (rows split over pages).
// Returns the number of failures and counts the cases.
static int run_sparse_pages(
    const sparse_api &api, const std::string &label, const csr &R,
    const csr &S, int K, std::vector<std::pair<std::string, std::string>> o,
    int &cases
) {
    int failures = 0;
    for (const char *page : {"0", "64"}) {
        auto so = o;
        so.push_back({"ss_output_page_entries", page});
        failures += !run_sparse_case<double>(api, label.c_str(), R, S, K, so);
        failures += !run_sparse_case<float>(api, label.c_str(), R, S, K, so);
        cases += 2;
    }
    return failures;
}

// The caller's side of spalm_sparse_or_dense(): a dense result handed out
// on request (filled with `fill`), or null when `fail` is set.
template <typename T> struct dense_request_ctx {
    std::vector<T> buf;
    T fill = T(0);
    bool fail = false;
    int calls = 0;
    static void *provide(void *p) {
        auto *ctx = static_cast<dense_request_ctx *>(p);
        ++ctx->calls;
        if (ctx->fail) return nullptr;
        std::fill(ctx->buf.begin(), ctx->buf.end(), ctx->fill);
        return ctx->buf.data();
    }
};

// spalm_sparse_or_dense(): the result is either row blocks (callback not
// called, a handle) or the dense buffer (called once, no handle), and must
// match the reference bit for bit. expect_dense: 1 / 0 = which one the
// decision must take, -1 = either. unzeroed: the buffer starts as NaN with
// result_zeroed = 0. fail: the callback returns null, so a dense decision
// must fail with status 4 (and leave no handle). With `log`, the
// library's stdout is kept there.
template <typename T>
static bool run_sparse_or_dense_case(
    spalm_sparse_or_dense_fn run, const sparse_api &api, const char *label,
    const csr &R, const csr &S, int K,
    std::vector<std::pair<std::string, std::string>> options, int expect_dense,
    bool unzeroed = false, bool fail = false, std::string *log = nullptr
) {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    std::vector<T> rv(R.val.begin(), R.val.end()),
        sv(S.val.begin(), S.val.end());
    std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
    rv.reserve(1), sv.reserve(1), rj.reserve(1), sj.reserve(1);
    if (unzeroed) options.push_back({"result_zeroed", "0"});
    std::vector<const char *> keys, vals;
    for (auto &kv : options) {
        keys.push_back(kv.first.c_str());
        vals.push_back(kv.second.c_str());
    }
    dense_request_ctx<T> ctx;
    ctx.buf.assign(size_t(R.rows) * S.rows, T(0));
    ctx.fill = unzeroed ? T(NAN) : T(0);
    ctx.fail = fail;
    fflush(stdout);
    char log_path[] = "/tmp/spalm-native-log-XXXXXX";
    const int saved = dup(1),
              null = log ? mkstemp(log_path) : open("/dev/null", O_WRONLY);
    dup2(null, 1);
    void *handle = reinterpret_cast<void *>(1);
    const int status = run(
        type, ri.data(), rj.data(), rv.data(), si.data(), sj.data(), sv.data(),
        R.rows, K, S.rows, int(keys.size()), keys.data(), vals.data(),
        &dense_request_ctx<T>::provide, &ctx, &handle
    );
    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    close(null);
    if (log) {
        log->clear();
        if (FILE *f = std::fopen(log_path, "r")) {
            char buf[4096];
            for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;)
                log->append(buf, n);
            std::fclose(f);
        }
        unlink(log_path);
    }
    const auto expected = reference<T>(R, S, K);
    size_t bad = 0;
    const bool dense = ctx.calls > 0;
    const char *why = nullptr;
    if (ctx.calls > 1) why = "callback called more than once";
    else if (fail && dense) {
        if (status != 4 || handle) why = "failed callback: want status 4, no handle";
    } else if (status != 0) why = "status";
    else if (dense == bool(handle)) why = "handle and callback disagree";
    else if (expect_dense >= 0 && dense != bool(expect_dense))
        why = dense ? "dense result, want sparse" : "sparse result, want dense";
    else if (dense) {
        for (size_t i = 0; i < expected.size(); ++i)
            if (!(ctx.buf[i] == expected[i])) ++bad;
    } else {
        std::vector<char> seen(expected.size(), 0);
        long long count = 0;
        const spalm_csr_block *blocks = api.blocks(handle, &count);
        for (long long b = 0; b < count; ++b) {
            const spalm_csr_block &blk = blocks[b];
            const T *v = static_cast<const T *>(blk.vals);
            for (int r = 0; r < blk.rows; ++r) {
                const int i = blk.row_begin + r;
                for (int p = blk.starts[r]; p < blk.ends[r]; ++p) {
                    const int j = blk.cols[p];
                    if (i < 0 || i >= R.rows || j < 0 || j >= S.rows) {
                        ++bad;
                        continue;
                    }
                    const size_t e = size_t(i) * S.rows + j;
                    if (seen[e]++ || !(v[p] == expected[e])) ++bad;
                }
            }
        }
        for (size_t e = 0; e < expected.size(); ++e)
            if (!seen[e] && expected[e] != T(0)) ++bad;
    }
    if (status == 0 && handle) api.free(handle);
    if (fail && !dense && expect_dense == 1) why = "callback not called";
    if (why || bad) {
        std::string opts;
        for (auto &kv : options) opts += " " + kv.first + "=" + kv.second;
        std::printf(
            "FAIL sparse-or-dense %s %s status=%d calls=%d mismatches=%zu %s%s\n",
            label, type ? "double" : "float", status, ctx.calls, bad,
            why ? why : "", opts.c_str()
        );
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const char *lib = argc > 1 ? argv[1] : "bin/spalm.so";
    void *handle = dlopen(lib, RTLD_NOW);
    if (!handle) {
        std::printf("cannot load %s: %s\n", lib, dlerror());
        return 2;
    }
    auto spalm = reinterpret_cast<spalm_fn>(dlsym(handle, "spalm"));
    assert(spalm);
    // The DuckDB extension skips zeroing for MKL only when this is exported.
    assert(dlsym(handle, "spalm_accepts_unzeroed_result"));
    // Sparse output (init_method sparse / FORCE_MKL_SPARSE).
    const sparse_api sparse = {
        reinterpret_cast<spalm_sparse_fn>(dlsym(handle, "spalm_sparse")),
        reinterpret_cast<spalm_sparse_blocks_fn>(
            dlsym(handle, "spalm_sparse_result_blocks")
        ),
        reinterpret_cast<spalm_sparse_free_fn>(
            dlsym(handle, "spalm_sparse_result_free")
        )
    };
    assert(sparse.run && sparse.blocks && sparse.free);
    // Row blocks or the caller's dense result, after the decision.
    const auto sparse_or_dense = reinterpret_cast<spalm_sparse_or_dense_fn>(
        dlsym(handle, "spalm_sparse_or_dense")
    );
    assert(sparse_or_dense);
    using abi_fn = int (*)();
    assert(
        reinterpret_cast<abi_fn>(dlsym(handle, "spalm_sparse_result_abi"))() ==
        SPALM_SPARSE_RESULT_ABI
    );

    std::mt19937 rng(20261027);
    std::uniform_real_distribution<double> u(0, 1);
    struct shape {
        const char *name;
        int M, N, K;
    };
    // Tails: M not a multiple of 12, N not a multiple of 16/32, K over
    // several 512-key slices plus a partial one.
    const shape shapes[] = {
        {"small", 37, 45, 1100},
        {"tall", 157, 23, 700},
        // Wide output: room for SS output-column blocks (>= 256 columns each).
        {"wide", 29, 1100, 600}
    };
    int failures = 0, cases = 0;
    for (const shape &sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K;
        struct pattern {
            const char *name;
            csr R, S;
            // Methods whose packed side has no repeated key (packing keeps
            // only the last of repeated entries: an old, separate issue).
            std::vector<const char *> methods = {
                "sparse", "dense_left_spmm_inner_kernel",
                "dense_right_spmm_inner_kernel", "dense_bm_inner_kernel"
            };
        };
        std::vector<pattern> patterns;
        // Uniform 5%: uneven subrow lengths and scattered output columns.
        patterns.push_back(
            {"uniform",
             make_csr(
                 M, K, [&](int, int) { return u(rng) < 0.05; }, rng
             ),
             make_csr(N, K, [&](int, int) { return u(rng) < 0.05; }, rng)}
        );
        // Uniform 1%: very short subrows, so the automatic slice length grows.
        patterns.push_back(
            {"uniform-1pct",
             make_csr(
                 M, K, [&](int, int) { return u(rng) < 0.01; }, rng
             ),
             make_csr(N, K, [&](int, int) { return u(rng) < 0.01; }, rng)}
        );
        // Column pattern: a few full keys, every row has them -> long runs of
        // consecutive output columns in SS, equal subrow lengths in SD/DS.
        std::vector<char> hot(K);
        for (int k = 0; k < K; ++k) hot[k] = u(rng) < 0.04;
        patterns.push_back(
            {"col",
             make_csr(
                 M, K, [&](int, int k) { return hot[k]; }, rng
             ),
             make_csr(
                 N, K, [&](int r, int k) { return hot[k] && r % 7 != 3; }, rng
             )}
        );
        // Some subrows full (every key of the slice column), others not.
        patterns.push_back(
            {"full-mixed",
             make_csr(
                 M, K,
                 [&](int r, int k) {
                     return hot[k] && (r % 3 == 0 || u(rng) < 0.5);
                 },
                 rng
             ),
             make_csr(
                 N, K,
                 [&](int r, int k) {
                     return hot[k] && (r % 2 == 0 || u(rng) < 0.4);
                 },
                 rng
             )}
        );
        // A repeated key on the sparse side: subrows as long as a full one
        // that miss a key must take the general path.
        std::vector<int> hot_keys;
        for (int k = 0; k < K; ++k)
            if (hot[k]) hot_keys.push_back(k);
        const csr all_hot_R =
            make_csr(M, K, [&](int, int k) { return hot[k]; }, rng);
        const csr all_hot_S =
            make_csr(N, K, [&](int, int k) { return hot[k]; }, rng);
        patterns.push_back(
            {"repeat-R",
             make_repeated(M, hot_keys, rng),
             all_hot_S,
             {"sparse", "dense_left_spmm_inner_kernel"}}
        );
        patterns.push_back(
            {"repeat-S",
             all_hot_R,
             make_repeated(N, hot_keys, rng),
             {"dense_right_spmm_inner_kernel"}}
        );
        // Row pattern with empty rows on both sides.
        patterns.push_back(
            {"row",
             make_csr(
                 M, K, [&](int r, int) { return r % 5 == 1 && u(rng) < 0.6; },
                 rng
             ),
             make_csr(
                 N, K, [&](int r, int) { return r % 3 != 0 && u(rng) < 0.3; },
                 rng
             )}
        );
        for (auto &pt : patterns) {
            for (const char *method : pt.methods) {
                const bool ss = std::string(method) == "sparse";
                const std::vector<const char *> column_blocks =
                    ss ? std::vector<const char *>{"-1", "1", "3"}
                       : std::vector<const char *>{"-1"};
                const bool ds =
                    std::string(method) == "dense_right_spmm_inner_kernel";
                const std::vector<const char *> aligned_panels =
                    ds ? std::vector<const char *>{"0", "1"}
                       : std::vector<const char *>{"1"};
                for (const char *aligned : aligned_panels) {
                    for (const char *blocks : column_blocks) {
                        for (const char *full : {"0", "1"}) {
                            for (const char *selective : {"0", "1"}) {
                                for (const char *contiguous : {"0", "1"}) {
                                    for (const char *threads : {"1", "4"}) {
                                        for (const char *slice :
                                             {"512", "64", "auto"}) {
                                            std::vector<std::pair<
                                                std::string, std::string>>
                                                o = {
                                                    {"init_method", method},
                                                    {"threads", threads},
                                                    {"selective_packing", selective},
                                                    {"sd_ds_full_rows", full},
                                                    {"ss_column_blocks", blocks},
                                                    {"ds_aligned_panels", aligned},
                                                    {"ss_contiguous", contiguous}
                                            };
                                            if (std::string(slice) != "auto")
                                                o.push_back(
                                                    {"slice_length", slice}
                                                );
                                            std::string label =
                                                std::string(sh.name) + "/" +
                                                pt.name + "/" + method;
                                            failures += !run_case<double>(
                                                spalm, label.c_str(), pt.R,
                                                pt.S, K, o
                                            );
                                            failures += !run_case<float>(
                                                spalm, label.c_str(), pt.R,
                                                pt.S, K, o
                                            );
                                            cases += 2;
                                            // SS with sparse output: the
                                            // same switches (SS ignores
                                            // full rows and packing).
                                            if (ss && full[0] == '1' &&
                                                selective[0] == '1') {
                                                failures += run_sparse_pages(
                                                    sparse, label, pt.R, pt.S,
                                                    K, o, cases
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Tunables (execution_options.h) that change what the kernels do:
    // packing buffers always or never on huge pages, more SS column blocks,
    // longer slices at any cost. Results must stay exact.
    {
        const int M = 61, N = 700, K = 2100;
        const csr R =
            make_csr(M, K, [&](int, int) { return u(rng) < 0.02; }, rng);
        const csr S =
            make_csr(N, K, [&](int, int) { return u(rng) < 0.02; }, rng);
        using kv = std::vector<std::pair<std::string, std::string>>;
        const kv sets[] = {
            {{"r_block_huge_bytes", "0"}, {"s_blocks_huge_bytes", "0"}},
            {{"r_block_huge_bytes", "-1"}, {"s_blocks_huge_bytes", "-1"}},
            {{"ss_column_blocks", "5"}, {"ss_column_block_min_cols", "1"}},
            {{"ss_column_blocks", "700"}, {"ss_column_block_min_cols", "1"}},
            {{"ss_column_block_min_cols", "1"},
             {"ss_column_block_cache_divisor", "1000"}},
            {{"asl_min_cost", "0"}},
            {{"asl_min_cost", "0"}, {"asl_max_levels", "2"},
             {"asl_min_cost_ratio", "10"}, {"asl_subrow_write_entries", "1000"}},
            {{"asl_min_cost", "0"}, {"asl_subrow_write_entries", "0"}},
        };
        for (const kv &set : sets) {
            for (const char *method :
                 {"sparse", "dense_left_spmm_inner_kernel",
                  "dense_right_spmm_inner_kernel", "dense_bm_inner_kernel"}) {
                for (const char *threads : {"1", "4"}) {
                    kv o = {{"init_method", method}, {"threads", threads}};
                    o.insert(o.end(), set.begin(), set.end());
                    const std::string label = std::string("tunables/") + method;
                    failures += !run_case<double>(spalm, label.c_str(), R, S, K, o);
                    failures += !run_case<float>(spalm, label.c_str(), R, S, K, o);
                    cases += 2;
                }
            }
        }
    }
    // The automatic dispatch with a decision table from thresholds_file that
    // sends slice pairs to all four kernels: R slices (12 rows) cycle
    // through 1%, 30% and 60% density, S slices (16 rows) through 1%, 10%
    // and 50%. R below bucket 5: SS; otherwise S below bucket 3: SS, below
    // 20: DS, else SD (R below bucket 40) or DD. The SS pairs run on the
    // mixed SS path, in the same call as the dense kernels.
    {
        char table[] = "/tmp/spalm-native-thresholds-XXXXXX";
        const int fd = mkstemp(table);
        assert(fd >= 0);
        std::string text = "num_buckets: 100\nss_upper:";
        for (int b = 0; b <= 100; ++b) text += b < 5 ? " 101" : " 3";
        text += "\nri_upper:";
        for (int b = 0; b <= 100; ++b) text += b < 5 ? " 0" : " 20";
        text += "\nleft_dense_threshold: 40\ndense_mm_threshold: 0\n";
        assert(write(fd, text.data(), text.size()) == ssize_t(text.size()));
        close(fd);
        const int M = 12 * 9 + 5, N = 16 * 9 + 7, K = 1100;
        const double r_density[] = {0.01, 0.3, 0.6}, s_density[] = {0.01, 0.1, 0.5};
        const csr R = make_csr(
            M, K, [&](int r, int) { return u(rng) < r_density[(r / 12) % 3]; }, rng
        );
        const csr S = make_csr(
            N, K, [&](int r, int) { return u(rng) < s_density[(r / 16) % 3]; }, rng
        );
        using kv = std::vector<std::pair<std::string, std::string>>;
        struct auto_case {
            kv conf;
            const char *backend; // expected for double (float: S slices of 32)
        } auto_cases[] = {
            {{{"auto_sparse_ratio", "0"}}, "[backend] spalm-internal sparse=1 dense=1"},
            {{{"auto_sparse_ratio", "0"}, {"cost_weight_ss", "30"},
              {"cost_weight_dd", "0.5"}, {"ss_column_block_min_cols", "1"}},
             "[backend] spalm-internal sparse=1 dense=1"},
            {{{"auto_sparse_ratio", "0"}, {"slice_length", "128"}},
             "[backend] spalm-internal sparse=1 dense=1"},
            {{{"auto_sparse_ratio", "0"}, {"selective_packing", "0"}},
             "[backend] spalm-internal sparse=1 dense=1"},
            {{{"cost_weight_ss", "0"}}, "[backend] spalm-internal sparse=0 dense=1"},
            {{{"dense_mm_threshold", "1e18"}}, "[backend] spalm-internal sparse=1 dense=0"},
        };
        for (const auto &c : auto_cases) {
            for (const char *threads : {"1", "4"}) {
                kv o = {{"init_method", "no_fallback"}, {"threads", threads},
                        {"thresholds_file", table}};
                o.insert(o.end(), c.conf.begin(), c.conf.end());
                std::string log;
                failures += !run_case<double>(spalm, "auto-table", R, S, K, o, false, &log);
                if (log.find(c.backend) == std::string::npos ||
                    log.find("[conf] init_method=no_fallback ") == std::string::npos) {
                    std::printf("FAIL auto-table: expected '%s'\n", c.backend);
                    ++failures;
                }
                failures += !run_case<float>(spalm, "auto-table", R, S, K, o);
                cases += 2;
                // spalm_sparse_or_dense(): row blocks exactly when the call
                // is all SS (float: either, its S slices differ).
                const bool all_ss =
                    std::string(c.backend).find("sparse=1 dense=0") !=
                    std::string::npos;
                failures += !run_sparse_or_dense_case<double>(
                    sparse_or_dense, sparse, "auto-table", R, S, K, o,
                    all_ss ? 0 : 1, false, false, &log
                );
                const char *choice = all_ss ? "[result-choice] sparse (all SS)"
                                     : std::string(c.backend).find("sparse=1") !=
                                             std::string::npos
                                         ? "[result-choice] dense (mixed)"
                                         : "[result-choice] dense (all dense)";
                if (log.find(choice) == std::string::npos) {
                    std::printf("FAIL sparse-or-dense auto-table: expected '%s'\n", choice);
                    ++failures;
                }
                failures += !run_sparse_or_dense_case<float>(
                    sparse_or_dense, sparse, "auto-table", R, S, K, o, -1
                );
                cases += 2;
                if (all_ss) {
                    // Rows split over 64-entry pages.
                    kv paged = o;
                    paged.push_back({"ss_output_page_entries", "64"});
                    failures += !run_sparse_or_dense_case<double>(
                        sparse_or_dense, sparse, "auto-table/pages", R, S, K,
                        paged, 0
                    );
                    ++cases;
                } else {
                    // Uninitialized buffer (result_zeroed = 0): every element
                    // written; a failed request fails the call.
                    failures += !run_sparse_or_dense_case<double>(
                        sparse_or_dense, sparse, "auto-table/unzeroed", R, S,
                        K, o, 1, true
                    );
                    failures += !run_sparse_or_dense_case<double>(
                        sparse_or_dense, sparse, "auto-table/no-buffer", R, S,
                        K, o, 1, false, true
                    );
                    cases += 2;
                }
            }
        }
        // auto (MKL fallback allowed): an all-SS decision goes to MKL
        // sparse-sparse, which writes the dense result; without that
        // fallback it stays on SS with row blocks.
        for (const char *threads : {"1", "4"}) {
            struct fallback_case {
                kv conf;
                int dense;
                const char *choice;
            } fallback_cases[] = {
                {{{"dense_mm_threshold", "1e18"}, {"auto_mkl_dense_ratio", "1e9"}},
                 1, "[result-choice] dense (mkl-sparse fallback)"},
                {{{"dense_mm_threshold", "1e18"}, {"auto_mkl_dense_ratio", "1e9"},
                  {"auto_mkl_sparse_fallback", "0"}},
                 0, "[result-choice] sparse (all SS)"},
                {{{"auto_mkl_dense_ratio", "0"}}, 1,
                 "[result-choice] dense (mkl-dense fallback)"},
            };
            for (const auto &c : fallback_cases) {
                kv o = {{"threads", threads}, {"thresholds_file", table}};
                o.insert(o.end(), c.conf.begin(), c.conf.end());
                std::string log;
                failures += !run_sparse_or_dense_case<double>(
                    sparse_or_dense, sparse, "auto-table/auto", R, S, K, o,
                    c.dense, false, false, &log
                );
                if (log.find(c.choice) == std::string::npos) {
                    std::printf("FAIL sparse-or-dense auto: expected '%s'\n", c.choice);
                    ++failures;
                }
                failures += !run_sparse_or_dense_case<float>(
                    sparse_or_dense, sparse, "auto-table/auto", R, S, K, o, c.dense
                );
                cases += 2;
            }
        }
        // Zero products: no request, a handle without entries.
        {
            const csr empty = make_csr(M, K, [](int, int) { return false; }, rng);
            const csr no_keys_R = make_csr(M, 0, [](int, int) { return true; }, rng);
            const csr no_keys_S = make_csr(N, 0, [](int, int) { return true; }, rng);
            const kv o = {{"init_method", "no_fallback"}, {"threads", "4"},
                          {"thresholds_file", table}};
            failures += !run_sparse_or_dense_case<double>(
                sparse_or_dense, sparse, "zero/empty-R", empty, S, K, o, 0
            );
            failures += !run_sparse_or_dense_case<double>(
                sparse_or_dense, sparse, "zero/K=0", no_keys_R, no_keys_S, 0, o, 0
            );
            cases += 2;
        }
        unlink(table);
    }
    // result_zeroed = 0: the MKL backends overwrite the result, the SPALM
    // kernels and every zero-work exit get it zeroed by the library.
    {
        const int M = 40, N = 36, K = 300;
        const csr R =
            make_csr(M, K, [&](int, int) { return u(rng) < 0.2; }, rng);
        const csr S =
            make_csr(N, K, [&](int, int) { return u(rng) < 0.2; }, rng);
        csr zeros = R;
        for (double &v : zeros.val) v = 0.0;
        const csr empty = make_csr(M, K, [](int, int) { return false; }, rng);
        const csr low = make_csr(
            M, K, [&](int, int k) { return k < K / 2 && u(rng) < 0.2; }, rng
        );
        const csr high = make_csr(
            N, K, [&](int, int k) { return k >= K / 2 && u(rng) < 0.2; }, rng
        );
        const csr no_keys_R =
            make_csr(M, 0, [](int, int) { return true; }, rng);
        const csr no_keys_S =
            make_csr(N, 0, [](int, int) { return true; }, rng);
        struct unzeroed_case {
            const char *name;
            const csr &R, &S;
            int K;
        } unzeroed_cases[] = {
            {"unzeroed/random", R, S, K},
            {"unzeroed/empty-R", empty, S, K},
            {"unzeroed/all-zero-R", zeros, S, K},
            {"unzeroed/no-common-keys", low, high, K},
            {"unzeroed/K=0", no_keys_R, no_keys_S, 0},
        };
        for (const auto &c : unzeroed_cases) {
            for (const char *method :
                 {"FORCE_MKL_DENSE", "FORCE_MKL_SPARSE", "sparse",
                  "dense_bm_inner_kernel", "dense_left_spmm_inner_kernel",
                  "dense_right_spmm_inner_kernel"}) {
                const std::vector<std::pair<std::string, std::string>> o = {
                    {"init_method", method}, {"threads", "4"}
                };
                const std::string label = std::string(c.name) + "/" + method;
                failures += !run_case<double>(
                    spalm, label.c_str(), c.R, c.S, c.K, o, true
                );
                failures += !run_case<float>(
                    spalm, label.c_str(), c.R, c.S, c.K, o, true
                );
                cases += 2;
            }
        }
    }
    // MKL sparse-sparse with sparse output (sp2m) on the pattern shapes, and
    // the methods without sparse output rejected.
    {
        const int M = 61, N = 300, K = 700;
        const csr R =
            make_csr(M, K, [&](int, int) { return u(rng) < 0.03; }, rng);
        const csr S =
            make_csr(N, K, [&](int, int) { return u(rng) < 0.03; }, rng);
        const csr dense_R = make_csr(
            M, K, [&](int r, int) { return r % 4 == 0 || u(rng) < 0.2; }, rng
        );
        for (const csr *r : {&R, &dense_R}) {
            for (const char *threads : {"1", "4"}) {
                const std::vector<std::pair<std::string, std::string>> o = {
                    {"init_method", "FORCE_MKL_SPARSE"}, {"threads", threads}
                };
                const char *name = "mkl-sp2m";
                failures += !run_sparse_case<double>(sparse, name, *r, S, K, o);
                failures += !run_sparse_case<float>(sparse, name, *r, S, K, o);
                cases += 2;
            }
        }
        for (const char *method :
             {"FORCE_MKL_DENSE", "dense_bm_inner_kernel",
              "dense_left_spmm_inner_kernel", "dense_right_spmm_inner_kernel",
              ""}) {
            std::vector<double> rv(R.val), sv(S.val);
            std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
            const char *keys[] = {"init_method"}, *vals[] = {method};
            void *result = reinterpret_cast<void *>(1);
            fflush(stdout);
            const int saved = dup(1), null = open("/dev/null", O_WRONLY);
            dup2(null, 1);
            const int status = sparse.run(
                1, ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
                sv.data(), M, K, N, 1, keys, vals, &result
            );
            fflush(stdout);
            dup2(saved, 1);
            close(saved);
            close(null);
            if (status != 1 || result) {
                std::printf(
                    "FAIL sparse output accepted init_method '%s'\n", method
                );
                ++failures;
            }
            ++cases;
        }
    }
    // spalm_sparse_or_dense() with the forced methods: SS and MKL sp2m keep
    // row blocks, the dense kernels and MKL dense ask for the dense result;
    // a null callback is rejected.
    {
        const int M = 61, N = 300, K = 700;
        const csr R =
            make_csr(M, K, [&](int, int) { return u(rng) < 0.03; }, rng);
        const csr S =
            make_csr(N, K, [&](int, int) { return u(rng) < 0.03; }, rng);
        struct forced_case {
            const char *method;
            int dense;
        } forced_cases[] = {
            {"sparse", 0},
            {"FORCE_MKL_SPARSE", 0},
            {"FORCE_MKL_DENSE", 1},
            {"dense_bm_inner_kernel", 1},
            {"dense_left_spmm_inner_kernel", 1},
            {"dense_right_spmm_inner_kernel", 1},
        };
        for (const auto &c : forced_cases) {
            for (const char *threads : {"1", "4"}) {
                const std::vector<std::pair<std::string, std::string>> o = {
                    {"init_method", c.method}, {"threads", threads}
                };
                const std::string label = std::string("forced/") + c.method;
                failures += !run_sparse_or_dense_case<double>(
                    sparse_or_dense, sparse, label.c_str(), R, S, K, o, c.dense
                );
                failures += !run_sparse_or_dense_case<float>(
                    sparse_or_dense, sparse, label.c_str(), R, S, K, o, c.dense
                );
                cases += 2;
                if (c.dense) {
                    failures += !run_sparse_or_dense_case<double>(
                        sparse_or_dense, sparse, label.c_str(), R, S, K, o, 1,
                        true
                    );
                    failures += !run_sparse_or_dense_case<double>(
                        sparse_or_dense, sparse, label.c_str(), R, S, K, o, 1,
                        false, true
                    );
                    cases += 2;
                }
            }
        }
        std::vector<double> rv(R.val), sv(S.val);
        std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
        const char *keys[] = {"init_method"}, *vals[] = {"sparse"};
        void *result = reinterpret_cast<void *>(1);
        const int status = sparse_or_dense(
            1, ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
            sv.data(), M, K, N, 1, keys, vals, nullptr, nullptr, &result
        );
        if (status != 1 || result) {
            std::printf("FAIL sparse-or-dense accepted a null callback\n");
            ++failures;
        }
        ++cases;
    }
    std::printf(
        "%s: %d native kernel cases (SS/SD/DS/DD x DS aligned panels x SS "
        "column blocks x full "
        "rows x selective "
        "packing x SS contiguous x threads x slice length (fixed / automatic) "
        "x float/double, plus tunables, mixed SS and dense calls from a "
        "thresholds file, uninitialized results for MKL and SPALM, and "
        "sparse output for SS and MKL sp2m, and row blocks or a requested "
        "dense result from spalm_sparse_or_dense), "
        "%d failed\n",
        failures ? "FAIL" : "PASS", cases, failures
    );
    return failures ? 1 : 0;
}
