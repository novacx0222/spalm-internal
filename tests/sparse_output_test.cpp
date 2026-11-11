// Sparse output (spalm_sparse()) on portable machines.
//
// 1. The SS row loop and its output pages (cpu/ss_sparse_output.h) with a
//    scalar kernel in place of the AVX-512 one: every combination of
//    threads, page size and output-column blocks against a dense reference,
//    including rows split across pages, repeated keys, empty and dense rows
//    and products that cancel to 0.
// 2. The C API through the real wrapper (spalm.cpp): init_method sparse and
//    FORCE_MKL_SPARSE give the same nonzeros as spalm()'s dense result;
//    other methods are rejected; the zero-work exits give no blocks. Only
//    the SIMD kernels, MKL and dynamic loading are scalar test doubles.
//
// Runs from the repository root (parameters/cpu/*/tile_sizes.txt).
#include "ss_mat.h"
#include "path_utils.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <random>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "cpu/execution_options.h"
#include "cpu/ss_sparse_output.h"
#include "spalm_sparse_result.h"
#define SPARSE_SPARSE_MATMUL_H
template <typename T>
void sparse_sparse_matmul(
    ss_matrix<T> &, ss_matrix<T> &, T *, bool, int, bool = false,
    const cpu_execution_options & = cpu_execution_options(),
    spalm_sparse_result * = nullptr
);

static void scalar_baseline(
    int, const char *, int *, int *, void *, void *, int *, int *, void *,
    void *, void *, int, int, int, int
);
static int scalar_sparse_output(
    int, int *, int *, void *, int *, int *, void *, const int, const int,
    const int, const int, void **, int **, int **, int **, void **
);
static void scalar_sparse_output_free(void *);
static int sparse_output_calls, sparse_output_frees;
static void *test_dlopen(const char *, int) {
    return reinterpret_cast<void *>(1);
}
static void *test_dlsym(void *, const char *name) {
    if (std::strcmp(name, "run_baseline_sparse_output") == 0)
        return reinterpret_cast<void *>(&scalar_sparse_output);
    if (std::strcmp(name, "run_baseline_sparse_output_free") == 0)
        return reinterpret_cast<void *>(&scalar_sparse_output_free);
    return reinterpret_cast<void *>(&scalar_baseline);
}
#define dlopen test_dlopen
#define dlsym test_dlsym
#include "../src/cpu/spalm.cpp"
#undef dlopen
#undef dlsym

namespace {

struct csr_input {
    int rows = 0, K = 0;
    std::vector<int> ptr{0}, col;
    std::vector<double> val;
};

// S^T by key (rows ascending within a key), and output-column block starts
// as ss_column_block_starts builds them.
template <typename T> struct transposed {
    std::vector<int> offsets, rows, block_starts;
    std::vector<T> vals;
    int blocks = 1;
};

template <typename T>
transposed<T> transpose(const int *ptr, const int *col, const T *val, int N,
                        int K, int blocks) {
    transposed<T> t;
    t.offsets.assign(size_t(K) + 1, 0);
    for (int s = 0; s < N; ++s)
        for (int q = ptr[s]; q < ptr[s + 1]; ++q) ++t.offsets[col[q] + 1];
    for (int k = 0; k < K; ++k) t.offsets[k + 1] += t.offsets[k];
    t.rows.resize(t.offsets[K]);
    t.vals.resize(t.offsets[K]);
    std::vector<int> next(t.offsets.begin(), t.offsets.end() - 1);
    for (int s = 0; s < N; ++s)
        for (int q = ptr[s]; q < ptr[s + 1]; ++q) {
            const int p = next[col[q]]++;
            t.rows[p] = s;
            t.vals[p] = val[q];
        }
    t.blocks = blocks;
    if (blocks > 1) {
        const int width = (N + blocks - 1) / blocks;
        t.block_starts.resize(size_t(blocks + 1) * K);
        for (int k = 0; k < K; ++k) {
            int p = t.offsets[k];
            for (int b = 0; b < blocks; ++b) {
                while (p < t.offsets[k + 1] && t.rows[p] < b * width) ++p;
                t.block_starts[size_t(b) * K + k] = p;
            }
            t.block_starts[size_t(blocks) * K + k] = t.offsets[k + 1];
        }
    }
    return t;
}

// The SS row loop with a scalar kernel (same per-element order as the
// AVX-512 one: the row's keys in order).
template <typename T>
void scalar_ss_product(ss_sparse_output<T> &out, int threads, int M, int N,
                       int K, const int *ptr, const int *col, const T *val,
                       const transposed<T> &t) {
    ss_sparse_product(
        out, threads, M, N, K, ptr, col, t.rows.data(), t.offsets.data(),
        t.blocks > 1 ? t.block_starts.data() : nullptr, t.blocks,
        [&](int p0, int p1, const int *first, const int *last, T *acc) {
            for (int p = p0; p < p1; ++p) {
                const int k = col[p];
                for (int q = first[k]; q < last[k]; ++q)
                    acc[t.rows[q]] += val[p] * t.vals[q];
            }
        }
    );
}

template <typename T> const char *type_name() {
    return sizeof(T) == 4 ? " float" : " double";
}

// Dense C = R * S^T with the same per-element addition order.
template <typename T>
std::vector<T> dense_reference(const csr_input &R, const csr_input &S) {
    std::vector<T> rv(R.val.begin(), R.val.end());
    std::vector<T> sv(S.val.begin(), S.val.end());
    const auto t = transpose<T>(S.ptr.data(), S.col.data(), sv.data(), S.rows,
                                R.K, 1);
    std::vector<T> out(size_t(R.rows) * S.rows, T(0));
    for (int i = 0; i < R.rows; ++i)
        for (int p = R.ptr[i]; p < R.ptr[i + 1]; ++p) {
            const int k = R.col[p];
            for (int q = t.offsets[k]; q < t.offsets[k + 1]; ++q)
                out[size_t(i) * S.rows + t.rows[q]] += rv[p] * t.vals[q];
        }
    return out;
}

// Checks the blocks' structure and returns the entries as (row, col) ->
// value, failing on a repeated (row, col).
template <typename T>
bool collect(const spalm_sparse_result &r, int M, int N, bool zeros_allowed,
             std::map<std::pair<int, int>, T> &entries, const char *label) {
    long long total = 0;
    for (const auto &b : r.blocks) {
        if (b.rows < 1 || b.row_begin < 0 || b.row_begin + b.rows > M) {
            std::printf("FAIL %s: block rows [%d, +%d) of %d\n", label,
                        b.row_begin, b.rows, M);
            return false;
        }
        const T *vals = static_cast<const T *>(b.vals);
        for (int i = 0; i < b.rows; ++i) {
            if (b.starts[i] > b.ends[i]) {
                std::printf("FAIL %s: row range reversed\n", label);
                return false;
            }
            for (int p = b.starts[i]; p < b.ends[i]; ++p) {
                const int j = b.cols[p];
                if (j < 0 || j >= N || (!zeros_allowed && vals[p] == T(0))) {
                    std::printf("FAIL %s: entry col %d value %g\n", label, j,
                                double(vals[p]));
                    return false;
                }
                const auto key = std::make_pair(b.row_begin + i, j);
                if (!entries.emplace(key, vals[p]).second) {
                    std::printf("FAIL %s: (%d, %d) twice\n", label,
                                b.row_begin + i, j);
                    return false;
                }
                ++total;
            }
        }
    }
    if (total != r.entries) {
        std::printf("FAIL %s: %lld entries, result says %lld\n", label, total,
                    r.entries);
        return false;
    }
    return true;
}

// The nonzeros of a dense M x N result and of `entries` (zeros dropped)
// must agree exactly.
template <typename T>
bool same_nonzeros(const std::vector<T> &dense, int M, int N,
                   const std::map<std::pair<int, int>, T> &entries,
                   const char *label) {
    size_t nonzeros = 0, bad = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            const T want = dense[size_t(i) * N + j];
            const auto it = entries.find({i, j});
            const bool stored = it != entries.end() && it->second != T(0);
            if (want != T(0)) {
                ++nonzeros;
                if (!stored || !(it->second == want ||
                                 (std::isnan(want) && std::isnan(it->second))))
                    ++bad;
            } else if (stored) {
                ++bad;
            }
        }
    if (bad) {
        std::printf("FAIL %s: %zu of %zu nonzeros differ\n", label, bad,
                    nonzeros);
        return false;
    }
    return true;
}

csr_input random_csr(int rows, int K, double density, std::mt19937 &rng,
                     int dense_every = 0, int empty_every = 0,
                     bool repeat_first = false) {
    std::uniform_real_distribution<double> u(0, 1);
    csr_input m;
    m.rows = rows;
    m.K = K;
    for (int r = 0; r < rows; ++r) {
        const bool empty = empty_every && r % empty_every == 1;
        const bool full = dense_every && r % dense_every == 0;
        bool first = true;
        for (int k = 0; k < K && !empty; ++k) {
            if (!full && u(rng) >= density) continue;
            const int copies = repeat_first && first ? 2 : 1;
            first = false;
            for (int c = 0; c < copies; ++c) {
                m.col.push_back(k);
                const int v = int(rng() % 7) - 3;
                m.val.push_back(v ? v : 2);
            }
        }
        m.ptr.push_back(int(m.col.size()));
    }
    return m;
}

int failures = 0, cases = 0;
long long scanned_total = 0, listed_total = 0;

// Part 1: the row loop and pages, direct.
template <typename T>
void check_product(const char *name, const csr_input &R, const csr_input &S) {
    const auto want = dense_reference<T>(R, S);
    std::vector<T> rv(R.val.begin(), R.val.end());
    std::vector<T> sv(S.val.begin(), S.val.end());
    for (int blocks : {1, 3}) {
        const auto t = transpose<T>(S.ptr.data(), S.col.data(), sv.data(),
                                    S.rows, R.K, blocks);
        for (int threads : {1, 3}) {
            for (size_t page : {size_t(0), size_t(64)}) {
                ss_sparse_output<T> out(page);
                scalar_ss_product(out, threads, R.rows, S.rows, R.K,
                                  R.ptr.data(), R.col.data(), rv.data(), t);
                spalm_sparse_result result;
                out.finish(result);
                scanned_total += out.scanned_rows();
                listed_total += out.listed_rows();
                std::map<std::pair<int, int>, T> got;
                const std::string label = std::string(name) + " blocks=" +
                                          std::to_string(blocks) + " threads=" +
                                          std::to_string(threads) + " page=" +
                                          std::to_string(page) +
                                          type_name<T>();
                const char *l = label.c_str();
                const bool ok =
                    collect(result, R.rows, S.rows, false, got, l) &&
                    same_nonzeros(want, R.rows, S.rows, got, l);
                failures += !ok;
                ++cases;
            }
        }
    }
}

template <typename T> void check_emit_semantics() {
    // NaN is kept, +0 and -0 dropped; the buffer is +0 afterwards.
    const int N = 40;
    for (int scan : {0, 1}) {
        ss_sparse_output<T> out(64);
        out.start(1, N);
        auto &rows = out.local(0);
        T *acc = rows.row_buffer();
        acc[3] = T(NAN);
        acc[5] = T(-0.0);
        acc[17] = T(2);
        acc[39] = T(-1);
        const int keys[] = {0, 1};
        const int Sa[] = {3, 5, 17, 3, 39, 0};
        const int first[] = {0, 3}, last[] = {3, 6};
        rows.begin_block(7);
        if (scan)
            ss_emit_range(rows, acc, 0, N);
        else
            ss_emit_listed(rows, acc, keys, 2, Sa, first, last);
        rows.end_row();
        rows.end_block();
        bool clean = true;
        for (int j = 0; j < N; ++j)
            clean = clean && acc[j] == T(0) && !std::signbit(acc[j]);
        spalm_sparse_result result;
        out.finish(result);
        std::map<std::pair<int, int>, T> got;
        const bool ok = clean &&
                        collect(result, 8, N, false, got, "emit semantics") &&
                        got.size() == 3 && std::isnan(got[{7, 3}]) &&
                        got[{7, 17}] == T(2) && got[{7, 39}] == T(-1);
        if (!ok) std::printf("FAIL emit semantics scan=%d\n", scan);
        failures += !ok;
        ++cases;
    }
}

void check_pages() {
    // One row of 1000 nonzeros with 64-entry pages: the row is split over
    // 16 blocks, which together hold it once.
    const int N = 1000;
    ss_sparse_output<double> out(64);
    out.start(1, N);
    auto &rows = out.local(0);
    double *acc = rows.row_buffer();
    for (int j = 0; j < N; ++j) acc[j] = j + 1;
    rows.begin_block(2);
    rows.end_row(); // row 2: empty
    ss_emit_range(rows, acc, 0, N);
    rows.end_row(); // row 3
    rows.end_row(); // row 4: empty
    rows.end_block();
    spalm_sparse_result result;
    out.finish(result);
    std::map<std::pair<int, int>, double> got;
    bool ok = collect(result, 5, N, false, got, "pages") &&
              got.size() == size_t(N) && result.blocks.size() == 16 &&
              out.pages() == 16;
    for (int j = 0; ok && j < N; ++j) ok = got[{3, j}] == j + 1;
    if (!ok) std::printf("FAIL pages: %zu blocks\n", result.blocks.size());
    failures += !ok;
    ++cases;
}

// Default pages: one row of kHugePage - kFirstPage + 3 nonzeros fills the
// pages below kHugePage entries (64 Ki doubling to 1 Mi) and starts one of
// kHugePage. The blocks hold the row once, in order. On Linux, only that
// last page is on huge pages, its arrays starting on 2 MiB boundaries.
void check_huge_pages() {
    using rows_t = ss_row_output<double>;
    const size_t big = rows_t::kHugePage;
    const int N = static_cast<int>(big - rows_t::kFirstPage + 3);
    ss_sparse_output<double> out;
    out.start(1, N);
    auto &rows = out.local(0);
    double *acc = rows.row_buffer();
    for (int j = 0; j < N; ++j) acc[j] = j + 1;
    rows.begin_block(0);
    ss_emit_range(rows, acc, 0, N);
    rows.end_row();
    rows.end_block();
    spalm_sparse_result result;
    out.finish(result);
    bool ok = out.pages() == 6 && result.blocks.size() == 6 &&
              result.entries == N &&
              out.page_bytes() == (2 * big - rows_t::kFirstPage) * 12;
    int next = 0;
    for (const auto &b : result.blocks) {
        const auto *vals = static_cast<const double *>(b.vals);
        for (int r = 0; ok && r < b.rows; ++r)
            for (int p = b.starts[r]; ok && p < b.ends[r]; ++p, ++next)
                ok = b.row_begin + r == 0 && b.cols[p] == next &&
                     vals[p] == next + 1;
    }
    ok = ok && next == N;
#if defined(__linux__)
    auto aligned = [](const void *p) {
        return reinterpret_cast<uintptr_t>(p) % spalm_huge::kHugeBytes == 0;
    };
    ok = ok && out.huge_page_bytes() == big * 12 &&
         aligned(result.blocks[5].cols) && aligned(result.blocks[5].vals);
#else
    ok = ok && out.huge_page_bytes() == 0;
#endif
    if (!ok)
        std::printf("FAIL huge pages: %zu blocks, %zu huge bytes\n",
                    result.blocks.size(), out.huge_page_bytes());
    failures += !ok;
    ++cases;
}

// Part 2: the C API through spalm.cpp.
template <typename T> int type_of() { return sizeof(T) == 4 ? 0 : 1; }

struct silence {
    int saved, null;
    silence() {
        std::fflush(stdout);
        saved = dup(1);
        null = open("/dev/null", O_WRONLY);
        dup2(null, 1);
    }
    ~silence() {
        std::fflush(stdout);
        dup2(saved, 1);
        close(saved);
        close(null);
    }
};

template <typename T>
int call_sparse(const csr_input &R, const csr_input &S,
                std::vector<std::pair<std::string, std::string>> options,
                void **handle) {
    std::vector<T> rv(R.val.begin(), R.val.end());
    std::vector<T> sv(S.val.begin(), S.val.end());
    std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
    rv.reserve(1), sv.reserve(1), rj.reserve(1), sj.reserve(1);
    std::vector<const char *> keys, vals;
    for (auto &kv : options) {
        keys.push_back(kv.first.c_str());
        vals.push_back(kv.second.c_str());
    }
    silence quiet;
    return spalm_sparse(type_of<T>(), ri.data(), rj.data(), rv.data(),
                        si.data(), sj.data(), sv.data(), R.rows, R.K, S.rows,
                        int(keys.size()), keys.data(), vals.data(), handle);
}

template <typename T>
std::vector<T> call_dense(
    const csr_input &R, const csr_input &S,
    std::vector<std::pair<std::string, std::string>> options
) {
    std::vector<T> rv(R.val.begin(), R.val.end());
    std::vector<T> sv(S.val.begin(), S.val.end());
    std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
    rv.reserve(1), sv.reserve(1), rj.reserve(1), sj.reserve(1);
    std::vector<const char *> keys, vals;
    for (auto &kv : options) {
        keys.push_back(kv.first.c_str());
        vals.push_back(kv.second.c_str());
    }
    std::vector<T> out(size_t(R.rows) * S.rows, T(0));
    out.reserve(1);
    silence quiet;
    const int status = spalm(
        type_of<T>(), ri.data(), rj.data(), rv.data(), si.data(), sj.data(),
        sv.data(), R.rows, R.K, S.rows, false, out.data(), nullptr, nullptr,
        nullptr, int(keys.size()), keys.data(), vals.data()
    );
    assert(status == 0);
    return out;
}

template <typename T>
void check_api(const char *name, const csr_input &R, const csr_input &S) {
    for (const char *method : {"sparse", "FORCE_MKL_SPARSE"}) {
        for (const char *threads : {"1", "4"}) {
            for (const char *page : {"0", "64"}) {
                const std::vector<std::pair<std::string, std::string>> o = {
                    {"init_method", method}, {"threads", threads},
                    {"ss_output_page_entries", page}};
                const auto want = call_dense<T>(R, S, o);
                void *handle = nullptr;
                const int frees = sparse_output_frees;
                const int status = call_sparse<T>(R, S, o, &handle);
                const std::string label = std::string("api ") + name + " " +
                                          method + " threads=" + threads +
                                          " page=" + page +
                                          type_name<T>();
                bool ok = status == 0 && handle;
                if (ok) {
                    long long count = -1;
                    const spalm_csr_block *blocks =
                        spalm_sparse_result_blocks(handle, &count);
                    spalm_sparse_result copy;
                    copy.blocks.assign(blocks, blocks + count);
                    copy.entries = spalm_sparse_result_entries(handle);
                    std::map<std::pair<int, int>, T> got;
                    const bool mkl =
                        std::strcmp(method, "FORCE_MKL_SPARSE") == 0;
                    const char *l = label.c_str();
                    ok = collect(copy, R.rows, S.rows, mkl, got, l) &&
                         same_nonzeros(want, R.rows, S.rows, got, l);
                    spalm_sparse_result_free(handle);
                    // MKL's result is released with the handle, once.
                    if (mkl && sparse_output_frees != frees + 1) ok = false;
                }
                if (!ok) {
                    std::printf("FAIL %s status=%d\n", label.c_str(), status);
                }
                failures += !ok;
                ++cases;
            }
        }
    }
}

void check_api_rejects_and_exits() {
    std::mt19937 rng(7);
    const csr_input R = random_csr(9, 30, 0.3, rng);
    const csr_input S = random_csr(8, 30, 0.3, rng);
    for (const char *method :
         {"", "dense", "FORCE_MKL_DENSE", "dense_bm_inner_kernel",
          "dense_left_spmm_inner_kernel", "dense_right_spmm_inner_kernel",
          "no_fallback"}) {
        void *handle = reinterpret_cast<void *>(1);
        const int status = call_sparse<double>(
            R, S, {{"init_method", method}, {"threads", "2"}}, &handle);
        const bool ok = status == SPALM_INVALID_ARGUMENT && handle == nullptr;
        if (!ok) {
            std::printf("FAIL rejects init_method '%s': %d\n", method, status);
        }
        failures += !ok;
        ++cases;
    }
    {
        // No init_method at all (automatic dispatch) is rejected too.
        void *handle = nullptr;
        const int status =
            call_sparse<double>(R, S, {{"threads", "2"}}, &handle);
        failures += !(status == SPALM_INVALID_ARGUMENT && !handle);
        ++cases;
    }
    {
        silence quiet;
        failures += spalm_sparse(1, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, 1, 1, 1, 0, nullptr, nullptr,
                                 nullptr) != SPALM_INVALID_ARGUMENT;
        ++cases;
    }
    failures += spalm_sparse_result_abi() != SPALM_SPARSE_RESULT_ABI;
    long long count = -1;
    failures +=
        spalm_sparse_result_blocks(nullptr, &count) != nullptr || count != 0;
    failures += spalm_sparse_result_entries(nullptr) != 0;
    spalm_sparse_result_free(nullptr);
    cases += 3;

    // Zero-work exits: OK and no blocks, for both methods.
    csr_input zeros = R;
    for (double &v : zeros.val) v = 0;
    const csr_input empty = random_csr(9, 30, 0.0, rng);
    // R uses only keys 0..14, S only keys 15..29.
    csr_input low = random_csr(9, 15, 0.4, rng);
    low.K = 30;
    csr_input high;
    high.rows = 8;
    high.K = 30;
    high.ptr = {0};
    high.col.clear();
    high.val.clear();
    for (int r = 0; r < 8; ++r) {
        for (int k = 15; k < 30; ++k)
            if (rng() % 3 == 0) {
                high.col.push_back(k);
                high.val.push_back(1);
            }
        high.ptr.push_back(int(high.col.size()));
    }
    const csr_input no_keys_R = random_csr(9, 0, 0.5, rng);
    const csr_input no_keys_S = random_csr(8, 0, 0.5, rng);
    struct exit_case {
        const char *name;
        const csr_input &R, &S;
    } exits[] = {{"empty-R", empty, S},
                 {"all-zero-R", zeros, S},
                 {"no-common-keys", low, high},
                 {"K=0", no_keys_R, no_keys_S}};
    for (const auto &c : exits) {
        for (const char *method : {"sparse", "FORCE_MKL_SPARSE"}) {
            void *handle = nullptr;
            const int status = call_sparse<double>(
                c.R, c.S, {{"init_method", method}, {"threads", "2"}}, &handle);
            long long count = -1;
            bool ok = status == 0 && handle;
            ok = ok && spalm_sparse_result_blocks(handle, &count) == nullptr;
            ok = ok && count == 0 && spalm_sparse_result_entries(handle) == 0;
            if (!ok) {
                std::printf("FAIL exit %s %s: %d\n", c.name, method, status);
            }
            spalm_sparse_result_free(handle);
            failures += !ok;
            ++cases;
        }
    }
}

} // namespace

// Test doubles for the SIMD kernels and MKL (as in dispatch_test.cpp).
template <class T>
void scalar_csr(int *ri, int *rj, T *rv, int *si, int *sj, T *sv, T *output,
                int m, int n) {
    for (int r = 0; r < m; ++r)
        for (int p = ri[r]; p < ri[r + 1]; ++p)
            for (int s = 0; s < n; ++s)
                for (int q = si[s]; q < si[s + 1]; ++q)
                    if (rj[p] == sj[q])
                        output[size_t(r) * n + s] += rv[p] * sv[q];
}

template <class T>
void dense_dense_matmul(ss_matrix<T> &r, ss_matrix<T> &s, int, int, T *output,
                        bool, int, const cpu_execution_options &) {
    scalar_csr(r.csr_mat.row_ptrs, r.csr_mat.col_idxs, r.csr_mat.vals,
               s.csr_mat.row_ptrs, s.csr_mat.col_idxs, s.csr_mat.vals, output,
               r.csr_mat.rows, s.csr_mat.rows);
}

// The SS path: dense into `output`, or the real sparse row loop with a
// scalar kernel (as spgemm_rows_sparse_result_vanilla does with the SIMD one).
template <class T>
void sparse_sparse_matmul(ss_matrix<T> &r, ss_matrix<T> &s, T *output, bool,
                          int threads, bool all_sparse,
                          const cpu_execution_options &options,
                          spalm_sparse_result *sparse_out) {
    if (!sparse_out) {
        scalar_csr(r.csr_mat.row_ptrs, r.csr_mat.col_idxs, r.csr_mat.vals,
                   s.csr_mat.row_ptrs, s.csr_mat.col_idxs, s.csr_mat.vals,
                   output, r.csr_mat.rows, s.csr_mat.rows);
        return;
    }
    assert(all_sparse && !output);
    const int N = s.csr_mat.rows, K = r.csr_mat.cols;
    const auto t = transpose<T>(s.csr_mat.row_ptrs, s.csr_mat.col_idxs,
                                s.csr_mat.vals, N, K, 1);
    auto out = std::make_shared<ss_sparse_output<T>>(
        size_t(std::max(0LL, options.ss_output_page_entries)));
    scalar_ss_product(*out, threads, r.csr_mat.rows, N, K, r.csr_mat.row_ptrs,
                      r.csr_mat.col_idxs, r.csr_mat.vals, t);
    out->finish(*sparse_out);
    sparse_out->storage = out;
}

template <class T>
void baseline_typed(const char *method, int *ri, int *rj, void *rv, void *rd,
                    int *si, int *sj, void *sv, void *sd, void *output, int m,
                    int k, int n) {
    if (std::strcmp(method, "MKL sparse-sparse") == 0) {
        scalar_csr(ri, rj, static_cast<T *>(rv), si, sj, static_cast<T *>(sv),
                   static_cast<T *>(output), m, n);
        return;
    }
    for (int r = 0; r < m; ++r)
        for (int s = 0; s < n; ++s)
            for (int j = 0; j < k; ++j)
                static_cast<T *>(output)[size_t(r) * n + s] +=
                    static_cast<T *>(rd)[size_t(r) * k + j] *
                    static_cast<T *>(sd)[size_t(s) * k + j];
}
static void scalar_baseline(int type, const char *method, int *ri, int *rj,
                            void *rv, void *rd, int *si, int *sj, void *sv,
                            void *sd, void *output, int m, int k, int n, int) {
    if (type == 0)
        baseline_typed<float>(method, ri, rj, rv, rd, si, sj, sv, sd, output, m,
                              k, n);
    else
        baseline_typed<double>(method, ri, rj, rv, rd, si, sj, sv, sd, output,
                               m, k, n);
}

// MKL sp2m double: CSR with every structural product entry, those that
// cancel to 0 included, columns in descending order (MKL does not promise
// sorted columns).
struct scalar_csr_result {
    std::vector<int> starts, ends, cols;
    std::vector<float> f;
    std::vector<double> d;
};
template <class T>
void scalar_sp2m(scalar_csr_result &c, int *ri, int *rj, const T *rv, int *si,
                 int *sj, const T *sv, int m, int n, std::vector<T> &vals) {
    for (int r = 0; r < m; ++r) {
        c.starts.push_back(int(c.cols.size()));
        for (int s = n - 1; s >= 0; --s) {
            bool structural = false;
            T sum = 0;
            for (int p = ri[r]; p < ri[r + 1]; ++p)
                for (int q = si[s]; q < si[s + 1]; ++q)
                    if (rj[p] == sj[q]) {
                        structural = true;
                        sum += rv[p] * sv[q];
                    }
            if (structural) {
                c.cols.push_back(s);
                vals.push_back(sum);
            }
        }
        c.ends.push_back(int(c.cols.size()));
    }
}
static int scalar_sparse_output(int type, int *ri, int *rj, void *rv, int *si,
                                int *sj, void *sv, const int m, const int,
                                const int n, const int, void **handle,
                                int **rows_start, int **rows_end, int **cols,
                                void **vals) {
    ++sparse_output_calls;
    auto *c = new scalar_csr_result;
    if (type == 0) {
        scalar_sp2m(*c, ri, rj, static_cast<float *>(rv), si, sj,
                    static_cast<float *>(sv), m, n, c->f);
        *vals = c->f.data();
    } else {
        scalar_sp2m(*c, ri, rj, static_cast<double *>(rv), si, sj,
                    static_cast<double *>(sv), m, n, c->d);
        *vals = c->d.data();
    }
    *handle = c;
    *rows_start = c->starts.data();
    *rows_end = c->ends.data();
    *cols = c->cols.data();
    return 0;
}
static void scalar_sparse_output_free(void *handle) {
    ++sparse_output_frees;
    delete static_cast<scalar_csr_result *>(handle);
}

int main() {
    std::mt19937 rng(20261108);
    // Shapes: tails of the 4-row tasks; dense rows (scanned) and sparse ones
    // (listed); empty rows; a repeated key per R row; output wider than a
    // 64-entry page.
    struct shape {
        const char *name;
        csr_input R, S;
    };
    std::vector<shape> shapes;
    shapes.push_back({"uniform", random_csr(37, 60, 0.08, rng),
                      random_csr(45, 60, 0.08, rng)});
    shapes.push_back({"dense-rows", random_csr(23, 40, 0.05, rng, 5, 0),
                      random_csr(130, 40, 0.3, rng)});
    shapes.push_back({"empty-rows", random_csr(30, 50, 0.1, rng, 0, 3),
                      random_csr(20, 50, 0.1, rng, 0, 4)});
    shapes.push_back({"repeated-key", random_csr(17, 25, 0.2, rng, 0, 0, true),
                      random_csr(33, 25, 0.2, rng)});
    shapes.push_back({"wide", random_csr(9, 30, 0.3, rng),
                      random_csr(400, 30, 0.3, rng)});
    shapes.push_back({"one-row", random_csr(1, 30, 0.5, rng),
                      random_csr(70, 30, 0.5, rng)});
    // Few listed columns per row: emitted by revisiting them, not scanning.
    shapes.push_back({"sparse-wide",
                      random_csr(30, 200, 0.012, rng, 7, 0, true),
                      random_csr(500, 200, 0.012, rng)});
    for (const auto &s : shapes) {
        check_product<double>(s.name, s.R, s.S);
        check_product<float>(s.name, s.R, s.S);
        check_api<double>(s.name, s.R, s.S);
        check_api<float>(s.name, s.R, s.S);
    }
    // Both ways of emitting a row ran.
    if (!scanned_total || !listed_total) {
        std::printf("FAIL rows scanned %lld, listed %lld\n", scanned_total,
                    listed_total);
        ++failures;
    }
    ++cases;
    check_emit_semantics<double>();
    check_emit_semantics<float>();
    check_pages();
    check_huge_pages();
    check_api_rejects_and_exits();
    std::printf("%s: %d sparse output cases (row loop x threads x pages x "
                "column blocks, emit semantics, page splits, default page "
                "growth, C API for SS and "
                "MKL sp2m vs the dense result, rejected methods, zero-work "
                "exits), %d failed\n",
                failures ? "FAIL" : "PASS", cases, failures);
    return failures ? 1 : 0;
}
