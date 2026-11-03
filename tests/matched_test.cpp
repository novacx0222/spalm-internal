// Runs spalm_matched of the built library and compares its bitmap of matched
// pairs with a brute-force one: bit j of row i is set when row i of R and row
// j of S share a key, whatever the values. The inputs have explicit zeros and
// sums that cancel, which the result values cannot show, for every SPALM
// method (with mixed SS and dense pairs among them), switch, thread count and
// slice length. Values are small integers, so the result must also match the
// reference bit for bit. MKL marks nothing: forced, or as the automatic
// fallback, it must say so (marked = 0) and leave the bitmap zero.
//
// Build/run from the repository root (needs bin/spalm.so and parameters/):
//   make test-matched
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using spalm_matched_fn = int (*)(
    int, int *, int *, void *, int *, int *, void *, int, int, int, bool,
    void *, int *, int *, void *, int, const char **, const char **,
    uint64_t *, int *
);

struct csr {
    int rows = 0;
    std::vector<int> ptr{0}, col;
    std::vector<double> val;
};

// Row r has key k when pick(r, k), with the value value(r, k).
static csr make_csr(
    int rows, int K, const std::function<bool(int, int)> &pick,
    const std::function<double(int, int)> &value
) {
    csr m;
    m.rows = rows;
    for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < K; ++k) {
            if (!pick(r, k)) continue;
            m.col.push_back(k);
            m.val.push_back(value(r, k));
        }
        m.ptr.push_back(int(m.col.size()));
    }
    return m;
}

static size_t words_of(int N) { return (size_t(N) + 63) / 64; }

static std::vector<uint64_t> reference_matched(const csr &R, const csr &S, int K) {
    const size_t words = words_of(S.rows);
    std::vector<uint64_t> out(size_t(R.rows) * words, 0);
    std::vector<char> keys(K);
    for (int i = 0; i < R.rows; ++i) {
        std::fill(keys.begin(), keys.end(), 0);
        for (int p = R.ptr[i]; p < R.ptr[i + 1]; ++p) keys[R.col[p]] = 1;
        for (int j = 0; j < S.rows; ++j)
            for (int q = S.ptr[j]; q < S.ptr[j + 1]; ++q)
                if (keys[S.col[q]]) {
                    out[i * words + j / 64] |= uint64_t(1) << (j % 64);
                    break;
                }
    }
    return out;
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

static int call(
    spalm_matched_fn spalm_matched, int type, const csr &R, const csr &S,
    int K, const std::vector<std::pair<std::string, std::string>> &options,
    void *out, uint64_t *matched, int *marked, bool sparse_result = false
) {
    std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
    std::vector<float> rf(R.val.begin(), R.val.end()),
        sf(S.val.begin(), S.val.end());
    std::vector<double> rd = R.val, sd = S.val;
    // Non-null data() for inputs without entries, as from the operator.
    rj.reserve(1), sj.reserve(1), rf.reserve(1), sf.reserve(1), rd.reserve(1),
        sd.reserve(1);
    void *rv = type == 0 ? (void *) rf.data() : (void *) rd.data();
    void *sv = type == 0 ? (void *) sf.data() : (void *) sd.data();
    std::vector<const char *> keys, vals;
    for (auto &kv : options) {
        keys.push_back(kv.first.c_str());
        vals.push_back(kv.second.c_str());
    }
    fflush(stdout);
    const int saved = dup(1), null = open("/dev/null", O_WRONLY);
    dup2(null, 1);
    const int status = spalm_matched(
        type, ri.data(), rj.data(), rv, si.data(), sj.data(), sv, R.rows, K,
        S.rows, sparse_result, out, nullptr, nullptr, nullptr,
        int(keys.size()), keys.data(), vals.data(), matched, marked
    );
    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    close(null);
    return status;
}

// mkl: 1 when MKL must run (if a pair matches), 0 when SPALM's kernels must,
// -1 either (the automatic choice); MKL runs are counted in mkl_runs.
template <typename T>
static bool run_case(
    spalm_matched_fn spalm_matched, const std::string &label, const csr &R,
    const csr &S, int K,
    const std::vector<std::pair<std::string, std::string>> &options, int mkl,
    int &mkl_runs
) {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    std::vector<T> out(size_t(R.rows) * S.rows, T(0));
    std::vector<uint64_t> matched(size_t(R.rows) * words_of(S.rows), 0);
    int marked = -1;
    const int status = call(
        spalm_matched, type, R, S, K, options, out.data(), matched.data(),
        &marked
    );
    mkl_runs += marked == 0;
    auto want_matched = reference_matched(R, S, K);
    const bool any_match =
        std::any_of(want_matched.begin(), want_matched.end(), [](uint64_t w) {
            return w != 0;
        });
    if (marked == 0) std::fill(want_matched.begin(), want_matched.end(), 0);
    const auto want = reference<T>(R, S, K);
    size_t bad_bits = 0, bad_values = 0, first = 0;
    for (size_t w = 0; w < matched.size(); ++w)
        if (matched[w] != want_matched[w]) {
            if (!bad_bits) first = w;
            bad_bits += __builtin_popcountll(matched[w] ^ want_matched[w]);
        }
    for (size_t i = 0; i < out.size(); ++i) bad_values += !(out[i] == want[i]);
    // Without a common key the library returns before any backend: the
    // bitmap is complete (all zero) and marked, whatever the method
    const bool bad_marked = (marked != 0 && marked != 1) ||
                            (mkl == 0 && marked != 1) ||
                            (mkl == 1 && marked != 0 && any_match);
    if (status != 0 || bad_marked || bad_bits || bad_values) {
        std::string opts;
        for (auto &kv : options) opts += " " + kv.first + "=" + kv.second;
        std::printf(
            "FAIL %s %s status=%d marked=%d wrong_bits=%zu (first word %zu: "
            "%016llx, want %016llx) wrong_values=%zu%s\n",
            label.c_str(), type ? "double" : "float", status, marked, bad_bits,
            first,
            bad_bits ? (unsigned long long) matched[first] : 0ULL,
            bad_bits ? (unsigned long long) want_matched[first] : 0ULL,
            bad_values, opts.c_str()
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
    auto spalm_matched =
        reinterpret_cast<spalm_matched_fn>(dlsym(handle, "spalm_matched"));
    if (!spalm_matched) {
        std::printf("%s has no spalm_matched\n", lib);
        return 2;
    }

    std::mt19937 rng(20261103);
    std::uniform_real_distribution<double> u(0, 1);
    // -3..3: a seventh of the stored values are explicit zeros
    auto small = [&](int, int) { return double(int(rng() % 7) - 3); };
    auto zero = [](int, int) { return 0.0; };
    struct shape {
        const char *name;
        int M, N, K;
    };
    // Tails: M not a multiple of 12, N not a multiple of 16/32 or of 64,
    // K over several 512-key slices plus a partial one.
    const shape shapes[] = {
        {"small", 37, 45, 1100}, {"tall", 157, 23, 700}, {"wide", 29, 1100, 600}
    };
    const char *methods[] = {
        // Every pair SS; SD, DS or DD only; then mixed by the calibrated
        // thresholds: all dense pairs, SS and dense pairs, the automatic
        // choice without and with the MKL fallback; MKL
        "sparse", "dense_left_spmm_inner_kernel",
        "dense_right_spmm_inner_kernel", "dense_bm_inner_kernel", "dense",
        "no_features", "no_fallback", "", "FORCE_MKL_DENSE", "FORCE_MKL_SPARSE"
    };
    int failures = 0, cases = 0, mkl_runs = 0, auto_runs = 0;
    for (const shape &sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K;
        struct pattern {
            const char *name;
            csr R, S;
        };
        std::vector<pattern> patterns;
        patterns.push_back(
            {"uniform",
             make_csr(M, K, [&](int, int) { return u(rng) < 0.05; }, small),
             make_csr(N, K, [&](int, int) { return u(rng) < 0.05; }, small)}
        );
        patterns.push_back(
            {"uniform-1pct",
             make_csr(M, K, [&](int, int) { return u(rng) < 0.01; }, small),
             make_csr(N, K, [&](int, int) { return u(rng) < 0.01; }, small)}
        );
        // Hot keys in every row of S with +1 and -1 in turn: an R row with
        // all of them sums to zero with every S row, others mostly not.
        std::vector<int> hot_rank(K, -1);
        int hot = 0;
        for (int k = 0; k < K; ++k)
            if (u(rng) < 0.04) hot_rank[k] = hot++;
        patterns.push_back(
            {"cancel",
             make_csr(
                 M, K,
                 [&](int r, int k) {
                     return hot_rank[k] >= 0 && (r % 2 == 0 || u(rng) < 0.5);
                 },
                 [](int, int) { return 1.0; }
             ),
             make_csr(
                 N, K, [&](int, int k) { return hot_rank[k] >= 0; },
                 [&](int, int k) { return hot_rank[k] % 2 ? -1.0 : 1.0; }
             )}
        );
        // Bands: R row i and S row j share keys when |i - j| < 40, and
        // consecutive S rows make continuous S slices
        patterns.push_back(
            {"band",
             make_csr(
                 M, K, [&](int r, int k) { return k >= r && k < r + 20; },
                 small
             ),
             make_csr(
                 N, K, [&](int r, int k) { return k >= r - 20 && k < r + 20; },
                 small
             )}
        );
        // Empty rows on both sides
        patterns.push_back(
            {"row",
             make_csr(
                 M, K, [&](int r, int) { return r % 5 == 1 && u(rng) < 0.6; },
                 small
             ),
             make_csr(
                 N, K, [&](int r, int) { return r % 3 != 0 && u(rng) < 0.3; },
                 small
             )}
        );
        // Only explicit zeros in R: every pair with a key matches, all sums 0
        patterns.push_back(
            {"zeros",
             make_csr(M, K, [&](int, int) { return u(rng) < 0.2; }, zero),
             make_csr(N, K, [&](int, int) { return u(rng) < 0.2; }, small)}
        );
        // No common key: nothing matches
        patterns.push_back(
            {"disjoint",
             make_csr(
                 M, K, [&](int, int k) { return k < K / 2 && u(rng) < 0.2; },
                 small
             ),
             make_csr(
                 N, K, [&](int, int k) { return k >= K / 2 && u(rng) < 0.2; },
                 small
             )}
        );
        for (auto &pt : patterns) {
            for (const char *method : methods) {
                const std::string name = method;
                const int mkl = name.rfind("FORCE_MKL", 0) == 0 ? 1
                                : name.empty()                  ? -1
                                                                : 0;
                for (const char *threads : {"1", "4"}) {
                    for (const char *slice : {"512", "64", "auto"}) {
                        for (const char *selective : {"0", "1"}) {
                            std::vector<std::pair<std::string, std::string>> o =
                                {{"init_method", method},
                                 {"threads", threads},
                                 {"selective_packing", selective}};
                            if (std::string(slice) != "auto")
                                o.push_back({"slice_length", slice});
                            const std::string label = std::string(sh.name) +
                                                      "/" + pt.name + "/" +
                                                      (*method ? method : "auto");
                            const int before = mkl_runs;
                            failures += !run_case<double>(
                                spalm_matched, label, pt.R, pt.S, K, o, mkl,
                                mkl_runs
                            );
                            failures += !run_case<float>(
                                spalm_matched, label, pt.R, pt.S, K, o, mkl,
                                mkl_runs
                            );
                            cases += 2;
                            if (mkl < 0) auto_runs += 2;
                            else mkl_runs = before;
                        }
                    }
                }
            }
        }
    }
    // Only a dense result is marked
    {
        const csr R =
            make_csr(20, 100, [&](int, int) { return u(rng) < 0.2; }, small);
        std::vector<double> out(size_t(20) * 20);
        std::vector<uint64_t> matched(20 * words_of(20));
        int marked = -1;
        const int status = call(
            spalm_matched, 1, R, R, 100, {{"init_method", "sparse"}},
            out.data(), matched.data(), &marked, true
        );
        failures += status != 1;
        if (status != 1)
            std::printf("FAIL sparse result with matched: status=%d\n", status);
        ++cases;
    }
    std::printf(
        "%s: %d matched cases (bitmap and values against brute force; "
        "SS/SD/DS/DD, mixed and MKL methods x threads x slice length x "
        "selective packing x float/double, plus a rejected sparse result; "
        "%d of %d automatic runs on MKL), %d failed\n",
        failures ? "FAIL" : "PASS", cases, mkl_runs, auto_runs, failures
    );
    return failures ? 1 : 0;
}
