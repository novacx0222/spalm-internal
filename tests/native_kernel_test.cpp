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
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using spalm_fn = int (*)(
    int, int *, int *, void *, int *, int *, void *, int, int, int, bool,
    void *, int *, int *, void *, int, const char **, const char **
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

template <typename T>
static bool run_case(
    spalm_fn spalm, const char *label, const csr &R, const csr &S, int K,
    const std::vector<std::pair<std::string, std::string>> &options
) {
    constexpr int type = sizeof(T) == sizeof(float) ? 0 : 1;
    std::vector<T> rv(R.val.begin(), R.val.end()),
        sv(S.val.begin(), S.val.end());
    std::vector<int> ri = R.ptr, rj = R.col, si = S.ptr, sj = S.col;
    // Pre-filled output: the kernels must add to it, not overwrite it.
    std::vector<T> out(size_t(R.rows) * S.rows, T(1));
    std::vector<const char *> keys, vals;
    for (auto &kv : options) {
        keys.push_back(kv.first.c_str());
        vals.push_back(kv.second.c_str());
    }
    fflush(stdout);
    const int saved = dup(1), null = open("/dev/null", O_WRONLY);
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
    const auto expected = reference<T>(R, S, K);
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != expected[i] + T(1)) {
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
            bad ? double(expected[first] + T(1)) : 0.0, opts.c_str()
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
    std::printf(
        "%s: %d native kernel cases (SS/SD/DS/DD x DS aligned panels x SS "
        "column blocks x full "
        "rows x selective "
        "packing x SS contiguous x threads x slice length (fixed / automatic) "
        "x float/double), "
        "%d failed\n",
        failures ? "FAIL" : "PASS", cases, failures
    );
    return failures ? 1 : 0;
}
