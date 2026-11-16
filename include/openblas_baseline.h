#ifndef OPENBLAS_BASELINE_H
#define OPENBLAS_BASELINE_H

// OpenBLAS dense-dense baseline (init_method FORCE_OPENBLAS_DENSE), used by
// run_baseline.so. libopenblas is opened with dlopen, RTLD_LOCAL: MKL, which
// the same process has loaded, exports the same cblas_?gemm names, so every
// OpenBLAS function is called through the pointers looked up here.
//
// Fairness with MKL dense (notes/openblas-baseline/plan.md):
// - OpenBLAS must be built with USE_OPENMP=1 (openblas_get_parallel() == 2):
//   its GEMM then runs as a parallel region of the calling thread on the
//   process's one libgomp, the same 30-thread team MKL's gnu_thread layer
//   and the SPALM kernels use. A pthreads build would add its own pool of
//   busy-waiting threads; it is rejected.
// - openblas_set_num_threads(threads), as mkl_set_num_threads(threads).
//   OpenBLAS's OpenMP server sizes its team from the caller's OpenMP thread
//   count, which is set to the same value.
// - OpenBLAS allocates one GEMM buffer per thread when it is loaded (its
//   constructor), with mmap on 4 KiB pages. With huge_pages, the
//   SPALM-patched build (scripts/openblas_thp_patch.py) aligns them to
//   2 MiB and marks them MADV_HUGEPAGE, as MKL's buffers are through
//   i_malloc (huge_pages.h). The patch reads SPALM_OPENBLAS_THP, which is
//   therefore set before dlopen; the choice holds for the whole process.

#include <dlfcn.h>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

namespace spalm_openblas {

// The directory next to bin/ that scripts/build_openblas.sh installs into.
inline constexpr const char *kVersionDir = "OpenBLAS-0.3.30";
inline constexpr const char *kThpEnv = "SPALM_OPENBLAS_THP";
// CBLAS enum values (cblas.h, the same in MKL and OpenBLAS).
inline constexpr int kRowMajor = 101, kNoTrans = 111, kTrans = 112;
// openblas_get_parallel(): 0 sequential, 1 pthreads, 2 OpenMP.
inline constexpr int kParallelOpenmp = 2;

using dgemm_fn = void (*)(
    int, int, int, int, int, int, double, const double *, int, const double *,
    int, double, double *, int
);
using sgemm_fn = void (*)(
    int, int, int, int, int, int, float, const float *, int, const float *,
    int, float, float *, int
);

struct library {
    void *handle = nullptr;
    dgemm_fn dgemm = nullptr;
    sgemm_fn sgemm = nullptr;
    void (*set_num_threads)(int) = nullptr;
    int (*get_num_threads)() = nullptr;
    int (*get_parallel)() = nullptr;
    char *(*get_corename)() = nullptr;
    char *(*get_config)() = nullptr;
    // Only in the SPALM-patched build: buffers put on huge pages so far,
    // or -1 when SPALM_OPENBLAS_THP was not 1 at load.
    long (*thp_buffers)() = nullptr;
    bool huge_pages = false;
    std::string path;
};

template <typename F> static bool lookup(void *handle, const char *name, F &f) {
    f = reinterpret_cast<F>(dlsym(handle, name));
    return f != nullptr;
}

// Loads libopenblas from `path` once per process. The first call decides
// huge_pages (OpenBLAS allocates its buffers while it is loaded); a later
// call asking for the other setting fails. Returns nullptr with `error` set
// on failure; a failed load is retried on the next call. load_ms is the
// time of this call's dlopen and lookups (0 when already loaded).
inline const library *load(
    const std::string &path, bool huge_pages, std::string &error,
    double &load_ms
) {
    static std::mutex lock;
    static library loaded;
    std::lock_guard<std::mutex> guard(lock);
    load_ms = 0;
    if (loaded.handle) {
        if (loaded.huge_pages != huge_pages) {
            error = std::string("OpenBLAS was loaded with huge_pages=") +
                    (loaded.huge_pages ? "1" : "0") +
                    " in this process; openblas_huge_pages cannot change";
            return nullptr;
        }
        return &loaded;
    }
    const auto begin = std::chrono::steady_clock::now();
    // Read by the patched allocator from the constructor on. Overwritten so
    // that spalm.conf decides, not the environment.
    setenv(kThpEnv, huge_pages ? "1" : "0", 1);
    void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char *why = dlerror();
        error = "cannot open " + path + ": " + (why ? why : "unknown error");
        return nullptr;
    }
    library lib;
    lib.handle = handle;
    lib.path = path;
    lib.huge_pages = huge_pages;
    const char *missing = nullptr;
    if (!lookup(handle, "cblas_dgemm", lib.dgemm)) missing = "cblas_dgemm";
    else if (!lookup(handle, "cblas_sgemm", lib.sgemm)) missing = "cblas_sgemm";
    else if (!lookup(handle, "openblas_set_num_threads", lib.set_num_threads))
        missing = "openblas_set_num_threads";
    else if (!lookup(handle, "openblas_get_num_threads", lib.get_num_threads))
        missing = "openblas_get_num_threads";
    else if (!lookup(handle, "openblas_get_parallel", lib.get_parallel))
        missing = "openblas_get_parallel";
    else if (!lookup(handle, "openblas_get_corename", lib.get_corename))
        missing = "openblas_get_corename";
    else if (!lookup(handle, "openblas_get_config", lib.get_config))
        missing = "openblas_get_config";
    lookup(handle, "openblas_spalm_thp_buffers", lib.thp_buffers);
    // Not dlclose'd on failure: OpenBLAS's destructor would run while its
    // buffers may already be registered; the mapping is harmless.
    if (missing) {
        error = path + " lacks " + missing;
        return nullptr;
    }
    if (lib.get_parallel() != kParallelOpenmp) {
        error = path + " is not an OpenMP build (openblas_get_parallel() = " +
                std::to_string(lib.get_parallel()) +
                "); build it with USE_OPENMP=1";
        return nullptr;
    }
    if (huge_pages && !lib.thp_buffers) {
        error = "openblas_huge_pages=1 needs the SPALM-patched OpenBLAS "
                "(openblas_spalm_thp_buffers missing in " +
                path + ")";
        return nullptr;
    }
    loaded = lib;
    load_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - begin
    )
                  .count();
    return &loaded;
}

// One line describing the library this call runs on (not a "<n> ms" line).
inline void print_config(const library &lib) {
    const long thp = lib.thp_buffers ? lib.thp_buffers() : -2;
    std::cout << "[openblas-config] parallel=" << lib.get_parallel()
              << " corename=" << lib.get_corename()
              << " threads=" << lib.get_num_threads()
              << " omp_max_threads=" << omp_get_max_threads()
              << " huge_pages=" << (lib.huge_pages ? 1 : 0) << " thp_buffers="
              << (thp == -2 ? std::string("unpatched") : std::to_string(thp))
              << " path=" << lib.path << " config=" << lib.get_config()
              << std::endl;
}

// ret = R * S^T with R M x K and S N x K, both dense and row major: the
// same call as MKL dense-dense (beta = 0, so ret need not be zeroed).
// type 0 = float, 1 = double.
inline void gemm(
    const library &lib, int type, void *R, void *S, void *ret, int M, int K,
    int N
) {
    if (type == 0) {
        lib.sgemm(
            kRowMajor, kNoTrans, kTrans, M, N, K, 1.0f,
            static_cast<const float *>(R), K, static_cast<const float *>(S), K,
            0.0f, static_cast<float *>(ret), N
        );
    } else {
        lib.dgemm(
            kRowMajor, kNoTrans, kTrans, M, N, K, 1.0,
            static_cast<const double *>(R), K, static_cast<const double *>(S),
            K, 0.0, static_cast<double *>(ret), N
        );
    }
}

// The whole baseline call: load (first call only), threads, configuration
// line, GEMM. Stage lines: [openblas-load] (when it loaded),
// [openblas-dense-mm-time], [openblas-baseline-total]. Returns 0, or 1
// after printing why on stderr.
template <typename PrintTime>
int run(
    const std::string &path, int type, void *R, void *S, void *ret, int M,
    int K, int N, int threads, bool huge_pages, PrintTime print_time
) {
    std::cout << "OpenBLAS dense-dense" << std::endl;
    std::cout << M << " " << K << " " << N << std::endl;
    std::cout << "THREADS: " << threads << std::endl;
    const auto e2e_start = std::chrono::high_resolution_clock::now();
    if (type != 0 && type != 1) {
        std::cerr << "OpenBLAS baseline: type must be 0 or 1" << std::endl;
        return 1;
    }
    std::string error;
    double load_ms = 0;
    const library *lib = load(path, huge_pages, error, load_ms);
    if (!lib) {
        std::cerr << "OpenBLAS baseline: " << error << std::endl;
        return 1;
    }
    if (load_ms > 0) {
        print_time(
            "[openblas-load]", std::chrono::duration<double, std::milli>(load_ms)
        );
    }
    // The caller's OpenMP thread count sizes OpenBLAS's team; the operator
    // already set it to the same value.
    omp_set_num_threads(threads);
    lib->set_num_threads(threads);
    print_config(*lib);
    const auto start = std::chrono::high_resolution_clock::now();
    gemm(*lib, type, R, S, ret, M, K, N);
    const auto end = std::chrono::high_resolution_clock::now();
    print_time("[openblas-dense-mm-time]", end - start);
    print_time("[openblas-baseline-total]", end - e2e_start);
    return 0;
}

} // namespace spalm_openblas

#endif
