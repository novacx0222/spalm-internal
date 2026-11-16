// Stand-in for libopenblas in tests/openblas_baseline_test.cpp: the symbols
// openblas_baseline.h looks up, a scalar row-major GEMM, and (FAKE_THP) the
// SPALM patch's openblas_spalm_thp_buffers, which reports whether
// SPALM_OPENBLAS_THP was 1 when the library was loaded, as the patched
// allocator reads it from OpenBLAS's constructor on.
#include <cstdlib>
#include <cstring>

#ifndef FAKE_PARALLEL
#define FAKE_PARALLEL 2
#endif

static int threads = 64;
#ifdef FAKE_THP
static int thp_at_load = -1;
__attribute__((constructor)) static void fake_init() {
    const char *v = std::getenv("SPALM_OPENBLAS_THP");
    thp_at_load = v && std::strcmp(v, "1") == 0 ? 1 : 0;
}
#endif

template <typename T>
static void gemm(
    int order, int ta, int tb, int M, int N, int K, T alpha, const T *A,
    int lda, const T *B, int ldb, T beta, T *C, int ldc
) {
    // Only the call the baseline makes: row major, A not transposed, B
    // transposed (C = A * B^T), beta = 0.
    if (order != 101 || ta != 111 || tb != 112 || beta != T(0)) std::abort();
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B[j * ldb + k];
            C[i * ldc + j] = alpha * sum;
        }
}

extern "C" {
void cblas_dgemm(
    int o, int ta, int tb, int M, int N, int K, double alpha, const double *A,
    int lda, const double *B, int ldb, double beta, double *C, int ldc
) {
    gemm(o, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
void cblas_sgemm(
    int o, int ta, int tb, int M, int N, int K, float alpha, const float *A,
    int lda, const float *B, int ldb, float beta, float *C, int ldc
) {
    gemm(o, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
void openblas_set_num_threads(int n) { threads = n; }
int openblas_get_num_threads() { return threads; }
int openblas_get_parallel() { return FAKE_PARALLEL; }
char *openblas_get_corename() { return const_cast<char *>("fakecore"); }
char *openblas_get_config() { return const_cast<char *>("fake OpenBLAS"); }
#ifdef FAKE_THP
long openblas_spalm_thp_buffers() { return thp_at_load == 1 ? 7 : -1; }
#endif
}
