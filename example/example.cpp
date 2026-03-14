// example.cpp
// Demonstrates spalm sparse matrix multiplication on:
//   A: 8000x8000 diagonal matrix (band width = 99)
//   B: 8000x8000 edges matrix   (edge width = 200)
//
// Matrices are initialized inline, converted to CSR, then passed to spalm.so
// via dlopen. Library path defaults to /app/bin/spalm.so (Docker) and can be
// overridden with the SPALM_LIB environment variable.

#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Matrix dimensions and sparsity parameters
// ---------------------------------------------------------------------------
static constexpr int M          = 8000;
static constexpr int K          = 8000;
static constexpr int N          = 8000;
static constexpr int DIAG_WIDTH = 50;
static constexpr int EDGE_SIZE  = 200;

// ---------------------------------------------------------------------------
// Matrix initialization
// ---------------------------------------------------------------------------
static void init_diagonal(float* A, int rows, int cols, int diagsize) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<> val(1, 10);
    memset(A, 0, sizeof(float) * rows * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if ((j >= i && (j - i) < diagsize) || (i > j && (i - j) < diagsize))
                A[i * cols + j] = (float)val(gen);
}

static void init_edges(float* A, int rows, int cols, int edge_size) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<> val(1, 10);
    memset(A, 0, sizeof(float) * rows * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (i < edge_size || (rows - i) < edge_size ||
                j < edge_size || (cols - j) < edge_size)
                A[i * cols + j] = (float)val(gen);
}

// ---------------------------------------------------------------------------
// Dense (row-major) -> CSR
// ---------------------------------------------------------------------------
static void dense_to_csr(const float* A, int rows, int cols,
                          std::vector<int>& row_ptrs,
                          std::vector<int>& col_idxs,
                          std::vector<float>& vals) {
    row_ptrs.assign(rows + 1, 0);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (A[i * cols + j] != 0.0f)
                row_ptrs[i + 1]++;
    for (int i = 0; i < rows; i++)
        row_ptrs[i + 1] += row_ptrs[i];

    int nnz = row_ptrs[rows];
    col_idxs.resize(nnz);
    vals.resize(nnz);
    std::vector<int> cur(rows, 0);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (A[i * cols + j] != 0.0f) {
                int idx       = row_ptrs[i] + cur[i]++;
                col_idxs[idx] = j;
                vals[idx]     = A[i * cols + j];
            }
}

// ---------------------------------------------------------------------------
// Build CSR for B^T (N x K) directly from B (K x N, row-major).
// spalm expects the second matrix stored as B^T in CSR format.
// ---------------------------------------------------------------------------
static void dense_to_csr_transposed(const float* B, int K_dim, int N_dim,
                                     std::vector<int>& row_ptrs,
                                     std::vector<int>& col_idxs,
                                     std::vector<float>& vals) {
    row_ptrs.assign(N_dim + 1, 0);
    for (int k = 0; k < K_dim; k++)
        for (int n = 0; n < N_dim; n++)
            if (B[k * N_dim + n] != 0.0f)
                row_ptrs[n + 1]++;
    for (int n = 0; n < N_dim; n++)
        row_ptrs[n + 1] += row_ptrs[n];

    int nnz = row_ptrs[N_dim];
    col_idxs.resize(nnz);
    vals.resize(nnz);
    std::vector<int> cur(N_dim, 0);
    for (int k = 0; k < K_dim; k++)
        for (int n = 0; n < N_dim; n++)
            if (B[k * N_dim + n] != 0.0f) {
                int idx       = row_ptrs[n] + cur[n]++;
                col_idxs[idx] = k;
                vals[idx]     = B[k * N_dim + n];
            }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const char* spalm_lib =
        getenv("SPALM_LIB") ? getenv("SPALM_LIB") : "/app/bin/spalm.so";

    std::cout << "Loading spalm: " << spalm_lib << "\n\n";
    void* sp_handle = dlopen(spalm_lib, RTLD_LAZY | RTLD_GLOBAL);
    if (!sp_handle) {
        std::cerr << "dlopen spalm: " << dlerror() << "\n";
        return 1;
    }

    //   void spalm(int type, int* Ri, int* Rj, void* Rv,
    //              int* Si, int* Sj, void* Sv,
    //              int M, int K, int N, bool sparse_result,
    //              void* ret, int* reti, int* retj, void* retv,
    //              int n, const char** keys, const char** vals)
    auto spalm_fn =
        reinterpret_cast<void(*)(int, int*, int*, void*,
                                  int*, int*, void*,
                                  int, int, int, bool,
                                  void*, int*, int*, void*,
                                  int, const char**, const char**)>(
            dlsym(sp_handle, "spalm"));
    if (!spalm_fn) {
        std::cerr << "dlsym spalm: " << dlerror() << "\n";
        return 1;
    }

    // -- Initialize matrices --------------------------------------------------
    std::vector<float> A_dense((long long)M * K);
    init_diagonal(A_dense.data(), M, K, DIAG_WIDTH);

    std::vector<float> B_dense((long long)K * N);
    init_edges(B_dense.data(), K, N, EDGE_SIZE);

    // -- Convert to CSR -------------------------------------------------------
    std::vector<int> A_rp, A_ci; std::vector<float> A_v;
    dense_to_csr(A_dense.data(), M, K, A_rp, A_ci, A_v);
    A_dense.clear(); A_dense.shrink_to_fit();

    std::vector<int> Bt_rp, Bt_ci; std::vector<float> Bt_v;
    dense_to_csr_transposed(B_dense.data(), K, N, Bt_rp, Bt_ci, Bt_v);
    B_dense.clear(); B_dense.shrink_to_fit();

    // -- Run spalm ------------------------------------------------------------
    std::vector<float> C((long long)M * N, 0.0f);

    const char* keys[] = { "threads" };
    const char* vals[] = { "4"       };

    spalm_fn(/*float*/0,
             A_rp.data(),  A_ci.data(),  A_v.data(),
             Bt_rp.data(), Bt_ci.data(), Bt_v.data(),
             M, K, N,
             false,
             C.data(), nullptr, nullptr, nullptr,
             1, keys, vals);

    dlclose(sp_handle);
    return 0;
}
