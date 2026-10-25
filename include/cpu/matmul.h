#include <vector>

template <typename T>
void spalm_matmul(
    std::vector<int> &R_i, std::vector<int> &R_k, std::vector<T> &R_v,
    std::vector<int> &S_j, std::vector<int> &S_k, std::vector<T> &S_v,
    const int M, const int K, const int N, const int sparse_threshold,
    const int dense_threshold, const int slice_length,
    const int tile_size_join_dim, const int tile_size_agg_dim,
    bool sparse_result, std::vector<T> &ret
);
