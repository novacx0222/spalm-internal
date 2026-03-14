#ifndef DENSE_DENSE_MATMUL_H
#define DENSE_DENSE_MATMUL_H

#include "ss_mat.h"

template <typename T>
void dense_dense_matmul(ss_matrix<T>& R_ssm, ss_matrix<T>& S_ssm,
                        const int tile_size_l2, const int tile_size_l3, T* ret,
                        bool sparse_result, const int threads);

#endif
