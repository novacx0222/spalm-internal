#ifndef UTIL
#define UTIL

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

void print_matrix(double *mat, const int dim0, const int dim1);

template <typename T>
void coo2dense(std::vector<int> &agg_idx, std::vector<int> &join_idx,
               std::vector<T> &vals, const int agg_dim, const int join_dim,
               T *ret, const bool agg_major) {
  size_t row_num = agg_idx.size();
  if (agg_major) {
    for (size_t i = 0; i < row_num; i++) {
      ret[join_idx[i] * agg_dim + agg_idx[i]] += vals[i];
    }
  } else {
    for (size_t i = 0; i < row_num; i++) {
      ret[agg_idx[i] * join_dim + join_idx[i]] += vals[i];
    }
  }
}

inline float print_time(
    std::string msg,
    std::chrono::duration<long int, std::ratio<1, 1000000000>> time,
    bool print = true) {
  auto casted_time =
      std::chrono::duration_cast<std::chrono::microseconds>(time);
  if (print) {
    std::cout << msg << (float)casted_time.count() / 1000000 << "s"
              << std::endl;
  }
  return (float)casted_time.count() / 1000000;
}

inline int div_ceil(const int N, const int D) { return (N + D - 1) / D; }

#define BIT_1 1ULL
#define BITS 64
#define BIT_LOG2 6
#define BIT_MOD 0b111111
#define BIT_DTYPE uint64_t

inline int popcount_range(BIT_DTYPE* bits, const BIT_DTYPE start,
                          const BIT_DTYPE end) {
  unsigned int curr = start;
  int ret = 0;
  while ((curr >> BIT_LOG2) != (end >> BIT_LOG2)) {
    ret += __builtin_popcountll(bits[curr >> BIT_LOG2] >> (curr & BIT_MOD));
    curr = ((curr >> BIT_LOG2) + 1) << BIT_LOG2;
  }
  ret += __builtin_popcountll(
      (bits[curr >> BIT_LOG2] & ((1ULL << (end & BIT_MOD)) - 1)) >>
      (curr & BIT_MOD));
  return ret;
}

inline void set_bit(BIT_DTYPE* bits, const int idx) {
  bits[idx >> BIT_LOG2] |= (BIT_1 << (idx & BIT_MOD));
}

inline void set_bit_zero(BIT_DTYPE* bits, const int idx) {
  bits[idx >> BIT_LOG2] &= ~(BIT_1 << (idx & BIT_MOD));
}

inline int get_bit(BIT_DTYPE* bits, const int idx) {
  return (bits[idx >> BIT_LOG2] >> (idx & BIT_MOD)) & BIT_1;
}

#endif