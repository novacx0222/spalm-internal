#include "cpu/sparse_sparse_matmul.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "csr_preparation.h"
#include "slice_decision.h"
#include "util.h"

void sparse_sparse_mm_count_results(
    const int start, const int end, const int *Rj, const int *Sa, const int *Sj,
    const int *Sj_1, const int write_val, const __m512i &write_val_reg,
    int *scratch, int *result_cnt
) {
    for (int r_kit = start; r_kit < end; r_kit++) {
        const int k = Rj[r_kit];

        const int iterations = Sj_1[k] - Sj[k];

        if (!iterations) {
            continue;
        }

        const int left = iterations & 0b1111;
        const int s_ait_start = Sj[k];
        for (int s_ait = 0; s_ait < iterations - left; s_ait += 16) {
            __m512i sagg = _mm512_loadu_si512(&Sa[s_ait + s_ait_start]);
            __m512i scratch_vals = _mm512_i32gather_epi32(sagg, scratch, 4);
            __mmask16 cmp_result =
                _mm512_cmplt_epi32_mask(write_val_reg, scratch_vals);
            (*result_cnt) += _mm_popcnt_u32(cmp_result);
            _mm512_mask_i32scatter_epi32(
                scratch, cmp_result, sagg, write_val_reg, 4
            );
        }

        __mmask16 mask = (__mmask16) ~0 >> (16 - left);
        const int s_ait = iterations - left;
        __m512i sagg = _mm512_maskz_loadu_epi32(mask, &Sa[s_ait + s_ait_start]);
        __m512i scratch_vals =
            _mm512_mask_i32gather_epi32(write_val_reg, mask, sagg, scratch, 4);
        __mmask16 cmp_result =
            _mm512_mask_cmplt_epi32_mask(mask, write_val_reg, scratch_vals);
        (*result_cnt) += _mm_popcnt_u32(cmp_result);
        _mm512_mask_i32scatter_epi32(
            scratch, mask & cmp_result, sagg, write_val_reg, 4
        );

        /*
    for (int s_ait = Sj[k]; s_ait < Sj_1[k]; s_ait++) {
            const int sagg = Sa[s_ait];
            if (write_val < scratch[sagg]) {
                    (*result_cnt)++;
                    scratch[sagg] = write_val;
            }
    }
    */
    }
}

template <typename T>
void sparse_sparse_mm_calculate_results(
    const int start, const int end, const int *Rj, const double *Rv,
    const int *Sa, const int *Sj, const int *Sj_1, const T *Sv, int *scratch,
    int *row_result_cnt, const int offset, csr<T> &result_csr
) {
    for (int r_kit = start; r_kit < end; r_kit++) {
        const int k = Rj[r_kit];

        const int iterations = Sj_1[k] - Sj[k];

        int *Sj_w_offset = &result_csr.join_idxs[offset];
        T *Sv_w_offset = &result_csr.vals[offset];

        if (!iterations) {
            continue;
        }

        for (int s_ait = Sj[k]; s_ait < Sj_1[k]; s_ait++) {
            const int sagg = Sa[s_ait];
            if (scratch[sagg] >= 0) {
                const int write_idx = scratch[sagg];
                Sv_w_offset[write_idx] += Rv[r_kit] * Sv[s_ait];
            } else {
                const int write_idx = (*row_result_cnt)++;
                scratch[sagg] = write_idx;
                Sj_w_offset[write_idx] = sagg;
                Sv_w_offset[write_idx] = Rv[r_kit] * Sv[s_ait];
            }
        }
    }
}

template <typename T>
void csr_csr_matmul_kernel(
    const int start, const int end, const int *Rj, const T *Rv, const int *Sa,
    const int *Sj, const int *Sj_1, const T *Sv, T *ret
) {
    for (int r_kit = start; r_kit < end; r_kit++) {
        const T rv = Rv[r_kit];
        const int k = Rj[r_kit];
        for (int s_ait = Sj[k]; s_ait < Sj_1[k]; s_ait++) {
            ret[Sa[s_ait]] += rv * Sv[s_ait];
        }
    }
}

void csr_csr_matmul_kernel(
    const int start, const int end, const int *Rj, const double *Rv,
    const int *Sa, const int *Sj, const int *Sj_1, const double *Sv,
    double *ret, bool contiguous
) {
    const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (int r_kit = start; r_kit < end; r_kit++) {
        const int k = Rj[r_kit];

        const int iterations = Sj_1[k] - Sj[k];

        __m128d rv_128 = _mm_set_sd(Rv[r_kit]);

        if (!iterations) {
            continue;
        } else if (!(iterations & 0xfffffff8)) {
            /*
      for (int s_ait = Sj[k]; s_ait < Sj_1[k]; s_ait++) {
        const int sagg = Sa[s_ait];
        __m128d res = _mm_fmadd_pd(_mm_loadu_pd(&Sv[s_ait]), rv_128,
                                   _mm_loadu_pd(&ret[sagg]));
        ret[sagg] = _mm_cvtsd_f64(res);
      }
      */
            const int s_ait_start = Sj[k];
            __m512d r_val = _mm512_broadcastsd_pd(rv_128);
            __mmask8 mask = (__mmask8) ~0 >> (8 - iterations);
            __m512d sv = _mm512_maskz_loadu_pd(mask, &Sv[s_ait_start]);
            __m512d subres = _mm512_mul_pd(r_val, sv);
            __m256i sagg =
                _mm256_maskz_loadu_epi32(mask, (__m256i *) &Sa[s_ait_start]);
            __m512d res = _mm512_mask_i32gather_pd(sv, mask, sagg, ret, 8);
            res = _mm512_add_pd(subres, res);
            _mm512_mask_i32scatter_pd(ret, mask, sagg, res, 8);
        } else {
            const double rv = Rv[r_kit];
            const int left = iterations & 0b111;
            const int s_ait_start = Sj[k];
            __m512d r_val = _mm512_broadcastsd_pd(rv_128);
            for (int s_ait = 0; s_ait < iterations - left; s_ait += 8) {
                const int base = s_ait + s_ait_start;
                __m512d sv = _mm512_loadu_pd(&Sv[base]);
                __m512d subres = _mm512_mul_pd(r_val, sv);

                const __m256i sagg =
                    _mm256_loadu_si256((const __m256i *) &Sa[base]);
                const int first = Sa[base];
                // Eight consecutive output columns (the indices are compared,
                // not assumed): one plain load/add/store, same arithmetic.
                if (contiguous &&
                    _mm256_cmpeq_epi32_mask(
                        sagg,
                        _mm256_add_epi32(_mm256_set1_epi32(first), lane_ids)
                    ) == 0xFF) {
                    double *dst = &ret[first];
                    _mm512_storeu_pd(
                        dst, _mm512_add_pd(subres, _mm512_loadu_pd(dst))
                    );
                } else {
                    // int32 indices; the old code widened them to int64.
                    __m512d res = _mm512_i32gather_pd(sagg, ret, 8);
                    res = _mm512_add_pd(subres, res);
                    _mm512_i32scatter_pd(ret, sagg, res, 8);
                }
            }
            __mmask8 mask = (__mmask8) ~0 >> (8 - left);
            const int s_ait = iterations - left;
            __m512d sv = _mm512_maskz_loadu_pd(mask, &Sv[s_ait + s_ait_start]);
            __m512d subres = _mm512_mul_pd(r_val, sv);
            __m256i sagg = _mm256_maskz_loadu_epi32(
                mask, (__m256i *) &Sa[s_ait + s_ait_start]
            );
            __m512d res = _mm512_mask_i32gather_pd(sv, mask, sagg, ret, 8);
            res = _mm512_add_pd(subres, res);
            _mm512_mask_i32scatter_pd(ret, mask, sagg, res, 8);
        }
    }
}

// The double kernel takes the contiguous-block switch; the generic
// (float) one has no SIMD paths.
template <typename T>
static inline void csr_csr_matmul_rows(
    const int start, const int end, const int *Rj, const T *Rv, const int *Sa,
    const int *Sj, const int *Sj_1, const T *Sv, T *ret, bool contiguous
) {
    if constexpr (std::is_same_v<T, double>) {
        csr_csr_matmul_kernel(
            start, end, Rj, Rv, Sa, Sj, Sj_1, Sv, ret, contiguous
        );
    } else {
        (void) contiguous;
        csr_csr_matmul_kernel(start, end, Rj, Rv, Sa, Sj, Sj_1, Sv, ret);
    }
}

template <typename T>
void spgemm_rows_dense_result(
    ss_matrix<T> &R_ssm, int *R_subrow_buckets, int *s_agg_idxs, T *s_vals,
    int *sj_buckets, const int N, T *ret, int threads, bool contiguous
) {
    const int M = R_ssm.csr_mat.rows;
    const int K = R_ssm.csr_mat.cols;

#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (int ra = 0; ra < M; ra++) {
        for (int si = R_ssm.sr_csr_mat.row_ptrs[ra];
             si < R_ssm.sr_csr_mat.row_ptrs[ra + 1]; si++) {
            const int bucket_idx = R_subrow_buckets[si];
            if (bucket_idx == 0) {
                continue;
            }
            const int *sj_starts = &sj_buckets[0];
            const int *sj_ends = &sj_buckets[(size_t) bucket_idx * K];
            const subrow &sr = R_ssm.sr_csr_mat.vals[si];
            csr_csr_matmul_rows(
                sr.start, sr.start + sr.len, R_ssm.csr_mat.col_idxs,
                R_ssm.csr_mat.vals, s_agg_idxs, sj_starts, sj_ends, s_vals,
                &ret[(size_t) ra * N], contiguous
            );
        }
    }
}

// Size of the cache shared by all cores (bytes), 0 if unknown.
static size_t shared_cache_bytes() {
#ifdef _SC_LEVEL3_CACHE_SIZE
    const long bytes = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (bytes > 0) return static_cast<size_t>(bytes);
#endif
    return 0;
}

// Output-column blocks for the full SS path. Without them every R row
// streams the whole transposed S (for random_col 5%: about 160 MB, more than
// the shared cache) from memory. With b blocks, one block's part of S^T,
// about total / b, is reused by all R rows while it is cached. Blocks only
// pay off when that part does stay cached: otherwise one sequential stream
// becomes many short jumps. Automatic: none when S^T already fits the
// shared cache; else the fewest blocks whose part fits a third of it (the
// rest holds the streamed R rows and output rows), at least 256 columns
// each, and none when that is impossible.
static int ss_column_block_count(
    const int requested, const size_t entries, const size_t entry_bytes,
    const int N
) {
    const int max_blocks = std::max(1, N / 256);
    if (requested >= 0) return std::max(1, std::min(requested, max_blocks));
    const size_t cache = shared_cache_bytes();
    if (cache == 0) return 1;
    const size_t total = entries * entry_bytes;
    if (total <= cache) return 1;
    const size_t target = cache / 3;
    const size_t blocks = (total + target - 1) / target;
    return blocks <= static_cast<size_t>(max_blocks) ? static_cast<int>(blocks)
                                                     : 1;
}

// starts[b * K + k], b = 0..blocks: the first entry of key k's row list
// with row >= b * width (row blocks: the list end), so block b of key k is
// [starts[b * K + k], starts[(b + 1) * K + k]). Returns false when a list is
// not in ascending row order, which the blocks rely on.
static bool ss_column_block_starts(
    const int *offsets, const int *rows, const int K, const int blocks,
    const int width, std::vector<int> &starts, const int threads
) {
    starts.resize(static_cast<size_t>(blocks + 1) * K);
    bool ascending = true;
#pragma omp parallel for num_threads(threads) schedule(static)                 \
    reduction(&& : ascending)
    for (int k = 0; k < K; ++k) {
        const int begin = offsets[k], end = offsets[k + 1];
        for (int p = begin + 1; p < end; ++p)
            if (rows[p] < rows[p - 1]) ascending = false;
        int p = begin;
        for (int b = 0; b < blocks; ++b) {
            const long long bound = static_cast<long long>(b) * width;
            while (p < end && rows[p] < bound) ++p;
            starts[static_cast<size_t>(b) * K + k] = p;
        }
        starts[static_cast<size_t>(blocks) * K + k] = end;
    }
    return ascending;
}

template <typename T>
void spgemm_rows_dense_result_vanilla(
    const int N, ss_matrix<T> &R_ssm, int *s_col_idxs, int *s_agg_idxs,
    T *s_vals, T *ret, int threads, bool contiguous, const int *block_starts,
    const int blocks
) {
    const int M = R_ssm.csr_mat.rows;
    csr<T> &R_csr = R_ssm.csr_mat;
    if (block_starts && blocks > 1) {
        // Blocks in order, each over all R rows: same updates per output
        // element, in the same order, so results are unchanged.
        const size_t K = static_cast<size_t>(R_csr.cols);
#pragma omp parallel num_threads(threads)
        for (int b = 0; b < blocks; ++b) {
            const int *first = block_starts + b * K;
            const int *last = block_starts + (b + 1) * K;
#pragma omp for schedule(dynamic, 4)
            for (int ra = 0; ra < M; ra++) {
                csr_csr_matmul_rows(
                    R_csr.row_ptrs[ra], R_csr.row_ptrs[ra + 1], R_csr.col_idxs,
                    R_csr.vals, s_agg_idxs, first, last, s_vals,
                    &ret[(size_t) ra * N], contiguous
                );
            }
        }
        return;
    }

#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (int ra = 0; ra < M; ra++) {
        csr_csr_matmul_rows(
            R_csr.row_ptrs[ra], R_csr.row_ptrs[ra + 1], R_csr.col_idxs,
            R_csr.vals, s_agg_idxs, s_col_idxs, s_col_idxs + 1, s_vals,
            &ret[(size_t) ra * N], contiguous
        );
    }
}

template <typename T>
void sparse_sparse_matmul(
    ss_matrix<T> &R_ssm, ss_matrix<T> &S_ssm, T *ret, bool sparse_output,
    const int threads, const bool all_sparse,
    const cpu_execution_options &options
) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end,
        sparse_matmul_start;
    sparse_matmul_start = std::chrono::high_resolution_clock::now();
    const int M = R_ssm.csr_mat.rows;
    const int N = S_ssm.csr_mat.rows;

    if (all_sparse) {
        auto transposed = transpose_all_sparse(S_ssm, threads);
        end = std::chrono::high_resolution_clock::now();
        print_time(
            "[sparse-sparse-s-transpose-time]", end - sparse_matmul_start
        );
        start = std::chrono::high_resolution_clock::now();
        const int K = R_ssm.csr_mat.cols;
        int blocks = ss_column_block_count(
            options.ss_column_blocks, static_cast<size_t>(transposed.entries),
            sizeof(int) + sizeof(T), N
        );
        std::vector<int> block_starts;
        if (blocks > 1 &&
            !ss_column_block_starts(
                transposed.offsets.get(), transposed.row_idxs.get(), K, blocks,
                div_ceil(N, blocks), block_starts, threads
            )) {
            blocks = 1;
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << "[ss-column-blocks] blocks=" << blocks
                  << " width=" << div_ceil(N, blocks)
                  << " shared_cache_bytes=" << shared_cache_bytes()
                  << std::endl;
        print_time("[sparse-sparse-column-blocks-prepare]", end - start);
        start = std::chrono::high_resolution_clock::now();
        spgemm_rows_dense_result_vanilla(
            N, R_ssm, transposed.offsets.get(), transposed.row_idxs.get(),
            transposed.values.get(), ret, threads, options.ss_contiguous,
            blocks > 1 ? block_starts.data() : nullptr, blocks
        );
        end = std::chrono::high_resolution_clock::now();
        print_time("[sparse-sparse-sparse-mm-time]", end - start);
        print_time(
            "[sparse-sparse-total-sparse-mm-time]", end - sparse_matmul_start
        );
        return;
    }

    // Only entries of S slices with bucket < total_buckets can be multiplied
    // here; the dense path handles the rest.
    const int total_buckets = get_total_sparse_buckets();
    if (total_buckets <= 0) return;
    auto transposed = transpose_sparse_buckets(S_ssm, total_buckets, threads);
    end = std::chrono::high_resolution_clock::now();
    print_time("[sparse-sparse-s-transpose-time]", end - sparse_matmul_start);
    if (transposed.entries == 0) return;

    start = std::chrono::high_resolution_clock::now();
    if (!sparse_output) {
        // Label each subrow in R's sr_csr_mat with its sparse threshold bucket.
        // Use sr_csc_to_csr_idx to map from slice CSC subrows back to CSR subrows.
        const int total_sr_csr_subrows = R_ssm.sr_csr_mat.row_ptrs[M];
        auto R_subrow_buckets =
            std::make_unique_for_overwrite<int[]>(total_sr_csr_subrows);
        memset(R_subrow_buckets.get(), 0, sizeof(int) * total_sr_csr_subrows);

        const int R_num_slice_cols = R_ssm.s_csc_mat.cols;
        for (int col = 0; col < R_num_slice_cols; col++) {
            for (int si = R_ssm.s_csc_mat.col_ptrs[col];
                 si < R_ssm.s_csc_mat.col_ptrs[col + 1]; si++) {
                const slice &s = R_ssm.s_csc_mat.vals[si];
                // The transpose only has bucket starts up to total_buckets;
                // clamp so a non-monotone thresholds file cannot index past it.
                const int thres = std::min<int>(ss_upper[s.b], total_buckets);
                // Label each subrow in this slice
                for (int sr_idx = s.start; sr_idx < s.start + s.len; sr_idx++) {
                    // Map CSC subrow index back to CSR subrow index
                    const int csr_idx = R_ssm.sr_csc_to_csr_idx[sr_idx];
                    R_subrow_buckets[csr_idx] = thres;
                }
            }
        }
        end = std::chrono::high_resolution_clock::now();
        print_time("[sparse-sparse-r-subrow-labeling-time]", end - start);

        spgemm_rows_dense_result(
            R_ssm, R_subrow_buckets.get(), transposed.row_idxs.get(),
            transposed.values.get(), transposed.offsets.get(), N, ret, threads,
            options.ss_contiguous
        );
    } else {
        // TODO: sparse output path
    }
    end = std::chrono::high_resolution_clock::now();
    print_time("[sparse-sparse-sparse-mm-time]", end - start);
    print_time(
        "[sparse-sparse-total-sparse-mm-time]", end - sparse_matmul_start
    );
}

template void sparse_sparse_matmul<float>(
    ss_matrix<float> &R_ssm, ss_matrix<float> &S_ssm, float *ret,
    bool sparse_output, const int threads, const bool all_sparse,
    const cpu_execution_options &options
);

template void sparse_sparse_matmul<double>(
    ss_matrix<double> &R_ssm, ss_matrix<double> &S_ssm, double *ret,
    bool sparse_output, const int threads, const bool all_sparse,
    const cpu_execution_options &options
);
