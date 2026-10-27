#include "cpu/dense_dense_matmul.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <thread>
#include <unordered_set>
#include <vector>

#include "cpu/kernel.h"
#include "slice_decision.h"
#include "util.h"

// Custom deleter for aligned memory allocated with posix_memalign
struct AlignedDeleter {
    void operator()(void *ptr) const { free(ptr); }
};

template <typename T>
using aligned_unique_ptr = std::unique_ptr<T[], AlignedDeleter>;

// Aligned allocation; returns nullptr on failure (safe inside OpenMP regions).
template <typename T>
aligned_unique_ptr<T> try_make_aligned(size_t count, size_t alignment = 64) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, sizeof(T) * count) != 0) return nullptr;
    return aligned_unique_ptr<T>(static_cast<T *>(ptr));
}

// Aligned allocation that throws std::bad_alloc on failure.
template <typename T>
aligned_unique_ptr<T> make_aligned_unique(size_t count, size_t alignment = 64) {
    auto ptr = try_make_aligned<T>(count, alignment);
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

template <typename T>
void dense_left_inner_kernel(
    const slice &R_s, ss_matrix<T> &R_ssm, const slice &S_s,
    const int *S_row_idxs, const int nr, const T *packed_mem, const int M,
    const int N, T *ret, const int k_offset, const bool full_rows
);

template <typename T>
void dense_right_inner_kernel(
    const slice &S_s, ss_matrix<T> &S_ssm, const slice &R_s,
    const int *R_row_idxs, const int mr, const T *packed_mem, const int M,
    const int N, T *ret, __m512i gather_idx_512, const int k_offset,
    const bool full_rows
);

// Which packed representations the dense-path algorithms of this call read:
// SD (left inner kernel) reads packed S, DS (right inner kernel) packed R,
// DD both plus their bitmaps. Without selective packing, everything is
// packed as before.
struct dense_packing_policy {
    bool pack_r = true;
    bool pack_s = true;
    bool bitmaps = true;
    // The one algorithm of every slice pair (no SS pairs), or -1 (mixed).
    int fixed_alg = -1;
    // Values per key in packed R: the slice height (12, what DD needs), or
    // 16 when only DS reads it, so each packed row is 128 bytes and the two
    // 64-byte loads per key are cache-line aligned in double.
    int r_stride = 0;
};

static dense_packing_policy get_dense_packing_policy(bool selective) {
    dense_packing_policy policy;
    bool used[4] = {};
    for (int left = 0; left <= num_buckets; ++left)
        for (int right = 0; right <= num_buckets; ++right)
            used[get_alg(left, right)] = true;
    const int algorithms = used[SPARSE_SPARSE] + used[DENSE_DENSE] +
                           used[DENSE_DENSE_LEFT_INNER_KERNEL] +
                           used[DENSE_DENSE_RIGHT_INNER_KERNEL];
    if (algorithms == 1)
        for (int alg = 0; alg < 4; ++alg)
            if (used[alg]) policy.fixed_alg = alg;
    if (!selective) return policy;
    policy.pack_r = used[DENSE_DENSE] || used[DENSE_DENSE_RIGHT_INNER_KERNEL];
    policy.pack_s = used[DENSE_DENSE] || used[DENSE_DENSE_LEFT_INNER_KERNEL];
    policy.bitmaps = used[DENSE_DENSE];
    return policy;
}

// FixedAlg >= 0: the algorithm is known for every slice pair, so the
// per-pair get_alg switch is resolved at compile time.
template <typename T, int FixedAlg>
void inline dense_dense_matmul_task_shared(
    const int slice_col, const int R_slice_start, const int R_slice_start_task,
    const int R_slice_end, const int S_slice_start, const int S_slice_end,
    T *R_packed_mem, BIT_DTYPE *R_bm, T *S_packed_mem, BIT_DTYPE *S_bm,
    ss_matrix<T> &R_ssm, ss_matrix<T> &S_ssm, T *ret, bool sparse_result,
    const dense_packing_policy &policy, const bool full_rows
) {
    const int M = R_ssm.csr_mat.rows;
    const int K = R_ssm.csr_mat.cols;
    const int N = S_ssm.csr_mat.rows;

    const int mr = policy.r_stride;
    const int nr = S_ssm.sh;

    const int sl = R_ssm.sl;

    const int k_offset = slice_col * sl;
    const int row_offset = 0; // unused by inner_kernel_bm; was uninitialized
    // One bitmap stride per slice: (s_idx * sl) / BITS made slices share
    // bitmap words when sl < BITS.
    const int bm_words = div_ceil(sl, BITS);

    __m512i gather_idx = _mm512_set_epi32(
        0, 0, 0, 0, N * 11, N * 10, N * 9, N * 8, N * 7, N * 6, N * 5, N * 4,
        N * 3, N * 2, N, 0
    );

    if (policy.pack_s && S_slice_end > S_slice_start) {
        if (policy.bitmaps) {
            memset(
                S_bm, 0,
                sizeof(BIT_DTYPE) * (S_slice_end - S_slice_start) * bm_words
            );
        }
        for (int S_si = S_slice_start; S_si < S_slice_end; S_si++) {
            const int s_idx = S_si - S_slice_start;
            memset(&S_packed_mem[s_idx * sl * nr], 0, sizeof(T) * sl * nr);
            const slice &S_s = S_ssm.s_csc_mat.vals[S_si];
            if (policy.bitmaps) {
                pack_slice_new(
                    S_s, S_ssm, k_offset, sl, nr,
                    &S_packed_mem[s_idx * sl * nr], &S_bm[s_idx * bm_words]
                );
            } else {
                pack_slice_new<T, false>(
                    S_s, S_ssm, k_offset, sl, nr,
                    &S_packed_mem[s_idx * sl * nr], nullptr
                );
            }
        }
    }

    for (int R_si = R_slice_start_task; R_si < R_slice_end; R_si++) {
        const int R_s_idx = R_si - R_slice_start;
        const slice &R_s = R_ssm.s_csc_mat.vals[R_si];
        const int *R_row_idxs = &R_ssm.sr_csc_mat.row_idxs[R_s.start];
        for (int S_si = S_slice_start; S_si < S_slice_end; S_si++) {
            const int S_s_idx = S_si - S_slice_start;
            const slice &S_s = S_ssm.s_csc_mat.vals[S_si];
            const int *S_row_idxs = &S_ssm.sr_csc_mat.row_idxs[S_s.start];
            const int alg = FixedAlg >= 0 ? FixedAlg : get_alg(R_s.b, S_s.b);
            switch (alg) {
            case SPARSE_SPARSE:
                break;
            case DENSE_DENSE_RIGHT_INNER_KERNEL:
                // Pass the panel itself and k_offset; the kernels index with
                // panel-local keys instead of a pointer before the buffer.
                dense_right_inner_kernel(
                    S_s, S_ssm, R_s, R_row_idxs, mr,
                    &R_packed_mem[R_s_idx * sl * mr], M, N, ret, gather_idx,
                    k_offset, full_rows
                );
                break;
            case DENSE_DENSE_LEFT_INNER_KERNEL:
                dense_left_inner_kernel(
                    R_s, R_ssm, S_s, S_row_idxs, nr,
                    &S_packed_mem[S_s_idx * sl * nr], M, N, ret, k_offset,
                    full_rows
                );
                break;
            case DENSE_DENSE:
                inner_kernel_bm(
                    &R_packed_mem[R_s_idx * sl * mr], &R_bm[R_s_idx * bm_words],
                    &S_packed_mem[S_s_idx * sl * nr], &S_bm[S_s_idx * bm_words],
                    sl, 1, 1, R_s, R_row_idxs, S_s, S_row_idxs, M, N, ret,
                    sparse_result, nullptr, row_offset
                );
                break;
            }
        }
    }
}

static const int divide_rule[][2] = {
    {0, 0},  {1, 1},  {1, 2},  {1, 3},  {2, 2},  {1, 5},  {2, 3},  {1, 7},
    {2, 4},  {3, 3},  {2, 5},  {1, 11}, {2, 6},  {1, 13}, {2, 7},  {3, 5},
    {16, 1}, {1, 17}, {3, 6},  {1, 19}, {4, 5},  {3, 7},  {2, 11}, {1, 23},
    {4, 6},  {5, 5},  {2, 13}, {3, 9},  {4, 7},  {1, 29}, {5, 6},  {1, 31},
    {4, 8},  {3, 11}, {2, 17}, {5, 7},  {6, 6},  {1, 37}, {2, 19}, {3, 13},
    {5, 8},  {1, 41}, {6, 7},  {1, 43}, {4, 11}, {5, 9},  {2, 23}, {1, 47},
    {6, 8},  {7, 7},  {5, 10}, {3, 17}, {4, 13}, {1, 53}, {6, 9},  {5, 11},
    {7, 8},  {3, 19}, {2, 29}, {1, 59}, {6, 10}, {1, 61}, {2, 31}, {7, 9},
    {8, 8},
};

template <typename T>
void dense_dense_matmul(
    ss_matrix<T> &R_ssm, ss_matrix<T> &S_ssm, const int tile_size_l2,
    const int tile_size_l3, T *ret, bool sparse_result, const int threads,
    const cpu_execution_options &options
) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    auto dense_mm_start = std::chrono::high_resolution_clock::now();
    const int M = R_ssm.csr_mat.rows;
    const int K = R_ssm.csr_mat.cols;
    const int N = S_ssm.csr_mat.rows;

    auto mm_start = std::chrono::high_resolution_clock::now();

    const int sl = R_ssm.sl;
    const int mr = R_ssm.sh;
    const int nr = S_ssm.sh;

    auto mem_alloc_start = std::chrono::high_resolution_clock::now();

    const int num_slice_cols = R_ssm.s_csc_mat.cols;

    // Size the packing buffers by the largest slice columns that take part
    // (both sides non-empty), not only by the configured tile sizes.
    int max_slices_in_col_R = 0, max_slices_in_col_S = 0;
    for (int col = 0; col < num_slice_cols; col++) {
        const int r_count =
            R_ssm.s_csc_mat.col_ptrs[col + 1] - R_ssm.s_csc_mat.col_ptrs[col];
        const int s_count =
            S_ssm.s_csc_mat.col_ptrs[col + 1] - S_ssm.s_csc_mat.col_ptrs[col];
        if (!r_count || !s_count) continue;
        max_slices_in_col_R = std::max(max_slices_in_col_R, r_count);
        max_slices_in_col_S = std::max(max_slices_in_col_S, s_count);
    }
    if (!max_slices_in_col_R || !max_slices_in_col_S) return;

    const int S_slices_per_agg_tile =
        std::min(div_ceil(tile_size_l2, nr), max_slices_in_col_S);
    const int R_slices_per_l3 =
        std::min(div_ceil(tile_size_l3, mr), max_slices_in_col_R);
    const int bm_words = div_ceil(sl, BITS);
    dense_packing_policy policy =
        get_dense_packing_policy(options.selective_packing);
    policy.r_stride =
        options.ds_aligned_panels && policy.pack_r && !policy.bitmaps ? 16 : mr;
    const int r_stride = policy.r_stride;
    const size_t R_block_size = (size_t) R_slices_per_l3 * sl * r_stride;
    const size_t R_bm_size = (size_t) R_slices_per_l3 * bm_words;
    // The SIMD kernels read past the last packed k row: DS loads 16 lanes
    // from 12-row R panels (up to 20 extra floats), and the DD kernel reads
    // one element past the S panel. Keep a zeroed tail after each buffer.
    constexpr size_t pad_bytes = 128;
    const size_t pad = pad_bytes / sizeof(T);
    const bool full_rows = options.sd_ds_full_rows;
    std::cout << "[dense-packing-policy] pack_r=" << policy.pack_r
              << " pack_s=" << policy.pack_s << " bitmaps=" << policy.bitmaps
              << " fixed_alg=" << policy.fixed_alg
              << " sd_ds_full_rows=" << full_rows << " r_stride=" << r_stride
              << std::endl;
    aligned_unique_ptr<T> R_packed_mem;
    if (policy.pack_r) {
        R_packed_mem = make_aligned_unique<T>(R_block_size + pad);
        std::memset(R_packed_mem.get() + R_block_size, 0, pad_bytes);
    }
    aligned_unique_ptr<BIT_DTYPE> R_bm;
    if (policy.bitmaps) R_bm = make_aligned_unique<BIT_DTYPE>(R_bm_size);

    auto mem_alloc_end = std::chrono::high_resolution_clock::now();
    print_time(
        "[dense-dense-memory-allocation-time]", mem_alloc_end - mem_alloc_start
    );

    // Per-worker S packing buffers are allocated on a worker's first task
    // rather than for every thread up front. An allocation failure inside the
    // parallel region must not throw (that terminates the process): it is
    // flagged, remaining tasks are skipped, and std::bad_alloc is thrown after.
    const size_t S_block_size = (size_t) S_slices_per_agg_tile * sl * nr;
    const size_t S_bm_size = (size_t) S_slices_per_agg_tile * bm_words;
    std::vector<aligned_unique_ptr<T>> S_packed_mem_threads(threads);
    std::vector<aligned_unique_ptr<BIT_DTYPE>> S_bm_threads(threads);
    std::atomic<bool> alloc_failed{false};

    omp_set_dynamic(0);
#pragma omp parallel num_threads(threads)
    {
        for (int R_slice_outer_it = 0; R_slice_outer_it < max_slices_in_col_R;
             R_slice_outer_it += R_slices_per_l3) {
            for (int slice_col = 0; slice_col < num_slice_cols; slice_col++) {
                const int k_offset = slice_col * sl;

                const int R_si_l3 =
                    R_ssm.s_csc_mat.col_ptrs[slice_col] + R_slice_outer_it;
                const int R_si_l3_end = std::min(
                    R_si_l3 + R_slices_per_l3,
                    R_ssm.s_csc_mat.col_ptrs[slice_col + 1]
                );
                // Skip panels with no R slices left or no S slices: nothing
                // to pack or multiply. Same decision on every thread.
                if (R_si_l3 >= R_si_l3_end ||
                    S_ssm.s_csc_mat.col_ptrs[slice_col] ==
                        S_ssm.s_csc_mat.col_ptrs[slice_col + 1]) {
                    continue;
                }

                // Packed R is read only by DS and DD (the same decision on
                // every thread, so the loop and its barrier are skipped
                // together).
                if (policy.pack_r) {
#pragma omp for schedule(static)
                    for (int R_si = R_si_l3; R_si < R_si_l3_end; R_si++) {
                        const int s_idx = R_si - R_si_l3;
                        memset(
                            &R_packed_mem[s_idx * sl * r_stride], 0,
                            sizeof(T) * sl * r_stride
                        );
                        const slice &R_s = R_ssm.s_csc_mat.vals[R_si];
                        if (policy.bitmaps) {
                            memset(
                                &R_bm[s_idx * bm_words], 0,
                                sizeof(BIT_DTYPE) * bm_words
                            );
                            pack_slice_new(
                                R_s, R_ssm, k_offset, sl, r_stride,
                                &R_packed_mem[s_idx * sl * r_stride],
                                &R_bm[s_idx * bm_words]
                            );
                        } else {
                            pack_slice_new<T, false>(
                                R_s, R_ssm, k_offset, sl, r_stride,
                                &R_packed_mem[s_idx * sl * r_stride], nullptr
                            );
                        }
                    }
                }

                const int task_num = div_ceil(
                    S_ssm.s_csc_mat.col_ptrs[slice_col + 1] -
                        S_ssm.s_csc_mat.col_ptrs[slice_col],
                    S_slices_per_agg_tile
                );
                const int R_div = div_ceil(threads, task_num);
                const int R_slices_per_it =
                    div_ceil(R_si_l3_end - R_si_l3, R_div);

#pragma omp for collapse(2) schedule(dynamic, 1)
                for (int R_slice_start_task = R_si_l3;
                     R_slice_start_task < R_si_l3_end;
                     R_slice_start_task += R_slices_per_it) {
                    for (int S_si = S_ssm.s_csc_mat.col_ptrs[slice_col];
                         S_si < S_ssm.s_csc_mat.col_ptrs[slice_col + 1];
                         S_si += S_slices_per_agg_tile) {
                        if (alloc_failed.load(std::memory_order_relaxed)) {
                            continue;
                        }
                        const int S_slice_end = std::min(
                            S_si + S_slices_per_agg_tile,
                            S_ssm.s_csc_mat.col_ptrs[slice_col + 1]
                        );

                        const int R_slice_end = std::min(
                            R_si_l3_end, R_slice_start_task + R_slices_per_it
                        );

                        const int worker = omp_get_thread_num();
                        if (policy.pack_s && !S_packed_mem_threads[worker]) {
                            S_packed_mem_threads[worker] =
                                try_make_aligned<T>(S_block_size + pad);
                            if (policy.bitmaps) {
                                S_bm_threads[worker] =
                                    try_make_aligned<BIT_DTYPE>(S_bm_size);
                            }
                            if (!S_packed_mem_threads[worker] ||
                                (policy.bitmaps && !S_bm_threads[worker])) {
                                S_packed_mem_threads[worker].reset();
                                alloc_failed.store(
                                    true, std::memory_order_relaxed
                                );
                                continue;
                            }
                            std::memset(
                                S_packed_mem_threads[worker].get() +
                                    S_block_size,
                                0, pad_bytes
                            );
                        }
                        auto task = [&](auto fixed) {
                            dense_dense_matmul_task_shared<
                                T, decltype(fixed)::value>(
                                slice_col, R_si_l3, R_slice_start_task,
                                R_slice_end, S_si, S_slice_end,
                                R_packed_mem.get(), R_bm.get(),
                                S_packed_mem_threads[worker].get(),
                                S_bm_threads[worker].get(), R_ssm, S_ssm, ret,
                                sparse_result, policy, full_rows
                            );
                        };
                        switch (policy.fixed_alg) {
                        case DENSE_DENSE:
                            task(std::integral_constant<int, DENSE_DENSE>{});
                            break;
                        case DENSE_DENSE_LEFT_INNER_KERNEL:
                            task(
                                std::integral_constant<
                                    int, DENSE_DENSE_LEFT_INNER_KERNEL>{}
                            );
                            break;
                        case DENSE_DENSE_RIGHT_INNER_KERNEL:
                            task(
                                std::integral_constant<
                                    int, DENSE_DENSE_RIGHT_INNER_KERNEL>{}
                            );
                            break;
                        default:
                            task(std::integral_constant<int, -1>{});
                        }
                    }
                }
            }
        }
    }
    if (alloc_failed.load()) throw std::bad_alloc();

    auto mm_end = std::chrono::high_resolution_clock::now();
    print_time("[dense-dense-dense-mm-time]", mm_end - mem_alloc_end);
    print_time("[dense-dense-total-dense-mm-time]", mm_end - dense_mm_start);
}

// packed_mem is the panel itself; keys are made local with k_offset.
void inline sparse_dense_inner_kernel_iter(
    const subrow &sr, csr<double> &csr_mat, const double *packed_mem,
    const int nr, const int k_offset, __m512d &res0, __m512d &res1
) {
    res0 = _mm512_setzero_pd();
    res1 = _mm512_setzero_pd();

    const int unroll_factor = 4;
    const int sr_end = sr.start + sr.len;

    int k_idx = sr.start;
    int iters = sr.len / unroll_factor;
    int k_end = k_idx + iters * unroll_factor;

    for (; k_idx < k_end; k_idx += unroll_factor) {
        int k = csr_mat.col_idxs[k_idx] - k_offset;
        __m512d r_val = _mm512_broadcastsd_pd(_mm_set_sd(csr_mat.vals[k_idx]));
        __m512d s_vals0 = _mm512_loadu_pd(&packed_mem[nr * k]);
        __m512d s_vals1 = _mm512_loadu_pd(&packed_mem[nr * k + 8]);
        res0 = _mm512_fmadd_pd(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_pd(r_val, s_vals1, res1);

        k = csr_mat.col_idxs[k_idx + 1] - k_offset;
        r_val = _mm512_broadcastsd_pd(_mm_set_sd(csr_mat.vals[k_idx + 1]));
        s_vals0 = _mm512_loadu_pd(&packed_mem[nr * k]);
        s_vals1 = _mm512_loadu_pd(&packed_mem[nr * k + 8]);
        res0 = _mm512_fmadd_pd(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_pd(r_val, s_vals1, res1);

        k = csr_mat.col_idxs[k_idx + 2] - k_offset;
        r_val = _mm512_broadcastsd_pd(_mm_set_sd(csr_mat.vals[k_idx + 2]));
        s_vals0 = _mm512_loadu_pd(&packed_mem[nr * k]);
        s_vals1 = _mm512_loadu_pd(&packed_mem[nr * k + 8]);
        res0 = _mm512_fmadd_pd(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_pd(r_val, s_vals1, res1);

        k = csr_mat.col_idxs[k_idx + 3] - k_offset;
        r_val = _mm512_broadcastsd_pd(_mm_set_sd(csr_mat.vals[k_idx + 3]));
        s_vals0 = _mm512_loadu_pd(&packed_mem[nr * k]);
        s_vals1 = _mm512_loadu_pd(&packed_mem[nr * k + 8]);
        res0 = _mm512_fmadd_pd(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_pd(r_val, s_vals1, res1);
    }
    for (; k_idx < sr_end; k_idx++) {
        int k = csr_mat.col_idxs[k_idx] - k_offset;
        __m512d r_val = _mm512_broadcastsd_pd(_mm_set_sd(csr_mat.vals[k_idx]));
        __m512d s_vals0 = _mm512_loadu_pd(&packed_mem[nr * k]);
        __m512d s_vals1 = _mm512_loadu_pd(&packed_mem[nr * k + 8]);
        res0 = _mm512_fmadd_pd(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_pd(r_val, s_vals1, res1);
    }
}

void inline sparse_dense_inner_kernel_iter(
    const subrow &sr, csr<float> &csr_mat, const float *packed_mem,
    const int nr, const int k_offset, __m512 &res0, __m512 &res1
) {
    res0 = _mm512_setzero_ps();
    res1 = _mm512_setzero_ps();

    const int unroll_factor = 4;
    const int sr_end = sr.start + sr.len;

    int k_idx = sr.start;
    int iters = sr.len / unroll_factor;
    int k_end = k_idx + iters * unroll_factor;

    for (; k_idx < k_end; k_idx += unroll_factor) {
        int k = csr_mat.col_idxs[k_idx] - k_offset;
        __m512 r_val = _mm512_broadcastss_ps(_mm_set_ss(csr_mat.vals[k_idx]));
        __m512 s_vals0 = _mm512_loadu_ps(&packed_mem[nr * k]);
        __m512 s_vals1 = _mm512_loadu_ps(&packed_mem[nr * k + 16]);
        res0 = _mm512_fmadd_ps(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_ps(r_val, s_vals1, res1);

        k = csr_mat.col_idxs[k_idx + 1] - k_offset;
        r_val = _mm512_broadcastss_ps(_mm_set_ss(csr_mat.vals[k_idx + 1]));
        s_vals0 = _mm512_loadu_ps(&packed_mem[nr * k]);
        s_vals1 = _mm512_loadu_ps(&packed_mem[nr * k + 16]);
        res0 = _mm512_fmadd_ps(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_ps(r_val, s_vals1, res1);

        k = csr_mat.col_idxs[k_idx + 2] - k_offset;
        r_val = _mm512_broadcastss_ps(_mm_set_ss(csr_mat.vals[k_idx + 2]));
        s_vals0 = _mm512_loadu_ps(&packed_mem[nr * k]);
        s_vals1 = _mm512_loadu_ps(&packed_mem[nr * k + 16]);
        res0 = _mm512_fmadd_ps(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_ps(r_val, s_vals1, res1);

        k = csr_mat.col_idxs[k_idx + 3] - k_offset;
        r_val = _mm512_broadcastss_ps(_mm_set_ss(csr_mat.vals[k_idx + 3]));
        s_vals0 = _mm512_loadu_ps(&packed_mem[nr * k]);
        s_vals1 = _mm512_loadu_ps(&packed_mem[nr * k + 16]);
        res0 = _mm512_fmadd_ps(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_ps(r_val, s_vals1, res1);
    }
    for (; k_idx < sr_end; k_idx++) {
        int k = csr_mat.col_idxs[k_idx] - k_offset;
        __m512 r_val = _mm512_broadcastss_ps(_mm_set_ss(csr_mat.vals[k_idx]));
        __m512 s_vals0 = _mm512_loadu_ps(&packed_mem[nr * k]);
        __m512 s_vals1 = _mm512_loadu_ps(&packed_mem[nr * k + 16]);
        res0 = _mm512_fmadd_ps(r_val, s_vals0, res0);
        res1 = _mm512_fmadd_ps(r_val, s_vals1, res1);
    }
}

// SIMD operations of the full-subrow helper below.
template <typename T> struct packed_simd;

template <> struct packed_simd<double> {
    using reg = __m512d;
    static constexpr int lanes = 8;
    static inline reg zero() { return _mm512_setzero_pd(); }
    static inline reg broadcast(const double *v) { return _mm512_set1_pd(*v); }
    static inline reg load(const double *p) { return _mm512_loadu_pd(p); }
    static inline reg fmadd(reg a, reg b, reg c) {
        return _mm512_fmadd_pd(a, b, c);
    }
};

template <> struct packed_simd<float> {
    using reg = __m512;
    static constexpr int lanes = 16;
    static inline reg zero() { return _mm512_setzero_ps(); }
    static inline reg broadcast(const float *v) { return _mm512_set1_ps(*v); }
    static inline reg load(const float *p) { return _mm512_loadu_ps(p); }
    static inline reg fmadd(reg a, reg b, reg c) {
        return _mm512_fmadd_ps(a, b, c);
    }
};

// Keys of this slice column, or 0 when full subrows cannot be recognized
// (path disabled or a row repeats a key). A subrow of that length holds every
// key of the column once, in order: its k-th value belongs to packed row k.
template <typename T>
static inline int full_subrow_keys(
    const ss_matrix<T> &sparse, const int k_offset, const bool enabled
) {
    if (!enabled || !sparse.keys_strictly_increasing) return 0;
    return std::max(0, std::min(sparse.sl, sparse.csr_mat.cols - k_offset));
}

// Rows full subrows at once: each packed row (one or two vectors) is loaded
// once and multiplied with every subrow's value for that key, instead of once
// per subrow; SD/DS are bound by these loads from L2. Every output still
// adds its entries in key order, so results are unchanged.
template <typename T, bool Two, int Rows>
static inline __attribute__((always_inline)) void packed_spmm_full_rows(
    const T *const *values, const int keys, const T *packed, const int stride,
    typename packed_simd<T>::reg *acc0, typename packed_simd<T>::reg *acc1
) {
    using S = packed_simd<T>;
    for (int k = 0; k < keys; ++k) {
        const T *row = packed + stride * k;
        const auto p0 = S::load(row);
        typename S::reg p1;
        if constexpr (Two) p1 = S::load(row + S::lanes);
#pragma GCC unroll 4
        for (int r = 0; r < Rows; ++r) {
            const auto v = S::broadcast(values[r] + k);
            acc0[r] = S::fmadd(v, p0, acc0[r]);
            if constexpr (Two) acc1[r] = S::fmadd(v, p1, acc1[r]);
        }
    }
}

// Multiplies the full subrows of a sparse slice with the packed panel, four
// (then two, then one) at a time, and passes each result to
// write(subrow index, acc0, acc1). Returns whether any subrow was full.
template <typename T, bool Two, typename Write>
static inline __attribute__((always_inline)) bool packed_spmm_full_subrows(
    const slice &sparse, const ss_matrix<T> &ssm, const T *packed,
    const int stride, const int keys, Write &&write
) {
    using S = packed_simd<T>;
    using reg = typename S::reg;
    if (keys == 0) return false;
    const subrow *subrows = &ssm.sr_csc_mat.vals[sparse.start];
    int full[32];
    int count = 0;
    for (int si = 0; si < sparse.len; si++)
        if (subrows[si].len == keys) full[count++] = si;
    int i = 0;
    auto group = [&](auto rows) __attribute__((always_inline)) {
        constexpr int Rows = decltype(rows)::value;
        const T *values[Rows];
        reg acc0[Rows], acc1[Rows];
#pragma GCC unroll 4
        for (int r = 0; r < Rows; ++r) {
            values[r] = ssm.csr_mat.vals + subrows[full[i + r]].start;
            acc0[r] = S::zero();
            acc1[r] = S::zero();
        }
        packed_spmm_full_rows<T, Two, Rows>(
            values, keys, packed, stride, acc0, acc1
        );
#pragma GCC unroll 4
        for (int r = 0; r < Rows; ++r) write(full[i + r], acc0[r], acc1[r]);
        i += Rows;
    };
    for (; i + 4 <= count;) group(std::integral_constant<int, 4>{});
    if (i + 2 <= count) group(std::integral_constant<int, 2>{});
    if (i < count) group(std::integral_constant<int, 1>{});
    return count > 0;
}

template <>
void dense_left_inner_kernel<double>(
    const slice &R_s, ss_matrix<double> &R_ssm, const slice &S_s,
    const int *S_row_idxs, const int nr, const double *packed_mem, const int M,
    const int N, double *ret, const int k_offset, const bool full_rows
) {
    __mmask8 k0 = 0b11111111;
    __mmask8 k1 = 0b11111111;
    __m256i scatter_idx0, scatter_idx1;
    if (S_s.continuous) {
        const int J = S_row_idxs[0];
        const int write_length = N - J;
        int shift = write_length >= 8 ? 0 : 8 - write_length;
        k0 >>= shift;
        shift = write_length >= 16 ? 0 : 16 - write_length;
        k1 >>= shift;
    } else {
        const int last_write_S =
            N - S_row_idxs[S_s.len - 1] < 1 ? N - S_row_idxs[S_s.len - 1] : 1;
        const int write_length = (S_s.len - 1) + last_write_S;

        int scatter_locations[16] = {};
        for (int i = 0; i < S_s.len; i++) {
            scatter_locations[i] = S_row_idxs[i];
        }

        scatter_idx0 = _mm256_loadu_si256((__m256i *) scatter_locations);
        scatter_idx1 = _mm256_loadu_si256((__m256i *) (scatter_locations + 8));
        int shift = write_length >= 8 ? 0 : 8 - write_length;
        k0 >>= shift;
        shift = write_length >= 16 ? 0 : 16 - write_length;
        k1 >>= shift;
    }

    const int J = S_row_idxs[0];

    auto write = [&](const int si, __m512d res0,
                     __m512d res1) __attribute__((always_inline)) {
        const int I = R_ssm.sr_csc_mat.row_idxs[R_s.start + si];
        if (S_s.continuous) {
            // Masked loads: unmasked ones read past the row (and past the
            // end of ret for the last row) when fewer than 16 columns remain.
            res0 =
                _mm512_add_pd(res0, _mm512_maskz_loadu_pd(k0, &ret[I * N + J]));
            res1 = _mm512_add_pd(
                res1, _mm512_maskz_loadu_pd(k1, &ret[I * N + J + 8])
            );
            _mm512_mask_storeu_pd(&ret[I * N + J], k0, res0);
            _mm512_mask_storeu_pd(&ret[I * N + J + 8], k1, res1);
        } else {
            double *write_location = &ret[I * N];
            _mm512_mask_i32scatter_pd(
                write_location, k0, scatter_idx0,
                _mm512_add_pd(
                    res0, _mm512_mask_i32gather_pd(
                              res0, k0, scatter_idx0, write_location, 8
                          )
                ),
                8
            );
            _mm512_mask_i32scatter_pd(
                write_location, k1, scatter_idx1,
                _mm512_add_pd(
                    res1, _mm512_mask_i32gather_pd(
                              res1, k1, scatter_idx1, write_location, 8
                          )
                ),
                8
            );
        }
    };
    // Full subrows first, four at a time with shared packed loads.
    const int keys = full_subrow_keys(R_ssm, k_offset, full_rows);
    packed_spmm_full_subrows<double, true>(
        R_s, R_ssm, packed_mem, nr, keys, write
    );
    for (int si = 0; si < R_s.len; si++) {
        const subrow &sr = R_ssm.sr_csc_mat.vals[R_s.start + si];
        if (sr.len == 0 || sr.len == keys) {
            continue;
        }

        __m512d res0, res1;
        sparse_dense_inner_kernel_iter(
            sr, R_ssm.csr_mat, packed_mem, nr, k_offset, res0, res1
        );
        write(si, res0, res1);
    }
}

template <>
void dense_left_inner_kernel<float>(
    const slice &R_s, ss_matrix<float> &R_ssm, const slice &S_s,
    const int *S_row_idxs, const int nr, const float *packed_mem, const int M,
    const int N, float *ret, const int k_offset, const bool full_rows
) {
    __mmask16 k0 = 0b1111111111111111;
    __mmask16 k1 = 0b1111111111111111;
    __m512i scatter_idx0, scatter_idx1;
    if (S_s.continuous) {
        const int J = S_row_idxs[0];
        const int write_length = N - J;
        int shift = write_length >= 16 ? 0 : 16 - write_length;
        k0 >>= shift;
        shift = write_length >= 32 ? 0 : 32 - write_length;
        k1 >>= shift;
    } else {
        const int last_write_S =
            N - S_row_idxs[S_s.len - 1] < 1 ? N - S_row_idxs[S_s.len - 1] : 1;
        const int write_length = (S_s.len - 1) + last_write_S;

        int scatter_locations[32] = {};
        for (int i = 0; i < S_s.len; i++) {
            scatter_locations[i] = S_row_idxs[i];
        }

        scatter_idx0 = _mm512_loadu_si512((__m512i *) scatter_locations);
        scatter_idx1 = _mm512_loadu_si512((__m512i *) (scatter_locations + 16));
        int shift = write_length >= 16 ? 0 : 16 - write_length;
        k0 >>= shift;
        shift = write_length >= 32 ? 0 : 32 - write_length;
        k1 >>= shift;
    }

    const int J = S_row_idxs[0];

    auto write = [&](const int si, __m512 res0,
                     __m512 res1) __attribute__((always_inline)) {
        const int I = R_ssm.sr_csc_mat.row_idxs[R_s.start + si];
        if (S_s.continuous) {
            // Masked loads, as in the double kernel above.
            res0 =
                _mm512_add_ps(res0, _mm512_maskz_loadu_ps(k0, &ret[I * N + J]));
            res1 = _mm512_add_ps(
                res1, _mm512_maskz_loadu_ps(k1, &ret[I * N + J + 16])
            );
            _mm512_mask_storeu_ps(&ret[I * N + J], k0, res0);
            _mm512_mask_storeu_ps(&ret[I * N + J + 16], k1, res1);
        } else {
            float *write_location = &ret[I * N];
            _mm512_mask_i32scatter_ps(
                write_location, k0, scatter_idx0,
                _mm512_add_ps(
                    res0, _mm512_mask_i32gather_ps(
                              res0, k0, scatter_idx0, write_location, 4
                          )
                ),
                4
            );
            _mm512_mask_i32scatter_ps(
                write_location, k1, scatter_idx1,
                _mm512_add_ps(
                    res1, _mm512_mask_i32gather_ps(
                              res1, k1, scatter_idx1, write_location, 4
                          )
                ),
                4
            );
        }
    };
    // Full subrows first, four at a time with shared packed loads.
    const int keys = full_subrow_keys(R_ssm, k_offset, full_rows);
    packed_spmm_full_subrows<float, true>(
        R_s, R_ssm, packed_mem, nr, keys, write
    );
    for (int si = 0; si < R_s.len; si++) {
        const subrow &sr = R_ssm.sr_csc_mat.vals[R_s.start + si];
        if (sr.len == 0 || sr.len == keys) {
            continue;
        }

        __m512 res0, res1;
        sparse_dense_inner_kernel_iter(
            sr, R_ssm.csr_mat, packed_mem, nr, k_offset, res0, res1
        );
        write(si, res0, res1);
    }
}

template <>
void dense_right_inner_kernel<double>(
    const slice &S_s, ss_matrix<double> &S_ssm, const slice &R_s,
    const int *R_row_idxs, const int mr, const double *packed_mem, const int M,
    const int N, double *ret, __m512i gather_idx_512, const int k_offset,
    const bool full_rows
) {
    __mmask8 k0 = 0b11111111;
    __mmask8 k1 = 0b1111;
    __m256i scatter_idx0, scatter_idx1;
    if (R_s.continuous) {
        const int I = R_row_idxs[0];
        int shift = I + 8 <= M ? 0 : 8 - (M - I);
        k0 >>= shift;
        shift = I + 12 <= M ? 0 : 12 - (M - I);
        k1 >>= shift;
    } else {
        const int last_write_R =
            M - R_row_idxs[R_s.len - 1] < 1 ? M - R_row_idxs[R_s.len - 1] : 1;
        const int write_height = (R_s.len - 1) + last_write_R;

        int scatter_locations[16] = {};
        for (int i = 0; i < R_s.len; i++) {
            scatter_locations[i] = R_row_idxs[i] * N;
        }

        scatter_idx0 = _mm256_loadu_si256((__m256i *) scatter_locations);
        scatter_idx1 = _mm256_loadu_si256((__m256i *) (scatter_locations + 8));
        int shift = write_height >= 8 ? 0 : 8 - write_height;
        k0 >>= shift;
        shift = write_height >= 12 ? 0 : 12 - write_height;
        k1 >>= shift;
    }

    const int I = R_row_idxs[0];

    __m256i gather_idx = _mm512_castsi512_si256(gather_idx_512);

    auto write = [&](const int si, __m512d res0,
                     __m512d res1) __attribute__((always_inline)) {
        const int J = S_ssm.sr_csc_mat.row_idxs[S_s.start + si];
        if (R_s.continuous) {
            res0 = _mm512_add_pd(
                res0, _mm512_mask_i32gather_pd(
                          res0, k0, gather_idx, &ret[I * N + J], 8
                      )
            );
            res1 = _mm512_add_pd(
                res1, _mm512_mask_i32gather_pd(
                          res1, k1, gather_idx, &ret[(I + 8) * N + J], 8
                      )
            );
            _mm512_mask_i32scatter_pd(&ret[I * N + J], k0, gather_idx, res0, 8);
            _mm512_mask_i32scatter_pd(
                &ret[(I + 8) * N + J], k1, gather_idx, res1, 8
            );
        } else {
            double *write_location = &ret[J];
            _mm512_mask_i32scatter_pd(
                write_location, k0, scatter_idx0,
                _mm512_add_pd(
                    res0, _mm512_mask_i32gather_pd(
                              res0, k0, scatter_idx0, write_location, 8
                          )
                ),
                8
            );
            _mm512_mask_i32scatter_pd(
                write_location, k1, scatter_idx1,
                _mm512_add_pd(
                    res1, _mm512_mask_i32gather_pd(
                              res1, k1, scatter_idx1, write_location, 8
                          )
                ),
                8
            );
        }
    };
    // Full subrows first, four at a time with shared packed loads.
    const int keys = full_subrow_keys(S_ssm, k_offset, full_rows);
    packed_spmm_full_subrows<double, true>(
        S_s, S_ssm, packed_mem, mr, keys, write
    );
    for (int si = 0; si < S_s.len; si++) {
        const subrow &sr = S_ssm.sr_csc_mat.vals[S_s.start + si];
        if (sr.len == 0 || sr.len == keys) {
            continue;
        }

        __m512d res0, res1;
        sparse_dense_inner_kernel_iter(
            sr, S_ssm.csr_mat, packed_mem, mr, k_offset, res0, res1
        );
        write(si, res0, res1);
    }
}

template <>
void dense_right_inner_kernel<float>(
    const slice &S_s, ss_matrix<float> &S_ssm, const slice &R_s,
    const int *R_row_idxs, const int mr, const float *packed_mem, const int M,
    const int N, float *ret, __m512i gather_idx, const int k_offset,
    const bool full_rows
) {
    __mmask16 k0 = 0b111111111111;
    __m512i scatter_idx0;
    if (R_s.continuous) {
        const int I = R_row_idxs[0];
        int shift = I + 12 <= M ? 0 : 12 - (M - I);
        k0 >>= shift;
    } else {
        const int last_write_R =
            M - R_row_idxs[R_s.len - 1] < 1 ? M - R_row_idxs[R_s.len - 1] : 1;
        const int write_height = (R_s.len - 1) + last_write_R;

        // A 512-bit index load reads 16 ints; [12] was a stack over-read.
        int scatter_locations[16] = {};
        for (int i = 0; i < R_s.len; i++) {
            scatter_locations[i] = R_row_idxs[i] * N;
        }

        scatter_idx0 = _mm512_loadu_si512((__m512i *) scatter_locations);
        int shift = write_height >= 12 ? 0 : 12 - write_height;
        k0 >>= shift;
    }

    const int I = R_row_idxs[0];

    auto write = [&](const int si, __m512 res0,
                     __m512 res1) __attribute__((always_inline)) {
        const int J = S_ssm.sr_csc_mat.row_idxs[S_s.start + si];
        if (R_s.continuous) {
            res0 = _mm512_add_ps(
                res0, _mm512_mask_i32gather_ps(
                          res0, k0, gather_idx, &ret[I * N + J], 4
                      )
            );
            _mm512_mask_i32scatter_ps(&ret[I * N + J], k0, gather_idx, res0, 4);
        } else {
            float *write_location = &ret[J];
            _mm512_mask_i32scatter_ps(
                write_location, k0, scatter_idx0,
                _mm512_add_ps(
                    res0, _mm512_mask_i32gather_ps(
                              res0, k0, scatter_idx0, write_location, 4
                          )
                ),
                4
            );
        }
    };
    // Full subrows first, four at a time with shared packed loads.
    const int keys = full_subrow_keys(S_ssm, k_offset, full_rows);
    packed_spmm_full_subrows<float, false>(
        S_s, S_ssm, packed_mem, mr, keys, write
    );
    for (int si = 0; si < S_s.len; si++) {
        const subrow &sr = S_ssm.sr_csc_mat.vals[S_s.start + si];
        if (sr.len == 0 || sr.len == keys) {
            continue;
        }

        __m512 res0, res1;
        sparse_dense_inner_kernel_iter(
            sr, S_ssm.csr_mat, packed_mem, mr, k_offset, res0, res1
        );
        write(si, res0, res1);
    }
}

template void pack_slice_new<float>(
    const slice &s, ss_matrix<float> &ssm, const int join_idx, const int sl,
    const int ik, float *packed_mem_original, BIT_DTYPE *bm_original
);

template void pack_slice_new<double>(
    const slice &s, ss_matrix<double> &ssm, const int join_idx, const int sl,
    const int ik, double *packed_mem_original, BIT_DTYPE *bm_original
);

template void dense_dense_matmul<float>(
    ss_matrix<float> &R_ssm, ss_matrix<float> &S_ssm, const int tile_size_l2,
    const int tile_size_l3, float *ret, bool sparse_result, const int threads,
    const cpu_execution_options &options
);

template void dense_dense_matmul<double>(
    ss_matrix<double> &R_ssm, ss_matrix<double> &S_ssm, const int tile_size_l2,
    const int tile_size_l3, double *ret, bool sparse_result, const int threads,
    const cpu_execution_options &options
);
