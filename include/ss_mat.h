#ifndef SS_MAT_CLASS_H
#define SS_MAT_CLASS_H

#include <chrono>
#include <memory>
#include <omp.h>
#include <string>

#include "slice_decision.h"
#include "util.h"

template <typename T> struct csr {
    int *row_ptrs;
    int *col_idxs;
    T *vals;
    int rows;
    int cols;
};

template <typename T> struct csc {
    int *col_ptrs;
    int *row_idxs;
    T *vals;
    int rows;
    int cols;
};

struct subrow {
    int start;
    uint16_t len;
};

struct slice {
    int start;
    uint16_t len;
    uint8_t b;
    bool continuous;
};

// ss_matrix stores unique_ptrs for memory ownership.
// The raw pointers in csr/csc structs point to data owned by these unique_ptrs.
template <typename T> class ss_matrix {
public:
    // Original CSR matrix (non-owning pointers to external data)
    csr<T> csr_mat;

    // Subrow CSR representation
    csr<subrow> sr_csr_mat;

    // Subrow CSC representation (transposed from sr_csr)
    csc<subrow> sr_csc_mat;

    // Slice CSC representation
    csc<slice> s_csc_mat;

    // Slice dimensions
    int sh; // slice height
    int sl; // slice length

    // Mapping from sr_csc index to sr_csr index
    // sr_csc_to_csr_idx[csc_idx] = csr_idx for each subrow
    int *sr_csc_to_csr_idx = nullptr;

    // Memory ownership: all dynamically allocated arrays are stored here
    std::vector<std::unique_ptr<int[]>> owned_int_arrays;
    std::vector<std::unique_ptr<subrow[]>> owned_subrow_arrays;
    std::vector<std::unique_ptr<slice[]>> owned_slice_arrays;

    // Helper to allocate and track an int array
    int *alloc_int_array(size_t count) {
        owned_int_arrays.push_back(
            std::make_unique_for_overwrite<int[]>(count)
        );
        return owned_int_arrays.back().get();
    }

    // Helper to allocate and track a subrow array
    subrow *alloc_subrow_array(size_t count) {
        owned_subrow_arrays.push_back(
            std::make_unique_for_overwrite<subrow[]>(count)
        );
        return owned_subrow_arrays.back().get();
    }

    // Helper to allocate and track a slice array
    slice *alloc_slice_array(size_t count) {
        owned_slice_arrays.push_back(
            std::make_unique_for_overwrite<slice[]>(count)
        );
        return owned_slice_arrays.back().get();
    }
};

// Temporary tuple used during subrow CSR construction.
struct sr_tuple {
    int col_idx;
    int start;
    uint16_t len;
    uint16_t _pad; // pad to 12 bytes for aligned stores
};

// Split [0, M) rows into `threads` NNZ-balanced ranges via binary search on Ri.
// out[t] and out[t+1] are the row range for thread t.
static void compute_thread_row_ranges(
    const int *Ri, const int M, const int threads, std::vector<int> &out
) {
    out.resize(threads + 1);
    out[0] = 0;
    const int total_nnz = Ri[M];
    for (int t = 1; t < threads; t++) {
        int target = (long long) t * total_nnz / threads;
        int lo = out[t - 1], hi = M;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (Ri[mid] < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        out[t] = lo;
    }
    out[threads] = M;
}

// Create subrow CSR from the original CSR matrix.
// Each subrow represents a contiguous run of elements within a single slice
// column range.
//
// Optimization: single NNZ scan using per-thread buffers, eliminating the
// second pass over Rj that the naive two-phase count-then-fill approach requires.
template <typename T>
void create_subrow_csr(ss_matrix<T> &ssm, const int threads) {
    const int M = ssm.csr_mat.rows;
    const int *Ri = ssm.csr_mat.row_ptrs;
    const int *Rj = ssm.csr_mat.col_idxs;
    const int sl_shift = (int) std::log2(ssm.sl);

    // Allocate row_ptrs for subrow CSR (one per row + 1)
    int *sr_row_ptrs = ssm.alloc_int_array(M + 1);
    sr_row_ptrs[0] = 0;

    // NNZ-balanced thread row ranges (better load balance than equal-row split)
    std::vector<int> trs;
    compute_thread_row_ranges(Ri, M, threads, trs);

    // Per-thread tuple buffers; worst-case size = thread's NNZ count
    // (every element is its own subrow). Released at function exit.
    std::vector<std::unique_ptr<sr_tuple[]>> tl_bufs(threads);
    std::vector<int> tl_counts(threads, 0);
    for (int t = 0; t < threads; t++) {
        int sz = Ri[trs[t + 1]] - Ri[trs[t]];
        if (sz > 0) tl_bufs[t] = std::make_unique_for_overwrite<sr_tuple[]>(sz);
    }

    // Phase 1+3 combined: single pass over Rj per thread.
    // Each thread counts subrows (writes sr_row_ptrs[i+1]) AND stores
    // (col_idx, start, len) tuples in its private buffer.
    // Loop over the logical ranges rather than omp_get_thread_num(): OpenMP
    // may give a smaller team than requested, and every range must run.
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int tid = 0; tid < threads; tid++) {
        sr_tuple *buf = tl_bufs[tid].get();
        int buf_pos = 0;

        for (int i = trs[tid]; i < trs[tid + 1]; i++) {
            const int nnz_start = Ri[i], nnz_end = Ri[i + 1];
            if (nnz_start == nnz_end) {
                sr_row_ptrs[i + 1] = 0;
                continue;
            }

            int curr_prefix = Rj[nnz_start] >> sl_shift;
            int subrow_start = nnz_start;
            int cnt = 0;

            for (int k = nnz_start + 1; k < nnz_end; k++) {
                int next_prefix = Rj[k] >> sl_shift;
                if (next_prefix != curr_prefix) {
                    buf[buf_pos++] = {
                        curr_prefix, subrow_start,
                        static_cast<uint16_t>(k - subrow_start), 0
                    };
                    cnt++;
                    subrow_start = k;
                    curr_prefix = next_prefix;
                }
            }
            // Last subrow in this row
            buf[buf_pos++] = {
                curr_prefix, subrow_start,
                static_cast<uint16_t>(nnz_end - subrow_start), 0
            };
            cnt++;
            sr_row_ptrs[i + 1] = cnt;
        }
        tl_counts[tid] = buf_pos;
    }

    // Phase 2: sequential prefix sum (O(M), unchanged)
    for (int i = 0; i < M; i++) sr_row_ptrs[i + 1] += sr_row_ptrs[i];
    const int total_subrows = sr_row_ptrs[M];

    // Allocate final output arrays now that total size is known
    int *sr_col_idxs = ssm.alloc_int_array(total_subrows);
    subrow *sr_vals = ssm.alloc_subrow_array(total_subrows);

    // Phase 4: parallel scatter-copy from thread buffers to output.
    // O(total_subrows), not O(NNZ) — fast when average subrow length > 1.
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int tid = 0; tid < threads; tid++) {
        const sr_tuple *buf = tl_bufs[tid].get();
        int buf_pos = 0;

        for (int i = trs[tid]; i < trs[tid + 1]; i++) {
            int out = sr_row_ptrs[i];
            const int out_end = sr_row_ptrs[i + 1];
            while (out < out_end) {
                sr_col_idxs[out] = buf[buf_pos].col_idx;
                sr_vals[out].start = buf[buf_pos].start;
                sr_vals[out].len = buf[buf_pos].len;
                buf_pos++;
                out++;
            }
        }
    }

    // Set up the subrow CSR struct
    ssm.sr_csr_mat.row_ptrs = sr_row_ptrs;
    ssm.sr_csr_mat.col_idxs = sr_col_idxs;
    ssm.sr_csr_mat.vals = sr_vals;
    ssm.sr_csr_mat.rows = M;
    ssm.sr_csr_mat.cols = div_ceil(ssm.csr_mat.cols, ssm.sl);
}

// Convert subrow CSR to subrow CSC (transpose)
template <typename T>
void create_subrow_csc(ss_matrix<T> &ssm, const int threads) {
    const int num_cols = ssm.sr_csr_mat.cols;
    const int num_rows = ssm.sr_csr_mat.rows;
    const int nnz = ssm.sr_csr_mat.row_ptrs[num_rows];

    // Allocate col_ptrs for CSC
    int *csc_col_ptrs = ssm.alloc_int_array(num_cols + 1);
    memset(csc_col_ptrs, 0, sizeof(int) * (num_cols + 1));

    // Count elements per column
    for (int i = 0; i < nnz; i++) {
        csc_col_ptrs[ssm.sr_csr_mat.col_idxs[i] + 1]++;
    }

    // Prefix sum
    for (int i = 0; i < num_cols; i++) {
        csc_col_ptrs[i + 1] += csc_col_ptrs[i];
    }

    // Allocate row_idxs and vals for CSC
    int *csc_row_idxs = ssm.alloc_int_array(nnz);
    subrow *csc_vals = ssm.alloc_subrow_array(nnz);

    // Allocate CSC-to-CSR index mapping
    int *csc_to_csr = ssm.alloc_int_array(nnz);

    // Temporary write positions
    auto write_pos = std::make_unique_for_overwrite<int[]>(num_cols);
    for (int i = 0; i < num_cols; i++) {
        write_pos[i] = csc_col_ptrs[i];
    }

    // Fill CSC data
    for (int row = 0; row < num_rows; row++) {
        for (int idx = ssm.sr_csr_mat.row_ptrs[row];
             idx < ssm.sr_csr_mat.row_ptrs[row + 1]; idx++) {
            int col = ssm.sr_csr_mat.col_idxs[idx];
            int pos = write_pos[col]++;
            csc_row_idxs[pos] = row;
            csc_vals[pos] = ssm.sr_csr_mat.vals[idx];
            csc_to_csr[pos] = idx;
        }
    }

    // Set up the subrow CSC struct and mapping
    ssm.sr_csc_mat.col_ptrs = csc_col_ptrs;
    ssm.sr_csc_mat.row_idxs = csc_row_idxs;
    ssm.sr_csc_mat.vals = csc_vals;
    ssm.sr_csc_mat.rows = num_rows;
    ssm.sr_csc_mat.cols = num_cols;
    ssm.sr_csc_to_csr_idx = csc_to_csr;
}

// Create slice CSC from subrow CSC.
// Groups subrows by slice height (sh) and assigns buckets.
template <typename T>
void create_slice_csc(ss_matrix<T> &ssm, const int threads) {
    const int sh = ssm.sh;
    const int sl = ssm.sl;
    const int num_cols = ssm.sr_csc_mat.cols;
    const int num_rows = ssm.sr_csc_mat.rows;
    const int slice_rows = div_ceil(num_rows, sh);
    const int nnz = ssm.sr_csc_mat.col_ptrs[num_cols];

    // Count slices per column: each slice groups sh subrows together
    int *s_col_ptrs = ssm.alloc_int_array(num_cols + 1);
    s_col_ptrs[0] = 0;

#pragma omp parallel for num_threads(threads)
    for (int col = 0; col < num_cols; col++) {
        int num_subrows =
            ssm.sr_csc_mat.col_ptrs[col + 1] - ssm.sr_csc_mat.col_ptrs[col];
        s_col_ptrs[col + 1] = div_ceil(num_subrows, sh);
    }

    // Prefix sum (sequential)
    for (int col = 0; col < num_cols; col++) {
        s_col_ptrs[col + 1] += s_col_ptrs[col];
    }

    const int total_slices = s_col_ptrs[num_cols];

    // Allocate row_idxs and vals for slice CSC
    int *s_row_idxs = ssm.alloc_int_array(total_slices);
    slice *s_vals = ssm.alloc_slice_array(total_slices);

    // Fill slice data: group every sh subrows into one slice
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (int col = 0; col < num_cols; col++) {
        const int col_start = ssm.sr_csc_mat.col_ptrs[col];
        const int col_end = ssm.sr_csc_mat.col_ptrs[col + 1];
        int write_idx = s_col_ptrs[col];

        for (int idx = col_start; idx < col_end; idx += sh) {
            const int slice_end = std::min(idx + sh, col_end);
            const int slice_len = slice_end - idx;

            // Sum nnz across subrows in this slice
            int slice_nnz = 0;
            for (int s = idx; s < slice_end; s++) {
                slice_nnz += ssm.sr_csc_mat.vals[s].len;
            }

            // Check continuity: each subrow's row should be prev + 1
            bool is_continuous = true;
            for (int s = idx + 1; s < slice_end; s++) {
                if (ssm.sr_csc_mat.row_idxs[s] !=
                    ssm.sr_csc_mat.row_idxs[s - 1] + 1) {
                    is_continuous = false;
                    break;
                }
            }

            // row_idx of the slice is the row of the first subrow
            s_row_idxs[write_idx] = ssm.sr_csc_mat.row_idxs[idx];
            s_vals[write_idx].start = idx;
            s_vals[write_idx].len = static_cast<uint16_t>(slice_len);
            s_vals[write_idx].b = get_bucket(slice_nnz, sh * sl);
            s_vals[write_idx].continuous = is_continuous;
            write_idx++;
        }
    }

    // Set up the slice CSC struct
    ssm.s_csc_mat.col_ptrs = s_col_ptrs;
    ssm.s_csc_mat.row_idxs = s_row_idxs;
    ssm.s_csc_mat.vals = s_vals;
    ssm.s_csc_mat.rows = slice_rows;
    ssm.s_csc_mat.cols = num_cols;
}

// Main conversion function: CSR -> ss_matrix with all representations
template <typename T>
void csr_to_ss_matrix(
    int *row_ptrs, int *col_idxs, T *vals, const int rows, const int cols,
    const int sl, const int sh, ss_matrix<T> &ssm, const int threads,
    int timing_side = 0
) {
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    auto create_start = std::chrono::high_resolution_clock::now();
    auto stage_start = create_start;
    // "[<side>-slice-<stage>]" timing labels; side 1 is R, 2 is S.
    const std::string side = "[" + std::to_string(timing_side) + "-slice-";
    auto report = [&](const char *stage) {
        end = std::chrono::high_resolution_clock::now();
        print_time(side + stage + "-cumulative]", end - create_start);
        print_time(side + stage + "]", end - stage_start);
        stage_start = end;
    };

    // Set dimensions
    ssm.sl = sl;
    ssm.sh = sh;

    // Set up original CSR (non-owning pointers)
    ssm.csr_mat.row_ptrs = row_ptrs;
    ssm.csr_mat.col_idxs = col_idxs;
    ssm.csr_mat.vals = vals;
    ssm.csr_mat.rows = rows;
    ssm.csr_mat.cols = cols;

    // Step 1: Create subrow CSR
    create_subrow_csr(ssm, threads);
    report("subrow-csr");

    // Step 2: Create subrow CSC (transpose)
    create_subrow_csc(ssm, threads);
    report("subrow-csc");

    // Step 3: Create slice CSC
    create_slice_csc(ssm, threads);
    report("csc");
}

template <typename T>
void mm_check(
    const int slice_col, ss_matrix<T> &R_ssm, ss_matrix<T> &S_ssm,
    unsigned long long &sparse_sparse_mm_cost,
    unsigned long long &sparse_dense_mm_cost,
    unsigned long long &dense_sparse_mm_cost,
    unsigned long long &dense_dense_mm_cost
) {
    if (R_ssm.s_csc_mat.col_ptrs[slice_col] ==
        R_ssm.s_csc_mat.col_ptrs[slice_col + 1]) {
        return;
    }
    if (S_ssm.s_csc_mat.col_ptrs[slice_col] ==
        S_ssm.s_csc_mat.col_ptrs[slice_col + 1]) {
        return;
    }

    const int sl = R_ssm.sl;
    const int mr = R_ssm.sh;
    const int nr = S_ssm.sh;
    const int R_slice_ele_num = sl * mr;
    const int S_slice_ele_num = sl * nr;

    std::vector<int> S_slice_buckets(num_buckets + 1);
    for (int si = S_ssm.s_csc_mat.col_ptrs[slice_col];
         si < S_ssm.s_csc_mat.col_ptrs[slice_col + 1]; si++) {
        S_slice_buckets[S_ssm.s_csc_mat.vals[si].b]++;
    }
    std::vector<int> R_slice_buckets(num_buckets + 1);
    for (int si = R_ssm.s_csc_mat.col_ptrs[slice_col];
         si < R_ssm.s_csc_mat.col_ptrs[slice_col + 1]; si++) {
        R_slice_buckets[R_ssm.s_csc_mat.vals[si].b]++;
    }

    sparse_sparse_mm_cost = 0;
    sparse_dense_mm_cost = 0;
    dense_sparse_mm_cost = 0;
    dense_dense_mm_cost = 0;
    for (size_t i = 0; i < R_slice_buckets.size(); i++) {
        if (R_slice_buckets[i] == 0) {
            continue;
        }
        // Compute sparse-sparse MM cost, assume uniform density
        // Approximate # of elements in S
        // 64-bit sums/products: slice counts times per-slice NNZ overflow int
        // for large matrices.
        unsigned long long S_slice_sum = 0;
        int S_slice_num = 0;
        for (int b = 0; b < ss_upper[i]; b++) {
            S_slice_sum += (unsigned long long) S_slice_buckets[b] *
                           bucket_to_nnz(b, S_slice_ele_num);
            S_slice_num += S_slice_buckets[b];
        }
        const float average_nums_per_col = (float) S_slice_sum / sl;
        unsigned long long sparse_sparse_mm_cost_one =
            (unsigned long long) R_slice_buckets[i] *
            bucket_to_nnz((int) i, R_slice_ele_num) * average_nums_per_col;
        if (R_slice_buckets[i] > 0 && S_slice_num > 0 &&
            sparse_sparse_mm_cost_one < 1) {
            sparse_sparse_mm_cost_one = 1;
        }
        sparse_sparse_mm_cost += sparse_sparse_mm_cost_one;
        const int sparse_end = ss_upper[i];
        const int dense_right_end = ri_upper[i];
        // Compute cost of sparse-dense or dense-dense
        S_slice_num = 0;
        for (int b = std::max(sparse_end, dense_right_end);
             b < (int) S_slice_buckets.size(); b++) {
            S_slice_num += S_slice_buckets[b];
        }
        unsigned long long dense_mm_cost_one = 0;
        if ((int) i < left_dense_threshold) {
            // sparse-dense
            dense_mm_cost_one = (unsigned long long) R_slice_buckets[i] *
                                bucket_to_nnz((int) i, R_slice_ele_num) *
                                S_slice_num * nr;
            if (R_slice_buckets[i] > 0 && S_slice_num > 0 &&
                dense_mm_cost_one < 1) {
                dense_mm_cost_one = 1;
            }
            sparse_dense_mm_cost += dense_mm_cost_one;
        } else {
            // dense-dense
            dense_mm_cost_one = (unsigned long long) R_slice_buckets[i] *
                                R_slice_ele_num * S_slice_num * nr;
            if (R_slice_buckets[i] > 0 && S_slice_num > 0 &&
                dense_mm_cost_one < 1) {
                dense_mm_cost_one = 1;
            }
            dense_dense_mm_cost += dense_mm_cost_one;
        }
        // Compute cost of dense-sparse
        S_slice_sum = 0;
        for (int b = sparse_end; b < dense_right_end; b++) {
            S_slice_sum += (unsigned long long) S_slice_buckets[b] *
                           bucket_to_nnz(b, S_slice_ele_num);
        }
        dense_sparse_mm_cost += S_slice_sum * mr * R_slice_buckets[i];
    }
}

// Pack a slice into dense memory for the inner kernel.
// Each subrow in the slice corresponds to one row in the packed block.
// Keys are made panel-local (k - join_idx) instead of offsetting the buffer
// pointers backwards: that formed pointers outside the allocations, and for
// slice lengths below 64 it set bits outside this slice's bitmap words.
template <typename T>
void pack_slice_new(
    const slice &s, ss_matrix<T> &ssm, const int join_idx, const int sl,
    const int ik, T *packed_mem_original, BIT_DTYPE *bm_original
) {
    BIT_DTYPE *bm = bm_original;
    for (int si = 0; si < s.len; si++) {
        const subrow &sr = ssm.sr_csc_mat.vals[s.start + si];
        T *packed_mem = packed_mem_original + si;

        const int unroll_factor = 4;
        const int sr_end = sr.start + sr.len;
        const int k_idx_end =
            sr.start + (sr.len / unroll_factor) * unroll_factor;
        int k_idx = sr.start;

        for (; k_idx < k_idx_end; k_idx += unroll_factor) {
            int k;
            T v;
            k = ssm.csr_mat.col_idxs[k_idx] - join_idx;
            v = ssm.csr_mat.vals[k_idx];
            set_bit(bm, k);
            packed_mem[ik * k] = v;

            k = ssm.csr_mat.col_idxs[k_idx + 1] - join_idx;
            v = ssm.csr_mat.vals[k_idx + 1];
            set_bit(bm, k);
            packed_mem[ik * k] = v;

            k = ssm.csr_mat.col_idxs[k_idx + 2] - join_idx;
            v = ssm.csr_mat.vals[k_idx + 2];
            set_bit(bm, k);
            packed_mem[ik * k] = v;

            k = ssm.csr_mat.col_idxs[k_idx + 3] - join_idx;
            v = ssm.csr_mat.vals[k_idx + 3];
            set_bit(bm, k);
            packed_mem[ik * k] = v;
        }

        for (; k_idx < sr_end; k_idx++) {
            int k = ssm.csr_mat.col_idxs[k_idx] - join_idx;
            T v = ssm.csr_mat.vals[k_idx];
            set_bit(bm, k);
            packed_mem[ik * k] = v;
        }
    }
}

#endif // SS_MAT_CLASS_H
