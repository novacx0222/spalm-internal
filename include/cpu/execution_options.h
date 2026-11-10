#pragma once

// Per-call execution switches, read from the spalm() key/value options so
// each optimization can be measured independently. Defaults are enabled.
struct cpu_execution_options {
    // Drop explicit finite zeros before slicing (key: prune_zeros).
    bool prune_zeros = true;
    // Keep only join keys used by both inputs and renumber them densely
    // (key: compact_keys).
    bool compact_keys = true;
    // Pack only the sides the dense-path algorithms read, and the DD
    // bitmaps only when DD runs (key: selective_packing).
    bool selective_packing = true;
    // SS: blocks of 8 consecutive output columns use plain vector loads and
    // stores instead of gather/scatter (key: ss_contiguous).
    bool ss_contiguous = true;
    // SD/DS: full sparse subrows (one value per key of the slice column) are
    // multiplied four at a time, sharing every packed-row load
    // (key: sd_ds_full_rows).
    bool sd_ds_full_rows = true;
    // SS without dense slices: output-column blocks, so one block of the
    // transposed S stays in the shared cache while all R rows use it.
    // -1: from the data size and the shared (L3) cache size, read at run
    // time; 0 or 1: off; n > 1: n blocks (key: ss_column_blocks).
    int ss_column_blocks = -1;
    // DS without DD: pack R with 16 values per key instead of the 12 R rows
    // of a slice, so the packed rows DS loads are cache-line aligned (the
    // DD kernel needs 12) (key: ds_aligned_panels).
    bool ds_aligned_panels = true;
    // Slice length (key: auto_slice_length; 0 when slice_length is given
    // explicitly). 0: the configured length. 1: up to 4 times it, from
    // exact counts of subrows and packed slices (slice_length_rule.h).
    // 2: the earlier rule from global density alone, kept for comparison.
    int auto_slice_length = 1;
    // The caller zeroed the result (key: result_zeroed, default 1). With 0
    // it holds garbage and the library writes every element: the forced MKL
    // backends overwrite it (GEMM with beta = 0, spmmd), every other path,
    // the zero-work exits included, zeroes it first.
    bool result_zeroed = true;
    // SS with sparse output (spalm_sparse()): entries per output page,
    // 0 = pages growing from 64 Ki to 4 Mi entries. Small fixed pages are
    // for tests (key: ss_output_page_entries).
    long long ss_output_page_entries = 0;

    // Tunables below default to the constants they replace, so a call
    // without these keys behaves exactly as before. spalm_impl() validates
    // them and prints the effective values on one "[conf]" line per call.

    // Automatic dispatch (init_method "" and its variants; spalm.cpp). The
    // per-kernel costs of the decision table are multiplied by these weights
    // before the two rules below (cost units are not time: SS takes far
    // longer per unit than DD). Keys: cost_weight_ss / _sd / _ds / _dd.
    double cost_weight_ss = 1.0;
    double cost_weight_sd = 1.0;
    double cost_weight_ds = 1.0;
    double cost_weight_dd = 1.0;
    // Every slice pair goes to the dense kernels when the weighted SS cost is
    // below this fraction of the weighted dense cost (key: auto_sparse_ratio;
    // compared in float, as before).
    double auto_sparse_ratio = 0.1;
    // With fallback: MKL dense when the (unweighted) dense cost exceeds this
    // fraction of M * K * N (key: auto_mkl_dense_ratio).
    double auto_mkl_dense_ratio = 0.8;
    // With fallback: MKL sparse when only SS slice pairs remain
    // (key: auto_mkl_sparse_fallback, 0 or 1).
    bool auto_mkl_sparse_fallback = true;

    // Slice length rule, auto_slice_length = 1 (slice_length_rule.h).
    // Keys: asl_min_cost, asl_subrow_write_entries, asl_min_cost_ratio,
    // asl_max_levels (1 to 3: lengths up to 1x, 2x or 4x the configured).
    double asl_min_cost = 3e9;
    double asl_subrow_write_entries = 16.0;
    double asl_min_cost_ratio = 0.9;
    int asl_max_levels = 3;

    // Packing buffers of SD/DS/DD (packing_buffers.h): R's block is a
    // huge-page mapping from r_block_huge_bytes on, the S blocks of all
    // workers share one from s_blocks_huge_bytes on (never below 2 MiB).
    // -1: never (keys: r_block_huge_bytes, s_blocks_huge_bytes).
    long long r_block_huge_bytes = 8LL << 20;
    long long s_blocks_huge_bytes = 2LL << 20;

    // Automatic SS column blocks (ss_column_blocks = -1): at least this many
    // output columns per block, and each block's part of S^T sized to the
    // shared cache divided by this (keys: ss_column_block_min_cols,
    // ss_column_block_cache_divisor). The minimum also caps an explicit
    // ss_column_blocks count.
    int ss_column_block_min_cols = 256;
    int ss_column_block_cache_divisor = 3;
};
