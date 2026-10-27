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
    // Lengthen the slices (up to 2048 keys) while the sparse side's subrows
    // would hold fewer than 16 entries on average; off when slice_length is
    // given explicitly (key: auto_slice_length).
    bool auto_slice_length = true;
};
