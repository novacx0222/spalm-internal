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
};
