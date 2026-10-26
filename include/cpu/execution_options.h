#pragma once

// Per-call execution switches, read from the spalm() key/value options so
// each optimization can be measured independently. Defaults are enabled.
struct cpu_execution_options {
    // Drop explicit finite zeros before slicing (key: prune_zeros).
    bool prune_zeros = true;
};
