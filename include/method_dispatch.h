#pragma once

// Backend selection shared by the forced-method and automatic paths.
namespace spalm_dispatch {

enum class Backend { Internal, MklDense, MklSparse };

// Explicit methods take precedence over every automatic cost heuristic; the
// automatic MKL fallbacks apply only when fallback is enabled.
inline Backend select(
    bool force_dense, bool force_sparse, bool fallback, bool auto_dense,
    bool auto_sparse
) {
    if (force_dense) return Backend::MklDense;
    if (force_sparse) return Backend::MklSparse;
    if (fallback && auto_dense) return Backend::MklDense;
    if (fallback && auto_sparse) return Backend::MklSparse;
    return Backend::Internal;
}

} // namespace spalm_dispatch
