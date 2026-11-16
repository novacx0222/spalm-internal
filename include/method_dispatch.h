#pragma once

// Backend selection shared by the forced-method and automatic paths.
namespace spalm_dispatch {

// OpenblasDense: OpenBLAS cblas_?gemm on the densified inputs, the same
// path as MklDense except for the GEMM call (forced only, never a
// fallback).
enum class Backend { Internal, MklDense, MklSparse, OpenblasDense };

// Explicit methods take precedence over every automatic cost heuristic; the
// automatic MKL fallbacks apply only when fallback is enabled. At most one
// force flag is set (init_decision_thresholds()).
inline Backend select(
    bool force_dense, bool force_sparse, bool fallback, bool auto_dense,
    bool auto_sparse, bool force_openblas_dense = false
) {
    if (force_openblas_dense) return Backend::OpenblasDense;
    if (force_dense) return Backend::MklDense;
    if (force_sparse) return Backend::MklSparse;
    if (fallback && auto_dense) return Backend::MklDense;
    if (fallback && auto_sparse) return Backend::MklSparse;
    return Backend::Internal;
}

// The dense baselines: both inputs expanded to dense M x K and N x K
// arrays, then one GEMM.
inline bool is_dense_baseline(Backend backend) {
    return backend == Backend::MklDense || backend == Backend::OpenblasDense;
}

inline const char *name(Backend backend) {
    switch (backend) {
    case Backend::MklDense: return "mkl-dense";
    case Backend::MklSparse: return "mkl-sparse";
    case Backend::OpenblasDense: return "openblas-dense";
    default: return "spalm-internal";
    }
}

} // namespace spalm_dispatch
