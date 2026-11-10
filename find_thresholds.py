#!/usr/bin/env python3
"""
Find optimal algorithm decision thresholds for spalm matmul.

Finds ss_upper, ri_upper, and left_dense_threshold for CPU.

Key optimisations over a naive BxB sweep:
  - Dense-dense time is measured once (constant across all densities).
  - Right-inner-kernel time is measured until RI > DD for --early-stop consecutive
    buckets; remaining buckets are never measured (RI can't win there).
  - Left-inner-kernel time is measured until LI > DD for --early-stop consecutive
    buckets; remaining buckets are never measured (LI can't win there).
  - Sparse-sparse is measured per (left, right) with early stopping:
    if a non-SS algorithm wins for --early-stop consecutive right buckets,
    the first of those buckets becomes the threshold and the row is done.

Produces (stage "table"):
  - decision_thresholds.txt  (threshold file for spalm)
  - algorithm_map.csv        (detailed timings)
  - algorithm_map.png        (heatmap visualization)

Stages "rblock" and "asl" measure two defaults of calibration.txt
(parameters/<platform>/<type>/, read by the library on every call; a key
given to spalm() wins), on inputs generated here, never on SuiteSparse or
grid matrices:
  - r_block_huge_bytes: the R packing block size from which a huge-page
    mapping pays (DS and DD, mapped against not mapped);
  - asl_min_cost: the smallest modelled cost at which the slice-length rule
    may lengthen (DD, SD and DS at the configured and the chosen length).
Each writes its cells (rblock_cells.csv, asl_cells.csv) next to it.
The third key, ss_page_huge_entries, is measured end to end through DuckDB
(calibrate_run.py --page-sweep, an experiment script kept outside this
repository), with the same file helpers (spalm_calibration.py).
"""

import argparse
import csv
import ctypes
import math
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.sparse import random as sparse_random, csr_matrix
from sklearn.isotonic import IsotonicRegression
from wurlitzer import STDOUT, pipes

import spalm_calibration as calib
from spalm_timing import (DENSE_MM_LABEL, MM_TOTAL_LABEL,
                          SLICE_MM_TOTAL_LABEL, SPARSE_MM_LABEL,
                          parse_r_block, parse_slice_length, parse_time)

SCRIPT_DIR = Path(__file__).resolve().parent
SPALM_BIN = SCRIPT_DIR / "bin"


def load_so(lib_name):
    """Load a .so from this package's bin/ (e.g. spalm.so)."""
    lib = ctypes.CDLL(str(SPALM_BIN / lib_name))
    if hasattr(lib, "spalm"):
        # spalm() returns a status code: 0 on success.
        lib.spalm.restype = ctypes.c_int
    return lib


class BenchmarkError(RuntimeError):
    """A calibration run failed; no thresholds may be derived from it."""


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------

def generate_uniform_csr(rows, cols, density, dtype=np.float32, seed=None):
    """Generate a CSR matrix with uniform random non-zero pattern."""
    if density <= 0:
        return csr_matrix((rows, cols), dtype=dtype)
    density = min(density, 1.0)
    rng = np.random.default_rng(seed)
    mat = sparse_random(rows, cols, density=density, format="csr",
                        dtype=dtype, random_state=rng.integers(2 ** 31))
    if mat.nnz > 0:
        mat.data[:] = rng.standard_normal(mat.nnz).astype(dtype)
        mat.data[mat.data == 0] = dtype(1e-6)
        mat.eliminate_zeros()
    return mat


# ---------------------------------------------------------------------------
# Library callers
# ---------------------------------------------------------------------------

def _make_kv_params(d):
    """Build (n, keys_arr, vals_arr) for const char** key-value params."""
    items = [(k.encode("utf-8"), str(v).encode("utf-8")) for k, v in d.items()]
    n = len(items)
    if n == 0:
        return 0, None, None
    KeysArr = ctypes.c_char_p * n
    ValsArr = ctypes.c_char_p * n
    return n, KeysArr(*[k for k, v in items]), ValsArr(*[v for k, v in items])


def call_spalm(lib, A_csr, B_csr, C, init_method, dtype, slice_length,
               threads=4, extra=None):
    """Call spalm with a given init_method, return captured stdout.

    slice_length None leaves the key out, so the library applies its
    slice-length rule (auto_slice_length = 1) from the configured length.
    `extra` adds further keys (e.g. asl_min_cost, r_block_huge_bytes)."""
    M, K = A_csr.shape
    N, _ = B_csr.shape
    C.fill(0)
    ct = ctypes.c_float if dtype == np.float32 else ctypes.c_double
    type_num = 0 if dtype == np.float32 else 1

    func = lib.spalm
    params = {"init_method": init_method, "threads": threads}
    if slice_length is not None:
        params["slice_length"] = slice_length
    params.update(extra or {})
    n, keys_arr, vals_arr = _make_kv_params(params)

    out = StringIO()
    with pipes(stdout=out, stderr=STDOUT):
        status = func(
            type_num,
            A_csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            A_csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            A_csr.data.ctypes.data_as(ctypes.POINTER(ct)),
            B_csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B_csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            B_csr.data.ctypes.data_as(ctypes.POINTER(ct)),
            M, K, N,
            False,
            C.ctypes.data_as(ctypes.POINTER(ct)),
            None, None, None,
            n, keys_arr, vals_arr,
        )
    text = out.getvalue()
    if status != 0:
        raise BenchmarkError(
            f"{init_method}: spalm() returned status {status}\n"
            f"--- native output ---\n{text.strip()}\n--- end ---")
    return text


# ---------------------------------------------------------------------------
# Timing parsers
# ---------------------------------------------------------------------------

def parse_sparse_time(text):
    return parse_time(text, SPARSE_MM_LABEL)


def parse_dense_time(text):
    return parse_time(text, DENSE_MM_LABEL)


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def benchmark(call_fn, A_csr, B_csr, C, init_method, warmup, trials,
              parse_fn):
    """Run warmup + trials, return the median kernel time.

    Every run, including warmups, must report exactly one valid kernel timing
    and must not fall back to MKL; otherwise BenchmarkError is raised with the
    native output (missing timings usually mean a stale bin/spalm.so or an
    initialization error), instead of silently scoring the method as inf.
    """
    results = []
    for t in range(warmup + trials):
        out = call_fn(A_csr=A_csr, B_csr=B_csr, C=C, init_method=init_method)
        try:
            if re.search(r"(?m)^\s*Fallback:", out):
                raise ValueError("unexpected MKL fallback in a forced internal mode")
            val = parse_fn(out)
            if val is None:
                raise ValueError(f"no timing found by {parse_fn.__name__}")
        except ValueError as exc:
            raise BenchmarkError(
                f"{init_method}: run {t + 1}/{warmup + trials}: {exc}\n"
                f"--- native output ---\n{out.strip()}\n--- end ---") from exc
        if t >= warmup:
            results.append(val)
    return float(np.median(results))


def interpolate_thresholds(sampled, num_buckets):
    """Linearly interpolate sampled thresholds to the full 0..num_buckets range."""
    arr = [0] * (num_buckets + 1)
    keys = sorted(sampled.keys())
    for b in range(num_buckets + 1):
        if b <= keys[0]:
            arr[b] = sampled[keys[0]]
        elif b >= keys[-1]:
            arr[b] = sampled[keys[-1]]
        else:
            lo = max(k for k in keys if k <= b)
            hi = min(k for k in keys if k >= b)
            if lo == hi:
                arr[b] = sampled[lo]
            else:
                frac = (b - lo) / (hi - lo)
                val = sampled[lo] * (1 - frac) + sampled[hi] * frac
                arr[b] = int(round(val))
    return [max(0, min(255, v)) for v in arr]


def early_stop_boundary(comparisons, count):
    """Return the first bucket where *a* loses for `count` consecutive entries.

    `comparisons` is a list of (bucket, a_wins: bool).
    Returns the bucket of the first loss in the consecutive streak,
    or None if a never loses that many times in a row.
    """
    consecutive = 0
    first_bucket = None
    for bucket, a_wins in comparisons:
        if not a_wins:
            if consecutive == 0:
                first_bucket = bucket
            consecutive += 1
            if consecutive >= count:
                return first_bucket
        else:
            consecutive = 0
            first_bucket = None
    return None


def print_smoothing_summary(name, keys, before_vals, after_vals,
                            rel_tol=1e-9, abs_tol=1e-12):
    """Print a compact summary of entries modified by smoothing."""
    changed = []
    for key, before, after in zip(keys, before_vals, after_vals):
        if not math.isclose(float(before), float(after),
                            rel_tol=rel_tol, abs_tol=abs_tol):
            changed.append((key, before, after))

    print(f"{name} smoothing summary: {len(changed)}/{len(keys)} values changed")
    if not changed:
        print("  - no changes")
        return

    for key, before, after in changed:
        print(f"  - bucket {key:3d}: {before:.8g} -> {after:.8g}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_table_stage(args, lib, dtype):
    """Stage "table": decision_thresholds.txt (ss_upper, ri_upper,
    left_dense_threshold) from kernel times on uniform square inputs."""
    M = K = N = args.size
    BK = args.slice_length
    num_buckets = args.num_buckets
    step = args.step
    platform = args.platform
    es = args.early_stop

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        type_name = "float" if dtype == np.float32 else "double"
        if args.log_buckets:
            bucket_dir = f"log_{num_buckets}_c{args.curvature}"
        else:
            bucket_dir = f"linear_{num_buckets}"
        output_dir = SCRIPT_DIR / "parameters" / platform / type_name / bucket_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    buckets = list(range(1, num_buckets + 1, step))
    if buckets[-1] != num_buckets:
        buckets.append(num_buckets)
    n = len(buckets)

    def call_fn(*, A_csr, B_csr, C, init_method):
        return call_spalm(lib, A_csr, B_csr, C, init_method,
                          dtype, BK, threads=args.threads)

    print(f"Config: {M}x{K} matrices, platform={platform}, BK={BK}, "
          f"{args.dtype}")
    print(f"Sampling {n} density levels: {buckets}")
    print(f"Trials: {args.warmup} warmup + {args.trials} timed, "
          f"early stop={es}")
    print(f"Output: {output_dir}\n")

    def bucket_to_density(b):
        """Convert a bucket index to representative density."""
        if not args.log_buckets:
            return b / num_buckets
        C = float(args.curvature)
        normalized = b / (num_buckets - 1)
        return (math.exp(normalized * math.log(1 + C)) - 1) / C

    # -- Pre-generate one CSR matrix per density level -------------------------
    print(f"Generating {n} matrices...")
    matrices = {}
    for i, b in enumerate(buckets):
        density = bucket_to_density(b)
        print(f"  [{i + 1}/{n}] bucket {b:3d}: density={density * 100:.4f}% ...", end="", flush=True)
        mat = generate_uniform_csr(M, K, density, dtype=dtype, seed=b + 42)
        matrices[b] = mat
        print(f"  nnz={mat.nnz}")

    C = np.zeros(M * N, dtype=dtype)
    mid = buckets[n // 2]
    A_mid = matrices[mid]
    B_mid = matrices[mid]

    # ======================================================================
    # 1. Measure dense-dense time ONCE (constant across all densities)
    # ======================================================================
    print("\n[1/4] Measuring dense-dense time (constant)...")
    dd_time = benchmark(call_fn, A_mid, B_mid, C, "dense_bm_inner_kernel",
                        args.warmup, args.trials, parse_dense_time)
    print(f"  Dense-dense time: {dd_time:.6f}s\n")

    # ======================================================================
    # 2. Measure RI per right bucket, LI per left bucket (B each)
    # ======================================================================
    ri_times = {}
    li_times = {}
    print("[2/4] Measuring right-inner-kernel per right_bucket "
          f"(early stop if RI > DD for {es} consecutive buckets)...")
    ri_consecutive = 0
    ri_first_over = None
    for rb in buckets:
        t = benchmark(call_fn, A_mid, matrices[rb], C,
                      "dense_right_spmm_inner_kernel",
                      args.warmup, args.trials, parse_dense_time)
        ri_times[rb] = t
        over_dd = t > dd_time
        print(f"  R={rb:3d}  RI={t:.6f}s{'  (> DD)' if over_dd else ''}")
        if over_dd:
            if ri_consecutive == 0:
                ri_first_over = rb
            ri_consecutive += 1
            if ri_consecutive >= es:
                print(f"  ** Early stop: RI > DD for {es} consecutive buckets "
                      f"(first at bucket {ri_first_over})")
                break
        else:
            ri_consecutive = 0
            ri_first_over = None
    for rb in buckets:
        if rb not in ri_times:
            ri_times[rb] = np.inf

    print(f"\n[2/4] Measuring left-inner-kernel per left_bucket "
          f"(early stop if LI > DD for {es} consecutive buckets)...")
    li_consecutive = 0
    li_first_over = None
    for lb in buckets:
        t = benchmark(call_fn, matrices[lb], B_mid, C,
                      "dense_left_spmm_inner_kernel",
                      args.warmup, args.trials, parse_dense_time)
        li_times[lb] = t
        over_dd = t > dd_time
        print(f"  L={lb:3d}  LI={t:.6f}s{'  (> DD)' if over_dd else ''}")
        if over_dd:
            if li_consecutive == 0:
                li_first_over = lb
            li_consecutive += 1
            if li_consecutive >= es:
                print(f"  ** Early stop: LI > DD for {es} consecutive buckets "
                      f"(first at bucket {li_first_over})")
                break
        else:
            li_consecutive = 0
            li_first_over = None
    for lb in buckets:
        if lb not in li_times:
            li_times[lb] = np.inf
    print()

    # Apply isotonic regression smoothing to finite measured values only
    ri_measured_keys = [k for k in sorted(ri_times) if ri_times[k] < np.inf]
    ri_vals = [ri_times[k] for k in ri_measured_keys]
    if ri_measured_keys:
        ir_ri = IsotonicRegression(increasing=True)
        ri_vals_iso = ir_ri.fit_transform(ri_measured_keys, ri_vals)
        print_smoothing_summary("RI", ri_measured_keys, ri_vals, ri_vals_iso)
        for k, v in zip(ri_measured_keys, ri_vals_iso):
            ri_times[k] = v
    else:
        print("RI smoothing summary: no finite measured values")

    li_measured_keys = [k for k in sorted(li_times) if li_times[k] < np.inf]
    li_vals = [li_times[k] for k in li_measured_keys]
    if li_measured_keys:
        ir_li = IsotonicRegression(increasing=True)
        li_vals_iso = ir_li.fit_transform(li_measured_keys, li_vals)
        print_smoothing_summary("LI", li_measured_keys, li_vals, li_vals_iso)
        for k, v in zip(li_measured_keys, li_vals_iso):
            li_times[k] = v
    else:
        print("LI smoothing summary: no finite measured values")

    # ======================================================================
    # 3. Benchmark sparse-sparse per (left, right) with early stopping
    # ======================================================================
    print("[3/4] Benchmarking sparse-sparse per row (early stop on "
          f"{es} consecutive non-SS wins)...")
    ss_sampled = {}
    results = []

    for lb in buckets:
        A_csr = matrices[lb]
        consecutive = 0
        first_bucket = None

        for rb in buckets:
            B_csr = matrices[rb]

            if A_csr.nnz == 0 or B_csr.nnz == 0:
                results.append((lb, rb, 0.0, dd_time, "SS"))
                consecutive = 0
                first_bucket = None
                continue

            ss_time = benchmark(call_fn, A_csr, B_csr, C, "sparse",
                                args.warmup, args.trials, parse_sparse_time)

            ri_t = ri_times.get(rb, np.inf)
            li_t = li_times.get(lb, np.inf)
            best_other = min(ri_t, li_t, dd_time)
            if best_other == ri_t:
                other_tag = "RI"
            elif best_other == li_t:
                other_tag = "LI"
            else:
                other_tag = "DD"

            ss_wins = ss_time <= best_other
            winner = "SS" if ss_wins else other_tag
            results.append((lb, rb, ss_time, best_other, winner))
            print(f"  L={lb:3d} R={rb:3d}  SS={ss_time:.6f}s  "
                  f"{other_tag}={best_other:.6f}s  -> {winner}")

            if not ss_wins:
                if consecutive == 0:
                    first_bucket = rb
                consecutive += 1
                if consecutive >= es:
                    ss_sampled[lb] = first_bucket
                    print(f"  ** Early stop: ss_upper[{lb}] = {first_bucket}")
                    break
            else:
                consecutive = 0
                first_bucket = None
        else:
            ss_sampled[lb] = num_buckets + 1

    # ======================================================================
    # 4. Derive thresholds
    # ======================================================================
    print(f"\n[4/4] Computing thresholds...")
    ss_upper_raw = interpolate_thresholds(ss_sampled, num_buckets)
    # Isotonic regression for non-increasing ss_upper
    ir_ss = IsotonicRegression(increasing=False)
    ss_keys = list(range(len(ss_upper_raw)))
    ss_upper = ir_ss.fit_transform(ss_keys, ss_upper_raw)
    print_smoothing_summary("SS", ss_keys, ss_upper_raw, ss_upper)
    ss_upper = [int(round(v)) for v in ss_upper]
    print(f"\n  ss_upper summary ({len(ss_upper)} values, "
          f"range [{min(ss_upper)}, {max(ss_upper)}]):")
    print("  Isotonic regression applied for non-increasing smoothing.")

    # ri_upper: for each left row, first right bucket where RI is not the
    # fastest method (excluding SS). Scans all right buckets independently
    # of the SS boundary, so ri_upper can be smaller than ss_upper.
    ri_sampled = {}
    for lb in buckets:
        comparisons = []
        for rb in buckets:
            ri_t = ri_times.get(rb, np.inf)
            best_non_ri = min(li_times.get(lb, np.inf), dd_time)
            comparisons.append((rb, ri_t <= best_non_ri))
        boundary = early_stop_boundary(comparisons, es)
        ri_sampled[lb] = boundary if boundary is not None else (
                num_buckets + 1)

    ri_upper = interpolate_thresholds(ri_sampled, num_buckets)

    # left_dense_threshold: where DD consistently beats LI
    comparisons = []
    for lb in buckets:
        li_t = li_times.get(lb, np.inf)
        comparisons.append((lb, li_t <= dd_time))
    boundary = early_stop_boundary(comparisons, es)
    left_dense_threshold = boundary if boundary is not None else (
            num_buckets + 1)
    print(f"  left_dense_threshold = {left_dense_threshold}")

    # -- Save decision_thresholds.txt ------------------------------------------
    thres_path = output_dir / "decision_thresholds.txt"
    with open(thres_path, "w") as f:
        f.write(f"num_buckets: {num_buckets}\n")
        f.write(f"ss_upper: {' '.join(map(str, ss_upper))}\n")
        f.write(f"ri_upper: {' '.join(map(str, ri_upper))}\n")
        f.write(f"left_dense_threshold: {left_dense_threshold}\n")
        f.write(f"dense_mm_threshold: 500000000\n")
    print(f"\nSaved {thres_path}")

    # -- Save CSV --------------------------------------------------------------
    csv_path = output_dir / "algorithm_map.csv"
    with open(csv_path, "w") as f:
        f.write("left_bucket,right_bucket,sparse_time,best_other_time,"
                "winner\n")
        for lb, rb, st, ot, w in results:
            f.write(f"{lb},{rb},{st:.8f},{ot:.8f},{w}\n")
    print(f"Saved {csv_path}")

    # -- Save heatmap PNG ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Compute density tick positions and labels from bucket indices
        num_ticks = 10
        tick_step = max(1, num_buckets // num_ticks)
        tick_buckets = list(range(0, num_buckets + 1, tick_step))
        if tick_buckets[-1] != num_buckets:
            tick_buckets.append(num_buckets)
        tick_labels = [f"{bucket_to_density(b) * 100:.1f}" for b in tick_buckets]

        def apply_density_ticks(ax):
            ax.set_xticks(tick_buckets)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(tick_buckets)
            ax.set_yticklabels(tick_labels, fontsize=7)

        # Left panel: algorithm decision map from derived thresholds
        ax = axes[0]
        alg_grid = np.zeros((num_buckets + 1, num_buckets + 1))
        for lb_i in range(num_buckets + 1):
            for rb_i in range(num_buckets + 1):
                if rb_i < ss_upper[lb_i]:
                    alg_grid[lb_i, rb_i] = 0
                elif rb_i < ri_upper[lb_i]:
                    alg_grid[lb_i, rb_i] = 1
                elif lb_i >= left_dense_threshold:
                    alg_grid[lb_i, rb_i] = 2
                else:
                    alg_grid[lb_i, rb_i] = 3

        alg_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        alg_labels = ["SS (sparse-sparse)", "RI (right-inner)",
                      "DD (dense-dense)", "LI (left-inner)"]
        cmap = ListedColormap(alg_colors)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        ax.imshow(alg_grid, origin="lower", cmap=cmap, norm=norm,
                  aspect="equal", extent=[0, num_buckets, 0, num_buckets])
        ax.plot(ss_upper, range(num_buckets + 1), "k-", lw=2,
                label="ss_upper")
        ax.plot(ri_upper, range(num_buckets + 1), "w--", lw=2,
                label="ri_upper")
        ax.axhline(y=left_dense_threshold, color="yellow", ls=":", lw=2,
                   label=f"left_dense={left_dense_threshold}")
        apply_density_ticks(ax)
        ax.set_xlabel("Right density (%)")
        ax.set_ylabel("Left density (%)")
        ax.set_title("Algorithm decision map")

        # Two separate legends: boundary lines and algorithm colour patches
        boundary_legend = ax.legend(loc="upper right", fontsize=8)
        alg_patches = [Patch(facecolor=c, label=l)
                       for c, l in zip(alg_colors, alg_labels)]
        ax.add_artist(boundary_legend)
        ax.legend(handles=alg_patches, loc="lower right", fontsize=8,
                  title="Algorithm")

        # Right panel: measured SS times
        ax = axes[1]
        ss_grid = np.full((num_buckets + 1, num_buckets + 1), np.nan)
        for lb, rb, st, ot, w in results:
            if st < np.inf:
                ss_grid[lb, rb] = st
        im = ax.imshow(ss_grid, origin="lower", aspect="equal",
                       extent=[0, num_buckets, 0, num_buckets])
        apply_density_ticks(ax)
        ax.set_xlabel("Right density (%)")
        ax.set_ylabel("Left density (%)")
        ax.set_title("Sparse-sparse kernel time (s)")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        png_path = output_dir / "algorithm_map.png"
        fig.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved {png_path}")
    except ImportError:
        print("matplotlib not available, skipping heatmap")


# ---------------------------------------------------------------------------
# Stages "asl" and "rblock": calibration.txt (spalm_calibration.py)
# ---------------------------------------------------------------------------

# The dense-path kernels whose packing buffers and slice length these stages
# calibrate (forced init_method, as in the table stage).
DENSE_METHODS = {
    "DD": "dense_bm_inner_kernel",
    "SD": "dense_left_spmm_inner_kernel",
    "DS": "dense_right_spmm_inner_kernel",
}

def run_dense(lib, A_csr, B_csr, C, method, dtype, threads, label,
              slice_length=None, extra=None):
    """One forced dense-path call: (seconds at `label`, native output)."""
    M, N = A_csr.shape[0], B_csr.shape[0]
    out = call_spalm(lib, A_csr, B_csr, C[:M * N], DENSE_METHODS[method],
                     dtype, slice_length, threads, extra)
    try:
        if re.search(r"(?m)^\s*Fallback:", out):
            raise ValueError("unexpected MKL fallback in a forced internal mode")
        value = parse_time(out, label)
        if value is None:
            raise ValueError(f"no {label} line")
    except ValueError as exc:
        raise BenchmarkError(
            f"{method}: {exc}\n--- native output ---\n{out.strip()}\n"
            f"--- end ---") from exc
    return value, out


def interleaved(run_a, run_b, warmup, trials):
    """Median times of two settings, run alternately (a b, b a, ...) so that
    drift over the stage hits both alike. Each run returns (seconds, out);
    also returns the last output of each."""
    for _ in range(warmup):
        run_a()
        run_b()
    times_a, times_b = [], []
    out_a = out_b = None
    for t in range(trials):
        order = ((run_a, times_a), (run_b, times_b))
        for fn, times in (order if t % 2 == 0 else order[::-1]):
            value, out = fn()
            times.append(value)
            if times is times_a:
                out_a = out
            else:
                out_b = out
    return calib.median(times_a), calib.median(times_b), out_a, out_b


def as_spalm_csr(mat, dtype):
    """CSR with sorted, duplicate-free int32 indices and contiguous values,
    as spalm() reads them."""
    mat = csr_matrix(mat, dtype=dtype)
    mat.sum_duplicates()
    mat.indptr = mat.indptr.astype(np.int32)
    mat.indices = mat.indices.astype(np.int32)
    mat.data = np.ascontiguousarray(mat.data, dtype=dtype)
    return mat


def _random_values(rng, count, dtype):
    values = rng.standard_normal(count).astype(dtype)
    values[values == 0] = dtype(1e-6)
    return values


def generate_powerlaw_csr(rows, cols, mean_degree, dtype=np.float64,
                          seed=None, alpha=2.0, zipf=0.7):
    """Graph-like rows: Pareto(alpha) row degrees of the given mean, keys
    drawn with Zipf(zipf) popularity over a random permutation of the keys
    (hub columns spread over the key range). Repeated keys in a row merge,
    so the stored mean is somewhat below `mean_degree`."""
    rng = np.random.default_rng(seed)
    scale = mean_degree * (alpha - 1) / alpha  # Pareto(alpha, 1) mean is alpha/(alpha-1)
    degree = np.clip(np.rint(scale * (rng.pareto(alpha, rows) + 1.0)), 1, cols)
    degree = degree.astype(np.int64)
    weights = np.arange(1, cols + 1, dtype=np.float64) ** -zipf
    weights /= weights.sum()
    keys = rng.permutation(cols)[
        rng.choice(cols, size=int(degree.sum()), p=weights)]
    row_of = np.repeat(np.arange(rows, dtype=np.int64), degree)
    flat = np.unique(row_of * cols + keys)
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(flat // cols, minlength=rows), out=indptr[1:])
    mat = csr_matrix(
        (_random_values(rng, flat.size, dtype), flat % cols, indptr),
        shape=(rows, cols))
    return as_spalm_csr(mat, dtype)


def generate_fixed_rows_csr(rows, cols, per_row, dtype=np.float64, seed=None):
    """Every row holds `per_row` distinct keys, uniform over [0, cols)."""
    rng = np.random.default_rng(seed)
    keys = np.sort(np.stack(
        [rng.choice(cols, per_row, replace=False) for _ in range(rows)]),
        axis=1)
    indptr = np.arange(0, rows * per_row + 1, per_row, dtype=np.int64)
    mat = csr_matrix(
        (_random_values(rng, rows * per_row, dtype), keys.ravel(), indptr),
        shape=(rows, cols))
    return as_spalm_csr(mat, dtype)


# asl stage inputs: square n x n, R and S drawn independently. Uniform rows
# at these densities hold fewer than asl_subrow_write_entries (16) entries
# per subrow at 512, so the rule considers longer slices; the power-law
# pairs stand in for the graph-like inputs on which the model overrated
# longer slices (slice_length_rule.h).
ASL_UNIFORM_DENSITIES = (0.002, 0.005, 0.01, 0.02)
ASL_POWERLAW_DEGREES = (8, 32)


def asl_cases(sizes):
    """(name, n, kind, density or mean degree, seed) per asl input pair."""
    seed = 1000
    for n in sizes:
        for density in ASL_UNIFORM_DENSITIES:
            seed += 2
            yield f"u{n}_d{density:g}", n, "uniform", density, seed
        for degree in ASL_POWERLAW_DEGREES:
            seed += 2
            yield f"p{n}_k{degree}", n, "powerlaw", degree, seed


def make_asl_pair(n, kind, param, seed, dtype):
    if kind == "uniform":
        R = generate_uniform_csr(n, n, param, dtype=dtype, seed=seed)
        S = generate_uniform_csr(n, n, param, dtype=dtype, seed=seed + 1)
        return as_spalm_csr(R, dtype), as_spalm_csr(S, dtype)
    return (generate_powerlaw_csr(n, n, param, dtype=dtype, seed=seed),
            generate_powerlaw_csr(n, n, param, dtype=dtype, seed=seed + 1))


def run_asl_stage(args, lib, dtype, calibration_path):
    """Stage "asl": asl_min_cost, the smallest modelled cost at which the
    slice-length rule (auto_slice_length = 1) may lengthen.

    For each synthetic pair and each of DD, SD and DS, one call without a
    gate (asl_min_cost = 0) gives the rule's choice and its cost0. Where the
    rule lengthens, the call is timed ([slice-mm-total]: slice preparation
    plus multiplication) at the configured and at the chosen length. The
    gate is then the cut on cost0 that minimises the geometric mean time
    over all those cells (choose_min_cost)."""
    sizes = [int(s) for s in args.asl_sizes.split(",")]
    print(f"\n[asl] slice-length gate: sizes {sizes}, "
          f"{args.warmup} warmup + {args.asl_trials} timed runs per length")
    C = np.empty(max(sizes) ** 2, dtype=dtype)
    rows = []
    for name, n, kind, param, seed in asl_cases(sizes):
        R, S = make_asl_pair(n, kind, param, seed, dtype)
        print(f"  {name}: nnz R={R.nnz} S={S.nnz}")
        for method in DENSE_METHODS:
            _, out = run_dense(lib, R, S, C, method, dtype, args.threads,
                               SLICE_MM_TOTAL_LABEL,
                               extra={"asl_min_cost": 0})
            try:
                probe = parse_slice_length(out)
            except ValueError as exc:
                raise BenchmarkError(f"{name} {method}: {exc}") from exc
            row = {"case": name, "method": method, "n": n,
                   "nnz_r": R.nnz, "nnz_s": S.nnz}
            if probe is None:
                print(f"    {method}: rule did not run")
                rows.append(row)
                continue
            configured, chosen, sides, cost0 = probe
            row.update(configured=configured, chosen=chosen, sides=sides,
                       cost0=cost0)
            if chosen == configured:
                print(f"    {method}: cost0={cost0:.3g} keeps {configured}")
                rows.append(row)
                continue

            def at(length):
                return lambda: run_dense(lib, R, S, C, method, dtype,
                                         args.threads, SLICE_MM_TOTAL_LABEL,
                                         slice_length=length)

            t_base, t_long, _, _ = interleaved(
                at(configured), at(chosen), args.warmup, args.asl_trials)
            row.update(t_base=t_base, t_long=t_long)
            print(f"    {method}: cost0={cost0:.3g} {configured}->{chosen} "
                  f"{t_base * 1e3:.2f} -> {t_long * 1e3:.2f} ms "
                  f"({t_long / t_base:.3f}x)")
            rows.append(row)
        del R, S
    del C

    timed = [r for r in rows if "t_base" in r]
    out_dir = calibration_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if not timed:
        print("[asl] the rule lengthened no input; asl_min_cost unchanged")
        write_rows(out_dir / "asl_cells.csv", rows, ASL_COLUMNS)
        return
    cells = [(r["cost0"], r["t_base"], r["t_long"]) for r in timed]
    min_cost, lengthened = calib.choose_min_cost(cells)
    for i, r in enumerate(timed):
        r["lengthened"] = int(i in lengthened)
    write_rows(out_dir / "asl_cells.csv", rows, ASL_COLUMNS)

    base_sum = sum(c[1] for c in cells)
    gated_sum = sum(c[2] if i in lengthened else c[1]
                    for i, c in enumerate(cells))
    long_sum = sum(c[2] for c in cells)
    gated_log = sum(math.log(c[2] / c[1]) for i, c in enumerate(cells)
                    if i in lengthened)
    print(f"\n[asl] {len(cells)} cells the rule lengthens without a gate; "
          f"the gate keeps {len(lengthened)} lengthened")
    print(f"  sum: base {base_sum * 1e3:.1f} ms, ungated "
          f"{long_sum * 1e3:.1f} ms, gated {gated_sum * 1e3:.1f} ms; "
          f"gated gmean {math.exp(gated_log / len(cells)):.3f}x of base")
    if min_cost == calib.NEVER_MIN_COST:
        note = "no synthetic input ran faster longer: never lengthen"
    else:
        note = f"{len(lengthened)} of {len(cells)} lengthened cells kept"
    print(f"  asl_min_cost = {calib.format_value(min_cost)} ({note})")
    if len(cells) < 6:
        print(f"  warning: only {len(cells)} cells; add --asl-sizes")
    calib.update_calibration(
        calibration_path, {"asl_min_cost": min_cost},
        f"find_thresholds.py asl {time.strftime('%Y-%m-%d')}, "
        f"threads={args.threads}: {note}")
    print(f"Updated {calibration_path}")


ASL_COLUMNS = ("case", "method", "n", "nnz_r", "nnz_s", "configured",
               "chosen", "sides", "cost0", "t_base", "t_long", "lengthened")

# rblock stage inputs: R with 12 * slices rows (one R slice per 12 rows, every
# slice nonempty in every slice column), RBLOCK_KEYS keys per row over
# RBLOCK_K keys; S with RBLOCK_S_ROWS rows. The kernel stays small, so the
# per-call cost of mapping R's packing block dominates [mm-total]. The R
# block holds min(192, slices) slices (tile_size_l3 / 12) of slice length x
# 16 (DS) or 12 (DD) values: 0.75 to 48 MiB over these configurations.
RBLOCK_CONFIGS = ((512, (16, 32, 48, 64, 96, 128, 160, 192)),
                  (1024, (128, 192)),
                  (2048, (128, 192)))
RBLOCK_K = 4096
RBLOCK_KEYS = 8
RBLOCK_S_ROWS = 64
RBLOCK_METHODS = ("DS", "DD")  # SD packs no R block
RBLOCK_COLUMNS = ("method", "slice_length", "r_slices", "r_bytes", "r_huge",
                  "t_huge", "t_never", "delta")


def run_rblock_stage(args, lib, dtype, calibration_path):
    """Stage "rblock": r_block_huge_bytes, the R packing block size from
    which a huge-page mapping pays.

    For each block size, DS and DD calls run alternately with the block on
    huge pages (r_block_huge_bytes = 0) and never (-1). The difference of the
    median [mm-total] times is fitted by a line in the block size; its zero,
    rounded up to a whole MiB, is the threshold (huge_break_even). On the
    8488C the mapping cost about 0.5 ms flat (one 2 MiB page zeroed in one
    fault behind the packing barrier) and saved about 60 us per MiB at the
    free (packing_buffers.h)."""
    print(f"\n[rblock] R block huge pages: {args.warmup} warmup + "
          f"{args.rblock_trials} timed runs per setting")
    S = generate_fixed_rows_csr(RBLOCK_S_ROWS, RBLOCK_K, RBLOCK_KEYS,
                                dtype=dtype, seed=7)
    max_slices = max(max(n) for _, n in RBLOCK_CONFIGS)
    C = np.empty(12 * max_slices * RBLOCK_S_ROWS, dtype=dtype)
    rows, points = [], []
    for sl, slice_counts in RBLOCK_CONFIGS:
        for slices in slice_counts:
            R = generate_fixed_rows_csr(12 * slices, RBLOCK_K, RBLOCK_KEYS,
                                        dtype=dtype, seed=100 + slices)
            for method in RBLOCK_METHODS:
                def setting(value):
                    return lambda: run_dense(
                        lib, R, S, C, method, dtype, args.threads,
                        MM_TOTAL_LABEL, slice_length=sl,
                        extra={"r_block_huge_bytes": value})

                t_huge, t_never, out_huge, out_never = interleaved(
                    setting(0), setting(-1), args.warmup, args.rblock_trials)
                try:
                    r_bytes, r_huge = parse_r_block(out_huge)
                    never_bytes, never_huge = parse_r_block(out_never)
                except ValueError as exc:
                    raise BenchmarkError(f"{method} {sl} {slices}: {exc}") \
                        from exc
                if never_huge or never_bytes != r_bytes:
                    raise BenchmarkError(
                        f"{method} {sl} {slices}: r_block_huge_bytes=-1 gave "
                        f"r_bytes={never_bytes} r_huge={int(never_huge)}")
                delta = t_huge - t_never
                rows.append({"method": method, "slice_length": sl,
                             "r_slices": slices, "r_bytes": r_bytes,
                             "r_huge": int(r_huge), "t_huge": t_huge,
                             "t_never": t_never, "delta": delta})
                if r_huge:
                    points.append((r_bytes, delta))
                print(f"  {method} sl={sl:4d} slices={slices:3d} "
                      f"R={r_bytes / calib.MIB:6.2f} MiB huge={int(r_huge)}  "
                      f"{t_never * 1e3:.3f} -> {t_huge * 1e3:.3f} ms "
                      f"({delta * 1e3:+.3f})")
            del R

    out_dir = calibration_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "rblock_cells.csv", rows, RBLOCK_COLUMNS)
    if len({b for b, _ in points}) < 2:
        print("[rblock] fewer than two block sizes were mapped (huge pages "
              "unavailable?); r_block_huge_bytes unchanged")
        return
    value, (a, b) = calib.huge_break_even(points)
    print(f"\n[rblock] t_huge - t_never = {a * 1e3:+.3f} ms "
          f"{b * 1e3:+.4f} ms/MiB over {len(points)} mapped cells")
    if value < 0:
        note = "mapping never paid: never"
    else:
        note = f"break-even {value / calib.MIB:.0f} MiB"
    print(f"  r_block_huge_bytes = {value} ({note})")
    calib.update_calibration(
        calibration_path, {"r_block_huge_bytes": value},
        f"find_thresholds.py rblock {time.strftime('%Y-%m-%d')}, "
        f"threads={args.threads}: {note}")
    print(f"Updated {calibration_path}")


def write_rows(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


STAGES = ("table", "rblock", "asl")


def main():
    parser = argparse.ArgumentParser(
        description="Find algorithm decision thresholds and the "
                    "calibration.txt defaults")
    parser.add_argument("--stages", default=",".join(STAGES),
                        help="Comma-separated subset of table, rblock, asl "
                             "(default: all; always run in that order)")
    parser.add_argument("--size", type=int, default=2048,
                        help="Square matrix dimension (default: 2048)")
    parser.add_argument("--step", type=int, default=1,
                        help="Bucket step size for sampling (default: 5)")
    parser.add_argument("--dtype", choices=["float32", "float64"],
                        default="float32")
    parser.add_argument("--slice-length", type=int, default=1024,
                        help="Slice length / BK (default: 1024)")
    parser.add_argument("--num-buckets", type=int, default=100)
    parser.add_argument("--platform", choices=["cpu"], default="cpu",
                        help="Target platform (default: cpu)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: parameters/<platform>/<type>/)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs before timing (default: 1)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Timed runs, keep best (default: 3)")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads (default: 4, CPU platform only)")
    parser.add_argument("--early-stop", type=int, default=3,
                        help="Consecutive non-winner tests to stop a row "
                             "(default: 3)")
    parser.add_argument("--log-buckets", action="store_true", default=False,
                        help="Use logarithmic bucket spacing")
    parser.add_argument("--curvature", type=int, default=50,
                        help="Log curvature parameter C (int, default 50 "
                             "=> ~10%% of buckets below 1%% density)")
    parser.add_argument("--calibration-file", type=str, default=None,
                        help="calibration.txt the rblock and asl stages "
                             "update (default: parameters/<platform>/<type>/"
                             "calibration.txt, the one the library reads)")
    parser.add_argument("--asl-sizes", default="2048,4096,8192,16384",
                        help="Square sizes of the asl stage's inputs "
                             "(default: 2048,4096,8192,16384)")
    parser.add_argument("--asl-trials", type=int, default=3,
                        help="Timed runs per slice length, asl stage "
                             "(default: 3)")
    parser.add_argument("--rblock-trials", type=int, default=15,
                        help="Timed runs per setting, rblock stage "
                             "(default: 15)")
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = sorted(set(stages) - set(STAGES))
    if unknown or not stages:
        parser.error(f"--stages: unknown {unknown}; choose from {STAGES}")
    dtype = np.float32 if args.dtype == "float32" else np.float64
    type_name = "float" if dtype == np.float32 else "double"
    calibration_path = Path(args.calibration_file) if args.calibration_file \
        else SCRIPT_DIR / "parameters" / args.platform / type_name / \
        "calibration.txt"

    lib = load_so("spalm.so")
    # rblock before asl: the asl stage's timings then use the R block
    # threshold this run measured, as the library will.
    for stage in STAGES:
        if stage not in stages:
            continue
        if stage == "table":
            run_table_stage(args, lib, dtype)
        elif stage == "rblock":
            run_rblock_stage(args, lib, dtype, calibration_path)
        else:
            run_asl_stage(args, lib, dtype, calibration_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
