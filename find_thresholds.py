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

Before the thresholds, step 0 measures the machine parameters (dense-path
tile sizes, the base slice length and the automatic slice length rule, the
huge-page result offset) on
synthetic uniform matrices and writes them to
parameters/<platform>/<type>/machine_params.txt, which the library reads; the
threshold measurements then use them. --machine-params use skips step 0.

Produces:
  - machine_params.txt       (machine parameters, step 0)
  - machine_params.csv       (step 0 timings)
  - decision_thresholds.txt  (threshold file for spalm)
  - algorithm_map.csv        (detailed timings)
  - algorithm_map.png        (heatmap visualization)
"""

import argparse
import ctypes
import math
import mmap
import os
import platform as host_platform
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.sparse import random as sparse_random, csr_matrix
from sklearn.isotonic import IsotonicRegression
from wurlitzer import STDOUT, pipes

import machine_tuning
from spalm_timing import DENSE_MM_LABEL, SPARSE_MM_LABEL, parse_time

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

    slice_length None leaves it to the library (tile_sizes.txt, then the
    automatic length), as at run time. `extra` adds key/value options.
    """
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
              parse_fn, allow_fallback=False):
    """Run warmup + trials, return the median kernel time.

    Every run, including warmups, must report exactly one valid kernel timing
    and must not fall back to MKL (unless allow_fallback: the forced MKL
    methods); otherwise BenchmarkError is raised with the native output
    (missing timings usually mean a stale bin/spalm.so or an initialization
    error), instead of silently scoring the method as inf.
    """
    results = []
    for t in range(warmup + trials):
        out = call_fn(A_csr=A_csr, B_csr=B_csr, C=C, init_method=init_method)
        try:
            if not allow_fallback and re.search(r"(?m)^\s*Fallback:", out):
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
# Step 0: machine parameters
# ---------------------------------------------------------------------------

HUGE_PAGE_BYTES = 2 << 20
TOTAL_LABEL = "[spalm-total]"
DENSE_METHODS = {"SD": "dense_left_spmm_inner_kernel",
                 "DS": "dense_right_spmm_inner_kernel",
                 "DD": "dense_bm_inner_kernel"}
SPARSE_OUTPUT_METHODS = {"SS": "sparse", "MKL-S": "FORCE_MKL_SPARSE"}
# Candidates. The tile sizes are R rows (L3, shared packed R block) and S rows
# (L2, per-thread packed S block) per slice column.
TILE_L2_CANDIDATES = (32, 64, 128, 256)
TILE_L3_CANDIDATES = (288, 576, 1152, 2304, 4608, 9216)
TILE_DENSITIES = (0.05, 0.25)
# Average entries per subrow at the configured slice length: SD, DS and DD
# (automatic length), and SS (always the base length).
SLICE_ENTRIES = (2, 4, 8, 16, 32, 64, 128)
SS_SLICE_ENTRIES = (4, 32)
SLICE_BASES = (256, 512, 1024)
SLICE_TARGETS = (1, 2, 4, 8, 16, 32, 64, 128)
SLICE_MAXIMA = (512, 1024, 2048, 4096)
OFFSETS = (0, 16, 32, 64)
OFFSET_DENSITY = 0.05
# The offset effect is small next to run-to-run noise: every offset is
# measured in each of OFFSET_ROUNDS rounds, in rotated order, and the
# median of the round medians is used.
OFFSET_ROUNDS = 5
# A candidate must beat the default by more than this to replace it.
TOLERANCE = 0.01
OFFSET_TOLERANCE = 0.02


def huge_page_array(count, dtype, offset):
    """A 1-D array starting `offset` bytes past a 2 MiB boundary of an
    anonymous mapping marked MADV_HUGEPAGE, placed as the DuckDB extension
    places its dense result (AllocateDenseResult in result_storage.hpp).
    Returns (array, mapping, advised); keep the mapping while the array is used.
    """
    itemsize = np.dtype(dtype).itemsize
    length = ((count * itemsize + offset + HUGE_PAGE_BYTES - 1)
              // HUGE_PAGE_BYTES + 1) * HUGE_PAGE_BYTES
    mapping = mmap.mmap(-1, length, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    view = ctypes.c_char.from_buffer(mapping)
    start = -ctypes.addressof(view) % HUGE_PAGE_BYTES
    del view
    advised = False
    if hasattr(mmap, "MADV_HUGEPAGE"):
        try:
            mapping.madvise(mmap.MADV_HUGEPAGE, start, length - start)
            advised = True
        except OSError:
            pass
    array = np.frombuffer(mapping, dtype=dtype, count=count,
                          offset=start + offset)
    return array, mapping, advised


def configured_slice_length(platform, type_name):
    """slice_length of parameters/<platform>/<type>/tile_sizes.txt."""
    path = SCRIPT_DIR / "parameters" / platform / type_name / "tile_sizes.txt"
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "slice_length":
            return int(value)
    raise BenchmarkError(f"no slice_length in {path}")


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return host_platform.processor() or "unknown"


def tune_machine_params(lib, args, dtype, platform, type_name, params_path):
    """Measure and write machine_params.txt; return the parameters.

    Every call passes the parameters it depends on explicitly, so an existing
    machine_params.txt does not influence the measurements. Synthetic uniform
    size x size matrices, the threshold run's threads and dtype.
    """
    size = args.tune_size
    configured = configured_slice_length(platform, type_name)
    defaults = machine_tuning.DEFAULTS
    csv_rows = []

    def measure(stage, A, B, C, name, method, label, candidate, slice_length,
                extra, case):
        def parse(text):
            return parse_time(text, label)

        def call_fn(*, A_csr, B_csr, C, init_method):
            return call_spalm(lib, A_csr, B_csr, C, init_method, dtype,
                              slice_length, threads=args.threads, extra=extra)

        t = benchmark(call_fn, A, B, C, method, args.warmup, args.trials,
                      parse, allow_fallback=method.startswith("FORCE_MKL"))
        csv_rows.append((stage, name, case, candidate, t))
        print(f"  {stage} {name} {case} {candidate}: {t:.6f}s")
        return t

    def uniform_pair(density, seed):
        return (generate_uniform_csr(size, size, density, dtype, seed),
                generate_uniform_csr(size, size, density, dtype, seed + 1))

    print(f"[0/4] Machine parameters: {size}x{size} uniform matrices, "
          f"configured slice length {configured}")
    C = np.zeros(size * size, dtype=dtype)

    # -- Tile sizes: SD, DS, DD at the configured slice length ---------------
    tile_pairs = {d: uniform_pair(d, 1000 + i)
                  for i, d in enumerate(TILE_DENSITIES)}
    tile_table = {}
    for l2 in TILE_L2_CANDIDATES:
        for l3 in TILE_L3_CANDIDATES:
            extra = {"tile_size_l2": l2, "tile_size_l3": l3}
            times = {}
            for name, method in DENSE_METHODS.items():
                for d in TILE_DENSITIES:
                    times[(name, d)] = measure(
                        "tiles", *tile_pairs[d], C, name, method,
                        DENSE_MM_LABEL, f"l2={l2} l3={l3}", configured,
                        extra, f"density={d}")
            tile_table[(l2, l3)] = times
    # One pair for calls mixing SD, DS and DD, and one per algorithm (they
    # differ: SD prefers small S tiles, DS larger ones).
    default_tiles = (defaults["tile_size_l2"], defaults["tile_size_l3"])
    tile_params = {}
    method_tiles = {}
    for suffix, names in [("", tuple(DENSE_METHODS))] + [
            (f"_{name.lower()}", (name,)) for name in DENSE_METHODS]:
        table = {candidate: {case: t for case, t in times.items()
                             if case[0] in names}
                 for candidate, times in tile_table.items()}
        (l2, l3), costs = machine_tuning.pick_candidate(
            table, prefer=default_tiles, tolerance=TOLERANCE)
        print(f"  tiles{suffix or ' (all)'}: l2={l2} l3={l3} (relative cost "
              f"{costs[(l2, l3)]:.3f}, default "
              f"{costs.get(default_tiles, math.nan):.3f})")
        tile_params[f"tile_size_l2{suffix}"] = l2
        tile_params[f"tile_size_l3{suffix}"] = l3
        if len(names) == 1:
            method_tiles[names[0]] = {"tile_size_l2": l2, "tile_size_l3": l3}
    # SS ignores them.
    method_tiles["SS"] = method_tiles["MKL-S"] = {}

    # -- Slice lengths: base length and automatic rule ------------------------
    # SD, DS and DD over slice lengths at several densities (they follow the
    # automatic rule), SS at two (it always runs at the base length).
    lengths = [min(SLICE_BASES)]
    while lengths[-1] * 2 <= min(max(SLICE_MAXIMA), size):
        lengths.append(lengths[-1] * 2)
    slice_times = {}
    methods = [(name, method, SLICE_ENTRIES, True)
               for name, method in DENSE_METHODS.items()]
    methods.append(("SS", SPARSE_OUTPUT_METHODS["SS"], SS_SLICE_ENTRIES, False))
    pairs = {}
    for i, entries in enumerate(sorted(set(SLICE_ENTRIES + SS_SLICE_ENTRIES))):
        pairs[entries] = uniform_pair(entries / configured, 2000 + 2 * i)
    for name, method, entry_list, automatic in methods:
        label = DENSE_MM_LABEL if automatic else SPARSE_MM_LABEL
        for entries in entry_list:
            A, B = pairs[entries]
            density = min(A.nnz / (size * size), B.nnz / (size * size))
            measured = {}
            for length in lengths:
                measured[length] = measure(
                    "slice", A, B, C, name, method, label, f"length={length}",
                    length, method_tiles[name], f"entries={entries}")
            slice_times[(name, entries)] = (density, measured, automatic)
    bases = [b for b in SLICE_BASES if b in lengths]
    maxima = [m for m in SLICE_MAXIMA if m <= lengths[-1]]
    (base, target, max_length), costs = machine_tuning.fit_slice_rule(
        slice_times, size, bases, SLICE_TARGETS, maxima)
    default_rule = (configured, defaults["slice_target_entries"],
                    defaults["slice_length_max"])
    if (default_rule in costs and costs[default_rule]
            <= costs[(base, target, max_length)] * (1 + TOLERANCE)):
        base, target, max_length = default_rule
    print(f"  slice rule: base={base} target={target} max={max_length} "
          f"(relative cost {costs[(base, target, max_length)]:.3f}, default "
          f"{costs.get(default_rule, math.nan):.3f})")
    rule = {"slice_length": base, "slice_target_entries": target,
            "slice_length_max": max_length}
    # The rule per algorithm, at that base length: SD, DS and DD gain from
    # longer slices at different subrow sizes (one shared rule made DD on
    # uniform 5% 31% slower in encoder-1025 v12 while SD and DS gained).
    method_rule = {}
    for name in DENSE_METHODS:
        cases = {case: v for case, v in slice_times.items() if case[0] == name}
        (_, t, m), costs = machine_tuning.fit_slice_rule(
            cases, size, (base,), SLICE_TARGETS, maxima)
        default = (base, defaults[f"slice_target_entries_{name.lower()}"],
                   defaults[f"slice_length_max_{name.lower()}"])
        if default in costs and costs[default] <= costs[(base, t, m)] * (
                1 + TOLERANCE):
            _, t, m = default
        print(f"  slice rule {name}: target={t} max={m} (relative cost "
              f"{costs[(base, t, m)]:.3f}, default "
              f"{costs.get(default, math.nan):.3f})")
        rule[f"slice_target_entries_{name.lower()}"] = t
        rule[f"slice_length_max_{name.lower()}"] = m
        method_rule[name] = (t, m)
    del C

    # -- Result offset on huge pages, whole library call ----------------------
    A, B = tile_pairs[OFFSET_DENSITY] if OFFSET_DENSITY in tile_pairs else \
        uniform_pair(OFFSET_DENSITY, 3000)
    # The slice length the library would choose with the parameters above
    # (an explicit slice_length key turns the automatic rule off).
    density = min(A.nnz / (size * size), B.nnz / (size * size))
    run_length = {"sparse": base, "FORCE_MKL_SPARSE": base}
    for name, method in DENSE_METHODS.items():
        run_length[method] = machine_tuning.auto_slice_length(
            base, density, size, *method_rule[name])
    offsets = {}
    advised_all = True
    for key, methods in (("result_base_offset_sparse", SPARSE_OUTPUT_METHODS),
                         ("result_base_offset_dense", DENSE_METHODS)):
        buffers = {}
        for offset in OFFSETS:
            buffers[offset] = huge_page_array(size * size, dtype, offset)
            advised_all = advised_all and buffers[offset][2]
        rounds = {offset: {name: [] for name in methods} for offset in OFFSETS}
        for r in range(OFFSET_ROUNDS):
            order = OFFSETS[r % len(OFFSETS):] + OFFSETS[:r % len(OFFSETS)]
            for offset in order:
                for name, method in methods.items():
                    rounds[offset][name].append(measure(
                        "offset", A, B, buffers[offset][0], name, method,
                        TOTAL_LABEL, f"offset={offset} round={r + 1}",
                        run_length[method], method_tiles[name],
                        f"density={OFFSET_DENSITY}"))
        del buffers
        table = {offset: {name: float(np.median(ts))
                          for name, ts in times.items()}
                 for offset, times in rounds.items()}
        offsets[key], costs = machine_tuning.pick_candidate(
            table, prefer=defaults[key], tolerance=OFFSET_TOLERANCE)
        print(f"  {key}={offsets[key]} (relative cost "
              f"{costs[offsets[key]]:.3f}, default {costs[defaults[key]]:.3f})")
    if not advised_all:
        print("  warning: MADV_HUGEPAGE failed; offsets measured on ordinary pages")

    params = {**tile_params, **rule, **offsets}
    comments = [
        "Measured by find_thresholds.py (step 0); read by the library.",
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"cpu={cpu_model()}",
        f"threads={args.threads} dtype={args.dtype} size={size} "
        f"warmup={args.warmup} trials={args.trials} "
        f"configured_slice_length={configured} huge_pages={int(advised_all)}",
    ]
    params_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = params_path.with_suffix(".txt.tmp")
    tmp.write_text(machine_tuning.format_machine_params(params, comments))
    os.replace(tmp, params_path)
    csv_path = params_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("stage,method,case,candidate,seconds\n")
        for row in csv_rows:
            f.write(",".join(str(v) for v in row[:4]) + f",{row[4]:.8f}\n")
    print(f"  Saved {params_path} and {csv_path}\n")
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find algorithm decision thresholds")
    parser.add_argument("--size", type=int, default=2048,
                        help="Square matrix dimension (default: 2048)")
    parser.add_argument("--step", type=int, default=1,
                        help="Bucket step size for sampling (default: 5)")
    parser.add_argument("--dtype", choices=["float32", "float64"],
                        default="float32")
    parser.add_argument("--slice-length", type=int, default=None,
                        help="Fixed slice length for the threshold runs "
                             "(default: the library's, as at run time: "
                             "tile_sizes.txt and the automatic length)")
    parser.add_argument("--num-buckets", type=int, default=100)
    parser.add_argument("--platform", choices=["cpu"], default="cpu",
                        help="Target platform (default: cpu)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: parameters/<platform>/<type>/)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs before timing (default: 1)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Timed runs, keep best (default: 3)")
    parser.add_argument("--threads", type=int,
                        default=len(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity") else os.cpu_count(),
                        help="CPU threads; use the count the library will run "
                             "with (default: all available CPUs)")
    parser.add_argument("--machine-params", choices=["tune", "use"],
                        default="tune",
                        help="tune: measure and write machine_params.txt "
                             "first (step 0); use: keep the existing file "
                             "(or the defaults without one)")
    parser.add_argument("--machine-params-only", action="store_true",
                        help="Stop after step 0")
    parser.add_argument("--tune-size", type=int, default=6000,
                        help="Matrix dimension for step 0 (default: 6000)")
    parser.add_argument("--early-stop", type=int, default=3,
                        help="Consecutive non-winner tests to stop a row "
                             "(default: 3)")
    parser.add_argument("--log-buckets", action="store_true", default=False,
                        help="Use logarithmic bucket spacing")
    parser.add_argument("--curvature", type=int, default=50,
                        help="Log curvature parameter C (int, default 50 "
                             "=> ~10%% of buckets below 1%% density)")
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    M = K = N = args.size
    BK = args.slice_length
    num_buckets = args.num_buckets
    step = args.step
    platform = args.platform
    es = args.early_stop

    type_name = "float" if dtype == np.float32 else "double"
    if args.machine_params_only and args.machine_params != "tune":
        parser.error("--machine-params-only needs --machine-params tune")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
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

    # -- Load library and build a platform-specific call closure ---------------
    lib = load_so("spalm.so")

    params_path = (SCRIPT_DIR / "parameters" / platform / type_name /
                   "machine_params.txt")
    if args.machine_params == "tune":
        tune_machine_params(lib, args, dtype, platform, type_name, params_path)
        if args.machine_params_only:
            return
    else:
        print(f"Machine parameters: "
              f"{params_path if params_path.exists() else 'defaults (no file)'}")

    def call_fn(*, A_csr, B_csr, C, init_method):
        return call_spalm(lib, A_csr, B_csr, C, init_method,
                          dtype, BK, threads=args.threads)

    print(f"Config: {M}x{K} matrices, platform={platform}, "
          f"BK={BK if BK is not None else 'library default'}, "
          f"{args.dtype}, threads={args.threads}")
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

    print("\nDone!")


if __name__ == "__main__":
    main()
