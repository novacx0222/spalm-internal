# SPALM

This repository contains code used in the paper 'SPALM: A Sparsity-Pattern-Adaptive Library for Matrices'.

The DuckDB extension that runs SPALM as a query operator is
[novacx0222/duckdb-spalm](https://github.com/novacx0222/duckdb-spalm).

## Acknowledgments

This repository is a fork of
[junyoungkim22/spalm-release](https://github.com/junyoungkim22/spalm-release),
the original SPALM implementation by Junyoung Kim, accompanying:

> Junyoung Kim and Kenneth A. Ross. 2026. SPALM: A Sparsity-Pattern-Adaptive
> Library for Matrices. *Proc. ACM Manag. Data* 4, 3 (SIGMOD), Article 236.
> https://doi.org/10.1145/3802113

## Building and running via Docker (`setup.sh`)

All Docker-based commands are in a single script. Run from this directory:

```bash
./setup.sh <command> [args...]
```

### Commands

| Command                       | Description                                                                                                                                                                                                                       |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **build**                     | Build the Docker image (`spalm-env:latest`) from this directory. Run once before other commands.                                                                                                                                  |
| **run** \<cmd...\>            | Run an arbitrary command in the container. Example: `./setup.sh run make baseline spalm`                                                                                                                                          |
| **bash**                      | Start an interactive bash shell in the container.                                                                                                                                                                                 |
| **build-libs**                | Compile all libraries in the container (baseline, spalm). Outputs land in `bin/`.                                                                                                                                                 |
| **build-libs clean**          | Remove compiled `.so` and object files.                                                                                                                                                                                           |
| **build-openblas**            | Download OpenBLAS 0.3.30 (sha256-checked), apply the optional huge-page patch and build it for `FORCE_OPENBLAS_DENSE` into `OpenBLAS-0.3.30/libopenblas.so` (see [OpenBLAS baseline](#openblas-baseline)). |
| **find-thresholds** [args...] | Run `find_thresholds.py` in the container. Writes `decision_thresholds.txt`, `algorithm_map.csv`, and `algorithm_map.png` under `parameters/`. Pass any [find_thresholds.py](find_thresholds.py) options, e.g. `--dtype float32`. |
| **help**                      | Show usage.                                                                                                                                                                                                                       |

### Examples

```bash
# One-time: build the image
./setup.sh build

# Compile libraries
./setup.sh build-libs

# Clean and rebuild
./setup.sh build-libs clean
./setup.sh build-libs

# Find thresholds (float32)
./setup.sh find-thresholds --dtype float32

# Find thresholds (custom size)
./setup.sh find-thresholds --size 4096

# Run the matrix multiplication example
cd example && ./compile.sh && ./run.sh
```

## Calling `spalm()`

`spalm()` returns `0` on success; the output is valid only then. Other values:
`1` invalid argument (including unsorted or out-of-range CSR column
indices), `2` missing or invalid parameter files, `3` the MKL
or OpenBLAS baseline could not be loaded, `4` internal error (e.g. out of memory). Options
are passed as key/value strings:

| Key | Values |
|-----|--------|
| `init_method` | empty (automatic, needs calibrated thresholds), `sparse`, `dense_bm_inner_kernel`, `dense_right_spmm_inner_kernel`, `dense_left_spmm_inner_kernel`, `FORCE_MKL_DENSE`, `FORCE_MKL_SPARSE`, `FORCE_OPENBLAS_DENSE`, `dense`, `no_fallback`, `no_threshold`, `no_features` |
| `threads` | OpenMP threads (default 4) |
| `prune_zeros` | `1` (default) drops explicit zeros when all values are finite; `0` disables it |
| `compact_keys` | `1` (default) keeps only join keys used by both inputs and renumbers them densely, so every kernel and MKL backend runs on K' <= K; `0` disables it |
| `bucket_method` | `linear_100` (default) or `log_100_cC` |
| `slice_length`, `tile_size_l2`, `tile_size_l3` | override `parameters/cpu/<type>/tile_sizes.txt` |
| `auto_slice_length` | for SD, DS and DD only, when `slice_length` is not given: `1` (default) lengthens the configured slice length up to 4x, at most to K, from exact counts of subrows and packed slices (`include/slice_length_rule.h`), and only for inputs whose modelled cost reaches `kMinCost` (below it the length is kept, and inputs too small to reach it are not counted); `0` keeps it; `2` is the earlier rule from global density alone, for comparison. One `[slice-length]` line gives the rule, the counts and the choice |

Tunables. Each defaults to the constant it replaces, so leaving it out keeps
the behaviour unchanged. Values must be numbers (integers where noted; `5e8`
counts as one); anything else, or a value out of range, returns `1`.

| Key | Default | Meaning |
|-----|---------|---------|
| `thresholds_file` | empty | decision table read by the automatic modes instead of `parameters/cpu/<type>/<buckets>/decision_thresholds.txt` |
| `dense_mm_threshold` | `-1` (from the file) | integer >= 0: replaces the file's value (dense work below it: every slice pair on SS). The `dense` and `no_threshold` variants still set it to 0 |
| `auto_sparse_ratio` | `0.1` | every slice pair goes dense when the SS cost is below this fraction of the dense cost (compared in float) |
| `cost_weight_ss`, `cost_weight_sd`, `cost_weight_ds`, `cost_weight_dd` | `1` | >= 0: each kernel's cost is multiplied by its weight before the two rules above (cost units are not time). With a weight other than 1, a `[weighted-cost]` line gives the weighted costs |
| `auto_mkl_dense_ratio` | `0.8` | with fallback: MKL dense when the (unweighted) dense cost exceeds this fraction of M·K·N |
| `auto_mkl_sparse_fallback` | `1` | with fallback: `1` runs MKL sparse when only SS slice pairs remain, `0` keeps them on SPALM SS |
| `asl_min_cost` | `3e9` | `auto_slice_length = 1`: smallest modelled cost that is lengthened at all (`kMinCost`) |
| `asl_subrow_write_entries` | `16` | entries one subrow write costs, and the subrow length from which nothing is lengthened (`kSubrowWriteEntries`) |
| `asl_min_cost_ratio` | `0.9` | a doubling must bring the modelled cost to at most this fraction (`kMinCostRatio`) |
| `asl_max_levels` | `3` | integer 1 to 3: lengths tried, up to 1x, 2x or 4x the configured one |
| `r_block_huge_bytes` | `8388608` | integer: SD/DS/DD pack R into a huge-page mapping from this size on (never below 2 MiB); `-1` never |
| `s_blocks_huge_bytes` | `2097152` | integer: the S packing blocks of all workers share one huge-page mapping from this total on (never below 2 MiB); `-1` never |
| `ss_column_block_min_cols` | `256` | integer >= 1: fewest output columns per SS column block (automatic or explicit `ss_column_blocks`) |
| `ss_column_block_cache_divisor` | `3` | integer >= 1: automatic SS column blocks size each block's part of S^T to the shared cache divided by this |
| `openblas_huge_pages` | `0` | `FORCE_OPENBLAS_DENSE` only, `0` or `1`: `1` puts OpenBLAS's GEMM buffers on 2 MiB transparent huge pages, as MKL's are; needs the patched build of `build-openblas`, and holds for the whole process once OpenBLAS is loaded |
| `log_candidate_work` | `0` | `1`: diagnostics for comparing whole-call alternatives offline (`include/candidate_work.h`), printed after the MM cost lines; decisions and results do not change. `[candidate-features]`: M, N, K after key compaction, slice length and heights, stored entries per side, `flops` (multiply-adds of the product: the sum over keys of R entries x S entries), `row_bound` (the sum over R rows of min(N, the row's multiply-adds): an upper bound on result entries), slices per side, and slice pairs that share a column. `[candidate-work] <table> bucket\|nnz ss= sd= ds= dd=`: the cost model for the decision table as loaded (`table`, before the whole-call rules), with every pair SS (`all_ss`), and after `set_to_all_dense()` (`all_dense`). `bucket` is the cost model itself (so `table bucket` repeats the MM cost lines); `nnz` uses each density bucket's stored entries instead of its representative count. `[candidate-entries] budget= estimate= exact= draws= rows= long_rows= visits=`: the result entries, twice, with budgets of 2^27 and 2^20 S entries read. Counted exactly when `flops` is at most the budget; otherwise estimated from R rows drawn in proportion to their row bound (standard error at most `row_bound` / (2 sqrt(draws))). A drawn row with more than 8 N multiply-adds (`long_rows`) is estimated from 64 of its result columns, each tested for a key in common with the row. `[candidate-pairs] <table> ss= sd= ds= dd= pack_r= pack_s= bitmaps= r_stride= fixed_alg=`: the slice pairs per kernel, and what the dense path would pack under that table. `[candidate-dense-schedule]`: the R slices the dense path packs (each once) and the S slices (once per R task), with their stored entries, and its (R task, S tile) tasks |

Every call prints one `[conf] ...` line with the effective settings (after
the thresholds file, the `init_method` variant and the tile sizes), and a
`[conf-unused] keys=...` line naming any key neither the library nor the
DuckDB operator reads, so a misspelled key shows in the log.

The four SPALM methods (`sparse`, `dense_*`) use one algorithm for every slice
pair and never fall back to MKL. The forced baselines (`FORCE_MKL_*`,
`FORCE_OPENBLAS_DENSE`) skip slicing and the cost model entirely and need no
thresholds file. Automatic mode may fall back to MKL (never to OpenBLAS); one
`[backend] ...` line says what ran: `spalm-internal`, `mkl-dense`/`mkl-sparse`
(`forced` or `fallback`), `openblas-dense (forced)`, or `zero-work`.

### OpenBLAS baseline

`FORCE_OPENBLAS_DENSE` is MKL dense with OpenBLAS's GEMM: the same CSR
preparation, the same dense inputs on huge pages filled by the same threads
(`[mkl-dense-input-*]` lines for both), then `cblas_?gemm` with the same
arguments. OpenBLAS has no sparse BLAS, so there is no OpenBLAS sparse
method. `run_baseline.so` opens `OpenBLAS-0.3.30/libopenblas.so` (next to
`bin/`) with `RTLD_LOCAL` and calls it through looked-up pointers, since MKL
exports the same names (`include/openblas_baseline.h`).

`./setup.sh build-openblas` (`scripts/build_openblas.sh`) builds the pinned
0.3.30 release with `USE_OPENMP=1` (its GEMM runs on the process's libgomp
team, like MKL's gnu_thread layer; pthreads builds are rejected at load),
`TARGET=SAPPHIRERAPIDS`, `NUM_THREADS=64` and `NO_AFFINITY=1`, and writes
`OpenBLAS-0.3.30/BUILD_INFO.txt`. Each call sets `openblas_set_num_threads`
and the OpenMP thread count to `threads`, and prints `[openblas-config]
parallel= corename= threads= omp_max_threads= huge_pages= thp_buffers=
path= config=`.

OpenBLAS maps a 128 MiB GEMM buffer per thread when it is loaded, on 4 KiB
pages. `scripts/openblas_thp_patch.py` patches its allocator: with
`openblas_huge_pages = 1`, `run_baseline.so` sets `SPALM_OPENBLAS_THP=1`
before loading it, and the buffers are 2 MiB aligned and marked
`MADV_HUGEPAGE` (`thp_buffers` counts them). The default `0` maps them as
the stock library does.

### Sparse output

`spalm_sparse()` returns C as row blocks in CSR form
(`include/spalm_sparse_result.h`) instead of filling a dense result, for
`init_method` `sparse` and `FORCE_MKL_SPARSE` only.

`spalm_sparse_or_dense()` takes any `init_method` (e.g. `no_fallback`) and a
callback, `spalm_dense_result_fn`, that returns the caller's dense M x N
result. After the dispatch decision:

- **Row blocks:** the call runs entirely on the SS path (every slice pair SS,
  no MKL fallback) or on `FORCE_MKL_SPARSE`. The callback is not called.
- **Dense result:** any other decision (mixed, all dense, an MKL fallback,
  the dense kernels, `FORCE_MKL_DENSE`) calls the callback once, writes the
  dense result there, and returns no handle.

A `[result-choice] sparse|dense (<why>)` line says which, and
`[dense-result-request]` times the callback. A callback that returns null
fails the call with `4`.

## Timing output

Each stage prints `[stage-name] <milliseconds> ms` (six decimals). Nested
stages overlap, so only totals at the same level may be compared or added.

| Label | Covers |
|-------|--------|
| `[library-initialize]` | argument checks, options, thresholds and tile sizes |
| `[csr-zero-prepare]` | all CSR preparation: zero pruning, key compaction, the compacted copy (`[csr-zero-stats]` gives entries before and after) |
| `[csr-key-domain]` | marking the keys of both inputs and ranking the common ones (`[csr-key-stats] K=<K>-><K'> <result>`) |
| `[1-slice-…]`, `[2-slice-…]` | building the slice structures of R (1) and S (2): `subrow-csr`, `subrow-csc`, `csc` (plus `-cumulative`) |
| `[slice-prepare-total]` | both slice structures |
| `[mm-select-total]` | cost model over all slice columns |
| `[candidate-entries-time]`, `[candidate-entries-small-time]` | `log_candidate_work = 1` only: the `[candidate-entries]` counts with budgets 2^27 and 2^20; inside `[candidate-work-time]` |
| `[candidate-work-time]` | `log_candidate_work = 1` only: all the `[candidate-*]` counts; inside `[mm-total]` |
| `[sparse-sparse-…]`, `[dense-dense-…]` | kernel stages; `[sparse-sparse-sparse-mm-time]` and `[dense-dense-dense-mm-time]` are the kernel times used by `find_thresholds.py` |
| `[mm-total]` | cost model + multiplication |
| `[slice-mm-total]` | slicing + cost model + multiplication |
| `[forced-backend-total]` | a `FORCE_MKL_*` or `FORCE_OPENBLAS_DENSE` call; `[mkl-…]` and `[openblas-…]` lines break it down |
| `[openblas-load]` | `FORCE_OPENBLAS_DENSE`, first call in the process: loading libopenblas (its constructor allocates the thread buffers); inside `[openblas-baseline-total]` |
| `[openblas-dense-mm-time]`, `[openblas-baseline-total]`, `[openblas-call-total]` | the GEMM; the whole OpenBLAS call in `run_baseline.so` (load included); the same seen from `spalm.so` (MKL dense: `[mkl-dense-mm-time]`, `[mkl-baseline-total]`, `[mkl-call-total]`) |
| `[slice-deallocate]`, `[fallback-slice-deallocate]`, `[thresholds-deallocate]` | releasing slices and decision arrays |
| `[spalm-total]` | the whole `spalm()` call inside the library |

## Tests

Portable tests (no MKL, AVX-512 or built libraries), run from this directory:

```bash
make test CXX=g++-16   # any C++20 compiler with OpenMP; also runs the Python parser tests
```
