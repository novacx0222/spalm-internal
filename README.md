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
baseline could not be loaded, `4` internal error (e.g. out of memory). Options
are passed as key/value strings:

| Key | Values |
|-----|--------|
| `init_method` | empty (automatic, needs calibrated thresholds), `sparse`, `dense_bm_inner_kernel`, `dense_right_spmm_inner_kernel`, `dense_left_spmm_inner_kernel`, `FORCE_MKL_DENSE`, `FORCE_MKL_SPARSE`, `dense`, `no_fallback`, `no_threshold`, `no_features` |
| `threads` | OpenMP threads (default 4) |
| `prune_zeros` | `1` (default) drops explicit zeros when all values are finite; `0` disables it |
| `compact_keys` | `1` (default) keeps only join keys used by both inputs and renumbers them densely, so every kernel and MKL backend runs on K' <= K; `0` disables it |
| `bucket_method` | `linear_100` (default) or `log_100_cC` |
| `slice_length`, `tile_size_l2`, `tile_size_l3` | override `parameters/cpu/<type>/tile_sizes.txt` |

The four SPALM methods (`sparse`, `dense_*`) use one algorithm for every slice
pair and never fall back to MKL. The two `FORCE_MKL_*` methods skip slicing and
the cost model entirely and need no thresholds file. Automatic mode may fall
back to MKL; one `[backend] ...` line says what ran: `spalm-internal`,
`mkl-dense`/`mkl-sparse` (`forced` or `fallback`), or `zero-work`.

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
| `[sparse-sparse-…]`, `[dense-dense-…]` | kernel stages; `[sparse-sparse-sparse-mm-time]` and `[dense-dense-dense-mm-time]` are the kernel times used by `find_thresholds.py` |
| `[mm-total]` | cost model + multiplication |
| `[slice-mm-total]` | slicing + cost model + multiplication |
| `[forced-backend-total]` | a `FORCE_MKL_*` call; `[mkl-…]` lines break it down |
| `[slice-deallocate]`, `[fallback-slice-deallocate]`, `[thresholds-deallocate]` | releasing slices and decision arrays |
| `[spalm-total]` | the whole `spalm()` call inside the library |

## Tests

Portable tests (no MKL, AVX-512 or built libraries), run from this directory:

```bash
make test CXX=g++-16   # any C++20 compiler with OpenMP; also runs the Python parser tests
```
