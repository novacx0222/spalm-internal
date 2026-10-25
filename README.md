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
