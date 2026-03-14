# spalm

Matrix multiplication library with a CPU (Intel MKL) backend.

## Building and running via Docker (`setup.sh`)

All Docker-based commands are in a single script. Run from this directory:

```bash
./setup.sh <command> [args...]
```

### Commands

| Command | Description |
|--------|-------------|
| **build** | Build the Docker image (`spalm-env:latest`) from this directory. Run once before other commands. |
| **run** \<cmd...\> | Run an arbitrary command in the container. Example: `./setup.sh run make baseline spalm` |
| **bash** | Start an interactive bash shell in the container. |
| **build-libs** | Compile all libraries in the container (baseline, spalm). Outputs land in `bin/`. |
| **build-libs clean** | Remove compiled `.so` and object files. |
| **find-thresholds** [args...] | Run `find_thresholds.py` in the container. Writes `decision_thresholds.txt`, `algorithm_map.csv`, and `algorithm_map.png` under `parameters/`. Pass any [find_thresholds.py](find_thresholds.py) options, e.g. `--dtype float32`. |
| **help** | Show usage. |

### Examples

```bash
# One-time: build the image
./setup.sh build

# Run the matrix multiplication example
cd example && ./compile.sh && ./run.sh

# Compile libraries
./setup.sh build-libs

# Clean and rebuild
./setup.sh build-libs clean
./setup.sh build-libs

# Find thresholds (float32)
./setup.sh find-thresholds --dtype float32

# Find thresholds (custom size)
./setup.sh find-thresholds --size 4096

# Run a custom command in the container
./setup.sh run make clean
./setup.sh bash
```
