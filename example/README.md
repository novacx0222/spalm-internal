# spalm example

Demonstrates sparse matrix multiplication using `spalm.so` on two 8000×8000 matrices:

- **A**: diagonal band pattern (band width = 99)
- **B**: edges pattern (border width = 200)

## Prerequisites

You must build the Docker image from the `spalm/` directory before compiling or running:

```bash
cd ..
./setup.sh build
```

## Compile

From the `example/` directory:

```bash
./compile.sh
```

This builds `spalm.so` and the example binary (`bin/example`) inside the container.

## Run

```bash
./run.sh
```
