#!/bin/bash
# Builds the OpenBLAS baseline library (init_method FORCE_OPENBLAS_DENSE):
# OpenBLAS 0.3.30, OpenMP (libgomp), Sapphire Rapids kernels, with the
# optional huge-page patch (scripts/openblas_thp_patch.py; off unless
# spalm.conf has openblas_huge_pages = 1). Run in the spalm-env container
# from the repository root: ./setup.sh build-openblas.
#
# Output: OpenBLAS-0.3.30/libopenblas.so (what run_baseline.so loads) and
# OpenBLAS-0.3.30/BUILD_INFO.txt. Source and build tree:
# OpenBLAS-0.3.30-build/. Both are ignored by git.
set -euo pipefail

VERSION=0.3.30
URL="https://github.com/OpenMathLib/OpenBLAS/releases/download/v${VERSION}/OpenBLAS-${VERSION}.tar.gz"
# The release asset's sha256 (GitHub releases API, digest field).
SHA256=27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d
# Fixed target instead of DYNAMIC_ARCH: the experiments run on a Xeon 8488C
# (Sapphire Rapids); [openblas-config] corename shows what was built.
TARGET="${OPENBLAS_TARGET:-SAPPHIRERAPIDS}"
JOBS="${JOBS:-$(nproc)}"

ROOT="$(pwd)"
BUILD="$ROOT/OpenBLAS-${VERSION}-build"
OUT="$ROOT/OpenBLAS-${VERSION}"
TARBALL="$BUILD/OpenBLAS-${VERSION}.tar.gz"
SRC="$BUILD/src"

mkdir -p "$BUILD"
if [ ! -f "$TARBALL" ]; then
    wget -q -O "$TARBALL.part" "$URL"
    mv "$TARBALL.part" "$TARBALL"
fi
echo "${SHA256}  ${TARBALL}" | sha256sum -c -

rm -rf "$SRC"
mkdir -p "$SRC"
tar -xzf "$TARBALL" -C "$SRC" --strip-components=1
python3 "$ROOT/scripts/openblas_thp_patch.py" "$SRC"

# USE_OPENMP=1: GEMM runs on the process's libgomp team, like MKL's
#   gnu_thread layer and the SPALM kernels (a pthreads build would add a
#   busy-waiting pool of its own).
# NUM_THREADS=64: the most threads it supports (the 8488C has 64 CPUs);
#   run_baseline.so sets the number used, 30 in the experiments.
# NO_AFFINITY=1: no thread binding, as nothing is bound in v35.
# NO_LAPACK, NOFORTRAN: only BLAS is needed.
# USE_TLS stays unset (the allocator the patch expects).
OPTIONS=(TARGET="$TARGET" DYNAMIC_ARCH=0 BINARY=64 USE_OPENMP=1 USE_THREAD=1
         NUM_THREADS=64 NO_AFFINITY=1 NO_LAPACK=1 NOFORTRAN=1 CC=gcc)
make -C "$SRC" -j"$JOBS" "${OPTIONS[@]}" libs shared > "$BUILD/make.log" 2>&1 \
    || { tail -n 40 "$BUILD/make.log"; exit 1; }

mkdir -p "$OUT"
cp -L "$SRC/libopenblas.so" "$OUT/libopenblas.so.tmp"
mv "$OUT/libopenblas.so.tmp" "$OUT/libopenblas.so"

# The library must use libgomp (not libiomp5) and carry the patch. The
# outputs are captured first: with pipefail, "nm | grep -q" fails when grep
# stops at its first match and nm then dies of SIGPIPE.
LDD="$(ldd "$OUT/libopenblas.so")"
grep -q 'libgomp.so' <<< "$LDD" || { echo "libopenblas.so does not load libgomp:"; echo "$LDD"; exit 1; }
if grep -q 'libiomp5' <<< "$LDD"; then echo "libopenblas.so loads libiomp5"; exit 1; fi
SYMBOLS="$(nm -D --defined-only "$OUT/libopenblas.so")"
grep -q ' openblas_spalm_thp_buffers$' <<< "$SYMBOLS" \
    || { echo "libopenblas.so lacks openblas_spalm_thp_buffers (patch not applied?)"; exit 1; }

{
    echo "version: $VERSION"
    echo "url: $URL"
    echo "tarball_sha256: $SHA256"
    echo "patch: scripts/openblas_thp_patch.py (SPALM_OPENBLAS_THP, off unless openblas_huge_pages = 1)"
    echo "options: ${OPTIONS[*]}"
    echo "gcc: $(gcc --version | head -n 1)"
    echo "library_sha256: $(sha256sum "$OUT/libopenblas.so" | cut -d' ' -f1)"
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "ldd:"
    echo "$LDD"
} > "$OUT/BUILD_INFO.txt"
cat "$OUT/BUILD_INFO.txt"
