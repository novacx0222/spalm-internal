#!/bin/bash
# run.sh
# Runs the example binary inside the Docker container.
# Run compile.sh first.

set -e

SPALM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="spalm-env:latest"

if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Image '$IMAGE_NAME' not found. Run compile.sh first."
    exit 1
fi

if [ ! -f "$SPALM_DIR/example/bin/example" ]; then
    echo "Binary not found. Run compile.sh first."
    exit 1
fi

echo "==> Running example..."
echo ""
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$SPALM_DIR":/app \
    -w /app \
    -e SPALM_LIB=/app/bin/spalm.so \
    "$IMAGE_NAME" \
    /app/example/bin/example
