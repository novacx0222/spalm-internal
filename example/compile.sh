#!/bin/bash
# compile.sh
# Builds spalm.so and the example binary inside the Docker container.
# Must be run from the spalm/ directory (or anywhere — script locates itself).

set -e

SPALM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="spalm-env:latest"

echo "==> spalm dir : $SPALM_DIR"
echo "==> image     : $IMAGE_NAME"
echo ""

if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "==> Image not found. Building from $SPALM_DIR/Dockerfile..."
    docker build -t "$IMAGE_NAME" "$SPALM_DIR"
fi

echo "==> Building spalm libs and example inside container..."
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$SPALM_DIR":/app \
    -w /app \
    "$IMAGE_NAME" \
    bash -c "
        set -e
        echo '--- spalm libs ---'
        make baseline spalm

        echo ''
        echo '--- example binary ---'
        cd /app/example
        make example
    "

echo ""
echo "==> Done. Binary at: spalm/example/bin/example"
