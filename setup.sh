#!/bin/bash
# Single entry point for Docker-based build, run, and threshold finding.
# Run from spalm/ only. See README for details.

set -e
SPALM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SPALM_DIR"

IMAGE_NAME="spalm-env:latest"

docker_run() {
    docker run --rm --user "$(id -u):$(id -g)" -v "$SPALM_DIR":/app -w /app "${IMAGE_NAME}" "$@"
}

docker_run_interactive() {
    docker run --rm -it --user "$(id -u):$(id -g)" -v "$SPALM_DIR":/app -w /app "${IMAGE_NAME}" "$@"
}

show_help() {
    cat <<'HELP'
Usage: ./setup.sh <command> [args...]

Docker image:
  build              Build the Docker image (spalm-env:latest) from this directory.
                      Do this once before using other commands.

Running in container:
  run <cmd...>       Run an arbitrary command in the container.
                      Example: ./setup.sh run make baseline spalm
  bash               Start an interactive bash shell in the container.

Build libraries (in container):
    build-libs         Compile all .so files (baseline, spalm).
    build-libs clean   Remove compiled .so and object files.

Threshold finding:
  find-thresholds [args...]   Run find_thresholds.py in the container.
                      Outputs go to parameters/. Pass any find_thresholds.py options.
                      Example: ./setup.sh find-thresholds --dtype float32

Other:
  help               Show this message.
HELP
}

COMMAND="${1:-help}"
shift

case "$COMMAND" in
    build)
        echo "Building image: ${IMAGE_NAME}..."
        docker build -t "${IMAGE_NAME}" .
        ;;
    run)
        docker_run "$@"
        ;;
    bash)
        docker_run_interactive bash
        ;;
    build-libs)
        if [ "$1" = "clean" ]; then
            echo "Cleaning build artifacts..."
            docker_run make clean
            echo "Clean done."
        else
            echo "Building CPU libraries..."
            docker_run make baseline spalm
        fi
        ;;
    find-thresholds)
        docker_run python find_thresholds.py "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
