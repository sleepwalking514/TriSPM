#!/bin/bash
# Build + run the dma_l2_warming microbenchmark.
#
# Usage:
#   ./build_and_run.sh [--buf-bytes N] [--build-only] [--sweep]
#
# --buf-bytes N   Set BUF_BYTES (default 4096)
# --build-only    Compile but don't run gem5
# --sweep         Run the working-set sweep (4K, 8K, 16K, 32K)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../../env.sh"

BUF_BYTES=4096
BUILD_ONLY=0
SWEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --buf-bytes)  BUF_BYTES="$2"; shift 2 ;;
        --build-only) BUILD_ONLY=1; shift ;;
        --sweep)      SWEEP=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

BUILD_DIR="$TRISPM_ROOT/workloads/build/dma_l2_warming/buf${BUF_BYTES}"
M5OUT_DIR="$TRISPM_ROOT/workloads/m5out/dma_l2_warming/buf${BUF_BYTES}"

build_one() {
    local buf=$1
    local bdir="$TRISPM_ROOT/workloads/build/dma_l2_warming/buf${buf}"
    mkdir -p "$bdir"

    echo "===== Building dma_l2_warming (BUF_BYTES=$buf) ====="
    $CLANG $CLANG_FLAGS \
        -DBUF_BYTES="$buf" \
        -I"$TRISPM_ROOT/simulator/src/scratchpad_mem" \
        "$SCRIPT_DIR/harness.c" \
        -lm \
        -o "$bdir/dma_l2_warming_test"
    echo "  → $bdir/dma_l2_warming_test"
}

run_one() {
    local buf=$1
    local bdir="$TRISPM_ROOT/workloads/build/dma_l2_warming/buf${buf}"
    local mdir="$TRISPM_ROOT/workloads/m5out/dma_l2_warming/buf${buf}"
    mkdir -p "$mdir"

    echo "===== Running dma_l2_warming (BUF_BYTES=$buf) on gem5 ====="
    $GEM5 --outdir="$mdir" \
        "$GEM5_RUN_SCRIPT" \
        --binary "$bdir/dma_l2_warming_test"
    echo "  → stats: $mdir/stats.txt"
}

if [ "$SWEEP" -eq 1 ]; then
    for sz in 4096 8192 16384 32768; do
        build_one "$sz"
        if [ "$BUILD_ONLY" -eq 0 ]; then
            run_one "$sz"
        fi
    done
else
    build_one "$BUF_BYTES"
    if [ "$BUILD_ONLY" -eq 0 ]; then
        run_one "$BUF_BYTES"
    fi
fi

echo ""
echo "Done."
