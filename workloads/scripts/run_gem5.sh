#!/bin/bash
# ============================================================
# Run a compiled kernel binary on the gem5 simulator.
#
# Usage:  ./scripts/run_gem5.sh <kernel_name> [gem5_flags...]
# Example:
#   ./scripts/run_gem5.sh vector_add --cache_baseline
#   ./scripts/run_gem5.sh vector_add --spm
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

KERNEL="${1:?Usage: $0 <kernel_name> [gem5_flags...]}"
shift
GEM5_EXTRA_FLAGS="$*"

BINARY="$TRISPM_ROOT/workloads/$KERNEL/build/${KERNEL}_test"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found: $BINARY" >&2
    echo "       Run './scripts/build_kernel.sh $KERNEL' first." >&2
    exit 1
fi

if [ ! -f "$GEM5" ]; then
    echo "ERROR: gem5 binary not found: $GEM5" >&2
    exit 1
fi

if [ ! -f "$GEM5_RUN_SCRIPT" ]; then
    echo "ERROR: gem5 run script not found: $GEM5_RUN_SCRIPT" >&2
    exit 1
fi

echo "===== Running $KERNEL on gem5 ====="
echo "  binary: $BINARY"
echo "  flags:  ${GEM5_EXTRA_FLAGS:-<none>}"
echo ""

$GEM5 "$GEM5_RUN_SCRIPT" --binary "$BINARY" $GEM5_EXTRA_FLAGS

echo ""
echo "Stats written to m5out/stats.txt"
