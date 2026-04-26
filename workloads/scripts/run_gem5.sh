#!/bin/bash
# ============================================================
# Run a compiled kernel binary on the gem5 simulator.
#
# Usage:  ./scripts/run_gem5.sh <kernel_name> [--no-spm] [gem5_flags...]
# Example:
#   ./scripts/run_gem5.sh matmul --no-spm --cache_baseline   # cache baseline
#   ./scripts/run_gem5.sh matmul                             # SPM mode
#
# --no-spm: pick the cache-baseline binary built in build_nospm/.
#           This flag is consumed locally and not forwarded to gem5.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

KERNEL="${1:?Usage: $0 <kernel_name> [--no-spm] [gem5_flags...]}"
shift

NO_SPM=0
GEM5_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-spm) NO_SPM=1 ;;
        *)        GEM5_ARGS+=("$arg") ;;
    esac
done

if [ "$NO_SPM" = "1" ]; then
    BINARY="$TRISPM_ROOT/workloads/$KERNEL/build_nospm/${KERNEL}_test"
    BUILD_HINT="./scripts/build_kernel.sh $KERNEL --no-spm"
    MODE="cache"
else
    BINARY="$TRISPM_ROOT/workloads/$KERNEL/build/${KERNEL}_test"
    BUILD_HINT="./scripts/build_kernel.sh $KERNEL"
    MODE="spm"
fi

if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found: $BINARY" >&2
    echo "       Run '$BUILD_HINT' first." >&2
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
echo "  flags:  ${GEM5_ARGS[*]:-<none>}"
echo ""

M5OUT_DIR="$TRISPM_ROOT/workloads/m5out"
mkdir -p "$M5OUT_DIR"

$GEM5 --outdir="$M5OUT_DIR" "$GEM5_RUN_SCRIPT" --binary "$BINARY" "${GEM5_ARGS[@]}"

NAMED_STATS="$M5OUT_DIR/${MODE}-${KERNEL}-stats.txt"
# Keep the named artifact focused on the explicit kernel ROI dump. gem5 also
# appends a final dump at process exit, which includes post-kernel harness work.
awk '
    /---------- Begin Simulation Statistics ----------/ { in_block = 1 }
    in_block { print }
    /---------- End Simulation Statistics/ && in_block { exit }
' "$M5OUT_DIR/stats.txt" > "$NAMED_STATS"

echo ""
echo "Stats written to $NAMED_STATS"
