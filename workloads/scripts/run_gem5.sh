#!/bin/bash
# ============================================================
# Run a compiled kernel on gem5.
#
# Usage:  ./scripts/run_gem5.sh <kernel> --mode {spm,cache} [--tag TAG] [gem5_flags...]
# Example:
#   ./scripts/run_gem5.sh matmul --mode cache --cache_baseline
#   ./scripts/run_gem5.sh matmul --mode spm --tag n256-bs32
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

KERNEL=""
MODE=""
TAG="default"
GEM5_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --tag)  TAG="$2";  shift 2 ;;
        --) shift; GEM5_ARGS+=("$@"); break ;;
        *)  if [ -z "$KERNEL" ]; then KERNEL="$1"; shift
            else GEM5_ARGS+=("$1"); shift; fi ;;
    esac
done

[ -n "$KERNEL" ] || { echo "Usage: $0 <kernel> --mode {spm,cache} [--tag TAG] [gem5_flags...]" >&2; exit 2; }
[ "$MODE" = "spm" ] || [ "$MODE" = "cache" ] || { echo "--mode must be spm or cache" >&2; exit 2; }

BINARY="$(python3 "$SCRIPT_DIR/trispm_paths.py" binary "$KERNEL" "$MODE" --tag "$TAG")"
M5OUT_DIR="$(python3 "$SCRIPT_DIR/trispm_paths.py" m5out_dir "$KERNEL" "$MODE" --tag "$TAG")"
ROI_STATS="$(python3 "$SCRIPT_DIR/trispm_paths.py" roi_stats "$KERNEL" "$MODE" --tag "$TAG")"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found: $BINARY" >&2
    echo "       Run './scripts/build_kernel.sh $KERNEL --mode $MODE --tag $TAG' first." >&2
    exit 1
fi
[ -f "$GEM5" ] || { echo "ERROR: gem5 binary not found: $GEM5" >&2; exit 1; }
[ -f "$GEM5_RUN_SCRIPT" ] || { echo "ERROR: gem5 run script not found: $GEM5_RUN_SCRIPT" >&2; exit 1; }

mkdir -p "$M5OUT_DIR"
echo "===== Running $KERNEL ($MODE, tag=$TAG) on gem5 ====="
echo "  binary: $BINARY"
echo "  outdir: $M5OUT_DIR"
echo "  flags:  ${GEM5_ARGS[*]:-<none>}"
echo ""

$GEM5 --outdir="$M5OUT_DIR" "$GEM5_RUN_SCRIPT" --binary "$BINARY" "${GEM5_ARGS[@]}"

# gem5 dumps stats both at the explicit ROI and again at exit. Keep only
# the explicit ROI dump for downstream comparison.
awk '
    /---------- Begin Simulation Statistics ----------/ { in_block = 1 }
    in_block { print }
    /---------- End Simulation Statistics/ && in_block { exit }
' "$M5OUT_DIR/stats.txt" > "$ROI_STATS"

echo ""
echo "ROI stats written to $ROI_STATS"
