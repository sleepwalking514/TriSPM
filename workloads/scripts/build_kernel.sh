#!/bin/bash
# ============================================================
# Build a Triton kernel for RISC-V.
#
# Usage:  ./scripts/build_kernel.sh <kernel> --mode {spm,cache} [--tag TAG]
# Example:
#   ./scripts/build_kernel.sh matmul --mode spm
#   ./scripts/build_kernel.sh matmul --mode cache --tag n256-bs32
#
# --mode cache: skip ConvertMemoryToSPM pass, build cache-baseline binary.
# --tag:        artifact label; default "default". Build dir is
#               workloads/build/<kernel>/<mode>-<tag>/ (single source of
#               truth lives in scripts/trispm_paths.py).
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

KERNEL=""
MODE=""
TAG="default"
while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --tag)  TAG="$2";  shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; exit 2 ;;
        *)  if [ -z "$KERNEL" ]; then KERNEL="$1"; shift
            else echo "unexpected arg: $1" >&2; exit 2; fi ;;
    esac
done

[ -n "$KERNEL" ] || { echo "Usage: $0 <kernel> --mode {spm,cache} [--tag TAG]" >&2; exit 2; }
[ "$MODE" = "spm" ] || [ "$MODE" = "cache" ] || { echo "--mode must be spm or cache" >&2; exit 2; }

KERNEL_DIR="$TRISPM_ROOT/workloads/$KERNEL"
[ -d "$KERNEL_DIR" ] || { echo "ERROR: kernel directory not found: $KERNEL_DIR" >&2; exit 1; }

source "$KERNEL_DIR/config.sh"

BUILD_DIR="$(python3 "$SCRIPT_DIR/trispm_paths.py" build_dir "$KERNEL" "$MODE" --tag "$TAG")"

if [ "$MODE" = "cache" ]; then
    export TRITON_DISABLE_SPM=1
    # Triton's compile cache key omits TRITON_DISABLE_SPM, so spm and cache
    # builds collide unless we point them at separate caches.
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_NOSPM:-$HOME/.triton/cache_nospm}"
    echo "(SPM pass disabled — cache-baseline build)"
fi

mkdir -p "$BUILD_DIR"
export KERNEL_AUX_FILE_DIR="$BUILD_DIR"

echo "===== [1/3] Triton kernel → LLVM IR ====="
python3 "$KERNEL_DIR/kernel.py" 2>"$BUILD_DIR/triton_stderr.log"
if [ ! -f "$BUILD_DIR/${KERNEL}.llir" ]; then
    echo "ERROR: Triton did not produce $BUILD_DIR/${KERNEL}.llir" >&2
    echo "       See $BUILD_DIR/triton_stderr.log" >&2
    exit 1
fi
echo "  → $BUILD_DIR/${KERNEL}.llir"

echo "===== [2/3] LLVM IR → RISC-V assembly ====="
$LLC $LLC_FLAGS "$BUILD_DIR/${KERNEL}.llir" -o "$BUILD_DIR/${KERNEL}.s"
echo "  → $BUILD_DIR/${KERNEL}.s"

echo "===== [3/3] Link with harness → RISC-V binary ====="
$CLANG $CLANG_FLAGS \
    $KERNEL_CFLAGS \
    -I"$BUILD_DIR" \
    -I"$TRISPM_ROOT/simulator/src/scratchpad_mem" \
    "$BUILD_DIR/${KERNEL}.s" \
    "$BUILD_DIR/${KERNEL}_launcher.c" \
    "$KERNEL_DIR/harness.c" \
    -lm \
    -o "$BUILD_DIR/${KERNEL}_test"
echo "  → $BUILD_DIR/${KERNEL}_test"

echo ""
echo "Build complete: $BUILD_DIR/${KERNEL}_test"
