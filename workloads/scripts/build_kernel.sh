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
# --tag:        artifact label rendered from the kernel's experiment.toml
#               tag_template (slashes are flattened to '-' in build dirs).
#               Build dir is workloads/build/<kernel>/<mode>-<flat-tag>/
#               (single source of truth lives in scripts/trispm_paths.py).
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

KERNEL=""
MODE=""
TAG=""
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
[ -n "$TAG" ] || { echo "--tag is required (driver renders it from experiment.toml)" >&2; exit 2; }

KERNEL_DIR="$TRISPM_ROOT/workloads/kernels/$KERNEL"
[ -d "$KERNEL_DIR" ] || { echo "ERROR: kernel directory not found: $KERNEL_DIR" >&2; exit 1; }

if [ -z "${KERNEL_CFLAGS:-}" ]; then
    echo "ERROR: KERNEL_CFLAGS is not set; build through scripts/run_experiment.py so experiment.toml params are exported." >&2
    exit 1
fi

BUILD_DIR="$(python3 "$SCRIPT_DIR/trispm_paths.py" build_dir "$KERNEL" "$MODE" --tag "$TAG")"

if [ "$MODE" = "cache" ]; then
    export TRITON_DISABLE_SPM=1
    # Triton's compile cache key omits TRITON_DISABLE_SPM, so spm and cache
    # builds collide unless we point them at separate caches.
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_NOSPM:-$HOME/.triton/cache_nospm}"
    echo "(SPM pass disabled — cache-baseline build)"
else
    # The cache key also omits SPM policy env vars.  Keep the default
    # cache-only reduction policy separate from opt-in reduction SPM builds.
    # KERNEL_TIER_OVERRIDE affects the generated launcher allocation cases, so
    # tier experiments need separate compile caches as well.
    if [ "${TRITON_ENABLE_SPM_REDUCTIONS:-0}" = "1" ]; then
        SPM_CACHE_DIR="${TRITON_CACHE_DIR_SPM_REDUCE:-$HOME/.triton/cache_spm_reduce}"
    else
        SPM_CACHE_DIR="${TRITON_CACHE_DIR_SPM_NOREDUCE:-$HOME/.triton/cache_spm_noreduce}"
    fi
    if [ -n "${KERNEL_TIER_OVERRIDE:-}" ]; then
        TIER_KEY="$(printf '%s' "$KERNEL_TIER_OVERRIDE" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_tier_${TIER_KEY}"
    fi
    if [ "${TRITON_SPM_PROMOTION_REPORT:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_promotion_report"
    fi
    export TRITON_CACHE_DIR="$SPM_CACHE_DIR"
fi

mkdir -p "$BUILD_DIR"
export KERNEL_AUX_FILE_DIR="$BUILD_DIR"
# AOT sidecars are pass side effects, not cached artifacts.  Force compilation
# so every tag gets matching tier/promotion JSON and launcher allocation cases.
export TRITON_ALWAYS_COMPILE=1

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
