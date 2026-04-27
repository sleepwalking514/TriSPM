#!/bin/bash
# ============================================================
# Build a Triton kernel for RISC-V.
#
# Usage:  ./scripts/build_kernel.sh <kernel_name> [--no-spm]
# Example: ./scripts/build_kernel.sh vector_add
#          ./scripts/build_kernel.sh matmul --no-spm
#
# --no-spm: skip the ConvertMemoryToSPM pass (no DMA MMIO writes,
#           no addrspace(3) loads).  Output goes to build_nospm/
#           so it doesn't clobber the SPM build.  Required for the
#           cache-baseline gem5 run.
#
# Pipeline:
#   1. python kernel.py  → LLVM IR (.llir) + launcher (.c/.h)
#   2. llc               → RISC-V assembly (.s)
#   3. clang             → static RISC-V binary
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

NO_SPM=0
POS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-spm) NO_SPM=1 ;;
        *)        POS_ARGS+=("$arg") ;;
    esac
done

KERNEL="${POS_ARGS[0]:?Usage: $0 <kernel_name> [--no-spm]}"
KERNEL_DIR="$TRISPM_ROOT/workloads/$KERNEL"

if [ ! -d "$KERNEL_DIR" ]; then
    echo "ERROR: kernel directory not found: $KERNEL_DIR" >&2
    exit 1
fi

# Source per-kernel config
source "$KERNEL_DIR/config.sh"

BUILD_SUFFIX="${KERNEL_BUILD_SUFFIX:-}"

if [ "$NO_SPM" = "1" ]; then
    BUILD_DIR="$KERNEL_DIR/build_nospm${BUILD_SUFFIX}"
    export TRITON_DISABLE_SPM=1
    # Triton's compile cache key is derived from CPUOptions, which does NOT
    # include TRITON_DISABLE_SPM.  Without a separate cache directory the
    # SPM-enabled and SPM-disabled builds collide and return whichever LLIR
    # was cached first.
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_NOSPM:-$HOME/.triton/cache_nospm}"
    echo "(SPM pass disabled — building cache-baseline binary)"
else
    BUILD_DIR="$KERNEL_DIR/build${BUILD_SUFFIX}"
fi
mkdir -p "$BUILD_DIR"

# Tell the Triton AOT pipeline where to write artifacts (LLIR + launcher).
# env.sh already exports TRITON_CPU_AOT=1, so the compiler and driver 
# pick them up at import time.
export KERNEL_AUX_FILE_DIR="$BUILD_DIR"

# ---- Step 1: Triton → LLVM IR + launcher ----
echo "===== [1/3] Triton kernel → LLVM IR ====="
python3 "$KERNEL_DIR/kernel.py" 2>"$BUILD_DIR/triton_stderr.log"
if [ ! -f "$BUILD_DIR/${KERNEL}.llir" ]; then
    echo "ERROR: Triton did not produce $BUILD_DIR/${KERNEL}.llir" >&2
    echo "       Check $BUILD_DIR/triton_stderr.log for details." >&2
    exit 1
fi
echo "  → $BUILD_DIR/${KERNEL}.llir"

# ---- Step 2: LLVM IR → RISC-V assembly ----
echo "===== [2/3] LLVM IR → RISC-V assembly ====="
$LLC $LLC_FLAGS "$BUILD_DIR/${KERNEL}.llir" -o "$BUILD_DIR/${KERNEL}.s"
echo "  → $BUILD_DIR/${KERNEL}.s"

# ---- Step 3: Cross-compile harness + launcher + kernel → binary ----
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
