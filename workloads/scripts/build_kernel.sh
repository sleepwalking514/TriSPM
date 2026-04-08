#!/bin/bash
# ============================================================
# Build a Triton kernel for RISC-V.
#
# Usage:  ./scripts/build_kernel.sh <kernel_name>
# Example: ./scripts/build_kernel.sh vector_add
#
# Pipeline:
#   1. python kernel.py          → LLVM IR  (.llir) + launcher (.c/.h)
#   2. opt --strip-debug         → strip DWARF metadata
#   3. llc                       → RISC-V assembly (.s)
#   4. riscv64-unknown-linux-gcc → static binary
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

KERNEL="${1:?Usage: $0 <kernel_name>}"
KERNEL_DIR="$TRISPM_ROOT/workloads/$KERNEL"

if [ ! -d "$KERNEL_DIR" ]; then
    echo "ERROR: kernel directory not found: $KERNEL_DIR" >&2
    exit 1
fi

# Source per-kernel config
source "$KERNEL_DIR/config.sh"

BUILD_DIR="$KERNEL_DIR/build"
mkdir -p "$BUILD_DIR"

# Tell the Triton AOT pipeline where to write artifacts (LLIR + launcher).
# env.sh already exports TRITON_CPU_AOT=1 and TRITON_CPU_AOT_FEATURES,
# so the compiler and driver pick them up at import time.
export KERNEL_AUX_FILE_DIR="$BUILD_DIR"

# ---- Step 1: Triton → LLVM IR + launcher ----
echo "===== [1/4] Triton kernel → LLVM IR ====="
python3 "$KERNEL_DIR/kernel.py" 2>"$BUILD_DIR/triton_stderr.log"
if [ ! -f "$BUILD_DIR/${KERNEL}.llir" ]; then
    echo "ERROR: Triton did not produce $BUILD_DIR/${KERNEL}.llir" >&2
    echo "       Check $BUILD_DIR/triton_stderr.log for details." >&2
    exit 1
fi
echo "  → $BUILD_DIR/${KERNEL}.llir"

# ---- Step 2: Strip debug metadata ----
# Triton emits DWARF 5 directives that older cross-toolchain assemblers
# cannot parse.  Strip them — debug info is not needed for gem5 simulation.
echo "===== [2/4] Strip debug metadata ====="
$OPT --strip-debug "$BUILD_DIR/${KERNEL}.llir" -S -o "$BUILD_DIR/${KERNEL}.llir"
echo "  done"

# ---- Step 3: LLVM IR → RISC-V assembly ----
echo "===== [3/4] LLVM IR → RISC-V assembly ====="
$LLC $LLC_FLAGS "$BUILD_DIR/${KERNEL}.llir" -o "$BUILD_DIR/${KERNEL}.s"
echo "  → $BUILD_DIR/${KERNEL}.s"

# ---- Step 4: Cross-compile harness + launcher + kernel → binary ----
echo "===== [4/4] Link with harness → RISC-V binary ====="
$GCC $GCC_FLAGS \
    $KERNEL_CFLAGS \
    -I"$BUILD_DIR" \
    "$BUILD_DIR/${KERNEL}.s" \
    "$BUILD_DIR/${KERNEL}_launcher.c" \
    "$KERNEL_DIR/harness.c" \
    -lm \
    -o "$BUILD_DIR/${KERNEL}_test"
echo "  → $BUILD_DIR/${KERNEL}_test"

echo ""
echo "Build complete: $BUILD_DIR/${KERNEL}_test"
