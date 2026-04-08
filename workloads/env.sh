#!/bin/bash
# ============================================================
# Shared environment for TriSPM workload pipelines.
# Source this file from any build/run script.
# ============================================================

TRISPM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- Python venv (Triton + PyTorch) ----------
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$TRISPM_ROOT/compiler/.venv/bin/activate" ]; then
    source "$TRISPM_ROOT/compiler/.venv/bin/activate"
fi

# ---------- Triton AOT cross-compilation ----------
# TRITON_CPU_AOT is the master switch: it gates the compilation pipeline
# (skip host-specific passes), execution skip, LLIR saving, and launcher
# generation.  All three vars are exported so Python inherits them before
# `import triton` evaluates module-level _AOT_MODE flags.
export TRITON_CPU_AOT=1
export TRITON_CPU_AOT_FEATURES="+m,+a,+f,+d,+v"
# KERNEL_AUX_FILE_DIR is set per-kernel in build_kernel.sh.

# ---------- Tool paths ----------
LLC="$TRISPM_ROOT/compiler/llvm-project/build/bin/llc"
OPT="$TRISPM_ROOT/compiler/llvm-project/build/bin/opt"
GCC="riscv64-unknown-linux-gnu-gcc"
GEM5="$TRISPM_ROOT/simulator/build/RISCV/gem5.opt"
GEM5_RUN_SCRIPT="$TRISPM_ROOT/simulator/src/scratchpad_mem/run_spm.py"

# ---------- Default RISC-V llc flags ----------
# VLEN=256 bits → 8 x float per vector register.
# LMUL can extend up to 8 (64 floats) automatically.
LLC_FLAGS="-mtriple=riscv64-unknown-linux-gnu \
           -march=riscv64 -mattr=+m,+a,+f,+d,+v \
           -riscv-v-vector-bits-min=256 \
           -riscv-v-vector-bits-max=256 \
           -O3 -filetype=asm"

GCC_FLAGS="-O2 -static -march=rv64gcv -mabi=lp64d"
