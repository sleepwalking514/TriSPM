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
# KERNEL_AUX_FILE_DIR is set per-kernel in build_kernel.sh.

# ---------- RISC-V cross-compilation toolchain ----------
# Requires a clang with RISC-V sysroot. Set RISCV_TOOLCHAIN_ROOT to
# override (e.g., export RISCV_TOOLCHAIN_ROOT=/opt/riscv before sourcing).
RISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT:-/llvm_rvv/rvv/riscv}"

# ---------- Tool paths ----------
LLC="$TRISPM_ROOT/compiler/llvm-project/build/bin/llc"
CLANG="${RISCV_TOOLCHAIN_ROOT}/bin/clang"
GEM5="$TRISPM_ROOT/simulator/build/RISCV/gem5.opt"
GEM5_RUN_SCRIPT="$TRISPM_ROOT/simulator/src/scratchpad_mem/run_spm.py"

# ---------- Default RISC-V llc flags ----------
# VLEN=256 bits → 8 x float per vector register.
# -mattr: +m (integer mul/div), +a (atomics), +f (single-float),
#         +d (double-float), +c (compressed insns), +v (vector 1.0).
LLC_FLAGS="-mtriple=riscv64-unknown-linux-gnu \
           -march=riscv64 -mattr=+m,+a,+f,+d,+v \
           -riscv-v-vector-bits-min=256 \
           -riscv-v-vector-bits-max=256 \
           -O3 -filetype=asm"

# Clang flags for harness/launcher C code (not the kernel — that comes from llc).
# -fno-vectorize -fno-slp-vectorize: disable auto-vectorization because gem5's
# O3 CPU has dependency-graph bugs with LMUL>=4 RVV instructions. The Triton
# kernel assembly (from llc) is unaffected by these flags.
CLANG_FLAGS="--target=riscv64-unknown-linux-gnu \
             --sysroot=${RISCV_TOOLCHAIN_ROOT}/sysroot \
             -O2 -static -march=rv64gcv -mabi=lp64d \
             -fno-vectorize -fno-slp-vectorize"
