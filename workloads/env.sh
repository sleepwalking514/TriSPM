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

# ---------- SPM configuration (single source of truth) ----------
# Override these to change SPM geometry for the entire pipeline:
# compiler pass, runtime library, and gem5 simulator.
export TRITON_SPM_BASE="${TRITON_SPM_BASE:-0x40000000}"
export TRITON_SPM_SIZE="${TRITON_SPM_SIZE:-32768}"     # 32 KiB
export SPM_SIZE_BYTES="${SPM_SIZE_BYTES:-$TRITON_SPM_SIZE}"

# ---------- Triton AOT cross-compilation ----------
# TRITON_CPU_AOT is the master switch: it gates the compilation pipeline
# (skip host-specific passes), execution skip, LLIR saving, and launcher
# generation.  All three vars are exported so Python inherits them before
# `import triton` evaluates module-level _AOT_MODE flags.
export TRITON_CPU_AOT=1
# KERNEL_AUX_FILE_DIR is set per-kernel in build_kernel.sh.

# ---------- RISC-V cross-compilation toolchain ----------
# Requires a clang with a RISC-V sysroot. The root must contain bin/clang and
# sysroot/. Keep this explicit so public checkouts do not inherit a local path.
if [ -z "${RISCV_TOOLCHAIN_ROOT:-}" ]; then
    cat >&2 <<'EOF'
error: RISCV_TOOLCHAIN_ROOT is not set.
Set it to a RISC-V cross-compilation toolchain root, for example:
  export RISCV_TOOLCHAIN_ROOT=/opt/riscv
Expected layout:
  $RISCV_TOOLCHAIN_ROOT/bin/clang
  $RISCV_TOOLCHAIN_ROOT/sysroot/
EOF
    return 1 2>/dev/null || exit 1
fi

# ---------- Tool paths ----------
LLC="${LLC:-$TRISPM_ROOT/compiler/llvm-project/build/bin/llc}"
CLANG="${CLANG:-${RISCV_TOOLCHAIN_ROOT}/bin/clang}"
GEM5="${GEM5:-$TRISPM_ROOT/simulator/build/RISCV/gem5.opt}"
GEM5_RUN_SCRIPT="${GEM5_RUN_SCRIPT:-$TRISPM_ROOT/simulator/src/scratchpad_mem/run_spm.py}"

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
