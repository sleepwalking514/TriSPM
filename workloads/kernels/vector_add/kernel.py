"""
Triton AOT compilation: vector_add kernel → LLVM IR for RISC-V.

Env vars (set by build_kernel.sh via env.sh before this script runs):
  TRITON_CPU_AOT=1              — master AOT switch (compiler skips host passes)
  KERNEL_AUX_FILE_DIR           — output directory for .llir and launcher files

The Triton runtime writes <kernel_name>.llir and <kernel_name>_launcher.{c,h}
directly to KERNEL_AUX_FILE_DIR.  No manual cache extraction needed.
"""
import os

import torch
import triton
import triton.language as tl

def env_int(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"{name} must be exported from experiment.toml by run_experiment.py")
    return int(value)


SIZE = env_int("SIZE")
BLOCK_SIZE = env_int("BLOCK_SIZE")
GRID_X = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE


@triton.jit
def vector_add(x_ptr, y_ptr, out_ptr,
               n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# --- AOT cross-compilation ---
#
# With TRITON_CPU_AOT=1 (exported by env.sh), the Triton-CPU backend:
#   1. Skips host-specific passes (AMX, AVX512, NEON, …)
#   2. Stops the pipeline at LLVM IR (no .so generation)
#   3. Writes the .llir to KERNEL_AUX_FILE_DIR
#   4. Generates a C launcher (grid dispatch) in the same directory
#   5. Skips kernel execution (returns immediately)
#
# The caller (build_kernel.sh) then runs llc + clang to produce a
# RISC-V binary.

x = torch.empty(SIZE, dtype=torch.float32)
y = torch.empty(SIZE, dtype=torch.float32)
out = torch.empty(SIZE, dtype=torch.float32)

vector_add[(GRID_X,)](x, y, out, SIZE, BLOCK_SIZE)
