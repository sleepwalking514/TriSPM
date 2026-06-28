"""
Triton AOT compilation: residual_add kernel -> LLVM IR for RISC-V.

Transformer-facing residual add coverage: out = x + residual.
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


SIZE = env_int("RESIDUAL_ADD_SIZE")
BLOCK_SIZE = env_int("RESIDUAL_ADD_BLOCK_SIZE")
GRID_X = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE


@triton.jit
def residual_add(x_ptr, residual_ptr, out_ptr,
                 n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + residual, mask=mask)


# --- AOT cross-compilation ---
x = torch.empty(SIZE, dtype=torch.float32)
residual = torch.empty(SIZE, dtype=torch.float32)
out = torch.empty(SIZE, dtype=torch.float32)

residual_add[(GRID_X,)](x, residual, out, SIZE, BLOCK_SIZE)
