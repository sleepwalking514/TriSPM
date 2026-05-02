"""
Triton AOT compilation: activation kernel -> LLVM IR for RISC-V.

Transformer-facing elementwise activation coverage.  The first activation is
SiLU because it is common in modern decoder blocks and exercises tl.exp without
introducing another producer/consumer dependency.
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


SIZE = env_int("ACTIVATION_SIZE")
BLOCK_SIZE = env_int("ACTIVATION_BLOCK_SIZE")
GRID_X = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE


@triton.jit
def activation(x_ptr, out_ptr,
               n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, y, mask=mask)


# --- AOT cross-compilation ---
x = torch.empty(SIZE, dtype=torch.float32)
out = torch.empty(SIZE, dtype=torch.float32)

activation[(GRID_X,)](x, out, SIZE, BLOCK_SIZE)
