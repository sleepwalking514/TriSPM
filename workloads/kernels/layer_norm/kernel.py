"""
Triton AOT compilation: layer_norm kernel -> LLVM IR for RISC-V.

Per-row normalization: out[i,:] = gamma * (x[i,:] - mean) / sqrt(var + eps) + beta

N is a *runtime* (non-constexpr) parameter and BLOCK_SIZE_N is kept at 8 (one
RISC-V vector register, LMUL=1 at VLEN=256).  This avoids two gem5 RVV bugs:
  1. LMUL=8 operations are broken outright.
  2. vfredusum.vs produces wrong results when LMUL>=2 and the destination
     register overlaps the source register group.
Three-pass algorithm: mean, variance, normalize+store.
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


M = env_int("M_SIZE")        # rows
N = env_int("N_SIZE")        # features per row
BLOCK_SIZE_N = 8             # process N in chunks (LMUL=1 at VLEN=256)
GRID_X = M                   # one program per row


@triton.jit
def layer_norm(x_ptr, gamma_ptr, beta_ptr, out_ptr,
               N,                               # runtime — prevents loop fusion
               M: tl.constexpr,
               BLOCK_SIZE_N: tl.constexpr):
    """Layer-normalise one row of the [M, N] input tensor."""
    row = tl.program_id(0)

    # ---- Pass 1: mean ----
    _sum = tl.zeros((1,), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        offs = off + tl.arange(0, BLOCK_SIZE_N)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.sum(x, axis=0)
    mean = _sum / N

    # ---- Pass 2: variance ----
    _var = tl.zeros((1,), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        offs = off + tl.arange(0, BLOCK_SIZE_N)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        d = x - mean
        _var += tl.sum(d * d, axis=0)
    var = _var / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    # ---- Pass 3: normalize, scale, store ----
    for off in range(0, N, BLOCK_SIZE_N):
        offs = off + tl.arange(0, BLOCK_SIZE_N)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(beta_ptr  + offs, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * inv_std
        out = x_norm * g + b
        tl.store(out_ptr + row * N + offs, out, mask=mask)


# --- AOT cross-compilation ---
x     = torch.empty(M, N, dtype=torch.float32)
gamma = torch.ones(N, dtype=torch.float32)
beta  = torch.zeros(N, dtype=torch.float32)
out   = torch.empty(M, N, dtype=torch.float32)

layer_norm[(GRID_X,)](x, gamma, beta, out, N, M, BLOCK_SIZE_N)
