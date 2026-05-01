"""
Triton AOT compilation: layer_norm kernel -> LLVM IR for RISC-V.

Per-row normalization: out[i,:] = gamma * (x[i,:] - mean) / sqrt(var + eps) + beta

N and BLOCK_N are constexprs, with BLOCK_N kept at 8 (one
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


M = env_int("M")        # rows
N = env_int("N")        # features per row
BLOCK_N = 8             # process N in chunks (LMUL=1 at VLEN=256)
GRID_X = M                   # one program per row


@triton.jit
def layer_norm(x_ptr, gamma_ptr, beta_ptr, out_ptr,
               M: tl.constexpr, N: tl.constexpr,
               BLOCK_N: tl.constexpr):
    """Layer-normalise one row of the [M, N] input tensor."""
    row = tl.program_id(0)

    # Use block pointers so the SPM conversion pass sees canonical
    # vector.transfer_read ops.
    x_mean_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    x_var_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    x_norm_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    gamma_block_ptr = tl.make_block_ptr(
        base=gamma_ptr, shape=(N,), strides=(1,),
        offsets=(0,), block_shape=(BLOCK_N,), order=(0,))
    beta_block_ptr = tl.make_block_ptr(
        base=beta_ptr, shape=(N,), strides=(1,),
        offsets=(0,), block_shape=(BLOCK_N,), order=(0,))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))

    # ---- Pass 1: mean ----
    _sum = tl.zeros((1,), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        x = tl.load(x_mean_ptr).to(tl.float32)
        _sum += tl.sum(x, axis=0)
        x_mean_ptr = tl.advance(x_mean_ptr, (BLOCK_N,))
    mean = _sum / N

    # ---- Pass 2: variance ----
    _var = tl.zeros((1,), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        x = tl.load(x_var_ptr).to(tl.float32)
        d = x - mean
        _var += tl.sum(d * d, axis=0)
        x_var_ptr = tl.advance(x_var_ptr, (BLOCK_N,))
    var = _var / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    # ---- Pass 3: normalize, scale, store ----
    for off in range(0, N, BLOCK_N):
        x = tl.load(x_norm_ptr).to(tl.float32)
        g = tl.load(gamma_block_ptr).to(tl.float32)
        b = tl.load(beta_block_ptr).to(tl.float32)
        x_norm = (x - mean) * inv_std
        out = x_norm * g + b
        tl.store(out_block_ptr, out)
        x_norm_ptr = tl.advance(x_norm_ptr, (BLOCK_N,))
        gamma_block_ptr = tl.advance(gamma_block_ptr, (BLOCK_N,))
        beta_block_ptr = tl.advance(beta_block_ptr, (BLOCK_N,))
        out_block_ptr = tl.advance(out_block_ptr, (BLOCK_N,))


# --- AOT cross-compilation ---
x     = torch.empty(M, N, dtype=torch.float32)
gamma = torch.ones(N, dtype=torch.float32)
beta  = torch.zeros(N, dtype=torch.float32)
out   = torch.empty(M, N, dtype=torch.float32)

layer_norm[(GRID_X,)](x, gamma, beta, out, M, N, BLOCK_N)
