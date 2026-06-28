"""
Triton AOT compilation: RMSNorm kernel -> LLVM IR for RISC-V.

Per-row root-mean-square normalization:

    out[i, :] = gamma[:] * x[i, :] / sqrt(mean(x[i, :]^2) + eps)

The kernel is intentionally written as two row passes so the existing
row-block SPM lowering can stage one row block and reuse it for statistics and
normalize/store.
"""
import os

import torch
import triton
import triton.language as tl


def env_int(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(
            f"{name} must be exported from experiment.toml by run_experiment.py"
        )
    return int(value)


M = env_int("M")
N = env_int("N")
BLOCK_N = env_int("BLOCK_N")
GRID_X = M

if BLOCK_N <= 0:
    raise ValueError("rms_norm requires positive BLOCK_N")
if N % BLOCK_N != 0:
    raise ValueError("rms_norm workload requires N to be divisible by BLOCK_N")


@triton.jit
def rms_norm(x_ptr, gamma_ptr, out_ptr,
             M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)

    x_stats_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    x_norm_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    gamma_block_ptr = tl.make_block_ptr(
        base=gamma_ptr, shape=(N,), strides=(1,),
        offsets=(0,), block_shape=(BLOCK_N,), order=(0,))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))

    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        x = tl.load(x_stats_ptr).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        x_stats_ptr = tl.advance(x_stats_ptr, (BLOCK_N,))

    inv_rms = 1.0 / tl.sqrt(sum_sq / N + 1e-5)

    for off in range(0, N, BLOCK_N):
        x = tl.load(x_norm_ptr).to(tl.float32)
        g = tl.load(gamma_block_ptr).to(tl.float32)
        out = x * inv_rms * g
        tl.store(out_block_ptr, out)
        x_norm_ptr = tl.advance(x_norm_ptr, (BLOCK_N,))
        gamma_block_ptr = tl.advance(gamma_block_ptr, (BLOCK_N,))
        out_block_ptr = tl.advance(out_block_ptr, (BLOCK_N,))


x = torch.empty(M, N, dtype=torch.float32)
gamma = torch.ones(N, dtype=torch.float32)
out = torch.empty(M, N, dtype=torch.float32)

rms_norm[(GRID_X,)](x, gamma, out, M, N, BLOCK_N)
