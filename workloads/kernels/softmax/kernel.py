"""
Triton AOT compilation: softmax kernel -> LLVM IR for RISC-V.

Row-wise softmax coverage for attention-facing reductions.  The first workload
keeps each row in a single block so the access pattern is simple and future
row-resident promotion experiments have a canonical block-pointer load to
target.
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


M = env_int("SOFTMAX_M")
N = env_int("SOFTMAX_N")
BLOCK_N = env_int("SOFTMAX_BLOCK_N")
GRID_X = M

if N != BLOCK_N:
    raise ValueError("softmax first workload requires N == BLOCK_N")


@triton.jit
def softmax(x_ptr, out_ptr,
            M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))

    x = tl.load(x_block_ptr).to(tl.float32)
    row_max = tl.max(x, axis=0)
    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator
    tl.store(out_block_ptr, y)


# --- AOT cross-compilation ---
x = torch.empty(M, N, dtype=torch.float32)
out = torch.empty(M, N, dtype=torch.float32)

softmax[(GRID_X,)](x, out, M, N, BLOCK_N)
