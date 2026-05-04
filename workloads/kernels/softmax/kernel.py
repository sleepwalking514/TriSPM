"""Triton AOT compilation: canonical row-wise softmax -> LLVM IR for RISC-V."""
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


M = env_int("SOFTMAX_M")
N = env_int("SOFTMAX_N")
BLOCK_N = env_int("SOFTMAX_BLOCK_N")

if N % BLOCK_N != 0:
    raise ValueError("softmax workload requires N to be divisible by BLOCK_N")


@triton.jit
def softmax(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
            BLOCK_N: tl.constexpr):
    row = tl.program_id(0)

    x_max_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    x_sum_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    x_norm_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))

    row_max = tl.full((1,), -3.4028234663852886e38, dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        x = tl.load(x_max_ptr).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
        x_max_ptr = tl.advance(x_max_ptr, (BLOCK_N,))

    denominator = tl.zeros((1,), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        x = tl.load(x_sum_ptr).to(tl.float32)
        numerator = tl.exp(x - row_max)
        denominator += tl.sum(numerator, axis=0)
        x_sum_ptr = tl.advance(x_sum_ptr, (BLOCK_N,))

    for off in range(0, N, BLOCK_N):
        x = tl.load(x_norm_ptr).to(tl.float32)
        numerator = tl.exp(x - row_max)
        y = numerator / denominator
        tl.store(out_block_ptr, y)
        x_norm_ptr = tl.advance(x_norm_ptr, (BLOCK_N,))
        out_block_ptr = tl.advance(out_block_ptr, (BLOCK_N,))


# --- AOT cross-compilation ---
x = torch.empty(M, N, dtype=torch.float32)
out = torch.empty(M, N, dtype=torch.float32)

softmax[(M,)](x, out, M, N, BLOCK_N)
