"""Triton AOT compilation: exact online row-wise softmax -> LLVM IR."""
import os

import torch
import triton
import triton.language as tl


def env_int(name: str, default: int | None = None) -> int:
    value = os.getenv(name)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(
            f"{name} must be exported from experiment.toml by run_experiment.py"
        )
    return int(value)


M = env_int("SOFTMAX_ONLINE_M")
N = env_int("SOFTMAX_ONLINE_N")
BLOCK_N = env_int("SOFTMAX_ONLINE_BLOCK_N")
CAUSAL = env_int("SOFTMAX_ONLINE_CAUSAL", 0)

if N % BLOCK_N != 0:
    raise ValueError("softmax_online workload requires N to be divisible by BLOCK_N")
if CAUSAL not in (0, 1):
    raise ValueError("softmax_online CAUSAL must be 0 or 1")


@triton.jit
def softmax_online(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                   BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    neg_inf = -3.4028234663852886e38

    x_scan_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    x_norm_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(M * N,), strides=(1,),
        offsets=(row * N,), block_shape=(BLOCK_N,), order=(0,))

    row_max = tl.full((1,), neg_inf, dtype=tl.float32)
    denominator = tl.zeros((1,), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        x = tl.load(x_scan_ptr).to(tl.float32)
        if CAUSAL:
            cols = off + offs
            x = tl.where(cols <= row, x, neg_inf)

        tile_max = tl.max(x, axis=0)
        next_max = tl.maximum(row_max, tile_max)
        denominator = denominator * tl.exp(row_max - next_max)
        denominator += tl.sum(tl.exp(x - next_max), axis=0)
        row_max = next_max
        x_scan_ptr = tl.advance(x_scan_ptr, (BLOCK_N,))

    for off in range(0, N, BLOCK_N):
        x = tl.load(x_norm_ptr).to(tl.float32)
        if CAUSAL:
            cols = off + offs
            x = tl.where(cols <= row, x, neg_inf)
        y = tl.exp(x - row_max) / denominator
        tl.store(out_block_ptr, y)
        x_norm_ptr = tl.advance(x_norm_ptr, (BLOCK_N,))
        out_block_ptr = tl.advance(out_block_ptr, (BLOCK_N,))


# --- AOT cross-compilation ---
x = torch.empty(M, N, dtype=torch.float32)
out = torch.empty(M, N, dtype=torch.float32)

softmax_online[(M,)](x, out, M, N, BLOCK_N, CAUSAL)
