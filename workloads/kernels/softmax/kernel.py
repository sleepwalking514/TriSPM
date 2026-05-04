"""
Triton AOT compilation: row-wise softmax -> LLVM IR for RISC-V.

This workload keeps the canonical one-row schedule separate from the row-block
schedule used by the Phase 3.5 DMA experiment.  Both lower through the same
`softmax` symbol so the existing AOT launcher and harness stay unchanged.
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


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


M = env_int("SOFTMAX_M")
N = env_int("SOFTMAX_N")
BLOCK_N = env_int("SOFTMAX_BLOCK_N")
SCHEDULE_ID = env_int("SOFTMAX_SCHEDULE_ID")
ROW_BLOCK = env_int("SOFTMAX_ROW_BLOCK")
ROW_GROUP_BLOCKS = env_int("SOFTMAX_ROW_GROUP_BLOCKS")
SCHEDULE = env_str("SOFTMAX_SCHEDULE", "canonical")

if SCHEDULE == "canonical":
    if SCHEDULE_ID != 0:
        raise ValueError("canonical softmax requires SOFTMAX_SCHEDULE_ID=0")
elif SCHEDULE == "row_block":
    if SCHEDULE_ID != 1:
        raise ValueError("row_block softmax requires SOFTMAX_SCHEDULE_ID=1")
else:
    raise ValueError(
        f"SOFTMAX_SCHEDULE must be canonical or row_block, got {SCHEDULE!r}"
    )

if N % BLOCK_N != 0:
    raise ValueError("softmax workload requires N to be divisible by BLOCK_N")
if SCHEDULE_ID == 0:
    if ROW_BLOCK != 1 or ROW_GROUP_BLOCKS != 1:
        raise ValueError(
            "canonical softmax requires ROW_BLOCK=1 and ROW_GROUP_BLOCKS=1"
        )
    GRID_X = M
else:
    if ROW_BLOCK <= 1:
        raise ValueError("row_block softmax requires ROW_BLOCK > 1")
    if ROW_GROUP_BLOCKS <= 0:
        raise ValueError("row_block softmax requires ROW_GROUP_BLOCKS > 0")
    if M % (ROW_BLOCK * ROW_GROUP_BLOCKS) != 0:
        raise ValueError(
            "row_block softmax requires M to be divisible by "
            "ROW_BLOCK * ROW_GROUP_BLOCKS"
        )
    GRID_X = triton.cdiv(M, ROW_BLOCK * ROW_GROUP_BLOCKS)


@triton.jit
def softmax(x_ptr, out_ptr,
            M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
            SCHEDULE_ID: tl.constexpr, ROW_BLOCK: tl.constexpr,
            ROW_GROUP_BLOCKS: tl.constexpr):
    if SCHEDULE_ID == 0:
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
    else:
        group_row_base = tl.program_id(0) * ROW_BLOCK * ROW_GROUP_BLOCKS
        for rb in range(0, ROW_GROUP_BLOCKS):
            row_base = group_row_base + rb * ROW_BLOCK

            x_max_ptr = tl.make_block_ptr(
                base=x_ptr, shape=(N, M), strides=(1, N),
                offsets=(0, row_base), block_shape=(BLOCK_N, ROW_BLOCK),
                order=(0, 1))
            x_sum_ptr = tl.make_block_ptr(
                base=x_ptr, shape=(N, M), strides=(1, N),
                offsets=(0, row_base), block_shape=(BLOCK_N, ROW_BLOCK),
                order=(0, 1))
            x_norm_ptr = tl.make_block_ptr(
                base=x_ptr, shape=(N, M), strides=(1, N),
                offsets=(0, row_base), block_shape=(BLOCK_N, ROW_BLOCK),
                order=(0, 1))
            out_block_ptr = tl.make_block_ptr(
                base=out_ptr, shape=(N, M), strides=(1, N),
                offsets=(0, row_base), block_shape=(BLOCK_N, ROW_BLOCK),
                order=(0, 1))

            row_max = tl.full((ROW_BLOCK,), -3.4028234663852886e38,
                              dtype=tl.float32)
            for off in range(0, N, BLOCK_N):
                x = tl.load(x_max_ptr).to(tl.float32)
                row_max = tl.maximum(row_max, tl.max(x, axis=0))
                x_max_ptr = tl.advance(x_max_ptr, (BLOCK_N, 0))

            denominator = tl.zeros((ROW_BLOCK,), dtype=tl.float32)
            for off in range(0, N, BLOCK_N):
                x = tl.load(x_sum_ptr).to(tl.float32)
                numerator = tl.exp(x - row_max)
                denominator += tl.sum(numerator, axis=0)
                x_sum_ptr = tl.advance(x_sum_ptr, (BLOCK_N, 0))

            for off in range(0, N, BLOCK_N):
                x = tl.load(x_norm_ptr).to(tl.float32)
                numerator = tl.exp(x - row_max)
                y = numerator / denominator
                tl.store(out_block_ptr, y)
                x_norm_ptr = tl.advance(x_norm_ptr, (BLOCK_N, 0))
                out_block_ptr = tl.advance(out_block_ptr, (BLOCK_N, 0))


# --- AOT cross-compilation ---
x = torch.empty(M, N, dtype=torch.float32)
out = torch.empty(M, N, dtype=torch.float32)

softmax[(GRID_X,)](
    x, out, M, N, BLOCK_N, SCHEDULE_ID, ROW_BLOCK, ROW_GROUP_BLOCKS
)
