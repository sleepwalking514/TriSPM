"""
Triton AOT compilation: strided batched GEMM -> LLVM IR for RISC-V.

Computes C[b, :, :] = A[b, :, :] @ B[b, :, :] for a fixed batch count.
The batch dimension is folded into the block-pointer base offsets so the
compiler still sees rank-2 tiled GEMM loads feeding a GEMM-shaped contraction.
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


BATCH = env_int("BATCHED_GEMM_BATCH")
M = env_int("BATCHED_GEMM_M")
N = env_int("BATCHED_GEMM_N")
K = env_int("BATCHED_GEMM_K")
BLOCK_SIZE_M = env_int("BATCHED_GEMM_BLOCK_SIZE_M")
BLOCK_SIZE_N = env_int("BATCHED_GEMM_BLOCK_SIZE_N")
BLOCK_SIZE_K = env_int("BATCHED_GEMM_BLOCK_SIZE_K")
GROUP_SIZE_M = env_int("BATCHED_GEMM_GROUP_SIZE_M")

if BATCH <= 0:
    raise ValueError("batched_gemm requires positive BATCH")
if M % BLOCK_SIZE_M != 0 or N % BLOCK_SIZE_N != 0 or K % BLOCK_SIZE_K != 0:
    raise ValueError("batched_gemm dimensions must be exact multiples of block sizes")

GRID_X = BATCH * (M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N)


@triton.jit
def batched_gemm(a_ptr, b_ptr, c_ptr,
                 BATCH: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                 K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                 BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                 GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tiles_per_batch = num_pid_m * num_pid_n

    batch = pid // tiles_per_batch
    tile_pid = pid - batch * tiles_per_batch

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_pid % group_size_m)
    pid_n = (tile_pid % num_pid_in_group) // group_size_m

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(BATCH * M, K), strides=(K, 1),
        offsets=(batch * M + pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(BATCH * K, N), strides=(N, 1),
        offsets=(batch * K, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_SIZE_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(BATCH * M, N), strides=(N, 1),
        offsets=(batch * M + pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, acc)


a = torch.empty(BATCH * M, K, dtype=torch.float32)
b = torch.empty(BATCH * K, N, dtype=torch.float32)
c = torch.empty(BATCH * M, N, dtype=torch.float32)

batched_gemm[(GRID_X,)](a, b, c, BATCH, M, N, K,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                        GROUP_SIZE_M)
