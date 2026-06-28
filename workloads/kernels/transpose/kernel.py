"""Triton AOT compilation: row-major transpose -> LLVM IR for RISC-V."""
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


M = env_int("TRANSPOSE_M")
N = env_int("TRANSPOSE_N")
BLOCK_M = env_int("TRANSPOSE_BLOCK_M")
BLOCK_N = env_int("TRANSPOSE_BLOCK_N")
GRID_X = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)


@triton.jit
def transpose(x_ptr, out_ptr,
              M: tl.constexpr, N: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(
        x_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + offs_n[None, :] * M + offs_m[:, None],
        x,
        mask=mask,
    )


# --- AOT cross-compilation ---
x = torch.empty(M, N, dtype=torch.float32)
out = torch.empty(N, M, dtype=torch.float32)

transpose[(GRID_X,)](x, out, M, N, BLOCK_M, BLOCK_N)
