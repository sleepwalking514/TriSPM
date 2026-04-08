"""
Triton AOT compilation: matmul kernel -> LLVM IR for RISC-V.

Tiled matrix multiply: C[M,N] = A[M,K] @ B[K,N]
All dimensions are exact multiples of their block sizes (no masking).
Row-major contiguous layout; strides baked in as constexprs.
"""
import torch
import triton
import triton.language as tl

M = 64
N = 64
K = 64
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16
GROUP_SIZE_M = 4

GRID_X = (M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N)  # 16


@triton.jit
def matmul(a_ptr, b_ptr, c_ptr,
           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
           BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Tiled matmul with L1-friendly GROUP_SIZE_M tile grouping."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row-major contiguous: stride_am=K, stride_ak=1, stride_bk=N, stride_bn=1
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K          # move along K in A
        b_ptrs += BLOCK_SIZE_K * N      # move along K in B

    # Store C tile
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    tl.store(c_ptrs, acc)


# --- AOT cross-compilation ---
a = torch.empty(M, K, dtype=torch.float32)
b = torch.empty(K, N, dtype=torch.float32)
c = torch.empty(M, N, dtype=torch.float32)

matmul[(GRID_X,)](a, b, c, M, N, K,
                  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
