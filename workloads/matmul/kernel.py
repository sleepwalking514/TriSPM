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

    # Block pointers: produce vector.transfer_read from memref, which the
    # ConvertMemoryToSPM pass can match and transform into DMA+SPM transfers.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(K, 1),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(N, 1),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_SIZE_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    # Store C tile (block pointer for consistent lowering path)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, acc)


# --- AOT cross-compilation ---
a = torch.empty(M, K, dtype=torch.float32)
b = torch.empty(K, N, dtype=torch.float32)
c = torch.empty(M, N, dtype=torch.float32)

matmul[(GRID_X,)](a, b, c, M, N, K,
                  BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
