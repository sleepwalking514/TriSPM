# updated version

import torch
import triton
import triton.language as tl
import os
from utils import save_matrices_to_txt

DTYPE = getattr(torch, (os.getenv("DTYPE", "float32")))

def jacobi_kernel(
        a_ptr, b_ptr,
        M, N,
        stride_m, stride_n,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N - 2, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    block_offset_m = pid_m * BLOCK_SIZE_M
    block_offset_n = pid_n * BLOCK_SIZE_N

    a_up_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, N), strides=(stride_m, stride_n),
                                    offsets=(block_offset_m, block_offset_n+1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                                    order=(1, 0))
    a_mid_tile_ptr = tl.advance(a_up_tile_ptr, (1, 0))
    a_down_tile_ptr = tl.advance(a_mid_tile_ptr, (1, 0))
    a_left_tile_ptr = tl.advance(a_mid_tile_ptr, (0, -1))
    a_right_tile_ptr = tl.advance(a_mid_tile_ptr, (0, 1))

    a_up = tl.load(a_up_tile_ptr)
    tmp = a_up

    a_mid = tl.load(a_mid_tile_ptr)
    tmp += a_mid

    a_down = tl.load(a_down_tile_ptr)
    tmp += a_down

    a_left = tl.load(a_left_tile_ptr)
    tmp += a_left

    a_right = tl.load(a_right_tile_ptr)
    tmp += a_right

    # Jacobi iteration: compute the average of the neighbors
    b = 0.2 * tmp

    b_ptrs = tl.make_block_ptr(base=b_ptr, shape=(M, N), strides=(stride_m, stride_n),
                             offsets=(block_offset_m+1, block_offset_n+1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                             order=(1, 0))

    tl.store(b_ptrs, b)


# Triton Benchmark
def get_jacobi_kernel_autotune_config(num_threads=0):
    configs = []
    
    block_sizes_M = [4, 8, 16, 32, 64]
    block_sizes_N = [4, 8, 16, 32, 64]

    for block_m in block_sizes_M:
        for block_n in block_sizes_N:
            print(f"Config: BLOCK_SIZE_M={block_m}, BLOCK_SIZE_N={block_n}")
            configs.append(
                triton.Config({
                    'BLOCK_SIZE_M': block_m,
                    'BLOCK_SIZE_N': block_n
                }, num_threads=num_threads)
            )
    
    return configs


def benchmark_triton(shape, a, b, parallel=False):
    fn = jacobi_kernel
    fn_jit = triton.jit(fn)
    fn_jit_tuned = triton.runtime.Autotuner(fn_jit, fn_jit.arg_names, 
        reset_to_zero=None, 
        restore_value=None,
        configs=get_jacobi_kernel_autotune_config(0 if parallel else 1),
        key=[],
    )

    M, N = shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M-2, META["BLOCK_SIZE_M"]) * triton.cdiv(N-2, META["BLOCK_SIZE_N"]),
    )

    fn_jit_tuned[grid](
        a, b, M, N,
        a.stride(0), a.stride(1)
    )


def jacobi(shape, a, b):
    M, N = shape
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 16

    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M-2, BLOCK_SIZE_M) * triton.cdiv(N-2, BLOCK_SIZE_N), )

    jacobi_jit = triton.jit(jacobi_kernel)
    jacobi_jit[grid](
        a, b, M, N,
        a.stride(0), a.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    return b

if __name__ == "__main__":
    torch.manual_seed(0)
    triton.runtime.driver.set_active_to_cpu()

    a = torch.randn((1026, 1026), device='cpu', dtype=DTYPE)
    b = torch.empty((1026, 1026), device='cpu', dtype=DTYPE)

    shape = (1026, 1026)

    benchmark_triton(shape, a, b, parallel=False)

    b = jacobi(shape, a, b)
    out = b[1:-1, 1:-1]
    ans = 0.2 * (a[1:-1, 1:-1] + a[:-2, 1:-1] + a[1:-1, :-2] + a[2:, 1:-1] + a[1:-1, 2:])
    print(torch.allclose(out, ans, atol=1e-5))

    # # Save matrices to txt files
    # save_matrices_to_txt(
    #     a, b,
    #     output_dir="test_data",
    #     precision=9,
    #     dtype="float",
    #     prefix="matrix",
    #     delimiter=" ",
    #     create_manifest=False,
    #     start_idx=1
    # )