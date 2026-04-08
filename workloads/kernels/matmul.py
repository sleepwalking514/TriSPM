'''
Currently masked load is not supported yet
'''

import torch
import triton
import triton.language as tl
import os

# KERNEL_LAUNCHER_INCLUDE_DIR=os.getenv("KERNEL_LAUNCHER_INCLUDE_DIR", "/home/yuhao/T_RVV/benchmark/src/launcher/include")
# KERNEL_AUX_FILE_DIR=os.getenv("KERNEL_AUX_FILE_DIR", "/home/yuhao/T_RVV/benchmark/src/launcher/src/matmul")

# KERNEL_LAUNCHER_INCLUDE_DIR="/home/yuhao/T_RVV/benchmark/src/launcher/include" KERNEL_AUX_FILE_DIR="/home/yuhao/T_RVV/benchmark/src/launcher/src/matmul"

DTYPE = getattr(torch, (os.getenv("DTYPE", "float32")))
# Choose block size depending on dtype. We have more register
# capacity for bfloat16/float16 compared to float32.
# BLOCK_SIZE_M = 8 if DTYPE == torch.float32 else 32
# BLOCK_SIZE_K = 8 if DTYPE == torch.float32 else 32
# BLOCK_SIZE_N = 8
GROUP_SIZE_M = 8
USE_BLOCK_POINTERS = os.getenv("USE_BLOCK_POINTERS", "1") != "0"

def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,  
        GROUP_SIZE_M: tl.constexpr,
        USE_BLOCK_POINTERS: tl.constexpr,  #
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    if USE_BLOCK_POINTERS:
        block_offset_m = pid_m * BLOCK_SIZE_M
        block_offset_n = pid_n * BLOCK_SIZE_N
        a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       offsets=(block_offset_m, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                       order=(1, 0))
        b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                       offsets=(0, block_offset_n), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                       order=(1, 0))
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_tile_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_tile_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.

        a = tl.load(a_tile_ptr)
        b = tl.load(b_tile_ptr)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        # Advance the ptrs to the next K block.
        if USE_BLOCK_POINTERS:
            a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_SIZE_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_SIZE_K, 0])
        else:
            a_tile_ptr += BLOCK_SIZE_K * stride_ak
            b_tile_ptr += BLOCK_SIZE_K * stride_bk

    # Convert the accumulator to the output matrix C's type if needed.
    c = accumulator
    # -----------------------------------------------------------
    # Write back the block of the output matrix C.
    if USE_BLOCK_POINTERS:
        c_tile_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       offsets=(block_offset_m, block_offset_n),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
        tl.store(c_tile_ptr, c)
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_tile_ptr = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_tile_ptr, c)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    # offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # tl.store(c_ptrs, c, mask=c_mask)


# def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):

#     assert a.shape[1] == b.shape[0], "Incompatible dimensions"
#     assert a.is_contiguous(), "Matrix A must be contiguous"
#     M, K = a.shape
#     K, N = b.shape
#     assert c.shape == (M, N), "Incompatible dimensions"

#     # 1D launch kernel where each block gets its own program.
#     grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
#     matmul_kernel[grid](
#         a, b, c,  #
#         M, N, K,  #
#         a.stride(0), a.stride(1),  #
#         b.stride(0), b.stride(1),  #
#         c.stride(0), c.stride(1),  #
#         BLOCK_SIZE_M=BLOCK_SIZE_M, 
#         BLOCK_SIZE_N=BLOCK_SIZE_N, 
#         BLOCK_SIZE_K=BLOCK_SIZE_K,  #
#         GROUP_SIZE_M=GROUP_SIZE_M,  #
#         USE_BLOCK_POINTERS=USE_BLOCK_POINTERS,  #
#     )
#     return c


# Triton Benchmark
def get_matmul_kernel_autotune_config(num_threads=0):
    configs = []
    
    # All possible block sizes for each dimension
    # block_sizes_M = [4, 8, 16, 32, 64, 128]
    # block_sizes_N = [8, 16, 32, 64, 128, 256]
    # block_sizes_K = [8, 16, 32, 64, 128]
    
    block_sizes_M = [4, 8]
    block_sizes_N = [8, 16]
    block_sizes_K = [8]

    # Generate unique total block sizes from all combinations
    total_block_sizes = set()
    for block_m in block_sizes_M:
        for block_n in block_sizes_N:
            for block_k in block_sizes_K:
                total_block_sizes.add(block_m * block_n * block_k)
    
    # Sort the total block sizes for systematic exploration
    total_block_sizes = sorted(list(total_block_sizes))
    
    # For each total block size, find all combinations that achieve it
    for total_size in total_block_sizes:
        for block_m in block_sizes_M:
            for block_n in block_sizes_N:
                # Calculate the required block_k
                if total_size % (block_m * block_n) == 0:
                    block_k = total_size // (block_m * block_n)
                    # Only include if block_k is a valid block size
                    if block_k in block_sizes_K:
                        print(f"Config: BLOCK_SIZE_M={block_m}, BLOCK_SIZE_N={block_n}, BLOCK_SIZE_K={block_k}")
                        configs.append(
                            triton.Config({
                                'BLOCK_SIZE_M': block_m,
                                'BLOCK_SIZE_N': block_n,
                                'BLOCK_SIZE_K': block_k,
                                'GROUP_SIZE_M': GROUP_SIZE_M,
                                'USE_BLOCK_POINTERS': USE_BLOCK_POINTERS
                            }, num_threads=num_threads)
                        )
    
    return configs


def benchmark_triton(shape, a, b, parallel=False):
    fn = matmul_kernel
    fn_jit = triton.jit(fn)
    fn_jit_tuned = triton.runtime.Autotuner(fn_jit, fn_jit.arg_names, 
        reset_to_zero=None, 
        restore_value=None,
        configs=get_matmul_kernel_autotune_config(0 if parallel else 1),
        key=[],
    )

    M, N, K = shape
    c = torch.empty((M, N), dtype=torch.float32, device="cpu")
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    def run_triton_kernel():
        # don't need to include the Meta parameters in the call
        # to the kernel, they are already included in the config
        fn_jit_tuned[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1)
        )

    run_triton_kernel() # generate IR for all configs

    # # Warm up.
    # for _ in range(25):
    #     run_triton_kernel()

    # times = []
    # # Repeat to execute.
    # for _ in range(100):
    #     start = time.perf_counter()
    #     run_triton_kernel()
    #     end = time.perf_counter()
    #     times.append(end - start)
    # return np.mean(times), c.numpy(), tuning_time



if __name__ == "__main__":

    # Unit Test
    torch.manual_seed(0)
    triton.runtime.driver.set_active_to_cpu()

    a = torch.randn((512, 512), device='cpu', dtype=DTYPE)
    b = torch.randn((512, 512), device='cpu', dtype=DTYPE)
    c = torch.empty((512, 512), device='cpu', dtype=DTYPE)

    # torch_output = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    # triton_output = matmul(a, b, c, 512, 512, 512)

    # print(f"triton_cpu_output_with_{a.dtype}_inputs={triton_output}")
    # print(f"torch_cpu_output_with_{a.dtype}_inputs={torch_output}")
    # rtol = 0
    # if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    #     print("✅ TritonCPU and TorchCPU match")
    # else:
    #     print("❌ TritonCPU and TorchCPU differ, the maximum difference is "f'{torch.max(torch.abs(triton_output - torch_output))}')

    # Autotune:
    shape = (512, 512, 512)
    benchmark_triton(shape, a, b, parallel=False)