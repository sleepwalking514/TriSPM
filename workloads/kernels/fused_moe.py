import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import os
import numpy as np

# --- Configuration ---
DEVICE = 'cpu'
DTYPE = torch.float32
DTYPE_ACC = tl.float32

# Set Triton to run on the CPU backend
triton.runtime.driver.set_active_to_cpu()

def transpose_moe_b_kernel(
        B_ptr, B_out_ptr,
        E, N, K,
        stride_be, stride_bn, stride_bk,
        stride_out_e, stride_out_k, stride_out_n,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        top_k: tl.constexpr,
):
    """
    Transpose B matrix from [E, N, K] to [E, K, N] layout
    """
    pid = tl.program_id(axis=0)
    
    # Calculate which expert, K block, and N block we're processing
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Calculate expert, k_block, and n_block indices
    expert_kn_idx = pid // (num_k_blocks * num_n_blocks)
    kn_block_idx = pid % (num_k_blocks * num_n_blocks)
    
    k_block_idx = kn_block_idx // num_n_blocks
    n_block_idx = kn_block_idx % num_n_blocks
    
    # Input block pointer for B[E, N, K]
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(E, N, K),
        strides=(stride_be, stride_bn, stride_bk),
        offsets=(expert_kn_idx, n_block_idx * BLOCK_SIZE_N, k_block_idx * BLOCK_SIZE_K),
        block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(2, 1, 0)
    )

    # Output block pointer for B_out[E, K, N]
    b_block_out_ptr = tl.make_block_ptr(
        base=B_out_ptr,
        shape=(E, K, N),
        strides=(stride_out_e, stride_out_k, stride_out_n),
        offsets=(expert_kn_idx, k_block_idx * BLOCK_SIZE_K, n_block_idx * BLOCK_SIZE_N),
        block_shape=(1, BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(2, 1, 0)
    )

    # Load, transpose, and store
    b_block = tl.load(b_block_ptr)
    b_block_T = tl.trans(b_block, (0, 2, 1))  # Transpose last two dimensions
    tl.store(b_block_out_ptr, b_block_T)

def fused_moe_kernel(
    # Pointers to matrices
    a_ptr, b_ptr_transposed, c_ptr,  # Note: b_ptr_transposed is now [E, K, N]
    # These will be updated by pre_hook:
    sorted_token_ids_ptr,
    expert_ids_ptr,
    # Matrix dimensions
    E, N, K, num_tokens_post_padded,
    num_valid_tokens,
    stride_am, stride_ak, # [M, K]
    stride_bk, stride_bn,  # For transposed B: [E, K, N]
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using token and expert matrices.
    Note: b_ptr_transposed should be pre-transposed from [E, N, K] to [E, K, N] layout.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = (offs_token < num_valid_tokens) & (offs_token >= 0)

    # A pointer setup (cannot use make_block_ptr due to special mask)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    # Get expert ID for this block
    off_experts = tl.load(expert_ids_ptr + pid_m)
    # off_experts * stride_be  = off_experts * K * N

    # Use make_block_ptr for B (now transposed to [E, K, N])
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr_transposed,
        shape=(E * K, N),
        strides=(stride_bk, stride_bn), 
        offsets=(off_experts * K, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A (manual pointer due to special mask)
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        
        # Load the next block of B using make_block_ptr
        b = tl.load(b_block_ptr)  # Check K and N dimensions
        # tl.store(b, pos)

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_block_ptr = tl.advance(b_block_ptr, [BLOCK_SIZE_K, 0])
        # b_block_ptr = tl.advance(b_block_ptr, [3, 0])

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):

    num_tokens, top_k = topk_ids.shape # (M, TopK)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device) # (E,)

    expert_counts.index_add_(0, topk_ids.flatten(), torch.ones_like(topk_ids.flatten(), dtype=torch.int32))

    # Count how many tokens each Expert have (With padding)
    padded_expert_counts = torch.ceil(expert_counts.float() / block_size).long() * block_size
    # Count new total number of tokens (newM,)
    total_padded_tokens = torch.sum(padded_expert_counts).item()

    # # Debug print
    # print("TopK IDs flattened:", topk_ids.flatten())
    # print("Ones tensor:", torch.ones_like(topk_ids.flatten(), dtype=torch.int32))
    # print("Expert counts before:", expert_counts)
    # print("Expert counts after:", expert_counts)
    # print("Total padded tokens:", total_padded_tokens)
    # print("Padded Expert counts:", padded_expert_counts)

    # Start to create expert_tokens
    expert_tokens = [[] for _ in range(num_experts)]
    padding = padded_expert_counts.clone()
    for k in range(top_k):  
        for i in range(num_tokens):
            expert_id = topk_ids[i, k].item()
            expert_tokens[expert_id].append(i)
            padding[expert_id] -= 1

    for expert_id in range(num_experts):
        # Sort the tokens
        expert_tokens[expert_id].sort()
        while padding[expert_id] > 0:
            # Padding with -1 for this expert
            expert_tokens[expert_id].append(-1)
            padding[expert_id] -= 1

    # # Debug print
    # print("Expert tokens after padding:", expert_tokens)
    # # Check correct or not
    # total_tokens = sum(len(tokens) for tokens in expert_tokens) # Total tokens in expert_tokens
    # if total_tokens != total_padded_tokens:
    #     print(f"Error: Length mismatch! Expected {total_padded_tokens}, got {total_tokens}")
    # else:
    #     print("Length check passed")

    # Create the expert ids for each block
    expert_ids_for_blocks = []
    for expert_id in range(num_experts):
        num_blocks = padded_expert_counts[expert_id].item() // block_size
        expert_ids_for_blocks.extend([expert_id] * num_blocks)


    # If expert_tokens need flat 1D
    flat_expert_tokens = [token for expert in expert_tokens for token in expert]

    return torch.tensor(flat_expert_tokens, dtype=torch.int32, device=topk_ids.device), torch.tensor(expert_ids_for_blocks, dtype=torch.int32, device=topk_ids.device), total_padded_tokens

def get_moe_fused_autotune_config(block_size_m, num_threads=1):
    configs = []
    block_sizes_n = [8, 16, 32, 64]
    block_sizes_k = [8, 16, 32, 64]
    
    for bn in block_sizes_n:
        for bk in block_sizes_k:
            configs.append(
                triton.Config(
                    {
                        'BLOCK_SIZE_M': block_size_m, 
                        'BLOCK_SIZE_N': bn, 
                        'BLOCK_SIZE_K': bk, 
                        'GROUP_SIZE_M': 8, 
                        'top_k': 2
                    }, num_threads=num_threads
                )
            )
    return configs

# Reorder Kernel Function Here
def reorder_kernel(
    a_ptr, expert_ptr, r_ptr, # Matrix a, Expert e, Roerder Matrix r
    M, newM, K,
    stride_am, stride_ak,
    stride_rm, stride_rk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
):
    '''
    This function is used to reorder Matrix A from (M, K) to (newM, K)

    Matrix e: sorted_token_ids
    Matrix r: Matrix reordered_a

    For the meta-parameters:
    BLOCK_SIZE_N, GROUP_SIZE_M, top_k will not be used here
    '''

    pid = tl.program_id(axis=0)

    # Block offsets (M: rows, K: columns)
    block_offset_m = (pid * 1)
    block_offset_k = 0

    # Matrix a offsets
    offset_e = tl.load(expert_ptr + pid)

    if offset_e == -1:
        return

    r_tile_ptr = tl.make_block_ptr(
        base=r_ptr,
        shape=(newM, K),
        strides=(stride_rm, stride_rk),
        offsets=(block_offset_m, block_offset_k),
        block_shape=(1, BLOCK_SIZE_K),
        order=(1, 0)
    )

    a_tile_ptr = tl.make_block_ptr(
        base=a_ptr, 
        shape=(M, K), 
        strides=(stride_am, stride_ak),
        offsets=(offset_e, block_offset_k), 
        block_shape=(1, BLOCK_SIZE_K), 
        order=(1, 0)
    )

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_data = tl.load(a_tile_ptr)
        tl.store(r_tile_ptr, input_data)

        # Advance the pointers to the next BLOCK_SIZE_K block
        a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_SIZE_K])
        r_tile_ptr = tl.advance(r_tile_ptr, [0, BLOCK_SIZE_K])

def benchmark_triton(a, b, c,
                     sorted_token_ids, expert_ids_ptr,
                     num_tokens_post_padded, num_valid_tokens,
                     block_size_m):

    # a: Matrix A, shape(M, K)
    # b: Matrix B, shape(E, N, K)
    # c: Matrix C, the output Matrix, shape(M, topK, N)
    # sorted_token_ids: Sorted token IDs for each block, shape(E, M')
    # expert_ids_ptr: Expert IDs for each block, shape(E,)
    # num_tokens_post_padded: Total number of tokens after padding
    # num_valid_tokens: Number of valid tokens
    # block_size_m: Block size for M dimension

    M, K_a = a.shape
    E, N, K_b = b.shape
    assert K_a == K_b

    # First auto-tuner for transpose kernel
    fn_transpose_jit = triton.jit(transpose_moe_b_kernel)
    fn_transpose_tuned = triton.runtime.Autotuner(
        fn_transpose_jit, 
        fn_transpose_jit.arg_names, 
        reset_to_zero=None, 
        restore_value=None,
        configs=get_moe_fused_autotune_config(block_size_m),  # Configs for transpose kernel
        key=[],
    )

    # Step 1: Transpose B from [E, N, K] to [E, K, N]
    b_transposed = torch.empty((E, K_b, N), dtype=b.dtype, device=b.device)
    
    # Grid for transpose
    transpose_grid = lambda META: (E * triton.cdiv(K_b, META['BLOCK_SIZE_K']) 
                                   * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    fn_transpose_tuned[transpose_grid](
        b, b_transposed,
        E, N, K_b,
        b.stride(0), b.stride(1), b.stride(2),  # [E, N, K] strides
        b_transposed.stride(0), b_transposed.stride(1), b_transposed.stride(2),  # [E, K, N] strides
    )

    '''
    Step 2: Reorder Matrix A
    This step reorders the matrix A based on the sorted token IDs.

    Matrix A shape (M, K_a)
    Reordered Matrix A shape (newM, K_a)
    sorted_token_ids shape (newM,)

    The sorted_token_ids is a 1D tensor that contains the row index of Matrix A
    To adjust the sorted_token_ids, go to moe_align_block_size() function
    ________________________________________________________________________________________________________________________________________

    For Reordering the Matrix A, it will call the reorder_kernel() function
    How Reorder Matrix A works:
    1. pid represent the row index of the reordered matrix A and the index for the sorted_token_ids
    2. Using the value from sorted_token_ids (offset_e), you can get the row offset for Matrix A
    3. Using the offset_e, you can check if the value == -1, which will just return
        Since, when the value == -1, the reordered matrix row elements should all be zero, which is already the default. Hence, no need to do anything
   '''
    # Second auto-turner for reorder kernel
    fn_reorder_jit = triton.jit(reorder_kernel)
    fn_reorder_tuned = triton.runtime.Autotuner(
        fn_reorder_jit, 
        fn_reorder_jit.arg_names, 
        reset_to_zero=None, 
        restore_value=None,
        configs=get_moe_fused_autotune_config(block_size_m),
        key=[],
    )

    # Step 2: Reordered Matrix A
    newM = num_tokens_post_padded
    reordered_a = torch.zeros((newM, K_a), device=a.device, dtype=a.dtype)

    # Grid for reorder
    reorder_grid = lambda META: (newM * 1, ) # Grid for reordering matrix A = the row size of matrix reordered_a

    fn_reorder_tuned[reorder_grid](
        a, sorted_token_ids, reordered_a,
        M, newM, K_a,
        a.stride(0), a.stride(1), 
        reordered_a.stride(0), reordered_a.stride(1),
    )

    # Create a reference reordered_a for comparison
    ref_reordered_a = []
    zero_row = torch.zeros((K_a,), device=a.device, dtype=a.dtype)
    for token_id in sorted_token_ids:
        if token_id == -1:
            ref_reordered_a.append(zero_row.clone())
        else:
            ref_reordered_a.append(a[token_id].clone())

    # Check the output of the reordered using the reference
    is_match = False

    ref_reordered_a_tensor = torch.stack(ref_reordered_a)
    if reordered_a.shape == ref_reordered_a_tensor.shape:
        print("Comparing Triton output with PyTorch reference output...")
        are_close = torch.allclose(ref_reordered_a_tensor, reordered_a, atol=1e-5, rtol=1e-3)
        
        if are_close:
            print("✅ SUCCESS: Triton kernel output matches PyTorch reference output.")
            is_match = True
        else:
            print("❌ FAILURE: Triton kernel output does not match PyTorch reference output.")

            print(f"Triton out:\n{reordered_a[:2,:5]}")
            print(f"PyTorch out:\n{ref_reordered_a_tensor[:2,:5]}")

            max_diff = torch.max(torch.abs(ref_reordered_a_tensor - reordered_a)).item()
            print(f"Max absolute difference: {max_diff}")

    # Third auto-tuner for MOE kernel
    fn_moe_jit = triton.jit(fused_moe_kernel)
    fn_moe_tuned = triton.runtime.Autotuner(
        fn_moe_jit, 
        fn_moe_jit.arg_names, 
        reset_to_zero=None, 
        restore_value=None,
        configs=get_moe_fused_autotune_config(block_size_m),  # Configs for MOE kernel with specific block_size_m
        key=[],
    )

    # Grid for MOE kernel
    moe_grid = lambda META: (
        triton.cdiv(num_tokens_post_padded, META['BLOCK_SIZE_M'])
          * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Step 3: Run the fused MOE kernel with transposed B
    fn_moe_tuned[moe_grid](
        a, b_transposed, c, 
        sorted_token_ids, 
        expert_ids_ptr,
        E, N, K_a, num_tokens_post_padded,
        num_valid_tokens,
        a.stride(0), a.stride(1),  # [M, K] strides
        b_transposed.stride(1), b_transposed.stride(2),  # [E, K, N] -> use K and N strides
        c.stride(1), c.stride(2)  # [M, topK, N] strides
    )

def run_autotune_for_all_block_sizes(a, b, topk_ids, E):
    """
    Run autotuning for all block sizes and find the best configuration
    """
    block_sizes_m = [4, 8, 16, 32, 64]
    
    for block_size_m in block_sizes_m:        
        # Prepare data for this block size
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, E
        )

        M, topK = topk_ids.shape
        N = b.shape[1]
        c = torch.zeros((M, topK, N), device=DEVICE, dtype=DTYPE)
                
        benchmark_triton(a, b, c, 
                        sorted_token_ids=sorted_token_ids, 
                        expert_ids_ptr=expert_ids, 
                        num_tokens_post_padded=num_tokens_post_padded, 
                        num_valid_tokens=topk_ids.numel(),
                        block_size_m=block_size_m)

if __name__ == "__main__":
    torch.manual_seed(0)
    
    M = 1000
    E = 8
    K = 128
    N = 512
    topK = 2

    a = torch.randn((M, K), device=DEVICE, dtype=DTYPE)  # 1000 tokens
    b = torch.randn((E, N, K), device=DEVICE, dtype=DTYPE)  # 8 experts, each with K inputs and N outputs

    gating_output = torch.randn((M, E), device=DEVICE)  # Gating scores for each token across experts
    _, topk_ids = torch.topk(gating_output, topK, dim=-1)
    
    # Run autotuning for all block sizes
    run_autotune_for_all_block_sizes(a, b, topk_ids, E)
    
    # sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
    #             topk_ids, block_sizes_m[0], E
    # )
    # c = torch.zeros((M, topK, N), device=DEVICE, dtype=DTYPE)
    
    # benchmark_triton(a, b, c, sorted_token_ids=sorted_token_ids, expert_ids_ptr=expert_ids, num_tokens_post_padded=num_tokens_post_padded, num_valid_tokens=topk_ids.numel())

    # Save input matrices a and b once (they don't change across block sizes)
    # print("Saving input matrices a and b...")
    # save_matrices_to_txt(a, b, 
    #                     output_dir="/home/yuhao/T_RVV/benchmark/auto-tuner/fused_moe/run/test_data", 
    #                     precision=9,)
    
    # # Track results for each block size
    # successful_configs = []
    # failed_configs = []
    
    # print(f"\nTesting {len(block_sizes_m)} different block sizes: {block_sizes_m}")
    # print("=" * 60)
    
    # for block_size_m in block_sizes_m:
    #     print(f"\n--- Testing BLOCK_SIZE_M = {block_size_m} ---")
        
    #     # Create configuration for this block size
    #     config = {
    #         'BLOCK_SIZE_M': block_size_m, 
    #         'BLOCK_SIZE_N': 64, 
    #         'BLOCK_SIZE_K': 32, 
    #         'GROUP_SIZE_M': 8
    #     }
        
    #     try:
    #         # Call moe_align_block_size for this block size
    #         sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
    #             topk_ids, block_size_m, E
    #         )
            
    #         # print(f"moe_align_block_size completed:")
    #         # print(f"  - sorted_token_ids shape: {sorted_token_ids.shape}")
    #         # print(f"  - expert_ids shape: {expert_ids.shape}")
    #         # print(f"  - num_tokens_post_padded: {num_tokens_post_padded}")
            
    #         # Create output tensor for this configuration
    #         c = torch.zeros((M, topK, N), device=DEVICE, dtype=DTYPE)
            
    #         # Run and verify the Triton kernel
    #         triton_out, is_match = run_and_verify_triton_kernel(
    #             a, b, c, sorted_token_ids, expert_ids,
    #             num_tokens_post_padded, topk_ids.numel(), topK, config
    #         )
            
    #         if is_match:
    #             print(f"✅ SUCCESS for BLOCK_SIZE_M = {block_size_m}")
    #             successful_configs.append(block_size_m)
                
    #             # Save the successful configuration data
    #             # output_dir = f"/home/yuhao/T_RVV/benchmark/auto-tuner/fused_moe/run/test_data"
    #             # print(f"Saving results for BLOCK_SIZE_M = {block_size_m} to {output_dir}...")
                
    #             # save_matrices_to_txt(
    #             #     triton_out,           # Output from Triton kernel
    #             #     start_idx=3,
    #             #     output_dir=output_dir,
    #             #     prefix=f"matrix_BLOCK_SIZE_M_{block_size_m}",
    #             # )
                
    #             # save_matrices_to_txt(
    #             #     sorted_token_ids,     # Processed token IDs
    #             #     expert_ids,           # Expert IDs for blocks
    #             #     start_idx=4,
    #             #     dtype="int",
    #             #     output_dir=output_dir,
    #             #     prefix=f"matrix_BLOCK_SIZE_M_{block_size_m}",
    #             # )

    #         else:
    #             print(f"❌ FAILURE for BLOCK_SIZE_M = {block_size_m}")
    #             failed_configs.append(block_size_m)
                
    #     except Exception as e:
    #         print(f"❌ ERROR for BLOCK_SIZE_M = {block_size_m}: {str(e)}")
    #         failed_configs.append(block_size_m)
    #         import traceback
    #         traceback.print_exc()
    
    # # Final summary
    # print("\n" + "=" * 60)
    # print("FINAL SUMMARY")
    # print("=" * 60)
    
    # if successful_configs:
    #     print(f"✅ SUCCESSFUL configurations (BLOCK_SIZE_M): {successful_configs}")
    #     print(f"   Data saved for {len(successful_configs)} configurations")
    # else:
    #     print("❌ No configurations were successful")
    
    # if failed_configs:
    #     print(f"❌ FAILED configurations (BLOCK_SIZE_M): {failed_configs}")