import torch
import triton
import triton.language as tl
import math

'''The name of the kernel must match the file name for Triton to find it.'''
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, # Tensor pointers
    M_LogSumExp_ptr, # Pointer to store m_i and l_i (combined for stability)
    
    stride_qz, stride_qh, stride_qm, stride_qk, # Strides for Q
    stride_kz, stride_kh, stride_km, stride_kk, # Strides for K (km for N_CTX)
    stride_vz, stride_vh, stride_vm, stride_vk, # Strides for V (vm for N_CTX)
    stride_oz, stride_oh, stride_om, stride_ok, # Strides for Out

    Z, H, N_CTX, D_HEAD, # Dimensions: Batch, Heads, SeqLenQ, SeqLenK, HeadDim
    sm_scale, # Softmax scale factor (usually 1/sqrt(D_HEAD))
    
    BLOCK_M: tl.constexpr, # Tile size for Q sequence length dimension
    BLOCK_N: tl.constexpr, # Tile size for K sequence length dimension
    BLOCK_DMODEL: tl.constexpr, # Head dimension (must be == D_HEAD)
    IS_CAUSAL: tl.constexpr
):
    # Program IDs
    start_m_idx = tl.program_id(axis=0) # Index of the current BLOCK_M tile along N_CTX
    off_zh_idx = tl.program_id(axis=1)  # Combined batch and head index

    off_z = off_zh_idx // H
    off_h = off_zh_idx % H

    # Calculate offsets for Q, K, V, Out based on batch and head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    out_offset = off_z * stride_oz + off_h * stride_oh
    m_logsumexp_offset = off_z * H * N_CTX + off_h * N_CTX # Assuming M_LogSumExp is [Z, H, N_CTX]

    # Pointers to Q, K, V, Out for the current batch/head
    Q_batch_head_ptr = Q_ptr + q_offset
    K_batch_head_ptr = K_ptr + k_offset
    V_batch_head_ptr = V_ptr + v_offset
    Out_batch_head_ptr = Out_ptr + out_offset
    M_LogSumExp_batch_head_ptr = M_LogSumExp_ptr + m_logsumexp_offset


    # Initialize accumulators and stats for the current Q_block
    # tl.zeros needs a shape, so if BLOCK_M or BLOCK_DMODEL is 1, handle carefully.
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=DTYPE_TRITON_KERNEL)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=DTYPE_TRITON_KERNEL) # Running max
    l_i = tl.zeros((BLOCK_M,), dtype=DTYPE_TRITON_KERNEL)              # Running sum of exps

    # Offsets for the current Q block (BLOCK_M rows of queries)
    offs_m_q = start_m_idx * BLOCK_M + tl.arange(0, BLOCK_M) # Rows in Q to process
    offs_d = tl.arange(0, BLOCK_DMODEL)                 # Columns (head dimension)

    # Load current Q block
    # Q_block_ptr: base, shape, strides, offsets, block_shape, order
    q_ptrs = Q_batch_head_ptr + offs_m_q[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Mask for Q if N_CTX is not a multiple of BLOCK_M
    mask_q_rows = offs_m_q < N_CTX
    q_block = tl.load(q_ptrs, mask=mask_q_rows[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)
    q_block = q_block.to(DTYPE_TRITON_KERNEL)


    # Loop over K and V blocks (N_CTX dimension)
    # N_CTX is the sequence length of K and V
    for start_n_idx in range(0, tl.cdiv(N_CTX, BLOCK_N)):
        offs_m_k = start_n_idx * BLOCK_N + tl.arange(0, BLOCK_N) # Rows in K/V to process in this block

        # Load K block (transposed for dot product with Q)
        # K is [D_HEAD, N_CTX] effectively for this head after transposing for QK^T
        # K_block will be [BLOCK_DMODEL, BLOCK_N]
        k_ptrs = K_batch_head_ptr + offs_d[:, None] * stride_kk + offs_m_k[None, :] * stride_km
        # Mask for K if N_CTX is not a multiple of BLOCK_N
        mask_k_cols = offs_m_k < N_CTX
        k_block = tl.load(k_ptrs, mask=(offs_d[:, None] < D_HEAD) & mask_k_cols[None, :], other=0.0)
        k_block = k_block.to(DTYPE_TRITON_KERNEL)

        # Compute QK^T score for the block: (BLOCK_M, D_MODEL) @ (D_MODEL, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        qk_scores = tl.dot(q_block, k_block) * sm_scale
        qk_scores = qk_scores.to(DTYPE_TRITON_KERNEL)

        # Apply causal mask if needed
        if IS_CAUSAL:
            # offs_m_q are global query indices for current Q block
            # offs_m_k are global key indices for current K block
            # For causality, q_idx >= k_idx
            causal_mask = offs_m_q[:, None] >= offs_m_k[None, :]
            qk_scores += tl.where(causal_mask, 0, float("-inf"))

        # Online softmax update
        m_i_prev = m_i
        m_i_block_max = tl.max(qk_scores, axis=1) # Max score for each query in Q_block against current K_block
        m_i = tl.maximum(m_i, m_i_block_max)    # Update running max

        # Numerator: p_ij = exp(qk_scores - m_i_new)
        # Corrected for Triton: p_ij is actually exp(scores_ij - m_ij)
        # where m_ij is the max of current qk_scores and m_i_old
        # Let's use the standard FlashAttention update:
        p_block = tl.exp(qk_scores - m_i[:, None]) # Subtract new running max, m_i is (BLOCK_M,)
        p_block = p_block.to(DTYPE_TRITON_KERNEL)

        # Denominator update: l_i_new = l_i_old * exp(m_i_old - m_i_new) + sum(p_block_new_m, axis=1)
        alpha = tl.exp(m_i_prev - m_i) # m_i_prev is (BLOCK_M,), m_i is (BLOCK_M,)
        alpha = alpha.to(DTYPE_TRITON_KERNEL)
        
        acc = acc * alpha[:, None] # Scale existing accumulator
        l_i = l_i * alpha          # Scale existing l_i

        # Load V block
        # V_block will be [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = V_batch_head_ptr + offs_m_k[:, None] * stride_vm + offs_d[None, :] * stride_vk
        mask_v_rows = offs_m_k < N_CTX
        v_block = tl.load(v_ptrs, mask=mask_v_rows[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)
        v_block = v_block.to(DTYPE_TRITON_KERNEL) # Or Q.dtype.element_ty for dot product type

        # Update accumulator: acc += P_block @ V_block
        # p_block is (BLOCK_M, BLOCK_N), v_block is (BLOCK_N, BLOCK_DMODEL)
        acc = tl.dot(p_block.to(v_block.dtype), v_block, acc) # Add to scaled accumulator
        
        l_i += tl.sum(p_block, axis=1) # Update l_i with sum of current p_block

    # Final normalization
    acc = acc / l_i[:, None]
    acc = acc.to(Q_ptr.dtype.element_ty) # Cast to output type

    # Store output block
    out_ptrs = Out_batch_head_ptr + offs_m_q[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=mask_q_rows[:, None] & (offs_d[None, :] < D_HEAD))

    # Store m_i (max for stability) and l_i (logsumexp for backward) if needed
    # Here, storing m_i + log(l_i) which is the log of the true denominator (logsumexp)
    # For simplicity if M_LogSumExp_ptr is only for m_i for stability for a fwd-only test,
    # we can just store m_i. If it's for backward, logsumexp is needed.
    # The original Triton code stores m_i + tl.math.log2(l_i) when using exp2.
    # For tl.exp, it would be m_i + tl.log(l_i)
    final_logsumexp = m_i + tl.log(l_i)
    m_logsumexp_store_ptrs = M_LogSumExp_batch_head_ptr + offs_m_q
    tl.store(m_logsumexp_store_ptrs, final_logsumexp, mask=mask_q_rows)


def get_flash_attention_fwd_kernel_autotune_config(num_threads=0):
    configs = []
    
    block_sizes_M = [4, 8]
    block_sizes_N = [8]
    block_sizes_DMODEL = [8]

    # Generate unique total block sizes from all combinations
    total_block_sizes = set()
    for block_m in block_sizes_M:
        for block_n in block_sizes_N:
            for block_dmodel in block_sizes_DMODEL:
                total_block_sizes.add(block_m * block_n * block_dmodel)
    
    # Sort the total block sizes for systematic exploration
    total_block_sizes = sorted(list(total_block_sizes))
    
    # For each total block size, find all combinations that achieve it
    for total_size in total_block_sizes:
        for block_m in block_sizes_M:
            for block_n in block_sizes_N:
                # Calculate the required block_k
                if total_size % (block_m * block_n) == 0:
                    block_dmodel = total_size // (block_m * block_n)
                    # Only include if block_k is a valid block size
                    if block_dmodel in block_sizes_DMODEL:
                        #print(f"Config: BLOCK_SIZE_M={block_m}, BLOCK_SIZE_N={block_n}, BLOCK_SIZE_K={block_k}")
                        configs.append(
                            triton.Config({
                                'BLOCK_M': block_m,
                                'BLOCK_N': block_n,
                                'BLOCK_DMODEL': block_dmodel,
                                'IS_CAUSAL': False
                            }, num_threads=num_threads)
                        )
    
    return configs


def benchmark_triton(q, k, v, sm_scale, parallel=False):
    fn = flash_attention_fwd_kernel
    fn_jit = triton.jit(fn)
    fn_jit_tuned = triton.runtime.Autotuner(fn_jit, fn_jit.arg_names, 
        reset_to_zero=None, 
        restore_value=None,
        configs=get_flash_attention_fwd_kernel_autotune_config(0 if parallel else 1),
        key=[],
    )

    # Ensure inputs are 4D: (Batch, Heads, SeqLen, HeadDim)
    assert q.dim() == k.dim() == v.dim() == 4

    Z,   H,   N_CTX_q, D_HEAD   = q.shape
    Z_k, H_k, N_CTX_k, D_HEAD_k = k.shape
    Z_v, H_v, N_CTX_v, D_HEAD_v = v.shape

    # Check that dimensions match where they should
    assert Z == Z_k == Z_v and H == H_k == H_v and D_HEAD == D_HEAD_k == D_HEAD_v
    assert N_CTX_k == N_CTX_v  # SeqLen_K == SeqLen_V

    # Convert to Triton-compatible device and dtype
    q = q.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    k = k.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    v = v.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)

    output = torch.empty_like(q)
    # M_LogSumExp is used for numerical stability checks or backward pass
    M_LogSumExp = torch.empty((Z, H, N_CTX_q), device=DEVICE, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(N_CTX_q, META["BLOCK_M"]), Z * H,
    )

    def run_triton_kernel():
        # don't need to include the Meta parameters in the call
        # to the kernel, they are already included in the config
        fn_jit_tuned[grid](
            q, k, v, output, M_LogSumExp,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            Z, H, N_CTX_q, D_HEAD,
            sm_scale
        )

    run_triton_kernel() # generate IR for all configs


# def flash_attention_fwd(q, k, v, sm_scale, is_causal=False):
#     # Ensure inputs are 4D: (Batch, Heads, SeqLen, HeadDim)
#     assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
#     assert q.shape[-1] == k.shape[-1] == v.shape[-1] # D_HEAD matches
#     assert k.shape[2] == v.shape[2] # SeqLen_K == SeqLen_V

#     Z, H, N_CTX, D_HEAD = q.shape
#     _Z_k, _H_k, N_CTX_K, _D_HEAD_k = k.shape
#     _Z_v, _H_v, _N_CTX_K_v, _D_HEAD_v = v.shape

#     assert Z == _Z_k == _Z_v and H == _H_k == _H_v and D_HEAD == _D_HEAD_k == _D_HEAD_v

#     # Convert to Triton-compatible device and dtype
#     q = q.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
#     k = k.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
#     v = v.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)

#     output = torch.empty_like(q)
#     # M_LogSumExp is used for numerical stability checks or backward pass
#     M_LogSumExp = torch.empty((Z, H, N_CTX), device=DEVICE, dtype=torch.float32)


#     # Block sizes - these need tuning for CPU.
#     # BLOCK_M and BLOCK_N are tile sizes along sequence lengths.
#     # BLOCK_DMODEL must be D_HEAD.
#     BLOCK_M = 64  # Tile size for Q sequence length
#     BLOCK_N = 64  # Tile size for K sequence length
#     if D_HEAD > 128: # Example simple heuristic
#         BLOCK_M = 32
#         BLOCK_N = 32
#     elif D_HEAD <=32:
#         BLOCK_M = 128
#         BLOCK_N = 128


#     # Grid: (num_q_blocks, num_batch_heads)
#     grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

#     print(f"Grid: {grid}, BLOCK_M: {BLOCK_M}, BLOCK_N: {BLOCK_N}, BLOCK_DMODEL: {D_HEAD}")
#     print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}, Out: {output.shape}")
#     print(f"Strides Q: {q.stride()}, K: {k.stride()}, V: {v.stride()}, Out: {output.stride()}")


#     flash_attention_fwd_kernel[grid](
#         q, k, v, output, M_LogSumExp,
#         q.stride(0), q.stride(1), q.stride(2), q.stride(3),
#         k.stride(0), k.stride(1), k.stride(2), k.stride(3),
#         v.stride(0), v.stride(1), v.stride(2), v.stride(3),
#         output.stride(0), output.stride(1), output.stride(2), output.stride(3),
#         Z, H, N_CTX, D_HEAD,
#         sm_scale,
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D_HEAD,
#         IS_CAUSAL=is_causal
#         # num_warps and num_stages are GPU-specific, not used for CPU
#     )
#     return output, M_LogSumExp


# PyTorch reference implementation for testing
# def pytorch_attention(q, k, v, sm_scale, is_causal=False):
#     # q, k, v: (Batch, Heads, SeqLen, HeadDim)
#     # Transpose K for matmul: (B, H, D_HEAD, SeqLen_K)
#     k_t = k.transpose(-2, -1)
#     # Scores: (B, H, SeqLen_Q, SeqLen_K)
#     scores = (q @ k_t) * sm_scale

#     if is_causal:
#         # Create a causal mask
#         # Mask elements where key_idx > query_idx
#         mask_value = -float('inf') if scores.dtype == torch.float32 else -1e4 # Approx -inf for float16
#         rows = q.size(2)
#         cols = k.size(2)
#         causal_mask = torch.ones(rows, cols, dtype=torch.bool, device=q.device).tril(diagonal=0)
#         # Expand mask to match scores shape (B, H, Sq, Sk)
#         # For tril, if Sq != Sk, it applies to the min(Sq, Sk) x min(Sq,Sk) bottom-left submatrix
#         # Here we want to mask where key_pos > query_pos
#         # query_pos is indexed by M (rows of scores), key_pos by N (cols of scores)
#         # So we want M_indices < N_indices to be masked. tril(0) gives M_indices >= N_indices
#         scores = scores.masked_fill(~causal_mask[None, None, :, :], mask_value)

#     attn_weights = torch.softmax(scores, dim=-1)
#     output = attn_weights @ v
#     return output, attn_weights # Return weights for potential inspection



if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = 'cpu'
    DTYPE_TRITON_KERNEL = tl.float32 # Kernel internal computation dtype
    DTYPE_TORCH_INPUT = torch.float32  # Dtype for torch tensors
    triton.runtime.driver.set_active_to_cpu()

    # Test parameters
    Z_test = 1
    H_test = 2
    N_CTX_test = 128 # Sequence length
    D_HEAD_test = 64
    sm_scale_test = 1.0 / math.sqrt(D_HEAD_test)

    q_torch = torch.randn((Z_test, H_test, N_CTX_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    k_torch = torch.randn((Z_test, H_test, N_CTX_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    v_torch = torch.randn((Z_test, H_test, N_CTX_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)

    benchmark_triton(q_torch, k_torch, v_torch, sm_scale_test)
