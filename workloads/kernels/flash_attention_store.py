import torch
import triton
import triton.language as tl
import os
import math
import numpy as np # Needed for save_tensor_to_txt

# --- Configuration ---
DEVICE = 'cpu'
DTYPE_TRITON_KERNEL = tl.float32
DTYPE_TORCH_INPUT = torch.float32
triton.runtime.driver.set_active_to_cpu()


@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    M_LogSumExp_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_vz, stride_vh, stride_vm, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX_Q, N_CTX_K, D_HEAD,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    start_m_idx = tl.program_id(axis=0)
    off_zh_idx = tl.program_id(axis=1)
    off_z = off_zh_idx // H
    off_h = off_zh_idx % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    out_offset = off_z * stride_oz + off_h * stride_oh
    m_logsumexp_offset = off_z * H * N_CTX_Q + off_h * N_CTX_Q

    Q_batch_head_ptr = Q_ptr + q_offset
    K_batch_head_ptr = K_ptr + k_offset
    V_batch_head_ptr = V_ptr + v_offset
    Out_batch_head_ptr = Out_ptr + out_offset
    M_LogSumExp_batch_head_ptr = M_LogSumExp_ptr + m_logsumexp_offset

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=DTYPE_TRITON_KERNEL)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=DTYPE_TRITON_KERNEL)
    l_i = tl.zeros((BLOCK_M,), dtype=DTYPE_TRITON_KERNEL)

    offs_m_q = start_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_q_rows = offs_m_q < N_CTX_Q

    q_ptrs = Q_batch_head_ptr + offs_m_q[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_block = tl.load(q_ptrs, mask=mask_q_rows[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)
    q_block = q_block.to(DTYPE_TRITON_KERNEL)

    for start_n_idx in range(0, tl.cdiv(N_CTX_K, BLOCK_N)):
        offs_m_k_block = start_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        k_ptrs = K_batch_head_ptr + offs_d[:, None] * stride_kk + offs_m_k_block[None, :] * stride_km
        mask_k_cols = offs_m_k_block < N_CTX_K
        k_block = tl.load(k_ptrs, mask=(offs_d[:, None] < D_HEAD) & mask_k_cols[None, :], other=0.0)
        k_block = k_block.to(DTYPE_TRITON_KERNEL)

        qk_scores = tl.dot(q_block, k_block) * sm_scale
        qk_scores = qk_scores.to(DTYPE_TRITON_KERNEL)

        if IS_CAUSAL:
            causal_mask = offs_m_q[:, None] >= offs_m_k_block[None, :]
            qk_scores += tl.where(causal_mask, 0, float("-inf"))

        m_i_prev = m_i
        m_i_block_max = tl.max(qk_scores, axis=1)
        m_i = tl.maximum(m_i, m_i_block_max)

        p_block = tl.exp(qk_scores - m_i[:, None])
        p_block = p_block.to(DTYPE_TRITON_KERNEL)

        alpha = tl.exp(m_i_prev - m_i)
        alpha = alpha.to(DTYPE_TRITON_KERNEL)
        
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        v_ptrs = V_batch_head_ptr + offs_m_k_block[:, None] * stride_vm + offs_d[None, :] * stride_vk
        mask_v_rows = offs_m_k_block < N_CTX_K # This should be mask_k_cols or similar for V
        v_block = tl.load(v_ptrs, mask=mask_v_rows[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)
        v_block = v_block.to(DTYPE_TRITON_KERNEL)
        
        acc = tl.dot(p_block.to(v_block.dtype), v_block, acc)
        l_i += tl.sum(p_block, axis=1)

    acc = acc / l_i[:, None]
    acc = acc.to(Q_ptr.dtype.element_ty)

    out_ptrs = Out_batch_head_ptr + offs_m_q[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=mask_q_rows[:, None] & (offs_d[None, :] < D_HEAD))

    final_logsumexp = m_i + tl.log(l_i)
    m_logsumexp_store_ptrs = M_LogSumExp_batch_head_ptr + offs_m_q
    tl.store(m_logsumexp_store_ptrs, final_logsumexp, mask=mask_q_rows)


def flash_attention_fwd(q, k, v, sm_scale, is_causal=False):
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]

    Z, H, N_CTX_Q, D_HEAD = q.shape
    _Z_k, _H_k, N_CTX_K, _D_HEAD_k = k.shape

    q = q.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    k = k.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    v = v.contiguous().to(device=DEVICE, dtype=DTYPE_TORCH_INPUT)

    output = torch.empty_like(q)
    M_LogSumExp = torch.empty((Z, H, N_CTX_Q), device=DEVICE, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    if D_HEAD > 128: 
        BLOCK_M = 32
        BLOCK_N = 32
    elif D_HEAD <= 32: 
        BLOCK_M = 128
        BLOCK_N = 128

    grid = (triton.cdiv(N_CTX_Q, BLOCK_M), Z * H)
    
    print(f"FlashAttention Fwd Kernel Grid: {grid}, BLOCK_M: {BLOCK_M}, BLOCK_N: {BLOCK_N}, BLOCK_DMODEL: {D_HEAD}")

    flash_attention_fwd_kernel[grid](
        q, k, v, output, M_LogSumExp,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        Z, H, N_CTX_Q, N_CTX_K, D_HEAD,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D_HEAD,
        IS_CAUSAL=is_causal
    )
    return output, M_LogSumExp

def pytorch_attention(q, k, v, sm_scale, is_causal=False):
    k_t = k.transpose(-2, -1)
    scores = (q @ k_t) * sm_scale
    if is_causal:
        mask_value = -float('inf') if scores.dtype == torch.float32 else -1e4
        rows, cols = q.size(2), k.size(2)
        # Correct causal mask generation:
        # For each query position i (0 to rows-1), it can attend to key positions j (0 to cols-1) where j <= i.
        # So, we need a mask that is True where j <= i.
        # `torch.ones(rows, cols).tril(0)` creates a lower triangular matrix (including diagonal).
        # If cols (N_CTX_K) > rows (N_CTX_Q), tril will apply to the min(rows,cols) square.
        # We need to ensure the mask is broadcastable to (B,H,Sq,Sk).
        # A simple way for q_idx >= k_idx is:
        query_indices = torch.arange(rows, device=q.device).unsqueeze(1) # Shape (Sq, 1)
        key_indices = torch.arange(cols, device=q.device).unsqueeze(0)   # Shape (1, Sk)
        causal_mask_bool = query_indices >= key_indices                  # Shape (Sq, Sk)
        
        scores = scores.masked_fill(~causal_mask_bool[None, None, :, :], mask_value)

    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ v
    return output, attn_weights

# --- Function to save tensors in the C++ expected format ---
def save_tensor_to_txt(tensor, filename, fmt='%.8f', is_matrix=False, rows=0, cols=0):
    """Saves a PyTorch tensor to a text file."""
    print(f"Saving tensor to {filename}...")
    if not isinstance(tensor, torch.Tensor):
        print(f"ERROR: Input is not a PyTorch tensor. Got type: {type(tensor)}")
        return
    data = tensor.detach().cpu().numpy()
    try:
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        with open(filename, 'w') as f:
            if is_matrix and rows > 0 and cols > 0:
                f.write(f"{rows} {cols}\n")
                if data.shape != (rows, cols) and not (data.ndim == 1 and rows == data.shape[0] and cols == 1): # check for 1D array saved as Mx1
                     print(f"Warning: Tensor shape {data.shape} does not match provided rows={rows}, cols={cols} for file header in {filename}.")
                
                current_rows, current_cols = data.shape if data.ndim == 2 else (data.shape[0], 1) if data.ndim == 1 else (0,0)

                for i in range(current_rows):
                    row_data = data[i] if data.ndim == 2 else [data[i]] if data.ndim == 1 else []
                    if hasattr(row_data, '__iter__'):
                        row_str = ' '.join([fmt % val for val in row_data])
                    else:
                        row_str = fmt % row_data
                    f.write(row_str + '\n')

            elif not is_matrix and data.ndim == 1:
                for val in data:
                    f.write((fmt % val) + '\n')
            elif data.ndim == 0:
                 f.write(fmt % data.item() + '\n')
            else: # Fallback for other cases
                print(f"Warning: Using generic np.savetxt for tensor shape {data.shape} into {filename}")
                np.savetxt(f, data.reshape(-1, data.shape[-1]) if data.ndim > 1 else data, fmt=fmt)
        print(f"Tensor saved successfully to {filename}")
    except Exception as e:
        print(f"ERROR: Failed to save tensor to {filename}: {e}")
        import traceback
        traceback.print_exc()

# --- Testing and Data Saving ---
if __name__ == "__main__":
    torch.manual_seed(0)

    Z_test, H_test, N_CTX_Q_test, N_CTX_K_test, D_HEAD_test = 1, 2, 64, 64, 32 # Smaller for faster testing
    sm_scale_test = 1.0 / math.sqrt(D_HEAD_test)

    q_torch = torch.randn((Z_test, H_test, N_CTX_Q_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    k_torch = torch.randn((Z_test, H_test, N_CTX_K_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)
    v_torch = torch.randn((Z_test, H_test, N_CTX_K_test, D_HEAD_test), device=DEVICE, dtype=DTYPE_TORCH_INPUT)

    print(f"Testing FlashAttention with Z={Z_test}, H={H_test}, N_CTX_Q={N_CTX_Q_test}, N_CTX_K={N_CTX_K_test}, D_HEAD={D_HEAD_test}")

    for is_causal_test in [False, True]:
        test_type = "Causal" if is_causal_test else "Non-Causal"
        print(f"\n--- {test_type} Attention Test ---")
        
        triton_output, pytorch_output = None, None # Initialize
        try:
            triton_output, _ = flash_attention_fwd(
                q_torch.clone(), k_torch.clone(), v_torch.clone(), sm_scale_test, is_causal=is_causal_test
            )
            pytorch_output, _ = pytorch_attention(
                q_torch.clone(), k_torch.clone(), v_torch.clone(), sm_scale_test, is_causal=is_causal_test
            )

            atol = 1e-5 if DTYPE_TORCH_INPUT == torch.float32 else 1e-2
            rtol = 1e-3 if DTYPE_TORCH_INPUT == torch.float32 else 1e-2
            
            print(f"Triton out ({test_type}, first sample, first head, first 2 queries, first 5 features):\n{triton_output[0,0,:2,:5]}")
            print(f"PyTorch out ({test_type}, first sample, first head, first 2 queries, first 5 features):\n{pytorch_output[0,0,:2,:5]}")

            all_close = torch.allclose(triton_output, pytorch_output, atol=atol, rtol=rtol)
            if all_close:
                print(f"✅ {test_type}: Triton and PyTorch match.")
            else:
                print(f"❌ {test_type}: Triton and PyTorch differ.")
                print(f"   Max diff: {torch.max(torch.abs(triton_output - pytorch_output))}")
        except Exception as e:
            print(f"ERROR during {test_type} test: {e}")
            import traceback
            traceback.print_exc()

        # --- Save test data files for this configuration (Causal/Non-Causal) ---
        if pytorch_output is not None: # Only save if reference generation was successful
            print(f"\n--- Saving test data for C++ benchmark ({test_type}) ---")
            causal_suffix = "_causal" if is_causal_test else "_noncausal"
            output_dir = f"./test_data_flash_attention{causal_suffix}"
            os.makedirs(output_dir, exist_ok=True)
            DB_PREFIX = os.path.join(output_dir, "tensor")

            # Flatten tensors for saving as 1D arrays (as C++ driver expects with readLoss)
            q_flat_elements = q_torch.numel()
            k_flat_elements = k_torch.numel()
            v_flat_elements = v_torch.numel()
            out_flat_elements = pytorch_output.numel()

            # Naming convention: tensor_NUMELEMENTS_INDEX.txt
            input_q_filename = f"{DB_PREFIX}_{q_flat_elements}_1.txt"
            input_k_filename = f"{DB_PREFIX}_{k_flat_elements}_2.txt"
            input_v_filename = f"{DB_PREFIX}_{v_flat_elements}_3.txt"
            ref_output_filename = f"{DB_PREFIX}_{out_flat_elements}_4.txt" # Using PyTorch output as reference

            save_tensor_to_txt(q_torch.flatten(), input_q_filename, is_matrix=False)
            save_tensor_to_txt(k_torch.flatten(), input_k_filename, is_matrix=False)
            save_tensor_to_txt(v_torch.flatten(), input_v_filename, is_matrix=False)
            save_tensor_to_txt(pytorch_output.flatten(), ref_output_filename, is_matrix=False)

            print(f"\nGenerated test data files in {output_dir}/ for FlashAttention ({test_type}):")
            print(f"  Input Q:   {input_q_filename}")
            print(f"  Input K:   {input_k_filename}")
            print(f"  Input V:   {input_v_filename}")
            print(f"  Ref Out:   {ref_output_filename}")
            print(f"\nTo use these files with your C++ benchmark ({test_type}):")
            print(f"1. Transfer the '{output_dir}' directory to your RISC-V target machine.")
            print(f"2. Set the DB_FILE environment variable on the target, e.g.:")
            print(f"   export DB_FILE=\"/path/to/your/transferred/{output_dir}/tensor\"")
            print(f"3. Ensure your C++ 'flash_attention_kernel.cpp' uses getDB with correct SHAPE string and indices:")
            print(f"   - For Q:   getDB(\"{q_flat_elements}\", 1)")
            print(f"   - For K:   getDB(\"{k_flat_elements}\", 2)")
            print(f"   - For V:   getDB(\"{v_flat_elements}\", 3)")
            print(f"   - For Out: getDB(\"{out_flat_elements}\", 4)")
            print(f"   And ensure C++ reads them as flattened 1D arrays.")

    print("\nCPU FlashAttention forward pass test run finished.")
    print("Backward pass is not implemented. Performance will differ significantly from GPU.")