import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# --- Configuration ---
DTYPE = torch.float32
DEVICE = 'cpu'

triton.runtime.driver.set_active_to_cpu()

# --- Triton Kernel ---
@triton.jit
def cross_entropy_kernel(
    logits_ptr,       # Pointer to input logits (BatchSize, NumClasses)
    labels_ptr,       # Pointer to input labels (BatchSize,)
    loss_ptr,         # Pointer to output loss per sample (BatchSize,)
    M,                # BatchSize
    N,                # NumClasses
    stride_logits_m,  # Row stride for logits
    stride_logits_n,  # Column stride for logits
    stride_labels,    # Stride for labels (usually 1)
    stride_loss,      # Stride for loss (usually 1)
    BLOCK_SIZE_N: tl.constexpr, # Block size for the N dimension
):
    pid = tl.program_id(axis=0) # Corresponds to row index m

    # --- Load Label ---
    label_ptr = labels_ptr + pid * stride_labels
    label = tl.load(label_ptr) # Load the true label for this sample

    # --- Compute LogSumExp ---
    row_start_ptr = logits_ptr + pid * stride_logits_m
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    logit_ptrs = row_start_ptr + offs_n * stride_logits_n
    mask = offs_n < N

    # Load logits and IMMEDIATELY cast to float32
    row_logits = tl.load(logit_ptrs, mask=mask, other=-float('inf'))
    # row_logits = row_logits.to(tl.float32) # <<< FORCE float32

    # Calculations
    row_max = tl.max(row_logits, axis=0)
    # row_max = row_max.to(tl.float32) # <<< FORCE float32 for max value
    logits_minus_max = row_logits - row_max # Should be float32 - float32

    # Ensure input to exp is float32 (redundant but safe)
    # numerator = tl.exp(logits_minus_max.to(tl.float32))
    numerator = tl.exp(logits_minus_max)
    # Ensure numerator is float32 (usually inherits type from exp)
    # numerator = numerator.to(tl.float32)
    denominator = tl.sum(numerator, axis=0)
    # Ensure denominator and input to log are float32
    # denominator = denominator.to(tl.float32) # <<< FORCE float32 for sum
    log_sum_exp = row_max + tl.log(denominator) # float32 + tl.log(float32)
    # log_sum_exp = log_sum_exp.to(tl.float32) # <<< FORCE float32 for result

    # --- Compute NLL Loss ---
    target_logit_ptr = row_start_ptr + label * stride_logits_n
    target_logit = tl.load(target_logit_ptr)
    # target_logit = target_logit.to(tl.float32) # <<< FORCE float32

    # Final calculation
    log_softmax_target = target_logit - log_sum_exp # float32 - float32
    nll_loss = -log_softmax_target
    # nll_loss = nll_loss.to(tl.float32) # <<< FORCE float32 before store

    # --- Store Loss ---
    loss_out_ptr = loss_ptr + pid * stride_loss
    # Store the explicitly float32 value
    tl.store(loss_out_ptr, nll_loss)


# --- Python Wrapper for Triton Kernel ---
def cross_entropy(logits: torch.Tensor, labels: torch.Tensor):
    """
    Python wrapper to launch the Triton cross-entropy kernel.

    Args:
        logits: Raw model output tensor (BatchSize, NumClasses), float type.
        labels: True class labels tensor (BatchSize,), long type (int64).

    Returns:
        Scalar tensor representing the mean cross-entropy loss over the batch.
    """
    M, N = logits.shape # M = BatchSize, N = NumClasses

    # Input validation (using torch tensor properties)
    assert logits.is_contiguous(), "Logits tensor must be contiguous"
    assert labels.is_contiguous(), "Labels tensor must be contiguous"
    assert logits.dim() == 2, "Logits must be 2D (BatchSize, NumClasses)"
    assert labels.dim() == 1, "Labels must be 1D (BatchSize,)"
    assert M == labels.shape[0], f"Batch size mismatch: {M} vs {labels.shape[0]}"
    # Note: Label range check (0 <= label < N) is implicitly handled by PyTorch's F.cross_entropy
    # and expected for the kernel logic to be correct.

    # Allocate intermediate tensor for per-sample losses (using torch)
    loss_per_sample = torch.empty((M,), dtype=logits.dtype, device=logits.device)

    # Kernel launch configuration
    grid = (M,) # Launch one program instance per sample

    # Determine block size for the N dimension
    # Use next power of 2 for efficiency, Triton handles masking
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    # Launch the kernel
    cross_entropy_kernel[grid](
        logits, labels, loss_per_sample,
        M, N,
        logits.stride(0), logits.stride(1), # Strides for logits
        labels.stride(0),                   # Stride for labels
        loss_per_sample.stride(0),          # Stride for output loss tensor
        BLOCK_SIZE_N=BLOCK_SIZE_N,          # Pass block size as constexpr
        # num_warps=4                       # Optional: tuning parameter
    )

    # Final reduction (mean) performed using torch after kernel execution
    final_loss = torch.mean(loss_per_sample)
    return final_loss

# --- Testing ---
torch.manual_seed(0)

# Test parameters
BATCH_SIZE = 128
NUM_CLASSES = 1000 # Number of classes

# Create input data on the target device
logits = torch.randn((BATCH_SIZE, NUM_CLASSES), device=DEVICE, dtype=DTYPE, requires_grad=False)
labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE, dtype=torch.long) # Labels must be LongTensor

# Ensure tensors are on the correct device (redundant if created there, but good practice)
logits = logits.to(DEVICE)
labels = labels.to(DEVICE)

# --- Triton Calculation ---
triton_output = cross_entropy(logits, labels)

# --- PyTorch Calculation ---
# PyTorch handles softmax and NLL loss internally.
# Use float32 for PyTorch calculation for stable comparison.
torch_output = F.cross_entropy(logits.to(torch.float32), labels, reduction='mean')

print(f"Triton Output ({DEVICE}, {logits.dtype}): {triton_output}")
print(f"PyTorch Output ({DEVICE}, float32): {torch_output}")

# --- Comparison ---
# Use tolerance due to potential minor floating-point differences
atol = 1e-5 # Absolute tolerance
rtol = 1e-4 # Relative tolerance

# Compare outputs (convert Triton output to float32 for comparison consistency)
are_close = torch.allclose(triton_output.to(torch.float32), torch_output, atol=atol, rtol=rtol)

if are_close:
    print(f"✅ Triton ({DEVICE}) and PyTorch ({DEVICE}) match within tolerance (atol={atol}, rtol={rtol})")
else:
    diff = torch.abs(triton_output.to(torch.float32) - torch_output)
    print(f"❌ Triton ({DEVICE}) and PyTorch ({DEVICE}) differ.")
    print(f"   Max difference: {torch.max(diff)}")
    print(f"   Mean difference: {torch.mean(diff)}")


# # 保存test data
# import numpy as np

# # --- Function to save tensors in the C++ expected format ---
# def save_tensor_to_txt(tensor, filename, fmt='%.8f', is_matrix=False, rows=0, cols=0):
#     """Saves a PyTorch tensor to a text file."""
#     print(f"Saving tensor to {filename}...")
#     data = tensor.cpu().numpy()
#     with open(filename, 'w') as f:
#         if is_matrix and rows > 0 and cols > 0: # For 2D matrices like logits
#             f.write(f"{rows} {cols}\n") # Write dimensions if support.cpp readMatrix expects it
#             for i in range(rows):
#                 row_str = ' '.join([fmt % val for val in data[i]])
#                 f.write(row_str + '\n')
#         elif not is_matrix and tensor.dim() == 1: # For 1D arrays like labels, loss
#             # If readLabels/readLoss in support.cpp expect dimensions, add them
#             # f.write(f"{len(data)} 1\n") # Example if dimensions are needed
#             for val in data:
#                 f.write((fmt % val) + '\n') # One value per line
#         else: # Fallback for other 2D or generic cases
#              np.savetxt(f, data, fmt=fmt)

#     print(f"Tensor saved to {filename}")

# if __name__ == "__main__": # Ensures this part only runs when script is executed directly
#     # torch.manual_seed(0)
#     # BATCH_SIZE = 128
#     # NUM_CLASSES = 1000
#     # DTYPE = torch.float32
#     # DEVICE = 'cpu'
#     # logits = torch.randn((BATCH_SIZE, NUM_CLASSES), device=DEVICE, dtype=DTYPE)
#     # labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE, dtype=torch.long)
#     # torch_output_loss_scalar = F.cross_entropy(logits, labels, reduction='mean') # PyTorch scalar loss

#     # --- Re-generate data for saving (or use existing logits, labels from test) ---
#     M_save = 128  # 与 cross_entropy_kernel.cpp 中的 M 匹配
#     N_save = 1000 # 与 cross_entropy_kernel.cpp 中的 N 匹配
#     torch.manual_seed(42) # Use a fixed seed for reproducible data
#     save_dtype = torch.float32
#     save_device = 'cpu'

#     logits_to_save = torch.randn((M_save, N_save), device=save_device, dtype=save_dtype)
#     labels_to_save = torch.randint(0, N_save, (M_save,), device=save_device, dtype=torch.long)

#     # Calculate per-sample reference loss using PyTorch
#     # F.cross_entropy with reduction='none' gives per-sample loss
#     ref_loss_per_sample_pytorch = F.cross_entropy(logits_to_save, labels_to_save, reduction='none')
#     # Ensure it's float32 for saving
#     ref_loss_per_sample_pytorch = ref_loss_per_sample_pytorch.to(save_dtype)


#     # --- Define base path and filenames based on getDB logic ---
#     # 这个 DB_PREFIX 应该与您在远程服务器上设置的 DB_FILE 环境变量的值匹配
#     # 或者，如果您在本地生成并传输，请确保远程的 DB_FILE 指向正确的文件名前缀
#     DB_PREFIX = "./tests/matrix" # 在本地生成数据到这个子目录
#     # 创建目录 (如果不存在)
#     import os
#     os.makedirs(os.path.dirname(DB_PREFIX), exist_ok=True)


#     # 根据 cross_entropy_kernel.cpp 中的 getDB 调用来命名文件
#     # getDB(std::to_string(M) + "x" + std::to_string(N), 1); // for logits
#     # getDB(std::to_string(M), 2); // for labels
#     # getDB(std::to_string(M), 3); // for reference loss

#     logits_filename = f"{DB_PREFIX}_{M_save}x{N_save}_1.txt"
#     labels_filename = f"{DB_PREFIX}_{M_save}_2.txt"
#     ref_loss_filename = f"{DB_PREFIX}_{M_save}_3.txt"

#     # 保存 logits (2D matrix)
#     # 假设您的 C++ readMatrix 会读取 M 和 N 这两个维度
#     save_tensor_to_txt(logits_to_save, logits_filename, is_matrix=True, rows=M_save, cols=N_save)

#     # 保存 labels (1D long tensor)
#     # 假设您的 C++ readLabels 期望的是原始的 long 值，每行一个
#     save_tensor_to_txt(labels_to_save, labels_filename, fmt='%d', is_matrix=False) # %d for long integers

#     # 保存 reference loss (1D float tensor)
#     # QC++ readLoss 期望的是 float 值，每行一个
#     save_tensor_to_txt(ref_loss_per_sample_pytorch, ref_loss_filename, is_matrix=False)

#     print(f"\nGenerated test data files in {os.path.dirname(DB_PREFIX)}/")
#     print(f"Logits: {logits_filename}")
#     print(f"Labels: {labels_filename}")
#     print(f"Reference Loss: {ref_loss_filename}")