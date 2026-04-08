import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import os
import numpy as np # For saving text files

# --- Configuration ---
DTYPE = torch.float32
DEVICE = 'cpu' # For CPU benchmarking with Triton's CPU backend
EPSILON = 1e-5   # Epsilon for LayerNorm

triton.runtime.driver.set_active_to_cpu()

# --- Triton Kernel for Layer Normalization ---
@triton.jit
def layer_norm_kernel(
    x_ptr,             # Pointer to input tensor (M, N)
    gamma_ptr,         # Pointer to gamma tensor (N,)
    beta_ptr,          # Pointer to beta tensor (N,)
    output_ptr,        # Pointer to output tensor (M, N)
    M,                 # Number of rows (e.g., BatchSize or BatchSize * SeqLen)
    N,                 # Number of features (last dimension, normalized over)
    stride_x_m,        # Stride for M dimension of x
    stride_x_n,        # Stride for N dimension of x
    stride_gamma_n,    # Stride for N dimension of gamma (usually 1 or 0 if scalar-like broadcast)
    stride_beta_n,     # Stride for N dimension of beta (usually 1 or 0)
    stride_output_m,   # Stride for M dimension of output
    stride_output_n,   # Stride for N dimension of output
    epsilon,           # Epsilon value for numerical stability
    BLOCK_SIZE_N: tl.constexpr, # Block size for the N dimension (features)
):
    # Each program instance processes one row (one sample or one token's features)
    pid_m = tl.program_id(axis=0)  # Row index

    # --- Load one row of x ---
    # Create pointers to the current row of x
    row_x_start_ptr = x_ptr + pid_m * stride_x_m
    # Create offsets for the N dimension
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    # Load the row of x, handling masking for N not being a multiple of BLOCK_SIZE_N
    # Iteratively load if N > BLOCK_SIZE_N
    # For simplicity, this kernel assumes BLOCK_SIZE_N >= N or uses one iteration
    # More robust implementation would loop over blocks of N if N is large.
    # Let's assume for now BLOCK_SIZE_N is large enough, or N is processed in one go.
    # If BLOCK_SIZE_N < N, this needs a loop.

    # For a simpler kernel focusing on the core logic of one row:
    # We'll compute stats for the full row, then apply.
    # This requires loading the full row if N > BLOCK_SIZE_N by iterating.
    # Let's simplify for this example and assume we process N in blocks.

    # To make it fully work for any N with BLOCK_SIZE_N:
    current_x_row = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32) # Accumulator for the row
    mean_val = 0.0
    var_val = 0.0

    # Step 1 & 2: Calculate mean and variance for the current row
    # First pass for mean: sum and count
    _sum = tl.zeros((1,), dtype=tl.float32) # Scalar sum
    for off_n_base in range(0, tl.cdiv(N, BLOCK_SIZE_N)): # Loop if N > BLOCK_SIZE_N
        current_offs_n = off_n_base * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        current_mask_n = current_offs_n < N
        x_block = tl.load(row_x_start_ptr + current_offs_n * stride_x_n,
                          mask=current_mask_n, other=0.0)
        x_block = x_block.to(tl.float32) # Ensure float32
        _sum += tl.sum(x_block, axis=0) # Sum along the feature dimension (axis=0 for 1D block)
    mean_val = _sum / N
    mean_val = mean_val.to(tl.float32)

    # Second pass for variance: sum of squared differences
    _var_sum = tl.zeros((1,), dtype=tl.float32) # Scalar variance sum
    for off_n_base in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        current_offs_n = off_n_base * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        current_mask_n = current_offs_n < N
        x_block = tl.load(row_x_start_ptr + current_offs_n * stride_x_n,
                          mask=current_mask_n, other=0.0)
        x_block = x_block.to(tl.float32)
        diff = x_block - mean_val # Broadcasting mean_val
        diff_sq = diff * diff
        _var_sum += tl.sum(diff_sq, axis=0)
    var_val = _var_sum / N
    var_val = var_val.to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_val + epsilon)
    inv_std = inv_std.to(tl.float32)

    # Step 3 & 4: Normalize, scale, and shift
    # Iterate again to apply normalization and store
    row_output_start_ptr = output_ptr + pid_m * stride_output_m

    for off_n_base in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        current_offs_n = off_n_base * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        current_mask_n = current_offs_n < N

        x_block = tl.load(row_x_start_ptr + current_offs_n * stride_x_n,
                          mask=current_mask_n, other=0.0)
        x_block = x_block.to(tl.float32)

        # Load gamma and beta for the current block
        # Assuming gamma_ptr and beta_ptr are 1D arrays of size N
        gamma_block_ptr = gamma_ptr + current_offs_n * stride_gamma_n # stride_gamma_n is likely 1
        beta_block_ptr = beta_ptr + current_offs_n * stride_beta_n   # stride_beta_n is likely 1

        gamma_block = tl.load(gamma_block_ptr, mask=current_mask_n, other=1.0) # Default gamma to 1
        beta_block = tl.load(beta_block_ptr, mask=current_mask_n, other=0.0)   # Default beta to 0
        gamma_block = gamma_block.to(tl.float32)
        beta_block = beta_block.to(tl.float32)

        # Normalize
        normalized_x = (x_block - mean_val) * inv_std
        # Scale and shift
        output_block = normalized_x * gamma_block + beta_block

        # Store the output block
        tl.store(row_output_start_ptr + current_offs_n * stride_output_n,
                 output_block, mask=current_mask_n)


# --- Python Wrapper for LayerNorm Triton Kernel ---
def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = EPSILON):
    """
    Layer Normalization using Triton kernel.
    Assumes x is 2D [M, N] or 3D [Batch, SeqLen, N] which will be reshaped to 2D.
    Gamma and Beta are 1D [N,].
    """
    input_shape = x.shape
    if x.dim() == 3:
        M, S, N_features = input_shape
        x_reshaped = x.reshape(-1, N_features) # Reshape to [M*S, N]
    elif x.dim() == 2:
        M_rows, N_features = input_shape
        x_reshaped = x
    else:
        raise ValueError("Input tensor x must be 2D or 3D")

    M, N = x_reshaped.shape

    # Ensure gamma and beta have the correct shape [N]
    assert gamma.shape == (N,), f"Gamma shape mismatch: expected ({N},), got {gamma.shape}"
    assert beta.shape == (N,), f"Beta shape mismatch: expected ({N},), got {beta.shape}"

    # Ensure contiguity and correct dtype
    x_reshaped = x_reshaped.contiguous().to(DTYPE).to(DEVICE)
    gamma = gamma.contiguous().to(DTYPE).to(DEVICE)
    beta = beta.contiguous().to(DTYPE).to(DEVICE)

    # Allocate output tensor
    output = torch.empty_like(x_reshaped)

    # Kernel launch configuration
    grid = (M,) # One program instance per row/sample

    # BLOCK_SIZE_N should be a power of 2 and >= N for best single-pass, or tuned
    # This should ideally come from config.json or be tuned.
    # For kernel simplicity, if BLOCK_SIZE_N < N, the kernel itself loops.
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    # A common choice is a fixed larger block if N varies a lot e.g. 1024
    # For this kernel version that loops internally over N-dim blocks if N > BLOCK_SIZE_N,
    # a smaller BLOCK_SIZE_N (e.g., 256, 512) is also fine.
    # Let's use a common default from previous examples for consistency for now.
    # BLOCK_SIZE_N_CONFIG = 256 # Example, should match config.json's blk[1]

    layer_norm_kernel[grid](
        x_reshaped, gamma, beta, output,
        M, N,
        x_reshaped.stride(0), x_reshaped.stride(1),
        gamma.stride(0) if gamma.dim() > 0 else 0, # Stride for 1D gamma (or 0 if it behaves like scalar)
        beta.stride(0) if beta.dim() > 0 else 0,    # Stride for 1D beta
        output.stride(0), output.stride(1),
        epsilon,
        BLOCK_SIZE_N=BLOCK_SIZE_N # Pass the dynamically determined or configured block size
    )

    if x.dim() == 3:
        output = output.reshape(input_shape) # Reshape back to original 3D if needed

    return output

# --- Function to save tensors in the C++ expected format ---
def save_tensor_to_txt(tensor, filename, fmt='%.8f', is_matrix=False, rows=0, cols=0):
    """Saves a PyTorch tensor to a text file."""
    print(f"Saving tensor to {filename}...")
    # Ensure tensor is on CPU and convert to NumPy
    data = tensor.detach().cpu().numpy()
    with open(filename, 'w') as f:
        if is_matrix and rows > 0 and cols > 0: # For 2D matrices
            f.write(f"{rows} {cols}\n") # Write dimensions if support.cpp readMatrix expects it
            for i in range(rows):
                row_str = ' '.join([fmt % val for val in data[i]])
                f.write(row_str + '\n')
        elif not is_matrix and data.ndim == 1: # For 1D arrays
            # Assuming readLabels/readLoss/etc. for 1D don't expect dimensions header
            # If they do, add: f.write(f"{len(data)} 1\n")
            for val in data:
                f.write((fmt % val) + '\n') # One value per line
        elif data.ndim == 0: # Scalar
             f.write(fmt % data + '\n')
        else: # Fallback for other (e.g., if gamma/beta were saved as matrix Nx1)
             # This part needs to match how your C++ will read gamma/beta if they are matrices
             if rows > 0 and cols > 0 :
                 f.write(f"{rows} {cols}\n")
             np.savetxt(f, data, fmt=fmt)
    print(f"Tensor saved to {filename}")


# --- Testing and Data Generation/Saving ---
if __name__ == "__main__":
    torch.manual_seed(0)

    # Test parameters
    BATCH_SIZE = 4 # Keep smaller for easier debugging / smaller files
    SEQ_LEN = 3 # Example sequence length for 3D input
    NUM_FEATURES = 1024  # N dimension, should ideally be multiple of BLOCK_SIZE_N for efficiency
                         # but kernel handles general N now.

    # Create input data
    # Test with 2D input: (BATCH_SIZE, NUM_FEATURES)
    # Test with 3D input: (BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
    # For simplicity of M in kernel, we often reshape 3D to 2D [Batch*SeqLen, Features]
    # Let M be the number of rows the kernel processes (BatchSize or BatchSize*SeqLen)

    M_test = BATCH_SIZE * SEQ_LEN # Total rows kernel will see
    N_test = NUM_FEATURES

    # Input x can be 2D or 3D. The wrapper handles reshaping.
    # Let's generate a 3D input for a more common use case.
    x_input_torch = torch.randn((BATCH_SIZE, SEQ_LEN, N_test), device=DEVICE, dtype=DTYPE)
    # Or 2D: x_input_torch = torch.randn((M_test, N_test), device=DEVICE, dtype=DTYPE)


    # Gamma and Beta are 1D tensors of size NUM_FEATURES
    gamma_torch = torch.rand(N_test, device=DEVICE, dtype=DTYPE) * 0.5 + 0.8 # Near 1
    beta_torch = torch.rand(N_test, device=DEVICE, dtype=DTYPE) * 0.2 - 0.1   # Near 0

    print(f"Input x shape: {x_input_torch.shape}")
    print(f"Gamma shape: {gamma_torch.shape}, Beta shape: {beta_torch.shape}")


    # --- Triton Calculation ---
    triton_output = layer_norm(x_input_torch.clone(), gamma_torch.clone(), beta_torch.clone(), epsilon=EPSILON)

    # --- PyTorch Calculation (Reference) ---
    # PyTorch layer_norm expects normalized_shape as the last dim(s)
    # For input [B, S, N], normalized_shape is [N]
    # For input [B, N], normalized_shape is [N]
    ref_output_pytorch = F.layer_norm(x_input_torch.clone().to(torch.float32),
                                      normalized_shape=[N_test], # Normalize over the last dimension
                                      weight=gamma_torch.clone().to(torch.float32),
                                      bias=beta_torch.clone().to(torch.float32),
                                      eps=EPSILON)

    print(f"\nTriton LayerNorm Output (first few values of first sample):\n{triton_output.view(-1, N_test)[0, :5]}")
    print(f"PyTorch LayerNorm Output (first few values of first sample):\n{ref_output_pytorch.view(-1, N_test)[0, :5]}")

    # --- Comparison ---
    atol = 1e-4 # LayerNorm might need slightly higher tolerance due to sqrt/div
    rtol = 1e-3
    are_close = torch.allclose(triton_output.to(torch.float32), ref_output_pytorch, atol=atol, rtol=rtol)

    if are_close:
        print(f"\n✅ Triton ({DEVICE}) and PyTorch ({DEVICE}) LayerNorm match within tolerance (atol={atol}, rtol={rtol})")
    else:
        diff = torch.abs(triton_output.to(torch.float32) - ref_output_pytorch)
        print(f"\n❌ Triton ({DEVICE}) and PyTorch ({DEVICE}) LayerNorm differ.")
        print(f"   Max difference: {torch.max(diff)}")
        print(f"   Mean difference: {torch.mean(diff)}")


    # # --- Save test data files ---
    # print("\n--- Saving test data for C++ benchmark ---")
    # # This DB_PREFIX should match the prefix part of the DB_FILE environment variable
    # # that your C++ main driver (layer_norm_kernel.cpp) will use.
    # # Or, it's the prefix if files are directly in the test_data dir.
    # output_dir = "../../test_layer_norm" # Create a subdirectory for these files
    # os.makedirs(output_dir, exist_ok=True)
    # DB_PREFIX = os.path.join(output_dir, "matrix")


    # # Files for cross_entropy_kernel.cpp (example, adapt for LayerNorm)
    # # Input x: SHAPE is MxN where M is total rows, N is features
    # # Gamma: SHAPE is N (or Nx1), index 2
    # # Beta: SHAPE is N (or Nx1), index 3
    # # Reference Output: SHAPE is MxN, index 4 (or another index)

    # # Reshape x_input_torch to 2D [M_test, N_test] for saving if it was 3D
    # x_to_save = x_input_torch.reshape(M_test, N_test)
    # ref_output_to_save = ref_output_pytorch.reshape(M_test, N_test)

    # # File naming convention based on previous discussions
    # # (adjust SHAPE string and index as per your C++ getDB expectations)
    # input_x_filename     = f"{DB_PREFIX}_{M_test}x{N_test}_1.txt" # Index 1 for main input
    # gamma_filename       = f"{DB_PREFIX}_{N_test}_2.txt"          # Index 2 for gamma
    # beta_filename        = f"{DB_PREFIX}_{N_test}_3.txt"          # Index 3 for beta
    # ref_output_filename  = f"{DB_PREFIX}_{M_test}x{N_test}_4.txt" # Index 4 for reference output

    # # Save input x (MxN matrix)
    # save_tensor_to_txt(x_to_save, input_x_filename, is_matrix=True, rows=M_test, cols=N_test)

    # # Save gamma (N vector, saved as Nx1 matrix or 1 per line depending on C++ reader)
    # # Assuming your C++ 'readGamma' (or adapted readMatrix/readLoss) reads it as a 1D array (N elements)
    # save_tensor_to_txt(gamma_torch, gamma_filename, is_matrix=False) # rows=N_test, cols=1 for matrix format

    # # Save beta (N vector)
    # save_tensor_to_txt(beta_torch, beta_filename, is_matrix=False) # rows=N_test, cols=1 for matrix format

    # # Save reference output (MxN matrix)
    # save_tensor_to_txt(ref_output_to_save, ref_output_filename, is_matrix=True, rows=M_test, cols=N_test)
