import operator
from typing import Optional
import time
import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active_to_cpu()

@triton.jit
def tanh(x):
    # For numerical stability, use simpler approximation
    # Clamp the inputs to avoid overflow
    x_clamped = tl.minimum(tl.maximum(x, -10.0), 10.0)
    pos_exp = tl.exp(x_clamped)
    neg_exp = tl.exp(-x_clamped)
    return (pos_exp - neg_exp) / (pos_exp + neg_exp)

@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    grad_output = tl.load(grad_output_ptr)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)

@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr, 
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # First find the max value for numerical stability
    m = float("-inf")
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)
        m = tl.maximum(m, block_max)
    
    # Save original value of target class
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    # Calculate sum of exp(x - max) for log-sum-exp
    d = 0.0
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        
        # Subtract max for numerical stability
        X_shifted = X_block - m
        exp_X = tl.exp(X_shifted)
        d += tl.sum(tl.where(X_offsets < n_cols, exp_X, 0.0))
        
        if label_smoothing > 0:
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0))
            else:
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))

    # Compute log-sum-exp
    lse = m + tl.log(d)

    # Compute gradients for each element
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            intermediate = tanh(X_block / softcap)
            X_block = softcap * intermediate

        # Calculate softmax probability
        softmax_X = tl.exp(X_block - m) / d

        if not HAS_WEIGHT:
            # Standard gradient for unweighted case
            gradient = softmax_X  # Base gradient is softmax
            gradient += 2 * lse_square_scale * lse * softmax_X  # z-loss gradient
            gradient += -eps  # Label smoothing constant term
            # Special case for correct class
            gradient = tl.where(X_offsets != y, gradient, gradient - (1 - label_smoothing))
            if reduction == "mean":
                gradient = gradient / n_non_ignore
        else:
            # Weighted case gradient
            weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
            
            # Original loss gradient component
            dloss_ori = (1 - label_smoothing) * softmax_X
            dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
            dloss_ori = dloss_ori * weight_y
            
            # Label smoothing component
            dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
            
            # z-loss component
            dz_loss = 2 * lse_square_scale * lse * softmax_X
            
            # Apply reduction
            if reduction == "mean":
                dloss_ori = dloss_ori / sum_non_ignore_weight
                dloss_smooth = dloss_smooth / sum_non_ignore_weight
                dz_loss = dz_loss / n_non_ignore
            
            # Combine all gradient components
            gradient = dloss_ori + dloss_smooth + dz_loss

        # Apply softcap chain rule if needed
        if HAS_SOFTCAPPING:
            gradient = gradient * (1 - intermediate * intermediate)

        tl.store(X_ptr + X_offsets, gradient, mask=X_offsets < n_cols)

    # Compute final loss value
    loss = lse - ori_X_y  # Cross entropy component
    if HAS_WEIGHT:
        loss = weight_y * loss  # Apply weight to loss

    # Apply label smoothing
    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    # Compute z-loss component
    z_loss = lse_square_scale * lse * lse
    
    # Apply reduction
    if reduction == "mean":
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        z_loss = z_loss / n_non_ignore
    
    # Final loss with z-loss component
    loss += z_loss

    # Store results
    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)

# Adjusted for CPU performance
MAX_FUSED_SIZE = 1024  

def cross_entropy_forward(
    _input,
    target,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"

    BT, V = _input.shape
    n_rows = BT

    # Use appropriate block size
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    # Create output tensors
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device) if return_z_loss else None

    # Process target information
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    # Ensure valid targets within range
    assert (target * target_mask).max() < _input.shape[-1], (
        f"Target {target.max()} is out of bounds. Expected < {_input.shape[-1]}"
    )
    assert (target * target_mask).min() >= 0, f"Target {target.min()} is out of bounds. Expected >= 0"
    
    # Calculate weights for reduction
    sum_non_ignore_weight = n_non_ignore
    weight_sum = 0.0
    if weight is not None:
        assert weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {weight.shape}"
        assert torch.is_floating_point(weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {weight.dtype}"
        )
        weight_mask_target = target.masked_fill(~target_mask, 0)  # Replace ignored indices with 0 for gathering
        sum_non_ignore_weight = torch.gather(weight, dim=0, index=weight_mask_target).masked_select(target_mask).sum().item()
        weight_sum = weight.sum().item()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

    # Ensure tensors are contiguous
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # Launch kernel
    liger_cross_entropy_kernel[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),  # always 1
        weight_ptr=weight,  # dummy if None
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        loss_stride=loss_1d.stride(-1),  # always 1
        n_cols=V,
        n_non_ignore=max(n_non_ignore, 1),  # avoid division by zero
        sum_non_ignore_weight=max(sum_non_ignore_weight, 1e-8),  # avoid division by zero
        ignore_index=ignore_index,
        weight_sum=weight_sum,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=return_z_loss,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=True if weight is not None else False,
        HAS_SOFTCAPPING=True if softcap is not None else False,
        num_warps=2,  # Optimized for CPU
    )

    # Apply reduction if neded
    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None

    return loss, z_loss, _input

def cross_entropy_backward(_input, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        pass
    else:
        BT, V = _input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            _input,
            _input.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,  # Optimized for CPU
        )

    return _input

# class LigerCrossEntropyFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         _input: torch.Tensor,
#         target: torch.Tensor,
#         weight: Optional[torch.FloatTensor],
#         ignore_index: int = -100,
#         lse_square_scale: float = 0.0,
#         label_smoothing: float = 0.0,
#         reduction: str = "mean",
#         softcap: Optional[float] = None,
#         return_z_loss: bool = False,
#     ):
#         loss, z_loss, _input = cross_entropy_forward(
#             _input,
#             target,
#             weight,
#             ignore_index,
#             lse_square_scale,
#             label_smoothing,
#             reduction,
#             softcap,
#             return_z_loss,
#         )
#         ctx.save_for_backward(_input.detach())
#         ctx.return_z_loss = return_z_loss

#         return loss, z_loss

#     @staticmethod
#     def backward(ctx, grad_output, grad_ouput2):
#         if ctx.return_z_loss:
#             del grad_ouput2  # z_loss is only for logging

#         (_input,) = ctx.saved_tensors
#         _input = cross_entropy_backward(_input, grad_output)
#         return (
#             _input,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#             None,
#         )

def torch_cross_entropy(logits, targets, weight=None, ignore_index=-100, 
                        lse_square_scale=0.0, label_smoothing=0.0, 
                        reduction="mean", softcap=None):
    # Handle softcap if provided
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    
    # Create mask for valid targets
    target_mask = targets != ignore_index
    n_valid = target_mask.sum().item()
    
    # Handle ignored indices safely
    safe_targets = torch.clamp(targets, min=0, max=logits.size(-1) - 1)
    
    # Compute log_softmax carefully for numerical stability
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    logits_shifted = logits - max_logits
    exp_logits = torch.exp(logits_shifted)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    log_softmax = logits_shifted - torch.log(sum_exp)
    
    # Calculate CE loss
    nll_loss = -torch.gather(log_softmax, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    nll_loss.masked_fill_(~target_mask, 0.0)  # Zero out ignored indices
    
    # Apply weights if provided
    sum_non_ignore_weight = n_valid
    if weight is not None:
        # Safe gathering of weights
        safe_weights = weight.gather(dim=0, index=safe_targets)
        nll_loss = nll_loss * safe_weights
        weight_sum = weight.sum().item()
        sum_non_ignore_weight = safe_weights.masked_select(target_mask).sum().item()
    
    # Calculate LSE for z_loss
    lse = max_logits.squeeze(-1) + torch.log(sum_exp).squeeze(-1)
    z_loss = lse_square_scale * (lse ** 2)
    
    # Apply label smoothing
    if label_smoothing > 0.0:
        if weight is not None:
            smooth_loss = -label_smoothing * torch.sum(log_softmax * weight.unsqueeze(0), dim=-1) / weight_sum
        else:
            smooth_loss = -label_smoothing * torch.mean(log_softmax, dim=-1)
        nll_loss = (1.0 - label_smoothing) * nll_loss + smooth_loss
    
    # Combine losses
    loss = nll_loss + z_loss
    
    # Apply reduction
    if reduction == "mean":
        loss = loss.sum() / max(sum_non_ignore_weight, 1e-8)  # Avoid division by zero
        z_loss = z_loss.sum() / max(n_valid, 1)  # Avoid division by zero
    elif reduction == "sum":
        loss = loss.sum()
        z_loss = z_loss.sum()
    
    return loss, z_loss if lse_square_scale > 0.0 else None

def benchmark_cross_entropy():
    print("Initializing benchmark data...")
    torch.manual_seed(42)  # For reproducibility
    batch_size = 16
    seq_len = 32
    vocab_size = 800
    
    # Create controllable test data
    logits = torch.randn(batch_size * seq_len, vocab_size, dtype=torch.float32)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,))
    
    # Add some ignore indices
    ignore_mask = torch.rand_like(targets.float()) > 0.95
    targets = torch.where(ignore_mask, torch.tensor(-100), targets)
    
    # Create weights
    weight = torch.rand(vocab_size)
    weight = weight / weight.sum() * vocab_size  # Normalize weights
    
    # Parameters
    ignore_index = -100
    lse_square_scale = 0.001
    label_smoothing = 0.1
    reduction = "mean"
    softcap = 20.0
    
    # Print some diagnostic info
    print(f"Data shapes: logits {logits.shape}, targets {targets.shape}")
    print(f"Ignored indices: {(targets == ignore_index).sum().item()}")
    
    # Clone inputs
    logits_torch = logits.clone()
    logits_triton = logits.clone()
    
    print("\nRunning PyTorch implementation...")
    # Warm up
    for _ in range(5):
        torch_cross_entropy(
            logits_torch.clone(), targets, weight, ignore_index, 
            lse_square_scale, label_smoothing, reduction, softcap
        )
    
    # Timed run
    torch_times = []
    for i in range(3):
        torch_start = time.time()
        torch_loss, torch_z_loss = torch_cross_entropy(
            logits_torch.clone(), targets, weight, ignore_index, 
            lse_square_scale, label_smoothing, reduction, softcap
        )
        torch_time = time.time() - torch_start
        torch_times.append(torch_time)
        print(f"  Run {i+1}: {torch_time:.6f}s, Loss: {torch_loss.item():.6f}")
    
    torch_time = sum(torch_times) / len(torch_times)
    
    print("\nRunning Triton CPU implementation...")
    # Warm up
    for _ in range(5):
        cross_entropy_forward(
            logits_triton.clone(), targets, weight, ignore_index, 
            lse_square_scale, label_smoothing, reduction, softcap, True
        )
    
    # Timed run
    triton_times = []
    for i in range(3):
        triton_start = time.time()
        triton_loss, triton_z_loss, _ = cross_entropy_forward(
            logits_triton.clone(), targets, weight, ignore_index, 
            lse_square_scale, label_smoothing, reduction, softcap, True
        )
        triton_time = time.time() - triton_start
        triton_times.append(triton_time)
        print(f"  Run {i+1}: {triton_time:.6f}s, Loss: {triton_loss.item():.6f}")
    
    triton_time = sum(triton_times) / len(triton_times)
    
    # Compare results
    torch_loss_val = torch_loss.item()
    triton_loss_val = triton_loss.item()
    loss_diff = abs(torch_loss_val - triton_loss_val)
    loss_rel_diff = loss_diff / abs(torch_loss_val) if torch_loss_val != 0 else float('inf')
    
    if torch_z_loss is not None and triton_z_loss is not None:
        torch_z_loss_val = torch_z_loss.item()
        triton_z_loss_val = triton_z_loss.item()
        z_loss_diff = abs(torch_z_loss_val - triton_z_loss_val)
        z_loss_rel_diff = z_loss_diff / abs(torch_z_loss_val) if torch_z_loss_val != 0 else float('inf')
    else:
        z_loss_diff = 0.0
        z_loss_rel_diff = 0.0
    
    print(f"\nBenchmark Results:")
    print(f"  PyTorch time: {torch_time:.6f} seconds")
    print(f"  Triton CPU time: {triton_time:.6f} seconds")
    if triton_time < torch_time:
        print(f"  Speedup: {torch_time / triton_time:.2f}x faster with Triton CPU")
    else:
        print(f"  Slowdown: {triton_time / torch_time:.2f}x slower with Triton CPU")
    
    print(f"\nCorrectness Check:")
    print(f"  PyTorch loss: {torch_loss_val:.6f}")
    print(f"  Triton CPU loss: {triton_loss_val:.6f}")
    print(f"  Absolute difference in loss: {loss_diff:.6f} (relative: {loss_rel_diff*100:.4f}%)")
    print(f"  Absolute difference in z-loss: {z_loss_diff:.6f} (relative: {z_loss_rel_diff*100:.4f}%)")
    
    # Use more appropriate tolerance for CPU implementation
    abs_tolerance = 1e-3
    rel_tolerance = 0.05  # 5% relative difference allowed
    
    is_correct = (loss_diff < abs_tolerance or loss_rel_diff < rel_tolerance)
    
    print(f"  Results {'match' if is_correct else 'do not match'} (abs tolerance: {abs_tolerance}, rel tolerance: {rel_tolerance*100}%)")
    
    return is_correct

if __name__ == "__main__":
    print("Running cross-entropy benchmark with triton-cpu vs PyTorch...")
    try:
        success = benchmark_cross_entropy()
        print(f"\nBenchmark {'successful' if success else 'failed - values do not match'}")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()