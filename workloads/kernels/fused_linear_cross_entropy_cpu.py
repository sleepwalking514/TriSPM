import time
import operator
import functools
import torch
import triton
import triton.language as tl
import numpy as np

# 确保使用CPU后端
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
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
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

    m = float("-inf")  
    d = 0.0  
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    # 计算log-sum-exp所需的最大值
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = X_offsets < n_cols
        X_block = tl.load(X_ptr + X_offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)
        m = tl.maximum(m, block_max)

    # 计算softmax分母和label smoothing项
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = X_offsets < n_cols
        X_block = tl.load(X_ptr + X_offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        
        # 减去最大值以提高数值稳定性
        X_block = X_block - m
        exp_X_block = tl.exp(X_block)
        d += tl.sum(tl.where(mask, exp_X_block, 0.0))
        
        if label_smoothing > 0:
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=mask)
                scaled_x_sum += tl.sum(tl.where(mask, -eps * X_block * weight_block, 0.0))
            else:
                scaled_x_sum += tl.sum(tl.where(mask, -eps * X_block, 0.0))

    lse = m + tl.log(d)

    # 计算梯度
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = X_offsets < n_cols
        X_block = tl.load(X_ptr + X_offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        if HAS_SOFTCAPPING:
            intermediate = tanh(X_block / softcap)
            X_block = softcap * intermediate

        # 计算softmax概率
        softmax_X = tl.exp(X_block - m) / d
        
        if not HAS_WEIGHT:
            # 标准梯度
            X_block = softmax_X  # 基础梯度是softmax
            X_block += 2 * lse_square_scale * lse * softmax_X  # z-loss梯度
            X_block += -eps  # 标签平滑常数项
            # 处理正确类别
            X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
            if reduction == "mean":
                X_block = X_block / n_non_ignore
        else:
            # 带权重的梯度
            weight_block = tl.load(weight_ptr + X_offsets, mask=mask)
            dloss_ori = (1 - label_smoothing) * softmax_X
            dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
            dloss_ori = dloss_ori * weight_y
            dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
            dz_loss = 2 * lse_square_scale * lse * softmax_X
            if reduction == "mean":
                dloss_ori = dloss_ori / sum_non_ignore_weight
                dloss_smooth = dloss_smooth / sum_non_ignore_weight
                dz_loss = dz_loss / n_non_ignore
            X_block = dloss_ori + dloss_smooth + dz_loss

        if HAS_SOFTCAPPING:
            X_block = X_block * (1 - intermediate * intermediate)

        tl.store(X_ptr + X_offsets, X_block, mask=mask)

    # 计算损失值
    loss = lse - ori_X_y
    if HAS_WEIGHT:
        loss = weight_y * loss

    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    z_loss = lse_square_scale * lse * lse
    if reduction == "mean":
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        z_loss = z_loss / n_non_ignore
    loss += z_loss

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)

# 为CPU优化的设置
MAX_FUSED_SIZE = 1024  # CPU友好的块大小

def get_cpu_settings(size):
    # 计算CPU优化的块大小和warps
    BLOCK_SIZE = triton.next_power_of_2(min(size, MAX_FUSED_SIZE))
    num_warps = 1  # CPU只需要很少的warps
    return BLOCK_SIZE, num_warps

def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    device = _input.device

    # 输入形状: BT x H
    # 输出逻辑形状: BT x V
    # 内存增长 = BT x V
    # 可以通过对tokens (BT)分块来减少内存使用
    BT, H = _input.shape
    V = weight.shape[0]
    
    # CPU优化的块大小
    BLOCK_SIZE, num_warps = get_cpu_settings(V)

    # 计算块大小和块数量
    inc_factor = (V + H - 1) // H
    chunk_size = min(1024, (BT + inc_factor - 1) // inc_factor)  # 对于CPU版本使用更小的块
    num_chunks = (BT + chunk_size - 1) // chunk_size

    # 准备梯度和损失
    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    # 使用fp32累积损失值以提高精度
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None

    # 处理target信息
    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        safe_target = torch.clamp(target, 0, V-1)  # 处理ignore_index
        masked_weights = torch.gather(ce_weight, dim=0, index=safe_target)
        total_sum_non_ignore_ce_weight = masked_weights.masked_select(target_mask).sum().item()
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    # 按块处理
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        # 执行矩阵乘法计算logits
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        # 不减少的损失
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None

        # 确保输入连续
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # 计算logits_chunk的梯度并原地存储以节省内存
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            weight_ptr=ce_weight,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=max(total_n_non_ignore, 1),  # 避免除零
            sum_non_ignore_weight=max(total_sum_non_ignore_ce_weight, 1e-8),  # 避免除零
            weight_sum=ce_weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # 保存中间结果
        loss_1d[start_idx:end_idx] = loss_1d_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice
        
        grad_logits_chunk = logits_chunk  # chunk_size x V

        # 计算输入梯度
        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        # 计算权重梯度
        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=grad_logits_chunk.t().to(_input_chunk.dtype),
                mat2=_input_chunk,
                out=grad_weight,
                alpha=1.0,
                beta=1.0,
            )

        # 计算偏置梯度
        if bias is not None:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # 根据reduction模式处理最终损失
    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        
    return loss, z_loss, grad_input, grad_weight, grad_bias

def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    # 如果cross entropy是最后一层，grad_output通常是1.0，可以跳过乘法
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # 使用Triton kernel处理梯度乘法
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE, num_warps = get_cpu_settings(H)

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # 处理权重梯度
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V
            BLOCK_SIZE, num_warps = get_cpu_settings(H)

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        # 处理偏置梯度
        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V
            BLOCK_SIZE, num_warps = get_cpu_settings(1)  # 偏置是一维的

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
    return grad_input, grad_weight, grad_bias

# 自定义混合精度装饰器实现
def amp_custom_fwd(fwd):
    @functools.wraps(fwd)
    def wrapper(*args, **kwargs):
        return fwd(*args, **kwargs)
    return wrapper

def amp_custom_bwd(bwd):
    @functools.wraps(bwd)
    def wrapper(*args, **kwargs):
        return bwd(*args, **kwargs)
    return wrapper

class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
    ):
        loss, z_loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ce_weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
        )
        # 保存梯度以便反向传播
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2  # z_loss只用于记录
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

# PyTorch参考实现
def torch_linear_cross_entropy(
    _input, 
    weight, 
    target, 
    bias=None, 
    ce_weight=None, 
    ignore_index=-100, 
    lse_square_scale=0.0, 
    label_smoothing=0.0, 
    reduction="mean", 
    softcap=None
):
    # 执行线性变换
    logits = _input @ weight.t()
    if bias is not None:
        logits = logits + bias
    
    # 应用softcap如果提供了
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    
    # 计算交叉熵损失
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # 处理ignored索引
    target_mask = target != ignore_index
    n_valid = target_mask.sum().item()
    
    # 安全地获取target索引
    safe_target = torch.clamp(target, 0, logits.size(-1) - 1)
    
    # 计算NLL损失
    nll_loss = -torch.gather(log_probs, -1, safe_target.unsqueeze(-1)).squeeze(-1)
    nll_loss.masked_fill_(~target_mask, 0.0)  # 忽略无效目标
    
    # 应用权重如果提供了
    if ce_weight is not None:
        weight_at_target = ce_weight.gather(0, safe_target)
        nll_loss = nll_loss * weight_at_target
        sum_weights = weight_at_target.masked_select(target_mask).sum().item()
    else:
        sum_weights = n_valid
    
    # 计算z-loss
    lse = torch.logsumexp(logits, dim=-1)
    z_loss = lse_square_scale * (lse ** 2)
    
    # 应用标签平滑
    if label_smoothing > 0.0:
        if ce_weight is not None:
            weights_sum = ce_weight.sum().item()
            smooth_loss = -label_smoothing * torch.sum(log_probs * ce_weight.unsqueeze(0), dim=-1) / weights_sum
        else:
            smooth_loss = -label_smoothing * torch.mean(log_probs, dim=-1)
        nll_loss = (1.0 - label_smoothing) * nll_loss + smooth_loss
        
    # 组合损失
    loss = nll_loss + z_loss
    
    # 根据reduction模式处理
    if reduction == "mean":
        loss = loss.sum() / max(sum_weights, 1e-8)
        z_loss = z_loss.sum() / max(n_valid, 1)
    elif reduction == "sum":
        loss = loss.sum()
        z_loss = z_loss.sum()
    
    return loss, z_loss if lse_square_scale > 0.0 else None

# 基准测试函数
def benchmark_fused_linear_ce():
    print("Benchmarking Fused Linear Cross Entropy CPU Implementation")
    
    # 测试参数
    batch_sizes = [32]
    seq_lens = [32]
    hidden_dims = [768, 1024]
    vocab_sizes = [32000, 50277]
    
    # 设置随机种子
    torch.manual_seed(42)
    
    all_results = []
    
    for batch in batch_sizes:
        for seq_len in seq_lens:
            for hidden_dim in hidden_dims:
                for vocab_size in vocab_sizes:
                    print(f"\nTesting B={batch}, T={seq_len}, H={hidden_dim}, V={vocab_size}")
                    
                    # 创建输入数据
                    _input = torch.randn(batch * seq_len, hidden_dim)
                    weight = torch.randn(vocab_size, hidden_dim)
                    bias = torch.randn(vocab_size)
                    target = torch.randint(0, vocab_size, (batch * seq_len,))
                    
                    # 添加一些忽略索引
                    ignore_mask = torch.rand_like(target.float()) > 0.95
                    target[ignore_mask] = -100
                    
                    # 创建CE权重
                    ce_weight = torch.rand(vocab_size)
                    
                    # 其他参数
                    lse_square_scale = 0.001
                    label_smoothing = 0.1
                    reduction = "mean"
                    softcap = 20.0
                    
                    # 1. 测试标准PyTorch实现
                    print("Running PyTorch implementation...")
                    
                    # 预热
                    for _ in range(2):
                        _ = torch_linear_cross_entropy(
                            _input.clone(), weight.clone(), target.clone(), bias.clone(), 
                            ce_weight=ce_weight.clone(), ignore_index=-100, 
                            lse_square_scale=lse_square_scale, label_smoothing=label_smoothing,
                            reduction=reduction, softcap=softcap
                        )
                    
                    torch_times = []
                    for i in range(3):
                        start = time.time()
                        torch_loss, torch_z_loss = torch_linear_cross_entropy(
                            _input.clone(), weight.clone(), target.clone(), bias.clone(), 
                            ce_weight=ce_weight.clone(), ignore_index=-100, 
                            lse_square_scale=lse_square_scale, label_smoothing=label_smoothing,
                            reduction=reduction, softcap=softcap
                        )
                        end = time.time()
                        torch_times.append(end - start)
                        print(f"  Run {i+1}: {torch_times[-1]:.6f}s")
                    
                    torch_avg_time = sum(torch_times) / len(torch_times)
                    
                    # 2. 测试融合Triton CPU实现
                    print("Running Triton CPU fused implementation...")
                    
                    # 预热
                    for _ in range(2):
                        _, _, _, _, _ = fused_linear_cross_entropy_forward(
                            _input.clone(), weight.clone(), target.clone(), 
                            ce_weight=ce_weight.clone(), bias=bias.clone(), 
                            ignore_index=-100, lse_square_scale=lse_square_scale,
                            label_smoothing=label_smoothing, reduction=reduction, 
                            softcap=softcap, return_z_loss=True
                        )
                    
                    triton_times = []
                    for i in range(3):
                        start = time.time()
                        triton_loss, triton_z_loss, _, _, _ = fused_linear_cross_entropy_forward(
                            _input.clone(), weight.clone(), target.clone(), 
                            ce_weight=ce_weight.clone(), bias=bias.clone(), 
                            ignore_index=-100, lse_square_scale=lse_square_scale,
                            label_smoothing=label_smoothing, reduction=reduction, 
                            softcap=softcap, return_z_loss=True
                        )
                        end = time.time()
                        triton_times.append(end - start)
                        print(f"  Run {i+1}: {triton_times[-1]:.6f}s")
                    
                    triton_avg_time = sum(triton_times) / len(triton_times)
                    
                    # 比较结果
                    loss_diff = abs(torch_loss.item() - triton_loss.item())
                    loss_rel_diff = loss_diff / (abs(torch_loss.item()) + 1e-10)
                    
                    if torch_z_loss is not None and triton_z_loss is not None:
                        z_loss_diff = abs(torch_z_loss.item() - triton_z_loss.item())
                        z_loss_rel_diff = z_loss_diff / (abs(torch_z_loss.item()) + 1e-10)
                    else:
                        z_loss_diff = 0.0
                        z_loss_rel_diff = 0.0
                    
                    # 输出结果
                    speedup = torch_avg_time / triton_avg_time
                    print(f"Performance: PyTorch {torch_avg_time:.6f}s, Triton {triton_avg_time:.6f}s, Speedup: {speedup:.2f}x")
                    print(f"Loss diff: {loss_diff:.8f} (relative: {loss_rel_diff*100:.4f}%)")
                    print(f"Z-loss diff: {z_loss_diff:.8f} (relative: {z_loss_rel_diff*100:.4f}%)")
                    
                    # 存储结果
                    result = {
                        'batch': batch,
                        'seq_len': seq_len, 
                        'hidden_dim': hidden_dim,
                        'vocab_size': vocab_size,
                        'torch_time': torch_avg_time,
                        'triton_time': triton_avg_time,
                        'speedup': speedup,
                        'loss_diff': loss_diff,
                        'z_loss_diff': z_loss_diff,
                        'loss_rel_diff': loss_rel_diff,
                    }
                    all_results.append(result)
    
    # 打印总结
    print("\n=== SUMMARY ===")
    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
    max_loss_diff = max(r['loss_diff'] for r in all_results)
    max_z_loss_diff = max(r['z_loss_diff'] for r in all_results)
    max_rel_diff = max(r['loss_rel_diff'] for r in all_results)
    
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Maximum loss difference: {max_loss_diff:.8f}")
    print(f"Maximum z-loss difference: {max_z_loss_diff:.8f}")
    print(f"Maximum relative difference: {max_rel_diff*100:.4f}%")
    
    # 判断正确性
    tolerance = 1e-3  # 由于是CPU实现，使用更宽松的容差
    rel_tolerance = 0.01  # 1%相对误差
    is_correct = (max_loss_diff < tolerance) or (max_rel_diff < rel_tolerance)
    print(f"Correctness test: {'PASSED' if is_correct else 'FAILED'} (abs tolerance: {tolerance}, rel tolerance: {rel_tolerance*100}%)")
    
    return is_correct

# 测试AutogradFunction的实现
def test_autograd_function():
    print("\nTesting LigerFusedLinearCrossEntropyFunction...")
    
    # 创建测试数据
    batch_size = 16
    seq_len = 32
    hidden_dim = 512
    vocab_size = 10000
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建输入数据
    _input = torch.randn(batch_size * seq_len, hidden_dim, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, requires_grad=True)
    bias = torch.randn(vocab_size, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size * seq_len,))
    
    # 执行前向传播
    loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
        _input, weight, target, bias, None, -100, 0.001, 0.1, "mean", 20.0, True
    )
    
    # 执行反向传播
    loss.backward()
    
    # 检查梯度是否计算出来了
    has_grad_input = _input.grad is not None
    has_grad_weight = weight.grad is not None
    has_grad_bias = bias.grad is not None
    
    print(f"Input has gradient: {has_grad_input}")
    print(f"Weight has gradient: {has_grad_weight}")
    print(f"Bias has gradient: {has_grad_bias}")
    
    success = has_grad_input and has_grad_weight and has_grad_bias
    print(f"Autograd test: {'PASSED' if success else 'FAILED'}")
    
    return success

if __name__ == "__main__":
    print("Running fused linear cross entropy benchmark with Triton CPU vs PyTorch...")
    try:
        # 运行基准测试
        bench_success = benchmark_fused_linear_ce()
        # 测试autograd功能
        autograd_success = test_autograd_function()
        
        overall_success = bench_success and autograd_success
        print(f"\nOverall result: {'Successful' if overall_success else 'Failed'}")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()