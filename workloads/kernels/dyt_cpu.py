import operator
import functools
import time
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

def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper

def calculate_settings(n):
    # 对于CPU，使用更保守的块大小
    MAX_FUSED_SIZE = 1024
    BLOCK_SIZE = triton.next_power_of_2(min(n, MAX_FUSED_SIZE))
    num_warps = 1  # CPU模式下使用最小warps数
    return BLOCK_SIZE, num_warps

@triton.jit
def _dyt_fwd_kernel(
    x_ptr,
    x_row_stride,
    alpha_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    DyT前向传播核函数
    """
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # 计算行指针位置
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    # 加载参数
    alpha = tl.load(alpha_ptr)
    gamma = tl.load(gamma_ptr + offsets, mask=mask)
    beta = tl.load(beta_ptr + offsets, mask=mask)
    
    # 加载输入
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 计算DyT: y = gamma * tanh(alpha * x) + beta
    y = gamma * tanh((alpha * x).cast(tl.float32)) + beta
    
    # 存储结果
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def _dyt_bwd_kernel(
    x_ptr,
    x_row_stride,
    dy_ptr,
    dy_row_stride,
    dx_ptr,
    dx_row_stride,
    alpha_ptr,
    dalpha_ptr,
    gamma_ptr,
    dgamma_ptr,
    dgamma_row_stride,
    dbeta_ptr,
    dbeta_stride,
    n_cols,
    n_rows,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    DyT反向传播核函数
    """
    pid = tl.program_id(0)

    # 计算当前程序处理的行范围
    row_start = pid * ROWS_PER_PROGRAM
    row_end = tl.minimum((pid + 1) * ROWS_PER_PROGRAM, n_rows)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # 初始化梯度累积器
    dalpha = 0.0
    dgamma = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dbeta = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 计算起始指针位置
    x_ptr += row_start * x_row_stride
    dx_ptr += row_start * dx_row_stride
    dy_ptr += row_start * dy_row_stride
    
    # 加载固定参数
    alpha = tl.load(alpha_ptr)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)

    # 遍历分配的行
    for _ in tl.range(row_start, row_end):
        # 加载当前行的梯度和输入
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # 计算中间值
        alpha_x = (alpha * x).cast(tl.float32)
        tanh_ax = tanh(alpha_x)
        sech2_ax = 1 - tanh_ax * tanh_ax  # sech^2(x) = 1 - tanh^2(x)

        # 计算各个梯度
        dx = dy * gamma * sech2_ax * alpha
        dalpha += tl.sum(dy * gamma * sech2_ax * x)
        dgamma += dy * tanh_ax
        dbeta += dy
        
        # 存储dx梯度
        tl.store(dx_ptr + offsets, dx, mask=mask)

        # 更新指针到下一行
        dy_ptr += dy_row_stride
        x_ptr += x_row_stride
        dx_ptr += dx_row_stride

    # 存储累积的梯度
    tl.store(dgamma_ptr + pid * dgamma_row_stride + offsets, dgamma, mask=mask)
    tl.store(dbeta_ptr + pid * dbeta_stride + offsets, dbeta, mask=mask)
    tl.store(dalpha_ptr + pid, dalpha)

def liger_dyt_fwd(x, alpha, gamma, beta):
    shape = x.shape
    dim = shape[-1]
    x = x.view(-1, dim)
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    
    # 计算块大小
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    
    # 分块处理 - 需要多次调用kernel
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # 创建切片视图
        col_count = min(BLOCK_SIZE, n_cols - col_start)
        x_slice = x[:, col_start:col_start+col_count]
        y_slice = y[:, col_start:col_start+col_count]
        gamma_slice = gamma[col_start:col_start+col_count]
        beta_slice = beta[col_start:col_start+col_count]
        
        _dyt_fwd_kernel[(n_rows,)](
            x_ptr=x_slice,
            x_row_stride=x_slice.stride(0),
            alpha_ptr=alpha,
            gamma_ptr=gamma_slice,
            beta_ptr=beta_slice,
            y_ptr=y_slice,
            y_row_stride=y_slice.stride(0),
            n_cols=col_count,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    
    return y.view(*shape)

def liger_dyt_bwd(dy, x, alpha, gamma):
    shape = dy.shape
    dtype = x.dtype
    dim = shape[-1]
    dy = dy.view(-1, dim)
    x = x.view(-1, dim)
    n_rows, n_cols = dy.shape
    
    # 计算块大小和SM数量
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    sm_count = max(1, min(4, n_rows // 128))  # 根据行数调整SM数量
    
    # 准备梯度输出
    dx = torch.zeros_like(x, dtype=torch.float32)
    dalpha_acc = torch.zeros(1, dtype=torch.float32, device=x.device)
    dgamma_acc = torch.zeros(n_cols, dtype=torch.float32, device=x.device)
    dbeta_acc = torch.zeros(n_cols, dtype=torch.float32, device=x.device)
    
    # 分块处理 - 按列分块
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_count = min(BLOCK_SIZE, n_cols - col_start)
        
        # 创建切片视图
        x_slice = x[:, col_start:col_start+col_count]
        dy_slice = dy[:, col_start:col_start+col_count]
        dx_slice = dx[:, col_start:col_start+col_count]
        gamma_slice = gamma[col_start:col_start+col_count]
        
        # 创建临时梯度累积器 
        _dalpha = torch.zeros(sm_count, dtype=torch.float32, device=x.device)
        _dgamma = torch.zeros((sm_count, col_count), dtype=torch.float32, device=x.device)
        _dbeta = torch.zeros((sm_count, col_count), dtype=torch.float32, device=x.device)
        
        grid = (sm_count,)
        rows_per_program = (n_rows + sm_count - 1) // sm_count
        
        _dyt_bwd_kernel[grid](
            x_ptr=x_slice,
            x_row_stride=x_slice.stride(0),
            dy_ptr=dy_slice,
            dy_row_stride=dy_slice.stride(0),
            dx_ptr=dx_slice,
            dx_row_stride=dx_slice.stride(0),
            alpha_ptr=alpha,
            dalpha_ptr=_dalpha,
            gamma_ptr=gamma_slice,
            dgamma_ptr=_dgamma,
            dgamma_row_stride=_dgamma.stride(0),
            dbeta_ptr=_dbeta,
            dbeta_stride=_dbeta.stride(0),
            n_cols=col_count,
            n_rows=n_rows,
            ROWS_PER_PROGRAM=rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        # 累积梯度
        dalpha_acc += _dalpha.sum()
        dgamma_acc[col_start:col_start+col_count] = _dgamma.sum(dim=0)
        dbeta_acc[col_start:col_start+col_count] = _dbeta.sum(dim=0)
    
    return dx.view(*shape), dalpha_acc.to(dtype), dgamma_acc.to(dtype), dbeta_acc.to(dtype)

class LigerDyTFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, alpha, gamma, beta):
        y = liger_dyt_fwd(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma)
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        x, alpha, gamma = ctx.saved_tensors
        dx, dalpha, dgamma, dbeta = liger_dyt_bwd(
            grad_output,
            x,
            alpha,
            gamma,
        )
        return (dx, dalpha, dgamma, dbeta)

# PyTorch参考实现
class TorchDyT(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta

# Triton实现
class TritonDyT(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return LigerDyTFunction.apply(x, self.alpha, self.gamma, self.beta)

# 运行性能和准确度测试
def benchmark_dyt():
    print("Benchmarking DyT (Dynamic Token-wise Transformation)")
    
    # 测试参数 - 减少测试组合以加快测试
    batch_sizes = [16]  # 简化测试
    seq_lens = [128, 256]
    dims = [768, 1536]  # 避免太大的维度
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    
    all_results = []
    
    for batch in batch_sizes:
        for seq_len in seq_lens:
            for dim in dims:
                print(f"\nTesting batch={batch}, seq_len={seq_len}, dim={dim}")
                
                # 创建测试数据
                x = torch.randn(batch, seq_len, dim)
                
                # 创建两个模型
                torch_dyt = TorchDyT(dim)
                triton_dyt = TritonDyT(dim)
                
                # 确保两个模型有相同的参数
                triton_dyt.alpha.data.copy_(torch_dyt.alpha.data)
                triton_dyt.gamma.data.copy_(torch_dyt.gamma.data)
                triton_dyt.beta.data.copy_(torch_dyt.beta.data)
                
                # 前向测试
                # 热身
                for _ in range(2):
                    _ = torch_dyt(x)
                    _ = triton_dyt(x)
                
                # 计时并运行
                torch_times = []
                for i in range(3):
                    start_time = time.time()
                    torch_output = torch_dyt(x)
                    torch_time = time.time() - start_time
                    torch_times.append(torch_time)
                    print(f"  Torch forward run {i+1}: {torch_time:.6f}s")
                
                triton_times = []
                for i in range(3):
                    start_time = time.time()
                    triton_output = triton_dyt(x)
                    triton_time = time.time() - start_time
                    triton_times.append(triton_time)
                    print(f"  Triton forward run {i+1}: {triton_time:.6f}s")
                
                # 计算平均时间
                torch_avg_time = sum(torch_times) / len(torch_times)
                triton_avg_time = sum(triton_times) / len(triton_times)
                
                # 检查输出差异
                output_diff = torch.abs(torch_output - triton_output).max().item()
                rel_diff = output_diff / (torch.abs(torch_output).mean().item() + 1e-10)
                
                # 反向传播测试
                grad_output = torch.randn_like(x)
                
                # 热身
                for _ in range(2):
                    torch_output = torch_dyt(x)
                    torch_output.backward(grad_output, retain_graph=True)
                    
                    triton_output = triton_dyt(x)
                    triton_output.backward(grad_output, retain_graph=True)
                
                # 复制参数以便重新测试
                torch_dyt_bwd = TorchDyT(dim)
                triton_dyt_bwd = TritonDyT(dim)
                torch_dyt_bwd.alpha.data.copy_(torch_dyt.alpha.data)
                torch_dyt_bwd.gamma.data.copy_(torch_dyt.gamma.data)
                torch_dyt_bwd.beta.data.copy_(torch_dyt.beta.data)
                triton_dyt_bwd.alpha.data.copy_(torch_dyt.alpha.data)
                triton_dyt_bwd.gamma.data.copy_(torch_dyt.gamma.data)
                triton_dyt_bwd.beta.data.copy_(torch_dyt.beta.data)
                
                # 清除梯度
                torch_dyt_bwd.zero_grad()
                triton_dyt_bwd.zero_grad()
                
                # 计时反向传播
                torch_bwd_times = []
                for i in range(3):
                    torch_output = torch_dyt_bwd(x)
                    start_time = time.time()
                    torch_output.backward(grad_output, retain_graph=True)
                    torch_time = time.time() - start_time
                    torch_bwd_times.append(torch_time)
                    print(f"  Torch backward run {i+1}: {torch_time:.6f}s")
                
                triton_bwd_times = []
                for i in range(3):
                    triton_output = triton_dyt_bwd(x)
                    start_time = time.time()
                    triton_output.backward(grad_output, retain_graph=True)
                    triton_time = time.time() - start_time
                    triton_bwd_times.append(triton_time)
                    print(f"  Triton backward run {i+1}: {triton_time:.6f}s")
                
                # 计算平均反向传播时间
                torch_avg_bwd_time = sum(torch_bwd_times) / len(torch_bwd_times)
                triton_avg_bwd_time = sum(triton_bwd_times) / len(triton_bwd_times)
                
                # 检查梯度差异
                alpha_grad_diff = torch.abs(torch_dyt_bwd.alpha.grad - triton_dyt_bwd.alpha.grad).max().item()
                gamma_grad_diff = torch.abs(torch_dyt_bwd.gamma.grad - triton_dyt_bwd.gamma.grad).max().item()
                beta_grad_diff = torch.abs(torch_dyt_bwd.beta.grad - triton_dyt_bwd.beta.grad).max().item()
                
                # 格式化输出结果
                speedup_fwd = torch_avg_time / triton_avg_time
                speedup_bwd = torch_avg_bwd_time / triton_avg_bwd_time
                
                print(f"Forward: PyTorch {torch_avg_time:.6f}s, Triton {triton_avg_time:.6f}s, Speedup: {speedup_fwd:.2f}x")
                print(f"Backward: PyTorch {torch_avg_bwd_time:.6f}s, Triton {triton_avg_bwd_time:.6f}s, Speedup: {speedup_bwd:.2f}x")
                print(f"Output diff: {output_diff:.8f} (relative: {rel_diff:.8f})")
                print(f"Gradient diffs: alpha={alpha_grad_diff:.8f}, gamma={gamma_grad_diff:.8f}, beta={beta_grad_diff:.8f}")
                
                # 保存结果
                result = {
                    'batch': batch,
                    'seq_len': seq_len,
                    'dim': dim,
                    'torch_fwd_time': torch_avg_time,
                    'triton_fwd_time': triton_avg_time,
                    'torch_bwd_time': torch_avg_bwd_time,
                    'triton_bwd_time': triton_avg_bwd_time,
                    'speedup_fwd': speedup_fwd,
                    'speedup_bwd': speedup_bwd,
                    'output_diff': output_diff,
                    'alpha_grad_diff': alpha_grad_diff,
                    'gamma_grad_diff': gamma_grad_diff,
                    'beta_grad_diff': beta_grad_diff,
                }
                all_results.append(result)
    
    # 打印总结
    print("\n=== SUMMARY ===")
    avg_speedup_fwd = sum(r['speedup_fwd'] for r in all_results) / len(all_results)
    avg_speedup_bwd = sum(r['speedup_bwd'] for r in all_results) / len(all_results)
    max_output_diff = max(r['output_diff'] for r in all_results)
    max_grad_diff = max(max(r['alpha_grad_diff'], r['gamma_grad_diff'], r['beta_grad_diff']) for r in all_results)
    
    print(f"Average Forward Speedup: {avg_speedup_fwd:.2f}x")
    print(f"Average Backward Speedup: {avg_speedup_bwd:.2f}x")
    print(f"Maximum Output Difference: {max_output_diff:.8f}")
    print(f"Maximum Gradient Difference: {max_grad_diff:.8f}")
    
    # 判断正确性
    tolerance = 5e-3  # 由于是CPU实现，稍微放宽容差
    is_correct = max_output_diff < tolerance and max_grad_diff < tolerance
    print(f"Correctness test: {'PASSED' if is_correct else 'FAILED'} (tolerance: {tolerance})")
    
    return is_correct

if __name__ == "__main__":
    print("Running DyT benchmark with triton-cpu vs PyTorch...")
    try:
        success = benchmark_dyt()
        print(f"\nBenchmark {'successful' if success else 'failed - values do not match'}")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()