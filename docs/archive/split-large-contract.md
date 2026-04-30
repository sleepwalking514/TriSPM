# SplitLargeContract Pass — 寄存器微分块

> **Status: ARCHIVED RESULT — 2026-04-30**
> SplitLargeContract has landed and been validated for the Phase 3 large-tile
> matmul path. Keep this file as the design and measurement record.
> Follow-on DMA reuse work is tracked in
> [`../plans/spm-dma-reuse.md`](../plans/spm-dma-reuse.md).

## 问题

256×256×256 matmul（32×32 tile, T3-GSM1）的 ROI 统计显示：

| 指标 | 值 | 解读 |
|---|---|---|
| numCycles | 4,393,944 | |
| IPC | 1.09 | 中等 |
| **L1I miss rate** | **63.6%** | 极差 |
| **icacheStallCycles** | **2,988,441 (68% of cycles)** | 头号瓶颈 |
| L1D miss rate | 0.30% | 优 |
| spm.bankConflicts | 0 | 优 |
| spm_dma.waitStallCycles | 47,754 (1%) | 良 |

SPM 子系统本身工作良好，但 CPU 前端被 I-cache miss 卡死了 68% 周期。
T3-GSM1 vs T2-GSM1 仅省 4.3% cycles — SPM 改进被前端瓶颈淹没。

### 根因

`tl.dot(a[32,32], b[32,32], acc[32,32])` 经 `ConvertDotGeneric` 降为
`vector.contract<32×32×32>`，LLVM 展开为 32 行 × `<32xf32>` 的 `fmuladd`。

VLEN=256 下每个 `<32xf32>` 需要 LMUL=4（4 个物理 vreg），32 行累加器共需
**128 个物理 vreg**，而 RISC-V 只有 **32 个**。LLVM 被迫将 ~75% 累加器
spill 到栈上：

- `vl4r.v` × 917, `vs4r.v` × 949（spill/reload）
- `vfmacc.vf` × 1024（实际 MAC）
- 每条 MAC 伴随 ~1.8 对 spill/reload + 地址计算

结果：内层循环代码体积 10K 行 asm，远超 L1I 容量。

## 方案

新增 `SplitLargeContract` pass，在 SPM 变换之后、LLVM lowering 之前，
将包含大 `vector.contract` 的 K-loop 拆成多个独立的 K-loop，每个只累加
MICRO_M 行。

### 为什么必须拆 loop 而非仅拆 contract

最初尝试在同一 K-loop 内将 contract 拆成 8 个小 contract + extract/insert，
但 LLVM 的 canonicalization 会看穿 extract_strided_slice / insert_strided_slice
并将累加器重新合并为 `[32 x <32xf32>]` phi 节点，spill 不变。

正确做法：将单个 K-loop 替换为 8 个顺序 K-loop，每个有自己独立的
`[4 x <32xf32>]` loop-carried phi。LLVM 无法跨 loop 合并。

### 变换

```
原始:
  scf.for %k = 0 to K step BK iter_args(%acc<32×32>) {
    %a = load A[..., %k]           // <32×32>
    %b = load B[%k, ...]           // <32×32>
    %acc' = vector.contract %a, %b, %acc
    yield %acc'
  }
  store %acc to C

拆成 (MICRO_M=4):
  for m_start in [0, 4, 8, ..., 28]:
    %micro_init = extract_strided_slice %full_acc [m_start, 0] [4, 32]
    scf.for %k = 0 to K step BK iter_args(%micro_acc<4×32>) {
      %a = load A[..., %k]         // <32×32> (完整 tile，从 SPM 读)
      %a_slice = extract_strided_slice %a [m_start, 0] [4, 32]
      %b = load B[%k, ...]         // <32×32> (完整 tile，从 SPM 读)
      %micro_acc' = vector.contract %a_slice, %b, %micro_acc
      yield %micro_acc'
    }
    %full_acc = insert_strided_slice %micro_acc, %full_acc [m_start, 0]
  store %full_acc to C
```

### 为什么在 SPM pass 之后

`ConvertMemoryToSPM` 通过 `readFeedsDot` 检查 `transfer_read` 的直接 user
是否是 `vector.ContractionOp`。如果先拆分 contract，`transfer_read` 的直接
user 变成 `extract_strided_slice`，SPM pass 无法匹配 GEMM 模式。

Pipeline 顺序：
```
add_convert_dot_generic          // tt.dot → vector.contract (32×32)
add_spm_tensor_placement         // 标注 tier
add_convert_memory_to_spm        // 匹配 transfer_read→contract，插入 DMA
add_split_large_contract         // 新 pass：拆 contract 为微块
```

### 关键设计点

1. **DMA 粒度不变**：A 和 B 仍然是 32×32 的完整 tile DMA
2. **DMA reuse 不由本 pass 解决**：拆成多个独立 K-loop 后，每个 micro-loop
   都有自己的 SPM K-loop 和 body prefetch。B 会在 micro-loop 之间重复搬运；
   re-prime 只修复 prologue 正确性，不能消除 K-loop body 内的重复 B DMA。
   后续的 DMA reuse 必须在 `ConvertMemoryToSPM` 中生成 microM-aware fused
   GEMM schedule。
3. **寄存器压力**：4 行 × LMUL4 = 16 物理 vreg（累加器），剩 16 个给
   A load、B load、临时值 → 零 spill
4. **只匹配标准 GEMM contract**：检查 indexing maps 为
   `(m,n,k)→(m,k)`, `(m,n,k)→(k,n)`, `(m,n,k)→(m,n)`

## 文件变更

| 文件 | 变更 |
|---|---|
| `third_party/cpu/lib/TritonCPUTransforms/SplitLargeContract.cpp` | 新增 pass 实现 |
| `third_party/cpu/include/TritonCPUTransforms/Passes.td` | 注册 pass 定义 |
| `third_party/cpu/include/TritonCPUTransforms/Passes.h` | 声明 create 函数 |
| `third_party/cpu/lib/TritonCPUTransforms/CMakeLists.txt` | 加入编译 |
| `third_party/cpu/triton_cpu.cc` | Python binding |
| `third_party/cpu/backend/compiler.py` | 插入 AOT pipeline |

## 环境变量

- `TRITON_MICRO_M`：控制微块行数，默认 4。设为 ≥ BLOCK_SIZE_M 可禁用拆分。

## DMA 再灌注（Re-priming）

### 问题

`ConvertMemoryToSPM` 在 K-loop 之前发出 prologue DMA enqueue，将第一个
K-tile 的 A/B 搬入 SPM buffer 0。拆成 8 个独立 K-loop 后，只有第 1 个
micro-loop 能读到正确的 prologue 数据；第 2–8 个 micro-loop 启动时
SPM buffer 0 里是上一个 micro-loop 最后一轮的残留数据。

### 修复

在每个 micro-loop（第 2 个起）之前，clone 原始 prologue DMA enqueue ops，
重新将第一个 K-tile 搬入 SPM buffer 0。

注意：不能只 clone A、跳过 B。上一条 micro-loop 跑完整个 K-loop 后，
B buffer 0/1 中保留的是末尾 K tile；下一条 micro-loop 若从 `buf_idx=0`
读取残留 B，会把 K=0 的 B tile 错用成末尾 tile。这也是
`docs/plans/spm-dma-reuse.md` 采用 fused micro-scheduler 的原因。

收集 prologue DMA 的方法：从 `origFor` 的前一个 op 开始向前扫描，收集
连续的 `triton::cpu::DmaEnqueue2DOp`，遇到非 DMA 且有 side effect 的 op
就停止。跳过 `isPure()` 的 op（如 `arith.constant`），因为
`ConvertMemoryToSPM` 会在 DMA enqueue 和 loop 之间插入 `initBufIdx = i64Cst(0)`。
收集后反转顺序，保持原始 A/B enqueue 顺序。

不 clone `DmaWaitOp`：`ConvertMemoryToSPM` 已经把 wait 放在 loop body 顶部，
每个 micro-loop 首轮迭代会自动等待前面的 prologue DMA 完成。

### 调试历程

1. 初始实现（intra-loop 方案）：在同一 K-loop 内拆 contract 为 8 个小
   contract + 8 个 `[4 x <32xf32>]` iter-args。结果正确但性能崩溃
   （6.12M cycles，比 cache 还慢 10%），因为 LLVM 仍然看到 128 个 phi
   节点并大量 spill。
2. 改为 separate K-loops + DMA re-prime：`collectPrologueDma` 最初只收集
   `DmaEnqueue2DOp`，遇到任何非 DMA op 就停止。但 `ConvertMemoryToSPM`
   在 DMA enqueue 和 loop 之间插入了 `arith.constant 0`（initBufIdx），
   导致收集到 0 个 DMA op，re-prime 未生效。
3. 修复：在 `collectPrologueDma` 中加入 `isPure()` 跳过分支，跳过
   side-effect-free 的 op。需要 `#include "mlir/Interfaces/SideEffectInterfaces.h"`。

## 实测结果

### 代码形态（256×256×256 / 32×32×32 tile）

| 指标 | 拆分前（cache baseline） | 拆分后（SPM） |
|---|---|---|
| 累加器 phi 类型 | `[32 x <32xf32>]` | `[4 x <32xf32>]` × 8 |
| 累加器物理 vreg | 128 | 16 per loop |
| vl4r.v (spill reload) | 928 | 38 |
| vs4r.v (spill store) | 960 | 70 |
| vfmacc.vf | 1,024 | 1,024 |
| flw (scalar A load) | 1,457 | 1,024 |
| vl4re32.v (vector B load) | 32 | 256 |
| K-loop 内层体大小 | ~8,000 行 | ~200 行 × 8 loops |

Spill 从 928/960 降到 38/70（-96%）。B 的 vector load 增加 8× 是因为
每个 micro-loop 独立重新加载完整 B tile。

### gem5 仿真（256×256×256 / 32×32×32）

| 指标 | SPM | Cache | Δ |
|---|---|---|---|
| numCycles | 3,777,998 | 5,560,678 | **-32.1%** |
| simInsts | 1,805,834 | 5,012,404 | -64.0% |
| IPC | 0.478 | 0.901 | -47.0% |
| l1d.demandMisses | 10,807 | 425,626 | -97.5% |
| l2.demandMisses | 1,531 | 18,540 | -91.7% |
| issued.MemRead | 4,874,680 | 60,178 | +8000% |
| issued.FloatMemRead | 2,086,443 | 756,107 | +176% |

功能验证：PASS（65536/65536 正确）。

### issued.MemRead 为什么高 80×

SPM 的 `issued.MemRead` = 4.87M 几乎全部来自 **DMA 轮询**。每个 K-iteration
开头有一个 tight polling loop：

```asm
.LBB0_6:
    ld t1, 24(a6)       # 读 DMA status register (MMIO)
    beqz t1, .LBB0_7
    j .LBB0_6
```

共 4096 次 DMA wait（64 tiles × 8 micro-loops × 8 K-iters）。O3 CPU 的
分支预测器持续预测 "继续轮询"，投机发射大量 `ld` 微操作，即使最终被
squash 也计入 `issued.MemRead`。平均每次 wait 约 1190 次投机 poll。

Cache 模式没有 DMA，integer load 仅来自循环控制（~60K）。

这不影响性能：CPU 本来就在等 DMA 完成，轮询 load 命中 MMIO 寄存器，
不污染 cache。
