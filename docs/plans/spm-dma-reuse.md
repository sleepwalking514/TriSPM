# SPM DMA Reuse 优化计划 — 终极版 C：融合式 micro-scheduler

## 结论

原先的“只在 `SplitLargeContract` re-prime 时跳过 B prologue DMA clone”
不是正确方案。它只能减少每个 output tile 中 micro-loop 之间的 B0
重新灌注，无法消除 K-loop body 内每轮 B prefetch 的重复搬运，而且会让
后续 micro-loop 从残留的 B double-buffer 内容开始计算，存在正确性风险。

本计划采用终极版 C：在 SPM GEMM lowering 阶段直接生成 microM-aware 的
融合式调度，同时做到：

- B tile/window 在一组 microM 循环之间只搬一次
- A 只搬当前 microM 行，而不是完整 BM 行
- accumulator 放入 SPM，在 micro-loop 之间保存完整 C tile
- 保持每个 micro-loop 的寄存器压力为 `microM x BN`

## 当前问题

1024x1024x1024 matmul 在 64x64x32 tile 下：

| 指标 | 当前值 |
|---|---:|
| SPM cycles | 225M |
| Cache cycles | 292M |
| SPM 优势 | -23.0% |
| DMA transfers | 262,144 |
| DMA bytes | 2.15 GB |
| DMA wait fraction | 46.7% |
| SPM 使用量 | 32 KiB / 256 KiB |

根因不是单一 prologue DMA，而是 `ConvertMemoryToSPM` 先生成 full-tile
double-buffered K-loop，随后 `SplitLargeContract` 把整条 K-loop 复制成
`BM / microM` 个 micro-loop。结果每个 micro-loop 都独立搬运 A/B：

- B 对所有 microM 块完全相同，却被重复搬运
- A 每次搬完整 `BM x BK`，但 contract 只使用其中 `microM x BK`
- DMA wait 被复制到每个 micro-loop，polling 成本也随之放大

## 为什么旧方案不做

旧方案建议：在 `SplitLargeContract` 的 re-prime 中只 clone A DMA，跳过 B DMA。

这个方案有三个问题：

1. 收益远低于预期  
   `SplitLargeContract` re-prime 只覆盖 micro-loop 开头的 prologue B0。
   K-loop body 中的 B prefetch 仍然被每个 micro-loop clone，262,144
   transfers 最多只减少几千次，而不是降到 139,264。

2. 正确性不稳  
   micro-loop 0 跑完整条 K-loop 后，B0/B1 中保留的是最后几个 K tile。
   micro-loop 1 如果不重新灌 B0，却仍从 `buf_idx=0` 开始读，会把 K=0
   的输入错误替换成末尾 K tile 的残留。

3. 依赖隐式顺序  
   当前 `ConvertMemoryToSPM` 确实按 A0、B0 顺序发 prologue DMA，但把
   correctness 建在 `prologueDma[0]`/`[1]` 上不适合长期维护。

## 终极方案 C

在 `ConvertMemoryToSPM` 的 GEMM lowering 中，当 accumulator 的 M 维大于
`microM` 时，不再先生成 full-tile SPM K-loop 再交给
`SplitLargeContract` 复制；而是直接生成 microM-aware 的 SPM GEMM schedule。

目标调度：

```text
acc_spm = zero full BM x BN tile in SPM

for kWindow in K by windowK:
  DMA B[kWindow] into resident B buffers once

  for mOff in 0..BM step microM:
    microAcc = load acc_spm[mOff:mOff+microM, :]

    for k in kWindow:
      DMA A[mOff:mOff+microM, k] only
      read resident B[k]
      microAcc = contract(A_micro, B, microAcc)

    store microAcc back to acc_spm[mOff:mOff+microM, :]

finalAcc = load full acc_spm
store finalAcc to C
```

### Window 大小

第一版使用 `windowK = min(4, K / BK)`。

原因：当前 DMA engine 默认 `max_descriptors=4`。一次性提交过多 B resident
descriptor 会触发 queue full，并且当前实现会 drop descriptor。因此第一版以
4 个 BK tile 为窗口，和硬件队列深度匹配。

后续可以把 windowK 变成 pass option 或从 DMA engine 参数传入。

### SPM 空间估算

以 64x64x32, microM=4, windowK=4 为例：

| Buffer | 大小 |
|---|---:|
| B resident window | `4 * 32 * 64 * 4 = 32 KiB` |
| A micro buffer | `4 * 32 * 4 = 512 B` |
| accumulator full tile | `64 * 64 * 4 = 16 KiB` |
| 合计 | 48.5 KiB |

128x128x32, microM=4, windowK=4：

| Buffer | 大小 |
|---|---:|
| B resident window | 64 KiB |
| A micro buffer | 512 B |
| accumulator full tile | 64 KiB |
| 合计 | 128.5 KiB |

仍然小于 256 KiB SPM。

## 预期收益

对 1024x1024x1024 / 64x64x32：

| 指标 | 当前 | 目标 |
|---|---:|---:|
| DMA transfers | 262,144 | 约 139,264 |
| DRAM DMA bytes | 2.15 GB | 约 128 MiB |
| B DRAM DMA | 重复 16 次 | 每个 K tile 1 次 |
| A DRAM DMA | 每次 64 行 | 每次 4 行 |

目标 transfers 由两部分构成：

- B：`output_tiles * K_iters = 256 * 32 = 8,192`
- A micro：`output_tiles * micro_loops * K_iters = 256 * 16 * 32 = 131,072`

合计 139,264。

## 实施路线

### Step 1：文档和防误导

- 更新本文件，废弃旧方案 1
- 在 [`../archive/split-large-contract.md`](../archive/split-large-contract.md) 中修正“B 不重复搬运”的历史描述

### Step 2：实现新的 GEMM lowering

文件：`compiler/third_party/cpu/lib/TritonCPUTransforms/ConvertMemoryToSPM.cpp`

新增 microM-aware GEMM path：

- 从 `vector.contract` 分析 `BM/BN/BK`
- 当 `BM > microM` 且 SPM 空间足够时，走 fused micro-scheduler
- 生成 A micro DMA descriptor，而不是 full A DMA descriptor
- 生成 B resident window descriptor
- 生成 acc_spm 的 full-tile load/store
- 让 `SplitLargeContract` 不再需要处理这条 GEMM loop

### Step 3：保留 fallback

如果下列条件不满足，回退到当前 double-buffered lowering：

- loop trip count、step、tile shape 必须静态可知
- accumulator 必须是 rank-2 vector
- contract indexing maps 必须是标准 GEMM
- `BM > microM`
- SPM 空间必须容纳 B window、A micro buffers、acc tile

### Step 4：验证

功能：

- `make cmp-matmul`
- 1024x1024x1024 / 64x64x32 with `CHECK_RESULT=1`

性能：

- DMA transfers 目标约 139,264
- DMA bytes 目标约 128 MiB
- wait fraction 应显著低于当前 46.7%

回归：

- 32x32x32
- 64x64x32
- 128x128x32

## 后续优化

- `windowK` 自动调参：根据 SPM 容量和 DMA queue depth 选择 2/4/8
- B window 内部可继续做 double-buffer/pipeline
- A micro DMA 可按 rows 合并或用专门的 2D descriptor 优化 MMIO 开销
- A micro pipelining 需要轻量实现。一次性把 window 内所有 A micro tile
  stage 到 SPM 的实验版本 64³ 正确，但 512³ 明显变慢，说明额外 loop/control
  开销超过了减少 wait 的收益；不进入本轮主线。
- launcher/grid 层面的跨 program A tile reuse 作为更后续的优化，不进入本轮

## 当前实现状态

已实现第一版 correctness-first fused scheduler：

- `ConvertMemoryToSPM` 新增 `microM`/`windowK` pass option
- Python backend 从 `TRITON_MICRO_M` 和 `TRITON_SPM_WINDOW_K` 读取参数
- GEMM loop 优先走 fused micro-scheduler，失败时回退旧 double-buffer path
- B 以 window 为单位常驻 SPM，A 每次只 DMA `microM x BK`
- accumulator full tile spill 到 SPM，在 micro-loop 之间保存
- 默认 `microM=8`。初版 `microM=4` 正确但 descriptor/wait 偏多；
  512³ 调参显示 `microM=8` 更优，`microM=16` build 可行但 gem5 明显变慢，
  不作为默认。

验证结果：

| Case | 结果 |
|---|---:|
| 64³ / 32x32x32 / CHECK_RESULT=1 | PASS，4096/4096 正确 |
| 64³ DMA transfers (`microM=8`) | 40 |
| 64³ DMA bytes | 65,536 B |
| 512³ / 64x64x32 / CHECK_RESULT=0 (`microM=8`) | 完成 |
| 512³ cycles | 18,652,222 |
| 512³ vs cache cycles | -48.5% |
| 512³ DMA transfers | 9,216 |
| 512³ DMA bytes | 16 MiB |
| 512³ queueFullStalls | 0 |

512³ 的 transfers/bytes 与公式吻合。按默认 `microM=8`，1024³ / 64x64x32
应为 73,728 transfers、128 MiB DMA bytes。1024³ perf run 已能 build
并进入 gem5，但 correctness-first 版本在大规模下仿真耗时过长，本轮先不把
1024³ 完整 perf 作为阻塞项。
