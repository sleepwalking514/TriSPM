# TriSPM Docs Index

## 当前工作 Head（2026-04-30）

**已完成**：Phase 3 matmul P3 cold-start headline 收敛。大尺寸 1024³/32³ SPM 比 cache 快 25.1%（288M vs 386M cycles）；64×64 smoke +3.7% 在 ≤1.05× 回归线内。Stage 1–3 + 2.5 + 2.6 全部落地（fair baseline、MMIO packing、prologue wait 消除、size/steady sweep）。SplitLargeContract pass 已验证。Tier sidecar 覆盖审计完成（`3tier.md` §4.1）。`verify-spm-fires` 工具已落地。**Stage 4 完成**：`transformReductionLoop` 2-D 非连续 leading dim prefetch 地址 bug 已修复（使用 IV 所索引维度的 stride 而非始终取 strides[0]），新增 lit 测试覆盖。**Stage 5 完成**：`dma_l2_warming` microbenchmark 已跑通，L2-warming 效应在 4K–32K tile 全尺寸确认（随机访问 2.8× 加速，Phase B 近 100% L2 命中 vs Phase D 近 100% miss）。

**当前阻塞/待做**（按 [`plans/next_steps.md`](plans/next_steps.md) 推荐顺序）：
1. ~~**Tier sidecar 覆盖审计**~~ ✅ 完成。matmul 正常；vector_add 无循环（设计预期）；layer_norm 需 kernel 改写 + matcher 泛化
2. ~~**评测工具化**~~ 部分完成。`verify-spm-fires` ✅、`make cmp-<kernel>` ✅、stats CSV export ✅（`--csv` / `--spm-only-csv`）
3. ~~**Reduction 2D 地址 bug**~~ ✅ 完成。`transformReductionLoop` 内 IV 维度 stride 查找已修复 + lit 测试
4. ~~**L2-warming microbenchmark**~~ ✅ 完成。随机访问模式确认 L2-warming 2.8× 加速，数据见 [`evidence/l2_warming.md`](evidence/l2_warming.md)
5. **Reduction 双缓冲流水** — 当前 reduction 路径是单缓冲串行，DMA 延迟全暴露（依赖 #3 ✅）
6. **Compiler robustness backlog** — robust GEMM matcher、reduction matcher 泛化、DmaOpsToLLVM 选项

---

按生命周期分三档管理项目文档：活跃路线图、验证证据、被取代的快照。
所有文档都为之后写 paper 服务，**优先保留决策路径而不是合并**。

## Layout

| 目录 | 用途 |
| --- | --- |
| [`architecture/`](architecture/) | 长期架构说明 — 给新 AI agent / paper 写作快速建立 simulator 和系统设计上下文 |
| [`plans/`](plans/) | 活跃路线图、todo、设计 — 当前还在执行或仍可能被引用 |
| [`evidence/`](evidence/) | 验证计划、实验数据、per-checkpoint 记录 — paper 的 claim 仓库 |
| [`archive/`](archive/) | 被新文档取代的快照 — 头部标注 superseded |

## Architecture (`architecture/`)

| 文件 | 角色 |
| --- | --- |
| [`architecture/simulator-spm-architecture.md`](architecture/simulator-spm-architecture.md) | Simulator-side SPM architecture map: `stable..spm-dev` diff summary, SPM/DMA/Xspm/O3-LSQ topology, code entry points, and paper-facing design notes |

## Active plans (`plans/`)

| 文件 | 角色 |
| --- | --- |
| [`plans/plan.md`](plans/plan.md) | Master roadmap — Phase 1-6 完整设计 |
| [`plans/phase3.md`](plans/phase3.md) | Phase 3 current status page — 已完成、当前 blocker、placement 状态 |
| [`plans/next_steps.md`](plans/next_steps.md) | 当前执行顺序指针，主线任务列表 |
| [`plans/todo.md`](plans/todo.md) | Phase 3 audit & 剩余 compiler robustness backlog |
| [`plans/spm-lowering.md`](plans/spm-lowering.md) | matmul SPM lowering 优化（P0-P3，DMA latency gap） |
| [`plans/3tier.md`](plans/3tier.md) | 三层 placement 设计 + MVP 里程碑（Tier 1/2/3） |

## Evidence (`evidence/`)

| 文件 | 角色 |
| --- | --- |
| [`evidence/l2_warming.md`](evidence/l2_warming.md) | Tier 2 L2-warming 验证 — 源码分析 + 微基准实验数据（2.8× 加速确认） |

## Archive (`archive/`)

| 文件 | 取代来源 |
| --- | --- |
| [`archive/phase2.md`](archive/phase2.md) | 内容拆入 [`plans/todo.md`](plans/todo.md) §Phase-3 plumbing 与 [`plans/3tier.md`](plans/3tier.md) |

---

## 判断规则

新建或处理文档时按以下三条决定放哪。规则的目的是**保留决策路径**，让以后写 paper 时能复盘"我们为什么没走另一条路"。

### 1. 新分支文档默认进 `plans/`

当原计划发现问题、需要拆出新文档继续推进时：
- 直接在 `plans/` 下新建（命名建议短，去掉 `-todo` 后缀）。
- 在 `plans/next_steps.md` 或对应母文档里加一行链接说明衔接关系。
- **不要**把新分支的内容塞回母文档 — 那会冲淡 motivation。

### 2. 老文档被取代时进 `archive/`，并加 superseded header

当一份文档不再是行动依据（被拆解、被新版本取代、阶段结束）：
- 移到 `archive/`。
- 在文件顶部加这种 block：
  ```markdown
  > **Status: SUPERSEDED — YYYY-MM-DD**
  > 一句话说明这份文档原本承载什么。
  > - 关键内容 A → 迁移到 [`../plans/X.md`](../plans/X.md) §章节
  > - 关键内容 B → 演化为 [`../plans/Y.md`](../plans/Y.md)
  >
  > 保留作为 ___ 的快照，不再是行动文档。
  ```
- **必须写明 reason / migration target**，否则 archive 等于丢失。
- 不要删 — 这份文档可能就是 paper 里 "we initially attempted X but ..." 的依据。

### 3. 有数据 / 实验 / 验证的文档进 `evidence/`

凡是产出可发表数据的文档（即使实验还没跑）：
- 微基准计划、per-checkpoint stats、sweep 结果、对照实验。
- 哪怕只是"实验设计 + 预期结果"也算 — 真跑出来后把数据补回同一份。
- **不要**和 `plans/` 混 — `plans/` 是 "下一步要做什么"，`evidence/` 是 "做出来 / 要做出来给 paper 引用的东西"。

### 4. 跨档移动时同步引用

任何一次 plans ↔ archive ↔ evidence 的迁移，都要：
- 用 `grep -rE "<旧文件名>" docs/` 检查是否还有反向引用。
- 全部更新成新路径（同档同级用 bare filename，跨档用 `../<dir>/`）。
- 不留 dangling link — 否则索引就开始衰退。
