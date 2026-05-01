---
name: TriSPM Phase 3 Execution Timeline
overview: `../archive/matmul-spm-lowering-closure.md` 的 matmul P3 cold-start 主线已收敛；Tier sidecar、L2-warming、评测工具化、reduction 2-D 地址修复、single-load reduction 双缓冲和 multi-load reduction matcher 均已完成。当前剩余 compiler 主线是 bail-out cleanup/verification。
tasks:
  - id: p3-profile-overlap
    content: 画清 matmul prefetch enqueue、current buffer 读取、下一轮 dma_wait 成功之间的时间线，确认是否存在真实 overlap。
    status: completed
    note: Stage 1 完成。结论见 `../archive/matmul-spm-lowering-closure.md` §P3.1：DMA 已 62.3% 重叠（avgWaitStall 60.68/avgLatency 160.92），瓶颈不在 prefetch 完成，而是 MMIO descriptor stores 引发的 back-end 阻塞（占 cycle gap 68.8%）。
  - id: fair-baseline-flush
    content: harness 在 measured region 之前 flush cache（Zicbom 或 scrub buffer）、加速 SPM 端 host check（DMA copy 到 cacheable）；SPM/cache 两侧 cold-cache 重测，更新 ../archive/matmul-spm-lowering-closure.md §P3.0/§P3.1。
    status: completed
    note: Stage 2.5 完成。harness 加 publish_input + flush_caches；cold-cache fair baseline gap 从 +75.9% 降到 +6.7%。结果记入 `../archive/matmul-spm-lowering-closure.md` §P3.0。后续 Stage 2.6 已补 sweep 数据；cold-start 是 matmul P3 的关键口径，steady-state warm-cache 仅作辅助参考。
  - id: p3-wait-semantics
    content: 在确认 dma_wait 等待对象后，再判断是否存在可跳过的冗余 wait；不要先假设最后一轮顶部 wait 一定安全可删。
    status: completed
    note: Stage 3 完成 prologue wait 移除：body 顶部 wait 已经覆盖 prologue DMAs，prologue wait 是冗余的。其他 4 个 body waits 经映射确认每个都在等待来自 prior iter / prologue 的具体 buffer，不可删。
  - id: p3-reduce-wait-cost
    content: 若仍有明显差距，减少每次 DMA enqueue 的 MMIO store 数量（如合并 descriptor 写入）。
    status: completed
    note: Stage 3 完成 MMIO descriptor packing：新增 REG_STRIDES_PACKED (0x38) + REG_LEN 上 32 位接收 HEIGHT。每个 DMA descriptor 从 6 stores 降到 4 stores。SPM kernel cycles 从 39,480 → 38,361 (-1,119 cy)，gap +6.7% → +3.7%。
  - id: p3-prefetch-timing
    content: 尝试在 iteration body 中更早发出 prefetch，让 DMA 有更多计算时间来隐藏延迟（与 MMIO 优化是独立方向）。
    status: pending
    note: 暂未启用：Stage 3 后剩余 wait stall ~4.7K cy 主要来自 64×64 算量太小、单 iter compute 无法摊销 161-cy DMA 延迟，结构性问题。Stage 2.6 已确认大尺寸 cold-start 下 SPM beat cache；本项不再是 P3 blocker，只作为未来小尺寸优化 lever 保留。
  - id: steady-state-and-size-sweep
    content: harness 加 N-iter steady-state 测量模式 + 128×128/256×256 size sweep；补 `../archive/matmul-spm-lowering-closure.md` §P3.5 的 warm-cache 与大尺寸数据。
    status: completed
    note: Stage 2.6 完成。统一 `make cmp-matmul` / `make sweep-matmul SWEEP=size` 入口已落地并补跑；steady-state warm-cache 不是 P3 headline 口径，只作为 cache residency 的辅助参考。当前可引用的大尺寸 cold-start 记录为 1024×1024×1024 / 32×32×32：SPM 288,976,339 cycles vs cache 386,049,495 cycles，SPM 快 25.1%。
  - id: tier-sidecar-verify
    content: 解释 `three-tier-placement.md` 与当前生成物的状态冲突：重新 build 三个 workload，核对 `_tiers.json` 和 launcher 分配路径；同时回溯 M8 验证时的实际输出，确认当时是否真的通过还是验证标准过宽。空 JSON 需定位是 pass 未跑、matcher 未命中还是无候选 tensor。
    status: completed
    note: 2026-04-29 审计完成，2026-05-01 刷新。matmul 正常（args 0,1 → Tier 3）；vector_add 空 JSON 是设计预期（单 block 无循环，无 tile reuse）。layer_norm 的 mean/variance/final normalize 都已改成 block pointer，当前 layer_norm args 0,1,2 → Tier 3，作为 reduction-path coverage workload。详见 `three-tier-placement.md` §4.1。
  - id: reduction-single-buffer-pipeline
    priority: high
    content: 升级 `transformReductionLoop` 单缓冲 prefetch 为真双缓冲（layer_norm/softmax/未来 reduction kernel），让 DMA 与 CPU 计算真正流水起来；当前实现是 read-then-prefetch 串行，DMA 延迟完全暴露在关键路径上。
    status: completed
    note: 2026-04-30 完成 single-load 双缓冲；2026-05-01 multi-load matcher 覆盖 final normalize。`transformReductionLoop` 已拆成两个 SPM buffer，prologue 发首块 DMA，body 顶部 `dma_wait` 等当前 buffer，随后向 alternate buffer 发 next prefetch 并翻转 `buf_idx`。lit 测试覆盖 body-top wait / buffer select / alternate-buffer enqueue / 2-D non-leading-IV stride / multi-load shared-IV streams。`make verify-layer_norm` 通过（38 `addrspace(3)`，78 `fence iorw`，tier JSON `{\"0\":3,\"1\":3,\"2\":3}`），`make cmp-layer_norm` 功能 PASS。32x64 ROI 基线：SPM 125,593 cycles vs cache 6,138 cycles；1,280 DMA transfers / 40,960 bytes；waitFraction 0.4002、avgWaitStall 65.45、avgLatency 66.23。该数据证明路径正确并暴露小尺寸 MMIO/DMA 控制开销，不作为 performance win。
  - id: l2-warming-bench
    content: 实现并运行 `dma_l2_warming` microbenchmark，接通 per-checkpoint stats parser，并做 cacheable/UC/无 DMA 对照与 working-set sweep。前置条件：tier-sidecar-verify 确认 Tier 2 分配链路正常。
    status: completed
    note: 2026-04-30 完成。随机访问模式（Fisher-Yates shuffle）击败 L2 prefetcher，清晰展示 warming 效应。4K–32K sweep 全尺寸确认 Phase B（DMA 后）近 100% L2 命中 vs Phase D（冷）近 100% miss，2.8× cycle 加速。数据记录于 `../evidence/l2_warming.md`。
  - id: eval-tooling
    content: 补 `verify-spm-fires`、统一 run/compare target、stats CSV export 和后续 Phase 6 对比工具。
    status: completed
    note: 2026-04-29 `verify-spm-fires` 已落地（`make verify` / `make verify-<kernel>`，`run_experiment.py --mode verify`）。统一 run target 已由 `make run-<kernel>` / `make cmp-<kernel>` 覆盖。Stats CSV export 已完成（`compare_stats.py --csv` / `--spm-only-csv`）。剩余：Phase 6 对比工具。
  - id: reduction-2d-addr-bug
    content: 修复 `transformReductionLoop` 2-D 非连续 leading dimension 时 prefetch DRAM offset 计算错误（`phase3-compiler-backlog.md` §C.2），这是正确性 bug 而非 robustness 问题，应在 reduction matcher 泛化之前修复。
    status: completed
    note: 2026-04-30 修复。根因：prefetch 地址计算始终使用 strides[0]（leading stride），但 IV 可能索引非 leading 维度。修复为动态查找 IV 所索引维度的 stride。新增 lit 测试 `@reduction_2d_non_leading_iv`（memref<8x64xf32> IV 在 dim 1）验证 byte offset 使用 stride=1*4=4 而非 64*4=256。
  - id: compiler-robustness-backlog
    content: "P3 收敛后补 compiler robustness：robust GEMM matcher 已完成 A/B lhs/rhs 识别、`IRMapping` cloned-read lookup 和 extra-load 覆盖；reduction matcher 已泛化到多 load 共享 loop IV；`DmaOpsToLLVM` 已增加 MMIO base/useXspmInsn 选项。剩余：修复/验证 bail-out 路径 prologue DMA 残留。"
    status: pending
  - id: tier1-backlog
    content: 在 P3 和 Tier 2 证据链稳定后，规划 `three-tier-placement.md` §6.1 的 Tier 1 resident SPM 完整实现。
    status: pending
  - id: reuse-rules-backlog
    content: 仅在新 workload 需要时扩展 `three-tier-placement.md` §6.2 的 `has_scalar_reuse` 规则。
    status: pending
  - id: doc-refresh
    content: P3 收敛后统一刷新 `phase3.md` 和 `phase3-compiler-backlog.md` 中已解决但仍标记为 NOT VERIFIED / NOT IMPLEMENTED 的条目，消除文档状态滞后。
    status: completed
    note: 2026-04-29 完成。README head section、phase3-execution-timeline.md task 状态、phase3.md 近期优先级均已同步刷新。
isProject: false
---

# TriSPM Phase 3 Execution Timeline

## 当前判断
- Phase 1 / Phase 2 基础完成，Phase 3 的 GEMM SPM lowering 已能在 `matmul` 产出真实 DMA/SPM 代码。
- `phase3.md` 已更新为当前状态页：记录 P1/P2 plumbing、matmul P3 cold-start headline、reduction 双缓冲、multi-load matcher 与 layer_norm coverage baseline。
- `../archive/matmul-spm-lowering-closure.md` 的 P0-P2 已完成：matmul 汇编形态已修复，`vfmacc.vf=256`、`vrgather=0`、spill/reload 与 cache baseline 对齐。
- `SplitLargeContract` pass 已实现并验证：256×256×256 / 32×32×32 tile 下 SPM 比 cache 快 32.1%（3,777,998 vs 5,560,678 cycles）。通过 DMA 再灌注（re-priming）解决了多 micro-loop 的正确性问题。
- P3 已通过 Stage 2.5（fair cold-cache baseline）+ Stage 3（MMIO packing + 移除 prologue wait）+ Stage 2.6（size/steady sweep）收敛：64×64 cold-start smoke 仅慢 +3.7%（38,361 vs 37,004 cycles），大尺寸 cold-start 1024×1024×1024 / 32×32×32 已反超 cache（288,976,339 vs 386,049,495 cycles，-25.1%）。P3 headline 采用 cold-start 口径；steady-state warm-cache 仅作辅助参考，不作为关键对象。
- `three-tier-placement.md` 的 MVP 代码框架基本落地。当前覆盖：`matmul` 正常命中 Tier 3（args 0,1）；`vector_add` 空 JSON 是设计预期（单 block 无循环）；`layer_norm` 在 mean/variance/final normalize 命中 Tier 3（args 0,1,2）。详见 `three-tier-placement.md` §4.1。
- 文档中"三个 workload 全部命中 Tier 3"的说法已修正。`verify-spm-fires` 工具已落地：`matmul` 与 `layer_norm` 通过 `make verify-<kernel>`；`vector_add` 预期未命中 SPM（见 `three-tier-placement.md` §4.1）。
- `../evidence/l2_warming.md` Tier 2 L2-warming 验证完成（源码分析 + 微基准 2.8× 加速数据）。
- `phase3-compiler-backlog.md` 里的 P1 稳健性事项已大幅收敛：GEMM >2 loads 和 reduction matcher 泛化已完成，剩余 bail-out 清理/验证仍会影响后续 workload 扩展。`DmaOpsToLLVM` 的 MMIO base/useXspmInsn 选项已完成。
- `three-tier-placement.md` §6 是明确 backlog：§6.3 的验证补完已由工具化和 L2-warming evidence 覆盖；§6.1 Tier 1 和 §6.2 reuse 规则扩展应排在 reduction 性能和 Tier 2 数据闭环之后。

## Timeline Reading Order

> **P3 成功标准**：`matmul` 以 cold-start fair baseline 为主口径。64×64 smoke case 用来守住代码形态和小尺寸回归（≤ cache × 1.05），论文 headline 应采用已实测的大尺寸 cold-start 数据；steady-state warm-cache 是 cache residency 辅助实验，不是 P3 关键对象。

> **可并行**：Tier sidecar 核查、L2-warming microbenchmark、评测工具化和 reduction 正确性修复互不完全依赖，可按资源并行推进。

1. Done: treat matmul P3 as closed under the cold-start headline metric. Keep `p3-prefetch-timing` only as a future small-size optimization lever.
2. Done: resolve Tier sidecar coverage mismatch. `matmul` hits Tier 3; `vector_add` is intentionally empty; `layer_norm` now hits Tier 3 for mean/variance reduction and final normalize.
3. Done: complete Tier 2 / L2-warming evidence via `../evidence/l2_warming.md`.
4. Done: complete the Phase 3 tooling baseline: `make verify`, unified `make run-<kernel>` / `make cmp-<kernel>`, and stats CSV export. Remaining Phase 6 comparison tooling moves to the roadmap.
5. Done: fix the `transformReductionLoop` 2-D non-leading-IV prefetch address bug.
6. Done: implement reduction double-buffer pipelining (`reduction-single-buffer-pipeline`) and record the first `layer_norm` SPM coverage/perf baseline.
7. Current: finish remaining `phase3-compiler-backlog.md` P1 robustness work: bail-out cleanup/verification.
8. Later: after Tier 2 evidence and reduction performance stabilize, enter `three-tier-placement.md` §6.1 for Tier 1 resident SPM and §6.2 for workload-driven scalar-reuse rule expansion.

## 阶段化执行计划

每个阶段独立可验证、改动范围受控；除非显式声明，前一阶段不阻塞后一阶段（标注 *依赖* 的除外）。

### Stage 1 — P3 timeline 诊断（completed）
- **范围**：基于已有 `m5out/spm-matmul-stats.txt` 与 `cache-matmul-stats.txt` 拆解 16K+ cycle 差距来源（`waitStallCycles`、MMIO traffic、额外指令、未解释余量）；推断 prefetch 是否真的与 FMA 重叠、loop 顶部 `dma_wait` 当前等的对象。
- **交付物**：在 `../archive/matmul-spm-lowering-closure.md` 新增 "P3 Timeline" 小节，给出 cycle 预算表 + 单次 iteration 时序示意 + Stage 3 候选干预的优先级排序。
- **验证**：分析覆盖 ≥80% 的 cycle 差距；结论可由现有 stats 直接复现，不引入新 build。
- **范围限制**：仅写 doc / 小脚本；不改编译器、不改 simulator。
- **对应 task**：`p3-profile-overlap`、`p3-wait-semantics`（诊断部分）。

### Stage 2 — Tier sidecar 覆盖审计（completed 2026-04-29）
- **结论**：matmul 正常（args 0,1 → Tier 3）；vector_add 空 JSON 是设计预期（单 block 无循环）。该审计时 layer_norm 还因 pointer arithmetic 未命中；后续已把 mean/variance/final normalize 改为 block pointer，当前 layer_norm args 0,1,2 → Tier 3。详见 `three-tier-placement.md` §4.1。
- **交付物**：`three-tier-placement.md` §4.1 覆盖审计表 + `make verify` / `make verify-<kernel>` 工具。
- **对应 task**：`tier-sidecar-verify`（completed）。

### Stage 2.5 — Fair baseline (completed)
- **背景**：当前 harness 在 init / reference 阶段写过 A、B，cache 是 write-allocate，kernel 启动时 L1/L2 已被加热；64×64 matmul 的 A+B≈32 KiB 完全装得下 L1d。SPM 这边却要从 DRAM 重新搬，这一段差距是初始化阶段副产物，不是 scheme 本身性质。Stage 1 的 16,479 cy gap 数值在公平性修正前需打折看。
- **范围**：
  - (a) 在 measured region 之前、`m5_reset_stats()` 之后插入 cache 清扫（首选 RISC-V Zicbom `cbo.flush` 逐 line；若 gem5 实现不完整，退回到 scrub buffer：分配 > L2 大小、逐 line 触摸），SPM 与 cache 两侧都要做。
  - (b) 加速 harness verification：当前 SPM 走 uncacheable DMA buffer，host 端 `for` 比对极慢。改为 measured region 结束、`m5_dump_stats()` 之后用 DMA 把结果搬到 cacheable buffer 再 host 比较；或直接把 reference 也算到 SPM 输出 buffer 邻近的 cacheable 镜像中比较。**关键约束**：所有 DMA / cacheable copy 都必须在 stats 区外完成，不能污染 measured cycles。
  - (c) 主跑 cold-cache 一种作为主表；steady-state（同 kernel 跑 N 次取稳态）作为 follow-up，留 stub。
- **交付物**：harness 改动（`workloads/matmul/...` + 共享 helper） + 一份 fair-baseline 数据表（cold-cache SPM vs cold-cache cache）写进 `../archive/matmul-spm-lowering-closure.md` §P3.1 之后的新小节 `P3.0 Fair baseline`，并在 §P3.1 顶部加 caveat 说明原数据未做 flush。
- **验证**：
  - check 阶段在 SPM mode 下耗时显著下降（grep host wall-clock 或 gem5 simSeconds 区分 measured vs unmeasured）；功能 PASS 不退化。
  - cache baseline 在 flush 后 `numCycles` 显著上升（cold misses 进入测量区），可由 `dcache.overall_miss_rate` 与 `l2.overall_misses` 同步上升交叉验证。
  - SPM 在 flush 后 `numCycles` 几乎不变（输入路径本就走 DMA，不依赖 cache 预热）。
- **范围限制**：harness-only；不改编译器、不改 simulator MMIO 行为。仅当 Zicbom 不可用时引入 scrub buffer，其他情况不引入新构建路径。
- **对应 task**：`fair-baseline-flush`、配套 harness check 加速。

### Stage 2.6 — Steady-state warm-cache 复测 + 大尺寸 sweep（completed）
- **背景**：Stage 2.5 给的是 cold-cache 公平 baseline。补跑 steady-state 与 size sweep 后，P3 口径明确为 cold-start：steady-state warm-cache 主要测 cache residency，不作为 matmul P3 的关键对象。
- **范围**：(a) 在 harness 增加 N 次 kernel 重复跑、丢弃前 K 次的 measure 模式；(b) 暴露 `MATMUL_SIZE` 配置或新增 128/256 尺寸 config；(c) 通过统一 `make cmp-matmul` / `make sweep-matmul SWEEP=size` 入口跑 compare 与 size sweep。
- **交付物**：`../archive/matmul-spm-lowering-closure.md` §P3.5 补 steady-state + 大尺寸数据；论文头数应从大尺寸里挑。
- **当前记录（2026-04-29）**：工具链已落地，默认 64 cold-cache 行为保持不变。已跑 steady preset compare：64×64 steady warm-cache PASS，SPM 29,999.0 cy/iter vs cache 19,485.8 (+54.0%)。已跑 size sweep：128×128 cold-cache PASS，SPM 249,322 vs cache 219,288 (+13.7%)。后续 256×256 cold-cache 与 128×128/256×256 steady-state sweep 已补完；当前大尺寸 cold-start 1024×1024×1024 / 32×32×32 PASS，SPM 288,976,339 cycles vs cache 386,049,495 cycles，SPM 快 25.1%。
- **验证**：steady-state cache mode 比 cold-cache 显著更快，说明 warm-cache 重复 launch 主要反映 cache residency；P3 headline 使用 cold-start 大尺寸数据。相关 sweep 不再是 pending。
- **范围限制**：harness/Makefile 改动；不动编译器或 simulator。
- **对应 task**：`steady-state-and-size-sweep`。

### Stage 3 — P3 back-end 压力修复（completed）
- **范围**：基于 Stage 1 证据，主攻 68.8% cycle gap 所在的 MMIO descriptor store 路径。优先级（与 P3.2 一致）：(1) 合并 / 压缩 MMIO descriptor stores 为更少的 64-bit 写或单条 descriptor-block 触发命令——目标 back-end 压力 11,341 cy 桶；(2) 跳过最后一轮顶部 `dma_wait`（前置：先按 buffer 映射验证冗余）——~960 cy；(3) 把 prefetch 提前到 body 更早位置——最多 ~2,400 cy（DMA wait 桶上限）。每次只改一项并重测。
- **交付物**：`transformGemmLoop`（或 `DmaOpsToLLVM`）的最小 patch + before/after stats 对比记录。
- **验证**：`make cmp-matmul` 通过；`numCycles` 相比 Stage 2.5 的 39,480 显著下降，目标 ≤ fair cache × 1.05（37,004 × 1.05 ≈ 38,854）。64×64 是 smoke/regression guard；headline 由 Stage 2.6 的实测大尺寸 cold-start 数据决定，steady-state 只作辅助参考。
- **范围限制**：不动 reduction 路径；不重构 matcher。
- **对应 task**：`p3-wait-semantics`、`p3-reduce-wait-cost`、`p3-prefetch-timing`。

### Stage 4 — 2-D reduction prefetch DRAM offset 修复（completed）
- **范围**：`transformReductionLoop` 在 leading dimension 非连续时 prefetch 地址错误（`phase3-compiler-backlog.md` §C.2）。
- **交付物**：代码修复 + 一个针对 2-D stride>1 reduction 的 lit 测试。
- **验证**：新 lit 测试通过；现有 `layer_norm` / `matmul` SPM 路径功能 PASS 不退化。
- **范围限制**：只改 `transformReductionLoop` 内部地址计算；不动 GEMM 路径，不泛化 matcher。
- **对应 task**：`reduction-2d-addr-bug`。

### Stage 4.5 — Reduction 双缓冲流水（completed 2026-04-30）
- **背景**：`transformGemmLoop` 已是双缓冲 + prologue prefetch（../archive/matmul-spm-lowering-closure.md §P3.1 实测 DMA 延迟 62.3% 被 overlap 隐藏）。但 `transformReductionLoop` 仍是 *单缓冲* prefetch：每轮 body 先 `vector.transfer_read` 当前 chunk → 再 `dma_enqueue_2d` 下一 chunk → `dma_wait`，CPU 必须等 DMA 完成才能进入下一轮。这意味着 layer_norm / softmax / 未来所有 reduction kernel 的 DMA 延迟全部串行暴露在关键路径，是 reduction 路径性能基线偏弱的结构性原因。
- **范围**：把 reduction 路径升级到与 GEMM 同等的双缓冲方案：(a) 分配两个 SPM staging buffer；(b) prologue 发首轮 prefetch + wait；(c) body 顶 wait 当前 buffer、读完之后发下一轮 prefetch（不在 body 末尾 wait）；(d) bail-out 路径清残留 enqueue（GEMM 已踩过的坑）；(e) 复用 GEMM 的 buffer-flip 计数 / index 计算结构，避免再造一套。
- **交付物**：`transformReductionLoop` patch + `layer_norm` 的 block-pointer mean/variance/final normalize path + 32x64 gem5 baseline。
- **验证**：`compiler/test/TritonCPU/convert-memory-to-spm.mlir` 通过，覆盖 reduction body-top wait、buffer select、alternate-buffer prefetch、multi-load shared-IV streams 与 non-leading-IV stride。`make verify-layer_norm` 通过：LLIR `addrspace(3)=38`、`fence iorw=78`，tier JSON `{"0":3,"1":3,"2":3}`。`make cmp-layer_norm` 两侧功能 PASS；32x64 ROI SPM 125,593 cycles vs cache 6,138 cycles，1,280 DMA transfers / 40,960 bytes，waitFraction 0.4002，avgWaitStall 65.45 vs avgLatency 66.23。小尺寸仍慢，说明下一步应减少 reduction-path MMIO/control 开销，而不是把该点当性能 headline。
- **范围限制**：只改 `transformReductionLoop`，不动 matcher，不动 GEMM 路径。
- **依赖**：Stage 4（2-D 地址 bug 必须先修，否则双缓冲化只会放大错误地址的影响），并建议在 `tier-sidecar-verify` 之后做（先确认 layer_norm 真正进入了 SPM lowering 路径）。
- **对应 task**：`reduction-single-buffer-pipeline`（completed）。

### Stage 5 — L2-warming microbenchmark（completed 2026-04-30）
- **范围**：实现 `dma_l2_warming` workload，带 cacheable / UC / 无 DMA 三个变体 + per-checkpoint stats 解析 + working-set sweep。
- **交付物**：`../evidence/l2_warming.md` 记录源码分析、实验设计、4K-32K sweep 表和 cycle 数据。
- **验证**：Phase B（DMA 后随机读）近 100% L2 命中；Phase D（冷随机读）近 100% miss；4K-32K 全尺寸约 2.8x cycle 加速。
- **范围限制**：只新增 workload 与配套脚本；不改编译器主路径。
- **对应 task**：`l2-warming-bench`。

### Stage 6 — 评测工具化（baseline completed，Phase 6 comparator pending）
- **已完成**：`make verify` / `make verify-<kernel>` 落地（`run_experiment.py --mode verify`）；`make run-<kernel>` / `make cmp-<kernel>` 已覆盖统一 run/compare target；`compare_stats.py` 提取 21 symmetric + 15 SPM-only 指标到 `.txt`，并支持机器可读 CSV（`--csv` / `--spm-only-csv`）。
- **剩余**：Phase 6 对比工具。
- **验证**：matmul 与 layer_norm 通过 `make verify-<kernel>`；vector_add 预期 FAIL（SPM pass 未命中，见 `three-tier-placement.md` §4.1）。
- **范围限制**：不改编译器；不修改 simulator 接口。
- **对应 task**：`eval-tooling`.

### Stage 7 — Compiler robustness backlog（partially completed）
- **范围**：robust GEMM matcher（已完成 A/B lhs/rhs 识别、`IRMapping` cloned-read lookup 和 >2 loads 容忍）；reduction matcher 已泛化到多 load 共享 loop IV；`DmaOpsToLLVM` MMIO base pass option 与 `useXspmInsn` 开关已完成。剩余 bail-out cleanup/verification。
- **交付物**：已提交 GEMM A/B matcher、cloned-read lookup、extra-load coverage、reduction multi-load matcher、DMA lowering options 和对应 lit/pytest。剩余分子项继续保持最小 patch + 对应 lit 测试。
- **验证**：已通过 `dma-ops-to-llvm.mlir`、`convert-memory-to-spm.mlir`、`python/test/unit/cpu/test_dma.py`、`make verify-matmul verify-layer_norm`；multi-load matcher 另由 `make cmp-layer_norm` 做 gem5 功能覆盖。
- **范围限制**：不动 placement pass、不动 SPM lowering 主体。
- **对应 task**：`compiler-robustness-backlog`。

### Stage 8 — 文档刷新（completed）
- **范围**：`phase3.md` 与 `phase3-compiler-backlog.md` 中已被前面阶段解决但仍标记为 NOT VERIFIED / NOT IMPLEMENTED 的条目同步消除；`three-tier-placement.md` 状态对齐 Stage 2 的实际生成物。
- **交付物**：docs diff，无代码改动。
- **验证**：交叉引用一致；不再有"文档与代码状态冲突"的句子。
- **对应 task**：`doc-refresh`.

### 不在本计划内（保留为 backlog）
- `tier1-backlog`：Tier 1 resident SPM 完整实现，待 Stage 5 数据闭环后再排期。
- `reuse-rules-backlog`：仅在新 workload 触发时按需扩展。

## 暂不抢做但不能丢
- `compiler-roadmap.md` Phase 4 / Phase 5 的 attention、SPM output writeback、多 kernel SPM lifetime、transformer block 是完整论文故事的一部分，但当前不应抢在 matmul P3 和 Tier 2 证据链前面。
- `compiler-roadmap.md` Phase 6 的 workload coverage、performance breakdown、area-equivalent comparison、sensitivity analysis 是后续论文评测闭环；近期工具化应尽量让这些实验能复用同一套 run target 和 stats parser。
- `phase3.md` 现在是同步后的 Phase 3 状态页；如果它与本计划冲突，应优先重新核对 `../archive/matmul-spm-lowering-closure.md`、`three-tier-placement.md`、`../evidence/l2_warming.md` 和实际生成物。

## 关键参考文件
- [`phase3.md`](phase3.md): 当前 Phase 3 状态页。
- [`../archive/matmul-spm-lowering-closure.md`](../archive/matmul-spm-lowering-closure.md): matmul P3 closure and measured result log.
- [`three-tier-placement.md`](three-tier-placement.md): 三层 placement 的设计与 MVP 里程碑。
- [`../evidence/l2_warming.md`](../evidence/l2_warming.md): Tier 2 论文 claim 的实验结果。
- [`phase3-compiler-backlog.md`](phase3-compiler-backlog.md): Phase 3 审计与剩余 compiler robustness 事项。
