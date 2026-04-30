> **Status: SUPERSEDED — 2026-04-27**
> 原 Phase 2 收尾笔记，记录三处对齐项与"是否 cacheable"的最初想法。
> - MMIO base pass option / `useXspmInsn` 开关 / `addIllegalOp<DmaEnqueue2DOp, DmaWaitOp>()` → 已迁移到 [`../plans/phase3-compiler-backlog.md`](../plans/phase3-compiler-backlog.md) §Phase-3 plumbing；其中 `addIllegalOp` 已落地，另两项仍为 backlog。
> - "让编译器决定数据放 cacheable 还是 uncacheable" → 演化为 [`../plans/three-tier-placement.md`](../plans/three-tier-placement.md) 的三层 placement 设计。
>
> 保留作为决策起点的快照，不再是行动文档。

---

写法没有违背 Phase 2，寄存器下发顺序、fence 位置、MMIO 偏移都照着文档来了。但相对于 Phase 2 的完整设计，还差三点小的对齐：

把 MMIO 基址改成 pass option
留一个 useXspmInsn 之类的开关，接口对 Phase 4d 开放
补上 addIllegalOp<DmaEnqueue2DOp, DmaWaitOp>()
另外要单独验证一下 SPM 地址空间在上下游（TypeConverter + ConvertMemoryToSPM）是不是真的一致用的 addrspace 3，这个 pass 自身看不出来。

让编译器“决定数据放 cacheable 还是 uncacheable”