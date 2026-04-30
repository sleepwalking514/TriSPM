# Three-Tier Data Placement — Plan & Discussion (P0+ #8)

> 对应 `phase3-compiler-backlog.md` §B.3c / §E P0+ #8 / §E P1 #9。
> Phase 3 论文核心论点 "SPM never worse than cache" 的实现支撑。

---

## 0. 物理地址空间清单（前置概念）

| 空间 | 位置 | Cache 行为 | 分配 API | 默认范围 |
|---|---|---|---|---|
| SPM | 片上 SRAM | 旁路 cache，CPU 通过 spm_port 直访 | `spm_malloc` | `0x40000000`，256 KiB |
| Cacheable DRAM | DRAM | 走 L1/L2 cache | `malloc` | `0x80000000+`（系统堆） |
| Uncacheable DMA buf | DRAM | gem5 routing 标记为 uncacheable | `dma_buf_malloc` | `0x30000000`，1 MiB |

`dma_buf_malloc` **不是** "走 SPM 必经的 staging"，而是 DRAM 的一段特殊区域。三个空间互不重叠。

---

## 1. 三层 Tier 的数据流

### Tier 1 — SPM-resident
- **分配位置**：input tensor 在 SPM (`spm_malloc`)；output tensor 仍在 DRAM (`malloc`)，kernel 内 SPM accumulator → 收尾 DMA 写回。
- **harness 行为**：启动前 DMA 从 DRAM 初值预拷到 SPM。
- **kernel 行为**：`ConvertMemoryToSPM` 不为 Tier-1 张量生成 staging — 数据已在 SPM。
- **何时赢**：tensor 装得下 SPM 且有 scalar reuse（如 softmax 的 `max`/`sum`、attention 的 `lse`）。
- **MVP 状态**：**不实现**。需要 (a) addrspace(3) 注入 func arg；(b) harness pre-DMA；(c) 与 ConvertMemoryToSPM staging buffer 共享 `SPMSpaceManager` 解决冲突。框架预留 tag。

### Tier 2 — Cacheable DRAM + DMA tiling
- **分配位置**：`malloc`（普通 cacheable DRAM）。
- **kernel 行为**：`ConvertMemoryToSPM` 把 tile DMA 到 SPM staging。
- **L2-warming 副作用**：DMA 读经过 L2XBar，命中前会在 L2 allocate cache line。后续 scalar 访问命中 L2。
- **何时赢 cache baseline**：当 tensor 体积 > SPM 但有 scalar reuse 时；纯 cache baseline 因为 vector / scalar 抢 cache line 反而更差。
- **MVP 状态**：**作为兜底实现**。当分析无法判定时默认走这条路径。

### Tier 3 — Uncacheable DMA buffer
- **分配位置**：`dma_buf_malloc`（uncacheable DRAM）。
- **kernel 行为**：与 Tier 2 一样走 ConvertMemoryToSPM staging。
- **优势**：DMA 读不污染 cache，纯向量 workload 拿到完整 SPM 收益。
- **何时赢**：tensor 无 scalar reuse（纯向量消费）。
- **MVP 状态**：**实现**。`matmul` 命中此 tier（args 0,1 → Tier 3）；
  `layer_norm` 的输入 `x` 也在 mean/variance block-pointer reduction path
  中命中 Tier 3。`vector_add` 的 tier JSON 为空——见 §4.1 覆盖审计。

### 关于 write-back
- Tier 1 input：用完即弃，无写回。
- Tier 1 output：DRAM 分配 + SPM accumulate + 收尾 DMA 写回。
- Tier 2 / 3 output：通过 `ConvertMemoryToSPM` 的 SPM→DRAM 写回（当前已有路径，写回到 tensor 原分配地址）。

---

## 2. 用户确认的设计决策（2026-04-26）

| ID | 决策 | 备注 |
|---|---|---|
| D1 | 新增独立 pass `SPMTensorPlacement`，跑在 `ConvertMemoryToSPM` 之前 | 关注点分离 |
| D2 | JSON sidecar `<kernel>_tiers.json` → driver.py 生成 `<kernel>_alloc(idx, size)` 派发函数 | harness 单点改 malloc 即可 |
| D3 | MVP 只做 Tier 2/3 plumbing；Tier 1 框架预留 tag 不实现 | Tier 1 完整实现 + 配套 workload 进 §6 |
| D4 | `has_scalar_reuse` 保守语义：scalar load **或** `vector<1xT>` transfer_read 算 reuse | 扩展规则进 §6 |
| D5 | 进入 placement 的张量必中一个 tier，无 conservative 开关。`no scalar reuse → Tier 3` 直接生效 | 仅 matmul 命中（见 §4.1） |
| D6 | 三个 workload harness 全部改造 | 否则 alloc 派发链路无法验证 |
| D7 | 选择 **b1**：MVP 就引入统一 `SPMSpaceManager`，`ConvertMemoryToSPM` staging 也从它分配 | Tier 1 之后接入时复用同一套 SPM layout，不再重构 staging |

> D5 含义补充：tier 分析只对"被 ConvertMemoryToSPM 视为候选的张量"打标。其他 memref（小常量、未被 tile 的张量）走默认 `malloc`，与 tier 系统正交，无冲突。
>
> D7 含义补充：统一 allocator 是**编译期 SPM 地址空间/offset 管理器**，不是把所有 tier 都改成 runtime `spm_malloc`。Tier 1 tensor 才是 resident SPM allocation；Tier 2/3 tensor 仍分别由 `malloc` / `dma_buf_malloc` 分配在 DRAM，只是它们进入 kernel 时使用的 temporary staging buffer 也由 `SPMSpaceManager` 分配 SPM offset。

### Q1 / Q2 结论（b1 后）

- **Q1 临时搬运 vs 显式分配仍不完全一样**：两者都占 SPM 地址、都由同一个 `SPMSpaceManager` 防冲突；区别仍是生命周期。Tier 1 是 tensor 常驻（跨 kernel call，直到显式释放），Tier 2/3 staging 是 kernel/loop 临时 buffer。
- **Q2 "装得下 + 无 reuse" 仍默认 Tier 3**：统一 allocator 只解决 SPM 地址冲突，不改变性能判断。无 scalar reuse 时，Tier 1 常驻不会让 vector load 更快，却会付出 pre-DMA 整 tensor 成本并占住 SPM；Tier 3 用 staging 拿到同样的 SPM vector load，通常更划算。

---

## 3. 实现计划（MVP — Tier 2/3）

### 3.1 编译器侧：统一 SPM 空间管理

**新 helper** `SPMSpaceManager`（建议放在 `compiler/third_party/cpu/lib/TritonCPUTransforms/` 或 include 对应 utility 目录）
- 管理 SPM logical layout：`base`, `size`, `top`，按 alignment 返回 offset/address。
- MVP 只需要 bump-pointer/high-water 行为；free list / lifetime reuse 先只保留接口，不实现复杂复用。
- `ConvertMemoryToSPM` 当前 hardcode 的 staging offset（`+0`, `+tileA`, `+2*tileA`, ...）改成 `alloc()` 返回值。
- MVP 行为应与现状等价：因为 Tier 1 还不占空间，staging 仍从 SPM base 开始连续分配。
- 后续 Tier 1 接入时：`SPMTensorPlacement` 先为 resident tensor alloc long-lifetime range，`ConvertMemoryToSPM` 再为 staging alloc short-lifetime range，天然避免冲突。

### 3.2 编译器侧：tier placement

**新文件** `compiler/third_party/cpu/lib/TritonCPUTransforms/SPMTensorPlacement.cpp`
- 注册为 pass `convert-memory-to-spm-placement` 或 `spm-tensor-placement`。
- 输入：TTCIR (post `convert_memory_ops`)。
- 流程：
  1. 遍历 `func.func` 的每个 memref-类型 arg。
  2. 仅对 "被某个 scf.for 内的 tiled `vector.transfer_read` 消费" 的 arg 做分析；其他保持无 tag。
  3. `has_scalar_reuse(arg)` = ∃ user 是 `memref.load` 或返回 `vector<1xT>` 的 `vector.transfer_read`。
  4. tier 决策表：
     | 条件 | tier |
     |---|---|
     | scalar reuse + 装得下 SPM | 1（占位，MVP 回落 2） |
     | scalar reuse + 装不下 | 2 |
     | no scalar reuse | 3 |
  5. 把 tier 写入 func arg attribute `tt_cpu.spm_tier = i32`（仅 IR 内传递，不依赖 launcher 读取）。
- **不修改** IR 语义；纯 annotation pass，便于后续 `ConvertMemoryToSPM` 读取做不同生成。

**Pass 注册**
- `compiler/include/cpu/include/TritonCPUTransforms/Passes.td` 增 `ConvertMemoryToSPMPlacement`。
- `compiler/third_party/cpu/triton_cpu.cc` 暴露 `add_convert_memory_to_spm_placement(pm)`。
- `compiler.py::make_tttcir` 在 `add_convert_memory_to_spm` *之前* 调用。

**Sidecar 写出**
- 在 placement pass 末尾把 `{arg_index → tier}` 写入 `${KERNEL_AUX_FILE_DIR}/<kernel>_tiers.json`。
- 复用现有 `KERNEL_AUX_FILE_DIR` 机制（与 launcher.c/.h 同目录）。

### 3.3 Driver 侧（launcher 生成）

**`compiler/third_party/cpu/backend/driver.py::make_aot_launcher`**
- 读取同目录下 `<kernel>_tiers.json`（不存在时全部按 Tier 2 处理 → 与现状一致）。
- 在 `<kernel>_launcher.h` 增声明：
  ```c
  void *<kernel>_alloc(int arg_index, size_t nbytes);
  void  <kernel>_free_all(void);
  ```
- 在 `<kernel>_launcher.c` 生成对应实现（switch 派发 → `spm_malloc` / `malloc` / `dma_buf_malloc`）。

### 3.4 Harness 侧

三个 workload (`workloads/{matmul,layer_norm,vector_add}/harness.c`) 改造：
- `#include "libspm.h"` → 通过 KERNEL_AUX_FILE_DIR 或 env.sh 加 include path。
- `malloc(n)` → `<kernel>_alloc(arg_idx, n)`。
- 程序结束 `<kernel>_free_all()` 替代逐个 free（线性 allocator 简化）。
- 注意：用 SPM 的 input tensor 在 Tier 3 模式下用 `dma_buf_malloc`，初值仍要 `memcpy` 进去（因为该区域 uncacheable，但仍是 DRAM，CPU 可以普通 store；或考虑用 spm_dma_copy 搬运。MVP 用普通 store）。

### 3.5 stats / 验证

- 验证手段（按用户要求）：**只看 IR**，不跑 gem5。
  - `<kernel>.llir` 中：Tier 3 张量地址应在 `0x30000000+` 区域；Tier 2 在 heap 区。
  - `<kernel>_tiers.json` 内容正确。
  - `<kernel>_launcher.c` 含 `<kernel>_alloc` 派发逻辑。
  - harness binary `objdump` 能找到对 `dma_buf_malloc` 的调用。

---

## 4. 实现里程碑

- [🆗] M1: 新增 `SPMSpaceManager` helper（bump-pointer + alignment + 预留 lifetime/free 接口）
- [🆗] M2: `ConvertMemoryToSPM` staging offset 改为通过 `SPMSpaceManager::alloc()` 生成，验证 IR 与现状等价
- [🆗] M3: `SPMTensorPlacement` pass 骨架 + tablegen 注册 + pybind 暴露
- [🆗] M4: `has_scalar_reuse` 分析 + tier 决策 + arg attribute 写入
- [🆗] M5: JSON sidecar 写出（pass 内 file I/O 或 helper）
- [🆗] M6: `make_aot_launcher` 读 sidecar，生成 `<kernel>_alloc/_free_all`
- [🆗] M7: 三个 workload harness 改造
- [🆗] M8: IR 验证（每个 workload 的 `.llir` + `_tiers.json` + `_launcher.c`）

### 4.1 覆盖审计（2026-04-29，2026-04-30 更新）

`make verify` 重新 build 三个 workload 后的实际结果：

| workload | `_tiers.json` | LLIR `addrspace(3)` | LLIR `fence iorw` | SPM pass 生效？ |
|---|---|---|---|---|
| matmul | `{"0":3,"1":3}` | 583 | 80 | ✅ |
| layer_norm | `{"0":3}` | 32 | 68 | ✅ mean/variance reduction only |
| vector_add | `{}` | 0 | 0 | ❌ |

**根因分析**：

- **vector_add**：kernel 使用 pointer arithmetic（`x_ptr + offs`），无 `tl.make_block_ptr`。`ConvertMemoryOps` 将其降为 gather-style load，不产生 `vector.transfer_read %memref[%idx]` 的规范形式。此外 vector_add 是单 block 无循环 kernel，没有 `scf.for` 可供 `findTiledLoads` 匹配。**这是设计上的预期行为**——elementwise kernel 无 tile reuse，SPM tiling 无收益。
- **layer_norm**：mean / variance 两个 single-load reduction pass 已改为
  `tl.make_block_ptr`，并把 `N` 固定为 constexpr，使
  `ConvertMemoryToSPM` 的静态 loop guard 可以证明边界。该路径现在命中
  SPM：`make verify-layer_norm` 通过，LLIR 有 32 个 `addrspace(3)` 和
  68 个 `fence iorw`。最终 normalize pass 仍有 3 个 load（x, gamma,
  beta），需要 reduction matcher 泛化到多 load 才能覆盖。

**结论**：

- `matmul` 是当前成熟的 SPM performance workload，tier sidecar 工作正常。
- `layer_norm` 现在是 reduction-path SPM coverage workload；32x64
  `make cmp-layer_norm` 功能 PASS，但 SPM 656,898 cycles vs cache
  121,284 cycles，说明该小尺寸点主要暴露 MMIO/DMA 控制开销。
- `vector_add` 不需要 SPM（无 tile reuse），空 JSON 是正确行为。文档中"三个 workload 全部命中 Tier 3"的说法不准确，已修正。
- `layer_norm` 剩余前置修复是泛化 reduction matcher 接受多 load。这属于
  `compiler-robustness-backlog` 范畴。

---

## 5. 风险与开放问题

1. **MLIR pass 写文件副作用**：MLIR pass 一般不该有 I/O 副作用。可选：
   - (a) pass 写 attribute，driver.py 在 `make_tttcir` 后从 `mod` 直接读取
   - (b) pass 写文件
   M5 已选 (b) 的 MVP 做法：placement pass 直接写 `<kernel>_tiers.json`。后续如需减少 pass I/O 副作用，可再切回 (a)。
2. **Triton AOT 编译路径中 `<kernel>_tiers.json` 写出时机**：必须早于 launcher.c 生成。如果用 (a)，driver.py 在 launcher 生成时直接读 IR attribute，时序天然正确。
3. **arg_index 的稳定性**：MLIR 里 func arg 顺序与 launcher.c 里 `<kernel>_alloc` 派发用的索引必须一致。需检查 `make_aot_launcher` 里的 `signature_flat` 顺序与 IR 一致。
4. **uncacheable DRAM 上的初始化**：harness 用普通 store 写 `dma_buf_malloc` 返回的 buffer，不经 cache，每次 store 直入 DRAM — 性能差但功能正确。如果初始化耗时影响测量，可以考虑 m5_reset_stats 之后才开始计时（已有 m5op 支持）。

---

## 6. Backlog（暂不在 MVP 内）

### 6.1 Tier 1 完整实现 (D3 选项 C)
- func arg `addrspace(3)` 注入到 LLVM lowering。
- harness pre-launch DMA：从 `malloc` 区把初值搬到 `spm_malloc` 区。
- `SPMSpaceManager` long-lifetime allocation：Tier 1 resident tensor 先占 SPM range，`ConvertMemoryToSPM` staging 后占临时 range。
- SPM layout manifest：把 Tier 1 resident allocation 的 offset/size 暴露给 harness 和 compiler，保证 pre-DMA 目标地址与 IR 中的 SPM 地址一致。
- 配套 workload：softmax / 小型 attention，要求有 scalar reuse + 整 tensor 能装下 SPM。

### 6.2 `has_scalar_reuse` 扩展规则 (D4)
- `vector.extract` 抽出 scalar 后再 broadcast 算 scalar reuse。
- scf.for 外的 load 也算（当前只看循环内 user）。
- `tt.reduce` 内的中间 scalar 算 reuse。
- 触发条件：当未来引入需要更宽语义的 workload 时再加，避免过早泛化。

### 6.3 验证补完
- `make verify-spm-fires` 引入 tier 检查（phase3-compiler-backlog.md §E P2 #14 的延伸）— 已由 `make verify` / `make verify-<kernel>` 覆盖。
- L2-warming 实验数据点：tier 2 vs cache baseline，要求新写 long-vector + scalar-tail workload — 已由 `../evidence/l2_warming.md` 的 `dma_l2_warming` microbenchmark 覆盖。
