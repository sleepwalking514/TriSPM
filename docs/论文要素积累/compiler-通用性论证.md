# Compiler 通用性论证：Pattern Matching 不等于 Library Approach

## 核心判断

这个 work 仍然有价值。Reviewer 质疑的重点不是“用了 pattern
matching”，而是 pass 是否依赖 kernel-specific 模板。如果 pass 匹配的是
`matmul`、`conv`、`softmax` 这些具体 kernel 的名字或专门形状，那么它会更像
library/template approach；但如果 pass 匹配的是 IR 层面的通用 data-movement
idiom，例如 tiled affine access、tile footprint、memory-space promotion、
DMA descriptor generation 和 dependence-aware synchronization，那么它仍然是
compiler approach。

一句话口径：

> Pattern matching is not the problem; kernel-specific pattern matching is the
> problem. Our pass should be framed as a generic lowering from analyzable tiled
> memory footprints to DMA transfers, not as a collection of per-kernel DMA
> templates.

## Reviewer 可能的质疑

Reviewer 可能会问：

> Your pass currently relies on pattern matching, such as recognizing tiled load
> patterns and inserting DMA. If a new kernel requires a new pattern, is this
> really a compiler technique, or just a library approach encoded inside the
> compiler?

这个问题需要正面回答。不能只说“编译器也有 pattern matching”，而要进一步证明：

- pass 没有识别 kernel 名字；
- pass 没有 hard-code 某个 operator 的 loop nest；
- pass 的输入 contract 是一个通用 IR 形态；
- 新 kernel 只要 canonicalize 到这个 IR 形态，就不需要修改 DMA insertion pass；
- 不支持的 kernel 会安全 fallback，而不是要求用户手写新模板。

## 建议的 Compiler Contract

不要把 pass 描述成“识别 tiled load 然后插 DMA”。更稳妥的说法是：

> The pass targets tiled affine kernels whose global-memory accesses can be
> represented as statically analyzable tile footprints. Given a canonical tiled
> loop nest, affine memory access maps, memory-space annotations, and dependence
> information, the pass derives DMA descriptors, inserts asynchronous copy
> operations, and places waits/fences before the corresponding tile consumers.

中文解释：

这个 pass 的输入不是“某个 kernel”，而是“已经被 tiling/canonicalization 整理过的
affine memory-access IR”。如果 `matmul`、`conv2d`、`stencil`、`transpose`
都能降到这种形式，DMA insertion pass 可以复用同一套逻辑。

## 区分 Kernel Pattern 和 Data-Movement Pattern

需要在论文或 rebuttal 里明确区分两类 pattern：

| 类型 | 例子 | 性质 |
| --- | --- | --- |
| Kernel-specific pattern | `if op == matmul`, `if loop shape matches a hand-written conv template` | 更像 library/template approach |
| IR/data-movement pattern | tiled affine footprint, source/destination memory spaces, DMA-capable strides, producer-consumer dependence | compiler approach |

建议强调：

> Our rewrite rules do not encode operator-specific schedules. They encode a
> data-movement lowering rule parameterized by affine maps, tile sizes, memory
> spaces, DMA capabilities, and dependence constraints.

## Pass 应该被描述成什么

推荐把 pass 拆成以下 compiler steps：

1. Identify candidate memory accesses in a tiled loop nest.
2. Infer the tile footprint from affine maps or structured subviews.
3. Check DMA legality, including contiguity/stride support, alignment, bounds,
   memory-space compatibility, and dependence constraints.
4. Materialize DMA descriptors from the inferred footprint.
5. Insert asynchronous DMA operations.
6. Place waits/fences before tile consumers.
7. Fall back to the original cache/load path when legality or profitability
   checks fail.

这样叙述后，pattern matching 只是实现 rewrite 的机制，不是 contribution 的全部。
真正的 contribution 是从 IR 语义到 SPM/DMA data movement 的自动 lowering。

## 可直接使用的 Rebuttal 段落

> Although our implementation uses rewrite patterns, these patterns are not
> kernel-specific templates. They match a semantic data-movement idiom in the IR:
> a statically analyzable tile footprint moved between memory spaces and consumed
> within a tiled loop body. The same DMA insertion rule is parameterized by
> affine access maps, tile sizes, memory-space annotations, DMA capability, and
> dependence information. Therefore, a new kernel does not require modifying the
> DMA insertion pass as long as it is lowered to the same canonical tiled affine
> IR form. Kernels outside this analyzable class safely fall back to the original
> cache-based lowering.

更强一点的版本：

> The distinction from a library approach is that the compiler does not select
> from hand-written operator implementations. Instead, it derives DMA transfers
> from the program representation itself. Operator-specific front-end lowering
> may be needed to expose canonical tiled affine accesses, but the subsequent
> SPM promotion and DMA insertion logic is shared across kernels.

## 论文中可以这样写

方法章节可以写成：

> We formulate SPM promotion as a lowering problem over canonical tiled affine
> IR. For each candidate access, the compiler computes the accessed tile
> footprint, checks whether the footprint is representable by the target DMA
> engine, and replaces repeated scalar/vector memory accesses with explicit
> asynchronous transfers into SPM. Synchronization is inserted according to the
> producer-consumer dependence between the DMA operation and the compute region.
> This design separates kernel-specific canonicalization from kernel-independent
> DMA lowering.

贡献点可以写成：

> A compiler pass that automatically derives SPM/DMA data movement from tiled
> memory-access IR, avoiding hand-written per-kernel DMA libraries for regular
> affine kernels.

## 实验上应该补的证据

最好补一个小表，证明 same pass 可以处理多个 kernel，且没有修改 pass：

| Kernel | Access shape | Pass change needed? | Notes |
| --- | --- | --- | --- |
| Matmul | 2D affine tiles | No | Dense tile loads |
| Conv2D | Sliding-window affine tiles | No, if canonicalized | May need padding/boundary handling |
| Stencil | Tile plus halo | No, if footprint inference supports halo | Tests footprint generality |
| Transpose | Strided tile | No, if DMA supports stride | Tests non-unit-stride descriptors |
| Row reduction / Softmax | Row or row-block resident tile | No for shared residency logic | Tests producer-consumer reuse |

即使暂时只跑了两个或三个 kernel，也要报告：

> All evaluated kernels are handled by the same DMA insertion pass without
> adding kernel-specific rewrite rules.

如果某个 kernel 需要额外处理，要把它说成 canonicalization 或 legality extension，而不是
“新增 kernel 模板”：

> Supporting this kernel required adding a canonicalization that exposes its
> access region as an affine tile; the DMA insertion pass itself was unchanged.

## 代码层面的自查清单

为了让论证站得住，需要检查实现里有没有这些风险：

- 是否出现 `matmul`、`conv`、`softmax` 等 kernel 名字作为 pass 分支条件；
- 是否 hard-code 某个 loop depth、某个 operand role 或某个固定 tile shape；
- 是否只能识别某一种 load/store 顺序；
- 是否把 DMA descriptor 生成和某个 operator 的 schedule 绑死；
- 是否缺少 legality failure 的 fallback 路径。

建议把 kernel-specific 信息压成参数或接口：

- affine map / indexing map；
- tile sizes；
- source and destination memory spaces；
- DMA descriptor capability，例如 contiguous、strided、2D tile；
- boundary/mask policy；
- dependence and synchronization points；
- profitability/admission policy。

## 诚实的 Scope

不要声称支持所有 kernel。更可信的 scope 是：

> The current implementation targets regular tiled kernels with statically
> analyzable affine or structured memory accesses. Irregular pointer chasing,
> data-dependent gather/scatter, and highly dynamic access regions are outside
> the current scope and fall back to the baseline lowering.

中文口径：

当前 work 面向 dense tensor computation 中最常见、也最值得做 SPM/DMA 优化的规则
affine/tiled kernel。这个限制是合理的 compiler scope，不是失败。关键是 fallback
要安全，且 supported class 要清楚。

## 最终答辩口径

可以把回答压缩成：

> 我们确实使用 rewrite pattern，但它不是 per-kernel template。这个 pass 匹配的是
> tiled affine IR 中的 data-movement idiom：一个可静态分析的 tile footprint 从全局
> memory 被搬到 SPM/local memory，并在 tiled compute region 中被消费。新 kernel
> 只要通过前端 lowering/canonicalization 暴露出同样的 IR 形态，就不需要修改 DMA
> insertion pass。不满足条件的 irregular kernel 会 fallback 到原始 cache path。因此，
> 这不是把 library 模板塞进 compiler，而是一个带有明确适用范围和 legality check 的
> compiler lowering。

## 对 Work 价值的定位

这个 work 的价值不在于“为几个 kernel 手写了 DMA 版本”，而在于：

- 建立了从 tiled memory-access IR 到 DMA descriptor 的自动 lowering；
- 把 SPM/DMA 优化从程序员手写库代码转移到 compiler pass；
- 明确了 legality、synchronization 和 fallback 边界；
- 对 affine dense kernels 给出了可复用的 SPM promotion 机制；
- 为后续更多 kernel 的支持提供了 canonicalization + shared DMA lowering 的扩展路径。

