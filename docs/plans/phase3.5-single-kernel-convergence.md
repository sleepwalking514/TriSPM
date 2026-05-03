# Phase 3.5 Single-Kernel SPM Convergence Plan

> Status: active plan, 2026-05-03.
> This phase exists because the LayerNorm re-audit showed that reduction SPM
> was not fundamentally settled.  Phase 4 graph/fusion work should wait until
> the single-kernel reduction story has a measured, reuse-aware baseline.

## Goal

Phase 3.5 closes the single-kernel SPM performance policy before graph-level
Phase 4/5 work.  The target is not "make every kernel use SPM"; it is:

- `matmul` remains the mature SPM path.
- Low-reuse elementwise kernels stay cache path unless fused.
- Reduction kernels (`layer_norm`, `softmax`) get a shared row/block-resident
  promotion framework with explicit admission, lifetime, buffer rotation, and
  profitability evidence.

The immediate success bar is stricter than "correct and near parity":

| Kernel | Default target | Opt-in target before default enablement |
|---|---|---|
| `layer_norm` | cache path until measured win | row-resident SPM beats or matches cache on 32x64 and 512x1024 |
| `softmax` | cache path until measured win | row/block-resident SPM beats or matches cache on row-wise smoke and one larger row case |
| `matmul` | SPM path | no regression under existing `make verify-matmul` and headline compare |
| `activation`, `residual_add`, `vector_add` | cache path | only revisit inside fusion or producer-consumer schedules |

## Lessons To Borrow From Triton GPU

This is the useful shared-memory model to copy, not a literal GPU feature
clone:

| Idea | GPU analogue | Current TriSPM status | Phase 3.5 action |
|---|---|---|---|
| Admission | Only some loads become shared-memory/async-copy operands; normal `tl.load` may stay global/cache | D1/D3 evidence exists, but reduction thresholds were based on flawed row-DMA data | Refit reduction admission around reuse and measured SPM store/read overhead |
| Lifetime | `ttg.local_alloc` / `local_load` / `local_store` are scoped to a block/kernel region | `SPMSpaceManager` tracks Function/Loop/Temporary, but `free()` is still a stub and reductions are ad hoc | Introduce a reduction residency plan with explicit row/block lifetime and report fields |
| Replacement | GPU pipelines use explicit double/ring buffers and stage indices, not hardware LRU | GEMM and old streaming reduction have local double-buffer logic; row-resident reduction has one resident row; no generic reduction buffer plan | Add explicit buffer role/rotation model: resident row, optional ping-pong chunk, and optional output/temp slots |
| Cross-kernel | Shared memory does not persist across normal Triton launches; cross-kernel data goes through global memory/L2 unless fused | Three-tier graph placement already plans cacheable Tier 2 backbone plus optional fusion/Tier 1 later | Keep Phase 4 as cacheable-boundary first, fusion second; do not depend on cross-kernel SPM persistence for Phase 3.5 |

Important correction: Triton GPU LayerNorm forward is not proof that every
LayerNorm row must be staged in shared memory.  The official tutorial expresses
the row as `tl.load` + vector reductions.  The lesson is not "always use shared
memory"; it is "make scratchpad residency explicit only when the schedule proves
reuse."

## Current Pass Map

| Component | File | Current role | Kernels affected |
|---|---|---|---|
| `SPMTensorPlacement` | `compiler/third_party/cpu/lib/TritonCPUTransforms/SPMTensorPlacement.cpp` | Kernel-local Tier 1/2/3 sidecar and arg annotation | `matmul` default Tier 3 inputs; old opt-in streaming reductions when enabled |
| `SPMSpaceManager` | `compiler/third_party/cpu/include/TritonCPUTransforms/SPMSpaceManager.h` | Compile-time SPM address allocator | All SPM lowering paths |
| `ConvertMemoryToSPM` | `compiler/third_party/cpu/lib/TritonCPUTransforms/ConvertMemoryToSPM.cpp` | Main SPM promotion/lowering pass | `matmul`, opt-in `layer_norm`, future `softmax` |
| `transformFusedMicroGemmLoop` | same | Mature GEMM path: B-window, A-micro, accumulator residency | `matmul` |
| `transformGemmLoop` | same | Older double-buffer GEMM fallback | `matmul` fallback |
| `transformReductionLoop` | same | Old streaming reduction SPM path with chunk DMA | opt-in `layer_norm`; candidate fallback only |
| row-resident reduction matcher/lowering | same | LayerNorm-specific fill-on-first-pass path | opt-in `layer_norm`; should become generalized reduction residency |
| `SplitLargeContract` | `compiler/third_party/cpu/lib/TritonCPUTransforms/SplitLargeContract.cpp` | Register-pressure reduction after SPM GEMM lowering | `matmul` |
| `DmaOpsToLLVM` | `compiler/third_party/cpu/lib/TritonCPUToLLVM/DmaOpsToLLVM.cpp` | DMA/fence/MMIO lowering | DMA-based SPM paths |
| workload manifests | `workloads/kernels/*/experiment.toml` | Expected default policy and gem5 smoke shape | all current workloads |

## Work Items

### P0. Stabilize The Measurement Baseline

Files:

- `workloads/scripts/run_experiment.py`
- `workloads/scripts/compare_stats.py`
- `workloads/kernels/layer_norm/experiment.toml`
- `workloads/kernels/softmax/experiment.toml`

Tasks:

- Add named Phase 3.5 presets for `layer_norm` and `softmax`: small row,
  large row, and flushed ROI compare.
- Keep policy verification aware of SPM-without-DMA (`addrspace(3)` present,
  `fence iorw` absent).
- Add CSV fields for reduction-specific counters: SPM CPU reads/writes, DMA
  transfers, DMA wait stall cycles, bank conflicts, `simInsts`, and line count
  if available.

Exit criteria:

- Reproduce the current fill-on-first-pass LayerNorm numbers.
- Preserve `make verify-matmul`, `make verify-layer_norm`,
  `make verify-softmax`.

### P1. Generalize Row/Block-Resident Reduction Planning

Files:

- `compiler/third_party/cpu/lib/TritonCPUTransforms/ConvertMemoryToSPM.cpp`
- optional new helper:
  `compiler/third_party/cpu/include/TritonCPUTransforms/SPMReductionPlan.h`
  and `.cpp`
- `compiler/test/TritonCPU/convert-memory-to-spm-row-resident.mlir`
- new/extended lit for softmax row/block residency

Tasks:

- Replace the LayerNorm-only row-resident record with a generic
  `ReductionResidencyPlan`.
- Represent:
  - source arg / row or block shape,
  - producer pass (`fill_on_first_pass`, `producer_store`, or `dma_prefetch`),
  - consumer passes,
  - required SPM slots,
  - buffer role (`resident_row`, `ping_pong_chunk`, `temp_vector`,
    `output_tile`),
  - rotation policy (`none`, `double_buffer`, `ring_N`).
- Keep LayerNorm as the first concrete lowering.
- Add Softmax matching after LayerNorm: one row/block load, max reduction,
  exp/sum, normalize/store.

Exit criteria:

- LayerNorm still emits fill-on-first-pass SPM with no DMA fences.
- Softmax has an opt-in row/block-resident SPM path with promotion evidence.
- Unsupported reductions leave clean rejection records and cache-path IR.

### P2. Reduce Fill-On-First-Pass Overhead

Files:

- `ConvertMemoryToSPM.cpp`
- `workloads/kernels/layer_norm/kernel.py`
- `workloads/kernels/softmax/kernel.py`

Tasks:

- Inspect generated LLIR/assembly for the extra SPM store/read overhead that
  remains after DMA wait removal.
- Try schedule variants:
  - keep the first-pass vector value in registers for mean while writing SPM;
  - avoid redundant SPM read when a consumer can be fused into the fill pass;
  - use larger row chunks where RVV/gem5 constraints allow;
  - specialize gamma/beta policy separately from `x` residency.
- Do not make streaming DMA reduction default unless it beats the resident
  plan; it is now a fallback/debug path.

Exit criteria:

- LayerNorm 512x1024 is at least break-even under flushed ROI.
- 32x64 either breaks even or has an explained small-row rejection threshold.

### P3. Profitability Gate Refit

Files:

- `ConvertMemoryToSPM.cpp`
- promotion sidecar tests under `compiler/test/TritonCPU/`
- docs in `docs/plans/spm-explicit-promotion.md`

Tasks:

- Replace row-DMA constants with measured fill-on-first-pass costs:
  SPM writes, later SPM reads, extra instructions, bank conflicts.
- Keep D3 deterministic: no runtime profiling.
- Add separate decisions for:
  - `streaming_reduction_no_residency`,
  - `small_row_spm_overhead`,
  - `accepted_row_resident_fill_first`,
  - `accepted_block_resident_fill_first`.

Exit criteria:

- Small LayerNorm can reject by model.
- Large LayerNorm can accept as opt-in evidence.
- Softmax accepts only after measured row/block-resident win or rejects with a
  clear reason.

### P4. Documentation And Default Policy Update

Files:

- `docs/README.md`
- `docs/plans/phase3.md`
- `docs/plans/spm-explicit-promotion.md`
- `docs/plans/three-tier-placement.md`
- `docs/plans/compiler-roadmap.md`

Tasks:

- Keep Phase 4 graph/fusion as next major topic only after Phase 3.5 reaches
  its reduction exit criteria.
- Document that Tier 2 is the cross-kernel default, while SPM residency is
  single-kernel or fused-region scoped unless Tier 1 is explicitly implemented.
- Record measured reduction thresholds rather than treating cache-favorable
  old data as settled truth.

## Universal Or Per-Kernel?

The planned reduction pass should be **generic in admission and lifetime**, but
**kernel-specific in schedule templates**:

- Generic:
  - detect repeated row/block use,
  - allocate resident SPM slots,
  - write promotion records,
  - model costs,
  - reject unsupported lifetimes cleanly.
- Kernel templates:
  - LayerNorm: fill `x`, mean/variance/normalize reuse, gamma/beta stay cache
    unless cross-row reuse is made explicit.
  - Softmax: max pass, exp/sum pass, normalize/store; optional temp/vector
    policy depending on whether numerator is materialized or recomputed.
  - Future attention: Q/K/V tile residency and softmax hot state are fused
    region work, not required for Phase 3.5 default closure.

This mirrors the matmul situation: the promotion framework should be common,
but high-performance schedules still need pattern-specific lowering.

