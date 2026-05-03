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
| `softmax` | cache path until measured win | row/block-resident SPM is measured on a multi-block row case and either wins or rejects with evidence |
| `matmul` | SPM path | no regression under existing `make verify-matmul` and headline compare |
| `activation`, `residual_add`, `vector_add` | cache path | only revisit inside fusion or producer-consumer schedules |

Softmax has an extra admission caveat.  The current smoke workload keeps
`N == BLOCK_N`, so the row is loaded once and reused mainly in registers.  That
is useful coverage, but it is not enough evidence for SPM row/block residency.
Phase 3.5 must add a larger multi-block row shape before judging Softmax
promotion profitable.

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

## Progress Log

### 2026-05-03 P0 Baseline Tooling

Landed:

- Phase 3.5 presets for LayerNorm and Softmax manifests.
  LayerNorm now has small, large, row-resident, and D3 evidence presets.
  Softmax now has smoke and large-row presets.
- Preset-scoped environment variables in `run_experiment.py`, so opt-in
  row-resident and D3 runs no longer require hand-written `ENV=...` for the
  common Phase 3.5 cases.
- Compare output now writes `compare.csv`, `spm_stats.csv`, and
  `artifacts.csv` next to the existing text tables.  `artifacts.csv` records
  build artifact line counts and simple SPM markers such as `addrspace(3)` and
  `fence iorw`.
- Softmax kernel coverage now supports `N % BLOCK_N == 0` and uses three passes
  over the row: max, exp/sum, normalize/store.  This creates a valid
  multi-block row baseline for Phase 3.5, while default Softmax still verifies
  as cache path.
- A helper script, `workloads/scripts/phase35_baseline.sh`, runs the Phase 3.5
  verify/smoke/full baseline suites.

Validation already run:

- `make verify-layer_norm`
- `make verify-softmax`
- `python3 scripts/run_experiment.py softmax --mode verify --preset phase35-large-row`
- LayerNorm row-resident small verify, including accepted `LayerNorm x row`
  promotion evidence and empty tier JSON.
- LayerNorm D3 small verify, including `insufficient_row_work` rejection and
  cache-path IR.
- Softmax `phase35-smoke` compare under gem5 with result checking.

Interpretation caveat:

- The Softmax smoke compare showed nearly identical SPM-enabled and
  cache-baseline stats, and both artifacts have no `addrspace(3)` or DMA
  markers.  That is expected: default Softmax is still cache path.  Treat that
  run as a harness/baseline check, not an SPM promotion result.
- LayerNorm row-resident small remains an opt-in SPM-without-DMA path.  The
  smoke run generated `addrspace(3)` stores/loads and zero DMA fences, but small
  rows still have overhead.  It remains evidence for the small-row rejection
  threshold, not a default-enable result.

Next:

1. Run the full Phase 3.5 baseline suite with
   `workloads/scripts/phase35_baseline.sh full`.
2. Use the large LayerNorm row-resident/D3 runs to confirm whether 512x1024 is
   still near break-even after the tooling changes.
3. Use Softmax `phase35-large-row` as the cache-path baseline before adding any
   Softmax row/block-resident lowering.  Until that lowering exists, Softmax
   `spm` mode means SPM-enabled runtime with cache-path IR.
4. Start P1 by extracting a common `ReductionResidencyPlan` data structure from
   the current LayerNorm-only matcher, while keeping schedule lowering
   kernel-specific.

## Work Items

### P0. Stabilize The Measurement Baseline

Files:

- `workloads/scripts/run_experiment.py`
- `workloads/scripts/compare_stats.py`
- `workloads/kernels/layer_norm/experiment.toml`
- `workloads/kernels/softmax/experiment.toml`

Tasks:

- Add named Phase 3.5 presets for `layer_norm` and `softmax`: small row,
  large row, and flushed ROI compare.  For Softmax, the large row preset must
  allow `N > BLOCK_N` or introduce a second kernel variant before it is used as
  promotion evidence.
- Keep policy verification aware of SPM-without-DMA (`addrspace(3)` present,
  `fence iorw` absent).
- Add CSV fields for reduction-specific counters: SPM CPU reads/writes, DMA
  transfers, DMA wait stall cycles, bank conflicts, `simInsts`, and line count
  if available.

Exit criteria:

- Reproduce the current fill-on-first-pass LayerNorm numbers.
- Establish a Softmax baseline split into single-block smoke and multi-block
  row evidence; do not treat the single-block smoke as proof of SPM benefit.
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
- Treat this as a medium refactor, not a rename: the existing matcher is
  LayerNorm-specific and assumes three same-shape top-level loops.  The first
  landing should separate common evidence/lifetime/planning data from
  kernel-specific schedule templates.
- Represent:
  - source arg / row or block shape,
  - producer pass (`fill_on_first_pass`, `producer_store`, or `dma_prefetch`),
  - consumer passes,
  - required SPM slots,
  - buffer role (`resident_row`, `ping_pong_chunk`, `temp_vector`,
    `output_tile`),
  - rotation policy (`none`, `double_buffer`, `ring_N`).
- Keep LayerNorm as the first concrete lowering.
- Add Softmax matching after LayerNorm only for a schedule with real memory
  reuse: max pass, exp/sum pass, normalize/store over a multi-block row, or a
  clearly rejected single-block record.

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
- Default reduction promotion changes only after the measured preset suite has
  a stable threshold and `make verify-*` keeps the expected default cache/SPM
  policies.  Until then, D3 remains evidence-only.

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
