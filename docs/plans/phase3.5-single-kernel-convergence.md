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

## Terminology: CPU-Direct SPM Residency

The row-resident LayerNorm path is **CPU-direct SPM residency**, not
DMA-based staging.

It works like this:

- The first reduction pass reads `x` from the normal DRAM/cache path.
- That same pass writes each loaded vector chunk into SPM through
  `addrspace(3)` stores.
- Later passes read `x` back from SPM through `addrspace(3)` loads.
- There are no DMA descriptors, MMIO programming stores, `spm.dma.w` waits, or
  `fence iorw` instructions on this path.

So "SPM without DMA" means: **the kernel really uses SPM, but CPU vector
loads/stores access it directly instead of asking the DMA engine to fill it**.
Artifact checks should therefore look for `addrspace(3)` with zero DMA/fence
markers, and stats should show nonzero `system.spm.*` CPU read/write counters
with `system.spm_dma.transfers = 0`.

## Progress Log

### 2026-05-03 P0 Baseline Tooling

Landed:

- Phase 3.5 presets for LayerNorm and Softmax manifests.
  LayerNorm now has small, large, row-resident, and D3 evidence presets.
  Softmax now has smoke and large-row presets.
- Preset-scoped environment variables in `run_experiment.py`, so opt-in
  row-resident and D3 runs no longer require hand-written `ENV=...` for the
  common Phase 3.5 cases.
- Compare tooling records text tables for symmetric stats, SPM-only stats, and
  build artifact markers such as `addrspace(3)` and `fence iorw`.
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

Full baseline results:

- Default LayerNorm remains cache path.  `phase35-small` measured 5,947 vs
  6,114 cycles and `phase35-large` measured 1,013,150 vs 1,013,120 cycles;
  both have `addrspace(3) = 0`, so these are baseline/noise checks.
- LayerNorm CPU-direct row residency is real but not yet profitable.  The small
  row run has `addrspace(3) = 44`, zero DMA/fences, and 7,018 vs 6,179 cycles
  (`+13.6%`).  The large row run has `addrspace(3) = 6`, zero DMA/fences,
  nonzero `system.spm.*` CPU read/write counters, and 1,015,329 vs 1,010,763
  cycles (`+0.5%`).
- D3 behaves as a conservative evidence gate: small rows reject as
  `insufficient_row_work` and stay cache path; large rows accept as opt-in
  evidence but still measure slightly slower (1,016,009 vs 1,013,120 cycles,
  `+0.3%`).  This is not enough to default-enable reduction promotion.
- Softmax smoke and large-row runs are cache-path baselines.  They have no
  `addrspace(3)`, no promotion sidecar, and identical instruction counts.
  `phase35-smoke` measured 46,125 vs 46,245 cycles (`-0.3%`) and
  `phase35-large-row` measured 7,705,586 vs 7,700,781 cycles (`+0.1%`);
  these numbers are runtime/cache noise, not SPM wins.

### 2026-05-03 P1a Reduction Residency Plan Extraction

Landed:

- The row-resident reduction path now uses an internal
  `ReductionResidencyPlan`: matcher -> plan -> lowering.  The first concrete
  lowering is still LayerNorm fill-on-first-pass CPU-direct row residency, so
  generated LayerNorm IR stays on the same SPM/no-DMA schedule.
- Promotion sidecars now include `residency_plan` evidence for row-resident
  reductions: producer pass, consumer passes, buffer role, rotation policy,
  copy-in mode, required SPM slots, and expected SPM/stat markers.
- At P1a time, Softmax row-resident detection learned to recognize the Phase 3.5 large-row
  three-pass pattern (`max`, `exp_sum`, `normalize_store`) and emits a clean
  rejected `Softmax x row` plan with
  `unsupported_reduction_residency_plan`.  Its IR remained cache path with no
  `addrspace(3)` until the P2a lowering below.
- `run_experiment.py`, the workload Makefile, and
  `phase35_baseline.sh verify` can now check rejection source and
  `residency_plan` evidence directly.

Validation run:

- `ninja -C compiler/build/cmake.linux-x86_64-cpython-3.12 triton-opt`
- `ninja -C compiler/build/cmake.linux-x86_64-cpython-3.12 /home/feige/TriSPM/compiler/python/triton/_C/libtriton.so`
- `lit -sv .../test/TritonCPU/convert-memory-to-spm-row-resident.mlir`
- `lit -sv .../test/TritonCPU/convert-memory-to-spm-promotion-report.mlir`
- `workloads/scripts/phase35_baseline.sh verify`
- `make -C workloads verify-matmul`

Next: P2 fill-on-first-pass overhead reduction, plus a row-block DMA
double-buffer design probe.  The scheduler still needs a measured performance
win before reduction promotion can become the default.

### 2026-05-03 P2a Softmax Row-Resident Lowering And Producer-Store Probe

Landed:

- Softmax row-resident plans now have a concrete CPU-direct lowering instead
  of stopping at `unsupported_reduction_residency_plan`.  For the large-row
  pattern, the first pass (`max`) can fill the resident SPM row, and the
  `exp_sum` plus `normalize_store` consumers read `x` from SPM.
- A schedule variant is selectable with
  `TRITON_SPM_ROW_RESIDENT_PRODUCER_PASS=producer_store`.  This leaves the
  first reduction pass on the original cache path, writes `x` into SPM during
  the next consumer pass, and uses SPM for the final consumer.  It was added to
  test whether avoiding one SPM read beats fill-on-first-pass.
- Phase 3.5 manifests now include producer-store presets for LayerNorm and
  Softmax.  The build script includes the producer-pass mode in the Triton
  compile-cache key so the two schedules cannot accidentally reuse stale IR.
- `phase35_baseline.sh verify` now checks both Softmax SPM schedules as
  accepted row-resident plans, with empty tier sidecars, no fences, and
  promotion sidecars that expose the selected producer pass.

Validation run:

- `ninja -C compiler/build/cmake.linux-x86_64-cpython-3.12 triton-opt`
- `ninja -C compiler/build/cmake.linux-x86_64-cpython-3.12 /home/feige/TriSPM/compiler/python/triton/_C/libtriton.so`
- Manual `triton-opt | FileCheck` checks for Softmax row-resident IR,
  producer-store IR, and Softmax promotion sidecar.  Direct `llvm-lit` from
  this checkout still needs the generated site config wiring, so the manual
  FileCheck commands were used for the changed prefixes.
- `workloads/scripts/phase35_baseline.sh verify`
- `make -C workloads verify-matmul`
- Gem5 flushed ROI compares for:
  - `layer_norm phase35-row-resident-producer-large`
  - `layer_norm phase35-row-resident-producer-small`
  - `softmax phase35-row-resident-large-row`
  - `softmax phase35-row-resident-producer-large-row`

Measured findings:

| Kernel preset | Schedule | Result |
|---|---|---|
| `layer_norm phase35-row-resident-large` | fill-on-first-pass | 1,015,329 SPM vs 1,010,763 cache cycles (`+0.5%`); zero DMA/fence; SPM reads 4,227,072 B and writes 2,097,152 B |
| `layer_norm phase35-row-resident-producer-large` | producer-store | 1,191,772 SPM vs 1,013,124 cache cycles (`+17.6%`); fewer SPM reads, but the extra cache-path pass is worse |
| `layer_norm phase35-row-resident-producer-small` | producer-store | 5,635 SPM vs 5,618 cache cycles (`+0.3%`); near parity, but small rows still do not justify default promotion |
| `softmax phase35-row-resident-large-row` | fill-on-first-pass | 7,174,501 SPM vs 7,709,817 cache cycles (`-6.9%`); zero DMA/fence; SPM reads 1,023,392 B and writes 524,288 B |
| `softmax phase35-row-resident-producer-large-row` | producer-store | 7,658,442 SPM vs 7,706,343 cache cycles (`-0.6%`); fewer SPM reads, but slower than fill-on-first-pass |

Conclusion:

- Softmax is now the strongest reduction SPM candidate.  The large-row
  fill-on-first-pass schedule has a measured SPM win and should feed the P3
  profitability refit.
- Producer-store is useful as evidence, but not as the preferred schedule.
  It reduces SPM read bytes, yet for large LayerNorm the saved SPM read loses
  badly to the extra cache-path read.  For Softmax it remains positive but
  much weaker than first-pass fill.
- LayerNorm remains opt-in/conservative.  Small producer-store is near parity
  and large fill-on-first-pass is near parity, but there is no stable win on
  both required shapes.

Next: stay in P2 and try the row-block DMA alternative.  Do not move to the P3
profitability refit until the row-block DMA evidence says whether coarse
row-block staging is a useful alternative to CPU-direct fill-on-first-pass.

### 2026-05-03 P2b DMA-Prefetch Probe

Landed:

- `row-resident-producer-pass=dma_prefetch` is now an opt-in lowering.  The
  current Triton reduction kernels expose one program row at a time, so this is a
  **chunk-DMA prefetch into a resident row buffer**, not the full future
  row-block `[r, r + R)` double buffer.  The producer pass waits for each
  DMA-filled chunk, prefetches the next chunk while computing the current one,
  and the later passes read the resident row through `addrspace(3)`.
- Softmax and LayerNorm manifests now have DMA-prefetch Phase 3.5 presets, and
  `run_experiment.py --expect-dma` can verify DMA/fence marker expectations
  separately from CPU-direct row residency.

Validation run:

- `ninja -C compiler/build/cmake.linux-x86_64-cpython-3.12 triton-opt
  /home/feige/TriSPM/compiler/python/triton/_C/libtriton.so`
- Softmax DMA verify:
  `python3 ./scripts/run_experiment.py softmax --mode verify --preset phase35-row-resident-dma-large-row --expect-spm true --expect-tier-json empty --expect-dma true --expect-promotion-source 'Softmax x row' --expect-residency-plan 'Softmax x row'`
- LayerNorm DMA verify for small and large rows with the same `--expect-dma`
  policy, source, and residency-plan checks.
- Existing CPU-direct row-resident verifies for Softmax large-row and LayerNorm
  large still pass with zero fence markers, proving the DMA path did not replace
  the preferred fill-on-first-pass lowering.
- Gem5 flushed ROI compares for:
  - `softmax phase35-row-resident-dma-large-row`
  - `layer_norm phase35-row-resident-dma-small`
  - `layer_norm phase35-row-resident-dma-large`

Measured findings:

| Kernel preset | Schedule | Result |
|---|---|---|
| `softmax phase35-row-resident-dma-large-row` | chunk-DMA prefetch into resident row | 7,398,839 SPM vs 7,707,609 cache cycles (`-4.0%`); 2,048 DMA transfers, 524,288 DMA bytes, 166,326 wait-stall cycles |
| `layer_norm phase35-row-resident-dma-small` | chunk-DMA prefetch into resident row | 30,538 SPM vs 5,618 cache cycles (`+443.6%`); 256 DMA transfers, 8,192 DMA bytes, 8,478 wait-stall cycles |
| `layer_norm phase35-row-resident-dma-large` | chunk-DMA prefetch into resident row | 7,178,094 SPM vs 1,010,763 cache cycles (`+610.2%`); 65,536 DMA transfers, 2,097,152 DMA bytes, 1,713,120 wait-stall cycles |

Conclusion:

- Softmax DMA-prefetch is a real SPM path and still beats cache on the large-row
  case, but it is weaker than CPU-direct fill-on-first-pass (`-4.0%` vs
  `-6.9%`).  The extra DMA descriptors, MMIO/fence cost, and wait polling do not
  improve the current one-row program schedule.
- LayerNorm DMA-prefetch should be rejected for the current kernel shape.  It
  restores the exact tiny-chunk DMA pathology Phase 3.5 was meant to escape:
  65,536 DMA transfers on the 512x1024 case dominate the run despite row reuse in
  later passes.
- This does **not** reject the future row-block DMA idea.  The current IR cannot
  expose a multi-row block lifetime, so a true row-block double buffer still
  requires a different kernel schedule or pass structure that processes multiple
  rows per program/block and amortizes one coarse DMA phase over many rows.

### 2026-05-03/04 P2c Softmax Row-Block A/B DMA Prototype

Landed:

- Softmax now has explicit row-block controls: `ROW_BLOCK` and
  `ROW_GROUP_BLOCKS`.  The default remains the original one-row path; the
  `phase35-row-block-dma-large-row` preset now uses `ROW_BLOCK=2` and
  `ROW_GROUP_BLOCKS=8` so one program exposes a longer outer row-block loop.
- `row_block_dma` now matches that outer loop for Softmax.  The lowering
  allocates two resident row-block input buffers, emits a prologue DMA for the
  first row block, waits for the current buffer at the top of each outer-loop
  iteration, prefetches the next row block into the alternate SPM slot, then
  runs max, exp/sum, and normalize/store from the current SPM slot.
- The current row-block DMA shape is a full-row-block copy: 1024 columns x 2
  rows x fp32 = 8 KiB per row block.  The SPM view uses full-row strides, so the
  compute loops read the same layout that the 2D DMA writes.
- Promotion evidence records `source = "Softmax x row block"`, scope
  `program-row-block-group`, `buffer_role = resident_row_block`,
  `rotation_policy = double_buffer`, and `required_spm_slots = 2`.

Validation run:

- `ninja -C compiler/build/cmake.linux-x86_64-cpython-3.12 triton-opt
  /home/feige/TriSPM/compiler/python/triton/_C/libtriton.so`
- Existing CPU-direct Softmax verify:
  `python3 ./scripts/run_experiment.py softmax --mode verify --preset phase35-row-resident-large-row --expect-spm true --expect-tier-json empty --expect-dma false --expect-promotion-source 'Softmax x row' --expect-residency-plan 'Softmax x row'`
- Row-block DMA verify:
  `python3 ./scripts/run_experiment.py softmax --mode verify --preset phase35-row-block-dma-large-row --expect-spm true --expect-tier-json empty --expect-dma true --expect-promotion-source 'Softmax x row block' --expect-residency-plan 'Softmax x row block'`
- Gem5 flushed ROI compare:
  `python3 ./scripts/run_experiment.py softmax --mode spm-compare --preset phase35-row-block-dma-large-row`

Measured findings:

| Kernel preset | Schedule | Result |
|---|---|---|
| `softmax phase35-row-block-dma-large-row` old `ROW_BLOCK=4`, `ROW_GROUP_BLOCKS=2` | true row-block A/B DMA, 16 KiB row-block buffers | 5,346,794 SPM vs 5,101,703 cache cycles (`+4.8%`); 32 2D DMA transfers, 524,288 DMA bytes, 57,169 wait-stall cycles, 0 SPM bank conflicts |
| `softmax p2c-rb4-rg4` | same 16 KiB buffers, longer row-block group | 5,075,308 SPM vs 5,107,055 cache cycles (`-0.6%`); wait-stall cycles fall to 28,091 |
| `softmax p2c-rb4-rg8` | same 16 KiB buffers, still longer row-block group | 5,061,998 SPM vs 5,104,950 cache cycles (`-0.8%`); wait-stall cycles fall to 14,498 |
| `softmax p2c-rb2-rg4` | 8 KiB row-block buffers | 3,966,595 SPM vs 4,350,227 cache cycles (`-8.8%`); 64 2D DMA transfers, avg DMA latency 1,700.7 cycles, 25,059 wait-stall cycles |
| `softmax phase35-row-block-dma-large-row` updated preset | 8 KiB row-block buffers, longer row-block group | 3,953,189 SPM vs 4,346,982 cache cycles (`-9.1%`); 64 2D DMA transfers, 524,288 DMA bytes, 12,297 wait-stall cycles, 0 SPM bank conflicts |
| `softmax p2c-rb8-rg4` | 32 KiB row-block buffers | 7,793,190 SPM vs 7,670,844 cache cycles (`+1.6%`); 16 2D DMA transfers, avg DMA latency 6,920.3 cycles, 28,338 wait-stall cycles |

Interpretation:

- This is the first true row-block DMA prototype in the Softmax path.  It is no
  longer the P2b per-chunk DMA schedule: descriptor count drops from 2,048 in
  one-row DMA-prefetch to 64 full-row-block DMA transfers at the current 8 KiB
  granularity.
- The `ROW_BLOCK=4`, `ROW_GROUP_BLOCKS=2` result was correct but not
  profitable.  Increasing `ROW_GROUP_BLOCKS` reduced exposed prologue/wait cost,
  while reducing the row-block from 16 KiB to 8 KiB cut average DMA latency from
  about 3.3K cycles to about 1.7K cycles.  Going coarser to 32 KiB regresses
  again.
- `ROW_BLOCK=2`, `ROW_GROUP_BLOCKS=8` is the current best measured SPM schedule
  for this Softmax shape (`-9.1%` vs cache), ahead of CPU-direct row residency
  (`-6.9%`).  P2c should now feed D3/P3 profitability refit rather than changing
  default reduction promotion immediately.

### 2026-05-04 P2d Softmax Measurement Audit And Result Gate

Correction:

- The P2c row-block results above remain useful as implementation evidence for
  the `row_block_dma` lowering, but the performance interpretation is now
  downgraded.  That comparison used the same row-block schedule for both modes,
  while later sweeps showed that `ROW_BLOCK` and `ROW_GROUP_BLOCKS` dominate
  runtime.  Same-schedule evidence can show whether an SPM lowering helps a
  particular schedule, but it cannot prove that SPM beats the best cache kernel.
- A more serious correctness hole was found in the Softmax schedule controls:
  `GRID_X = M / (ROW_BLOCK * ROW_GROUP_BLOCKS)` was applied even when
  `ROW_BLOCK == 1`, but that branch computes only `row = tl.program_id(0)` and
  does not loop over `ROW_GROUP_BLOCKS`.  Therefore `ROW_BLOCK=1,
  ROW_GROUP_BLOCKS>1` runs only `M / ROW_GROUP_BLOCKS` rows.  Any results from
  that shape, including cases where SPM counters are zero and both modes look
  extremely fast, must be treated as invalid partial-compute data.
- The earlier `ROW_BLOCK=2` / `ROW_BLOCK=4` row-block runs may still be valid
  same-schedule diagnostics after result checking, but they are not sufficient
  evidence for default Softmax SPM policy.  Default enablement now requires a
  decoupled search: best legal cache schedule vs best legal SPM schedule.

Landed:

- `run_gem5.sh` now captures gem5 stdout/stderr in
  `workloads/m5out/<kernel>/<tag>/<mode>/run.log`.
- `run_experiment.py` now enforces a result gate after every gem5 run when
  `CHECK_RESULT=1`.  The run must contain `PASS:` and must not contain `FAIL:`,
  `MISMATCH`, or `SKIP:`.  If the gate fails, the experiment exits with an error
  that points at `run.log`.
- `spm-compare` removes stale `compare_vs_cache_best.txt`, `spm_stats.txt`, and
  `artifacts.txt` before running.  These files are generated only after the SPM
  run passes the result gate, so failed correctness no longer leaves a misleading
  old comparison table behind.
- `run_experiment.py --mode cache-search --sweep blocking` runs cache candidates
  under the cache path and selects the minimum `system.cpu.numCycles` for each
  shape, recording the winner in
  `workloads/m5out/<kernel>/<shape>/cache_best.json`.
- `cache-search` removes stale `cache_best.json` for the shape before running
  candidates, so a failed search cannot silently leave an old best baseline in
  place.
- `spm-compare` no longer runs a same-schedule cache baseline.  It loads the
  shape's `cache_best.json`, runs only the requested SPM candidate, and writes
  `compare_vs_cache_best.txt`, `spm_stats.txt`, and `artifacts.txt` under that
  SPM blocking directory.
- The m5out layout is now
  `workloads/m5out/<kernel>/<shape>/<cache|spm>/<blocking>/...`; compare files
  live under the SPM blocking because they are evidence for that SPM candidate.
  Preset names prefix the blocking/schedule component when present, so variants
  with the same numeric tile shape but different policy env do not collide.
- Single-kernel compare artifacts are text-only.  Spreadsheet-style outputs and driver flags
  were removed from the current compare path.
- Softmax now has an explicit `SOFTMAX_SCHEDULE` axis in the workload manifest
  and kernel.  `canonical` requires `ROW_BLOCK=1`, `ROW_GROUP_BLOCKS=1`, and one
  row per program.  `row_block` requires `ROW_BLOCK>1` and exposes the outer
  row-block group loop used by the row-block DMA lowering.  Illegal combinations
  fail at build time instead of silently changing the work count.

Next methodology:

- Split Softmax performance reporting into two policy numbers:
  `best_cache` and SPM candidate/best-SPM cycles.  The policy headline is SPM
  vs `best_cache`; same-schedule A/B is intentionally out of the default
  driver.
- The main workflow is:
  `cache-search --sweep blocking` for the shape first, then `spm-compare` for
  one SPM candidate.  If `cache_best.json` already exists, later SPM compares
  reuse it and skip the cache run.
- Keep one `workloads/kernels/softmax/kernel.py` for now.  The implemented first
  split is `SOFTMAX_SCHEDULE = canonical | row_block`: `canonical` means one row
  per program, and `row_block` means one program processes vector row-block
  tiles.  A scalar `row_group` schedule can still be added later if cache search
  needs it, but it is no longer implicit in `ROW_GROUP_BLOCKS`.
- Add manifest presets/sweeps for cache search and SPM search separately.  Cache
  search should vary legal schedule shape without SPM env vars.  SPM search
  should use the same correctness gate plus explicit promotion/DMA marker
  expectations, but it is allowed to prefer a different schedule.
- Only split into separate `kernel_cache.py` and `kernel_spm.py` after the shared
  schedule-axis version proves too restrictive.  Keeping one file first avoids
  hidden semantic drift between cache and SPM kernels while the methodology is
  being repaired.

Immediate next step:

1. Treat the 128x1024 Softmax results below as the new policy baseline.
2. Keep standalone Softmax default-off unless a later SPM schedule beats
   best-cache by a stable margin across repeated runs and adjacent shapes.
3. After Softmax is stable, repeat the same cache-best-first pattern for the
   other single kernels.  Kernels with no meaningful blocking axis can use a
   single `default` candidate so their directory layout still matches the
   cache-best workflow.

### 2026-05-04 P2e Softmax Schedule Cleanup And Best-Cache Result

Landed:

- `workloads/kernels/softmax/kernel.py` now has explicit schedule semantics:
  `canonical` is the normal one-row-per-program kernel, while `row_block` is the
  vector row-block schedule used by the DMA prototype.  Both still lower through
  the same AOT `softmax` symbol.
- `workloads/kernels/softmax/harness.c` computes `gridX` from
  `SOFTMAX_SCHEDULE_ID`, prints the schedule name, and rejects illegal
  schedule/blocking combinations at compile time.
- `workloads/kernels/softmax/experiment.toml` now has named canonical presets
  and a blocking sweep with only legal cache candidates:
  canonical `rb1/rg1`, row-block `rb2/rg4`, row-block `rb2/rg8`, and row-block
  `rb4/rg4`.

Validation:

- Default canonical verify: `make verify-softmax`
- Canonical cache-path verify:
  `python3 ./scripts/run_experiment.py softmax --mode verify --preset canonical-large-row --expect-spm false --expect-tier-json empty --expect-dma false`
- Canonical SPM-direct verify:
  `python3 ./scripts/run_experiment.py softmax --mode verify --preset canonical-spm-direct-large-row --expect-spm true --expect-tier-json empty --expect-dma false --expect-promotion-source 'Softmax x row' --expect-residency-plan 'Softmax x row'`
- Row-block DMA verify:
  `python3 ./scripts/run_experiment.py softmax --mode verify --preset phase35-row-block-dma-large-row --expect-spm true --expect-tier-json empty --expect-dma true --expect-promotion-source 'Softmax x row block' --expect-residency-plan 'Softmax x row block'`

Best-cache search for 128x1024:

| Cache candidate | Cycles |
|---|---:|
| canonical `rb1/rg1` | 7,778,735 |
| row-block `rb2/rg4` | 4,292,282 |
| row-block `rb2/rg8` | 4,291,669 |
| row-block `rb4/rg4` | 5,614,819 |

SPM candidates versus best-cache `row_block rb2/rg8`:

| SPM candidate | Cycles | Result |
|---|---:|---|
| canonical SPM-direct fill-on-first-pass | 7,175,796 | `+67.2%` vs best-cache; faster than canonical cache but not policy-competitive |
| row-block A/B DMA `rb2/rg8` | 4,312,695 | `+0.5%` vs best-cache; correct and near parity, but not a default win |
| row-block A/B DMA `rb2/rg16` diagnostic | 4,313,160 | `+0.5%` vs best-cache; wait stalls fall to 6,342 cycles, but total time does not improve |

Interpretation:

- The clean canonical experiment says CPU-direct SPM residency is real
  (`addrspace(3)`, zero DMA/fence, 1,023,488 SPM read bytes and 524,288 SPM
  write bytes), but it is not the right standalone Softmax default because the
  best cache schedule is row-blocked and much faster.
- The row-block DMA lowering is correct and close to best-cache.  It performs 64
  2D DMA transfers, moves 524,288 DMA bytes, has 12,528 wait-stall cycles, and
  has zero SPM bank conflicts in this run.  That is useful implementation
  evidence, not enough default-policy evidence.
- A follow-up `ROW_GROUP_BLOCKS=16` diagnostic halves DMA wait stalls
  (`12,528 -> 6,342`) but leaves total cycles flat
  (`4,312,695 -> 4,313,160`).  A `ROW_BLOCK=4`, `ROW_GROUP_BLOCKS=8` CLI probe
  is much worse at 5,522,771 cycles because the generated row-block IR/assembly
  roughly doubles (`addrspace(3) = 1,536`, 11,191 asm lines).  The current
  bottleneck is therefore not exposed DMA wait; it is the SPM row-block
  load/issue path and code shape, visible as about `+40.6%` issued
  `FloatMemRead` versus best-cache even when wait stalls are reduced.
- Current policy: standalone Softmax stays cache path by default.  Keep
  row-block DMA as an opt-in schedule and revisit Softmax SPM inside fused
  attention or after a broader adjacent-shape sweep shows a stable win.

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
- Keep policy verification aware of CPU-direct SPM residency (`addrspace(3)`
  present, `fence iorw` absent, and zero DMA transfers).
- Add text-table fields for reduction-specific counters: SPM CPU reads/writes,
  DMA transfers, DMA wait stall cycles, bank conflicts, `simInsts`, and line
  count if available.
- Gate gem5 compares on workload correctness.  When `CHECK_RESULT=1`, compare
  artifacts must only be generated after both `spm` and `cache` logs report
  `PASS:` and no mismatch/failure markers.

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
  `ReductionResidencyPlan`.  Done in P1a for the internal plan/evidence shape.
- Treat this as a medium refactor, not a rename: the existing matcher is
  LayerNorm-specific and assumes three same-shape top-level loops.  The first
  landing should separate common evidence/lifetime/planning data from
  kernel-specific schedule templates.
- P1a scope: preserve current LayerNorm IR while changing the internal shape to
  `match -> ReductionResidencyPlan -> lower`.  Done; performance was not
  optimized in this step.
- Represent:
  - source arg / row or block shape,
  - producer pass (`fill_on_first_pass`, `producer_store`, or `dma_prefetch`),
  - consumer passes,
  - required SPM slots,
  - buffer role (`resident_row`, future `resident_row_block`,
    `ping_pong_chunk`, `temp_vector`, `output_tile`),
  - rotation policy (`none`, `double_buffer`, `ring_N`).
- Keep LayerNorm as the first concrete lowering.
- Add Softmax matching after LayerNorm only for a schedule with real memory
  reuse: max pass, exp/sum pass, normalize/store over a multi-block row, or a
  clearly rejected single-block record.  P1a detected the large-row reuse shape
  and left a rejected unsupported plan record; P2a adds the measured
  CPU-direct row-resident lowering.

Exit criteria:

- P1a complete: LayerNorm still emits fill-on-first-pass SPM with no DMA
  fences, and unsupported Softmax row residency leaves clean rejection records
  plus cache-path IR.
- P2a complete: Softmax large-row emits accepted row-resident SPM IR and
  promotion evidence.  Default enablement still waits for P3 profitability
  refit.

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
- Try producer-store as a concrete SPM-path alternative.  Done in P2a:
  it reduces SPM reads but is not the preferred schedule after measurement.
- Do not make streaming DMA reduction default unless it beats the resident
  plan; it is now a fallback/debug path.

P2b/P2c result: chunk-DMA was measured and rejected as too descriptor-heavy.
True Softmax row-block A/B DMA now builds, verifies, and wins after P2c
granularity tuning: the old 16 KiB row-block cut exposed too much wait, while
the current 8 KiB row-block cut with a longer row-block group beats cache.

This is a different idea from the old streaming reduction DMA path.  The old
path DMAed tiny per-loop chunks and paid descriptor/MMIO/wait overhead for each
small piece of reuse, which is why it lost badly.  A more plausible DMA-based
reduction schedule is to stage a **block of rows** at a time:

1. DMA rows `[r, r + R)` into SPM buffer A using one coarse 2D row-block copy
   or a small bounded descriptor group.
2. Compute all reduction passes for those `R` rows, reusing the resident
   row-block from SPM.
3. While computing buffer A, prefetch rows `[r + R, r + 2R)` into SPM buffer B.
4. Swap A/B and continue.

The expected benefit is descriptor/wait amortization: one DMA phase feeds many
row computations and multiple row passes.  For LayerNorm this means mean,
variance, and normalize/store reuse the same row-block.  For Softmax this
means max, exp/sum, and normalize/store reuse the same row-block.  This should
be represented as `producer_pass = dma_prefetch`, `buffer_role =
resident_row_block` or `ping_pong_chunk`, and `rotation_policy =
double_buffer`.

Admission requirements before implementation:

- The kernel schedule must process a row block, not only one row/program.
- SPM must fit two input row-block buffers plus per-row temporaries:
  roughly `2 * R * N * elem_bytes + temps`.
- Rows must be contiguous or cheaply describable by a bounded 2D DMA
  descriptor sequence.
- `R * row_compute` must be large enough to hide or amortize DMA wait and MMIO
  overhead.
- The DMA wait placement must not reintroduce the RVV codegen/fence regression
  seen in earlier DMA paths.
- Small rows should still reject; this is meant for large rows or multi-row
  batches where reuse and overlap can pay for DMA setup.

Exit criteria:

- LayerNorm 512x1024 is at least break-even under flushed ROI.  P2a result:
  fill-on-first-pass remains near parity (`+0.5%`); producer-store is rejected
  for large LayerNorm (`+17.6%`).
- 32x64 either breaks even or has an explained small-row rejection threshold.
  P2a result: producer-store reaches near parity (`+0.3%`) but does not
  justify default promotion by itself.
- Softmax large-row gets a measured row-resident result.  P2a result:
  fill-on-first-pass wins (`-6.9%`), producer-store is weaker (`-0.6%`).
- Row-block DMA double buffering has a buildable prototype or a documented
  rejection with measured descriptor/wait overhead and SPM-capacity math.
  P2b built and measured the chunk-DMA prefetch variant; P2c now builds and
  measures the true Softmax row-block A/B DMA variant.  P2d downgrades this to
  same-schedule evidence only: the row-block lowering is real, but the
  performance policy must be re-evaluated against the best legal cache schedule.

### P3. Profitability Gate Refit

Files:

- `ConvertMemoryToSPM.cpp`
- promotion sidecar tests under `compiler/test/TritonCPU/`
- docs in `docs/plans/spm-explicit-promotion.md`

Tasks:

- Replace row-DMA constants with measured fill-on-first-pass costs:
  SPM writes, later SPM reads, extra instructions, bank conflicts.
- Keep D3 deterministic: no runtime profiling.
- Refit Softmax profitability against `best_spm` vs `best_cache`, not only
  same-schedule A/B.  The same-schedule A/B remains useful for debugging the
  lowering, but it must not drive default policy by itself.
- Add separate decisions for:
  - `streaming_reduction_no_residency`,
  - `small_row_spm_overhead`,
  - `accepted_row_resident_fill_first`,
  - `accepted_block_resident_fill_first`.

Exit criteria:

- Small LayerNorm can reject by model.
- Large LayerNorm can accept as opt-in evidence.
- Softmax accepts only after measured row/block-resident win or rejects with a
  clear reason against the best legal cache schedule.
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
