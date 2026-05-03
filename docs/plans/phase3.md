# TriSPM Compiler — Phase 3 Current Status

This file is the Phase 3 status page. The older "how to add
`ConvertMemoryToSPM`" checklist is now superseded by `phase3-compiler-backlog.md`,
`../archive/matmul-spm-lowering-closure.md`, `three-tier-placement.md`, and
`../evidence/l2_warming.md`.

## What Phase 3 Covers

Phase 3 makes Triton CPU kernels use the scratchpad memory path
automatically:

1. Lower tiled DRAM loads to DMA-to-SPM staging in `ConvertMemoryToSPM`.
2. Preserve good RVV codegen after SPM rewriting.
3. Add three-tier tensor placement so the launcher can choose cacheable
   DRAM, uncacheable DMA buffer, or future SPM-resident allocations.
4. Build the verification and stats tooling needed for Phase 6 evaluation.

## Current State

### Done

- Phase 1 / Phase 2 plumbing is in place: RISC-V AOT, DMA dialect ops,
  `DmaOpsToLLVM`, SPM address space 3, and gem5 SPM/DMA execution all exist.
- `ConvertMemoryToSPM` fires on real `matmul` output and emits DMA/SPM code.
- The main GEMM codegen regression is fixed. Current `matmul` SPM assembly
  has `vfmacc.vf=256`, `vrgather=0`, `vfmadd.vv=0`, and whole-register
  spill/reload counts matching the cache baseline.
- The earlier gem5 DMA VA-to-PA page-boundary bug is resolved; `matmul`,
  `vector_add`, `layer_norm`, `activation`, `residual_add`, and `softmax`
  pass functionally in both SPM-enabled and cache-baseline modes for the
  current smoke coverage.
- SPM size defaults have been unified to 256 KiB across the current path.
- Tier sidecar coverage has been audited; `matmul` is the mature real SPM
  workload, while `vector_add` and default `layer_norm` are intentionally
  cache-path kernels. LayerNorm can still opt into SPM reduction coverage for
  mean, variance, and final normalize with `TRITON_ENABLE_SPM_REDUCTIONS=1`;
  final normalize exercises the multi-load reduction/streaming matcher with
  `x`, `gamma`, and `beta`.
- Reduction support is not a standalone workload. It is the
  `transformReductionLoop` branch of `ConvertMemoryToSPM`; `layer_norm` is
  the opt-in production workload used to exercise it, while
  `convert-memory-to-spm.mlir` provides the synthetic structural lit coverage
  and verifies that the default reduction-off policy leaves reductions on the
  cache path.
- Compiler robustness for the current Phase 3 workload set is closed:
  GEMM matching derives A/B identity from the contract operands, cloned reads
  use `IRMapping`, extra non-dot loads are tolerated, reduction matching accepts
  multiple shared-IV streams, and GEMM/reduction bail-out paths clean up
  partially inserted prologue work.
- Tier 2 L2-warming has been verified by the `dma_l2_warming`
  microbenchmark; the 4K-32K working-set sweep confirms near-100% L2 hits
  after DMA and about 2.8x speedup over the cold scalar-read phase.
- Transformer-facing workload coverage before Phase 4 attention has landed:
  `activation` (SiLU), `residual_add`, and row-wise `softmax` now have
  `kernel.py`, `harness.c`, `experiment.toml`, flushed ROI measurement,
  result checks, `make verify-<kernel>` policy checks, and SPM-vs-cache smoke
  compares. These are coverage/harness workloads; they do not add new default
  SPM promotion policy.
- Graph-level conservative placement has a build/verify MVP:
  `workloads/scripts/graph_placement.py` reads graph tensor-edge metadata,
  emits per-node `KERNEL_TIER_OVERRIDE`, builds SPM artifacts, and verifies
  launcher allocation dispatch. The first fixture, `layer_norm_qkv`, keeps the
  LayerNorm producer output / QKV activation input cacheable Tier 2 while
  allowing external read-only Q/K/V weights to use Tier 3. This does not yet
  link or run a multi-kernel graph, implement Tier 1, or perform fused
  promotion.
- Explicit promotion D1 evidence/reporting has landed for matmul. The existing
  fused scheduler now writes promotion records for the B tile window, A micro
  tile, and accumulator tile when `TRITON_SPM_PROMOTION_REPORT=1` is enabled.
  D1b made the sidecar a versioned debug/evidence schema with accepted/rejected
  status, structural `reason_code`, exact vs estimated `field_kinds`, an
  explicit non-durable-contract string, accepted matmul coverage, default
  reduction/cache-path rejection coverage, and report-off coverage. This is
  evidence/debug output only; it does not yet make promotion records drive
  scheduling or profitability.
- Explicit promotion D2 has landed as an opt-in row-resident LayerNorm
  prototype. `TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS=1` enables a separate
  lowering that materializes `x[row, :]` into SPM on the first reduction pass,
  reuses it across variance and normalize, and leaves `gamma` / `beta` on
  cache. Default `layer_norm` still verifies as cache path, and the old
  `TRITON_ENABLE_SPM_REDUCTIONS=1` streaming reduction coverage remains
  separate. D2 uses the D1 sidecar only as debug/evidence; it does not drive
  planner, scheduling, buffer layout, `windowK`, or profitability. The
  2026-05-03 fill-on-first-pass rework removed serialized row DMA/wait overhead
  and improved LayerNorm from clear regressions to near parity (32x64 +4.4%,
  512x1024 +0.5%), so reduction SPM remains active optimization work rather
  than a closed negative result.
- Explicit promotion D3 has landed as a conservative opt-in profitability gate.
  `TRITON_ENABLE_SPM_PROMOTION_PROFITABILITY=1` records static
  descriptor/MMIO/wait/fence/byte/use evidence in the D1 sidecar, accepts the
  existing fused matmul B-window / accumulator evidence, rejects streaming
  reductions and small row-resident LayerNorm reductions, and can accept large
  fill-on-first-pass row-resident evidence for opt-in experiments. Default
  LayerNorm stays cache path until measured wins are clear. The sidecar remains
  debug/evidence only; it is not read back by planning, placement, scheduling,
  `windowK`, or buffer-layout code.
- A no-regression issue found during D1 validation is fixed: DMA fences still
  lower to `fence iorw, iorw`, but no longer carry a generic inline-asm memory
  clobber. The clobber caused the 256x256x256 SPM matmul to regress from the
  archived ~1.729M-cycle baseline to ~1.987M cycles by increasing reloads around
  fence sites. After the fix, 64x64x64 SPM-only correctness passes and
  D1b validation kept 256x256x256 SPM-only cycles at 1,729,209 with the fast
  5913-line assembly shape.

### Current State of P3

Phase 3 matmul and compiler robustness are converged for the current
single-kernel scope, but the broader single-kernel SPM policy is not fully
converged for reductions. The 2026-05-03 LayerNorm re-audit found a concrete
implementation artifact in the old row-resident path: every row paid a
serialized DMA descriptor and wait before reduction work began. Removing that
artifact with fill-on-first-pass SPM materialization collapses the old
regression to near parity, so reduction promotion remains an active
optimization line. The next compiler gate is
`phase3.5-single-kernel-convergence.md`: close LayerNorm/Softmax
row/block-resident SPM before Phase 4 graph/attention/fusion becomes the main
line. Phase 4/5/6 producer-consumer promotion, broader evaluation, and final
blocking sweeps remain open on top of this corrected baseline.

`matmul` is functionally correct, has the right compute shape, and has crossed
over under the cold-start P3 headline metric. SPM beats the cache baseline on
the current large runs, and the small 64-case is no longer a regression point.
Keep this page qualitative for now: the best SPM and cache blocking choices are
not identical, so a fair final headline needs a blocking sweep rather than a
single fixed number. This state is the result of four landed pieces:

- **Stage 2.5 (harness, fair baseline)**: scrub L1+L2 before `m5_reset_stats`
  so SPM and cache both start with cold DRAM; init / reference run on
  cacheable shadow buffers and DMA-publish into the launcher's input
  buffers in one shot, keeping the SPM check phase fast even when inputs
  live in the uncacheable DMA buffer.  The fair baseline alone shrinks
  the headline gap from +75.9% to +6.7%.
- **Stage 3 (compiler + simulator, MMIO compaction)**: pack
  `SRC_STRIDE`+`DST_STRIDE` into `REG_STRIDES_PACKED` (offset 0x38) and
  pack `HEIGHT` into the upper 32 bits of `REG_LEN`, dropping the per-
  descriptor MMIO store sequence from 6 to 4.  Compiler `DmaOpsToLLVM`
  emits the packed form; the simulator accepts both packed and legacy
  layouts.
- **Stage 3 (compiler, prologue wait elimination)**: the body-top
  `DmaWait` already covers prologue DMAs on iteration 0, so the
  redundant prologue wait was removed from `transformGemmLoop`.
- **Stage 2.6 (measurement, size/steady sweep)**: large cold-start data is
  now available and shows crossover. Steady-state warm-cache is retained only
  as an auxiliary sensitivity point because repeated launches mainly measure
  cache residency.

The archived measurement log still contains older concrete points, but do not
use those as the current headline without rerunning a fair blocking sweep for
both SPM and cache. See `../archive/matmul-spm-lowering-closure.md` §P3.0 /
§P3.5 for the historical measured tables and interpretation.

Stage 2.6 also produced steady-state warm-cache data.  Those runs are not
the critical object for P3: cache keeps tiny repeated working sets resident,
while SPM repeats DMA/MMIO setup each launch.  Keep them as sensitivity data,
not as the headline criterion.

## Three-Tier Placement Status

`three-tier-placement.md` describes the intended placement framework:

- Tier 1: SPM-resident tensor, future work.
- Tier 2: cacheable DRAM plus DMA tiling, needed for the L2-warming claim.
- Tier 3: uncacheable DMA buffer, useful for pure vector workloads.

For single-kernel Phase 3 verification, the current MVP sidecar can still
classify no-scalar-reuse inputs as Tier 3. For end-to-end transformer work,
the policy is stricter and graph-aware: intermediate activations and kernel
outputs stay Tier 2 cacheable by default; Tier 3 is reserved for external
read-only DMA-only streaming inputs/weights; Tier 1 is future SPM-resident hot
state. The first graph-level build/verify planner implements this policy above
the compiler lowering layer; executable graph harnesses remain next work. See
`three-tier-placement.md` §2.1 / §6.2.

The MVP framework is landed. Current workload verification confirms:

- `matmul`: args 0,1 → Tier 3. LLIR still has SPM markers; it is the
  performance workload.
- `vector_add`: empty tier JSON. Expected — single-block kernel with no loop,
  no tile reuse. SPM tiling has no benefit here.
- `activation`: empty tier JSON and no SPM markers. Expected — SiLU
  elementwise cache-path workload.
- `residual_add`: empty tier JSON and no SPM markers. Expected — cache-path
  residual elementwise workload.
- `softmax`: empty tier JSON and no SPM markers by default. Expected for the
  first row-wise smoke workload; future row/block-resident promotion experiments
  should stay opt-in and measured before changing the default. The latest
  LayerNorm result shows this boundary should be revisited with a reuse-aware
  reduction schedule rather than closed based on the old DMA-row experiment.
- `layer_norm`: default `expect_spm = false`, empty tier JSON, and no SPM
  markers. New flushed compares show the SPM-enabled runtime with cache-path
  LayerNorm is effectively equal to cache baseline: 32x64 is -3.6% cycles in
  the latest run, and 512x1024 is -0.3%.
- `layer_norm` reduction SPM remains available as opt-in coverage:
  `TRITON_ENABLE_SPM_REDUCTIONS=1` produces Tier 3 args 0,1,2 and SPM markers
  for mean/variance/final normalize. This path is correctness/coverage only.
- `layer_norm` row-resident promotion is also opt-in:
  `TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS=1` generates SPM markers and a
  `LayerNorm x row` promotion sidecar while keeping the tier JSON empty. It is
  experiment evidence, not a default policy. The current fill-on-first-pass
  implementation removes DMA fences from this path; with
  `TRITON_ENABLE_SPM_PROMOTION_PROFITABILITY=1`, D3 still rejects small rows as
  `insufficient_row_work` but accepts large rows as opt-in evidence.
  Forcing the old streaming reduction inputs to Tier 2 cacheable still leaves
  that path far slower than cache, so the default-off decision for streaming
  SPM is not just a Tier 3 uncacheable-buffer workaround.

See `three-tier-placement.md` §4.1 for full analysis.

## Near-Term Priority Order

1. ~~Resolve the Tier sidecar coverage mismatch.~~ Done (2026-04-29). See `three-tier-placement.md` §4.1.
2. ~~Implement `../evidence/l2_warming.md`'s `dma_l2_warming` microbenchmark.~~
   Done (2026-04-30): cacheable DMA, UC DMA, no-DMA scalar baseline,
   per-checkpoint stats, and 4K-32K working-set sweep are recorded in
   `../evidence/l2_warming.md`.
3. ~~Add Phase 6 tooling: `verify-spm-policy`, unified run/compare targets,
   and a stats CSV parser.~~ Done: `make verify` landed; `make run-<kernel>` / `make cmp-<kernel>` cover unified run/compare targets; `compare_stats.py` extracts metrics to `.txt` and CSV (`--csv` / `--spm-only-csv`). Remaining: Phase 6 comparison tooling.
4. ~~Upgrade reduction lowering from single-buffer prefetch to true
   double-buffer pipelining after the 2-D address fix.~~ Done: the lit
   test now verifies body-top wait + buffer flip + alternate-buffer
   prefetch, and `TRITON_ENABLE_SPM_REDUCTIONS=1` keeps the real LayerNorm
   SPM coverage path available. Archived opt-in compares made the performance
   caveat explicit: flushed 32x64, 512x1024, and 1024x1024 SPM-reduction runs
   were much slower than cache. Default AOT therefore disables reduction SPM
   and keeps LayerNorm cache-path.
5. ~~Finish compiler robustness for current Phase 3 coverage:~~ Done.
   GEMM A/B identity, cloned-read lookup, extra-load tolerance, reduction
   multi-load matching, and `DmaOpsToLLVM` MMIO base / future `useXspmInsn`
   options are done. GEMM/reduction bail-out cleanup is also done; lit tests
   cover dynamic-step no-DMA cases and partial prologue cleanup.
6. ~~Before Phase 4 attention or the transformer pipeline, add
   transformer-facing single-kernel coverage.~~ Done (2026-05-02): `activation`
   (SiLU), `residual_add`, and row-wise `softmax` build, verify clean cache-path
   policy, run flushed ROI compares, and check results under gem5 in both modes.
7. ~~Before the transformer pipeline, implement graph-level conservative
   placement build/verify (`three-tier-placement.md` §2.1 / §6.2).~~ Done:
   the `layer_norm_qkv` graph fixture verifies the cacheable activation
   backbone and selective uncacheable external weights. Remaining work is an
   executable multi-kernel graph harness plus cache-baseline comparison.

## Configuration Notes

Current expected SPM defaults:

- `TRITON_SPM_BASE=0x40000000`
- `TRITON_SPM_SIZE=262144` (256 KiB)
- `TRITON_ENABLE_SPM_REDUCTIONS=0` by default. Set it to `1` only for
  reduction-path correctness/coverage experiments.

Older notes in this file mentioned 64 KiB. Treat those as obsolete unless a
specific experiment intentionally overrides the environment.

## What Is Not Phase 3 Critical Path

- Graph-level conservative placement build/verify is not needed for the
  single-kernel P3 headline and is now in place; the executable graph harness
  and graph-vs-cache comparison are P0 for Phase 5 transformer evaluation.
- Elementwise activation/residual kernels usually should not get an SPM pass
  in isolation. They should have harness/verify coverage and normally remain
  cache-path kernels unless fused into a neighboring matmul/reduction/attention
  kernel.
- Tier 1 resident SPM needs func arg addrspace(3), harness pre-launch DMA,
  and an SPM layout manifest. Keep it in `three-tier-placement.md` §6.1 until Tier 2
  evidence is stable.
- `has_scalar_reuse` should only be expanded when a new workload needs it.
  Candidate extensions include scalar from `vector.extract`, loop-external
  loads, and `tt.reduce` intermediate scalar patterns.
- Attention, SPM output writeback, multi-kernel lifetime management, and the
  transformer block belong to later Phase 4 / Phase 5 work.

## Source Of Truth

- `../archive/matmul-spm-lowering-closure.md`: archived matmul SPM lowering closure and P3 measurements.
- `three-tier-placement.md`: three-tier placement design and backlog.
- `../evidence/l2_warming.md`: Tier 2 L2-warming verification results.
- `phase3-compiler-backlog.md`: broader Phase 3 compiler audit and closure record.
- `phase3-execution-timeline.md`: current execution order and task list.
