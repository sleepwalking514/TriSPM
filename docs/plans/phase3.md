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
  `vector_add`, and `layer_norm` pass functionally in both SPM and cache
  baseline modes.
- SPM size defaults have been unified to 256 KiB across the current path.
- Tier sidecar coverage has been audited; `matmul` is the mature real SPM
  workload, `vector_add` is intentionally not transformed, and `layer_norm`
  now enters the SPM path for mean, variance, and final normalize. Final
  normalize exercises the multi-load reduction/streaming matcher with `x`,
  `gamma`, and `beta`.
- Reduction support is not a standalone workload. It is the
  `transformReductionLoop` branch of `ConvertMemoryToSPM`; `layer_norm` is
  the production workload currently used to exercise it, while
  `convert-memory-to-spm.mlir` provides the synthetic structural lit coverage.
- Tier 2 L2-warming has been verified by the `dma_l2_warming`
  microbenchmark; the 4K-32K working-set sweep confirms near-100% L2 hits
  after DMA and about 2.8x speedup over the cold scalar-read phase.

### Current State of P3

`matmul` is functionally correct, has the right compute shape, and now has a
closed cold-start P3 headline: the large 1024×1024×1024 / 32×32×32 run beats
cache by 25.1% (SPM 288,976,339 cycles vs cache 386,049,495).  The 64×64
smoke case remains within +3.7% under a fair cold-cache comparison, which is
inside the ≤ cache × 1.05 regression guard.  This is the result of four
landed pieces:

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
  now available and is the P3 headline metric.  Steady-state warm-cache is
  retained only as an auxiliary sensitivity point because repeated launches
  mainly measure cache residency.

Small-case fair-baseline cycle count: SPM 38,361 vs cache 37,004 (+3.7%).
Large cold-start headline: SPM 288,976,339 vs cache 386,049,495 (-25.1%).
DMA stats on the 64×64 smoke case remain unchanged (128 transfers, 131,072
bytes, 4,096 SPM reads, 0 bank conflicts).  See `../archive/matmul-spm-lowering-closure.md` §P3.0 /
§P3.5 for the measured tables and interpretation.

Stage 2.6 also produced steady-state warm-cache data.  Those runs are not
the critical object for P3: cache keeps tiny repeated working sets resident,
while SPM repeats DMA/MMIO setup each launch.  Keep them as sensitivity data,
not as the headline criterion.

## Three-Tier Placement Status

`three-tier-placement.md` describes the intended placement framework:

- Tier 1: SPM-resident tensor, future work.
- Tier 2: cacheable DRAM plus DMA tiling, needed for the L2-warming claim.
- Tier 3: uncacheable DMA buffer, useful for pure vector workloads.

The MVP framework is landed. Coverage audit (2026-04-29, `make verify`) confirmed:

- `matmul`: args 0,1 → Tier 3. LLIR has 583 `addrspace(3)` + 80 `fence iorw`. Working.
- `vector_add`: empty tier JSON. Expected — single-block kernel with no loop, no tile reuse. SPM tiling has no benefit here.
- `layer_norm`: args 0,1,2 -> Tier 3 after rewriting all three passes to
  block pointers with constexpr `N`. LLIR has 38 `addrspace(3)` + 78
  `fence iorw`; gem5 compare passes functionally.  This verifies production
  use of `transformReductionLoop` for the two single-load reductions and the
  final 3-load normalize loop; there is no separate `reduction` workload.

See `three-tier-placement.md` §4.1 for full analysis.

## Near-Term Priority Order

1. ~~Resolve the Tier sidecar coverage mismatch.~~ Done (2026-04-29). See `three-tier-placement.md` §4.1.
2. ~~Implement `../evidence/l2_warming.md`'s `dma_l2_warming` microbenchmark.~~
   Done (2026-04-30): cacheable DMA, UC DMA, no-DMA scalar baseline,
   per-checkpoint stats, and 4K-32K working-set sweep are recorded in
   `../evidence/l2_warming.md`.
3. ~~Add Phase 6 tooling: `verify-spm-fires`, unified run/compare targets,
   and a stats CSV parser.~~ Done: `make verify` landed; `make run-<kernel>` / `make cmp-<kernel>` cover unified run/compare targets; `compare_stats.py` extracts metrics to `.txt` and CSV (`--csv` / `--spm-only-csv`). Remaining: Phase 6 comparison tooling.
4. ~~Upgrade reduction lowering from single-buffer prefetch to true
   double-buffer pipelining after the 2-D address fix.~~ Done: the lit
   test now verifies body-top wait + buffer flip + alternate-buffer
   prefetch, and `make cmp-layer_norm` passes with real SPM markers.
   Current 32x64 ROI result after final normalize also enters SPM: SPM
   125,593 cycles vs cache 6,138 cycles, 1,280 DMA transfers / 40,960
   bytes, waitFraction 0.4002. This is a
   correctness/coverage baseline, not a performance win.
5. Continue compiler robustness:
   GEMM A/B identity, cloned-read lookup, extra-load tolerance, reduction
   multi-load matching, and `DmaOpsToLLVM` MMIO base / future `useXspmInsn`
   options are done. The active remaining compiler item is bail-out cleanup /
   verification for partially-mutated paths.

## Configuration Notes

Current expected SPM defaults:

- `TRITON_SPM_BASE=0x40000000`
- `TRITON_SPM_SIZE=262144` (256 KiB)

Older notes in this file mentioned 64 KiB. Treat those as obsolete unless a
specific experiment intentionally overrides the environment.

## What Is Not Phase 3 Critical Path

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
- `phase3-compiler-backlog.md`: broader Phase 3 audit and remaining compiler robustness tasks.
- `phase3-execution-timeline.md`: current execution order and task list.
