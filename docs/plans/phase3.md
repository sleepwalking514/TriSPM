# TriSPM Compiler — Phase 3 Current Status

This file is the Phase 3 status page. The older "how to add
`ConvertMemoryToSPM`" checklist is now superseded by `todo.md`,
`spm-lowering.md`, `3tier.md`, and
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
bytes, 4,096 SPM reads, 0 bank conflicts).  See `spm-lowering.md` §P3.0 /
§P3.5 for the measured tables and interpretation.

Stage 2.6 also produced steady-state warm-cache data.  Those runs are not
the critical object for P3: cache keeps tiny repeated working sets resident,
while SPM repeats DMA/MMIO setup each launch.  Keep them as sensitivity data,
not as the headline criterion.

## Three-Tier Placement Status

`3tier.md` describes the intended placement framework:

- Tier 1: SPM-resident tensor, future work.
- Tier 2: cacheable DRAM plus DMA tiling, needed for the L2-warming claim.
- Tier 3: uncacheable DMA buffer, useful for pure vector workloads.

The MVP framework is landed. Coverage audit (2026-04-29, `make verify`) confirmed:

- `matmul`: args 0,1 → Tier 3. LLIR has 583 `addrspace(3)` + 80 `fence iorw`. Working.
- `vector_add`: empty tier JSON. Expected — single-block kernel with no loop, no tile reuse. SPM tiling has no benefit here.
- `layer_norm`: empty tier JSON. Root cause: pointer arithmetic loads + reduction matcher only accepts 1 non-dot load, but layer_norm pass 3 has 3 loads (x, gamma, beta). Needs kernel rewrite to block pointers + reduction matcher generalization.

See `3tier.md` §4.1 for full analysis.

## Near-Term Priority Order

1. ~~Resolve the Tier sidecar coverage mismatch.~~ Done (2026-04-29). See `3tier.md` §4.1.
2. Implement `../evidence/l2_warming.md`'s `dma_l2_warming` microbenchmark
   with cacheable DMA, UC DMA, no-DMA scalar baseline, per-checkpoint stats,
   and a working-set sweep.
3. ~~Add Phase 6 tooling: `verify-spm-fires`, unified run/compare targets,
   and a stats CSV parser.~~ Done: `make verify` landed; `make run-<kernel>` / `make cmp-<kernel>` cover unified run/compare targets; `compare_stats.py` extracts metrics to `.txt` and CSV (`--csv` / `--spm-only-csv`). Remaining: Phase 6 comparison tooling.
4. After placement coverage stabilizes, clean up compiler robustness:
   derive GEMM A/B identity and K dimension from `vector.contract` indexing
   maps, generalize reduction matching to multiple loads sharing the loop IV,
   and add `DmaOpsToLLVM` options for MMIO base and future `useXspmInsn`.

## Configuration Notes

Current expected SPM defaults:

- `TRITON_SPM_BASE=0x40000000`
- `TRITON_SPM_SIZE=262144` (256 KiB)

Older notes in this file mentioned 64 KiB. Treat those as obsolete unless a
specific experiment intentionally overrides the environment.

## What Is Not Phase 3 Critical Path

- Tier 1 resident SPM needs func arg addrspace(3), harness pre-launch DMA,
  and an SPM layout manifest. Keep it in `3tier.md` §6.1 until Tier 2
  evidence is stable.
- `has_scalar_reuse` should only be expanded when a new workload needs it.
  Candidate extensions include scalar from `vector.extract`, loop-external
  loads, and `tt.reduce` intermediate scalar patterns.
- Attention, SPM output writeback, multi-kernel lifetime management, and the
  transformer block belong to later Phase 4 / Phase 5 work.

## Source Of Truth

- `spm-lowering.md`: current `matmul` SPM lowering and P3 DMA-latency work.
- `3tier.md`: three-tier placement design and backlog.
- `../evidence/l2_warming.md`: Tier 2 L2-warming verification plan.
- `todo.md`: broader Phase 3 audit and remaining compiler robustness tasks.
- `next_steps.md`: current execution order and task list.
