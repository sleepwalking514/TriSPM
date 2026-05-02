# Explicit SPM Promotion Plan

## Why This Exists

The current SPM compiler path has two different ideas mixed together:

- **Backing placement**: where a tensor lives before the kernel starts.
  Tier 2 means cacheable DRAM, Tier 3 means uncacheable DMA-buffer DRAM.
- **SPM promotion**: which tile/value is explicitly copied into SPM, for how
  long, and how many compute uses consume it before eviction.

GPU shared memory compilers mostly win through the second idea.  They do not
use shared memory merely as a latency-hiding staging cache; they promote a tile
because the schedule will reuse it.  Double buffering is then a pipeline detail
around a promoted tile, not the whole policy.

TriSPM currently has partial promotion:

- `transformGemmLoop` is mostly DMA staging: A/B tiles are double-buffered for
  one K iteration.
- `transformFusedMicroGemmLoop` is closer to GPU shared memory: B is resident
  for a small K window, A is streamed as micro-tiles, and the accumulator tile
  is spilled in SPM between microM loops.
- `transformReductionLoop` is only streaming staging.  LayerNorm showed that
  this is a bad fit: even Tier 2 cacheable inputs still produce many tiny DMA
  transactions and lose to cache.

The next design step should make promotion explicit and general, instead of
adding more ad hoc double-buffer cases.

## Revised Principle

Do not ask "can this load be DMAed into SPM?"

Ask "can the compiler prove enough reuse inside a bounded lifetime to justify
promoting this tile/value into SPM?"

If the answer is no, leave the operation on the cache path.  Tier 2 remains the
safe backing placement for graph edges and mixed cache/SPM pipelines, but it is
not a substitute for SPM residency.

## Promotion Model

For each candidate tile/value, build a promotion record:

| Field | Meaning |
|---|---|
| `source` | memref arg, producer result, or scalar/vector temporary |
| `scope` | function, program/block, loop window, or single iteration |
| `shape` | promoted tile shape in elements |
| `uses` | number of compute uses before eviction |
| `copy_in` | DMA, CPU store, or producer writes directly to SPM |
| `copy_out` | none, DMA writeback, or CPU/vector transfer write |
| `bytes` | SPM footprint, with alignment |
| `overhead` | descriptor stores, waits, fences, extra loop/control ops |
| `benefit` | avoided DRAM/cache reads/writes and removed redundant DMA |

The compiler should promote only if:

```text
benefit(tile, uses) > copy_in + copy_out + sync/control overhead
and live_spm_bytes(scope) <= SPM capacity
and the access pattern is statically describable
```

This makes LayerNorm's current reduction SPM fail the profitability test:
each 8-float chunk has too little reuse per DMA transaction, and the pass reads
`x` across separate passes instead of keeping the row resident.

## Scopes

### 1. Iteration Staging

Current double-buffer path.  This is useful when each tile feeds expensive
compute immediately, such as GEMM, but it is not enough by itself.

Keep this as the fallback for GEMM-like loops.

### 2. Loop-Window Residency

A tile is loaded once and reused across an inner loop nest.  This already exists
in `transformFusedMicroGemmLoop` for the B window.

Near-term extension:

- Represent B-window, A-micro, and accumulator allocations as promotion records.
- Add counters to the pass/debug output: promoted bytes, use count, avoided DMA.
- Use the same framework to decide when `windowK` should be 1, 2, 4, or disabled.

### 3. Whole-Row / Whole-Block Residency

Useful for reductions only when the whole row/block fits and is reused across
multiple passes.

LayerNorm example for small/medium N:

```text
DMA x[row, :] once into SPM
mean = reduce(spm_x)
var  = reduce(spm_x)
normalize using spm_x + gamma/beta
```

This removes the three independent `x` streams.  Gamma/beta may still stay on
cache path or be promoted only if their reuse across rows is made visible by a
different schedule.

Profitability guard:

- Require `N * sizeof(T)` to fit in a row-resident SPM budget.
- Require at least two uses of the promoted row.
- Prefer cache path if the row would be split into many tiny DMA descriptors
  with no per-row reuse.

### 4. Producer-to-Consumer SPM Residency

This is the real "shared-memory-like" story for transformer fusion.

Examples:

- `layer_norm + qkv`: write `X_norm` tile to SPM and consume it by Q/K/V matmuls
  before writing or evicting.
- `residual + layer_norm`: keep residual sum tile in SPM for the following
  reduction when the fused schedule proves bounded lifetime.
- `flash attention`: keep Q resident while streaming K/V blocks; keep softmax
  stats in SPM as small hot state.

This requires either fusion or a graph-level schedule.  Tier 2 cacheable DRAM is
still the safe unfused boundary.

## Compiler Architecture

### P0: Separate Placement From Promotion

Rename the mental model:

- `SPMTensorPlacement`: chooses backing allocation / graph safety
  (Tier 2 vs Tier 3, future Tier 1 for external resident inputs).
- `SPMPromotionPlanner`: chooses intra-kernel SPM residency and schedule.

Promotion should not be inferred from Tier 3.  A Tier 2 tensor can be promoted
into SPM for compute.  A Tier 3 tensor can be left unpromoted if the access is
not profitable.

### P1: Promotion Records In ConvertMemoryToSPM

Refactor existing GEMM fused scheduler to build explicit promotion records:

- B window: loop-window resident, DMA in, many uses.
- A micro tile: iteration staging, DMA in, one use.
- Accumulator tile: loop-window resident temporary, SPM read/write.

This is mostly structural and should preserve current matmul behavior.

### P1: Row-Resident Reduction Prototype

Add a new opt-in lowering before the streaming reduction path:

```text
if reduction row bytes <= budget and x is reused by >=2 loops:
  promote x row once
  rewrite mean/variance/final-normalize x loads to SPM reads
else:
  leave reduction on cache path by default
```

This should be a separate experiment from the existing `transformReductionLoop`.
The existing streaming reduction SPM path has already shown poor performance.

First target: LayerNorm with `N <= 1024` and `BLOCK_N = 8`, where one row is
small enough but currently creates thousands of tiny DMA transactions across
three passes.

### P2: Promotion Profitability Analysis

Add a conservative cost model:

- DMA descriptor count and MMIO store count.
- Wait/fence count.
- Tile bytes and SPM bytes.
- Static use count from the loop nest.
- Optional measured constants from existing gem5 stats.

If the model cannot prove reuse, do not promote.

### P2: Graph/Fusion Promotion

For transformer work, add a graph-level promotion manifest that can describe:

- fused producer/consumer groups,
- tensor tile lifetimes,
- SPM-resident intermediate ranges,
- fallback DRAM materialization points.

This should integrate with graph-level Tier 2/3 placement, but remain a separate
decision: graph placement says where tensors live between kernels; promotion says
what lives in SPM during a scheduled fused region.

## What Not To Do

- Do not make every cacheable Tier 2 tensor automatically enter SPM lowering.
  LayerNorm Tier 2 experiments showed this is not enough.
- Do not treat double buffering as the optimization.  It hides latency only
  after a profitable promotion decision has been made.
- Do not use Tier 1 as "whole tensor in SPM" by default.  Whole tensor residency
  is only profitable for small hot state or explicitly scheduled fused regions.
- Do not rely on cache for the data that the compiler can prove is reused inside
  a tight SPM-resident scope.  Cache remains the fallback, not the reuse plan.

## Minimal Milestones

1. **Promotion terminology/documentation**: update roadmap and placement docs so
   Tier placement and SPM promotion are no longer conflated.
2. **Matmul promotion records**: refactor the existing fused scheduler without
   changing generated behavior.
3. **LayerNorm row-resident prototype**: opt-in path that copies one row of `x`
   once and reuses it across mean/variance/normalize.
4. **Profitability gate**: keep reduction SPM default-off unless row-resident
   promotion beats cache on 32x64 and 512x1024.
5. **Fusion promotion plan**: prototype `layer_norm + qkv` or attention-style
   Q/K/V tile residency once transformer harness exists.
