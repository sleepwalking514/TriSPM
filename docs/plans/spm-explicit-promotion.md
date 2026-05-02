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

## Top-Priority Implementation Decisions

These decisions are intentionally narrow.  The first goal is to prove the
promotion model without destabilizing the working matmul path; only after each
validation gate passes should the next generalization happen.

### D1. Start With An Internal Promotion Record, Then Export Evidence

D1 status (2026-05-02): **completed**.

D1a and D1b have landed and are validated as metadata-only. D2 row-resident
reductions have now landed as a separate opt-in prototype. Only the actual cost
model and automatic scheduling decisions belong to D3.

Completed in D1a:

- `ConvertMemoryToSPM` now emits pass-local promotion records for the existing
  fused matmul schedule: B tile window, A micro tile, and accumulator tile.
- `TRITON_SPM_PROMOTION_REPORT=1` / the pass `promotion-report` option writes a
  `<kernel>_promotions.json` sidecar next to the tier sidecar.
- The AOT build path now forces recompilation when sidecars are needed, so tier
  and promotion reports are not silently lost on Triton cache hits.
- Validation kept generated matmul behavior intact after fixing an unrelated
  DMA fence codegen regression: `make verify-matmul` passes; a 64x64x64
  SPM-only run passes result checking; 256x256x256 SPM-only cycles are
  `1,729,063`, matching the archived `1,729,209` baseline within noise.

Completed in D1b:

- The promotion sidecar now declares `schema_version = 1`,
  `schema = "triton_cpu_spm_promotion_d1"`, and an explicit debug/evidence
  contract string. It is not a graph manifest and not a durable IR contract.
- Accepted records carry `status = "accepted"`, `reason_code`,
  `reason`, and `field_kinds` that distinguish exact/static fields from
  D1 structural estimates such as overhead and benefit.
- Rejected candidate records carry `status = "rejected"`, `pattern`, `source`,
  `scope`, `shape`, `uses`, `bytes`, `copy_in`, `copy_out`, `reason_code`,
  `reason`, and field-kind annotations. Current structural reason codes include
  `policy_disabled`, `unsupported_pattern`, `unsupported_config`,
  `dynamic_shape_or_stride`, `no_bounded_lifetime`, and
  `spm_capacity_overflow`.
- The report test now covers accepted fused matmul records, a rejected default
  reduction/cache-path candidate, and report-off behavior.
- Real AOT matmul promotion reports use the new schema after a full compiler
  rebuild. Cache-baseline builds still do not emit promotion reports.
- D1b no-regression validation kept generated matmul behavior intact:
  `make verify-matmul` passes, 64x64x64 SPM-only result check passes, and
  256x256x256 SPM-only cycles are `1,729,209` with the fast 5913-line assembly
  shape.

Important finding:

- The observed `~1,987,453` cycle matmul regression was not caused by promotion
  reporting.  It was introduced earlier by adding a generic `"~{memory}"`
  clobber to DMA fence inline assembly.  That clobber forced extra reloads
  around every DMA fence and changed the 256x256x256 matmul assembly from the
  fast 5913-line / 529-reload shape to a slower 6161-line / 608-reload shape.
  Removing the clobber restored byte-identical fast assembly while keeping
  `fence iorw, iorw` and volatile MMIO accesses.

D1 is now closed. If D1 report fields change again, rerun the same guard set:
promotion-report lit/FileCheck, `make verify-matmul`, 64x64x64 SPM-only
correctness, and 256x256x256 SPM-only no-regression.

Not D1 work:

- Promotion records do not need to drive `windowK`, `microM`, buffer layout, or
  fused matmul lowering yet. Those scheduling decisions are D3 and later.
- Row-resident LayerNorm, softmax/block-resident paths, fused
  producer-consumer promotion, and end-to-end experiments are later gates.  They
  should build on D1 evidence/reporting rather than changing the D1 schema into
  a planner interface.

Open pitfall:

- Do not reintroduce a generic DMA fence memory clobber as a quick correctness
  fix without rerunning the 64x64x64 correctness check and 256x256x256
  no-regression check.  If stronger compiler ordering is needed later, make it
  a targeted lowering mode with its own performance gate.

Implemented D1 shape:

- `ConvertMemoryToSPM` owns a pass-local C++ promotion report.
- The report records `source`, `scope`, `shape`, `uses`, `bytes`, `copy_in`,
  `copy_out`, exact/estimated field kinds, accepted status, and rejection
  reason codes.
- The existing fused matmul scheduler populates B-window, A-micro, and
  accumulator records without changing generated IR.

Validated D1 guard set:

- `make verify-matmul` produces the same SPM policy as before.
- 64x64x64 SPM-only result check passes.
- 256x256x256 SPM-only stays at the archived ~1.729M-cycle baseline.
- Debug sidecar data makes promoted bytes, use count, and rejected candidates
  visible.

Only after D1 validation:

- Use the records to drive `windowK` selection.
- Consider making promotion records a more durable IR attribute, JSON sidecar,
  or graph manifest.  Do not design a permanent manifest before the single-kernel
  and first fusion experiments show what information is actually needed.

### D2. Keep Row-Resident Reduction Separate From Streaming Reduction

D2 status (2026-05-03): **completed as opt-in prototype; not default-worthy**.

First step:

- Add a new opt-in row-resident lowering before `transformReductionLoop`.
- Keep the existing streaming reduction path as opt-in correctness coverage.
- First target is LayerNorm `x[row, :]` only: DMA one row into SPM, reduce mean
  from SPM, reduce variance from the same SPM row, and normalize from the same
  SPM row.
- Keep `gamma` and `beta` on the cache path in the first version unless their
  cross-row reuse is made explicit by a later schedule.

Initial scope:

- Static `N`.
- fp32 only unless another dtype is needed by a concrete experiment.
- `row_bytes = N * sizeof(T)` must fit a conservative row-resident budget.
  Start with `N <= 1024` / 4 KiB rows for the prototype, then sweep the budget
  upward only after measurements justify it.
- Require at least two uses of the row before eviction; LayerNorm should expose
  three uses of `x`.

Implemented in D2:

- New pass/backend controls:
  `enable-row-resident-reductions`,
  `row-resident-max-bytes`,
  `TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS=1`, and
  `TRITON_SPM_ROW_RESIDENT_MAX_BYTES` (default 4096).
- The row-resident path is separate from the old streaming reduction path.  It
  matches the current LayerNorm-style trio of top-level loops over the same
  `x[row, :]`: mean, variance, and normalize.
- Accepted candidates must use rank-1 contiguous fp32 `x` loads, static loop
  bounds/step, and a row footprint within the row-resident budget.
- The lowering emits one row DMA and one wait per program row, then rewrites
  only the `x` loads in mean/variance/normalize to read from SPM.  `gamma` and
  `beta` stay on the cache path.
- `TRITON_ENABLE_SPM_REDUCTIONS=1` still exercises the older streaming
  reduction coverage path.  If both reduction flags are set, the row-resident
  path stays separate and the streaming path is not mixed in.
- The D1 sidecar is reused only as debug/evidence output.  D2 records accepted
  `LayerNorm x row` promotions and row-resident rejections, but the sidecar does
  not drive planning, scheduling, `windowK`, buffer layout, or profitability.

Validation:

- `make verify-layer_norm` remains cache path by default.
- `TRITON_ENABLE_SPM_REDUCTIONS=1` continues to exercise the old streaming
  coverage path.
- A separate opt-in row-resident flag verifies that `x` is promoted once per row
  and reused across mean/variance/normalize.
- 32x64 and 512x1024 must beat or match the cache baseline before this path can
  become a default policy.

D2 validation results:

- Focused compiler lit:
  `llvm-lit -v test --filter='convert-memory-to-spm-row-resident|convert-memory-to-spm-promotion-report|convert-memory-to-spm\.mlir'`
  passed 3/3 tests.
- Default LayerNorm:
  `make verify-layer_norm` passed with no SPM LLIR markers and an empty tier
  sidecar.
- Old streaming reduction:
  `make verify-layer_norm ENV='TRITON_ENABLE_SPM_REDUCTIONS=1' EXPECT_SPM=true EXPECT_TIER_JSON=non_empty`
  passed and still emits non-empty tier evidence for the old path.
- New row-resident opt-in:
  `make verify-layer_norm ENV='TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS=1 TRITON_SPM_PROMOTION_REPORT=1' EXPECT_SPM=true EXPECT_TIER_JSON=empty EXPECT_PROMOTION_SOURCE='LayerNorm x row'`
  passed.  The promotion sidecar records `accepted_d2_opt_in_row_resident`; the
  tier sidecar remains `{}`.
- Both flags together:
  `TRITON_ENABLE_SPM_REDUCTIONS=1 TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS=1`
  still verifies as row-resident evidence with an empty tier sidecar, proving the
  old streaming coverage path is not mixed into D2.
- Performance is still cache-favorable.  32x64 measured 10,279 SPM cycles vs
  6,138 cache cycles (`+67.5%`).  512x1024 measured 1,245,422 SPM cycles vs
  1,012,922 cache cycles (`+23.0%`).

D2 conclusion:

- The prototype proves the compiler can explicitly promote a whole row and
  explain the decision without changing the default policy.
- The prototype must remain opt-in because it does not beat or match cache on
  the required 32x64 and 512x1024 comparisons.
- D3 may now start with a conservative profitability/rejection model.  D3 should
  keep the D1/D2 sidecar as evidence only, not as a planner or scheduling
  interface.

Only after D3 or later validation:

- Generalize from LayerNorm to softmax or another canonical block-pointer
  reduction.
- Consider row/block residency for producer results, not just memref arguments.

### D3. Use A Conservative Profitability Gate Before Default Promotion

First step:

- Implement static guards first: static access pattern, bounded lifetime,
  `uses >= 2`, row/tile bytes within budget, and live SPM bytes within capacity.
- Count DMA descriptors, MMIO stores, waits, fences, copy bytes, and avoided
  repeated reads/writes.
- If the model cannot prove reuse and bounded overhead, leave the operation on
  the cache path.

Validation:

- The model must reject the current tiny-chunk streaming LayerNorm SPM path by
  default.
- The model must accept the existing matmul fused scheduler cases that already
  perform well.
- The model must explain decisions in debug output so failed experiments are
  diagnosable without reading generated IR by hand.

Only after this validation:

- Fit constants from gem5 stats for descriptor/MMIO/fence overhead.
- Use measured constants to tune thresholds.  Do not make the first version
  profile-guided; keep it deterministic and conservative.

### D4. Delay Graph/Fusion Manifest Design Until One Fusion Exists

First step:

- Treat graph/fusion promotion as the next layer after single-kernel promotion
  is working.
- Keep graph-level Tier 2/3 placement conservative: unfused tensor boundaries
  stay cacheable unless they are external read-only DMA-only streams.
- Pick one concrete fusion prototype, preferably `layer_norm + qkv`, before
  freezing any manifest format.

Validation:

- The fused schedule must show a bounded SPM lifetime for the promoted producer
  result.
- The fallback materialization point must be explicit: if fusion cannot be used,
  the value writes back to Tier 2 cacheable DRAM.
- The manifest, if introduced, must describe producer/consumer groups, promoted
  ranges, lifetimes, and fallback points.  It should not duplicate backing
  placement decisions already handled by `SPMTensorPlacement`.

Only after this validation:

- Extend to attention-style Q/K/V tile residency and softmax hot state.
- Use the graph manifest as paper evidence for multi-kernel scheduling rather
  than as a hand-written configuration mechanism.

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

D2 implemented the first version as a new opt-in lowering before the streaming
reduction path:

```text
if reduction row bytes <= budget and x is reused by >=2 loops:
  promote x row once
  rewrite mean/variance/final-normalize x loads to SPM reads
else:
  leave reduction on cache path by default
```

This should be a separate experiment from the existing `transformReductionLoop`.
The existing streaming reduction SPM path has already shown poor performance.

First target completed: LayerNorm with `N <= 1024` and `BLOCK_N = 8`.  It emits
one row DMA per program row and reuses that row across three `x` uses, while
leaving `gamma` and `beta` cached.  The experiment still loses to cache, so it
remains opt-in evidence for D3 rather than a default policy.

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

## Milestones And Validation Gates

### Gate A: Single-Kernel Promotion Closure

Goal: make promotion a real compiler decision for standalone kernels.

Steps:

1. **Terminology/documentation**: update roadmap and placement docs so Tier
   placement and SPM promotion are no longer conflated.
2. **Matmul promotion records**: refactor the existing fused scheduler without
   changing generated behavior.  Done: the fused matmul scheduler reports
   B-window, A-micro, and accumulator promotion records without changing the
   generated schedule; D1b added the versioned schema, structural rejected
   candidates, report-off coverage, and no-regression validation.
3. **Promotion evidence**: report promoted bytes, use counts, descriptor counts,
   and rejected candidates.  D1 now exports stable evidence fields and
   structural rejection records. D3 may add measured profitability scores later,
   but it should not turn the D1 evidence sidecar into a planner contract.
4. **LayerNorm row-resident prototype**: done in D2 as a separate opt-in path
   that copies one row of `x` once and reuses it across
   mean/variance/normalize.
5. **Single-kernel profitability gate**: keep reduction SPM default-off unless
   row-resident promotion beats or matches cache on 32x64 and 512x1024. D2 did
   not meet this bar, so D3 must implement a conservative profitability gate
   before any default enablement.
6. ~~**Workload breadth**: add at least one more reduction/streaming workload
   such as softmax and at least one cache-only elementwise/residual workload to
   show the gate can both promote and reject.~~ Done for first smoke coverage
   (2026-05-02): `activation` (SiLU), `residual_add`, and row-wise `softmax`
   now build, verify as cache path by default, and pass flushed ROI gem5 smoke
   compares in both SPM-enabled and cache-baseline modes. This supplies the
   single-kernel rejection/coverage fixtures; it does not complete the
   profitability gate or row/block-resident promotion work.

Gate A is complete only when the default policy is automatic for the single
kernel set: matmul promotes, cache-only elementwise kernels stay clean, and
LayerNorm/softmax promote only when the row/block-resident strategy is measured
profitable.

### Gate B: Multi-Kernel / Fusion Promotion

Goal: show that SPM residency is useful across producer-consumer boundaries, not
just inside one loop nest.

Steps:

1. Build a transformer-facing harness for at least one fused region.
2. Prototype `layer_norm + qkv` or an attention-style schedule.
3. Keep the producer result in SPM only when the fused schedule proves bounded
   lifetime and enough consumer reuse.
4. Materialize to Tier 2 cacheable DRAM at explicit fallback boundaries.
5. Compare against cache-only and against single-kernel SPM promotion.

Gate B is complete only when the compiler can explain both the promoted fused
case and the fallback unfused case.  A hand-only fused schedule is useful for a
prototype, but the paper story needs the promotion decision and fallback policy
to be visible in compiler output.

### Gate C: End-To-End Evaluation

Goal: turn the mechanism into a paper-quality system result.

Steps:

1. Run the single-kernel suite with ablations: cache-only, streaming SPM,
   row/block-resident promotion, Tier 2/3 backing placement, and fused promotion.
2. Run at least one transformer-facing end-to-end or near end-to-end experiment.
3. Report not only speedups, but also why the policy avoids slowdowns:
   rejected candidates, descriptor counts, wait/fence counts, and cache fallback
   cases.
4. Include L2-warming/Tier 2 evidence as a CPU-specific result, not as a GPU
   shared-memory copy.
5. Prepare artifact scripts so `make verify`, selected compares, and figure
   generation are reproducible.

Gate C is complete when the evaluation supports the main claim: GPU-style SPM
promotion needs to be redefined for cache-rich CPUs as a combination of backing
placement, explicit residency, and conservative profitability gating.

### Paper Gate

The likely paper shape is:

- **Problem**: blindly lowering tiled loads into CPU SPM can lose badly because
  cache, scalar/vector reuse, DMA descriptor overhead, MMIO stores, and fences
  change the shared-memory cost model.
- **Design**: separate backing placement from explicit SPM promotion; keep cache
  as the default fallback; promote only when bounded reuse is proven.
- **Implementation**: TriSPM compiler passes, tier sidecar, DMA lowering,
  promotion records, row/block residency, and fusion promotion.
- **Evaluation**: single-kernel wins and rejections, multi-kernel/fusion wins,
  end-to-end transformer-facing result, and ablations.

This is not ready after Gate A alone.  Gate A is enough for a strong
single-kernel story; Gate B makes it a compiler/scheduling contribution; Gate C
is the minimum target for a CGO-style submission.

Submission target:

- CGO 2027 Round 1 paper submission is 2026-06-11 AoE.
- Round 1 should be treated as the aggressive target for a complete Gate C
  story.  The daily push plan should prioritize runnable evidence over perfect
  generality.
- If Gate B is incomplete or the end-to-end result is weak by the internal
  paper-freeze point, target Round 2 on 2026-09-10 rather than submitting a
  single-kernel-only paper with an overclaimed story.

Suggested Round 1 sprint:

1. **Week 1**: promotion records, evidence export, and no-regression matmul
   validation are complete as of 2026-05-02. Keep this gate closed unless the
   schema or fused matmul lowering changes again.
2. **Week 2**: LayerNorm row-resident promotion is implemented and measured as
   of 2026-05-03. It must remain opt-in because the required 32x64 and 512x1024
   comparisons are slower than cache.
3. **Week 3**: softmax plus cache-only elementwise / residual smoke coverage is
   already landed; use this week to connect those fixtures to the profitability
   gate and any row/block-resident reduction prototype before claiming Gate A.
4. **Week 4**: build the first fusion prototype, preferably `layer_norm + qkv`,
   with explicit SPM lifetime and Tier 2 fallback.
5. **Week 5**: run end-to-end or near end-to-end transformer-facing evaluation,
   ablations, and artifact scripts.
6. **Final days**: freeze experiments, write the paper, and keep only bug fixes
   or figure-regeneration changes.

Daily rule: every day should either land runnable compiler/workload support,
new measurement evidence, or paper/artifact text.  Avoid large speculative
abstractions that cannot feed one of the three gates before the paper freeze.
