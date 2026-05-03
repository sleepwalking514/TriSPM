# TriSPM Docs Map

This directory is the navigation layer for the TriSPM project notes. It should answer three questions quickly:

- What is the project trying to prove?
- Which documents are completed records, and which ones still drive work?
- Where are we on the timeline: past, current, and next?

## Current Position

As of 2026-05-03, Phase 3 matmul and compiler robustness are closed for the
current scope.  Matmul remains the mature SPM performance path, while
low-reuse elementwise kernels stay on the cache path unless future fusion makes
SPM residency profitable.  The graph-level conservative placement MVP is also
landed: intermediate activations default to cacheable Tier 2, selective external
streaming weights may use Tier 3, and Tier 1/fusion remain later work.

The active compiler gate is Phase 3.5: single-kernel reduction SPM convergence.
The old streaming reduction SPM path is correctness/coverage only; it still
loses badly because it emits many tiny DMA transactions.  The newer LayerNorm
row-resident path is different: it uses CPU-direct SPM residency.  The first
pass reads `x` from DRAM/cache and writes chunks into SPM with `addrspace(3)`
stores; later passes read those chunks back from SPM with `addrspace(3)` loads.
There are no DMA descriptors, waits, or fences on that path.

Phase 3.5 P0 baseline tooling is in place.  LayerNorm and Softmax have named
Phase 3.5 presets, compare runs emit CSV plus build-artifact marker counts, and
`workloads/scripts/phase35_baseline.sh` provides verify/smoke/full suites.  The
full baseline showed:

- Default LayerNorm remains cache path: `phase35-small` and `phase35-large`
  have no SPM markers and are effectively noise-level comparisons.
- Opt-in LayerNorm CPU-direct row residency really uses SPM but still does not
  win: 32x64 is +13.6%, and 512x1024 is +0.5%.
- D3 correctly rejects small LayerNorm rows as `insufficient_row_work`; it can
  accept large rows as opt-in evidence, but the measured result is still +0.3%,
  so reduction promotion is not default-enabled.
- Softmax smoke and large-row runs are cache-path baselines only.  They have no
  `addrspace(3)` and identical instruction counts; their small deltas are
  runtime/cache noise, not SPM wins.

Phase 3.5 P1a is now landed.  The reduction pass has an internal
`ReductionResidencyPlan` shape, LayerNorm still lowers through the same
CPU-direct row-resident schedule, and promotion sidecars report producer pass,
consumer passes, buffer role, rotation policy, copy-in mode, SPM slots, and
expected markers.  Softmax large-row detection now emits a rejected
`Softmax x row` plan with `unsupported_reduction_residency_plan`; Softmax IR
stays cache path until a measured lowering is added.  Next work is P2:
reduce fill-on-first-pass overhead enough to justify any default reduction SPM
promotion.

## Document Inventory

| Status | File | What it is for |
| --- | --- | --- |
| Current | [`plans/phase3.md`](plans/phase3.md) | Phase 3 status page: what is done, what is current, what still blocks Phase 4, and what is explicitly out of scope. Start here for the live compiler state. |
| Current | [`plans/phase3.5-single-kernel-convergence.md`](plans/phase3.5-single-kernel-convergence.md) | Active Phase 3.5 plan: close single-kernel reduction SPM using admission, lifetime, explicit buffer rotation, and measured row/block-resident profitability before Phase 4. |
| Current | [`plans/phase3-execution-timeline.md`](plans/phase3-execution-timeline.md) | Ordered execution timeline for Phase 3. Shows completed stages, current stage, and next stages. |
| Current | [`plans/phase3-compiler-backlog.md`](plans/phase3-compiler-backlog.md) | Phase 3 compiler audit/closure record: GEMM/reduction robustness is done for the current coverage; output-tile SPM writeback is deferred to Phase 4b. |
| Current | [`plans/three-tier-placement.md`](plans/three-tier-placement.md) | Three-tier placement design and MVP state: Tier 2/3 plumbing landed; graph-level conservative placement build/verify MVP landed; executable graph harness remains P0. |
| Current | [`plans/spm-explicit-promotion.md`](plans/spm-explicit-promotion.md) | Explicit promotion record: D1 evidence, D2 opt-in row-resident LayerNorm, and D3 conservative profitability/rejection policy are landed for current single-kernel coverage. |
| Current | [`plans/spm-dma-reuse.md`](plans/spm-dma-reuse.md) | Follow-on fused micro-scheduler DMA reuse plan and first correctness implementation after SplitLargeContract exposed repeated B/A DMA costs. |
| Roadmap | [`plans/compiler-roadmap.md`](plans/compiler-roadmap.md) | Full Phase 1-6 compiler roadmap: foundation, Phase 3, attention/multi-kernel work, end-to-end inference, and evaluation. |
| Evidence | [`evidence/l2_warming.md`](evidence/l2_warming.md) | Completed Tier 2 L2-warming evidence: source verification plus `dma_l2_warming` microbenchmark, 4K-32K sweep, and 2.8x speedup result. |
| Architecture | [`architecture/simulator-spm-architecture.md`](architecture/simulator-spm-architecture.md) | Simulator-side SPM/DMA/Xspm/O3-LSQ architecture map, code entry points, and paper-facing design notes. |
| Architecture | [`architecture/graph.html`](architecture/graph.html) | Architecture graph asset moved out of archive because it describes current structure rather than an obsolete plan. |
| Archived result | [`archive/matmul-spm-lowering-closure.md`](archive/matmul-spm-lowering-closure.md) | Completed matmul lowering record: codegen fix, fair baseline, MMIO packing, size/steady sweep, and final P3 headline. |
| Archived result | [`archive/split-large-contract.md`](archive/split-large-contract.md) | Completed SplitLargeContract design and measurement record for large-tile register micro-tiling. |
| Superseded | [`archive/phase2.md`](archive/phase2.md) | Old Phase 2 closeout note. Its live leftovers moved into Phase 3 backlog and three-tier placement docs. |

## Timeline

| Time | Status | Main document path | Outcome / next edge |
| --- | --- | --- | --- |
| Past | Done | [`archive/phase2.md`](archive/phase2.md) | Phase 2 leftovers split into [`plans/phase3-compiler-backlog.md`](plans/phase3-compiler-backlog.md) and [`plans/three-tier-placement.md`](plans/three-tier-placement.md). |
| Past | Done | [`archive/matmul-spm-lowering-closure.md`](archive/matmul-spm-lowering-closure.md) P0-P3 | Matmul SPM lowering preserves RVV code shape, uses fair cold-cache measurement, and closes the Phase 3 matmul headline. |
| Past | Done | [`archive/split-large-contract.md`](archive/split-large-contract.md) | Large-tile register pressure fixed; follow-on DMA reuse moves to [`plans/spm-dma-reuse.md`](plans/spm-dma-reuse.md). |
| Past | Done | [`plans/three-tier-placement.md`](plans/three-tier-placement.md) M1-M8 | Tier 2/3 placement plumbing landed; `matmul` enters the SPM path, while `layer_norm` and `vector_add` default to cache path. LayerNorm reduction SPM remains opt-in coverage. |
| Past | Done | [`evidence/l2_warming.md`](evidence/l2_warming.md) | Tier 2 L2-warming claim has source-level and microbenchmark evidence. |
| Past | Done | [`plans/phase3-compiler-backlog.md`](plans/phase3-compiler-backlog.md) P1 | GEMM extra-load matching, reduction multi-load matching, DMA lowering options, and GEMM/reduction bail-out cleanup are complete for the current coverage. |
| Past | Done | [`plans/phase3.md`](plans/phase3.md) + [`plans/compiler-roadmap.md`](plans/compiler-roadmap.md) Phase 6c | Transformer-facing single-kernel coverage landed for `activation`, `residual_add`, and `softmax`; each builds/verifies as cache path and has flushed ROI smoke compares. |
| Current | Done MVP | [`plans/three-tier-placement.md`](plans/three-tier-placement.md) §2.1 / §6.2 | Graph-level conservative placement planner landed for build/verify: cacheable activation backbone, selective UC streaming inputs/weights, and explicit Tier 1/fusion non-goals. |
| Current | Active prototype | [`plans/spm-explicit-promotion.md`](plans/spm-explicit-promotion.md) D2 | Opt-in row-resident LayerNorm promotion now uses fill-on-first-pass SPM materialization. It validates separately from both default cache path and old streaming reduction coverage, and it has improved from large regressions to near parity. |
| Current | Active gate | [`plans/spm-explicit-promotion.md`](plans/spm-explicit-promotion.md) D3 | Conservative profitability evidence landed: accepts existing fused matmul evidence, rejects streaming reductions and small row-resident reductions, and can accept large fill-on-first-pass row-resident evidence while default LayerNorm remains cache path. |
| Current | Active compiler gate | [`plans/phase3.5-single-kernel-convergence.md`](plans/phase3.5-single-kernel-convergence.md) | P1a plan extraction is landed; next is P2 overhead reduction and measured LayerNorm/Softmax row-resident profitability before default promotion. |
| Later | Planned | [`plans/compiler-roadmap.md`](plans/compiler-roadmap.md) Phase 4/5 + [`plans/spm-explicit-promotion.md`](plans/spm-explicit-promotion.md) Gate B | Move to executable attention/fusion and producer-consumer promotion after Phase 3.5 reduction closure. |
| Current | Active optimization | [`plans/spm-dma-reuse.md`](plans/spm-dma-reuse.md) | First fused microM-aware scheduler implementation exists; continue correctness/performance tuning and larger-run evaluation. |
| Later | Planned | [`plans/three-tier-placement.md`](plans/three-tier-placement.md) §6.1 -> [`plans/compiler-roadmap.md`](plans/compiler-roadmap.md) Phase 4/5 | Tier 1 resident SPM, attention/multi-kernel SPM management, then end-to-end transformer inference. |
| Later | Planned | [`plans/compiler-roadmap.md`](plans/compiler-roadmap.md) Phase 6 | Paper evaluation: cache baseline, workload coverage, breakdowns, area-equivalent comparison, and sensitivity analysis. |

## Directory Rules

| Directory | Use it for |
| --- | --- |
| [`plans/`](plans/) | Active plans, roadmaps, current status pages, and still-actionable backlog. File names should describe the owned topic, not generic placeholders. |
| [`evidence/`](evidence/) | Verification notes, experiment design, measured data, sweep results, and paper-claim support. |
| [`architecture/`](architecture/) | Long-lived system architecture notes and diagrams that are useful for future agents or paper writing. |
| [`archive/`](archive/) | Superseded notes and completed result logs that should not drive current work but preserve decisions and measurements. |

When moving a document, update this README and all references under `docs/`. Archived files should start with a short status block saying why they are archived and where the live follow-up lives.
