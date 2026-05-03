# TriSPM Docs Map

This directory is the navigation layer for the TriSPM project notes. It should answer three questions quickly:

- What is the project trying to prove?
- Which documents are completed records, and which ones still drive work?
- Where are we on the timeline: past, current, and next?

## Current Position

As of 2026-05-03, Phase 3 matmul has crossed over under the cold-start headline metric: SPM beats the cache baseline on the current large runs, and the small 64-case is no longer a regression point. Do not quote a fixed headline number here yet; the best SPM and cache blocking choices differ, so the final fair comparison should come from a blocking sweep rather than a single stale point.

The Phase 3 compiler robustness line is closed for the current matmul/layer_norm/vector_add coverage: `layer_norm` can still opt into mean, variance, and final normalize through SPM, including the multi-load reduction/streaming matcher, and GEMM/reduction bail-out cleanup is implemented and covered by lit tests. However, the stronger "single-kernel SPM policy is converged" claim is no longer accurate for reductions. A 2026-05-03 LayerNorm re-audit found that the old row-resident result was dominated by serialized per-row DMA/wait overhead; a fill-on-first-pass prototype removes that artifact and collapses the gap to near parity. Transformer-facing single-kernel workload coverage has also landed for cache-path `activation` (SiLU), `residual_add`, and row-wise `softmax`; all three have manifests, AOT launchers, flushed ROI harnesses, result checks, `make verify-<kernel>` policy coverage, and gem5 smoke compares in both SPM-enabled and cache-baseline modes. The first graph-level conservative placement build/verify MVP is also in place; executable graph harnesses and attention/fusion work remain open.

Important performance caveat: the old streaming reduction SPM path is still only correctness/coverage. It issues many tiny DMA transactions and remains far behind cache even with Tier 2 cacheable inputs. The row-resident LayerNorm path is different: the original DMA-row prototype lost badly (32x64: 10,279 vs 6,138 cycles; 512x1024: 1,245,422 vs 1,012,922 cycles), but the fill-on-first-pass prototype writes each first-pass `x` chunk into SPM and reuses it for variance/normalize. That removes row DMA descriptors and waits, producing 32x64 at 6,150 vs 5,892 cycles (+4.4%) and 512x1024 at 1,015,917 vs 1,011,115 cycles (+0.5%). Default `layer_norm` still verifies as cache path until this becomes a clear win, but reduction SPM should be treated as an active optimization line, not a closed failure.

For end-to-end transformer work, the placement rule is now conservative and graph-aware: keep intermediate activations and kernel outputs cacheable by default, use Tier 3 uncacheable only for external read-only DMA-only streaming inputs/weights, and reserve Tier 1 for future small SPM-resident hot state. See `plans/three-tier-placement.md` §2.1.

The first graph-level placement MVP is an upper-layer build/verify tool, not a
compiler promotion change: `workloads/scripts/graph_placement.py` reads a graph
manifest, applies conservative tensor-edge rules, builds per-node SPM artifacts
with `KERNEL_TIER_OVERRIDE`, and verifies the generated launcher allocation
dispatch. It proves that intermediate activations stay Tier 2 while external
DMA-only weights can be Tier 3. It does not yet link or run a multi-kernel graph,
implement Tier 1, or perform fused SPM promotion.

Explicit SPM promotion D1 is now closed. The existing fused matmul scheduler
reports B-window, A-micro, and accumulator promotion records without changing
generated code, and the D1 sidecar now has a versioned debug/evidence schema
with accepted/rejected status, structural rejection reason codes, exact vs
estimated field annotations, accepted matmul coverage, reduction/cache-path
rejection coverage, and report-off coverage. During D1 validation we found and
fixed an unrelated DMA fence codegen regression: a generic inline-asm memory
clobber had increased reload pressure and moved 256x256x256 matmul from the
archived ~1.729M-cycle baseline to ~1.987M cycles. Removing the clobber
restored the baseline while keeping the real `fence iorw, iorw` instruction.
D1b no-regression validation kept 64x64x64 SPM-only correctness passing and
256x256x256 SPM-only at 1,729,209 cycles / 5913 assembly lines. Promotion
records are still debug/evidence output, not yet a scheduler or profitability
planner.

Explicit SPM promotion D2 and D3 are landed as opt-in infrastructure, but their
reduction policy is not settled. D2 now uses a fill-on-first-pass row-resident
LayerNorm prototype (`TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS=1`): the mean
pass reads `x` from the original cache/DRAM path and writes the loaded chunks
into SPM, then variance and normalize reuse the SPM row while `gamma` / `beta`
stay cached. D3 records static descriptor/MMIO/wait/fence/byte/use evidence in
the existing D1 debug sidecar; after the fill-on-first-pass change, large rows
can be accepted as opt-in evidence while small rows still reject as
`insufficient_row_work`. The sidecar is still evidence only; it is not a
planner, scheduler, placement, layout, `windowK`, or profitability interface.

Phase 3 matmul and compiler robustness are converged for the current scope, but
single-kernel SPM policy is not fully converged for reductions. Matmul is still
the mature SPM performance path and low-reuse elementwise kernels stay cache
path. Reduction SPM needs another measured optimization pass: the latest
LayerNorm data shows the old cache-favorable conclusion was partly caused by a
serialized DMA implementation artifact, not by an inherent SPM limitation.
The next compiler gate is Phase 3.5 single-kernel convergence: finish the
reduction row/block-resident policy before treating Phase 4 graph/fusion as the
main line. Phase 4/5/6 graph, attention/fusion, producer-consumer promotion, and
broader evaluation remain open on top of that corrected single-kernel baseline.

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
| Current | Next compiler gate | [`plans/phase3.5-single-kernel-convergence.md`](plans/phase3.5-single-kernel-convergence.md) | Close reduction single-kernel SPM first: LayerNorm break-even/win, Softmax row/block-resident evidence, and refit D3 profitability. |
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
