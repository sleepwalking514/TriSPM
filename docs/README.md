# TriSPM Docs Map

This directory is the navigation layer for the TriSPM project notes. It should answer three questions quickly:

- What is the project trying to prove?
- Which documents are completed records, and which ones still drive work?
- Where are we on the timeline: past, current, and next?

## Current Position

As of 2026-04-30, Phase 3 matmul is closed under the cold-start headline metric. The large 1024x1024x1024 / 32x32x32 run beats cache by 25.1% (SPM 288,976,339 cycles vs cache 386,049,495), while the 64x64 smoke case stays within the <= cache x 1.05 guardrail.

The current active line is reduction-path performance and compiler robustness: make `transformReductionLoop` truly double-buffered, then broaden the matchers and pass options needed for more workloads and Phase 4.

## Document Inventory

| Status | File | What it is for |
| --- | --- | --- |
| Current | [`plans/phase3.md`](plans/phase3.md) | Phase 3 status page: what is done, what is current, and what is explicitly out of scope. Start here for the live compiler state. |
| Current | [`plans/phase3-execution-timeline.md`](plans/phase3-execution-timeline.md) | Ordered execution timeline for Phase 3. Shows completed stages, current stage, and next stages. |
| Current | [`plans/phase3-compiler-backlog.md`](plans/phase3-compiler-backlog.md) | Remaining compiler robustness backlog: GEMM matcher, reduction matcher, DMA lowering options, and deferred writeback traceability. |
| Current | [`plans/three-tier-placement.md`](plans/three-tier-placement.md) | Three-tier placement design and MVP state: Tier 2/3 plumbing landed, Tier 1 and reuse-rule expansion deferred. |
| Current | [`plans/spm-dma-reuse.md`](plans/spm-dma-reuse.md) | Active follow-on plan for fused micro-scheduler DMA reuse after SplitLargeContract exposed repeated B/A DMA costs. |
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
| Past | Done | [`plans/three-tier-placement.md`](plans/three-tier-placement.md) M1-M8 | Tier 2/3 placement plumbing landed; coverage audit says only `matmul` currently enters the SPM path. |
| Past | Done | [`evidence/l2_warming.md`](evidence/l2_warming.md) | Tier 2 L2-warming claim has source-level and microbenchmark evidence. |
| Current | In progress | [`plans/phase3-execution-timeline.md`](plans/phase3-execution-timeline.md) Stage 4.5 | Upgrade reduction lowering from single-buffer prefetch to true double-buffer pipelining. |
| Next | Planned | [`plans/phase3-compiler-backlog.md`](plans/phase3-compiler-backlog.md) P1 | Robust GEMM matcher -> generalized reduction matcher -> `DmaOpsToLLVM` MMIO base / `useXspmInsn` options. |
| Next | Planned | [`plans/spm-dma-reuse.md`](plans/spm-dma-reuse.md) | Replace repeated full-tile DMA after SplitLargeContract with a fused microM-aware scheduler. |
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
