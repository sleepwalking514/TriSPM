# L2-Warming Side Effect — Verification & Results

## Purpose

DMA reads from **cacheable** addresses cause L2Cache to allocate a line for
the transferred data. Subsequent CPU scalar accesses to the same address hit
L2 rather than miss to DRAM. This is the core justification for the **Tier 2**
placement policy (cacheable + DMA tiling) in the three-tier design.

This document records:
1. **Section A** — gem5 source-level evidence that the mechanism exists.
2. **Section B** — microbenchmark design and experimental results.

---

## A. Source-Level Verification

### A.1 DMA engine issues cacheable `ReadReq`

`simulator/src/scratchpad_mem/spm_dma_engine.cc` — `beginRead` (1D) and
`beginPipelinedRead` (2D) both call:

```
dmaPort.dmaAction(MemCmd::ReadReq, addr, len, ..., buffer, 0);
```

The final `0` is `Request::Flags flag=0`. No `Request::UNCACHEABLE` bit is
set, so every cache in the hierarchy treats the request as a normal cacheable
read.

### A.2 DMA port connects to L2XBar

`run_spm.py` line 133: `self.spm_dma.dma = self.l2bus.cpu_side_ports`

The DMA engine's port connects to the L2 crossbar's CPU-side. DMA reads and
writes traverse L2 just like CPU accesses.

### A.3 L2 allocates on fill for `ReadReq`

`src/mem/cache/base.hh:443-451` — `allocOnFill()` returns true for
`MemCmd::ReadReq`. When a DMA read misses L2 and the DRAM response returns,
`handleFill()` allocates a new L2 block and copies the data in.

### A.4 Uncacheable path bypasses L2

For addresses in the UC DMA buffer range, `findPort` resolves to `uc_bridge`
(not `l2cache.cpu_side`). L2Cache is never accessed for these addresses.

### Verdict

The L2-warming mechanism is **faithful to the gem5 source**. DMA from
cacheable addresses warms L2; DMA from UC addresses does not.

---

## B. Microbenchmark Results

### B.1 Design

Workload: `workloads/kernels/dma_l2_warming/`

Four phases, each bracketed by `m5_reset_stats` / `m5_dump_stats`:

| Phase | Operation | Expected |
|-------|-----------|----------|
| A | DMA cacheable buffer → SPM | L2 misses (allocate-on-fill) |
| B | Random read of same buffer | L2 hits (warmed by DMA) |
| C | DMA from UC buffer → SPM | Zero L2 accesses |
| D | Random read of cold buffer | L2 misses (never warmed) |

Key design choices:
- **Random access pattern** (Fisher-Yates shuffle) defeats the L2 stride
  prefetcher, which otherwise hides DRAM latency and masks the warming effect.
- **Throttled flush** (4 MiB scrub in 8 KB chunks with fences) evicts all
  working-set lines from L2 before Phase D measurement.
- **Same permutation** for B and D ensures identical access patterns.

### B.2 Results — Working-Set Sweep (2026-04-30)

| BUF_BYTES | Lines | Phase B hits | Phase B misses | Phase D hits | Phase D misses | B speedup vs D |
|-----------|-------|-------------|----------------|-------------|----------------|----------------|
| 4 KiB | 64 | 66 | 5 | 2 | 69 | 2.74× |
| 8 KiB | 128 | 134 | 5 | 6 | 133 | 2.76× |
| 16 KiB | 256 | 270 | 5 | 14 | 261 | 2.80× |
| 32 KiB | 512 | 542 | 4 | 30 | 516 | 2.80× |

Cycle counts (Phase B / Phase D):

| BUF_BYTES | Phase B cycles | Phase D cycles | Ratio |
|-----------|---------------|---------------|-------|
| 4 KiB | 2,515 | 6,898 | 2.74× |
| 8 KiB | 4,747 | 13,093 | 2.76× |
| 16 KiB | 9,181 | 25,707 | 2.80× |
| 32 KiB | 18,119 | 50,755 | 2.80× |

### B.3 Analysis

**L2-warming confirmed at all tile sizes.** Phase B (post-DMA random read)
achieves near-100% L2 hit rate, while Phase D (cold random read) shows
near-100% miss rate. The hit count difference matches the expected number of
cache lines exactly (64, 128, 256, 512 for the four sizes).

The consistent **2.8× cycle speedup** from L2-warming demonstrates that
Tier 2 placement (cacheable + DMA) provides a substantial performance
advantage over cold-cache access for subsequent scalar reads.

Phase C (DMA from UC buffer) shows zero or near-zero L2 accesses, confirming
that the uncacheable path truly bypasses L2.

### B.4 Implications for Tier 2 Policy

1. **"SPM never worse than cache" floor guarantee holds**: after DMA prefetch
   of a tile, the CPU's scalar epilogue (e.g., reduction accumulation, store-
   back) hits L2 rather than DRAM. This is strictly better than a pure-cache
   baseline that would miss on first access.

2. **Tile sizes up to 32 KiB** (512 cache lines) are validated. Larger tiles
   exceed the gem5 O3 CPU's in-flight request capacity for random access
   patterns, but real kernel access patterns are sequential (where the
   prefetcher provides similar benefit regardless of warming state).

3. **The warming effect is most valuable for non-sequential access patterns**
   (reductions, scatter/gather, indirect indexing) where the prefetcher cannot
   help.

### B.5 Limitations

- Sweep limited to ≤ 32 KiB due to gem5 L2XBar routing table capacity (512
  entries). Larger random-access working sets overflow the crossbar. Real
  kernels use sequential access which doesn't trigger this limitation.
- The benchmark uses `fence r, r` between random reads to prevent crossbar
  overflow, which serializes loads. Real workloads benefit from ILP that
  partially hides L2 hit latency.

---

## C. Completed Tasks

- [x] **T1** — Create `workloads/kernels/dma_l2_warming/` harness
- [x] **T2** — m5ops wired up via inline asm in `libspm.h`
- [x] **T3** — Multiple `dump_stats` produce multiple stats blocks (no config change needed)
- [x] **T4** — `parse_l2_warming.py` extracts per-checkpoint L2 counters
- [x] **T5** — Run §B.2 with default config → L2-warming confirmed
- [x] **T6** — Working-set sweep (4K–32K) → consistent 2.8× speedup
