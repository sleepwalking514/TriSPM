# Simulator SPM Architecture Notes

This document summarizes the simulator-side changes in the `simulator`
submodule, comparing `stable..spm-dev`. It is written as a quick map for new
AI agents and as reusable architecture material for paper writing.

The main simulator contribution is a gem5-based scratchpad memory (SPM)
system for RISC-V SE-mode experiments. The branch adds an explicitly managed
on-chip SRAM model, a DMA engine, RISC-V custom DMA instructions, an O3 CPU
direct SPM port, and workload/run scripts for SPM-vs-cache experiments.

## Quick Map For New Agents

Start with these files:

| File | Why it matters |
| --- | --- |
| `simulator/src/scratchpad_mem/ScratchpadMemory.py` | Python SimObject parameters for the SPM SRAM model. |
| `simulator/src/scratchpad_mem/scratchpad_memory.hh/.cc` | Dual-port, multi-bank scratchpad implementation and stats. |
| `simulator/src/scratchpad_mem/SpmDmaEngine.py` | Python SimObject parameters for the DMA engine. |
| `simulator/src/scratchpad_mem/spm_dma_engine.hh/.cc` | MMIO and ISA-facing DMA engine, descriptor queue, 1D/2D transfers. |
| `simulator/src/scratchpad_mem/spm_dma_iface.hh` | Free-function interface used by RISC-V custom instructions. |
| `simulator/src/scratchpad_mem/run_spm.py` | Current SE-mode system topology for cache baseline and SPM runs. |
| `simulator/src/scratchpad_mem/libspm.h` | User-space SPM allocation, DMA APIs, Xspm inline assembly, m5ops helpers. |
| `simulator/src/arch/riscv/isa/formats/spm.isa` | RISC-V Xspm instruction formats and execution semantics. |
| `simulator/src/arch/riscv/isa/decoder.isa` | Custom-0 opcode decode for Xspm instructions. |
| `simulator/src/cpu/o3/lsq.hh/.cc` and `lsq_unit.hh/.cc` | O3 LSQ routing of SPM physical addresses to the direct SPM port. |
| `simulator/src/arch/riscv/tlb.cc` | SE-mode uncacheable flag propagation fix. |

The current run entry point is `simulator/src/scratchpad_mem/run_spm.py`.
`run_spm_v1.py`, `run_spm_v2.py`, and `run_spm_v3.py` are preserved older
topology snapshots.

## What Changed Relative To `stable`

The diff is concentrated in four areas:

1. New `src/scratchpad_mem/` module.
   - Adds `ScratchpadMemory`, `SpmDmaEngine`, debug flags, user-space
     `libspm.h`, run scripts, and C test programs.
   - Adds GEMM, 2D DMA, step-by-step DMA, and Xspm instruction tests.

2. RISC-V ISA extension.
   - Adds `src/arch/riscv/insts/spm.hh/.cc`.
   - Adds `src/arch/riscv/isa/formats/spm.isa`.
   - Extends `decoder.isa`, `formats.isa`, `includes.isa`, and the RISC-V
     instruction `SConscript`.

3. O3 CPU memory-system changes.
   - Adds `BaseO3CPU.spm_port`, `spmAddrStart`, and `spmAddrSize`.
   - Extends O3 CPU `getPort()` so `spm_port` connects to the LSQ.
   - Extends the LSQ/LSQUnit to route SPM addresses to a dedicated
     non-snooping request port instead of the normal dcache port.

4. RISC-V SE-mode TLB fix.
   - Changes SE translation to use `pTable->lookup()` so
     `process.map(cacheable=False)` propagates `UNCACHEABLE` and
     `STRICT_ORDER` request flags.

No simulator source files for gem5's RVV/vector instruction implementation are
added by this branch relative to `stable`. RVV appears as a workload/codegen
caveat: `spm_gemmX.c` notes that auto-vectorized init code can generate RVV
micro-ops that crash gem5 O3 when touching uncacheable DMA-buffer memory, so
the init helpers are compiled at `O1`. The simulator ISA added here is the
Xspm custom DMA extension.

## System Topology

`run_spm.py` creates either a pure cache baseline or an SPM-enabled SE-mode
system.

In cache-baseline mode:

```text
O3CPU
  icache_port -> L1I -> L2XBar -> L2 -> SystemXBar -> DDR3
  dcache_port -> L1D -> L2XBar -> L2 -> SystemXBar -> DDR3
  PTW caches  -> L2XBar -> L2 -> SystemXBar -> DDR3
```

In SPM mode, normal memory still uses the cache hierarchy, but SPM traffic has
two additional paths:

```text
CPU SPM loads/stores:
  O3CPU.spm_port -> ScratchpadMemory.cpu_port

DMA and MMIO traffic:
  SpmDmaEngine.dma -> L2XBar -> ScratchpadMemory.port or DRAM
  SpmDmaEngine.pio <- L2XBar MMIO requests

Uncacheable DMA-buffer DRAM:
  L2XBar -> Bridge -> SystemXBar -> DDR3
```

The important architectural split is:

- `ScratchpadMemory.cpu_port`: tightly coupled CPU data path.
- `ScratchpadMemory.port`: interconnect/DMA-side path.

Both ports share the same SPM backing store. The split lets CPU SPM accesses
avoid the dcache path while still allowing the DMA engine to fill or drain SPM
through the memory interconnect.

### Address Map In `run_spm.py`

When SPM is enabled:

| Region | Address/size | Mapping behavior |
| --- | --- | --- |
| SPM | base `0x40000000`, size from `--spm_size` | `process.map(..., cacheable=False)`; LSQ later clears strict ordering only for SPM. |
| DMA MMIO | base `0xF0000000`, mapped window `0x10000`, active PIO size `0x40` | Uncacheable and strictly ordered. |
| DMA buffer | base `SPM_BASE - 512MiB`, currently `0x20000000`, size `512MiB` | Uncacheable DRAM staging area routed through the bridge. |
| Cacheable DRAM | lower `512MiB` cacheable range in SPM mode | Normal L1/L2 path. |

`run_spm.py` also passes these environment variables to the workload:

- `SPM_SIZE_BYTES`
- `DMA_BUF_BASE`
- `DMA_BUF_SIZE`

`libspm.h` has fallback values for these, but normal experiments should rely
on the environment from `run_spm.py`.

## ScratchpadMemory

`ScratchpadMemory` is an `AbstractMemory` subclass that models a multi-bank,
dual-port SRAM.

Key behavior:

- Two response ports: `port` for bus/DMA traffic and `cpu_port` for direct CPU
  traffic.
- Shared backing store, so both ports see the same bytes.
- Bank selection uses `(addr - range.start()) / bank_interleave_size %
  num_banks`.
- Each bank tracks busy time independently for the bus port and CPU port.
  CPU-vs-DMA access to the same bank is allowed in the same cycle; conflicts
  are counted only when the same port re-accesses a bank before that port's
  prior access completes.
- Timing uses configurable base latency, latency variance, bandwidth, bank
  busy time, and a shared response queue tagged with the originating port.
- Drain waits for the response queue to empty.

Main parameters:

| Parameter | Meaning |
| --- | --- |
| `latency` | Per-bank access latency. |
| `latency_var` | Optional randomized latency variation. |
| `bandwidth` | Port bandwidth limit. |
| `num_banks` | SRAM bank count. |
| `bank_interleave_size` | Address interleave granularity. |

Main stats:

- total reads/writes
- cumulative and average read/write latency
- total bank conflicts
- per-bank reads, writes, and conflicts

## SpmDmaEngine

`SpmDmaEngine` is a `ClockedObject` with:

- `pio`: response port for MMIO register access
- `dma`: request port for DMA reads/writes
- per-System registry used by Xspm instruction helpers
- a descriptor queue, default depth 4
- 1D contiguous DMA support
- 2D strided DMA support
- polling status semantics for waits
- stats for transfers, bytes, busy cycles, queue-full stalls, and wait polling

### MMIO Register Interface

The PIO base is `0xF0000000` in `run_spm.py`.

| Offset | Register | Behavior |
| --- | --- | --- |
| `0x00` | `SRC` | Staged source address. |
| `0x08` | `DST` | Staged destination address. |
| `0x10` | `LEN` | Writing enqueues a transfer. Low 32 bits are bytes/row. Upper 32 bits can override `HEIGHT`. |
| `0x18` | `STATUS` | Read returns pending descriptor count plus active transfer. `0` means idle. |
| `0x20` | `SRC_STRIDE` | Staged source row pitch for 2D copies. |
| `0x28` | `DST_STRIDE` | Staged destination row pitch for 2D copies. |
| `0x30` | `HEIGHT` | Staged row count for 2D copies. |
| `0x38` | `STRIDES_PACKED` | Lower 32 bits are source stride, upper 32 bits are destination stride. |

The packed registers keep the legacy interface functional while allowing the
compiler to reduce descriptor MMIO traffic.

### Transfer Semantics

1D transfer:

```text
write SRC
write DST
write LEN              # enqueues len bytes
poll STATUS until 0
```

2D transfer:

```text
write SRC
write DST
write SRC_STRIDE / DST_STRIDE, or STRIDES_PACKED
write HEIGHT, or put height in LEN[63:32]
write LEN[31:0]         # enqueues width bytes per row
poll STATUS until 0
```

For 2D transfers, the engine allocates one row buffer per row, issues row
reads in parallel through `DmaPort`, and starts each row write as soon as that
row's read finishes. This models pipelined row transfers rather than a purely
serial row loop.

The engine treats workload addresses as SE-mode virtual addresses and
translates them through the process page table. DMA actions are split at
4 KiB virtual-page boundaries because consecutive virtual pages are not
guaranteed to map to consecutive physical pages in gem5 SE mode.

## Xspm RISC-V Custom Instructions

The branch uses RISC-V custom-0 opcode space, full opcode `0x0B`.

| Instruction | Encoding role | Semantics |
| --- | --- | --- |
| `spm.dma rd, rs1, rs2` | R-type, `funct3=0` | Enqueue a 1D copy from `rs1` source to `rd` destination with `rs2` bytes. |
| `spm.dma.w rd` | I-type, `funct3=1` | Poll DMA `STATUS`; result is pending count. Compiler/user code loops until zero. |
| `spm.dma.stride rs1, rs2` | R-type, `funct3=2` | Stage source and destination strides for the next 2D DMA. |
| `spm.dma.2d rd, rs1, rs2` | R-type, `funct3=3` | Enqueue a 2D copy. `rd` is destination, `rs1` is source, `rs2` packs `width | height << 32`. |

`spm.dma` operations are marked non-speculative. `spm.dma.w` is implemented as
a load from the DMA `STATUS` register with read/write barrier flags, so it can
be used in a polling loop without forcing every wait to stall at the ROB head.

`libspm.h` exposes inline assembly wrappers when `USE_XSPM_INSN` is defined:

- `xspm_dma()`
- `xspm_dma_wait()`
- `xspm_dma_stride()`
- `xspm_dma_2d()`
- `xspm_dma_copy_2d()`

The MMIO APIs remain available and are useful for hand-written tests and for
compiler paths that do not use Xspm instructions.

## O3 CPU And LSQ Routing

The O3 CPU gains a dedicated `spm_port` plus physical address-window
parameters:

- `spmAddrStart`
- `spmAddrSize`

The LSQ owns a new `SpmPort`. During packet send:

1. The request is translated normally.
2. If the physical address falls inside the configured SPM range, LSQUnit sends
   the packet through `spmPort`.
3. Otherwise the packet follows the existing dcache path.

SPM routing deliberately does not consume normal dcache port availability and
does not set the normal dcache blocked state. It has its own `spmPortBlocked`
state for retry handling.

This matters because SPM is mapped as uncacheable in SE mode. The TLB marks
uncacheable mappings as `UNCACHEABLE | STRICT_ORDER`; that is correct for DMA
MMIO and the uncacheable DMA buffer, but would serialize every SPM load/store
through the O3 pipeline. The LSQ translation-finish path therefore detects SPM
physical addresses and clears `STRICT_ORDER` on both:

- the `Request`
- the `DynInst` strictly-ordered state

Clearing both is required; clearing only one side leaves unnecessary O3
serialization.

## RISC-V SE-Mode TLB Fix

The branch fixes a gem5 RISC-V SE-mode issue where
`process.map(cacheable=False)` did not affect translated requests. The old
path used `pTable->translate()`, which returned only the physical address and
discarded page-table flags.

The new path uses `pTable->lookup()`, sets the physical address manually, and
propagates `EmulationPageTable::Uncacheable` into
`Request::UNCACHEABLE | Request::STRICT_ORDER`.

This fix is essential for the simulator architecture:

- DMA MMIO must be uncacheable and ordered.
- DMA-buffer DRAM must be uncacheable for the Tier-3 path.
- SPM is initially marked uncacheable for routing correctness, then the LSQ
  removes strict ordering only for the SPM direct port.

## User-Space Runtime And Tests

`libspm.h` provides:

- `spm_malloc()`, `spm_memset()`, `spm_free_all()`
- `dma_buf_malloc()`, `dma_buf_free_all()`
- blocking 1D DMA: `spm_dma_copy()`
- blocking 2D DMA: `spm_dma_copy_2d()`
- async 2D enqueue: `spm_dma_enqueue_2d()`
- wait: `spm_dma_wait()`
- cache/measurement helpers: `publish_input()`, `flush_caches()`,
  `m5_reset_stats()`, `m5_dump_stats()`
- Xspm wrappers under `USE_XSPM_INSN`

Important tests and examples:

| File | Purpose |
| --- | --- |
| `test/spm_xinsn_test.c` | Basic Xspm copy and GEMM checks. |
| `test/spm_2d_dma_test.c` | 2D load/store, 2D-vs-1D equivalence, tiled GEMM. |
| `test/spm_gemmX.c` | Block-contiguous GEMM using Xspm 1D DMA and descriptor-queue scheduling. |
| `test/spm_gemm_2d.c` | Row-major GEMM using 2D DMA. |
| `test/spm_step_test.c` | Incremental DMA debugging. |
| `test/cache_gemm.c` | Cache baseline. |

Run helpers:

- `run_spm.py --binary ...`: current topology.
- `run_spm.py --binary ... --cache_baseline`: pure cache baseline.
- `run_gemm.sh [N] [BS]`: builds and runs cache and SPM GEMM variants.
- `run_xinsn_test.sh`: builds and runs the Xspm instruction test.

## Design Points Useful For Paper Writing

The simulator models an explicitly managed memory hierarchy with three
architectural ideas:

1. CPU-direct SPM access.
   - SPM loads/stores bypass the dcache and access the SRAM through a dedicated
     O3 LSQ port.
   - This avoids cache tag/data-array behavior for data the compiler has
     deliberately staged into SPM.

2. DMA-managed data movement.
   - The DMA engine moves tiles between DRAM and SPM using MMIO or Xspm
     instructions.
   - 2D strided descriptors match row-major tensor tiles without requiring
     software to issue one DMA per row.
   - A descriptor queue supports software/compiler scheduling and
     double-buffering.

3. Separate cacheable and uncacheable DRAM paths.
   - Cacheable DRAM supports Tier-2 experiments where DMA may warm L2 for later
     scalar reuse.
   - Uncacheable DMA-buffer DRAM supports Tier-3 experiments where DMA staging
     should avoid cache pollution.

The key simulator mechanism is not merely "add an SPM object"; it is the
combination of:

- SE page mappings that mark special regions uncacheable.
- RISC-V TLB propagation of those flags.
- LSQ routing based on translated physical address.
- Targeted removal of strict ordering only for SPM direct accesses.
- DMA access through the interconnect for fills/drains.

## Known Caveats

- The current Xspm DMA registry is per `System` and rejects multiple DMA
  engines in one system. Multi-core or per-core DMA would need a finer key.
- If the DMA descriptor queue is full, `startCopy()` returns false. The MMIO
  path warns and drops the descriptor. Tests and compiler-generated schedules
  should respect the queue depth or insert waits.
- `run_spm.py` is the current topology; older `run_spm_v*.py` files are
  historical backups.
- The direct SPM port is implemented for O3 CPU. Other CPU models would need
  their own routing support.
- The branch does not add new gem5 RVV instruction implementation relative to
  `stable`. RVV-related notes in tests are compatibility/workaround notes for
  existing vector support and compiler-generated code.
- `libspm.h` fallback DMA-buffer constants are not the authoritative runtime
  map when launched through `run_spm.py`; use the environment variables set by
  the simulator config.

