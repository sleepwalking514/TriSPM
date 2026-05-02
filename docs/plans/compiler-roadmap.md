# TriSPM Compiler Roadmap: Detailed Implementation Plan

## Context

The TriSPM simulator (gem5) is largely complete and can execute kernels with SPM, but performance doesn't yet beat cache — because hand-written C code doesn't exploit tiling/double-buffering/DMA scheduling. The compiler (triton-cpu fork) must generate optimized code that demonstrates SPM's value.

**End goal: a small but complete transformer inference pipeline** (not just a single kernel) running on gem5 with SPM, showing that compiler-controlled SPM is practical for real-world AI workloads. Individual kernel benchmarks (GEMM, attention, etc.) are stepping stones, not the destination.

## Viability Assessment

**The roadmap is viable and Phase 1 is significantly simpler than initially planned.** Key insights:

- Triton's programming model **already solves tiling** — `BLOCK_SIZE` constexprs define tile shapes, and the K-loop naturally iterates over tiles. No "tiling pass" needed.
- **Prior art exists**: Terapines' [AI-Benchmark](https://github.com/Terapines/AI-Benchmark) project has already solved RISC-V cross-compilation for triton-cpu with a clean 3-patch approach. We can adapt this directly.
- The real compiler work is: (1) adapt the Terapines AOT flow, (2) add SPM dialect ops + DMA lowering, (3) build the automatic double-buffering pass, (4) generalize across kernel types and chain them into an inference pipeline.

### Target Inference Pipeline

A minimal transformer decoder block exercising all major kernel types:

```
Input tensor (seq_len × d_model)
  │
  ├─ Layer Norm                    [reduction kernel]
  ├─ Q/K/V Linear Projections     [matmul kernel × 3]
  ├─ Flash Attention               [attention kernel: multi-tensor SPM]
  ├─ Output Projection             [matmul kernel]
  ├─ Residual Add                  [elementwise — cache-friendly, no SPM needed]
  ├─ Layer Norm                    [reduction kernel]
  ├─ FFN Up Projection             [matmul kernel]
  ├─ Activation (GELU/SiLU)       [elementwise — cache-friendly]
  ├─ FFN Down Projection           [matmul kernel]
  └─ Residual Add                  [elementwise]
  │
  Output tensor (seq_len × d_model)
```

**SPM-relevant kernel categories:**
1. **Matmul (GEMM)** — double-buffered A/B tiles in K-loop (highest SPM benefit)
2. **Reduction (LayerNorm, Softmax)** — double-buffered DMA prefetch along reduction axis for single-load tiled reductions
3. **Attention** — multi-tensor SPM allocation for Q/K/V/O tiles, nested loops
4. **Elementwise (activation, residual add)** — normally leave in cache unless fused into a higher-reuse kernel

---

## Phase 1: RISC-V AOT Cross-Compilation (Foundation) ✅ COMPLETE

**Goal:** Produce standalone RISC-V ELF binaries from Triton kernels, runnable in gem5 SE mode with RVV.

### What was done

**Compiler changes (triton-cpu `spm-dev` branch):**

1. **`third_party/cpu/backend/compiler.py`** — `TRITON_CPU_AOT` env var gates AOT mode:
   - Pipeline stops at LLVM IR (skip `make_asm`/`make_so` stages)
   - Host-specific passes skipped (AMX, AVX512, NEON dot product, ukernels, vec-lib)
   - Generic dot lowering (`add_convert_dot_generic`) used for RVV compatibility
   - Type promotion driven by `TRITON_CPU_AOT_FEATURES` instead of host CPU detection
   - `llvm.set_host_target()` skipped in AOT mode

2. **`third_party/cpu/backend/driver.py`** — New `make_aot_launcher()`:
   - Generates standalone C header + source (not Python C extension)
   - Sequential 3-D grid loop (suitable for single-threaded gem5 SE mode)
   - Files written to `KERNEL_AUX_FILE_DIR`

3. **`python/triton/runtime/jit.py`** — AOT flow in JIT:
   - Skips kernel execution when `TRITON_CPU_AOT` is set
   - Dumps compiled LLVM IR to `KERNEL_AUX_FILE_DIR`

4. **`python/triton/compiler/compiler.py`** — `CompiledKernel` AOT handling:
   - Gracefully handles missing binary in AOT mode
   - Eagerly creates launcher to emit .c/.h artifacts

**Workloads infrastructure (TriSPM top-level `workloads/` directory):**

- Makefile-based build system for cross-compiling Triton kernels and running on gem5
- `scripts/build_kernel.sh` — AOT compile + RISC-V cross-compile pipeline
- `scripts/run_gem5.sh` — gem5 SE-mode execution
- Kernel definitions: vector_add, matmul, layer_norm (+ more in `kernels/`)
- Each workload has: `kernel.py`, `harness.c`, `config.sh`, `build/` output

**Verified:** All three kernels (vector_add, matmul, layer_norm) produce `.llir` → `.s` → linked ELF, and gem5 runs completed (m5out/stats.txt exists).

---

## Phase 2: SPM Dialect Extensions + DMA Lowering

**Goal:** Define DMA operations in the TritonCPU dialect and implement their lowering to RISC-V MMIO/Xspm instructions.

### New dialect ops (in `TritonCPUOps.td`)

```tablegen
def TTC_DmaEnqueue2DOp : TTC_Op<"dma_enqueue_2d"> {
  let summary = "Enqueue a 2D strided async DMA transfer";
  let arguments = (ins
    I64:$dst,           // SPM destination address
    I64:$src,           // DRAM source address
    I64:$width,         // bytes per row
    I64:$height,        // number of rows
    I64:$src_stride,    // source row stride in bytes
    I64:$dst_stride     // destination row stride in bytes
  );
}

def TTC_DmaWaitOp : TTC_Op<"dma_wait"> {
  let summary = "Block until all pending DMA transfers complete";
}
```

### LLVM address spaces for memory correctness

- Define `SPM_ADDR_SPACE = 3` in the TritonCPU dialect
- SPM memrefs use `memref<MxNxf32, 3>` (address space 3)
- DRAM memrefs stay in default address space 0
- This creates a hard optimization boundary: LLVM knows SPM and DRAM pointers never alias
- Additionally: all DMA MMIO stores are `volatile`, bracketed by `fence iorw, iorw`

### Files to modify/create

1. **`include/triton/Dialect/TritonCPU/IR/TritonCPUOps.td`** — Add `DmaEnqueue2DOp`, `DmaWaitOp`
2. **`lib/Dialect/TritonCPU/IR/Ops.cpp`** — Add verification for new ops
3. **NEW: `third_party/cpu/lib/TritonCPUToLLVM/DmaOpsToLLVM.cpp`** — Lower DMA ops to:
   - **MMIO path** (default): volatile LLVM stores to DMA registers at 0xF0000000 (safe: `run_spm.py` maps via `process.map()`)
   - **Xspm path** (flag): LLVM inline assembly for `.insn r 0x0B, ...`
4. **`third_party/cpu/lib/TritonCPUToLLVM/TypeConverter.cpp`** — Map SPM address space 3 to LLVM
5. **`third_party/cpu/lib/TritonCPUToLLVM/CMakeLists.txt`** — Add `DmaOpsToLLVM.cpp`
6. **`third_party/cpu/include/TritonCPUToLLVM/Passes.td`** — Add `DmaOpsToLLVM` pass
7. **`third_party/cpu/triton_cpu.cc`** — Register `add_dma_ops_to_llvmir` pybind
8. **`third_party/cpu/backend/compiler.py`** — Insert DMA lowering pass in AOT pipeline

### DMA lowering detail (MMIO path)

`ttc.dma_enqueue_2d(dst, src, width, height, src_stride, dst_stride)` lowers to:
```
%mmio = 0xF0000000
volatile store %src  → %mmio+0x00   // SRC register
volatile store %dst  → %mmio+0x08   // DST register
volatile store %src_stride → %mmio+0x20
volatile store %dst_stride → %mmio+0x28
volatile store %height     → %mmio+0x30
fence iorw, iorw
volatile store %width      → %mmio+0x10   // LEN register (triggers enqueue)
fence iorw, iorw
```

### Verification
- Write a test MLIR module with DMA ops
- Lower through the pass, verify RISC-V assembly output
- Cross-compile, run on gem5, verify DMA transfers execute correctly

---

## Phase 3: SPM Transformation Pass — GEMM First (Core) ✅ CURRENTLY CLOSED FOR SINGLE-KERNEL COVERAGE

**Goal:** An MLIR pass that transforms tiled DRAM loads into double-buffered DMA-to-SPM transfers. Start with GEMM (the highest-value pattern), then generalize.

### 3a. GEMM double-buffering (first target)

**Input** (Triton GEMM at tttcir level — the K-loop):
```
scf.for %k = 0 to K step BLOCK_K iter_args(%acc, %a_ptr, %b_ptr) {
  %a_memref = ttc.extract_memref(%a_ptr)
  %a_tile = vector.transfer_read %a_memref      // DRAM load
  %b_memref = ttc.extract_memref(%b_ptr)
  %b_tile = vector.transfer_read %b_memref      // DRAM load
  %acc_new = <dot>(a_tile, b_tile, acc)
  %a_next = tt.advance(%a_ptr, [0, BLOCK_K])
  %b_next = tt.advance(%b_ptr, [BLOCK_K, 0])
  yield %acc_new, %a_next, %b_next
}
```

**Output** (after ConvertMemoryToSPM):
```
// Prologue: prime buffer 0
ttc.dma_enqueue_2d(spm_a0, dram_a_start, ...)
ttc.dma_enqueue_2d(spm_b0, dram_b_start, ...)
ttc.dma_wait()

scf.for %k iter_args(%buf_idx=0, %acc, %dram_a_addr, %dram_b_addr) {
  %spm_a_cur = select(%buf_idx==0, spm_a0, spm_a1)
  %spm_b_cur = select(%buf_idx==0, spm_b0, spm_b1)

  // Prefetch NEXT tiles async (overlaps with compute)
  scf.if (%k + BLOCK_K < K) {
    %spm_a_nxt = select(%buf_idx==0, spm_a1, spm_a0)
    ttc.dma_enqueue_2d(%spm_a_nxt, %dram_a_next, ...)
    ttc.dma_enqueue_2d(%spm_b_nxt, %dram_b_next, ...)
  }

  // Compute on CURRENT tiles from SPM (address space 3)
  %a_tile = vector.transfer_read memref<BM×BK×f16, 3> @ spm_a_cur
  %b_tile = vector.transfer_read memref<BK×BN×f16, 3> @ spm_b_cur
  %acc_new = <dot>(a_tile, b_tile, acc)

  ttc.dma_wait()  // wait before swapping buffers
  yield (1-%buf_idx), %acc_new, %dram_a_next, %dram_b_next
}
```

### 3b. Reduction kernels (LayerNorm, Softmax)

Reduction is handled by the existing `ConvertMemoryToSPM` compiler pass, not
by a standalone workload-specific pass. The reduction branch is
`transformReductionLoop` in
`compiler/third_party/cpu/lib/TritonCPUTransforms/ConvertMemoryToSPM.cpp`.

Current status:
- Single-load tiled reductions are double-buffered: prologue DMA fills the
  first SPM buffer, the loop waits at the body top, reads the current SPM
  buffer, enqueues the next tile into the alternate buffer, and flips the
  buffer index.
- Structural coverage is in
  `compiler/test/TritonCPU/convert-memory-to-spm.mlir`, including the
  reduction body-top wait, buffer select, alternate-buffer enqueue, and
  2-D non-leading-IV address case.
- Opt-in production coverage uses `layer_norm`: its mean and variance passes
  were rewritten to `tl.make_block_ptr` + `tl.advance`, and its final normalize
  loop now uses block pointers for `x`, `gamma`, `beta`, and `out`. With
  `TRITON_ENABLE_SPM_REDUCTIONS=1`, these lower into the reduction/streaming
  `transformReductionLoop` pattern.
- Performance status is not acceptable for standalone reduction SPM: archived
  opt-in runs showed SPM LayerNorm far slower than cache on flushed 32x64,
  512x1024, and 1024x1024 comparisons, and still slower on noflush 32x64.
  Default AOT therefore leaves reduction/streaming kernels on cache path unless
  explicitly opted in for coverage/debugging.

### 3d. Transformer-facing single-kernel coverage before Phase 4

Before treating Phase 4 attention as unblocked, add workload harness coverage
for the non-attention kernels that the transformer block will need:

- **Activation**: GELU/SiLU approximation or whichever activation the target
  FFN uses. This should normally stay cache-only; verify that SPM mode builds
  and runs without accidentally inserting SPM for low-reuse elementwise loads.
- **Residual/add**: cache-path elementwise producer for the activation
  backbone; again, coverage/harness matters more than an SPM transform.
- **Softmax or another reduction/streaming kernel**: reuse the canonical
  block-pointer `transformReductionLoop` path and expose whether the current
  layer_norm performance issue generalizes.
- **Matmul variants used by Q/K/V/FFN**: already based on GEMM, but should be
  represented in manifests with transformer-like shapes and blocking sweeps.

Recommended rule: do not write a special SPM pass for standalone activation or
residual/add. Keep them as cache-path kernels and let graph placement preserve
their producer/consumer activations as Tier 2. Only revisit SPM for elementwise
work if it is fused with matmul, layer_norm, softmax, or attention and gets real
tile reuse.

### 3c. Data placement policy

Not all loads benefit from SPM. The pass must classify tensors into placement tiers.

Important distinction: tier placement chooses the tensor's backing allocation
and graph-safety policy; it is not the full SPM reuse strategy. GPU-style shared
memory performance requires an explicit promotion/residency planner that chooses
which tile or temporary lives in SPM, at what scope, and for how many uses. The
next compiler direction is tracked in `spm-explicit-promotion.md`.

#### Three-tier placement policy

The critical insight (raised by advisor): if a vector exceeds SPM capacity and is tiled through SPM, earlier tiles are evicted before scalar work begins. If the vector lives in **uncacheable** DRAM, subsequent scalar access hits DRAM directly — destroying performance. The solution is a compile-time placement decision:

| Tier | Condition | Placement | Rationale |
|------|-----------|-----------|-----------|
| **1. SPM-resident** | `total_size ≤ SPM_capacity` AND has scalar reuse | SPM (uncacheable) | Entire vector fits — scalar loads go through `spm_port` at ≈L1 latency |
| **2. Cacheable + DMA tiling** | `total_size > SPM_capacity` AND has scalar reuse | **Cacheable** DRAM | Vector work uses DMA→SPM tiling; scalar work hits L2 (see below) |
| **3. Uncacheable DMA buffer** | No scalar reuse, no downstream cache/CPU consumer, and graph metadata proves external read-only DMA-only use | Uncacheable DRAM | Maximum SPM benefit, zero cache pollution |

**Tier 2 is the key design point.** It guarantees SPM is **never worse than cache baseline**, and is often better:

#### L2-warming side effect (why Tier 2 beats pure cache)

When DMA reads a tile from **cacheable** DRAM into SPM, the request passes through L2XBar. Per the coherence analysis (Cases B/C in Phase 5 notes):
- **L2 hit (Case B):** Data already in L2, no extra cost.
- **L2 miss (Case C):** L2 allocates a new cache line for the data fetched from DRAM.

Either way, **after the DMA read, L2 holds a copy of the tile.** This means the tiled DMA process acts as a structured L2 prefetcher. By the time vector processing finishes and scalar work begins, the data is already warm in L2. Result:
- **Vector work:** SPM-speed, deterministic latency, no cache-line thrashing
- **Scalar work:** L2-speed, warmed by DMA reads as a side effect
- **Pure cache baseline:** Both vector and scalar ops compete for the same cache lines, causing mutual evictions

The only scenario where L2 warmth fails is if total vector size exceeds L2 capacity (512 KiB in our config), causing early tiles to be evicted from L2 before scalar access. But pure cache would also thrash in this case — so Tier 2 is still "no worse."

#### Applies to both input and output tensors

The placement decision is made at **tensor allocation time** (in the harness or compiler), before the kernel runs. It applies to outputs too: if matmul's C matrix will later be consumed by another kernel, C should be allocated in cacheable range from the start. Once Phase 4b output-tile DMA writeback exists, that writeback should target the tensor's pre-allocated address; there should not be a separate runtime "writeback destination" decision.

For an end-to-end transformer pipeline, the default policy is graph-aware and conservative: **intermediate activations and producer outputs are Tier 2 cacheable unless the graph proves they are terminal DMA-only streaming tensors.** Tier 3 is reserved for external read-only inputs/weights that have no scalar/fallback/downstream cache consumer. This keeps a cacheable activation backbone across matmul, layer norm, residual, attention, and FFN kernels while still allowing selective uncacheable inputs where they are genuinely useful.

#### Compiler analysis for tier classification

Detection of `has_scalar_reuse` at compile time on Triton IR:
- After the tiled vector loop (`scf.for` with block-pointer advancement), are there non-tiled `transfer_read` ops (element-level or irregular access) on the same tensor?
- Conservative default: if analysis is inconclusive, use **Tier 2** (cacheable) — safe fallback

**Basic heuristic** (sufficient for MVP):
- Loads inside `scf.for` with block-pointer advancement → SPM-eligible (Tier 1 or 2 based on size)
- Kernel-local loads with no subsequent scalar consumers and predictable access → Tier 3 only in single-kernel MVP verification
- Graph-level transformer mode: producer outputs / intermediate activations → Tier 2; external read-only DMA-only streaming inputs/weights → Tier 3
- Scalar loads, irregular access, low-reuse elementwise ops, accumulators → cache (not SPM at all)

#### Paper angle

> "The compiler's data placement policy provides a performance floor guarantee: by analyzing access patterns, the compiler routes tensors with mixed vector/scalar access through cacheable memory, ensuring SPM never degrades below cache baseline. Pure vector workloads get the full SPM advantage through uncacheable DMA buffers with zero cache pollution. This access-pattern-aware placement is a capability that hardware-managed cache cannot offer."

This directly addresses the advisor's vector→scalar coherence concern: the compiler decides, not the hardware.

### SPM buffer layout (compile-time constants)

| Buffer | Address | Size |
|--------|---------|------|
| spm_a0 | `SPM_BASE + 0` | `BLOCK_M × BLOCK_K × sizeof(dtype)` |
| spm_a1 | `SPM_BASE + tile_a` | same |
| spm_b0 | `SPM_BASE + 2×tile_a` | `BLOCK_K × BLOCK_N × sizeof(dtype)` |
| spm_b1 | `SPM_BASE + 2×tile_a + tile_b` | same |

Total: `2×(tile_a + tile_b)` must fit in SPM. Validated at compile time.

### Loop boundary handling

- **MVP:** Require `K % BLOCK_K == 0` (standard practice — autotuner selects compatible block sizes)
- **Later:** Clamp DMA width/height for final partial tile

### Files to create/modify

1. **NEW: `third_party/cpu/lib/TritonCPUTransforms/ConvertMemoryToSPM.cpp`** — Core pass
2. **`third_party/cpu/include/TritonCPUTransforms/Passes.td`** — Pass tablegen
3. **`third_party/cpu/include/TritonCPUTransforms/Passes.h`** — Pass factory
4. **`third_party/cpu/lib/TritonCPUTransforms/CMakeLists.txt`** — Build
5. **`third_party/cpu/triton_cpu.cc`** — Register `add_convert_memory_to_spm`
6. **`third_party/cpu/backend/compiler.py`** — Insert in `make_tttcir()`:
   ```python
   if self.cpu_arch == "riscv64":
       cpu.passes.ttcpuir.add_convert_memory_to_spm(pm, spm_base, spm_size)
   ```

### Verification
- GEMM: Run on gem5 with SPM, compare cycles vs cache-only baseline
- LayerNorm: Verify DMA prefetch in reduction loop
- Each kernel verified independently before integration

---

## Phase 4: Attention + Multi-Kernel SPM Management

**Goal:** Support flash attention (the most complex kernel for SPM) and enable multiple kernels to share SPM in sequence.

### 4a. Flash Attention SPM support

Flash attention is the hardest kernel because it has multiple tensors competing for SPM:
- Q tile (loaded once per outer loop), K/V tiles (streamed in inner loop), O tile (accumulated)
- SPM allocation must fit: `Q_tile + 2×K_tile (double-buffered) + 2×V_tile (double-buffered) + O_tile`
- Inner loop over K/V sequence length with double-buffering; Q tile pinned in SPM

### 4b. SPM writeback for output tiles
- DMA SPM→DRAM for output tiles (C in GEMM, O in attention) after compute loops
- Writeback destination is the tensor's pre-allocated address (determined by Phase 3c placement policy — not a separate runtime decision)
- If the output tensor was allocated in cacheable range (Tier 2: default for producer outputs / intermediate activations), writeback warms L2 automatically
- If graph-level placement proves a terminal output has no downstream cache/CPU consumer, it may be allocated in uncacheable range; otherwise outputs stay cacheable

### 4c. Inter-kernel SPM lifetime management
- Each kernel gets full SPM (no cross-kernel sharing needed — kernels run sequentially)
- But the harness must ensure SPM state is clean between kernel calls
- Compile-time SPM allocator validates each kernel fits independently

### 4d. Custom Xspm instruction emission
- Switch DmaOpsToLLVM from MMIO to `.insn` inline assembly (fewer instructions per DMA)

### 4e. Loop boundary handling for non-aligned dimensions
- Clamp DMA bounds for partial tiles; translate Triton masks to DMA bounds

---

## Phase 5: End-to-End Inference Pipeline

**Goal:** Chain multiple Triton kernels into a complete transformer decoder block, demonstrating that compiler-controlled SPM works for real-world inference, not just microbenchmarks.

### Target: Minimal Transformer Decoder Block

```
// One decoder layer — all kernels AOT-compiled from Triton
layer_norm(X, ...)                          // reduction
matmul(X_norm, W_q, Q)                     // GEMM
matmul(X_norm, W_k, K)                     // GEMM
matmul(X_norm, W_v, V)                     // GEMM
flash_attention(Q, K, V, O)                // attention
matmul(O, W_o, attn_out)                   // GEMM
residual_add(X, attn_out, X2)              // elementwise (cache)
layer_norm(X2, ...)                         // reduction
matmul(X2_norm, W_up, hidden)              // GEMM (FFN)
activation(hidden)                          // elementwise (cache)
matmul(hidden, W_down, ffn_out)            // GEMM (FFN)
residual_add(X2, ffn_out, output)          // elementwise (cache)
```

### Implementation

1. **Inference harness** (`workloads/transformer_block/harness.c`):
   - Allocates all weight matrices and activation buffers
   - Calls each kernel launcher in sequence
   - Pre-loads weights into memory (simulating model loading)
   - Small but realistic dimensions (e.g., d_model=256, n_heads=4, seq_len=64)

2. **Kernel compilation**: Each Triton kernel AOT-compiled independently, all linked into one ELF

3. **Comparison runs on gem5**:
   - **Baseline**: all kernels use cache only (no SPM)
   - **SPM**: compiler inserts DMA + double-buffering for SPM-eligible kernels
   - **Selective SPM**: only GEMM + attention use SPM; elementwise stays in cache
   - Collect: total cycles, cache miss rate, DMA utilization, SPM hit rate

### Graph-level placement first; fusion second

`SPMTensorPlacement` is currently a **per-kernel** pass. Its tier classification
analyzes scalar reuse only within the function being compiled, so a chain like
`matmul(A, B, C) → matmul(C, W, D)` can misclassify `C` if the second matmul is
viewed in isolation. The transformer pipeline therefore needs a lightweight
graph-level placement manifest before Phase 5 evaluation.

The conservative graph rule is:

- Intermediate activations and producer outputs form a **cacheable activation
  backbone** and default to Tier 2.
- Tier 3 is limited to external read-only DMA-only streaming inputs/weights with
  no scalar, fallback, or downstream cache consumer.
- Tier 1 is reserved for future small SPM-resident hot state.
- Unknown lifetime or consumer information falls back to Tier 2.

This makes cross-kernel chaining correct without requiring fusion. Fusion remains
valuable as a local optimization, but it should not be the mechanism that makes
tensor placement safe.

Fusion candidates for the decoder block:

| Fusion | Eliminates DRAM round-trip for | Notes |
|---|---|---|
| `LN + QKV` | `X_norm` (3-way reused as input to Q/K/V matmuls) | High value: LN output is small + reused 3× |
| `softmax + dropout` (training only) | softmax probabilities | Inference uses bare softmax inside flash attention — already fused |
| `GELU/SiLU + FFN-down` | activation output | Standard MLP fusion |
| `residual + LN` | residual sum | Common pre-norm transformer pattern |
| `flash attention` | Q·Kᵀ, softmax, ·V intermediates | Already a fusion by construction |

Each fusion is a **kernel-source change** (one `@triton.jit` function instead of two),
not a compiler requirement. The SPM pipeline (placement, double-buffer, sidecar) treats
the fused kernel as a normal single kernel, while graph-level placement handles the
unfused boundaries that still remain.

### Paper evaluation metrics
- End-to-end latency: full decoder block, SPM vs cache
- Per-kernel breakdown: which kernels benefit most from SPM
- SPM utilization: how much of SPM capacity is used by each kernel
- Scalability: vary d_model, seq_len to show trends
- Comparison: compiler-managed SPM vs hand-written SPM code vs cache-only

### Why this matters for the paper
- Moves beyond "GEMM is 2× faster" microbenchmark claims
- Shows the compiler handles heterogeneous kernel types automatically
- Demonstrates SPM data placement policy works across a real workload
- Proves the system is practical, not just a toy

### Inter-kernel coherence: no compiler intervention needed

When kernels are chained (e.g., `residual_add` writes activations through L1D/L2, then `layer_norm`
DMA-reads them into SPM), cache–SPM coherence is maintained automatically by gem5's hardware.

**No `CBO.FLUSH`, no `FENCE`, and no compiler-inserted coherence logic is required.**

#### DMA read flow: three cases (code-verified against gem5 source)

DMA issues `ReadReq` via `DmaPort` with default `flags=0` (no uncacheable flag;
`dma_device.hh:205`). Routing through L2XBar is purely by address range of `mem_side_ports`:
cacheable range → L2Cache; DMA buffer → uc_bridge; SPM → SPM.port.

**Case A — L1D has dirty data (snoop hit, L2 completely uninvolved):**

1. DMA `ReadReq` arrives at L2XBar `cpu_side_ports` (`spm_dma_engine.cc:322`).
2. L2XBar snoops L1D via `forwardTiming()` (`coherent_xbar.cc:242`).
3. L1D `handleSnoop`: block is dirty → `setCacheResponding()` → `doTimingSupplyResponse()`
   provides data via snoop response (`cache.cc:1168,1206,1230`).
4. Back in L2XBar: `sinkPacket()` returns **true** via condition 4:
   `cacheResponding() && !needsWritable()` (`coherent_xbar.cc:1107-1108`).
   Note: `L2XBar.point_of_coherency = False` (`XBar.py:154-173`), but condition 4
   does not require PoC — it fires because ReadReq does not need writable.
   **The request is NOT forwarded to L2Cache.**
5. Snoop response is converted to normal response and routed to DMA via
   `cpuSidePorts[dest]->schedTimingResp()` (`coherent_xbar.cc:658,681`).
6. L1D transitions Modified → Owned (WritableBit cleared, `cache.cc:1197`).
   DirtyBit remains set. L2 receives the data only when L1D later naturally evicts the line.

**Case B — L2Cache hit (L1D snoop miss):**

1. L1D snoop miss → `cacheResponding` stays false → `sinkPacket` = false.
2. `forwardPacket()` returns true (`coherent_xbar.cc:1122`: `pkt->isRead()` → true).
3. Request forwarded to L2Cache → cache hit → response returned to DMA.
4. No new L2 allocation needed (data already in L2).

**Case C — L2Cache miss → DRAM (L2 allocates a new line):**

1. Same as Case B up to L2Cache, but L2 misses.
2. L2 creates MSHR with `allocOnFill(ReadReq) = true` (`base.hh:443-451`).
3. Request forwarded to membus → DRAM → response returns.
4. **L2Cache allocates a new cache line**: `handleFill()` → `allocateBlock()` →
   `updateBlockData()` copies data into the new block (`base.cc:1584-1600,1652-1658`).
5. Response sent back through L2XBar to DMA engine.
6. **L2 now holds a copy** of data that was headed to SPM.

#### DMA engine two-step mechanism

`SpmDmaEngine` performs each transfer as **two separate L2XBar transactions** (`spm_dma_engine.cc`):

- **1D** (lines 318-343): `beginRead()` issues `ReadReq` to source address → `readDone()` callback
  issues `WriteReq` to destination address.
- **2D pipelined** (lines 347-390): all row reads issued in parallel via `beginPipelinedRead()`,
  each row's `pipelinedRowReadDone()` immediately starts the write for that row.

The read and write are independent XBar transactions — data is buffered inside the DMA engine
(`buffer` for 1D, `rowBuffers[]` for 2D) between read completion and write initiation.

#### L2 cache pollution from DMA reads (theoretical, negligible in practice)

In Case C, L2Cache allocates a line for DMA read data headed to SPM. If the CPU never re-accesses
that data through cache, the L2 copy is wasted. However, **Case C is rare in practice:**

- **Inter-kernel warm data** (the primary DMA-from-cacheable scenario): Kernel A writes activations
  through cache → data is in L1D or L2 → Case A or B, not C. Case C only occurs if the data
  was evicted from *both* L1D (32 KiB) and L2 (512 KiB) between kernels, which requires an
  inter-kernel working set exceeding L2 capacity — unlikely for our target dimensions
  (d_model=256, seq_len=64).
- **Cold data** (weights, inputs): Should be allocated in the DMA buffer region
  (0x20000000–0x3FFFFFFF, uncacheable) by design. Data there is routed through
  `uc_bridge → membus → DRAM`, bypassing L2 entirely. So this scenario doesn't arise.

**Conclusion:** L2 pollution from DMA is a non-issue for our system. The address map design
(cacheable vs. UC DMA buffer) and the natural warmth of inter-kernel data ensure that DMA reads
from cacheable addresses almost always hit Case A (L1D snoop) or Case B (L2 hit).

---

## Phase 6: Evaluation — SPM vs Cache Baseline (Paper-Critical)

**Goal:** Produce comprehensive comparison data between SPM and cache-only execution. This is the single biggest factor determining which conference to target.

### 6a. Cache Baseline Comparison [P0 — Highest Priority]

Run the same kernels on gem5 in cache-only mode (no SPM hardware, no ConvertMemoryToSPM pass). Collect:
- Total cycles, L1D/L2 miss rate, memory bandwidth utilization
- Same workloads, same tile sizes, same gem5 config minus SPM

**Comparison configurations:**

| Config | Description |
|--------|-------------|
| Cache-only | Pure cache hierarchy, no SPM hardware |
| SPM (Tier 3) | Uncacheable DMA buffer, current implementation |
| SPM (Tier 2) | Cacheable + DMA tiling; placement plumbing exists and L2-warming is verified, but broad workload integration remains future work |

**Expected speedup vs. target venue:**

| Speedup (SPM vs Cache) | Target Level | Venues |
|------------------------|-------------|--------|
| < 1.2x | B-tier | DATE / LCTES / Workshop |
| 1.3x – 1.8x | A-tier | CGO / PACT / ICS |
| 2x+ | Top-tier possible | ASPLOS / MICRO |

### 6b. Tier 2 Placement Policy And Evidence [P0]

The three-tier placement policy is the paper's most valuable insight. The Tier 2/3 MVP plumbing has landed, and the L2-warming side effect is verified in `../evidence/l2_warming.md`; the remaining work is to connect Tier 2 to broader scalar-reuse workloads, add graph-level conservative placement for transformer edges, and compare it against cache baselines.

- Extend Tier 2 workload coverage: tensors with scalar reuse that exceed SPM capacity should go to cacheable DRAM.
- Graph-level conservative placement build/verify MVP is in place:
  `workloads/scripts/graph_placement.py` reads graph tensor-edge metadata,
  forces intermediate activations and producer outputs to Tier 2, and reserves
  Tier 3 for external read-only DMA-only tensors.  The remaining work is an
  executable multi-kernel graph harness plus cache-baseline comparison.
- Use the completed L2-warming evidence as the mechanism proof: after DMA reads from cacheable DRAM, L2 holds data copies.
- Compare Tier 2 vs pure cache on real kernels: prove "cacheable + DMA tiling" is never worse than cache (performance floor guarantee).

**This is the only experimental evidence for the core claim "SPM never worse than cache."**

### 6b.1 Explicit SPM Promotion / Residency [P0]

Tier 2 is a safe backing policy, not a replacement for explicit SPM reuse.
`transformFusedMicroGemmLoop` already demonstrates the right direction by
keeping a B window and accumulator tile resident in SPM, but that idea is not
yet a planner-driven compiler abstraction. D1 has landed: matmul now reports
explicit B-window, A-micro, and accumulator promotion records without changing
generated behavior, and the D1 sidecar has a versioned debug/evidence schema
with structural rejected-candidate records and report tests. A separate DMA
fence regression was fixed at the same gate by removing the generic inline-asm
memory clobber; D1b validation kept 64x64x64 SPM-only correctness passing and
256x256x256 SPM-only at 1,729,209 cycles / 5913 assembly lines.

D2 has landed as a separate opt-in row-resident reduction prototype, not as a
default policy. `TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS=1` copies one
LayerNorm `x[row, :]` into SPM, reuses it across mean, variance, and normalize,
and keeps `gamma` / `beta` on cache. Default `make verify-layer_norm` still
proves the cache path; `TRITON_ENABLE_SPM_REDUCTIONS=1` remains the old
streaming reduction coverage path; the D1 sidecar remains debug/evidence only.
D2 measurements are still slower than cache (32x64: 10,279 vs 6,138 cycles;
512x1024: 1,245,422 vs 1,012,922 cycles), so D3 is now the next compiler gate:
add a conservative `SPMPromotionPlanner` concept that records promoted tile
shape, scope, use count, copy-in/copy-out, SPM footprint, rejected candidates,
and profitability before any automatic row/block-resident policy. See
`spm-explicit-promotion.md`.

First targets:

- Refactor the existing fused matmul scheduler into explicit promotion records
  without changing behavior.  Done in D1; the records remain debug/evidence
  output and do not yet drive scheduling.
- Prototype a row-resident LayerNorm path that DMA-copies `x[row, :]` once and
  reuses it across mean, variance, and normalize, instead of streaming 8-float
  chunks three times.  Done in D2 as opt-in evidence only.
- Keep reduction SPM default-off unless the promotion path beats cache on the
  existing 32x64 and 512x1024 comparisons.  D2 did not meet this bar, so D3
  must reject it by default until a stronger cost model or schedule changes the
  measurements.

### 6c. Expand Workload Coverage [P1]

`matmul` is the mature GEMM workload. `layer_norm` is now the cache-path
reduction workload by default, with opt-in SPM reduction coverage retained via
`TRITON_ENABLE_SPM_REDUCTIONS=1`; there is no separate synthetic `reduction`
workload. `vector_add` is intentionally too trivial for SPM tiling. Before
Phase 4 attention, add transformer-facing harness coverage even for kernels
that should remain cache-only. Need at least 5 representative kernels:

| Kernel | Type | SPM Pattern | Priority |
|--------|------|-------------|----------|
| matmul | GEMM | double-buffer / fused micro scheduler ✅ | Done; final headline needs blocking sweep |
| layer_norm | reduction | cache path by default; opt-in double-buffer coverage | Default fixed; SPM reduction remains performance blocker |
| activation (SiLU) | elementwise | cache path expected | Done: workload + verify + flushed ROI smoke compare |
| residual_add | elementwise | cache path expected | Done: workload + verify + flushed ROI smoke compare |
| softmax | reduction | cache path by default; opt-in SPM only if measured useful | Done for row-wise smoke coverage; promotion/perf experiments remain |
| layer_norm -> qkv graph | graph placement | activation Tier 2, external weights Tier 3 | Build/verify MVP done; executable harness pending |
| flash_attention | multi-tensor | Q pinned + K/V double-buffer | P1 — depends on Phase 4 |
| cross_entropy | reduction | cache path by default | P2 — diversify workload mix |

### 6d. Performance Breakdown Analysis [P1]

Reviewers will demand understanding of where speedup comes from. Extract from gem5 stats:
- DMA latency hiding ratio: compute cycles vs DMA wait cycles
- Bank conflict statistics: per-port per-bank read/write conflicts
- SPM utilization: actual SPM usage / total SPM capacity
- Cache pollution reduction: L1D/L2 miss rate decrease under SPM mode

### 6e. Area-Equivalent Comparison [P2]

SPM vs cache cost-effectiveness at equal silicon area:
- Use CACTI or published data to estimate: 32KB SPM + 32KB L1D vs 64KB L1D
- If "small cache + SPM" outperforms "large cache," this is a very compelling argument
- No RTL synthesis needed — cite existing literature

### 6f. Sensitivity Analysis [P2]

| Parameter | Range | Purpose |
|-----------|-------|---------|
| SPM size | 16KB / 32KB / 64KB | Find cost-effectiveness sweet spot |
| DMA queue depth | 1 / 2 / 4 descriptors | Validate pipelining value |
| Bank count | 4 / 8 / 16 | Measure bank conflict impact |
| Tile size | Various BLOCK_M/N/K | SPM capacity utilization |

---

## Execution Dependencies

```
Phase 1 (AOT cross-compile) ✅ COMPLETE
  └─> Phase 2 (DMA dialect ops + address spaces + lowering) ✅ COMPLETE
        └─> Phase 3 (SPM pass: GEMM → reduction → data placement) ✅ current single-kernel coverage closed
              ├─> Phase 4 (Attention + multi-kernel SPM + optimizations) ← CURRENT NEXT
              │     └─> Phase 5 (End-to-end transformer inference pipeline)
              └─> Phase 6 (Evaluation — SPM vs Cache) ← Critical path for publication
                    ├─ 6a Cache baseline [Phase 3 matmul/layer_norm baseline ready]
                    ├─ 6b Tier 2 workload integration [placement MVP + L2 evidence + graph build/verify done; executable graph pending]
                    ├─ 6c Additional workloads [after Phase 3/4 pattern support]
                    ├─ 6d Breakdown analysis [built on 6a data]
                    ├─ 6e Area comparison [independent, can start anytime]
                    └─ 6f Sensitivity analysis [built on 6a framework]
```

**Minimum viable paper:** Phase 1 + 2 + 3 + 6a + 6b (GEMM + LayerNorm, SPM vs Cache comparison, Tier 2 verification)
**Solid paper (CGO/PACT level):** + 6c + 6d (5 workloads, breakdown analysis)
**Strong paper (ASPLOS/MICRO attempt):** + Phase 5 + 6e + 6f (end-to-end pipeline, area comparison, sensitivity analysis)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Wide vectors scalarized on RISC-V | External compiler flags: `-mrvv-vector-bits=256 --riscv-disable-rvv-fixedlen=false`. Verify RVV insns in output. |
| SPM loads reordered past DMA wait | LLVM address spaces (SPM=3) + volatile + fence |
| ConvertMemoryToSPM pattern too fragile | Start with GEMM, generalize incrementally; graceful fallback to cache for unrecognized patterns |
| Cross-compilation toolchain issues | Static linking; test toolchain independently |
| SPM capacity overflow | Compile-time validation: `2×(tile_a+tile_b) ≤ spm_size` |
| Attention SPM pressure too high | Reduce tile sizes; pin only K/V in SPM, keep Q in cache |
| gem5 simulation too slow for full inference | Small dimensions (d_model=256); single decoder layer, not full model |
| Non-aligned dimensions | MVP: require alignment; later: DMA bound clamping |
| Long vector + scalar reuse (advisor concern) | Three-tier placement policy (§3c): vectors exceeding SPM with scalar consumers go to cacheable range; DMA tiling warms L2 as side effect; guarantees ≥ cache baseline |

---

## Key References

| Resource | Purpose |
|----------|---------|
| [Terapines/AI-Benchmark](https://github.com/Terapines/AI-Benchmark) | Reference RISC-V AOT patches for triton-cpu |
| [RISC-V blog: Triton on RISC-V](https://riscv.org/blog/triton-kernel-performance-on-risc-v-cpu/) | Prior art showing triton-cpu → RISC-V RVV works |
| `third_party/cpu/backend/compiler.py` | Pass pipeline, target config |
| `third_party/cpu/backend/driver.py` | Launcher generation (AOT rewrite target) |
| `third_party/cpu/lib/TritonToTritonCPU/ConvertMemoryOps.cpp` | tt.load → vector.transfer_read |
| `third_party/cpu/lib/TritonCPUToLLVM/MemoryOpToLLVM.cpp` | ttc ops → LLVM dialect |
| `include/triton/Dialect/TritonCPU/IR/TritonCPUOps.td` | Dialect op definitions |
| `third_party/cpu/triton_cpu.cc` | Pass registration + pybind |
| `simulator/src/scratchpad_mem/libspm.h` | SPM/DMA hardware interface |
| `simulator/src/scratchpad_mem/run_spm.py` | gem5 config (process.map, RVV settings) |
