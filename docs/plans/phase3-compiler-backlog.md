# Phase 3 Compiler Backlog

> Audit performed against `~/.claude/plans/moonlit-napping-origami.md` ("Phase 3 — SPM Transformation Pass") and the existing repo state.

---

## A. What is in place

The skeleton of Phase 3 is wired end-to-end:

- `compiler/third_party/cpu/lib/TritonCPUTransforms/ConvertMemoryToSPM.cpp`
  - GEMM K-loop pattern (`transformGemmLoop`, 2 dot-feeding loads).
  - Reduction pattern (`transformReductionLoop`, non-dot tiled loads), gated
    by `enable-reductions`.
  - Pass options `spm-base` / `spm-size` / `enable-reductions` declared in
    `Passes.td`.
- `compiler/include/triton/Dialect/TritonCPU/IR/SPMAttrs.h`
  - Single source of truth for `kSPMAddressSpace = 3`, included by both Phase 2 and Phase 3 files.
- `compiler/third_party/cpu/lib/TritonCPUToLLVM/TypeConverter.cpp`
  - Explicit `addTypeAttributeConversion` callback for `memref<…, 3>` → `!llvm.ptr<3>`.
- `compiler/third_party/cpu/triton_cpu.cc`
  - pybind `add_convert_memory_to_spm(pm, spmBase, spmSize)` registered.
- `compiler/third_party/cpu/backend/compiler.py` (lines 216–221)
  - SPM pass inserted inside `make_tttcir()` under `_AOT_MODE`, with env-var overrides `TRITON_SPM_BASE` / `TRITON_SPM_SIZE`.
  - DMA-to-LLVM pass inserted in `make_llir()` (line 303).
- Tests:
  - `compiler/test/TritonCPU/convert-memory-to-spm.mlir` (lit, FileCheck).
  - `compiler/test/TritonCPU/dma-ops-to-llvm.mlir` (lit, FileCheck).
  - `compiler/test/TritonCPU/spm-address-space.mlir` (lit, pins `memref<…, 3>` → `!llvm.ptr<3>` contract).
  - `compiler/python/test/unit/cpu/test_dma.py` (DMA dialect parsing + lowering).

These were the initially audited plumbing deliverables. The current Phase 3
compiler-robustness closure status is tracked in §E.

Clarification on reduction coverage: there is no standalone `reduction`
workload. Reduction lowering is the `transformReductionLoop` branch inside
`ConvertMemoryToSPM`. It is structurally tested in
`convert-memory-to-spm.mlir`, including both the opt-in SPM path and the
default reduction-off cache path. Its production workload coverage is
`layer_norm` when `TRITON_ENABLE_SPM_REDUCTIONS=1`: the mean, variance, and
final normalize loops use block pointers and hit the reduction/streaming path.
The final normalize loop exercises P1 #11's multi-load matcher with `x`,
`gamma`, and `beta`. Performance analysis showed standalone LayerNorm is
cache-favorable, so default AOT leaves reductions on cache path.

---

## B. Audit Findings And Remaining Work

This section preserves the original audit findings, but the final status is
tracked in §E. All P0/P1/P2 compiler-robustness items for the current Phase 3
coverage are now resolved; only explicitly deferred Phase 4+ items remain
active backlog.

### 3a. GEMM double-buffering — **mostly resolved** (P0 #1, #3, #5; writeback deferred to Phase 4b)

1. ~~**No real compute/DMA overlap.**~~ Resolved by P0 #3. The GEMM path now
   prefetches before `read_current` / compute and waits at the end of the
   iteration; measured overlap is recorded in
   `../archive/matmul-spm-lowering-closure.md`.
2. **No SPM writeback for output tile.** Phase 3 plan keeps writeback in §4b, but the GEMM pass should at least leave a hook (or stop transforming when the output is also SPM-resident). Today the C tile still goes through cacheable `vector.transfer_write`.
3. ~~**MVP gating leaves all current workloads untransformed.**~~ Resolved for
   `matmul`; `vector_add` is intentionally not transformed, and `layer_norm`
   can opt into the reduction path including its final multi-load normalize loop.

### 3b. Reduction prefetch — **resolved for current coverage** (P0 #2 race fix, double buffering, and P1 #11 multi-load matcher done)

1. ~~**Single-buffer race (correctness bug).**~~ Resolved by reordering:
   read current → reduce → prefetch → `dma_wait`. The single-load tiled
   reduction path now uses two SPM buffers with a prologue prefetch, per-iteration
   `dma_wait`, alternate-buffer prefetch, and flipped buffer index. Tiny
   reductions remain MMIO/control-heavy even after overlap; standalone
   LayerNorm defaults to cache path for this reason.
2. ~~**Pattern is too narrow** (`nonDotLoads.size() == 1`).~~ Resolved.
   Loops with multiple non-dot tiled loads now transform when the loads share
   the loop IV and fit in SPM. Covered structurally by
   `@reduction_multi_load_prefetch` and in production by LayerNorm final
   normalize (`x`, `gamma`, `beta`).

### 3c. Three-tier data-placement policy — **MVP landed** (see `three-tier-placement.md` M1-M8; coverage audited 2026-04-29)

> ✅ Tier 2/3 MVP plumbing implemented via `three-tier-placement.md` M1-M8: `SPMSpaceManager`、`SPMTensorPlacement` pass、JSON sidecar、launcher `_alloc/_free_all`、harness 改造。当前默认覆盖：`matmul` 正常命中 Tier 3；`vector_add` 和 `layer_norm` 空 JSON 是设计预期（无收益 / reduction SPM 默认关闭）。`layer_norm` 的 mean/variance/final normalize 仍可通过 `TRITON_ENABLE_SPM_REDUCTIONS=1` 命中 Tier 3。详见 `three-tier-placement.md` §4.1。

The table below reflects the state after the 2026-04-29 audit; see `three-tier-placement.md` for full details.

| Required piece | Status |
|---|---|
| `has_scalar_reuse` analysis on Triton/TTCIR | ✅ implemented in `SPMTensorPlacement` |
| Tier classification per tensor (size vs SPM, scalar reuse) | ✅ implemented (matmul args → Tier 3) |
| Compiler-driven choice of cacheable vs uncacheable allocation | ✅ implemented via `_alloc` dispatch |
| Harness API to allocate in cacheable / uncacheable / SPM regions | ✅ all 3 harnesses use `<kernel>_alloc` |
| L2-warming experimental verification (Tier 2) | ✅ completed — see `../evidence/l2_warming.md` |
| `verify-spm-policy` tooling | ✅ `make verify` / `make verify-<kernel>` |

### 3d. Loop boundary handling — **RESOLVED** (P0 #6)

> ✅ Compile-time bail-out added. See §E P0 #6.

Plan §3 requires MVP guard: `K % BLOCK_K == 0`. Neither `transformGemmLoop` nor `transformReductionLoop` checks or asserts this. If a workload violates the alignment assumption, the final DMA will read past the tensor boundary.

Fix: add a compile-time assert (or emit a runtime guard that skips the SPM path) when the trip count is not provably a multiple of the tile size.

### Phase-3 plumbing items still missing (raised in `../archive/phase2.md`)

1. ~~`DMA_MMIO_BASE` / `DMA_REG_*` are `static constexpr` in `DmaOpsToLLVM.cpp:37-44`.~~ ✅ Resolved. `DmaOpsToLLVM` now exposes `dma-mmio-base`, and the AOT pipeline forwards `TRITON_DMA_MMIO_BASE`.
2. ~~No `useXspmInsn` switch on `DmaOpsToLLVM`.~~ ✅ Resolved. The pass now exposes `use-xspm-insn` / `TRITON_USE_XSPM_INSN`; the switch currently fails explicitly until the Phase 4d instruction path exists.
3. ~~**Default-size mismatch**~~ ✅ Resolved. All unified to 256 KiB. See §E P0 #4.

---

## C. Quality review of the parts marked "done"

### 1. `transformGemmLoop` (correctness + brittleness)

- ~~**A/B identification is brittle.**~~ Resolved for the standard GEMM fallback/fused path. The matcher now identifies A/B by asking `analyzeGemmContract()` whether the two reads are the `vector.contract` lhs/rhs, and retries with reversed read order before deciding the loop is not a supported GEMM.
- ~~**Cloned-read rediscovery via `getLoc()` is fragile** (`:401-408`).~~ Resolved. Cloned reads are now recovered through `IRMapping::lookupOrNull(read.getResult())`, avoiding location-based collisions.
- ~~**Extra loads inside the GEMM K-loop were not explicitly verified.**~~
  Resolved for the current GEMM contract matcher: `ConvertMemoryToSPM` now
  classifies by the number of dot-feeding transfer reads, so a loop with
  extra non-dot tiled loads still transforms when exactly two reads feed the
  `vector.contract`. Covered by `@gemm_extra_non_dot_load`. Single-load
  GEMV and true multi-input contractions remain outside the current GEMM
  pattern.
- ~~**Non-overlapping prefetch (§B.3a above)**~~ — resolved for GEMM; keep
  watching this shape if `transformGemmLoop` is refactored.
- ~~**Unused `TiledLoadInfo::feedsDot` for non-dot loads** is fine, but the early `return false` cases in `transformGemmLoop` left the IR partially mutated.~~ Resolved by the bail-out cleanup pass. `InsertedBeforeGuard` now removes speculative work on failure, and lit coverage verifies dynamic-step no-DMA plus partial-prologue cleanup cases.

### 2. `SplitLargeContract` (new pass)

- ✅ Correctness verified: 256×256×256 / 32×32×32 PASS 65536/65536.
- ✅ DMA re-priming: `collectPrologueDma` correctly skips `isPure()` ops between DMA enqueues and the loop.
- **Only matches standard GEMM contract** (`(m,k)×(k,n)→(m,n)`). Non-standard layouts or batched contracts are silently skipped.
- **Clones full prologue DMA ops** including their operands. If `ConvertMemoryToSPM` changes the prologue structure (e.g. adds non-DMA side-effecting ops between enqueues), `collectPrologueDma` may collect fewer ops than expected.
- **`microM` default is 4**. Optimal value depends on VLEN and LMUL; currently hardcoded via `TRITON_MICRO_M` env var.
- **B tile is re-loaded 8× per K-iteration** (once per micro-loop). This is the cost of separate loops — each micro-loop independently loads the full B tile from SPM. Not a correctness issue but a known performance trade-off vs. a hypothetical shared-B approach.

### 3. `transformReductionLoop`

- ~~**Correctness bug** (race) documented in §B.3b.1.~~ Resolved by
  reordering the single-buffer prefetch after the current read.
- ~~**2-D non-leading-IV prefetch offset bug.**~~ Resolved on 2026-04-30 by
  using the stride of the dimension indexed by the IV instead of always
  using `strides[0]`; covered by `@reduction_2d_non_leading_iv`.
- ~~**Remaining performance issue:** the path is still single-buffered, so DMA
  latency is serially exposed.~~ Resolved for the single-load reduction path.
- ~~**LayerNorm final normalize remained a multi-load miss.**~~ Resolved by
  allowing multiple non-dot tiled loads sharing the loop IV. The path remains
  MMIO/control-heavy on tiny shapes, but the matcher coverage is in place.

### 4. `DmaOpsToLLVM.cpp` (Phase 2 lowering, used by Phase 3)

- Register layout, fence positions, MMIO offsets all line up with `simulator/src/scratchpad_mem/libspm.h`. Good.
- `addIllegalOp<DmaEnqueue2DOp, DmaWaitOp>()` is set in `TritonLLVMConversionTarget` (`:54-55`) — the `../archive/phase2.md` concern is resolved.
- Inline-asm `fence iorw, iorw` instead of `LLVM::FenceOp(seq_cst)` — comment in source explains why. Keep.

### 5. SPM buffer layout correctness

Plan §3 specifies a precise buffer address layout:

| Buffer | Address |
|--------|---------|
| spm_a0 | `SPM_BASE + 0` |
| spm_a1 | `SPM_BASE + tile_a` |
| spm_b0 | `SPM_BASE + 2×tile_a` |
| spm_b1 | `SPM_BASE + 2×tile_a + tile_b` |

Verified in P0 #7: A_front=0x0, A_back=+0x400, B_front=+0x800,
B_back=+0xC00 for the 64x64 / 16x16 matmul smoke case.

### 6. End-to-end pipeline integration

- `compiler.py:216-221` inserts the SPM pass and `:303` lowers the DMA ops. Both gated on `_AOT_MODE`. Good.
- `matmul` now exercises the GEMM branch in production. `layer_norm` exercises
  the reduction branch only when `TRITON_ENABLE_SPM_REDUCTIONS=1`, through the
  mean/variance loops and the final 3-load normalize loop (`x`, `gamma`,
  `beta`). `vector_add` remains a designed no-op.

### 7. Tests

- `convert-memory-to-spm.mlir` exercises the pass on synthetic IR that already contains `vector.transfer_read` from `memref` (i.e. the post-`ConvertMemoryOps` form). It does **not** prove that anything from a Triton kernel actually reaches that form. The lit test passing tells us the pattern matches; it does **not** tell us the pattern fires in production.
- Production policy is checked separately by workload verification. Default
  `make verify-layer_norm` now confirms no SPM markers and an empty tier JSON;
  `TRITON_ENABLE_SPM_REDUCTIONS=1` remains the opt-in check for
  mean/variance/final-normalize SPM coverage.

---

## D. End-to-end verification gap — **RESOLVED** (P0 #1)

> ✅ `matmul/kernel.py` rewritten to `tl.make_block_ptr`; LLIR now contains `addrspace(3)` loads and `fence iorw` DMA sequences. See §E P0 #1.

The description below is a historical initial-audit note. Current production
coverage is resolved: `matmul` and `layer_norm` now use block-pointer forms
where needed, and verification checks generated LLIR for SPM markers.

```bash
$ rg 'addrspace\(3\)|fence iorw|store volatile' workloads/*/build/*.llir
# (no matches in matmul.llir or layer_norm.llir)
```

Initial root cause: `workloads/matmul/kernel.py` and `workloads/layer_norm/kernel.py` used **pointer arithmetic** (`a_ptrs += BLOCK_SIZE_K`), not `tl.make_block_ptr`. In `compiler/third_party/cpu/lib/TritonToTritonCPU/ConvertMemoryOps.cpp`, only the `triton::isTensorPointerType(ptr)` branch produces the canonical `vector.transfer_read %memref[%idx]` form (`:179-209`). The pointer-arithmetic branch fell into `lowerToContiguousRowMajor` / `lowerToScalarLoads`, which emitted gather-style or scalar loads — those did not match `findTiledLoads` in `ConvertMemoryToSPM.cpp`.

Two ways forward:

1. Rewrite the workloads to use `tl.make_block_ptr` + `tt.advance` (smallest delta to the compiler).
2. Teach `ConvertMemoryToSPM` to recognise the contiguous-row-major form as well (broader, but harder).

At that point, **until this was fixed nothing in Phase 3 was actually exercised in production** — the unit tests passed on hand-written IR, but the production pipeline was a no-op. This is now fixed for the current `matmul` and `layer_norm` coverage.

---

## E. Concrete Backlog Items

### P0 — blocks any benchmark / paper number

1. ~~**Make the pass actually fire on a real workload.**~~
   ✅ Done. Rewrote `matmul/kernel.py` to `tl.make_block_ptr`. LLIR now contains `addrspace(3)` loads and `fence iorw` DMA sequences.
2. ~~**Fix the reduction single-buffer race**~~ (`ConvertMemoryToSPM.cpp:637-650`)
   ✅ Done. Reordered `enqueue` after `read`.
3. ~~**Reorder GEMM body for real compute/DMA overlap.**~~
   ✅ Done. Prefetch issued before `read_current`/compute; `dma_wait` at end of iteration.
4. ~~**Reconcile the three SPM-size defaults.**~~
   ✅ Done. All unified to 256 KiB (`Passes.td`, `compiler.py`, `env.sh`, `run_spm.py`).
5. ~~**Verify on gem5.**~~
   ✅ Done. matmul / vector_add / layer_norm all PASS in both SPM and cache_baseline modes.
   - DMA plumbing: 128 transfers / 4096 SPM reads / 2048 SPM writes for matmul.
   - **Root cause found** for Bug A (row 14 errors) and Bug B (tile_n=3 errors): both were a single bug — `SpmDmaEngine::translateAddr()` only translated the start of each DMA row read, but in gem5 SE mode consecutive virtual pages are not mapped to consecutive physical pages, so any row read crossing a 4 KiB VA boundary fetched garbage from `PA_start + 0x1000`.
   - **Fix**: added `issuePagedDmaAction()` in `spm_dma_engine.cc` that splits any DMA read/write at VA page boundaries, translates each chunk separately, and fires the row's `doneEvent` once all sub-actions complete. Applied to all four issue sites (1D read, 1D write-back, 2D row read, 2D row write).
   - **Cache-collision fix in Triton**: `make matmul` and `make matmul-nospm` shared the same `~/.triton/cache` because `TRITON_DISABLE_SPM` is not part of `CPUOptions.hash()`. `build_kernel.sh` now exports `TRITON_CACHE_DIR=~/.triton/cache_nospm` for the no-SPM build so the two LLIRs do not collide.
6. ~~**Verify loop boundary guard.**~~
   ✅ Done. Compile-time bail-out added.
7. ~~**Verify SPM buffer layout addresses.**~~
   ✅ Done. Layout confirmed: A_front=0x0, A_back=+0x400, B_front=+0x800, B_back=+0xC00.

### P0+ — start immediately after P0 completes (Phase 3 core claim)

8. ~~**Design Tier classification framework (§3c).**~~
   ✅ Done. `SPMTensorPlacement` pass implements `has_scalar_reuse` analysis, tier decision table, JSON sidecar export, and launcher `_alloc/_free_all` generation. See `three-tier-placement.md` M1-M8.

### P1 — Phase 3 spec compliance

9. ~~**Implement Tier classification (§3c).**~~
   ✅ Done. `SPMTensorPlacement` pass tags each tensor arg with tier 1/2/3. Launcher dispatches to `spm_malloc`/`malloc`/`dma_buf_malloc`. Current default coverage: `matmul` args 0,1 → Tier 3; `layer_norm` and `vector_add` intentionally have no eligible default tiers. `layer_norm` args 0,1,2 → Tier 3 only under `TRITON_ENABLE_SPM_REDUCTIONS=1`. See `three-tier-placement.md` §4.1.
10. **Robust GEMM matcher.**
    - ~~Derive A/B identity from `vector.contract` lhs/rhs operands instead of transfer-read walk order.~~ ✅ Done for the standard GEMM fallback/fused path.
    - ~~Replace `getLoc()`-based cloned-read lookup with `IRMapping`-based lookup.~~ ✅ Done.
    - ~~Allow >2 loads in the loop as long as exactly two feed the contract.~~ ✅ Verified by `@gemm_extra_non_dot_load`.
11. ~~**Generalise reduction matcher** to "any number of loads sharing the loop IV", so LayerNorm's 3-load pass and softmax's typical body are accepted.~~ ✅ Done for canonical `vector.transfer_read` loops. Covered by `@reduction_multi_load_prefetch` and opt-in LayerNorm reduction builds.
12. ~~**Make `DMA_MMIO_BASE` a pass option** on `DmaOpsToLLVM`, and add a `useXspmInsn` boolean (default off) so Phase 4d only needs to flip a flag.~~ ✅ Done. `use-xspm-insn` is present and intentionally errors until the instruction lowering is implemented.

### P1-deferred — explicitly out of Phase 3 scope

13. **SPM writeback for output tile** (C tile in GEMM, O tile in attention). Currently the C tile goes through cacheable `vector.transfer_write`. *(Deferred to Phase 4b per plan — not a Phase 3 deliverable, listed here for traceability.)*

14. ~~**Transformer-facing workload coverage before Phase 4 attention.**~~
    ✅ Done (2026-05-02). Added `activation` (SiLU), `residual_add`, and
    row-wise `softmax` workloads with AOT kernels, harnesses, manifests,
    flushed ROI measurement, result checks, and manifest-driven verify policy.
    `make verify-activation`, `make verify-residual_add`, and
    `make verify-softmax` all confirm clean cache-path policy by default.
    Smoke gem5 compares passed in both SPM-enabled and cache-baseline modes
    (`SIZE=128 BLOCK_SIZE=32` for activation/residual_add; `M=4 N=32
    BLOCK_N=32` for softmax). This closes the workload/harness coverage item;
    it does not add a default SPM pass for elementwise or softmax.

### P2 — Tooling / verification

15. ~~**End-to-end smoke test**~~: ✅ Done. `make verify` / `make verify-<kernel>` builds both modes and checks each manifest's SPM policy: expected SPM kernels must have `addrspace(3)` + `fence iorw` and non-empty tier JSON, while cache-path kernels must remain clean with empty tier JSON.
16. ~~**Cache-baseline harness path.**~~ ✅ Done. `make cmp-<kernel>` runs both SPM and cache modes; `make run-<kernel>` runs SPM only. Unified via `run_experiment.py`.
17. ~~**Stats wiring**~~: ✅ Done. `compare_stats.py` extracts 21 symmetric + 15 SPM-only metrics into `.txt` tables. Current Phase 3.5 compare uses `cache-search` plus `spm-compare` against `cache_best.json`; remaining Phase 6 comparison tooling is separate.

---

## F. P0-5 gem5 verification — matmul 869/4096 computation error investigation (RESOLVED)

### F.1 Earlier fix: VA→PA start-address translation

gem5 SE mode allocates physical pages from a free pool and maps them via the page table. The first iteration of the fix added `translateAddr(Addr vaddr)` using `process->pTable->translate()` to all DMA address calculations. This made `spm_probe` (single-row DMA) pass.

### F.2 Final fix: paged DMA splitting at VA boundaries

After F.1 the simple DMA round-trip worked, but matmul still produced 869/4096 errors. The remaining bug:

**Symptom**: certain rows whose 64-byte transfer crossed a 4 KiB virtual page boundary read garbage in the second half. Concretely, for `pid_m=0`, A's row 14 K-block 3 read `VA [0x86FD0, 0x87010)` which spans pages `0x86000` and `0x87000`. The two pages mapped to `PA 0x86000 → 0x7B000` and `PA 0x87000 → 0x80000` (non-consecutive). The DMA engine translated only `VA 0x86FD0 → PA 0x7BFD0` and then issued a single contiguous 64-byte read, so the last 16 bytes (cols 60–63) came from `PA 0x7C000+`, which contained whatever happened to live in the next physical page rather than the user's data at `PA 0x80000`. The accumulated FMA error showed up across every column of `acc[14]` because the corrupted A elements multiplied with the full B row.

The "tile_n=3" pattern (cols 52–63 wrong for every row) was the same bug at a different VA: the B prefetch for `pid_n=3` reads from `b_ptr + 192` and ran past a page boundary on every K-iteration whose row crossed one.

**Fix** (`spm_dma_engine.cc`):
- New helper `issuePagedDmaAction(cmd, vaddr, len, doneEvent, buf)` splits a single logical DMA action at every 4 KiB VA boundary, translates each chunk separately, and uses an auto-deleting wrapper event with a refcount so the original `doneEvent` fires only after all sub-actions complete.
- Replaced the four direct `dmaPort.dmaAction(...)` call sites: `beginRead()`, `readDone()`, `beginPipelinedRead()` (per-row read), `pipelinedRowReadDone()` (per-row write).

Verified: `make run-matmul-spm` PASS 4096/4096; `make run-vector_add-spm` and `make run-layer_norm-spm` also PASS. SPM stats: 4096 SPM reads, 2048 SPM writes, 128 DMA transfers for matmul.

### F.3 Compiler-side fix: Triton cache collision

`make matmul` and `make matmul-nospm` produced binaries that shared the same `~/.triton/cache` entry because `TRITON_DISABLE_SPM` is not part of `CPUOptions.hash()`. After the fix above, the cache_baseline run still page-faulted at `0xF0000000` because it was running an SPM-enabled binary out of the cache.

**Fix** (`workloads/scripts/build_kernel.sh`): when `--no-spm` is set, also export `TRITON_CACHE_DIR=~/.triton/cache_nospm` so the two cache trees never collide.

### F.4 Probe test files (in `workloads/matmul/`)

| File | Purpose |
|---|---|
| `spm_probe.c` | Basic 1D DMA round-trip (PASS) |
| `row14_probe.c` | 2D DMA 16×16 to SPM with `src_stride=64` packed source — never crossed a page boundary, always passed |
| `rvv_matmul_probe.c` | Pure RVV intrinsic 16×16 matmul, no SPM (PASS) |
| `vrgather_probe.c` | `vrgather.vi` LMUL=2 correctness for indices 0–15 (PASS) |
| `spill_probe.c` | 16 LMUL=2 accumulators under register pressure (PASS) |
| `delta_analysis.c` | Offline hypothesis testing for row-14 error pattern (run natively) |
