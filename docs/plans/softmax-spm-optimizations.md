# Softmax SPM Optimizations

> Status: active, 2026-05-05.
> Tracks compiler-pass optimizations for the Softmax row-block DMA SPM path.

## Baseline

The Softmax row-block DMA path (`ROW_BLOCK=2`, `ROW_GROUP_BLOCKS=8`) uses
double-buffered 2D DMA to prefetch row blocks into SPM.  Three inner loops
(max, exp_sum, normalize_store) read `x` from the current SPM buffer.

Before optimization, the normalize pass redundantly recomputes `exp(x - max)`
which was already computed in the exp_sum pass.  This causes ~2x FloatMemRead
inflation vs cache because `x` is read three times from SPM (once per pass)
and the exp computation is duplicated.

## Optimization 1: Exp-Cache (implemented)

**Idea:** Cache `exp(x - max)` results in a third SPM buffer during the
exp_sum pass; read them directly in the normalize pass, eliminating the
redundant x read + sub + exp chain.

**Implementation:** `ConvertMemoryToSPM.cpp`, controlled by env var
`TRITON_SPM_SOFTMAX_CACHE_EXP=1`.

Changes to `cloneLoopBodyWithRowBlockSpm`:
- New parameters: `expSpmBase`, `writeExpToSpm`, `readExpFromSpm`
- exp_sum loop (`writeExpToSpm=true`): after cloning `math::ExpOp`, emit an
  SPM write of the exp result to the exp buffer
- normalize loop (`readExpFromSpm=true`): skip the x read, sub, and exp ops;
  replace with an SPM read from the exp buffer

SPM allocation: a third buffer of the same size as one row-block input buffer
is allocated via `SPMSpaceManager` when the env var is set.

**Results (vs cache best, flushed ROI):**

| Shape | Before (row-block DMA only) | After (+ exp-cache) | Improvement |
|---|---|---|---|
| 64x512 | -37.8% | -58.0% | +20.2pp |
| 128x512 | -37.6% | -58.1% | +20.5pp |
| 128x1024 | -42.8% | -62.8% | +20.0pp |

Correctness verified on all shapes (CHECK_RESULT=1, all elements correct).
No regression on matmul.

**Reproduction:**

```bash
# Build compiler
ninja -C compiler/build/cmake.linux-x86_64-cpython-3.12 triton-opt \
  /home/feige/TriSPM/compiler/python/triton/_C/libtriton.so

# Run softmax with exp-cache (128x1024 example)
TRITON_SPM_SOFTMAX_CACHE_EXP=1 python3 workloads/scripts/run_experiment.py \
  softmax --mode spm-compare \
  --preset phase35-row-block-dma-large-row

# Other shapes: override M and N
TRITON_SPM_SOFTMAX_CACHE_EXP=1 python3 workloads/scripts/run_experiment.py \
  softmax --mode spm-compare \
  --preset phase35-row-block-dma-large-row \
  --override M=128 N=512

TRITON_SPM_SOFTMAX_CACHE_EXP=1 python3 workloads/scripts/run_experiment.py \
  softmax --mode spm-compare \
  --preset phase35-row-block-dma-large-row \
  --override M=64 N=512
```

## Optimization 2: Row-Contiguous SPM Layout (planned)

**Idea:** Current SPM layout uses column-major strides
(`spmMemStrides = {1, trips * vecShape[0]}`), meaning consecutive elements
within one inner-loop iteration are not contiguous in SPM.  Switching to
row-contiguous layout would make each vector load a single contiguous SPM
access, enabling LLVM to generate wider/more efficient vector loads.

**Expected benefit:** Reduce instruction count and improve IPC by eliminating
strided SPM access patterns in the inner loops.

**Status:** Not yet implemented.

## Optimization 3: Fused Max+ExpSum Pass (future)

**Idea:** For cases where the row fits in SPM, fuse the max and exp_sum passes
into a single pass using online softmax (log-sum-exp trick), reducing the
number of SPM reads from 3 to 2.

**Status:** Requires kernel-level changes, not just compiler pass optimization.
Deferred to future work.
