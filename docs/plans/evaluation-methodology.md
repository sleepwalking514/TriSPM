# Evaluation Methodology

> Status: active, 2026-05-05.

## Core Positioning

This project demonstrates **compiler-controlled SPM** on CPU. The value
proposition is: the programmer writes a portable, high-level Triton kernel;
the compiler automatically handles hardware-specific data placement (SPM vs
cache), DMA scheduling, double-buffering, and buffer allocation.

Algorithm choice is the programmer's responsibility. Data placement is the
compiler's responsibility. These are orthogonal concerns.

Analogy: Triton vs hand-written CUDA. Triton doesn't claim to beat expert
CUDA — it claims that a simple kernel + compiler gets close to expert
performance. Similarly, we don't claim SPM compiler beats a perfectly tuned
cache kernel — we claim the compiler automatically exploits SPM hardware from
the same portable kernel.

## Comparison Matrix

For each workload, present four configurations:

| Configuration | What it demonstrates |
|---|---|
| Canonical kernel + cache | Baseline: no SPM hardware benefit |
| Canonical kernel + SPM (compiler auto) | **Compiler value**: same kernel, compiler exploits SPM |
| Tuned kernel + cache | Upper bound: what a smart programmer achieves without SPM |
| Tuned kernel + SPM (compiler auto) | **Hardware + compiler value**: SPM still helps good kernels |

The key comparisons:

1. **SPM hardware value** = canonical+SPM vs canonical+cache
2. **Compiler intelligence** = what the compiler does automatically (DMA,
   double-buffer, CSE-to-SPM) that the programmer didn't write
3. **SPM on tuned kernels** = tuned+SPM vs tuned+cache (proves SPM value is
   not just compensating for a bad algorithm)
4. **Compiler fallback** = canonical+SPM vs tuned+cache (can the compiler
   close the gap between a naive kernel on SPM and a tuned kernel on cache?)

## Compiler Optimization Categories

The compiler pass performs optimizations that are orthogonal to kernel
algorithm quality:

| Optimization | Category | Example |
|---|---|---|
| DMA double-buffering | Data movement scheduling | Overlap compute with prefetch |
| Row-block grouping | Amortization | Reduce DMA descriptor overhead |
| Exp-cache (cross-loop CSE to SPM) | Redundancy elimination | Cache exp(x-max) across loops |
| SPM buffer allocation | Resource management | Fit working set in 256 KiB |
| Profitability gating | Admission control | Reject when SPM overhead > benefit |

Some optimizations (like exp-cache) compensate for naive kernel structure.
If the programmer writes online softmax (2-pass), the compiler's CSE has
nothing to do — just as GCC's CSE pass is idle when the programmer already
eliminated redundant expressions. This is expected and correct behavior.

## Kernel Variants

For softmax specifically:

- **Canonical (3-pass)**: natural Triton-style implementation. Three loops
  over the row: max, exp+sum, normalize+store. This is what a programmer
  writes without thinking about hardware.
- **Online softmax (2-pass)**: fused max+sum using log-sum-exp correction,
  then normalize. Fewer passes = fewer memory reads. This is an algorithmic
  improvement independent of SPM.

Both kernels should be evaluated on both cache and SPM paths to separate
algorithm value from hardware/compiler value.

## What NOT To Do

- Don't only show canonical+SPM vs canonical+cache. This doesn't prove SPM
  helps beyond compensating for a naive algorithm.
- Don't only show tuned+SPM vs tuned+cache. This doesn't show the compiler's
  ability to handle naive code.
- Don't compare tuned+cache vs canonical+SPM as if they're the same thing.
  That conflates algorithm quality with hardware benefit.
- Don't hand-tune the SPM kernel (manual DMA, manual buffer management).
  The point is that the compiler does this automatically.
