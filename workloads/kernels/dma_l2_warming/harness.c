#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "libspm.h"

/*
 * L2-warming microbenchmark.
 *
 * Demonstrates that DMA from a cacheable address warms L2, so subsequent
 * scalar reads hit L2 instead of going to DRAM.  This is the empirical
 * evidence for the Tier 2 placement policy.
 *
 * Phases (each bracketed by m5_reset_stats / m5_dump_stats):
 *   A: DMA cacheable buffer → SPM (expect L2 misses + allocation)
 *   B: Scalar read of same cacheable buffer (expect L2 hits)
 *   C: DMA from UC buffer → SPM (expect zero L2 accesses)
 *   D: Scalar read of cacheable buffer WITHOUT prior DMA (expect L2 misses)
 *
 * Key comparison: B vs D.  If B has L2 hits where D has misses, L2-warming
 * is confirmed.
 *
 * Build: cross-compile with CLANG for RISC-V, link with libspm.h.
 *        No Triton kernel needed.
 *
 * Config macros (via -D):
 *   BUF_BYTES  — tile size in bytes (default 4096 = 1 KiB of floats)
 */

#ifndef BUF_BYTES
#define BUF_BYTES 4096
#endif

#define NUM_FLOATS (BUF_BYTES / sizeof(float))

int main(void)
{
    printf("dma_l2_warming: BUF_BYTES=%d  NUM_FLOATS=%d\n",
           BUF_BYTES, (int)NUM_FLOATS);

    /* Cacheable source buffer (malloc'd in normal DRAM range). */
    float *src_cached = (float *)aligned_alloc(64, BUF_BYTES);
    if (!src_cached) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* Uncacheable source buffer (in DMA_BUF region). */
    float *src_uc = (float *)dma_buf_malloc(BUF_BYTES);
    if (!src_uc) { fprintf(stderr, "dma_buf_malloc failed\n"); return 1; }

    /* SPM destination. */
    float *spm_dst = (float *)spm_malloc(BUF_BYTES);
    if (!spm_dst) { fprintf(stderr, "spm_malloc failed\n"); return 1; }

    /* Initialize both source buffers. */
    for (size_t i = 0; i < NUM_FLOATS; i++) {
        src_cached[i] = (float)i;
    }
    /* UC buffer: write via DMA from a temp cacheable copy. */
    spm_dma_copy(src_uc, src_cached, BUF_BYTES);

    /* ====== Phase A: DMA cacheable buffer → SPM ====== */
    flush_caches();
    m5_reset_stats(0, 0);

    spm_dma_copy((void *)spm_dst, (void *)src_cached, BUF_BYTES);

    m5_dump_stats(0, 0);
    /* Checkpoint A: expect l2cache misses from DMA + allocation */

    /* ====== Phase B: Scalar read of src_cached (should hit L2) ====== */
    m5_reset_stats(0, 0);

    volatile float acc = 0.f;
    for (size_t i = 0; i < NUM_FLOATS; i++)
        acc += src_cached[i];

    m5_dump_stats(0, 0);
    /* Checkpoint B: expect l2cache hits (warmed by DMA in phase A) */

    /* ====== Phase C: DMA from UC buffer → SPM ====== */
    flush_caches();
    m5_reset_stats(0, 0);

    spm_dma_copy((void *)spm_dst, (void *)src_uc, BUF_BYTES);

    m5_dump_stats(0, 0);
    /* Checkpoint C: expect zero l2cache accesses (UC bypasses L2) */

    /* ====== Phase D: Scalar read WITHOUT prior DMA (cold baseline) ====== */
    flush_caches();
    m5_reset_stats(0, 0);

    volatile float acc2 = 0.f;
    for (size_t i = 0; i < NUM_FLOATS; i++)
        acc2 += src_cached[i];

    m5_dump_stats(0, 0);
    /* Checkpoint D: expect l2cache misses (no warming) */

    printf("\nPhase results (prevent optimization): acc=%.2f acc2=%.2f\n",
           (double)acc, (double)acc2);
    printf("DONE: 4 stat checkpoints written to m5out/stats.txt\n");

    free(src_cached);
    dma_buf_free_all();
    spm_free_all();
    return 0;
}
