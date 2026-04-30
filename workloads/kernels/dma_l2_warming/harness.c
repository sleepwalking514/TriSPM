#ifndef BUF_BYTES
#define BUF_BYTES 4096
#endif

#ifndef SCRUB_BUFFER_BYTES
#define SCRUB_BUFFER_BYTES (4U << 20)
#endif

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
 * Uses random (but deterministic) access patterns to defeat the L2 stride
 * prefetcher — otherwise prefetch hides DRAM latency and masks the warming
 * effect.
 *
 * Phases (each bracketed by m5_reset_stats / m5_dump_stats):
 *   A: DMA cacheable buffer → SPM (warms L2 with src_cached lines)
 *   B: Random read of src_cached (expect L2 hits — warmed by DMA)
 *   C: DMA from UC buffer → SPM (no L2 involvement)
 *   D: Random read of src_cold (expect L2 misses — never warmed)
 *
 * Key comparison: B vs D.  If B has L2 hits where D has misses, L2-warming
 * is confirmed.
 */

#define NUM_FLOATS (BUF_BYTES / sizeof(float))
#define NUM_LINES  (BUF_BYTES / 64)

/* Simple xorshift32 PRNG for deterministic random access. */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Generate a random permutation of cache-line indices [0, NUM_LINES).
 * Each index is touched exactly once → every line accessed once. */
static void generate_permutation(uint32_t *perm, uint32_t n, uint32_t seed)
{
    for (uint32_t i = 0; i < n; i++)
        perm[i] = i;
    uint32_t rng = seed;
    for (uint32_t i = n - 1; i > 0; i--) {
        uint32_t j = xorshift32(&rng) % (i + 1);
        uint32_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}

/* Throttled cache flush: walk the scrub buffer in 8KB chunks with a fence
 * between each chunk to prevent the O3 CPU from overwhelming the L2 crossbar
 * routing table on large working sets. */
static void flush_caches_throttled(void)
{
    static volatile uint8_t scrub[SCRUB_BUFFER_BYTES]
        __attribute__((aligned(CACHE_LINE_BYTES)));
    for (size_t base = 0; base < SCRUB_BUFFER_BYTES; base += 8192) {
        size_t end = base + 8192;
        if (end > SCRUB_BUFFER_BYTES) end = SCRUB_BUFFER_BYTES;
        for (size_t i = base; i < end; i += CACHE_LINE_BYTES)
            scrub[i] = (uint8_t)(scrub[i] + 1);
        asm volatile("fence rw, rw" ::: "memory");
    }
}

/* Random read: access one float from each cache line in permuted order.
 * Fence between accesses prevents the O3 CPU from issuing too many
 * outstanding misses that overflow the L2 crossbar routing table. */
static float random_read(const float *buf, const uint32_t *perm, uint32_t n)
{
    volatile float acc = 0.f;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t line_idx = perm[i];
        acc += buf[line_idx * (64 / sizeof(float))];
        asm volatile("fence r, r" ::: "memory");
    }
    return acc;
}

int main(void)
{
    printf("dma_l2_warming: BUF_BYTES=%d  NUM_LINES=%d\n",
           BUF_BYTES, (int)NUM_LINES);

    float *src_cached = (float *)aligned_alloc(64, BUF_BYTES);
    if (!src_cached) { fprintf(stderr, "malloc failed\n"); return 1; }

    float *src_uc = (float *)dma_buf_malloc(BUF_BYTES);
    if (!src_uc) { fprintf(stderr, "dma_buf_malloc failed\n"); return 1; }

    float *spm_dst = (float *)spm_malloc(BUF_BYTES);
    if (!spm_dst) { fprintf(stderr, "spm_malloc failed\n"); return 1; }

    /* Cold buffer: initialized via CPU, then flushed before measurement. */
    float *src_cold = (float *)aligned_alloc(64, BUF_BYTES);
    if (!src_cold) { fprintf(stderr, "malloc src_cold failed\n"); return 1; }

    /* Random access permutation (same seed for B and D → same pattern). */
    uint32_t *perm = (uint32_t *)malloc(NUM_LINES * sizeof(uint32_t));
    if (!perm) { fprintf(stderr, "malloc perm failed\n"); return 1; }
    generate_permutation(perm, NUM_LINES, 0xDEADBEEF);

    /* Initialize source buffers. */
    for (size_t i = 0; i < NUM_FLOATS; i++) {
        src_cached[i] = (float)i;
        src_cold[i] = (float)i;
    }
    spm_dma_copy(src_uc, src_cached, BUF_BYTES);

    /* ====== Phase A: DMA cacheable buffer → SPM (warms L2) ====== */
    flush_caches_throttled();
    m5_reset_stats(0, 0);

    spm_dma_copy((void *)spm_dst, (void *)src_cached, BUF_BYTES);

    m5_dump_stats(0, 0);

    /* ====== Phase B: Random read of src_cached (should hit L2) ====== */
    m5_reset_stats(0, 0);

    float r1 = random_read(src_cached, perm, NUM_LINES);

    m5_dump_stats(0, 0);

    /* ====== Phase C: DMA from UC buffer → SPM ====== */
    flush_caches_throttled();
    m5_reset_stats(0, 0);

    spm_dma_copy((void *)spm_dst, (void *)src_uc, BUF_BYTES);

    m5_dump_stats(0, 0);

    /* ====== Phase D: Random read of src_cold (L2 cold) ====== */
    flush_caches_throttled();
    m5_reset_stats(0, 0);

    float r2 = random_read(src_cold, perm, NUM_LINES);

    m5_dump_stats(0, 0);

    printf("\nPhase results (prevent optimization): r1=%.2f r2=%.2f\n",
           (double)r1, (double)r2);
    printf("DONE: 4 stat checkpoints written to m5out/stats.txt\n");

    free(src_cached);
    free(src_cold);
    free(perm);
    dma_buf_free_all();
    spm_free_all();
    return 0;
}
