#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "matmul_launcher.h"
#include "libspm.h"

/*
 * Test harness for the Triton-compiled matmul kernel.
 *
 * Build with -DM=64 -DN=64 -DK=64 -DBLOCK_SIZE_M=16 -DBLOCK_SIZE_N=16
 * -DBLOCK_SIZE_K=16 (injected by build_kernel.sh from config.sh).
 */

#ifndef M
#error "M must be defined via -D flag"
#endif
#ifndef N
#error "N must be defined via -D flag"
#endif
#ifndef K
#error "K must be defined via -D flag"
#endif
#ifndef BLOCK_SIZE_M
#error "BLOCK_SIZE_M must be defined via -D flag"
#endif
#ifndef BLOCK_SIZE_N
#error "BLOCK_SIZE_N must be defined via -D flag"
#endif

#define GRID_X  (((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) \
               * ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N))

int main(void)
{
    printf("matmul: M=%d  N=%d  K=%d  GRID_X=%d\n", M, N, K, GRID_X);

    /* Cacheable shadows for inputs.  In SPM mode the launcher places A/B
     * in the uncacheable DMA buffer; doing the host-side init and the
     * naive reference on those addresses is extremely slow because every
     * scalar load/store bypasses the cache.  We init/ref on cacheable
     * shadows and DMA-publish into the launcher's buffers in one shot. */
    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)K * N * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    float *a_shadow = (float *)malloc(a_bytes);
    float *b_shadow = (float *)malloc(b_bytes);
    float *a   = (float *)matmul_alloc(0, a_bytes);
    float *b   = (float *)matmul_alloc(1, b_bytes);
    float *c   = (float *)matmul_alloc(2, c_bytes);
    float *ref = (float *)malloc(c_bytes);

    if (!a_shadow || !b_shadow || !a || !b || !c || !ref) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Init: deterministic, small values to avoid fp overflow. */
    for (int i = 0; i < M * K; i++)
        a_shadow[i] = (float)((i % 17) - 8) * 0.1f;
    for (int i = 0; i < K * N; i++)
        b_shadow[i] = (float)((i % 13) - 6) * 0.1f;

    /* Reference matmul on cacheable shadows. */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++)
                sum += a_shadow[i * K + kk] * b_shadow[kk * N + j];
            ref[i * N + j] = sum;
        }

    /* Force shadow + ref dirty lines back to DRAM so the DMA engine
     * reading a_shadow/b_shadow on the next step sees the data the
     * scalar code just wrote.  In a coherent hierarchy gem5's L2 bus
     * snoops L1d, but flushing first makes the contract explicit and
     * also matches the pattern we use for the kernel cold-start. */
    flush_caches();

    /* Publish inputs into the launcher-chosen buffers (DMA-bulk copy
     * when the destination is in the uncacheable DMA buffer, plain
     * memcpy in cache-baseline mode). */
    publish_input(a, a_shadow, a_bytes);
    publish_input(b, b_shadow, b_bytes);
    free(a_shadow);
    free(b_shadow);

    /* Zero output (cacheable in both modes). */
    memset(c, 0, c_bytes);

    /* Cold-cache fair baseline: scrub L1+L2 so SPM and cache modes both
     * face a DRAM-cold starting state in the measured ROI.  Without this
     * the cache baseline gets an unearned warmup bonus from the init /
     * reference phase, which biases comparisons against SPM. */
    flush_caches();

    /* Measure only the Triton kernel ROI; init/ref/publish stay outside. */
    m5_reset_stats(0, 0);

    /* Launch Triton kernel over the 1-D grid. */
    matmul_launch(GRID_X, 1, 1, a, b, c);

    m5_dump_stats(0, 0);

    /* Verify — print per-tile summary to locate which tiles fail. */
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(c[i] - ref[i]) > 1e-3f) {
            if (errors < 20) {
                int row = i / N, col = i % N;
                int tm = row / BLOCK_SIZE_M, tn = col / BLOCK_SIZE_N;
                printf("MISMATCH [%d] (r=%d,c=%d tile_m=%d,tile_n=%d): "
                       "got %.6f, expected %.6f\n",
                       i, row, col, tm, tn, c[i], ref[i]);
            }
            errors++;
        }
    }

    if (errors == 0)
        printf("\nPASS: all %d elements correct\n", M * N);
    else
        printf("\nFAIL: %d / %d mismatches\n", errors, M * N);

    free(ref);
    matmul_free_all();

    return (errors > 0) ? 1 : 0;
}
