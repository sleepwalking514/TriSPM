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

    float *a   = (float *)matmul_alloc(0, M * K * sizeof(float));
    float *b   = (float *)matmul_alloc(1, K * N * sizeof(float));
    float *c   = (float *)matmul_alloc(2, M * N * sizeof(float));
    float *ref = (float *)malloc(M * N * sizeof(float));

    if (!a || !b || !c || !ref) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Init: deterministic, small values to avoid fp overflow. */
    for (int i = 0; i < M * K; i++)
        a[i] = (float)((i % 17) - 8) * 0.1f;
    for (int i = 0; i < K * N; i++)
        b[i] = (float)((i % 13) - 6) * 0.1f;
    for (int i = 0; i < M * N; i++)
        c[i] = 0.0f;

    /* Reference matmul (naive triple loop). */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++)
                sum += a[i * K + kk] * b[kk * N + j];
            ref[i * N + j] = sum;
        }

    /* Measure only the Triton kernel ROI; init/ref stay outside this window. */
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
