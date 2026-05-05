#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transpose_launcher.h"
#include "libspm.h"

#ifndef M
#error "M must be defined via -D flag"
#endif
#ifndef N
#error "N must be defined via -D flag"
#endif
#ifndef BLOCK_M
#error "BLOCK_M must be defined via -D flag"
#endif
#ifndef BLOCK_N
#error "BLOCK_N must be defined via -D flag"
#endif
#ifndef TRANSPOSE_WARMUP_ITERS
#define TRANSPOSE_WARMUP_ITERS 0
#endif
#ifndef TRANSPOSE_MEASURE_ITERS
#define TRANSPOSE_MEASURE_ITERS 1
#endif
#ifndef TRANSPOSE_FLUSH_BEFORE_ROI
#define TRANSPOSE_FLUSH_BEFORE_ROI 1
#endif
#ifndef TRANSPOSE_CHECK_RESULT
#define TRANSPOSE_CHECK_RESULT 1
#endif

#define GRID_X (((M + BLOCK_M - 1) / BLOCK_M) * ((N + BLOCK_N - 1) / BLOCK_N))

int main(void)
{
    printf("transpose: M=%d N=%d BLOCK=%dx%d GRID_X=%d warmup=%d measure=%d flush=%d check=%d\n",
           M, N, BLOCK_M, BLOCK_N, GRID_X, TRANSPOSE_WARMUP_ITERS,
           TRANSPOSE_MEASURE_ITERS, TRANSPOSE_FLUSH_BEFORE_ROI,
           TRANSPOSE_CHECK_RESULT);

    const size_t elems = (size_t)M * (size_t)N;
    const size_t bytes = elems * sizeof(float);
    float *x_shadow = (float *)malloc(bytes);
    float *x = (float *)transpose_alloc(0, bytes);
    float *out = (float *)transpose_alloc(1, bytes);
    if (!x_shadow || !x || !out) {
        fprintf(stderr, "malloc failed\n");
        free(x_shadow);
        transpose_free_all();
        return 1;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            x_shadow[i * N + j] = (float)(((i * 13 + j * 7) % 29) - 14) * 0.0625f;
    }

    flush_caches();
    publish_input(x, x_shadow, bytes);
    memset(out, 0, bytes);

    if (TRANSPOSE_FLUSH_BEFORE_ROI)
        flush_caches();

    for (int iter = 0; iter < TRANSPOSE_WARMUP_ITERS; iter++)
        transpose_launch(GRID_X, 1, 1, x, out);

    m5_reset_stats(0, 0);

    for (int iter = 0; iter < TRANSPOSE_MEASURE_ITERS; iter++)
        transpose_launch(GRID_X, 1, 1, x, out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (TRANSPOSE_CHECK_RESULT) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float got = out[j * M + i];
                float expected = x_shadow[i * N + j];
                if (fabsf(got - expected) > 1e-6f) {
                    if (errors < 10)
                        printf("MISMATCH [%d,%d]: got %.6f, expected %.6f\n",
                               j, i, got, expected);
                    errors++;
                }
            }
        }
        if (errors == 0)
            printf("PASS: all %zu elements correct\n", elems);
        else
            printf("FAIL: %d / %zu mismatches\n", errors, elems);
    } else {
        printf("SKIP: result check disabled\n");
    }

    free(x_shadow);
    transpose_free_all();
    return (errors > 0) ? 1 : 0;
}
