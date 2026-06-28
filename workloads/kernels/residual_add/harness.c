#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "residual_add_launcher.h"
#include "libspm.h"

/*
 * Test harness for the Triton-compiled residual_add kernel.
 *
 * Build with -DSIZE=4096 -DBLOCK_SIZE=64 (rendered from experiment.toml by
 * run_experiment.py).
 */

#ifndef SIZE
#error "SIZE must be defined via -D flag"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined via -D flag"
#endif
#ifndef RESIDUAL_ADD_WARMUP_ITERS
#define RESIDUAL_ADD_WARMUP_ITERS 0
#endif
#ifndef RESIDUAL_ADD_MEASURE_ITERS
#define RESIDUAL_ADD_MEASURE_ITERS 1
#endif
#ifndef RESIDUAL_ADD_FLUSH_BEFORE_ROI
#define RESIDUAL_ADD_FLUSH_BEFORE_ROI 1
#endif
#ifndef RESIDUAL_ADD_CHECK_RESULT
#define RESIDUAL_ADD_CHECK_RESULT 1
#endif

#define GRID_X  ((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE)

static volatile int residual_add_check_result = 1;

int main(void)
{
    residual_add_check_result = RESIDUAL_ADD_CHECK_RESULT;

    printf("residual_add: SIZE=%d  BLOCK_SIZE=%d  GRID_X=%d  warmup=%d  measure=%d  flush=%d  check=%d\n",
           SIZE, BLOCK_SIZE, GRID_X, RESIDUAL_ADD_WARMUP_ITERS,
           RESIDUAL_ADD_MEASURE_ITERS, RESIDUAL_ADD_FLUSH_BEFORE_ROI,
           residual_add_check_result);

    size_t bytes = (size_t)SIZE * sizeof(float);
    float *x_shadow = (float *)malloc(bytes);
    float *residual_shadow = (float *)malloc(bytes);
    float *x = (float *)residual_add_alloc(0, bytes);
    float *residual = (float *)residual_add_alloc(1, bytes);
    float *out = (float *)residual_add_alloc(2, bytes);
    float *ref = NULL;

    if (!x_shadow || !residual_shadow || !x || !residual || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < SIZE; i++) {
        x_shadow[i] = (float)((i % 19) - 9) * 0.125f;
        residual_shadow[i] = (float)((i % 23) - 11) * 0.0625f;
    }

    flush_caches();
    publish_input(x, x_shadow, bytes);
    publish_input(residual, residual_shadow, bytes);
    memset(out, 0, bytes);

    if (RESIDUAL_ADD_FLUSH_BEFORE_ROI)
        flush_caches();

    for (int iter = 0; iter < RESIDUAL_ADD_WARMUP_ITERS; iter++)
        residual_add_launch(GRID_X, 1, 1, x, residual, out);

    m5_reset_stats(0, 0);

    for (int iter = 0; iter < RESIDUAL_ADD_MEASURE_ITERS; iter++)
        residual_add_launch(GRID_X, 1, 1, x, residual, out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (residual_add_check_result) {
        ref = (float *)malloc(bytes);
        if (!ref) {
            fprintf(stderr, "malloc failed\n");
            free(x_shadow);
            free(residual_shadow);
            residual_add_free_all();
            return 1;
        }

        for (int i = 0; i < SIZE; i++)
            ref[i] = x_shadow[i] + residual_shadow[i];

        for (int i = 0; i < SIZE; i++) {
            if (fabsf(out[i] - ref[i]) > 1e-5f) {
                if (errors < 10)
                    printf("MISMATCH [%d]: got %.6f, expected %.6f\n",
                           i, out[i], ref[i]);
                errors++;
            }
        }

        if (errors == 0)
            printf("PASS: all %d elements correct\n", SIZE);
        else
            printf("FAIL: %d / %d mismatches\n", errors, SIZE);

        free(ref);
    } else {
        printf("SKIP: result check disabled\n");
    }

    free(x_shadow);
    free(residual_shadow);
    residual_add_free_all();

    return (errors > 0) ? 1 : 0;
}
