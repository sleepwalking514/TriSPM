#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "activation_launcher.h"
#include "libspm.h"

/*
 * Test harness for the Triton-compiled activation kernel.
 *
 * Build with -DSIZE=4096 -DBLOCK_SIZE=64 (rendered from experiment.toml by
 * run_experiment.py).  The activation is SiLU: y = x / (1 + exp(-x)).
 */

#ifndef SIZE
#error "SIZE must be defined via -D flag"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined via -D flag"
#endif
#ifndef ACTIVATION_WARMUP_ITERS
#define ACTIVATION_WARMUP_ITERS 0
#endif
#ifndef ACTIVATION_MEASURE_ITERS
#define ACTIVATION_MEASURE_ITERS 1
#endif
#ifndef ACTIVATION_FLUSH_BEFORE_ROI
#define ACTIVATION_FLUSH_BEFORE_ROI 1
#endif
#ifndef ACTIVATION_CHECK_RESULT
#define ACTIVATION_CHECK_RESULT 1
#endif

#define GRID_X  ((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE)

static float silu_ref(float x)
{
    return x / (1.0f + expf(-x));
}

int main(void)
{
    printf("activation(silu): SIZE=%d  BLOCK_SIZE=%d  GRID_X=%d  warmup=%d  measure=%d  flush=%d  check=%d\n",
           SIZE, BLOCK_SIZE, GRID_X, ACTIVATION_WARMUP_ITERS,
           ACTIVATION_MEASURE_ITERS, ACTIVATION_FLUSH_BEFORE_ROI,
           ACTIVATION_CHECK_RESULT);

    size_t bytes = (size_t)SIZE * sizeof(float);
    float *x_shadow = (float *)malloc(bytes);
    float *x = (float *)activation_alloc(0, bytes);
    float *out = (float *)activation_alloc(1, bytes);
#if ACTIVATION_CHECK_RESULT
    float *ref = (float *)malloc(bytes);
#endif

    if (!x_shadow || !x || !out
#if ACTIVATION_CHECK_RESULT
        || !ref
#endif
    ) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < SIZE; i++) {
        float v = (float)((i % 37) - 18) / 6.0f;
        x_shadow[i] = v;
#if ACTIVATION_CHECK_RESULT
        ref[i] = silu_ref(v);
#endif
    }

    flush_caches();
    publish_input(x, x_shadow, bytes);
    free(x_shadow);
    memset(out, 0, bytes);

    if (ACTIVATION_FLUSH_BEFORE_ROI)
        flush_caches();

    for (int iter = 0; iter < ACTIVATION_WARMUP_ITERS; iter++)
        activation_launch(GRID_X, 1, 1, x, out);

    m5_reset_stats(0, 0);

    for (int iter = 0; iter < ACTIVATION_MEASURE_ITERS; iter++)
        activation_launch(GRID_X, 1, 1, x, out);

    m5_dump_stats(0, 0);

#if ACTIVATION_CHECK_RESULT
    int errors = 0;
    for (int i = 0; i < SIZE; i++) {
        if (fabsf(out[i] - ref[i]) > 1e-3f) {
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
#else
    printf("SKIP: result check disabled\n");
#endif

    activation_free_all();

#if ACTIVATION_CHECK_RESULT
    return (errors > 0) ? 1 : 0;
#else
    return 0;
#endif
}
