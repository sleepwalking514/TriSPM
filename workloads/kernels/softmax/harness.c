#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "softmax_launcher.h"
#include "libspm.h"

/*
 * Test harness for the Triton-compiled row-wise softmax kernel.
 *
 * Build with -DM=32 -DN=64 -DBLOCK_N=64, or larger Phase 3.5 rows where
 * N is a multiple of BLOCK_N. Values are rendered from experiment.toml by
 * run_experiment.py. SOFTMAX_SCHEDULE_ID=0 is the canonical one-row schedule;
 * SOFTMAX_SCHEDULE_ID=1 is the row-block schedule used by the DMA experiment.
 */

#ifndef M
#error "M must be defined via -D flag"
#endif
#ifndef N
#error "N must be defined via -D flag"
#endif
#ifndef BLOCK_N
#error "BLOCK_N must be defined via -D flag"
#endif
#ifndef ROW_BLOCK
#define ROW_BLOCK 1
#endif
#ifndef ROW_GROUP_BLOCKS
#define ROW_GROUP_BLOCKS 1
#endif
#ifndef SOFTMAX_WARMUP_ITERS
#define SOFTMAX_WARMUP_ITERS 0
#endif
#ifndef SOFTMAX_MEASURE_ITERS
#define SOFTMAX_MEASURE_ITERS 1
#endif
#ifndef SOFTMAX_FLUSH_BEFORE_ROI
#define SOFTMAX_FLUSH_BEFORE_ROI 1
#endif
#ifndef SOFTMAX_CHECK_RESULT
#define SOFTMAX_CHECK_RESULT 1
#endif
#ifndef SOFTMAX_SCHEDULE_ID
#define SOFTMAX_SCHEDULE_ID 0
#endif

static volatile int softmax_check_result = 1;

#if (N % BLOCK_N) != 0
#error "softmax workload requires N to be divisible by BLOCK_N"
#endif
#if SOFTMAX_SCHEDULE_ID == 0
#if ROW_BLOCK != 1
#error "canonical softmax requires ROW_BLOCK=1"
#endif
#if ROW_GROUP_BLOCKS != 1
#error "canonical softmax requires ROW_GROUP_BLOCKS=1"
#endif
#elif SOFTMAX_SCHEDULE_ID == 1
#if ROW_BLOCK <= 1
#error "row-block softmax requires ROW_BLOCK > 1"
#endif
#if ROW_GROUP_BLOCKS <= 0
#error "row-block softmax requires ROW_GROUP_BLOCKS > 0"
#endif
#if (M % (ROW_BLOCK * ROW_GROUP_BLOCKS)) != 0
#error "softmax row-block workload requires M to be divisible by ROW_BLOCK * ROW_GROUP_BLOCKS"
#endif
#else
#error "unknown SOFTMAX_SCHEDULE_ID"
#endif

static int softmax_grid_x(void)
{
#if SOFTMAX_SCHEDULE_ID == 0
    return M;
#else
    return M / (ROW_BLOCK * ROW_GROUP_BLOCKS);
#endif
}

static const char *softmax_schedule_name(void)
{
#if SOFTMAX_SCHEDULE_ID == 0
    return "canonical";
#else
    return "row_block";
#endif
}

int main(void)
{
    softmax_check_result = SOFTMAX_CHECK_RESULT;
    int grid_x = softmax_grid_x();

    printf("softmax: schedule=%s  M=%d  N=%d  BLOCK_N=%d  ROW_BLOCK=%d  ROW_GROUP_BLOCKS=%d  gridX=%d  warmup=%d  measure=%d  flush=%d  check=%d\n",
           softmax_schedule_name(), M, N, BLOCK_N, ROW_BLOCK, ROW_GROUP_BLOCKS,
           grid_x,
           SOFTMAX_WARMUP_ITERS, SOFTMAX_MEASURE_ITERS, SOFTMAX_FLUSH_BEFORE_ROI,
           softmax_check_result);

    size_t bytes = (size_t)M * N * sizeof(float);
    float *x_shadow = (float *)malloc(bytes);
    float *x = (float *)softmax_alloc(0, bytes);
    float *out = (float *)softmax_alloc(1, bytes);
    float *ref = NULL;

    if (!x_shadow || !x || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            x_shadow[idx] = (float)(((i * 7 + j * 3) % 29) - 14) * 0.1f;
        }
    }

    flush_caches();
    publish_input(x, x_shadow, bytes);
    memset(out, 0, bytes);

    if (SOFTMAX_FLUSH_BEFORE_ROI)
        flush_caches();

    for (int iter = 0; iter < SOFTMAX_WARMUP_ITERS; iter++)
        softmax_launch(grid_x, 1, 1, x, out);

    m5_reset_stats(0, 0);

    for (int iter = 0; iter < SOFTMAX_MEASURE_ITERS; iter++)
        softmax_launch(grid_x, 1, 1, x, out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (softmax_check_result) {
        ref = (float *)malloc(bytes);
        if (!ref) {
            fprintf(stderr, "malloc failed\n");
            free(x_shadow);
            softmax_free_all();
            return 1;
        }

        for (int i = 0; i < M; i++) {
            float max_v = x_shadow[i * N];
            for (int j = 1; j < N; j++) {
                float v = x_shadow[i * N + j];
                if (v > max_v)
                    max_v = v;
            }

            float denom = 0.0f;
            for (int j = 0; j < N; j++) {
                float e = expf(x_shadow[i * N + j] - max_v);
                ref[i * N + j] = e;
                denom += e;
            }
            for (int j = 0; j < N; j++)
                ref[i * N + j] /= denom;
        }

        for (int i = 0; i < M * N; i++) {
            if (fabsf(out[i] - ref[i]) > 1e-3f) {
                if (errors < 10) {
                    int row = i / N, col = i % N;
                    printf("MISMATCH [%d,%d]: got %.6f, expected %.6f\n",
                           row, col, out[i], ref[i]);
                }
                errors++;
            }
        }

        if (errors == 0)
            printf("PASS: all %d elements correct\n", M * N);
        else
            printf("FAIL: %d / %d mismatches\n", errors, M * N);

        free(ref);
    } else {
        printf("SKIP: result check disabled\n");
    }

    free(x_shadow);
    softmax_free_all();

    return (errors > 0) ? 1 : 0;
}
