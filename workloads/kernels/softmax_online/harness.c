#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "softmax_online_launcher.h"
#include "libspm.h"

/*
 * Test harness for the Triton-compiled exact online row-wise softmax kernel.
 *
 * This kernel fuses canonical max and exp-sum into one online scan, then rereads
 * the row once for normalize/store. Cache and SPM builds use the same Triton
 * source; any SPM traffic comes only from compiler policy/env selection.
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
#ifndef SOFTMAX_ONLINE_WARMUP_ITERS
#define SOFTMAX_ONLINE_WARMUP_ITERS 0
#endif
#ifndef SOFTMAX_ONLINE_MEASURE_ITERS
#define SOFTMAX_ONLINE_MEASURE_ITERS 1
#endif
#ifndef SOFTMAX_ONLINE_FLUSH_BEFORE_ROI
#define SOFTMAX_ONLINE_FLUSH_BEFORE_ROI 1
#endif
#ifndef SOFTMAX_ONLINE_CHECK_RESULT
#define SOFTMAX_ONLINE_CHECK_RESULT 1
#endif
#ifndef SOFTMAX_ONLINE_CAUSAL
#define SOFTMAX_ONLINE_CAUSAL 0
#endif

static volatile int softmax_online_check_result = 1;

#if (N % BLOCK_N) != 0
#error "softmax_online workload requires N to be divisible by BLOCK_N"
#endif

int main(void)
{
    softmax_online_check_result = SOFTMAX_ONLINE_CHECK_RESULT;

    printf("softmax_online: schedule=online  M=%d  N=%d  BLOCK_N=%d  causal=%d  gridX=%d  warmup=%d  measure=%d  flush=%d  check=%d\n",
           M, N, BLOCK_N, SOFTMAX_ONLINE_CAUSAL, M,
           SOFTMAX_ONLINE_WARMUP_ITERS, SOFTMAX_ONLINE_MEASURE_ITERS,
           SOFTMAX_ONLINE_FLUSH_BEFORE_ROI, softmax_online_check_result);

    size_t bytes = (size_t)M * N * sizeof(float);
    float *x_shadow = (float *)malloc(bytes);
    float *x = (float *)softmax_online_alloc(0, bytes);
    float *out = (float *)softmax_online_alloc(1, bytes);
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

    if (SOFTMAX_ONLINE_FLUSH_BEFORE_ROI)
        flush_caches();

    for (int iter = 0; iter < SOFTMAX_ONLINE_WARMUP_ITERS; iter++)
        softmax_online_launch(M, 1, 1, x, out);

    m5_reset_stats(0, 0);

    for (int iter = 0; iter < SOFTMAX_ONLINE_MEASURE_ITERS; iter++)
        softmax_online_launch(M, 1, 1, x, out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (softmax_online_check_result) {
        ref = (float *)malloc(bytes);
        if (!ref) {
            fprintf(stderr, "malloc failed\n");
            free(x_shadow);
            softmax_online_free_all();
            return 1;
        }

        for (int i = 0; i < M; i++) {
            float max_v = -3.4028234663852886e38f;
            for (int j = 0; j < N; j++) {
                if (SOFTMAX_ONLINE_CAUSAL && j > i)
                    continue;
                float v = x_shadow[i * N + j];
                if (v > max_v)
                    max_v = v;
            }

            float denom = 0.0f;
            for (int j = 0; j < N; j++) {
                if (SOFTMAX_ONLINE_CAUSAL && j > i) {
                    ref[i * N + j] = 0.0f;
                    continue;
                }
                float e = expf(x_shadow[i * N + j] - max_v);
                ref[i * N + j] = e;
                denom += e;
            }
            for (int j = 0; j < N; j++) {
                if (SOFTMAX_ONLINE_CAUSAL && j > i)
                    continue;
                ref[i * N + j] /= denom;
            }
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
    softmax_online_free_all();

    return (errors > 0) ? 1 : 0;
}
