#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "layer_norm_launcher.h"
#include "libspm.h"

/*
 * Test harness for the Triton-compiled layer_norm kernel.
 *
 * Build with -DM=32 -DN=64 (rendered from experiment.toml
 * by run_experiment.py).
 */

#ifndef M
#error "M must be defined via -D flag"
#endif
#ifndef N
#error "N must be defined via -D flag"
#endif
#ifndef CHECK_RESULT
#define CHECK_RESULT 1
#endif
#ifndef LAYERNORM_FLUSH_BEFORE_ROI
#define LAYERNORM_FLUSH_BEFORE_ROI 1
#endif

int main(void)
{
    printf("layer_norm: M=%d  N=%d  flush=%d  check=%d\n",
           M, N, LAYERNORM_FLUSH_BEFORE_ROI, CHECK_RESULT);

    size_t x_bytes = (size_t)M * N * sizeof(float);
    size_t param_bytes = (size_t)N * sizeof(float);

    float *x_shadow = (float *)malloc(x_bytes);
    float *gamma_shadow = (float *)malloc(param_bytes);
    float *beta_shadow = (float *)malloc(param_bytes);
    float *x     = (float *)layer_norm_alloc(0, x_bytes);
    float *gamma = (float *)layer_norm_alloc(1, param_bytes);
    float *beta  = (float *)layer_norm_alloc(2, param_bytes);
    float *out   = (float *)layer_norm_alloc(3, x_bytes);
#if CHECK_RESULT
    float *ref = (float *)malloc(x_bytes);
#endif

    if (!x_shadow || !gamma_shadow || !beta_shadow || !x || !gamma || !beta || !out
#if CHECK_RESULT
        || !ref
#endif
    ) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Deterministic init on cacheable shadows.  In SPM mode, x is placed in
     * the uncacheable DMA buffer, so scalar host init/reference there would
     * dominate the measured run and make the comparison unfair. */
    for (int i = 0; i < M * N; i++)
        x_shadow[i] = (float)((i % 23) - 11) * 0.1f;
    for (int j = 0; j < N; j++) {
        gamma_shadow[j] = 0.8f + (float)(j % 5) * 0.1f;   /* 0.8 .. 1.2 */
        beta_shadow[j]  = (float)(j % 3) * 0.05f;         /* 0.0, 0.05, 0.10 */
    }

#if CHECK_RESULT
    /* Reference layer-norm on cacheable shadows. */
    for (int i = 0; i < M; i++) {
        float mean = 0.0f;
        for (int j = 0; j < N; j++)
            mean += x_shadow[i * N + j];
        mean /= N;

        float var = 0.0f;
        for (int j = 0; j < N; j++) {
            float d = x_shadow[i * N + j] - mean;
            var += d * d;
        }
        var /= N;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);

        for (int j = 0; j < N; j++) {
            ref[i * N + j] =
                (x_shadow[i * N + j] - mean) * inv_std
                * gamma_shadow[j] + beta_shadow[j];
        }
    }
#endif

    flush_caches();
    publish_input(x, x_shadow, x_bytes);
    publish_input(gamma, gamma_shadow, param_bytes);
    publish_input(beta, beta_shadow, param_bytes);
    free(x_shadow);
    free(gamma_shadow);
    free(beta_shadow);

    memset(out, 0, x_bytes);

    if (LAYERNORM_FLUSH_BEFORE_ROI)
        flush_caches();

    /* Measure only the Triton kernel ROI; init/ref/publish/check stay outside. */
    m5_reset_stats(0, 0);

    /* Launch kernel: grid_x = M (one program per row). */
    layer_norm_launch(M, 1, 1, x, gamma, beta, out);

    m5_dump_stats(0, 0);

#if CHECK_RESULT
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(out[i] - ref[i]) > 1e-4f) {
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
#else
    printf("SKIP: result check disabled\n");
#endif

    layer_norm_free_all();

#if CHECK_RESULT
    return (errors > 0) ? 1 : 0;
#else
    return 0;
#endif
}
