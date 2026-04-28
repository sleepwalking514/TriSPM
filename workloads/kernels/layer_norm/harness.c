#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "layer_norm_launcher.h"

/*
 * Test harness for the Triton-compiled layer_norm kernel.
 *
 * Build with -DM_SIZE=32 -DN_SIZE=64 (rendered from experiment.toml
 * by run_experiment.py).
 */

#ifndef M_SIZE
#error "M_SIZE must be defined via -D flag"
#endif
#ifndef N_SIZE
#error "N_SIZE must be defined via -D flag"
#endif
#ifndef CHECK_RESULT
#define CHECK_RESULT 1
#endif

int main(void)
{
    printf("layer_norm: M=%d  N=%d  check=%d\n", M_SIZE, N_SIZE, CHECK_RESULT);

    float *x     = (float *)layer_norm_alloc(0, M_SIZE * N_SIZE * sizeof(float));
    float *gamma = (float *)layer_norm_alloc(1, N_SIZE * sizeof(float));
    float *beta  = (float *)layer_norm_alloc(2, N_SIZE * sizeof(float));
    float *out   = (float *)layer_norm_alloc(3, M_SIZE * N_SIZE * sizeof(float));

    if (!x || !gamma || !beta || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Deterministic init. */
    for (int i = 0; i < M_SIZE * N_SIZE; i++)
        x[i] = (float)((i % 23) - 11) * 0.1f;
    for (int j = 0; j < N_SIZE; j++) {
        gamma[j] = 0.8f + (float)(j % 5) * 0.1f;   /* 0.8 .. 1.2 */
        beta[j]  = (float)(j % 3) * 0.05f;           /* 0.0, 0.05, 0.10 */
    }
    for (int i = 0; i < M_SIZE * N_SIZE; i++)
        out[i] = 0.0f;

    /* Launch kernel: grid_x = M (one program per row).
     * N_SIZE is passed as a runtime arg (not constexpr) to prevent the
     * Triton compiler from merging loop iterations into LMUL=8 ops. */
    layer_norm_launch(M_SIZE, 1, 1, x, gamma, beta, out, N_SIZE);

#if CHECK_RESULT
    /* Reference layer-norm and verification. */
    int errors = 0;
    for (int i = 0; i < M_SIZE; i++) {
        /* Mean */
        float mean = 0.0f;
        for (int j = 0; j < N_SIZE; j++)
            mean += x[i * N_SIZE + j];
        mean /= N_SIZE;

        /* Variance */
        float var = 0.0f;
        for (int j = 0; j < N_SIZE; j++) {
            float d = x[i * N_SIZE + j] - mean;
            var += d * d;
        }
        var /= N_SIZE;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);

        /* Check each element in the row. */
        for (int j = 0; j < N_SIZE; j++) {
            float expected = (x[i * N_SIZE + j] - mean) * inv_std
                           * gamma[j] + beta[j];
            if (fabsf(out[i * N_SIZE + j] - expected) > 1e-4f) {
                if (errors < 10)
                    printf("MISMATCH [%d,%d]: got %.6f, expected %.6f\n",
                           i, j, out[i * N_SIZE + j], expected);
                errors++;
            }
        }
    }

    if (errors == 0)
        printf("PASS: all %d elements correct\n", M_SIZE * N_SIZE);
    else
        printf("FAIL: %d / %d mismatches\n", errors, M_SIZE * N_SIZE);
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
