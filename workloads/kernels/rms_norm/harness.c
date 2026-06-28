#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rms_norm_launcher.h"
#include "libspm.h"

#ifndef M
#error "M must be defined via -D flag"
#endif
#ifndef N
#error "N must be defined via -D flag"
#endif
#ifndef BLOCK_N
#error "BLOCK_N must be defined via -D flag"
#endif
#ifndef RMSNORM_CHECK_RESULT
#define RMSNORM_CHECK_RESULT 1
#endif
#ifndef RMSNORM_FLUSH_BEFORE_ROI
#define RMSNORM_FLUSH_BEFORE_ROI 1
#endif
#ifndef RMSNORM_SPM_ROW_BLOCK
#define RMSNORM_SPM_ROW_BLOCK 1
#endif
#ifndef RMSNORM_SPM_ROW_GROUP_BLOCKS
#define RMSNORM_SPM_ROW_GROUP_BLOCKS 1
#endif
#ifndef RMSNORM_SPM_INTERNAL_ROW_BLOCK
#define RMSNORM_SPM_INTERNAL_ROW_BLOCK 0
#endif

#define RMSNORM_SPM_EFFECTIVE_INTERNAL_ROW_BLOCK \
    (RMSNORM_SPM_INTERNAL_ROW_BLOCK && (BLOCK_N < N))

static volatile int rms_norm_check_result = 1;

#if (N % BLOCK_N) != 0
#error "rms_norm workload requires N to be divisible by BLOCK_N"
#endif
#if RMSNORM_SPM_EFFECTIVE_INTERNAL_ROW_BLOCK
#if RMSNORM_SPM_ROW_BLOCK <= 1
#error "SPM internal row-block rms_norm requires RMSNORM_SPM_ROW_BLOCK > 1"
#endif
#if RMSNORM_SPM_ROW_GROUP_BLOCKS <= 0
#error "SPM internal row-block rms_norm requires RMSNORM_SPM_ROW_GROUP_BLOCKS > 0"
#endif
#if (M % (RMSNORM_SPM_ROW_BLOCK * RMSNORM_SPM_ROW_GROUP_BLOCKS)) != 0
#error "SPM internal row-block rms_norm requires M divisible by row-block * row-group-blocks"
#endif
#endif

static int rms_norm_grid_x(void)
{
#if RMSNORM_SPM_EFFECTIVE_INTERNAL_ROW_BLOCK
    return M / (RMSNORM_SPM_ROW_BLOCK * RMSNORM_SPM_ROW_GROUP_BLOCKS);
#endif
    return M;
}

static const char *rms_norm_schedule_name(void)
{
#if RMSNORM_SPM_EFFECTIVE_INTERNAL_ROW_BLOCK
    return "canonical+spm_row_block";
#else
    return "canonical";
#endif
}

int main(void)
{
    rms_norm_check_result = RMSNORM_CHECK_RESULT;
    int grid_x = rms_norm_grid_x();

    printf("rms_norm: schedule=%s M=%d N=%d BLOCK_N=%d spm_ROW_BLOCK=%d "
           "spm_ROW_GROUP_BLOCKS=%d spm_INTERNAL_ROW_BLOCK=%d gridX=%d "
           "flush=%d check=%d\n",
           rms_norm_schedule_name(), M, N, BLOCK_N, RMSNORM_SPM_ROW_BLOCK,
           RMSNORM_SPM_ROW_GROUP_BLOCKS, RMSNORM_SPM_EFFECTIVE_INTERNAL_ROW_BLOCK,
           grid_x, RMSNORM_FLUSH_BEFORE_ROI, rms_norm_check_result);

    size_t x_bytes = (size_t)M * N * sizeof(float);
    size_t param_bytes = (size_t)N * sizeof(float);

    float *x_shadow = (float *)malloc(x_bytes);
    float *gamma_shadow = (float *)malloc(param_bytes);
    float *x = (float *)rms_norm_alloc(0, x_bytes);
    float *gamma = (float *)rms_norm_alloc(1, param_bytes);
    float *out = (float *)rms_norm_alloc(2, x_bytes);
    float *ref = NULL;

    if (!x_shadow || !gamma_shadow || !x || !gamma || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < M * N; i++)
        x_shadow[i] = (float)((i % 31) - 15) * 0.07f;
    for (int j = 0; j < N; j++)
        gamma_shadow[j] = 0.75f + (float)(j % 7) * 0.08f;

    flush_caches();
    publish_input(x, x_shadow, x_bytes);
    publish_input(gamma, gamma_shadow, param_bytes);
    memset(out, 0, x_bytes);

    if (RMSNORM_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);
    rms_norm_launch(grid_x, 1, 1, x, gamma, out);
    m5_dump_stats(0, 0);

    int errors = 0;
    if (rms_norm_check_result) {
        ref = (float *)malloc(x_bytes);
        if (!ref) {
            fprintf(stderr, "malloc failed\n");
            free(x_shadow);
            free(gamma_shadow);
            rms_norm_free_all();
            return 1;
        }

        for (int i = 0; i < M; i++) {
            float sum_sq = 0.0f;
            for (int j = 0; j < N; j++) {
                float v = x_shadow[i * N + j];
                sum_sq += v * v;
            }
            float inv_rms = 1.0f / sqrtf(sum_sq / (float)N + 1e-5f);
            for (int j = 0; j < N; j++)
                ref[i * N + j] = x_shadow[i * N + j] * inv_rms * gamma_shadow[j];
        }

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
    } else {
        printf("SKIP: result check disabled\n");
    }

    free(x_shadow);
    free(gamma_shadow);
    rms_norm_free_all();

    return (errors > 0) ? 1 : 0;
}
