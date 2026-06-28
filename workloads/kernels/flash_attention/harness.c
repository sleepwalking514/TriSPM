#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "flash_attention_launcher.h"
#include "libspm.h"

#ifndef BATCH
#define BATCH 1
#endif
#ifndef HEADS
#define HEADS 1
#endif
#ifndef SEQ
#error "SEQ must be defined via -D flag"
#endif
#ifndef HEAD_DIM
#error "HEAD_DIM must be defined via -D flag"
#endif
#ifndef FLASH_BLOCK_M
#define FLASH_BLOCK_M 16
#endif
#ifndef FLASH_BLOCK_N
#define FLASH_BLOCK_N 16
#endif
#ifndef FLASH_ATTENTION_WARMUP_ITERS
#define FLASH_ATTENTION_WARMUP_ITERS 0
#endif
#ifndef FLASH_ATTENTION_MEASURE_ITERS
#define FLASH_ATTENTION_MEASURE_ITERS 1
#endif
#ifndef FLASH_ATTENTION_FLUSH_BEFORE_ROI
#define FLASH_ATTENTION_FLUSH_BEFORE_ROI 1
#endif
#ifndef FLASH_ATTENTION_CHECK_RESULT
#define FLASH_ATTENTION_CHECK_RESULT 1
#endif
#ifndef FLASH_ATTENTION_CAUSAL
#define FLASH_ATTENTION_CAUSAL 1
#endif
#ifndef FLASH_ATTENTION_TOLERANCE
#define FLASH_ATTENTION_TOLERANCE 2e-2f
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define GRID_X CEIL_DIV(SEQ, FLASH_BLOCK_M)
#define GRID_Y (BATCH * HEADS)
#define QKV_ELEMS (BATCH * HEADS * SEQ * HEAD_DIM)

static void init_matrix(float *x, int rows, int cols, int salt)
{
    for (int i = 0; i < rows * cols; i++)
        x[i] = (float)(((i * 7 + salt * 11) % 31) - 15) * 0.05f;
}

static void reference_flash_attention(
    const float *q, const float *k, const float *v, float *out, float sm_scale)
{
    for (int bh = 0; bh < BATCH * HEADS; bh++) {
        int base = bh * SEQ * HEAD_DIM;
        for (int i = 0; i < SEQ; i++) {
            float max_v = -3.4028234663852886e38f;
            for (int j = 0; j < SEQ; j++) {
                if (FLASH_ATTENTION_CAUSAL && j > i)
                    continue;
                float score = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++)
                    score += q[base + i * HEAD_DIM + d] *
                             k[base + j * HEAD_DIM + d];
                score *= sm_scale;
                if (score > max_v)
                    max_v = score;
            }

            float denom = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++)
                out[base + i * HEAD_DIM + d] = 0.0f;

            for (int j = 0; j < SEQ; j++) {
                if (FLASH_ATTENTION_CAUSAL && j > i)
                    continue;
                float score = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++)
                    score += q[base + i * HEAD_DIM + d] *
                             k[base + j * HEAD_DIM + d];
                score *= sm_scale;
                float weight = expf(score - max_v);
                denom += weight;
                for (int d = 0; d < HEAD_DIM; d++)
                    out[base + i * HEAD_DIM + d] +=
                        weight * v[base + j * HEAD_DIM + d];
            }

            for (int d = 0; d < HEAD_DIM; d++)
                out[base + i * HEAD_DIM + d] /= denom;
        }
    }
}

int main(void)
{
    const float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);
    printf("flash_attention: BATCH=%d HEADS=%d SEQ=%d HEAD_DIM=%d BLOCK=%dx%d grid=%dx%d causal=%d scale=%.6g warmup=%d measure=%d flush=%d check=%d tol=%.6g\n",
           BATCH, HEADS, SEQ, HEAD_DIM, FLASH_BLOCK_M, FLASH_BLOCK_N,
           GRID_X, GRID_Y, FLASH_ATTENTION_CAUSAL, (double)sm_scale,
           FLASH_ATTENTION_WARMUP_ITERS, FLASH_ATTENTION_MEASURE_ITERS,
           FLASH_ATTENTION_FLUSH_BEFORE_ROI, FLASH_ATTENTION_CHECK_RESULT,
           (double)FLASH_ATTENTION_TOLERANCE);

    const size_t bytes = (size_t)QKV_ELEMS * sizeof(float);
    float *q_shadow = (float *)malloc(bytes);
    float *k_shadow = (float *)malloc(bytes);
    float *v_shadow = (float *)malloc(bytes);
    float *q = (float *)flash_attention_alloc(0, bytes);
    float *k = (float *)flash_attention_alloc(1, bytes);
    float *v = (float *)flash_attention_alloc(2, bytes);
    float *out = (float *)flash_attention_alloc(3, bytes);

    if (!q_shadow || !k_shadow || !v_shadow || !q || !k || !v || !out) {
        fprintf(stderr, "malloc failed\n");
        free(q_shadow);
        free(k_shadow);
        free(v_shadow);
        flash_attention_free_all();
        return 1;
    }

    init_matrix(q_shadow, BATCH * HEADS * SEQ, HEAD_DIM, 1);
    init_matrix(k_shadow, BATCH * HEADS * SEQ, HEAD_DIM, 2);
    init_matrix(v_shadow, BATCH * HEADS * SEQ, HEAD_DIM, 3);

    flush_caches();
    publish_input(q, q_shadow, bytes);
    publish_input(k, k_shadow, bytes);
    publish_input(v, v_shadow, bytes);
    memset(out, 0, bytes);

    for (int i = 0; i < FLASH_ATTENTION_WARMUP_ITERS; i++) {
        if (FLASH_ATTENTION_FLUSH_BEFORE_ROI)
            flush_caches();
        flash_attention_launch(GRID_X, GRID_Y, 1, q, k, v, out, sm_scale);
    }

    if (FLASH_ATTENTION_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);
    for (int i = 0; i < FLASH_ATTENTION_MEASURE_ITERS; i++)
        flash_attention_launch(GRID_X, GRID_Y, 1, q, k, v, out, sm_scale);
    m5_dump_stats(0, 0);

    int errors = 0;
    if (FLASH_ATTENTION_CHECK_RESULT) {
        float *ref = (float *)malloc(bytes);
        if (!ref) {
            fprintf(stderr, "malloc failed\n");
            free(q_shadow);
            free(k_shadow);
            free(v_shadow);
            flash_attention_free_all();
            return 1;
        }

        reference_flash_attention(q_shadow, k_shadow, v_shadow, ref, sm_scale);
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (int i = 0; i < QKV_ELEMS; i++) {
            float abs_err = fabsf(out[i] - ref[i]);
            float rel_err = abs_err / fmaxf(fabsf(ref[i]), 1e-6f);
            if (abs_err > max_abs)
                max_abs = abs_err;
            if (rel_err > max_rel)
                max_rel = rel_err;
            if (abs_err > FLASH_ATTENTION_TOLERANCE) {
                if (errors < 10) {
                    int bh = i / (SEQ * HEAD_DIM);
                    int local = i % (SEQ * HEAD_DIM);
                    int row = local / HEAD_DIM;
                    int col = i % HEAD_DIM;
                    printf("MISMATCH [bh=%d,%d,%d]: got %.6f, expected %.6f\n",
                           bh, row, col, out[i], ref[i]);
                }
                errors++;
            }
        }
        if (errors == 0)
            printf("PASS: all %d elements correct (max_abs=%.6g max_rel=%.6g)\n",
                   QKV_ELEMS, (double)max_abs, (double)max_rel);
        else
            printf("FAIL: %d / %d mismatches (max_abs=%.6g max_rel=%.6g)\n",
                   errors, QKV_ELEMS, (double)max_abs, (double)max_rel);
        free(ref);
    } else {
        printf("SKIP: result check disabled\n");
    }

    free(q_shadow);
    free(k_shadow);
    free(v_shadow);
    flash_attention_free_all();
    return (errors > 0) ? 1 : 0;
}
