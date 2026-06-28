#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph_nodes.h"
#include "libspm.h"

#ifndef GRAPH_SEQ
#define GRAPH_SEQ 32
#endif
#ifndef GRAPH_HEAD_DIM
#define GRAPH_HEAD_DIM 16
#endif
#ifndef GRAPH_QK_BLOCK_M
#define GRAPH_QK_BLOCK_M 32
#endif
#ifndef GRAPH_QK_BLOCK_N
#define GRAPH_QK_BLOCK_N 16
#endif
#ifndef GRAPH_QK_BLOCK_K
#define GRAPH_QK_BLOCK_K 8
#endif
#ifndef GRAPH_PV_BLOCK_M
#define GRAPH_PV_BLOCK_M 32
#endif
#ifndef GRAPH_PV_BLOCK_N
#define GRAPH_PV_BLOCK_N 16
#endif
#ifndef GRAPH_PV_BLOCK_K
#define GRAPH_PV_BLOCK_K 16
#endif
#ifndef GRAPH_SOFTMAX_BLOCK_N
#define GRAPH_SOFTMAX_BLOCK_N GRAPH_SEQ
#endif
#ifndef GRAPH_SOFTMAX_ROW_BLOCK
#define GRAPH_SOFTMAX_ROW_BLOCK 1
#endif
#ifndef GRAPH_SOFTMAX_ROW_GROUP_BLOCKS
#define GRAPH_SOFTMAX_ROW_GROUP_BLOCKS 1
#endif
#ifndef GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK
#define GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK 0
#endif
#ifndef GRAPH_CAUSAL
#define GRAPH_CAUSAL 0
#endif
#ifndef GRAPH_WARMUP_ITERS
#define GRAPH_WARMUP_ITERS 0
#endif
#ifndef GRAPH_MEASURE_ITERS
#define GRAPH_MEASURE_ITERS 1
#endif
#ifndef GRAPH_CHECK_RESULT
#define GRAPH_CHECK_RESULT 1
#endif
#ifndef GRAPH_FLUSH_BEFORE_ROI
#define GRAPH_FLUSH_BEFORE_ROI 1
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define SCORE_ELEMS (GRAPH_SEQ * GRAPH_SEQ)
#define QKV_ELEMS (GRAPH_SEQ * GRAPH_HEAD_DIM)
#define MATMUL_GRID(m, n, bm, bn) (CEIL_DIV((m), (bm)) * CEIL_DIV((n), (bn)))
#if GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK
#define SOFTMAX_GRID_X (GRAPH_SEQ / (GRAPH_SOFTMAX_ROW_BLOCK * GRAPH_SOFTMAX_ROW_GROUP_BLOCKS))
#else
#define SOFTMAX_GRID_X GRAPH_SEQ
#endif

#if GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK
#if GRAPH_SOFTMAX_ROW_BLOCK <= 1
#error "SPM internal row-block softmax requires GRAPH_SOFTMAX_ROW_BLOCK > 1"
#endif
#if GRAPH_SOFTMAX_ROW_GROUP_BLOCKS <= 0
#error "SPM internal row-block softmax requires GRAPH_SOFTMAX_ROW_GROUP_BLOCKS > 0"
#endif
#if (GRAPH_SEQ % (GRAPH_SOFTMAX_ROW_BLOCK * GRAPH_SOFTMAX_ROW_GROUP_BLOCKS)) != 0
#error "SPM internal row-block softmax requires GRAPH_SEQ divisible by row-block * row-group-blocks"
#endif
#endif

static void init_matrix(float *x, int rows, int cols, int salt)
{
    for (int i = 0; i < rows * cols; i++)
        x[i] = (float)(((i * 7 + salt * 11) % 31) - 15) * 0.05f;
}

static void scale_matrix(float *x, int elems, float scale)
{
    for (int i = 0; i < elems; i++)
        x[i] *= scale;
}

static void reference_matmul(
    const float *a, const float *b, float *c, int m, int n, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++)
                sum += a[i * k + kk] * b[kk * n + j];
            c[i * n + j] = sum;
        }
    }
}

static void reference_softmax(const float *x, float *out)
{
    for (int i = 0; i < GRAPH_SEQ; i++) {
        float max_v = -3.4028234663852886e38f;
        for (int j = 0; j < GRAPH_SEQ; j++) {
            if (GRAPH_CAUSAL && j > i)
                continue;
            float v = x[i * GRAPH_SEQ + j];
            if (v > max_v)
                max_v = v;
        }

        float denom = 0.0f;
        for (int j = 0; j < GRAPH_SEQ; j++) {
            if (GRAPH_CAUSAL && j > i) {
                out[i * GRAPH_SEQ + j] = 0.0f;
                continue;
            }
            float e = expf(x[i * GRAPH_SEQ + j] - max_v);
            out[i * GRAPH_SEQ + j] = e;
            denom += e;
        }
        for (int j = 0; j < GRAPH_SEQ; j++) {
            if (GRAPH_CAUSAL && j > i)
                continue;
            out[i * GRAPH_SEQ + j] /= denom;
        }
    }
}

static int check_tensor(const char *name, const float *got, const float *ref,
                        int elems, float tolerance)
{
    int errors = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < elems; i++) {
        float abs_err = fabsf(got[i] - ref[i]);
        float rel_err = abs_err / fmaxf(fabsf(ref[i]), 1e-6f);
        if (abs_err > max_abs)
            max_abs = abs_err;
        if (rel_err > max_rel)
            max_rel = rel_err;
        if (abs_err > tolerance) {
            if (errors < 10)
                printf("MISMATCH %s[%d]: got %.6f, expected %.6f\n",
                       name, i, got[i], ref[i]);
            errors++;
        }
    }
    if (errors == 0)
        printf("PASS %s: max_abs=%.6g max_rel=%.6g\n",
               name, (double)max_abs, (double)max_rel);
    else
        printf("FAIL %s: %d / %d mismatches max_abs=%.6g max_rel=%.6g\n",
               name, errors, elems, (double)max_abs, (double)max_rel);
    return errors;
}

static void free_all_nodes(void)
{
    qk_free_all();
    softmax_free_all();
    pv_free_all();
}

static void run_canonical_attention_graph(float *q, float *k_t, float *v,
                                          float *scores, float *probs,
                                          float *out)
{
    qk_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_SEQ,
                          GRAPH_QK_BLOCK_M, GRAPH_QK_BLOCK_N),
              1, 1, q, k_t, scores);
    softmax_launch(SOFTMAX_GRID_X, 1, 1, scores, probs);
    pv_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM,
                          GRAPH_PV_BLOCK_M, GRAPH_PV_BLOCK_N),
              1, 1, probs, v, out);
}

int main(void)
{
    printf("graph canonical_attention: SEQ=%d HEAD_DIM=%d qk_block=%dx%dx%d pv_block=%dx%dx%d softmax_bn=%d softmax_rb=%d softmax_rg=%d causal=%d warmup=%d measure=%d check=%d flush=%d\n",
           GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QK_BLOCK_M, GRAPH_QK_BLOCK_N,
           GRAPH_QK_BLOCK_K, GRAPH_PV_BLOCK_M, GRAPH_PV_BLOCK_N,
           GRAPH_PV_BLOCK_K, GRAPH_SOFTMAX_BLOCK_N, GRAPH_SOFTMAX_ROW_BLOCK,
           GRAPH_SOFTMAX_ROW_GROUP_BLOCKS, GRAPH_CAUSAL,
           GRAPH_WARMUP_ITERS, GRAPH_MEASURE_ITERS, GRAPH_CHECK_RESULT,
           GRAPH_FLUSH_BEFORE_ROI);

    const float sm_scale = 1.0f / sqrtf((float)GRAPH_HEAD_DIM);
    const size_t qkv_bytes = (size_t)QKV_ELEMS * sizeof(float);
    const size_t score_bytes = (size_t)SCORE_ELEMS * sizeof(float);

    float *q_shadow = (float *)malloc(qkv_bytes);
    float *k_t_shadow = (float *)malloc(qkv_bytes);
    float *v_shadow = (float *)malloc(qkv_bytes);

    float *q = (float *)qk_alloc(0, qkv_bytes);
    float *k_t = (float *)qk_alloc(1, qkv_bytes);
    float *scores = (float *)qk_alloc(2, score_bytes);
    float *probs = (float *)softmax_alloc(1, score_bytes);
    float *v = (float *)pv_alloc(1, qkv_bytes);
    float *out = (float *)pv_alloc(2, qkv_bytes);

    if (!q_shadow || !k_t_shadow || !v_shadow || !q || !k_t || !scores ||
        !probs || !v || !out) {
        fprintf(stderr, "malloc failed\n");
        free(q_shadow);
        free(k_t_shadow);
        free(v_shadow);
        free_all_nodes();
        return 1;
    }

    init_matrix(q_shadow, GRAPH_SEQ, GRAPH_HEAD_DIM, 1);
    init_matrix(k_t_shadow, GRAPH_HEAD_DIM, GRAPH_SEQ, 2);
    init_matrix(v_shadow, GRAPH_SEQ, GRAPH_HEAD_DIM, 3);
    scale_matrix(q_shadow, QKV_ELEMS, sm_scale);

    flush_caches();
    publish_input(q, q_shadow, qkv_bytes);
    publish_input(k_t, k_t_shadow, qkv_bytes);
    publish_input(v, v_shadow, qkv_bytes);
    memset(scores, 0, score_bytes);
    memset(probs, 0, score_bytes);
    memset(out, 0, qkv_bytes);

    for (int iter = 0; iter < GRAPH_WARMUP_ITERS; iter++) {
        if (GRAPH_FLUSH_BEFORE_ROI)
            flush_caches();
        run_canonical_attention_graph(q, k_t, v, scores, probs, out);
    }

    if (GRAPH_FLUSH_BEFORE_ROI)
        flush_caches();
    m5_reset_stats(0, 0);

    for (int iter = 0; iter < GRAPH_MEASURE_ITERS; iter++)
        run_canonical_attention_graph(q, k_t, v, scores, probs, out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (GRAPH_CHECK_RESULT) {
        float *scores_ref = (float *)malloc(score_bytes);
        float *probs_ref = (float *)malloc(score_bytes);
        float *out_ref = (float *)malloc(qkv_bytes);
        if (!scores_ref || !probs_ref || !out_ref) {
            fprintf(stderr, "malloc failed\n");
            free(scores_ref);
            free(probs_ref);
            free(out_ref);
            free(q_shadow);
            free(k_t_shadow);
            free(v_shadow);
            free_all_nodes();
            return 1;
        }

        reference_matmul(q_shadow, k_t_shadow, scores_ref,
                         GRAPH_SEQ, GRAPH_SEQ, GRAPH_HEAD_DIM);
        reference_softmax(scores_ref, probs_ref);
        reference_matmul(probs_ref, v_shadow, out_ref,
                         GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_SEQ);
        errors += check_tensor("out", out, out_ref, QKV_ELEMS, 2e-2f);
        if (errors == 0)
            printf("PASS: graph outputs correct\n");
        else
            printf("FAIL: graph has %d mismatches\n", errors);

        free(scores_ref);
        free(probs_ref);
        free(out_ref);
    } else {
        printf("SKIP: graph result check disabled\n");
    }

    free(q_shadow);
    free(k_t_shadow);
    free(v_shadow);
    free_all_nodes();
    return (errors > 0) ? 1 : 0;
}
