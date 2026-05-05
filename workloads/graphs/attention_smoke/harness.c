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
#ifndef GRAPH_D_MODEL
#define GRAPH_D_MODEL 32
#endif
#ifndef GRAPH_HEAD_DIM
#define GRAPH_HEAD_DIM 16
#endif
#ifndef GRAPH_BLOCK
#define GRAPH_BLOCK 16
#endif
#ifndef GRAPH_CHECK_RESULT
#define GRAPH_CHECK_RESULT 1
#endif
#ifndef GRAPH_FLUSH_BEFORE_ROI
#define GRAPH_FLUSH_BEFORE_ROI 1
#endif

#define LN_ELEMS (GRAPH_SEQ * GRAPH_D_MODEL)
#define QKV_ELEMS (GRAPH_SEQ * GRAPH_HEAD_DIM)
#define SCORE_ELEMS (GRAPH_SEQ * GRAPH_SEQ)
#define ELEMENTWISE_GRID_X ((QKV_ELEMS + GRAPH_BLOCK - 1) / GRAPH_BLOCK)

static void init_matrix(float *x, int rows, int cols, int salt)
{
    for (int i = 0; i < rows * cols; i++)
        x[i] = (float)(((i * 7 + salt * 11) % 31) - 15) * 0.05f;
}

static void init_layer_norm_params(float *gamma, float *beta)
{
    for (int j = 0; j < GRAPH_D_MODEL; j++) {
        gamma[j] = 0.75f + (float)(j % 7) * 0.05f;
        beta[j] = (float)((j % 5) - 2) * 0.025f;
    }
}

static void reference_layer_norm(
    const float *x, const float *gamma, const float *beta, float *out)
{
    for (int i = 0; i < GRAPH_SEQ; i++) {
        float mean = 0.0f;
        for (int j = 0; j < GRAPH_D_MODEL; j++)
            mean += x[i * GRAPH_D_MODEL + j];
        mean /= GRAPH_D_MODEL;

        float var = 0.0f;
        for (int j = 0; j < GRAPH_D_MODEL; j++) {
            float d = x[i * GRAPH_D_MODEL + j] - mean;
            var += d * d;
        }
        var /= GRAPH_D_MODEL;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < GRAPH_D_MODEL; j++) {
            out[i * GRAPH_D_MODEL + j] =
                (x[i * GRAPH_D_MODEL + j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
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

static void reference_transpose(const float *x, float *out, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j * rows + i] = x[i * cols + j];
}

static void reference_softmax(const float *x, float *out)
{
    for (int i = 0; i < GRAPH_SEQ; i++) {
        float max_v = x[i * GRAPH_SEQ];
        for (int j = 1; j < GRAPH_SEQ; j++) {
            float v = x[i * GRAPH_SEQ + j];
            if (v > max_v)
                max_v = v;
        }

        float denom = 0.0f;
        for (int j = 0; j < GRAPH_SEQ; j++) {
            float e = expf(x[i * GRAPH_SEQ + j] - max_v);
            out[i * GRAPH_SEQ + j] = e;
            denom += e;
        }
        for (int j = 0; j < GRAPH_SEQ; j++)
            out[i * GRAPH_SEQ + j] /= denom;
    }
}

static void reference_residual_add(const float *x, const float *residual, float *out)
{
    for (int i = 0; i < QKV_ELEMS; i++)
        out[i] = x[i] + residual[i];
}

static void reference_activation(const float *x, float *out)
{
    for (int i = 0; i < QKV_ELEMS; i++)
        out[i] = x[i] / (1.0f + expf(-x[i]));
}

static int check_tensor(const char *name, const float *got, const float *ref,
                        int elems, float tolerance)
{
    int errors = 0;
    for (int i = 0; i < elems; i++) {
        if (fabsf(got[i] - ref[i]) > tolerance) {
            if (errors < 10) {
                printf("MISMATCH %s[%d]: got %.6f, expected %.6f\n",
                       name, i, got[i], ref[i]);
            }
            errors++;
        }
    }
    return errors;
}

static void free_all_nodes(void)
{
    layer_norm_free_all();
    q_proj_free_all();
    k_proj_free_all();
    v_proj_free_all();
    k_transpose_free_all();
    qk_free_all();
    softmax_free_all();
    pv_free_all();
    residual_add_free_all();
    activation_free_all();
}

int main(void)
{
    printf("graph attention_smoke: SEQ=%d D_MODEL=%d HEAD_DIM=%d BLOCK=%d check=%d flush=%d\n",
           GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_HEAD_DIM, GRAPH_BLOCK,
           GRAPH_CHECK_RESULT, GRAPH_FLUSH_BEFORE_ROI);

    const size_t ln_bytes = (size_t)LN_ELEMS * sizeof(float);
    const size_t qkv_bytes = (size_t)QKV_ELEMS * sizeof(float);
    const size_t score_bytes = (size_t)SCORE_ELEMS * sizeof(float);
    const size_t ln_param_bytes = (size_t)GRAPH_D_MODEL * sizeof(float);
    const size_t qkv_weight_bytes =
        (size_t)GRAPH_D_MODEL * (size_t)GRAPH_HEAD_DIM * sizeof(float);

    float *x_shadow = (float *)malloc(ln_bytes);
    float *gamma_shadow = (float *)malloc(ln_param_bytes);
    float *beta_shadow = (float *)malloc(ln_param_bytes);
    float *wq_shadow = (float *)malloc(qkv_weight_bytes);
    float *wk_shadow = (float *)malloc(qkv_weight_bytes);
    float *wv_shadow = (float *)malloc(qkv_weight_bytes);
    float *residual_shadow = (float *)malloc(qkv_bytes);

    float *x = (float *)layer_norm_alloc(0, ln_bytes);
    float *gamma = (float *)layer_norm_alloc(1, ln_param_bytes);
    float *beta = (float *)layer_norm_alloc(2, ln_param_bytes);
    float *ln_out = (float *)layer_norm_alloc(3, ln_bytes);
    float *wq = (float *)q_proj_alloc(1, qkv_weight_bytes);
    float *wk = (float *)k_proj_alloc(1, qkv_weight_bytes);
    float *wv = (float *)v_proj_alloc(1, qkv_weight_bytes);
    float *q = (float *)q_proj_alloc(2, qkv_bytes);
    float *k = (float *)k_proj_alloc(2, qkv_bytes);
    float *v = (float *)v_proj_alloc(2, qkv_bytes);
    float *k_t = (float *)k_transpose_alloc(1, qkv_bytes);
    float *scores = (float *)qk_alloc(2, score_bytes);
    float *probs = (float *)softmax_alloc(1, score_bytes);
    float *attn = (float *)pv_alloc(2, qkv_bytes);
    float *residual = (float *)residual_add_alloc(1, qkv_bytes);
    float *resid_out = (float *)residual_add_alloc(2, qkv_bytes);
    float *act_out = (float *)activation_alloc(1, qkv_bytes);

    if (!x_shadow || !gamma_shadow || !beta_shadow || !wq_shadow ||
        !wk_shadow || !wv_shadow || !residual_shadow || !x || !gamma ||
        !beta || !ln_out || !wq || !wk || !wv || !q || !k || !v ||
        !k_t || !scores || !probs || !attn || !residual || !resid_out ||
        !act_out) {
        fprintf(stderr, "malloc failed\n");
        free_all_nodes();
        free(x_shadow);
        free(gamma_shadow);
        free(beta_shadow);
        free(wq_shadow);
        free(wk_shadow);
        free(wv_shadow);
        free(residual_shadow);
        return 1;
    }

    init_matrix(x_shadow, GRAPH_SEQ, GRAPH_D_MODEL, 1);
    init_layer_norm_params(gamma_shadow, beta_shadow);
    init_matrix(wq_shadow, GRAPH_D_MODEL, GRAPH_HEAD_DIM, 2);
    init_matrix(wk_shadow, GRAPH_D_MODEL, GRAPH_HEAD_DIM, 3);
    init_matrix(wv_shadow, GRAPH_D_MODEL, GRAPH_HEAD_DIM, 4);
    init_matrix(residual_shadow, GRAPH_SEQ, GRAPH_HEAD_DIM, 5);

    flush_caches();
    publish_input(x, x_shadow, ln_bytes);
    publish_input(gamma, gamma_shadow, ln_param_bytes);
    publish_input(beta, beta_shadow, ln_param_bytes);
    publish_input(wq, wq_shadow, qkv_weight_bytes);
    publish_input(wk, wk_shadow, qkv_weight_bytes);
    publish_input(wv, wv_shadow, qkv_weight_bytes);
    publish_input(residual, residual_shadow, qkv_bytes);

    memset(ln_out, 0, ln_bytes);
    memset(q, 0, qkv_bytes);
    memset(k, 0, qkv_bytes);
    memset(v, 0, qkv_bytes);
    memset(k_t, 0, qkv_bytes);
    memset(scores, 0, score_bytes);
    memset(probs, 0, score_bytes);
    memset(attn, 0, qkv_bytes);
    memset(resid_out, 0, qkv_bytes);
    memset(act_out, 0, qkv_bytes);

    if (GRAPH_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);

    layer_norm_launch(GRAPH_SEQ, 1, 1, x, gamma, beta, ln_out);
    q_proj_launch(1, 1, 1, ln_out, wq, q);
    k_proj_launch(1, 1, 1, ln_out, wk, k);
    v_proj_launch(1, 1, 1, ln_out, wv, v);
    k_transpose_launch(2, 1, 1, k, k_t);
    qk_launch(2, 1, 1, q, k_t, scores);
    softmax_launch(GRAPH_SEQ, 1, 1, scores, probs);
    pv_launch(1, 1, 1, probs, v, attn);
    residual_add_launch(ELEMENTWISE_GRID_X, 1, 1, attn, residual, resid_out);
    activation_launch(ELEMENTWISE_GRID_X, 1, 1, resid_out, act_out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (GRAPH_CHECK_RESULT) {
        float *ln_ref = (float *)malloc(ln_bytes);
        float *q_ref = (float *)malloc(qkv_bytes);
        float *k_ref = (float *)malloc(qkv_bytes);
        float *v_ref = (float *)malloc(qkv_bytes);
        float *k_t_ref = (float *)malloc(qkv_bytes);
        float *scores_ref = (float *)malloc(score_bytes);
        float *probs_ref = (float *)malloc(score_bytes);
        float *attn_ref = (float *)malloc(qkv_bytes);
        float *resid_ref = (float *)malloc(qkv_bytes);
        float *act_ref = (float *)malloc(qkv_bytes);
        if (!ln_ref || !q_ref || !k_ref || !v_ref || !k_t_ref ||
            !scores_ref || !probs_ref || !attn_ref || !resid_ref || !act_ref) {
            fprintf(stderr, "malloc failed\n");
            free(ln_ref);
            free(q_ref);
            free(k_ref);
            free(v_ref);
            free(k_t_ref);
            free(scores_ref);
            free(probs_ref);
            free(attn_ref);
            free(resid_ref);
            free(act_ref);
            free_all_nodes();
            free(x_shadow);
            free(gamma_shadow);
            free(beta_shadow);
            free(wq_shadow);
            free(wk_shadow);
            free(wv_shadow);
            free(residual_shadow);
            return 1;
        }

        reference_layer_norm(x_shadow, gamma_shadow, beta_shadow, ln_ref);
        reference_matmul(ln_ref, wq_shadow, q_ref, GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
        reference_matmul(ln_ref, wk_shadow, k_ref, GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
        reference_matmul(ln_ref, wv_shadow, v_ref, GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
        reference_transpose(k_ref, k_t_ref, GRAPH_SEQ, GRAPH_HEAD_DIM);
        reference_matmul(q_ref, k_t_ref, scores_ref, GRAPH_SEQ, GRAPH_SEQ, GRAPH_HEAD_DIM);
        reference_softmax(scores_ref, probs_ref);
        reference_matmul(probs_ref, v_ref, attn_ref, GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_SEQ);
        reference_residual_add(attn_ref, residual_shadow, resid_ref);
        reference_activation(resid_ref, act_ref);

        errors += check_tensor("act_out", act_out, act_ref, QKV_ELEMS, 1e-2f);
        if (errors == 0)
            printf("PASS: graph outputs correct\n");
        else
            printf("FAIL: graph has %d mismatches\n", errors);

        free(ln_ref);
        free(q_ref);
        free(k_ref);
        free(v_ref);
        free(k_t_ref);
        free(scores_ref);
        free(probs_ref);
        free(attn_ref);
        free(resid_ref);
        free(act_ref);
    } else {
        printf("SKIP: graph result check disabled\n");
    }

    free_all_nodes();
    free(x_shadow);
    free(gamma_shadow);
    free(beta_shadow);
    free(wq_shadow);
    free(wk_shadow);
    free(wv_shadow);
    free(residual_shadow);

    return (errors > 0) ? 1 : 0;
}
