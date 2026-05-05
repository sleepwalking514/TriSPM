#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activation_launcher.h"
#include "layer_norm_launcher.h"
#include "matmul_launcher.h"
#include "residual_add_launcher.h"
#include "softmax_launcher.h"
#include "libspm.h"

#ifndef GRAPH_DIM
#define GRAPH_DIM 32
#endif
#ifndef GRAPH_BLOCK
#define GRAPH_BLOCK 32
#endif
#ifndef GRAPH_CHECK_RESULT
#define GRAPH_CHECK_RESULT 1
#endif
#ifndef GRAPH_FLUSH_BEFORE_ROI
#define GRAPH_FLUSH_BEFORE_ROI 1
#endif

#if GRAPH_DIM != GRAPH_BLOCK
#error "attention_smoke uses one square block so all matmul nodes share one AOT symbol"
#endif

#define ELEM_COUNT (GRAPH_DIM * GRAPH_DIM)
#define BYTES ((size_t)ELEM_COUNT * sizeof(float))
#define MATMUL_GRID_X 1
#define SOFTMAX_GRID_X GRAPH_DIM
#define ELEMENTWISE_GRID_X ((ELEM_COUNT + GRAPH_BLOCK - 1) / GRAPH_BLOCK)

static void init_matrix(float *x, int salt)
{
    for (int i = 0; i < ELEM_COUNT; i++)
        x[i] = (float)(((i * 7 + salt * 11) % 31) - 15) * 0.05f;
}

static void init_layer_norm_params(float *gamma, float *beta)
{
    for (int j = 0; j < GRAPH_DIM; j++) {
        gamma[j] = 0.75f + (float)(j % 7) * 0.05f;
        beta[j] = (float)((j % 5) - 2) * 0.025f;
    }
}

static void reference_layer_norm(
    const float *x, const float *gamma, const float *beta, float *out)
{
    for (int i = 0; i < GRAPH_DIM; i++) {
        float mean = 0.0f;
        for (int j = 0; j < GRAPH_DIM; j++)
            mean += x[i * GRAPH_DIM + j];
        mean /= GRAPH_DIM;

        float var = 0.0f;
        for (int j = 0; j < GRAPH_DIM; j++) {
            float d = x[i * GRAPH_DIM + j] - mean;
            var += d * d;
        }
        var /= GRAPH_DIM;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < GRAPH_DIM; j++) {
            out[i * GRAPH_DIM + j] =
                (x[i * GRAPH_DIM + j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

static void reference_matmul(const float *a, const float *b, float *c)
{
    for (int i = 0; i < GRAPH_DIM; i++) {
        for (int j = 0; j < GRAPH_DIM; j++) {
            float sum = 0.0f;
            for (int k = 0; k < GRAPH_DIM; k++)
                sum += a[i * GRAPH_DIM + k] * b[k * GRAPH_DIM + j];
            c[i * GRAPH_DIM + j] = sum;
        }
    }
}

static void reference_softmax(const float *x, float *out)
{
    for (int i = 0; i < GRAPH_DIM; i++) {
        float max_v = x[i * GRAPH_DIM];
        for (int j = 1; j < GRAPH_DIM; j++) {
            float v = x[i * GRAPH_DIM + j];
            if (v > max_v)
                max_v = v;
        }

        float denom = 0.0f;
        for (int j = 0; j < GRAPH_DIM; j++) {
            float e = expf(x[i * GRAPH_DIM + j] - max_v);
            out[i * GRAPH_DIM + j] = e;
            denom += e;
        }
        for (int j = 0; j < GRAPH_DIM; j++)
            out[i * GRAPH_DIM + j] /= denom;
    }
}

static void reference_residual_add(const float *x, const float *residual, float *out)
{
    for (int i = 0; i < ELEM_COUNT; i++)
        out[i] = x[i] + residual[i];
}

static void reference_activation(const float *x, float *out)
{
    for (int i = 0; i < ELEM_COUNT; i++)
        out[i] = x[i] / (1.0f + expf(-x[i]));
}

static int check_tensor(const char *name, const float *got, const float *ref,
                        float tolerance)
{
    int errors = 0;
    for (int i = 0; i < ELEM_COUNT; i++) {
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

int main(void)
{
    printf("graph attention_smoke: DIM=%d BLOCK=%d check=%d flush=%d\n",
           GRAPH_DIM, GRAPH_BLOCK, GRAPH_CHECK_RESULT, GRAPH_FLUSH_BEFORE_ROI);

    const size_t param_bytes = (size_t)GRAPH_DIM * sizeof(float);

    float *x_shadow = (float *)malloc(BYTES);
    float *gamma_shadow = (float *)malloc(param_bytes);
    float *beta_shadow = (float *)malloc(param_bytes);
    float *wq_shadow = (float *)malloc(BYTES);
    float *wk_shadow = (float *)malloc(BYTES);
    float *wv_shadow = (float *)malloc(BYTES);
    float *residual_shadow = (float *)malloc(BYTES);

    float *x = (float *)layer_norm_alloc(0, BYTES);
    float *gamma = (float *)layer_norm_alloc(1, param_bytes);
    float *beta = (float *)layer_norm_alloc(2, param_bytes);
    float *ln_out = (float *)layer_norm_alloc(3, BYTES);
    float *wq = (float *)matmul_alloc(1, BYTES);
    float *wk = (float *)matmul_alloc(1, BYTES);
    float *wv = (float *)matmul_alloc(1, BYTES);
    float *q = (float *)matmul_alloc(2, BYTES);
    float *k = (float *)matmul_alloc(2, BYTES);
    float *v = (float *)matmul_alloc(2, BYTES);
    float *scores = (float *)matmul_alloc(2, BYTES);
    float *probs = (float *)softmax_alloc(1, BYTES);
    float *attn = (float *)matmul_alloc(2, BYTES);
    float *residual = (float *)residual_add_alloc(1, BYTES);
    float *resid_out = (float *)residual_add_alloc(2, BYTES);
    float *act_out = (float *)activation_alloc(1, BYTES);

    if (!x_shadow || !gamma_shadow || !beta_shadow || !wq_shadow ||
        !wk_shadow || !wv_shadow || !residual_shadow || !x || !gamma ||
        !beta || !ln_out || !wq || !wk || !wv || !q || !k || !v ||
        !scores || !probs || !attn || !residual || !resid_out || !act_out) {
        fprintf(stderr, "malloc failed\n");
        layer_norm_free_all();
        matmul_free_all();
        softmax_free_all();
        residual_add_free_all();
        activation_free_all();
        free(x_shadow);
        free(gamma_shadow);
        free(beta_shadow);
        free(wq_shadow);
        free(wk_shadow);
        free(wv_shadow);
        free(residual_shadow);
        return 1;
    }

    init_matrix(x_shadow, 1);
    init_layer_norm_params(gamma_shadow, beta_shadow);
    init_matrix(wq_shadow, 2);
    init_matrix(wk_shadow, 3);
    init_matrix(wv_shadow, 4);
    init_matrix(residual_shadow, 5);

    flush_caches();
    publish_input(x, x_shadow, BYTES);
    publish_input(gamma, gamma_shadow, param_bytes);
    publish_input(beta, beta_shadow, param_bytes);
    publish_input(wq, wq_shadow, BYTES);
    publish_input(wk, wk_shadow, BYTES);
    publish_input(wv, wv_shadow, BYTES);
    publish_input(residual, residual_shadow, BYTES);

    memset(ln_out, 0, BYTES);
    memset(q, 0, BYTES);
    memset(k, 0, BYTES);
    memset(v, 0, BYTES);
    memset(scores, 0, BYTES);
    memset(probs, 0, BYTES);
    memset(attn, 0, BYTES);
    memset(resid_out, 0, BYTES);
    memset(act_out, 0, BYTES);

    if (GRAPH_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);

    layer_norm_launch(GRAPH_DIM, 1, 1, x, gamma, beta, ln_out);
    matmul_launch(MATMUL_GRID_X, 1, 1, ln_out, wq, q);
    matmul_launch(MATMUL_GRID_X, 1, 1, ln_out, wk, k);
    matmul_launch(MATMUL_GRID_X, 1, 1, ln_out, wv, v);
    matmul_launch(MATMUL_GRID_X, 1, 1, q, k, scores);
    softmax_launch(SOFTMAX_GRID_X, 1, 1, scores, probs);
    matmul_launch(MATMUL_GRID_X, 1, 1, probs, v, attn);
    residual_add_launch(ELEMENTWISE_GRID_X, 1, 1, attn, residual, resid_out);
    activation_launch(ELEMENTWISE_GRID_X, 1, 1, resid_out, act_out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (GRAPH_CHECK_RESULT) {
        float *ln_ref = (float *)malloc(BYTES);
        float *q_ref = (float *)malloc(BYTES);
        float *k_ref = (float *)malloc(BYTES);
        float *v_ref = (float *)malloc(BYTES);
        float *scores_ref = (float *)malloc(BYTES);
        float *probs_ref = (float *)malloc(BYTES);
        float *attn_ref = (float *)malloc(BYTES);
        float *resid_ref = (float *)malloc(BYTES);
        float *act_ref = (float *)malloc(BYTES);
        if (!ln_ref || !q_ref || !k_ref || !v_ref || !scores_ref ||
            !probs_ref || !attn_ref || !resid_ref || !act_ref) {
            fprintf(stderr, "malloc failed\n");
            free(ln_ref);
            free(q_ref);
            free(k_ref);
            free(v_ref);
            free(scores_ref);
            free(probs_ref);
            free(attn_ref);
            free(resid_ref);
            free(act_ref);
            layer_norm_free_all();
            matmul_free_all();
            softmax_free_all();
            residual_add_free_all();
            activation_free_all();
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
        reference_matmul(ln_ref, wq_shadow, q_ref);
        reference_matmul(ln_ref, wk_shadow, k_ref);
        reference_matmul(ln_ref, wv_shadow, v_ref);
        reference_matmul(q_ref, k_ref, scores_ref);
        reference_softmax(scores_ref, probs_ref);
        reference_matmul(probs_ref, v_ref, attn_ref);
        reference_residual_add(attn_ref, residual_shadow, resid_ref);
        reference_activation(resid_ref, act_ref);

        errors += check_tensor("act_out", act_out, act_ref, 1e-2f);
        if (errors == 0)
            printf("PASS: graph outputs correct\n");
        else
            printf("FAIL: graph has %d mismatches\n", errors);

        free(ln_ref);
        free(q_ref);
        free(k_ref);
        free(v_ref);
        free(scores_ref);
        free(probs_ref);
        free(attn_ref);
        free(resid_ref);
        free(act_ref);
    } else {
        printf("SKIP: graph result check disabled\n");
    }

    layer_norm_free_all();
    matmul_free_all();
    softmax_free_all();
    residual_add_free_all();
    activation_free_all();
    free(x_shadow);
    free(gamma_shadow);
    free(beta_shadow);
    free(wq_shadow);
    free(wk_shadow);
    free(wv_shadow);
    free(residual_shadow);

    return (errors > 0) ? 1 : 0;
}
