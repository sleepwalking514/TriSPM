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
#ifndef GRAPH_FFN_DIM
#define GRAPH_FFN_DIM 64
#endif
#ifndef GRAPH_BLOCK
#define GRAPH_BLOCK 16
#endif
#ifndef GRAPH_QKV_BLOCK_M
#define GRAPH_QKV_BLOCK_M 32
#endif
#ifndef GRAPH_QKV_BLOCK_N
#define GRAPH_QKV_BLOCK_N 16
#endif
#ifndef GRAPH_QK_BLOCK_M
#define GRAPH_QK_BLOCK_M 32
#endif
#ifndef GRAPH_QK_BLOCK_N
#define GRAPH_QK_BLOCK_N 16
#endif
#ifndef GRAPH_PV_BLOCK_M
#define GRAPH_PV_BLOCK_M 32
#endif
#ifndef GRAPH_PV_BLOCK_N
#define GRAPH_PV_BLOCK_N 16
#endif
#ifndef GRAPH_O_PROJ_BLOCK_M
#define GRAPH_O_PROJ_BLOCK_M 32
#endif
#ifndef GRAPH_O_PROJ_BLOCK_N
#define GRAPH_O_PROJ_BLOCK_N 16
#endif
#ifndef GRAPH_FFN_UP_BLOCK_M
#define GRAPH_FFN_UP_BLOCK_M 32
#endif
#ifndef GRAPH_FFN_UP_BLOCK_N
#define GRAPH_FFN_UP_BLOCK_N 32
#endif
#ifndef GRAPH_FFN_DOWN_BLOCK_M
#define GRAPH_FFN_DOWN_BLOCK_M 32
#endif
#ifndef GRAPH_FFN_DOWN_BLOCK_N
#define GRAPH_FFN_DOWN_BLOCK_N 16
#endif
#ifndef GRAPH_K_TRANSPOSE_BLOCK_M
#define GRAPH_K_TRANSPOSE_BLOCK_M 16
#endif
#ifndef GRAPH_K_TRANSPOSE_BLOCK_N
#define GRAPH_K_TRANSPOSE_BLOCK_N 16
#endif
#ifndef GRAPH_CHECK_RESULT
#define GRAPH_CHECK_RESULT 1
#endif
#ifndef GRAPH_FLUSH_BEFORE_ROI
#define GRAPH_FLUSH_BEFORE_ROI 1
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MODEL_ELEMS (GRAPH_SEQ * GRAPH_D_MODEL)
#define QKV_ELEMS (GRAPH_SEQ * GRAPH_HEAD_DIM)
#define SCORE_ELEMS (GRAPH_SEQ * GRAPH_SEQ)
#define FFN_ELEMS (GRAPH_SEQ * GRAPH_FFN_DIM)
#define MODEL_GRID_X CEIL_DIV(MODEL_ELEMS, GRAPH_BLOCK)
#define FFN_GRID_X CEIL_DIV(FFN_ELEMS, GRAPH_BLOCK)
#define MATMUL_GRID(m, n, bm, bn) (CEIL_DIV((m), (bm)) * CEIL_DIV((n), (bn)))

static void init_matrix(float *x, int rows, int cols, int salt)
{
    for (int i = 0; i < rows * cols; i++)
        x[i] = (float)(((i * 7 + salt * 11) % 31) - 15) * 0.05f;
}

static void init_layer_norm_params(float *gamma, float *beta, int cols, int salt)
{
    for (int j = 0; j < cols; j++) {
        gamma[j] = 0.75f + (float)((j + salt) % 7) * 0.05f;
        beta[j] = (float)(((j + salt) % 5) - 2) * 0.025f;
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

static void reference_residual_add(
    const float *x, const float *residual, float *out, int elems)
{
    for (int i = 0; i < elems; i++)
        out[i] = x[i] + residual[i];
}

static void reference_activation(const float *x, float *out, int elems)
{
    for (int i = 0; i < elems; i++)
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
    o_proj_free_all();
    attn_residual_add_free_all();
    ln2_free_all();
    ffn_up_free_all();
    ffn_activation_free_all();
    ffn_down_free_all();
    final_residual_add_free_all();
}

int main(void)
{
    printf("graph attention_smoke: SEQ=%d D_MODEL=%d HEAD_DIM=%d FFN_DIM=%d BLOCK=%d check=%d flush=%d\n",
           GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_HEAD_DIM, GRAPH_FFN_DIM, GRAPH_BLOCK,
           GRAPH_CHECK_RESULT, GRAPH_FLUSH_BEFORE_ROI);

    const size_t model_bytes = (size_t)MODEL_ELEMS * sizeof(float);
    const size_t qkv_bytes = (size_t)QKV_ELEMS * sizeof(float);
    const size_t score_bytes = (size_t)SCORE_ELEMS * sizeof(float);
    const size_t ffn_bytes = (size_t)FFN_ELEMS * sizeof(float);
    const size_t ln_param_bytes = (size_t)GRAPH_D_MODEL * sizeof(float);
    const size_t qkv_weight_bytes =
        (size_t)GRAPH_D_MODEL * (size_t)GRAPH_HEAD_DIM * sizeof(float);
    const size_t o_weight_bytes =
        (size_t)GRAPH_HEAD_DIM * (size_t)GRAPH_D_MODEL * sizeof(float);
    const size_t ffn_up_weight_bytes =
        (size_t)GRAPH_D_MODEL * (size_t)GRAPH_FFN_DIM * sizeof(float);
    const size_t ffn_down_weight_bytes =
        (size_t)GRAPH_FFN_DIM * (size_t)GRAPH_D_MODEL * sizeof(float);

    float *x_shadow = (float *)malloc(model_bytes);
    float *gamma_shadow = (float *)malloc(ln_param_bytes);
    float *beta_shadow = (float *)malloc(ln_param_bytes);
    float *wq_shadow = (float *)malloc(qkv_weight_bytes);
    float *wk_shadow = (float *)malloc(qkv_weight_bytes);
    float *wv_shadow = (float *)malloc(qkv_weight_bytes);
    float *wo_shadow = (float *)malloc(o_weight_bytes);
    float *residual_shadow = (float *)malloc(model_bytes);
    float *gamma2_shadow = (float *)malloc(ln_param_bytes);
    float *beta2_shadow = (float *)malloc(ln_param_bytes);
    float *w_up_shadow = (float *)malloc(ffn_up_weight_bytes);
    float *w_down_shadow = (float *)malloc(ffn_down_weight_bytes);

    float *x = (float *)layer_norm_alloc(0, model_bytes);
    float *gamma = (float *)layer_norm_alloc(1, ln_param_bytes);
    float *beta = (float *)layer_norm_alloc(2, ln_param_bytes);
    float *ln_out = (float *)layer_norm_alloc(3, model_bytes);
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
    float *wo = (float *)o_proj_alloc(1, o_weight_bytes);
    float *attn_proj = (float *)o_proj_alloc(2, model_bytes);
    float *residual = (float *)attn_residual_add_alloc(1, model_bytes);
    float *resid_out = (float *)attn_residual_add_alloc(2, model_bytes);
    float *gamma2 = (float *)ln2_alloc(1, ln_param_bytes);
    float *beta2 = (float *)ln2_alloc(2, ln_param_bytes);
    float *ln2_out = (float *)ln2_alloc(3, model_bytes);
    float *w_up = (float *)ffn_up_alloc(1, ffn_up_weight_bytes);
    float *ffn_hidden = (float *)ffn_up_alloc(2, ffn_bytes);
    float *ffn_act = (float *)ffn_activation_alloc(1, ffn_bytes);
    float *w_down = (float *)ffn_down_alloc(1, ffn_down_weight_bytes);
    float *ffn_out = (float *)ffn_down_alloc(2, model_bytes);
    float *block_out = (float *)final_residual_add_alloc(2, model_bytes);

    if (!x_shadow || !gamma_shadow || !beta_shadow || !wq_shadow ||
        !wk_shadow || !wv_shadow || !wo_shadow || !residual_shadow ||
        !gamma2_shadow || !beta2_shadow || !w_up_shadow || !w_down_shadow ||
        !x || !gamma || !beta || !ln_out || !wq || !wk || !wv || !q || !k || !v ||
        !k_t || !scores || !probs || !attn || !wo || !attn_proj || !residual ||
        !resid_out || !gamma2 || !beta2 || !ln2_out || !w_up || !ffn_hidden ||
        !ffn_act || !w_down || !ffn_out || !block_out) {
        fprintf(stderr, "malloc failed\n");
        free_all_nodes();
        free(x_shadow);
        free(gamma_shadow);
        free(beta_shadow);
        free(wq_shadow);
        free(wk_shadow);
        free(wv_shadow);
        free(wo_shadow);
        free(residual_shadow);
        free(gamma2_shadow);
        free(beta2_shadow);
        free(w_up_shadow);
        free(w_down_shadow);
        return 1;
    }

    init_matrix(x_shadow, GRAPH_SEQ, GRAPH_D_MODEL, 1);
    init_layer_norm_params(gamma_shadow, beta_shadow, GRAPH_D_MODEL, 0);
    init_matrix(wq_shadow, GRAPH_D_MODEL, GRAPH_HEAD_DIM, 2);
    init_matrix(wk_shadow, GRAPH_D_MODEL, GRAPH_HEAD_DIM, 3);
    init_matrix(wv_shadow, GRAPH_D_MODEL, GRAPH_HEAD_DIM, 4);
    init_matrix(wo_shadow, GRAPH_HEAD_DIM, GRAPH_D_MODEL, 5);
    init_matrix(residual_shadow, GRAPH_SEQ, GRAPH_D_MODEL, 6);
    init_layer_norm_params(gamma2_shadow, beta2_shadow, GRAPH_D_MODEL, 3);
    init_matrix(w_up_shadow, GRAPH_D_MODEL, GRAPH_FFN_DIM, 7);
    init_matrix(w_down_shadow, GRAPH_FFN_DIM, GRAPH_D_MODEL, 8);

    flush_caches();
    publish_input(x, x_shadow, model_bytes);
    publish_input(gamma, gamma_shadow, ln_param_bytes);
    publish_input(beta, beta_shadow, ln_param_bytes);
    publish_input(wq, wq_shadow, qkv_weight_bytes);
    publish_input(wk, wk_shadow, qkv_weight_bytes);
    publish_input(wv, wv_shadow, qkv_weight_bytes);
    publish_input(wo, wo_shadow, o_weight_bytes);
    publish_input(residual, residual_shadow, model_bytes);
    publish_input(gamma2, gamma2_shadow, ln_param_bytes);
    publish_input(beta2, beta2_shadow, ln_param_bytes);
    publish_input(w_up, w_up_shadow, ffn_up_weight_bytes);
    publish_input(w_down, w_down_shadow, ffn_down_weight_bytes);

    memset(ln_out, 0, model_bytes);
    memset(q, 0, qkv_bytes);
    memset(k, 0, qkv_bytes);
    memset(v, 0, qkv_bytes);
    memset(k_t, 0, qkv_bytes);
    memset(scores, 0, score_bytes);
    memset(probs, 0, score_bytes);
    memset(attn, 0, qkv_bytes);
    memset(attn_proj, 0, model_bytes);
    memset(resid_out, 0, model_bytes);
    memset(ln2_out, 0, model_bytes);
    memset(ffn_hidden, 0, ffn_bytes);
    memset(ffn_act, 0, ffn_bytes);
    memset(ffn_out, 0, model_bytes);
    memset(block_out, 0, model_bytes);

    if (GRAPH_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);

    layer_norm_launch(GRAPH_SEQ, 1, 1, x, gamma, beta, ln_out);
    q_proj_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wq, q);
    k_proj_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wk, k);
    v_proj_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wv, v);
    k_transpose_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_K_TRANSPOSE_BLOCK_M, GRAPH_K_TRANSPOSE_BLOCK_N), 1, 1, k, k_t);
    qk_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_SEQ, GRAPH_QK_BLOCK_M, GRAPH_QK_BLOCK_N), 1, 1, q, k_t, scores);
    softmax_launch(GRAPH_SEQ, 1, 1, scores, probs);
    pv_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_PV_BLOCK_M, GRAPH_PV_BLOCK_N), 1, 1, probs, v, attn);
    o_proj_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_O_PROJ_BLOCK_M, GRAPH_O_PROJ_BLOCK_N), 1, 1, attn, wo, attn_proj);
    attn_residual_add_launch(MODEL_GRID_X, 1, 1, attn_proj, residual, resid_out);
    ln2_launch(GRAPH_SEQ, 1, 1, resid_out, gamma2, beta2, ln2_out);
    ffn_up_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_FFN_DIM, GRAPH_FFN_UP_BLOCK_M, GRAPH_FFN_UP_BLOCK_N), 1, 1, ln2_out, w_up, ffn_hidden);
    ffn_activation_launch(FFN_GRID_X, 1, 1, ffn_hidden, ffn_act);
    ffn_down_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_FFN_DOWN_BLOCK_M, GRAPH_FFN_DOWN_BLOCK_N), 1, 1, ffn_act, w_down, ffn_out);
    final_residual_add_launch(MODEL_GRID_X, 1, 1, ffn_out, resid_out, block_out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (GRAPH_CHECK_RESULT) {
        float *ln_ref = (float *)malloc(model_bytes);
        float *q_ref = (float *)malloc(qkv_bytes);
        float *k_ref = (float *)malloc(qkv_bytes);
        float *v_ref = (float *)malloc(qkv_bytes);
        float *k_t_ref = (float *)malloc(qkv_bytes);
        float *scores_ref = (float *)malloc(score_bytes);
        float *probs_ref = (float *)malloc(score_bytes);
        float *attn_ref = (float *)malloc(qkv_bytes);
        float *attn_proj_ref = (float *)malloc(model_bytes);
        float *resid_ref = (float *)malloc(model_bytes);
        float *ln2_ref = (float *)malloc(model_bytes);
        float *ffn_hidden_ref = (float *)malloc(ffn_bytes);
        float *ffn_act_ref = (float *)malloc(ffn_bytes);
        float *ffn_out_ref = (float *)malloc(model_bytes);
        float *block_ref = (float *)malloc(model_bytes);
        if (!ln_ref || !q_ref || !k_ref || !v_ref || !k_t_ref ||
            !scores_ref || !probs_ref || !attn_ref || !attn_proj_ref ||
            !resid_ref || !ln2_ref || !ffn_hidden_ref || !ffn_act_ref ||
            !ffn_out_ref || !block_ref) {
            fprintf(stderr, "malloc failed\n");
            free(ln_ref);
            free(q_ref);
            free(k_ref);
            free(v_ref);
            free(k_t_ref);
            free(scores_ref);
            free(probs_ref);
            free(attn_ref);
            free(attn_proj_ref);
            free(resid_ref);
            free(ln2_ref);
            free(ffn_hidden_ref);
            free(ffn_act_ref);
            free(ffn_out_ref);
            free(block_ref);
            free_all_nodes();
            free(x_shadow);
            free(gamma_shadow);
            free(beta_shadow);
            free(wq_shadow);
            free(wk_shadow);
            free(wv_shadow);
            free(wo_shadow);
            free(residual_shadow);
            free(gamma2_shadow);
            free(beta2_shadow);
            free(w_up_shadow);
            free(w_down_shadow);
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
        reference_matmul(attn_ref, wo_shadow, attn_proj_ref, GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_HEAD_DIM);
        reference_residual_add(attn_proj_ref, residual_shadow, resid_ref, MODEL_ELEMS);
        reference_layer_norm(resid_ref, gamma2_shadow, beta2_shadow, ln2_ref);
        reference_matmul(ln2_ref, w_up_shadow, ffn_hidden_ref, GRAPH_SEQ, GRAPH_FFN_DIM, GRAPH_D_MODEL);
        reference_activation(ffn_hidden_ref, ffn_act_ref, FFN_ELEMS);
        reference_matmul(ffn_act_ref, w_down_shadow, ffn_out_ref, GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_FFN_DIM);
        reference_residual_add(ffn_out_ref, resid_ref, block_ref, MODEL_ELEMS);

        errors += check_tensor("block_out", block_out, block_ref, MODEL_ELEMS, 1e-1f);
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
        free(attn_proj_ref);
        free(resid_ref);
        free(ln2_ref);
        free(ffn_hidden_ref);
        free(ffn_act_ref);
        free(ffn_out_ref);
        free(block_ref);
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
    free(wo_shadow);
    free(residual_shadow);
    free(gamma2_shadow);
    free(beta2_shadow);
    free(w_up_shadow);
    free(w_down_shadow);

    return (errors > 0) ? 1 : 0;
}
