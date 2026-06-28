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
#ifndef GRAPH_HEADS
#define GRAPH_HEADS 2
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
#ifndef GRAPH_FLASH_BLOCK_M
#define GRAPH_FLASH_BLOCK_M 16
#endif
#ifndef GRAPH_FLASH_BLOCK_N
#define GRAPH_FLASH_BLOCK_N 16
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
#define GRAPH_CAUSAL 1
#endif
#ifndef GRAPH_FOLD_ATTENTION_SCALE_IN_Q
#define GRAPH_FOLD_ATTENTION_SCALE_IN_Q 1
#endif
#ifndef GRAPH_USE_FLASH_ATTENTION
#define GRAPH_USE_FLASH_ATTENTION 0
#endif
#ifndef GRAPH_CHECK_RESULT
#define GRAPH_CHECK_RESULT 1
#endif
#ifndef GRAPH_CHECK_INTERMEDIATES
#define GRAPH_CHECK_INTERMEDIATES 0
#endif
#ifndef GRAPH_FLUSH_BEFORE_ROI
#define GRAPH_FLUSH_BEFORE_ROI 1
#endif
#ifndef GRAPH_TRACE_PROGRESS
#define GRAPH_TRACE_PROGRESS 0
#endif

#if GRAPH_TRACE_PROGRESS
static void trace_step(const char *label)
{
    void *probe;
    printf("TRACE %s\n", label);
    fflush(stdout);
    probe = malloc(16);
    if (probe)
        free(probe);
    printf("TRACE_HEAP_OK %s\n", label);
    fflush(stdout);
}
#define TRACE_STEP(label) trace_step(label)
#else
#define TRACE_STEP(label) do { } while (0)
#endif

#if GRAPH_HEADS != 2
#error "attention_mh_causal_smoke harness currently expects GRAPH_HEADS == 2"
#endif
#if GRAPH_D_MODEL != (GRAPH_HEADS * GRAPH_HEAD_DIM)
#error "attention_mh_causal_smoke expects GRAPH_D_MODEL == GRAPH_HEADS * GRAPH_HEAD_DIM"
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MODEL_ELEMS (GRAPH_SEQ * GRAPH_D_MODEL)
#define HEAD_ELEMS (GRAPH_SEQ * GRAPH_HEAD_DIM)
#define SCORE_ELEMS (GRAPH_SEQ * GRAPH_SEQ)
#define FFN_ELEMS (GRAPH_SEQ * GRAPH_FFN_DIM)
#define MODEL_GRID_X CEIL_DIV(MODEL_ELEMS, GRAPH_BLOCK)
#define FFN_GRID_X CEIL_DIV(FFN_ELEMS, GRAPH_BLOCK)
#define MATMUL_GRID(m, n, bm, bn) (CEIL_DIV((m), (bm)) * CEIL_DIV((n), (bn)))
#define FLASH_GRID_X CEIL_DIV(GRAPH_SEQ, GRAPH_FLASH_BLOCK_M)
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
            float e = 0.0f;
            if (!GRAPH_CAUSAL || j <= i)
                e = expf(x[i * GRAPH_SEQ + j] - max_v);
            out[i * GRAPH_SEQ + j] = e;
            denom += e;
        }
        for (int j = 0; j < GRAPH_SEQ; j++)
            out[i * GRAPH_SEQ + j] /= denom;
    }
}

static void reference_flash_attention(
    const float *q, const float *k, const float *v, float *out, float sm_scale)
{
    for (int i = 0; i < GRAPH_SEQ; i++) {
        float max_v = -3.4028234663852886e38f;
        for (int j = 0; j < GRAPH_SEQ; j++) {
            if (GRAPH_CAUSAL && j > i)
                continue;
            float score = 0.0f;
            for (int d = 0; d < GRAPH_HEAD_DIM; d++)
                score += q[i * GRAPH_HEAD_DIM + d] * k[j * GRAPH_HEAD_DIM + d];
            score *= sm_scale;
            if (score > max_v)
                max_v = score;
        }

        float denom = 0.0f;
        for (int d = 0; d < GRAPH_HEAD_DIM; d++)
            out[i * GRAPH_HEAD_DIM + d] = 0.0f;

        for (int j = 0; j < GRAPH_SEQ; j++) {
            if (GRAPH_CAUSAL && j > i)
                continue;
            float score = 0.0f;
            for (int d = 0; d < GRAPH_HEAD_DIM; d++)
                score += q[i * GRAPH_HEAD_DIM + d] * k[j * GRAPH_HEAD_DIM + d];
            score *= sm_scale;
            float weight = expf(score - max_v);
            denom += weight;
            for (int d = 0; d < GRAPH_HEAD_DIM; d++)
                out[i * GRAPH_HEAD_DIM + d] += weight * v[j * GRAPH_HEAD_DIM + d];
        }

        for (int d = 0; d < GRAPH_HEAD_DIM; d++)
            out[i * GRAPH_HEAD_DIM + d] /= denom;
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
            if (errors < 10) {
                printf("MISMATCH %s[%d]: got %.6f, expected %.6f\n",
                       name, i, got[i], ref[i]);
            }
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

static void report_tensor_error(const char *name, const float *got,
                                const float *ref, int elems)
{
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int max_index = 0;
    for (int i = 0; i < elems; i++) {
        float abs_err = fabsf(got[i] - ref[i]);
        float rel_err = abs_err / fmaxf(fabsf(ref[i]), 1e-6f);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            max_index = i;
        }
        if (rel_err > max_rel)
            max_rel = rel_err;
    }
    printf("ERRSUM %s: max_abs=%.6g max_rel=%.6g idx=%d got=%.6f ref=%.6f\n",
           name, (double)max_abs, (double)max_rel, max_index,
           got[max_index], ref[max_index]);
}

static void free_all_nodes(void)
{
    layer_norm_free_all();
    q_proj_h0_free_all();
    k_proj_h0_free_all();
    v_proj_h0_free_all();
    q_proj_h1_free_all();
    k_proj_h1_free_all();
    v_proj_h1_free_all();
#if GRAPH_USE_FLASH_ATTENTION
    flash_attention_h0_free_all();
    flash_attention_h1_free_all();
#else
    k_transpose_h0_free_all();
    qk_h0_free_all();
    softmax_h0_free_all();
    pv_h0_free_all();
    k_transpose_h1_free_all();
    qk_h1_free_all();
    softmax_h1_free_all();
    pv_h1_free_all();
#endif
    o_proj_h0_free_all();
    o_proj_h1_free_all();
    attn_head_sum_free_all();
    attn_residual_add_free_all();
    ln2_free_all();
    ffn_up_free_all();
    ffn_activation_free_all();
    ffn_down_free_all();
    final_residual_add_free_all();
}

static void free_head_arrays(float *a[GRAPH_HEADS])
{
    for (int h = 0; h < GRAPH_HEADS; h++)
        free(a[h]);
}

int main(void)
{
    const char *graph_name =
        GRAPH_USE_FLASH_ATTENTION ? "attention_mh_flash_smoke" :
                                    "attention_mh_causal_smoke";
    printf("graph %s: SEQ=%d HEADS=%d D_MODEL=%d HEAD_DIM=%d FFN_DIM=%d flash=%d causal=%d fold_scale=%d check=%d flush=%d\n",
           graph_name,
           GRAPH_SEQ, GRAPH_HEADS, GRAPH_D_MODEL, GRAPH_HEAD_DIM, GRAPH_FFN_DIM,
           GRAPH_USE_FLASH_ATTENTION, GRAPH_CAUSAL, GRAPH_FOLD_ATTENTION_SCALE_IN_Q,
           GRAPH_CHECK_RESULT, GRAPH_FLUSH_BEFORE_ROI);

    const float attention_scale = 1.0f / sqrtf((float)GRAPH_HEAD_DIM);
    const float flash_sm_scale =
        GRAPH_FOLD_ATTENTION_SCALE_IN_Q ? 1.0f : attention_scale;

    const size_t model_bytes = (size_t)MODEL_ELEMS * sizeof(float);
    const size_t head_bytes = (size_t)HEAD_ELEMS * sizeof(float);
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
    float *wq_shadow[GRAPH_HEADS] = {NULL, NULL};
    float *wk_shadow[GRAPH_HEADS] = {NULL, NULL};
    float *wv_shadow[GRAPH_HEADS] = {NULL, NULL};
    float *wo_shadow[GRAPH_HEADS] = {NULL, NULL};
    float *residual_shadow = (float *)malloc(model_bytes);
    float *gamma2_shadow = (float *)malloc(ln_param_bytes);
    float *beta2_shadow = (float *)malloc(ln_param_bytes);
    float *w_up_shadow = (float *)malloc(ffn_up_weight_bytes);
    float *w_down_shadow = (float *)malloc(ffn_down_weight_bytes);
    for (int h = 0; h < GRAPH_HEADS; h++) {
        wq_shadow[h] = (float *)malloc(qkv_weight_bytes);
        wk_shadow[h] = (float *)malloc(qkv_weight_bytes);
        wv_shadow[h] = (float *)malloc(qkv_weight_bytes);
        wo_shadow[h] = (float *)malloc(o_weight_bytes);
    }
    TRACE_STEP("shadow_alloc");

    float *x = (float *)layer_norm_alloc(0, model_bytes);
    float *gamma = (float *)layer_norm_alloc(1, ln_param_bytes);
    float *beta = (float *)layer_norm_alloc(2, ln_param_bytes);
    float *ln_out = (float *)layer_norm_alloc(3, model_bytes);

    float *wq[GRAPH_HEADS];
    float *wk[GRAPH_HEADS];
    float *wv[GRAPH_HEADS];
    float *wo[GRAPH_HEADS];
    float *q[GRAPH_HEADS];
    float *k[GRAPH_HEADS];
    float *v[GRAPH_HEADS];
    float *k_t[GRAPH_HEADS] = {NULL, NULL};
    float *scores[GRAPH_HEADS] = {NULL, NULL};
    float *probs[GRAPH_HEADS] = {NULL, NULL};
    float *attn[GRAPH_HEADS];
    float *attn_proj[GRAPH_HEADS];

    wq[0] = (float *)q_proj_h0_alloc(1, qkv_weight_bytes);
    wk[0] = (float *)k_proj_h0_alloc(1, qkv_weight_bytes);
    wv[0] = (float *)v_proj_h0_alloc(1, qkv_weight_bytes);
    q[0] = (float *)q_proj_h0_alloc(2, head_bytes);
    k[0] = (float *)k_proj_h0_alloc(2, head_bytes);
    v[0] = (float *)v_proj_h0_alloc(2, head_bytes);

    wq[1] = (float *)q_proj_h1_alloc(1, qkv_weight_bytes);
    wk[1] = (float *)k_proj_h1_alloc(1, qkv_weight_bytes);
    wv[1] = (float *)v_proj_h1_alloc(1, qkv_weight_bytes);
    q[1] = (float *)q_proj_h1_alloc(2, head_bytes);
    k[1] = (float *)k_proj_h1_alloc(2, head_bytes);
    v[1] = (float *)v_proj_h1_alloc(2, head_bytes);

#if GRAPH_USE_FLASH_ATTENTION
    attn[0] = (float *)flash_attention_h0_alloc(3, head_bytes);
    attn[1] = (float *)flash_attention_h1_alloc(3, head_bytes);
#else
    k_t[0] = (float *)k_transpose_h0_alloc(1, head_bytes);
    scores[0] = (float *)qk_h0_alloc(2, score_bytes);
    probs[0] = (float *)softmax_h0_alloc(1, score_bytes);
    attn[0] = (float *)pv_h0_alloc(2, head_bytes);
    k_t[1] = (float *)k_transpose_h1_alloc(1, head_bytes);
    scores[1] = (float *)qk_h1_alloc(2, score_bytes);
    probs[1] = (float *)softmax_h1_alloc(1, score_bytes);
    attn[1] = (float *)pv_h1_alloc(2, head_bytes);
#endif

    wo[0] = (float *)o_proj_h0_alloc(1, o_weight_bytes);
    attn_proj[0] = (float *)o_proj_h0_alloc(2, model_bytes);
    wo[1] = (float *)o_proj_h1_alloc(1, o_weight_bytes);
    attn_proj[1] = (float *)o_proj_h1_alloc(2, model_bytes);
    float *attn_sum = (float *)attn_head_sum_alloc(2, model_bytes);
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
    TRACE_STEP("node_alloc");

    int malloc_failed =
        !x_shadow || !gamma_shadow || !beta_shadow || !residual_shadow ||
        !gamma2_shadow || !beta2_shadow || !w_up_shadow || !w_down_shadow ||
        !x || !gamma || !beta || !ln_out || !attn_sum || !residual ||
        !resid_out || !gamma2 || !beta2 || !ln2_out || !w_up || !ffn_hidden ||
        !ffn_act || !w_down || !ffn_out || !block_out;
    for (int h = 0; h < GRAPH_HEADS; h++) {
        malloc_failed = malloc_failed || !wq_shadow[h] || !wk_shadow[h] ||
                        !wv_shadow[h] || !wo_shadow[h] || !wq[h] || !wk[h] ||
                        !wv[h] || !wo[h] || !q[h] || !k[h] || !v[h] ||
                        !attn[h] || !attn_proj[h];
#if !GRAPH_USE_FLASH_ATTENTION
        malloc_failed = malloc_failed || !k_t[h] || !scores[h] || !probs[h];
#endif
    }
    if (malloc_failed) {
        fprintf(stderr, "malloc failed\n");
        free_all_nodes();
        free(x_shadow);
        free(gamma_shadow);
        free(beta_shadow);
        free_head_arrays(wq_shadow);
        free_head_arrays(wk_shadow);
        free_head_arrays(wv_shadow);
        free_head_arrays(wo_shadow);
        free(residual_shadow);
        free(gamma2_shadow);
        free(beta2_shadow);
        free(w_up_shadow);
        free(w_down_shadow);
        return 1;
    }

    init_matrix(x_shadow, GRAPH_SEQ, GRAPH_D_MODEL, 1);
    init_layer_norm_params(gamma_shadow, beta_shadow, GRAPH_D_MODEL, 0);
    for (int h = 0; h < GRAPH_HEADS; h++) {
        init_matrix(wq_shadow[h], GRAPH_D_MODEL, GRAPH_HEAD_DIM, 2 + h * 10);
        if (GRAPH_FOLD_ATTENTION_SCALE_IN_Q)
            scale_matrix(wq_shadow[h], GRAPH_D_MODEL * GRAPH_HEAD_DIM, attention_scale);
        init_matrix(wk_shadow[h], GRAPH_D_MODEL, GRAPH_HEAD_DIM, 3 + h * 10);
        init_matrix(wv_shadow[h], GRAPH_D_MODEL, GRAPH_HEAD_DIM, 4 + h * 10);
        init_matrix(wo_shadow[h], GRAPH_HEAD_DIM, GRAPH_D_MODEL, 5 + h * 10);
    }
    init_matrix(residual_shadow, GRAPH_SEQ, GRAPH_D_MODEL, 6);
    init_layer_norm_params(gamma2_shadow, beta2_shadow, GRAPH_D_MODEL, 3);
    init_matrix(w_up_shadow, GRAPH_D_MODEL, GRAPH_FFN_DIM, 7);
    init_matrix(w_down_shadow, GRAPH_FFN_DIM, GRAPH_D_MODEL, 8);
    TRACE_STEP("init_inputs");

    flush_caches();
    publish_input(x, x_shadow, model_bytes);
    publish_input(gamma, gamma_shadow, ln_param_bytes);
    publish_input(beta, beta_shadow, ln_param_bytes);
    for (int h = 0; h < GRAPH_HEADS; h++) {
        publish_input(wq[h], wq_shadow[h], qkv_weight_bytes);
        publish_input(wk[h], wk_shadow[h], qkv_weight_bytes);
        publish_input(wv[h], wv_shadow[h], qkv_weight_bytes);
        publish_input(wo[h], wo_shadow[h], o_weight_bytes);
    }
    publish_input(residual, residual_shadow, model_bytes);
    publish_input(gamma2, gamma2_shadow, ln_param_bytes);
    publish_input(beta2, beta2_shadow, ln_param_bytes);
    publish_input(w_up, w_up_shadow, ffn_up_weight_bytes);
    publish_input(w_down, w_down_shadow, ffn_down_weight_bytes);
    TRACE_STEP("publish_inputs");

    memset(ln_out, 0, model_bytes);
    for (int h = 0; h < GRAPH_HEADS; h++) {
        memset(q[h], 0, head_bytes);
        memset(k[h], 0, head_bytes);
        memset(v[h], 0, head_bytes);
#if !GRAPH_USE_FLASH_ATTENTION
        memset(k_t[h], 0, head_bytes);
        memset(scores[h], 0, score_bytes);
        memset(probs[h], 0, score_bytes);
#endif
        memset(attn[h], 0, head_bytes);
        memset(attn_proj[h], 0, model_bytes);
    }
    memset(attn_sum, 0, model_bytes);
    memset(resid_out, 0, model_bytes);
    memset(ln2_out, 0, model_bytes);
    memset(ffn_hidden, 0, ffn_bytes);
    memset(ffn_act, 0, ffn_bytes);
    memset(ffn_out, 0, model_bytes);
    memset(block_out, 0, model_bytes);
    TRACE_STEP("zero_outputs");

    if (GRAPH_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);

    layer_norm_launch(GRAPH_SEQ, 1, 1, x, gamma, beta, ln_out);
    TRACE_STEP("layer_norm");
    q_proj_h0_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wq[0], q[0]);
    TRACE_STEP("q_proj_h0");
    k_proj_h0_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wk[0], k[0]);
    TRACE_STEP("k_proj_h0");
    v_proj_h0_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wv[0], v[0]);
    TRACE_STEP("v_proj_h0");
    q_proj_h1_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wq[1], q[1]);
    TRACE_STEP("q_proj_h1");
    k_proj_h1_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wk[1], k[1]);
    TRACE_STEP("k_proj_h1");
    v_proj_h1_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wv[1], v[1]);
    TRACE_STEP("v_proj_h1");
#if GRAPH_USE_FLASH_ATTENTION
    flash_attention_h0_launch(FLASH_GRID_X, 1, 1, q[0], k[0], v[0], attn[0], flash_sm_scale);
    TRACE_STEP("flash_attention_h0");
    flash_attention_h1_launch(FLASH_GRID_X, 1, 1, q[1], k[1], v[1], attn[1], flash_sm_scale);
    TRACE_STEP("flash_attention_h1");
#else
    k_transpose_h0_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_K_TRANSPOSE_BLOCK_M, GRAPH_K_TRANSPOSE_BLOCK_N), 1, 1, k[0], k_t[0]);
    TRACE_STEP("k_transpose_h0");
    qk_h0_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_SEQ, GRAPH_QK_BLOCK_M, GRAPH_QK_BLOCK_N), 1, 1, q[0], k_t[0], scores[0]);
    TRACE_STEP("qk_h0");
    softmax_h0_launch(SOFTMAX_GRID_X, 1, 1, scores[0], probs[0]);
    TRACE_STEP("softmax_h0");
    pv_h0_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_PV_BLOCK_M, GRAPH_PV_BLOCK_N), 1, 1, probs[0], v[0], attn[0]);
    TRACE_STEP("pv_h0");
    k_transpose_h1_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_K_TRANSPOSE_BLOCK_M, GRAPH_K_TRANSPOSE_BLOCK_N), 1, 1, k[1], k_t[1]);
    TRACE_STEP("k_transpose_h1");
    qk_h1_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_SEQ, GRAPH_QK_BLOCK_M, GRAPH_QK_BLOCK_N), 1, 1, q[1], k_t[1], scores[1]);
    TRACE_STEP("qk_h1");
    softmax_h1_launch(SOFTMAX_GRID_X, 1, 1, scores[1], probs[1]);
    TRACE_STEP("softmax_h1");
    pv_h1_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_PV_BLOCK_M, GRAPH_PV_BLOCK_N), 1, 1, probs[1], v[1], attn[1]);
    TRACE_STEP("pv_h1");
#endif
    o_proj_h0_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_O_PROJ_BLOCK_M, GRAPH_O_PROJ_BLOCK_N), 1, 1, attn[0], wo[0], attn_proj[0]);
    TRACE_STEP("o_proj_h0");
    o_proj_h1_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_O_PROJ_BLOCK_M, GRAPH_O_PROJ_BLOCK_N), 1, 1, attn[1], wo[1], attn_proj[1]);
    TRACE_STEP("o_proj_h1");
    attn_head_sum_launch(MODEL_GRID_X, 1, 1, attn_proj[0], attn_proj[1], attn_sum);
    TRACE_STEP("attn_head_sum");
    attn_residual_add_launch(MODEL_GRID_X, 1, 1, attn_sum, residual, resid_out);
    TRACE_STEP("attn_residual_add");
    ln2_launch(GRAPH_SEQ, 1, 1, resid_out, gamma2, beta2, ln2_out);
    TRACE_STEP("ln2");
    ffn_up_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_FFN_DIM, GRAPH_FFN_UP_BLOCK_M, GRAPH_FFN_UP_BLOCK_N), 1, 1, ln2_out, w_up, ffn_hidden);
    TRACE_STEP("ffn_up");
    ffn_activation_launch(FFN_GRID_X, 1, 1, ffn_hidden, ffn_act);
    TRACE_STEP("ffn_activation");
    ffn_down_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_FFN_DOWN_BLOCK_M, GRAPH_FFN_DOWN_BLOCK_N), 1, 1, ffn_act, w_down, ffn_out);
    TRACE_STEP("ffn_down");
    final_residual_add_launch(MODEL_GRID_X, 1, 1, ffn_out, resid_out, block_out);
    TRACE_STEP("final_residual_add");

    m5_dump_stats(0, 0);
    TRACE_STEP("after_dump_stats");

    int errors = 0;
    if (GRAPH_CHECK_RESULT) {
        TRACE_STEP("before_ref_alloc");
        float *ln_ref = (float *)malloc(model_bytes);
        float *q_ref[GRAPH_HEADS] = {NULL, NULL};
        float *k_ref[GRAPH_HEADS] = {NULL, NULL};
        float *v_ref[GRAPH_HEADS] = {NULL, NULL};
        float *k_t_ref[GRAPH_HEADS] = {NULL, NULL};
        float *scores_ref[GRAPH_HEADS] = {NULL, NULL};
        float *probs_ref[GRAPH_HEADS] = {NULL, NULL};
        float *attn_ref[GRAPH_HEADS] = {NULL, NULL};
        float *attn_proj_ref[GRAPH_HEADS] = {NULL, NULL};
        float *attn_sum_ref = (float *)malloc(model_bytes);
        float *resid_ref = (float *)malloc(model_bytes);
        float *ln2_ref = (float *)malloc(model_bytes);
        float *ffn_hidden_ref = (float *)malloc(ffn_bytes);
        float *ffn_act_ref = (float *)malloc(ffn_bytes);
        float *ffn_out_ref = (float *)malloc(model_bytes);
        float *block_ref = (float *)malloc(model_bytes);
        for (int h = 0; h < GRAPH_HEADS; h++) {
            q_ref[h] = (float *)malloc(head_bytes);
            k_ref[h] = (float *)malloc(head_bytes);
            v_ref[h] = (float *)malloc(head_bytes);
            k_t_ref[h] = (float *)malloc(head_bytes);
            scores_ref[h] = (float *)malloc(score_bytes);
            probs_ref[h] = (float *)malloc(score_bytes);
            attn_ref[h] = (float *)malloc(head_bytes);
            attn_proj_ref[h] = (float *)malloc(model_bytes);
        }

        int ref_malloc_failed =
            !ln_ref || !attn_sum_ref || !resid_ref || !ln2_ref ||
            !ffn_hidden_ref || !ffn_act_ref || !ffn_out_ref || !block_ref;
        for (int h = 0; h < GRAPH_HEADS; h++) {
            ref_malloc_failed = ref_malloc_failed || !q_ref[h] || !k_ref[h] ||
                                !v_ref[h] || !k_t_ref[h] || !scores_ref[h] ||
                                !probs_ref[h] || !attn_ref[h] ||
                                !attn_proj_ref[h];
        }
        if (ref_malloc_failed) {
            fprintf(stderr, "malloc failed\n");
            free(ln_ref);
            free_head_arrays(q_ref);
            free_head_arrays(k_ref);
            free_head_arrays(v_ref);
            free_head_arrays(k_t_ref);
            free_head_arrays(scores_ref);
            free_head_arrays(probs_ref);
            free_head_arrays(attn_ref);
            free_head_arrays(attn_proj_ref);
            free(attn_sum_ref);
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
            free_head_arrays(wq_shadow);
            free_head_arrays(wk_shadow);
            free_head_arrays(wv_shadow);
            free_head_arrays(wo_shadow);
            free(residual_shadow);
            free(gamma2_shadow);
            free(beta2_shadow);
            free(w_up_shadow);
            free(w_down_shadow);
            return 1;
        }

        reference_layer_norm(x_shadow, gamma_shadow, beta_shadow, ln_ref);
        for (int h = 0; h < GRAPH_HEADS; h++) {
            reference_matmul(ln_ref, wq_shadow[h], q_ref[h],
                             GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
            reference_matmul(ln_ref, wk_shadow[h], k_ref[h],
                             GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
            reference_matmul(ln_ref, wv_shadow[h], v_ref[h],
                             GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_D_MODEL);
            if (GRAPH_USE_FLASH_ATTENTION) {
                reference_flash_attention(
                    q_ref[h], k_ref[h], v_ref[h], attn_ref[h], flash_sm_scale);
            } else {
                reference_transpose(k_ref[h], k_t_ref[h], GRAPH_SEQ, GRAPH_HEAD_DIM);
                reference_matmul(q_ref[h], k_t_ref[h], scores_ref[h],
                                 GRAPH_SEQ, GRAPH_SEQ, GRAPH_HEAD_DIM);
                reference_softmax(scores_ref[h], probs_ref[h]);
                reference_matmul(probs_ref[h], v_ref[h], attn_ref[h],
                                 GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_SEQ);
            }
            reference_matmul(attn_ref[h], wo_shadow[h], attn_proj_ref[h],
                             GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_HEAD_DIM);
        }
        reference_residual_add(attn_proj_ref[0], attn_proj_ref[1],
                               attn_sum_ref, MODEL_ELEMS);
        reference_residual_add(attn_sum_ref, residual_shadow, resid_ref, MODEL_ELEMS);
        reference_layer_norm(resid_ref, gamma2_shadow, beta2_shadow, ln2_ref);
        reference_matmul(ln2_ref, w_up_shadow, ffn_hidden_ref,
                         GRAPH_SEQ, GRAPH_FFN_DIM, GRAPH_D_MODEL);
        reference_activation(ffn_hidden_ref, ffn_act_ref, FFN_ELEMS);
        reference_matmul(ffn_act_ref, w_down_shadow, ffn_out_ref,
                         GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_FFN_DIM);
        reference_residual_add(ffn_out_ref, resid_ref, block_ref, MODEL_ELEMS);

#if GRAPH_CHECK_INTERMEDIATES
        report_tensor_error("q_h0", q[0], q_ref[0], HEAD_ELEMS);
        report_tensor_error("q_h1", q[1], q_ref[1], HEAD_ELEMS);
        report_tensor_error("attn_h0", attn[0], attn_ref[0], HEAD_ELEMS);
        report_tensor_error("attn_h1", attn[1], attn_ref[1], HEAD_ELEMS);
        report_tensor_error("attn_sum", attn_sum, attn_sum_ref, MODEL_ELEMS);
        report_tensor_error("resid_out", resid_out, resid_ref, MODEL_ELEMS);
        report_tensor_error("ln2_out", ln2_out, ln2_ref, MODEL_ELEMS);
        report_tensor_error("ffn_out", ffn_out, ffn_out_ref, MODEL_ELEMS);
#endif
        errors += check_tensor("block_out", block_out, block_ref, MODEL_ELEMS, 1e-1f);
        if (errors == 0)
            printf("PASS: graph outputs correct\n");
        else
            printf("FAIL: graph has %d mismatches\n", errors);

        free(ln_ref);
        free_head_arrays(q_ref);
        free_head_arrays(k_ref);
        free_head_arrays(v_ref);
        free_head_arrays(k_t_ref);
        free_head_arrays(scores_ref);
        free_head_arrays(probs_ref);
        free_head_arrays(attn_ref);
        free_head_arrays(attn_proj_ref);
        free(attn_sum_ref);
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
    free_head_arrays(wq_shadow);
    free_head_arrays(wk_shadow);
    free_head_arrays(wv_shadow);
    free_head_arrays(wo_shadow);
    free(residual_shadow);
    free(gamma2_shadow);
    free(beta2_shadow);
    free(w_up_shadow);
    free(w_down_shadow);

    return (errors > 0) ? 1 : 0;
}
