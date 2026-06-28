#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph_nodes.h"
#include "libspm.h"

#ifndef GRAPH_SEQ
#define GRAPH_SEQ 256
#endif
#ifndef GRAPH_HEADS
#define GRAPH_HEADS 4
#endif
#ifndef GRAPH_D_MODEL
#define GRAPH_D_MODEL 256
#endif
#ifndef GRAPH_HEAD_DIM
#define GRAPH_HEAD_DIM 64
#endif
#ifndef GRAPH_FFN_DIM
#define GRAPH_FFN_DIM 1024
#endif
#ifndef GRAPH_BLOCK
#define GRAPH_BLOCK 64
#endif
#ifndef GRAPH_QKV_BLOCK_M
#define GRAPH_QKV_BLOCK_M 32
#endif
#ifndef GRAPH_QKV_BLOCK_N
#define GRAPH_QKV_BLOCK_N 32
#endif
#ifndef GRAPH_QK_BLOCK_M
#define GRAPH_QK_BLOCK_M 32
#endif
#ifndef GRAPH_QK_BLOCK_N
#define GRAPH_QK_BLOCK_N 32
#endif
#ifndef GRAPH_PV_BLOCK_M
#define GRAPH_PV_BLOCK_M 32
#endif
#ifndef GRAPH_PV_BLOCK_N
#define GRAPH_PV_BLOCK_N 32
#endif
#ifndef GRAPH_O_PROJ_BLOCK_M
#define GRAPH_O_PROJ_BLOCK_M 32
#endif
#ifndef GRAPH_O_PROJ_BLOCK_N
#define GRAPH_O_PROJ_BLOCK_N 32
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
#define GRAPH_FFN_DOWN_BLOCK_N 32
#endif
#ifndef GRAPH_K_TRANSPOSE_BLOCK_M
#define GRAPH_K_TRANSPOSE_BLOCK_M 16
#endif
#ifndef GRAPH_K_TRANSPOSE_BLOCK_N
#define GRAPH_K_TRANSPOSE_BLOCK_N 16
#endif
#ifndef GRAPH_LAYERNORM_ROW_BLOCK
#define GRAPH_LAYERNORM_ROW_BLOCK 1
#endif
#ifndef GRAPH_LAYERNORM_ROW_GROUP_BLOCKS
#define GRAPH_LAYERNORM_ROW_GROUP_BLOCKS 1
#endif
#ifndef GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK
#define GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK 0
#endif
#ifndef GRAPH_SOFTMAX_ROW_BLOCK
#define GRAPH_SOFTMAX_ROW_BLOCK 2
#endif
#ifndef GRAPH_SOFTMAX_ROW_GROUP_BLOCKS
#define GRAPH_SOFTMAX_ROW_GROUP_BLOCKS 8
#endif
#ifndef GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK
#define GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK 1
#endif
#ifndef GRAPH_CAUSAL
#define GRAPH_CAUSAL 1
#endif
#ifndef GRAPH_FOLD_ATTENTION_SCALE_IN_Q
#define GRAPH_FOLD_ATTENTION_SCALE_IN_Q 1
#endif
#ifndef GRAPH_CHECK_RESULT
#define GRAPH_CHECK_RESULT 0
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
#ifndef GRAPH_DUMP_KERNEL_STATS
#define GRAPH_DUMP_KERNEL_STATS 0
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

#if GRAPH_DUMP_KERNEL_STATS
static void dump_kernel_stats(const char *label)
{
    printf("KERNEL_STATS %s\n", label);
    fflush(stdout);
    m5_dump_reset_stats(0, 0);
}
#define KERNEL_DONE(label) do { TRACE_STEP(label); dump_kernel_stats(label); } while (0)
#else
#define KERNEL_DONE(label) TRACE_STEP(label)
#endif

#if GRAPH_D_MODEL != (GRAPH_HEADS * GRAPH_HEAD_DIM)
#error "decoder_canonical_small_mh4 expects GRAPH_D_MODEL == GRAPH_HEADS * GRAPH_HEAD_DIM"
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MODEL_ELEMS (GRAPH_SEQ * GRAPH_D_MODEL)
#define HEAD_ELEMS (GRAPH_SEQ * GRAPH_HEAD_DIM)
#define SCORE_ELEMS (GRAPH_SEQ * GRAPH_SEQ)
#define FFN_ELEMS (GRAPH_SEQ * GRAPH_FFN_DIM)
#define MODEL_GRID_X CEIL_DIV(MODEL_ELEMS, GRAPH_BLOCK)
#define FFN_GRID_X CEIL_DIV(FFN_ELEMS, GRAPH_BLOCK)
#define MATMUL_GRID(m, n, bm, bn) (CEIL_DIV((m), (bm)) * CEIL_DIV((n), (bn)))
#if GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK
#define LAYERNORM_GRID_X (GRAPH_SEQ / (GRAPH_LAYERNORM_ROW_BLOCK * GRAPH_LAYERNORM_ROW_GROUP_BLOCKS))
#else
#define LAYERNORM_GRID_X GRAPH_SEQ
#endif
#if GRAPH_SOFTMAX_INTERNAL_ROW_BLOCK
#define SOFTMAX_GRID_X (GRAPH_SEQ / (GRAPH_SOFTMAX_ROW_BLOCK * GRAPH_SOFTMAX_ROW_GROUP_BLOCKS))
#else
#define SOFTMAX_GRID_X GRAPH_SEQ
#endif

#if GRAPH_LAYERNORM_INTERNAL_ROW_BLOCK
#if GRAPH_LAYERNORM_ROW_BLOCK <= 1
#error "SPM internal row-block layer_norm requires GRAPH_LAYERNORM_ROW_BLOCK > 1"
#endif
#if GRAPH_LAYERNORM_ROW_GROUP_BLOCKS <= 0
#error "SPM internal row-block layer_norm requires GRAPH_LAYERNORM_ROW_GROUP_BLOCKS > 0"
#endif
#if (GRAPH_SEQ % (GRAPH_LAYERNORM_ROW_BLOCK * GRAPH_LAYERNORM_ROW_GROUP_BLOCKS)) != 0
#error "SPM internal row-block layer_norm requires GRAPH_SEQ divisible by row-block * row-group-blocks"
#endif
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

typedef void *(*alloc_fn)(int, size_t);
typedef void (*free_fn)(void);
typedef void (*binary_launch_fn)(int32_t, int32_t, int32_t, void *, void *, void *);
typedef void (*unary_launch_fn)(int32_t, int32_t, int32_t, void *, void *);

static alloc_fn q_proj_allocs[GRAPH_HEADS] = { q_proj_h0_alloc, q_proj_h1_alloc, q_proj_h2_alloc, q_proj_h3_alloc };
static alloc_fn k_proj_allocs[GRAPH_HEADS] = { k_proj_h0_alloc, k_proj_h1_alloc, k_proj_h2_alloc, k_proj_h3_alloc };
static alloc_fn v_proj_allocs[GRAPH_HEADS] = { v_proj_h0_alloc, v_proj_h1_alloc, v_proj_h2_alloc, v_proj_h3_alloc };
static alloc_fn k_transpose_allocs[GRAPH_HEADS] = { k_transpose_h0_alloc, k_transpose_h1_alloc, k_transpose_h2_alloc, k_transpose_h3_alloc };
static alloc_fn qk_allocs[GRAPH_HEADS] = { qk_h0_alloc, qk_h1_alloc, qk_h2_alloc, qk_h3_alloc };
static alloc_fn softmax_allocs[GRAPH_HEADS] = { softmax_h0_alloc, softmax_h1_alloc, softmax_h2_alloc, softmax_h3_alloc };
static alloc_fn pv_allocs[GRAPH_HEADS] = { pv_h0_alloc, pv_h1_alloc, pv_h2_alloc, pv_h3_alloc };
static alloc_fn o_proj_allocs[GRAPH_HEADS] = { o_proj_h0_alloc, o_proj_h1_alloc, o_proj_h2_alloc, o_proj_h3_alloc };

static binary_launch_fn q_proj_launches[GRAPH_HEADS] = { q_proj_h0_launch, q_proj_h1_launch, q_proj_h2_launch, q_proj_h3_launch };
static binary_launch_fn k_proj_launches[GRAPH_HEADS] = { k_proj_h0_launch, k_proj_h1_launch, k_proj_h2_launch, k_proj_h3_launch };
static binary_launch_fn v_proj_launches[GRAPH_HEADS] = { v_proj_h0_launch, v_proj_h1_launch, v_proj_h2_launch, v_proj_h3_launch };
static unary_launch_fn k_transpose_launches[GRAPH_HEADS] = { k_transpose_h0_launch, k_transpose_h1_launch, k_transpose_h2_launch, k_transpose_h3_launch };
static binary_launch_fn qk_launches[GRAPH_HEADS] = { qk_h0_launch, qk_h1_launch, qk_h2_launch, qk_h3_launch };
static unary_launch_fn softmax_launches[GRAPH_HEADS] = { softmax_h0_launch, softmax_h1_launch, softmax_h2_launch, softmax_h3_launch };
static binary_launch_fn pv_launches[GRAPH_HEADS] = { pv_h0_launch, pv_h1_launch, pv_h2_launch, pv_h3_launch };
static binary_launch_fn o_proj_launches[GRAPH_HEADS] = { o_proj_h0_launch, o_proj_h1_launch, o_proj_h2_launch, o_proj_h3_launch };

static free_fn q_proj_frees[GRAPH_HEADS] = { q_proj_h0_free_all, q_proj_h1_free_all, q_proj_h2_free_all, q_proj_h3_free_all };
static free_fn k_proj_frees[GRAPH_HEADS] = { k_proj_h0_free_all, k_proj_h1_free_all, k_proj_h2_free_all, k_proj_h3_free_all };
static free_fn v_proj_frees[GRAPH_HEADS] = { v_proj_h0_free_all, v_proj_h1_free_all, v_proj_h2_free_all, v_proj_h3_free_all };
static free_fn k_transpose_frees[GRAPH_HEADS] = { k_transpose_h0_free_all, k_transpose_h1_free_all, k_transpose_h2_free_all, k_transpose_h3_free_all };
static free_fn qk_frees[GRAPH_HEADS] = { qk_h0_free_all, qk_h1_free_all, qk_h2_free_all, qk_h3_free_all };
static free_fn softmax_frees[GRAPH_HEADS] = { softmax_h0_free_all, softmax_h1_free_all, softmax_h2_free_all, softmax_h3_free_all };
static free_fn pv_frees[GRAPH_HEADS] = { pv_h0_free_all, pv_h1_free_all, pv_h2_free_all, pv_h3_free_all };
static free_fn o_proj_frees[GRAPH_HEADS] = { o_proj_h0_free_all, o_proj_h1_free_all, o_proj_h2_free_all, o_proj_h3_free_all };

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

static void free_head_arrays(float *a[GRAPH_HEADS])
{
    for (int h = 0; h < GRAPH_HEADS; h++)
        free(a[h]);
}

static void free_all_nodes(void)
{
    layer_norm_free_all();
    for (int h = 0; h < GRAPH_HEADS; h++) {
        q_proj_frees[h]();
        k_proj_frees[h]();
        v_proj_frees[h]();
        k_transpose_frees[h]();
        qk_frees[h]();
        softmax_frees[h]();
        pv_frees[h]();
        o_proj_frees[h]();
    }
    attn_head_sum_0_1_free_all();
    attn_head_sum_2_3_free_all();
    attn_head_sum_0_3_free_all();
    attn_residual_add_free_all();
    ln2_free_all();
    ffn_up_free_all();
    ffn_activation_free_all();
    ffn_down_free_all();
    final_residual_add_free_all();
}

static void label_head(char *buf, size_t n, const char *prefix, int h)
{
    snprintf(buf, n, "%s_h%d", prefix, h);
}

int main(void)
{
    printf("graph decoder_canonical_small_mh4: SEQ=%d HEADS=%d D_MODEL=%d HEAD_DIM=%d FFN_DIM=%d canonical_attention=1 causal=%d fold_scale=%d check=%d flush=%d kernel_stats=%d\n",
           GRAPH_SEQ, GRAPH_HEADS, GRAPH_D_MODEL, GRAPH_HEAD_DIM, GRAPH_FFN_DIM,
           GRAPH_CAUSAL, GRAPH_FOLD_ATTENTION_SCALE_IN_Q, GRAPH_CHECK_RESULT,
           GRAPH_FLUSH_BEFORE_ROI, GRAPH_DUMP_KERNEL_STATS);

    const float attention_scale = 1.0f / sqrtf((float)GRAPH_HEAD_DIM);
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
    float *wq_shadow[GRAPH_HEADS] = {0};
    float *wk_shadow[GRAPH_HEADS] = {0};
    float *wv_shadow[GRAPH_HEADS] = {0};
    float *wo_shadow[GRAPH_HEADS] = {0};
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
    float *k_t[GRAPH_HEADS];
    float *scores[GRAPH_HEADS];
    float *probs[GRAPH_HEADS];
    float *attn[GRAPH_HEADS];
    float *attn_proj[GRAPH_HEADS];
    for (int h = 0; h < GRAPH_HEADS; h++) {
        wq[h] = (float *)q_proj_allocs[h](1, qkv_weight_bytes);
        wk[h] = (float *)k_proj_allocs[h](1, qkv_weight_bytes);
        wv[h] = (float *)v_proj_allocs[h](1, qkv_weight_bytes);
        q[h] = (float *)q_proj_allocs[h](2, head_bytes);
        k[h] = (float *)k_proj_allocs[h](2, head_bytes);
        v[h] = (float *)v_proj_allocs[h](2, head_bytes);
        k_t[h] = (float *)k_transpose_allocs[h](1, head_bytes);
        scores[h] = (float *)qk_allocs[h](2, score_bytes);
        probs[h] = (float *)softmax_allocs[h](1, score_bytes);
        attn[h] = (float *)pv_allocs[h](2, head_bytes);
        wo[h] = (float *)o_proj_allocs[h](1, o_weight_bytes);
        attn_proj[h] = (float *)o_proj_allocs[h](2, model_bytes);
    }

    float *attn_sum_0_1 = (float *)attn_head_sum_0_1_alloc(2, model_bytes);
    float *attn_sum_2_3 = (float *)attn_head_sum_2_3_alloc(2, model_bytes);
    float *attn_sum = (float *)attn_head_sum_0_3_alloc(2, model_bytes);
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
        !x || !gamma || !beta || !ln_out
        || !attn_sum_0_1
        || !attn_sum_2_3
        || !attn_sum
        || !residual || !resid_out || !gamma2 || !beta2 ||
        !ln2_out || !w_up || !ffn_hidden || !ffn_act || !w_down ||
        !ffn_out || !block_out;
    for (int h = 0; h < GRAPH_HEADS; h++) {
        malloc_failed = malloc_failed || !wq_shadow[h] || !wk_shadow[h] ||
                        !wv_shadow[h] || !wo_shadow[h] || !wq[h] || !wk[h] ||
                        !wv[h] || !wo[h] || !q[h] || !k[h] || !v[h] ||
                        !k_t[h] || !scores[h] || !probs[h] || !attn[h] ||
                        !attn_proj[h];
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
        memset(k_t[h], 0, head_bytes);
        memset(scores[h], 0, score_bytes);
        memset(probs[h], 0, score_bytes);
        memset(attn[h], 0, head_bytes);
        memset(attn_proj[h], 0, model_bytes);
    }
    memset(attn_sum_0_1, 0, model_bytes);
    memset(attn_sum_2_3, 0, model_bytes);
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

    char label[64];
    m5_reset_stats(0, 0);

    layer_norm_launch(LAYERNORM_GRID_X, 1, 1, x, gamma, beta, ln_out);
    KERNEL_DONE("layer_norm");
    for (int h = 0; h < GRAPH_HEADS; h++) {
        label_head(label, sizeof(label), "q_proj", h);
        q_proj_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wq[h], q[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "k_proj", h);
        k_proj_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wk[h], k[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "v_proj", h);
        v_proj_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_QKV_BLOCK_M, GRAPH_QKV_BLOCK_N), 1, 1, ln_out, wv[h], v[h]);
        KERNEL_DONE(label);
    }
    for (int h = 0; h < GRAPH_HEADS; h++) {
        label_head(label, sizeof(label), "k_transpose", h);
        k_transpose_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_K_TRANSPOSE_BLOCK_M, GRAPH_K_TRANSPOSE_BLOCK_N), 1, 1, k[h], k_t[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "qk", h);
        qk_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_SEQ, GRAPH_QK_BLOCK_M, GRAPH_QK_BLOCK_N), 1, 1, q[h], k_t[h], scores[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "softmax", h);
        softmax_launches[h](SOFTMAX_GRID_X, 1, 1, scores[h], probs[h]);
        KERNEL_DONE(label);
        label_head(label, sizeof(label), "pv", h);
        pv_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_PV_BLOCK_M, GRAPH_PV_BLOCK_N), 1, 1, probs[h], v[h], attn[h]);
        KERNEL_DONE(label);
    }
    for (int h = 0; h < GRAPH_HEADS; h++) {
        label_head(label, sizeof(label), "o_proj", h);
        o_proj_launches[h](MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_O_PROJ_BLOCK_M, GRAPH_O_PROJ_BLOCK_N), 1, 1, attn[h], wo[h], attn_proj[h]);
        KERNEL_DONE(label);
    }
    attn_head_sum_0_1_launch(MODEL_GRID_X, 1, 1, attn_proj[0], attn_proj[1], attn_sum_0_1);
    KERNEL_DONE("attn_head_sum_0_1");
    attn_head_sum_2_3_launch(MODEL_GRID_X, 1, 1, attn_proj[2], attn_proj[3], attn_sum_2_3);
    KERNEL_DONE("attn_head_sum_2_3");
    attn_head_sum_0_3_launch(MODEL_GRID_X, 1, 1, attn_sum_0_1, attn_sum_2_3, attn_sum);
    KERNEL_DONE("attn_head_sum_0_3");
    attn_residual_add_launch(MODEL_GRID_X, 1, 1, attn_sum, residual, resid_out);
    KERNEL_DONE("attn_residual_add");
    ln2_launch(LAYERNORM_GRID_X, 1, 1, resid_out, gamma2, beta2, ln2_out);
    KERNEL_DONE("ln2");
    ffn_up_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_FFN_DIM, GRAPH_FFN_UP_BLOCK_M, GRAPH_FFN_UP_BLOCK_N), 1, 1, ln2_out, w_up, ffn_hidden);
    KERNEL_DONE("ffn_up");
    ffn_activation_launch(FFN_GRID_X, 1, 1, ffn_hidden, ffn_act);
    KERNEL_DONE("ffn_activation");
    ffn_down_launch(MATMUL_GRID(GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_FFN_DOWN_BLOCK_M, GRAPH_FFN_DOWN_BLOCK_N), 1, 1, ffn_act, w_down, ffn_out);
    KERNEL_DONE("ffn_down");
    final_residual_add_launch(MODEL_GRID_X, 1, 1, ffn_out, resid_out, block_out);
    KERNEL_DONE("final_residual_add");

#if !GRAPH_DUMP_KERNEL_STATS
    m5_dump_stats(0, 0);
    TRACE_STEP("after_dump_stats");
#else
    TRACE_STEP("after_kernel_stats");
#endif

    int errors = 0;
    if (GRAPH_CHECK_RESULT) {
        float *ln_ref = (float *)malloc(model_bytes);
        float *q_ref[GRAPH_HEADS] = {0};
        float *k_ref[GRAPH_HEADS] = {0};
        float *v_ref[GRAPH_HEADS] = {0};
        float *k_t_ref[GRAPH_HEADS] = {0};
        float *scores_ref[GRAPH_HEADS] = {0};
        float *probs_ref[GRAPH_HEADS] = {0};
        float *attn_ref[GRAPH_HEADS] = {0};
        float *attn_proj_ref[GRAPH_HEADS] = {0};
        float *attn_sum_0_1_ref = (float *)malloc(model_bytes);
        float *attn_sum_2_3_ref = (float *)malloc(model_bytes);
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
            !ln_ref
            || !attn_sum_0_1_ref
            || !attn_sum_2_3_ref
            || !attn_sum_ref
            || !resid_ref ||
            !ln2_ref || !ffn_hidden_ref || !ffn_act_ref || !ffn_out_ref ||
            !block_ref;
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
            free(attn_sum_0_1_ref);
            free(attn_sum_2_3_ref);
            free(attn_sum_ref);
            free(resid_ref);
            free(ln2_ref);
            free(ffn_hidden_ref);
            free(ffn_act_ref);
            free(ffn_out_ref);
            free(block_ref);
            free_all_nodes();
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
            reference_transpose(k_ref[h], k_t_ref[h], GRAPH_SEQ, GRAPH_HEAD_DIM);
            reference_matmul(q_ref[h], k_t_ref[h], scores_ref[h],
                             GRAPH_SEQ, GRAPH_SEQ, GRAPH_HEAD_DIM);
            reference_softmax(scores_ref[h], probs_ref[h]);
            reference_matmul(probs_ref[h], v_ref[h], attn_ref[h],
                             GRAPH_SEQ, GRAPH_HEAD_DIM, GRAPH_SEQ);
            reference_matmul(attn_ref[h], wo_shadow[h], attn_proj_ref[h],
                             GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_HEAD_DIM);
        }
        reference_residual_add(attn_proj_ref[0], attn_proj_ref[1], attn_sum_0_1_ref, MODEL_ELEMS);
        reference_residual_add(attn_proj_ref[2], attn_proj_ref[3], attn_sum_2_3_ref, MODEL_ELEMS);
        reference_residual_add(attn_sum_0_1_ref, attn_sum_2_3_ref, attn_sum_ref, MODEL_ELEMS);
        reference_residual_add(attn_sum_ref, residual_shadow, resid_ref, MODEL_ELEMS);
        reference_layer_norm(resid_ref, gamma2_shadow, beta2_shadow, ln2_ref);
        reference_matmul(ln2_ref, w_up_shadow, ffn_hidden_ref,
                         GRAPH_SEQ, GRAPH_FFN_DIM, GRAPH_D_MODEL);
        reference_activation(ffn_hidden_ref, ffn_act_ref, FFN_ELEMS);
        reference_matmul(ffn_act_ref, w_down_shadow, ffn_out_ref,
                         GRAPH_SEQ, GRAPH_D_MODEL, GRAPH_FFN_DIM);
        reference_residual_add(ffn_out_ref, resid_ref, block_ref, MODEL_ELEMS);

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
        free(attn_sum_0_1_ref);
            free(attn_sum_2_3_ref);
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
