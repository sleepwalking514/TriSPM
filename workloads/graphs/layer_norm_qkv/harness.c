#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layer_norm_launcher.h"
#include "matmul_launcher.h"
#include "libspm.h"

#ifndef GRAPH_M
#define GRAPH_M 32
#endif
#ifndef GRAPH_D_MODEL
#define GRAPH_D_MODEL 64
#endif
#ifndef GRAPH_PROJ_N
#define GRAPH_PROJ_N 64
#endif
#ifndef GRAPH_BLOCK_SIZE_M
#define GRAPH_BLOCK_SIZE_M 32
#endif
#ifndef GRAPH_BLOCK_SIZE_N
#define GRAPH_BLOCK_SIZE_N 32
#endif
#ifndef GRAPH_BLOCK_SIZE_K
#define GRAPH_BLOCK_SIZE_K 32
#endif
#ifndef GRAPH_CHECK_RESULT
#define GRAPH_CHECK_RESULT 1
#endif
#ifndef GRAPH_FLUSH_BEFORE_ROI
#define GRAPH_FLUSH_BEFORE_ROI 1
#endif

#define LN_GRID_X GRAPH_M
#define MATMUL_GRID_X (((GRAPH_M + GRAPH_BLOCK_SIZE_M - 1) / GRAPH_BLOCK_SIZE_M) \
                     * ((GRAPH_PROJ_N + GRAPH_BLOCK_SIZE_N - 1) / GRAPH_BLOCK_SIZE_N))

static void init_x(float *x)
{
    for (int i = 0; i < GRAPH_M * GRAPH_D_MODEL; i++)
        x[i] = (float)((i % 23) - 11) * 0.1f;
}

static void init_layer_norm_params(float *gamma, float *beta)
{
    for (int j = 0; j < GRAPH_D_MODEL; j++) {
        gamma[j] = 0.8f + (float)(j % 5) * 0.1f;
        beta[j] = (float)(j % 3) * 0.05f;
    }
}

static void init_weight(float *w, int salt)
{
    for (int i = 0; i < GRAPH_D_MODEL * GRAPH_PROJ_N; i++)
        w[i] = (float)(((i + salt) % 13) - 6) * 0.1f;
}

static void reference_layer_norm(
    const float *x, const float *gamma, const float *beta, float *out)
{
    for (int i = 0; i < GRAPH_M; i++) {
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
                (x[i * GRAPH_D_MODEL + j] - mean) * inv_std
                * gamma[j] + beta[j];
        }
    }
}

static void reference_matmul(const float *a, const float *b, float *c)
{
    for (int i = 0; i < GRAPH_M; i++) {
        for (int j = 0; j < GRAPH_PROJ_N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < GRAPH_D_MODEL; k++)
                sum += a[i * GRAPH_D_MODEL + k] * b[k * GRAPH_PROJ_N + j];
            c[i * GRAPH_PROJ_N + j] = sum;
        }
    }
}

static int check_tensor(const char *name, const float *got, const float *ref,
                        int count, float tolerance)
{
    int errors = 0;
    for (int i = 0; i < count; i++) {
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
    printf("graph layer_norm_qkv: M=%d D=%d N=%d matmul_grid=%d check=%d flush=%d\n",
           GRAPH_M, GRAPH_D_MODEL, GRAPH_PROJ_N, MATMUL_GRID_X,
           GRAPH_CHECK_RESULT, GRAPH_FLUSH_BEFORE_ROI);

    const size_t x_bytes = (size_t)GRAPH_M * GRAPH_D_MODEL * sizeof(float);
    const size_t param_bytes = (size_t)GRAPH_D_MODEL * sizeof(float);
    const size_t weight_bytes = (size_t)GRAPH_D_MODEL * GRAPH_PROJ_N * sizeof(float);
    const size_t proj_bytes = (size_t)GRAPH_M * GRAPH_PROJ_N * sizeof(float);

    float *x_shadow = (float *)malloc(x_bytes);
    float *gamma_shadow = (float *)malloc(param_bytes);
    float *beta_shadow = (float *)malloc(param_bytes);
    float *wq_shadow = (float *)malloc(weight_bytes);
    float *wk_shadow = (float *)malloc(weight_bytes);
    float *wv_shadow = (float *)malloc(weight_bytes);

    float *x = (float *)layer_norm_alloc(0, x_bytes);
    float *gamma = (float *)layer_norm_alloc(1, param_bytes);
    float *beta = (float *)layer_norm_alloc(2, param_bytes);
    float *ln_out = (float *)layer_norm_alloc(3, x_bytes);
    float *wq = (float *)matmul_alloc(1, weight_bytes);
    float *wk = (float *)matmul_alloc(1, weight_bytes);
    float *wv = (float *)matmul_alloc(1, weight_bytes);
    float *q = (float *)matmul_alloc(2, proj_bytes);
    float *k = (float *)matmul_alloc(2, proj_bytes);
    float *v = (float *)matmul_alloc(2, proj_bytes);

    if (!x_shadow || !gamma_shadow || !beta_shadow ||
        !wq_shadow || !wk_shadow || !wv_shadow ||
        !x || !gamma || !beta || !ln_out || !wq || !wk || !wv || !q || !k || !v) {
        fprintf(stderr, "malloc failed\n");
        layer_norm_free_all();
        matmul_free_all();
        free(x_shadow);
        free(gamma_shadow);
        free(beta_shadow);
        free(wq_shadow);
        free(wk_shadow);
        free(wv_shadow);
        return 1;
    }

    init_x(x_shadow);
    init_layer_norm_params(gamma_shadow, beta_shadow);
    init_weight(wq_shadow, 0);
    init_weight(wk_shadow, 5);
    init_weight(wv_shadow, 9);

    flush_caches();
    publish_input(x, x_shadow, x_bytes);
    publish_input(gamma, gamma_shadow, param_bytes);
    publish_input(beta, beta_shadow, param_bytes);
    publish_input(wq, wq_shadow, weight_bytes);
    publish_input(wk, wk_shadow, weight_bytes);
    publish_input(wv, wv_shadow, weight_bytes);

    memset(ln_out, 0, x_bytes);
    memset(q, 0, proj_bytes);
    memset(k, 0, proj_bytes);
    memset(v, 0, proj_bytes);

    if (GRAPH_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);

    layer_norm_launch(LN_GRID_X, 1, 1, x, gamma, beta, ln_out);
    matmul_launch(MATMUL_GRID_X, 1, 1, ln_out, wq, q);
    matmul_launch(MATMUL_GRID_X, 1, 1, ln_out, wk, k);
    matmul_launch(MATMUL_GRID_X, 1, 1, ln_out, wv, v);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (GRAPH_CHECK_RESULT) {
        float *ln_ref = (float *)malloc(x_bytes);
        float *q_ref = (float *)malloc(proj_bytes);
        float *k_ref = (float *)malloc(proj_bytes);
        float *v_ref = (float *)malloc(proj_bytes);
        if (!ln_ref || !q_ref || !k_ref || !v_ref) {
            fprintf(stderr, "malloc failed\n");
            free(ln_ref);
            free(q_ref);
            free(k_ref);
            free(v_ref);
            layer_norm_free_all();
            matmul_free_all();
            free(x_shadow);
            free(gamma_shadow);
            free(beta_shadow);
            free(wq_shadow);
            free(wk_shadow);
            free(wv_shadow);
            return 1;
        }

        reference_layer_norm(x_shadow, gamma_shadow, beta_shadow, ln_ref);
        reference_matmul(ln_ref, wq_shadow, q_ref);
        reference_matmul(ln_ref, wk_shadow, k_ref);
        reference_matmul(ln_ref, wv_shadow, v_ref);

        errors += check_tensor("ln_out", ln_out, ln_ref,
                               GRAPH_M * GRAPH_D_MODEL, 1e-4f);
        errors += check_tensor("q", q, q_ref, GRAPH_M * GRAPH_PROJ_N, 1e-3f);
        errors += check_tensor("k", k, k_ref, GRAPH_M * GRAPH_PROJ_N, 1e-3f);
        errors += check_tensor("v", v, v_ref, GRAPH_M * GRAPH_PROJ_N, 1e-3f);

        if (errors == 0)
            printf("PASS: graph outputs correct\n");
        else
            printf("FAIL: graph has %d mismatches\n", errors);

        free(ln_ref);
        free(q_ref);
        free(k_ref);
        free(v_ref);
    } else {
        printf("SKIP: graph result check disabled\n");
    }

    layer_norm_free_all();
    matmul_free_all();
    free(x_shadow);
    free(gamma_shadow);
    free(beta_shadow);
    free(wq_shadow);
    free(wk_shadow);
    free(wv_shadow);

    return (errors > 0) ? 1 : 0;
}
