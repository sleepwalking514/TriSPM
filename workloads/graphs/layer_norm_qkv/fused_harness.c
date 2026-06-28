#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#ifndef GRAPH_FUSION_MICRO_M
#define GRAPH_FUSION_MICRO_M 8
#endif
#ifndef GRAPH_FUSION_WINDOW_K
#define GRAPH_FUSION_WINDOW_K 4
#endif
#ifndef GRAPH_FUSION_VARIANT
#define GRAPH_FUSION_VARIANT 1
#endif

#define FUSION_VARIANT_CACHE 0
#define FUSION_VARIANT_SPM_RESIDENT 1
#define FUSION_VARIANT_FORCED_MATERIALIZE 2

#if GRAPH_FUSION_VARIANT == FUSION_VARIANT_CACHE
#define FUSION_VARIANT_NAME "A2_fused_cache"
#define FUSION_USES_DMA 0
#define FUSION_NEEDS_LN_WORKSPACE 1
#elif GRAPH_FUSION_VARIANT == FUSION_VARIANT_SPM_RESIDENT
#define FUSION_VARIANT_NAME "A3_fused_spm_resident"
#define FUSION_USES_DMA 1
#define FUSION_NEEDS_LN_WORKSPACE 0
#elif GRAPH_FUSION_VARIANT == FUSION_VARIANT_FORCED_MATERIALIZE
#define FUSION_VARIANT_NAME "A4_fused_spm_forced_materialize"
#define FUSION_USES_DMA 1
#define FUSION_NEEDS_LN_WORKSPACE 1
#else
#error "GRAPH_FUSION_VARIANT must be 0 (cache), 1 (SPM resident), or 2 (forced materialize)"
#endif

#ifndef SPM_MAX_BYTES
#define SPM_MAX_BYTES (256 * 1024)
#endif
#ifndef DMA_QUEUE_DEPTH
#define DMA_QUEUE_DEPTH 32
#endif

#define BM GRAPH_BLOCK_SIZE_M
#define BN GRAPH_BLOCK_SIZE_N
#define BK GRAPH_BLOCK_SIZE_K
#define MICRO_M GRAPH_FUSION_MICRO_M
#define WINDOW_K GRAPH_FUSION_WINDOW_K

#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

#define X_TILE_BYTES (BM * GRAPH_D_MODEL * sizeof(float))
#define B_TILE_BYTES (BK * BN * sizeof(float))
#define B_WINDOW_BYTES (B_TILE_BYTES * WINDOW_K)
#define ACC_BYTES (BM * BN * sizeof(float))

#define SPM_X_TILE SPM_BASE
#define SPM_B_WINDOW (SPM_X_TILE + ALIGN_UP(X_TILE_BYTES, 64))
#define SPM_ACC (SPM_B_WINDOW + ALIGN_UP(B_WINDOW_BYTES, 64))
#define SPM_FUSION_BYTES (ALIGN_UP(X_TILE_BYTES, 64) + \
                          ALIGN_UP(B_WINDOW_BYTES, 64) + \
                          ALIGN_UP(ACC_BYTES, 64))

_Static_assert(GRAPH_M % BM == 0, "GRAPH_M must be divisible by BLOCK_SIZE_M");
_Static_assert(GRAPH_PROJ_N % BN == 0,
               "GRAPH_PROJ_N must be divisible by BLOCK_SIZE_N");
_Static_assert(GRAPH_D_MODEL % BK == 0,
               "GRAPH_D_MODEL must be divisible by BLOCK_SIZE_K");
_Static_assert(BM % MICRO_M == 0,
               "BLOCK_SIZE_M must be divisible by GRAPH_FUSION_MICRO_M");
_Static_assert(WINDOW_K <= DMA_QUEUE_DEPTH,
               "GRAPH_FUSION_WINDOW_K exceeds DMA queue depth");
_Static_assert(SPM_FUSION_BYTES <= SPM_MAX_BYTES,
               "fused layer_norm+qkv SPM layout exceeds SPM_MAX_BYTES");

#if BN == 32
typedef vfloat32m4_t vacc_t;
#define VLE_M __riscv_vle32_v_f32m4
#define VSE_M __riscv_vse32_v_f32m4
#define VFMACC_M __riscv_vfmacc_vf_f32m4
#define VSETVL __riscv_vsetvl_e32m4
#define VREGS_PER_ROW 4
#elif BN == 16
typedef vfloat32m2_t vacc_t;
#define VLE_M __riscv_vle32_v_f32m2
#define VSE_M __riscv_vse32_v_f32m2
#define VFMACC_M __riscv_vfmacc_vf_f32m2
#define VSETVL __riscv_vsetvl_e32m2
#define VREGS_PER_ROW 2
#elif BN == 8
typedef vfloat32m1_t vacc_t;
#define VLE_M __riscv_vle32_v_f32m1
#define VSE_M __riscv_vse32_v_f32m1
#define VFMACC_M __riscv_vfmacc_vf_f32m1
#define VSETVL __riscv_vsetvl_e32m1
#define VREGS_PER_ROW 1
#else
#error "GRAPH_BLOCK_SIZE_N must be 8, 16, or 32"
#endif

_Static_assert(MICRO_M * VREGS_PER_ROW <= 32,
               "MICRO_M * vector-registers-per-row must fit in 32 vregs");

#define LOAD_ACC(i) vacc_t c##i = VLE_M(acc + (i) * BN, vl)
#define FMACC_ACC(i) c##i = VFMACC_M(c##i, a_base[(i) * GRAPH_D_MODEL + k], bv, vl)
#define STORE_ACC(i) VSE_M(acc + (i) * BN, c##i, vl)

static int g_dma_wait_failed = 0;

#if MICRO_M == 2
#define LOAD_MICRO_ACC() LOAD_ACC(0); LOAD_ACC(1)
#define FMACC_MICRO_ACC() FMACC_ACC(0); FMACC_ACC(1)
#define STORE_MICRO_ACC() STORE_ACC(0); STORE_ACC(1)
#elif MICRO_M == 4
#define LOAD_MICRO_ACC() LOAD_ACC(0); LOAD_ACC(1); LOAD_ACC(2); LOAD_ACC(3)
#define FMACC_MICRO_ACC() FMACC_ACC(0); FMACC_ACC(1); FMACC_ACC(2); FMACC_ACC(3)
#define STORE_MICRO_ACC() STORE_ACC(0); STORE_ACC(1); STORE_ACC(2); STORE_ACC(3)
#elif MICRO_M == 8
#define LOAD_MICRO_ACC() LOAD_ACC(0); LOAD_ACC(1); LOAD_ACC(2); LOAD_ACC(3); \
                         LOAD_ACC(4); LOAD_ACC(5); LOAD_ACC(6); LOAD_ACC(7)
#define FMACC_MICRO_ACC() FMACC_ACC(0); FMACC_ACC(1); FMACC_ACC(2); FMACC_ACC(3); \
                          FMACC_ACC(4); FMACC_ACC(5); FMACC_ACC(6); FMACC_ACC(7)
#define STORE_MICRO_ACC() STORE_ACC(0); STORE_ACC(1); STORE_ACC(2); STORE_ACC(3); \
                          STORE_ACC(4); STORE_ACC(5); STORE_ACC(6); STORE_ACC(7)
#elif MICRO_M == 16
#define LOAD_MICRO_ACC() LOAD_ACC(0); LOAD_ACC(1); LOAD_ACC(2); LOAD_ACC(3); \
                         LOAD_ACC(4); LOAD_ACC(5); LOAD_ACC(6); LOAD_ACC(7); \
                         LOAD_ACC(8); LOAD_ACC(9); LOAD_ACC(10); LOAD_ACC(11); \
                         LOAD_ACC(12); LOAD_ACC(13); LOAD_ACC(14); LOAD_ACC(15)
#define FMACC_MICRO_ACC() FMACC_ACC(0); FMACC_ACC(1); FMACC_ACC(2); FMACC_ACC(3); \
                          FMACC_ACC(4); FMACC_ACC(5); FMACC_ACC(6); FMACC_ACC(7); \
                          FMACC_ACC(8); FMACC_ACC(9); FMACC_ACC(10); FMACC_ACC(11); \
                          FMACC_ACC(12); FMACC_ACC(13); FMACC_ACC(14); FMACC_ACC(15)
#define STORE_MICRO_ACC() STORE_ACC(0); STORE_ACC(1); STORE_ACC(2); STORE_ACC(3); \
                          STORE_ACC(4); STORE_ACC(5); STORE_ACC(6); STORE_ACC(7); \
                          STORE_ACC(8); STORE_ACC(9); STORE_ACC(10); STORE_ACC(11); \
                          STORE_ACC(12); STORE_ACC(13); STORE_ACC(14); STORE_ACC(15)
#else
#error "GRAPH_FUSION_MICRO_M must be 2, 4, 8, or 16"
#endif

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
        for (int j = 0; j < GRAPH_D_MODEL; j++)
            out[i * GRAPH_D_MODEL + j] =
                (x[i * GRAPH_D_MODEL + j] - mean) * inv_std
                * gamma[j] + beta[j];
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
            if (errors < 10)
                printf("MISMATCH %s[%d]: got %.6f, expected %.6f\n",
                       name, i, got[i], ref[i]);
            errors++;
        }
    }
    return errors;
}

static inline __attribute__((always_inline))
int dma_wait_all(void)
{
    const uint64_t max_iters = 20000000ULL;
    for (uint64_t it = 0; it < max_iters; ++it) {
        if (dma_read64(DMA_REG_STATUS) == 0) {
            _fence_io();
            return 0;
        }
    }
    return -1;
}

static inline __attribute__((always_inline))
void dma_enqueue_tile(uintptr_t spm_dst, const float *dram_src,
                      int rows, int cols_bytes,
                      int src_stride_bytes, int dst_stride_bytes)
{
    dma_write64(DMA_REG_SRC, (uint64_t)(uintptr_t)dram_src);
    dma_write64(DMA_REG_DST, (uint64_t)spm_dst);
    dma_write64(DMA_REG_SRC_STRIDE, (uint64_t)(size_t)src_stride_bytes);
    dma_write64(DMA_REG_DST_STRIDE, (uint64_t)(size_t)dst_stride_bytes);
    dma_write64(DMA_REG_HEIGHT, (uint64_t)(size_t)rows);
    _fence_io();
    dma_write64(DMA_REG_LEN, (uint64_t)(size_t)cols_bytes);
    _fence_io();
}

static void normalize_tile_to_buffer(const float *x, const float *gamma,
                                     const float *beta, int i0,
                                     float *tile)
{
    for (int mi = 0; mi < BM; mi++) {
        const float *row = &x[(i0 + mi) * GRAPH_D_MODEL];
        float *out = &tile[mi * GRAPH_D_MODEL];

        float mean = 0.0f;
        for (int j = 0; j < GRAPH_D_MODEL; j++)
            mean += row[j];
        mean /= GRAPH_D_MODEL;

        float var = 0.0f;
        for (int j = 0; j < GRAPH_D_MODEL; j++) {
            float d = row[j] - mean;
            var += d * d;
        }
        var /= GRAPH_D_MODEL;

        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < GRAPH_D_MODEL; j++)
            out[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

static void normalize_tile_to_spm(const float *x, const float *gamma,
                                  const float *beta, int i0)
{
    normalize_tile_to_buffer(x, gamma, beta, i0, (float *)SPM_X_TILE);
}

static void normalize_tile_to_cache(const float *x, const float *gamma,
                                    const float *beta, int i0,
                                    float *tile)
{
    normalize_tile_to_buffer(x, gamma, beta, i0, tile);
    asm volatile("fence rw, rw" ::: "memory");
}

static inline __attribute__((always_inline))
void micro_vec_from_resident_tile(int mOff, int kw, int actual_window,
                                  float *restrict acc)
{
    const float *tile = (const float *)SPM_X_TILE;
    size_t vl = VSETVL(BN);
    LOAD_MICRO_ACC();

    for (int wi = 0; wi < actual_window; wi++) {
        const float *b_spm = (const float *)(SPM_B_WINDOW + wi * B_TILE_BYTES);
        const float *a_base =
            &tile[mOff * GRAPH_D_MODEL + (kw + wi) * BK];
        for (int k = 0; k < BK; k++) {
            vacc_t bv = VLE_M(b_spm + k * BN, vl);
            FMACC_MICRO_ACC();
        }
    }

    STORE_MICRO_ACC();
}

static inline __attribute__((always_inline))
void micro_vec_from_cache_tile(const float *restrict tile,
                               const float *restrict B,
                               int mOff, int j0, float *restrict acc)
{
    size_t vl = VSETVL(BN);
    LOAD_MICRO_ACC();

    const float *a_base = &tile[mOff * GRAPH_D_MODEL];
    for (int k = 0; k < GRAPH_D_MODEL; k++) {
        vacc_t bv = VLE_M(&B[k * GRAPH_PROJ_N + j0], vl);
        FMACC_MICRO_ACC();
    }

    STORE_MICRO_ACC();
}

static void matmul_from_cache_tile(const float *restrict tile,
                                   const float *restrict B,
                                   float *restrict C)
{
    float acc_buf[BM * BN] __attribute__((aligned(64)));

    for (int j0 = 0; j0 < GRAPH_PROJ_N; j0 += BN) {
        for (int x = 0; x < BM * BN; x++)
            acc_buf[x] = 0.0f;

        for (int mOff = 0; mOff < BM; mOff += MICRO_M) {
            float *acc_row = &acc_buf[mOff * BN];
            micro_vec_from_cache_tile(tile, B, mOff, j0, acc_row);
        }

        for (int mi = 0; mi < BM; mi++)
            for (int ni = 0; ni < BN; ni++)
                C[mi * GRAPH_PROJ_N + (j0 + ni)] = acc_buf[mi * BN + ni];
    }
}

static void matmul_from_spm_tile(const float *restrict a_reload_src,
                                 const float *restrict B,
                                 float *restrict C)
{
    float *spm_acc = (float *)SPM_ACC;

    for (int j0 = 0; j0 < GRAPH_PROJ_N; j0 += BN) {
        if (a_reload_src) {
            dma_enqueue_tile(SPM_X_TILE, a_reload_src,
                             BM, GRAPH_D_MODEL * sizeof(float),
                             GRAPH_D_MODEL * sizeof(float),
                             GRAPH_D_MODEL * sizeof(float));
        }

        for (int x = 0; x < BM * BN; x++)
            spm_acc[x] = 0.0f;

        if (a_reload_src && dma_wait_all() != 0) {
            g_dma_wait_failed = 1;
            return;
        }

        int k_trips = GRAPH_D_MODEL / BK;
        for (int kw = 0; kw < k_trips; kw += WINDOW_K) {
            int actual_window =
                (kw + WINDOW_K <= k_trips) ? WINDOW_K : (k_trips - kw);

            for (int wi = 0; wi < actual_window; wi++) {
                int k0 = (kw + wi) * BK;
                uintptr_t b_spm_addr = SPM_B_WINDOW + wi * B_TILE_BYTES;
                const float *b_dram = &B[k0 * GRAPH_PROJ_N + j0];
                dma_enqueue_tile(b_spm_addr, b_dram,
                                 BK, BN * sizeof(float),
                                 GRAPH_PROJ_N * sizeof(float),
                                 BN * sizeof(float));
            }
            if (dma_wait_all() != 0) {
                g_dma_wait_failed = 1;
                return;
            }

            for (int mOff = 0; mOff < BM; mOff += MICRO_M) {
                float *acc_row = &spm_acc[mOff * BN];
                micro_vec_from_resident_tile(mOff, kw, actual_window, acc_row);
            }
        }

        for (int mi = 0; mi < BM; mi++)
            for (int ni = 0; ni < BN; ni++)
                C[mi * GRAPH_PROJ_N + (j0 + ni)] = spm_acc[mi * BN + ni];
    }
}

static void matmul_from_resident_tile(const float *restrict B,
                                      float *restrict C)
{
    matmul_from_spm_tile(NULL, B, C);
}

static void matmul_from_materialized_tile(const float *restrict tile,
                                          const float *restrict B,
                                          float *restrict C)
{
    matmul_from_spm_tile(tile, B, C);
}

static void fused_layer_norm_qkv(const float *x, const float *gamma,
                                 const float *beta, const float *wq,
                                 const float *wk, const float *wv,
                                 float *ln_out, float *q, float *k, float *v)
{
    for (int i0 = 0; i0 < GRAPH_M; i0 += BM) {
#if GRAPH_FUSION_VARIANT == FUSION_VARIANT_CACHE
        float *tile = ln_out;
        normalize_tile_to_cache(x, gamma, beta, i0, tile);
        matmul_from_cache_tile(tile, wq, &q[i0 * GRAPH_PROJ_N]);
        matmul_from_cache_tile(tile, wk, &k[i0 * GRAPH_PROJ_N]);
        matmul_from_cache_tile(tile, wv, &v[i0 * GRAPH_PROJ_N]);
#elif GRAPH_FUSION_VARIANT == FUSION_VARIANT_SPM_RESIDENT
        normalize_tile_to_spm(x, gamma, beta, i0);
        matmul_from_resident_tile(wq, &q[i0 * GRAPH_PROJ_N]);
        if (g_dma_wait_failed)
            return;
        matmul_from_resident_tile(wk, &k[i0 * GRAPH_PROJ_N]);
        if (g_dma_wait_failed)
            return;
        matmul_from_resident_tile(wv, &v[i0 * GRAPH_PROJ_N]);
        if (g_dma_wait_failed)
            return;
#elif GRAPH_FUSION_VARIANT == FUSION_VARIANT_FORCED_MATERIALIZE
        float *tile = &ln_out[i0 * GRAPH_D_MODEL];
        normalize_tile_to_cache(x, gamma, beta, i0, tile);
        matmul_from_materialized_tile(tile, wq, &q[i0 * GRAPH_PROJ_N]);
        if (g_dma_wait_failed)
            return;
        matmul_from_materialized_tile(tile, wk, &k[i0 * GRAPH_PROJ_N]);
        if (g_dma_wait_failed)
            return;
        matmul_from_materialized_tile(tile, wv, &v[i0 * GRAPH_PROJ_N]);
        if (g_dma_wait_failed)
            return;
#endif
    }
}

int main(void)
{
    printf("graph layer_norm_qkv fused-ablation: variant=%s "
           "M=%d D=%d N=%d BLK=%dx%dx%d "
           "microM=%d windowK=%d spm_layout=%lu check=%d flush=%d\n",
           FUSION_VARIANT_NAME, GRAPH_M, GRAPH_D_MODEL, GRAPH_PROJ_N,
           BM, BN, BK,
           MICRO_M, WINDOW_K, (unsigned long)SPM_FUSION_BYTES,
           GRAPH_CHECK_RESULT, GRAPH_FLUSH_BEFORE_ROI);

    const size_t x_bytes = (size_t)GRAPH_M * GRAPH_D_MODEL * sizeof(float);
    const size_t param_bytes = (size_t)GRAPH_D_MODEL * sizeof(float);
    const size_t weight_bytes =
        (size_t)GRAPH_D_MODEL * GRAPH_PROJ_N * sizeof(float);
    const size_t proj_bytes = (size_t)GRAPH_M * GRAPH_PROJ_N * sizeof(float);

    float *x_shadow = (float *)malloc(x_bytes);
    float *gamma_shadow = (float *)malloc(param_bytes);
    float *beta_shadow = (float *)malloc(param_bytes);
    float *wq_shadow = (float *)malloc(weight_bytes);
    float *wk_shadow = (float *)malloc(weight_bytes);
    float *wv_shadow = (float *)malloc(weight_bytes);

    float *x = (float *)malloc(x_bytes);
    float *gamma = (float *)malloc(param_bytes);
    float *beta = (float *)malloc(param_bytes);
    float *ln_out = NULL;
#if FUSION_NEEDS_LN_WORKSPACE
    ln_out = (float *)malloc(x_bytes);
#endif
#if FUSION_USES_DMA
    float *wq = (float *)dma_buf_malloc(weight_bytes);
    float *wk = (float *)dma_buf_malloc(weight_bytes);
    float *wv = (float *)dma_buf_malloc(weight_bytes);
#else
    float *wq = (float *)malloc(weight_bytes);
    float *wk = (float *)malloc(weight_bytes);
    float *wv = (float *)malloc(weight_bytes);
#endif
    float *q = (float *)malloc(proj_bytes);
    float *k = (float *)malloc(proj_bytes);
    float *v = (float *)malloc(proj_bytes);

    if (!x_shadow || !gamma_shadow || !beta_shadow ||
        !wq_shadow || !wk_shadow || !wv_shadow ||
        !x || !gamma || !beta || !wq || !wk || !wv ||
        !q || !k || !v
#if FUSION_NEEDS_LN_WORKSPACE
        || !ln_out
#endif
    ) {
        fprintf(stderr, "malloc failed\n");
#if FUSION_USES_DMA
        dma_buf_free_all();
#else
        free(wq);
        free(wk);
        free(wv);
#endif
        free(x_shadow);
        free(gamma_shadow);
        free(beta_shadow);
        free(wq_shadow);
        free(wk_shadow);
        free(wv_shadow);
        free(x);
        free(gamma);
        free(beta);
        free(ln_out);
        free(q);
        free(k);
        free(v);
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

    memset(q, 0, proj_bytes);
    memset(k, 0, proj_bytes);
    memset(v, 0, proj_bytes);
    if (ln_out)
        memset(ln_out, 0, x_bytes);

    if (GRAPH_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);

    fused_layer_norm_qkv(x, gamma, beta, wq, wk, wv, ln_out, q, k, v);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (g_dma_wait_failed) {
        printf("FAIL: dma wait timeout\n");
        errors++;
    }
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
#if FUSION_USES_DMA
            dma_buf_free_all();
#else
            free(wq);
            free(wk);
            free(wv);
#endif
            free(x_shadow);
            free(gamma_shadow);
            free(beta_shadow);
            free(wq_shadow);
            free(wk_shadow);
            free(wv_shadow);
            free(x);
            free(gamma);
            free(beta);
            free(ln_out);
            free(q);
            free(k);
            free(v);
            return 1;
        }

        reference_layer_norm(x_shadow, gamma_shadow, beta_shadow, ln_ref);
        reference_matmul(ln_ref, wq_shadow, q_ref);
        reference_matmul(ln_ref, wk_shadow, k_ref);
        reference_matmul(ln_ref, wv_shadow, v_ref);

#if GRAPH_FUSION_VARIANT == FUSION_VARIANT_FORCED_MATERIALIZE
        errors += check_tensor("ln_out", ln_out, ln_ref,
                               GRAPH_M * GRAPH_D_MODEL, 1e-4f);
#endif
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

#if FUSION_USES_DMA
    dma_buf_free_all();
#else
    free(wq);
    free(wk);
    free(wv);
#endif
    free(x_shadow);
    free(gamma_shadow);
    free(beta_shadow);
    free(wq_shadow);
    free(wk_shadow);
    free(wv_shadow);
    free(x);
    free(gamma);
    free(beta);
    free(ln_out);
    free(q);
    free(k);
    free(v);

    return (errors > 0) ? 1 : 0;
}
