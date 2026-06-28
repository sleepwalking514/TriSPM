#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "batched_gemm_launcher.h"
#include "libspm.h"

#ifndef BATCH
#error "BATCH must be defined via -D flag"
#endif
#ifndef M
#error "M must be defined via -D flag"
#endif
#ifndef N
#error "N must be defined via -D flag"
#endif
#ifndef K
#error "K must be defined via -D flag"
#endif
#ifndef BLOCK_SIZE_M
#error "BLOCK_SIZE_M must be defined via -D flag"
#endif
#ifndef BLOCK_SIZE_N
#error "BLOCK_SIZE_N must be defined via -D flag"
#endif

#define GRID_X (BATCH * ((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) * \
                ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N))

#ifndef BATCHED_GEMM_WARMUP_ITERS
#define BATCHED_GEMM_WARMUP_ITERS 0
#endif
#ifndef BATCHED_GEMM_MEASURE_ITERS
#define BATCHED_GEMM_MEASURE_ITERS 1
#endif
#ifndef BATCHED_GEMM_FLUSH_BEFORE_ROI
#define BATCHED_GEMM_FLUSH_BEFORE_ROI 1
#endif
#ifndef BATCHED_GEMM_CHECK_RESULT
#define BATCHED_GEMM_CHECK_RESULT 1
#endif
#ifndef BATCHED_GEMM_TOLERANCE
#define BATCHED_GEMM_TOLERANCE 1e-3f
#endif

static volatile int batched_gemm_check_result = 1;

static float init_a_value(int batch, int row, int col)
{
    int v = (batch * 31 + row * 7 + col * 3) % 17;
    return (float)(v - 8) * 0.1f;
}

static float init_b_value(int batch, int row, int col)
{
    int v = (batch * 29 + row * 5 + col * 11) % 13;
    return (float)(v - 6) * 0.1f;
}

int main(void)
{
    batched_gemm_check_result = BATCHED_GEMM_CHECK_RESULT;

    printf("batched_gemm: B=%d  M=%d  N=%d  K=%d  GRID_X=%d  "
           "block=%dx%dx%d  warmup=%d  measure=%d  flush=%d  check=%d\n",
           BATCH, M, N, K, GRID_X, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
           BATCHED_GEMM_WARMUP_ITERS, BATCHED_GEMM_MEASURE_ITERS,
           BATCHED_GEMM_FLUSH_BEFORE_ROI, batched_gemm_check_result);

    size_t a_elems = (size_t)BATCH * M * K;
    size_t b_elems = (size_t)BATCH * K * N;
    size_t c_elems = (size_t)BATCH * M * N;
    size_t a_bytes = a_elems * sizeof(float);
    size_t b_bytes = b_elems * sizeof(float);
    size_t c_bytes = c_elems * sizeof(float);

    float *a_shadow = (float *)malloc(a_bytes);
    float *b_shadow = (float *)malloc(b_bytes);
    float *a = (float *)batched_gemm_alloc(0, a_bytes);
    float *b = (float *)batched_gemm_alloc(1, b_bytes);
    float *c = (float *)batched_gemm_alloc(2, c_bytes);
    float *ref = NULL;

    if (!a_shadow || !b_shadow || !a || !b || !c) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int batch = 0; batch < BATCH; batch++) {
        for (int i = 0; i < M; i++) {
            for (int kk = 0; kk < K; kk++) {
                size_t idx = ((size_t)batch * M + i) * K + kk;
                a_shadow[idx] = init_a_value(batch, i, kk);
            }
        }
        for (int kk = 0; kk < K; kk++) {
            for (int j = 0; j < N; j++) {
                size_t idx = ((size_t)batch * K + kk) * N + j;
                b_shadow[idx] = init_b_value(batch, kk, j);
            }
        }
    }

    flush_caches();
    publish_input(a, a_shadow, a_bytes);
    publish_input(b, b_shadow, b_bytes);
    memset(c, 0, c_bytes);

    if (BATCHED_GEMM_FLUSH_BEFORE_ROI)
        flush_caches();

    for (int iter = 0; iter < BATCHED_GEMM_WARMUP_ITERS; iter++)
        batched_gemm_launch(GRID_X, 1, 1, a, b, c);

    m5_reset_stats(0, 0);

    for (int iter = 0; iter < BATCHED_GEMM_MEASURE_ITERS; iter++)
        batched_gemm_launch(GRID_X, 1, 1, a, b, c);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (batched_gemm_check_result) {
        ref = (float *)malloc(c_bytes);
        if (!ref) {
            fprintf(stderr, "malloc failed\n");
            free(a_shadow);
            free(b_shadow);
            batched_gemm_free_all();
            return 1;
        }

        for (int batch = 0; batch < BATCH; batch++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < K; kk++) {
                        size_t a_idx = ((size_t)batch * M + i) * K + kk;
                        size_t b_idx = ((size_t)batch * K + kk) * N + j;
                        sum += a_shadow[a_idx] * b_shadow[b_idx];
                    }
                    ref[((size_t)batch * M + i) * N + j] = sum;
                }
            }
        }

        for (size_t idx = 0; idx < c_elems; idx++) {
            float diff = fabsf(c[idx] - ref[idx]);
            if (diff > BATCHED_GEMM_TOLERANCE) {
                if (errors < 20) {
                    size_t batch_stride = (size_t)M * N;
                    int batch = (int)(idx / batch_stride);
                    size_t local = idx % batch_stride;
                    int row = (int)(local / N);
                    int col = (int)(local % N);
                    printf("MISMATCH [%zu] (b=%d,r=%d,c=%d): got %.6f, "
                           "expected %.6f, diff %.6f\n",
                           idx, batch, row, col, c[idx], ref[idx], diff);
                }
                errors++;
            }
        }

        if (errors == 0)
            printf("\nPASS: all %zu elements correct\n", c_elems);
        else
            printf("\nFAIL: %d / %zu mismatches\n", errors, c_elems);

        free(ref);
    } else {
        printf("\nSKIP: result check disabled\n");
    }

    free(a_shadow);
    free(b_shadow);
    batched_gemm_free_all();

    return (errors > 0) ? 1 : 0;
}
