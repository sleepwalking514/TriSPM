#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "paged_kv_decode_launcher.h"
#include "libspm.h"

/*
 * Test harness for the single-query paged KV-decode microkernel.
 *
 * Computes online-softmax attention over NUM_PAGES * PAGE_SIZE tokens
 * selected by an int32 page table:
 *
 *   for p in 0..NUM_PAGES:
 *     phys = page_ids[p]
 *     K_page = k_cache[phys, :, :]
 *     V_page = v_cache[phys, :, :]
 *     scores = K_page @ q * sm_scale
 *     update (m, l, acc) with scores and V_page
 *   out = acc / l
 *
 * W1 invariants:
 *   - exactly one batch, one head, one query;
 *   - fixed NUM_PAGES, NUM_PHYS_PAGES, PAGE_SIZE, HEAD_DIM;
 *   - page_ids selects scattered physical pages from [0, NUM_PHYS_PAGES);
 *     default is deterministic without-replacement (PAGE_ID_UNIQUE=1) so
 *     no cache hits come from duplicate physical pages.  PAGE_ID_UNIQUE=0
 *     falls back to with-replacement as a duplicate-handling ablation.
 */

#ifndef PAGED_KV_DECODE_NUM_PAGES
#error "PAGED_KV_DECODE_NUM_PAGES must be defined via -D flag"
#endif
#ifndef PAGED_KV_DECODE_NUM_PHYS_PAGES
#error "PAGED_KV_DECODE_NUM_PHYS_PAGES must be defined via -D flag"
#endif
#ifndef PAGED_KV_DECODE_PAGE_SIZE
#error "PAGED_KV_DECODE_PAGE_SIZE must be defined via -D flag"
#endif
#ifndef PAGED_KV_DECODE_HEAD_DIM
#error "PAGED_KV_DECODE_HEAD_DIM must be defined via -D flag"
#endif
#ifndef PAGED_KV_DECODE_WARMUP_ITERS
#define PAGED_KV_DECODE_WARMUP_ITERS 0
#endif
#ifndef PAGED_KV_DECODE_MEASURE_ITERS
#define PAGED_KV_DECODE_MEASURE_ITERS 1
#endif
#ifndef PAGED_KV_DECODE_FLUSH_BEFORE_ROI
#define PAGED_KV_DECODE_FLUSH_BEFORE_ROI 1
#endif
#ifndef PAGED_KV_DECODE_CHECK_RESULT
#define PAGED_KV_DECODE_CHECK_RESULT 1
#endif
#ifndef PAGED_KV_DECODE_TOLERANCE
#define PAGED_KV_DECODE_TOLERANCE 1e-4f
#endif
#ifndef PAGED_KV_DECODE_REL_TOLERANCE
#define PAGED_KV_DECODE_REL_TOLERANCE 1e-4f
#endif
#ifndef PAGED_KV_DECODE_PAGE_ID_UNIQUE
#define PAGED_KV_DECODE_PAGE_ID_UNIQUE 1
#endif
#ifndef PAGED_KV_DECODE_SEED
#define PAGED_KV_DECODE_SEED 0xc01du
#endif

#define NUM_PAGES      PAGED_KV_DECODE_NUM_PAGES
#define NUM_PHYS_PAGES PAGED_KV_DECODE_NUM_PHYS_PAGES
#define PAGE_SIZE      PAGED_KV_DECODE_PAGE_SIZE
#define HEAD_DIM       PAGED_KV_DECODE_HEAD_DIM

#define KV_CACHE_ELEMS  ((size_t)NUM_PHYS_PAGES * PAGE_SIZE * HEAD_DIM)
#define KV_CACHE_BYTES  (KV_CACHE_ELEMS * sizeof(float))
#define Q_BYTES         ((size_t)HEAD_DIM * sizeof(float))
#define OUT_BYTES       Q_BYTES
#define PAGE_IDS_BYTES  ((size_t)NUM_PAGES * sizeof(int32_t))

static inline uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void init_kv_cache(float *cache, uint32_t salt)
{
    uint32_t rng = (uint32_t)PAGED_KV_DECODE_SEED ^ salt;
    for (size_t i = 0; i < KV_CACHE_ELEMS; ++i) {
        uint32_t v = xorshift32(&rng);
        float f = ((float)(v & 0xFFFFFFu) / (float)0xFFFFFFu) * 0.2f - 0.1f;
        cache[i] = f;
    }
}

static void init_query(float *q)
{
    uint32_t rng = (uint32_t)PAGED_KV_DECODE_SEED ^ 0xa11ce11eu;
    for (int d = 0; d < HEAD_DIM; ++d) {
        uint32_t v = xorshift32(&rng);
        float f = ((float)(v & 0xFFFFFFu) / (float)0xFFFFFFu) * 0.2f - 0.1f;
        q[d] = f;
    }
}

static void init_page_ids(int32_t *page_ids)
{
    uint32_t rng = (uint32_t)PAGED_KV_DECODE_SEED ^ 0xdec0deu;
#if PAGED_KV_DECODE_PAGE_ID_UNIQUE
    /* Without replacement: avoids duplicate-page reuse that would let cache
     * absorb hits the irregular gather scenario is meant to expose. */
    int32_t pool[NUM_PHYS_PAGES];
    for (int i = 0; i < NUM_PHYS_PAGES; ++i)
        pool[i] = (int32_t)i;
    for (int i = 0; i < NUM_PAGES; ++i) {
        int j = i + (int)(xorshift32(&rng) % (uint32_t)(NUM_PHYS_PAGES - i));
        int32_t tmp = pool[i];
        pool[i] = pool[j];
        pool[j] = tmp;
        page_ids[i] = pool[i];
    }
#else
    /* Ablation: exercises kernel correctness under duplicate physical pages. */
    for (int p = 0; p < NUM_PAGES; ++p) {
        page_ids[p] = (int32_t)(xorshift32(&rng) % (uint32_t)NUM_PHYS_PAGES);
    }
#endif
}

static void reference_paged_kv_decode(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    const int32_t *page_ids,
    float *out,
    float sm_scale)
{
    const float neg_inf = -3.4028234663852886e38f;
    float m = neg_inf;
    float l = 0.0f;
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d)
        acc[d] = 0.0f;

    float scores[PAGE_SIZE];
    float p_block[PAGE_SIZE];

    for (int p = 0; p < NUM_PAGES; ++p) {
        int phys = page_ids[p];
        const float *k_page = k_cache + (size_t)phys * PAGE_SIZE * HEAD_DIM;
        const float *v_page = v_cache + (size_t)phys * PAGE_SIZE * HEAD_DIM;

        float page_max = neg_inf;
        for (int i = 0; i < PAGE_SIZE; ++i) {
            float s = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d)
                s += k_page[i * HEAD_DIM + d] * q[d];
            s *= sm_scale;
            scores[i] = s;
            if (s > page_max)
                page_max = s;
        }

        float m_new = (m > page_max) ? m : page_max;
        float alpha = expf(m - m_new);
        float sum_p = 0.0f;
        for (int i = 0; i < PAGE_SIZE; ++i) {
            p_block[i] = expf(scores[i] - m_new);
            sum_p += p_block[i];
        }
        l = l * alpha + sum_p;

        for (int d = 0; d < HEAD_DIM; ++d) {
            float pv = 0.0f;
            for (int i = 0; i < PAGE_SIZE; ++i)
                pv += p_block[i] * v_page[i * HEAD_DIM + d];
            acc[d] = acc[d] * alpha + pv;
        }
        m = m_new;
    }

    for (int d = 0; d < HEAD_DIM; ++d)
        out[d] = acc[d] / l;
}

int main(void)
{
    const float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);

    printf("paged_kv_decode: NUM_PAGES=%d NUM_PHYS_PAGES=%d PAGE_SIZE=%d HEAD_DIM=%d "
           "tokens=%d scale=%.6g warmup=%d measure=%d flush=%d check=%d unique=%d "
           "atol=%.6g rtol=%.6g\n",
           NUM_PAGES, NUM_PHYS_PAGES, PAGE_SIZE, HEAD_DIM,
           NUM_PAGES * PAGE_SIZE, (double)sm_scale,
           PAGED_KV_DECODE_WARMUP_ITERS, PAGED_KV_DECODE_MEASURE_ITERS,
           PAGED_KV_DECODE_FLUSH_BEFORE_ROI, PAGED_KV_DECODE_CHECK_RESULT,
           PAGED_KV_DECODE_PAGE_ID_UNIQUE,
           (double)PAGED_KV_DECODE_TOLERANCE,
           (double)PAGED_KV_DECODE_REL_TOLERANCE);

    float   *q_shadow        = (float   *)malloc(Q_BYTES);
    float   *k_cache_shadow  = (float   *)malloc(KV_CACHE_BYTES);
    float   *v_cache_shadow  = (float   *)malloc(KV_CACHE_BYTES);
    int32_t *page_ids_shadow = (int32_t *)malloc(PAGE_IDS_BYTES);

    float   *q        = (float   *)paged_kv_decode_alloc(0, Q_BYTES);
    float   *k_cache  = (float   *)paged_kv_decode_alloc(1, KV_CACHE_BYTES);
    float   *v_cache  = (float   *)paged_kv_decode_alloc(2, KV_CACHE_BYTES);
    int32_t *page_ids = (int32_t *)paged_kv_decode_alloc(3, PAGE_IDS_BYTES);
    float   *out      = (float   *)paged_kv_decode_alloc(4, OUT_BYTES);

    if (!q_shadow || !k_cache_shadow || !v_cache_shadow || !page_ids_shadow
        || !q || !k_cache || !v_cache || !page_ids || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    init_query(q_shadow);
    init_kv_cache(k_cache_shadow, 0x4u);
    init_kv_cache(v_cache_shadow, 0x5u);
    init_page_ids(page_ids_shadow);

    flush_caches();
    publish_input(q,        q_shadow,        Q_BYTES);
    publish_input(k_cache,  k_cache_shadow,  KV_CACHE_BYTES);
    publish_input(v_cache,  v_cache_shadow,  KV_CACHE_BYTES);
    publish_input(page_ids, page_ids_shadow, PAGE_IDS_BYTES);
    memset(out, 0, OUT_BYTES);

    for (int i = 0; i < PAGED_KV_DECODE_WARMUP_ITERS; ++i) {
        if (PAGED_KV_DECODE_FLUSH_BEFORE_ROI)
            flush_caches();
        paged_kv_decode_launch(1, 1, 1, q, k_cache, v_cache, page_ids, out, sm_scale);
    }

    if (PAGED_KV_DECODE_FLUSH_BEFORE_ROI)
        flush_caches();

    m5_reset_stats(0, 0);
    for (int i = 0; i < PAGED_KV_DECODE_MEASURE_ITERS; ++i)
        paged_kv_decode_launch(1, 1, 1, q, k_cache, v_cache, page_ids, out, sm_scale);
    m5_dump_stats(0, 0);

    int errors = 0;
    if (PAGED_KV_DECODE_CHECK_RESULT) {
        float *ref = (float *)malloc(OUT_BYTES);
        if (!ref) {
            fprintf(stderr, "malloc failed\n");
            free(q_shadow);
            free(k_cache_shadow);
            free(v_cache_shadow);
            free(page_ids_shadow);
            paged_kv_decode_free_all();
            return 1;
        }

        reference_paged_kv_decode(q_shadow, k_cache_shadow, v_cache_shadow,
                                  page_ids_shadow, ref, sm_scale);

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            float got = out[d];
            float expected = ref[d];
            float abs_err = fabsf(got - expected);
            float rel_err = abs_err / fmaxf(fabsf(expected), 1e-6f);
            if (abs_err > max_abs)
                max_abs = abs_err;
            if (rel_err > max_rel)
                max_rel = rel_err;
            float allowed =
                PAGED_KV_DECODE_TOLERANCE +
                PAGED_KV_DECODE_REL_TOLERANCE * fabsf(expected);
            if (abs_err > allowed) {
                if (errors < 10) {
                    printf("MISMATCH [d=%d]: got %.6f, expected %.6f, "
                           "abs=%.6g rel=%.6g allowed=%.6g\n",
                           d, got, expected, abs_err, rel_err, allowed);
                }
                errors++;
            }
        }

        if (errors == 0)
            printf("PASS: all %d elements correct (max_abs=%.6g max_rel=%.6g)\n",
                   HEAD_DIM, (double)max_abs, (double)max_rel);
        else
            printf("FAIL: %d / %d mismatches (max_abs=%.6g max_rel=%.6g)\n",
                   errors, HEAD_DIM, (double)max_abs, (double)max_rel);
        free(ref);
    } else {
        printf("SKIP: result check disabled\n");
    }

    free(q_shadow);
    free(k_cache_shadow);
    free(v_cache_shadow);
    free(page_ids_shadow);
    paged_kv_decode_free_all();
    return (errors > 0) ? 1 : 0;
}
