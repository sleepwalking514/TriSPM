#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "embedding_bag_launcher.h"
#include "libspm.h"

/*
 * Test harness for the Triton-compiled embedding_bag kernel.
 *
 * Pooled embedding gather:
 *   for bag b:
 *     for k in 0..L_MAX:
 *       idx = indices[offsets[b] + k]
 *       out[b, :] += table[idx, :]
 *
 * Build with -DEMB_B=... -DEMB_L_MAX=... -DEMB_D=... -DEMB_NUM_ROWS=...
 * -DEMB_BLOCK_D=... -DEMB_INDEX_DIST={0,1} (rendered from experiment.toml).
 *
 * W1 invariants:
 *   - Fixed-length bags: offsets[b+1] - offsets[b] == L_MAX for every bag.
 *   - Uniform (EMB_INDEX_DIST=0) or zipf-like (EMB_INDEX_DIST=1) indices.
 *   - Reproducible PRNG seeded by a fixed seed.
 */

#ifndef EMB_B
#error "EMB_B must be defined via -D flag"
#endif
#ifndef EMB_L_MAX
#error "EMB_L_MAX must be defined via -D flag"
#endif
#ifndef EMB_D
#error "EMB_D must be defined via -D flag"
#endif
#ifndef EMB_NUM_ROWS
#error "EMB_NUM_ROWS must be defined via -D flag"
#endif
#ifndef EMB_INDEX_DIST
#define EMB_INDEX_DIST 0
#endif
#ifndef EMB_BAG_GROUP
#define EMB_BAG_GROUP 1
#endif
#ifndef CHECK_RESULT
#define CHECK_RESULT 1
#endif
#ifndef EMB_FLUSH_BEFORE_ROI
#define EMB_FLUSH_BEFORE_ROI 1
#endif
#ifndef EMB_SEED
#define EMB_SEED 0x5eedu
#endif

static volatile int embedding_bag_check_result = 1;

/* Simple xorshift32: deterministic and free of libc dependencies. */
static inline uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int pick_index_uniform(uint32_t *rng)
{
    return (int)(xorshift32(rng) % (uint32_t)EMB_NUM_ROWS);
}

/* Zipf-like: hot 1% of rows account for ~50% of lookups.
 * Implemented as a 2-mixture: 50% from rows [0, NUM_ROWS/100),
 * 50% uniform over [0, NUM_ROWS). */
static int pick_index_zipf(uint32_t *rng)
{
    uint32_t hot_count = (uint32_t)EMB_NUM_ROWS / 100u;
    if (hot_count == 0) hot_count = 1u;
    if ((xorshift32(rng) & 1u) == 0u) {
        return (int)(xorshift32(rng) % hot_count);
    }
    return (int)(xorshift32(rng) % (uint32_t)EMB_NUM_ROWS);
}

static int pick_index(uint32_t *rng)
{
#if EMB_INDEX_DIST == 1
    return pick_index_zipf(rng);
#else
    return pick_index_uniform(rng);
#endif
}

static const char *index_dist_name(void)
{
#if EMB_INDEX_DIST == 1
    return "zipf";
#else
    return "uniform";
#endif
}

int main(void)
{
    embedding_bag_check_result = CHECK_RESULT;

    if (EMB_B % EMB_BAG_GROUP != 0) {
        fprintf(stderr, "EMB_B (%d) must be divisible by EMB_BAG_GROUP (%d)\n",
                EMB_B, EMB_BAG_GROUP);
        return 1;
    }
    int grid_x = EMB_B / EMB_BAG_GROUP;

    printf("embedding_bag: B=%d  L_MAX=%d  D=%d  NUM_ROWS=%d"
           "  bag_group=%d  grid=%d"
           "  index_dist=%s  flush=%d  check=%d\n",
           EMB_B, EMB_L_MAX, EMB_D, EMB_NUM_ROWS,
           EMB_BAG_GROUP, grid_x,
           index_dist_name(), EMB_FLUSH_BEFORE_ROI, embedding_bag_check_result);

    size_t table_bytes   = (size_t)EMB_NUM_ROWS * EMB_D * sizeof(float);
    size_t indices_bytes = (size_t)EMB_B * EMB_L_MAX * sizeof(int32_t);
    size_t offsets_bytes = (size_t)(EMB_B + 1) * sizeof(int32_t);
    size_t out_bytes     = (size_t)EMB_B * EMB_D * sizeof(float);

    float   *table_shadow   = (float   *)malloc(table_bytes);
    int32_t *indices_shadow = (int32_t *)malloc(indices_bytes);
    int32_t *offsets_shadow = (int32_t *)malloc(offsets_bytes);

    float   *table   = (float   *)embedding_bag_alloc(0, table_bytes);
    int32_t *indices = (int32_t *)embedding_bag_alloc(1, indices_bytes);
    int32_t *offsets = (int32_t *)embedding_bag_alloc(2, offsets_bytes);
    float   *out     = (float   *)embedding_bag_alloc(3, out_bytes);
    float   *ref     = NULL;

    if (!table_shadow || !indices_shadow || !offsets_shadow
        || !table || !indices || !offsets || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Table: deterministic pseudo-random fp32 in roughly [-1, 1]. */
    uint32_t rng_table = (uint32_t)EMB_SEED ^ 0xa5a5a5a5u;
    for (size_t i = 0; i < (size_t)EMB_NUM_ROWS * EMB_D; ++i) {
        uint32_t v = xorshift32(&rng_table);
        /* Map to [-1, 1] with finite precision; sign + 23-bit magnitude. */
        float f = ((float)(v & 0xFFFFFFu) / (float)0xFFFFFFu) * 2.0f - 1.0f;
        table_shadow[i] = f;
    }

    /* Offsets: fixed-length bags, offsets[b] = b * L_MAX. */
    for (int b = 0; b <= EMB_B; ++b) {
        offsets_shadow[b] = b * EMB_L_MAX;
    }

    /* Indices: deterministic, depends on EMB_INDEX_DIST. */
    uint32_t rng_idx = (uint32_t)EMB_SEED;
    for (int b = 0; b < EMB_B; ++b) {
        int start = offsets_shadow[b];
        for (int k = 0; k < EMB_L_MAX; ++k) {
            indices_shadow[start + k] = pick_index(&rng_idx);
        }
    }

    flush_caches();
    publish_input(table,   table_shadow,   table_bytes);
    publish_input(indices, indices_shadow, indices_bytes);
    publish_input(offsets, offsets_shadow, offsets_bytes);

    memset(out, 0, out_bytes);

    if (EMB_FLUSH_BEFORE_ROI)
        flush_caches();

    /* Measure only the Triton kernel ROI; init/ref/publish/check stay outside. */
    m5_reset_stats(0, 0);

    embedding_bag_launch(grid_x, 1, 1, table, indices, offsets, out);

    m5_dump_stats(0, 0);

    int errors = 0;
    if (embedding_bag_check_result) {
        ref = (float *)malloc(out_bytes);
        if (!ref) {
            fprintf(stderr, "malloc failed\n");
            free(table_shadow);
            free(indices_shadow);
            free(offsets_shadow);
            embedding_bag_free_all();
            return 1;
        }

        /* Reference embedding_bag on private shadow buffers after the ROI. */
        for (int b = 0; b < EMB_B; ++b) {
            int start = offsets_shadow[b];
            for (int d = 0; d < EMB_D; ++d) {
                float acc = 0.0f;
                for (int k = 0; k < EMB_L_MAX; ++k) {
                    int idx = indices_shadow[start + k];
                    acc += table_shadow[(size_t)idx * EMB_D + d];
                }
                ref[b * EMB_D + d] = acc;
            }
        }

        const float tol = 1e-3f;
        for (int i = 0; i < EMB_B * EMB_D; ++i) {
            float got = out[i];
            float expected = ref[i];
            float diff = fabsf(got - expected);
            float abstol = tol * (1.0f + fabsf(expected));
            if (diff > abstol) {
                if (errors < 10) {
                    int b = i / EMB_D, d = i % EMB_D;
                    printf("MISMATCH [bag=%d,d=%d]: got %.6f, expected %.6f, "
                           "diff %.6f\n",
                           b, d, got, expected, diff);
                }
                errors++;
            }
        }

        if (errors == 0)
            printf("PASS: all %d elements correct\n", EMB_B * EMB_D);
        else
            printf("FAIL: %d / %d mismatches\n", errors, EMB_B * EMB_D);

        free(ref);
    } else {
        printf("SKIP: result check disabled\n");
    }

    free(table_shadow);
    free(indices_shadow);
    free(offsets_shadow);
    embedding_bag_free_all();

    return (errors > 0) ? 1 : 0;
}
