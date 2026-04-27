#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "vector_add_launcher.h"

/*
 * Test harness for the Triton-compiled vector_add kernel.
 *
 * Build with -DBLOCK_SIZE=64 -DSIZE=4096 (injected by build_kernel.sh
 * from config.sh).
 *
 * The generated launcher (vector_add_launcher.c) handles the grid
 * dispatch loop and the extern kernel declaration, so the harness
 * only deals with data init and verification.
 */

#ifndef BLOCK_SIZE
#error "BLOCK_SIZE must be defined via -D flag"
#endif
#ifndef SIZE
#error "SIZE must be defined via -D flag"
#endif
#ifndef CHECK_RESULT
#define CHECK_RESULT 1
#endif

#define GRID_X  ((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE)

int main(void)
{
    printf("vector_add: SIZE=%d  BLOCK_SIZE=%d  GRID_X=%d  check=%d\n",
           SIZE, BLOCK_SIZE, GRID_X, CHECK_RESULT);

    float *x   = (float *)vector_add_alloc(0, SIZE * sizeof(float));
    float *y   = (float *)vector_add_alloc(1, SIZE * sizeof(float));
    float *out = (float *)vector_add_alloc(2, SIZE * sizeof(float));

    if (!x || !y || !out) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < SIZE; i++) {
        x[i]   = (float)(i + 1);
        y[i]   = (float)(i + 1) * 0.5f;
        out[i] = 0.0f;
    }

    /* Launch kernel over the 1-D grid via the generated launcher. */
    vector_add_launch(GRID_X, 1, 1, x, y, out);

#if CHECK_RESULT
    /* Verify */
    int errors = 0;
    for (int i = 0; i < SIZE; i++) {
        float expected = x[i] + y[i];
        if (fabsf(out[i] - expected) > 1e-5f) {
            if (errors < 10)
                printf("MISMATCH [%d]: got %.4f, expected %.4f\n",
                       i, out[i], expected);
            errors++;
        }
    }

    if (errors == 0)
        printf("PASS: all %d elements correct\n", SIZE);
    else
        printf("FAIL: %d / %d mismatches\n", errors, SIZE);
#else
    printf("SKIP: result check disabled\n");
#endif

    vector_add_free_all();

#if CHECK_RESULT
    return (errors > 0) ? 1 : 0;
#else
    return 0;
#endif
}
