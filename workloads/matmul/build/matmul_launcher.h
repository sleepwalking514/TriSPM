#ifndef MATMUL_LAUNCHER_H
#define MATMUL_LAUNCHER_H

#include <stdint.h>

/* Launch the Triton kernel over a 3-D grid (sequential, for gem5 SE mode). */
void matmul_launch(
    int32_t gridX, int32_t gridY, int32_t gridZ
    , void* arg0, void* arg1, void* arg2);

#endif /* MATMUL_LAUNCHER_H */
