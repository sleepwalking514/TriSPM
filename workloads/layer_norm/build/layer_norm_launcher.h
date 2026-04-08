#ifndef LAYER_NORM_LAUNCHER_H
#define LAYER_NORM_LAUNCHER_H

#include <stdint.h>

/* Launch the Triton kernel over a 3-D grid (sequential, for gem5 SE mode). */
void layer_norm_launch(
    int32_t gridX, int32_t gridY, int32_t gridZ
    , void* arg0, void* arg1, void* arg2, void* arg3, int32_t arg4);

#endif /* LAYER_NORM_LAUNCHER_H */
