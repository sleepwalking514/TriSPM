#ifndef VECTOR_ADD_LAUNCHER_H
#define VECTOR_ADD_LAUNCHER_H

#include <stdint.h>

/* Launch the Triton kernel over a 3-D grid (sequential, for gem5 SE mode). */
void vector_add_launch(
    int32_t gridX, int32_t gridY, int32_t gridZ
    , void* arg0, void* arg1, void* arg2);

#endif /* VECTOR_ADD_LAUNCHER_H */
