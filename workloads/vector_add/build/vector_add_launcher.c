#include "vector_add_launcher.h"

/* Triton-generated kernel symbol (from the cross-compiled .s file). */
extern void vector_add(void*, void*, void*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

void vector_add_launch(
    int32_t gridX, int32_t gridY, int32_t gridZ
    , void* arg0, void* arg1, void* arg2)
{
    for (int32_t z = 0; z < gridZ; ++z)
        for (int32_t y = 0; y < gridY; ++y)
            for (int32_t x = 0; x < gridX; ++x)
                vector_add(arg0, arg1, arg2, x, y, z, gridX, gridY, gridZ);
}
