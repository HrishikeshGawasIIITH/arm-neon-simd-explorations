/*
 * vector_add.c - element-wise addition of two float arrays: c = a + b.
 *
 * This is a memory-bandwidth-bound kernel: there is only one add per element,
 * so performance is limited by how fast data moves to and from memory rather
 * than by compute. It is the simplest possible NEON example - load, add, store.
 */
#include "kernels.h"
#include <arm_neon.h>

void vector_add_scalar(const float *a, const float *b, float *c, size_t n)
{
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vector_add_neon(const float *a, const float *b, float *c, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vaddq_f32(va, vb));
    }
    /* Tail for the remaining (< 4) elements. */
    for (; i < n; i++)
        c[i] = a[i] + b[i];
}
