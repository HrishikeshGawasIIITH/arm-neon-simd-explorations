/*
 * kernels.h - scalar and NEON versions of every benchmarked kernel.
 *
 * Each kernel comes in two flavours with identical results:
 *   *_scalar : plain C, what a compiler sees without help
 *   *_neon   : hand-written Arm NEON intrinsics (arm_neon.h)
 *
 * The benchmark in bench/benchmark.c links these and times scalar vs NEON.
 * All kernels are written for AArch64 (Raspberry Pi 4B / Cortex-A72), where
 * NEON is part of the baseline ISA.
 */
#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>

/* Collision detection (from "PPT 1", advanced de-interleaved example).
 * Tests a single collider circle (cx, cy, cr) against n circles held in
 * de-interleaved arrays. out[i] = 1 if circle i overlaps the collider. */
void collision_scalar(const float *xs, const float *ys, const float *radii,
                      size_t n, float cx, float cy, float cr,
                      unsigned char *out);
void collision_neon(const float *xs, const float *ys, const float *radii,
                    size_t n, float cx, float cy, float cr,
                    unsigned char *out);

/* Element-wise vector add: c[i] = a[i] + b[i]. Memory-bandwidth bound. */
void vector_add_scalar(const float *a, const float *b, float *c, size_t n);
void vector_add_neon(const float *a, const float *b, float *c, size_t n);

/* Square matrix multiply C = A * B (column-major, n must be a multiple of 4).
 * NEON version multiplies in 4x4 blocks with fused multiply-accumulate. */
void matmul_scalar(const float *A, const float *B, float *C, unsigned n);
void matmul_neon(const float *A, const float *B, float *C, unsigned n);

/* FIR filter (valid convolution): y[i] = sum_k h[k] * x[i+k].
 * n_out outputs, ntaps coefficients. Classic DSP multiply-accumulate. */
void fir_scalar(const float *x, const float *h, float *y,
                size_t n_out, size_t ntaps);
void fir_neon(const float *x, const float *h, float *y,
              size_t n_out, size_t ntaps);

#endif /* KERNELS_H */
