/*
 * matrix_multiply.c - square matrix multiply C = A * B (column-major).
 *
 * Matrices are n x n with n a multiple of 4, stored column-major so that a
 * column of four consecutive values maps onto one 128-bit NEON register.
 *
 * The NEON version walks the result in 4x4 blocks. For each block it keeps four
 * accumulator registers (the four result columns) and updates them with
 * vfmaq_laneq_f32 - a fused multiply-accumulate that multiplies a whole vector
 * by one lane of another. This is compute-bound, so NEON's parallel FMAs give
 * the largest speedup of all the kernels here.
 *
 * Adapted from the Arm NEON matrix-multiply tutorial example.
 */
#include "kernels.h"
#include <arm_neon.h>

void matmul_scalar(const float *A, const float *B, float *C, unsigned n)
{
    for (unsigned i = 0; i < n; i++) {
        for (unsigned j = 0; j < n; j++) {
            float acc = 0.0f;
            for (unsigned k = 0; k < n; k++)
                acc += A[n * k + i] * B[n * j + k];
            C[n * j + i] = acc;
        }
    }
}

void matmul_neon(const float *A, const float *B, float *C, unsigned n)
{
    for (unsigned i = 0; i < n; i += 4) {
        for (unsigned j = 0; j < n; j += 4) {
            /* Four accumulators = the four columns of this 4x4 result block. */
            float32x4_t C0 = vmovq_n_f32(0);
            float32x4_t C1 = vmovq_n_f32(0);
            float32x4_t C2 = vmovq_n_f32(0);
            float32x4_t C3 = vmovq_n_f32(0);

            for (unsigned k = 0; k < n; k += 4) {
                unsigned a_idx = i + n * k;
                unsigned b_idx = n * j + k;

                /* Four columns of the 4x4 sub-block of A. */
                float32x4_t A0 = vld1q_f32(A + a_idx);
                float32x4_t A1 = vld1q_f32(A + a_idx + n);
                float32x4_t A2 = vld1q_f32(A + a_idx + 2 * n);
                float32x4_t A3 = vld1q_f32(A + a_idx + 3 * n);

                float32x4_t B0 = vld1q_f32(B + b_idx);
                C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
                C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
                C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
                C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

                float32x4_t B1 = vld1q_f32(B + b_idx + n);
                C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
                C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
                C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
                C1 = vfmaq_laneq_f32(C1, A3, B1, 3);

                float32x4_t B2 = vld1q_f32(B + b_idx + 2 * n);
                C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
                C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
                C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
                C2 = vfmaq_laneq_f32(C2, A3, B2, 3);

                float32x4_t B3 = vld1q_f32(B + b_idx + 3 * n);
                C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
                C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
                C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
                C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
            }

            unsigned c_idx = n * j + i;
            vst1q_f32(C + c_idx, C0);
            vst1q_f32(C + c_idx + n, C1);
            vst1q_f32(C + c_idx + 2 * n, C2);
            vst1q_f32(C + c_idx + 3 * n, C3);
        }
    }
}
