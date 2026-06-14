/*
 * fir_filter.c - finite impulse response (FIR) filter, a core DSP operation.
 *
 * Valid convolution: y[i] = sum_{k=0}^{ntaps-1} h[k] * x[i + k]
 *
 * The NEON version vectorises over the OUTPUT index: it computes four output
 * samples at once. For each tap it broadcasts the coefficient h[k] across a
 * register and fused-multiply-accumulates it with four input samples. This
 * keeps the inner loop branch-free and reuses each loaded input vector.
 */
#include "kernels.h"
#include <arm_neon.h>

void fir_scalar(const float *x, const float *h, float *y,
                size_t n_out, size_t ntaps)
{
    for (size_t i = 0; i < n_out; i++) {
        float acc = 0.0f;
        for (size_t k = 0; k < ntaps; k++)
            acc += h[k] * x[i + k];
        y[i] = acc;
    }
}

void fir_neon(const float *x, const float *h, float *y,
              size_t n_out, size_t ntaps)
{
    size_t i = 0;
    for (; i + 4 <= n_out; i += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t k = 0; k < ntaps; k++) {
            float32x4_t hv = vdupq_n_f32(h[k]);      /* coefficient in all lanes */
            float32x4_t xv = vld1q_f32(x + i + k);   /* four input samples       */
            acc = vfmaq_f32(acc, hv, xv);            /* acc += h[k] * x[i+k..]   */
        }
        vst1q_f32(y + i, acc);
    }
    /* Tail for the remaining (< 4) outputs. */
    for (; i < n_out; i++) {
        float acc = 0.0f;
        for (size_t k = 0; k < ntaps; k++)
            acc += h[k] * x[i + k];
        y[i] = acc;
    }
}
