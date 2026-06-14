/*
 * collision.c - circle/circle collision detection.
 *
 * Two circles collide when the squared distance between their centres is no
 * greater than the squared sum of their radii. We test one "collider" circle
 * against a batch of n circles stored in de-interleaved arrays (separate xs,
 * ys, radii), which is the layout NEON loads most efficiently.
 */
#include "kernels.h"
#include <arm_neon.h>

void collision_scalar(const float *xs, const float *ys, const float *radii,
                      size_t n, float cx, float cy, float cr,
                      unsigned char *out)
{
    for (size_t i = 0; i < n; i++) {
        float dx = cx - xs[i];
        float dy = cy - ys[i];
        float dist2 = dx * dx + dy * dy;
        float rsum = cr + radii[i];
        out[i] = (dist2 <= rsum * rsum) ? 1 : 0;
    }
}

void collision_neon(const float *xs, const float *ys, const float *radii,
                    size_t n, float cx, float cy, float cr,
                    unsigned char *out)
{
    /* Broadcast the single collider's properties into every lane. */
    float32x4_t vcx = vdupq_n_f32(cx);
    float32x4_t vcy = vdupq_n_f32(cy);
    float32x4_t vcr = vdupq_n_f32(cr);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(xs + i);
        float32x4_t y = vld1q_f32(ys + i);
        float32x4_t r = vld1q_f32(radii + i);

        float32x4_t dx = vsubq_f32(vcx, x);
        float32x4_t dy = vsubq_f32(vcy, y);
        float32x4_t dist2 = vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy));

        float32x4_t rsum = vaddq_f32(vcr, r);
        float32x4_t rsum2 = vmulq_f32(rsum, rsum);

        /* vcleq_f32 sets all bits of a lane to 1 where dist2 <= rsum2. */
        uint32x4_t mask = vcleq_f32(dist2, rsum2);
        out[i + 0] = (unsigned char)(1u & vgetq_lane_u32(mask, 0));
        out[i + 1] = (unsigned char)(1u & vgetq_lane_u32(mask, 1));
        out[i + 2] = (unsigned char)(1u & vgetq_lane_u32(mask, 2));
        out[i + 3] = (unsigned char)(1u & vgetq_lane_u32(mask, 3));
    }

    /* Tail: handle the remaining (< 4) circles with scalar code. */
    for (; i < n; i++) {
        float dx = cx - xs[i];
        float dy = cy - ys[i];
        float dist2 = dx * dx + dy * dy;
        float rsum = cr + radii[i];
        out[i] = (dist2 <= rsum * rsum) ? 1 : 0;
    }
}
