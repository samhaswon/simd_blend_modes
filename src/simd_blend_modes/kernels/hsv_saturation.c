#include "blend_common.h"

static inline rgb_triplet hsv_saturation_comp(rgb_triplet bg, rgb_triplet fg)
{
    hsv_triplet bg_hsv = rgb_to_hsv_triplet(bg);
    hsv_triplet fg_hsv = rgb_to_hsv_triplet(fg);
    bg_hsv.s = fg_hsv.s;
    return hsv_to_rgb_triplet(bg_hsv);
}

#if SIMD_BLEND_MODES_X86
static inline void hsv_saturation_comp_ps128(__m128 bg_r, __m128 bg_g, __m128 bg_b,
                                             __m128 fg_r, __m128 fg_g, __m128 fg_b,
                                             __m128 *out_r, __m128 *out_g, __m128 *out_b)
{
    __m128 bg_h, bg_s, bg_v;
    __m128 fg_h, fg_s, fg_v;
    rgb_to_hsv_ps128(bg_r, bg_g, bg_b, &bg_h, &bg_s, &bg_v);
    rgb_to_hsv_ps128(fg_r, fg_g, fg_b, &fg_h, &fg_s, &fg_v);
    hsv_to_rgb_ps128(bg_h, fg_s, bg_v, out_r, out_g, out_b);
}

static inline void hsv_saturation_comp_ps256(__m256 bg_r, __m256 bg_g, __m256 bg_b,
                                             __m256 fg_r, __m256 fg_g, __m256 fg_b,
                                             __m256 *out_r, __m256 *out_g, __m256 *out_b)
{
    __m256 bg_h, bg_s, bg_v;
    __m256 fg_h, fg_s, fg_v;
    rgb_to_hsv_ps256(bg_r, bg_g, bg_b, &bg_h, &bg_s, &bg_v);
    rgb_to_hsv_ps256(fg_r, fg_g, fg_b, &fg_h, &fg_s, &fg_v);
    hsv_to_rgb_ps256(bg_h, fg_s, bg_v, out_r, out_g, out_b);
}
#endif

PyObject *blend_hsv_saturation(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_rgb_mode_simd(
        args,
        hsv_saturation_comp,
        SIMD_BLEND_MODES_RGB_SIMD_ARGS(hsv_saturation_comp_ps128,
                                       hsv_saturation_comp_ps256)
    );
}
