#include "blend_common.h"

static inline rgb_triplet hsv_hue_comp(rgb_triplet bg, rgb_triplet fg)
{
    hsv_triplet bg_hsv = rgb_to_hsv_triplet(bg);
    hsv_triplet fg_hsv = rgb_to_hsv_triplet(fg);
    if (fg_hsv.s != 0.0f) {
        bg_hsv.h = fg_hsv.h;
    }
    return hsv_to_rgb_triplet(bg_hsv);
}

#if SIMD_BLEND_MODES_X86
static inline void hsv_hue_comp_ps128(__m128 bg_r, __m128 bg_g, __m128 bg_b,
                                      __m128 fg_r, __m128 fg_g, __m128 fg_b,
                                      __m128 *out_r, __m128 *out_g, __m128 *out_b)
{
    __m128 bg_h, bg_s, bg_v;
    __m128 fg_h, fg_s, fg_v;
    rgb_to_hsv_ps128(bg_r, bg_g, bg_b, &bg_h, &bg_s, &bg_v);
    rgb_to_hsv_ps128(fg_r, fg_g, fg_b, &fg_h, &fg_s, &fg_v);
    bg_h = select_ps128(_mm_cmpneq_ps(fg_s, _mm_set1_ps(0.0f)), fg_h, bg_h);
    hsv_to_rgb_ps128(bg_h, bg_s, bg_v, out_r, out_g, out_b);
}

static inline void hsv_hue_comp_ps256(__m256 bg_r, __m256 bg_g, __m256 bg_b,
                                      __m256 fg_r, __m256 fg_g, __m256 fg_b,
                                      __m256 *out_r, __m256 *out_g, __m256 *out_b)
{
    __m256 bg_h, bg_s, bg_v;
    __m256 fg_h, fg_s, fg_v;
    rgb_to_hsv_ps256(bg_r, bg_g, bg_b, &bg_h, &bg_s, &bg_v);
    rgb_to_hsv_ps256(fg_r, fg_g, fg_b, &fg_h, &fg_s, &fg_v);
    bg_h = select_ps256(_mm256_cmp_ps(fg_s, _mm256_set1_ps(0.0f), _CMP_NEQ_OQ),
                        fg_h, bg_h);
    hsv_to_rgb_ps256(bg_h, bg_s, bg_v, out_r, out_g, out_b);
}
#endif

PyObject *blend_hsv_hue(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_rgb_mode_simd(
        args,
        hsv_hue_comp,
        SIMD_BLEND_MODES_RGB_SIMD_ARGS(hsv_hue_comp_ps128, hsv_hue_comp_ps256)
    );
}
