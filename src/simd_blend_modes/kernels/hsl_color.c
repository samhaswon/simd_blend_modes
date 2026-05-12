#include "blend_common.h"

static inline rgb_triplet hsl_color_comp(rgb_triplet bg, rgb_triplet fg)
{
    hsl_triplet bg_hsl = rgb_to_hsl_triplet(bg);
    hsl_triplet fg_hsl = rgb_to_hsl_triplet(fg);
    hsl_triplet out_hsl = {fg_hsl.h, fg_hsl.s, bg_hsl.l};
    return hsl_to_rgb_triplet(out_hsl);
}

#if SIMD_BLEND_MODES_X86
static inline void hsl_color_comp_ps128(__m128 bg_r, __m128 bg_g, __m128 bg_b,
                                        __m128 fg_r, __m128 fg_g, __m128 fg_b,
                                        __m128 *out_r, __m128 *out_g, __m128 *out_b)
{
    __m128 bg_h, bg_s, bg_l;
    __m128 fg_h, fg_s, fg_l;
    rgb_to_hsl_ps128(bg_r, bg_g, bg_b, &bg_h, &bg_s, &bg_l);
    rgb_to_hsl_ps128(fg_r, fg_g, fg_b, &fg_h, &fg_s, &fg_l);
    hsl_to_rgb_ps128(fg_h, fg_s, bg_l, out_r, out_g, out_b);
}

static inline void hsl_color_comp_ps256(__m256 bg_r, __m256 bg_g, __m256 bg_b,
                                        __m256 fg_r, __m256 fg_g, __m256 fg_b,
                                        __m256 *out_r, __m256 *out_g, __m256 *out_b)
{
    __m256 bg_h, bg_s, bg_l;
    __m256 fg_h, fg_s, fg_l;
    rgb_to_hsl_ps256(bg_r, bg_g, bg_b, &bg_h, &bg_s, &bg_l);
    rgb_to_hsl_ps256(fg_r, fg_g, fg_b, &fg_h, &fg_s, &fg_l);
    hsl_to_rgb_ps256(fg_h, fg_s, bg_l, out_r, out_g, out_b);
}
#endif

PyObject *blend_hsl_color(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_rgb_mode_simd(
        args,
        hsl_color_comp,
        SIMD_BLEND_MODES_RGB_SIMD_ARGS(hsl_color_comp_ps128, hsl_color_comp_ps256)
    );
}
