#include "blend_common.h"

static inline rgb_triplet lch_color_comp(rgb_triplet bg, rgb_triplet fg)
{
    return lch_blend_triplet(bg, fg, 0, 1, 1);
}

#if SIMD_BLEND_MODES_X86
static inline void lch_color_comp_ps128(__m128 bg_r, __m128 bg_g, __m128 bg_b,
                                        __m128 fg_r, __m128 fg_g, __m128 fg_b,
                                        __m128 *out_r, __m128 *out_g, __m128 *out_b)
{
    apply_rgb_comp_ps128(bg_r, bg_g, bg_b, fg_r, fg_g, fg_b,
                         out_r, out_g, out_b, lch_color_comp);
}

static inline void lch_color_comp_ps256(__m256 bg_r, __m256 bg_g, __m256 bg_b,
                                        __m256 fg_r, __m256 fg_g, __m256 fg_b,
                                        __m256 *out_r, __m256 *out_g, __m256 *out_b)
{
    apply_rgb_comp_ps256(bg_r, bg_g, bg_b, fg_r, fg_g, fg_b,
                         out_r, out_g, out_b, lch_color_comp);
}
#endif

PyObject *blend_lch_color(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_rgb_mode_simd(
        args,
        lch_color_comp,
        SIMD_BLEND_MODES_RGB_SIMD_ARGS(lch_color_comp_ps128, lch_color_comp_ps256)
    );
}
