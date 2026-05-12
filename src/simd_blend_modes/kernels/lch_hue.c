#include "blend_common.h"

static inline rgb_triplet lch_hue_comp(rgb_triplet bg, rgb_triplet fg)
{
    return lch_blend_triplet(bg, fg, 0, 0, 1);
}

#if SIMD_BLEND_MODES_X86
static inline void lch_hue_comp_ps128(__m128 bg_r, __m128 bg_g, __m128 bg_b,
                                      __m128 fg_r, __m128 fg_g, __m128 fg_b,
                                      __m128 *out_r, __m128 *out_g, __m128 *out_b)
{
    lch_blend_ps128(bg_r, bg_g, bg_b, fg_r, fg_g, fg_b, 0, 0, 1,
                    out_r, out_g, out_b);
}

static inline void lch_hue_comp_ps256(__m256 bg_r, __m256 bg_g, __m256 bg_b,
                                      __m256 fg_r, __m256 fg_g, __m256 fg_b,
                                      __m256 *out_r, __m256 *out_g, __m256 *out_b)
{
    lch_blend_ps256(bg_r, bg_g, bg_b, fg_r, fg_g, fg_b, 0, 0, 1,
                    out_r, out_g, out_b);
}
#endif

PyObject *blend_lch_hue(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_rgb_mode_simd(
        args,
        lch_hue_comp,
        SIMD_BLEND_MODES_RGB_SIMD_ARGS(lch_hue_comp_ps128, lch_hue_comp_ps256)
    );
}
