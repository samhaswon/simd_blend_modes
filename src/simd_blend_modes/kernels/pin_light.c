#include "blend_common.h"

static inline float pin_light_comp(float in_c, float layer_c)
{
    if (layer_c < 0.5f) {
        return fminf(in_c, 2.0f * layer_c);
    }
    return fmaxf(in_c, 2.0f * layer_c - 1.0f);
}

#if SIMD_BLEND_MODES_X86
static inline __m128 pin_light_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 doubled = _mm_add_ps(layer_c, layer_c);
    __m128 low = _mm_min_ps(in_c, doubled);
    __m128 high = _mm_max_ps(in_c, _mm_sub_ps(doubled, _mm_set1_ps(1.0f)));
    return _mm_blendv_ps(high, low, _mm_cmplt_ps(layer_c, _mm_set1_ps(0.5f)));
}

static inline __m256 pin_light_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 doubled = _mm256_add_ps(layer_c, layer_c);
    __m256 low = _mm256_min_ps(in_c, doubled);
    __m256 high = _mm256_max_ps(in_c, _mm256_sub_ps(doubled, _mm256_set1_ps(1.0f)));
    return _mm256_blendv_ps(high, low,
                            _mm256_cmp_ps(layer_c, _mm256_set1_ps(0.5f), _CMP_LT_OQ));
}
#endif

PyObject *blend_pin_light(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_ratio_mode_simd(
        args,
        pin_light_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(pin_light_comp_ps128, pin_light_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(NULL, NULL),
        0
    );
}
