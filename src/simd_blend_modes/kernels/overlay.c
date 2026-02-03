#include "blend_common.h"

static inline float overlay_comp(float in_c, float layer_c)
{
    if (in_c < 0.5f) {
        return 2.0f * in_c * layer_c;
    }
    return 1.0f - (2.0f * (1.0f - in_c) * (1.0f - layer_c));
}

#if SIMD_BLEND_MODES_X86
static inline __m128 overlay_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 half = _mm_set1_ps(0.5f);
    __m128 mask = _mm_cmplt_ps(in_c, half);
    __m128 low = _mm_mul_ps(_mm_mul_ps(in_c, layer_c), _mm_set1_ps(2.0f));
    __m128 high = _mm_sub_ps(_mm_set1_ps(1.0f),
                             _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), in_c),
                                                    _mm_sub_ps(_mm_set1_ps(1.0f), layer_c)),
                                        _mm_set1_ps(2.0f)));
    return _mm_blendv_ps(high, low, mask);
}

static inline __m256 overlay_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 mask = _mm256_cmp_ps(in_c, half, _CMP_LT_OQ);
    __m256 low = _mm256_mul_ps(_mm256_mul_ps(in_c, layer_c), _mm256_set1_ps(2.0f));
    __m256 high = _mm256_sub_ps(_mm256_set1_ps(1.0f),
                                _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), in_c),
                                                           _mm256_sub_ps(_mm256_set1_ps(1.0f), layer_c)),
                                              _mm256_set1_ps(2.0f)));
    return _mm256_blendv_ps(high, low, mask);
}
#endif

PyObject *blend_overlay(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(
        args,
        overlay_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(overlay_comp_ps128, overlay_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(NULL, NULL),
        0
    );
}
