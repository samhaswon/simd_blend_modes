#include "blend_common.h"

static inline float divide_comp(float in_c, float layer_c)
{
    float denom = (1.0f / 255.0f) + layer_c;
    float value = (256.0f / 255.0f * in_c) / denom;
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static inline __m128 divide_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 denom = _mm_add_ps(_mm_set1_ps(1.0f / 255.0f), layer_c);
    __m128 value = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(256.0f / 255.0f), in_c), denom);
    return _mm_min_ps(value, _mm_set1_ps(1.0f));
}

static inline __m256 divide_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f / 255.0f), layer_c);
    __m256 value = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(256.0f / 255.0f), in_c), denom);
    return _mm256_min_ps(value, _mm256_set1_ps(1.0f));
}

PyObject *blend_divide(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(args, divide_comp, divide_comp_ps128,
                                 divide_comp_ps256, 0);
}
