#include "blend_common.h"

static float hard_light_comp(float in_c, float layer_c)
{
    if (layer_c > 0.5f) {
        float value = 1.0f - ((1.0f - in_c) * (1.0f - (layer_c - 0.5f) * 2.0f));
        if (value > 1.0f) {
            return 1.0f;
        }
        return value;
    }

    float value = in_c * (layer_c * 2.0f);
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static __m128 hard_light_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 half = _mm_set1_ps(0.5f);
    __m128 mask = _mm_cmpgt_ps(layer_c, half);
    __m128 low = _mm_min_ps(_mm_mul_ps(in_c, _mm_mul_ps(layer_c, _mm_set1_ps(2.0f))),
                            _mm_set1_ps(1.0f));
    __m128 high_term = _mm_sub_ps(layer_c, half);
    high_term = _mm_mul_ps(high_term, _mm_set1_ps(2.0f));
    __m128 high = _mm_sub_ps(_mm_set1_ps(1.0f),
                             _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), in_c),
                                        _mm_sub_ps(_mm_set1_ps(1.0f), high_term)));
    high = _mm_min_ps(high, _mm_set1_ps(1.0f));
    return _mm_blendv_ps(low, high, mask);
}

static __m256 hard_light_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 mask = _mm256_cmp_ps(layer_c, half, _CMP_GT_OQ);
    __m256 low = _mm256_min_ps(_mm256_mul_ps(in_c, _mm256_mul_ps(layer_c, _mm256_set1_ps(2.0f))),
                               _mm256_set1_ps(1.0f));
    __m256 high_term = _mm256_sub_ps(layer_c, half);
    high_term = _mm256_mul_ps(high_term, _mm256_set1_ps(2.0f));
    __m256 high = _mm256_sub_ps(_mm256_set1_ps(1.0f),
                                _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), in_c),
                                              _mm256_sub_ps(_mm256_set1_ps(1.0f), high_term)));
    high = _mm256_min_ps(high, _mm256_set1_ps(1.0f));
    return _mm256_blendv_ps(low, high, mask);
}

PyObject *blend_hard_light(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(args, hard_light_comp, hard_light_comp_ps128,
                                 hard_light_comp_ps256, 0);
}
