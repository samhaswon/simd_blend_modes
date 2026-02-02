#include "blend_common.h"

static inline float dodge_comp(float in_c, float layer_c)
{
    float denom = 1.0f - layer_c;
    if (denom <= 0.0f) {
        return in_c <= 0.0f ? 0.0f : 1.0f;
    }
    float value = in_c / denom;
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static inline __m128 dodge_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 denom = _mm_sub_ps(_mm_set1_ps(1.0f), layer_c);
    __m128 value = _mm_div_ps(in_c, denom);
    value = _mm_min_ps(value, _mm_set1_ps(1.0f));
    __m128 denom_mask = _mm_cmple_ps(denom, _mm_set1_ps(0.0f));
    __m128 in_zero_mask = _mm_cmple_ps(in_c, _mm_set1_ps(0.0f));
    __m128 ones = _mm_set1_ps(1.0f);
    value = _mm_blendv_ps(value, ones, denom_mask);
    __m128 zero_mask = _mm_and_ps(denom_mask, in_zero_mask);
    value = _mm_blendv_ps(value, _mm_set1_ps(0.0f), zero_mask);
    return value;
}

static inline __m256 dodge_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 denom = _mm256_sub_ps(_mm256_set1_ps(1.0f), layer_c);
    __m256 value = _mm256_div_ps(in_c, denom);
    value = _mm256_min_ps(value, _mm256_set1_ps(1.0f));
    __m256 denom_mask = _mm256_cmp_ps(denom, _mm256_set1_ps(0.0f), _CMP_LE_OQ);
    __m256 in_zero_mask = _mm256_cmp_ps(in_c, _mm256_set1_ps(0.0f), _CMP_LE_OQ);
    __m256 ones = _mm256_set1_ps(1.0f);
    value = _mm256_blendv_ps(value, ones, denom_mask);
    __m256 zero_mask = _mm256_and_ps(denom_mask, in_zero_mask);
    value = _mm256_blendv_ps(value, _mm256_set1_ps(0.0f), zero_mask);
    return value;
}

PyObject *blend_dodge(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(args, dodge_comp, dodge_comp_ps128, dodge_comp_ps256, 0);
}
