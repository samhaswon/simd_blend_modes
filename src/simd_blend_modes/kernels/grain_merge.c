#include "blend_common.h"

static float grain_merge_comp(float in_c, float layer_c)
{
    return clamp01(in_c + layer_c - 0.5f);
}

static __m128 grain_merge_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 value = _mm_sub_ps(_mm_add_ps(in_c, layer_c), _mm_set1_ps(0.5f));
    return _mm_min_ps(_mm_max_ps(value, _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f));
}

static __m256 grain_merge_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 value = _mm256_sub_ps(_mm256_add_ps(in_c, layer_c), _mm256_set1_ps(0.5f));
    return _mm256_min_ps(_mm256_max_ps(value, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
}

PyObject *blend_grain_merge(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(args, grain_merge_comp, grain_merge_comp_ps128,
                                 grain_merge_comp_ps256, 0);
}
