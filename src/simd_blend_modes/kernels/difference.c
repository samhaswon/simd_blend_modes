#include "blend_common.h"

static inline float difference_comp(float in_c, float layer_c)
{
    return fabsf(in_c - layer_c);
}

static inline __m128 difference_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 diff = _mm_sub_ps(in_c, layer_c);
    return _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
}

static inline __m256 difference_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 diff = _mm256_sub_ps(in_c, layer_c);
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);
}

PyObject *blend_difference(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(args, difference_comp, difference_comp_ps128,
                                 difference_comp_ps256, 0);
}
