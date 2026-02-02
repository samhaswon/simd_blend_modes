#include "blend_common.h"

static inline float soft_light_comp(float in_c, float layer_c)
{
    float term1 = (1.0f - in_c) * in_c * layer_c;
    float term2 = in_c * (1.0f - (1.0f - in_c) * (1.0f - layer_c));
    return term1 + term2;
}

#if SIMD_BLEND_MODES_X86
static inline __m128 soft_light_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 one = _mm_set1_ps(1.0f);
    __m128 term1 = _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(one, in_c), in_c), layer_c);
    __m128 inner = _mm_sub_ps(one,
                              _mm_mul_ps(_mm_sub_ps(one, in_c),
                                         _mm_sub_ps(one, layer_c)));
    return mul_add_ps128(in_c, inner, term1);
}

static inline __m256 soft_light_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 term1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(one, in_c), in_c), layer_c);
    __m256 inner = _mm256_sub_ps(one,
                                 _mm256_mul_ps(_mm256_sub_ps(one, in_c),
                                               _mm256_sub_ps(one, layer_c)));
    return mul_add_ps256(in_c, inner, term1);
}
#endif

PyObject *blend_soft_light(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(
        args,
        soft_light_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(soft_light_comp_ps128, soft_light_comp_ps256),
        0
    );
}
