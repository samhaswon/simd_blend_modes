#include "blend_common.h"

static inline float burn_comp(float in_c, float layer_c)
{
    if (layer_c == 0.0f) {
        return in_c == 1.0f ? 1.0f : 0.0f;
    }
    return 1.0f - fminf(1.0f, (1.0f - in_c) / layer_c);
}

#if SIMD_BLEND_MODES_X86
static inline __m128 burn_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 one = _mm_set1_ps(1.0f);
    __m128 zero = _mm_set1_ps(0.0f);
    __m128 value = _mm_sub_ps(one, _mm_min_ps(one, _mm_div_ps(_mm_sub_ps(one, in_c), layer_c)));
    __m128 layer_zero = _mm_cmpeq_ps(layer_c, zero);
    __m128 in_one = _mm_cmpeq_ps(in_c, one);
    __m128 zero_or_one = _mm_blendv_ps(zero, one, in_one);
    return _mm_blendv_ps(value, zero_or_one, layer_zero);
}

static inline __m256 burn_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 zero = _mm256_set1_ps(0.0f);
    __m256 value = _mm256_sub_ps(
        one,
        _mm256_min_ps(one, _mm256_div_ps(_mm256_sub_ps(one, in_c), layer_c))
    );
    __m256 layer_zero = _mm256_cmp_ps(layer_c, zero, _CMP_EQ_OQ);
    __m256 in_one = _mm256_cmp_ps(in_c, one, _CMP_EQ_OQ);
    __m256 zero_or_one = _mm256_blendv_ps(zero, one, in_one);
    return _mm256_blendv_ps(value, zero_or_one, layer_zero);
}
#endif

PyObject *blend_burn(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_ratio_mode_simd(
        args,
        burn_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(burn_comp_ps128, burn_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(NULL, NULL),
        0
    );
}
