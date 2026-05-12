#include "blend_common.h"

static inline float linear_burn_comp(float in_c, float layer_c)
{
    return fmaxf(in_c + layer_c - 1.0f, 0.0f);
}

#if SIMD_BLEND_MODES_X86
static inline __m128 linear_burn_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_max_ps(_mm_add_ps(_mm_sub_ps(in_c, _mm_set1_ps(1.0f)), layer_c),
                      _mm_set1_ps(0.0f));
}

static inline __m256 linear_burn_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_max_ps(
        _mm256_add_ps(_mm256_sub_ps(in_c, _mm256_set1_ps(1.0f)), layer_c),
        _mm256_set1_ps(0.0f)
    );
}

static inline __m128 linear_burn_comp_u8_ps128(__m128i in_c, __m128i layer_c, __m128 inv255)
{
    __m128i sum = _mm_sub_epi32(_mm_add_epi32(in_c, layer_c), _mm_set1_epi32(255));
    return _mm_max_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), inv255), _mm_set1_ps(0.0f));
}

static inline __m256 linear_burn_comp_u8_ps256(__m256i in_c, __m256i layer_c, __m256 inv255)
{
    __m256i sum = _mm256_sub_epi32(_mm256_add_epi32(in_c, layer_c), _mm256_set1_epi32(255));
    return _mm256_max_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sum), inv255),
                         _mm256_set1_ps(0.0f));
}
#endif

PyObject *blend_linear_burn(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_ratio_mode_simd(
        args,
        linear_burn_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(linear_burn_comp_ps128, linear_burn_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(linear_burn_comp_u8_ps128,
                                      linear_burn_comp_u8_ps256),
        0
    );
}
