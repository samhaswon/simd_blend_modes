#include "blend_common.h"

static inline float multiply_comp(float in_c, float layer_c)
{
    return in_c * layer_c;
}

#if SIMD_BLEND_MODES_X86
static inline __m128 multiply_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_mul_ps(in_c, layer_c);
}

static inline __m256 multiply_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_mul_ps(in_c, layer_c);
}

static inline __m128 multiply_comp_u8_ps128(__m128i in_c, __m128i layer_c, __m128 inv255)
{
    __m128i prod = _mm_mullo_epi32(in_c, layer_c);
    __m128 inv255_sq = _mm_mul_ps(inv255, inv255);
    return _mm_mul_ps(_mm_cvtepi32_ps(prod), inv255_sq);
}

static inline __m256 multiply_comp_u8_ps256(__m256i in_c, __m256i layer_c, __m256 inv255)
{
    __m256i prod = _mm256_mullo_epi32(in_c, layer_c);
    __m256 inv255_sq = _mm256_mul_ps(inv255, inv255);
    return _mm256_mul_ps(_mm256_cvtepi32_ps(prod), inv255_sq);
}
#endif

PyObject *blend_multiply(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(
        args,
        multiply_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(multiply_comp_ps128, multiply_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(multiply_comp_u8_ps128, multiply_comp_u8_ps256),
        0
    );
}
