#include "blend_common.h"

static inline float addition_comp(float in_c, float layer_c)
{
    return in_c + layer_c;
}

#if SIMD_BLEND_MODES_X86
static inline __m128 addition_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_add_ps(in_c, layer_c);
}

static inline __m256 addition_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_add_ps(in_c, layer_c);
}

static inline __m128 addition_comp_u8_ps128(__m128i in_c, __m128i layer_c, __m128 inv255)
{
    __m128i sum = _mm_add_epi32(in_c, layer_c);
    return _mm_mul_ps(_mm_cvtepi32_ps(sum), inv255);
}

static inline __m256 addition_comp_u8_ps256(__m256i in_c, __m256i layer_c, __m256 inv255)
{
    __m256i sum = _mm256_add_epi32(in_c, layer_c);
    return _mm256_mul_ps(_mm256_cvtepi32_ps(sum), inv255);
}
#endif

PyObject *blend_addition(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(
        args,
        addition_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(addition_comp_ps128, addition_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(addition_comp_u8_ps128, addition_comp_u8_ps256),
        1
    );
}
