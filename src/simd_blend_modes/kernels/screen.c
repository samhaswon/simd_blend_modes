#include "blend_common.h"

static inline float screen_comp(float in_c, float layer_c)
{
    return 1.0f - (1.0f - in_c) * (1.0f - layer_c);
}

#if SIMD_BLEND_MODES_X86
static inline __m128 screen_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_sub_ps(_mm_set1_ps(1.0f),
                      _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.0f), in_c),
                                 _mm_sub_ps(_mm_set1_ps(1.0f), layer_c)));
}

static inline __m256 screen_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_sub_ps(_mm256_set1_ps(1.0f),
                         _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), in_c),
                                       _mm256_sub_ps(_mm256_set1_ps(1.0f), layer_c)));
}

static inline __m128 screen_comp_u8_ps128(__m128i in_c, __m128i layer_c, __m128 inv255)
{
    __m128i inv_in = _mm_sub_epi32(_mm_set1_epi32(255), in_c);
    __m128i inv_layer = _mm_sub_epi32(_mm_set1_epi32(255), layer_c);
    __m128i prod = _mm_mullo_epi32(inv_in, inv_layer);
    __m128 inv255_sq = _mm_mul_ps(inv255, inv255);
    return _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(_mm_cvtepi32_ps(prod), inv255_sq));
}

static inline __m256 screen_comp_u8_ps256(__m256i in_c, __m256i layer_c, __m256 inv255)
{
    __m256i inv_in = _mm256_sub_epi32(_mm256_set1_epi32(255), in_c);
    __m256i inv_layer = _mm256_sub_epi32(_mm256_set1_epi32(255), layer_c);
    __m256i prod = _mm256_mullo_epi32(inv_in, inv_layer);
    __m256 inv255_sq = _mm256_mul_ps(inv255, inv255);
    return _mm256_sub_ps(_mm256_set1_ps(1.0f),
                         _mm256_mul_ps(_mm256_cvtepi32_ps(prod), inv255_sq));
}
#endif

PyObject *blend_screen(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(
        args,
        screen_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(screen_comp_ps128, screen_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(screen_comp_u8_ps128, screen_comp_u8_ps256),
        0
    );
}
