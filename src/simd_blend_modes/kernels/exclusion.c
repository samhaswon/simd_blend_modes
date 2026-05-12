#include "blend_common.h"

static inline float exclusion_comp(float in_c, float layer_c)
{
    return in_c + layer_c - 2.0f * in_c * layer_c;
}

#if SIMD_BLEND_MODES_X86
static inline __m128 exclusion_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_sub_ps(_mm_add_ps(in_c, layer_c),
                      _mm_mul_ps(_mm_mul_ps(in_c, layer_c), _mm_set1_ps(2.0f)));
}

static inline __m256 exclusion_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_sub_ps(_mm256_add_ps(in_c, layer_c),
                         _mm256_mul_ps(_mm256_mul_ps(in_c, layer_c),
                                       _mm256_set1_ps(2.0f)));
}
#endif

PyObject *blend_exclusion(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_ratio_mode_simd(
        args,
        exclusion_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(exclusion_comp_ps128, exclusion_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(NULL, NULL),
        0
    );
}
