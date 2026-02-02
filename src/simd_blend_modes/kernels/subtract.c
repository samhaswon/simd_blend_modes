#include "blend_common.h"

static inline float subtract_comp(float in_c, float layer_c)
{
    return in_c * 255.0f - layer_c;
}

#if SIMD_BLEND_MODES_X86
static inline __m128 subtract_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_sub_ps(_mm_mul_ps(in_c, _mm_set1_ps(255.0f)), layer_c);
}

static inline __m256 subtract_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_sub_ps(_mm256_mul_ps(in_c, _mm256_set1_ps(255.0f)), layer_c);
}
#endif

PyObject *blend_subtract(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(
        args,
        subtract_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(subtract_comp_ps128, subtract_comp_ps256),
        1
    );
}
