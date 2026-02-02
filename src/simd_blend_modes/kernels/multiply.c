#include "blend_common.h"

static inline float multiply_comp(float in_c, float layer_c)
{
    return in_c * layer_c;
}

static inline __m128 multiply_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_mul_ps(in_c, layer_c);
}

static inline __m256 multiply_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_mul_ps(in_c, layer_c);
}

PyObject *blend_multiply(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(args, multiply_comp, multiply_comp_ps128,
                                 multiply_comp_ps256, 0);
}
