#include "blend_common.h"

static float addition_comp(float in_c, float layer_c)
{
    return in_c + layer_c;
}

static __m128 addition_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_add_ps(in_c, layer_c);
}

static __m256 addition_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_add_ps(in_c, layer_c);
}

PyObject *blend_addition(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(args, addition_comp, addition_comp_ps128,
                                 addition_comp_ps256, 1);
}
