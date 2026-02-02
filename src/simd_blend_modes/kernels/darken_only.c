#include "blend_common.h"

static inline float darken_only_comp(float in_c, float layer_c)
{
    return fminf(in_c, layer_c);
}

static inline __m128 darken_only_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_min_ps(in_c, layer_c);
}

static inline __m256 darken_only_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_min_ps(in_c, layer_c);
}

PyObject *blend_darken_only(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(args, darken_only_comp, darken_only_comp_ps128,
                                 darken_only_comp_ps256, 0);
}
