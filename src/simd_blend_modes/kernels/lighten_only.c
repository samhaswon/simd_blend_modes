#include "blend_common.h"

static inline float lighten_only_comp(float in_c, float layer_c)
{
    return fmaxf(in_c, layer_c);
}

#if SIMD_BLEND_MODES_X86
static inline __m128 lighten_only_comp_ps128(__m128 in_c, __m128 layer_c)
{
    return _mm_max_ps(in_c, layer_c);
}

static inline __m256 lighten_only_comp_ps256(__m256 in_c, __m256 layer_c)
{
    return _mm256_max_ps(in_c, layer_c);
}
#endif

PyObject *blend_lighten_only(PyObject *self, PyObject *args)
{
    return blend_ratio_mode_simd(
        args,
        lighten_only_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(lighten_only_comp_ps128, lighten_only_comp_ps256),
        0
    );
}
