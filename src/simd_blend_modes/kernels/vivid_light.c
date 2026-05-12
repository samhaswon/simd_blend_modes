#include "blend_common.h"

static inline float vivid_light_comp(float in_c, float layer_c)
{
    return vivid_light_channel(in_c, layer_c);
}

#if SIMD_BLEND_MODES_X86
static inline __m128 vivid_light_comp_ps128(__m128 in_c, __m128 layer_c)
{
    __m128 one = _mm_set1_ps(1.0f);
    __m128 zero = _mm_set1_ps(0.0f);
    __m128 half = _mm_set1_ps(0.5f);
    __m128 burn_layer = _mm_add_ps(layer_c, layer_c);
    __m128 dodge_layer = _mm_sub_ps(_mm_add_ps(layer_c, layer_c), one);
    __m128 burned = _mm_sub_ps(
        one,
        _mm_min_ps(one, _mm_div_ps(_mm_sub_ps(one, in_c), burn_layer))
    );
    __m128 burn_zero = _mm_cmpeq_ps(burn_layer, zero);
    __m128 in_one = _mm_cmpeq_ps(in_c, one);
    burned = _mm_blendv_ps(burned, _mm_blendv_ps(zero, one, in_one), burn_zero);

    __m128 dodged = _mm_min_ps(one, _mm_div_ps(in_c, _mm_sub_ps(one, dodge_layer)));
    dodged = _mm_blendv_ps(dodged, one, _mm_cmpeq_ps(dodge_layer, one));
    return _mm_blendv_ps(dodged, burned, _mm_cmplt_ps(layer_c, half));
}

static inline __m256 vivid_light_comp_ps256(__m256 in_c, __m256 layer_c)
{
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 zero = _mm256_set1_ps(0.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 burn_layer = _mm256_add_ps(layer_c, layer_c);
    __m256 dodge_layer = _mm256_sub_ps(_mm256_add_ps(layer_c, layer_c), one);
    __m256 burned = _mm256_sub_ps(
        one,
        _mm256_min_ps(one, _mm256_div_ps(_mm256_sub_ps(one, in_c), burn_layer))
    );
    __m256 burn_zero = _mm256_cmp_ps(burn_layer, zero, _CMP_EQ_OQ);
    __m256 in_one = _mm256_cmp_ps(in_c, one, _CMP_EQ_OQ);
    burned = _mm256_blendv_ps(burned, _mm256_blendv_ps(zero, one, in_one), burn_zero);

    __m256 dodged = _mm256_min_ps(one, _mm256_div_ps(in_c, _mm256_sub_ps(one, dodge_layer)));
    dodged = _mm256_blendv_ps(dodged, one, _mm256_cmp_ps(dodge_layer, one, _CMP_EQ_OQ));
    return _mm256_blendv_ps(dodged, burned, _mm256_cmp_ps(layer_c, half, _CMP_LT_OQ));
}
#endif

PyObject *blend_vivid_light(PyObject *self, PyObject *args)
{
    (void)self;
    return blend_ratio_mode_simd(
        args,
        vivid_light_comp,
        SIMD_BLEND_MODES_SIMD_ARGS(vivid_light_comp_ps128, vivid_light_comp_ps256),
        SIMD_BLEND_MODES_SIMD_U8_ARGS(NULL, NULL),
        0
    );
}
