#include "blend_common.h"

static float grain_merge_comp(float in_c, float layer_c)
{
    return clamp01(in_c + layer_c - 0.5f);
}

PyObject *blend_grain_merge(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, grain_merge_comp, 0);
}
