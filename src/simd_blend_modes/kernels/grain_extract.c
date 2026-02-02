#include "blend_common.h"

static float grain_extract_comp(float in_c, float layer_c)
{
    return clamp01(in_c - layer_c + 0.5f);
}

PyObject *blend_grain_extract(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, grain_extract_comp, 0);
}
