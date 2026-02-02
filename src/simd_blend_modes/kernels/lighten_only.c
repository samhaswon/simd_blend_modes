#include "blend_common.h"

static float lighten_only_comp(float in_c, float layer_c)
{
    return fmaxf(in_c, layer_c);
}

PyObject *blend_lighten_only(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, lighten_only_comp, 0);
}
