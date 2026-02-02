#include "blend_common.h"

static float darken_only_comp(float in_c, float layer_c)
{
    return fminf(in_c, layer_c);
}

PyObject *blend_darken_only(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, darken_only_comp, 0);
}
