#include "blend_common.h"

static float subtract_comp(float in_c, float layer_c)
{
    return in_c * 255.0f - layer_c;
}

PyObject *blend_subtract(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, subtract_comp, 1);
}
