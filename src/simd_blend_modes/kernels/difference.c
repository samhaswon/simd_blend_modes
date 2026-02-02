#include "blend_common.h"

static float difference_comp(float in_c, float layer_c)
{
    return fabsf(in_c - layer_c);
}

PyObject *blend_difference(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, difference_comp, 0);
}
