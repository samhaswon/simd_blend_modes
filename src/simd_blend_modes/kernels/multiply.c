#include "blend_common.h"

static float multiply_comp(float in_c, float layer_c)
{
    return in_c * layer_c;
}

PyObject *blend_multiply(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, multiply_comp, 0);
}
