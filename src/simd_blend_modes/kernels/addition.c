#include "blend_common.h"

static float addition_comp(float in_c, float layer_c)
{
    return in_c + layer_c;
}

PyObject *blend_addition(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, addition_comp, 1);
}
