#include "blend_common.h"

static float soft_light_comp(float in_c, float layer_c)
{
    float term1 = (1.0f - in_c) * in_c * layer_c;
    float term2 = in_c * (1.0f - (1.0f - in_c) * (1.0f - layer_c));
    return term1 + term2;
}

PyObject *blend_soft_light(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, soft_light_comp, 0);
}
