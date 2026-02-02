#include "blend_common.h"

static float divide_comp(float in_c, float layer_c)
{
    float denom = (1.0f / 255.0f) + layer_c;
    float value = (256.0f / 255.0f * in_c) / denom;
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

PyObject *blend_divide(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, divide_comp, 0);
}
