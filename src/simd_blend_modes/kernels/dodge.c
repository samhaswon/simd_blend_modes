#include "blend_common.h"

static float dodge_comp(float in_c, float layer_c)
{
    float denom = 1.0f - layer_c;
    if (denom <= 0.0f) {
        return 1.0f;
    }
    float value = in_c / denom;
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

PyObject *blend_dodge(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, dodge_comp, 0);
}
