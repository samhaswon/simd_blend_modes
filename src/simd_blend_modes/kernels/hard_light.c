#include "blend_common.h"

static float hard_light_comp(float in_c, float layer_c)
{
    if (layer_c > 0.5f) {
        float value = 1.0f - ((1.0f - in_c) * (1.0f - (layer_c - 0.5f) * 2.0f));
        if (value > 1.0f) {
            return 1.0f;
        }
        return value;
    }

    float value = in_c * (layer_c * 2.0f);
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

PyObject *blend_hard_light(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, hard_light_comp, 0);
}
