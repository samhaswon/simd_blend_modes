#include "blend_common.h"

static float overlay_comp(float in_c, float layer_c)
{
    if (in_c < 0.5f) {
        return 2.0f * in_c * layer_c;
    }
    return 1.0f - (2.0f * (1.0f - in_c) * (1.0f - layer_c));
}

PyObject *blend_overlay(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, overlay_comp, 0);
}
