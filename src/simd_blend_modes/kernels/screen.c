#include "blend_common.h"

static float screen_comp(float in_c, float layer_c)
{
    return 1.0f - (1.0f - in_c) * (1.0f - layer_c);
}

PyObject *blend_screen(PyObject *self, PyObject *args)
{
    return blend_ratio_mode(args, screen_comp, 0);
}
