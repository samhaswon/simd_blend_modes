#include "blend_common.h"

PyObject *blend_soft_light(PyObject *self, PyObject *args)
{
    return blend_return_background(args);
}
