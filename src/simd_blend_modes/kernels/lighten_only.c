#include "blend_common.h"

PyObject *blend_lighten_only(PyObject *self, PyObject *args)
{
    return blend_return_background(args);
}
