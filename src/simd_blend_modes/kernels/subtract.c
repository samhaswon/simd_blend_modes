#include "blend_common.h"

PyObject *blend_subtract(PyObject *self, PyObject *args)
{
    return blend_return_background(args);
}
