#include "blend_common.h"

PyObject *blend_darken_only(PyObject *self, PyObject *args)
{
    return blend_return_background(args);
}
