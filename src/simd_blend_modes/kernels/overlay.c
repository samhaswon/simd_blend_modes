#include "blend_common.h"

PyObject *blend_overlay(PyObject *self, PyObject *args)
{
    return blend_return_background(args);
}
