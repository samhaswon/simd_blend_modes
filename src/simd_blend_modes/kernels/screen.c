#include "blend_common.h"

PyObject *blend_screen(PyObject *self, PyObject *args)
{
    return blend_return_background(args);
}
