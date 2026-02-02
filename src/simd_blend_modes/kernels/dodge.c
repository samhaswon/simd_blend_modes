#include "blend_common.h"

PyObject *blend_dodge(PyObject *self, PyObject *args)
{
    return blend_return_background(args);
}
