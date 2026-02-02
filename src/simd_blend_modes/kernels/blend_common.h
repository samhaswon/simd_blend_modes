#ifndef SIMD_BLEND_MODES_BLEND_COMMON_H
#define SIMD_BLEND_MODES_BLEND_COMMON_H

#include <Python.h>

static inline PyObject *blend_return_background(PyObject *args)
{
    PyObject *background = NULL;
    PyObject *foreground = NULL;
    PyObject *opacity = NULL;

    if (!PyArg_UnpackTuple(args, "blend", 3, 3, &background, &foreground, &opacity)) {
        return NULL;
    }

    Py_INCREF(background);
    return background;
}

#endif
