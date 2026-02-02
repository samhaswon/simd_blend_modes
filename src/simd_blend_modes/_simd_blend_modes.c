#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *blend_normal(PyObject *self, PyObject *args);
PyObject *blend_soft_light(PyObject *self, PyObject *args);
PyObject *blend_lighten_only(PyObject *self, PyObject *args);
PyObject *blend_screen(PyObject *self, PyObject *args);
PyObject *blend_dodge(PyObject *self, PyObject *args);
PyObject *blend_addition(PyObject *self, PyObject *args);
PyObject *blend_darken_only(PyObject *self, PyObject *args);
PyObject *blend_multiply(PyObject *self, PyObject *args);
PyObject *blend_hard_light(PyObject *self, PyObject *args);
PyObject *blend_difference(PyObject *self, PyObject *args);
PyObject *blend_subtract(PyObject *self, PyObject *args);
PyObject *blend_grain_extract(PyObject *self, PyObject *args);
PyObject *blend_grain_merge(PyObject *self, PyObject *args);
PyObject *blend_divide(PyObject *self, PyObject *args);
PyObject *blend_overlay(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"normal", blend_normal, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"soft_light", blend_soft_light, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"lighten_only", blend_lighten_only, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"screen", blend_screen, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"dodge", blend_dodge, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"addition", blend_addition, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"darken_only", blend_darken_only, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"multiply", blend_multiply, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"hard_light", blend_hard_light, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"difference", blend_difference, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"subtract", blend_subtract, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"grain_extract", blend_grain_extract, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"grain_merge", blend_grain_merge, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"divide", blend_divide, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {"overlay", blend_overlay, METH_VARARGS, "Blend (background, foreground, opacity)."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_simd_blend_modes",
    "SIMD blend mode kernels (skeleton).",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__simd_blend_modes(void)
{
    PyObject *module = PyModule_Create(&module_def);
    if (!module) {
        return NULL;
    }

    import_array();

    return module;
}
