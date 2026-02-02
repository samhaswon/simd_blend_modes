#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#define PY_ARRAY_UNIQUE_SYMBOL SIMD_BLEND_MODES_ARRAY_API
#include <numpy/arrayobject.h>

/* ----- Runtime CPU feature detection (GCC/Clang + MSVC) ----- */
#if defined(_MSC_VER)
  #include <intrin.h>
  static int os_supports_avx(void) {
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      int osxsave = (ecx >> 27) & 1;
      if (!osxsave) return 0;
      unsigned long long xcr0 = _xgetbv(0);
      return ((xcr0 & 0x6) == 0x6);
  }

  static int cpu_supports_avx2(void) {
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      int avx = (ecx >> 28) & 1;
      int osxsave = (ecx >> 27) & 1;
      if (!(avx && osxsave && os_supports_avx())) return 0;

      int ex[4];
      __cpuidex(ex, 7, 0);
      int ebx = ex[1];
      return (ebx >> 5) & 1;
  }

  static int cpu_supports_sse42(void) {
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      return (ecx >> 20) & 1;
  }
#else
  static int os_supports_avx(void) {
  #if defined(__GNUC__) || defined(__clang__)
      return 1;
  #else
      return 0;
  #endif
  }

  static int cpu_supports_avx2(void) {
  #if defined(__GNUC__) || defined(__clang__)
      return __builtin_cpu_supports("avx2");
  #else
      return 0;
  #endif
  }

  static int cpu_supports_sse42(void) {
  #if defined(__GNUC__) || defined(__clang__)
      return __builtin_cpu_supports("sse4.2");
  #else
      return 0;
  #endif
  }
#endif

static PyObject *kernel_available(PyObject *self, PyObject *args)
{
    const char *name = NULL;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        return NULL;
    }

    if (strcmp(name, "scalar") == 0 || strcmp(name, "auto") == 0) {
        Py_RETURN_TRUE;
    }
    if (strcmp(name, "sse42") == 0) {
        if (cpu_supports_sse42()) {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    }
    if (strcmp(name, "avx2") == 0) {
        if (cpu_supports_avx2() && os_supports_avx()) {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    }

    PyErr_SetString(PyExc_ValueError, "unknown kernel kind");
    return NULL;
}

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
    {"kernel_available", kernel_available, METH_VARARGS, "Check whether a kernel is supported."},
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

    if (_import_array() < 0) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
