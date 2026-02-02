#ifndef SIMD_BLEND_MODES_BLEND_COMMON_H
#define SIMD_BLEND_MODES_BLEND_COMMON_H

#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SIMD_BLEND_MODES_ARRAY_API
#include <numpy/arrayobject.h>

/* ----- Runtime CPU feature detection (GCC/Clang + MSVC) ----- */
#if defined(_MSC_VER)
  #include <intrin.h>
  static int os_supports_avx(void) {
      /* Check OSXSAVE + XCR0[2:1] == 11b so OS saves YMM state */
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      int osxsave = (ecx >> 27) & 1;
      if (!osxsave) return 0;
      unsigned long long xcr0 = _xgetbv(0);
      return ((xcr0 & 0x6) == 0x6); /* XMM (bit1) and YMM (bit2) state enabled */
  }

  static int cpu_supports_avx2(void) {
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      int avx   = (ecx >> 28) & 1;
      int osxsave = (ecx >> 27) & 1;
      if (!(avx && osxsave && os_supports_avx())) return 0;

      /* Leaf 7, subleaf 0: EBX bit 5 = AVX2 */
      int ex[4];
      __cpuidex(ex, 7, 0);
      int ebx = ex[1];
      return (ebx >> 5) & 1;
  }

  static int cpu_supports_sse42(void) {
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      return (ecx >> 20) & 1; /* SSE4.2 */
  }
#else
  /* GCC/Clang path */
  static int os_supports_avx(void) {
  #if defined(__GNUC__) || defined(__clang__)
      /* If we’re here, assume OS supports AVX when the CPU supports it.
         For full rigor you can also call xgetbv via inline asm, but it’s uncommon to lack it. */
      return 1;
  #else
      return 0;
  #endif
  }

  static int cpu_supports_avx2(void) {
  #if defined(__GNUC__) || defined(__clang__)
      /* Requires -mavx2 at compile, but we only *call* the AVX2 kernel if true. */
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

typedef enum {
    KERNEL_AUTO = 0,
    KERNEL_SCALAR = 1,
    KERNEL_SSE42 = 2,  /* 128-bit vectors */
    KERNEL_AVX2 = 3  /* 256-bit vectors */
} kernel_kind;

/* ---------- Kernel dispatch ---------- */

static kernel_kind pick_kernel(const char *force_name) {
    if (force_name) {
        if (strcmp(force_name, "scalar") == 0) return KERNEL_SCALAR;
        if (strcmp(force_name, "sse42")  == 0) return KERNEL_SSE42;
        if (strcmp(force_name, "avx2")   == 0) return KERNEL_AVX2;
        if (strcmp(force_name, "auto")   == 0) {/* fall through */}
    }
    /* Auto: prefer AVX2, then SSE4.2, else scalar */
    if (cpu_supports_avx2() && os_supports_avx()) return KERNEL_AVX2;
    if (cpu_supports_sse42()) return KERNEL_SSE42;
    return KERNEL_SCALAR;
}

typedef struct {
    PyArrayObject *array;
    int channels;
    int is_uint8;
    const uint8_t *u8;
    const float *f32;
} BlendArray;

typedef struct {
    PyArrayObject *array;
    int channels;
    int is_uint8;
    uint8_t *u8;
    float *f32;
} BlendOutput;

typedef float (*blend_comp_fn)(float in_c, float layer_c);

static inline float clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static inline float read_channel(const BlendArray *array, npy_intp index) {
    if (array->is_uint8) {
        return ((float)array->u8[index]) / 255.0f;
    }
    return array->f32[index] / 255.0f;
}

static inline void write_channel(BlendOutput *output, npy_intp index, float value) {
    float scaled = value * 255.0f;
    if (output->is_uint8) {
        if (scaled < 0.0f) {
            scaled = 0.0f;
        } else if (scaled > 255.0f) {
            scaled = 255.0f;
        }
        output->u8[index] = (uint8_t)lroundf(scaled);
        return;
    }
    output->f32[index] = scaled;
}

static int parse_blend_inputs(PyObject *args,
                              BlendArray *background,
                              BlendArray *foreground,
                              float *opacity,
                              npy_intp *height,
                              npy_intp *width) {
    PyObject *background_obj = NULL;
    PyObject *foreground_obj = NULL;
    PyObject *opacity_obj = NULL;

    if (!PyArg_ParseTuple(args, "OO|O", &background_obj, &foreground_obj, &opacity_obj)) {
        return 0;
    }

    background->array = (PyArrayObject *)PyArray_FromAny(
        background_obj, NULL, 3, 3, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!background->array) {
        return 0;
    }

    foreground->array = (PyArrayObject *)PyArray_FromAny(
        foreground_obj, NULL, 3, 3, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!foreground->array) {
        Py_DECREF(background->array);
        return 0;
    }

    int bg_type = PyArray_TYPE(background->array);
    int fg_type = PyArray_TYPE(foreground->array);
    if (!((bg_type == NPY_UINT8 || bg_type == NPY_FLOAT32) &&
          (fg_type == NPY_UINT8 || fg_type == NPY_FLOAT32))) {
        PyErr_SetString(PyExc_TypeError, "inputs must be uint8 or float32 numpy arrays");
        Py_DECREF(background->array);
        Py_DECREF(foreground->array);
        return 0;
    }

    background->channels = (int)PyArray_DIM(background->array, 2);
    foreground->channels = (int)PyArray_DIM(foreground->array, 2);
    if (!((background->channels == 3 || background->channels == 4) &&
          (foreground->channels == 3 || foreground->channels == 4))) {
        PyErr_SetString(PyExc_ValueError, "inputs must have 3 or 4 channels");
        Py_DECREF(background->array);
        Py_DECREF(foreground->array);
        return 0;
    }

    *height = PyArray_DIM(background->array, 0);
    *width = PyArray_DIM(background->array, 1);
    if (*height != PyArray_DIM(foreground->array, 0) ||
        *width != PyArray_DIM(foreground->array, 1)) {
        PyErr_SetString(PyExc_ValueError, "background and foreground must match in height and width");
        Py_DECREF(background->array);
        Py_DECREF(foreground->array);
        return 0;
    }

    if (!opacity_obj) {
        *opacity = 1.0f;
    } else {
        *opacity = (float)PyFloat_AsDouble(opacity_obj);
        if (PyErr_Occurred()) {
            Py_DECREF(background->array);
            Py_DECREF(foreground->array);
            return 0;
        }
    }

    background->is_uint8 = (bg_type == NPY_UINT8);
    foreground->is_uint8 = (fg_type == NPY_UINT8);
    background->u8 = background->is_uint8 ? (const uint8_t *)PyArray_DATA(background->array) : NULL;
    background->f32 = background->is_uint8 ? NULL : (const float *)PyArray_DATA(background->array);
    foreground->u8 = foreground->is_uint8 ? (const uint8_t *)PyArray_DATA(foreground->array) : NULL;
    foreground->f32 = foreground->is_uint8 ? NULL : (const float *)PyArray_DATA(foreground->array);

    return 1;
}

static inline void release_blend_inputs(BlendArray *background, BlendArray *foreground) {
    Py_DECREF(background->array);
    Py_DECREF(foreground->array);
}

static PyObject *blend_ratio_mode(PyObject *args, blend_comp_fn comp_fn, int clip_output) {
    BlendArray background = {0};
    BlendArray foreground = {0};
    float opacity = 0.0f;
    npy_intp height = 0;
    npy_intp width = 0;

    if (!parse_blend_inputs(args, &background, &foreground, &opacity, &height, &width)) {
        return NULL;
    }

    npy_intp dims[3] = {height, width, background.channels};
    int out_type = background.is_uint8 ? NPY_UINT8 : NPY_FLOAT32;
    PyArrayObject *output_array = (PyArrayObject *)PyArray_SimpleNew(3, dims, out_type);
    if (!output_array) {
        release_blend_inputs(&background, &foreground);
        return NULL;
    }

    BlendOutput output = {0};
    output.array = output_array;
    output.channels = background.channels;
    output.is_uint8 = background.is_uint8;
    output.u8 = output.is_uint8 ? (uint8_t *)PyArray_DATA(output_array) : NULL;
    output.f32 = output.is_uint8 ? NULL : (float *)PyArray_DATA(output_array);

    npy_intp pixels = height * width;

    for (npy_intp index = 0; index < pixels; ++index) {
        npy_intp bg_offset = index * background.channels;
        npy_intp fg_offset = index * foreground.channels;

        if (opacity <= 0.0f) {
            for (int c = 0; c < background.channels; ++c) {
                float value = read_channel(&background, bg_offset + c);
                write_channel(&output, bg_offset + c, value);
            }
            continue;
        }

        float in_a = 1.0f;
        float layer_a = 1.0f;

        if (background.channels == 4) {
            in_a = read_channel(&background, bg_offset + 3);
        }
        if (foreground.channels == 4) {
            layer_a = read_channel(&foreground, fg_offset + 3);
        }

        float comp_alpha = fminf(in_a, layer_a) * opacity;

        float new_alpha = in_a + (1.0f - in_a) * comp_alpha;
        float ratio = 0.0f;
        if (new_alpha > 0.0f) {
            ratio = comp_alpha / new_alpha;
        }

        for (int c = 0; c < 3; ++c) {
            float in_c = read_channel(&background, bg_offset + c);
            float layer_c = read_channel(&foreground, fg_offset + c);
            float comp = comp_fn(in_c, layer_c);
            float out_c = comp * ratio + in_c * (1.0f - ratio);
            if (clip_output) {
                out_c = clamp01(out_c);
            }
            write_channel(&output, bg_offset + c, out_c);
        }

        if (background.channels == 4) {
            write_channel(&output, bg_offset + 3, in_a);
        }
    }

    release_blend_inputs(&background, &foreground);
    return (PyObject *)output_array;
}

static PyObject *blend_normal_mode(PyObject *args) {
    BlendArray background = {0};
    BlendArray foreground = {0};
    float opacity = 0.0f;
    npy_intp height = 0;
    npy_intp width = 0;

    if (!parse_blend_inputs(args, &background, &foreground, &opacity, &height, &width)) {
        return NULL;
    }

    npy_intp dims[3] = {height, width, background.channels};
    int out_type = background.is_uint8 ? NPY_UINT8 : NPY_FLOAT32;
    PyArrayObject *output_array = (PyArrayObject *)PyArray_SimpleNew(3, dims, out_type);
    if (!output_array) {
        release_blend_inputs(&background, &foreground);
        return NULL;
    }

    BlendOutput output = {0};
    output.array = output_array;
    output.channels = background.channels;
    output.is_uint8 = background.is_uint8;
    output.u8 = output.is_uint8 ? (uint8_t *)PyArray_DATA(output_array) : NULL;
    output.f32 = output.is_uint8 ? NULL : (float *)PyArray_DATA(output_array);

    npy_intp pixels = height * width;

    for (npy_intp index = 0; index < pixels; ++index) {
        npy_intp bg_offset = index * background.channels;
        npy_intp fg_offset = index * foreground.channels;

        if (opacity <= 0.0f) {
            for (int c = 0; c < background.channels; ++c) {
                float value = read_channel(&background, bg_offset + c);
                write_channel(&output, bg_offset + c, value);
            }
            continue;
        }

        float in_a = 1.0f;
        float layer_a = 1.0f;

        if (background.channels == 4) {
            in_a = read_channel(&background, bg_offset + 3);
        }
        if (foreground.channels == 4) {
            layer_a = read_channel(&foreground, fg_offset + 3);
        }

        float layer_opacity = layer_a * opacity;

        float denom = layer_opacity + in_a * (1.0f - layer_opacity);
        for (int c = 0; c < 3; ++c) {
            float in_c = read_channel(&background, bg_offset + c);
            float layer_c = read_channel(&foreground, fg_offset + c);
            float out_c = 0.0f;
            if (denom > 0.0f) {
                out_c = (layer_c * layer_opacity + in_c * in_a * (1.0f - layer_opacity)) / denom;
            }
            write_channel(&output, bg_offset + c, out_c);
        }

        if (background.channels == 4) {
            float out_a = layer_opacity + in_a * (1.0f - layer_opacity);
            write_channel(&output, bg_offset + 3, out_a);
        }
    }

    release_blend_inputs(&background, &foreground);
    return (PyObject *)output_array;
}

#endif
