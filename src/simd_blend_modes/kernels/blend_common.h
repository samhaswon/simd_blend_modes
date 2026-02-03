#ifndef SIMD_BLEND_MODES_BLEND_COMMON_H
#define SIMD_BLEND_MODES_BLEND_COMMON_H

#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define SIMD_BLEND_MODES_X86 1
#else
  #define SIMD_BLEND_MODES_X86 0
#endif

#if SIMD_BLEND_MODES_X86
  #include <smmintrin.h>
  #include <tmmintrin.h>
  #include <immintrin.h>
#endif

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SIMD_BLEND_MODES_ARRAY_API
#include <numpy/arrayobject.h>

/* ----- Runtime CPU feature detection (GCC/Clang + MSVC) ----- */
#if SIMD_BLEND_MODES_X86
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
#else
  static int os_supports_avx(void) { return 0; }
  static int cpu_supports_avx2(void) { return 0; }
  static int cpu_supports_sse42(void) { return 0; }
#endif

typedef enum {
    KERNEL_AUTO = 0,
    KERNEL_SCALAR = 1,
    KERNEL_SSE42 = 2,  /* 128-bit vectors */
    KERNEL_AVX2 = 3  /* 256-bit vectors */
} kernel_kind;

/* ---------- Kernel dispatch ---------- */

static kernel_kind pick_kernel(const char *force_name) {
#if !SIMD_BLEND_MODES_X86
    (void)force_name;
    return KERNEL_SCALAR;
#endif
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

#if SIMD_BLEND_MODES_X86
typedef __m128 (*blend_comp_fn128)(__m128 in_c, __m128 layer_c);
typedef __m256 (*blend_comp_fn256)(__m256 in_c, __m256 layer_c);
#else
typedef void *blend_comp_fn128;
typedef void *blend_comp_fn256;
#endif

#if SIMD_BLEND_MODES_X86
#define SIMD_BLEND_MODES_SIMD_ARGS(comp_sse, comp_avx) (comp_sse), (comp_avx)
#else
#define SIMD_BLEND_MODES_SIMD_ARGS(comp_sse, comp_avx) NULL, NULL
#endif

#if SIMD_BLEND_MODES_X86
static inline __m256 mul_add_ps256(__m256 a, __m256 b, __m256 c) {
#ifdef __FMA__
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

static inline __m256 fnmadd_ps256(__m256 a, __m256 b, __m256 c) {
#ifdef __FMA__
    return _mm256_fnmadd_ps(a, b, c);
#else
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
#endif
}

/* Convert 8 consecutive u8 to float32 in [0,1] (for grayscale im_alpha). */
static inline __m256 load8_u8_to_unit_f32_avx2(const uint8_t *p, __m256 inv255) {
    __m128i v8  = _mm_loadl_epi64((const __m128i*)p);        /* 8 bytes -> XMM */
    __m256i v32 = _mm256_cvtepu8_epi32(v8);                  /* widen to 8 x u32 */
    return _mm256_mul_ps(_mm256_cvtepi32_ps(v32), inv255);
}

static inline void load16_u8_to_unit_f32_avx2(const uint8_t *p, __m256 inv255,
                                              __m256 *lo, __m256 *hi) {
    __m128i v16 = _mm_loadu_si128((const __m128i*)p);        /* 16 bytes */
    __m256i v32_lo = _mm256_cvtepu8_epi32(v16);
    __m128i v8_hi = _mm_srli_si128(v16, 8);
    __m256i v32_hi = _mm256_cvtepu8_epi32(v8_hi);
    *lo = _mm256_mul_ps(_mm256_cvtepi32_ps(v32_lo), inv255);
    *hi = _mm256_mul_ps(_mm256_cvtepi32_ps(v32_hi), inv255);
}

static inline void load16_u8_to_unit_f32_avx2_from_xmm(__m128i v16, __m256 inv255,
                                                       __m256 *lo, __m256 *hi) {
    __m256i v32_lo = _mm256_cvtepu8_epi32(v16);
    __m128i v8_hi = _mm_srli_si128(v16, 8);
    __m256i v32_hi = _mm256_cvtepu8_epi32(v8_hi);
    *lo = _mm256_mul_ps(_mm256_cvtepi32_ps(v32_lo), inv255);
    *hi = _mm256_mul_ps(_mm256_cvtepi32_ps(v32_hi), inv255);
}

static inline __m256 clamp01_ps(__m256 x) {
    return _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
}

static inline void load_rgba8_u8_to_unit_f32_avx2(const uint8_t *p, __m256 inv255,
                                                  __m256 *r, __m256 *g, __m256 *b,
                                                  __m256 *a) {
    __m256i pixels = _mm256_loadu_si256((const __m256i *)p);
    __m128i lo = _mm256_castsi256_si128(pixels);
    __m128i hi = _mm256_extracti128_si256(pixels, 1);

    const __m128i mask_r = _mm_setr_epi8(
        0, 4, 8, 12, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80);
    const __m128i mask_g = _mm_setr_epi8(
        1, 5, 9, 13, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80);
    const __m128i mask_b = _mm_setr_epi8(
        2, 6, 10, 14, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80);
    const __m128i mask_a = _mm_setr_epi8(
        3, 7, 11, 15, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80);

    __m128 r0 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, mask_r)));
    __m128 r1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, mask_r)));
    __m128 g0 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, mask_g)));
    __m128 g1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, mask_g)));
    __m128 b0 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, mask_b)));
    __m128 b1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, mask_b)));
    __m128 a0 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(lo, mask_a)));
    __m128 a1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(hi, mask_a)));

    __m256 r256 = _mm256_insertf128_ps(_mm256_castps128_ps256(r0), r1, 1);
    __m256 g256 = _mm256_insertf128_ps(_mm256_castps128_ps256(g0), g1, 1);
    __m256 b256 = _mm256_insertf128_ps(_mm256_castps128_ps256(b0), b1, 1);
    __m256 a256 = _mm256_insertf128_ps(_mm256_castps128_ps256(a0), a1, 1);

    *r = _mm256_mul_ps(r256, inv255);
    *g = _mm256_mul_ps(g256, inv255);
    *b = _mm256_mul_ps(b256, inv255);
    *a = _mm256_mul_ps(a256, inv255);
}

/* Replace NaN with 0.0f (Inf is not expected from uint8-origin math). */
static inline __m256 nan_to_num_ps(__m256 x) {
    __m256 cmp = _mm256_cmp_ps(x, x, _CMP_ORD_Q); /* 0 for NaN lanes */
    return _mm256_blendv_ps(_mm256_set1_ps(0.0f), x, cmp);
}

/* Convert 4 float32 RGB vectors in [0,1] to uint8_t RGBRGBRGBRGB without branches. */
static inline __m128i pack_unit_f32_to_u8_rgb4(__m128 fr, __m128 fg, __m128 fb) {
    const __m128 scale = _mm_set1_ps(255.0f);
    const __m128i zero = _mm_setzero_si128();
    const __m128i max255 = _mm_set1_epi32(255);

    __m128i ir = _mm_cvttps_epi32(_mm_mul_ps(fr, scale));
    __m128i ig = _mm_cvttps_epi32(_mm_mul_ps(fg, scale));
    __m128i ib = _mm_cvttps_epi32(_mm_mul_ps(fb, scale));

    ir = _mm_min_epi32(_mm_max_epi32(ir, zero), max255);
    ig = _mm_min_epi32(_mm_max_epi32(ig, zero), max255);
    ib = _mm_min_epi32(_mm_max_epi32(ib, zero), max255);

    __m128i ir16 = _mm_packus_epi32(ir, zero);
    __m128i ig16 = _mm_packus_epi32(ig, zero);
    __m128i ib16 = _mm_packus_epi32(ib, zero);

    __m128i ir8 = _mm_packus_epi16(ir16, zero);
    __m128i ig8 = _mm_packus_epi16(ig16, zero);
    __m128i ib8 = _mm_packus_epi16(ib16, zero);

    const __m128i mask_r = _mm_setr_epi8(
        0, (char)0x80, (char)0x80, 1,
        (char)0x80, (char)0x80, 2, (char)0x80,
        (char)0x80, 3, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    const __m128i mask_g = _mm_setr_epi8(
        (char)0x80, 0, (char)0x80, (char)0x80,
        1, (char)0x80, (char)0x80, 2,
        (char)0x80, (char)0x80, 3, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    const __m128i mask_b = _mm_setr_epi8(
        (char)0x80, (char)0x80, 0, (char)0x80,
        (char)0x80, 1, (char)0x80, (char)0x80,
        2, (char)0x80, (char)0x80, 3,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);

    __m128i packed = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(ir8, mask_r),
                     _mm_shuffle_epi8(ig8, mask_g)),
        _mm_shuffle_epi8(ib8, mask_b));

    return packed;
}

static inline __m128i pack_unit_f32_to_u8_rgba4(__m128 fr, __m128 fg, __m128 fb, __m128 fa) {
    const __m128 scale = _mm_set1_ps(255.0f);
    const __m128i zero = _mm_setzero_si128();
    const __m128i max255 = _mm_set1_epi32(255);

    __m128i ir = _mm_cvtps_epi32(_mm_mul_ps(fr, scale));
    __m128i ig = _mm_cvtps_epi32(_mm_mul_ps(fg, scale));
    __m128i ib = _mm_cvtps_epi32(_mm_mul_ps(fb, scale));
    __m128i ia = _mm_cvtps_epi32(_mm_mul_ps(fa, scale));

    ir = _mm_min_epi32(_mm_max_epi32(ir, zero), max255);
    ig = _mm_min_epi32(_mm_max_epi32(ig, zero), max255);
    ib = _mm_min_epi32(_mm_max_epi32(ib, zero), max255);
    ia = _mm_min_epi32(_mm_max_epi32(ia, zero), max255);

    __m128i ir16 = _mm_packus_epi32(ir, zero);
    __m128i ig16 = _mm_packus_epi32(ig, zero);
    __m128i ib16 = _mm_packus_epi32(ib, zero);
    __m128i ia16 = _mm_packus_epi32(ia, zero);

    __m128i ir8 = _mm_packus_epi16(ir16, zero);
    __m128i ig8 = _mm_packus_epi16(ig16, zero);
    __m128i ib8 = _mm_packus_epi16(ib16, zero);
    __m128i ia8 = _mm_packus_epi16(ia16, zero);

    __m128i rg = _mm_unpacklo_epi8(ir8, ig8);
    __m128i ba = _mm_unpacklo_epi8(ib8, ia8);
    __m128i rgba_lo = _mm_unpacklo_epi16(rg, ba);
    return rgba_lo;
}

static inline void store_unit_f32_to_u8_rgb4(__m128 fr, __m128 fg, __m128 fb,
                                             uint8_t *out_ptr) {
    __m128i packed = pack_unit_f32_to_u8_rgb4(fr, fg, fb);
    _mm_storel_epi64((__m128i*)out_ptr, packed);
    __m128i tail_vec = _mm_srli_si128(packed, 8);
    uint32_t tail = (uint32_t)_mm_cvtsi128_si32(tail_vec);
    memcpy(out_ptr + 8, &tail, sizeof(tail));
}

static inline void store_unit_f32_to_u8_rgb4_u16(__m128 fr, __m128 fg, __m128 fb,
                                                 uint8_t *out_ptr) {
    __m128i packed = pack_unit_f32_to_u8_rgb4(fr, fg, fb);
    _mm_storeu_si128((__m128i*)out_ptr, packed);
}

static inline void store_unit_f32_to_u8_rgba4(__m128 fr, __m128 fg, __m128 fb, __m128 fa,
                                              uint8_t *out_ptr) {
    __m128i packed = pack_unit_f32_to_u8_rgba4(fr, fg, fb, fa);
    _mm_storeu_si128((__m128i*)out_ptr, packed);
}

static inline void store_rgba4_u8(BlendOutput *output, npy_intp index,
                                  __m128 r, __m128 g, __m128 b, __m128 a) {
    uint8_t *out_ptr = output->u8 + (index * output->channels);
    store_unit_f32_to_u8_rgba4(r, g, b, a, out_ptr);
}

static inline void store_rgba8_u8(BlendOutput *output, npy_intp index,
                                  __m256 r, __m256 g, __m256 b, __m256 a) {
    uint8_t *out_ptr = output->u8 + (index * output->channels);
    __m128 r0 = _mm256_castps256_ps128(r);
    __m128 r1 = _mm256_extractf128_ps(r, 1);
    __m128 g0 = _mm256_castps256_ps128(g);
    __m128 g1 = _mm256_extractf128_ps(g, 1);
    __m128 b0 = _mm256_castps256_ps128(b);
    __m128 b1 = _mm256_extractf128_ps(b, 1);
    __m128 a0 = _mm256_castps256_ps128(a);
    __m128 a1 = _mm256_extractf128_ps(a, 1);

    store_unit_f32_to_u8_rgba4(r0, g0, b0, a0, out_ptr);
    store_unit_f32_to_u8_rgba4(r1, g1, b1, a1, out_ptr + 16);
}

/* ---------- SSE4.2 skeleton (process 4 pixels via manual loads) ---------- */

static inline __m128 load4_u8_to_unit_f32(const uint8_t *p) {
    /* p[0..3] are consecutive bytes (for im_alpha) */
    __m128i v8  = _mm_cvtsi32_si128(*(const int*)p);  /* 4 bytes into xmm */
    __m128i v16 = _mm_cvtepu8_epi16(v8);              /* widen to 8 x u16, we use low 4 */
    __m128i v32 = _mm_cvtepu16_epi32(v16);
    return _mm_mul_ps(_mm_cvtepi32_ps(v32), _mm_set1_ps(1.0f/255.0f));
}

static inline __m128 clamp01_ps128(__m128 x) {
    return _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f));
}

static inline void load_rgba4_u8_to_unit_f32_sse(const uint8_t *p, __m128 inv255,
                                                 __m128 *r, __m128 *g, __m128 *b,
                                                 __m128 *a) {
    __m128i pixels = _mm_loadu_si128((const __m128i *)p);
    const __m128i mask_r = _mm_setr_epi8(
        0, 4, 8, 12, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80);
    const __m128i mask_g = _mm_setr_epi8(
        1, 5, 9, 13, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80);
    const __m128i mask_b = _mm_setr_epi8(
        2, 6, 10, 14, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80);
    const __m128i mask_a = _mm_setr_epi8(
        3, 7, 11, 15, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80);

    __m128i r8 = _mm_shuffle_epi8(pixels, mask_r);
    __m128i g8 = _mm_shuffle_epi8(pixels, mask_g);
    __m128i b8 = _mm_shuffle_epi8(pixels, mask_b);
    __m128i a8 = _mm_shuffle_epi8(pixels, mask_a);

    *r = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(r8)), inv255);
    *g = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(g8)), inv255);
    *b = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(b8)), inv255);
    *a = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(a8)), inv255);
}

static inline __m128 nan_to_num_ps128(__m128 x) {
    __m128 cmp = _mm_cmpord_ps(x, x); /* 0 for NaN lanes */
    return _mm_blendv_ps(_mm_set1_ps(0.0f), x, cmp);
}

static inline __m128 mul_add_ps128(__m128 a, __m128 b, __m128 c) {
#ifdef __FMA__
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

static inline __m128 fnmadd_ps128(__m128 a, __m128 b, __m128 c) {
#ifdef __FMA__
    return _mm_fnmadd_ps(a, b, c);
#else
    return _mm_sub_ps(c, _mm_mul_ps(a, b));
#endif
}
#endif

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

#if SIMD_BLEND_MODES_X86
static inline void load_rgb4(const BlendArray *array, npy_intp index,
                             __m128 *r, __m128 *g, __m128 *b) {
    npy_intp base0 = (index + 0) * array->channels;
    npy_intp base1 = (index + 1) * array->channels;
    npy_intp base2 = (index + 2) * array->channels;
    npy_intp base3 = (index + 3) * array->channels;

    float r0 = read_channel(array, base0 + 0);
    float g0 = read_channel(array, base0 + 1);
    float b0 = read_channel(array, base0 + 2);
    float r1 = read_channel(array, base1 + 0);
    float g1 = read_channel(array, base1 + 1);
    float b1 = read_channel(array, base1 + 2);
    float r2 = read_channel(array, base2 + 0);
    float g2 = read_channel(array, base2 + 1);
    float b2 = read_channel(array, base2 + 2);
    float r3 = read_channel(array, base3 + 0);
    float g3 = read_channel(array, base3 + 1);
    float b3 = read_channel(array, base3 + 2);

    *r = _mm_set_ps(r3, r2, r1, r0);
    *g = _mm_set_ps(g3, g2, g1, g0);
    *b = _mm_set_ps(b3, b2, b1, b0);
}

static inline __m128 load_alpha4(const BlendArray *array, npy_intp index) {
    if (array->channels == 4) {
        npy_intp base0 = (index + 0) * array->channels + 3;
        npy_intp base1 = (index + 1) * array->channels + 3;
        npy_intp base2 = (index + 2) * array->channels + 3;
        npy_intp base3 = (index + 3) * array->channels + 3;
        float a0 = read_channel(array, base0);
        float a1 = read_channel(array, base1);
        float a2 = read_channel(array, base2);
        float a3 = read_channel(array, base3);
        return _mm_set_ps(a3, a2, a1, a0);
    }
    return _mm_set1_ps(1.0f);
}

static inline void store_rgb4(BlendOutput *output, npy_intp index,
                              __m128 r, __m128 g, __m128 b) {
    float rv[4];
    float gv[4];
    float bv[4];
    _mm_storeu_ps(rv, r);
    _mm_storeu_ps(gv, g);
    _mm_storeu_ps(bv, b);

    for (int i = 0; i < 4; ++i) {
        npy_intp base = (index + i) * output->channels;
        write_channel(output, base + 0, rv[i]);
        write_channel(output, base + 1, gv[i]);
        write_channel(output, base + 2, bv[i]);
    }
}

static inline void store_alpha4(BlendOutput *output, npy_intp index, __m128 a) {
    if (output->channels != 4) {
        return;
    }
    float av[4];
    _mm_storeu_ps(av, a);
    for (int i = 0; i < 4; ++i) {
        npy_intp base = (index + i) * output->channels + 3;
        write_channel(output, base, av[i]);
    }
}

static inline void load_rgb8(const BlendArray *array, npy_intp index,
                             __m256 *r, __m256 *g, __m256 *b) {
    float rv[8];
    float gv[8];
    float bv[8];
    for (int i = 0; i < 8; ++i) {
        npy_intp base = (index + i) * array->channels;
        rv[i] = read_channel(array, base + 0);
        gv[i] = read_channel(array, base + 1);
        bv[i] = read_channel(array, base + 2);
    }
    *r = _mm256_set_ps(rv[7], rv[6], rv[5], rv[4], rv[3], rv[2], rv[1], rv[0]);
    *g = _mm256_set_ps(gv[7], gv[6], gv[5], gv[4], gv[3], gv[2], gv[1], gv[0]);
    *b = _mm256_set_ps(bv[7], bv[6], bv[5], bv[4], bv[3], bv[2], bv[1], bv[0]);
}

static inline __m256 load_alpha8(const BlendArray *array, npy_intp index) {
    if (array->channels == 4) {
        float av[8];
        for (int i = 0; i < 8; ++i) {
            npy_intp base = (index + i) * array->channels + 3;
            av[i] = read_channel(array, base);
        }
        return _mm256_set_ps(av[7], av[6], av[5], av[4], av[3], av[2], av[1], av[0]);
    }
    return _mm256_set1_ps(1.0f);
}

static inline void store_rgb8(BlendOutput *output, npy_intp index,
                              __m256 r, __m256 g, __m256 b) {
    float rv[8];
    float gv[8];
    float bv[8];
    _mm256_storeu_ps(rv, r);
    _mm256_storeu_ps(gv, g);
    _mm256_storeu_ps(bv, b);

    for (int i = 0; i < 8; ++i) {
        npy_intp base = (index + i) * output->channels;
        write_channel(output, base + 0, rv[i]);
        write_channel(output, base + 1, gv[i]);
        write_channel(output, base + 2, bv[i]);
    }
}

static inline void store_alpha8(BlendOutput *output, npy_intp index, __m256 a) {
    if (output->channels != 4) {
        return;
    }
    float av[8];
    _mm256_storeu_ps(av, a);
    for (int i = 0; i < 8; ++i) {
        npy_intp base = (index + i) * output->channels + 3;
        write_channel(output, base, av[i]);
    }
}
#endif

static inline void release_blend_inputs(BlendArray *background, BlendArray *foreground) {
    Py_DECREF(background->array);
    Py_DECREF(foreground->array);
}

static inline int parse_blend_inputs(PyObject *args,
                              BlendArray *background,
                              BlendArray *foreground,
                              float *opacity,
                              npy_intp *height,
                              npy_intp *width,
                              const char **kernel_name,
                              PyObject **kernel_hold) {
    PyObject *background_obj = NULL;
    PyObject *foreground_obj = NULL;
    PyObject *opacity_obj = NULL;
    PyObject *kernel_obj = NULL;
    PyObject *kernel_value_obj = NULL;
    const char *kernel_value = NULL;

    if (!PyArg_ParseTuple(args, "OO|OO", &background_obj, &foreground_obj, &opacity_obj,
                          &kernel_obj)) {
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
    } else if (PyUnicode_Check(opacity_obj) && !kernel_obj) {
        kernel_obj = opacity_obj;
        *opacity = 1.0f;
    } else {
        *opacity = (float)PyFloat_AsDouble(opacity_obj);
        if (PyErr_Occurred()) {
            Py_DECREF(background->array);
            Py_DECREF(foreground->array);
            return 0;
        }
    }

    if (kernel_obj) {
        if (PyUnicode_Check(kernel_obj)) {
            kernel_value = PyUnicode_AsUTF8(kernel_obj);
        } else {
            kernel_value_obj = PyObject_GetAttrString(kernel_obj, "value");
            if (kernel_value_obj && PyUnicode_Check(kernel_value_obj)) {
                kernel_value = PyUnicode_AsUTF8(kernel_value_obj);
            }
        }

        if (!kernel_value) {
            Py_XDECREF(kernel_value_obj);
            PyErr_SetString(PyExc_TypeError, "kernel must be a string or KernelKind");
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

    if (kernel_name) {
        *kernel_name = kernel_value;
    }
    if (kernel_hold) {
        *kernel_hold = kernel_value_obj;
    } else {
        Py_XDECREF(kernel_value_obj);
    }

    return 1;
}

static inline int alloc_output(const BlendArray *background,
                               npy_intp height,
                               npy_intp width,
                               BlendOutput *output) {
    npy_intp dims[3] = {height, width, background->channels};
    int out_type = background->is_uint8 ? NPY_UINT8 : NPY_FLOAT32;
    PyArrayObject *output_array = (PyArrayObject *)PyArray_SimpleNew(3, dims, out_type);
    if (!output_array) {
        return 0;
    }

    output->array = output_array;
    output->channels = background->channels;
    output->is_uint8 = background->is_uint8;
    output->u8 = output->is_uint8 ? (uint8_t *)PyArray_DATA(output_array) : NULL;
    output->f32 = output->is_uint8 ? NULL : (float *)PyArray_DATA(output_array);
    return 1;
}

static inline int copy_background(const BlendArray *background, BlendOutput *output,
                                  npy_intp height, npy_intp width) {
    if (!alloc_output(background, height, width, output)) {
        return 0;
    }
    size_t bytes = (size_t)height * (size_t)width * (size_t)background->channels *
                   (background->is_uint8 ? sizeof(uint8_t) : sizeof(float));
    memcpy(background->is_uint8 ? (void *)output->u8 : (void *)output->f32,
           background->is_uint8 ? (const void *)background->u8 : (const void *)background->f32,
           bytes);
    return 1;
}

static inline PyObject *blend_ratio_mode_simd(PyObject *args,
                                       blend_comp_fn comp_scalar,
                                       blend_comp_fn128 comp_sse,
                                       blend_comp_fn256 comp_avx,
                                       int clip_output) {
    BlendArray background = {0};
    BlendArray foreground = {0};
    float opacity = 0.0f;
    npy_intp height = 0;
    npy_intp width = 0;
    const char *kernel_name = NULL;
    PyObject *kernel_hold = NULL;

    if (!parse_blend_inputs(args, &background, &foreground, &opacity, &height, &width,
                            &kernel_name, &kernel_hold)) {
        return NULL;
    }

    BlendOutput output = {0};
    if (opacity <= 0.0f) {
        if (!copy_background(&background, &output, height, width)) {
            release_blend_inputs(&background, &foreground);
            return NULL;
        }
        release_blend_inputs(&background, &foreground);
        return (PyObject *)output.array;
    }

    if (!alloc_output(&background, height, width, &output)) {
        Py_XDECREF(kernel_hold);
        release_blend_inputs(&background, &foreground);
        return NULL;
    }

    kernel_kind kernel = pick_kernel(kernel_name);

    npy_intp pixels = height * width;
    npy_intp index = 0;
#if SIMD_BLEND_MODES_X86
    const __m128 one = _mm_set1_ps(1.0f);
    const __m256 one256 = _mm256_set1_ps(1.0f);
    const __m128 opacity128 = _mm_set1_ps(opacity);
    const __m256 opacity256 = _mm256_set1_ps(opacity);
    const __m128 inv255128 = _mm_set1_ps(1.0f / 255.0f);
    const __m256 inv255256 = _mm256_set1_ps(1.0f / 255.0f);

#if defined(__AVX2__)
    if (kernel == KERNEL_AVX2) {
        const npy_intp prefetch_distance = 16;
        for (; index + 7 < pixels; index += 8) {
            npy_intp prefetch_index = index + prefetch_distance;
            if (background.is_uint8) {
                _mm_prefetch((const char *)(background.u8 + (prefetch_index * background.channels)),
                             _MM_HINT_T0);
            } else {
                _mm_prefetch((const char *)(background.f32 + (prefetch_index * background.channels)),
                             _MM_HINT_T0);
            }
            if (foreground.is_uint8) {
                _mm_prefetch((const char *)(foreground.u8 + (prefetch_index * foreground.channels)),
                             _MM_HINT_T0);
            } else {
                _mm_prefetch((const char *)(foreground.f32 + (prefetch_index * foreground.channels)),
                             _MM_HINT_T0);
            }
            __m256 in_r, in_g, in_b, in_a;
            __m256 layer_r, layer_g, layer_b, layer_a;
            if (background.is_uint8 && background.channels == 4) {
                load_rgba8_u8_to_unit_f32_avx2(
                    background.u8 + (index * background.channels),
                    inv255256,
                    &in_r,
                    &in_g,
                    &in_b,
                    &in_a
                );
            } else {
                load_rgb8(&background, index, &in_r, &in_g, &in_b);
                in_a = load_alpha8(&background, index);
            }

            if (foreground.is_uint8 && foreground.channels == 4) {
                load_rgba8_u8_to_unit_f32_avx2(
                    foreground.u8 + (index * foreground.channels),
                    inv255256,
                    &layer_r,
                    &layer_g,
                    &layer_b,
                    &layer_a
                );
            } else {
                load_rgb8(&foreground, index, &layer_r, &layer_g, &layer_b);
                layer_a = load_alpha8(&foreground, index);
            }

            __m256 comp_alpha = _mm256_mul_ps(_mm256_min_ps(in_a, layer_a), opacity256);
            __m256 new_alpha = mul_add_ps256(_mm256_sub_ps(one256, in_a), comp_alpha, in_a);
            __m256 ratio = _mm256_div_ps(comp_alpha, new_alpha);
            __m256 mask = _mm256_cmp_ps(new_alpha, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
            ratio = _mm256_blendv_ps(_mm256_set1_ps(0.0f), ratio, mask);

            __m256 comp_r = comp_avx(in_r, layer_r);
            __m256 comp_g = comp_avx(in_g, layer_g);
            __m256 comp_b = comp_avx(in_b, layer_b);

            __m256 out_r = mul_add_ps256(comp_r, ratio,
                                         _mm256_mul_ps(in_r, _mm256_sub_ps(one256, ratio)));
            __m256 out_g = mul_add_ps256(comp_g, ratio,
                                         _mm256_mul_ps(in_g, _mm256_sub_ps(one256, ratio)));
            __m256 out_b = mul_add_ps256(comp_b, ratio,
                                         _mm256_mul_ps(in_b, _mm256_sub_ps(one256, ratio)));

            if (clip_output) {
                out_r = _mm256_min_ps(_mm256_max_ps(out_r, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
                out_g = _mm256_min_ps(_mm256_max_ps(out_g, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
                out_b = _mm256_min_ps(_mm256_max_ps(out_b, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
            }

            if (output.is_uint8 && output.channels == 4) {
                store_rgba8_u8(&output, index, out_r, out_g, out_b, in_a);
            } else {
                store_rgb8(&output, index, out_r, out_g, out_b);
                store_alpha8(&output, index, in_a);
            }
        }
    }
#endif

#if defined(__SSE4_1__)
    if (kernel == KERNEL_SSE42 || kernel == KERNEL_AVX2) {
        for (; index + 3 < pixels; index += 4) {
            __m128 in_r, in_g, in_b, in_a;
            __m128 layer_r, layer_g, layer_b, layer_a;
            if (background.is_uint8 && background.channels == 4) {
                load_rgba4_u8_to_unit_f32_sse(
                    background.u8 + (index * background.channels),
                    inv255128,
                    &in_r,
                    &in_g,
                    &in_b,
                    &in_a
                );
            } else {
                load_rgb4(&background, index, &in_r, &in_g, &in_b);
                in_a = load_alpha4(&background, index);
            }

            if (foreground.is_uint8 && foreground.channels == 4) {
                load_rgba4_u8_to_unit_f32_sse(
                    foreground.u8 + (index * foreground.channels),
                    inv255128,
                    &layer_r,
                    &layer_g,
                    &layer_b,
                    &layer_a
                );
            } else {
                load_rgb4(&foreground, index, &layer_r, &layer_g, &layer_b);
                layer_a = load_alpha4(&foreground, index);
            }

            __m128 comp_alpha = _mm_mul_ps(_mm_min_ps(in_a, layer_a), opacity128);
            __m128 new_alpha = mul_add_ps128(_mm_sub_ps(one, in_a), comp_alpha, in_a);
            __m128 ratio = _mm_div_ps(comp_alpha, new_alpha);
            __m128 mask = _mm_cmpgt_ps(new_alpha, _mm_set1_ps(0.0f));
            ratio = _mm_blendv_ps(_mm_set1_ps(0.0f), ratio, mask);

            __m128 comp_r = comp_sse(in_r, layer_r);
            __m128 comp_g = comp_sse(in_g, layer_g);
            __m128 comp_b = comp_sse(in_b, layer_b);

            __m128 out_r = mul_add_ps128(comp_r, ratio,
                                         _mm_mul_ps(in_r, _mm_sub_ps(one, ratio)));
            __m128 out_g = mul_add_ps128(comp_g, ratio,
                                         _mm_mul_ps(in_g, _mm_sub_ps(one, ratio)));
            __m128 out_b = mul_add_ps128(comp_b, ratio,
                                         _mm_mul_ps(in_b, _mm_sub_ps(one, ratio)));

            if (clip_output) {
                out_r = _mm_min_ps(_mm_max_ps(out_r, _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f));
                out_g = _mm_min_ps(_mm_max_ps(out_g, _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f));
                out_b = _mm_min_ps(_mm_max_ps(out_b, _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f));
            }

            if (output.is_uint8 && output.channels == 4) {
                store_rgba4_u8(&output, index, out_r, out_g, out_b, in_a);
            } else {
                store_rgb4(&output, index, out_r, out_g, out_b);
                store_alpha4(&output, index, in_a);
            }
        }
    }
#endif
#endif

    for (; index < pixels; ++index) {
        npy_intp bg_offset = index * background.channels;
        npy_intp fg_offset = index * foreground.channels;

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
            float comp = comp_scalar(in_c, layer_c);
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

    Py_XDECREF(kernel_hold);
    release_blend_inputs(&background, &foreground);
    return (PyObject *)output.array;
}

static inline PyObject *blend_normal_mode(PyObject *args) {
    BlendArray background = {0};
    BlendArray foreground = {0};
    float opacity = 0.0f;
    npy_intp height = 0;
    npy_intp width = 0;
    const char *kernel_name = NULL;
    PyObject *kernel_hold = NULL;

    if (!parse_blend_inputs(args, &background, &foreground, &opacity, &height, &width,
                            &kernel_name, &kernel_hold)) {
        return NULL;
    }

    BlendOutput output = {0};
    if (opacity <= 0.0f) {
        if (!copy_background(&background, &output, height, width)) {
            release_blend_inputs(&background, &foreground);
            return NULL;
        }
        release_blend_inputs(&background, &foreground);
        return (PyObject *)output.array;
    }
    if (opacity >= 1.0f && foreground.channels == 3) {
        if (!alloc_output(&background, height, width, &output)) {
            Py_XDECREF(kernel_hold);
            release_blend_inputs(&background, &foreground);
            return NULL;
        }
        npy_intp pixels = height * width;
        for (npy_intp index = 0; index < pixels; ++index) {
            npy_intp fg_offset = index * foreground.channels;
            npy_intp out_offset = index * output.channels;
            for (int c = 0; c < 3; ++c) {
                float value = read_channel(&foreground, fg_offset + c);
                write_channel(&output, out_offset + c, value);
            }
            if (output.channels == 4) {
                write_channel(&output, out_offset + 3, 1.0f);
            }
        }
        Py_XDECREF(kernel_hold);
        release_blend_inputs(&background, &foreground);
        return (PyObject *)output.array;
    }

    if (!alloc_output(&background, height, width, &output)) {
        Py_XDECREF(kernel_hold);
        release_blend_inputs(&background, &foreground);
        return NULL;
    }

    kernel_kind kernel = pick_kernel(kernel_name);

    npy_intp pixels = height * width;
    npy_intp index = 0;
#if SIMD_BLEND_MODES_X86
    const __m128 one = _mm_set1_ps(1.0f);
    const __m256 one256 = _mm256_set1_ps(1.0f);
    const __m128 opacity128 = _mm_set1_ps(opacity);
    const __m256 opacity256 = _mm256_set1_ps(opacity);
    const __m128 inv255128 = _mm_set1_ps(1.0f / 255.0f);
    const __m256 inv255256 = _mm256_set1_ps(1.0f / 255.0f);

#if defined(__AVX2__)
    if (kernel == KERNEL_AVX2) {
        const npy_intp prefetch_distance = 16;
        for (; index + 7 < pixels; index += 8) {
            npy_intp prefetch_index = index + prefetch_distance;
            if (background.is_uint8) {
                _mm_prefetch((const char *)(background.u8 + (prefetch_index * background.channels)),
                             _MM_HINT_T0);
            } else {
                _mm_prefetch((const char *)(background.f32 + (prefetch_index * background.channels)),
                             _MM_HINT_T0);
            }
            if (foreground.is_uint8) {
                _mm_prefetch((const char *)(foreground.u8 + (prefetch_index * foreground.channels)),
                             _MM_HINT_T0);
            } else {
                _mm_prefetch((const char *)(foreground.f32 + (prefetch_index * foreground.channels)),
                             _MM_HINT_T0);
            }
            __m256 in_r, in_g, in_b, in_a;
            __m256 layer_r, layer_g, layer_b, layer_a;
            if (background.is_uint8 && background.channels == 4) {
                load_rgba8_u8_to_unit_f32_avx2(
                    background.u8 + (index * background.channels),
                    inv255256,
                    &in_r,
                    &in_g,
                    &in_b,
                    &in_a
                );
            } else {
                load_rgb8(&background, index, &in_r, &in_g, &in_b);
                in_a = load_alpha8(&background, index);
            }

            if (foreground.is_uint8 && foreground.channels == 4) {
                load_rgba8_u8_to_unit_f32_avx2(
                    foreground.u8 + (index * foreground.channels),
                    inv255256,
                    &layer_r,
                    &layer_g,
                    &layer_b,
                    &layer_a
                );
            } else {
                load_rgb8(&foreground, index, &layer_r, &layer_g, &layer_b);
                layer_a = load_alpha8(&foreground, index);
            }
            __m256 layer_opacity = _mm256_mul_ps(layer_a, opacity256);

            __m256 denom = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity), layer_opacity);
            __m256 mask = _mm256_cmp_ps(denom, _mm256_set1_ps(0.0f), _CMP_GT_OQ);

            __m256 layer_r_opacity = _mm256_mul_ps(layer_r, layer_opacity);
            __m256 layer_g_opacity = _mm256_mul_ps(layer_g, layer_opacity);
            __m256 layer_b_opacity = _mm256_mul_ps(layer_b, layer_opacity);
            __m256 inv_layer_opacity = _mm256_sub_ps(one256, layer_opacity);
            __m256 in_a_scaled = _mm256_mul_ps(in_a, inv_layer_opacity);
            __m256 num_r = mul_add_ps256(in_r, in_a_scaled, layer_r_opacity);
            __m256 num_g = mul_add_ps256(in_g, in_a_scaled, layer_g_opacity);
            __m256 num_b = mul_add_ps256(in_b, in_a_scaled, layer_b_opacity);

            __m256 out_r = _mm256_div_ps(num_r, denom);
            __m256 out_g = _mm256_div_ps(num_g, denom);
            __m256 out_b = _mm256_div_ps(num_b, denom);
            out_r = _mm256_blendv_ps(_mm256_set1_ps(0.0f), out_r, mask);
            out_g = _mm256_blendv_ps(_mm256_set1_ps(0.0f), out_g, mask);
            out_b = _mm256_blendv_ps(_mm256_set1_ps(0.0f), out_b, mask);

            __m256 out_a = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity), layer_opacity);

            if (output.is_uint8 && output.channels == 4) {
                store_rgba8_u8(&output, index, out_r, out_g, out_b, out_a);
            } else {
                store_rgb8(&output, index, out_r, out_g, out_b);
                store_alpha8(&output, index, out_a);
            }
        }
    }
#endif

#if defined(__SSE4_1__)
    if (kernel == KERNEL_SSE42 || kernel == KERNEL_AVX2) {
        for (; index + 3 < pixels; index += 4) {
            __m128 in_r, in_g, in_b, in_a;
            __m128 layer_r, layer_g, layer_b, layer_a;
            if (background.is_uint8 && background.channels == 4) {
                load_rgba4_u8_to_unit_f32_sse(
                    background.u8 + (index * background.channels),
                    inv255128,
                    &in_r,
                    &in_g,
                    &in_b,
                    &in_a
                );
            } else {
                load_rgb4(&background, index, &in_r, &in_g, &in_b);
                in_a = load_alpha4(&background, index);
            }

            if (foreground.is_uint8 && foreground.channels == 4) {
                load_rgba4_u8_to_unit_f32_sse(
                    foreground.u8 + (index * foreground.channels),
                    inv255128,
                    &layer_r,
                    &layer_g,
                    &layer_b,
                    &layer_a
                );
            } else {
                load_rgb4(&foreground, index, &layer_r, &layer_g, &layer_b);
                layer_a = load_alpha4(&foreground, index);
            }
            __m128 layer_opacity = _mm_mul_ps(layer_a, opacity128);

            __m128 denom = mul_add_ps128(in_a, _mm_sub_ps(one, layer_opacity), layer_opacity);
            __m128 mask = _mm_cmpgt_ps(denom, _mm_set1_ps(0.0f));

            __m128 layer_r_opacity = _mm_mul_ps(layer_r, layer_opacity);
            __m128 layer_g_opacity = _mm_mul_ps(layer_g, layer_opacity);
            __m128 layer_b_opacity = _mm_mul_ps(layer_b, layer_opacity);
            __m128 inv_layer_opacity = _mm_sub_ps(one, layer_opacity);
            __m128 in_a_scaled = _mm_mul_ps(in_a, inv_layer_opacity);
            __m128 num_r = mul_add_ps128(in_r, in_a_scaled, layer_r_opacity);
            __m128 num_g = mul_add_ps128(in_g, in_a_scaled, layer_g_opacity);
            __m128 num_b = mul_add_ps128(in_b, in_a_scaled, layer_b_opacity);

            __m128 out_r = _mm_div_ps(num_r, denom);
            __m128 out_g = _mm_div_ps(num_g, denom);
            __m128 out_b = _mm_div_ps(num_b, denom);
            out_r = _mm_blendv_ps(_mm_set1_ps(0.0f), out_r, mask);
            out_g = _mm_blendv_ps(_mm_set1_ps(0.0f), out_g, mask);
            out_b = _mm_blendv_ps(_mm_set1_ps(0.0f), out_b, mask);

            __m128 out_a = mul_add_ps128(in_a, _mm_sub_ps(one, layer_opacity), layer_opacity);

            if (output.is_uint8 && output.channels == 4) {
                store_rgba4_u8(&output, index, out_r, out_g, out_b, out_a);
            } else {
                store_rgb4(&output, index, out_r, out_g, out_b);
                store_alpha4(&output, index, out_a);
            }
        }
    }
#endif
#endif

    for (; index < pixels; ++index) {
        npy_intp bg_offset = index * background.channels;
        npy_intp fg_offset = index * foreground.channels;

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

    Py_XDECREF(kernel_hold);
    release_blend_inputs(&background, &foreground);
    return (PyObject *)output.array;
}

#endif
