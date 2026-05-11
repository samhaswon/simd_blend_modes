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
typedef __m128 (*blend_comp_fn128_u8)(__m128i in_c, __m128i layer_c, __m128 inv255);
typedef __m256 (*blend_comp_fn256_u8)(__m256i in_c, __m256i layer_c, __m256 inv255);
#else
typedef void *blend_comp_fn128;
typedef void *blend_comp_fn256;
typedef void *blend_comp_fn128_u8;
typedef void *blend_comp_fn256_u8;
#endif

#if SIMD_BLEND_MODES_X86
#define SIMD_BLEND_MODES_SIMD_ARGS(comp_sse, comp_avx) (comp_sse), (comp_avx)
#define SIMD_BLEND_MODES_SIMD_U8_ARGS(comp_sse, comp_avx) (comp_sse), (comp_avx)
#else
#define SIMD_BLEND_MODES_SIMD_ARGS(comp_sse, comp_avx) NULL, NULL
#define SIMD_BLEND_MODES_SIMD_U8_ARGS(comp_sse, comp_avx) NULL, NULL
#endif

#if SIMD_BLEND_MODES_X86
#if defined(_MSC_VER)
  #define SIMD_BLEND_MODES_ALIGN16 __declspec(align(16))
#else
  #define SIMD_BLEND_MODES_ALIGN16 __attribute__((aligned(16)))
#endif

static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_rgba_r[16] = {
    0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_rgba_g[16] = {
    1, 5, 9, 13, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_rgba_b[16] = {
    2, 6, 10, 14, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_rgba_a[16] = {
    3, 7, 11, 15, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_rgb_r[16] = {
    0, 3, 6, 9, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_rgb_g[16] = {
    1, 4, 7, 10, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_rgb_b[16] = {
    2, 5, 8, 11, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_pack_rgb_r[16] = {
    0, 0x80, 0x80, 1,
    0x80, 0x80, 2, 0x80,
    0x80, 3, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_pack_rgb_g[16] = {
    0x80, 0, 0x80, 0x80,
    1, 0x80, 0x80, 2,
    0x80, 0x80, 3, 0x80,
    0x80, 0x80, 0x80, 0x80
};
static const SIMD_BLEND_MODES_ALIGN16 uint8_t k_mask_pack_rgb_b[16] = {
    0x80, 0x80, 0, 0x80,
    0x80, 1, 0x80, 0x80,
    2, 0x80, 0x80, 3,
    0x80, 0x80, 0x80, 0x80
};

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

static inline __m128 u32_to_unit_f32_sse(__m128i v, __m128 inv255) {
    return _mm_mul_ps(_mm_cvtepi32_ps(v), inv255);
}

static inline __m256 u32_to_unit_f32_avx2(__m256i v, __m256 inv255) {
    return _mm256_mul_ps(_mm256_cvtepi32_ps(v), inv255);
}

/* Convert 8 consecutive u8 to float32 in [0,1] (for grayscale im_alpha). */
static inline __m256 load8_u8_to_unit_f32_avx2(const uint8_t *p, __m256 inv255) {
    __m128i v8  = _mm_loadl_epi64((const __m128i*)p);        /* 8 bytes -> XMM */
    __m256i v32 = _mm256_cvtepu8_epi32(v8);                  /* widen to 8 x u32 */
    return _mm256_mul_ps(_mm256_cvtepi32_ps(v32), inv255);
}

static inline void load_rgba8_u8_to_u32_avx2(const uint8_t *p,
                                             __m256i *r, __m256i *g,
                                             __m256i *b, __m256i *a) {
    __m256i pixels = _mm256_loadu_si256((const __m256i *)p);
    __m128i lo = _mm256_castsi256_si128(pixels);
    __m128i hi = _mm256_extracti128_si256(pixels, 1);

    const __m128i mask_r = _mm_load_si128((const __m128i *)k_mask_rgba_r);
    const __m128i mask_g = _mm_load_si128((const __m128i *)k_mask_rgba_g);
    const __m128i mask_b = _mm_load_si128((const __m128i *)k_mask_rgba_b);
    const __m128i mask_a = _mm_load_si128((const __m128i *)k_mask_rgba_a);

    __m128i r_lo = _mm_shuffle_epi8(lo, mask_r);
    __m128i r_hi = _mm_shuffle_epi8(hi, mask_r);
    __m128i g_lo = _mm_shuffle_epi8(lo, mask_g);
    __m128i g_hi = _mm_shuffle_epi8(hi, mask_g);
    __m128i b_lo = _mm_shuffle_epi8(lo, mask_b);
    __m128i b_hi = _mm_shuffle_epi8(hi, mask_b);
    __m128i a_lo = _mm_shuffle_epi8(lo, mask_a);
    __m128i a_hi = _mm_shuffle_epi8(hi, mask_a);

    __m128i r32_lo = _mm_cvtepu8_epi32(r_lo);
    __m128i r32_hi = _mm_cvtepu8_epi32(r_hi);
    __m128i g32_lo = _mm_cvtepu8_epi32(g_lo);
    __m128i g32_hi = _mm_cvtepu8_epi32(g_hi);
    __m128i b32_lo = _mm_cvtepu8_epi32(b_lo);
    __m128i b32_hi = _mm_cvtepu8_epi32(b_hi);
    __m128i a32_lo = _mm_cvtepu8_epi32(a_lo);
    __m128i a32_hi = _mm_cvtepu8_epi32(a_hi);

    *r = _mm256_inserti128_si256(_mm256_castsi128_si256(r32_lo), r32_hi, 1);
    *g = _mm256_inserti128_si256(_mm256_castsi128_si256(g32_lo), g32_hi, 1);
    *b = _mm256_inserti128_si256(_mm256_castsi128_si256(b32_lo), b32_hi, 1);
    *a = _mm256_inserti128_si256(_mm256_castsi128_si256(a32_lo), a32_hi, 1);
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

    const __m128i mask_r = _mm_load_si128((const __m128i *)k_mask_rgba_r);
    const __m128i mask_g = _mm_load_si128((const __m128i *)k_mask_rgba_g);
    const __m128i mask_b = _mm_load_si128((const __m128i *)k_mask_rgba_b);
    const __m128i mask_a = _mm_load_si128((const __m128i *)k_mask_rgba_a);

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

    const __m128i mask_r = _mm_load_si128((const __m128i *)k_mask_pack_rgb_r);
    const __m128i mask_g = _mm_load_si128((const __m128i *)k_mask_pack_rgb_g);
    const __m128i mask_b = _mm_load_si128((const __m128i *)k_mask_pack_rgb_b);

    __m128i packed = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(ir8, mask_r),
                     _mm_shuffle_epi8(ig8, mask_g)),
        _mm_shuffle_epi8(ib8, mask_b));

    return packed;
}

static inline __m128i pack_u32_to_u8_rgb4(__m128i r, __m128i g, __m128i b) {
    const __m128i zero = _mm_setzero_si128();
    __m128i r16 = _mm_packus_epi32(r, zero);
    __m128i g16 = _mm_packus_epi32(g, zero);
    __m128i b16 = _mm_packus_epi32(b, zero);

    __m128i r8 = _mm_packus_epi16(r16, zero);
    __m128i g8 = _mm_packus_epi16(g16, zero);
    __m128i b8 = _mm_packus_epi16(b16, zero);

    const __m128i mask_r = _mm_load_si128((const __m128i *)k_mask_pack_rgb_r);
    const __m128i mask_g = _mm_load_si128((const __m128i *)k_mask_pack_rgb_g);
    const __m128i mask_b = _mm_load_si128((const __m128i *)k_mask_pack_rgb_b);

    __m128i packed = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(r8, mask_r),
                     _mm_shuffle_epi8(g8, mask_g)),
        _mm_shuffle_epi8(b8, mask_b));

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

static inline void store_u32_to_u8_rgb4(__m128i r, __m128i g, __m128i b, uint8_t *out_ptr) {
    __m128i packed = pack_u32_to_u8_rgb4(r, g, b);
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

static inline void load_rgba4_f32_to_unit_f32_sse(const float *p, __m128 inv255,
                                                  __m128 *r, __m128 *g, __m128 *b,
                                                  __m128 *a) {
    __m128 row0 = _mm_loadu_ps(p + 0);
    __m128 row1 = _mm_loadu_ps(p + 4);
    __m128 row2 = _mm_loadu_ps(p + 8);
    __m128 row3 = _mm_loadu_ps(p + 12);

    __m128 t0 = _mm_unpacklo_ps(row0, row1);
    __m128 t1 = _mm_unpackhi_ps(row0, row1);
    __m128 t2 = _mm_unpacklo_ps(row2, row3);
    __m128 t3 = _mm_unpackhi_ps(row2, row3);

    *r = _mm_mul_ps(_mm_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)), inv255);
    *g = _mm_mul_ps(_mm_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)), inv255);
    *b = _mm_mul_ps(_mm_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)), inv255);
    *a = _mm_mul_ps(_mm_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2)), inv255);
}

static inline void load_rgb4_f32_to_unit_f32_sse(const float *p, __m128 inv255,
                                                 __m128 *r, __m128 *g, __m128 *b) {
    __m128 v0 = _mm_loadu_ps(p + 0);
    __m128 v1 = _mm_loadu_ps(p + 4);
    __m128 v2 = _mm_loadu_ps(p + 8);

    __m128 r01 = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(3, 3, 0, 0));
    __m128 r23 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 1, 2, 2));
    __m128 g01 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(0, 0, 1, 1));
    __m128 g23 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(2, 2, 3, 3));
    __m128 b01 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(1, 1, 2, 2));
    __m128 b23 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 3, 0, 0));

    *r = _mm_mul_ps(_mm_shuffle_ps(r01, r23, _MM_SHUFFLE(2, 0, 2, 0)), inv255);
    *g = _mm_mul_ps(_mm_shuffle_ps(g01, g23, _MM_SHUFFLE(2, 0, 2, 0)), inv255);
    *b = _mm_mul_ps(_mm_shuffle_ps(b01, b23, _MM_SHUFFLE(2, 0, 2, 0)), inv255);
}

static inline void store_rgba4_f32_from_unit_sse(float *p, __m128 r, __m128 g, __m128 b,
                                                 __m128 a) {
    __m128 scale = _mm_set1_ps(255.0f);
    r = _mm_mul_ps(r, scale);
    g = _mm_mul_ps(g, scale);
    b = _mm_mul_ps(b, scale);
    a = _mm_mul_ps(a, scale);

    __m128 t0 = _mm_unpacklo_ps(r, g);
    __m128 t1 = _mm_unpackhi_ps(r, g);
    __m128 t2 = _mm_unpacklo_ps(b, a);
    __m128 t3 = _mm_unpackhi_ps(b, a);

    __m128 row0 = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 row1 = _mm_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 row2 = _mm_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 row3 = _mm_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));

    _mm_storeu_ps(p + 0, row0);
    _mm_storeu_ps(p + 4, row1);
    _mm_storeu_ps(p + 8, row2);
    _mm_storeu_ps(p + 12, row3);
}

static inline void load_rgb8_f32_to_unit_f32_avx2(const float *p, __m128 inv255,
                                                  __m256 *r, __m256 *g, __m256 *b) {
    __m128 r0, g0, b0;
    __m128 r1, g1, b1;
    load_rgb4_f32_to_unit_f32_sse(p, inv255, &r0, &g0, &b0);
    load_rgb4_f32_to_unit_f32_sse(p + 12, inv255, &r1, &g1, &b1);
    *r = _mm256_set_m128(r1, r0);
    *g = _mm256_set_m128(g1, g0);
    *b = _mm256_set_m128(b1, b0);
}

static inline void load_rgba8_f32_to_unit_f32_avx2(const float *p, __m128 inv255,
                                                   __m256 *r, __m256 *g, __m256 *b,
                                                   __m256 *a) {
    __m128 r0, g0, b0, a0;
    __m128 r1, g1, b1, a1;
    load_rgba4_f32_to_unit_f32_sse(p, inv255, &r0, &g0, &b0, &a0);
    load_rgba4_f32_to_unit_f32_sse(p + 16, inv255, &r1, &g1, &b1, &a1);
    *r = _mm256_set_m128(r1, r0);
    *g = _mm256_set_m128(g1, g0);
    *b = _mm256_set_m128(b1, b0);
    *a = _mm256_set_m128(a1, a0);
}

static inline void store_rgba8_f32_from_unit_avx2(float *p, __m256 r, __m256 g, __m256 b,
                                                  __m256 a) {
    __m128 r0 = _mm256_castps256_ps128(r);
    __m128 r1 = _mm256_extractf128_ps(r, 1);
    __m128 g0 = _mm256_castps256_ps128(g);
    __m128 g1 = _mm256_extractf128_ps(g, 1);
    __m128 b0 = _mm256_castps256_ps128(b);
    __m128 b1 = _mm256_extractf128_ps(b, 1);
    __m128 a0 = _mm256_castps256_ps128(a);
    __m128 a1 = _mm256_extractf128_ps(a, 1);

    store_rgba4_f32_from_unit_sse(p, r0, g0, b0, a0);
    store_rgba4_f32_from_unit_sse(p + 16, r1, g1, b1, a1);
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
    const __m128i mask_r = _mm_load_si128((const __m128i *)k_mask_rgba_r);
    const __m128i mask_g = _mm_load_si128((const __m128i *)k_mask_rgba_g);
    const __m128i mask_b = _mm_load_si128((const __m128i *)k_mask_rgba_b);
    const __m128i mask_a = _mm_load_si128((const __m128i *)k_mask_rgba_a);

    __m128i r8 = _mm_shuffle_epi8(pixels, mask_r);
    __m128i g8 = _mm_shuffle_epi8(pixels, mask_g);
    __m128i b8 = _mm_shuffle_epi8(pixels, mask_b);
    __m128i a8 = _mm_shuffle_epi8(pixels, mask_a);

    *r = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(r8)), inv255);
    *g = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(g8)), inv255);
    *b = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(b8)), inv255);
    *a = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(a8)), inv255);
}

static inline __m128i load_rgb4_u8_bytes_sse(const uint8_t *p) {
    __m128i lo = _mm_loadl_epi64((const __m128i*)p);
    __m128i hi = _mm_cvtsi32_si128(*(const int*)(p + 8));
    return _mm_unpacklo_epi64(lo, hi);
}

static inline void load_rgb4_u8_to_unit_f32_sse(const uint8_t *p, __m128 inv255,
                                                __m128 *r, __m128 *g, __m128 *b) {
    __m128i pixels = load_rgb4_u8_bytes_sse(p);
    const __m128i mask_r = _mm_load_si128((const __m128i *)k_mask_rgb_r);
    const __m128i mask_g = _mm_load_si128((const __m128i *)k_mask_rgb_g);
    const __m128i mask_b = _mm_load_si128((const __m128i *)k_mask_rgb_b);

    __m128i r8 = _mm_shuffle_epi8(pixels, mask_r);
    __m128i g8 = _mm_shuffle_epi8(pixels, mask_g);
    __m128i b8 = _mm_shuffle_epi8(pixels, mask_b);

    *r = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(r8)), inv255);
    *g = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(g8)), inv255);
    *b = _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(b8)), inv255);
}

static inline void load_rgb4_u8_to_u32_sse_from_ptr(const uint8_t *p, __m128i *r,
                                                    __m128i *g, __m128i *b) {
    __m128i pixels = load_rgb4_u8_bytes_sse(p);
    const __m128i mask_r = _mm_load_si128((const __m128i *)k_mask_rgb_r);
    const __m128i mask_g = _mm_load_si128((const __m128i *)k_mask_rgb_g);
    const __m128i mask_b = _mm_load_si128((const __m128i *)k_mask_rgb_b);

    __m128i r8 = _mm_shuffle_epi8(pixels, mask_r);
    __m128i g8 = _mm_shuffle_epi8(pixels, mask_g);
    __m128i b8 = _mm_shuffle_epi8(pixels, mask_b);

    *r = _mm_cvtepu8_epi32(r8);
    *g = _mm_cvtepu8_epi32(g8);
    *b = _mm_cvtepu8_epi32(b8);
}

static inline void load_rgb4_u8_to_u32_sse_with_alpha(const uint8_t *p, __m128i *r,
                                                      __m128i *g, __m128i *b,
                                                      __m128i *a) {
    load_rgb4_u8_to_u32_sse_from_ptr(p, r, g, b);
    *a = _mm_set1_epi32(255);
}

static inline void load_rgba4_u8_to_u32_sse(const uint8_t *p,
                                            __m128i *r, __m128i *g,
                                            __m128i *b, __m128i *a) {
    __m128i pixels = _mm_loadu_si128((const __m128i *)p);
    const __m128i mask_r = _mm_load_si128((const __m128i *)k_mask_rgba_r);
    const __m128i mask_g = _mm_load_si128((const __m128i *)k_mask_rgba_g);
    const __m128i mask_b = _mm_load_si128((const __m128i *)k_mask_rgba_b);
    const __m128i mask_a = _mm_load_si128((const __m128i *)k_mask_rgba_a);

    __m128i r8 = _mm_shuffle_epi8(pixels, mask_r);
    __m128i g8 = _mm_shuffle_epi8(pixels, mask_g);
    __m128i b8 = _mm_shuffle_epi8(pixels, mask_b);
    __m128i a8 = _mm_shuffle_epi8(pixels, mask_a);

    *r = _mm_cvtepu8_epi32(r8);
    *g = _mm_cvtepu8_epi32(g8);
    *b = _mm_cvtepu8_epi32(b8);
    *a = _mm_cvtepu8_epi32(a8);
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

static inline float read_channel_u8(const uint8_t *data, npy_intp index) {
    return ((float)data[index]) / 255.0f;
}

static inline float read_channel_f32(const float *data, npy_intp index) {
    return data[index] / 255.0f;
}

static inline float read_channel(const BlendArray *array, npy_intp index) {
    if (array->is_uint8) {
        return read_channel_u8(array->u8, index);
    }
    return read_channel_f32(array->f32, index);
}

static inline void write_channel_u8(uint8_t *data, npy_intp index, float value) {
    float scaled = value * 255.0f;
    if (scaled < 0.0f) {
        scaled = 0.0f;
    } else if (scaled > 255.0f) {
        scaled = 255.0f;
    }
    data[index] = (uint8_t)lroundf(scaled);
}

static inline void write_channel_f32(float *data, npy_intp index, float value) {
    data[index] = value * 255.0f;
}

static inline void write_channel(BlendOutput *output, npy_intp index, float value) {
    if (output->is_uint8) {
        write_channel_u8(output->u8, index, value);
        return;
    }
    write_channel_f32(output->f32, index, value);
}

#if SIMD_BLEND_MODES_X86
static inline int ptr_is_aligned(const void *ptr, uintptr_t alignment) {
    return (((uintptr_t)ptr) & (alignment - 1)) == 0;
}

static inline __m128i pack_two_unit_f32_avx2_to_u8(__m256 lo, __m256 hi) {
    const __m256 scale = _mm256_set1_ps(255.0f);
    const __m256 zero_ps = _mm256_set1_ps(0.0f);
    const __m256 max_ps = _mm256_set1_ps(255.0f);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max255 = _mm256_set1_epi32(255);

    __m256i lo_i = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(lo, scale),
                                                                  zero_ps), max_ps));
    __m256i hi_i = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(hi, scale),
                                                                  zero_ps), max_ps));
    lo_i = _mm256_min_epi32(_mm256_max_epi32(lo_i, zero), max255);
    hi_i = _mm256_min_epi32(_mm256_max_epi32(hi_i, zero), max255);

    __m128i lo_0 = _mm256_castsi256_si128(lo_i);
    __m128i lo_1 = _mm256_extracti128_si256(lo_i, 1);
    __m128i hi_0 = _mm256_castsi256_si128(hi_i);
    __m128i hi_1 = _mm256_extracti128_si256(hi_i, 1);
    __m128i lo_16 = _mm_packus_epi32(lo_0, lo_1);
    __m128i hi_16 = _mm_packus_epi32(hi_0, hi_1);
    return _mm_packus_epi16(lo_16, hi_16);
}

static inline __m128i pack_unit_f32_sse_to_u8(__m128 value) {
    const __m128 scale = _mm_set1_ps(255.0f);
    const __m128 zero_ps = _mm_set1_ps(0.0f);
    const __m128 max_ps = _mm_set1_ps(255.0f);
    const __m128i zero = _mm_setzero_si128();
    const __m128i max255 = _mm_set1_epi32(255);

    __m128i value_i = _mm_cvtps_epi32(_mm_min_ps(_mm_max_ps(_mm_mul_ps(value, scale),
                                                           zero_ps), max_ps));
    value_i = _mm_min_epi32(_mm_max_epi32(value_i, zero), max255);
    value_i = _mm_packus_epi32(value_i, zero);
    return _mm_packus_epi16(value_i, zero);
}

static inline __m256 blend_flat_ps256(__m256 in_c, __m256 layer_c, __m256 opacity,
                                      __m256 one, blend_comp_fn256 comp_avx,
                                      int clip_output) {
    __m256 comp = comp_avx(in_c, layer_c);
    __m256 out = mul_add_ps256(comp, opacity,
                               _mm256_mul_ps(in_c, _mm256_sub_ps(one, opacity)));
    if (clip_output) {
        out = clamp01_ps(out);
    }
    return out;
}

static inline __m128 blend_flat_ps128(__m128 in_c, __m128 layer_c, __m128 opacity,
                                      __m128 one, blend_comp_fn128 comp_sse,
                                      int clip_output) {
    __m128 comp = comp_sse(in_c, layer_c);
    __m128 out = mul_add_ps128(comp, opacity, _mm_mul_ps(in_c, _mm_sub_ps(one, opacity)));
    if (clip_output) {
        out = clamp01_ps128(out);
    }
    return out;
}

static inline __m256 load_f32x8(const float *p, int aligned) {
    if (aligned) {
        return _mm256_load_ps(p);
    }
    return _mm256_loadu_ps(p);
}

static inline void store_f32x8(float *p, __m256 value, int aligned) {
    if (aligned) {
        _mm256_store_ps(p, value);
        return;
    }
    _mm256_storeu_ps(p, value);
}

static inline __m128 load_f32x4(const float *p, int aligned) {
    if (aligned) {
        return _mm_load_ps(p);
    }
    return _mm_loadu_ps(p);
}

static inline void store_f32x4(float *p, __m128 value, int aligned) {
    if (aligned) {
        _mm_store_ps(p, value);
        return;
    }
    _mm_storeu_ps(p, value);
}

static inline void load_rgb4_u8(const uint8_t *data, int channels, npy_intp index,
                                __m128 *r, __m128 *g, __m128 *b) {
    if (channels == 3) {
        const uint8_t *p = data + (index * channels);
        load_rgb4_u8_to_unit_f32_sse(p, _mm_set1_ps(1.0f / 255.0f), r, g, b);
        return;
    }
    if (channels == 4) {
        const uint8_t *p = data + (index * channels);
        __m128 a_unused;
        load_rgba4_u8_to_unit_f32_sse(p, _mm_set1_ps(1.0f / 255.0f), r, g, b, &a_unused);
        return;
    }

    npy_intp base0 = (index + 0) * channels;
    npy_intp base1 = (index + 1) * channels;
    npy_intp base2 = (index + 2) * channels;
    npy_intp base3 = (index + 3) * channels;

    float r0 = read_channel_u8(data, base0 + 0);
    float g0 = read_channel_u8(data, base0 + 1);
    float b0 = read_channel_u8(data, base0 + 2);
    float r1 = read_channel_u8(data, base1 + 0);
    float g1 = read_channel_u8(data, base1 + 1);
    float b1 = read_channel_u8(data, base1 + 2);
    float r2 = read_channel_u8(data, base2 + 0);
    float g2 = read_channel_u8(data, base2 + 1);
    float b2 = read_channel_u8(data, base2 + 2);
    float r3 = read_channel_u8(data, base3 + 0);
    float g3 = read_channel_u8(data, base3 + 1);
    float b3 = read_channel_u8(data, base3 + 2);

    *r = _mm_set_ps(r3, r2, r1, r0);
    *g = _mm_set_ps(g3, g2, g1, g0);
    *b = _mm_set_ps(b3, b2, b1, b0);
}

static inline void load_rgb4_u8_to_u32_sse(const uint8_t *data, int channels, npy_intp index,
                                           __m128i *r, __m128i *g, __m128i *b) {
    if (channels == 3) {
        const uint8_t *p = data + (index * channels);
        load_rgb4_u8_to_u32_sse_from_ptr(p, r, g, b);
        return;
    }
    if (channels == 4) {
        const uint8_t *p = data + (index * channels);
        __m128i a_unused;
        load_rgba4_u8_to_u32_sse(p, r, g, b, &a_unused);
        return;
    }

    npy_intp base0 = (index + 0) * channels;
    npy_intp base1 = (index + 1) * channels;
    npy_intp base2 = (index + 2) * channels;
    npy_intp base3 = (index + 3) * channels;

    uint32_t r0 = data[base0 + 0];
    uint32_t g0 = data[base0 + 1];
    uint32_t b0 = data[base0 + 2];
    uint32_t r1 = data[base1 + 0];
    uint32_t g1 = data[base1 + 1];
    uint32_t b1 = data[base1 + 2];
    uint32_t r2 = data[base2 + 0];
    uint32_t g2 = data[base2 + 1];
    uint32_t b2 = data[base2 + 2];
    uint32_t r3 = data[base3 + 0];
    uint32_t g3 = data[base3 + 1];
    uint32_t b3 = data[base3 + 2];

    *r = _mm_set_epi32((int)r3, (int)r2, (int)r1, (int)r0);
    *g = _mm_set_epi32((int)g3, (int)g2, (int)g1, (int)g0);
    *b = _mm_set_epi32((int)b3, (int)b2, (int)b1, (int)b0);
}

static inline void load_rgb4_f32(const float *data, int channels, npy_intp index,
                                 __m128 *r, __m128 *g, __m128 *b) {
    npy_intp base0 = (index + 0) * channels;
    npy_intp base1 = (index + 1) * channels;
    npy_intp base2 = (index + 2) * channels;
    npy_intp base3 = (index + 3) * channels;

    float r0 = read_channel_f32(data, base0 + 0);
    float g0 = read_channel_f32(data, base0 + 1);
    float b0 = read_channel_f32(data, base0 + 2);
    float r1 = read_channel_f32(data, base1 + 0);
    float g1 = read_channel_f32(data, base1 + 1);
    float b1 = read_channel_f32(data, base1 + 2);
    float r2 = read_channel_f32(data, base2 + 0);
    float g2 = read_channel_f32(data, base2 + 1);
    float b2 = read_channel_f32(data, base2 + 2);
    float r3 = read_channel_f32(data, base3 + 0);
    float g3 = read_channel_f32(data, base3 + 1);
    float b3 = read_channel_f32(data, base3 + 2);

    *r = _mm_set_ps(r3, r2, r1, r0);
    *g = _mm_set_ps(g3, g2, g1, g0);
    *b = _mm_set_ps(b3, b2, b1, b0);
}

static inline __m128 load_alpha4_u8(const uint8_t *data, int channels, npy_intp index) {
    if (channels == 4) {
        npy_intp base0 = (index + 0) * channels + 3;
        npy_intp base1 = (index + 1) * channels + 3;
        npy_intp base2 = (index + 2) * channels + 3;
        npy_intp base3 = (index + 3) * channels + 3;
        float a0 = read_channel_u8(data, base0);
        float a1 = read_channel_u8(data, base1);
        float a2 = read_channel_u8(data, base2);
        float a3 = read_channel_u8(data, base3);
        return _mm_set_ps(a3, a2, a1, a0);
    }
    return _mm_set1_ps(1.0f);
}

static inline __m128 load_alpha4_f32(const float *data, int channels, npy_intp index) {
    if (channels == 4) {
        npy_intp base0 = (index + 0) * channels + 3;
        npy_intp base1 = (index + 1) * channels + 3;
        npy_intp base2 = (index + 2) * channels + 3;
        npy_intp base3 = (index + 3) * channels + 3;
        float a0 = read_channel_f32(data, base0);
        float a1 = read_channel_f32(data, base1);
        float a2 = read_channel_f32(data, base2);
        float a3 = read_channel_f32(data, base3);
        return _mm_set_ps(a3, a2, a1, a0);
    }
    return _mm_set1_ps(1.0f);
}

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
    if (output->is_uint8 && output->channels == 3) {
        uint8_t *out_ptr = output->u8 + (index * output->channels);
        store_unit_f32_to_u8_rgb4(r, g, b, out_ptr);
        return;
    }

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

static inline void load_rgb8_u8(const uint8_t *data, int channels, npy_intp index,
                                __m256 *r, __m256 *g, __m256 *b) {
    if (channels == 3) {
        __m128 r0, g0, b0;
        __m128 r1, g1, b1;
        __m128 inv255 = _mm_set1_ps(1.0f / 255.0f);
        load_rgb4_u8_to_unit_f32_sse(data + (index * channels), inv255, &r0, &g0, &b0);
        load_rgb4_u8_to_unit_f32_sse(data + ((index + 4) * channels), inv255, &r1, &g1,
                                     &b1);
        *r = _mm256_set_m128(r1, r0);
        *g = _mm256_set_m128(g1, g0);
        *b = _mm256_set_m128(b1, b0);
        return;
    }
    if (channels == 4) {
        __m256 a_unused;
        load_rgba8_u8_to_unit_f32_avx2(data + (index * channels), _mm256_set1_ps(1.0f / 255.0f),
                                       r, g, b, &a_unused);
        return;
    }

    float rv[8];
    float gv[8];
    float bv[8];
    for (int i = 0; i < 8; ++i) {
        npy_intp base = (index + i) * channels;
        rv[i] = read_channel_u8(data, base + 0);
        gv[i] = read_channel_u8(data, base + 1);
        bv[i] = read_channel_u8(data, base + 2);
    }
    *r = _mm256_set_ps(rv[7], rv[6], rv[5], rv[4], rv[3], rv[2], rv[1], rv[0]);
    *g = _mm256_set_ps(gv[7], gv[6], gv[5], gv[4], gv[3], gv[2], gv[1], gv[0]);
    *b = _mm256_set_ps(bv[7], bv[6], bv[5], bv[4], bv[3], bv[2], bv[1], bv[0]);
}

static inline void load_rgb8_u8_to_u32_avx2(const uint8_t *data, int channels, npy_intp index,
                                            __m256i *r, __m256i *g, __m256i *b) {
    if (channels == 3) {
        __m128i r0, g0, b0;
        __m128i r1, g1, b1;
        load_rgb4_u8_to_u32_sse_from_ptr(data + (index * channels), &r0, &g0, &b0);
        load_rgb4_u8_to_u32_sse_from_ptr(data + ((index + 4) * channels), &r1, &g1, &b1);
        *r = _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
        *g = _mm256_inserti128_si256(_mm256_castsi128_si256(g0), g1, 1);
        *b = _mm256_inserti128_si256(_mm256_castsi128_si256(b0), b1, 1);
        return;
    }
    if (channels == 4) {
        __m256i a_unused;
        load_rgba8_u8_to_u32_avx2(data + (index * channels), r, g, b, &a_unused);
        return;
    }

    uint32_t rv[8];
    uint32_t gv[8];
    uint32_t bv[8];
    for (int i = 0; i < 8; ++i) {
        npy_intp base = (index + i) * channels;
        rv[i] = data[base + 0];
        gv[i] = data[base + 1];
        bv[i] = data[base + 2];
    }
    *r = _mm256_set_epi32((int)rv[7], (int)rv[6], (int)rv[5], (int)rv[4],
                          (int)rv[3], (int)rv[2], (int)rv[1], (int)rv[0]);
    *g = _mm256_set_epi32((int)gv[7], (int)gv[6], (int)gv[5], (int)gv[4],
                          (int)gv[3], (int)gv[2], (int)gv[1], (int)gv[0]);
    *b = _mm256_set_epi32((int)bv[7], (int)bv[6], (int)bv[5], (int)bv[4],
                          (int)bv[3], (int)bv[2], (int)bv[1], (int)bv[0]);
}

static inline void load_rgb8_u8_to_u32_avx2_with_alpha(const uint8_t *data, npy_intp index,
                                                       __m256i *r, __m256i *g, __m256i *b,
                                                       __m256i *a) {
    __m128i r0, g0, b0;
    __m128i r1, g1, b1;
    load_rgb4_u8_to_u32_sse_from_ptr(data + (index * 3), &r0, &g0, &b0);
    load_rgb4_u8_to_u32_sse_from_ptr(data + ((index + 4) * 3), &r1, &g1, &b1);
    *r = _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
    *g = _mm256_inserti128_si256(_mm256_castsi128_si256(g0), g1, 1);
    *b = _mm256_inserti128_si256(_mm256_castsi128_si256(b0), b1, 1);
    *a = _mm256_set1_epi32(255);
}

static inline void load_rgb8_f32(const float *data, int channels, npy_intp index,
                                 __m256 *r, __m256 *g, __m256 *b) {
    float rv[8];
    float gv[8];
    float bv[8];
    for (int i = 0; i < 8; ++i) {
        npy_intp base = (index + i) * channels;
        rv[i] = read_channel_f32(data, base + 0);
        gv[i] = read_channel_f32(data, base + 1);
        bv[i] = read_channel_f32(data, base + 2);
    }
    *r = _mm256_set_ps(rv[7], rv[6], rv[5], rv[4], rv[3], rv[2], rv[1], rv[0]);
    *g = _mm256_set_ps(gv[7], gv[6], gv[5], gv[4], gv[3], gv[2], gv[1], gv[0]);
    *b = _mm256_set_ps(bv[7], bv[6], bv[5], bv[4], bv[3], bv[2], bv[1], bv[0]);
}

static inline __m256 load_alpha8_u8(const uint8_t *data, int channels, npy_intp index) {
    if (channels == 4) {
        float av[8];
        for (int i = 0; i < 8; ++i) {
            npy_intp base = (index + i) * channels + 3;
            av[i] = read_channel_u8(data, base);
        }
        return _mm256_set_ps(av[7], av[6], av[5], av[4], av[3], av[2], av[1], av[0]);
    }
    return _mm256_set1_ps(1.0f);
}

static inline __m256 load_alpha8_f32(const float *data, int channels, npy_intp index) {
    if (channels == 4) {
        float av[8];
        for (int i = 0; i < 8; ++i) {
            npy_intp base = (index + i) * channels + 3;
            av[i] = read_channel_f32(data, base);
        }
        return _mm256_set_ps(av[7], av[6], av[5], av[4], av[3], av[2], av[1], av[0]);
    }
    return _mm256_set1_ps(1.0f);
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
    if (output->is_uint8 && output->channels == 3) {
        uint8_t *out_ptr = output->u8 + (index * output->channels);
        __m128 r0 = _mm256_castps256_ps128(r);
        __m128 r1 = _mm256_extractf128_ps(r, 1);
        __m128 g0 = _mm256_castps256_ps128(g);
        __m128 g1 = _mm256_extractf128_ps(g, 1);
        __m128 b0 = _mm256_castps256_ps128(b);
        __m128 b1 = _mm256_extractf128_ps(b, 1);

        store_unit_f32_to_u8_rgb4(r0, g0, b0, out_ptr);
        store_unit_f32_to_u8_rgb4(r1, g1, b1, out_ptr + 12);
        return;
    }

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
                                       blend_comp_fn128_u8 comp_sse_u8,
                                       blend_comp_fn256_u8 comp_avx_u8,
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
    NPY_BEGIN_ALLOW_THREADS
#if SIMD_BLEND_MODES_X86
    const __m128 one = _mm_set1_ps(1.0f);
    const __m256 one256 = _mm256_set1_ps(1.0f);
    const __m128 opacity128 = _mm_set1_ps(opacity);
    const __m256 opacity256 = _mm256_set1_ps(opacity);
    const __m128 inv255128 = _mm_set1_ps(1.0f / 255.0f);
    const __m256 inv255256 = _mm256_set1_ps(1.0f / 255.0f);

    if (background.channels == 3 && foreground.channels == 3) {
        const npy_intp elements = pixels * 3;
        npy_intp elem = 0;

#if defined(__AVX2__)
        if (kernel == KERNEL_AVX2) {
            if (background.is_uint8 && foreground.is_uint8 &&
                    ptr_is_aligned(background.u8, 16) &&
                    ptr_is_aligned(foreground.u8, 16) &&
                    ptr_is_aligned(output.u8, 16)) {
                for (; elem + 15 < elements; elem += 16) {
                    __m128i in_bytes = _mm_load_si128((const __m128i *)(background.u8 + elem));
                    __m128i layer_bytes = _mm_load_si128((const __m128i *)(foreground.u8 + elem));
                    __m256i in_lo_u = _mm256_cvtepu8_epi32(in_bytes);
                    __m256i layer_lo_u = _mm256_cvtepu8_epi32(layer_bytes);
                    __m128i in_hi_bytes = _mm_srli_si128(in_bytes, 8);
                    __m128i layer_hi_bytes = _mm_srli_si128(layer_bytes, 8);
                    __m256i in_hi_u = _mm256_cvtepu8_epi32(in_hi_bytes);
                    __m256i layer_hi_u = _mm256_cvtepu8_epi32(layer_hi_bytes);
                    __m256 in_lo = u32_to_unit_f32_avx2(in_lo_u, inv255256);
                    __m256 in_hi = u32_to_unit_f32_avx2(in_hi_u, inv255256);
                    __m256 comp_lo = comp_avx_u8 ? comp_avx_u8(in_lo_u, layer_lo_u, inv255256) :
                                      comp_avx(in_lo, u32_to_unit_f32_avx2(layer_lo_u, inv255256));
                    __m256 comp_hi = comp_avx_u8 ? comp_avx_u8(in_hi_u, layer_hi_u, inv255256) :
                                      comp_avx(in_hi, u32_to_unit_f32_avx2(layer_hi_u, inv255256));
                    __m256 out_lo = mul_add_ps256(comp_lo, opacity256,
                                                  _mm256_mul_ps(in_lo,
                                                               _mm256_sub_ps(one256, opacity256)));
                    __m256 out_hi = mul_add_ps256(comp_hi, opacity256,
                                                  _mm256_mul_ps(in_hi,
                                                               _mm256_sub_ps(one256, opacity256)));
                    if (clip_output) {
                        out_lo = clamp01_ps(out_lo);
                        out_hi = clamp01_ps(out_hi);
                    }
                    _mm_store_si128((__m128i *)(output.u8 + elem),
                                    pack_two_unit_f32_avx2_to_u8(out_lo, out_hi));
                }
            } else if (!background.is_uint8 && !foreground.is_uint8) {
                int bg_aligned = ptr_is_aligned(background.f32, 32);
                int fg_aligned = ptr_is_aligned(foreground.f32, 32);
                int out_aligned = ptr_is_aligned(output.f32, 32);
                for (; elem + 7 < elements; elem += 8) {
                    __m256 in_c = _mm256_mul_ps(load_f32x8(background.f32 + elem, bg_aligned),
                                                inv255256);
                    __m256 layer_c = _mm256_mul_ps(load_f32x8(foreground.f32 + elem, fg_aligned),
                                                   inv255256);
                    __m256 out_c = blend_flat_ps256(in_c, layer_c, opacity256, one256,
                                                    comp_avx, clip_output);
                    store_f32x8(output.f32 + elem, _mm256_mul_ps(out_c,
                                                                 _mm256_set1_ps(255.0f)),
                                out_aligned);
                }
            } else if (background.is_uint8 && !foreground.is_uint8) {
                int bg_aligned = ptr_is_aligned(background.u8, 16);
                int fg_aligned = ptr_is_aligned(foreground.f32, 32);
                int out_aligned = ptr_is_aligned(output.u8, 16);
                for (; elem + 15 < elements; elem += 16) {
                    __m128i in_bytes = bg_aligned ?
                        _mm_load_si128((const __m128i *)(background.u8 + elem)) :
                        _mm_loadu_si128((const __m128i *)(background.u8 + elem));
                    __m256i in_lo_u = _mm256_cvtepu8_epi32(in_bytes);
                    __m256i in_hi_u = _mm256_cvtepu8_epi32(_mm_srli_si128(in_bytes, 8));
                    __m256 in_lo = u32_to_unit_f32_avx2(in_lo_u, inv255256);
                    __m256 in_hi = u32_to_unit_f32_avx2(in_hi_u, inv255256);
                    __m256 layer_lo = _mm256_mul_ps(load_f32x8(foreground.f32 + elem, fg_aligned),
                                                    inv255256);
                    __m256 layer_hi = _mm256_mul_ps(load_f32x8(foreground.f32 + elem + 8,
                                                               fg_aligned),
                                                    inv255256);
                    __m256 out_lo = blend_flat_ps256(in_lo, layer_lo, opacity256, one256,
                                                     comp_avx, clip_output);
                    __m256 out_hi = blend_flat_ps256(in_hi, layer_hi, opacity256, one256,
                                                     comp_avx, clip_output);
                    __m128i packed = pack_two_unit_f32_avx2_to_u8(out_lo, out_hi);
                    if (out_aligned) {
                        _mm_store_si128((__m128i *)(output.u8 + elem), packed);
                    } else {
                        _mm_storeu_si128((__m128i *)(output.u8 + elem), packed);
                    }
                }
            } else if (!background.is_uint8 && foreground.is_uint8) {
                int bg_aligned = ptr_is_aligned(background.f32, 32);
                int fg_aligned = ptr_is_aligned(foreground.u8, 16);
                int out_aligned = ptr_is_aligned(output.f32, 32);
                for (; elem + 15 < elements; elem += 16) {
                    __m128i layer_bytes = fg_aligned ?
                        _mm_load_si128((const __m128i *)(foreground.u8 + elem)) :
                        _mm_loadu_si128((const __m128i *)(foreground.u8 + elem));
                    __m256i layer_lo_u = _mm256_cvtepu8_epi32(layer_bytes);
                    __m256i layer_hi_u = _mm256_cvtepu8_epi32(_mm_srli_si128(layer_bytes, 8));
                    __m256 in_lo = _mm256_mul_ps(load_f32x8(background.f32 + elem, bg_aligned),
                                                 inv255256);
                    __m256 in_hi = _mm256_mul_ps(load_f32x8(background.f32 + elem + 8,
                                                            bg_aligned),
                                                 inv255256);
                    __m256 layer_lo = u32_to_unit_f32_avx2(layer_lo_u, inv255256);
                    __m256 layer_hi = u32_to_unit_f32_avx2(layer_hi_u, inv255256);
                    __m256 out_lo = blend_flat_ps256(in_lo, layer_lo, opacity256, one256,
                                                     comp_avx, clip_output);
                    __m256 out_hi = blend_flat_ps256(in_hi, layer_hi, opacity256, one256,
                                                     comp_avx, clip_output);
                    store_f32x8(output.f32 + elem, _mm256_mul_ps(out_lo,
                                                                 _mm256_set1_ps(255.0f)),
                                out_aligned);
                    store_f32x8(output.f32 + elem + 8, _mm256_mul_ps(out_hi,
                                                                     _mm256_set1_ps(255.0f)),
                                out_aligned);
                }
            }
        }
#endif

#if defined(__SSE4_1__)
        if (elem == 0 && (kernel == KERNEL_SSE42 || kernel == KERNEL_AVX2)) {
            if (background.is_uint8 && foreground.is_uint8 &&
                    ptr_is_aligned(background.u8, 16) &&
                    ptr_is_aligned(foreground.u8, 16) &&
                    ptr_is_aligned(output.u8, 16)) {
                for (; elem + 15 < elements; elem += 16) {
                    __m128i in_bytes = _mm_load_si128((const __m128i *)(background.u8 + elem));
                    __m128i layer_bytes = _mm_load_si128((const __m128i *)(foreground.u8 + elem));
                    __m128i in_u0 = _mm_cvtepu8_epi32(in_bytes);
                    __m128i layer_u0 = _mm_cvtepu8_epi32(layer_bytes);
                    __m128i in_u1 = _mm_cvtepu8_epi32(_mm_srli_si128(in_bytes, 4));
                    __m128i layer_u1 = _mm_cvtepu8_epi32(_mm_srli_si128(layer_bytes, 4));
                    __m128i in_u2 = _mm_cvtepu8_epi32(_mm_srli_si128(in_bytes, 8));
                    __m128i layer_u2 = _mm_cvtepu8_epi32(_mm_srli_si128(layer_bytes, 8));
                    __m128i in_u3 = _mm_cvtepu8_epi32(_mm_srli_si128(in_bytes, 12));
                    __m128i layer_u3 = _mm_cvtepu8_epi32(_mm_srli_si128(layer_bytes, 12));

                    __m128 in_c0 = u32_to_unit_f32_sse(in_u0, inv255128);
                    __m128 in_c1 = u32_to_unit_f32_sse(in_u1, inv255128);
                    __m128 in_c2 = u32_to_unit_f32_sse(in_u2, inv255128);
                    __m128 in_c3 = u32_to_unit_f32_sse(in_u3, inv255128);
                    __m128 comp0 = comp_sse_u8 ? comp_sse_u8(in_u0, layer_u0, inv255128) :
                                   comp_sse(in_c0, u32_to_unit_f32_sse(layer_u0, inv255128));
                    __m128 comp1 = comp_sse_u8 ? comp_sse_u8(in_u1, layer_u1, inv255128) :
                                   comp_sse(in_c1, u32_to_unit_f32_sse(layer_u1, inv255128));
                    __m128 comp2 = comp_sse_u8 ? comp_sse_u8(in_u2, layer_u2, inv255128) :
                                   comp_sse(in_c2, u32_to_unit_f32_sse(layer_u2, inv255128));
                    __m128 comp3 = comp_sse_u8 ? comp_sse_u8(in_u3, layer_u3, inv255128) :
                                   comp_sse(in_c3, u32_to_unit_f32_sse(layer_u3, inv255128));
                    __m128 out_c0 = mul_add_ps128(comp0, opacity128,
                                                  _mm_mul_ps(in_c0,
                                                             _mm_sub_ps(one, opacity128)));
                    __m128 out_c1 = mul_add_ps128(comp1, opacity128,
                                                  _mm_mul_ps(in_c1,
                                                             _mm_sub_ps(one, opacity128)));
                    __m128 out_c2 = mul_add_ps128(comp2, opacity128,
                                                  _mm_mul_ps(in_c2,
                                                             _mm_sub_ps(one, opacity128)));
                    __m128 out_c3 = mul_add_ps128(comp3, opacity128,
                                                  _mm_mul_ps(in_c3,
                                                             _mm_sub_ps(one, opacity128)));
                    if (clip_output) {
                        out_c0 = clamp01_ps128(out_c0);
                        out_c1 = clamp01_ps128(out_c1);
                        out_c2 = clamp01_ps128(out_c2);
                        out_c3 = clamp01_ps128(out_c3);
                    }
                    uint32_t packed0 = (uint32_t)_mm_cvtsi128_si32(
                        pack_unit_f32_sse_to_u8(out_c0));
                    uint32_t packed1 = (uint32_t)_mm_cvtsi128_si32(
                        pack_unit_f32_sse_to_u8(out_c1));
                    uint32_t packed2 = (uint32_t)_mm_cvtsi128_si32(
                        pack_unit_f32_sse_to_u8(out_c2));
                    uint32_t packed3 = (uint32_t)_mm_cvtsi128_si32(
                        pack_unit_f32_sse_to_u8(out_c3));
                    memcpy(output.u8 + elem, &packed0, sizeof(packed0));
                    memcpy(output.u8 + elem + 4, &packed1, sizeof(packed1));
                    memcpy(output.u8 + elem + 8, &packed2, sizeof(packed2));
                    memcpy(output.u8 + elem + 12, &packed3, sizeof(packed3));
                }
            } else if (!background.is_uint8 && !foreground.is_uint8) {
                int bg_aligned = ptr_is_aligned(background.f32, 16);
                int fg_aligned = ptr_is_aligned(foreground.f32, 16);
                int out_aligned = ptr_is_aligned(output.f32, 16);
                for (; elem + 3 < elements; elem += 4) {
                    __m128 in_c = _mm_mul_ps(load_f32x4(background.f32 + elem, bg_aligned),
                                             inv255128);
                    __m128 layer_c = _mm_mul_ps(load_f32x4(foreground.f32 + elem, fg_aligned),
                                                inv255128);
                    __m128 out_c = blend_flat_ps128(in_c, layer_c, opacity128, one,
                                                    comp_sse, clip_output);
                    store_f32x4(output.f32 + elem, _mm_mul_ps(out_c, _mm_set1_ps(255.0f)),
                                out_aligned);
                }
            }
        }
#endif

        if (elem > 0) {
            if (background.is_uint8) {
                const uint8_t *bg = background.u8;
                uint8_t *out = output.u8;
                if (foreground.is_uint8) {
                    const uint8_t *fg = foreground.u8;
                    for (; elem < elements; ++elem) {
                        float in_c = read_channel_u8(bg, elem);
                        float layer_c = read_channel_u8(fg, elem);
                        float out_c = comp_scalar(in_c, layer_c) * opacity +
                                      in_c * (1.0f - opacity);
                        if (clip_output) {
                            out_c = clamp01(out_c);
                        }
                        write_channel_u8(out, elem, out_c);
                    }
                } else {
                    const float *fg = foreground.f32;
                    for (; elem < elements; ++elem) {
                        float in_c = read_channel_u8(bg, elem);
                        float layer_c = read_channel_f32(fg, elem);
                        float out_c = comp_scalar(in_c, layer_c) * opacity +
                                      in_c * (1.0f - opacity);
                        if (clip_output) {
                            out_c = clamp01(out_c);
                        }
                        write_channel_u8(out, elem, out_c);
                    }
                }
            } else {
                const float *bg = background.f32;
                float *out = output.f32;
                if (foreground.is_uint8) {
                    const uint8_t *fg = foreground.u8;
                    for (; elem < elements; ++elem) {
                        float in_c = read_channel_f32(bg, elem);
                        float layer_c = read_channel_u8(fg, elem);
                        float out_c = comp_scalar(in_c, layer_c) * opacity +
                                      in_c * (1.0f - opacity);
                        if (clip_output) {
                            out_c = clamp01(out_c);
                        }
                        write_channel_f32(out, elem, out_c);
                    }
                } else {
                    const float *fg = foreground.f32;
                    for (; elem < elements; ++elem) {
                        float in_c = read_channel_f32(bg, elem);
                        float layer_c = read_channel_f32(fg, elem);
                        float out_c = comp_scalar(in_c, layer_c) * opacity +
                                      in_c * (1.0f - opacity);
                        if (clip_output) {
                            out_c = clamp01(out_c);
                        }
                        write_channel_f32(out, elem, out_c);
                    }
                }
            }
            index = pixels;
        }
    }

#if defined(__AVX2__)
    if (index < pixels && kernel == KERNEL_AVX2) {
        const npy_intp prefetch_distance = 16;
        if (background.is_uint8) {
            if (foreground.is_uint8) {
                if (comp_avx_u8) {
                    for (; index + 7 < pixels; index += 8) {
                        npy_intp prefetch_index = index + prefetch_distance;
                        _mm_prefetch((const char *)(background.u8 + (prefetch_index * background.channels)),
                                     _MM_HINT_T0);
                        _mm_prefetch((const char *)(foreground.u8 + (prefetch_index * foreground.channels)),
                                     _MM_HINT_T0);

                        __m256i in_r_u, in_g_u, in_b_u, in_a_u;
                        __m256i layer_r_u, layer_g_u, layer_b_u, layer_a_u;
                        __m256 in_r, in_g, in_b, in_a;
                        __m256 layer_a;

                        if (background.channels == 4) {
                            load_rgba8_u8_to_u32_avx2(
                                background.u8 + (index * background.channels),
                                &in_r_u,
                                &in_g_u,
                                &in_b_u,
                                &in_a_u
                            );
                            in_a = u32_to_unit_f32_avx2(in_a_u, inv255256);
                        } else {
                            load_rgb8_u8_to_u32_avx2_with_alpha(background.u8, index,
                                                                &in_r_u, &in_g_u, &in_b_u,
                                                                &in_a_u);
                            in_a = u32_to_unit_f32_avx2(in_a_u, inv255256);
                        }
                        in_r = u32_to_unit_f32_avx2(in_r_u, inv255256);
                        in_g = u32_to_unit_f32_avx2(in_g_u, inv255256);
                        in_b = u32_to_unit_f32_avx2(in_b_u, inv255256);

                        if (foreground.channels == 4) {
                            load_rgba8_u8_to_u32_avx2(
                                foreground.u8 + (index * foreground.channels),
                                &layer_r_u,
                                &layer_g_u,
                                &layer_b_u,
                                &layer_a_u
                            );
                            layer_a = u32_to_unit_f32_avx2(layer_a_u, inv255256);
                        } else {
                            load_rgb8_u8_to_u32_avx2_with_alpha(foreground.u8, index,
                                                                &layer_r_u, &layer_g_u,
                                                                &layer_b_u, &layer_a_u);
                            layer_a = u32_to_unit_f32_avx2(layer_a_u, inv255256);
                        }

                        __m256 comp_alpha = _mm256_mul_ps(_mm256_min_ps(in_a, layer_a), opacity256);
                        __m256 new_alpha = mul_add_ps256(_mm256_sub_ps(one256, in_a), comp_alpha, in_a);
                        __m256 ratio = _mm256_div_ps(comp_alpha, new_alpha);
                        __m256 mask = _mm256_cmp_ps(new_alpha, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
                        ratio = _mm256_blendv_ps(_mm256_set1_ps(0.0f), ratio, mask);

                        __m256 comp_r = comp_avx_u8(in_r_u, layer_r_u, inv255256);
                        __m256 comp_g = comp_avx_u8(in_g_u, layer_g_u, inv255256);
                        __m256 comp_b = comp_avx_u8(in_b_u, layer_b_u, inv255256);

                        __m256 out_r = mul_add_ps256(comp_r, ratio,
                                                     _mm256_mul_ps(in_r, _mm256_sub_ps(one256, ratio)));
                        __m256 out_g = mul_add_ps256(comp_g, ratio,
                                                     _mm256_mul_ps(in_g, _mm256_sub_ps(one256, ratio)));
                        __m256 out_b = mul_add_ps256(comp_b, ratio,
                                                     _mm256_mul_ps(in_b, _mm256_sub_ps(one256, ratio)));

                        if (clip_output) {
                            out_r = _mm256_min_ps(_mm256_max_ps(out_r, _mm256_set1_ps(0.0f)),
                                                  _mm256_set1_ps(1.0f));
                            out_g = _mm256_min_ps(_mm256_max_ps(out_g, _mm256_set1_ps(0.0f)),
                                                  _mm256_set1_ps(1.0f));
                            out_b = _mm256_min_ps(_mm256_max_ps(out_b, _mm256_set1_ps(0.0f)),
                                                  _mm256_set1_ps(1.0f));
                        }

                        if (output.channels == 4) {
                            store_rgba8_u8(&output, index, out_r, out_g, out_b, in_a);
                        } else {
                            store_rgb8(&output, index, out_r, out_g, out_b);
                            store_alpha8(&output, index, in_a);
                        }
                    }
                } else {
                    for (; index + 7 < pixels; index += 8) {
                        npy_intp prefetch_index = index + prefetch_distance;
                        _mm_prefetch((const char *)(background.u8 + (prefetch_index * background.channels)),
                                     _MM_HINT_T0);
                        _mm_prefetch((const char *)(foreground.u8 + (prefetch_index * foreground.channels)),
                                     _MM_HINT_T0);

                        __m256 in_r, in_g, in_b, in_a;
                        __m256 layer_r, layer_g, layer_b, layer_a;
                        if (background.channels == 4) {
                            load_rgba8_u8_to_unit_f32_avx2(
                                background.u8 + (index * background.channels),
                                inv255256,
                                &in_r,
                                &in_g,
                                &in_b,
                                &in_a
                            );
                        } else {
                            load_rgb8_u8(background.u8, background.channels, index,
                                         &in_r, &in_g, &in_b);
                            in_a = _mm256_set1_ps(1.0f);
                        }

                        if (foreground.channels == 4) {
                            load_rgba8_u8_to_unit_f32_avx2(
                                foreground.u8 + (index * foreground.channels),
                                inv255256,
                                &layer_r,
                                &layer_g,
                                &layer_b,
                                &layer_a
                            );
                        } else {
                            load_rgb8_u8(foreground.u8, foreground.channels, index,
                                         &layer_r, &layer_g, &layer_b);
                            layer_a = _mm256_set1_ps(1.0f);
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
                            out_r = _mm256_min_ps(_mm256_max_ps(out_r, _mm256_set1_ps(0.0f)),
                                                  _mm256_set1_ps(1.0f));
                            out_g = _mm256_min_ps(_mm256_max_ps(out_g, _mm256_set1_ps(0.0f)),
                                                  _mm256_set1_ps(1.0f));
                            out_b = _mm256_min_ps(_mm256_max_ps(out_b, _mm256_set1_ps(0.0f)),
                                                  _mm256_set1_ps(1.0f));
                        }

                        if (output.channels == 4) {
                            store_rgba8_u8(&output, index, out_r, out_g, out_b, in_a);
                        } else {
                            store_rgb8(&output, index, out_r, out_g, out_b);
                            store_alpha8(&output, index, in_a);
                        }
                    }
                }
            } else {
                for (; index + 7 < pixels; index += 8) {
                    npy_intp prefetch_index = index + prefetch_distance;
                    _mm_prefetch((const char *)(background.u8 + (prefetch_index * background.channels)),
                                 _MM_HINT_T0);
                    _mm_prefetch((const char *)(foreground.f32 + (prefetch_index * foreground.channels)),
                                 _MM_HINT_T0);

                    __m256 in_r, in_g, in_b, in_a;
                    __m256 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba8_u8_to_unit_f32_avx2(
                            background.u8 + (index * background.channels),
                            inv255256,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb8_u8(background.u8, background.channels, index, &in_r, &in_g, &in_b);
                        in_a = _mm256_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba8_f32_to_unit_f32_avx2(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb8_f32_to_unit_f32_avx2(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b
                        );
                        layer_a = _mm256_set1_ps(1.0f);
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
                        out_r = _mm256_min_ps(_mm256_max_ps(out_r, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                        out_g = _mm256_min_ps(_mm256_max_ps(out_g, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                        out_b = _mm256_min_ps(_mm256_max_ps(out_b, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                    }

                    if (output.channels == 4) {
                        store_rgba8_u8(&output, index, out_r, out_g, out_b, in_a);
                    } else {
                        store_rgb8(&output, index, out_r, out_g, out_b);
                        store_alpha8(&output, index, in_a);
                    }
                }
            }
        } else {
            if (foreground.is_uint8) {
                for (; index + 7 < pixels; index += 8) {
                    npy_intp prefetch_index = index + prefetch_distance;
                    _mm_prefetch((const char *)(background.f32 + (prefetch_index * background.channels)),
                                 _MM_HINT_T0);
                    _mm_prefetch((const char *)(foreground.u8 + (prefetch_index * foreground.channels)),
                                 _MM_HINT_T0);

                    __m256 in_r, in_g, in_b, in_a;
                    __m256 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba8_f32_to_unit_f32_avx2(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb8_f32_to_unit_f32_avx2(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b
                        );
                        in_a = _mm256_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba8_u8_to_unit_f32_avx2(
                            foreground.u8 + (index * foreground.channels),
                            inv255256,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb8_u8(foreground.u8, foreground.channels, index,
                                     &layer_r, &layer_g, &layer_b);
                        layer_a = _mm256_set1_ps(1.0f);
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
                        out_r = _mm256_min_ps(_mm256_max_ps(out_r, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                        out_g = _mm256_min_ps(_mm256_max_ps(out_g, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                        out_b = _mm256_min_ps(_mm256_max_ps(out_b, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                    }

                    if (output.channels == 4) {
                        store_rgba8_f32_from_unit_avx2(
                            output.f32 + (index * output.channels),
                            out_r,
                            out_g,
                            out_b,
                            in_a
                        );
                    } else {
                        store_rgb8(&output, index, out_r, out_g, out_b);
                        store_alpha8(&output, index, in_a);
                    }
                }
            } else {
                for (; index + 7 < pixels; index += 8) {
                    __m256 in_r, in_g, in_b, in_a;
                    __m256 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba8_f32_to_unit_f32_avx2(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb8_f32_to_unit_f32_avx2(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b
                        );
                        in_a = _mm256_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba8_f32_to_unit_f32_avx2(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb8_f32_to_unit_f32_avx2(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b
                        );
                        layer_a = _mm256_set1_ps(1.0f);
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
                        out_r = _mm256_min_ps(_mm256_max_ps(out_r, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                        out_g = _mm256_min_ps(_mm256_max_ps(out_g, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                        out_b = _mm256_min_ps(_mm256_max_ps(out_b, _mm256_set1_ps(0.0f)),
                                              _mm256_set1_ps(1.0f));
                    }

                    if (output.channels == 4) {
                        store_rgba8_f32_from_unit_avx2(
                            output.f32 + (index * output.channels),
                            out_r,
                            out_g,
                            out_b,
                            in_a
                        );
                    } else {
                        store_rgb8(&output, index, out_r, out_g, out_b);
                        store_alpha8(&output, index, in_a);
                    }
                }
            }
        }
    }
#endif

#if defined(__SSE4_1__)
    if (kernel == KERNEL_SSE42 || kernel == KERNEL_AVX2) {
        if (background.is_uint8) {
            if (foreground.is_uint8) {
                if (comp_sse_u8) {
                    for (; index + 3 < pixels; index += 4) {
                        __m128i in_r_u, in_g_u, in_b_u, in_a_u;
                        __m128i layer_r_u, layer_g_u, layer_b_u, layer_a_u;
                        __m128 in_r, in_g, in_b, in_a;
                        __m128 layer_a;

                        if (background.channels == 4) {
                            load_rgba4_u8_to_u32_sse(
                                background.u8 + (index * background.channels),
                                &in_r_u,
                                &in_g_u,
                                &in_b_u,
                                &in_a_u
                            );
                            in_a = u32_to_unit_f32_sse(in_a_u, inv255128);
                        } else {
                            load_rgb4_u8_to_u32_sse_with_alpha(
                                background.u8 + (index * background.channels),
                                &in_r_u,
                                &in_g_u,
                                &in_b_u,
                                &in_a_u
                            );
                            in_a = u32_to_unit_f32_sse(in_a_u, inv255128);
                        }
                        in_r = u32_to_unit_f32_sse(in_r_u, inv255128);
                        in_g = u32_to_unit_f32_sse(in_g_u, inv255128);
                        in_b = u32_to_unit_f32_sse(in_b_u, inv255128);

                        if (foreground.channels == 4) {
                            load_rgba4_u8_to_u32_sse(
                                foreground.u8 + (index * foreground.channels),
                                &layer_r_u,
                                &layer_g_u,
                                &layer_b_u,
                                &layer_a_u
                            );
                            layer_a = u32_to_unit_f32_sse(layer_a_u, inv255128);
                        } else {
                            load_rgb4_u8_to_u32_sse_with_alpha(
                                foreground.u8 + (index * foreground.channels),
                                &layer_r_u,
                                &layer_g_u,
                                &layer_b_u,
                                &layer_a_u
                            );
                            layer_a = u32_to_unit_f32_sse(layer_a_u, inv255128);
                        }

                        __m128 comp_alpha = _mm_mul_ps(_mm_min_ps(in_a, layer_a), opacity128);
                        __m128 new_alpha = mul_add_ps128(_mm_sub_ps(one, in_a), comp_alpha, in_a);
                        __m128 ratio = _mm_div_ps(comp_alpha, new_alpha);
                        __m128 mask = _mm_cmpgt_ps(new_alpha, _mm_set1_ps(0.0f));
                        ratio = _mm_blendv_ps(_mm_set1_ps(0.0f), ratio, mask);

                        __m128 comp_r = comp_sse_u8(in_r_u, layer_r_u, inv255128);
                        __m128 comp_g = comp_sse_u8(in_g_u, layer_g_u, inv255128);
                        __m128 comp_b = comp_sse_u8(in_b_u, layer_b_u, inv255128);

                        __m128 out_r = mul_add_ps128(comp_r, ratio,
                                                     _mm_mul_ps(in_r, _mm_sub_ps(one, ratio)));
                        __m128 out_g = mul_add_ps128(comp_g, ratio,
                                                     _mm_mul_ps(in_g, _mm_sub_ps(one, ratio)));
                        __m128 out_b = mul_add_ps128(comp_b, ratio,
                                                     _mm_mul_ps(in_b, _mm_sub_ps(one, ratio)));

                        if (clip_output) {
                            out_r = _mm_min_ps(_mm_max_ps(out_r, _mm_set1_ps(0.0f)),
                                               _mm_set1_ps(1.0f));
                            out_g = _mm_min_ps(_mm_max_ps(out_g, _mm_set1_ps(0.0f)),
                                               _mm_set1_ps(1.0f));
                            out_b = _mm_min_ps(_mm_max_ps(out_b, _mm_set1_ps(0.0f)),
                                               _mm_set1_ps(1.0f));
                        }

                        if (output.channels == 4) {
                            store_rgba4_u8(&output, index, out_r, out_g, out_b, in_a);
                        } else {
                            store_rgb4(&output, index, out_r, out_g, out_b);
                            store_alpha4(&output, index, in_a);
                        }
                    }
                } else {
                    for (; index + 3 < pixels; index += 4) {
                        __m128 in_r, in_g, in_b, in_a;
                        __m128 layer_r, layer_g, layer_b, layer_a;
                        if (background.channels == 4) {
                            load_rgba4_u8_to_unit_f32_sse(
                                background.u8 + (index * background.channels),
                                inv255128,
                                &in_r,
                                &in_g,
                                &in_b,
                                &in_a
                            );
                        } else {
                            load_rgb4_u8(background.u8, background.channels, index,
                                         &in_r, &in_g, &in_b);
                            in_a = _mm_set1_ps(1.0f);
                        }

                        if (foreground.channels == 4) {
                            load_rgba4_u8_to_unit_f32_sse(
                                foreground.u8 + (index * foreground.channels),
                                inv255128,
                                &layer_r,
                                &layer_g,
                                &layer_b,
                                &layer_a
                            );
                        } else {
                            load_rgb4_u8(foreground.u8, foreground.channels, index,
                                         &layer_r, &layer_g, &layer_b);
                            layer_a = _mm_set1_ps(1.0f);
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
                            out_r = _mm_min_ps(_mm_max_ps(out_r, _mm_set1_ps(0.0f)),
                                               _mm_set1_ps(1.0f));
                            out_g = _mm_min_ps(_mm_max_ps(out_g, _mm_set1_ps(0.0f)),
                                               _mm_set1_ps(1.0f));
                            out_b = _mm_min_ps(_mm_max_ps(out_b, _mm_set1_ps(0.0f)),
                                               _mm_set1_ps(1.0f));
                        }

                        if (output.channels == 4) {
                            store_rgba4_u8(&output, index, out_r, out_g, out_b, in_a);
                        } else {
                            store_rgb4(&output, index, out_r, out_g, out_b);
                            store_alpha4(&output, index, in_a);
                        }
                    }
                }
            } else {
                for (; index + 3 < pixels; index += 4) {
                    __m128 in_r, in_g, in_b, in_a;
                    __m128 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba4_u8_to_unit_f32_sse(
                            background.u8 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb4_u8(background.u8, background.channels, index,
                                     &in_r, &in_g, &in_b);
                        in_a = _mm_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba4_f32_to_unit_f32_sse(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb4_f32_to_unit_f32_sse(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b
                        );
                        layer_a = _mm_set1_ps(1.0f);
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
                        out_r = _mm_min_ps(_mm_max_ps(out_r, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                        out_g = _mm_min_ps(_mm_max_ps(out_g, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                        out_b = _mm_min_ps(_mm_max_ps(out_b, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                    }

                    if (output.channels == 4) {
                        store_rgba4_u8(&output, index, out_r, out_g, out_b, in_a);
                    } else {
                        store_rgb4(&output, index, out_r, out_g, out_b);
                        store_alpha4(&output, index, in_a);
                    }
                }
            }
        } else {
            if (foreground.is_uint8) {
                for (; index + 3 < pixels; index += 4) {
                    __m128 in_r, in_g, in_b, in_a;
                    __m128 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba4_f32_to_unit_f32_sse(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb4_f32_to_unit_f32_sse(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b
                        );
                        in_a = _mm_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba4_u8_to_unit_f32_sse(
                            foreground.u8 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb4_u8(foreground.u8, foreground.channels, index,
                                     &layer_r, &layer_g, &layer_b);
                        layer_a = _mm_set1_ps(1.0f);
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
                        out_r = _mm_min_ps(_mm_max_ps(out_r, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                        out_g = _mm_min_ps(_mm_max_ps(out_g, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                        out_b = _mm_min_ps(_mm_max_ps(out_b, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                    }

                    if (output.channels == 4) {
                        store_rgba4_f32_from_unit_sse(
                            output.f32 + (index * output.channels),
                            out_r,
                            out_g,
                            out_b,
                            in_a
                        );
                    } else {
                        store_rgb4(&output, index, out_r, out_g, out_b);
                        store_alpha4(&output, index, in_a);
                    }
                }
            } else {
                for (; index + 3 < pixels; index += 4) {
                    __m128 in_r, in_g, in_b, in_a;
                    __m128 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba4_f32_to_unit_f32_sse(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb4_f32_to_unit_f32_sse(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b
                        );
                        in_a = _mm_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba4_f32_to_unit_f32_sse(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb4_f32_to_unit_f32_sse(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b
                        );
                        layer_a = _mm_set1_ps(1.0f);
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
                        out_r = _mm_min_ps(_mm_max_ps(out_r, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                        out_g = _mm_min_ps(_mm_max_ps(out_g, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                        out_b = _mm_min_ps(_mm_max_ps(out_b, _mm_set1_ps(0.0f)),
                                           _mm_set1_ps(1.0f));
                    }

                    if (output.channels == 4) {
                        store_rgba4_f32_from_unit_sse(
                            output.f32 + (index * output.channels),
                            out_r,
                            out_g,
                            out_b,
                            in_a
                        );
                    } else {
                        store_rgb4(&output, index, out_r, out_g, out_b);
                        store_alpha4(&output, index, in_a);
                    }
                }
            }
        }
    }
#endif
#endif

    if (background.is_uint8) {
        const uint8_t *bg = background.u8;
        uint8_t *out = output.u8;
        if (foreground.is_uint8) {
            const uint8_t *fg = foreground.u8;
            for (; index < pixels; ++index) {
                npy_intp bg_offset = index * background.channels;
                npy_intp fg_offset = index * foreground.channels;

                float in_a = 1.0f;
                float layer_a = 1.0f;

                if (background.channels == 4) {
                    in_a = read_channel_u8(bg, bg_offset + 3);
                }
                if (foreground.channels == 4) {
                    layer_a = read_channel_u8(fg, fg_offset + 3);
                }

                float comp_alpha = fminf(in_a, layer_a) * opacity;
                float new_alpha = in_a + (1.0f - in_a) * comp_alpha;
                float ratio = 0.0f;
                if (new_alpha > 0.0f) {
                    ratio = comp_alpha / new_alpha;
                }

                for (int c = 0; c < 3; ++c) {
                    float in_c = read_channel_u8(bg, bg_offset + c);
                    float layer_c = read_channel_u8(fg, fg_offset + c);
                    float comp = comp_scalar(in_c, layer_c);
                    float out_c = comp * ratio + in_c * (1.0f - ratio);
                    if (clip_output) {
                        out_c = clamp01(out_c);
                    }
                    write_channel_u8(out, bg_offset + c, out_c);
                }

                if (background.channels == 4) {
                    write_channel_u8(out, bg_offset + 3, in_a);
                }
            }
        } else {
            const float *fg = foreground.f32;
            for (; index < pixels; ++index) {
                npy_intp bg_offset = index * background.channels;
                npy_intp fg_offset = index * foreground.channels;

                float in_a = 1.0f;
                float layer_a = 1.0f;

                if (background.channels == 4) {
                    in_a = read_channel_u8(bg, bg_offset + 3);
                }
                if (foreground.channels == 4) {
                    layer_a = read_channel_f32(fg, fg_offset + 3);
                }

                float comp_alpha = fminf(in_a, layer_a) * opacity;
                float new_alpha = in_a + (1.0f - in_a) * comp_alpha;
                float ratio = 0.0f;
                if (new_alpha > 0.0f) {
                    ratio = comp_alpha / new_alpha;
                }

                for (int c = 0; c < 3; ++c) {
                    float in_c = read_channel_u8(bg, bg_offset + c);
                    float layer_c = read_channel_f32(fg, fg_offset + c);
                    float comp = comp_scalar(in_c, layer_c);
                    float out_c = comp * ratio + in_c * (1.0f - ratio);
                    if (clip_output) {
                        out_c = clamp01(out_c);
                    }
                    write_channel_u8(out, bg_offset + c, out_c);
                }

                if (background.channels == 4) {
                    write_channel_u8(out, bg_offset + 3, in_a);
                }
            }
        }
    } else {
        const float *bg = background.f32;
        float *out = output.f32;
        if (foreground.is_uint8) {
            const uint8_t *fg = foreground.u8;
            for (; index < pixels; ++index) {
                npy_intp bg_offset = index * background.channels;
                npy_intp fg_offset = index * foreground.channels;

                float in_a = 1.0f;
                float layer_a = 1.0f;

                if (background.channels == 4) {
                    in_a = read_channel_f32(bg, bg_offset + 3);
                }
                if (foreground.channels == 4) {
                    layer_a = read_channel_u8(fg, fg_offset + 3);
                }

                float comp_alpha = fminf(in_a, layer_a) * opacity;
                float new_alpha = in_a + (1.0f - in_a) * comp_alpha;
                float ratio = 0.0f;
                if (new_alpha > 0.0f) {
                    ratio = comp_alpha / new_alpha;
                }

                for (int c = 0; c < 3; ++c) {
                    float in_c = read_channel_f32(bg, bg_offset + c);
                    float layer_c = read_channel_u8(fg, fg_offset + c);
                    float comp = comp_scalar(in_c, layer_c);
                    float out_c = comp * ratio + in_c * (1.0f - ratio);
                    if (clip_output) {
                        out_c = clamp01(out_c);
                    }
                    write_channel_f32(out, bg_offset + c, out_c);
                }

                if (background.channels == 4) {
                    write_channel_f32(out, bg_offset + 3, in_a);
                }
            }
        } else {
            const float *fg = foreground.f32;
            for (; index < pixels; ++index) {
                npy_intp bg_offset = index * background.channels;
                npy_intp fg_offset = index * foreground.channels;

                float in_a = 1.0f;
                float layer_a = 1.0f;

                if (background.channels == 4) {
                    in_a = read_channel_f32(bg, bg_offset + 3);
                }
                if (foreground.channels == 4) {
                    layer_a = read_channel_f32(fg, fg_offset + 3);
                }

                float comp_alpha = fminf(in_a, layer_a) * opacity;
                float new_alpha = in_a + (1.0f - in_a) * comp_alpha;
                float ratio = 0.0f;
                if (new_alpha > 0.0f) {
                    ratio = comp_alpha / new_alpha;
                }

                for (int c = 0; c < 3; ++c) {
                    float in_c = read_channel_f32(bg, bg_offset + c);
                    float layer_c = read_channel_f32(fg, fg_offset + c);
                    float comp = comp_scalar(in_c, layer_c);
                    float out_c = comp * ratio + in_c * (1.0f - ratio);
                    if (clip_output) {
                        out_c = clamp01(out_c);
                    }
                    write_channel_f32(out, bg_offset + c, out_c);
                }

                if (background.channels == 4) {
                    write_channel_f32(out, bg_offset + 3, in_a);
                }
            }
        }
    }
    NPY_END_ALLOW_THREADS
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
        NPY_BEGIN_ALLOW_THREADS
        if (output.is_uint8) {
            uint8_t *out = output.u8;
            if (foreground.is_uint8) {
                const uint8_t *fg = foreground.u8;
                if (output.channels == 3) {
                    memcpy(out, fg, (size_t)pixels * 3);
                } else {
                    for (npy_intp index = 0; index < pixels; ++index) {
                        npy_intp fg_offset = index * foreground.channels;
                        npy_intp out_offset = index * output.channels;
                        out[out_offset + 0] = fg[fg_offset + 0];
                        out[out_offset + 1] = fg[fg_offset + 1];
                        out[out_offset + 2] = fg[fg_offset + 2];
                        out[out_offset + 3] = 255;
                    }
                }
            } else {
                const float *fg = foreground.f32;
                for (npy_intp index = 0; index < pixels; ++index) {
                    npy_intp fg_offset = index * foreground.channels;
                    npy_intp out_offset = index * output.channels;
                    for (int c = 0; c < 3; ++c) {
                        float value = read_channel_f32(fg, fg_offset + c);
                        write_channel_u8(out, out_offset + c, value);
                    }
                    if (output.channels == 4) {
                        write_channel_u8(out, out_offset + 3, 1.0f);
                    }
                }
            }
        } else {
            float *out = output.f32;
            if (foreground.is_uint8) {
                const uint8_t *fg = foreground.u8;
                for (npy_intp index = 0; index < pixels; ++index) {
                    npy_intp fg_offset = index * foreground.channels;
                    npy_intp out_offset = index * output.channels;
                    for (int c = 0; c < 3; ++c) {
                        float value = read_channel_u8(fg, fg_offset + c);
                        write_channel_f32(out, out_offset + c, value);
                    }
                    if (output.channels == 4) {
                        write_channel_f32(out, out_offset + 3, 1.0f);
                    }
                }
            } else {
                const float *fg = foreground.f32;
                if (output.channels == 3) {
                    memcpy(out, fg, (size_t)pixels * 3 * sizeof(float));
                } else {
                    for (npy_intp index = 0; index < pixels; ++index) {
                        npy_intp fg_offset = index * foreground.channels;
                        npy_intp out_offset = index * output.channels;
                        out[out_offset + 0] = fg[fg_offset + 0];
                        out[out_offset + 1] = fg[fg_offset + 1];
                        out[out_offset + 2] = fg[fg_offset + 2];
                        out[out_offset + 3] = 255.0f;
                    }
                }
            }
        }
        NPY_END_ALLOW_THREADS
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
    NPY_BEGIN_ALLOW_THREADS
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
        if (background.is_uint8) {
            if (foreground.is_uint8) {
                if (background.channels == 3 && foreground.channels == 3) {
                    uint8_t *out_ptr = output.u8 + (index * output.channels);
                    int alpha_u8 = (int)lroundf(opacity * 255.0f);
                    if (alpha_u8 < 0) {
                        alpha_u8 = 0;
                    } else if (alpha_u8 > 255) {
                        alpha_u8 = 255;
                    }
                    int inv_alpha_u8 = 255 - alpha_u8;
                    const __m256i alpha256 = _mm256_set1_epi32(alpha_u8);
                    const __m256i inv_alpha256 = _mm256_set1_epi32(inv_alpha_u8);
                    const __m256i rounding = _mm256_set1_epi32(128);

                    for (; index + 7 < pixels; index += 8) {
                        npy_intp prefetch_index = index + prefetch_distance;
                        _mm_prefetch((const char *)(background.u8 + (prefetch_index * 3)),
                                     _MM_HINT_T0);
                        _mm_prefetch((const char *)(foreground.u8 + (prefetch_index * 3)),
                                     _MM_HINT_T0);

                        __m256i in_r_u, in_g_u, in_b_u, in_a_u;
                        __m256i layer_r_u, layer_g_u, layer_b_u, layer_a_u;
                        load_rgb8_u8_to_u32_avx2_with_alpha(background.u8, index,
                                                            &in_r_u, &in_g_u, &in_b_u,
                                                            &in_a_u);
                        load_rgb8_u8_to_u32_avx2_with_alpha(foreground.u8, index,
                                                            &layer_r_u, &layer_g_u, &layer_b_u,
                                                            &layer_a_u);

                        __m256i sum_r = _mm256_add_epi32(_mm256_mullo_epi32(in_r_u, inv_alpha256),
                                                         _mm256_mullo_epi32(layer_r_u, alpha256));
                        __m256i sum_g = _mm256_add_epi32(_mm256_mullo_epi32(in_g_u, inv_alpha256),
                                                         _mm256_mullo_epi32(layer_g_u, alpha256));
                        __m256i sum_b = _mm256_add_epi32(_mm256_mullo_epi32(in_b_u, inv_alpha256),
                                                         _mm256_mullo_epi32(layer_b_u, alpha256));

                        sum_r = _mm256_add_epi32(sum_r, rounding);
                        sum_g = _mm256_add_epi32(sum_g, rounding);
                        sum_b = _mm256_add_epi32(sum_b, rounding);
                        sum_r = _mm256_add_epi32(sum_r, _mm256_srli_epi32(sum_r, 8));
                        sum_g = _mm256_add_epi32(sum_g, _mm256_srli_epi32(sum_g, 8));
                        sum_b = _mm256_add_epi32(sum_b, _mm256_srli_epi32(sum_b, 8));
                        __m256i out_r_u = _mm256_srli_epi32(sum_r, 8);
                        __m256i out_g_u = _mm256_srli_epi32(sum_g, 8);
                        __m256i out_b_u = _mm256_srli_epi32(sum_b, 8);

                        __m128i r0 = _mm256_castsi256_si128(out_r_u);
                        __m128i r1 = _mm256_extracti128_si256(out_r_u, 1);
                        __m128i g0 = _mm256_castsi256_si128(out_g_u);
                        __m128i g1 = _mm256_extracti128_si256(out_g_u, 1);
                        __m128i b0 = _mm256_castsi256_si128(out_b_u);
                        __m128i b1 = _mm256_extracti128_si256(out_b_u, 1);

                        store_u32_to_u8_rgb4(r0, g0, b0, out_ptr);
                        store_u32_to_u8_rgb4(r1, g1, b1, out_ptr + 12);

                        out_ptr += 24;
                    }
                } else {
                for (; index + 7 < pixels; index += 8) {
                    npy_intp prefetch_index = index + prefetch_distance;
                    _mm_prefetch((const char *)(background.u8 + (prefetch_index * background.channels)),
                                 _MM_HINT_T0);
                    _mm_prefetch((const char *)(foreground.u8 + (prefetch_index * foreground.channels)),
                                 _MM_HINT_T0);

                    __m256i in_r_u, in_g_u, in_b_u, in_a_u;
                    __m256i layer_r_u, layer_g_u, layer_b_u, layer_a_u;
                    __m256 in_r, in_g, in_b, in_a;
                    __m256 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba8_u8_to_u32_avx2(
                            background.u8 + (index * background.channels),
                            &in_r_u,
                            &in_g_u,
                            &in_b_u,
                            &in_a_u
                        );
                        in_a = u32_to_unit_f32_avx2(in_a_u, inv255256);
                    } else {
                        load_rgb8_u8_to_u32_avx2_with_alpha(
                            background.u8,
                            index,
                            &in_r_u,
                            &in_g_u,
                            &in_b_u,
                            &in_a_u
                        );
                        in_a = u32_to_unit_f32_avx2(in_a_u, inv255256);
                    }
                    in_r = u32_to_unit_f32_avx2(in_r_u, inv255256);
                    in_g = u32_to_unit_f32_avx2(in_g_u, inv255256);
                    in_b = u32_to_unit_f32_avx2(in_b_u, inv255256);

                    if (foreground.channels == 4) {
                        load_rgba8_u8_to_u32_avx2(
                            foreground.u8 + (index * foreground.channels),
                            &layer_r_u,
                            &layer_g_u,
                            &layer_b_u,
                            &layer_a_u
                        );
                        layer_a = u32_to_unit_f32_avx2(layer_a_u, inv255256);
                    } else {
                        load_rgb8_u8_to_u32_avx2_with_alpha(
                            foreground.u8,
                            index,
                            &layer_r_u,
                            &layer_g_u,
                            &layer_b_u,
                            &layer_a_u
                        );
                        layer_a = u32_to_unit_f32_avx2(layer_a_u, inv255256);
                    }
                    layer_r = u32_to_unit_f32_avx2(layer_r_u, inv255256);
                    layer_g = u32_to_unit_f32_avx2(layer_g_u, inv255256);
                    layer_b = u32_to_unit_f32_avx2(layer_b_u, inv255256);
                    __m256 layer_opacity = _mm256_mul_ps(layer_a, opacity256);

                    __m256 denom = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity),
                                                 layer_opacity);
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

                    __m256 out_a = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity),
                                                 layer_opacity);

                    if (output.channels == 4) {
                        store_rgba8_u8(&output, index, out_r, out_g, out_b, out_a);
                    } else {
                        store_rgb8(&output, index, out_r, out_g, out_b);
                        store_alpha8(&output, index, out_a);
                    }
                }
                }
            } else {
                for (; index + 7 < pixels; index += 8) {
                    npy_intp prefetch_index = index + prefetch_distance;
                    _mm_prefetch((const char *)(background.u8 + (prefetch_index * background.channels)),
                                 _MM_HINT_T0);
                    _mm_prefetch((const char *)(foreground.f32 + (prefetch_index * foreground.channels)),
                                 _MM_HINT_T0);

                    __m256 in_r, in_g, in_b, in_a;
                    __m256 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba8_u8_to_unit_f32_avx2(
                            background.u8 + (index * background.channels),
                            inv255256,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb8_u8(background.u8, background.channels, index, &in_r, &in_g, &in_b);
                        in_a = _mm256_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba8_f32_to_unit_f32_avx2(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb8_f32_to_unit_f32_avx2(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b
                        );
                        layer_a = _mm256_set1_ps(1.0f);
                    }
                    __m256 layer_opacity = _mm256_mul_ps(layer_a, opacity256);

                    __m256 denom = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity),
                                                 layer_opacity);
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

                    __m256 out_a = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity),
                                                 layer_opacity);

                    if (output.channels == 4) {
                        store_rgba8_u8(&output, index, out_r, out_g, out_b, out_a);
                    } else {
                        store_rgb8(&output, index, out_r, out_g, out_b);
                        store_alpha8(&output, index, out_a);
                    }
                }
            }
        } else {
            if (foreground.is_uint8) {
                for (; index + 7 < pixels; index += 8) {
                    npy_intp prefetch_index = index + prefetch_distance;
                    _mm_prefetch((const char *)(background.f32 + (prefetch_index * background.channels)),
                                 _MM_HINT_T0);
                    _mm_prefetch((const char *)(foreground.u8 + (prefetch_index * foreground.channels)),
                                 _MM_HINT_T0);

                    __m256 in_r, in_g, in_b, in_a;
                    __m256 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba8_f32_to_unit_f32_avx2(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb8_f32_to_unit_f32_avx2(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b
                        );
                        in_a = _mm256_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba8_u8_to_unit_f32_avx2(
                            foreground.u8 + (index * foreground.channels),
                            inv255256,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb8_u8(foreground.u8, foreground.channels, index,
                                     &layer_r, &layer_g, &layer_b);
                        layer_a = _mm256_set1_ps(1.0f);
                    }
                    __m256 layer_opacity = _mm256_mul_ps(layer_a, opacity256);

                    __m256 denom = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity),
                                                 layer_opacity);
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

                    __m256 out_a = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity),
                                                 layer_opacity);

                    if (output.channels == 4) {
                        store_rgba8_f32_from_unit_avx2(
                            output.f32 + (index * output.channels),
                            out_r,
                            out_g,
                            out_b,
                            out_a
                        );
                    } else {
                        store_rgb8(&output, index, out_r, out_g, out_b);
                        store_alpha8(&output, index, out_a);
                    }
                }
            } else {
                for (; index + 7 < pixels; index += 8) {
                    npy_intp prefetch_index = index + prefetch_distance;
                    _mm_prefetch((const char *)(background.f32 + (prefetch_index * background.channels)),
                                 _MM_HINT_T0);
                    _mm_prefetch((const char *)(foreground.f32 + (prefetch_index * foreground.channels)),
                                 _MM_HINT_T0);

                    __m256 in_r, in_g, in_b, in_a;
                    __m256 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba8_f32_to_unit_f32_avx2(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb8_f32_to_unit_f32_avx2(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b
                        );
                        in_a = _mm256_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba8_f32_to_unit_f32_avx2(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb8_f32_to_unit_f32_avx2(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b
                        );
                        layer_a = _mm256_set1_ps(1.0f);
                    }
                    __m256 layer_opacity = _mm256_mul_ps(layer_a, opacity256);

                    __m256 denom = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity),
                                                 layer_opacity);
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

                    __m256 out_a = mul_add_ps256(in_a, _mm256_sub_ps(one256, layer_opacity),
                                                 layer_opacity);

                    if (output.channels == 4) {
                        store_rgba8_f32_from_unit_avx2(
                            output.f32 + (index * output.channels),
                            out_r,
                            out_g,
                            out_b,
                            out_a
                        );
                    } else {
                        store_rgb8(&output, index, out_r, out_g, out_b);
                        store_alpha8(&output, index, out_a);
                    }
                }
            }
        }
    }
#endif

#if defined(__SSE4_1__)
    if (kernel == KERNEL_SSE42 || kernel == KERNEL_AVX2) {
        if (background.is_uint8) {
            if (foreground.is_uint8) {
                if (background.channels == 3 && foreground.channels == 3) {
                    uint8_t *out_ptr = output.u8 + (index * output.channels);
                    int alpha_u8 = (int)lroundf(opacity * 255.0f);
                    if (alpha_u8 < 0) {
                        alpha_u8 = 0;
                    } else if (alpha_u8 > 255) {
                        alpha_u8 = 255;
                    }
                    int inv_alpha_u8 = 255 - alpha_u8;
                    const __m128i alpha128 = _mm_set1_epi32(alpha_u8);
                    const __m128i inv_alpha128 = _mm_set1_epi32(inv_alpha_u8);
                    const __m128i rounding128 = _mm_set1_epi32(128);

                    for (; index + 3 < pixels; index += 4) {
                        __m128i in_r_u, in_g_u, in_b_u, in_a_u;
                        __m128i layer_r_u, layer_g_u, layer_b_u, layer_a_u;
                        load_rgb4_u8_to_u32_sse_with_alpha(
                            background.u8 + (index * 3),
                            &in_r_u,
                            &in_g_u,
                            &in_b_u,
                            &in_a_u
                        );
                        load_rgb4_u8_to_u32_sse_with_alpha(
                            foreground.u8 + (index * 3),
                            &layer_r_u,
                            &layer_g_u,
                            &layer_b_u,
                            &layer_a_u
                        );

                        __m128i sum_r = _mm_add_epi32(_mm_mullo_epi32(in_r_u, inv_alpha128),
                                                      _mm_mullo_epi32(layer_r_u, alpha128));
                        __m128i sum_g = _mm_add_epi32(_mm_mullo_epi32(in_g_u, inv_alpha128),
                                                      _mm_mullo_epi32(layer_g_u, alpha128));
                        __m128i sum_b = _mm_add_epi32(_mm_mullo_epi32(in_b_u, inv_alpha128),
                                                      _mm_mullo_epi32(layer_b_u, alpha128));

                        sum_r = _mm_add_epi32(sum_r, rounding128);
                        sum_g = _mm_add_epi32(sum_g, rounding128);
                        sum_b = _mm_add_epi32(sum_b, rounding128);
                        sum_r = _mm_add_epi32(sum_r, _mm_srli_epi32(sum_r, 8));
                        sum_g = _mm_add_epi32(sum_g, _mm_srli_epi32(sum_g, 8));
                        sum_b = _mm_add_epi32(sum_b, _mm_srli_epi32(sum_b, 8));
                        __m128i out_r_u = _mm_srli_epi32(sum_r, 8);
                        __m128i out_g_u = _mm_srli_epi32(sum_g, 8);
                        __m128i out_b_u = _mm_srli_epi32(sum_b, 8);

                        store_u32_to_u8_rgb4(out_r_u, out_g_u, out_b_u, out_ptr);
                        out_ptr += 12;
                    }
                } else {
                for (; index + 3 < pixels; index += 4) {
                    __m128i in_r_u, in_g_u, in_b_u, in_a_u;
                    __m128i layer_r_u, layer_g_u, layer_b_u, layer_a_u;
                    __m128 in_r, in_g, in_b, in_a;
                    __m128 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba4_u8_to_u32_sse(
                            background.u8 + (index * background.channels),
                            &in_r_u,
                            &in_g_u,
                            &in_b_u,
                            &in_a_u
                        );
                        in_a = u32_to_unit_f32_sse(in_a_u, inv255128);
                    } else {
                        load_rgb4_u8_to_u32_sse_with_alpha(
                            background.u8 + (index * background.channels),
                            &in_r_u,
                            &in_g_u,
                            &in_b_u,
                            &in_a_u
                        );
                        in_a = u32_to_unit_f32_sse(in_a_u, inv255128);
                    }
                    in_r = u32_to_unit_f32_sse(in_r_u, inv255128);
                    in_g = u32_to_unit_f32_sse(in_g_u, inv255128);
                    in_b = u32_to_unit_f32_sse(in_b_u, inv255128);

                    if (foreground.channels == 4) {
                        load_rgba4_u8_to_u32_sse(
                            foreground.u8 + (index * foreground.channels),
                            &layer_r_u,
                            &layer_g_u,
                            &layer_b_u,
                            &layer_a_u
                        );
                        layer_a = u32_to_unit_f32_sse(layer_a_u, inv255128);
                    } else {
                        load_rgb4_u8_to_u32_sse_with_alpha(
                            foreground.u8 + (index * foreground.channels),
                            &layer_r_u,
                            &layer_g_u,
                            &layer_b_u,
                            &layer_a_u
                        );
                        layer_a = u32_to_unit_f32_sse(layer_a_u, inv255128);
                    }
                    layer_r = u32_to_unit_f32_sse(layer_r_u, inv255128);
                    layer_g = u32_to_unit_f32_sse(layer_g_u, inv255128);
                    layer_b = u32_to_unit_f32_sse(layer_b_u, inv255128);
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

                    if (output.channels == 4) {
                        store_rgba4_u8(&output, index, out_r, out_g, out_b, out_a);
                    } else {
                        store_rgb4(&output, index, out_r, out_g, out_b);
                        store_alpha4(&output, index, out_a);
                    }
                }
                }
            } else {
                for (; index + 3 < pixels; index += 4) {
                    __m128 in_r, in_g, in_b, in_a;
                    __m128 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba4_u8_to_unit_f32_sse(
                            background.u8 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb4_u8(background.u8, background.channels, index,
                                     &in_r, &in_g, &in_b);
                        in_a = _mm_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba4_f32_to_unit_f32_sse(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb4_f32_to_unit_f32_sse(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b
                        );
                        layer_a = _mm_set1_ps(1.0f);
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

                    if (output.channels == 4) {
                        store_rgba4_u8(&output, index, out_r, out_g, out_b, out_a);
                    } else {
                        store_rgb4(&output, index, out_r, out_g, out_b);
                        store_alpha4(&output, index, out_a);
                    }
                }
            }
        } else {
            if (foreground.is_uint8) {
                for (; index + 3 < pixels; index += 4) {
                    __m128 in_r, in_g, in_b, in_a;
                    __m128 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba4_f32_to_unit_f32_sse(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb4_f32_to_unit_f32_sse(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b
                        );
                        in_a = _mm_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba4_u8_to_unit_f32_sse(
                            foreground.u8 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb4_u8(foreground.u8, foreground.channels, index,
                                     &layer_r, &layer_g, &layer_b);
                        layer_a = _mm_set1_ps(1.0f);
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

                    if (output.channels == 4) {
                        store_rgba4_f32_from_unit_sse(
                            output.f32 + (index * output.channels),
                            out_r,
                            out_g,
                            out_b,
                            out_a
                        );
                    } else {
                        store_rgb4(&output, index, out_r, out_g, out_b);
                        store_alpha4(&output, index, out_a);
                    }
                }
            } else {
                for (; index + 3 < pixels; index += 4) {
                    __m128 in_r, in_g, in_b, in_a;
                    __m128 layer_r, layer_g, layer_b, layer_a;
                    if (background.channels == 4) {
                        load_rgba4_f32_to_unit_f32_sse(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b,
                            &in_a
                        );
                    } else {
                        load_rgb4_f32_to_unit_f32_sse(
                            background.f32 + (index * background.channels),
                            inv255128,
                            &in_r,
                            &in_g,
                            &in_b
                        );
                        in_a = _mm_set1_ps(1.0f);
                    }

                    if (foreground.channels == 4) {
                        load_rgba4_f32_to_unit_f32_sse(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b,
                            &layer_a
                        );
                    } else {
                        load_rgb4_f32_to_unit_f32_sse(
                            foreground.f32 + (index * foreground.channels),
                            inv255128,
                            &layer_r,
                            &layer_g,
                            &layer_b
                        );
                        layer_a = _mm_set1_ps(1.0f);
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

                    if (output.channels == 4) {
                        store_rgba4_f32_from_unit_sse(
                            output.f32 + (index * output.channels),
                            out_r,
                            out_g,
                            out_b,
                            out_a
                        );
                    } else {
                        store_rgb4(&output, index, out_r, out_g, out_b);
                        store_alpha4(&output, index, out_a);
                    }
                }
            }
        }
    }
#endif
#endif

    if (background.is_uint8) {
        const uint8_t *bg = background.u8;
        uint8_t *out = output.u8;
        if (foreground.is_uint8) {
            const uint8_t *fg = foreground.u8;
            for (; index < pixels; ++index) {
                npy_intp bg_offset = index * background.channels;
                npy_intp fg_offset = index * foreground.channels;

                float in_a = 1.0f;
                float layer_a = 1.0f;

                if (background.channels == 4) {
                    in_a = read_channel_u8(bg, bg_offset + 3);
                }
                if (foreground.channels == 4) {
                    layer_a = read_channel_u8(fg, fg_offset + 3);
                }

                float layer_opacity = layer_a * opacity;
                float denom = layer_opacity + in_a * (1.0f - layer_opacity);
                for (int c = 0; c < 3; ++c) {
                    float in_c = read_channel_u8(bg, bg_offset + c);
                    float layer_c = read_channel_u8(fg, fg_offset + c);
                    float out_c = 0.0f;
                    if (denom > 0.0f) {
                        out_c = (layer_c * layer_opacity +
                                 in_c * in_a * (1.0f - layer_opacity)) / denom;
                    }
                    write_channel_u8(out, bg_offset + c, out_c);
                }

                if (background.channels == 4) {
                    float out_a = layer_opacity + in_a * (1.0f - layer_opacity);
                    write_channel_u8(out, bg_offset + 3, out_a);
                }
            }
        } else {
            const float *fg = foreground.f32;
            for (; index < pixels; ++index) {
                npy_intp bg_offset = index * background.channels;
                npy_intp fg_offset = index * foreground.channels;

                float in_a = 1.0f;
                float layer_a = 1.0f;

                if (background.channels == 4) {
                    in_a = read_channel_u8(bg, bg_offset + 3);
                }
                if (foreground.channels == 4) {
                    layer_a = read_channel_f32(fg, fg_offset + 3);
                }

                float layer_opacity = layer_a * opacity;
                float denom = layer_opacity + in_a * (1.0f - layer_opacity);
                for (int c = 0; c < 3; ++c) {
                    float in_c = read_channel_u8(bg, bg_offset + c);
                    float layer_c = read_channel_f32(fg, fg_offset + c);
                    float out_c = 0.0f;
                    if (denom > 0.0f) {
                        out_c = (layer_c * layer_opacity +
                                 in_c * in_a * (1.0f - layer_opacity)) / denom;
                    }
                    write_channel_u8(out, bg_offset + c, out_c);
                }

                if (background.channels == 4) {
                    float out_a = layer_opacity + in_a * (1.0f - layer_opacity);
                    write_channel_u8(out, bg_offset + 3, out_a);
                }
            }
        }
    } else {
        const float *bg = background.f32;
        float *out = output.f32;
        if (foreground.is_uint8) {
            const uint8_t *fg = foreground.u8;
            for (; index < pixels; ++index) {
                npy_intp bg_offset = index * background.channels;
                npy_intp fg_offset = index * foreground.channels;

                float in_a = 1.0f;
                float layer_a = 1.0f;

                if (background.channels == 4) {
                    in_a = read_channel_f32(bg, bg_offset + 3);
                }
                if (foreground.channels == 4) {
                    layer_a = read_channel_u8(fg, fg_offset + 3);
                }

                float layer_opacity = layer_a * opacity;
                float denom = layer_opacity + in_a * (1.0f - layer_opacity);
                for (int c = 0; c < 3; ++c) {
                    float in_c = read_channel_f32(bg, bg_offset + c);
                    float layer_c = read_channel_u8(fg, fg_offset + c);
                    float out_c = 0.0f;
                    if (denom > 0.0f) {
                        out_c = (layer_c * layer_opacity +
                                 in_c * in_a * (1.0f - layer_opacity)) / denom;
                    }
                    write_channel_f32(out, bg_offset + c, out_c);
                }

                if (background.channels == 4) {
                    float out_a = layer_opacity + in_a * (1.0f - layer_opacity);
                    write_channel_f32(out, bg_offset + 3, out_a);
                }
            }
        } else {
            const float *fg = foreground.f32;
            for (; index < pixels; ++index) {
                npy_intp bg_offset = index * background.channels;
                npy_intp fg_offset = index * foreground.channels;

                float in_a = 1.0f;
                float layer_a = 1.0f;

                if (background.channels == 4) {
                    in_a = read_channel_f32(bg, bg_offset + 3);
                }
                if (foreground.channels == 4) {
                    layer_a = read_channel_f32(fg, fg_offset + 3);
                }

                float layer_opacity = layer_a * opacity;
                float denom = layer_opacity + in_a * (1.0f - layer_opacity);
                for (int c = 0; c < 3; ++c) {
                    float in_c = read_channel_f32(bg, bg_offset + c);
                    float layer_c = read_channel_f32(fg, fg_offset + c);
                    float out_c = 0.0f;
                    if (denom > 0.0f) {
                        out_c = (layer_c * layer_opacity +
                                 in_c * in_a * (1.0f - layer_opacity)) / denom;
                    }
                    write_channel_f32(out, bg_offset + c, out_c);
                }

                if (background.channels == 4) {
                    float out_a = layer_opacity + in_a * (1.0f - layer_opacity);
                    write_channel_f32(out, bg_offset + 3, out_a);
                }
            }
        }
    }
    NPY_END_ALLOW_THREADS
    Py_XDECREF(kernel_hold);
    release_blend_inputs(&background, &foreground);
    return (PyObject *)output.array;
}

#endif
