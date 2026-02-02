from setuptools import Extension, setup

import numpy

KERNEL_SOURCES = [
    "src/simd_blend_modes/kernels/normal.c",
    "src/simd_blend_modes/kernels/soft_light.c",
    "src/simd_blend_modes/kernels/lighten_only.c",
    "src/simd_blend_modes/kernels/screen.c",
    "src/simd_blend_modes/kernels/dodge.c",
    "src/simd_blend_modes/kernels/addition.c",
    "src/simd_blend_modes/kernels/darken_only.c",
    "src/simd_blend_modes/kernels/multiply.c",
    "src/simd_blend_modes/kernels/hard_light.c",
    "src/simd_blend_modes/kernels/difference.c",
    "src/simd_blend_modes/kernels/subtract.c",
    "src/simd_blend_modes/kernels/grain_extract.c",
    "src/simd_blend_modes/kernels/grain_merge.c",
    "src/simd_blend_modes/kernels/divide.c",
    "src/simd_blend_modes/kernels/overlay.c",
]

extension = Extension(
    "simd_blend_modes._simd_blend_modes",
    sources=["src/simd_blend_modes/_simd_blend_modes.c", *KERNEL_SOURCES],
    include_dirs=[numpy.get_include(), "src/simd_blend_modes/kernels"],
)

setup(ext_modules=[extension])
