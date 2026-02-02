import platform
from setuptools import Extension, setup
import sys

import numpy


arch = platform.machine().lower()
extra_compile_args = []

if sys.platform == "win32":
    extra_compile_args += ["/O2", "/arch:AVX2", "/Qpar"]  # enables AVX/AVX2; SSE4.2 implied
elif "arm" in arch or "aarch64" in arch:
    # Likely won't compile on ARM, but just in case
    extra_compile_args += ["-O3", "-flto", "-ffp-contract=fast",]
else:
    extra_compile_args += ["-O3", "-march=x86-64", "-mavx2", "-msse4.2", "-flto", "-mfma", "-ffp-contract=fast",]

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
    extra_compile_args=extra_compile_args,
)

setup(ext_modules=[extension])
