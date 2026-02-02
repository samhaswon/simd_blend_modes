"""SIMD-accelerated blend modes."""

from enum import Enum

from ._simd_blend_modes import (
    addition,
    darken_only,
    difference,
    divide,
    dodge,
    grain_extract,
    grain_merge,
    hard_light,
    lighten_only,
    multiply,
    normal,
    overlay,
    screen,
    soft_light,
    subtract,
)


class KernelKind(Enum):
    KERNEL_AUTO = "auto"
    KERNEL_SCALAR = "scalar"
    KERNEL_SSE42 = "sse42"
    KERNEL_AVX2 = "avx2"


__all__ = [
    "KernelKind",
    "addition",
    "darken_only",
    "difference",
    "divide",
    "dodge",
    "grain_extract",
    "grain_merge",
    "hard_light",
    "lighten_only",
    "multiply",
    "normal",
    "overlay",
    "screen",
    "soft_light",
    "subtract",
]
