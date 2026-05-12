"""SIMD-accelerated blend modes."""

from enum import Enum

from ._simd_blend_modes import (
    addition,
    burn,
    darken_only,
    difference,
    divide,
    dodge,
    exclusion,
    grain_extract,
    grain_merge,
    hard_light,
    hsl_color,
    hsv_hue,
    hsv_saturation,
    hsv_value,
    lch_chroma,
    lch_color,
    lch_hue,
    lch_lightness,
    lighten_only,
    linear_burn,
    multiply,
    normal,
    overlay,
    pin_light,
    screen,
    soft_light,
    subtract,
    vivid_light,
)


class KernelKind(Enum):
    KERNEL_AUTO = "auto"
    KERNEL_SCALAR = "scalar"
    KERNEL_SSE42 = "sse42"
    KERNEL_AVX2 = "avx2"


__all__ = [
    "KernelKind",
    "addition",
    "burn",
    "darken_only",
    "difference",
    "divide",
    "dodge",
    "exclusion",
    "grain_extract",
    "grain_merge",
    "hard_light",
    "hsl_color",
    "hsv_hue",
    "hsv_saturation",
    "hsv_value",
    "lch_chroma",
    "lch_color",
    "lch_hue",
    "lch_lightness",
    "lighten_only",
    "linear_burn",
    "multiply",
    "normal",
    "overlay",
    "pin_light",
    "screen",
    "soft_light",
    "subtract",
    "vivid_light",
]
