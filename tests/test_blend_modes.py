import sys
import unittest
from pathlib import Path

import numpy as np

import simd_blend_modes
from additional_blends import additional_blends

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
BLEND_MODES_PATH = ROOT / "blend_modes"
if str(BLEND_MODES_PATH) not in sys.path:
    sys.path.insert(0, str(BLEND_MODES_PATH))

from blend_modes import blending_functions as reference


BLEND_MODE_NAMES = [
    "normal",
    "soft_light",
    "lighten_only",
    "screen",
    "dodge",
    "addition",
    "darken_only",
    "multiply",
    "hard_light",
    "difference",
    "subtract",
    "grain_extract",
    "grain_merge",
    "divide",
    "overlay",
]

ADDITIONAL_BLEND_MODES = {
    "hsv_hue": additional_blends.hsv_hue,
    "hsv_saturation": additional_blends.hsv_saturation,
    "hsv_value": additional_blends.hsv_value,
    "hsl_color": additional_blends.hsl_color,
    "lch_hue": additional_blends.lch_hue,
    "lch_chroma": additional_blends.lch_chroma,
    "lch_color": additional_blends.lch_color,
    "lch_lightness": additional_blends.lch_lightness,
    "burn": additional_blends.burn,
    "linear_burn": additional_blends.linear_burn,
    "exclusion": additional_blends.exclusion,
    "vivid_light": additional_blends.vivid_light,
    "pin_light": additional_blends.pin_light,
}

ADDITIONAL_SIMD_BLEND_MODE_NAMES = list(ADDITIONAL_BLEND_MODES)


WIDE_BACKGROUND_VALUES = np.array(
    [
        [[0, 128, 255], [64, 200, 10], [255, 0, 32]],
        [[120, 180, 240], [15, 253, 143], [237, 104, 229]],
        [[242, 72, 162], [28, 246, 137], [251, 173, 207]],
    ],
    dtype=np.uint8,
)
WIDE_FOREGROUND_VALUES = np.array(
    [
        [[10, 220, 30], [90, 40, 200], [200, 10, 250]],
        [[5, 130, 80], [0, 0, 0], [28, 31, 27]],
        [[208, 204, 211], [255, 252, 249], [76, 77, 72]],
    ],
    dtype=np.uint8,
)
WIDE_ALPHA_VALUES = np.array(
    [
        [[0], [128], [255]],
        [[64], [49], [255]],
        [[255], [2], [1]],
    ],
    dtype=np.uint8,
)


def _with_alpha(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:
        return image
    alpha = np.full(image.shape[:2] + (1,), 255.0, dtype=image.dtype)
    return np.dstack((image, alpha))


def _read_fixture_rgba(name: str) -> np.ndarray:
    image = cv2.imread(str(ROOT / "tests" / name), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(name)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA).astype(np.float32)


def _make_wide_image(channels: int, dtype: np.dtype, base: np.ndarray) -> np.ndarray:
    image = base.astype(dtype)
    if channels == 4:
        image = np.dstack((image, WIDE_ALPHA_VALUES.astype(dtype)))
    return image


class TestBlendModes(unittest.TestCase):
    def setUp(self) -> None:
        self.background_values = np.array(
            [
                [[0, 128, 255], [64, 200, 10]],
                [[255, 0, 32], [120, 180, 240]],
            ],
            dtype=np.uint8,
        )
        self.foreground_values = np.array(
            [
                [[10, 220, 30], [90, 40, 200]],
                [[200, 10, 250], [5, 130, 80]],
            ],
            dtype=np.uint8,
        )
        self.alpha_values = np.array(
            [
                [[0], [128]],
                [[255], [64]],
            ],
            dtype=np.uint8,
        )

    def _make_image(
        self,
        channels: int,
        dtype: np.dtype,
        base: np.ndarray,
    ) -> np.ndarray:
        image = base.astype(dtype)
        if channels == 4:
            alpha = self.alpha_values.astype(dtype)
            image = np.dstack((image, alpha))
        return image

    def _reference_blend(
        self,
        mode_name: str,
        background: np.ndarray,
        foreground: np.ndarray,
        opacity: float,
        background_channels: int,
    ) -> np.ndarray:
        func = getattr(reference, mode_name)
        bg_ref = _with_alpha(background.astype(np.float32))
        fg_ref = _with_alpha(foreground.astype(np.float32))
        blended = func(bg_ref, fg_ref, opacity)
        if background_channels == 3:
            blended = blended[:, :, :3]
        return blended

    def _assert_close(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        context: str = "",
    ) -> None:
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        max_diff = np.max(diff)
        message = f"max diff {max_diff} exceeded tolerance"
        if context:
            message = f"{context}: {message}"
        self.assertLessEqual(
            max_diff,
            1.0,
            msg=message,
        )

    def test_blend_modes(self) -> None:
        opacities = [0.0, 0.5, 1.0]
        dtype_pairs = [
            (np.uint8, np.uint8),
            (np.uint8, np.float32),
            (np.float32, np.uint8),
            (np.float32, np.float32),
        ]
        channel_pairs = [(3, 3), (3, 4), (4, 3), (4, 4)]

        kernels = ["auto", "scalar", "sse42", "avx2"]
        available_kernels = [
            kernel for kernel in kernels
            if simd_blend_modes._simd_blend_modes.kernel_available(kernel)
        ]
        for mode_name in BLEND_MODE_NAMES:
            func = getattr(simd_blend_modes, mode_name)
            for background_dtype, foreground_dtype in dtype_pairs:
                for background_channels, foreground_channels in channel_pairs:
                    background = self._make_image(
                        background_channels,
                        background_dtype,
                        self.background_values,
                    )
                    foreground = self._make_image(
                        foreground_channels,
                        foreground_dtype,
                        self.foreground_values,
                    )
                    for opacity in opacities:
                        if opacity == 0.0:
                            expected = background
                        else:
                            expected = self._reference_blend(
                                mode_name,
                                background,
                                foreground,
                                opacity,
                                background_channels,
                            )
                        for kernel in available_kernels:
                            actual = func(background, foreground, float(opacity), kernel)
                            context = (
                                f"{mode_name} bg={background_dtype.__name__} "
                                f"fg={foreground_dtype.__name__} "
                                f"bg_ch={background_channels} "
                                f"fg_ch={foreground_channels} "
                                f"opacity={opacity} kernel={kernel}"
                            )
                            self._assert_close(actual, expected, context=context)

    def test_exhaustive_uint8_values(self) -> None:
        total_values = 256 * 256 * 256
        chunk_size = 65536
        opacity = 0.5

        kernels = ["auto", "scalar", "sse42", "avx2"]
        available_kernels = [
            kernel for kernel in kernels
            if simd_blend_modes._simd_blend_modes.kernel_available(kernel)
        ]

        for mode_name in BLEND_MODE_NAMES:
            func = getattr(simd_blend_modes, mode_name)
            for start in range(0, total_values, chunk_size):
                end = min(start + chunk_size, total_values)
                indices = np.arange(start, end, dtype=np.uint32)
                r = (indices >> 16) & 0xFF
                g = (indices >> 8) & 0xFF
                b = indices & 0xFF
                a = (indices & 0xFF).astype(np.uint8)
                background = np.stack([r, g, b, a], axis=1).astype(np.uint8)
                foreground = (255 - background).astype(np.uint8)
                background = background[:, None, :]
                foreground = foreground[:, None, :]
                expected = self._reference_blend(
                    mode_name,
                    background,
                    foreground,
                    opacity,
                    4,
                )
                for kernel in available_kernels:
                    actual = func(background, foreground, float(opacity), kernel)
                    context = (
                        f"{mode_name} uint8 exhaustive bg_ch=4 fg_ch=4 opacity={opacity} "
                        f"kernel={kernel} range={start}-{end - 1}"
                    )
                    self._assert_close(actual, expected, context=context)

    def test_exhaustive_float32_values(self) -> None:
        total_values = 256 * 256 * 256
        chunk_size = 65536
        opacity = 0.5

        kernels = ["auto", "scalar", "sse42", "avx2"]
        available_kernels = [
            kernel for kernel in kernels
            if simd_blend_modes._simd_blend_modes.kernel_available(kernel)
        ]

        for mode_name in BLEND_MODE_NAMES:
            func = getattr(simd_blend_modes, mode_name)
            for start in range(0, total_values, chunk_size):
                end = min(start + chunk_size, total_values)
                indices = np.arange(start, end, dtype=np.uint32)
                r = (indices >> 16) & 0xFF
                g = (indices >> 8) & 0xFF
                b = indices & 0xFF
                a = (indices & 0xFF).astype(np.uint8)
                background = np.stack([r, g, b, a], axis=1).astype(np.float32)
                foreground = (255.0 - background).astype(np.float32)
                background = background[:, None, :]
                foreground = foreground[:, None, :]
                expected = self._reference_blend(
                    mode_name,
                    background,
                    foreground,
                    opacity,
                    4,
                )
                for kernel in available_kernels:
                    actual = func(background, foreground, float(opacity), kernel)
                    context = (
                        f"{mode_name} float32 exhaustive bg_ch=4 fg_ch=4 opacity={opacity} "
                        f"kernel={kernel} range={start}-{end - 1}"
                    )
                    self._assert_close(actual, expected, context=context)

    def test_input_shape_validation(self) -> None:
        valid_background = np.ones((2, 2, 3), dtype=np.uint8)
        valid_foreground = np.ones((2, 2, 3), dtype=np.uint8)
        func = simd_blend_modes.normal

        with self.assertRaises(ValueError):
            func(valid_background[:1, :2], valid_foreground[:, :, :])

        with self.assertRaises(ValueError):
            func(valid_background, valid_foreground[:1, :, :])

        with self.assertRaises(ValueError):
            func(valid_background, valid_foreground[:, :1, :])

        with self.assertRaises(ValueError):
            func(np.zeros((2, 2), dtype=np.uint8), valid_foreground)

        with self.assertRaises(ValueError):
            func(valid_background, np.zeros((2, 2, 3, 1), dtype=np.uint8))

        with self.assertRaises(ValueError):
            func(np.zeros((2, 2, 2), dtype=np.uint8), valid_foreground)

        with self.assertRaises(ValueError):
            func(np.zeros((2, 2, 5), dtype=np.uint8), valid_foreground)

    def test_opacity_default(self) -> None:
        background = self._make_image(3, np.uint8, self.background_values)
        foreground = self._make_image(3, np.uint8, self.foreground_values)
        expected = self._reference_blend("normal", background, foreground, 1.0, 3)
        actual = simd_blend_modes.normal(background, foreground)
        self._assert_close(actual, expected, context="normal opacity default")

    def test_output_dtype_matches_background(self) -> None:
        background_u8 = self._make_image(3, np.uint8, self.background_values)
        foreground_f32 = self._make_image(3, np.float32, self.foreground_values)
        out_u8 = simd_blend_modes.screen(background_u8, foreground_f32, 1.0)
        self.assertEqual(out_u8.dtype, np.uint8)

        background_f32 = self._make_image(3, np.float32, self.background_values)
        foreground_u8 = self._make_image(3, np.uint8, self.foreground_values)
        out_f32 = simd_blend_modes.screen(background_f32, foreground_u8, 1.0)
        self.assertEqual(out_f32.dtype, np.float32)

    def test_opacity_zero_returns_background(self) -> None:
        background = self._make_image(4, np.float32, self.background_values)
        foreground = self._make_image(4, np.float32, self.foreground_values)
        out = simd_blend_modes.multiply(background, foreground, 0.0)
        self._assert_close(out, background, context="opacity zero returns background")

    @unittest.skipIf(cv2 is None, "cv2 is required for PNG fixture tests")
    def test_additional_blend_modes_match_gimp_fixtures(self) -> None:
        background = _read_fixture_rgba("orig.png")
        foreground = _read_fixture_rgba("layer.png")
        opaque_layer = foreground[:, :, 3] == 255.0

        for mode_name, func in ADDITIONAL_BLEND_MODES.items():
            with self.subTest(mode=mode_name):
                expected = _read_fixture_rgba(f"{mode_name}.png")
                actual = np.clip(np.rint(func(background, foreground, 1.0)), 0.0, 255.0)
                diff = np.abs(actual - expected)[opaque_layer]

                if mode_name == "lch_hue":
                    self.assertLessEqual(np.mean(diff), 4.0)
                    self.assertLessEqual(np.percentile(diff, 95), 4.0)
                else:
                    self.assertLessEqual(np.max(diff), 52.0)
                    self.assertLessEqual(np.percentile(diff, 99), 6.0)

    def test_additional_blend_modes_opacity_zero_returns_background(self) -> None:
        background = self._make_image(4, np.float32, self.background_values)
        foreground = self._make_image(4, np.float32, self.foreground_values)

        for mode_name, func in ADDITIONAL_BLEND_MODES.items():
            with self.subTest(mode=mode_name):
                actual = func(background, foreground, 0.0)
                self._assert_close(actual, background, context=f"{mode_name} opacity zero")

    def test_additional_blend_modes_return_finite_rgba(self) -> None:
        background = self._make_image(4, np.float32, self.background_values)
        foreground = self._make_image(4, np.float32, self.foreground_values)

        for mode_name, func in ADDITIONAL_BLEND_MODES.items():
            with self.subTest(mode=mode_name):
                actual = func(background, foreground, 1.0)
                self.assertEqual(actual.shape, background.shape)
                self.assertTrue(np.isfinite(actual).all())

    def test_additional_simd_blend_modes_match_python_reference(self) -> None:
        opacities = [0.0, 0.5, 1.0]
        dtype_pairs = [
            (np.uint8, np.uint8),
            (np.uint8, np.float32),
            (np.float32, np.uint8),
            (np.float32, np.float32),
        ]
        channel_pairs = [(3, 3), (3, 4), (4, 3), (4, 4)]

        kernels = ["auto", "scalar", "sse42", "avx2"]
        available_kernels = [
            kernel for kernel in kernels
            if simd_blend_modes._simd_blend_modes.kernel_available(kernel)
        ]

        for mode_name in ADDITIONAL_SIMD_BLEND_MODE_NAMES:
            simd_func = getattr(simd_blend_modes, mode_name)
            reference_func = ADDITIONAL_BLEND_MODES[mode_name]
            for background_dtype, foreground_dtype in dtype_pairs:
                for background_channels, foreground_channels in channel_pairs:
                    background = _make_wide_image(
                        background_channels,
                        background_dtype,
                        WIDE_BACKGROUND_VALUES,
                    )
                    foreground = _make_wide_image(
                        foreground_channels,
                        foreground_dtype,
                        WIDE_FOREGROUND_VALUES,
                    )
                    for opacity in opacities:
                        expected = reference_func(background, foreground, float(opacity))
                        if background_channels == 3:
                            expected = expected[:, :, :3]
                        if background_dtype == np.uint8:
                            expected = np.clip(np.rint(expected), 0.0, 255.0)
                        for kernel in available_kernels:
                            actual = simd_func(background, foreground, float(opacity), kernel)
                            context = (
                                f"{mode_name} bg={background_dtype.__name__} "
                                f"fg={foreground_dtype.__name__} "
                                f"bg_ch={background_channels} "
                                f"fg_ch={foreground_channels} "
                                f"opacity={opacity} kernel={kernel}"
                            )
                            self._assert_close(actual, expected, context=context)

    def test_existing_simd_kernel_paths_match_reference_at_vector_width(self) -> None:
        dtype_pairs = [
            (np.uint8, np.uint8),
            (np.uint8, np.float32),
            (np.float32, np.uint8),
            (np.float32, np.float32),
        ]
        channel_pairs = [(3, 3), (3, 4), (4, 3), (4, 4)]
        kernels = ["auto", "scalar", "sse42", "avx2"]
        available_kernels = [
            kernel for kernel in kernels
            if simd_blend_modes._simd_blend_modes.kernel_available(kernel)
        ]

        for mode_name in BLEND_MODE_NAMES:
            func = getattr(simd_blend_modes, mode_name)
            for background_dtype, foreground_dtype in dtype_pairs:
                for background_channels, foreground_channels in channel_pairs:
                    background = _make_wide_image(
                        background_channels,
                        background_dtype,
                        WIDE_BACKGROUND_VALUES,
                    )
                    foreground = _make_wide_image(
                        foreground_channels,
                        foreground_dtype,
                        WIDE_FOREGROUND_VALUES,
                    )
                    expected = self._reference_blend(
                        mode_name,
                        background,
                        foreground,
                        0.5,
                        background_channels,
                    )
                    for kernel in available_kernels:
                        actual = func(background, foreground, 0.5, kernel)
                        context = (
                            f"{mode_name} vector-width parity "
                            f"bg={background_dtype.__name__} "
                            f"fg={foreground_dtype.__name__} "
                            f"bg_ch={background_channels} "
                            f"fg_ch={foreground_channels} kernel={kernel}"
                        )
                        self._assert_close(actual, expected, context=context)


if __name__ == "__main__":
    unittest.main()
