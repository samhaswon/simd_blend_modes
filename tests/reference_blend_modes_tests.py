"""
Reference tests adapted from the blend_modes project.

Source: https://github.com/flrs/blend_modes (tests/test_blend_modes.py)
Original license: MIT (see blend_modes/LICENSE.txt in that repository)
This file preserves key tests for image-based validation and type checks.

Note: These tests are skipped unless the original blend_modes package and test
assets are available in the environment.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from blend_modes import (
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
    )
    from blend_modes.type_checks import assert_image_format, assert_opacity
except Exception:  # pragma: no cover - optional dependency
    addition = None
    assert_image_format = None
    assert_opacity = None


_TEST_LIMIT = 10
_TEST_TOLERANCE = 0.001


def _test_criteria(out: np.ndarray, comp: np.ndarray) -> bool:
    return (
        np.sum(np.absolute(out - comp) > _TEST_LIMIT) / np.prod(comp.shape)
    ) < _TEST_TOLERANCE


class TestBlendModesReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if cv2 is None or addition is None:
            raise unittest.SkipTest("blend_modes and cv2 are required for reference tests")

        cls._assets_dir = Path(__file__).resolve().parent.parent / "blend_modes" / "tests"
        if not cls._assets_dir.exists():
            raise unittest.SkipTest("blend_modes test assets not available")

    def _load_image(self, name: str) -> np.ndarray:
        path = self._assets_dir / name
        if not path.exists():
            self.skipTest(f"missing test asset: {path}")
        image = cv2.imread(str(path), -1)
        if image is None:
            self.skipTest(f"unable to load test asset: {path}")
        return image.astype(float)

    def test_addition(self) -> None:
        out = soft_light(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("soft_light.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_darken_only(self) -> None:
        out = darken_only(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("darken_only.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_multiply(self) -> None:
        out = multiply(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("multiply.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_difference(self) -> None:
        out = difference(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("difference.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_divide(self) -> None:
        out = divide(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("divide.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_dodge(self) -> None:
        out = dodge(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("dodge.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_grain_extract(self) -> None:
        out = grain_extract(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("grain_extract.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_grain_merge(self) -> None:
        out = grain_merge(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("grain_merge.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_hard_light(self) -> None:
        out = hard_light(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("hard_light.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_lighten_only(self) -> None:
        out = lighten_only(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("lighten_only.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_soft_light_50p(self) -> None:
        out = soft_light(self._load_image("orig.png"), self._load_image("layer_50p.png"), 0.8)
        comp = self._load_image("soft_light_50p.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_overlay(self) -> None:
        out = overlay(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("overlay.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_normal_50p(self) -> None:
        out = normal(self._load_image("orig.png"), self._load_image("layer.png"), 0.5)
        comp = self._load_image("normal_50p.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_normal_100p(self) -> None:
        out = normal(self._load_image("orig.png"), self._load_image("layer.png"), 1.0)
        comp = self._load_image("normal_100p.png")
        self.assertTrue(_test_criteria(out, comp))

    def test_assert_image_format_dims_force_alpha(self) -> None:
        if assert_image_format is None:
            self.skipTest("blend_modes.type_checks not available")
        with self.assertRaises(TypeError):
            assert_image_format(
                np.ndarray(dtype=float, shape=[640, 640, 3]),
                fcn_name="",
                arg_name="",
                force_alpha=True,
            )

    def test_assert_image_format_dims_not_force_alpha(self) -> None:
        if assert_image_format is None:
            self.skipTest("blend_modes.type_checks not available")
        assert_image_format(
            np.ndarray(dtype=float, shape=[640, 640, 3]),
            fcn_name="",
            arg_name="",
            force_alpha=False,
        )

    def test_assert_image_format_dims(self) -> None:
        if assert_image_format is None:
            self.skipTest("blend_modes.type_checks not available")
        with self.assertRaises(TypeError):
            assert_image_format(
                np.ndarray(dtype=float, shape=[640, 640, 2]),
                fcn_name="",
                arg_name="",
            )

    def test_assert_image_format_shape(self) -> None:
        if assert_image_format is None:
            self.skipTest("blend_modes.type_checks not available")
        with self.assertRaises(TypeError):
            assert_image_format(
                np.ndarray(dtype=float, shape=[640, 640]),
                fcn_name="",
                arg_name="",
            )

    def test_assert_image_format_kind(self) -> None:
        if assert_image_format is None:
            self.skipTest("blend_modes.type_checks not available")
        with self.assertRaises(TypeError):
            assert_image_format(
                np.ndarray(dtype=int, shape=[640, 640, 4]),
                fcn_name="",
                arg_name="",
            )

    def test_assert_image_format_type(self) -> None:
        if assert_image_format is None:
            self.skipTest("blend_modes.type_checks not available")
        with self.assertRaises(TypeError):
            assert_image_format(2.0, fcn_name="", arg_name="")

    def test_assert_opacity_wrong_variable_type(self) -> None:
        if assert_opacity is None:
            self.skipTest("blend_modes.type_checks not available")
        opacity = "0.5"
        with self.assertRaises(TypeError):
            assert_opacity(opacity, "")

    def test_assert_opacity_right_variable_type(self) -> None:
        if assert_opacity is None:
            self.skipTest("blend_modes.type_checks not available")
        assert_opacity(0.5, "")

    def test_assert_opacity_wrong_variable_range_low(self) -> None:
        if assert_opacity is None:
            self.skipTest("blend_modes.type_checks not available")
        with self.assertRaises(ValueError):
            assert_opacity(-5.0, "")

    def test_assert_opacity_wrong_variable_range_high(self) -> None:
        if assert_opacity is None:
            self.skipTest("blend_modes.type_checks not available")
        with self.assertRaises(ValueError):
            assert_opacity(1.01, "")


if __name__ == "__main__":
    unittest.main()
