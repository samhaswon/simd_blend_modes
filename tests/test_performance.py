import gc
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

import simd_blend_modes
from simd_blend_modes import _simd_blend_modes

from blend_modes import blending_functions as reference

WRITE_RESULTS_TO_README = True
README_PATH = Path(__file__).resolve().parents[1] / "README.md"
PERF_RESULTS_START = "<!-- PERF_RESULTS_START -->"
PERF_RESULTS_END = "<!-- PERF_RESULTS_END -->"

@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    height: int
    width: int


def _time_call(fn: Callable[[], None], iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    end = time.perf_counter()
    return (end - start) / float(iterations)


def _iterations_for_size(height: int, width: int) -> int:
    pixels = height * width
    if pixels <= 256 * 256:
        return 5
    if pixels <= 1024 * 1024:
        return 4
    return 3


def _blend_reference(func: Callable[..., np.ndarray], bg: np.ndarray, fg: np.ndarray) -> None:
    func(bg, fg, 0.5)


def _blend_reference_with_alpha(
    func: Callable[..., np.ndarray],
    bg: np.ndarray,
    fg: np.ndarray,
) -> None:
    bg_alpha = np.full(bg.shape[:2] + (1,), 255.0, dtype=bg.dtype)
    fg_alpha = np.full(fg.shape[:2] + (1,), 255.0, dtype=fg.dtype)
    bg_ref = np.dstack((bg, bg_alpha))
    fg_ref = np.dstack((fg, fg_alpha))
    func(bg_ref, fg_ref, 0.5)


def _blend_simd(
    func: Callable[..., np.ndarray],
    bg: np.ndarray,
    fg: np.ndarray,
    kernel: str,
) -> None:
    func(bg, fg, 0.5, kernel)


def _blend_reference_with_cast(
    func: Callable[..., np.ndarray],
    bg: np.ndarray,
    fg: np.ndarray,
) -> None:
    bg_ref = bg.astype(np.float32)
    fg_ref = fg.astype(np.float32)
    func(bg_ref, fg_ref, 0.5)


def _blend_reference_with_cast_and_alpha(
    func: Callable[..., np.ndarray],
    bg: np.ndarray,
    fg: np.ndarray,
) -> None:
    bg_ref = bg.astype(np.float32)
    fg_ref = fg.astype(np.float32)
    bg_alpha = np.full(bg_ref.shape[:2] + (1,), 255.0, dtype=bg_ref.dtype)
    fg_alpha = np.full(fg_ref.shape[:2] + (1,), 255.0, dtype=fg_ref.dtype)
    bg_ref = np.dstack((bg_ref, bg_alpha))
    fg_ref = np.dstack((fg_ref, fg_alpha))
    func(bg_ref, fg_ref, 0.5)


class TestPerformance(unittest.TestCase):
    def test_kernel_speedups(self) -> None:
        cases = [
            BenchmarkCase("256x256", 256, 256),
            BenchmarkCase("512x512", 512, 512),
            BenchmarkCase("1024x1024", 1024, 1024),
            BenchmarkCase("2048x2048", 2048, 2048),
            BenchmarkCase("1280x720", 720, 1280),
            BenchmarkCase("1920x1080", 1080, 1920),
            BenchmarkCase("2560x1440", 1440, 2560),
            BenchmarkCase("3840x2160", 2160, 3840),
        ]
        modes = [
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
        kernels = ["scalar", "sse42", "avx2"]
        available_kernels = [
            kernel for kernel in kernels if _simd_blend_modes.kernel_available(kernel)
        ]

        input_kinds = [
            ("uint8", np.uint8),
            ("float32", np.float32),
        ]
        results: Dict[Tuple[str, str, str, str, int], Dict[str, float]] = {}

        for case in cases:
            iterations = _iterations_for_size(case.height, case.width)
            for input_label, input_dtype in input_kinds:
                try:
                    rng = np.random.default_rng(1234)
                    background_u8_4 = rng.integers(
                        0,
                        256,
                        size=(case.height, case.width, 4),
                        dtype=np.uint8,
                    )
                    foreground_u8_4 = rng.integers(
                        0,
                        256,
                        size=(case.height, case.width, 4),
                        dtype=np.uint8,
                    )
                except MemoryError:
                    print(f"Skipping {case.name} {input_label} due to memory constraints")
                    continue

                background_u8_3 = background_u8_4[:, :, :3]
                foreground_u8_3 = foreground_u8_4[:, :, :3]

                for channels in (3, 4):
                    background_u8 = background_u8_3 if channels == 3 else background_u8_4
                    foreground_u8 = foreground_u8_3 if channels == 3 else foreground_u8_4

                    if input_dtype == np.float32:
                        background = background_u8.astype(np.float32)
                        foreground = foreground_u8.astype(np.float32)
                    else:
                        background = background_u8
                        foreground = foreground_u8

                    for mode in modes:
                        ref_func = getattr(reference, mode)
                        simd_func = getattr(simd_blend_modes, mode)

                        gc.collect()
                        if channels == 3:
                            if input_dtype == np.float32:
                                ref_time = _time_call(
                                    lambda: _blend_reference_with_alpha(
                                        ref_func,
                                        background,
                                        foreground,
                                    ),
                                    iterations,
                                )
                            else:
                                ref_time = _time_call(
                                    lambda: _blend_reference_with_cast_and_alpha(
                                        ref_func,
                                        background,
                                        foreground,
                                    ),
                                    iterations,
                                )
                        else:
                            if input_dtype == np.float32:
                                ref_time = _time_call(
                                    lambda: _blend_reference(ref_func, background, foreground),
                                    iterations,
                                )
                            else:
                                ref_time = _time_call(
                                    lambda: _blend_reference_with_cast(ref_func, background, foreground),
                                    iterations,
                                )

                        for kernel in available_kernels:
                            gc.collect()
                            simd_time = _time_call(
                                lambda: _blend_simd(simd_func, background, foreground, kernel),
                                iterations,
                            )
                            speedup = ref_time / simd_time if simd_time > 0 else float("inf")
                            results[(case.name, input_label, mode, kernel, channels)] = {
                                "ref_time": ref_time,
                                "simd_time": simd_time,
                                "speedup": speedup,
                            }

        headers = [
            "Case",
            "Input",
            "Channels",
            "Mode",
            "Kernel",
            "Ref (s)",
            "Kernel (s)",
            "Speedup",
            "Percent Change",
        ]
        rows: List[List[str]] = []
        for case in cases:
            for input_label, _ in input_kinds:
                for channels in (3, 4):
                    for mode in modes:
                        for kernel in available_kernels:
                            key = (case.name, input_label, mode, kernel, channels)
                            if key not in results:
                                continue
                            entry = results[key]
                            rows.append(
                                [
                                    case.name,
                                    input_label,
                                    str(channels),
                                    mode,
                                    kernel,
                                    f"{entry['ref_time']:.6f}",
                                    f"{entry['simd_time']:.6f}",
                                    f"{entry['speedup']:.2f}x",
                                    f"{(entry['simd_time'] / entry['ref_time'] - 1.0) * 100.0:.2f}%",
                                ]
                            )

        columns = list(zip(*([headers] + rows))) if rows else [headers]
        widths = [max(len(item) for item in col) for col in columns]

        def _format_row(values: List[str]) -> str:
            padded = [val.ljust(widths[i]) for i, val in enumerate(values)]
            return "| " + " | ".join(padded) + " |"

        def _format_row_width(values: List[str], widths: List[int]) -> str:
            padded = [val.ljust(widths[i]) for i, val in enumerate(values)]
            return "| " + " | ".join(padded) + " |"

        summary_headers = [
            "Mode",
            "Kernel",
            "Ref (s)",
            "Kernel (s)",
            "Speedup",
            "Percent Change",
        ]
        summary_rows: List[List[str]] = []
        for mode in modes:
            for kernel in available_kernels:
                entries = [
                    results[(case.name, input_label, mode, kernel, channels)]
                    for case in cases
                    for input_label, _ in input_kinds
                    for channels in (3, 4)
                    if (case.name, input_label, mode, kernel, channels) in results
                ]
                if not entries:
                    continue
                avg_ref = sum(entry["ref_time"] for entry in entries) / len(entries)
                avg_simd = sum(entry["simd_time"] for entry in entries) / len(entries)
                avg_speedup = avg_ref / avg_simd if avg_simd > 0 else float("inf")
                avg_percent = (avg_simd / avg_ref - 1.0) * 100.0 if avg_ref > 0 else 0.0
                summary_rows.append(
                    [
                        mode,
                        kernel,
                        f"{avg_ref:.6f}",
                        f"{avg_simd:.6f}",
                        f"{avg_speedup:.2f}x",
                        f"{avg_percent:.2f}%",
                    ]
                )
        if summary_rows:
            summary_columns = list(zip(*([summary_headers] + summary_rows)))
            summary_widths = [max(len(item) for item in col) for col in summary_columns]
            output_lines = [
                "",
                _format_row_width(summary_headers, summary_widths),
                _format_row_width(["-" * width for width in summary_widths], summary_widths),
            ]
            for row in summary_rows:
                output_lines.append(_format_row_width(row, summary_widths))
        else:
            output_lines = []

        output_lines.extend(
            [
                "",
                "<details>",
                "<summary>Per-kernel, size, and type results</summary>",
                "",
                _format_row(headers),
                _format_row(["-" * width for width in widths]),
            ]
        )
        for row in rows:
            output_lines.append(_format_row(row))
        output_lines.append("</details>")

        output_text = "\n".join(output_lines).lstrip("\n")
        print(output_text)

        if WRITE_RESULTS_TO_README:
            readme_text = README_PATH.read_text(encoding="utf-8")
            start_idx = readme_text.find(PERF_RESULTS_START)
            end_idx = readme_text.find(PERF_RESULTS_END)
            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                raise AssertionError("Performance markers not found in README.md")
            start_idx += len(PERF_RESULTS_START)
            replacement = "\n" + output_text.rstrip("\n") + "\n"
            updated = readme_text[:start_idx] + replacement + readme_text[end_idx:]
            README_PATH.write_text(updated, encoding="utf-8")

        self.assertTrue(results, "No performance results collected")


if __name__ == "__main__":
    unittest.main()
