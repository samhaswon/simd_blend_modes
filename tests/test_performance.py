import gc
import time
import unittest
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

import simd_blend_modes
from simd_blend_modes import _simd_blend_modes

from blend_modes import blending_functions as reference


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
        return 3
    return 1


def _blend_reference(func: Callable[..., np.ndarray], bg: np.ndarray, fg: np.ndarray) -> None:
    func(bg, fg, 0.5)


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
        results: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}

        for case in cases:
            iterations = _iterations_for_size(case.height, case.width)
            for input_label, input_dtype in input_kinds:
                try:
                    rng = np.random.default_rng(1234)
                    background_u8 = rng.integers(
                        0,
                        256,
                        size=(case.height, case.width, 4),
                        dtype=np.uint8,
                    )
                    foreground_u8 = rng.integers(
                        0,
                        256,
                        size=(case.height, case.width, 4),
                        dtype=np.uint8,
                    )
                except MemoryError:
                    print(f"Skipping {case.name} {input_label} due to memory constraints")
                    continue

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
                        results[(case.name, input_label, mode, kernel)] = {
                            "ref_time": ref_time,
                            "simd_time": simd_time,
                            "speedup": speedup,
                        }

        headers = [
            "Case",
            "Input",
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
                for mode in modes:
                    for kernel in available_kernels:
                        key = (case.name, input_label, mode, kernel)
                        if key not in results:
                            continue
                        entry = results[key]
                        rows.append(
                            [
                                case.name,
                                input_label,
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
                    results[(case.name, input_label, mode, kernel)]
                    for case in cases
                    for input_label, _ in input_kinds
                    if (case.name, input_label, mode, kernel) in results
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
            print("")
            print(_format_row_width(summary_headers, summary_widths))
            print(_format_row_width(["-" * width for width in summary_widths], summary_widths))
            for row in summary_rows:
                print(_format_row_width(row, summary_widths))

        print("")
        print("<details>")
        print("<summary>Per-kernel results</summary>")
        print("")
        print(_format_row(headers))
        print(_format_row(["-" * width for width in widths]))
        for row in rows:
            print(_format_row(row))
        print("</details>")

        self.assertTrue(results, "No performance results collected")


if __name__ == "__main__":
    unittest.main()
