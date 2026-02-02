# SIMD Blend Modes

This project reimplements the blend modes from [`blend_modes`](https://github.com/flrs/blend_modes) with C kernels and SIMD
(SSE4.2/AVX2) acceleration. It supports uint8 and float32 NumPy inputs in the range 0..255
and returns output dtype/channel count matching the background image. Missing alpha channels
are treated as fully opaque (255). Opacity defaults to 1.0.

## Build and Install

```bash
python3 -m pip install -e .
```

## Usage

```python
import numpy as np
import simd_blend_modes as sbm

background = np.zeros((512, 512, 4), dtype=np.uint8)
foreground = np.zeros((512, 512, 4), dtype=np.uint8)

out = sbm.screen(background, foreground, 0.5)
```

You can force a kernel by passing a string (or `KernelKind`):

```python
out = sbm.screen(background, foreground, 0.5, "avx2")
```

## Tests

Correctness:

```bash
python3 -m unittest discover tests/
```

Performance:

```bash
python3 -m unittest tests.test_performance
```

The performance test prints a markdown table of per-kernel speedups vs the NumPy reference
for common square sizes and screen resolutions.

## Notes

- SIMD kernels are selected at runtime: AVX2 → SSE4.2 → scalar.
- The reference implementation is not included in this repository (it is ignored by git).
- Reference tests adapted from the original project live in `tests/reference_blend_modes_tests.py`
  and are skipped unless the `blend_modes` package and test assets are available.
- The SIMD paths currently assume contiguous arrays (the input validation enforces this).

## Performance 

| Case      | Mode     | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | -------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | normal   | scalar | 0.006662 | 0.001362   | 4.89x   | -79.55%        |
| 256x256   | normal   | sse42  | 0.006662 | 0.001044   | 6.38x   | -84.33%        |
| 256x256   | normal   | avx2   | 0.006662 | 0.001087   | 6.13x   | -83.69%        |
| 256x256   | multiply | scalar | 0.007513 | 0.001621   | 4.64x   | -78.43%        |
| 256x256   | multiply | sse42  | 0.007513 | 0.001054   | 7.13x   | -85.97%        |
| 256x256   | multiply | avx2   | 0.007513 | 0.001110   | 6.77x   | -85.23%        |
| 256x256   | screen   | scalar | 0.006496 | 0.001703   | 3.81x   | -73.78%        |
| 256x256   | screen   | sse42  | 0.006496 | 0.001045   | 6.21x   | -83.91%        |
| 256x256   | screen   | avx2   | 0.006496 | 0.001091   | 5.96x   | -83.21%        |
| 512x512   | normal   | scalar | 0.022320 | 0.005378   | 4.15x   | -75.90%        |
| 512x512   | normal   | sse42  | 0.022320 | 0.004145   | 5.39x   | -81.43%        |
| 512x512   | normal   | avx2   | 0.022320 | 0.004331   | 5.15x   | -80.59%        |
| 512x512   | multiply | scalar | 0.025804 | 0.006692   | 3.86x   | -74.06%        |
| 512x512   | multiply | sse42  | 0.025804 | 0.006040   | 4.27x   | -76.59%        |
| 512x512   | multiply | avx2   | 0.025804 | 0.006980   | 3.70x   | -72.95%        |
| 512x512   | screen   | scalar | 0.029229 | 0.009057   | 3.23x   | -69.01%        |
| 512x512   | screen   | sse42  | 0.029229 | 0.004556   | 6.42x   | -84.41%        |
| 512x512   | screen   | avx2   | 0.029229 | 0.004395   | 6.65x   | -84.96%        |
| 1024x1024 | normal   | scalar | 0.067396 | 0.021487   | 3.14x   | -68.12%        |
| 1024x1024 | normal   | sse42  | 0.067396 | 0.016433   | 4.10x   | -75.62%        |
| 1024x1024 | normal   | avx2   | 0.067396 | 0.016945   | 3.98x   | -74.86%        |
| 1024x1024 | multiply | scalar | 0.071204 | 0.033039   | 2.16x   | -53.60%        |
| 1024x1024 | multiply | sse42  | 0.071204 | 0.017486   | 4.07x   | -75.44%        |
| 1024x1024 | multiply | avx2   | 0.071204 | 0.018661   | 3.82x   | -73.79%        |
| 1024x1024 | screen   | scalar | 0.069659 | 0.026621   | 2.62x   | -61.78%        |
| 1024x1024 | screen   | sse42  | 0.069659 | 0.016562   | 4.21x   | -76.22%        |
| 1024x1024 | screen   | avx2   | 0.069659 | 0.017369   | 4.01x   | -75.07%        |
| 2048x2048 | normal   | scalar | 0.254364 | 0.085124   | 2.99x   | -66.53%        |
| 2048x2048 | normal   | sse42  | 0.254364 | 0.065440   | 3.89x   | -74.27%        |
| 2048x2048 | normal   | avx2   | 0.254364 | 0.067702   | 3.76x   | -73.38%        |
| 2048x2048 | multiply | scalar | 0.258615 | 0.103388   | 2.50x   | -60.02%        |
| 2048x2048 | multiply | sse42  | 0.258615 | 0.067015   | 3.86x   | -74.09%        |
| 2048x2048 | multiply | avx2   | 0.258615 | 0.070783   | 3.65x   | -72.63%        |
| 2048x2048 | screen   | scalar | 0.281895 | 0.106401   | 2.65x   | -62.26%        |
| 2048x2048 | screen   | sse42  | 0.281895 | 0.070217   | 4.01x   | -75.09%        |
| 2048x2048 | screen   | avx2   | 0.281895 | 0.073929   | 3.81x   | -73.77%        |
| 1280x720  | normal   | scalar | 0.065743 | 0.019400   | 3.39x   | -70.49%        |
| 1280x720  | normal   | sse42  | 0.065743 | 0.014380   | 4.57x   | -78.13%        |
| 1280x720  | normal   | avx2   | 0.065743 | 0.014980   | 4.39x   | -77.22%        |
| 1280x720  | multiply | scalar | 0.070777 | 0.022507   | 3.14x   | -68.20%        |
| 1280x720  | multiply | sse42  | 0.070777 | 0.014748   | 4.80x   | -79.16%        |
| 1280x720  | multiply | avx2   | 0.070777 | 0.015763   | 4.49x   | -77.73%        |
| 1280x720  | screen   | scalar | 0.071781 | 0.023369   | 3.07x   | -67.44%        |
| 1280x720  | screen   | sse42  | 0.071781 | 0.014876   | 4.83x   | -79.28%        |
| 1280x720  | screen   | avx2   | 0.071781 | 0.015722   | 4.57x   | -78.10%        |
| 1920x1080 | normal   | scalar | 0.131687 | 0.042640   | 3.09x   | -67.62%        |
| 1920x1080 | normal   | sse42  | 0.131687 | 0.032533   | 4.05x   | -75.29%        |
| 1920x1080 | normal   | avx2   | 0.131687 | 0.034093   | 3.86x   | -74.11%        |
| 1920x1080 | multiply | scalar | 0.134145 | 0.052983   | 2.53x   | -60.50%        |
| 1920x1080 | multiply | sse42  | 0.134145 | 0.033170   | 4.04x   | -75.27%        |
| 1920x1080 | multiply | avx2   | 0.134145 | 0.035005   | 3.83x   | -73.91%        |
| 1920x1080 | screen   | scalar | 0.135322 | 0.052253   | 2.59x   | -61.39%        |
| 1920x1080 | screen   | sse42  | 0.135322 | 0.033601   | 4.03x   | -75.17%        |
| 1920x1080 | screen   | avx2   | 0.135322 | 0.034687   | 3.90x   | -74.37%        |
| 2560x1440 | normal   | scalar | 0.224093 | 0.075571   | 2.97x   | -66.28%        |
| 2560x1440 | normal   | sse42  | 0.224093 | 0.057906   | 3.87x   | -74.16%        |
| 2560x1440 | normal   | avx2   | 0.224093 | 0.059906   | 3.74x   | -73.27%        |
| 2560x1440 | multiply | scalar | 0.224900 | 0.090359   | 2.49x   | -59.82%        |
| 2560x1440 | multiply | sse42  | 0.224900 | 0.059160   | 3.80x   | -73.69%        |
| 2560x1440 | multiply | avx2   | 0.224900 | 0.062218   | 3.61x   | -72.34%        |
| 2560x1440 | screen   | scalar | 0.237774 | 0.094557   | 2.51x   | -60.23%        |
| 2560x1440 | screen   | sse42  | 0.237774 | 0.058509   | 4.06x   | -75.39%        |
| 2560x1440 | screen   | avx2   | 0.237774 | 0.061436   | 3.87x   | -74.16%        |
| 3840x2160 | normal   | scalar | 0.509620 | 0.168835   | 3.02x   | -66.87%        |
| 3840x2160 | normal   | sse42  | 0.509620 | 0.129727   | 3.93x   | -74.54%        |
| 3840x2160 | normal   | avx2   | 0.509620 | 0.134291   | 3.79x   | -73.65%        |
| 3840x2160 | multiply | scalar | 0.498596 | 0.203224   | 2.45x   | -59.24%        |
| 3840x2160 | multiply | sse42  | 0.498596 | 0.132480   | 3.76x   | -73.43%        |
| 3840x2160 | multiply | avx2   | 0.498596 | 0.140142   | 3.56x   | -71.89%        |
| 3840x2160 | screen   | scalar | 0.517649 | 0.211524   | 2.45x   | -59.14%        |
| 3840x2160 | screen   | sse42  | 0.517649 | 0.131242   | 3.94x   | -74.65%        |
| 3840x2160 | screen   | avx2   | 0.517649 | 0.137666   | 3.76x   | -73.41%        |
