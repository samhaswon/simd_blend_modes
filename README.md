# SIMD Blend Modes

This project reimplements the blend modes from [`blend_modes`](https://github.com/flrs/blend_modes) with C kernels and SIMD
(SSE4.2/AVX2) acceleration. It supports uint8 and float32 NumPy inputs in the range 0..255
and returns output dtype/channel count matching the background image. Missing alpha channels
are treated as fully opaque (255). Opacity defaults to 1.0.

This is mostly intended to be a mostly drop-in replacement, but with a more permissive 
API that allows you to go faster if you don't need FP32 arrays or the information of an
Alpha channel for some layers.

## Build and Install

### General

```bash
pip install simd-blend-modes
```

### Development

```bash
pip install -r requirements-dev.txt
pip install -e .
```

## Usage

```python
import numpy as np
import simd_blend_modes as sbm

background = np.zeros((512, 512, 4), dtype=np.uint8)
foreground = np.zeros((512, 512, 4), dtype=np.uint8)

out = sbm.screen(background, foreground, 0.5)
```

Inputs:

- Dtypes: `np.uint8` or `np.float32` only.
- Value range: 0..255 for both dtypes.
  - This expects float32 inputs to be cast from uint8, not normalized as well.
- Shapes: `H x W x C` with `C` = 3 (RGB) or 4 (RGBA).
- Output: dtype and channel count match the background image.
- Alpha: if a source is RGB (3 channels), alpha is treated as 255 (fully opaque).
- Opacity: the third argument is optional; defaults to `1.0`.

Supported blend modes:

- [`normal`](https://en.wikipedia.org/wiki/Blend_modes#Normal_blend_mode)
- [`soft_light`](https://en.wikipedia.org/wiki/Blend_modes#Soft_Light)
- [`lighten_only`](https://en.wikipedia.org/wiki/Blend_modes#Lighten_Only)
- [`screen`](https://en.wikipedia.org/wiki/Blend_modes#Screen)
- [`dodge`](https://en.wikipedia.org/wiki/Blend_modes#Dodge_and_burn)
- [`addition`](https://en.wikipedia.org/wiki/Blend_modes#Addition)
- [`darken_only`](https://en.wikipedia.org/wiki/Blend_modes#Darken_Only)
- [`multiply`](https://en.wikipedia.org/wiki/Blend_modes#Multiply)
- [`hard_light`](https://en.wikipedia.org/wiki/Blend_modes#Hard_Light)
- [`difference`](https://en.wikipedia.org/wiki/Blend_modes#Difference)
- [`subtract`](https://en.wikipedia.org/wiki/Blend_modes#Subtract)
- `grain_extract` (known from GIMP)
- `grain_merge` (known from GIMP)
- [`divide`](https://en.wikipedia.org/wiki/Blend_modes#Divide)
- [`overlay`](https://en.wikipedia.org/wiki/Blend_modes#Overlay)

You can force a kernel by passing a string (or `KernelKind` value):

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

## ARM

ARM isn't properly supported as I do not have a new enough ARM CPU to test on. 
Nor do I wish to use a cloud VM to test it. So, if you want ARM support, open a PR.
It should build and be faster, but there's no SIMD support there (yet).

ARM builds run in scalar-only mode (x86 SIMD is compile-time gated). To test ARM under Docker,
enable emulation and then build with the ARM platform. 

If you don't already have buildx/binfmt configured, run:

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

Then build or run the ARM container:

```bash
docker compose up --build
```

This is incredibly slow. I wouldn't actually do this, but it's here.

## Notes

- SIMD kernels are selected at runtime: AVX2 → SSE4.2 → scalar.
- ARM builds are supported in scalar-only mode; x86 SIMD is compile-time gated. CI does not emit
  ARM artifacts.
- Reference tests adapted from the original project live in `tests/reference_blend_modes_tests.py`
  and are skipped unless the `blend_modes` package and test assets are available.
- The SIMD paths currently assume contiguous arrays (the input validation enforces this).

## Performance 

<!--
The performance test prints large tables. If your terminal buffer is limited, you can write the
output into this README instead by setting `WRITE_RESULTS_TO_README = True` in
`tests/test_performance.py`. When enabled, it replaces the block between the markers below.
-->

<!-- PERF_RESULTS_START -->
| Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| ------------- | ------ | -------- | ---------- | ------- | -------------- |
| normal        | scalar | 0.205344 | 0.043612   | 4.71x   | -78.76%        |
| normal        | sse42  | 0.205344 | 0.021944   | 9.36x   | -89.31%        |
| normal        | avx2   | 0.205344 | 0.022534   | 9.11x   | -89.03%        |
| soft_light    | scalar | 0.273715 | 0.053343   | 5.13x   | -80.51%        |
| soft_light    | sse42  | 0.273715 | 0.024034   | 11.39x  | -91.22%        |
| soft_light    | avx2   | 0.273715 | 0.023142   | 11.83x  | -91.55%        |
| lighten_only  | scalar | 0.204830 | 0.056793   | 3.61x   | -72.27%        |
| lighten_only  | sse42  | 0.204830 | 0.023721   | 8.63x   | -88.42%        |
| lighten_only  | avx2   | 0.204830 | 0.023265   | 8.80x   | -88.64%        |
| screen        | scalar | 0.219813 | 0.051124   | 4.30x   | -76.74%        |
| screen        | sse42  | 0.219813 | 0.023636   | 9.30x   | -89.25%        |
| screen        | avx2   | 0.219813 | 0.022663   | 9.70x   | -89.69%        |
| dodge         | scalar | 0.223351 | 0.055607   | 4.02x   | -75.10%        |
| dodge         | sse42  | 0.223351 | 0.024499   | 9.12x   | -89.03%        |
| dodge         | avx2   | 0.223351 | 0.023496   | 9.51x   | -89.48%        |
| addition      | scalar | 0.216508 | 0.080848   | 2.68x   | -62.66%        |
| addition      | sse42  | 0.216508 | 0.024711   | 8.76x   | -88.59%        |
| addition      | avx2   | 0.216508 | 0.023335   | 9.28x   | -89.22%        |
| darken_only   | scalar | 0.205050 | 0.055595   | 3.69x   | -72.89%        |
| darken_only   | sse42  | 0.205050 | 0.023619   | 8.68x   | -88.48%        |
| darken_only   | avx2   | 0.205050 | 0.022744   | 9.02x   | -88.91%        |
| multiply      | scalar | 0.209459 | 0.049581   | 4.22x   | -76.33%        |
| multiply      | sse42  | 0.209459 | 0.023948   | 8.75x   | -88.57%        |
| multiply      | avx2   | 0.209459 | 0.023080   | 9.08x   | -88.98%        |
| hard_light    | scalar | 0.307069 | 0.098995   | 3.10x   | -67.76%        |
| hard_light    | sse42  | 0.307069 | 0.024433   | 12.57x  | -92.04%        |
| hard_light    | avx2   | 0.307069 | 0.023364   | 13.14x  | -92.39%        |
| difference    | scalar | 0.278796 | 0.050392   | 5.53x   | -81.92%        |
| difference    | sse42  | 0.278796 | 0.023629   | 11.80x  | -91.52%        |
| difference    | avx2   | 0.278796 | 0.022940   | 12.15x  | -91.77%        |
| subtract      | scalar | 0.210091 | 0.054978   | 3.82x   | -73.83%        |
| subtract      | sse42  | 0.210091 | 0.024745   | 8.49x   | -88.22%        |
| subtract      | avx2   | 0.210091 | 0.023053   | 9.11x   | -89.03%        |
| grain_extract | scalar | 0.213886 | 0.065259   | 3.28x   | -69.49%        |
| grain_extract | sse42  | 0.213886 | 0.023802   | 8.99x   | -88.87%        |
| grain_extract | avx2   | 0.213886 | 0.023165   | 9.23x   | -89.17%        |
| grain_merge   | scalar | 0.213529 | 0.066217   | 3.22x   | -68.99%        |
| grain_merge   | sse42  | 0.213529 | 0.024020   | 8.89x   | -88.75%        |
| grain_merge   | avx2   | 0.213529 | 0.022937   | 9.31x   | -89.26%        |
| divide        | scalar | 0.216148 | 0.053688   | 4.03x   | -75.16%        |
| divide        | sse42  | 0.216148 | 0.023968   | 9.02x   | -88.91%        |
| divide        | avx2   | 0.216148 | 0.023376   | 9.25x   | -89.19%        |
| overlay       | scalar | 0.288172 | 0.093346   | 3.09x   | -67.61%        |
| overlay       | sse42  | 0.288172 | 0.024282   | 11.87x  | -91.57%        |
| overlay       | avx2   | 0.288172 | 0.023223   | 12.41x  | -91.94%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.010638 | 0.001683   | 6.32x   | -84.18%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.010638 | 0.001371   | 7.76x   | -87.11%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.010638 | 0.001817   | 5.85x   | -82.92%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.010655 | 0.001913   | 5.57x   | -82.05%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.010655 | 0.001420   | 7.50x   | -86.67%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.010655 | 0.001961   | 5.43x   | -81.60%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.009153 | 0.001959   | 4.67x   | -78.60%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.009153 | 0.001466   | 6.24x   | -83.98%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.009153 | 0.001463   | 6.26x   | -84.02%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.010116 | 0.001975   | 5.12x   | -80.48%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.010116 | 0.001530   | 6.61x   | -84.87%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.010116 | 0.001503   | 6.73x   | -85.14%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.010548 | 0.001901   | 5.55x   | -81.98%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.010548 | 0.001425   | 7.40x   | -86.49%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.010548 | 0.001506   | 7.00x   | -85.72%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.009860 | 0.002661   | 3.70x   | -73.01%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.009860 | 0.001401   | 7.04x   | -85.79%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.009860 | 0.001495   | 6.59x   | -84.84%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.009394 | 0.001915   | 4.90x   | -79.61%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.009394 | 0.001395   | 6.73x   | -85.15%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.009394 | 0.001465   | 6.41x   | -84.40%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.009586 | 0.001764   | 5.44x   | -81.60%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.009586 | 0.001412   | 6.79x   | -85.27%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.009586 | 0.001475   | 6.50x   | -84.61%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.011080 | 0.003057   | 3.62x   | -72.41%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.011080 | 0.001401   | 7.91x   | -87.35%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.011080 | 0.001477   | 7.50x   | -86.67%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.011562 | 0.001745   | 6.63x   | -84.91%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.011562 | 0.001384   | 8.36x   | -88.03%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.011562 | 0.001499   | 7.71x   | -87.03%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.009477 | 0.001854   | 5.11x   | -80.43%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.009477 | 0.001404   | 6.75x   | -85.19%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.009477 | 0.001478   | 6.41x   | -84.41%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.009425 | 0.002168   | 4.35x   | -77.00%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.009425 | 0.001398   | 6.74x   | -85.16%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.009425 | 0.001466   | 6.43x   | -84.45%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.009519 | 0.002165   | 4.40x   | -77.25%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.009519 | 0.001398   | 6.81x   | -85.32%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.009519 | 0.001464   | 6.50x   | -84.62%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.009467 | 0.001831   | 5.17x   | -80.66%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.009467 | 0.001396   | 6.78x   | -85.25%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.009467 | 0.001663   | 5.69x   | -82.44%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.011032 | 0.002918   | 3.78x   | -73.55%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.011032 | 0.001411   | 7.82x   | -87.21%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.011032 | 0.001477   | 7.47x   | -86.61%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.005568 | 0.001318   | 4.22x   | -76.33%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.005568 | 0.000193   | 28.80x  | -96.53%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.005568 | 0.000160   | 34.83x  | -97.13%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.008679 | 0.001692   | 5.13x   | -80.50%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.008679 | 0.000231   | 37.50x  | -97.33%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.008679 | 0.000196   | 44.23x  | -97.74%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.007031 | 0.001689   | 4.16x   | -75.97%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.007031 | 0.000219   | 32.11x  | -96.89%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.007031 | 0.000190   | 37.00x  | -97.30%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.007067 | 0.001658   | 4.26x   | -76.54%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.007067 | 0.000224   | 31.50x  | -96.83%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.007067 | 0.000191   | 36.93x  | -97.29%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.007896 | 0.001880   | 4.20x   | -76.18%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.007896 | 0.000250   | 31.62x  | -96.84%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.007896 | 0.000199   | 39.70x  | -97.48%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.007655 | 0.002091   | 3.66x   | -72.68%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.007655 | 0.000273   | 28.01x  | -96.43%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.007655 | 0.000202   | 37.94x  | -97.36%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.007205 | 0.001762   | 4.09x   | -75.54%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.007205 | 0.000224   | 32.16x  | -96.89%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.007205 | 0.000191   | 37.65x  | -97.34%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.007435 | 0.001732   | 4.29x   | -76.70%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.007435 | 0.000220   | 33.73x  | -97.04%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.007435 | 0.000195   | 38.21x  | -97.38%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.009368 | 0.002758   | 3.40x   | -70.56%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.009368 | 0.000260   | 35.96x  | -97.22%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.009368 | 0.000195   | 47.93x  | -97.91%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.009994 | 0.001633   | 6.12x   | -83.66%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.009994 | 0.000220   | 45.41x  | -97.80%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.009994 | 0.000192   | 51.93x  | -98.07%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.008146 | 0.001711   | 4.76x   | -79.00%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.008146 | 0.000280   | 29.13x  | -96.57%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.008146 | 0.000202   | 40.32x  | -97.52%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.008231 | 0.002033   | 4.05x   | -75.30%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.008231 | 0.000230   | 35.84x  | -97.21%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.008231 | 0.000192   | 42.93x  | -97.67%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.007370 | 0.002064   | 3.57x   | -71.99%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.007370 | 0.000228   | 32.26x  | -96.90%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.007370 | 0.000192   | 38.42x  | -97.40%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.007380 | 0.001754   | 4.21x   | -76.23%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.007380 | 0.000243   | 30.42x  | -96.71%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.007380 | 0.000191   | 38.63x  | -97.41%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.008448 | 0.002592   | 3.26x   | -69.32%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.008448 | 0.000241   | 35.09x  | -97.15%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.008448 | 0.000193   | 43.78x  | -97.72%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.007591 | 0.000584   | 13.00x  | -92.31%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.007591 | 0.000170   | 44.74x  | -97.76%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.007591 | 0.000179   | 42.45x  | -97.64%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.010407 | 0.000736   | 14.14x  | -92.93%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.010407 | 0.000244   | 42.56x  | -97.65%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.010407 | 0.000206   | 50.43x  | -98.02%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.008792 | 0.000795   | 11.06x  | -90.96%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.008792 | 0.000218   | 40.28x  | -97.52%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.008792 | 0.000193   | 45.62x  | -97.81%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.010295 | 0.000742   | 13.87x  | -92.79%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.010295 | 0.000244   | 42.26x  | -97.63%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.010295 | 0.000215   | 47.93x  | -97.91%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.011703 | 0.000756   | 15.49x  | -93.54%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.011703 | 0.000278   | 42.05x  | -97.62%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.011703 | 0.000229   | 51.20x  | -98.05%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.010126 | 0.001726   | 5.87x   | -82.95%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.010126 | 0.000280   | 36.15x  | -97.23%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.010126 | 0.000221   | 45.80x  | -97.82%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.009900 | 0.000880   | 11.25x  | -91.11%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.009900 | 0.000226   | 43.74x  | -97.71%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.009900 | 0.000196   | 50.55x  | -98.02%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.009765 | 0.000671   | 14.55x  | -93.13%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.009765 | 0.000263   | 37.15x  | -97.31%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.009765 | 0.000198   | 49.29x  | -97.97%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.011243 | 0.001990   | 5.65x   | -82.30%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.011243 | 0.000275   | 40.88x  | -97.55%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.011243 | 0.000217   | 51.72x  | -98.07%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.011116 | 0.000657   | 16.92x  | -94.09%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.011116 | 0.000234   | 47.52x  | -97.90%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.011116 | 0.000197   | 56.51x  | -98.23%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.009562 | 0.000817   | 11.70x  | -91.45%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.009562 | 0.000281   | 34.06x  | -97.06%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.009562 | 0.000210   | 45.60x  | -97.81%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.009509 | 0.001122   | 8.47x   | -88.20%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.009509 | 0.000254   | 37.42x  | -97.33%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.009509 | 0.000207   | 45.94x  | -97.82%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.011010 | 0.001492   | 7.38x   | -86.45%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.011010 | 0.000347   | 31.76x  | -96.85%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.011010 | 0.000356   | 30.94x  | -96.77%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.012540 | 0.000735   | 17.07x  | -94.14%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.012540 | 0.000268   | 46.75x  | -97.86%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.012540 | 0.000277   | 45.23x  | -97.79%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.012654 | 0.001922   | 6.58x   | -84.81%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.012654 | 0.000275   | 46.06x  | -97.83%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.012654 | 0.000224   | 56.48x  | -98.23%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.005931 | 0.000749   | 7.92x   | -87.37%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.005931 | 0.000328   | 18.09x  | -94.47%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.005931 | 0.000266   | 22.29x  | -95.51%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.008858 | 0.000926   | 9.57x   | -89.55%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.008858 | 0.000385   | 23.03x  | -95.66%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.008858 | 0.000285   | 31.13x  | -96.79%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.007012 | 0.001003   | 6.99x   | -85.70%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.007012 | 0.000391   | 17.95x  | -94.43%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.007012 | 0.000277   | 25.32x  | -96.05%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.007445 | 0.000910   | 8.18x   | -87.77%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.007445 | 0.000389   | 19.13x  | -94.77%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.007445 | 0.000292   | 25.54x  | -96.08%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.007484 | 0.000967   | 7.74x   | -87.08%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.007484 | 0.000412   | 18.15x  | -94.49%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.007484 | 0.000285   | 26.25x  | -96.19%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.007124 | 0.001635   | 4.36x   | -77.05%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.007124 | 0.000402   | 17.74x  | -94.36%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.007124 | 0.000294   | 24.25x  | -95.88%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.007226 | 0.001001   | 7.22x   | -86.15%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.007226 | 0.000400   | 18.06x  | -94.46%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.007226 | 0.000285   | 25.37x  | -96.06%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.007639 | 0.000830   | 9.21x   | -89.14%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.007639 | 0.000385   | 19.85x  | -94.96%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.007639 | 0.000286   | 26.72x  | -96.26%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.009203 | 0.002107   | 4.37x   | -77.11%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.009203 | 0.000390   | 23.61x  | -95.76%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.009203 | 0.000286   | 32.21x  | -96.90%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.009624 | 0.000840   | 11.46x  | -91.27%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.009624 | 0.000389   | 24.77x  | -95.96%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.009624 | 0.000279   | 34.47x  | -97.10%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.007183 | 0.001070   | 6.71x   | -85.11%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.007183 | 0.000402   | 17.86x  | -94.40%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.007183 | 0.000325   | 22.11x  | -95.48%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.006700 | 0.001241   | 5.40x   | -81.49%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.006700 | 0.000390   | 17.19x  | -94.18%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.006700 | 0.000314   | 21.35x  | -95.32%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.006410 | 0.001223   | 5.24x   | -80.93%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.006410 | 0.000362   | 17.72x  | -94.36%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.006410 | 0.000261   | 24.58x  | -95.93%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.006364 | 0.000889   | 7.16x   | -86.03%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.006364 | 0.000368   | 17.29x  | -94.22%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.006364 | 0.000260   | 24.46x  | -95.91%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.007827 | 0.002290   | 3.42x   | -70.74%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.007827 | 0.000356   | 22.00x  | -95.46%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.007827 | 0.000267   | 29.35x  | -96.59%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.034165 | 0.006414   | 5.33x   | -81.23%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.034165 | 0.005478   | 6.24x   | -83.97%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.034165 | 0.005751   | 5.94x   | -83.17%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.044207 | 0.007524   | 5.88x   | -82.98%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.044207 | 0.005713   | 7.74x   | -87.08%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.044207 | 0.005941   | 7.44x   | -86.56%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.037609 | 0.007564   | 4.97x   | -79.89%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.037609 | 0.005696   | 6.60x   | -84.85%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.037609 | 0.005777   | 6.51x   | -84.64%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.039398 | 0.006918   | 5.70x   | -82.44%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.039398 | 0.005648   | 6.98x   | -85.67%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.039398 | 0.005820   | 6.77x   | -85.23%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.037770 | 0.007323   | 5.16x   | -80.61%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.037770 | 0.005662   | 6.67x   | -85.01%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.037770 | 0.005978   | 6.32x   | -84.17%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.037249 | 0.010507   | 3.54x   | -71.79%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.037249 | 0.005533   | 6.73x   | -85.15%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.037249 | 0.005926   | 6.29x   | -84.09%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.037969 | 0.008043   | 4.72x   | -78.82%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.037969 | 0.005715   | 6.64x   | -84.95%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.037969 | 0.005944   | 6.39x   | -84.35%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.038645 | 0.007110   | 5.44x   | -81.60%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.038645 | 0.005637   | 6.86x   | -85.41%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.038645 | 0.005871   | 6.58x   | -84.81%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.048580 | 0.012250   | 3.97x   | -74.78%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.048580 | 0.005632   | 8.63x   | -88.41%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.048580 | 0.005914   | 8.21x   | -87.83%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.044556 | 0.006918   | 6.44x   | -84.47%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.044556 | 0.005584   | 7.98x   | -87.47%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.044556 | 0.005757   | 7.74x   | -87.08%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.037672 | 0.007049   | 5.34x   | -81.29%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.037672 | 0.005784   | 6.51x   | -84.65%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.037672 | 0.006079   | 6.20x   | -83.86%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.039984 | 0.008970   | 4.46x   | -77.57%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.039984 | 0.005718   | 6.99x   | -85.70%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.039984 | 0.005966   | 6.70x   | -85.08%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.038684 | 0.008834   | 4.38x   | -77.16%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.038684 | 0.005751   | 6.73x   | -85.13%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.038684 | 0.006074   | 6.37x   | -84.30%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.041041 | 0.007720   | 5.32x   | -81.19%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.041041 | 0.005685   | 7.22x   | -86.15%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.041041 | 0.006040   | 6.79x   | -85.28%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.046477 | 0.012221   | 3.80x   | -73.71%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.046477 | 0.005703   | 8.15x   | -87.73%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.046477 | 0.006085   | 7.64x   | -86.91%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.026423 | 0.005416   | 4.88x   | -79.50%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.026423 | 0.000789   | 33.48x  | -97.01%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.026423 | 0.000623   | 42.44x  | -97.64%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.040144 | 0.008253   | 4.86x   | -79.44%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.040144 | 0.001149   | 34.93x  | -97.14%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.040144 | 0.000805   | 49.88x  | -98.00%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.031894 | 0.006962   | 4.58x   | -78.17%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.031894 | 0.000885   | 36.02x  | -97.22%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.031894 | 0.000752   | 42.42x  | -97.64%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.030747 | 0.006923   | 4.44x   | -77.49%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.030747 | 0.000917   | 33.55x  | -97.02%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.030747 | 0.000762   | 40.33x  | -97.52%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.031178 | 0.007015   | 4.44x   | -77.50%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.031178 | 0.000988   | 31.56x  | -96.83%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.031178 | 0.000782   | 39.86x  | -97.49%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.029003 | 0.008376   | 3.46x   | -71.12%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.029003 | 0.001093   | 26.52x  | -96.23%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.029003 | 0.000780   | 37.19x  | -97.31%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.028816 | 0.006951   | 4.15x   | -75.88%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.028816 | 0.000872   | 33.03x  | -96.97%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.028816 | 0.000752   | 38.33x  | -97.39%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.029032 | 0.006763   | 4.29x   | -76.71%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.029032 | 0.000890   | 32.64x  | -96.94%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.029032 | 0.000767   | 37.84x  | -97.36%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.038191 | 0.011041   | 3.46x   | -71.09%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.038191 | 0.001011   | 37.76x  | -97.35%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.038191 | 0.000778   | 49.11x  | -97.96%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.035886 | 0.006391   | 5.62x   | -82.19%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.035886 | 0.000883   | 40.62x  | -97.54%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.035886 | 0.000756   | 47.48x  | -97.89%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.028658 | 0.006456   | 4.44x   | -77.47%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.028658 | 0.001097   | 26.14x  | -96.17%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.028658 | 0.000849   | 33.77x  | -97.04%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.029260 | 0.007861   | 3.72x   | -73.13%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.029260 | 0.000903   | 32.41x  | -96.91%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.029260 | 0.000749   | 39.08x  | -97.44%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.029982 | 0.007875   | 3.81x   | -73.73%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.029982 | 0.000928   | 32.31x  | -96.90%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.029982 | 0.000765   | 39.18x  | -97.45%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.033115 | 0.007368   | 4.49x   | -77.75%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.033115 | 0.000966   | 34.27x  | -97.08%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.033115 | 0.000774   | 42.80x  | -97.66%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.038015 | 0.010687   | 3.56x   | -71.89%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.038015 | 0.001015   | 37.44x  | -97.33%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.038015 | 0.000771   | 49.32x  | -97.97%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.029180 | 0.002106   | 13.86x  | -92.78%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.029180 | 0.000737   | 39.60x  | -97.47%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.029180 | 0.000815   | 35.81x  | -97.21%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.041331 | 0.002785   | 14.84x  | -93.26%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.041331 | 0.001051   | 39.34x  | -97.46%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.041331 | 0.000859   | 48.09x  | -97.92%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.033124 | 0.002925   | 11.32x  | -91.17%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.033124 | 0.000993   | 33.36x  | -97.00%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.033124 | 0.000909   | 36.43x  | -97.25%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.035164 | 0.002470   | 14.24x  | -92.98%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.035164 | 0.000953   | 36.90x  | -97.29%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.035164 | 0.000806   | 43.65x  | -97.71%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.033406 | 0.002729   | 12.24x  | -91.83%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.033406 | 0.001115   | 29.96x  | -96.66%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.033406 | 0.000897   | 37.24x  | -97.31%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.034219 | 0.006649   | 5.15x   | -80.57%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.034219 | 0.001092   | 31.33x  | -96.81%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.034219 | 0.000863   | 39.65x  | -97.48%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.035267 | 0.002951   | 11.95x  | -91.63%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.035267 | 0.000924   | 38.17x  | -97.38%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.035267 | 0.000825   | 42.75x  | -97.66%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.033562 | 0.002407   | 13.95x  | -92.83%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.033562 | 0.000882   | 38.05x  | -97.37%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.033562 | 0.000813   | 41.26x  | -97.58%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.041908 | 0.007696   | 5.45x   | -81.64%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.041908 | 0.001067   | 39.27x  | -97.45%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.041908 | 0.000862   | 48.63x  | -97.94%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.040043 | 0.002546   | 15.73x  | -93.64%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.040043 | 0.000944   | 42.40x  | -97.64%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.040043 | 0.000829   | 48.31x  | -97.93%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.032464 | 0.003010   | 10.78x  | -90.73%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.032464 | 0.001103   | 29.44x  | -96.60%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.032464 | 0.000831   | 39.07x  | -97.44%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.032635 | 0.004197   | 7.78x   | -87.14%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.032635 | 0.000994   | 32.83x  | -96.95%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.032635 | 0.000825   | 39.54x  | -97.47%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.032612 | 0.004218   | 7.73x   | -87.07%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.032612 | 0.001056   | 30.88x  | -96.76%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.032612 | 0.000854   | 38.20x  | -97.38%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.034173 | 0.002655   | 12.87x  | -92.23%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.034173 | 0.000997   | 34.28x  | -97.08%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.034173 | 0.000871   | 39.25x  | -97.45%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.039872 | 0.007195   | 5.54x   | -81.95%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.039872 | 0.001034   | 38.55x  | -97.41%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.039872 | 0.000823   | 48.47x  | -97.94%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.020830 | 0.002527   | 8.24x   | -87.87%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.020830 | 0.001248   | 16.69x  | -94.01%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.020830 | 0.001051   | 19.81x  | -94.95%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.032728 | 0.003299   | 9.92x   | -89.92%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.032728 | 0.001468   | 22.29x  | -95.51%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.032728 | 0.001061   | 30.84x  | -96.76%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.026681 | 0.003613   | 7.39x   | -86.46%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.026681 | 0.001538   | 17.34x  | -94.23%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.026681 | 0.001065   | 25.05x  | -96.01%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.026742 | 0.003151   | 8.49x   | -88.22%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.026742 | 0.001471   | 18.18x  | -94.50%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.026742 | 0.001081   | 24.74x  | -95.96%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.026364 | 0.003366   | 7.83x   | -87.23%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.026364 | 0.001475   | 17.87x  | -94.40%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.026364 | 0.001066   | 24.72x  | -95.95%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.025673 | 0.005683   | 4.52x   | -77.87%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.025673 | 0.001565   | 16.41x  | -93.91%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.025673 | 0.001133   | 22.67x  | -95.59%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.025564 | 0.003597   | 7.11x   | -85.93%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.025564 | 0.001483   | 17.24x  | -94.20%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.025564 | 0.001056   | 24.20x  | -95.87%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.025311 | 0.002876   | 8.80x   | -88.64%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.025311 | 0.001514   | 16.72x  | -94.02%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.025311 | 0.001079   | 23.45x  | -95.74%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.034395 | 0.007693   | 4.47x   | -77.63%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.034395 | 0.001437   | 23.94x  | -95.82%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.034395 | 0.001100   | 31.27x  | -96.80%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.032749 | 0.002927   | 11.19x  | -91.06%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.032749 | 0.001502   | 21.80x  | -95.41%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.032749 | 0.001072   | 30.54x  | -96.73%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.024628 | 0.003940   | 6.25x   | -84.00%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.024628 | 0.001537   | 16.02x  | -93.76%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.024628 | 0.001086   | 22.67x  | -95.59%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.025892 | 0.004574   | 5.66x   | -82.33%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.025892 | 0.001473   | 17.57x  | -94.31%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.025892 | 0.001083   | 23.92x  | -95.82%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.026416 | 0.004646   | 5.69x   | -82.41%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.026416 | 0.001495   | 17.67x  | -94.34%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.026416 | 0.001093   | 24.18x  | -95.86%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.026623 | 0.003482   | 7.65x   | -86.92%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.026623 | 0.001556   | 17.11x  | -94.16%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.026623 | 0.001250   | 21.30x  | -95.31%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.033609 | 0.007346   | 4.58x   | -78.14%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.033609 | 0.001427   | 23.55x  | -95.75%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.033609 | 0.001115   | 30.13x  | -96.68%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.097679 | 0.026634   | 3.67x   | -72.73%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.097679 | 0.021750   | 4.49x   | -77.73%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.097679 | 0.023059   | 4.24x   | -76.39%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.134074 | 0.031639   | 4.24x   | -76.40%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.134074 | 0.023129   | 5.80x   | -82.75%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.134074 | 0.023773   | 5.64x   | -82.27%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.105168 | 0.031170   | 3.37x   | -70.36%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.105168 | 0.022263   | 4.72x   | -78.83%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.105168 | 0.023270   | 4.52x   | -77.87%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.120501 | 0.029318   | 4.11x   | -75.67%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.120501 | 0.022406   | 5.38x   | -81.41%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.120501 | 0.023942   | 5.03x   | -80.13%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.106121 | 0.029958   | 3.54x   | -71.77%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.106121 | 0.022693   | 4.68x   | -78.62%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.106121 | 0.024299   | 4.37x   | -77.10%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.107271 | 0.042974   | 2.50x   | -59.94%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.107271 | 0.022151   | 4.84x   | -79.35%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.107271 | 0.023550   | 4.56x   | -78.05%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.103640 | 0.031246   | 3.32x   | -69.85%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.103640 | 0.022419   | 4.62x   | -78.37%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.103640 | 0.023357   | 4.44x   | -77.46%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.103856 | 0.030139   | 3.45x   | -70.98%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.103856 | 0.022517   | 4.61x   | -78.32%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.103856 | 0.025059   | 4.14x   | -75.87%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.159894 | 0.051575   | 3.10x   | -67.74%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.159894 | 0.022631   | 7.07x   | -85.85%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.159894 | 0.024020   | 6.66x   | -84.98%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.148088 | 0.028637   | 5.17x   | -80.66%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.148088 | 0.022282   | 6.65x   | -84.95%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.148088 | 0.026182   | 5.66x   | -82.32%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.131463 | 0.032596   | 4.03x   | -75.21%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.131463 | 0.025555   | 5.14x   | -80.56%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.131463 | 0.024063   | 5.46x   | -81.70%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.125303 | 0.037595   | 3.33x   | -70.00%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.125303 | 0.025006   | 5.01x   | -80.04%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.125303 | 0.029501   | 4.25x   | -76.46%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.113800 | 0.040462   | 2.81x   | -64.44%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.113800 | 0.023947   | 4.75x   | -78.96%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.113800 | 0.024116   | 4.72x   | -78.81%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.128623 | 0.047488   | 2.71x   | -63.08%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.128623 | 0.024268   | 5.30x   | -81.13%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.128623 | 0.024389   | 5.27x   | -81.04%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.135595 | 0.047128   | 2.88x   | -65.24%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.135595 | 0.023149   | 5.86x   | -82.93%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.135595 | 0.024487   | 5.54x   | -81.94%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.076558 | 0.021874   | 3.50x   | -71.43%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.076558 | 0.003099   | 24.70x  | -95.95%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.076558 | 0.002474   | 30.94x  | -96.77%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.107318 | 0.027375   | 3.92x   | -74.49%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.107318 | 0.003683   | 29.14x  | -96.57%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.107318 | 0.003135   | 34.23x  | -97.08%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.085436 | 0.028522   | 3.00x   | -66.62%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.085436 | 0.003734   | 22.88x  | -95.63%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.085436 | 0.003188   | 26.80x  | -96.27%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.091240 | 0.028264   | 3.23x   | -69.02%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.091240 | 0.003739   | 24.40x  | -95.90%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.091240 | 0.003101   | 29.42x  | -96.60%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.087471 | 0.035864   | 2.44x   | -59.00%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.087471 | 0.004070   | 21.49x  | -95.35%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.087471 | 0.003133   | 27.92x  | -96.42%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.089300 | 0.034689   | 2.57x   | -61.16%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.089300 | 0.004407   | 20.26x  | -95.06%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.089300 | 0.003217   | 27.76x  | -96.40%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.083927 | 0.027889   | 3.01x   | -66.77%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.083927 | 0.003554   | 23.62x  | -95.77%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.083927 | 0.003034   | 27.66x  | -96.38%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.098995 | 0.029813   | 3.32x   | -69.88%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.098995 | 0.004559   | 21.71x  | -95.39%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.098995 | 0.003734   | 26.51x  | -96.23%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.124434 | 0.053201   | 2.34x   | -57.25%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.124434 | 0.004162   | 29.90x  | -96.66%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.124434 | 0.003184   | 39.09x  | -97.44%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.127435 | 0.031948   | 3.99x   | -74.93%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.127435 | 0.003689   | 34.54x  | -97.11%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.127435 | 0.003040   | 41.92x  | -97.61%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.088273 | 0.028738   | 3.07x   | -67.44%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.088273 | 0.004754   | 18.57x  | -94.61%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.088273 | 0.003580   | 24.66x  | -95.94%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.088177 | 0.033281   | 2.65x   | -62.26%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.088177 | 0.003749   | 23.52x  | -95.75%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.088177 | 0.003099   | 28.45x  | -96.49%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.096732 | 0.035106   | 2.76x   | -63.71%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.096732 | 0.003761   | 25.72x  | -96.11%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.096732 | 0.003908   | 24.75x  | -95.96%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.114667 | 0.040030   | 2.86x   | -65.09%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.114667 | 0.004078   | 28.12x  | -96.44%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.114667 | 0.003255   | 35.23x  | -97.16%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.118328 | 0.043037   | 2.75x   | -63.63%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.118328 | 0.004001   | 29.58x  | -96.62%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.118328 | 0.003128   | 37.83x  | -97.36%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.089912 | 0.009171   | 9.80x   | -89.80%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.089912 | 0.002970   | 30.27x  | -96.70%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.089912 | 0.003426   | 26.25x  | -96.19%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.135344 | 0.011709   | 11.56x  | -91.35%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.135344 | 0.004283   | 31.60x  | -96.84%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.135344 | 0.004121   | 32.84x  | -96.96%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.106496 | 0.013839   | 7.70x   | -87.01%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.106496 | 0.005586   | 19.07x  | -94.75%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.106496 | 0.003630   | 29.33x  | -96.59%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.112116 | 0.010729   | 10.45x  | -90.43%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.112116 | 0.004353   | 25.75x  | -96.12%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.112116 | 0.003461   | 32.40x  | -96.91%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.100186 | 0.011885   | 8.43x   | -88.14%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.100186 | 0.004487   | 22.33x  | -95.52%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.100186 | 0.003763   | 26.62x  | -96.24%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.099359 | 0.026664   | 3.73x   | -73.16%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.099359 | 0.004417   | 22.50x  | -95.55%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.099359 | 0.003496   | 28.42x  | -96.48%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.095126 | 0.012850   | 7.40x   | -86.49%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.095126 | 0.003724   | 25.55x  | -96.09%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.095126 | 0.004761   | 19.98x  | -95.00%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.097465 | 0.010006   | 9.74x   | -89.73%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.097465 | 0.003857   | 25.27x  | -96.04%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.097465 | 0.003382   | 28.82x  | -96.53%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.135989 | 0.031595   | 4.30x   | -76.77%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.135989 | 0.004259   | 31.93x  | -96.87%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.135989 | 0.003568   | 38.11x  | -97.38%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.131473 | 0.011220   | 11.72x  | -91.47%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.131473 | 0.004423   | 29.72x  | -96.64%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.131473 | 0.003532   | 37.22x  | -97.31%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.106258 | 0.016555   | 6.42x   | -84.42%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.106258 | 0.004768   | 22.29x  | -95.51%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.106258 | 0.004788   | 22.19x  | -95.49%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.117656 | 0.018248   | 6.45x   | -84.49%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.117656 | 0.005889   | 19.98x  | -94.99%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.117656 | 0.003534   | 33.29x  | -97.00%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.117437 | 0.017825   | 6.59x   | -84.82%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.117437 | 0.004088   | 28.73x  | -96.52%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.117437 | 0.003408   | 34.45x  | -97.10%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.103653 | 0.011374   | 9.11x   | -89.03%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.103653 | 0.004139   | 25.05x  | -96.01%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.103653 | 0.003585   | 28.91x  | -96.54%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.139905 | 0.029599   | 4.73x   | -78.84%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.139905 | 0.004078   | 34.31x  | -97.09%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.139905 | 0.003509   | 39.87x  | -97.49%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.066586 | 0.010787   | 6.17x   | -83.80%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.066586 | 0.004916   | 13.54x  | -92.62%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.066586 | 0.003826   | 17.40x  | -94.25%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.097550 | 0.013817   | 7.06x   | -85.84%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.097550 | 0.005799   | 16.82x  | -94.06%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.097550 | 0.004549   | 21.44x  | -95.34%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.070332 | 0.015408   | 4.56x   | -78.09%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.070332 | 0.005913   | 11.89x  | -91.59%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.070332 | 0.004333   | 16.23x  | -93.84%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.072999 | 0.013273   | 5.50x   | -81.82%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.072999 | 0.005776   | 12.64x  | -92.09%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.072999 | 0.004412   | 16.55x  | -93.96%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.072455 | 0.015422   | 4.70x   | -78.72%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.072455 | 0.006024   | 12.03x  | -91.69%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.072455 | 0.004728   | 15.32x  | -93.47%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.070592 | 0.024757   | 2.85x   | -64.93%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.070592 | 0.006457   | 10.93x  | -90.85%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.070592 | 0.004492   | 15.71x  | -93.64%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.070287 | 0.015197   | 4.63x   | -78.38%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.070287 | 0.005942   | 11.83x  | -91.55%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.070287 | 0.004396   | 15.99x  | -93.75%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.072907 | 0.012362   | 5.90x   | -83.04%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.072907 | 0.006091   | 11.97x  | -91.65%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.072907 | 0.004462   | 16.34x  | -93.88%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.110646 | 0.032473   | 3.41x   | -70.65%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.110646 | 0.005756   | 19.22x  | -94.80%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.110646 | 0.004427   | 25.00x  | -96.00%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.101877 | 0.012775   | 7.97x   | -87.46%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.101877 | 0.006135   | 16.61x  | -93.98%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.101877 | 0.004531   | 22.49x  | -95.55%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.075319 | 0.018243   | 4.13x   | -75.78%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.075319 | 0.006639   | 11.34x  | -91.18%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.075319 | 0.005014   | 15.02x  | -93.34%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.083085 | 0.022348   | 3.72x   | -73.10%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.083085 | 0.006307   | 13.17x  | -92.41%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.083085 | 0.005211   | 15.94x  | -93.73%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.080701 | 0.020868   | 3.87x   | -74.14%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.080701 | 0.005960   | 13.54x  | -92.61%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.080701 | 0.004783   | 16.87x  | -94.07%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.081080 | 0.014573   | 5.56x   | -82.03%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.081080 | 0.006315   | 12.84x  | -92.21%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.081080 | 0.004790   | 16.93x  | -94.09%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.120235 | 0.032140   | 3.74x   | -73.27%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.120235 | 0.006053   | 19.86x  | -94.97%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.120235 | 0.004443   | 27.06x  | -96.30%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.513658 | 0.119747   | 4.29x   | -76.69%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.513658 | 0.091506   | 5.61x   | -82.19%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.513658 | 0.110305   | 4.66x   | -78.53%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.545103 | 0.145559   | 3.74x   | -73.30%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.545103 | 0.097290   | 5.60x   | -82.15%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.545103 | 0.096924   | 5.62x   | -82.22%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.404875 | 0.126728   | 3.19x   | -68.70%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.404875 | 0.091899   | 4.41x   | -77.30%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.404875 | 0.106138   | 3.81x   | -73.79%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.424464 | 0.117717   | 3.61x   | -72.27%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.424464 | 0.092861   | 4.57x   | -78.12%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.424464 | 0.097453   | 4.36x   | -77.04%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.430221 | 0.120405   | 3.57x   | -72.01%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.430221 | 0.093895   | 4.58x   | -78.18%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.430221 | 0.095278   | 4.52x   | -77.85%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.412100 | 0.175731   | 2.35x   | -57.36%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.412100 | 0.095051   | 4.34x   | -76.93%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.412100 | 0.101704   | 4.05x   | -75.32%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.437926 | 0.124763   | 3.51x   | -71.51%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.437926 | 0.092359   | 4.74x   | -78.91%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.437926 | 0.094756   | 4.62x   | -78.36%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.408945 | 0.117494   | 3.48x   | -71.27%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.408945 | 0.093971   | 4.35x   | -77.02%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.408945 | 0.099067   | 4.13x   | -75.77%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.611484 | 0.203005   | 3.01x   | -66.80%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.611484 | 0.092707   | 6.60x   | -84.84%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.611484 | 0.096361   | 6.35x   | -84.24%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.534587 | 0.121072   | 4.42x   | -77.35%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.534587 | 0.098470   | 5.43x   | -81.58%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.534587 | 0.099103   | 5.39x   | -81.46%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.426619 | 0.118075   | 3.61x   | -72.32%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.426619 | 0.089800   | 4.75x   | -78.95%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.426619 | 0.094279   | 4.53x   | -77.90%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.385825 | 0.141194   | 2.73x   | -63.40%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.385825 | 0.089524   | 4.31x   | -76.80%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.385825 | 0.097045   | 3.98x   | -74.85%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.378516 | 0.139635   | 2.71x   | -63.11%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.378516 | 0.089344   | 4.24x   | -76.40%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.378516 | 0.096786   | 3.91x   | -74.43%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.392311 | 0.119954   | 3.27x   | -69.42%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.392311 | 0.089375   | 4.39x   | -77.22%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.392311 | 0.097193   | 4.04x   | -75.23%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.508685 | 0.191892   | 2.65x   | -62.28%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.508685 | 0.094242   | 5.40x   | -81.47%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.508685 | 0.095264   | 5.34x   | -81.27%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.298034 | 0.088892   | 3.35x   | -70.17%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.298034 | 0.012244   | 24.34x  | -95.89%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.298034 | 0.009833   | 30.31x  | -96.70%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.390080 | 0.108289   | 3.60x   | -72.24%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.390080 | 0.015198   | 25.67x  | -96.10%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.390080 | 0.012196   | 31.99x  | -96.87%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.276003 | 0.115557   | 2.39x   | -58.13%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.276003 | 0.014014   | 19.70x  | -94.92%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.276003 | 0.012503   | 22.08x  | -95.47%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.311924 | 0.109945   | 2.84x   | -64.75%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.311924 | 0.014467   | 21.56x  | -95.36%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.311924 | 0.012949   | 24.09x  | -95.85%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.361405 | 0.116658   | 3.10x   | -67.72%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.361405 | 0.015723   | 22.99x  | -95.65%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.361405 | 0.012486   | 28.94x  | -96.55%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.302618 | 0.139691   | 2.17x   | -53.84%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.302618 | 0.017443   | 17.35x  | -94.24%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.302618 | 0.013934   | 21.72x  | -95.40%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.289491 | 0.112580   | 2.57x   | -61.11%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.289491 | 0.018059   | 16.03x  | -93.76%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.289491 | 0.013985   | 20.70x  | -95.17%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.318311 | 0.110943   | 2.87x   | -65.15%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.318311 | 0.014439   | 22.05x  | -95.46%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.318311 | 0.014075   | 22.61x  | -95.58%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.445737 | 0.177887   | 2.51x   | -60.09%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.445737 | 0.016865   | 26.43x  | -96.22%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.445737 | 0.012583   | 35.42x  | -97.18%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.409144 | 0.107989   | 3.79x   | -73.61%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.409144 | 0.014111   | 28.99x  | -96.55%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.409144 | 0.012077   | 33.88x  | -97.05%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.353967 | 0.113717   | 3.11x   | -67.87%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.353967 | 0.017573   | 20.14x  | -95.04%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.353967 | 0.013235   | 26.74x  | -96.26%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.326601 | 0.127764   | 2.56x   | -60.88%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.326601 | 0.015116   | 21.61x  | -95.37%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.326601 | 0.012396   | 26.35x  | -96.20%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.334612 | 0.131597   | 2.54x   | -60.67%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.334612 | 0.014686   | 22.78x  | -95.61%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.334612 | 0.012257   | 27.30x  | -96.34%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.334569 | 0.120429   | 2.78x   | -64.00%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.334569 | 0.016463   | 20.32x  | -95.08%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.334569 | 0.012845   | 26.05x  | -96.16%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.451496 | 0.173479   | 2.60x   | -61.58%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.451496 | 0.016673   | 27.08x  | -96.31%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.451496 | 0.013498   | 33.45x  | -97.01%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.360730 | 0.038919   | 9.27x   | -89.21%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.360730 | 0.017520   | 20.59x  | -95.14%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.360730 | 0.022487   | 16.04x  | -93.77%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.475253 | 0.050359   | 9.44x   | -89.40%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.475253 | 0.021715   | 21.89x  | -95.43%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.475253 | 0.020613   | 23.06x  | -95.66%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.358115 | 0.055527   | 6.45x   | -84.49%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.358115 | 0.022743   | 15.75x  | -93.65%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.358115 | 0.019199   | 18.65x  | -94.64%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.381367 | 0.045693   | 8.35x   | -88.02%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.381367 | 0.021864   | 17.44x  | -94.27%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.381367 | 0.020033   | 19.04x  | -94.75%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.382259 | 0.052773   | 7.24x   | -86.19%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.382259 | 0.023429   | 16.32x  | -93.87%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.382259 | 0.021020   | 18.19x  | -94.50%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.410307 | 0.113506   | 3.61x   | -72.34%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.410307 | 0.023382   | 17.55x  | -94.30%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.410307 | 0.020776   | 19.75x  | -94.94%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.345213 | 0.051640   | 6.68x   | -85.04%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.345213 | 0.020889   | 16.53x  | -93.95%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.345213 | 0.018088   | 19.08x  | -94.76%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.356956 | 0.042418   | 8.42x   | -88.12%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.356956 | 0.020388   | 17.51x  | -94.29%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.356956 | 0.018448   | 19.35x  | -94.83%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.513513 | 0.127795   | 4.02x   | -75.11%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.513513 | 0.022804   | 22.52x  | -95.56%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.513513 | 0.019827   | 25.90x  | -96.14%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.459295 | 0.045731   | 10.04x  | -90.04%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.459295 | 0.020415   | 22.50x  | -95.56%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.459295 | 0.018753   | 24.49x  | -95.92%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.346862 | 0.055039   | 6.30x   | -84.13%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.346862 | 0.022762   | 15.24x  | -93.44%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.346862 | 0.020517   | 16.91x  | -94.09%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.379891 | 0.074465   | 5.10x   | -80.40%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.379891 | 0.021488   | 17.68x  | -94.34%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.379891 | 0.018715   | 20.30x  | -95.07%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.358351 | 0.073870   | 4.85x   | -79.39%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.358351 | 0.021306   | 16.82x  | -94.05%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.358351 | 0.018812   | 19.05x  | -94.75%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.365784 | 0.048853   | 7.49x   | -86.64%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.365784 | 0.021758   | 16.81x  | -94.05%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.365784 | 0.019308   | 18.94x  | -94.72%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.522489 | 0.121912   | 4.29x   | -76.67%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.522489 | 0.022933   | 22.78x  | -95.61%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.522489 | 0.019494   | 26.80x  | -96.27%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.276090 | 0.047717   | 5.79x   | -82.72%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.276090 | 0.028850   | 9.57x   | -89.55%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.276090 | 0.023152   | 11.92x  | -91.61%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.395403 | 0.062284   | 6.35x   | -84.25%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.395403 | 0.029046   | 13.61x  | -92.65%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.395403 | 0.024943   | 15.85x  | -93.69%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.292445 | 0.073545   | 3.98x   | -74.85%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.292445 | 0.031599   | 9.25x   | -89.19%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.292445 | 0.026708   | 10.95x  | -90.87%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.304269 | 0.060654   | 5.02x   | -80.07%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.304269 | 0.030173   | 10.08x  | -90.08%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.304269 | 0.024082   | 12.63x  | -92.09%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.297940 | 0.064855   | 4.59x   | -78.23%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.297940 | 0.032234   | 9.24x   | -89.18%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.297940 | 0.024908   | 11.96x  | -91.64%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.287864 | 0.101204   | 2.84x   | -64.84%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.287864 | 0.031384   | 9.17x   | -89.10%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.287864 | 0.025506   | 11.29x  | -91.14%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.286950 | 0.070381   | 4.08x   | -75.47%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.286950 | 0.030967   | 9.27x   | -89.21%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.286950 | 0.025482   | 11.26x  | -91.12%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.282372 | 0.058429   | 4.83x   | -79.31%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.282372 | 0.031955   | 8.84x   | -88.68%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.282372 | 0.024316   | 11.61x  | -91.39%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.456215 | 0.136923   | 3.33x   | -69.99%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.456215 | 0.031041   | 14.70x  | -93.20%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.456215 | 0.026593   | 17.16x  | -94.17%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.398403 | 0.059040   | 6.75x   | -85.18%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.398403 | 0.030875   | 12.90x  | -92.25%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.398403 | 0.025025   | 15.92x  | -93.72%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.281782 | 0.070760   | 3.98x   | -74.89%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.281782 | 0.031330   | 8.99x   | -88.88%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.281782 | 0.026512   | 10.63x  | -90.59%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.298947 | 0.081887   | 3.65x   | -72.61%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.298947 | 0.030216   | 9.89x   | -89.89%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.298947 | 0.025329   | 11.80x  | -91.53%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.293657 | 0.084393   | 3.48x   | -71.26%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.293657 | 0.029786   | 9.86x   | -89.86%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.293657 | 0.024863   | 11.81x  | -91.53%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.304267 | 0.065925   | 4.62x   | -78.33%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.304267 | 0.029934   | 10.16x  | -90.16%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.304267 | 0.024794   | 12.27x  | -91.85%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.411254 | 0.130617   | 3.15x   | -68.24%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.411254 | 0.031711   | 12.97x  | -92.29%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.411254 | 0.025233   | 16.30x  | -93.86%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.080693 | 0.024028   | 3.36x   | -70.22%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.080693 | 0.019848   | 4.07x   | -75.40%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.080693 | 0.020362   | 3.96x   | -74.77%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.112877 | 0.027675   | 4.08x   | -75.48%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.112877 | 0.020064   | 5.63x   | -82.23%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.112877 | 0.021291   | 5.30x   | -81.14%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.090648 | 0.027559   | 3.29x   | -69.60%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.090648 | 0.019428   | 4.67x   | -78.57%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.090648 | 0.020857   | 4.35x   | -76.99%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.092080 | 0.025890   | 3.56x   | -71.88%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.092080 | 0.019924   | 4.62x   | -78.36%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.092080 | 0.020624   | 4.46x   | -77.60%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.092036 | 0.025983   | 3.54x   | -71.77%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.092036 | 0.020068   | 4.59x   | -78.20%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.092036 | 0.020911   | 4.40x   | -77.28%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.089068 | 0.037010   | 2.41x   | -58.45%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.089068 | 0.019572   | 4.55x   | -78.03%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.089068 | 0.020821   | 4.28x   | -76.62%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.088479 | 0.027390   | 3.23x   | -69.04%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.088479 | 0.019543   | 4.53x   | -77.91%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.088479 | 0.021317   | 4.15x   | -75.91%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.092759 | 0.026934   | 3.44x   | -70.96%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.092759 | 0.020586   | 4.51x   | -77.81%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.092759 | 0.021784   | 4.26x   | -76.52%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.132783 | 0.044449   | 2.99x   | -66.53%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.132783 | 0.020048   | 6.62x   | -84.90%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.132783 | 0.021023   | 6.32x   | -84.17%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.116825 | 0.024840   | 4.70x   | -78.74%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.116825 | 0.020129   | 5.80x   | -82.77%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.116825 | 0.021392   | 5.46x   | -81.69%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.087093 | 0.025461   | 3.42x   | -70.77%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.087093 | 0.020774   | 4.19x   | -76.15%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.087093 | 0.020649   | 4.22x   | -76.29%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.091756 | 0.030996   | 2.96x   | -66.22%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.091756 | 0.020047   | 4.58x   | -78.15%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.091756 | 0.021306   | 4.31x   | -76.78%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.089795 | 0.032028   | 2.80x   | -64.33%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.089795 | 0.019573   | 4.59x   | -78.20%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.089795 | 0.021079   | 4.26x   | -76.53%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.090912 | 0.026065   | 3.49x   | -71.33%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.090912 | 0.019987   | 4.55x   | -78.01%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.090912 | 0.021612   | 4.21x   | -76.23%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.114587 | 0.041503   | 2.76x   | -63.78%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.114587 | 0.019931   | 5.75x   | -82.61%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.114587 | 0.021110   | 5.43x   | -81.58%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.064809 | 0.019636   | 3.30x   | -69.70%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.064809 | 0.002695   | 24.05x  | -95.84%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.064809 | 0.002174   | 29.80x  | -96.64%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.098995 | 0.024200   | 4.09x   | -75.55%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.098995 | 0.003660   | 27.05x  | -96.30%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.098995 | 0.002699   | 36.68x  | -97.27%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.075945 | 0.024493   | 3.10x   | -67.75%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.075945 | 0.003158   | 24.05x  | -95.84%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.075945 | 0.002678   | 28.36x  | -96.47%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.078307 | 0.024219   | 3.23x   | -69.07%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.078307 | 0.003198   | 24.49x  | -95.92%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.078307 | 0.002688   | 29.13x  | -96.57%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.078298 | 0.025171   | 3.11x   | -67.85%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.078298 | 0.003515   | 22.27x  | -95.51%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.078298 | 0.002785   | 28.11x  | -96.44%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.073416 | 0.030215   | 2.43x   | -58.84%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.073416 | 0.004229   | 17.36x  | -94.24%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.073416 | 0.002782   | 26.39x  | -96.21%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.073640 | 0.024922   | 2.95x   | -66.16%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.073640 | 0.003098   | 23.77x  | -95.79%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.073640 | 0.002663   | 27.65x  | -96.38%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.083894 | 0.024944   | 3.36x   | -70.27%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.083894 | 0.003076   | 27.27x  | -96.33%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.083894 | 0.002650   | 31.65x  | -96.84%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.108221 | 0.039258   | 2.76x   | -63.72%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.108221 | 0.003663   | 29.54x  | -96.61%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.108221 | 0.002750   | 39.35x  | -97.46%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.104771 | 0.025160   | 4.16x   | -75.99%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.104771 | 0.003116   | 33.63x  | -97.03%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.104771 | 0.002786   | 37.61x  | -97.34%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.073425 | 0.024533   | 2.99x   | -66.59%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.073425 | 0.003826   | 19.19x  | -94.79%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.073425 | 0.002846   | 25.80x  | -96.12%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.076789 | 0.028153   | 2.73x   | -63.34%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.076789 | 0.003244   | 23.67x  | -95.78%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.076789 | 0.002683   | 28.62x  | -96.51%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.076035 | 0.027911   | 2.72x   | -63.29%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.076035 | 0.003165   | 24.03x  | -95.84%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.076035 | 0.002640   | 28.80x  | -96.53%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.078232 | 0.025002   | 3.13x   | -68.04%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.078232 | 0.003331   | 23.49x  | -95.74%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.078232 | 0.002742   | 28.53x  | -96.50%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.103287 | 0.037936   | 2.72x   | -63.27%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.103287 | 0.003545   | 29.13x  | -96.57%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.103287 | 0.002834   | 36.44x  | -97.26%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.078339 | 0.007519   | 10.42x  | -90.40%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.078339 | 0.002570   | 30.48x  | -96.72%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.078339 | 0.002947   | 26.58x  | -96.24%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.112017 | 0.009394   | 11.92x  | -91.61%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.112017 | 0.003848   | 29.11x  | -96.56%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.112017 | 0.003168   | 35.36x  | -97.17%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.085129 | 0.010211   | 8.34x   | -88.00%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.085129 | 0.003203   | 26.58x  | -96.24%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.085129 | 0.002904   | 29.32x  | -96.59%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.087308 | 0.009476   | 9.21x   | -89.15%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.087308 | 0.003342   | 26.12x  | -96.17%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.087308 | 0.003358   | 26.00x  | -96.15%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.088964 | 0.010766   | 8.26x   | -87.90%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.088964 | 0.003791   | 23.47x  | -95.74%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.088964 | 0.003309   | 26.89x  | -96.28%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.085261 | 0.023050   | 3.70x   | -72.97%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.085261 | 0.003778   | 22.56x  | -95.57%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.085261 | 0.003065   | 27.82x  | -96.41%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.085513 | 0.010258   | 8.34x   | -88.00%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.085513 | 0.003197   | 26.75x  | -96.26%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.085513 | 0.002874   | 29.75x  | -96.64%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.086158 | 0.008211   | 10.49x  | -90.47%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.086158 | 0.003320   | 25.95x  | -96.15%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.086158 | 0.002973   | 28.98x  | -96.55%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.119567 | 0.027568   | 4.34x   | -76.94%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.119567 | 0.003748   | 31.90x  | -96.87%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.119567 | 0.003529   | 33.89x  | -97.05%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.115098 | 0.008896   | 12.94x  | -92.27%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.115098 | 0.003380   | 34.05x  | -97.06%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.115098 | 0.003425   | 33.60x  | -97.02%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.083793 | 0.011232   | 7.46x   | -86.60%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.083793 | 0.003934   | 21.30x  | -95.30%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.083793 | 0.003176   | 26.39x  | -96.21%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.087789 | 0.015265   | 5.75x   | -82.61%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.087789 | 0.003565   | 24.62x  | -95.94%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.087789 | 0.003086   | 28.45x  | -96.48%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.088491 | 0.015590   | 5.68x   | -82.38%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.088491 | 0.003807   | 23.24x  | -95.70%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.088491 | 0.003020   | 29.31x  | -96.59%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.090865 | 0.009521   | 9.54x   | -89.52%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.090865 | 0.003607   | 25.19x  | -96.03%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.090865 | 0.003134   | 29.00x  | -96.55%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.114340 | 0.025501   | 4.48x   | -77.70%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.114340 | 0.003707   | 30.84x  | -96.76%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.114340 | 0.003341   | 34.22x  | -97.08%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.060047 | 0.008908   | 6.74x   | -85.16%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.060047 | 0.004385   | 13.69x  | -92.70%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.060047 | 0.003790   | 15.84x  | -93.69%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.099895 | 0.011726   | 8.52x   | -88.26%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.099895 | 0.005159   | 19.36x  | -94.84%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.099895 | 0.004342   | 23.01x  | -95.65%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.071887 | 0.013161   | 5.46x   | -81.69%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.071887 | 0.005409   | 13.29x  | -92.48%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.071887 | 0.003932   | 18.28x  | -94.53%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.077677 | 0.011817   | 6.57x   | -84.79%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.077677 | 0.005296   | 14.67x  | -93.18%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.077677 | 0.004313   | 18.01x  | -94.45%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.077267 | 0.012684   | 6.09x   | -83.58%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.077267 | 0.005186   | 14.90x  | -93.29%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.077267 | 0.003720   | 20.77x  | -95.19%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.072542 | 0.020472   | 3.54x   | -71.78%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.072542 | 0.005861   | 12.38x  | -91.92%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.072542 | 0.004117   | 17.62x  | -94.32%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.075428 | 0.013452   | 5.61x   | -82.17%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.075428 | 0.005216   | 14.46x  | -93.08%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.075428 | 0.003934   | 19.17x  | -94.78%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.073800 | 0.011367   | 6.49x   | -84.60%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.073800 | 0.005519   | 13.37x  | -92.52%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.073800 | 0.003924   | 18.81x  | -94.68%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.110565 | 0.028208   | 3.92x   | -74.49%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.110565 | 0.005199   | 21.27x  | -95.30%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.110565 | 0.004200   | 26.33x  | -96.20%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.116615 | 0.011469   | 10.17x  | -90.17%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.116615 | 0.005310   | 21.96x  | -95.45%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.116615 | 0.004331   | 26.92x  | -96.29%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.071483 | 0.014449   | 4.95x   | -79.79%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.071483 | 0.005676   | 12.59x  | -92.06%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.071483 | 0.004143   | 17.25x  | -94.20%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.076691 | 0.016328   | 4.70x   | -78.71%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.076691 | 0.005173   | 14.83x  | -93.26%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.076691 | 0.003901   | 19.66x  | -94.91%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.076828 | 0.017419   | 4.41x   | -77.33%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.076828 | 0.005454   | 14.09x  | -92.90%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.076828 | 0.004354   | 17.65x  | -94.33%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.079484 | 0.012082   | 6.58x   | -84.80%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.079484 | 0.005484   | 14.49x  | -93.10%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.079484 | 0.005390   | 14.75x  | -93.22%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.115724 | 0.026542   | 4.36x   | -77.06%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.115724 | 0.005097   | 22.70x  | -95.60%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.115724 | 0.003798   | 30.47x  | -96.72%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.196374 | 0.058853   | 3.34x   | -70.03%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.196374 | 0.043625   | 4.50x   | -77.78%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.196374 | 0.048074   | 4.08x   | -75.52%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.259831 | 0.066618   | 3.90x   | -74.36%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.259831 | 0.046171   | 5.63x   | -82.23%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.259831 | 0.048564   | 5.35x   | -81.31%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.200758 | 0.069175   | 2.90x   | -65.54%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.200758 | 0.045372   | 4.42x   | -77.40%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.200758 | 0.046974   | 4.27x   | -76.60%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.206701 | 0.062789   | 3.29x   | -69.62%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.206701 | 0.045317   | 4.56x   | -78.08%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.206701 | 0.048959   | 4.22x   | -76.31%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.209470 | 0.064376   | 3.25x   | -69.27%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.209470 | 0.048326   | 4.33x   | -76.93%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.209470 | 0.056224   | 3.73x   | -73.16%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.229496 | 0.091685   | 2.50x   | -60.05%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.229496 | 0.046459   | 4.94x   | -79.76%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.229496 | 0.048529   | 4.73x   | -78.85%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.214139 | 0.071250   | 3.01x   | -66.73%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.214139 | 0.047160   | 4.54x   | -77.98%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.214139 | 0.049808   | 4.30x   | -76.74%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.225096 | 0.066572   | 3.38x   | -70.43%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.225096 | 0.048209   | 4.67x   | -78.58%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.225096 | 0.049966   | 4.50x   | -77.80%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.276894 | 0.102263   | 2.71x   | -63.07%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.276894 | 0.047276   | 5.86x   | -82.93%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.276894 | 0.047355   | 5.85x   | -82.90%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.267704 | 0.060130   | 4.45x   | -77.54%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.267704 | 0.046567   | 5.75x   | -82.61%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.267704 | 0.047912   | 5.59x   | -82.10%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.220384 | 0.060765   | 3.63x   | -72.43%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.220384 | 0.045497   | 4.84x   | -79.36%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.220384 | 0.046364   | 4.75x   | -78.96%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.206660 | 0.075766   | 2.73x   | -63.34%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.206660 | 0.046396   | 4.45x   | -77.55%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.206660 | 0.049210   | 4.20x   | -76.19%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.211001 | 0.073044   | 2.89x   | -65.38%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.211001 | 0.046550   | 4.53x   | -77.94%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.211001 | 0.047040   | 4.49x   | -77.71%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.203342 | 0.067081   | 3.03x   | -67.01%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.203342 | 0.046504   | 4.37x   | -77.13%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.203342 | 0.047618   | 4.27x   | -76.58%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.264236 | 0.097028   | 2.72x   | -63.28%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.264236 | 0.046287   | 5.71x   | -82.48%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.264236 | 0.051547   | 5.13x   | -80.49%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.143665 | 0.047962   | 3.00x   | -66.62%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.143665 | 0.006080   | 23.63x  | -95.77%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.143665 | 0.004870   | 29.50x  | -96.61%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.203321 | 0.059378   | 3.42x   | -70.80%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.203321 | 0.008765   | 23.20x  | -95.69%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.203321 | 0.006136   | 33.14x  | -96.98%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.155534 | 0.078863   | 1.97x   | -49.30%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.155534 | 0.008366   | 18.59x  | -94.62%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.155534 | 0.006316   | 24.62x  | -95.94%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.160381 | 0.056642   | 2.83x   | -64.68%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.160381 | 0.007238   | 22.16x  | -95.49%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.160381 | 0.006068   | 26.43x  | -96.22%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.160959 | 0.060059   | 2.68x   | -62.69%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.160959 | 0.007833   | 20.55x  | -95.13%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.160959 | 0.006175   | 26.07x  | -96.16%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.154421 | 0.069659   | 2.22x   | -54.89%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.154421 | 0.008591   | 17.97x  | -94.44%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.154421 | 0.006410   | 24.09x  | -95.85%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.148074 | 0.058496   | 2.53x   | -60.50%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.148074 | 0.006885   | 21.51x  | -95.35%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.148074 | 0.005971   | 24.80x  | -95.97%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.153216 | 0.053966   | 2.84x   | -64.78%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.153216 | 0.007916   | 19.35x  | -94.83%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.153216 | 0.006002   | 25.53x  | -96.08%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.221096 | 0.088259   | 2.51x   | -60.08%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.221096 | 0.008085   | 27.35x  | -96.34%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.221096 | 0.006112   | 36.17x  | -97.24%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.209187 | 0.057203   | 3.66x   | -72.65%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.209187 | 0.006988   | 29.93x  | -96.66%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.209187 | 0.005929   | 35.28x  | -97.17%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.150110 | 0.061724   | 2.43x   | -58.88%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.150110 | 0.008658   | 17.34x  | -94.23%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.150110 | 0.006223   | 24.12x  | -95.85%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.152691 | 0.064958   | 2.35x   | -57.46%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.152691 | 0.007218   | 21.15x  | -95.27%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.152691 | 0.005952   | 25.65x  | -96.10%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.150258 | 0.064796   | 2.32x   | -56.88%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.150258 | 0.007112   | 21.13x  | -95.27%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.150258 | 0.005948   | 25.26x  | -96.04%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.158777 | 0.058399   | 2.72x   | -63.22%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.158777 | 0.007613   | 20.86x  | -95.21%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.158777 | 0.006000   | 26.46x  | -96.22%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.207337 | 0.085220   | 2.43x   | -58.90%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.207337 | 0.008637   | 24.00x  | -95.83%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.207337 | 0.006058   | 34.23x  | -97.08%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.164903 | 0.018774   | 8.78x   | -88.61%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.164903 | 0.005900   | 27.95x  | -96.42%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.164903 | 0.006123   | 26.93x  | -96.29%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.224110 | 0.023933   | 9.36x   | -89.32%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.224110 | 0.008006   | 27.99x  | -96.43%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.224110 | 0.007772   | 28.83x  | -96.53%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.168735 | 0.028008   | 6.02x   | -83.40%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.168735 | 0.007053   | 23.92x  | -95.82%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.168735 | 0.006487   | 26.01x  | -96.16%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.175278 | 0.022511   | 7.79x   | -87.16%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.175278 | 0.007504   | 23.36x  | -95.72%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.175278 | 0.006515   | 26.90x  | -96.28%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.177036 | 0.024152   | 7.33x   | -86.36%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.177036 | 0.008391   | 21.10x  | -95.26%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.177036 | 0.007710   | 22.96x  | -95.64%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.192535 | 0.057782   | 3.33x   | -69.99%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.192535 | 0.008523   | 22.59x  | -95.57%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.192535 | 0.007405   | 26.00x  | -96.15%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.170754 | 0.025733   | 6.64x   | -84.93%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.170754 | 0.007005   | 24.38x  | -95.90%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.170754 | 0.006197   | 27.55x  | -96.37%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.173318 | 0.022077   | 7.85x   | -87.26%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.173318 | 0.007188   | 24.11x  | -95.85%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.173318 | 0.006231   | 27.81x  | -96.40%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.239560 | 0.063894   | 3.75x   | -73.33%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.239560 | 0.008281   | 28.93x  | -96.54%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.239560 | 0.007301   | 32.81x  | -96.95%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.228112 | 0.021735   | 10.49x  | -90.47%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.228112 | 0.007354   | 31.02x  | -96.78%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.228112 | 0.006536   | 34.90x  | -97.13%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.173831 | 0.026662   | 6.52x   | -84.66%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.173831 | 0.008719   | 19.94x  | -94.98%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.173831 | 0.006929   | 25.09x  | -96.01%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.193256 | 0.036260   | 5.33x   | -81.24%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.193256 | 0.007769   | 24.87x  | -95.98%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.193256 | 0.006429   | 30.06x  | -96.67%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.175287 | 0.035600   | 4.92x   | -79.69%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.175287 | 0.007881   | 22.24x  | -95.50%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.175287 | 0.006697   | 26.17x  | -96.18%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.181719 | 0.031192   | 5.83x   | -82.84%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.181719 | 0.008093   | 22.45x  | -95.55%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.181719 | 0.006979   | 26.04x  | -96.16%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.230482 | 0.059517   | 3.87x   | -74.18%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.230482 | 0.009953   | 23.16x  | -95.68%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.230482 | 0.006980   | 33.02x  | -96.97%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.122258 | 0.020026   | 6.11x   | -83.62%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.122258 | 0.009672   | 12.64x  | -92.09%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.122258 | 0.008212   | 14.89x  | -93.28%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.180524 | 0.025649   | 7.04x   | -85.79%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.180524 | 0.011162   | 16.17x  | -93.82%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.180524 | 0.008238   | 21.91x  | -95.44%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.129798 | 0.028166   | 4.61x   | -78.30%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.129798 | 0.011578   | 11.21x  | -91.08%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.129798 | 0.008829   | 14.70x  | -93.20%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.138190 | 0.024594   | 5.62x   | -82.20%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.138190 | 0.011902   | 11.61x  | -91.39%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.138190 | 0.008801   | 15.70x  | -93.63%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.138120 | 0.029740   | 4.64x   | -78.47%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.138120 | 0.011424   | 12.09x  | -91.73%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.138120 | 0.008573   | 16.11x  | -93.79%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.136594 | 0.044739   | 3.05x   | -67.25%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.136594 | 0.012185   | 11.21x  | -91.08%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.136594 | 0.008796   | 15.53x  | -93.56%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.130127 | 0.030440   | 4.27x   | -76.61%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.130127 | 0.011589   | 11.23x  | -91.09%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.130127 | 0.008608   | 15.12x  | -93.39%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.134764 | 0.022642   | 5.95x   | -83.20%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.134764 | 0.012172   | 11.07x  | -90.97%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.134764 | 0.008479   | 15.89x  | -93.71%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.207565 | 0.061824   | 3.36x   | -70.21%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.207565 | 0.011083   | 18.73x  | -94.66%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.207565 | 0.008451   | 24.56x  | -95.93%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.192813 | 0.023693   | 8.14x   | -87.71%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.192813 | 0.011342   | 17.00x  | -94.12%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.192813 | 0.009628   | 20.03x  | -95.01%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.133404 | 0.031691   | 4.21x   | -76.24%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.133404 | 0.012087   | 11.04x  | -90.94%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.133404 | 0.008914   | 14.97x  | -93.32%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.140941 | 0.035857   | 3.93x   | -74.56%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.140941 | 0.011559   | 12.19x  | -91.80%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.140941 | 0.008531   | 16.52x  | -93.95%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.146147 | 0.035679   | 4.10x   | -75.59%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.146147 | 0.011567   | 12.63x  | -92.09%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.146147 | 0.008915   | 16.39x  | -93.90%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.139452 | 0.024651   | 5.66x   | -82.32%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.139452 | 0.012549   | 11.11x  | -91.00%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.139452 | 0.008455   | 16.49x  | -93.94%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.188737 | 0.058453   | 3.23x   | -69.03%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.188737 | 0.012816   | 14.73x  | -93.21%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.188737 | 0.008271   | 22.82x  | -95.62%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.354070 | 0.094495   | 3.75x   | -73.31%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.354070 | 0.077584   | 4.56x   | -78.09%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.354070 | 0.083558   | 4.24x   | -76.40%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.440345 | 0.107391   | 4.10x   | -75.61%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.440345 | 0.079638   | 5.53x   | -81.91%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.440345 | 0.083679   | 5.26x   | -81.00%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.348467 | 0.110419   | 3.16x   | -68.31%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.348467 | 0.082633   | 4.22x   | -76.29%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.348467 | 0.085285   | 4.09x   | -75.53%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.364993 | 0.105131   | 3.47x   | -71.20%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.364993 | 0.082099   | 4.45x   | -77.51%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.364993 | 0.084159   | 4.34x   | -76.94%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.391889 | 0.107133   | 3.66x   | -72.66%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.391889 | 0.082730   | 4.74x   | -78.89%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.391889 | 0.085453   | 4.59x   | -78.19%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.379568 | 0.155747   | 2.44x   | -58.97%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.379568 | 0.083155   | 4.56x   | -78.09%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.379568 | 0.087315   | 4.35x   | -77.00%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.356189 | 0.112102   | 3.18x   | -68.53%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.356189 | 0.082032   | 4.34x   | -76.97%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.356189 | 0.084771   | 4.20x   | -76.20%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.356963 | 0.101303   | 3.52x   | -71.62%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.356963 | 0.084971   | 4.20x   | -76.20%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.356963 | 0.083481   | 4.28x   | -76.61%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.505525 | 0.182390   | 2.77x   | -63.92%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.505525 | 0.084965   | 5.95x   | -83.19%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.505525 | 0.088353   | 5.72x   | -82.52%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.455729 | 0.110650   | 4.12x   | -75.72%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.455729 | 0.081102   | 5.62x   | -82.20%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.455729 | 0.084569   | 5.39x   | -81.44%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.352198 | 0.101945   | 3.45x   | -71.05%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.352198 | 0.082965   | 4.25x   | -76.44%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.352198 | 0.084271   | 4.18x   | -76.07%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.361226 | 0.123772   | 2.92x   | -65.74%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.361226 | 0.081492   | 4.43x   | -77.44%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.361226 | 0.086979   | 4.15x   | -75.92%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.388231 | 0.133504   | 2.91x   | -65.61%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.388231 | 0.087121   | 4.46x   | -77.56%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.388231 | 0.089861   | 4.32x   | -76.85%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.370897 | 0.107754   | 3.44x   | -70.95%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.370897 | 0.082682   | 4.49x   | -77.71%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.370897 | 0.086202   | 4.30x   | -76.76%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.467097 | 0.168070   | 2.78x   | -64.02%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.467097 | 0.081404   | 5.74x   | -82.57%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.467097 | 0.084906   | 5.50x   | -81.82%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.261075 | 0.076821   | 3.40x   | -70.58%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.261075 | 0.010949   | 23.84x  | -95.81%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.261075 | 0.008939   | 29.20x  | -96.58%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.367234 | 0.101390   | 3.62x   | -72.39%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.367234 | 0.013532   | 27.14x  | -96.32%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.367234 | 0.011953   | 30.72x  | -96.75%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.264836 | 0.101908   | 2.60x   | -61.52%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.264836 | 0.013199   | 20.06x  | -95.02%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.264836 | 0.011022   | 24.03x  | -95.84%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.298596 | 0.101419   | 2.94x   | -66.03%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.298596 | 0.013297   | 22.46x  | -95.55%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.298596 | 0.010947   | 27.28x  | -96.33%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.294860 | 0.101400   | 2.91x   | -65.61%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.294860 | 0.014207   | 20.75x  | -95.18%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.294860 | 0.011354   | 25.97x  | -96.15%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.293035 | 0.131017   | 2.24x   | -55.29%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.293035 | 0.018131   | 16.16x  | -93.81%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.293035 | 0.011407   | 25.69x  | -96.11%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.279567 | 0.105780   | 2.64x   | -62.16%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.279567 | 0.013200   | 21.18x  | -95.28%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.279567 | 0.011660   | 23.98x  | -95.83%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.279328 | 0.094816   | 2.95x   | -66.06%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.279328 | 0.012766   | 21.88x  | -95.43%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.279328 | 0.011554   | 24.18x  | -95.86%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.455646 | 0.164669   | 2.77x   | -63.86%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.455646 | 0.015687   | 29.05x  | -96.56%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.455646 | 0.011762   | 38.74x  | -97.42%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.382561 | 0.099620   | 3.84x   | -73.96%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.382561 | 0.012854   | 29.76x  | -96.64%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.382561 | 0.010639   | 35.96x  | -97.22%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.299917 | 0.109940   | 2.73x   | -63.34%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.299917 | 0.016387   | 18.30x  | -94.54%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.299917 | 0.011588   | 25.88x  | -96.14%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.292394 | 0.113388   | 2.58x   | -61.22%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.292394 | 0.014226   | 20.55x  | -95.13%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.292394 | 0.010967   | 26.66x  | -96.25%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.296621 | 0.118007   | 2.51x   | -60.22%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.296621 | 0.016256   | 18.25x  | -94.52%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.296621 | 0.010979   | 27.02x  | -96.30%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.284575 | 0.099843   | 2.85x   | -64.92%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.284575 | 0.013698   | 20.77x  | -95.19%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.284575 | 0.011669   | 24.39x  | -95.90%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.418926 | 0.149006   | 2.81x   | -64.43%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.418926 | 0.014186   | 29.53x  | -96.61%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.418926 | 0.010833   | 38.67x  | -97.41%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.329398 | 0.033616   | 9.80x   | -89.79%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.329398 | 0.015738   | 20.93x  | -95.22%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.329398 | 0.015176   | 21.70x  | -95.39%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.421094 | 0.045076   | 9.34x   | -89.30%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.421094 | 0.018969   | 22.20x  | -95.50%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.421094 | 0.017160   | 24.54x  | -95.92%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.300377 | 0.046276   | 6.49x   | -84.59%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.300377 | 0.018537   | 16.20x  | -93.83%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.300377 | 0.016601   | 18.09x  | -94.47%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.318975 | 0.041638   | 7.66x   | -86.95%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.318975 | 0.019110   | 16.69x  | -94.01%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.318975 | 0.017016   | 18.75x  | -94.67%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.318101 | 0.044355   | 7.17x   | -86.06%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.318101 | 0.021118   | 15.06x  | -93.36%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.318101 | 0.017432   | 18.25x  | -94.52%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.304676 | 0.097113   | 3.14x   | -68.13%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.304676 | 0.020682   | 14.73x  | -93.21%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.304676 | 0.017051   | 17.87x  | -94.40%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.295145 | 0.045965   | 6.42x   | -84.43%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.295145 | 0.017453   | 16.91x  | -94.09%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.295145 | 0.016664   | 17.71x  | -94.35%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.306521 | 0.039383   | 7.78x   | -87.15%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.306521 | 0.018709   | 16.38x  | -93.90%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.306521 | 0.016436   | 18.65x  | -94.64%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.448208 | 0.114066   | 3.93x   | -74.55%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.448208 | 0.021408   | 20.94x  | -95.22%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.448208 | 0.017662   | 25.38x  | -96.06%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.413842 | 0.042721   | 9.69x   | -89.68%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.413842 | 0.019334   | 21.41x  | -95.33%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.413842 | 0.019463   | 21.26x  | -95.30%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.317271 | 0.047500   | 6.68x   | -85.03%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.317271 | 0.020258   | 15.66x  | -93.61%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.317271 | 0.017597   | 18.03x  | -94.45%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.319066 | 0.066299   | 4.81x   | -79.22%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.319066 | 0.020612   | 15.48x  | -93.54%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.319066 | 0.017317   | 18.42x  | -94.57%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.319064 | 0.065064   | 4.90x   | -79.61%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.319064 | 0.019453   | 16.40x  | -93.90%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.319064 | 0.016942   | 18.83x  | -94.69%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.322630 | 0.042406   | 7.61x   | -86.86%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.322630 | 0.021122   | 15.27x  | -93.45%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.322630 | 0.018853   | 17.11x  | -94.16%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.429716 | 0.108210   | 3.97x   | -74.82%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.429716 | 0.019595   | 21.93x  | -95.44%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.429716 | 0.017993   | 23.88x  | -95.81%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.245045 | 0.042286   | 5.79x   | -82.74%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.245045 | 0.023671   | 10.35x  | -90.34%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.245045 | 0.019389   | 12.64x  | -92.09%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.347279 | 0.047128   | 7.37x   | -86.43%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.347279 | 0.020681   | 16.79x  | -94.04%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.347279 | 0.018484   | 18.79x  | -94.68%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.238223 | 0.058212   | 4.09x   | -75.56%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.238223 | 0.025781   | 9.24x   | -89.18%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.238223 | 0.021515   | 11.07x  | -90.97%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.258864 | 0.045900   | 5.64x   | -82.27%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.258864 | 0.021019   | 12.32x  | -91.88%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.258864 | 0.018382   | 14.08x  | -92.90%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.265312 | 0.053843   | 4.93x   | -79.71%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.265312 | 0.027151   | 9.77x   | -89.77%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.265312 | 0.025277   | 10.50x  | -90.47%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.262685 | 0.081371   | 3.23x   | -69.02%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.262685 | 0.022270   | 11.80x  | -91.52%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.262685 | 0.018886   | 13.91x  | -92.81%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.237753 | 0.056958   | 4.17x   | -76.04%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.237753 | 0.026189   | 9.08x   | -88.98%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.237753 | 0.020543   | 11.57x  | -91.36%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.240630 | 0.040988   | 5.87x   | -82.97%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.240630 | 0.021077   | 11.42x  | -91.24%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.240630 | 0.021313   | 11.29x  | -91.14%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.398577 | 0.117834   | 3.38x   | -70.44%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.398577 | 0.025388   | 15.70x  | -93.63%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.398577 | 0.021417   | 18.61x  | -94.63%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.343682 | 0.042918   | 8.01x   | -87.51%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.343682 | 0.021333   | 16.11x  | -93.79%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.343682 | 0.019607   | 17.53x  | -94.30%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.246783 | 0.061434   | 4.02x   | -75.11%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.246783 | 0.027379   | 9.01x   | -88.91%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.246783 | 0.022198   | 11.12x  | -91.00%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.258877 | 0.066575   | 3.89x   | -74.28%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.258877 | 0.020796   | 12.45x  | -91.97%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.258877 | 0.020095   | 12.88x  | -92.24%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.253456 | 0.070305   | 3.61x   | -72.26%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.253456 | 0.026935   | 9.41x   | -89.37%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.253456 | 0.020006   | 12.67x  | -92.11%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.252142 | 0.046984   | 5.37x   | -81.37%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.252142 | 0.020902   | 12.06x  | -91.71%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.252142 | 0.018090   | 13.94x  | -92.83%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.355364 | 0.110797   | 3.21x   | -68.82%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.355364 | 0.024996   | 14.22x  | -92.97%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.355364 | 0.020647   | 17.21x  | -94.19%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.765982 | 0.215670   | 3.55x   | -71.84%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.765982 | 0.176256   | 4.35x   | -76.99%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.765982 | 0.187832   | 4.08x   | -75.48%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 0.988702 | 0.244889   | 4.04x   | -75.23%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 0.988702 | 0.187244   | 5.28x   | -81.06%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 0.988702 | 0.193405   | 5.11x   | -80.44%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.746954 | 0.250149   | 2.99x   | -66.51%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.746954 | 0.180877   | 4.13x   | -75.78%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.746954 | 0.191358   | 3.90x   | -74.38%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.779058 | 0.232404   | 3.35x   | -70.17%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.779058 | 0.180874   | 4.31x   | -76.78%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.779058 | 0.187844   | 4.15x   | -75.89%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.794044 | 0.244881   | 3.24x   | -69.16%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.794044 | 0.181514   | 4.37x   | -77.14%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.794044 | 0.192107   | 4.13x   | -75.81%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.768608 | 0.357102   | 2.15x   | -53.54%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.768608 | 0.180161   | 4.27x   | -76.56%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.768608 | 0.186938   | 4.11x   | -75.68%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.745141 | 0.257797   | 2.89x   | -65.40%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.745141 | 0.178423   | 4.18x   | -76.06%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.745141 | 0.188098   | 3.96x   | -74.76%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.770154 | 0.234005   | 3.29x   | -69.62%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.770154 | 0.186746   | 4.12x   | -75.75%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.770154 | 0.193410   | 3.98x   | -74.89%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.077088 | 0.398545   | 2.70x   | -63.00%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.077088 | 0.182769   | 5.89x   | -83.03%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.077088 | 0.193179   | 5.58x   | -82.06%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.991143 | 0.231896   | 4.27x   | -76.60%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.991143 | 0.180818   | 5.48x   | -81.76%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.991143 | 0.191662   | 5.17x   | -80.66%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.763133 | 0.234906   | 3.25x   | -69.22%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.763133 | 0.179937   | 4.24x   | -76.42%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.763133 | 0.190339   | 4.01x   | -75.06%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.788573 | 0.285034   | 2.77x   | -63.85%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.788573 | 0.180963   | 4.36x   | -77.05%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.788573 | 0.189628   | 4.16x   | -75.95%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.776919 | 0.282263   | 2.75x   | -63.67%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.776919 | 0.180722   | 4.30x   | -76.74%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.776919 | 0.189493   | 4.10x   | -75.61%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.796175 | 0.237950   | 3.35x   | -70.11%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.796175 | 0.181174   | 4.39x   | -77.24%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.796175 | 0.196329   | 4.06x   | -75.34%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 1.012301 | 0.378189   | 2.68x   | -62.64%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 1.012301 | 0.182739   | 5.54x   | -81.95%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 1.012301 | 0.192764   | 5.25x   | -80.96%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.566880 | 0.176326   | 3.21x   | -68.90%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.566880 | 0.024366   | 23.27x  | -95.70%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.566880 | 0.020508   | 27.64x  | -96.38%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.788741 | 0.219866   | 3.59x   | -72.12%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.788741 | 0.030328   | 26.01x  | -96.15%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.788741 | 0.024809   | 31.79x  | -96.85%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.654654 | 0.247879   | 2.64x   | -62.14%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.654654 | 0.028425   | 23.03x  | -95.66%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.654654 | 0.025324   | 25.85x  | -96.13%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.750567 | 0.229437   | 3.27x   | -69.43%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.750567 | 0.030601   | 24.53x  | -95.92%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.750567 | 0.024394   | 30.77x  | -96.75%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.756306 | 0.282145   | 2.68x   | -62.69%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.756306 | 0.033454   | 22.61x  | -95.58%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.756306 | 0.025184   | 30.03x  | -96.67%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.700781 | 0.281597   | 2.49x   | -59.82%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.700781 | 0.035890   | 19.53x  | -94.88%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.700781 | 0.025722   | 27.24x  | -96.33%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.608579 | 0.231356   | 2.63x   | -61.98%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.608579 | 0.027874   | 21.83x  | -95.42%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.608579 | 0.025250   | 24.10x  | -95.85%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.603959 | 0.218539   | 2.76x   | -63.82%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.603959 | 0.028409   | 21.26x  | -95.30%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.603959 | 0.023687   | 25.50x  | -96.08%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.901701 | 0.361391   | 2.50x   | -59.92%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.901701 | 0.033107   | 27.24x  | -96.33%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.901701 | 0.027005   | 33.39x  | -97.01%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.825560 | 0.217724   | 3.79x   | -73.63%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.825560 | 0.027921   | 29.57x  | -96.62%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.825560 | 0.024344   | 33.91x  | -97.05%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.571259 | 0.229746   | 2.49x   | -59.78%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.571259 | 0.036051   | 15.85x  | -93.69%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.571259 | 0.026242   | 21.77x  | -95.41%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.591419 | 0.257965   | 2.29x   | -56.38%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.591419 | 0.029335   | 20.16x  | -95.04%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.591419 | 0.024861   | 23.79x  | -95.80%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.589555 | 0.262234   | 2.25x   | -55.52%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.589555 | 0.029534   | 19.96x  | -94.99%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.589555 | 0.024529   | 24.04x  | -95.84%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.598445 | 0.230121   | 2.60x   | -61.55%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.598445 | 0.030621   | 19.54x  | -94.88%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.598445 | 0.025867   | 23.14x  | -95.68%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.830174 | 0.339093   | 2.45x   | -59.15%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.830174 | 0.032107   | 25.86x  | -96.13%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.830174 | 0.025650   | 32.37x  | -96.91%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.671411 | 0.073578   | 9.13x   | -89.04%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.671411 | 0.033995   | 19.75x  | -94.94%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.671411 | 0.033602   | 19.98x  | -95.00%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.889616 | 0.097911   | 9.09x   | -88.99%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.889616 | 0.041969   | 21.20x  | -95.28%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.889616 | 0.038260   | 23.25x  | -95.70%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.647270 | 0.105637   | 6.13x   | -83.68%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.647270 | 0.038910   | 16.64x  | -93.99%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.647270 | 0.036228   | 17.87x  | -94.40%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.698571 | 0.088023   | 7.94x   | -87.40%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.698571 | 0.039778   | 17.56x  | -94.31%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.698571 | 0.035070   | 19.92x  | -94.98%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.694253 | 0.098606   | 7.04x   | -85.80%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.694253 | 0.042134   | 16.48x  | -93.93%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.694253 | 0.037882   | 18.33x  | -94.54%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.681984 | 0.217450   | 3.14x   | -68.12%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.681984 | 0.044292   | 15.40x  | -93.51%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.681984 | 0.037484   | 18.19x  | -94.50%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.656687 | 0.105256   | 6.24x   | -83.97%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.656687 | 0.039138   | 16.78x  | -94.04%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.656687 | 0.035596   | 18.45x  | -94.58%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.668302 | 0.084607   | 7.90x   | -87.34%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.668302 | 0.037669   | 17.74x  | -94.36%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.668302 | 0.037903   | 17.63x  | -94.33%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.989322 | 0.251633   | 3.93x   | -74.57%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.989322 | 0.042720   | 23.16x  | -95.68%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.989322 | 0.038688   | 25.57x  | -96.09%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.891419 | 0.085812   | 10.39x  | -90.37%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.891419 | 0.039021   | 22.84x  | -95.62%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.891419 | 0.034523   | 25.82x  | -96.13%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.670556 | 0.104605   | 6.41x   | -84.40%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.670556 | 0.043198   | 15.52x  | -93.56%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.670556 | 0.035734   | 18.77x  | -94.67%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.689521 | 0.144710   | 4.76x   | -79.01%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.689521 | 0.042179   | 16.35x  | -93.88%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.689521 | 0.040253   | 17.13x  | -94.16%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.702469 | 0.144558   | 4.86x   | -79.42%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.702469 | 0.041237   | 17.03x  | -94.13%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.702469 | 0.037323   | 18.82x  | -94.69%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.706920 | 0.092077   | 7.68x   | -86.97%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.706920 | 0.042542   | 16.62x  | -93.98%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.706920 | 0.039269   | 18.00x  | -94.45%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.950350 | 0.237563   | 4.00x   | -75.00%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.950350 | 0.041299   | 23.01x  | -95.65%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.950350 | 0.038188   | 24.89x  | -95.98%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.546495 | 0.092566   | 5.90x   | -83.06%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.546495 | 0.051693   | 10.57x  | -90.54%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.546495 | 0.046308   | 11.80x  | -91.53%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.747160 | 0.116589   | 6.41x   | -84.40%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.747160 | 0.058086   | 12.86x  | -92.23%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.747160 | 0.049032   | 15.24x  | -93.44%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.515164 | 0.130444   | 3.95x   | -74.68%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.515164 | 0.057985   | 8.88x   | -88.74%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.515164 | 0.048589   | 10.60x  | -90.57%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.562602 | 0.113739   | 4.95x   | -79.78%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.562602 | 0.058838   | 9.56x   | -89.54%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.562602 | 0.045969   | 12.24x  | -91.83%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.605923 | 0.120376   | 5.03x   | -80.13%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.605923 | 0.058953   | 10.28x  | -90.27%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.605923 | 0.047207   | 12.84x  | -92.21%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.563262 | 0.192582   | 2.92x   | -65.81%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.563262 | 0.060657   | 9.29x   | -89.23%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.563262 | 0.052400   | 10.75x  | -90.70%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.522498 | 0.128247   | 4.07x   | -75.46%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.522498 | 0.058660   | 8.91x   | -88.77%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.522498 | 0.045282   | 11.54x  | -91.33%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.553044 | 0.100474   | 5.50x   | -81.83%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.553044 | 0.059030   | 9.37x   | -89.33%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.553044 | 0.045545   | 12.14x  | -91.76%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.872020 | 0.258538   | 3.37x   | -70.35%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.872020 | 0.056744   | 15.37x  | -93.49%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.872020 | 0.047472   | 18.37x  | -94.56%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.760584 | 0.106026   | 7.17x   | -86.06%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.760584 | 0.058010   | 13.11x  | -92.37%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.760584 | 0.044510   | 17.09x  | -94.15%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.539937 | 0.133069   | 4.06x   | -75.35%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.539937 | 0.061415   | 8.79x   | -88.63%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.539937 | 0.047428   | 11.38x  | -91.22%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.549590 | 0.158027   | 3.48x   | -71.25%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.549590 | 0.058431   | 9.41x   | -89.37%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.549590 | 0.044454   | 12.36x  | -91.91%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.556946 | 0.164667   | 3.38x   | -70.43%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.556946 | 0.057828   | 9.63x   | -89.62%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.556946 | 0.045159   | 12.33x  | -91.89%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.566512 | 0.111835   | 5.07x   | -80.26%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.566512 | 0.059244   | 9.56x   | -89.54%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.566512 | 0.048336   | 11.72x  | -91.47%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.812914 | 0.247472   | 3.28x   | -69.56%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.812914 | 0.056437   | 14.40x  | -93.06%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.812914 | 0.048193   | 16.87x  | -94.07%        |
</details>
<!-- PERF_RESULTS_END -->
