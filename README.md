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

Correctness and performance:

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
| normal        | scalar | 0.197829 | 0.042381   | 4.67x   | -78.58%        |
| normal        | sse42  | 0.197829 | 0.021604   | 9.16x   | -89.08%        |
| normal        | avx2   | 0.197829 | 0.022000   | 8.99x   | -88.88%        |
| soft_light    | scalar | 0.268651 | 0.052382   | 5.13x   | -80.50%        |
| soft_light    | sse42  | 0.268651 | 0.023966   | 11.21x  | -91.08%        |
| soft_light    | avx2   | 0.268651 | 0.022879   | 11.74x  | -91.48%        |
| lighten_only  | scalar | 0.198012 | 0.054348   | 3.64x   | -72.55%        |
| lighten_only  | sse42  | 0.198012 | 0.023264   | 8.51x   | -88.25%        |
| lighten_only  | avx2   | 0.198012 | 0.022423   | 8.83x   | -88.68%        |
| screen        | scalar | 0.208087 | 0.049475   | 4.21x   | -76.22%        |
| screen        | sse42  | 0.208087 | 0.023801   | 8.74x   | -88.56%        |
| screen        | avx2   | 0.208087 | 0.022455   | 9.27x   | -89.21%        |
| dodge         | scalar | 0.208409 | 0.052182   | 3.99x   | -74.96%        |
| dodge         | sse42  | 0.208409 | 0.023900   | 8.72x   | -88.53%        |
| dodge         | avx2   | 0.208409 | 0.022801   | 9.14x   | -89.06%        |
| addition      | scalar | 0.202234 | 0.078005   | 2.59x   | -61.43%        |
| addition      | sse42  | 0.202234 | 0.024208   | 8.35x   | -88.03%        |
| addition      | avx2   | 0.202234 | 0.022852   | 8.85x   | -88.70%        |
| darken_only   | scalar | 0.197071 | 0.054534   | 3.61x   | -72.33%        |
| darken_only   | sse42  | 0.197071 | 0.023379   | 8.43x   | -88.14%        |
| darken_only   | avx2   | 0.197071 | 0.022659   | 8.70x   | -88.50%        |
| multiply      | scalar | 0.201614 | 0.049442   | 4.08x   | -75.48%        |
| multiply      | sse42  | 0.201614 | 0.023555   | 8.56x   | -88.32%        |
| multiply      | avx2   | 0.201614 | 0.022948   | 8.79x   | -88.62%        |
| hard_light    | scalar | 0.301130 | 0.097317   | 3.09x   | -67.68%        |
| hard_light    | sse42  | 0.301130 | 0.024288   | 12.40x  | -91.93%        |
| hard_light    | avx2   | 0.301130 | 0.022907   | 13.15x  | -92.39%        |
| difference    | scalar | 0.272341 | 0.049422   | 5.51x   | -81.85%        |
| difference    | sse42  | 0.272341 | 0.024177   | 11.26x  | -91.12%        |
| difference    | avx2   | 0.272341 | 0.022592   | 12.05x  | -91.70%        |
| subtract      | scalar | 0.202555 | 0.053035   | 3.82x   | -73.82%        |
| subtract      | sse42  | 0.202555 | 0.024132   | 8.39x   | -88.09%        |
| subtract      | avx2   | 0.202555 | 0.022621   | 8.95x   | -88.83%        |
| grain_extract | scalar | 0.207463 | 0.064410   | 3.22x   | -68.95%        |
| grain_extract | sse42  | 0.207463 | 0.023653   | 8.77x   | -88.60%        |
| grain_extract | avx2   | 0.207463 | 0.022563   | 9.19x   | -89.12%        |
| grain_merge   | scalar | 0.209493 | 0.065076   | 3.22x   | -68.94%        |
| grain_merge   | sse42  | 0.209493 | 0.023546   | 8.90x   | -88.76%        |
| grain_merge   | avx2   | 0.209493 | 0.022588   | 9.27x   | -89.22%        |
| divide        | scalar | 0.211735 | 0.051211   | 4.13x   | -75.81%        |
| divide        | sse42  | 0.211735 | 0.023542   | 8.99x   | -88.88%        |
| divide        | avx2   | 0.211735 | 0.022978   | 9.21x   | -89.15%        |
| overlay       | scalar | 0.278261 | 0.092845   | 3.00x   | -66.63%        |
| overlay       | sse42  | 0.278261 | 0.023753   | 11.71x  | -91.46%        |
| overlay       | avx2   | 0.278261 | 0.023029   | 12.08x  | -91.72%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.006785 | 0.001673   | 4.06x   | -75.34%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.006785 | 0.001385   | 4.90x   | -79.59%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.006785 | 0.001470   | 4.62x   | -78.34%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.008597 | 0.001878   | 4.58x   | -78.15%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.008597 | 0.001442   | 5.96x   | -83.23%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.008597 | 0.001524   | 5.64x   | -82.27%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.007374 | 0.001982   | 3.72x   | -73.12%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.007374 | 0.001467   | 5.03x   | -80.11%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.007374 | 0.001523   | 4.84x   | -79.34%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.007156 | 0.001833   | 3.90x   | -74.38%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.007156 | 0.001438   | 4.98x   | -79.91%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.007156 | 0.001481   | 4.83x   | -79.30%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.007552 | 0.001917   | 3.94x   | -74.62%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.007552 | 0.001446   | 5.22x   | -80.85%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.007552 | 0.001506   | 5.02x   | -80.06%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.007271 | 0.002653   | 2.74x   | -63.52%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.007271 | 0.001408   | 5.16x   | -80.64%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.007271 | 0.001503   | 4.84x   | -79.33%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.007308 | 0.001926   | 3.79x   | -73.64%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.007308 | 0.001414   | 5.17x   | -80.65%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.007308 | 0.001481   | 4.93x   | -79.73%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.007464 | 0.001844   | 4.05x   | -75.29%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.007464 | 0.001491   | 5.00x   | -80.02%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.007464 | 0.001615   | 4.62x   | -78.36%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.010101 | 0.003214   | 3.14x   | -68.19%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.010101 | 0.001475   | 6.85x   | -85.39%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.010101 | 0.001549   | 6.52x   | -84.67%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.009067 | 0.001824   | 4.97x   | -79.89%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.009067 | 0.001446   | 6.27x   | -84.06%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.009067 | 0.001545   | 5.87x   | -82.96%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.007759 | 0.001965   | 3.95x   | -74.68%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.007759 | 0.001763   | 4.40x   | -77.28%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.007759 | 0.001554   | 4.99x   | -79.97%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.007735 | 0.002441   | 3.17x   | -68.44%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.007735 | 0.001517   | 5.10x   | -80.39%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.007735 | 0.001514   | 5.11x   | -80.42%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.007553 | 0.002256   | 3.35x   | -70.13%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.007553 | 0.001449   | 5.21x   | -80.81%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.007553 | 0.001536   | 4.92x   | -79.66%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.007705 | 0.001871   | 4.12x   | -75.72%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.007705 | 0.001435   | 5.37x   | -81.37%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.007705 | 0.001529   | 5.04x   | -80.15%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.009464 | 0.003034   | 3.12x   | -67.94%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.009464 | 0.001578   | 6.00x   | -83.33%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.009464 | 0.001650   | 5.73x   | -82.56%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003514 | 0.001380   | 2.55x   | -60.74%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003514 | 0.000214   | 16.44x  | -93.92%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003514 | 0.000157   | 22.44x  | -95.54%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.007294 | 0.001761   | 4.14x   | -75.85%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.007294 | 0.000252   | 28.98x  | -96.55%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.007294 | 0.000193   | 37.86x  | -97.36%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.005989 | 0.001757   | 3.41x   | -70.66%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.005989 | 0.000246   | 24.33x  | -95.89%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.005989 | 0.000198   | 30.31x  | -96.70%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.005797 | 0.001795   | 3.23x   | -69.04%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.005797 | 0.000231   | 25.11x  | -96.02%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.005797 | 0.000193   | 29.99x  | -96.67%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.006244 | 0.001734   | 3.60x   | -72.23%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.006244 | 0.000248   | 25.17x  | -96.03%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.006244 | 0.000194   | 32.24x  | -96.90%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.006238 | 0.002199   | 2.84x   | -64.75%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.006238 | 0.000276   | 22.64x  | -95.58%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.006238 | 0.000207   | 30.19x  | -96.69%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.005850 | 0.001802   | 3.25x   | -69.19%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.005850 | 0.000219   | 26.77x  | -96.26%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.005850 | 0.000193   | 30.34x  | -96.70%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.005677 | 0.001815   | 3.13x   | -68.03%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.005677 | 0.000232   | 24.51x  | -95.92%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.005677 | 0.000193   | 29.35x  | -96.59%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.007802 | 0.002771   | 2.82x   | -64.48%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.007802 | 0.000256   | 30.53x  | -96.72%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.007802 | 0.000194   | 40.23x  | -97.51%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.007803 | 0.001954   | 3.99x   | -74.95%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.007803 | 0.000234   | 33.40x  | -97.01%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.007803 | 0.000197   | 39.59x  | -97.47%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.005989 | 0.001712   | 3.50x   | -71.40%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.005989 | 0.000277   | 21.59x  | -95.37%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.005989 | 0.000205   | 29.22x  | -96.58%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.005643 | 0.001971   | 2.86x   | -65.06%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.005643 | 0.000227   | 24.83x  | -95.97%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.005643 | 0.000197   | 28.62x  | -96.51%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.005639 | 0.001992   | 2.83x   | -64.68%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.005639 | 0.000232   | 24.28x  | -95.88%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.005639 | 0.000196   | 28.79x  | -96.53%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.006108 | 0.001803   | 3.39x   | -70.48%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.006108 | 0.000239   | 25.51x  | -96.08%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.006108 | 0.000226   | 27.02x  | -96.30%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.007395 | 0.002684   | 2.76x   | -63.71%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.007395 | 0.000255   | 29.03x  | -96.55%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.007395 | 0.000212   | 34.82x  | -97.13%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.006306 | 0.000527   | 11.96x  | -91.64%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.006306 | 0.000184   | 34.26x  | -97.08%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.006306 | 0.000186   | 33.99x  | -97.06%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.009368 | 0.000708   | 13.23x  | -92.44%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.009368 | 0.000252   | 37.11x  | -97.31%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.009368 | 0.000207   | 45.34x  | -97.79%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.007716 | 0.000774   | 9.97x   | -89.97%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.007716 | 0.000221   | 34.91x  | -97.14%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.007716 | 0.000217   | 35.58x  | -97.19%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.008063 | 0.000695   | 11.61x  | -91.38%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.008063 | 0.000236   | 34.10x  | -97.07%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.008063 | 0.000219   | 36.84x  | -97.29%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.007740 | 0.000682   | 11.35x  | -91.19%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.007740 | 0.000260   | 29.79x  | -96.64%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.007740 | 0.000250   | 30.91x  | -96.76%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.007753 | 0.001712   | 4.53x   | -77.91%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.007753 | 0.000297   | 26.13x  | -96.17%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.007753 | 0.000198   | 39.20x  | -97.45%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.007811 | 0.000769   | 10.15x  | -90.15%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.007811 | 0.000237   | 32.96x  | -96.97%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.007811 | 0.000201   | 38.78x  | -97.42%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.007876 | 0.000602   | 13.09x  | -92.36%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.007876 | 0.000227   | 34.76x  | -97.12%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.007876 | 0.000194   | 40.67x  | -97.54%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.009944 | 0.002011   | 4.94x   | -79.78%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.009944 | 0.000260   | 38.21x  | -97.38%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.009944 | 0.000239   | 41.61x  | -97.60%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.010062 | 0.000750   | 13.42x  | -92.55%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.010062 | 0.000230   | 43.80x  | -97.72%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.010062 | 0.000221   | 45.58x  | -97.81%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.007821 | 0.000785   | 9.96x   | -89.96%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.007821 | 0.000279   | 27.99x  | -96.43%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.007821 | 0.000208   | 37.59x  | -97.34%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.007903 | 0.001110   | 7.12x   | -85.95%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.007903 | 0.000257   | 30.73x  | -96.75%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.007903 | 0.000216   | 36.63x  | -97.27%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.007630 | 0.001110   | 6.88x   | -85.46%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.007630 | 0.000269   | 28.33x  | -96.47%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.007630 | 0.000211   | 36.10x  | -97.23%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.007770 | 0.000681   | 11.41x  | -91.24%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.007770 | 0.000276   | 28.17x  | -96.45%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.007770 | 0.000208   | 37.31x  | -97.32%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.008922 | 0.001816   | 4.91x   | -79.65%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.008922 | 0.000252   | 35.37x  | -97.17%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.008922 | 0.000219   | 40.77x  | -97.55%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.004324 | 0.000636   | 6.80x   | -85.30%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.004324 | 0.000309   | 13.99x  | -92.85%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.004324 | 0.000230   | 18.78x  | -94.67%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.006977 | 0.000813   | 8.58x   | -88.35%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.006977 | 0.000362   | 19.28x  | -94.81%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.006977 | 0.000259   | 26.98x  | -96.29%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.005399 | 0.000891   | 6.06x   | -83.50%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.005399 | 0.000370   | 14.57x  | -93.14%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.005399 | 0.000268   | 20.14x  | -95.04%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.005580 | 0.000800   | 6.97x   | -85.66%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.005580 | 0.000366   | 15.24x  | -93.44%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.005580 | 0.000254   | 21.99x  | -95.45%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.005864 | 0.000908   | 6.46x   | -84.51%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.005864 | 0.000375   | 15.62x  | -93.60%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.005864 | 0.000352   | 16.64x  | -93.99%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.005957 | 0.001459   | 4.08x   | -75.51%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.005957 | 0.000404   | 14.75x  | -93.22%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.005957 | 0.000289   | 20.64x  | -95.16%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.005584 | 0.000922   | 6.05x   | -83.48%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.005584 | 0.000374   | 14.93x  | -93.30%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.005584 | 0.000270   | 20.67x  | -95.16%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.005572 | 0.000735   | 7.58x   | -86.80%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.005572 | 0.000370   | 15.07x  | -93.36%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.005572 | 0.000265   | 21.03x  | -95.25%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.007243 | 0.001959   | 3.70x   | -72.95%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.007243 | 0.000361   | 20.05x  | -95.01%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.007243 | 0.000272   | 26.64x  | -96.25%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.007371 | 0.000776   | 9.50x   | -89.48%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.007371 | 0.000368   | 20.05x  | -95.01%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.007371 | 0.000272   | 27.09x  | -96.31%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.005921 | 0.000994   | 5.96x   | -83.22%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.005921 | 0.000389   | 15.20x  | -93.42%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.005921 | 0.000279   | 21.22x  | -95.29%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.005673 | 0.001153   | 4.92x   | -79.67%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.005673 | 0.000383   | 14.80x  | -93.25%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.005673 | 0.000269   | 21.09x  | -95.26%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.005721 | 0.001142   | 5.01x   | -80.04%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.005721 | 0.000368   | 15.54x  | -93.56%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.005721 | 0.000255   | 22.41x  | -95.54%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.005909 | 0.000819   | 7.22x   | -86.14%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.005909 | 0.000403   | 14.65x  | -93.17%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.005909 | 0.000274   | 21.54x  | -95.36%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.007530 | 0.001932   | 3.90x   | -74.34%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.007530 | 0.000380   | 19.80x  | -94.95%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.007530 | 0.000292   | 25.83x  | -96.13%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.035062 | 0.006576   | 5.33x   | -81.25%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.035062 | 0.005525   | 6.35x   | -84.24%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.035062 | 0.005891   | 5.95x   | -83.20%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.047223 | 0.007728   | 6.11x   | -83.64%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.047223 | 0.005850   | 8.07x   | -87.61%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.047223 | 0.006206   | 7.61x   | -86.86%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.038923 | 0.007713   | 5.05x   | -80.18%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.038923 | 0.005649   | 6.89x   | -85.49%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.038923 | 0.005864   | 6.64x   | -84.93%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.039823 | 0.007105   | 5.61x   | -82.16%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.039823 | 0.005758   | 6.92x   | -85.54%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.039823 | 0.006067   | 6.56x   | -84.76%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.040283 | 0.007964   | 5.06x   | -80.23%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.040283 | 0.005989   | 6.73x   | -85.13%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.040283 | 0.006611   | 6.09x   | -83.59%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.039688 | 0.010744   | 3.69x   | -72.93%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.039688 | 0.005658   | 7.01x   | -85.74%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.039688 | 0.006035   | 6.58x   | -84.79%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.039372 | 0.008163   | 4.82x   | -79.27%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.039372 | 0.005774   | 6.82x   | -85.33%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.039372 | 0.005995   | 6.57x   | -84.77%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.039133 | 0.007283   | 5.37x   | -81.39%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.039133 | 0.005902   | 6.63x   | -84.92%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.039133 | 0.006274   | 6.24x   | -83.97%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.047872 | 0.012273   | 3.90x   | -74.36%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.047872 | 0.006018   | 7.96x   | -87.43%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.047872 | 0.005898   | 8.12x   | -87.68%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.045588 | 0.006952   | 6.56x   | -84.75%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.045588 | 0.005602   | 8.14x   | -87.71%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.045588 | 0.005759   | 7.92x   | -87.37%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.037672 | 0.007683   | 4.90x   | -79.60%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.037672 | 0.005650   | 6.67x   | -85.00%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.037672 | 0.005976   | 6.30x   | -84.14%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.039321 | 0.008975   | 4.38x   | -77.17%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.039321 | 0.005648   | 6.96x   | -85.64%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.039321 | 0.005893   | 6.67x   | -85.01%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.038331 | 0.008589   | 4.46x   | -77.59%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.038331 | 0.005590   | 6.86x   | -85.42%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.038331 | 0.005887   | 6.51x   | -84.64%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.039862 | 0.007643   | 5.22x   | -80.83%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.039862 | 0.005797   | 6.88x   | -85.46%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.039862 | 0.006176   | 6.45x   | -84.51%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.046802 | 0.011708   | 4.00x   | -74.98%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.046802 | 0.005775   | 8.10x   | -87.66%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.046802 | 0.006018   | 7.78x   | -87.14%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.025847 | 0.005775   | 4.48x   | -77.66%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.025847 | 0.000799   | 32.34x  | -96.91%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.025847 | 0.000649   | 39.83x  | -97.49%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.037556 | 0.006899   | 5.44x   | -81.63%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.037556 | 0.000968   | 38.80x  | -97.42%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.037556 | 0.000764   | 49.17x  | -97.97%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.029316 | 0.007049   | 4.16x   | -75.95%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.029316 | 0.000894   | 32.81x  | -96.95%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.029316 | 0.000762   | 38.45x  | -97.40%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.033405 | 0.007251   | 4.61x   | -78.29%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.033405 | 0.000918   | 36.37x  | -97.25%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.033405 | 0.000772   | 43.25x  | -97.69%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.030309 | 0.007187   | 4.22x   | -76.29%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.030309 | 0.001003   | 30.22x  | -96.69%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.030309 | 0.000785   | 38.60x  | -97.41%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.029261 | 0.008665   | 3.38x   | -70.39%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.029261 | 0.001103   | 26.52x  | -96.23%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.029261 | 0.000789   | 37.08x  | -97.30%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.029664 | 0.007445   | 3.98x   | -74.90%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.029664 | 0.000894   | 33.19x  | -96.99%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.029664 | 0.000757   | 39.19x  | -97.45%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.030109 | 0.006799   | 4.43x   | -77.42%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.030109 | 0.000984   | 30.61x  | -96.73%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.030109 | 0.000775   | 38.85x  | -97.43%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.038871 | 0.011000   | 3.53x   | -71.70%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.038871 | 0.001117   | 34.80x  | -97.13%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.038871 | 0.000783   | 49.65x  | -97.99%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.036871 | 0.007581   | 4.86x   | -79.44%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.036871 | 0.001180   | 31.25x  | -96.80%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.036871 | 0.000808   | 45.66x  | -97.81%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.029430 | 0.006962   | 4.23x   | -76.34%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.029430 | 0.001179   | 24.96x  | -95.99%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.029430 | 0.000836   | 35.21x  | -97.16%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.031151 | 0.008255   | 3.77x   | -73.50%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.031151 | 0.000915   | 34.04x  | -97.06%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.031151 | 0.000763   | 40.85x  | -97.55%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.031606 | 0.008145   | 3.88x   | -74.23%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.031606 | 0.000944   | 33.47x  | -97.01%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.031606 | 0.000787   | 40.18x  | -97.51%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.031159 | 0.006954   | 4.48x   | -77.68%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.031159 | 0.000952   | 32.72x  | -96.94%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.031159 | 0.000788   | 39.55x  | -97.47%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.039246 | 0.010745   | 3.65x   | -72.62%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.039246 | 0.001029   | 38.15x  | -97.38%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.039246 | 0.000819   | 47.94x  | -97.91%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.029383 | 0.002525   | 11.63x  | -91.40%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.029383 | 0.000754   | 38.99x  | -97.44%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.029383 | 0.000800   | 36.73x  | -97.28%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.040710 | 0.003372   | 12.07x  | -91.72%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.040710 | 0.001083   | 37.58x  | -97.34%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.040710 | 0.000907   | 44.90x  | -97.77%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.034175 | 0.003387   | 10.09x  | -90.09%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.034175 | 0.000918   | 37.24x  | -97.32%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.034175 | 0.000829   | 41.22x  | -97.57%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.034827 | 0.003062   | 11.37x  | -91.21%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.034827 | 0.001078   | 32.32x  | -96.91%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.034827 | 0.000984   | 35.40x  | -97.18%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.035233 | 0.003343   | 10.54x  | -90.51%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.035233 | 0.001185   | 29.74x  | -96.64%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.035233 | 0.001012   | 34.80x  | -97.13%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.033816 | 0.007034   | 4.81x   | -79.20%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.033816 | 0.001098   | 30.79x  | -96.75%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.033816 | 0.000920   | 36.74x  | -97.28%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.034342 | 0.003537   | 9.71x   | -89.70%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.034342 | 0.000954   | 36.01x  | -97.22%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.034342 | 0.000800   | 42.91x  | -97.67%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.034058 | 0.002869   | 11.87x  | -91.58%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.034058 | 0.000954   | 35.69x  | -97.20%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.034058 | 0.000894   | 38.11x  | -97.38%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.042969 | 0.008694   | 4.94x   | -79.77%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.042969 | 0.001203   | 35.71x  | -97.20%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.042969 | 0.000974   | 44.12x  | -97.73%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.042902 | 0.003125   | 13.73x  | -92.72%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.042902 | 0.001001   | 42.86x  | -97.67%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.042902 | 0.000963   | 44.55x  | -97.76%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.036356 | 0.003654   | 9.95x   | -89.95%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.036356 | 0.001147   | 31.68x  | -96.84%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.036356 | 0.000871   | 41.73x  | -97.60%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.033898 | 0.004748   | 7.14x   | -85.99%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.033898 | 0.001040   | 32.61x  | -96.93%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.033898 | 0.001107   | 30.63x  | -96.74%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.035282 | 0.004869   | 7.25x   | -86.20%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.035282 | 0.001076   | 32.80x  | -96.95%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.035282 | 0.000884   | 39.90x  | -97.49%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.035387 | 0.003409   | 10.38x  | -90.37%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.035387 | 0.001090   | 32.45x  | -96.92%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.035387 | 0.000909   | 38.91x  | -97.43%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.042722 | 0.007977   | 5.36x   | -81.33%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.042722 | 0.001151   | 37.11x  | -97.31%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.042722 | 0.000994   | 42.99x  | -97.67%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.021538 | 0.002658   | 8.10x   | -87.66%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.021538 | 0.001392   | 15.47x  | -93.54%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.021538 | 0.001154   | 18.67x  | -94.64%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.033442 | 0.003444   | 9.71x   | -89.70%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.033442 | 0.001447   | 23.12x  | -95.67%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.033442 | 0.001061   | 31.51x  | -96.83%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.025293 | 0.003663   | 6.91x   | -85.52%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.025293 | 0.001563   | 16.18x  | -93.82%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.025293 | 0.001075   | 23.53x  | -95.75%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.026507 | 0.003124   | 8.48x   | -88.21%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.026507 | 0.001464   | 18.11x  | -94.48%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.026507 | 0.001067   | 24.83x  | -95.97%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.024937 | 0.003283   | 7.60x   | -86.84%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.024937 | 0.001473   | 16.93x  | -94.09%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.024937 | 0.001051   | 23.72x  | -95.78%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.023667 | 0.005680   | 4.17x   | -76.00%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.023667 | 0.001556   | 15.21x  | -93.42%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.023667 | 0.001091   | 21.69x  | -95.39%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.024028 | 0.003511   | 6.84x   | -85.39%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.024028 | 0.001485   | 16.18x  | -93.82%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.024028 | 0.001044   | 23.02x  | -95.66%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.024879 | 0.002850   | 8.73x   | -88.54%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.024879 | 0.001481   | 16.80x  | -94.05%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.024879 | 0.001078   | 23.08x  | -95.67%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.033234 | 0.007680   | 4.33x   | -76.89%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.033234 | 0.001465   | 22.68x  | -95.59%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.033234 | 0.001060   | 31.35x  | -96.81%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.032225 | 0.003172   | 10.16x  | -90.16%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.032225 | 0.001480   | 21.77x  | -95.41%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.032225 | 0.001071   | 30.10x  | -96.68%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.024870 | 0.004051   | 6.14x   | -83.71%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.024870 | 0.001556   | 15.98x  | -93.74%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.024870 | 0.001121   | 22.19x  | -95.49%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.025822 | 0.004675   | 5.52x   | -81.90%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.025822 | 0.001494   | 17.28x  | -94.21%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.025822 | 0.001225   | 21.07x  | -95.26%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.025594 | 0.004549   | 5.63x   | -82.22%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.025594 | 0.001499   | 17.07x  | -94.14%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.025594 | 0.001142   | 22.40x  | -95.54%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.026094 | 0.003260   | 8.00x   | -87.51%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.026094 | 0.001506   | 17.33x  | -94.23%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.026094 | 0.001117   | 23.35x  | -95.72%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.033090 | 0.007376   | 4.49x   | -77.71%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.033090 | 0.001433   | 23.09x  | -95.67%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.033090 | 0.001140   | 29.03x  | -96.56%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.099931 | 0.025703   | 3.89x   | -74.28%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.099931 | 0.021588   | 4.63x   | -78.40%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.099931 | 0.023584   | 4.24x   | -76.40%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.133579 | 0.030232   | 4.42x   | -77.37%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.133579 | 0.023122   | 5.78x   | -82.69%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.133579 | 0.023868   | 5.60x   | -82.13%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.109130 | 0.031995   | 3.41x   | -70.68%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.109130 | 0.022547   | 4.84x   | -79.34%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.109130 | 0.023240   | 4.70x   | -78.70%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.108778 | 0.027728   | 3.92x   | -74.51%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.108778 | 0.022214   | 4.90x   | -79.58%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.108778 | 0.023215   | 4.69x   | -78.66%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.110719 | 0.029552   | 3.75x   | -73.31%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.110719 | 0.023158   | 4.78x   | -79.08%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.110719 | 0.023675   | 4.68x   | -78.62%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.104255 | 0.042056   | 2.48x   | -59.66%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.104255 | 0.022445   | 4.64x   | -78.47%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.104255 | 0.023941   | 4.35x   | -77.04%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.104402 | 0.030929   | 3.38x   | -70.37%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.104402 | 0.022447   | 4.65x   | -78.50%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.104402 | 0.023264   | 4.49x   | -77.72%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.104127 | 0.028022   | 3.72x   | -73.09%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.104127 | 0.022387   | 4.65x   | -78.50%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.104127 | 0.023470   | 4.44x   | -77.46%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.142178 | 0.049089   | 2.90x   | -65.47%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.142178 | 0.022982   | 6.19x   | -83.84%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.142178 | 0.023663   | 6.01x   | -83.36%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.135579 | 0.028377   | 4.78x   | -79.07%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.135579 | 0.022338   | 6.07x   | -83.52%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.135579 | 0.023411   | 5.79x   | -82.73%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.103943 | 0.030279   | 3.43x   | -70.87%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.103943 | 0.022410   | 4.64x   | -78.44%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.103943 | 0.023181   | 4.48x   | -77.70%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.105913 | 0.034451   | 3.07x   | -67.47%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.105913 | 0.022295   | 4.75x   | -78.95%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.105913 | 0.023308   | 4.54x   | -77.99%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.107111 | 0.035595   | 3.01x   | -66.77%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.107111 | 0.022808   | 4.70x   | -78.71%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.107111 | 0.023548   | 4.55x   | -78.02%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.109496 | 0.029442   | 3.72x   | -73.11%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.109496 | 0.022610   | 4.84x   | -79.35%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.109496 | 0.023779   | 4.60x   | -78.28%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.136110 | 0.047578   | 2.86x   | -65.04%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.136110 | 0.023075   | 5.90x   | -83.05%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.136110 | 0.025300   | 5.38x   | -81.41%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.073668 | 0.023676   | 3.11x   | -67.86%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.073668 | 0.003087   | 23.86x  | -95.81%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.073668 | 0.002472   | 29.80x  | -96.64%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.109640 | 0.027281   | 4.02x   | -75.12%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.109640 | 0.003759   | 29.17x  | -96.57%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.109640 | 0.003095   | 35.43x  | -97.18%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.079411 | 0.029218   | 2.72x   | -63.21%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.079411 | 0.003590   | 22.12x  | -95.48%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.079411 | 0.003057   | 25.97x  | -96.15%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.084056 | 0.027520   | 3.05x   | -67.26%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.084056 | 0.003741   | 22.47x  | -95.55%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.084056 | 0.003019   | 27.84x  | -96.41%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.080844 | 0.027201   | 2.97x   | -66.35%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.080844 | 0.003927   | 20.59x  | -95.14%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.080844 | 0.003108   | 26.01x  | -96.16%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.078194 | 0.033210   | 2.35x   | -57.53%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.078194 | 0.004427   | 17.66x  | -94.34%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.078194 | 0.003183   | 24.57x  | -95.93%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.079425 | 0.028176   | 2.82x   | -64.53%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.079425 | 0.003508   | 22.64x  | -95.58%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.079425 | 0.003024   | 26.26x  | -96.19%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.079659 | 0.027103   | 2.94x   | -65.98%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.079659 | 0.003558   | 22.39x  | -95.53%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.079659 | 0.003111   | 25.60x  | -96.09%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.122930 | 0.044800   | 2.74x   | -63.56%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.122930 | 0.004207   | 29.22x  | -96.58%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.122930 | 0.003143   | 39.11x  | -97.44%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.114697 | 0.026745   | 4.29x   | -76.68%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.114697 | 0.003743   | 30.64x  | -96.74%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.114697 | 0.003136   | 36.58x  | -97.27%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.083440 | 0.026500   | 3.15x   | -68.24%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.083440 | 0.004340   | 19.23x  | -94.80%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.083440 | 0.003193   | 26.13x  | -96.17%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.080811 | 0.031860   | 2.54x   | -60.57%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.080811 | 0.003717   | 21.74x  | -95.40%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.080811 | 0.003065   | 26.37x  | -96.21%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.082378 | 0.031459   | 2.62x   | -61.81%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.082378 | 0.003679   | 22.39x  | -95.53%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.082378 | 0.003068   | 26.85x  | -96.28%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.082869 | 0.026464   | 3.13x   | -68.07%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.082869 | 0.003723   | 22.26x  | -95.51%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.082869 | 0.003094   | 26.78x  | -96.27%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.110409 | 0.042498   | 2.60x   | -61.51%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.110409 | 0.003967   | 27.83x  | -96.41%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.110409 | 0.003127   | 35.30x  | -97.17%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.087300 | 0.008839   | 9.88x   | -89.87%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.087300 | 0.002971   | 29.38x  | -96.60%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.087300 | 0.003358   | 26.00x  | -96.15%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.121504 | 0.011905   | 10.21x  | -90.20%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.121504 | 0.004210   | 28.86x  | -96.54%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.121504 | 0.003570   | 34.04x  | -97.06%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.093286 | 0.012433   | 7.50x   | -86.67%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.093286 | 0.003789   | 24.62x  | -95.94%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.093286 | 0.003426   | 27.23x  | -96.33%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.098103 | 0.011163   | 8.79x   | -88.62%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.098103 | 0.003922   | 25.01x  | -96.00%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.098103 | 0.003401   | 28.84x  | -96.53%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.095510 | 0.011586   | 8.24x   | -87.87%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.095510 | 0.004422   | 21.60x  | -95.37%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.095510 | 0.003703   | 25.80x  | -96.12%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.094807 | 0.027248   | 3.48x   | -71.26%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.094807 | 0.004422   | 21.44x  | -95.34%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.094807 | 0.003439   | 27.57x  | -96.37%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.097542 | 0.012224   | 7.98x   | -87.47%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.097542 | 0.003728   | 26.17x  | -96.18%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.097542 | 0.003342   | 29.19x  | -96.57%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.093633 | 0.010076   | 9.29x   | -89.24%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.093633 | 0.003770   | 24.84x  | -95.97%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.093633 | 0.003363   | 27.85x  | -96.41%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.129115 | 0.031670   | 4.08x   | -75.47%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.129115 | 0.004245   | 30.42x  | -96.71%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.129115 | 0.003523   | 36.64x  | -97.27%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.123724 | 0.010092   | 12.26x  | -91.84%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.123724 | 0.003695   | 33.48x  | -97.01%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.123724 | 0.003243   | 38.16x  | -97.38%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.091204 | 0.012629   | 7.22x   | -86.15%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.091204 | 0.004460   | 20.45x  | -95.11%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.091204 | 0.003392   | 26.89x  | -96.28%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.095078 | 0.017527   | 5.42x   | -81.57%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.095078 | 0.004079   | 23.31x  | -95.71%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.095078 | 0.003418   | 27.82x  | -96.40%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.095456 | 0.017516   | 5.45x   | -81.65%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.095456 | 0.003932   | 24.28x  | -95.88%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.095456 | 0.003254   | 29.33x  | -96.59%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.098110 | 0.011626   | 8.44x   | -88.15%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.098110 | 0.004128   | 23.76x  | -95.79%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.098110 | 0.003700   | 26.51x  | -96.23%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.124408 | 0.029610   | 4.20x   | -76.20%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.124408 | 0.004051   | 30.71x  | -96.74%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.124408 | 0.003488   | 35.67x  | -97.20%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.064533 | 0.010071   | 6.41x   | -84.39%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.064533 | 0.004962   | 13.01x  | -92.31%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.064533 | 0.003896   | 16.56x  | -93.96%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.099407 | 0.012980   | 7.66x   | -86.94%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.099407 | 0.005868   | 16.94x  | -94.10%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.099407 | 0.004731   | 21.01x  | -95.24%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.074658 | 0.014254   | 5.24x   | -80.91%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.074658 | 0.005980   | 12.48x  | -91.99%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.074658 | 0.004509   | 16.56x  | -93.96%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.077117 | 0.012311   | 6.26x   | -84.04%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.077117 | 0.005708   | 13.51x  | -92.60%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.077117 | 0.004137   | 18.64x  | -94.64%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.084988 | 0.013851   | 6.14x   | -83.70%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.084988 | 0.005871   | 14.48x  | -93.09%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.084988 | 0.004500   | 18.89x  | -94.70%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.072923 | 0.022784   | 3.20x   | -68.76%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.072923 | 0.006194   | 11.77x  | -91.51%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.072923 | 0.004510   | 16.17x  | -93.82%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.072938 | 0.014135   | 5.16x   | -80.62%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.072938 | 0.005928   | 12.30x  | -91.87%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.072938 | 0.004434   | 16.45x  | -93.92%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.075221 | 0.011318   | 6.65x   | -84.95%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.075221 | 0.006029   | 12.48x  | -91.99%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.075221 | 0.005782   | 13.01x  | -92.31%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.112051 | 0.031235   | 3.59x   | -72.12%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.112051 | 0.005772   | 19.41x  | -94.85%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.112051 | 0.004450   | 25.18x  | -96.03%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.103980 | 0.011733   | 8.86x   | -88.72%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.103980 | 0.005850   | 17.78x  | -94.37%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.103980 | 0.004549   | 22.86x  | -95.62%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.072947 | 0.015279   | 4.77x   | -79.05%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.072947 | 0.006174   | 11.82x  | -91.54%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.072947 | 0.004552   | 16.03x  | -93.76%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.073738 | 0.018060   | 4.08x   | -75.51%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.073738 | 0.005812   | 12.69x  | -92.12%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.073738 | 0.004402   | 16.75x  | -94.03%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.075314 | 0.020753   | 3.63x   | -72.44%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.075314 | 0.005831   | 12.92x  | -92.26%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.075314 | 0.005205   | 14.47x  | -93.09%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.076967 | 0.012777   | 6.02x   | -83.40%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.076967 | 0.006169   | 12.48x  | -91.99%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.076967 | 0.004548   | 16.92x  | -94.09%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.101974 | 0.029386   | 3.47x   | -71.18%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.101974 | 0.005625   | 18.13x  | -94.48%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.101974 | 0.004474   | 22.79x  | -95.61%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.381713 | 0.102504   | 3.72x   | -73.15%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.381713 | 0.086612   | 4.41x   | -77.31%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.381713 | 0.092152   | 4.14x   | -75.86%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.501368 | 0.118773   | 4.22x   | -76.31%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.501368 | 0.090790   | 5.52x   | -81.89%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.501368 | 0.095309   | 5.26x   | -80.99%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.385143 | 0.127172   | 3.03x   | -66.98%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.385143 | 0.090387   | 4.26x   | -76.53%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.385143 | 0.096035   | 4.01x   | -75.07%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.394091 | 0.117338   | 3.36x   | -70.23%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.394091 | 0.093743   | 4.20x   | -76.21%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.394091 | 0.094749   | 4.16x   | -75.96%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.404379 | 0.118253   | 3.42x   | -70.76%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.404379 | 0.090528   | 4.47x   | -77.61%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.404379 | 0.095503   | 4.23x   | -76.38%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.401937 | 0.171019   | 2.35x   | -57.45%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.401937 | 0.089278   | 4.50x   | -77.79%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.401937 | 0.093693   | 4.29x   | -76.69%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.386356 | 0.128830   | 3.00x   | -66.65%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.386356 | 0.092438   | 4.18x   | -76.07%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.386356 | 0.096835   | 3.99x   | -74.94%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.395222 | 0.123522   | 3.20x   | -68.75%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.395222 | 0.093999   | 4.20x   | -76.22%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.395222 | 0.100561   | 3.93x   | -74.56%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.577060 | 0.200104   | 2.88x   | -65.32%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.577060 | 0.092837   | 6.22x   | -83.91%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.577060 | 0.095269   | 6.06x   | -83.49%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.496629 | 0.111218   | 4.47x   | -77.61%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.496629 | 0.099219   | 5.01x   | -80.02%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.496629 | 0.093380   | 5.32x   | -81.20%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.384286 | 0.116367   | 3.30x   | -69.72%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.384286 | 0.089894   | 4.27x   | -76.61%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.384286 | 0.094466   | 4.07x   | -75.42%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.396192 | 0.138395   | 2.86x   | -65.07%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.396192 | 0.089727   | 4.42x   | -77.35%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.396192 | 0.093768   | 4.23x   | -76.33%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.401572 | 0.138440   | 2.90x   | -65.53%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.401572 | 0.089892   | 4.47x   | -77.61%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.401572 | 0.094891   | 4.23x   | -76.37%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.408808 | 0.117231   | 3.49x   | -71.32%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.408808 | 0.090449   | 4.52x   | -77.87%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.408808 | 0.096097   | 4.25x   | -76.49%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.510314 | 0.193993   | 2.63x   | -61.99%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.510314 | 0.091755   | 5.56x   | -82.02%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.510314 | 0.097504   | 5.23x   | -80.89%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.288479 | 0.085592   | 3.37x   | -70.33%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.288479 | 0.012271   | 23.51x  | -95.75%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.288479 | 0.009933   | 29.04x  | -96.56%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.400929 | 0.113758   | 3.52x   | -71.63%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.400929 | 0.015500   | 25.87x  | -96.13%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.400929 | 0.012490   | 32.10x  | -96.88%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.279973 | 0.112591   | 2.49x   | -59.78%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.279973 | 0.014202   | 19.71x  | -94.93%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.279973 | 0.012178   | 22.99x  | -95.65%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.300975 | 0.105118   | 2.86x   | -65.07%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.300975 | 0.014414   | 20.88x  | -95.21%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.300975 | 0.012012   | 25.06x  | -96.01%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.303567 | 0.108968   | 2.79x   | -64.10%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.303567 | 0.015716   | 19.32x  | -94.82%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.303567 | 0.012402   | 24.48x  | -95.91%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.288450 | 0.131904   | 2.19x   | -54.27%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.288450 | 0.017270   | 16.70x  | -94.01%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.288450 | 0.012624   | 22.85x  | -95.62%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.277005 | 0.117589   | 2.36x   | -57.55%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.277005 | 0.014551   | 19.04x  | -94.75%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.277005 | 0.012287   | 22.54x  | -95.56%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.287523 | 0.105899   | 2.72x   | -63.17%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.287523 | 0.014030   | 20.49x  | -95.12%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.287523 | 0.011990   | 23.98x  | -95.83%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.444917 | 0.172573   | 2.58x   | -61.21%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.444917 | 0.017013   | 26.15x  | -96.18%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.444917 | 0.012494   | 35.61x  | -97.19%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.410013 | 0.104008   | 3.94x   | -74.63%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.410013 | 0.014093   | 29.09x  | -96.56%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.410013 | 0.012259   | 33.45x  | -97.01%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.292372 | 0.104345   | 2.80x   | -64.31%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.292372 | 0.017360   | 16.84x  | -94.06%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.292372 | 0.012737   | 22.95x  | -95.64%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.302710 | 0.126742   | 2.39x   | -58.13%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.302710 | 0.014521   | 20.85x  | -95.20%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.302710 | 0.012209   | 24.79x  | -95.97%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.295306 | 0.125435   | 2.35x   | -57.52%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.295306 | 0.014485   | 20.39x  | -95.09%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.295306 | 0.012092   | 24.42x  | -95.91%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.300816 | 0.110078   | 2.73x   | -63.41%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.300816 | 0.015260   | 19.71x  | -94.93%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.300816 | 0.012185   | 24.69x  | -95.95%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.414023 | 0.167239   | 2.48x   | -59.61%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.414023 | 0.016130   | 25.67x  | -96.10%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.414023 | 0.012507   | 33.10x  | -96.98%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.344621 | 0.037792   | 9.12x   | -89.03%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.344621 | 0.016611   | 20.75x  | -95.18%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.344621 | 0.018805   | 18.33x  | -94.54%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.458798 | 0.055311   | 8.29x   | -87.94%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.458798 | 0.022578   | 20.32x  | -95.08%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.458798 | 0.020912   | 21.94x  | -95.44%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.357364 | 0.055491   | 6.44x   | -84.47%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.357364 | 0.021121   | 16.92x  | -94.09%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.357364 | 0.020132   | 17.75x  | -94.37%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.370255 | 0.047977   | 7.72x   | -87.04%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.370255 | 0.021609   | 17.13x  | -94.16%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.370255 | 0.019050   | 19.44x  | -94.85%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.357963 | 0.051146   | 7.00x   | -85.71%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.357963 | 0.022610   | 15.83x  | -93.68%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.357963 | 0.019833   | 18.05x  | -94.46%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.361765 | 0.111084   | 3.26x   | -69.29%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.361765 | 0.025640   | 14.11x  | -92.91%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.361765 | 0.024344   | 14.86x  | -93.27%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.371895 | 0.053447   | 6.96x   | -85.63%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.371895 | 0.021543   | 17.26x  | -94.21%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.371895 | 0.019568   | 19.01x  | -94.74%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.365746 | 0.043143   | 8.48x   | -88.20%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.365746 | 0.021140   | 17.30x  | -94.22%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.365746 | 0.020400   | 17.93x  | -94.42%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.537909 | 0.130004   | 4.14x   | -75.83%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.537909 | 0.022514   | 23.89x  | -95.81%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.537909 | 0.019990   | 26.91x  | -96.28%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.480436 | 0.045355   | 10.59x  | -90.56%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.480436 | 0.021293   | 22.56x  | -95.57%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.480436 | 0.019110   | 25.14x  | -96.02%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.352517 | 0.053238   | 6.62x   | -84.90%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.352517 | 0.022548   | 15.63x  | -93.60%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.352517 | 0.018580   | 18.97x  | -94.73%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.343982 | 0.072321   | 4.76x   | -78.98%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.343982 | 0.020731   | 16.59x  | -93.97%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.343982 | 0.018156   | 18.95x  | -94.72%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.363517 | 0.075233   | 4.83x   | -79.30%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.363517 | 0.021568   | 16.85x  | -94.07%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.363517 | 0.019176   | 18.96x  | -94.72%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.375052 | 0.046874   | 8.00x   | -87.50%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.375052 | 0.020702   | 18.12x  | -94.48%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.375052 | 0.018489   | 20.29x  | -95.07%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.472964 | 0.123955   | 3.82x   | -73.79%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.472964 | 0.022877   | 20.67x  | -95.16%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.472964 | 0.020179   | 23.44x  | -95.73%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.274895 | 0.049193   | 5.59x   | -82.10%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.274895 | 0.028385   | 9.68x   | -89.67%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.274895 | 0.024831   | 11.07x  | -90.97%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.391394 | 0.064763   | 6.04x   | -83.45%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.391394 | 0.030305   | 12.92x  | -92.26%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.391394 | 0.025879   | 15.12x  | -93.39%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.262592 | 0.068467   | 3.84x   | -73.93%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.262592 | 0.030386   | 8.64x   | -88.43%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.262592 | 0.024716   | 10.62x  | -90.59%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.279428 | 0.059956   | 4.66x   | -78.54%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.279428 | 0.030001   | 9.31x   | -89.26%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.279428 | 0.025637   | 10.90x  | -90.83%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.286456 | 0.063341   | 4.52x   | -77.89%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.286456 | 0.030381   | 9.43x   | -89.39%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.286456 | 0.026675   | 10.74x  | -90.69%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.269228 | 0.101652   | 2.65x   | -62.24%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.269228 | 0.033060   | 8.14x   | -87.72%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.269228 | 0.027314   | 9.86x   | -89.85%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.262523 | 0.067402   | 3.89x   | -74.33%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.262523 | 0.031122   | 8.44x   | -88.15%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.262523 | 0.025574   | 10.27x  | -90.26%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.277050 | 0.060040   | 4.61x   | -78.33%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.277050 | 0.030814   | 8.99x   | -88.88%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.277050 | 0.025531   | 10.85x  | -90.78%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.440431 | 0.136688   | 3.22x   | -68.97%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.440431 | 0.032791   | 13.43x  | -92.55%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.440431 | 0.027097   | 16.25x  | -93.85%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.403430 | 0.061411   | 6.57x   | -84.78%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.403430 | 0.031807   | 12.68x  | -92.12%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.403430 | 0.026005   | 15.51x  | -93.55%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.283334 | 0.075872   | 3.73x   | -73.22%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.283334 | 0.031897   | 8.88x   | -88.74%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.283334 | 0.026470   | 10.70x  | -90.66%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.296589 | 0.084704   | 3.50x   | -71.44%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.296589 | 0.031546   | 9.40x   | -89.36%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.296589 | 0.025725   | 11.53x  | -91.33%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.290800 | 0.083616   | 3.48x   | -71.25%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.290800 | 0.031415   | 9.26x   | -89.20%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.290800 | 0.025824   | 11.26x  | -91.12%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.292352 | 0.065341   | 4.47x   | -77.65%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.292352 | 0.030336   | 9.64x   | -89.62%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.292352 | 0.025324   | 11.54x  | -91.34%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.393777 | 0.127874   | 3.08x   | -67.53%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.393777 | 0.029616   | 13.30x  | -92.48%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.393777 | 0.025883   | 15.21x  | -93.43%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.084909 | 0.023177   | 3.66x   | -72.70%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.084909 | 0.019628   | 4.33x   | -76.88%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.084909 | 0.020875   | 4.07x   | -75.42%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.114998 | 0.026270   | 4.38x   | -77.16%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.114998 | 0.020636   | 5.57x   | -82.06%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.114998 | 0.021182   | 5.43x   | -81.58%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.088959 | 0.026636   | 3.34x   | -70.06%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.088959 | 0.019644   | 4.53x   | -77.92%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.088959 | 0.020329   | 4.38x   | -77.15%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.090565 | 0.024697   | 3.67x   | -72.73%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.090565 | 0.019546   | 4.63x   | -78.42%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.090565 | 0.020355   | 4.45x   | -77.52%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.089666 | 0.025836   | 3.47x   | -71.19%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.089666 | 0.020807   | 4.31x   | -76.79%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.089666 | 0.021092   | 4.25x   | -76.48%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.086396 | 0.037166   | 2.32x   | -56.98%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.086396 | 0.019915   | 4.34x   | -76.95%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.086396 | 0.020898   | 4.13x   | -75.81%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.088131 | 0.026786   | 3.29x   | -69.61%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.088131 | 0.019905   | 4.43x   | -77.41%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.088131 | 0.020814   | 4.23x   | -76.38%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.089246 | 0.024985   | 3.57x   | -72.00%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.089246 | 0.020045   | 4.45x   | -77.54%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.089246 | 0.021044   | 4.24x   | -76.42%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.120575 | 0.043328   | 2.78x   | -64.07%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.120575 | 0.019806   | 6.09x   | -83.57%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.120575 | 0.020752   | 5.81x   | -82.79%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.116878 | 0.024648   | 4.74x   | -78.91%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.116878 | 0.019609   | 5.96x   | -83.22%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.116878 | 0.020289   | 5.76x   | -82.64%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.086012 | 0.025522   | 3.37x   | -70.33%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.086012 | 0.019326   | 4.45x   | -77.53%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.086012 | 0.020536   | 4.19x   | -76.12%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.090632 | 0.030168   | 3.00x   | -66.71%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.090632 | 0.019812   | 4.57x   | -78.14%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.090632 | 0.020684   | 4.38x   | -77.18%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.090362 | 0.031403   | 2.88x   | -65.25%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.090362 | 0.019593   | 4.61x   | -78.32%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.090362 | 0.020558   | 4.40x   | -77.25%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.092291 | 0.025552   | 3.61x   | -72.31%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.092291 | 0.019786   | 4.66x   | -78.56%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.092291 | 0.020758   | 4.45x   | -77.51%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.113917 | 0.040760   | 2.79x   | -64.22%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.113917 | 0.019814   | 5.75x   | -82.61%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.113917 | 0.021609   | 5.27x   | -81.03%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.063064 | 0.018994   | 3.32x   | -69.88%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.063064 | 0.002929   | 21.53x  | -95.36%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.063064 | 0.002192   | 28.77x  | -96.52%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.099263 | 0.024803   | 4.00x   | -75.01%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.099263 | 0.003383   | 29.34x  | -96.59%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.099263 | 0.002699   | 36.77x  | -97.28%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.075748 | 0.024500   | 3.09x   | -67.66%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.075748 | 0.003049   | 24.84x  | -95.97%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.075748 | 0.002617   | 28.94x  | -96.54%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.080190 | 0.022809   | 3.52x   | -71.56%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.080190 | 0.003180   | 25.21x  | -96.03%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.080190 | 0.002660   | 30.15x  | -96.68%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.077922 | 0.024333   | 3.20x   | -68.77%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.077922 | 0.003543   | 21.99x  | -95.45%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.077922 | 0.003085   | 25.26x  | -96.04%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.074700 | 0.029244   | 2.55x   | -60.85%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.074700 | 0.003902   | 19.14x  | -94.78%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.074700 | 0.002841   | 26.30x  | -96.20%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.074797 | 0.024895   | 3.00x   | -66.72%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.074797 | 0.003074   | 24.33x  | -95.89%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.074797 | 0.002622   | 28.53x  | -96.49%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.075843 | 0.023696   | 3.20x   | -68.76%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.075843 | 0.003079   | 24.63x  | -95.94%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.075843 | 0.002608   | 29.08x  | -96.56%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.107547 | 0.038601   | 2.79x   | -64.11%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.107547 | 0.003612   | 29.77x  | -96.64%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.107547 | 0.002845   | 37.80x  | -97.35%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.103059 | 0.023646   | 4.36x   | -77.06%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.103059 | 0.003206   | 32.14x  | -96.89%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.103059 | 0.002676   | 38.51x  | -97.40%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.074750 | 0.022816   | 3.28x   | -69.48%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.074750 | 0.003858   | 19.38x  | -94.84%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.074750 | 0.002808   | 26.62x  | -96.24%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.077000 | 0.027424   | 2.81x   | -64.38%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.077000 | 0.003223   | 23.89x  | -95.81%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.077000 | 0.002725   | 28.26x  | -96.46%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.076282 | 0.027640   | 2.76x   | -63.77%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.076282 | 0.003224   | 23.66x  | -95.77%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.076282 | 0.002681   | 28.45x  | -96.49%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.080921 | 0.024458   | 3.31x   | -69.78%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.080921 | 0.003337   | 24.25x  | -95.88%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.080921 | 0.002724   | 29.71x  | -96.63%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.102968 | 0.037991   | 2.71x   | -63.10%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.102968 | 0.003485   | 29.54x  | -96.62%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.102968 | 0.002727   | 37.76x  | -97.35%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.080857 | 0.007168   | 11.28x  | -91.13%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.080857 | 0.002569   | 31.47x  | -96.82%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.080857 | 0.002827   | 28.60x  | -96.50%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.113678 | 0.009400   | 12.09x  | -91.73%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.113678 | 0.003479   | 32.67x  | -96.94%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.113678 | 0.003037   | 37.43x  | -97.33%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.081291 | 0.010722   | 7.58x   | -86.81%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.081291 | 0.003617   | 22.47x  | -95.55%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.081291 | 0.003176   | 25.60x  | -96.09%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.084321 | 0.008882   | 9.49x   | -89.47%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.084321 | 0.003418   | 24.67x  | -95.95%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.084321 | 0.003042   | 27.71x  | -96.39%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.085452 | 0.009738   | 8.78x   | -88.60%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.085452 | 0.003810   | 22.43x  | -95.54%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.085452 | 0.003558   | 24.02x  | -95.84%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.095222 | 0.024587   | 3.87x   | -74.18%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.095222 | 0.003901   | 24.41x  | -95.90%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.095222 | 0.004449   | 21.40x  | -95.33%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.095711 | 0.010937   | 8.75x   | -88.57%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.095711 | 0.003602   | 26.57x  | -96.24%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.095711 | 0.003061   | 31.27x  | -96.80%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.083891 | 0.008195   | 10.24x  | -90.23%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.083891 | 0.003131   | 26.79x  | -96.27%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.083891 | 0.002816   | 29.79x  | -96.64%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.111275 | 0.027246   | 4.08x   | -75.51%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.111275 | 0.003981   | 27.95x  | -96.42%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.111275 | 0.003187   | 34.91x  | -97.14%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.108993 | 0.008647   | 12.60x  | -92.07%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.108993 | 0.003613   | 30.17x  | -96.69%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.108993 | 0.003341   | 32.62x  | -96.93%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.079282 | 0.011146   | 7.11x   | -85.94%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.079282 | 0.003903   | 20.31x  | -95.08%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.079282 | 0.003095   | 25.62x  | -96.10%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.083736 | 0.014886   | 5.63x   | -82.22%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.083736 | 0.003474   | 24.10x  | -95.85%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.083736 | 0.002964   | 28.25x  | -96.46%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.082397 | 0.014822   | 5.56x   | -82.01%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.082397 | 0.003466   | 23.77x  | -95.79%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.082397 | 0.002920   | 28.22x  | -96.46%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.085056 | 0.009277   | 9.17x   | -89.09%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.085056 | 0.003561   | 23.89x  | -95.81%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.085056 | 0.003132   | 27.16x  | -96.32%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.108389 | 0.025565   | 4.24x   | -76.41%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.108389 | 0.003578   | 30.29x  | -96.70%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.108389 | 0.003071   | 35.30x  | -97.17%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.058313 | 0.008909   | 6.55x   | -84.72%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.058313 | 0.004375   | 13.33x  | -92.50%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.058313 | 0.003399   | 17.16x  | -94.17%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.090383 | 0.011262   | 8.03x   | -87.54%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.090383 | 0.005017   | 18.02x  | -94.45%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.090383 | 0.003627   | 24.92x  | -95.99%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.067266 | 0.012617   | 5.33x   | -81.24%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.067266 | 0.005189   | 12.96x  | -92.29%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.067266 | 0.003896   | 17.26x  | -94.21%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.070462 | 0.011429   | 6.16x   | -83.78%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.070462 | 0.005108   | 13.80x  | -92.75%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.070462 | 0.003958   | 17.80x  | -94.38%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.069575 | 0.011576   | 6.01x   | -83.36%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.069575 | 0.005115   | 13.60x  | -92.65%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.069575 | 0.003827   | 18.18x  | -94.50%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.065686 | 0.020126   | 3.26x   | -69.36%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.065686 | 0.005439   | 12.08x  | -91.72%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.065686 | 0.004024   | 16.32x  | -93.87%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.067469 | 0.012534   | 5.38x   | -81.42%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.067469 | 0.005193   | 12.99x  | -92.30%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.067469 | 0.003877   | 17.40x  | -94.25%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.069726 | 0.010206   | 6.83x   | -85.36%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.069726 | 0.005182   | 13.46x  | -92.57%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.069726 | 0.003793   | 18.38x  | -94.56%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.099357 | 0.027799   | 3.57x   | -72.02%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.099357 | 0.005083   | 19.55x  | -94.88%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.099357 | 0.003995   | 24.87x  | -95.98%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.096640 | 0.011046   | 8.75x   | -88.57%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.096640 | 0.005144   | 18.79x  | -94.68%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.096640 | 0.004031   | 23.97x  | -95.83%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.070831 | 0.013666   | 5.18x   | -80.71%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.070831 | 0.005401   | 13.11x  | -92.38%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.070831 | 0.004076   | 17.38x  | -94.25%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.068639 | 0.015905   | 4.32x   | -76.83%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.068639 | 0.005154   | 13.32x  | -92.49%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.068639 | 0.003902   | 17.59x  | -94.32%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.071853 | 0.016429   | 4.37x   | -77.14%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.071853 | 0.005488   | 13.09x  | -92.36%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.071853 | 0.004274   | 16.81x  | -94.05%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.073949 | 0.011236   | 6.58x   | -84.81%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.073949 | 0.005240   | 14.11x  | -92.91%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.073949 | 0.003848   | 19.22x  | -94.80%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.093875 | 0.026452   | 3.55x   | -71.82%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.093875 | 0.004942   | 19.00x  | -94.74%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.093875 | 0.003822   | 24.56x  | -95.93%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.197262 | 0.051043   | 3.86x   | -74.12%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.197262 | 0.044062   | 4.48x   | -77.66%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.197262 | 0.047455   | 4.16x   | -75.94%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.246416 | 0.058582   | 4.21x   | -76.23%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.246416 | 0.044848   | 5.49x   | -81.80%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.246416 | 0.046765   | 5.27x   | -81.02%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.198747 | 0.060233   | 3.30x   | -69.69%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.198747 | 0.044588   | 4.46x   | -77.57%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.198747 | 0.047661   | 4.17x   | -76.02%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.208663 | 0.057886   | 3.60x   | -72.26%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.208663 | 0.045116   | 4.63x   | -78.38%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.208663 | 0.046280   | 4.51x   | -77.82%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.198087 | 0.059414   | 3.33x   | -70.01%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.198087 | 0.045283   | 4.37x   | -77.14%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.198087 | 0.047273   | 4.19x   | -76.14%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.201532 | 0.085256   | 2.36x   | -57.70%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.201532 | 0.045121   | 4.47x   | -77.61%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.201532 | 0.047027   | 4.29x   | -76.67%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.198612 | 0.061814   | 3.21x   | -68.88%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.198612 | 0.047473   | 4.18x   | -76.10%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.198612 | 0.048497   | 4.10x   | -75.58%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.197944 | 0.056496   | 3.50x   | -71.46%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.197944 | 0.045292   | 4.37x   | -77.12%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.197944 | 0.047288   | 4.19x   | -76.11%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.265009 | 0.098595   | 2.69x   | -62.80%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.265009 | 0.047453   | 5.58x   | -82.09%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.265009 | 0.046850   | 5.66x   | -82.32%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.253858 | 0.055153   | 4.60x   | -78.27%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.253858 | 0.044475   | 5.71x   | -82.48%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.253858 | 0.046298   | 5.48x   | -81.76%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.198908 | 0.058100   | 3.42x   | -70.79%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.198908 | 0.043774   | 4.54x   | -77.99%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.198908 | 0.046390   | 4.29x   | -76.68%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.197350 | 0.069071   | 2.86x   | -65.00%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.197350 | 0.046701   | 4.23x   | -76.34%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.197350 | 0.046035   | 4.29x   | -76.67%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.193568 | 0.069816   | 2.77x   | -63.93%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.193568 | 0.045345   | 4.27x   | -76.57%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.193568 | 0.046668   | 4.15x   | -75.89%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.201030 | 0.057147   | 3.52x   | -71.57%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.201030 | 0.044723   | 4.49x   | -77.75%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.201030 | 0.048375   | 4.16x   | -75.94%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.277264 | 0.094358   | 2.94x   | -65.97%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.277264 | 0.046383   | 5.98x   | -83.27%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.277264 | 0.048490   | 5.72x   | -82.51%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.138705 | 0.046201   | 3.00x   | -66.69%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.138705 | 0.006531   | 21.24x  | -95.29%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.138705 | 0.004993   | 27.78x  | -96.40%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.197646 | 0.054915   | 3.60x   | -72.22%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.197646 | 0.007344   | 26.91x  | -96.28%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.197646 | 0.006003   | 32.92x  | -96.96%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.159902 | 0.056165   | 2.85x   | -64.88%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.159902 | 0.006877   | 23.25x  | -95.70%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.159902 | 0.005927   | 26.98x  | -96.29%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.157226 | 0.052082   | 3.02x   | -66.87%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.157226 | 0.007057   | 22.28x  | -95.51%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.157226 | 0.005906   | 26.62x  | -96.24%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.151613 | 0.057105   | 2.65x   | -62.33%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.151613 | 0.007838   | 19.34x  | -94.83%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.151613 | 0.006292   | 24.10x  | -95.85%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.145762 | 0.064795   | 2.25x   | -55.55%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.145762 | 0.008494   | 17.16x  | -94.17%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.145762 | 0.006165   | 23.64x  | -95.77%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.141619 | 0.054724   | 2.59x   | -61.36%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.141619 | 0.007130   | 19.86x  | -94.97%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.141619 | 0.005938   | 23.85x  | -95.81%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.157751 | 0.056256   | 2.80x   | -64.34%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.157751 | 0.006988   | 22.58x  | -95.57%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.157751 | 0.005909   | 26.70x  | -96.25%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.230572 | 0.087758   | 2.63x   | -61.94%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.230572 | 0.007982   | 28.89x  | -96.54%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.230572 | 0.006034   | 38.21x  | -97.38%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.203173 | 0.054376   | 3.74x   | -73.24%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.203173 | 0.007019   | 28.95x  | -96.55%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.203173 | 0.005939   | 34.21x  | -97.08%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.144971 | 0.051170   | 2.83x   | -64.70%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.144971 | 0.008609   | 16.84x  | -94.06%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.144971 | 0.006269   | 23.12x  | -95.68%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.147485 | 0.065265   | 2.26x   | -55.75%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.147485 | 0.007268   | 20.29x  | -95.07%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.147485 | 0.006026   | 24.48x  | -95.91%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.149981 | 0.065613   | 2.29x   | -56.25%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.149981 | 0.007218   | 20.78x  | -95.19%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.149981 | 0.006239   | 24.04x  | -95.84%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.156471 | 0.052906   | 2.96x   | -66.19%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.156471 | 0.007415   | 21.10x  | -95.26%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.156471 | 0.006071   | 25.77x  | -96.12%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.229674 | 0.082474   | 2.78x   | -64.09%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.229674 | 0.008082   | 28.42x  | -96.48%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.229674 | 0.006107   | 37.61x  | -97.34%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.164164 | 0.018730   | 8.76x   | -88.59%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.164164 | 0.005718   | 28.71x  | -96.52%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.164164 | 0.006303   | 26.04x  | -96.16%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.224458 | 0.024254   | 9.25x   | -89.19%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.224458 | 0.007897   | 28.42x  | -96.48%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.224458 | 0.006572   | 34.16x  | -97.07%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.173785 | 0.025007   | 6.95x   | -85.61%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.173785 | 0.007023   | 24.74x  | -95.96%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.173785 | 0.006207   | 28.00x  | -96.43%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.172223 | 0.022096   | 7.79x   | -87.17%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.172223 | 0.007711   | 22.33x  | -95.52%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.172223 | 0.006589   | 26.14x  | -96.17%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.171486 | 0.024470   | 7.01x   | -85.73%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.171486 | 0.008556   | 20.04x  | -95.01%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.171486 | 0.007254   | 23.64x  | -95.77%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.171124 | 0.054678   | 3.13x   | -68.05%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.171124 | 0.008605   | 19.89x  | -94.97%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.171124 | 0.006837   | 25.03x  | -96.00%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.167439 | 0.025944   | 6.45x   | -84.51%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.167439 | 0.007128   | 23.49x  | -95.74%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.167439 | 0.006496   | 25.78x  | -96.12%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.175539 | 0.021263   | 8.26x   | -87.89%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.175539 | 0.007121   | 24.65x  | -95.94%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.175539 | 0.007830   | 22.42x  | -95.54%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.238928 | 0.063588   | 3.76x   | -73.39%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.238928 | 0.008188   | 29.18x  | -96.57%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.238928 | 0.007237   | 33.02x  | -96.97%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.234960 | 0.022730   | 10.34x  | -90.33%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.234960 | 0.007198   | 32.64x  | -96.94%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.234960 | 0.006270   | 37.47x  | -97.33%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.169986 | 0.026735   | 6.36x   | -84.27%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.169986 | 0.008705   | 19.53x  | -94.88%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.169986 | 0.006514   | 26.10x  | -96.17%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.172041 | 0.036799   | 4.68x   | -78.61%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.172041 | 0.008116   | 21.20x  | -95.28%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.172041 | 0.006938   | 24.80x  | -95.97%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.174294 | 0.037034   | 4.71x   | -78.75%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.174294 | 0.008068   | 21.60x  | -95.37%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.174294 | 0.006986   | 24.95x  | -95.99%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.176589 | 0.024363   | 7.25x   | -86.20%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.176589 | 0.008144   | 21.68x  | -95.39%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.176589 | 0.007170   | 24.63x  | -95.94%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.226485 | 0.059970   | 3.78x   | -73.52%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.226485 | 0.008159   | 27.76x  | -96.40%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.226485 | 0.006661   | 34.00x  | -97.06%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.126739 | 0.019768   | 6.41x   | -84.40%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.126739 | 0.009793   | 12.94x  | -92.27%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.126739 | 0.007615   | 16.64x  | -93.99%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.191064 | 0.026425   | 7.23x   | -86.17%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.191064 | 0.011179   | 17.09x  | -94.15%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.191064 | 0.008156   | 23.43x  | -95.73%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.132244 | 0.028928   | 4.57x   | -78.13%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.132244 | 0.011581   | 11.42x  | -91.24%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.132244 | 0.008617   | 15.35x  | -93.48%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.140572 | 0.024759   | 5.68x   | -82.39%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.140572 | 0.011244   | 12.50x  | -92.00%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.140572 | 0.008605   | 16.34x  | -93.88%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.143463 | 0.027618   | 5.19x   | -80.75%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.143463 | 0.011381   | 12.61x  | -92.07%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.143463 | 0.008847   | 16.22x  | -93.83%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.140750 | 0.045502   | 3.09x   | -67.67%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.140750 | 0.012144   | 11.59x  | -91.37%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.140750 | 0.008956   | 15.72x  | -93.64%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.136533 | 0.028513   | 4.79x   | -79.12%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.136533 | 0.011656   | 11.71x  | -91.46%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.136533 | 0.008994   | 15.18x  | -93.41%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.138268 | 0.023358   | 5.92x   | -83.11%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.138268 | 0.011697   | 11.82x  | -91.54%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.138268 | 0.008696   | 15.90x  | -93.71%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.208559 | 0.062856   | 3.32x   | -69.86%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.208559 | 0.011089   | 18.81x  | -94.68%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.208559 | 0.008434   | 24.73x  | -95.96%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.201923 | 0.024492   | 8.24x   | -87.87%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.201923 | 0.011460   | 17.62x  | -94.32%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.201923 | 0.008456   | 23.88x  | -95.81%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.137224 | 0.030918   | 4.44x   | -77.47%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.137224 | 0.012253   | 11.20x  | -91.07%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.137224 | 0.009380   | 14.63x  | -93.16%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.141591 | 0.037315   | 3.79x   | -73.65%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.141591 | 0.011645   | 12.16x  | -91.78%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.141591 | 0.008744   | 16.19x  | -93.82%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.140081 | 0.035919   | 3.90x   | -74.36%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.140081 | 0.011264   | 12.44x  | -91.96%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.140081 | 0.008702   | 16.10x  | -93.79%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.141185 | 0.029974   | 4.71x   | -78.77%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.141185 | 0.011994   | 11.77x  | -91.50%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.141185 | 0.009804   | 14.40x  | -93.06%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.198067 | 0.060192   | 3.29x   | -69.61%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.198067 | 0.011134   | 17.79x  | -94.38%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.198067 | 0.008532   | 23.22x  | -95.69%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.333183 | 0.091139   | 3.66x   | -72.65%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.333183 | 0.078074   | 4.27x   | -76.57%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.333183 | 0.082865   | 4.02x   | -75.13%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.439557 | 0.105209   | 4.18x   | -76.06%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.439557 | 0.082207   | 5.35x   | -81.30%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.439557 | 0.087287   | 5.04x   | -80.14%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.339362 | 0.112132   | 3.03x   | -66.96%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.339362 | 0.081416   | 4.17x   | -76.01%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.339362 | 0.084506   | 4.02x   | -75.10%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.358886 | 0.100723   | 3.56x   | -71.93%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.358886 | 0.078868   | 4.55x   | -78.02%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.358886 | 0.082750   | 4.34x   | -76.94%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.349078 | 0.107010   | 3.26x   | -69.34%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.349078 | 0.081710   | 4.27x   | -76.59%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.349078 | 0.084210   | 4.15x   | -75.88%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.348447 | 0.146555   | 2.38x   | -57.94%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.348447 | 0.077777   | 4.48x   | -77.68%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.348447 | 0.084530   | 4.12x   | -75.74%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.336426 | 0.110753   | 3.04x   | -67.08%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.336426 | 0.081248   | 4.14x   | -75.85%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.336426 | 0.085680   | 3.93x   | -74.53%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.336504 | 0.098941   | 3.40x   | -70.60%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.336504 | 0.079266   | 4.25x   | -76.44%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.336504 | 0.082838   | 4.06x   | -75.38%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.479129 | 0.172793   | 2.77x   | -63.94%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.479129 | 0.079674   | 6.01x   | -83.37%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.479129 | 0.087018   | 5.51x   | -81.84%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.437820 | 0.097955   | 4.47x   | -77.63%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.437820 | 0.078571   | 5.57x   | -82.05%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.437820 | 0.082869   | 5.28x   | -81.07%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.341185 | 0.099580   | 3.43x   | -70.81%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.341185 | 0.077665   | 4.39x   | -77.24%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.341185 | 0.082471   | 4.14x   | -75.83%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.350612 | 0.122427   | 2.86x   | -65.08%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.350612 | 0.080854   | 4.34x   | -76.94%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.350612 | 0.085130   | 4.12x   | -75.72%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.351348 | 0.121569   | 2.89x   | -65.40%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.351348 | 0.079250   | 4.43x   | -77.44%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.351348 | 0.082730   | 4.25x   | -76.45%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.352004 | 0.103381   | 3.40x   | -70.63%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.352004 | 0.079182   | 4.45x   | -77.51%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.352004 | 0.083419   | 4.22x   | -76.30%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.448342 | 0.164844   | 2.72x   | -63.23%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.448342 | 0.080270   | 5.59x   | -82.10%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.448342 | 0.083788   | 5.35x   | -81.31%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.244532 | 0.075753   | 3.23x   | -69.02%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.244532 | 0.010829   | 22.58x  | -95.57%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.244532 | 0.008942   | 27.35x  | -96.34%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.349931 | 0.100009   | 3.50x   | -71.42%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.349931 | 0.013364   | 26.18x  | -96.18%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.349931 | 0.011326   | 30.90x  | -96.76%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.256654 | 0.101499   | 2.53x   | -60.45%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.256654 | 0.012628   | 20.32x  | -95.08%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.256654 | 0.010748   | 23.88x  | -95.81%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.272091 | 0.092527   | 2.94x   | -65.99%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.272091 | 0.012626   | 21.55x  | -95.36%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.272091 | 0.010603   | 25.66x  | -96.10%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.264729 | 0.099704   | 2.66x   | -62.34%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.264729 | 0.014735   | 17.97x  | -94.43%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.264729 | 0.011209   | 23.62x  | -95.77%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.254292 | 0.116706   | 2.18x   | -54.11%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.254292 | 0.015444   | 16.47x  | -93.93%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.254292 | 0.010972   | 23.18x  | -95.69%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.246525 | 0.099002   | 2.49x   | -59.84%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.246525 | 0.012325   | 20.00x  | -95.00%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.246525 | 0.010690   | 23.06x  | -95.66%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.251964 | 0.094828   | 2.66x   | -62.36%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.251964 | 0.012819   | 19.66x  | -94.91%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.251964 | 0.010833   | 23.26x  | -95.70%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.405978 | 0.154055   | 2.64x   | -62.05%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.405978 | 0.014242   | 28.51x  | -96.49%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.405978 | 0.011152   | 36.40x  | -97.25%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.347868 | 0.095824   | 3.63x   | -72.45%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.347868 | 0.012612   | 27.58x  | -96.37%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.347868 | 0.010679   | 32.57x  | -96.93%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.256541 | 0.093489   | 2.74x   | -63.56%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.256541 | 0.015452   | 16.60x  | -93.98%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.256541 | 0.011336   | 22.63x  | -95.58%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.263274 | 0.111552   | 2.36x   | -57.63%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.263274 | 0.013018   | 20.22x  | -95.06%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.263274 | 0.010871   | 24.22x  | -95.87%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.265877 | 0.112356   | 2.37x   | -57.74%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.265877 | 0.013229   | 20.10x  | -95.02%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.265877 | 0.010776   | 24.67x  | -95.95%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.260761 | 0.092818   | 2.81x   | -64.41%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.260761 | 0.013360   | 19.52x  | -94.88%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.260761 | 0.010846   | 24.04x  | -95.84%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.360648 | 0.147121   | 2.45x   | -59.21%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.360648 | 0.013923   | 25.90x  | -96.14%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.360648 | 0.010863   | 33.20x  | -96.99%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.301701 | 0.028636   | 10.54x  | -90.51%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.301701 | 0.010349   | 29.15x  | -96.57%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.301701 | 0.011375   | 26.52x  | -96.23%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.413366 | 0.038043   | 10.87x  | -90.80%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.413366 | 0.017550   | 23.55x  | -95.75%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.413366 | 0.012694   | 32.56x  | -96.93%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.315277 | 0.042191   | 7.47x   | -86.62%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.315277 | 0.012884   | 24.47x  | -95.91%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.315277 | 0.012451   | 25.32x  | -96.05%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.328649 | 0.036871   | 8.91x   | -88.78%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.328649 | 0.014214   | 23.12x  | -95.68%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.328649 | 0.012998   | 25.29x  | -96.05%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.320413 | 0.038224   | 8.38x   | -88.07%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.320413 | 0.014605   | 21.94x  | -95.44%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.320413 | 0.013254   | 24.17x  | -95.86%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.324233 | 0.093090   | 3.48x   | -71.29%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.324233 | 0.015020   | 21.59x  | -95.37%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.324233 | 0.011788   | 27.51x  | -96.36%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.290007 | 0.040385   | 7.18x   | -86.07%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.290007 | 0.012669   | 22.89x  | -95.63%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.290007 | 0.012242   | 23.69x  | -95.78%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.310236 | 0.034050   | 9.11x   | -89.02%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.310236 | 0.012797   | 24.24x  | -95.88%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.310236 | 0.011431   | 27.14x  | -96.32%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.436340 | 0.109011   | 4.00x   | -75.02%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.436340 | 0.014814   | 29.45x  | -96.60%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.436340 | 0.012437   | 35.08x  | -97.15%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.402756 | 0.033686   | 11.96x  | -91.64%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.402756 | 0.013211   | 30.49x  | -96.72%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.402756 | 0.012309   | 32.72x  | -96.94%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.310678 | 0.041943   | 7.41x   | -86.50%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.310678 | 0.015500   | 20.04x  | -95.01%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.310678 | 0.012935   | 24.02x  | -95.84%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.356282 | 0.062326   | 5.72x   | -82.51%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.356282 | 0.013959   | 25.52x  | -96.08%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.356282 | 0.011950   | 29.82x  | -96.65%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.311896 | 0.060903   | 5.12x   | -80.47%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.311896 | 0.015185   | 20.54x  | -95.13%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.311896 | 0.012098   | 25.78x  | -96.12%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.325583 | 0.038808   | 8.39x   | -88.08%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.325583 | 0.015014   | 21.68x  | -95.39%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.325583 | 0.012980   | 25.08x  | -96.01%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.425546 | 0.103531   | 4.11x   | -75.67%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.425546 | 0.014823   | 28.71x  | -96.52%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.425546 | 0.013218   | 32.20x  | -96.89%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.263992 | 0.041820   | 6.31x   | -84.16%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.263992 | 0.023269   | 11.34x  | -91.19%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.263992 | 0.019625   | 13.45x  | -92.57%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.349918 | 0.053008   | 6.60x   | -84.85%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.349918 | 0.027775   | 12.60x  | -92.06%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.349918 | 0.022639   | 15.46x  | -93.53%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.242846 | 0.057958   | 4.19x   | -76.13%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.242846 | 0.027714   | 8.76x   | -88.59%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.242846 | 0.021379   | 11.36x  | -91.20%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.263685 | 0.052540   | 5.02x   | -80.07%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.263685 | 0.026287   | 10.03x  | -90.03%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.263685 | 0.023327   | 11.30x  | -91.15%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.271843 | 0.055815   | 4.87x   | -79.47%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.271843 | 0.026499   | 10.26x  | -90.25%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.271843 | 0.021862   | 12.43x  | -91.96%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.259717 | 0.088266   | 2.94x   | -66.01%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.259717 | 0.028652   | 9.06x   | -88.97%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.259717 | 0.023093   | 11.25x  | -91.11%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.240325 | 0.057180   | 4.20x   | -76.21%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.240325 | 0.026812   | 8.96x   | -88.84%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.240325 | 0.021414   | 11.22x  | -91.09%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.245974 | 0.047208   | 5.21x   | -80.81%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.245974 | 0.026912   | 9.14x   | -89.06%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.245974 | 0.021403   | 11.49x  | -91.30%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.390678 | 0.116988   | 3.34x   | -70.06%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.390678 | 0.026115   | 14.96x  | -93.32%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.390678 | 0.021462   | 18.20x  | -94.51%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.351330 | 0.049871   | 7.04x   | -85.81%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.351330 | 0.028064   | 12.52x  | -92.01%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.351330 | 0.025259   | 13.91x  | -92.81%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.265760 | 0.061064   | 4.35x   | -77.02%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.265760 | 0.027535   | 9.65x   | -89.64%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.265760 | 0.021922   | 12.12x  | -91.75%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.274974 | 0.071195   | 3.86x   | -74.11%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.274974 | 0.026644   | 10.32x  | -90.31%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.274974 | 0.021671   | 12.69x  | -92.12%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.271268 | 0.073048   | 3.71x   | -73.07%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.271268 | 0.026786   | 10.13x  | -90.13%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.271268 | 0.021832   | 12.43x  | -91.95%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.292625 | 0.056863   | 5.15x   | -80.57%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.292625 | 0.026927   | 10.87x  | -90.80%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.292625 | 0.022188   | 13.19x  | -92.42%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.363355 | 0.120889   | 3.01x   | -66.73%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.363355 | 0.026949   | 13.48x  | -92.58%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.363355 | 0.024182   | 15.03x  | -93.34%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.810524 | 0.211859   | 3.83x   | -73.86%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.810524 | 0.177108   | 4.58x   | -78.15%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.810524 | 0.198618   | 4.08x   | -75.50%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 0.977934 | 0.246600   | 3.97x   | -74.78%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 0.977934 | 0.185617   | 5.27x   | -81.02%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 0.977934 | 0.189562   | 5.16x   | -80.62%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.732703 | 0.245210   | 2.99x   | -66.53%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.732703 | 0.179559   | 4.08x   | -75.49%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.732703 | 0.184923   | 3.96x   | -74.76%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.772148 | 0.229198   | 3.37x   | -70.32%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.772148 | 0.192954   | 4.00x   | -75.01%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.772148 | 0.186764   | 4.13x   | -75.81%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.796234 | 0.245862   | 3.24x   | -69.12%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.796234 | 0.180435   | 4.41x   | -77.34%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.796234 | 0.187935   | 4.24x   | -76.40%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.751179 | 0.335505   | 2.24x   | -55.34%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.751179 | 0.176189   | 4.26x   | -76.55%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.751179 | 0.183978   | 4.08x   | -75.51%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.747998 | 0.251293   | 2.98x   | -66.40%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.747998 | 0.178269   | 4.20x   | -76.17%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.747998 | 0.187520   | 3.99x   | -74.93%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.747514 | 0.238968   | 3.13x   | -68.03%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.747514 | 0.182843   | 4.09x   | -75.54%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.747514 | 0.195190   | 3.83x   | -73.89%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.103315 | 0.392556   | 2.81x   | -64.42%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.103315 | 0.186109   | 5.93x   | -83.13%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.103315 | 0.188689   | 5.85x   | -82.90%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.986955 | 0.233516   | 4.23x   | -76.34%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.986955 | 0.194650   | 5.07x   | -80.28%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.986955 | 0.187575   | 5.26x   | -80.99%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.759649 | 0.230539   | 3.30x   | -69.65%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.759649 | 0.176436   | 4.31x   | -76.77%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.759649 | 0.188194   | 4.04x   | -75.23%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.771425 | 0.279990   | 2.76x   | -63.70%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.771425 | 0.184710   | 4.18x   | -76.06%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.771425 | 0.186736   | 4.13x   | -75.79%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.868237 | 0.289065   | 3.00x   | -66.71%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.868237 | 0.181051   | 4.80x   | -79.15%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.868237 | 0.191624   | 4.53x   | -77.93%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.792788 | 0.238517   | 3.32x   | -69.91%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.792788 | 0.179029   | 4.43x   | -77.42%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.792788 | 0.189381   | 4.19x   | -76.11%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 1.015153 | 0.376014   | 2.70x   | -62.96%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 1.015153 | 0.179932   | 5.64x   | -82.28%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 1.015153 | 0.191231   | 5.31x   | -81.16%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.554462 | 0.181603   | 3.05x   | -67.25%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.554462 | 0.025143   | 22.05x  | -95.47%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.554462 | 0.020359   | 27.23x  | -96.33%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.792810 | 0.223272   | 3.55x   | -71.84%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.792810 | 0.029477   | 26.90x  | -96.28%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.792810 | 0.024514   | 32.34x  | -96.91%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.541147 | 0.228128   | 2.37x   | -57.84%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.541147 | 0.027662   | 19.56x  | -94.89%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.541147 | 0.023802   | 22.73x  | -95.60%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.578389 | 0.211028   | 2.74x   | -63.51%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.578389 | 0.028241   | 20.48x  | -95.12%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.578389 | 0.023762   | 24.34x  | -95.89%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.580284 | 0.219092   | 2.65x   | -62.24%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.580284 | 0.031030   | 18.70x  | -94.65%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.580284 | 0.024439   | 23.74x  | -95.79%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.554270 | 0.267695   | 2.07x   | -51.70%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.554270 | 0.034458   | 16.09x  | -93.78%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.554270 | 0.025052   | 22.12x  | -95.48%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.544786 | 0.227410   | 2.40x   | -58.26%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.544786 | 0.027577   | 19.75x  | -94.94%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.544786 | 0.023722   | 22.97x  | -95.65%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.560730 | 0.219711   | 2.55x   | -60.82%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.560730 | 0.029887   | 18.76x  | -94.67%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.560730 | 0.023668   | 23.69x  | -95.78%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.891432 | 0.355592   | 2.51x   | -60.11%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.891432 | 0.033585   | 26.54x  | -96.23%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.891432 | 0.024884   | 35.82x  | -97.21%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.789762 | 0.219623   | 3.60x   | -72.19%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.789762 | 0.028833   | 27.39x  | -96.35%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.789762 | 0.024398   | 32.37x  | -96.91%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.590876 | 0.224541   | 2.63x   | -62.00%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.590876 | 0.035199   | 16.79x  | -94.04%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.590876 | 0.025676   | 23.01x  | -95.65%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.578481 | 0.255979   | 2.26x   | -55.75%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.578481 | 0.028659   | 20.19x  | -95.05%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.578481 | 0.027569   | 20.98x  | -95.23%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.582410 | 0.263751   | 2.21x   | -54.71%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.582410 | 0.028594   | 20.37x  | -95.09%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.582410 | 0.024015   | 24.25x  | -95.88%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.602323 | 0.220464   | 2.73x   | -63.40%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.602323 | 0.029640   | 20.32x  | -95.08%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.602323 | 0.025130   | 23.97x  | -95.83%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.810268 | 0.335673   | 2.41x   | -58.57%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.810268 | 0.031103   | 26.05x  | -96.16%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.810268 | 0.024627   | 32.90x  | -96.96%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.652738 | 0.074521   | 8.76x   | -88.58%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.652738 | 0.032745   | 19.93x  | -94.98%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.652738 | 0.035122   | 18.58x  | -94.62%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.857086 | 0.095533   | 8.97x   | -88.85%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.857086 | 0.040975   | 20.92x  | -95.22%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.857086 | 0.037014   | 23.16x  | -95.68%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.636909 | 0.104662   | 6.09x   | -83.57%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.636909 | 0.039448   | 16.15x  | -93.81%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.636909 | 0.037305   | 17.07x  | -94.14%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.675464 | 0.090485   | 7.46x   | -86.60%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.675464 | 0.041524   | 16.27x  | -93.85%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.675464 | 0.037990   | 17.78x  | -94.38%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.683533 | 0.097518   | 7.01x   | -85.73%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.683533 | 0.043037   | 15.88x  | -93.70%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.683533 | 0.038007   | 17.98x  | -94.44%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.666286 | 0.215093   | 3.10x   | -67.72%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.666286 | 0.044180   | 15.08x  | -93.37%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.666286 | 0.037457   | 17.79x  | -94.38%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.630644 | 0.101749   | 6.20x   | -83.87%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.630644 | 0.038764   | 16.27x  | -93.85%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.630644 | 0.036504   | 17.28x  | -94.21%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.652586 | 0.082905   | 7.87x   | -87.30%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.652586 | 0.037765   | 17.28x  | -94.21%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.652586 | 0.034646   | 18.84x  | -94.69%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.986865 | 0.256221   | 3.85x   | -74.04%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.986865 | 0.043139   | 22.88x  | -95.63%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.986865 | 0.037996   | 25.97x  | -96.15%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.870507 | 0.085896   | 10.13x  | -90.13%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.870507 | 0.040641   | 21.42x  | -95.33%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.870507 | 0.036016   | 24.17x  | -95.86%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.658518 | 0.106372   | 6.19x   | -83.85%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.658518 | 0.044576   | 14.77x  | -93.23%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.658518 | 0.036498   | 18.04x  | -94.46%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.675374 | 0.144825   | 4.66x   | -78.56%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.675374 | 0.041878   | 16.13x  | -93.80%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.675374 | 0.037779   | 17.88x  | -94.41%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.669940 | 0.145587   | 4.60x   | -78.27%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.669940 | 0.042944   | 15.60x  | -93.59%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.669940 | 0.037176   | 18.02x  | -94.45%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.686159 | 0.091970   | 7.46x   | -86.60%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.686159 | 0.041745   | 16.44x  | -93.92%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.686159 | 0.041492   | 16.54x  | -93.95%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.911117 | 0.240236   | 3.79x   | -73.63%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.911117 | 0.042590   | 21.39x  | -95.33%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.911117 | 0.036722   | 24.81x  | -95.97%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.507475 | 0.091761   | 5.53x   | -81.92%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.507475 | 0.051142   | 9.92x   | -89.92%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.507475 | 0.041856   | 12.12x  | -91.75%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.730543 | 0.117022   | 6.24x   | -83.98%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.730543 | 0.058378   | 12.51x  | -92.01%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.730543 | 0.048081   | 15.19x  | -93.42%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.497790 | 0.123713   | 4.02x   | -75.15%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.497790 | 0.058247   | 8.55x   | -88.30%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.497790 | 0.045964   | 10.83x  | -90.77%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.531283 | 0.110400   | 4.81x   | -79.22%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.531283 | 0.057702   | 9.21x   | -89.14%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.531283 | 0.046724   | 11.37x  | -91.21%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.533133 | 0.115593   | 4.61x   | -78.32%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.533133 | 0.057827   | 9.22x   | -89.15%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.533133 | 0.046318   | 11.51x  | -91.31%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.506684 | 0.190794   | 2.66x   | -62.34%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.506684 | 0.060867   | 8.32x   | -87.99%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.506684 | 0.049132   | 10.31x  | -90.30%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.493217 | 0.130347   | 3.78x   | -73.57%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.493217 | 0.058691   | 8.40x   | -88.10%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.493217 | 0.047947   | 10.29x  | -90.28%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.524997 | 0.107146   | 4.90x   | -79.59%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.524997 | 0.061568   | 8.53x   | -88.27%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.524997 | 0.048858   | 10.75x  | -90.69%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.855964 | 0.261376   | 3.27x   | -69.46%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.855964 | 0.057815   | 14.81x  | -93.25%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.855964 | 0.049443   | 17.31x  | -94.22%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.748053 | 0.115309   | 6.49x   | -84.59%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.748053 | 0.061792   | 12.11x  | -91.74%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.748053 | 0.050607   | 14.78x  | -93.23%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.516720 | 0.137208   | 3.77x   | -73.45%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.516720 | 0.062694   | 8.24x   | -87.87%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.516720 | 0.048151   | 10.73x  | -90.68%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.537751 | 0.158611   | 3.39x   | -70.50%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.537751 | 0.057869   | 9.29x   | -89.24%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.537751 | 0.047071   | 11.42x  | -91.25%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.535175 | 0.156793   | 3.41x   | -70.70%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.535175 | 0.057720   | 9.27x   | -89.21%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.535175 | 0.045594   | 11.74x  | -91.48%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.551323 | 0.114753   | 4.80x   | -79.19%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.551323 | 0.059171   | 9.32x   | -89.27%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.551323 | 0.049518   | 11.13x  | -91.02%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.760148 | 0.245560   | 3.10x   | -67.70%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.760148 | 0.055991   | 13.58x  | -92.63%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.760148 | 0.047482   | 16.01x  | -93.75%        |
</details>
<!-- PERF_RESULTS_END -->
