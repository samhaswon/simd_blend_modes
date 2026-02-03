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
| normal        | scalar | 0.192243 | 0.042739   | 4.50x   | -77.77%        |
| normal        | sse42  | 0.192243 | 0.021129   | 9.10x   | -89.01%        |
| normal        | avx2   | 0.192243 | 0.020708   | 9.28x   | -89.23%        |
| soft_light    | scalar | 0.261741 | 0.052995   | 4.94x   | -79.75%        |
| soft_light    | sse42  | 0.261741 | 0.022875   | 11.44x  | -91.26%        |
| soft_light    | avx2   | 0.261741 | 0.021854   | 11.98x  | -91.65%        |
| lighten_only  | scalar | 0.191711 | 0.060322   | 3.18x   | -68.54%        |
| lighten_only  | sse42  | 0.191711 | 0.021988   | 8.72x   | -88.53%        |
| lighten_only  | avx2   | 0.191711 | 0.021439   | 8.94x   | -88.82%        |
| screen        | scalar | 0.204561 | 0.051933   | 3.94x   | -74.61%        |
| screen        | sse42  | 0.204561 | 0.022339   | 9.16x   | -89.08%        |
| screen        | avx2   | 0.204561 | 0.021674   | 9.44x   | -89.40%        |
| dodge         | scalar | 0.207655 | 0.054406   | 3.82x   | -73.80%        |
| dodge         | sse42  | 0.207655 | 0.023557   | 8.81x   | -88.66%        |
| dodge         | avx2   | 0.207655 | 0.022258   | 9.33x   | -89.28%        |
| addition      | scalar | 0.198924 | 0.079281   | 2.51x   | -60.14%        |
| addition      | sse42  | 0.198924 | 0.023308   | 8.53x   | -88.28%        |
| addition      | avx2   | 0.198924 | 0.021902   | 9.08x   | -88.99%        |
| darken_only   | scalar | 0.194581 | 0.060552   | 3.21x   | -68.88%        |
| darken_only   | sse42  | 0.194581 | 0.021845   | 8.91x   | -88.77%        |
| darken_only   | avx2   | 0.194581 | 0.021407   | 9.09x   | -89.00%        |
| multiply      | scalar | 0.197827 | 0.052015   | 3.80x   | -73.71%        |
| multiply      | sse42  | 0.197827 | 0.021864   | 9.05x   | -88.95%        |
| multiply      | avx2   | 0.197827 | 0.021629   | 9.15x   | -89.07%        |
| hard_light    | scalar | 0.292409 | 0.095780   | 3.05x   | -67.24%        |
| hard_light    | sse42  | 0.292409 | 0.023224   | 12.59x  | -92.06%        |
| hard_light    | avx2   | 0.292409 | 0.021757   | 13.44x  | -92.56%        |
| difference    | scalar | 0.264410 | 0.051512   | 5.13x   | -80.52%        |
| difference    | sse42  | 0.264410 | 0.022023   | 12.01x  | -91.67%        |
| difference    | avx2   | 0.264410 | 0.021347   | 12.39x  | -91.93%        |
| subtract      | scalar | 0.197403 | 0.053788   | 3.67x   | -72.75%        |
| subtract      | sse42  | 0.197403 | 0.023078   | 8.55x   | -88.31%        |
| subtract      | avx2   | 0.197403 | 0.021952   | 8.99x   | -88.88%        |
| grain_extract | scalar | 0.203771 | 0.065976   | 3.09x   | -67.62%        |
| grain_extract | sse42  | 0.203771 | 0.022499   | 9.06x   | -88.96%        |
| grain_extract | avx2   | 0.203771 | 0.021948   | 9.28x   | -89.23%        |
| grain_merge   | scalar | 0.205503 | 0.065955   | 3.12x   | -67.91%        |
| grain_merge   | sse42  | 0.205503 | 0.022592   | 9.10x   | -89.01%        |
| grain_merge   | avx2   | 0.205503 | 0.021681   | 9.48x   | -89.45%        |
| divide        | scalar | 0.205822 | 0.053855   | 3.82x   | -73.83%        |
| divide        | sse42  | 0.205822 | 0.022871   | 9.00x   | -88.89%        |
| divide        | avx2   | 0.205822 | 0.021766   | 9.46x   | -89.42%        |
| overlay       | scalar | 0.272738 | 0.096961   | 2.81x   | -64.45%        |
| overlay       | sse42  | 0.272738 | 0.023370   | 11.67x  | -91.43%        |
| overlay       | avx2   | 0.272738 | 0.022250   | 12.26x  | -91.84%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.007108 | 0.001688   | 4.21x   | -76.25%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.007108 | 0.001553   | 4.58x   | -78.15%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.007108 | 0.001583   | 4.49x   | -77.73%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.009376 | 0.002292   | 4.09x   | -75.56%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.009376 | 0.001571   | 5.97x   | -83.24%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.009376 | 0.001646   | 5.70x   | -82.45%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.007921 | 0.002284   | 3.47x   | -71.17%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.007921 | 0.001618   | 4.90x   | -79.58%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.007921 | 0.001572   | 5.04x   | -80.15%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.007962 | 0.001934   | 4.12x   | -75.71%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.007962 | 0.001498   | 5.31x   | -81.18%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.007962 | 0.001494   | 5.33x   | -81.23%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.007498 | 0.002018   | 3.72x   | -73.08%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.007498 | 0.001571   | 4.77x   | -79.05%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.007498 | 0.001615   | 4.64x   | -78.46%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.007312 | 0.002702   | 2.71x   | -63.05%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.007312 | 0.001487   | 4.92x   | -79.66%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.007312 | 0.001631   | 4.48x   | -77.70%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.007769 | 0.002214   | 3.51x   | -71.50%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.007769 | 0.001576   | 4.93x   | -79.72%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.007769 | 0.001555   | 5.00x   | -79.98%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.007448 | 0.001982   | 3.76x   | -73.39%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.007448 | 0.001522   | 4.89x   | -79.56%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.007448 | 0.001593   | 4.68x   | -78.61%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.009009 | 0.003205   | 2.81x   | -64.43%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.009009 | 0.001794   | 5.02x   | -80.09%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.009009 | 0.001821   | 4.95x   | -79.78%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.009156 | 0.001866   | 4.91x   | -79.62%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.009156 | 0.001459   | 6.28x   | -84.07%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.009156 | 0.001502   | 6.10x   | -83.60%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.006990 | 0.001797   | 3.89x   | -74.29%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.006990 | 0.001486   | 4.70x   | -78.74%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.006990 | 0.001575   | 4.44x   | -77.47%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.007230 | 0.002268   | 3.19x   | -68.64%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.007230 | 0.001502   | 4.81x   | -79.23%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.007230 | 0.001503   | 4.81x   | -79.21%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.007345 | 0.002268   | 3.24x   | -69.12%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.007345 | 0.001453   | 5.05x   | -80.21%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.007345 | 0.001518   | 4.84x   | -79.33%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.007311 | 0.001955   | 3.74x   | -73.26%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.007311 | 0.001492   | 4.90x   | -79.59%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.007311 | 0.001489   | 4.91x   | -79.64%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.008463 | 0.003076   | 2.75x   | -63.65%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.008463 | 0.001487   | 5.69x   | -82.43%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.008463 | 0.001536   | 5.51x   | -81.86%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003152 | 0.001409   | 2.24x   | -55.31%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003152 | 0.000186   | 16.98x  | -94.11%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003152 | 0.000169   | 18.61x  | -94.63%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.006782 | 0.001820   | 3.73x   | -73.16%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.006782 | 0.000216   | 31.45x  | -96.82%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.006782 | 0.000189   | 35.96x  | -97.22%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.005349 | 0.002097   | 2.55x   | -60.80%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.005349 | 0.000199   | 26.90x  | -96.28%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.005349 | 0.000184   | 29.04x  | -96.56%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.005630 | 0.001705   | 3.30x   | -69.72%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.005630 | 0.000200   | 28.20x  | -96.45%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.005630 | 0.000185   | 30.41x  | -96.71%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.005688 | 0.001835   | 3.10x   | -67.74%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.005688 | 0.000238   | 23.85x  | -95.81%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.005688 | 0.000190   | 29.99x  | -96.67%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.005983 | 0.002078   | 2.88x   | -65.27%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.005983 | 0.000260   | 22.99x  | -95.65%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.005983 | 0.000193   | 31.06x  | -96.78%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.005353 | 0.001976   | 2.71x   | -63.08%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.005353 | 0.000199   | 26.91x  | -96.28%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.005353 | 0.000186   | 28.72x  | -96.52%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.005582 | 0.001755   | 3.18x   | -68.57%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.005582 | 0.000201   | 27.80x  | -96.40%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.005582 | 0.000189   | 29.49x  | -96.61%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.007508 | 0.002778   | 2.70x   | -63.00%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.007508 | 0.000237   | 31.69x  | -96.84%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.007508 | 0.000188   | 39.97x  | -97.50%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.007302 | 0.001681   | 4.34x   | -76.97%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.007302 | 0.000199   | 36.63x  | -97.27%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.007302 | 0.000186   | 39.17x  | -97.45%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.005806 | 0.001692   | 3.43x   | -70.86%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.005806 | 0.000262   | 22.19x  | -95.49%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.005806 | 0.000194   | 29.95x  | -96.66%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.005391 | 0.001991   | 2.71x   | -63.07%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.005391 | 0.000207   | 26.09x  | -96.17%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.005391 | 0.000190   | 28.42x  | -96.48%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.005578 | 0.002008   | 2.78x   | -64.00%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.005578 | 0.000210   | 26.52x  | -96.23%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.005578 | 0.000187   | 29.88x  | -96.65%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.005533 | 0.001799   | 3.08x   | -67.48%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.005533 | 0.000209   | 26.42x  | -96.22%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.005533 | 0.000185   | 29.85x  | -96.65%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.006833 | 0.002788   | 2.45x   | -59.20%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.006833 | 0.000227   | 30.07x  | -96.67%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.006833 | 0.000186   | 36.68x  | -97.27%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.005439 | 0.000510   | 10.67x  | -90.63%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.005439 | 0.000188   | 28.88x  | -96.54%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.005439 | 0.000143   | 37.91x  | -97.36%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.008393 | 0.000668   | 12.57x  | -92.04%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.008393 | 0.000244   | 34.41x  | -97.09%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.008393 | 0.000194   | 43.22x  | -97.69%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.006995 | 0.000865   | 8.09x   | -87.64%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.006995 | 0.000212   | 33.02x  | -96.97%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.006995 | 0.000187   | 37.37x  | -97.32%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.007115 | 0.000673   | 10.57x  | -90.54%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.007115 | 0.000228   | 31.21x  | -96.80%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.007115 | 0.000190   | 37.46x  | -97.33%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.007323 | 0.000724   | 10.12x  | -90.12%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.007323 | 0.000258   | 28.39x  | -96.48%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.007323 | 0.000193   | 37.87x  | -97.36%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.007148 | 0.001662   | 4.30x   | -76.75%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.007148 | 0.000236   | 30.25x  | -96.69%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.007148 | 0.000187   | 38.20x  | -97.38%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.006772 | 0.000855   | 7.92x   | -87.37%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.006772 | 0.000224   | 30.28x  | -96.70%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.006772 | 0.000180   | 37.60x  | -97.34%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.006901 | 0.000658   | 10.50x  | -90.47%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.006901 | 0.000212   | 32.51x  | -96.92%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.006901 | 0.000179   | 38.45x  | -97.40%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.008630 | 0.001761   | 4.90x   | -79.60%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.008630 | 0.000249   | 34.66x  | -97.12%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.008630 | 0.000186   | 46.30x  | -97.84%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.008646 | 0.000670   | 12.90x  | -92.25%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.008646 | 0.000227   | 38.14x  | -97.38%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.008646 | 0.000184   | 46.91x  | -97.87%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.007297 | 0.000742   | 9.84x   | -89.83%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.007297 | 0.000242   | 30.21x  | -96.69%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.007297 | 0.000190   | 38.42x  | -97.40%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.007213 | 0.001060   | 6.81x   | -85.31%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.007213 | 0.000241   | 29.94x  | -96.66%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.007213 | 0.000182   | 39.55x  | -97.47%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.007263 | 0.001065   | 6.82x   | -85.34%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.007263 | 0.000233   | 31.12x  | -96.79%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.007263 | 0.000184   | 39.53x  | -97.47%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.007165 | 0.000714   | 10.04x  | -90.04%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.007165 | 0.000241   | 29.69x  | -96.63%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.007165 | 0.000189   | 37.99x  | -97.37%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.008686 | 0.001848   | 4.70x   | -78.73%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.008686 | 0.000298   | 29.20x  | -96.57%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.008686 | 0.000227   | 38.26x  | -97.39%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.004368 | 0.000620   | 7.05x   | -85.81%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.004368 | 0.000189   | 23.09x  | -95.67%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.004368 | 0.000160   | 27.36x  | -96.34%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.006872 | 0.000851   | 8.08x   | -87.62%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.006872 | 0.000244   | 28.18x  | -96.45%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.006872 | 0.000192   | 35.82x  | -97.21%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.005506 | 0.001064   | 5.17x   | -80.67%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.005506 | 0.000210   | 26.25x  | -96.19%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.005506 | 0.000202   | 27.24x  | -96.33%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.005532 | 0.000868   | 6.37x   | -84.30%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.005532 | 0.000238   | 23.27x  | -95.70%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.005532 | 0.000184   | 30.05x  | -96.67%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.005535 | 0.000900   | 6.15x   | -83.74%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.005535 | 0.000241   | 22.96x  | -95.64%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.005535 | 0.000192   | 28.81x  | -96.53%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.005783 | 0.001481   | 3.91x   | -74.39%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.005783 | 0.000224   | 25.86x  | -96.13%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.005783 | 0.000195   | 29.66x  | -96.63%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.005332 | 0.000970   | 5.49x   | -81.80%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.005332 | 0.000211   | 25.26x  | -96.04%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.005332 | 0.000185   | 28.80x  | -96.53%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.005388 | 0.000812   | 6.64x   | -84.93%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.005388 | 0.000214   | 25.15x  | -96.02%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.005388 | 0.000201   | 26.74x  | -96.26%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.007311 | 0.001876   | 3.90x   | -74.34%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.007311 | 0.000244   | 29.96x  | -96.66%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.007311 | 0.000189   | 38.73x  | -97.42%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.007212 | 0.000803   | 8.98x   | -88.87%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.007212 | 0.000214   | 33.70x  | -97.03%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.007212 | 0.000187   | 38.61x  | -97.41%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.005662 | 0.000972   | 5.83x   | -82.84%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.005662 | 0.000240   | 23.57x  | -95.76%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.005662 | 0.000196   | 28.85x  | -96.53%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.005438 | 0.001171   | 4.64x   | -78.46%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.005438 | 0.000248   | 21.90x  | -95.43%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.005438 | 0.000188   | 28.96x  | -96.55%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.005585 | 0.001166   | 4.79x   | -79.12%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.005585 | 0.000235   | 23.76x  | -95.79%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.005585 | 0.000187   | 29.84x  | -96.65%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.005667 | 0.000859   | 6.60x   | -84.84%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.005667 | 0.000237   | 23.88x  | -95.81%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.005667 | 0.000191   | 29.69x  | -96.63%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.007093 | 0.001944   | 3.65x   | -72.59%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.007093 | 0.000236   | 30.07x  | -96.67%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.007093 | 0.000194   | 36.49x  | -97.26%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.033359 | 0.006297   | 5.30x   | -81.12%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.033359 | 0.005638   | 5.92x   | -83.10%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.033359 | 0.005889   | 5.66x   | -82.35%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.043766 | 0.007830   | 5.59x   | -82.11%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.043766 | 0.006053   | 7.23x   | -86.17%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.043766 | 0.006145   | 7.12x   | -85.96%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.037839 | 0.008159   | 4.64x   | -78.44%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.037839 | 0.005723   | 6.61x   | -84.87%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.037839 | 0.005925   | 6.39x   | -84.34%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.038186 | 0.007425   | 5.14x   | -80.55%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.038186 | 0.005837   | 6.54x   | -84.71%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.038186 | 0.005923   | 6.45x   | -84.49%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.038749 | 0.008419   | 4.60x   | -78.27%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.038749 | 0.006376   | 6.08x   | -83.55%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.038749 | 0.006413   | 6.04x   | -83.45%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.037859 | 0.010509   | 3.60x   | -72.24%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.037859 | 0.006014   | 6.30x   | -84.12%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.037859 | 0.006077   | 6.23x   | -83.95%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.037748 | 0.008896   | 4.24x   | -76.43%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.037748 | 0.005770   | 6.54x   | -84.72%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.037748 | 0.006020   | 6.27x   | -84.05%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.038618 | 0.007539   | 5.12x   | -80.48%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.038618 | 0.005806   | 6.65x   | -84.96%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.038618 | 0.005931   | 6.51x   | -84.64%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.046010 | 0.012373   | 3.72x   | -73.11%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.046010 | 0.005952   | 7.73x   | -87.06%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.046010 | 0.006052   | 7.60x   | -86.85%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.043978 | 0.007391   | 5.95x   | -83.19%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.043978 | 0.005726   | 7.68x   | -86.98%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.043978 | 0.005899   | 7.45x   | -86.59%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.036430 | 0.007356   | 4.95x   | -79.81%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.036430 | 0.006014   | 6.06x   | -83.49%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.036430 | 0.007127   | 5.11x   | -80.44%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.039069 | 0.009180   | 4.26x   | -76.50%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.039069 | 0.006024   | 6.49x   | -84.58%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.039069 | 0.006143   | 6.36x   | -84.28%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.037754 | 0.009042   | 4.18x   | -76.05%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.037754 | 0.005836   | 6.47x   | -84.54%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.037754 | 0.005928   | 6.37x   | -84.30%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.038241 | 0.007612   | 5.02x   | -80.09%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.038241 | 0.005944   | 6.43x   | -84.46%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.038241 | 0.006005   | 6.37x   | -84.30%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.044993 | 0.012341   | 3.65x   | -72.57%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.044993 | 0.005919   | 7.60x   | -86.85%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.044993 | 0.006035   | 7.46x   | -86.59%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.023709 | 0.005681   | 4.17x   | -76.04%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.023709 | 0.000731   | 32.42x  | -96.92%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.023709 | 0.000652   | 36.36x  | -97.25%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.035264 | 0.006966   | 5.06x   | -80.25%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.035264 | 0.000862   | 40.90x  | -97.56%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.035264 | 0.000735   | 48.00x  | -97.92%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.028031 | 0.007902   | 3.55x   | -71.81%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.028031 | 0.000778   | 36.01x  | -97.22%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.028031 | 0.000731   | 38.35x  | -97.39%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.028746 | 0.006771   | 4.25x   | -76.44%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.028746 | 0.000798   | 36.03x  | -97.22%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.028746 | 0.000733   | 39.20x  | -97.45%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.028662 | 0.006949   | 4.12x   | -75.76%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.028662 | 0.000945   | 30.33x  | -96.70%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.028662 | 0.000737   | 38.88x  | -97.43%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.029055 | 0.008968   | 3.24x   | -69.14%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.029055 | 0.001065   | 27.28x  | -96.33%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.029055 | 0.000770   | 37.72x  | -97.35%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.028850 | 0.008041   | 3.59x   | -72.13%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.028850 | 0.000778   | 37.09x  | -97.30%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.028850 | 0.000746   | 38.67x  | -97.41%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.028718 | 0.006883   | 4.17x   | -76.03%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.028718 | 0.000783   | 36.67x  | -97.27%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.028718 | 0.000735   | 39.06x  | -97.44%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.036795 | 0.011369   | 3.24x   | -69.10%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.036795 | 0.000967   | 38.07x  | -97.37%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.036795 | 0.000749   | 49.11x  | -97.96%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.036552 | 0.007203   | 5.07x   | -80.29%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.036552 | 0.000833   | 43.90x  | -97.72%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.036552 | 0.000777   | 47.07x  | -97.88%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.029491 | 0.006928   | 4.26x   | -76.51%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.029491 | 0.001048   | 28.15x  | -96.45%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.029491 | 0.000792   | 37.25x  | -97.32%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.030051 | 0.008107   | 3.71x   | -73.02%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.030051 | 0.000829   | 36.25x  | -97.24%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.030051 | 0.000756   | 39.78x  | -97.49%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.030974 | 0.008555   | 3.62x   | -72.38%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.030974 | 0.000885   | 35.01x  | -97.14%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.030974 | 0.000766   | 40.45x  | -97.53%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.031236 | 0.007194   | 4.34x   | -76.97%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.031236 | 0.000848   | 36.82x  | -97.28%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.031236 | 0.000747   | 41.82x  | -97.61%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.037672 | 0.011072   | 3.40x   | -70.61%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.037672 | 0.000893   | 42.21x  | -97.63%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.037672 | 0.000754   | 49.96x  | -98.00%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.028845 | 0.002511   | 11.49x  | -91.30%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.028845 | 0.000867   | 33.29x  | -97.00%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.028845 | 0.000676   | 42.66x  | -97.66%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.043608 | 0.003237   | 13.47x  | -92.58%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.043608 | 0.001007   | 43.30x  | -97.69%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.043608 | 0.000880   | 49.54x  | -97.98%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.036999 | 0.005018   | 7.37x   | -86.44%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.036999 | 0.000922   | 40.15x  | -97.51%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.036999 | 0.000842   | 43.97x  | -97.73%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.035808 | 0.004050   | 8.84x   | -88.69%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.035808 | 0.001036   | 34.55x  | -97.11%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.035808 | 0.000989   | 36.22x  | -97.24%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.038853 | 0.003732   | 10.41x  | -90.39%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.038853 | 0.001441   | 26.96x  | -96.29%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.038853 | 0.001085   | 35.81x  | -97.21%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.037143 | 0.007355   | 5.05x   | -80.20%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.037143 | 0.001043   | 35.61x  | -97.19%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.037143 | 0.000896   | 41.46x  | -97.59%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.034360 | 0.003949   | 8.70x   | -88.51%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.034360 | 0.000914   | 37.61x  | -97.34%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.034360 | 0.000818   | 42.03x  | -97.62%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.034551 | 0.003251   | 10.63x  | -90.59%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.034551 | 0.000947   | 36.49x  | -97.26%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.034551 | 0.000777   | 44.46x  | -97.75%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.043419 | 0.007932   | 5.47x   | -81.73%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.043419 | 0.001021   | 42.52x  | -97.65%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.043419 | 0.000779   | 55.75x  | -98.21%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.039742 | 0.003148   | 12.63x  | -92.08%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.039742 | 0.000931   | 42.71x  | -97.66%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.039742 | 0.000784   | 50.70x  | -98.03%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.033176 | 0.003405   | 9.74x   | -89.74%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.033176 | 0.000960   | 34.55x  | -97.11%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.033176 | 0.000760   | 43.65x  | -97.71%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.032857 | 0.004708   | 6.98x   | -85.67%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.032857 | 0.000933   | 35.21x  | -97.16%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.032857 | 0.000769   | 42.74x  | -97.66%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.033937 | 0.004726   | 7.18x   | -86.07%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.033937 | 0.000929   | 36.53x  | -97.26%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.033937 | 0.000762   | 44.53x  | -97.75%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.032785 | 0.003292   | 9.96x   | -89.96%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.032785 | 0.000956   | 34.30x  | -97.08%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.032785 | 0.000748   | 43.81x  | -97.72%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.039154 | 0.007907   | 4.95x   | -79.81%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.039154 | 0.000949   | 41.27x  | -97.58%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.039154 | 0.000751   | 52.11x  | -98.08%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.021002 | 0.002508   | 8.38x   | -88.06%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.021002 | 0.000889   | 23.63x  | -95.77%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.021002 | 0.000836   | 25.12x  | -96.02%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.031996 | 0.003562   | 8.98x   | -88.87%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.031996 | 0.001046   | 30.59x  | -96.73%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.031996 | 0.000939   | 34.08x  | -97.07%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.025013 | 0.004172   | 6.00x   | -83.32%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.025013 | 0.000866   | 28.88x  | -96.54%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.025013 | 0.000817   | 30.61x  | -96.73%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.025354 | 0.003518   | 7.21x   | -86.12%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.025354 | 0.001018   | 24.92x  | -95.99%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.025354 | 0.000876   | 28.93x  | -96.54%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.025452 | 0.003571   | 7.13x   | -85.97%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.025452 | 0.001003   | 25.38x  | -96.06%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.025452 | 0.000865   | 29.43x  | -96.60%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.024028 | 0.005846   | 4.11x   | -75.67%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.024028 | 0.001040   | 23.10x  | -95.67%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.024028 | 0.000886   | 27.11x  | -96.31%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.024331 | 0.003942   | 6.17x   | -83.80%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.024331 | 0.000876   | 27.78x  | -96.40%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.024331 | 0.000838   | 29.02x  | -96.55%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.024355 | 0.003304   | 7.37x   | -86.43%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.024355 | 0.000895   | 27.22x  | -96.33%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.024355 | 0.000830   | 29.35x  | -96.59%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.034850 | 0.007757   | 4.49x   | -77.74%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.034850 | 0.001037   | 33.60x  | -97.02%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.034850 | 0.001014   | 34.35x  | -97.09%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.033633 | 0.003621   | 9.29x   | -89.23%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.033633 | 0.000905   | 37.18x  | -97.31%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.033633 | 0.000877   | 38.35x  | -97.39%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.027477 | 0.004282   | 6.42x   | -84.42%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.027477 | 0.001161   | 23.68x  | -95.78%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.027477 | 0.000888   | 30.96x  | -96.77%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.025432 | 0.004693   | 5.42x   | -81.55%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.025432 | 0.001003   | 25.36x  | -96.06%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.025432 | 0.000920   | 27.63x  | -96.38%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.025776 | 0.004624   | 5.57x   | -82.06%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.025776 | 0.000956   | 26.98x  | -96.29%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.025776 | 0.000792   | 32.53x  | -96.93%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.024802 | 0.003406   | 7.28x   | -86.27%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.024802 | 0.000983   | 25.23x  | -96.04%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.024802 | 0.000830   | 29.87x  | -96.65%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.035307 | 0.008182   | 4.32x   | -76.83%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.035307 | 0.001065   | 33.16x  | -96.98%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.035307 | 0.000949   | 37.22x  | -97.31%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.108019 | 0.025506   | 4.24x   | -76.39%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.108019 | 0.022560   | 4.79x   | -79.11%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.108019 | 0.023475   | 4.60x   | -78.27%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.131072 | 0.031751   | 4.13x   | -75.78%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.131072 | 0.023776   | 5.51x   | -81.86%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.131072 | 0.024584   | 5.33x   | -81.24%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.103103 | 0.033576   | 3.07x   | -67.43%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.103103 | 0.023093   | 4.46x   | -77.60%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.103103 | 0.023472   | 4.39x   | -77.23%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.106398 | 0.029632   | 3.59x   | -72.15%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.106398 | 0.023340   | 4.56x   | -78.06%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.106398 | 0.023791   | 4.47x   | -77.64%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.106627 | 0.031018   | 3.44x   | -70.91%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.106627 | 0.023880   | 4.47x   | -77.60%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.106627 | 0.024855   | 4.29x   | -76.69%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.100988 | 0.041791   | 2.42x   | -58.62%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.100988 | 0.023141   | 4.36x   | -77.08%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.100988 | 0.023508   | 4.30x   | -76.72%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.105488 | 0.033252   | 3.17x   | -68.48%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.105488 | 0.023468   | 4.49x   | -77.75%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.105488 | 0.023615   | 4.47x   | -77.61%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.104774 | 0.030164   | 3.47x   | -71.21%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.104774 | 0.023839   | 4.39x   | -77.25%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.104774 | 0.025485   | 4.11x   | -75.68%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.140648 | 0.051220   | 2.75x   | -63.58%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.140648 | 0.023957   | 5.87x   | -82.97%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.140648 | 0.024167   | 5.82x   | -82.82%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.134118 | 0.030126   | 4.45x   | -77.54%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.134118 | 0.022996   | 5.83x   | -82.85%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.134118 | 0.023520   | 5.70x   | -82.46%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.103366 | 0.029720   | 3.48x   | -71.25%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.103366 | 0.023387   | 4.42x   | -77.37%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.103366 | 0.023786   | 4.35x   | -76.99%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.105542 | 0.036123   | 2.92x   | -65.77%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.105542 | 0.023663   | 4.46x   | -77.58%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.105542 | 0.024278   | 4.35x   | -77.00%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.105069 | 0.035724   | 2.94x   | -66.00%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.105069 | 0.023316   | 4.51x   | -77.81%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.105069 | 0.023510   | 4.47x   | -77.62%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.105508 | 0.030274   | 3.49x   | -71.31%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.105508 | 0.024174   | 4.36x   | -77.09%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.105508 | 0.023769   | 4.44x   | -77.47%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.135187 | 0.051631   | 2.62x   | -61.81%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.135187 | 0.023743   | 5.69x   | -82.44%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.135187 | 0.023769   | 5.69x   | -82.42%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.071002 | 0.022443   | 3.16x   | -68.39%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.071002 | 0.002856   | 24.86x  | -95.98%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.071002 | 0.002583   | 27.49x  | -96.36%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.103981 | 0.028120   | 3.70x   | -72.96%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.103981 | 0.003475   | 29.92x  | -96.66%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.103981 | 0.003038   | 34.22x  | -97.08%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.078271 | 0.031995   | 2.45x   | -59.12%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.078271 | 0.003071   | 25.49x  | -96.08%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.078271 | 0.002907   | 26.93x  | -96.29%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.079929 | 0.027503   | 2.91x   | -65.59%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.079929 | 0.003211   | 24.89x  | -95.98%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.079929 | 0.003008   | 26.57x  | -96.24%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.080777 | 0.028495   | 2.83x   | -64.72%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.080777 | 0.003714   | 21.75x  | -95.40%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.080777 | 0.003027   | 26.68x  | -96.25%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.077569 | 0.033935   | 2.29x   | -56.25%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.077569 | 0.004161   | 18.64x  | -94.64%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.077569 | 0.003073   | 25.24x  | -96.04%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.076573 | 0.032470   | 2.36x   | -57.60%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.076573 | 0.003169   | 24.16x  | -95.86%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.076573 | 0.002915   | 26.27x  | -96.19%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.082245 | 0.027810   | 2.96x   | -66.19%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.082245 | 0.003224   | 25.51x  | -96.08%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.082245 | 0.003004   | 27.37x  | -96.35%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.115042 | 0.045700   | 2.52x   | -60.28%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.115042 | 0.003736   | 30.79x  | -96.75%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.115042 | 0.003013   | 38.18x  | -97.38%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.111647 | 0.027508   | 4.06x   | -75.36%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.111647 | 0.003239   | 34.47x  | -97.10%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.111647 | 0.002928   | 38.13x  | -97.38%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.077583 | 0.028025   | 2.77x   | -63.88%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.077583 | 0.004158   | 18.66x  | -94.64%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.077583 | 0.003089   | 25.12x  | -96.02%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.081874 | 0.032592   | 2.51x   | -60.19%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.081874 | 0.003325   | 24.63x  | -95.94%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.081874 | 0.002983   | 27.45x  | -96.36%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.081850 | 0.032650   | 2.51x   | -60.11%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.081850 | 0.003306   | 24.76x  | -95.96%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.081850 | 0.003048   | 26.85x  | -96.28%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.088170 | 0.030017   | 2.94x   | -65.96%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.088170 | 0.003341   | 26.39x  | -96.21%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.088170 | 0.002945   | 29.94x  | -96.66%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.109241 | 0.044798   | 2.44x   | -58.99%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.109241 | 0.003607   | 30.28x  | -96.70%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.109241 | 0.002988   | 36.56x  | -97.27%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.086276 | 0.008701   | 9.92x   | -89.91%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.086276 | 0.003453   | 24.99x  | -96.00%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.086276 | 0.002809   | 30.71x  | -96.74%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.117989 | 0.011519   | 10.24x  | -90.24%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.117989 | 0.003996   | 29.53x  | -96.61%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.117989 | 0.003295   | 35.81x  | -97.21%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.092421 | 0.014352   | 6.44x   | -84.47%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.092421 | 0.003743   | 24.69x  | -95.95%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.092421 | 0.003051   | 30.30x  | -96.70%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.094795 | 0.011386   | 8.33x   | -87.99%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.094795 | 0.003745   | 25.31x  | -96.05%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.094795 | 0.003124   | 30.35x  | -96.71%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.096372 | 0.013035   | 7.39x   | -86.47%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.096372 | 0.004319   | 22.31x  | -95.52%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.096372 | 0.003389   | 28.44x  | -96.48%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.092422 | 0.027654   | 3.34x   | -70.08%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.092422 | 0.003925   | 23.54x  | -95.75%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.092422 | 0.003089   | 29.92x  | -96.66%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.091081 | 0.014381   | 6.33x   | -84.21%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.091081 | 0.003575   | 25.48x  | -96.07%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.091081 | 0.003322   | 27.42x  | -96.35%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.094837 | 0.011448   | 8.28x   | -87.93%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.094837 | 0.003607   | 26.29x  | -96.20%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.094837 | 0.003165   | 29.97x  | -96.66%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.126145 | 0.028862   | 4.37x   | -77.12%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.126145 | 0.004068   | 31.01x  | -96.78%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.126145 | 0.003198   | 39.45x  | -97.46%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.120298 | 0.011329   | 10.62x  | -90.58%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.120298 | 0.003755   | 32.04x  | -96.88%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.120298 | 0.004310   | 27.91x  | -96.42%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.092004 | 0.012676   | 7.26x   | -86.22%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.092004 | 0.003901   | 23.58x  | -95.76%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.092004 | 0.003179   | 28.94x  | -96.54%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.093678 | 0.017826   | 5.26x   | -80.97%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.093678 | 0.003917   | 23.91x  | -95.82%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.093678 | 0.003283   | 28.53x  | -96.50%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.093071 | 0.017726   | 5.25x   | -80.95%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.093071 | 0.003746   | 24.84x  | -95.97%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.093071 | 0.003184   | 29.23x  | -96.58%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.095728 | 0.012804   | 7.48x   | -86.62%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.095728 | 0.003957   | 24.19x  | -95.87%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.095728 | 0.003249   | 29.46x  | -96.61%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.124304 | 0.030286   | 4.10x   | -75.64%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.124304 | 0.004005   | 31.04x  | -96.78%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.124304 | 0.003076   | 40.41x  | -97.53%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.064229 | 0.009934   | 6.47x   | -84.53%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.064229 | 0.003398   | 18.90x  | -94.71%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.064229 | 0.002782   | 23.09x  | -95.67%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.096934 | 0.013360   | 7.26x   | -86.22%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.096934 | 0.003921   | 24.72x  | -95.95%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.096934 | 0.003255   | 29.78x  | -96.64%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.070508 | 0.015411   | 4.58x   | -78.14%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.070508 | 0.003564   | 19.78x  | -94.94%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.070508 | 0.003205   | 22.00x  | -95.45%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.075991 | 0.013245   | 5.74x   | -82.57%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.075991 | 0.003821   | 19.89x  | -94.97%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.075991 | 0.003260   | 23.31x  | -95.71%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.074614 | 0.013568   | 5.50x   | -81.82%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.074614 | 0.003992   | 18.69x  | -94.65%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.074614 | 0.003229   | 23.10x  | -95.67%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.070808 | 0.023239   | 3.05x   | -67.18%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.070808 | 0.003863   | 18.33x  | -94.54%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.070808 | 0.003237   | 21.87x  | -95.43%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.071612 | 0.015578   | 4.60x   | -78.25%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.071612 | 0.003649   | 19.63x  | -94.90%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.071612 | 0.003206   | 22.34x  | -95.52%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.072498 | 0.012640   | 5.74x   | -82.56%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.072498 | 0.003603   | 20.12x  | -95.03%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.072498 | 0.003220   | 22.52x  | -95.56%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.106478 | 0.030402   | 3.50x   | -71.45%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.106478 | 0.004057   | 26.25x  | -96.19%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.106478 | 0.003161   | 33.68x  | -97.03%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.101666 | 0.012758   | 7.97x   | -87.45%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.101666 | 0.003499   | 29.06x  | -96.56%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.101666 | 0.003047   | 33.37x  | -97.00%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.075148 | 0.015644   | 4.80x   | -79.18%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.075148 | 0.003945   | 19.05x  | -94.75%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.075148 | 0.003280   | 22.91x  | -95.64%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.073193 | 0.018595   | 3.94x   | -74.59%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.073193 | 0.003734   | 19.60x  | -94.90%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.073193 | 0.003112   | 23.52x  | -95.75%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.072919 | 0.018502   | 3.94x   | -74.63%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.072919 | 0.003746   | 19.47x  | -94.86%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.072919 | 0.003100   | 23.53x  | -95.75%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.073988 | 0.013507   | 5.48x   | -81.74%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.073988 | 0.003771   | 19.62x  | -94.90%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.073988 | 0.003060   | 24.18x  | -95.86%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.098471 | 0.030995   | 3.18x   | -68.52%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.098471 | 0.003889   | 25.32x  | -96.05%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.098471 | 0.003213   | 30.65x  | -96.74%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.384275 | 0.102524   | 3.75x   | -73.32%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.384275 | 0.090658   | 4.24x   | -76.41%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.384275 | 0.093663   | 4.10x   | -75.63%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.487413 | 0.123646   | 3.94x   | -74.63%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.487413 | 0.095381   | 5.11x   | -80.43%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.487413 | 0.095426   | 5.11x   | -80.42%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.371235 | 0.130800   | 2.84x   | -64.77%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.371235 | 0.092229   | 4.03x   | -75.16%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.371235 | 0.093732   | 3.96x   | -74.75%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.405620 | 0.120835   | 3.36x   | -70.21%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.405620 | 0.094082   | 4.31x   | -76.81%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.405620 | 0.096199   | 4.22x   | -76.28%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.396243 | 0.123259   | 3.21x   | -68.89%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.396243 | 0.095435   | 4.15x   | -75.92%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.396243 | 0.096050   | 4.13x   | -75.76%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.388288 | 0.167402   | 2.32x   | -56.89%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.388288 | 0.103048   | 3.77x   | -73.46%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.388288 | 0.094272   | 4.12x   | -75.72%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.371534 | 0.131371   | 2.83x   | -64.64%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.371534 | 0.092236   | 4.03x   | -75.17%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.371534 | 0.093933   | 3.96x   | -74.72%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.375551 | 0.118930   | 3.16x   | -68.33%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.375551 | 0.092601   | 4.06x   | -75.34%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.375551 | 0.094593   | 3.97x   | -74.81%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.559894 | 0.199178   | 2.81x   | -64.43%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.559894 | 0.095938   | 5.84x   | -82.86%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.559894 | 0.096520   | 5.80x   | -82.76%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.488396 | 0.118462   | 4.12x   | -75.74%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.488396 | 0.092033   | 5.31x   | -81.16%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.488396 | 0.093921   | 5.20x   | -80.77%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.379639 | 0.114263   | 3.32x   | -69.90%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.379639 | 0.095363   | 3.98x   | -74.88%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.379639 | 0.094721   | 4.01x   | -75.05%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.390800 | 0.144271   | 2.71x   | -63.08%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.390800 | 0.093260   | 4.19x   | -76.14%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.390800 | 0.094928   | 4.12x   | -75.71%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.393410 | 0.144798   | 2.72x   | -63.19%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.393410 | 0.093671   | 4.20x   | -76.19%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.393410 | 0.095514   | 4.12x   | -75.72%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.399226 | 0.121851   | 3.28x   | -69.48%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.399226 | 0.095361   | 4.19x   | -76.11%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.399226 | 0.096628   | 4.13x   | -75.80%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.502490 | 0.200123   | 2.51x   | -60.17%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.502490 | 0.096352   | 5.22x   | -80.83%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.502490 | 0.098329   | 5.11x   | -80.43%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.283797 | 0.091324   | 3.11x   | -67.82%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.283797 | 0.011630   | 24.40x  | -95.90%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.283797 | 0.010469   | 27.11x  | -96.31%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.390871 | 0.113606   | 3.44x   | -70.94%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.390871 | 0.014067   | 27.79x  | -96.40%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.390871 | 0.012029   | 32.49x  | -96.92%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.278765 | 0.126896   | 2.20x   | -54.48%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.278765 | 0.012717   | 21.92x  | -95.44%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.278765 | 0.011797   | 23.63x  | -95.77%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.293024 | 0.112343   | 2.61x   | -61.66%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.293024 | 0.012913   | 22.69x  | -95.59%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.293024 | 0.011845   | 24.74x  | -95.96%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.307527 | 0.118010   | 2.61x   | -61.63%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.307527 | 0.014881   | 20.67x  | -95.16%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.307527 | 0.011904   | 25.83x  | -96.13%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.284776 | 0.139202   | 2.05x   | -51.12%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.284776 | 0.018834   | 15.12x  | -93.39%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.284776 | 0.013953   | 20.41x  | -95.10%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.290486 | 0.133531   | 2.18x   | -54.03%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.290486 | 0.012851   | 22.60x  | -95.58%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.290486 | 0.011995   | 24.22x  | -95.87%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.284397 | 0.111224   | 2.56x   | -60.89%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.284397 | 0.012595   | 22.58x  | -95.57%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.284397 | 0.011938   | 23.82x  | -95.80%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.455897 | 0.182826   | 2.49x   | -59.90%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.455897 | 0.015319   | 29.76x  | -96.64%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.455897 | 0.012166   | 37.47x  | -97.33%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.396977 | 0.115151   | 3.45x   | -70.99%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.396977 | 0.012903   | 30.77x  | -96.75%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.396977 | 0.012039   | 32.97x  | -96.97%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.302567 | 0.115138   | 2.63x   | -61.95%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.302567 | 0.016609   | 18.22x  | -94.51%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.302567 | 0.012459   | 24.29x  | -95.88%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.315674 | 0.135371   | 2.33x   | -57.12%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.315674 | 0.014582   | 21.65x  | -95.38%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.315674 | 0.012362   | 25.54x  | -96.08%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.327925 | 0.131400   | 2.50x   | -59.93%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.327925 | 0.013483   | 24.32x  | -95.89%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.327925 | 0.012094   | 27.12x  | -96.31%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.298939 | 0.116113   | 2.57x   | -61.16%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.298939 | 0.013609   | 21.97x  | -95.45%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.298939 | 0.011836   | 25.26x  | -96.04%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.415111 | 0.177538   | 2.34x   | -57.23%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.415111 | 0.014406   | 28.82x  | -96.53%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.415111 | 0.012300   | 33.75x  | -97.04%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.326958 | 0.037440   | 8.73x   | -88.55%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.326958 | 0.019015   | 17.19x  | -94.18%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.326958 | 0.015907   | 20.55x  | -95.13%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.443199 | 0.048447   | 9.15x   | -89.07%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.443199 | 0.020856   | 21.25x  | -95.29%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.443199 | 0.018149   | 24.42x  | -95.91%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.323923 | 0.060185   | 5.38x   | -81.42%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.323923 | 0.019524   | 16.59x  | -93.97%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.323923 | 0.017426   | 18.59x  | -94.62%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.347237 | 0.049050   | 7.08x   | -85.87%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.347237 | 0.020193   | 17.20x  | -94.18%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.347237 | 0.017522   | 19.82x  | -94.95%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.347322 | 0.052969   | 6.56x   | -84.75%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.347322 | 0.021710   | 16.00x  | -93.75%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.347322 | 0.018319   | 18.96x  | -94.73%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.333878 | 0.111284   | 3.00x   | -66.67%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.333878 | 0.021088   | 15.83x  | -93.68%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.333878 | 0.018469   | 18.08x  | -94.47%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.328188 | 0.061808   | 5.31x   | -81.17%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.328188 | 0.018868   | 17.39x  | -94.25%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.328188 | 0.017941   | 18.29x  | -94.53%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.357133 | 0.053636   | 6.66x   | -84.98%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.357133 | 0.020113   | 17.76x  | -94.37%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.357133 | 0.018131   | 19.70x  | -94.92%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.514495 | 0.119579   | 4.30x   | -76.76%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.514495 | 0.021490   | 23.94x  | -95.82%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.514495 | 0.017698   | 29.07x  | -96.56%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.450629 | 0.048779   | 9.24x   | -89.18%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.450629 | 0.020329   | 22.17x  | -95.49%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.450629 | 0.019074   | 23.63x  | -95.77%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.340849 | 0.054090   | 6.30x   | -84.13%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.340849 | 0.020653   | 16.50x  | -93.94%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.340849 | 0.018145   | 18.78x  | -94.68%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.350692 | 0.074879   | 4.68x   | -78.65%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.350692 | 0.021171   | 16.56x  | -93.96%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.350692 | 0.018067   | 19.41x  | -94.85%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.354583 | 0.075865   | 4.67x   | -78.60%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.354583 | 0.024370   | 14.55x  | -93.13%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.354583 | 0.017819   | 19.90x  | -94.97%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.352519 | 0.058010   | 6.08x   | -83.54%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.352519 | 0.020444   | 17.24x  | -94.20%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.352519 | 0.018196   | 19.37x  | -94.84%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.468878 | 0.124950   | 3.75x   | -73.35%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.468878 | 0.020679   | 22.67x  | -95.59%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.468878 | 0.017667   | 26.54x  | -96.23%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.270041 | 0.047584   | 5.68x   | -82.38%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.270041 | 0.021383   | 12.63x  | -92.08%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.270041 | 0.018621   | 14.50x  | -93.10%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.369557 | 0.062266   | 5.94x   | -83.15%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.369557 | 0.023501   | 15.73x  | -93.64%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.369557 | 0.021004   | 17.59x  | -94.32%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.253718 | 0.070283   | 3.61x   | -72.30%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.253718 | 0.021223   | 11.96x  | -91.64%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.253718 | 0.020157   | 12.59x  | -92.06%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.269642 | 0.062145   | 4.34x   | -76.95%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.269642 | 0.022553   | 11.96x  | -91.64%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.269642 | 0.020005   | 13.48x  | -92.58%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.278158 | 0.067492   | 4.12x   | -75.74%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.278158 | 0.023144   | 12.02x  | -91.68%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.278158 | 0.020755   | 13.40x  | -92.54%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.269184 | 0.102363   | 2.63x   | -61.97%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.269184 | 0.023036   | 11.69x  | -91.44%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.269184 | 0.020632   | 13.05x  | -92.34%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.256751 | 0.071942   | 3.57x   | -71.98%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.256751 | 0.022845   | 11.24x  | -91.10%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.256751 | 0.020292   | 12.65x  | -92.10%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.259611 | 0.061740   | 4.20x   | -76.22%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.259611 | 0.021442   | 12.11x  | -91.74%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.259611 | 0.019467   | 13.34x  | -92.50%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.413107 | 0.130224   | 3.17x   | -68.48%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.413107 | 0.023384   | 17.67x  | -94.34%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.413107 | 0.019107   | 21.62x  | -95.37%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.368849 | 0.067092   | 5.50x   | -81.81%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.368849 | 0.023463   | 15.72x  | -93.64%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.368849 | 0.020081   | 18.37x  | -94.56%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.269181 | 0.072578   | 3.71x   | -73.04%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.269181 | 0.024107   | 11.17x  | -91.04%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.269181 | 0.021570   | 12.48x  | -91.99%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.272989 | 0.084292   | 3.24x   | -69.12%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.272989 | 0.022528   | 12.12x  | -91.75%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.272989 | 0.021049   | 12.97x  | -92.29%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.282724 | 0.083709   | 3.38x   | -70.39%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.282724 | 0.022990   | 12.30x  | -91.87%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.282724 | 0.020722   | 13.64x  | -92.67%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.277890 | 0.066591   | 4.17x   | -76.04%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.277890 | 0.023567   | 11.79x  | -91.52%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.277890 | 0.021576   | 12.88x  | -92.24%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.389595 | 0.134045   | 2.91x   | -65.59%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.389595 | 0.022590   | 17.25x  | -94.20%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.389595 | 0.020585   | 18.93x  | -94.72%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.081102 | 0.022358   | 3.63x   | -72.43%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.081102 | 0.019914   | 4.07x   | -75.45%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.081102 | 0.020562   | 3.94x   | -74.65%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.109125 | 0.027561   | 3.96x   | -74.74%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.109125 | 0.020948   | 5.21x   | -80.80%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.109125 | 0.020959   | 5.21x   | -80.79%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.086222 | 0.028937   | 2.98x   | -66.44%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.086222 | 0.020186   | 4.27x   | -76.59%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.086222 | 0.020514   | 4.20x   | -76.21%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.090056 | 0.027019   | 3.33x   | -70.00%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.090056 | 0.020597   | 4.37x   | -77.13%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.090056 | 0.020957   | 4.30x   | -76.73%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.090130 | 0.028312   | 3.18x   | -68.59%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.090130 | 0.021100   | 4.27x   | -76.59%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.090130 | 0.021186   | 4.25x   | -76.49%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.083700 | 0.036726   | 2.28x   | -56.12%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.083700 | 0.020342   | 4.11x   | -75.70%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.083700 | 0.020684   | 4.05x   | -75.29%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.087483 | 0.029243   | 2.99x   | -66.57%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.087483 | 0.020151   | 4.34x   | -76.97%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.087483 | 0.020890   | 4.19x   | -76.12%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.092638 | 0.026211   | 3.53x   | -71.71%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.092638 | 0.020685   | 4.48x   | -77.67%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.092638 | 0.020817   | 4.45x   | -77.53%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.119015 | 0.044180   | 2.69x   | -62.88%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.119015 | 0.021206   | 5.61x   | -82.18%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.119015 | 0.021305   | 5.59x   | -82.10%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.112991 | 0.026048   | 4.34x   | -76.95%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.112991 | 0.020321   | 5.56x   | -82.02%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.112991 | 0.021872   | 5.17x   | -80.64%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.084114 | 0.024762   | 3.40x   | -70.56%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.084114 | 0.020699   | 4.06x   | -75.39%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.084114 | 0.020905   | 4.02x   | -75.15%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.088146 | 0.031668   | 2.78x   | -64.07%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.088146 | 0.020834   | 4.23x   | -76.36%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.088146 | 0.021605   | 4.08x   | -75.49%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.090216 | 0.033140   | 2.72x   | -63.27%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.090216 | 0.021203   | 4.25x   | -76.50%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.090216 | 0.021113   | 4.27x   | -76.60%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.089459 | 0.026896   | 3.33x   | -69.94%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.089459 | 0.020888   | 4.28x   | -76.65%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.089459 | 0.021155   | 4.23x   | -76.35%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.115708 | 0.043955   | 2.63x   | -62.01%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.115708 | 0.021389   | 5.41x   | -81.51%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.115708 | 0.021315   | 5.43x   | -81.58%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.063524 | 0.019732   | 3.22x   | -68.94%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.063524 | 0.002584   | 24.58x  | -95.93%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.063524 | 0.002293   | 27.70x  | -96.39%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.097189 | 0.024468   | 3.97x   | -74.82%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.097189 | 0.003025   | 32.13x  | -96.89%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.097189 | 0.002632   | 36.93x  | -97.29%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.074534 | 0.028221   | 2.64x   | -62.14%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.074534 | 0.002811   | 26.52x  | -96.23%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.074534 | 0.002607   | 28.59x  | -96.50%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.078467 | 0.024193   | 3.24x   | -69.17%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.078467 | 0.002817   | 27.85x  | -96.41%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.078467 | 0.002665   | 29.45x  | -96.60%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.077053 | 0.025296   | 3.05x   | -67.17%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.077053 | 0.003241   | 23.78x  | -95.79%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.077053 | 0.002595   | 29.69x  | -96.63%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.071685 | 0.029892   | 2.40x   | -58.30%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.071685 | 0.003648   | 19.65x  | -94.91%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.071685 | 0.002747   | 26.09x  | -96.17%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.074548 | 0.028361   | 2.63x   | -61.96%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.074548 | 0.002838   | 26.27x  | -96.19%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.074548 | 0.002635   | 28.29x  | -96.47%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.076221 | 0.024577   | 3.10x   | -67.76%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.076221 | 0.002765   | 27.57x  | -96.37%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.076221 | 0.002582   | 29.52x  | -96.61%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.103035 | 0.039747   | 2.59x   | -61.42%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.103035 | 0.003378   | 30.50x  | -96.72%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.103035 | 0.002664   | 38.68x  | -97.41%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.102171 | 0.023870   | 4.28x   | -76.64%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.102171 | 0.002774   | 36.83x  | -97.29%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.102171 | 0.002589   | 39.47x  | -97.47%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.070962 | 0.024162   | 2.94x   | -65.95%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.070962 | 0.003655   | 19.41x  | -94.85%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.070962 | 0.002727   | 26.02x  | -96.16%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.075754 | 0.028518   | 2.66x   | -62.35%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.075754 | 0.002937   | 25.79x  | -96.12%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.075754 | 0.002677   | 28.30x  | -96.47%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.076530 | 0.028207   | 2.71x   | -63.14%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.076530 | 0.002892   | 26.46x  | -96.22%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.076530 | 0.002610   | 29.32x  | -96.59%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.076427 | 0.024272   | 3.15x   | -68.24%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.076427 | 0.002940   | 25.99x  | -96.15%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.076427 | 0.002592   | 29.49x  | -96.61%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.099484 | 0.039002   | 2.55x   | -60.80%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.099484 | 0.003077   | 32.33x  | -96.91%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.099484 | 0.002644   | 37.63x  | -97.34%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.069585 | 0.007115   | 9.78x   | -89.78%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.069585 | 0.003111   | 22.37x  | -95.53%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.069585 | 0.002324   | 29.94x  | -96.66%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.101884 | 0.009468   | 10.76x  | -90.71%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.101884 | 0.003332   | 30.58x  | -96.73%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.101884 | 0.002772   | 36.76x  | -97.28%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.079471 | 0.012121   | 6.56x   | -84.75%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.079471 | 0.003007   | 26.43x  | -96.22%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.079471 | 0.002686   | 29.59x  | -96.62%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.080828 | 0.009342   | 8.65x   | -88.44%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.080828 | 0.003200   | 25.26x  | -96.04%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.080828 | 0.002602   | 31.06x  | -96.78%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.080988 | 0.010132   | 7.99x   | -87.49%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.080988 | 0.003517   | 23.03x  | -95.66%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.080988 | 0.002891   | 28.02x  | -96.43%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.077396 | 0.023230   | 3.33x   | -69.99%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.077396 | 0.003365   | 23.00x  | -95.65%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.077396 | 0.002873   | 26.94x  | -96.29%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.078456 | 0.012090   | 6.49x   | -84.59%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.078456 | 0.003058   | 25.65x  | -96.10%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.078456 | 0.002598   | 30.20x  | -96.69%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.079165 | 0.009338   | 8.48x   | -88.20%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.079165 | 0.003099   | 25.54x  | -96.09%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.079165 | 0.002668   | 29.67x  | -96.63%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.109863 | 0.024936   | 4.41x   | -77.30%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.109863 | 0.003591   | 30.60x  | -96.73%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.109863 | 0.002936   | 37.42x  | -97.33%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.105326 | 0.009585   | 10.99x  | -90.90%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.105326 | 0.003091   | 34.07x  | -97.06%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.105326 | 0.002706   | 38.92x  | -97.43%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.077114 | 0.010458   | 7.37x   | -86.44%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.077114 | 0.003360   | 22.95x  | -95.64%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.077114 | 0.002748   | 28.06x  | -96.44%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.081533 | 0.015075   | 5.41x   | -81.51%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.081533 | 0.003198   | 25.49x  | -96.08%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.081533 | 0.002663   | 30.62x  | -96.73%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.080479 | 0.014888   | 5.41x   | -81.50%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.080479 | 0.003304   | 24.36x  | -95.89%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.080479 | 0.002706   | 29.74x  | -96.64%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.081140 | 0.009943   | 8.16x   | -87.75%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.081140 | 0.003446   | 23.55x  | -95.75%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.081140 | 0.002809   | 28.88x  | -96.54%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.104349 | 0.026510   | 3.94x   | -74.59%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.104349 | 0.003421   | 30.50x  | -96.72%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.104349 | 0.002761   | 37.79x  | -97.35%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.054076 | 0.008693   | 6.22x   | -83.92%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.054076 | 0.002885   | 18.74x  | -94.66%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.054076 | 0.002507   | 21.57x  | -95.36%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.088220 | 0.011459   | 7.70x   | -87.01%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.088220 | 0.003394   | 25.99x  | -96.15%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.088220 | 0.002819   | 31.29x  | -96.80%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.065375 | 0.013595   | 4.81x   | -79.20%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.065375 | 0.003175   | 20.59x  | -95.14%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.065375 | 0.002750   | 23.77x  | -95.79%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.068608 | 0.011441   | 6.00x   | -83.32%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.068608 | 0.003222   | 21.30x  | -95.30%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.068608 | 0.002646   | 25.92x  | -96.14%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.067506 | 0.012188   | 5.54x   | -81.94%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.067506 | 0.003517   | 19.20x  | -94.79%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.067506 | 0.002809   | 24.03x  | -95.84%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.063953 | 0.020456   | 3.13x   | -68.01%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.063953 | 0.003289   | 19.45x  | -94.86%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.063953 | 0.002825   | 22.64x  | -95.58%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.065291 | 0.013583   | 4.81x   | -79.20%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.065291 | 0.003032   | 21.54x  | -95.36%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.065291 | 0.002707   | 24.12x  | -95.85%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.067514 | 0.012433   | 5.43x   | -81.58%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.067514 | 0.003079   | 21.93x  | -95.44%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.067514 | 0.002845   | 23.73x  | -95.79%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.096512 | 0.026444   | 3.65x   | -72.60%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.096512 | 0.003469   | 27.82x  | -96.41%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.096512 | 0.002823   | 34.19x  | -97.08%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.091413 | 0.011203   | 8.16x   | -87.74%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.091413 | 0.003048   | 29.99x  | -96.67%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.091413 | 0.002760   | 33.12x  | -96.98%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.064320 | 0.013810   | 4.66x   | -78.53%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.064320 | 0.003476   | 18.51x  | -94.60%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.064320 | 0.002850   | 22.57x  | -95.57%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.067399 | 0.016495   | 4.09x   | -75.53%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.067399 | 0.003516   | 19.17x  | -94.78%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.067399 | 0.002801   | 24.06x  | -95.84%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.069781 | 0.018247   | 3.82x   | -73.85%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.069781 | 0.003451   | 20.22x  | -95.05%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.069781 | 0.002926   | 23.85x  | -95.81%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.070573 | 0.012049   | 5.86x   | -82.93%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.070573 | 0.003461   | 20.39x  | -95.10%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.070573 | 0.002797   | 25.23x  | -96.04%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.090275 | 0.027635   | 3.27x   | -69.39%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.090275 | 0.003521   | 25.64x  | -96.10%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.090275 | 0.002847   | 31.71x  | -96.85%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.190947 | 0.050324   | 3.79x   | -73.64%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.190947 | 0.046287   | 4.13x   | -75.76%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.190947 | 0.049267   | 3.88x   | -74.20%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.251713 | 0.062052   | 4.06x   | -75.35%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.251713 | 0.047634   | 5.28x   | -81.08%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.251713 | 0.047783   | 5.27x   | -81.02%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.193336 | 0.069615   | 2.78x   | -63.99%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.193336 | 0.048991   | 3.95x   | -74.66%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.193336 | 0.048829   | 3.96x   | -74.74%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.209224 | 0.062441   | 3.35x   | -70.16%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.209224 | 0.050288   | 4.16x   | -75.96%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.209224 | 0.047499   | 4.40x   | -77.30%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.207474 | 0.063344   | 3.28x   | -69.47%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.207474 | 0.048717   | 4.26x   | -76.52%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.207474 | 0.048413   | 4.29x   | -76.67%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.206214 | 0.086358   | 2.39x   | -58.12%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.206214 | 0.050120   | 4.11x   | -75.70%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.206214 | 0.050072   | 4.12x   | -75.72%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.215372 | 0.066488   | 3.24x   | -69.13%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.215372 | 0.046215   | 4.66x   | -78.54%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.215372 | 0.047297   | 4.55x   | -78.04%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.197741 | 0.060800   | 3.25x   | -69.25%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.197741 | 0.047208   | 4.19x   | -76.13%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.197741 | 0.048063   | 4.11x   | -75.69%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.265855 | 0.100091   | 2.66x   | -62.35%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.265855 | 0.048082   | 5.53x   | -81.91%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.265855 | 0.048389   | 5.49x   | -81.80%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.257903 | 0.060179   | 4.29x   | -76.67%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.257903 | 0.046844   | 5.51x   | -81.84%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.257903 | 0.046652   | 5.53x   | -81.91%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.191325 | 0.061432   | 3.11x   | -67.89%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.191325 | 0.049588   | 3.86x   | -74.08%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.191325 | 0.047505   | 4.03x   | -75.17%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.195224 | 0.072090   | 2.71x   | -63.07%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.195224 | 0.047035   | 4.15x   | -75.91%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.195224 | 0.048180   | 4.05x   | -75.32%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.206524 | 0.072810   | 2.84x   | -64.75%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.206524 | 0.046925   | 4.40x   | -77.28%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.206524 | 0.047669   | 4.33x   | -76.92%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.226860 | 0.063072   | 3.60x   | -72.20%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.226860 | 0.053062   | 4.28x   | -76.61%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.226860 | 0.051026   | 4.45x   | -77.51%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.261324 | 0.104146   | 2.51x   | -60.15%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.261324 | 0.061952   | 4.22x   | -76.29%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.261324 | 0.057957   | 4.51x   | -77.82%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.153716 | 0.047220   | 3.26x   | -69.28%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.153716 | 0.006159   | 24.96x  | -95.99%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.153716 | 0.005485   | 28.02x  | -96.43%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.219477 | 0.056821   | 3.86x   | -74.11%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.219477 | 0.006867   | 31.96x  | -96.87%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.219477 | 0.005877   | 37.34x  | -97.32%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.149634 | 0.066722   | 2.24x   | -55.41%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.149634 | 0.007304   | 20.49x  | -95.12%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.149634 | 0.006035   | 24.79x  | -95.97%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.156375 | 0.055637   | 2.81x   | -64.42%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.156375 | 0.006503   | 24.05x  | -95.84%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.156375 | 0.005999   | 26.07x  | -96.16%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.156757 | 0.061977   | 2.53x   | -60.46%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.156757 | 0.008397   | 18.67x  | -94.64%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.156757 | 0.005967   | 26.27x  | -96.19%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.149968 | 0.067969   | 2.21x   | -54.68%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.149968 | 0.008168   | 18.36x  | -94.55%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.149968 | 0.006140   | 24.42x  | -95.91%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.151415 | 0.065664   | 2.31x   | -56.63%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.151415 | 0.008776   | 17.25x  | -94.20%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.151415 | 0.006911   | 21.91x  | -95.44%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.149057 | 0.058282   | 2.56x   | -60.90%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.149057 | 0.006304   | 23.64x  | -95.77%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.149057 | 0.005995   | 24.86x  | -95.98%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.224158 | 0.093818   | 2.39x   | -58.15%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.224158 | 0.007492   | 29.92x  | -96.66%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.224158 | 0.006006   | 37.32x  | -97.32%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.204514 | 0.055323   | 3.70x   | -72.95%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.204514 | 0.006309   | 32.41x  | -96.91%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.204514 | 0.005881   | 34.78x  | -97.12%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.149125 | 0.058078   | 2.57x   | -61.05%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.149125 | 0.008542   | 17.46x  | -94.27%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.149125 | 0.006338   | 23.53x  | -95.75%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.154534 | 0.064723   | 2.39x   | -58.12%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.154534 | 0.006606   | 23.39x  | -95.73%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.154534 | 0.005914   | 26.13x  | -96.17%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.163470 | 0.068574   | 2.38x   | -58.05%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.163470 | 0.006585   | 24.82x  | -95.97%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.163470 | 0.005854   | 27.92x  | -96.42%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.149206 | 0.055314   | 2.70x   | -62.93%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.149206 | 0.006621   | 22.53x  | -95.56%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.149206 | 0.005818   | 25.64x  | -96.10%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.200823 | 0.086766   | 2.31x   | -56.79%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.200823 | 0.006940   | 28.94x  | -96.54%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.200823 | 0.005868   | 34.22x  | -97.08%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.157803 | 0.018446   | 8.55x   | -88.31%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.157803 | 0.007049   | 22.39x  | -95.53%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.157803 | 0.005040   | 31.31x  | -96.81%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.219343 | 0.023807   | 9.21x   | -89.15%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.219343 | 0.007743   | 28.33x  | -96.47%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.219343 | 0.006298   | 34.83x  | -97.13%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.166483 | 0.031727   | 5.25x   | -80.94%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.166483 | 0.006845   | 24.32x  | -95.89%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.166483 | 0.006690   | 24.89x  | -95.98%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.170533 | 0.023345   | 7.30x   | -86.31%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.170533 | 0.007401   | 23.04x  | -95.66%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.170533 | 0.006080   | 28.05x  | -96.43%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.172457 | 0.025232   | 6.83x   | -85.37%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.172457 | 0.008164   | 21.12x  | -95.27%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.172457 | 0.006030   | 28.60x  | -96.50%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.166492 | 0.054487   | 3.06x   | -67.27%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.166492 | 0.007602   | 21.90x  | -95.43%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.166492 | 0.006652   | 25.03x  | -96.00%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.165700 | 0.029658   | 5.59x   | -82.10%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.165700 | 0.007094   | 23.36x  | -95.72%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.165700 | 0.005925   | 27.97x  | -96.42%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.170401 | 0.023673   | 7.20x   | -86.11%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.170401 | 0.006957   | 24.49x  | -95.92%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.170401 | 0.005940   | 28.68x  | -96.51%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.235011 | 0.058253   | 4.03x   | -75.21%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.235011 | 0.008001   | 29.37x  | -96.60%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.235011 | 0.006294   | 37.34x  | -97.32%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.223087 | 0.024107   | 9.25x   | -89.19%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.223087 | 0.007216   | 30.91x  | -96.77%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.223087 | 0.005938   | 37.57x  | -97.34%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.166445 | 0.025948   | 6.41x   | -84.41%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.166445 | 0.007744   | 21.49x  | -95.35%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.166445 | 0.006415   | 25.95x  | -96.15%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.168472 | 0.036024   | 4.68x   | -78.62%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.168472 | 0.007511   | 22.43x  | -95.54%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.168472 | 0.005708   | 29.52x  | -96.61%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.167062 | 0.036266   | 4.61x   | -78.29%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.167062 | 0.007268   | 22.98x  | -95.65%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.167062 | 0.005811   | 28.75x  | -96.52%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.169794 | 0.025054   | 6.78x   | -85.24%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.169794 | 0.007654   | 22.18x  | -95.49%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.169794 | 0.005871   | 28.92x  | -96.54%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.225962 | 0.061605   | 3.67x   | -72.74%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.225962 | 0.008134   | 27.78x  | -96.40%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.225962 | 0.008027   | 28.15x  | -96.45%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.122943 | 0.019427   | 6.33x   | -84.20%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.122943 | 0.006441   | 19.09x  | -94.76%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.122943 | 0.005593   | 21.98x  | -95.45%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.184740 | 0.025886   | 7.14x   | -85.99%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.184740 | 0.007841   | 23.56x  | -95.76%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.184740 | 0.006209   | 29.75x  | -96.64%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.130729 | 0.030297   | 4.31x   | -76.82%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.130729 | 0.006644   | 19.67x  | -94.92%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.130729 | 0.005948   | 21.98x  | -95.45%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.134592 | 0.025072   | 5.37x   | -81.37%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.134592 | 0.007071   | 19.04x  | -94.75%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.134592 | 0.006458   | 20.84x  | -95.20%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.134444 | 0.026512   | 5.07x   | -80.28%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.134444 | 0.007522   | 17.87x  | -94.40%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.134444 | 0.006633   | 20.27x  | -95.07%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.132721 | 0.045699   | 2.90x   | -65.57%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.132721 | 0.007253   | 18.30x  | -94.53%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.132721 | 0.006252   | 21.23x  | -95.29%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.128820 | 0.029799   | 4.32x   | -76.87%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.128820 | 0.006626   | 19.44x  | -94.86%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.128820 | 0.006204   | 20.76x  | -95.18%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.131823 | 0.024860   | 5.30x   | -81.14%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.131823 | 0.006679   | 19.74x  | -94.93%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.131823 | 0.006041   | 21.82x  | -95.42%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.198931 | 0.059017   | 3.37x   | -70.33%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.198931 | 0.007761   | 25.63x  | -96.10%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.198931 | 0.005986   | 33.23x  | -96.99%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.191050 | 0.025206   | 7.58x   | -86.81%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.191050 | 0.007459   | 25.61x  | -96.10%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.191050 | 0.006210   | 30.76x  | -96.75%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.132765 | 0.030998   | 4.28x   | -76.65%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.132765 | 0.007848   | 16.92x  | -94.09%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.132765 | 0.006292   | 21.10x  | -95.26%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.136352 | 0.036459   | 3.74x   | -73.26%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.136352 | 0.007820   | 17.44x  | -94.26%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.136352 | 0.006313   | 21.60x  | -95.37%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.134796 | 0.036315   | 3.71x   | -73.06%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.134796 | 0.007741   | 17.41x  | -94.26%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.134796 | 0.006008   | 22.44x  | -95.54%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.136018 | 0.039301   | 3.46x   | -71.11%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.136018 | 0.010648   | 12.77x  | -92.17%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.136018 | 0.007714   | 17.63x  | -94.33%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.201658 | 0.062293   | 3.24x   | -69.11%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.201658 | 0.007692   | 26.22x  | -96.19%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.201658 | 0.006312   | 31.95x  | -96.87%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.330118 | 0.106349   | 3.10x   | -67.78%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.330118 | 0.081797   | 4.04x   | -75.22%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.330118 | 0.083667   | 3.95x   | -74.66%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.433322 | 0.109966   | 3.94x   | -74.62%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.433322 | 0.084370   | 5.14x   | -80.53%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.433322 | 0.084680   | 5.12x   | -80.46%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.330505 | 0.118163   | 2.80x   | -64.25%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.330505 | 0.082391   | 4.01x   | -75.07%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.330505 | 0.084070   | 3.93x   | -74.56%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.349637 | 0.108057   | 3.24x   | -69.09%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.349637 | 0.081826   | 4.27x   | -76.60%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.349637 | 0.084819   | 4.12x   | -75.74%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.347665 | 0.108866   | 3.19x   | -68.69%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.347665 | 0.084566   | 4.11x   | -75.68%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.347665 | 0.085386   | 4.07x   | -75.44%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.337686 | 0.149281   | 2.26x   | -55.79%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.337686 | 0.082125   | 4.11x   | -75.68%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.337686 | 0.084262   | 4.01x   | -75.05%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.332318 | 0.124301   | 2.67x   | -62.60%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.332318 | 0.080546   | 4.13x   | -75.76%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.332318 | 0.082363   | 4.03x   | -75.22%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.332403 | 0.104409   | 3.18x   | -68.59%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.332403 | 0.080611   | 4.12x   | -75.75%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.332403 | 0.082326   | 4.04x   | -75.23%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.467836 | 0.172854   | 2.71x   | -63.05%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.467836 | 0.082748   | 5.65x   | -82.31%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.467836 | 0.083212   | 5.62x   | -82.21%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.430457 | 0.104001   | 4.14x   | -75.84%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.430457 | 0.080422   | 5.35x   | -81.32%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.430457 | 0.083199   | 5.17x   | -80.67%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.361620 | 0.109494   | 3.30x   | -69.72%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.361620 | 0.083191   | 4.35x   | -76.99%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.361620 | 0.093616   | 3.86x   | -74.11%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.366712 | 0.125106   | 2.93x   | -65.88%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.366712 | 0.084392   | 4.35x   | -76.99%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.366712 | 0.095296   | 3.85x   | -74.01%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.355942 | 0.129782   | 2.74x   | -63.54%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.355942 | 0.083278   | 4.27x   | -76.60%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.355942 | 0.085941   | 4.14x   | -75.86%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.355892 | 0.108634   | 3.28x   | -69.48%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.355892 | 0.083626   | 4.26x   | -76.50%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.355892 | 0.083326   | 4.27x   | -76.59%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.440401 | 0.175531   | 2.51x   | -60.14%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.440401 | 0.082579   | 5.33x   | -81.25%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.440401 | 0.083293   | 5.29x   | -81.09%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.239254 | 0.078531   | 3.05x   | -67.18%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.239254 | 0.010023   | 23.87x  | -95.81%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.239254 | 0.009166   | 26.10x  | -96.17%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.344790 | 0.097847   | 3.52x   | -71.62%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.344790 | 0.011981   | 28.78x  | -96.53%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.344790 | 0.010370   | 33.25x  | -96.99%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.248300 | 0.113104   | 2.20x   | -54.45%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.248300 | 0.013379   | 18.56x  | -94.61%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.248300 | 0.011937   | 20.80x  | -95.19%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.257540 | 0.096051   | 2.68x   | -62.70%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.257540 | 0.011070   | 23.26x  | -95.70%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.257540 | 0.010135   | 25.41x  | -96.06%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.260071 | 0.101258   | 2.57x   | -61.07%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.260071 | 0.014887   | 17.47x  | -94.28%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.260071 | 0.010543   | 24.67x  | -95.95%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.246878 | 0.116464   | 2.12x   | -52.83%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.246878 | 0.014567   | 16.95x  | -94.10%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.246878 | 0.010908   | 22.63x  | -95.58%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.246128 | 0.113812   | 2.16x   | -53.76%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.246128 | 0.010999   | 22.38x  | -95.53%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.246128 | 0.010308   | 23.88x  | -95.81%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.252087 | 0.099038   | 2.55x   | -60.71%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.252087 | 0.010913   | 23.10x  | -95.67%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.252087 | 0.010339   | 24.38x  | -95.90%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.394495 | 0.156149   | 2.53x   | -60.42%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.394495 | 0.013264   | 29.74x  | -96.64%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.394495 | 0.010512   | 37.53x  | -97.34%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.340291 | 0.095032   | 3.58x   | -72.07%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.340291 | 0.010955   | 31.06x  | -96.78%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.340291 | 0.010190   | 33.39x  | -97.01%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.256217 | 0.097445   | 2.63x   | -61.97%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.256217 | 0.014400   | 17.79x  | -94.38%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.256217 | 0.010821   | 23.68x  | -95.78%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.247505 | 0.112569   | 2.20x   | -54.52%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.247505 | 0.011487   | 21.55x  | -95.36%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.247505 | 0.010399   | 23.80x  | -95.80%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.260846 | 0.111507   | 2.34x   | -57.25%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.260846 | 0.011491   | 22.70x  | -95.59%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.260846 | 0.010496   | 24.85x  | -95.98%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.253484 | 0.096042   | 2.64x   | -62.11%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.253484 | 0.011541   | 21.96x  | -95.45%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.253484 | 0.010221   | 24.80x  | -95.97%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.358694 | 0.156953   | 2.29x   | -56.24%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.358694 | 0.013131   | 27.32x  | -96.34%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.358694 | 0.010714   | 33.48x  | -97.01%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.291706 | 0.028038   | 10.40x  | -90.39%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.291706 | 0.011759   | 24.81x  | -95.97%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.291706 | 0.009458   | 30.84x  | -96.76%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.390009 | 0.038280   | 10.19x  | -90.18%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.390009 | 0.013751   | 28.36x  | -96.47%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.390009 | 0.011179   | 34.89x  | -97.13%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.285290 | 0.047672   | 5.98x   | -83.29%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.285290 | 0.011931   | 23.91x  | -95.82%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.285290 | 0.010373   | 27.50x  | -96.36%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.299300 | 0.036917   | 8.11x   | -87.67%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.299300 | 0.012491   | 23.96x  | -95.83%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.299300 | 0.010579   | 28.29x  | -96.47%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.311526 | 0.040863   | 7.62x   | -86.88%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.311526 | 0.014042   | 22.19x  | -95.49%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.311526 | 0.011513   | 27.06x  | -96.30%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.305467 | 0.094723   | 3.22x   | -68.99%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.305467 | 0.013851   | 22.05x  | -95.47%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.305467 | 0.011718   | 26.07x  | -96.16%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.293755 | 0.048902   | 6.01x   | -83.35%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.293755 | 0.012427   | 23.64x  | -95.77%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.293755 | 0.010935   | 26.86x  | -96.28%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.310731 | 0.038299   | 8.11x   | -87.67%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.310731 | 0.012239   | 25.39x  | -96.06%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.310731 | 0.010919   | 28.46x  | -96.49%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.464136 | 0.101560   | 4.57x   | -78.12%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.464136 | 0.014526   | 31.95x  | -96.87%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.464136 | 0.011572   | 40.11x  | -97.51%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.398167 | 0.037790   | 10.54x  | -90.51%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.398167 | 0.013092   | 30.41x  | -96.71%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.398167 | 0.011341   | 35.11x  | -97.15%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.295542 | 0.041833   | 7.06x   | -85.85%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.295542 | 0.013174   | 22.43x  | -95.54%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.295542 | 0.010907   | 27.10x  | -96.31%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.299362 | 0.059922   | 5.00x   | -79.98%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.299362 | 0.012956   | 23.11x  | -95.67%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.299362 | 0.010574   | 28.31x  | -96.47%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.304873 | 0.060700   | 5.02x   | -80.09%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.304873 | 0.013211   | 23.08x  | -95.67%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.304873 | 0.010860   | 28.07x  | -96.44%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.307314 | 0.040432   | 7.60x   | -86.84%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.307314 | 0.013488   | 22.78x  | -95.61%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.307314 | 0.011151   | 27.56x  | -96.37%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.408907 | 0.104575   | 3.91x   | -74.43%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.408907 | 0.013503   | 30.28x  | -96.70%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.408907 | 0.011214   | 36.46x  | -97.26%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.237552 | 0.040738   | 5.83x   | -82.85%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.237552 | 0.017789   | 13.35x  | -92.51%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.237552 | 0.015447   | 15.38x  | -93.50%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.327819 | 0.052468   | 6.25x   | -83.99%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.327819 | 0.019949   | 16.43x  | -93.91%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.327819 | 0.017533   | 18.70x  | -94.65%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.234236 | 0.068899   | 3.40x   | -70.59%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.234236 | 0.019345   | 12.11x  | -91.74%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.234236 | 0.017432   | 13.44x  | -92.56%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.250057 | 0.053179   | 4.70x   | -78.73%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.250057 | 0.019990   | 12.51x  | -92.01%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.250057 | 0.017570   | 14.23x  | -92.97%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.254791 | 0.054406   | 4.68x   | -78.65%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.254791 | 0.019948   | 12.77x  | -92.17%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.254791 | 0.017114   | 14.89x  | -93.28%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.239435 | 0.088404   | 2.71x   | -63.08%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.239435 | 0.019633   | 12.20x  | -91.80%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.239435 | 0.017346   | 13.80x  | -92.76%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.231827 | 0.061273   | 3.78x   | -73.57%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.231827 | 0.018741   | 12.37x  | -91.92%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.231827 | 0.016581   | 13.98x  | -92.85%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.237788 | 0.050689   | 4.69x   | -78.68%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.237788 | 0.018016   | 13.20x  | -92.42%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.237788 | 0.016655   | 14.28x  | -93.00%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.376552 | 0.112190   | 3.36x   | -70.21%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.376552 | 0.020126   | 18.71x  | -94.66%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.376552 | 0.016943   | 22.22x  | -95.50%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.336373 | 0.051537   | 6.53x   | -84.68%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.336373 | 0.018731   | 17.96x  | -94.43%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.336373 | 0.017583   | 19.13x  | -94.77%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.250088 | 0.062082   | 4.03x   | -75.18%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.250088 | 0.019745   | 12.67x  | -92.10%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.250088 | 0.017028   | 14.69x  | -93.19%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.253690 | 0.071833   | 3.53x   | -71.68%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.253690 | 0.019346   | 13.11x  | -92.37%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.253690 | 0.016393   | 15.48x  | -93.54%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.246909 | 0.071716   | 3.44x   | -70.95%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.246909 | 0.019680   | 12.55x  | -92.03%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.246909 | 0.016799   | 14.70x  | -93.20%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.251923 | 0.054158   | 4.65x   | -78.50%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.251923 | 0.019721   | 12.77x  | -92.17%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.251923 | 0.016916   | 14.89x  | -93.29%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.351344 | 0.116015   | 3.03x   | -66.98%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.351344 | 0.020088   | 17.49x  | -94.28%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.351344 | 0.017020   | 20.64x  | -95.16%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.738773 | 0.205412   | 3.60x   | -72.20%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.738773 | 0.180350   | 4.10x   | -75.59%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.738773 | 0.187929   | 3.93x   | -74.56%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 0.964534 | 0.257673   | 3.74x   | -73.29%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 0.964534 | 0.190000   | 5.08x   | -80.30%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 0.964534 | 0.191700   | 5.03x   | -80.13%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.716608 | 0.266441   | 2.69x   | -62.82%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.716608 | 0.182667   | 3.92x   | -74.51%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.716608 | 0.187016   | 3.83x   | -73.90%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.793332 | 0.245283   | 3.23x   | -69.08%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.793332 | 0.187090   | 4.24x   | -76.42%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.793332 | 0.191668   | 4.14x   | -75.84%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.832748 | 0.256363   | 3.25x   | -69.21%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.832748 | 0.196960   | 4.23x   | -76.35%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.832748 | 0.201894   | 4.12x   | -75.76%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.793472 | 0.342378   | 2.32x   | -56.85%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.793472 | 0.186279   | 4.26x   | -76.52%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.793472 | 0.189607   | 4.18x   | -76.10%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.746644 | 0.269552   | 2.77x   | -63.90%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.746644 | 0.184881   | 4.04x   | -75.24%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.746644 | 0.188047   | 3.97x   | -74.81%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.745222 | 0.248711   | 3.00x   | -66.63%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.745222 | 0.188322   | 3.96x   | -74.73%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.745222 | 0.192838   | 3.86x   | -74.12%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.048269 | 0.400960   | 2.61x   | -61.75%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.048269 | 0.191594   | 5.47x   | -81.72%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.048269 | 0.192235   | 5.45x   | -81.66%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.970269 | 0.240398   | 4.04x   | -75.22%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.970269 | 0.190561   | 5.09x   | -80.36%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.970269 | 0.185094   | 5.24x   | -80.92%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.733179 | 0.229431   | 3.20x   | -68.71%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.733179 | 0.185695   | 3.95x   | -74.67%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.733179 | 0.187500   | 3.91x   | -74.43%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.802187 | 0.305090   | 2.63x   | -61.97%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.802187 | 0.186982   | 4.29x   | -76.69%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.802187 | 0.187516   | 4.28x   | -76.62%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.767531 | 0.293006   | 2.62x   | -61.82%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.767531 | 0.188961   | 4.06x   | -75.38%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.767531 | 0.191634   | 4.01x   | -75.03%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.771867 | 0.251909   | 3.06x   | -67.36%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.771867 | 0.187940   | 4.11x   | -75.65%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.771867 | 0.189320   | 4.08x   | -75.47%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 0.991868 | 0.395076   | 2.51x   | -60.17%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 0.991868 | 0.191182   | 5.19x   | -80.73%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 0.991868 | 0.195503   | 5.07x   | -80.29%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.545829 | 0.184110   | 2.96x   | -66.27%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.545829 | 0.022705   | 24.04x  | -95.84%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.545829 | 0.020741   | 26.32x  | -96.20%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.754019 | 0.223062   | 3.38x   | -70.42%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.754019 | 0.027060   | 27.86x  | -96.41%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.754019 | 0.023208   | 32.49x  | -96.92%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.519680 | 0.256060   | 2.03x   | -50.73%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.519680 | 0.024412   | 21.29x  | -95.30%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.519680 | 0.023305   | 22.30x  | -95.52%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.573348 | 0.218859   | 2.62x   | -61.83%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.573348 | 0.025965   | 22.08x  | -95.47%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.573348 | 0.023521   | 24.38x  | -95.90%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.591681 | 0.227198   | 2.60x   | -61.60%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.591681 | 0.029073   | 20.35x  | -95.09%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.591681 | 0.023519   | 25.16x  | -96.02%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.546267 | 0.268234   | 2.04x   | -50.90%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.546267 | 0.032603   | 16.76x  | -94.03%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.546267 | 0.024406   | 22.38x  | -95.53%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.538300 | 0.257761   | 2.09x   | -52.12%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.538300 | 0.024503   | 21.97x  | -95.45%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.538300 | 0.023051   | 23.35x  | -95.72%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.551348 | 0.223723   | 2.46x   | -59.42%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.551348 | 0.024864   | 22.17x  | -95.49%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.551348 | 0.023403   | 23.56x  | -95.76%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.861331 | 0.356705   | 2.41x   | -58.59%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.861331 | 0.029618   | 29.08x  | -96.56%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.861331 | 0.023687   | 36.36x  | -97.25%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.770052 | 0.221747   | 3.47x   | -71.20%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.770052 | 0.024813   | 31.03x  | -96.78%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.770052 | 0.023015   | 33.46x  | -97.01%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.544853 | 0.224076   | 2.43x   | -58.87%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.544853 | 0.032645   | 16.69x  | -94.01%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.544853 | 0.024362   | 22.37x  | -95.53%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.564991 | 0.258003   | 2.19x   | -54.33%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.564991 | 0.026390   | 21.41x  | -95.33%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.564991 | 0.023875   | 23.66x  | -95.77%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.616076 | 0.258988   | 2.38x   | -57.96%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.616076 | 0.026022   | 23.68x  | -95.78%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.616076 | 0.023905   | 25.77x  | -96.12%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.606872 | 0.222622   | 2.73x   | -63.32%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.606872 | 0.026407   | 22.98x  | -95.65%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.606872 | 0.023456   | 25.87x  | -96.14%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.815322 | 0.359010   | 2.27x   | -55.97%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.815322 | 0.028271   | 28.84x  | -96.53%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.815322 | 0.023866   | 34.16x  | -97.07%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.656187 | 0.075353   | 8.71x   | -88.52%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.656187 | 0.036245   | 18.10x  | -94.48%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.656187 | 0.029413   | 22.31x  | -95.52%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.854423 | 0.099157   | 8.62x   | -88.39%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.854423 | 0.041000   | 20.84x  | -95.20%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.854423 | 0.036734   | 23.26x  | -95.70%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.626930 | 0.119740   | 5.24x   | -80.90%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.626930 | 0.038848   | 16.14x  | -93.80%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.626930 | 0.032827   | 19.10x  | -94.76%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.652922 | 0.093457   | 6.99x   | -85.69%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.652922 | 0.038030   | 17.17x  | -94.18%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.652922 | 0.033324   | 19.59x  | -94.90%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.659527 | 0.101666   | 6.49x   | -84.59%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.659527 | 0.041129   | 16.04x  | -93.76%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.659527 | 0.033726   | 19.56x  | -94.89%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.633578 | 0.218664   | 2.90x   | -65.49%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.633578 | 0.039127   | 16.19x  | -93.82%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.633578 | 0.035043   | 18.08x  | -94.47%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.621741 | 0.120164   | 5.17x   | -80.67%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.621741 | 0.038080   | 16.33x  | -93.88%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.621741 | 0.033866   | 18.36x  | -94.55%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.638187 | 0.092972   | 6.86x   | -85.43%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.638187 | 0.036487   | 17.49x  | -94.28%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.638187 | 0.032469   | 19.66x  | -94.91%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.960527 | 0.233499   | 4.11x   | -75.69%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.960527 | 0.042043   | 22.85x  | -95.62%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.960527 | 0.034720   | 27.66x  | -96.39%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.850302 | 0.093193   | 9.12x   | -89.04%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.850302 | 0.037534   | 22.65x  | -95.59%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.850302 | 0.032950   | 25.81x  | -96.12%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.650048 | 0.104967   | 6.19x   | -83.85%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.650048 | 0.039563   | 16.43x  | -93.91%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.650048 | 0.033391   | 19.47x  | -94.86%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.664236 | 0.144314   | 4.60x   | -78.27%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.664236 | 0.039454   | 16.84x  | -94.06%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.664236 | 0.034871   | 19.05x  | -94.75%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.651603 | 0.143768   | 4.53x   | -77.94%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.651603 | 0.039439   | 16.52x  | -93.95%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.651603 | 0.034501   | 18.89x  | -94.71%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.664447 | 0.099392   | 6.69x   | -85.04%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.664447 | 0.039792   | 16.70x  | -94.01%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.664447 | 0.034264   | 19.39x  | -94.84%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.880130 | 0.243370   | 3.62x   | -72.35%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.880130 | 0.039884   | 22.07x  | -95.47%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.880130 | 0.033163   | 26.54x  | -96.23%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.497074 | 0.089122   | 5.58x   | -82.07%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.497074 | 0.035832   | 13.87x  | -92.79%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.497074 | 0.033331   | 14.91x  | -93.29%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.708048 | 0.115933   | 6.11x   | -83.63%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.708048 | 0.042883   | 16.51x  | -93.94%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.708048 | 0.036875   | 19.20x  | -94.79%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.501809 | 0.143918   | 3.49x   | -71.32%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.501809 | 0.041987   | 11.95x  | -91.63%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.501809 | 0.036816   | 13.63x  | -92.66%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.554163 | 0.118471   | 4.68x   | -78.62%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.554163 | 0.042583   | 13.01x  | -92.32%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.554163 | 0.037730   | 14.69x  | -93.19%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.554750 | 0.121384   | 4.57x   | -78.12%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.554750 | 0.045914   | 12.08x  | -91.72%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.554750 | 0.039212   | 14.15x  | -92.93%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.538423 | 0.206568   | 2.61x   | -61.63%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.538423 | 0.041433   | 12.99x  | -92.30%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.538423 | 0.038267   | 14.07x  | -92.89%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.506566 | 0.141834   | 3.57x   | -72.00%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.506566 | 0.039852   | 12.71x  | -92.13%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.506566 | 0.036959   | 13.71x  | -92.70%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.515522 | 0.112691   | 4.57x   | -78.14%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.515522 | 0.039801   | 12.95x  | -92.28%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.515522 | 0.038788   | 13.29x  | -92.48%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.806332 | 0.247517   | 3.26x   | -69.30%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.806332 | 0.042818   | 18.83x  | -94.69%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.806332 | 0.036947   | 21.82x  | -95.42%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.717962 | 0.111584   | 6.43x   | -84.46%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.717962 | 0.038853   | 18.48x  | -94.59%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.717962 | 0.035801   | 20.05x  | -95.01%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.496502 | 0.132935   | 3.73x   | -73.23%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.496502 | 0.041624   | 11.93x  | -91.62%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.496502 | 0.037125   | 13.37x  | -92.52%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.517440 | 0.156214   | 3.31x   | -69.81%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.517440 | 0.042348   | 12.22x  | -91.82%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.517440 | 0.036854   | 14.04x  | -92.88%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.517705 | 0.158814   | 3.26x   | -69.32%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.517705 | 0.042122   | 12.29x  | -91.86%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.517705 | 0.035630   | 14.53x  | -93.12%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.530333 | 0.118280   | 4.48x   | -77.70%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.530333 | 0.041515   | 12.77x  | -92.17%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.530333 | 0.036431   | 14.56x  | -93.13%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.749893 | 0.256803   | 2.92x   | -65.75%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.749893 | 0.042719   | 17.55x  | -94.30%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.749893 | 0.036923   | 20.31x  | -95.08%        |
</details>
<!-- PERF_RESULTS_END -->
