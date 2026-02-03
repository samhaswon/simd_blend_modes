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
| normal        | scalar | 0.187794 | 0.040472   | 4.64x   | -78.45%        |
| normal        | sse42  | 0.187794 | 0.019107   | 9.83x   | -89.83%        |
| normal        | avx2   | 0.187794 | 0.018936   | 9.92x   | -89.92%        |
| soft_light    | scalar | 0.252953 | 0.047342   | 5.34x   | -81.28%        |
| soft_light    | sse42  | 0.252953 | 0.021950   | 11.52x  | -91.32%        |
| soft_light    | avx2   | 0.252953 | 0.021939   | 11.53x  | -91.33%        |
| lighten_only  | scalar | 0.186238 | 0.050894   | 3.66x   | -72.67%        |
| lighten_only  | sse42  | 0.186238 | 0.020045   | 9.29x   | -89.24%        |
| lighten_only  | avx2   | 0.186238 | 0.019405   | 9.60x   | -89.58%        |
| screen        | scalar | 0.196732 | 0.045528   | 4.32x   | -76.86%        |
| screen        | sse42  | 0.196732 | 0.021539   | 9.13x   | -89.05%        |
| screen        | avx2   | 0.196732 | 0.022045   | 8.92x   | -88.79%        |
| dodge         | scalar | 0.198893 | 0.048265   | 4.12x   | -75.73%        |
| dodge         | sse42  | 0.198893 | 0.022607   | 8.80x   | -88.63%        |
| dodge         | avx2   | 0.198893 | 0.022186   | 8.96x   | -88.85%        |
| addition      | scalar | 0.190430 | 0.073369   | 2.60x   | -61.47%        |
| addition      | sse42  | 0.190430 | 0.020968   | 9.08x   | -88.99%        |
| addition      | avx2   | 0.190430 | 0.019600   | 9.72x   | -89.71%        |
| darken_only   | scalar | 0.187890 | 0.051229   | 3.67x   | -72.73%        |
| darken_only   | sse42  | 0.187890 | 0.019978   | 9.40x   | -89.37%        |
| darken_only   | avx2   | 0.187890 | 0.019331   | 9.72x   | -89.71%        |
| multiply      | scalar | 0.192403 | 0.044681   | 4.31x   | -76.78%        |
| multiply      | sse42  | 0.192403 | 0.021097   | 9.12x   | -89.04%        |
| multiply      | avx2   | 0.192403 | 0.021855   | 8.80x   | -88.64%        |
| hard_light    | scalar | 0.284409 | 0.090488   | 3.14x   | -68.18%        |
| hard_light    | sse42  | 0.284409 | 0.022683   | 12.54x  | -92.02%        |
| hard_light    | avx2   | 0.284409 | 0.022082   | 12.88x  | -92.24%        |
| difference    | scalar | 0.257683 | 0.044553   | 5.78x   | -82.71%        |
| difference    | sse42  | 0.257683 | 0.019904   | 12.95x  | -92.28%        |
| difference    | avx2   | 0.257683 | 0.019172   | 13.44x  | -92.56%        |
| subtract      | scalar | 0.190798 | 0.047022   | 4.06x   | -75.35%        |
| subtract      | sse42  | 0.190798 | 0.022377   | 8.53x   | -88.27%        |
| subtract      | avx2   | 0.190798 | 0.022114   | 8.63x   | -88.41%        |
| grain_extract | scalar | 0.196758 | 0.060136   | 3.27x   | -69.44%        |
| grain_extract | sse42  | 0.196758 | 0.021596   | 9.11x   | -89.02%        |
| grain_extract | avx2   | 0.196758 | 0.022049   | 8.92x   | -88.79%        |
| grain_merge   | scalar | 0.196584 | 0.060973   | 3.22x   | -68.98%        |
| grain_merge   | sse42  | 0.196584 | 0.021619   | 9.09x   | -89.00%        |
| grain_merge   | avx2   | 0.196584 | 0.021819   | 9.01x   | -88.90%        |
| divide        | scalar | 0.200758 | 0.046864   | 4.28x   | -76.66%        |
| divide        | sse42  | 0.200758 | 0.021977   | 9.14x   | -89.05%        |
| divide        | avx2   | 0.200758 | 0.021874   | 9.18x   | -89.10%        |
| overlay       | scalar | 0.261447 | 0.086409   | 3.03x   | -66.95%        |
| overlay       | sse42  | 0.261447 | 0.022162   | 11.80x  | -91.52%        |
| overlay       | avx2   | 0.261447 | 0.022206   | 11.77x  | -91.51%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.006193 | 0.001647   | 3.76x   | -73.41%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.006193 | 0.001355   | 4.57x   | -78.11%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.006193 | 0.001297   | 4.78x   | -79.06%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.007986 | 0.001759   | 4.54x   | -77.97%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.007986 | 0.001476   | 5.41x   | -81.52%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.007986 | 0.001540   | 5.19x   | -80.72%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.006665 | 0.001853   | 3.60x   | -72.20%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.006665 | 0.001305   | 5.11x   | -80.42%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.006665 | 0.001261   | 5.28x   | -81.07%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.007073 | 0.001694   | 4.17x   | -76.05%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.007073 | 0.001448   | 4.88x   | -79.52%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.007073 | 0.001527   | 4.63x   | -78.41%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.007089 | 0.001772   | 4.00x   | -75.00%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.007089 | 0.001485   | 4.77x   | -79.05%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.007089 | 0.001567   | 4.52x   | -77.90%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.007208 | 0.002570   | 2.80x   | -64.34%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.007208 | 0.001428   | 5.05x   | -80.19%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.007208 | 0.001336   | 5.39x   | -81.46%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.006843 | 0.001853   | 3.69x   | -72.92%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.006843 | 0.001378   | 4.97x   | -79.86%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.006843 | 0.001273   | 5.38x   | -81.40%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.007380 | 0.001697   | 4.35x   | -77.01%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.007380 | 0.001437   | 5.14x   | -80.53%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.007380 | 0.001531   | 4.82x   | -79.26%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.008642 | 0.002902   | 2.98x   | -66.42%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.008642 | 0.001476   | 5.85x   | -82.92%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.008642 | 0.001536   | 5.63x   | -82.23%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.008584 | 0.001684   | 5.10x   | -80.38%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.008584 | 0.001317   | 6.52x   | -84.66%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.008584 | 0.001259   | 6.82x   | -85.33%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.006835 | 0.001565   | 4.37x   | -77.10%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.006835 | 0.001449   | 4.72x   | -78.80%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.006835 | 0.001522   | 4.49x   | -77.73%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.006774 | 0.002068   | 3.28x   | -69.48%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.006774 | 0.001448   | 4.68x   | -78.63%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.006774 | 0.001529   | 4.43x   | -77.42%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.007192 | 0.002154   | 3.34x   | -70.04%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.007192 | 0.001563   | 4.60x   | -78.26%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.007192 | 0.001587   | 4.53x   | -77.93%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.007295 | 0.001784   | 4.09x   | -75.55%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.007295 | 0.001516   | 4.81x   | -79.22%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.007295 | 0.001566   | 4.66x   | -78.53%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.008326 | 0.002843   | 2.93x   | -65.86%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.008326 | 0.001516   | 5.49x   | -81.79%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.008326 | 0.001541   | 5.40x   | -81.50%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.002981 | 0.001346   | 2.21x   | -54.84%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.002981 | 0.000150   | 19.82x  | -94.95%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.002981 | 0.000186   | 15.99x  | -93.75%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.006724 | 0.001675   | 4.01x   | -75.09%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.006724 | 0.000221   | 30.38x  | -96.71%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.006724 | 0.000207   | 32.45x  | -96.92%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.005298 | 0.001758   | 3.01x   | -66.82%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.005298 | 0.000218   | 24.26x  | -95.88%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.005298 | 0.000196   | 27.10x  | -96.31%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.005351 | 0.001535   | 3.48x   | -71.30%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.005351 | 0.000209   | 25.66x  | -96.10%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.005351 | 0.000207   | 25.90x  | -96.14%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.005395 | 0.001618   | 3.34x   | -70.02%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.005395 | 0.000237   | 22.81x  | -95.62%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.005395 | 0.000207   | 26.11x  | -96.17%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.005856 | 0.001965   | 2.98x   | -66.44%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.005856 | 0.000251   | 23.35x  | -95.72%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.005856 | 0.000218   | 26.90x  | -96.28%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.005352 | 0.001699   | 3.15x   | -68.25%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.005352 | 0.000190   | 28.15x  | -96.45%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.005352 | 0.000199   | 26.96x  | -96.29%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.005438 | 0.001537   | 3.54x   | -71.74%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.005438 | 0.000198   | 27.42x  | -96.35%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.005438 | 0.000207   | 26.32x  | -96.20%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.007129 | 0.002726   | 2.62x   | -61.77%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.007129 | 0.000238   | 29.99x  | -96.67%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.007129 | 0.000206   | 34.60x  | -97.11%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.007122 | 0.001546   | 4.61x   | -78.29%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.007122 | 0.000191   | 37.36x  | -97.32%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.007122 | 0.000208   | 34.32x  | -97.09%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.005808 | 0.001490   | 3.90x   | -74.35%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.005808 | 0.000263   | 22.06x  | -95.47%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.005808 | 0.000221   | 26.25x  | -96.19%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.005374 | 0.001864   | 2.88x   | -65.32%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.005374 | 0.000212   | 25.34x  | -96.05%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.005374 | 0.000206   | 26.03x  | -96.16%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.005418 | 0.001891   | 2.86x   | -65.09%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.005418 | 0.000211   | 25.72x  | -96.11%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.005418 | 0.000207   | 26.23x  | -96.19%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.005393 | 0.001602   | 3.37x   | -70.30%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.005393 | 0.000219   | 24.58x  | -95.93%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.005393 | 0.000211   | 25.56x  | -96.09%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.007227 | 0.002524   | 2.86x   | -65.08%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.007227 | 0.000226   | 31.99x  | -96.87%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.007227 | 0.000211   | 34.23x  | -97.08%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.005870 | 0.000506   | 11.60x  | -91.38%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.005870 | 0.000143   | 41.19x  | -97.57%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.005870 | 0.000146   | 40.17x  | -97.51%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.008666 | 0.000615   | 14.10x  | -92.91%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.008666 | 0.000229   | 37.80x  | -97.35%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.008666 | 0.000184   | 47.04x  | -97.87%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.006928 | 0.000732   | 9.46x   | -89.43%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.006928 | 0.000211   | 32.83x  | -96.95%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.006928 | 0.000181   | 38.37x  | -97.39%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.006962 | 0.000560   | 12.43x  | -91.96%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.006962 | 0.000223   | 31.22x  | -96.80%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.006962 | 0.000186   | 37.33x  | -97.32%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.007055 | 0.000629   | 11.22x  | -91.08%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.007055 | 0.000243   | 28.99x  | -96.55%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.007055 | 0.000184   | 38.32x  | -97.39%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.007019 | 0.001546   | 4.54x   | -77.98%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.007019 | 0.000231   | 30.41x  | -96.71%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.007019 | 0.000190   | 36.87x  | -97.29%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.006793 | 0.000729   | 9.32x   | -89.28%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.006793 | 0.000212   | 32.07x  | -96.88%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.006793 | 0.000176   | 38.67x  | -97.41%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.006888 | 0.000539   | 12.77x  | -92.17%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.006888 | 0.000208   | 33.08x  | -96.98%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.006888 | 0.000177   | 38.87x  | -97.43%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.008437 | 0.001791   | 4.71x   | -78.77%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.008437 | 0.000239   | 35.33x  | -97.17%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.008437 | 0.000185   | 45.67x  | -97.81%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.008622 | 0.000546   | 15.79x  | -93.67%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.008622 | 0.000211   | 40.80x  | -97.55%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.008622 | 0.000175   | 49.15x  | -97.97%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.006933 | 0.000683   | 10.15x  | -90.15%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.006933 | 0.000233   | 29.73x  | -96.64%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.006933 | 0.000200   | 34.71x  | -97.12%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.006783 | 0.000994   | 6.82x   | -85.34%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.006783 | 0.000225   | 30.12x  | -96.68%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.006783 | 0.000182   | 37.33x  | -97.32%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.007369 | 0.001074   | 6.86x   | -85.43%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.007369 | 0.000229   | 32.12x  | -96.89%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.007369 | 0.000187   | 39.33x  | -97.46%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.007587 | 0.000615   | 12.34x  | -91.89%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.007587 | 0.000232   | 32.66x  | -96.94%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.007587 | 0.000206   | 36.81x  | -97.28%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.008646 | 0.001669   | 5.18x   | -80.70%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.008646 | 0.000238   | 36.27x  | -97.24%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.008646 | 0.000180   | 48.02x  | -97.92%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.004233 | 0.000615   | 6.89x   | -85.48%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.004233 | 0.000132   | 32.09x  | -96.88%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.004233 | 0.000181   | 23.40x  | -95.73%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.006585 | 0.000720   | 9.15x   | -89.07%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.006585 | 0.000182   | 36.09x  | -97.23%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.006585 | 0.000181   | 36.40x  | -97.25%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.005245 | 0.000766   | 6.85x   | -85.40%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.005245 | 0.000171   | 30.64x  | -96.74%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.005245 | 0.000180   | 29.14x  | -96.57%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.005250 | 0.000675   | 7.77x   | -87.13%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.005250 | 0.000171   | 30.65x  | -96.74%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.005250 | 0.000177   | 29.70x  | -96.63%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.005371 | 0.000751   | 7.16x   | -86.03%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.005371 | 0.000221   | 24.25x  | -95.88%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.005371 | 0.000188   | 28.64x  | -96.51%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.005543 | 0.001354   | 4.09x   | -75.57%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.005543 | 0.000196   | 28.26x  | -96.46%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.005543 | 0.000185   | 29.98x  | -96.66%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.005217 | 0.000760   | 6.86x   | -85.42%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.005217 | 0.000167   | 31.28x  | -96.80%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.005217 | 0.000177   | 29.40x  | -96.60%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.005186 | 0.000642   | 8.08x   | -87.62%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.005186 | 0.000163   | 31.77x  | -96.85%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.005186 | 0.000173   | 29.90x  | -96.65%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.006979 | 0.001856   | 3.76x   | -73.41%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.006979 | 0.000239   | 29.20x  | -96.57%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.006979 | 0.000183   | 38.19x  | -97.38%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.006993 | 0.000655   | 10.68x  | -90.63%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.006993 | 0.000170   | 41.06x  | -97.56%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.006993 | 0.000175   | 39.99x  | -97.50%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.005524 | 0.000883   | 6.26x   | -84.02%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.005524 | 0.000201   | 27.51x  | -96.36%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.005524 | 0.000189   | 29.18x  | -96.57%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.005195 | 0.001057   | 4.92x   | -79.66%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.005195 | 0.000182   | 28.52x  | -96.49%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.005195 | 0.000181   | 28.69x  | -96.51%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.005565 | 0.001076   | 5.17x   | -80.67%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.005565 | 0.000178   | 31.26x  | -96.80%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.005565 | 0.000173   | 32.22x  | -96.90%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.005435 | 0.000717   | 7.58x   | -86.80%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.005435 | 0.000188   | 28.98x  | -96.55%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.005435 | 0.000178   | 30.46x  | -96.72%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.006732 | 0.001728   | 3.90x   | -74.33%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.006732 | 0.000196   | 34.28x  | -97.08%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.006732 | 0.000174   | 38.77x  | -97.42%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.032795 | 0.006325   | 5.19x   | -80.71%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.032795 | 0.005476   | 5.99x   | -83.30%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.032795 | 0.005176   | 6.34x   | -84.22%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.042898 | 0.007098   | 6.04x   | -83.45%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.042898 | 0.005893   | 7.28x   | -86.26%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.042898 | 0.006143   | 6.98x   | -85.68%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.036514 | 0.007532   | 4.85x   | -79.37%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.036514 | 0.005525   | 6.61x   | -84.87%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.036514 | 0.005151   | 7.09x   | -85.89%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.037621 | 0.006899   | 5.45x   | -81.66%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.037621 | 0.005791   | 6.50x   | -84.61%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.037621 | 0.006139   | 6.13x   | -83.68%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.036947 | 0.007135   | 5.18x   | -80.69%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.036947 | 0.005949   | 6.21x   | -83.90%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.036947 | 0.006216   | 5.94x   | -83.18%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.036918 | 0.009919   | 3.72x   | -73.13%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.036918 | 0.005352   | 6.90x   | -85.50%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.036918 | 0.005068   | 7.28x   | -86.27%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.036100 | 0.007816   | 4.62x   | -78.35%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.036100 | 0.005269   | 6.85x   | -85.40%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.036100 | 0.005122   | 7.05x   | -85.81%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.036936 | 0.006755   | 5.47x   | -81.71%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.036936 | 0.005724   | 6.45x   | -84.50%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.036936 | 0.006140   | 6.02x   | -83.38%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.044595 | 0.011664   | 3.82x   | -73.85%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.044595 | 0.005935   | 7.51x   | -86.69%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.044595 | 0.006144   | 7.26x   | -86.22%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.043484 | 0.006728   | 6.46x   | -84.53%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.043484 | 0.005319   | 8.17x   | -87.77%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.043484 | 0.005049   | 8.61x   | -88.39%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.036036 | 0.006343   | 5.68x   | -82.40%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.036036 | 0.005769   | 6.25x   | -83.99%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.036036 | 0.006099   | 5.91x   | -83.08%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.037093 | 0.008255   | 4.49x   | -77.74%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.037093 | 0.005841   | 6.35x   | -84.25%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.037093 | 0.006149   | 6.03x   | -83.42%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.037925 | 0.008359   | 4.54x   | -77.96%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.037925 | 0.005828   | 6.51x   | -84.63%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.037925 | 0.006139   | 6.18x   | -83.81%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.037352 | 0.006916   | 5.40x   | -81.48%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.037352 | 0.005932   | 6.30x   | -84.12%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.037352 | 0.006175   | 6.05x   | -83.47%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.043774 | 0.011232   | 3.90x   | -74.34%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.043774 | 0.005837   | 7.50x   | -86.67%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.043774 | 0.006162   | 7.10x   | -85.92%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.023884 | 0.005193   | 4.60x   | -78.26%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.023884 | 0.000594   | 40.19x  | -97.51%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.023884 | 0.000740   | 32.27x  | -96.90%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.035080 | 0.006438   | 5.45x   | -81.65%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.035080 | 0.000878   | 39.95x  | -97.50%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.035080 | 0.000823   | 42.60x  | -97.65%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.028113 | 0.006763   | 4.16x   | -75.94%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.028113 | 0.000751   | 37.44x  | -97.33%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.028113 | 0.000784   | 35.86x  | -97.21%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.028992 | 0.006126   | 4.73x   | -78.87%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.028992 | 0.000826   | 35.08x  | -97.15%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.028992 | 0.000823   | 35.24x  | -97.16%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.028643 | 0.006557   | 4.37x   | -77.11%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.028643 | 0.000941   | 30.43x  | -96.71%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.028643 | 0.000830   | 34.51x  | -97.10%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.028251 | 0.008009   | 3.53x   | -71.65%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.028251 | 0.000994   | 28.42x  | -96.48%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.028251 | 0.000853   | 33.12x  | -96.98%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.028080 | 0.006821   | 4.12x   | -75.71%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.028080 | 0.000759   | 37.00x  | -97.30%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.028080 | 0.000793   | 35.39x  | -97.17%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.028602 | 0.006188   | 4.62x   | -78.37%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.028602 | 0.000789   | 36.23x  | -97.24%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.028602 | 0.000822   | 34.81x  | -97.13%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.036867 | 0.010279   | 3.59x   | -72.12%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.036867 | 0.000943   | 39.09x  | -97.44%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.036867 | 0.000822   | 44.87x  | -97.77%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.035385 | 0.006241   | 5.67x   | -82.36%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.035385 | 0.000772   | 45.81x  | -97.82%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.035385 | 0.000809   | 43.76x  | -97.71%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.028064 | 0.006011   | 4.67x   | -78.58%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.028064 | 0.001059   | 26.51x  | -96.23%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.028064 | 0.000861   | 32.59x  | -96.93%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.028350 | 0.007408   | 3.83x   | -73.87%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.028350 | 0.000835   | 33.94x  | -97.05%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.028350 | 0.000823   | 34.44x  | -97.10%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.028713 | 0.007550   | 3.80x   | -73.71%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.028713 | 0.000858   | 33.48x  | -97.01%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.028713 | 0.000819   | 35.06x  | -97.15%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.028983 | 0.006263   | 4.63x   | -78.39%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.028983 | 0.000858   | 33.77x  | -97.04%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.028983 | 0.000820   | 35.35x  | -97.17%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.035705 | 0.010373   | 3.44x   | -70.95%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.035705 | 0.000916   | 39.00x  | -97.44%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.035705 | 0.000836   | 42.72x  | -97.66%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.027929 | 0.002357   | 11.85x  | -91.56%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.027929 | 0.000615   | 45.42x  | -97.80%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.027929 | 0.000631   | 44.27x  | -97.74%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.038600 | 0.002782   | 13.87x  | -92.79%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.038600 | 0.000926   | 41.70x  | -97.60%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.038600 | 0.000732   | 52.74x  | -98.10%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.031958 | 0.003282   | 9.74x   | -89.73%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.031958 | 0.000869   | 36.79x  | -97.28%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.031958 | 0.000747   | 42.79x  | -97.66%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.032533 | 0.002739   | 11.88x  | -91.58%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.032533 | 0.000903   | 36.03x  | -97.22%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.032533 | 0.000758   | 42.90x  | -97.67%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.032301 | 0.002916   | 11.08x  | -90.97%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.032301 | 0.001016   | 31.78x  | -96.85%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.032301 | 0.000774   | 41.71x  | -97.60%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.031843 | 0.006563   | 4.85x   | -79.39%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.031843 | 0.000952   | 33.45x  | -97.01%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.031843 | 0.000790   | 40.31x  | -97.52%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.032322 | 0.003325   | 9.72x   | -89.71%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.032322 | 0.000902   | 35.82x  | -97.21%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.032322 | 0.000741   | 43.64x  | -97.71%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.033785 | 0.002546   | 13.27x  | -92.46%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.033785 | 0.000881   | 38.35x  | -97.39%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.033785 | 0.000749   | 45.08x  | -97.78%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.040148 | 0.007397   | 5.43x   | -81.57%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.040148 | 0.000986   | 40.74x  | -97.55%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.040148 | 0.000750   | 53.55x  | -98.13%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.038807 | 0.002540   | 15.28x  | -93.46%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.038807 | 0.000898   | 43.19x  | -97.68%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.038807 | 0.000750   | 51.75x  | -98.07%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.031770 | 0.003094   | 10.27x  | -90.26%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.031770 | 0.000970   | 32.74x  | -96.95%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.031770 | 0.000764   | 41.57x  | -97.59%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.032302 | 0.004864   | 6.64x   | -84.94%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.032302 | 0.000971   | 33.26x  | -96.99%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.032302 | 0.000807   | 40.03x  | -97.50%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.033901 | 0.004398   | 7.71x   | -87.03%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.033901 | 0.000907   | 37.39x  | -97.33%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.033901 | 0.000742   | 45.72x  | -97.81%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.032278 | 0.002820   | 11.45x  | -91.26%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.032278 | 0.000938   | 34.41x  | -97.09%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.032278 | 0.000746   | 43.24x  | -97.69%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.038826 | 0.007017   | 5.53x   | -81.93%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.038826 | 0.000938   | 41.41x  | -97.58%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.038826 | 0.000734   | 52.88x  | -98.11%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.019884 | 0.002455   | 8.10x   | -87.65%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.019884 | 0.000671   | 29.62x  | -96.62%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.019884 | 0.000744   | 26.73x  | -96.26%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.029825 | 0.002851   | 10.46x  | -90.44%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.029825 | 0.000751   | 39.69x  | -97.48%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.029825 | 0.000769   | 38.76x  | -97.42%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.022978 | 0.003128   | 7.35x   | -86.39%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.022978 | 0.000717   | 32.04x  | -96.88%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.022978 | 0.000766   | 30.01x  | -96.67%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.023974 | 0.002673   | 8.97x   | -88.85%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.023974 | 0.000766   | 31.30x  | -96.80%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.023974 | 0.000785   | 30.53x  | -96.72%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.024946 | 0.003198   | 7.80x   | -87.18%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.024946 | 0.000991   | 25.18x  | -96.03%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.024946 | 0.000867   | 28.78x  | -96.53%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.024691 | 0.005476   | 4.51x   | -77.82%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.024691 | 0.000794   | 31.10x  | -96.78%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.024691 | 0.000812   | 30.39x  | -96.71%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.024152 | 0.003198   | 7.55x   | -86.76%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.024152 | 0.000791   | 30.52x  | -96.72%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.024152 | 0.000795   | 30.36x  | -96.71%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.024168 | 0.002554   | 9.46x   | -89.43%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.024168 | 0.000734   | 32.91x  | -96.96%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.024168 | 0.000774   | 31.24x  | -96.80%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.031841 | 0.007400   | 4.30x   | -76.76%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.031841 | 0.000940   | 33.89x  | -97.05%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.031841 | 0.000782   | 40.69x  | -97.54%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.031150 | 0.002639   | 11.80x  | -91.53%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.031150 | 0.000781   | 39.89x  | -97.49%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.031150 | 0.000778   | 40.06x  | -97.50%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.024041 | 0.003492   | 6.88x   | -85.47%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.024041 | 0.000822   | 29.23x  | -96.58%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.024041 | 0.000783   | 30.71x  | -96.74%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.023727 | 0.004223   | 5.62x   | -82.20%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.023727 | 0.000735   | 32.27x  | -96.90%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.023727 | 0.000749   | 31.70x  | -96.85%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.023865 | 0.004217   | 5.66x   | -82.33%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.023865 | 0.000783   | 30.47x  | -96.72%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.023865 | 0.000764   | 31.24x  | -96.80%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.024303 | 0.002863   | 8.49x   | -88.22%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.024303 | 0.000789   | 30.81x  | -96.75%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.024303 | 0.000771   | 31.52x  | -96.83%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.031394 | 0.007017   | 4.47x   | -77.65%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.031394 | 0.000835   | 37.61x  | -97.34%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.031394 | 0.000784   | 40.06x  | -97.50%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.095016 | 0.025470   | 3.73x   | -73.19%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.095016 | 0.021846   | 4.35x   | -77.01%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.095016 | 0.020781   | 4.57x   | -78.13%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.127934 | 0.029184   | 4.38x   | -77.19%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.127934 | 0.024609   | 5.20x   | -80.76%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.127934 | 0.025027   | 5.11x   | -80.44%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.100674 | 0.029728   | 3.39x   | -70.47%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.100674 | 0.021058   | 4.78x   | -79.08%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.100674 | 0.020236   | 4.97x   | -79.90%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.103023 | 0.027051   | 3.81x   | -73.74%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.103023 | 0.023282   | 4.42x   | -77.40%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.103023 | 0.025184   | 4.09x   | -75.56%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.103673 | 0.028361   | 3.66x   | -72.64%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.103673 | 0.023862   | 4.34x   | -76.98%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.103673 | 0.024712   | 4.20x   | -76.16%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.102412 | 0.039758   | 2.58x   | -61.18%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.102412 | 0.021228   | 4.82x   | -79.27%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.102412 | 0.020168   | 5.08x   | -80.31%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.100110 | 0.029485   | 3.40x   | -70.55%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.100110 | 0.021077   | 4.75x   | -78.95%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.100110 | 0.020358   | 4.92x   | -79.66%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.102310 | 0.027164   | 3.77x   | -73.45%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.102310 | 0.023181   | 4.41x   | -77.34%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.102310 | 0.024431   | 4.19x   | -76.12%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.136460 | 0.046584   | 2.93x   | -65.86%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.136460 | 0.023968   | 5.69x   | -82.44%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.136460 | 0.025088   | 5.44x   | -81.61%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.129417 | 0.026910   | 4.81x   | -79.21%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.129417 | 0.021028   | 6.15x   | -83.75%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.129417 | 0.020188   | 6.41x   | -84.40%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.098898 | 0.025352   | 3.90x   | -74.37%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.098898 | 0.023499   | 4.21x   | -76.24%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.098898 | 0.024707   | 4.00x   | -75.02%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.103073 | 0.033377   | 3.09x   | -67.62%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.103073 | 0.023470   | 4.39x   | -77.23%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.103073 | 0.024506   | 4.21x   | -76.22%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.102364 | 0.034160   | 3.00x   | -66.63%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.102364 | 0.023416   | 4.37x   | -77.13%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.102364 | 0.024494   | 4.18x   | -76.07%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.103340 | 0.029306   | 3.53x   | -71.64%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.103340 | 0.024090   | 4.29x   | -76.69%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.103340 | 0.024790   | 4.17x   | -76.01%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.128287 | 0.044835   | 2.86x   | -65.05%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.128287 | 0.023501   | 5.46x   | -81.68%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.128287 | 0.024662   | 5.20x   | -80.78%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.069048 | 0.020735   | 3.33x   | -69.97%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.069048 | 0.002456   | 28.11x  | -96.44%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.069048 | 0.002931   | 23.56x  | -95.76%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.100656 | 0.025467   | 3.95x   | -74.70%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.100656 | 0.003512   | 28.66x  | -96.51%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.100656 | 0.003294   | 30.55x  | -96.73%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.074851 | 0.027002   | 2.77x   | -63.93%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.074851 | 0.002993   | 25.01x  | -96.00%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.074851 | 0.003155   | 23.72x  | -95.78%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.077879 | 0.025453   | 3.06x   | -67.32%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.077879 | 0.003343   | 23.29x  | -95.71%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.077879 | 0.003353   | 23.23x  | -95.69%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.078807 | 0.026149   | 3.01x   | -66.82%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.078807 | 0.003707   | 21.26x  | -95.30%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.078807 | 0.003300   | 23.88x  | -95.81%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.077237 | 0.031585   | 2.45x   | -59.11%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.077237 | 0.003953   | 19.54x  | -94.88%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.077237 | 0.003441   | 22.45x  | -95.55%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.075398 | 0.027862   | 2.71x   | -63.05%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.075398 | 0.003063   | 24.61x  | -95.94%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.075398 | 0.003202   | 23.55x  | -95.75%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.078418 | 0.024452   | 3.21x   | -68.82%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.078418 | 0.003222   | 24.34x  | -95.89%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.078418 | 0.003315   | 23.65x  | -95.77%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.110247 | 0.041168   | 2.68x   | -62.66%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.110247 | 0.003830   | 28.78x  | -96.53%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.110247 | 0.003301   | 33.40x  | -97.01%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.107337 | 0.024393   | 4.40x   | -77.27%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.107337 | 0.003057   | 35.11x  | -97.15%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.107337 | 0.003202   | 33.53x  | -97.02%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.076221 | 0.025046   | 3.04x   | -67.14%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.076221 | 0.004293   | 17.75x  | -94.37%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.076221 | 0.003475   | 21.94x  | -95.44%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.077446 | 0.029592   | 2.62x   | -61.79%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.077446 | 0.003336   | 23.21x  | -95.69%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.077446 | 0.003276   | 23.64x  | -95.77%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.076522 | 0.029674   | 2.58x   | -61.22%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.076522 | 0.003386   | 22.60x  | -95.58%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.076522 | 0.003257   | 23.49x  | -95.74%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.077795 | 0.025177   | 3.09x   | -67.64%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.077795 | 0.003435   | 22.65x  | -95.58%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.077795 | 0.003282   | 23.70x  | -95.78%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.106611 | 0.040096   | 2.66x   | -62.39%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.106611 | 0.003674   | 29.02x  | -96.55%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.106611 | 0.003317   | 32.14x  | -96.89%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.083577 | 0.008443   | 9.90x   | -89.90%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.083577 | 0.002684   | 31.13x  | -96.79%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.083577 | 0.002547   | 32.81x  | -96.95%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.113895 | 0.010187   | 11.18x  | -91.06%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.113895 | 0.003725   | 30.58x  | -96.73%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.113895 | 0.002985   | 38.16x  | -97.38%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.088042 | 0.012048   | 7.31x   | -86.32%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.088042 | 0.003494   | 25.20x  | -96.03%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.088042 | 0.002875   | 30.63x  | -96.73%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.092725 | 0.009563   | 9.70x   | -89.69%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.092725 | 0.003739   | 24.80x  | -95.97%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.092725 | 0.003012   | 30.78x  | -96.75%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.090949 | 0.010456   | 8.70x   | -88.50%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.090949 | 0.004036   | 22.54x  | -95.56%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.090949 | 0.003036   | 29.96x  | -96.66%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.088589 | 0.025209   | 3.51x   | -71.54%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.088589 | 0.003815   | 23.22x  | -95.69%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.088589 | 0.003022   | 29.31x  | -96.59%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.088103 | 0.012158   | 7.25x   | -86.20%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.088103 | 0.003562   | 24.74x  | -95.96%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.088103 | 0.002907   | 30.31x  | -96.70%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.089580 | 0.009180   | 9.76x   | -89.75%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.089580 | 0.003521   | 25.44x  | -96.07%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.089580 | 0.002936   | 30.51x  | -96.72%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.123947 | 0.028857   | 4.30x   | -76.72%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.123947 | 0.004059   | 30.54x  | -96.73%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.123947 | 0.003030   | 40.91x  | -97.56%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.117230 | 0.009097   | 12.89x  | -92.24%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.117230 | 0.003544   | 33.08x  | -96.98%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.117230 | 0.002963   | 39.57x  | -97.47%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.088136 | 0.011485   | 7.67x   | -86.97%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.088136 | 0.003916   | 22.51x  | -95.56%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.088136 | 0.003088   | 28.54x  | -96.50%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.091383 | 0.016607   | 5.50x   | -81.83%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.091383 | 0.003777   | 24.19x  | -95.87%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.091383 | 0.003007   | 30.39x  | -96.71%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.090676 | 0.016728   | 5.42x   | -81.55%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.090676 | 0.003625   | 25.01x  | -96.00%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.090676 | 0.002952   | 30.71x  | -96.74%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.091352 | 0.010150   | 9.00x   | -88.89%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.091352 | 0.003819   | 23.92x  | -95.82%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.091352 | 0.002952   | 30.94x  | -96.77%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.116181 | 0.027049   | 4.30x   | -76.72%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.116181 | 0.003785   | 30.70x  | -96.74%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.116181 | 0.002920   | 39.78x  | -97.49%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.062819 | 0.009750   | 6.44x   | -84.48%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.062819 | 0.002624   | 23.94x  | -95.82%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.062819 | 0.002805   | 22.40x  | -95.54%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.094850 | 0.011354   | 8.35x   | -88.03%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.094850 | 0.003009   | 31.52x  | -96.83%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.094850 | 0.002964   | 32.00x  | -96.87%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.069308 | 0.012146   | 5.71x   | -82.48%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.069308 | 0.003104   | 22.33x  | -95.52%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.069308 | 0.003073   | 22.55x  | -95.57%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.072426 | 0.010632   | 6.81x   | -85.32%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.072426 | 0.003059   | 23.68x  | -95.78%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.072426 | 0.002957   | 24.50x  | -95.92%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.071691 | 0.012045   | 5.95x   | -83.20%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.071691 | 0.003561   | 20.13x  | -95.03%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.071691 | 0.003006   | 23.85x  | -95.81%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.069388 | 0.021661   | 3.20x   | -68.78%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.069388 | 0.003279   | 21.16x  | -95.27%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.069388 | 0.003155   | 21.99x  | -95.45%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.069264 | 0.012551   | 5.52x   | -81.88%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.069264 | 0.003095   | 22.38x  | -95.53%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.069264 | 0.003032   | 22.85x  | -95.62%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.070497 | 0.010178   | 6.93x   | -85.56%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.070497 | 0.002880   | 24.48x  | -95.91%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.070497 | 0.003070   | 22.96x  | -95.64%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.103047 | 0.029549   | 3.49x   | -71.32%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.103047 | 0.003628   | 28.40x  | -96.48%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.103047 | 0.003001   | 34.33x  | -97.09%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.099351 | 0.010481   | 9.48x   | -89.45%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.099351 | 0.002916   | 34.07x  | -97.06%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.099351 | 0.002983   | 33.31x  | -97.00%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.073098 | 0.014575   | 5.02x   | -80.06%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.073098 | 0.003551   | 20.59x  | -95.14%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.073098 | 0.003305   | 22.12x  | -95.48%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.074218 | 0.017054   | 4.35x   | -77.02%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.074218 | 0.003171   | 23.40x  | -95.73%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.074218 | 0.002974   | 24.96x  | -95.99%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.071429 | 0.017096   | 4.18x   | -76.07%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.071429 | 0.003201   | 22.31x  | -95.52%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.071429 | 0.003159   | 22.61x  | -95.58%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.074909 | 0.011341   | 6.61x   | -84.86%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.074909 | 0.003024   | 24.77x  | -95.96%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.074909 | 0.002962   | 25.29x  | -96.05%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.098654 | 0.027795   | 3.55x   | -71.83%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.098654 | 0.003171   | 31.11x  | -96.79%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.098654 | 0.003032   | 32.54x  | -96.93%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.410226 | 0.104825   | 3.91x   | -74.45%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.410226 | 0.089916   | 4.56x   | -78.08%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.410226 | 0.083715   | 4.90x   | -79.59%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.488855 | 0.113747   | 4.30x   | -76.73%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.488855 | 0.095188   | 5.14x   | -80.53%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.488855 | 0.099000   | 4.94x   | -79.75%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.399721 | 0.121212   | 3.30x   | -69.68%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.399721 | 0.089516   | 4.47x   | -77.61%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.399721 | 0.086976   | 4.60x   | -78.24%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.413967 | 0.116785   | 3.54x   | -71.79%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.413967 | 0.093681   | 4.42x   | -77.37%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.413967 | 0.101155   | 4.09x   | -75.56%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.430129 | 0.113515   | 3.79x   | -73.61%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.430129 | 0.094620   | 4.55x   | -78.00%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.430129 | 0.098589   | 4.36x   | -77.08%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.367385 | 0.158035   | 2.32x   | -56.98%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.367385 | 0.084858   | 4.33x   | -76.90%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.367385 | 0.080975   | 4.54x   | -77.96%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.364367 | 0.119711   | 3.04x   | -67.15%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.364367 | 0.084504   | 4.31x   | -76.81%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.364367 | 0.081356   | 4.48x   | -77.67%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.373259 | 0.108518   | 3.44x   | -70.93%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.373259 | 0.091827   | 4.06x   | -75.40%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.373259 | 0.097467   | 3.83x   | -73.89%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.524126 | 0.187225   | 2.80x   | -64.28%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.524126 | 0.094882   | 5.52x   | -81.90%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.524126 | 0.098860   | 5.30x   | -81.14%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.482183 | 0.107015   | 4.51x   | -77.81%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.482183 | 0.084600   | 5.70x   | -82.45%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.482183 | 0.081758   | 5.90x   | -83.04%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.377002 | 0.100690   | 3.74x   | -73.29%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.377002 | 0.092377   | 4.08x   | -75.50%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.377002 | 0.097802   | 3.85x   | -74.06%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.379916 | 0.133561   | 2.84x   | -64.84%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.379916 | 0.093353   | 4.07x   | -75.43%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.379916 | 0.098445   | 3.86x   | -74.09%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.383619 | 0.131973   | 2.91x   | -65.60%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.383619 | 0.092854   | 4.13x   | -75.80%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.383619 | 0.097675   | 3.93x   | -74.54%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.391564 | 0.110413   | 3.55x   | -71.80%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.391564 | 0.094192   | 4.16x   | -75.94%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.391564 | 0.097900   | 4.00x   | -75.00%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.490935 | 0.179377   | 2.74x   | -63.46%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.490935 | 0.094049   | 5.22x   | -80.84%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.490935 | 0.098804   | 4.97x   | -79.87%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.272135 | 0.083224   | 3.27x   | -69.42%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.272135 | 0.009523   | 28.58x  | -96.50%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.272135 | 0.011876   | 22.92x  | -95.64%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.380869 | 0.102889   | 3.70x   | -72.99%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.380869 | 0.013969   | 27.26x  | -96.33%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.380869 | 0.013192   | 28.87x  | -96.54%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.268588 | 0.108384   | 2.48x   | -59.65%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.268588 | 0.012083   | 22.23x  | -95.50%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.268588 | 0.012606   | 21.31x  | -95.31%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.283683 | 0.099734   | 2.84x   | -64.84%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.283683 | 0.013063   | 21.72x  | -95.40%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.283683 | 0.013295   | 21.34x  | -95.31%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.290848 | 0.104487   | 2.78x   | -64.08%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.290848 | 0.015079   | 19.29x  | -94.82%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.290848 | 0.013360   | 21.77x  | -95.41%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.292282 | 0.126120   | 2.32x   | -56.85%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.292282 | 0.015712   | 18.60x  | -94.62%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.292282 | 0.013564   | 21.55x  | -95.36%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.265443 | 0.109252   | 2.43x   | -58.84%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.265443 | 0.012106   | 21.93x  | -95.44%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.265443 | 0.012689   | 20.92x  | -95.22%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.283645 | 0.100041   | 2.84x   | -64.73%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.283645 | 0.012610   | 22.49x  | -95.55%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.283645 | 0.013206   | 21.48x  | -95.34%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.431533 | 0.163343   | 2.64x   | -62.15%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.431533 | 0.015217   | 28.36x  | -96.47%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.431533 | 0.013099   | 32.94x  | -96.96%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.382427 | 0.097834   | 3.91x   | -74.42%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.382427 | 0.012199   | 31.35x  | -96.81%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.382427 | 0.012764   | 29.96x  | -96.66%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.274161 | 0.094534   | 2.90x   | -65.52%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.274161 | 0.016945   | 16.18x  | -93.82%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.274161 | 0.013765   | 19.92x  | -94.98%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.281415 | 0.118514   | 2.37x   | -57.89%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.281415 | 0.013380   | 21.03x  | -95.25%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.281415 | 0.013150   | 21.40x  | -95.33%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.280876 | 0.118408   | 2.37x   | -57.84%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.280876 | 0.013457   | 20.87x  | -95.21%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.280876 | 0.013146   | 21.37x  | -95.32%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.289392 | 0.100019   | 2.89x   | -65.44%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.289392 | 0.013828   | 20.93x  | -95.22%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.289392 | 0.013114   | 22.07x  | -95.47%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.390816 | 0.157957   | 2.47x   | -59.58%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.390816 | 0.014471   | 27.01x  | -96.30%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.390816 | 0.013047   | 29.95x  | -96.66%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.321857 | 0.036564   | 8.80x   | -88.64%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.321857 | 0.014599   | 22.05x  | -95.46%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.321857 | 0.014806   | 21.74x  | -95.40%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.427981 | 0.044245   | 9.67x   | -89.66%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.427981 | 0.019437   | 22.02x  | -95.46%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.427981 | 0.016659   | 25.69x  | -96.11%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.314043 | 0.051480   | 6.10x   | -83.61%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.314043 | 0.018158   | 17.30x  | -94.22%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.314043 | 0.016484   | 19.05x  | -94.75%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.332597 | 0.040989   | 8.11x   | -87.68%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.332597 | 0.019320   | 17.22x  | -94.19%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.332597 | 0.016859   | 19.73x  | -94.93%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.334974 | 0.045960   | 7.29x   | -86.28%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.334974 | 0.020854   | 16.06x  | -93.77%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.334974 | 0.017159   | 19.52x  | -94.88%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.323418 | 0.104440   | 3.10x   | -67.71%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.323418 | 0.020828   | 15.53x  | -93.56%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.323418 | 0.017683   | 18.29x  | -94.53%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.327526 | 0.051398   | 6.37x   | -84.31%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.327526 | 0.018133   | 18.06x  | -94.46%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.327526 | 0.016598   | 19.73x  | -94.93%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.322226 | 0.040005   | 8.05x   | -87.58%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.322226 | 0.019205   | 16.78x  | -94.04%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.322226 | 0.016678   | 19.32x  | -94.82%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.483079 | 0.117556   | 4.11x   | -75.67%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.483079 | 0.020904   | 23.11x  | -95.67%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.483079 | 0.017496   | 27.61x  | -96.38%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.438405 | 0.039440   | 11.12x  | -91.00%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.438405 | 0.018690   | 23.46x  | -95.74%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.438405 | 0.016911   | 25.92x  | -96.14%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.346718 | 0.049000   | 7.08x   | -85.87%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.346718 | 0.020476   | 16.93x  | -94.09%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.346718 | 0.017767   | 19.52x  | -94.88%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.347092 | 0.070473   | 4.93x   | -79.70%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.347092 | 0.020373   | 17.04x  | -94.13%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.347092 | 0.017310   | 20.05x  | -95.01%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.346034 | 0.070835   | 4.89x   | -79.53%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.346034 | 0.019789   | 17.49x  | -94.28%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.346034 | 0.016951   | 20.41x  | -95.10%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.346829 | 0.044745   | 7.75x   | -87.10%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.346829 | 0.020419   | 16.99x  | -94.11%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.346829 | 0.017272   | 20.08x  | -95.02%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.456885 | 0.113162   | 4.04x   | -75.23%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.456885 | 0.020242   | 22.57x  | -95.57%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.456885 | 0.017540   | 26.05x  | -96.16%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.262092 | 0.045982   | 5.70x   | -82.46%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.262092 | 0.018672   | 14.04x  | -92.88%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.262092 | 0.018330   | 14.30x  | -93.01%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.363545 | 0.053616   | 6.78x   | -85.25%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.363545 | 0.019560   | 18.59x  | -94.62%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.363545 | 0.018703   | 19.44x  | -94.86%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.245770 | 0.062916   | 3.91x   | -74.40%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.245770 | 0.023538   | 10.44x  | -90.42%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.245770 | 0.020639   | 11.91x  | -91.60%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.276669 | 0.050541   | 5.47x   | -81.73%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.276669 | 0.019531   | 14.17x  | -92.94%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.276669 | 0.018579   | 14.89x  | -93.28%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.271483 | 0.056463   | 4.81x   | -79.20%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.271483 | 0.022152   | 12.26x  | -91.84%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.271483 | 0.025283   | 10.74x  | -90.69%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.255958 | 0.095276   | 2.69x   | -62.78%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.255958 | 0.021048   | 12.16x  | -91.78%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.255958 | 0.019144   | 13.37x  | -92.52%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.246114 | 0.057212   | 4.30x   | -76.75%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.246114 | 0.018772   | 13.11x  | -92.37%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.246114 | 0.018536   | 13.28x  | -92.47%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.252025 | 0.048545   | 5.19x   | -80.74%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.252025 | 0.018400   | 13.70x  | -92.70%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.252025 | 0.018477   | 13.64x  | -92.67%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.403721 | 0.126464   | 3.19x   | -68.68%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.403721 | 0.021958   | 18.39x  | -94.56%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.403721 | 0.018745   | 21.54x  | -95.36%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.358818 | 0.048733   | 7.36x   | -86.42%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.358818 | 0.018443   | 19.46x  | -94.86%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.358818 | 0.018472   | 19.42x  | -94.85%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.251007 | 0.063566   | 3.95x   | -74.68%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.251007 | 0.020037   | 12.53x  | -92.02%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.251007 | 0.018974   | 13.23x  | -92.44%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.259174 | 0.075036   | 3.45x   | -71.05%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.259174 | 0.018422   | 14.07x  | -92.89%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.259174 | 0.018376   | 14.10x  | -92.91%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.261487 | 0.074552   | 3.51x   | -71.49%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.261487 | 0.019341   | 13.52x  | -92.60%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.261487 | 0.018469   | 14.16x  | -92.94%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.266422 | 0.052593   | 5.07x   | -80.26%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.266422 | 0.019376   | 13.75x  | -92.73%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.266422 | 0.018609   | 14.32x  | -93.02%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.370646 | 0.118491   | 3.13x   | -68.03%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.370646 | 0.019001   | 19.51x  | -94.87%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.370646 | 0.018304   | 20.25x  | -95.06%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.079270 | 0.022223   | 3.57x   | -71.97%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.079270 | 0.019079   | 4.15x   | -75.93%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.079270 | 0.018148   | 4.37x   | -77.11%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.104875 | 0.024988   | 4.20x   | -76.17%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.104875 | 0.020671   | 5.07x   | -80.29%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.104875 | 0.021650   | 4.84x   | -79.36%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.082287 | 0.025667   | 3.21x   | -68.81%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.082287 | 0.018401   | 4.47x   | -77.64%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.082287 | 0.017723   | 4.64x   | -78.46%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.084734 | 0.023798   | 3.56x   | -71.91%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.084734 | 0.020079   | 4.22x   | -76.30%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.084734 | 0.021398   | 3.96x   | -74.75%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.085385 | 0.024826   | 3.44x   | -70.92%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.085385 | 0.020726   | 4.12x   | -75.73%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.085385 | 0.021651   | 3.94x   | -74.64%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.081786 | 0.034788   | 2.35x   | -57.46%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.081786 | 0.018594   | 4.40x   | -77.27%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.081786 | 0.017712   | 4.62x   | -78.34%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.084002 | 0.026233   | 3.20x   | -68.77%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.084002 | 0.019037   | 4.41x   | -77.34%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.084002 | 0.018077   | 4.65x   | -78.48%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.084533 | 0.023727   | 3.56x   | -71.93%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.084533 | 0.020059   | 4.21x   | -76.27%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.084533 | 0.021465   | 3.94x   | -74.61%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.113401 | 0.041085   | 2.76x   | -63.77%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.113401 | 0.020771   | 5.46x   | -81.68%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.113401 | 0.021600   | 5.25x   | -80.95%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.109010 | 0.023579   | 4.62x   | -78.37%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.109010 | 0.018401   | 5.92x   | -83.12%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.109010 | 0.017636   | 6.18x   | -83.82%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.080534 | 0.022040   | 3.65x   | -72.63%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.080534 | 0.020345   | 3.96x   | -74.74%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.080534 | 0.021615   | 3.73x   | -73.16%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.084678 | 0.028927   | 2.93x   | -65.84%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.084678 | 0.020293   | 4.17x   | -76.04%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.084678 | 0.021427   | 3.95x   | -74.70%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.084441 | 0.029229   | 2.89x   | -65.38%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.084441 | 0.020318   | 4.16x   | -75.94%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.084441 | 0.021519   | 3.92x   | -74.52%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.086173 | 0.024189   | 3.56x   | -71.93%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.086173 | 0.020615   | 4.18x   | -76.08%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.086173 | 0.021444   | 4.02x   | -75.11%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.108245 | 0.039441   | 2.74x   | -63.56%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.108245 | 0.020529   | 5.27x   | -81.03%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.108245 | 0.022187   | 4.88x   | -79.50%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.061927 | 0.018009   | 3.44x   | -70.92%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.061927 | 0.002078   | 29.80x  | -96.64%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.061927 | 0.002573   | 24.07x  | -95.84%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.092287 | 0.022261   | 4.15x   | -75.88%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.092287 | 0.003044   | 30.32x  | -96.70%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.092287 | 0.002895   | 31.88x  | -96.86%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.070439 | 0.023619   | 2.98x   | -66.47%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.070439 | 0.002669   | 26.39x  | -96.21%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.070439 | 0.002752   | 25.60x  | -96.09%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.073374 | 0.021603   | 3.40x   | -70.56%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.073374 | 0.002862   | 25.64x  | -96.10%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.073374 | 0.002905   | 25.26x  | -96.04%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.076554 | 0.023437   | 3.27x   | -69.38%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.076554 | 0.003301   | 23.19x  | -95.69%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.076554 | 0.002899   | 26.41x  | -96.21%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.070291 | 0.027628   | 2.54x   | -60.70%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.070291 | 0.003462   | 20.30x  | -95.07%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.070291 | 0.002976   | 23.62x  | -95.77%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.070519 | 0.023928   | 2.95x   | -66.07%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.070519 | 0.002651   | 26.60x  | -96.24%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.070519 | 0.002791   | 25.27x  | -96.04%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.072735 | 0.021685   | 3.35x   | -70.19%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.072735 | 0.002753   | 26.42x  | -96.21%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.072735 | 0.002873   | 25.32x  | -96.05%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.100945 | 0.036069   | 2.80x   | -64.27%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.100945 | 0.003298   | 30.61x  | -96.73%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.100945 | 0.002861   | 35.28x  | -97.17%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.097848 | 0.021348   | 4.58x   | -78.18%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.097848 | 0.002683   | 36.48x  | -97.26%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.097848 | 0.002796   | 35.00x  | -97.14%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.068931 | 0.020648   | 3.34x   | -70.05%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.068931 | 0.003679   | 18.73x  | -94.66%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.068931 | 0.003002   | 22.96x  | -95.64%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.072605 | 0.026094   | 2.78x   | -64.06%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.072605 | 0.002927   | 24.81x  | -95.97%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.072605 | 0.002870   | 25.30x  | -96.05%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.073305 | 0.025825   | 2.84x   | -64.77%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.073305 | 0.002964   | 24.73x  | -95.96%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.073305 | 0.002863   | 25.60x  | -96.09%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.074218 | 0.022037   | 3.37x   | -70.31%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.074218 | 0.002989   | 24.83x  | -95.97%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.074218 | 0.002863   | 25.92x  | -96.14%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.095431 | 0.034602   | 2.76x   | -63.74%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.095431 | 0.003141   | 30.39x  | -96.71%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.095431 | 0.002902   | 32.89x  | -96.96%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.067667 | 0.006940   | 9.75x   | -89.74%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.067667 | 0.002007   | 33.72x  | -97.03%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.067667 | 0.002116   | 31.97x  | -96.87%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.098031 | 0.008544   | 11.47x  | -91.28%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.098031 | 0.003217   | 30.47x  | -96.72%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.098031 | 0.002591   | 37.83x  | -97.36%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.074841 | 0.010247   | 7.30x   | -86.31%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.074841 | 0.002915   | 25.68x  | -96.11%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.074841 | 0.002491   | 30.05x  | -96.67%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.077428 | 0.008089   | 9.57x   | -89.55%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.077428 | 0.003042   | 25.45x  | -96.07%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.077428 | 0.002585   | 29.95x  | -96.66%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.077911 | 0.008882   | 8.77x   | -88.60%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.077911 | 0.003426   | 22.74x  | -95.60%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.077911 | 0.002609   | 29.86x  | -96.65%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.077145 | 0.022100   | 3.49x   | -71.35%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.077145 | 0.003307   | 23.33x  | -95.71%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.077145 | 0.002689   | 28.69x  | -96.51%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.076401 | 0.010252   | 7.45x   | -86.58%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.076401 | 0.002953   | 25.87x  | -96.13%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.076401 | 0.002497   | 30.60x  | -96.73%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.075923 | 0.007693   | 9.87x   | -89.87%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.075923 | 0.002941   | 25.81x  | -96.13%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.075923 | 0.002496   | 30.42x  | -96.71%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.105355 | 0.024627   | 4.28x   | -76.62%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.105355 | 0.003395   | 31.03x  | -96.78%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.105355 | 0.002637   | 39.95x  | -97.50%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.101218 | 0.007621   | 13.28x  | -92.47%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.101218 | 0.003011   | 33.61x  | -97.03%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.101218 | 0.002536   | 39.92x  | -97.49%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.074611 | 0.009695   | 7.70x   | -87.01%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.074611 | 0.003312   | 22.52x  | -95.56%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.074611 | 0.002670   | 27.95x  | -96.42%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.078196 | 0.014251   | 5.49x   | -81.78%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.078196 | 0.003151   | 24.82x  | -95.97%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.078196 | 0.002589   | 30.21x  | -96.69%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.076999 | 0.014271   | 5.40x   | -81.47%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.076999 | 0.003131   | 24.59x  | -95.93%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.076999 | 0.002595   | 29.67x  | -96.63%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.078373 | 0.008627   | 9.08x   | -88.99%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.078373 | 0.003271   | 23.96x  | -95.83%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.078373 | 0.002612   | 30.00x  | -96.67%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.100333 | 0.023229   | 4.32x   | -76.85%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.100333 | 0.003260   | 30.78x  | -96.75%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.100333 | 0.002552   | 39.32x  | -97.46%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.053590 | 0.008522   | 6.29x   | -84.10%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.053590 | 0.002242   | 23.91x  | -95.82%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.053590 | 0.002459   | 21.80x  | -95.41%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.086296 | 0.010000   | 8.63x   | -88.41%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.086296 | 0.002659   | 32.46x  | -96.92%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.086296 | 0.002647   | 32.60x  | -96.93%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.063262 | 0.010735   | 5.89x   | -83.03%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.063262 | 0.002539   | 24.92x  | -95.99%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.063262 | 0.002607   | 24.27x  | -95.88%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.065528 | 0.009408   | 6.97x   | -85.64%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.065528 | 0.002570   | 25.50x  | -96.08%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.065528 | 0.002659   | 24.64x  | -95.94%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.065823 | 0.010508   | 6.26x   | -84.04%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.065823 | 0.003107   | 21.19x  | -95.28%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.065823 | 0.002686   | 24.51x  | -95.92%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.062141 | 0.018898   | 3.29x   | -69.59%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.062141 | 0.002766   | 22.47x  | -95.55%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.062141 | 0.002726   | 22.79x  | -95.61%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.063830 | 0.011036   | 5.78x   | -82.71%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.063830 | 0.002539   | 25.14x  | -96.02%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.063830 | 0.002689   | 23.74x  | -95.79%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.064699 | 0.009005   | 7.18x   | -86.08%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.064699 | 0.002554   | 25.33x  | -96.05%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.064699 | 0.002640   | 24.51x  | -95.92%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.093753 | 0.026222   | 3.58x   | -72.03%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.093753 | 0.003200   | 29.29x  | -96.59%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.093753 | 0.002650   | 35.38x  | -97.17%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.088905 | 0.009094   | 9.78x   | -89.77%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.088905 | 0.002556   | 34.78x  | -97.12%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.088905 | 0.002614   | 34.01x  | -97.06%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.061935 | 0.012202   | 5.08x   | -80.30%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.061935 | 0.002885   | 21.47x  | -95.34%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.061935 | 0.002687   | 23.05x  | -95.66%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.065861 | 0.014814   | 4.45x   | -77.51%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.065861 | 0.002688   | 24.50x  | -95.92%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.065861 | 0.002759   | 23.87x  | -95.81%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.068980 | 0.014838   | 4.65x   | -78.49%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.068980 | 0.002582   | 26.72x  | -96.26%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.068980 | 0.002580   | 26.73x  | -96.26%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.066662 | 0.009900   | 6.73x   | -85.15%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.066662 | 0.002645   | 25.20x  | -96.03%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.066662 | 0.002597   | 25.67x  | -96.10%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.087822 | 0.024301   | 3.61x   | -72.33%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.087822 | 0.002693   | 32.62x  | -96.93%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.087822 | 0.002677   | 32.80x  | -96.95%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.175109 | 0.049748   | 3.52x   | -71.59%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.175109 | 0.042912   | 4.08x   | -75.49%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.175109 | 0.040797   | 4.29x   | -76.70%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.237710 | 0.055646   | 4.27x   | -76.59%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.237710 | 0.046582   | 5.10x   | -80.40%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.237710 | 0.049766   | 4.78x   | -79.06%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.183760 | 0.058211   | 3.16x   | -68.32%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.183760 | 0.041683   | 4.41x   | -77.32%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.183760 | 0.039859   | 4.61x   | -78.31%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.185943 | 0.053485   | 3.48x   | -71.24%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.185943 | 0.045139   | 4.12x   | -75.72%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.185943 | 0.048205   | 3.86x   | -74.08%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.186663 | 0.055729   | 3.35x   | -70.14%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.186663 | 0.046627   | 4.00x   | -75.02%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.186663 | 0.048912   | 3.82x   | -73.80%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.182258 | 0.077901   | 2.34x   | -57.26%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.182258 | 0.041699   | 4.37x   | -77.12%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.182258 | 0.039860   | 4.57x   | -78.13%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.181760 | 0.058243   | 3.12x   | -67.96%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.181760 | 0.041544   | 4.38x   | -77.14%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.181760 | 0.040027   | 4.54x   | -77.98%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.183457 | 0.053335   | 3.44x   | -70.93%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.183457 | 0.045125   | 4.07x   | -75.40%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.183457 | 0.048001   | 3.82x   | -73.84%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.247158 | 0.091591   | 2.70x   | -62.94%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.247158 | 0.046622   | 5.30x   | -81.14%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.247158 | 0.048531   | 5.09x   | -80.36%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.237511 | 0.052812   | 4.50x   | -77.76%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.237511 | 0.041345   | 5.74x   | -82.59%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.237511 | 0.039773   | 5.97x   | -83.25%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.181351 | 0.049613   | 3.66x   | -72.64%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.181351 | 0.045524   | 3.98x   | -74.90%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.181351 | 0.048158   | 3.77x   | -73.44%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.185266 | 0.065888   | 2.81x   | -64.44%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.185266 | 0.046530   | 3.98x   | -74.88%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.185266 | 0.049011   | 3.78x   | -73.55%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.186691 | 0.065189   | 2.86x   | -65.08%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.186691 | 0.045648   | 4.09x   | -75.55%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.186691 | 0.048082   | 3.88x   | -74.25%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.187035 | 0.054374   | 3.44x   | -70.93%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.187035 | 0.046386   | 4.03x   | -75.20%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.187035 | 0.048236   | 3.88x   | -74.21%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.238441 | 0.088469   | 2.70x   | -62.90%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.238441 | 0.046214   | 5.16x   | -80.62%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.238441 | 0.048253   | 4.94x   | -79.76%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.126211 | 0.040533   | 3.11x   | -67.88%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.126211 | 0.004726   | 26.70x  | -96.26%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.126211 | 0.005770   | 21.87x  | -95.43%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.185032 | 0.050236   | 3.68x   | -72.85%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.185032 | 0.006872   | 26.93x  | -96.29%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.185032 | 0.006497   | 28.48x  | -96.49%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.134032 | 0.053248   | 2.52x   | -60.27%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.134032 | 0.005867   | 22.84x  | -95.62%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.134032 | 0.006193   | 21.64x  | -95.38%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.139506 | 0.048235   | 2.89x   | -65.42%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.139506 | 0.006489   | 21.50x  | -95.35%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.139506 | 0.006465   | 21.58x  | -95.37%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.140106 | 0.050741   | 2.76x   | -63.78%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.140106 | 0.007312   | 19.16x  | -94.78%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.140106 | 0.006465   | 21.67x  | -95.39%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.136123 | 0.062039   | 2.19x   | -54.42%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.136123 | 0.007755   | 17.55x  | -94.30%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.136123 | 0.006677   | 20.39x  | -95.09%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.135495 | 0.053631   | 2.53x   | -60.42%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.135495 | 0.006000   | 22.58x  | -95.57%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.135495 | 0.006191   | 21.89x  | -95.43%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.136541 | 0.048119   | 2.84x   | -64.76%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.136541 | 0.006215   | 21.97x  | -95.45%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.136541 | 0.006434   | 21.22x  | -95.29%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.201141 | 0.080333   | 2.50x   | -60.06%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.201141 | 0.007411   | 27.14x  | -96.32%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.201141 | 0.006416   | 31.35x  | -96.81%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.192378 | 0.048103   | 4.00x   | -75.00%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.192378 | 0.005983   | 32.16x  | -96.89%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.192378 | 0.006302   | 30.53x  | -96.72%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.136832 | 0.046191   | 2.96x   | -66.24%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.136832 | 0.008287   | 16.51x  | -93.94%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.136832 | 0.006788   | 20.16x  | -95.04%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.140649 | 0.058276   | 2.41x   | -58.57%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.140649 | 0.006587   | 21.35x  | -95.32%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.140649 | 0.006426   | 21.89x  | -95.43%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.139599 | 0.058102   | 2.40x   | -58.38%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.139599 | 0.006548   | 21.32x  | -95.31%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.139599 | 0.006453   | 21.63x  | -95.38%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.141575 | 0.049386   | 2.87x   | -65.12%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.141575 | 0.006731   | 21.03x  | -95.25%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.141575 | 0.006444   | 21.97x  | -95.45%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.190361 | 0.078034   | 2.44x   | -59.01%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.190361 | 0.006989   | 27.24x  | -96.33%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.190361 | 0.006527   | 29.17x  | -96.57%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.150505 | 0.016620   | 9.06x   | -88.96%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.150505 | 0.004550   | 33.08x  | -96.98%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.150505 | 0.004826   | 31.18x  | -96.79%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.206183 | 0.020276   | 10.17x  | -90.17%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.206183 | 0.007325   | 28.15x  | -96.45%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.206183 | 0.005786   | 35.63x  | -97.19%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.157410 | 0.023696   | 6.64x   | -84.95%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.157410 | 0.006618   | 23.79x  | -95.80%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.157410 | 0.005659   | 27.82x  | -96.41%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.163915 | 0.018364   | 8.93x   | -88.80%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.163915 | 0.006995   | 23.43x  | -95.73%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.163915 | 0.005791   | 28.31x  | -96.47%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.164614 | 0.020689   | 7.96x   | -87.43%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.164614 | 0.007878   | 20.89x  | -95.21%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.164614 | 0.005855   | 28.12x  | -96.44%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.160835 | 0.049855   | 3.23x   | -69.00%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.160835 | 0.007419   | 21.68x  | -95.39%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.160835 | 0.005998   | 26.81x  | -96.27%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.157124 | 0.024329   | 6.46x   | -84.52%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.157124 | 0.006581   | 23.88x  | -95.81%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.157124 | 0.005693   | 27.60x  | -96.38%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.159001 | 0.019844   | 8.01x   | -87.52%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.159001 | 0.006669   | 23.84x  | -95.81%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.159001 | 0.005663   | 28.08x  | -96.44%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.233065 | 0.056598   | 4.12x   | -75.72%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.233065 | 0.007759   | 30.04x  | -96.67%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.233065 | 0.005847   | 39.86x  | -97.49%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.214929 | 0.018021   | 11.93x  | -91.62%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.214929 | 0.006880   | 31.24x  | -96.80%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.214929 | 0.005679   | 37.85x  | -97.36%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.159986 | 0.022789   | 7.02x   | -85.76%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.159986 | 0.007440   | 21.50x  | -95.35%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.159986 | 0.006002   | 26.65x  | -96.25%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.163416 | 0.032908   | 4.97x   | -79.86%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.163416 | 0.007054   | 23.17x  | -95.68%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.163416 | 0.005802   | 28.16x  | -96.45%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.161110 | 0.032966   | 4.89x   | -79.54%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.161110 | 0.007130   | 22.60x  | -95.57%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.161110 | 0.005793   | 27.81x  | -96.40%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.163775 | 0.019939   | 8.21x   | -87.83%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.163775 | 0.007353   | 22.27x  | -95.51%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.163775 | 0.005773   | 28.37x  | -96.48%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.212473 | 0.053040   | 4.01x   | -75.04%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.212473 | 0.007296   | 29.12x  | -96.57%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.212473 | 0.005782   | 36.75x  | -97.28%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.116246 | 0.019084   | 6.09x   | -83.58%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.116246 | 0.004950   | 23.49x  | -95.74%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.116246 | 0.005507   | 21.11x  | -95.26%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.173938 | 0.022420   | 7.76x   | -87.11%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.173938 | 0.005820   | 29.89x  | -96.65%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.173938 | 0.005935   | 29.31x  | -96.59%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.124277 | 0.024047   | 5.17x   | -80.65%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.124277 | 0.005541   | 22.43x  | -95.54%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.124277 | 0.005844   | 21.26x  | -95.30%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.130979 | 0.021006   | 6.24x   | -83.96%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.130979 | 0.005645   | 23.20x  | -95.69%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.130979 | 0.005854   | 22.37x  | -95.53%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.129684 | 0.023499   | 5.52x   | -81.88%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.129684 | 0.006913   | 18.76x  | -94.67%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.129684 | 0.005851   | 22.17x  | -95.49%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.127013 | 0.042522   | 2.99x   | -66.52%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.127013 | 0.006102   | 20.81x  | -95.20%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.127013 | 0.005941   | 21.38x  | -95.32%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.124320 | 0.024093   | 5.16x   | -80.62%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.124320 | 0.005523   | 22.51x  | -95.56%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.124320 | 0.005739   | 21.66x  | -95.38%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.127503 | 0.020065   | 6.35x   | -84.26%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.127503 | 0.005559   | 22.93x  | -95.64%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.127503 | 0.005745   | 22.19x  | -95.49%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.192205 | 0.057971   | 3.32x   | -69.84%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.192205 | 0.007079   | 27.15x  | -96.32%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.192205 | 0.005795   | 33.17x  | -96.99%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.184353 | 0.020485   | 9.00x   | -88.89%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.184353 | 0.005653   | 32.61x  | -96.93%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.184353 | 0.005830   | 31.62x  | -96.84%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.127422 | 0.027485   | 4.64x   | -78.43%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.127422 | 0.006171   | 20.65x  | -95.16%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.127422 | 0.006042   | 21.09x  | -95.26%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.130014 | 0.033327   | 3.90x   | -74.37%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.130014 | 0.005643   | 23.04x  | -95.66%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.130014 | 0.005772   | 22.53x  | -95.56%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.129601 | 0.033344   | 3.89x   | -74.27%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.129601 | 0.005744   | 22.56x  | -95.57%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.129601 | 0.005763   | 22.49x  | -95.55%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.132589 | 0.022300   | 5.95x   | -83.18%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.132589 | 0.005782   | 22.93x  | -95.64%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.132589 | 0.005793   | 22.89x  | -95.63%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.178621 | 0.054573   | 3.27x   | -69.45%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.178621 | 0.006041   | 29.57x  | -96.62%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.178621 | 0.005779   | 30.91x  | -96.76%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.311445 | 0.088960   | 3.50x   | -71.44%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.311445 | 0.076135   | 4.09x   | -75.55%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.311445 | 0.072477   | 4.30x   | -76.73%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.409168 | 0.099265   | 4.12x   | -75.74%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.409168 | 0.082705   | 4.95x   | -79.79%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.409168 | 0.086181   | 4.75x   | -78.94%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.312492 | 0.103040   | 3.03x   | -67.03%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.312492 | 0.073509   | 4.25x   | -76.48%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.312492 | 0.071180   | 4.39x   | -77.22%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.328827 | 0.094837   | 3.47x   | -71.16%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.328827 | 0.080339   | 4.09x   | -75.57%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.328827 | 0.085778   | 3.83x   | -73.91%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.329431 | 0.099145   | 3.32x   | -69.90%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.329431 | 0.082887   | 3.97x   | -74.84%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.329431 | 0.086121   | 3.83x   | -73.86%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.321710 | 0.138785   | 2.32x   | -56.86%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.321710 | 0.074478   | 4.32x   | -76.85%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.321710 | 0.071418   | 4.50x   | -77.80%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.312238 | 0.103647   | 3.01x   | -66.81%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.312238 | 0.074000   | 4.22x   | -76.30%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.312238 | 0.071213   | 4.38x   | -77.19%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.321406 | 0.094618   | 3.40x   | -70.56%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.321406 | 0.080384   | 4.00x   | -74.99%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.321406 | 0.087889   | 3.66x   | -72.65%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.459765 | 0.163150   | 2.82x   | -64.51%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.459765 | 0.082787   | 5.55x   | -81.99%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.459765 | 0.086365   | 5.32x   | -81.22%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.414305 | 0.093593   | 4.43x   | -77.41%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.414305 | 0.073879   | 5.61x   | -82.17%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.414305 | 0.070657   | 5.86x   | -82.95%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.318575 | 0.088339   | 3.61x   | -72.27%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.318575 | 0.081263   | 3.92x   | -74.49%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.318575 | 0.085605   | 3.72x   | -73.13%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.328744 | 0.117155   | 2.81x   | -64.36%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.328744 | 0.081299   | 4.04x   | -75.27%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.328744 | 0.085549   | 3.84x   | -73.98%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.328429 | 0.115947   | 2.83x   | -64.70%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.328429 | 0.081113   | 4.05x   | -75.30%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.328429 | 0.085525   | 3.84x   | -73.96%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.331216 | 0.096602   | 3.43x   | -70.83%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.331216 | 0.082753   | 4.00x   | -75.02%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.331216 | 0.086030   | 3.85x   | -74.03%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.421426 | 0.157498   | 2.68x   | -62.63%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.421426 | 0.081933   | 5.14x   | -80.56%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.421426 | 0.086522   | 4.87x   | -79.47%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.240081 | 0.073086   | 3.28x   | -69.56%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.240081 | 0.008287   | 28.97x  | -96.55%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.240081 | 0.010289   | 23.33x  | -95.71%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.331942 | 0.089284   | 3.72x   | -73.10%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.331942 | 0.012170   | 27.27x  | -96.33%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.331942 | 0.011478   | 28.92x  | -96.54%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.241151 | 0.095035   | 2.54x   | -60.59%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.241151 | 0.010578   | 22.80x  | -95.61%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.241151 | 0.011059   | 21.81x  | -95.41%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.245811 | 0.086084   | 2.86x   | -64.98%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.245811 | 0.011464   | 21.44x  | -95.34%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.245811 | 0.011516   | 21.35x  | -95.32%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.249158 | 0.090908   | 2.74x   | -63.51%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.249158 | 0.013075   | 19.06x  | -94.75%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.249158 | 0.011618   | 21.45x  | -95.34%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.238306 | 0.110936   | 2.15x   | -53.45%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.238306 | 0.013761   | 17.32x  | -94.23%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.238306 | 0.012039   | 19.80x  | -94.95%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.230729 | 0.095682   | 2.41x   | -58.53%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.230729 | 0.010543   | 21.89x  | -95.43%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.230729 | 0.010992   | 20.99x  | -95.24%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.233730 | 0.085909   | 2.72x   | -63.24%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.233730 | 0.011006   | 21.24x  | -95.29%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.233730 | 0.011518   | 20.29x  | -95.07%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.378496 | 0.143412   | 2.64x   | -62.11%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.378496 | 0.013190   | 28.70x  | -96.52%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.378496 | 0.011559   | 32.74x  | -96.95%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.331883 | 0.086265   | 3.85x   | -74.01%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.331883 | 0.010673   | 31.10x  | -96.78%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.331883 | 0.011260   | 29.47x  | -96.61%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.246031 | 0.085583   | 2.87x   | -65.21%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.246031 | 0.014893   | 16.52x  | -93.95%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.246031 | 0.012104   | 20.33x  | -95.08%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.248253 | 0.104877   | 2.37x   | -57.75%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.248253 | 0.011731   | 21.16x  | -95.27%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.248253 | 0.011536   | 21.52x  | -95.35%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.245944 | 0.104367   | 2.36x   | -57.56%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.245944 | 0.011768   | 20.90x  | -95.22%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.245944 | 0.011478   | 21.43x  | -95.33%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.249248 | 0.088689   | 2.81x   | -64.42%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.249248 | 0.012103   | 20.59x  | -95.14%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.249248 | 0.011559   | 21.56x  | -95.36%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.349430 | 0.139464   | 2.51x   | -60.09%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.349430 | 0.012598   | 27.74x  | -96.39%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.349430 | 0.012358   | 28.28x  | -96.46%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.279338 | 0.027936   | 10.00x  | -90.00%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.279338 | 0.008426   | 33.15x  | -96.98%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.279338 | 0.008860   | 31.53x  | -96.83%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.391280 | 0.035353   | 11.07x  | -90.96%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.391280 | 0.013276   | 29.47x  | -96.61%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.391280 | 0.010414   | 37.57x  | -97.34%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.276570 | 0.041104   | 6.73x   | -85.14%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.276570 | 0.012024   | 23.00x  | -95.65%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.276570 | 0.010233   | 27.03x  | -96.30%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.295296 | 0.032017   | 9.22x   | -89.16%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.295296 | 0.012556   | 23.52x  | -95.75%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.295296 | 0.010588   | 27.89x  | -96.41%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.296806 | 0.035862   | 8.28x   | -87.92%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.296806 | 0.013913   | 21.33x  | -95.31%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.296806 | 0.010604   | 27.99x  | -96.43%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.286271 | 0.086891   | 3.29x   | -69.65%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.286271 | 0.013254   | 21.60x  | -95.37%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.286271 | 0.010666   | 26.84x  | -96.27%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.286169 | 0.042884   | 6.67x   | -85.01%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.286169 | 0.012162   | 23.53x  | -95.75%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.286169 | 0.010241   | 27.94x  | -96.42%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.286500 | 0.030959   | 9.25x   | -89.19%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.286500 | 0.012014   | 23.85x  | -95.81%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.286500 | 0.010329   | 27.74x  | -96.39%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.421232 | 0.099960   | 4.21x   | -76.27%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.421232 | 0.013871   | 30.37x  | -96.71%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.421232 | 0.011005   | 38.28x  | -97.39%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.376933 | 0.032107   | 11.74x  | -91.48%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.376933 | 0.012270   | 30.72x  | -96.74%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.376933 | 0.010227   | 36.86x  | -97.29%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.283904 | 0.043060   | 6.59x   | -84.83%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.283904 | 0.014174   | 20.03x  | -95.01%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.283904 | 0.010838   | 26.20x  | -96.18%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.312712 | 0.057061   | 5.48x   | -81.75%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.312712 | 0.012876   | 24.29x  | -95.88%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.312712 | 0.010366   | 30.17x  | -96.68%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.291764 | 0.060039   | 4.86x   | -79.42%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.291764 | 0.012709   | 22.96x  | -95.64%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.291764 | 0.010316   | 28.28x  | -96.46%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.316569 | 0.035128   | 9.01x   | -88.90%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.316569 | 0.013348   | 23.72x  | -95.78%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.316569 | 0.010489   | 30.18x  | -96.69%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.391986 | 0.092972   | 4.22x   | -76.28%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.391986 | 0.013317   | 29.44x  | -96.60%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.391986 | 0.011297   | 34.70x  | -97.12%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.251020 | 0.040135   | 6.25x   | -84.01%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.251020 | 0.015022   | 16.71x  | -94.02%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.251020 | 0.017106   | 14.67x  | -93.19%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.327993 | 0.046779   | 7.01x   | -85.74%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.327993 | 0.017075   | 19.21x  | -94.79%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.327993 | 0.016519   | 19.85x  | -94.96%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.228554 | 0.049669   | 4.60x   | -78.27%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.228554 | 0.016262   | 14.05x  | -92.89%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.228554 | 0.016295   | 14.03x  | -92.87%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.247125 | 0.045029   | 5.49x   | -81.78%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.247125 | 0.017125   | 14.43x  | -93.07%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.247125 | 0.016691   | 14.81x  | -93.25%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.260071 | 0.049496   | 5.25x   | -80.97%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.260071 | 0.018906   | 13.76x  | -92.73%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.260071 | 0.016845   | 15.44x  | -93.52%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.242061 | 0.084935   | 2.85x   | -64.91%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.242061 | 0.018674   | 12.96x  | -92.29%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.242061 | 0.018281   | 13.24x  | -92.45%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.266460 | 0.052213   | 5.10x   | -80.40%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.266460 | 0.017776   | 14.99x  | -93.33%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.266460 | 0.017478   | 15.25x  | -93.44%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.249470 | 0.042553   | 5.86x   | -82.94%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.249470 | 0.017427   | 14.31x  | -93.01%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.249470 | 0.017280   | 14.44x  | -93.07%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.383524 | 0.110657   | 3.47x   | -71.15%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.383524 | 0.019517   | 19.65x  | -94.91%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.383524 | 0.016621   | 23.07x  | -95.67%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.344181 | 0.043076   | 7.99x   | -87.48%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.344181 | 0.016664   | 20.65x  | -95.16%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.344181 | 0.016507   | 20.85x  | -95.20%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.245978 | 0.055219   | 4.45x   | -77.55%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.245978 | 0.017492   | 14.06x  | -92.89%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.245978 | 0.016949   | 14.51x  | -93.11%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.278360 | 0.066093   | 4.21x   | -76.26%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.278360 | 0.017264   | 16.12x  | -93.80%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.278360 | 0.016287   | 17.09x  | -94.15%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.240488 | 0.065010   | 3.70x   | -72.97%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.240488 | 0.016679   | 14.42x  | -93.06%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.240488 | 0.016854   | 14.27x  | -92.99%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.250080 | 0.046232   | 5.41x   | -81.51%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.250080 | 0.017036   | 14.68x  | -93.19%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.250080 | 0.016366   | 15.28x  | -93.46%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.338463 | 0.103587   | 3.27x   | -69.39%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.338463 | 0.017770   | 19.05x  | -94.75%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.338463 | 0.016897   | 20.03x  | -95.01%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.748858 | 0.203829   | 3.67x   | -72.78%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.748858 | 0.173482   | 4.32x   | -76.83%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.748858 | 0.163069   | 4.59x   | -78.22%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 0.912404 | 0.224899   | 4.06x   | -75.35%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 0.912404 | 0.186370   | 4.90x   | -79.57%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 0.912404 | 0.193670   | 4.71x   | -78.77%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.698018 | 0.234977   | 2.97x   | -66.34%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.698018 | 0.165547   | 4.22x   | -76.28%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.698018 | 0.161400   | 4.32x   | -76.88%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.740187 | 0.219776   | 3.37x   | -70.31%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.740187 | 0.189292   | 3.91x   | -74.43%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.740187 | 0.195761   | 3.78x   | -73.55%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.754625 | 0.225751   | 3.34x   | -70.08%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.754625 | 0.187342   | 4.03x   | -75.17%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.754625 | 0.194155   | 3.89x   | -74.27%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.729349 | 0.313531   | 2.33x   | -57.01%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.729349 | 0.168507   | 4.33x   | -76.90%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.729349 | 0.162692   | 4.48x   | -77.69%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.721116 | 0.237710   | 3.03x   | -67.04%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.721116 | 0.168044   | 4.29x   | -76.70%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.721116 | 0.162107   | 4.45x   | -77.52%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.731921 | 0.215958   | 3.39x   | -70.49%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.731921 | 0.181752   | 4.03x   | -75.17%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.731921 | 0.193151   | 3.79x   | -73.61%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.098455 | 0.373368   | 2.94x   | -66.01%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.098455 | 0.187397   | 5.86x   | -82.94%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.098455 | 0.196365   | 5.59x   | -82.12%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.944496 | 0.214402   | 4.41x   | -77.30%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.944496 | 0.167160   | 5.65x   | -82.30%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.944496 | 0.160499   | 5.88x   | -83.01%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.732793 | 0.207316   | 3.53x   | -71.71%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.732793 | 0.183944   | 3.98x   | -74.90%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.732793 | 0.194436   | 3.77x   | -73.47%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.740497 | 0.263239   | 2.81x   | -64.45%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.740497 | 0.184581   | 4.01x   | -75.07%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.740497 | 0.199166   | 3.72x   | -73.10%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.789616 | 0.268400   | 2.94x   | -66.01%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.789616 | 0.186128   | 4.24x   | -76.43%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.789616 | 0.193513   | 4.08x   | -75.49%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.760829 | 0.221281   | 3.44x   | -70.92%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.760829 | 0.188135   | 4.04x   | -75.27%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.760829 | 0.194421   | 3.91x   | -74.45%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 0.942256 | 0.356683   | 2.64x   | -62.15%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 0.942256 | 0.187759   | 5.02x   | -80.07%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 0.942256 | 0.199883   | 4.71x   | -78.79%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.521299 | 0.163984   | 3.18x   | -68.54%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.521299 | 0.018529   | 28.13x  | -96.45%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.521299 | 0.023058   | 22.61x  | -95.58%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.733900 | 0.201826   | 3.64x   | -72.50%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.733900 | 0.027303   | 26.88x  | -96.28%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.733900 | 0.025704   | 28.55x  | -96.50%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.519056 | 0.214555   | 2.42x   | -58.66%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.519056 | 0.023525   | 22.06x  | -95.47%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.519056 | 0.024565   | 21.13x  | -95.27%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.549547 | 0.194557   | 2.82x   | -64.60%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.549547 | 0.025422   | 21.62x  | -95.37%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.549547 | 0.025840   | 21.27x  | -95.30%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.561650 | 0.207443   | 2.71x   | -63.07%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.561650 | 0.029182   | 19.25x  | -94.80%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.561650 | 0.025807   | 21.76x  | -95.41%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.531803 | 0.250465   | 2.12x   | -52.90%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.531803 | 0.030867   | 17.23x  | -94.20%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.531803 | 0.026692   | 19.92x  | -94.98%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.515789 | 0.216508   | 2.38x   | -58.02%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.515789 | 0.025163   | 20.50x  | -95.12%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.515789 | 0.026774   | 19.26x  | -94.81%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.580825 | 0.194348   | 2.99x   | -66.54%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.580825 | 0.024927   | 23.30x  | -95.71%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.580825 | 0.025997   | 22.34x  | -95.52%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.841460 | 0.325125   | 2.59x   | -61.36%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.841460 | 0.029730   | 28.30x  | -96.47%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.841460 | 0.025814   | 32.60x  | -96.93%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.755409 | 0.195804   | 3.86x   | -74.08%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.755409 | 0.024068   | 31.39x  | -96.81%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.755409 | 0.025065   | 30.14x  | -96.68%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.538231 | 0.188642   | 2.85x   | -64.95%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.538231 | 0.033210   | 16.21x  | -93.83%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.538231 | 0.027226   | 19.77x  | -94.94%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.555133 | 0.235693   | 2.36x   | -57.54%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.555133 | 0.026355   | 21.06x  | -95.25%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.555133 | 0.026283   | 21.12x  | -95.27%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.549783 | 0.255079   | 2.16x   | -53.60%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.549783 | 0.026651   | 20.63x  | -95.15%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.549783 | 0.026292   | 20.91x  | -95.22%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.623415 | 0.203914   | 3.06x   | -67.29%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.623415 | 0.027341   | 22.80x  | -95.61%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.623415 | 0.025857   | 24.11x  | -95.85%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.786758 | 0.315461   | 2.49x   | -59.90%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.786758 | 0.028125   | 27.97x  | -96.43%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.786758 | 0.025905   | 30.37x  | -96.71%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.615697 | 0.071885   | 8.56x   | -88.32%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.615697 | 0.027335   | 22.52x  | -95.56%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.615697 | 0.028957   | 21.26x  | -95.30%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.835860 | 0.086889   | 9.62x   | -89.60%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.835860 | 0.038822   | 21.53x  | -95.36%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.835860 | 0.032830   | 25.46x  | -96.07%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.620715 | 0.101809   | 6.10x   | -83.60%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.620715 | 0.036532   | 16.99x  | -94.11%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.620715 | 0.032596   | 19.04x  | -94.75%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.648287 | 0.080634   | 8.04x   | -87.56%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.648287 | 0.036852   | 17.59x  | -94.32%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.648287 | 0.033141   | 19.56x  | -94.89%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.660157 | 0.089239   | 7.40x   | -86.48%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.660157 | 0.040590   | 16.26x  | -93.85%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.660157 | 0.033178   | 19.90x  | -94.97%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.628914 | 0.204785   | 3.07x   | -67.44%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.628914 | 0.038810   | 16.20x  | -93.83%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.628914 | 0.033773   | 18.62x  | -94.63%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.619089 | 0.104266   | 5.94x   | -83.16%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.619089 | 0.036147   | 17.13x  | -94.16%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.619089 | 0.032419   | 19.10x  | -94.76%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.632412 | 0.078806   | 8.02x   | -87.54%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.632412 | 0.035931   | 17.60x  | -94.32%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.632412 | 0.032248   | 19.61x  | -94.90%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.938377 | 0.234415   | 4.00x   | -75.02%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.938377 | 0.039961   | 23.48x  | -95.74%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.938377 | 0.033398   | 28.10x  | -96.44%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.852228 | 0.078808   | 10.81x  | -90.75%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.852228 | 0.036870   | 23.11x  | -95.67%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.852228 | 0.032506   | 26.22x  | -96.19%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.625587 | 0.095819   | 6.53x   | -84.68%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.625587 | 0.039364   | 15.89x  | -93.71%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.625587 | 0.033929   | 18.44x  | -94.58%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.639335 | 0.136061   | 4.70x   | -78.72%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.639335 | 0.037557   | 17.02x  | -94.13%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.639335 | 0.032761   | 19.52x  | -94.88%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.646449 | 0.138174   | 4.68x   | -78.63%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.646449 | 0.037506   | 17.24x  | -94.20%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.646449 | 0.032937   | 19.63x  | -94.90%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.647181 | 0.088853   | 7.28x   | -86.27%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.647181 | 0.038648   | 16.75x  | -94.03%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.647181 | 0.032762   | 19.75x  | -94.94%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.858021 | 0.219547   | 3.91x   | -74.41%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.858021 | 0.040660   | 21.10x  | -95.26%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.858021 | 0.033231   | 25.82x  | -96.13%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.510601 | 0.088157   | 5.79x   | -82.73%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.510601 | 0.030203   | 16.91x  | -94.08%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.510601 | 0.033033   | 15.46x  | -93.53%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.692649 | 0.101662   | 6.81x   | -85.32%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.692649 | 0.034934   | 19.83x  | -94.96%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.692649 | 0.035066   | 19.75x  | -94.94%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.468059 | 0.108220   | 4.33x   | -76.88%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.468059 | 0.033533   | 13.96x  | -92.84%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.468059 | 0.035194   | 13.30x  | -92.48%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.518213 | 0.096336   | 5.38x   | -81.41%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.518213 | 0.034035   | 15.23x  | -93.43%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.518213 | 0.035259   | 14.70x  | -93.20%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.505645 | 0.106326   | 4.76x   | -78.97%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.505645 | 0.039296   | 12.87x  | -92.23%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.505645 | 0.035420   | 14.28x  | -93.00%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.493768 | 0.182258   | 2.71x   | -63.09%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.493768 | 0.036601   | 13.49x  | -92.59%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.493768 | 0.036442   | 13.55x  | -92.62%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.486262 | 0.108830   | 4.47x   | -77.62%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.486262 | 0.034659   | 14.03x  | -92.87%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.486262 | 0.035699   | 13.62x  | -92.66%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.495893 | 0.092612   | 5.35x   | -81.32%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.495893 | 0.034789   | 14.25x  | -92.98%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.495893 | 0.035463   | 13.98x  | -92.85%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.791947 | 0.244279   | 3.24x   | -69.15%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.791947 | 0.040435   | 19.59x  | -94.89%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.791947 | 0.035918   | 22.05x  | -95.46%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.704937 | 0.094097   | 7.49x   | -86.65%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.704937 | 0.034709   | 20.31x  | -95.08%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.704937 | 0.035157   | 20.05x  | -95.01%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.492569 | 0.122262   | 4.03x   | -75.18%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.492569 | 0.038235   | 12.88x  | -92.24%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.492569 | 0.036074   | 13.65x  | -92.68%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.513216 | 0.144744   | 3.55x   | -71.80%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.513216 | 0.034795   | 14.75x  | -93.22%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.513216 | 0.035303   | 14.54x  | -93.12%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.514529 | 0.146225   | 3.52x   | -71.58%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.514529 | 0.035576   | 14.46x  | -93.09%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.514529 | 0.034917   | 14.74x  | -93.21%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.525092 | 0.100865   | 5.21x   | -80.79%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.525092 | 0.035256   | 14.89x  | -93.29%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.525092 | 0.035155   | 14.94x  | -93.31%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.726606 | 0.231032   | 3.15x   | -68.20%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.726606 | 0.038261   | 18.99x  | -94.73%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.726606 | 0.035602   | 20.41x  | -95.10%        |
</details>
<!-- PERF_RESULTS_END -->
