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
| normal        | scalar | 0.197744 | 0.043501   | 4.55x   | -78.00%        |
| normal        | sse42  | 0.197744 | 0.021745   | 9.09x   | -89.00%        |
| normal        | avx2   | 0.197744 | 0.021496   | 9.20x   | -89.13%        |
| soft_light    | scalar | 0.274421 | 0.053890   | 5.09x   | -80.36%        |
| soft_light    | sse42  | 0.274421 | 0.023863   | 11.50x  | -91.30%        |
| soft_light    | avx2   | 0.274421 | 0.023118   | 11.87x  | -91.58%        |
| lighten_only  | scalar | 0.197696 | 0.054865   | 3.60x   | -72.25%        |
| lighten_only  | sse42  | 0.197696 | 0.023638   | 8.36x   | -88.04%        |
| lighten_only  | avx2   | 0.197696 | 0.023211   | 8.52x   | -88.26%        |
| screen        | scalar | 0.210168 | 0.049706   | 4.23x   | -76.35%        |
| screen        | sse42  | 0.210168 | 0.024067   | 8.73x   | -88.55%        |
| screen        | avx2   | 0.210168 | 0.022526   | 9.33x   | -89.28%        |
| dodge         | scalar | 0.210161 | 0.052879   | 3.97x   | -74.84%        |
| dodge         | sse42  | 0.210161 | 0.024109   | 8.72x   | -88.53%        |
| dodge         | avx2   | 0.210161 | 0.023221   | 9.05x   | -88.95%        |
| addition      | scalar | 0.203964 | 0.078938   | 2.58x   | -61.30%        |
| addition      | sse42  | 0.203964 | 0.024045   | 8.48x   | -88.21%        |
| addition      | avx2   | 0.203964 | 0.022688   | 8.99x   | -88.88%        |
| darken_only   | scalar | 0.199378 | 0.055185   | 3.61x   | -72.32%        |
| darken_only   | sse42  | 0.199378 | 0.023432   | 8.51x   | -88.25%        |
| darken_only   | avx2   | 0.199378 | 0.022796   | 8.75x   | -88.57%        |
| multiply      | scalar | 0.205424 | 0.049611   | 4.14x   | -75.85%        |
| multiply      | sse42  | 0.205424 | 0.023467   | 8.75x   | -88.58%        |
| multiply      | avx2   | 0.205424 | 0.022516   | 9.12x   | -89.04%        |
| hard_light    | scalar | 0.302939 | 0.097438   | 3.11x   | -67.84%        |
| hard_light    | sse42  | 0.302939 | 0.024044   | 12.60x  | -92.06%        |
| hard_light    | avx2   | 0.302939 | 0.022909   | 13.22x  | -92.44%        |
| difference    | scalar | 0.272482 | 0.049036   | 5.56x   | -82.00%        |
| difference    | sse42  | 0.272482 | 0.023346   | 11.67x  | -91.43%        |
| difference    | avx2   | 0.272482 | 0.022278   | 12.23x  | -91.82%        |
| subtract      | scalar | 0.201906 | 0.053218   | 3.79x   | -73.64%        |
| subtract      | sse42  | 0.201906 | 0.024159   | 8.36x   | -88.03%        |
| subtract      | avx2   | 0.201906 | 0.022549   | 8.95x   | -88.83%        |
| grain_extract | scalar | 0.208158 | 0.064569   | 3.22x   | -68.98%        |
| grain_extract | sse42  | 0.208158 | 0.023461   | 8.87x   | -88.73%        |
| grain_extract | avx2   | 0.208158 | 0.022744   | 9.15x   | -89.07%        |
| grain_merge   | scalar | 0.209586 | 0.064876   | 3.23x   | -69.05%        |
| grain_merge   | sse42  | 0.209586 | 0.023360   | 8.97x   | -88.85%        |
| grain_merge   | avx2   | 0.209586 | 0.022750   | 9.21x   | -89.15%        |
| divide        | scalar | 0.212087 | 0.051542   | 4.11x   | -75.70%        |
| divide        | sse42  | 0.212087 | 0.023574   | 9.00x   | -88.88%        |
| divide        | avx2   | 0.212087 | 0.023034   | 9.21x   | -89.14%        |
| overlay       | scalar | 0.277370 | 0.092330   | 3.00x   | -66.71%        |
| overlay       | sse42  | 0.277370 | 0.023780   | 11.66x  | -91.43%        |
| overlay       | avx2   | 0.277370 | 0.023007   | 12.06x  | -91.71%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.006713 | 0.001625   | 4.13x   | -75.80%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.006713 | 0.001359   | 4.94x   | -79.75%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.006713 | 0.001484   | 4.52x   | -77.90%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.008879 | 0.001943   | 4.57x   | -78.12%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.008879 | 0.001447   | 6.13x   | -83.70%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.008879 | 0.001487   | 5.97x   | -83.25%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.007218 | 0.001970   | 3.66x   | -72.71%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.007218 | 0.001435   | 5.03x   | -80.12%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.007218 | 0.001491   | 4.84x   | -79.34%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.007382 | 0.001747   | 4.22x   | -76.33%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.007382 | 0.001399   | 5.28x   | -81.05%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.007382 | 0.001461   | 5.05x   | -80.20%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.007221 | 0.001900   | 3.80x   | -73.69%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.007221 | 0.001412   | 5.11x   | -80.44%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.007221 | 0.001517   | 4.76x   | -78.99%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.007316 | 0.002634   | 2.78x   | -63.99%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.007316 | 0.001367   | 5.35x   | -81.31%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.007316 | 0.001444   | 5.07x   | -80.26%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.006975 | 0.001922   | 3.63x   | -72.44%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.006975 | 0.001421   | 4.91x   | -79.63%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.006975 | 0.001508   | 4.63x   | -78.38%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.007255 | 0.001799   | 4.03x   | -75.20%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.007255 | 0.001414   | 5.13x   | -80.51%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.007255 | 0.001461   | 4.97x   | -79.87%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.008895 | 0.003036   | 2.93x   | -65.87%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.008895 | 0.001452   | 6.13x   | -83.67%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.008895 | 0.001486   | 5.99x   | -83.29%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.008782 | 0.001729   | 5.08x   | -80.32%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.008782 | 0.001427   | 6.15x   | -83.75%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.008782 | 0.001489   | 5.90x   | -83.05%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.007418 | 0.001857   | 3.99x   | -74.97%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.007418 | 0.001473   | 5.04x   | -80.15%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.007418 | 0.001461   | 5.08x   | -80.30%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.007150 | 0.002142   | 3.34x   | -70.04%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.007150 | 0.001438   | 4.97x   | -79.89%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.007150 | 0.001494   | 4.78x   | -79.10%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.007899 | 0.002265   | 3.49x   | -71.32%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.007899 | 0.001427   | 5.54x   | -81.93%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.007899 | 0.001460   | 5.41x   | -81.52%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.007098 | 0.001857   | 3.82x   | -73.84%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.007098 | 0.001413   | 5.02x   | -80.10%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.007098 | 0.001479   | 4.80x   | -79.17%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.008328 | 0.002896   | 2.88x   | -65.22%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.008328 | 0.001408   | 5.91x   | -83.09%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.008328 | 0.001471   | 5.66x   | -82.34%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003192 | 0.001327   | 2.40x   | -58.41%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003192 | 0.000194   | 16.44x  | -93.92%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003192 | 0.000155   | 20.64x  | -95.15%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.007056 | 0.001859   | 3.80x   | -73.66%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.007056 | 0.000233   | 30.30x  | -96.70%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.007056 | 0.000197   | 35.73x  | -97.20%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.005877 | 0.001755   | 3.35x   | -70.13%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.005877 | 0.000224   | 26.27x  | -96.19%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.005877 | 0.000201   | 29.18x  | -96.57%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.005974 | 0.001666   | 3.59x   | -72.12%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.005974 | 0.000226   | 26.38x  | -96.21%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.005974 | 0.000190   | 31.44x  | -96.82%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.006058 | 0.001803   | 3.36x   | -70.23%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.006058 | 0.000280   | 21.64x  | -95.38%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.006058 | 0.000196   | 30.96x  | -96.77%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.006208 | 0.002116   | 2.93x   | -65.91%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.006208 | 0.000271   | 22.93x  | -95.64%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.006208 | 0.000202   | 30.78x  | -96.75%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.005960 | 0.001819   | 3.28x   | -69.48%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.005960 | 0.000219   | 27.19x  | -96.32%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.005960 | 0.000194   | 30.77x  | -96.75%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.005887 | 0.002592   | 2.27x   | -55.97%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.005887 | 0.000307   | 19.18x  | -94.79%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.005887 | 0.000223   | 26.38x  | -96.21%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.008266 | 0.002760   | 3.00x   | -66.61%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.008266 | 0.000255   | 32.42x  | -96.92%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.008266 | 0.000197   | 41.86x  | -97.61%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.007576 | 0.001687   | 4.49x   | -77.74%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.007576 | 0.000222   | 34.09x  | -97.07%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.007576 | 0.000190   | 39.80x  | -97.49%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.005922 | 0.001691   | 3.50x   | -71.44%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.005922 | 0.000271   | 21.85x  | -95.42%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.005922 | 0.000201   | 29.41x  | -96.60%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.005566 | 0.001940   | 2.87x   | -65.15%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.005566 | 0.000226   | 24.57x  | -95.93%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.005566 | 0.000192   | 29.00x  | -96.55%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.005618 | 0.001954   | 2.87x   | -65.21%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.005618 | 0.000227   | 24.77x  | -95.96%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.005618 | 0.000189   | 29.68x  | -96.63%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.005773 | 0.001689   | 3.42x   | -70.75%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.005773 | 0.000235   | 24.59x  | -95.93%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.005773 | 0.000193   | 29.90x  | -96.66%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.006878 | 0.002575   | 2.67x   | -62.56%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.006878 | 0.000241   | 28.50x  | -96.49%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.006878 | 0.000191   | 35.96x  | -97.22%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.005340 | 0.000502   | 10.65x  | -90.61%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.005340 | 0.000172   | 31.10x  | -96.78%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.005340 | 0.000180   | 29.62x  | -96.62%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.008591 | 0.000673   | 12.77x  | -92.17%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.008591 | 0.000253   | 33.94x  | -97.05%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.008591 | 0.000213   | 40.42x  | -97.53%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.007191 | 0.000826   | 8.71x   | -88.52%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.007191 | 0.000227   | 31.61x  | -96.84%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.007191 | 0.000196   | 36.75x  | -97.28%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.007460 | 0.000615   | 12.12x  | -91.75%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.007460 | 0.000238   | 31.34x  | -96.81%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.007460 | 0.000203   | 36.82x  | -97.28%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.008366 | 0.000696   | 12.02x  | -91.68%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.008366 | 0.000270   | 31.04x  | -96.78%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.008366 | 0.000219   | 38.27x  | -97.39%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.007229 | 0.001640   | 4.41x   | -77.31%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.007229 | 0.000285   | 25.33x  | -96.05%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.007229 | 0.000208   | 34.80x  | -97.13%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.007610 | 0.000721   | 10.55x  | -90.52%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.007610 | 0.000222   | 34.28x  | -97.08%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.007610 | 0.000200   | 38.06x  | -97.37%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.007160 | 0.000583   | 12.28x  | -91.86%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.007160 | 0.000304   | 23.55x  | -95.75%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.007160 | 0.000199   | 36.04x  | -97.23%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.009075 | 0.001942   | 4.67x   | -78.60%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.009075 | 0.000263   | 34.52x  | -97.10%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.009075 | 0.000218   | 41.69x  | -97.60%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.009202 | 0.000608   | 15.12x  | -93.39%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.009202 | 0.000230   | 40.02x  | -97.50%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.009202 | 0.000196   | 46.94x  | -97.87%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.007666 | 0.000760   | 10.09x  | -90.09%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.007666 | 0.000271   | 28.28x  | -96.46%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.007666 | 0.000206   | 37.23x  | -97.31%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.007655 | 0.001086   | 7.05x   | -85.81%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.007655 | 0.000251   | 30.51x  | -96.72%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.007655 | 0.000199   | 38.47x  | -97.40%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.007627 | 0.001064   | 7.17x   | -86.05%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.007627 | 0.000249   | 30.59x  | -96.73%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.007627 | 0.000201   | 38.03x  | -97.37%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.007849 | 0.000681   | 11.53x  | -91.32%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.007849 | 0.000246   | 31.97x  | -96.87%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.007849 | 0.000210   | 37.34x  | -97.32%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.009024 | 0.001829   | 4.94x   | -79.74%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.009024 | 0.000276   | 32.75x  | -96.95%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.009024 | 0.000208   | 43.37x  | -97.69%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.004633 | 0.000640   | 7.24x   | -86.19%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.004633 | 0.000299   | 15.52x  | -93.56%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.004633 | 0.000227   | 20.43x  | -95.11%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.007597 | 0.000809   | 9.39x   | -89.35%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.007597 | 0.000368   | 20.62x  | -95.15%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.007597 | 0.000280   | 27.17x  | -96.32%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.005629 | 0.000898   | 6.27x   | -84.05%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.005629 | 0.000369   | 15.25x  | -93.44%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.005629 | 0.000261   | 21.56x  | -95.36%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.005889 | 0.000771   | 7.64x   | -86.91%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.005889 | 0.000368   | 16.00x  | -93.75%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.005889 | 0.000255   | 23.13x  | -95.68%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.005573 | 0.000821   | 6.79x   | -85.27%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.005573 | 0.000364   | 15.29x  | -93.46%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.005573 | 0.000261   | 21.37x  | -95.32%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.005897 | 0.001466   | 4.02x   | -75.14%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.005897 | 0.000397   | 14.86x  | -93.27%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.005897 | 0.000266   | 22.16x  | -95.49%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.005718 | 0.000988   | 5.79x   | -82.72%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.005718 | 0.000376   | 15.22x  | -93.43%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.005718 | 0.000257   | 22.29x  | -95.51%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.005818 | 0.000710   | 8.19x   | -87.79%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.005818 | 0.000371   | 15.66x  | -93.62%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.005818 | 0.000266   | 21.91x  | -95.44%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.007478 | 0.001965   | 3.80x   | -73.72%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.007478 | 0.000377   | 19.83x  | -94.96%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.007478 | 0.000267   | 28.01x  | -96.43%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.007445 | 0.000746   | 9.97x   | -89.97%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.007445 | 0.000392   | 18.99x  | -94.73%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.007445 | 0.000252   | 29.52x  | -96.61%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.005783 | 0.000947   | 6.11x   | -83.63%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.005783 | 0.000385   | 15.03x  | -93.35%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.005783 | 0.000261   | 22.16x  | -95.49%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.005545 | 0.001183   | 4.69x   | -78.67%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.005545 | 0.000371   | 14.96x  | -93.31%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.005545 | 0.000256   | 21.63x  | -95.38%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.005637 | 0.001128   | 5.00x   | -79.99%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.005637 | 0.000363   | 15.52x  | -93.56%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.005637 | 0.000258   | 21.86x  | -95.43%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.005782 | 0.000791   | 7.31x   | -86.33%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.005782 | 0.000372   | 15.56x  | -93.57%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.005782 | 0.000275   | 21.00x  | -95.24%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.007329 | 0.001845   | 3.97x   | -74.83%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.007329 | 0.000350   | 20.92x  | -95.22%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.007329 | 0.000264   | 27.79x  | -96.40%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.033997 | 0.006396   | 5.32x   | -81.19%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.033997 | 0.005386   | 6.31x   | -84.16%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.033997 | 0.005735   | 5.93x   | -83.13%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.044691 | 0.007389   | 6.05x   | -83.47%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.044691 | 0.005628   | 7.94x   | -87.41%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.044691 | 0.005891   | 7.59x   | -86.82%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.037611 | 0.007746   | 4.86x   | -79.40%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.037611 | 0.005630   | 6.68x   | -85.03%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.037611 | 0.005844   | 6.44x   | -84.46%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.038953 | 0.007241   | 5.38x   | -81.41%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.038953 | 0.005801   | 6.71x   | -85.11%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.038953 | 0.006032   | 6.46x   | -84.51%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.038668 | 0.007482   | 5.17x   | -80.65%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.038668 | 0.005642   | 6.85x   | -85.41%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.038668 | 0.005879   | 6.58x   | -84.80%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.037099 | 0.010597   | 3.50x   | -71.44%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.037099 | 0.005582   | 6.65x   | -84.95%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.037099 | 0.005892   | 6.30x   | -84.12%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.038016 | 0.007709   | 4.93x   | -79.72%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.038016 | 0.005783   | 6.57x   | -84.79%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.038016 | 0.006013   | 6.32x   | -84.18%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.038680 | 0.007324   | 5.28x   | -81.07%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.038680 | 0.005842   | 6.62x   | -84.90%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.038680 | 0.006001   | 6.45x   | -84.49%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.048071 | 0.012699   | 3.79x   | -73.58%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.048071 | 0.005971   | 8.05x   | -87.58%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.048071 | 0.005942   | 8.09x   | -87.64%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.045345 | 0.007298   | 6.21x   | -83.90%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.045345 | 0.005628   | 8.06x   | -87.59%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.045345 | 0.006012   | 7.54x   | -86.74%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.038016 | 0.007447   | 5.10x   | -80.41%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.038016 | 0.005606   | 6.78x   | -85.25%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.038016 | 0.005822   | 6.53x   | -84.68%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.037786 | 0.008790   | 4.30x   | -76.74%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.037786 | 0.005572   | 6.78x   | -85.25%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.037786 | 0.005856   | 6.45x   | -84.50%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.037729 | 0.008650   | 4.36x   | -77.07%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.037729 | 0.005582   | 6.76x   | -85.20%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.037729 | 0.005984   | 6.30x   | -84.14%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.039477 | 0.007632   | 5.17x   | -80.67%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.039477 | 0.005738   | 6.88x   | -85.47%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.039477 | 0.006061   | 6.51x   | -84.65%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.045677 | 0.011623   | 3.93x   | -74.55%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.045677 | 0.005639   | 8.10x   | -87.65%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.045677 | 0.006298   | 7.25x   | -86.21%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.025401 | 0.005429   | 4.68x   | -78.63%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.025401 | 0.000779   | 32.60x  | -96.93%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.025401 | 0.000617   | 41.17x  | -97.57%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.037456 | 0.007237   | 5.18x   | -80.68%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.037456 | 0.000953   | 39.29x  | -97.46%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.037456 | 0.000770   | 48.64x  | -97.94%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.029367 | 0.007082   | 4.15x   | -75.89%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.029367 | 0.000916   | 32.05x  | -96.88%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.029367 | 0.000753   | 38.99x  | -97.44%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.029682 | 0.006620   | 4.48x   | -77.70%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.029682 | 0.000893   | 33.26x  | -96.99%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.029682 | 0.000753   | 39.40x  | -97.46%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.030389 | 0.006881   | 4.42x   | -77.36%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.030389 | 0.001033   | 29.42x  | -96.60%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.030389 | 0.000829   | 36.64x  | -97.27%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.029689 | 0.008370   | 3.55x   | -71.81%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.029689 | 0.001087   | 27.30x  | -96.34%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.029689 | 0.000781   | 38.00x  | -97.37%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.029275 | 0.007228   | 4.05x   | -75.31%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.029275 | 0.000933   | 31.38x  | -96.81%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.029275 | 0.000752   | 38.95x  | -97.43%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.029838 | 0.006911   | 4.32x   | -76.84%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.029838 | 0.000996   | 29.97x  | -96.66%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.029838 | 0.000757   | 39.41x  | -97.46%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.038624 | 0.011186   | 3.45x   | -71.04%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.038624 | 0.001040   | 37.12x  | -97.31%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.038624 | 0.000778   | 49.66x  | -97.99%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.036999 | 0.007012   | 5.28x   | -81.05%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.036999 | 0.000886   | 41.75x  | -97.60%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.036999 | 0.000753   | 49.16x  | -97.97%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.029853 | 0.006913   | 4.32x   | -76.84%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.029853 | 0.001143   | 26.11x  | -96.17%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.029853 | 0.000799   | 37.35x  | -97.32%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.030286 | 0.008190   | 3.70x   | -72.96%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.030286 | 0.000954   | 31.73x  | -96.85%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.030286 | 0.000758   | 39.98x  | -97.50%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.030847 | 0.008058   | 3.83x   | -73.88%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.030847 | 0.000906   | 34.06x  | -97.06%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.030847 | 0.000766   | 40.29x  | -97.52%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.030911 | 0.006583   | 4.70x   | -78.70%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.030911 | 0.000944   | 32.75x  | -96.95%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.030911 | 0.000770   | 40.15x  | -97.51%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.036713 | 0.010507   | 3.49x   | -71.38%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.036713 | 0.001039   | 35.32x  | -97.17%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.036713 | 0.000787   | 46.64x  | -97.86%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.029035 | 0.002519   | 11.52x  | -91.32%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.029035 | 0.000709   | 40.93x  | -97.56%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.029035 | 0.000772   | 37.61x  | -97.34%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.039956 | 0.003137   | 12.74x  | -92.15%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.039956 | 0.001000   | 39.94x  | -97.50%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.039956 | 0.000869   | 45.95x  | -97.82%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.032873 | 0.003617   | 9.09x   | -89.00%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.032873 | 0.000958   | 34.32x  | -97.09%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.032873 | 0.000996   | 33.01x  | -96.97%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.034693 | 0.003022   | 11.48x  | -91.29%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.034693 | 0.000959   | 36.18x  | -97.24%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.034693 | 0.000821   | 42.27x  | -97.63%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.035988 | 0.003356   | 10.72x  | -90.68%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.035988 | 0.001116   | 32.25x  | -96.90%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.035988 | 0.000927   | 38.81x  | -97.42%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.034126 | 0.007087   | 4.82x   | -79.23%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.034126 | 0.001122   | 30.41x  | -96.71%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.034126 | 0.000828   | 41.22x  | -97.57%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.033036 | 0.003377   | 9.78x   | -89.78%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.033036 | 0.000914   | 36.15x  | -97.23%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.033036 | 0.000800   | 41.28x  | -97.58%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.033741 | 0.002866   | 11.77x  | -91.51%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.033741 | 0.000895   | 37.69x  | -97.35%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.033741 | 0.000806   | 41.88x  | -97.61%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.043666 | 0.009098   | 4.80x   | -79.17%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.043666 | 0.001082   | 40.37x  | -97.52%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.043666 | 0.000884   | 49.37x  | -97.97%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.044163 | 0.003348   | 13.19x  | -92.42%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.044163 | 0.000980   | 45.05x  | -97.78%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.044163 | 0.000858   | 51.50x  | -98.06%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.033001 | 0.003553   | 9.29x   | -89.23%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.033001 | 0.001092   | 30.23x  | -96.69%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.033001 | 0.000824   | 40.04x  | -97.50%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.034308 | 0.004772   | 7.19x   | -86.09%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.034308 | 0.001018   | 33.71x  | -97.03%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.034308 | 0.000833   | 41.17x  | -97.57%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.033289 | 0.004775   | 6.97x   | -85.65%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.033289 | 0.000991   | 33.59x  | -97.02%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.033289 | 0.000816   | 40.82x  | -97.55%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.034587 | 0.003177   | 10.89x  | -90.81%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.034587 | 0.000987   | 35.06x  | -97.15%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.034587 | 0.000850   | 40.67x  | -97.54%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.042082 | 0.007802   | 5.39x   | -81.46%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.042082 | 0.001094   | 38.45x  | -97.40%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.042082 | 0.000869   | 48.41x  | -97.93%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.021291 | 0.002567   | 8.29x   | -87.94%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.021291 | 0.001342   | 15.87x  | -93.70%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.021291 | 0.000987   | 21.58x  | -95.37%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.031208 | 0.003311   | 9.43x   | -89.39%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.031208 | 0.001512   | 20.64x  | -95.16%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.031208 | 0.001069   | 29.19x  | -96.57%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.025133 | 0.003566   | 7.05x   | -85.81%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.025133 | 0.001484   | 16.93x  | -94.09%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.025133 | 0.001207   | 20.83x  | -95.20%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.026558 | 0.003132   | 8.48x   | -88.21%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.026558 | 0.001440   | 18.44x  | -94.58%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.026558 | 0.001099   | 24.17x  | -95.86%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.025453 | 0.003326   | 7.65x   | -86.93%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.025453 | 0.001490   | 17.09x  | -94.15%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.025453 | 0.001081   | 23.56x  | -95.75%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.024039 | 0.005761   | 4.17x   | -76.04%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.024039 | 0.001545   | 15.56x  | -93.57%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.024039 | 0.001105   | 21.74x  | -95.40%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.024616 | 0.003725   | 6.61x   | -84.87%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.024616 | 0.001518   | 16.22x  | -93.83%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.024616 | 0.001247   | 19.74x  | -94.93%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.025025 | 0.002860   | 8.75x   | -88.57%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.025025 | 0.001533   | 16.32x  | -93.87%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.025025 | 0.001171   | 21.37x  | -95.32%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.032948 | 0.007738   | 4.26x   | -76.51%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.032948 | 0.001735   | 18.99x  | -94.73%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.032948 | 0.001138   | 28.95x  | -96.55%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.033991 | 0.003069   | 11.08x  | -90.97%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.033991 | 0.001533   | 22.18x  | -95.49%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.033991 | 0.001189   | 28.59x  | -96.50%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.025266 | 0.004098   | 6.17x   | -83.78%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.025266 | 0.001621   | 15.58x  | -93.58%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.025266 | 0.001186   | 21.31x  | -95.31%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.025284 | 0.004632   | 5.46x   | -81.68%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.025284 | 0.001461   | 17.30x  | -94.22%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.025284 | 0.001047   | 24.16x  | -95.86%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.024404 | 0.004640   | 5.26x   | -80.99%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.024404 | 0.001457   | 16.75x  | -94.03%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.024404 | 0.001065   | 22.92x  | -95.64%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.025794 | 0.003266   | 7.90x   | -87.34%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.025794 | 0.001541   | 16.74x  | -94.03%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.025794 | 0.001258   | 20.50x  | -95.12%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.032915 | 0.007386   | 4.46x   | -77.56%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.032915 | 0.001435   | 22.93x  | -95.64%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.032915 | 0.001045   | 31.50x  | -96.83%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.101193 | 0.026011   | 3.89x   | -74.30%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.101193 | 0.022215   | 4.56x   | -78.05%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.101193 | 0.023669   | 4.28x   | -76.61%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.131524 | 0.029993   | 4.39x   | -77.20%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.131524 | 0.022939   | 5.73x   | -82.56%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.131524 | 0.028016   | 4.69x   | -78.70%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.110890 | 0.031694   | 3.50x   | -71.42%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.110890 | 0.023727   | 4.67x   | -78.60%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.110890 | 0.029136   | 3.81x   | -73.73%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.106990 | 0.033043   | 3.24x   | -69.12%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.106990 | 0.025428   | 4.21x   | -76.23%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.106990 | 0.023717   | 4.51x   | -77.83%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.108484 | 0.030967   | 3.50x   | -71.45%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.108484 | 0.023118   | 4.69x   | -78.69%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.108484 | 0.024504   | 4.43x   | -77.41%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.106670 | 0.042188   | 2.53x   | -60.45%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.106670 | 0.024132   | 4.42x   | -77.38%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.106670 | 0.025058   | 4.26x   | -76.51%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.107143 | 0.032253   | 3.32x   | -69.90%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.107143 | 0.022644   | 4.73x   | -78.87%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.107143 | 0.023615   | 4.54x   | -77.96%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.109701 | 0.028273   | 3.88x   | -74.23%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.109701 | 0.022657   | 4.84x   | -79.35%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.109701 | 0.023639   | 4.64x   | -78.45%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.144134 | 0.049398   | 2.92x   | -65.73%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.144134 | 0.024309   | 5.93x   | -83.13%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.144134 | 0.024287   | 5.93x   | -83.15%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.135466 | 0.028280   | 4.79x   | -79.12%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.135466 | 0.022523   | 6.01x   | -83.37%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.135466 | 0.023338   | 5.80x   | -82.77%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.106423 | 0.028694   | 3.71x   | -73.04%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.106423 | 0.022576   | 4.71x   | -78.79%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.106423 | 0.024003   | 4.43x   | -77.45%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.109297 | 0.035442   | 3.08x   | -67.57%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.109297 | 0.022591   | 4.84x   | -79.33%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.109297 | 0.023581   | 4.63x   | -78.42%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.112751 | 0.034590   | 3.26x   | -69.32%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.112751 | 0.022555   | 5.00x   | -80.00%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.112751 | 0.024288   | 4.64x   | -78.46%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.112414 | 0.029607   | 3.80x   | -73.66%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.112414 | 0.022725   | 4.95x   | -79.78%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.112414 | 0.023858   | 4.71x   | -78.78%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.136684 | 0.046778   | 2.92x   | -65.78%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.136684 | 0.022732   | 6.01x   | -83.37%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.136684 | 0.027007   | 5.06x   | -80.24%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.075500 | 0.022005   | 3.43x   | -70.85%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.075500 | 0.003096   | 24.38x  | -95.90%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.075500 | 0.002520   | 29.96x  | -96.66%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.107165 | 0.027786   | 3.86x   | -74.07%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.107165 | 0.003951   | 27.13x  | -96.31%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.107165 | 0.003124   | 34.30x  | -97.08%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.081288 | 0.028959   | 2.81x   | -64.37%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.081288 | 0.003613   | 22.50x  | -95.56%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.081288 | 0.003019   | 26.93x  | -96.29%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.081422 | 0.025893   | 3.14x   | -68.20%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.081422 | 0.003573   | 22.79x  | -95.61%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.081422 | 0.003019   | 26.97x  | -96.29%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.080345 | 0.027936   | 2.88x   | -65.23%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.080345 | 0.004004   | 20.07x  | -95.02%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.080345 | 0.003227   | 24.90x  | -95.98%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.079935 | 0.034083   | 2.35x   | -57.36%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.079935 | 0.004433   | 18.03x  | -94.45%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.079935 | 0.003157   | 25.32x  | -96.05%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.076193 | 0.027795   | 2.74x   | -63.52%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.076193 | 0.003510   | 21.71x  | -95.39%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.076193 | 0.003031   | 25.14x  | -96.02%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.080968 | 0.027793   | 2.91x   | -65.67%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.080968 | 0.003664   | 22.10x  | -95.47%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.080968 | 0.003060   | 26.46x  | -96.22%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.118122 | 0.044231   | 2.67x   | -62.55%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.118122 | 0.004066   | 29.05x  | -96.56%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.118122 | 0.003126   | 37.78x  | -97.35%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.110073 | 0.027609   | 3.99x   | -74.92%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.110073 | 0.003709   | 29.68x  | -96.63%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.110073 | 0.003145   | 35.00x  | -97.14%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.081501 | 0.027198   | 3.00x   | -66.63%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.081501 | 0.004371   | 18.65x  | -94.64%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.081501 | 0.003165   | 25.75x  | -96.12%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.083287 | 0.031426   | 2.65x   | -62.27%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.083287 | 0.003665   | 22.72x  | -95.60%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.083287 | 0.003046   | 27.34x  | -96.34%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.081888 | 0.031455   | 2.60x   | -61.59%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.081888 | 0.003636   | 22.52x  | -95.56%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.081888 | 0.003035   | 26.98x  | -96.29%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.083404 | 0.026665   | 3.13x   | -68.03%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.083404 | 0.003749   | 22.24x  | -95.50%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.083404 | 0.003074   | 27.13x  | -96.31%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.109068 | 0.041401   | 2.63x   | -62.04%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.109068 | 0.003882   | 28.09x  | -96.44%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.109068 | 0.003092   | 35.28x  | -97.17%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.085360 | 0.009126   | 9.35x   | -89.31%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.085360 | 0.002984   | 28.60x  | -96.50%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.085360 | 0.003166   | 26.97x  | -96.29%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.119128 | 0.011516   | 10.34x  | -90.33%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.119128 | 0.004035   | 29.53x  | -96.61%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.119128 | 0.003389   | 35.15x  | -97.16%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.091376 | 0.012056   | 7.58x   | -86.81%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.091376 | 0.003607   | 25.33x  | -96.05%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.091376 | 0.003203   | 28.53x  | -96.49%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.096368 | 0.010982   | 8.77x   | -88.60%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.096368 | 0.003858   | 24.98x  | -96.00%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.096368 | 0.003452   | 27.91x  | -96.42%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.095293 | 0.011754   | 8.11x   | -87.67%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.095293 | 0.004311   | 22.10x  | -95.48%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.095293 | 0.003660   | 26.04x  | -96.16%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.094856 | 0.027163   | 3.49x   | -71.36%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.094856 | 0.004421   | 21.45x  | -95.34%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.094856 | 0.003435   | 27.62x  | -96.38%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.094192 | 0.014570   | 6.47x   | -84.53%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.094192 | 0.005383   | 17.50x  | -94.29%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.094192 | 0.004033   | 23.36x  | -95.72%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.101688 | 0.010704   | 9.50x   | -89.47%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.101688 | 0.003971   | 25.60x  | -96.09%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.101688 | 0.003422   | 29.72x  | -96.64%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.129788 | 0.031646   | 4.10x   | -75.62%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.129788 | 0.004311   | 30.10x  | -96.68%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.129788 | 0.003755   | 34.57x  | -97.11%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.121895 | 0.010193   | 11.96x  | -91.64%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.121895 | 0.003799   | 32.08x  | -96.88%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.121895 | 0.003488   | 34.95x  | -97.14%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.099616 | 0.012805   | 7.78x   | -87.15%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.099616 | 0.004469   | 22.29x  | -95.51%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.099616 | 0.003457   | 28.82x  | -96.53%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.096199 | 0.017530   | 5.49x   | -81.78%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.096199 | 0.004066   | 23.66x  | -95.77%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.096199 | 0.003450   | 27.89x  | -96.41%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.097413 | 0.017657   | 5.52x   | -81.87%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.097413 | 0.004283   | 22.74x  | -95.60%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.097413 | 0.003443   | 28.29x  | -96.47%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.097499 | 0.011483   | 8.49x   | -88.22%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.097499 | 0.004157   | 23.46x  | -95.74%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.097499 | 0.003648   | 26.73x  | -96.26%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.124999 | 0.029786   | 4.20x   | -76.17%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.124999 | 0.005835   | 21.42x  | -95.33%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.124999 | 0.004099   | 30.50x  | -96.72%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.068407 | 0.010175   | 6.72x   | -85.13%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.068407 | 0.004951   | 13.82x  | -92.76%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.068407 | 0.003840   | 17.81x  | -94.39%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.103137 | 0.013395   | 7.70x   | -87.01%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.103137 | 0.005793   | 17.80x  | -94.38%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.103137 | 0.004533   | 22.75x  | -95.60%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.074078 | 0.014297   | 5.18x   | -80.70%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.074078 | 0.005832   | 12.70x  | -92.13%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.074078 | 0.004318   | 17.16x  | -94.17%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.078502 | 0.013655   | 5.75x   | -82.61%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.078502 | 0.005838   | 13.45x  | -92.56%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.078502 | 0.004452   | 17.63x  | -94.33%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.078384 | 0.013542   | 5.79x   | -82.72%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.078384 | 0.006008   | 13.05x  | -92.34%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.078384 | 0.004480   | 17.50x  | -94.28%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.073431 | 0.022920   | 3.20x   | -68.79%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.073431 | 0.006200   | 11.84x  | -91.56%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.073431 | 0.004498   | 16.33x  | -93.88%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.077998 | 0.014818   | 5.26x   | -81.00%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.077998 | 0.005946   | 13.12x  | -92.38%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.077998 | 0.004389   | 17.77x  | -94.37%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.076211 | 0.011590   | 6.58x   | -84.79%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.076211 | 0.005935   | 12.84x  | -92.21%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.076211 | 0.004460   | 17.09x  | -94.15%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.109993 | 0.031107   | 3.54x   | -71.72%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.109993 | 0.005737   | 19.17x  | -94.78%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.109993 | 0.004629   | 23.76x  | -95.79%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.107699 | 0.012301   | 8.76x   | -88.58%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.107699 | 0.006044   | 17.82x  | -94.39%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.107699 | 0.004391   | 24.53x  | -95.92%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.074629 | 0.016095   | 4.64x   | -78.43%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.074629 | 0.006146   | 12.14x  | -91.76%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.074629 | 0.004535   | 16.46x  | -93.92%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.076893 | 0.018114   | 4.24x   | -76.44%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.076893 | 0.005777   | 13.31x  | -92.49%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.076893 | 0.004352   | 17.67x  | -94.34%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.075325 | 0.017993   | 4.19x   | -76.11%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.075325 | 0.006032   | 12.49x  | -91.99%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.075325 | 0.005166   | 14.58x  | -93.14%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.077434 | 0.012706   | 6.09x   | -83.59%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.077434 | 0.005873   | 13.18x  | -92.42%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.077434 | 0.004280   | 18.09x  | -94.47%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.102289 | 0.029501   | 3.47x   | -71.16%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.102289 | 0.005611   | 18.23x  | -94.51%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.102289 | 0.004700   | 21.76x  | -95.41%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.388261 | 0.106116   | 3.66x   | -72.67%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.388261 | 0.087426   | 4.44x   | -77.48%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.388261 | 0.091443   | 4.25x   | -76.45%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.504603 | 0.121780   | 4.14x   | -75.87%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.504603 | 0.090370   | 5.58x   | -82.09%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.504603 | 0.094576   | 5.34x   | -81.26%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.379672 | 0.126715   | 3.00x   | -66.63%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.379672 | 0.089283   | 4.25x   | -76.48%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.379672 | 0.108356   | 3.50x   | -71.46%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.418168 | 0.115455   | 3.62x   | -72.39%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.418168 | 0.111052   | 3.77x   | -73.44%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.418168 | 0.096409   | 4.34x   | -76.95%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.427821 | 0.126966   | 3.37x   | -70.32%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.427821 | 0.093497   | 4.58x   | -78.15%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.427821 | 0.098921   | 4.32x   | -76.88%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.423761 | 0.169415   | 2.50x   | -60.02%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.423761 | 0.087642   | 4.84x   | -79.32%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.423761 | 0.093446   | 4.53x   | -77.95%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.393323 | 0.126919   | 3.10x   | -67.73%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.393323 | 0.093196   | 4.22x   | -76.31%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.393323 | 0.096221   | 4.09x   | -75.54%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.414829 | 0.113887   | 3.64x   | -72.55%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.414829 | 0.093064   | 4.46x   | -77.57%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.414829 | 0.096848   | 4.28x   | -76.65%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.577451 | 0.200540   | 2.88x   | -65.27%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.577451 | 0.090947   | 6.35x   | -84.25%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.577451 | 0.095731   | 6.03x   | -83.42%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.511650 | 0.113747   | 4.50x   | -77.77%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.511650 | 0.092098   | 5.56x   | -82.00%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.511650 | 0.095410   | 5.36x   | -81.35%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.403292 | 0.117384   | 3.44x   | -70.89%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.403292 | 0.090052   | 4.48x   | -77.67%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.403292 | 0.095625   | 4.22x   | -76.29%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.422900 | 0.142735   | 2.96x   | -66.25%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.422900 | 0.092114   | 4.59x   | -78.22%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.422900 | 0.103675   | 4.08x   | -75.48%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.423327 | 0.146116   | 2.90x   | -65.48%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.423327 | 0.091100   | 4.65x   | -78.48%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.423327 | 0.098969   | 4.28x   | -76.62%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.417096 | 0.124995   | 3.34x   | -70.03%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.417096 | 0.091445   | 4.56x   | -78.08%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.417096 | 0.097107   | 4.30x   | -76.72%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.542896 | 0.188562   | 2.88x   | -65.27%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.542896 | 0.091015   | 5.96x   | -83.24%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.542896 | 0.097018   | 5.60x   | -82.13%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.305744 | 0.086413   | 3.54x   | -71.74%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.305744 | 0.012558   | 24.35x  | -95.89%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.305744 | 0.009969   | 30.67x  | -96.74%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.410946 | 0.109027   | 3.77x   | -73.47%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.410946 | 0.015129   | 27.16x  | -96.32%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.410946 | 0.012348   | 33.28x  | -97.00%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.279104 | 0.112802   | 2.47x   | -59.58%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.279104 | 0.016731   | 16.68x  | -94.01%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.279104 | 0.012293   | 22.70x  | -95.60%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.298543 | 0.104150   | 2.87x   | -65.11%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.298543 | 0.014300   | 20.88x  | -95.21%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.298543 | 0.012082   | 24.71x  | -95.95%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.299232 | 0.108629   | 2.75x   | -63.70%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.299232 | 0.015856   | 18.87x  | -94.70%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.299232 | 0.013282   | 22.53x  | -95.56%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.309515 | 0.146970   | 2.11x   | -52.52%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.309515 | 0.018472   | 16.76x  | -94.03%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.309515 | 0.013364   | 23.16x  | -95.68%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.296323 | 0.114111   | 2.60x   | -61.49%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.296323 | 0.014625   | 20.26x  | -95.06%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.296323 | 0.012442   | 23.82x  | -95.80%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.304729 | 0.110089   | 2.77x   | -63.87%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.304729 | 0.014720   | 20.70x  | -95.17%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.304729 | 0.012335   | 24.70x  | -95.95%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.468702 | 0.178057   | 2.63x   | -62.01%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.468702 | 0.016799   | 27.90x  | -96.42%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.468702 | 0.012348   | 37.96x  | -97.37%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.429388 | 0.105918   | 4.05x   | -75.33%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.429388 | 0.014220   | 30.20x  | -96.69%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.429388 | 0.012083   | 35.54x  | -97.19%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.289350 | 0.108909   | 2.66x   | -62.36%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.289350 | 0.017844   | 16.22x  | -93.83%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.289350 | 0.013026   | 22.21x  | -95.50%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.303143 | 0.127808   | 2.37x   | -57.84%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.303143 | 0.014535   | 20.86x  | -95.21%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.303143 | 0.012157   | 24.94x  | -95.99%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.301935 | 0.137989   | 2.19x   | -54.30%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.301935 | 0.014991   | 20.14x  | -95.03%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.301935 | 0.012292   | 24.56x  | -95.93%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.309432 | 0.107501   | 2.88x   | -65.26%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.309432 | 0.015209   | 20.35x  | -95.08%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.309432 | 0.012311   | 25.13x  | -96.02%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.422398 | 0.167106   | 2.53x   | -60.44%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.422398 | 0.015588   | 27.10x  | -96.31%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.422398 | 0.012165   | 34.72x  | -97.12%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.338620 | 0.037620   | 9.00x   | -88.89%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.338620 | 0.017208   | 19.68x  | -94.92%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.338620 | 0.019077   | 17.75x  | -94.37%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.465480 | 0.049708   | 9.36x   | -89.32%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.465480 | 0.021738   | 21.41x  | -95.33%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.465480 | 0.019440   | 23.94x  | -95.82%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.344544 | 0.052429   | 6.57x   | -84.78%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.344544 | 0.020102   | 17.14x  | -94.17%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.344544 | 0.018968   | 18.16x  | -94.49%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.360529 | 0.044662   | 8.07x   | -87.61%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.360529 | 0.020479   | 17.60x  | -94.32%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.360529 | 0.019078   | 18.90x  | -94.71%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.356029 | 0.049682   | 7.17x   | -86.05%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.356029 | 0.024135   | 14.75x  | -93.22%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.356029 | 0.022786   | 15.62x  | -93.60%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.358787 | 0.108616   | 3.30x   | -69.73%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.358787 | 0.023166   | 15.49x  | -93.54%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.358787 | 0.019393   | 18.50x  | -94.59%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.334007 | 0.052010   | 6.42x   | -84.43%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.334007 | 0.019229   | 17.37x  | -94.24%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.334007 | 0.018053   | 18.50x  | -94.60%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.348715 | 0.044396   | 7.85x   | -87.27%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.348715 | 0.020925   | 16.67x  | -94.00%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.348715 | 0.018649   | 18.70x  | -94.65%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.505848 | 0.129311   | 3.91x   | -74.44%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.505848 | 0.022309   | 22.67x  | -95.59%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.505848 | 0.019979   | 25.32x  | -96.05%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.460187 | 0.047336   | 9.72x   | -89.71%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.460187 | 0.021560   | 21.34x  | -95.32%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.460187 | 0.020157   | 22.83x  | -95.62%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.360895 | 0.053971   | 6.69x   | -85.05%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.360895 | 0.023648   | 15.26x  | -93.45%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.360895 | 0.019251   | 18.75x  | -94.67%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.357038 | 0.075236   | 4.75x   | -78.93%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.357038 | 0.021538   | 16.58x  | -93.97%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.357038 | 0.019946   | 17.90x  | -94.41%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.367450 | 0.074617   | 4.92x   | -79.69%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.367450 | 0.020893   | 17.59x  | -94.31%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.367450 | 0.019150   | 19.19x  | -94.79%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.364925 | 0.049431   | 7.38x   | -86.45%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.364925 | 0.021414   | 17.04x  | -94.13%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.364925 | 0.019385   | 18.83x  | -94.69%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.461821 | 0.121774   | 3.79x   | -73.63%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.461821 | 0.021435   | 21.55x  | -95.36%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.461821 | 0.019075   | 24.21x  | -95.87%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.266780 | 0.047552   | 5.61x   | -82.18%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.266780 | 0.026530   | 10.06x  | -90.06%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.266780 | 0.022740   | 11.73x  | -91.48%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.375083 | 0.060234   | 6.23x   | -83.94%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.375083 | 0.029723   | 12.62x  | -92.08%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.375083 | 0.024356   | 15.40x  | -93.51%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.262926 | 0.063562   | 4.14x   | -75.83%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.262926 | 0.029759   | 8.84x   | -88.68%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.262926 | 0.024592   | 10.69x  | -90.65%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.285339 | 0.059053   | 4.83x   | -79.30%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.285339 | 0.029809   | 9.57x   | -89.55%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.285339 | 0.024328   | 11.73x  | -91.47%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.278519 | 0.059516   | 4.68x   | -78.63%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.278519 | 0.029709   | 9.38x   | -89.33%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.278519 | 0.023362   | 11.92x  | -91.61%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.265269 | 0.097123   | 2.73x   | -63.39%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.265269 | 0.032388   | 8.19x   | -87.79%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.265269 | 0.024951   | 10.63x  | -90.59%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.261170 | 0.065772   | 3.97x   | -74.82%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.261170 | 0.030688   | 8.51x   | -88.25%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.261170 | 0.024327   | 10.74x  | -90.69%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.280786 | 0.052798   | 5.32x   | -81.20%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.280786 | 0.029684   | 9.46x   | -89.43%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.280786 | 0.025030   | 11.22x  | -91.09%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.461177 | 0.133998   | 3.44x   | -70.94%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.461177 | 0.030215   | 15.26x  | -93.45%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.461177 | 0.024542   | 18.79x  | -94.68%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.373032 | 0.055406   | 6.73x   | -85.15%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.373032 | 0.030109   | 12.39x  | -91.93%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.373032 | 0.023821   | 15.66x  | -93.61%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.265034 | 0.069827   | 3.80x   | -73.65%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.265034 | 0.031321   | 8.46x   | -88.18%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.265034 | 0.024661   | 10.75x  | -90.70%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.277721 | 0.078796   | 3.52x   | -71.63%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.277721 | 0.029173   | 9.52x   | -89.50%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.277721 | 0.024263   | 11.45x  | -91.26%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.287384 | 0.079143   | 3.63x   | -72.46%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.287384 | 0.029200   | 9.84x   | -89.84%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.287384 | 0.024163   | 11.89x  | -91.59%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.281969 | 0.057740   | 4.88x   | -79.52%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.281969 | 0.030440   | 9.26x   | -89.20%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.281969 | 0.024065   | 11.72x  | -91.47%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.395857 | 0.126827   | 3.12x   | -67.96%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.395857 | 0.030183   | 13.12x  | -92.38%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.395857 | 0.024874   | 15.91x  | -93.72%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.087859 | 0.022686   | 3.87x   | -74.18%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.087859 | 0.019229   | 4.57x   | -78.11%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.087859 | 0.020351   | 4.32x   | -76.84%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.110914 | 0.026413   | 4.20x   | -76.19%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.110914 | 0.020415   | 5.43x   | -81.59%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.110914 | 0.021207   | 5.23x   | -80.88%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.087793 | 0.028566   | 3.07x   | -67.46%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.087793 | 0.020045   | 4.38x   | -77.17%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.087793 | 0.021016   | 4.18x   | -76.06%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.090672 | 0.026592   | 3.41x   | -70.67%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.090672 | 0.020116   | 4.51x   | -77.81%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.090672 | 0.021174   | 4.28x   | -76.65%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.093637 | 0.026685   | 3.51x   | -71.50%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.093637 | 0.020113   | 4.66x   | -78.52%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.093637 | 0.021561   | 4.34x   | -76.97%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.086739 | 0.037065   | 2.34x   | -57.27%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.086739 | 0.019955   | 4.35x   | -76.99%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.086739 | 0.021435   | 4.05x   | -75.29%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.088607 | 0.027526   | 3.22x   | -68.94%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.088607 | 0.019870   | 4.46x   | -77.58%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.088607 | 0.021221   | 4.18x   | -76.05%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.089913 | 0.024792   | 3.63x   | -72.43%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.089913 | 0.020686   | 4.35x   | -76.99%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.089913 | 0.021736   | 4.14x   | -75.83%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.123760 | 0.043249   | 2.86x   | -65.05%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.123760 | 0.020306   | 6.09x   | -83.59%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.123760 | 0.021333   | 5.80x   | -82.76%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.116193 | 0.024530   | 4.74x   | -78.89%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.116193 | 0.019628   | 5.92x   | -83.11%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.116193 | 0.020619   | 5.64x   | -82.25%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.084782 | 0.024954   | 3.40x   | -70.57%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.084782 | 0.019562   | 4.33x   | -76.93%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.084782 | 0.020687   | 4.10x   | -75.60%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.088141 | 0.030557   | 2.88x   | -65.33%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.088141 | 0.019795   | 4.45x   | -77.54%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.088141 | 0.020709   | 4.26x   | -76.51%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.089305 | 0.030352   | 2.94x   | -66.01%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.089305 | 0.019778   | 4.52x   | -77.85%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.089305 | 0.020863   | 4.28x   | -76.64%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.093619 | 0.025872   | 3.62x   | -72.36%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.093619 | 0.019976   | 4.69x   | -78.66%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.093619 | 0.020944   | 4.47x   | -77.63%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.112890 | 0.041389   | 2.73x   | -63.34%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.112890 | 0.020255   | 5.57x   | -82.06%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.112890 | 0.020888   | 5.40x   | -81.50%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.064610 | 0.022668   | 2.85x   | -64.92%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.064610 | 0.002767   | 23.35x  | -95.72%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.064610 | 0.002236   | 28.89x  | -96.54%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.100608 | 0.024690   | 4.07x   | -75.46%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.100608 | 0.003381   | 29.75x  | -96.64%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.100608 | 0.002749   | 36.59x  | -97.27%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.075569 | 0.025384   | 2.98x   | -66.41%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.075569 | 0.003312   | 22.82x  | -95.62%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.075569 | 0.002810   | 26.89x  | -96.28%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.083285 | 0.023903   | 3.48x   | -71.30%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.083285 | 0.003170   | 26.27x  | -96.19%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.083285 | 0.003132   | 26.60x  | -96.24%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.078864 | 0.024821   | 3.18x   | -68.53%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.078864 | 0.003522   | 22.39x  | -95.53%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.078864 | 0.002743   | 28.75x  | -96.52%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.074499 | 0.029877   | 2.49x   | -59.90%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.074499 | 0.003852   | 19.34x  | -94.83%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.074499 | 0.002839   | 26.24x  | -96.19%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.075249 | 0.024978   | 3.01x   | -66.81%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.075249 | 0.003141   | 23.96x  | -95.83%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.075249 | 0.002730   | 27.57x  | -96.37%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.078578 | 0.024044   | 3.27x   | -69.40%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.078578 | 0.003122   | 25.17x  | -96.03%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.078578 | 0.002696   | 29.15x  | -96.57%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.109312 | 0.038122   | 2.87x   | -65.13%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.109312 | 0.003587   | 30.48x  | -96.72%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.109312 | 0.002733   | 39.99x  | -97.50%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.103461 | 0.023158   | 4.47x   | -77.62%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.103461 | 0.003160   | 32.74x  | -96.95%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.103461 | 0.002663   | 38.86x  | -97.43%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.073732 | 0.022766   | 3.24x   | -69.12%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.073732 | 0.003820   | 19.30x  | -94.82%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.073732 | 0.002807   | 26.27x  | -96.19%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.078883 | 0.027455   | 2.87x   | -65.20%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.078883 | 0.003181   | 24.79x  | -95.97%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.078883 | 0.002720   | 29.00x  | -96.55%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.076555 | 0.027563   | 2.78x   | -64.00%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.076555 | 0.003322   | 23.04x  | -95.66%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.076555 | 0.002711   | 28.23x  | -96.46%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.077694 | 0.023614   | 3.29x   | -69.61%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.077694 | 0.003314   | 23.44x  | -95.73%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.077694 | 0.002677   | 29.02x  | -96.55%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.100545 | 0.036854   | 2.73x   | -63.35%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.100545 | 0.003472   | 28.96x  | -96.55%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.100545 | 0.002707   | 37.15x  | -97.31%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.072900 | 0.007136   | 10.22x  | -90.21%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.072900 | 0.002442   | 29.85x  | -96.65%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.072900 | 0.002704   | 26.96x  | -96.29%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.105609 | 0.010439   | 10.12x  | -90.12%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.105609 | 0.003617   | 29.20x  | -96.58%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.105609 | 0.003155   | 33.47x  | -97.01%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.083306 | 0.009961   | 8.36x   | -88.04%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.083306 | 0.003125   | 26.66x  | -96.25%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.083306 | 0.002781   | 29.95x  | -96.66%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.084461 | 0.008674   | 9.74x   | -89.73%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.084461 | 0.003235   | 26.10x  | -96.17%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.084461 | 0.002950   | 28.63x  | -96.51%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.084221 | 0.010096   | 8.34x   | -88.01%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.084221 | 0.003796   | 22.19x  | -95.49%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.084221 | 0.003350   | 25.14x  | -96.02%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.080092 | 0.023001   | 3.48x   | -71.28%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.080092 | 0.003770   | 21.24x  | -95.29%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.080092 | 0.002898   | 27.63x  | -96.38%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.086435 | 0.010003   | 8.64x   | -88.43%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.086435 | 0.003086   | 28.01x  | -96.43%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.086435 | 0.002757   | 31.35x  | -96.81%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.080723 | 0.008102   | 9.96x   | -89.96%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.080723 | 0.003110   | 25.96x  | -96.15%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.080723 | 0.002748   | 29.37x  | -96.60%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.117466 | 0.027476   | 4.28x   | -76.61%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.117466 | 0.003695   | 31.79x  | -96.85%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.117466 | 0.003183   | 36.90x  | -97.29%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.107939 | 0.008677   | 12.44x  | -91.96%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.107939 | 0.003270   | 33.01x  | -96.97%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.107939 | 0.002966   | 36.39x  | -97.25%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.079573 | 0.010505   | 7.57x   | -86.80%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.079573 | 0.003799   | 20.94x  | -95.23%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.079573 | 0.002828   | 28.14x  | -96.45%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.081310 | 0.014990   | 5.42x   | -81.56%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.081310 | 0.003580   | 22.71x  | -95.60%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.081310 | 0.003133   | 25.95x  | -96.15%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.083460 | 0.014902   | 5.60x   | -82.14%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.083460 | 0.003514   | 23.75x  | -95.79%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.083460 | 0.003008   | 27.75x  | -96.40%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.083965 | 0.009455   | 8.88x   | -88.74%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.083965 | 0.003410   | 24.62x  | -95.94%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.083965 | 0.002948   | 28.48x  | -96.49%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.115134 | 0.026052   | 4.42x   | -77.37%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.115134 | 0.003782   | 30.45x  | -96.72%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.115134 | 0.003220   | 35.76x  | -97.20%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.057943 | 0.009053   | 6.40x   | -84.38%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.057943 | 0.004387   | 13.21x  | -92.43%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.057943 | 0.003376   | 17.16x  | -94.17%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.090149 | 0.011660   | 7.73x   | -87.07%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.090149 | 0.004996   | 18.05x  | -94.46%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.090149 | 0.003931   | 22.93x  | -95.64%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.067518 | 0.012448   | 5.42x   | -81.56%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.067518 | 0.005200   | 12.98x  | -92.30%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.067518 | 0.003945   | 17.12x  | -94.16%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.074067 | 0.010891   | 6.80x   | -85.30%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.074067 | 0.005105   | 14.51x  | -93.11%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.074067 | 0.003943   | 18.78x  | -94.68%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.068659 | 0.011563   | 5.94x   | -83.16%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.068659 | 0.005065   | 13.56x  | -92.62%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.068659 | 0.003711   | 18.50x  | -94.59%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.064368 | 0.020813   | 3.09x   | -67.67%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.064368 | 0.005454   | 11.80x  | -91.53%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.064368 | 0.003973   | 16.20x  | -93.83%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.067266 | 0.013996   | 4.81x   | -79.19%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.067266 | 0.005168   | 13.02x  | -92.32%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.067266 | 0.003872   | 17.37x  | -94.24%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.069551 | 0.011403   | 6.10x   | -83.61%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.069551 | 0.005181   | 13.42x  | -92.55%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.069551 | 0.003847   | 18.08x  | -94.47%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.098495 | 0.027387   | 3.60x   | -72.19%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.098495 | 0.004974   | 19.80x  | -94.95%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.098495 | 0.003792   | 25.97x  | -96.15%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.091443 | 0.010323   | 8.86x   | -88.71%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.091443 | 0.005158   | 17.73x  | -94.36%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.091443 | 0.003822   | 23.93x  | -95.82%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.065501 | 0.013997   | 4.68x   | -78.63%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.065501 | 0.005457   | 12.00x  | -91.67%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.065501 | 0.004127   | 15.87x  | -93.70%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.071342 | 0.016565   | 4.31x   | -76.78%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.071342 | 0.005248   | 13.59x  | -92.64%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.071342 | 0.004005   | 17.81x  | -94.39%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.070736 | 0.016315   | 4.34x   | -76.94%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.070736 | 0.005264   | 13.44x  | -92.56%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.070736 | 0.004091   | 17.29x  | -94.22%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.072558 | 0.011714   | 6.19x   | -83.86%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.072558 | 0.005267   | 13.77x  | -92.74%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.072558 | 0.004056   | 17.89x  | -94.41%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.095480 | 0.026536   | 3.60x   | -72.21%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.095480 | 0.004992   | 19.13x  | -94.77%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.095480 | 0.004117   | 23.19x  | -95.69%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.189352 | 0.052515   | 3.61x   | -72.27%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.189352 | 0.043768   | 4.33x   | -76.89%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.189352 | 0.046158   | 4.10x   | -75.62%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.240707 | 0.062191   | 3.87x   | -74.16%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.240707 | 0.045909   | 5.24x   | -80.93%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.240707 | 0.048098   | 5.00x   | -80.02%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.189709 | 0.062396   | 3.04x   | -67.11%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.189709 | 0.046260   | 4.10x   | -75.62%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.189709 | 0.046791   | 4.05x   | -75.34%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.194202 | 0.055600   | 3.49x   | -71.37%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.194202 | 0.044279   | 4.39x   | -77.20%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.194202 | 0.046473   | 4.18x   | -76.07%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.193340 | 0.058606   | 3.30x   | -69.69%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.193340 | 0.044814   | 4.31x   | -76.82%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.193340 | 0.047812   | 4.04x   | -75.27%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.195872 | 0.082437   | 2.38x   | -57.91%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.195872 | 0.043885   | 4.46x   | -77.60%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.195872 | 0.046848   | 4.18x   | -76.08%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.188622 | 0.061605   | 3.06x   | -67.34%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.188622 | 0.044344   | 4.25x   | -76.49%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.188622 | 0.046234   | 4.08x   | -75.49%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.190744 | 0.055691   | 3.43x   | -70.80%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.190744 | 0.044751   | 4.26x   | -76.54%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.190744 | 0.046876   | 4.07x   | -75.42%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.258691 | 0.096430   | 2.68x   | -62.72%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.258691 | 0.044835   | 5.77x   | -82.67%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.258691 | 0.046846   | 5.52x   | -81.89%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.247724 | 0.057233   | 4.33x   | -76.90%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.247724 | 0.046415   | 5.34x   | -81.26%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.247724 | 0.046635   | 5.31x   | -81.17%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.203621 | 0.059400   | 3.43x   | -70.83%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.203621 | 0.044603   | 4.57x   | -78.10%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.203621 | 0.046728   | 4.36x   | -77.05%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.195938 | 0.073408   | 2.67x   | -62.53%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.195938 | 0.044580   | 4.40x   | -77.25%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.195938 | 0.046836   | 4.18x   | -76.10%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.195373 | 0.069845   | 2.80x   | -64.25%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.195373 | 0.045631   | 4.28x   | -76.64%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.195373 | 0.046753   | 4.18x   | -76.07%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.194549 | 0.058635   | 3.32x   | -69.86%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.194549 | 0.044873   | 4.34x   | -76.94%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.194549 | 0.050765   | 3.83x   | -73.91%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.249749 | 0.093590   | 2.67x   | -62.53%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.249749 | 0.045002   | 5.55x   | -81.98%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.249749 | 0.047284   | 5.28x   | -81.07%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.134668 | 0.044266   | 3.04x   | -67.13%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.134668 | 0.006208   | 21.69x  | -95.39%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.134668 | 0.004895   | 27.51x  | -96.37%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.197384 | 0.054406   | 3.63x   | -72.44%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.197384 | 0.007530   | 26.21x  | -96.19%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.197384 | 0.006032   | 32.72x  | -96.94%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.143360 | 0.057043   | 2.51x   | -60.21%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.143360 | 0.006950   | 20.63x  | -95.15%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.143360 | 0.005897   | 24.31x  | -95.89%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.151267 | 0.054203   | 2.79x   | -64.17%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.151267 | 0.007267   | 20.82x  | -95.20%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.151267 | 0.006006   | 25.19x  | -96.03%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.148284 | 0.053498   | 2.77x   | -63.92%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.148284 | 0.007837   | 18.92x  | -94.72%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.148284 | 0.006089   | 24.35x  | -95.89%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.141475 | 0.065338   | 2.17x   | -53.82%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.141475 | 0.008605   | 16.44x  | -93.92%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.141475 | 0.006167   | 22.94x  | -95.64%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.147517 | 0.056024   | 2.63x   | -62.02%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.147517 | 0.007035   | 20.97x  | -95.23%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.147517 | 0.005956   | 24.77x  | -95.96%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.149502 | 0.054607   | 2.74x   | -63.47%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.149502 | 0.007506   | 19.92x  | -94.98%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.149502 | 0.005989   | 24.96x  | -95.99%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.220303 | 0.086117   | 2.56x   | -60.91%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.220303 | 0.009000   | 24.48x  | -95.91%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.220303 | 0.006239   | 35.31x  | -97.17%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.208904 | 0.052370   | 3.99x   | -74.93%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.208904 | 0.006964   | 30.00x  | -96.67%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.208904 | 0.005908   | 35.36x  | -97.17%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.146671 | 0.054408   | 2.70x   | -62.91%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.146671 | 0.008598   | 17.06x  | -94.14%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.146671 | 0.006290   | 23.32x  | -95.71%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.155276 | 0.062204   | 2.50x   | -59.94%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.155276 | 0.007202   | 21.56x  | -95.36%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.155276 | 0.006011   | 25.83x  | -96.13%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.146942 | 0.061810   | 2.38x   | -57.94%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.146942 | 0.007177   | 20.47x  | -95.12%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.146942 | 0.005959   | 24.66x  | -95.94%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.147016 | 0.051932   | 2.83x   | -64.68%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.147016 | 0.007497   | 19.61x  | -94.90%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.147016 | 0.006011   | 24.46x  | -95.91%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.199580 | 0.081787   | 2.44x   | -59.02%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.199580 | 0.007610   | 26.22x  | -96.19%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.199580 | 0.006045   | 33.02x  | -96.97%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.156498 | 0.018541   | 8.44x   | -88.15%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.156498 | 0.005631   | 27.79x  | -96.40%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.156498 | 0.006024   | 25.98x  | -96.15%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.216663 | 0.024128   | 8.98x   | -88.86%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.216663 | 0.007980   | 27.15x  | -96.32%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.216663 | 0.007101   | 30.51x  | -96.72%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.163128 | 0.025128   | 6.49x   | -84.60%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.163128 | 0.007040   | 23.17x  | -95.68%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.163128 | 0.006346   | 25.71x  | -96.11%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.172916 | 0.021964   | 7.87x   | -87.30%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.172916 | 0.007302   | 23.68x  | -95.78%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.172916 | 0.006584   | 26.26x  | -96.19%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.168258 | 0.023939   | 7.03x   | -85.77%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.168258 | 0.008356   | 20.14x  | -95.03%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.168258 | 0.006922   | 24.31x  | -95.89%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.168320 | 0.054318   | 3.10x   | -67.73%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.168320 | 0.008509   | 19.78x  | -94.94%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.168320 | 0.006717   | 25.06x  | -96.01%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.174713 | 0.029098   | 6.00x   | -83.35%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.174713 | 0.008163   | 21.40x  | -95.33%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.174713 | 0.007014   | 24.91x  | -95.99%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.185483 | 0.022687   | 8.18x   | -87.77%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.185483 | 0.007636   | 24.29x  | -95.88%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.185483 | 0.006938   | 26.73x  | -96.26%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.248140 | 0.063949   | 3.88x   | -74.23%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.248140 | 0.008675   | 28.60x  | -96.50%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.248140 | 0.007423   | 33.43x  | -97.01%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.234989 | 0.021560   | 10.90x  | -90.82%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.234989 | 0.007438   | 31.59x  | -96.83%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.234989 | 0.006870   | 34.21x  | -97.08%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.177086 | 0.027616   | 6.41x   | -84.41%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.177086 | 0.008914   | 19.87x  | -94.97%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.177086 | 0.006883   | 25.73x  | -96.11%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.171567 | 0.035773   | 4.80x   | -79.15%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.171567 | 0.007722   | 22.22x  | -95.50%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.171567 | 0.006416   | 26.74x  | -96.26%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.175278 | 0.036172   | 4.85x   | -79.36%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.175278 | 0.008036   | 21.81x  | -95.42%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.175278 | 0.006792   | 25.81x  | -96.12%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.174657 | 0.023182   | 7.53x   | -86.73%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.174657 | 0.007717   | 22.63x  | -95.58%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.174657 | 0.007145   | 24.45x  | -95.91%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.230536 | 0.064004   | 3.60x   | -72.24%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.230536 | 0.008875   | 25.98x  | -96.15%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.230536 | 0.008758   | 26.32x  | -96.20%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.128936 | 0.019863   | 6.49x   | -84.60%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.128936 | 0.009664   | 13.34x  | -92.50%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.128936 | 0.007392   | 17.44x  | -94.27%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.187884 | 0.026300   | 7.14x   | -86.00%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.187884 | 0.011380   | 16.51x  | -93.94%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.187884 | 0.008629   | 21.77x  | -95.41%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.131959 | 0.029605   | 4.46x   | -77.57%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.131959 | 0.011569   | 11.41x  | -91.23%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.131959 | 0.008801   | 14.99x  | -93.33%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.144300 | 0.026623   | 5.42x   | -81.55%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.144300 | 0.012315   | 11.72x  | -91.47%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.144300 | 0.008553   | 16.87x  | -94.07%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.143432 | 0.027534   | 5.21x   | -80.80%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.143432 | 0.011552   | 12.42x  | -91.95%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.143432 | 0.008578   | 16.72x  | -94.02%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.139979 | 0.046761   | 2.99x   | -66.59%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.139979 | 0.012408   | 11.28x  | -91.14%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.139979 | 0.009364   | 14.95x  | -93.31%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.134421 | 0.029590   | 4.54x   | -77.99%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.134421 | 0.011642   | 11.55x  | -91.34%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.134421 | 0.008373   | 16.05x  | -93.77%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.135924 | 0.022722   | 5.98x   | -83.28%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.135924 | 0.011600   | 11.72x  | -91.47%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.135924 | 0.008420   | 16.14x  | -93.81%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.203270 | 0.063650   | 3.19x   | -68.69%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.203270 | 0.011487   | 17.70x  | -94.35%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.203270 | 0.008886   | 22.87x  | -95.63%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.194340 | 0.024057   | 8.08x   | -87.62%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.194340 | 0.011434   | 17.00x  | -94.12%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.194340 | 0.008224   | 23.63x  | -95.77%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.135082 | 0.030586   | 4.42x   | -77.36%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.135082 | 0.013870   | 9.74x   | -89.73%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.135082 | 0.008438   | 16.01x  | -93.75%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.138601 | 0.035677   | 3.88x   | -74.26%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.138601 | 0.011339   | 12.22x  | -91.82%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.138601 | 0.008259   | 16.78x  | -94.04%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.149335 | 0.036293   | 4.11x   | -75.70%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.149335 | 0.011308   | 13.21x  | -92.43%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.149335 | 0.008376   | 17.83x  | -94.39%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.145038 | 0.025005   | 5.80x   | -82.76%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.145038 | 0.011594   | 12.51x  | -92.01%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.145038 | 0.008270   | 17.54x  | -94.30%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.190438 | 0.058461   | 3.26x   | -69.30%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.190438 | 0.011094   | 17.17x  | -94.17%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.190438 | 0.008348   | 22.81x  | -95.62%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.341971 | 0.091623   | 3.73x   | -73.21%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.341971 | 0.077114   | 4.43x   | -77.45%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.341971 | 0.084262   | 4.06x   | -75.36%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.451570 | 0.108002   | 4.18x   | -76.08%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.451570 | 0.080442   | 5.61x   | -82.19%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.451570 | 0.087068   | 5.19x   | -80.72%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.342443 | 0.113048   | 3.03x   | -66.99%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.342443 | 0.081170   | 4.22x   | -76.30%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.342443 | 0.084151   | 4.07x   | -75.43%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.380768 | 0.101633   | 3.75x   | -73.31%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.380768 | 0.079598   | 4.78x   | -79.10%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.380768 | 0.082510   | 4.61x   | -78.33%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.372851 | 0.110684   | 3.37x   | -70.31%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.372851 | 0.080470   | 4.63x   | -78.42%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.372851 | 0.083706   | 4.45x   | -77.55%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.352271 | 0.146845   | 2.40x   | -58.31%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.352271 | 0.077475   | 4.55x   | -78.01%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.352271 | 0.082971   | 4.25x   | -76.45%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.383278 | 0.114954   | 3.33x   | -70.01%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.383278 | 0.080345   | 4.77x   | -79.04%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.383278 | 0.087532   | 4.38x   | -77.16%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.348260 | 0.103479   | 3.37x   | -70.29%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.348260 | 0.080964   | 4.30x   | -76.75%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.348260 | 0.084299   | 4.13x   | -75.79%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.495963 | 0.171679   | 2.89x   | -65.38%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.495963 | 0.079834   | 6.21x   | -83.90%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.495963 | 0.083565   | 5.94x   | -83.15%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.438117 | 0.101907   | 4.30x   | -76.74%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.438117 | 0.079542   | 5.51x   | -81.84%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.438117 | 0.082823   | 5.29x   | -81.10%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.355767 | 0.104563   | 3.40x   | -70.61%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.355767 | 0.078975   | 4.50x   | -77.80%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.355767 | 0.082572   | 4.31x   | -76.79%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.348974 | 0.123238   | 2.83x   | -64.69%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.348974 | 0.081322   | 4.29x   | -76.70%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.348974 | 0.082876   | 4.21x   | -76.25%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.348562 | 0.121947   | 2.86x   | -65.01%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.348562 | 0.079222   | 4.40x   | -77.27%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.348562 | 0.082775   | 4.21x   | -76.25%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.398419 | 0.116821   | 3.41x   | -70.68%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.398419 | 0.079243   | 5.03x   | -80.11%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.398419 | 0.084910   | 4.69x   | -78.69%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.476358 | 0.167278   | 2.85x   | -64.88%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.476358 | 0.079944   | 5.96x   | -83.22%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.476358 | 0.085349   | 5.58x   | -82.08%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.260854 | 0.077819   | 3.35x   | -70.17%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.260854 | 0.010820   | 24.11x  | -95.85%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.260854 | 0.008604   | 30.32x  | -96.70%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.348808 | 0.109657   | 3.18x   | -68.56%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.348808 | 0.013068   | 26.69x  | -96.25%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.348808 | 0.010893   | 32.02x  | -96.88%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.242472 | 0.101997   | 2.38x   | -57.93%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.242472 | 0.013506   | 17.95x  | -94.43%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.242472 | 0.010511   | 23.07x  | -95.67%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.264979 | 0.091356   | 2.90x   | -65.52%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.264979 | 0.012714   | 20.84x  | -95.20%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.264979 | 0.010660   | 24.86x  | -95.98%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.270131 | 0.100410   | 2.69x   | -62.83%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.270131 | 0.014364   | 18.81x  | -94.68%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.270131 | 0.010976   | 24.61x  | -95.94%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.249399 | 0.117063   | 2.13x   | -53.06%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.249399 | 0.015337   | 16.26x  | -93.85%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.249399 | 0.011004   | 22.66x  | -95.59%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.246624 | 0.104347   | 2.36x   | -57.69%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.246624 | 0.012362   | 19.95x  | -94.99%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.246624 | 0.010615   | 23.23x  | -95.70%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.250056 | 0.094705   | 2.64x   | -62.13%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.250056 | 0.012665   | 19.74x  | -94.94%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.250056 | 0.010776   | 23.20x  | -95.69%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.399240 | 0.156120   | 2.56x   | -60.90%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.399240 | 0.014442   | 27.64x  | -96.38%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.399240 | 0.010865   | 36.74x  | -97.28%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.359968 | 0.098618   | 3.65x   | -72.60%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.359968 | 0.013510   | 26.64x  | -96.25%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.359968 | 0.010809   | 33.30x  | -97.00%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.257017 | 0.096537   | 2.66x   | -62.44%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.257017 | 0.015578   | 16.50x  | -93.94%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.257017 | 0.011360   | 22.62x  | -95.58%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.256619 | 0.110363   | 2.33x   | -56.99%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.256619 | 0.012929   | 19.85x  | -94.96%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.256619 | 0.010725   | 23.93x  | -95.82%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.285129 | 0.112934   | 2.52x   | -60.39%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.285129 | 0.012775   | 22.32x  | -95.52%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.285129 | 0.010687   | 26.68x  | -96.25%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.268114 | 0.096096   | 2.79x   | -64.16%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.268114 | 0.013490   | 19.87x  | -94.97%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.268114 | 0.010716   | 25.02x  | -96.00%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.387731 | 0.146019   | 2.66x   | -62.34%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.387731 | 0.013779   | 28.14x  | -96.45%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.387731 | 0.010784   | 35.95x  | -97.22%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.322413 | 0.028901   | 11.16x  | -91.04%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.322413 | 0.011091   | 29.07x  | -96.56%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.322413 | 0.012458   | 25.88x  | -96.14%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.416765 | 0.041036   | 10.16x  | -90.15%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.416765 | 0.013975   | 29.82x  | -96.65%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.416765 | 0.012636   | 32.98x  | -96.97%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.296726 | 0.042628   | 6.96x   | -85.63%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.296726 | 0.013254   | 22.39x  | -95.53%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.296726 | 0.011896   | 24.94x  | -95.99%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.317445 | 0.035018   | 9.07x   | -88.97%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.317445 | 0.013318   | 23.84x  | -95.80%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.317445 | 0.012639   | 25.12x  | -96.02%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.321386 | 0.038498   | 8.35x   | -88.02%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.321386 | 0.014767   | 21.76x  | -95.41%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.321386 | 0.012533   | 25.64x  | -96.10%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.310771 | 0.091714   | 3.39x   | -70.49%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.310771 | 0.017525   | 17.73x  | -94.36%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.310771 | 0.012030   | 25.83x  | -96.13%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.294986 | 0.041847   | 7.05x   | -85.81%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.294986 | 0.013022   | 22.65x  | -95.59%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.294986 | 0.012094   | 24.39x  | -95.90%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.306268 | 0.034944   | 8.76x   | -88.59%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.306268 | 0.013095   | 23.39x  | -95.72%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.306268 | 0.012164   | 25.18x  | -96.03%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.456430 | 0.110292   | 4.14x   | -75.84%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.456430 | 0.014684   | 31.08x  | -96.78%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.456430 | 0.013273   | 34.39x  | -97.09%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.412858 | 0.034134   | 12.10x  | -91.73%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.412858 | 0.013292   | 31.06x  | -96.78%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.412858 | 0.012695   | 32.52x  | -96.93%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.311327 | 0.042569   | 7.31x   | -86.33%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.311327 | 0.015322   | 20.32x  | -95.08%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.311327 | 0.012355   | 25.20x  | -96.03%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.316230 | 0.059625   | 5.30x   | -81.14%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.316230 | 0.014245   | 22.20x  | -95.50%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.316230 | 0.012890   | 24.53x  | -95.92%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.326514 | 0.060782   | 5.37x   | -81.38%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.326514 | 0.014607   | 22.35x  | -95.53%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.326514 | 0.016116   | 20.26x  | -95.06%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.322661 | 0.039599   | 8.15x   | -87.73%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.322661 | 0.014385   | 22.43x  | -95.54%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.322661 | 0.013641   | 23.65x  | -95.77%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.410292 | 0.102324   | 4.01x   | -75.06%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.410292 | 0.014201   | 28.89x  | -96.54%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.410292 | 0.012701   | 32.30x  | -96.90%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.244572 | 0.042043   | 5.82x   | -82.81%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.244572 | 0.024084   | 10.15x  | -90.15%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.244572 | 0.020575   | 11.89x  | -91.59%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.337002 | 0.054688   | 6.16x   | -83.77%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.337002 | 0.027512   | 12.25x  | -91.84%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.337002 | 0.021984   | 15.33x  | -93.48%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.245572 | 0.058189   | 4.22x   | -76.30%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.245572 | 0.027221   | 9.02x   | -88.92%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.245572 | 0.021888   | 11.22x  | -91.09%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.259545 | 0.053771   | 4.83x   | -79.28%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.259545 | 0.027577   | 9.41x   | -89.37%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.259545 | 0.022758   | 11.40x  | -91.23%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.259269 | 0.054292   | 4.78x   | -79.06%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.259269 | 0.029155   | 8.89x   | -88.75%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.259269 | 0.022715   | 11.41x  | -91.24%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.253338 | 0.088758   | 2.85x   | -64.96%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.253338 | 0.027967   | 9.06x   | -88.96%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.253338 | 0.029867   | 8.48x   | -88.21%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.238275 | 0.057386   | 4.15x   | -75.92%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.238275 | 0.027040   | 8.81x   | -88.65%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.238275 | 0.021713   | 10.97x  | -90.89%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.256700 | 0.049832   | 5.15x   | -80.59%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.256700 | 0.026816   | 9.57x   | -89.55%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.256700 | 0.021317   | 12.04x  | -91.70%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.401411 | 0.115747   | 3.47x   | -71.17%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.401411 | 0.027326   | 14.69x  | -93.19%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.401411 | 0.022837   | 17.58x  | -94.31%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.346604 | 0.049745   | 6.97x   | -85.65%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.346604 | 0.029156   | 11.89x  | -91.59%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.346604 | 0.023389   | 14.82x  | -93.25%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.260844 | 0.062714   | 4.16x   | -75.96%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.260844 | 0.027657   | 9.43x   | -89.40%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.260844 | 0.022257   | 11.72x  | -91.47%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.267191 | 0.072436   | 3.69x   | -72.89%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.267191 | 0.027460   | 9.73x   | -89.72%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.267191 | 0.021732   | 12.29x  | -91.87%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.257246 | 0.070242   | 3.66x   | -72.69%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.257246 | 0.026084   | 9.86x   | -89.86%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.257246 | 0.025479   | 10.10x  | -90.10%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.274156 | 0.052221   | 5.25x   | -80.95%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.274156 | 0.027038   | 10.14x  | -90.14%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.274156 | 0.022036   | 12.44x  | -91.96%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.366282 | 0.111022   | 3.30x   | -69.69%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.366282 | 0.025640   | 14.29x  | -93.00%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.366282 | 0.021772   | 16.82x  | -94.06%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.758857 | 0.237848   | 3.19x   | -68.66%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.758857 | 0.175648   | 4.32x   | -76.85%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.758857 | 0.181017   | 4.19x   | -76.15%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 1.068068 | 0.259038   | 4.12x   | -75.75%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 1.068068 | 0.188211   | 5.67x   | -82.38%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 1.068068 | 0.196991   | 5.42x   | -81.56%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.773249 | 0.253351   | 3.05x   | -67.24%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.773249 | 0.187392   | 4.13x   | -75.77%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.773249 | 0.193033   | 4.01x   | -75.04%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.782257 | 0.228597   | 3.42x   | -70.78%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.782257 | 0.183136   | 4.27x   | -76.59%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.782257 | 0.189210   | 4.13x   | -75.81%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.799069 | 0.244194   | 3.27x   | -69.44%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.799069 | 0.180387   | 4.43x   | -77.43%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.799069 | 0.191573   | 4.17x   | -76.03%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.758499 | 0.335927   | 2.26x   | -55.71%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.758499 | 0.172329   | 4.40x   | -77.28%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.758499 | 0.183897   | 4.12x   | -75.76%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.745522 | 0.256297   | 2.91x   | -65.62%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.745522 | 0.180336   | 4.13x   | -75.81%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.745522 | 0.191885   | 3.89x   | -74.26%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.764632 | 0.230765   | 3.31x   | -69.82%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.764632 | 0.180591   | 4.23x   | -76.38%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.764632 | 0.187830   | 4.07x   | -75.44%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.087909 | 0.398472   | 2.73x   | -63.37%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.087909 | 0.181305   | 6.00x   | -83.33%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.087909 | 0.191961   | 5.67x   | -82.36%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.996030 | 0.226560   | 4.40x   | -77.25%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.996030 | 0.176934   | 5.63x   | -82.24%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.996030 | 0.183596   | 5.43x   | -81.57%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.742985 | 0.229076   | 3.24x   | -69.17%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.742985 | 0.174707   | 4.25x   | -76.49%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.742985 | 0.185741   | 4.00x   | -75.00%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.774643 | 0.279962   | 2.77x   | -63.86%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.774643 | 0.177701   | 4.36x   | -77.06%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.774643 | 0.189127   | 4.10x   | -75.59%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.800348 | 0.288494   | 2.77x   | -63.95%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.800348 | 0.179293   | 4.46x   | -77.60%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.800348 | 0.186991   | 4.28x   | -76.64%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.803819 | 0.240743   | 3.34x   | -70.05%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.803819 | 0.179182   | 4.49x   | -77.71%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.803819 | 0.189326   | 4.25x   | -76.45%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 0.995222 | 0.386792   | 2.57x   | -61.14%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 0.995222 | 0.184667   | 5.39x   | -81.44%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 0.995222 | 0.191872   | 5.19x   | -80.72%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.576989 | 0.186339   | 3.10x   | -67.71%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.576989 | 0.024825   | 23.24x  | -95.70%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.576989 | 0.019971   | 28.89x  | -96.54%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.843521 | 0.234560   | 3.60x   | -72.19%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.843521 | 0.030753   | 27.43x  | -96.35%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.843521 | 0.024902   | 33.87x  | -97.05%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.562358 | 0.233586   | 2.41x   | -58.46%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.562358 | 0.028906   | 19.45x  | -94.86%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.562358 | 0.024356   | 23.09x  | -95.67%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.621960 | 0.216818   | 2.87x   | -65.14%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.621960 | 0.028606   | 21.74x  | -95.40%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.621960 | 0.024277   | 25.62x  | -96.10%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.620259 | 0.237047   | 2.62x   | -61.78%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.620259 | 0.033016   | 18.79x  | -94.68%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.620259 | 0.026288   | 23.59x  | -95.76%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.601906 | 0.289391   | 2.08x   | -51.92%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.601906 | 0.035979   | 16.73x  | -94.02%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.601906 | 0.025763   | 23.36x  | -95.72%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.580079 | 0.229167   | 2.53x   | -60.49%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.580079 | 0.028451   | 20.39x  | -95.10%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.580079 | 0.024559   | 23.62x  | -95.77%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.619544 | 0.235207   | 2.63x   | -62.04%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.619544 | 0.028003   | 22.12x  | -95.48%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.619544 | 0.023855   | 25.97x  | -96.15%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.965797 | 0.353766   | 2.73x   | -63.37%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.965797 | 0.032586   | 29.64x  | -96.63%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.965797 | 0.024666   | 39.16x  | -97.45%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.797073 | 0.217655   | 3.66x   | -72.69%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.797073 | 0.029130   | 27.36x  | -96.35%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.797073 | 0.024977   | 31.91x  | -96.87%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.577202 | 0.219240   | 2.63x   | -62.02%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.577202 | 0.035106   | 16.44x  | -93.92%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.577202 | 0.025646   | 22.51x  | -95.56%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.619464 | 0.260681   | 2.38x   | -57.92%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.619464 | 0.030079   | 20.59x  | -95.14%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.619464 | 0.024445   | 25.34x  | -96.05%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.603395 | 0.256917   | 2.35x   | -57.42%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.603395 | 0.028419   | 21.23x  | -95.29%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.603395 | 0.023753   | 25.40x  | -96.06%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.590331 | 0.222489   | 2.65x   | -62.31%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.590331 | 0.030120   | 19.60x  | -94.90%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.590331 | 0.023925   | 24.67x  | -95.95%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.825301 | 0.331217   | 2.49x   | -59.87%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.825301 | 0.031002   | 26.62x  | -96.24%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.825301 | 0.024459   | 33.74x  | -97.04%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.671677 | 0.073933   | 9.08x   | -88.99%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.671677 | 0.040142   | 16.73x  | -94.02%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.671677 | 0.039788   | 16.88x  | -94.08%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.924338 | 0.111117   | 8.32x   | -87.98%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.924338 | 0.041867   | 22.08x  | -95.47%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.924338 | 0.037975   | 24.34x  | -95.89%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.638254 | 0.103272   | 6.18x   | -83.82%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.638254 | 0.039505   | 16.16x  | -93.81%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.638254 | 0.037081   | 17.21x  | -94.19%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.686070 | 0.090105   | 7.61x   | -86.87%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.686070 | 0.039972   | 17.16x  | -94.17%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.686070 | 0.037604   | 18.24x  | -94.52%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.691782 | 0.098022   | 7.06x   | -85.83%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.691782 | 0.043435   | 15.93x  | -93.72%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.691782 | 0.039847   | 17.36x  | -94.24%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.671413 | 0.215813   | 3.11x   | -67.86%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.671413 | 0.043659   | 15.38x  | -93.50%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.671413 | 0.036762   | 18.26x  | -94.52%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.635880 | 0.105414   | 6.03x   | -83.42%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.635880 | 0.038490   | 16.52x  | -93.95%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.635880 | 0.037613   | 16.91x  | -94.08%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.658060 | 0.086357   | 7.62x   | -86.88%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.658060 | 0.040688   | 16.17x  | -93.82%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.658060 | 0.036545   | 18.01x  | -94.45%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.965677 | 0.255766   | 3.78x   | -73.51%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.965677 | 0.044124   | 21.89x  | -95.43%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.965677 | 0.039346   | 24.54x  | -95.93%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.890481 | 0.084673   | 10.52x  | -90.49%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.890481 | 0.038899   | 22.89x  | -95.63%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.890481 | 0.035235   | 25.27x  | -96.04%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.648623 | 0.103807   | 6.25x   | -84.00%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.648623 | 0.044193   | 14.68x  | -93.19%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.648623 | 0.037272   | 17.40x  | -94.25%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.670494 | 0.146040   | 4.59x   | -78.22%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.670494 | 0.041393   | 16.20x  | -93.83%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.670494 | 0.036636   | 18.30x  | -94.54%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.674796 | 0.144700   | 4.66x   | -78.56%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.674796 | 0.041871   | 16.12x  | -93.79%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.674796 | 0.036628   | 18.42x  | -94.57%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.679810 | 0.093167   | 7.30x   | -86.30%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.679810 | 0.041573   | 16.35x  | -93.88%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.679810 | 0.043861   | 15.50x  | -93.55%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.893251 | 0.239000   | 3.74x   | -73.24%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.893251 | 0.042074   | 21.23x  | -95.29%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.893251 | 0.036799   | 24.27x  | -95.88%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.498253 | 0.090791   | 5.49x   | -81.78%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.498253 | 0.050818   | 9.80x   | -89.80%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.498253 | 0.041482   | 12.01x  | -91.67%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.738970 | 0.116353   | 6.35x   | -84.25%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.738970 | 0.057497   | 12.85x  | -92.22%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.738970 | 0.045854   | 16.12x  | -93.79%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.508075 | 0.129103   | 3.94x   | -74.59%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.508075 | 0.058054   | 8.75x   | -88.57%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.508075 | 0.046607   | 10.90x  | -90.83%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.534716 | 0.113144   | 4.73x   | -78.84%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.534716 | 0.056781   | 9.42x   | -89.38%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.534716 | 0.045018   | 11.88x  | -91.58%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.529891 | 0.116982   | 4.53x   | -77.92%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.529891 | 0.058600   | 9.04x   | -88.94%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.529891 | 0.049553   | 10.69x  | -90.65%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.514071 | 0.192735   | 2.67x   | -62.51%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.514071 | 0.060226   | 8.54x   | -88.28%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.514071 | 0.045463   | 11.31x  | -91.16%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.501062 | 0.127961   | 3.92x   | -74.46%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.501062 | 0.060713   | 8.25x   | -87.88%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.501062 | 0.048218   | 10.39x  | -90.38%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.518585 | 0.103028   | 5.03x   | -80.13%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.518585 | 0.058248   | 8.90x   | -88.77%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.518585 | 0.046158   | 11.23x  | -91.10%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.829958 | 0.261078   | 3.18x   | -68.54%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.829958 | 0.057692   | 14.39x  | -93.05%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.829958 | 0.046819   | 17.73x  | -94.36%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.730418 | 0.107656   | 6.78x   | -85.26%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.730418 | 0.057789   | 12.64x  | -92.09%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.730418 | 0.044902   | 16.27x  | -93.85%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.507520 | 0.138090   | 3.68x   | -72.79%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.507520 | 0.060638   | 8.37x   | -88.05%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.507520 | 0.047095   | 10.78x  | -90.72%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.546334 | 0.157400   | 3.47x   | -71.19%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.546334 | 0.058226   | 9.38x   | -89.34%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.546334 | 0.046188   | 11.83x  | -91.55%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.523254 | 0.154674   | 3.38x   | -70.44%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.523254 | 0.057329   | 9.13x   | -89.04%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.523254 | 0.045763   | 11.43x  | -91.25%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.558933 | 0.113005   | 4.95x   | -79.78%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.558933 | 0.059204   | 9.44x   | -89.41%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.558933 | 0.047032   | 11.88x  | -91.59%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.742098 | 0.244027   | 3.04x   | -67.12%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.742098 | 0.056798   | 13.07x  | -92.35%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.742098 | 0.047966   | 15.47x  | -93.54%        |
</details>
<!-- PERF_RESULTS_END -->
