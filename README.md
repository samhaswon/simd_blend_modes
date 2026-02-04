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
| normal        | scalar | 0.200207 | 0.042092   | 4.76x   | -78.98%        |
| normal        | sse42  | 0.200207 | 0.014064   | 14.24x  | -92.98%        |
| normal        | avx2   | 0.200207 | 0.013356   | 14.99x  | -93.33%        |
| soft_light    | scalar | 0.273639 | 0.049037   | 5.58x   | -82.08%        |
| soft_light    | sse42  | 0.273639 | 0.016607   | 16.48x  | -93.93%        |
| soft_light    | avx2   | 0.273639 | 0.015255   | 17.94x  | -94.43%        |
| lighten_only  | scalar | 0.200017 | 0.053192   | 3.76x   | -73.41%        |
| lighten_only  | sse42  | 0.200017 | 0.015282   | 13.09x  | -92.36%        |
| lighten_only  | avx2   | 0.200017 | 0.014691   | 13.62x  | -92.66%        |
| screen        | scalar | 0.213643 | 0.046891   | 4.56x   | -78.05%        |
| screen        | sse42  | 0.213643 | 0.015909   | 13.43x  | -92.55%        |
| screen        | avx2   | 0.213643 | 0.015008   | 14.23x  | -92.97%        |
| dodge         | scalar | 0.214396 | 0.050080   | 4.28x   | -76.64%        |
| dodge         | sse42  | 0.214396 | 0.017373   | 12.34x  | -91.90%        |
| dodge         | avx2   | 0.214396 | 0.015568   | 13.77x  | -92.74%        |
| addition      | scalar | 0.206058 | 0.075378   | 2.73x   | -63.42%        |
| addition      | sse42  | 0.206058 | 0.016390   | 12.57x  | -92.05%        |
| addition      | avx2   | 0.206058 | 0.015153   | 13.60x  | -92.65%        |
| darken_only   | scalar | 0.201230 | 0.053439   | 3.77x   | -73.44%        |
| darken_only   | sse42  | 0.201230 | 0.015304   | 13.15x  | -92.39%        |
| darken_only   | avx2   | 0.201230 | 0.014711   | 13.68x  | -92.69%        |
| multiply      | scalar | 0.206920 | 0.046745   | 4.43x   | -77.41%        |
| multiply      | sse42  | 0.206920 | 0.015593   | 13.27x  | -92.46%        |
| multiply      | avx2   | 0.206920 | 0.014986   | 13.81x  | -92.76%        |
| hard_light    | scalar | 0.306952 | 0.093280   | 3.29x   | -69.61%        |
| hard_light    | sse42  | 0.306952 | 0.017610   | 17.43x  | -94.26%        |
| hard_light    | avx2   | 0.306952 | 0.015595   | 19.68x  | -94.92%        |
| difference    | scalar | 0.277872 | 0.046439   | 5.98x   | -83.29%        |
| difference    | sse42  | 0.277872 | 0.015597   | 17.82x  | -94.39%        |
| difference    | avx2   | 0.277872 | 0.014993   | 18.53x  | -94.60%        |
| subtract      | scalar | 0.207039 | 0.049087   | 4.22x   | -76.29%        |
| subtract      | sse42  | 0.207039 | 0.017608   | 11.76x  | -91.50%        |
| subtract      | avx2   | 0.207039 | 0.015622   | 13.25x  | -92.45%        |
| grain_extract | scalar | 0.213208 | 0.062544   | 3.41x   | -70.67%        |
| grain_extract | sse42  | 0.213208 | 0.016450   | 12.96x  | -92.28%        |
| grain_extract | avx2   | 0.213208 | 0.015483   | 13.77x  | -92.74%        |
| grain_merge   | scalar | 0.215160 | 0.062312   | 3.45x   | -71.04%        |
| grain_merge   | sse42  | 0.215160 | 0.016667   | 12.91x  | -92.25%        |
| grain_merge   | avx2   | 0.215160 | 0.015334   | 14.03x  | -92.87%        |
| divide        | scalar | 0.217853 | 0.049344   | 4.42x   | -77.35%        |
| divide        | sse42  | 0.217853 | 0.017288   | 12.60x  | -92.06%        |
| divide        | avx2   | 0.217853 | 0.015734   | 13.85x  | -92.78%        |
| overlay       | scalar | 0.286022 | 0.089595   | 3.19x   | -68.68%        |
| overlay       | sse42  | 0.286022 | 0.016833   | 16.99x  | -94.11%        |
| overlay       | avx2   | 0.286022 | 0.015514   | 18.44x  | -94.58%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.006570 | 0.001631   | 4.03x   | -75.18%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.006570 | 0.000697   | 9.43x   | -89.40%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.006570 | 0.000705   | 9.32x   | -89.27%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.008765 | 0.001803   | 4.86x   | -79.43%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.008765 | 0.000872   | 10.05x  | -90.05%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.008765 | 0.000812   | 10.80x  | -90.74%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.007296 | 0.001986   | 3.67x   | -72.78%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.007296 | 0.000788   | 9.26x   | -89.21%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.007296 | 0.000796   | 9.16x   | -89.08%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.007447 | 0.001811   | 4.11x   | -75.68%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.007447 | 0.000915   | 8.14x   | -87.72%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.007447 | 0.000866   | 8.60x   | -88.37%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.007917 | 0.001893   | 4.18x   | -76.09%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.007917 | 0.000970   | 8.16x   | -87.75%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.007917 | 0.000841   | 9.41x   | -89.37%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.008313 | 0.002586   | 3.21x   | -68.89%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.008313 | 0.000899   | 9.25x   | -89.19%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.008313 | 0.000876   | 9.49x   | -89.46%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.007417 | 0.001920   | 3.86x   | -74.11%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.007417 | 0.000789   | 9.40x   | -89.36%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.007417 | 0.000766   | 9.69x   | -89.68%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.007188 | 0.001866   | 3.85x   | -74.04%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.007188 | 0.000849   | 8.46x   | -88.18%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.007188 | 0.000789   | 9.11x   | -89.02%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.009434 | 0.003074   | 3.07x   | -67.42%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.009434 | 0.000899   | 10.49x  | -90.47%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.009434 | 0.000824   | 11.45x  | -91.27%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.009260 | 0.001739   | 5.32x   | -81.22%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.009260 | 0.000861   | 10.76x  | -90.71%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.009260 | 0.000777   | 11.92x  | -91.61%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.007825 | 0.001746   | 4.48x   | -77.68%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.007825 | 0.000892   | 8.77x   | -88.60%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.007825 | 0.000898   | 8.71x   | -88.52%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.007762 | 0.002182   | 3.56x   | -71.88%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.007762 | 0.000882   | 8.80x   | -88.63%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.007762 | 0.000935   | 8.30x   | -87.96%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.007734 | 0.002166   | 3.57x   | -71.99%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.007734 | 0.000933   | 8.29x   | -87.93%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.007734 | 0.000814   | 9.50x   | -89.47%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.007703 | 0.001904   | 4.05x   | -75.28%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.007703 | 0.000885   | 8.70x   | -88.51%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.007703 | 0.000928   | 8.30x   | -87.95%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.009154 | 0.002870   | 3.19x   | -68.65%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.009154 | 0.000877   | 10.43x  | -90.42%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.009154 | 0.000812   | 11.27x  | -91.12%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003566 | 0.001339   | 2.66x   | -62.45%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003566 | 0.000185   | 19.25x  | -94.81%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003566 | 0.000165   | 21.60x  | -95.37%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.007084 | 0.001636   | 4.33x   | -76.90%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.007084 | 0.000275   | 25.79x  | -96.12%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.007084 | 0.000199   | 35.60x  | -97.19%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.005867 | 0.001818   | 3.23x   | -69.01%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.005867 | 0.000228   | 25.72x  | -96.11%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.005867 | 0.000193   | 30.38x  | -96.71%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.005735 | 0.001586   | 3.62x   | -72.34%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.005735 | 0.000220   | 26.10x  | -96.17%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.005735 | 0.000197   | 29.14x  | -96.57%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.006204 | 0.001938   | 3.20x   | -68.77%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.006204 | 0.000237   | 26.14x  | -96.17%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.006204 | 0.000196   | 31.69x  | -96.84%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.006020 | 0.002056   | 2.93x   | -65.84%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.006020 | 0.000263   | 22.93x  | -95.64%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.006020 | 0.000199   | 30.25x  | -96.69%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.005789 | 0.001835   | 3.15x   | -68.29%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.005789 | 0.000202   | 28.67x  | -96.51%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.005789 | 0.000186   | 31.16x  | -96.79%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.006067 | 0.001615   | 3.76x   | -73.38%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.006067 | 0.000208   | 29.19x  | -96.57%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.006067 | 0.000195   | 31.13x  | -96.79%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.007812 | 0.002649   | 2.95x   | -66.10%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.007812 | 0.000270   | 28.96x  | -96.55%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.007812 | 0.000204   | 38.34x  | -97.39%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.007755 | 0.001664   | 4.66x   | -78.55%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.007755 | 0.000200   | 38.84x  | -97.43%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.007755 | 0.000193   | 40.26x  | -97.52%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.006310 | 0.001643   | 3.84x   | -73.96%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.006310 | 0.000270   | 23.38x  | -95.72%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.006310 | 0.000209   | 30.13x  | -96.68%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.005918 | 0.001944   | 3.04x   | -67.16%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.005918 | 0.000223   | 26.56x  | -96.24%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.005918 | 0.000203   | 29.18x  | -96.57%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.006059 | 0.001914   | 3.17x   | -68.41%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.006059 | 0.000228   | 26.55x  | -96.23%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.006059 | 0.000199   | 30.42x  | -96.71%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.006541 | 0.001615   | 4.05x   | -75.31%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.006541 | 0.000221   | 29.61x  | -96.62%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.006541 | 0.000197   | 33.15x  | -96.98%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.007450 | 0.002648   | 2.81x   | -64.46%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.007450 | 0.000229   | 32.46x  | -96.92%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.007450 | 0.000204   | 36.52x  | -97.26%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.005993 | 0.000502   | 11.93x  | -91.62%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.005993 | 0.000263   | 22.78x  | -95.61%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.005993 | 0.000311   | 19.26x  | -94.81%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.009903 | 0.000643   | 15.41x  | -93.51%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.009903 | 0.000232   | 42.76x  | -97.66%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.009903 | 0.000221   | 44.83x  | -97.77%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.007720 | 0.000787   | 9.81x   | -89.80%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.007720 | 0.000214   | 36.00x  | -97.22%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.007720 | 0.000221   | 34.99x  | -97.14%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.007738 | 0.000651   | 11.88x  | -91.58%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.007738 | 0.000349   | 22.20x  | -95.49%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.007738 | 0.000199   | 38.93x  | -97.43%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.008056 | 0.000653   | 12.34x  | -91.90%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.008056 | 0.000409   | 19.72x  | -94.93%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.008056 | 0.000185   | 43.47x  | -97.70%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.007742 | 0.001596   | 4.85x   | -79.39%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.007742 | 0.000240   | 32.31x  | -96.91%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.007742 | 0.000188   | 41.23x  | -97.57%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.007807 | 0.000736   | 10.61x  | -90.58%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.007807 | 0.000217   | 35.92x  | -97.22%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.007807 | 0.000186   | 41.88x  | -97.61%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.007682 | 0.000562   | 13.66x  | -92.68%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.007682 | 0.000209   | 36.78x  | -97.28%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.007682 | 0.000185   | 41.61x  | -97.60%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.009715 | 0.001825   | 5.32x   | -81.21%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.009715 | 0.000244   | 39.80x  | -97.49%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.009715 | 0.000188   | 51.64x  | -98.06%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.009380 | 0.000555   | 16.90x  | -94.08%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.009380 | 0.000224   | 41.79x  | -97.61%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.009380 | 0.000189   | 49.71x  | -97.99%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.007795 | 0.000694   | 11.23x  | -91.09%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.007795 | 0.000283   | 27.56x  | -96.37%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.007795 | 0.000199   | 39.18x  | -97.45%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.007746 | 0.001116   | 6.94x   | -85.59%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.007746 | 0.000254   | 30.53x  | -96.73%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.007746 | 0.000186   | 41.54x  | -97.59%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.008441 | 0.001113   | 7.58x   | -86.81%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.008441 | 0.000306   | 27.60x  | -96.38%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.008441 | 0.000188   | 44.92x  | -97.77%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.008844 | 0.000645   | 13.71x  | -92.71%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.008844 | 0.000244   | 36.23x  | -97.24%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.008844 | 0.000188   | 47.02x  | -97.87%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.009237 | 0.001738   | 5.32x   | -81.19%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.009237 | 0.000310   | 29.77x  | -96.64%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.009237 | 0.000187   | 49.37x  | -97.97%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.004823 | 0.000625   | 7.71x   | -87.04%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.004823 | 0.000214   | 22.51x  | -95.56%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.004823 | 0.000160   | 30.18x  | -96.69%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.007119 | 0.000705   | 10.10x  | -90.10%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.007119 | 0.000190   | 37.42x  | -97.33%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.007119 | 0.000184   | 38.78x  | -97.42%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.005771 | 0.000805   | 7.17x   | -86.06%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.005771 | 0.000175   | 32.93x  | -96.96%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.005771 | 0.000182   | 31.72x  | -96.85%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.005733 | 0.000681   | 8.42x   | -88.13%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.005733 | 0.000178   | 32.25x  | -96.90%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.005733 | 0.000186   | 30.78x  | -96.75%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.006020 | 0.000757   | 7.95x   | -87.42%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.006020 | 0.000226   | 26.60x  | -96.24%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.006020 | 0.000192   | 31.43x  | -96.82%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.005881 | 0.001355   | 4.34x   | -76.96%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.005881 | 0.000201   | 29.29x  | -96.59%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.005881 | 0.000193   | 30.42x  | -96.71%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.005655 | 0.000824   | 6.86x   | -85.43%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.005655 | 0.000176   | 32.12x  | -96.89%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.005655 | 0.000183   | 30.90x  | -96.76%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.005575 | 0.000819   | 6.81x   | -85.31%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.005575 | 0.000170   | 32.71x  | -96.94%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.005575 | 0.000181   | 30.88x  | -96.76%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.007400 | 0.001880   | 3.94x   | -74.59%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.007400 | 0.000238   | 31.06x  | -96.78%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.007400 | 0.000182   | 40.57x  | -97.54%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.007485 | 0.000656   | 11.42x  | -91.24%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.007485 | 0.000195   | 38.33x  | -97.39%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.007485 | 0.000181   | 41.45x  | -97.59%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.005925 | 0.000926   | 6.40x   | -84.37%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.005925 | 0.000200   | 29.63x  | -96.63%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.005925 | 0.000189   | 31.38x  | -96.81%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.005538 | 0.001090   | 5.08x   | -80.33%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.005538 | 0.000178   | 31.17x  | -96.79%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.005538 | 0.000180   | 30.75x  | -96.75%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.005824 | 0.001083   | 5.38x   | -81.40%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.005824 | 0.000183   | 31.90x  | -96.87%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.005824 | 0.000180   | 32.33x  | -96.91%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.006074 | 0.000730   | 8.32x   | -87.98%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.006074 | 0.000192   | 31.56x  | -96.83%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.006074 | 0.000185   | 32.82x  | -96.95%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.007394 | 0.001770   | 4.18x   | -76.06%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.007394 | 0.000196   | 37.73x  | -97.35%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.007394 | 0.000182   | 40.61x  | -97.54%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.035268 | 0.006900   | 5.11x   | -80.44%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.035268 | 0.002808   | 12.56x  | -92.04%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.035268 | 0.002826   | 12.48x  | -91.99%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.045440 | 0.007495   | 6.06x   | -83.51%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.045440 | 0.003532   | 12.86x  | -92.23%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.045440 | 0.003265   | 13.92x  | -92.82%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.038189 | 0.007609   | 5.02x   | -80.08%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.038189 | 0.003372   | 11.32x  | -91.17%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.038189 | 0.003180   | 12.01x  | -91.67%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.039552 | 0.007217   | 5.48x   | -81.75%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.039552 | 0.003358   | 11.78x  | -91.51%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.039552 | 0.003156   | 12.53x  | -92.02%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.039553 | 0.007438   | 5.32x   | -81.20%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.039553 | 0.003581   | 11.04x  | -90.95%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.039553 | 0.003406   | 11.61x  | -91.39%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.038135 | 0.010165   | 3.75x   | -73.35%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.038135 | 0.003256   | 11.71x  | -91.46%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.038135 | 0.003061   | 12.46x  | -91.97%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.038734 | 0.007964   | 4.86x   | -79.44%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.038734 | 0.003168   | 12.23x  | -91.82%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.038734 | 0.003144   | 12.32x  | -91.88%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.039645 | 0.007003   | 5.66x   | -82.34%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.039645 | 0.003406   | 11.64x  | -91.41%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.039645 | 0.003077   | 12.89x  | -92.24%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.047820 | 0.011982   | 3.99x   | -74.94%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.047820 | 0.003629   | 13.18x  | -92.41%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.047820 | 0.003329   | 14.37x  | -93.04%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.046261 | 0.007259   | 6.37x   | -84.31%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.046261 | 0.003316   | 13.95x  | -92.83%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.046261 | 0.003173   | 14.58x  | -93.14%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.039166 | 0.007050   | 5.56x   | -82.00%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.039166 | 0.003556   | 11.01x  | -90.92%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.039166 | 0.003279   | 11.94x  | -91.63%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.040619 | 0.009392   | 4.32x   | -76.88%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.040619 | 0.003491   | 11.64x  | -91.41%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.040619 | 0.003396   | 11.96x  | -91.64%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.039412 | 0.008513   | 4.63x   | -78.40%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.039412 | 0.003503   | 11.25x  | -91.11%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.039412 | 0.003252   | 12.12x  | -91.75%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.039247 | 0.007233   | 5.43x   | -81.57%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.039247 | 0.003752   | 10.46x  | -90.44%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.039247 | 0.003224   | 12.17x  | -91.78%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.046822 | 0.011679   | 4.01x   | -75.06%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.046822 | 0.003494   | 13.40x  | -92.54%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.046822 | 0.003426   | 13.67x  | -92.68%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.026201 | 0.005548   | 4.72x   | -78.83%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.026201 | 0.000726   | 36.08x  | -97.23%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.026201 | 0.000632   | 41.48x  | -97.59%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.036582 | 0.006770   | 5.40x   | -81.49%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.036582 | 0.000959   | 38.13x  | -97.38%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.036582 | 0.000789   | 46.37x  | -97.84%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.029006 | 0.007034   | 4.12x   | -75.75%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.029006 | 0.000772   | 37.56x  | -97.34%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.029006 | 0.000750   | 38.67x  | -97.41%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.030894 | 0.006594   | 4.69x   | -78.66%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.030894 | 0.000856   | 36.09x  | -97.23%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.030894 | 0.000813   | 37.99x  | -97.37%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.030661 | 0.006702   | 4.58x   | -78.14%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.030661 | 0.001066   | 28.76x  | -96.52%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.030661 | 0.000798   | 38.43x  | -97.40%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.029566 | 0.008116   | 3.64x   | -72.55%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.029566 | 0.001024   | 28.88x  | -96.54%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.029566 | 0.000788   | 37.51x  | -97.33%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.030477 | 0.007204   | 4.23x   | -76.36%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.030477 | 0.000778   | 39.18x  | -97.45%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.030477 | 0.000764   | 39.90x  | -97.49%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.032167 | 0.006467   | 4.97x   | -79.89%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.032167 | 0.000807   | 39.88x  | -97.49%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.032167 | 0.000772   | 41.68x  | -97.60%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.041967 | 0.010655   | 3.94x   | -74.61%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.041967 | 0.000995   | 42.19x  | -97.63%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.041967 | 0.000882   | 47.57x  | -97.90%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.038351 | 0.006442   | 5.95x   | -83.20%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.038351 | 0.000823   | 46.59x  | -97.85%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.038351 | 0.000777   | 49.35x  | -97.97%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.029624 | 0.006539   | 4.53x   | -77.93%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.029624 | 0.001076   | 27.53x  | -96.37%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.029624 | 0.000820   | 36.15x  | -97.23%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.030519 | 0.007731   | 3.95x   | -74.67%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.030519 | 0.000979   | 31.17x  | -96.79%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.030519 | 0.000796   | 38.34x  | -97.39%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.030819 | 0.007736   | 3.98x   | -74.90%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.030819 | 0.000868   | 35.52x  | -97.18%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.030819 | 0.000789   | 39.08x  | -97.44%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.031365 | 0.006983   | 4.49x   | -77.73%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.031365 | 0.000893   | 35.14x  | -97.15%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.031365 | 0.000800   | 39.21x  | -97.45%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.037909 | 0.010532   | 3.60x   | -72.22%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.037909 | 0.000927   | 40.91x  | -97.56%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.037909 | 0.000801   | 47.33x  | -97.89%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.029162 | 0.002358   | 12.37x  | -91.91%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.029162 | 0.001076   | 27.10x  | -96.31%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.029162 | 0.000665   | 43.85x  | -97.72%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.041237 | 0.002865   | 14.39x  | -93.05%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.041237 | 0.000946   | 43.59x  | -97.71%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.041237 | 0.000776   | 53.16x  | -98.12%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.033263 | 0.003274   | 10.16x  | -90.16%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.033263 | 0.000918   | 36.23x  | -97.24%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.033263 | 0.000825   | 40.33x  | -97.52%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.037016 | 0.004008   | 9.24x   | -89.17%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.037016 | 0.001242   | 29.80x  | -96.64%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.037016 | 0.001089   | 33.99x  | -97.06%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.040767 | 0.002990   | 13.63x  | -92.67%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.040767 | 0.001200   | 33.98x  | -97.06%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.040767 | 0.000850   | 47.95x  | -97.91%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.033616 | 0.006718   | 5.00x   | -80.01%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.033616 | 0.000970   | 34.66x  | -97.11%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.033616 | 0.000777   | 43.28x  | -97.69%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.034223 | 0.003424   | 9.99x   | -89.99%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.034223 | 0.000905   | 37.83x  | -97.36%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.034223 | 0.000767   | 44.63x  | -97.76%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.034649 | 0.002673   | 12.96x  | -92.28%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.034649 | 0.001016   | 34.10x  | -97.07%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.034649 | 0.000912   | 37.98x  | -97.37%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.043250 | 0.007627   | 5.67x   | -82.37%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.043250 | 0.000999   | 43.31x  | -97.69%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.043250 | 0.000774   | 55.88x  | -98.21%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.041462 | 0.002575   | 16.10x  | -93.79%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.041462 | 0.000902   | 45.99x  | -97.83%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.041462 | 0.000883   | 46.97x  | -97.87%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.034146 | 0.003208   | 10.65x  | -90.61%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.034146 | 0.001230   | 27.76x  | -96.40%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.034146 | 0.000827   | 41.29x  | -97.58%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.034301 | 0.004455   | 7.70x   | -87.01%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.034301 | 0.000909   | 37.75x  | -97.35%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.034301 | 0.000763   | 44.96x  | -97.78%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.034490 | 0.004469   | 7.72x   | -87.04%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.034490 | 0.000910   | 37.90x  | -97.36%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.034490 | 0.000784   | 44.01x  | -97.73%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.034191 | 0.002799   | 12.21x  | -91.81%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.034191 | 0.000954   | 35.84x  | -97.21%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.034191 | 0.000764   | 44.78x  | -97.77%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.041668 | 0.007219   | 5.77x   | -82.67%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.041668 | 0.000950   | 43.87x  | -97.72%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.041668 | 0.000820   | 50.81x  | -98.03%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.021167 | 0.002494   | 8.49x   | -88.22%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.021167 | 0.000832   | 25.45x  | -96.07%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.021167 | 0.000752   | 28.16x  | -96.45%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.032317 | 0.002921   | 11.06x  | -90.96%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.032317 | 0.000931   | 34.70x  | -97.12%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.032317 | 0.000793   | 40.73x  | -97.54%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.025059 | 0.003164   | 7.92x   | -87.38%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.025059 | 0.000749   | 33.45x  | -97.01%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.025059 | 0.000803   | 31.19x  | -96.79%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.025596 | 0.002767   | 9.25x   | -89.19%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.025596 | 0.000784   | 32.64x  | -96.94%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.025596 | 0.000791   | 32.35x  | -96.91%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.025445 | 0.002993   | 8.50x   | -88.24%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.025445 | 0.000930   | 27.35x  | -96.34%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.025445 | 0.000799   | 31.86x  | -96.86%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.024585 | 0.005580   | 4.41x   | -77.30%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.024585 | 0.000837   | 29.39x  | -96.60%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.024585 | 0.001053   | 23.36x  | -95.72%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.024378 | 0.003187   | 7.65x   | -86.93%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.024378 | 0.000778   | 31.34x  | -96.81%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.024378 | 0.000814   | 29.93x  | -96.66%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.025892 | 0.002697   | 9.60x   | -89.58%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.025892 | 0.000784   | 33.02x  | -96.97%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.025892 | 0.000782   | 33.12x  | -96.98%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.034359 | 0.007519   | 4.57x   | -78.12%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.034359 | 0.001006   | 34.17x  | -97.07%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.034359 | 0.000983   | 34.95x  | -97.14%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.033035 | 0.002667   | 12.39x  | -91.93%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.033035 | 0.000746   | 44.27x  | -97.74%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.033035 | 0.000778   | 42.46x  | -97.65%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.024829 | 0.003482   | 7.13x   | -85.98%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.024829 | 0.000817   | 30.40x  | -96.71%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.024829 | 0.000815   | 30.48x  | -96.72%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.025173 | 0.004256   | 5.92x   | -83.09%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.025173 | 0.000738   | 34.11x  | -97.07%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.025173 | 0.000769   | 32.75x  | -96.95%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.025653 | 0.004442   | 5.78x   | -82.69%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.025653 | 0.000793   | 32.33x  | -96.91%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.025653 | 0.000813   | 31.57x  | -96.83%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.026171 | 0.002969   | 8.81x   | -88.65%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.026171 | 0.000840   | 31.15x  | -96.79%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.026171 | 0.000783   | 33.41x  | -97.01%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.032324 | 0.007159   | 4.52x   | -77.85%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.032324 | 0.000803   | 40.25x  | -97.52%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.032324 | 0.000757   | 42.70x  | -97.66%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.101151 | 0.026672   | 3.79x   | -73.63%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.101151 | 0.011935   | 8.48x   | -88.20%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.101151 | 0.011664   | 8.67x   | -88.47%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.136414 | 0.029314   | 4.65x   | -78.51%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.136414 | 0.014134   | 9.65x   | -89.64%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.136414 | 0.013337   | 10.23x  | -90.22%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.110766 | 0.031680   | 3.50x   | -71.40%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.110766 | 0.012872   | 8.61x   | -88.38%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.110766 | 0.012496   | 8.86x   | -88.72%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.116964 | 0.028999   | 4.03x   | -75.21%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.116964 | 0.013280   | 8.81x   | -88.65%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.116964 | 0.013075   | 8.95x   | -88.82%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.112867 | 0.032221   | 3.50x   | -71.45%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.112867 | 0.015209   | 7.42x   | -86.53%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.112867 | 0.014092   | 8.01x   | -87.51%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.114049 | 0.041366   | 2.76x   | -63.73%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.114049 | 0.012987   | 8.78x   | -88.61%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.114049 | 0.012753   | 8.94x   | -88.82%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.108569 | 0.031823   | 3.41x   | -70.69%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.108569 | 0.013030   | 8.33x   | -88.00%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.108569 | 0.012287   | 8.84x   | -88.68%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.112598 | 0.028183   | 4.00x   | -74.97%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.112598 | 0.012986   | 8.67x   | -88.47%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.112598 | 0.013012   | 8.65x   | -88.44%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.161288 | 0.047569   | 3.39x   | -70.51%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.161288 | 0.014452   | 11.16x  | -91.04%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.161288 | 0.013853   | 11.64x  | -91.41%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.138282 | 0.029238   | 4.73x   | -78.86%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.138282 | 0.013012   | 10.63x  | -90.59%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.138282 | 0.012992   | 10.64x  | -90.61%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.108601 | 0.027283   | 3.98x   | -74.88%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.108601 | 0.013954   | 7.78x   | -87.15%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.108601 | 0.012937   | 8.39x   | -88.09%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.108122 | 0.034135   | 3.17x   | -68.43%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.108122 | 0.013886   | 7.79x   | -87.16%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.108122 | 0.012986   | 8.33x   | -87.99%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.108823 | 0.034198   | 3.18x   | -68.57%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.108823 | 0.013857   | 7.85x   | -87.27%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.108823 | 0.012865   | 8.46x   | -88.18%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.110052 | 0.029278   | 3.76x   | -73.40%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.110052 | 0.014258   | 7.72x   | -87.04%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.110052 | 0.013145   | 8.37x   | -88.06%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.135828 | 0.046617   | 2.91x   | -65.68%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.135828 | 0.014165   | 9.59x   | -89.57%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.135828 | 0.013479   | 10.08x  | -90.08%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.074670 | 0.021553   | 3.46x   | -71.14%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.074670 | 0.002821   | 26.47x  | -96.22%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.074670 | 0.002585   | 28.88x  | -96.54%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.108453 | 0.026538   | 4.09x   | -75.53%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.108453 | 0.003558   | 30.48x  | -96.72%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.108453 | 0.003205   | 33.83x  | -97.04%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.079969 | 0.028230   | 2.83x   | -64.70%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.079969 | 0.003225   | 24.80x  | -95.97%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.079969 | 0.003035   | 26.35x  | -96.20%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.084292 | 0.025246   | 3.34x   | -70.05%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.084292 | 0.003415   | 24.68x  | -95.95%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.084292 | 0.003142   | 26.83x  | -96.27%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.083541 | 0.026566   | 3.14x   | -68.20%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.083541 | 0.003743   | 22.32x  | -95.52%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.083541 | 0.003221   | 25.94x  | -96.14%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.080920 | 0.031877   | 2.54x   | -60.61%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.080920 | 0.004106   | 19.71x  | -94.93%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.080920 | 0.003211   | 25.20x  | -96.03%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.080557 | 0.029218   | 2.76x   | -63.73%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.080557 | 0.003146   | 25.61x  | -96.09%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.080557 | 0.003149   | 25.58x  | -96.09%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.084171 | 0.026154   | 3.22x   | -68.93%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.084171 | 0.003412   | 24.67x  | -95.95%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.084171 | 0.003299   | 25.51x  | -96.08%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.120560 | 0.042916   | 2.81x   | -64.40%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.120560 | 0.003948   | 30.53x  | -96.73%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.120560 | 0.003220   | 37.44x  | -97.33%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.116452 | 0.025918   | 4.49x   | -77.74%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.116452 | 0.003264   | 35.68x  | -97.20%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.116452 | 0.003124   | 37.28x  | -97.32%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.085641 | 0.024608   | 3.48x   | -71.27%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.085641 | 0.004476   | 19.13x  | -94.77%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.085641 | 0.003345   | 25.60x  | -96.09%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.085953 | 0.030734   | 2.80x   | -64.24%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.085953 | 0.003423   | 25.11x  | -96.02%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.085953 | 0.003170   | 27.12x  | -96.31%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.088437 | 0.030757   | 2.88x   | -65.22%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.088437 | 0.003420   | 25.86x  | -96.13%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.088437 | 0.003211   | 27.54x  | -96.37%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.087993 | 0.026460   | 3.33x   | -69.93%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.087993 | 0.003544   | 24.83x  | -95.97%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.087993 | 0.003164   | 27.81x  | -96.40%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.116998 | 0.041783   | 2.80x   | -64.29%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.116998 | 0.003781   | 30.94x  | -96.77%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.116998 | 0.003353   | 34.89x  | -97.13%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.090941 | 0.008809   | 10.32x  | -90.31%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.090941 | 0.003734   | 24.36x  | -95.89%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.090941 | 0.003072   | 29.60x  | -96.62%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.123938 | 0.011749   | 10.55x  | -90.52%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.123938 | 0.003860   | 32.11x  | -96.89%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.123938 | 0.003444   | 35.99x  | -97.22%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.101566 | 0.013370   | 7.60x   | -86.84%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.101566 | 0.004631   | 21.93x  | -95.44%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.101566 | 0.003123   | 32.53x  | -96.93%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.103321 | 0.009810   | 10.53x  | -90.51%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.103321 | 0.003765   | 27.44x  | -96.36%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.103321 | 0.003358   | 30.77x  | -96.75%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.102554 | 0.011550   | 8.88x   | -88.74%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.102554 | 0.004273   | 24.00x  | -95.83%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.102554 | 0.003458   | 29.66x  | -96.63%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.098392 | 0.026694   | 3.69x   | -72.87%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.098392 | 0.004266   | 23.06x  | -95.66%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.098392 | 0.003275   | 30.04x  | -96.67%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.096178 | 0.012337   | 7.80x   | -87.17%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.096178 | 0.003846   | 25.01x  | -96.00%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.096178 | 0.003091   | 31.11x  | -96.79%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.100016 | 0.009560   | 10.46x  | -90.44%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.100016 | 0.003776   | 26.49x  | -96.22%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.100016 | 0.003138   | 31.87x  | -96.86%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.137161 | 0.029860   | 4.59x   | -78.23%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.137161 | 0.004455   | 30.79x  | -96.75%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.137161 | 0.003212   | 42.71x  | -97.66%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.129669 | 0.009986   | 12.98x  | -92.30%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.129669 | 0.003836   | 33.81x  | -97.04%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.129669 | 0.003215   | 40.33x  | -97.52%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.097533 | 0.011958   | 8.16x   | -87.74%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.097533 | 0.004106   | 23.75x  | -95.79%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.097533 | 0.003638   | 26.81x  | -96.27%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.098339 | 0.017332   | 5.67x   | -82.38%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.098339 | 0.003766   | 26.12x  | -96.17%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.098339 | 0.003272   | 30.06x  | -96.67%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.097438 | 0.017091   | 5.70x   | -82.46%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.097438 | 0.004158   | 23.44x  | -95.73%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.097438 | 0.003078   | 31.66x  | -96.84%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.099616 | 0.010459   | 9.52x   | -89.50%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.099616 | 0.003948   | 25.23x  | -96.04%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.099616 | 0.003141   | 31.72x  | -96.85%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.128187 | 0.027799   | 4.61x   | -78.31%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.128187 | 0.003938   | 32.55x  | -96.93%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.128187 | 0.003106   | 41.27x  | -97.58%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.066090 | 0.009953   | 6.64x   | -84.94%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.066090 | 0.002969   | 22.26x  | -95.51%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.066090 | 0.003020   | 21.88x  | -95.43%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.100884 | 0.011570   | 8.72x   | -88.53%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.100884 | 0.003621   | 27.86x  | -96.41%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.100884 | 0.004597   | 21.95x  | -95.44%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.075664 | 0.013016   | 5.81x   | -82.80%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.075664 | 0.003499   | 21.63x  | -95.38%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.075664 | 0.003273   | 23.11x  | -95.67%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.077544 | 0.011010   | 7.04x   | -85.80%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.077544 | 0.003655   | 21.22x  | -95.29%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.077544 | 0.004921   | 15.76x  | -93.65%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.078507 | 0.012384   | 6.34x   | -84.23%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.078507 | 0.003837   | 20.46x  | -95.11%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.078507 | 0.005037   | 15.59x  | -93.58%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.074498 | 0.021849   | 3.41x   | -70.67%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.074498 | 0.003471   | 21.46x  | -95.34%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.074498 | 0.004367   | 17.06x  | -94.14%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.073969 | 0.012996   | 5.69x   | -82.43%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.073969 | 0.003347   | 22.10x  | -95.47%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.073969 | 0.003330   | 22.21x  | -95.50%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.076249 | 0.010405   | 7.33x   | -86.35%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.076249 | 0.003191   | 23.89x  | -95.81%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.076249 | 0.003511   | 21.72x  | -95.40%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.109947 | 0.029975   | 3.67x   | -72.74%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.109947 | 0.004107   | 26.77x  | -96.26%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.109947 | 0.004777   | 23.02x  | -95.66%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.104388 | 0.011308   | 9.23x   | -89.17%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.104388 | 0.003337   | 31.28x  | -96.80%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.104388 | 0.004329   | 24.11x  | -95.85%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.074292 | 0.014034   | 5.29x   | -81.11%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.074292 | 0.003517   | 21.13x  | -95.27%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.074292 | 0.004871   | 15.25x  | -93.44%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.075906 | 0.017326   | 4.38x   | -77.17%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.075906 | 0.003332   | 22.78x  | -95.61%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.075906 | 0.003521   | 21.56x  | -95.36%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.077394 | 0.017494   | 4.42x   | -77.40%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.077394 | 0.003586   | 21.58x  | -95.37%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.077394 | 0.004675   | 16.56x  | -93.96%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.077702 | 0.012074   | 6.44x   | -84.46%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.077702 | 0.003401   | 22.84x  | -95.62%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.077702 | 0.004602   | 16.88x  | -94.08%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.104928 | 0.028475   | 3.68x   | -72.86%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.104928 | 0.003504   | 29.95x  | -96.66%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.104928 | 0.004791   | 21.90x  | -95.43%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.393878 | 0.104528   | 3.77x   | -73.46%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.393878 | 0.044860   | 8.78x   | -88.61%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.393878 | 0.046006   | 8.56x   | -88.32%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.509161 | 0.116425   | 4.37x   | -77.13%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.509161 | 0.057390   | 8.87x   | -88.73%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.509161 | 0.053150   | 9.58x   | -89.56%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.390307 | 0.122962   | 3.17x   | -68.50%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.390307 | 0.051264   | 7.61x   | -86.87%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.390307 | 0.049196   | 7.93x   | -87.40%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.414538 | 0.111426   | 3.72x   | -73.12%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.414538 | 0.052929   | 7.83x   | -87.23%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.414538 | 0.050157   | 8.26x   | -87.90%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.422911 | 0.119565   | 3.54x   | -71.73%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.422911 | 0.057561   | 7.35x   | -86.39%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.422911 | 0.054014   | 7.83x   | -87.23%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.402317 | 0.162739   | 2.47x   | -59.55%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.402317 | 0.052023   | 7.73x   | -87.07%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.402317 | 0.050694   | 7.94x   | -87.40%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.393946 | 0.123379   | 3.19x   | -68.68%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.393946 | 0.050913   | 7.74x   | -87.08%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.393946 | 0.049125   | 8.02x   | -87.53%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.406491 | 0.113019   | 3.60x   | -72.20%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.406491 | 0.052117   | 7.80x   | -87.18%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.406491 | 0.050008   | 8.13x   | -87.70%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.566547 | 0.191390   | 2.96x   | -66.22%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.566547 | 0.057382   | 9.87x   | -89.87%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.566547 | 0.052291   | 10.83x  | -90.77%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.508820 | 0.110832   | 4.59x   | -78.22%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.508820 | 0.050640   | 10.05x  | -90.05%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.508820 | 0.049177   | 10.35x  | -90.34%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.395868 | 0.107127   | 3.70x   | -72.94%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.395868 | 0.055655   | 7.11x   | -85.94%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.395868 | 0.051026   | 7.76x   | -87.11%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.411094 | 0.138205   | 2.97x   | -66.38%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.411094 | 0.055104   | 7.46x   | -86.60%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.411094 | 0.051739   | 7.95x   | -87.41%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.411049 | 0.137325   | 2.99x   | -66.59%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.411049 | 0.056629   | 7.26x   | -86.22%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.411049 | 0.051718   | 7.95x   | -87.42%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.446365 | 0.121099   | 3.69x   | -72.87%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.446365 | 0.057447   | 7.77x   | -87.13%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.446365 | 0.052071   | 8.57x   | -88.33%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.525320 | 0.185624   | 2.83x   | -64.66%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.525320 | 0.056161   | 9.35x   | -89.31%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.525320 | 0.051986   | 10.10x  | -90.10%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.294908 | 0.091268   | 3.23x   | -69.05%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.294908 | 0.011460   | 25.73x  | -96.11%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.294908 | 0.010204   | 28.90x  | -96.54%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.412048 | 0.105412   | 3.91x   | -74.42%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.412048 | 0.014488   | 28.44x  | -96.48%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.412048 | 0.012781   | 32.24x  | -96.90%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.285422 | 0.113179   | 2.52x   | -60.35%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.285422 | 0.012757   | 22.37x  | -95.53%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.285422 | 0.012111   | 23.57x  | -95.76%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.309275 | 0.101521   | 3.05x   | -67.17%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.309275 | 0.013707   | 22.56x  | -95.57%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.309275 | 0.012454   | 24.83x  | -95.97%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.307620 | 0.106089   | 2.90x   | -65.51%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.307620 | 0.015415   | 19.96x  | -94.99%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.307620 | 0.012685   | 24.25x  | -95.88%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.291897 | 0.128515   | 2.27x   | -55.97%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.291897 | 0.016370   | 17.83x  | -94.39%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.291897 | 0.012865   | 22.69x  | -95.59%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.285738 | 0.115474   | 2.47x   | -59.59%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.285738 | 0.012412   | 23.02x  | -95.66%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.285738 | 0.012053   | 23.71x  | -95.78%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.297590 | 0.103413   | 2.88x   | -65.25%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.297590 | 0.012846   | 23.17x  | -95.68%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.297590 | 0.012310   | 24.17x  | -95.86%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.469179 | 0.169380   | 2.77x   | -63.90%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.469179 | 0.015709   | 29.87x  | -96.65%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.469179 | 0.013001   | 36.09x  | -97.23%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.412527 | 0.102142   | 4.04x   | -75.24%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.412527 | 0.012888   | 32.01x  | -96.88%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.412527 | 0.012170   | 33.90x  | -97.05%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.299801 | 0.098537   | 3.04x   | -67.13%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.299801 | 0.017432   | 17.20x  | -94.19%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.299801 | 0.013281   | 22.57x  | -95.57%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.311509 | 0.122867   | 2.54x   | -60.56%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.311509 | 0.014442   | 21.57x  | -95.36%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.311509 | 0.013185   | 23.63x  | -95.77%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.307636 | 0.123523   | 2.49x   | -59.85%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.307636 | 0.014138   | 21.76x  | -95.40%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.307636 | 0.012635   | 24.35x  | -95.89%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.316216 | 0.105281   | 3.00x   | -66.71%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.316216 | 0.014299   | 22.11x  | -95.48%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.316216 | 0.012634   | 25.03x  | -96.00%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.425511 | 0.165563   | 2.57x   | -61.09%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.425511 | 0.015513   | 27.43x  | -96.35%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.425511 | 0.012718   | 33.46x  | -97.01%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.353850 | 0.037873   | 9.34x   | -89.30%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.353850 | 0.020488   | 17.27x  | -94.21%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.353850 | 0.016518   | 21.42x  | -95.33%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.471892 | 0.047082   | 10.02x  | -90.02%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.471892 | 0.021495   | 21.95x  | -95.44%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.471892 | 0.018885   | 24.99x  | -96.00%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.349672 | 0.052119   | 6.71x   | -85.09%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.349672 | 0.020545   | 17.02x  | -94.12%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.349672 | 0.018558   | 18.84x  | -94.69%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.374285 | 0.042602   | 8.79x   | -88.62%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.374285 | 0.020547   | 18.22x  | -94.51%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.374285 | 0.018554   | 20.17x  | -95.04%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.369275 | 0.047498   | 7.77x   | -87.14%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.369275 | 0.022284   | 16.57x  | -93.97%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.369275 | 0.018706   | 19.74x  | -94.93%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.353186 | 0.109751   | 3.22x   | -68.93%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.353186 | 0.021705   | 16.27x  | -93.85%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.353186 | 0.019121   | 18.47x  | -94.59%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.360334 | 0.052732   | 6.83x   | -85.37%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.360334 | 0.020095   | 17.93x  | -94.42%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.360334 | 0.018775   | 19.19x  | -94.79%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.357948 | 0.042519   | 8.42x   | -88.12%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.357948 | 0.020381   | 17.56x  | -94.31%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.357948 | 0.018418   | 19.43x  | -94.85%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.521844 | 0.122148   | 4.27x   | -76.59%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.521844 | 0.022235   | 23.47x  | -95.74%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.521844 | 0.018896   | 27.62x  | -96.38%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.483397 | 0.041555   | 11.63x  | -91.40%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.483397 | 0.020906   | 23.12x  | -95.68%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.483397 | 0.018276   | 26.45x  | -96.22%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.360036 | 0.050542   | 7.12x   | -85.96%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.360036 | 0.021746   | 16.56x  | -93.96%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.360036 | 0.018721   | 19.23x  | -94.80%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.362614 | 0.072713   | 4.99x   | -79.95%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.362614 | 0.021256   | 17.06x  | -94.14%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.362614 | 0.018634   | 19.46x  | -94.86%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.365690 | 0.072454   | 5.05x   | -80.19%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.365690 | 0.020904   | 17.49x  | -94.28%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.365690 | 0.018711   | 19.54x  | -94.88%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.375056 | 0.045546   | 8.23x   | -87.86%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.375056 | 0.021629   | 17.34x  | -94.23%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.375056 | 0.018385   | 20.40x  | -95.10%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.490080 | 0.115740   | 4.23x   | -76.38%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.490080 | 0.021353   | 22.95x  | -95.64%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.490080 | 0.017983   | 27.25x  | -96.33%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.275993 | 0.048140   | 5.73x   | -82.56%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.275993 | 0.019162   | 14.40x  | -93.06%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.275993 | 0.018813   | 14.67x  | -93.18%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.389345 | 0.056163   | 6.93x   | -85.58%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.389345 | 0.021717   | 17.93x  | -94.42%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.389345 | 0.021202   | 18.36x  | -94.55%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.266547 | 0.060676   | 4.39x   | -77.24%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.266547 | 0.020790   | 12.82x  | -92.20%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.266547 | 0.020147   | 13.23x  | -92.44%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.291502 | 0.052528   | 5.55x   | -81.98%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.291502 | 0.022395   | 13.02x  | -92.32%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.291502 | 0.020634   | 14.13x  | -92.92%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.289435 | 0.057660   | 5.02x   | -80.08%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.289435 | 0.023063   | 12.55x  | -92.03%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.289435 | 0.021074   | 13.73x  | -92.72%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.275422 | 0.095006   | 2.90x   | -65.51%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.275422 | 0.021955   | 12.54x  | -92.03%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.275422 | 0.020593   | 13.37x  | -92.52%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.266845 | 0.059595   | 4.48x   | -77.67%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.266845 | 0.020772   | 12.85x  | -92.22%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.266845 | 0.021161   | 12.61x  | -92.07%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.272541 | 0.050340   | 5.41x   | -81.53%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.272541 | 0.021528   | 12.66x  | -92.10%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.272541 | 0.020410   | 13.35x  | -92.51%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.437978 | 0.128902   | 3.40x   | -70.57%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.437978 | 0.023519   | 18.62x  | -94.63%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.437978 | 0.020244   | 21.64x  | -95.38%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.382149 | 0.049779   | 7.68x   | -86.97%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.382149 | 0.021233   | 18.00x  | -94.44%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.382149 | 0.019906   | 19.20x  | -94.79%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.276466 | 0.063808   | 4.33x   | -76.92%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.276466 | 0.021483   | 12.87x  | -92.23%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.276466 | 0.020350   | 13.59x  | -92.64%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.289092 | 0.076469   | 3.78x   | -73.55%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.289092 | 0.021001   | 13.77x  | -92.74%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.289092 | 0.019992   | 14.46x  | -93.08%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.280021 | 0.075847   | 3.69x   | -72.91%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.280021 | 0.020508   | 13.65x  | -92.68%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.280021 | 0.020585   | 13.60x  | -92.65%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.288216 | 0.054454   | 5.29x   | -81.11%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.288216 | 0.020796   | 13.86x  | -92.78%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.288216 | 0.021004   | 13.72x  | -92.71%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.403582 | 0.120920   | 3.34x   | -70.04%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.403582 | 0.021475   | 18.79x  | -94.68%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.403582 | 0.020245   | 19.94x  | -94.98%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.083499 | 0.022916   | 3.64x   | -72.56%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.083499 | 0.009712   | 8.60x   | -88.37%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.083499 | 0.009971   | 8.37x   | -88.06%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.112748 | 0.025526   | 4.42x   | -77.36%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.112748 | 0.012230   | 9.22x   | -89.15%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.112748 | 0.011392   | 9.90x   | -89.90%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.088411 | 0.026633   | 3.32x   | -69.88%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.088411 | 0.011238   | 7.87x   | -87.29%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.088411 | 0.010809   | 8.18x   | -87.77%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.091523 | 0.024615   | 3.72x   | -73.11%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.091523 | 0.011261   | 8.13x   | -87.70%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.091523 | 0.010948   | 8.36x   | -88.04%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.091949 | 0.025792   | 3.57x   | -71.95%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.091949 | 0.013080   | 7.03x   | -85.77%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.091949 | 0.011857   | 7.75x   | -87.10%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.088053 | 0.035362   | 2.49x   | -59.84%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.088053 | 0.011474   | 7.67x   | -86.97%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.088053 | 0.010885   | 8.09x   | -87.64%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.088918 | 0.027027   | 3.29x   | -69.61%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.088918 | 0.011462   | 7.76x   | -87.11%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.088918 | 0.010659   | 8.34x   | -88.01%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.089263 | 0.024368   | 3.66x   | -72.70%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.089263 | 0.011024   | 8.10x   | -87.65%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.089263 | 0.010889   | 8.20x   | -87.80%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.119902 | 0.041456   | 2.89x   | -65.43%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.119902 | 0.012459   | 9.62x   | -89.61%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.119902 | 0.011496   | 10.43x  | -90.41%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.116565 | 0.024657   | 4.73x   | -78.85%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.116565 | 0.011285   | 10.33x  | -90.32%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.116565 | 0.010856   | 10.74x  | -90.69%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.086306 | 0.023977   | 3.60x   | -72.22%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.086306 | 0.012013   | 7.18x   | -86.08%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.086306 | 0.011201   | 7.71x   | -87.02%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.090701 | 0.029700   | 3.05x   | -67.25%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.090701 | 0.012021   | 7.55x   | -86.75%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.090701 | 0.011136   | 8.15x   | -87.72%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.089878 | 0.029597   | 3.04x   | -67.07%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.089878 | 0.012084   | 7.44x   | -86.56%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.089878 | 0.011173   | 8.04x   | -87.57%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.093010 | 0.024763   | 3.76x   | -73.38%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.093010 | 0.012333   | 7.54x   | -86.74%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.093010 | 0.011511   | 8.08x   | -87.62%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.114420 | 0.040370   | 2.83x   | -64.72%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.114420 | 0.012201   | 9.38x   | -89.34%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.114420 | 0.011363   | 10.07x  | -90.07%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.065189 | 0.018650   | 3.50x   | -71.39%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.065189 | 0.002432   | 26.81x  | -96.27%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.065189 | 0.002209   | 29.50x  | -96.61%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.100499 | 0.022998   | 4.37x   | -77.12%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.100499 | 0.003202   | 31.39x  | -96.81%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.100499 | 0.002857   | 35.17x  | -97.16%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.075653 | 0.024591   | 3.08x   | -67.49%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.075653 | 0.002755   | 27.46x  | -96.36%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.075653 | 0.002671   | 28.33x  | -96.47%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.080177 | 0.022221   | 3.61x   | -72.29%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.080177 | 0.003102   | 25.85x  | -96.13%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.080177 | 0.002671   | 30.02x  | -96.67%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.079034 | 0.023278   | 3.40x   | -70.55%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.079034 | 0.003407   | 23.20x  | -95.69%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.079034 | 0.002764   | 28.59x  | -96.50%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.075671 | 0.028114   | 2.69x   | -62.85%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.075671 | 0.003551   | 21.31x  | -95.31%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.075671 | 0.002837   | 26.67x  | -96.25%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.076179 | 0.025130   | 3.03x   | -67.01%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.076179 | 0.002682   | 28.40x  | -96.48%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.076179 | 0.002607   | 29.22x  | -96.58%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.077836 | 0.022369   | 3.48x   | -71.26%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.077836 | 0.002862   | 27.19x  | -96.32%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.077836 | 0.002888   | 26.96x  | -96.29%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.108796 | 0.037039   | 2.94x   | -65.96%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.108796 | 0.003359   | 32.39x  | -96.91%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.108796 | 0.002804   | 38.80x  | -97.42%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.104218 | 0.022062   | 4.72x   | -78.83%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.104218 | 0.002710   | 38.46x  | -97.40%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.104218 | 0.002653   | 39.28x  | -97.45%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.074558 | 0.020970   | 3.56x   | -71.87%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.074558 | 0.003825   | 19.49x  | -94.87%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.074558 | 0.003036   | 24.56x  | -95.93%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.077487 | 0.026477   | 2.93x   | -65.83%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.077487 | 0.002963   | 26.15x  | -96.18%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.077487 | 0.002707   | 28.62x  | -96.51%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.078056 | 0.026552   | 2.94x   | -65.98%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.078056 | 0.003026   | 25.79x  | -96.12%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.078056 | 0.002811   | 27.76x  | -96.40%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.079479 | 0.022610   | 3.52x   | -71.55%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.079479 | 0.003047   | 26.08x  | -96.17%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.079479 | 0.002733   | 29.08x  | -96.56%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.103683 | 0.035599   | 2.91x   | -65.67%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.103683 | 0.003240   | 32.00x  | -96.88%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.103683 | 0.002779   | 37.31x  | -97.32%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.072272 | 0.007107   | 10.17x  | -90.17%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.072272 | 0.003210   | 22.51x  | -95.56%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.072272 | 0.002291   | 31.55x  | -96.83%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.104873 | 0.008727   | 12.02x  | -91.68%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.104873 | 0.003294   | 31.84x  | -96.86%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.104873 | 0.002693   | 38.94x  | -97.43%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.081934 | 0.010247   | 8.00x   | -87.49%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.081934 | 0.003007   | 27.25x  | -96.33%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.081934 | 0.002624   | 31.22x  | -96.80%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.084249 | 0.008079   | 10.43x  | -90.41%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.084249 | 0.003292   | 25.59x  | -96.09%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.084249 | 0.002597   | 32.44x  | -96.92%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.084356 | 0.009224   | 9.15x   | -89.07%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.084356 | 0.003455   | 24.41x  | -95.90%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.084356 | 0.002848   | 29.62x  | -96.62%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.080892 | 0.022315   | 3.63x   | -72.41%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.080892 | 0.003506   | 23.07x  | -95.67%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.080892 | 0.002725   | 29.68x  | -96.63%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.082360 | 0.010508   | 7.84x   | -87.24%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.082360 | 0.003126   | 26.35x  | -96.21%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.082360 | 0.002626   | 31.36x  | -96.81%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.084189 | 0.008280   | 10.17x  | -90.16%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.084189 | 0.003105   | 27.12x  | -96.31%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.084189 | 0.002599   | 32.39x  | -96.91%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.115484 | 0.025283   | 4.57x   | -78.11%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.115484 | 0.003668   | 31.49x  | -96.82%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.115484 | 0.002968   | 38.91x  | -97.43%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.109842 | 0.007818   | 14.05x  | -92.88%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.109842 | 0.003226   | 34.05x  | -97.06%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.109842 | 0.002742   | 40.06x  | -97.50%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.080303 | 0.009825   | 8.17x   | -87.76%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.080303 | 0.003523   | 22.79x  | -95.61%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.080303 | 0.002958   | 27.14x  | -96.32%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.083199 | 0.014668   | 5.67x   | -82.37%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.083199 | 0.003419   | 24.34x  | -95.89%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.083199 | 0.002757   | 30.18x  | -96.69%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.083843 | 0.014653   | 5.72x   | -82.52%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.083843 | 0.003315   | 25.29x  | -96.05%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.083843 | 0.002660   | 31.52x  | -96.83%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.086199 | 0.008896   | 9.69x   | -89.68%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.086199 | 0.003432   | 25.12x  | -96.02%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.086199 | 0.002947   | 29.25x  | -96.58%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.107808 | 0.024155   | 4.46x   | -77.59%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.107808 | 0.003379   | 31.91x  | -96.87%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.107808 | 0.002711   | 39.77x  | -97.49%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.058377 | 0.008781   | 6.65x   | -84.96%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.058377 | 0.002573   | 22.69x  | -95.59%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.058377 | 0.002476   | 23.58x  | -95.76%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.091170 | 0.010163   | 8.97x   | -88.85%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.091170 | 0.002944   | 30.97x  | -96.77%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.091170 | 0.004389   | 20.77x  | -95.19%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.067559 | 0.011562   | 5.84x   | -82.89%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.067559 | 0.003134   | 21.56x  | -95.36%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.067559 | 0.004511   | 14.98x  | -93.32%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.071181 | 0.009881   | 7.20x   | -86.12%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.071181 | 0.003179   | 22.39x  | -95.53%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.071181 | 0.004870   | 14.62x  | -93.16%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.070931 | 0.011031   | 6.43x   | -84.45%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.070931 | 0.003369   | 21.05x  | -95.25%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.070931 | 0.004575   | 15.50x  | -93.55%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.066766 | 0.019299   | 3.46x   | -71.09%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.066766 | 0.003013   | 22.16x  | -95.49%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.066766 | 0.004642   | 14.38x  | -93.05%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.067745 | 0.011290   | 6.00x   | -83.34%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.067745 | 0.002793   | 24.25x  | -95.88%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.067745 | 0.004696   | 14.43x  | -93.07%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.068703 | 0.009376   | 7.33x   | -86.35%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.068703 | 0.002863   | 24.00x  | -95.83%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.068703 | 0.004351   | 15.79x  | -93.67%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.100111 | 0.026451   | 3.78x   | -73.58%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.100111 | 0.003365   | 29.75x  | -96.64%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.100111 | 0.004371   | 22.91x  | -95.63%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.094149 | 0.009369   | 10.05x  | -90.05%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.094149 | 0.002838   | 33.17x  | -96.99%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.094149 | 0.004424   | 21.28x  | -95.30%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.066358 | 0.012397   | 5.35x   | -81.32%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.066358 | 0.003261   | 20.35x  | -95.09%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.066358 | 0.004656   | 14.25x  | -92.98%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.069759 | 0.015321   | 4.55x   | -78.04%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.069759 | 0.003025   | 23.06x  | -95.66%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.069759 | 0.004690   | 14.88x  | -93.28%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.070275 | 0.015162   | 4.64x   | -78.43%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.070275 | 0.003220   | 21.83x  | -95.42%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.070275 | 0.004559   | 15.42x  | -93.51%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.072620 | 0.010426   | 6.97x   | -85.64%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.072620 | 0.003055   | 23.77x  | -95.79%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.072620 | 0.004393   | 16.53x  | -93.95%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.095409 | 0.025633   | 3.72x   | -73.13%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.095409 | 0.003177   | 30.03x  | -96.67%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.095409 | 0.003439   | 27.74x  | -96.40%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.188792 | 0.052641   | 3.59x   | -72.12%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.188792 | 0.022348   | 8.45x   | -88.16%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.188792 | 0.023470   | 8.04x   | -87.57%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.251787 | 0.058431   | 4.31x   | -76.79%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.251787 | 0.027662   | 9.10x   | -89.01%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.251787 | 0.026273   | 9.58x   | -89.57%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.195903 | 0.060680   | 3.23x   | -69.03%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.195903 | 0.025043   | 7.82x   | -87.22%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.195903 | 0.024398   | 8.03x   | -87.55%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.203467 | 0.055988   | 3.63x   | -72.48%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.203467 | 0.025882   | 7.86x   | -87.28%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.203467 | 0.024872   | 8.18x   | -87.78%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.203152 | 0.058947   | 3.45x   | -70.98%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.203152 | 0.028580   | 7.11x   | -85.93%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.203152 | 0.027222   | 7.46x   | -86.60%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.201484 | 0.081635   | 2.47x   | -59.48%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.201484 | 0.026136   | 7.71x   | -87.03%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.201484 | 0.025232   | 7.99x   | -87.48%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.197552 | 0.061093   | 3.23x   | -69.07%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.197552 | 0.026097   | 7.57x   | -86.79%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.197552 | 0.024779   | 7.97x   | -87.46%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.201168 | 0.056434   | 3.56x   | -71.95%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.201168 | 0.025511   | 7.89x   | -87.32%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.201168 | 0.025349   | 7.94x   | -87.40%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.269898 | 0.096438   | 2.80x   | -64.27%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.269898 | 0.028816   | 9.37x   | -89.32%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.269898 | 0.026196   | 10.30x  | -90.29%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.260917 | 0.057389   | 4.55x   | -78.00%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.260917 | 0.026015   | 10.03x  | -90.03%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.260917 | 0.024916   | 10.47x  | -90.45%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.198843 | 0.054658   | 3.64x   | -72.51%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.198843 | 0.027433   | 7.25x   | -86.20%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.198843 | 0.025764   | 7.72x   | -87.04%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.206955 | 0.067083   | 3.09x   | -67.59%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.206955 | 0.027577   | 7.50x   | -86.67%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.206955 | 0.025533   | 8.11x   | -87.66%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.202478 | 0.066875   | 3.03x   | -66.97%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.202478 | 0.026866   | 7.54x   | -86.73%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.202478 | 0.025798   | 7.85x   | -87.26%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.206185 | 0.056372   | 3.66x   | -72.66%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.206185 | 0.027566   | 7.48x   | -86.63%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.206185 | 0.025629   | 8.05x   | -87.57%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.259265 | 0.091917   | 2.82x   | -64.55%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.259265 | 0.027936   | 9.28x   | -89.23%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.259265 | 0.025743   | 10.07x  | -90.07%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.140571 | 0.042250   | 3.33x   | -69.94%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.140571 | 0.005467   | 25.71x  | -96.11%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.140571 | 0.005175   | 27.16x  | -96.32%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.200525 | 0.052658   | 3.81x   | -73.74%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.200525 | 0.007346   | 27.30x  | -96.34%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.200525 | 0.006272   | 31.97x  | -96.87%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.148752 | 0.056145   | 2.65x   | -62.26%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.148752 | 0.006179   | 24.08x  | -95.85%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.148752 | 0.006028   | 24.68x  | -95.95%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.158196 | 0.051301   | 3.08x   | -67.57%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.158196 | 0.006694   | 23.63x  | -95.77%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.158196 | 0.006165   | 25.66x  | -96.10%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.154925 | 0.054261   | 2.86x   | -64.98%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.154925 | 0.008045   | 19.26x  | -94.81%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.154925 | 0.006128   | 25.28x  | -96.04%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.149927 | 0.063585   | 2.36x   | -57.59%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.149927 | 0.008235   | 18.21x  | -94.51%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.149927 | 0.006261   | 23.95x  | -95.82%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.150059 | 0.056136   | 2.67x   | -62.59%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.150059 | 0.006197   | 24.21x  | -95.87%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.150059 | 0.005893   | 25.46x  | -96.07%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.150840 | 0.050658   | 2.98x   | -66.42%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.150840 | 0.006418   | 23.50x  | -95.75%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.150840 | 0.006175   | 24.43x  | -95.91%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.219917 | 0.083192   | 2.64x   | -62.17%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.219917 | 0.007509   | 29.29x  | -96.59%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.219917 | 0.006285   | 34.99x  | -97.14%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.208284 | 0.051195   | 4.07x   | -75.42%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.208284 | 0.006232   | 33.42x  | -97.01%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.208284 | 0.006124   | 34.01x  | -97.06%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.150256 | 0.049709   | 3.02x   | -66.92%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.150256 | 0.008367   | 17.96x  | -94.43%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.150256 | 0.006472   | 23.22x  | -95.69%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.151305 | 0.060739   | 2.49x   | -59.86%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.151305 | 0.006787   | 22.29x  | -95.51%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.151305 | 0.006051   | 25.00x  | -96.00%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.153032 | 0.060312   | 2.54x   | -60.59%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.153032 | 0.006770   | 22.61x  | -95.58%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.153032 | 0.006169   | 24.81x  | -95.97%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.155712 | 0.051351   | 3.03x   | -67.02%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.155712 | 0.006997   | 22.25x  | -95.51%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.155712 | 0.006212   | 25.07x  | -96.01%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.208373 | 0.081163   | 2.57x   | -61.05%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.208373 | 0.007253   | 28.73x  | -96.52%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.208373 | 0.006378   | 32.67x  | -96.94%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.167324 | 0.016796   | 9.96x   | -89.96%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.167324 | 0.007143   | 23.42x  | -95.73%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.167324 | 0.005747   | 29.11x  | -96.57%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.226462 | 0.021179   | 10.69x  | -90.65%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.226462 | 0.007801   | 29.03x  | -96.56%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.226462 | 0.006191   | 36.58x  | -97.27%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.173513 | 0.024164   | 7.18x   | -86.07%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.173513 | 0.007265   | 23.88x  | -95.81%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.173513 | 0.006128   | 28.31x  | -96.47%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.185180 | 0.019630   | 9.43x   | -89.40%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.185180 | 0.007256   | 25.52x  | -96.08%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.185180 | 0.006351   | 29.16x  | -96.57%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.178350 | 0.021936   | 8.13x   | -87.70%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.178350 | 0.008059   | 22.13x  | -95.48%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.178350 | 0.006416   | 27.80x  | -96.40%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.207940 | 0.051047   | 4.07x   | -75.45%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.207940 | 0.007768   | 26.77x  | -96.26%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.207940 | 0.006057   | 34.33x  | -97.09%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.170622 | 0.023430   | 7.28x   | -86.27%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.170622 | 0.007214   | 23.65x  | -95.77%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.170622 | 0.006323   | 26.98x  | -96.29%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.174329 | 0.018653   | 9.35x   | -89.30%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.174329 | 0.007036   | 24.78x  | -95.96%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.174329 | 0.006016   | 28.98x  | -96.55%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.244157 | 0.057838   | 4.22x   | -76.31%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.244157 | 0.007883   | 30.97x  | -96.77%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.244157 | 0.006030   | 40.49x  | -97.53%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.233008 | 0.018801   | 12.39x  | -91.93%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.233008 | 0.007786   | 29.93x  | -96.66%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.233008 | 0.005808   | 40.12x  | -97.51%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.176184 | 0.023362   | 7.54x   | -86.74%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.176184 | 0.008105   | 21.74x  | -95.40%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.176184 | 0.006038   | 29.18x  | -96.57%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.178446 | 0.033749   | 5.29x   | -81.09%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.178446 | 0.007538   | 23.67x  | -95.78%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.178446 | 0.006098   | 29.26x  | -96.58%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.180985 | 0.034468   | 5.25x   | -80.96%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.180985 | 0.007897   | 22.92x  | -95.64%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.180985 | 0.006326   | 28.61x  | -96.50%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.186350 | 0.020565   | 9.06x   | -88.96%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.186350 | 0.008411   | 22.15x  | -95.49%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.186350 | 0.006095   | 30.57x  | -96.73%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.235925 | 0.054574   | 4.32x   | -76.87%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.235925 | 0.008122   | 29.05x  | -96.56%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.235925 | 0.005981   | 39.45x  | -97.46%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.128811 | 0.019494   | 6.61x   | -84.87%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.128811 | 0.005676   | 22.69x  | -95.59%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.128811 | 0.006010   | 21.43x  | -95.33%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.191589 | 0.022488   | 8.52x   | -88.26%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.191589 | 0.006298   | 30.42x  | -96.71%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.191589 | 0.006054   | 31.65x  | -96.84%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.135676 | 0.025930   | 5.23x   | -80.89%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.135676 | 0.006125   | 22.15x  | -95.49%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.135676 | 0.006778   | 20.02x  | -95.00%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.144283 | 0.021919   | 6.58x   | -84.81%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.144283 | 0.007232   | 19.95x  | -94.99%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.144283 | 0.007698   | 18.74x  | -94.66%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.145749 | 0.024862   | 5.86x   | -82.94%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.145749 | 0.007554   | 19.29x  | -94.82%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.145749 | 0.008056   | 18.09x  | -94.47%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.137747 | 0.043078   | 3.20x   | -68.73%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.137747 | 0.006732   | 20.46x  | -95.11%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.137747 | 0.009161   | 15.04x  | -93.35%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.134666 | 0.025440   | 5.29x   | -81.11%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.134666 | 0.006059   | 22.23x  | -95.50%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.134666 | 0.007334   | 18.36x  | -94.55%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.138441 | 0.020908   | 6.62x   | -84.90%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.138441 | 0.006038   | 22.93x  | -95.64%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.138441 | 0.008030   | 17.24x  | -94.20%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.207388 | 0.059158   | 3.51x   | -71.47%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.207388 | 0.007469   | 27.77x  | -96.40%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.207388 | 0.008616   | 24.07x  | -95.85%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.195984 | 0.020439   | 9.59x   | -89.57%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.195984 | 0.006539   | 29.97x  | -96.66%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.195984 | 0.007766   | 25.24x  | -96.04%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.140438 | 0.027364   | 5.13x   | -80.52%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.140438 | 0.006589   | 21.31x  | -95.31%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.140438 | 0.008248   | 17.03x  | -94.13%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.140804 | 0.034380   | 4.10x   | -75.58%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.140804 | 0.006163   | 22.85x  | -95.62%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.140804 | 0.008569   | 16.43x  | -93.91%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.143887 | 0.033916   | 4.24x   | -76.43%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.143887 | 0.006115   | 23.53x  | -95.75%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.143887 | 0.008093   | 17.78x  | -94.38%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.145251 | 0.023270   | 6.24x   | -83.98%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.145251 | 0.007060   | 20.57x  | -95.14%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.145251 | 0.007137   | 20.35x  | -95.09%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.199085 | 0.056160   | 3.54x   | -71.79%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.199085 | 0.006584   | 30.24x  | -96.69%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.199085 | 0.008362   | 23.81x  | -95.80%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.341750 | 0.092595   | 3.69x   | -72.91%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.341750 | 0.039548   | 8.64x   | -88.43%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.341750 | 0.039814   | 8.58x   | -88.35%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.452435 | 0.102874   | 4.40x   | -77.26%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.452435 | 0.049925   | 9.06x   | -88.97%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.452435 | 0.045712   | 9.90x   | -89.90%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.342840 | 0.107118   | 3.20x   | -68.76%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.342840 | 0.044488   | 7.71x   | -87.02%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.342840 | 0.042847   | 8.00x   | -87.50%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.360471 | 0.098138   | 3.67x   | -72.78%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.360471 | 0.045710   | 7.89x   | -87.32%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.360471 | 0.043936   | 8.20x   | -87.81%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.366123 | 0.103423   | 3.54x   | -71.75%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.366123 | 0.050145   | 7.30x   | -86.30%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.366123 | 0.046758   | 7.83x   | -87.23%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.350988 | 0.143200   | 2.45x   | -59.20%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.350988 | 0.046533   | 7.54x   | -86.74%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.350988 | 0.043779   | 8.02x   | -87.53%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.343212 | 0.107938   | 3.18x   | -68.55%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.343212 | 0.044903   | 7.64x   | -86.92%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.343212 | 0.042705   | 8.04x   | -87.56%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.354614 | 0.100425   | 3.53x   | -71.68%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.354614 | 0.045065   | 7.87x   | -87.29%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.354614 | 0.043741   | 8.11x   | -87.67%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.499654 | 0.170913   | 2.92x   | -65.79%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.499654 | 0.051520   | 9.70x   | -89.69%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.499654 | 0.047230   | 10.58x  | -90.55%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.450714 | 0.099292   | 4.54x   | -77.97%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.450714 | 0.045967   | 9.81x   | -89.80%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.450714 | 0.046779   | 9.63x   | -89.62%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.354458 | 0.098554   | 3.60x   | -72.20%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.354458 | 0.049904   | 7.10x   | -85.92%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.354458 | 0.046621   | 7.60x   | -86.85%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.362281 | 0.123254   | 2.94x   | -65.98%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.362281 | 0.050190   | 7.22x   | -86.15%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.362281 | 0.046216   | 7.84x   | -87.24%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.361725 | 0.121597   | 2.97x   | -66.38%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.361725 | 0.049340   | 7.33x   | -86.36%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.361725 | 0.046988   | 7.70x   | -87.01%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.367042 | 0.105173   | 3.49x   | -71.35%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.367042 | 0.049607   | 7.40x   | -86.48%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.367042 | 0.046041   | 7.97x   | -87.46%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.459598 | 0.163446   | 2.81x   | -64.44%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.459598 | 0.049310   | 9.32x   | -89.27%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.459598 | 0.045682   | 10.06x  | -90.06%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.250229 | 0.075790   | 3.30x   | -69.71%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.250229 | 0.010457   | 23.93x  | -95.82%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.250229 | 0.008956   | 27.94x  | -96.42%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.361162 | 0.092992   | 3.88x   | -74.25%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.361162 | 0.012737   | 28.36x  | -96.47%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.361162 | 0.011201   | 32.24x  | -96.90%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.254590 | 0.102131   | 2.49x   | -59.88%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.254590 | 0.011231   | 22.67x  | -95.59%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.254590 | 0.010764   | 23.65x  | -95.77%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.273978 | 0.090018   | 3.04x   | -67.14%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.273978 | 0.012114   | 22.62x  | -95.58%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.273978 | 0.010870   | 25.20x  | -96.03%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.274110 | 0.094392   | 2.90x   | -65.56%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.274110 | 0.014416   | 19.01x  | -94.74%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.274110 | 0.011259   | 24.35x  | -95.89%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.265217 | 0.115436   | 2.30x   | -56.47%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.265217 | 0.014911   | 17.79x  | -94.38%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.265217 | 0.011446   | 23.17x  | -95.68%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.261158 | 0.103117   | 2.53x   | -60.52%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.261158 | 0.011048   | 23.64x  | -95.77%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.261158 | 0.010636   | 24.55x  | -95.93%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.257959 | 0.090711   | 2.84x   | -64.84%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.257959 | 0.011505   | 22.42x  | -95.54%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.257959 | 0.011073   | 23.30x  | -95.71%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.422412 | 0.149799   | 2.82x   | -64.54%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.422412 | 0.013709   | 30.81x  | -96.75%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.422412 | 0.011200   | 37.71x  | -97.35%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.356230 | 0.089774   | 3.97x   | -74.80%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.356230 | 0.011310   | 31.50x  | -96.83%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.356230 | 0.010904   | 32.67x  | -96.94%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.261693 | 0.086654   | 3.02x   | -66.89%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.261693 | 0.015193   | 17.22x  | -94.19%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.261693 | 0.011876   | 22.03x  | -95.46%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.270340 | 0.109574   | 2.47x   | -59.47%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.270340 | 0.012199   | 22.16x  | -95.49%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.270340 | 0.011163   | 24.22x  | -95.87%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.274284 | 0.108570   | 2.53x   | -60.42%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.274284 | 0.012761   | 21.49x  | -95.35%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.274284 | 0.010995   | 24.95x  | -95.99%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.276407 | 0.096144   | 2.87x   | -65.22%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.276407 | 0.013275   | 20.82x  | -95.20%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.276407 | 0.011157   | 24.77x  | -95.96%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.382683 | 0.146874   | 2.61x   | -61.62%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.382683 | 0.013549   | 28.24x  | -96.46%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.382683 | 0.011319   | 33.81x  | -97.04%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.313189 | 0.028609   | 10.95x  | -90.87%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.313189 | 0.013362   | 23.44x  | -95.73%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.313189 | 0.009408   | 33.29x  | -97.00%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.416157 | 0.036390   | 11.44x  | -91.26%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.416157 | 0.014658   | 28.39x  | -96.48%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.416157 | 0.011436   | 36.39x  | -97.25%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.303200 | 0.042512   | 7.13x   | -85.98%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.303200 | 0.013059   | 23.22x  | -95.69%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.303200 | 0.011640   | 26.05x  | -96.16%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.323173 | 0.033246   | 9.72x   | -89.71%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.323173 | 0.013266   | 24.36x  | -95.90%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.323173 | 0.011056   | 29.23x  | -96.58%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.325303 | 0.037975   | 8.57x   | -88.33%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.325303 | 0.014742   | 22.07x  | -95.47%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.325303 | 0.011371   | 28.61x  | -96.50%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.320344 | 0.091228   | 3.51x   | -71.52%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.320344 | 0.014240   | 22.50x  | -95.55%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.320344 | 0.011717   | 27.34x  | -96.34%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.309303 | 0.041559   | 7.44x   | -86.56%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.309303 | 0.012336   | 25.07x  | -96.01%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.309303 | 0.011260   | 27.47x  | -96.36%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.314254 | 0.032160   | 9.77x   | -89.77%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.314254 | 0.012483   | 25.17x  | -96.03%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.314254 | 0.011008   | 28.55x  | -96.50%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.474209 | 0.101083   | 4.69x   | -78.68%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.474209 | 0.014294   | 33.18x  | -96.99%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.474209 | 0.011071   | 42.83x  | -97.67%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.409102 | 0.032015   | 12.78x  | -92.17%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.409102 | 0.013199   | 31.00x  | -96.77%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.409102 | 0.011469   | 35.67x  | -97.20%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.311564 | 0.041127   | 7.58x   | -86.80%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.311564 | 0.014410   | 21.62x  | -95.37%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.311564 | 0.011700   | 26.63x  | -96.24%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.314930 | 0.059503   | 5.29x   | -81.11%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.314930 | 0.013760   | 22.89x  | -95.63%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.314930 | 0.011017   | 28.59x  | -96.50%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.315943 | 0.058402   | 5.41x   | -81.52%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.315943 | 0.013229   | 23.88x  | -95.81%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.315943 | 0.011455   | 27.58x  | -96.37%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.322589 | 0.035230   | 9.16x   | -89.08%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.322589 | 0.013933   | 23.15x  | -95.68%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.322589 | 0.011161   | 28.90x  | -96.54%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.429014 | 0.098393   | 4.36x   | -77.07%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.429014 | 0.014305   | 29.99x  | -96.67%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.429014 | 0.011430   | 37.54x  | -97.34%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.246939 | 0.042421   | 5.82x   | -82.82%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.246939 | 0.016954   | 14.56x  | -93.13%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.246939 | 0.016344   | 15.11x  | -93.38%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.360581 | 0.048876   | 7.38x   | -86.45%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.360581 | 0.019363   | 18.62x  | -94.63%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.360581 | 0.017789   | 20.27x  | -95.07%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.244263 | 0.053330   | 4.58x   | -78.17%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.244263 | 0.018508   | 13.20x  | -92.42%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.244263 | 0.018075   | 13.51x  | -92.60%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.266655 | 0.045452   | 5.87x   | -82.95%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.266655 | 0.018082   | 14.75x  | -93.22%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.266655 | 0.017852   | 14.94x  | -93.31%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.262199 | 0.049448   | 5.30x   | -81.14%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.262199 | 0.019872   | 13.19x  | -92.42%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.262199 | 0.018343   | 14.29x  | -93.00%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.246502 | 0.083285   | 2.96x   | -66.21%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.246502 | 0.018447   | 13.36x  | -92.52%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.246502 | 0.017803   | 13.85x  | -92.78%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.239961 | 0.052017   | 4.61x   | -78.32%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.239961 | 0.017847   | 13.45x  | -92.56%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.239961 | 0.017445   | 13.76x  | -92.73%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.252676 | 0.043872   | 5.76x   | -82.64%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.252676 | 0.017421   | 14.50x  | -93.11%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.252676 | 0.017423   | 14.50x  | -93.10%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.394771 | 0.111848   | 3.53x   | -71.67%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.394771 | 0.020150   | 19.59x  | -94.90%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.394771 | 0.017530   | 22.52x  | -95.56%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.348547 | 0.043761   | 7.96x   | -87.44%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.348547 | 0.017373   | 20.06x  | -95.02%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.348547 | 0.018123   | 19.23x  | -94.80%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.249966 | 0.056058   | 4.46x   | -77.57%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.249966 | 0.019291   | 12.96x  | -92.28%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.249966 | 0.017754   | 14.08x  | -92.90%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.258967 | 0.066676   | 3.88x   | -74.25%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.258967 | 0.017773   | 14.57x  | -93.14%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.258967 | 0.017676   | 14.65x  | -93.17%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.259600 | 0.067078   | 3.87x   | -74.16%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.259600 | 0.018635   | 13.93x  | -92.82%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.259600 | 0.018071   | 14.37x  | -93.04%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.266703 | 0.047286   | 5.64x   | -82.27%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.266703 | 0.018033   | 14.79x  | -93.24%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.266703 | 0.017930   | 14.88x  | -93.28%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.363181 | 0.106366   | 3.41x   | -70.71%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.363181 | 0.018709   | 19.41x  | -94.85%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.363181 | 0.017686   | 20.54x  | -95.13%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.768948 | 0.209657   | 3.67x   | -72.73%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.768948 | 0.088208   | 8.72x   | -88.53%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.768948 | 0.091554   | 8.40x   | -88.09%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 0.999666 | 0.231761   | 4.31x   | -76.82%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 0.999666 | 0.108844   | 9.18x   | -89.11%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 0.999666 | 0.101502   | 9.85x   | -89.85%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.759913 | 0.245710   | 3.09x   | -67.67%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.759913 | 0.101450   | 7.49x   | -86.65%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.759913 | 0.099659   | 7.63x   | -86.89%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.816827 | 0.226459   | 3.61x   | -72.28%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.816827 | 0.107004   | 7.63x   | -86.90%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.816827 | 0.100518   | 8.13x   | -87.69%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.821389 | 0.237451   | 3.46x   | -71.09%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.821389 | 0.114111   | 7.20x   | -86.11%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.821389 | 0.104840   | 7.83x   | -87.24%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.792950 | 0.328167   | 2.42x   | -58.61%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.792950 | 0.103866   | 7.63x   | -86.90%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.792950 | 0.100432   | 7.90x   | -87.33%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.771287 | 0.246452   | 3.13x   | -68.05%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.771287 | 0.100903   | 7.64x   | -86.92%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.771287 | 0.097266   | 7.93x   | -87.39%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.785611 | 0.224460   | 3.50x   | -71.43%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.785611 | 0.102417   | 7.67x   | -86.96%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.785611 | 0.099567   | 7.89x   | -87.33%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.111962 | 0.382638   | 2.91x   | -65.59%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.111962 | 0.113792   | 9.77x   | -89.77%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.111962 | 0.103987   | 10.69x  | -90.65%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 1.038627 | 0.224535   | 4.63x   | -78.38%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 1.038627 | 0.101977   | 10.18x  | -90.18%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 1.038627 | 0.097778   | 10.62x  | -90.59%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.793940 | 0.218121   | 3.64x   | -72.53%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.793940 | 0.113740   | 6.98x   | -85.67%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.793940 | 0.103618   | 7.66x   | -86.95%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.808542 | 0.279056   | 2.90x   | -65.49%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.808542 | 0.110480   | 7.32x   | -86.34%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.808542 | 0.103357   | 7.82x   | -87.22%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.801495 | 0.272401   | 2.94x   | -66.01%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.801495 | 0.111766   | 7.17x   | -86.06%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.801495 | 0.102789   | 7.80x   | -87.18%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.804225 | 0.230816   | 3.48x   | -71.30%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.804225 | 0.113042   | 7.11x   | -85.94%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.804225 | 0.104815   | 7.67x   | -86.97%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 1.020451 | 0.369756   | 2.76x   | -63.77%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 1.020451 | 0.112180   | 9.10x   | -89.01%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 1.020451 | 0.103271   | 9.88x   | -89.88%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.577405 | 0.173287   | 3.33x   | -69.99%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.577405 | 0.022308   | 25.88x  | -96.14%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.577405 | 0.020027   | 28.83x  | -96.53%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.813887 | 0.212692   | 3.83x   | -73.87%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.813887 | 0.029092   | 27.98x  | -96.43%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.813887 | 0.026284   | 30.96x  | -96.77%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.585153 | 0.231552   | 2.53x   | -60.43%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.585153 | 0.025543   | 22.91x  | -95.63%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.585153 | 0.024171   | 24.21x  | -95.87%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.626360 | 0.205338   | 3.05x   | -67.22%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.626360 | 0.027694   | 22.62x  | -95.58%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.626360 | 0.025240   | 24.82x  | -95.97%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.618345 | 0.212459   | 2.91x   | -65.64%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.618345 | 0.030016   | 20.60x  | -95.15%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.618345 | 0.024891   | 24.84x  | -95.97%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.588861 | 0.254758   | 2.31x   | -56.74%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.588861 | 0.032397   | 18.18x  | -94.50%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.588861 | 0.024755   | 23.79x  | -95.80%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.569623 | 0.226342   | 2.52x   | -60.26%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.569623 | 0.024369   | 23.37x  | -95.72%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.569623 | 0.024184   | 23.55x  | -95.75%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.581622 | 0.203543   | 2.86x   | -65.00%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.581622 | 0.025228   | 23.05x  | -95.66%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.581622 | 0.024237   | 24.00x  | -95.83%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.900851 | 0.335075   | 2.69x   | -62.80%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.900851 | 0.030278   | 29.75x  | -96.64%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.900851 | 0.025103   | 35.89x  | -97.21%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.811731 | 0.200910   | 4.04x   | -75.25%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.811731 | 0.024918   | 32.58x  | -96.93%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.811731 | 0.024211   | 33.53x  | -97.02%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.584768 | 0.195104   | 3.00x   | -66.64%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.584768 | 0.034135   | 17.13x  | -94.16%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.584768 | 0.025994   | 22.50x  | -95.55%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.607024 | 0.242655   | 2.50x   | -60.03%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.607024 | 0.027571   | 22.02x  | -95.46%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.607024 | 0.024845   | 24.43x  | -95.91%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.609408 | 0.243690   | 2.50x   | -60.01%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.609408 | 0.027024   | 22.55x  | -95.57%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.609408 | 0.024772   | 24.60x  | -95.94%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.619493 | 0.207143   | 2.99x   | -66.56%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.619493 | 0.027983   | 22.14x  | -95.48%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.619493 | 0.024890   | 24.89x  | -95.98%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.841947 | 0.322025   | 2.61x   | -61.75%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.841947 | 0.028944   | 29.09x  | -96.56%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.841947 | 0.027865   | 30.22x  | -96.69%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.674566 | 0.073720   | 9.15x   | -89.07%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.674566 | 0.040504   | 16.65x  | -94.00%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.674566 | 0.031184   | 21.63x  | -95.38%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.896392 | 0.089472   | 10.02x  | -90.02%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.896392 | 0.041194   | 21.76x  | -95.40%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.896392 | 0.034908   | 25.68x  | -96.11%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.658061 | 0.102284   | 6.43x   | -84.46%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.658061 | 0.038862   | 16.93x  | -94.09%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.658061 | 0.034693   | 18.97x  | -94.73%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.697933 | 0.083420   | 8.37x   | -88.05%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.697933 | 0.040159   | 17.38x  | -94.25%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.697933 | 0.035440   | 19.69x  | -94.92%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.706168 | 0.092477   | 7.64x   | -86.90%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.706168 | 0.042978   | 16.43x  | -93.91%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.706168 | 0.035943   | 19.65x  | -94.91%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.676164 | 0.211634   | 3.19x   | -68.70%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.676164 | 0.041785   | 16.18x  | -93.82%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.676164 | 0.035716   | 18.93x  | -94.72%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.659188 | 0.102220   | 6.45x   | -84.49%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.659188 | 0.039047   | 16.88x  | -94.08%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.659188 | 0.034907   | 18.88x  | -94.70%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.671086 | 0.083323   | 8.05x   | -87.58%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.671086 | 0.039725   | 16.89x  | -94.08%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.671086 | 0.034554   | 19.42x  | -94.85%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.988336 | 0.239905   | 4.12x   | -75.73%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.988336 | 0.042721   | 23.13x  | -95.68%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.988336 | 0.035840   | 27.58x  | -96.37%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.898995 | 0.081058   | 11.09x  | -90.98%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.898995 | 0.039557   | 22.73x  | -95.60%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.898995 | 0.035058   | 25.64x  | -96.10%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.673456 | 0.098218   | 6.86x   | -85.42%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.673456 | 0.041589   | 16.19x  | -93.82%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.673456 | 0.035359   | 19.05x  | -94.75%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.686935 | 0.139938   | 4.91x   | -79.63%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.686935 | 0.040242   | 17.07x  | -94.14%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.686935 | 0.034880   | 19.69x  | -94.92%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.759165 | 0.143840   | 5.28x   | -81.05%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.759165 | 0.041533   | 18.28x  | -94.53%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.759165 | 0.034860   | 21.78x  | -95.41%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.732662 | 0.095421   | 7.68x   | -86.98%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.732662 | 0.043945   | 16.67x  | -94.00%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.732662 | 0.036656   | 19.99x  | -95.00%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.980262 | 0.231591   | 4.23x   | -76.37%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.980262 | 0.044159   | 22.20x  | -95.50%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.980262 | 0.036290   | 27.01x  | -96.30%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.544545 | 0.092030   | 5.92x   | -83.10%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.544545 | 0.035928   | 15.16x  | -93.40%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.544545 | 0.034666   | 15.71x  | -93.63%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.735928 | 0.102868   | 7.15x   | -86.02%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.735928 | 0.036621   | 20.10x  | -95.02%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.735928 | 0.035556   | 20.70x  | -95.17%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.473024 | 0.115852   | 4.08x   | -75.51%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.473024 | 0.034338   | 13.78x  | -92.74%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.473024 | 0.035419   | 13.36x  | -92.51%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.521490 | 0.096351   | 5.41x   | -81.52%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.521490 | 0.035581   | 14.66x  | -93.18%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.521490 | 0.035595   | 14.65x  | -93.17%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.547246 | 0.106702   | 5.13x   | -80.50%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.547246 | 0.040088   | 13.65x  | -92.67%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.547246 | 0.035345   | 15.48x  | -93.54%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.499812 | 0.183974   | 2.72x   | -63.19%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.499812 | 0.037324   | 13.39x  | -92.53%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.499812 | 0.037424   | 13.36x  | -92.51%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.496919 | 0.125710   | 3.95x   | -74.70%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.496919 | 0.039073   | 12.72x  | -92.14%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.496919 | 0.037640   | 13.20x  | -92.43%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.552370 | 0.099011   | 5.58x   | -82.08%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.552370 | 0.042600   | 12.97x  | -92.29%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.552370 | 0.040650   | 13.59x  | -92.64%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.918340 | 0.257481   | 3.57x   | -71.96%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.918340 | 0.048451   | 18.95x  | -94.72%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.918340 | 0.041456   | 22.15x  | -95.49%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.786324 | 0.098657   | 7.97x   | -87.45%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.786324 | 0.041787   | 18.82x  | -94.69%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.786324 | 0.040032   | 19.64x  | -94.91%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.568283 | 0.131494   | 4.32x   | -76.86%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.568283 | 0.051398   | 11.06x  | -90.96%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.568283 | 0.043194   | 13.16x  | -92.40%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.614774 | 0.156687   | 3.92x   | -74.51%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.614774 | 0.040818   | 15.06x  | -93.36%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.614774 | 0.045042   | 13.65x  | -92.67%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.606151 | 0.156730   | 3.87x   | -74.14%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.606151 | 0.044828   | 13.52x  | -92.60%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.606151 | 0.038659   | 15.68x  | -93.62%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.596025 | 0.113998   | 5.23x   | -80.87%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.596025 | 0.054188   | 11.00x  | -90.91%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.596025 | 0.048969   | 12.17x  | -91.78%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.829196 | 0.240897   | 3.44x   | -70.95%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.829196 | 0.037937   | 21.86x  | -95.42%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.829196 | 0.041287   | 20.08x  | -95.02%        |
</details>
<!-- PERF_RESULTS_END -->
