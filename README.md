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
| normal        | scalar | 0.202726 | 0.042567   | 4.76x   | -79.00%        |
| normal        | sse42  | 0.202726 | 0.014943   | 13.57x  | -92.63%        |
| normal        | avx2   | 0.202726 | 0.013875   | 14.61x  | -93.16%        |
| soft_light    | scalar | 0.274885 | 0.050004   | 5.50x   | -81.81%        |
| soft_light    | sse42  | 0.274885 | 0.016791   | 16.37x  | -93.89%        |
| soft_light    | avx2   | 0.274885 | 0.015433   | 17.81x  | -94.39%        |
| lighten_only  | scalar | 0.202197 | 0.054170   | 3.73x   | -73.21%        |
| lighten_only  | sse42  | 0.202197 | 0.015383   | 13.14x  | -92.39%        |
| lighten_only  | avx2   | 0.202197 | 0.014964   | 13.51x  | -92.60%        |
| screen        | scalar | 0.215419 | 0.048196   | 4.47x   | -77.63%        |
| screen        | sse42  | 0.215419 | 0.016018   | 13.45x  | -92.56%        |
| screen        | avx2   | 0.215419 | 0.015136   | 14.23x  | -92.97%        |
| dodge         | scalar | 0.216169 | 0.050629   | 4.27x   | -76.58%        |
| dodge         | sse42  | 0.216169 | 0.017625   | 12.26x  | -91.85%        |
| dodge         | avx2   | 0.216169 | 0.015544   | 13.91x  | -92.81%        |
| addition      | scalar | 0.208097 | 0.075851   | 2.74x   | -63.55%        |
| addition      | sse42  | 0.208097 | 0.016657   | 12.49x  | -92.00%        |
| addition      | avx2   | 0.208097 | 0.015235   | 13.66x  | -92.68%        |
| darken_only   | scalar | 0.203791 | 0.054533   | 3.74x   | -73.24%        |
| darken_only   | sse42  | 0.203791 | 0.015550   | 13.11x  | -92.37%        |
| darken_only   | avx2   | 0.203791 | 0.014858   | 13.72x  | -92.71%        |
| multiply      | scalar | 0.209584 | 0.047338   | 4.43x   | -77.41%        |
| multiply      | sse42  | 0.209584 | 0.015699   | 13.35x  | -92.51%        |
| multiply      | avx2   | 0.209584 | 0.014978   | 13.99x  | -92.85%        |
| hard_light    | scalar | 0.308733 | 0.093807   | 3.29x   | -69.62%        |
| hard_light    | sse42  | 0.308733 | 0.017876   | 17.27x  | -94.21%        |
| hard_light    | avx2   | 0.308733 | 0.015630   | 19.75x  | -94.94%        |
| difference    | scalar | 0.279202 | 0.047006   | 5.94x   | -83.16%        |
| difference    | sse42  | 0.279202 | 0.015837   | 17.63x  | -94.33%        |
| difference    | avx2   | 0.279202 | 0.015096   | 18.50x  | -94.59%        |
| subtract      | scalar | 0.208186 | 0.049912   | 4.17x   | -76.03%        |
| subtract      | sse42  | 0.208186 | 0.017391   | 11.97x  | -91.65%        |
| subtract      | avx2   | 0.208186 | 0.015590   | 13.35x  | -92.51%        |
| grain_extract | scalar | 0.214372 | 0.062674   | 3.42x   | -70.76%        |
| grain_extract | sse42  | 0.214372 | 0.016577   | 12.93x  | -92.27%        |
| grain_extract | avx2   | 0.214372 | 0.015269   | 14.04x  | -92.88%        |
| grain_merge   | scalar | 0.212850 | 0.062570   | 3.40x   | -70.60%        |
| grain_merge   | sse42  | 0.212850 | 0.016579   | 12.84x  | -92.21%        |
| grain_merge   | avx2   | 0.212850 | 0.015224   | 13.98x  | -92.85%        |
| divide        | scalar | 0.218603 | 0.048823   | 4.48x   | -77.67%        |
| divide        | sse42  | 0.218603 | 0.016882   | 12.95x  | -92.28%        |
| divide        | avx2   | 0.218603 | 0.015434   | 14.16x  | -92.94%        |
| overlay       | scalar | 0.285608 | 0.090168   | 3.17x   | -68.43%        |
| overlay       | sse42  | 0.285608 | 0.017180   | 16.62x  | -93.98%        |
| overlay       | avx2   | 0.285608 | 0.015447   | 18.49x  | -94.59%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.006718 | 0.001936   | 3.47x   | -71.18%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.006718 | 0.000773   | 8.69x   | -88.49%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.006718 | 0.000940   | 7.15x   | -86.01%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.008764 | 0.001910   | 4.59x   | -78.20%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.008764 | 0.000857   | 10.22x  | -90.22%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.008764 | 0.000833   | 10.52x  | -90.50%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.007256 | 0.001918   | 3.78x   | -73.56%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.007256 | 0.000794   | 9.14x   | -89.06%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.007256 | 0.000763   | 9.51x   | -89.48%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.007571 | 0.001741   | 4.35x   | -77.01%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.007571 | 0.000797   | 9.50x   | -89.47%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.007571 | 0.000878   | 8.62x   | -88.40%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.007887 | 0.001801   | 4.38x   | -77.16%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.007887 | 0.000899   | 8.77x   | -88.60%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.007887 | 0.000806   | 9.79x   | -89.78%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.007218 | 0.002582   | 2.80x   | -64.22%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.007218 | 0.000868   | 8.32x   | -87.98%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.007218 | 0.000758   | 9.53x   | -89.50%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.007183 | 0.001954   | 3.68x   | -72.79%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.007183 | 0.000784   | 9.16x   | -89.09%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.007183 | 0.000753   | 9.54x   | -89.52%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.007140 | 0.001951   | 3.66x   | -72.68%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.007140 | 0.000781   | 9.15x   | -89.07%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.007140 | 0.000998   | 7.15x   | -86.02%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.009429 | 0.003014   | 3.13x   | -68.04%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.009429 | 0.000883   | 10.68x  | -90.64%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.009429 | 0.000807   | 11.69x  | -91.45%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.009254 | 0.001720   | 5.38x   | -81.42%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.009254 | 0.000771   | 12.00x  | -91.66%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.009254 | 0.000751   | 12.32x  | -91.89%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.007381 | 0.001754   | 4.21x   | -76.24%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.007381 | 0.000855   | 8.63x   | -88.41%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.007381 | 0.001006   | 7.34x   | -86.37%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.007489 | 0.002096   | 3.57x   | -72.02%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.007489 | 0.000878   | 8.53x   | -88.28%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.007489 | 0.000785   | 9.54x   | -89.51%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.007326 | 0.002291   | 3.20x   | -68.73%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.007326 | 0.000853   | 8.58x   | -88.35%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.007326 | 0.000799   | 9.17x   | -89.09%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.007757 | 0.001835   | 4.23x   | -76.34%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.007757 | 0.000863   | 8.98x   | -88.87%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.007757 | 0.000801   | 9.69x   | -89.67%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.009103 | 0.002923   | 3.11x   | -67.89%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.009103 | 0.000893   | 10.19x  | -90.19%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.009103 | 0.000801   | 11.37x  | -91.20%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003422 | 0.001328   | 2.58x   | -61.19%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003422 | 0.000177   | 19.39x  | -94.84%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003422 | 0.000162   | 21.17x  | -95.28%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.007311 | 0.001843   | 3.97x   | -74.80%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.007311 | 0.000223   | 32.77x  | -96.95%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.007311 | 0.000198   | 36.87x  | -97.29%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.005357 | 0.001744   | 3.07x   | -67.44%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.005357 | 0.000193   | 27.78x  | -96.40%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.005357 | 0.000189   | 28.35x  | -96.47%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.005830 | 0.001794   | 3.25x   | -69.22%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.005830 | 0.000218   | 26.77x  | -96.26%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.005830 | 0.000210   | 27.71x  | -96.39%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.005786 | 0.001837   | 3.15x   | -68.26%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.005786 | 0.000242   | 23.89x  | -95.81%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.005786 | 0.000198   | 29.26x  | -96.58%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.005972 | 0.001947   | 3.07x   | -67.39%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.005972 | 0.000262   | 22.84x  | -95.62%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.005972 | 0.000199   | 29.99x  | -96.67%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.005680 | 0.001715   | 3.31x   | -69.80%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.005680 | 0.000192   | 29.66x  | -96.63%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.005680 | 0.000219   | 25.99x  | -96.15%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.005674 | 0.001585   | 3.58x   | -72.08%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.005674 | 0.000201   | 28.16x  | -96.45%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.005674 | 0.000191   | 29.66x  | -96.63%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.008013 | 0.002581   | 3.10x   | -67.79%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.008013 | 0.000240   | 33.40x  | -97.01%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.008013 | 0.000200   | 40.02x  | -97.50%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.007545 | 0.001620   | 4.66x   | -78.53%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.007545 | 0.000198   | 38.04x  | -97.37%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.007545 | 0.000190   | 39.64x  | -97.48%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.006279 | 0.001530   | 4.10x   | -75.63%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.006279 | 0.000268   | 23.40x  | -95.73%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.006279 | 0.000209   | 30.09x  | -96.68%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.006144 | 0.001872   | 3.28x   | -69.53%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.006144 | 0.000213   | 28.89x  | -96.54%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.006144 | 0.000198   | 31.09x  | -96.78%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.006263 | 0.001936   | 3.23x   | -69.09%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.006263 | 0.000215   | 29.13x  | -96.57%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.006263 | 0.000200   | 31.25x  | -96.80%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.006205 | 0.001559   | 3.98x   | -74.87%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.006205 | 0.000220   | 28.25x  | -96.46%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.006205 | 0.000194   | 32.04x  | -96.88%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.007882 | 0.002547   | 3.09x   | -67.69%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.007882 | 0.000226   | 34.83x  | -97.13%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.007882 | 0.000222   | 35.48x  | -97.18%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.006137 | 0.000511   | 12.02x  | -91.68%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.006137 | 0.000261   | 23.49x  | -95.74%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.006137 | 0.000151   | 40.54x  | -97.53%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.010353 | 0.000707   | 14.65x  | -93.17%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.010353 | 0.000232   | 44.58x  | -97.76%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.010353 | 0.000187   | 55.34x  | -98.19%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.007765 | 0.000758   | 10.24x  | -90.23%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.007765 | 0.000213   | 36.41x  | -97.25%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.007765 | 0.000182   | 42.69x  | -97.66%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.007798 | 0.000583   | 13.37x  | -92.52%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.007798 | 0.000227   | 34.36x  | -97.09%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.007798 | 0.000184   | 42.35x  | -97.64%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.007783 | 0.000646   | 12.04x  | -91.70%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.007783 | 0.000252   | 30.85x  | -96.76%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.007783 | 0.000192   | 40.58x  | -97.54%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.008173 | 0.001606   | 5.09x   | -80.35%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.008173 | 0.000241   | 33.92x  | -97.05%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.008173 | 0.000187   | 43.68x  | -97.71%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.007556 | 0.000718   | 10.52x  | -90.50%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.007556 | 0.000211   | 35.84x  | -97.21%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.007556 | 0.000178   | 42.51x  | -97.65%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.007684 | 0.000601   | 12.79x  | -92.18%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.007684 | 0.000213   | 36.09x  | -97.23%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.007684 | 0.000179   | 42.83x  | -97.67%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.009577 | 0.001814   | 5.28x   | -81.06%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.009577 | 0.000242   | 39.54x  | -97.47%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.009577 | 0.000190   | 50.48x  | -98.02%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.009572 | 0.000546   | 17.53x  | -94.30%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.009572 | 0.000216   | 44.40x  | -97.75%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.009572 | 0.000182   | 52.67x  | -98.10%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.007438 | 0.000697   | 10.67x  | -90.63%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.007438 | 0.000239   | 31.17x  | -96.79%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.007438 | 0.000188   | 39.57x  | -97.47%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.007515 | 0.001147   | 6.55x   | -84.74%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.007515 | 0.000236   | 31.86x  | -96.86%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.007515 | 0.000185   | 40.61x  | -97.54%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.007460 | 0.001103   | 6.76x   | -85.22%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.007460 | 0.000230   | 32.50x  | -96.92%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.007460 | 0.000190   | 39.28x  | -97.45%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.007681 | 0.000626   | 12.27x  | -91.85%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.007681 | 0.000335   | 22.93x  | -95.64%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.007681 | 0.000188   | 40.89x  | -97.55%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.008771 | 0.001668   | 5.26x   | -80.99%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.008771 | 0.000232   | 37.82x  | -97.36%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.008771 | 0.000179   | 48.88x  | -97.95%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.004513 | 0.000617   | 7.31x   | -86.32%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.004513 | 0.000158   | 28.61x  | -96.50%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.004513 | 0.000154   | 29.35x  | -96.59%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.007085 | 0.000722   | 9.81x   | -89.81%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.007085 | 0.000189   | 37.57x  | -97.34%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.007085 | 0.000184   | 38.51x  | -97.40%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.005385 | 0.000793   | 6.79x   | -85.28%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.005385 | 0.000171   | 31.42x  | -96.82%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.005385 | 0.000182   | 29.66x  | -96.63%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.005814 | 0.000674   | 8.63x   | -88.41%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.005814 | 0.000239   | 24.33x  | -95.89%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.005814 | 0.000177   | 32.81x  | -96.95%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.005733 | 0.000797   | 7.19x   | -86.10%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.005733 | 0.000233   | 24.57x  | -95.93%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.005733 | 0.000182   | 31.50x  | -96.83%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.006429 | 0.001385   | 4.64x   | -78.46%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.006429 | 0.000206   | 31.23x  | -96.80%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.006429 | 0.000184   | 34.94x  | -97.14%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.005875 | 0.000780   | 7.53x   | -86.72%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.005875 | 0.000171   | 34.34x  | -97.09%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.005875 | 0.000185   | 31.73x  | -96.85%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.005799 | 0.000645   | 8.99x   | -88.88%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.005799 | 0.000186   | 31.18x  | -96.79%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.005799 | 0.000187   | 31.01x  | -96.78%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.007711 | 0.001891   | 4.08x   | -75.47%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.007711 | 0.000251   | 30.70x  | -96.74%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.007711 | 0.000184   | 41.90x  | -97.61%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.007829 | 0.000678   | 11.54x  | -91.34%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.007829 | 0.000189   | 41.52x  | -97.59%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.007829 | 0.000188   | 41.62x  | -97.60%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.005769 | 0.000872   | 6.62x   | -84.89%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.005769 | 0.000198   | 29.10x  | -96.56%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.005769 | 0.000189   | 30.49x  | -96.72%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.005695 | 0.001061   | 5.37x   | -81.37%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.005695 | 0.000180   | 31.61x  | -96.84%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.005695 | 0.000177   | 32.09x  | -96.88%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.006015 | 0.001059   | 5.68x   | -82.39%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.006015 | 0.000181   | 33.21x  | -96.99%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.006015 | 0.000189   | 31.76x  | -96.85%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.006028 | 0.000718   | 8.39x   | -88.09%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.006028 | 0.000185   | 32.55x  | -96.93%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.006028 | 0.000212   | 28.42x  | -96.48%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.007865 | 0.001793   | 4.39x   | -77.20%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.007865 | 0.000211   | 37.23x  | -97.31%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.007865 | 0.000199   | 39.58x  | -97.47%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.034391 | 0.006580   | 5.23x   | -80.87%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.034391 | 0.003511   | 9.80x   | -89.79%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.034391 | 0.003504   | 9.81x   | -89.81%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.045265 | 0.007521   | 6.02x   | -83.38%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.045265 | 0.003502   | 12.93x  | -92.26%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.045265 | 0.003336   | 13.57x  | -92.63%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.038058 | 0.007779   | 4.89x   | -79.56%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.038058 | 0.003426   | 11.11x  | -91.00%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.038058 | 0.003010   | 12.65x  | -92.09%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.040513 | 0.007163   | 5.66x   | -82.32%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.040513 | 0.003253   | 12.45x  | -91.97%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.040513 | 0.003262   | 12.42x  | -91.95%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.039715 | 0.007374   | 5.39x   | -81.43%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.039715 | 0.003834   | 10.36x  | -90.35%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.039715 | 0.004028   | 9.86x   | -89.86%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.038981 | 0.010321   | 3.78x   | -73.52%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.038981 | 0.003333   | 11.70x  | -91.45%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.038981 | 0.003216   | 12.12x  | -91.75%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.039278 | 0.007747   | 5.07x   | -80.28%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.039278 | 0.003271   | 12.01x  | -91.67%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.039278 | 0.003272   | 12.00x  | -91.67%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.037792 | 0.007800   | 4.84x   | -79.36%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.037792 | 0.003115   | 12.13x  | -91.76%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.037792 | 0.003111   | 12.15x  | -91.77%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.047623 | 0.012499   | 3.81x   | -73.75%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.047623 | 0.003719   | 12.81x  | -92.19%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.047623 | 0.003366   | 14.15x  | -92.93%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.044658 | 0.007111   | 6.28x   | -84.08%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.044658 | 0.003095   | 14.43x  | -93.07%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.044658 | 0.003183   | 14.03x  | -92.87%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.037979 | 0.006721   | 5.65x   | -82.30%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.037979 | 0.003823   | 9.93x   | -89.93%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.037979 | 0.003640   | 10.43x  | -90.42%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.038032 | 0.008785   | 4.33x   | -76.90%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.038032 | 0.003414   | 11.14x  | -91.02%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.038032 | 0.003386   | 11.23x  | -91.10%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.038573 | 0.008560   | 4.51x   | -77.81%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.038573 | 0.003444   | 11.20x  | -91.07%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.038573 | 0.003160   | 12.21x  | -91.81%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.038854 | 0.008175   | 4.75x   | -78.96%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.038854 | 0.003484   | 11.15x  | -91.03%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.038854 | 0.003366   | 11.54x  | -91.34%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.046152 | 0.011830   | 3.90x   | -74.37%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.046152 | 0.003457   | 13.35x  | -92.51%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.046152 | 0.003459   | 13.34x  | -92.51%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.025209 | 0.005464   | 4.61x   | -78.33%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.025209 | 0.000883   | 28.54x  | -96.50%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.025209 | 0.000645   | 39.06x  | -97.44%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.036450 | 0.006990   | 5.21x   | -80.82%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.036450 | 0.001143   | 31.89x  | -96.86%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.036450 | 0.000987   | 36.92x  | -97.29%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.030644 | 0.006873   | 4.46x   | -77.57%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.030644 | 0.000777   | 39.44x  | -97.46%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.030644 | 0.000750   | 40.87x  | -97.55%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.030332 | 0.006557   | 4.63x   | -78.38%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.030332 | 0.000846   | 35.87x  | -97.21%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.030332 | 0.000784   | 38.68x  | -97.41%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.030787 | 0.006714   | 4.59x   | -78.19%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.030787 | 0.001097   | 28.06x  | -96.44%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.030787 | 0.000915   | 33.65x  | -97.03%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.030163 | 0.007858   | 3.84x   | -73.95%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.030163 | 0.001085   | 27.81x  | -96.40%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.030163 | 0.000804   | 37.53x  | -97.34%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.028884 | 0.007274   | 3.97x   | -74.82%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.028884 | 0.000773   | 37.35x  | -97.32%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.028884 | 0.000752   | 38.39x  | -97.40%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.030453 | 0.006817   | 4.47x   | -77.61%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.030453 | 0.000808   | 37.70x  | -97.35%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.030453 | 0.000755   | 40.34x  | -97.52%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.039048 | 0.010457   | 3.73x   | -73.22%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.039048 | 0.000939   | 41.57x  | -97.59%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.039048 | 0.000788   | 49.58x  | -97.98%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.038996 | 0.006281   | 6.21x   | -83.89%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.038996 | 0.000787   | 49.58x  | -97.98%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.038996 | 0.000752   | 51.87x  | -98.07%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.030472 | 0.006348   | 4.80x   | -79.17%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.030472 | 0.001080   | 28.22x  | -96.46%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.030472 | 0.000857   | 35.54x  | -97.19%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.031210 | 0.008116   | 3.85x   | -74.00%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.031210 | 0.000868   | 35.95x  | -97.22%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.031210 | 0.000791   | 39.44x  | -97.46%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.030916 | 0.007950   | 3.89x   | -74.28%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.030916 | 0.000855   | 36.16x  | -97.23%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.030916 | 0.000870   | 35.54x  | -97.19%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.031954 | 0.006553   | 4.88x   | -79.49%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.031954 | 0.000894   | 35.74x  | -97.20%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.031954 | 0.000785   | 40.69x  | -97.54%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.037123 | 0.010252   | 3.62x   | -72.38%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.037123 | 0.000924   | 40.18x  | -97.51%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.037123 | 0.000811   | 45.80x  | -97.82%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.029571 | 0.002472   | 11.96x  | -91.64%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.029571 | 0.001207   | 24.49x  | -95.92%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.029571 | 0.000674   | 43.85x  | -97.72%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.041303 | 0.002985   | 13.83x  | -92.77%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.041303 | 0.000989   | 41.77x  | -97.61%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.041303 | 0.000825   | 50.05x  | -98.00%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.034333 | 0.003567   | 9.62x   | -89.61%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.034333 | 0.000884   | 38.82x  | -97.42%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.034333 | 0.000813   | 42.25x  | -97.63%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.035328 | 0.002753   | 12.83x  | -92.21%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.035328 | 0.000926   | 38.15x  | -97.38%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.035328 | 0.000763   | 46.27x  | -97.84%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.034295 | 0.003099   | 11.07x  | -90.96%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.034295 | 0.001071   | 32.03x  | -96.88%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.034295 | 0.001128   | 30.42x  | -96.71%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.034354 | 0.006835   | 5.03x   | -80.10%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.034354 | 0.001007   | 34.12x  | -97.07%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.034354 | 0.000823   | 41.72x  | -97.60%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.035661 | 0.003335   | 10.69x  | -90.65%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.035661 | 0.000875   | 40.77x  | -97.55%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.035661 | 0.000761   | 46.87x  | -97.87%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.035253 | 0.002877   | 12.25x  | -91.84%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.035253 | 0.000944   | 37.34x  | -97.32%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.035253 | 0.000814   | 43.32x  | -97.69%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.044136 | 0.007702   | 5.73x   | -82.55%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.044136 | 0.001282   | 34.43x  | -97.10%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.044136 | 0.000807   | 54.69x  | -98.17%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.042358 | 0.002789   | 15.19x  | -93.42%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.042358 | 0.001089   | 38.91x  | -97.43%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.042358 | 0.000814   | 52.02x  | -98.08%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.034865 | 0.003324   | 10.49x  | -90.47%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.034865 | 0.001048   | 33.26x  | -96.99%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.034865 | 0.000799   | 43.66x  | -97.71%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.035427 | 0.004560   | 7.77x   | -87.13%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.035427 | 0.000978   | 36.21x  | -97.24%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.035427 | 0.000785   | 45.11x  | -97.78%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.035561 | 0.004715   | 7.54x   | -86.74%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.035561 | 0.000956   | 37.18x  | -97.31%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.035561 | 0.000793   | 44.84x  | -97.77%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.036380 | 0.002940   | 12.38x  | -91.92%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.036380 | 0.001041   | 34.94x  | -97.14%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.036380 | 0.000843   | 43.14x  | -97.68%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.042859 | 0.007296   | 5.87x   | -82.98%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.042859 | 0.001091   | 39.27x  | -97.45%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.042859 | 0.000869   | 49.32x  | -97.97%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.021370 | 0.002584   | 8.27x   | -87.91%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.021370 | 0.000777   | 27.51x  | -96.36%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.021370 | 0.000922   | 23.18x  | -95.69%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.033408 | 0.003151   | 10.60x  | -90.57%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.033408 | 0.000847   | 39.42x  | -97.46%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.033408 | 0.000847   | 39.46x  | -97.47%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.025600 | 0.003219   | 7.95x   | -87.43%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.025600 | 0.000786   | 32.57x  | -96.93%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.025600 | 0.000832   | 30.78x  | -96.75%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.026008 | 0.002852   | 9.12x   | -89.03%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.026008 | 0.000800   | 32.50x  | -96.92%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.026008 | 0.000791   | 32.87x  | -96.96%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.026902 | 0.003273   | 8.22x   | -87.84%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.026902 | 0.000997   | 26.97x  | -96.29%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.026902 | 0.000851   | 31.60x  | -96.84%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.025574 | 0.005679   | 4.50x   | -77.79%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.025574 | 0.000869   | 29.44x  | -96.60%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.025574 | 0.000843   | 30.34x  | -96.70%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.026084 | 0.003275   | 7.96x   | -87.44%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.026084 | 0.000777   | 33.58x  | -97.02%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.026084 | 0.000823   | 31.71x  | -96.85%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.025905 | 0.002818   | 9.19x   | -89.12%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.025905 | 0.000858   | 30.20x  | -96.69%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.025905 | 0.000849   | 30.52x  | -96.72%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.035194 | 0.007544   | 4.67x   | -78.56%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.035194 | 0.000961   | 36.60x  | -97.27%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.035194 | 0.000873   | 40.29x  | -97.52%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.032975 | 0.002747   | 12.00x  | -91.67%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.032975 | 0.000995   | 33.14x  | -96.98%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.032975 | 0.000870   | 37.88x  | -97.36%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.025737 | 0.003633   | 7.08x   | -85.88%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.025737 | 0.000900   | 28.59x  | -96.50%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.025737 | 0.000802   | 32.09x  | -96.88%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.026076 | 0.004427   | 5.89x   | -83.02%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.026076 | 0.000789   | 33.04x  | -96.97%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.026076 | 0.000778   | 33.51x  | -97.02%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.026588 | 0.004616   | 5.76x   | -82.64%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.026588 | 0.000789   | 33.72x  | -97.03%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.026588 | 0.000780   | 34.08x  | -97.07%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.027114 | 0.003063   | 8.85x   | -88.70%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.027114 | 0.000855   | 31.70x  | -96.85%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.027114 | 0.000965   | 28.10x  | -96.44%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.034025 | 0.007196   | 4.73x   | -78.85%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.034025 | 0.000982   | 34.66x  | -97.11%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.034025 | 0.000832   | 40.89x  | -97.55%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.106099 | 0.027792   | 3.82x   | -73.81%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.106099 | 0.013261   | 8.00x   | -87.50%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.106099 | 0.012953   | 8.19x   | -87.79%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.141395 | 0.030473   | 4.64x   | -78.45%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.141395 | 0.014738   | 9.59x   | -89.58%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.141395 | 0.013905   | 10.17x  | -90.17%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.112129 | 0.032660   | 3.43x   | -70.87%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.112129 | 0.013300   | 8.43x   | -88.14%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.112129 | 0.012506   | 8.97x   | -88.85%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.115376 | 0.028912   | 3.99x   | -74.94%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.115376 | 0.013780   | 8.37x   | -88.06%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.115376 | 0.012882   | 8.96x   | -88.84%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.114321 | 0.030441   | 3.76x   | -73.37%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.114321 | 0.014931   | 7.66x   | -86.94%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.114321 | 0.013924   | 8.21x   | -87.82%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.112758 | 0.042438   | 2.66x   | -62.36%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.112758 | 0.013718   | 8.22x   | -87.83%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.112758 | 0.012647   | 8.92x   | -88.78%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.111466 | 0.034965   | 3.19x   | -68.63%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.111466 | 0.013378   | 8.33x   | -88.00%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.111466 | 0.012622   | 8.83x   | -88.68%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.111393 | 0.029232   | 3.81x   | -73.76%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.111393 | 0.013343   | 8.35x   | -88.02%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.111393 | 0.012769   | 8.72x   | -88.54%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.148911 | 0.049348   | 3.02x   | -66.86%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.148911 | 0.014673   | 10.15x  | -90.15%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.148911 | 0.013726   | 10.85x  | -90.78%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.140701 | 0.027831   | 5.06x   | -80.22%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.140701 | 0.012792   | 11.00x  | -90.91%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.140701 | 0.012574   | 11.19x  | -91.06%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.108621 | 0.027553   | 3.94x   | -74.63%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.108621 | 0.014024   | 7.75x   | -87.09%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.108621 | 0.012935   | 8.40x   | -88.09%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.112395 | 0.034623   | 3.25x   | -69.20%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.112395 | 0.014522   | 7.74x   | -87.08%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.112395 | 0.012806   | 8.78x   | -88.61%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.111400 | 0.034343   | 3.24x   | -69.17%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.111400 | 0.014130   | 7.88x   | -87.32%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.111400 | 0.012947   | 8.60x   | -88.38%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.113729 | 0.028449   | 4.00x   | -74.99%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.113729 | 0.014011   | 8.12x   | -87.68%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.113729 | 0.013085   | 8.69x   | -88.49%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.137729 | 0.046919   | 2.94x   | -65.93%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.137729 | 0.014162   | 9.73x   | -89.72%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.137729 | 0.013264   | 10.38x  | -90.37%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.073412 | 0.021277   | 3.45x   | -71.02%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.073412 | 0.002976   | 24.67x  | -95.95%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.073412 | 0.002591   | 28.34x  | -96.47%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.108315 | 0.026673   | 4.06x   | -75.37%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.108315 | 0.003690   | 29.35x  | -96.59%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.108315 | 0.003182   | 34.04x  | -97.06%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.081872 | 0.028706   | 2.85x   | -64.94%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.081872 | 0.003300   | 24.81x  | -95.97%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.081872 | 0.003145   | 26.04x  | -96.16%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.086879 | 0.026813   | 3.24x   | -69.14%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.086879 | 0.003410   | 25.48x  | -96.08%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.086879 | 0.003163   | 27.47x  | -96.36%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.086297 | 0.027455   | 3.14x   | -68.19%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.086297 | 0.003958   | 21.80x  | -95.41%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.086297 | 0.003410   | 25.31x  | -96.05%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.086084 | 0.032417   | 2.66x   | -62.34%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.086084 | 0.004163   | 20.68x  | -95.16%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.086084 | 0.003221   | 26.73x  | -96.26%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.081194 | 0.028676   | 2.83x   | -64.68%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.081194 | 0.003435   | 23.64x  | -95.77%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.081194 | 0.003054   | 26.59x  | -96.24%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.085238 | 0.025963   | 3.28x   | -69.54%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.085238 | 0.003179   | 26.81x  | -96.27%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.085238 | 0.003205   | 26.59x  | -96.24%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.119660 | 0.043907   | 2.73x   | -63.31%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.119660 | 0.004047   | 29.57x  | -96.62%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.119660 | 0.003207   | 37.31x  | -97.32%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.113305 | 0.025355   | 4.47x   | -77.62%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.113305 | 0.003107   | 36.46x  | -97.26%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.113305 | 0.002989   | 37.91x  | -97.36%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.082142 | 0.025529   | 3.22x   | -68.92%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.082142 | 0.004373   | 18.78x  | -94.68%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.082142 | 0.003484   | 23.58x  | -95.76%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.085456 | 0.031169   | 2.74x   | -63.53%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.085456 | 0.003475   | 24.59x  | -95.93%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.085456 | 0.003173   | 26.93x  | -96.29%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.083664 | 0.031166   | 2.68x   | -62.75%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.083664 | 0.003538   | 23.65x  | -95.77%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.083664 | 0.003303   | 25.33x  | -96.05%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.087409 | 0.026503   | 3.30x   | -69.68%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.087409 | 0.003489   | 25.06x  | -96.01%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.087409 | 0.003289   | 26.58x  | -96.24%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.121034 | 0.040883   | 2.96x   | -66.22%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.121034 | 0.003747   | 32.30x  | -96.90%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.121034 | 0.003234   | 37.43x  | -97.33%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.088487 | 0.008599   | 10.29x  | -90.28%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.088487 | 0.003657   | 24.19x  | -95.87%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.088487 | 0.002660   | 33.27x  | -96.99%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.122766 | 0.010767   | 11.40x  | -91.23%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.122766 | 0.003996   | 30.72x  | -96.75%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.122766 | 0.003222   | 38.11x  | -97.38%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.096139 | 0.012261   | 7.84x   | -87.25%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.096139 | 0.003727   | 25.80x  | -96.12%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.096139 | 0.003248   | 29.60x  | -96.62%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.103943 | 0.010063   | 10.33x  | -90.32%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.103943 | 0.003875   | 26.83x  | -96.27%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.103943 | 0.003049   | 34.09x  | -97.07%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.098404 | 0.010925   | 9.01x   | -88.90%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.098404 | 0.004211   | 23.37x  | -95.72%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.098404 | 0.003302   | 29.80x  | -96.64%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.095842 | 0.026164   | 3.66x   | -72.70%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.095842 | 0.004038   | 23.73x  | -95.79%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.095842 | 0.003389   | 28.28x  | -96.46%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.095456 | 0.012219   | 7.81x   | -87.20%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.095456 | 0.003781   | 25.25x  | -96.04%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.095456 | 0.003114   | 30.66x  | -96.74%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.096505 | 0.009591   | 10.06x  | -90.06%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.096505 | 0.003677   | 26.24x  | -96.19%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.096505 | 0.003175   | 30.40x  | -96.71%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.132353 | 0.029231   | 4.53x   | -77.91%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.132353 | 0.004098   | 32.30x  | -96.90%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.132353 | 0.003295   | 40.17x  | -97.51%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.126876 | 0.009688   | 13.10x  | -92.36%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.126876 | 0.004155   | 30.54x  | -96.73%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.126876 | 0.003018   | 42.04x  | -97.62%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.096889 | 0.012198   | 7.94x   | -87.41%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.096889 | 0.004077   | 23.76x  | -95.79%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.096889 | 0.003421   | 28.32x  | -96.47%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.098212 | 0.016796   | 5.85x   | -82.90%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.098212 | 0.003754   | 26.16x  | -96.18%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.098212 | 0.003114   | 31.54x  | -96.83%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.098789 | 0.016963   | 5.82x   | -82.83%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.098789 | 0.004045   | 24.42x  | -95.91%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.098789 | 0.003205   | 30.83x  | -96.76%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.100743 | 0.010677   | 9.44x   | -89.40%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.100743 | 0.003981   | 25.31x  | -96.05%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.100743 | 0.003172   | 31.76x  | -96.85%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.127022 | 0.027414   | 4.63x   | -78.42%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.127022 | 0.004089   | 31.06x  | -96.78%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.127022 | 0.003473   | 36.57x  | -97.27%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.067734 | 0.010259   | 6.60x   | -84.85%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.067734 | 0.003069   | 22.07x  | -95.47%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.067734 | 0.002985   | 22.69x  | -95.59%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.105693 | 0.011912   | 8.87x   | -88.73%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.105693 | 0.003757   | 28.13x  | -96.45%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.105693 | 0.004039   | 26.17x  | -96.18%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.075395 | 0.012679   | 5.95x   | -83.18%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.075395 | 0.003666   | 20.57x  | -95.14%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.075395 | 0.003842   | 19.62x  | -94.90%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.080440 | 0.011406   | 7.05x   | -85.82%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.080440 | 0.003449   | 23.33x  | -95.71%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.080440 | 0.003910   | 20.57x  | -95.14%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.079228 | 0.012660   | 6.26x   | -84.02%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.079228 | 0.004536   | 17.47x  | -94.28%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.079228 | 0.003695   | 21.44x  | -95.34%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.075535 | 0.022173   | 3.41x   | -70.65%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.075535 | 0.003635   | 20.78x  | -95.19%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.075535 | 0.003991   | 18.93x  | -94.72%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.076276 | 0.012774   | 5.97x   | -83.25%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.076276 | 0.003437   | 22.19x  | -95.49%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.076276 | 0.004015   | 19.00x  | -94.74%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.075924 | 0.010459   | 7.26x   | -86.22%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.075924 | 0.003603   | 21.07x  | -95.25%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.075924 | 0.003619   | 20.98x  | -95.23%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.111572 | 0.029816   | 3.74x   | -73.28%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.111572 | 0.003953   | 28.23x  | -96.46%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.111572 | 0.003880   | 28.76x  | -96.52%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.107455 | 0.010587   | 10.15x  | -90.15%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.107455 | 0.003660   | 29.36x  | -96.59%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.107455 | 0.003768   | 28.52x  | -96.49%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.073769 | 0.014404   | 5.12x   | -80.47%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.073769 | 0.003720   | 19.83x  | -94.96%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.073769 | 0.004088   | 18.05x  | -94.46%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.079166 | 0.017323   | 4.57x   | -78.12%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.079166 | 0.003725   | 21.25x  | -95.29%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.079166 | 0.003959   | 20.00x  | -95.00%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.079983 | 0.017761   | 4.50x   | -77.79%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.079983 | 0.003784   | 21.14x  | -95.27%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.079983 | 0.004669   | 17.13x  | -94.16%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.086855 | 0.012136   | 7.16x   | -86.03%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.086855 | 0.003856   | 22.52x  | -95.56%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.086855 | 0.004519   | 19.22x  | -94.80%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.107394 | 0.028337   | 3.79x   | -73.61%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.107394 | 0.003846   | 27.92x  | -96.42%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.107394 | 0.003871   | 27.75x  | -96.40%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.433135 | 0.114061   | 3.80x   | -73.67%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.433135 | 0.050715   | 8.54x   | -88.29%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.433135 | 0.048948   | 8.85x   | -88.70%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.509726 | 0.115954   | 4.40x   | -77.25%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.509726 | 0.057120   | 8.92x   | -88.79%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.509726 | 0.051984   | 9.81x   | -89.80%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.396257 | 0.125931   | 3.15x   | -68.22%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.396257 | 0.051029   | 7.77x   | -87.12%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.396257 | 0.054727   | 7.24x   | -86.19%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.427275 | 0.116455   | 3.67x   | -72.74%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.427275 | 0.052309   | 8.17x   | -87.76%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.427275 | 0.051003   | 8.38x   | -88.06%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.418703 | 0.116938   | 3.58x   | -72.07%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.418703 | 0.058475   | 7.16x   | -86.03%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.418703 | 0.052670   | 7.95x   | -87.42%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.409296 | 0.163531   | 2.50x   | -60.05%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.409296 | 0.052129   | 7.85x   | -87.26%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.409296 | 0.049596   | 8.25x   | -87.88%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.395132 | 0.126457   | 3.12x   | -68.00%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.395132 | 0.052394   | 7.54x   | -86.74%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.395132 | 0.050059   | 7.89x   | -87.33%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.409465 | 0.115137   | 3.56x   | -71.88%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.409465 | 0.051163   | 8.00x   | -87.50%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.409465 | 0.050010   | 8.19x   | -87.79%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.576011 | 0.194667   | 2.96x   | -66.20%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.576011 | 0.058309   | 9.88x   | -89.88%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.576011 | 0.053074   | 10.85x  | -90.79%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.535315 | 0.117391   | 4.56x   | -78.07%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.535315 | 0.052954   | 10.11x  | -90.11%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.535315 | 0.049578   | 10.80x  | -90.74%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.411479 | 0.109832   | 3.75x   | -73.31%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.411479 | 0.056297   | 7.31x   | -86.32%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.411479 | 0.052453   | 7.84x   | -87.25%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.414977 | 0.136527   | 3.04x   | -67.10%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.414977 | 0.055884   | 7.43x   | -86.53%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.414977 | 0.051620   | 8.04x   | -87.56%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.413871 | 0.137916   | 3.00x   | -66.68%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.413871 | 0.055088   | 7.51x   | -86.69%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.413871 | 0.051975   | 7.96x   | -87.44%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.415518 | 0.115253   | 3.61x   | -72.26%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.415518 | 0.056703   | 7.33x   | -86.35%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.415518 | 0.052367   | 7.93x   | -87.40%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.529084 | 0.197830   | 2.67x   | -62.61%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.529084 | 0.057455   | 9.21x   | -89.14%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.529084 | 0.052812   | 10.02x  | -90.02%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.297597 | 0.085803   | 3.47x   | -71.17%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.297597 | 0.011312   | 26.31x  | -96.20%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.297597 | 0.010365   | 28.71x  | -96.52%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.408834 | 0.107870   | 3.79x   | -73.62%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.408834 | 0.014414   | 28.36x  | -96.47%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.408834 | 0.012677   | 32.25x  | -96.90%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.286007 | 0.114104   | 2.51x   | -60.10%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.286007 | 0.012356   | 23.15x  | -95.68%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.286007 | 0.012339   | 23.18x  | -95.69%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.313314 | 0.104068   | 3.01x   | -66.78%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.313314 | 0.013858   | 22.61x  | -95.58%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.313314 | 0.012676   | 24.72x  | -95.95%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.308274 | 0.108009   | 2.85x   | -64.96%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.308274 | 0.015423   | 19.99x  | -95.00%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.308274 | 0.012853   | 23.98x  | -95.83%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.316017 | 0.127612   | 2.48x   | -59.62%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.316017 | 0.016537   | 19.11x  | -94.77%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.316017 | 0.012963   | 24.38x  | -95.90%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.295302 | 0.117925   | 2.50x   | -60.07%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.295302 | 0.013024   | 22.67x  | -95.59%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.295302 | 0.012338   | 23.93x  | -95.82%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.312680 | 0.103911   | 3.01x   | -66.77%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.312680 | 0.012953   | 24.14x  | -95.86%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.312680 | 0.012389   | 25.24x  | -96.04%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.472015 | 0.169848   | 2.78x   | -64.02%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.472015 | 0.016293   | 28.97x  | -96.55%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.472015 | 0.012836   | 36.77x  | -97.28%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.417181 | 0.103180   | 4.04x   | -75.27%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.417181 | 0.012684   | 32.89x  | -96.96%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.417181 | 0.012600   | 33.11x  | -96.98%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.296651 | 0.100374   | 2.96x   | -66.16%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.296651 | 0.017240   | 17.21x  | -94.19%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.296651 | 0.013287   | 22.33x  | -95.52%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.322822 | 0.124302   | 2.60x   | -61.50%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.322822 | 0.013993   | 23.07x  | -95.67%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.322822 | 0.012550   | 25.72x  | -96.11%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.305454 | 0.124861   | 2.45x   | -59.12%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.305454 | 0.013807   | 22.12x  | -95.48%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.305454 | 0.012678   | 24.09x  | -95.85%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.314614 | 0.101904   | 3.09x   | -67.61%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.314614 | 0.014202   | 22.15x  | -95.49%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.314614 | 0.012403   | 25.37x  | -96.06%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.423314 | 0.164434   | 2.57x   | -61.16%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.423314 | 0.015045   | 28.14x  | -96.45%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.423314 | 0.012725   | 33.27x  | -96.99%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.340892 | 0.037836   | 9.01x   | -88.90%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.340892 | 0.020448   | 16.67x  | -94.00%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.340892 | 0.015857   | 21.50x  | -95.35%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.458260 | 0.045948   | 9.97x   | -89.97%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.458260 | 0.021482   | 21.33x  | -95.31%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.458260 | 0.017849   | 25.67x  | -96.10%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.341425 | 0.053868   | 6.34x   | -84.22%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.341425 | 0.020287   | 16.83x  | -94.06%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.341425 | 0.018192   | 18.77x  | -94.67%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.366637 | 0.042949   | 8.54x   | -88.29%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.366637 | 0.020406   | 17.97x  | -94.43%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.366637 | 0.017862   | 20.53x  | -95.13%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.365124 | 0.047999   | 7.61x   | -86.85%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.365124 | 0.021959   | 16.63x  | -93.99%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.365124 | 0.018446   | 19.79x  | -94.95%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.350954 | 0.107286   | 3.27x   | -69.43%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.350954 | 0.021321   | 16.46x  | -93.92%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.350954 | 0.018145   | 19.34x  | -94.83%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.338628 | 0.053611   | 6.32x   | -84.17%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.338628 | 0.020040   | 16.90x  | -94.08%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.338628 | 0.018453   | 18.35x  | -94.55%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.369247 | 0.041753   | 8.84x   | -88.69%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.369247 | 0.020667   | 17.87x  | -94.40%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.369247 | 0.017632   | 20.94x  | -95.22%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.522062 | 0.121401   | 4.30x   | -76.75%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.522062 | 0.022309   | 23.40x  | -95.73%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.522062 | 0.018229   | 28.64x  | -96.51%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.470792 | 0.041747   | 11.28x  | -91.13%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.470792 | 0.020431   | 23.04x  | -95.66%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.470792 | 0.017767   | 26.50x  | -96.23%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.357940 | 0.053270   | 6.72x   | -85.12%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.357940 | 0.021897   | 16.35x  | -93.88%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.357940 | 0.018296   | 19.56x  | -94.89%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.365178 | 0.072030   | 5.07x   | -80.28%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.365178 | 0.021588   | 16.92x  | -94.09%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.365178 | 0.017654   | 20.69x  | -95.17%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.362894 | 0.070829   | 5.12x   | -80.48%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.362894 | 0.020956   | 17.32x  | -94.23%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.362894 | 0.018042   | 20.11x  | -95.03%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.367250 | 0.045101   | 8.14x   | -87.72%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.367250 | 0.021645   | 16.97x  | -94.11%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.367250 | 0.018251   | 20.12x  | -95.03%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.481820 | 0.112726   | 4.27x   | -76.60%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.481820 | 0.020962   | 22.99x  | -95.65%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.481820 | 0.018147   | 26.55x  | -96.23%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.283864 | 0.047542   | 5.97x   | -83.25%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.283864 | 0.018985   | 14.95x  | -93.31%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.283864 | 0.018582   | 15.28x  | -93.45%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.387372 | 0.057409   | 6.75x   | -85.18%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.387372 | 0.022285   | 17.38x  | -94.25%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.387372 | 0.020732   | 18.69x  | -94.65%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.266081 | 0.060156   | 4.42x   | -77.39%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.266081 | 0.020574   | 12.93x  | -92.27%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.266081 | 0.020048   | 13.27x  | -92.47%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.290654 | 0.052438   | 5.54x   | -81.96%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.290654 | 0.020863   | 13.93x  | -92.82%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.290654 | 0.020479   | 14.19x  | -92.95%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.293972 | 0.059889   | 4.91x   | -79.63%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.293972 | 0.022973   | 12.80x  | -92.19%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.293972 | 0.020806   | 14.13x  | -92.92%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.273126 | 0.095569   | 2.86x   | -65.01%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.273126 | 0.021639   | 12.62x  | -92.08%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.273126 | 0.020392   | 13.39x  | -92.53%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.262408 | 0.060701   | 4.32x   | -76.87%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.262408 | 0.020707   | 12.67x  | -92.11%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.262408 | 0.019873   | 13.20x  | -92.43%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.276766 | 0.051167   | 5.41x   | -81.51%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.276766 | 0.021344   | 12.97x  | -92.29%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.276766 | 0.020362   | 13.59x  | -92.64%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.449680 | 0.127955   | 3.51x   | -71.55%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.449680 | 0.024499   | 18.36x  | -94.55%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.449680 | 0.020710   | 21.71x  | -95.39%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.386572 | 0.051100   | 7.56x   | -86.78%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.386572 | 0.020590   | 18.78x  | -94.67%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.386572 | 0.020290   | 19.05x  | -94.75%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.277393 | 0.064894   | 4.27x   | -76.61%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.277393 | 0.022450   | 12.36x  | -91.91%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.277393 | 0.020651   | 13.43x  | -92.56%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.286091 | 0.077313   | 3.70x   | -72.98%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.286091 | 0.020637   | 13.86x  | -92.79%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.286091 | 0.020257   | 14.12x  | -92.92%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.284656 | 0.076926   | 3.70x   | -72.98%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.284656 | 0.021227   | 13.41x  | -92.54%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.284656 | 0.020371   | 13.97x  | -92.84%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.298294 | 0.055729   | 5.35x   | -81.32%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.298294 | 0.021102   | 14.14x  | -92.93%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.298294 | 0.020994   | 14.21x  | -92.96%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.404662 | 0.120967   | 3.35x   | -70.11%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.404662 | 0.021593   | 18.74x  | -94.66%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.404662 | 0.020408   | 19.83x  | -94.96%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.083692 | 0.023561   | 3.55x   | -71.85%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.083692 | 0.011047   | 7.58x   | -86.80%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.083692 | 0.010595   | 7.90x   | -87.34%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.113002 | 0.025539   | 4.42x   | -77.40%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.113002 | 0.012317   | 9.17x   | -89.10%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.113002 | 0.011545   | 9.79x   | -89.78%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.088649 | 0.027668   | 3.20x   | -68.79%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.088649 | 0.011258   | 7.87x   | -87.30%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.088649 | 0.010694   | 8.29x   | -87.94%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.092130 | 0.024555   | 3.75x   | -73.35%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.092130 | 0.012010   | 7.67x   | -86.96%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.092130 | 0.010946   | 8.42x   | -88.12%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.092421 | 0.025663   | 3.60x   | -72.23%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.092421 | 0.012631   | 7.32x   | -86.33%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.092421 | 0.011419   | 8.09x   | -87.65%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.086937 | 0.035594   | 2.44x   | -59.06%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.086937 | 0.012021   | 7.23x   | -86.17%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.086937 | 0.010960   | 7.93x   | -87.39%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.089794 | 0.027657   | 3.25x   | -69.20%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.089794 | 0.010892   | 8.24x   | -87.87%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.089794 | 0.010836   | 8.29x   | -87.93%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.090957 | 0.024733   | 3.68x   | -72.81%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.090957 | 0.011833   | 7.69x   | -86.99%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.090957 | 0.011218   | 8.11x   | -87.67%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.122647 | 0.042450   | 2.89x   | -65.39%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.122647 | 0.012872   | 9.53x   | -89.50%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.122647 | 0.011542   | 10.63x  | -90.59%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.118278 | 0.025088   | 4.71x   | -78.79%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.118278 | 0.011483   | 10.30x  | -90.29%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.118278 | 0.010827   | 10.92x  | -90.85%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.091255 | 0.024271   | 3.76x   | -73.40%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.091255 | 0.012428   | 7.34x   | -86.38%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.091255 | 0.011537   | 7.91x   | -87.36%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.092646 | 0.030314   | 3.06x   | -67.28%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.092646 | 0.012115   | 7.65x   | -86.92%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.092646 | 0.011315   | 8.19x   | -87.79%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.091983 | 0.030402   | 3.03x   | -66.95%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.091983 | 0.012969   | 7.09x   | -85.90%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.091983 | 0.011878   | 7.74x   | -87.09%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.099875 | 0.025704   | 3.89x   | -74.26%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.099875 | 0.012891   | 7.75x   | -87.09%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.099875 | 0.011859   | 8.42x   | -88.13%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.137669 | 0.042477   | 3.24x   | -69.15%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.137669 | 0.012640   | 10.89x  | -90.82%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.137669 | 0.011517   | 11.95x  | -91.63%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.068117 | 0.020602   | 3.31x   | -69.75%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.068117 | 0.002641   | 25.79x  | -96.12%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.068117 | 0.002296   | 29.67x  | -96.63%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.105977 | 0.024850   | 4.26x   | -76.55%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.105977 | 0.003390   | 31.26x  | -96.80%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.105977 | 0.002793   | 37.94x  | -97.36%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.091284 | 0.025057   | 3.64x   | -72.55%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.091284 | 0.002690   | 33.94x  | -97.05%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.091284 | 0.002711   | 33.67x  | -97.03%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.091036 | 0.025371   | 3.59x   | -72.13%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.091036 | 0.003665   | 24.84x  | -95.97%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.091036 | 0.002840   | 32.06x  | -96.88%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.091949 | 0.024464   | 3.76x   | -73.39%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.091949 | 0.003627   | 25.35x  | -96.05%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.091949 | 0.003060   | 30.05x  | -96.67%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.086738 | 0.030385   | 2.85x   | -64.97%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.086738 | 0.003925   | 22.10x  | -95.48%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.086738 | 0.003012   | 28.80x  | -96.53%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.082981 | 0.028992   | 2.86x   | -65.06%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.082981 | 0.002756   | 30.11x  | -96.68%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.082981 | 0.002673   | 31.05x  | -96.78%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.081881 | 0.022332   | 3.67x   | -72.73%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.081881 | 0.002823   | 29.00x  | -96.55%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.081881 | 0.003610   | 22.68x  | -95.59%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.111934 | 0.037490   | 2.99x   | -66.51%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.111934 | 0.003340   | 33.51x  | -97.02%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.111934 | 0.002752   | 40.67x  | -97.54%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.104692 | 0.021804   | 4.80x   | -79.17%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.104692 | 0.002843   | 36.83x  | -97.28%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.104692 | 0.002671   | 39.20x  | -97.45%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.075189 | 0.021479   | 3.50x   | -71.43%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.075189 | 0.003749   | 20.05x  | -95.01%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.075189 | 0.002897   | 25.95x  | -96.15%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.081814 | 0.029208   | 2.80x   | -64.30%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.081814 | 0.003033   | 26.98x  | -96.29%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.081814 | 0.002828   | 28.93x  | -96.54%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.079317 | 0.028269   | 2.81x   | -64.36%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.079317 | 0.003096   | 25.62x  | -96.10%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.079317 | 0.002794   | 28.38x  | -96.48%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.084344 | 0.024215   | 3.48x   | -71.29%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.084344 | 0.003106   | 27.16x  | -96.32%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.084344 | 0.002716   | 31.06x  | -96.78%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.116102 | 0.037758   | 3.07x   | -67.48%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.116102 | 0.003257   | 35.65x  | -97.19%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.116102 | 0.002893   | 40.14x  | -97.51%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.079508 | 0.007125   | 11.16x  | -91.04%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.079508 | 0.003328   | 23.89x  | -95.81%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.079508 | 0.002327   | 34.16x  | -97.07%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.113552 | 0.009224   | 12.31x  | -91.88%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.113552 | 0.003454   | 32.88x  | -96.96%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.113552 | 0.002856   | 39.76x  | -97.49%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.086195 | 0.010494   | 8.21x   | -87.83%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.086195 | 0.003209   | 26.86x  | -96.28%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.086195 | 0.002669   | 32.30x  | -96.90%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.087690 | 0.008280   | 10.59x  | -90.56%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.087690 | 0.003377   | 25.97x  | -96.15%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.087690 | 0.002824   | 31.06x  | -96.78%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.094323 | 0.009584   | 9.84x   | -89.84%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.094323 | 0.003674   | 25.68x  | -96.11%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.094323 | 0.002826   | 33.38x  | -97.00%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.082740 | 0.022351   | 3.70x   | -72.99%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.082740 | 0.003268   | 25.32x  | -96.05%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.082740 | 0.002628   | 31.49x  | -96.82%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.083394 | 0.010108   | 8.25x   | -87.88%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.083394 | 0.003036   | 27.47x  | -96.36%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.083394 | 0.002531   | 32.95x  | -96.97%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.082662 | 0.007968   | 10.37x  | -90.36%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.082662 | 0.003058   | 27.03x  | -96.30%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.082662 | 0.003420   | 24.17x  | -95.86%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.116530 | 0.025963   | 4.49x   | -77.72%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.116530 | 0.003553   | 32.80x  | -96.95%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.116530 | 0.002838   | 41.06x  | -97.56%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.112935 | 0.007685   | 14.70x  | -93.20%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.112935 | 0.003503   | 32.24x  | -96.90%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.112935 | 0.002644   | 42.72x  | -97.66%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.084784 | 0.011192   | 7.58x   | -86.80%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.084784 | 0.003636   | 23.32x  | -95.71%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.084784 | 0.002753   | 30.80x  | -96.75%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.088907 | 0.016170   | 5.50x   | -81.81%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.088907 | 0.004101   | 21.68x  | -95.39%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.088907 | 0.003042   | 29.23x  | -96.58%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.090694 | 0.014212   | 6.38x   | -84.33%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.090694 | 0.004860   | 18.66x  | -94.64%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.090694 | 0.002619   | 34.62x  | -97.11%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.086859 | 0.008679   | 10.01x  | -90.01%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.086859 | 0.003373   | 25.75x  | -96.12%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.086859 | 0.002648   | 32.80x  | -96.95%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.108730 | 0.023392   | 4.65x   | -78.49%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.108730 | 0.003460   | 31.42x  | -96.82%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.108730 | 0.002596   | 41.88x  | -97.61%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.058669 | 0.008657   | 6.78x   | -85.24%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.058669 | 0.002545   | 23.05x  | -95.66%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.058669 | 0.002510   | 23.37x  | -95.72%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.090257 | 0.011905   | 7.58x   | -86.81%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.090257 | 0.002841   | 31.77x  | -96.85%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.090257 | 0.004003   | 22.55x  | -95.56%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.067247 | 0.011132   | 6.04x   | -83.45%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.067247 | 0.002942   | 22.86x  | -95.63%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.067247 | 0.003714   | 18.11x  | -94.48%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.073024 | 0.010501   | 6.95x   | -85.62%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.073024 | 0.002992   | 24.41x  | -95.90%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.073024 | 0.003647   | 20.02x  | -95.01%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.069329 | 0.010378   | 6.68x   | -85.03%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.069329 | 0.003149   | 22.02x  | -95.46%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.069329 | 0.003019   | 22.96x  | -95.64%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.064381 | 0.019433   | 3.31x   | -69.82%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.064381 | 0.002953   | 21.80x  | -95.41%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.064381 | 0.003820   | 16.85x  | -94.07%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.066619 | 0.010834   | 6.15x   | -83.74%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.066619 | 0.002718   | 24.51x  | -95.92%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.066619 | 0.003639   | 18.31x  | -94.54%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.067735 | 0.009488   | 7.14x   | -85.99%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.067735 | 0.002577   | 26.28x  | -96.20%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.067735 | 0.003326   | 20.37x  | -95.09%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.096822 | 0.026172   | 3.70x   | -72.97%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.096822 | 0.003420   | 28.31x  | -96.47%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.096822 | 0.003909   | 24.77x  | -95.96%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.092748 | 0.009761   | 9.50x   | -89.48%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.092748 | 0.002863   | 32.39x  | -96.91%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.092748 | 0.004617   | 20.09x  | -95.02%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.066442 | 0.012939   | 5.14x   | -80.53%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.066442 | 0.003525   | 18.85x  | -94.70%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.066442 | 0.003795   | 17.51x  | -94.29%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.071189 | 0.015036   | 4.73x   | -78.88%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.071189 | 0.004843   | 14.70x  | -93.20%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.071189 | 0.003391   | 20.99x  | -95.24%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.072529 | 0.016170   | 4.49x   | -77.71%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.072529 | 0.002886   | 25.13x  | -96.02%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.072529 | 0.003208   | 22.61x  | -95.58%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.073881 | 0.012095   | 6.11x   | -83.63%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.073881 | 0.003264   | 22.64x  | -95.58%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.073881 | 0.004247   | 17.40x  | -94.25%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.099784 | 0.025091   | 3.98x   | -74.85%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.099784 | 0.003550   | 28.11x  | -96.44%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.099784 | 0.003635   | 27.45x  | -96.36%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.191473 | 0.053395   | 3.59x   | -72.11%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.191473 | 0.025139   | 7.62x   | -86.87%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.191473 | 0.027003   | 7.09x   | -85.90%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.261490 | 0.058022   | 4.51x   | -77.81%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.261490 | 0.028053   | 9.32x   | -89.27%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.261490 | 0.026583   | 9.84x   | -89.83%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.201484 | 0.062849   | 3.21x   | -68.81%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.201484 | 0.024950   | 8.08x   | -87.62%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.201484 | 0.024519   | 8.22x   | -87.83%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.203960 | 0.056217   | 3.63x   | -72.44%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.203960 | 0.026099   | 7.81x   | -87.20%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.203960 | 0.025520   | 7.99x   | -87.49%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.211274 | 0.058641   | 3.60x   | -72.24%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.211274 | 0.029006   | 7.28x   | -86.27%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.211274 | 0.027023   | 7.82x   | -87.21%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.204530 | 0.080802   | 2.53x   | -60.49%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.204530 | 0.026230   | 7.80x   | -87.18%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.204530 | 0.024642   | 8.30x   | -87.95%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.198080 | 0.062586   | 3.16x   | -68.40%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.198080 | 0.025250   | 7.84x   | -87.25%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.198080 | 0.024353   | 8.13x   | -87.71%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.205222 | 0.055907   | 3.67x   | -72.76%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.205222 | 0.026029   | 7.88x   | -87.32%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.205222 | 0.024518   | 8.37x   | -88.05%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.272242 | 0.096481   | 2.82x   | -64.56%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.272242 | 0.028793   | 9.46x   | -89.42%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.272242 | 0.026183   | 10.40x  | -90.38%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.257766 | 0.054905   | 4.69x   | -78.70%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.257766 | 0.028446   | 9.06x   | -88.96%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.257766 | 0.028321   | 9.10x   | -89.01%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.208556 | 0.054481   | 3.83x   | -73.88%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.208556 | 0.027892   | 7.48x   | -86.63%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.208556 | 0.026081   | 8.00x   | -87.49%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.205211 | 0.068916   | 2.98x   | -66.42%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.205211 | 0.027669   | 7.42x   | -86.52%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.205211 | 0.025879   | 7.93x   | -87.39%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.204840 | 0.068610   | 2.99x   | -66.51%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.204840 | 0.027580   | 7.43x   | -86.54%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.204840 | 0.025908   | 7.91x   | -87.35%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.208482 | 0.057317   | 3.64x   | -72.51%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.208482 | 0.028563   | 7.30x   | -86.30%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.208482 | 0.026030   | 8.01x   | -87.51%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.259402 | 0.093980   | 2.76x   | -63.77%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.259402 | 0.028739   | 9.03x   | -88.92%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.259402 | 0.026102   | 9.94x   | -89.94%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.140995 | 0.043237   | 3.26x   | -69.33%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.140995 | 0.005717   | 24.66x  | -95.95%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.140995 | 0.005180   | 27.22x  | -96.33%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.205817 | 0.053575   | 3.84x   | -73.97%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.205817 | 0.007253   | 28.38x  | -96.48%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.205817 | 0.006229   | 33.04x  | -96.97%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.151199 | 0.070555   | 2.14x   | -53.34%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.151199 | 0.006879   | 21.98x  | -95.45%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.151199 | 0.006278   | 24.09x  | -95.85%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.160218 | 0.051954   | 3.08x   | -67.57%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.160218 | 0.006792   | 23.59x  | -95.76%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.160218 | 0.006212   | 25.79x  | -96.12%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.175725 | 0.053815   | 3.27x   | -69.38%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.175725 | 0.007585   | 23.17x  | -95.68%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.175725 | 0.006265   | 28.05x  | -96.43%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.153502 | 0.064526   | 2.38x   | -57.96%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.153502 | 0.008467   | 18.13x  | -94.48%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.153502 | 0.006515   | 23.56x  | -95.76%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.156699 | 0.062544   | 2.51x   | -60.09%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.156699 | 0.006570   | 23.85x  | -95.81%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.156699 | 0.006157   | 25.45x  | -96.07%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.155740 | 0.052001   | 2.99x   | -66.61%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.155740 | 0.006651   | 23.42x  | -95.73%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.155740 | 0.006172   | 25.23x  | -96.04%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.227413 | 0.084767   | 2.68x   | -62.73%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.227413 | 0.008050   | 28.25x  | -96.46%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.227413 | 0.006258   | 36.34x  | -97.25%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.215861 | 0.052144   | 4.14x   | -75.84%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.215861 | 0.006621   | 32.60x  | -96.93%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.215861 | 0.006356   | 33.96x  | -97.06%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.152824 | 0.050712   | 3.01x   | -66.82%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.152824 | 0.009033   | 16.92x  | -94.09%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.152824 | 0.006625   | 23.07x  | -95.67%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.158990 | 0.061542   | 2.58x   | -61.29%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.158990 | 0.006872   | 23.14x  | -95.68%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.158990 | 0.006534   | 24.33x  | -95.89%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.155224 | 0.061390   | 2.53x   | -60.45%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.155224 | 0.006700   | 23.17x  | -95.68%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.155224 | 0.006247   | 24.85x  | -95.98%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.157953 | 0.052634   | 3.00x   | -66.68%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.157953 | 0.007067   | 22.35x  | -95.53%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.157953 | 0.006261   | 25.23x  | -96.04%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.217319 | 0.082410   | 2.64x   | -62.08%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.217319 | 0.007920   | 27.44x  | -96.36%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.217319 | 0.006311   | 34.44x  | -97.10%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.168567 | 0.017197   | 9.80x   | -89.80%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.168567 | 0.007565   | 22.28x  | -95.51%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.168567 | 0.004833   | 34.88x  | -97.13%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.230144 | 0.022474   | 10.24x  | -90.23%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.230144 | 0.007722   | 29.80x  | -96.64%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.230144 | 0.006355   | 36.22x  | -97.24%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.174749 | 0.024788   | 7.05x   | -85.82%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.174749 | 0.006814   | 25.64x  | -96.10%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.174749 | 0.006318   | 27.66x  | -96.38%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.180933 | 0.019689   | 9.19x   | -89.12%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.180933 | 0.007525   | 24.05x  | -95.84%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.180933 | 0.006355   | 28.47x  | -96.49%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.180903 | 0.022548   | 8.02x   | -87.54%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.180903 | 0.008245   | 21.94x  | -95.44%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.180903 | 0.006639   | 27.25x  | -96.33%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.176927 | 0.052276   | 3.38x   | -70.45%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.176927 | 0.007899   | 22.40x  | -95.54%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.176927 | 0.006506   | 27.19x  | -96.32%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.173973 | 0.024919   | 6.98x   | -85.68%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.173973 | 0.007627   | 22.81x  | -95.62%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.173973 | 0.006199   | 28.07x  | -96.44%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.180597 | 0.019608   | 9.21x   | -89.14%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.180597 | 0.007007   | 25.77x  | -96.12%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.180597 | 0.006114   | 29.54x  | -96.61%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.248737 | 0.058588   | 4.25x   | -76.45%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.248737 | 0.008603   | 28.91x  | -96.54%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.248737 | 0.006402   | 38.85x  | -97.43%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.236866 | 0.019192   | 12.34x  | -91.90%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.236866 | 0.007932   | 29.86x  | -96.65%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.236866 | 0.005912   | 40.06x  | -97.50%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.177891 | 0.024397   | 7.29x   | -86.29%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.177891 | 0.008513   | 20.90x  | -95.21%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.177891 | 0.006185   | 28.76x  | -96.52%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.182045 | 0.034091   | 5.34x   | -81.27%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.182045 | 0.007616   | 23.90x  | -95.82%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.182045 | 0.006853   | 26.56x  | -96.24%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.195037 | 0.034111   | 5.72x   | -82.51%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.195037 | 0.007790   | 25.04x  | -96.01%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.195037 | 0.006111   | 31.91x  | -96.87%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.182003 | 0.021138   | 8.61x   | -88.39%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.182003 | 0.008173   | 22.27x  | -95.51%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.182003 | 0.006158   | 29.56x  | -96.62%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.246080 | 0.055941   | 4.40x   | -77.27%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.246080 | 0.008591   | 28.65x  | -96.51%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.246080 | 0.006302   | 39.05x  | -97.44%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.127986 | 0.020001   | 6.40x   | -84.37%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.127986 | 0.006277   | 20.39x  | -95.10%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.127986 | 0.005692   | 22.49x  | -95.55%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.193794 | 0.024624   | 7.87x   | -87.29%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.193794 | 0.006613   | 29.31x  | -96.59%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.193794 | 0.007596   | 25.51x  | -96.08%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.138987 | 0.026292   | 5.29x   | -81.08%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.138987 | 0.006522   | 21.31x  | -95.31%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.138987 | 0.007326   | 18.97x  | -94.73%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.147975 | 0.022535   | 6.57x   | -84.77%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.147975 | 0.006817   | 21.71x  | -95.39%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.147975 | 0.008198   | 18.05x  | -94.46%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.146489 | 0.025056   | 5.85x   | -82.90%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.146489 | 0.007888   | 18.57x  | -94.62%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.146489 | 0.006826   | 21.46x  | -95.34%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.139519 | 0.043854   | 3.18x   | -68.57%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.139519 | 0.006900   | 20.22x  | -95.05%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.139519 | 0.007696   | 18.13x  | -94.48%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.136599 | 0.025992   | 5.26x   | -80.97%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.136599 | 0.006449   | 21.18x  | -95.28%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.136599 | 0.007634   | 17.89x  | -94.41%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.140809 | 0.020941   | 6.72x   | -85.13%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.140809 | 0.007001   | 20.11x  | -95.03%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.140809 | 0.007370   | 19.11x  | -94.77%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.211790 | 0.059040   | 3.59x   | -72.12%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.211790 | 0.007929   | 26.71x  | -96.26%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.211790 | 0.007128   | 29.71x  | -96.63%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.199667 | 0.021116   | 9.46x   | -89.42%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.199667 | 0.006292   | 31.73x  | -96.85%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.199667 | 0.007618   | 26.21x  | -96.18%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.137614 | 0.028769   | 4.78x   | -79.09%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.137614 | 0.007183   | 19.16x  | -94.78%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.137614 | 0.008287   | 16.61x  | -93.98%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.143890 | 0.034585   | 4.16x   | -75.96%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.143890 | 0.006388   | 22.52x  | -95.56%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.143890 | 0.008296   | 17.34x  | -94.23%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.142414 | 0.034313   | 4.15x   | -75.91%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.142414 | 0.006630   | 21.48x  | -95.34%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.142414 | 0.007293   | 19.53x  | -94.88%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.145538 | 0.025379   | 5.73x   | -82.56%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.145538 | 0.006291   | 23.13x  | -95.68%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.145538 | 0.007190   | 20.24x  | -95.06%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.196957 | 0.056664   | 3.48x   | -71.23%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.196957 | 0.006867   | 28.68x  | -96.51%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.196957 | 0.007747   | 25.42x  | -96.07%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.343469 | 0.093357   | 3.68x   | -72.82%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.343469 | 0.044404   | 7.74x   | -87.07%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.343469 | 0.043170   | 7.96x   | -87.43%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.453740 | 0.104216   | 4.35x   | -77.03%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.453740 | 0.049737   | 9.12x   | -89.04%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.453740 | 0.046139   | 9.83x   | -89.83%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.349368 | 0.111891   | 3.12x   | -67.97%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.349368 | 0.045230   | 7.72x   | -87.05%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.349368 | 0.044097   | 7.92x   | -87.38%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.366577 | 0.107369   | 3.41x   | -70.71%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.366577 | 0.049554   | 7.40x   | -86.48%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.366577 | 0.045190   | 8.11x   | -87.67%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.370528 | 0.103141   | 3.59x   | -72.16%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.370528 | 0.050647   | 7.32x   | -86.33%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.370528 | 0.046123   | 8.03x   | -87.55%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.349277 | 0.147529   | 2.37x   | -57.76%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.349277 | 0.047164   | 7.41x   | -86.50%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.349277 | 0.048294   | 7.23x   | -86.17%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.357872 | 0.111735   | 3.20x   | -68.78%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.357872 | 0.045503   | 7.86x   | -87.29%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.357872 | 0.044030   | 8.13x   | -87.70%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.358512 | 0.103198   | 3.47x   | -71.21%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.358512 | 0.046482   | 7.71x   | -87.03%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.358512 | 0.044751   | 8.01x   | -87.52%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.520383 | 0.173434   | 3.00x   | -66.67%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.520383 | 0.052506   | 9.91x   | -89.91%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.520383 | 0.048939   | 10.63x  | -90.60%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.471255 | 0.099063   | 4.76x   | -78.98%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.471255 | 0.046757   | 10.08x  | -90.08%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.471255 | 0.043799   | 10.76x  | -90.71%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.349064 | 0.095952   | 3.64x   | -72.51%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.349064 | 0.049257   | 7.09x   | -85.89%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.349064 | 0.045295   | 7.71x   | -87.02%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.366871 | 0.123885   | 2.96x   | -66.23%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.366871 | 0.049289   | 7.44x   | -86.56%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.366871 | 0.046102   | 7.96x   | -87.43%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.369353 | 0.124057   | 2.98x   | -66.41%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.369353 | 0.050612   | 7.30x   | -86.30%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.369353 | 0.046762   | 7.90x   | -87.34%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.371450 | 0.101299   | 3.67x   | -72.73%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.371450 | 0.050257   | 7.39x   | -86.47%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.371450 | 0.046689   | 7.96x   | -87.43%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.462350 | 0.166243   | 2.78x   | -64.04%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.462350 | 0.051172   | 9.04x   | -88.93%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.462350 | 0.046566   | 9.93x   | -89.93%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.261227 | 0.079749   | 3.28x   | -69.47%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.261227 | 0.010121   | 25.81x  | -96.13%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.261227 | 0.009241   | 28.27x  | -96.46%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.365297 | 0.094707   | 3.86x   | -74.07%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.365297 | 0.012580   | 29.04x  | -96.56%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.365297 | 0.011150   | 32.76x  | -96.95%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.251906 | 0.099255   | 2.54x   | -60.60%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.251906 | 0.011006   | 22.89x  | -95.63%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.251906 | 0.010690   | 23.56x  | -95.76%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.272915 | 0.092411   | 2.95x   | -66.14%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.272915 | 0.011965   | 22.81x  | -95.62%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.272915 | 0.011086   | 24.62x  | -95.94%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.273653 | 0.096490   | 2.84x   | -64.74%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.273653 | 0.013579   | 20.15x  | -95.04%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.273653 | 0.011311   | 24.19x  | -95.87%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.263792 | 0.114189   | 2.31x   | -56.71%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.263792 | 0.014626   | 18.04x  | -94.46%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.263792 | 0.011417   | 23.11x  | -95.67%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.253650 | 0.100333   | 2.53x   | -60.44%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.253650 | 0.011162   | 22.72x  | -95.60%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.253650 | 0.010789   | 23.51x  | -95.75%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.262905 | 0.091294   | 2.88x   | -65.28%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.262905 | 0.011308   | 23.25x  | -95.70%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.262905 | 0.011078   | 23.73x  | -95.79%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.414744 | 0.148575   | 2.79x   | -64.18%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.414744 | 0.013643   | 30.40x  | -96.71%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.414744 | 0.011245   | 36.88x  | -97.29%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.361731 | 0.088641   | 4.08x   | -75.50%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.361731 | 0.011727   | 30.85x  | -96.76%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.361731 | 0.010681   | 33.87x  | -97.05%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.259553 | 0.087741   | 2.96x   | -66.20%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.259553 | 0.014965   | 17.34x  | -94.23%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.259553 | 0.011583   | 22.41x  | -95.54%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.265384 | 0.107052   | 2.48x   | -59.66%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.265384 | 0.012062   | 22.00x  | -95.45%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.265384 | 0.010972   | 24.19x  | -95.87%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.268058 | 0.107005   | 2.51x   | -60.08%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.268058 | 0.011989   | 22.36x  | -95.53%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.268058 | 0.011027   | 24.31x  | -95.89%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.293290 | 0.089993   | 3.26x   | -69.32%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.293290 | 0.012581   | 23.31x  | -95.71%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.293290 | 0.010805   | 27.14x  | -96.32%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.371362 | 0.144248   | 2.57x   | -61.16%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.371362 | 0.012999   | 28.57x  | -96.50%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.371362 | 0.011207   | 33.14x  | -96.98%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.299843 | 0.028107   | 10.67x  | -90.63%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.299843 | 0.012851   | 23.33x  | -95.71%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.299843 | 0.008898   | 33.70x  | -97.03%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.410966 | 0.036525   | 11.25x  | -91.11%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.410966 | 0.014103   | 29.14x  | -96.57%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.410966 | 0.011539   | 35.62x  | -97.19%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.301940 | 0.042068   | 7.18x   | -86.07%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.301940 | 0.013040   | 23.16x  | -95.68%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.301940 | 0.010802   | 27.95x  | -96.42%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.317562 | 0.033131   | 9.58x   | -89.57%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.317562 | 0.013200   | 24.06x  | -95.84%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.317562 | 0.011170   | 28.43x  | -96.48%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.320691 | 0.036687   | 8.74x   | -88.56%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.320691 | 0.014323   | 22.39x  | -95.53%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.320691 | 0.011157   | 28.74x  | -96.52%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.311933 | 0.090221   | 3.46x   | -71.08%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.311933 | 0.014028   | 22.24x  | -95.50%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.311933 | 0.011066   | 28.19x  | -96.45%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.299649 | 0.041721   | 7.18x   | -86.08%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.299649 | 0.013125   | 22.83x  | -95.62%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.299649 | 0.010717   | 27.96x  | -96.42%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.314884 | 0.031779   | 9.91x   | -89.91%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.314884 | 0.012839   | 24.53x  | -95.92%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.314884 | 0.011150   | 28.24x  | -96.46%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.456706 | 0.100647   | 4.54x   | -77.96%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.456706 | 0.014381   | 31.76x  | -96.85%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.456706 | 0.011010   | 41.48x  | -97.59%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.403724 | 0.031710   | 12.73x  | -92.15%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.403724 | 0.013156   | 30.69x  | -96.74%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.403724 | 0.010741   | 37.59x  | -97.34%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.306701 | 0.040543   | 7.56x   | -86.78%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.306701 | 0.014098   | 21.76x  | -95.40%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.306701 | 0.011452   | 26.78x  | -96.27%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.315267 | 0.058029   | 5.43x   | -81.59%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.315267 | 0.013550   | 23.27x  | -95.70%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.315267 | 0.010784   | 29.23x  | -96.58%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.313943 | 0.057481   | 5.46x   | -81.69%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.313943 | 0.013660   | 22.98x  | -95.65%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.313943 | 0.010894   | 28.82x  | -96.53%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.321753 | 0.035855   | 8.97x   | -88.86%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.321753 | 0.013911   | 23.13x  | -95.68%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.321753 | 0.011155   | 28.84x  | -96.53%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.418176 | 0.095587   | 4.37x   | -77.14%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.418176 | 0.014234   | 29.38x  | -96.60%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.418176 | 0.011304   | 36.99x  | -97.30%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.245412 | 0.041426   | 5.92x   | -83.12%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.245412 | 0.016659   | 14.73x  | -93.21%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.245412 | 0.016808   | 14.60x  | -93.15%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.348281 | 0.048359   | 7.20x   | -86.11%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.348281 | 0.018266   | 19.07x  | -94.76%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.348281 | 0.017631   | 19.75x  | -94.94%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.242257 | 0.052027   | 4.66x   | -78.52%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.242257 | 0.018077   | 13.40x  | -92.54%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.242257 | 0.017793   | 13.61x  | -92.66%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.263067 | 0.045785   | 5.75x   | -82.60%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.263067 | 0.018541   | 14.19x  | -92.95%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.263067 | 0.017725   | 14.84x  | -93.26%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.262153 | 0.049692   | 5.28x   | -81.04%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.262153 | 0.020253   | 12.94x  | -92.27%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.262153 | 0.017389   | 15.08x  | -93.37%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.248605 | 0.083811   | 2.97x   | -66.29%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.248605 | 0.018989   | 13.09x  | -92.36%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.248605 | 0.017650   | 14.09x  | -92.90%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.261743 | 0.052452   | 4.99x   | -79.96%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.261743 | 0.018446   | 14.19x  | -92.95%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.261743 | 0.017469   | 14.98x  | -93.33%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.258049 | 0.043725   | 5.90x   | -83.06%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.258049 | 0.018144   | 14.22x  | -92.97%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.258049 | 0.017487   | 14.76x  | -93.22%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.396788 | 0.110363   | 3.60x   | -72.19%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.396788 | 0.020363   | 19.49x  | -94.87%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.396788 | 0.017691   | 22.43x  | -95.54%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.349466 | 0.044306   | 7.89x   | -87.32%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.349466 | 0.018209   | 19.19x  | -94.79%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.349466 | 0.017801   | 19.63x  | -94.91%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.247102 | 0.056538   | 4.37x   | -77.12%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.247102 | 0.018812   | 13.14x  | -92.39%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.247102 | 0.017937   | 13.78x  | -92.74%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.259227 | 0.067098   | 3.86x   | -74.12%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.259227 | 0.019084   | 13.58x  | -92.64%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.259227 | 0.017624   | 14.71x  | -93.20%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.259468 | 0.067288   | 3.86x   | -74.07%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.259468 | 0.018379   | 14.12x  | -92.92%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.259468 | 0.016990   | 15.27x  | -93.45%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.267044 | 0.047986   | 5.56x   | -82.03%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.267044 | 0.018476   | 14.45x  | -93.08%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.267044 | 0.017367   | 15.38x  | -93.50%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.372928 | 0.107258   | 3.48x   | -71.24%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.372928 | 0.021055   | 17.71x  | -94.35%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.372928 | 0.018189   | 20.50x  | -95.12%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.777226 | 0.209868   | 3.70x   | -73.00%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.777226 | 0.098482   | 7.89x   | -87.33%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.777226 | 0.096447   | 8.06x   | -87.59%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 1.001978 | 0.234630   | 4.27x   | -76.58%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 1.001978 | 0.111559   | 8.98x   | -88.87%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 1.001978 | 0.104277   | 9.61x   | -89.59%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.762246 | 0.252981   | 3.01x   | -66.81%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.762246 | 0.101035   | 7.54x   | -86.75%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.762246 | 0.099221   | 7.68x   | -86.98%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.800773 | 0.226843   | 3.53x   | -71.67%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.800773 | 0.103239   | 7.76x   | -87.11%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.800773 | 0.100694   | 7.95x   | -87.43%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.815498 | 0.239696   | 3.40x   | -70.61%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.815498 | 0.116677   | 6.99x   | -85.69%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.815498 | 0.107267   | 7.60x   | -86.85%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.796095 | 0.328626   | 2.42x   | -58.72%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.796095 | 0.105904   | 7.52x   | -86.70%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.796095 | 0.101312   | 7.86x   | -87.27%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.808652 | 0.260813   | 3.10x   | -67.75%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.808652 | 0.104252   | 7.76x   | -87.11%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.808652 | 0.099973   | 8.09x   | -87.64%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.807482 | 0.228973   | 3.53x   | -71.64%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.807482 | 0.104569   | 7.72x   | -87.05%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.807482 | 0.101460   | 7.96x   | -87.43%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.142282 | 0.393325   | 2.90x   | -65.57%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.142282 | 0.116900   | 9.77x   | -89.77%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.142282 | 0.108227   | 10.55x  | -90.53%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 1.021413 | 0.227278   | 4.49x   | -77.75%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 1.021413 | 0.103373   | 9.88x   | -89.88%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 1.021413 | 0.103105   | 9.91x   | -89.91%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.806679 | 0.221012   | 3.65x   | -72.60%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.806679 | 0.113932   | 7.08x   | -85.88%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.806679 | 0.104837   | 7.69x   | -87.00%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.805649 | 0.275660   | 2.92x   | -65.78%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.805649 | 0.111147   | 7.25x   | -86.20%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.805649 | 0.103661   | 7.77x   | -87.13%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.807312 | 0.276106   | 2.92x   | -65.80%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.807312 | 0.110333   | 7.32x   | -86.33%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.807312 | 0.102387   | 7.88x   | -87.32%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.807938 | 0.232767   | 3.47x   | -71.19%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.807938 | 0.113891   | 7.09x   | -85.90%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.807938 | 0.105621   | 7.65x   | -86.93%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 1.040889 | 0.376450   | 2.77x   | -63.83%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 1.040889 | 0.113657   | 9.16x   | -89.08%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 1.040889 | 0.105458   | 9.87x   | -89.87%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.590228 | 0.174770   | 3.38x   | -70.39%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.590228 | 0.022941   | 25.73x  | -96.11%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.590228 | 0.020352   | 29.00x  | -96.55%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.806591 | 0.217673   | 3.71x   | -73.01%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.806591 | 0.028704   | 28.10x  | -96.44%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.806591 | 0.025388   | 31.77x  | -96.85%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.575430 | 0.225978   | 2.55x   | -60.73%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.575430 | 0.025059   | 22.96x  | -95.65%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.575430 | 0.024545   | 23.44x  | -95.73%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.615049 | 0.210342   | 2.92x   | -65.80%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.615049 | 0.027813   | 22.11x  | -95.48%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.615049 | 0.024930   | 24.67x  | -95.95%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.615212 | 0.218137   | 2.82x   | -64.54%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.615212 | 0.030696   | 20.04x  | -95.01%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.615212 | 0.025714   | 23.92x  | -95.82%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.588293 | 0.254423   | 2.31x   | -56.75%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.588293 | 0.032789   | 17.94x  | -94.43%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.588293 | 0.025705   | 22.89x  | -95.63%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.563506 | 0.226899   | 2.48x   | -59.73%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.563506 | 0.025151   | 22.41x  | -95.54%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.563506 | 0.024052   | 23.43x  | -95.73%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.578786 | 0.208121   | 2.78x   | -64.04%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.578786 | 0.025938   | 22.31x  | -95.52%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.578786 | 0.024275   | 23.84x  | -95.81%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.914000 | 0.335741   | 2.72x   | -63.27%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.914000 | 0.030935   | 29.55x  | -96.62%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.914000 | 0.025097   | 36.42x  | -97.25%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.804711 | 0.204143   | 3.94x   | -74.63%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.804711 | 0.025269   | 31.85x  | -96.86%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.804711 | 0.024162   | 33.31x  | -97.00%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.596982 | 0.201209   | 2.97x   | -66.30%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.596982 | 0.033925   | 17.60x  | -94.32%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.596982 | 0.026546   | 22.49x  | -95.55%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.621070 | 0.247779   | 2.51x   | -60.10%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.621070 | 0.027396   | 22.67x  | -95.59%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.621070 | 0.025233   | 24.61x  | -95.94%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.608897 | 0.245084   | 2.48x   | -59.75%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.608897 | 0.027262   | 22.33x  | -95.52%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.608897 | 0.024904   | 24.45x  | -95.91%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.631874 | 0.204787   | 3.09x   | -67.59%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.631874 | 0.028031   | 22.54x  | -95.56%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.631874 | 0.024960   | 25.32x  | -96.05%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.838448 | 0.326505   | 2.57x   | -61.06%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.838448 | 0.029275   | 28.64x  | -96.51%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.838448 | 0.025350   | 33.08x  | -96.98%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.677016 | 0.074076   | 9.14x   | -89.06%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.677016 | 0.039938   | 16.95x  | -94.10%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.677016 | 0.030355   | 22.30x  | -95.52%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.904196 | 0.093369   | 9.68x   | -89.67%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.904196 | 0.041329   | 21.88x  | -95.43%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.904196 | 0.035328   | 25.59x  | -96.09%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.657110 | 0.106404   | 6.18x   | -83.81%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.657110 | 0.039307   | 16.72x  | -94.02%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.657110 | 0.034426   | 19.09x  | -94.76%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.705962 | 0.086393   | 8.17x   | -87.76%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.705962 | 0.041198   | 17.14x  | -94.16%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.705962 | 0.036512   | 19.34x  | -94.83%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.708598 | 0.092979   | 7.62x   | -86.88%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.708598 | 0.043212   | 16.40x  | -93.90%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.708598 | 0.035567   | 19.92x  | -94.98%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.685239 | 0.213921   | 3.20x   | -68.78%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.685239 | 0.041828   | 16.38x  | -93.90%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.685239 | 0.035225   | 19.45x  | -94.86%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.660198 | 0.105766   | 6.24x   | -83.98%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.660198 | 0.039241   | 16.82x  | -94.06%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.660198 | 0.034722   | 19.01x  | -94.74%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.681861 | 0.081546   | 8.36x   | -88.04%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.681861 | 0.039430   | 17.29x  | -94.22%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.681861 | 0.034585   | 19.72x  | -94.93%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 1.015207 | 0.243124   | 4.18x   | -76.05%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 1.015207 | 0.044563   | 22.78x  | -95.61%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 1.015207 | 0.035661   | 28.47x  | -96.49%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.919986 | 0.084612   | 10.87x  | -90.80%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.919986 | 0.040068   | 22.96x  | -95.64%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.919986 | 0.035428   | 25.97x  | -96.15%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.698299 | 0.103726   | 6.73x   | -85.15%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.698299 | 0.042447   | 16.45x  | -93.92%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.698299 | 0.036740   | 19.01x  | -94.74%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.732233 | 0.143587   | 5.10x   | -80.39%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.732233 | 0.041276   | 17.74x  | -94.36%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.732233 | 0.035649   | 20.54x  | -95.13%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.699970 | 0.144241   | 4.85x   | -79.39%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.699970 | 0.042845   | 16.34x  | -93.88%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.699970 | 0.035785   | 19.56x  | -94.89%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.723007 | 0.091497   | 7.90x   | -87.34%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.723007 | 0.042209   | 17.13x  | -94.16%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.723007 | 0.035689   | 20.26x  | -95.06%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.938730 | 0.226467   | 4.15x   | -75.88%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.938730 | 0.042556   | 22.06x  | -95.47%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.938730 | 0.035269   | 26.62x  | -96.24%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.551262 | 0.092343   | 5.97x   | -83.25%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.551262 | 0.036345   | 15.17x  | -93.41%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.551262 | 0.036193   | 15.23x  | -93.43%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.758930 | 0.107596   | 7.05x   | -85.82%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.758930 | 0.039932   | 19.01x  | -94.74%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.758930 | 0.039460   | 19.23x  | -94.80%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.520565 | 0.116970   | 4.45x   | -77.53%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.520565 | 0.038746   | 13.44x  | -92.56%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.520565 | 0.038287   | 13.60x  | -92.65%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.570831 | 0.103667   | 5.51x   | -81.84%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.570831 | 0.038524   | 14.82x  | -93.25%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.570831 | 0.038421   | 14.86x  | -93.27%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.565439 | 0.113293   | 4.99x   | -79.96%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.565439 | 0.043719   | 12.93x  | -92.27%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.565439 | 0.038383   | 14.73x  | -93.21%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.544106 | 0.189885   | 2.87x   | -65.10%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.544106 | 0.040972   | 13.28x  | -92.47%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.544106 | 0.039706   | 13.70x  | -92.70%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.515835 | 0.117575   | 4.39x   | -77.21%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.515835 | 0.038188   | 13.51x  | -92.60%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.515835 | 0.039228   | 13.15x  | -92.40%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.545692 | 0.100909   | 5.41x   | -81.51%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.545692 | 0.039638   | 13.77x  | -92.74%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.545692 | 0.038513   | 14.17x  | -92.94%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.878250 | 0.251995   | 3.49x   | -71.31%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.878250 | 0.045491   | 19.31x  | -94.82%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.878250 | 0.039114   | 22.45x  | -95.55%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.771986 | 0.102373   | 7.54x   | -86.74%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.771986 | 0.040513   | 19.06x  | -94.75%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.771986 | 0.038865   | 19.86x  | -94.97%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.542228 | 0.129304   | 4.19x   | -76.15%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.542228 | 0.040634   | 13.34x  | -92.51%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.542228 | 0.040035   | 13.54x  | -92.62%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.547634 | 0.150476   | 3.64x   | -72.52%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.547634 | 0.038898   | 14.08x  | -92.90%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.547634 | 0.038227   | 14.33x  | -93.02%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.552752 | 0.150512   | 3.67x   | -72.77%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.552752 | 0.038833   | 14.23x  | -92.97%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.552752 | 0.038202   | 14.47x  | -93.09%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.593610 | 0.109767   | 5.41x   | -81.51%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.593610 | 0.041265   | 14.39x  | -93.05%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.593610 | 0.039065   | 15.20x  | -93.42%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.788703 | 0.239898   | 3.29x   | -69.58%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.788703 | 0.040860   | 19.30x  | -94.82%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.788703 | 0.038547   | 20.46x  | -95.11%        |
</details>
<!-- PERF_RESULTS_END -->
