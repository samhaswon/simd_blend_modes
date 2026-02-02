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

| Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| ------------- | ------ | -------- | ---------- | ------- | -------------- |
| normal        | scalar | 0.182605 | 0.041871   | 4.36x   | -77.07%        |
| normal        | sse42  | 0.182605 | 0.011882   | 15.37x  | -93.49%        |
| normal        | avx2   | 0.182605 | 0.009732   | 18.76x  | -94.67%        |
| soft_light    | scalar | 0.252954 | 0.053357   | 4.74x   | -78.91%        |
| soft_light    | sse42  | 0.252954 | 0.013451   | 18.81x  | -94.68%        |
| soft_light    | avx2   | 0.252954 | 0.010885   | 23.24x  | -95.70%        |
| lighten_only  | scalar | 0.177642 | 0.057645   | 3.08x   | -67.55%        |
| lighten_only  | sse42  | 0.177642 | 0.013625   | 13.04x  | -92.33%        |
| lighten_only  | avx2   | 0.177642 | 0.011146   | 15.94x  | -93.73%        |
| screen        | scalar | 0.195678 | 0.052231   | 3.75x   | -73.31%        |
| screen        | sse42  | 0.195678 | 0.012961   | 15.10x  | -93.38%        |
| screen        | avx2   | 0.195678 | 0.010717   | 18.26x  | -94.52%        |
| dodge         | scalar | 0.197062 | 0.054329   | 3.63x   | -72.43%        |
| dodge         | sse42  | 0.197062 | 0.014062   | 14.01x  | -92.86%        |
| dodge         | avx2   | 0.197062 | 0.011184   | 17.62x  | -94.32%        |
| addition      | scalar | 0.184431 | 0.072055   | 2.56x   | -60.93%        |
| addition      | sse42  | 0.184431 | 0.014458   | 12.76x  | -92.16%        |
| addition      | avx2   | 0.184431 | 0.010918   | 16.89x  | -94.08%        |
| darken_only   | scalar | 0.177120 | 0.055675   | 3.18x   | -68.57%        |
| darken_only   | sse42  | 0.177120 | 0.013798   | 12.84x  | -92.21%        |
| darken_only   | avx2   | 0.177120 | 0.010648   | 16.63x  | -93.99%        |
| multiply      | scalar | 0.181514 | 0.050204   | 3.62x   | -72.34%        |
| multiply      | sse42  | 0.181514 | 0.013347   | 13.60x  | -92.65%        |
| multiply      | avx2   | 0.181514 | 0.011021   | 16.47x  | -93.93%        |
| hard_light    | scalar | 0.290821 | 0.096411   | 3.02x   | -66.85%        |
| hard_light    | sse42  | 0.290821 | 0.014138   | 20.57x  | -95.14%        |
| hard_light    | avx2   | 0.290821 | 0.011452   | 25.39x  | -96.06%        |
| difference    | scalar | 0.256406 | 0.051031   | 5.02x   | -80.10%        |
| difference    | sse42  | 0.256406 | 0.012933   | 19.83x  | -94.96%        |
| difference    | avx2   | 0.256406 | 0.011081   | 23.14x  | -95.68%        |
| subtract      | scalar | 0.198317 | 0.056039   | 3.54x   | -71.74%        |
| subtract      | sse42  | 0.198317 | 0.014909   | 13.30x  | -92.48%        |
| subtract      | avx2   | 0.198317 | 0.011109   | 17.85x  | -94.40%        |
| grain_extract | scalar | 0.189953 | 0.069139   | 2.75x   | -63.60%        |
| grain_extract | sse42  | 0.189953 | 0.013173   | 14.42x  | -93.06%        |
| grain_extract | avx2   | 0.189953 | 0.010809   | 17.57x  | -94.31%        |
| grain_merge   | scalar | 0.194364 | 0.066230   | 2.93x   | -65.92%        |
| grain_merge   | sse42  | 0.194364 | 0.013342   | 14.57x  | -93.14%        |
| grain_merge   | avx2   | 0.194364 | 0.010882   | 17.86x  | -94.40%        |
| divide        | scalar | 0.198578 | 0.053781   | 3.69x   | -72.92%        |
| divide        | sse42  | 0.198578 | 0.013777   | 14.41x  | -93.06%        |
| divide        | avx2   | 0.198578 | 0.011097   | 17.89x  | -94.41%        |
| overlay       | scalar | 0.266443 | 0.093156   | 2.86x   | -65.04%        |
| overlay       | sse42  | 0.266443 | 0.013901   | 19.17x  | -94.78%        |
| overlay       | avx2   | 0.266443 | 0.011400   | 23.37x  | -95.72%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | normal        | scalar | 0.003661 | 0.001402   | 2.61x   | -61.70%        |
| 256x256   | uint8   | normal        | sse42  | 0.003661 | 0.000226   | 16.17x  | -93.81%        |
| 256x256   | uint8   | normal        | avx2   | 0.003661 | 0.000162   | 22.64x  | -95.58%        |
| 256x256   | uint8   | soft_light    | scalar | 0.005125 | 0.001818   | 2.82x   | -64.53%        |
| 256x256   | uint8   | soft_light    | sse42  | 0.005125 | 0.000238   | 21.49x  | -95.35%        |
| 256x256   | uint8   | soft_light    | avx2   | 0.005125 | 0.000193   | 26.49x  | -96.23%        |
| 256x256   | uint8   | lighten_only  | scalar | 0.002859 | 0.001740   | 1.64x   | -39.15%        |
| 256x256   | uint8   | lighten_only  | sse42  | 0.002859 | 0.000222   | 12.90x  | -92.25%        |
| 256x256   | uint8   | lighten_only  | avx2   | 0.002859 | 0.000192   | 14.90x  | -93.29%        |
| 256x256   | uint8   | screen        | scalar | 0.003078 | 0.001628   | 1.89x   | -47.10%        |
| 256x256   | uint8   | screen        | sse42  | 0.003078 | 0.000229   | 13.46x  | -92.57%        |
| 256x256   | uint8   | screen        | avx2   | 0.003078 | 0.000194   | 15.83x  | -93.68%        |
| 256x256   | uint8   | dodge         | scalar | 0.003205 | 0.001787   | 1.79x   | -44.23%        |
| 256x256   | uint8   | dodge         | sse42  | 0.003205 | 0.000252   | 12.70x  | -92.13%        |
| 256x256   | uint8   | dodge         | avx2   | 0.003205 | 0.000199   | 16.11x  | -93.79%        |
| 256x256   | uint8   | addition      | scalar | 0.004704 | 0.002193   | 2.15x   | -53.39%        |
| 256x256   | uint8   | addition      | sse42  | 0.004704 | 0.000277   | 16.99x  | -94.11%        |
| 256x256   | uint8   | addition      | avx2   | 0.004704 | 0.000209   | 22.47x  | -95.55%        |
| 256x256   | uint8   | darken_only   | scalar | 0.003183 | 0.001738   | 1.83x   | -45.39%        |
| 256x256   | uint8   | darken_only   | sse42  | 0.003183 | 0.000219   | 14.52x  | -93.11%        |
| 256x256   | uint8   | darken_only   | avx2   | 0.003183 | 0.000193   | 16.47x  | -93.93%        |
| 256x256   | uint8   | multiply      | scalar | 0.003048 | 0.001665   | 1.83x   | -45.36%        |
| 256x256   | uint8   | multiply      | sse42  | 0.003048 | 0.000239   | 12.76x  | -92.16%        |
| 256x256   | uint8   | multiply      | avx2   | 0.003048 | 0.000192   | 15.87x  | -93.70%        |
| 256x256   | uint8   | hard_light    | scalar | 0.004713 | 0.002733   | 1.72x   | -42.02%        |
| 256x256   | uint8   | hard_light    | sse42  | 0.004713 | 0.000262   | 18.02x  | -94.45%        |
| 256x256   | uint8   | hard_light    | avx2   | 0.004713 | 0.000195   | 24.12x  | -95.85%        |
| 256x256   | uint8   | difference    | scalar | 0.005442 | 0.001751   | 3.11x   | -67.82%        |
| 256x256   | uint8   | difference    | sse42  | 0.005442 | 0.000246   | 22.09x  | -95.47%        |
| 256x256   | uint8   | difference    | avx2   | 0.005442 | 0.000192   | 28.27x  | -96.46%        |
| 256x256   | uint8   | subtract      | scalar | 0.004091 | 0.001677   | 2.44x   | -59.02%        |
| 256x256   | uint8   | subtract      | sse42  | 0.004091 | 0.000280   | 14.59x  | -93.15%        |
| 256x256   | uint8   | subtract      | avx2   | 0.004091 | 0.000204   | 20.04x  | -95.01%        |
| 256x256   | uint8   | grain_extract | scalar | 0.003254 | 0.001945   | 1.67x   | -40.24%        |
| 256x256   | uint8   | grain_extract | sse42  | 0.003254 | 0.000230   | 14.13x  | -92.92%        |
| 256x256   | uint8   | grain_extract | avx2   | 0.003254 | 0.000192   | 16.94x  | -94.10%        |
| 256x256   | uint8   | grain_merge   | scalar | 0.003022 | 0.001986   | 1.52x   | -34.30%        |
| 256x256   | uint8   | grain_merge   | sse42  | 0.003022 | 0.000240   | 12.61x  | -92.07%        |
| 256x256   | uint8   | grain_merge   | avx2   | 0.003022 | 0.000199   | 15.21x  | -93.42%        |
| 256x256   | uint8   | divide        | scalar | 0.003783 | 0.001908   | 1.98x   | -49.58%        |
| 256x256   | uint8   | divide        | sse42  | 0.003783 | 0.000243   | 15.60x  | -93.59%        |
| 256x256   | uint8   | divide        | avx2   | 0.003783 | 0.000197   | 19.19x  | -94.79%        |
| 256x256   | uint8   | overlay       | scalar | 0.004542 | 0.002655   | 1.71x   | -41.54%        |
| 256x256   | uint8   | overlay       | sse42  | 0.004542 | 0.000250   | 18.16x  | -94.49%        |
| 256x256   | uint8   | overlay       | avx2   | 0.004542 | 0.000196   | 23.23x  | -95.69%        |
| 256x256   | float32 | normal        | scalar | 0.002906 | 0.000638   | 4.55x   | -78.03%        |
| 256x256   | float32 | normal        | sse42  | 0.002906 | 0.000313   | 9.30x   | -89.25%        |
| 256x256   | float32 | normal        | avx2   | 0.002906 | 0.000231   | 12.59x  | -92.06%        |
| 256x256   | float32 | soft_light    | scalar | 0.004283 | 0.000851   | 5.03x   | -80.13%        |
| 256x256   | float32 | soft_light    | sse42  | 0.004283 | 0.000369   | 11.60x  | -91.38%        |
| 256x256   | float32 | soft_light    | avx2   | 0.004283 | 0.000306   | 14.01x  | -92.86%        |
| 256x256   | float32 | lighten_only  | scalar | 0.003270 | 0.000936   | 3.49x   | -71.38%        |
| 256x256   | float32 | lighten_only  | sse42  | 0.003270 | 0.000426   | 7.68x   | -86.98%        |
| 256x256   | float32 | lighten_only  | avx2   | 0.003270 | 0.000307   | 10.65x  | -90.61%        |
| 256x256   | float32 | screen        | scalar | 0.002928 | 0.000789   | 3.71x   | -73.06%        |
| 256x256   | float32 | screen        | sse42  | 0.002928 | 0.000382   | 7.67x   | -86.97%        |
| 256x256   | float32 | screen        | avx2   | 0.002928 | 0.000265   | 11.06x  | -90.96%        |
| 256x256   | float32 | dodge         | scalar | 0.002987 | 0.000843   | 3.54x   | -71.77%        |
| 256x256   | float32 | dodge         | sse42  | 0.002987 | 0.000367   | 8.15x   | -87.72%        |
| 256x256   | float32 | dodge         | avx2   | 0.002987 | 0.000261   | 11.46x  | -91.27%        |
| 256x256   | float32 | addition      | scalar | 0.004167 | 0.001464   | 2.85x   | -64.86%        |
| 256x256   | float32 | addition      | sse42  | 0.004167 | 0.000390   | 10.68x  | -90.63%        |
| 256x256   | float32 | addition      | avx2   | 0.004167 | 0.000266   | 15.65x  | -93.61%        |
| 256x256   | float32 | darken_only   | scalar | 0.003236 | 0.001013   | 3.19x   | -68.69%        |
| 256x256   | float32 | darken_only   | sse42  | 0.003236 | 0.000385   | 8.40x   | -88.10%        |
| 256x256   | float32 | darken_only   | avx2   | 0.003236 | 0.000267   | 12.12x  | -91.75%        |
| 256x256   | float32 | multiply      | scalar | 0.003052 | 0.000716   | 4.26x   | -76.53%        |
| 256x256   | float32 | multiply      | sse42  | 0.003052 | 0.000371   | 8.23x   | -87.85%        |
| 256x256   | float32 | multiply      | avx2   | 0.003052 | 0.000262   | 11.64x  | -91.41%        |
| 256x256   | float32 | hard_light    | scalar | 0.004707 | 0.001939   | 2.43x   | -58.82%        |
| 256x256   | float32 | hard_light    | sse42  | 0.004707 | 0.000359   | 13.12x  | -92.38%        |
| 256x256   | float32 | hard_light    | avx2   | 0.004707 | 0.000269   | 17.50x  | -94.29%        |
| 256x256   | float32 | difference    | scalar | 0.004881 | 0.000734   | 6.65x   | -84.97%        |
| 256x256   | float32 | difference    | sse42  | 0.004881 | 0.000370   | 13.19x  | -92.42%        |
| 256x256   | float32 | difference    | avx2   | 0.004881 | 0.000264   | 18.46x  | -94.58%        |
| 256x256   | float32 | subtract      | scalar | 0.004106 | 0.001089   | 3.77x   | -73.47%        |
| 256x256   | float32 | subtract      | sse42  | 0.004106 | 0.000714   | 5.75x   | -82.60%        |
| 256x256   | float32 | subtract      | avx2   | 0.004106 | 0.000328   | 12.51x  | -92.01%        |
| 256x256   | float32 | grain_extract | scalar | 0.003435 | 0.001163   | 2.95x   | -66.13%        |
| 256x256   | float32 | grain_extract | sse42  | 0.003435 | 0.000363   | 9.46x   | -89.43%        |
| 256x256   | float32 | grain_extract | avx2   | 0.003435 | 0.000270   | 12.74x  | -92.15%        |
| 256x256   | float32 | grain_merge   | scalar | 0.002893 | 0.001316   | 2.20x   | -54.51%        |
| 256x256   | float32 | grain_merge   | sse42  | 0.002893 | 0.000367   | 7.89x   | -87.32%        |
| 256x256   | float32 | grain_merge   | avx2   | 0.002893 | 0.000260   | 11.12x  | -91.01%        |
| 256x256   | float32 | divide        | scalar | 0.003339 | 0.000855   | 3.90x   | -74.38%        |
| 256x256   | float32 | divide        | sse42  | 0.003339 | 0.000373   | 8.96x   | -88.83%        |
| 256x256   | float32 | divide        | avx2   | 0.003339 | 0.000309   | 10.80x  | -90.74%        |
| 256x256   | float32 | overlay       | scalar | 0.004617 | 0.001860   | 2.48x   | -59.72%        |
| 256x256   | float32 | overlay       | sse42  | 0.004617 | 0.000354   | 13.06x  | -92.34%        |
| 256x256   | float32 | overlay       | avx2   | 0.004617 | 0.000257   | 17.98x  | -94.44%        |
| 512x512   | uint8   | normal        | scalar | 0.024506 | 0.005376   | 4.56x   | -78.06%        |
| 512x512   | uint8   | normal        | sse42  | 0.024506 | 0.000777   | 31.54x  | -96.83%        |
| 512x512   | uint8   | normal        | avx2   | 0.024506 | 0.000654   | 37.47x  | -97.33%        |
| 512x512   | uint8   | soft_light    | scalar | 0.036493 | 0.007977   | 4.57x   | -78.14%        |
| 512x512   | uint8   | soft_light    | sse42  | 0.036493 | 0.001004   | 36.34x  | -97.25%        |
| 512x512   | uint8   | soft_light    | avx2   | 0.036493 | 0.000772   | 47.29x  | -97.89%        |
| 512x512   | uint8   | lighten_only  | scalar | 0.033703 | 0.009848   | 3.42x   | -70.78%        |
| 512x512   | uint8   | lighten_only  | sse42  | 0.033703 | 0.000913   | 36.91x  | -97.29%        |
| 512x512   | uint8   | lighten_only  | avx2   | 0.033703 | 0.000791   | 42.62x  | -97.65%        |
| 512x512   | uint8   | screen        | scalar | 0.030170 | 0.007288   | 4.14x   | -75.84%        |
| 512x512   | uint8   | screen        | sse42  | 0.030170 | 0.001007   | 29.96x  | -96.66%        |
| 512x512   | uint8   | screen        | avx2   | 0.030170 | 0.000765   | 39.44x  | -97.46%        |
| 512x512   | uint8   | dodge         | scalar | 0.031417 | 0.007020   | 4.48x   | -77.66%        |
| 512x512   | uint8   | dodge         | sse42  | 0.031417 | 0.001062   | 29.58x  | -96.62%        |
| 512x512   | uint8   | dodge         | avx2   | 0.031417 | 0.000806   | 39.00x  | -97.44%        |
| 512x512   | uint8   | addition      | scalar | 0.028176 | 0.008633   | 3.26x   | -69.36%        |
| 512x512   | uint8   | addition      | sse42  | 0.028176 | 0.001125   | 25.05x  | -96.01%        |
| 512x512   | uint8   | addition      | avx2   | 0.028176 | 0.000793   | 35.54x  | -97.19%        |
| 512x512   | uint8   | darken_only   | scalar | 0.028970 | 0.006946   | 4.17x   | -76.02%        |
| 512x512   | uint8   | darken_only   | sse42  | 0.028970 | 0.000896   | 32.32x  | -96.91%        |
| 512x512   | uint8   | darken_only   | avx2   | 0.028970 | 0.000753   | 38.48x  | -97.40%        |
| 512x512   | uint8   | multiply      | scalar | 0.028280 | 0.007162   | 3.95x   | -74.68%        |
| 512x512   | uint8   | multiply      | sse42  | 0.028280 | 0.000922   | 30.67x  | -96.74%        |
| 512x512   | uint8   | multiply      | avx2   | 0.028280 | 0.000787   | 35.96x  | -97.22%        |
| 512x512   | uint8   | hard_light    | scalar | 0.038027 | 0.012171   | 3.12x   | -67.99%        |
| 512x512   | uint8   | hard_light    | sse42  | 0.038027 | 0.001063   | 35.78x  | -97.21%        |
| 512x512   | uint8   | hard_light    | avx2   | 0.038027 | 0.000781   | 48.71x  | -97.95%        |
| 512x512   | uint8   | difference    | scalar | 0.037317 | 0.006807   | 5.48x   | -81.76%        |
| 512x512   | uint8   | difference    | sse42  | 0.037317 | 0.000889   | 41.98x  | -97.62%        |
| 512x512   | uint8   | difference    | avx2   | 0.037317 | 0.000761   | 49.01x  | -97.96%        |
| 512x512   | uint8   | subtract      | scalar | 0.028939 | 0.006624   | 4.37x   | -77.11%        |
| 512x512   | uint8   | subtract      | sse42  | 0.028939 | 0.001102   | 26.25x  | -96.19%        |
| 512x512   | uint8   | subtract      | avx2   | 0.028939 | 0.000820   | 35.27x  | -97.17%        |
| 512x512   | uint8   | grain_extract | scalar | 0.028545 | 0.008131   | 3.51x   | -71.52%        |
| 512x512   | uint8   | grain_extract | sse42  | 0.028545 | 0.000922   | 30.96x  | -96.77%        |
| 512x512   | uint8   | grain_extract | avx2   | 0.028545 | 0.000758   | 37.68x  | -97.35%        |
| 512x512   | uint8   | grain_merge   | scalar | 0.028097 | 0.008259   | 3.40x   | -70.60%        |
| 512x512   | uint8   | grain_merge   | sse42  | 0.028097 | 0.000946   | 29.71x  | -96.63%        |
| 512x512   | uint8   | grain_merge   | avx2   | 0.028097 | 0.000773   | 36.36x  | -97.25%        |
| 512x512   | uint8   | divide        | scalar | 0.029915 | 0.006900   | 4.34x   | -76.93%        |
| 512x512   | uint8   | divide        | sse42  | 0.029915 | 0.000955   | 31.32x  | -96.81%        |
| 512x512   | uint8   | divide        | avx2   | 0.029915 | 0.000767   | 39.00x  | -97.44%        |
| 512x512   | uint8   | overlay       | scalar | 0.035415 | 0.010901   | 3.25x   | -69.22%        |
| 512x512   | uint8   | overlay       | sse42  | 0.035415 | 0.000998   | 35.50x  | -97.18%        |
| 512x512   | uint8   | overlay       | avx2   | 0.035415 | 0.000857   | 41.35x  | -97.58%        |
| 512x512   | float32 | normal        | scalar | 0.021083 | 0.002566   | 8.22x   | -87.83%        |
| 512x512   | float32 | normal        | sse42  | 0.021083 | 0.001261   | 16.72x  | -94.02%        |
| 512x512   | float32 | normal        | avx2   | 0.021083 | 0.001005   | 20.97x  | -95.23%        |
| 512x512   | float32 | soft_light    | scalar | 0.034075 | 0.003310   | 10.29x  | -90.29%        |
| 512x512   | float32 | soft_light    | sse42  | 0.034075 | 0.001521   | 22.41x  | -95.54%        |
| 512x512   | float32 | soft_light    | avx2   | 0.034075 | 0.001089   | 31.28x  | -96.80%        |
| 512x512   | float32 | lighten_only  | scalar | 0.025765 | 0.003803   | 6.77x   | -85.24%        |
| 512x512   | float32 | lighten_only  | sse42  | 0.025765 | 0.001574   | 16.37x  | -93.89%        |
| 512x512   | float32 | lighten_only  | avx2   | 0.025765 | 0.001179   | 21.86x  | -95.42%        |
| 512x512   | float32 | screen        | scalar | 0.028401 | 0.003351   | 8.47x   | -88.20%        |
| 512x512   | float32 | screen        | sse42  | 0.028401 | 0.001495   | 19.00x  | -94.74%        |
| 512x512   | float32 | screen        | avx2   | 0.028401 | 0.001105   | 25.70x  | -96.11%        |
| 512x512   | float32 | dodge         | scalar | 0.027563 | 0.003382   | 8.15x   | -87.73%        |
| 512x512   | float32 | dodge         | sse42  | 0.027563 | 0.001533   | 17.98x  | -94.44%        |
| 512x512   | float32 | dodge         | avx2   | 0.027563 | 0.001156   | 23.84x  | -95.81%        |
| 512x512   | float32 | addition      | scalar | 0.025976 | 0.005794   | 4.48x   | -77.70%        |
| 512x512   | float32 | addition      | sse42  | 0.025976 | 0.001612   | 16.12x  | -93.79%        |
| 512x512   | float32 | addition      | avx2   | 0.025976 | 0.001124   | 23.11x  | -95.67%        |
| 512x512   | float32 | darken_only   | scalar | 0.027477 | 0.003665   | 7.50x   | -86.66%        |
| 512x512   | float32 | darken_only   | sse42  | 0.027477 | 0.001516   | 18.13x  | -94.48%        |
| 512x512   | float32 | darken_only   | avx2   | 0.027477 | 0.001110   | 24.77x  | -95.96%        |
| 512x512   | float32 | multiply      | scalar | 0.025787 | 0.002883   | 8.94x   | -88.82%        |
| 512x512   | float32 | multiply      | sse42  | 0.025787 | 0.001549   | 16.65x  | -93.99%        |
| 512x512   | float32 | multiply      | avx2   | 0.025787 | 0.001080   | 23.88x  | -95.81%        |
| 512x512   | float32 | hard_light    | scalar | 0.036920 | 0.007796   | 4.74x   | -78.88%        |
| 512x512   | float32 | hard_light    | sse42  | 0.036920 | 0.001433   | 25.77x  | -96.12%        |
| 512x512   | float32 | hard_light    | avx2   | 0.036920 | 0.001087   | 33.97x  | -97.06%        |
| 512x512   | float32 | difference    | scalar | 0.034991 | 0.003058   | 11.44x  | -91.26%        |
| 512x512   | float32 | difference    | sse42  | 0.034991 | 0.001531   | 22.86x  | -95.63%        |
| 512x512   | float32 | difference    | avx2   | 0.034991 | 0.001087   | 32.20x  | -96.89%        |
| 512x512   | float32 | subtract      | scalar | 0.025225 | 0.004161   | 6.06x   | -83.50%        |
| 512x512   | float32 | subtract      | sse42  | 0.025225 | 0.001598   | 15.78x  | -93.66%        |
| 512x512   | float32 | subtract      | avx2   | 0.025225 | 0.001251   | 20.17x  | -95.04%        |
| 512x512   | float32 | grain_extract | scalar | 0.028864 | 0.004608   | 6.26x   | -84.04%        |
| 512x512   | float32 | grain_extract | sse42  | 0.028864 | 0.001487   | 19.41x  | -94.85%        |
| 512x512   | float32 | grain_extract | avx2   | 0.028864 | 0.001087   | 26.54x  | -96.23%        |
| 512x512   | float32 | grain_merge   | scalar | 0.027855 | 0.004642   | 6.00x   | -83.34%        |
| 512x512   | float32 | grain_merge   | sse42  | 0.027855 | 0.001525   | 18.27x  | -94.53%        |
| 512x512   | float32 | grain_merge   | avx2   | 0.027855 | 0.001183   | 23.55x  | -95.75%        |
| 512x512   | float32 | divide        | scalar | 0.027038 | 0.003251   | 8.32x   | -87.98%        |
| 512x512   | float32 | divide        | sse42  | 0.027038 | 0.001505   | 17.97x  | -94.44%        |
| 512x512   | float32 | divide        | avx2   | 0.027038 | 0.001112   | 24.31x  | -95.89%        |
| 512x512   | float32 | overlay       | scalar | 0.035042 | 0.007811   | 4.49x   | -77.71%        |
| 512x512   | float32 | overlay       | sse42  | 0.035042 | 0.001405   | 24.94x  | -95.99%        |
| 512x512   | float32 | overlay       | avx2   | 0.035042 | 0.001049   | 33.42x  | -97.01%        |
| 1024x1024 | uint8   | normal        | scalar | 0.077235 | 0.021641   | 3.57x   | -71.98%        |
| 1024x1024 | uint8   | normal        | sse42  | 0.077235 | 0.003114   | 24.80x  | -95.97%        |
| 1024x1024 | uint8   | normal        | avx2   | 0.077235 | 0.002874   | 26.87x  | -96.28%        |
| 1024x1024 | uint8   | soft_light    | scalar | 0.111686 | 0.034872   | 3.20x   | -68.78%        |
| 1024x1024 | uint8   | soft_light    | sse42  | 0.111686 | 0.003811   | 29.31x  | -96.59%        |
| 1024x1024 | uint8   | soft_light    | avx2   | 0.111686 | 0.003083   | 36.23x  | -97.24%        |
| 1024x1024 | uint8   | lighten_only  | scalar | 0.082009 | 0.028339   | 2.89x   | -65.44%        |
| 1024x1024 | uint8   | lighten_only  | sse42  | 0.082009 | 0.003600   | 22.78x  | -95.61%        |
| 1024x1024 | uint8   | lighten_only  | avx2   | 0.082009 | 0.003024   | 27.12x  | -96.31%        |
| 1024x1024 | uint8   | screen        | scalar | 0.085439 | 0.027824   | 3.07x   | -67.43%        |
| 1024x1024 | uint8   | screen        | sse42  | 0.085439 | 0.003663   | 23.32x  | -95.71%        |
| 1024x1024 | uint8   | screen        | avx2   | 0.085439 | 0.003064   | 27.89x  | -96.41%        |
| 1024x1024 | uint8   | dodge         | scalar | 0.086055 | 0.028393   | 3.03x   | -67.01%        |
| 1024x1024 | uint8   | dodge         | sse42  | 0.086055 | 0.004078   | 21.10x  | -95.26%        |
| 1024x1024 | uint8   | dodge         | avx2   | 0.086055 | 0.003220   | 26.72x  | -96.26%        |
| 1024x1024 | uint8   | addition      | scalar | 0.083776 | 0.034122   | 2.46x   | -59.27%        |
| 1024x1024 | uint8   | addition      | sse42  | 0.083776 | 0.004417   | 18.97x  | -94.73%        |
| 1024x1024 | uint8   | addition      | avx2   | 0.083776 | 0.003215   | 26.06x  | -96.16%        |
| 1024x1024 | uint8   | darken_only   | scalar | 0.082962 | 0.028585   | 2.90x   | -65.54%        |
| 1024x1024 | uint8   | darken_only   | sse42  | 0.082962 | 0.003535   | 23.47x  | -95.74%        |
| 1024x1024 | uint8   | darken_only   | avx2   | 0.082962 | 0.003053   | 27.18x  | -96.32%        |
| 1024x1024 | uint8   | multiply      | scalar | 0.083445 | 0.027853   | 3.00x   | -66.62%        |
| 1024x1024 | uint8   | multiply      | sse42  | 0.083445 | 0.003558   | 23.45x  | -95.74%        |
| 1024x1024 | uint8   | multiply      | avx2   | 0.083445 | 0.003105   | 26.88x  | -96.28%        |
| 1024x1024 | uint8   | hard_light    | scalar | 0.121351 | 0.044491   | 2.73x   | -63.34%        |
| 1024x1024 | uint8   | hard_light    | sse42  | 0.121351 | 0.004289   | 28.29x  | -96.47%        |
| 1024x1024 | uint8   | hard_light    | avx2   | 0.121351 | 0.003126   | 38.82x  | -97.42%        |
| 1024x1024 | uint8   | difference    | scalar | 0.117757 | 0.028818   | 4.09x   | -75.53%        |
| 1024x1024 | uint8   | difference    | sse42  | 0.117757 | 0.003618   | 32.55x  | -96.93%        |
| 1024x1024 | uint8   | difference    | avx2   | 0.117757 | 0.003019   | 39.00x  | -97.44%        |
| 1024x1024 | uint8   | subtract      | scalar | 0.084949 | 0.029240   | 2.91x   | -65.58%        |
| 1024x1024 | uint8   | subtract      | sse42  | 0.084949 | 0.004646   | 18.28x  | -94.53%        |
| 1024x1024 | uint8   | subtract      | avx2   | 0.084949 | 0.003366   | 25.24x  | -96.04%        |
| 1024x1024 | uint8   | grain_extract | scalar | 0.087088 | 0.033130   | 2.63x   | -61.96%        |
| 1024x1024 | uint8   | grain_extract | sse42  | 0.087088 | 0.003732   | 23.34x  | -95.71%        |
| 1024x1024 | uint8   | grain_extract | avx2   | 0.087088 | 0.003110   | 28.00x  | -96.43%        |
| 1024x1024 | uint8   | grain_merge   | scalar | 0.081859 | 0.033195   | 2.47x   | -59.45%        |
| 1024x1024 | uint8   | grain_merge   | sse42  | 0.081859 | 0.003684   | 22.22x  | -95.50%        |
| 1024x1024 | uint8   | grain_merge   | avx2   | 0.081859 | 0.003114   | 26.29x  | -96.20%        |
| 1024x1024 | uint8   | divide        | scalar | 0.086951 | 0.030504   | 2.85x   | -64.92%        |
| 1024x1024 | uint8   | divide        | sse42  | 0.086951 | 0.003909   | 22.24x  | -95.50%        |
| 1024x1024 | uint8   | divide        | avx2   | 0.086951 | 0.003303   | 26.32x  | -96.20%        |
| 1024x1024 | uint8   | overlay       | scalar | 0.110272 | 0.043442   | 2.54x   | -60.60%        |
| 1024x1024 | uint8   | overlay       | sse42  | 0.110272 | 0.004081   | 27.02x  | -96.30%        |
| 1024x1024 | uint8   | overlay       | avx2   | 0.110272 | 0.003103   | 35.54x  | -97.19%        |
| 1024x1024 | float32 | normal        | scalar | 0.065527 | 0.010865   | 6.03x   | -83.42%        |
| 1024x1024 | float32 | normal        | sse42  | 0.065527 | 0.004940   | 13.27x  | -92.46%        |
| 1024x1024 | float32 | normal        | avx2   | 0.065527 | 0.003914   | 16.74x  | -94.03%        |
| 1024x1024 | float32 | soft_light    | scalar | 0.100829 | 0.013688   | 7.37x   | -86.42%        |
| 1024x1024 | float32 | soft_light    | sse42  | 0.100829 | 0.005802   | 17.38x  | -94.25%        |
| 1024x1024 | float32 | soft_light    | avx2   | 0.100829 | 0.004280   | 23.56x  | -95.76%        |
| 1024x1024 | float32 | lighten_only  | scalar | 0.070850 | 0.014978   | 4.73x   | -78.86%        |
| 1024x1024 | float32 | lighten_only  | sse42  | 0.070850 | 0.005925   | 11.96x  | -91.64%        |
| 1024x1024 | float32 | lighten_only  | avx2   | 0.070850 | 0.004358   | 16.26x  | -93.85%        |
| 1024x1024 | float32 | screen        | scalar | 0.077777 | 0.013608   | 5.72x   | -82.50%        |
| 1024x1024 | float32 | screen        | sse42  | 0.077777 | 0.005823   | 13.36x  | -92.51%        |
| 1024x1024 | float32 | screen        | avx2   | 0.077777 | 0.004327   | 17.97x  | -94.44%        |
| 1024x1024 | float32 | dodge         | scalar | 0.077828 | 0.014311   | 5.44x   | -81.61%        |
| 1024x1024 | float32 | dodge         | sse42  | 0.077828 | 0.005924   | 13.14x  | -92.39%        |
| 1024x1024 | float32 | dodge         | avx2   | 0.077828 | 0.004430   | 17.57x  | -94.31%        |
| 1024x1024 | float32 | addition      | scalar | 0.073472 | 0.023699   | 3.10x   | -67.74%        |
| 1024x1024 | float32 | addition      | sse42  | 0.073472 | 0.006229   | 11.80x  | -91.52%        |
| 1024x1024 | float32 | addition      | avx2   | 0.073472 | 0.004817   | 15.25x  | -93.44%        |
| 1024x1024 | float32 | darken_only   | scalar | 0.071543 | 0.015233   | 4.70x   | -78.71%        |
| 1024x1024 | float32 | darken_only   | sse42  | 0.071543 | 0.005922   | 12.08x  | -91.72%        |
| 1024x1024 | float32 | darken_only   | avx2   | 0.071543 | 0.004343   | 16.47x  | -93.93%        |
| 1024x1024 | float32 | multiply      | scalar | 0.075408 | 0.012225   | 6.17x   | -83.79%        |
| 1024x1024 | float32 | multiply      | sse42  | 0.075408 | 0.006016   | 12.53x  | -92.02%        |
| 1024x1024 | float32 | multiply      | avx2   | 0.075408 | 0.004978   | 15.15x  | -93.40%        |
| 1024x1024 | float32 | hard_light    | scalar | 0.109080 | 0.031441   | 3.47x   | -71.18%        |
| 1024x1024 | float32 | hard_light    | sse42  | 0.109080 | 0.005686   | 19.18x  | -94.79%        |
| 1024x1024 | float32 | hard_light    | avx2   | 0.109080 | 0.004338   | 25.15x  | -96.02%        |
| 1024x1024 | float32 | difference    | scalar | 0.101896 | 0.012181   | 8.37x   | -88.05%        |
| 1024x1024 | float32 | difference    | sse42  | 0.101896 | 0.005798   | 17.58x  | -94.31%        |
| 1024x1024 | float32 | difference    | avx2   | 0.101896 | 0.004189   | 24.33x  | -95.89%        |
| 1024x1024 | float32 | subtract      | scalar | 0.071114 | 0.016199   | 4.39x   | -77.22%        |
| 1024x1024 | float32 | subtract      | sse42  | 0.071114 | 0.006166   | 11.53x  | -91.33%        |
| 1024x1024 | float32 | subtract      | avx2   | 0.071114 | 0.005028   | 14.14x  | -92.93%        |
| 1024x1024 | float32 | grain_extract | scalar | 0.074568 | 0.019034   | 3.92x   | -74.47%        |
| 1024x1024 | float32 | grain_extract | sse42  | 0.074568 | 0.005850   | 12.75x  | -92.15%        |
| 1024x1024 | float32 | grain_extract | avx2   | 0.074568 | 0.004409   | 16.91x  | -94.09%        |
| 1024x1024 | float32 | grain_merge   | scalar | 0.075562 | 0.019411   | 3.89x   | -74.31%        |
| 1024x1024 | float32 | grain_merge   | sse42  | 0.075562 | 0.005822   | 12.98x  | -92.30%        |
| 1024x1024 | float32 | grain_merge   | avx2   | 0.075562 | 0.004317   | 17.50x  | -94.29%        |
| 1024x1024 | float32 | divide        | scalar | 0.076990 | 0.014162   | 5.44x   | -81.61%        |
| 1024x1024 | float32 | divide        | sse42  | 0.076990 | 0.005941   | 12.96x  | -92.28%        |
| 1024x1024 | float32 | divide        | avx2   | 0.076990 | 0.004612   | 16.69x  | -94.01%        |
| 1024x1024 | float32 | overlay       | scalar | 0.101769 | 0.029989   | 3.39x   | -70.53%        |
| 1024x1024 | float32 | overlay       | sse42  | 0.101769 | 0.005688   | 17.89x  | -94.41%        |
| 1024x1024 | float32 | overlay       | avx2   | 0.101769 | 0.004450   | 22.87x  | -95.63%        |
| 2048x2048 | uint8   | normal        | scalar | 0.303609 | 0.088250   | 3.44x   | -70.93%        |
| 2048x2048 | uint8   | normal        | sse42  | 0.303609 | 0.012831   | 23.66x  | -95.77%        |
| 2048x2048 | uint8   | normal        | avx2   | 0.303609 | 0.009929   | 30.58x  | -96.73%        |
| 2048x2048 | uint8   | soft_light    | scalar | 0.465309 | 0.122280   | 3.81x   | -73.72%        |
| 2048x2048 | uint8   | soft_light    | sse42  | 0.465309 | 0.015396   | 30.22x  | -96.69%        |
| 2048x2048 | uint8   | soft_light    | avx2   | 0.465309 | 0.012341   | 37.70x  | -97.35%        |
| 2048x2048 | uint8   | lighten_only  | scalar | 0.325531 | 0.110993   | 2.93x   | -65.90%        |
| 2048x2048 | uint8   | lighten_only  | sse42  | 0.325531 | 0.014196   | 22.93x  | -95.64%        |
| 2048x2048 | uint8   | lighten_only  | avx2   | 0.325531 | 0.012415   | 26.22x  | -96.19%        |
| 2048x2048 | uint8   | screen        | scalar | 0.355823 | 0.123478   | 2.88x   | -65.30%        |
| 2048x2048 | uint8   | screen        | sse42  | 0.355823 | 0.015521   | 22.93x  | -95.64%        |
| 2048x2048 | uint8   | screen        | avx2   | 0.355823 | 0.012368   | 28.77x  | -96.52%        |
| 2048x2048 | uint8   | dodge         | scalar | 0.344005 | 0.109831   | 3.13x   | -68.07%        |
| 2048x2048 | uint8   | dodge         | sse42  | 0.344005 | 0.017101   | 20.12x  | -95.03%        |
| 2048x2048 | uint8   | dodge         | avx2   | 0.344005 | 0.013427   | 25.62x  | -96.10%        |
| 2048x2048 | uint8   | addition      | scalar | 0.333962 | 0.139731   | 2.39x   | -58.16%        |
| 2048x2048 | uint8   | addition      | sse42  | 0.333962 | 0.017774   | 18.79x  | -94.68%        |
| 2048x2048 | uint8   | addition      | avx2   | 0.333962 | 0.012855   | 25.98x  | -96.15%        |
| 2048x2048 | uint8   | darken_only   | scalar | 0.284072 | 0.112708   | 2.52x   | -60.32%        |
| 2048x2048 | uint8   | darken_only   | sse42  | 0.284072 | 0.014066   | 20.20x  | -95.05%        |
| 2048x2048 | uint8   | darken_only   | avx2   | 0.284072 | 0.012659   | 22.44x  | -95.54%        |
| 2048x2048 | uint8   | multiply      | scalar | 0.289462 | 0.110137   | 2.63x   | -61.95%        |
| 2048x2048 | uint8   | multiply      | sse42  | 0.289462 | 0.014046   | 20.61x  | -95.15%        |
| 2048x2048 | uint8   | multiply      | avx2   | 0.289462 | 0.011977   | 24.17x  | -95.86%        |
| 2048x2048 | uint8   | hard_light    | scalar | 0.527752 | 0.186464   | 2.83x   | -64.67%        |
| 2048x2048 | uint8   | hard_light    | sse42  | 0.527752 | 0.016865   | 31.29x  | -96.80%        |
| 2048x2048 | uint8   | hard_light    | avx2   | 0.527752 | 0.012442   | 42.42x  | -97.64%        |
| 2048x2048 | uint8   | difference    | scalar | 0.433408 | 0.108038   | 4.01x   | -75.07%        |
| 2048x2048 | uint8   | difference    | sse42  | 0.433408 | 0.014160   | 30.61x  | -96.73%        |
| 2048x2048 | uint8   | difference    | avx2   | 0.433408 | 0.012065   | 35.92x  | -97.22%        |
| 2048x2048 | uint8   | subtract      | scalar | 0.469278 | 0.112361   | 4.18x   | -76.06%        |
| 2048x2048 | uint8   | subtract      | sse42  | 0.469278 | 0.018212   | 25.77x  | -96.12%        |
| 2048x2048 | uint8   | subtract      | avx2   | 0.469278 | 0.012887   | 36.41x  | -97.25%        |
| 2048x2048 | uint8   | grain_extract | scalar | 0.321108 | 0.191520   | 1.68x   | -40.36%        |
| 2048x2048 | uint8   | grain_extract | sse42  | 0.321108 | 0.015041   | 21.35x  | -95.32%        |
| 2048x2048 | uint8   | grain_extract | avx2   | 0.321108 | 0.012808   | 25.07x  | -96.01%        |
| 2048x2048 | uint8   | grain_merge   | scalar | 0.350654 | 0.135757   | 2.58x   | -61.28%        |
| 2048x2048 | uint8   | grain_merge   | sse42  | 0.350654 | 0.015700   | 22.34x  | -95.52%        |
| 2048x2048 | uint8   | grain_merge   | avx2   | 0.350654 | 0.012815   | 27.36x  | -96.35%        |
| 2048x2048 | uint8   | divide        | scalar | 0.385103 | 0.116700   | 3.30x   | -69.70%        |
| 2048x2048 | uint8   | divide        | sse42  | 0.385103 | 0.015592   | 24.70x  | -95.95%        |
| 2048x2048 | uint8   | divide        | avx2   | 0.385103 | 0.013594   | 28.33x  | -96.47%        |
| 2048x2048 | uint8   | overlay       | scalar | 0.454401 | 0.172927   | 2.63x   | -61.94%        |
| 2048x2048 | uint8   | overlay       | sse42  | 0.454401 | 0.016067   | 28.28x  | -96.46%        |
| 2048x2048 | uint8   | overlay       | avx2   | 0.454401 | 0.013434   | 33.83x  | -97.04%        |
| 2048x2048 | float32 | normal        | scalar | 0.274699 | 0.048135   | 5.71x   | -82.48%        |
| 2048x2048 | float32 | normal        | sse42  | 0.274699 | 0.028008   | 9.81x   | -89.80%        |
| 2048x2048 | float32 | normal        | avx2   | 0.274699 | 0.022933   | 11.98x  | -91.65%        |
| 2048x2048 | float32 | soft_light    | scalar | 0.384017 | 0.062259   | 6.17x   | -83.79%        |
| 2048x2048 | float32 | soft_light    | sse42  | 0.384017 | 0.029940   | 12.83x  | -92.20%        |
| 2048x2048 | float32 | soft_light    | avx2   | 0.384017 | 0.025810   | 14.88x  | -93.28%        |
| 2048x2048 | float32 | lighten_only  | scalar | 0.273407 | 0.085831   | 3.19x   | -68.61%        |
| 2048x2048 | float32 | lighten_only  | sse42  | 0.273407 | 0.030704   | 8.90x   | -88.77%        |
| 2048x2048 | float32 | lighten_only  | avx2   | 0.273407 | 0.025389   | 10.77x  | -90.71%        |
| 2048x2048 | float32 | screen        | scalar | 0.323815 | 0.063898   | 5.07x   | -80.27%        |
| 2048x2048 | float32 | screen        | sse42  | 0.323815 | 0.031845   | 10.17x  | -90.17%        |
| 2048x2048 | float32 | screen        | avx2   | 0.323815 | 0.028525   | 11.35x  | -91.19%        |
| 2048x2048 | float32 | dodge         | scalar | 0.320575 | 0.063445   | 5.05x   | -80.21%        |
| 2048x2048 | float32 | dodge         | sse42  | 0.320575 | 0.032766   | 9.78x   | -89.78%        |
| 2048x2048 | float32 | dodge         | avx2   | 0.320575 | 0.026216   | 12.23x  | -91.82%        |
| 2048x2048 | float32 | addition      | scalar | 0.275482 | 0.100947   | 2.73x   | -63.36%        |
| 2048x2048 | float32 | addition      | sse42  | 0.275482 | 0.034533   | 7.98x   | -87.46%        |
| 2048x2048 | float32 | addition      | avx2   | 0.275482 | 0.026612   | 10.35x  | -90.34%        |
| 2048x2048 | float32 | darken_only   | scalar | 0.298802 | 0.068697   | 4.35x   | -77.01%        |
| 2048x2048 | float32 | darken_only   | sse42  | 0.298802 | 0.031892   | 9.37x   | -89.33%        |
| 2048x2048 | float32 | darken_only   | avx2   | 0.298802 | 0.025449   | 11.74x  | -91.48%        |
| 2048x2048 | float32 | multiply      | scalar | 0.304767 | 0.057218   | 5.33x   | -81.23%        |
| 2048x2048 | float32 | multiply      | sse42  | 0.304767 | 0.032533   | 9.37x   | -89.33%        |
| 2048x2048 | float32 | multiply      | avx2   | 0.304767 | 0.025262   | 12.06x  | -91.71%        |
| 2048x2048 | float32 | hard_light    | scalar | 0.449448 | 0.133464   | 3.37x   | -70.30%        |
| 2048x2048 | float32 | hard_light    | sse42  | 0.449448 | 0.029747   | 15.11x  | -93.38%        |
| 2048x2048 | float32 | hard_light    | avx2   | 0.449448 | 0.025664   | 17.51x  | -94.29%        |
| 2048x2048 | float32 | difference    | scalar | 0.397206 | 0.058288   | 6.81x   | -85.33%        |
| 2048x2048 | float32 | difference    | sse42  | 0.397206 | 0.031494   | 12.61x  | -92.07%        |
| 2048x2048 | float32 | difference    | avx2   | 0.397206 | 0.028332   | 14.02x  | -92.87%        |
| 2048x2048 | float32 | subtract      | scalar | 0.287251 | 0.070810   | 4.06x   | -75.35%        |
| 2048x2048 | float32 | subtract      | sse42  | 0.287251 | 0.032393   | 8.87x   | -88.72%        |
| 2048x2048 | float32 | subtract      | avx2   | 0.287251 | 0.025503   | 11.26x  | -91.12%        |
| 2048x2048 | float32 | grain_extract | scalar | 0.288486 | 0.084271   | 3.42x   | -70.79%        |
| 2048x2048 | float32 | grain_extract | sse42  | 0.288486 | 0.030779   | 9.37x   | -89.33%        |
| 2048x2048 | float32 | grain_extract | avx2   | 0.288486 | 0.029954   | 9.63x   | -89.62%        |
| 2048x2048 | float32 | grain_merge   | scalar | 0.315823 | 0.084692   | 3.73x   | -73.18%        |
| 2048x2048 | float32 | grain_merge   | sse42  | 0.315823 | 0.030843   | 10.24x  | -90.23%        |
| 2048x2048 | float32 | grain_merge   | avx2   | 0.315823 | 0.025532   | 12.37x  | -91.92%        |
| 2048x2048 | float32 | divide        | scalar | 0.299737 | 0.061984   | 4.84x   | -79.32%        |
| 2048x2048 | float32 | divide        | sse42  | 0.299737 | 0.031416   | 9.54x   | -89.52%        |
| 2048x2048 | float32 | divide        | avx2   | 0.299737 | 0.026056   | 11.50x  | -91.31%        |
| 2048x2048 | float32 | overlay       | scalar | 0.415367 | 0.128588   | 3.23x   | -69.04%        |
| 2048x2048 | float32 | overlay       | sse42  | 0.415367 | 0.029788   | 13.94x  | -92.83%        |
| 2048x2048 | float32 | overlay       | avx2   | 0.415367 | 0.025252   | 16.45x  | -93.92%        |
| 1280x720  | uint8   | normal        | scalar | 0.065418 | 0.019282   | 3.39x   | -70.52%        |
| 1280x720  | uint8   | normal        | sse42  | 0.065418 | 0.002904   | 22.53x  | -95.56%        |
| 1280x720  | uint8   | normal        | avx2   | 0.065418 | 0.002223   | 29.43x  | -96.60%        |
| 1280x720  | uint8   | soft_light    | scalar | 0.102925 | 0.025282   | 4.07x   | -75.44%        |
| 1280x720  | uint8   | soft_light    | sse42  | 0.102925 | 0.003460   | 29.74x  | -96.64%        |
| 1280x720  | uint8   | soft_light    | avx2   | 0.102925 | 0.002818   | 36.52x  | -97.26%        |
| 1280x720  | uint8   | lighten_only  | scalar | 0.077020 | 0.024812   | 3.10x   | -67.78%        |
| 1280x720  | uint8   | lighten_only  | sse42  | 0.077020 | 0.003184   | 24.19x  | -95.87%        |
| 1280x720  | uint8   | lighten_only  | avx2   | 0.077020 | 0.002696   | 28.57x  | -96.50%        |
| 1280x720  | uint8   | screen        | scalar | 0.079226 | 0.025781   | 3.07x   | -67.46%        |
| 1280x720  | uint8   | screen        | sse42  | 0.079226 | 0.003157   | 25.10x  | -96.02%        |
| 1280x720  | uint8   | screen        | avx2   | 0.079226 | 0.002723   | 29.10x  | -96.56%        |
| 1280x720  | uint8   | dodge         | scalar | 0.079655 | 0.024497   | 3.25x   | -69.25%        |
| 1280x720  | uint8   | dodge         | sse42  | 0.079655 | 0.003466   | 22.98x  | -95.65%        |
| 1280x720  | uint8   | dodge         | avx2   | 0.079655 | 0.002727   | 29.21x  | -96.58%        |
| 1280x720  | uint8   | addition      | scalar | 0.073449 | 0.029117   | 2.52x   | -60.36%        |
| 1280x720  | uint8   | addition      | sse42  | 0.073449 | 0.003811   | 19.27x  | -94.81%        |
| 1280x720  | uint8   | addition      | avx2   | 0.073449 | 0.002745   | 26.75x  | -96.26%        |
| 1280x720  | uint8   | darken_only   | scalar | 0.073442 | 0.024018   | 3.06x   | -67.30%        |
| 1280x720  | uint8   | darken_only   | sse42  | 0.073442 | 0.003058   | 24.02x  | -95.84%        |
| 1280x720  | uint8   | darken_only   | avx2   | 0.073442 | 0.002628   | 27.95x  | -96.42%        |
| 1280x720  | uint8   | multiply      | scalar | 0.075777 | 0.024024   | 3.15x   | -68.30%        |
| 1280x720  | uint8   | multiply      | sse42  | 0.075777 | 0.003062   | 24.75x  | -95.96%        |
| 1280x720  | uint8   | multiply      | avx2   | 0.075777 | 0.002863   | 26.46x  | -96.22%        |
| 1280x720  | uint8   | hard_light    | scalar | 0.108004 | 0.038444   | 2.81x   | -64.41%        |
| 1280x720  | uint8   | hard_light    | sse42  | 0.108004 | 0.003602   | 29.99x  | -96.67%        |
| 1280x720  | uint8   | hard_light    | avx2   | 0.108004 | 0.002712   | 39.82x  | -97.49%        |
| 1280x720  | uint8   | difference    | scalar | 0.101540 | 0.022814   | 4.45x   | -77.53%        |
| 1280x720  | uint8   | difference    | sse42  | 0.101540 | 0.003087   | 32.89x  | -96.96%        |
| 1280x720  | uint8   | difference    | avx2   | 0.101540 | 0.002621   | 38.73x  | -97.42%        |
| 1280x720  | uint8   | subtract      | scalar | 0.074603 | 0.024409   | 3.06x   | -67.28%        |
| 1280x720  | uint8   | subtract      | sse42  | 0.074603 | 0.003874   | 19.26x  | -94.81%        |
| 1280x720  | uint8   | subtract      | avx2   | 0.074603 | 0.002826   | 26.39x  | -96.21%        |
| 1280x720  | uint8   | grain_extract | scalar | 0.076637 | 0.027971   | 2.74x   | -63.50%        |
| 1280x720  | uint8   | grain_extract | sse42  | 0.076637 | 0.003181   | 24.09x  | -95.85%        |
| 1280x720  | uint8   | grain_extract | avx2   | 0.076637 | 0.002691   | 28.48x  | -96.49%        |
| 1280x720  | uint8   | grain_merge   | scalar | 0.074892 | 0.036750   | 2.04x   | -50.93%        |
| 1280x720  | uint8   | grain_merge   | sse42  | 0.074892 | 0.003192   | 23.47x  | -95.74%        |
| 1280x720  | uint8   | grain_merge   | avx2   | 0.074892 | 0.002673   | 28.02x  | -96.43%        |
| 1280x720  | uint8   | divide        | scalar | 0.077430 | 0.023835   | 3.25x   | -69.22%        |
| 1280x720  | uint8   | divide        | sse42  | 0.077430 | 0.003279   | 23.61x  | -95.77%        |
| 1280x720  | uint8   | divide        | avx2   | 0.077430 | 0.002695   | 28.74x  | -96.52%        |
| 1280x720  | uint8   | overlay       | scalar | 0.099515 | 0.036507   | 2.73x   | -63.32%        |
| 1280x720  | uint8   | overlay       | sse42  | 0.099515 | 0.003482   | 28.58x  | -96.50%        |
| 1280x720  | uint8   | overlay       | avx2   | 0.099515 | 0.002716   | 36.64x  | -97.27%        |
| 1280x720  | float32 | normal        | scalar | 0.058640 | 0.009530   | 6.15x   | -83.75%        |
| 1280x720  | float32 | normal        | sse42  | 0.058640 | 0.004372   | 13.41x  | -92.54%        |
| 1280x720  | float32 | normal        | avx2   | 0.058640 | 0.003540   | 16.57x  | -93.96%        |
| 1280x720  | float32 | soft_light    | scalar | 0.089443 | 0.011899   | 7.52x   | -86.70%        |
| 1280x720  | float32 | soft_light    | sse42  | 0.089443 | 0.005046   | 17.73x  | -94.36%        |
| 1280x720  | float32 | soft_light    | avx2   | 0.089443 | 0.003845   | 23.26x  | -95.70%        |
| 1280x720  | float32 | lighten_only  | scalar | 0.066600 | 0.013136   | 5.07x   | -80.28%        |
| 1280x720  | float32 | lighten_only  | sse42  | 0.066600 | 0.005221   | 12.76x  | -92.16%        |
| 1280x720  | float32 | lighten_only  | avx2   | 0.066600 | 0.003865   | 17.23x  | -94.20%        |
| 1280x720  | float32 | screen        | scalar | 0.069480 | 0.011963   | 5.81x   | -82.78%        |
| 1280x720  | float32 | screen        | sse42  | 0.069480 | 0.005426   | 12.80x  | -92.19%        |
| 1280x720  | float32 | screen        | avx2   | 0.069480 | 0.004249   | 16.35x  | -93.88%        |
| 1280x720  | float32 | dodge         | scalar | 0.071842 | 0.012385   | 5.80x   | -82.76%        |
| 1280x720  | float32 | dodge         | sse42  | 0.071842 | 0.005170   | 13.90x  | -92.80%        |
| 1280x720  | float32 | dodge         | avx2   | 0.071842 | 0.003984   | 18.03x  | -94.45%        |
| 1280x720  | float32 | addition      | scalar | 0.065334 | 0.020681   | 3.16x   | -68.35%        |
| 1280x720  | float32 | addition      | sse42  | 0.065334 | 0.005406   | 12.08x  | -91.72%        |
| 1280x720  | float32 | addition      | avx2   | 0.065334 | 0.003874   | 16.87x  | -94.07%        |
| 1280x720  | float32 | darken_only   | scalar | 0.066511 | 0.013008   | 5.11x   | -80.44%        |
| 1280x720  | float32 | darken_only   | sse42  | 0.066511 | 0.005193   | 12.81x  | -92.19%        |
| 1280x720  | float32 | darken_only   | avx2   | 0.066511 | 0.003806   | 17.48x  | -94.28%        |
| 1280x720  | float32 | multiply      | scalar | 0.069171 | 0.010930   | 6.33x   | -84.20%        |
| 1280x720  | float32 | multiply      | sse42  | 0.069171 | 0.005295   | 13.06x  | -92.35%        |
| 1280x720  | float32 | multiply      | avx2   | 0.069171 | 0.003819   | 18.11x  | -94.48%        |
| 1280x720  | float32 | hard_light    | scalar | 0.100195 | 0.027746   | 3.61x   | -72.31%        |
| 1280x720  | float32 | hard_light    | sse42  | 0.100195 | 0.005062   | 19.79x  | -94.95%        |
| 1280x720  | float32 | hard_light    | avx2   | 0.100195 | 0.003981   | 25.17x  | -96.03%        |
| 1280x720  | float32 | difference    | scalar | 0.093440 | 0.010983   | 8.51x   | -88.25%        |
| 1280x720  | float32 | difference    | sse42  | 0.093440 | 0.005155   | 18.13x  | -94.48%        |
| 1280x720  | float32 | difference    | avx2   | 0.093440 | 0.003842   | 24.32x  | -95.89%        |
| 1280x720  | float32 | subtract      | scalar | 0.066311 | 0.014253   | 4.65x   | -78.51%        |
| 1280x720  | float32 | subtract      | sse42  | 0.066311 | 0.005428   | 12.22x  | -91.81%        |
| 1280x720  | float32 | subtract      | avx2   | 0.066311 | 0.003994   | 16.60x  | -93.98%        |
| 1280x720  | float32 | grain_extract | scalar | 0.068979 | 0.016664   | 4.14x   | -75.84%        |
| 1280x720  | float32 | grain_extract | sse42  | 0.068979 | 0.005093   | 13.54x  | -92.62%        |
| 1280x720  | float32 | grain_extract | avx2   | 0.068979 | 0.003815   | 18.08x  | -94.47%        |
| 1280x720  | float32 | grain_merge   | scalar | 0.068447 | 0.016697   | 4.10x   | -75.61%        |
| 1280x720  | float32 | grain_merge   | sse42  | 0.068447 | 0.005101   | 13.42x  | -92.55%        |
| 1280x720  | float32 | grain_merge   | avx2   | 0.068447 | 0.003781   | 18.10x  | -94.48%        |
| 1280x720  | float32 | divide        | scalar | 0.071225 | 0.011841   | 6.02x   | -83.38%        |
| 1280x720  | float32 | divide        | sse42  | 0.071225 | 0.005200   | 13.70x  | -92.70%        |
| 1280x720  | float32 | divide        | avx2   | 0.071225 | 0.003852   | 18.49x  | -94.59%        |
| 1280x720  | float32 | overlay       | scalar | 0.099901 | 0.026490   | 3.77x   | -73.48%        |
| 1280x720  | float32 | overlay       | sse42  | 0.099901 | 0.004952   | 20.18x  | -95.04%        |
| 1280x720  | float32 | overlay       | avx2   | 0.099901 | 0.003800   | 26.29x  | -96.20%        |
| 1920x1080 | uint8   | normal        | scalar | 0.138848 | 0.044718   | 3.10x   | -67.79%        |
| 1920x1080 | uint8   | normal        | sse42  | 0.138848 | 0.006040   | 22.99x  | -95.65%        |
| 1920x1080 | uint8   | normal        | avx2   | 0.138848 | 0.004828   | 28.76x  | -96.52%        |
| 1920x1080 | uint8   | soft_light    | scalar | 0.193238 | 0.053789   | 3.59x   | -72.16%        |
| 1920x1080 | uint8   | soft_light    | sse42  | 0.193238 | 0.007300   | 26.47x  | -96.22%        |
| 1920x1080 | uint8   | soft_light    | avx2   | 0.193238 | 0.005986   | 32.28x  | -96.90%        |
| 1920x1080 | uint8   | lighten_only  | scalar | 0.136868 | 0.053873   | 2.54x   | -60.64%        |
| 1920x1080 | uint8   | lighten_only  | sse42  | 0.136868 | 0.006867   | 19.93x  | -94.98%        |
| 1920x1080 | uint8   | lighten_only  | avx2   | 0.136868 | 0.005945   | 23.02x  | -95.66%        |
| 1920x1080 | uint8   | screen        | scalar | 0.146941 | 0.051330   | 2.86x   | -65.07%        |
| 1920x1080 | uint8   | screen        | sse42  | 0.146941 | 0.007001   | 20.99x  | -95.24%        |
| 1920x1080 | uint8   | screen        | avx2   | 0.146941 | 0.005861   | 25.07x  | -96.01%        |
| 1920x1080 | uint8   | dodge         | scalar | 0.145535 | 0.053933   | 2.70x   | -62.94%        |
| 1920x1080 | uint8   | dodge         | sse42  | 0.145535 | 0.007715   | 18.86x  | -94.70%        |
| 1920x1080 | uint8   | dodge         | avx2   | 0.145535 | 0.006129   | 23.74x  | -95.79%        |
| 1920x1080 | uint8   | addition      | scalar | 0.140008 | 0.065165   | 2.15x   | -53.46%        |
| 1920x1080 | uint8   | addition      | sse42  | 0.140008 | 0.008620   | 16.24x  | -93.84%        |
| 1920x1080 | uint8   | addition      | avx2   | 0.140008 | 0.006172   | 22.69x  | -95.59%        |
| 1920x1080 | uint8   | darken_only   | scalar | 0.139759 | 0.053645   | 2.61x   | -61.62%        |
| 1920x1080 | uint8   | darken_only   | sse42  | 0.139759 | 0.006865   | 20.36x  | -95.09%        |
| 1920x1080 | uint8   | darken_only   | avx2   | 0.139759 | 0.005880   | 23.77x  | -95.79%        |
| 1920x1080 | uint8   | multiply      | scalar | 0.140494 | 0.051779   | 2.71x   | -63.14%        |
| 1920x1080 | uint8   | multiply      | sse42  | 0.140494 | 0.006868   | 20.46x  | -95.11%        |
| 1920x1080 | uint8   | multiply      | avx2   | 0.140494 | 0.005862   | 23.97x  | -95.83%        |
| 1920x1080 | uint8   | hard_light    | scalar | 0.207118 | 0.085374   | 2.43x   | -58.78%        |
| 1920x1080 | uint8   | hard_light    | sse42  | 0.207118 | 0.007958   | 26.03x  | -96.16%        |
| 1920x1080 | uint8   | hard_light    | avx2   | 0.207118 | 0.006038   | 34.30x  | -97.08%        |
| 1920x1080 | uint8   | difference    | scalar | 0.197435 | 0.054205   | 3.64x   | -72.55%        |
| 1920x1080 | uint8   | difference    | sse42  | 0.197435 | 0.007164   | 27.56x  | -96.37%        |
| 1920x1080 | uint8   | difference    | avx2   | 0.197435 | 0.005906   | 33.43x  | -97.01%        |
| 1920x1080 | uint8   | subtract      | scalar | 0.140046 | 0.051615   | 2.71x   | -63.14%        |
| 1920x1080 | uint8   | subtract      | sse42  | 0.140046 | 0.008576   | 16.33x  | -93.88%        |
| 1920x1080 | uint8   | subtract      | avx2   | 0.140046 | 0.006201   | 22.58x  | -95.57%        |
| 1920x1080 | uint8   | grain_extract | scalar | 0.142875 | 0.061070   | 2.34x   | -57.26%        |
| 1920x1080 | uint8   | grain_extract | sse42  | 0.142875 | 0.010055   | 14.21x  | -92.96%        |
| 1920x1080 | uint8   | grain_extract | avx2   | 0.142875 | 0.007230   | 19.76x  | -94.94%        |
| 1920x1080 | uint8   | grain_merge   | scalar | 0.148838 | 0.061865   | 2.41x   | -58.43%        |
| 1920x1080 | uint8   | grain_merge   | sse42  | 0.148838 | 0.007173   | 20.75x  | -95.18%        |
| 1920x1080 | uint8   | grain_merge   | avx2   | 0.148838 | 0.005999   | 24.81x  | -95.97%        |
| 1920x1080 | uint8   | divide        | scalar | 0.147516 | 0.054063   | 2.73x   | -63.35%        |
| 1920x1080 | uint8   | divide        | sse42  | 0.147516 | 0.007374   | 20.00x  | -95.00%        |
| 1920x1080 | uint8   | divide        | avx2   | 0.147516 | 0.006127   | 24.08x  | -95.85%        |
| 1920x1080 | uint8   | overlay       | scalar | 0.192576 | 0.081690   | 2.36x   | -57.58%        |
| 1920x1080 | uint8   | overlay       | sse42  | 0.192576 | 0.007554   | 25.49x  | -96.08%        |
| 1920x1080 | uint8   | overlay       | avx2   | 0.192576 | 0.006000   | 32.09x  | -96.88%        |
| 1920x1080 | float32 | normal        | scalar | 0.119239 | 0.019554   | 6.10x   | -83.60%        |
| 1920x1080 | float32 | normal        | sse42  | 0.119239 | 0.009458   | 12.61x  | -92.07%        |
| 1920x1080 | float32 | normal        | avx2   | 0.119239 | 0.007276   | 16.39x  | -93.90%        |
| 1920x1080 | float32 | soft_light    | scalar | 0.178459 | 0.025073   | 7.12x   | -85.95%        |
| 1920x1080 | float32 | soft_light    | sse42  | 0.178459 | 0.013326   | 13.39x  | -92.53%        |
| 1920x1080 | float32 | soft_light    | avx2   | 0.178459 | 0.008483   | 21.04x  | -95.25%        |
| 1920x1080 | float32 | lighten_only  | scalar | 0.127971 | 0.027652   | 4.63x   | -78.39%        |
| 1920x1080 | float32 | lighten_only  | sse42  | 0.127971 | 0.011377   | 11.25x  | -91.11%        |
| 1920x1080 | float32 | lighten_only  | avx2   | 0.127971 | 0.008345   | 15.34x  | -93.48%        |
| 1920x1080 | float32 | screen        | scalar | 0.138441 | 0.025078   | 5.52x   | -81.89%        |
| 1920x1080 | float32 | screen        | sse42  | 0.138441 | 0.011223   | 12.34x  | -91.89%        |
| 1920x1080 | float32 | screen        | avx2   | 0.138441 | 0.008671   | 15.97x  | -93.74%        |
| 1920x1080 | float32 | dodge         | scalar | 0.140259 | 0.026050   | 5.38x   | -81.43%        |
| 1920x1080 | float32 | dodge         | sse42  | 0.140259 | 0.011495   | 12.20x  | -91.80%        |
| 1920x1080 | float32 | dodge         | avx2   | 0.140259 | 0.008566   | 16.37x  | -93.89%        |
| 1920x1080 | float32 | addition      | scalar | 0.133955 | 0.045696   | 2.93x   | -65.89%        |
| 1920x1080 | float32 | addition      | sse42  | 0.133955 | 0.012138   | 11.04x  | -90.94%        |
| 1920x1080 | float32 | addition      | avx2   | 0.133955 | 0.008950   | 14.97x  | -93.32%        |
| 1920x1080 | float32 | darken_only   | scalar | 0.129843 | 0.028220   | 4.60x   | -78.27%        |
| 1920x1080 | float32 | darken_only   | sse42  | 0.129843 | 0.011377   | 11.41x  | -91.24%        |
| 1920x1080 | float32 | darken_only   | avx2   | 0.129843 | 0.007999   | 16.23x  | -93.84%        |
| 1920x1080 | float32 | multiply      | scalar | 0.131430 | 0.022429   | 5.86x   | -82.93%        |
| 1920x1080 | float32 | multiply      | sse42  | 0.131430 | 0.011369   | 11.56x  | -91.35%        |
| 1920x1080 | float32 | multiply      | avx2   | 0.131430 | 0.008382   | 15.68x  | -93.62%        |
| 1920x1080 | float32 | hard_light    | scalar | 0.205010 | 0.060762   | 3.37x   | -70.36%        |
| 1920x1080 | float32 | hard_light    | sse42  | 0.205010 | 0.011058   | 18.54x  | -94.61%        |
| 1920x1080 | float32 | hard_light    | avx2   | 0.205010 | 0.008182   | 25.06x  | -96.01%        |
| 1920x1080 | float32 | difference    | scalar | 0.187268 | 0.023311   | 8.03x   | -87.55%        |
| 1920x1080 | float32 | difference    | sse42  | 0.187268 | 0.011278   | 16.60x  | -93.98%        |
| 1920x1080 | float32 | difference    | avx2   | 0.187268 | 0.008152   | 22.97x  | -95.65%        |
| 1920x1080 | float32 | subtract      | scalar | 0.135574 | 0.030767   | 4.41x   | -77.31%        |
| 1920x1080 | float32 | subtract      | sse42  | 0.135574 | 0.012039   | 11.26x  | -91.12%        |
| 1920x1080 | float32 | subtract      | avx2   | 0.135574 | 0.008588   | 15.79x  | -93.67%        |
| 1920x1080 | float32 | grain_extract | scalar | 0.144617 | 0.036228   | 3.99x   | -74.95%        |
| 1920x1080 | float32 | grain_extract | sse42  | 0.144617 | 0.011398   | 12.69x  | -92.12%        |
| 1920x1080 | float32 | grain_extract | avx2   | 0.144617 | 0.008878   | 16.29x  | -93.86%        |
| 1920x1080 | float32 | grain_merge   | scalar | 0.149801 | 0.036254   | 4.13x   | -75.80%        |
| 1920x1080 | float32 | grain_merge   | sse42  | 0.149801 | 0.011442   | 13.09x  | -92.36%        |
| 1920x1080 | float32 | grain_merge   | avx2   | 0.149801 | 0.008636   | 17.35x  | -94.23%        |
| 1920x1080 | float32 | divide        | scalar | 0.139193 | 0.025160   | 5.53x   | -81.92%        |
| 1920x1080 | float32 | divide        | sse42  | 0.139193 | 0.011538   | 12.06x  | -91.71%        |
| 1920x1080 | float32 | divide        | avx2   | 0.139193 | 0.008676   | 16.04x  | -93.77%        |
| 1920x1080 | float32 | overlay       | scalar | 0.194692 | 0.058005   | 3.36x   | -70.21%        |
| 1920x1080 | float32 | overlay       | sse42  | 0.194692 | 0.011094   | 17.55x  | -94.30%        |
| 1920x1080 | float32 | overlay       | avx2   | 0.194692 | 0.008247   | 23.61x  | -95.76%        |
| 2560x1440 | uint8   | normal        | scalar | 0.255139 | 0.077829   | 3.28x   | -69.50%        |
| 2560x1440 | uint8   | normal        | sse42  | 0.255139 | 0.011028   | 23.14x  | -95.68%        |
| 2560x1440 | uint8   | normal        | avx2   | 0.255139 | 0.008760   | 29.12x  | -96.57%        |
| 2560x1440 | uint8   | soft_light    | scalar | 0.348927 | 0.097442   | 3.58x   | -72.07%        |
| 2560x1440 | uint8   | soft_light    | sse42  | 0.348927 | 0.013289   | 26.26x  | -96.19%        |
| 2560x1440 | uint8   | soft_light    | avx2   | 0.348927 | 0.011637   | 29.98x  | -96.66%        |
| 2560x1440 | uint8   | lighten_only  | scalar | 0.242312 | 0.095663   | 2.53x   | -60.52%        |
| 2560x1440 | uint8   | lighten_only  | sse42  | 0.242312 | 0.012381   | 19.57x  | -94.89%        |
| 2560x1440 | uint8   | lighten_only  | avx2   | 0.242312 | 0.011120   | 21.79x  | -95.41%        |
| 2560x1440 | uint8   | screen        | scalar | 0.267929 | 0.092352   | 2.90x   | -65.53%        |
| 2560x1440 | uint8   | screen        | sse42  | 0.267929 | 0.012717   | 21.07x  | -95.25%        |
| 2560x1440 | uint8   | screen        | avx2   | 0.267929 | 0.010565   | 25.36x  | -96.06%        |
| 2560x1440 | uint8   | dodge         | scalar | 0.367163 | 0.112946   | 3.25x   | -69.24%        |
| 2560x1440 | uint8   | dodge         | sse42  | 0.367163 | 0.014865   | 24.70x  | -95.95%        |
| 2560x1440 | uint8   | dodge         | avx2   | 0.367163 | 0.011830   | 31.04x  | -96.78%        |
| 2560x1440 | uint8   | addition      | scalar | 0.326326 | 0.119525   | 2.73x   | -63.37%        |
| 2560x1440 | uint8   | addition      | sse42  | 0.326326 | 0.015485   | 21.07x  | -95.25%        |
| 2560x1440 | uint8   | addition      | avx2   | 0.326326 | 0.011542   | 28.27x  | -96.46%        |
| 2560x1440 | uint8   | darken_only   | scalar | 0.250035 | 0.100302   | 2.49x   | -59.88%        |
| 2560x1440 | uint8   | darken_only   | sse42  | 0.250035 | 0.012284   | 20.35x  | -95.09%        |
| 2560x1440 | uint8   | darken_only   | avx2   | 0.250035 | 0.010632   | 23.52x  | -95.75%        |
| 2560x1440 | uint8   | multiply      | scalar | 0.259294 | 0.099554   | 2.60x   | -61.61%        |
| 2560x1440 | uint8   | multiply      | sse42  | 0.259294 | 0.012598   | 20.58x  | -95.14%        |
| 2560x1440 | uint8   | multiply      | avx2   | 0.259294 | 0.012497   | 20.75x  | -95.18%        |
| 2560x1440 | uint8   | hard_light    | scalar | 0.429509 | 0.154498   | 2.78x   | -64.03%        |
| 2560x1440 | uint8   | hard_light    | sse42  | 0.429509 | 0.015017   | 28.60x  | -96.50%        |
| 2560x1440 | uint8   | hard_light    | avx2   | 0.429509 | 0.011256   | 38.16x  | -97.38%        |
| 2560x1440 | uint8   | difference    | scalar | 0.385166 | 0.097813   | 3.94x   | -74.60%        |
| 2560x1440 | uint8   | difference    | sse42  | 0.385166 | 0.013437   | 28.66x  | -96.51%        |
| 2560x1440 | uint8   | difference    | avx2   | 0.385166 | 0.010800   | 35.66x  | -97.20%        |
| 2560x1440 | uint8   | subtract      | scalar | 0.323918 | 0.099078   | 3.27x   | -69.41%        |
| 2560x1440 | uint8   | subtract      | sse42  | 0.323918 | 0.015697   | 20.64x  | -95.15%        |
| 2560x1440 | uint8   | subtract      | avx2   | 0.323918 | 0.011622   | 27.87x  | -96.41%        |
| 2560x1440 | uint8   | grain_extract | scalar | 0.277849 | 0.119929   | 2.32x   | -56.84%        |
| 2560x1440 | uint8   | grain_extract | sse42  | 0.277849 | 0.013157   | 21.12x  | -95.26%        |
| 2560x1440 | uint8   | grain_extract | avx2   | 0.277849 | 0.010977   | 25.31x  | -96.05%        |
| 2560x1440 | uint8   | grain_merge   | scalar | 0.300567 | 0.117643   | 2.55x   | -60.86%        |
| 2560x1440 | uint8   | grain_merge   | sse42  | 0.300567 | 0.013182   | 22.80x  | -95.61%        |
| 2560x1440 | uint8   | grain_merge   | avx2   | 0.300567 | 0.011182   | 26.88x  | -96.28%        |
| 2560x1440 | uint8   | divide        | scalar | 0.297638 | 0.112564   | 2.64x   | -62.18%        |
| 2560x1440 | uint8   | divide        | sse42  | 0.297638 | 0.018271   | 16.29x  | -93.86%        |
| 2560x1440 | uint8   | divide        | avx2   | 0.297638 | 0.012489   | 23.83x  | -95.80%        |
| 2560x1440 | uint8   | overlay       | scalar | 0.472751 | 0.168850   | 2.80x   | -64.28%        |
| 2560x1440 | uint8   | overlay       | sse42  | 0.472751 | 0.015881   | 29.77x  | -96.64%        |
| 2560x1440 | uint8   | overlay       | avx2   | 0.472751 | 0.012566   | 37.62x  | -97.34%        |
| 2560x1440 | float32 | normal        | scalar | 0.325487 | 0.047543   | 6.85x   | -85.39%        |
| 2560x1440 | float32 | normal        | sse42  | 0.325487 | 0.027379   | 11.89x  | -91.59%        |
| 2560x1440 | float32 | normal        | avx2   | 0.325487 | 0.023271   | 13.99x  | -92.85%        |
| 2560x1440 | float32 | soft_light    | scalar | 0.417767 | 0.048470   | 8.62x   | -88.40%        |
| 2560x1440 | float32 | soft_light    | sse42  | 0.417767 | 0.022103   | 18.90x  | -94.71%        |
| 2560x1440 | float32 | soft_light    | avx2   | 0.417767 | 0.018060   | 23.13x  | -95.68%        |
| 2560x1440 | float32 | lighten_only  | scalar | 0.269722 | 0.090701   | 2.97x   | -66.37%        |
| 2560x1440 | float32 | lighten_only  | sse42  | 0.269722 | 0.031770   | 8.49x   | -88.22%        |
| 2560x1440 | float32 | lighten_only  | avx2   | 0.269722 | 0.026973   | 10.00x  | -90.00%        |
| 2560x1440 | float32 | screen        | scalar | 0.312516 | 0.047079   | 6.64x   | -84.94%        |
| 2560x1440 | float32 | screen        | sse42  | 0.312516 | 0.021132   | 14.79x  | -93.24%        |
| 2560x1440 | float32 | screen        | avx2   | 0.312516 | 0.016966   | 18.42x  | -94.57%        |
| 2560x1440 | float32 | dodge         | scalar | 0.296207 | 0.058345   | 5.08x   | -80.30%        |
| 2560x1440 | float32 | dodge         | sse42  | 0.296207 | 0.027085   | 10.94x  | -90.86%        |
| 2560x1440 | float32 | dodge         | avx2   | 0.296207 | 0.023087   | 12.83x  | -92.21%        |
| 2560x1440 | float32 | addition      | scalar | 0.260459 | 0.083637   | 3.11x   | -67.89%        |
| 2560x1440 | float32 | addition      | sse42  | 0.260459 | 0.022067   | 11.80x  | -91.53%        |
| 2560x1440 | float32 | addition      | avx2   | 0.260459 | 0.016975   | 15.34x  | -93.48%        |
| 2560x1440 | float32 | darken_only   | scalar | 0.242680 | 0.058591   | 4.14x   | -75.86%        |
| 2560x1440 | float32 | darken_only   | sse42  | 0.242680 | 0.027296   | 8.89x   | -88.75%        |
| 2560x1440 | float32 | darken_only   | avx2   | 0.242680 | 0.021337   | 11.37x  | -91.21%        |
| 2560x1440 | float32 | multiply      | scalar | 0.251209 | 0.042303   | 5.94x   | -83.16%        |
| 2560x1440 | float32 | multiply      | sse42  | 0.251209 | 0.023858   | 10.53x  | -90.50%        |
| 2560x1440 | float32 | multiply      | avx2   | 0.251209 | 0.022320   | 11.25x  | -91.11%        |
| 2560x1440 | float32 | hard_light    | scalar | 0.450454 | 0.119631   | 3.77x   | -73.44%        |
| 2560x1440 | float32 | hard_light    | sse42  | 0.450454 | 0.027444   | 16.41x  | -93.91%        |
| 2560x1440 | float32 | hard_light    | avx2   | 0.450454 | 0.024696   | 18.24x  | -94.52%        |
| 2560x1440 | float32 | difference    | scalar | 0.360172 | 0.043289   | 8.32x   | -87.98%        |
| 2560x1440 | float32 | difference    | sse42  | 0.360172 | 0.020325   | 17.72x  | -94.36%        |
| 2560x1440 | float32 | difference    | avx2   | 0.360172 | 0.014881   | 24.20x  | -95.87%        |
| 2560x1440 | float32 | subtract      | scalar | 0.256480 | 0.064032   | 4.01x   | -75.03%        |
| 2560x1440 | float32 | subtract      | sse42  | 0.256480 | 0.027565   | 9.30x   | -89.25%        |
| 2560x1440 | float32 | subtract      | avx2   | 0.256480 | 0.021358   | 12.01x  | -91.67%        |
| 2560x1440 | float32 | grain_extract | scalar | 0.261604 | 0.064975   | 4.03x   | -75.16%        |
| 2560x1440 | float32 | grain_extract | sse42  | 0.261604 | 0.020957   | 12.48x  | -91.99%        |
| 2560x1440 | float32 | grain_extract | avx2   | 0.261604 | 0.014808   | 17.67x  | -94.34%        |
| 2560x1440 | float32 | grain_merge   | scalar | 0.252671 | 0.073489   | 3.44x   | -70.92%        |
| 2560x1440 | float32 | grain_merge   | sse42  | 0.252671 | 0.026257   | 9.62x   | -89.61%        |
| 2560x1440 | float32 | grain_merge   | avx2   | 0.252671 | 0.021567   | 11.72x  | -91.46%        |
| 2560x1440 | float32 | divide        | scalar | 0.255273 | 0.045106   | 5.66x   | -82.33%        |
| 2560x1440 | float32 | divide        | sse42  | 0.255273 | 0.024162   | 10.57x  | -90.53%        |
| 2560x1440 | float32 | divide        | avx2   | 0.255273 | 0.017356   | 14.71x  | -93.20%        |
| 2560x1440 | float32 | overlay       | scalar | 0.359658 | 0.112368   | 3.20x   | -68.76%        |
| 2560x1440 | float32 | overlay       | sse42  | 0.359658 | 0.028160   | 12.77x  | -92.17%        |
| 2560x1440 | float32 | overlay       | avx2   | 0.359658 | 0.021950   | 16.39x  | -93.90%        |
| 3840x2160 | uint8   | normal        | scalar | 0.606416 | 0.176192   | 3.44x   | -70.95%        |
| 3840x2160 | uint8   | normal        | sse42  | 0.606416 | 0.025395   | 23.88x  | -95.81%        |
| 3840x2160 | uint8   | normal        | avx2   | 0.606416 | 0.020258   | 29.93x  | -96.66%        |
| 3840x2160 | uint8   | soft_light    | scalar | 0.771831 | 0.223426   | 3.45x   | -71.05%        |
| 3840x2160 | uint8   | soft_light    | sse42  | 0.771831 | 0.031147   | 24.78x  | -95.96%        |
| 3840x2160 | uint8   | soft_light    | avx2   | 0.771831 | 0.023919   | 32.27x  | -96.90%        |
| 3840x2160 | uint8   | lighten_only  | scalar | 0.548321 | 0.231142   | 2.37x   | -57.85%        |
| 3840x2160 | uint8   | lighten_only  | sse42  | 0.548321 | 0.027764   | 19.75x  | -94.94%        |
| 3840x2160 | uint8   | lighten_only  | avx2   | 0.548321 | 0.023837   | 23.00x  | -95.65%        |
| 3840x2160 | uint8   | screen        | scalar | 0.598712 | 0.221648   | 2.70x   | -62.98%        |
| 3840x2160 | uint8   | screen        | sse42  | 0.598712 | 0.028783   | 20.80x  | -95.19%        |
| 3840x2160 | uint8   | screen        | avx2   | 0.598712 | 0.023990   | 24.96x  | -95.99%        |
| 3840x2160 | uint8   | dodge         | scalar | 0.594928 | 0.233698   | 2.55x   | -60.72%        |
| 3840x2160 | uint8   | dodge         | sse42  | 0.594928 | 0.032694   | 18.20x  | -94.50%        |
| 3840x2160 | uint8   | dodge         | avx2   | 0.594928 | 0.025612   | 23.23x  | -95.69%        |
| 3840x2160 | uint8   | addition      | scalar | 0.591811 | 0.276889   | 2.14x   | -53.21%        |
| 3840x2160 | uint8   | addition      | sse42  | 0.591811 | 0.035643   | 16.60x  | -93.98%        |
| 3840x2160 | uint8   | addition      | avx2   | 0.591811 | 0.025372   | 23.33x  | -95.71%        |
| 3840x2160 | uint8   | darken_only   | scalar | 0.626804 | 0.243417   | 2.58x   | -61.17%        |
| 3840x2160 | uint8   | darken_only   | sse42  | 0.626804 | 0.037686   | 16.63x  | -93.99%        |
| 3840x2160 | uint8   | darken_only   | avx2   | 0.626804 | 0.024508   | 25.58x  | -96.09%        |
| 3840x2160 | uint8   | multiply      | scalar | 0.626420 | 0.224295   | 2.79x   | -64.19%        |
| 3840x2160 | uint8   | multiply      | sse42  | 0.626420 | 0.029775   | 21.04x  | -95.25%        |
| 3840x2160 | uint8   | multiply      | avx2   | 0.626420 | 0.024443   | 25.63x  | -96.10%        |
| 3840x2160 | uint8   | hard_light    | scalar | 0.957533 | 0.367457   | 2.61x   | -61.62%        |
| 3840x2160 | uint8   | hard_light    | sse42  | 0.957533 | 0.033131   | 28.90x  | -96.54%        |
| 3840x2160 | uint8   | hard_light    | avx2   | 0.957533 | 0.025999   | 36.83x  | -97.28%        |
| 3840x2160 | uint8   | difference    | scalar | 0.850651 | 0.232033   | 3.67x   | -72.72%        |
| 3840x2160 | uint8   | difference    | sse42  | 0.850651 | 0.029543   | 28.79x  | -96.53%        |
| 3840x2160 | uint8   | difference    | avx2   | 0.850651 | 0.027695   | 30.71x  | -96.74%        |
| 3840x2160 | uint8   | subtract      | scalar | 0.647738 | 0.224477   | 2.89x   | -65.34%        |
| 3840x2160 | uint8   | subtract      | sse42  | 0.647738 | 0.035112   | 18.45x  | -94.58%        |
| 3840x2160 | uint8   | subtract      | avx2   | 0.647738 | 0.025394   | 25.51x  | -96.08%        |
| 3840x2160 | uint8   | grain_extract | scalar | 0.668233 | 0.270575   | 2.47x   | -59.51%        |
| 3840x2160 | uint8   | grain_extract | sse42  | 0.668233 | 0.029611   | 22.57x  | -95.57%        |
| 3840x2160 | uint8   | grain_extract | avx2   | 0.668233 | 0.024251   | 27.55x  | -96.37%        |
| 3840x2160 | uint8   | grain_merge   | scalar | 0.652997 | 0.265846   | 2.46x   | -59.29%        |
| 3840x2160 | uint8   | grain_merge   | sse42  | 0.652997 | 0.029088   | 22.45x  | -95.55%        |
| 3840x2160 | uint8   | grain_merge   | avx2   | 0.652997 | 0.024294   | 26.88x  | -96.28%        |
| 3840x2160 | uint8   | divide        | scalar | 0.703346 | 0.233714   | 3.01x   | -66.77%        |
| 3840x2160 | uint8   | divide        | sse42  | 0.703346 | 0.030241   | 23.26x  | -95.70%        |
| 3840x2160 | uint8   | divide        | avx2   | 0.703346 | 0.024751   | 28.42x  | -96.48%        |
| 3840x2160 | uint8   | overlay       | scalar | 0.885705 | 0.354499   | 2.50x   | -59.98%        |
| 3840x2160 | uint8   | overlay       | sse42  | 0.885705 | 0.031919   | 27.75x  | -96.40%        |
| 3840x2160 | uint8   | overlay       | avx2   | 0.885705 | 0.028739   | 30.82x  | -96.76%        |
| 3840x2160 | float32 | normal        | scalar | 0.579262 | 0.096408   | 6.01x   | -83.36%        |
| 3840x2160 | float32 | normal        | sse42  | 0.579262 | 0.052067   | 11.13x  | -91.01%        |
| 3840x2160 | float32 | normal        | avx2   | 0.579262 | 0.043858   | 13.21x  | -92.43%        |
| 3840x2160 | float32 | soft_light    | scalar | 0.802854 | 0.121282   | 6.62x   | -84.89%        |
| 3840x2160 | float32 | soft_light    | sse42  | 0.802854 | 0.061460   | 13.06x  | -92.34%        |
| 3840x2160 | float32 | soft_light    | avx2   | 0.802854 | 0.051540   | 15.58x  | -93.58%        |
| 3840x2160 | float32 | lighten_only  | scalar | 0.556059 | 0.128874   | 4.31x   | -76.82%        |
| 3840x2160 | float32 | lighten_only  | sse42  | 0.556059 | 0.061872   | 8.99x   | -88.87%        |
| 3840x2160 | float32 | lighten_only  | avx2   | 0.556059 | 0.047902   | 11.61x  | -91.39%        |
| 3840x2160 | float32 | screen        | scalar | 0.610163 | 0.118599   | 5.14x   | -80.56%        |
| 3840x2160 | float32 | screen        | sse42  | 0.610163 | 0.057976   | 10.52x  | -90.50%        |
| 3840x2160 | float32 | screen        | avx2   | 0.610163 | 0.047838   | 12.75x  | -92.16%        |
| 3840x2160 | float32 | dodge         | scalar | 0.563762 | 0.118401   | 4.76x   | -79.00%        |
| 3840x2160 | float32 | dodge         | sse42  | 0.563762 | 0.059427   | 9.49x   | -89.46%        |
| 3840x2160 | float32 | dodge         | avx2   | 0.563762 | 0.047300   | 11.92x  | -91.61%        |
| 3840x2160 | float32 | addition      | scalar | 0.529833 | 0.195591   | 2.71x   | -63.08%        |
| 3840x2160 | float32 | addition      | sse42  | 0.529833 | 0.061809   | 8.57x   | -88.33%        |
| 3840x2160 | float32 | addition      | avx2   | 0.529833 | 0.049163   | 10.78x  | -90.72%        |
| 3840x2160 | float32 | darken_only   | scalar | 0.504600 | 0.131018   | 3.85x   | -74.04%        |
| 3840x2160 | float32 | darken_only   | sse42  | 0.504600 | 0.058584   | 8.61x   | -88.39%        |
| 3840x2160 | float32 | darken_only   | avx2   | 0.504600 | 0.045752   | 11.03x  | -90.93%        |
| 3840x2160 | float32 | multiply      | scalar | 0.537175 | 0.108095   | 4.97x   | -79.88%        |
| 3840x2160 | float32 | multiply      | sse42  | 0.537175 | 0.061488   | 8.74x   | -88.55%        |
| 3840x2160 | float32 | multiply      | avx2   | 0.537175 | 0.048510   | 11.07x  | -90.97%        |
| 3840x2160 | float32 | hard_light    | scalar | 0.903319 | 0.268163   | 3.37x   | -70.31%        |
| 3840x2160 | float32 | hard_light    | sse42  | 0.903319 | 0.063234   | 14.29x  | -93.00%        |
| 3840x2160 | float32 | hard_light    | avx2   | 0.903319 | 0.052474   | 17.21x  | -94.19%        |
| 3840x2160 | float32 | difference    | scalar | 0.793935 | 0.112376   | 7.06x   | -85.85%        |
| 3840x2160 | float32 | difference    | sse42  | 0.793935 | 0.058830   | 13.50x  | -92.59%        |
| 3840x2160 | float32 | difference    | avx2   | 0.793935 | 0.053492   | 14.84x  | -93.26%        |
| 3840x2160 | float32 | subtract      | scalar | 0.553447 | 0.145829   | 3.80x   | -73.65%        |
| 3840x2160 | float32 | subtract      | sse42  | 0.553447 | 0.065136   | 8.50x   | -88.23%        |
| 3840x2160 | float32 | subtract      | avx2   | 0.553447 | 0.048380   | 11.44x  | -91.26%        |
| 3840x2160 | float32 | grain_extract | scalar | 0.563107 | 0.165016   | 3.41x   | -70.70%        |
| 3840x2160 | float32 | grain_extract | sse42  | 0.563107 | 0.058917   | 9.56x   | -89.54%        |
| 3840x2160 | float32 | grain_extract | avx2   | 0.563107 | 0.047713   | 11.80x  | -91.53%        |
| 3840x2160 | float32 | grain_merge   | scalar | 0.575855 | 0.161879   | 3.56x   | -71.89%        |
| 3840x2160 | float32 | grain_merge   | sse42  | 0.575855 | 0.058910   | 9.78x   | -89.77%        |
| 3840x2160 | float32 | grain_merge   | avx2   | 0.575855 | 0.047779   | 12.05x  | -91.70%        |
| 3840x2160 | float32 | divide        | scalar | 0.572773 | 0.117954   | 4.86x   | -79.41%        |
| 3840x2160 | float32 | divide        | sse42  | 0.572773 | 0.060439   | 9.48x   | -89.45%        |
| 3840x2160 | float32 | divide        | avx2   | 0.572773 | 0.051661   | 11.09x  | -90.98%        |
| 3840x2160 | float32 | overlay       | scalar | 0.796863 | 0.253915   | 3.14x   | -68.14%        |
| 3840x2160 | float32 | overlay       | sse42  | 0.796863 | 0.060748   | 13.12x  | -92.38%        |
| 3840x2160 | float32 | overlay       | avx2   | 0.796863 | 0.049789   | 16.00x  | -93.75%        |
</details>
