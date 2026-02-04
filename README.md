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
| normal        | scalar | 0.194601 | 0.041577   | 4.68x   | -78.63%        |
| normal        | sse42  | 0.194601 | 0.015128   | 12.86x  | -92.23%        |
| normal        | avx2   | 0.194601 | 0.014023   | 13.88x  | -92.79%        |
| soft_light    | scalar | 0.269600 | 0.049026   | 5.50x   | -81.82%        |
| soft_light    | sse42  | 0.269600 | 0.016655   | 16.19x  | -93.82%        |
| soft_light    | avx2   | 0.269600 | 0.015088   | 17.87x  | -94.40%        |
| lighten_only  | scalar | 0.198358 | 0.052808   | 3.76x   | -73.38%        |
| lighten_only  | sse42  | 0.198358 | 0.015187   | 13.06x  | -92.34%        |
| lighten_only  | avx2   | 0.198358 | 0.014617   | 13.57x  | -92.63%        |
| screen        | scalar | 0.211788 | 0.046399   | 4.56x   | -78.09%        |
| screen        | sse42  | 0.211788 | 0.015586   | 13.59x  | -92.64%        |
| screen        | avx2   | 0.211788 | 0.014757   | 14.35x  | -93.03%        |
| dodge         | scalar | 0.214393 | 0.049545   | 4.33x   | -76.89%        |
| dodge         | sse42  | 0.214393 | 0.017346   | 12.36x  | -91.91%        |
| dodge         | avx2   | 0.214393 | 0.015312   | 14.00x  | -92.86%        |
| addition      | scalar | 0.204252 | 0.075454   | 2.71x   | -63.06%        |
| addition      | sse42  | 0.204252 | 0.016326   | 12.51x  | -92.01%        |
| addition      | avx2   | 0.204252 | 0.014879   | 13.73x  | -92.72%        |
| darken_only   | scalar | 0.199977 | 0.053474   | 3.74x   | -73.26%        |
| darken_only   | sse42  | 0.199977 | 0.015375   | 13.01x  | -92.31%        |
| darken_only   | avx2   | 0.199977 | 0.014762   | 13.55x  | -92.62%        |
| multiply      | scalar | 0.207291 | 0.046513   | 4.46x   | -77.56%        |
| multiply      | sse42  | 0.207291 | 0.015529   | 13.35x  | -92.51%        |
| multiply      | avx2   | 0.207291 | 0.015180   | 13.66x  | -92.68%        |
| hard_light    | scalar | 0.302457 | 0.093957   | 3.22x   | -68.94%        |
| hard_light    | sse42  | 0.302457 | 0.017239   | 17.55x  | -94.30%        |
| hard_light    | avx2   | 0.302457 | 0.015327   | 19.73x  | -94.93%        |
| difference    | scalar | 0.268923 | 0.045805   | 5.87x   | -82.97%        |
| difference    | sse42  | 0.268923 | 0.015096   | 17.81x  | -94.39%        |
| difference    | avx2   | 0.268923 | 0.014442   | 18.62x  | -94.63%        |
| subtract      | scalar | 0.201079 | 0.048849   | 4.12x   | -75.71%        |
| subtract      | sse42  | 0.201079 | 0.017004   | 11.83x  | -91.54%        |
| subtract      | avx2   | 0.201079 | 0.015347   | 13.10x  | -92.37%        |
| grain_extract | scalar | 0.211358 | 0.061946   | 3.41x   | -70.69%        |
| grain_extract | sse42  | 0.211358 | 0.016269   | 12.99x  | -92.30%        |
| grain_extract | avx2   | 0.211358 | 0.014980   | 14.11x  | -92.91%        |
| grain_merge   | scalar | 0.208199 | 0.062727   | 3.32x   | -69.87%        |
| grain_merge   | sse42  | 0.208199 | 0.016249   | 12.81x  | -92.20%        |
| grain_merge   | avx2   | 0.208199 | 0.014995   | 13.88x  | -92.80%        |
| divide        | scalar | 0.212321 | 0.048152   | 4.41x   | -77.32%        |
| divide        | sse42  | 0.212321 | 0.016507   | 12.86x  | -92.23%        |
| divide        | avx2   | 0.212321 | 0.015132   | 14.03x  | -92.87%        |
| overlay       | scalar | 0.277336 | 0.088554   | 3.13x   | -68.07%        |
| overlay       | sse42  | 0.277336 | 0.016506   | 16.80x  | -94.05%        |
| overlay       | avx2   | 0.277336 | 0.014991   | 18.50x  | -94.59%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.006338 | 0.001605   | 3.95x   | -74.67%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.006338 | 0.000806   | 7.86x   | -87.28%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.006338 | 0.000770   | 8.23x   | -87.85%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.008549 | 0.002098   | 4.07x   | -75.46%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.008549 | 0.000860   | 9.95x   | -89.95%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.008549 | 0.000805   | 10.63x  | -90.59%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.007013 | 0.001950   | 3.60x   | -72.19%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.007013 | 0.000825   | 8.50x   | -88.23%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.007013 | 0.000746   | 9.40x   | -89.36%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.007259 | 0.001732   | 4.19x   | -76.14%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.007259 | 0.000826   | 8.79x   | -88.62%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.007259 | 0.000793   | 9.15x   | -89.07%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.007110 | 0.001847   | 3.85x   | -74.02%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.007110 | 0.000884   | 8.04x   | -87.56%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.007110 | 0.000810   | 8.78x   | -88.61%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.007220 | 0.002510   | 2.88x   | -65.23%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.007220 | 0.000802   | 9.01x   | -88.90%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.007220 | 0.000755   | 9.57x   | -89.55%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.007118 | 0.001947   | 3.66x   | -72.64%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.007118 | 0.000800   | 8.89x   | -88.75%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.007118 | 0.000779   | 9.13x   | -89.05%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.006915 | 0.001724   | 4.01x   | -75.07%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.006915 | 0.000788   | 8.78x   | -88.61%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.006915 | 0.000762   | 9.07x   | -88.98%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.009345 | 0.003027   | 3.09x   | -67.61%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.009345 | 0.000894   | 10.45x  | -90.43%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.009345 | 0.000839   | 11.14x  | -91.03%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.009134 | 0.001709   | 5.34x   | -81.29%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.009134 | 0.000789   | 11.58x  | -91.37%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.009134 | 0.000758   | 12.05x  | -91.70%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.007382 | 0.001854   | 3.98x   | -74.89%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.007382 | 0.000855   | 8.63x   | -88.42%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.007382 | 0.000803   | 9.19x   | -89.12%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.007301 | 0.002120   | 3.44x   | -70.96%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.007301 | 0.000864   | 8.45x   | -88.16%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.007301 | 0.000813   | 8.98x   | -88.87%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.007637 | 0.002147   | 3.56x   | -71.88%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.007637 | 0.000860   | 8.88x   | -88.74%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.007637 | 0.000799   | 9.56x   | -89.54%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.007885 | 0.001769   | 4.46x   | -77.56%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.007885 | 0.000870   | 9.06x   | -88.96%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.007885 | 0.000799   | 9.87x   | -89.87%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.008754 | 0.002866   | 3.05x   | -67.26%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.008754 | 0.000872   | 10.04x  | -90.04%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.008754 | 0.000912   | 9.60x   | -89.58%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003250 | 0.001320   | 2.46x   | -59.40%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003250 | 0.000202   | 16.11x  | -93.79%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003250 | 0.000183   | 17.78x  | -94.38%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.006886 | 0.001634   | 4.22x   | -76.28%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.006886 | 0.000233   | 29.59x  | -96.62%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.006886 | 0.000198   | 34.75x  | -97.12%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.005629 | 0.001716   | 3.28x   | -69.52%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.005629 | 0.000196   | 28.75x  | -96.52%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.005629 | 0.000193   | 29.18x  | -96.57%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.005569 | 0.001593   | 3.50x   | -71.40%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.005569 | 0.000213   | 26.17x  | -96.18%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.005569 | 0.000201   | 27.67x  | -96.39%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.005832 | 0.001650   | 3.53x   | -71.70%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.005832 | 0.000240   | 24.35x  | -95.89%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.005832 | 0.000198   | 29.41x  | -96.60%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.005853 | 0.001991   | 2.94x   | -65.99%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.005853 | 0.000280   | 20.89x  | -95.21%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.005853 | 0.000205   | 28.52x  | -96.49%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.005574 | 0.001752   | 3.18x   | -68.56%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.005574 | 0.000201   | 27.69x  | -96.39%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.005574 | 0.000196   | 28.49x  | -96.49%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.006022 | 0.001702   | 3.54x   | -71.74%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.006022 | 0.000217   | 27.75x  | -96.40%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.006022 | 0.000198   | 30.44x  | -96.72%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.007676 | 0.002624   | 2.93x   | -65.82%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.007676 | 0.000242   | 31.72x  | -96.85%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.007676 | 0.000202   | 37.97x  | -97.37%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.007430 | 0.001621   | 4.58x   | -78.18%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.007430 | 0.000199   | 37.28x  | -97.32%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.007430 | 0.000192   | 38.64x  | -97.41%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.005978 | 0.001624   | 3.68x   | -72.84%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.005978 | 0.000267   | 22.40x  | -95.54%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.005978 | 0.000209   | 28.59x  | -96.50%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.005986 | 0.001882   | 3.18x   | -68.56%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.005986 | 0.000212   | 28.17x  | -96.45%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.005986 | 0.000195   | 30.73x  | -96.75%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.005486 | 0.001867   | 2.94x   | -65.97%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.005486 | 0.000221   | 24.84x  | -95.97%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.005486 | 0.000198   | 27.74x  | -96.39%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.005889 | 0.001565   | 3.76x   | -73.42%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.005889 | 0.000220   | 26.82x  | -96.27%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.005889 | 0.000193   | 30.44x  | -96.71%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.007146 | 0.002580   | 2.77x   | -63.89%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.007146 | 0.000240   | 29.72x  | -96.63%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.007146 | 0.000201   | 35.48x  | -97.18%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.005724 | 0.000511   | 11.20x  | -91.07%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.005724 | 0.000227   | 25.16x  | -96.03%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.005724 | 0.000145   | 39.35x  | -97.46%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.012984 | 0.000635   | 20.43x  | -95.11%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.012984 | 0.000242   | 53.69x  | -98.14%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.012984 | 0.000190   | 68.37x  | -98.54%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.006998 | 0.000735   | 9.52x   | -89.49%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.006998 | 0.000209   | 33.49x  | -97.01%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.006998 | 0.000182   | 38.53x  | -97.40%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.007373 | 0.000561   | 13.15x  | -92.40%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.007373 | 0.000216   | 34.18x  | -97.07%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.007373 | 0.000179   | 41.18x  | -97.57%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.007407 | 0.000660   | 11.22x  | -91.09%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.007407 | 0.000244   | 30.38x  | -96.71%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.007407 | 0.000190   | 38.97x  | -97.43%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.007310 | 0.001588   | 4.60x   | -78.28%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.007310 | 0.000233   | 31.34x  | -96.81%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.007310 | 0.000196   | 37.33x  | -97.32%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.007159 | 0.000775   | 9.24x   | -89.18%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.007159 | 0.000213   | 33.54x  | -97.02%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.007159 | 0.000182   | 39.44x  | -97.46%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.007100 | 0.000563   | 12.61x  | -92.07%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.007100 | 0.000210   | 33.80x  | -97.04%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.007100 | 0.000181   | 39.26x  | -97.45%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.009017 | 0.001864   | 4.84x   | -79.33%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.009017 | 0.000244   | 36.95x  | -97.29%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.009017 | 0.000183   | 49.35x  | -97.97%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.009021 | 0.000543   | 16.62x  | -93.98%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.009021 | 0.000216   | 41.74x  | -97.60%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.009021 | 0.000179   | 50.37x  | -98.01%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.007354 | 0.000710   | 10.36x  | -90.34%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.007354 | 0.000240   | 30.69x  | -96.74%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.007354 | 0.000188   | 39.07x  | -97.44%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.007327 | 0.001004   | 7.30x   | -86.30%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.007327 | 0.000224   | 32.72x  | -96.94%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.007327 | 0.000301   | 24.38x  | -95.90%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.007107 | 0.001014   | 7.01x   | -85.73%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.007107 | 0.000564   | 12.60x  | -92.07%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.007107 | 0.000180   | 39.50x  | -97.47%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.007889 | 0.000612   | 12.90x  | -92.25%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.007889 | 0.000238   | 33.21x  | -96.99%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.007889 | 0.000188   | 42.04x  | -97.62%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.009238 | 0.001654   | 5.59x   | -82.10%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.009238 | 0.000232   | 39.80x  | -97.49%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.009238 | 0.000209   | 44.19x  | -97.74%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.004486 | 0.000623   | 7.20x   | -86.11%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.004486 | 0.000150   | 29.85x  | -96.65%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.004486 | 0.000154   | 29.11x  | -96.56%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.006834 | 0.000733   | 9.33x   | -89.28%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.006834 | 0.000187   | 36.56x  | -97.27%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.006834 | 0.000181   | 37.69x  | -97.35%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.005665 | 0.000785   | 7.21x   | -86.14%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.005665 | 0.000184   | 30.80x  | -96.75%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.005665 | 0.000182   | 31.05x  | -96.78%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.005634 | 0.000676   | 8.33x   | -88.00%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.005634 | 0.000183   | 30.76x  | -96.75%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.005634 | 0.000183   | 30.85x  | -96.76%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.005449 | 0.000746   | 7.30x   | -86.31%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.005449 | 0.000237   | 23.03x  | -95.66%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.005449 | 0.000186   | 29.35x  | -96.59%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.005830 | 0.001395   | 4.18x   | -76.08%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.005830 | 0.000209   | 27.90x  | -96.42%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.005830 | 0.000196   | 29.71x  | -96.63%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.005427 | 0.000789   | 6.88x   | -85.46%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.005427 | 0.000173   | 31.38x  | -96.81%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.005427 | 0.000192   | 28.31x  | -96.47%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.005516 | 0.000643   | 8.57x   | -88.34%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.005516 | 0.000175   | 31.46x  | -96.82%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.005516 | 0.000177   | 31.18x  | -96.79%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.007440 | 0.001948   | 3.82x   | -73.82%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.007440 | 0.000242   | 30.80x  | -96.75%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.007440 | 0.000200   | 37.24x  | -97.31%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.007150 | 0.000653   | 10.95x  | -90.87%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.007150 | 0.000193   | 37.07x  | -97.30%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.007150 | 0.000197   | 36.34x  | -97.25%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.005580 | 0.000881   | 6.34x   | -84.22%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.005580 | 0.000198   | 28.17x  | -96.45%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.005580 | 0.000188   | 29.68x  | -96.63%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.005494 | 0.001060   | 5.18x   | -80.70%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.005494 | 0.000177   | 31.01x  | -96.77%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.005494 | 0.000180   | 30.55x  | -96.73%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.005557 | 0.001058   | 5.25x   | -80.96%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.005557 | 0.000193   | 28.81x  | -96.53%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.005557 | 0.000179   | 31.05x  | -96.78%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.005620 | 0.000788   | 7.13x   | -85.97%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.005620 | 0.000186   | 30.21x  | -96.69%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.005620 | 0.000181   | 30.97x  | -96.77%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.007223 | 0.001747   | 4.13x   | -75.81%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.007223 | 0.000193   | 37.46x  | -97.33%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.007223 | 0.000179   | 40.34x  | -97.52%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.034286 | 0.006411   | 5.35x   | -81.30%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.034286 | 0.003293   | 10.41x  | -90.39%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.034286 | 0.003105   | 11.04x  | -90.94%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.044984 | 0.007141   | 6.30x   | -84.13%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.044984 | 0.003431   | 13.11x  | -92.37%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.044984 | 0.003189   | 14.11x  | -92.91%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.037292 | 0.007994   | 4.67x   | -78.56%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.037292 | 0.003120   | 11.95x  | -91.63%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.037292 | 0.003038   | 12.28x  | -91.85%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.038429 | 0.006820   | 5.63x   | -82.25%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.038429 | 0.003231   | 11.90x  | -91.59%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.038429 | 0.003090   | 12.44x  | -91.96%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.040764 | 0.009111   | 4.47x   | -77.65%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.040764 | 0.004121   | 9.89x   | -89.89%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.040764 | 0.003524   | 11.57x  | -91.36%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.042894 | 0.010416   | 4.12x   | -75.72%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.042894 | 0.003247   | 13.21x  | -92.43%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.042894 | 0.003366   | 12.74x  | -92.15%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.039824 | 0.007782   | 5.12x   | -80.46%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.039824 | 0.003104   | 12.83x  | -92.21%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.039824 | 0.003332   | 11.95x  | -91.63%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.038551 | 0.007013   | 5.50x   | -81.81%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.038551 | 0.003165   | 12.18x  | -91.79%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.038551 | 0.003140   | 12.28x  | -91.85%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.051785 | 0.011984   | 4.32x   | -76.86%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.051785 | 0.003597   | 14.40x  | -93.05%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.051785 | 0.003413   | 15.17x  | -93.41%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.045581 | 0.006712   | 6.79x   | -85.28%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.045581 | 0.003382   | 13.48x  | -92.58%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.045581 | 0.003015   | 15.12x  | -93.39%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.037692 | 0.006770   | 5.57x   | -82.04%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.037692 | 0.003404   | 11.07x  | -90.97%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.037692 | 0.003230   | 11.67x  | -91.43%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.037972 | 0.008563   | 4.43x   | -77.45%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.037972 | 0.003497   | 10.86x  | -90.79%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.037972 | 0.003177   | 11.95x  | -91.63%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.038100 | 0.008483   | 4.49x   | -77.73%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.038100 | 0.003391   | 11.24x  | -91.10%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.038100 | 0.003425   | 11.12x  | -91.01%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.038418 | 0.007030   | 5.46x   | -81.70%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.038418 | 0.003865   | 9.94x   | -89.94%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.038418 | 0.003281   | 11.71x  | -91.46%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.044851 | 0.011516   | 3.89x   | -74.32%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.044851 | 0.003454   | 12.98x  | -92.30%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.044851 | 0.003348   | 13.40x  | -92.54%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.024605 | 0.005322   | 4.62x   | -78.37%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.024605 | 0.000750   | 32.82x  | -96.95%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.024605 | 0.000712   | 34.57x  | -97.11%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.035962 | 0.006503   | 5.53x   | -81.92%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.035962 | 0.000893   | 40.27x  | -97.52%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.035962 | 0.000789   | 45.56x  | -97.80%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.028737 | 0.006858   | 4.19x   | -76.13%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.028737 | 0.000832   | 34.54x  | -97.10%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.028737 | 0.000752   | 38.23x  | -97.38%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.029777 | 0.006311   | 4.72x   | -78.81%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.029777 | 0.001079   | 27.59x  | -96.38%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.029777 | 0.000785   | 37.95x  | -97.36%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.031686 | 0.006616   | 4.79x   | -79.12%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.031686 | 0.000945   | 33.52x  | -97.02%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.031686 | 0.000786   | 40.31x  | -97.52%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.028762 | 0.007860   | 3.66x   | -72.67%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.028762 | 0.001022   | 28.15x  | -96.45%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.028762 | 0.000795   | 36.16x  | -97.23%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.028937 | 0.006997   | 4.14x   | -75.82%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.028937 | 0.000772   | 37.49x  | -97.33%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.028937 | 0.000760   | 38.06x  | -97.37%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.029360 | 0.006237   | 4.71x   | -78.76%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.029360 | 0.000781   | 37.61x  | -97.34%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.029360 | 0.000758   | 38.71x  | -97.42%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.038171 | 0.010640   | 3.59x   | -72.13%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.038171 | 0.000952   | 40.10x  | -97.51%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.038171 | 0.000783   | 48.75x  | -97.95%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.036145 | 0.006946   | 5.20x   | -80.78%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.036145 | 0.000792   | 45.61x  | -97.81%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.036145 | 0.000754   | 47.92x  | -97.91%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.029339 | 0.006485   | 4.52x   | -77.90%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.029339 | 0.001063   | 27.61x  | -96.38%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.029339 | 0.000826   | 35.50x  | -97.18%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.030157 | 0.007705   | 3.91x   | -74.45%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.030157 | 0.000849   | 35.54x  | -97.19%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.030157 | 0.000784   | 38.49x  | -97.40%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.029691 | 0.007473   | 3.97x   | -74.83%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.029691 | 0.000840   | 35.33x  | -97.17%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.029691 | 0.000770   | 38.55x  | -97.41%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.029472 | 0.006129   | 4.81x   | -79.20%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.029472 | 0.000926   | 31.84x  | -96.86%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.029472 | 0.000788   | 37.41x  | -97.33%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.037357 | 0.010146   | 3.68x   | -72.84%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.037357 | 0.000922   | 40.51x  | -97.53%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.037357 | 0.000801   | 46.66x  | -97.86%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.028329 | 0.002366   | 11.97x  | -91.65%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.028329 | 0.000896   | 31.63x  | -96.84%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.028329 | 0.000627   | 45.17x  | -97.79%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.039893 | 0.002827   | 14.11x  | -92.91%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.039893 | 0.000919   | 43.40x  | -97.70%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.039893 | 0.000743   | 53.70x  | -98.14%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.033553 | 0.003280   | 10.23x  | -90.22%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.033553 | 0.000846   | 39.64x  | -97.48%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.033553 | 0.000800   | 41.95x  | -97.62%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.033816 | 0.002642   | 12.80x  | -92.19%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.033816 | 0.000943   | 35.86x  | -97.21%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.033816 | 0.000965   | 35.04x  | -97.15%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.033746 | 0.002896   | 11.65x  | -91.42%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.033746 | 0.000998   | 33.83x  | -97.04%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.033746 | 0.000758   | 44.54x  | -97.75%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.033447 | 0.006784   | 4.93x   | -79.72%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.033447 | 0.000936   | 35.72x  | -97.20%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.033447 | 0.000770   | 43.46x  | -97.70%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.033297 | 0.003278   | 10.16x  | -90.16%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.033297 | 0.000895   | 37.20x  | -97.31%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.033297 | 0.000753   | 44.21x  | -97.74%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.033873 | 0.002554   | 13.26x  | -92.46%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.033873 | 0.000862   | 39.31x  | -97.46%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.033873 | 0.000734   | 46.14x  | -97.83%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.041963 | 0.007472   | 5.62x   | -82.19%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.041963 | 0.001002   | 41.89x  | -97.61%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.041963 | 0.000776   | 54.05x  | -98.15%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.040905 | 0.002583   | 15.84x  | -93.69%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.040905 | 0.000879   | 46.55x  | -97.85%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.040905 | 0.000750   | 54.55x  | -98.17%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.033053 | 0.003155   | 10.48x  | -90.45%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.033053 | 0.001008   | 32.80x  | -96.95%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.033053 | 0.000804   | 41.13x  | -97.57%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.034356 | 0.004479   | 7.67x   | -86.96%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.034356 | 0.000933   | 36.82x  | -97.28%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.034356 | 0.000755   | 45.52x  | -97.80%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.034057 | 0.004479   | 7.60x   | -86.85%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.034057 | 0.000937   | 36.33x  | -97.25%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.034057 | 0.000755   | 45.14x  | -97.78%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.033656 | 0.002776   | 12.13x  | -91.75%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.033656 | 0.000924   | 36.42x  | -97.25%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.033656 | 0.000731   | 46.03x  | -97.83%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.039711 | 0.006985   | 5.69x   | -82.41%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.039711 | 0.000955   | 41.59x  | -97.60%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.039711 | 0.000750   | 52.93x  | -98.11%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.020535 | 0.002470   | 8.31x   | -87.97%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.020535 | 0.000689   | 29.82x  | -96.65%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.020535 | 0.000707   | 29.03x  | -96.56%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.031211 | 0.002899   | 10.77x  | -90.71%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.031211 | 0.000826   | 37.79x  | -97.35%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.031211 | 0.000799   | 39.04x  | -97.44%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.024007 | 0.003206   | 7.49x   | -86.65%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.024007 | 0.000752   | 31.91x  | -96.87%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.024007 | 0.000820   | 29.29x  | -96.59%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.024895 | 0.002690   | 9.26x   | -89.20%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.024895 | 0.000743   | 33.50x  | -97.02%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.024895 | 0.000773   | 32.19x  | -96.89%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.024828 | 0.003006   | 8.26x   | -87.89%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.024828 | 0.000929   | 26.73x  | -96.26%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.024828 | 0.000802   | 30.95x  | -96.77%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.024286 | 0.005441   | 4.46x   | -77.60%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.024286 | 0.000793   | 30.64x  | -96.74%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.024286 | 0.000793   | 30.62x  | -96.73%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.024157 | 0.003150   | 7.67x   | -86.96%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.024157 | 0.000756   | 31.93x  | -96.87%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.024157 | 0.000789   | 30.60x  | -96.73%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.024598 | 0.002657   | 9.26x   | -89.20%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.024598 | 0.000787   | 31.25x  | -96.80%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.024598 | 0.000788   | 31.20x  | -96.79%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.033976 | 0.007321   | 4.64x   | -78.45%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.033976 | 0.000987   | 34.41x  | -97.09%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.033976 | 0.000807   | 42.08x  | -97.62%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.031740 | 0.002622   | 12.11x  | -91.74%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.031740 | 0.000776   | 40.90x  | -97.56%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.031740 | 0.000782   | 40.61x  | -97.54%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.024193 | 0.003541   | 6.83x   | -85.36%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.024193 | 0.000846   | 28.61x  | -96.50%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.024193 | 0.000845   | 28.64x  | -96.51%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.024717 | 0.004330   | 5.71x   | -82.48%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.024717 | 0.000757   | 32.64x  | -96.94%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.024717 | 0.000750   | 32.94x  | -96.96%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.024916 | 0.004240   | 5.88x   | -82.98%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.024916 | 0.000825   | 30.19x  | -96.69%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.024916 | 0.000801   | 31.09x  | -96.78%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.024765 | 0.002921   | 8.48x   | -88.20%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.024765 | 0.000828   | 29.91x  | -96.66%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.024765 | 0.000803   | 30.83x  | -96.76%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.031875 | 0.006981   | 4.57x   | -78.10%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.031875 | 0.000800   | 39.84x  | -97.49%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.031875 | 0.000762   | 41.82x  | -97.61%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.098626 | 0.025908   | 3.81x   | -73.73%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.098626 | 0.012923   | 7.63x   | -86.90%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.098626 | 0.012448   | 7.92x   | -87.38%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.130322 | 0.028702   | 4.54x   | -77.98%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.130322 | 0.013975   | 9.33x   | -89.28%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.130322 | 0.012767   | 10.21x  | -90.20%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.104340 | 0.030982   | 3.37x   | -70.31%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.104340 | 0.012558   | 8.31x   | -87.96%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.104340 | 0.012292   | 8.49x   | -88.22%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.108119 | 0.028156   | 3.84x   | -73.96%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.108119 | 0.012846   | 8.42x   | -88.12%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.108119 | 0.012454   | 8.68x   | -88.48%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.107207 | 0.028775   | 3.73x   | -73.16%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.107207 | 0.014181   | 7.56x   | -86.77%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.107207 | 0.012957   | 8.27x   | -87.91%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.103996 | 0.040808   | 2.55x   | -60.76%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.103996 | 0.012876   | 8.08x   | -87.62%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.103996 | 0.012504   | 8.32x   | -87.98%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.103301 | 0.031664   | 3.26x   | -69.35%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.103301 | 0.012589   | 8.21x   | -87.81%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.103301 | 0.012313   | 8.39x   | -88.08%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.104903 | 0.028272   | 3.71x   | -73.05%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.104903 | 0.013014   | 8.06x   | -87.59%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.104903 | 0.012393   | 8.46x   | -88.19%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.140432 | 0.047836   | 2.94x   | -65.94%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.140432 | 0.014077   | 9.98x   | -89.98%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.140432 | 0.013066   | 10.75x  | -90.70%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.133142 | 0.027048   | 4.92x   | -79.68%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.133142 | 0.013357   | 9.97x   | -89.97%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.133142 | 0.013402   | 9.93x   | -89.93%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.113732 | 0.026659   | 4.27x   | -76.56%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.113732 | 0.013848   | 8.21x   | -87.82%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.113732 | 0.012781   | 8.90x   | -88.76%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.106637 | 0.033917   | 3.14x   | -68.19%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.106637 | 0.013695   | 7.79x   | -87.16%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.106637 | 0.012755   | 8.36x   | -88.04%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.106143 | 0.034483   | 3.08x   | -67.51%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.106143 | 0.013978   | 7.59x   | -86.83%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.106143 | 0.012800   | 8.29x   | -87.94%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.107868 | 0.028345   | 3.81x   | -73.72%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.107868 | 0.013940   | 7.74x   | -87.08%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.107868 | 0.013134   | 8.21x   | -87.82%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.133268 | 0.046286   | 2.88x   | -65.27%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.133268 | 0.014105   | 9.45x   | -89.42%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.133268 | 0.012898   | 10.33x  | -90.32%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.071598 | 0.021852   | 3.28x   | -69.48%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.071598 | 0.003058   | 23.41x  | -95.73%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.071598 | 0.002889   | 24.78x  | -95.96%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.105589 | 0.026823   | 3.94x   | -74.60%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.105589 | 0.003564   | 29.63x  | -96.62%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.105589 | 0.003228   | 32.71x  | -96.94%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.085761 | 0.028208   | 3.04x   | -67.11%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.085761 | 0.003056   | 28.06x  | -96.44%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.085761 | 0.003111   | 27.56x  | -96.37%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.087230 | 0.025660   | 3.40x   | -70.58%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.087230 | 0.003410   | 25.58x  | -96.09%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.087230 | 0.003212   | 27.16x  | -96.32%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.085710 | 0.026805   | 3.20x   | -68.73%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.085710 | 0.003779   | 22.68x  | -95.59%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.085710 | 0.003271   | 26.20x  | -96.18%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.078078 | 0.031529   | 2.48x   | -59.62%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.078078 | 0.004158   | 18.78x  | -94.67%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.078078 | 0.003202   | 24.38x  | -95.90%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.078355 | 0.027927   | 2.81x   | -64.36%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.078355 | 0.003153   | 24.85x  | -95.98%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.078355 | 0.003048   | 25.71x  | -96.11%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.079700 | 0.025330   | 3.15x   | -68.22%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.079700 | 0.003123   | 25.52x  | -96.08%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.079700 | 0.003098   | 25.73x  | -96.11%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.115082 | 0.044792   | 2.57x   | -61.08%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.115082 | 0.004408   | 26.11x  | -96.17%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.115082 | 0.003166   | 36.35x  | -97.25%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.119510 | 0.025024   | 4.78x   | -79.06%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.119510 | 0.003397   | 35.18x  | -97.16%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.119510 | 0.003032   | 39.41x  | -97.46%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.082464 | 0.024712   | 3.34x   | -70.03%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.082464 | 0.004273   | 19.30x  | -94.82%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.082464 | 0.003383   | 24.37x  | -95.90%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.080266 | 0.030815   | 2.60x   | -61.61%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.080266 | 0.003444   | 23.30x  | -95.71%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.080266 | 0.003165   | 25.36x  | -96.06%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.092006 | 0.044011   | 2.09x   | -52.16%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.092006 | 0.004714   | 19.52x  | -94.88%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.092006 | 0.003727   | 24.69x  | -95.95%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.086159 | 0.025359   | 3.40x   | -70.57%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.086159 | 0.003497   | 24.64x  | -95.94%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.086159 | 0.003128   | 27.55x  | -96.37%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.107245 | 0.040855   | 2.63x   | -61.91%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.107245 | 0.003631   | 29.54x  | -96.61%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.107245 | 0.003195   | 33.56x  | -97.02%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.085272 | 0.008512   | 10.02x  | -90.02%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.085272 | 0.003690   | 23.11x  | -95.67%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.085272 | 0.002622   | 32.52x  | -96.92%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.119664 | 0.010766   | 11.11x  | -91.00%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.119664 | 0.004112   | 29.10x  | -96.56%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.119664 | 0.003053   | 39.19x  | -97.45%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.094611 | 0.013657   | 6.93x   | -85.56%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.094611 | 0.003558   | 26.59x  | -96.24%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.094611 | 0.003195   | 29.61x  | -96.62%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.096017 | 0.009664   | 9.94x   | -89.94%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.096017 | 0.003789   | 25.34x  | -96.05%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.096017 | 0.003024   | 31.75x  | -96.85%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.095069 | 0.010600   | 8.97x   | -88.85%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.095069 | 0.004172   | 22.78x  | -95.61%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.095069 | 0.003097   | 30.70x  | -96.74%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.091842 | 0.026109   | 3.52x   | -71.57%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.091842 | 0.003935   | 23.34x  | -95.72%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.091842 | 0.003356   | 27.36x  | -96.35%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.093563 | 0.012351   | 7.58x   | -86.80%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.093563 | 0.003682   | 25.41x  | -96.06%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.093563 | 0.002999   | 31.19x  | -96.79%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.093122 | 0.009387   | 9.92x   | -89.92%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.093122 | 0.003499   | 26.61x  | -96.24%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.093122 | 0.003019   | 30.85x  | -96.76%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.127889 | 0.028856   | 4.43x   | -77.44%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.127889 | 0.004069   | 31.43x  | -96.82%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.127889 | 0.003080   | 41.52x  | -97.59%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.122242 | 0.009657   | 12.66x  | -92.10%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.122242 | 0.003686   | 33.16x  | -96.98%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.122242 | 0.003011   | 40.59x  | -97.54%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.091346 | 0.011973   | 7.63x   | -86.89%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.091346 | 0.003934   | 23.22x  | -95.69%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.091346 | 0.003102   | 29.45x  | -96.60%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.095942 | 0.016684   | 5.75x   | -82.61%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.095942 | 0.003580   | 26.80x  | -96.27%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.095942 | 0.003137   | 30.58x  | -96.73%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.095049 | 0.016840   | 5.64x   | -82.28%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.095049 | 0.003744   | 25.38x  | -96.06%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.095049 | 0.003244   | 29.30x  | -96.59%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.095452 | 0.010448   | 9.14x   | -89.05%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.095452 | 0.003965   | 24.08x  | -95.85%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.095452 | 0.003098   | 30.81x  | -96.75%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.123245 | 0.027436   | 4.49x   | -77.74%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.123245 | 0.003953   | 31.18x  | -96.79%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.123245 | 0.002984   | 41.30x  | -97.58%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.064886 | 0.009831   | 6.60x   | -84.85%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.064886 | 0.002982   | 21.76x  | -95.40%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.064886 | 0.003321   | 19.54x  | -94.88%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.098004 | 0.011640   | 8.42x   | -88.12%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.098004 | 0.003405   | 28.78x  | -96.53%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.098004 | 0.003759   | 26.07x  | -96.16%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.076296 | 0.012841   | 5.94x   | -83.17%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.076296 | 0.004082   | 18.69x  | -94.65%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.076296 | 0.003492   | 21.85x  | -95.42%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.075975 | 0.010875   | 6.99x   | -85.69%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.075975 | 0.003296   | 23.05x  | -95.66%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.075975 | 0.004082   | 18.61x  | -94.63%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.075317 | 0.012258   | 6.14x   | -83.72%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.075317 | 0.004222   | 17.84x  | -94.39%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.075317 | 0.004023   | 18.72x  | -94.66%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.072612 | 0.022035   | 3.30x   | -69.65%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.072612 | 0.003455   | 21.02x  | -95.24%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.072612 | 0.004047   | 17.94x  | -94.43%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.072256 | 0.012790   | 5.65x   | -82.30%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.072256 | 0.003198   | 22.60x  | -95.57%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.072256 | 0.003919   | 18.44x  | -94.58%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.078941 | 0.010516   | 7.51x   | -86.68%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.078941 | 0.003311   | 23.84x  | -95.81%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.078941 | 0.003549   | 22.24x  | -95.50%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.112627 | 0.029826   | 3.78x   | -73.52%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.112627 | 0.004087   | 27.55x  | -96.37%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.112627 | 0.004285   | 26.29x  | -96.20%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.103437 | 0.010626   | 9.73x   | -89.73%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.103437 | 0.003296   | 31.38x  | -96.81%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.103437 | 0.004032   | 25.65x  | -96.10%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.076911 | 0.014430   | 5.33x   | -81.24%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.076911 | 0.003469   | 22.17x  | -95.49%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.076911 | 0.004081   | 18.85x  | -94.69%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.077890 | 0.017209   | 4.53x   | -77.91%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.077890 | 0.003429   | 22.72x  | -95.60%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.077890 | 0.004091   | 19.04x  | -94.75%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.079664 | 0.017546   | 4.54x   | -77.97%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.079664 | 0.003409   | 23.37x  | -95.72%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.079664 | 0.004295   | 18.55x  | -94.61%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.080249 | 0.011835   | 6.78x   | -85.25%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.080249 | 0.003599   | 22.30x  | -95.52%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.080249 | 0.003874   | 20.72x  | -95.17%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.104725 | 0.028321   | 3.70x   | -72.96%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.104725 | 0.003483   | 30.07x  | -96.67%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.104725 | 0.004151   | 25.23x  | -96.04%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.397586 | 0.105708   | 3.76x   | -73.41%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.397586 | 0.052478   | 7.58x   | -86.80%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.397586 | 0.049844   | 7.98x   | -87.46%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.503369 | 0.115635   | 4.35x   | -77.03%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.503369 | 0.055492   | 9.07x   | -88.98%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.503369 | 0.052196   | 9.64x   | -89.63%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.384179 | 0.125331   | 3.07x   | -67.38%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.384179 | 0.051425   | 7.47x   | -86.61%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.384179 | 0.049192   | 7.81x   | -87.20%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.443229 | 0.113163   | 3.92x   | -74.47%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.443229 | 0.052912   | 8.38x   | -88.06%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.443229 | 0.051280   | 8.64x   | -88.43%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.428553 | 0.114375   | 3.75x   | -73.31%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.428553 | 0.056126   | 7.64x   | -86.90%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.428553 | 0.051068   | 8.39x   | -88.08%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.387673 | 0.161034   | 2.41x   | -58.46%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.387673 | 0.050827   | 7.63x   | -86.89%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.387673 | 0.048693   | 7.96x   | -87.44%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.389974 | 0.123407   | 3.16x   | -68.35%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.389974 | 0.049531   | 7.87x   | -87.30%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.389974 | 0.048538   | 8.03x   | -87.55%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.417741 | 0.112679   | 3.71x   | -73.03%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.417741 | 0.050293   | 8.31x   | -87.96%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.417741 | 0.048717   | 8.57x   | -88.34%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.549170 | 0.188942   | 2.91x   | -65.60%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.549170 | 0.056950   | 9.64x   | -89.63%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.549170 | 0.051865   | 10.59x  | -90.56%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.500671 | 0.107252   | 4.67x   | -78.58%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.500671 | 0.050376   | 9.94x   | -89.94%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.500671 | 0.048002   | 10.43x  | -90.41%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.388008 | 0.106050   | 3.66x   | -72.67%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.388008 | 0.056170   | 6.91x   | -85.52%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.388008 | 0.051975   | 7.47x   | -86.60%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.438351 | 0.137606   | 3.19x   | -68.61%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.438351 | 0.055447   | 7.91x   | -87.35%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.438351 | 0.051300   | 8.54x   | -88.30%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.447665 | 0.141110   | 3.17x   | -68.48%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.447665 | 0.059300   | 7.55x   | -86.75%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.447665 | 0.052797   | 8.48x   | -88.21%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.444860 | 0.117854   | 3.77x   | -73.51%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.444860 | 0.055598   | 8.00x   | -87.50%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.444860 | 0.051068   | 8.71x   | -88.52%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.519336 | 0.184072   | 2.82x   | -64.56%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.519336 | 0.055327   | 9.39x   | -89.35%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.519336 | 0.051067   | 10.17x  | -90.17%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.284631 | 0.084040   | 3.39x   | -70.47%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.284631 | 0.012066   | 23.59x  | -95.76%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.284631 | 0.011440   | 24.88x  | -95.98%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.395978 | 0.104819   | 3.78x   | -73.53%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.395978 | 0.014122   | 28.04x  | -96.43%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.395978 | 0.012358   | 32.04x  | -96.88%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.282161 | 0.109937   | 2.57x   | -61.04%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.282161 | 0.014053   | 20.08x  | -95.02%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.282161 | 0.012961   | 21.77x  | -95.41%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.342469 | 0.103091   | 3.32x   | -69.90%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.342469 | 0.013932   | 24.58x  | -95.93%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.342469 | 0.012510   | 27.37x  | -96.35%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.308825 | 0.107753   | 2.87x   | -65.11%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.308825 | 0.016046   | 19.25x  | -94.80%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.308825 | 0.013324   | 23.18x  | -95.69%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.316964 | 0.128482   | 2.47x   | -59.46%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.316964 | 0.017800   | 17.81x  | -94.38%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.316964 | 0.013287   | 23.86x  | -95.81%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.314451 | 0.117081   | 2.69x   | -62.77%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.314451 | 0.013645   | 23.05x  | -95.66%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.314451 | 0.012770   | 24.62x  | -95.94%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.330089 | 0.102866   | 3.21x   | -68.84%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.330089 | 0.013134   | 25.13x  | -96.02%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.330089 | 0.012868   | 25.65x  | -96.10%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.478146 | 0.167937   | 2.85x   | -64.88%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.478146 | 0.015821   | 30.22x  | -96.69%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.478146 | 0.012768   | 37.45x  | -97.33%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.396690 | 0.100647   | 3.94x   | -74.63%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.396690 | 0.012899   | 30.75x  | -96.75%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.396690 | 0.012321   | 32.20x  | -96.89%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.300429 | 0.099576   | 3.02x   | -66.86%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.300429 | 0.018376   | 16.35x  | -93.88%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.300429 | 0.013716   | 21.90x  | -95.43%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.345513 | 0.122529   | 2.82x   | -64.54%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.345513 | 0.013961   | 24.75x  | -95.96%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.345513 | 0.012928   | 26.73x  | -96.26%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.319651 | 0.122719   | 2.60x   | -61.61%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.319651 | 0.013960   | 22.90x  | -95.63%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.319651 | 0.012543   | 25.48x  | -96.08%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.303238 | 0.101188   | 3.00x   | -66.63%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.303238 | 0.014110   | 21.49x  | -95.35%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.303238 | 0.012398   | 24.46x  | -95.91%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.437852 | 0.163801   | 2.67x   | -62.59%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.437852 | 0.014989   | 29.21x  | -96.58%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.437852 | 0.013295   | 32.93x  | -96.96%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.350851 | 0.038098   | 9.21x   | -89.14%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.350851 | 0.021055   | 16.66x  | -94.00%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.350851 | 0.017992   | 19.50x  | -94.87%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.495510 | 0.046652   | 10.62x  | -90.58%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.495510 | 0.021793   | 22.74x  | -95.60%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.495510 | 0.018090   | 27.39x  | -96.35%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.339782 | 0.052673   | 6.45x   | -84.50%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.339782 | 0.020056   | 16.94x  | -94.10%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.339782 | 0.017777   | 19.11x  | -94.77%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.364555 | 0.042603   | 8.56x   | -88.31%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.364555 | 0.020792   | 17.53x  | -94.30%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.364555 | 0.018162   | 20.07x  | -95.02%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.377410 | 0.048655   | 7.76x   | -87.11%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.377410 | 0.021785   | 17.32x  | -94.23%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.377410 | 0.018261   | 20.67x  | -95.16%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.351636 | 0.109485   | 3.21x   | -68.86%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.351636 | 0.021101   | 16.66x  | -94.00%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.351636 | 0.018226   | 19.29x  | -94.82%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.341839 | 0.052781   | 6.48x   | -84.56%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.341839 | 0.020296   | 16.84x  | -94.06%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.341839 | 0.018226   | 18.76x  | -94.67%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.348366 | 0.042444   | 8.21x   | -87.82%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.348366 | 0.022546   | 15.45x  | -93.53%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.348366 | 0.021065   | 16.54x  | -93.95%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.534638 | 0.121605   | 4.40x   | -77.25%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.534638 | 0.021735   | 24.60x  | -95.93%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.534638 | 0.018927   | 28.25x  | -96.46%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.464146 | 0.042032   | 11.04x  | -90.94%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.464146 | 0.022011   | 21.09x  | -95.26%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.464146 | 0.018810   | 24.68x  | -95.95%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.361308 | 0.052657   | 6.86x   | -85.43%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.361308 | 0.022488   | 16.07x  | -93.78%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.361308 | 0.023065   | 15.66x  | -93.62%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.373949 | 0.073988   | 5.05x   | -80.21%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.373949 | 0.021813   | 17.14x  | -94.17%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.373949 | 0.019415   | 19.26x  | -94.81%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.380928 | 0.076522   | 4.98x   | -79.91%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.380928 | 0.021859   | 17.43x  | -94.26%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.380928 | 0.019101   | 19.94x  | -94.99%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.372394 | 0.046768   | 7.96x   | -87.44%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.372394 | 0.021996   | 16.93x  | -94.09%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.372394 | 0.018202   | 20.46x  | -95.11%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.479338 | 0.116897   | 4.10x   | -75.61%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.479338 | 0.021394   | 22.41x  | -95.54%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.479338 | 0.018279   | 26.22x  | -96.19%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.264243 | 0.047769   | 5.53x   | -81.92%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.264243 | 0.018926   | 13.96x  | -92.84%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.264243 | 0.018427   | 14.34x  | -93.03%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.375047 | 0.057636   | 6.51x   | -84.63%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.375047 | 0.020760   | 18.07x  | -94.46%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.375047 | 0.019975   | 18.78x  | -94.67%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.258052 | 0.060160   | 4.29x   | -76.69%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.258052 | 0.019956   | 12.93x  | -92.27%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.258052 | 0.019731   | 13.08x  | -92.35%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.278623 | 0.052270   | 5.33x   | -81.24%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.278623 | 0.020129   | 13.84x  | -92.78%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.278623 | 0.020241   | 13.77x  | -92.74%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.276700 | 0.057347   | 4.82x   | -79.27%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.276700 | 0.022686   | 12.20x  | -91.80%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.276700 | 0.019987   | 13.84x  | -92.78%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.266116 | 0.097842   | 2.72x   | -63.23%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.266116 | 0.022249   | 11.96x  | -91.64%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.266116 | 0.021201   | 12.55x  | -92.03%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.257048 | 0.060561   | 4.24x   | -76.44%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.257048 | 0.020829   | 12.34x  | -91.90%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.257048 | 0.019556   | 13.14x  | -92.39%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.318400 | 0.054173   | 5.88x   | -82.99%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.318400 | 0.022380   | 14.23x  | -92.97%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.318400 | 0.021051   | 15.13x  | -93.39%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.468529 | 0.129157   | 3.63x   | -72.43%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.468529 | 0.022753   | 20.59x  | -95.14%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.468529 | 0.020093   | 23.32x  | -95.71%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.375847 | 0.050308   | 7.47x   | -86.61%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.375847 | 0.019532   | 19.24x  | -94.80%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.375847 | 0.019209   | 19.57x  | -94.89%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.264138 | 0.065660   | 4.02x   | -75.14%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.264138 | 0.021348   | 12.37x  | -91.92%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.264138 | 0.020086   | 13.15x  | -92.40%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.279623 | 0.078113   | 3.58x   | -72.06%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.279623 | 0.022905   | 12.21x  | -91.81%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.279623 | 0.021153   | 13.22x  | -92.44%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.278194 | 0.078399   | 3.55x   | -71.82%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.278194 | 0.020495   | 13.57x  | -92.63%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.278194 | 0.020125   | 13.82x  | -92.77%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.300516 | 0.054587   | 5.51x   | -81.84%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.300516 | 0.020608   | 14.58x  | -93.14%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.300516 | 0.020572   | 14.61x  | -93.15%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.414794 | 0.120405   | 3.45x   | -70.97%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.414794 | 0.019870   | 20.88x  | -95.21%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.414794 | 0.020540   | 20.19x  | -95.05%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.085360 | 0.022570   | 3.78x   | -73.56%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.085360 | 0.011322   | 7.54x   | -86.74%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.085360 | 0.010880   | 7.85x   | -87.25%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.108644 | 0.025281   | 4.30x   | -76.73%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.108644 | 0.012167   | 8.93x   | -88.80%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.108644 | 0.011463   | 9.48x   | -89.45%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.092088 | 0.027261   | 3.38x   | -70.40%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.092088 | 0.011140   | 8.27x   | -87.90%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.092088 | 0.010830   | 8.50x   | -88.24%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.091723 | 0.024020   | 3.82x   | -73.81%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.091723 | 0.011330   | 8.10x   | -87.65%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.091723 | 0.010820   | 8.48x   | -88.20%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.089003 | 0.025472   | 3.49x   | -71.38%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.089003 | 0.012259   | 7.26x   | -86.23%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.089003 | 0.011240   | 7.92x   | -87.37%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.082402 | 0.034878   | 2.36x   | -57.67%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.082402 | 0.011045   | 7.46x   | -86.60%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.082402 | 0.010623   | 7.76x   | -87.11%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.084715 | 0.026910   | 3.15x   | -68.23%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.084715 | 0.011006   | 7.70x   | -87.01%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.084715 | 0.010484   | 8.08x   | -87.62%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.095388 | 0.026289   | 3.63x   | -72.44%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.095388 | 0.011795   | 8.09x   | -87.64%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.095388 | 0.010825   | 8.81x   | -88.65%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.119959 | 0.041796   | 2.87x   | -65.16%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.119959 | 0.012764   | 9.40x   | -89.36%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.119959 | 0.011943   | 10.04x  | -90.04%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.120173 | 0.026279   | 4.57x   | -78.13%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.120173 | 0.010999   | 10.93x  | -90.85%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.120173 | 0.010764   | 11.16x  | -91.04%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.088308 | 0.023391   | 3.78x   | -73.51%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.088308 | 0.012179   | 7.25x   | -86.21%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.088308 | 0.011103   | 7.95x   | -87.43%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.087642 | 0.029141   | 3.01x   | -66.75%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.087642 | 0.011966   | 7.32x   | -86.35%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.087642 | 0.010931   | 8.02x   | -87.53%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.087276 | 0.029529   | 2.96x   | -66.17%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.087276 | 0.011977   | 7.29x   | -86.28%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.087276 | 0.011135   | 7.84x   | -87.24%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.089775 | 0.024071   | 3.73x   | -73.19%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.089775 | 0.012065   | 7.44x   | -86.56%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.089775 | 0.011183   | 8.03x   | -87.54%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.117933 | 0.040781   | 2.89x   | -65.42%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.117933 | 0.012045   | 9.79x   | -89.79%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.117933 | 0.011170   | 10.56x  | -90.53%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.061446 | 0.018203   | 3.38x   | -70.38%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.061446 | 0.002593   | 23.70x  | -95.78%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.061446 | 0.002481   | 24.77x  | -95.96%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.099140 | 0.023044   | 4.30x   | -76.76%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.099140 | 0.003083   | 32.16x  | -96.89%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.099140 | 0.002729   | 36.33x  | -97.25%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.075457 | 0.024246   | 3.11x   | -67.87%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.075457 | 0.002668   | 28.28x  | -96.46%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.075457 | 0.002652   | 28.45x  | -96.49%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.081839 | 0.022288   | 3.67x   | -72.77%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.081839 | 0.003031   | 27.00x  | -96.30%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.081839 | 0.002754   | 29.72x  | -96.64%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.075508 | 0.023120   | 3.27x   | -69.38%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.075508 | 0.003271   | 23.08x  | -95.67%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.075508 | 0.002748   | 27.48x  | -96.36%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.074114 | 0.027858   | 2.66x   | -62.41%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.074114 | 0.003604   | 20.56x  | -95.14%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.074114 | 0.002865   | 25.87x  | -96.13%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.079725 | 0.024905   | 3.20x   | -68.76%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.079725 | 0.002837   | 28.10x  | -96.44%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.079725 | 0.002766   | 28.82x  | -96.53%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.083671 | 0.023070   | 3.63x   | -72.43%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.083671 | 0.002739   | 30.55x  | -96.73%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.083671 | 0.002651   | 31.56x  | -96.83%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.106780 | 0.037948   | 2.81x   | -64.46%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.106780 | 0.003931   | 27.16x  | -96.32%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.106780 | 0.002757   | 38.74x  | -97.42%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.110585 | 0.025104   | 4.41x   | -77.30%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.110585 | 0.002949   | 37.50x  | -97.33%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.110585 | 0.002688   | 41.15x  | -97.57%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.086049 | 0.022239   | 3.87x   | -74.16%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.086049 | 0.003860   | 22.29x  | -95.51%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.086049 | 0.002917   | 29.50x  | -96.61%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.076723 | 0.026208   | 2.93x   | -65.84%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.076723 | 0.002943   | 26.07x  | -96.16%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.076723 | 0.002717   | 28.24x  | -96.46%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.075385 | 0.026578   | 2.84x   | -64.74%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.075385 | 0.003058   | 24.65x  | -95.94%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.075385 | 0.002769   | 27.23x  | -96.33%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.079956 | 0.028170   | 2.84x   | -64.77%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.079956 | 0.003057   | 26.15x  | -96.18%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.079956 | 0.002798   | 28.57x  | -96.50%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.109246 | 0.036019   | 3.03x   | -67.03%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.109246 | 0.003203   | 34.10x  | -97.07%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.109246 | 0.002792   | 39.12x  | -97.44%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.080940 | 0.007133   | 11.35x  | -91.19%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.080940 | 0.003130   | 25.86x  | -96.13%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.080940 | 0.002094   | 38.65x  | -97.41%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.100160 | 0.008581   | 11.67x  | -91.43%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.100160 | 0.003359   | 29.81x  | -96.65%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.100160 | 0.002599   | 38.54x  | -97.41%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.077301 | 0.010864   | 7.12x   | -85.95%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.077301 | 0.003212   | 24.06x  | -95.84%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.077301 | 0.002758   | 28.03x  | -96.43%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.083250 | 0.007825   | 10.64x  | -90.60%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.083250 | 0.003189   | 26.11x  | -96.17%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.083250 | 0.002572   | 32.37x  | -96.91%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.078561 | 0.008840   | 8.89x   | -88.75%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.078561 | 0.003429   | 22.91x  | -95.63%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.078561 | 0.002602   | 30.20x  | -96.69%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.078429 | 0.022416   | 3.50x   | -71.42%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.078429 | 0.003321   | 23.62x  | -95.77%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.078429 | 0.002647   | 29.63x  | -96.63%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.081820 | 0.010141   | 8.07x   | -87.61%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.081820 | 0.003030   | 27.01x  | -96.30%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.081820 | 0.002530   | 32.34x  | -96.91%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.080571 | 0.007790   | 10.34x  | -90.33%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.080571 | 0.003068   | 26.26x  | -96.19%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.080571 | 0.002548   | 31.63x  | -96.84%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.113856 | 0.024818   | 4.59x   | -78.20%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.113856 | 0.003527   | 32.28x  | -96.90%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.113856 | 0.002667   | 42.69x  | -97.66%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.104962 | 0.007681   | 13.67x  | -92.68%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.104962 | 0.002991   | 35.09x  | -97.15%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.104962 | 0.002594   | 40.46x  | -97.53%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.078344 | 0.009885   | 7.93x   | -87.38%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.078344 | 0.003331   | 23.52x  | -95.75%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.078344 | 0.002655   | 29.51x  | -96.61%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.080597 | 0.014228   | 5.66x   | -82.35%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.080597 | 0.003216   | 25.06x  | -96.01%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.080597 | 0.002555   | 31.55x  | -96.83%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.093022 | 0.018870   | 4.93x   | -79.71%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.093022 | 0.003911   | 23.78x  | -95.80%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.093022 | 0.003555   | 26.17x  | -96.18%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.087538 | 0.008824   | 9.92x   | -89.92%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.087538 | 0.003435   | 25.49x  | -96.08%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.087538 | 0.002776   | 31.54x  | -96.83%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.103653 | 0.023436   | 4.42x   | -77.39%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.103653 | 0.003398   | 30.51x  | -96.72%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.103653 | 0.002567   | 40.38x  | -97.52%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.056640 | 0.008692   | 6.52x   | -84.65%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.056640 | 0.002992   | 18.93x  | -94.72%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.056640 | 0.002759   | 20.53x  | -95.13%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.098139 | 0.010673   | 9.19x   | -89.12%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.098139 | 0.003025   | 32.44x  | -96.92%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.098139 | 0.003639   | 26.97x  | -96.29%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.067956 | 0.011049   | 6.15x   | -83.74%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.067956 | 0.002627   | 25.86x  | -96.13%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.067956 | 0.003075   | 22.10x  | -95.47%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.067173 | 0.009562   | 7.02x   | -85.76%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.067173 | 0.002844   | 23.62x  | -95.77%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.067173 | 0.003474   | 19.34x  | -94.83%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.066462 | 0.010393   | 6.39x   | -84.36%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.066462 | 0.003153   | 21.08x  | -95.26%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.066462 | 0.002942   | 22.59x  | -95.57%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.062636 | 0.019173   | 3.27x   | -69.39%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.062636 | 0.002887   | 21.70x  | -95.39%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.062636 | 0.003480   | 18.00x  | -94.44%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.064405 | 0.010873   | 5.92x   | -83.12%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.064405 | 0.002613   | 24.65x  | -95.94%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.064405 | 0.003243   | 19.86x  | -94.96%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.065683 | 0.009023   | 7.28x   | -86.26%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.065683 | 0.002672   | 24.59x  | -95.93%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.065683 | 0.003306   | 19.87x  | -94.97%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.097903 | 0.025750   | 3.80x   | -73.70%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.097903 | 0.003301   | 29.66x  | -96.63%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.097903 | 0.003685   | 26.57x  | -96.24%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.098511 | 0.009436   | 10.44x  | -90.42%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.098511 | 0.002879   | 34.22x  | -97.08%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.098511 | 0.003889   | 25.33x  | -96.05%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.067955 | 0.012811   | 5.30x   | -81.15%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.067955 | 0.003571   | 19.03x  | -94.75%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.067955 | 0.004386   | 15.49x  | -93.55%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.084357 | 0.017500   | 4.82x   | -79.26%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.084357 | 0.002994   | 28.17x  | -96.45%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.084357 | 0.003834   | 22.00x  | -95.45%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.071392 | 0.015283   | 4.67x   | -78.59%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.071392 | 0.002876   | 24.82x  | -95.97%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.071392 | 0.004157   | 17.17x  | -94.18%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.072578 | 0.010915   | 6.65x   | -84.96%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.072578 | 0.003039   | 23.88x  | -95.81%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.072578 | 0.004113   | 17.64x  | -94.33%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.094151 | 0.025229   | 3.73x   | -73.20%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.094151 | 0.002876   | 32.74x  | -96.95%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.094151 | 0.003986   | 23.62x  | -95.77%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.179613 | 0.051675   | 3.48x   | -71.23%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.179613 | 0.025578   | 7.02x   | -85.76%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.179613 | 0.024540   | 7.32x   | -86.34%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.293012 | 0.066063   | 4.44x   | -77.45%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.293012 | 0.029698   | 9.87x   | -89.86%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.293012 | 0.027227   | 10.76x  | -90.71%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.218037 | 0.064492   | 3.38x   | -70.42%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.218037 | 0.027263   | 8.00x   | -87.50%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.218037 | 0.025884   | 8.42x   | -88.13%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.204677 | 0.055230   | 3.71x   | -73.02%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.204677 | 0.025804   | 7.93x   | -87.39%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.204677 | 0.024564   | 8.33x   | -88.00%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.197415 | 0.059515   | 3.32x   | -69.85%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.197415 | 0.028197   | 7.00x   | -85.72%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.197415 | 0.031593   | 6.25x   | -84.00%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.200324 | 0.078840   | 2.54x   | -60.64%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.200324 | 0.025189   | 7.95x   | -87.43%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.200324 | 0.024009   | 8.34x   | -88.01%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.184907 | 0.061125   | 3.03x   | -66.94%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.184907 | 0.025172   | 7.35x   | -86.39%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.184907 | 0.023729   | 7.79x   | -87.17%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.185390 | 0.053997   | 3.43x   | -70.87%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.185390 | 0.024945   | 7.43x   | -86.54%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.185390 | 0.024091   | 7.70x   | -87.01%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.253051 | 0.093241   | 2.71x   | -63.15%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.253051 | 0.027928   | 9.06x   | -88.96%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.253051 | 0.025766   | 9.82x   | -89.82%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.250578 | 0.054645   | 4.59x   | -78.19%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.250578 | 0.024706   | 10.14x  | -90.14%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.250578 | 0.024115   | 10.39x  | -90.38%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.201285 | 0.054566   | 3.69x   | -72.89%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.201285 | 0.028129   | 7.16x   | -86.03%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.201285 | 0.026137   | 7.70x   | -87.02%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.212826 | 0.067583   | 3.15x   | -68.24%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.212826 | 0.027391   | 7.77x   | -87.13%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.212826 | 0.025857   | 8.23x   | -87.85%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.204514 | 0.072634   | 2.82x   | -64.48%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.204514 | 0.028341   | 7.22x   | -86.14%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.204514 | 0.026640   | 7.68x   | -86.97%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.204414 | 0.056205   | 3.64x   | -72.50%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.204414 | 0.027654   | 7.39x   | -86.47%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.204414 | 0.025513   | 8.01x   | -87.52%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.260317 | 0.092100   | 2.83x   | -64.62%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.260317 | 0.030445   | 8.55x   | -88.30%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.260317 | 0.025906   | 10.05x  | -90.05%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.133248 | 0.041876   | 3.18x   | -68.57%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.133248 | 0.006044   | 22.05x  | -95.46%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.133248 | 0.005694   | 23.40x  | -95.73%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.194130 | 0.052539   | 3.69x   | -72.94%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.194130 | 0.006968   | 27.86x  | -96.41%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.194130 | 0.006225   | 31.19x  | -96.79%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.143107 | 0.054889   | 2.61x   | -61.64%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.143107 | 0.006388   | 22.40x  | -95.54%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.143107 | 0.006082   | 23.53x  | -95.75%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.156994 | 0.050281   | 3.12x   | -67.97%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.156994 | 0.006609   | 23.76x  | -95.79%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.156994 | 0.006127   | 25.62x  | -96.10%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.147274 | 0.051983   | 2.83x   | -64.70%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.147274 | 0.007408   | 19.88x  | -94.97%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.147274 | 0.006224   | 23.66x  | -95.77%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.157226 | 0.062508   | 2.52x   | -60.24%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.157226 | 0.008668   | 18.14x  | -94.49%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.157226 | 0.006358   | 24.73x  | -95.96%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.150690 | 0.054902   | 2.74x   | -63.57%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.150690 | 0.006125   | 24.60x  | -95.94%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.150690 | 0.006032   | 24.98x  | -96.00%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.145884 | 0.049439   | 2.95x   | -66.11%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.145884 | 0.006202   | 23.52x  | -95.75%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.145884 | 0.005973   | 24.42x  | -95.91%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.209168 | 0.082346   | 2.54x   | -60.63%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.209168 | 0.007473   | 27.99x  | -96.43%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.209168 | 0.006244   | 33.50x  | -97.01%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.202446 | 0.049533   | 4.09x   | -75.53%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.202446 | 0.006248   | 32.40x  | -96.91%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.202446 | 0.005962   | 33.96x  | -97.06%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.151532 | 0.051862   | 2.92x   | -65.78%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.151532 | 0.008755   | 17.31x  | -94.22%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.151532 | 0.006699   | 22.62x  | -95.58%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.170919 | 0.059932   | 2.85x   | -64.94%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.170919 | 0.007538   | 22.67x  | -95.59%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.170919 | 0.006392   | 26.74x  | -96.26%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.170885 | 0.059877   | 2.85x   | -64.96%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.170885 | 0.006667   | 25.63x  | -96.10%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.170885 | 0.006156   | 27.76x  | -96.40%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.158048 | 0.052417   | 3.02x   | -66.83%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.158048 | 0.007292   | 21.67x  | -95.39%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.158048 | 0.006231   | 25.37x  | -96.06%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.197327 | 0.080165   | 2.46x   | -59.37%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.197327 | 0.007126   | 27.69x  | -96.39%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.197327 | 0.006229   | 31.68x  | -96.84%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.158207 | 0.016667   | 9.49x   | -89.47%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.158207 | 0.007012   | 22.56x  | -95.57%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.158207 | 0.004959   | 31.90x  | -96.87%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.220955 | 0.020382   | 10.84x  | -90.78%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.220955 | 0.007431   | 29.73x  | -96.64%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.220955 | 0.005952   | 37.12x  | -97.31%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.173428 | 0.024670   | 7.03x   | -85.78%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.173428 | 0.007178   | 24.16x  | -95.86%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.173428 | 0.005895   | 29.42x  | -96.60%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.188827 | 0.018903   | 9.99x   | -89.99%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.188827 | 0.007317   | 25.81x  | -96.12%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.188827 | 0.006162   | 30.65x  | -96.74%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.174902 | 0.021524   | 8.13x   | -87.69%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.174902 | 0.008049   | 21.73x  | -95.40%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.174902 | 0.005981   | 29.24x  | -96.58%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.169073 | 0.051605   | 3.28x   | -69.48%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.169073 | 0.007385   | 22.89x  | -95.63%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.169073 | 0.006044   | 27.98x  | -96.43%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.171012 | 0.023855   | 7.17x   | -86.05%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.171012 | 0.006815   | 25.09x  | -96.01%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.171012 | 0.005797   | 29.50x  | -96.61%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.169033 | 0.018523   | 9.13x   | -89.04%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.169033 | 0.007131   | 23.70x  | -95.78%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.169033 | 0.005967   | 28.33x  | -96.47%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.243328 | 0.057340   | 4.24x   | -76.44%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.243328 | 0.007867   | 30.93x  | -96.77%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.243328 | 0.006003   | 40.53x  | -97.53%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.226461 | 0.018646   | 12.15x  | -91.77%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.226461 | 0.006972   | 32.48x  | -96.92%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.226461 | 0.005957   | 38.02x  | -97.37%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.168608 | 0.023096   | 7.30x   | -86.30%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.168608 | 0.007741   | 21.78x  | -95.41%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.168608 | 0.005984   | 28.18x  | -96.45%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.170513 | 0.033091   | 5.15x   | -80.59%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.170513 | 0.007262   | 23.48x  | -95.74%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.170513 | 0.005900   | 28.90x  | -96.54%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.172654 | 0.033362   | 5.18x   | -80.68%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.172654 | 0.007395   | 23.35x  | -95.72%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.172654 | 0.005842   | 29.55x  | -96.62%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.180396 | 0.020725   | 8.70x   | -88.51%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.180396 | 0.007924   | 22.77x  | -95.61%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.180396 | 0.006423   | 28.08x  | -96.44%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.225755 | 0.053501   | 4.22x   | -76.30%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.225755 | 0.007709   | 29.28x  | -96.59%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.225755 | 0.005856   | 38.55x  | -97.41%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.120684 | 0.019378   | 6.23x   | -83.94%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.120684 | 0.005364   | 22.50x  | -95.56%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.120684 | 0.005440   | 22.19x  | -95.49%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.182649 | 0.022568   | 8.09x   | -87.64%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.182649 | 0.006080   | 30.04x  | -96.67%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.182649 | 0.006412   | 28.48x  | -96.49%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.131214 | 0.025252   | 5.20x   | -80.76%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.131214 | 0.006982   | 18.79x  | -94.68%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.131214 | 0.008109   | 16.18x  | -93.82%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.137185 | 0.021332   | 6.43x   | -84.45%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.137185 | 0.006057   | 22.65x  | -95.59%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.137185 | 0.007242   | 18.94x  | -94.72%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.137219 | 0.023811   | 5.76x   | -82.65%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.137219 | 0.007153   | 19.18x  | -94.79%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.137219 | 0.006295   | 21.80x  | -95.41%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.131281 | 0.043149   | 3.04x   | -67.13%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.131281 | 0.006308   | 20.81x  | -95.20%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.131281 | 0.006966   | 18.85x  | -94.69%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.129508 | 0.024569   | 5.27x   | -81.03%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.129508 | 0.006211   | 20.85x  | -95.20%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.129508 | 0.006750   | 19.19x  | -94.79%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.133200 | 0.020435   | 6.52x   | -84.66%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.133200 | 0.005750   | 23.16x  | -95.68%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.133200 | 0.007542   | 17.66x  | -94.34%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.197319 | 0.057847   | 3.41x   | -70.68%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.197319 | 0.007310   | 26.99x  | -96.30%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.197319 | 0.008015   | 24.62x  | -95.94%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.190377 | 0.020816   | 9.15x   | -89.07%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.190377 | 0.006092   | 31.25x  | -96.80%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.190377 | 0.007067   | 26.94x  | -96.29%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.131435 | 0.027685   | 4.75x   | -78.94%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.131435 | 0.006369   | 20.64x  | -95.15%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.131435 | 0.008045   | 16.34x  | -93.88%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.136841 | 0.033666   | 4.06x   | -75.40%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.136841 | 0.005930   | 23.08x  | -95.67%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.136841 | 0.008473   | 16.15x  | -93.81%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.137470 | 0.033521   | 4.10x   | -75.62%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.137470 | 0.005877   | 23.39x  | -95.72%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.137470 | 0.006569   | 20.93x  | -95.22%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.156188 | 0.023196   | 6.73x   | -85.15%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.156188 | 0.006501   | 24.03x  | -95.84%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.156188 | 0.007738   | 20.18x  | -95.05%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.188345 | 0.056112   | 3.36x   | -70.21%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.188345 | 0.006358   | 29.62x  | -96.62%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.188345 | 0.007666   | 24.57x  | -95.93%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.345066 | 0.094333   | 3.66x   | -72.66%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.345066 | 0.045386   | 7.60x   | -86.85%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.345066 | 0.043474   | 7.94x   | -87.40%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.428927 | 0.100708   | 4.26x   | -76.52%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.428927 | 0.048374   | 8.87x   | -88.72%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.428927 | 0.044918   | 9.55x   | -89.53%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.333575 | 0.120974   | 2.76x   | -63.73%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.333575 | 0.043666   | 7.64x   | -86.91%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.333575 | 0.042283   | 7.89x   | -87.32%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.343069 | 0.095795   | 3.58x   | -72.08%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.343069 | 0.044908   | 7.64x   | -86.91%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.343069 | 0.043223   | 7.94x   | -87.40%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.399892 | 0.106035   | 3.77x   | -73.48%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.399892 | 0.055219   | 7.24x   | -86.19%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.399892 | 0.048160   | 8.30x   | -87.96%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.356405 | 0.141775   | 2.51x   | -60.22%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.356405 | 0.045725   | 7.79x   | -87.17%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.356405 | 0.045459   | 7.84x   | -87.24%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.362947 | 0.110037   | 3.30x   | -69.68%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.362947 | 0.045400   | 7.99x   | -87.49%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.362947 | 0.044620   | 8.13x   | -87.71%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.357533 | 0.096704   | 3.70x   | -72.95%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.357533 | 0.048013   | 7.45x   | -86.57%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.357533 | 0.057944   | 6.17x   | -83.79%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.522394 | 0.191298   | 2.73x   | -63.38%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.522394 | 0.051565   | 10.13x  | -90.13%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.522394 | 0.046445   | 11.25x  | -91.11%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.454189 | 0.094716   | 4.80x   | -79.15%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.454189 | 0.043624   | 10.41x  | -90.40%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.454189 | 0.042414   | 10.71x  | -90.66%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.346570 | 0.094894   | 3.65x   | -72.62%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.346570 | 0.049897   | 6.95x   | -85.60%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.346570 | 0.046002   | 7.53x   | -86.73%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.386483 | 0.120103   | 3.22x   | -68.92%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.386483 | 0.048695   | 7.94x   | -87.40%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.386483 | 0.045175   | 8.56x   | -88.31%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.358272 | 0.119050   | 3.01x   | -66.77%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.358272 | 0.048049   | 7.46x   | -86.59%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.358272 | 0.044311   | 8.09x   | -87.63%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.360629 | 0.101153   | 3.57x   | -71.95%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.360629 | 0.049783   | 7.24x   | -86.20%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.360629 | 0.049029   | 7.36x   | -86.40%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.473868 | 0.164375   | 2.88x   | -65.31%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.473868 | 0.048995   | 9.67x   | -89.66%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.473868 | 0.045196   | 10.48x  | -90.46%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.246852 | 0.076152   | 3.24x   | -69.15%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.246852 | 0.010465   | 23.59x  | -95.76%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.246852 | 0.010196   | 24.21x  | -95.87%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.363176 | 0.092226   | 3.94x   | -74.61%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.363176 | 0.012939   | 28.07x  | -96.44%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.363176 | 0.011020   | 32.96x  | -96.97%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.254475 | 0.096980   | 2.62x   | -61.89%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.254475 | 0.010591   | 24.03x  | -95.84%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.254475 | 0.010612   | 23.98x  | -95.83%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.263347 | 0.090416   | 2.91x   | -65.67%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.263347 | 0.011833   | 22.26x  | -95.51%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.263347 | 0.010856   | 24.26x  | -95.88%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.260777 | 0.092069   | 2.83x   | -64.69%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.260777 | 0.013135   | 19.85x  | -94.96%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.260777 | 0.010956   | 23.80x  | -95.80%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.250441 | 0.109900   | 2.28x   | -56.12%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.250441 | 0.014637   | 17.11x  | -94.16%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.250441 | 0.011209   | 22.34x  | -95.52%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.242996 | 0.098196   | 2.47x   | -59.59%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.242996 | 0.010772   | 22.56x  | -95.57%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.242996 | 0.010546   | 23.04x  | -95.66%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.267146 | 0.087637   | 3.05x   | -67.20%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.267146 | 0.011065   | 24.14x  | -95.86%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.267146 | 0.010634   | 25.12x  | -96.02%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.390062 | 0.149302   | 2.61x   | -61.72%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.390062 | 0.013766   | 28.33x  | -96.47%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.390062 | 0.011335   | 34.41x  | -97.09%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.364632 | 0.088639   | 4.11x   | -75.69%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.364632 | 0.010970   | 33.24x  | -96.99%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.364632 | 0.010666   | 34.19x  | -97.07%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.285353 | 0.097022   | 2.94x   | -66.00%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.285353 | 0.015577   | 18.32x  | -94.54%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.285353 | 0.012255   | 23.28x  | -95.71%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.273353 | 0.107070   | 2.55x   | -60.83%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.273353 | 0.012238   | 22.34x  | -95.52%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.273353 | 0.010992   | 24.87x  | -95.98%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.264927 | 0.107926   | 2.45x   | -59.26%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.264927 | 0.012097   | 21.90x  | -95.43%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.264927 | 0.010996   | 24.09x  | -95.85%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.274030 | 0.087217   | 3.14x   | -68.17%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.274030 | 0.012509   | 21.91x  | -95.44%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.274030 | 0.010799   | 25.37x  | -96.06%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.366286 | 0.144161   | 2.54x   | -60.64%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.366286 | 0.013334   | 27.47x  | -96.36%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.366286 | 0.011563   | 31.68x  | -96.84%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.307616 | 0.028067   | 10.96x  | -90.88%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.307616 | 0.012680   | 24.26x  | -95.88%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.307616 | 0.008827   | 34.85x  | -97.13%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.404756 | 0.035083   | 11.54x  | -91.33%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.404756 | 0.013508   | 29.96x  | -96.66%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.404756 | 0.010822   | 37.40x  | -97.33%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.296453 | 0.040592   | 7.30x   | -86.31%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.296453 | 0.012495   | 23.73x  | -95.79%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.296453 | 0.010591   | 27.99x  | -96.43%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.328804 | 0.033423   | 9.84x   | -89.83%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.328804 | 0.013246   | 24.82x  | -95.97%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.328804 | 0.011226   | 29.29x  | -96.59%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.327215 | 0.036893   | 8.87x   | -88.73%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.327215 | 0.014497   | 22.57x  | -95.57%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.327215 | 0.010824   | 30.23x  | -96.69%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.321471 | 0.089449   | 3.59x   | -72.18%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.321471 | 0.013335   | 24.11x  | -95.85%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.321471 | 0.010932   | 29.41x  | -96.60%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.310860 | 0.044426   | 7.00x   | -85.71%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.310860 | 0.012915   | 24.07x  | -95.85%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.310860 | 0.010555   | 29.45x  | -96.60%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.320153 | 0.031898   | 10.04x  | -90.04%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.320153 | 0.012961   | 24.70x  | -95.95%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.320153 | 0.010544   | 30.36x  | -96.71%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.449438 | 0.103448   | 4.34x   | -76.98%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.449438 | 0.014273   | 31.49x  | -96.82%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.449438 | 0.011134   | 40.37x  | -97.52%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.399503 | 0.031243   | 12.79x  | -92.18%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.399503 | 0.012292   | 32.50x  | -96.92%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.399503 | 0.010793   | 37.02x  | -97.30%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.298135 | 0.039398   | 7.57x   | -86.79%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.298135 | 0.013547   | 22.01x  | -95.46%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.298135 | 0.010787   | 27.64x  | -96.38%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.343342 | 0.060820   | 5.65x   | -82.29%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.343342 | 0.013993   | 24.54x  | -95.92%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.343342 | 0.010736   | 31.98x  | -96.87%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.321328 | 0.058155   | 5.53x   | -81.90%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.321328 | 0.013144   | 24.45x  | -95.91%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.321328 | 0.010705   | 30.02x  | -96.67%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.330646 | 0.035284   | 9.37x   | -89.33%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.330646 | 0.013651   | 24.22x  | -95.87%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.330646 | 0.011296   | 29.27x  | -96.58%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.460671 | 0.096964   | 4.75x   | -78.95%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.460671 | 0.014398   | 32.00x  | -96.87%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.460671 | 0.011291   | 40.80x  | -97.55%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.257547 | 0.042505   | 6.06x   | -83.50%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.257547 | 0.018122   | 14.21x  | -92.96%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.257547 | 0.018082   | 14.24x  | -92.98%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.376180 | 0.048574   | 7.74x   | -87.09%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.376180 | 0.019268   | 19.52x  | -94.88%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.376180 | 0.018891   | 19.91x  | -94.98%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.279000 | 0.054154   | 5.15x   | -80.59%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.279000 | 0.019532   | 14.28x  | -93.00%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.279000 | 0.018984   | 14.70x  | -93.20%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.296284 | 0.043963   | 6.74x   | -85.16%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.296284 | 0.019065   | 15.54x  | -93.57%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.296284 | 0.017750   | 16.69x  | -94.01%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.292291 | 0.049884   | 5.86x   | -82.93%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.292291 | 0.019952   | 14.65x  | -93.17%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.292291 | 0.017593   | 16.61x  | -93.98%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.270147 | 0.084145   | 3.21x   | -68.85%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.270147 | 0.019571   | 13.80x  | -92.76%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.270147 | 0.018813   | 14.36x  | -93.04%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.242292 | 0.052576   | 4.61x   | -78.30%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.242292 | 0.018652   | 12.99x  | -92.30%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.242292 | 0.018119   | 13.37x  | -92.52%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.239078 | 0.042286   | 5.65x   | -82.31%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.239078 | 0.016349   | 14.62x  | -93.16%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.239078 | 0.016186   | 14.77x  | -93.23%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.374526 | 0.108556   | 3.45x   | -71.02%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.374526 | 0.019168   | 19.54x  | -94.88%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.374526 | 0.018950   | 19.76x  | -94.94%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.335906 | 0.042787   | 7.85x   | -87.26%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.335906 | 0.017765   | 18.91x  | -94.71%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.335906 | 0.018780   | 17.89x  | -94.41%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.231622 | 0.054398   | 4.26x   | -76.51%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.231622 | 0.017472   | 13.26x  | -92.46%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.231622 | 0.016728   | 13.85x  | -92.78%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.239501 | 0.064980   | 3.69x   | -72.87%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.239501 | 0.016522   | 14.50x  | -93.10%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.239501 | 0.016496   | 14.52x  | -93.11%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.261942 | 0.065566   | 4.00x   | -74.97%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.261942 | 0.018145   | 14.44x  | -93.07%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.261942 | 0.017321   | 15.12x  | -93.39%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.276036 | 0.049018   | 5.63x   | -82.24%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.276036 | 0.021559   | 12.80x  | -92.19%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.276036 | 0.017204   | 16.04x  | -93.77%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.346461 | 0.103984   | 3.33x   | -69.99%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.346461 | 0.019170   | 18.07x  | -94.47%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.346461 | 0.017805   | 19.46x  | -94.86%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.746626 | 0.210211   | 3.55x   | -71.85%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.746626 | 0.102219   | 7.30x   | -86.31%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.746626 | 0.098152   | 7.61x   | -86.85%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 0.973048 | 0.234555   | 4.15x   | -75.89%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 0.973048 | 0.110909   | 8.77x   | -88.60%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 0.973048 | 0.103707   | 9.38x   | -89.34%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.741259 | 0.241871   | 3.06x   | -67.37%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.741259 | 0.097362   | 7.61x   | -86.87%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.741259 | 0.098163   | 7.55x   | -86.76%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.791492 | 0.215443   | 3.67x   | -72.78%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.791492 | 0.100035   | 7.91x   | -87.36%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.791492 | 0.096927   | 8.17x   | -87.75%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.790101 | 0.225666   | 3.50x   | -71.44%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.790101 | 0.112420   | 7.03x   | -85.77%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.790101 | 0.103476   | 7.64x   | -86.90%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.788405 | 0.337046   | 2.34x   | -57.25%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.788405 | 0.105033   | 7.51x   | -86.68%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.788405 | 0.099678   | 7.91x   | -87.36%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.764564 | 0.258484   | 2.96x   | -66.19%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.764564 | 0.099315   | 7.70x   | -87.01%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.764564 | 0.103341   | 7.40x   | -86.48%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.794125 | 0.238870   | 3.32x   | -69.92%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.794125 | 0.104393   | 7.61x   | -86.85%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.794125 | 0.099510   | 7.98x   | -87.47%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.095836 | 0.376679   | 2.91x   | -65.63%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.095836 | 0.112956   | 9.70x   | -89.69%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.095836 | 0.105673   | 10.37x  | -90.36%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.962260 | 0.223460   | 4.31x   | -76.78%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.962260 | 0.100246   | 9.60x   | -89.58%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.962260 | 0.095557   | 10.07x  | -90.07%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.746823 | 0.211572   | 3.53x   | -71.67%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.746823 | 0.108011   | 6.91x   | -85.54%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.746823 | 0.100017   | 7.47x   | -86.61%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.774166 | 0.267903   | 2.89x   | -65.39%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.774166 | 0.108622   | 7.13x   | -85.97%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.774166 | 0.100615   | 7.69x   | -87.00%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.759438 | 0.269519   | 2.82x   | -64.51%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.759438 | 0.107609   | 7.06x   | -85.83%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.759438 | 0.100744   | 7.54x   | -86.73%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.776983 | 0.222581   | 3.49x   | -71.35%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.776983 | 0.110037   | 7.06x   | -85.84%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.776983 | 0.102374   | 7.59x   | -86.82%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 0.978494 | 0.367301   | 2.66x   | -62.46%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 0.978494 | 0.109621   | 8.93x   | -88.80%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 0.978494 | 0.101064   | 9.68x   | -89.67%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.544748 | 0.168321   | 3.24x   | -69.10%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.544748 | 0.023506   | 23.18x  | -95.69%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.544748 | 0.022593   | 24.11x  | -95.85%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.759830 | 0.210416   | 3.61x   | -72.31%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.759830 | 0.027682   | 27.45x  | -96.36%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.759830 | 0.024651   | 30.82x  | -96.76%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.537592 | 0.218403   | 2.46x   | -59.37%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.537592 | 0.024046   | 22.36x  | -95.53%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.537592 | 0.023757   | 22.63x  | -95.58%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.571761 | 0.206317   | 2.77x   | -63.92%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.571761 | 0.028002   | 20.42x  | -95.10%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.571761 | 0.025193   | 22.69x  | -95.59%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.664718 | 0.217009   | 3.06x   | -67.35%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.664718 | 0.031029   | 21.42x  | -95.33%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.664718 | 0.025113   | 26.47x  | -96.22%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.589144 | 0.258149   | 2.28x   | -56.18%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.589144 | 0.032318   | 18.23x  | -94.51%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.589144 | 0.025112   | 23.46x  | -95.74%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.563719 | 0.228327   | 2.47x   | -59.50%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.563719 | 0.026231   | 21.49x  | -95.35%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.563719 | 0.024483   | 23.02x  | -95.66%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.600044 | 0.200015   | 3.00x   | -66.67%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.600044 | 0.024995   | 24.01x  | -95.83%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.600044 | 0.023820   | 25.19x  | -96.03%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.943742 | 0.369989   | 2.55x   | -60.80%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.943742 | 0.029992   | 31.47x  | -96.82%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.943742 | 0.024845   | 37.98x  | -97.37%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.785148 | 0.201382   | 3.90x   | -74.35%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.785148 | 0.024724   | 31.76x  | -96.85%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.785148 | 0.023757   | 33.05x  | -96.97%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.547724 | 0.191496   | 2.86x   | -65.04%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.547724 | 0.033494   | 16.35x  | -93.88%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.547724 | 0.025986   | 21.08x  | -95.26%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.558178 | 0.239041   | 2.34x   | -57.17%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.558178 | 0.026484   | 21.08x  | -95.26%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.558178 | 0.024385   | 22.89x  | -95.63%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.575053 | 0.252330   | 2.28x   | -56.12%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.575053 | 0.027344   | 21.03x  | -95.24%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.575053 | 0.025062   | 22.95x  | -95.64%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.622309 | 0.211567   | 2.94x   | -66.00%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.622309 | 0.028731   | 21.66x  | -95.38%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.622309 | 0.025309   | 24.59x  | -95.93%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.829849 | 0.321902   | 2.58x   | -61.21%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.829849 | 0.028348   | 29.27x  | -96.58%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.829849 | 0.024727   | 33.56x  | -97.02%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.645911 | 0.072219   | 8.94x   | -88.82%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.645911 | 0.037956   | 17.02x  | -94.12%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.645911 | 0.028554   | 22.62x  | -95.58%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.856237 | 0.087178   | 9.82x   | -89.82%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.856237 | 0.040406   | 21.19x  | -95.28%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.856237 | 0.033782   | 25.35x  | -96.05%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.642439 | 0.102828   | 6.25x   | -83.99%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.642439 | 0.039736   | 16.17x  | -93.81%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.642439 | 0.033026   | 19.45x  | -94.86%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.660981 | 0.081762   | 8.08x   | -87.63%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.660981 | 0.038379   | 17.22x  | -94.19%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.660981 | 0.033603   | 19.67x  | -94.92%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.698607 | 0.092230   | 7.57x   | -86.80%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.698607 | 0.041818   | 16.71x  | -94.01%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.698607 | 0.033842   | 20.64x  | -95.16%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.654031 | 0.213593   | 3.06x   | -67.34%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.654031 | 0.041249   | 15.86x  | -93.69%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.654031 | 0.034133   | 19.16x  | -94.78%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.662579 | 0.102432   | 6.47x   | -84.54%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.662579 | 0.039416   | 16.81x  | -94.05%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.662579 | 0.033061   | 20.04x  | -95.01%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.637406 | 0.079093   | 8.06x   | -87.59%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.637406 | 0.036800   | 17.32x  | -94.23%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.637406 | 0.033008   | 19.31x  | -94.82%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.930471 | 0.232667   | 4.00x   | -74.99%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.930471 | 0.040226   | 23.13x  | -95.68%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.930471 | 0.033303   | 27.94x  | -96.42%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.852499 | 0.078578   | 10.85x  | -90.78%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.852499 | 0.037131   | 22.96x  | -95.64%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.852499 | 0.032757   | 26.02x  | -96.16%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.666783 | 0.099845   | 6.68x   | -85.03%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.666783 | 0.042672   | 15.63x  | -93.60%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.666783 | 0.035218   | 18.93x  | -94.72%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.706050 | 0.136829   | 5.16x   | -80.62%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.706050 | 0.039464   | 17.89x  | -94.41%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.706050 | 0.032610   | 21.65x  | -95.38%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.634862 | 0.135969   | 4.67x   | -78.58%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.634862 | 0.038299   | 16.58x  | -93.97%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.634862 | 0.032575   | 19.49x  | -94.87%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.647205 | 0.086681   | 7.47x   | -86.61%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.647205 | 0.039472   | 16.40x  | -93.90%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.647205 | 0.033107   | 19.55x  | -94.88%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.881636 | 0.220389   | 4.00x   | -75.00%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.881636 | 0.039423   | 22.36x  | -95.53%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.881636 | 0.032717   | 26.95x  | -96.29%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.511467 | 0.090112   | 5.68x   | -82.38%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.511467 | 0.035533   | 14.39x  | -93.05%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.511467 | 0.034626   | 14.77x  | -93.23%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.757429 | 0.103819   | 7.30x   | -86.29%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.757429 | 0.043263   | 17.51x  | -94.29%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.757429 | 0.036459   | 20.77x  | -95.19%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.509986 | 0.111003   | 4.59x   | -78.23%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.509986 | 0.035400   | 14.41x  | -93.06%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.509986 | 0.036597   | 13.94x  | -92.82%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.560839 | 0.099707   | 5.62x   | -82.22%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.560839 | 0.038578   | 14.54x  | -93.12%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.560839 | 0.037810   | 14.83x  | -93.26%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.549017 | 0.107885   | 5.09x   | -80.35%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.549017 | 0.042492   | 12.92x  | -92.26%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.549017 | 0.037142   | 14.78x  | -93.23%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.526003 | 0.184738   | 2.85x   | -64.88%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.526003 | 0.038253   | 13.75x  | -92.73%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.526003 | 0.036214   | 14.52x  | -93.12%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.500256 | 0.114371   | 4.37x   | -77.14%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.500256 | 0.041659   | 12.01x  | -91.67%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.500256 | 0.037983   | 13.17x  | -92.41%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.535804 | 0.094577   | 5.67x   | -82.35%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.535804 | 0.039753   | 13.48x  | -92.58%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.535804 | 0.038705   | 13.84x  | -92.78%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.904887 | 0.247783   | 3.65x   | -72.62%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.904887 | 0.043537   | 20.78x  | -95.19%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.904887 | 0.037235   | 24.30x  | -95.89%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.744510 | 0.096844   | 7.69x   | -86.99%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.744510 | 0.036703   | 20.28x  | -95.07%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.744510 | 0.035952   | 20.71x  | -95.17%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.509094 | 0.122259   | 4.16x   | -75.98%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.509094 | 0.037735   | 13.49x  | -92.59%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.509094 | 0.036915   | 13.79x  | -92.75%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.510486 | 0.162167   | 3.15x   | -68.23%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.510486 | 0.039560   | 12.90x  | -92.25%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.510486 | 0.036778   | 13.88x  | -92.80%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.522092 | 0.146701   | 3.56x   | -71.90%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.522092 | 0.035895   | 14.54x  | -93.12%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.522092 | 0.035577   | 14.68x  | -93.19%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.533203 | 0.102850   | 5.18x   | -80.71%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.533203 | 0.036163   | 14.74x  | -93.22%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.533203 | 0.035906   | 14.85x  | -93.27%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.734805 | 0.234762   | 3.13x   | -68.05%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.734805 | 0.037311   | 19.69x  | -94.92%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.734805 | 0.035601   | 20.64x  | -95.16%        |
</details>
<!-- PERF_RESULTS_END -->
