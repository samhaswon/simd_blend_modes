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
| normal        | scalar | 0.196542 | 0.042276   | 4.65x   | -78.49%        |
| normal        | sse42  | 0.196542 | 0.021300   | 9.23x   | -89.16%        |
| normal        | avx2   | 0.196542 | 0.021446   | 9.16x   | -89.09%        |
| soft_light    | scalar | 0.270178 | 0.050696   | 5.33x   | -81.24%        |
| soft_light    | sse42  | 0.270178 | 0.022373   | 12.08x  | -91.72%        |
| soft_light    | avx2   | 0.270178 | 0.021060   | 12.83x  | -92.21%        |
| lighten_only  | scalar | 0.196856 | 0.054992   | 3.58x   | -72.06%        |
| lighten_only  | sse42  | 0.196856 | 0.021316   | 9.24x   | -89.17%        |
| lighten_only  | avx2   | 0.196856 | 0.020744   | 9.49x   | -89.46%        |
| screen        | scalar | 0.210263 | 0.048266   | 4.36x   | -77.04%        |
| screen        | sse42  | 0.210263 | 0.022024   | 9.55x   | -89.53%        |
| screen        | avx2   | 0.210263 | 0.020773   | 10.12x  | -90.12%        |
| dodge         | scalar | 0.210616 | 0.050163   | 4.20x   | -76.18%        |
| dodge         | sse42  | 0.210616 | 0.022895   | 9.20x   | -89.13%        |
| dodge         | avx2   | 0.210616 | 0.021098   | 9.98x   | -89.98%        |
| addition      | scalar | 0.199914 | 0.075566   | 2.65x   | -62.20%        |
| addition      | sse42  | 0.199914 | 0.022708   | 8.80x   | -88.64%        |
| addition      | avx2   | 0.199914 | 0.020966   | 9.54x   | -89.51%        |
| darken_only   | scalar | 0.197377 | 0.055244   | 3.57x   | -72.01%        |
| darken_only   | sse42  | 0.197377 | 0.021404   | 9.22x   | -89.16%        |
| darken_only   | avx2   | 0.197377 | 0.020745   | 9.51x   | -89.49%        |
| multiply      | scalar | 0.201587 | 0.048110   | 4.19x   | -76.13%        |
| multiply      | sse42  | 0.201587 | 0.021653   | 9.31x   | -89.26%        |
| multiply      | avx2   | 0.201587 | 0.020628   | 9.77x   | -89.77%        |
| hard_light    | scalar | 0.296998 | 0.095924   | 3.10x   | -67.70%        |
| hard_light    | sse42  | 0.296998 | 0.023164   | 12.82x  | -92.20%        |
| hard_light    | avx2   | 0.296998 | 0.020956   | 14.17x  | -92.94%        |
| difference    | scalar | 0.270532 | 0.047608   | 5.68x   | -82.40%        |
| difference    | sse42  | 0.270532 | 0.021428   | 12.63x  | -92.08%        |
| difference    | avx2   | 0.270532 | 0.020478   | 13.21x  | -92.43%        |
| subtract      | scalar | 0.199823 | 0.049253   | 4.06x   | -75.35%        |
| subtract      | sse42  | 0.199823 | 0.022650   | 8.82x   | -88.67%        |
| subtract      | avx2   | 0.199823 | 0.021005   | 9.51x   | -89.49%        |
| grain_extract | scalar | 0.206590 | 0.063866   | 3.23x   | -69.09%        |
| grain_extract | sse42  | 0.206590 | 0.022142   | 9.33x   | -89.28%        |
| grain_extract | avx2   | 0.206590 | 0.020779   | 9.94x   | -89.94%        |
| grain_merge   | scalar | 0.207314 | 0.063604   | 3.26x   | -69.32%        |
| grain_merge   | sse42  | 0.207314 | 0.021920   | 9.46x   | -89.43%        |
| grain_merge   | avx2   | 0.207314 | 0.020765   | 9.98x   | -89.98%        |
| divide        | scalar | 0.210411 | 0.049962   | 4.21x   | -76.26%        |
| divide        | sse42  | 0.210411 | 0.022265   | 9.45x   | -89.42%        |
| divide        | avx2   | 0.210411 | 0.020622   | 10.20x  | -90.20%        |
| overlay       | scalar | 0.275802 | 0.091657   | 3.01x   | -66.77%        |
| overlay       | sse42  | 0.275802 | 0.022698   | 12.15x  | -91.77%        |
| overlay       | avx2   | 0.275802 | 0.021202   | 13.01x  | -92.31%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.004687 | 0.001639   | 2.86x   | -65.02%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.004687 | 0.001592   | 2.94x   | -66.04%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.004687 | 0.001625   | 2.89x   | -65.34%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.008059 | 0.002028   | 3.97x   | -74.84%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.008059 | 0.001624   | 4.96x   | -79.85%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.008059 | 0.001386   | 5.82x   | -82.80%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.005999 | 0.002161   | 2.78x   | -63.98%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.005999 | 0.001693   | 3.54x   | -71.77%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.005999 | 0.001522   | 3.94x   | -74.62%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.006671 | 0.002024   | 3.30x   | -69.67%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.006671 | 0.001485   | 4.49x   | -77.75%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.006671 | 0.001403   | 4.75x   | -78.97%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.005888 | 0.001936   | 3.04x   | -67.13%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.005888 | 0.001528   | 3.85x   | -74.05%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.005888 | 0.001396   | 4.22x   | -76.30%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.006012 | 0.002591   | 2.32x   | -56.91%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.006012 | 0.001472   | 4.08x   | -75.51%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.006012 | 0.001421   | 4.23x   | -76.37%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.005822 | 0.002353   | 2.47x   | -59.58%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.005822 | 0.001450   | 4.02x   | -75.10%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.005822 | 0.001370   | 4.25x   | -76.46%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.005701 | 0.001990   | 2.87x   | -65.10%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.005701 | 0.001439   | 3.96x   | -74.75%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.005701 | 0.001406   | 4.06x   | -75.34%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.007697 | 0.003533   | 2.18x   | -54.09%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.007697 | 0.001595   | 4.83x   | -79.28%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.007697 | 0.001478   | 5.21x   | -80.79%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.007648 | 0.001927   | 3.97x   | -74.80%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.007648 | 0.001539   | 4.97x   | -79.87%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.007648 | 0.001420   | 5.38x   | -81.43%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.005958 | 0.001685   | 3.54x   | -71.72%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.005958 | 0.001481   | 4.02x   | -75.14%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.005958 | 0.001436   | 4.15x   | -75.90%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.005721 | 0.002235   | 2.56x   | -60.93%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.005721 | 0.001594   | 3.59x   | -72.14%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.005721 | 0.001381   | 4.14x   | -75.86%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.005801 | 0.002301   | 2.52x   | -60.34%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.005801 | 0.001464   | 3.96x   | -74.76%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.005801 | 0.001415   | 4.10x   | -75.60%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.006370 | 0.001968   | 3.24x   | -69.10%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.006370 | 0.001553   | 4.10x   | -75.62%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.006370 | 0.001414   | 4.51x   | -77.81%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.007488 | 0.003052   | 2.45x   | -59.25%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.007488 | 0.001478   | 5.07x   | -80.27%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.007488 | 0.001522   | 4.92x   | -79.68%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003319 | 0.001487   | 2.23x   | -55.20%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003319 | 0.000195   | 16.99x  | -94.12%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003319 | 0.000172   | 19.25x  | -94.80%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.006619 | 0.002210   | 3.00x   | -66.62%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.006619 | 0.000225   | 29.42x  | -96.60%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.006619 | 0.000223   | 29.69x  | -96.63%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.005358 | 0.001783   | 3.00x   | -66.72%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.005358 | 0.000201   | 26.71x  | -96.26%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.005358 | 0.000217   | 24.69x  | -95.95%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.004874 | 0.001666   | 2.93x   | -65.82%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.004874 | 0.000270   | 18.08x  | -94.47%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.004874 | 0.000218   | 22.33x  | -95.52%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.005456 | 0.001880   | 2.90x   | -65.53%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.005456 | 0.000268   | 20.34x  | -95.08%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.005456 | 0.000226   | 24.16x  | -95.86%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.004789 | 0.002045   | 2.34x   | -57.29%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.004789 | 0.000276   | 17.32x  | -94.23%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.004789 | 0.000221   | 21.66x  | -95.38%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.004653 | 0.001870   | 2.49x   | -59.80%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.004653 | 0.000201   | 23.18x  | -95.69%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.004653 | 0.000217   | 21.44x  | -95.34%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.004599 | 0.001628   | 2.82x   | -64.59%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.004599 | 0.000217   | 21.20x  | -95.28%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.004599 | 0.000221   | 20.82x  | -95.20%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.007905 | 0.002951   | 2.68x   | -62.67%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.007905 | 0.000264   | 29.92x  | -96.66%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.007905 | 0.000226   | 34.92x  | -97.14%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.006798 | 0.001685   | 4.03x   | -75.21%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.006798 | 0.000241   | 28.19x  | -96.45%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.006798 | 0.000224   | 30.35x  | -96.71%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.005250 | 0.002335   | 2.25x   | -55.52%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.005250 | 0.000299   | 17.57x  | -94.31%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.005250 | 0.000290   | 18.13x  | -94.48%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.006562 | 0.002463   | 2.66x   | -62.47%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.006562 | 0.000228   | 28.84x  | -96.53%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.006562 | 0.000233   | 28.20x  | -96.45%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.005062 | 0.002103   | 2.41x   | -58.46%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.005062 | 0.000217   | 23.34x  | -95.71%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.005062 | 0.000239   | 21.22x  | -95.29%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.005575 | 0.001801   | 3.09x   | -67.69%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.005575 | 0.000222   | 25.11x  | -96.02%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.005575 | 0.000224   | 24.86x  | -95.98%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.006791 | 0.002742   | 2.48x   | -59.63%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.006791 | 0.000230   | 29.56x  | -96.62%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.006791 | 0.000233   | 29.20x  | -96.58%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.003875 | 0.000518   | 7.49x   | -86.64%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.003875 | 0.000204   | 18.96x  | -94.73%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.003875 | 0.000156   | 24.81x  | -95.97%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.007133 | 0.000698   | 10.21x  | -90.21%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.007133 | 0.000265   | 26.90x  | -96.28%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.007133 | 0.000187   | 38.04x  | -97.37%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.005230 | 0.000795   | 6.58x   | -84.80%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.005230 | 0.000248   | 21.11x  | -95.26%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.005230 | 0.000194   | 27.02x  | -96.30%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.005597 | 0.000574   | 9.75x   | -89.74%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.005597 | 0.000250   | 22.40x  | -95.54%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.005597 | 0.000196   | 28.51x  | -96.49%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.005680 | 0.000647   | 8.78x   | -88.61%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.005680 | 0.000251   | 22.63x  | -95.58%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.005680 | 0.000195   | 29.17x  | -96.57%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.005843 | 0.001616   | 3.62x   | -72.34%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.005843 | 0.000245   | 23.87x  | -95.81%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.005843 | 0.000198   | 29.58x  | -96.62%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.005386 | 0.000748   | 7.20x   | -86.11%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.005386 | 0.000282   | 19.13x  | -94.77%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.005386 | 0.000192   | 28.10x  | -96.44%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.006795 | 0.000783   | 8.68x   | -88.47%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.006795 | 0.000272   | 25.01x  | -96.00%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.006795 | 0.000220   | 30.90x  | -96.76%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.007671 | 0.001888   | 4.06x   | -75.38%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.007671 | 0.000272   | 28.25x  | -96.46%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.007671 | 0.000228   | 33.70x  | -97.03%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.007723 | 0.000558   | 13.85x  | -92.78%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.007723 | 0.000228   | 33.89x  | -97.05%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.007723 | 0.000184   | 42.01x  | -97.62%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.005780 | 0.000714   | 8.10x   | -87.66%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.005780 | 0.000248   | 23.34x  | -95.72%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.005780 | 0.000195   | 29.69x  | -96.63%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.005458 | 0.001106   | 4.94x   | -79.75%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.005458 | 0.000269   | 20.31x  | -95.08%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.005458 | 0.000186   | 29.37x  | -96.60%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.005382 | 0.001052   | 5.11x   | -80.45%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.005382 | 0.000228   | 23.64x  | -95.77%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.005382 | 0.000188   | 28.70x  | -96.52%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.005654 | 0.000640   | 8.83x   | -88.68%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.005654 | 0.000239   | 23.63x  | -95.77%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.005654 | 0.000234   | 24.16x  | -95.86%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.007190 | 0.001762   | 4.08x   | -75.50%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.007190 | 0.000239   | 30.08x  | -96.68%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.007190 | 0.000190   | 37.78x  | -97.35%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.003182 | 0.000641   | 4.96x   | -79.85%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.003182 | 0.000196   | 16.24x  | -93.84%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.003182 | 0.000217   | 14.70x  | -93.20%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.005781 | 0.000719   | 8.04x   | -87.56%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.005781 | 0.000190   | 30.47x  | -96.72%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.005781 | 0.000187   | 30.88x  | -96.76%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.004379 | 0.000871   | 5.03x   | -80.12%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.004379 | 0.000175   | 25.07x  | -96.01%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.004379 | 0.000185   | 23.72x  | -95.78%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.004652 | 0.000708   | 6.57x   | -84.78%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.004652 | 0.000187   | 24.85x  | -95.98%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.004652 | 0.000242   | 19.25x  | -94.80%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.004737 | 0.000756   | 6.27x   | -84.04%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.004737 | 0.000241   | 19.67x  | -94.92%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.004737 | 0.000186   | 25.53x  | -96.08%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.004239 | 0.001399   | 3.03x   | -66.99%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.004239 | 0.000203   | 20.93x  | -95.22%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.004239 | 0.000195   | 21.72x  | -95.40%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.004817 | 0.000817   | 5.90x   | -83.04%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.004817 | 0.000166   | 29.03x  | -96.56%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.004817 | 0.000183   | 26.35x  | -96.21%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.004402 | 0.000662   | 6.66x   | -84.97%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.004402 | 0.000174   | 25.32x  | -96.05%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.004402 | 0.000193   | 22.76x  | -95.61%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.006379 | 0.001919   | 3.32x   | -69.91%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.006379 | 0.000242   | 26.36x  | -96.21%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.006379 | 0.000191   | 33.40x  | -97.01%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.006501 | 0.000715   | 9.10x   | -89.01%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.006501 | 0.000173   | 37.58x  | -97.34%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.006501 | 0.000208   | 31.33x  | -96.81%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.004099 | 0.000887   | 4.62x   | -78.35%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.004099 | 0.000199   | 20.60x  | -95.15%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.004099 | 0.000192   | 21.34x  | -95.31%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.004243 | 0.001082   | 3.92x   | -74.50%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.004243 | 0.000180   | 23.64x  | -95.77%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.004243 | 0.000188   | 22.53x  | -95.56%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.004299 | 0.001087   | 3.95x   | -74.71%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.004299 | 0.000182   | 23.60x  | -95.76%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.004299 | 0.000193   | 22.31x  | -95.52%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.004626 | 0.000757   | 6.11x   | -83.65%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.004626 | 0.000181   | 25.55x  | -96.09%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.004626 | 0.000183   | 25.29x  | -96.05%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.005950 | 0.001782   | 3.34x   | -70.05%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.005950 | 0.000203   | 29.35x  | -96.59%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.005950 | 0.000209   | 28.48x  | -96.49%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.035422 | 0.006572   | 5.39x   | -81.45%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.035422 | 0.005805   | 6.10x   | -83.61%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.035422 | 0.006493   | 5.46x   | -81.67%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.045704 | 0.008004   | 5.71x   | -82.49%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.045704 | 0.006258   | 7.30x   | -86.31%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.045704 | 0.005698   | 8.02x   | -87.53%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.038257 | 0.008632   | 4.43x   | -77.44%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.038257 | 0.005717   | 6.69x   | -85.06%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.038257 | 0.005662   | 6.76x   | -85.20%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.040698 | 0.010989   | 3.70x   | -73.00%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.040698 | 0.008841   | 4.60x   | -78.28%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.040698 | 0.005538   | 7.35x   | -86.39%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.038801 | 0.008138   | 4.77x   | -79.03%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.038801 | 0.006077   | 6.38x   | -84.34%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.038801 | 0.005678   | 6.83x   | -85.37%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.037454 | 0.010522   | 3.56x   | -71.91%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.037454 | 0.005873   | 6.38x   | -84.32%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.037454 | 0.005617   | 6.67x   | -85.00%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.039779 | 0.009131   | 4.36x   | -77.05%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.039779 | 0.005871   | 6.78x   | -85.24%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.039779 | 0.005608   | 7.09x   | -85.90%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.038661 | 0.007768   | 4.98x   | -79.91%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.038661 | 0.005967   | 6.48x   | -84.57%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.038661 | 0.005742   | 6.73x   | -85.15%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.047655 | 0.013173   | 3.62x   | -72.36%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.047655 | 0.006139   | 7.76x   | -87.12%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.047655 | 0.005729   | 8.32x   | -87.98%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.045327 | 0.007656   | 5.92x   | -83.11%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.045327 | 0.006060   | 7.48x   | -86.63%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.045327 | 0.005617   | 8.07x   | -87.61%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.039500 | 0.007037   | 5.61x   | -82.19%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.039500 | 0.006161   | 6.41x   | -84.40%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.039500 | 0.005737   | 6.88x   | -85.48%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.040304 | 0.009725   | 4.14x   | -75.87%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.040304 | 0.005963   | 6.76x   | -85.21%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.040304 | 0.005701   | 7.07x   | -85.85%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.039228 | 0.009151   | 4.29x   | -76.67%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.039228 | 0.005965   | 6.58x   | -84.79%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.039228 | 0.005594   | 7.01x   | -85.74%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.039094 | 0.007784   | 5.02x   | -80.09%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.039094 | 0.006056   | 6.46x   | -84.51%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.039094 | 0.005621   | 6.96x   | -85.62%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.046690 | 0.012326   | 3.79x   | -73.60%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.046690 | 0.006143   | 7.60x   | -86.84%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.046690 | 0.005768   | 8.09x   | -87.65%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.027237 | 0.005488   | 4.96x   | -79.85%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.027237 | 0.000768   | 35.48x  | -97.18%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.027237 | 0.000692   | 39.38x  | -97.46%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.037521 | 0.007202   | 5.21x   | -80.81%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.037521 | 0.000870   | 43.14x  | -97.68%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.037521 | 0.000910   | 41.24x  | -97.58%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.032359 | 0.007148   | 4.53x   | -77.91%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.032359 | 0.000845   | 38.29x  | -97.39%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.032359 | 0.000860   | 37.61x  | -97.34%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.031462 | 0.006519   | 4.83x   | -79.28%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.031462 | 0.000873   | 36.03x  | -97.22%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.031462 | 0.000905   | 34.77x  | -97.12%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.031156 | 0.006682   | 4.66x   | -78.55%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.031156 | 0.000943   | 33.05x  | -96.97%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.031156 | 0.000883   | 35.29x  | -97.17%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.030014 | 0.008067   | 3.72x   | -73.12%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.030014 | 0.001089   | 27.57x  | -96.37%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.030014 | 0.000873   | 34.39x  | -97.09%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.029892 | 0.007100   | 4.21x   | -76.25%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.029892 | 0.000793   | 37.70x  | -97.35%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.029892 | 0.000863   | 34.62x  | -97.11%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.030334 | 0.006527   | 4.65x   | -78.48%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.030334 | 0.000783   | 38.75x  | -97.42%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.030334 | 0.000852   | 35.61x  | -97.19%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.039453 | 0.011164   | 3.53x   | -71.70%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.039453 | 0.000985   | 40.05x  | -97.50%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.039453 | 0.000864   | 45.64x  | -97.81%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.038285 | 0.006562   | 5.83x   | -82.86%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.038285 | 0.000808   | 47.41x  | -97.89%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.038285 | 0.000879   | 43.56x  | -97.70%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.030604 | 0.006305   | 4.85x   | -79.40%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.030604 | 0.001123   | 27.24x  | -96.33%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.030604 | 0.000898   | 34.09x  | -97.07%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.031238 | 0.008227   | 3.80x   | -73.66%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.031238 | 0.000840   | 37.18x  | -97.31%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.031238 | 0.000867   | 36.04x  | -97.23%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.030643 | 0.007928   | 3.87x   | -74.13%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.030643 | 0.000836   | 36.65x  | -97.27%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.030643 | 0.000870   | 35.24x  | -97.16%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.031348 | 0.006696   | 4.68x   | -78.64%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.031348 | 0.000901   | 34.80x  | -97.13%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.031348 | 0.000934   | 33.55x  | -97.02%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.039036 | 0.010852   | 3.60x   | -72.20%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.039036 | 0.000902   | 43.26x  | -97.69%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.039036 | 0.000874   | 44.69x  | -97.76%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.029239 | 0.002093   | 13.97x  | -92.84%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.029239 | 0.000861   | 33.95x  | -97.05%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.029239 | 0.000776   | 37.67x  | -97.35%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.040401 | 0.002504   | 16.14x  | -93.80%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.040401 | 0.000963   | 41.93x  | -97.62%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.040401 | 0.000806   | 50.15x  | -98.01%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.033591 | 0.002953   | 11.38x  | -91.21%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.033591 | 0.000936   | 35.90x  | -97.21%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.033591 | 0.000773   | 43.46x  | -97.70%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.035071 | 0.002298   | 15.26x  | -93.45%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.035071 | 0.001171   | 29.94x  | -96.66%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.035071 | 0.000803   | 43.68x  | -97.71%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.034471 | 0.002567   | 13.43x  | -92.55%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.034471 | 0.001008   | 34.20x  | -97.08%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.034471 | 0.000775   | 44.50x  | -97.75%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.033473 | 0.006449   | 5.19x   | -80.73%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.033473 | 0.000981   | 34.13x  | -97.07%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.033473 | 0.000846   | 39.56x  | -97.47%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.033687 | 0.002998   | 11.24x  | -91.10%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.033687 | 0.000926   | 36.40x  | -97.25%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.033687 | 0.000815   | 41.33x  | -97.58%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.032844 | 0.002633   | 12.47x  | -91.98%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.032844 | 0.000905   | 36.30x  | -97.24%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.032844 | 0.000829   | 39.61x  | -97.48%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.042199 | 0.007345   | 5.75x   | -82.59%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.042199 | 0.000996   | 42.36x  | -97.64%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.042199 | 0.000810   | 52.07x  | -98.08%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.042035 | 0.002296   | 18.31x  | -94.54%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.042035 | 0.001052   | 39.97x  | -97.50%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.042035 | 0.000781   | 53.82x  | -98.14%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.033596 | 0.002872   | 11.70x  | -91.45%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.033596 | 0.001000   | 33.61x  | -97.02%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.033596 | 0.000807   | 41.63x  | -97.60%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.033701 | 0.004124   | 8.17x   | -87.76%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.033701 | 0.000927   | 36.34x  | -97.25%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.033701 | 0.000783   | 43.07x  | -97.68%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.033687 | 0.004273   | 7.88x   | -87.31%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.033687 | 0.000930   | 36.20x  | -97.24%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.033687 | 0.000784   | 42.97x  | -97.67%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.034580 | 0.002542   | 13.60x  | -92.65%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.034580 | 0.000949   | 36.43x  | -97.25%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.034580 | 0.000769   | 44.94x  | -97.77%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.040663 | 0.006935   | 5.86x   | -82.94%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.040663 | 0.000947   | 42.92x  | -97.67%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.040663 | 0.000769   | 52.91x  | -98.11%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.020681 | 0.002570   | 8.05x   | -87.57%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.020681 | 0.000846   | 24.44x  | -95.91%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.020681 | 0.000749   | 27.62x  | -96.38%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.030984 | 0.002877   | 10.77x  | -90.72%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.030984 | 0.000794   | 39.00x  | -97.44%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.030984 | 0.000799   | 38.77x  | -97.42%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.023603 | 0.003121   | 7.56x   | -86.78%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.023603 | 0.000745   | 31.69x  | -96.84%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.023603 | 0.000795   | 29.70x  | -96.63%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.025612 | 0.002828   | 9.06x   | -88.96%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.025612 | 0.000807   | 31.75x  | -96.85%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.025612 | 0.000834   | 30.73x  | -96.75%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.026331 | 0.003133   | 8.41x   | -88.10%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.026331 | 0.001028   | 25.62x  | -96.10%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.026331 | 0.000889   | 29.61x  | -96.62%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.024225 | 0.005519   | 4.39x   | -77.22%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.024225 | 0.000828   | 29.24x  | -96.58%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.024225 | 0.000852   | 28.44x  | -96.48%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.024396 | 0.003228   | 7.56x   | -86.77%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.024396 | 0.000772   | 31.62x  | -96.84%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.024396 | 0.000785   | 31.07x  | -96.78%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.024975 | 0.002609   | 9.57x   | -89.55%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.024975 | 0.000775   | 32.22x  | -96.90%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.024975 | 0.000793   | 31.49x  | -96.82%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.034141 | 0.007676   | 4.45x   | -77.52%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.034141 | 0.001043   | 32.75x  | -96.95%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.034141 | 0.000882   | 38.72x  | -97.42%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.032769 | 0.002743   | 11.95x  | -91.63%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.032769 | 0.000775   | 42.29x  | -97.64%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.032769 | 0.000826   | 39.69x  | -97.48%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.025135 | 0.003564   | 7.05x   | -85.82%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.025135 | 0.000874   | 28.75x  | -96.52%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.025135 | 0.000832   | 30.20x  | -96.69%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.024229 | 0.004317   | 5.61x   | -82.18%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.024229 | 0.000808   | 30.00x  | -96.67%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.024229 | 0.000790   | 30.67x  | -96.74%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.024596 | 0.004490   | 5.48x   | -81.74%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.024596 | 0.000944   | 26.07x  | -96.16%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.024596 | 0.000906   | 27.15x  | -96.32%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.025856 | 0.002894   | 8.94x   | -88.81%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.025856 | 0.000799   | 32.36x  | -96.91%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.025856 | 0.000815   | 31.74x  | -96.85%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.031756 | 0.007160   | 4.44x   | -77.45%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.031756 | 0.000810   | 39.21x  | -97.45%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.031756 | 0.000805   | 39.46x  | -97.47%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.102990 | 0.026095   | 3.95x   | -74.66%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.102990 | 0.022944   | 4.49x   | -77.72%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.102990 | 0.024411   | 4.22x   | -76.30%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.133581 | 0.031484   | 4.24x   | -76.43%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.133581 | 0.023652   | 5.65x   | -82.29%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.133581 | 0.022360   | 5.97x   | -83.26%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.106370 | 0.034388   | 3.09x   | -67.67%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.106370 | 0.022982   | 4.63x   | -78.39%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.106370 | 0.022095   | 4.81x   | -79.23%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.111687 | 0.032468   | 3.44x   | -70.93%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.111687 | 0.023993   | 4.66x   | -78.52%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.111687 | 0.022755   | 4.91x   | -79.63%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.112640 | 0.030548   | 3.69x   | -72.88%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.112640 | 0.024109   | 4.67x   | -78.60%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.112640 | 0.022181   | 5.08x   | -80.31%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.107664 | 0.041564   | 2.59x   | -61.39%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.107664 | 0.024527   | 4.39x   | -77.22%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.107664 | 0.023359   | 4.61x   | -78.30%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.108353 | 0.035097   | 3.09x   | -67.61%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.108353 | 0.023671   | 4.58x   | -78.15%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.108353 | 0.022565   | 4.80x   | -79.17%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.112469 | 0.031106   | 3.62x   | -72.34%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.112469 | 0.023621   | 4.76x   | -79.00%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.112469 | 0.022829   | 4.93x   | -79.70%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.147227 | 0.050486   | 2.92x   | -65.71%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.147227 | 0.023989   | 6.14x   | -83.71%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.147227 | 0.022376   | 6.58x   | -84.80%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.138699 | 0.030120   | 4.60x   | -78.28%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.138699 | 0.023046   | 6.02x   | -83.38%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.138699 | 0.022250   | 6.23x   | -83.96%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.111315 | 0.027424   | 4.06x   | -75.36%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.111315 | 0.023711   | 4.69x   | -78.70%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.111315 | 0.024958   | 4.46x   | -77.58%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.114493 | 0.036525   | 3.13x   | -68.10%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.114493 | 0.023280   | 4.92x   | -79.67%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.114493 | 0.022285   | 5.14x   | -80.54%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.116293 | 0.036098   | 3.22x   | -68.96%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.116293 | 0.023403   | 4.97x   | -79.88%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.116293 | 0.022097   | 5.26x   | -81.00%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.110144 | 0.030770   | 3.58x   | -72.06%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.110144 | 0.023700   | 4.65x   | -78.48%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.110144 | 0.021726   | 5.07x   | -80.27%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.140710 | 0.048647   | 2.89x   | -65.43%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.140710 | 0.024020   | 5.86x   | -82.93%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.140710 | 0.022087   | 6.37x   | -84.30%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.075240 | 0.021920   | 3.43x   | -70.87%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.075240 | 0.002947   | 25.53x  | -96.08%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.075240 | 0.002660   | 28.28x  | -96.46%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.108477 | 0.026600   | 4.08x   | -75.48%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.108477 | 0.003452   | 31.43x  | -96.82%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.108477 | 0.003441   | 31.52x  | -96.83%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.079945 | 0.028165   | 2.84x   | -64.77%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.079945 | 0.003127   | 25.57x  | -96.09%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.079945 | 0.003412   | 23.43x  | -95.73%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.086225 | 0.026610   | 3.24x   | -69.14%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.086225 | 0.003288   | 26.22x  | -96.19%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.086225 | 0.003441   | 25.06x  | -96.01%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.084333 | 0.026823   | 3.14x   | -68.19%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.084333 | 0.003824   | 22.05x  | -95.47%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.084333 | 0.003550   | 23.76x  | -95.79%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.083707 | 0.032400   | 2.58x   | -61.29%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.083707 | 0.004304   | 19.45x  | -94.86%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.083707 | 0.003492   | 23.97x  | -95.83%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.080799 | 0.027983   | 2.89x   | -65.37%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.080799 | 0.003127   | 25.84x  | -96.13%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.080799 | 0.003414   | 23.67x  | -95.78%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.083370 | 0.027735   | 3.01x   | -66.73%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.083370 | 0.003186   | 26.16x  | -96.18%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.083370 | 0.003832   | 21.76x  | -95.40%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.119815 | 0.044567   | 2.69x   | -62.80%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.119815 | 0.003873   | 30.93x  | -96.77%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.119815 | 0.003560   | 33.65x  | -97.03%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.113976 | 0.024814   | 4.59x   | -78.23%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.113976 | 0.003147   | 36.22x  | -97.24%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.113976 | 0.003437   | 33.16x  | -96.98%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.080664 | 0.025810   | 3.13x   | -68.00%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.080664 | 0.004208   | 19.17x  | -94.78%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.080664 | 0.003529   | 22.86x  | -95.62%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.084870 | 0.031957   | 2.66x   | -62.35%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.084870 | 0.003730   | 22.76x  | -95.61%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.084870 | 0.003438   | 24.69x  | -95.95%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.084988 | 0.031393   | 2.71x   | -63.06%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.084988 | 0.003339   | 25.45x  | -96.07%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.084988 | 0.003470   | 24.50x  | -95.92%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.084607 | 0.026708   | 3.17x   | -68.43%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.084607 | 0.003385   | 24.99x  | -96.00%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.084607 | 0.003443   | 24.58x  | -95.93%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.109840 | 0.042955   | 2.56x   | -60.89%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.109840 | 0.003654   | 30.06x  | -96.67%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.109840 | 0.003514   | 31.25x  | -96.80%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.086485 | 0.008754   | 9.88x   | -89.88%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.086485 | 0.003186   | 27.14x  | -96.32%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.086485 | 0.002868   | 30.16x  | -96.68%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.121881 | 0.010961   | 11.12x  | -91.01%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.121881 | 0.003844   | 31.71x  | -96.85%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.121881 | 0.003203   | 38.05x  | -97.37%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.094384 | 0.012334   | 7.65x   | -86.93%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.094384 | 0.003577   | 26.39x  | -96.21%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.094384 | 0.003057   | 30.88x  | -96.76%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.098206 | 0.010200   | 9.63x   | -89.61%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.098206 | 0.004168   | 23.56x  | -95.76%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.098206 | 0.003528   | 27.83x  | -96.41%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.098549 | 0.011186   | 8.81x   | -88.65%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.098549 | 0.004137   | 23.82x  | -95.80%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.098549 | 0.003812   | 25.85x  | -96.13%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.096307 | 0.026117   | 3.69x   | -72.88%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.096307 | 0.003893   | 24.74x  | -95.96%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.096307 | 0.003213   | 29.97x  | -96.66%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.093369 | 0.012322   | 7.58x   | -86.80%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.093369 | 0.003422   | 27.29x  | -96.34%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.093369 | 0.003194   | 29.23x  | -96.58%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.096342 | 0.009689   | 9.94x   | -89.94%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.096342 | 0.003596   | 26.79x  | -96.27%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.096342 | 0.003034   | 31.76x  | -96.85%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.132288 | 0.029804   | 4.44x   | -77.47%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.132288 | 0.004200   | 31.50x  | -96.82%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.132288 | 0.003278   | 40.36x  | -97.52%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.126448 | 0.009485   | 13.33x  | -92.50%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.126448 | 0.003760   | 33.63x  | -97.03%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.126448 | 0.003163   | 39.98x  | -97.50%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.094800 | 0.012081   | 7.85x   | -87.26%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.094800 | 0.003964   | 23.92x  | -95.82%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.094800 | 0.003220   | 29.44x  | -96.60%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.096154 | 0.016971   | 5.67x   | -82.35%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.096154 | 0.003634   | 26.46x  | -96.22%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.096154 | 0.003036   | 31.67x  | -96.84%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.096917 | 0.016955   | 5.72x   | -82.51%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.096917 | 0.003742   | 25.90x  | -96.14%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.096917 | 0.003205   | 30.24x  | -96.69%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.099177 | 0.010813   | 9.17x   | -89.10%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.099177 | 0.003979   | 24.93x  | -95.99%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.099177 | 0.003296   | 30.09x  | -96.68%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.126019 | 0.028629   | 4.40x   | -77.28%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.126019 | 0.003958   | 31.84x  | -96.86%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.126019 | 0.003164   | 39.83x  | -97.49%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.067333 | 0.010083   | 6.68x   | -85.02%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.067333 | 0.003416   | 19.71x  | -94.93%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.067333 | 0.003104   | 21.69x  | -95.39%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.102285 | 0.011922   | 8.58x   | -88.34%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.102285 | 0.003641   | 28.09x  | -96.44%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.102285 | 0.003265   | 31.32x  | -96.81%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.074826 | 0.013472   | 5.55x   | -82.00%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.074826 | 0.003204   | 23.35x  | -95.72%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.074826 | 0.003194   | 23.43x  | -95.73%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.078214 | 0.011219   | 6.97x   | -85.66%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.078214 | 0.003261   | 23.99x  | -95.83%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.078214 | 0.003194   | 24.49x  | -95.92%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.077603 | 0.012279   | 6.32x   | -84.18%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.077603 | 0.003728   | 20.82x  | -95.20%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.077603 | 0.003159   | 24.57x  | -95.93%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.073514 | 0.021760   | 3.38x   | -70.40%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.073514 | 0.003326   | 22.10x  | -95.48%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.073514 | 0.003226   | 22.79x  | -95.61%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.074642 | 0.012664   | 5.89x   | -83.03%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.074642 | 0.003286   | 22.71x  | -95.60%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.074642 | 0.003246   | 22.99x  | -95.65%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.076172 | 0.011071   | 6.88x   | -85.47%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.076172 | 0.003135   | 24.30x  | -95.88%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.076172 | 0.003183   | 23.93x  | -95.82%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.113255 | 0.030242   | 3.74x   | -73.30%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.113255 | 0.003729   | 30.37x  | -96.71%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.113255 | 0.003104   | 36.49x  | -97.26%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.104259 | 0.010656   | 9.78x   | -89.78%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.104259 | 0.003207   | 32.51x  | -96.92%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.104259 | 0.003240   | 32.18x  | -96.89%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.074218 | 0.016360   | 4.54x   | -77.96%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.074218 | 0.003394   | 21.87x  | -95.43%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.074218 | 0.003289   | 22.56x  | -95.57%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.081080 | 0.017356   | 4.67x   | -78.59%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.081080 | 0.003262   | 24.85x  | -95.98%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.081080 | 0.003244   | 24.99x  | -96.00%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.076276 | 0.017024   | 4.48x   | -77.68%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.076276 | 0.003191   | 23.90x  | -95.82%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.076276 | 0.003210   | 23.76x  | -95.79%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.076998 | 0.011838   | 6.50x   | -84.63%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.076998 | 0.003229   | 23.85x  | -95.81%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.076998 | 0.003257   | 23.64x  | -95.77%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.105054 | 0.028329   | 3.71x   | -73.03%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.105054 | 0.003504   | 29.98x  | -96.66%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.105054 | 0.003294   | 31.89x  | -96.86%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.393977 | 0.106384   | 3.70x   | -73.00%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.393977 | 0.089620   | 4.40x   | -77.25%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.393977 | 0.097577   | 4.04x   | -75.23%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.522030 | 0.131000   | 3.98x   | -74.91%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.522030 | 0.098874   | 5.28x   | -81.06%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.522030 | 0.093401   | 5.59x   | -82.11%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.394995 | 0.136706   | 2.89x   | -65.39%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.394995 | 0.092283   | 4.28x   | -76.64%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.394995 | 0.089939   | 4.39x   | -77.23%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.416750 | 0.122926   | 3.39x   | -70.50%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.416750 | 0.095037   | 4.39x   | -77.20%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.416750 | 0.089317   | 4.67x   | -78.57%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.428377 | 0.124218   | 3.45x   | -71.00%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.428377 | 0.096461   | 4.44x   | -77.48%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.428377 | 0.089632   | 4.78x   | -79.08%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.394848 | 0.163462   | 2.42x   | -58.60%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.394848 | 0.093835   | 4.21x   | -76.24%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.394848 | 0.089864   | 4.39x   | -77.24%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.386858 | 0.139476   | 2.77x   | -63.95%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.386858 | 0.093041   | 4.16x   | -75.95%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.386858 | 0.088994   | 4.35x   | -77.00%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.397611 | 0.119231   | 3.33x   | -70.01%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.397611 | 0.091704   | 4.34x   | -76.94%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.397611 | 0.087325   | 4.55x   | -78.04%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.563908 | 0.201819   | 2.79x   | -64.21%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.563908 | 0.094747   | 5.95x   | -83.20%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.563908 | 0.090261   | 6.25x   | -83.99%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.509898 | 0.120138   | 4.24x   | -76.44%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.509898 | 0.091179   | 5.59x   | -82.12%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.509898 | 0.087921   | 5.80x   | -82.76%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.395999 | 0.106753   | 3.71x   | -73.04%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.395999 | 0.092688   | 4.27x   | -76.59%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.395999 | 0.088471   | 4.48x   | -77.66%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.405330 | 0.142069   | 2.85x   | -64.95%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.405330 | 0.095688   | 4.24x   | -76.39%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.405330 | 0.087054   | 4.66x   | -78.52%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.432563 | 0.148375   | 2.92x   | -65.70%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.432563 | 0.097004   | 4.46x   | -77.57%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.432563 | 0.088915   | 4.86x   | -79.44%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.411983 | 0.124979   | 3.30x   | -69.66%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.411983 | 0.097180   | 4.24x   | -76.41%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.411983 | 0.088450   | 4.66x   | -78.53%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.522539 | 0.192308   | 2.72x   | -63.20%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.522539 | 0.096362   | 5.42x   | -81.56%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.522539 | 0.090235   | 5.79x   | -82.73%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.291790 | 0.087554   | 3.33x   | -69.99%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.291790 | 0.011730   | 24.88x  | -95.98%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.291790 | 0.010483   | 27.83x  | -96.41%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.407083 | 0.107182   | 3.80x   | -73.67%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.407083 | 0.013829   | 29.44x  | -96.60%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.407083 | 0.013873   | 29.34x  | -96.59%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.282093 | 0.111762   | 2.52x   | -60.38%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.282093 | 0.012526   | 22.52x  | -95.56%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.282093 | 0.013678   | 20.62x  | -95.15%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.306892 | 0.101537   | 3.02x   | -66.91%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.306892 | 0.012943   | 23.71x  | -95.78%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.306892 | 0.013664   | 22.46x  | -95.55%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.306970 | 0.107526   | 2.85x   | -64.97%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.306970 | 0.015288   | 20.08x  | -95.02%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.306970 | 0.014137   | 21.71x  | -95.39%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.288367 | 0.128529   | 2.24x   | -55.43%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.288367 | 0.016996   | 16.97x  | -94.11%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.288367 | 0.013741   | 20.99x  | -95.23%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.279310 | 0.111830   | 2.50x   | -59.96%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.279310 | 0.012576   | 22.21x  | -95.50%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.279310 | 0.013724   | 20.35x  | -95.09%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.300195 | 0.105706   | 2.84x   | -64.79%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.300195 | 0.012630   | 23.77x  | -95.79%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.300195 | 0.013896   | 21.60x  | -95.37%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.461494 | 0.179748   | 2.57x   | -61.05%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.461494 | 0.015754   | 29.29x  | -96.59%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.461494 | 0.014037   | 32.88x  | -96.96%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.412832 | 0.105349   | 3.92x   | -74.48%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.412832 | 0.012981   | 31.80x  | -96.86%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.412832 | 0.013889   | 29.72x  | -96.64%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.291445 | 0.098236   | 2.97x   | -66.29%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.291445 | 0.016706   | 17.45x  | -94.27%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.291445 | 0.014154   | 20.59x  | -95.14%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.304229 | 0.125995   | 2.41x   | -58.59%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.304229 | 0.013430   | 22.65x  | -95.59%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.304229 | 0.013873   | 21.93x  | -95.44%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.306086 | 0.125545   | 2.44x   | -58.98%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.306086 | 0.013415   | 22.82x  | -95.62%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.306086 | 0.014043   | 21.80x  | -95.41%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.320121 | 0.104803   | 3.05x   | -67.26%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.320121 | 0.013974   | 22.91x  | -95.63%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.320121 | 0.013864   | 23.09x  | -95.67%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.423811 | 0.170773   | 2.48x   | -59.71%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.423811 | 0.014495   | 29.24x  | -96.58%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.423811 | 0.013886   | 30.52x  | -96.72%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.357652 | 0.038078   | 9.39x   | -89.35%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.357652 | 0.019179   | 18.65x  | -94.64%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.357652 | 0.016322   | 21.91x  | -95.44%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.459076 | 0.047007   | 9.77x   | -89.76%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.459076 | 0.020680   | 22.20x  | -95.50%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.459076 | 0.018056   | 25.42x  | -96.07%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.343086 | 0.052477   | 6.54x   | -84.70%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.343086 | 0.019548   | 17.55x  | -94.30%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.343086 | 0.017642   | 19.45x  | -94.86%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.355319 | 0.041558   | 8.55x   | -88.30%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.355319 | 0.020073   | 17.70x  | -94.35%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.355319 | 0.019679   | 18.06x  | -94.46%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.357233 | 0.047135   | 7.58x   | -86.81%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.357233 | 0.021346   | 16.74x  | -94.02%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.357233 | 0.020212   | 17.67x  | -94.34%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.358012 | 0.106622   | 3.36x   | -70.22%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.358012 | 0.020828   | 17.19x  | -94.18%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.358012 | 0.017875   | 20.03x  | -95.01%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.350588 | 0.054577   | 6.42x   | -84.43%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.350588 | 0.020013   | 17.52x  | -94.29%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.350588 | 0.017990   | 19.49x  | -94.87%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.345944 | 0.041152   | 8.41x   | -88.10%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.345944 | 0.019326   | 17.90x  | -94.41%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.345944 | 0.017537   | 19.73x  | -94.93%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.518538 | 0.124283   | 4.17x   | -76.03%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.518538 | 0.021264   | 24.39x  | -95.90%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.518538 | 0.018077   | 28.69x  | -96.51%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.461890 | 0.042379   | 10.90x  | -90.82%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.461890 | 0.020214   | 22.85x  | -95.62%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.461890 | 0.018207   | 25.37x  | -96.06%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.353713 | 0.051625   | 6.85x   | -85.40%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.353713 | 0.021526   | 16.43x  | -93.91%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.353713 | 0.018460   | 19.16x  | -94.78%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.365488 | 0.071905   | 5.08x   | -80.33%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.365488 | 0.023153   | 15.79x  | -93.67%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.365488 | 0.018370   | 19.90x  | -94.97%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.362310 | 0.071961   | 5.03x   | -80.14%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.362310 | 0.020257   | 17.89x  | -94.41%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.362310 | 0.018093   | 20.02x  | -95.01%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.369610 | 0.044641   | 8.28x   | -87.92%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.369610 | 0.020116   | 18.37x  | -94.56%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.369610 | 0.017479   | 21.15x  | -95.27%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.465645 | 0.115918   | 4.02x   | -75.11%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.465645 | 0.020748   | 22.44x  | -95.54%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.465645 | 0.018112   | 25.71x  | -96.11%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.270274 | 0.046876   | 5.77x   | -82.66%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.270274 | 0.020253   | 13.34x  | -92.51%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.270274 | 0.019207   | 14.07x  | -92.89%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.387664 | 0.053763   | 7.21x   | -86.13%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.387664 | 0.020289   | 19.11x  | -94.77%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.387664 | 0.020230   | 19.16x  | -94.78%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.260659 | 0.058578   | 4.45x   | -77.53%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.260659 | 0.019679   | 13.25x  | -92.45%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.260659 | 0.019151   | 13.61x  | -92.65%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.286809 | 0.051431   | 5.58x   | -82.07%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.286809 | 0.019588   | 14.64x  | -93.17%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.286809 | 0.019266   | 14.89x  | -93.28%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.290346 | 0.055399   | 5.24x   | -80.92%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.290346 | 0.023592   | 12.31x  | -91.87%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.290346 | 0.020441   | 14.20x  | -92.96%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.270338 | 0.096374   | 2.81x   | -64.35%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.270338 | 0.021417   | 12.62x  | -92.08%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.270338 | 0.019784   | 13.66x  | -92.68%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.264228 | 0.059740   | 4.42x   | -77.39%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.264228 | 0.020298   | 13.02x  | -92.32%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.264228 | 0.019750   | 13.38x  | -92.53%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.270792 | 0.049192   | 5.50x   | -81.83%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.270792 | 0.021329   | 12.70x  | -92.12%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.270792 | 0.019349   | 14.00x  | -92.85%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.441604 | 0.132772   | 3.33x   | -69.93%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.441604 | 0.025190   | 17.53x  | -94.30%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.441604 | 0.021043   | 20.99x  | -95.23%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.388163 | 0.051014   | 7.61x   | -86.86%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.388163 | 0.020036   | 19.37x  | -94.84%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.388163 | 0.019334   | 20.08x  | -95.02%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.274652 | 0.067461   | 4.07x   | -75.44%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.274652 | 0.021904   | 12.54x  | -92.02%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.274652 | 0.020583   | 13.34x  | -92.51%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.283967 | 0.078042   | 3.64x   | -72.52%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.283967 | 0.020331   | 13.97x  | -92.84%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.283967 | 0.020381   | 13.93x  | -92.82%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.283831 | 0.077015   | 3.69x   | -72.87%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.283831 | 0.020415   | 13.90x  | -92.81%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.283831 | 0.019753   | 14.37x  | -93.04%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.290572 | 0.054611   | 5.32x   | -81.21%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.290572 | 0.021591   | 13.46x  | -92.57%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.290572 | 0.020114   | 14.45x  | -93.08%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.402690 | 0.122568   | 3.29x   | -69.56%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.402690 | 0.021783   | 18.49x  | -94.59%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.402690 | 0.020510   | 19.63x  | -94.91%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.083778 | 0.023324   | 3.59x   | -72.16%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.083778 | 0.020792   | 4.03x   | -75.18%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.083778 | 0.020591   | 4.07x   | -75.42%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.110988 | 0.027237   | 4.07x   | -75.46%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.110988 | 0.021198   | 5.24x   | -80.90%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.110988 | 0.019733   | 5.62x   | -82.22%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.086852 | 0.029635   | 2.93x   | -65.88%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.086852 | 0.021018   | 4.13x   | -75.80%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.086852 | 0.019316   | 4.50x   | -77.76%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.095153 | 0.026387   | 3.61x   | -72.27%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.095153 | 0.020313   | 4.68x   | -78.65%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.095153 | 0.019482   | 4.88x   | -79.53%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.090750 | 0.026923   | 3.37x   | -70.33%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.090750 | 0.021097   | 4.30x   | -76.75%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.090750 | 0.019715   | 4.60x   | -78.28%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.086506 | 0.035986   | 2.40x   | -58.40%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.086506 | 0.020976   | 4.12x   | -75.75%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.086506 | 0.019591   | 4.42x   | -77.35%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.087095 | 0.029698   | 2.93x   | -65.90%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.087095 | 0.020066   | 4.34x   | -76.96%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.087095 | 0.019158   | 4.55x   | -78.00%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.087385 | 0.026802   | 3.26x   | -69.33%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.087385 | 0.020265   | 4.31x   | -76.81%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.087385 | 0.019408   | 4.50x   | -77.79%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.118655 | 0.044045   | 2.69x   | -62.88%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.118655 | 0.021679   | 5.47x   | -81.73%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.118655 | 0.020013   | 5.93x   | -83.13%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.114981 | 0.026504   | 4.34x   | -76.95%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.114981 | 0.020159   | 5.70x   | -82.47%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.114981 | 0.019464   | 5.91x   | -83.07%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.084340 | 0.023412   | 3.60x   | -72.24%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.084340 | 0.020489   | 4.12x   | -75.71%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.084340 | 0.019579   | 4.31x   | -76.79%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.090137 | 0.032944   | 2.74x   | -63.45%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.090137 | 0.021209   | 4.25x   | -76.47%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.090137 | 0.019787   | 4.56x   | -78.05%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.090663 | 0.031665   | 2.86x   | -65.07%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.090663 | 0.020770   | 4.37x   | -77.09%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.090663 | 0.019411   | 4.67x   | -78.59%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.093691 | 0.026684   | 3.51x   | -71.52%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.093691 | 0.020613   | 4.55x   | -78.00%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.093691 | 0.019048   | 4.92x   | -79.67%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.113122 | 0.042779   | 2.64x   | -62.18%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.113122 | 0.021200   | 5.34x   | -81.26%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.113122 | 0.019648   | 5.76x   | -82.63%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.067176 | 0.018746   | 3.58x   | -72.09%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.067176 | 0.002556   | 26.28x  | -96.19%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.067176 | 0.002720   | 24.69x  | -95.95%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.101922 | 0.023875   | 4.27x   | -76.58%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.101922 | 0.003102   | 32.85x  | -96.96%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.101922 | 0.002998   | 34.00x  | -97.06%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.078081 | 0.025613   | 3.05x   | -67.20%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.078081 | 0.002788   | 28.01x  | -96.43%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.078081 | 0.003025   | 25.81x  | -96.13%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.080656 | 0.022873   | 3.53x   | -71.64%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.080656 | 0.002794   | 28.87x  | -96.54%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.080656 | 0.003335   | 24.18x  | -95.86%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.079180 | 0.022825   | 3.47x   | -71.17%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.079180 | 0.003635   | 21.78x  | -95.41%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.079180 | 0.003039   | 26.05x  | -96.16%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.074371 | 0.027880   | 2.67x   | -62.51%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.074371 | 0.003679   | 20.22x  | -95.05%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.074371 | 0.003115   | 23.88x  | -95.81%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.075676 | 0.024316   | 3.11x   | -67.87%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.075676 | 0.002696   | 28.07x  | -96.44%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.075676 | 0.003026   | 25.01x  | -96.00%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.082980 | 0.022900   | 3.62x   | -72.40%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.082980 | 0.002850   | 29.12x  | -96.57%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.082980 | 0.003039   | 27.30x  | -96.34%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.120168 | 0.039436   | 3.05x   | -67.18%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.120168 | 0.003379   | 35.56x  | -97.19%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.120168 | 0.003025   | 39.73x  | -97.48%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.106918 | 0.022347   | 4.78x   | -79.10%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.106918 | 0.002875   | 37.19x  | -97.31%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.106918 | 0.003096   | 34.53x  | -97.10%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.075927 | 0.021796   | 3.48x   | -71.29%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.075927 | 0.003718   | 20.42x  | -95.10%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.075927 | 0.003094   | 24.54x  | -95.93%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.081279 | 0.028846   | 2.82x   | -64.51%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.081279 | 0.003059   | 26.57x  | -96.24%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.081279 | 0.003099   | 26.23x  | -96.19%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.082773 | 0.028265   | 2.93x   | -65.85%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.082773 | 0.003087   | 26.82x  | -96.27%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.082773 | 0.003079   | 26.88x  | -96.28%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.086292 | 0.023337   | 3.70x   | -72.96%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.086292 | 0.003102   | 27.82x  | -96.41%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.086292 | 0.003546   | 24.33x  | -95.89%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.106928 | 0.037889   | 2.82x   | -64.57%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.106928 | 0.003288   | 32.52x  | -96.92%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.106928 | 0.003213   | 33.28x  | -96.99%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.074026 | 0.007147   | 10.36x  | -90.35%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.074026 | 0.002965   | 24.97x  | -96.00%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.074026 | 0.002344   | 31.59x  | -96.83%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.107296 | 0.008983   | 11.94x  | -91.63%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.107296 | 0.003538   | 30.32x  | -96.70%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.107296 | 0.002843   | 37.74x  | -97.35%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.086533 | 0.010612   | 8.15x   | -87.74%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.086533 | 0.003242   | 26.69x  | -96.25%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.086533 | 0.002778   | 31.15x  | -96.79%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.087830 | 0.008545   | 10.28x  | -90.27%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.087830 | 0.003377   | 26.01x  | -96.16%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.087830 | 0.002818   | 31.16x  | -96.79%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.088203 | 0.009472   | 9.31x   | -89.26%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.088203 | 0.003603   | 24.48x  | -95.91%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.088203 | 0.002878   | 30.65x  | -96.74%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.083210 | 0.022899   | 3.63x   | -72.48%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.083210 | 0.003433   | 24.24x  | -95.87%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.083210 | 0.002780   | 29.93x  | -96.66%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.083605 | 0.010669   | 7.84x   | -87.24%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.083605 | 0.003211   | 26.04x  | -96.16%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.083605 | 0.002803   | 29.83x  | -96.65%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.083528 | 0.007805   | 10.70x  | -90.66%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.083528 | 0.003131   | 26.68x  | -96.25%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.083528 | 0.002716   | 30.75x  | -96.75%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.115439 | 0.026504   | 4.36x   | -77.04%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.115439 | 0.003641   | 31.71x  | -96.85%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.115439 | 0.002793   | 41.34x  | -97.58%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.111184 | 0.008781   | 12.66x  | -92.10%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.111184 | 0.003311   | 33.58x  | -97.02%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.111184 | 0.002740   | 40.58x  | -97.54%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.081091 | 0.010169   | 7.97x   | -87.46%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.081091 | 0.003440   | 23.57x  | -95.76%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.081091 | 0.002786   | 29.10x  | -96.56%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.085230 | 0.014663   | 5.81x   | -82.80%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.085230 | 0.003326   | 25.63x  | -96.10%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.085230 | 0.002678   | 31.82x  | -96.86%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.085830 | 0.014782   | 5.81x   | -82.78%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.085830 | 0.003304   | 25.98x  | -96.15%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.085830 | 0.002737   | 31.36x  | -96.81%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.085904 | 0.009463   | 9.08x   | -88.98%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.085904 | 0.003432   | 25.03x  | -96.00%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.085904 | 0.002729   | 31.48x  | -96.82%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.110655 | 0.024765   | 4.47x   | -77.62%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.110655 | 0.003415   | 32.40x  | -96.91%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.110655 | 0.002731   | 40.51x  | -97.53%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.059847 | 0.008977   | 6.67x   | -85.00%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.059847 | 0.003258   | 18.37x  | -94.56%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.059847 | 0.002747   | 21.78x  | -95.41%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.094331 | 0.010246   | 9.21x   | -89.14%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.094331 | 0.002925   | 32.25x  | -96.90%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.094331 | 0.002851   | 33.08x  | -96.98%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.067704 | 0.010965   | 6.17x   | -83.80%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.067704 | 0.002774   | 24.40x  | -95.90%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.067704 | 0.002791   | 24.26x  | -95.88%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.072546 | 0.009726   | 7.46x   | -86.59%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.072546 | 0.002954   | 24.56x  | -95.93%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.072546 | 0.002882   | 25.17x  | -96.03%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.073137 | 0.010589   | 6.91x   | -85.52%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.073137 | 0.003588   | 20.38x  | -95.09%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.073137 | 0.002788   | 26.23x  | -96.19%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.065546 | 0.019649   | 3.34x   | -70.02%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.065546 | 0.003066   | 21.38x  | -95.32%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.065546 | 0.002906   | 22.55x  | -95.57%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.068406 | 0.011066   | 6.18x   | -83.82%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.068406 | 0.002747   | 24.91x  | -95.98%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.068406 | 0.002786   | 24.55x  | -95.93%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.068890 | 0.009434   | 7.30x   | -86.31%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.068890 | 0.002731   | 25.22x  | -96.04%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.068890 | 0.002809   | 24.52x  | -95.92%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.100586 | 0.026891   | 3.74x   | -73.27%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.100586 | 0.003284   | 30.63x  | -96.74%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.100586 | 0.002792   | 36.03x  | -97.22%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.094240 | 0.009339   | 10.09x  | -90.09%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.094240 | 0.002631   | 35.81x  | -97.21%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.094240 | 0.002703   | 34.86x  | -97.13%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.065181 | 0.013030   | 5.00x   | -80.01%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.065181 | 0.003050   | 21.37x  | -95.32%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.065181 | 0.002926   | 22.27x  | -95.51%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.070418 | 0.015226   | 4.62x   | -78.38%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.070418 | 0.002926   | 24.07x  | -95.84%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.070418 | 0.002807   | 25.09x  | -96.01%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.070431 | 0.015055   | 4.68x   | -78.62%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.070431 | 0.002918   | 24.14x  | -95.86%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.070431 | 0.002786   | 25.28x  | -96.04%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.071921 | 0.010250   | 7.02x   | -85.75%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.071921 | 0.002896   | 24.84x  | -95.97%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.071921 | 0.002823   | 25.48x  | -96.07%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.097496 | 0.024885   | 3.92x   | -74.48%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.097496 | 0.002876   | 33.90x  | -97.05%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.097496 | 0.002817   | 34.61x  | -97.11%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.183654 | 0.051044   | 3.60x   | -72.21%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.183654 | 0.044617   | 4.12x   | -75.71%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.183654 | 0.046728   | 3.93x   | -74.56%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.244810 | 0.061942   | 3.95x   | -74.70%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.244810 | 0.047802   | 5.12x   | -80.47%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.244810 | 0.044842   | 5.46x   | -81.68%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.192974 | 0.068973   | 2.80x   | -64.26%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.192974 | 0.044631   | 4.32x   | -76.87%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.192974 | 0.042678   | 4.52x   | -77.88%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.212234 | 0.061524   | 3.45x   | -71.01%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.212234 | 0.048407   | 4.38x   | -77.19%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.212234 | 0.044715   | 4.75x   | -78.93%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.204722 | 0.061017   | 3.36x   | -70.20%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.204722 | 0.048133   | 4.25x   | -76.49%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.204722 | 0.045374   | 4.51x   | -77.84%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.201563 | 0.083205   | 2.42x   | -58.72%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.201563 | 0.047536   | 4.24x   | -76.42%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.201563 | 0.045101   | 4.47x   | -77.62%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.197103 | 0.070747   | 2.79x   | -64.11%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.197103 | 0.046742   | 4.22x   | -76.29%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.197103 | 0.045887   | 4.30x   | -76.72%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.197488 | 0.060843   | 3.25x   | -69.19%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.197488 | 0.046164   | 4.28x   | -76.62%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.197488 | 0.043811   | 4.51x   | -77.82%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.264189 | 0.098499   | 2.68x   | -62.72%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.264189 | 0.047597   | 5.55x   | -81.98%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.264189 | 0.044231   | 5.97x   | -83.26%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.254417 | 0.058984   | 4.31x   | -76.82%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.254417 | 0.045396   | 5.60x   | -82.16%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.254417 | 0.043066   | 5.91x   | -83.07%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.193359 | 0.052099   | 3.71x   | -73.06%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.193359 | 0.045271   | 4.27x   | -76.59%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.193359 | 0.044084   | 4.39x   | -77.20%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.196720 | 0.069070   | 2.85x   | -64.89%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.196720 | 0.047050   | 4.18x   | -76.08%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.196720 | 0.044089   | 4.46x   | -77.59%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.197470 | 0.070742   | 2.79x   | -64.18%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.197470 | 0.046031   | 4.29x   | -76.69%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.197470 | 0.043567   | 4.53x   | -77.94%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.202723 | 0.064723   | 3.13x   | -68.07%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.202723 | 0.047411   | 4.28x   | -76.61%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.202723 | 0.042710   | 4.75x   | -78.93%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.251240 | 0.097583   | 2.57x   | -61.16%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.251240 | 0.048226   | 5.21x   | -80.80%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.251240 | 0.047127   | 5.33x   | -81.24%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.147416 | 0.047197   | 3.12x   | -67.98%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.147416 | 0.005960   | 24.73x  | -95.96%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.147416 | 0.005470   | 26.95x  | -96.29%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.206616 | 0.054171   | 3.81x   | -73.78%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.206616 | 0.007015   | 29.45x  | -96.60%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.206616 | 0.006937   | 29.78x  | -96.64%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.150908 | 0.058603   | 2.58x   | -61.17%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.150908 | 0.006273   | 24.06x  | -95.84%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.150908 | 0.006823   | 22.12x  | -95.48%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.167374 | 0.053048   | 3.16x   | -68.31%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.167374 | 0.006380   | 26.24x  | -96.19%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.167374 | 0.006910   | 24.22x  | -95.87%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.159365 | 0.053585   | 2.97x   | -66.38%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.159365 | 0.007522   | 21.19x  | -95.28%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.159365 | 0.007100   | 22.44x  | -95.54%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.153593 | 0.065954   | 2.33x   | -57.06%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.153593 | 0.008777   | 17.50x  | -94.29%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.153593 | 0.006984   | 21.99x  | -95.45%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.161361 | 0.063299   | 2.55x   | -60.77%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.161361 | 0.006835   | 23.61x  | -95.76%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.161361 | 0.007294   | 22.12x  | -95.48%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.175982 | 0.060600   | 2.90x   | -65.56%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.175982 | 0.006951   | 25.32x  | -96.05%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.175982 | 0.006971   | 25.24x  | -96.04%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.224384 | 0.089581   | 2.50x   | -60.08%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.224384 | 0.007592   | 29.56x  | -96.62%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.224384 | 0.006816   | 32.92x  | -96.96%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.207661 | 0.052765   | 3.94x   | -74.59%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.207661 | 0.006300   | 32.96x  | -96.97%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.207661 | 0.006700   | 30.99x  | -96.77%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.147033 | 0.050362   | 2.92x   | -65.75%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.147033 | 0.008249   | 17.82x  | -94.39%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.147033 | 0.006927   | 21.22x  | -95.29%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.153099 | 0.062148   | 2.46x   | -59.41%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.153099 | 0.006585   | 23.25x  | -95.70%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.153099 | 0.006760   | 22.65x  | -95.58%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.150907 | 0.062331   | 2.42x   | -58.70%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.150907 | 0.006806   | 22.17x  | -95.49%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.150907 | 0.006998   | 21.56x  | -95.36%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.159615 | 0.053335   | 2.99x   | -66.59%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.159615 | 0.006886   | 23.18x  | -95.69%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.159615 | 0.006701   | 23.82x  | -95.80%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.201160 | 0.083957   | 2.40x   | -58.26%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.201160 | 0.006902   | 29.14x  | -96.57%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.201160 | 0.006797   | 29.60x  | -96.62%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.164904 | 0.017248   | 9.56x   | -89.54%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.164904 | 0.006690   | 24.65x  | -95.94%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.164904 | 0.005599   | 29.45x  | -96.60%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.222393 | 0.020637   | 10.78x  | -90.72%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.222393 | 0.007530   | 29.53x  | -96.61%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.222393 | 0.006242   | 35.63x  | -97.19%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.168887 | 0.023794   | 7.10x   | -85.91%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.168887 | 0.006846   | 24.67x  | -95.95%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.168887 | 0.005790   | 29.17x  | -96.57%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.175925 | 0.019349   | 9.09x   | -89.00%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.175925 | 0.007105   | 24.76x  | -95.96%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.175925 | 0.006123   | 28.73x  | -96.52%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.179185 | 0.021332   | 8.40x   | -88.10%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.179185 | 0.008032   | 22.31x  | -95.52%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.179185 | 0.006070   | 29.52x  | -96.61%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.169932 | 0.051692   | 3.29x   | -69.58%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.169932 | 0.007728   | 21.99x  | -95.45%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.169932 | 0.006165   | 27.56x  | -96.37%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.172289 | 0.024310   | 7.09x   | -85.89%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.172289 | 0.007068   | 24.38x  | -95.90%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.172289 | 0.006097   | 28.26x  | -96.46%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.173464 | 0.019577   | 8.86x   | -88.71%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.173464 | 0.007104   | 24.42x  | -95.90%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.173464 | 0.005979   | 29.01x  | -96.55%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.244969 | 0.058673   | 4.18x   | -76.05%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.244969 | 0.007876   | 31.10x  | -96.78%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.244969 | 0.006409   | 38.22x  | -97.38%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.231603 | 0.018369   | 12.61x  | -92.07%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.231603 | 0.007276   | 31.83x  | -96.86%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.231603 | 0.006260   | 37.00x  | -97.30%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.171105 | 0.024412   | 7.01x   | -85.73%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.171105 | 0.007867   | 21.75x  | -95.40%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.171105 | 0.006187   | 27.66x  | -96.38%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.175314 | 0.033844   | 5.18x   | -80.70%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.175314 | 0.007522   | 23.31x  | -95.71%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.175314 | 0.006295   | 27.85x  | -96.41%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.176689 | 0.033961   | 5.20x   | -80.78%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.176689 | 0.007565   | 23.36x  | -95.72%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.176689 | 0.006348   | 27.83x  | -96.41%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.177937 | 0.020744   | 8.58x   | -88.34%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.177937 | 0.007495   | 23.74x  | -95.79%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.177937 | 0.006360   | 27.98x  | -96.43%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.227177 | 0.055248   | 4.11x   | -75.68%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.227177 | 0.007514   | 30.24x  | -96.69%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.227177 | 0.006147   | 36.96x  | -97.29%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.128969 | 0.019643   | 6.57x   | -84.77%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.128969 | 0.006173   | 20.89x  | -95.21%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.128969 | 0.005663   | 22.77x  | -95.61%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.188238 | 0.022364   | 8.42x   | -88.12%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.188238 | 0.006333   | 29.73x  | -96.64%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.188238 | 0.006111   | 30.80x  | -96.75%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.133064 | 0.024410   | 5.45x   | -81.66%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.133064 | 0.006060   | 21.96x  | -95.45%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.133064 | 0.006307   | 21.10x  | -95.26%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.142144 | 0.021783   | 6.53x   | -84.68%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.142144 | 0.006542   | 21.73x  | -95.40%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.142144 | 0.006460   | 22.00x  | -95.46%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.145554 | 0.024597   | 5.92x   | -83.10%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.145554 | 0.007350   | 19.80x  | -94.95%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.145554 | 0.006134   | 23.73x  | -95.79%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.141299 | 0.043457   | 3.25x   | -69.24%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.141299 | 0.006785   | 20.83x  | -95.20%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.141299 | 0.006468   | 21.85x  | -95.42%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.136836 | 0.024835   | 5.51x   | -81.85%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.136836 | 0.006206   | 22.05x  | -95.46%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.136836 | 0.006203   | 22.06x  | -95.47%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.138605 | 0.020997   | 6.60x   | -84.85%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.138605 | 0.006326   | 21.91x  | -95.44%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.138605 | 0.006159   | 22.50x  | -95.56%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.207365 | 0.060288   | 3.44x   | -70.93%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.207365 | 0.007645   | 27.12x  | -96.31%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.207365 | 0.006308   | 32.87x  | -96.96%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.201462 | 0.021519   | 9.36x   | -89.32%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.201462 | 0.006227   | 32.35x  | -96.91%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.201462 | 0.006439   | 31.29x  | -96.80%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.140115 | 0.028935   | 4.84x   | -79.35%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.140115 | 0.006970   | 20.10x  | -95.03%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.140115 | 0.006709   | 20.89x  | -95.21%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.143257 | 0.034460   | 4.16x   | -75.95%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.143257 | 0.006217   | 23.04x  | -95.66%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.143257 | 0.006291   | 22.77x  | -95.61%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.144388 | 0.035462   | 4.07x   | -75.44%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.144388 | 0.006472   | 22.31x  | -95.52%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.144388 | 0.006500   | 22.21x  | -95.50%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.152217 | 0.023481   | 6.48x   | -84.57%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.152217 | 0.006686   | 22.77x  | -95.61%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.152217 | 0.006631   | 22.95x  | -95.64%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.197362 | 0.056285   | 3.51x   | -71.48%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.197362 | 0.006490   | 30.41x  | -96.71%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.197362 | 0.006410   | 30.79x  | -96.75%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.341731 | 0.094793   | 3.61x   | -72.26%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.341731 | 0.082867   | 4.12x   | -75.75%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.341731 | 0.087369   | 3.91x   | -74.43%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.450919 | 0.111607   | 4.04x   | -75.25%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.450919 | 0.084039   | 5.37x   | -81.36%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.450919 | 0.078658   | 5.73x   | -82.56%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.340616 | 0.122307   | 2.78x   | -64.09%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.340616 | 0.080512   | 4.23x   | -76.36%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.340616 | 0.076183   | 4.47x   | -77.63%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.357712 | 0.106359   | 3.36x   | -70.27%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.357712 | 0.081580   | 4.38x   | -77.19%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.357712 | 0.075513   | 4.74x   | -78.89%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.351791 | 0.107567   | 3.27x   | -69.42%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.351791 | 0.082821   | 4.25x   | -76.46%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.351791 | 0.077571   | 4.54x   | -77.95%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.338354 | 0.142101   | 2.38x   | -58.00%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.338354 | 0.082262   | 4.11x   | -75.69%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.338354 | 0.079843   | 4.24x   | -76.40%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.352555 | 0.119403   | 2.95x   | -66.13%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.352555 | 0.080074   | 4.40x   | -77.29%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.352555 | 0.077107   | 4.57x   | -78.13%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.338950 | 0.108305   | 3.13x   | -68.05%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.338950 | 0.086266   | 3.93x   | -74.55%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.338950 | 0.077970   | 4.35x   | -77.00%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.487453 | 0.179640   | 2.71x   | -63.15%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.487453 | 0.086163   | 5.66x   | -82.32%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.487453 | 0.078283   | 6.23x   | -83.94%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.445981 | 0.105635   | 4.22x   | -76.31%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.445981 | 0.080848   | 5.52x   | -81.87%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.445981 | 0.077739   | 5.74x   | -82.57%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.340838 | 0.096657   | 3.53x   | -71.64%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.340838 | 0.081333   | 4.19x   | -76.14%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.340838 | 0.077240   | 4.41x   | -77.34%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.345065 | 0.126194   | 2.73x   | -63.43%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.345065 | 0.083280   | 4.14x   | -75.87%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.345065 | 0.077191   | 4.47x   | -77.63%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.343630 | 0.126895   | 2.71x   | -63.07%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.343630 | 0.081530   | 4.21x   | -76.27%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.343630 | 0.076588   | 4.49x   | -77.71%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.353788 | 0.108536   | 3.26x   | -69.32%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.353788 | 0.082741   | 4.28x   | -76.61%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.353788 | 0.076892   | 4.60x   | -78.27%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.446284 | 0.170034   | 2.62x   | -61.90%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.446284 | 0.082657   | 5.40x   | -81.48%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.446284 | 0.077265   | 5.78x   | -82.69%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.243635 | 0.075297   | 3.24x   | -69.09%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.243635 | 0.010322   | 23.60x  | -95.76%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.243635 | 0.009522   | 25.59x  | -96.09%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.361652 | 0.098730   | 3.66x   | -72.70%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.361652 | 0.012305   | 29.39x  | -96.60%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.361652 | 0.012407   | 29.15x  | -96.57%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.250403 | 0.100298   | 2.50x   | -59.95%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.250403 | 0.010944   | 22.88x  | -95.63%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.250403 | 0.011957   | 20.94x  | -95.23%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.267615 | 0.090872   | 2.94x   | -66.04%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.267615 | 0.011640   | 22.99x  | -95.65%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.267615 | 0.012191   | 21.95x  | -95.44%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.273559 | 0.095716   | 2.86x   | -65.01%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.273559 | 0.013397   | 20.42x  | -95.10%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.273559 | 0.012291   | 22.26x  | -95.51%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.261408 | 0.115776   | 2.26x   | -55.71%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.261408 | 0.015032   | 17.39x  | -94.25%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.261408 | 0.012309   | 21.24x  | -95.29%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.259357 | 0.102752   | 2.52x   | -60.38%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.259357 | 0.011374   | 22.80x  | -95.61%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.259357 | 0.012256   | 21.16x  | -95.27%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.256624 | 0.092733   | 2.77x   | -63.86%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.256624 | 0.011398   | 22.51x  | -95.56%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.256624 | 0.012829   | 20.00x  | -95.00%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.411996 | 0.153536   | 2.68x   | -62.73%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.411996 | 0.013266   | 31.06x  | -96.78%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.411996 | 0.012056   | 34.17x  | -97.07%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.362975 | 0.092359   | 3.93x   | -74.55%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.362975 | 0.011296   | 32.13x  | -96.89%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.362975 | 0.012320   | 29.46x  | -96.61%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.266096 | 0.093514   | 2.85x   | -64.86%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.266096 | 0.015122   | 17.60x  | -94.32%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.266096 | 0.012803   | 20.78x  | -95.19%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.272914 | 0.126973   | 2.15x   | -53.48%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.272914 | 0.012171   | 22.42x  | -95.54%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.272914 | 0.012183   | 22.40x  | -95.54%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.267229 | 0.108027   | 2.47x   | -59.58%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.267229 | 0.011496   | 23.25x  | -95.70%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.267229 | 0.011917   | 22.42x  | -95.54%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.267954 | 0.092104   | 2.91x   | -65.63%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.267954 | 0.012272   | 21.84x  | -95.42%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.267954 | 0.012275   | 21.83x  | -95.42%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.376787 | 0.153387   | 2.46x   | -59.29%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.376787 | 0.012655   | 29.77x  | -96.64%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.376787 | 0.012403   | 30.38x  | -96.71%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.305517 | 0.029123   | 10.49x  | -90.47%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.305517 | 0.012015   | 25.43x  | -96.07%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.305517 | 0.009560   | 31.96x  | -96.87%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.408800 | 0.036121   | 11.32x  | -91.16%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.408800 | 0.013144   | 31.10x  | -96.78%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.408800 | 0.011473   | 35.63x  | -97.19%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.295340 | 0.042451   | 6.96x   | -85.63%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.295340 | 0.012083   | 24.44x  | -95.91%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.295340 | 0.011502   | 25.68x  | -96.11%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.313871 | 0.034463   | 9.11x   | -89.02%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.313871 | 0.013583   | 23.11x  | -95.67%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.313871 | 0.012750   | 24.62x  | -95.94%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.316639 | 0.036345   | 8.71x   | -88.52%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.316639 | 0.014005   | 22.61x  | -95.58%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.316639 | 0.013587   | 23.31x  | -95.71%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.299737 | 0.089580   | 3.35x   | -70.11%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.299737 | 0.013761   | 21.78x  | -95.41%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.299737 | 0.010823   | 27.69x  | -96.39%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.294298 | 0.040900   | 7.20x   | -86.10%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.294298 | 0.012503   | 23.54x  | -95.75%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.294298 | 0.010720   | 27.45x  | -96.36%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.300986 | 0.030923   | 9.73x   | -89.73%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.300986 | 0.012593   | 23.90x  | -95.82%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.300986 | 0.010852   | 27.74x  | -96.39%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.441612 | 0.102053   | 4.33x   | -76.89%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.441612 | 0.013880   | 31.82x  | -96.86%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.441612 | 0.011320   | 39.01x  | -97.44%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.421776 | 0.033208   | 12.70x  | -92.13%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.421776 | 0.012775   | 33.02x  | -96.97%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.421776 | 0.010322   | 40.86x  | -97.55%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.313903 | 0.041755   | 7.52x   | -86.70%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.313903 | 0.014180   | 22.14x  | -95.48%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.313903 | 0.011696   | 26.84x  | -96.27%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.334773 | 0.058128   | 5.76x   | -82.64%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.334773 | 0.013425   | 24.94x  | -95.99%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.334773 | 0.011439   | 29.26x  | -96.58%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.310197 | 0.059041   | 5.25x   | -80.97%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.310197 | 0.013428   | 23.10x  | -95.67%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.310197 | 0.011655   | 26.61x  | -96.24%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.320064 | 0.035838   | 8.93x   | -88.80%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.320064 | 0.013432   | 23.83x  | -95.80%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.320064 | 0.010993   | 29.12x  | -96.57%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.413006 | 0.098103   | 4.21x   | -76.25%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.413006 | 0.014397   | 28.69x  | -96.51%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.413006 | 0.013165   | 31.37x  | -96.81%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.256291 | 0.041672   | 6.15x   | -83.74%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.256291 | 0.018557   | 13.81x  | -92.76%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.256291 | 0.017141   | 14.95x  | -93.31%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.367444 | 0.048439   | 7.59x   | -86.82%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.367444 | 0.017631   | 20.84x  | -95.20%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.367444 | 0.016636   | 22.09x  | -95.47%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.255630 | 0.055481   | 4.61x   | -78.30%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.255630 | 0.018815   | 13.59x  | -92.64%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.255630 | 0.017755   | 14.40x  | -93.05%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.275828 | 0.045345   | 6.08x   | -83.56%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.275828 | 0.017895   | 15.41x  | -93.51%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.275828 | 0.017071   | 16.16x  | -93.81%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.262059 | 0.048106   | 5.45x   | -81.64%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.262059 | 0.019240   | 13.62x  | -92.66%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.262059 | 0.016765   | 15.63x  | -93.60%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.245661 | 0.082615   | 2.97x   | -66.37%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.245661 | 0.017821   | 13.78x  | -92.75%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.245661 | 0.017370   | 14.14x  | -92.93%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.236636 | 0.050062   | 4.73x   | -78.84%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.236636 | 0.016390   | 14.44x  | -93.07%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.236636 | 0.016655   | 14.21x  | -92.96%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.243727 | 0.042963   | 5.67x   | -82.37%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.243727 | 0.016717   | 14.58x  | -93.14%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.243727 | 0.016868   | 14.45x  | -93.08%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.383879 | 0.112094   | 3.42x   | -70.80%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.383879 | 0.019739   | 19.45x  | -94.86%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.383879 | 0.017266   | 22.23x  | -95.50%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.341531 | 0.043870   | 7.78x   | -87.15%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.341531 | 0.017368   | 19.66x  | -94.91%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.341531 | 0.016756   | 20.38x  | -95.09%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.247166 | 0.056263   | 4.39x   | -77.24%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.247166 | 0.018464   | 13.39x  | -92.53%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.247166 | 0.017502   | 14.12x  | -92.92%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.257253 | 0.066571   | 3.86x   | -74.12%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.257253 | 0.016874   | 15.25x  | -93.44%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.257253 | 0.016514   | 15.58x  | -93.58%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.255229 | 0.068578   | 3.72x   | -73.13%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.255229 | 0.017690   | 14.43x  | -93.07%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.255229 | 0.016693   | 15.29x  | -93.46%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.259514 | 0.047232   | 5.49x   | -81.80%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.259514 | 0.018551   | 13.99x  | -92.85%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.259514 | 0.016702   | 15.54x  | -93.56%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.369206 | 0.106108   | 3.48x   | -71.26%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.369206 | 0.018854   | 19.58x  | -94.89%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.369206 | 0.017660   | 20.91x  | -95.22%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.759845 | 0.210865   | 3.60x   | -72.25%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.759845 | 0.186465   | 4.08x   | -75.46%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.759845 | 0.199835   | 3.80x   | -73.70%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 1.002996 | 0.252910   | 3.97x   | -74.78%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 1.002996 | 0.189806   | 5.28x   | -81.08%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 1.002996 | 0.178521   | 5.62x   | -82.20%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.767081 | 0.274525   | 2.79x   | -64.21%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.767081 | 0.185318   | 4.14x   | -75.84%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.767081 | 0.180545   | 4.25x   | -76.46%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.816200 | 0.242382   | 3.37x   | -70.30%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.816200 | 0.187498   | 4.35x   | -77.03%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.816200 | 0.170585   | 4.78x   | -79.10%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.791191 | 0.238913   | 3.31x   | -69.80%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.791191 | 0.188059   | 4.21x   | -76.23%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.791191 | 0.175808   | 4.50x   | -77.78%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.764032 | 0.327696   | 2.33x   | -57.11%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.764032 | 0.186291   | 4.10x   | -75.62%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.764032 | 0.176533   | 4.33x   | -76.89%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.755106 | 0.276745   | 2.73x   | -63.35%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.755106 | 0.184314   | 4.10x   | -75.59%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.755106 | 0.176071   | 4.29x   | -76.68%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.774928 | 0.240710   | 3.22x   | -68.94%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.774928 | 0.185091   | 4.19x   | -76.12%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.774928 | 0.174838   | 4.43x   | -77.44%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.084018 | 0.397572   | 2.73x   | -63.32%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.084018 | 0.190577   | 5.69x   | -82.42%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.084018 | 0.177654   | 6.10x   | -83.61%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.985409 | 0.240007   | 4.11x   | -75.64%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.985409 | 0.185515   | 5.31x   | -81.17%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.985409 | 0.171198   | 5.76x   | -82.63%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.758036 | 0.215881   | 3.51x   | -71.52%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.758036 | 0.187039   | 4.05x   | -75.33%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.758036 | 0.176333   | 4.30x   | -76.74%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.775801 | 0.287987   | 2.69x   | -62.88%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.775801 | 0.188320   | 4.12x   | -75.73%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.775801 | 0.179411   | 4.32x   | -76.87%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.822291 | 0.297607   | 2.76x   | -63.81%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.822291 | 0.187178   | 4.39x   | -77.24%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.822291 | 0.178777   | 4.60x   | -78.26%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.827018 | 0.257217   | 3.22x   | -68.90%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.827018 | 0.187724   | 4.41x   | -77.30%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.827018 | 0.175108   | 4.72x   | -78.83%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 1.049704 | 0.390755   | 2.69x   | -62.77%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 1.049704 | 0.194421   | 5.40x   | -81.48%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 1.049704 | 0.182049   | 5.77x   | -82.66%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.576847 | 0.178634   | 3.23x   | -69.03%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.576847 | 0.023481   | 24.57x  | -95.93%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.576847 | 0.021238   | 27.16x  | -96.32%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.802917 | 0.210396   | 3.82x   | -73.80%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.802917 | 0.027060   | 29.67x  | -96.63%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.802917 | 0.027197   | 29.52x  | -96.61%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.543699 | 0.227471   | 2.39x   | -58.16%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.543699 | 0.024686   | 22.02x  | -95.46%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.543699 | 0.026512   | 20.51x  | -95.12%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.592076 | 0.200379   | 2.95x   | -66.16%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.592076 | 0.027003   | 21.93x  | -95.44%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.592076 | 0.027080   | 21.86x  | -95.43%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.640938 | 0.212604   | 3.01x   | -66.83%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.640938 | 0.029629   | 21.63x  | -95.38%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.640938 | 0.027398   | 23.39x  | -95.73%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.565809 | 0.264950   | 2.14x   | -53.17%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.565809 | 0.034994   | 16.17x  | -93.82%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.565809 | 0.027672   | 20.45x  | -95.11%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.563926 | 0.228318   | 2.47x   | -59.51%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.563926 | 0.025162   | 22.41x  | -95.54%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.563926 | 0.027694   | 20.36x  | -95.09%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.566694 | 0.204774   | 2.77x   | -63.87%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.566694 | 0.026382   | 21.48x  | -95.34%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.566694 | 0.027040   | 20.96x  | -95.23%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.866878 | 0.347056   | 2.50x   | -59.96%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.866878 | 0.030245   | 28.66x  | -96.51%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.866878 | 0.027291   | 31.76x  | -96.85%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.763483 | 0.199296   | 3.83x   | -73.90%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.763483 | 0.024988   | 30.55x  | -96.73%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.763483 | 0.026862   | 28.42x  | -96.48%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.552055 | 0.197195   | 2.80x   | -64.28%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.552055 | 0.033344   | 16.56x  | -93.96%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.552055 | 0.027945   | 19.76x  | -94.94%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.575842 | 0.249007   | 2.31x   | -56.76%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.575842 | 0.026510   | 21.72x  | -95.40%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.575842 | 0.027160   | 21.20x  | -95.28%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.563675 | 0.244155   | 2.31x   | -56.69%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.563675 | 0.025838   | 21.82x  | -95.42%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.563675 | 0.026714   | 21.10x  | -95.26%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.570979 | 0.202732   | 2.82x   | -64.49%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.570979 | 0.026654   | 21.42x  | -95.33%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.570979 | 0.026791   | 21.31x  | -95.31%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.778367 | 0.336292   | 2.31x   | -56.80%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.778367 | 0.027880   | 27.92x  | -96.42%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.778367 | 0.027217   | 28.60x  | -96.50%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.626596 | 0.072429   | 8.65x   | -88.44%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.626596 | 0.034820   | 18.00x  | -94.44%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.626596 | 0.029332   | 21.36x  | -95.32%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.836907 | 0.086833   | 9.64x   | -89.62%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.836907 | 0.037698   | 22.20x  | -95.50%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.836907 | 0.033503   | 24.98x  | -96.00%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.611715 | 0.100274   | 6.10x   | -83.61%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.611715 | 0.035375   | 17.29x  | -94.22%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.611715 | 0.032858   | 18.62x  | -94.63%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.655010 | 0.079482   | 8.24x   | -87.87%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.655010 | 0.036710   | 17.84x  | -94.40%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.655010 | 0.037253   | 17.58x  | -94.31%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.656757 | 0.089793   | 7.31x   | -86.33%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.656757 | 0.039855   | 16.48x  | -93.93%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.656757 | 0.036860   | 17.82x  | -94.39%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.631061 | 0.207709   | 3.04x   | -67.09%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.631061 | 0.038294   | 16.48x  | -93.93%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.631061 | 0.033095   | 19.07x  | -94.76%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.610309 | 0.100223   | 6.09x   | -83.58%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.610309 | 0.035152   | 17.36x  | -94.24%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.610309 | 0.032409   | 18.83x  | -94.69%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.634072 | 0.077782   | 8.15x   | -87.73%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.634072 | 0.035987   | 17.62x  | -94.32%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.634072 | 0.032459   | 19.53x  | -94.88%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.939193 | 0.242054   | 3.88x   | -74.23%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.939193 | 0.040299   | 23.31x  | -95.71%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.939193 | 0.033313   | 28.19x  | -96.45%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.853033 | 0.078513   | 10.86x  | -90.80%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.853033 | 0.035999   | 23.70x  | -95.78%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.853033 | 0.033058   | 25.80x  | -96.12%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.635300 | 0.097430   | 6.52x   | -84.66%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.635300 | 0.039042   | 16.27x  | -93.85%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.635300 | 0.033466   | 18.98x  | -94.73%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.653619 | 0.136933   | 4.77x   | -79.05%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.653619 | 0.037646   | 17.36x  | -94.24%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.653619 | 0.032710   | 19.98x  | -95.00%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.649490 | 0.136789   | 4.75x   | -78.94%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.649490 | 0.036940   | 17.58x  | -94.31%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.649490 | 0.032861   | 19.76x  | -94.94%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.666149 | 0.087120   | 7.65x   | -86.92%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.666149 | 0.039165   | 17.01x  | -94.12%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.666149 | 0.033665   | 19.79x  | -94.95%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.878391 | 0.224781   | 3.91x   | -74.41%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.878391 | 0.038523   | 22.80x  | -95.61%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.878391 | 0.033074   | 26.56x  | -96.23%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.495727 | 0.089952   | 5.51x   | -81.85%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.495727 | 0.036305   | 13.65x  | -92.68%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.495727 | 0.032908   | 15.06x  | -93.36%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.713180 | 0.101614   | 7.02x   | -85.75%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.713180 | 0.035360   | 20.17x  | -95.04%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.713180 | 0.034942   | 20.41x  | -95.10%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.484756 | 0.108979   | 4.45x   | -77.52%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.484756 | 0.033256   | 14.58x  | -93.14%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.484756 | 0.034621   | 14.00x  | -92.86%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.521516 | 0.096457   | 5.41x   | -81.50%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.521516 | 0.034755   | 15.01x  | -93.34%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.521516 | 0.034582   | 15.08x  | -93.37%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.518123 | 0.104987   | 4.94x   | -79.74%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.518123 | 0.038846   | 13.34x  | -92.50%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.518123 | 0.034418   | 15.05x  | -93.36%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.496346 | 0.181931   | 2.73x   | -63.35%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.496346 | 0.036122   | 13.74x  | -92.72%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.496346 | 0.035370   | 14.03x  | -92.87%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.474921 | 0.108545   | 4.38x   | -77.14%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.474921 | 0.034480   | 13.77x  | -92.74%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.474921 | 0.034777   | 13.66x  | -92.68%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.495267 | 0.092902   | 5.33x   | -81.24%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.495267 | 0.033867   | 14.62x  | -93.16%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.495267 | 0.035116   | 14.10x  | -92.91%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.801933 | 0.248279   | 3.23x   | -69.04%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.801933 | 0.040099   | 20.00x  | -95.00%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.801933 | 0.034893   | 22.98x  | -95.65%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.717111 | 0.093863   | 7.64x   | -86.91%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.717111 | 0.034294   | 20.91x  | -95.22%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.717111 | 0.034981   | 20.50x  | -95.12%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.496067 | 0.122025   | 4.07x   | -75.40%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.496067 | 0.037721   | 13.15x  | -92.40%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.496067 | 0.035826   | 13.85x  | -92.78%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.513086 | 0.146628   | 3.50x   | -71.42%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.513086 | 0.035102   | 14.62x  | -93.16%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.513086 | 0.034698   | 14.79x  | -93.24%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.515198 | 0.145211   | 3.55x   | -71.81%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.515198 | 0.034857   | 14.78x  | -93.23%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.515198 | 0.034872   | 14.77x  | -93.23%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.521065 | 0.101741   | 5.12x   | -80.47%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.521065 | 0.035356   | 14.74x  | -93.21%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.521065 | 0.035118   | 14.84x  | -93.26%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.726922 | 0.233442   | 3.11x   | -67.89%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.726922 | 0.037552   | 19.36x  | -94.83%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.726922 | 0.035578   | 20.43x  | -95.11%        |
</details>
<!-- PERF_RESULTS_END -->
