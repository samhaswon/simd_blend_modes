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
| normal        | scalar | 0.197654 | 0.042761   | 4.62x   | -78.37%        |
| normal        | sse42  | 0.197654 | 0.021663   | 9.12x   | -89.04%        |
| normal        | avx2   | 0.197654 | 0.021751   | 9.09x   | -89.00%        |
| soft_light    | scalar | 0.268099 | 0.051905   | 5.17x   | -80.64%        |
| soft_light    | sse42  | 0.268099 | 0.023330   | 11.49x  | -91.30%        |
| soft_light    | avx2   | 0.268099 | 0.022521   | 11.90x  | -91.60%        |
| lighten_only  | scalar | 0.196839 | 0.054581   | 3.61x   | -72.27%        |
| lighten_only  | sse42  | 0.196839 | 0.023264   | 8.46x   | -88.18%        |
| lighten_only  | avx2   | 0.196839 | 0.021995   | 8.95x   | -88.83%        |
| screen        | scalar | 0.208626 | 0.049238   | 4.24x   | -76.40%        |
| screen        | sse42  | 0.208626 | 0.022784   | 9.16x   | -89.08%        |
| screen        | avx2   | 0.208626 | 0.022123   | 9.43x   | -89.40%        |
| dodge         | scalar | 0.210086 | 0.053298   | 3.94x   | -74.63%        |
| dodge         | sse42  | 0.210086 | 0.024498   | 8.58x   | -88.34%        |
| dodge         | avx2   | 0.210086 | 0.023146   | 9.08x   | -88.98%        |
| addition      | scalar | 0.202519 | 0.079320   | 2.55x   | -60.83%        |
| addition      | sse42  | 0.202519 | 0.023517   | 8.61x   | -88.39%        |
| addition      | avx2   | 0.202519 | 0.022336   | 9.07x   | -88.97%        |
| darken_only   | scalar | 0.199671 | 0.054371   | 3.67x   | -72.77%        |
| darken_only   | sse42  | 0.199671 | 0.022634   | 8.82x   | -88.66%        |
| darken_only   | avx2   | 0.199671 | 0.022126   | 9.02x   | -88.92%        |
| multiply      | scalar | 0.201891 | 0.048310   | 4.18x   | -76.07%        |
| multiply      | sse42  | 0.201891 | 0.023082   | 8.75x   | -88.57%        |
| multiply      | avx2   | 0.201891 | 0.022540   | 8.96x   | -88.84%        |
| hard_light    | scalar | 0.299268 | 0.097070   | 3.08x   | -67.56%        |
| hard_light    | sse42  | 0.299268 | 0.024079   | 12.43x  | -91.95%        |
| hard_light    | avx2   | 0.299268 | 0.022773   | 13.14x  | -92.39%        |
| difference    | scalar | 0.271178 | 0.048395   | 5.60x   | -82.15%        |
| difference    | sse42  | 0.271178 | 0.022580   | 12.01x  | -91.67%        |
| difference    | avx2   | 0.271178 | 0.022370   | 12.12x  | -91.75%        |
| subtract      | scalar | 0.202934 | 0.052904   | 3.84x   | -73.93%        |
| subtract      | sse42  | 0.202934 | 0.023542   | 8.62x   | -88.40%        |
| subtract      | avx2   | 0.202934 | 0.022316   | 9.09x   | -89.00%        |
| grain_extract | scalar | 0.206477 | 0.064492   | 3.20x   | -68.77%        |
| grain_extract | sse42  | 0.206477 | 0.023268   | 8.87x   | -88.73%        |
| grain_extract | avx2   | 0.206477 | 0.022531   | 9.16x   | -89.09%        |
| grain_merge   | scalar | 0.206704 | 0.064383   | 3.21x   | -68.85%        |
| grain_merge   | sse42  | 0.206704 | 0.023435   | 8.82x   | -88.66%        |
| grain_merge   | avx2   | 0.206704 | 0.022729   | 9.09x   | -89.00%        |
| divide        | scalar | 0.216468 | 0.052359   | 4.13x   | -75.81%        |
| divide        | sse42  | 0.216468 | 0.024030   | 9.01x   | -88.90%        |
| divide        | avx2   | 0.216468 | 0.022700   | 9.54x   | -89.51%        |
| overlay       | scalar | 0.279752 | 0.092357   | 3.03x   | -66.99%        |
| overlay       | sse42  | 0.279752 | 0.023488   | 11.91x  | -91.60%        |
| overlay       | avx2   | 0.279752 | 0.022689   | 12.33x  | -91.89%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.007170 | 0.001780   | 4.03x   | -75.17%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.007170 | 0.001521   | 4.71x   | -78.78%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.007170 | 0.001561   | 4.59x   | -78.23%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.009906 | 0.002041   | 4.85x   | -79.40%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.009906 | 0.001590   | 6.23x   | -83.95%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.009906 | 0.001619   | 6.12x   | -83.65%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.007343 | 0.001987   | 3.70x   | -72.94%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.007343 | 0.001487   | 4.94x   | -79.75%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.007343 | 0.001505   | 4.88x   | -79.50%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.007973 | 0.001789   | 4.46x   | -77.56%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.007973 | 0.001599   | 4.99x   | -79.95%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.007973 | 0.001556   | 5.12x   | -80.49%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.008031 | 0.002015   | 3.99x   | -74.91%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.008031 | 0.001542   | 5.21x   | -80.80%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.008031 | 0.001568   | 5.12x   | -80.47%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.007888 | 0.002741   | 2.88x   | -65.25%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.007888 | 0.001523   | 5.18x   | -80.69%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.007888 | 0.001516   | 5.20x   | -80.78%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.007548 | 0.001996   | 3.78x   | -73.56%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.007548 | 0.001499   | 5.04x   | -80.14%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.007548 | 0.001516   | 4.98x   | -79.91%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.007557 | 0.001847   | 4.09x   | -75.56%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.007557 | 0.001521   | 4.97x   | -79.88%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.007557 | 0.001635   | 4.62x   | -78.37%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.009630 | 0.003058   | 3.15x   | -68.24%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.009630 | 0.001482   | 6.50x   | -84.61%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.009630 | 0.001519   | 6.34x   | -84.23%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.011070 | 0.001906   | 5.81x   | -82.78%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.011070 | 0.001522   | 7.27x   | -86.25%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.011070 | 0.001506   | 7.35x   | -86.39%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.007405 | 0.001830   | 4.05x   | -75.28%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.007405 | 0.001484   | 4.99x   | -79.96%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.007405 | 0.001496   | 4.95x   | -79.80%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.007324 | 0.002317   | 3.16x   | -68.37%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.007324 | 0.001605   | 4.56x   | -78.08%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.007324 | 0.001515   | 4.83x   | -79.31%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.007497 | 0.002432   | 3.08x   | -67.56%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.007497 | 0.001600   | 4.69x   | -78.66%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.007497 | 0.001627   | 4.61x   | -78.29%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.007521 | 0.001935   | 3.89x   | -74.27%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.007521 | 0.001575   | 4.77x   | -79.05%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.007521 | 0.001587   | 4.74x   | -78.90%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.009277 | 0.002985   | 3.11x   | -67.82%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.009277 | 0.001608   | 5.77x   | -82.66%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.009277 | 0.001648   | 5.63x   | -82.24%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003448 | 0.001439   | 2.40x   | -58.26%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003448 | 0.000187   | 18.48x  | -94.59%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003448 | 0.000172   | 20.08x  | -95.02%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.007632 | 0.001813   | 4.21x   | -76.25%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.007632 | 0.000228   | 33.43x  | -97.01%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.007632 | 0.000189   | 40.34x  | -97.52%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.006099 | 0.001926   | 3.17x   | -68.42%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.006099 | 0.000209   | 29.23x  | -96.58%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.006099 | 0.000213   | 28.64x  | -96.51%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.006447 | 0.001893   | 3.41x   | -70.64%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.006447 | 0.000229   | 28.15x  | -96.45%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.006447 | 0.000196   | 32.81x  | -96.95%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.005979 | 0.001771   | 3.38x   | -70.38%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.005979 | 0.000237   | 25.27x  | -96.04%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.005979 | 0.000194   | 30.75x  | -96.75%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.006536 | 0.002340   | 2.79x   | -64.20%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.006536 | 0.000275   | 23.80x  | -95.80%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.006536 | 0.000199   | 32.89x  | -96.96%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.006490 | 0.001802   | 3.60x   | -72.23%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.006490 | 0.000205   | 31.73x  | -96.85%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.006490 | 0.000220   | 29.54x  | -96.61%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.005747 | 0.001812   | 3.17x   | -68.47%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.005747 | 0.000227   | 25.32x  | -96.05%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.005747 | 0.000212   | 27.13x  | -96.31%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.009011 | 0.002930   | 3.07x   | -67.48%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.009011 | 0.000266   | 33.91x  | -97.05%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.009011 | 0.000218   | 41.37x  | -97.58%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.007748 | 0.001639   | 4.73x   | -78.85%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.007748 | 0.000283   | 27.42x  | -96.35%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.007748 | 0.000214   | 36.24x  | -97.24%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.006761 | 0.001737   | 3.89x   | -74.30%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.006761 | 0.000268   | 25.19x  | -96.03%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.006761 | 0.000196   | 34.46x  | -97.10%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.005606 | 0.002011   | 2.79x   | -64.13%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.005606 | 0.000213   | 26.29x  | -96.20%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.005606 | 0.000194   | 28.88x  | -96.54%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.005750 | 0.001949   | 2.95x   | -66.11%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.005750 | 0.000207   | 27.74x  | -96.40%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.005750 | 0.000186   | 31.00x  | -96.77%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.005700 | 0.001681   | 3.39x   | -70.52%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.005700 | 0.000215   | 26.55x  | -96.23%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.005700 | 0.000187   | 30.56x  | -96.73%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.007048 | 0.002574   | 2.74x   | -63.48%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.007048 | 0.000227   | 31.05x  | -96.78%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.007048 | 0.000191   | 36.91x  | -97.29%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.005552 | 0.000510   | 10.90x  | -90.82%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.005552 | 0.000248   | 22.39x  | -95.53%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.005552 | 0.000236   | 23.56x  | -95.76%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.008694 | 0.000793   | 10.97x  | -90.88%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.008694 | 0.000292   | 29.74x  | -96.64%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.008694 | 0.000248   | 35.11x  | -97.15%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.007332 | 0.000750   | 9.77x   | -89.77%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.007332 | 0.000273   | 26.90x  | -96.28%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.007332 | 0.000251   | 29.20x  | -96.58%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.007981 | 0.000664   | 12.02x  | -91.68%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.007981 | 0.000317   | 25.15x  | -96.02%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.007981 | 0.000270   | 29.59x  | -96.62%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.007573 | 0.000702   | 10.78x  | -90.73%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.007573 | 0.000320   | 23.68x  | -95.78%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.007573 | 0.000249   | 30.46x  | -96.72%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.007982 | 0.001654   | 4.83x   | -79.28%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.007982 | 0.000282   | 28.30x  | -96.47%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.007982 | 0.000235   | 33.98x  | -97.06%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.007513 | 0.000807   | 9.31x   | -89.26%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.007513 | 0.000278   | 27.02x  | -96.30%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.007513 | 0.000223   | 33.67x  | -97.03%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.007392 | 0.000580   | 12.75x  | -92.16%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.007392 | 0.000279   | 26.49x  | -96.22%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.007392 | 0.000242   | 30.56x  | -96.73%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.009812 | 0.001929   | 5.09x   | -80.34%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.009812 | 0.000317   | 30.98x  | -96.77%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.009812 | 0.000268   | 36.60x  | -97.27%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.009038 | 0.000599   | 15.08x  | -93.37%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.009038 | 0.000275   | 32.90x  | -96.96%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.009038 | 0.000229   | 39.43x  | -97.46%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.007327 | 0.000778   | 9.41x   | -89.38%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.007327 | 0.000286   | 25.65x  | -96.10%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.007327 | 0.000254   | 28.86x  | -96.53%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.007181 | 0.001043   | 6.88x   | -85.47%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.007181 | 0.000274   | 26.22x  | -96.19%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.007181 | 0.000235   | 30.55x  | -96.73%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.007332 | 0.001097   | 6.68x   | -85.03%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.007332 | 0.000322   | 22.74x  | -95.60%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.007332 | 0.000239   | 30.71x  | -96.74%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.007437 | 0.000652   | 11.41x  | -91.23%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.007437 | 0.000298   | 24.91x  | -95.99%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.007437 | 0.000263   | 28.24x  | -96.46%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.009000 | 0.001790   | 5.03x   | -80.11%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.009000 | 0.000294   | 30.61x  | -96.73%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.009000 | 0.000250   | 36.00x  | -97.22%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.004295 | 0.000633   | 6.78x   | -85.25%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.004295 | 0.000202   | 21.22x  | -95.29%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.004295 | 0.000164   | 26.25x  | -96.19%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.006819 | 0.000803   | 8.49x   | -88.23%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.006819 | 0.000258   | 26.46x  | -96.22%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.006819 | 0.000186   | 36.66x  | -97.27%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.005517 | 0.000883   | 6.25x   | -84.00%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.005517 | 0.000242   | 22.83x  | -95.62%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.005517 | 0.000187   | 29.52x  | -96.61%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.005594 | 0.000828   | 6.76x   | -85.20%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.005594 | 0.000237   | 23.58x  | -95.76%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.005594 | 0.000186   | 30.04x  | -96.67%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.005665 | 0.000817   | 6.93x   | -85.58%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.005665 | 0.000269   | 21.07x  | -95.25%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.005665 | 0.000188   | 30.11x  | -96.68%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.006042 | 0.001462   | 4.13x   | -75.79%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.006042 | 0.000230   | 26.25x  | -96.19%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.006042 | 0.000195   | 31.04x  | -96.78%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.005645 | 0.000908   | 6.22x   | -83.91%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.005645 | 0.000234   | 24.17x  | -95.86%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.005645 | 0.000196   | 28.76x  | -96.52%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.005492 | 0.000727   | 7.56x   | -86.77%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.005492 | 0.000241   | 22.82x  | -95.62%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.005492 | 0.000190   | 28.91x  | -96.54%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.007345 | 0.001959   | 3.75x   | -73.33%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.007345 | 0.000262   | 28.00x  | -96.43%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.007345 | 0.000184   | 39.91x  | -97.49%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.007192 | 0.000719   | 10.00x  | -90.00%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.007192 | 0.000381   | 18.89x  | -94.71%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.007192 | 0.000180   | 39.97x  | -97.50%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.005823 | 0.000997   | 5.84x   | -82.87%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.005823 | 0.000241   | 24.18x  | -95.86%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.005823 | 0.000203   | 28.62x  | -96.51%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.005655 | 0.001233   | 4.59x   | -78.20%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.005655 | 0.000268   | 21.11x  | -95.26%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.005655 | 0.000208   | 27.20x  | -96.32%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.005602 | 0.001133   | 4.94x   | -79.77%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.005602 | 0.000253   | 22.15x  | -95.48%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.005602 | 0.000185   | 30.26x  | -96.70%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.005737 | 0.000784   | 7.32x   | -86.34%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.005737 | 0.000257   | 22.36x  | -95.53%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.005737 | 0.000197   | 29.16x  | -96.57%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.007015 | 0.001826   | 3.84x   | -73.97%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.007015 | 0.000265   | 26.44x  | -96.22%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.007015 | 0.000183   | 38.31x  | -97.39%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.035662 | 0.006457   | 5.52x   | -81.89%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.035662 | 0.005648   | 6.31x   | -84.16%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.035662 | 0.006143   | 5.81x   | -82.77%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.045595 | 0.007453   | 6.12x   | -83.65%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.045595 | 0.005771   | 7.90x   | -87.34%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.045595 | 0.006040   | 7.55x   | -86.75%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.039417 | 0.007625   | 5.17x   | -80.65%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.039417 | 0.005630   | 7.00x   | -85.72%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.039417 | 0.005872   | 6.71x   | -85.10%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.038892 | 0.006996   | 5.56x   | -82.01%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.038892 | 0.006028   | 6.45x   | -84.50%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.038892 | 0.005913   | 6.58x   | -84.80%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.038108 | 0.007418   | 5.14x   | -80.54%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.038108 | 0.005843   | 6.52x   | -84.67%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.038108 | 0.005944   | 6.41x   | -84.40%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.037353 | 0.010583   | 3.53x   | -71.67%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.037353 | 0.005721   | 6.53x   | -84.68%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.037353 | 0.005828   | 6.41x   | -84.40%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.038777 | 0.010109   | 3.84x   | -73.93%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.038777 | 0.006215   | 6.24x   | -83.97%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.038777 | 0.005892   | 6.58x   | -84.80%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.040619 | 0.007762   | 5.23x   | -80.89%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.040619 | 0.006303   | 6.44x   | -84.48%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.040619 | 0.006015   | 6.75x   | -85.19%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.049707 | 0.012370   | 4.02x   | -75.11%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.049707 | 0.006092   | 8.16x   | -87.74%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.049707 | 0.006489   | 7.66x   | -86.95%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.049227 | 0.007263   | 6.78x   | -85.25%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.049227 | 0.005903   | 8.34x   | -88.01%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.049227 | 0.006504   | 7.57x   | -86.79%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.042354 | 0.007455   | 5.68x   | -82.40%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.042354 | 0.005935   | 7.14x   | -85.99%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.042354 | 0.005884   | 7.20x   | -86.11%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.038351 | 0.008607   | 4.46x   | -77.56%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.038351 | 0.005786   | 6.63x   | -84.91%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.038351 | 0.005967   | 6.43x   | -84.44%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.039396 | 0.008864   | 4.44x   | -77.50%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.039396 | 0.005998   | 6.57x   | -84.77%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.039396 | 0.005879   | 6.70x   | -85.08%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.038780 | 0.007496   | 5.17x   | -80.67%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.038780 | 0.006057   | 6.40x   | -84.38%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.038780 | 0.006035   | 6.43x   | -84.44%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.045268 | 0.011563   | 3.91x   | -74.46%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.045268 | 0.005818   | 7.78x   | -87.15%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.045268 | 0.005946   | 7.61x   | -86.87%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.024722 | 0.005502   | 4.49x   | -77.74%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.024722 | 0.000741   | 33.37x  | -97.00%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.024722 | 0.000668   | 36.99x  | -97.30%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.036272 | 0.007133   | 5.09x   | -80.34%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.036272 | 0.000856   | 42.37x  | -97.64%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.036272 | 0.000744   | 48.75x  | -97.95%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.030000 | 0.006797   | 4.41x   | -77.34%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.030000 | 0.000787   | 38.14x  | -97.38%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.030000 | 0.000765   | 39.23x  | -97.45%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.030114 | 0.006418   | 4.69x   | -78.69%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.030114 | 0.000822   | 36.64x  | -97.27%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.030114 | 0.000738   | 40.82x  | -97.55%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.029892 | 0.006831   | 4.38x   | -77.15%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.029892 | 0.000913   | 32.75x  | -96.95%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.029892 | 0.000740   | 40.39x  | -97.52%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.028318 | 0.008331   | 3.40x   | -70.58%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.028318 | 0.001048   | 27.02x  | -96.30%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.028318 | 0.000780   | 36.32x  | -97.25%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.029068 | 0.006857   | 4.24x   | -76.41%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.029068 | 0.000781   | 37.24x  | -97.31%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.029068 | 0.000756   | 38.43x  | -97.40%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.029657 | 0.006459   | 4.59x   | -78.22%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.029657 | 0.000796   | 37.27x  | -97.32%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.029657 | 0.000744   | 39.88x  | -97.49%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.037895 | 0.011024   | 3.44x   | -70.91%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.037895 | 0.000940   | 40.30x  | -97.52%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.037895 | 0.000762   | 49.75x  | -97.99%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.037590 | 0.006655   | 5.65x   | -82.30%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.037590 | 0.000812   | 46.27x  | -97.84%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.037590 | 0.000754   | 49.88x  | -98.00%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.030247 | 0.006983   | 4.33x   | -76.91%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.030247 | 0.001044   | 28.96x  | -96.55%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.030247 | 0.000769   | 39.32x  | -97.46%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.030123 | 0.008035   | 3.75x   | -73.33%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.030123 | 0.000849   | 35.49x  | -97.18%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.030123 | 0.000760   | 39.63x  | -97.48%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.029833 | 0.007828   | 3.81x   | -73.76%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.029833 | 0.000830   | 35.96x  | -97.22%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.029833 | 0.000746   | 40.00x  | -97.50%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.029877 | 0.006866   | 4.35x   | -77.02%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.029877 | 0.000845   | 35.37x  | -97.17%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.029877 | 0.000737   | 40.55x  | -97.53%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.036195 | 0.010515   | 3.44x   | -70.95%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.036195 | 0.000880   | 41.12x  | -97.57%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.036195 | 0.000751   | 48.22x  | -97.93%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.028340 | 0.002488   | 11.39x  | -91.22%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.028340 | 0.001043   | 27.17x  | -96.32%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.028340 | 0.000922   | 30.72x  | -96.75%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.039577 | 0.003210   | 12.33x  | -91.89%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.039577 | 0.001255   | 31.53x  | -96.83%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.039577 | 0.001206   | 32.83x  | -96.95%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.034038 | 0.003594   | 9.47x   | -89.44%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.034038 | 0.001143   | 29.78x  | -96.64%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.034038 | 0.000955   | 35.65x  | -97.20%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.033952 | 0.002989   | 11.36x  | -91.20%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.033952 | 0.001254   | 27.09x  | -96.31%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.033952 | 0.000959   | 35.42x  | -97.18%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.033812 | 0.003944   | 8.57x   | -88.34%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.033812 | 0.001275   | 26.52x  | -96.23%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.033812 | 0.001132   | 29.87x  | -96.65%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.033465 | 0.007025   | 4.76x   | -79.01%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.033465 | 0.001089   | 30.73x  | -96.75%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.033465 | 0.000933   | 35.87x  | -97.21%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.032581 | 0.003384   | 9.63x   | -89.61%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.032581 | 0.001104   | 29.52x  | -96.61%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.032581 | 0.000878   | 37.12x  | -97.31%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.033071 | 0.002922   | 11.32x  | -91.17%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.033071 | 0.001145   | 28.89x  | -96.54%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.033071 | 0.000944   | 35.02x  | -97.14%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.044972 | 0.008155   | 5.51x   | -81.87%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.044972 | 0.001223   | 36.76x  | -97.28%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.044972 | 0.001077   | 41.74x  | -97.60%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.044985 | 0.003201   | 14.06x  | -92.89%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.044985 | 0.001205   | 37.33x  | -97.32%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.044985 | 0.001224   | 36.76x  | -97.28%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.033951 | 0.003471   | 9.78x   | -89.78%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.033951 | 0.001112   | 30.52x  | -96.72%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.033951 | 0.000948   | 35.80x  | -97.21%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.033102 | 0.004701   | 7.04x   | -85.80%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.033102 | 0.001095   | 30.24x  | -96.69%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.033102 | 0.000943   | 35.10x  | -97.15%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.033426 | 0.004750   | 7.04x   | -85.79%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.033426 | 0.001098   | 30.44x  | -96.71%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.033426 | 0.000942   | 35.48x  | -97.18%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.034992 | 0.004540   | 7.71x   | -87.03%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.034992 | 0.001209   | 28.94x  | -96.55%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.034992 | 0.001035   | 33.80x  | -97.04%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.045752 | 0.007857   | 5.82x   | -82.83%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.045752 | 0.001181   | 38.74x  | -97.42%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.045752 | 0.001038   | 44.07x  | -97.73%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.021916 | 0.002573   | 8.52x   | -88.26%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.021916 | 0.000812   | 26.98x  | -96.29%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.021916 | 0.000777   | 28.20x  | -96.45%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.033394 | 0.003747   | 8.91x   | -88.78%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.033394 | 0.001139   | 29.32x  | -96.59%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.033394 | 0.000876   | 38.13x  | -97.38%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.025056 | 0.003679   | 6.81x   | -85.32%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.025056 | 0.001099   | 22.79x  | -95.61%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.025056 | 0.000868   | 28.86x  | -96.53%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.027326 | 0.003281   | 8.33x   | -87.99%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.027326 | 0.001514   | 18.05x  | -94.46%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.027326 | 0.001215   | 22.49x  | -95.55%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.030493 | 0.003644   | 8.37x   | -88.05%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.030493 | 0.001067   | 28.57x  | -96.50%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.030493 | 0.000839   | 36.35x  | -97.25%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.025428 | 0.005834   | 4.36x   | -77.06%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.025428 | 0.000961   | 26.46x  | -96.22%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.025428 | 0.000835   | 30.47x  | -96.72%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.025758 | 0.003556   | 7.24x   | -86.19%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.025758 | 0.000974   | 26.44x  | -96.22%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.025758 | 0.000802   | 32.13x  | -96.89%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.024920 | 0.003060   | 8.14x   | -87.72%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.024920 | 0.000995   | 25.05x  | -96.01%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.024920 | 0.000845   | 29.47x  | -96.61%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.033728 | 0.007979   | 4.23x   | -76.34%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.033728 | 0.001078   | 31.30x  | -96.80%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.033728 | 0.000810   | 41.62x  | -97.60%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.033897 | 0.003159   | 10.73x  | -90.68%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.033897 | 0.001014   | 33.44x  | -97.01%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.033897 | 0.000802   | 42.26x  | -97.63%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.024403 | 0.003856   | 6.33x   | -84.20%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.024403 | 0.000971   | 25.13x  | -96.02%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.024403 | 0.000850   | 28.71x  | -96.52%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.024838 | 0.004638   | 5.36x   | -81.33%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.024838 | 0.000998   | 24.89x  | -95.98%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.024838 | 0.000916   | 27.13x  | -96.31%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.025177 | 0.004512   | 5.58x   | -82.08%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.025177 | 0.000993   | 25.35x  | -96.06%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.025177 | 0.000820   | 30.72x  | -96.75%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.024886 | 0.003091   | 8.05x   | -87.58%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.024886 | 0.000986   | 25.24x  | -96.04%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.024886 | 0.000819   | 30.37x  | -96.71%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.031756 | 0.007264   | 4.37x   | -77.13%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.031756 | 0.001011   | 31.41x  | -96.82%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.031756 | 0.000839   | 37.86x  | -97.36%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.101841 | 0.026132   | 3.90x   | -74.34%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.101841 | 0.022662   | 4.49x   | -77.75%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.101841 | 0.023536   | 4.33x   | -76.89%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.139677 | 0.030708   | 4.55x   | -78.01%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.139677 | 0.023799   | 5.87x   | -82.96%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.139677 | 0.024394   | 5.73x   | -82.54%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.106342 | 0.032023   | 3.32x   | -69.89%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.106342 | 0.023150   | 4.59x   | -78.23%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.106342 | 0.023877   | 4.45x   | -77.55%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.114128 | 0.029451   | 3.88x   | -74.20%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.114128 | 0.023760   | 4.80x   | -79.18%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.114128 | 0.024099   | 4.74x   | -78.88%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.110642 | 0.030985   | 3.57x   | -71.99%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.110642 | 0.027034   | 4.09x   | -75.57%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.110642 | 0.025432   | 4.35x   | -77.01%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.108259 | 0.044451   | 2.44x   | -58.94%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.108259 | 0.024439   | 4.43x   | -77.43%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.108259 | 0.025211   | 4.29x   | -76.71%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.109301 | 0.032639   | 3.35x   | -70.14%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.109301 | 0.024069   | 4.54x   | -77.98%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.109301 | 0.024716   | 4.42x   | -77.39%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.109919 | 0.032732   | 3.36x   | -70.22%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.109919 | 0.023783   | 4.62x   | -78.36%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.109919 | 0.024134   | 4.55x   | -78.04%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.142525 | 0.049337   | 2.89x   | -65.38%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.142525 | 0.024244   | 5.88x   | -82.99%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.142525 | 0.024347   | 5.85x   | -82.92%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.135548 | 0.028576   | 4.74x   | -78.92%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.135548 | 0.022784   | 5.95x   | -83.19%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.135548 | 0.023680   | 5.72x   | -82.53%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.105929 | 0.028566   | 3.71x   | -73.03%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.105929 | 0.023544   | 4.50x   | -77.77%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.105929 | 0.023867   | 4.44x   | -77.47%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.108791 | 0.034962   | 3.11x   | -67.86%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.108791 | 0.027530   | 3.95x   | -74.69%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.108791 | 0.031574   | 3.45x   | -70.98%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.107926 | 0.035725   | 3.02x   | -66.90%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.107926 | 0.023919   | 4.51x   | -77.84%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.107926 | 0.024558   | 4.39x   | -77.25%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.109669 | 0.029461   | 3.72x   | -73.14%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.109669 | 0.024022   | 4.57x   | -78.10%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.109669 | 0.024225   | 4.53x   | -77.91%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.143556 | 0.047077   | 3.05x   | -67.21%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.143556 | 0.024363   | 5.89x   | -83.03%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.143556 | 0.025186   | 5.70x   | -82.46%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.075240 | 0.021916   | 3.43x   | -70.87%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.075240 | 0.002986   | 25.20x  | -96.03%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.075240 | 0.002737   | 27.49x  | -96.36%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.108242 | 0.028071   | 3.86x   | -74.07%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.108242 | 0.003441   | 31.45x  | -96.82%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.108242 | 0.002998   | 36.10x  | -97.23%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.078726 | 0.028518   | 2.76x   | -63.78%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.078726 | 0.003209   | 24.54x  | -95.92%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.078726 | 0.003062   | 25.71x  | -96.11%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.084607 | 0.026512   | 3.19x   | -68.66%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.084607 | 0.003301   | 25.63x  | -96.10%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.084607 | 0.003017   | 28.04x  | -96.43%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.084954 | 0.028764   | 2.95x   | -66.14%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.084954 | 0.003784   | 22.45x  | -95.55%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.084954 | 0.003090   | 27.49x  | -96.36%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.081391 | 0.034331   | 2.37x   | -57.82%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.081391 | 0.004310   | 18.88x  | -94.70%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.081391 | 0.003149   | 25.85x  | -96.13%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.081429 | 0.028769   | 2.83x   | -64.67%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.081429 | 0.003192   | 25.51x  | -96.08%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.081429 | 0.002973   | 27.39x  | -96.35%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.081843 | 0.028959   | 2.83x   | -64.62%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.081843 | 0.003349   | 24.44x  | -95.91%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.081843 | 0.003112   | 26.30x  | -96.20%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.120395 | 0.045284   | 2.66x   | -62.39%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.120395 | 0.003877   | 31.06x  | -96.78%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.120395 | 0.003053   | 39.43x  | -97.46%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.113894 | 0.027955   | 4.07x   | -75.46%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.113894 | 0.003267   | 34.86x  | -97.13%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.113894 | 0.003036   | 37.52x  | -97.33%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.079268 | 0.027497   | 2.88x   | -65.31%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.079268 | 0.004173   | 18.99x  | -94.74%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.079268 | 0.003112   | 25.48x  | -96.07%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.084010 | 0.033057   | 2.54x   | -60.65%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.084010 | 0.003391   | 24.77x  | -95.96%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.084010 | 0.003047   | 27.57x  | -96.37%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.081086 | 0.032195   | 2.52x   | -60.30%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.081086 | 0.003353   | 24.18x  | -95.86%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.081086 | 0.003059   | 26.51x  | -96.23%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.085148 | 0.028254   | 3.01x   | -66.82%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.085148 | 0.003424   | 24.86x  | -95.98%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.085148 | 0.002990   | 28.48x  | -96.49%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.110949 | 0.041822   | 2.65x   | -62.30%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.110949 | 0.003877   | 28.61x  | -96.51%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.110949 | 0.003026   | 36.67x  | -97.27%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.085831 | 0.009450   | 9.08x   | -88.99%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.085831 | 0.004056   | 21.16x  | -95.27%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.085831 | 0.003675   | 23.35x  | -95.72%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.121434 | 0.011927   | 10.18x  | -90.18%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.121434 | 0.005140   | 23.63x  | -95.77%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.121434 | 0.004069   | 29.84x  | -96.65%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.095203 | 0.012930   | 7.36x   | -86.42%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.095203 | 0.004553   | 20.91x  | -95.22%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.095203 | 0.003878   | 24.55x  | -95.93%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.101282 | 0.010772   | 9.40x   | -89.36%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.101282 | 0.004716   | 21.47x  | -95.34%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.101282 | 0.003912   | 25.89x  | -96.14%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.098254 | 0.011570   | 8.49x   | -88.22%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.098254 | 0.005132   | 19.15x  | -94.78%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.098254 | 0.004221   | 23.28x  | -95.70%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.094339 | 0.026915   | 3.51x   | -71.47%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.094339 | 0.004557   | 20.70x  | -95.17%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.094339 | 0.004078   | 23.13x  | -95.68%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.094382 | 0.012320   | 7.66x   | -86.95%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.094382 | 0.004458   | 21.17x  | -95.28%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.094382 | 0.003718   | 25.39x  | -96.06%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.094480 | 0.010021   | 9.43x   | -89.39%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.094480 | 0.004464   | 21.16x  | -95.27%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.094480 | 0.003748   | 25.21x  | -96.03%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.129966 | 0.030936   | 4.20x   | -76.20%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.129966 | 0.004805   | 27.05x  | -96.30%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.129966 | 0.004073   | 31.91x  | -96.87%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.124411 | 0.010363   | 12.01x  | -91.67%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.124411 | 0.004833   | 25.74x  | -96.12%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.124411 | 0.003835   | 32.44x  | -96.92%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.094287 | 0.012828   | 7.35x   | -86.39%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.094287 | 0.004623   | 20.39x  | -95.10%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.094287 | 0.004017   | 23.47x  | -95.74%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.096699 | 0.017866   | 5.41x   | -81.52%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.096699 | 0.004456   | 21.70x  | -95.39%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.096699 | 0.003935   | 24.57x  | -95.93%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.096586 | 0.018166   | 5.32x   | -81.19%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.096586 | 0.004757   | 20.30x  | -95.07%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.096586 | 0.004122   | 23.43x  | -95.73%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.102357 | 0.011027   | 9.28x   | -89.23%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.102357 | 0.004757   | 21.52x  | -95.35%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.102357 | 0.004035   | 25.37x  | -96.06%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.123061 | 0.029084   | 4.23x   | -76.37%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.123061 | 0.004631   | 26.57x  | -96.24%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.123061 | 0.003832   | 32.12x  | -96.89%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.066331 | 0.010035   | 6.61x   | -84.87%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.066331 | 0.003390   | 19.56x  | -94.89%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.066331 | 0.003106   | 21.35x  | -95.32%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.101315 | 0.013300   | 7.62x   | -86.87%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.101315 | 0.004195   | 24.15x  | -95.86%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.101315 | 0.003681   | 27.52x  | -96.37%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.073319 | 0.013954   | 5.25x   | -80.97%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.073319 | 0.003936   | 18.63x  | -94.63%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.073319 | 0.003235   | 22.66x  | -95.59%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.076605 | 0.012488   | 6.13x   | -83.70%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.076605 | 0.004072   | 18.81x  | -94.68%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.076605 | 0.003473   | 22.06x  | -95.47%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.077354 | 0.013308   | 5.81x   | -82.80%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.077354 | 0.004338   | 17.83x  | -94.39%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.077354 | 0.003341   | 23.15x  | -95.68%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.072846 | 0.022990   | 3.17x   | -68.44%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.072846 | 0.003898   | 18.69x  | -94.65%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.072846 | 0.003374   | 21.59x  | -95.37%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.072198 | 0.014281   | 5.06x   | -80.22%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.072198 | 0.004096   | 17.63x  | -94.33%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.072198 | 0.003455   | 20.89x  | -95.21%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.076121 | 0.011447   | 6.65x   | -84.96%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.076121 | 0.004203   | 18.11x  | -94.48%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.076121 | 0.003282   | 23.20x  | -95.69%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.111010 | 0.031243   | 3.55x   | -71.86%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.111010 | 0.004383   | 25.33x  | -96.05%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.111010 | 0.003274   | 33.90x  | -97.05%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.105816 | 0.011723   | 9.03x   | -88.92%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.105816 | 0.004029   | 26.27x  | -96.19%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.105816 | 0.003362   | 31.47x  | -96.82%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.074261 | 0.015418   | 4.82x   | -79.24%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.074261 | 0.003980   | 18.66x  | -94.64%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.074261 | 0.003332   | 22.28x  | -95.51%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.076669 | 0.018738   | 4.09x   | -75.56%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.076669 | 0.004228   | 18.13x  | -94.49%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.076669 | 0.003366   | 22.78x  | -95.61%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.077294 | 0.018399   | 4.20x   | -76.20%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.077294 | 0.004195   | 18.42x  | -94.57%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.077294 | 0.003298   | 23.43x  | -95.73%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.078089 | 0.013215   | 5.91x   | -83.08%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.078089 | 0.004263   | 18.32x  | -94.54%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.078089 | 0.003351   | 23.30x  | -95.71%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.105007 | 0.029521   | 3.56x   | -71.89%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.105007 | 0.004637   | 22.65x  | -95.58%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.105007 | 0.003358   | 31.27x  | -96.80%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.392494 | 0.106185   | 3.70x   | -72.95%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.392494 | 0.093228   | 4.21x   | -76.25%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.392494 | 0.097539   | 4.02x   | -75.15%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.511991 | 0.119628   | 4.28x   | -76.63%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.511991 | 0.094335   | 5.43x   | -81.57%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.511991 | 0.096808   | 5.29x   | -81.09%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.380335 | 0.134704   | 2.82x   | -64.58%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.380335 | 0.105014   | 3.62x   | -72.39%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.380335 | 0.093862   | 4.05x   | -75.32%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.407664 | 0.112347   | 3.63x   | -72.44%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.407664 | 0.092265   | 4.42x   | -77.37%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.407664 | 0.093589   | 4.36x   | -77.04%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.408921 | 0.119017   | 3.44x   | -70.89%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.408921 | 0.096586   | 4.23x   | -76.38%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.408921 | 0.098441   | 4.15x   | -75.93%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.400188 | 0.171837   | 2.33x   | -57.06%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.400188 | 0.091933   | 4.35x   | -77.03%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.400188 | 0.093261   | 4.29x   | -76.70%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.379820 | 0.123735   | 3.07x   | -67.42%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.379820 | 0.091442   | 4.15x   | -75.92%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.379820 | 0.093597   | 4.06x   | -75.36%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.407586 | 0.113501   | 3.59x   | -72.15%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.407586 | 0.100387   | 4.06x   | -75.37%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.407586 | 0.097432   | 4.18x   | -76.10%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.560594 | 0.196222   | 2.86x   | -65.00%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.560594 | 0.097441   | 5.75x   | -82.62%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.560594 | 0.098066   | 5.72x   | -82.51%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.506620 | 0.112310   | 4.51x   | -77.83%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.506620 | 0.090299   | 5.61x   | -82.18%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.506620 | 0.094312   | 5.37x   | -81.38%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.398004 | 0.114496   | 3.48x   | -71.23%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.398004 | 0.094435   | 4.21x   | -76.27%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.398004 | 0.095778   | 4.16x   | -75.94%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.411028 | 0.146277   | 2.81x   | -64.41%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.411028 | 0.099111   | 4.15x   | -75.89%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.411028 | 0.098711   | 4.16x   | -75.98%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.405014 | 0.137451   | 2.95x   | -66.06%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.405014 | 0.093122   | 4.35x   | -77.01%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.405014 | 0.097054   | 4.17x   | -76.04%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.431190 | 0.120634   | 3.57x   | -72.02%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.431190 | 0.095911   | 4.50x   | -77.76%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.431190 | 0.095476   | 4.52x   | -77.86%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.509918 | 0.184910   | 2.76x   | -63.74%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.509918 | 0.092719   | 5.50x   | -81.82%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.509918 | 0.095046   | 5.36x   | -81.36%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.287132 | 0.092112   | 3.12x   | -67.92%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.287132 | 0.012014   | 23.90x  | -95.82%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.287132 | 0.011046   | 25.99x  | -96.15%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.417550 | 0.109148   | 3.83x   | -73.86%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.417550 | 0.013729   | 30.41x  | -96.71%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.417550 | 0.012377   | 33.74x  | -97.04%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.285248 | 0.114161   | 2.50x   | -59.98%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.285248 | 0.013148   | 21.69x  | -95.39%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.285248 | 0.012306   | 23.18x  | -95.69%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.317012 | 0.106367   | 2.98x   | -66.45%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.317012 | 0.014072   | 22.53x  | -95.56%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.317012 | 0.012419   | 25.53x  | -96.08%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.317981 | 0.126559   | 2.51x   | -60.20%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.317981 | 0.015309   | 20.77x  | -95.19%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.317981 | 0.014034   | 22.66x  | -95.59%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.296140 | 0.138149   | 2.14x   | -53.35%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.296140 | 0.016947   | 17.47x  | -94.28%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.296140 | 0.012389   | 23.90x  | -95.82%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.282304 | 0.110912   | 2.55x   | -60.71%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.282304 | 0.012842   | 21.98x  | -95.45%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.282304 | 0.012010   | 23.51x  | -95.75%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.291136 | 0.102063   | 2.85x   | -64.94%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.291136 | 0.012328   | 23.62x  | -95.77%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.291136 | 0.011670   | 24.95x  | -95.99%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.458034 | 0.173573   | 2.64x   | -62.10%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.458034 | 0.014828   | 30.89x  | -96.76%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.458034 | 0.011807   | 38.79x  | -97.42%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.401964 | 0.103454   | 3.89x   | -74.26%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.401964 | 0.012819   | 31.36x  | -96.81%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.401964 | 0.011809   | 34.04x  | -97.06%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.289043 | 0.104914   | 2.76x   | -63.70%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.289043 | 0.016499   | 17.52x  | -94.29%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.289043 | 0.012253   | 23.59x  | -95.76%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.301029 | 0.131071   | 2.30x   | -56.46%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.301029 | 0.013474   | 22.34x  | -95.52%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.301029 | 0.012090   | 24.90x  | -95.98%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.306142 | 0.126101   | 2.43x   | -58.81%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.306142 | 0.013089   | 23.39x  | -95.72%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.306142 | 0.011786   | 25.97x  | -96.15%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.316591 | 0.111267   | 2.85x   | -64.85%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.316591 | 0.013525   | 23.41x  | -95.73%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.316591 | 0.011826   | 26.77x  | -96.26%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.421875 | 0.169118   | 2.49x   | -59.91%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.421875 | 0.014875   | 28.36x  | -96.47%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.421875 | 0.011999   | 35.16x  | -97.16%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.338225 | 0.037657   | 8.98x   | -88.87%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.338225 | 0.020827   | 16.24x  | -93.84%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.338225 | 0.019844   | 17.04x  | -94.13%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.455301 | 0.048879   | 9.31x   | -89.26%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.455301 | 0.023882   | 19.06x  | -94.75%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.455301 | 0.021303   | 21.37x  | -95.32%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.328780 | 0.051105   | 6.43x   | -84.46%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.328780 | 0.022415   | 14.67x  | -93.18%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.328780 | 0.020757   | 15.84x  | -93.69%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.357532 | 0.045101   | 7.93x   | -87.39%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.357532 | 0.023780   | 15.04x  | -93.35%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.357532 | 0.021733   | 16.45x  | -93.92%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.353383 | 0.055391   | 6.38x   | -84.33%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.353383 | 0.027769   | 12.73x  | -92.14%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.353383 | 0.023766   | 14.87x  | -93.27%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.364741 | 0.110374   | 3.30x   | -69.74%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.364741 | 0.023955   | 15.23x  | -93.43%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.364741 | 0.021462   | 16.99x  | -94.12%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.344768 | 0.054129   | 6.37x   | -84.30%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.344768 | 0.023615   | 14.60x  | -93.15%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.344768 | 0.021160   | 16.29x  | -93.86%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.351372 | 0.045320   | 7.75x   | -87.10%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.351372 | 0.024900   | 14.11x  | -92.91%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.351372 | 0.023473   | 14.97x  | -93.32%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.519166 | 0.128552   | 4.04x   | -75.24%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.519166 | 0.025759   | 20.15x  | -95.04%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.519166 | 0.022138   | 23.45x  | -95.74%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.458694 | 0.046771   | 9.81x   | -89.80%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.458694 | 0.023739   | 19.32x  | -94.82%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.458694 | 0.021548   | 21.29x  | -95.30%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.347321 | 0.055464   | 6.26x   | -84.03%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.347321 | 0.023815   | 14.58x  | -93.14%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.347321 | 0.021232   | 16.36x  | -93.89%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.371333 | 0.075033   | 4.95x   | -79.79%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.371333 | 0.024128   | 15.39x  | -93.50%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.371333 | 0.021301   | 17.43x  | -94.26%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.369540 | 0.076959   | 4.80x   | -79.17%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.369540 | 0.025494   | 14.50x  | -93.10%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.369540 | 0.024039   | 15.37x  | -93.49%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.387107 | 0.047677   | 8.12x   | -87.68%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.387107 | 0.024474   | 15.82x  | -93.68%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.387107 | 0.023356   | 16.57x  | -93.97%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.468299 | 0.121733   | 3.85x   | -74.01%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.468299 | 0.024131   | 19.41x  | -94.85%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.468299 | 0.021646   | 21.63x  | -95.38%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.271125 | 0.048788   | 5.56x   | -82.01%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.271125 | 0.021360   | 12.69x  | -92.12%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.271125 | 0.019166   | 14.15x  | -92.93%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.384320 | 0.062174   | 6.18x   | -83.82%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.384320 | 0.024999   | 15.37x  | -93.50%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.384320 | 0.021254   | 18.08x  | -94.47%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.261050 | 0.068868   | 3.79x   | -73.62%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.261050 | 0.026036   | 10.03x  | -90.03%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.261050 | 0.020547   | 12.70x  | -92.13%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.285146 | 0.059892   | 4.76x   | -79.00%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.285146 | 0.024202   | 11.78x  | -91.51%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.285146 | 0.020063   | 14.21x  | -92.96%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.283124 | 0.063445   | 4.46x   | -77.59%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.283124 | 0.025819   | 10.97x  | -90.88%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.283124 | 0.021840   | 12.96x  | -92.29%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.276021 | 0.103502   | 2.67x   | -62.50%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.276021 | 0.024767   | 11.14x  | -91.03%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.276021 | 0.021689   | 12.73x  | -92.14%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.293732 | 0.069827   | 4.21x   | -76.23%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.293732 | 0.024427   | 12.02x  | -91.68%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.293732 | 0.020444   | 14.37x  | -93.04%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.272168 | 0.056045   | 4.86x   | -79.41%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.272168 | 0.024566   | 11.08x  | -90.97%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.272168 | 0.020238   | 13.45x  | -92.56%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.441734 | 0.139057   | 3.18x   | -68.52%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.441734 | 0.026513   | 16.66x  | -94.00%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.441734 | 0.021779   | 20.28x  | -95.07%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.383938 | 0.056611   | 6.78x   | -85.26%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.383938 | 0.024076   | 15.95x  | -93.73%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.383938 | 0.020656   | 18.59x  | -94.62%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.278859 | 0.071747   | 3.89x   | -74.27%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.278859 | 0.024620   | 11.33x  | -91.17%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.278859 | 0.021405   | 13.03x  | -92.32%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.273498 | 0.081571   | 3.35x   | -70.17%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.273498 | 0.024644   | 11.10x  | -90.99%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.273498 | 0.020125   | 13.59x  | -92.64%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.279065 | 0.081884   | 3.41x   | -70.66%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.279065 | 0.028251   | 9.88x   | -89.88%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.279065 | 0.023479   | 11.89x  | -91.59%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.309981 | 0.063116   | 4.91x   | -79.64%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.309981 | 0.026971   | 11.49x  | -91.30%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.309981 | 0.020883   | 14.84x  | -93.26%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.410778 | 0.126650   | 3.24x   | -69.17%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.410778 | 0.024726   | 16.61x  | -93.98%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.410778 | 0.020466   | 20.07x  | -95.02%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.085482 | 0.023979   | 3.56x   | -71.95%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.085482 | 0.020048   | 4.26x   | -76.55%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.085482 | 0.020977   | 4.08x   | -75.46%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.112350 | 0.026572   | 4.23x   | -76.35%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.112350 | 0.020497   | 5.48x   | -81.76%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.112350 | 0.021142   | 5.31x   | -81.18%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.090996 | 0.027570   | 3.30x   | -69.70%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.090996 | 0.020366   | 4.47x   | -77.62%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.090996 | 0.020609   | 4.42x   | -77.35%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.091570 | 0.024692   | 3.71x   | -73.04%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.091570 | 0.020881   | 4.39x   | -77.20%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.091570 | 0.021368   | 4.29x   | -76.66%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.091900 | 0.026719   | 3.44x   | -70.93%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.091900 | 0.021957   | 4.19x   | -76.11%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.091900 | 0.021994   | 4.18x   | -76.07%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.086524 | 0.037530   | 2.31x   | -56.62%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.086524 | 0.020478   | 4.23x   | -76.33%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.086524 | 0.020839   | 4.15x   | -75.92%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.085884 | 0.027037   | 3.18x   | -68.52%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.085884 | 0.020046   | 4.28x   | -76.66%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.085884 | 0.020755   | 4.14x   | -75.83%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.090659 | 0.024847   | 3.65x   | -72.59%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.090659 | 0.020261   | 4.47x   | -77.65%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.090659 | 0.021130   | 4.29x   | -76.69%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.117586 | 0.045782   | 2.57x   | -61.06%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.117586 | 0.021399   | 5.50x   | -81.80%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.117586 | 0.022092   | 5.32x   | -81.21%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.116278 | 0.024546   | 4.74x   | -78.89%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.116278 | 0.019970   | 5.82x   | -82.83%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.116278 | 0.020720   | 5.61x   | -82.18%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.084500 | 0.024939   | 3.39x   | -70.49%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.084500 | 0.020513   | 4.12x   | -75.72%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.084500 | 0.020761   | 4.07x   | -75.43%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.090103 | 0.030878   | 2.92x   | -65.73%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.090103 | 0.020574   | 4.38x   | -77.17%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.090103 | 0.021246   | 4.24x   | -76.42%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.093039 | 0.031013   | 3.00x   | -66.67%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.093039 | 0.021821   | 4.26x   | -76.55%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.093039 | 0.021514   | 4.32x   | -76.88%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.092482 | 0.027067   | 3.42x   | -70.73%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.092482 | 0.020914   | 4.42x   | -77.39%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.092482 | 0.021719   | 4.26x   | -76.52%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.117454 | 0.043670   | 2.69x   | -62.82%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.117454 | 0.021014   | 5.59x   | -82.11%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.117454 | 0.021531   | 5.46x   | -81.67%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.067117 | 0.019127   | 3.51x   | -71.50%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.067117 | 0.002576   | 26.06x  | -96.16%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.067117 | 0.002321   | 28.91x  | -96.54%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.099471 | 0.024126   | 4.12x   | -75.75%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.099471 | 0.003172   | 31.36x  | -96.81%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.099471 | 0.002682   | 37.09x  | -97.30%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.076425 | 0.024791   | 3.08x   | -67.56%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.076425 | 0.002905   | 26.31x  | -96.20%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.076425 | 0.002608   | 29.31x  | -96.59%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.078150 | 0.022740   | 3.44x   | -70.90%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.078150 | 0.002884   | 27.10x  | -96.31%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.078150 | 0.002628   | 29.73x  | -96.64%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.077855 | 0.023678   | 3.29x   | -69.59%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.077855 | 0.003195   | 24.37x  | -95.90%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.077855 | 0.002602   | 29.92x  | -96.66%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.073667 | 0.029453   | 2.50x   | -60.02%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.073667 | 0.003651   | 20.18x  | -95.04%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.073667 | 0.002681   | 27.48x  | -96.36%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.075868 | 0.025371   | 2.99x   | -66.56%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.075868 | 0.002783   | 27.26x  | -96.33%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.075868 | 0.002610   | 29.07x  | -96.56%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.076285 | 0.023315   | 3.27x   | -69.44%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.076285 | 0.002732   | 27.92x  | -96.42%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.076285 | 0.002591   | 29.44x  | -96.60%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.108646 | 0.041485   | 2.62x   | -61.82%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.108646 | 0.003296   | 32.97x  | -96.97%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.108646 | 0.002617   | 41.51x  | -97.59%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.102213 | 0.023812   | 4.29x   | -76.70%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.102213 | 0.002831   | 36.10x  | -97.23%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.102213 | 0.002616   | 39.08x  | -97.44%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.073834 | 0.025259   | 2.92x   | -65.79%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.073834 | 0.003694   | 19.99x  | -95.00%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.073834 | 0.002772   | 26.64x  | -96.25%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.079498 | 0.028308   | 2.81x   | -64.39%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.079498 | 0.002921   | 27.21x  | -96.33%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.079498 | 0.002617   | 30.38x  | -96.71%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.079791 | 0.027538   | 2.90x   | -65.49%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.079791 | 0.002983   | 26.75x  | -96.26%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.079791 | 0.002598   | 30.72x  | -96.74%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.079445 | 0.024069   | 3.30x   | -69.70%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.079445 | 0.003099   | 25.64x  | -96.10%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.079445 | 0.002606   | 30.48x  | -96.72%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.100162 | 0.036304   | 2.76x   | -63.75%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.100162 | 0.003086   | 32.45x  | -96.92%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.100162 | 0.002659   | 37.67x  | -97.35%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.070865 | 0.007092   | 9.99x   | -89.99%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.070865 | 0.003418   | 20.73x  | -95.18%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.070865 | 0.003213   | 22.06x  | -95.47%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.103573 | 0.009333   | 11.10x  | -90.99%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.103573 | 0.004075   | 25.42x  | -96.07%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.103573 | 0.003396   | 30.50x  | -96.72%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.078646 | 0.010217   | 7.70x   | -87.01%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.078646 | 0.003815   | 20.62x  | -95.15%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.078646 | 0.003117   | 25.23x  | -96.04%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.082875 | 0.008570   | 9.67x   | -89.66%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.082875 | 0.003885   | 21.33x  | -95.31%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.082875 | 0.003303   | 25.09x  | -96.01%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.084112 | 0.010268   | 8.19x   | -87.79%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.084112 | 0.004419   | 19.04x  | -94.75%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.084112 | 0.003733   | 22.53x  | -95.56%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.079013 | 0.022947   | 3.44x   | -70.96%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.079013 | 0.003845   | 20.55x  | -95.13%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.079013 | 0.003404   | 23.21x  | -95.69%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.081457 | 0.010752   | 7.58x   | -86.80%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.081457 | 0.003823   | 21.31x  | -95.31%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.081457 | 0.003145   | 25.90x  | -96.14%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.081448 | 0.008961   | 9.09x   | -89.00%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.081448 | 0.004182   | 19.48x  | -94.87%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.081448 | 0.003490   | 23.34x  | -95.72%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.111101 | 0.026922   | 4.13x   | -75.77%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.111101 | 0.004502   | 24.68x  | -95.95%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.111101 | 0.003993   | 27.82x  | -96.41%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.106988 | 0.008821   | 12.13x  | -91.76%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.106988 | 0.003954   | 27.06x  | -96.30%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.106988 | 0.003323   | 32.20x  | -96.89%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.079061 | 0.010526   | 7.51x   | -86.69%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.079061 | 0.003856   | 20.50x  | -95.12%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.079061 | 0.003282   | 24.09x  | -95.85%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.083079 | 0.015058   | 5.52x   | -81.87%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.083079 | 0.004047   | 20.53x  | -95.13%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.083079 | 0.003476   | 23.90x  | -95.82%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.087292 | 0.015013   | 5.81x   | -82.80%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.087292 | 0.003981   | 21.93x  | -95.44%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.087292 | 0.003425   | 25.49x  | -96.08%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.085530 | 0.009323   | 9.17x   | -89.10%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.085530 | 0.004178   | 20.47x  | -95.12%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.085530 | 0.003546   | 24.12x  | -95.85%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.107796 | 0.025181   | 4.28x   | -76.64%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.107796 | 0.004139   | 26.04x  | -96.16%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.107796 | 0.003551   | 30.35x  | -96.71%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.056797 | 0.009072   | 6.26x   | -84.03%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.056797 | 0.002958   | 19.20x  | -94.79%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.056797 | 0.002573   | 22.07x  | -95.47%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.090928 | 0.011701   | 7.77x   | -87.13%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.090928 | 0.003925   | 23.17x  | -95.68%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.090928 | 0.002919   | 31.15x  | -96.79%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.067424 | 0.012659   | 5.33x   | -81.22%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.067424 | 0.003518   | 19.16x  | -94.78%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.067424 | 0.002825   | 23.86x  | -95.81%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.069454 | 0.010754   | 6.46x   | -84.52%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.069454 | 0.003588   | 19.36x  | -94.83%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.069454 | 0.002795   | 24.85x  | -95.98%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.070206 | 0.011681   | 6.01x   | -83.36%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.070206 | 0.004053   | 17.32x  | -94.23%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.070206 | 0.002978   | 23.57x  | -95.76%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.065078 | 0.020312   | 3.20x   | -68.79%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.065078 | 0.003386   | 19.22x  | -94.80%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.065078 | 0.002968   | 21.93x  | -95.44%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.067876 | 0.012894   | 5.26x   | -81.00%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.067876 | 0.003860   | 17.59x  | -94.31%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.067876 | 0.002969   | 22.86x  | -95.63%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.068353 | 0.010299   | 6.64x   | -84.93%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.068353 | 0.003738   | 18.29x  | -94.53%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.068353 | 0.002926   | 23.36x  | -95.72%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.107177 | 0.028288   | 3.79x   | -73.61%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.107177 | 0.004057   | 26.42x  | -96.21%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.107177 | 0.003053   | 35.11x  | -97.15%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.096591 | 0.010623   | 9.09x   | -89.00%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.096591 | 0.003609   | 26.76x  | -96.26%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.096591 | 0.002897   | 33.34x  | -97.00%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.066458 | 0.013901   | 4.78x   | -79.08%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.066458 | 0.003612   | 18.40x  | -94.56%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.066458 | 0.002992   | 22.21x  | -95.50%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.069536 | 0.015949   | 4.36x   | -77.06%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.069536 | 0.003656   | 19.02x  | -94.74%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.069536 | 0.002818   | 24.68x  | -95.95%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.069149 | 0.016547   | 4.18x   | -76.07%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.069149 | 0.003849   | 17.97x  | -94.43%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.069149 | 0.003031   | 22.82x  | -95.62%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.071883 | 0.011266   | 6.38x   | -84.33%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.071883 | 0.003752   | 19.16x  | -94.78%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.071883 | 0.002973   | 24.18x  | -95.86%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.095028 | 0.025649   | 3.70x   | -73.01%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.095028 | 0.003928   | 24.19x  | -95.87%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.095028 | 0.002875   | 33.06x  | -96.98%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.193244 | 0.052819   | 3.66x   | -72.67%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.193244 | 0.046503   | 4.16x   | -75.94%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.193244 | 0.048850   | 3.96x   | -74.72%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.261152 | 0.061685   | 4.23x   | -76.38%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.261152 | 0.048521   | 5.38x   | -81.42%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.261152 | 0.049561   | 5.27x   | -81.02%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.197545 | 0.061471   | 3.21x   | -68.88%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.197545 | 0.047070   | 4.20x   | -76.17%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.197545 | 0.047445   | 4.16x   | -75.98%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.200826 | 0.056386   | 3.56x   | -71.92%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.200826 | 0.046210   | 4.35x   | -76.99%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.200826 | 0.048579   | 4.13x   | -75.81%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.203378 | 0.062220   | 3.27x   | -69.41%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.203378 | 0.049292   | 4.13x   | -75.76%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.203378 | 0.049371   | 4.12x   | -75.72%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.196786 | 0.085027   | 2.31x   | -56.79%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.196786 | 0.047047   | 4.18x   | -76.09%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.196786 | 0.047813   | 4.12x   | -75.70%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.192329 | 0.065655   | 2.93x   | -65.86%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.192329 | 0.046958   | 4.10x   | -75.58%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.192329 | 0.048663   | 3.95x   | -74.70%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.199145 | 0.057640   | 3.45x   | -71.06%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.199145 | 0.047675   | 4.18x   | -76.06%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.199145 | 0.048402   | 4.11x   | -75.70%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.267543 | 0.098246   | 2.72x   | -63.28%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.267543 | 0.047766   | 5.60x   | -82.15%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.267543 | 0.050035   | 5.35x   | -81.30%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.254287 | 0.056710   | 4.48x   | -77.70%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.254287 | 0.046695   | 5.45x   | -81.64%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.254287 | 0.047891   | 5.31x   | -81.17%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.195320 | 0.059217   | 3.30x   | -69.68%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.195320 | 0.047779   | 4.09x   | -75.54%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.195320 | 0.048993   | 3.99x   | -74.92%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.197467 | 0.070447   | 2.80x   | -64.32%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.197467 | 0.047765   | 4.13x   | -75.81%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.197467 | 0.047428   | 4.16x   | -75.98%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.198126 | 0.070852   | 2.80x   | -64.24%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.198126 | 0.047071   | 4.21x   | -76.24%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.198126 | 0.049231   | 4.02x   | -75.15%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.202772 | 0.060902   | 3.33x   | -69.97%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.202772 | 0.048491   | 4.18x   | -76.09%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.202772 | 0.048947   | 4.14x   | -75.86%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.249470 | 0.093511   | 2.67x   | -62.52%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.249470 | 0.046639   | 5.35x   | -81.30%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.249470 | 0.047921   | 5.21x   | -80.79%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.136421 | 0.042616   | 3.20x   | -68.76%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.136421 | 0.005768   | 23.65x  | -95.77%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.136421 | 0.005222   | 26.12x  | -96.17%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.196020 | 0.058699   | 3.34x   | -70.05%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.196020 | 0.007061   | 27.76x  | -96.40%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.196020 | 0.006025   | 32.53x  | -96.93%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.147452 | 0.054164   | 2.72x   | -63.27%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.147452 | 0.008030   | 18.36x  | -94.55%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.147452 | 0.005791   | 25.46x  | -96.07%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.149675 | 0.051696   | 2.90x   | -65.46%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.149675 | 0.006281   | 23.83x  | -95.80%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.149675 | 0.005790   | 25.85x  | -96.13%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.147778 | 0.054037   | 2.73x   | -63.43%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.147778 | 0.007120   | 20.76x  | -95.18%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.147778 | 0.005820   | 25.39x  | -96.06%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.143987 | 0.066832   | 2.15x   | -53.58%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.143987 | 0.008302   | 17.34x  | -94.23%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.143987 | 0.006548   | 21.99x  | -95.45%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.144367 | 0.056530   | 2.55x   | -60.84%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.144367 | 0.006120   | 23.59x  | -95.76%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.144367 | 0.005763   | 25.05x  | -96.01%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.144616 | 0.051250   | 2.82x   | -64.56%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.144616 | 0.006115   | 23.65x  | -95.77%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.144616 | 0.005895   | 24.53x  | -95.92%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.217788 | 0.087211   | 2.50x   | -59.96%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.217788 | 0.007423   | 29.34x  | -96.59%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.217788 | 0.005910   | 36.85x  | -97.29%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.205765 | 0.051679   | 3.98x   | -74.88%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.205765 | 0.006379   | 32.26x  | -96.90%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.205765 | 0.005843   | 35.22x  | -97.16%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.145901 | 0.054823   | 2.66x   | -62.42%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.145901 | 0.008232   | 17.72x  | -94.36%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.145901 | 0.006176   | 23.62x  | -95.77%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.149514 | 0.061454   | 2.43x   | -58.90%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.149514 | 0.006464   | 23.13x  | -95.68%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.149514 | 0.005836   | 25.62x  | -96.10%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.149713 | 0.062251   | 2.40x   | -58.42%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.149713 | 0.006735   | 22.23x  | -95.50%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.149713 | 0.005966   | 25.09x  | -96.01%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.153368 | 0.058676   | 2.61x   | -61.74%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.153368 | 0.007060   | 21.72x  | -95.40%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.153368 | 0.005858   | 26.18x  | -96.18%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.210976 | 0.085972   | 2.45x   | -59.25%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.210976 | 0.007283   | 28.97x  | -96.55%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.210976 | 0.006058   | 34.82x  | -97.13%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.161001 | 0.018819   | 8.56x   | -88.31%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.161001 | 0.008340   | 19.30x  | -94.82%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.161001 | 0.007693   | 20.93x  | -95.22%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.224235 | 0.024635   | 9.10x   | -89.01%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.224235 | 0.010350   | 21.66x  | -95.38%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.224235 | 0.008166   | 27.46x  | -96.36%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.177640 | 0.026570   | 6.69x   | -85.04%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.177640 | 0.008917   | 19.92x  | -94.98%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.177640 | 0.007237   | 24.55x  | -95.93%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.180299 | 0.023416   | 7.70x   | -87.01%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.180299 | 0.010435   | 17.28x  | -94.21%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.180299 | 0.007966   | 22.63x  | -95.58%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.178444 | 0.025171   | 7.09x   | -85.89%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.178444 | 0.010089   | 17.69x  | -94.35%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.178444 | 0.008552   | 20.87x  | -95.21%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.179658 | 0.055681   | 3.23x   | -69.01%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.179658 | 0.009273   | 19.37x  | -94.84%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.179658 | 0.007791   | 23.06x  | -95.66%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.172910 | 0.027070   | 6.39x   | -84.34%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.172910 | 0.009491   | 18.22x  | -94.51%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.172910 | 0.007508   | 23.03x  | -95.66%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.177390 | 0.021960   | 8.08x   | -87.62%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.177390 | 0.009142   | 19.40x  | -94.85%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.177390 | 0.007431   | 23.87x  | -95.81%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.245538 | 0.065038   | 3.78x   | -73.51%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.245538 | 0.009801   | 25.05x  | -96.01%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.245538 | 0.008249   | 29.77x  | -96.64%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.243597 | 0.024756   | 9.84x   | -89.84%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.243597 | 0.009192   | 26.50x  | -96.23%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.243597 | 0.007691   | 31.67x  | -96.84%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.177621 | 0.027594   | 6.44x   | -84.46%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.177621 | 0.009690   | 18.33x  | -94.54%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.177621 | 0.008015   | 22.16x  | -95.49%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.178979 | 0.036976   | 4.84x   | -79.34%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.178979 | 0.009033   | 19.81x  | -94.95%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.178979 | 0.007989   | 22.40x  | -95.54%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.184808 | 0.037607   | 4.91x   | -79.65%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.184808 | 0.009201   | 20.08x  | -95.02%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.184808 | 0.007836   | 23.59x  | -95.76%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.175776 | 0.025904   | 6.79x   | -85.26%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.175776 | 0.010883   | 16.15x  | -93.81%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.175776 | 0.008290   | 21.20x  | -95.28%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.227339 | 0.059664   | 3.81x   | -73.76%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.227339 | 0.009390   | 24.21x  | -95.87%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.227339 | 0.007889   | 28.82x  | -96.53%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.127307 | 0.020091   | 6.34x   | -84.22%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.127307 | 0.006666   | 19.10x  | -94.76%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.127307 | 0.005506   | 23.12x  | -95.68%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.189265 | 0.026435   | 7.16x   | -86.03%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.189265 | 0.008308   | 22.78x  | -95.61%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.189265 | 0.006335   | 29.88x  | -96.65%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.136444 | 0.029917   | 4.56x   | -78.07%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.136444 | 0.008305   | 16.43x  | -93.91%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.136444 | 0.006593   | 20.70x  | -95.17%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.149180 | 0.025332   | 5.89x   | -83.02%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.149180 | 0.008197   | 18.20x  | -94.51%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.149180 | 0.006392   | 23.34x  | -95.72%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.139418 | 0.026949   | 5.17x   | -80.67%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.139418 | 0.008407   | 16.58x  | -93.97%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.139418 | 0.006694   | 20.83x  | -95.20%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.136343 | 0.046014   | 2.96x   | -66.25%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.136343 | 0.007810   | 17.46x  | -94.27%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.136343 | 0.006751   | 20.20x  | -95.05%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.136062 | 0.028240   | 4.82x   | -79.24%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.136062 | 0.008174   | 16.65x  | -93.99%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.136062 | 0.006424   | 21.18x  | -95.28%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.137234 | 0.023166   | 5.92x   | -83.12%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.137234 | 0.008070   | 17.01x  | -94.12%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.137234 | 0.006379   | 21.51x  | -95.35%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.210454 | 0.062447   | 3.37x   | -70.33%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.210454 | 0.008307   | 25.33x  | -96.05%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.210454 | 0.006220   | 33.84x  | -97.04%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.194756 | 0.023889   | 8.15x   | -87.73%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.194756 | 0.007831   | 24.87x  | -95.98%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.194756 | 0.006263   | 31.10x  | -96.78%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.133645 | 0.030594   | 4.37x   | -77.11%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.133645 | 0.007361   | 18.16x  | -94.49%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.133645 | 0.006370   | 20.98x  | -95.23%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.136674 | 0.035518   | 3.85x   | -74.01%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.136674 | 0.007766   | 17.60x  | -94.32%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.136674 | 0.005968   | 22.90x  | -95.63%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.137052 | 0.036512   | 3.75x   | -73.36%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.137052 | 0.008522   | 16.08x  | -93.78%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.137052 | 0.006466   | 21.20x  | -95.28%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.142961 | 0.024718   | 5.78x   | -82.71%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.142961 | 0.008158   | 17.52x  | -94.29%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.142961 | 0.006499   | 22.00x  | -95.45%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.192870 | 0.057753   | 3.34x   | -70.06%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.192870 | 0.008096   | 23.82x  | -95.80%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.192870 | 0.006134   | 31.44x  | -96.82%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.337001 | 0.105166   | 3.20x   | -68.79%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.337001 | 0.082659   | 4.08x   | -75.47%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.337001 | 0.086164   | 3.91x   | -74.43%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.464063 | 0.109079   | 4.25x   | -76.49%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.464063 | 0.083730   | 5.54x   | -81.96%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.464063 | 0.085683   | 5.42x   | -81.54%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.331954 | 0.120612   | 2.75x   | -63.67%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.331954 | 0.085321   | 3.89x   | -74.30%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.331954 | 0.087584   | 3.79x   | -73.62%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.364003 | 0.102980   | 3.53x   | -71.71%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.364003 | 0.082455   | 4.41x   | -77.35%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.364003 | 0.084971   | 4.28x   | -76.66%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.383165 | 0.105896   | 3.62x   | -72.36%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.383165 | 0.084297   | 4.55x   | -78.00%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.383165 | 0.084892   | 4.51x   | -77.84%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.340944 | 0.156722   | 2.18x   | -54.03%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.340944 | 0.087327   | 3.90x   | -74.39%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.340944 | 0.085740   | 3.98x   | -74.85%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.334476 | 0.110282   | 3.03x   | -67.03%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.334476 | 0.081484   | 4.10x   | -75.64%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.334476 | 0.084184   | 3.97x   | -74.83%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.348643 | 0.100900   | 3.46x   | -71.06%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.348643 | 0.081740   | 4.27x   | -76.55%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.348643 | 0.084569   | 4.12x   | -75.74%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.494474 | 0.173126   | 2.86x   | -64.99%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.494474 | 0.085086   | 5.81x   | -82.79%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.494474 | 0.085875   | 5.76x   | -82.63%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.445917 | 0.100089   | 4.46x   | -77.55%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.445917 | 0.080876   | 5.51x   | -81.86%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.445917 | 0.083691   | 5.33x   | -81.23%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.349536 | 0.107663   | 3.25x   | -69.20%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.349536 | 0.084813   | 4.12x   | -75.74%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.349536 | 0.085193   | 4.10x   | -75.63%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.355661 | 0.123499   | 2.88x   | -65.28%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.355661 | 0.082183   | 4.33x   | -76.89%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.355661 | 0.084427   | 4.21x   | -76.26%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.354020 | 0.123024   | 2.88x   | -65.25%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.354020 | 0.089039   | 3.98x   | -74.85%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.354020 | 0.086376   | 4.10x   | -75.60%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.377137 | 0.112141   | 3.36x   | -70.27%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.377137 | 0.090427   | 4.17x   | -76.02%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.377137 | 0.086019   | 4.38x   | -77.19%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.473356 | 0.167970   | 2.82x   | -64.51%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.473356 | 0.081888   | 5.78x   | -82.70%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.473356 | 0.087876   | 5.39x   | -81.44%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.260875 | 0.074797   | 3.49x   | -71.33%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.260875 | 0.010163   | 25.67x  | -96.10%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.260875 | 0.009154   | 28.50x  | -96.49%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.355373 | 0.095788   | 3.71x   | -73.05%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.355373 | 0.011896   | 29.87x  | -96.65%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.355373 | 0.010309   | 34.47x  | -97.10%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.274731 | 0.097059   | 2.83x   | -64.67%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.274731 | 0.010938   | 25.12x  | -96.02%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.274731 | 0.010277   | 26.73x  | -96.26%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.258493 | 0.091702   | 2.82x   | -64.52%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.258493 | 0.011329   | 22.82x  | -95.62%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.258493 | 0.010495   | 24.63x  | -95.94%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.272567 | 0.101097   | 2.70x   | -62.91%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.272567 | 0.013017   | 20.94x  | -95.22%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.272567 | 0.010621   | 25.66x  | -96.10%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.261529 | 0.117032   | 2.23x   | -55.25%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.261529 | 0.014466   | 18.08x  | -94.47%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.261529 | 0.010702   | 24.44x  | -95.91%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.247708 | 0.098777   | 2.51x   | -60.12%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.247708 | 0.011097   | 22.32x  | -95.52%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.247708 | 0.010443   | 23.72x  | -95.78%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.263115 | 0.090916   | 2.89x   | -65.45%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.263115 | 0.010911   | 24.12x  | -95.85%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.263115 | 0.010452   | 25.17x  | -96.03%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.407523 | 0.153966   | 2.65x   | -62.22%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.407523 | 0.013378   | 30.46x  | -96.72%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.407523 | 0.010575   | 38.54x  | -97.40%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.355473 | 0.092824   | 3.83x   | -73.89%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.355473 | 0.011210   | 31.71x  | -96.85%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.355473 | 0.010357   | 34.32x  | -97.09%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.252506 | 0.091949   | 2.75x   | -63.59%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.252506 | 0.014492   | 17.42x  | -94.26%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.252506 | 0.010799   | 23.38x  | -95.72%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.264138 | 0.111384   | 2.37x   | -57.83%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.264138 | 0.012237   | 21.59x  | -95.37%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.264138 | 0.010712   | 24.66x  | -95.94%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.277558 | 0.113384   | 2.45x   | -59.15%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.277558 | 0.012049   | 23.04x  | -95.66%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.277558 | 0.010726   | 25.88x  | -96.14%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.269042 | 0.096708   | 2.78x   | -64.05%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.269042 | 0.012054   | 22.32x  | -95.52%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.269042 | 0.010557   | 25.49x  | -96.08%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.364036 | 0.146354   | 2.49x   | -59.80%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.364036 | 0.012330   | 29.53x  | -96.61%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.364036 | 0.010410   | 34.97x  | -97.14%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.309541 | 0.028221   | 10.97x  | -90.88%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.309541 | 0.013919   | 22.24x  | -95.50%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.309541 | 0.013649   | 22.68x  | -95.59%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.395785 | 0.038468   | 10.29x  | -90.28%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.395785 | 0.016321   | 24.25x  | -95.88%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.395785 | 0.014191   | 27.89x  | -96.41%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.294656 | 0.040615   | 7.25x   | -86.22%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.294656 | 0.015320   | 19.23x  | -94.80%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.294656 | 0.013240   | 22.25x  | -95.51%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.325572 | 0.035910   | 9.07x   | -88.97%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.325572 | 0.015822   | 20.58x  | -95.14%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.325572 | 0.013275   | 24.53x  | -95.92%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.323877 | 0.038569   | 8.40x   | -88.09%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.323877 | 0.017059   | 18.99x  | -94.73%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.323877 | 0.015248   | 21.24x  | -95.29%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.314070 | 0.094410   | 3.33x   | -69.94%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.314070 | 0.016258   | 19.32x  | -94.82%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.314070 | 0.015015   | 20.92x  | -95.22%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.364100 | 0.044547   | 8.17x   | -87.77%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.364100 | 0.016053   | 22.68x  | -95.59%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.364100 | 0.015789   | 23.06x  | -95.66%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.325345 | 0.035644   | 9.13x   | -89.04%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.325345 | 0.016323   | 19.93x  | -94.98%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.325345 | 0.014024   | 23.20x  | -95.69%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.487778 | 0.107793   | 4.53x   | -77.90%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.487778 | 0.017066   | 28.58x  | -96.50%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.487778 | 0.014472   | 33.71x  | -97.03%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.410734 | 0.035204   | 11.67x  | -91.43%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.410734 | 0.015631   | 26.28x  | -96.19%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.410734 | 0.014995   | 27.39x  | -96.35%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.305492 | 0.043659   | 7.00x   | -85.71%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.305492 | 0.017270   | 17.69x  | -94.35%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.305492 | 0.015921   | 19.19x  | -94.79%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.322315 | 0.059338   | 5.43x   | -81.59%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.322315 | 0.016250   | 19.83x  | -94.96%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.322315 | 0.014464   | 22.28x  | -95.51%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.320308 | 0.059789   | 5.36x   | -81.33%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.320308 | 0.016222   | 19.74x  | -94.94%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.320308 | 0.015156   | 21.13x  | -95.27%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.337892 | 0.036903   | 9.16x   | -89.08%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.337892 | 0.017229   | 19.61x  | -94.90%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.337892 | 0.014872   | 22.72x  | -95.60%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.422758 | 0.100568   | 4.20x   | -76.21%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.422758 | 0.017252   | 24.50x  | -95.92%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.422758 | 0.014865   | 28.44x  | -96.48%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.254838 | 0.041846   | 6.09x   | -83.58%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.254838 | 0.020035   | 12.72x  | -92.14%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.254838 | 0.018251   | 13.96x  | -92.84%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.349653 | 0.052431   | 6.67x   | -85.00%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.349653 | 0.020937   | 16.70x  | -94.01%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.349653 | 0.018605   | 18.79x  | -94.68%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.238487 | 0.057278   | 4.16x   | -75.98%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.238487 | 0.020222   | 11.79x  | -91.52%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.238487 | 0.017129   | 13.92x  | -92.82%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.254316 | 0.050243   | 5.06x   | -80.24%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.254316 | 0.020427   | 12.45x  | -91.97%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.254316 | 0.017598   | 14.45x  | -93.08%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.263365 | 0.054455   | 4.84x   | -79.32%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.263365 | 0.021253   | 12.39x  | -91.93%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.263365 | 0.018333   | 14.37x  | -93.04%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.242340 | 0.085534   | 2.83x   | -64.71%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.242340 | 0.019084   | 12.70x  | -92.13%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.242340 | 0.017163   | 14.12x  | -92.92%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.236710 | 0.059111   | 4.00x   | -75.03%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.236710 | 0.020355   | 11.63x  | -91.40%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.236710 | 0.017023   | 13.91x  | -92.81%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.241646 | 0.046417   | 5.21x   | -80.79%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.241646 | 0.019666   | 12.29x  | -91.86%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.241646 | 0.016530   | 14.62x  | -93.16%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.390494 | 0.116889   | 3.34x   | -70.07%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.390494 | 0.020988   | 18.61x  | -94.63%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.390494 | 0.017515   | 22.29x  | -95.51%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.345246 | 0.048589   | 7.11x   | -85.93%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.345246 | 0.020531   | 16.82x  | -94.05%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.345246 | 0.018278   | 18.89x  | -94.71%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.245937 | 0.061308   | 4.01x   | -75.07%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.245937 | 0.020537   | 11.98x  | -91.65%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.245937 | 0.017899   | 13.74x  | -92.72%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.254082 | 0.069303   | 3.67x   | -72.72%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.254082 | 0.019956   | 12.73x  | -92.15%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.254082 | 0.016525   | 15.38x  | -93.50%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.246830 | 0.069263   | 3.56x   | -71.94%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.246830 | 0.020474   | 12.06x  | -91.71%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.246830 | 0.017771   | 13.89x  | -92.80%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.272030 | 0.050738   | 5.36x   | -81.35%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.272030 | 0.021011   | 12.95x  | -92.28%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.272030 | 0.017599   | 15.46x  | -93.53%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.368929 | 0.108822   | 3.39x   | -70.50%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.368929 | 0.020571   | 17.93x  | -94.42%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.368929 | 0.017001   | 21.70x  | -95.39%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.756075 | 0.212738   | 3.55x   | -71.86%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.756075 | 0.177409   | 4.26x   | -76.54%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.756075 | 0.189216   | 4.00x   | -74.97%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 0.954727 | 0.238133   | 4.01x   | -75.06%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 0.954727 | 0.182698   | 5.23x   | -80.86%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 0.954727 | 0.188179   | 5.07x   | -80.29%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.747342 | 0.248858   | 3.00x   | -66.70%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.747342 | 0.184830   | 4.04x   | -75.27%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.747342 | 0.188561   | 3.96x   | -74.77%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.780926 | 0.226451   | 3.45x   | -71.00%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.780926 | 0.179545   | 4.35x   | -77.01%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.780926 | 0.187800   | 4.16x   | -75.95%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.782958 | 0.252809   | 3.10x   | -67.71%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.782958 | 0.197957   | 3.96x   | -74.72%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.782958 | 0.199697   | 3.92x   | -74.49%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.755788 | 0.342596   | 2.21x   | -54.67%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.755788 | 0.183485   | 4.12x   | -75.72%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.755788 | 0.187363   | 4.03x   | -75.21%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.740631 | 0.249003   | 2.97x   | -66.38%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.740631 | 0.179855   | 4.12x   | -75.72%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.740631 | 0.186719   | 3.97x   | -74.79%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.755049 | 0.228909   | 3.30x   | -69.68%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.755049 | 0.185087   | 4.08x   | -75.49%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.755049 | 0.199457   | 3.79x   | -73.58%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.055864 | 0.393605   | 2.68x   | -62.72%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.055864 | 0.190548   | 5.54x   | -81.95%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.055864 | 0.195467   | 5.40x   | -81.49%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.973262 | 0.229707   | 4.24x   | -76.40%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.973262 | 0.183412   | 5.31x   | -81.15%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.973262 | 0.198230   | 4.91x   | -79.63%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.797669 | 0.236041   | 3.38x   | -70.41%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.797669 | 0.185396   | 4.30x   | -76.76%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.797669 | 0.187092   | 4.26x   | -76.55%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.765939 | 0.277868   | 2.76x   | -63.72%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.765939 | 0.182548   | 4.20x   | -76.17%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.765939 | 0.187744   | 4.08x   | -75.49%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.762204 | 0.277724   | 2.74x   | -63.56%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.762204 | 0.183732   | 4.15x   | -75.89%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.762204 | 0.193354   | 3.94x   | -74.63%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.809180 | 0.238338   | 3.40x   | -70.55%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.809180 | 0.192566   | 4.20x   | -76.20%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.809180 | 0.190514   | 4.25x   | -76.46%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 0.986854 | 0.379022   | 2.60x   | -61.59%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 0.986854 | 0.188039   | 5.25x   | -80.95%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 0.986854 | 0.196724   | 5.02x   | -80.07%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.572116 | 0.171865   | 3.33x   | -69.96%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.572116 | 0.022765   | 25.13x  | -96.02%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.572116 | 0.020571   | 27.81x  | -96.40%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.760738 | 0.214518   | 3.55x   | -71.80%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.760738 | 0.026940   | 28.24x  | -96.46%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.760738 | 0.023318   | 32.62x  | -96.93%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.531621 | 0.220887   | 2.41x   | -58.45%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.531621 | 0.024850   | 21.39x  | -95.33%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.531621 | 0.023112   | 23.00x  | -95.65%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.570344 | 0.207819   | 2.74x   | -63.56%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.570344 | 0.025261   | 22.58x  | -95.57%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.570344 | 0.023301   | 24.48x  | -95.91%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.572730 | 0.216544   | 2.64x   | -62.19%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.572730 | 0.028877   | 19.83x  | -94.96%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.572730 | 0.023371   | 24.51x  | -95.92%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.562867 | 0.267565   | 2.10x   | -52.46%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.562867 | 0.032637   | 17.25x  | -94.20%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.562867 | 0.024117   | 23.34x  | -95.72%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.544961 | 0.221744   | 2.46x   | -59.31%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.544961 | 0.024459   | 22.28x  | -95.51%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.544961 | 0.024693   | 22.07x  | -95.47%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.559357 | 0.209035   | 2.68x   | -62.63%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.559357 | 0.024524   | 22.81x  | -95.62%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.559357 | 0.023177   | 24.13x  | -95.86%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 0.871403 | 0.349828   | 2.49x   | -59.85%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 0.871403 | 0.029581   | 29.46x  | -96.61%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 0.871403 | 0.023591   | 36.94x  | -97.29%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.777388 | 0.205490   | 3.78x   | -73.57%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.777388 | 0.024955   | 31.15x  | -96.79%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.777388 | 0.023166   | 33.56x  | -97.02%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.558021 | 0.209433   | 2.66x   | -62.47%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.558021 | 0.032700   | 17.06x  | -94.14%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.558021 | 0.024232   | 23.03x  | -95.66%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.577648 | 0.252988   | 2.28x   | -56.20%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.577648 | 0.026069   | 22.16x  | -95.49%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.577648 | 0.023461   | 24.62x  | -95.94%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.572338 | 0.256154   | 2.23x   | -55.24%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.572338 | 0.028390   | 20.16x  | -95.04%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.572338 | 0.024956   | 22.93x  | -95.64%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.645473 | 0.233782   | 2.76x   | -63.78%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.645473 | 0.027077   | 23.84x  | -95.81%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.645473 | 0.026077   | 24.75x  | -95.96%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.846680 | 0.347187   | 2.44x   | -58.99%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.846680 | 0.028882   | 29.32x  | -96.59%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.846680 | 0.023888   | 35.44x  | -97.18%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.664842 | 0.074439   | 8.93x   | -88.80%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.664842 | 0.041475   | 16.03x  | -93.76%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.664842 | 0.038331   | 17.34x  | -94.23%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.867169 | 0.097369   | 8.91x   | -88.77%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.867169 | 0.047749   | 18.16x  | -94.49%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.867169 | 0.042768   | 20.28x  | -95.07%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.650742 | 0.102381   | 6.36x   | -84.27%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.650742 | 0.044092   | 14.76x  | -93.22%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.650742 | 0.038236   | 17.02x  | -94.12%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.669069 | 0.090082   | 7.43x   | -86.54%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.669069 | 0.045683   | 14.65x  | -93.17%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.669069 | 0.040611   | 16.48x  | -93.93%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.681299 | 0.098507   | 6.92x   | -85.54%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.681299 | 0.048551   | 14.03x  | -92.87%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.681299 | 0.042691   | 15.96x  | -93.73%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.671699 | 0.220641   | 3.04x   | -67.15%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.671699 | 0.046455   | 14.46x  | -93.08%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.671699 | 0.041886   | 16.04x  | -93.76%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.655433 | 0.103064   | 6.36x   | -84.28%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.655433 | 0.045908   | 14.28x  | -93.00%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.655433 | 0.041651   | 15.74x  | -93.65%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.646486 | 0.085053   | 7.60x   | -86.84%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.646486 | 0.045185   | 14.31x  | -93.01%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.646486 | 0.039708   | 16.28x  | -93.86%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 0.968219 | 0.248467   | 3.90x   | -74.34%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 0.968219 | 0.047539   | 20.37x  | -95.09%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 0.968219 | 0.041681   | 23.23x  | -95.70%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.859425 | 0.085171   | 10.09x  | -90.09%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.859425 | 0.044626   | 19.26x  | -94.81%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.859425 | 0.039539   | 21.74x  | -95.40%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.690103 | 0.104199   | 6.62x   | -84.90%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.690103 | 0.044109   | 15.65x  | -93.61%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.690103 | 0.039944   | 17.28x  | -94.21%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.676995 | 0.146133   | 4.63x   | -78.41%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.676995 | 0.046063   | 14.70x  | -93.20%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.676995 | 0.041609   | 16.27x  | -93.85%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.666029 | 0.142266   | 4.68x   | -78.64%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.666029 | 0.044048   | 15.12x  | -93.39%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.666029 | 0.039234   | 16.98x  | -94.11%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.666866 | 0.095469   | 6.99x   | -85.68%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.666866 | 0.048017   | 13.89x  | -92.80%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.666866 | 0.042503   | 15.69x  | -93.63%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.919465 | 0.236988   | 3.88x   | -74.23%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.919465 | 0.048277   | 19.05x  | -94.75%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.919465 | 0.042690   | 21.54x  | -95.36%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.522073 | 0.091992   | 5.68x   | -82.38%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.522073 | 0.037600   | 13.88x  | -92.80%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.522073 | 0.033042   | 15.80x  | -93.67%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.726942 | 0.121146   | 6.00x   | -83.33%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.726942 | 0.045472   | 15.99x  | -93.74%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.726942 | 0.039412   | 18.44x  | -94.58%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.492942 | 0.128036   | 3.85x   | -74.03%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.492942 | 0.043618   | 11.30x  | -91.15%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.492942 | 0.037337   | 13.20x  | -92.43%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.549010 | 0.119064   | 4.61x   | -78.31%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.549010 | 0.044046   | 12.46x  | -91.98%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.549010 | 0.037725   | 14.55x  | -93.13%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.555532 | 0.120749   | 4.60x   | -78.26%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.555532 | 0.047155   | 11.78x  | -91.51%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.555532 | 0.039071   | 14.22x  | -92.97%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.523381 | 0.197427   | 2.65x   | -62.28%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.523381 | 0.043100   | 12.14x  | -91.77%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.523381 | 0.038829   | 13.48x  | -92.58%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.497392 | 0.123772   | 4.02x   | -75.12%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.497392 | 0.044409   | 11.20x  | -91.07%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.497392 | 0.037126   | 13.40x  | -92.54%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.506649 | 0.102339   | 4.95x   | -79.80%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.506649 | 0.043799   | 11.57x  | -91.36%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.506649 | 0.037215   | 13.61x  | -92.65%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.829451 | 0.263537   | 3.15x   | -68.23%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.829451 | 0.046281   | 17.92x  | -94.42%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.829451 | 0.037532   | 22.10x  | -95.48%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.758159 | 0.103815   | 7.30x   | -86.31%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.758159 | 0.043605   | 17.39x  | -94.25%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.758159 | 0.036690   | 20.66x  | -95.16%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.513034 | 0.133783   | 3.83x   | -73.92%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.513034 | 0.042251   | 12.14x  | -91.76%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.513034 | 0.038062   | 13.48x  | -92.58%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.530410 | 0.157498   | 3.37x   | -70.31%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.530410 | 0.044986   | 11.79x  | -91.52%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.530410 | 0.039784   | 13.33x  | -92.50%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.539620 | 0.161866   | 3.33x   | -70.00%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.539620 | 0.044303   | 12.18x  | -91.79%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.539620 | 0.037676   | 14.32x  | -93.02%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.570087 | 0.117790   | 4.84x   | -79.34%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.570087 | 0.045257   | 12.60x  | -92.06%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.570087 | 0.040823   | 13.96x  | -92.84%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.784146 | 0.244535   | 3.21x   | -68.82%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.784146 | 0.045544   | 17.22x  | -94.19%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.784146 | 0.038585   | 20.32x  | -95.08%        |
</details>
<!-- PERF_RESULTS_END -->
