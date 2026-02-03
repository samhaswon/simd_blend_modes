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
| normal        | scalar | 0.201590 | 0.043132   | 4.67x   | -78.60%        |
| normal        | sse42  | 0.201590 | 0.021821   | 9.24x   | -89.18%        |
| normal        | avx2   | 0.201590 | 0.021293   | 9.47x   | -89.44%        |
| soft_light    | scalar | 0.271493 | 0.055382   | 4.90x   | -79.60%        |
| soft_light    | sse42  | 0.271493 | 0.023530   | 11.54x  | -91.33%        |
| soft_light    | avx2   | 0.271493 | 0.022450   | 12.09x  | -91.73%        |
| lighten_only  | scalar | 0.198780 | 0.061401   | 3.24x   | -69.11%        |
| lighten_only  | sse42  | 0.198780 | 0.022627   | 8.79x   | -88.62%        |
| lighten_only  | avx2   | 0.198780 | 0.022032   | 9.02x   | -88.92%        |
| screen        | scalar | 0.211612 | 0.053034   | 3.99x   | -74.94%        |
| screen        | sse42  | 0.211612 | 0.022649   | 9.34x   | -89.30%        |
| screen        | avx2   | 0.211612 | 0.022165   | 9.55x   | -89.53%        |
| dodge         | scalar | 0.212331 | 0.055215   | 3.85x   | -74.00%        |
| dodge         | sse42  | 0.212331 | 0.023736   | 8.95x   | -88.82%        |
| dodge         | avx2   | 0.212331 | 0.022431   | 9.47x   | -89.44%        |
| addition      | scalar | 0.205306 | 0.079390   | 2.59x   | -61.33%        |
| addition      | sse42  | 0.205306 | 0.023691   | 8.67x   | -88.46%        |
| addition      | avx2   | 0.205306 | 0.022631   | 9.07x   | -88.98%        |
| darken_only   | scalar | 0.202302 | 0.061431   | 3.29x   | -69.63%        |
| darken_only   | sse42  | 0.202302 | 0.023591   | 8.58x   | -88.34%        |
| darken_only   | avx2   | 0.202302 | 0.023492   | 8.61x   | -88.39%        |
| multiply      | scalar | 0.208987 | 0.053841   | 3.88x   | -74.24%        |
| multiply      | sse42  | 0.208987 | 0.022635   | 9.23x   | -89.17%        |
| multiply      | avx2   | 0.208987 | 0.022267   | 9.39x   | -89.35%        |
| hard_light    | scalar | 0.309637 | 0.100052   | 3.09x   | -67.69%        |
| hard_light    | sse42  | 0.309637 | 0.024081   | 12.86x  | -92.22%        |
| hard_light    | avx2   | 0.309637 | 0.022678   | 13.65x  | -92.68%        |
| difference    | scalar | 0.274962 | 0.052030   | 5.28x   | -81.08%        |
| difference    | sse42  | 0.274962 | 0.022237   | 12.37x  | -91.91%        |
| difference    | avx2   | 0.274962 | 0.022297   | 12.33x  | -91.89%        |
| subtract      | scalar | 0.204950 | 0.053975   | 3.80x   | -73.66%        |
| subtract      | sse42  | 0.204950 | 0.023560   | 8.70x   | -88.50%        |
| subtract      | avx2   | 0.204950 | 0.022740   | 9.01x   | -88.90%        |
| grain_extract | scalar | 0.210799 | 0.067902   | 3.10x   | -67.79%        |
| grain_extract | sse42  | 0.210799 | 0.022978   | 9.17x   | -89.10%        |
| grain_extract | avx2   | 0.210799 | 0.022447   | 9.39x   | -89.35%        |
| grain_merge   | scalar | 0.210284 | 0.067870   | 3.10x   | -67.72%        |
| grain_merge   | sse42  | 0.210284 | 0.023037   | 9.13x   | -89.04%        |
| grain_merge   | avx2   | 0.210284 | 0.022762   | 9.24x   | -89.18%        |
| divide        | scalar | 0.214662 | 0.055155   | 3.89x   | -74.31%        |
| divide        | sse42  | 0.214662 | 0.023854   | 9.00x   | -88.89%        |
| divide        | avx2   | 0.214662 | 0.022209   | 9.67x   | -89.65%        |
| overlay       | scalar | 0.280377 | 0.095566   | 2.93x   | -65.92%        |
| overlay       | sse42  | 0.280377 | 0.023513   | 11.92x  | -91.61%        |
| overlay       | avx2   | 0.280377 | 0.022046   | 12.72x  | -92.14%        |

<details>
<summary>Per-kernel, size, and type results</summary>

| Case      | Input   | Channels | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | -------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | 3        | normal        | scalar | 0.004670 | 0.001666   | 2.80x   | -64.34%        |
| 256x256   | uint8   | 3        | normal        | sse42  | 0.004670 | 0.001464   | 3.19x   | -68.65%        |
| 256x256   | uint8   | 3        | normal        | avx2   | 0.004670 | 0.001528   | 3.06x   | -67.28%        |
| 256x256   | uint8   | 3        | soft_light    | scalar | 0.007339 | 0.002117   | 3.47x   | -71.16%        |
| 256x256   | uint8   | 3        | soft_light    | sse42  | 0.007339 | 0.001566   | 4.69x   | -78.66%        |
| 256x256   | uint8   | 3        | soft_light    | avx2   | 0.007339 | 0.001562   | 4.70x   | -78.71%        |
| 256x256   | uint8   | 3        | lighten_only  | scalar | 0.006055 | 0.002403   | 2.52x   | -60.31%        |
| 256x256   | uint8   | 3        | lighten_only  | sse42  | 0.006055 | 0.001491   | 4.06x   | -75.37%        |
| 256x256   | uint8   | 3        | lighten_only  | avx2   | 0.006055 | 0.001574   | 3.85x   | -74.01%        |
| 256x256   | uint8   | 3        | screen        | scalar | 0.006000 | 0.001981   | 3.03x   | -66.98%        |
| 256x256   | uint8   | 3        | screen        | sse42  | 0.006000 | 0.001496   | 4.01x   | -75.07%        |
| 256x256   | uint8   | 3        | screen        | avx2   | 0.006000 | 0.001507   | 3.98x   | -74.88%        |
| 256x256   | uint8   | 3        | dodge         | scalar | 0.005943 | 0.002060   | 2.89x   | -65.34%        |
| 256x256   | uint8   | 3        | dodge         | sse42  | 0.005943 | 0.001572   | 3.78x   | -73.54%        |
| 256x256   | uint8   | 3        | dodge         | avx2   | 0.005943 | 0.001585   | 3.75x   | -73.32%        |
| 256x256   | uint8   | 3        | addition      | scalar | 0.005872 | 0.002659   | 2.21x   | -54.71%        |
| 256x256   | uint8   | 3        | addition      | sse42  | 0.005872 | 0.001622   | 3.62x   | -72.37%        |
| 256x256   | uint8   | 3        | addition      | avx2   | 0.005872 | 0.001496   | 3.93x   | -74.53%        |
| 256x256   | uint8   | 3        | darken_only   | scalar | 0.005799 | 0.002245   | 2.58x   | -61.29%        |
| 256x256   | uint8   | 3        | darken_only   | sse42  | 0.005799 | 0.001470   | 3.94x   | -74.64%        |
| 256x256   | uint8   | 3        | darken_only   | avx2   | 0.005799 | 0.001503   | 3.86x   | -74.08%        |
| 256x256   | uint8   | 3        | multiply      | scalar | 0.005693 | 0.002014   | 2.83x   | -64.62%        |
| 256x256   | uint8   | 3        | multiply      | sse42  | 0.005693 | 0.001491   | 3.82x   | -73.82%        |
| 256x256   | uint8   | 3        | multiply      | avx2   | 0.005693 | 0.001517   | 3.75x   | -73.35%        |
| 256x256   | uint8   | 3        | hard_light    | scalar | 0.007534 | 0.003165   | 2.38x   | -57.99%        |
| 256x256   | uint8   | 3        | hard_light    | sse42  | 0.007534 | 0.001574   | 4.79x   | -79.10%        |
| 256x256   | uint8   | 3        | hard_light    | avx2   | 0.007534 | 0.001527   | 4.93x   | -79.73%        |
| 256x256   | uint8   | 3        | difference    | scalar | 0.007595 | 0.001937   | 3.92x   | -74.49%        |
| 256x256   | uint8   | 3        | difference    | sse42  | 0.007595 | 0.001477   | 5.14x   | -80.56%        |
| 256x256   | uint8   | 3        | difference    | avx2   | 0.007595 | 0.001495   | 5.08x   | -80.31%        |
| 256x256   | uint8   | 3        | subtract      | scalar | 0.005727 | 0.001826   | 3.14x   | -68.12%        |
| 256x256   | uint8   | 3        | subtract      | sse42  | 0.005727 | 0.001504   | 3.81x   | -73.74%        |
| 256x256   | uint8   | 3        | subtract      | avx2   | 0.005727 | 0.001527   | 3.75x   | -73.33%        |
| 256x256   | uint8   | 3        | grain_extract | scalar | 0.006089 | 0.002508   | 2.43x   | -58.81%        |
| 256x256   | uint8   | 3        | grain_extract | sse42  | 0.006089 | 0.001539   | 3.96x   | -74.72%        |
| 256x256   | uint8   | 3        | grain_extract | avx2   | 0.006089 | 0.001534   | 3.97x   | -74.81%        |
| 256x256   | uint8   | 3        | grain_merge   | scalar | 0.006296 | 0.002450   | 2.57x   | -61.09%        |
| 256x256   | uint8   | 3        | grain_merge   | sse42  | 0.006296 | 0.001609   | 3.91x   | -74.45%        |
| 256x256   | uint8   | 3        | grain_merge   | avx2   | 0.006296 | 0.001652   | 3.81x   | -73.76%        |
| 256x256   | uint8   | 3        | divide        | scalar | 0.006552 | 0.002028   | 3.23x   | -69.04%        |
| 256x256   | uint8   | 3        | divide        | sse42  | 0.006552 | 0.001614   | 4.06x   | -75.37%        |
| 256x256   | uint8   | 3        | divide        | avx2   | 0.006552 | 0.001616   | 4.05x   | -75.34%        |
| 256x256   | uint8   | 3        | overlay       | scalar | 0.007601 | 0.003123   | 2.43x   | -58.91%        |
| 256x256   | uint8   | 3        | overlay       | sse42  | 0.007601 | 0.001533   | 4.96x   | -79.83%        |
| 256x256   | uint8   | 3        | overlay       | avx2   | 0.007601 | 0.001587   | 4.79x   | -79.13%        |
| 256x256   | uint8   | 4        | normal        | scalar | 0.003253 | 0.001527   | 2.13x   | -53.06%        |
| 256x256   | uint8   | 4        | normal        | sse42  | 0.003253 | 0.000184   | 17.73x  | -94.36%        |
| 256x256   | uint8   | 4        | normal        | avx2   | 0.003253 | 0.000169   | 19.27x  | -94.81%        |
| 256x256   | uint8   | 4        | soft_light    | scalar | 0.005074 | 0.002024   | 2.51x   | -60.11%        |
| 256x256   | uint8   | 4        | soft_light    | sse42  | 0.005074 | 0.000240   | 21.12x  | -95.27%        |
| 256x256   | uint8   | 4        | soft_light    | avx2   | 0.005074 | 0.000193   | 26.27x  | -96.19%        |
| 256x256   | uint8   | 4        | lighten_only  | scalar | 0.003129 | 0.001993   | 1.57x   | -36.31%        |
| 256x256   | uint8   | 4        | lighten_only  | sse42  | 0.003129 | 0.000195   | 16.02x  | -93.76%        |
| 256x256   | uint8   | 4        | lighten_only  | avx2   | 0.003129 | 0.000190   | 16.48x  | -93.93%        |
| 256x256   | uint8   | 4        | screen        | scalar | 0.003448 | 0.001813   | 1.90x   | -47.44%        |
| 256x256   | uint8   | 4        | screen        | sse42  | 0.003448 | 0.000217   | 15.92x  | -93.72%        |
| 256x256   | uint8   | 4        | screen        | avx2   | 0.003448 | 0.000198   | 17.40x  | -94.25%        |
| 256x256   | uint8   | 4        | dodge         | scalar | 0.003514 | 0.002126   | 1.65x   | -39.50%        |
| 256x256   | uint8   | 4        | dodge         | sse42  | 0.003514 | 0.000231   | 15.21x  | -93.43%        |
| 256x256   | uint8   | 4        | dodge         | avx2   | 0.003514 | 0.000194   | 18.10x  | -94.47%        |
| 256x256   | uint8   | 4        | addition      | scalar | 0.004266 | 0.002130   | 2.00x   | -50.08%        |
| 256x256   | uint8   | 4        | addition      | sse42  | 0.004266 | 0.000258   | 16.53x  | -93.95%        |
| 256x256   | uint8   | 4        | addition      | avx2   | 0.004266 | 0.000213   | 20.07x  | -95.02%        |
| 256x256   | uint8   | 4        | darken_only   | scalar | 0.003145 | 0.001985   | 1.58x   | -36.89%        |
| 256x256   | uint8   | 4        | darken_only   | sse42  | 0.003145 | 0.000203   | 15.52x  | -93.56%        |
| 256x256   | uint8   | 4        | darken_only   | avx2   | 0.003145 | 0.000192   | 16.38x  | -93.89%        |
| 256x256   | uint8   | 4        | multiply      | scalar | 0.003319 | 0.001759   | 1.89x   | -47.00%        |
| 256x256   | uint8   | 4        | multiply      | sse42  | 0.003319 | 0.000196   | 16.92x  | -94.09%        |
| 256x256   | uint8   | 4        | multiply      | avx2   | 0.003319 | 0.000217   | 15.28x  | -93.45%        |
| 256x256   | uint8   | 4        | hard_light    | scalar | 0.004966 | 0.002918   | 1.70x   | -41.24%        |
| 256x256   | uint8   | 4        | hard_light    | sse42  | 0.004966 | 0.000249   | 19.91x  | -94.98%        |
| 256x256   | uint8   | 4        | hard_light    | avx2   | 0.004966 | 0.000187   | 26.53x  | -96.23%        |
| 256x256   | uint8   | 4        | difference    | scalar | 0.005078 | 0.001771   | 2.87x   | -65.12%        |
| 256x256   | uint8   | 4        | difference    | sse42  | 0.005078 | 0.000198   | 25.65x  | -96.10%        |
| 256x256   | uint8   | 4        | difference    | avx2   | 0.005078 | 0.000186   | 27.30x  | -96.34%        |
| 256x256   | uint8   | 4        | subtract      | scalar | 0.004123 | 0.001664   | 2.48x   | -59.64%        |
| 256x256   | uint8   | 4        | subtract      | sse42  | 0.004123 | 0.000286   | 14.40x  | -93.06%        |
| 256x256   | uint8   | 4        | subtract      | avx2   | 0.004123 | 0.000197   | 20.93x  | -95.22%        |
| 256x256   | uint8   | 4        | grain_extract | scalar | 0.003422 | 0.002090   | 1.64x   | -38.91%        |
| 256x256   | uint8   | 4        | grain_extract | sse42  | 0.003422 | 0.000332   | 10.31x  | -90.30%        |
| 256x256   | uint8   | 4        | grain_extract | avx2   | 0.003422 | 0.000190   | 18.02x  | -94.45%        |
| 256x256   | uint8   | 4        | grain_merge   | scalar | 0.003241 | 0.002113   | 1.53x   | -34.81%        |
| 256x256   | uint8   | 4        | grain_merge   | sse42  | 0.003241 | 0.000234   | 13.85x  | -92.78%        |
| 256x256   | uint8   | 4        | grain_merge   | avx2   | 0.003241 | 0.000196   | 16.55x  | -93.96%        |
| 256x256   | uint8   | 4        | divide        | scalar | 0.003330 | 0.001822   | 1.83x   | -45.28%        |
| 256x256   | uint8   | 4        | divide        | sse42  | 0.003330 | 0.000221   | 15.08x  | -93.37%        |
| 256x256   | uint8   | 4        | divide        | avx2   | 0.003330 | 0.000196   | 17.02x  | -94.12%        |
| 256x256   | uint8   | 4        | overlay       | scalar | 0.004513 | 0.002792   | 1.62x   | -38.13%        |
| 256x256   | uint8   | 4        | overlay       | sse42  | 0.004513 | 0.000224   | 20.12x  | -95.03%        |
| 256x256   | uint8   | 4        | overlay       | avx2   | 0.004513 | 0.000202   | 22.36x  | -95.53%        |
| 256x256   | float32 | 3        | normal        | scalar | 0.003721 | 0.000529   | 7.04x   | -85.80%        |
| 256x256   | float32 | 3        | normal        | sse42  | 0.003721 | 0.000200   | 18.62x  | -94.63%        |
| 256x256   | float32 | 3        | normal        | avx2   | 0.003721 | 0.000207   | 17.96x  | -94.43%        |
| 256x256   | float32 | 3        | soft_light    | scalar | 0.006655 | 0.000700   | 9.50x   | -89.48%        |
| 256x256   | float32 | 3        | soft_light    | sse42  | 0.006655 | 0.000258   | 25.83x  | -96.13%        |
| 256x256   | float32 | 3        | soft_light    | avx2   | 0.006655 | 0.000195   | 34.19x  | -97.08%        |
| 256x256   | float32 | 3        | lighten_only  | scalar | 0.005310 | 0.000879   | 6.04x   | -83.45%        |
| 256x256   | float32 | 3        | lighten_only  | sse42  | 0.005310 | 0.000216   | 24.58x  | -95.93%        |
| 256x256   | float32 | 3        | lighten_only  | avx2   | 0.005310 | 0.000205   | 25.90x  | -96.14%        |
| 256x256   | float32 | 3        | screen        | scalar | 0.005311 | 0.000683   | 7.78x   | -87.15%        |
| 256x256   | float32 | 3        | screen        | sse42  | 0.005311 | 0.000247   | 21.49x  | -95.35%        |
| 256x256   | float32 | 3        | screen        | avx2   | 0.005311 | 0.000185   | 28.73x  | -96.52%        |
| 256x256   | float32 | 3        | dodge         | scalar | 0.005824 | 0.000727   | 8.02x   | -87.53%        |
| 256x256   | float32 | 3        | dodge         | sse42  | 0.005824 | 0.000256   | 22.73x  | -95.60%        |
| 256x256   | float32 | 3        | dodge         | avx2   | 0.005824 | 0.000192   | 30.29x  | -96.70%        |
| 256x256   | float32 | 3        | addition      | scalar | 0.005445 | 0.001682   | 3.24x   | -69.11%        |
| 256x256   | float32 | 3        | addition      | sse42  | 0.005445 | 0.000248   | 21.93x  | -95.44%        |
| 256x256   | float32 | 3        | addition      | avx2   | 0.005445 | 0.000218   | 24.97x  | -96.00%        |
| 256x256   | float32 | 3        | darken_only   | scalar | 0.006342 | 0.001097   | 5.78x   | -82.71%        |
| 256x256   | float32 | 3        | darken_only   | sse42  | 0.006342 | 0.000257   | 24.67x  | -95.95%        |
| 256x256   | float32 | 3        | darken_only   | avx2   | 0.006342 | 0.000205   | 30.98x  | -96.77%        |
| 256x256   | float32 | 3        | multiply      | scalar | 0.005562 | 0.000758   | 7.34x   | -86.38%        |
| 256x256   | float32 | 3        | multiply      | sse42  | 0.005562 | 0.000226   | 24.56x  | -95.93%        |
| 256x256   | float32 | 3        | multiply      | avx2   | 0.005562 | 0.000198   | 28.09x  | -96.44%        |
| 256x256   | float32 | 3        | hard_light    | scalar | 0.007251 | 0.001954   | 3.71x   | -73.06%        |
| 256x256   | float32 | 3        | hard_light    | sse42  | 0.007251 | 0.000261   | 27.83x  | -96.41%        |
| 256x256   | float32 | 3        | hard_light    | avx2   | 0.007251 | 0.000194   | 37.31x  | -97.32%        |
| 256x256   | float32 | 3        | difference    | scalar | 0.006986 | 0.000739   | 9.46x   | -89.43%        |
| 256x256   | float32 | 3        | difference    | sse42  | 0.006986 | 0.000225   | 31.08x  | -96.78%        |
| 256x256   | float32 | 3        | difference    | avx2   | 0.006986 | 0.000199   | 35.04x  | -97.15%        |
| 256x256   | float32 | 3        | subtract      | scalar | 0.005399 | 0.000781   | 6.91x   | -85.53%        |
| 256x256   | float32 | 3        | subtract      | sse42  | 0.005399 | 0.000244   | 22.15x  | -95.49%        |
| 256x256   | float32 | 3        | subtract      | avx2   | 0.005399 | 0.000190   | 28.38x  | -96.48%        |
| 256x256   | float32 | 3        | grain_extract | scalar | 0.005194 | 0.001083   | 4.80x   | -79.15%        |
| 256x256   | float32 | 3        | grain_extract | sse42  | 0.005194 | 0.000235   | 22.07x  | -95.47%        |
| 256x256   | float32 | 3        | grain_extract | avx2   | 0.005194 | 0.000187   | 27.84x  | -96.41%        |
| 256x256   | float32 | 3        | grain_merge   | scalar | 0.005110 | 0.001097   | 4.66x   | -78.54%        |
| 256x256   | float32 | 3        | grain_merge   | sse42  | 0.005110 | 0.000232   | 22.04x  | -95.46%        |
| 256x256   | float32 | 3        | grain_merge   | avx2   | 0.005110 | 0.000237   | 21.56x  | -95.36%        |
| 256x256   | float32 | 3        | divide        | scalar | 0.005275 | 0.000691   | 7.63x   | -86.90%        |
| 256x256   | float32 | 3        | divide        | sse42  | 0.005275 | 0.000245   | 21.57x  | -95.36%        |
| 256x256   | float32 | 3        | divide        | avx2   | 0.005275 | 0.000193   | 27.37x  | -96.35%        |
| 256x256   | float32 | 3        | overlay       | scalar | 0.006601 | 0.001777   | 3.71x   | -73.07%        |
| 256x256   | float32 | 3        | overlay       | sse42  | 0.006601 | 0.000247   | 26.73x  | -96.26%        |
| 256x256   | float32 | 3        | overlay       | avx2   | 0.006601 | 0.000189   | 34.85x  | -97.13%        |
| 256x256   | float32 | 4        | normal        | scalar | 0.003015 | 0.000633   | 4.76x   | -79.01%        |
| 256x256   | float32 | 4        | normal        | sse42  | 0.003015 | 0.000181   | 16.68x  | -94.00%        |
| 256x256   | float32 | 4        | normal        | avx2   | 0.003015 | 0.000163   | 18.49x  | -94.59%        |
| 256x256   | float32 | 4        | soft_light    | scalar | 0.004379 | 0.000866   | 5.06x   | -80.22%        |
| 256x256   | float32 | 4        | soft_light    | sse42  | 0.004379 | 0.000242   | 18.08x  | -94.47%        |
| 256x256   | float32 | 4        | soft_light    | avx2   | 0.004379 | 0.000185   | 23.65x  | -95.77%        |
| 256x256   | float32 | 4        | lighten_only  | scalar | 0.002995 | 0.001006   | 2.98x   | -66.42%        |
| 256x256   | float32 | 4        | lighten_only  | sse42  | 0.002995 | 0.000239   | 12.55x  | -92.03%        |
| 256x256   | float32 | 4        | lighten_only  | avx2   | 0.002995 | 0.000189   | 15.83x  | -93.68%        |
| 256x256   | float32 | 4        | screen        | scalar | 0.003367 | 0.000855   | 3.94x   | -74.62%        |
| 256x256   | float32 | 4        | screen        | sse42  | 0.003367 | 0.000233   | 14.44x  | -93.08%        |
| 256x256   | float32 | 4        | screen        | avx2   | 0.003367 | 0.000185   | 18.24x  | -94.52%        |
| 256x256   | float32 | 4        | dodge         | scalar | 0.003181 | 0.000884   | 3.60x   | -72.22%        |
| 256x256   | float32 | 4        | dodge         | sse42  | 0.003181 | 0.000328   | 9.71x   | -89.70%        |
| 256x256   | float32 | 4        | dodge         | avx2   | 0.003181 | 0.000196   | 16.20x  | -93.83%        |
| 256x256   | float32 | 4        | addition      | scalar | 0.004038 | 0.001568   | 2.57x   | -61.16%        |
| 256x256   | float32 | 4        | addition      | sse42  | 0.004038 | 0.000234   | 17.24x  | -94.20%        |
| 256x256   | float32 | 4        | addition      | avx2   | 0.004038 | 0.000201   | 20.13x  | -95.03%        |
| 256x256   | float32 | 4        | darken_only   | scalar | 0.003140 | 0.001074   | 2.92x   | -65.79%        |
| 256x256   | float32 | 4        | darken_only   | sse42  | 0.003140 | 0.000215   | 14.64x  | -93.17%        |
| 256x256   | float32 | 4        | darken_only   | avx2   | 0.003140 | 0.000200   | 15.68x  | -93.62%        |
| 256x256   | float32 | 4        | multiply      | scalar | 0.003146 | 0.000919   | 3.42x   | -70.80%        |
| 256x256   | float32 | 4        | multiply      | sse42  | 0.003146 | 0.000249   | 12.64x  | -92.09%        |
| 256x256   | float32 | 4        | multiply      | avx2   | 0.003146 | 0.000216   | 14.59x  | -93.15%        |
| 256x256   | float32 | 4        | hard_light    | scalar | 0.005423 | 0.002095   | 2.59x   | -61.37%        |
| 256x256   | float32 | 4        | hard_light    | sse42  | 0.005423 | 0.000306   | 17.72x  | -94.36%        |
| 256x256   | float32 | 4        | hard_light    | avx2   | 0.005423 | 0.000197   | 27.53x  | -96.37%        |
| 256x256   | float32 | 4        | difference    | scalar | 0.005066 | 0.000914   | 5.54x   | -81.96%        |
| 256x256   | float32 | 4        | difference    | sse42  | 0.005066 | 0.000244   | 20.73x  | -95.18%        |
| 256x256   | float32 | 4        | difference    | avx2   | 0.005066 | 0.000395   | 12.83x  | -92.20%        |
| 256x256   | float32 | 4        | subtract      | scalar | 0.004459 | 0.001104   | 4.04x   | -75.24%        |
| 256x256   | float32 | 4        | subtract      | sse42  | 0.004459 | 0.000276   | 16.15x  | -93.81%        |
| 256x256   | float32 | 4        | subtract      | avx2   | 0.004459 | 0.000260   | 17.17x  | -94.18%        |
| 256x256   | float32 | 4        | grain_extract | scalar | 0.003191 | 0.001231   | 2.59x   | -61.43%        |
| 256x256   | float32 | 4        | grain_extract | sse42  | 0.003191 | 0.000239   | 13.34x  | -92.50%        |
| 256x256   | float32 | 4        | grain_extract | avx2   | 0.003191 | 0.000188   | 16.94x  | -94.10%        |
| 256x256   | float32 | 4        | grain_merge   | scalar | 0.003054 | 0.001522   | 2.01x   | -50.17%        |
| 256x256   | float32 | 4        | grain_merge   | sse42  | 0.003054 | 0.000239   | 12.77x  | -92.17%        |
| 256x256   | float32 | 4        | grain_merge   | avx2   | 0.003054 | 0.000217   | 14.09x  | -92.90%        |
| 256x256   | float32 | 4        | divide        | scalar | 0.003204 | 0.000845   | 3.79x   | -73.62%        |
| 256x256   | float32 | 4        | divide        | sse42  | 0.003204 | 0.000249   | 12.86x  | -92.22%        |
| 256x256   | float32 | 4        | divide        | avx2   | 0.003204 | 0.000202   | 15.84x  | -93.68%        |
| 256x256   | float32 | 4        | overlay       | scalar | 0.004567 | 0.001872   | 2.44x   | -59.00%        |
| 256x256   | float32 | 4        | overlay       | sse42  | 0.004567 | 0.000258   | 17.73x  | -94.36%        |
| 256x256   | float32 | 4        | overlay       | avx2   | 0.004567 | 0.000196   | 23.30x  | -95.71%        |
| 512x512   | uint8   | 3        | normal        | scalar | 0.034457 | 0.006754   | 5.10x   | -80.40%        |
| 512x512   | uint8   | 3        | normal        | sse42  | 0.034457 | 0.005812   | 5.93x   | -83.13%        |
| 512x512   | uint8   | 3        | normal        | avx2   | 0.034457 | 0.005926   | 5.81x   | -82.80%        |
| 512x512   | uint8   | 3        | soft_light    | scalar | 0.045565 | 0.008320   | 5.48x   | -81.74%        |
| 512x512   | uint8   | 3        | soft_light    | sse42  | 0.045565 | 0.006402   | 7.12x   | -85.95%        |
| 512x512   | uint8   | 3        | soft_light    | avx2   | 0.045565 | 0.006210   | 7.34x   | -86.37%        |
| 512x512   | uint8   | 3        | lighten_only  | scalar | 0.038230 | 0.008845   | 4.32x   | -76.86%        |
| 512x512   | uint8   | 3        | lighten_only  | sse42  | 0.038230 | 0.005775   | 6.62x   | -84.89%        |
| 512x512   | uint8   | 3        | lighten_only  | avx2   | 0.038230 | 0.005993   | 6.38x   | -84.32%        |
| 512x512   | uint8   | 3        | screen        | scalar | 0.039794 | 0.007716   | 5.16x   | -80.61%        |
| 512x512   | uint8   | 3        | screen        | sse42  | 0.039794 | 0.005998   | 6.63x   | -84.93%        |
| 512x512   | uint8   | 3        | screen        | avx2   | 0.039794 | 0.006212   | 6.41x   | -84.39%        |
| 512x512   | uint8   | 3        | dodge         | scalar | 0.038619 | 0.007906   | 4.88x   | -79.53%        |
| 512x512   | uint8   | 3        | dodge         | sse42  | 0.038619 | 0.006015   | 6.42x   | -84.42%        |
| 512x512   | uint8   | 3        | dodge         | avx2   | 0.038619 | 0.006074   | 6.36x   | -84.27%        |
| 512x512   | uint8   | 3        | addition      | scalar | 0.038265 | 0.010448   | 3.66x   | -72.70%        |
| 512x512   | uint8   | 3        | addition      | sse42  | 0.038265 | 0.005809   | 6.59x   | -84.82%        |
| 512x512   | uint8   | 3        | addition      | avx2   | 0.038265 | 0.005915   | 6.47x   | -84.54%        |
| 512x512   | uint8   | 3        | darken_only   | scalar | 0.037445 | 0.008978   | 4.17x   | -76.02%        |
| 512x512   | uint8   | 3        | darken_only   | sse42  | 0.037445 | 0.005935   | 6.31x   | -84.15%        |
| 512x512   | uint8   | 3        | darken_only   | avx2   | 0.037445 | 0.006015   | 6.23x   | -83.94%        |
| 512x512   | uint8   | 3        | multiply      | scalar | 0.039021 | 0.007722   | 5.05x   | -80.21%        |
| 512x512   | uint8   | 3        | multiply      | sse42  | 0.039021 | 0.005845   | 6.68x   | -85.02%        |
| 512x512   | uint8   | 3        | multiply      | avx2   | 0.039021 | 0.006064   | 6.43x   | -84.46%        |
| 512x512   | uint8   | 3        | hard_light    | scalar | 0.048485 | 0.013133   | 3.69x   | -72.91%        |
| 512x512   | uint8   | 3        | hard_light    | sse42  | 0.048485 | 0.006422   | 7.55x   | -86.75%        |
| 512x512   | uint8   | 3        | hard_light    | avx2   | 0.048485 | 0.006256   | 7.75x   | -87.10%        |
| 512x512   | uint8   | 3        | difference    | scalar | 0.045039 | 0.007430   | 6.06x   | -83.50%        |
| 512x512   | uint8   | 3        | difference    | sse42  | 0.045039 | 0.005809   | 7.75x   | -87.10%        |
| 512x512   | uint8   | 3        | difference    | avx2   | 0.045039 | 0.005969   | 7.55x   | -86.75%        |
| 512x512   | uint8   | 3        | subtract      | scalar | 0.040439 | 0.007265   | 5.57x   | -82.03%        |
| 512x512   | uint8   | 3        | subtract      | sse42  | 0.040439 | 0.005970   | 6.77x   | -85.24%        |
| 512x512   | uint8   | 3        | subtract      | avx2   | 0.040439 | 0.006858   | 5.90x   | -83.04%        |
| 512x512   | uint8   | 3        | grain_extract | scalar | 0.037903 | 0.009010   | 4.21x   | -76.23%        |
| 512x512   | uint8   | 3        | grain_extract | sse42  | 0.037903 | 0.005836   | 6.49x   | -84.60%        |
| 512x512   | uint8   | 3        | grain_extract | avx2   | 0.037903 | 0.006004   | 6.31x   | -84.16%        |
| 512x512   | uint8   | 3        | grain_merge   | scalar | 0.038926 | 0.009355   | 4.16x   | -75.97%        |
| 512x512   | uint8   | 3        | grain_merge   | sse42  | 0.038926 | 0.005882   | 6.62x   | -84.89%        |
| 512x512   | uint8   | 3        | grain_merge   | avx2   | 0.038926 | 0.006110   | 6.37x   | -84.30%        |
| 512x512   | uint8   | 3        | divide        | scalar | 0.038711 | 0.008038   | 4.82x   | -79.23%        |
| 512x512   | uint8   | 3        | divide        | sse42  | 0.038711 | 0.006071   | 6.38x   | -84.32%        |
| 512x512   | uint8   | 3        | divide        | avx2   | 0.038711 | 0.006126   | 6.32x   | -84.18%        |
| 512x512   | uint8   | 3        | overlay       | scalar | 0.045252 | 0.012095   | 3.74x   | -73.27%        |
| 512x512   | uint8   | 3        | overlay       | sse42  | 0.045252 | 0.005986   | 7.56x   | -86.77%        |
| 512x512   | uint8   | 3        | overlay       | avx2   | 0.045252 | 0.006067   | 7.46x   | -86.59%        |
| 512x512   | uint8   | 4        | normal        | scalar | 0.024798 | 0.005552   | 4.47x   | -77.61%        |
| 512x512   | uint8   | 4        | normal        | sse42  | 0.024798 | 0.000774   | 32.04x  | -96.88%        |
| 512x512   | uint8   | 4        | normal        | avx2   | 0.024798 | 0.000688   | 36.05x  | -97.23%        |
| 512x512   | uint8   | 4        | soft_light    | scalar | 0.036260 | 0.007542   | 4.81x   | -79.20%        |
| 512x512   | uint8   | 4        | soft_light    | sse42  | 0.036260 | 0.000899   | 40.35x  | -97.52%        |
| 512x512   | uint8   | 4        | soft_light    | avx2   | 0.036260 | 0.000748   | 48.49x  | -97.94%        |
| 512x512   | uint8   | 4        | lighten_only  | scalar | 0.029398 | 0.007887   | 3.73x   | -73.17%        |
| 512x512   | uint8   | 4        | lighten_only  | sse42  | 0.029398 | 0.000793   | 37.09x  | -97.30%        |
| 512x512   | uint8   | 4        | lighten_only  | avx2   | 0.029398 | 0.000782   | 37.57x  | -97.34%        |
| 512x512   | uint8   | 4        | screen        | scalar | 0.029858 | 0.007153   | 4.17x   | -76.04%        |
| 512x512   | uint8   | 4        | screen        | sse42  | 0.029858 | 0.000799   | 37.37x  | -97.32%        |
| 512x512   | uint8   | 4        | screen        | avx2   | 0.029858 | 0.000741   | 40.28x  | -97.52%        |
| 512x512   | uint8   | 4        | dodge         | scalar | 0.029213 | 0.007348   | 3.98x   | -74.85%        |
| 512x512   | uint8   | 4        | dodge         | sse42  | 0.029213 | 0.000914   | 31.97x  | -96.87%        |
| 512x512   | uint8   | 4        | dodge         | avx2   | 0.029213 | 0.000743   | 39.31x  | -97.46%        |
| 512x512   | uint8   | 4        | addition      | scalar | 0.027967 | 0.008500   | 3.29x   | -69.61%        |
| 512x512   | uint8   | 4        | addition      | sse42  | 0.027967 | 0.001035   | 27.01x  | -96.30%        |
| 512x512   | uint8   | 4        | addition      | avx2   | 0.027967 | 0.000782   | 35.78x  | -97.20%        |
| 512x512   | uint8   | 4        | darken_only   | scalar | 0.028267 | 0.007790   | 3.63x   | -72.44%        |
| 512x512   | uint8   | 4        | darken_only   | sse42  | 0.028267 | 0.000821   | 34.44x  | -97.10%        |
| 512x512   | uint8   | 4        | darken_only   | avx2   | 0.028267 | 0.000739   | 38.27x  | -97.39%        |
| 512x512   | uint8   | 4        | multiply      | scalar | 0.028650 | 0.006891   | 4.16x   | -75.95%        |
| 512x512   | uint8   | 4        | multiply      | sse42  | 0.028650 | 0.000840   | 34.09x  | -97.07%        |
| 512x512   | uint8   | 4        | multiply      | avx2   | 0.028650 | 0.000736   | 38.93x  | -97.43%        |
| 512x512   | uint8   | 4        | hard_light    | scalar | 0.037072 | 0.011420   | 3.25x   | -69.19%        |
| 512x512   | uint8   | 4        | hard_light    | sse42  | 0.037072 | 0.001009   | 36.74x  | -97.28%        |
| 512x512   | uint8   | 4        | hard_light    | avx2   | 0.037072 | 0.000749   | 49.46x  | -97.98%        |
| 512x512   | uint8   | 4        | difference    | scalar | 0.036253 | 0.006932   | 5.23x   | -80.88%        |
| 512x512   | uint8   | 4        | difference    | sse42  | 0.036253 | 0.000801   | 45.28x  | -97.79%        |
| 512x512   | uint8   | 4        | difference    | avx2   | 0.036253 | 0.000749   | 48.41x  | -97.93%        |
| 512x512   | uint8   | 4        | subtract      | scalar | 0.028873 | 0.007094   | 4.07x   | -75.43%        |
| 512x512   | uint8   | 4        | subtract      | sse42  | 0.028873 | 0.001060   | 27.25x  | -96.33%        |
| 512x512   | uint8   | 4        | subtract      | avx2   | 0.028873 | 0.000796   | 36.28x  | -97.24%        |
| 512x512   | uint8   | 4        | grain_extract | scalar | 0.030568 | 0.008546   | 3.58x   | -72.04%        |
| 512x512   | uint8   | 4        | grain_extract | sse42  | 0.030568 | 0.000838   | 36.50x  | -97.26%        |
| 512x512   | uint8   | 4        | grain_extract | avx2   | 0.030568 | 0.000743   | 41.15x  | -97.57%        |
| 512x512   | uint8   | 4        | grain_merge   | scalar | 0.031317 | 0.008458   | 3.70x   | -72.99%        |
| 512x512   | uint8   | 4        | grain_merge   | sse42  | 0.031317 | 0.000835   | 37.51x  | -97.33%        |
| 512x512   | uint8   | 4        | grain_merge   | avx2   | 0.031317 | 0.000747   | 41.92x  | -97.61%        |
| 512x512   | uint8   | 4        | divide        | scalar | 0.029975 | 0.007237   | 4.14x   | -75.86%        |
| 512x512   | uint8   | 4        | divide        | sse42  | 0.029975 | 0.000842   | 35.60x  | -97.19%        |
| 512x512   | uint8   | 4        | divide        | avx2   | 0.029975 | 0.000741   | 40.43x  | -97.53%        |
| 512x512   | uint8   | 4        | overlay       | scalar | 0.036500 | 0.011186   | 3.26x   | -69.35%        |
| 512x512   | uint8   | 4        | overlay       | sse42  | 0.036500 | 0.000909   | 40.17x  | -97.51%        |
| 512x512   | uint8   | 4        | overlay       | avx2   | 0.036500 | 0.000755   | 48.32x  | -97.93%        |
| 512x512   | float32 | 3        | normal        | scalar | 0.029076 | 0.002087   | 13.93x  | -92.82%        |
| 512x512   | float32 | 3        | normal        | sse42  | 0.029076 | 0.000995   | 29.21x  | -96.58%        |
| 512x512   | float32 | 3        | normal        | avx2   | 0.029076 | 0.000734   | 39.59x  | -97.47%        |
| 512x512   | float32 | 3        | soft_light    | scalar | 0.041548 | 0.002875   | 14.45x  | -93.08%        |
| 512x512   | float32 | 3        | soft_light    | sse42  | 0.041548 | 0.001008   | 41.22x  | -97.57%        |
| 512x512   | float32 | 3        | soft_light    | avx2   | 0.041548 | 0.000890   | 46.68x  | -97.86%        |
| 512x512   | float32 | 3        | lighten_only  | scalar | 0.034081 | 0.003835   | 8.89x   | -88.75%        |
| 512x512   | float32 | 3        | lighten_only  | sse42  | 0.034081 | 0.000991   | 34.39x  | -97.09%        |
| 512x512   | float32 | 3        | lighten_only  | avx2   | 0.034081 | 0.000886   | 38.45x  | -97.40%        |
| 512x512   | float32 | 3        | screen        | scalar | 0.036240 | 0.002734   | 13.26x  | -92.46%        |
| 512x512   | float32 | 3        | screen        | sse42  | 0.036240 | 0.000980   | 36.99x  | -97.30%        |
| 512x512   | float32 | 3        | screen        | avx2   | 0.036240 | 0.000840   | 43.12x  | -97.68%        |
| 512x512   | float32 | 3        | dodge         | scalar | 0.033690 | 0.002895   | 11.64x  | -91.41%        |
| 512x512   | float32 | 3        | dodge         | sse42  | 0.033690 | 0.001085   | 31.06x  | -96.78%        |
| 512x512   | float32 | 3        | dodge         | avx2   | 0.033690 | 0.000876   | 38.46x  | -97.40%        |
| 512x512   | float32 | 3        | addition      | scalar | 0.033519 | 0.006843   | 4.90x   | -79.58%        |
| 512x512   | float32 | 3        | addition      | sse42  | 0.033519 | 0.001138   | 29.45x  | -96.60%        |
| 512x512   | float32 | 3        | addition      | avx2   | 0.033519 | 0.000900   | 37.23x  | -97.31%        |
| 512x512   | float32 | 3        | darken_only   | scalar | 0.033028 | 0.003537   | 9.34x   | -89.29%        |
| 512x512   | float32 | 3        | darken_only   | sse42  | 0.033028 | 0.000967   | 34.16x  | -97.07%        |
| 512x512   | float32 | 3        | darken_only   | avx2   | 0.033028 | 0.000835   | 39.55x  | -97.47%        |
| 512x512   | float32 | 3        | multiply      | scalar | 0.032858 | 0.002576   | 12.75x  | -92.16%        |
| 512x512   | float32 | 3        | multiply      | sse42  | 0.032858 | 0.000929   | 35.37x  | -97.17%        |
| 512x512   | float32 | 3        | multiply      | avx2   | 0.032858 | 0.000948   | 34.66x  | -97.11%        |
| 512x512   | float32 | 3        | hard_light    | scalar | 0.042601 | 0.007712   | 5.52x   | -81.90%        |
| 512x512   | float32 | 3        | hard_light    | sse42  | 0.042601 | 0.001044   | 40.80x  | -97.55%        |
| 512x512   | float32 | 3        | hard_light    | avx2   | 0.042601 | 0.000816   | 52.19x  | -98.08%        |
| 512x512   | float32 | 3        | difference    | scalar | 0.039548 | 0.002717   | 14.56x  | -93.13%        |
| 512x512   | float32 | 3        | difference    | sse42  | 0.039548 | 0.000969   | 40.82x  | -97.55%        |
| 512x512   | float32 | 3        | difference    | avx2   | 0.039548 | 0.000923   | 42.87x  | -97.67%        |
| 512x512   | float32 | 3        | subtract      | scalar | 0.032911 | 0.003565   | 9.23x   | -89.17%        |
| 512x512   | float32 | 3        | subtract      | sse42  | 0.032911 | 0.000983   | 33.49x  | -97.01%        |
| 512x512   | float32 | 3        | subtract      | avx2   | 0.032911 | 0.000835   | 39.41x  | -97.46%        |
| 512x512   | float32 | 3        | grain_extract | scalar | 0.033540 | 0.004297   | 7.81x   | -87.19%        |
| 512x512   | float32 | 3        | grain_extract | sse42  | 0.033540 | 0.000943   | 35.57x  | -97.19%        |
| 512x512   | float32 | 3        | grain_extract | avx2   | 0.033540 | 0.000817   | 41.05x  | -97.56%        |
| 512x512   | float32 | 3        | grain_merge   | scalar | 0.032194 | 0.004306   | 7.48x   | -86.62%        |
| 512x512   | float32 | 3        | grain_merge   | sse42  | 0.032194 | 0.000933   | 34.51x  | -97.10%        |
| 512x512   | float32 | 3        | grain_merge   | avx2   | 0.032194 | 0.000816   | 39.45x  | -97.47%        |
| 512x512   | float32 | 3        | divide        | scalar | 0.032811 | 0.002816   | 11.65x  | -91.42%        |
| 512x512   | float32 | 3        | divide        | sse42  | 0.032811 | 0.001096   | 29.94x  | -96.66%        |
| 512x512   | float32 | 3        | divide        | avx2   | 0.032811 | 0.000878   | 37.39x  | -97.33%        |
| 512x512   | float32 | 3        | overlay       | scalar | 0.040574 | 0.007122   | 5.70x   | -82.45%        |
| 512x512   | float32 | 3        | overlay       | sse42  | 0.040574 | 0.000988   | 41.05x  | -97.56%        |
| 512x512   | float32 | 3        | overlay       | avx2   | 0.040574 | 0.000883   | 45.95x  | -97.82%        |
| 512x512   | float32 | 4        | normal        | scalar | 0.020622 | 0.002635   | 7.83x   | -87.22%        |
| 512x512   | float32 | 4        | normal        | sse42  | 0.020622 | 0.000992   | 20.79x  | -95.19%        |
| 512x512   | float32 | 4        | normal        | avx2   | 0.020622 | 0.000875   | 23.57x  | -95.76%        |
| 512x512   | float32 | 4        | soft_light    | scalar | 0.034278 | 0.003382   | 10.13x  | -90.13%        |
| 512x512   | float32 | 4        | soft_light    | sse42  | 0.034278 | 0.001107   | 30.95x  | -96.77%        |
| 512x512   | float32 | 4        | soft_light    | avx2   | 0.034278 | 0.000993   | 34.51x  | -97.10%        |
| 512x512   | float32 | 4        | lighten_only  | scalar | 0.026514 | 0.004314   | 6.15x   | -83.73%        |
| 512x512   | float32 | 4        | lighten_only  | sse42  | 0.026514 | 0.001134   | 23.39x  | -95.72%        |
| 512x512   | float32 | 4        | lighten_only  | avx2   | 0.026514 | 0.001075   | 24.66x  | -95.95%        |
| 512x512   | float32 | 4        | screen        | scalar | 0.028152 | 0.003367   | 8.36x   | -88.04%        |
| 512x512   | float32 | 4        | screen        | sse42  | 0.028152 | 0.001137   | 24.75x  | -95.96%        |
| 512x512   | float32 | 4        | screen        | avx2   | 0.028152 | 0.000902   | 31.21x  | -96.80%        |
| 512x512   | float32 | 4        | dodge         | scalar | 0.027329 | 0.003511   | 7.78x   | -87.15%        |
| 512x512   | float32 | 4        | dodge         | sse42  | 0.027329 | 0.001157   | 23.61x  | -95.77%        |
| 512x512   | float32 | 4        | dodge         | avx2   | 0.027329 | 0.000938   | 29.12x  | -96.57%        |
| 512x512   | float32 | 4        | addition      | scalar | 0.026257 | 0.005894   | 4.46x   | -77.55%        |
| 512x512   | float32 | 4        | addition      | sse42  | 0.026257 | 0.001139   | 23.04x  | -95.66%        |
| 512x512   | float32 | 4        | addition      | avx2   | 0.026257 | 0.000995   | 26.39x  | -96.21%        |
| 512x512   | float32 | 4        | darken_only   | scalar | 0.025737 | 0.003965   | 6.49x   | -84.59%        |
| 512x512   | float32 | 4        | darken_only   | sse42  | 0.025737 | 0.001085   | 23.73x  | -95.79%        |
| 512x512   | float32 | 4        | darken_only   | avx2   | 0.025737 | 0.000892   | 28.84x  | -96.53%        |
| 512x512   | float32 | 4        | multiply      | scalar | 0.026064 | 0.003316   | 7.86x   | -87.28%        |
| 512x512   | float32 | 4        | multiply      | sse42  | 0.026064 | 0.001099   | 23.71x  | -95.78%        |
| 512x512   | float32 | 4        | multiply      | avx2   | 0.026064 | 0.000939   | 27.74x  | -96.40%        |
| 512x512   | float32 | 4        | hard_light    | scalar | 0.034992 | 0.007971   | 4.39x   | -77.22%        |
| 512x512   | float32 | 4        | hard_light    | sse42  | 0.034992 | 0.001215   | 28.80x  | -96.53%        |
| 512x512   | float32 | 4        | hard_light    | avx2   | 0.034992 | 0.001034   | 33.85x  | -97.05%        |
| 512x512   | float32 | 4        | difference    | scalar | 0.033373 | 0.003279   | 10.18x  | -90.18%        |
| 512x512   | float32 | 4        | difference    | sse42  | 0.033373 | 0.001123   | 29.72x  | -96.64%        |
| 512x512   | float32 | 4        | difference    | avx2   | 0.033373 | 0.001016   | 32.84x  | -96.96%        |
| 512x512   | float32 | 4        | subtract      | scalar | 0.026258 | 0.004049   | 6.48x   | -84.58%        |
| 512x512   | float32 | 4        | subtract      | sse42  | 0.026258 | 0.001156   | 22.71x  | -95.60%        |
| 512x512   | float32 | 4        | subtract      | avx2   | 0.026258 | 0.000989   | 26.56x  | -96.24%        |
| 512x512   | float32 | 4        | grain_extract | scalar | 0.026725 | 0.005106   | 5.23x   | -80.89%        |
| 512x512   | float32 | 4        | grain_extract | sse42  | 0.026725 | 0.001634   | 16.35x  | -93.89%        |
| 512x512   | float32 | 4        | grain_extract | avx2   | 0.026725 | 0.001020   | 26.20x  | -96.18%        |
| 512x512   | float32 | 4        | grain_merge   | scalar | 0.028828 | 0.005019   | 5.74x   | -82.59%        |
| 512x512   | float32 | 4        | grain_merge   | sse42  | 0.028828 | 0.001187   | 24.28x  | -95.88%        |
| 512x512   | float32 | 4        | grain_merge   | avx2   | 0.028828 | 0.000967   | 29.82x  | -96.65%        |
| 512x512   | float32 | 4        | divide        | scalar | 0.027692 | 0.003597   | 7.70x   | -87.01%        |
| 512x512   | float32 | 4        | divide        | sse42  | 0.027692 | 0.001092   | 25.37x  | -96.06%        |
| 512x512   | float32 | 4        | divide        | avx2   | 0.027692 | 0.000885   | 31.30x  | -96.81%        |
| 512x512   | float32 | 4        | overlay       | scalar | 0.034879 | 0.007781   | 4.48x   | -77.69%        |
| 512x512   | float32 | 4        | overlay       | sse42  | 0.034879 | 0.001141   | 30.58x  | -96.73%        |
| 512x512   | float32 | 4        | overlay       | avx2   | 0.034879 | 0.001003   | 34.78x  | -97.12%        |
| 1024x1024 | uint8   | 3        | normal        | scalar | 0.106035 | 0.027322   | 3.88x   | -74.23%        |
| 1024x1024 | uint8   | 3        | normal        | sse42  | 0.106035 | 0.023601   | 4.49x   | -77.74%        |
| 1024x1024 | uint8   | 3        | normal        | avx2   | 0.106035 | 0.024948   | 4.25x   | -76.47%        |
| 1024x1024 | uint8   | 3        | soft_light    | scalar | 0.131195 | 0.031344   | 4.19x   | -76.11%        |
| 1024x1024 | uint8   | 3        | soft_light    | sse42  | 0.131195 | 0.025082   | 5.23x   | -80.88%        |
| 1024x1024 | uint8   | 3        | soft_light    | avx2   | 0.131195 | 0.025377   | 5.17x   | -80.66%        |
| 1024x1024 | uint8   | 3        | lighten_only  | scalar | 0.111527 | 0.036297   | 3.07x   | -67.45%        |
| 1024x1024 | uint8   | 3        | lighten_only  | sse42  | 0.111527 | 0.023047   | 4.84x   | -79.33%        |
| 1024x1024 | uint8   | 3        | lighten_only  | avx2   | 0.111527 | 0.023570   | 4.73x   | -78.87%        |
| 1024x1024 | uint8   | 3        | screen        | scalar | 0.109268 | 0.030060   | 3.64x   | -72.49%        |
| 1024x1024 | uint8   | 3        | screen        | sse42  | 0.109268 | 0.023540   | 4.64x   | -78.46%        |
| 1024x1024 | uint8   | 3        | screen        | avx2   | 0.109268 | 0.023934   | 4.57x   | -78.10%        |
| 1024x1024 | uint8   | 3        | dodge         | scalar | 0.112652 | 0.031945   | 3.53x   | -71.64%        |
| 1024x1024 | uint8   | 3        | dodge         | sse42  | 0.112652 | 0.024113   | 4.67x   | -78.59%        |
| 1024x1024 | uint8   | 3        | dodge         | avx2   | 0.112652 | 0.024055   | 4.68x   | -78.65%        |
| 1024x1024 | uint8   | 3        | addition      | scalar | 0.107683 | 0.041930   | 2.57x   | -61.06%        |
| 1024x1024 | uint8   | 3        | addition      | sse42  | 0.107683 | 0.023900   | 4.51x   | -77.81%        |
| 1024x1024 | uint8   | 3        | addition      | avx2   | 0.107683 | 0.023864   | 4.51x   | -77.84%        |
| 1024x1024 | uint8   | 3        | darken_only   | scalar | 0.106606 | 0.034401   | 3.10x   | -67.73%        |
| 1024x1024 | uint8   | 3        | darken_only   | sse42  | 0.106606 | 0.023696   | 4.50x   | -77.77%        |
| 1024x1024 | uint8   | 3        | darken_only   | avx2   | 0.106606 | 0.025560   | 4.17x   | -76.02%        |
| 1024x1024 | uint8   | 3        | multiply      | scalar | 0.107367 | 0.032351   | 3.32x   | -69.87%        |
| 1024x1024 | uint8   | 3        | multiply      | sse42  | 0.107367 | 0.023595   | 4.55x   | -78.02%        |
| 1024x1024 | uint8   | 3        | multiply      | avx2   | 0.107367 | 0.024105   | 4.45x   | -77.55%        |
| 1024x1024 | uint8   | 3        | hard_light    | scalar | 0.147524 | 0.052047   | 2.83x   | -64.72%        |
| 1024x1024 | uint8   | 3        | hard_light    | sse42  | 0.147524 | 0.025041   | 5.89x   | -83.03%        |
| 1024x1024 | uint8   | 3        | hard_light    | avx2   | 0.147524 | 0.024562   | 6.01x   | -83.35%        |
| 1024x1024 | uint8   | 3        | difference    | scalar | 0.140928 | 0.031133   | 4.53x   | -77.91%        |
| 1024x1024 | uint8   | 3        | difference    | sse42  | 0.140928 | 0.023795   | 5.92x   | -83.12%        |
| 1024x1024 | uint8   | 3        | difference    | avx2   | 0.140928 | 0.023776   | 5.93x   | -83.13%        |
| 1024x1024 | uint8   | 3        | subtract      | scalar | 0.110315 | 0.029571   | 3.73x   | -73.19%        |
| 1024x1024 | uint8   | 3        | subtract      | sse42  | 0.110315 | 0.024065   | 4.58x   | -78.18%        |
| 1024x1024 | uint8   | 3        | subtract      | avx2   | 0.110315 | 0.024273   | 4.54x   | -78.00%        |
| 1024x1024 | uint8   | 3        | grain_extract | scalar | 0.111973 | 0.037146   | 3.01x   | -66.83%        |
| 1024x1024 | uint8   | 3        | grain_extract | sse42  | 0.111973 | 0.023895   | 4.69x   | -78.66%        |
| 1024x1024 | uint8   | 3        | grain_extract | avx2   | 0.111973 | 0.024417   | 4.59x   | -78.19%        |
| 1024x1024 | uint8   | 3        | grain_merge   | scalar | 0.109032 | 0.037296   | 2.92x   | -65.79%        |
| 1024x1024 | uint8   | 3        | grain_merge   | sse42  | 0.109032 | 0.024159   | 4.51x   | -77.84%        |
| 1024x1024 | uint8   | 3        | grain_merge   | avx2   | 0.109032 | 0.024139   | 4.52x   | -77.86%        |
| 1024x1024 | uint8   | 3        | divide        | scalar | 0.113694 | 0.032126   | 3.54x   | -71.74%        |
| 1024x1024 | uint8   | 3        | divide        | sse42  | 0.113694 | 0.025049   | 4.54x   | -77.97%        |
| 1024x1024 | uint8   | 3        | divide        | avx2   | 0.113694 | 0.025799   | 4.41x   | -77.31%        |
| 1024x1024 | uint8   | 3        | overlay       | scalar | 0.137578 | 0.048158   | 2.86x   | -65.00%        |
| 1024x1024 | uint8   | 3        | overlay       | sse42  | 0.137578 | 0.023600   | 5.83x   | -82.85%        |
| 1024x1024 | uint8   | 3        | overlay       | avx2   | 0.137578 | 0.024321   | 5.66x   | -82.32%        |
| 1024x1024 | uint8   | 4        | normal        | scalar | 0.074337 | 0.021820   | 3.41x   | -70.65%        |
| 1024x1024 | uint8   | 4        | normal        | sse42  | 0.074337 | 0.002941   | 25.28x  | -96.04%        |
| 1024x1024 | uint8   | 4        | normal        | avx2   | 0.074337 | 0.002668   | 27.86x  | -96.41%        |
| 1024x1024 | uint8   | 4        | soft_light    | scalar | 0.110557 | 0.031396   | 3.52x   | -71.60%        |
| 1024x1024 | uint8   | 4        | soft_light    | sse42  | 0.110557 | 0.003577   | 30.91x  | -96.76%        |
| 1024x1024 | uint8   | 4        | soft_light    | avx2   | 0.110557 | 0.003013   | 36.69x  | -97.27%        |
| 1024x1024 | uint8   | 4        | lighten_only  | scalar | 0.082950 | 0.032850   | 2.53x   | -60.40%        |
| 1024x1024 | uint8   | 4        | lighten_only  | sse42  | 0.082950 | 0.003290   | 25.21x  | -96.03%        |
| 1024x1024 | uint8   | 4        | lighten_only  | avx2   | 0.082950 | 0.003041   | 27.28x  | -96.33%        |
| 1024x1024 | uint8   | 4        | screen        | scalar | 0.086277 | 0.028211   | 3.06x   | -67.30%        |
| 1024x1024 | uint8   | 4        | screen        | sse42  | 0.086277 | 0.003196   | 27.00x  | -96.30%        |
| 1024x1024 | uint8   | 4        | screen        | avx2   | 0.086277 | 0.002983   | 28.92x  | -96.54%        |
| 1024x1024 | uint8   | 4        | dodge         | scalar | 0.084572 | 0.029759   | 2.84x   | -64.81%        |
| 1024x1024 | uint8   | 4        | dodge         | sse42  | 0.084572 | 0.003712   | 22.78x  | -95.61%        |
| 1024x1024 | uint8   | 4        | dodge         | avx2   | 0.084572 | 0.003061   | 27.63x  | -96.38%        |
| 1024x1024 | uint8   | 4        | addition      | scalar | 0.085701 | 0.034148   | 2.51x   | -60.15%        |
| 1024x1024 | uint8   | 4        | addition      | sse42  | 0.085701 | 0.004100   | 20.91x  | -95.22%        |
| 1024x1024 | uint8   | 4        | addition      | avx2   | 0.085701 | 0.003104   | 27.61x  | -96.38%        |
| 1024x1024 | uint8   | 4        | darken_only   | scalar | 0.083704 | 0.030781   | 2.72x   | -63.23%        |
| 1024x1024 | uint8   | 4        | darken_only   | sse42  | 0.083704 | 0.003188   | 26.26x  | -96.19%        |
| 1024x1024 | uint8   | 4        | darken_only   | avx2   | 0.083704 | 0.002965   | 28.23x  | -96.46%        |
| 1024x1024 | uint8   | 4        | multiply      | scalar | 0.083534 | 0.029304   | 2.85x   | -64.92%        |
| 1024x1024 | uint8   | 4        | multiply      | sse42  | 0.083534 | 0.003689   | 22.65x  | -95.58%        |
| 1024x1024 | uint8   | 4        | multiply      | avx2   | 0.083534 | 0.003489   | 23.94x  | -95.82%        |
| 1024x1024 | uint8   | 4        | hard_light    | scalar | 0.144881 | 0.046533   | 3.11x   | -67.88%        |
| 1024x1024 | uint8   | 4        | hard_light    | sse42  | 0.144881 | 0.004013   | 36.10x  | -97.23%        |
| 1024x1024 | uint8   | 4        | hard_light    | avx2   | 0.144881 | 0.003100   | 46.74x  | -97.86%        |
| 1024x1024 | uint8   | 4        | difference    | scalar | 0.130092 | 0.029889   | 4.35x   | -77.02%        |
| 1024x1024 | uint8   | 4        | difference    | sse42  | 0.130092 | 0.003449   | 37.71x  | -97.35%        |
| 1024x1024 | uint8   | 4        | difference    | avx2   | 0.130092 | 0.003050   | 42.65x  | -97.66%        |
| 1024x1024 | uint8   | 4        | subtract      | scalar | 0.086567 | 0.029248   | 2.96x   | -66.21%        |
| 1024x1024 | uint8   | 4        | subtract      | sse42  | 0.086567 | 0.004178   | 20.72x  | -95.17%        |
| 1024x1024 | uint8   | 4        | subtract      | avx2   | 0.086567 | 0.003119   | 27.76x  | -96.40%        |
| 1024x1024 | uint8   | 4        | grain_extract | scalar | 0.085615 | 0.035499   | 2.41x   | -58.54%        |
| 1024x1024 | uint8   | 4        | grain_extract | sse42  | 0.085615 | 0.003369   | 25.42x  | -96.07%        |
| 1024x1024 | uint8   | 4        | grain_extract | avx2   | 0.085615 | 0.003002   | 28.52x  | -96.49%        |
| 1024x1024 | uint8   | 4        | grain_merge   | scalar | 0.087146 | 0.035011   | 2.49x   | -59.82%        |
| 1024x1024 | uint8   | 4        | grain_merge   | sse42  | 0.087146 | 0.003318   | 26.26x  | -96.19%        |
| 1024x1024 | uint8   | 4        | grain_merge   | avx2   | 0.087146 | 0.002943   | 29.62x  | -96.62%        |
| 1024x1024 | uint8   | 4        | divide        | scalar | 0.086155 | 0.028421   | 3.03x   | -67.01%        |
| 1024x1024 | uint8   | 4        | divide        | sse42  | 0.086155 | 0.003400   | 25.34x  | -96.05%        |
| 1024x1024 | uint8   | 4        | divide        | avx2   | 0.086155 | 0.002994   | 28.78x  | -96.53%        |
| 1024x1024 | uint8   | 4        | overlay       | scalar | 0.115672 | 0.044577   | 2.59x   | -61.46%        |
| 1024x1024 | uint8   | 4        | overlay       | sse42  | 0.115672 | 0.003575   | 32.36x  | -96.91%        |
| 1024x1024 | uint8   | 4        | overlay       | avx2   | 0.115672 | 0.003048   | 37.95x  | -97.36%        |
| 1024x1024 | float32 | 3        | normal        | scalar | 0.090008 | 0.008837   | 10.18x  | -90.18%        |
| 1024x1024 | float32 | 3        | normal        | sse42  | 0.090008 | 0.003859   | 23.32x  | -95.71%        |
| 1024x1024 | float32 | 3        | normal        | avx2   | 0.090008 | 0.003008   | 29.93x  | -96.66%        |
| 1024x1024 | float32 | 3        | soft_light    | scalar | 0.129712 | 0.011702   | 11.08x  | -90.98%        |
| 1024x1024 | float32 | 3        | soft_light    | sse42  | 0.129712 | 0.003828   | 33.88x  | -97.05%        |
| 1024x1024 | float32 | 3        | soft_light    | avx2   | 0.129712 | 0.003117   | 41.62x  | -97.60%        |
| 1024x1024 | float32 | 3        | lighten_only  | scalar | 0.100972 | 0.013752   | 7.34x   | -86.38%        |
| 1024x1024 | float32 | 3        | lighten_only  | sse42  | 0.100972 | 0.003587   | 28.15x  | -96.45%        |
| 1024x1024 | float32 | 3        | lighten_only  | avx2   | 0.100972 | 0.003088   | 32.70x  | -96.94%        |
| 1024x1024 | float32 | 3        | screen        | scalar | 0.102020 | 0.011621   | 8.78x   | -88.61%        |
| 1024x1024 | float32 | 3        | screen        | sse42  | 0.102020 | 0.003944   | 25.87x  | -96.13%        |
| 1024x1024 | float32 | 3        | screen        | avx2   | 0.102020 | 0.003443   | 29.63x  | -96.62%        |
| 1024x1024 | float32 | 3        | dodge         | scalar | 0.100534 | 0.014542   | 6.91x   | -85.53%        |
| 1024x1024 | float32 | 3        | dodge         | sse42  | 0.100534 | 0.004918   | 20.44x  | -95.11%        |
| 1024x1024 | float32 | 3        | dodge         | avx2   | 0.100534 | 0.003689   | 27.25x  | -96.33%        |
| 1024x1024 | float32 | 3        | addition      | scalar | 0.103042 | 0.026905   | 3.83x   | -73.89%        |
| 1024x1024 | float32 | 3        | addition      | sse42  | 0.103042 | 0.004123   | 24.99x  | -96.00%        |
| 1024x1024 | float32 | 3        | addition      | avx2   | 0.103042 | 0.003424   | 30.09x  | -96.68%        |
| 1024x1024 | float32 | 3        | darken_only   | scalar | 0.096901 | 0.015034   | 6.45x   | -84.49%        |
| 1024x1024 | float32 | 3        | darken_only   | sse42  | 0.096901 | 0.003805   | 25.47x  | -96.07%        |
| 1024x1024 | float32 | 3        | darken_only   | avx2   | 0.096901 | 0.003552   | 27.28x  | -96.33%        |
| 1024x1024 | float32 | 3        | multiply      | scalar | 0.098779 | 0.012314   | 8.02x   | -87.53%        |
| 1024x1024 | float32 | 3        | multiply      | sse42  | 0.098779 | 0.003831   | 25.79x  | -96.12%        |
| 1024x1024 | float32 | 3        | multiply      | avx2   | 0.098779 | 0.003335   | 29.62x  | -96.62%        |
| 1024x1024 | float32 | 3        | hard_light    | scalar | 0.133655 | 0.031750   | 4.21x   | -76.25%        |
| 1024x1024 | float32 | 3        | hard_light    | sse42  | 0.133655 | 0.004223   | 31.65x  | -96.84%        |
| 1024x1024 | float32 | 3        | hard_light    | avx2   | 0.133655 | 0.003462   | 38.61x  | -97.41%        |
| 1024x1024 | float32 | 3        | difference    | scalar | 0.127919 | 0.011848   | 10.80x  | -90.74%        |
| 1024x1024 | float32 | 3        | difference    | sse42  | 0.127919 | 0.003946   | 32.42x  | -96.92%        |
| 1024x1024 | float32 | 3        | difference    | avx2   | 0.127919 | 0.003480   | 36.76x  | -97.28%        |
| 1024x1024 | float32 | 3        | subtract      | scalar | 0.106314 | 0.013044   | 8.15x   | -87.73%        |
| 1024x1024 | float32 | 3        | subtract      | sse42  | 0.106314 | 0.004299   | 24.73x  | -95.96%        |
| 1024x1024 | float32 | 3        | subtract      | avx2   | 0.106314 | 0.005217   | 20.38x  | -95.09%        |
| 1024x1024 | float32 | 3        | grain_extract | scalar | 0.118195 | 0.019598   | 6.03x   | -83.42%        |
| 1024x1024 | float32 | 3        | grain_extract | sse42  | 0.118195 | 0.005710   | 20.70x  | -95.17%        |
| 1024x1024 | float32 | 3        | grain_extract | avx2   | 0.118195 | 0.005015   | 23.57x  | -95.76%        |
| 1024x1024 | float32 | 3        | grain_merge   | scalar | 0.109041 | 0.019397   | 5.62x   | -82.21%        |
| 1024x1024 | float32 | 3        | grain_merge   | sse42  | 0.109041 | 0.004108   | 26.54x  | -96.23%        |
| 1024x1024 | float32 | 3        | grain_merge   | avx2   | 0.109041 | 0.003216   | 33.91x  | -97.05%        |
| 1024x1024 | float32 | 3        | divide        | scalar | 0.103674 | 0.013290   | 7.80x   | -87.18%        |
| 1024x1024 | float32 | 3        | divide        | sse42  | 0.103674 | 0.004067   | 25.49x  | -96.08%        |
| 1024x1024 | float32 | 3        | divide        | avx2   | 0.103674 | 0.003542   | 29.27x  | -96.58%        |
| 1024x1024 | float32 | 3        | overlay       | scalar | 0.132340 | 0.029699   | 4.46x   | -77.56%        |
| 1024x1024 | float32 | 3        | overlay       | sse42  | 0.132340 | 0.004050   | 32.68x  | -96.94%        |
| 1024x1024 | float32 | 3        | overlay       | avx2   | 0.132340 | 0.003259   | 40.60x  | -97.54%        |
| 1024x1024 | float32 | 4        | normal        | scalar | 0.069776 | 0.011177   | 6.24x   | -83.98%        |
| 1024x1024 | float32 | 4        | normal        | sse42  | 0.069776 | 0.003743   | 18.64x  | -94.64%        |
| 1024x1024 | float32 | 4        | normal        | avx2   | 0.069776 | 0.003078   | 22.67x  | -95.59%        |
| 1024x1024 | float32 | 4        | soft_light    | scalar | 0.105504 | 0.014592   | 7.23x   | -86.17%        |
| 1024x1024 | float32 | 4        | soft_light    | sse42  | 0.105504 | 0.004036   | 26.14x  | -96.17%        |
| 1024x1024 | float32 | 4        | soft_light    | avx2   | 0.105504 | 0.003316   | 31.81x  | -96.86%        |
| 1024x1024 | float32 | 4        | lighten_only  | scalar | 0.070727 | 0.017834   | 3.97x   | -74.79%        |
| 1024x1024 | float32 | 4        | lighten_only  | sse42  | 0.070727 | 0.004210   | 16.80x  | -94.05%        |
| 1024x1024 | float32 | 4        | lighten_only  | avx2   | 0.070727 | 0.003677   | 19.23x  | -94.80%        |
| 1024x1024 | float32 | 4        | screen        | scalar | 0.077334 | 0.015963   | 4.84x   | -79.36%        |
| 1024x1024 | float32 | 4        | screen        | sse42  | 0.077334 | 0.003948   | 19.59x  | -94.89%        |
| 1024x1024 | float32 | 4        | screen        | avx2   | 0.077334 | 0.003538   | 21.86x  | -95.42%        |
| 1024x1024 | float32 | 4        | dodge         | scalar | 0.076836 | 0.015326   | 5.01x   | -80.05%        |
| 1024x1024 | float32 | 4        | dodge         | sse42  | 0.076836 | 0.004307   | 17.84x  | -94.39%        |
| 1024x1024 | float32 | 4        | dodge         | avx2   | 0.076836 | 0.003255   | 23.61x  | -95.76%        |
| 1024x1024 | float32 | 4        | addition      | scalar | 0.078053 | 0.024349   | 3.21x   | -68.80%        |
| 1024x1024 | float32 | 4        | addition      | sse42  | 0.078053 | 0.004037   | 19.34x  | -94.83%        |
| 1024x1024 | float32 | 4        | addition      | avx2   | 0.078053 | 0.003424   | 22.80x  | -95.61%        |
| 1024x1024 | float32 | 4        | darken_only   | scalar | 0.078281 | 0.017016   | 4.60x   | -78.26%        |
| 1024x1024 | float32 | 4        | darken_only   | sse42  | 0.078281 | 0.004546   | 17.22x  | -94.19%        |
| 1024x1024 | float32 | 4        | darken_only   | avx2   | 0.078281 | 0.003446   | 22.72x  | -95.60%        |
| 1024x1024 | float32 | 4        | multiply      | scalar | 0.078643 | 0.014497   | 5.42x   | -81.57%        |
| 1024x1024 | float32 | 4        | multiply      | sse42  | 0.078643 | 0.004327   | 18.17x  | -94.50%        |
| 1024x1024 | float32 | 4        | multiply      | avx2   | 0.078643 | 0.003652   | 21.53x  | -95.36%        |
| 1024x1024 | float32 | 4        | hard_light    | scalar | 0.113353 | 0.033878   | 3.35x   | -70.11%        |
| 1024x1024 | float32 | 4        | hard_light    | sse42  | 0.113353 | 0.004177   | 27.14x  | -96.32%        |
| 1024x1024 | float32 | 4        | hard_light    | avx2   | 0.113353 | 0.003394   | 33.40x  | -97.01%        |
| 1024x1024 | float32 | 4        | difference    | scalar | 0.103351 | 0.014128   | 7.32x   | -86.33%        |
| 1024x1024 | float32 | 4        | difference    | sse42  | 0.103351 | 0.003813   | 27.10x  | -96.31%        |
| 1024x1024 | float32 | 4        | difference    | avx2   | 0.103351 | 0.003525   | 29.32x  | -96.59%        |
| 1024x1024 | float32 | 4        | subtract      | scalar | 0.072715 | 0.017391   | 4.18x   | -76.08%        |
| 1024x1024 | float32 | 4        | subtract      | sse42  | 0.072715 | 0.004200   | 17.31x  | -94.22%        |
| 1024x1024 | float32 | 4        | subtract      | avx2   | 0.072715 | 0.003398   | 21.40x  | -95.33%        |
| 1024x1024 | float32 | 4        | grain_extract | scalar | 0.077301 | 0.020620   | 3.75x   | -73.33%        |
| 1024x1024 | float32 | 4        | grain_extract | sse42  | 0.077301 | 0.004213   | 18.35x  | -94.55%        |
| 1024x1024 | float32 | 4        | grain_extract | avx2   | 0.077301 | 0.003570   | 21.65x  | -95.38%        |
| 1024x1024 | float32 | 4        | grain_merge   | scalar | 0.076684 | 0.020636   | 3.72x   | -73.09%        |
| 1024x1024 | float32 | 4        | grain_merge   | sse42  | 0.076684 | 0.004267   | 17.97x  | -94.44%        |
| 1024x1024 | float32 | 4        | grain_merge   | avx2   | 0.076684 | 0.003655   | 20.98x  | -95.23%        |
| 1024x1024 | float32 | 4        | divide        | scalar | 0.079458 | 0.016461   | 4.83x   | -79.28%        |
| 1024x1024 | float32 | 4        | divide        | sse42  | 0.079458 | 0.004571   | 17.38x  | -94.25%        |
| 1024x1024 | float32 | 4        | divide        | avx2   | 0.079458 | 0.003653   | 21.75x  | -95.40%        |
| 1024x1024 | float32 | 4        | overlay       | scalar | 0.107011 | 0.032121   | 3.33x   | -69.98%        |
| 1024x1024 | float32 | 4        | overlay       | sse42  | 0.107011 | 0.004526   | 23.64x  | -95.77%        |
| 1024x1024 | float32 | 4        | overlay       | avx2   | 0.107011 | 0.003547   | 30.17x  | -96.69%        |
| 2048x2048 | uint8   | 3        | normal        | scalar | 0.400740 | 0.110300   | 3.63x   | -72.48%        |
| 2048x2048 | uint8   | 3        | normal        | sse42  | 0.400740 | 0.094039   | 4.26x   | -76.53%        |
| 2048x2048 | uint8   | 3        | normal        | avx2   | 0.400740 | 0.098383   | 4.07x   | -75.45%        |
| 2048x2048 | uint8   | 3        | soft_light    | scalar | 0.515943 | 0.135654   | 3.80x   | -73.71%        |
| 2048x2048 | uint8   | 3        | soft_light    | sse42  | 0.515943 | 0.097596   | 5.29x   | -81.08%        |
| 2048x2048 | uint8   | 3        | soft_light    | avx2   | 0.515943 | 0.105103   | 4.91x   | -79.63%        |
| 2048x2048 | uint8   | 3        | lighten_only  | scalar | 0.388284 | 0.147439   | 2.63x   | -62.03%        |
| 2048x2048 | uint8   | 3        | lighten_only  | sse42  | 0.388284 | 0.096120   | 4.04x   | -75.25%        |
| 2048x2048 | uint8   | 3        | lighten_only  | avx2   | 0.388284 | 0.096331   | 4.03x   | -75.19%        |
| 2048x2048 | uint8   | 3        | screen        | scalar | 0.407827 | 0.127448   | 3.20x   | -68.75%        |
| 2048x2048 | uint8   | 3        | screen        | sse42  | 0.407827 | 0.099025   | 4.12x   | -75.72%        |
| 2048x2048 | uint8   | 3        | screen        | avx2   | 0.407827 | 0.097395   | 4.19x   | -76.12%        |
| 2048x2048 | uint8   | 3        | dodge         | scalar | 0.415744 | 0.129014   | 3.22x   | -68.97%        |
| 2048x2048 | uint8   | 3        | dodge         | sse42  | 0.415744 | 0.101216   | 4.11x   | -75.65%        |
| 2048x2048 | uint8   | 3        | dodge         | avx2   | 0.415744 | 0.102739   | 4.05x   | -75.29%        |
| 2048x2048 | uint8   | 3        | addition      | scalar | 0.439302 | 0.180913   | 2.43x   | -58.82%        |
| 2048x2048 | uint8   | 3        | addition      | sse42  | 0.439302 | 0.101441   | 4.33x   | -76.91%        |
| 2048x2048 | uint8   | 3        | addition      | avx2   | 0.439302 | 0.109678   | 4.01x   | -75.03%        |
| 2048x2048 | uint8   | 3        | darken_only   | scalar | 0.432477 | 0.155990   | 2.77x   | -63.93%        |
| 2048x2048 | uint8   | 3        | darken_only   | sse42  | 0.432477 | 0.131681   | 3.28x   | -69.55%        |
| 2048x2048 | uint8   | 3        | darken_only   | avx2   | 0.432477 | 0.123415   | 3.50x   | -71.46%        |
| 2048x2048 | uint8   | 3        | multiply      | scalar | 0.433854 | 0.143009   | 3.03x   | -67.04%        |
| 2048x2048 | uint8   | 3        | multiply      | sse42  | 0.433854 | 0.099928   | 4.34x   | -76.97%        |
| 2048x2048 | uint8   | 3        | multiply      | avx2   | 0.433854 | 0.098686   | 4.40x   | -77.25%        |
| 2048x2048 | uint8   | 3        | hard_light    | scalar | 0.608444 | 0.205730   | 2.96x   | -66.19%        |
| 2048x2048 | uint8   | 3        | hard_light    | sse42  | 0.608444 | 0.097144   | 6.26x   | -84.03%        |
| 2048x2048 | uint8   | 3        | hard_light    | avx2   | 0.608444 | 0.096714   | 6.29x   | -84.10%        |
| 2048x2048 | uint8   | 3        | difference    | scalar | 0.513136 | 0.123294   | 4.16x   | -75.97%        |
| 2048x2048 | uint8   | 3        | difference    | sse42  | 0.513136 | 0.094276   | 5.44x   | -81.63%        |
| 2048x2048 | uint8   | 3        | difference    | avx2   | 0.513136 | 0.103487   | 4.96x   | -79.83%        |
| 2048x2048 | uint8   | 3        | subtract      | scalar | 0.395619 | 0.117024   | 3.38x   | -70.42%        |
| 2048x2048 | uint8   | 3        | subtract      | sse42  | 0.395619 | 0.095331   | 4.15x   | -75.90%        |
| 2048x2048 | uint8   | 3        | subtract      | avx2   | 0.395619 | 0.100767   | 3.93x   | -74.53%        |
| 2048x2048 | uint8   | 3        | grain_extract | scalar | 0.402119 | 0.146199   | 2.75x   | -63.64%        |
| 2048x2048 | uint8   | 3        | grain_extract | sse42  | 0.402119 | 0.097112   | 4.14x   | -75.85%        |
| 2048x2048 | uint8   | 3        | grain_extract | avx2   | 0.402119 | 0.100582   | 4.00x   | -74.99%        |
| 2048x2048 | uint8   | 3        | grain_merge   | scalar | 0.407627 | 0.149998   | 2.72x   | -63.20%        |
| 2048x2048 | uint8   | 3        | grain_merge   | sse42  | 0.407627 | 0.096247   | 4.24x   | -76.39%        |
| 2048x2048 | uint8   | 3        | grain_merge   | avx2   | 0.407627 | 0.097935   | 4.16x   | -75.97%        |
| 2048x2048 | uint8   | 3        | divide        | scalar | 0.412169 | 0.130790   | 3.15x   | -68.27%        |
| 2048x2048 | uint8   | 3        | divide        | sse42  | 0.412169 | 0.099451   | 4.14x   | -75.87%        |
| 2048x2048 | uint8   | 3        | divide        | avx2   | 0.412169 | 0.100187   | 4.11x   | -75.69%        |
| 2048x2048 | uint8   | 3        | overlay       | scalar | 0.509361 | 0.194459   | 2.62x   | -61.82%        |
| 2048x2048 | uint8   | 3        | overlay       | sse42  | 0.509361 | 0.098961   | 5.15x   | -80.57%        |
| 2048x2048 | uint8   | 3        | overlay       | avx2   | 0.509361 | 0.096333   | 5.29x   | -81.09%        |
| 2048x2048 | uint8   | 4        | normal        | scalar | 0.295864 | 0.087459   | 3.38x   | -70.44%        |
| 2048x2048 | uint8   | 4        | normal        | sse42  | 0.295864 | 0.011551   | 25.61x  | -96.10%        |
| 2048x2048 | uint8   | 4        | normal        | avx2   | 0.295864 | 0.010533   | 28.09x  | -96.44%        |
| 2048x2048 | uint8   | 4        | soft_light    | scalar | 0.401672 | 0.117904   | 3.41x   | -70.65%        |
| 2048x2048 | uint8   | 4        | soft_light    | sse42  | 0.401672 | 0.013837   | 29.03x  | -96.56%        |
| 2048x2048 | uint8   | 4        | soft_light    | avx2   | 0.401672 | 0.011939   | 33.64x  | -97.03%        |
| 2048x2048 | uint8   | 4        | lighten_only  | scalar | 0.286264 | 0.124692   | 2.30x   | -56.44%        |
| 2048x2048 | uint8   | 4        | lighten_only  | sse42  | 0.286264 | 0.012579   | 22.76x  | -95.61%        |
| 2048x2048 | uint8   | 4        | lighten_only  | avx2   | 0.286264 | 0.011793   | 24.27x  | -95.88%        |
| 2048x2048 | uint8   | 4        | screen        | scalar | 0.309720 | 0.113319   | 2.73x   | -63.41%        |
| 2048x2048 | uint8   | 4        | screen        | sse42  | 0.309720 | 0.013193   | 23.48x  | -95.74%        |
| 2048x2048 | uint8   | 4        | screen        | avx2   | 0.309720 | 0.012078   | 25.64x  | -96.10%        |
| 2048x2048 | uint8   | 4        | dodge         | scalar | 0.313924 | 0.118252   | 2.65x   | -62.33%        |
| 2048x2048 | uint8   | 4        | dodge         | sse42  | 0.313924 | 0.014773   | 21.25x  | -95.29%        |
| 2048x2048 | uint8   | 4        | dodge         | avx2   | 0.313924 | 0.012027   | 26.10x  | -96.17%        |
| 2048x2048 | uint8   | 4        | addition      | scalar | 0.299865 | 0.136475   | 2.20x   | -54.49%        |
| 2048x2048 | uint8   | 4        | addition      | sse42  | 0.299865 | 0.016691   | 17.97x  | -94.43%        |
| 2048x2048 | uint8   | 4        | addition      | avx2   | 0.299865 | 0.012443   | 24.10x  | -95.85%        |
| 2048x2048 | uint8   | 4        | darken_only   | scalar | 0.297635 | 0.125120   | 2.38x   | -57.96%        |
| 2048x2048 | uint8   | 4        | darken_only   | sse42  | 0.297635 | 0.012524   | 23.77x  | -95.79%        |
| 2048x2048 | uint8   | 4        | darken_only   | avx2   | 0.297635 | 0.012539   | 23.74x  | -95.79%        |
| 2048x2048 | uint8   | 4        | multiply      | scalar | 0.295073 | 0.112422   | 2.62x   | -61.90%        |
| 2048x2048 | uint8   | 4        | multiply      | sse42  | 0.295073 | 0.012772   | 23.10x  | -95.67%        |
| 2048x2048 | uint8   | 4        | multiply      | avx2   | 0.295073 | 0.013845   | 21.31x  | -95.31%        |
| 2048x2048 | uint8   | 4        | hard_light    | scalar | 0.482680 | 0.182364   | 2.65x   | -62.22%        |
| 2048x2048 | uint8   | 4        | hard_light    | sse42  | 0.482680 | 0.014929   | 32.33x  | -96.91%        |
| 2048x2048 | uint8   | 4        | hard_light    | avx2   | 0.482680 | 0.011955   | 40.37x  | -97.52%        |
| 2048x2048 | uint8   | 4        | difference    | scalar | 0.405832 | 0.112840   | 3.60x   | -72.20%        |
| 2048x2048 | uint8   | 4        | difference    | sse42  | 0.405832 | 0.012975   | 31.28x  | -96.80%        |
| 2048x2048 | uint8   | 4        | difference    | avx2   | 0.405832 | 0.013546   | 29.96x  | -96.66%        |
| 2048x2048 | uint8   | 4        | subtract      | scalar | 0.299529 | 0.114968   | 2.61x   | -61.62%        |
| 2048x2048 | uint8   | 4        | subtract      | sse42  | 0.299529 | 0.017402   | 17.21x  | -94.19%        |
| 2048x2048 | uint8   | 4        | subtract      | avx2   | 0.299529 | 0.012469   | 24.02x  | -95.84%        |
| 2048x2048 | uint8   | 4        | grain_extract | scalar | 0.313528 | 0.138373   | 2.27x   | -55.87%        |
| 2048x2048 | uint8   | 4        | grain_extract | sse42  | 0.313528 | 0.014010   | 22.38x  | -95.53%        |
| 2048x2048 | uint8   | 4        | grain_extract | avx2   | 0.313528 | 0.012274   | 25.54x  | -96.09%        |
| 2048x2048 | uint8   | 4        | grain_merge   | scalar | 0.318249 | 0.133929   | 2.38x   | -57.92%        |
| 2048x2048 | uint8   | 4        | grain_merge   | sse42  | 0.318249 | 0.013424   | 23.71x  | -95.78%        |
| 2048x2048 | uint8   | 4        | grain_merge   | avx2   | 0.318249 | 0.012033   | 26.45x  | -96.22%        |
| 2048x2048 | uint8   | 4        | divide        | scalar | 0.325632 | 0.122161   | 2.67x   | -62.48%        |
| 2048x2048 | uint8   | 4        | divide        | sse42  | 0.325632 | 0.013749   | 23.68x  | -95.78%        |
| 2048x2048 | uint8   | 4        | divide        | avx2   | 0.325632 | 0.011998   | 27.14x  | -96.32%        |
| 2048x2048 | uint8   | 4        | overlay       | scalar | 0.425427 | 0.177161   | 2.40x   | -58.36%        |
| 2048x2048 | uint8   | 4        | overlay       | sse42  | 0.425427 | 0.014311   | 29.73x  | -96.64%        |
| 2048x2048 | uint8   | 4        | overlay       | avx2   | 0.425427 | 0.012304   | 34.58x  | -97.11%        |
| 2048x2048 | float32 | 3        | normal        | scalar | 0.349954 | 0.037599   | 9.31x   | -89.26%        |
| 2048x2048 | float32 | 3        | normal        | sse42  | 0.349954 | 0.018530   | 18.89x  | -94.71%        |
| 2048x2048 | float32 | 3        | normal        | avx2   | 0.349954 | 0.015838   | 22.10x  | -95.47%        |
| 2048x2048 | float32 | 3        | soft_light    | scalar | 0.457853 | 0.049983   | 9.16x   | -89.08%        |
| 2048x2048 | float32 | 3        | soft_light    | sse42  | 0.457853 | 0.020468   | 22.37x  | -95.53%        |
| 2048x2048 | float32 | 3        | soft_light    | avx2   | 0.457853 | 0.017404   | 26.31x  | -96.20%        |
| 2048x2048 | float32 | 3        | lighten_only  | scalar | 0.337332 | 0.058751   | 5.74x   | -82.58%        |
| 2048x2048 | float32 | 3        | lighten_only  | sse42  | 0.337332 | 0.020567   | 16.40x  | -93.90%        |
| 2048x2048 | float32 | 3        | lighten_only  | avx2   | 0.337332 | 0.017690   | 19.07x  | -94.76%        |
| 2048x2048 | float32 | 3        | screen        | scalar | 0.361015 | 0.049420   | 7.30x   | -86.31%        |
| 2048x2048 | float32 | 3        | screen        | sse42  | 0.361015 | 0.020619   | 17.51x  | -94.29%        |
| 2048x2048 | float32 | 3        | screen        | avx2   | 0.361015 | 0.018116   | 19.93x  | -94.98%        |
| 2048x2048 | float32 | 3        | dodge         | scalar | 0.363005 | 0.051619   | 7.03x   | -85.78%        |
| 2048x2048 | float32 | 3        | dodge         | sse42  | 0.363005 | 0.022036   | 16.47x  | -93.93%        |
| 2048x2048 | float32 | 3        | dodge         | avx2   | 0.363005 | 0.018294   | 19.84x  | -94.96%        |
| 2048x2048 | float32 | 3        | addition      | scalar | 0.345282 | 0.109289   | 3.16x   | -68.35%        |
| 2048x2048 | float32 | 3        | addition      | sse42  | 0.345282 | 0.020770   | 16.62x  | -93.98%        |
| 2048x2048 | float32 | 3        | addition      | avx2   | 0.345282 | 0.017992   | 19.19x  | -94.79%        |
| 2048x2048 | float32 | 3        | darken_only   | scalar | 0.343424 | 0.058368   | 5.88x   | -83.00%        |
| 2048x2048 | float32 | 3        | darken_only   | sse42  | 0.343424 | 0.019912   | 17.25x  | -94.20%        |
| 2048x2048 | float32 | 3        | darken_only   | avx2   | 0.343424 | 0.018275   | 18.79x  | -94.68%        |
| 2048x2048 | float32 | 3        | multiply      | scalar | 0.355415 | 0.048014   | 7.40x   | -86.49%        |
| 2048x2048 | float32 | 3        | multiply      | sse42  | 0.355415 | 0.020356   | 17.46x  | -94.27%        |
| 2048x2048 | float32 | 3        | multiply      | avx2   | 0.355415 | 0.017777   | 19.99x  | -95.00%        |
| 2048x2048 | float32 | 3        | hard_light    | scalar | 0.508976 | 0.131771   | 3.86x   | -74.11%        |
| 2048x2048 | float32 | 3        | hard_light    | sse42  | 0.508976 | 0.021706   | 23.45x  | -95.74%        |
| 2048x2048 | float32 | 3        | hard_light    | avx2   | 0.508976 | 0.018221   | 27.93x  | -96.42%        |
| 2048x2048 | float32 | 3        | difference    | scalar | 0.457511 | 0.045753   | 10.00x  | -90.00%        |
| 2048x2048 | float32 | 3        | difference    | sse42  | 0.457511 | 0.019514   | 23.45x  | -95.73%        |
| 2048x2048 | float32 | 3        | difference    | avx2   | 0.457511 | 0.018891   | 24.22x  | -95.87%        |
| 2048x2048 | float32 | 3        | subtract      | scalar | 0.352058 | 0.054064   | 6.51x   | -84.64%        |
| 2048x2048 | float32 | 3        | subtract      | sse42  | 0.352058 | 0.021191   | 16.61x  | -93.98%        |
| 2048x2048 | float32 | 3        | subtract      | avx2   | 0.352058 | 0.018776   | 18.75x  | -94.67%        |
| 2048x2048 | float32 | 3        | grain_extract | scalar | 0.351683 | 0.075298   | 4.67x   | -78.59%        |
| 2048x2048 | float32 | 3        | grain_extract | sse42  | 0.351683 | 0.019792   | 17.77x  | -94.37%        |
| 2048x2048 | float32 | 3        | grain_extract | avx2   | 0.351683 | 0.017704   | 19.86x  | -94.97%        |
| 2048x2048 | float32 | 3        | grain_merge   | scalar | 0.353660 | 0.074413   | 4.75x   | -78.96%        |
| 2048x2048 | float32 | 3        | grain_merge   | sse42  | 0.353660 | 0.020622   | 17.15x  | -94.17%        |
| 2048x2048 | float32 | 3        | grain_merge   | avx2   | 0.353660 | 0.018056   | 19.59x  | -94.89%        |
| 2048x2048 | float32 | 3        | divide        | scalar | 0.362296 | 0.050176   | 7.22x   | -86.15%        |
| 2048x2048 | float32 | 3        | divide        | sse42  | 0.362296 | 0.020646   | 17.55x  | -94.30%        |
| 2048x2048 | float32 | 3        | divide        | avx2   | 0.362296 | 0.017706   | 20.46x  | -95.11%        |
| 2048x2048 | float32 | 3        | overlay       | scalar | 0.462199 | 0.120910   | 3.82x   | -73.84%        |
| 2048x2048 | float32 | 3        | overlay       | sse42  | 0.462199 | 0.020628   | 22.41x  | -95.54%        |
| 2048x2048 | float32 | 3        | overlay       | avx2   | 0.462199 | 0.017296   | 26.72x  | -96.26%        |
| 2048x2048 | float32 | 4        | normal        | scalar | 0.278183 | 0.047128   | 5.90x   | -83.06%        |
| 2048x2048 | float32 | 4        | normal        | sse42  | 0.278183 | 0.020859   | 13.34x  | -92.50%        |
| 2048x2048 | float32 | 4        | normal        | avx2   | 0.278183 | 0.017747   | 15.67x  | -93.62%        |
| 2048x2048 | float32 | 4        | soft_light    | scalar | 0.387054 | 0.061085   | 6.34x   | -84.22%        |
| 2048x2048 | float32 | 4        | soft_light    | sse42  | 0.387054 | 0.023637   | 16.37x  | -93.89%        |
| 2048x2048 | float32 | 4        | soft_light    | avx2   | 0.387054 | 0.019901   | 19.45x  | -94.86%        |
| 2048x2048 | float32 | 4        | lighten_only  | scalar | 0.265670 | 0.068919   | 3.85x   | -74.06%        |
| 2048x2048 | float32 | 4        | lighten_only  | sse42  | 0.265670 | 0.022562   | 11.78x  | -91.51%        |
| 2048x2048 | float32 | 4        | lighten_only  | avx2   | 0.265670 | 0.023298   | 11.40x  | -91.23%        |
| 2048x2048 | float32 | 4        | screen        | scalar | 0.297909 | 0.058487   | 5.09x   | -80.37%        |
| 2048x2048 | float32 | 4        | screen        | sse42  | 0.297909 | 0.022795   | 13.07x  | -92.35%        |
| 2048x2048 | float32 | 4        | screen        | avx2   | 0.297909 | 0.024145   | 12.34x  | -91.90%        |
| 2048x2048 | float32 | 4        | dodge         | scalar | 0.293157 | 0.064412   | 4.55x   | -78.03%        |
| 2048x2048 | float32 | 4        | dodge         | sse42  | 0.293157 | 0.024520   | 11.96x  | -91.64%        |
| 2048x2048 | float32 | 4        | dodge         | avx2   | 0.293157 | 0.020433   | 14.35x  | -93.03%        |
| 2048x2048 | float32 | 4        | addition      | scalar | 0.278464 | 0.099489   | 2.80x   | -64.27%        |
| 2048x2048 | float32 | 4        | addition      | sse42  | 0.278464 | 0.022556   | 12.35x  | -91.90%        |
| 2048x2048 | float32 | 4        | addition      | avx2   | 0.278464 | 0.019446   | 14.32x  | -93.02%        |
| 2048x2048 | float32 | 4        | darken_only   | scalar | 0.276939 | 0.070881   | 3.91x   | -74.41%        |
| 2048x2048 | float32 | 4        | darken_only   | sse42  | 0.276939 | 0.022305   | 12.42x  | -91.95%        |
| 2048x2048 | float32 | 4        | darken_only   | avx2   | 0.276939 | 0.023339   | 11.87x  | -91.57%        |
| 2048x2048 | float32 | 4        | multiply      | scalar | 0.279439 | 0.057996   | 4.82x   | -79.25%        |
| 2048x2048 | float32 | 4        | multiply      | sse42  | 0.279439 | 0.022100   | 12.64x  | -92.09%        |
| 2048x2048 | float32 | 4        | multiply      | avx2   | 0.279439 | 0.023471   | 11.91x  | -91.60%        |
| 2048x2048 | float32 | 4        | hard_light    | scalar | 0.437204 | 0.134382   | 3.25x   | -69.26%        |
| 2048x2048 | float32 | 4        | hard_light    | sse42  | 0.437204 | 0.023682   | 18.46x  | -94.58%        |
| 2048x2048 | float32 | 4        | hard_light    | avx2   | 0.437204 | 0.021981   | 19.89x  | -94.97%        |
| 2048x2048 | float32 | 4        | difference    | scalar | 0.386168 | 0.060208   | 6.41x   | -84.41%        |
| 2048x2048 | float32 | 4        | difference    | sse42  | 0.386168 | 0.023621   | 16.35x  | -93.88%        |
| 2048x2048 | float32 | 4        | difference    | avx2   | 0.386168 | 0.024563   | 15.72x  | -93.64%        |
| 2048x2048 | float32 | 4        | subtract      | scalar | 0.279424 | 0.069473   | 4.02x   | -75.14%        |
| 2048x2048 | float32 | 4        | subtract      | sse42  | 0.279424 | 0.023487   | 11.90x  | -91.59%        |
| 2048x2048 | float32 | 4        | subtract      | avx2   | 0.279424 | 0.020690   | 13.51x  | -92.60%        |
| 2048x2048 | float32 | 4        | grain_extract | scalar | 0.288249 | 0.085915   | 3.36x   | -70.19%        |
| 2048x2048 | float32 | 4        | grain_extract | sse42  | 0.288249 | 0.023490   | 12.27x  | -91.85%        |
| 2048x2048 | float32 | 4        | grain_extract | avx2   | 0.288249 | 0.025497   | 11.31x  | -91.15%        |
| 2048x2048 | float32 | 4        | grain_merge   | scalar | 0.284775 | 0.084694   | 3.36x   | -70.26%        |
| 2048x2048 | float32 | 4        | grain_merge   | sse42  | 0.284775 | 0.022012   | 12.94x  | -92.27%        |
| 2048x2048 | float32 | 4        | grain_merge   | avx2   | 0.284775 | 0.023629   | 12.05x  | -91.70%        |
| 2048x2048 | float32 | 4        | divide        | scalar | 0.293519 | 0.061153   | 4.80x   | -79.17%        |
| 2048x2048 | float32 | 4        | divide        | sse42  | 0.293519 | 0.023020   | 12.75x  | -92.16%        |
| 2048x2048 | float32 | 4        | divide        | avx2   | 0.293519 | 0.021491   | 13.66x  | -92.68%        |
| 2048x2048 | float32 | 4        | overlay       | scalar | 0.418593 | 0.129246   | 3.24x   | -69.12%        |
| 2048x2048 | float32 | 4        | overlay       | sse42  | 0.418593 | 0.022647   | 18.48x  | -94.59%        |
| 2048x2048 | float32 | 4        | overlay       | avx2   | 0.418593 | 0.019379   | 21.60x  | -95.37%        |
| 1280x720  | uint8   | 3        | normal        | scalar | 0.077624 | 0.024094   | 3.22x   | -68.96%        |
| 1280x720  | uint8   | 3        | normal        | sse42  | 0.077624 | 0.022020   | 3.53x   | -71.63%        |
| 1280x720  | uint8   | 3        | normal        | avx2   | 0.077624 | 0.020899   | 3.71x   | -73.08%        |
| 1280x720  | uint8   | 3        | soft_light    | scalar | 0.112118 | 0.027340   | 4.10x   | -75.61%        |
| 1280x720  | uint8   | 3        | soft_light    | sse42  | 0.112118 | 0.021001   | 5.34x   | -81.27%        |
| 1280x720  | uint8   | 3        | soft_light    | avx2   | 0.112118 | 0.021198   | 5.29x   | -81.09%        |
| 1280x720  | uint8   | 3        | lighten_only  | scalar | 0.088219 | 0.031584   | 2.79x   | -64.20%        |
| 1280x720  | uint8   | 3        | lighten_only  | sse42  | 0.088219 | 0.024038   | 3.67x   | -72.75%        |
| 1280x720  | uint8   | 3        | lighten_only  | avx2   | 0.088219 | 0.020813   | 4.24x   | -76.41%        |
| 1280x720  | uint8   | 3        | screen        | scalar | 0.094204 | 0.026758   | 3.52x   | -71.60%        |
| 1280x720  | uint8   | 3        | screen        | sse42  | 0.094204 | 0.020762   | 4.54x   | -77.96%        |
| 1280x720  | uint8   | 3        | screen        | avx2   | 0.094204 | 0.021101   | 4.46x   | -77.60%        |
| 1280x720  | uint8   | 3        | dodge         | scalar | 0.095127 | 0.027421   | 3.47x   | -71.17%        |
| 1280x720  | uint8   | 3        | dodge         | sse42  | 0.095127 | 0.021031   | 4.52x   | -77.89%        |
| 1280x720  | uint8   | 3        | dodge         | avx2   | 0.095127 | 0.021458   | 4.43x   | -77.44%        |
| 1280x720  | uint8   | 3        | addition      | scalar | 0.087605 | 0.036759   | 2.38x   | -58.04%        |
| 1280x720  | uint8   | 3        | addition      | sse42  | 0.087605 | 0.020575   | 4.26x   | -76.51%        |
| 1280x720  | uint8   | 3        | addition      | avx2   | 0.087605 | 0.020821   | 4.21x   | -76.23%        |
| 1280x720  | uint8   | 3        | darken_only   | scalar | 0.087264 | 0.029939   | 2.91x   | -65.69%        |
| 1280x720  | uint8   | 3        | darken_only   | sse42  | 0.087264 | 0.020379   | 4.28x   | -76.65%        |
| 1280x720  | uint8   | 3        | darken_only   | avx2   | 0.087264 | 0.021139   | 4.13x   | -75.78%        |
| 1280x720  | uint8   | 3        | multiply      | scalar | 0.091008 | 0.027131   | 3.35x   | -70.19%        |
| 1280x720  | uint8   | 3        | multiply      | sse42  | 0.091008 | 0.020559   | 4.43x   | -77.41%        |
| 1280x720  | uint8   | 3        | multiply      | avx2   | 0.091008 | 0.021912   | 4.15x   | -75.92%        |
| 1280x720  | uint8   | 3        | hard_light    | scalar | 0.117807 | 0.045417   | 2.59x   | -61.45%        |
| 1280x720  | uint8   | 3        | hard_light    | sse42  | 0.117807 | 0.021730   | 5.42x   | -81.55%        |
| 1280x720  | uint8   | 3        | hard_light    | avx2   | 0.117807 | 0.021154   | 5.57x   | -82.04%        |
| 1280x720  | uint8   | 3        | difference    | scalar | 0.114495 | 0.026699   | 4.29x   | -76.68%        |
| 1280x720  | uint8   | 3        | difference    | sse42  | 0.114495 | 0.020834   | 5.50x   | -81.80%        |
| 1280x720  | uint8   | 3        | difference    | avx2   | 0.114495 | 0.021301   | 5.38x   | -81.40%        |
| 1280x720  | uint8   | 3        | subtract      | scalar | 0.086441 | 0.025255   | 3.42x   | -70.78%        |
| 1280x720  | uint8   | 3        | subtract      | sse42  | 0.086441 | 0.020548   | 4.21x   | -76.23%        |
| 1280x720  | uint8   | 3        | subtract      | avx2   | 0.086441 | 0.020872   | 4.14x   | -75.85%        |
| 1280x720  | uint8   | 3        | grain_extract | scalar | 0.089395 | 0.031590   | 2.83x   | -64.66%        |
| 1280x720  | uint8   | 3        | grain_extract | sse42  | 0.089395 | 0.020573   | 4.35x   | -76.99%        |
| 1280x720  | uint8   | 3        | grain_extract | avx2   | 0.089395 | 0.021233   | 4.21x   | -76.25%        |
| 1280x720  | uint8   | 3        | grain_merge   | scalar | 0.089344 | 0.031810   | 2.81x   | -64.40%        |
| 1280x720  | uint8   | 3        | grain_merge   | sse42  | 0.089344 | 0.020686   | 4.32x   | -76.85%        |
| 1280x720  | uint8   | 3        | grain_merge   | avx2   | 0.089344 | 0.021638   | 4.13x   | -75.78%        |
| 1280x720  | uint8   | 3        | divide        | scalar | 0.090919 | 0.028000   | 3.25x   | -69.20%        |
| 1280x720  | uint8   | 3        | divide        | sse42  | 0.090919 | 0.021518   | 4.23x   | -76.33%        |
| 1280x720  | uint8   | 3        | divide        | avx2   | 0.090919 | 0.021234   | 4.28x   | -76.65%        |
| 1280x720  | uint8   | 3        | overlay       | scalar | 0.114545 | 0.042666   | 2.68x   | -62.75%        |
| 1280x720  | uint8   | 3        | overlay       | sse42  | 0.114545 | 0.021160   | 5.41x   | -81.53%        |
| 1280x720  | uint8   | 3        | overlay       | avx2   | 0.114545 | 0.021799   | 5.25x   | -80.97%        |
| 1280x720  | uint8   | 4        | normal        | scalar | 0.062976 | 0.018890   | 3.33x   | -70.00%        |
| 1280x720  | uint8   | 4        | normal        | sse42  | 0.062976 | 0.002554   | 24.65x  | -95.94%        |
| 1280x720  | uint8   | 4        | normal        | avx2   | 0.062976 | 0.002301   | 27.36x  | -96.35%        |
| 1280x720  | uint8   | 4        | soft_light    | scalar | 0.097718 | 0.026878   | 3.64x   | -72.49%        |
| 1280x720  | uint8   | 4        | soft_light    | sse42  | 0.097718 | 0.003181   | 30.72x  | -96.75%        |
| 1280x720  | uint8   | 4        | soft_light    | avx2   | 0.097718 | 0.002688   | 36.35x  | -97.25%        |
| 1280x720  | uint8   | 4        | lighten_only  | scalar | 0.074452 | 0.027879   | 2.67x   | -62.55%        |
| 1280x720  | uint8   | 4        | lighten_only  | sse42  | 0.074452 | 0.002755   | 27.02x  | -96.30%        |
| 1280x720  | uint8   | 4        | lighten_only  | avx2   | 0.074452 | 0.002588   | 28.77x  | -96.52%        |
| 1280x720  | uint8   | 4        | screen        | scalar | 0.077001 | 0.024814   | 3.10x   | -67.77%        |
| 1280x720  | uint8   | 4        | screen        | sse42  | 0.077001 | 0.002797   | 27.53x  | -96.37%        |
| 1280x720  | uint8   | 4        | screen        | avx2   | 0.077001 | 0.002595   | 29.68x  | -96.63%        |
| 1280x720  | uint8   | 4        | dodge         | scalar | 0.076621 | 0.026060   | 2.94x   | -65.99%        |
| 1280x720  | uint8   | 4        | dodge         | sse42  | 0.076621 | 0.003386   | 22.63x  | -95.58%        |
| 1280x720  | uint8   | 4        | dodge         | avx2   | 0.076621 | 0.002657   | 28.83x  | -96.53%        |
| 1280x720  | uint8   | 4        | addition      | scalar | 0.072031 | 0.030048   | 2.40x   | -58.28%        |
| 1280x720  | uint8   | 4        | addition      | sse42  | 0.072031 | 0.003754   | 19.19x  | -94.79%        |
| 1280x720  | uint8   | 4        | addition      | avx2   | 0.072031 | 0.002743   | 26.26x  | -96.19%        |
| 1280x720  | uint8   | 4        | darken_only   | scalar | 0.074407 | 0.028628   | 2.60x   | -61.53%        |
| 1280x720  | uint8   | 4        | darken_only   | sse42  | 0.074407 | 0.002808   | 26.50x  | -96.23%        |
| 1280x720  | uint8   | 4        | darken_only   | avx2   | 0.074407 | 0.002602   | 28.60x  | -96.50%        |
| 1280x720  | uint8   | 4        | multiply      | scalar | 0.075182 | 0.025287   | 2.97x   | -66.37%        |
| 1280x720  | uint8   | 4        | multiply      | sse42  | 0.075182 | 0.002906   | 25.88x  | -96.14%        |
| 1280x720  | uint8   | 4        | multiply      | avx2   | 0.075182 | 0.002651   | 28.36x  | -96.47%        |
| 1280x720  | uint8   | 4        | hard_light    | scalar | 0.107177 | 0.039971   | 2.68x   | -62.71%        |
| 1280x720  | uint8   | 4        | hard_light    | sse42  | 0.107177 | 0.003324   | 32.25x  | -96.90%        |
| 1280x720  | uint8   | 4        | hard_light    | avx2   | 0.107177 | 0.003017   | 35.52x  | -97.19%        |
| 1280x720  | uint8   | 4        | difference    | scalar | 0.098911 | 0.024109   | 4.10x   | -75.63%        |
| 1280x720  | uint8   | 4        | difference    | sse42  | 0.098911 | 0.002812   | 35.17x  | -97.16%        |
| 1280x720  | uint8   | 4        | difference    | avx2   | 0.098911 | 0.002587   | 38.23x  | -97.38%        |
| 1280x720  | uint8   | 4        | subtract      | scalar | 0.071039 | 0.022953   | 3.09x   | -67.69%        |
| 1280x720  | uint8   | 4        | subtract      | sse42  | 0.071039 | 0.003646   | 19.48x  | -94.87%        |
| 1280x720  | uint8   | 4        | subtract      | avx2   | 0.071039 | 0.002709   | 26.22x  | -96.19%        |
| 1280x720  | uint8   | 4        | grain_extract | scalar | 0.075866 | 0.029336   | 2.59x   | -61.33%        |
| 1280x720  | uint8   | 4        | grain_extract | sse42  | 0.075866 | 0.003018   | 25.14x  | -96.02%        |
| 1280x720  | uint8   | 4        | grain_extract | avx2   | 0.075866 | 0.002660   | 28.52x  | -96.49%        |
| 1280x720  | uint8   | 4        | grain_merge   | scalar | 0.074690 | 0.028921   | 2.58x   | -61.28%        |
| 1280x720  | uint8   | 4        | grain_merge   | sse42  | 0.074690 | 0.002895   | 25.80x  | -96.12%        |
| 1280x720  | uint8   | 4        | grain_merge   | avx2   | 0.074690 | 0.002593   | 28.80x  | -96.53%        |
| 1280x720  | uint8   | 4        | divide        | scalar | 0.078207 | 0.025094   | 3.12x   | -67.91%        |
| 1280x720  | uint8   | 4        | divide        | sse42  | 0.078207 | 0.002959   | 26.43x  | -96.22%        |
| 1280x720  | uint8   | 4        | divide        | avx2   | 0.078207 | 0.002614   | 29.92x  | -96.66%        |
| 1280x720  | uint8   | 4        | overlay       | scalar | 0.100081 | 0.038831   | 2.58x   | -61.20%        |
| 1280x720  | uint8   | 4        | overlay       | sse42  | 0.100081 | 0.003163   | 31.64x  | -96.84%        |
| 1280x720  | uint8   | 4        | overlay       | avx2   | 0.100081 | 0.002658   | 37.66x  | -97.34%        |
| 1280x720  | float32 | 3        | normal        | scalar | 0.076536 | 0.007223   | 10.60x  | -90.56%        |
| 1280x720  | float32 | 3        | normal        | sse42  | 0.076536 | 0.003099   | 24.70x  | -95.95%        |
| 1280x720  | float32 | 3        | normal        | avx2   | 0.076536 | 0.002340   | 32.70x  | -96.94%        |
| 1280x720  | float32 | 3        | soft_light    | scalar | 0.107791 | 0.010464   | 10.30x  | -90.29%        |
| 1280x720  | float32 | 3        | soft_light    | sse42  | 0.107791 | 0.003605   | 29.90x  | -96.66%        |
| 1280x720  | float32 | 3        | soft_light    | avx2   | 0.107791 | 0.002893   | 37.26x  | -97.32%        |
| 1280x720  | float32 | 3        | lighten_only  | scalar | 0.086114 | 0.011760   | 7.32x   | -86.34%        |
| 1280x720  | float32 | 3        | lighten_only  | sse42  | 0.086114 | 0.003010   | 28.61x  | -96.50%        |
| 1280x720  | float32 | 3        | lighten_only  | avx2   | 0.086114 | 0.002634   | 32.70x  | -96.94%        |
| 1280x720  | float32 | 3        | screen        | scalar | 0.089715 | 0.009097   | 9.86x   | -89.86%        |
| 1280x720  | float32 | 3        | screen        | sse42  | 0.089715 | 0.003437   | 26.10x  | -96.17%        |
| 1280x720  | float32 | 3        | screen        | avx2   | 0.089715 | 0.002899   | 30.94x  | -96.77%        |
| 1280x720  | float32 | 3        | dodge         | scalar | 0.092291 | 0.010629   | 8.68x   | -88.48%        |
| 1280x720  | float32 | 3        | dodge         | sse42  | 0.092291 | 0.003862   | 23.90x  | -95.82%        |
| 1280x720  | float32 | 3        | dodge         | avx2   | 0.092291 | 0.003127   | 29.51x  | -96.61%        |
| 1280x720  | float32 | 3        | addition      | scalar | 0.084877 | 0.023203   | 3.66x   | -72.66%        |
| 1280x720  | float32 | 3        | addition      | sse42  | 0.084877 | 0.003536   | 24.01x  | -95.83%        |
| 1280x720  | float32 | 3        | addition      | avx2   | 0.084877 | 0.002913   | 29.14x  | -96.57%        |
| 1280x720  | float32 | 3        | darken_only   | scalar | 0.085750 | 0.011694   | 7.33x   | -86.36%        |
| 1280x720  | float32 | 3        | darken_only   | sse42  | 0.085750 | 0.003097   | 27.69x  | -96.39%        |
| 1280x720  | float32 | 3        | darken_only   | avx2   | 0.085750 | 0.002642   | 32.46x  | -96.92%        |
| 1280x720  | float32 | 3        | multiply      | scalar | 0.086044 | 0.009047   | 9.51x   | -89.49%        |
| 1280x720  | float32 | 3        | multiply      | sse42  | 0.086044 | 0.003065   | 28.07x  | -96.44%        |
| 1280x720  | float32 | 3        | multiply      | avx2   | 0.086044 | 0.002676   | 32.15x  | -96.89%        |
| 1280x720  | float32 | 3        | hard_light    | scalar | 0.118248 | 0.027093   | 4.36x   | -77.09%        |
| 1280x720  | float32 | 3        | hard_light    | sse42  | 0.118248 | 0.003748   | 31.55x  | -96.83%        |
| 1280x720  | float32 | 3        | hard_light    | avx2   | 0.118248 | 0.003116   | 37.95x  | -97.37%        |
| 1280x720  | float32 | 3        | difference    | scalar | 0.111560 | 0.008941   | 12.48x  | -91.99%        |
| 1280x720  | float32 | 3        | difference    | sse42  | 0.111560 | 0.003304   | 33.77x  | -97.04%        |
| 1280x720  | float32 | 3        | difference    | avx2   | 0.111560 | 0.002786   | 40.05x  | -97.50%        |
| 1280x720  | float32 | 3        | subtract      | scalar | 0.082445 | 0.010601   | 7.78x   | -87.14%        |
| 1280x720  | float32 | 3        | subtract      | sse42  | 0.082445 | 0.003466   | 23.78x  | -95.80%        |
| 1280x720  | float32 | 3        | subtract      | avx2   | 0.082445 | 0.003006   | 27.42x  | -96.35%        |
| 1280x720  | float32 | 3        | grain_extract | scalar | 0.087231 | 0.015166   | 5.75x   | -82.61%        |
| 1280x720  | float32 | 3        | grain_extract | sse42  | 0.087231 | 0.003363   | 25.94x  | -96.15%        |
| 1280x720  | float32 | 3        | grain_extract | avx2   | 0.087231 | 0.002807   | 31.08x  | -96.78%        |
| 1280x720  | float32 | 3        | grain_merge   | scalar | 0.086686 | 0.015448   | 5.61x   | -82.18%        |
| 1280x720  | float32 | 3        | grain_merge   | sse42  | 0.086686 | 0.003274   | 26.48x  | -96.22%        |
| 1280x720  | float32 | 3        | grain_merge   | avx2   | 0.086686 | 0.002751   | 31.51x  | -96.83%        |
| 1280x720  | float32 | 3        | divide        | scalar | 0.090089 | 0.010217   | 8.82x   | -88.66%        |
| 1280x720  | float32 | 3        | divide        | sse42  | 0.090089 | 0.003434   | 26.23x  | -96.19%        |
| 1280x720  | float32 | 3        | divide        | avx2   | 0.090089 | 0.002969   | 30.34x  | -96.70%        |
| 1280x720  | float32 | 3        | overlay       | scalar | 0.111735 | 0.025442   | 4.39x   | -77.23%        |
| 1280x720  | float32 | 3        | overlay       | sse42  | 0.111735 | 0.003358   | 33.28x  | -97.00%        |
| 1280x720  | float32 | 3        | overlay       | avx2   | 0.111735 | 0.002729   | 40.95x  | -97.56%        |
| 1280x720  | float32 | 4        | normal        | scalar | 0.058059 | 0.008943   | 6.49x   | -84.60%        |
| 1280x720  | float32 | 4        | normal        | sse42  | 0.058059 | 0.002933   | 19.80x  | -94.95%        |
| 1280x720  | float32 | 4        | normal        | avx2   | 0.058059 | 0.002602   | 22.31x  | -95.52%        |
| 1280x720  | float32 | 4        | soft_light    | scalar | 0.089689 | 0.011941   | 7.51x   | -86.69%        |
| 1280x720  | float32 | 4        | soft_light    | sse42  | 0.089689 | 0.003483   | 25.75x  | -96.12%        |
| 1280x720  | float32 | 4        | soft_light    | avx2   | 0.089689 | 0.002914   | 30.78x  | -96.75%        |
| 1280x720  | float32 | 4        | lighten_only  | scalar | 0.066074 | 0.013797   | 4.79x   | -79.12%        |
| 1280x720  | float32 | 4        | lighten_only  | sse42  | 0.066074 | 0.003283   | 20.13x  | -95.03%        |
| 1280x720  | float32 | 4        | lighten_only  | avx2   | 0.066074 | 0.002995   | 22.06x  | -95.47%        |
| 1280x720  | float32 | 4        | screen        | scalar | 0.069192 | 0.011944   | 5.79x   | -82.74%        |
| 1280x720  | float32 | 4        | screen        | sse42  | 0.069192 | 0.003484   | 19.86x  | -94.96%        |
| 1280x720  | float32 | 4        | screen        | avx2   | 0.069192 | 0.002986   | 23.17x  | -95.68%        |
| 1280x720  | float32 | 4        | dodge         | scalar | 0.070615 | 0.012646   | 5.58x   | -82.09%        |
| 1280x720  | float32 | 4        | dodge         | sse42  | 0.070615 | 0.003551   | 19.88x  | -94.97%        |
| 1280x720  | float32 | 4        | dodge         | avx2   | 0.070615 | 0.003555   | 19.86x  | -94.97%        |
| 1280x720  | float32 | 4        | addition      | scalar | 0.065451 | 0.020401   | 3.21x   | -68.83%        |
| 1280x720  | float32 | 4        | addition      | sse42  | 0.065451 | 0.003363   | 19.46x  | -94.86%        |
| 1280x720  | float32 | 4        | addition      | avx2   | 0.065451 | 0.002948   | 22.20x  | -95.50%        |
| 1280x720  | float32 | 4        | darken_only   | scalar | 0.067486 | 0.014104   | 4.78x   | -79.10%        |
| 1280x720  | float32 | 4        | darken_only   | sse42  | 0.067486 | 0.003254   | 20.74x  | -95.18%        |
| 1280x720  | float32 | 4        | darken_only   | avx2   | 0.067486 | 0.003279   | 20.58x  | -95.14%        |
| 1280x720  | float32 | 4        | multiply      | scalar | 0.068519 | 0.011885   | 5.77x   | -82.65%        |
| 1280x720  | float32 | 4        | multiply      | sse42  | 0.068519 | 0.003431   | 19.97x  | -94.99%        |
| 1280x720  | float32 | 4        | multiply      | avx2   | 0.068519 | 0.003160   | 21.68x  | -95.39%        |
| 1280x720  | float32 | 4        | hard_light    | scalar | 0.099259 | 0.028011   | 3.54x   | -71.78%        |
| 1280x720  | float32 | 4        | hard_light    | sse42  | 0.099259 | 0.003604   | 27.54x  | -96.37%        |
| 1280x720  | float32 | 4        | hard_light    | avx2   | 0.099259 | 0.002907   | 34.14x  | -97.07%        |
| 1280x720  | float32 | 4        | difference    | scalar | 0.095781 | 0.011613   | 8.25x   | -87.88%        |
| 1280x720  | float32 | 4        | difference    | sse42  | 0.095781 | 0.003165   | 30.27x  | -96.70%        |
| 1280x720  | float32 | 4        | difference    | avx2   | 0.095781 | 0.003090   | 31.00x  | -96.77%        |
| 1280x720  | float32 | 4        | subtract      | scalar | 0.065555 | 0.014239   | 4.60x   | -78.28%        |
| 1280x720  | float32 | 4        | subtract      | sse42  | 0.065555 | 0.003365   | 19.48x  | -94.87%        |
| 1280x720  | float32 | 4        | subtract      | avx2   | 0.065555 | 0.002979   | 22.01x  | -95.46%        |
| 1280x720  | float32 | 4        | grain_extract | scalar | 0.069529 | 0.016615   | 4.18x   | -76.10%        |
| 1280x720  | float32 | 4        | grain_extract | sse42  | 0.069529 | 0.003433   | 20.25x  | -95.06%        |
| 1280x720  | float32 | 4        | grain_extract | avx2   | 0.069529 | 0.003140   | 22.14x  | -95.48%        |
| 1280x720  | float32 | 4        | grain_merge   | scalar | 0.069784 | 0.017262   | 4.04x   | -75.26%        |
| 1280x720  | float32 | 4        | grain_merge   | sse42  | 0.069784 | 0.003428   | 20.36x  | -95.09%        |
| 1280x720  | float32 | 4        | grain_merge   | avx2   | 0.069784 | 0.003101   | 22.50x  | -95.56%        |
| 1280x720  | float32 | 4        | divide        | scalar | 0.072254 | 0.012131   | 5.96x   | -83.21%        |
| 1280x720  | float32 | 4        | divide        | sse42  | 0.072254 | 0.003539   | 20.42x  | -95.10%        |
| 1280x720  | float32 | 4        | divide        | avx2   | 0.072254 | 0.002926   | 24.69x  | -95.95%        |
| 1280x720  | float32 | 4        | overlay       | scalar | 0.094784 | 0.026466   | 3.58x   | -72.08%        |
| 1280x720  | float32 | 4        | overlay       | sse42  | 0.094784 | 0.003915   | 24.21x  | -95.87%        |
| 1280x720  | float32 | 4        | overlay       | avx2   | 0.094784 | 0.002997   | 31.62x  | -96.84%        |
| 1920x1080 | uint8   | 3        | normal        | scalar | 0.190987 | 0.054713   | 3.49x   | -71.35%        |
| 1920x1080 | uint8   | 3        | normal        | sse42  | 0.190987 | 0.047053   | 4.06x   | -75.36%        |
| 1920x1080 | uint8   | 3        | normal        | avx2   | 0.190987 | 0.047536   | 4.02x   | -75.11%        |
| 1920x1080 | uint8   | 3        | soft_light    | scalar | 0.248863 | 0.063987   | 3.89x   | -74.29%        |
| 1920x1080 | uint8   | 3        | soft_light    | sse42  | 0.248863 | 0.047609   | 5.23x   | -80.87%        |
| 1920x1080 | uint8   | 3        | soft_light    | avx2   | 0.248863 | 0.048257   | 5.16x   | -80.61%        |
| 1920x1080 | uint8   | 3        | lighten_only  | scalar | 0.196779 | 0.072298   | 2.72x   | -63.26%        |
| 1920x1080 | uint8   | 3        | lighten_only  | sse42  | 0.196779 | 0.046195   | 4.26x   | -76.52%        |
| 1920x1080 | uint8   | 3        | lighten_only  | avx2   | 0.196779 | 0.046603   | 4.22x   | -76.32%        |
| 1920x1080 | uint8   | 3        | screen        | scalar | 0.205193 | 0.063532   | 3.23x   | -69.04%        |
| 1920x1080 | uint8   | 3        | screen        | sse42  | 0.205193 | 0.046500   | 4.41x   | -77.34%        |
| 1920x1080 | uint8   | 3        | screen        | avx2   | 0.205193 | 0.049557   | 4.14x   | -75.85%        |
| 1920x1080 | uint8   | 3        | dodge         | scalar | 0.206178 | 0.064389   | 3.20x   | -68.77%        |
| 1920x1080 | uint8   | 3        | dodge         | sse42  | 0.206178 | 0.048251   | 4.27x   | -76.60%        |
| 1920x1080 | uint8   | 3        | dodge         | avx2   | 0.206178 | 0.048511   | 4.25x   | -76.47%        |
| 1920x1080 | uint8   | 3        | addition      | scalar | 0.201845 | 0.086365   | 2.34x   | -57.21%        |
| 1920x1080 | uint8   | 3        | addition      | sse42  | 0.201845 | 0.046502   | 4.34x   | -76.96%        |
| 1920x1080 | uint8   | 3        | addition      | avx2   | 0.201845 | 0.048467   | 4.16x   | -75.99%        |
| 1920x1080 | uint8   | 3        | darken_only   | scalar | 0.198058 | 0.070410   | 2.81x   | -64.45%        |
| 1920x1080 | uint8   | 3        | darken_only   | sse42  | 0.198058 | 0.045949   | 4.31x   | -76.80%        |
| 1920x1080 | uint8   | 3        | darken_only   | avx2   | 0.198058 | 0.047113   | 4.20x   | -76.21%        |
| 1920x1080 | uint8   | 3        | multiply      | scalar | 0.204659 | 0.063318   | 3.23x   | -69.06%        |
| 1920x1080 | uint8   | 3        | multiply      | sse42  | 0.204659 | 0.046390   | 4.41x   | -77.33%        |
| 1920x1080 | uint8   | 3        | multiply      | avx2   | 0.204659 | 0.047660   | 4.29x   | -76.71%        |
| 1920x1080 | uint8   | 3        | hard_light    | scalar | 0.264774 | 0.101466   | 2.61x   | -61.68%        |
| 1920x1080 | uint8   | 3        | hard_light    | sse42  | 0.264774 | 0.047971   | 5.52x   | -81.88%        |
| 1920x1080 | uint8   | 3        | hard_light    | avx2   | 0.264774 | 0.047966   | 5.52x   | -81.88%        |
| 1920x1080 | uint8   | 3        | difference    | scalar | 0.255235 | 0.064096   | 3.98x   | -74.89%        |
| 1920x1080 | uint8   | 3        | difference    | sse42  | 0.255235 | 0.046094   | 5.54x   | -81.94%        |
| 1920x1080 | uint8   | 3        | difference    | avx2   | 0.255235 | 0.048097   | 5.31x   | -81.16%        |
| 1920x1080 | uint8   | 3        | subtract      | scalar | 0.199959 | 0.060904   | 3.28x   | -69.54%        |
| 1920x1080 | uint8   | 3        | subtract      | sse42  | 0.199959 | 0.048145   | 4.15x   | -75.92%        |
| 1920x1080 | uint8   | 3        | subtract      | avx2   | 0.199959 | 0.048710   | 4.11x   | -75.64%        |
| 1920x1080 | uint8   | 3        | grain_extract | scalar | 0.200893 | 0.074361   | 2.70x   | -62.98%        |
| 1920x1080 | uint8   | 3        | grain_extract | sse42  | 0.200893 | 0.048145   | 4.17x   | -76.03%        |
| 1920x1080 | uint8   | 3        | grain_extract | avx2   | 0.200893 | 0.047755   | 4.21x   | -76.23%        |
| 1920x1080 | uint8   | 3        | grain_merge   | scalar | 0.198901 | 0.076012   | 2.62x   | -61.78%        |
| 1920x1080 | uint8   | 3        | grain_merge   | sse42  | 0.198901 | 0.049351   | 4.03x   | -75.19%        |
| 1920x1080 | uint8   | 3        | grain_merge   | avx2   | 0.198901 | 0.047928   | 4.15x   | -75.90%        |
| 1920x1080 | uint8   | 3        | divide        | scalar | 0.207571 | 0.064686   | 3.21x   | -68.84%        |
| 1920x1080 | uint8   | 3        | divide        | sse42  | 0.207571 | 0.048180   | 4.31x   | -76.79%        |
| 1920x1080 | uint8   | 3        | divide        | avx2   | 0.207571 | 0.050124   | 4.14x   | -75.85%        |
| 1920x1080 | uint8   | 3        | overlay       | scalar | 0.257792 | 0.099996   | 2.58x   | -61.21%        |
| 1920x1080 | uint8   | 3        | overlay       | sse42  | 0.257792 | 0.048026   | 5.37x   | -81.37%        |
| 1920x1080 | uint8   | 3        | overlay       | avx2   | 0.257792 | 0.049393   | 5.22x   | -80.84%        |
| 1920x1080 | uint8   | 4        | normal        | scalar | 0.143007 | 0.046455   | 3.08x   | -67.52%        |
| 1920x1080 | uint8   | 4        | normal        | sse42  | 0.143007 | 0.005945   | 24.05x  | -95.84%        |
| 1920x1080 | uint8   | 4        | normal        | avx2   | 0.143007 | 0.005323   | 26.86x  | -96.28%        |
| 1920x1080 | uint8   | 4        | soft_light    | scalar | 0.202951 | 0.060982   | 3.33x   | -69.95%        |
| 1920x1080 | uint8   | 4        | soft_light    | sse42  | 0.202951 | 0.006908   | 29.38x  | -96.60%        |
| 1920x1080 | uint8   | 4        | soft_light    | avx2   | 0.202951 | 0.005957   | 34.07x  | -97.06%        |
| 1920x1080 | uint8   | 4        | lighten_only  | scalar | 0.150408 | 0.065550   | 2.29x   | -56.42%        |
| 1920x1080 | uint8   | 4        | lighten_only  | sse42  | 0.150408 | 0.006225   | 24.16x  | -95.86%        |
| 1920x1080 | uint8   | 4        | lighten_only  | avx2   | 0.150408 | 0.005950   | 25.28x  | -96.04%        |
| 1920x1080 | uint8   | 4        | screen        | scalar | 0.155913 | 0.060568   | 2.57x   | -61.15%        |
| 1920x1080 | uint8   | 4        | screen        | sse42  | 0.155913 | 0.006481   | 24.06x  | -95.84%        |
| 1920x1080 | uint8   | 4        | screen        | avx2   | 0.155913 | 0.005896   | 26.44x  | -96.22%        |
| 1920x1080 | uint8   | 4        | dodge         | scalar | 0.156377 | 0.061477   | 2.54x   | -60.69%        |
| 1920x1080 | uint8   | 4        | dodge         | sse42  | 0.156377 | 0.007464   | 20.95x  | -95.23%        |
| 1920x1080 | uint8   | 4        | dodge         | avx2   | 0.156377 | 0.005942   | 26.32x  | -96.20%        |
| 1920x1080 | uint8   | 4        | addition      | scalar | 0.151837 | 0.072932   | 2.08x   | -51.97%        |
| 1920x1080 | uint8   | 4        | addition      | sse42  | 0.151837 | 0.008440   | 17.99x  | -94.44%        |
| 1920x1080 | uint8   | 4        | addition      | avx2   | 0.151837 | 0.006207   | 24.46x  | -95.91%        |
| 1920x1080 | uint8   | 4        | darken_only   | scalar | 0.151614 | 0.065099   | 2.33x   | -57.06%        |
| 1920x1080 | uint8   | 4        | darken_only   | sse42  | 0.151614 | 0.006300   | 24.07x  | -95.84%        |
| 1920x1080 | uint8   | 4        | darken_only   | avx2   | 0.151614 | 0.005929   | 25.57x  | -96.09%        |
| 1920x1080 | uint8   | 4        | multiply      | scalar | 0.154562 | 0.059275   | 2.61x   | -61.65%        |
| 1920x1080 | uint8   | 4        | multiply      | sse42  | 0.154562 | 0.006261   | 24.69x  | -95.95%        |
| 1920x1080 | uint8   | 4        | multiply      | avx2   | 0.154562 | 0.005847   | 26.43x  | -96.22%        |
| 1920x1080 | uint8   | 4        | hard_light    | scalar | 0.221960 | 0.094044   | 2.36x   | -57.63%        |
| 1920x1080 | uint8   | 4        | hard_light    | sse42  | 0.221960 | 0.007634   | 29.07x  | -96.56%        |
| 1920x1080 | uint8   | 4        | hard_light    | avx2   | 0.221960 | 0.005937   | 37.38x  | -97.33%        |
| 1920x1080 | uint8   | 4        | difference    | scalar | 0.211866 | 0.060693   | 3.49x   | -71.35%        |
| 1920x1080 | uint8   | 4        | difference    | sse42  | 0.211866 | 0.006738   | 31.44x  | -96.82%        |
| 1920x1080 | uint8   | 4        | difference    | avx2   | 0.211866 | 0.005820   | 36.40x  | -97.25%        |
| 1920x1080 | uint8   | 4        | subtract      | scalar | 0.154057 | 0.054912   | 2.81x   | -64.36%        |
| 1920x1080 | uint8   | 4        | subtract      | sse42  | 0.154057 | 0.008475   | 18.18x  | -94.50%        |
| 1920x1080 | uint8   | 4        | subtract      | avx2   | 0.154057 | 0.006282   | 24.53x  | -95.92%        |
| 1920x1080 | uint8   | 4        | grain_extract | scalar | 0.162264 | 0.069214   | 2.34x   | -57.35%        |
| 1920x1080 | uint8   | 4        | grain_extract | sse42  | 0.162264 | 0.006738   | 24.08x  | -95.85%        |
| 1920x1080 | uint8   | 4        | grain_extract | avx2   | 0.162264 | 0.005952   | 27.26x  | -96.33%        |
| 1920x1080 | uint8   | 4        | grain_merge   | scalar | 0.152417 | 0.070552   | 2.16x   | -53.71%        |
| 1920x1080 | uint8   | 4        | grain_merge   | sse42  | 0.152417 | 0.006703   | 22.74x  | -95.60%        |
| 1920x1080 | uint8   | 4        | grain_merge   | avx2   | 0.152417 | 0.006076   | 25.08x  | -96.01%        |
| 1920x1080 | uint8   | 4        | divide        | scalar | 0.160230 | 0.062026   | 2.58x   | -61.29%        |
| 1920x1080 | uint8   | 4        | divide        | sse42  | 0.160230 | 0.006850   | 23.39x  | -95.73%        |
| 1920x1080 | uint8   | 4        | divide        | avx2   | 0.160230 | 0.005911   | 27.11x  | -96.31%        |
| 1920x1080 | uint8   | 4        | overlay       | scalar | 0.207172 | 0.091159   | 2.27x   | -56.00%        |
| 1920x1080 | uint8   | 4        | overlay       | sse42  | 0.207172 | 0.007255   | 28.56x  | -96.50%        |
| 1920x1080 | uint8   | 4        | overlay       | avx2   | 0.207172 | 0.006090   | 34.02x  | -97.06%        |
| 1920x1080 | float32 | 3        | normal        | scalar | 0.169568 | 0.019362   | 8.76x   | -88.58%        |
| 1920x1080 | float32 | 3        | normal        | sse42  | 0.169568 | 0.007501   | 22.61x  | -95.58%        |
| 1920x1080 | float32 | 3        | normal        | avx2   | 0.169568 | 0.006107   | 27.76x  | -96.40%        |
| 1920x1080 | float32 | 3        | soft_light    | scalar | 0.222333 | 0.025726   | 8.64x   | -88.43%        |
| 1920x1080 | float32 | 3        | soft_light    | sse42  | 0.222333 | 0.008100   | 27.45x  | -96.36%        |
| 1920x1080 | float32 | 3        | soft_light    | avx2   | 0.222333 | 0.006769   | 32.85x  | -96.96%        |
| 1920x1080 | float32 | 3        | lighten_only  | scalar | 0.172307 | 0.031782   | 5.42x   | -81.55%        |
| 1920x1080 | float32 | 3        | lighten_only  | sse42  | 0.172307 | 0.007656   | 22.51x  | -95.56%        |
| 1920x1080 | float32 | 3        | lighten_only  | avx2   | 0.172307 | 0.006171   | 27.92x  | -96.42%        |
| 1920x1080 | float32 | 3        | screen        | scalar | 0.178311 | 0.023730   | 7.51x   | -86.69%        |
| 1920x1080 | float32 | 3        | screen        | sse42  | 0.178311 | 0.007813   | 22.82x  | -95.62%        |
| 1920x1080 | float32 | 3        | screen        | avx2   | 0.178311 | 0.006602   | 27.01x  | -96.30%        |
| 1920x1080 | float32 | 3        | dodge         | scalar | 0.182679 | 0.025436   | 7.18x   | -86.08%        |
| 1920x1080 | float32 | 3        | dodge         | sse42  | 0.182679 | 0.008024   | 22.77x  | -95.61%        |
| 1920x1080 | float32 | 3        | dodge         | avx2   | 0.182679 | 0.006554   | 27.87x  | -96.41%        |
| 1920x1080 | float32 | 3        | addition      | scalar | 0.171504 | 0.054618   | 3.14x   | -68.15%        |
| 1920x1080 | float32 | 3        | addition      | sse42  | 0.171504 | 0.007603   | 22.56x  | -95.57%        |
| 1920x1080 | float32 | 3        | addition      | avx2   | 0.171504 | 0.006468   | 26.52x  | -96.23%        |
| 1920x1080 | float32 | 3        | darken_only   | scalar | 0.167175 | 0.028922   | 5.78x   | -82.70%        |
| 1920x1080 | float32 | 3        | darken_only   | sse42  | 0.167175 | 0.006787   | 24.63x  | -95.94%        |
| 1920x1080 | float32 | 3        | darken_only   | avx2   | 0.167175 | 0.005873   | 28.46x  | -96.49%        |
| 1920x1080 | float32 | 3        | multiply      | scalar | 0.171109 | 0.022683   | 7.54x   | -86.74%        |
| 1920x1080 | float32 | 3        | multiply      | sse42  | 0.171109 | 0.006970   | 24.55x  | -95.93%        |
| 1920x1080 | float32 | 3        | multiply      | avx2   | 0.171109 | 0.006085   | 28.12x  | -96.44%        |
| 1920x1080 | float32 | 3        | hard_light    | scalar | 0.239230 | 0.063516   | 3.77x   | -73.45%        |
| 1920x1080 | float32 | 3        | hard_light    | sse42  | 0.239230 | 0.008574   | 27.90x  | -96.42%        |
| 1920x1080 | float32 | 3        | hard_light    | avx2   | 0.239230 | 0.006778   | 35.30x  | -97.17%        |
| 1920x1080 | float32 | 3        | difference    | scalar | 0.229715 | 0.023458   | 9.79x   | -89.79%        |
| 1920x1080 | float32 | 3        | difference    | sse42  | 0.229715 | 0.007446   | 30.85x  | -96.76%        |
| 1920x1080 | float32 | 3        | difference    | avx2   | 0.229715 | 0.006908   | 33.25x  | -96.99%        |
| 1920x1080 | float32 | 3        | subtract      | scalar | 0.174943 | 0.026895   | 6.50x   | -84.63%        |
| 1920x1080 | float32 | 3        | subtract      | sse42  | 0.174943 | 0.007924   | 22.08x  | -95.47%        |
| 1920x1080 | float32 | 3        | subtract      | avx2   | 0.174943 | 0.006338   | 27.60x  | -96.38%        |
| 1920x1080 | float32 | 3        | grain_extract | scalar | 0.177617 | 0.037067   | 4.79x   | -79.13%        |
| 1920x1080 | float32 | 3        | grain_extract | sse42  | 0.177617 | 0.007212   | 24.63x  | -95.94%        |
| 1920x1080 | float32 | 3        | grain_extract | avx2   | 0.177617 | 0.006745   | 26.33x  | -96.20%        |
| 1920x1080 | float32 | 3        | grain_merge   | scalar | 0.171976 | 0.036905   | 4.66x   | -78.54%        |
| 1920x1080 | float32 | 3        | grain_merge   | sse42  | 0.171976 | 0.007454   | 23.07x  | -95.67%        |
| 1920x1080 | float32 | 3        | grain_merge   | avx2   | 0.171976 | 0.006096   | 28.21x  | -96.46%        |
| 1920x1080 | float32 | 3        | divide        | scalar | 0.180444 | 0.025449   | 7.09x   | -85.90%        |
| 1920x1080 | float32 | 3        | divide        | sse42  | 0.180444 | 0.009357   | 19.28x  | -94.81%        |
| 1920x1080 | float32 | 3        | divide        | avx2   | 0.180444 | 0.006625   | 27.24x  | -96.33%        |
| 1920x1080 | float32 | 3        | overlay       | scalar | 0.232932 | 0.059387   | 3.92x   | -74.50%        |
| 1920x1080 | float32 | 3        | overlay       | sse42  | 0.232932 | 0.007598   | 30.66x  | -96.74%        |
| 1920x1080 | float32 | 3        | overlay       | avx2   | 0.232932 | 0.006439   | 36.18x  | -97.24%        |
| 1920x1080 | float32 | 4        | normal        | scalar | 0.122869 | 0.019973   | 6.15x   | -83.74%        |
| 1920x1080 | float32 | 4        | normal        | sse42  | 0.122869 | 0.006429   | 19.11x  | -94.77%        |
| 1920x1080 | float32 | 4        | normal        | avx2   | 0.122869 | 0.005371   | 22.88x  | -95.63%        |
| 1920x1080 | float32 | 4        | soft_light    | scalar | 0.180907 | 0.030980   | 5.84x   | -82.88%        |
| 1920x1080 | float32 | 4        | soft_light    | sse42  | 0.180907 | 0.007786   | 23.24x  | -95.70%        |
| 1920x1080 | float32 | 4        | soft_light    | avx2   | 0.180907 | 0.006673   | 27.11x  | -96.31%        |
| 1920x1080 | float32 | 4        | lighten_only  | scalar | 0.129155 | 0.030885   | 4.18x   | -76.09%        |
| 1920x1080 | float32 | 4        | lighten_only  | sse42  | 0.129155 | 0.006815   | 18.95x  | -94.72%        |
| 1920x1080 | float32 | 4        | lighten_only  | avx2   | 0.129155 | 0.007091   | 18.21x  | -94.51%        |
| 1920x1080 | float32 | 4        | screen        | scalar | 0.138339 | 0.027872   | 4.96x   | -79.85%        |
| 1920x1080 | float32 | 4        | screen        | sse42  | 0.138339 | 0.008027   | 17.23x  | -94.20%        |
| 1920x1080 | float32 | 4        | screen        | avx2   | 0.138339 | 0.007206   | 19.20x  | -94.79%        |
| 1920x1080 | float32 | 4        | dodge         | scalar | 0.138465 | 0.027971   | 4.95x   | -79.80%        |
| 1920x1080 | float32 | 4        | dodge         | sse42  | 0.138465 | 0.007762   | 17.84x  | -94.39%        |
| 1920x1080 | float32 | 4        | dodge         | avx2   | 0.138465 | 0.006155   | 22.50x  | -95.55%        |
| 1920x1080 | float32 | 4        | addition      | scalar | 0.130622 | 0.046308   | 2.82x   | -64.55%        |
| 1920x1080 | float32 | 4        | addition      | sse42  | 0.130622 | 0.007791   | 16.76x  | -94.04%        |
| 1920x1080 | float32 | 4        | addition      | avx2   | 0.130622 | 0.006340   | 20.60x  | -95.15%        |
| 1920x1080 | float32 | 4        | darken_only   | scalar | 0.130236 | 0.031522   | 4.13x   | -75.80%        |
| 1920x1080 | float32 | 4        | darken_only   | sse42  | 0.130236 | 0.006675   | 19.51x  | -94.87%        |
| 1920x1080 | float32 | 4        | darken_only   | avx2   | 0.130236 | 0.007286   | 17.88x  | -94.41%        |
| 1920x1080 | float32 | 4        | multiply      | scalar | 0.130259 | 0.025052   | 5.20x   | -80.77%        |
| 1920x1080 | float32 | 4        | multiply      | sse42  | 0.130259 | 0.007220   | 18.04x  | -94.46%        |
| 1920x1080 | float32 | 4        | multiply      | avx2   | 0.130259 | 0.006634   | 19.64x  | -94.91%        |
| 1920x1080 | float32 | 4        | hard_light    | scalar | 0.199235 | 0.062389   | 3.19x   | -68.69%        |
| 1920x1080 | float32 | 4        | hard_light    | sse42  | 0.199235 | 0.008099   | 24.60x  | -95.93%        |
| 1920x1080 | float32 | 4        | hard_light    | avx2   | 0.199235 | 0.006456   | 30.86x  | -96.76%        |
| 1920x1080 | float32 | 4        | difference    | scalar | 0.191217 | 0.024945   | 7.67x   | -86.95%        |
| 1920x1080 | float32 | 4        | difference    | sse42  | 0.191217 | 0.006945   | 27.53x  | -96.37%        |
| 1920x1080 | float32 | 4        | difference    | avx2   | 0.191217 | 0.006558   | 29.16x  | -96.57%        |
| 1920x1080 | float32 | 4        | subtract      | scalar | 0.132456 | 0.030683   | 4.32x   | -76.84%        |
| 1920x1080 | float32 | 4        | subtract      | sse42  | 0.132456 | 0.007866   | 16.84x  | -94.06%        |
| 1920x1080 | float32 | 4        | subtract      | avx2   | 0.132456 | 0.006672   | 19.85x  | -94.96%        |
| 1920x1080 | float32 | 4        | grain_extract | scalar | 0.133471 | 0.036957   | 3.61x   | -72.31%        |
| 1920x1080 | float32 | 4        | grain_extract | sse42  | 0.133471 | 0.007599   | 17.56x  | -94.31%        |
| 1920x1080 | float32 | 4        | grain_extract | avx2   | 0.133471 | 0.006774   | 19.70x  | -94.92%        |
| 1920x1080 | float32 | 4        | grain_merge   | scalar | 0.132006 | 0.037967   | 3.48x   | -71.24%        |
| 1920x1080 | float32 | 4        | grain_merge   | sse42  | 0.132006 | 0.007620   | 17.32x  | -94.23%        |
| 1920x1080 | float32 | 4        | grain_merge   | avx2   | 0.132006 | 0.007108   | 18.57x  | -94.62%        |
| 1920x1080 | float32 | 4        | divide        | scalar | 0.139725 | 0.026806   | 5.21x   | -80.82%        |
| 1920x1080 | float32 | 4        | divide        | sse42  | 0.139725 | 0.007805   | 17.90x  | -94.41%        |
| 1920x1080 | float32 | 4        | divide        | avx2   | 0.139725 | 0.006734   | 20.75x  | -95.18%        |
| 1920x1080 | float32 | 4        | overlay       | scalar | 0.193398 | 0.065825   | 2.94x   | -65.96%        |
| 1920x1080 | float32 | 4        | overlay       | sse42  | 0.193398 | 0.008290   | 23.33x  | -95.71%        |
| 1920x1080 | float32 | 4        | overlay       | avx2   | 0.193398 | 0.006894   | 28.05x  | -96.44%        |
| 2560x1440 | uint8   | 3        | normal        | scalar | 0.383768 | 0.101262   | 3.79x   | -73.61%        |
| 2560x1440 | uint8   | 3        | normal        | sse42  | 0.383768 | 0.086093   | 4.46x   | -77.57%        |
| 2560x1440 | uint8   | 3        | normal        | avx2   | 0.383768 | 0.087259   | 4.40x   | -77.26%        |
| 2560x1440 | uint8   | 3        | soft_light    | scalar | 0.475929 | 0.113790   | 4.18x   | -76.09%        |
| 2560x1440 | uint8   | 3        | soft_light    | sse42  | 0.475929 | 0.084069   | 5.66x   | -82.34%        |
| 2560x1440 | uint8   | 3        | soft_light    | avx2   | 0.475929 | 0.087906   | 5.41x   | -81.53%        |
| 2560x1440 | uint8   | 3        | lighten_only  | scalar | 0.341492 | 0.123576   | 2.76x   | -63.81%        |
| 2560x1440 | uint8   | 3        | lighten_only  | sse42  | 0.341492 | 0.081413   | 4.19x   | -76.16%        |
| 2560x1440 | uint8   | 3        | lighten_only  | avx2   | 0.341492 | 0.082547   | 4.14x   | -75.83%        |
| 2560x1440 | uint8   | 3        | screen        | scalar | 0.357933 | 0.105577   | 3.39x   | -70.50%        |
| 2560x1440 | uint8   | 3        | screen        | sse42  | 0.357933 | 0.082168   | 4.36x   | -77.04%        |
| 2560x1440 | uint8   | 3        | screen        | avx2   | 0.357933 | 0.083676   | 4.28x   | -76.62%        |
| 2560x1440 | uint8   | 3        | dodge         | scalar | 0.359429 | 0.112063   | 3.21x   | -68.82%        |
| 2560x1440 | uint8   | 3        | dodge         | sse42  | 0.359429 | 0.084100   | 4.27x   | -76.60%        |
| 2560x1440 | uint8   | 3        | dodge         | avx2   | 0.359429 | 0.084406   | 4.26x   | -76.52%        |
| 2560x1440 | uint8   | 3        | addition      | scalar | 0.352547 | 0.148407   | 2.38x   | -57.90%        |
| 2560x1440 | uint8   | 3        | addition      | sse42  | 0.352547 | 0.081709   | 4.31x   | -76.82%        |
| 2560x1440 | uint8   | 3        | addition      | avx2   | 0.352547 | 0.083381   | 4.23x   | -76.35%        |
| 2560x1440 | uint8   | 3        | darken_only   | scalar | 0.343603 | 0.120724   | 2.85x   | -64.87%        |
| 2560x1440 | uint8   | 3        | darken_only   | sse42  | 0.343603 | 0.081516   | 4.22x   | -76.28%        |
| 2560x1440 | uint8   | 3        | darken_only   | avx2   | 0.343603 | 0.085292   | 4.03x   | -75.18%        |
| 2560x1440 | uint8   | 3        | multiply      | scalar | 0.356297 | 0.111388   | 3.20x   | -68.74%        |
| 2560x1440 | uint8   | 3        | multiply      | sse42  | 0.356297 | 0.083665   | 4.26x   | -76.52%        |
| 2560x1440 | uint8   | 3        | multiply      | avx2   | 0.356297 | 0.084211   | 4.23x   | -76.36%        |
| 2560x1440 | uint8   | 3        | hard_light    | scalar | 0.515085 | 0.180268   | 2.86x   | -65.00%        |
| 2560x1440 | uint8   | 3        | hard_light    | sse42  | 0.515085 | 0.085988   | 5.99x   | -83.31%        |
| 2560x1440 | uint8   | 3        | hard_light    | avx2   | 0.515085 | 0.085547   | 6.02x   | -83.39%        |
| 2560x1440 | uint8   | 3        | difference    | scalar | 0.456213 | 0.106556   | 4.28x   | -76.64%        |
| 2560x1440 | uint8   | 3        | difference    | sse42  | 0.456213 | 0.085065   | 5.36x   | -81.35%        |
| 2560x1440 | uint8   | 3        | difference    | avx2   | 0.456213 | 0.083769   | 5.45x   | -81.64%        |
| 2560x1440 | uint8   | 3        | subtract      | scalar | 0.353599 | 0.101092   | 3.50x   | -71.41%        |
| 2560x1440 | uint8   | 3        | subtract      | sse42  | 0.353599 | 0.085906   | 4.12x   | -75.71%        |
| 2560x1440 | uint8   | 3        | subtract      | avx2   | 0.353599 | 0.087786   | 4.03x   | -75.17%        |
| 2560x1440 | uint8   | 3        | grain_extract | scalar | 0.368356 | 0.130675   | 2.82x   | -64.52%        |
| 2560x1440 | uint8   | 3        | grain_extract | sse42  | 0.368356 | 0.083748   | 4.40x   | -77.26%        |
| 2560x1440 | uint8   | 3        | grain_extract | avx2   | 0.368356 | 0.085064   | 4.33x   | -76.91%        |
| 2560x1440 | uint8   | 3        | grain_merge   | scalar | 0.362793 | 0.127251   | 2.85x   | -64.92%        |
| 2560x1440 | uint8   | 3        | grain_merge   | sse42  | 0.362793 | 0.084575   | 4.29x   | -76.69%        |
| 2560x1440 | uint8   | 3        | grain_merge   | avx2   | 0.362793 | 0.091946   | 3.95x   | -74.66%        |
| 2560x1440 | uint8   | 3        | divide        | scalar | 0.362053 | 0.113340   | 3.19x   | -68.70%        |
| 2560x1440 | uint8   | 3        | divide        | sse42  | 0.362053 | 0.088030   | 4.11x   | -75.69%        |
| 2560x1440 | uint8   | 3        | divide        | avx2   | 0.362053 | 0.085977   | 4.21x   | -76.25%        |
| 2560x1440 | uint8   | 3        | overlay       | scalar | 0.470777 | 0.171763   | 2.74x   | -63.51%        |
| 2560x1440 | uint8   | 3        | overlay       | sse42  | 0.470777 | 0.085819   | 5.49x   | -81.77%        |
| 2560x1440 | uint8   | 3        | overlay       | avx2   | 0.470777 | 0.084619   | 5.56x   | -82.03%        |
| 2560x1440 | uint8   | 4        | normal        | scalar | 0.265066 | 0.077989   | 3.40x   | -70.58%        |
| 2560x1440 | uint8   | 4        | normal        | sse42  | 0.265066 | 0.010417   | 25.44x  | -96.07%        |
| 2560x1440 | uint8   | 4        | normal        | avx2   | 0.265066 | 0.009333   | 28.40x  | -96.48%        |
| 2560x1440 | uint8   | 4        | soft_light    | scalar | 0.359367 | 0.105934   | 3.39x   | -70.52%        |
| 2560x1440 | uint8   | 4        | soft_light    | sse42  | 0.359367 | 0.012397   | 28.99x  | -96.55%        |
| 2560x1440 | uint8   | 4        | soft_light    | avx2   | 0.359367 | 0.010619   | 33.84x  | -97.05%        |
| 2560x1440 | uint8   | 4        | lighten_only  | scalar | 0.256717 | 0.110573   | 2.32x   | -56.93%        |
| 2560x1440 | uint8   | 4        | lighten_only  | sse42  | 0.256717 | 0.011241   | 22.84x  | -95.62%        |
| 2560x1440 | uint8   | 4        | lighten_only  | avx2   | 0.256717 | 0.010380   | 24.73x  | -95.96%        |
| 2560x1440 | uint8   | 4        | screen        | scalar | 0.279760 | 0.099736   | 2.81x   | -64.35%        |
| 2560x1440 | uint8   | 4        | screen        | sse42  | 0.279760 | 0.011684   | 23.94x  | -95.82%        |
| 2560x1440 | uint8   | 4        | screen        | avx2   | 0.279760 | 0.010855   | 25.77x  | -96.12%        |
| 2560x1440 | uint8   | 4        | dodge         | scalar | 0.276919 | 0.104966   | 2.64x   | -62.10%        |
| 2560x1440 | uint8   | 4        | dodge         | sse42  | 0.276919 | 0.013253   | 20.89x  | -95.21%        |
| 2560x1440 | uint8   | 4        | dodge         | avx2   | 0.276919 | 0.010840   | 25.55x  | -96.09%        |
| 2560x1440 | uint8   | 4        | addition      | scalar | 0.262110 | 0.118149   | 2.22x   | -54.92%        |
| 2560x1440 | uint8   | 4        | addition      | sse42  | 0.262110 | 0.014770   | 17.75x  | -94.37%        |
| 2560x1440 | uint8   | 4        | addition      | avx2   | 0.262110 | 0.010859   | 24.14x  | -95.86%        |
| 2560x1440 | uint8   | 4        | darken_only   | scalar | 0.254987 | 0.113005   | 2.26x   | -55.68%        |
| 2560x1440 | uint8   | 4        | darken_only   | sse42  | 0.254987 | 0.011279   | 22.61x  | -95.58%        |
| 2560x1440 | uint8   | 4        | darken_only   | avx2   | 0.254987 | 0.010534   | 24.21x  | -95.87%        |
| 2560x1440 | uint8   | 4        | multiply      | scalar | 0.264925 | 0.100612   | 2.63x   | -62.02%        |
| 2560x1440 | uint8   | 4        | multiply      | sse42  | 0.264925 | 0.011466   | 23.11x  | -95.67%        |
| 2560x1440 | uint8   | 4        | multiply      | avx2   | 0.264925 | 0.010486   | 25.26x  | -96.04%        |
| 2560x1440 | uint8   | 4        | hard_light    | scalar | 0.415288 | 0.161672   | 2.57x   | -61.07%        |
| 2560x1440 | uint8   | 4        | hard_light    | sse42  | 0.415288 | 0.013506   | 30.75x  | -96.75%        |
| 2560x1440 | uint8   | 4        | hard_light    | avx2   | 0.415288 | 0.010481   | 39.62x  | -97.48%        |
| 2560x1440 | uint8   | 4        | difference    | scalar | 0.371220 | 0.098874   | 3.75x   | -73.37%        |
| 2560x1440 | uint8   | 4        | difference    | sse42  | 0.371220 | 0.011371   | 32.65x  | -96.94%        |
| 2560x1440 | uint8   | 4        | difference    | avx2   | 0.371220 | 0.010448   | 35.53x  | -97.19%        |
| 2560x1440 | uint8   | 4        | subtract      | scalar | 0.263494 | 0.095870   | 2.75x   | -63.62%        |
| 2560x1440 | uint8   | 4        | subtract      | sse42  | 0.263494 | 0.014926   | 17.65x  | -94.34%        |
| 2560x1440 | uint8   | 4        | subtract      | avx2   | 0.263494 | 0.010871   | 24.24x  | -95.87%        |
| 2560x1440 | uint8   | 4        | grain_extract | scalar | 0.272352 | 0.116689   | 2.33x   | -57.16%        |
| 2560x1440 | uint8   | 4        | grain_extract | sse42  | 0.272352 | 0.012049   | 22.60x  | -95.58%        |
| 2560x1440 | uint8   | 4        | grain_extract | avx2   | 0.272352 | 0.010510   | 25.91x  | -96.14%        |
| 2560x1440 | uint8   | 4        | grain_merge   | scalar | 0.273891 | 0.120681   | 2.27x   | -55.94%        |
| 2560x1440 | uint8   | 4        | grain_merge   | sse42  | 0.273891 | 0.011946   | 22.93x  | -95.64%        |
| 2560x1440 | uint8   | 4        | grain_merge   | avx2   | 0.273891 | 0.010496   | 26.10x  | -96.17%        |
| 2560x1440 | uint8   | 4        | divide        | scalar | 0.285913 | 0.106192   | 2.69x   | -62.86%        |
| 2560x1440 | uint8   | 4        | divide        | sse42  | 0.285913 | 0.012045   | 23.74x  | -95.79%        |
| 2560x1440 | uint8   | 4        | divide        | avx2   | 0.285913 | 0.010267   | 27.85x  | -96.41%        |
| 2560x1440 | uint8   | 4        | overlay       | scalar | 0.377951 | 0.155440   | 2.43x   | -58.87%        |
| 2560x1440 | uint8   | 4        | overlay       | sse42  | 0.377951 | 0.012552   | 30.11x  | -96.68%        |
| 2560x1440 | uint8   | 4        | overlay       | avx2   | 0.377951 | 0.010673   | 35.41x  | -97.18%        |
| 2560x1440 | float32 | 3        | normal        | scalar | 0.303949 | 0.033214   | 9.15x   | -89.07%        |
| 2560x1440 | float32 | 3        | normal        | sse42  | 0.303949 | 0.017242   | 17.63x  | -94.33%        |
| 2560x1440 | float32 | 3        | normal        | avx2   | 0.303949 | 0.014128   | 21.51x  | -95.35%        |
| 2560x1440 | float32 | 3        | soft_light    | scalar | 0.394276 | 0.043992   | 8.96x   | -88.84%        |
| 2560x1440 | float32 | 3        | soft_light    | sse42  | 0.394276 | 0.019021   | 20.73x  | -95.18%        |
| 2560x1440 | float32 | 3        | soft_light    | avx2   | 0.394276 | 0.016528   | 23.86x  | -95.81%        |
| 2560x1440 | float32 | 3        | lighten_only  | scalar | 0.296720 | 0.051088   | 5.81x   | -82.78%        |
| 2560x1440 | float32 | 3        | lighten_only  | sse42  | 0.296720 | 0.018235   | 16.27x  | -93.85%        |
| 2560x1440 | float32 | 3        | lighten_only  | avx2   | 0.296720 | 0.016106   | 18.42x  | -94.57%        |
| 2560x1440 | float32 | 3        | screen        | scalar | 0.324505 | 0.042536   | 7.63x   | -86.89%        |
| 2560x1440 | float32 | 3        | screen        | sse42  | 0.324505 | 0.018107   | 17.92x  | -94.42%        |
| 2560x1440 | float32 | 3        | screen        | avx2   | 0.324505 | 0.015961   | 20.33x  | -95.08%        |
| 2560x1440 | float32 | 3        | dodge         | scalar | 0.315437 | 0.045021   | 7.01x   | -85.73%        |
| 2560x1440 | float32 | 3        | dodge         | sse42  | 0.315437 | 0.019407   | 16.25x  | -93.85%        |
| 2560x1440 | float32 | 3        | dodge         | avx2   | 0.315437 | 0.017313   | 18.22x  | -94.51%        |
| 2560x1440 | float32 | 3        | addition      | scalar | 0.318496 | 0.097223   | 3.28x   | -69.47%        |
| 2560x1440 | float32 | 3        | addition      | sse42  | 0.318496 | 0.019576   | 16.27x  | -93.85%        |
| 2560x1440 | float32 | 3        | addition      | avx2   | 0.318496 | 0.016493   | 19.31x  | -94.82%        |
| 2560x1440 | float32 | 3        | darken_only   | scalar | 0.300457 | 0.054449   | 5.52x   | -81.88%        |
| 2560x1440 | float32 | 3        | darken_only   | sse42  | 0.300457 | 0.018139   | 16.56x  | -93.96%        |
| 2560x1440 | float32 | 3        | darken_only   | avx2   | 0.300457 | 0.015810   | 19.00x  | -94.74%        |
| 2560x1440 | float32 | 3        | multiply      | scalar | 0.308970 | 0.041442   | 7.46x   | -86.59%        |
| 2560x1440 | float32 | 3        | multiply      | sse42  | 0.308970 | 0.017681   | 17.47x  | -94.28%        |
| 2560x1440 | float32 | 3        | multiply      | avx2   | 0.308970 | 0.016995   | 18.18x  | -94.50%        |
| 2560x1440 | float32 | 3        | hard_light    | scalar | 0.454656 | 0.113089   | 4.02x   | -75.13%        |
| 2560x1440 | float32 | 3        | hard_light    | sse42  | 0.454656 | 0.019530   | 23.28x  | -95.70%        |
| 2560x1440 | float32 | 3        | hard_light    | avx2   | 0.454656 | 0.016880   | 26.94x  | -96.29%        |
| 2560x1440 | float32 | 3        | difference    | scalar | 0.401022 | 0.042242   | 9.49x   | -89.47%        |
| 2560x1440 | float32 | 3        | difference    | sse42  | 0.401022 | 0.018397   | 21.80x  | -95.41%        |
| 2560x1440 | float32 | 3        | difference    | avx2   | 0.401022 | 0.016527   | 24.27x  | -95.88%        |
| 2560x1440 | float32 | 3        | subtract      | scalar | 0.311619 | 0.047506   | 6.56x   | -84.76%        |
| 2560x1440 | float32 | 3        | subtract      | sse42  | 0.311619 | 0.018718   | 16.65x  | -93.99%        |
| 2560x1440 | float32 | 3        | subtract      | avx2   | 0.311619 | 0.016660   | 18.70x  | -94.65%        |
| 2560x1440 | float32 | 3        | grain_extract | scalar | 0.321804 | 0.068454   | 4.70x   | -78.73%        |
| 2560x1440 | float32 | 3        | grain_extract | sse42  | 0.321804 | 0.018711   | 17.20x  | -94.19%        |
| 2560x1440 | float32 | 3        | grain_extract | avx2   | 0.321804 | 0.016910   | 19.03x  | -94.75%        |
| 2560x1440 | float32 | 3        | grain_merge   | scalar | 0.334334 | 0.069140   | 4.84x   | -79.32%        |
| 2560x1440 | float32 | 3        | grain_merge   | sse42  | 0.334334 | 0.018302   | 18.27x  | -94.53%        |
| 2560x1440 | float32 | 3        | grain_merge   | avx2   | 0.334334 | 0.016935   | 19.74x  | -94.93%        |
| 2560x1440 | float32 | 3        | divide        | scalar | 0.335945 | 0.044606   | 7.53x   | -86.72%        |
| 2560x1440 | float32 | 3        | divide        | sse42  | 0.335945 | 0.019473   | 17.25x  | -94.20%        |
| 2560x1440 | float32 | 3        | divide        | avx2   | 0.335945 | 0.016392   | 20.50x  | -95.12%        |
| 2560x1440 | float32 | 3        | overlay       | scalar | 0.428773 | 0.109438   | 3.92x   | -74.48%        |
| 2560x1440 | float32 | 3        | overlay       | sse42  | 0.428773 | 0.019558   | 21.92x  | -95.44%        |
| 2560x1440 | float32 | 3        | overlay       | avx2   | 0.428773 | 0.016442   | 26.08x  | -96.17%        |
| 2560x1440 | float32 | 4        | normal        | scalar | 0.250863 | 0.041919   | 5.98x   | -83.29%        |
| 2560x1440 | float32 | 4        | normal        | sse42  | 0.250863 | 0.018447   | 13.60x  | -92.65%        |
| 2560x1440 | float32 | 4        | normal        | avx2   | 0.250863 | 0.016605   | 15.11x  | -93.38%        |
| 2560x1440 | float32 | 4        | soft_light    | scalar | 0.354947 | 0.049746   | 7.14x   | -85.98%        |
| 2560x1440 | float32 | 4        | soft_light    | sse42  | 0.354947 | 0.018279   | 19.42x  | -94.85%        |
| 2560x1440 | float32 | 4        | soft_light    | avx2   | 0.354947 | 0.015570   | 22.80x  | -95.61%        |
| 2560x1440 | float32 | 4        | lighten_only  | scalar | 0.234673 | 0.063778   | 3.68x   | -72.82%        |
| 2560x1440 | float32 | 4        | lighten_only  | sse42  | 0.234673 | 0.020165   | 11.64x  | -91.41%        |
| 2560x1440 | float32 | 4        | lighten_only  | avx2   | 0.234673 | 0.019204   | 12.22x  | -91.82%        |
| 2560x1440 | float32 | 4        | screen        | scalar | 0.263320 | 0.048230   | 5.46x   | -81.68%        |
| 2560x1440 | float32 | 4        | screen        | sse42  | 0.263320 | 0.013759   | 19.14x  | -94.77%        |
| 2560x1440 | float32 | 4        | screen        | avx2   | 0.263320 | 0.013518   | 19.48x  | -94.87%        |
| 2560x1440 | float32 | 4        | dodge         | scalar | 0.255952 | 0.054876   | 4.66x   | -78.56%        |
| 2560x1440 | float32 | 4        | dodge         | sse42  | 0.255952 | 0.020677   | 12.38x  | -91.92%        |
| 2560x1440 | float32 | 4        | dodge         | avx2   | 0.255952 | 0.020622   | 12.41x  | -91.94%        |
| 2560x1440 | float32 | 4        | addition      | scalar | 0.251539 | 0.086019   | 2.92x   | -65.80%        |
| 2560x1440 | float32 | 4        | addition      | sse42  | 0.251539 | 0.018945   | 13.28x  | -92.47%        |
| 2560x1440 | float32 | 4        | addition      | avx2   | 0.251539 | 0.016316   | 15.42x  | -93.51%        |
| 2560x1440 | float32 | 4        | darken_only   | scalar | 0.243838 | 0.063686   | 3.83x   | -73.88%        |
| 2560x1440 | float32 | 4        | darken_only   | sse42  | 0.243838 | 0.020585   | 11.85x  | -91.56%        |
| 2560x1440 | float32 | 4        | darken_only   | avx2   | 0.243838 | 0.020620   | 11.83x  | -91.54%        |
| 2560x1440 | float32 | 4        | multiply      | scalar | 0.248644 | 0.047913   | 5.19x   | -80.73%        |
| 2560x1440 | float32 | 4        | multiply      | sse42  | 0.248644 | 0.013620   | 18.26x  | -94.52%        |
| 2560x1440 | float32 | 4        | multiply      | avx2   | 0.248644 | 0.012539   | 19.83x  | -94.96%        |
| 2560x1440 | float32 | 4        | hard_light    | scalar | 0.399351 | 0.119733   | 3.34x   | -70.02%        |
| 2560x1440 | float32 | 4        | hard_light    | sse42  | 0.399351 | 0.020943   | 19.07x  | -94.76%        |
| 2560x1440 | float32 | 4        | hard_light    | avx2   | 0.399351 | 0.018778   | 21.27x  | -95.30%        |
| 2560x1440 | float32 | 4        | difference    | scalar | 0.346599 | 0.046237   | 7.50x   | -86.66%        |
| 2560x1440 | float32 | 4        | difference    | sse42  | 0.346599 | 0.013245   | 26.17x  | -96.18%        |
| 2560x1440 | float32 | 4        | difference    | avx2   | 0.346599 | 0.012756   | 27.17x  | -96.32%        |
| 2560x1440 | float32 | 4        | subtract      | scalar | 0.250380 | 0.063982   | 3.91x   | -74.45%        |
| 2560x1440 | float32 | 4        | subtract      | sse42  | 0.250380 | 0.020915   | 11.97x  | -91.65%        |
| 2560x1440 | float32 | 4        | subtract      | avx2   | 0.250380 | 0.018542   | 13.50x  | -92.59%        |
| 2560x1440 | float32 | 4        | grain_extract | scalar | 0.252381 | 0.067446   | 3.74x   | -73.28%        |
| 2560x1440 | float32 | 4        | grain_extract | sse42  | 0.252381 | 0.014137   | 17.85x  | -94.40%        |
| 2560x1440 | float32 | 4        | grain_extract | avx2   | 0.252381 | 0.013378   | 18.87x  | -94.70%        |
| 2560x1440 | float32 | 4        | grain_merge   | scalar | 0.254006 | 0.077216   | 3.29x   | -69.60%        |
| 2560x1440 | float32 | 4        | grain_merge   | sse42  | 0.254006 | 0.021690   | 11.71x  | -91.46%        |
| 2560x1440 | float32 | 4        | grain_merge   | avx2   | 0.254006 | 0.019823   | 12.81x  | -92.20%        |
| 2560x1440 | float32 | 4        | divide        | scalar | 0.266006 | 0.049633   | 5.36x   | -81.34%        |
| 2560x1440 | float32 | 4        | divide        | sse42  | 0.266006 | 0.014220   | 18.71x  | -94.65%        |
| 2560x1440 | float32 | 4        | divide        | avx2   | 0.266006 | 0.011748   | 22.64x  | -95.58%        |
| 2560x1440 | float32 | 4        | overlay       | scalar | 0.363498 | 0.112688   | 3.23x   | -69.00%        |
| 2560x1440 | float32 | 4        | overlay       | sse42  | 0.363498 | 0.021220   | 17.13x  | -94.16%        |
| 2560x1440 | float32 | 4        | overlay       | avx2   | 0.363498 | 0.018347   | 19.81x  | -94.95%        |
| 3840x2160 | uint8   | 3        | normal        | scalar | 0.764169 | 0.208383   | 3.67x   | -72.73%        |
| 3840x2160 | uint8   | 3        | normal        | sse42  | 0.764169 | 0.179082   | 4.27x   | -76.57%        |
| 3840x2160 | uint8   | 3        | normal        | avx2   | 0.764169 | 0.189501   | 4.03x   | -75.20%        |
| 3840x2160 | uint8   | 3        | soft_light    | scalar | 0.983079 | 0.255796   | 3.84x   | -73.98%        |
| 3840x2160 | uint8   | 3        | soft_light    | sse42  | 0.983079 | 0.196053   | 5.01x   | -80.06%        |
| 3840x2160 | uint8   | 3        | soft_light    | avx2   | 0.983079 | 0.191380   | 5.14x   | -80.53%        |
| 3840x2160 | uint8   | 3        | lighten_only  | scalar | 0.739453 | 0.286557   | 2.58x   | -61.25%        |
| 3840x2160 | uint8   | 3        | lighten_only  | sse42  | 0.739453 | 0.190464   | 3.88x   | -74.24%        |
| 3840x2160 | uint8   | 3        | lighten_only  | avx2   | 0.739453 | 0.187554   | 3.94x   | -74.64%        |
| 3840x2160 | uint8   | 3        | screen        | scalar | 0.779696 | 0.243646   | 3.20x   | -68.75%        |
| 3840x2160 | uint8   | 3        | screen        | sse42  | 0.779696 | 0.185930   | 4.19x   | -76.15%        |
| 3840x2160 | uint8   | 3        | screen        | avx2   | 0.779696 | 0.187727   | 4.15x   | -75.92%        |
| 3840x2160 | uint8   | 3        | dodge         | scalar | 0.793023 | 0.251737   | 3.15x   | -68.26%        |
| 3840x2160 | uint8   | 3        | dodge         | sse42  | 0.793023 | 0.188846   | 4.20x   | -76.19%        |
| 3840x2160 | uint8   | 3        | dodge         | avx2   | 0.793023 | 0.189861   | 4.18x   | -76.06%        |
| 3840x2160 | uint8   | 3        | addition      | scalar | 0.758006 | 0.337960   | 2.24x   | -55.41%        |
| 3840x2160 | uint8   | 3        | addition      | sse42  | 0.758006 | 0.194065   | 3.91x   | -74.40%        |
| 3840x2160 | uint8   | 3        | addition      | avx2   | 0.758006 | 0.195230   | 3.88x   | -74.24%        |
| 3840x2160 | uint8   | 3        | darken_only   | scalar | 0.736529 | 0.271878   | 2.71x   | -63.09%        |
| 3840x2160 | uint8   | 3        | darken_only   | sse42  | 0.736529 | 0.182886   | 4.03x   | -75.17%        |
| 3840x2160 | uint8   | 3        | darken_only   | avx2   | 0.736529 | 0.196641   | 3.75x   | -73.30%        |
| 3840x2160 | uint8   | 3        | multiply      | scalar | 0.758993 | 0.250436   | 3.03x   | -67.00%        |
| 3840x2160 | uint8   | 3        | multiply      | sse42  | 0.758993 | 0.191316   | 3.97x   | -74.79%        |
| 3840x2160 | uint8   | 3        | multiply      | avx2   | 0.758993 | 0.191957   | 3.95x   | -74.71%        |
| 3840x2160 | uint8   | 3        | hard_light    | scalar | 1.086137 | 0.397284   | 2.73x   | -63.42%        |
| 3840x2160 | uint8   | 3        | hard_light    | sse42  | 1.086137 | 0.197772   | 5.49x   | -81.79%        |
| 3840x2160 | uint8   | 3        | hard_light    | avx2   | 1.086137 | 0.199060   | 5.46x   | -81.67%        |
| 3840x2160 | uint8   | 3        | difference    | scalar | 0.996773 | 0.243638   | 4.09x   | -75.56%        |
| 3840x2160 | uint8   | 3        | difference    | sse42  | 0.996773 | 0.183551   | 5.43x   | -81.59%        |
| 3840x2160 | uint8   | 3        | difference    | avx2   | 0.996773 | 0.188631   | 5.28x   | -81.08%        |
| 3840x2160 | uint8   | 3        | subtract      | scalar | 0.770045 | 0.234950   | 3.28x   | -69.49%        |
| 3840x2160 | uint8   | 3        | subtract      | sse42  | 0.770045 | 0.185552   | 4.15x   | -75.90%        |
| 3840x2160 | uint8   | 3        | subtract      | avx2   | 0.770045 | 0.190278   | 4.05x   | -75.29%        |
| 3840x2160 | uint8   | 3        | grain_extract | scalar | 0.774652 | 0.306304   | 2.53x   | -60.46%        |
| 3840x2160 | uint8   | 3        | grain_extract | sse42  | 0.774652 | 0.193207   | 4.01x   | -75.06%        |
| 3840x2160 | uint8   | 3        | grain_extract | avx2   | 0.774652 | 0.193335   | 4.01x   | -75.04%        |
| 3840x2160 | uint8   | 3        | grain_merge   | scalar | 0.810242 | 0.295861   | 2.74x   | -63.48%        |
| 3840x2160 | uint8   | 3        | grain_merge   | sse42  | 0.810242 | 0.187922   | 4.31x   | -76.81%        |
| 3840x2160 | uint8   | 3        | grain_merge   | avx2   | 0.810242 | 0.191391   | 4.23x   | -76.38%        |
| 3840x2160 | uint8   | 3        | divide        | scalar | 0.786236 | 0.261216   | 3.01x   | -66.78%        |
| 3840x2160 | uint8   | 3        | divide        | sse42  | 0.786236 | 0.206753   | 3.80x   | -73.70%        |
| 3840x2160 | uint8   | 3        | divide        | avx2   | 0.786236 | 0.191895   | 4.10x   | -75.59%        |
| 3840x2160 | uint8   | 3        | overlay       | scalar | 1.013446 | 0.385739   | 2.63x   | -61.94%        |
| 3840x2160 | uint8   | 3        | overlay       | sse42  | 1.013446 | 0.189305   | 5.35x   | -81.32%        |
| 3840x2160 | uint8   | 3        | overlay       | avx2   | 1.013446 | 0.187695   | 5.40x   | -81.48%        |
| 3840x2160 | uint8   | 4        | normal        | scalar | 0.571063 | 0.174668   | 3.27x   | -69.41%        |
| 3840x2160 | uint8   | 4        | normal        | sse42  | 0.571063 | 0.022990   | 24.84x  | -95.97%        |
| 3840x2160 | uint8   | 4        | normal        | avx2   | 0.571063 | 0.020593   | 27.73x  | -96.39%        |
| 3840x2160 | uint8   | 4        | soft_light    | scalar | 0.782661 | 0.240673   | 3.25x   | -69.25%        |
| 3840x2160 | uint8   | 4        | soft_light    | sse42  | 0.782661 | 0.027368   | 28.60x  | -96.50%        |
| 3840x2160 | uint8   | 4        | soft_light    | avx2   | 0.782661 | 0.023442   | 33.39x  | -97.00%        |
| 3840x2160 | uint8   | 4        | lighten_only  | scalar | 0.550172 | 0.249519   | 2.20x   | -54.65%        |
| 3840x2160 | uint8   | 4        | lighten_only  | sse42  | 0.550172 | 0.024729   | 22.25x  | -95.51%        |
| 3840x2160 | uint8   | 4        | lighten_only  | avx2   | 0.550172 | 0.023554   | 23.36x  | -95.72%        |
| 3840x2160 | uint8   | 4        | screen        | scalar | 0.595019 | 0.234561   | 2.54x   | -60.58%        |
| 3840x2160 | uint8   | 4        | screen        | sse42  | 0.595019 | 0.027282   | 21.81x  | -95.41%        |
| 3840x2160 | uint8   | 4        | screen        | avx2   | 0.595019 | 0.024440   | 24.35x  | -95.89%        |
| 3840x2160 | uint8   | 4        | dodge         | scalar | 0.596269 | 0.233317   | 2.56x   | -60.87%        |
| 3840x2160 | uint8   | 4        | dodge         | sse42  | 0.596269 | 0.029933   | 19.92x  | -94.98%        |
| 3840x2160 | uint8   | 4        | dodge         | avx2   | 0.596269 | 0.024030   | 24.81x  | -95.97%        |
| 3840x2160 | uint8   | 4        | addition      | scalar | 0.566671 | 0.280244   | 2.02x   | -50.55%        |
| 3840x2160 | uint8   | 4        | addition      | sse42  | 0.566671 | 0.033401   | 16.97x  | -94.11%        |
| 3840x2160 | uint8   | 4        | addition      | avx2   | 0.566671 | 0.025470   | 22.25x  | -95.51%        |
| 3840x2160 | uint8   | 4        | darken_only   | scalar | 0.578945 | 0.265486   | 2.18x   | -54.14%        |
| 3840x2160 | uint8   | 4        | darken_only   | sse42  | 0.578945 | 0.031422   | 18.42x  | -94.57%        |
| 3840x2160 | uint8   | 4        | darken_only   | avx2   | 0.578945 | 0.027597   | 20.98x  | -95.23%        |
| 3840x2160 | uint8   | 4        | multiply      | scalar | 0.664092 | 0.234521   | 2.83x   | -64.69%        |
| 3840x2160 | uint8   | 4        | multiply      | sse42  | 0.664092 | 0.028890   | 22.99x  | -95.65%        |
| 3840x2160 | uint8   | 4        | multiply      | avx2   | 0.664092 | 0.024214   | 27.43x  | -96.35%        |
| 3840x2160 | uint8   | 4        | hard_light    | scalar | 1.004486 | 0.375455   | 2.68x   | -62.62%        |
| 3840x2160 | uint8   | 4        | hard_light    | sse42  | 1.004486 | 0.031142   | 32.26x  | -96.90%        |
| 3840x2160 | uint8   | 4        | hard_light    | avx2   | 1.004486 | 0.024312   | 41.32x  | -97.58%        |
| 3840x2160 | uint8   | 4        | difference    | scalar | 0.811227 | 0.223041   | 3.64x   | -72.51%        |
| 3840x2160 | uint8   | 4        | difference    | sse42  | 0.811227 | 0.025720   | 31.54x  | -96.83%        |
| 3840x2160 | uint8   | 4        | difference    | avx2   | 0.811227 | 0.023571   | 34.42x  | -97.09%        |
| 3840x2160 | uint8   | 4        | subtract      | scalar | 0.582841 | 0.218339   | 2.67x   | -62.54%        |
| 3840x2160 | uint8   | 4        | subtract      | sse42  | 0.582841 | 0.032940   | 17.69x  | -94.35%        |
| 3840x2160 | uint8   | 4        | subtract      | avx2   | 0.582841 | 0.024562   | 23.73x  | -95.79%        |
| 3840x2160 | uint8   | 4        | grain_extract | scalar | 0.610981 | 0.271388   | 2.25x   | -55.58%        |
| 3840x2160 | uint8   | 4        | grain_extract | sse42  | 0.610981 | 0.027009   | 22.62x  | -95.58%        |
| 3840x2160 | uint8   | 4        | grain_extract | avx2   | 0.610981 | 0.023883   | 25.58x  | -96.09%        |
| 3840x2160 | uint8   | 4        | grain_merge   | scalar | 0.582330 | 0.267244   | 2.18x   | -54.11%        |
| 3840x2160 | uint8   | 4        | grain_merge   | sse42  | 0.582330 | 0.027086   | 21.50x  | -95.35%        |
| 3840x2160 | uint8   | 4        | grain_merge   | avx2   | 0.582330 | 0.025932   | 22.46x  | -95.55%        |
| 3840x2160 | uint8   | 4        | divide        | scalar | 0.609479 | 0.230829   | 2.64x   | -62.13%        |
| 3840x2160 | uint8   | 4        | divide        | sse42  | 0.609479 | 0.026807   | 22.74x  | -95.60%        |
| 3840x2160 | uint8   | 4        | divide        | avx2   | 0.609479 | 0.023174   | 26.30x  | -96.20%        |
| 3840x2160 | uint8   | 4        | overlay       | scalar | 0.819986 | 0.363509   | 2.26x   | -55.67%        |
| 3840x2160 | uint8   | 4        | overlay       | sse42  | 0.819986 | 0.030374   | 27.00x  | -96.30%        |
| 3840x2160 | uint8   | 4        | overlay       | avx2   | 0.819986 | 0.024654   | 33.26x  | -96.99%        |
| 3840x2160 | float32 | 3        | normal        | scalar | 0.698704 | 0.079033   | 8.84x   | -88.69%        |
| 3840x2160 | float32 | 3        | normal        | sse42  | 0.698704 | 0.039033   | 17.90x  | -94.41%        |
| 3840x2160 | float32 | 3        | normal        | avx2   | 0.698704 | 0.031206   | 22.39x  | -95.53%        |
| 3840x2160 | float32 | 3        | soft_light    | scalar | 0.905923 | 0.100559   | 9.01x   | -88.90%        |
| 3840x2160 | float32 | 3        | soft_light    | sse42  | 0.905923 | 0.042638   | 21.25x  | -95.29%        |
| 3840x2160 | float32 | 3        | soft_light    | avx2   | 0.905923 | 0.037496   | 24.16x  | -95.86%        |
| 3840x2160 | float32 | 3        | lighten_only  | scalar | 0.675860 | 0.120121   | 5.63x   | -82.23%        |
| 3840x2160 | float32 | 3        | lighten_only  | sse42  | 0.675860 | 0.039376   | 17.16x  | -94.17%        |
| 3840x2160 | float32 | 3        | lighten_only  | avx2   | 0.675860 | 0.035240   | 19.18x  | -94.79%        |
| 3840x2160 | float32 | 3        | screen        | scalar | 0.704962 | 0.098390   | 7.16x   | -86.04%        |
| 3840x2160 | float32 | 3        | screen        | sse42  | 0.704962 | 0.041392   | 17.03x  | -94.13%        |
| 3840x2160 | float32 | 3        | screen        | avx2   | 0.704962 | 0.036036   | 19.56x  | -94.89%        |
| 3840x2160 | float32 | 3        | dodge         | scalar | 0.725652 | 0.101800   | 7.13x   | -85.97%        |
| 3840x2160 | float32 | 3        | dodge         | sse42  | 0.725652 | 0.043125   | 16.83x  | -94.06%        |
| 3840x2160 | float32 | 3        | dodge         | avx2   | 0.725652 | 0.035860   | 20.24x  | -95.06%        |
| 3840x2160 | float32 | 3        | addition      | scalar | 0.675101 | 0.213446   | 3.16x   | -68.38%        |
| 3840x2160 | float32 | 3        | addition      | sse42  | 0.675101 | 0.040787   | 16.55x  | -93.96%        |
| 3840x2160 | float32 | 3        | addition      | avx2   | 0.675101 | 0.035051   | 19.26x  | -94.81%        |
| 3840x2160 | float32 | 3        | darken_only   | scalar | 0.650858 | 0.114759   | 5.67x   | -82.37%        |
| 3840x2160 | float32 | 3        | darken_only   | sse42  | 0.650858 | 0.037837   | 17.20x  | -94.19%        |
| 3840x2160 | float32 | 3        | darken_only   | avx2   | 0.650858 | 0.034707   | 18.75x  | -94.67%        |
| 3840x2160 | float32 | 3        | multiply      | scalar | 0.673494 | 0.094323   | 7.14x   | -85.99%        |
| 3840x2160 | float32 | 3        | multiply      | sse42  | 0.673494 | 0.038083   | 17.68x  | -94.35%        |
| 3840x2160 | float32 | 3        | multiply      | avx2   | 0.673494 | 0.035202   | 19.13x  | -94.77%        |
| 3840x2160 | float32 | 3        | hard_light    | scalar | 1.002496 | 0.252223   | 3.97x   | -74.84%        |
| 3840x2160 | float32 | 3        | hard_light    | sse42  | 1.002496 | 0.043398   | 23.10x  | -95.67%        |
| 3840x2160 | float32 | 3        | hard_light    | avx2   | 1.002496 | 0.035715   | 28.07x  | -96.44%        |
| 3840x2160 | float32 | 3        | difference    | scalar | 0.894659 | 0.089353   | 10.01x  | -90.01%        |
| 3840x2160 | float32 | 3        | difference    | sse42  | 0.894659 | 0.037912   | 23.60x  | -95.76%        |
| 3840x2160 | float32 | 3        | difference    | avx2   | 0.894659 | 0.033989   | 26.32x  | -96.20%        |
| 3840x2160 | float32 | 3        | subtract      | scalar | 0.666854 | 0.105441   | 6.32x   | -84.19%        |
| 3840x2160 | float32 | 3        | subtract      | sse42  | 0.666854 | 0.040761   | 16.36x  | -93.89%        |
| 3840x2160 | float32 | 3        | subtract      | avx2   | 0.666854 | 0.035838   | 18.61x  | -94.63%        |
| 3840x2160 | float32 | 3        | grain_extract | scalar | 0.682917 | 0.143622   | 4.75x   | -78.97%        |
| 3840x2160 | float32 | 3        | grain_extract | sse42  | 0.682917 | 0.039015   | 17.50x  | -94.29%        |
| 3840x2160 | float32 | 3        | grain_extract | avx2   | 0.682917 | 0.033254   | 20.54x  | -95.13%        |
| 3840x2160 | float32 | 3        | grain_merge   | scalar | 0.683916 | 0.144262   | 4.74x   | -78.91%        |
| 3840x2160 | float32 | 3        | grain_merge   | sse42  | 0.683916 | 0.040379   | 16.94x  | -94.10%        |
| 3840x2160 | float32 | 3        | grain_merge   | avx2   | 0.683916 | 0.034572   | 19.78x  | -94.95%        |
| 3840x2160 | float32 | 3        | divide        | scalar | 0.702717 | 0.103801   | 6.77x   | -85.23%        |
| 3840x2160 | float32 | 3        | divide        | sse42  | 0.702717 | 0.042046   | 16.71x  | -94.02%        |
| 3840x2160 | float32 | 3        | divide        | avx2   | 0.702717 | 0.036041   | 19.50x  | -94.87%        |
| 3840x2160 | float32 | 3        | overlay       | scalar | 0.916593 | 0.238180   | 3.85x   | -74.01%        |
| 3840x2160 | float32 | 3        | overlay       | sse42  | 0.916593 | 0.041581   | 22.04x  | -95.46%        |
| 3840x2160 | float32 | 3        | overlay       | avx2   | 0.916593 | 0.035873   | 25.55x  | -96.09%        |
| 3840x2160 | float32 | 4        | normal        | scalar | 0.523162 | 0.091090   | 5.74x   | -82.59%        |
| 3840x2160 | float32 | 4        | normal        | sse42  | 0.523162 | 0.037705   | 13.88x  | -92.79%        |
| 3840x2160 | float32 | 4        | normal        | avx2   | 0.523162 | 0.033787   | 15.48x  | -93.54%        |
| 3840x2160 | float32 | 4        | soft_light    | scalar | 0.748639 | 0.121953   | 6.14x   | -83.71%        |
| 3840x2160 | float32 | 4        | soft_light    | sse42  | 0.748639 | 0.047687   | 15.70x  | -93.63%        |
| 3840x2160 | float32 | 4        | soft_light    | avx2   | 0.748639 | 0.037961   | 19.72x  | -94.93%        |
| 3840x2160 | float32 | 4        | lighten_only  | scalar | 0.512929 | 0.142375   | 3.60x   | -72.24%        |
| 3840x2160 | float32 | 4        | lighten_only  | sse42  | 0.512929 | 0.041657   | 12.31x  | -91.88%        |
| 3840x2160 | float32 | 4        | lighten_only  | avx2   | 0.512929 | 0.042226   | 12.15x  | -91.77%        |
| 3840x2160 | float32 | 4        | screen        | scalar | 0.554997 | 0.115266   | 4.81x   | -79.23%        |
| 3840x2160 | float32 | 4        | screen        | sse42  | 0.554997 | 0.043786   | 12.68x  | -92.11%        |
| 3840x2160 | float32 | 4        | screen        | avx2   | 0.554997 | 0.041807   | 13.28x  | -92.47%        |
| 3840x2160 | float32 | 4        | dodge         | scalar | 0.545831 | 0.124752   | 4.38x   | -77.14%        |
| 3840x2160 | float32 | 4        | dodge         | sse42  | 0.545831 | 0.045735   | 11.93x  | -91.62%        |
| 3840x2160 | float32 | 4        | dodge         | avx2   | 0.545831 | 0.038545   | 14.16x  | -92.94%        |
| 3840x2160 | float32 | 4        | addition      | scalar | 0.536514 | 0.195159   | 2.75x   | -63.62%        |
| 3840x2160 | float32 | 4        | addition      | sse42  | 0.536514 | 0.044202   | 12.14x  | -91.76%        |
| 3840x2160 | float32 | 4        | addition      | avx2   | 0.536514 | 0.040403   | 13.28x  | -92.47%        |
| 3840x2160 | float32 | 4        | darken_only   | scalar | 0.543601 | 0.143222   | 3.80x   | -73.65%        |
| 3840x2160 | float32 | 4        | darken_only   | sse42  | 0.543601 | 0.043404   | 12.52x  | -92.02%        |
| 3840x2160 | float32 | 4        | darken_only   | avx2   | 0.543601 | 0.041007   | 13.26x  | -92.46%        |
| 3840x2160 | float32 | 4        | multiply      | scalar | 0.554417 | 0.122721   | 4.52x   | -77.86%        |
| 3840x2160 | float32 | 4        | multiply      | sse42  | 0.554417 | 0.041324   | 13.42x  | -92.55%        |
| 3840x2160 | float32 | 4        | multiply      | avx2   | 0.554417 | 0.041115   | 13.48x  | -92.58%        |
| 3840x2160 | float32 | 4        | hard_light    | scalar | 0.898149 | 0.271226   | 3.31x   | -69.80%        |
| 3840x2160 | float32 | 4        | hard_light    | sse42  | 0.898149 | 0.046642   | 19.26x  | -94.81%        |
| 3840x2160 | float32 | 4        | hard_light    | avx2   | 0.898149 | 0.043231   | 20.78x  | -95.19%        |
| 3840x2160 | float32 | 4        | difference    | scalar | 0.768404 | 0.115639   | 6.64x   | -84.95%        |
| 3840x2160 | float32 | 4        | difference    | sse42  | 0.768404 | 0.042735   | 17.98x  | -94.44%        |
| 3840x2160 | float32 | 4        | difference    | avx2   | 0.768404 | 0.041406   | 18.56x  | -94.61%        |
| 3840x2160 | float32 | 4        | subtract      | scalar | 0.541932 | 0.141458   | 3.83x   | -73.90%        |
| 3840x2160 | float32 | 4        | subtract      | sse42  | 0.541932 | 0.045132   | 12.01x  | -91.67%        |
| 3840x2160 | float32 | 4        | subtract      | avx2   | 0.541932 | 0.045222   | 11.98x  | -91.66%        |
| 3840x2160 | float32 | 4        | grain_extract | scalar | 0.570567 | 0.165472   | 3.45x   | -71.00%        |
| 3840x2160 | float32 | 4        | grain_extract | sse42  | 0.570567 | 0.044161   | 12.92x  | -92.26%        |
| 3840x2160 | float32 | 4        | grain_extract | avx2   | 0.570567 | 0.042172   | 13.53x  | -92.61%        |
| 3840x2160 | float32 | 4        | grain_merge   | scalar | 0.556580 | 0.165612   | 3.36x   | -70.24%        |
| 3840x2160 | float32 | 4        | grain_merge   | sse42  | 0.556580 | 0.044573   | 12.49x  | -91.99%        |
| 3840x2160 | float32 | 4        | grain_merge   | avx2   | 0.556580 | 0.043443   | 12.81x  | -92.19%        |
| 3840x2160 | float32 | 4        | divide        | scalar | 0.577240 | 0.119289   | 4.84x   | -79.33%        |
| 3840x2160 | float32 | 4        | divide        | sse42  | 0.577240 | 0.044922   | 12.85x  | -92.22%        |
| 3840x2160 | float32 | 4        | divide        | avx2   | 0.577240 | 0.037836   | 15.26x  | -93.45%        |
| 3840x2160 | float32 | 4        | overlay       | scalar | 0.779919 | 0.247500   | 3.15x   | -68.27%        |
| 3840x2160 | float32 | 4        | overlay       | sse42  | 0.779919 | 0.045646   | 17.09x  | -94.15%        |
| 3840x2160 | float32 | 4        | overlay       | avx2   | 0.779919 | 0.037785   | 20.64x  | -95.16%        |
</details>
<!-- PERF_RESULTS_END -->
