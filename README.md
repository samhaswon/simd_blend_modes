# SIMD Blend Modes

This project reimplements the blend modes from [`blend_modes`](https://github.com/flrs/blend_modes) with C kernels and SIMD
(SSE4.2/AVX2) acceleration. It supports uint8 and float32 NumPy inputs in the range 0..255
and returns output dtype/channel count matching the background image. Missing alpha channels
are treated as fully opaque (255). Opacity defaults to 1.0.

## Build and Install

```bash
python3 -m pip install -e .
```

## Usage

```python
import numpy as np
import simd_blend_modes as sbm

background = np.zeros((512, 512, 4), dtype=np.uint8)
foreground = np.zeros((512, 512, 4), dtype=np.uint8)

out = sbm.screen(background, foreground, 0.5)
```

You can force a kernel by passing a string (or `KernelKind`):

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

## Notes

- SIMD kernels are selected at runtime: AVX2 → SSE4.2 → scalar.
- The reference implementation is not included in this repository (it is ignored by git).
- Reference tests adapted from the original project live in `tests/reference_blend_modes_tests.py`
  and are skipped unless the `blend_modes` package and test assets are available.
- The SIMD paths currently assume contiguous arrays (the input validation enforces this).

## Performance 

| Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| ------------- | ------ | -------- | ---------- | ------- | -------------- |
| normal        | scalar | 0.181753 | 0.042561   | 4.27x   | -76.58%        |
| normal        | sse42  | 0.181753 | 0.012263   | 14.82x  | -93.25%        |
| normal        | avx2   | 0.181753 | 0.009976   | 18.22x  | -94.51%        |
| soft_light    | scalar | 0.259822 | 0.054440   | 4.77x   | -79.05%        |
| soft_light    | sse42  | 0.259822 | 0.013311   | 19.52x  | -94.88%        |
| soft_light    | avx2   | 0.259822 | 0.010868   | 23.91x  | -95.82%        |
| lighten_only  | scalar | 0.182286 | 0.057447   | 3.17x   | -68.49%        |
| lighten_only  | sse42  | 0.182286 | 0.013527   | 13.48x  | -92.58%        |
| lighten_only  | avx2   | 0.182286 | 0.011040   | 16.51x  | -93.94%        |
| screen        | scalar | 0.194978 | 0.053268   | 3.66x   | -72.68%        |
| screen        | sse42  | 0.194978 | 0.013173   | 14.80x  | -93.24%        |
| screen        | avx2   | 0.194978 | 0.010922   | 17.85x  | -94.40%        |
| dodge         | scalar | 0.197374 | 0.055555   | 3.55x   | -71.85%        |
| dodge         | sse42  | 0.197374 | 0.014462   | 13.65x  | -92.67%        |
| dodge         | avx2   | 0.197374 | 0.011400   | 17.31x  | -94.22%        |
| addition      | scalar | 0.187840 | 0.073851   | 2.54x   | -60.68%        |
| addition      | sse42  | 0.187840 | 0.014418   | 13.03x  | -92.32%        |
| addition      | avx2   | 0.187840 | 0.010961   | 17.14x  | -94.16%        |
| darken_only   | scalar | 0.183205 | 0.057978   | 3.16x   | -68.35%        |
| darken_only   | sse42  | 0.183205 | 0.013649   | 13.42x  | -92.55%        |
| darken_only   | avx2   | 0.183205 | 0.011073   | 16.54x  | -93.96%        |
| multiply      | scalar | 0.191881 | 0.051665   | 3.71x   | -73.07%        |
| multiply      | sse42  | 0.191881 | 0.013362   | 14.36x  | -93.04%        |
| multiply      | avx2   | 0.191881 | 0.010950   | 17.52x  | -94.29%        |
| hard_light    | scalar | 0.296249 | 0.097343   | 3.04x   | -67.14%        |
| hard_light    | sse42  | 0.296249 | 0.014372   | 20.61x  | -95.15%        |
| hard_light    | avx2   | 0.296249 | 0.011650   | 25.43x  | -96.07%        |
| difference    | scalar | 0.262670 | 0.051855   | 5.07x   | -80.26%        |
| difference    | sse42  | 0.262670 | 0.013292   | 19.76x  | -94.94%        |
| difference    | avx2   | 0.262670 | 0.010740   | 24.46x  | -95.91%        |
| subtract      | scalar | 0.190169 | 0.057013   | 3.34x   | -70.02%        |
| subtract      | sse42  | 0.190169 | 0.014655   | 12.98x  | -92.29%        |
| subtract      | avx2   | 0.190169 | 0.011425   | 16.64x  | -93.99%        |
| grain_extract | scalar | 0.196944 | 0.066344   | 2.97x   | -66.31%        |
| grain_extract | sse42  | 0.196944 | 0.013337   | 14.77x  | -93.23%        |
| grain_extract | avx2   | 0.196944 | 0.010748   | 18.32x  | -94.54%        |
| grain_merge   | scalar | 0.199405 | 0.067343   | 2.96x   | -66.23%        |
| grain_merge   | sse42  | 0.199405 | 0.014352   | 13.89x  | -92.80%        |
| grain_merge   | avx2   | 0.199405 | 0.012969   | 15.38x  | -93.50%        |
| divide        | scalar | 0.202540 | 0.054078   | 3.75x   | -73.30%        |
| divide        | sse42  | 0.202540 | 0.013108   | 15.45x  | -93.53%        |
| divide        | avx2   | 0.202540 | 0.011001   | 18.41x  | -94.57%        |
| overlay       | scalar | 0.273305 | 0.092819   | 2.94x   | -66.04%        |
| overlay       | sse42  | 0.273305 | 0.013874   | 19.70x  | -94.92%        |
| overlay       | avx2   | 0.273305 | 0.011919   | 22.93x  | -95.64%        |

<details>
<summary>Per-kernel results</summary>

| Case      | Input   | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | normal        | scalar | 0.008150 | 0.002548   | 3.20x   | -68.73%        |
| 256x256   | uint8   | normal        | sse42  | 0.008150 | 0.000199   | 40.92x  | -97.56%        |
| 256x256   | uint8   | normal        | avx2   | 0.008150 | 0.000162   | 50.17x  | -98.01%        |
| 256x256   | uint8   | soft_light    | scalar | 0.013957 | 0.001938   | 7.20x   | -86.11%        |
| 256x256   | uint8   | soft_light    | sse42  | 0.013957 | 0.000266   | 52.51x  | -98.10%        |
| 256x256   | uint8   | soft_light    | avx2   | 0.013957 | 0.000185   | 75.29x  | -98.67%        |
| 256x256   | uint8   | lighten_only  | scalar | 0.009071 | 0.001887   | 4.81x   | -79.20%        |
| 256x256   | uint8   | lighten_only  | sse42  | 0.009071 | 0.000224   | 40.50x  | -97.53%        |
| 256x256   | uint8   | lighten_only  | avx2   | 0.009071 | 0.000184   | 49.21x  | -97.97%        |
| 256x256   | uint8   | screen        | scalar | 0.009585 | 0.002237   | 4.28x   | -76.66%        |
| 256x256   | uint8   | screen        | sse42  | 0.009585 | 0.000235   | 40.87x  | -97.55%        |
| 256x256   | uint8   | screen        | avx2   | 0.009585 | 0.000186   | 51.42x  | -98.06%        |
| 256x256   | uint8   | dodge         | scalar | 0.009668 | 0.002002   | 4.83x   | -79.29%        |
| 256x256   | uint8   | dodge         | sse42  | 0.009668 | 0.000249   | 38.90x  | -97.43%        |
| 256x256   | uint8   | dodge         | avx2   | 0.009668 | 0.000184   | 52.62x  | -98.10%        |
| 256x256   | uint8   | addition      | scalar | 0.009297 | 0.002174   | 4.28x   | -76.61%        |
| 256x256   | uint8   | addition      | sse42  | 0.009297 | 0.000267   | 34.87x  | -97.13%        |
| 256x256   | uint8   | addition      | avx2   | 0.009297 | 0.000197   | 47.08x  | -97.88%        |
| 256x256   | uint8   | darken_only   | scalar | 0.009238 | 0.001851   | 4.99x   | -79.96%        |
| 256x256   | uint8   | darken_only   | sse42  | 0.009238 | 0.000219   | 42.27x  | -97.63%        |
| 256x256   | uint8   | darken_only   | avx2   | 0.009238 | 0.000182   | 50.84x  | -98.03%        |
| 256x256   | uint8   | multiply      | scalar | 0.008717 | 0.001788   | 4.88x   | -79.49%        |
| 256x256   | uint8   | multiply      | sse42  | 0.008717 | 0.000241   | 36.18x  | -97.24%        |
| 256x256   | uint8   | multiply      | avx2   | 0.008717 | 0.000181   | 48.12x  | -97.92%        |
| 256x256   | uint8   | hard_light    | scalar | 0.011106 | 0.002805   | 3.96x   | -74.75%        |
| 256x256   | uint8   | hard_light    | sse42  | 0.011106 | 0.000260   | 42.79x  | -97.66%        |
| 256x256   | uint8   | hard_light    | avx2   | 0.011106 | 0.000209   | 53.01x  | -98.11%        |
| 256x256   | uint8   | difference    | scalar | 0.010902 | 0.001953   | 5.58x   | -82.09%        |
| 256x256   | uint8   | difference    | sse42  | 0.010902 | 0.000224   | 48.73x  | -97.95%        |
| 256x256   | uint8   | difference    | avx2   | 0.010902 | 0.000182   | 59.82x  | -98.33%        |
| 256x256   | uint8   | subtract      | scalar | 0.008983 | 0.001722   | 5.22x   | -80.83%        |
| 256x256   | uint8   | subtract      | sse42  | 0.008983 | 0.000263   | 34.16x  | -97.07%        |
| 256x256   | uint8   | subtract      | avx2   | 0.008983 | 0.000199   | 45.21x  | -97.79%        |
| 256x256   | uint8   | grain_extract | scalar | 0.009473 | 0.002011   | 4.71x   | -78.77%        |
| 256x256   | uint8   | grain_extract | sse42  | 0.009473 | 0.000261   | 36.30x  | -97.25%        |
| 256x256   | uint8   | grain_extract | avx2   | 0.009473 | 0.000186   | 50.93x  | -98.04%        |
| 256x256   | uint8   | grain_merge   | scalar | 0.010380 | 0.002316   | 4.48x   | -77.69%        |
| 256x256   | uint8   | grain_merge   | sse42  | 0.010380 | 0.000341   | 30.41x  | -96.71%        |
| 256x256   | uint8   | grain_merge   | avx2   | 0.010380 | 0.000191   | 54.40x  | -98.16%        |
| 256x256   | uint8   | divide        | scalar | 0.010337 | 0.001820   | 5.68x   | -82.39%        |
| 256x256   | uint8   | divide        | sse42  | 0.010337 | 0.000241   | 42.88x  | -97.67%        |
| 256x256   | uint8   | divide        | avx2   | 0.010337 | 0.000182   | 56.74x  | -98.24%        |
| 256x256   | uint8   | overlay       | scalar | 0.010497 | 0.002657   | 3.95x   | -74.69%        |
| 256x256   | uint8   | overlay       | sse42  | 0.010497 | 0.000247   | 42.42x  | -97.64%        |
| 256x256   | uint8   | overlay       | avx2   | 0.010497 | 0.000217   | 48.26x  | -97.93%        |
| 256x256   | float32 | normal        | scalar | 0.005642 | 0.000730   | 7.73x   | -87.07%        |
| 256x256   | float32 | normal        | sse42  | 0.005642 | 0.000351   | 16.07x  | -93.78%        |
| 256x256   | float32 | normal        | avx2   | 0.005642 | 0.000227   | 24.81x  | -95.97%        |
| 256x256   | float32 | soft_light    | scalar | 0.009031 | 0.000908   | 9.94x   | -89.94%        |
| 256x256   | float32 | soft_light    | sse42  | 0.009031 | 0.000355   | 25.44x  | -96.07%        |
| 256x256   | float32 | soft_light    | avx2   | 0.009031 | 0.000261   | 34.63x  | -97.11%        |
| 256x256   | float32 | lighten_only  | scalar | 0.007168 | 0.000991   | 7.23x   | -86.18%        |
| 256x256   | float32 | lighten_only  | sse42  | 0.007168 | 0.000809   | 8.86x   | -88.72%        |
| 256x256   | float32 | lighten_only  | avx2   | 0.007168 | 0.000254   | 28.26x  | -96.46%        |
| 256x256   | float32 | screen        | scalar | 0.006918 | 0.000869   | 7.96x   | -87.44%        |
| 256x256   | float32 | screen        | sse42  | 0.006918 | 0.000385   | 17.99x  | -94.44%        |
| 256x256   | float32 | screen        | avx2   | 0.006918 | 0.000256   | 27.05x  | -96.30%        |
| 256x256   | float32 | dodge         | scalar | 0.007701 | 0.000925   | 8.33x   | -87.99%        |
| 256x256   | float32 | dodge         | sse42  | 0.007701 | 0.000368   | 20.90x  | -95.22%        |
| 256x256   | float32 | dodge         | avx2   | 0.007701 | 0.000257   | 29.97x  | -96.66%        |
| 256x256   | float32 | addition      | scalar | 0.007446 | 0.001636   | 4.55x   | -78.03%        |
| 256x256   | float32 | addition      | sse42  | 0.007446 | 0.000382   | 19.51x  | -94.87%        |
| 256x256   | float32 | addition      | avx2   | 0.007446 | 0.000359   | 20.76x  | -95.18%        |
| 256x256   | float32 | darken_only   | scalar | 0.007149 | 0.001192   | 6.00x   | -83.32%        |
| 256x256   | float32 | darken_only   | sse42  | 0.007149 | 0.000373   | 19.16x  | -94.78%        |
| 256x256   | float32 | darken_only   | avx2   | 0.007149 | 0.000250   | 28.59x  | -96.50%        |
| 256x256   | float32 | multiply      | scalar | 0.007464 | 0.000868   | 8.60x   | -88.37%        |
| 256x256   | float32 | multiply      | sse42  | 0.007464 | 0.000369   | 20.25x  | -95.06%        |
| 256x256   | float32 | multiply      | avx2   | 0.007464 | 0.000254   | 29.44x  | -96.60%        |
| 256x256   | float32 | hard_light    | scalar | 0.009224 | 0.002068   | 4.46x   | -77.58%        |
| 256x256   | float32 | hard_light    | sse42  | 0.009224 | 0.000357   | 25.83x  | -96.13%        |
| 256x256   | float32 | hard_light    | avx2   | 0.009224 | 0.000285   | 32.32x  | -96.91%        |
| 256x256   | float32 | difference    | scalar | 0.010072 | 0.000936   | 10.76x  | -90.71%        |
| 256x256   | float32 | difference    | sse42  | 0.010072 | 0.000362   | 27.79x  | -96.40%        |
| 256x256   | float32 | difference    | avx2   | 0.010072 | 0.000280   | 35.94x  | -97.22%        |
| 256x256   | float32 | subtract      | scalar | 0.007686 | 0.001090   | 7.05x   | -85.82%        |
| 256x256   | float32 | subtract      | sse42  | 0.007686 | 0.000404   | 19.02x  | -94.74%        |
| 256x256   | float32 | subtract      | avx2   | 0.007686 | 0.000265   | 28.97x  | -96.55%        |
| 256x256   | float32 | grain_extract | scalar | 0.007362 | 0.001249   | 5.89x   | -83.03%        |
| 256x256   | float32 | grain_extract | sse42  | 0.007362 | 0.000421   | 17.48x  | -94.28%        |
| 256x256   | float32 | grain_extract | avx2   | 0.007362 | 0.000258   | 28.54x  | -96.50%        |
| 256x256   | float32 | grain_merge   | scalar | 0.007621 | 0.001226   | 6.22x   | -83.91%        |
| 256x256   | float32 | grain_merge   | sse42  | 0.007621 | 0.000381   | 20.01x  | -95.00%        |
| 256x256   | float32 | grain_merge   | avx2   | 0.007621 | 0.000251   | 30.36x  | -96.71%        |
| 256x256   | float32 | divide        | scalar | 0.007509 | 0.000950   | 7.90x   | -87.35%        |
| 256x256   | float32 | divide        | sse42  | 0.007509 | 0.000361   | 20.78x  | -95.19%        |
| 256x256   | float32 | divide        | avx2   | 0.007509 | 0.000255   | 29.46x  | -96.61%        |
| 256x256   | float32 | overlay       | scalar | 0.008612 | 0.002007   | 4.29x   | -76.70%        |
| 256x256   | float32 | overlay       | sse42  | 0.008612 | 0.000372   | 23.14x  | -95.68%        |
| 256x256   | float32 | overlay       | avx2   | 0.008612 | 0.000276   | 31.25x  | -96.80%        |
| 512x512   | uint8   | normal        | scalar | 0.027838 | 0.005963   | 4.67x   | -78.58%        |
| 512x512   | uint8   | normal        | sse42  | 0.027838 | 0.000771   | 36.10x  | -97.23%        |
| 512x512   | uint8   | normal        | avx2   | 0.027838 | 0.000646   | 43.12x  | -97.68%        |
| 512x512   | uint8   | soft_light    | scalar | 0.039536 | 0.008199   | 4.82x   | -79.26%        |
| 512x512   | uint8   | soft_light    | sse42  | 0.039536 | 0.000925   | 42.76x  | -97.66%        |
| 512x512   | uint8   | soft_light    | avx2   | 0.039536 | 0.000788   | 50.15x  | -98.01%        |
| 512x512   | uint8   | lighten_only  | scalar | 0.031483 | 0.007482   | 4.21x   | -76.23%        |
| 512x512   | uint8   | lighten_only  | sse42  | 0.031483 | 0.000881   | 35.72x  | -97.20%        |
| 512x512   | uint8   | lighten_only  | avx2   | 0.031483 | 0.000717   | 43.88x  | -97.72%        |
| 512x512   | uint8   | screen        | scalar | 0.033079 | 0.008002   | 4.13x   | -75.81%        |
| 512x512   | uint8   | screen        | sse42  | 0.033079 | 0.000901   | 36.73x  | -97.28%        |
| 512x512   | uint8   | screen        | avx2   | 0.033079 | 0.000729   | 45.38x  | -97.80%        |
| 512x512   | uint8   | dodge         | scalar | 0.032181 | 0.007567   | 4.25x   | -76.49%        |
| 512x512   | uint8   | dodge         | sse42  | 0.032181 | 0.001080   | 29.81x  | -96.65%        |
| 512x512   | uint8   | dodge         | avx2   | 0.032181 | 0.000736   | 43.70x  | -97.71%        |
| 512x512   | uint8   | addition      | scalar | 0.031813 | 0.008615   | 3.69x   | -72.92%        |
| 512x512   | uint8   | addition      | sse42  | 0.031813 | 0.001066   | 29.83x  | -96.65%        |
| 512x512   | uint8   | addition      | avx2   | 0.031813 | 0.000859   | 37.04x  | -97.30%        |
| 512x512   | uint8   | darken_only   | scalar | 0.030386 | 0.007789   | 3.90x   | -74.37%        |
| 512x512   | uint8   | darken_only   | sse42  | 0.030386 | 0.000877   | 34.63x  | -97.11%        |
| 512x512   | uint8   | darken_only   | avx2   | 0.030386 | 0.000717   | 42.41x  | -97.64%        |
| 512x512   | uint8   | multiply      | scalar | 0.031704 | 0.007335   | 4.32x   | -76.86%        |
| 512x512   | uint8   | multiply      | sse42  | 0.031704 | 0.000899   | 35.27x  | -97.16%        |
| 512x512   | uint8   | multiply      | avx2   | 0.031704 | 0.000853   | 37.15x  | -97.31%        |
| 512x512   | uint8   | hard_light    | scalar | 0.041851 | 0.011386   | 3.68x   | -72.79%        |
| 512x512   | uint8   | hard_light    | sse42  | 0.041851 | 0.001046   | 40.00x  | -97.50%        |
| 512x512   | uint8   | hard_light    | avx2   | 0.041851 | 0.000817   | 51.20x  | -98.05%        |
| 512x512   | uint8   | difference    | scalar | 0.039233 | 0.007211   | 5.44x   | -81.62%        |
| 512x512   | uint8   | difference    | sse42  | 0.039233 | 0.000876   | 44.77x  | -97.77%        |
| 512x512   | uint8   | difference    | avx2   | 0.039233 | 0.000714   | 54.93x  | -98.18%        |
| 512x512   | uint8   | subtract      | scalar | 0.032368 | 0.006886   | 4.70x   | -78.73%        |
| 512x512   | uint8   | subtract      | sse42  | 0.032368 | 0.001155   | 28.02x  | -96.43%        |
| 512x512   | uint8   | subtract      | avx2   | 0.032368 | 0.000787   | 41.11x  | -97.57%        |
| 512x512   | uint8   | grain_extract | scalar | 0.032297 | 0.008433   | 3.83x   | -73.89%        |
| 512x512   | uint8   | grain_extract | sse42  | 0.032297 | 0.000946   | 34.13x  | -97.07%        |
| 512x512   | uint8   | grain_extract | avx2   | 0.032297 | 0.000953   | 33.89x  | -97.05%        |
| 512x512   | uint8   | grain_merge   | scalar | 0.032043 | 0.008415   | 3.81x   | -73.74%        |
| 512x512   | uint8   | grain_merge   | sse42  | 0.032043 | 0.000903   | 35.49x  | -97.18%        |
| 512x512   | uint8   | grain_merge   | avx2   | 0.032043 | 0.000794   | 40.38x  | -97.52%        |
| 512x512   | uint8   | divide        | scalar | 0.032698 | 0.007667   | 4.26x   | -76.55%        |
| 512x512   | uint8   | divide        | sse42  | 0.032698 | 0.000971   | 33.68x  | -97.03%        |
| 512x512   | uint8   | divide        | avx2   | 0.032698 | 0.000784   | 41.69x  | -97.60%        |
| 512x512   | uint8   | overlay       | scalar | 0.039729 | 0.010961   | 3.62x   | -72.41%        |
| 512x512   | uint8   | overlay       | sse42  | 0.039729 | 0.000980   | 40.55x  | -97.53%        |
| 512x512   | uint8   | overlay       | avx2   | 0.039729 | 0.000736   | 54.00x  | -98.15%        |
| 512x512   | float32 | normal        | scalar | 0.022677 | 0.003431   | 6.61x   | -84.87%        |
| 512x512   | float32 | normal        | sse42  | 0.022677 | 0.001266   | 17.92x  | -94.42%        |
| 512x512   | float32 | normal        | avx2   | 0.022677 | 0.001543   | 14.70x  | -93.20%        |
| 512x512   | float32 | soft_light    | scalar | 0.035186 | 0.003730   | 9.43x   | -89.40%        |
| 512x512   | float32 | soft_light    | sse42  | 0.035186 | 0.001427   | 24.65x  | -95.94%        |
| 512x512   | float32 | soft_light    | avx2   | 0.035186 | 0.001169   | 30.09x  | -96.68%        |
| 512x512   | float32 | lighten_only  | scalar | 0.029005 | 0.004271   | 6.79x   | -85.27%        |
| 512x512   | float32 | lighten_only  | sse42  | 0.029005 | 0.001521   | 19.07x  | -94.76%        |
| 512x512   | float32 | lighten_only  | avx2   | 0.029005 | 0.001071   | 27.07x  | -96.31%        |
| 512x512   | float32 | screen        | scalar | 0.029864 | 0.003806   | 7.85x   | -87.26%        |
| 512x512   | float32 | screen        | sse42  | 0.029864 | 0.001496   | 19.96x  | -94.99%        |
| 512x512   | float32 | screen        | avx2   | 0.029864 | 0.001091   | 27.38x  | -96.35%        |
| 512x512   | float32 | dodge         | scalar | 0.028960 | 0.004008   | 7.22x   | -86.16%        |
| 512x512   | float32 | dodge         | sse42  | 0.028960 | 0.001528   | 18.95x  | -94.72%        |
| 512x512   | float32 | dodge         | avx2   | 0.028960 | 0.001132   | 25.59x  | -96.09%        |
| 512x512   | float32 | addition      | scalar | 0.028241 | 0.006440   | 4.39x   | -77.20%        |
| 512x512   | float32 | addition      | sse42  | 0.028241 | 0.001583   | 17.84x  | -94.39%        |
| 512x512   | float32 | addition      | avx2   | 0.028241 | 0.001135   | 24.88x  | -95.98%        |
| 512x512   | float32 | darken_only   | scalar | 0.028804 | 0.004209   | 6.84x   | -85.39%        |
| 512x512   | float32 | darken_only   | sse42  | 0.028804 | 0.001493   | 19.29x  | -94.82%        |
| 512x512   | float32 | darken_only   | avx2   | 0.028804 | 0.001269   | 22.71x  | -95.60%        |
| 512x512   | float32 | multiply      | scalar | 0.028578 | 0.003310   | 8.63x   | -88.42%        |
| 512x512   | float32 | multiply      | sse42  | 0.028578 | 0.001527   | 18.72x  | -94.66%        |
| 512x512   | float32 | multiply      | avx2   | 0.028578 | 0.001105   | 25.85x  | -96.13%        |
| 512x512   | float32 | hard_light    | scalar | 0.037625 | 0.008809   | 4.27x   | -76.59%        |
| 512x512   | float32 | hard_light    | sse42  | 0.037625 | 0.001474   | 25.53x  | -96.08%        |
| 512x512   | float32 | hard_light    | avx2   | 0.037625 | 0.001066   | 35.29x  | -97.17%        |
| 512x512   | float32 | difference    | scalar | 0.035592 | 0.003414   | 10.42x  | -90.41%        |
| 512x512   | float32 | difference    | sse42  | 0.035592 | 0.001448   | 24.59x  | -95.93%        |
| 512x512   | float32 | difference    | avx2   | 0.035592 | 0.001187   | 29.99x  | -96.67%        |
| 512x512   | float32 | subtract      | scalar | 0.027382 | 0.004334   | 6.32x   | -84.17%        |
| 512x512   | float32 | subtract      | sse42  | 0.027382 | 0.001533   | 17.86x  | -94.40%        |
| 512x512   | float32 | subtract      | avx2   | 0.027382 | 0.001148   | 23.85x  | -95.81%        |
| 512x512   | float32 | grain_extract | scalar | 0.028166 | 0.004909   | 5.74x   | -82.57%        |
| 512x512   | float32 | grain_extract | sse42  | 0.028166 | 0.001461   | 19.28x  | -94.81%        |
| 512x512   | float32 | grain_extract | avx2   | 0.028166 | 0.001102   | 25.57x  | -96.09%        |
| 512x512   | float32 | grain_merge   | scalar | 0.028441 | 0.005001   | 5.69x   | -82.42%        |
| 512x512   | float32 | grain_merge   | sse42  | 0.028441 | 0.001547   | 18.38x  | -94.56%        |
| 512x512   | float32 | grain_merge   | avx2   | 0.028441 | 0.001103   | 25.79x  | -96.12%        |
| 512x512   | float32 | divide        | scalar | 0.028385 | 0.003574   | 7.94x   | -87.41%        |
| 512x512   | float32 | divide        | sse42  | 0.028385 | 0.001507   | 18.83x  | -94.69%        |
| 512x512   | float32 | divide        | avx2   | 0.028385 | 0.001090   | 26.05x  | -96.16%        |
| 512x512   | float32 | overlay       | scalar | 0.036094 | 0.008011   | 4.51x   | -77.80%        |
| 512x512   | float32 | overlay       | sse42  | 0.036094 | 0.001587   | 22.75x  | -95.60%        |
| 512x512   | float32 | overlay       | avx2   | 0.036094 | 0.001125   | 32.09x  | -96.88%        |
| 1024x1024 | uint8   | normal        | scalar | 0.079314 | 0.022340   | 3.55x   | -71.83%        |
| 1024x1024 | uint8   | normal        | sse42  | 0.079314 | 0.003123   | 25.39x  | -96.06%        |
| 1024x1024 | uint8   | normal        | avx2   | 0.079314 | 0.002569   | 30.87x  | -96.76%        |
| 1024x1024 | uint8   | soft_light    | scalar | 0.113452 | 0.029466   | 3.85x   | -74.03%        |
| 1024x1024 | uint8   | soft_light    | sse42  | 0.113452 | 0.004018   | 28.23x  | -96.46%        |
| 1024x1024 | uint8   | soft_light    | avx2   | 0.113452 | 0.003097   | 36.64x  | -97.27%        |
| 1024x1024 | uint8   | lighten_only  | scalar | 0.093977 | 0.029701   | 3.16x   | -68.40%        |
| 1024x1024 | uint8   | lighten_only  | sse42  | 0.093977 | 0.003904   | 24.07x  | -95.85%        |
| 1024x1024 | uint8   | lighten_only  | avx2   | 0.093977 | 0.002869   | 32.76x  | -96.95%        |
| 1024x1024 | uint8   | screen        | scalar | 0.090318 | 0.028534   | 3.17x   | -68.41%        |
| 1024x1024 | uint8   | screen        | sse42  | 0.090318 | 0.004494   | 20.10x  | -95.02%        |
| 1024x1024 | uint8   | screen        | avx2   | 0.090318 | 0.002934   | 30.79x  | -96.75%        |
| 1024x1024 | uint8   | dodge         | scalar | 0.090061 | 0.029026   | 3.10x   | -67.77%        |
| 1024x1024 | uint8   | dodge         | sse42  | 0.090061 | 0.004030   | 22.35x  | -95.52%        |
| 1024x1024 | uint8   | dodge         | avx2   | 0.090061 | 0.002994   | 30.08x  | -96.68%        |
| 1024x1024 | uint8   | addition      | scalar | 0.082457 | 0.034355   | 2.40x   | -58.34%        |
| 1024x1024 | uint8   | addition      | sse42  | 0.082457 | 0.004244   | 19.43x  | -94.85%        |
| 1024x1024 | uint8   | addition      | avx2   | 0.082457 | 0.003156   | 26.13x  | -96.17%        |
| 1024x1024 | uint8   | darken_only   | scalar | 0.082557 | 0.029265   | 2.82x   | -64.55%        |
| 1024x1024 | uint8   | darken_only   | sse42  | 0.082557 | 0.003479   | 23.73x  | -95.79%        |
| 1024x1024 | uint8   | darken_only   | avx2   | 0.082557 | 0.002874   | 28.73x  | -96.52%        |
| 1024x1024 | uint8   | multiply      | scalar | 0.083284 | 0.027533   | 3.02x   | -66.94%        |
| 1024x1024 | uint8   | multiply      | sse42  | 0.083284 | 0.003669   | 22.70x  | -95.59%        |
| 1024x1024 | uint8   | multiply      | avx2   | 0.083284 | 0.003004   | 27.73x  | -96.39%        |
| 1024x1024 | uint8   | hard_light    | scalar | 0.121079 | 0.045818   | 2.64x   | -62.16%        |
| 1024x1024 | uint8   | hard_light    | sse42  | 0.121079 | 0.004165   | 29.07x  | -96.56%        |
| 1024x1024 | uint8   | hard_light    | avx2   | 0.121079 | 0.003060   | 39.57x  | -97.47%        |
| 1024x1024 | uint8   | difference    | scalar | 0.119055 | 0.027204   | 4.38x   | -77.15%        |
| 1024x1024 | uint8   | difference    | sse42  | 0.119055 | 0.003815   | 31.20x  | -96.80%        |
| 1024x1024 | uint8   | difference    | avx2   | 0.119055 | 0.002958   | 40.25x  | -97.52%        |
| 1024x1024 | uint8   | subtract      | scalar | 0.084602 | 0.028256   | 2.99x   | -66.60%        |
| 1024x1024 | uint8   | subtract      | sse42  | 0.084602 | 0.004328   | 19.55x  | -94.88%        |
| 1024x1024 | uint8   | subtract      | avx2   | 0.084602 | 0.003306   | 25.59x  | -96.09%        |
| 1024x1024 | uint8   | grain_extract | scalar | 0.086021 | 0.032783   | 2.62x   | -61.89%        |
| 1024x1024 | uint8   | grain_extract | sse42  | 0.086021 | 0.003683   | 23.36x  | -95.72%        |
| 1024x1024 | uint8   | grain_extract | avx2   | 0.086021 | 0.002971   | 28.95x  | -96.55%        |
| 1024x1024 | uint8   | grain_merge   | scalar | 0.085939 | 0.033633   | 2.56x   | -60.86%        |
| 1024x1024 | uint8   | grain_merge   | sse42  | 0.085939 | 0.003877   | 22.17x  | -95.49%        |
| 1024x1024 | uint8   | grain_merge   | avx2   | 0.085939 | 0.003100   | 27.73x  | -96.39%        |
| 1024x1024 | uint8   | divide        | scalar | 0.103324 | 0.028416   | 3.64x   | -72.50%        |
| 1024x1024 | uint8   | divide        | sse42  | 0.103324 | 0.004097   | 25.22x  | -96.03%        |
| 1024x1024 | uint8   | divide        | avx2   | 0.103324 | 0.003035   | 34.05x  | -97.06%        |
| 1024x1024 | uint8   | overlay       | scalar | 0.117333 | 0.042911   | 2.73x   | -63.43%        |
| 1024x1024 | uint8   | overlay       | sse42  | 0.117333 | 0.003924   | 29.90x  | -96.66%        |
| 1024x1024 | uint8   | overlay       | avx2   | 0.117333 | 0.003038   | 38.62x  | -97.41%        |
| 1024x1024 | float32 | normal        | scalar | 0.069840 | 0.010973   | 6.36x   | -84.29%        |
| 1024x1024 | float32 | normal        | sse42  | 0.069840 | 0.005444   | 12.83x  | -92.21%        |
| 1024x1024 | float32 | normal        | avx2   | 0.069840 | 0.004333   | 16.12x  | -93.80%        |
| 1024x1024 | float32 | soft_light    | scalar | 0.102372 | 0.014610   | 7.01x   | -85.73%        |
| 1024x1024 | float32 | soft_light    | sse42  | 0.102372 | 0.005729   | 17.87x  | -94.40%        |
| 1024x1024 | float32 | soft_light    | avx2   | 0.102372 | 0.004675   | 21.90x  | -95.43%        |
| 1024x1024 | float32 | lighten_only  | scalar | 0.080974 | 0.016510   | 4.90x   | -79.61%        |
| 1024x1024 | float32 | lighten_only  | sse42  | 0.080974 | 0.006191   | 13.08x  | -92.35%        |
| 1024x1024 | float32 | lighten_only  | avx2   | 0.080974 | 0.004981   | 16.26x  | -93.85%        |
| 1024x1024 | float32 | screen        | scalar | 0.078883 | 0.013683   | 5.77x   | -82.65%        |
| 1024x1024 | float32 | screen        | sse42  | 0.078883 | 0.005926   | 13.31x  | -92.49%        |
| 1024x1024 | float32 | screen        | avx2   | 0.078883 | 0.005034   | 15.67x  | -93.62%        |
| 1024x1024 | float32 | dodge         | scalar | 0.077752 | 0.014439   | 5.38x   | -81.43%        |
| 1024x1024 | float32 | dodge         | sse42  | 0.077752 | 0.006150   | 12.64x  | -92.09%        |
| 1024x1024 | float32 | dodge         | avx2   | 0.077752 | 0.004805   | 16.18x  | -93.82%        |
| 1024x1024 | float32 | addition      | scalar | 0.075100 | 0.025082   | 2.99x   | -66.60%        |
| 1024x1024 | float32 | addition      | sse42  | 0.075100 | 0.006204   | 12.10x  | -91.74%        |
| 1024x1024 | float32 | addition      | avx2   | 0.075100 | 0.004706   | 15.96x  | -93.73%        |
| 1024x1024 | float32 | darken_only   | scalar | 0.075769 | 0.015633   | 4.85x   | -79.37%        |
| 1024x1024 | float32 | darken_only   | sse42  | 0.075769 | 0.006003   | 12.62x  | -92.08%        |
| 1024x1024 | float32 | darken_only   | avx2   | 0.075769 | 0.005024   | 15.08x  | -93.37%        |
| 1024x1024 | float32 | multiply      | scalar | 0.077350 | 0.012449   | 6.21x   | -83.91%        |
| 1024x1024 | float32 | multiply      | sse42  | 0.077350 | 0.006071   | 12.74x  | -92.15%        |
| 1024x1024 | float32 | multiply      | avx2   | 0.077350 | 0.004809   | 16.08x  | -93.78%        |
| 1024x1024 | float32 | hard_light    | scalar | 0.113261 | 0.032564   | 3.48x   | -71.25%        |
| 1024x1024 | float32 | hard_light    | sse42  | 0.113261 | 0.005734   | 19.75x  | -94.94%        |
| 1024x1024 | float32 | hard_light    | avx2   | 0.113261 | 0.004743   | 23.88x  | -95.81%        |
| 1024x1024 | float32 | difference    | scalar | 0.107709 | 0.013276   | 8.11x   | -87.67%        |
| 1024x1024 | float32 | difference    | sse42  | 0.107709 | 0.005872   | 18.34x  | -94.55%        |
| 1024x1024 | float32 | difference    | avx2   | 0.107709 | 0.004804   | 22.42x  | -95.54%        |
| 1024x1024 | float32 | subtract      | scalar | 0.077570 | 0.017137   | 4.53x   | -77.91%        |
| 1024x1024 | float32 | subtract      | sse42  | 0.077570 | 0.006199   | 12.51x  | -92.01%        |
| 1024x1024 | float32 | subtract      | avx2   | 0.077570 | 0.004523   | 17.15x  | -94.17%        |
| 1024x1024 | float32 | grain_extract | scalar | 0.079115 | 0.019427   | 4.07x   | -75.44%        |
| 1024x1024 | float32 | grain_extract | sse42  | 0.079115 | 0.005784   | 13.68x  | -92.69%        |
| 1024x1024 | float32 | grain_extract | avx2   | 0.079115 | 0.005040   | 15.70x  | -93.63%        |
| 1024x1024 | float32 | grain_merge   | scalar | 0.076644 | 0.019268   | 3.98x   | -74.86%        |
| 1024x1024 | float32 | grain_merge   | sse42  | 0.076644 | 0.005940   | 12.90x  | -92.25%        |
| 1024x1024 | float32 | grain_merge   | avx2   | 0.076644 | 0.004862   | 15.76x  | -93.66%        |
| 1024x1024 | float32 | divide        | scalar | 0.079326 | 0.013790   | 5.75x   | -82.62%        |
| 1024x1024 | float32 | divide        | sse42  | 0.079326 | 0.006111   | 12.98x  | -92.30%        |
| 1024x1024 | float32 | divide        | avx2   | 0.079326 | 0.004700   | 16.88x  | -94.07%        |
| 1024x1024 | float32 | overlay       | scalar | 0.106525 | 0.030804   | 3.46x   | -71.08%        |
| 1024x1024 | float32 | overlay       | sse42  | 0.106525 | 0.005907   | 18.03x  | -94.45%        |
| 1024x1024 | float32 | overlay       | avx2   | 0.106525 | 0.004954   | 21.50x  | -95.35%        |
| 2048x2048 | uint8   | normal        | scalar | 0.298142 | 0.089601   | 3.33x   | -69.95%        |
| 2048x2048 | uint8   | normal        | sse42  | 0.298142 | 0.012760   | 23.36x  | -95.72%        |
| 2048x2048 | uint8   | normal        | avx2   | 0.298142 | 0.010264   | 29.05x  | -96.56%        |
| 2048x2048 | uint8   | soft_light    | scalar | 0.425814 | 0.113987   | 3.74x   | -73.23%        |
| 2048x2048 | uint8   | soft_light    | sse42  | 0.425814 | 0.018108   | 23.52x  | -95.75%        |
| 2048x2048 | uint8   | soft_light    | avx2   | 0.425814 | 0.012648   | 33.67x  | -97.03%        |
| 2048x2048 | uint8   | lighten_only  | scalar | 0.296520 | 0.119641   | 2.48x   | -59.65%        |
| 2048x2048 | uint8   | lighten_only  | sse42  | 0.296520 | 0.013817   | 21.46x  | -95.34%        |
| 2048x2048 | uint8   | lighten_only  | avx2   | 0.296520 | 0.011624   | 25.51x  | -96.08%        |
| 2048x2048 | uint8   | screen        | scalar | 0.318399 | 0.110927   | 2.87x   | -65.16%        |
| 2048x2048 | uint8   | screen        | sse42  | 0.318399 | 0.014584   | 21.83x  | -95.42%        |
| 2048x2048 | uint8   | screen        | avx2   | 0.318399 | 0.012063   | 26.39x  | -96.21%        |
| 2048x2048 | uint8   | dodge         | scalar | 0.320656 | 0.113524   | 2.82x   | -64.60%        |
| 2048x2048 | uint8   | dodge         | sse42  | 0.320656 | 0.016897   | 18.98x  | -94.73%        |
| 2048x2048 | uint8   | dodge         | avx2   | 0.320656 | 0.012586   | 25.48x  | -96.07%        |
| 2048x2048 | uint8   | addition      | scalar | 0.307764 | 0.137705   | 2.23x   | -55.26%        |
| 2048x2048 | uint8   | addition      | sse42  | 0.307764 | 0.017101   | 18.00x  | -94.44%        |
| 2048x2048 | uint8   | addition      | avx2   | 0.307764 | 0.012530   | 24.56x  | -95.93%        |
| 2048x2048 | uint8   | darken_only   | scalar | 0.296146 | 0.118617   | 2.50x   | -59.95%        |
| 2048x2048 | uint8   | darken_only   | sse42  | 0.296146 | 0.014954   | 19.80x  | -94.95%        |
| 2048x2048 | uint8   | darken_only   | avx2   | 0.296146 | 0.011500   | 25.75x  | -96.12%        |
| 2048x2048 | uint8   | multiply      | scalar | 0.313756 | 0.112273   | 2.79x   | -64.22%        |
| 2048x2048 | uint8   | multiply      | sse42  | 0.313756 | 0.014962   | 20.97x  | -95.23%        |
| 2048x2048 | uint8   | multiply      | avx2   | 0.313756 | 0.011444   | 27.42x  | -96.35%        |
| 2048x2048 | uint8   | hard_light    | scalar | 0.497571 | 0.183427   | 2.71x   | -63.14%        |
| 2048x2048 | uint8   | hard_light    | sse42  | 0.497571 | 0.017335   | 28.70x  | -96.52%        |
| 2048x2048 | uint8   | hard_light    | avx2   | 0.497571 | 0.013160   | 37.81x  | -97.36%        |
| 2048x2048 | uint8   | difference    | scalar | 0.430310 | 0.113532   | 3.79x   | -73.62%        |
| 2048x2048 | uint8   | difference    | sse42  | 0.430310 | 0.015362   | 28.01x  | -96.43%        |
| 2048x2048 | uint8   | difference    | avx2   | 0.430310 | 0.011447   | 37.59x  | -97.34%        |
| 2048x2048 | uint8   | subtract      | scalar | 0.315272 | 0.116259   | 2.71x   | -63.12%        |
| 2048x2048 | uint8   | subtract      | sse42  | 0.315272 | 0.016938   | 18.61x  | -94.63%        |
| 2048x2048 | uint8   | subtract      | avx2   | 0.315272 | 0.012660   | 24.90x  | -95.98%        |
| 2048x2048 | uint8   | grain_extract | scalar | 0.322946 | 0.135413   | 2.38x   | -58.07%        |
| 2048x2048 | uint8   | grain_extract | sse42  | 0.322946 | 0.014873   | 21.71x  | -95.39%        |
| 2048x2048 | uint8   | grain_extract | avx2   | 0.322946 | 0.011799   | 27.37x  | -96.35%        |
| 2048x2048 | uint8   | grain_merge   | scalar | 0.327965 | 0.134473   | 2.44x   | -59.00%        |
| 2048x2048 | uint8   | grain_merge   | sse42  | 0.327965 | 0.015491   | 21.17x  | -95.28%        |
| 2048x2048 | uint8   | grain_merge   | avx2   | 0.327965 | 0.012381   | 26.49x  | -96.22%        |
| 2048x2048 | uint8   | divide        | scalar | 0.331717 | 0.116154   | 2.86x   | -64.98%        |
| 2048x2048 | uint8   | divide        | sse42  | 0.331717 | 0.015766   | 21.04x  | -95.25%        |
| 2048x2048 | uint8   | divide        | avx2   | 0.331717 | 0.012203   | 27.18x  | -96.32%        |
| 2048x2048 | uint8   | overlay       | scalar | 0.453329 | 0.175331   | 2.59x   | -61.32%        |
| 2048x2048 | uint8   | overlay       | sse42  | 0.453329 | 0.015734   | 28.81x  | -96.53%        |
| 2048x2048 | uint8   | overlay       | avx2   | 0.453329 | 0.012104   | 37.45x  | -97.33%        |
| 2048x2048 | float32 | normal        | scalar | 0.292231 | 0.050183   | 5.82x   | -82.83%        |
| 2048x2048 | float32 | normal        | sse42  | 0.292231 | 0.029556   | 9.89x   | -89.89%        |
| 2048x2048 | float32 | normal        | avx2   | 0.292231 | 0.024034   | 12.16x  | -91.78%        |
| 2048x2048 | float32 | soft_light    | scalar | 0.418918 | 0.065529   | 6.39x   | -84.36%        |
| 2048x2048 | float32 | soft_light    | sse42  | 0.418918 | 0.030712   | 13.64x  | -92.67%        |
| 2048x2048 | float32 | soft_light    | avx2   | 0.418918 | 0.025807   | 16.23x  | -93.84%        |
| 2048x2048 | float32 | lighten_only  | scalar | 0.286359 | 0.069390   | 4.13x   | -75.77%        |
| 2048x2048 | float32 | lighten_only  | sse42  | 0.286359 | 0.031777   | 9.01x   | -88.90%        |
| 2048x2048 | float32 | lighten_only  | avx2   | 0.286359 | 0.027052   | 10.59x  | -90.55%        |
| 2048x2048 | float32 | screen        | scalar | 0.312610 | 0.062269   | 5.02x   | -80.08%        |
| 2048x2048 | float32 | screen        | sse42  | 0.312610 | 0.034575   | 9.04x   | -88.94%        |
| 2048x2048 | float32 | screen        | avx2   | 0.312610 | 0.029529   | 10.59x  | -90.55%        |
| 2048x2048 | float32 | dodge         | scalar | 0.319824 | 0.064759   | 4.94x   | -79.75%        |
| 2048x2048 | float32 | dodge         | sse42  | 0.319824 | 0.033170   | 9.64x   | -89.63%        |
| 2048x2048 | float32 | dodge         | avx2   | 0.319824 | 0.028369   | 11.27x  | -91.13%        |
| 2048x2048 | float32 | addition      | scalar | 0.301168 | 0.102860   | 2.93x   | -65.85%        |
| 2048x2048 | float32 | addition      | sse42  | 0.301168 | 0.033709   | 8.93x   | -88.81%        |
| 2048x2048 | float32 | addition      | avx2   | 0.301168 | 0.026385   | 11.41x  | -91.24%        |
| 2048x2048 | float32 | darken_only   | scalar | 0.297546 | 0.070338   | 4.23x   | -76.36%        |
| 2048x2048 | float32 | darken_only   | sse42  | 0.297546 | 0.033034   | 9.01x   | -88.90%        |
| 2048x2048 | float32 | darken_only   | avx2   | 0.297546 | 0.026576   | 11.20x  | -91.07%        |
| 2048x2048 | float32 | multiply      | scalar | 0.304373 | 0.058819   | 5.17x   | -80.68%        |
| 2048x2048 | float32 | multiply      | sse42  | 0.304373 | 0.032085   | 9.49x   | -89.46%        |
| 2048x2048 | float32 | multiply      | avx2   | 0.304373 | 0.027364   | 11.12x  | -91.01%        |
| 2048x2048 | float32 | hard_light    | scalar | 0.475907 | 0.139437   | 3.41x   | -70.70%        |
| 2048x2048 | float32 | hard_light    | sse42  | 0.475907 | 0.031281   | 15.21x  | -93.43%        |
| 2048x2048 | float32 | hard_light    | avx2   | 0.475907 | 0.027960   | 17.02x  | -94.12%        |
| 2048x2048 | float32 | difference    | scalar | 0.421858 | 0.060096   | 7.02x   | -85.75%        |
| 2048x2048 | float32 | difference    | sse42  | 0.421858 | 0.032237   | 13.09x  | -92.36%        |
| 2048x2048 | float32 | difference    | avx2   | 0.421858 | 0.027524   | 15.33x  | -93.48%        |
| 2048x2048 | float32 | subtract      | scalar | 0.299990 | 0.076736   | 3.91x   | -74.42%        |
| 2048x2048 | float32 | subtract      | sse42  | 0.299990 | 0.032609   | 9.20x   | -89.13%        |
| 2048x2048 | float32 | subtract      | avx2   | 0.299990 | 0.026235   | 11.43x  | -91.25%        |
| 2048x2048 | float32 | grain_extract | scalar | 0.314075 | 0.085673   | 3.67x   | -72.72%        |
| 2048x2048 | float32 | grain_extract | sse42  | 0.314075 | 0.032746   | 9.59x   | -89.57%        |
| 2048x2048 | float32 | grain_extract | avx2   | 0.314075 | 0.027493   | 11.42x  | -91.25%        |
| 2048x2048 | float32 | grain_merge   | scalar | 0.350085 | 0.083535   | 4.19x   | -76.14%        |
| 2048x2048 | float32 | grain_merge   | sse42  | 0.350085 | 0.032421   | 10.80x  | -90.74%        |
| 2048x2048 | float32 | grain_merge   | avx2   | 0.350085 | 0.027152   | 12.89x  | -92.24%        |
| 2048x2048 | float32 | divide        | scalar | 0.331168 | 0.064566   | 5.13x   | -80.50%        |
| 2048x2048 | float32 | divide        | sse42  | 0.331168 | 0.029757   | 11.13x  | -91.01%        |
| 2048x2048 | float32 | divide        | avx2   | 0.331168 | 0.027740   | 11.94x  | -91.62%        |
| 2048x2048 | float32 | overlay       | scalar | 0.433277 | 0.130339   | 3.32x   | -69.92%        |
| 2048x2048 | float32 | overlay       | sse42  | 0.433277 | 0.031829   | 13.61x  | -92.65%        |
| 2048x2048 | float32 | overlay       | avx2   | 0.433277 | 0.026746   | 16.20x  | -93.83%        |
| 1280x720  | uint8   | normal        | scalar | 0.069527 | 0.021429   | 3.24x   | -69.18%        |
| 1280x720  | uint8   | normal        | sse42  | 0.069527 | 0.002777   | 25.04x  | -96.01%        |
| 1280x720  | uint8   | normal        | avx2   | 0.069527 | 0.002263   | 30.73x  | -96.75%        |
| 1280x720  | uint8   | soft_light    | scalar | 0.106778 | 0.025530   | 4.18x   | -76.09%        |
| 1280x720  | uint8   | soft_light    | sse42  | 0.106778 | 0.003575   | 29.87x  | -96.65%        |
| 1280x720  | uint8   | soft_light    | avx2   | 0.106778 | 0.002854   | 37.41x  | -97.33%        |
| 1280x720  | uint8   | lighten_only  | scalar | 0.081469 | 0.026441   | 3.08x   | -67.55%        |
| 1280x720  | uint8   | lighten_only  | sse42  | 0.081469 | 0.003152   | 25.85x  | -96.13%        |
| 1280x720  | uint8   | lighten_only  | avx2   | 0.081469 | 0.002551   | 31.93x  | -96.87%        |
| 1280x720  | uint8   | screen        | scalar | 0.087690 | 0.025673   | 3.42x   | -70.72%        |
| 1280x720  | uint8   | screen        | sse42  | 0.087690 | 0.003229   | 27.16x  | -96.32%        |
| 1280x720  | uint8   | screen        | avx2   | 0.087690 | 0.002568   | 34.15x  | -97.07%        |
| 1280x720  | uint8   | dodge         | scalar | 0.084142 | 0.026093   | 3.22x   | -68.99%        |
| 1280x720  | uint8   | dodge         | sse42  | 0.084142 | 0.003588   | 23.45x  | -95.74%        |
| 1280x720  | uint8   | dodge         | avx2   | 0.084142 | 0.002611   | 32.22x  | -96.90%        |
| 1280x720  | uint8   | addition      | scalar | 0.080709 | 0.030343   | 2.66x   | -62.40%        |
| 1280x720  | uint8   | addition      | sse42  | 0.080709 | 0.003935   | 20.51x  | -95.12%        |
| 1280x720  | uint8   | addition      | avx2   | 0.080709 | 0.002847   | 28.35x  | -96.47%        |
| 1280x720  | uint8   | darken_only   | scalar | 0.082272 | 0.026259   | 3.13x   | -68.08%        |
| 1280x720  | uint8   | darken_only   | sse42  | 0.082272 | 0.003232   | 25.46x  | -96.07%        |
| 1280x720  | uint8   | darken_only   | avx2   | 0.082272 | 0.002574   | 31.96x  | -96.87%        |
| 1280x720  | uint8   | multiply      | scalar | 0.082444 | 0.026108   | 3.16x   | -68.33%        |
| 1280x720  | uint8   | multiply      | sse42  | 0.082444 | 0.003325   | 24.80x  | -95.97%        |
| 1280x720  | uint8   | multiply      | avx2   | 0.082444 | 0.002647   | 31.14x  | -96.79%        |
| 1280x720  | uint8   | hard_light    | scalar | 0.115649 | 0.040272   | 2.87x   | -65.18%        |
| 1280x720  | uint8   | hard_light    | sse42  | 0.115649 | 0.003791   | 30.50x  | -96.72%        |
| 1280x720  | uint8   | hard_light    | avx2   | 0.115649 | 0.002665   | 43.39x  | -97.70%        |
| 1280x720  | uint8   | difference    | scalar | 0.111247 | 0.025589   | 4.35x   | -77.00%        |
| 1280x720  | uint8   | difference    | sse42  | 0.111247 | 0.003202   | 34.74x  | -97.12%        |
| 1280x720  | uint8   | difference    | avx2   | 0.111247 | 0.002539   | 43.82x  | -97.72%        |
| 1280x720  | uint8   | subtract      | scalar | 0.078043 | 0.025694   | 3.04x   | -67.08%        |
| 1280x720  | uint8   | subtract      | sse42  | 0.078043 | 0.003880   | 20.11x  | -95.03%        |
| 1280x720  | uint8   | subtract      | avx2   | 0.078043 | 0.002822   | 27.65x  | -96.38%        |
| 1280x720  | uint8   | grain_extract | scalar | 0.081986 | 0.029202   | 2.81x   | -64.38%        |
| 1280x720  | uint8   | grain_extract | sse42  | 0.081986 | 0.003420   | 23.97x  | -95.83%        |
| 1280x720  | uint8   | grain_extract | avx2   | 0.081986 | 0.002742   | 29.90x  | -96.66%        |
| 1280x720  | uint8   | grain_merge   | scalar | 0.083844 | 0.029119   | 2.88x   | -65.27%        |
| 1280x720  | uint8   | grain_merge   | sse42  | 0.083844 | 0.003198   | 26.22x  | -96.19%        |
| 1280x720  | uint8   | grain_merge   | avx2   | 0.083844 | 0.002529   | 33.15x  | -96.98%        |
| 1280x720  | uint8   | divide        | scalar | 0.085773 | 0.026163   | 3.28x   | -69.50%        |
| 1280x720  | uint8   | divide        | sse42  | 0.085773 | 0.003401   | 25.22x  | -96.03%        |
| 1280x720  | uint8   | divide        | avx2   | 0.085773 | 0.002749   | 31.20x  | -96.79%        |
| 1280x720  | uint8   | overlay       | scalar | 0.107918 | 0.038717   | 2.79x   | -64.12%        |
| 1280x720  | uint8   | overlay       | sse42  | 0.107918 | 0.003693   | 29.22x  | -96.58%        |
| 1280x720  | uint8   | overlay       | avx2   | 0.107918 | 0.002637   | 40.93x  | -97.56%        |
| 1280x720  | float32 | normal        | scalar | 0.064996 | 0.009788   | 6.64x   | -84.94%        |
| 1280x720  | float32 | normal        | sse42  | 0.064996 | 0.004606   | 14.11x  | -92.91%        |
| 1280x720  | float32 | normal        | avx2   | 0.064996 | 0.003649   | 17.81x  | -94.39%        |
| 1280x720  | float32 | soft_light    | scalar | 0.098854 | 0.012988   | 7.61x   | -86.86%        |
| 1280x720  | float32 | soft_light    | sse42  | 0.098854 | 0.005360   | 18.44x  | -94.58%        |
| 1280x720  | float32 | soft_light    | avx2   | 0.098854 | 0.004905   | 20.15x  | -95.04%        |
| 1280x720  | float32 | lighten_only  | scalar | 0.074389 | 0.014313   | 5.20x   | -80.76%        |
| 1280x720  | float32 | lighten_only  | sse42  | 0.074389 | 0.005367   | 13.86x  | -92.78%        |
| 1280x720  | float32 | lighten_only  | avx2   | 0.074389 | 0.004231   | 17.58x  | -94.31%        |
| 1280x720  | float32 | screen        | scalar | 0.079577 | 0.012434   | 6.40x   | -84.37%        |
| 1280x720  | float32 | screen        | sse42  | 0.079577 | 0.005160   | 15.42x  | -93.52%        |
| 1280x720  | float32 | screen        | avx2   | 0.079577 | 0.004254   | 18.71x  | -94.65%        |
| 1280x720  | float32 | dodge         | scalar | 0.081567 | 0.013928   | 5.86x   | -82.92%        |
| 1280x720  | float32 | dodge         | sse42  | 0.081567 | 0.005729   | 14.24x  | -92.98%        |
| 1280x720  | float32 | dodge         | avx2   | 0.081567 | 0.004598   | 17.74x  | -94.36%        |
| 1280x720  | float32 | addition      | scalar | 0.072954 | 0.021217   | 3.44x   | -70.92%        |
| 1280x720  | float32 | addition      | sse42  | 0.072954 | 0.005655   | 12.90x  | -92.25%        |
| 1280x720  | float32 | addition      | avx2   | 0.072954 | 0.003928   | 18.57x  | -94.62%        |
| 1280x720  | float32 | darken_only   | scalar | 0.074108 | 0.014187   | 5.22x   | -80.86%        |
| 1280x720  | float32 | darken_only   | sse42  | 0.074108 | 0.005261   | 14.09x  | -92.90%        |
| 1280x720  | float32 | darken_only   | avx2   | 0.074108 | 0.004452   | 16.65x  | -93.99%        |
| 1280x720  | float32 | multiply      | scalar | 0.075526 | 0.011091   | 6.81x   | -85.31%        |
| 1280x720  | float32 | multiply      | sse42  | 0.075526 | 0.005517   | 13.69x  | -92.70%        |
| 1280x720  | float32 | multiply      | avx2   | 0.075526 | 0.004245   | 17.79x  | -94.38%        |
| 1280x720  | float32 | hard_light    | scalar | 0.116174 | 0.028714   | 4.05x   | -75.28%        |
| 1280x720  | float32 | hard_light    | sse42  | 0.116174 | 0.005040   | 23.05x  | -95.66%        |
| 1280x720  | float32 | hard_light    | avx2   | 0.116174 | 0.004540   | 25.59x  | -96.09%        |
| 1280x720  | float32 | difference    | scalar | 0.105958 | 0.011514   | 9.20x   | -89.13%        |
| 1280x720  | float32 | difference    | sse42  | 0.105958 | 0.005239   | 20.23x  | -95.06%        |
| 1280x720  | float32 | difference    | avx2   | 0.105958 | 0.004269   | 24.82x  | -95.97%        |
| 1280x720  | float32 | subtract      | scalar | 0.076588 | 0.015015   | 5.10x   | -80.40%        |
| 1280x720  | float32 | subtract      | sse42  | 0.076588 | 0.005354   | 14.31x  | -93.01%        |
| 1280x720  | float32 | subtract      | avx2   | 0.076588 | 0.003908   | 19.60x  | -94.90%        |
| 1280x720  | float32 | grain_extract | scalar | 0.076514 | 0.017116   | 4.47x   | -77.63%        |
| 1280x720  | float32 | grain_extract | sse42  | 0.076514 | 0.005680   | 13.47x  | -92.58%        |
| 1280x720  | float32 | grain_extract | avx2   | 0.076514 | 0.004042   | 18.93x  | -94.72%        |
| 1280x720  | float32 | grain_merge   | scalar | 0.076546 | 0.017994   | 4.25x   | -76.49%        |
| 1280x720  | float32 | grain_merge   | sse42  | 0.076546 | 0.005108   | 14.99x  | -93.33%        |
| 1280x720  | float32 | grain_merge   | avx2   | 0.076546 | 0.004217   | 18.15x  | -94.49%        |
| 1280x720  | float32 | divide        | scalar | 0.078998 | 0.012426   | 6.36x   | -84.27%        |
| 1280x720  | float32 | divide        | sse42  | 0.078998 | 0.005313   | 14.87x  | -93.27%        |
| 1280x720  | float32 | divide        | avx2   | 0.078998 | 0.003933   | 20.09x  | -95.02%        |
| 1280x720  | float32 | overlay       | scalar | 0.102286 | 0.027081   | 3.78x   | -73.52%        |
| 1280x720  | float32 | overlay       | sse42  | 0.102286 | 0.006067   | 16.86x  | -94.07%        |
| 1280x720  | float32 | overlay       | avx2   | 0.102286 | 0.003948   | 25.91x  | -96.14%        |
| 1920x1080 | uint8   | normal        | scalar | 0.147203 | 0.044523   | 3.31x   | -69.75%        |
| 1920x1080 | uint8   | normal        | sse42  | 0.147203 | 0.006870   | 21.43x  | -95.33%        |
| 1920x1080 | uint8   | normal        | avx2   | 0.147203 | 0.004962   | 29.66x  | -96.63%        |
| 1920x1080 | uint8   | soft_light    | scalar | 0.215697 | 0.056479   | 3.82x   | -73.82%        |
| 1920x1080 | uint8   | soft_light    | sse42  | 0.215697 | 0.007217   | 29.89x  | -96.65%        |
| 1920x1080 | uint8   | soft_light    | avx2   | 0.215697 | 0.005737   | 37.60x  | -97.34%        |
| 1920x1080 | uint8   | lighten_only  | scalar | 0.151983 | 0.057836   | 2.63x   | -61.95%        |
| 1920x1080 | uint8   | lighten_only  | sse42  | 0.151983 | 0.006926   | 21.95x  | -95.44%        |
| 1920x1080 | uint8   | lighten_only  | avx2   | 0.151983 | 0.005677   | 26.77x  | -96.26%        |
| 1920x1080 | uint8   | screen        | scalar | 0.158132 | 0.057980   | 2.73x   | -63.33%        |
| 1920x1080 | uint8   | screen        | sse42  | 0.158132 | 0.006934   | 22.80x  | -95.61%        |
| 1920x1080 | uint8   | screen        | avx2   | 0.158132 | 0.005893   | 26.83x  | -96.27%        |
| 1920x1080 | uint8   | dodge         | scalar | 0.159393 | 0.056430   | 2.82x   | -64.60%        |
| 1920x1080 | uint8   | dodge         | sse42  | 0.159393 | 0.007688   | 20.73x  | -95.18%        |
| 1920x1080 | uint8   | dodge         | avx2   | 0.159393 | 0.006356   | 25.08x  | -96.01%        |
| 1920x1080 | uint8   | addition      | scalar | 0.154528 | 0.074392   | 2.08x   | -51.86%        |
| 1920x1080 | uint8   | addition      | sse42  | 0.154528 | 0.008844   | 17.47x  | -94.28%        |
| 1920x1080 | uint8   | addition      | avx2   | 0.154528 | 0.006039   | 25.59x  | -96.09%        |
| 1920x1080 | uint8   | darken_only   | scalar | 0.150324 | 0.059159   | 2.54x   | -60.65%        |
| 1920x1080 | uint8   | darken_only   | sse42  | 0.150324 | 0.006921   | 21.72x  | -95.40%        |
| 1920x1080 | uint8   | darken_only   | avx2   | 0.150324 | 0.005692   | 26.41x  | -96.21%        |
| 1920x1080 | uint8   | multiply      | scalar | 0.159026 | 0.055365   | 2.87x   | -65.19%        |
| 1920x1080 | uint8   | multiply      | sse42  | 0.159026 | 0.009035   | 17.60x  | -94.32%        |
| 1920x1080 | uint8   | multiply      | avx2   | 0.159026 | 0.006475   | 24.56x  | -95.93%        |
| 1920x1080 | uint8   | hard_light    | scalar | 0.229847 | 0.091336   | 2.52x   | -60.26%        |
| 1920x1080 | uint8   | hard_light    | sse42  | 0.229847 | 0.008240   | 27.89x  | -96.41%        |
| 1920x1080 | uint8   | hard_light    | avx2   | 0.229847 | 0.005894   | 39.00x  | -97.44%        |
| 1920x1080 | uint8   | difference    | scalar | 0.215360 | 0.057014   | 3.78x   | -73.53%        |
| 1920x1080 | uint8   | difference    | sse42  | 0.215360 | 0.007321   | 29.42x  | -96.60%        |
| 1920x1080 | uint8   | difference    | avx2   | 0.215360 | 0.006531   | 32.97x  | -96.97%        |
| 1920x1080 | uint8   | subtract      | scalar | 0.155677 | 0.054502   | 2.86x   | -64.99%        |
| 1920x1080 | uint8   | subtract      | sse42  | 0.155677 | 0.008770   | 17.75x  | -94.37%        |
| 1920x1080 | uint8   | subtract      | avx2   | 0.155677 | 0.006109   | 25.48x  | -96.08%        |
| 1920x1080 | uint8   | grain_extract | scalar | 0.160491 | 0.064471   | 2.49x   | -59.83%        |
| 1920x1080 | uint8   | grain_extract | sse42  | 0.160491 | 0.007184   | 22.34x  | -95.52%        |
| 1920x1080 | uint8   | grain_extract | avx2   | 0.160491 | 0.006023   | 26.65x  | -96.25%        |
| 1920x1080 | uint8   | grain_merge   | scalar | 0.159387 | 0.067584   | 2.36x   | -57.60%        |
| 1920x1080 | uint8   | grain_merge   | sse42  | 0.159387 | 0.011619   | 13.72x  | -92.71%        |
| 1920x1080 | uint8   | grain_merge   | avx2   | 0.159387 | 0.006016   | 26.50x  | -96.23%        |
| 1920x1080 | uint8   | divide        | scalar | 0.160209 | 0.054486   | 2.94x   | -65.99%        |
| 1920x1080 | uint8   | divide        | sse42  | 0.160209 | 0.007655   | 20.93x  | -95.22%        |
| 1920x1080 | uint8   | divide        | avx2   | 0.160209 | 0.005764   | 27.79x  | -96.40%        |
| 1920x1080 | uint8   | overlay       | scalar | 0.218937 | 0.085173   | 2.57x   | -61.10%        |
| 1920x1080 | uint8   | overlay       | sse42  | 0.218937 | 0.007741   | 28.28x  | -96.46%        |
| 1920x1080 | uint8   | overlay       | avx2   | 0.218937 | 0.005740   | 38.14x  | -97.38%        |
| 1920x1080 | float32 | normal        | scalar | 0.137573 | 0.020245   | 6.80x   | -85.28%        |
| 1920x1080 | float32 | normal        | sse42  | 0.137573 | 0.009879   | 13.93x  | -92.82%        |
| 1920x1080 | float32 | normal        | avx2   | 0.137573 | 0.008780   | 15.67x  | -93.62%        |
| 1920x1080 | float32 | soft_light    | scalar | 0.208219 | 0.028122   | 7.40x   | -86.49%        |
| 1920x1080 | float32 | soft_light    | sse42  | 0.208219 | 0.012303   | 16.92x  | -94.09%        |
| 1920x1080 | float32 | soft_light    | avx2   | 0.208219 | 0.011280   | 18.46x  | -94.58%        |
| 1920x1080 | float32 | lighten_only  | scalar | 0.150905 | 0.029237   | 5.16x   | -80.63%        |
| 1920x1080 | float32 | lighten_only  | sse42  | 0.150905 | 0.011591   | 13.02x  | -92.32%        |
| 1920x1080 | float32 | lighten_only  | avx2   | 0.150905 | 0.008383   | 18.00x  | -94.45%        |
| 1920x1080 | float32 | screen        | scalar | 0.148837 | 0.026484   | 5.62x   | -82.21%        |
| 1920x1080 | float32 | screen        | sse42  | 0.148837 | 0.011388   | 13.07x  | -92.35%        |
| 1920x1080 | float32 | screen        | avx2   | 0.148837 | 0.008689   | 17.13x  | -94.16%        |
| 1920x1080 | float32 | dodge         | scalar | 0.155928 | 0.027763   | 5.62x   | -82.19%        |
| 1920x1080 | float32 | dodge         | sse42  | 0.155928 | 0.011529   | 13.52x  | -92.61%        |
| 1920x1080 | float32 | dodge         | avx2   | 0.155928 | 0.008667   | 17.99x  | -94.44%        |
| 1920x1080 | float32 | addition      | scalar | 0.143480 | 0.047390   | 3.03x   | -66.97%        |
| 1920x1080 | float32 | addition      | sse42  | 0.143480 | 0.011908   | 12.05x  | -91.70%        |
| 1920x1080 | float32 | addition      | avx2   | 0.143480 | 0.008666   | 16.56x  | -93.96%        |
| 1920x1080 | float32 | darken_only   | scalar | 0.141549 | 0.030872   | 4.59x   | -78.19%        |
| 1920x1080 | float32 | darken_only   | sse42  | 0.141549 | 0.011574   | 12.23x  | -91.82%        |
| 1920x1080 | float32 | darken_only   | avx2   | 0.141549 | 0.008654   | 16.36x  | -93.89%        |
| 1920x1080 | float32 | multiply      | scalar | 0.149314 | 0.024408   | 6.12x   | -83.65%        |
| 1920x1080 | float32 | multiply      | sse42  | 0.149314 | 0.011652   | 12.81x  | -92.20%        |
| 1920x1080 | float32 | multiply      | avx2   | 0.149314 | 0.009469   | 15.77x  | -93.66%        |
| 1920x1080 | float32 | hard_light    | scalar | 0.252664 | 0.064419   | 3.92x   | -74.50%        |
| 1920x1080 | float32 | hard_light    | sse42  | 0.252664 | 0.011391   | 22.18x  | -95.49%        |
| 1920x1080 | float32 | hard_light    | avx2   | 0.252664 | 0.008714   | 29.00x  | -96.55%        |
| 1920x1080 | float32 | difference    | scalar | 0.207944 | 0.029141   | 7.14x   | -85.99%        |
| 1920x1080 | float32 | difference    | sse42  | 0.207944 | 0.011427   | 18.20x  | -94.50%        |
| 1920x1080 | float32 | difference    | avx2   | 0.207944 | 0.008650   | 24.04x  | -95.84%        |
| 1920x1080 | float32 | subtract      | scalar | 0.142464 | 0.031069   | 4.59x   | -78.19%        |
| 1920x1080 | float32 | subtract      | sse42  | 0.142464 | 0.012274   | 11.61x  | -91.38%        |
| 1920x1080 | float32 | subtract      | avx2   | 0.142464 | 0.010339   | 13.78x  | -92.74%        |
| 1920x1080 | float32 | grain_extract | scalar | 0.150826 | 0.036703   | 4.11x   | -75.67%        |
| 1920x1080 | float32 | grain_extract | sse42  | 0.150826 | 0.011403   | 13.23x  | -92.44%        |
| 1920x1080 | float32 | grain_extract | avx2   | 0.150826 | 0.008523   | 17.70x  | -94.35%        |
| 1920x1080 | float32 | grain_merge   | scalar | 0.147879 | 0.037017   | 3.99x   | -74.97%        |
| 1920x1080 | float32 | grain_merge   | sse42  | 0.147879 | 0.011458   | 12.91x  | -92.25%        |
| 1920x1080 | float32 | grain_merge   | avx2   | 0.147879 | 0.009218   | 16.04x  | -93.77%        |
| 1920x1080 | float32 | divide        | scalar | 0.148888 | 0.025964   | 5.73x   | -82.56%        |
| 1920x1080 | float32 | divide        | sse42  | 0.148888 | 0.011398   | 13.06x  | -92.34%        |
| 1920x1080 | float32 | divide        | avx2   | 0.148888 | 0.008997   | 16.55x  | -93.96%        |
| 1920x1080 | float32 | overlay       | scalar | 0.205824 | 0.059503   | 3.46x   | -71.09%        |
| 1920x1080 | float32 | overlay       | sse42  | 0.205824 | 0.011849   | 17.37x  | -94.24%        |
| 1920x1080 | float32 | overlay       | avx2   | 0.205824 | 0.008719   | 23.61x  | -95.76%        |
| 2560x1440 | uint8   | normal        | scalar | 0.275598 | 0.080080   | 3.44x   | -70.94%        |
| 2560x1440 | uint8   | normal        | sse42  | 0.275598 | 0.012020   | 22.93x  | -95.64%        |
| 2560x1440 | uint8   | normal        | avx2   | 0.275598 | 0.009128   | 30.19x  | -96.69%        |
| 2560x1440 | uint8   | soft_light    | scalar | 0.383637 | 0.105085   | 3.65x   | -72.61%        |
| 2560x1440 | uint8   | soft_light    | sse42  | 0.383637 | 0.014175   | 27.06x  | -96.31%        |
| 2560x1440 | uint8   | soft_light    | avx2   | 0.383637 | 0.010598   | 36.20x  | -97.24%        |
| 2560x1440 | uint8   | lighten_only  | scalar | 0.265651 | 0.106696   | 2.49x   | -59.84%        |
| 2560x1440 | uint8   | lighten_only  | sse42  | 0.265651 | 0.012204   | 21.77x  | -95.41%        |
| 2560x1440 | uint8   | lighten_only  | avx2   | 0.265651 | 0.010026   | 26.50x  | -96.23%        |
| 2560x1440 | uint8   | screen        | scalar | 0.290039 | 0.100229   | 2.89x   | -65.44%        |
| 2560x1440 | uint8   | screen        | sse42  | 0.290039 | 0.012870   | 22.54x  | -95.56%        |
| 2560x1440 | uint8   | screen        | avx2   | 0.290039 | 0.010163   | 28.54x  | -96.50%        |
| 2560x1440 | uint8   | dodge         | scalar | 0.294054 | 0.102669   | 2.86x   | -65.08%        |
| 2560x1440 | uint8   | dodge         | sse42  | 0.294054 | 0.014015   | 20.98x  | -95.23%        |
| 2560x1440 | uint8   | dodge         | avx2   | 0.294054 | 0.010338   | 28.44x  | -96.48%        |
| 2560x1440 | uint8   | addition      | scalar | 0.274345 | 0.121012   | 2.27x   | -55.89%        |
| 2560x1440 | uint8   | addition      | sse42  | 0.274345 | 0.015446   | 17.76x  | -94.37%        |
| 2560x1440 | uint8   | addition      | avx2   | 0.274345 | 0.010800   | 25.40x  | -96.06%        |
| 2560x1440 | uint8   | darken_only   | scalar | 0.268952 | 0.106113   | 2.53x   | -60.55%        |
| 2560x1440 | uint8   | darken_only   | sse42  | 0.268952 | 0.012188   | 22.07x  | -95.47%        |
| 2560x1440 | uint8   | darken_only   | avx2   | 0.268952 | 0.009990   | 26.92x  | -96.29%        |
| 2560x1440 | uint8   | multiply      | scalar | 0.273182 | 0.098091   | 2.78x   | -64.09%        |
| 2560x1440 | uint8   | multiply      | sse42  | 0.273182 | 0.012968   | 21.07x  | -95.25%        |
| 2560x1440 | uint8   | multiply      | avx2   | 0.273182 | 0.010492   | 26.04x  | -96.16%        |
| 2560x1440 | uint8   | hard_light    | scalar | 0.425744 | 0.160502   | 2.65x   | -62.30%        |
| 2560x1440 | uint8   | hard_light    | sse42  | 0.425744 | 0.016771   | 25.39x  | -96.06%        |
| 2560x1440 | uint8   | hard_light    | avx2   | 0.425744 | 0.011130   | 38.25x  | -97.39%        |
| 2560x1440 | uint8   | difference    | scalar | 0.381898 | 0.100232   | 3.81x   | -73.75%        |
| 2560x1440 | uint8   | difference    | sse42  | 0.381898 | 0.012561   | 30.40x  | -96.71%        |
| 2560x1440 | uint8   | difference    | avx2   | 0.381898 | 0.010728   | 35.60x  | -97.19%        |
| 2560x1440 | uint8   | subtract      | scalar | 0.283814 | 0.098866   | 2.87x   | -65.17%        |
| 2560x1440 | uint8   | subtract      | sse42  | 0.283814 | 0.014922   | 19.02x  | -94.74%        |
| 2560x1440 | uint8   | subtract      | avx2   | 0.283814 | 0.010939   | 25.94x  | -96.15%        |
| 2560x1440 | uint8   | grain_extract | scalar | 0.292445 | 0.113812   | 2.57x   | -61.08%        |
| 2560x1440 | uint8   | grain_extract | sse42  | 0.292445 | 0.013397   | 21.83x  | -95.42%        |
| 2560x1440 | uint8   | grain_extract | avx2   | 0.292445 | 0.010145   | 28.83x  | -96.53%        |
| 2560x1440 | uint8   | grain_merge   | scalar | 0.288894 | 0.117598   | 2.46x   | -59.29%        |
| 2560x1440 | uint8   | grain_merge   | sse42  | 0.288894 | 0.012974   | 22.27x  | -95.51%        |
| 2560x1440 | uint8   | grain_merge   | avx2   | 0.288894 | 0.010178   | 28.38x  | -96.48%        |
| 2560x1440 | uint8   | divide        | scalar | 0.297255 | 0.099431   | 2.99x   | -66.55%        |
| 2560x1440 | uint8   | divide        | sse42  | 0.297255 | 0.014253   | 20.86x  | -95.21%        |
| 2560x1440 | uint8   | divide        | avx2   | 0.297255 | 0.010315   | 28.82x  | -96.53%        |
| 2560x1440 | uint8   | overlay       | scalar | 0.388810 | 0.150759   | 2.58x   | -61.23%        |
| 2560x1440 | uint8   | overlay       | sse42  | 0.388810 | 0.014034   | 27.71x  | -96.39%        |
| 2560x1440 | uint8   | overlay       | avx2   | 0.388810 | 0.011058   | 35.16x  | -97.16%        |
| 2560x1440 | float32 | normal        | scalar | 0.250714 | 0.042329   | 5.92x   | -83.12%        |
| 2560x1440 | float32 | normal        | sse42  | 0.250714 | 0.024777   | 10.12x  | -90.12%        |
| 2560x1440 | float32 | normal        | avx2   | 0.250714 | 0.020573   | 12.19x  | -91.79%        |
| 2560x1440 | float32 | soft_light    | scalar | 0.358318 | 0.048541   | 7.38x   | -86.45%        |
| 2560x1440 | float32 | soft_light    | sse42  | 0.358318 | 0.020687   | 17.32x  | -94.23%        |
| 2560x1440 | float32 | soft_light    | avx2   | 0.358318 | 0.016853   | 21.26x  | -95.30%        |
| 2560x1440 | float32 | lighten_only  | scalar | 0.250149 | 0.061320   | 4.08x   | -75.49%        |
| 2560x1440 | float32 | lighten_only  | sse42  | 0.250149 | 0.029145   | 8.58x   | -88.35%        |
| 2560x1440 | float32 | lighten_only  | avx2   | 0.250149 | 0.025282   | 9.89x   | -89.89%        |
| 2560x1440 | float32 | screen        | scalar | 0.269560 | 0.048448   | 5.56x   | -82.03%        |
| 2560x1440 | float32 | screen        | sse42  | 0.269560 | 0.020815   | 12.95x  | -92.28%        |
| 2560x1440 | float32 | screen        | avx2   | 0.269560 | 0.016246   | 16.59x  | -93.97%        |
| 2560x1440 | float32 | dodge         | scalar | 0.276142 | 0.057349   | 4.82x   | -79.23%        |
| 2560x1440 | float32 | dodge         | sse42  | 0.276142 | 0.027886   | 9.90x   | -89.90%        |
| 2560x1440 | float32 | dodge         | avx2   | 0.276142 | 0.022807   | 12.11x  | -91.74%        |
| 2560x1440 | float32 | addition      | scalar | 0.259869 | 0.084730   | 3.07x   | -67.39%        |
| 2560x1440 | float32 | addition      | sse42  | 0.259869 | 0.021534   | 12.07x  | -91.71%        |
| 2560x1440 | float32 | addition      | avx2   | 0.259869 | 0.017087   | 15.21x  | -93.42%        |
| 2560x1440 | float32 | darken_only   | scalar | 0.253387 | 0.064300   | 3.94x   | -74.62%        |
| 2560x1440 | float32 | darken_only   | sse42  | 0.253387 | 0.028559   | 8.87x   | -88.73%        |
| 2560x1440 | float32 | darken_only   | avx2   | 0.253387 | 0.023834   | 10.63x  | -90.59%        |
| 2560x1440 | float32 | multiply      | scalar | 0.258357 | 0.044335   | 5.83x   | -82.84%        |
| 2560x1440 | float32 | multiply      | sse42  | 0.258357 | 0.021154   | 12.21x  | -91.81%        |
| 2560x1440 | float32 | multiply      | avx2   | 0.258357 | 0.016650   | 15.52x  | -93.56%        |
| 2560x1440 | float32 | hard_light    | scalar | 0.412607 | 0.119177   | 3.46x   | -71.12%        |
| 2560x1440 | float32 | hard_light    | sse42  | 0.412607 | 0.028996   | 14.23x  | -92.97%        |
| 2560x1440 | float32 | hard_light    | avx2   | 0.412607 | 0.025530   | 16.16x  | -93.81%        |
| 2560x1440 | float32 | difference    | scalar | 0.373262 | 0.043075   | 8.67x   | -88.46%        |
| 2560x1440 | float32 | difference    | sse42  | 0.373262 | 0.020912   | 17.85x  | -94.40%        |
| 2560x1440 | float32 | difference    | avx2   | 0.373262 | 0.015303   | 24.39x  | -95.90%        |
| 2560x1440 | float32 | subtract      | scalar | 0.256860 | 0.066653   | 3.85x   | -74.05%        |
| 2560x1440 | float32 | subtract      | sse42  | 0.256860 | 0.028626   | 8.97x   | -88.86%        |
| 2560x1440 | float32 | subtract      | avx2   | 0.256860 | 0.023871   | 10.76x  | -90.71%        |
| 2560x1440 | float32 | grain_extract | scalar | 0.263986 | 0.066669   | 3.96x   | -74.75%        |
| 2560x1440 | float32 | grain_extract | sse42  | 0.263986 | 0.021224   | 12.44x  | -91.96%        |
| 2560x1440 | float32 | grain_extract | avx2   | 0.263986 | 0.019083   | 13.83x  | -92.77%        |
| 2560x1440 | float32 | grain_merge   | scalar | 0.265622 | 0.072432   | 3.67x   | -72.73%        |
| 2560x1440 | float32 | grain_merge   | sse42  | 0.265622 | 0.027811   | 9.55x   | -89.53%        |
| 2560x1440 | float32 | grain_merge   | avx2   | 0.265622 | 0.026259   | 10.12x  | -90.11%        |
| 2560x1440 | float32 | divide        | scalar | 0.268610 | 0.046366   | 5.79x   | -82.74%        |
| 2560x1440 | float32 | divide        | sse42  | 0.268610 | 0.020533   | 13.08x  | -92.36%        |
| 2560x1440 | float32 | divide        | avx2   | 0.268610 | 0.016938   | 15.86x  | -93.69%        |
| 2560x1440 | float32 | overlay       | scalar | 0.378575 | 0.112807   | 3.36x   | -70.20%        |
| 2560x1440 | float32 | overlay       | sse42  | 0.378575 | 0.026615   | 14.22x  | -92.97%        |
| 2560x1440 | float32 | overlay       | avx2   | 0.378575 | 0.022977   | 16.48x  | -93.93%        |
| 3840x2160 | uint8   | normal        | scalar | 0.610403 | 0.182561   | 3.34x   | -70.09%        |
| 3840x2160 | uint8   | normal        | sse42  | 0.610403 | 0.027154   | 22.48x  | -95.55%        |
| 3840x2160 | uint8   | normal        | avx2   | 0.610403 | 0.020338   | 30.01x  | -96.67%        |
| 3840x2160 | uint8   | soft_light    | scalar | 0.843191 | 0.233125   | 3.62x   | -72.35%        |
| 3840x2160 | uint8   | soft_light    | sse42  | 0.843191 | 0.029676   | 28.41x  | -96.48%        |
| 3840x2160 | uint8   | soft_light    | avx2   | 0.843191 | 0.023548   | 35.81x  | -97.21%        |
| 3840x2160 | uint8   | lighten_only  | scalar | 0.572585 | 0.239340   | 2.39x   | -58.20%        |
| 3840x2160 | uint8   | lighten_only  | sse42  | 0.572585 | 0.028160   | 20.33x  | -95.08%        |
| 3840x2160 | uint8   | lighten_only  | avx2   | 0.572585 | 0.023074   | 24.82x  | -95.97%        |
| 3840x2160 | uint8   | screen        | scalar | 0.622213 | 0.228613   | 2.72x   | -63.26%        |
| 3840x2160 | uint8   | screen        | sse42  | 0.622213 | 0.027912   | 22.29x  | -95.51%        |
| 3840x2160 | uint8   | screen        | avx2   | 0.622213 | 0.023162   | 26.86x  | -96.28%        |
| 3840x2160 | uint8   | dodge         | scalar | 0.625303 | 0.242821   | 2.58x   | -61.17%        |
| 3840x2160 | uint8   | dodge         | sse42  | 0.625303 | 0.033411   | 18.72x  | -94.66%        |
| 3840x2160 | uint8   | dodge         | avx2   | 0.625303 | 0.022887   | 27.32x  | -96.34%        |
| 3840x2160 | uint8   | addition      | scalar | 0.600483 | 0.279841   | 2.15x   | -53.40%        |
| 3840x2160 | uint8   | addition      | sse42  | 0.600483 | 0.035461   | 16.93x  | -94.09%        |
| 3840x2160 | uint8   | addition      | avx2   | 0.600483 | 0.024557   | 24.45x  | -95.91%        |
| 3840x2160 | uint8   | darken_only   | scalar | 0.585788 | 0.238322   | 2.46x   | -59.32%        |
| 3840x2160 | uint8   | darken_only   | sse42  | 0.585788 | 0.027635   | 21.20x  | -95.28%        |
| 3840x2160 | uint8   | darken_only   | avx2   | 0.585788 | 0.022703   | 25.80x  | -96.12%        |
| 3840x2160 | uint8   | multiply      | scalar | 0.645848 | 0.228531   | 2.83x   | -64.62%        |
| 3840x2160 | uint8   | multiply      | sse42  | 0.645848 | 0.028967   | 22.30x  | -95.51%        |
| 3840x2160 | uint8   | multiply      | avx2   | 0.645848 | 0.023048   | 28.02x  | -96.43%        |
| 3840x2160 | uint8   | hard_light    | scalar | 0.970269 | 0.360863   | 2.69x   | -62.81%        |
| 3840x2160 | uint8   | hard_light    | sse42  | 0.970269 | 0.034917   | 27.79x  | -96.40%        |
| 3840x2160 | uint8   | hard_light    | avx2   | 0.970269 | 0.024432   | 39.71x  | -97.48%        |
| 3840x2160 | uint8   | difference    | scalar | 0.830939 | 0.218942   | 3.80x   | -73.65%        |
| 3840x2160 | uint8   | difference    | sse42  | 0.830939 | 0.027799   | 29.89x  | -96.65%        |
| 3840x2160 | uint8   | difference    | avx2   | 0.830939 | 0.023219   | 35.79x  | -97.21%        |
| 3840x2160 | uint8   | subtract      | scalar | 0.606483 | 0.228972   | 2.65x   | -62.25%        |
| 3840x2160 | uint8   | subtract      | sse42  | 0.606483 | 0.034566   | 17.55x  | -94.30%        |
| 3840x2160 | uint8   | subtract      | avx2   | 0.606483 | 0.024770   | 24.48x  | -95.92%        |
| 3840x2160 | uint8   | grain_extract | scalar | 0.631731 | 0.270116   | 2.34x   | -57.24%        |
| 3840x2160 | uint8   | grain_extract | sse42  | 0.631731 | 0.032463   | 19.46x  | -94.86%        |
| 3840x2160 | uint8   | grain_extract | avx2   | 0.631731 | 0.024086   | 26.23x  | -96.19%        |
| 3840x2160 | uint8   | grain_merge   | scalar | 0.671660 | 0.269635   | 2.49x   | -59.86%        |
| 3840x2160 | uint8   | grain_merge   | sse42  | 0.671660 | 0.030026   | 22.37x  | -95.53%        |
| 3840x2160 | uint8   | grain_merge   | avx2   | 0.671660 | 0.023722   | 28.31x  | -96.47%        |
| 3840x2160 | uint8   | divide        | scalar | 0.640821 | 0.227621   | 2.82x   | -64.48%        |
| 3840x2160 | uint8   | divide        | sse42  | 0.640821 | 0.030934   | 20.72x  | -95.17%        |
| 3840x2160 | uint8   | divide        | avx2   | 0.640821 | 0.023074   | 27.77x  | -96.40%        |
| 3840x2160 | uint8   | overlay       | scalar | 0.868035 | 0.345552   | 2.51x   | -60.19%        |
| 3840x2160 | uint8   | overlay       | sse42  | 0.868035 | 0.031268   | 27.76x  | -96.40%        |
| 3840x2160 | uint8   | overlay       | avx2   | 0.868035 | 0.023404   | 37.09x  | -97.30%        |
| 3840x2160 | float32 | normal        | scalar | 0.548193 | 0.094256   | 5.82x   | -82.81%        |
| 3840x2160 | float32 | normal        | sse42  | 0.548193 | 0.054646   | 10.03x  | -90.03%        |
| 3840x2160 | float32 | normal        | avx2   | 0.548193 | 0.046146   | 11.88x  | -91.58%        |
| 3840x2160 | float32 | soft_light    | scalar | 0.784198 | 0.122798   | 6.39x   | -84.34%        |
| 3840x2160 | float32 | soft_light    | sse42  | 0.784198 | 0.058446   | 13.42x  | -92.55%        |
| 3840x2160 | float32 | soft_light    | avx2   | 0.784198 | 0.049485   | 15.85x  | -93.69%        |
| 3840x2160 | float32 | lighten_only  | scalar | 0.534883 | 0.134094   | 3.99x   | -74.93%        |
| 3840x2160 | float32 | lighten_only  | sse42  | 0.534883 | 0.060763   | 8.80x   | -88.64%        |
| 3840x2160 | float32 | lighten_only  | avx2   | 0.534883 | 0.048658   | 10.99x  | -90.90%        |
| 3840x2160 | float32 | screen        | scalar | 0.583949 | 0.122091   | 4.78x   | -79.09%        |
| 3840x2160 | float32 | screen        | sse42  | 0.583949 | 0.059871   | 9.75x   | -89.75%        |
| 3840x2160 | float32 | screen        | avx2   | 0.583949 | 0.051954   | 11.24x  | -91.10%        |
| 3840x2160 | float32 | dodge         | scalar | 0.594659 | 0.125584   | 4.74x   | -78.88%        |
| 3840x2160 | float32 | dodge         | sse42  | 0.594659 | 0.064070   | 9.28x   | -89.23%        |
| 3840x2160 | float32 | dodge         | avx2   | 0.594659 | 0.053074   | 11.20x  | -91.07%        |
| 3840x2160 | float32 | addition      | scalar | 0.575785 | 0.203821   | 2.82x   | -64.60%        |
| 3840x2160 | float32 | addition      | sse42  | 0.575785 | 0.063356   | 9.09x   | -89.00%        |
| 3840x2160 | float32 | addition      | avx2   | 0.575785 | 0.052123   | 11.05x  | -90.95%        |
| 3840x2160 | float32 | darken_only   | scalar | 0.547298 | 0.139548   | 3.92x   | -74.50%        |
| 3840x2160 | float32 | darken_only   | sse42  | 0.547298 | 0.062579   | 8.75x   | -88.57%        |
| 3840x2160 | float32 | darken_only   | avx2   | 0.547298 | 0.050886   | 10.76x  | -90.70%        |
| 3840x2160 | float32 | multiply      | scalar | 0.571176 | 0.114330   | 5.00x   | -79.98%        |
| 3840x2160 | float32 | multiply      | sse42  | 0.571176 | 0.061350   | 9.31x   | -89.26%        |
| 3840x2160 | float32 | multiply      | avx2   | 0.571176 | 0.053155   | 10.75x  | -90.69%        |
| 3840x2160 | float32 | hard_light    | scalar | 0.909403 | 0.265897   | 3.42x   | -70.76%        |
| 3840x2160 | float32 | hard_light    | sse42  | 0.909403 | 0.059160   | 15.37x  | -93.49%        |
| 3840x2160 | float32 | hard_light    | avx2   | 0.909403 | 0.052195   | 17.42x  | -94.26%        |
| 3840x2160 | float32 | difference    | scalar | 0.801380 | 0.116555   | 6.88x   | -85.46%        |
| 3840x2160 | float32 | difference    | sse42  | 0.801380 | 0.064007   | 12.52x  | -92.01%        |
| 3840x2160 | float32 | difference    | avx2   | 0.801380 | 0.051511   | 15.56x  | -93.57%        |
| 3840x2160 | float32 | subtract      | scalar | 0.588923 | 0.139021   | 4.24x   | -76.39%        |
| 3840x2160 | float32 | subtract      | sse42  | 0.588923 | 0.062666   | 9.40x   | -89.36%        |
| 3840x2160 | float32 | subtract      | avx2   | 0.588923 | 0.050919   | 11.57x  | -91.35%        |
| 3840x2160 | float32 | grain_extract | scalar | 0.613665 | 0.173517   | 3.54x   | -71.72%        |
| 3840x2160 | float32 | grain_extract | sse42  | 0.613665 | 0.058452   | 10.50x  | -90.47%        |
| 3840x2160 | float32 | grain_extract | avx2   | 0.613665 | 0.047528   | 12.91x  | -92.26%        |
| 3840x2160 | float32 | grain_merge   | scalar | 0.577530 | 0.178249   | 3.24x   | -69.14%        |
| 3840x2160 | float32 | grain_merge   | sse42  | 0.577530 | 0.066543   | 8.68x   | -88.48%        |
| 3840x2160 | float32 | grain_merge   | avx2   | 0.577530 | 0.075533   | 7.65x   | -86.92%        |
| 3840x2160 | float32 | divide        | scalar | 0.635617 | 0.135847   | 4.68x   | -78.63%        |
| 3840x2160 | float32 | divide        | sse42  | 0.635617 | 0.057431   | 11.07x  | -90.96%        |
| 3840x2160 | float32 | divide        | avx2   | 0.635617 | 0.054264   | 11.71x  | -91.46%        |
| 3840x2160 | float32 | overlay       | scalar | 0.897097 | 0.262501   | 3.42x   | -70.74%        |
| 3840x2160 | float32 | overlay       | sse42  | 0.897097 | 0.060136   | 14.92x  | -93.30%        |
| 3840x2160 | float32 | overlay       | avx2   | 0.897097 | 0.063020   | 14.24x  | -92.98%        |
</details>
