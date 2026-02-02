# SIMD Blend Modes

This project reimplements the blend modes from [`blend_modes`](https://github.com/flrs/blend_modes) with C kernels and SIMD
(SSE4.2/AVX2) acceleration. It supports uint8 and float32 NumPy inputs in the range 0..255
and returns output dtype/channel count matching the background image. Missing alpha channels
are treated as fully opaque (255). Opacity defaults to 1.0.

This is mostly intended to be a drop-in replacement, but with a more permissive API that allows you to go faster if you don't need FP32 arrays.

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
| normal        | scalar | 0.176382 | 0.042440   | 4.16x   | -75.94%        |
| normal        | sse42  | 0.176382 | 0.011849   | 14.89x  | -93.28%        |
| normal        | avx2   | 0.176382 | 0.010290   | 17.14x  | -94.17%        |
| soft_light    | scalar | 0.248033 | 0.053965   | 4.60x   | -78.24%        |
| soft_light    | sse42  | 0.248033 | 0.012930   | 19.18x  | -94.79%        |
| soft_light    | avx2   | 0.248033 | 0.010588   | 23.42x  | -95.73%        |
| lighten_only  | scalar | 0.175407 | 0.057309   | 3.06x   | -67.33%        |
| lighten_only  | sse42  | 0.175407 | 0.013309   | 13.18x  | -92.41%        |
| lighten_only  | avx2   | 0.175407 | 0.010847   | 16.17x  | -93.82%        |
| screen        | scalar | 0.190324 | 0.050963   | 3.73x   | -73.22%        |
| screen        | sse42  | 0.190324 | 0.012639   | 15.06x  | -93.36%        |
| screen        | avx2   | 0.190324 | 0.010510   | 18.11x  | -94.48%        |
| dodge         | scalar | 0.187045 | 0.052270   | 3.58x   | -72.05%        |
| dodge         | sse42  | 0.187045 | 0.013680   | 13.67x  | -92.69%        |
| dodge         | avx2   | 0.187045 | 0.010941   | 17.10x  | -94.15%        |
| addition      | scalar | 0.177928 | 0.071643   | 2.48x   | -59.73%        |
| addition      | sse42  | 0.177928 | 0.013763   | 12.93x  | -92.26%        |
| addition      | avx2   | 0.177928 | 0.010651   | 16.71x  | -94.01%        |
| darken_only   | scalar | 0.176545 | 0.056689   | 3.11x   | -67.89%        |
| darken_only   | sse42  | 0.176545 | 0.013355   | 13.22x  | -92.44%        |
| darken_only   | avx2   | 0.176545 | 0.010811   | 16.33x  | -93.88%        |
| multiply      | scalar | 0.183595 | 0.050833   | 3.61x   | -72.31%        |
| multiply      | sse42  | 0.183595 | 0.012836   | 14.30x  | -93.01%        |
| multiply      | avx2   | 0.183595 | 0.010106   | 18.17x  | -94.50%        |
| hard_light    | scalar | 0.279343 | 0.095928   | 2.91x   | -65.66%        |
| hard_light    | sse42  | 0.279343 | 0.013578   | 20.57x  | -95.14%        |
| hard_light    | avx2   | 0.279343 | 0.010918   | 25.58x  | -96.09%        |
| difference    | scalar | 0.248033 | 0.049055   | 5.06x   | -80.22%        |
| difference    | sse42  | 0.248033 | 0.012638   | 19.63x  | -94.90%        |
| difference    | avx2   | 0.248033 | 0.010101   | 24.55x  | -95.93%        |
| subtract      | scalar | 0.182739 | 0.054110   | 3.38x   | -70.39%        |
| subtract      | sse42  | 0.182739 | 0.014454   | 12.64x  | -92.09%        |
| subtract      | avx2   | 0.182739 | 0.011361   | 16.08x  | -93.78%        |
| grain_extract | scalar | 0.189483 | 0.064054   | 2.96x   | -66.20%        |
| grain_extract | sse42  | 0.189483 | 0.012991   | 14.59x  | -93.14%        |
| grain_extract | avx2   | 0.189483 | 0.010193   | 18.59x  | -94.62%        |
| grain_merge   | scalar | 0.185452 | 0.065266   | 2.84x   | -64.81%        |
| grain_merge   | sse42  | 0.185452 | 0.013348   | 13.89x  | -92.80%        |
| grain_merge   | avx2   | 0.185452 | 0.010821   | 17.14x  | -94.17%        |
| divide        | scalar | 0.196253 | 0.052935   | 3.71x   | -73.03%        |
| divide        | sse42  | 0.196253 | 0.013584   | 14.45x  | -93.08%        |
| divide        | avx2   | 0.196253 | 0.010692   | 18.36x  | -94.55%        |
| overlay       | scalar | 0.264969 | 0.091859   | 2.88x   | -65.33%        |
| overlay       | sse42  | 0.264969 | 0.013770   | 19.24x  | -94.80%        |
| overlay       | avx2   | 0.264969 | 0.010714   | 24.73x  | -95.96%        |

<details>
<summary>Per-kernel results</summary>

| Case      | Input   | Mode          | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | ------- | ------------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | uint8   | normal        | scalar | 0.003810 | 0.001543   | 2.47x   | -59.50%        |
| 256x256   | uint8   | normal        | sse42  | 0.003810 | 0.000251   | 15.20x  | -93.42%        |
| 256x256   | uint8   | normal        | avx2   | 0.003810 | 0.000164   | 23.26x  | -95.70%        |
| 256x256   | uint8   | soft_light    | scalar | 0.007351 | 0.001728   | 4.25x   | -76.49%        |
| 256x256   | uint8   | soft_light    | sse42  | 0.007351 | 0.000236   | 31.18x  | -96.79%        |
| 256x256   | uint8   | soft_light    | avx2   | 0.007351 | 0.000188   | 39.20x  | -97.45%        |
| 256x256   | uint8   | lighten_only  | scalar | 0.005857 | 0.001853   | 3.16x   | -68.36%        |
| 256x256   | uint8   | lighten_only  | sse42  | 0.005857 | 0.000233   | 25.13x  | -96.02%        |
| 256x256   | uint8   | lighten_only  | avx2   | 0.005857 | 0.000191   | 30.74x  | -96.75%        |
| 256x256   | uint8   | screen        | scalar | 0.006309 | 0.001793   | 3.52x   | -71.58%        |
| 256x256   | uint8   | screen        | sse42  | 0.006309 | 0.000236   | 26.78x  | -96.27%        |
| 256x256   | uint8   | screen        | avx2   | 0.006309 | 0.000189   | 33.33x  | -97.00%        |
| 256x256   | uint8   | dodge         | scalar | 0.006082 | 0.001768   | 3.44x   | -70.93%        |
| 256x256   | uint8   | dodge         | sse42  | 0.006082 | 0.000260   | 23.38x  | -95.72%        |
| 256x256   | uint8   | dodge         | avx2   | 0.006082 | 0.000190   | 31.97x  | -96.87%        |
| 256x256   | uint8   | addition      | scalar | 0.006342 | 0.002145   | 2.96x   | -66.18%        |
| 256x256   | uint8   | addition      | sse42  | 0.006342 | 0.000271   | 23.42x  | -95.73%        |
| 256x256   | uint8   | addition      | avx2   | 0.006342 | 0.000228   | 27.78x  | -96.40%        |
| 256x256   | uint8   | darken_only   | scalar | 0.006420 | 0.001934   | 3.32x   | -69.88%        |
| 256x256   | uint8   | darken_only   | sse42  | 0.006420 | 0.000232   | 27.68x  | -96.39%        |
| 256x256   | uint8   | darken_only   | avx2   | 0.006420 | 0.000193   | 33.34x  | -97.00%        |
| 256x256   | uint8   | multiply      | scalar | 0.006249 | 0.002011   | 3.11x   | -67.82%        |
| 256x256   | uint8   | multiply      | sse42  | 0.006249 | 0.000229   | 27.29x  | -96.34%        |
| 256x256   | uint8   | multiply      | avx2   | 0.006249 | 0.000179   | 34.96x  | -97.14%        |
| 256x256   | uint8   | hard_light    | scalar | 0.007636 | 0.002794   | 2.73x   | -63.41%        |
| 256x256   | uint8   | hard_light    | sse42  | 0.007636 | 0.000257   | 29.74x  | -96.64%        |
| 256x256   | uint8   | hard_light    | avx2   | 0.007636 | 0.000192   | 39.76x  | -97.49%        |
| 256x256   | uint8   | difference    | scalar | 0.007852 | 0.001682   | 4.67x   | -78.58%        |
| 256x256   | uint8   | difference    | sse42  | 0.007852 | 0.000259   | 30.37x  | -96.71%        |
| 256x256   | uint8   | difference    | avx2   | 0.007852 | 0.000182   | 43.10x  | -97.68%        |
| 256x256   | uint8   | subtract      | scalar | 0.006579 | 0.001639   | 4.01x   | -75.08%        |
| 256x256   | uint8   | subtract      | sse42  | 0.006579 | 0.000266   | 24.71x  | -95.95%        |
| 256x256   | uint8   | subtract      | avx2   | 0.006579 | 0.000197   | 33.44x  | -97.01%        |
| 256x256   | uint8   | grain_extract | scalar | 0.005560 | 0.002011   | 2.76x   | -63.83%        |
| 256x256   | uint8   | grain_extract | sse42  | 0.005560 | 0.000228   | 24.44x  | -95.91%        |
| 256x256   | uint8   | grain_extract | avx2   | 0.005560 | 0.000190   | 29.31x  | -96.59%        |
| 256x256   | uint8   | grain_merge   | scalar | 0.005956 | 0.002031   | 2.93x   | -65.89%        |
| 256x256   | uint8   | grain_merge   | sse42  | 0.005956 | 0.000237   | 25.15x  | -96.02%        |
| 256x256   | uint8   | grain_merge   | avx2   | 0.005956 | 0.000194   | 30.74x  | -96.75%        |
| 256x256   | uint8   | divide        | scalar | 0.005806 | 0.001726   | 3.36x   | -70.27%        |
| 256x256   | uint8   | divide        | sse42  | 0.005806 | 0.000240   | 24.16x  | -95.86%        |
| 256x256   | uint8   | divide        | avx2   | 0.005806 | 0.000192   | 30.16x  | -96.68%        |
| 256x256   | uint8   | overlay       | scalar | 0.007054 | 0.002600   | 2.71x   | -63.14%        |
| 256x256   | uint8   | overlay       | sse42  | 0.007054 | 0.000274   | 25.72x  | -96.11%        |
| 256x256   | uint8   | overlay       | avx2   | 0.007054 | 0.000184   | 38.24x  | -97.38%        |
| 256x256   | float32 | normal        | scalar | 0.004299 | 0.000625   | 6.88x   | -85.47%        |
| 256x256   | float32 | normal        | sse42  | 0.004299 | 0.000314   | 13.69x  | -92.70%        |
| 256x256   | float32 | normal        | avx2   | 0.004299 | 0.000234   | 18.39x  | -94.56%        |
| 256x256   | float32 | soft_light    | scalar | 0.006786 | 0.000832   | 8.16x   | -87.74%        |
| 256x256   | float32 | soft_light    | sse42  | 0.006786 | 0.000371   | 18.28x  | -94.53%        |
| 256x256   | float32 | soft_light    | avx2   | 0.006786 | 0.000270   | 25.14x  | -96.02%        |
| 256x256   | float32 | lighten_only  | scalar | 0.005093 | 0.000893   | 5.70x   | -82.46%        |
| 256x256   | float32 | lighten_only  | sse42  | 0.005093 | 0.000362   | 14.07x  | -92.89%        |
| 256x256   | float32 | lighten_only  | avx2   | 0.005093 | 0.000272   | 18.75x  | -94.67%        |
| 256x256   | float32 | screen        | scalar | 0.005185 | 0.000866   | 5.99x   | -83.29%        |
| 256x256   | float32 | screen        | sse42  | 0.005185 | 0.000371   | 13.96x  | -92.84%        |
| 256x256   | float32 | screen        | avx2   | 0.005185 | 0.000262   | 19.77x  | -94.94%        |
| 256x256   | float32 | dodge         | scalar | 0.005516 | 0.000828   | 6.66x   | -84.99%        |
| 256x256   | float32 | dodge         | sse42  | 0.005516 | 0.000360   | 15.33x  | -93.48%        |
| 256x256   | float32 | dodge         | avx2   | 0.005516 | 0.000265   | 20.80x  | -95.19%        |
| 256x256   | float32 | addition      | scalar | 0.005744 | 0.001415   | 4.06x   | -75.37%        |
| 256x256   | float32 | addition      | sse42  | 0.005744 | 0.000378   | 15.18x  | -93.41%        |
| 256x256   | float32 | addition      | avx2   | 0.005744 | 0.000288   | 19.93x  | -94.98%        |
| 256x256   | float32 | darken_only   | scalar | 0.005076 | 0.000888   | 5.71x   | -82.50%        |
| 256x256   | float32 | darken_only   | sse42  | 0.005076 | 0.000367   | 13.82x  | -92.76%        |
| 256x256   | float32 | darken_only   | avx2   | 0.005076 | 0.000260   | 19.52x  | -94.88%        |
| 256x256   | float32 | multiply      | scalar | 0.005120 | 0.000720   | 7.12x   | -85.95%        |
| 256x256   | float32 | multiply      | sse42  | 0.005120 | 0.000363   | 14.10x  | -92.91%        |
| 256x256   | float32 | multiply      | avx2   | 0.005120 | 0.000257   | 19.95x  | -94.99%        |
| 256x256   | float32 | hard_light    | scalar | 0.007049 | 0.001947   | 3.62x   | -72.38%        |
| 256x256   | float32 | hard_light    | sse42  | 0.007049 | 0.000360   | 19.56x  | -94.89%        |
| 256x256   | float32 | hard_light    | avx2   | 0.007049 | 0.000265   | 26.64x  | -96.25%        |
| 256x256   | float32 | difference    | scalar | 0.007089 | 0.000737   | 9.62x   | -89.61%        |
| 256x256   | float32 | difference    | sse42  | 0.007089 | 0.000369   | 19.22x  | -94.80%        |
| 256x256   | float32 | difference    | avx2   | 0.007089 | 0.000262   | 27.08x  | -96.31%        |
| 256x256   | float32 | subtract      | scalar | 0.005926 | 0.000982   | 6.04x   | -83.44%        |
| 256x256   | float32 | subtract      | sse42  | 0.005926 | 0.000391   | 15.15x  | -93.40%        |
| 256x256   | float32 | subtract      | avx2   | 0.005926 | 0.000274   | 21.61x  | -95.37%        |
| 256x256   | float32 | grain_extract | scalar | 0.005352 | 0.001152   | 4.64x   | -78.47%        |
| 256x256   | float32 | grain_extract | sse42  | 0.005352 | 0.000360   | 14.85x  | -93.27%        |
| 256x256   | float32 | grain_extract | avx2   | 0.005352 | 0.000263   | 20.34x  | -95.08%        |
| 256x256   | float32 | grain_merge   | scalar | 0.005478 | 0.001129   | 4.85x   | -79.39%        |
| 256x256   | float32 | grain_merge   | sse42  | 0.005478 | 0.000359   | 15.27x  | -93.45%        |
| 256x256   | float32 | grain_merge   | avx2   | 0.005478 | 0.000252   | 21.73x  | -95.40%        |
| 256x256   | float32 | divide        | scalar | 0.005302 | 0.000788   | 6.73x   | -85.14%        |
| 256x256   | float32 | divide        | sse42  | 0.005302 | 0.000357   | 14.84x  | -93.26%        |
| 256x256   | float32 | divide        | avx2   | 0.005302 | 0.000259   | 20.47x  | -95.11%        |
| 256x256   | float32 | overlay       | scalar | 0.006717 | 0.001927   | 3.49x   | -71.32%        |
| 256x256   | float32 | overlay       | sse42  | 0.006717 | 0.000363   | 18.53x  | -94.60%        |
| 256x256   | float32 | overlay       | avx2   | 0.006717 | 0.000270   | 24.91x  | -95.99%        |
| 512x512   | uint8   | normal        | scalar | 0.026708 | 0.005783   | 4.62x   | -78.35%        |
| 512x512   | uint8   | normal        | sse42  | 0.026708 | 0.000792   | 33.74x  | -97.04%        |
| 512x512   | uint8   | normal        | avx2   | 0.026708 | 0.000629   | 42.49x  | -97.65%        |
| 512x512   | uint8   | soft_light    | scalar | 0.037203 | 0.007339   | 5.07x   | -80.27%        |
| 512x512   | uint8   | soft_light    | sse42  | 0.037203 | 0.000991   | 37.54x  | -97.34%        |
| 512x512   | uint8   | soft_light    | avx2   | 0.037203 | 0.000799   | 46.57x  | -97.85%        |
| 512x512   | uint8   | lighten_only  | scalar | 0.033255 | 0.007448   | 4.47x   | -77.60%        |
| 512x512   | uint8   | lighten_only  | sse42  | 0.033255 | 0.000897   | 37.08x  | -97.30%        |
| 512x512   | uint8   | lighten_only  | avx2   | 0.033255 | 0.000715   | 46.53x  | -97.85%        |
| 512x512   | uint8   | screen        | scalar | 0.033435 | 0.007094   | 4.71x   | -78.78%        |
| 512x512   | uint8   | screen        | sse42  | 0.033435 | 0.000889   | 37.61x  | -97.34%        |
| 512x512   | uint8   | screen        | avx2   | 0.033435 | 0.000733   | 45.64x  | -97.81%        |
| 512x512   | uint8   | dodge         | scalar | 0.031094 | 0.007776   | 4.00x   | -74.99%        |
| 512x512   | uint8   | dodge         | sse42  | 0.031094 | 0.000995   | 31.23x  | -96.80%        |
| 512x512   | uint8   | dodge         | avx2   | 0.031094 | 0.000744   | 41.82x  | -97.61%        |
| 512x512   | uint8   | addition      | scalar | 0.030222 | 0.008658   | 3.49x   | -71.35%        |
| 512x512   | uint8   | addition      | sse42  | 0.030222 | 0.001043   | 28.98x  | -96.55%        |
| 512x512   | uint8   | addition      | avx2   | 0.030222 | 0.000780   | 38.74x  | -97.42%        |
| 512x512   | uint8   | darken_only   | scalar | 0.029529 | 0.007447   | 3.97x   | -74.78%        |
| 512x512   | uint8   | darken_only   | sse42  | 0.029529 | 0.000910   | 32.45x  | -96.92%        |
| 512x512   | uint8   | darken_only   | avx2   | 0.029529 | 0.000717   | 41.17x  | -97.57%        |
| 512x512   | uint8   | multiply      | scalar | 0.029347 | 0.007125   | 4.12x   | -75.72%        |
| 512x512   | uint8   | multiply      | sse42  | 0.029347 | 0.000925   | 31.74x  | -96.85%        |
| 512x512   | uint8   | multiply      | avx2   | 0.029347 | 0.000722   | 40.64x  | -97.54%        |
| 512x512   | uint8   | hard_light    | scalar | 0.039374 | 0.011101   | 3.55x   | -71.81%        |
| 512x512   | uint8   | hard_light    | sse42  | 0.039374 | 0.001019   | 38.66x  | -97.41%        |
| 512x512   | uint8   | hard_light    | avx2   | 0.039374 | 0.000746   | 52.78x  | -98.11%        |
| 512x512   | uint8   | difference    | scalar | 0.036963 | 0.007017   | 5.27x   | -81.02%        |
| 512x512   | uint8   | difference    | sse42  | 0.036963 | 0.000878   | 42.10x  | -97.62%        |
| 512x512   | uint8   | difference    | avx2   | 0.036963 | 0.000714   | 51.77x  | -98.07%        |
| 512x512   | uint8   | subtract      | scalar | 0.029045 | 0.006862   | 4.23x   | -76.37%        |
| 512x512   | uint8   | subtract      | sse42  | 0.029045 | 0.001056   | 27.51x  | -96.36%        |
| 512x512   | uint8   | subtract      | avx2   | 0.029045 | 0.000779   | 37.29x  | -97.32%        |
| 512x512   | uint8   | grain_extract | scalar | 0.033970 | 0.008449   | 4.02x   | -75.13%        |
| 512x512   | uint8   | grain_extract | sse42  | 0.033970 | 0.000925   | 36.73x  | -97.28%        |
| 512x512   | uint8   | grain_extract | avx2   | 0.033970 | 0.000742   | 45.80x  | -97.82%        |
| 512x512   | uint8   | grain_merge   | scalar | 0.031947 | 0.008983   | 3.56x   | -71.88%        |
| 512x512   | uint8   | grain_merge   | sse42  | 0.031947 | 0.000987   | 32.37x  | -96.91%        |
| 512x512   | uint8   | grain_merge   | avx2   | 0.031947 | 0.000759   | 42.07x  | -97.62%        |
| 512x512   | uint8   | divide        | scalar | 0.033853 | 0.011809   | 2.87x   | -65.12%        |
| 512x512   | uint8   | divide        | sse42  | 0.033853 | 0.001345   | 25.18x  | -96.03%        |
| 512x512   | uint8   | divide        | avx2   | 0.033853 | 0.000834   | 40.60x  | -97.54%        |
| 512x512   | uint8   | overlay       | scalar | 0.041037 | 0.011356   | 3.61x   | -72.33%        |
| 512x512   | uint8   | overlay       | sse42  | 0.041037 | 0.001106   | 37.09x  | -97.30%        |
| 512x512   | uint8   | overlay       | avx2   | 0.041037 | 0.000799   | 51.38x  | -98.05%        |
| 512x512   | float32 | normal        | scalar | 0.024028 | 0.002995   | 8.02x   | -87.53%        |
| 512x512   | float32 | normal        | sse42  | 0.024028 | 0.001301   | 18.47x  | -94.59%        |
| 512x512   | float32 | normal        | avx2   | 0.024028 | 0.001037   | 23.18x  | -95.69%        |
| 512x512   | float32 | soft_light    | scalar | 0.034120 | 0.003673   | 9.29x   | -89.23%        |
| 512x512   | float32 | soft_light    | sse42  | 0.034120 | 0.001439   | 23.71x  | -95.78%        |
| 512x512   | float32 | soft_light    | avx2   | 0.034120 | 0.001106   | 30.86x  | -96.76%        |
| 512x512   | float32 | lighten_only  | scalar | 0.025875 | 0.003999   | 6.47x   | -84.55%        |
| 512x512   | float32 | lighten_only  | sse42  | 0.025875 | 0.001521   | 17.01x  | -94.12%        |
| 512x512   | float32 | lighten_only  | avx2   | 0.025875 | 0.001135   | 22.80x  | -95.61%        |
| 512x512   | float32 | screen        | scalar | 0.026872 | 0.003508   | 7.66x   | -86.95%        |
| 512x512   | float32 | screen        | sse42  | 0.026872 | 0.001518   | 17.70x  | -94.35%        |
| 512x512   | float32 | screen        | avx2   | 0.026872 | 0.001097   | 24.49x  | -95.92%        |
| 512x512   | float32 | dodge         | scalar | 0.028592 | 0.003917   | 7.30x   | -86.30%        |
| 512x512   | float32 | dodge         | sse42  | 0.028592 | 0.001493   | 19.15x  | -94.78%        |
| 512x512   | float32 | dodge         | avx2   | 0.028592 | 0.001424   | 20.07x  | -95.02%        |
| 512x512   | float32 | addition      | scalar | 0.032270 | 0.007994   | 4.04x   | -75.23%        |
| 512x512   | float32 | addition      | sse42  | 0.032270 | 0.001633   | 19.76x  | -94.94%        |
| 512x512   | float32 | addition      | avx2   | 0.032270 | 0.001667   | 19.36x  | -94.83%        |
| 512x512   | float32 | darken_only   | scalar | 0.030495 | 0.004216   | 7.23x   | -86.17%        |
| 512x512   | float32 | darken_only   | sse42  | 0.030495 | 0.001558   | 19.58x  | -94.89%        |
| 512x512   | float32 | darken_only   | avx2   | 0.030495 | 0.001301   | 23.44x  | -95.73%        |
| 512x512   | float32 | multiply      | scalar | 0.033015 | 0.004365   | 7.56x   | -86.78%        |
| 512x512   | float32 | multiply      | sse42  | 0.033015 | 0.001644   | 20.08x  | -95.02%        |
| 512x512   | float32 | multiply      | avx2   | 0.033015 | 0.001381   | 23.91x  | -95.82%        |
| 512x512   | float32 | hard_light    | scalar | 0.043344 | 0.008828   | 4.91x   | -79.63%        |
| 512x512   | float32 | hard_light    | sse42  | 0.043344 | 0.001710   | 25.34x  | -96.05%        |
| 512x512   | float32 | hard_light    | avx2   | 0.043344 | 0.001183   | 36.65x  | -97.27%        |
| 512x512   | float32 | difference    | scalar | 0.036534 | 0.003715   | 9.83x   | -89.83%        |
| 512x512   | float32 | difference    | sse42  | 0.036534 | 0.001630   | 22.41x  | -95.54%        |
| 512x512   | float32 | difference    | avx2   | 0.036534 | 0.001270   | 28.76x  | -96.52%        |
| 512x512   | float32 | subtract      | scalar | 0.028788 | 0.005021   | 5.73x   | -82.56%        |
| 512x512   | float32 | subtract      | sse42  | 0.028788 | 0.001617   | 17.80x  | -94.38%        |
| 512x512   | float32 | subtract      | avx2   | 0.028788 | 0.001579   | 18.23x  | -94.51%        |
| 512x512   | float32 | grain_extract | scalar | 0.028560 | 0.005030   | 5.68x   | -82.39%        |
| 512x512   | float32 | grain_extract | sse42  | 0.028560 | 0.001557   | 18.34x  | -94.55%        |
| 512x512   | float32 | grain_extract | avx2   | 0.028560 | 0.001660   | 17.21x  | -94.19%        |
| 512x512   | float32 | grain_merge   | scalar | 0.028238 | 0.005201   | 5.43x   | -81.58%        |
| 512x512   | float32 | grain_merge   | sse42  | 0.028238 | 0.001503   | 18.79x  | -94.68%        |
| 512x512   | float32 | grain_merge   | avx2   | 0.028238 | 0.001170   | 24.13x  | -95.86%        |
| 512x512   | float32 | divide        | scalar | 0.030914 | 0.004561   | 6.78x   | -85.25%        |
| 512x512   | float32 | divide        | sse42  | 0.030914 | 0.001543   | 20.04x  | -95.01%        |
| 512x512   | float32 | divide        | avx2   | 0.030914 | 0.001316   | 23.50x  | -95.74%        |
| 512x512   | float32 | overlay       | scalar | 0.035431 | 0.007866   | 4.50x   | -77.80%        |
| 512x512   | float32 | overlay       | sse42  | 0.035431 | 0.001517   | 23.35x  | -95.72%        |
| 512x512   | float32 | overlay       | avx2   | 0.035431 | 0.001212   | 29.24x  | -96.58%        |
| 1024x1024 | uint8   | normal        | scalar | 0.079097 | 0.023290   | 3.40x   | -70.56%        |
| 1024x1024 | uint8   | normal        | sse42  | 0.079097 | 0.003165   | 24.99x  | -96.00%        |
| 1024x1024 | uint8   | normal        | avx2   | 0.079097 | 0.002518   | 31.42x  | -96.82%        |
| 1024x1024 | uint8   | soft_light    | scalar | 0.114851 | 0.031124   | 3.69x   | -72.90%        |
| 1024x1024 | uint8   | soft_light    | sse42  | 0.114851 | 0.004207   | 27.30x  | -96.34%        |
| 1024x1024 | uint8   | soft_light    | avx2   | 0.114851 | 0.003031   | 37.89x  | -97.36%        |
| 1024x1024 | uint8   | lighten_only  | scalar | 0.087298 | 0.032806   | 2.66x   | -62.42%        |
| 1024x1024 | uint8   | lighten_only  | sse42  | 0.087298 | 0.004040   | 21.61x  | -95.37%        |
| 1024x1024 | uint8   | lighten_only  | avx2   | 0.087298 | 0.003299   | 26.46x  | -96.22%        |
| 1024x1024 | uint8   | screen        | scalar | 0.095733 | 0.026597   | 3.60x   | -72.22%        |
| 1024x1024 | uint8   | screen        | sse42  | 0.095733 | 0.003508   | 27.29x  | -96.34%        |
| 1024x1024 | uint8   | screen        | avx2   | 0.095733 | 0.002878   | 33.26x  | -96.99%        |
| 1024x1024 | uint8   | dodge         | scalar | 0.092382 | 0.028299   | 3.26x   | -69.37%        |
| 1024x1024 | uint8   | dodge         | sse42  | 0.092382 | 0.003932   | 23.49x  | -95.74%        |
| 1024x1024 | uint8   | dodge         | avx2   | 0.092382 | 0.002973   | 31.08x  | -96.78%        |
| 1024x1024 | uint8   | addition      | scalar | 0.079126 | 0.033851   | 2.34x   | -57.22%        |
| 1024x1024 | uint8   | addition      | sse42  | 0.079126 | 0.004249   | 18.62x  | -94.63%        |
| 1024x1024 | uint8   | addition      | avx2   | 0.079126 | 0.003175   | 24.92x  | -95.99%        |
| 1024x1024 | uint8   | darken_only   | scalar | 0.083960 | 0.030605   | 2.74x   | -63.55%        |
| 1024x1024 | uint8   | darken_only   | sse42  | 0.083960 | 0.003597   | 23.34x  | -95.72%        |
| 1024x1024 | uint8   | darken_only   | avx2   | 0.083960 | 0.003420   | 24.55x  | -95.93%        |
| 1024x1024 | uint8   | multiply      | scalar | 0.080073 | 0.026819   | 2.99x   | -66.51%        |
| 1024x1024 | uint8   | multiply      | sse42  | 0.080073 | 0.003479   | 23.02x  | -95.66%        |
| 1024x1024 | uint8   | multiply      | avx2   | 0.080073 | 0.002934   | 27.29x  | -96.34%        |
| 1024x1024 | uint8   | hard_light    | scalar | 0.124632 | 0.044827   | 2.78x   | -64.03%        |
| 1024x1024 | uint8   | hard_light    | sse42  | 0.124632 | 0.004338   | 28.73x  | -96.52%        |
| 1024x1024 | uint8   | hard_light    | avx2   | 0.124632 | 0.003081   | 40.45x  | -97.53%        |
| 1024x1024 | uint8   | difference    | scalar | 0.122693 | 0.026538   | 4.62x   | -78.37%        |
| 1024x1024 | uint8   | difference    | sse42  | 0.122693 | 0.003674   | 33.40x  | -97.01%        |
| 1024x1024 | uint8   | difference    | avx2   | 0.122693 | 0.002863   | 42.85x  | -97.67%        |
| 1024x1024 | uint8   | subtract      | scalar | 0.080838 | 0.025718   | 3.14x   | -68.19%        |
| 1024x1024 | uint8   | subtract      | sse42  | 0.080838 | 0.004157   | 19.45x  | -94.86%        |
| 1024x1024 | uint8   | subtract      | avx2   | 0.080838 | 0.003086   | 26.20x  | -96.18%        |
| 1024x1024 | uint8   | grain_extract | scalar | 0.083965 | 0.035037   | 2.40x   | -58.27%        |
| 1024x1024 | uint8   | grain_extract | sse42  | 0.083965 | 0.004233   | 19.83x  | -94.96%        |
| 1024x1024 | uint8   | grain_extract | avx2   | 0.083965 | 0.003120   | 26.91x  | -96.28%        |
| 1024x1024 | uint8   | grain_merge   | scalar | 0.086469 | 0.033248   | 2.60x   | -61.55%        |
| 1024x1024 | uint8   | grain_merge   | sse42  | 0.086469 | 0.003746   | 23.08x  | -95.67%        |
| 1024x1024 | uint8   | grain_merge   | avx2   | 0.086469 | 0.002939   | 29.42x  | -96.60%        |
| 1024x1024 | uint8   | divide        | scalar | 0.086696 | 0.028865   | 3.00x   | -66.71%        |
| 1024x1024 | uint8   | divide        | sse42  | 0.086696 | 0.004202   | 20.63x  | -95.15%        |
| 1024x1024 | uint8   | divide        | avx2   | 0.086696 | 0.002946   | 29.42x  | -96.60%        |
| 1024x1024 | uint8   | overlay       | scalar | 0.115668 | 0.042196   | 2.74x   | -63.52%        |
| 1024x1024 | uint8   | overlay       | sse42  | 0.115668 | 0.004007   | 28.87x  | -96.54%        |
| 1024x1024 | uint8   | overlay       | avx2   | 0.115668 | 0.002931   | 39.47x  | -97.47%        |
| 1024x1024 | float32 | normal        | scalar | 0.069555 | 0.010896   | 6.38x   | -84.33%        |
| 1024x1024 | float32 | normal        | sse42  | 0.069555 | 0.005124   | 13.57x  | -92.63%        |
| 1024x1024 | float32 | normal        | avx2   | 0.069555 | 0.004175   | 16.66x  | -94.00%        |
| 1024x1024 | float32 | soft_light    | scalar | 0.102902 | 0.014696   | 7.00x   | -85.72%        |
| 1024x1024 | float32 | soft_light    | sse42  | 0.102902 | 0.005655   | 18.20x  | -94.50%        |
| 1024x1024 | float32 | soft_light    | avx2   | 0.102902 | 0.004339   | 23.72x  | -95.78%        |
| 1024x1024 | float32 | lighten_only  | scalar | 0.076933 | 0.015583   | 4.94x   | -79.74%        |
| 1024x1024 | float32 | lighten_only  | sse42  | 0.076933 | 0.005937   | 12.96x  | -92.28%        |
| 1024x1024 | float32 | lighten_only  | avx2   | 0.076933 | 0.004617   | 16.66x  | -94.00%        |
| 1024x1024 | float32 | screen        | scalar | 0.078085 | 0.013062   | 5.98x   | -83.27%        |
| 1024x1024 | float32 | screen        | sse42  | 0.078085 | 0.005808   | 13.44x  | -92.56%        |
| 1024x1024 | float32 | screen        | avx2   | 0.078085 | 0.004328   | 18.04x  | -94.46%        |
| 1024x1024 | float32 | dodge         | scalar | 0.079869 | 0.013915   | 5.74x   | -82.58%        |
| 1024x1024 | float32 | dodge         | sse42  | 0.079869 | 0.005795   | 13.78x  | -92.74%        |
| 1024x1024 | float32 | dodge         | avx2   | 0.079869 | 0.004442   | 17.98x  | -94.44%        |
| 1024x1024 | float32 | addition      | scalar | 0.084905 | 0.024629   | 3.45x   | -70.99%        |
| 1024x1024 | float32 | addition      | sse42  | 0.084905 | 0.006120   | 13.87x  | -92.79%        |
| 1024x1024 | float32 | addition      | avx2   | 0.084905 | 0.006599   | 12.87x  | -92.23%        |
| 1024x1024 | float32 | darken_only   | scalar | 0.081133 | 0.017659   | 4.59x   | -78.23%        |
| 1024x1024 | float32 | darken_only   | sse42  | 0.081133 | 0.006301   | 12.88x  | -92.23%        |
| 1024x1024 | float32 | darken_only   | avx2   | 0.081133 | 0.004655   | 17.43x  | -94.26%        |
| 1024x1024 | float32 | multiply      | scalar | 0.080166 | 0.013328   | 6.01x   | -83.37%        |
| 1024x1024 | float32 | multiply      | sse42  | 0.080166 | 0.005982   | 13.40x  | -92.54%        |
| 1024x1024 | float32 | multiply      | avx2   | 0.080166 | 0.004392   | 18.25x  | -94.52%        |
| 1024x1024 | float32 | hard_light    | scalar | 0.121157 | 0.032936   | 3.68x   | -72.82%        |
| 1024x1024 | float32 | hard_light    | sse42  | 0.121157 | 0.006233   | 19.44x  | -94.86%        |
| 1024x1024 | float32 | hard_light    | avx2   | 0.121157 | 0.004954   | 24.46x  | -95.91%        |
| 1024x1024 | float32 | difference    | scalar | 0.111687 | 0.013303   | 8.40x   | -88.09%        |
| 1024x1024 | float32 | difference    | sse42  | 0.111687 | 0.005783   | 19.31x  | -94.82%        |
| 1024x1024 | float32 | difference    | avx2   | 0.111687 | 0.005060   | 22.07x  | -95.47%        |
| 1024x1024 | float32 | subtract      | scalar | 0.083133 | 0.017326   | 4.80x   | -79.16%        |
| 1024x1024 | float32 | subtract      | sse42  | 0.083133 | 0.006739   | 12.34x  | -91.89%        |
| 1024x1024 | float32 | subtract      | avx2   | 0.083133 | 0.006016   | 13.82x  | -92.76%        |
| 1024x1024 | float32 | grain_extract | scalar | 0.083836 | 0.020826   | 4.03x   | -75.16%        |
| 1024x1024 | float32 | grain_extract | sse42  | 0.083836 | 0.006328   | 13.25x  | -92.45%        |
| 1024x1024 | float32 | grain_extract | avx2   | 0.083836 | 0.004650   | 18.03x  | -94.45%        |
| 1024x1024 | float32 | grain_merge   | scalar | 0.083708 | 0.019454   | 4.30x   | -76.76%        |
| 1024x1024 | float32 | grain_merge   | sse42  | 0.083708 | 0.005787   | 14.46x  | -93.09%        |
| 1024x1024 | float32 | grain_merge   | avx2   | 0.083708 | 0.004490   | 18.64x  | -94.64%        |
| 1024x1024 | float32 | divide        | scalar | 0.082689 | 0.013798   | 5.99x   | -83.31%        |
| 1024x1024 | float32 | divide        | sse42  | 0.082689 | 0.006490   | 12.74x  | -92.15%        |
| 1024x1024 | float32 | divide        | avx2   | 0.082689 | 0.004365   | 18.95x  | -94.72%        |
| 1024x1024 | float32 | overlay       | scalar | 0.103877 | 0.029946   | 3.47x   | -71.17%        |
| 1024x1024 | float32 | overlay       | sse42  | 0.103877 | 0.005679   | 18.29x  | -94.53%        |
| 1024x1024 | float32 | overlay       | avx2   | 0.103877 | 0.004425   | 23.48x  | -95.74%        |
| 2048x2048 | uint8   | normal        | scalar | 0.285544 | 0.086102   | 3.32x   | -69.85%        |
| 2048x2048 | uint8   | normal        | sse42  | 0.285544 | 0.012280   | 23.25x  | -95.70%        |
| 2048x2048 | uint8   | normal        | avx2   | 0.285544 | 0.010094   | 28.29x  | -96.47%        |
| 2048x2048 | uint8   | soft_light    | scalar | 0.403962 | 0.109393   | 3.69x   | -72.92%        |
| 2048x2048 | uint8   | soft_light    | sse42  | 0.403962 | 0.014844   | 27.21x  | -96.33%        |
| 2048x2048 | uint8   | soft_light    | avx2   | 0.403962 | 0.011588   | 34.86x  | -97.13%        |
| 2048x2048 | uint8   | lighten_only  | scalar | 0.281837 | 0.116014   | 2.43x   | -58.84%        |
| 2048x2048 | uint8   | lighten_only  | sse42  | 0.281837 | 0.013857   | 20.34x  | -95.08%        |
| 2048x2048 | uint8   | lighten_only  | avx2   | 0.281837 | 0.012539   | 22.48x  | -95.55%        |
| 2048x2048 | uint8   | screen        | scalar | 0.308021 | 0.109675   | 2.81x   | -64.39%        |
| 2048x2048 | uint8   | screen        | sse42  | 0.308021 | 0.014049   | 21.92x  | -95.44%        |
| 2048x2048 | uint8   | screen        | avx2   | 0.308021 | 0.011498   | 26.79x  | -96.27%        |
| 2048x2048 | uint8   | dodge         | scalar | 0.310590 | 0.110010   | 2.82x   | -64.58%        |
| 2048x2048 | uint8   | dodge         | sse42  | 0.310590 | 0.016272   | 19.09x  | -94.76%        |
| 2048x2048 | uint8   | dodge         | avx2   | 0.310590 | 0.011813   | 26.29x  | -96.20%        |
| 2048x2048 | uint8   | addition      | scalar | 0.292832 | 0.132849   | 2.20x   | -54.63%        |
| 2048x2048 | uint8   | addition      | sse42  | 0.292832 | 0.016501   | 17.75x  | -94.37%        |
| 2048x2048 | uint8   | addition      | avx2   | 0.292832 | 0.012364   | 23.68x  | -95.78%        |
| 2048x2048 | uint8   | darken_only   | scalar | 0.283852 | 0.121433   | 2.34x   | -57.22%        |
| 2048x2048 | uint8   | darken_only   | sse42  | 0.283852 | 0.014353   | 19.78x  | -94.94%        |
| 2048x2048 | uint8   | darken_only   | avx2   | 0.283852 | 0.011779   | 24.10x  | -95.85%        |
| 2048x2048 | uint8   | multiply      | scalar | 0.302628 | 0.106341   | 2.85x   | -64.86%        |
| 2048x2048 | uint8   | multiply      | sse42  | 0.302628 | 0.013812   | 21.91x  | -95.44%        |
| 2048x2048 | uint8   | multiply      | avx2   | 0.302628 | 0.011449   | 26.43x  | -96.22%        |
| 2048x2048 | uint8   | hard_light    | scalar | 0.457081 | 0.189014   | 2.42x   | -58.65%        |
| 2048x2048 | uint8   | hard_light    | sse42  | 0.457081 | 0.015933   | 28.69x  | -96.51%        |
| 2048x2048 | uint8   | hard_light    | avx2   | 0.457081 | 0.012471   | 36.65x  | -97.27%        |
| 2048x2048 | uint8   | difference    | scalar | 0.412539 | 0.107889   | 3.82x   | -73.85%        |
| 2048x2048 | uint8   | difference    | sse42  | 0.412539 | 0.014177   | 29.10x  | -96.56%        |
| 2048x2048 | uint8   | difference    | avx2   | 0.412539 | 0.011263   | 36.63x  | -97.27%        |
| 2048x2048 | uint8   | subtract      | scalar | 0.316134 | 0.107310   | 2.95x   | -66.06%        |
| 2048x2048 | uint8   | subtract      | sse42  | 0.316134 | 0.016646   | 18.99x  | -94.73%        |
| 2048x2048 | uint8   | subtract      | avx2   | 0.316134 | 0.013206   | 23.94x  | -95.82%        |
| 2048x2048 | uint8   | grain_extract | scalar | 0.304469 | 0.127421   | 2.39x   | -58.15%        |
| 2048x2048 | uint8   | grain_extract | sse42  | 0.304469 | 0.016244   | 18.74x  | -94.66%        |
| 2048x2048 | uint8   | grain_extract | avx2   | 0.304469 | 0.011730   | 25.96x  | -96.15%        |
| 2048x2048 | uint8   | grain_merge   | scalar | 0.304721 | 0.131973   | 2.31x   | -56.69%        |
| 2048x2048 | uint8   | grain_merge   | sse42  | 0.304721 | 0.014688   | 20.75x  | -95.18%        |
| 2048x2048 | uint8   | grain_merge   | avx2   | 0.304721 | 0.011822   | 25.78x  | -96.12%        |
| 2048x2048 | uint8   | divide        | scalar | 0.312665 | 0.110475   | 2.83x   | -64.67%        |
| 2048x2048 | uint8   | divide        | sse42  | 0.312665 | 0.015904   | 19.66x  | -94.91%        |
| 2048x2048 | uint8   | divide        | avx2   | 0.312665 | 0.011500   | 27.19x  | -96.32%        |
| 2048x2048 | uint8   | overlay       | scalar | 0.418359 | 0.167610   | 2.50x   | -59.94%        |
| 2048x2048 | uint8   | overlay       | sse42  | 0.418359 | 0.016282   | 25.70x  | -96.11%        |
| 2048x2048 | uint8   | overlay       | avx2   | 0.418359 | 0.011464   | 36.49x  | -97.26%        |
| 2048x2048 | float32 | normal        | scalar | 0.270263 | 0.047806   | 5.65x   | -82.31%        |
| 2048x2048 | float32 | normal        | sse42  | 0.270263 | 0.027768   | 9.73x   | -89.73%        |
| 2048x2048 | float32 | normal        | avx2   | 0.270263 | 0.024311   | 11.12x  | -91.00%        |
| 2048x2048 | float32 | soft_light    | scalar | 0.382659 | 0.061356   | 6.24x   | -83.97%        |
| 2048x2048 | float32 | soft_light    | sse42  | 0.382659 | 0.029355   | 13.04x  | -92.33%        |
| 2048x2048 | float32 | soft_light    | avx2   | 0.382659 | 0.024977   | 15.32x  | -93.47%        |
| 2048x2048 | float32 | lighten_only  | scalar | 0.267144 | 0.065285   | 4.09x   | -75.56%        |
| 2048x2048 | float32 | lighten_only  | sse42  | 0.267144 | 0.030809   | 8.67x   | -88.47%        |
| 2048x2048 | float32 | lighten_only  | avx2   | 0.267144 | 0.025296   | 10.56x  | -90.53%        |
| 2048x2048 | float32 | screen        | scalar | 0.293644 | 0.057472   | 5.11x   | -80.43%        |
| 2048x2048 | float32 | screen        | sse42  | 0.293644 | 0.029889   | 9.82x   | -89.82%        |
| 2048x2048 | float32 | screen        | avx2   | 0.293644 | 0.025053   | 11.72x  | -91.47%        |
| 2048x2048 | float32 | dodge         | scalar | 0.292825 | 0.060815   | 4.82x   | -79.23%        |
| 2048x2048 | float32 | dodge         | sse42  | 0.292825 | 0.031481   | 9.30x   | -89.25%        |
| 2048x2048 | float32 | dodge         | avx2   | 0.292825 | 0.025265   | 11.59x  | -91.37%        |
| 2048x2048 | float32 | addition      | scalar | 0.281187 | 0.101851   | 2.76x   | -63.78%        |
| 2048x2048 | float32 | addition      | sse42  | 0.281187 | 0.030983   | 9.08x   | -88.98%        |
| 2048x2048 | float32 | addition      | avx2   | 0.281187 | 0.026039   | 10.80x  | -90.74%        |
| 2048x2048 | float32 | darken_only   | scalar | 0.282692 | 0.065644   | 4.31x   | -76.78%        |
| 2048x2048 | float32 | darken_only   | sse42  | 0.282692 | 0.031338   | 9.02x   | -88.91%        |
| 2048x2048 | float32 | darken_only   | avx2   | 0.282692 | 0.026590   | 10.63x  | -90.59%        |
| 2048x2048 | float32 | multiply      | scalar | 0.276169 | 0.056222   | 4.91x   | -79.64%        |
| 2048x2048 | float32 | multiply      | sse42  | 0.276169 | 0.030842   | 8.95x   | -88.83%        |
| 2048x2048 | float32 | multiply      | avx2   | 0.276169 | 0.025272   | 10.93x  | -90.85%        |
| 2048x2048 | float32 | hard_light    | scalar | 0.440073 | 0.134456   | 3.27x   | -69.45%        |
| 2048x2048 | float32 | hard_light    | sse42  | 0.440073 | 0.029255   | 15.04x  | -93.35%        |
| 2048x2048 | float32 | hard_light    | avx2   | 0.440073 | 0.028361   | 15.52x  | -93.56%        |
| 2048x2048 | float32 | difference    | scalar | 0.391009 | 0.056623   | 6.91x   | -85.52%        |
| 2048x2048 | float32 | difference    | sse42  | 0.391009 | 0.031040   | 12.60x  | -92.06%        |
| 2048x2048 | float32 | difference    | avx2   | 0.391009 | 0.026991   | 14.49x  | -93.10%        |
| 2048x2048 | float32 | subtract      | scalar | 0.280611 | 0.070279   | 3.99x   | -74.95%        |
| 2048x2048 | float32 | subtract      | sse42  | 0.280611 | 0.031938   | 8.79x   | -88.62%        |
| 2048x2048 | float32 | subtract      | avx2   | 0.280611 | 0.025710   | 10.91x  | -90.84%        |
| 2048x2048 | float32 | grain_extract | scalar | 0.287226 | 0.080460   | 3.57x   | -71.99%        |
| 2048x2048 | float32 | grain_extract | sse42  | 0.287226 | 0.030412   | 9.44x   | -89.41%        |
| 2048x2048 | float32 | grain_extract | avx2   | 0.287226 | 0.024827   | 11.57x  | -91.36%        |
| 2048x2048 | float32 | grain_merge   | scalar | 0.287428 | 0.081839   | 3.51x   | -71.53%        |
| 2048x2048 | float32 | grain_merge   | sse42  | 0.287428 | 0.029801   | 9.64x   | -89.63%        |
| 2048x2048 | float32 | grain_merge   | avx2   | 0.287428 | 0.026232   | 10.96x  | -90.87%        |
| 2048x2048 | float32 | divide        | scalar | 0.295609 | 0.058118   | 5.09x   | -80.34%        |
| 2048x2048 | float32 | divide        | sse42  | 0.295609 | 0.030705   | 9.63x   | -89.61%        |
| 2048x2048 | float32 | divide        | avx2   | 0.295609 | 0.024346   | 12.14x  | -91.76%        |
| 2048x2048 | float32 | overlay       | scalar | 0.411212 | 0.128840   | 3.19x   | -68.67%        |
| 2048x2048 | float32 | overlay       | sse42  | 0.411212 | 0.029486   | 13.95x  | -92.83%        |
| 2048x2048 | float32 | overlay       | avx2   | 0.411212 | 0.024710   | 16.64x  | -93.99%        |
| 1280x720  | uint8   | normal        | scalar | 0.066350 | 0.020099   | 3.30x   | -69.71%        |
| 1280x720  | uint8   | normal        | sse42  | 0.066350 | 0.002721   | 24.38x  | -95.90%        |
| 1280x720  | uint8   | normal        | avx2   | 0.066350 | 0.002181   | 30.42x  | -96.71%        |
| 1280x720  | uint8   | soft_light    | scalar | 0.102975 | 0.027410   | 3.76x   | -73.38%        |
| 1280x720  | uint8   | soft_light    | sse42  | 0.102975 | 0.003345   | 30.79x  | -96.75%        |
| 1280x720  | uint8   | soft_light    | avx2   | 0.102975 | 0.002961   | 34.78x  | -97.12%        |
| 1280x720  | uint8   | lighten_only  | scalar | 0.078453 | 0.025332   | 3.10x   | -67.71%        |
| 1280x720  | uint8   | lighten_only  | sse42  | 0.078453 | 0.003028   | 25.91x  | -96.14%        |
| 1280x720  | uint8   | lighten_only  | avx2   | 0.078453 | 0.002477   | 31.68x  | -96.84%        |
| 1280x720  | uint8   | screen        | scalar | 0.078882 | 0.023579   | 3.35x   | -70.11%        |
| 1280x720  | uint8   | screen        | sse42  | 0.078882 | 0.003092   | 25.51x  | -96.08%        |
| 1280x720  | uint8   | screen        | avx2   | 0.078882 | 0.002530   | 31.17x  | -96.79%        |
| 1280x720  | uint8   | dodge         | scalar | 0.080290 | 0.023981   | 3.35x   | -70.13%        |
| 1280x720  | uint8   | dodge         | sse42  | 0.080290 | 0.003368   | 23.84x  | -95.80%        |
| 1280x720  | uint8   | dodge         | avx2   | 0.080290 | 0.002571   | 31.22x  | -96.80%        |
| 1280x720  | uint8   | addition      | scalar | 0.077500 | 0.030167   | 2.57x   | -61.07%        |
| 1280x720  | uint8   | addition      | sse42  | 0.077500 | 0.003678   | 21.07x  | -95.25%        |
| 1280x720  | uint8   | addition      | avx2   | 0.077500 | 0.003265   | 23.73x  | -95.79%        |
| 1280x720  | uint8   | darken_only   | scalar | 0.095819 | 0.026147   | 3.66x   | -72.71%        |
| 1280x720  | uint8   | darken_only   | sse42  | 0.095819 | 0.004088   | 23.44x  | -95.73%        |
| 1280x720  | uint8   | darken_only   | avx2   | 0.095819 | 0.002603   | 36.81x  | -97.28%        |
| 1280x720  | uint8   | multiply      | scalar | 0.080675 | 0.023446   | 3.44x   | -70.94%        |
| 1280x720  | uint8   | multiply      | sse42  | 0.080675 | 0.003061   | 26.36x  | -96.21%        |
| 1280x720  | uint8   | multiply      | avx2   | 0.080675 | 0.002478   | 32.56x  | -96.93%        |
| 1280x720  | uint8   | hard_light    | scalar | 0.118376 | 0.038602   | 3.07x   | -67.39%        |
| 1280x720  | uint8   | hard_light    | sse42  | 0.118376 | 0.003512   | 33.70x  | -97.03%        |
| 1280x720  | uint8   | hard_light    | avx2   | 0.118376 | 0.002571   | 46.04x  | -97.83%        |
| 1280x720  | uint8   | difference    | scalar | 0.103911 | 0.023642   | 4.40x   | -77.25%        |
| 1280x720  | uint8   | difference    | sse42  | 0.103911 | 0.003042   | 34.16x  | -97.07%        |
| 1280x720  | uint8   | difference    | avx2   | 0.103911 | 0.002590   | 40.13x  | -97.51%        |
| 1280x720  | uint8   | subtract      | scalar | 0.078207 | 0.023670   | 3.30x   | -69.73%        |
| 1280x720  | uint8   | subtract      | sse42  | 0.078207 | 0.003794   | 20.62x  | -95.15%        |
| 1280x720  | uint8   | subtract      | avx2   | 0.078207 | 0.002715   | 28.81x  | -96.53%        |
| 1280x720  | uint8   | grain_extract | scalar | 0.078775 | 0.027462   | 2.87x   | -65.14%        |
| 1280x720  | uint8   | grain_extract | sse42  | 0.078775 | 0.003073   | 25.63x  | -96.10%        |
| 1280x720  | uint8   | grain_extract | avx2   | 0.078775 | 0.002627   | 29.99x  | -96.67%        |
| 1280x720  | uint8   | grain_merge   | scalar | 0.078372 | 0.028174   | 2.78x   | -64.05%        |
| 1280x720  | uint8   | grain_merge   | sse42  | 0.078372 | 0.003171   | 24.72x  | -95.95%        |
| 1280x720  | uint8   | grain_merge   | avx2   | 0.078372 | 0.002534   | 30.93x  | -96.77%        |
| 1280x720  | uint8   | divide        | scalar | 0.080159 | 0.024350   | 3.29x   | -69.62%        |
| 1280x720  | uint8   | divide        | sse42  | 0.080159 | 0.003238   | 24.76x  | -95.96%        |
| 1280x720  | uint8   | divide        | avx2   | 0.080159 | 0.002509   | 31.95x  | -96.87%        |
| 1280x720  | uint8   | overlay       | scalar | 0.105270 | 0.037676   | 2.79x   | -64.21%        |
| 1280x720  | uint8   | overlay       | sse42  | 0.105270 | 0.003460   | 30.43x  | -96.71%        |
| 1280x720  | uint8   | overlay       | avx2   | 0.105270 | 0.002609   | 40.35x  | -97.52%        |
| 1280x720  | float32 | normal        | scalar | 0.060678 | 0.009638   | 6.30x   | -84.12%        |
| 1280x720  | float32 | normal        | sse42  | 0.060678 | 0.004801   | 12.64x  | -92.09%        |
| 1280x720  | float32 | normal        | avx2   | 0.060678 | 0.003493   | 17.37x  | -94.24%        |
| 1280x720  | float32 | soft_light    | scalar | 0.094097 | 0.012707   | 7.41x   | -86.50%        |
| 1280x720  | float32 | soft_light    | sse42  | 0.094097 | 0.005032   | 18.70x  | -94.65%        |
| 1280x720  | float32 | soft_light    | avx2   | 0.094097 | 0.004010   | 23.47x  | -95.74%        |
| 1280x720  | float32 | lighten_only  | scalar | 0.071733 | 0.014005   | 5.12x   | -80.48%        |
| 1280x720  | float32 | lighten_only  | sse42  | 0.071733 | 0.005308   | 13.51x  | -92.60%        |
| 1280x720  | float32 | lighten_only  | avx2   | 0.071733 | 0.003794   | 18.91x  | -94.71%        |
| 1280x720  | float32 | screen        | scalar | 0.072065 | 0.011662   | 6.18x   | -83.82%        |
| 1280x720  | float32 | screen        | sse42  | 0.072065 | 0.005163   | 13.96x  | -92.84%        |
| 1280x720  | float32 | screen        | avx2   | 0.072065 | 0.003816   | 18.89x  | -94.71%        |
| 1280x720  | float32 | dodge         | scalar | 0.073199 | 0.013261   | 5.52x   | -81.88%        |
| 1280x720  | float32 | dodge         | sse42  | 0.073199 | 0.005150   | 14.21x  | -92.96%        |
| 1280x720  | float32 | dodge         | avx2   | 0.073199 | 0.004040   | 18.12x  | -94.48%        |
| 1280x720  | float32 | addition      | scalar | 0.069865 | 0.022269   | 3.14x   | -68.13%        |
| 1280x720  | float32 | addition      | sse42  | 0.069865 | 0.005821   | 12.00x  | -91.67%        |
| 1280x720  | float32 | addition      | avx2   | 0.069865 | 0.004381   | 15.95x  | -93.73%        |
| 1280x720  | float32 | darken_only   | scalar | 0.072260 | 0.013966   | 5.17x   | -80.67%        |
| 1280x720  | float32 | darken_only   | sse42  | 0.072260 | 0.005403   | 13.37x  | -92.52%        |
| 1280x720  | float32 | darken_only   | avx2   | 0.072260 | 0.003931   | 18.38x  | -94.56%        |
| 1280x720  | float32 | multiply      | scalar | 0.081642 | 0.012053   | 6.77x   | -85.24%        |
| 1280x720  | float32 | multiply      | sse42  | 0.081642 | 0.005625   | 14.51x  | -93.11%        |
| 1280x720  | float32 | multiply      | avx2   | 0.081642 | 0.003896   | 20.95x  | -95.23%        |
| 1280x720  | float32 | hard_light    | scalar | 0.105895 | 0.030086   | 3.52x   | -71.59%        |
| 1280x720  | float32 | hard_light    | sse42  | 0.105895 | 0.005051   | 20.97x  | -95.23%        |
| 1280x720  | float32 | hard_light    | avx2   | 0.105895 | 0.004053   | 26.13x  | -96.17%        |
| 1280x720  | float32 | difference    | scalar | 0.098004 | 0.011505   | 8.52x   | -88.26%        |
| 1280x720  | float32 | difference    | sse42  | 0.098004 | 0.005252   | 18.66x  | -94.64%        |
| 1280x720  | float32 | difference    | avx2   | 0.098004 | 0.003922   | 24.99x  | -96.00%        |
| 1280x720  | float32 | subtract      | scalar | 0.068328 | 0.014830   | 4.61x   | -78.30%        |
| 1280x720  | float32 | subtract      | sse42  | 0.068328 | 0.005456   | 12.52x  | -92.01%        |
| 1280x720  | float32 | subtract      | avx2   | 0.068328 | 0.004099   | 16.67x  | -94.00%        |
| 1280x720  | float32 | grain_extract | scalar | 0.072576 | 0.017413   | 4.17x   | -76.01%        |
| 1280x720  | float32 | grain_extract | sse42  | 0.072576 | 0.005188   | 13.99x  | -92.85%        |
| 1280x720  | float32 | grain_extract | avx2   | 0.072576 | 0.004413   | 16.45x  | -93.92%        |
| 1280x720  | float32 | grain_merge   | scalar | 0.075691 | 0.017849   | 4.24x   | -76.42%        |
| 1280x720  | float32 | grain_merge   | sse42  | 0.075691 | 0.005192   | 14.58x  | -93.14%        |
| 1280x720  | float32 | grain_merge   | avx2   | 0.075691 | 0.003993   | 18.96x  | -94.73%        |
| 1280x720  | float32 | divide        | scalar | 0.076981 | 0.012554   | 6.13x   | -83.69%        |
| 1280x720  | float32 | divide        | sse42  | 0.076981 | 0.005896   | 13.06x  | -92.34%        |
| 1280x720  | float32 | divide        | avx2   | 0.076981 | 0.004721   | 16.31x  | -93.87%        |
| 1280x720  | float32 | overlay       | scalar | 0.107789 | 0.027940   | 3.86x   | -74.08%        |
| 1280x720  | float32 | overlay       | sse42  | 0.107789 | 0.005649   | 19.08x  | -94.76%        |
| 1280x720  | float32 | overlay       | avx2   | 0.107789 | 0.003741   | 28.81x  | -96.53%        |
| 1920x1080 | uint8   | normal        | scalar | 0.161188 | 0.060482   | 2.67x   | -62.48%        |
| 1920x1080 | uint8   | normal        | sse42  | 0.161188 | 0.006368   | 25.31x  | -96.05%        |
| 1920x1080 | uint8   | normal        | avx2   | 0.161188 | 0.005249   | 30.71x  | -96.74%        |
| 1920x1080 | uint8   | soft_light    | scalar | 0.229531 | 0.062819   | 3.65x   | -72.63%        |
| 1920x1080 | uint8   | soft_light    | sse42  | 0.229531 | 0.007958   | 28.84x  | -96.53%        |
| 1920x1080 | uint8   | soft_light    | avx2   | 0.229531 | 0.006506   | 35.28x  | -97.17%        |
| 1920x1080 | uint8   | lighten_only  | scalar | 0.159923 | 0.058122   | 2.75x   | -63.66%        |
| 1920x1080 | uint8   | lighten_only  | sse42  | 0.159923 | 0.007438   | 21.50x  | -95.35%        |
| 1920x1080 | uint8   | lighten_only  | avx2   | 0.159923 | 0.005704   | 28.04x  | -96.43%        |
| 1920x1080 | uint8   | screen        | scalar | 0.156821 | 0.053937   | 2.91x   | -65.61%        |
| 1920x1080 | uint8   | screen        | sse42  | 0.156821 | 0.006970   | 22.50x  | -95.56%        |
| 1920x1080 | uint8   | screen        | avx2   | 0.156821 | 0.005698   | 27.52x  | -96.37%        |
| 1920x1080 | uint8   | dodge         | scalar | 0.161172 | 0.059407   | 2.71x   | -63.14%        |
| 1920x1080 | uint8   | dodge         | sse42  | 0.161172 | 0.008573   | 18.80x  | -94.68%        |
| 1920x1080 | uint8   | dodge         | avx2   | 0.161172 | 0.006041   | 26.68x  | -96.25%        |
| 1920x1080 | uint8   | addition      | scalar | 0.150270 | 0.066524   | 2.26x   | -55.73%        |
| 1920x1080 | uint8   | addition      | sse42  | 0.150270 | 0.008706   | 17.26x  | -94.21%        |
| 1920x1080 | uint8   | addition      | avx2   | 0.150270 | 0.006306   | 23.83x  | -95.80%        |
| 1920x1080 | uint8   | darken_only   | scalar | 0.169948 | 0.063625   | 2.67x   | -62.56%        |
| 1920x1080 | uint8   | darken_only   | sse42  | 0.169948 | 0.007997   | 21.25x  | -95.29%        |
| 1920x1080 | uint8   | darken_only   | avx2   | 0.169948 | 0.005759   | 29.51x  | -96.61%        |
| 1920x1080 | uint8   | multiply      | scalar | 0.190092 | 0.062848   | 3.02x   | -66.94%        |
| 1920x1080 | uint8   | multiply      | sse42  | 0.190092 | 0.007544   | 25.20x  | -96.03%        |
| 1920x1080 | uint8   | multiply      | avx2   | 0.190092 | 0.005671   | 33.52x  | -97.02%        |
| 1920x1080 | uint8   | hard_light    | scalar | 0.227223 | 0.088541   | 2.57x   | -61.03%        |
| 1920x1080 | uint8   | hard_light    | sse42  | 0.227223 | 0.008271   | 27.47x  | -96.36%        |
| 1920x1080 | uint8   | hard_light    | avx2   | 0.227223 | 0.005983   | 37.98x  | -97.37%        |
| 1920x1080 | uint8   | difference    | scalar | 0.217780 | 0.051313   | 4.24x   | -76.44%        |
| 1920x1080 | uint8   | difference    | sse42  | 0.217780 | 0.007262   | 29.99x  | -96.67%        |
| 1920x1080 | uint8   | difference    | avx2   | 0.217780 | 0.006218   | 35.02x  | -97.14%        |
| 1920x1080 | uint8   | subtract      | scalar | 0.160098 | 0.057240   | 2.80x   | -64.25%        |
| 1920x1080 | uint8   | subtract      | sse42  | 0.160098 | 0.008610   | 18.59x  | -94.62%        |
| 1920x1080 | uint8   | subtract      | avx2   | 0.160098 | 0.006192   | 25.85x  | -96.13%        |
| 1920x1080 | uint8   | grain_extract | scalar | 0.160433 | 0.064920   | 2.47x   | -59.53%        |
| 1920x1080 | uint8   | grain_extract | sse42  | 0.160433 | 0.008105   | 19.80x  | -94.95%        |
| 1920x1080 | uint8   | grain_extract | avx2   | 0.160433 | 0.005833   | 27.51x  | -96.36%        |
| 1920x1080 | uint8   | grain_merge   | scalar | 0.184266 | 0.068007   | 2.71x   | -63.09%        |
| 1920x1080 | uint8   | grain_merge   | sse42  | 0.184266 | 0.007331   | 25.14x  | -96.02%        |
| 1920x1080 | uint8   | grain_merge   | avx2   | 0.184266 | 0.005879   | 31.34x  | -96.81%        |
| 1920x1080 | uint8   | divide        | scalar | 0.164370 | 0.058444   | 2.81x   | -64.44%        |
| 1920x1080 | uint8   | divide        | sse42  | 0.164370 | 0.007345   | 22.38x  | -95.53%        |
| 1920x1080 | uint8   | divide        | avx2   | 0.164370 | 0.005872   | 27.99x  | -96.43%        |
| 1920x1080 | uint8   | overlay       | scalar | 0.226769 | 0.088325   | 2.57x   | -61.05%        |
| 1920x1080 | uint8   | overlay       | sse42  | 0.226769 | 0.007925   | 28.61x  | -96.51%        |
| 1920x1080 | uint8   | overlay       | avx2   | 0.226769 | 0.006868   | 33.02x  | -96.97%        |
| 1920x1080 | float32 | normal        | scalar | 0.134052 | 0.020049   | 6.69x   | -85.04%        |
| 1920x1080 | float32 | normal        | sse42  | 0.134052 | 0.009873   | 13.58x  | -92.63%        |
| 1920x1080 | float32 | normal        | avx2   | 0.134052 | 0.007863   | 17.05x  | -94.13%        |
| 1920x1080 | float32 | soft_light    | scalar | 0.231037 | 0.027501   | 8.40x   | -88.10%        |
| 1920x1080 | float32 | soft_light    | sse42  | 0.231037 | 0.014688   | 15.73x  | -93.64%        |
| 1920x1080 | float32 | soft_light    | avx2   | 0.231037 | 0.010737   | 21.52x  | -95.35%        |
| 1920x1080 | float32 | lighten_only  | scalar | 0.148098 | 0.041286   | 3.59x   | -72.12%        |
| 1920x1080 | float32 | lighten_only  | sse42  | 0.148098 | 0.013918   | 10.64x  | -90.60%        |
| 1920x1080 | float32 | lighten_only  | avx2   | 0.148098 | 0.012392   | 11.95x  | -91.63%        |
| 1920x1080 | float32 | screen        | scalar | 0.160237 | 0.024556   | 6.53x   | -84.67%        |
| 1920x1080 | float32 | screen        | sse42  | 0.160237 | 0.012009   | 13.34x  | -92.51%        |
| 1920x1080 | float32 | screen        | avx2   | 0.160237 | 0.013576   | 11.80x  | -91.53%        |
| 1920x1080 | float32 | dodge         | scalar | 0.163165 | 0.028287   | 5.77x   | -82.66%        |
| 1920x1080 | float32 | dodge         | sse42  | 0.163165 | 0.012211   | 13.36x  | -92.52%        |
| 1920x1080 | float32 | dodge         | avx2   | 0.163165 | 0.013318   | 12.25x  | -91.84%        |
| 1920x1080 | float32 | addition      | scalar | 0.163528 | 0.047391   | 3.45x   | -71.02%        |
| 1920x1080 | float32 | addition      | sse42  | 0.163528 | 0.013410   | 12.19x  | -91.80%        |
| 1920x1080 | float32 | addition      | avx2   | 0.163528 | 0.010119   | 16.16x  | -93.81%        |
| 1920x1080 | float32 | darken_only   | scalar | 0.156395 | 0.034559   | 4.53x   | -77.90%        |
| 1920x1080 | float32 | darken_only   | sse42  | 0.156395 | 0.012419   | 12.59x  | -92.06%        |
| 1920x1080 | float32 | darken_only   | avx2   | 0.156395 | 0.009352   | 16.72x  | -94.02%        |
| 1920x1080 | float32 | multiply      | scalar | 0.165886 | 0.026393   | 6.29x   | -84.09%        |
| 1920x1080 | float32 | multiply      | sse42  | 0.165886 | 0.012204   | 13.59x  | -92.64%        |
| 1920x1080 | float32 | multiply      | avx2   | 0.165886 | 0.009484   | 17.49x  | -94.28%        |
| 1920x1080 | float32 | hard_light    | scalar | 0.242777 | 0.063707   | 3.81x   | -73.76%        |
| 1920x1080 | float32 | hard_light    | sse42  | 0.242777 | 0.011504   | 21.10x  | -95.26%        |
| 1920x1080 | float32 | hard_light    | avx2   | 0.242777 | 0.009048   | 26.83x  | -96.27%        |
| 1920x1080 | float32 | difference    | scalar | 0.209845 | 0.026036   | 8.06x   | -87.59%        |
| 1920x1080 | float32 | difference    | sse42  | 0.209845 | 0.011711   | 17.92x  | -94.42%        |
| 1920x1080 | float32 | difference    | avx2   | 0.209845 | 0.008776   | 23.91x  | -95.82%        |
| 1920x1080 | float32 | subtract      | scalar | 0.157406 | 0.031616   | 4.98x   | -79.91%        |
| 1920x1080 | float32 | subtract      | sse42  | 0.157406 | 0.011874   | 13.26x  | -92.46%        |
| 1920x1080 | float32 | subtract      | avx2   | 0.157406 | 0.009176   | 17.15x  | -94.17%        |
| 1920x1080 | float32 | grain_extract | scalar | 0.156077 | 0.038961   | 4.01x   | -75.04%        |
| 1920x1080 | float32 | grain_extract | sse42  | 0.156077 | 0.011879   | 13.14x  | -92.39%        |
| 1920x1080 | float32 | grain_extract | avx2   | 0.156077 | 0.008814   | 17.71x  | -94.35%        |
| 1920x1080 | float32 | grain_merge   | scalar | 0.172679 | 0.038438   | 4.49x   | -77.74%        |
| 1920x1080 | float32 | grain_merge   | sse42  | 0.172679 | 0.011841   | 14.58x  | -93.14%        |
| 1920x1080 | float32 | grain_merge   | avx2   | 0.172679 | 0.008787   | 19.65x  | -94.91%        |
| 1920x1080 | float32 | divide        | scalar | 0.158866 | 0.026935   | 5.90x   | -83.05%        |
| 1920x1080 | float32 | divide        | sse42  | 0.158866 | 0.011575   | 13.73x  | -92.71%        |
| 1920x1080 | float32 | divide        | avx2   | 0.158866 | 0.009416   | 16.87x  | -94.07%        |
| 1920x1080 | float32 | overlay       | scalar | 0.207560 | 0.059324   | 3.50x   | -71.42%        |
| 1920x1080 | float32 | overlay       | sse42  | 0.207560 | 0.011488   | 18.07x  | -94.47%        |
| 1920x1080 | float32 | overlay       | avx2   | 0.207560 | 0.008589   | 24.17x  | -95.86%        |
| 2560x1440 | uint8   | normal        | scalar | 0.265932 | 0.080719   | 3.29x   | -69.65%        |
| 2560x1440 | uint8   | normal        | sse42  | 0.265932 | 0.012394   | 21.46x  | -95.34%        |
| 2560x1440 | uint8   | normal        | avx2   | 0.265932 | 0.009566   | 27.80x  | -96.40%        |
| 2560x1440 | uint8   | soft_light    | scalar | 0.379253 | 0.103177   | 3.68x   | -72.79%        |
| 2560x1440 | uint8   | soft_light    | sse42  | 0.379253 | 0.013302   | 28.51x  | -96.49%        |
| 2560x1440 | uint8   | soft_light    | avx2   | 0.379253 | 0.010464   | 36.24x  | -97.24%        |
| 2560x1440 | uint8   | lighten_only  | scalar | 0.271217 | 0.114079   | 2.38x   | -57.94%        |
| 2560x1440 | uint8   | lighten_only  | sse42  | 0.271217 | 0.012464   | 21.76x  | -95.40%        |
| 2560x1440 | uint8   | lighten_only  | avx2   | 0.271217 | 0.010450   | 25.95x  | -96.15%        |
| 2560x1440 | uint8   | screen        | scalar | 0.303026 | 0.096855   | 3.13x   | -68.04%        |
| 2560x1440 | uint8   | screen        | sse42  | 0.303026 | 0.013128   | 23.08x  | -95.67%        |
| 2560x1440 | uint8   | screen        | avx2   | 0.303026 | 0.010583   | 28.63x  | -96.51%        |
| 2560x1440 | uint8   | dodge         | scalar | 0.281992 | 0.096551   | 2.92x   | -65.76%        |
| 2560x1440 | uint8   | dodge         | sse42  | 0.281992 | 0.013752   | 20.51x  | -95.12%        |
| 2560x1440 | uint8   | dodge         | avx2   | 0.281992 | 0.010326   | 27.31x  | -96.34%        |
| 2560x1440 | uint8   | addition      | scalar | 0.264653 | 0.124688   | 2.12x   | -52.89%        |
| 2560x1440 | uint8   | addition      | sse42  | 0.264653 | 0.014718   | 17.98x  | -94.44%        |
| 2560x1440 | uint8   | addition      | avx2   | 0.264653 | 0.011044   | 23.96x  | -95.83%        |
| 2560x1440 | uint8   | darken_only   | scalar | 0.258568 | 0.101383   | 2.55x   | -60.79%        |
| 2560x1440 | uint8   | darken_only   | sse42  | 0.258568 | 0.012297   | 21.03x  | -95.24%        |
| 2560x1440 | uint8   | darken_only   | avx2   | 0.258568 | 0.011922   | 21.69x  | -95.39%        |
| 2560x1440 | uint8   | multiply      | scalar | 0.267405 | 0.100558   | 2.66x   | -62.39%        |
| 2560x1440 | uint8   | multiply      | sse42  | 0.267405 | 0.012606   | 21.21x  | -95.29%        |
| 2560x1440 | uint8   | multiply      | avx2   | 0.267405 | 0.010206   | 26.20x  | -96.18%        |
| 2560x1440 | uint8   | hard_light    | scalar | 0.424500 | 0.160134   | 2.65x   | -62.28%        |
| 2560x1440 | uint8   | hard_light    | sse42  | 0.424500 | 0.014580   | 29.12x  | -96.57%        |
| 2560x1440 | uint8   | hard_light    | avx2   | 0.424500 | 0.010630   | 39.93x  | -97.50%        |
| 2560x1440 | uint8   | difference    | scalar | 0.380163 | 0.097460   | 3.90x   | -74.36%        |
| 2560x1440 | uint8   | difference    | sse42  | 0.380163 | 0.013200   | 28.80x  | -96.53%        |
| 2560x1440 | uint8   | difference    | avx2   | 0.380163 | 0.010305   | 36.89x  | -97.29%        |
| 2560x1440 | uint8   | subtract      | scalar | 0.266982 | 0.096417   | 2.77x   | -63.89%        |
| 2560x1440 | uint8   | subtract      | sse42  | 0.266982 | 0.014981   | 17.82x  | -94.39%        |
| 2560x1440 | uint8   | subtract      | avx2   | 0.266982 | 0.011043   | 24.18x  | -95.86%        |
| 2560x1440 | uint8   | grain_extract | scalar | 0.277128 | 0.114686   | 2.42x   | -58.62%        |
| 2560x1440 | uint8   | grain_extract | sse42  | 0.277128 | 0.012919   | 21.45x  | -95.34%        |
| 2560x1440 | uint8   | grain_extract | avx2   | 0.277128 | 0.010154   | 27.29x  | -96.34%        |
| 2560x1440 | uint8   | grain_merge   | scalar | 0.273721 | 0.112658   | 2.43x   | -58.84%        |
| 2560x1440 | uint8   | grain_merge   | sse42  | 0.273721 | 0.012937   | 21.16x  | -95.27%        |
| 2560x1440 | uint8   | grain_merge   | avx2   | 0.273721 | 0.010231   | 26.75x  | -96.26%        |
| 2560x1440 | uint8   | divide        | scalar | 0.279315 | 0.101609   | 2.75x   | -63.62%        |
| 2560x1440 | uint8   | divide        | sse42  | 0.279315 | 0.013008   | 21.47x  | -95.34%        |
| 2560x1440 | uint8   | divide        | avx2   | 0.279315 | 0.010344   | 27.00x  | -96.30%        |
| 2560x1440 | uint8   | overlay       | scalar | 0.383161 | 0.150714   | 2.54x   | -60.67%        |
| 2560x1440 | uint8   | overlay       | sse42  | 0.383161 | 0.013988   | 27.39x  | -96.35%        |
| 2560x1440 | uint8   | overlay       | avx2   | 0.383161 | 0.011029   | 34.74x  | -97.12%        |
| 2560x1440 | float32 | normal        | scalar | 0.242976 | 0.042654   | 5.70x   | -82.45%        |
| 2560x1440 | float32 | normal        | sse42  | 0.242976 | 0.024509   | 9.91x   | -89.91%        |
| 2560x1440 | float32 | normal        | avx2   | 0.242976 | 0.031660   | 7.67x   | -86.97%        |
| 2560x1440 | float32 | soft_light    | scalar | 0.348741 | 0.047206   | 7.39x   | -86.46%        |
| 2560x1440 | float32 | soft_light    | sse42  | 0.348741 | 0.019677   | 17.72x  | -94.36%        |
| 2560x1440 | float32 | soft_light    | avx2   | 0.348741 | 0.015297   | 22.80x  | -95.61%        |
| 2560x1440 | float32 | lighten_only  | scalar | 0.224382 | 0.056826   | 3.95x   | -74.67%        |
| 2560x1440 | float32 | lighten_only  | sse42  | 0.224382 | 0.027458   | 8.17x   | -87.76%        |
| 2560x1440 | float32 | lighten_only  | avx2   | 0.224382 | 0.022049   | 10.18x  | -90.17%        |
| 2560x1440 | float32 | screen        | scalar | 0.259347 | 0.044097   | 5.88x   | -83.00%        |
| 2560x1440 | float32 | screen        | sse42  | 0.259347 | 0.020409   | 12.71x  | -92.13%        |
| 2560x1440 | float32 | screen        | avx2   | 0.259347 | 0.015495   | 16.74x  | -94.03%        |
| 2560x1440 | float32 | dodge         | scalar | 0.266339 | 0.055839   | 4.77x   | -79.03%        |
| 2560x1440 | float32 | dodge         | sse42  | 0.266339 | 0.026491   | 10.05x  | -90.05%        |
| 2560x1440 | float32 | dodge         | avx2   | 0.266339 | 0.021926   | 12.15x  | -91.77%        |
| 2560x1440 | float32 | addition      | scalar | 0.241495 | 0.081082   | 2.98x   | -66.42%        |
| 2560x1440 | float32 | addition      | sse42  | 0.241495 | 0.020930   | 11.54x  | -91.33%        |
| 2560x1440 | float32 | addition      | avx2   | 0.241495 | 0.015246   | 15.84x  | -93.69%        |
| 2560x1440 | float32 | darken_only   | scalar | 0.229739 | 0.057741   | 3.98x   | -74.87%        |
| 2560x1440 | float32 | darken_only   | sse42  | 0.229739 | 0.027497   | 8.36x   | -88.03%        |
| 2560x1440 | float32 | darken_only   | avx2   | 0.229739 | 0.022683   | 10.13x  | -90.13%        |
| 2560x1440 | float32 | multiply      | scalar | 0.245672 | 0.042329   | 5.80x   | -82.77%        |
| 2560x1440 | float32 | multiply      | sse42  | 0.245672 | 0.021223   | 11.58x  | -91.36%        |
| 2560x1440 | float32 | multiply      | avx2   | 0.245672 | 0.015800   | 15.55x  | -93.57%        |
| 2560x1440 | float32 | hard_light    | scalar | 0.395661 | 0.117533   | 3.37x   | -70.29%        |
| 2560x1440 | float32 | hard_light    | sse42  | 0.395661 | 0.026765   | 14.78x  | -93.24%        |
| 2560x1440 | float32 | hard_light    | avx2   | 0.395661 | 0.023149   | 17.09x  | -94.15%        |
| 2560x1440 | float32 | difference    | scalar | 0.335597 | 0.042384   | 7.92x   | -87.37%        |
| 2560x1440 | float32 | difference    | sse42  | 0.335597 | 0.020517   | 16.36x  | -93.89%        |
| 2560x1440 | float32 | difference    | avx2   | 0.335597 | 0.015220   | 22.05x  | -95.46%        |
| 2560x1440 | float32 | subtract      | scalar | 0.250278 | 0.063070   | 3.97x   | -74.80%        |
| 2560x1440 | float32 | subtract      | sse42  | 0.250278 | 0.027692   | 9.04x   | -88.94%        |
| 2560x1440 | float32 | subtract      | avx2   | 0.250278 | 0.021809   | 11.48x  | -91.29%        |
| 2560x1440 | float32 | grain_extract | scalar | 0.250035 | 0.066309   | 3.77x   | -73.48%        |
| 2560x1440 | float32 | grain_extract | sse42  | 0.250035 | 0.020094   | 12.44x  | -91.96%        |
| 2560x1440 | float32 | grain_extract | avx2   | 0.250035 | 0.015121   | 16.54x  | -93.95%        |
| 2560x1440 | float32 | grain_merge   | scalar | 0.244152 | 0.070283   | 3.47x   | -71.21%        |
| 2560x1440 | float32 | grain_merge   | sse42  | 0.244152 | 0.026123   | 9.35x   | -89.30%        |
| 2560x1440 | float32 | grain_merge   | avx2   | 0.244152 | 0.021851   | 11.17x  | -91.05%        |
| 2560x1440 | float32 | divide        | scalar | 0.273558 | 0.047013   | 5.82x   | -82.81%        |
| 2560x1440 | float32 | divide        | sse42  | 0.273558 | 0.020344   | 13.45x  | -92.56%        |
| 2560x1440 | float32 | divide        | avx2   | 0.273558 | 0.015397   | 17.77x  | -94.37%        |
| 2560x1440 | float32 | overlay       | scalar | 0.386841 | 0.120275   | 3.22x   | -68.91%        |
| 2560x1440 | float32 | overlay       | sse42  | 0.386841 | 0.027157   | 14.24x  | -92.98%        |
| 2560x1440 | float32 | overlay       | avx2   | 0.386841 | 0.022479   | 17.21x  | -94.19%        |
| 3840x2160 | uint8   | normal        | scalar | 0.624598 | 0.174012   | 3.59x   | -72.14%        |
| 3840x2160 | uint8   | normal        | sse42  | 0.624598 | 0.024307   | 25.70x  | -96.11%        |
| 3840x2160 | uint8   | normal        | avx2   | 0.624598 | 0.019443   | 32.12x  | -96.89%        |
| 3840x2160 | uint8   | soft_light    | scalar | 0.758438 | 0.227087   | 3.34x   | -70.06%        |
| 3840x2160 | uint8   | soft_light    | sse42  | 0.758438 | 0.030272   | 25.05x  | -96.01%        |
| 3840x2160 | uint8   | soft_light    | avx2   | 0.758438 | 0.023499   | 32.27x  | -96.90%        |
| 3840x2160 | uint8   | lighten_only  | scalar | 0.548886 | 0.239921   | 2.29x   | -56.29%        |
| 3840x2160 | uint8   | lighten_only  | sse42  | 0.548886 | 0.027664   | 19.84x  | -94.96%        |
| 3840x2160 | uint8   | lighten_only  | avx2   | 0.548886 | 0.022866   | 24.00x  | -95.83%        |
| 3840x2160 | uint8   | screen        | scalar | 0.607144 | 0.228416   | 2.66x   | -62.38%        |
| 3840x2160 | uint8   | screen        | sse42  | 0.607144 | 0.028325   | 21.44x  | -95.33%        |
| 3840x2160 | uint8   | screen        | avx2   | 0.607144 | 0.024627   | 24.65x  | -95.94%        |
| 3840x2160 | uint8   | dodge         | scalar | 0.590450 | 0.216372   | 2.73x   | -63.35%        |
| 3840x2160 | uint8   | dodge         | sse42  | 0.590450 | 0.030675   | 19.25x  | -94.80%        |
| 3840x2160 | uint8   | dodge         | avx2   | 0.590450 | 0.022981   | 25.69x  | -96.11%        |
| 3840x2160 | uint8   | addition      | scalar | 0.560772 | 0.268445   | 2.09x   | -52.13%        |
| 3840x2160 | uint8   | addition      | sse42  | 0.560772 | 0.033133   | 16.92x  | -94.09%        |
| 3840x2160 | uint8   | addition      | avx2   | 0.560772 | 0.024445   | 22.94x  | -95.64%        |
| 3840x2160 | uint8   | darken_only   | scalar | 0.539852 | 0.235298   | 2.29x   | -56.41%        |
| 3840x2160 | uint8   | darken_only   | sse42  | 0.539852 | 0.027773   | 19.44x  | -94.86%        |
| 3840x2160 | uint8   | darken_only   | avx2   | 0.539852 | 0.023616   | 22.86x  | -95.63%        |
| 3840x2160 | uint8   | multiply      | scalar | 0.560248 | 0.223572   | 2.51x   | -60.09%        |
| 3840x2160 | uint8   | multiply      | sse42  | 0.560248 | 0.028669   | 19.54x  | -94.88%        |
| 3840x2160 | uint8   | multiply      | avx2   | 0.560248 | 0.022535   | 24.86x  | -95.98%        |
| 3840x2160 | uint8   | hard_light    | scalar | 0.883938 | 0.352272   | 2.51x   | -60.15%        |
| 3840x2160 | uint8   | hard_light    | sse42  | 0.883938 | 0.031981   | 27.64x  | -96.38%        |
| 3840x2160 | uint8   | hard_light    | avx2   | 0.883938 | 0.023252   | 38.01x  | -97.37%        |
| 3840x2160 | uint8   | difference    | scalar | 0.775390 | 0.212386   | 3.65x   | -72.61%        |
| 3840x2160 | uint8   | difference    | sse42  | 0.775390 | 0.027530   | 28.17x  | -96.45%        |
| 3840x2160 | uint8   | difference    | avx2   | 0.775390 | 0.022357   | 34.68x  | -97.12%        |
| 3840x2160 | uint8   | subtract      | scalar | 0.608540 | 0.212400   | 2.87x   | -65.10%        |
| 3840x2160 | uint8   | subtract      | sse42  | 0.608540 | 0.033513   | 18.16x  | -94.49%        |
| 3840x2160 | uint8   | subtract      | avx2   | 0.608540 | 0.025121   | 24.22x  | -95.87%        |
| 3840x2160 | uint8   | grain_extract | scalar | 0.633490 | 0.260427   | 2.43x   | -58.89%        |
| 3840x2160 | uint8   | grain_extract | sse42  | 0.633490 | 0.028799   | 22.00x  | -95.45%        |
| 3840x2160 | uint8   | grain_extract | avx2   | 0.633490 | 0.023393   | 27.08x  | -96.31%        |
| 3840x2160 | uint8   | grain_merge   | scalar | 0.574301 | 0.262623   | 2.19x   | -54.27%        |
| 3840x2160 | uint8   | grain_merge   | sse42  | 0.574301 | 0.029174   | 19.69x  | -94.92%        |
| 3840x2160 | uint8   | grain_merge   | avx2   | 0.574301 | 0.022936   | 25.04x  | -96.01%        |
| 3840x2160 | uint8   | divide        | scalar | 0.617020 | 0.217455   | 2.84x   | -64.76%        |
| 3840x2160 | uint8   | divide        | sse42  | 0.617020 | 0.029445   | 20.95x  | -95.23%        |
| 3840x2160 | uint8   | divide        | avx2   | 0.617020 | 0.022864   | 26.99x  | -96.29%        |
| 3840x2160 | uint8   | overlay       | scalar | 0.792074 | 0.337116   | 2.35x   | -57.44%        |
| 3840x2160 | uint8   | overlay       | sse42  | 0.792074 | 0.031249   | 25.35x  | -96.05%        |
| 3840x2160 | uint8   | overlay       | avx2   | 0.792074 | 0.023317   | 33.97x  | -97.06%        |
| 3840x2160 | float32 | normal        | scalar | 0.503031 | 0.092346   | 5.45x   | -81.64%        |
| 3840x2160 | float32 | normal        | sse42  | 0.503031 | 0.053625   | 9.38x   | -89.34%        |
| 3840x2160 | float32 | normal        | avx2   | 0.503031 | 0.042022   | 11.97x  | -91.65%        |
| 3840x2160 | float32 | soft_light    | scalar | 0.734614 | 0.125389   | 5.86x   | -82.93%        |
| 3840x2160 | float32 | soft_light    | sse42  | 0.734614 | 0.055506   | 13.23x  | -92.44%        |
| 3840x2160 | float32 | soft_light    | avx2   | 0.734614 | 0.049644   | 14.80x  | -93.24%        |
| 3840x2160 | float32 | lighten_only  | scalar | 0.520521 | 0.123488   | 4.22x   | -76.28%        |
| 3840x2160 | float32 | lighten_only  | sse42  | 0.520521 | 0.058005   | 8.97x   | -88.86%        |
| 3840x2160 | float32 | lighten_only  | avx2   | 0.520521 | 0.045751   | 11.38x  | -91.21%        |
| 3840x2160 | float32 | screen        | scalar | 0.560386 | 0.112239   | 4.99x   | -79.97%        |
| 3840x2160 | float32 | screen        | sse42  | 0.560386 | 0.056860   | 9.86x   | -89.85%        |
| 3840x2160 | float32 | screen        | avx2   | 0.560386 | 0.045800   | 12.24x  | -91.83%        |
| 3840x2160 | float32 | dodge         | scalar | 0.529161 | 0.115292   | 4.59x   | -78.21%        |
| 3840x2160 | float32 | dodge         | sse42  | 0.529161 | 0.058070   | 9.11x   | -89.03%        |
| 3840x2160 | float32 | dodge         | avx2   | 0.529161 | 0.046740   | 11.32x  | -91.17%        |
| 3840x2160 | float32 | addition      | scalar | 0.506134 | 0.192338   | 2.63x   | -62.00%        |
| 3840x2160 | float32 | addition      | sse42  | 0.506134 | 0.058634   | 8.63x   | -88.42%        |
| 3840x2160 | float32 | addition      | avx2   | 0.506134 | 0.044469   | 11.38x  | -91.21%        |
| 3840x2160 | float32 | darken_only   | scalar | 0.498988 | 0.124485   | 4.01x   | -75.05%        |
| 3840x2160 | float32 | darken_only   | sse42  | 0.498988 | 0.057556   | 8.67x   | -88.47%        |
| 3840x2160 | float32 | darken_only   | avx2   | 0.498988 | 0.044188   | 11.29x  | -91.14%        |
| 3840x2160 | float32 | multiply      | scalar | 0.533135 | 0.105194   | 5.07x   | -80.27%        |
| 3840x2160 | float32 | multiply      | sse42  | 0.533135 | 0.057169   | 9.33x   | -89.28%        |
| 3840x2160 | float32 | multiply      | avx2   | 0.533135 | 0.045036   | 11.84x  | -91.55%        |
| 3840x2160 | float32 | hard_light    | scalar | 0.830769 | 0.258063   | 3.22x   | -68.94%        |
| 3840x2160 | float32 | hard_light    | sse42  | 0.830769 | 0.056473   | 14.71x  | -93.20%        |
| 3840x2160 | float32 | hard_light    | avx2   | 0.830769 | 0.044755   | 18.56x  | -94.61%        |
| 3840x2160 | float32 | difference    | scalar | 0.721473 | 0.102653   | 7.03x   | -85.77%        |
| 3840x2160 | float32 | difference    | sse42  | 0.721473 | 0.055877   | 12.91x  | -92.26%        |
| 3840x2160 | float32 | difference    | avx2   | 0.721473 | 0.043626   | 16.54x  | -93.95%        |
| 3840x2160 | float32 | subtract      | scalar | 0.502930 | 0.131374   | 3.83x   | -73.88%        |
| 3840x2160 | float32 | subtract      | sse42  | 0.502930 | 0.062528   | 8.04x   | -87.57%        |
| 3840x2160 | float32 | subtract      | avx2   | 0.502930 | 0.050782   | 9.90x   | -89.90%        |
| 3840x2160 | float32 | grain_extract | scalar | 0.570269 | 0.154298   | 3.70x   | -72.94%        |
| 3840x2160 | float32 | grain_extract | sse42  | 0.570269 | 0.057514   | 9.92x   | -89.91%        |
| 3840x2160 | float32 | grain_extract | avx2   | 0.570269 | 0.045558   | 12.52x  | -92.01%        |
| 3840x2160 | float32 | grain_merge   | scalar | 0.530104 | 0.162362   | 3.26x   | -69.37%        |
| 3840x2160 | float32 | grain_merge   | sse42  | 0.530104 | 0.060696   | 8.73x   | -88.55%        |
| 3840x2160 | float32 | grain_merge   | avx2   | 0.530104 | 0.049066   | 10.80x  | -90.74%        |
| 3840x2160 | float32 | divide        | scalar | 0.636247 | 0.128456   | 4.95x   | -79.81%        |
| 3840x2160 | float32 | divide        | sse42  | 0.636247 | 0.065707   | 9.68x   | -89.67%        |
| 3840x2160 | float32 | divide        | avx2   | 0.636247 | 0.054188   | 11.74x  | -91.48%        |
| 3840x2160 | float32 | overlay       | scalar | 0.890683 | 0.256035   | 3.48x   | -71.25%        |
| 3840x2160 | float32 | overlay       | sse42  | 0.890683 | 0.060687   | 14.68x  | -93.19%        |
| 3840x2160 | float32 | overlay       | avx2   | 0.890683 | 0.046793   | 19.03x  | -94.75%        |
</details>
