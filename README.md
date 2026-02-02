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

| Case      | Mode     | Kernel | Ref (s)  | Kernel (s) | Speedup | Percent Change |
| --------- | -------- | ------ | -------- | ---------- | ------- | -------------- |
| 256x256   | normal   | scalar | 0.003182 | 0.001361   | 2.34x   | -57.22%        |
| 256x256   | normal   | sse42  | 0.003182 | 0.000193   | 16.46x  | -93.93%        |
| 256x256   | normal   | avx2   | 0.003182 | 0.000154   | 20.61x  | -95.15%        |
| 256x256   | multiply | scalar | 0.005427 | 0.001769   | 3.07x   | -67.40%        |
| 256x256   | multiply | sse42  | 0.005427 | 0.000217   | 24.96x  | -95.99%        |
| 256x256   | multiply | avx2   | 0.005427 | 0.000178   | 30.49x  | -96.72%        |
| 256x256   | screen   | scalar | 0.005896 | 0.001736   | 3.40x   | -70.56%        |
| 256x256   | screen   | sse42  | 0.005896 | 0.000226   | 26.10x  | -96.17%        |
| 256x256   | screen   | avx2   | 0.005896 | 0.000182   | 32.32x  | -96.91%        |
| 512x512   | normal   | scalar | 0.022963 | 0.005679   | 4.04x   | -75.27%        |
| 512x512   | normal   | sse42  | 0.022963 | 0.000780   | 29.43x  | -96.60%        |
| 512x512   | normal   | avx2   | 0.022963 | 0.000622   | 36.92x  | -97.29%        |
| 512x512   | multiply | scalar | 0.027643 | 0.007039   | 3.93x   | -74.54%        |
| 512x512   | multiply | sse42  | 0.027643 | 0.000906   | 30.51x  | -96.72%        |
| 512x512   | multiply | avx2   | 0.027643 | 0.000712   | 38.81x  | -97.42%        |
| 512x512   | screen   | scalar | 0.026764 | 0.006812   | 3.93x   | -74.55%        |
| 512x512   | screen   | sse42  | 0.026764 | 0.001018   | 26.30x  | -96.20%        |
| 512x512   | screen   | avx2   | 0.026764 | 0.000732   | 36.55x  | -97.26%        |
| 1024x1024 | normal   | scalar | 0.068064 | 0.022136   | 3.07x   | -67.48%        |
| 1024x1024 | normal   | sse42  | 0.068064 | 0.003171   | 21.47x  | -95.34%        |
| 1024x1024 | normal   | avx2   | 0.068064 | 0.002497   | 27.26x  | -96.33%        |
| 1024x1024 | multiply | scalar | 0.076773 | 0.026614   | 2.88x   | -65.33%        |
| 1024x1024 | multiply | sse42  | 0.076773 | 0.003729   | 20.59x  | -95.14%        |
| 1024x1024 | multiply | avx2   | 0.076773 | 0.002853   | 26.91x  | -96.28%        |
| 1024x1024 | screen   | scalar | 0.074736 | 0.027306   | 2.74x   | -63.46%        |
| 1024x1024 | screen   | sse42  | 0.074736 | 0.003733   | 20.02x  | -95.00%        |
| 1024x1024 | screen   | avx2   | 0.074736 | 0.002892   | 25.85x  | -96.13%        |
| 2048x2048 | normal   | scalar | 0.269703 | 0.086647   | 3.11x   | -67.87%        |
| 2048x2048 | normal   | sse42  | 0.269703 | 0.013298   | 20.28x  | -95.07%        |
| 2048x2048 | normal   | avx2   | 0.269703 | 0.010535   | 25.60x  | -96.09%        |
| 2048x2048 | multiply | scalar | 0.275712 | 0.110451   | 2.50x   | -59.94%        |
| 2048x2048 | multiply | sse42  | 0.275712 | 0.014939   | 18.46x  | -94.58%        |
| 2048x2048 | multiply | avx2   | 0.275712 | 0.011395   | 24.20x  | -95.87%        |
| 2048x2048 | screen   | scalar | 0.289454 | 0.111139   | 2.60x   | -61.60%        |
| 2048x2048 | screen   | sse42  | 0.289454 | 0.014811   | 19.54x  | -94.88%        |
| 2048x2048 | screen   | avx2   | 0.289454 | 0.011965   | 24.19x  | -95.87%        |
| 1280x720  | normal   | scalar | 0.066490 | 0.020533   | 3.24x   | -69.12%        |
| 1280x720  | normal   | sse42  | 0.066490 | 0.003351   | 19.84x  | -94.96%        |
| 1280x720  | normal   | avx2   | 0.066490 | 0.002277   | 29.20x  | -96.58%        |
| 1280x720  | multiply | scalar | 0.074333 | 0.023916   | 3.11x   | -67.83%        |
| 1280x720  | multiply | sse42  | 0.074333 | 0.003078   | 24.15x  | -95.86%        |
| 1280x720  | multiply | avx2   | 0.074333 | 0.002561   | 29.03x  | -96.55%        |
| 1280x720  | screen   | scalar | 0.080717 | 0.026591   | 3.04x   | -67.06%        |
| 1280x720  | screen   | sse42  | 0.080717 | 0.003205   | 25.18x  | -96.03%        |
| 1280x720  | screen   | avx2   | 0.080717 | 0.002578   | 31.31x  | -96.81%        |
| 1920x1080 | normal   | scalar | 0.140149 | 0.045124   | 3.11x   | -67.80%        |
| 1920x1080 | normal   | sse42  | 0.140149 | 0.006472   | 21.65x  | -95.38%        |
| 1920x1080 | normal   | avx2   | 0.140149 | 0.005000   | 28.03x  | -96.43%        |
| 1920x1080 | multiply | scalar | 0.151167 | 0.060418   | 2.50x   | -60.03%        |
| 1920x1080 | multiply | sse42  | 0.151167 | 0.007352   | 20.56x  | -95.14%        |
| 1920x1080 | multiply | avx2   | 0.151167 | 0.005705   | 26.50x  | -96.23%        |
| 1920x1080 | screen   | scalar | 0.142945 | 0.055582   | 2.57x   | -61.12%        |
| 1920x1080 | screen   | sse42  | 0.142945 | 0.007287   | 19.62x  | -94.90%        |
| 1920x1080 | screen   | avx2   | 0.142945 | 0.005848   | 24.45x  | -95.91%        |
| 2560x1440 | normal   | scalar | 0.235611 | 0.076912   | 3.06x   | -67.36%        |
| 2560x1440 | normal   | sse42  | 0.235611 | 0.011077   | 21.27x  | -95.30%        |
| 2560x1440 | normal   | avx2   | 0.235611 | 0.009066   | 25.99x  | -96.15%        |
| 2560x1440 | multiply | scalar | 0.255727 | 0.096881   | 2.64x   | -62.12%        |
| 2560x1440 | multiply | sse42  | 0.255727 | 0.012342   | 20.72x  | -95.17%        |
| 2560x1440 | multiply | avx2   | 0.255727 | 0.010111   | 25.29x  | -96.05%        |
| 2560x1440 | screen   | scalar | 0.251971 | 0.094904   | 2.65x   | -62.34%        |
| 2560x1440 | screen   | sse42  | 0.251971 | 0.012369   | 20.37x  | -95.09%        |
| 2560x1440 | screen   | avx2   | 0.251971 | 0.010218   | 24.66x  | -95.94%        |
| 3840x2160 | normal   | scalar | 0.524053 | 0.178688   | 2.93x   | -65.90%        |
| 3840x2160 | normal   | sse42  | 0.524053 | 0.026184   | 20.01x  | -95.00%        |
| 3840x2160 | normal   | avx2   | 0.524053 | 0.019952   | 26.27x  | -96.19%        |
| 3840x2160 | multiply | scalar | 0.568292 | 0.225099   | 2.52x   | -60.39%        |
| 3840x2160 | multiply | sse42  | 0.568292 | 0.028683   | 19.81x  | -94.95%        |
| 3840x2160 | multiply | avx2   | 0.568292 | 0.022401   | 25.37x  | -96.06%        |
| 3840x2160 | screen   | scalar | 0.558074 | 0.222274   | 2.51x   | -60.17%        |
| 3840x2160 | screen   | sse42  | 0.558074 | 0.028484   | 19.59x  | -94.90%        |
| 3840x2160 | screen   | avx2   | 0.558074 | 0.023404   | 23.85x  | -95.81%        |
