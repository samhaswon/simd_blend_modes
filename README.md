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
| 256x256   | normal   | scalar | 0.006612 | 0.001362   | 4.86x   | -79.41%        |
| 256x256   | normal   | sse42  | 0.006612 | 0.001032   | 6.41x   | -84.40%        |
| 256x256   | normal   | avx2   | 0.006612 | 0.001117   | 5.92x   | -83.11%        |
| 256x256   | multiply | scalar | 0.007282 | 0.001667   | 4.37x   | -77.11%        |
| 256x256   | multiply | sse42  | 0.007282 | 0.001059   | 6.87x   | -85.45%        |
| 256x256   | multiply | avx2   | 0.007282 | 0.001149   | 6.34x   | -84.22%        |
| 256x256   | screen   | scalar | 0.006866 | 0.001739   | 3.95x   | -74.67%        |
| 256x256   | screen   | sse42  | 0.006866 | 0.001051   | 6.53x   | -84.70%        |
| 256x256   | screen   | avx2   | 0.006866 | 0.001099   | 6.25x   | -83.99%        |
| 512x512   | normal   | scalar | 0.023103 | 0.005372   | 4.30x   | -76.75%        |
| 512x512   | normal   | sse42  | 0.023103 | 0.004139   | 5.58x   | -82.09%        |
| 512x512   | normal   | avx2   | 0.023103 | 0.004507   | 5.13x   | -80.49%        |
| 512x512   | multiply | scalar | 0.027180 | 0.006407   | 4.24x   | -76.43%        |
| 512x512   | multiply | sse42  | 0.027180 | 0.004432   | 6.13x   | -83.70%        |
| 512x512   | multiply | avx2   | 0.027180 | 0.004468   | 6.08x   | -83.56%        |
| 512x512   | screen   | scalar | 0.027278 | 0.006886   | 3.96x   | -74.75%        |
| 512x512   | screen   | sse42  | 0.027278 | 0.004169   | 6.54x   | -84.72%        |
| 512x512   | screen   | avx2   | 0.027278 | 0.004390   | 6.21x   | -83.91%        |
| 1024x1024 | normal   | scalar | 0.072408 | 0.022040   | 3.29x   | -69.56%        |
| 1024x1024 | normal   | sse42  | 0.072408 | 0.016705   | 4.33x   | -76.93%        |
| 1024x1024 | normal   | avx2   | 0.072408 | 0.017411   | 4.16x   | -75.95%        |
| 1024x1024 | multiply | scalar | 0.076406 | 0.027117   | 2.82x   | -64.51%        |
| 1024x1024 | multiply | sse42  | 0.076406 | 0.017119   | 4.46x   | -77.60%        |
| 1024x1024 | multiply | avx2   | 0.076406 | 0.017763   | 4.30x   | -76.75%        |
| 1024x1024 | screen   | scalar | 0.076402 | 0.027761   | 2.75x   | -63.66%        |
| 1024x1024 | screen   | sse42  | 0.076402 | 0.016710   | 4.57x   | -78.13%        |
| 1024x1024 | screen   | avx2   | 0.076402 | 0.017835   | 4.28x   | -76.66%        |
| 2048x2048 | normal   | scalar | 0.272022 | 0.088715   | 3.07x   | -67.39%        |
| 2048x2048 | normal   | sse42  | 0.272022 | 0.066855   | 4.07x   | -75.42%        |
| 2048x2048 | normal   | avx2   | 0.272022 | 0.068833   | 3.95x   | -74.70%        |
| 2048x2048 | multiply | scalar | 0.271529 | 0.106428   | 2.55x   | -60.80%        |
| 2048x2048 | multiply | sse42  | 0.271529 | 0.067615   | 4.02x   | -75.10%        |
| 2048x2048 | multiply | avx2   | 0.271529 | 0.072048   | 3.77x   | -73.47%        |
| 2048x2048 | screen   | scalar | 0.289100 | 0.113637   | 2.54x   | -60.69%        |
| 2048x2048 | screen   | sse42  | 0.289100 | 0.066957   | 4.32x   | -76.84%        |
| 2048x2048 | screen   | avx2   | 0.289100 | 0.074440   | 3.88x   | -74.25%        |
| 1280x720  | normal   | scalar | 0.064980 | 0.019841   | 3.28x   | -69.47%        |
| 1280x720  | normal   | sse42  | 0.064980 | 0.014565   | 4.46x   | -77.59%        |
| 1280x720  | normal   | avx2   | 0.064980 | 0.015361   | 4.23x   | -76.36%        |
| 1280x720  | multiply | scalar | 0.072834 | 0.023570   | 3.09x   | -67.64%        |
| 1280x720  | multiply | sse42  | 0.072834 | 0.014834   | 4.91x   | -79.63%        |
| 1280x720  | multiply | avx2   | 0.072834 | 0.015810   | 4.61x   | -78.29%        |
| 1280x720  | screen   | scalar | 0.074620 | 0.023918   | 3.12x   | -67.95%        |
| 1280x720  | screen   | sse42  | 0.074620 | 0.014915   | 5.00x   | -80.01%        |
| 1280x720  | screen   | avx2   | 0.074620 | 0.015933   | 4.68x   | -78.65%        |
| 1920x1080 | normal   | scalar | 0.133402 | 0.043408   | 3.07x   | -67.46%        |
| 1920x1080 | normal   | sse42  | 0.133402 | 0.033076   | 4.03x   | -75.21%        |
| 1920x1080 | normal   | avx2   | 0.133402 | 0.036161   | 3.69x   | -72.89%        |
| 1920x1080 | multiply | scalar | 0.142154 | 0.057037   | 2.49x   | -59.88%        |
| 1920x1080 | multiply | sse42  | 0.142154 | 0.033358   | 4.26x   | -76.53%        |
| 1920x1080 | multiply | avx2   | 0.142154 | 0.035815   | 3.97x   | -74.81%        |
| 1920x1080 | screen   | scalar | 0.142957 | 0.056232   | 2.54x   | -60.66%        |
| 1920x1080 | screen   | sse42  | 0.142957 | 0.033012   | 4.33x   | -76.91%        |
| 1920x1080 | screen   | avx2   | 0.142957 | 0.035316   | 4.05x   | -75.30%        |
| 2560x1440 | normal   | scalar | 0.227574 | 0.076433   | 2.98x   | -66.41%        |
| 2560x1440 | normal   | sse42  | 0.227574 | 0.058909   | 3.86x   | -74.11%        |
| 2560x1440 | normal   | avx2   | 0.227574 | 0.061172   | 3.72x   | -73.12%        |
| 2560x1440 | multiply | scalar | 0.240427 | 0.093281   | 2.58x   | -61.20%        |
| 2560x1440 | multiply | sse42  | 0.240427 | 0.059871   | 4.02x   | -75.10%        |
| 2560x1440 | multiply | avx2   | 0.240427 | 0.063184   | 3.81x   | -73.72%        |
| 2560x1440 | screen   | scalar | 0.248361 | 0.095404   | 2.60x   | -61.59%        |
| 2560x1440 | screen   | sse42  | 0.248361 | 0.061161   | 4.06x   | -75.37%        |
| 2560x1440 | screen   | avx2   | 0.248361 | 0.062824   | 3.95x   | -74.70%        |
| 3840x2160 | normal   | scalar | 0.522782 | 0.172257   | 3.03x   | -67.05%        |
| 3840x2160 | normal   | sse42  | 0.522782 | 0.133522   | 3.92x   | -74.46%        |
| 3840x2160 | normal   | avx2   | 0.522782 | 0.138452   | 3.78x   | -73.52%        |
| 3840x2160 | multiply | scalar | 0.521318 | 0.211922   | 2.46x   | -59.35%        |
| 3840x2160 | multiply | sse42  | 0.521318 | 0.136252   | 3.83x   | -73.86%        |
| 3840x2160 | multiply | avx2   | 0.521318 | 0.143096   | 3.64x   | -72.55%        |
| 3840x2160 | screen   | scalar | 0.548890 | 0.215867   | 2.54x   | -60.67%        |
| 3840x2160 | screen   | sse42  | 0.548890 | 0.133441   | 4.11x   | -75.69%        |
| 3840x2160 | screen   | avx2   | 0.548890 | 0.139889   | 3.92x   | -74.51%        |
