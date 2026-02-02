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
| 256x256   | normal   | scalar | 0.006004 | 0.001496   | 4.01x   | -75.09%        |
| 256x256   | normal   | sse42  | 0.006004 | 0.001037   | 5.79x   | -82.73%        |
| 256x256   | normal   | avx2   | 0.006004 | 0.001062   | 5.65x   | -82.31%        |
| 256x256   | normal   | auto   | 0.006004 | 0.001102   | 5.45x   | -81.64%        |
| 256x256   | multiply | scalar | 0.007002 | 0.001635   | 4.28x   | -76.65%        |
| 256x256   | multiply | sse42  | 0.007002 | 0.001047   | 6.69x   | -85.05%        |
| 256x256   | multiply | avx2   | 0.007002 | 0.001108   | 6.32x   | -84.17%        |
| 256x256   | multiply | auto   | 0.007002 | 0.001096   | 6.39x   | -84.34%        |
| 256x256   | screen   | scalar | 0.006307 | 0.001683   | 3.75x   | -73.31%        |
| 256x256   | screen   | sse42  | 0.006307 | 0.001035   | 6.09x   | -83.59%        |
| 256x256   | screen   | avx2   | 0.006307 | 0.001090   | 5.79x   | -82.72%        |
| 256x256   | screen   | auto   | 0.006307 | 0.001083   | 5.82x   | -82.82%        |
| 512x512   | normal   | scalar | 0.021775 | 0.005323   | 4.09x   | -75.56%        |
| 512x512   | normal   | sse42  | 0.021775 | 0.004073   | 5.35x   | -81.30%        |
| 512x512   | normal   | avx2   | 0.021775 | 0.004214   | 5.17x   | -80.65%        |
| 512x512   | normal   | auto   | 0.021775 | 0.004279   | 5.09x   | -80.35%        |
| 512x512   | multiply | scalar | 0.026253 | 0.006603   | 3.98x   | -74.85%        |
| 512x512   | multiply | sse42  | 0.026253 | 0.004188   | 6.27x   | -84.05%        |
| 512x512   | multiply | avx2   | 0.026253 | 0.004710   | 5.57x   | -82.06%        |
| 512x512   | multiply | auto   | 0.026253 | 0.004692   | 5.60x   | -82.13%        |
| 512x512   | screen   | scalar | 0.026615 | 0.006623   | 4.02x   | -75.12%        |
| 512x512   | screen   | sse42  | 0.026615 | 0.004148   | 6.42x   | -84.42%        |
| 512x512   | screen   | avx2   | 0.026615 | 0.004616   | 5.77x   | -82.66%        |
| 512x512   | screen   | auto   | 0.026615 | 0.004395   | 6.06x   | -83.49%        |
| 1024x1024 | normal   | scalar | 0.065886 | 0.021259   | 3.10x   | -67.73%        |
| 1024x1024 | normal   | sse42  | 0.065886 | 0.016313   | 4.04x   | -75.24%        |
| 1024x1024 | normal   | avx2   | 0.065886 | 0.016948   | 3.89x   | -74.28%        |
| 1024x1024 | normal   | auto   | 0.065886 | 0.016921   | 3.89x   | -74.32%        |
| 1024x1024 | multiply | scalar | 0.070933 | 0.025619   | 2.77x   | -63.88%        |
| 1024x1024 | multiply | sse42  | 0.070933 | 0.016740   | 4.24x   | -76.40%        |
| 1024x1024 | multiply | avx2   | 0.070933 | 0.017640   | 4.02x   | -75.13%        |
| 1024x1024 | multiply | auto   | 0.070933 | 0.017686   | 4.01x   | -75.07%        |
| 1024x1024 | screen   | scalar | 0.071614 | 0.026774   | 2.67x   | -62.61%        |
| 1024x1024 | screen   | sse42  | 0.071614 | 0.016593   | 4.32x   | -76.83%        |
| 1024x1024 | screen   | avx2   | 0.071614 | 0.017388   | 4.12x   | -75.72%        |
| 1024x1024 | screen   | auto   | 0.071614 | 0.017408   | 4.11x   | -75.69%        |
| 2048x2048 | normal   | scalar | 0.255385 | 0.084978   | 3.01x   | -66.73%        |
| 2048x2048 | normal   | sse42  | 0.255385 | 0.065629   | 3.89x   | -74.30%        |
| 2048x2048 | normal   | avx2   | 0.255385 | 0.067880   | 3.76x   | -73.42%        |
| 2048x2048 | normal   | auto   | 0.255385 | 0.068304   | 3.74x   | -73.25%        |
| 2048x2048 | multiply | scalar | 0.260006 | 0.102502   | 2.54x   | -60.58%        |
| 2048x2048 | multiply | sse42  | 0.260006 | 0.068822   | 3.78x   | -73.53%        |
| 2048x2048 | multiply | avx2   | 0.260006 | 0.072012   | 3.61x   | -72.30%        |
| 2048x2048 | multiply | auto   | 0.260006 | 0.072611   | 3.58x   | -72.07%        |
| 2048x2048 | screen   | scalar | 0.318748 | 0.107727   | 2.96x   | -66.20%        |
| 2048x2048 | screen   | sse42  | 0.318748 | 0.066704   | 4.78x   | -79.07%        |
| 2048x2048 | screen   | avx2   | 0.318748 | 0.070786   | 4.50x   | -77.79%        |
| 2048x2048 | screen   | auto   | 0.318748 | 0.069646   | 4.58x   | -78.15%        |
| 1280x720  | normal   | scalar | 0.061804 | 0.019272   | 3.21x   | -68.82%        |
| 1280x720  | normal   | sse42  | 0.061804 | 0.014420   | 4.29x   | -76.67%        |
| 1280x720  | normal   | avx2   | 0.061804 | 0.014913   | 4.14x   | -75.87%        |
| 1280x720  | normal   | auto   | 0.061804 | 0.015009   | 4.12x   | -75.71%        |
| 1280x720  | multiply | scalar | 0.072218 | 0.022572   | 3.20x   | -68.75%        |
| 1280x720  | multiply | sse42  | 0.072218 | 0.014724   | 4.90x   | -79.61%        |
| 1280x720  | multiply | avx2   | 0.072218 | 0.015602   | 4.63x   | -78.40%        |
| 1280x720  | multiply | auto   | 0.072218 | 0.015483   | 4.66x   | -78.56%        |
| 1280x720  | screen   | scalar | 0.073607 | 0.023303   | 3.16x   | -68.34%        |
| 1280x720  | screen   | sse42  | 0.073607 | 0.015342   | 4.80x   | -79.16%        |
| 1280x720  | screen   | avx2   | 0.073607 | 0.015317   | 4.81x   | -79.19%        |
| 1280x720  | screen   | auto   | 0.073607 | 0.016084   | 4.58x   | -78.15%        |
| 1920x1080 | normal   | scalar | 0.132466 | 0.042486   | 3.12x   | -67.93%        |
| 1920x1080 | normal   | sse42  | 0.132466 | 0.032318   | 4.10x   | -75.60%        |
| 1920x1080 | normal   | avx2   | 0.132466 | 0.033623   | 3.94x   | -74.62%        |
| 1920x1080 | normal   | auto   | 0.132466 | 0.033686   | 3.93x   | -74.57%        |
| 1920x1080 | multiply | scalar | 0.143956 | 0.057071   | 2.52x   | -60.36%        |
| 1920x1080 | multiply | sse42  | 0.143956 | 0.034130   | 4.22x   | -76.29%        |
| 1920x1080 | multiply | avx2   | 0.143956 | 0.036043   | 3.99x   | -74.96%        |
| 1920x1080 | multiply | auto   | 0.143956 | 0.035634   | 4.04x   | -75.25%        |
| 1920x1080 | screen   | scalar | 0.144221 | 0.053623   | 2.69x   | -62.82%        |
| 1920x1080 | screen   | sse42  | 0.144221 | 0.033390   | 4.32x   | -76.85%        |
| 1920x1080 | screen   | avx2   | 0.144221 | 0.035123   | 4.11x   | -75.65%        |
| 1920x1080 | screen   | auto   | 0.144221 | 0.035158   | 4.10x   | -75.62%        |
| 2560x1440 | normal   | scalar | 0.216557 | 0.076216   | 2.84x   | -64.81%        |
| 2560x1440 | normal   | sse42  | 0.216557 | 0.058363   | 3.71x   | -73.05%        |
| 2560x1440 | normal   | avx2   | 0.216557 | 0.060553   | 3.58x   | -72.04%        |
| 2560x1440 | normal   | auto   | 0.216557 | 0.060850   | 3.56x   | -71.90%        |
| 2560x1440 | multiply | scalar | 0.239886 | 0.092330   | 2.60x   | -61.51%        |
| 2560x1440 | multiply | sse42  | 0.239886 | 0.060642   | 3.96x   | -74.72%        |
| 2560x1440 | multiply | avx2   | 0.239886 | 0.062758   | 3.82x   | -73.84%        |
| 2560x1440 | multiply | auto   | 0.239886 | 0.062706   | 3.83x   | -73.86%        |
| 2560x1440 | screen   | scalar | 0.240427 | 0.095784   | 2.51x   | -60.16%        |
| 2560x1440 | screen   | sse42  | 0.240427 | 0.058822   | 4.09x   | -75.53%        |
| 2560x1440 | screen   | avx2   | 0.240427 | 0.061671   | 3.90x   | -74.35%        |
| 2560x1440 | screen   | auto   | 0.240427 | 0.061862   | 3.89x   | -74.27%        |
| 3840x2160 | normal   | scalar | 0.492737 | 0.171048   | 2.88x   | -65.29%        |
| 3840x2160 | normal   | sse42  | 0.492737 | 0.130546   | 3.77x   | -73.51%        |
| 3840x2160 | normal   | avx2   | 0.492737 | 0.136078   | 3.62x   | -72.38%        |
| 3840x2160 | normal   | auto   | 0.492737 | 0.135510   | 3.64x   | -72.50%        |
| 3840x2160 | multiply | scalar | 0.513104 | 0.226117   | 2.27x   | -55.93%        |
| 3840x2160 | multiply | sse42  | 0.513104 | 0.139031   | 3.69x   | -72.90%        |
| 3840x2160 | multiply | avx2   | 0.513104 | 0.149606   | 3.43x   | -70.84%        |
| 3840x2160 | multiply | auto   | 0.513104 | 0.142730   | 3.59x   | -72.18%        |
| 3840x2160 | screen   | scalar | 0.552903 | 0.241320   | 2.29x   | -56.35%        |
| 3840x2160 | screen   | sse42  | 0.552903 | 0.141164   | 3.92x   | -74.47%        |
| 3840x2160 | screen   | avx2   | 0.552903 | 0.147013   | 3.76x   | -73.41%        |
| 3840x2160 | screen   | auto   | 0.552903 | 0.140781   | 3.93x   | -74.54%        |
