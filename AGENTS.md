# Agent Instructions

## Testing

- Build and install with `pip install --force-reinstall -e .`.

- Use the default Python `unittest` module for tests in this project.

  - Tests are run with `python3 -m unittest discover tests/`.
  - If an ARM Docker build fails with `exec /bin/sh: exec format error`, enable QEMU/binfmt
    emulation first: `docker run --privileged --rm tonistiigi/binfmt --install arm64`.

## Project Notes

- Build is wired via `pyproject.toml`, `setup.cfg`, and `setup.py` with a single extension
  module named `simd_blend_modes._simd_blend_modes`.
- C kernels live in `src/simd_blend_modes/kernels`, one file per blend mode, with a shared
  helper in `src/simd_blend_modes/kernels/blend_common.h`.
- Scalar kernels are fully implemented for all blend modes; SIMD paths are scaffolded but not
  yet implemented.
- Blend modes identified from `blend_modes/blend_modes/blending_functions.py`:
  `normal`, `soft_light`, `lighten_only`, `screen`, `dodge`, `addition`, `darken_only`,
  `multiply`, `hard_light`, `difference`, `subtract`, `grain_extract`, `grain_merge`,
  `divide`, `overlay`.
- Scalar kernels now implement the blend math from `blend_modes/blend_modes/blending_functions.py`
  for uint8/float32 NumPy arrays; output dtype and channel count match the background image, missing
  alpha channels are treated as 255 (opaque), and opacity defaults to 1.0 when omitted.
- Tests in `tests/test_blend_modes.py` now include full exhaustive uint8 and float32 RGBA sweeps
  (chunked for memory) plus parity checks against the reference implementation.
- ARM builds are scalar-only; SIMD intrinsics are x86-only and gated at compile time. CI does not
  emit ARM artifacts.


## Code Style

When writing Python, follow PEP8 standards with a line length maximum of 100 characters. Python function docstrings should follow the reStructuredText style with field lists (i.e., `:param x: ...`), also being succinct and clear. Variable names should make clear what the variable is for without being too long. Don't use comments for code with a clear purpose and actions.

Write secure code. 
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one, and preferably only one, obvious way to do it.
