# Agent Instructions

## Testing

- Use the default Python `unittest` module for tests in this project.

## Project Conventions

To be determined. Edit this section when relevant.

## Project Skeleton Notes

- Build is wired via `pyproject.toml`, `setup.cfg`, and `setup.py` with a single extension
  module named `simd_blend_modes._simd_blend_modes`.
- C kernels live in `src/simd_blend_modes/kernels`, one file per blend mode, with a shared
  helper in `src/simd_blend_modes/kernels/blend_common.h`.
- Stub kernels currently return the first image argument; they accept 3 or 4 positional args
  to match `(background, foreground, opacity)`.
- Blend modes identified from `blend_modes/blend_modes/blending_functions.py`:
  `normal`, `soft_light`, `lighten_only`, `screen`, `dodge`, `addition`, `darken_only`,
  `multiply`, `hard_light`, `difference`, `subtract`, `grain_extract`, `grain_merge`,
  `divide`, `overlay`.
- Tests in `tests/test_blend_modes.py` are skeleton `unittest` cases with TODO skips for the
  exhaustive uint8 sweep, float32 handling, and performance comparisons.


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
