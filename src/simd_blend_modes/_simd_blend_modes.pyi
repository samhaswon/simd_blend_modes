from typing import Literal

import numpy as np

from . import KernelKind

KernelType = KernelKind | Literal["auto", "scalar", "sse42", "avx2"]


def normal(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using normal mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def soft_light(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using soft light mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def lighten_only(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using lighten-only mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def screen(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using screen mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def dodge(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using dodge mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def addition(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using addition mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def darken_only(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using darken-only mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def multiply(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using multiply mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def hard_light(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using hard light mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def difference(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using difference mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def subtract(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using subtract mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def grain_extract(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using grain extract mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def grain_merge(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using grain merge mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def divide(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using divide mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def overlay(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using overlay mode.

    :param background: Background image array (H, W, 3/4), uint8 or float32.
    :param foreground: Foreground image array (H, W, 3/4), uint8 or float32.
    :param opacity: Blend opacity in [0, 1]. Defaults to 1.0.
    :param kernel: Kernel selection (KernelKind or string literal).
    :return: Blended image with dtype/channels matching background.
    """
    ...

def hsv_hue(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using HSV hue mode."""
    ...

def hsv_saturation(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using HSV saturation mode."""
    ...

def hsv_value(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using HSV value mode."""
    ...

def hsl_color(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using HSL color mode."""
    ...

def lch_hue(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using LCh hue mode."""
    ...

def lch_chroma(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using LCh chroma mode."""
    ...

def lch_color(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using LCh color mode."""
    ...

def lch_lightness(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using LCh lightness mode."""
    ...

def burn(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using burn mode."""
    ...

def linear_burn(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using linear burn mode."""
    ...

def exclusion(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using exclusion mode."""
    ...

def vivid_light(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using vivid light mode."""
    ...

def pin_light(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: KernelType = "auto",
) -> np.ndarray:
    """Blend using pin light mode."""
    ...

def kernel_available(name: KernelType) -> bool:
    """Return True if the requested kernel is supported on this machine.

    :param name: Kernel selection (KernelKind or string literal).
    :return: True if available on this machine.
    """
    ...
