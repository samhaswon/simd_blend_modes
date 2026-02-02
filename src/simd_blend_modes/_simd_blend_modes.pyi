from typing import Any

import numpy as np


def normal(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def soft_light(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def lighten_only(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def screen(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def dodge(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def addition(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def darken_only(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def multiply(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def hard_light(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def difference(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def subtract(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def grain_extract(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def grain_merge(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def divide(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def overlay(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
    kernel: str | Any = ...,
) -> np.ndarray: ...

def kernel_available(name: str) -> bool: ...
