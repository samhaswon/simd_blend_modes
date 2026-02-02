from typing import Any

import numpy as np


def normal(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def soft_light(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def lighten_only(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def screen(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def dodge(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def addition(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def darken_only(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def multiply(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def hard_light(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def difference(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def subtract(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def grain_extract(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def grain_merge(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def divide(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...

def overlay(
    background: np.ndarray,
    foreground: np.ndarray,
    opacity: float = ...,
) -> np.ndarray: ...
