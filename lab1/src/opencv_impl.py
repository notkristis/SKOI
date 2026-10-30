"""OpenCV wrappers for adaptive threshold binarization."""

from __future__ import annotations

import cv2
import numpy as np


def adaptive_threshold_mean(
    gray: np.ndarray,
    block_size: int = 11,
    C: float = 2.0,
    max_value: int = 255,
) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    gray = np.asarray(gray, dtype=np.uint8)
    return cv2.adaptiveThreshold(
        gray,
        max_value,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )


def adaptive_threshold_gaussian(
    gray: np.ndarray,
    block_size: int = 11,
    C: float = 2.0,
    max_value: int = 255,
) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    gray = np.asarray(gray, dtype=np.uint8)
    return cv2.adaptiveThreshold(
        gray,
        max_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )
