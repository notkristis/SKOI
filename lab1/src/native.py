"""
Native adaptive threshold binarization (mean and Gaussian).

Does not use OpenCV adaptiveThreshold; uses NumPy only for arrays and basic ops.
"""

from __future__ import annotations

import numpy as np


def _ensure_odd_block_size(block_size: int) -> int:
    if block_size < 3:
        raise ValueError("block_size must be >= 3")
    if block_size % 2 == 0:
        block_size += 1
    return block_size


def _integral_image(img: np.ndarray) -> np.ndarray:
    """Summed-area table; shape (h+1, w+1), top/left row is 0."""
    sat = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    sat[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    return sat


def _local_sum_from_integral(sat: np.ndarray, h: int, w: int, block_size: int) -> np.ndarray:
    """Local sum for each pixel; windows align with unpadded (y,x) on reflect-padded input."""
    y0 = np.arange(h)
    x0 = np.arange(w)
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1 = Y0 + block_size
    X1 = X0 + block_size
    return sat[Y1, X1] - sat[Y0, X1] - sat[Y1, X0] + sat[Y0, X0]


def adaptive_threshold_mean(
    gray: np.ndarray,
    block_size: int = 11,
    C: float = 2.0,
    max_value: int = 255,
) -> np.ndarray:
    """
    Adaptive binarization: pixel > (local_mean - C) -> max_value, else 0.

    gray: uint8 single channel.
    """
    block_size = _ensure_odd_block_size(block_size)
    gray = np.asarray(gray, dtype=np.uint8)
    pad = block_size // 2
    padded = np.pad(gray, pad, mode="reflect")
    sat = _integral_image(padded)
    h, w = gray.shape
    local_sum = _local_sum_from_integral(sat, h, w, block_size)
    area = block_size * block_size
    local_mean = local_sum / area
    thresh_map = local_mean - C
    out = np.where(gray.astype(np.float64) > thresh_map, max_value, 0).astype(np.uint8)
    return out


def _gaussian_kernel(block_size: int, sigma: float | None = None) -> np.ndarray:
    if sigma is None:
        sigma = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8
        if sigma <= 0:
            sigma = 0.1
    ax = np.arange(block_size) - block_size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float64)


def _convolve1d_reflect(src: np.ndarray, kernel1d: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1D convolution with reflect padding (vectorized along one axis)."""
    radius = len(kernel1d) // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(src, pad_width, mode="reflect")
    out = np.zeros_like(src, dtype=np.float64)
    k = kernel1d[::-1]
    h, w = src.shape
    if axis == 1:
        for dx in range(len(kernel1d)):
            out += k[dx] * padded[:, dx : dx + w]
    else:
        for dy in range(len(kernel1d)):
            out += k[dy] * padded[dy : dy + h, :]
    return out


def _convolve_valid_reflect(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable Gaussian convolution (2×1D), reflect borders."""
    k1 = kernel[kernel.shape[0] // 2, :].astype(np.float64)
    k1 /= k1.sum()
    tmp = _convolve1d_reflect(gray.astype(np.float64), k1, axis=1)
    return _convolve1d_reflect(tmp, k1, axis=0)


def adaptive_threshold_gaussian(
    gray: np.ndarray,
    block_size: int = 11,
    C: float = 2.0,
    max_value: int = 255,
) -> np.ndarray:
    """
    Adaptive binarization with Gaussian-weighted local mean (OpenCV-compatible sigma).
    """
    block_size = _ensure_odd_block_size(block_size)
    gray = np.asarray(gray, dtype=np.uint8)
    kernel = _gaussian_kernel(block_size)
    local_mean = _convolve_valid_reflect(gray, kernel)
    thresh_map = local_mean - C
    out = np.where(gray.astype(np.float64) > thresh_map, max_value, 0).astype(np.uint8)
    return out
