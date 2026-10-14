"""Load grayscale images and synthetic test patterns."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def make_test_image(height: int = 480, width: int = 640) -> np.ndarray:
    """Uneven illumination + text-like structure for adaptive threshold demo."""
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    background = 40 + 120 * (0.6 * x + 0.4 * y)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 8, (height, width))
    img = background + noise

    # Dark "characters"
    for i in range(8):
        cy = int(height * (0.15 + 0.1 * i))
        cx = int(width * (0.2 + 0.08 * (i % 4)))
        img[cy : cy + 40, cx : cx + 120] -= 55

    return np.clip(img, 0, 255).astype(np.uint8)


def ensure_sample_image(assets_dir: Path) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    path = assets_dir / "sample.png"
    if not path.exists():
        cv2.imwrite(str(path), make_test_image())
    return path
