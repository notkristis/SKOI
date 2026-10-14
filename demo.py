#!/usr/bin/env python3
"""Generate comparison images for the report."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from src import io_utils
from src import native as native_impl
from src import opencv_impl as cv_impl

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample = io_utils.ensure_sample_image(ASSETS)
    if len(sys.argv) > 1:
        gray = io_utils.load_grayscale(Path(sys.argv[1]))
    else:
        gray = io_utils.load_grayscale(sample)

    block_size = 11
    C = 2.0
    methods = {
        "mean_opencv": cv_impl.adaptive_threshold_mean(gray, block_size, C),
        "mean_native": native_impl.adaptive_threshold_mean(gray, block_size, C),
        "gaussian_opencv": cv_impl.adaptive_threshold_gaussian(gray, block_size, C),
        "gaussian_native": native_impl.adaptive_threshold_gaussian(gray, block_size, C),
    }

    cv2.imwrite(str(OUTPUT / "input_gray.png"), gray)
    for name, img in methods.items():
        cv2.imwrite(str(OUTPUT / f"{name}.png"), img)

    # Side-by-side: original | mean CV | mean native | gaussian CV | gaussian native
    h, w = gray.shape
    panel = np.zeros((h, w * 5), dtype=np.uint8)
    panel[:, 0:w] = gray
    panel[:, w : 2 * w] = methods["mean_opencv"]
    panel[:, 2 * w : 3 * w] = methods["mean_native"]
    panel[:, 3 * w : 4 * w] = methods["gaussian_opencv"]
    panel[:, 4 * w : 5 * w] = methods["gaussian_native"]
    cv2.imwrite(str(OUTPUT / "comparison_strip.png"), panel)

    print("Mean OpenCV vs native MAE:", mae(methods["mean_opencv"], methods["mean_native"]))
    print("Gaussian OpenCV vs native MAE:", mae(methods["gaussian_opencv"], methods["gaussian_native"]))
    print(f"Outputs in {OUTPUT}")


if __name__ == "__main__":
    main()
