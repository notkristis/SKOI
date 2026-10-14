#!/usr/bin/env python3
"""Compare OpenCV vs native adaptive threshold performance."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Writable cache for matplotlib in restricted environments
_mpl_dir = Path(__file__).resolve().parent / "output" / ".mpl"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib.pyplot as plt
import numpy as np

from src import io_utils
from src import native as native_impl
from src import opencv_impl as cv_impl

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"


def _time_call(fn, *args, warmup: int = 2, repeats: int = 5) -> float:
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_benchmark(sizes: list[tuple[int, int]]) -> dict:
    block_size = 11
    C = 2.0
    results = {"block_size": block_size, "C": C, "sizes": []}

    for h, w in sizes:
        gray = io_utils.make_test_image(h, w)
        row = {"height": h, "width": w, "methods": {}}
        pairs = [
            ("mean_opencv", cv_impl.adaptive_threshold_mean),
            ("mean_native", native_impl.adaptive_threshold_mean),
            ("gaussian_opencv", cv_impl.adaptive_threshold_gaussian),
            ("gaussian_native", native_impl.adaptive_threshold_gaussian),
        ]
        for name, func in pairs:
            sec = _time_call(func, gray, block_size, C)
            row["methods"][name] = {"seconds": sec, "mpix_per_sec": (h * w / 1e6) / sec}
        results["sizes"].append(row)
        print(f"{w}x{h}: " + ", ".join(f"{k}={row['methods'][k]['seconds']:.4f}s" for k in row["methods"]))

    return results


def plot_results(data: dict, out_path: Path) -> None:
    labels = [f"{s['width']}×{s['height']}" for s in data["sizes"]]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    series = [
        ("mean OpenCV", "mean_opencv", "#2ecc71"),
        ("mean native", "mean_native", "#27ae60"),
        ("Gaussian OpenCV", "gaussian_opencv", "#3498db"),
        ("Gaussian native", "gaussian_native", "#2980b9"),
    ]
    for i, (title, key, color) in enumerate(series):
        vals = [s["methods"][key]["seconds"] for s in data["sizes"]]
        ax.bar(x + (i - 1.5) * width, vals, width, label=title, color=color)
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Image size")
    ax.set_title("Adaptive threshold: OpenCV vs native implementation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    io_utils.ensure_sample_image(ASSETS)

    sizes = [(480, 640), (720, 1280), (900, 1600)]
    # Smaller set for gaussian native on huge images — still run all
    data = run_benchmark(sizes)
    json_path = OUTPUT / "benchmark.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot_results(data, OUTPUT / "benchmark_chart.png")
    print(f"Saved {json_path} and {OUTPUT / 'benchmark_chart.png'}")


if __name__ == "__main__":
    main()
