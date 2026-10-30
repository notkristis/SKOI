#!/usr/bin/env python3
"""Build PDF report from report.md (requires pandoc + XeLaTeX)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_MD = ROOT / "report.md"
OUT_PDF = ROOT / "output" / "LR1_report.pdf"


def ensure_assets() -> None:
    out = ROOT / "output"
    need = ["comparison_strip.png", "benchmark_chart.png"]
    if all((out / name).exists() for name in need):
        return
    print("Generating missing images (demo + benchmark)...")
    subprocess.run([sys.executable, str(ROOT / "demo.py")], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "benchmark.py")], check=True, cwd=ROOT)


def main() -> None:
    if shutil.which("pandoc") is None:
        sys.exit("pandoc not found. Install: brew install pandoc basictex (or MacTeX).")
    if shutil.which("xelatex") is None:
        sys.exit("xelatex not found. Install a TeX distribution (BasicTeX / MacTeX).")

    ensure_assets()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pandoc",
        str(REPORT_MD),
        "-o",
        str(OUT_PDF),
        "--pdf-engine=xelatex",
        f"--resource-path={ROOT}",
        "-V",
        "lang=ru-RU",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
