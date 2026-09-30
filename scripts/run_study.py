#!/usr/bin/env python3
from pathlib import Path

from lunar_band_design.study import run


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run(root, root / "configs/study.yaml")
