#!/usr/bin/env python3
"""Convert one or more .npy heatmaps to PNG images for quick inspection."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap after squeeze, got shape {arr.shape}")

    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)

    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize .npy heatmaps as PNG files.")
    parser.add_argument("inputs", nargs="+", help="Input .npy file(s) or directories containing .npy files.")
    parser.add_argument("--out-dir", required=True, help="Directory to save PNG visualizations.")
    parser.add_argument("--suffix", default="viz", help="Suffix added before .png in output filenames.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_paths = []
    for item in args.inputs:
        path = Path(item)
        if path.is_dir():
            npy_paths.extend(sorted(path.glob("*.npy")))
        else:
            npy_paths.append(path)

    saved = []
    for npy_path in npy_paths:
        arr = np.load(npy_path)
        png = _to_uint8(arr)
        out_path = out_dir / f"{npy_path.stem}_{args.suffix}.png"
        Image.fromarray(png).save(out_path)
        saved.append(out_path)

    print(f"Saved {len(saved)} PNG visualization(s) to {out_dir}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
