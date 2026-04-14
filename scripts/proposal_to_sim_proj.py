#!/usr/bin/env python3
"""
Split a clustered/proposal image into binary heatmaps.

When a SQL row / vlm_response is provided, only colors that appear in
vlm_response are exported, and output color names are exactly the vlm_response
keys normalized to filename-safe slugs. NPY outputs are sim_proj-style heatmaps
with shape (H, W), dtype float32, and values in {0.0, 1.0}. No blur, resize, or
post-processing is applied beyond matching proposal pixels to the visualization
palette.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.img_utils import get_palette


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "color"


def _normalize_color_name(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def _palette_by_normalized_name() -> dict:
    return {
        _normalize_color_name(name): (np.asarray(rgb, dtype=np.float32), name)
        for rgb, name in get_palette()
    }


def _load_vlm_response(args) -> dict:
    if not args.db_path:
        return {}

    if args.pipeline_info_id is not None:
        query = """
            SELECT category_name, instance_name, frame_idx, vlm_response
            FROM pipeline_info
            WHERE id = ?
        """
        params = (args.pipeline_info_id,)
    else:
        missing = [
            name for name in ("category_name", "instance_name", "frame_idx")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "--db-path requires either --pipeline-info-id or all of "
                "--category-name, --instance-name, and --frame-idx"
            )
        query = """
            SELECT category_name, instance_name, frame_idx, vlm_response
            FROM pipeline_info
            WHERE category_name = ? AND instance_name = ? AND frame_idx = ?
        """
        params = (args.category_name, args.instance_name, str(args.frame_idx))

    conn = sqlite3.connect(args.db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query, params).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError("No matching pipeline_info row found in database")

    if args.pipeline_info_id is not None:
        args.category_name = args.category_name or row["category_name"]
        args.instance_name = args.instance_name or row["instance_name"]
        args.frame_idx = args.frame_idx or row["frame_idx"]

    if not row["vlm_response"]:
        raise ValueError("Matched pipeline_info row has empty vlm_response")
    return json.loads(row["vlm_response"])


def _heatmap_for_palette_rgb(proposal_img: np.ndarray, rgb: np.ndarray, tolerance: float):
    image = proposal_img.astype(np.float32)
    dist = np.linalg.norm(image - rgb, axis=-1)
    return np.asarray(dist <= tolerance, dtype=np.float32)


def proposal_to_heatmaps(proposal_img: np.ndarray, tolerance: float, min_pixels: int, vlm_response: dict):
    palette_by_name = _palette_by_normalized_name()
    outputs = []

    if vlm_response:
        for response_color in vlm_response.keys():
            normalized_name = _normalize_color_name(response_color)
            if normalized_name not in palette_by_name:
                valid = ", ".join(sorted(name for name in palette_by_name))
                raise ValueError(
                    f"vlm_response color {response_color!r} is not in the visualization palette. "
                    f"Valid colors: {valid}"
                )
            rgb, palette_name = palette_by_name[normalized_name]
            heatmap = _heatmap_for_palette_rgb(proposal_img, rgb, tolerance)
            pixel_count = int(heatmap.sum())
            if pixel_count < min_pixels:
                raise ValueError(
                    f"vlm_response color {response_color!r} matched only {pixel_count} pixel(s) "
                    f"in proposal image; min_pixels={min_pixels}"
                )
            outputs.append((response_color, palette_name, heatmap, pixel_count))
        return outputs

    for normalized_name, (rgb, palette_name) in palette_by_name.items():
        heatmap = _heatmap_for_palette_rgb(proposal_img, rgb, tolerance)
        pixel_count = int(heatmap.sum())
        if pixel_count < min_pixels:
            continue
        outputs.append((palette_name, palette_name, heatmap, pixel_count))
    return outputs


def _require_name_parts(args):
    missing = [
        name for name in ("category_name", "instance_name", "frame_idx")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Missing output name fields: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate sim_proj-style binary NPY heatmaps from a proposal image."
    )
    parser.add_argument("--proposal-img", required=True, help="Path to clustered/proposal image.")
    parser.add_argument("--out-dir", required=True, help="Directory to save heatmap NPYs and manifest.")
    parser.add_argument("--proposal-name", default="proposal",
                        help="Name token used in output files: {category}_{instance}_{frame_idx}_{proposal}_{color}.npy.")
    parser.add_argument("--tolerance", type=float, default=8.0,
                        help="RGB Euclidean distance tolerance for palette color matching.")
    parser.add_argument("--min-pixels", type=int, default=1,
                        help="Minimum matched pixels required for each exported color.")
    parser.add_argument("--save-png", action="store_true",
                        help="Also save each heatmap as an 8-bit grayscale PNG for inspection.")
    parser.add_argument("--vlm-response-json", default=None,
                        help="Optional vlm_response JSON string or JSON file path. If set, only those colors are exported.")

    parser.add_argument("--db-path", default=None,
                        help="Optional SQLite db path. Used to load vlm_response from pipeline_info.")
    parser.add_argument("--pipeline-info-id", type=int, default=None,
                        help="Optional pipeline_info.id for reading vlm_response and output name fields.")
    parser.add_argument("--category-name", default=None,
                        help="category_name used in output filenames and optional SQL lookup.")
    parser.add_argument("--instance-name", default=None,
                        help="instance_name used in output filenames and optional SQL lookup.")
    parser.add_argument("--frame-idx", default=None,
                        help="frame_idx used in output filenames and optional SQL lookup.")

    args = parser.parse_args()

    proposal_path = Path(args.proposal_img)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vlm_response = _load_vlm_response(args)
    if args.vlm_response_json:
        vlm_response_path = Path(args.vlm_response_json)
        if vlm_response_path.exists():
            with vlm_response_path.open("r", encoding="utf-8") as f:
                vlm_response = json.load(f)
        else:
            vlm_response = json.loads(args.vlm_response_json)

    _require_name_parts(args)

    proposal_img = np.asarray(Image.open(proposal_path).convert("RGB"))
    base_name = "_".join(
        _slugify(part)
        for part in (args.category_name, args.instance_name, str(args.frame_idx), args.proposal_name)
    )
    manifest = {
        "proposal_img": str(proposal_path),
        "category_name": args.category_name,
        "instance_name": args.instance_name,
        "frame_idx": str(args.frame_idx),
        "proposal_name": args.proposal_name,
        "shape": list(proposal_img.shape[:2]),
        "tolerance": args.tolerance,
        "min_pixels": args.min_pixels,
        "strict_vlm_response_colors": bool(vlm_response),
        "colors": [],
    }

    for response_color, palette_name, heatmap, pixel_count in proposal_to_heatmaps(
        proposal_img,
        tolerance=args.tolerance,
        min_pixels=args.min_pixels,
        vlm_response=vlm_response,
    ):
        color_slug = _slugify(response_color)
        npy_path = out_dir / f"{base_name}_{color_slug}.npy"
        np.save(npy_path, heatmap.astype(np.float32))

        png_path = None
        if args.save_png:
            png_path = out_dir / f"{base_name}_{color_slug}.png"
            Image.fromarray((heatmap * 255).astype(np.uint8)).save(png_path)

        manifest["colors"].append({
            "color": response_color,
            "palette_color": palette_name,
            "heatmap_npy": str(npy_path),
            "heatmap_png": str(png_path) if png_path else None,
            "pixel_count": pixel_count,
            "vlm_response": vlm_response.get(response_color) if vlm_response else None,
        })

    manifest_path = out_dir / f"{base_name}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(manifest['colors'])} heatmap NPY(s) to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
