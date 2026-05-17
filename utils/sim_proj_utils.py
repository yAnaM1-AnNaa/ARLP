"""Utilities for converting colored proposal RGB images into sim_proj maps.

The expected input is a palette-colored proposal/cluster image, not an
unannotated natural RGB image. Each selected palette color is converted into a
binary heatmap with shape ``(H, W)`` and dtype ``float32``.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.img_utils import get_palette


def slugify(text: Any) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "color"


def normalize_color_name(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def palette_by_normalized_name() -> dict[str, tuple[np.ndarray, str]]:
    return {
        normalize_color_name(name): (np.asarray(rgb, dtype=np.float32), name)
        for rgb, name in get_palette()
    }


def load_vlm_response_from_db(
    db_path: str | Path,
    *,
    pipeline_info_id: int | None = None,
    category_name: str | None = None,
    instance_name: str | None = None,
    frame_idx: str | int | None = None,
) -> tuple[dict, dict[str, Any]]:
    """Load a pipeline_info row and parse its vlm_response JSON payload."""
    if pipeline_info_id is not None:
        query = """
            SELECT category_name, instance_name, frame_idx, vlm_response
            FROM pipeline_info
            WHERE id = ?
        """
        params = (pipeline_info_id,)
    else:
        missing = [
            name
            for name, value in (
                ("category_name", category_name),
                ("instance_name", instance_name),
                ("frame_idx", frame_idx),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "db lookup requires either pipeline_info_id or all of "
                f"category_name, instance_name, frame_idx; missing {missing}"
            )
        query = """
            SELECT category_name, instance_name, frame_idx, vlm_response
            FROM pipeline_info
            WHERE category_name = ? AND instance_name = ? AND frame_idx = ?
        """
        params = (category_name, instance_name, str(frame_idx))

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query, params).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError("No matching pipeline_info row found in database")
    if not row["vlm_response"]:
        raise ValueError("Matched pipeline_info row has empty vlm_response")

    row_dict = dict(row)
    return json.loads(row["vlm_response"]), row_dict


def load_vlm_response_json(value: str | Path | None) -> dict:
    """Load vlm_response from a JSON string, JSON file, or return empty dict."""
    if value is None:
        return {}

    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(str(value))


def heatmap_for_palette_rgb(
    proposal_img: np.ndarray,
    rgb: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    image = proposal_img.astype(np.float32)
    dist = np.linalg.norm(image - rgb, axis=-1)
    return np.asarray(dist <= tolerance, dtype=np.float32)


def proposal_to_heatmaps(
    proposal_img: np.ndarray,
    tolerance: float = 8.0,
    min_pixels: int = 1,
    vlm_response: dict | None = None,
) -> list[tuple[str, str, np.ndarray, int]]:
    """Convert a palette-colored proposal image into binary heatmaps.

    Returns tuples of ``(response_color, palette_name, heatmap, pixel_count)``.
    If ``vlm_response`` is provided, only colors in that mapping are exported.
    """
    palette_by_name = palette_by_normalized_name()
    outputs = []
    vlm_response = vlm_response or {}

    if vlm_response:
        for response_color in vlm_response.keys():
            normalized_name = normalize_color_name(response_color)
            if normalized_name not in palette_by_name:
                valid = ", ".join(sorted(palette_by_name))
                raise ValueError(
                    f"vlm_response color {response_color!r} is not in the visualization palette. "
                    f"Valid colors: {valid}"
                )
            rgb, palette_name = palette_by_name[normalized_name]
            heatmap = heatmap_for_palette_rgb(proposal_img, rgb, tolerance)
            pixel_count = int(heatmap.sum())
            if pixel_count < min_pixels:
                raise ValueError(
                    f"vlm_response color {response_color!r} matched only {pixel_count} pixel(s) "
                    f"in proposal image; min_pixels={min_pixels}"
                )
            outputs.append((response_color, palette_name, heatmap, pixel_count))
        return outputs

    for _, (rgb, palette_name) in palette_by_name.items():
        heatmap = heatmap_for_palette_rgb(proposal_img, rgb, tolerance)
        pixel_count = int(heatmap.sum())
        if pixel_count >= min_pixels:
            outputs.append((palette_name, palette_name, heatmap, pixel_count))
    return outputs


def write_sim_proj_from_proposal_rgb(
    proposal_img: str | Path | Image.Image | np.ndarray,
    out_dir: str | Path,
    *,
    category_name: str,
    instance_name: str,
    frame_idx: str | int,
    proposal_name: str = "proposal",
    vlm_response: dict | None = None,
    tolerance: float = 8.0,
    min_pixels: int = 1,
    save_png: bool = False,
) -> tuple[dict[str, str], dict]:
    """Write sim_proj NPY files and return ``(sim_proj_json, manifest)``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(proposal_img, (str, Path)):
        proposal_img_path = Path(proposal_img)
        proposal_arr = np.asarray(Image.open(proposal_img_path).convert("RGB"))
        proposal_ref = str(proposal_img_path)
    elif isinstance(proposal_img, Image.Image):
        proposal_arr = np.asarray(proposal_img.convert("RGB"))
        proposal_ref = None
    else:
        proposal_arr = np.asarray(proposal_img)
        proposal_ref = None
        if proposal_arr.ndim != 3 or proposal_arr.shape[-1] < 3:
            raise ValueError("proposal_img array must have shape (H, W, 3)")
        proposal_arr = proposal_arr[..., :3]

    base_name = "_".join(
        slugify(part)
        for part in (category_name, instance_name, str(frame_idx), proposal_name)
    )
    manifest = {
        "proposal_img": proposal_ref,
        "category_name": category_name,
        "instance_name": instance_name,
        "frame_idx": str(frame_idx),
        "proposal_name": proposal_name,
        "shape": list(proposal_arr.shape[:2]),
        "tolerance": tolerance,
        "min_pixels": min_pixels,
        "strict_vlm_response_colors": bool(vlm_response),
        "colors": [],
    }

    sim_proj = {}
    for response_color, palette_name, heatmap, pixel_count in proposal_to_heatmaps(
        proposal_arr,
        tolerance=tolerance,
        min_pixels=min_pixels,
        vlm_response=vlm_response,
    ):
        color_slug = slugify(response_color)
        npy_path = out_dir / f"{base_name}_{color_slug}.npy"
        np.save(npy_path, heatmap.astype(np.float32))
        sim_proj[response_color] = str(npy_path.resolve())

        png_path = None
        if save_png:
            png_path = out_dir / f"{base_name}_{color_slug}.png"
            Image.fromarray((heatmap * 255).astype(np.uint8)).save(png_path)

        manifest["colors"].append({
            "color": response_color,
            "palette_color": palette_name,
            "heatmap_npy": str(npy_path.resolve()),
            "heatmap_png": str(png_path.resolve()) if png_path else None,
            "pixel_count": pixel_count,
            "vlm_response": vlm_response.get(response_color) if vlm_response else None,
        })

    if not sim_proj:
        raise ValueError("No sim_proj files were generated from proposal image.")

    manifest_path = out_dir / f"{base_name}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = str(manifest_path.resolve())
    return sim_proj, manifest


def build_sim_proj_json(
    proposal_img_path: str | Path,
    sim_proj_dir: str | Path,
    category_name: str,
    instance_key: str,
    frame_idx: int,
    region_matching: dict,
    tolerance: float,
    min_pixels: int,
) -> dict[str, str]:
    """Pipeline helper: generate proposal-mask sim_proj files for SQL payloads."""
    sim_proj, _ = write_sim_proj_from_proposal_rgb(
        proposal_img_path,
        sim_proj_dir,
        category_name=category_name,
        instance_name=instance_key,
        frame_idx=frame_idx,
        proposal_name="proposal",
        vlm_response=region_matching,
        tolerance=tolerance,
        min_pixels=min_pixels,
        save_png=False,
    )
    return sim_proj


def _require_name_parts(args: argparse.Namespace) -> None:
    missing = [
        name for name in ("category_name", "instance_name", "frame_idx")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "Missing output name fields: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate sim_proj-style binary NPY heatmaps from a proposal image."
    )
    parser.add_argument("--proposal-img", required=True, help="Path to clustered/proposal image.")
    parser.add_argument("--out-dir", required=True, help="Directory to save heatmap NPYs and manifest.")
    parser.add_argument(
        "--proposal-name",
        default="proposal",
        help="Name token used in output files: {category}_{instance}_{frame_idx}_{proposal}_{color}.npy.",
    )
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--min-pixels", type=int, default=1)
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument(
        "--vlm-response-json",
        default=None,
        help="Optional vlm_response JSON string or JSON file path. If set, only those colors are exported.",
    )
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--pipeline-info-id", type=int, default=None)
    parser.add_argument("--category-name", default=None)
    parser.add_argument("--instance-name", default=None)
    parser.add_argument("--frame-idx", default=None)

    args = parser.parse_args(argv)

    vlm_response = {}
    if args.db_path:
        vlm_response, row = load_vlm_response_from_db(
            args.db_path,
            pipeline_info_id=args.pipeline_info_id,
            category_name=args.category_name,
            instance_name=args.instance_name,
            frame_idx=args.frame_idx,
        )
        args.category_name = args.category_name or row["category_name"]
        args.instance_name = args.instance_name or row["instance_name"]
        args.frame_idx = args.frame_idx or row["frame_idx"]
    if args.vlm_response_json:
        vlm_response = load_vlm_response_json(args.vlm_response_json)

    _require_name_parts(args)
    sim_proj, manifest = write_sim_proj_from_proposal_rgb(
        args.proposal_img,
        args.out_dir,
        category_name=args.category_name,
        instance_name=args.instance_name,
        frame_idx=args.frame_idx,
        proposal_name=args.proposal_name,
        vlm_response=vlm_response,
        tolerance=args.tolerance,
        min_pixels=args.min_pixels,
        save_png=args.save_png,
    )

    print(f"Saved {len(sim_proj)} heatmap NPY(s) to {args.out_dir}")
    print(f"Manifest: {manifest['manifest_path']}")


# Backward-compatible private aliases used by older scripts.
_slugify = slugify
_normalize_color_name = normalize_color_name
_palette_by_normalized_name = palette_by_normalized_name
_heatmap_for_palette_rgb = heatmap_for_palette_rgb
