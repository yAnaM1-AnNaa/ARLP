#!/usr/bin/env python3
"""Populate pipeline_info.text_embedding and pipeline_info.sim_proj.

For each Done pipeline_info row with vlm_response, this script:
1. reads the row's proposal/clustered image,
2. generates one proposal-mask sim_proj .npy per color in vlm_response,
3. computes one text embedding per description,
4. stores JSON in two new pipeline_info columns:
   - text_embedding: {response_color: [[...], ...]}
   - sim_proj: {response_color: "/abs/path/to/..._{color}.npy"}

The color keys in both JSON columns are exactly the keys from vlm_response.
"""

import argparse
import ast
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.img_utils import get_palette


def _slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "color"


def _normalize_color_name(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def _parse_description_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        parsed = parsed.strip()
        return [parsed] if parsed else []
    return [text]


def _load_rows(conn: sqlite3.Connection, limit: int | None):
    query = """
        SELECT id, category_name, instance_name, frame_idx, clustered_img_path, vlm_response
        FROM pipeline_info
        WHERE status = 'Done'
          AND vlm_response IS NOT NULL
          AND vlm_response != ''
        ORDER BY id ASC
    """
    if limit is not None:
        query += " LIMIT ?"
        return conn.execute(query, (limit,)).fetchall()
    return conn.execute(query).fetchall()


def _ensure_columns(conn: sqlite3.Connection):
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(pipeline_info)")}
    if "text_embedding" not in existing:
        conn.execute("ALTER TABLE pipeline_info ADD COLUMN text_embedding TEXT")
    if "sim_proj" not in existing:
        conn.execute("ALTER TABLE pipeline_info ADD COLUMN sim_proj TEXT")


def _resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def _palette_by_normalized_name() -> dict[str, tuple[np.ndarray, str]]:
    return {
        _normalize_color_name(name): (np.asarray(rgb, dtype=np.float32), name)
        for rgb, name in get_palette()
    }


def _heatmap_for_color(proposal_img: np.ndarray, rgb: np.ndarray, tolerance: float) -> np.ndarray:
    dist = np.linalg.norm(proposal_img.astype(np.float32) - rgb, axis=-1)
    return np.asarray(dist <= tolerance, dtype=np.float32)



def _collect_unique_descriptions(rows) -> list[str]:
    seen = set()
    descriptions = []
    for row in rows:
        vlm_response = json.loads(row["vlm_response"])
        for value in vlm_response.values():
            for description in _parse_description_list(value):
                if description not in seen:
                    seen.add(description)
                    descriptions.append(description)
    return descriptions


def _embed_texts_openai(texts: list[str], batch_size: int) -> dict[str, list[float]]:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.gptsapi.net/v1",
        api_key="sk-CWP2517500a761fd50613af3577f70d63bfa0fdd322Pef5Z",
        max_retries=5,
        timeout=60.0,
    )
    embedding_by_text = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large",
            dimensions=1024,
        )
        for text, item in zip(batch, response.data):
            embedding_by_text[text] = np.asarray(item.embedding, dtype=np.float32).tolist()
        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} unique text(s)", flush=True)
    return embedding_by_text


def _embed_texts_sentence_transformer(texts: list[str], batch_size: int) -> dict[str, list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return {
        text: np.asarray(embedding, dtype=np.float32).tolist()
        for text, embedding in zip(texts, embeddings)
    }


def _embed_texts(texts: list[str], embedding_type: str, batch_size: int) -> dict[str, list[float]]:
    if embedding_type == "embeddings_oai":
        return _embed_texts_openai(texts, batch_size)
    if embedding_type == "embeddings_st":
        return _embed_texts_sentence_transformer(texts, batch_size)
    raise ValueError(f"Unsupported embedding_type: {embedding_type}")


def _validate_rows(rows, repo_root: Path, tolerance: float, min_pixels: int):
    palette_by_name = _palette_by_normalized_name()
    errors = []

    for row in rows:
        row_id = row["id"]
        proposal_path = _resolve_path(row["clustered_img_path"], repo_root)
        if not proposal_path.exists():
            errors.append(f"id={row_id}: proposal image not found: {proposal_path}")
            continue

        try:
            vlm_response = json.loads(row["vlm_response"])
        except json.JSONDecodeError as exc:
            errors.append(f"id={row_id}: invalid vlm_response JSON: {exc}")
            continue

        if not isinstance(vlm_response, dict) or not vlm_response:
            errors.append(f"id={row_id}: vlm_response must be a non-empty object")
            continue

        try:
            proposal_img = np.asarray(Image.open(proposal_path).convert("RGB"))
        except Exception as exc:
            errors.append(f"id={row_id}: failed to load proposal image {proposal_path}: {exc}")
            continue

        for color, descriptions in vlm_response.items():
            normalized = _normalize_color_name(color)
            if normalized not in palette_by_name:
                valid = ", ".join(sorted(palette_by_name))
                errors.append(f"id={row_id}: color {color!r} is not in palette. Valid colors: {valid}")
                continue

            desc_list = _parse_description_list(descriptions)
            if not desc_list:
                errors.append(f"id={row_id}: color {color!r} has no descriptions")

            rgb, _ = palette_by_name[normalized]
            heatmap = _heatmap_for_color(proposal_img, rgb, tolerance)
            pixel_count = int(heatmap.sum())
            if pixel_count < min_pixels:
                errors.append(
                    f"id={row_id}: color {color!r} matched {pixel_count} pixel(s), min_pixels={min_pixels}"
                )

    return errors


def _process_row(row, repo_root: Path, out_dir: Path, embedding_by_text: dict,
                 tolerance: float, min_pixels: int):
    palette_by_name = _palette_by_normalized_name()
    category = str(row["category_name"])
    instance = str(row["instance_name"])
    frame_idx = str(row["frame_idx"])
    proposal_path = _resolve_path(row["clustered_img_path"], repo_root)
    proposal_img = np.asarray(Image.open(proposal_path).convert("RGB"))
    vlm_response = json.loads(row["vlm_response"])

    base_name = "_".join(_slugify(part) for part in (category, instance, frame_idx, "proposal"))
    sim_proj = {}
    text_embedding = {}

    for color, descriptions in vlm_response.items():
        normalized = _normalize_color_name(color)
        rgb, _ = palette_by_name[normalized]
        heatmap = _heatmap_for_color(proposal_img, rgb, tolerance)
        pixel_count = int(heatmap.sum())
        if pixel_count < min_pixels:
            raise ValueError(f"id={row['id']}: color {color!r} matched too few pixels: {pixel_count}")

        color_slug = _slugify(color)
        npy_path = out_dir / f"{base_name}_{color_slug}.npy"
        np.save(npy_path, heatmap.astype(np.float32))
        sim_proj[color] = str(npy_path.resolve())

        embeddings = []
        for description in _parse_description_list(descriptions):
            embeddings.append(embedding_by_text[description])
        text_embedding[color] = embeddings

    return text_embedding, sim_proj


def main():
    parser = argparse.ArgumentParser(
        description="Generate proposal-mask sim_proj npy files and store paths/text embeddings in pipeline_info."
    )
    parser.add_argument("--db-path", default="/root/autodl-tmp/ARLP/dataset/h5/db/pipeline_info.db")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/ARLP/dataset/h5/sim_proj")
    parser.add_argument("--embedding-type", default="embeddings_oai", choices=["embeddings_oai", "embeddings_st"])
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--min-pixels", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing.")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not create files or modify DB.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = _load_rows(conn, args.limit)
        print(f"Loaded {len(rows)} Done row(s) with vlm_response from {db_path}")

        errors = _validate_rows(rows, REPO_ROOT, args.tolerance, args.min_pixels)
        if errors:
            print("Validation failed; DB was not modified.", file=sys.stderr)
            for error in errors[:50]:
                print(error, file=sys.stderr)
            if len(errors) > 50:
                print(f"... {len(errors) - 50} more error(s)", file=sys.stderr)
            raise SystemExit(1)

        if args.dry_run:
            print("Dry run passed; DB was not modified.")
            return

        unique_descriptions = _collect_unique_descriptions(rows)
        print(f"Embedding {len(unique_descriptions)} unique text(s) with {args.embedding_type}", flush=True)
        embedding_by_text = _embed_texts(unique_descriptions, args.embedding_type, args.embedding_batch_size)
        updates = []

        for i, row in enumerate(rows, 1):
            text_embedding, sim_proj = _process_row(
                row,
                REPO_ROOT,
                out_dir,
                embedding_by_text,
                args.tolerance,
                args.min_pixels,
            )
            updates.append((
                json.dumps(text_embedding, ensure_ascii=False),
                json.dumps(sim_proj, ensure_ascii=False),
                row["id"],
            ))
            if i % 25 == 0 or i == len(rows):
                print(f"Prepared {i}/{len(rows)} row(s)")

        with conn:
            _ensure_columns(conn)
            conn.executemany(
                "UPDATE pipeline_info SET text_embedding = ?, sim_proj = ? WHERE id = ?",
                updates,
            )

        print(f"Updated {len(updates)} row(s).")
        print(f"sim_proj npy directory: {out_dir.resolve()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
