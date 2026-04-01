#!/usr/bin/env python3
"""
Create DINOv3 patch-feature PCA visualizations using Hugging Face weights.

This script avoids the official torch.hub dependency and works with local
Hugging Face model directories or public repo IDs supported by Transformers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoConfig, AutoModel


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DINOv3 patch features with PCA using HF weights."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        required=True,
        help="Local HF model directory or model id, e.g. facebook/dinov3-vitl16-pretrain-lvd1689m.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where visualization images will be saved.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Hidden-state layer index for visualization. -1 uses the final layer.",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=896,
        help="Resize the image so the long side matches this value before patch extraction.",
    )
    parser.add_argument(
        "--mask-image",
        default=None,
        help="Optional binary foreground mask image. Non-zero pixels are kept for PCA fitting.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["none", "alpha", "white"],
        default="none",
        help="Auto-generate a foreground mask from alpha or white background.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=245,
        help="Threshold for white-background masking when --mask-mode white is used.",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower percentile used for robust PCA color normalization.",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper percentile used for robust PCA color normalization.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Opacity of the PCA overlay on the resized input image.",
    )
    return parser.parse_args()


def load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    return image


def get_patch_size(config: AutoConfig) -> int:
    patch_size = getattr(config, "patch_size", None)
    if patch_size is None:
        raise ValueError("Could not infer patch_size from model config.")
    return int(patch_size)


def resize_to_multiple(
    image: Image.Image,
    long_side: int,
    patch_size: int,
) -> Image.Image:
    width, height = image.size
    scale = long_side / max(width, height)
    new_width = max(patch_size, int(round(width * scale)))
    new_height = max(patch_size, int(round(height * scale)))
    new_width = max(patch_size, (new_width // patch_size) * patch_size)
    new_height = max(patch_size, (new_height // patch_size) * patch_size)
    if new_width == width and new_height == height:
        return image
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def pil_to_tensor_rgb(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=tensor.device, dtype=tensor.dtype)
    std = IMAGENET_STD.to(device=tensor.device, dtype=tensor.dtype)
    return (tensor - mean) / std


def build_patch_mask(
    resized_image: Image.Image,
    grid_h: int,
    grid_w: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.mask_image:
        mask_image = Image.open(args.mask_image).convert("L").resize(
            (grid_w, grid_h), resample=Image.NEAREST
        )
        return np.asarray(mask_image, dtype=np.uint8) > 0

    if args.mask_mode == "alpha" and "A" in resized_image.getbands():
        alpha = resized_image.getchannel("A").resize((grid_w, grid_h), resample=Image.NEAREST)
        return np.asarray(alpha, dtype=np.uint8) > 0

    if args.mask_mode == "white":
        rgb_small = resized_image.convert("RGB").resize((grid_w, grid_h), resample=Image.BILINEAR)
        rgb_small = np.asarray(rgb_small, dtype=np.uint8)
        return np.any(rgb_small < args.white_threshold, axis=-1)

    return np.ones((grid_h, grid_w), dtype=bool)


def get_patch_tokens(
    model: AutoModel,
    pixel_values: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    selected = hidden_states[layer_idx]
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    patch_tokens = selected[:, 1 + num_register_tokens :, :]
    return patch_tokens


def fit_pca_map(
    patch_features: np.ndarray,
    patch_mask: np.ndarray,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    h, w, c = patch_features.shape
    flat_features = patch_features.reshape(-1, c)
    flat_mask = patch_mask.reshape(-1)
    if flat_mask.sum() < 3:
        raise ValueError("Foreground mask kept fewer than 3 patches; PCA is ill-defined.")

    pca = PCA(n_components=3)
    pca.fit(flat_features[flat_mask])
    projected = pca.transform(flat_features).reshape(h, w, 3)

    normalized = np.zeros_like(projected, dtype=np.float32)
    for channel in range(3):
        source = projected[..., channel][patch_mask]
        low = np.percentile(source, percentile_low)
        high = np.percentile(source, percentile_high)
        if high <= low:
            high = low + 1e-6
        channel_data = np.clip(projected[..., channel], low, high)
        normalized[..., channel] = (channel_data - low) / (high - low)

    return normalized


def upsample_map(rgb_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(rgb_map).permute(2, 0, 1).unsqueeze(0)
    upsampled = F.interpolate(
        tensor,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.squeeze(0).permute(1, 2, 0).numpy()


def save_outputs(
    image_resized: Image.Image,
    patch_mask: np.ndarray,
    pca_map: np.ndarray,
    output_dir: Path,
    overlay_alpha: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_np = np.asarray(image_resized.convert("RGB"), dtype=np.float32) / 255.0
    pca_full = upsample_map(pca_map, image_np.shape[:2])
    mask_full = upsample_map(patch_mask[..., None].astype(np.float32), image_np.shape[:2])[..., 0]
    mask_full = np.clip(mask_full, 0.0, 1.0)

    overlay = image_np * (1.0 - overlay_alpha * mask_full[..., None]) + pca_full * (
        overlay_alpha * mask_full[..., None]
    )
    overlay = np.clip(overlay, 0.0, 1.0)

    plt.imsave(output_dir / "pca_map.png", np.clip(pca_full, 0.0, 1.0))
    plt.imsave(output_dir / "overlay.png", overlay)
    plt.imsave(output_dir / "mask.png", mask_full, cmap="gray")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[1].imshow(mask_full, cmap="gray")
    axes[1].set_title("Foreground mask")
    axes[2].imshow(np.clip(pca_full, 0.0, 1.0))
    axes[2].set_title("Patch PCA")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    patch_size = get_patch_size(config)

    image = load_image(image_path)
    resized_image = resize_to_multiple(image, args.long_side, patch_size)
    pixel_values = normalize_tensor(pil_to_tensor_rgb(resized_image)).to(device)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=False).to(device).eval()
    patch_tokens = get_patch_tokens(model, pixel_values, args.layer)

    _, _, resized_h, resized_w = pixel_values.shape
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    hidden_dim = patch_tokens.shape[-1]
    patch_features = patch_tokens.reshape(1, grid_h, grid_w, hidden_dim)[0].cpu().numpy()

    patch_mask = build_patch_mask(resized_image, grid_h, grid_w, args)
    pca_map = fit_pca_map(
        patch_features=patch_features,
        patch_mask=patch_mask,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )
    save_outputs(
        image_resized=resized_image,
        patch_mask=patch_mask,
        pca_map=pca_map,
        output_dir=output_dir,
        overlay_alpha=args.overlay_alpha,
    )

    print(f"Saved PCA visualizations to: {output_dir}")
    print(f"Resized image size: {resized_w}x{resized_h}")
    print(f"Patch grid size: {grid_w}x{grid_h}")
    print(f"Layer used: {args.layer}")


if __name__ == "__main__":
    main()
