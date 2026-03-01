# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARLP (Affordance Recognition with Large-scale Panoramic learning) detects affordance regions on objects — functional areas that enable specific interactions (e.g., "the handle to grip", "the seat to sit on"). It extends the upstream UAD (Unsupported Affordance Detection) project to handle panoramic/360-degree environments and integrates Chain-of-Thought reasoning from GREAT.

The project is written in Chinese-commented Python. README and inline comments are predominantly in Chinese.

## Commands

### Data Curation Pipeline
```bash
# Full pipeline (current version): process H5 datasets into training data
python pipelinev3.py --base_dir <data_dir> --embedding_type embeddings_oai [--category_names chair table] [--top_k 3]

# Batch Chain-of-Thought affordance descriptions via VLM
python CoT.py --img_dir <clustered_imgs> --output_dir <json_out> --object_name <category> [--model_name qwen/qwen3-vl-30b-a3b-thinking]
```

### Training
```bash
python train.py --config configs/oai_emb.yaml --data <dataset1.pt> [<dataset2.pt> ...] --run_name <name> [--epochs N] [--batch N] [--lr F] [--resume_ckpt <path>] [--no_wandb]
```

### Inference
```bash
python inference.py --config configs/oai_emb.yaml --checkpoint checkpoints/oai_emb.pth
```

### Evaluation
```bash
# AGD20K benchmark
python eval_agd.py --config configs/eval_agd.yaml --checkpoint <path> --agd_root <agd20k_dir> [--viz_dir <output>]

# Panoramic sliding-window evaluation
python eval_pano.py --config <config.yaml> --checkpoint <path> --agd_root <data_dir> [--window_size 518] [--viz_dir <output>]
```

### Utilities
```bash
# Inspect HDF5 file contents
python scripts/print_hdf5.py <file.h5>

# Validate CoT JSON output
python scripts/test_cot_output.py <output.json>

# Interactive HDF5 editor
python src/h5_editor.py <file.h5>
```

## Architecture

### Data Flow

```
H5 files (Behavior1k multi-view RGB-D)
  → pipelinev3.py: CLIP best-angle selection, 3D fusion, DINOv2 clustering, MHACoT VLM prompting, text embedding
    → .pt dataset files (RegionSimDataset)
      → train.py: trains Conv2DFiLMNet
        → inference.py / eval_agd.py / eval_pano.py
```

### Model: Conv2DFiLMNet (`model/network.py`)

FiLM-conditioned (Feature-wise Linear Modulation) 2D CNN that takes DINOv2 visual features (384-dim ViT-S or 1024-dim ViT-L) and text embeddings, producing single-channel affordance heatmaps. FiLM layers modulate visual features based on language input. Loss: BCEWithLogits.

### MHACoT: Multi-Hop Affordance Chain-of-Thought (`src/MHACoT.py`)

4-step VLM prompting via OpenRouter API:
1. Identify interactable colored regions
2. Geometric shape reasoning
3. Affordance description per part
4. Structured JSON output mapping colors → affordance keywords

### Dataset: RegionSimDataset (`model/dataset.py`)

Loads `.pt` files containing per-instance samples: text description, text embedding, similarity projection map (target heatmap), and RGB image. Supports train/val splitting with cross-dataset weighted sampling.

### Key Modules

- **`src/fusion.py`** — Multi-view RGB-D to 3D point cloud with per-point DINOv2 features via camera projection
- **`src/cluster.py`** — K-means/MeanShift clustering on PCA-reduced features; generates color-coded proposal images per cluster
- **`utils/img_utils.py`** — DINOv2 feature extraction, image transforms, visualization
- **`utils/vlm_utils.py`** — Text embedding (OpenAI text-embedding-3-large 1024-dim or SentenceTransformer all-MiniLM-L6-v2 384-dim), color detection
- **`utils/eval_utils.py`** — Evaluation metrics: KL divergence, SIM (intersection similarity), NSS (Normalized Scanpath Saliency)
- **`utils/file_utils.py`** — YAML config loading, HDF5 read/write helpers

### Configuration (`configs/`)

YAML files with sections: `model` (architecture: filters, kernels, FiLM mode, embedding dims), `optim` (lr, weight decay, scheduler), `trainer` (batch size, epochs, val split), `text_embedding` (oai vs st).

Key config variants:
- `oai_emb.yaml` / `st_emb.yaml` — OpenAI vs SentenceTransformer embeddings with ViT-S
- `oai_emb_vitl.yaml` / `st_emb_vitl.yaml` — ViT-L backbone variants
- `eval_agd.yaml` — AGD20K evaluation config

### HDF5 Dataset Structure

Each instance stores: multi-view `rgb`, `depth`, camera `intrinsics`/`extrinsics`, `link_segs`, `clip_similarities`, `top_k_indices`/`top_k_rgb`, `color_label_names`, `color_name_features`, `similarity_projections`, `region_matching` (VLM JSON), and `embeddings_oai`/`embeddings_st` keyed by color.

## Dependencies

PyTorch, Transformers (CLIP, DINOv2), h5py, Open3D, scikit-learn, OpenAI API, SentenceTransformers, WandB, matplotlib, PIL. VLM calls use OpenRouter API.
