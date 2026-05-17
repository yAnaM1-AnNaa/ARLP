# ARLP Run Guide

This README is only a runbook. It describes the files to prepare, the configs
to edit, and the commands to run evaluation, split evaluation, local inference,
EarlyFiLM training, and sim-proj generation.

## 1. Environment

Use Python 3.10+ with CUDA PyTorch. From the repository root:

```bash
cd ARLP
pip install -r requirements.txt
```

The project expects these Python packages at runtime:

```text
torch, torchvision, timm, transformers, safetensors, h5py, pillow, numpy,
matplotlib, opencv-python, openai, open3d, tqdm, pyyaml
```

If `requirements.txt` does not install CUDA PyTorch for your machine, install
the correct PyTorch build first, then install the remaining requirements.

## 2. Required Files

Large model and dataset artifacts are not committed. Prepare them locally and
point the YAML configs to their paths.

Recommended layout:

```text
ARLP/
  grounding-dino-base/
    config.json
    preprocessor_config.json
    model.safetensors or pytorch_model.bin
    tokenizer files...
  checkpoints/
    eval_agd.pth
    early_film/best (1).pth
  steervit_dinov2_base.pth
  dataset/pano_eazy/dataset6/Seen/testset/
    egocentric/
    GT/
```

SteerViT also needs a local DINOv2-B Hugging Face `model.safetensors`. Set this
path in `configs/steervit_eval.yaml`:

```yaml
local_inferer:
  model:
    vision_checkpoint_path: "/path/to/facebook--dinov2-base/.../model.safetensors"
```

EarlyFiLM and SteerViT use `roberta-large`. Make sure it is available through
Hugging Face cache or network access.

## 3. Configs

Evaluation uses one detector and one local inferer. The local inferer is selected
by `local_inferer.name`.

Available local inferers:

```yaml
local_inferer:
  name: "latefilm"    # Conv2DFiLMNet over frozen DINO features
```

```yaml
local_inferer:
  name: "earlyfilm"   # Early FiLM inside DINOv2 ViT
```

```yaml
local_inferer:
  name: "steervit"    # Gated cross-attention inside DINOv2 ViT
```

Useful configs:

```text
configs/eval_agd.yaml                    # default evaluation config
configs/eval_agd.example.yaml            # schema examples for all inferers
configs/early_film.yaml                  # EarlyFiLM eval config
configs/early_film_layers_21_22_23.yaml  # EarlyFiLM train/eval config for last 3 ViT-L blocks
configs/steervit_eval.yaml               # SteerViT eval config
configs/agd_prompts.yaml                 # action/object prompts and detector classes
```

For EarlyFiLM, FiLM layers are controlled by:

```yaml
model:
  film_layers: [21, 22, 23]
```

For SteerViT, cross-attention layers are controlled by:

```yaml
model:
  cross_attn_layers: [1, 3, 5, 7, 9, 11]
```

## 4. Full Evaluation

Run SteerViT on a pano-eazy test split:

```bash
python eval_agd.py \
  --config configs/steervit_eval.yaml \
  --agd_root dataset/pano_eazy/dataset6/Seen/testset \
  --viz_dir runs/steervit_seen_test
```

Run EarlyFiLM:

```bash
python eval_agd.py \
  --config configs/early_film.yaml \
  --agd_root dataset/pano_eazy/dataset6/Seen/testset \
  --viz_dir runs/earlyfilm_seen_test
```

Save predicted heatmaps as `.npy` files:

```bash
python eval_agd.py \
  --config configs/steervit_eval.yaml \
  --agd_root dataset/pano_eazy/dataset6/Seen/testset \
  --viz_dir runs/steervit_seen_test \
  --save_npy
```

Outputs:

```text
runs/<name>/*.png
runs/<name>/*.npy                 # only with --save_npy
runs/<name>/missing_files.txt     # if dataset image/GT files are missing
```

The script prints:

```text
KL, SIM, NSS, NSS_05
Evaluated: <valid>/<candidate>
No detections: <count>/<valid>
```

## 5. Seen/Unseen Action Splits

`eval_agd.py` supports action filters:

```text
--include_actions ACTION...
--exclude_actions ACTION...
```

Action names accept spaces, hyphens, or underscores. They are normalized to
directory names such as `lean_back`, `rest_arm`, and `swing_open`.

Example seen split:

```bash
python eval_agd.py \
  --config configs/steervit_eval.yaml \
  --agd_root dataset/pano_eazy/dataset6/Seen/testset \
  --viz_dir runs/steervit_split_seen \
  --include_actions heating lean_back open pull refrigerate rest_arm sit swing_open
```

Example unseen split using the complement:

```bash
python eval_agd.py \
  --config configs/steervit_eval.yaml \
  --agd_root dataset/pano_eazy/dataset6/Seen/testset \
  --viz_dir runs/steervit_split_unseen \
  --exclude_actions heating lean_back open pull refrigerate rest_arm sit swing_open
```

## 6. Local Image/Folder Inference

Run a local affordance model over an image directory:

```bash
python local_inference.py \
  --config configs/steervit_eval.yaml \
  --checkpoint steervit_dinov2_base.pth \
  --img-dir /path/to/images \
  --text-query "people can sit on this part of chair and relax." \
  --output-dir runs/local_steervit
```

For EarlyFiLM, use an EarlyFiLM config and checkpoint:

```bash
python local_inference.py \
  --config configs/early_film.yaml \
  --checkpoint "checkpoints/early_film/best (1).pth" \
  --img-dir /path/to/images \
  --text-query "people can sit on this part of chair and relax." \
  --output-dir runs/local_earlyfilm
```

## 7. Train EarlyFiLM

`train_earlyfilm.py` expects one or more serialized `RegionSimDataset` `.pt`
files. It does not train directly from a raw image directory.

Run last-three-layer EarlyFiLM training:

```bash
python train_earlyfilm.py \
  --config configs/early_film_layers_21_22_23.yaml \
  --data /path/to/region_sim_dataset.pt \
  --run-name earlyfilm_layers_21_22_23 \
  --output-dir logs/earlyfilm \
  --no_wandb
```

Resume from a checkpoint:

```bash
python train_earlyfilm.py \
  --config configs/early_film_layers_21_22_23.yaml \
  --data /path/to/region_sim_dataset.pt \
  --resume-ckpt logs/earlyfilm/<date>/<run-name>/ckpts/best.pth \
  --run-name earlyfilm_resume \
  --no_wandb
```

Checkpoints are written under:

```text
logs/earlyfilm/<date>/<run-name>/ckpts/
```

After training, update:

```yaml
local_inferer:
  checkpoint_path: "logs/earlyfilm/<date>/<run-name>/ckpts/best.pth"
```

## 8. Generate sim_proj From Proposal RGB

The sim-proj utility converts a palette-colored proposal/cluster RGB image into
binary heatmap `.npy` files. It is not intended for unannotated natural RGB
images.

CLI:

```bash
python scripts/proposal_to_sim_proj.py \
  --proposal-img /path/to/proposal.png \
  --out-dir runs/sim_proj \
  --category-name chair \
  --instance-name chair_001 \
  --frame-idx 0 \
  --vlm-response-json '{"Red": ["handle"], "Green": ["seat"]}' \
  --save-png
```

Python API:

```python
from utils.sim_proj_utils import write_sim_proj_from_proposal_rgb

sim_proj, manifest = write_sim_proj_from_proposal_rgb(
    "/path/to/proposal.png",
    "runs/sim_proj",
    category_name="chair",
    instance_name="chair_001",
    frame_idx=0,
    vlm_response={"Red": ["handle"], "Green": ["seat"]},
)
```

The returned `sim_proj` dict maps each color name to the generated `.npy` path.

## 9. Common Checks

Verify imports and syntax:

```bash
python -m py_compile \
  eval_agd.py \
  local_inference.py \
  model/early_film.py \
  model/steervit.py \
  utils/sim_proj_utils.py
```

Check that a config builds the requested local inferer:

```bash
python - <<'PY'
import numpy as np
from local_inference import build_affordance_inference

inferer = build_affordance_inference("configs/steervit_eval.yaml")
img = np.full((128, 160, 3), 255, dtype=np.uint8)
out = inferer.predict(img, "people can sit on this part of chair and relax.", thresh=None)
print(type(inferer).__name__, out.shape, out.dtype, float(out.min()), float(out.max()))
PY
```
