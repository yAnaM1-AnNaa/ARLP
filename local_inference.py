import argparse
import os
from collections import defaultdict
from time import perf_counter
from typing import Optional, Sequence
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from model.network import Conv2DFiLMNet
from model.early_film import EarlyFiLMDINOv2
from utils.file_utils import SolveFolder
from utils.img_utils import transform_imgs, load_pretrained_dino, get_dino_features_from_transformed_imgs
from utils.vlm_utils import get_text_embedding_options
from utils.file_utils import load_config, save_image


def _load_checkpoint_state(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    trainable_only = False
    if isinstance(ckpt, dict):
        trainable_only = bool(ckpt.get("trainable_only", False))
        for key in ("model", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

    if isinstance(ckpt, dict) and ckpt and all(k.startswith("module.") for k in ckpt):
        ckpt = {k.removeprefix("module."): v for k, v in ckpt.items()}
    return ckpt, trainable_only


########################################
##### LateFiLM local inference
########################################

class AffordanceInference:
    def __init__(self, config_path, checkpoint_path=None, text_embedding_func=None):
        # This will load 3 models: the affordance model; DINO; text embedding model.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = load_config(config_path)
        if 'local_inferer' in cfg:
            local_cfg = cfg['local_inferer']
            if local_cfg.get('name', '').lower() != 'latefilm':
                raise ValueError(
                    "This AffordanceInference implementation currently supports "
                    "local_inferer.name='latefilm'."
                )
            model_cfg = local_cfg['model']
            checkpoint_path = checkpoint_path or local_cfg['checkpoint_path']
            dino_cfg = local_cfg.get('dino', {})
            torch_home = dino_cfg.get('torch_home', None)
            dino_model_type = dino_cfg.get('model_type')
            dino_use_registers = dino_cfg.get('use_registers', True)
        else:
            model_cfg = cfg['model']
            torch_home = cfg.get('torch_home', None)
            dino_model_type = cfg.get('dino_model_type')
            dino_use_registers = cfg.get("dino_use_registers", True)

        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided for local inference.")
        if text_embedding_func is None:
            text_embedding_func = get_text_embedding_options("embeddings_oai")

        self.model = Conv2DFiLMNet(**model_cfg)
        self.model.build()
        self.model.to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        self.dino = load_pretrained_dino(dino_model_type, use_registers=dino_use_registers, torch_path=torch_home).to(self.device).eval()

        self.text_embedding_func = text_embedding_func
        self._text_embedding_cache = {}

    def _get_text_embedding_tensor(self, text: str) -> torch.Tensor:
        if text not in self._text_embedding_cache:
            emb = torch.from_numpy(self.text_embedding_func(text)).to(self.device).unsqueeze(0).to(torch.float32)
            self._text_embedding_cache[text] = emb
        return self._text_embedding_cache[text]

    def clear_text_embedding_cache(self):
        self._text_embedding_cache.clear()

    @staticmethod
    def _resize_pred_to_image(pred_map: np.array, out_hw):
        # pred_map is the output of the model, which is 1/8 of the original image size. 
        out_h, out_w = out_hw
        return np.array(
            T.functional.resize(
                Image.fromarray(pred_map.astype(np.float32), mode="F"),
                (out_h, out_w),
                interpolation=T.InterpolationMode.BILINEAR))
    
    @torch.no_grad()
    def predict(self, img_np, text, thresh=0.5):
        # The main function to get the affordance map.
        proc = transform_imgs(img_np, blur=False)[0] # (3, H, W)
        proc = proc.unsqueeze(0).to(self.device) # (1, 3, H, W)
        lang_emb = self._get_text_embedding_tensor(text).to(self.device) # (1, D)

        feature = get_dino_features_from_transformed_imgs(self.dino, proc,
                                                       repeat_to_orig_size=False) 
        # feature=(1, H', W', C), H'=H/14(patch size), C=feature dim
        feature = feature.permute(0, 3, 1, 2) # (1, C, H', W')

        logits = self.model(feature, lang_emb).squeeze(0).squeeze(0) # (H', W')
        sim = torch.sigmoid(logits) # map the logits to [0, 1], which is the similarity map.
        if thresh is not None:
            sim = (sim > thresh).float() # if thresh is not None, binarize the similarity map.
        sim_np = sim.cpu().numpy()
        H, W = img_np.shape[:2]
        sim_np = self._resize_pred_to_image(sim_np, (H, W))
        return sim_np
    
    @torch.no_grad()
    def predict_batch(self, img_np_list, text, thresh=None):
        # A batch version of predict.
        transformed = []
        orig_sizes = []

        for img_np in img_np_list:
            proc = transform_imgs(img_np, blur=False)[0]
            transformed.append(proc)
            orig_sizes.append(img_np.shape[:2])

        # Put images of the same size into a batch, and record their original sizes for later resizing.
        # {(3, 480, 640): [(0, proc0), (2, proc2)], (512, 512): [(1, proc1)], ...}
        buckets = defaultdict(list) # key: (H, W), value: list of (index, transformed_img)
        for idx, proc in enumerate(transformed):
            buckets[tuple(proc.shape)].append((idx, proc))

        outputs = [None] * len(img_np_list) 
        base_lang_emb = self._get_text_embedding_tensor(text).to(self.device)

        for _, items in buckets.items():
            indices = [idx for idx, _ in items]
            batch_proc = torch.stack([proc for _, proc in items], dim=0).to(self.device)

            feature = get_dino_features_from_transformed_imgs(
                  self.dino,
                  batch_proc,
                  repeat_to_orig_size=False,
              )
            feature = feature.permute(0, 3, 1, 2)  # (B, C, H', W')

            batch_lang_emb = base_lang_emb.repeat(len(items), 1)
            logits = self.model(feature, batch_lang_emb).squeeze(1)  # (B, h, w)
            sim = torch.sigmoid(logits)
            if thresh is not None:
                sim = (sim > thresh).float()

            sim_np = sim.cpu().numpy()

            for local_i, global_idx in enumerate(indices):
                H, W = orig_sizes[global_idx]
                pred_resized = self._resize_pred_to_image(sim_np[local_i], (H, W))
                outputs[global_idx] = pred_resized
        return outputs


########################################
##### EarlyFiLM local inference
########################################

class EarlyFiLMAffordanceInference:
    def __init__(self, config_path, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = load_config(config_path)
        local_cfg = cfg.get('local_inferer', {})
        if local_cfg.get('name', '').lower() != 'earlyfilm':
            raise ValueError(
                "EarlyFiLMAffordanceInference requires local_inferer.name='earlyfilm'."
            )

        model_cfg = local_cfg.get('model', {})
        checkpoint_path = checkpoint_path or local_cfg.get('checkpoint_path')
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided for EarlyFiLM inference.")

        self.batch_size = int(local_cfg.get("batch_size", 1))
        self.model = EarlyFiLMDINOv2(**model_cfg).to(self.device)
        state, trainable_only = _load_checkpoint_state(checkpoint_path, self.device)
        strict = local_cfg.get("load_strict", not trainable_only)
        result = self.model.load_state_dict(state, strict=strict)
        if not strict:
            print(
                "Loaded EarlyFiLM checkpoint with strict=False: "
                f"{len(result.missing_keys)} missing keys, "
                f"{len(result.unexpected_keys)} unexpected keys."
            )
        self.model.eval()

        self.transform = self.model.get_transforms()

    @staticmethod
    def _resize_pred_to_image(pred_map: np.array, out_hw):
        out_h, out_w = out_hw
        return np.array(
            T.functional.resize(
                Image.fromarray(pred_map.astype(np.float32), mode="F"),
                (out_h, out_w),
                interpolation=T.InterpolationMode.BILINEAR))

    def _transform_image(self, img_np):
        temp = img_np.copy()
        if temp.max() <= 1.1:
            temp = temp * 255
        temp = temp.astype(np.uint8).clip(0, 255)
        return self.transform(Image.fromarray(temp).convert("RGB"))

    @torch.no_grad()
    def predict(self, img_np, text, thresh=0.5):
        proc = self._transform_image(img_np).unsqueeze(0).to(self.device)
        logits = self.model.get_heatmap_logits(
            proc,
            texts=[text],
            interpolate=False,
        ).squeeze(0).squeeze(0)

        sim = torch.sigmoid(logits)
        if thresh is not None:
            sim = (sim > thresh).float()

        sim_np = sim.cpu().numpy()
        H, W = img_np.shape[:2]
        return self._resize_pred_to_image(sim_np, (H, W))

    @torch.no_grad()
    def predict_batch(self, img_np_list, text, thresh=None):
        outputs = []
        if not img_np_list:
            return outputs

        transformed = []
        orig_sizes = []
        for img_np in img_np_list:
            transformed.append(self._transform_image(img_np))
            orig_sizes.append(img_np.shape[:2])

        for start in range(0, len(transformed), self.batch_size):
            end = start + self.batch_size
            batch_proc = torch.stack(transformed[start:end], dim=0).to(self.device)
            texts = [text] * len(transformed[start:end])
            logits = self.model.get_heatmap_logits(
                batch_proc,
                texts=texts,
                interpolate=False,
            ).squeeze(1)

            sim = torch.sigmoid(logits)
            if thresh is not None:
                sim = (sim > thresh).float()

            sim_np = sim.cpu().numpy()
            for pred, (H, W) in zip(sim_np, orig_sizes[start:end]):
                outputs.append(self._resize_pred_to_image(pred, (H, W)))

            del batch_proc, logits, sim
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return outputs


########################################
##### Detection-based local inference
########################################

class DetectionLocalInference:
    def __init__(
        self,
        affordance_inference: AffordanceInference,
        box_expand_ratio: float = 0.0,
        local_heatmap_thresh: Optional[float] = None,
        final_heatmap_thresh: Optional[float] = None,
        overlap_mode: str = "mean",
    ):
        self.affordance_inference = affordance_inference
        self.box_expand_ratio = box_expand_ratio
        self.local_heatmap_thresh = local_heatmap_thresh
        self.final_heatmap_thresh = final_heatmap_thresh
        self.overlap_mode = overlap_mode

    @staticmethod
    def expand_and_clip_box(xyxy: Sequence[int], width: int, height: int, expand_ratio: float):
        x1, y1, x2, y2 = xyxy
        box_w = x2 - x1
        box_h = y2 - y1

        pad_w = box_w * expand_ratio / 2.0
        pad_h = box_h * expand_ratio / 2.0

        x1 = max(0, int(np.floor(x1 - pad_w)))
        y1 = max(0, int(np.floor(y1 - pad_h)))
        x2 = min(width, int(np.ceil(x2 + pad_w)))
        y2 = min(height, int(np.ceil(y2 + pad_h)))
        return x1, y1, x2, y2

    def predict(self, image_rgb: np.ndarray, detections: Sequence, affordance_text: str) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        heatmap_sum = np.zeros((height, width), dtype=np.float32)
        heatmap_count = np.zeros((height, width), dtype=np.float32)
        heatmap_max = np.zeros((height, width), dtype=np.float32)
        crop_records = []
        crop_images = []

        for det in detections:
            x1, y1, x2, y2 = self.expand_and_clip_box(
                det.box_xyxy,
                width,
                height,
                self.box_expand_ratio,
            )
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_records.append((x1, y1, x2, y2, crop.shape[:2]))
            crop_images.append(crop)

        if crop_images:
            local_heatmaps = self.affordance_inference.predict_batch(
                crop_images,
                affordance_text,
                thresh=self.local_heatmap_thresh,
            )
        else:
            local_heatmaps = []

        for local_heatmap, (x1, y1, x2, y2, crop_hw) in zip(local_heatmaps, crop_records):
            crop_h, crop_w = crop_hw
            if local_heatmap.shape != (crop_h, crop_w):
                local_heatmap = np.array(
                    Image.fromarray(local_heatmap.astype(np.float32), mode="F").resize(
                        (crop_w, crop_h),
                        Image.BILINEAR,
                    )
                )

            if self.overlap_mode == "max":
                heatmap_max[y1:y2, x1:x2] = np.maximum(
                    heatmap_max[y1:y2, x1:x2],
                    local_heatmap,
                )
            else:
                heatmap_sum[y1:y2, x1:x2] += local_heatmap
            heatmap_count[y1:y2, x1:x2] += 1.0

        if self.overlap_mode == "mean":
            heatmap = np.divide(
                heatmap_sum,
                np.maximum(heatmap_count, 1e-6),
                out=np.zeros_like(heatmap_sum),
                where=heatmap_count > 0,
            )
        elif self.overlap_mode == "sum":
            heatmap = heatmap_sum
        elif self.overlap_mode == "max":
            heatmap = heatmap_max
        else:
            raise ValueError(f"Unsupported overlap_mode: {self.overlap_mode}")

        if self.final_heatmap_thresh is not None:
            heatmap = (heatmap > self.final_heatmap_thresh).astype(np.float32)
        return heatmap


########################################
##### Local inference builder
########################################

def build_affordance_inference(config_path: str, checkpoint_path: Optional[str] = None):
    cfg = load_config(config_path)
    local_cfg = cfg.get("local_inferer", {})
    local_name = local_cfg.get("name", "latefilm").lower()
    checkpoint_path = checkpoint_path or local_cfg.get("checkpoint_path")

    if local_name == "latefilm":
        text_embedding_option = local_cfg.get("text_embedding", {}).get(
            "name",
            cfg.get("text_embedding", "embeddings_oai"),
        )
        text_embedding_func = get_text_embedding_options(text_embedding_option)
        return AffordanceInference(config_path, checkpoint_path, text_embedding_func)

    if local_name == "earlyfilm":
        return EarlyFiLMAffordanceInference(config_path, checkpoint_path)

    raise ValueError(f"Unsupported local_inferer.name: {local_name}")


def build_detection_local_inference(config_path: str) -> DetectionLocalInference:
    cfg = load_config(config_path)
    affordance_inference = build_affordance_inference(config_path)

    fusion_cfg = cfg.get("pano_fusion", {})
    return DetectionLocalInference(
        affordance_inference=affordance_inference,
        box_expand_ratio=fusion_cfg.get("box_expand_ratio", 0.0),
        local_heatmap_thresh=fusion_cfg.get("local_heatmap_thresh", None),
        final_heatmap_thresh=fusion_cfg.get("final_heatmap_thresh", None),
        overlap_mode=fusion_cfg.get("overlap_mode", "mean"),
    )


########################################
##### CLI runner
########################################

def main(args):
    cfg = load_config(args.config)
    inference = build_affordance_inference(args.config, args.checkpoint)
    shutdown_thresh = cfg.get("thresh", args.thresh)
    image_paths = SolveFolder.list_image_paths_recursive(args.img_dir, IMAGE_EXTENSIONS)
    print(f"Found {len(image_paths)} image(s).")

    for idx, image_path in enumerate(image_paths, start=1):
        start_time = perf_counter()

        img = np.array(Image.open(image_path).convert("RGB"))
        result = inference.predict(img, args.text_query, shutdown_thresh)

        saved = SolveFolder.save_local_result(
            image_path=image_path,
            img=img,
            heatmap=result,
            text_query=args.text_query,
            output_dir=args.output_dir,
        )

        elapsed = perf_counter() - start_time

        print(f"[{idx}/{len(image_paths)}] {image_path}")
        print(f"  result shape: {result.shape}")
        print(f"  elapsed: {elapsed:.4f}s")
        for key, value in saved.items():
            print(f"  saved {key}: {value}")


if __name__ == "__main__":
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--img-dir", default='/root/autodl-tmp/yolo/dataset', help="Image directory, recursively processed")
    parser.add_argument("--config", default='configs/oai_vitl_cot.yaml', help="Path to config YAML file")
    parser.add_argument("--checkpoint", default='logs/20260413/oai_vitl_cot/ckpts/best.pth', help="Path to model checkpoint")
    parser.add_argument("--text-query", default="people can sit on this part of chair and relax. seats often locates at the middle area of chairs.", help="Affordance text query")
    parser.add_argument("--thresh", type=float, default=0.3, help="Prediction threshold")
    parser.add_argument("--output-dir", default="runs/local_inference", help="Output directory")
    args = parser.parse_args()
    main(args)
