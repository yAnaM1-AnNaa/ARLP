import argparse
import os
from collections import defaultdict
from time import perf_counter
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from model.network import Conv2DFiLMNet
from utils.file_utils import SolveFolder
from utils.img_utils import transform_imgs, load_pretrained_dino, get_dino_features_from_transformed_imgs
from utils.vlm_utils import get_text_embedding_options
from utils.file_utils import load_config, save_image

class AffordanceInference:
    def __init__(self, config_path, checkpoint_path, text_embedding_func):
        # This will load 3 models: the affordance model; DINO; text embedding model.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = load_config(config_path)
        model_cfg = cfg['model']

        self.model = Conv2DFiLMNet(**model_cfg)
        self.model.build()
        self.model.to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        torch_home = cfg.get('torch_home', None)
        dino_model_type = cfg.get('dino_model_type')
        dino_use_registers = cfg.get("dino_use_registers", True)
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


def main(args):
    cfg = load_config(args.config)
    text_embedding_option = cfg.get("text_embedding", "embeddings_oai")
    text_embedding_func = get_text_embedding_options(text_embedding_option)

    inference = AffordanceInference(args.config, args.checkpoint, text_embedding_func)
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
    parser.add_argument("--img-dir", default='dataset/tests', help="Image directory, recursively processed")
    parser.add_argument("--config", default='configs/eval_agd.yaml', help="Path to config YAML file")
    parser.add_argument("--checkpoint", default='checkpoints/eval_agd.pth', help="Path to model checkpoint")
    parser.add_argument("--text-query", default="the area to sit on", help="Affordance text query")
    parser.add_argument("--thresh", type=float, default=0.3, help="Prediction threshold")
    parser.add_argument("--output-dir", default="runs/local_inference", help="Output directory")
    args = parser.parse_args()
    main(args)