from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from model.film_model import Conv2DFiLMNet
from utils.img_utils import transform_imgs, load_pretrained_dino, get_dino_features_from_transformed_imgs
from utils.file_utils import load_config


class LateFiLMAffordanceInference:
    def __init__(self, cfg):
        ''' Late FiLM model uses original features from DINOv2, with 3 FiLM layers, which is later than SteerViT.
        This will load 3 models: the affordance model(LateFiLM); DINO; text embedding model.
        checkpoint_path: path to late film ckpt.'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_structure = cfg['model_structure'] # Note that late film and early film configs are different, careful about the config path
        self.model = Conv2DFiLMNet(**model_structure)
        self.model.build()
        self.model.to(self.device)

        ckpt = torch.load(cfg['local_inferer_path'], map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        dino_model_type = cfg['dino_model_type']
        dino_use_registers = cfg["dino_use_registers"]
        self.dino = load_pretrained_dino(dino_model_type, 
                                         use_registers=dino_use_registers).to(self.device).eval()

        self.text_embedding_func = cfg['text_embedding_func']
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
        """Resize a model heatmap from model/patch resolution back to image size.

        Args:
            pred_map: 2D numpy array, usually a patch-level heatmap from the
                local model after activation.
            out_hw: Target image size as (height, width).

        Returns:
            2D numpy array with shape (height, width).
        """
        out_h, out_w = out_hw
        return np.array(
            T.functional.resize(
                Image.fromarray(pred_map.astype(np.float32), mode="F"),
                (out_h, out_w),
                interpolation=T.InterpolationMode.BILINEAR))
    
    @torch.no_grad()
    def predict(self, img_np, text, thresh=0.5):
        """Predict an affordance heatmap for one image-text pair.

        Args:
            img_np: RGB image as a numpy array with shape (H, W, 3).
            text: Affordance description used to compute the language embedding.
            thresh: Optional threshold applied after sigmoid. If None, returns
                a continuous score map in [0, 1].

        Internal tensor meaning:
            DINO produces patch features with shape (1, H', W', C), then the
            Late FiLM head outputs patch-level logits with shape (H', W').
            These logits are sigmoid scores, where each patch is an independent
            affordance probability-like score.

        Returns:
            2D numpy array with shape (H, W), resized back to the input image
            size. Values are continuous scores if thresh is None, otherwise
            binary 0/1 values.
        """
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
        """Predict affordance heatmaps for a list of images with one text prompt.

        Args:
            img_np_list: List of RGB numpy arrays, each with shape (H, W, 3).
            text: One affordance description shared by all images in the batch.
            thresh: Optional threshold applied after sigmoid. If None, returns
                continuous score maps in [0, 1].

        Internal tensor meaning:
            Images with the same transformed size are batched together. The
            Late FiLM head outputs patch-level logits with shape (B, H', W'),
            then sigmoid converts them to independent patch scores.

        Returns:
            List of 2D numpy arrays. outputs[i] has shape equal to the original
            height and width of img_np_list[i].
        """
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
    
    


class SteerViTAffordanceInference:
    '''Backend using SteerViT. The model fuses text features into patch embed tokens within the ViT backbone, 
    which is earlier than LateFiLM.'''
    def __init__(self, config_path, checkpoint_path, text_embedding_func=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = load_config(config_path)
        steer_cfg = cfg.get("steervit", {})
        steer_checkpoint = steer_cfg.get("checkpoint", checkpoint_path)

        from model.steer_model import SteerViT

        self.model = SteerViT.from_pretrained(steer_checkpoint, device=self.device)
        self.model.eval()
        self.transform = self.model.get_transforms()
        self.output_activation = steer_cfg.get("output_activation", "sigmoid").lower()

    def clear_text_embedding_cache(self):
        pass

    @staticmethod
    def _resize_pred_to_image(pred_map: np.array, out_hw):
        """Resize a SteerViT heatmap from model resolution back to image size.

        Args:
            pred_map: 2D numpy array with shape equal to the SteerViT input
                resolution after patch logits have been upsampled.
            out_hw: Target image size as (height, width).

        Returns:
            2D numpy array with shape (height, width).
        """
        out_h, out_w = out_hw
        return np.array(
            T.functional.resize(
                Image.fromarray(pred_map.astype(np.float32), mode="F"),
                (out_h, out_w),
                interpolation=T.InterpolationMode.BILINEAR))

    def _heatmaps_from_batch(self, images: torch.Tensor, text: str):
        """Run SteerViT on a batch and return model-resolution heatmaps.

        Args:
            images: Preprocessed image tensor with shape (B, 3, S, S), where S
                is the SteerViT image size.
            text: One affordance description shared by all images.

        Tensor meaning:
            If output_activation == "softmax", SteerViT get_heatmaps() applies
            softmax over all image patches, so values describe a normalized
            spatial distribution for each image.

            Otherwise, the segmentation head produces patch-level logits with
            shape (B, H', W'). Sigmoid is applied per patch, so values are
            independent probability-like affordance scores.

        Returns:
            Torch tensor with shape (B, S, S), already upsampled from patch
            resolution to the SteerViT model input resolution.
        """
        texts = [text] * images.size(0) # duplicate the text for each image in the batch
        if self.output_activation == "softmax":
            return self.model.get_heatmaps(images, texts=texts).squeeze(1)

        feats = self.model.forward(images.to(self.device), texts)
        num_prefix_tokens = self.model.vision_model.trunk.num_prefix_tokens
        patch_feats = feats[:, num_prefix_tokens:, :]
        patch_logits = self.model.lin_seg_head(patch_feats).squeeze(-1)
        patch_h = self.model.image_size[0] // self.model.patch_size
        patch_w = self.model.image_size[1] // self.model.patch_size
        patch_logits = patch_logits.view(images.size(0), 1, patch_h, patch_w)
        logits = torch.nn.functional.interpolate(
            patch_logits,
            size=self.model.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(logits).squeeze(1)

    @torch.no_grad()
    def predict(self, img_np, text, thresh=0.5):
        """Predict a SteerViT affordance heatmap for one image-text pair.

        Args:
            img_np: RGB image as a numpy array with shape (H, W, 3).
            text: Affordance description passed into SteerViT's text encoder.
            thresh: Optional threshold applied after heatmap generation. If
                None, returns the continuous heatmap.

        Returns:
            2D numpy array with shape (H, W), resized back to the input image
            size. The value meaning depends on output_activation: softmax gives
            a normalized spatial distribution; sigmoid gives independent
            probability-like patch scores.
        """
        image = Image.fromarray(img_np).convert("RGB")
        proc = self.transform(image).unsqueeze(0).to(self.device)
        sim = self._heatmaps_from_batch(proc, text).squeeze(0)
        if thresh is not None:
            sim = (sim > thresh).float()
        sim_np = sim.cpu().numpy()
        H, W = img_np.shape[:2]
        return self._resize_pred_to_image(sim_np, (H, W))

    @torch.no_grad()
    def predict_batch(self, img_np_list, text, thresh=None):
        """Predict SteerViT affordance heatmaps for a list of images.

        Args:
            img_np_list: List of RGB numpy arrays, each with shape (H, W, 3).
            text: One affordance description shared by all images in the batch.
            thresh: Optional threshold applied after heatmap generation. If
                None, returns continuous heatmaps.

        Returns:
            List of 2D numpy arrays. outputs[i] has shape equal to the original
            height and width of img_np_list[i]. The value meaning follows
            output_activation: softmax distribution or sigmoid patch scores.
        """
        if not img_np_list:
            return []

        orig_sizes = [img_np.shape[:2] for img_np in img_np_list]
        batch_proc = torch.stack(
            [self.transform(Image.fromarray(img_np).convert("RGB")) for img_np in img_np_list],
            dim=0,
        ).to(self.device)

        sim = self._heatmaps_from_batch(batch_proc, text)
        if thresh is not None:
            sim = (sim > thresh).float()
        sim_np = sim.cpu().numpy()

        outputs = []
        for pred_map, orig_hw in zip(sim_np, orig_sizes):
            outputs.append(self._resize_pred_to_image(pred_map, orig_hw))
        return outputs
