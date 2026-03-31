import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage, signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift

from img_utils import load_pretrained_dino



def resize_transform(image: Image.Image, patch_size:int = 16) -> torch.Tensor:
    width, height = image.size
    h_patches = int(height // patch_size)
    w_patches = int(width // patch_size)
    resized = TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    return TF.to_tensor(resized)


def extract_patch_features(image_tensor: torch.Tensor, layer_index: int = -1,
                           imagenet_mean=(0.485, 0.456, 0.406),
                           imagenet_std=(0.229, 0.224, 0.225)):
    image_norm = TF.normalize(image_tensor, mean=imagenet_mean, std=imagenet_std)# shape (3, H, W)
    image_batch = image_norm.unsqueeze(0).to(device) # shape (1, 3, H, W)

    num_blocks = len(model.blocks)
    if layer_index < 0:
        layer_index = num_blocks + layer_index

    with torch.inference_mode():
        feats = model.get_intermediate_layers(
            image_batch,
            n=[layer_index],
            reshape=True, # [B, embed_dim, 28, 28]
            norm=True,
        )
    feat_map = feats[0].squeeze(0).detach().cpu() # [channels, h_patches, w_patches]
    channels, h_patches, w_patches = feat_map.shape
    patch_features = feat_map.view(channels, -1).permute(1, 0).contiguous() # [H_p * W_p, embed_dim]
    return patch_features, (h_patches, w_patches)

def robust_normalize(projected_patches: np.ndarray, foreground_mask: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    '''对三通道 PCA 投影结果做鲁棒归一化：仅使用前景区域的分位数作为上下界，
    对每个通道分别进行截断，并线性缩放到 [0, 1]。'''
    normalized = projected_patches.copy()
    for channel in range(3): # 对应 PCA 后的 3 个主成分
        channel_values = normalized[..., channel][foreground_mask]
        low, high = np.percentile(channel_values, percentiles)
        normalized[..., channel] = np.clip(normalized[..., channel], low, high)
        normalized[..., channel] = (normalized[..., channel] - low) / max(high - low, 1e-6)
    return normalized

def render_pca(projected_patches: np.ndarray, 
               fg_score: torch.Tensor, image_tensor: torch.Tensor, mask_power: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    pca_only: 仅 PCA 颜色的渲染结果, shape (H, W, 3)
    overlay: PCA 颜色与原图融合的结果, shape (H, W, 3)
    fullres_mask: 前景软掩码的全分辨率版本, shape'''
    patch_rgb = torch.from_numpy(projected_patches).permute(2, 0, 1).unsqueeze(0).float() # [H_p, W_p, 3]->[1, 3, H_p, W_p]
    soft_mask = fg_score.unsqueeze(0).unsqueeze(0).float() # [H_p, W_p]->[1, 1, H_p, W_p] for interpolation/blending
    target_size = image_tensor.shape[1:]

    # 将 patch 级 PCA 颜色图上采样到原图分辨率，并转成 [H, W, 3] 的 numpy RGB 图像
    fullres_rgb = F.interpolate(
        patch_rgb,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )[0].permute(1, 2, 0).clamp(0, 1).numpy()
    # 将 patch 级前景软掩码上采样到原图分辨率，并转成 [H, W] 的 numpy 数组
    fullres_mask = F.interpolate(
        soft_mask,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )[0, 0].clamp(0, 1).numpy()

    pca_only = fullres_rgb * np.power(fullres_mask[..., None], mask_power)
    image_np = image_tensor.permute(1, 2, 0).numpy()
    overlay = image_np * (1.0 - fullres_mask[..., None]) + pca_only * fullres_mask[..., None]
    overlay = np.clip(overlay, 0.0, 1.0) # 将结果限制在合法的显示范围 [0, 1]
    return pca_only, overlay, fullres_mask

def pca(image, fg_classifier, fg_threshold=0.5, percentiles=(1, 99), cluster_method='meanshift',
          num_clusters=None, bandwidth=0.15, xy_weight=0.35, min_area_ratio=0.005):
    image_tensor = resize_transform(image)
    patch_features, (h_patches, w_patches) = extract_patch_features(image_tensor, layer_index=-1)

    fg_score = fg_classifier.predict_proba(patch_features.numpy())[:, 1].reshape(h_patches, w_patches)
    fg_score = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3)).float()
    foreground_mask = (fg_score.numpy() > fg_threshold)
    fg_patches = patch_features.numpy()[foreground_mask.reshape(-1)]

    pca = PCA(n_components=3, whiten=True)
    pca.fit(fg_patches)

    projected_patches = pca.transform(patch_features.numpy()).reshape(h_patches, w_patches, 3)
    projected_patches = robust_normalize(projected_patches, foreground_mask, percentiles)

    pca_only, overlay, fullres_mask = render_pca(projected_patches, fg_score, image_tensor)
    patch_grid = projected_patches * foreground_mask[..., None]

    yy, xx = np.meshgrid(
          np.linspace(0.0, 1.0, h_patches, dtype=np.float32),
          np.linspace(0.0, 1.0, w_patches, dtype=np.float32),
          indexing='ij',
      )
    cluster_feat = np.concatenate(
        [projected_patches, xy_weight * np.stack([yy, xx], axis=-1)],
        axis=-1,
    )
    cluster_feat = cluster_feat[foreground_mask]

    if cluster_method == 'kmeans':
        cluster_ids = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(cluster_feat)
    else:
        cluster_ids = MeanShift(bandwidth=bandwidth, cluster_all=True, n_jobs=16).fit_predict(cluster_feat)

    label_map = np.full((h_patches, w_patches), -1, dtype=np.int32)
    label_map[foreground_mask] = cluster_ids.astype(np.int32)

    min_area = max(1, int(foreground_mask.sum() * min_area_ratio))
    for label in np.unique(label_map):
        if label < 0:
            continue
        cc_map, num_cc = ndimage.label(label_map == label)
        for cc_idx in range(1, num_cc + 1):
            cc_mask = (cc_map == cc_idx)
            if int(cc_mask.sum()) < min_area:
                label_map[cc_mask] = -1

    fullres_label_map = F.interpolate(
        torch.from_numpy(label_map).unsqueeze(0).unsqueeze(0).float(),
        size=image_tensor.shape[1:],
        mode='nearest',
    )[0, 0].numpy().astype(np.int32)
    return pca_only, overlay, label_map, fullres_label_map

if __name__ == '__main__':
    image_path = '/root/autodl-tmp/ARLP/dataset/h5/vlm_query_imgs/vase_wltgjn_original.png'
    save_path = '/root/autodl-tmp/ARLP/dataset/tests/'
    image = Image.open(image_path).convert('RGB')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fg_classifier_path = "/root/autodl-tmp/dinov3/weights/fg_classifier.pkl"
    with open(fg_classifier_path, "rb") as file:
        fg_classifier = pickle.load(file)
    print('Loaded foreground classifier.')
    model = load_pretrained_dino(model_name="dinov3_vits16plus", use_registers=True)

    pca_only, overlay, label_map, fullres_label_map = pca(
        image,
        fg_classifier,
        fg_threshold=0.5,
        percentiles=(1, 99),
        cluster_method='meanshift',
        bandwidth=0.15,
        xy_weight=0.35,
        min_area_ratio=0.005,
    )
    os.makedirs(save_path, exist_ok=True)
    Image.fromarray((pca_only * 255).astype(np.uint8)).save(save_path + 'pca_only.png')
    Image.fromarray((overlay * 255).astype(np.uint8)).save(save_path + 'overlay.png')