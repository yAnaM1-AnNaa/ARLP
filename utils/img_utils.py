"""
Util functions for image processing
"""
import matplotlib
matplotlib.use('Agg')
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModel

# Auto-detect CUDA device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")



#################################################
### Image Pre-processing
#################################################

def find_crop_box(img_array):
    """
    Find the crop box to remove white margin. 
    img_array: (H, W, 3) numpy array
    return: (left, top, right, bottom) crop box
    """
    original_size = img_array.shape[:2]

    # Assuming white background is [255, 255, 255]
    # Find rows and columns where all values are 255
    if np.max(img_array) > 1:
        rows_with_background = np.all(img_array[:, :, :3] == 255, axis=(1, 2))
        cols_with_background = np.all(img_array[:, :, :3] == 255, axis=(0, 2))
    else:
        rows_with_background = np.all(img_array[:, :, :3] == 1.0, axis=(1, 2))
        cols_with_background = np.all(img_array[:, :, :3] == 1.0, axis=(0, 2))

    # Find indices of first and last rows and columns that are not all white (background)
    row_indices = np.where(~rows_with_background)[0]
    col_indices = np.where(~cols_with_background)[0]
    if len(row_indices) == 0 or len(col_indices) == 0:
        # Image is all white; nothing to crop
        crop_box = (0, 0, original_size[0], original_size[1])
    else:
        top, bottom = row_indices[0], row_indices[-1]
        left, right = col_indices[0], col_indices[-1]

        # Determine the largest square crop
        crop_size = max(bottom-top+1, right-left+1)
        center_row = (top + bottom) // 2
        center_col = (left + right) // 2

        half_crop_size = crop_size // 2
        crop_box = (
            max(center_col - half_crop_size, 0),
            max(center_row - half_crop_size, 0),
            min(center_col + half_crop_size + crop_size % 2, original_size[0]),
            min(center_row + half_crop_size + crop_size % 2, original_size[1])
        )
        # turn into json serializable
        crop_box = list(crop_box)
        crop_box = [int(x) for x in crop_box]

    return crop_box

def crop_and_rescale_image(img_array, new_size, crop_box=None):
    """
    crop and rescale the image to new size. 
    img_array: (H, W, 3) numpy array
    crop_box: (left, top, right, bottom) crop box
    return: (H, W, 3) cropped image
    """
    if crop_box is None:
        crop_box = find_crop_box(img_array)
    (left, top, right, bottom) = crop_box
    cropped_image = img_array[top:bottom, left:right]
    rescaled_image = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_LINEAR)
    return rescaled_image

def adjust_intrinsics_and_crop(image, intrinsics, crop_box, new_size, depth=None, seg=None):
    """
    Crop the image and adjust the intrinsics accordingly.

    image: (H, W, 3) numpy array
    intrinsics: (3, 3) numpy array
    crop_box: (left, top, right, bottom) crop box
    new_size: (new_h, new_w) new size
    depth: (H, W) numpy array or None
    seg: (H, W) numpy array or None
    """
    (left, top, right, bottom) = crop_box
    (new_h, new_w) = new_size
    
    # Crop the image
    # Crop
    cropped_image = image[top:bottom, left:right]
    cropped_depth = depth[top:bottom, left:right] if depth is not None else None
    cropped_seg = seg[top:bottom, left:right] if seg is not None else None
    
    # Rescale
    rescaled_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rescaled_depth = cv2.resize(cropped_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST) if depth is not None else None
    rescaled_seg = cv2.resize(cropped_seg, (new_w, new_h), interpolation=cv2.INTER_NEAREST) if seg is not None else None
    
    # Calculate scale factors for the intrinsics
    scale_x = new_w / (right - left)
    scale_y = new_h / (bottom - top)
    
    # Adjust the camera intrinsics
    new_intrinsics = np.copy(intrinsics)
    new_intrinsics[0, 0] *= scale_x  # Scale fx
    new_intrinsics[1, 1] *= scale_y  # Scale fy
    new_intrinsics[0, 2] = (intrinsics[0, 2] - left) * scale_x  # Adjust cx
    new_intrinsics[1, 2] = (intrinsics[1, 2] - top) * scale_y   # Adjust cy
    
    return {"image": rescaled_image, "intrinsics": new_intrinsics, "depth": rescaled_depth, "seg": rescaled_seg}


#################################################
### Image Features
#################################################

### DINO Features old version w/o registers ###

def load_pretrained_dino(model_name, use_registers=False, torch_path=None):
    '''
    model_type: in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'] or dinov3
    use_registers: bool, whether to use registers
    torch_path: str, path to torch model cache directory (optional)
    '''
    model_map = {
        'dinov2_vits14': 'facebook/dinov2-small',
        'dinov2_vitb14': 'facebook/dinov2-base',
        'dinov2_vitl14': 'facebook/dinov2-large',
        'dinov2_vitg14': 'facebook/dinov2-giant',
        'dinov3_vits16plus': '/root/autodl-tmp/dinov3',
        'dinov2_with_registers_b': 'facebook/dinov2-with-registers-base',
        'dinov2_with_registers_l': 'facebook/dinov2-with-registers-large',
    }
    model_path = model_map[model_name]

    if model_name.split('_')[0] == 'dinov2':
        model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()
    else:
        model = torch.hub.load(
            repo_or_dir=model_path,
            model=model_name,
            source="local",
            weights=model_path + "/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
        model = model.to(DEVICE).eval()
    print(f"Loaded {model_name} model")
    return model

"""Include both image transormation and DINOv2 forward pass"""
def get_dino_features(dinov2, img, blur=True, repeat_to_orig_size=True):
    """
    Original code to get DINOv2 features for image.
    ::param img:: np.array of shape (H, W, C) or (bs, H, W, C)
    ::param blur:: bool, whether to apply Gaussian blur before resizing

    ::return:: np.array of shape (bs, patch_h, patch_w, n_features) if not repeat_to_orig_size
                np.array of shape (bs, H, W, n_features) if repeat_to_orig_size
    """
    transformed_imgs = transform_imgs(img, blur=blur)
    transformed_imgs = torch.stack(transformed_imgs).to(DEVICE)
    features = get_dino_features_from_transformed_imgs(dinov2, transformed_imgs, repeat_to_orig_size=repeat_to_orig_size)
    return features
    

def transform_imgs(imgs, blur=True):
    """
    Transform image before passing to DINO model
    ::param imgs:: np.array of shape (H, W, C) or (bs, H, W, C)
    ::param blur:: bool, whether to apply Gaussian blur before resizing

    ::return:: list of transformed images
    """
    # handles both single image and batch of images
    if len(imgs.shape) == 3:
        H, W, C = imgs.shape
        imgs = imgs[None, ...]
        bs = 1
    else:
        bs, H, W, C = imgs.shape

    H *= 2
    W *= 2

    patch_h = H // 14
    patch_w = W // 14

    if blur:
        transform_lst = [T.GaussianBlur(9, sigma=(1.0, 2.0))]
    else:
        transform_lst = []
    transform_lst += [
        T.Resize((patch_h * 14, patch_w * 14)),
        T.CenterCrop((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    transform = T.Compose(transform_lst)
    
    transformed_imgs = []
    for i in range(bs):
        temp = imgs[i].copy()
        if temp.max() <= 1.1: # handle images with values in [0, 1]
            temp = (temp * 255)
        temp = temp.astype(np.uint8).clip(0, 255)
        transformed_imgs.append(transform(Image.fromarray(temp)))
    
    return transformed_imgs

def get_dino_features_from_transformed_imgs(dinov2, imgs, repeat_to_orig_size=True):
    """
    Get DINOv2 features from transformed images
    ::param dinov2:: DINO model
    ::param imgs:: tensor of shape (bs, C, H, W)
    """
    
    device = next(dinov2.parameters()).device  # 获取模型所在设备
    imgs = imgs.to(device)  # 将输入张量移动到模型所在设备
    bs, C, H, W = imgs.shape
    patch_h = H // 14
    patch_w = W // 14

    with torch.no_grad():
        # [关键修改] 检查模型类型并调用正确的方法
        if hasattr(dinov2, 'forward_features'):
            # 情况 A: 原版 Facebook Research 模型 (torch.hub)
            features_dict = dinov2.forward_features(imgs)
            features = features_dict['x_norm_patchtokens']
        else:
            # 情况 B: Hugging Face Transformers 
            # HF 模型没有 forward_features 方法，直接调用 dinov2(imgs)
            outputs = dinov2(imgs)
            # HF 输出包含 CLS token (index 0)
            # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
            num_register_tokens = int(getattr(dinov2.config, "num_register_tokens", 0))
            features = outputs.last_hidden_state[:, 1 + num_register_tokens:, :]
            features = features.reshape(bs, patch_h, patch_w, -1)
        
        
    if not repeat_to_orig_size:
        return features # (bs, patch_h, patch_w, n_features)
    else:
        # repeat on batched dims to original size
        ratio = H // (patch_h*2)
        # (bs, patch_h, patch_w, n_features) -> (bs, H, W, n_features)
        # features = np.repeat(np.repeat(features, ratio, axis=1), ratio, axis=2)
        features = F.interpolate(features.permute(0, 3, 1, 2), scale_factor=ratio, mode='bilinear').permute(0, 2, 3, 1)
        features = features.cpu().numpy()

        # warn user if the image shape is not the same as the original image
        if features.shape[1] != H/2 or features.shape[2] != W/2:
            print(f"Warning: The image shape is not the same as the original image. Original shape: {H/2, W/2}, New shape: {features.shape[1], features.shape[2]}")

        return features


#################################################
### Visualization
#################################################
def grid_visualize(img_list, name_list=None, save_path=None, n_rows=2, title=None):
    '''
    Visualize the input images in a n_rows * m grid
    img_list: list of images
    name_list: list of names for each image
    '''
    if name_list and len(name_list) != len(img_list):
        print("Length of name_list does not match length of img_list, discarding name_list.")
        name_list = None

    n_cols = len(img_list) // n_rows + (len(img_list) % n_rows > 0)
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5), squeeze=False)

    axs = axs.flatten()
    for ax in axs:
        ax.axis('off')

    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.imshow(img)
        if name_list:
            # ax.set_title(name_list[i])
            ax.set_title(name_list[i], fontsize=12, pad=5)

    if title:
        plt.suptitle(title)

    plt.tight_layout() 
    
    if save_path:
        plt.savefig(save_path)
    # else:
        # plt.show()
    plt.close()

def get_palette():
    """Provides a list of colros for visualization"""
    colors = [
    ([230, 25, 75], "Red"),
    ([60, 180, 75], "Green"),
    ([0, 130, 200], "Blue"),
    ([255, 225, 25], "Yellow"),
    ([245, 130, 48], "Orange"),
    ([145, 30, 180], "Purple"),
    ([70, 240, 240], "Cyan"),
    ([240, 50, 230], "Magenta"),
    ([250, 190, 212], "Pink"),
    ([210, 245, 60], "Lime Green"),
    ([0, 128, 128], "Teal"),
    ([170, 110, 40], "Brown"),
    ([128, 0, 0], "Maroon"),
    ([0, 0, 128], "Navy"),
    ([107, 142, 35], "Olive"),
    ([128, 128, 128], "Gray"),
    ([220, 20, 60], "Crimson"),
    ([0, 0, 0], "Black"),
    ([204, 85, 0], "Burnt Orange"),
    ([0, 153, 143], "Jade"),
    ]

    return colors

def visualize_palette():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    full_palette = get_palette()
    # Set up the figure and axes for the color patches
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(full_palette) / 2)
    plt.axis('off')

    # Create a patch for each color
    for i, ((r, g, b), name) in enumerate(full_palette):
        ax.add_patch(patches.Rectangle((i % 10, i // 10), 1, 1, color=[r/255, g/255, b/255]))
        ax.text(i % 10 + 0.5, i // 10 - 0.1, name, va='top', ha='center', fontsize=8)

    plt.savefig("/root/autodl-tmp/public/public-e/unsup-affordance/runs/color_palette.png", bbox_inches='tight')
    plt.close()

def overlay_heatmap(img, heatmap, alpha=0.5, colormap=plt.cm.jet):
    """
    将热力图叠加到原图上
    
    Args:
        img: 原图 (H, W, 3), uint8 或 float [0,1]
        heatmap: 热力图 (H, W), float [0,1]
        alpha: 热力图透明度
        colormap: matplotlib colormap
        
    Returns:
        overlay: 叠加后的图像 (H, W, 3), float [0,1]
    """
    # 确保img是float [0,1]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    # 确保heatmap尺寸与img一致
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = np.array(Image.fromarray(heatmap).resize(
            (img.shape[1], img.shape[0]), Image.BILINEAR))
    
    # 将heatmap转换为彩色 (H, W, 4) -> (H, W, 3)
    heatmap_colored = colormap(heatmap)[:, :, :3]
    
    # 叠加
    overlay = (1 - alpha) * img + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return overlay