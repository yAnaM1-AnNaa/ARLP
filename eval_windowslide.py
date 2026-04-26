"""
eval_windowslide.py - brute-force sliding-window panorama baseline.

Role:
1. Simplest panorama solution: resize image to a window grid, crop windows, run local inference, and average pasted maps.
2. Does not use object detector boxes.
3. Kept as a baseline for comparing detector-crop pano inference.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from utils.file_utils import load_config
from src.local_inference import build_local_inferer
from utils.vlm_utils import get_text_embedding_options
from utils.eval_utils import cal_kl, cal_sim, cal_nss  # , cal_iou
from utils.img_utils import grid_visualize
from utils.postprocess_utils import integrated_postprocessing, load_depth_map, visualize_filtering_steps


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


def calc_resize_dim(orig_size, window_size):
    """
    智能计算resize后的尺寸，最小化变形程度
    
    策略：
    - k = orig_size // window_size (可放置的完整窗口数)
    - remainder = orig_size - window_size * k (剩余长度)
    - 如果 remainder < window_size / 2，则缩放到 window_size * k
    - 否则缩放到 window_size * (k + 1)
    - 特殊情况：如果 k == 0，则缩放到 window_size
    
    Args:
        orig_size: 原始尺寸（宽或高）
        window_size: 滑窗大小
        
    Returns:
        new_size: resize后的尺寸（window_size的整数倍）
    """
    k = orig_size // window_size
    remainder = orig_size - window_size * k
    
    if k == 0:
        # 图像比窗口还小，resize到一个窗口大小
        return window_size
    
    if remainder < window_size / 2:
        # 剩余不足半个窗口，缩小
        return window_size * k
    else:
        # 剩余超过半个窗口，扩展
        return window_size * (k + 1)


def predict_sliding_window(inference_model, img_np, text, window_size=518, thresh=None):
    """
    独立的滑窗推理函数（与inference.py解耦）
    
    核心思想：
    - 智能resize：根据剩余长度决定缩放到k还是k+1个窗口
    - 使用518×518正方形窗口进行滑窗推理（无重叠）
    - 对每个窗口调用原有的inference.predict()
    - 使用累加器和计数图融合重叠区域
    - 最后resize回原始尺寸
    
    Args:
        inference_model: AffordanceInference实例
        img_np: 全分辨率输入图像 (H, W, 3)
        text: 文本查询
        window_size: 窗口大小（正方形），默认518（与DINOv2预训练一致）
        thresh: 最终输出的二值化阈值，None则返回概率图
        
    Returns:
        final_map: 与输入图像相同尺寸的预测结果 (H, W)
    """
    H_orig, W_orig = img_np.shape[:2]
    
    # 1. 智能计算resize后的尺寸
    H_new = calc_resize_dim(H_orig, window_size)
    W_new = calc_resize_dim(W_orig, window_size)
    
    # 2. Resize图像
    if (H_new, W_new) != (H_orig, W_orig):
        img_resized = np.array(Image.fromarray(img_np).resize((W_new, H_new), Image.BILINEAR))
        print(f"[INFO] 图像从 {W_orig}x{H_orig} resize到 {W_new}x{H_new}")
    else:
        img_resized = img_np
    
    H, W = img_resized.shape[:2]
    win_h = win_w = window_size
    
    # 计算滑动步长（无重叠，步长等于窗口大小）
    stride_h = stride_w = window_size

    
    # 初始化累加器
    probs_map = np.zeros((H, W), dtype=np.float32)  # 概率累加
    count_map = np.zeros((H, W), dtype=np.float32)  # 计数图（用于平均）
    
    # 生成滑窗网格
    h_steps = list(range(0, H - win_h + 1, stride_h))
    w_steps = list(range(0, W - win_w + 1, stride_w))
    
    # 确保覆盖边缘
    if H > win_h and (h_steps[-1] + win_h < H):
        h_steps.append(H - win_h)
    if W > win_w and (w_steps[-1] + win_w < W):
        w_steps.append(W - win_w)
        
    # 如果图像小于窗口，则只推理一次
    if H <= win_h:
        h_steps = [0]
    if W <= win_w:
        w_steps = [0]
    
    # 滑窗推理循环
    for h_start in h_steps:
        for w_start in w_steps:
            # 1. 确定裁剪坐标
            h_end = min(h_start + win_h, H)
            w_end = min(w_start + win_w, W)
            
            # 2. 裁剪窗口（注意：从resize后的图像裁剪）
            crop = img_resized[h_start:h_end, w_start:w_end, :]
            crop_h, crop_w = crop.shape[:2]
            
            # 3. 处理边界情况：当crop小于窗口时，进行零填充
            if crop_h < win_h or crop_w < win_w:
                pad_h = max(0, win_h - crop_h)
                pad_w = max(0, win_w - crop_w)
                crop_padded = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            else:
                crop_padded = crop
            
            # 4. 调用原有推理接口（关键解耦点）
            # 传入thresh=None以获取原始概率图，便于后续融合
            local_pred = inference_model.predict(crop_padded, text, thresh=None)
            
            # 5. 裁剪回原始大小（去除填充）
            local_pred = local_pred[:crop_h, :crop_w]
            
            # 6. 累加到全局图
            probs_map[h_start:h_start+crop_h, w_start:w_start+crop_w] += local_pred
            count_map[h_start:h_start+crop_h, w_start:w_start+crop_w] += 1.0
    
    # 7. 平均融合（处理重叠区域）
    final_map = probs_map / (count_map + 1e-9)
    
    # 8. Resize回原始尺寸
    if (H, W) != (H_orig, W_orig):
        final_map = np.array(Image.fromarray(final_map).resize((W_orig, H_orig), Image.BILINEAR))
    
    # 9. 可选的阈值化
    if thresh is not None:
        final_map = (final_map > thresh).astype(np.float32)
    
    return final_map


def eval():
    """
    主评估函数
    """
    parser = argparse.ArgumentParser(description="PanoGibson滑窗评估脚本")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--agd_root", required=True, help="数据集根目录")
    parser.add_argument("--viz_dir", default=None, help="可视化结果保存路径")
    parser.add_argument("--window_size", type=int, default=518, 
                        help="滑窗大小（正方形），默认518与DINOv2预训练一致")
    # parser.add_argument("--stride_rate", type=float, default=0.5, 
    #                     help="步长比例，0.5表示50%%重叠")
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 初始化推理模型
    text_embedding_option = cfg["text_embedding_func"]
    print(f"[INFO] 使用文本编码: {text_embedding_option}")
    cfg["text_embedding_func"] = get_text_embedding_options(text_embedding_option)
    inference = build_local_inferer(cfg)
    
    # 创建可视化目录
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
    
    # 数据路径
    agd_ego_dir = os.path.join(args.agd_root, "egocentric")
    agd_gt_dir = os.path.join(args.agd_root, "GT")
    
    # 语义消歧词典（与AGD20k保持一致）
    disambiguation_set = {
        ("hit", "axe"): "handle of axe to hold during hitting",
        ("ride", "bicycle"): "region to sit on and push the bicycle",
        ("pour", "cup"): "handle of the cup to hold while pouring",
        ("wash", "cup"): "rim of the cup to wash",
        ("hold", "cup"): "handle to hold the cup",
        ("bathe", "bathtub"): "region to sit in while bathing",
        ("sit_on", "sofa_seat"): "region to sit on and relax",
        ("open", "door"): "push to open the door and enter the room behind it",
        ("place on", "table"): "top surface of the table to place objects on",
    }
    
    # 构建评估数据集
    eval_set = []
    if not os.path.exists(agd_gt_dir):
        print(f"[ERROR] GT目录不存在: {agd_gt_dir}")
        return
    
    for action in os.listdir(agd_gt_dir):
        action_dir = os.path.join(agd_gt_dir, action)
        if not os.path.isdir(action_dir):
            continue
            
        for obj in os.listdir(action_dir):
            obj_dir = os.path.join(action_dir, obj)
            if not os.path.isdir(obj_dir):
                continue
                
            for img_file in os.listdir(obj_dir):
                if img_file.startswith('.'):
                    continue
                
                gt_path = os.path.join(agd_gt_dir, action, obj, img_file)
                
                # 查找对应的RGB图像
                base_name = os.path.splitext(img_file)[0]
                img_path = None
                for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
                    candidate = os.path.join(agd_ego_dir, action, obj, base_name + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                
                if not img_path:
                    continue
                
                # 构建文本描述
                if (action, obj) in disambiguation_set:
                    text_desc = disambiguation_set[(action, obj)]
                else:
                    text_desc = f"region to {action} the {obj}"
                
                eval_set.append({
                    "img_path": img_path,
                    "gt_path": gt_path,
                    "text": text_desc,
                    "viz_name": f"{action}_{obj}_{base_name}"
                })
    
    print(f"[INFO] 共找到 {len(eval_set)} 张图像待评估")
    
    # 指标累加器
    metrics = {
        'kl': [],
        'sim': [],
        'nss01': [],
        'nss05': [],
        'iou': []
    }
    
    # 评估循环
    for data in tqdm(eval_set, desc="评估进度"):
        try:
            # 1. 加载图像和GT
            img = np.array(Image.open(data["img_path"]).convert("RGB"))
            gt_mask = plt.imread(data["gt_path"])
            
            # 处理灰度图
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            
            # 处理多通道GT
            if len(gt_mask.shape) > 2:
                gt_mask = gt_mask[:, :, 0]
            
            # 2. 滑窗推理（核心调用）
            pred_map = predict_sliding_window(
                inference,
                img,
                data["text"],
                window_size=args.window_size,
                thresh=None  # 保留概率图用于指标计算
            )
            
            # 3. 后处理
            pred_map = np.clip(pred_map, 1e-3, 1.0 - 1e-3)
            
            # 确保GT和预测尺寸一致
            if pred_map.shape != gt_mask.shape:
                gt_mask = np.array(
                    Image.fromarray(gt_mask).resize(
                        (pred_map.shape[1], pred_map.shape[0]), 
                        Image.NEAREST
                    )
                )
            
            # 4. 可视化
            if args.viz_dir:
                overlay_pred = overlay_heatmap(img, pred_map, alpha=0.5)
                overlay_gt = overlay_heatmap(img, gt_mask, alpha=0.5)
                grid_visualize(
                    # img_list=[img, overlay_pred, gt_mask],
                    # name_list=["原图", "预测", "GT"],
                    img_list=[overlay_pred, overlay_gt],
                    name_list=["predict", "GT"],
                    save_path=os.path.join(args.viz_dir, f"{data['viz_name']}.png"),
                    n_rows=1,
                    title=data['viz_name']
                )
            
            # 5. 计算指标
            metrics['kl'].append(cal_kl(pred_map, gt_mask))
            metrics['sim'].append(cal_sim(pred_map, gt_mask))
            metrics['nss01'].append(cal_nss(pred_map, gt_mask, threshold=0.1))
            metrics['nss05'].append(cal_nss(pred_map, gt_mask, threshold=0.5))
            # metrics['iou'].append(cal_iou(pred_map, gt_mask, threshold=0.5))
            
        except Exception as e:
            print(f"[ERROR] 评估失败 {data['img_path']}: {e}")
            continue
    
    # 6. 输出结果
    print("\n" + "="*50)
    print("评估完成！")
    print("="*50)
    print(f"KL散度:       {np.mean(metrics['kl']):.4f}")
    print(f"相似度(SIM):  {np.mean(metrics['sim']):.4f}")
    print(f"NSS(0.1):     {np.mean(metrics['nss01']):.4f}")
    print(f"NSS(0.5):     {np.mean(metrics['nss05']):.4f}")
    # print(f"IoU(0.5):     {np.mean(metrics['iou']):.4f}")
    print("="*50)


if __name__ == "__main__":
    eval()
