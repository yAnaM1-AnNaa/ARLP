#!/usr/bin/env python3
"""
后处理工具函数 - 用于改进affordance预测质量

主要功能:
1. 基于深度的过滤 - 移除深度不一致的区域
2. 连通域分析 - 移除小的孤立区域
3. 空间平滑 - 形态学操作
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing
import cv2


def depth_aware_filtering(pred_map, depth_map,
                          depth_percentile=50,
                          depth_tolerance=0.15,
                          min_confidence=0.3,
                          adaptive_tolerance=True):
    """
    基于深度信息过滤不合理的affordance预测区域

    核心思想:
    1. 找到高置信度预测区域的主要深度
    2. 过滤掉深度差异过大的区域(例如墙面通常比table远)
    3. 保持affordance在空间上的连贯性

    Args:
        pred_map: 预测的affordance map (H, W), float [0,1]
        depth_map: 深度图 (H, W), float, 单位可以是米或任意单位
        depth_percentile: 用于计算主深度的百分位数(50=中位数)
        depth_tolerance: 深度容差比例, 例如0.15表示±15%
        min_confidence: 只考虑置信度>该值的像素来计算主深度
        adaptive_tolerance: 是否根据深度值自适应调整容差

    Returns:
        filtered_pred: 过滤后的预测 (H, W), float [0,1]
    """
    # 确保尺寸一致
    if pred_map.shape != depth_map.shape:
        from PIL import Image
        depth_map = np.array(Image.fromarray(depth_map).resize(
            (pred_map.shape[1], pred_map.shape[0]), Image.BILINEAR))

    # 1. 找到高置信度区域
    high_conf_mask = pred_map > min_confidence

    if not high_conf_mask.any():
        print(f"[Depth Filter] 警告: 没有置信度>{min_confidence}的预测，跳过深度过滤")
        return pred_map

    # 2. 计算高置信度区域的主要深度
    valid_depths = depth_map[high_conf_mask]

    # 过滤无效深度值(0, inf, nan)
    valid_depths = valid_depths[np.isfinite(valid_depths) & (valid_depths > 0)]

    if len(valid_depths) == 0:
        print(f"[Depth Filter] 警告: 没有有效的深度值，跳过深度过滤")
        return pred_map

    # 使用百分位数计算主深度(比mean更robust)
    main_depth = np.percentile(valid_depths, depth_percentile)

    # 3. 计算深度容差
    if adaptive_tolerance:
        # 自适应容差: 近处物体容差小，远处物体容差大
        # 例如: 0.5m的table容差±0.075m, 2m的table容差±0.3m
        abs_tolerance = main_depth * depth_tolerance
    else:
        abs_tolerance = depth_tolerance  # 固定容差

    # 4. 创建深度mask: 保留深度在合理范围内的区域
    depth_diff = np.abs(depth_map - main_depth)
    depth_mask = depth_diff <= abs_tolerance

    # 5. 应用深度mask
    filtered_pred = pred_map * depth_mask.astype(np.float32)

    # 6. 统计信息
    removed_ratio = 1.0 - (filtered_pred > min_confidence).sum() / max(1, high_conf_mask.sum())
    print(f"[Depth Filter] 主深度={main_depth:.3f}, 容差=±{abs_tolerance:.3f}, "
          f"移除了{removed_ratio*100:.1f}%的高置信度像素")

    return filtered_pred


def connected_components_filtering(pred_map,
                                   min_area=100,
                                   threshold=0.5,
                                   keep_top_k=3):
    """
    基于连通域分析移除小的、孤立的预测区域

    Args:
        pred_map: 预测的affordance map (H, W), float [0,1]
        min_area: 最小连通域面积(像素数), 小于该值的区域会被移除
        threshold: 二值化阈值
        keep_top_k: 保留面积最大的前k个连通域, None表示保留所有满足min_area的

    Returns:
        filtered_pred: 过滤后的预测 (H, W), float [0,1]
    """
    # 1. 二值化
    binary_mask = (pred_map > threshold).astype(np.uint8)

    if not binary_mask.any():
        print(f"[CC Filter] 警告: 没有像素>{threshold}，跳过连通域过滤")
        return pred_map

    # 2. 连通域分析
    labeled, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        return pred_map

    # 3. 计算每个连通域的面积
    component_sizes = []
    for i in range(1, num_features + 1):
        area = np.sum(labeled == i)
        component_sizes.append((i, area))

    # 4. 按面积排序
    component_sizes.sort(key=lambda x: x[1], reverse=True)

    # 5. 确定要保留的连通域
    keep_labels = set()
    for i, (label, area) in enumerate(component_sizes):
        # 保留条件: 面积足够大 且 在top-k内(如果指定)
        if area >= min_area:
            if keep_top_k is None or i < keep_top_k:
                keep_labels.add(label)

    # 6. 创建mask
    keep_mask = np.zeros_like(binary_mask, dtype=bool)
    for label in keep_labels:
        keep_mask |= (labeled == label)

    # 7. 应用mask
    filtered_pred = pred_map * keep_mask.astype(np.float32)

    # 8. 统计信息
    removed_components = num_features - len(keep_labels)
    print(f"[CC Filter] 共{num_features}个连通域, 保留{len(keep_labels)}个, "
          f"移除{removed_components}个小区域")

    return filtered_pred


def morphological_smoothing(pred_map,
                            threshold=0.5,
                            kernel_size=5,
                            operation='close'):
    """
    形态学平滑 - 去除噪点和孔洞

    Args:
        pred_map: 预测的affordance map (H, W), float [0,1]
        threshold: 二值化阈值
        kernel_size: 形态学核大小
        operation: 'open'(去噪点), 'close'(填孔洞), 'both'

    Returns:
        smoothed_pred: 平滑后的预测 (H, W), float [0,1]
    """
    # 1. 二值化
    binary_mask = (pred_map > threshold).astype(np.uint8)

    if not binary_mask.any():
        return pred_map

    # 2. 创建形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 3. 形态学操作
    if operation == 'open':
        smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'both':
        # 先close填孔洞，再open去噪点
        temp = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        smoothed_mask = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    # 4. 保留原始概率值，只应用mask
    smoothed_pred = pred_map * smoothed_mask.astype(np.float32)

    return smoothed_pred


def adaptive_threshold_filter(pred_map,
                              method='otsu',
                              post_dilate=False):
    """
    自适应阈值过滤 - 根据预测分布自动确定阈值

    Args:
        pred_map: 预测的affordance map (H, W), float [0,1]
        method: 'otsu' 或 'percentile'
        post_dilate: 是否在阈值化后进行膨胀

    Returns:
        filtered_pred: 过滤后的预测 (H, W), float [0,1]
    """
    # 1. 计算自适应阈值
    pred_uint8 = (pred_map * 255).astype(np.uint8)

    if method == 'otsu':
        # Otsu's method
        threshold_val, _ = cv2.threshold(pred_uint8, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_val = threshold_val / 255.0
    elif method == 'percentile':
        # 使用75分位数作为阈值
        threshold_val = np.percentile(pred_map[pred_map > 0], 75)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"[Adaptive Threshold] 自动阈值={threshold_val:.3f}")

    # 2. 应用阈值
    binary_mask = (pred_map > threshold_val).astype(np.uint8)

    # 3. 可选: 膨胀操作
    if post_dilate and binary_mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # 4. 保留原始概率值
    filtered_pred = pred_map * binary_mask.astype(np.float32)

    return filtered_pred


def integrated_postprocessing(pred_map,
                              depth_map=None,
                              use_depth=True,
                              use_cc_filter=True,
                              use_morph=True,
                              depth_tolerance=0.15,
                              min_area=100,
                              keep_top_k=3,
                              morph_kernel=5,
                              verbose=True):
    """
    集成的后处理pipeline

    推荐的处理顺序:
    1. 深度过滤 (移除大范围的无关区域，如墙面)
    2. 形态学平滑 (去除小噪点和孔洞)
    3. 连通域过滤 (移除孤立小区域)

    Args:
        pred_map: 预测的affordance map (H, W), float [0,1]
        depth_map: 深度图 (H, W), 可选
        use_depth: 是否使用深度过滤
        use_cc_filter: 是否使用连通域过滤
        use_morph: 是否使用形态学平滑
        其他参数: 见各个函数的文档

    Returns:
        filtered_pred: 过滤后的预测 (H, W), float [0,1]
    """
    result = pred_map.copy()

    if verbose:
        print("\n" + "="*60)
        print("开始集成后处理...")
        print("="*60)

    # 1. 深度过滤
    if use_depth and depth_map is not None:
        if verbose:
            print("\n[步骤1/3] 深度约束过滤...")
        result = depth_aware_filtering(
            result, depth_map,
            depth_tolerance=depth_tolerance,
            min_confidence=0.3
        )
    elif use_depth and depth_map is None:
        print("[警告] 要求使用深度过滤但未提供depth_map，跳过")

    # 2. 形态学平滑
    if use_morph:
        if verbose:
            print(f"\n[步骤2/3] 形态学平滑 (kernel={morph_kernel})...")
        result = morphological_smoothing(
            result,
            threshold=0.3,
            kernel_size=morph_kernel,
            operation='both'
        )

    # 3. 连通域过滤
    if use_cc_filter:
        if verbose:
            print(f"\n[步骤3/3] 连通域过滤 (min_area={min_area}, top_k={keep_top_k})...")
        result = connected_components_filtering(
            result,
            min_area=min_area,
            threshold=0.3,
            keep_top_k=keep_top_k
        )

    if verbose:
        # 统计总体效果
        orig_active = (pred_map > 0.3).sum()
        final_active = (result > 0.3).sum()
        reduction = 1.0 - final_active / max(1, orig_active)
        print("\n" + "="*60)
        print(f"后处理完成! 激活像素减少了{reduction*100:.1f}%")
        print(f"原始: {orig_active} 像素 -> 过滤后: {final_active} 像素")
        print("="*60 + "\n")

    return result


# ============================================================================
# 辅助函数
# ============================================================================

def load_depth_map(depth_path):
    """
    加载深度图 (支持npy格式)

    Args:
        depth_path: 深度图路径 (.npy)

    Returns:
        depth_map: numpy array (H, W)
    """
    if depth_path.endswith('.npy'):
        depth = np.load(depth_path)
    else:
        raise ValueError(f"不支持的深度图格式: {depth_path}, 请使用.npy格式")

    return depth


def visualize_filtering_steps(pred_original, pred_filtered,
                               depth_map=None, save_path=None):
    """
    可视化过滤前后的对比

    Args:
        pred_original: 原始预测 (H, W)
        pred_filtered: 过滤后的预测 (H, W)
        depth_map: 可选的深度图
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt

    n_plots = 3 if depth_map is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))

    axes[0].imshow(pred_original, cmap='jet', vmin=0, vmax=1)
    axes[0].set_title("Original Prediction")
    axes[0].axis('off')

    axes[1].imshow(pred_filtered, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Filtered Prediction")
    axes[1].axis('off')

    if depth_map is not None:
        axes[2].imshow(depth_map, cmap='viridis')
        axes[2].set_title("Depth Map")
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # 测试代码
    print("后处理工具模块已加载")
    print("主要函数:")
    print("  - depth_aware_filtering: 基于深度的过滤")
    print("  - connected_components_filtering: 连通域过滤")
    print("  - morphological_smoothing: 形态学平滑")
    print("  - integrated_postprocessing: 集成后处理pipeline")
