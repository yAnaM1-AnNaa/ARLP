import os
import sys
import glob

import h5py
import numpy as np
from tqdm import tqdm
import json
import ast
import logging
import re
import sqlite3

from src.fusion import create_fusion
from src.cluster import cluster
from src.MHACoT import init_client, process_image, detect_colors

# from utils.utils import find_similarity
from utils.img_utils import grid_visualize, load_pretrained_dino
from utils.file_utils import store_or_update_dataset, save_image, save_response
from utils.vlm_utils import get_text_embedding_options
from utils.file_utils import load_config, store_logs, ResearchSql
from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPTextModelWithProjection


def parse_desctiption_list(descriptions):
    '''
    将任意形式的描述字段解析成 list[str] 
    '''
    # check whether descriptions is a list or a tuple, if so, convert each item to string and strip whitespace
    if isinstance(descriptions, (list, tuple)):
        return [str(item).strip() for item in descriptions if str(item).strip()]
    
    text = str(descriptions).strip()
    assert text, "Description is empty or invalid."

    # Formally the VLM model will answer in this format:"['red handle', 'black seat']"
    parsed = ast.literal_eval(text) # 将字符串解析成 Python 对象
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        parsed = parsed.strip()
        return [parsed] if parsed else []

    quoted_parts = re.findall(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', text)
    recovered = []
    for double_quoted, single_quoted in quoted_parts:
        part = double_quoted or single_quoted
        part = part.strip()
        if part:
            recovered.append(part)
    if recovered:
        return recovered

    return [text]

def _normalize_region_matching(region_matching):
    '''
    将 region_matching 字段解析成 list[dict] 的标准格式。
    {"Red": ["...", "..."]}
    '''
    if not isinstance(region_matching, dict):
        raise ValueError("region_matching must be a dictionary.")
    
    normalized = {}

    for key, value in region_matching.items():
        color = str(key).strip()
        parsed_value = value
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed_value = value
        if isinstance(parsed_value, (list, tuple)):
            descriptions = [str(item).strip() for item in parsed_value if str(item).strip()]
        else:
            descriptions = [str(parsed_value).strip()] if str(parsed_value).strip() else []

        if color and descriptions:
              normalized[color] = descriptions
    return normalized


def proxy_off():
    """
    关闭当前 Python 进程里的代理环境变量。
    这个函数在本脚本里主要用于切换到不兼容代理环境的组件，例如部分 OpenAI 相关客户端初始化阶段。
    """
    for var in ['http_proxy', 'https_proxy', 'all_proxy',
                'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
        os.environ.pop(var, None)
    print("😼 已关闭代理环境")

def proxy_on(proxy_url="http://127.0.0.1:7890"):
    """
    为当前 Python 进程设置代理环境变量。
    参数:
        proxy_url:
            代理地址，默认指向本机 `7890` 端口。
    """
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    os.environ['all_proxy'] = proxy_url
    print("🚀 代理已开启")

def find_best_camera_angle(h5_file_path, db, logger, top_k=5):
    """
    为一个类别对应的 HDF5 文件挑选“最能代表该物体”的 Top-K 视角。
    写回字段:
        - `clip_similarities`: 每帧与类别名的相似度
        - `top_k_indices`: 最佳视角对应的帧下标，按相似度从高到低排序
    参数:
        h5_file_path:
            类别级 HDF5 文件路径，一个文件里通常包含多个实例。
        top_k:
            需要保留的最佳视角数量。
    返回:
        无返回值。结果直接写入 HDF5。
    """
    def find_similarity(features_flat, query_feature):
        '''features_flat: (N, D), query_feature: (D,)
        return shape (N, 1)
        '''
        similarity = np.dot(features_flat, query_feature)/(np.linalg.norm(features_flat, axis=1) * np.linalg.norm(query_feature))
        return similarity.reshape(-1, 1)
    print('HF requires proxy, starting proxy.')
    proxy_on()
    print('Loading CLIP model...')
    model_vision = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32") #text embedding
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # the rgb photos are stored under the following path: 
    # {object_name}.h5/{instanec_name}/rgb: shape=(14, 378, 378, 3)
    category_name = os.path.basename(h5_file_path).split('.')[0]
    tokenized_text = tokenizer([category_name], padding=True, return_tensors="pt") # tokenize
    text_embeds = model_text(**tokenized_text).text_embeds
    text_embeds = text_embeds.squeeze(0).detach().numpy()

    # Dictionary to store similarities and top-k frame indices for each instance
    instance_data = {}

    # First pass: read the file and compute best frames
    # {obj_name}.h5
    #   {instance_name}
    #       rgb: shape=({num_frames}, 378, 378, 3), dtype=uint8
    with h5py.File(h5_file_path, 'r') as h5_file:
        instance_keys = list(h5_file.keys()) # instance_key = chair_1
        for instance_key in instance_keys:
            instance = h5_file[instance_key]
            rgb_images = instance['rgb'][:]
            all_image_features = []

            for frame_idx in range(rgb_images.shape[0]):
                rgb_image = rgb_images[frame_idx] # (378, 378, 3), uint8
                inputs = processor(images=rgb_image, return_tensors="pt") # tensor(224, 224, 3),normalized
                image_features = model_vision.get_image_features(**inputs) # tensor(1, 512)
                image_features = image_features.squeeze(0).detach().numpy() # (512,)
                all_image_features.append(image_features)
            
            all_image_features = np.array(all_image_features) # (num_frames, 512)
            clip_similarities = find_similarity(all_image_features, text_embeds)
            print('clip_similarities shape:\n', clip_similarities.shape)
            logger.record(f'clip_similarities shape: {clip_similarities.shape}\n')

            # Get top-k frame indices (ordered by similarity)
            top_k_indices = np.argsort(clip_similarities.flatten())[-top_k:][::-1]  # reverse to get descending order
            print(f"Instance {instance_key} top-k frames: {top_k_indices}")
            print(f"Top-k similarities: {clip_similarities[top_k_indices].flatten()}")
            logger.record(f"Instance {instance_key} top-k frames: {top_k_indices}\n")

            # save data into pd dataframe
            for i in range(len(instance_keys)):
                for i in range(len(top_k_indices)):
                    db.add_data(instance_key, str(top_k_indices[i]), None, 'Not yet', '')


def process_instance(instance, db, dino_model, 
                     query_original_path: str,
                     query_proposal_path: str,
                     use_data_link_segs: bool = False,
                     top_i: int = 3):
    '''处理单个instance, 步骤:
    1. 根据`top_k_indices`, 确定最佳视角和对应的2d图
    2. 调用 `create_fusion` 构建 3D 融合对象
    3. 在 3D 空间上执行聚类，得到每个点的标签
    4. 建立region matching = {color: [answer1, answer2,...]},这里也是CoT的嵌入地点
    5. 将聚合之后的3d点云投影回2d图(也就是第一步中的topk个最佳视角)
    input args:
        - `instance`: HDF5 group 对象，包含一个实例的所有数据
        - `db`: ResearchSql 对象，用于查询和更新数据库中的信息
        - `dino_model`: 预加载的 DINO 模型对象，用于特征提取
        - `query_original_path`: str, 保存原始查询图像的路径
        - `query_proposal_path`: str, 保存聚类结果图像的路径
        - `use_data_link_segs`: bool, 是否在聚类时使用数据链接分割结果作为辅助信息
    output args:
        - `color_label_names`: 每个聚类对应的颜色标签名
        - `color_name_features`: 每个颜色标签对应的聚类特征
        - `similarity_projections`: 每个簇投影到 Top-K 帧上的相似度图
        - `top_k_rgb`: Top-K 视角的原始 RGB 图像
    '''
    # 1. Retrieve the top-k frames from HDF5
    instance_name = instance.name
    top_k_indices = []
    instance_rows = db.get_instance_data(instance_name) # [{'name1':'name1, 'topk': topk..},{.}, {.}..]
    for i in range(len(instance_rows)):
        top_k_indices.append(int(instance_rows[i]['frame_idx']))
    query_frame_idx = top_k_indices[0]            # the "main" query frame

    # 2. Create the 3D fusion object from the HDF5 group and fuse frames
    fusion = create_fusion(instance, dino_model, use_data_link_segs=use_data_link_segs)
    for i in range(fusion.num_frames):
        _ = fusion.fuse_frame(i)

    # 3) Run clustering on the 3D fusion
    #    (pca_dim, use_loc, etc. are from your pipeline needs)
    cluster_results, proposal_img, used_colors = cluster(
        fusion,
        pca_dim=3,
        use_loc=0.0,
        frame_idx=top_k_indices[top_i],  # "Query" viewpoint for the color-coded result
        return_color_names=True,
        proj_3d=True,
        min_num_clusters=5,
        enable_per_link_cluster=True
    )

    # 4) Aggregate cluster features; build color->feature mapping
    unique_labels = np.unique(cluster_results[cluster_results != -1]).tolist()
    aggr_cluster_feat, similarities = fusion.aggregate_cluster_feature(
        cluster_results,
        return_similarities=True
    )
    # "color_name_feat" was originally a dict: { used_colors[label]: features_vector }
    color_name_feat = {}
    for label in unique_labels:
        color_str = used_colors[label]
        color_name_feat[color_str] = aggr_cluster_feat[label]

    # 5) Build similarity projections for top-3 frames only
    # We'll collect them into one big array: shape (num_clusters, 3, H, W)
    sim_proj_list = []
    color_label_names = []  # e.g. ["red", "blue", "brown", ...]
    for label in unique_labels:
        label_str = used_colors[label]
        color_label_names.append(label_str)

        frame_maps = []
        for idx in top_k_indices:
            # This projects the similarity scores for 'similarities[label]' onto frame idx
            proj = fusion.view_projection_to_cam_3d(
                idx,
                labels=similarities[label],
                n_neighbors=1,
                bg_val=0.0
            )
            frame_maps.append(proj)

        # shape of frame_maps: (3, H, W)
        sim_proj_list.append(frame_maps)

    # Convert to final shape: (num_clusters, 3, H, W)
    sim_proj_imgs = np.array(sim_proj_list, dtype=np.float32)

    # 6) Save new data to the HDF5 group using store_or_update_dataset
    # a) color_label_names as variable-length strings
    store_or_update_dataset(instance, "color_label_names", color_label_names)
    # b) color_name_features as shape (num_clusters, D)
    #    Build one big array from the dict
    feat_array = []
    for cname in color_label_names:  # in same order as color_label_names
        feat_vector = color_name_feat[cname]
        feat_array.append(feat_vector)
    feat_array = np.array(feat_array, dtype=np.float32)
    store_or_update_dataset(instance, "color_name_features", feat_array)
    # c) similarity_projections => shape (num_clusters, 3, H, W)
    #    also the top-3 rgb images corresponding to the similarity projections
    store_or_update_dataset(instance, "similarity_projections", sim_proj_imgs, compression="gzip")

    all_rgb = instance["rgb"][:]                
    top_k_rgb = all_rgb[top_k_indices]          # shape (top_k, H, W, 3)
    store_or_update_dataset(instance, "top_k_rgb", top_k_rgb)

    # 7) Save the original (query) and proposal images to disk
    original_img = fusion.images[query_frame_idx]  # shape (H, W, 3)
    save_image(original_img, query_original_path)
    save_image(proposal_img, query_proposal_path)

    print("Saved query original image to:", query_original_path)
    print("Saved proposal (cluster) image to:", query_proposal_path)
    print("Stored color_label_names, color_name_features, and similarity_projections in HDF5.")


def process_category(h5_path, db, dino_model, query_save_dir, embedding_type, text_embedding_func,
                     client, model_name, logger,
                     use_existing_cluster=True, use_data_link_segs=False, top_k=3):
    '''
    处理一个category中的所有instance.
        1. 调用 `process_instance` 生成聚类结果和 VLM 查询图片
        2. 使用 `detect_colors` 从 proposal 图中提取颜色标签
        3. 调用 `process_image` 让 MHACoT/VLM 根据原图和 proposal 图做区域匹配
        4. 将 VLM 返回的 `region_matching` 写回 HDF5
        5. 把每个颜色区域的文本描述编码成向量并保存
    input args:
        - `h5_path`: str, 类别级 HDF5 文件路径
        - `dino_model`: 预加载的 DINO 模型对象，用于特征提取
        - `query_save_dir`: str, 用于保存查询图像的目录路径
        - `embedding_type`: str, 文本描述的嵌入类型，例如 "openai" 或 "custom"
        - `text_embedding_func`: function, 用于将文本描述编码成向量的函数，接口为 func(list_of_str) -> np.array
        - `client`: 已初始化的 MHACoT 客户端对象
        - `model_name`: str, VLM 模型名称, 用于region mathing
        - `logger`: 日志记录对象，用于记录处理过程中的信息
    '''
    category_name = os.path.basename(h5_path).split('.')[0] # eg. chair
    with h5py.File(h5_path, 'r+') as h5_file:
        instance_keys = list(h5_file.keys())  # chair1, chair2...
        for instance_key in tqdm(instance_keys):
            instance = h5_file[instance_key]
            logger.record('Target: %s (category=%s)', instance_key, category_name)
            original_img_path = os.path.join(query_save_dir, f'{category_name}_{instance_key}_original.png')
            proposal_img_path = os.path.join(query_save_dir, f'{category_name}_{instance_key}_proposal.png')

            # decide whether to peocess the target instance based on the status recorded in database
            top_i = 0
            for top_i in range(top_k):
                status = db.get_instance_data(instance)[top_i]['status']
                frame_idx = db.get_instance_data(instance)[top_i]['frame_idx']
                if status is not 'Done':
                    process_instance(instance, dino_model, original_img_path, proposal_img_path, use_data_link_segs=use_data_link_segs, top_i=top_i)
                    logger.record(f'No existing cluster for {category_name}_{instance_key}_{frame_idx}, initializing...')
                    top_i += 1
                else:
                    print(f'Using existing cluster for {category_name}_{instance_key}')
                    logger.record(f'Using existing cluster for {category_name}_{instance_key}')
                    pass


def main(args):
    logger = store_logs('pipeline_new.log')
    logger.record('Pipeline started with args: %s', args)
    base_dir = args.base_dir
    embedding_type = args.embedding_type
    print('HF requires proxy, turning on clash.')
    proxy_on()
    text_embedding_func = get_text_embedding_options(embedding_type)
    use_data_link_segs = args.use_data_link_segs if args.use_data_link_segs is not None else False
    top_k = args.top_k if args.top_k is not None else 3
    query_save_dir = os.path.join(base_dir, 'vlm_query_imgs')
    os.makedirs(query_save_dir, exist_ok=True)

    # Initialize DINO model
    dino_name = 'dinov2_vitl14'
    dino_model = load_pretrained_dino(dino_name, use_registers=True, torch_path=args.torch_path)
    print(f'Loaded dino model, type {dino_name}')

    # Initialize MHACoT VLM model
    vlm_model_name = args.vlm_model_name
    




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='dataset/h5')
    parser.add_argument('--embedding_type', type=str, default='embeddings_oai', choices=['embeddings_oai', 'embeddings_st'])
    parser.add_argument('--use_data_link_segs', action='store_true')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--torch_path', type=str, default=None, help='Path to torch model cache directory')
    parser.add_argument('--vlm_model_name', type=str, default='qwen/qwen3.5-flash-02-23',
                        help='API VL model name (vision-capable)')
    args = parser.parse_args()

    
