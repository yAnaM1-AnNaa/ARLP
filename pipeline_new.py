import os
import sys
import glob

import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
import ast
import logging
import re
import sqlite3
from pathlib import Path

from src.fusion import create_fusion
from src.cluster import cluster
from src.MHACoT import init_client, process_image, detect_colors

# from utils.utils import find_similarity
from utils.img_utils import grid_visualize, load_pretrained_dino
from utils.file_utils import store_or_update_dataset, save_image, save_response
from utils.vlm_utils import get_text_embedding_options
from utils.file_utils import load_config, store_logs, ResearchSql
from scripts.proposal_to_sim_proj import proposal_to_heatmaps, _slugify
from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPTextModelWithProjection


def parse_description_list(descriptions):
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



def build_text_embedding_json(region_matching: dict, text_embedding_func, logger, category_name: str, instance_key: str, frame_idx: int):
    """Build DB JSON payload for region-matching text embeddings."""
    text_embedding = {}
    for color, descriptions in region_matching.items():
        description_list = parse_description_list(descriptions)
        if not description_list:
            logger.warning(
                'Skip empty descriptions for %s_%s_%s color=%s',
                category_name,
                instance_key,
                frame_idx,
                color,
            )
            continue

        embeddings = []
        for description in description_list:
            embedding = np.asarray(text_embedding_func(description), dtype=np.float32)
            embeddings.append(embedding.tolist())
        text_embedding[color] = embeddings

    if not text_embedding:
        raise ValueError("No text embeddings were generated from region_matching.")
    return text_embedding


def write_text_embeddings_to_h5(instance, embedding_type: str, text_embedding: dict):
    """Keep the legacy HDF5 embedding layout in sync with the SQL payload."""
    if embedding_type not in instance.keys():
        embedding_group = instance.create_group(embedding_type)
    else:
        embedding_group = instance[embedding_type]

    for color, embeddings in text_embedding.items():
        store_or_update_dataset(embedding_group, color, np.asarray(embeddings, dtype=np.float32))


def build_sim_proj_json(
    proposal_img_path: str,
    sim_proj_dir: str,
    category_name: str,
    instance_key: str,
    frame_idx: int,
    region_matching: dict,
    tolerance: float,
    min_pixels: int,
):
    """Generate proposal-mask sim_proj files and return the DB JSON payload."""
    proposal_img_path = Path(proposal_img_path)
    sim_proj_dir = Path(sim_proj_dir)
    sim_proj_dir.mkdir(parents=True, exist_ok=True)

    proposal_img = np.asarray(Image.open(proposal_img_path).convert("RGB"))
    base_name = "_".join(
        _slugify(part)
        for part in (category_name, instance_key, str(frame_idx), "proposal")
    )

    sim_proj = {}
    manifest = {
        "proposal_img": str(proposal_img_path),
        "category_name": category_name,
        "instance_name": instance_key,
        "frame_idx": str(frame_idx),
        "shape": list(proposal_img.shape[:2]),
        "tolerance": tolerance,
        "min_pixels": min_pixels,
        "colors": [],
    }

    for response_color, palette_name, heatmap, pixel_count in proposal_to_heatmaps(
        proposal_img,
        tolerance=tolerance,
        min_pixels=min_pixels,
        vlm_response=region_matching,
    ):
        color_slug = _slugify(response_color)
        npy_path = sim_proj_dir / f"{base_name}_{color_slug}.npy"
        np.save(npy_path, heatmap.astype(np.float32))
        sim_proj[response_color] = str(npy_path.resolve())
        manifest["colors"].append({
            "color": response_color,
            "palette_color": palette_name,
            "heatmap_npy": str(npy_path.resolve()),
            "pixel_count": pixel_count,
            "vlm_response": region_matching.get(response_color),
        })

    if not sim_proj:
        raise ValueError("No sim_proj files were generated from proposal image.")

    manifest_path = sim_proj_dir / f"{base_name}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return sim_proj

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
    logger.record('Start finding best camera angle.')
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
            for top_i in top_k_indices:
                db.add_data(category_name, instance_key, top_i, '', '', '', '')




def process_instance(instance, instance_key: str, 
                     db, dino_model, logger,
                     query_original_path: str,
                     query_proposal_path: str,
                     use_data_link_segs: bool,
                     frame_idx: int):
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
    top_k_indices = []
    instance_rows = db.get_instance_data(instance_key) # [{'name1':'name1, 'topk': topk..},{.}, {.}..]
    for i in range(len(instance_rows)):
        top_k_indices.append(int(instance_rows[i]['frame_idx']))
    query_frame_idx = int(frame_idx)               # use the current selected frame as the query view

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
        frame_idx=frame_idx,  # "Query" viewpoint for the color-coded result
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
    logger.record(f"Saved query original image to: {query_original_path}")
    logger.record(f"Saved proposal (cluster) image to: {query_proposal_path}")


def process_category(h5_path, db, dino_model, query_save_dir, sim_proj_dir, embedding_type, text_embedding_func,
                     client, model_name, logger,
                     use_existing_cluster=True, use_data_link_segs=False, top_k=3,
                     sim_proj_tolerance=8.0, sim_proj_min_pixels=1):
    category_name = os.path.basename(h5_path).split('.')[0] # eg. chair.h5 -> chair
    with h5py.File(h5_path, 'r+') as h5_file:
        instance_keys = list(h5_file.keys())  # chair1, chair2...
        for instance_key in tqdm(instance_keys):
            instance = h5_file[instance_key]
            logger.record(f'Target: {instance_key} (category={category_name})')

            if 'top_k_indices' not in instance:
                find_best_camera_angle(h5_path, db, logger, top_k)
            else:
                top_k_indices = instance['top_k_indices'][:]
                existing_rows = db.get_instance_data(instance_key)
                existing_frame_idxs = {int(str(row['frame_idx'])) for row in existing_rows}

                for top_i in top_k_indices[:top_k]:
                    top_i = int(top_i)
                    if top_i not in existing_frame_idxs:
                        db.add_data(category_name, instance_key, top_i, '', '', '', '')

            instance_rows = db.get_instance_data(instance_key)
            for row in instance_rows[:top_k]:
                status = row['status']
                frame_idx = int(str(row['frame_idx']))
                original_img_path = os.path.join(
                    query_save_dir,
                    f'{category_name}_{instance_key}_{frame_idx}_original.png'
                )
                proposal_img_path = os.path.join(
                    query_save_dir,
                    f'{category_name}_{instance_key}_{frame_idx}_proposal.png'
                )

                if status == 'Done' and row.get('text_embedding') and row.get('sim_proj'):
                    print(f'Using existing complete row for {category_name}_{instance_key}_{frame_idx}')
                    logger.record(f'Using existing complete row for {category_name}_{instance_key}_{frame_idx}, skipping...')
                    continue

                if status == 'Done' and row.get('vlm_response') and row.get('clustered_img_path'):
                    try:
                        region_matching = _normalize_region_matching(json.loads(row['vlm_response']))
                        text_embedding_json = build_text_embedding_json(
                            region_matching,
                            text_embedding_func,
                            logger,
                            category_name,
                            instance_key,
                            frame_idx,
                        )
                        write_text_embeddings_to_h5(instance, embedding_type, text_embedding_json)
                        sim_proj_json = build_sim_proj_json(
                            row['clustered_img_path'],
                            sim_proj_dir,
                            category_name,
                            instance_key,
                            frame_idx,
                            region_matching,
                            tolerance=sim_proj_tolerance,
                            min_pixels=sim_proj_min_pixels,
                        )
                        h5_file.flush()
                        db.update_content(
                            category_name,
                            instance_key,
                            frame_idx,
                            text_embedding=json.dumps(text_embedding_json, ensure_ascii=False),
                            sim_proj=json.dumps(sim_proj_json, ensure_ascii=False),
                            error_msg='',
                        )
                        logger.record('Populated missing training columns for existing Done row %s_%s_%s', category_name, instance_key, frame_idx)
                        continue
                    except Exception as e:
                        db.update_content(category_name, instance_key, frame_idx, status='Failed', error_msg=str(e))
                        logger.exception('Failed to populate training columns for existing row %s_%s_%s', category_name, instance_key, frame_idx)
                        continue

                try:
                    process_instance(
                        instance,
                        instance_key,
                        db,
                        dino_model,
                        logger,
                        original_img_path,
                        proposal_img_path,
                        use_data_link_segs=use_data_link_segs,
                        frame_idx=frame_idx,
                    )
                    db.update_content(
                        category_name,
                        instance_key,
                        frame_idx,
                        clustered_img_path=proposal_img_path,
                    )
                    logger.record(f'No existing cluster for {category_name}_{instance_key}_{frame_idx}, initializing...')

                    colors = detect_colors(proposal_img_path)
                    region_matching, _ = process_image(
                        client,
                        model_name,
                        image_pair_path=[original_img_path, proposal_img_path],
                        object_name=category_name,
                        colors=colors,
                    )
                    region_matching = _normalize_region_matching(region_matching)
                    if not region_matching:
                        logger.warning('Skip %s_%s_%s because normalized region_matching is empty', category_name, instance_key, frame_idx)
                        continue

                    region_matching_json = json.dumps(region_matching, ensure_ascii=False)
                    store_or_update_dataset(instance, f"region_matching_{frame_idx}", region_matching_json)
                    db.update_content(
                        category_name,
                        instance_key,
                        frame_idx,
                        vlm_response=region_matching_json,
                    )
                    logger.record('Region matching for %s_%s_%s: %s', category_name, instance_key, frame_idx, region_matching)

                    text_embedding_json = build_text_embedding_json(
                        region_matching,
                        text_embedding_func,
                        logger,
                        category_name,
                        instance_key,
                        frame_idx,
                    )
                    write_text_embeddings_to_h5(instance, embedding_type, text_embedding_json)

                    sim_proj_json = build_sim_proj_json(
                        proposal_img_path,
                        sim_proj_dir,
                        category_name,
                        instance_key,
                        frame_idx,
                        region_matching,
                        tolerance=sim_proj_tolerance,
                        min_pixels=sim_proj_min_pixels,
                    )

                    h5_file.flush()
                    db.update_content(
                        category_name,
                        instance_key,
                        frame_idx,
                        status='Done',
                        error_msg='',
                        text_embedding=json.dumps(text_embedding_json, ensure_ascii=False),
                        sim_proj=json.dumps(sim_proj_json, ensure_ascii=False),
                    )
                except Exception as e:
                    db.update_content(category_name, instance_key, frame_idx, status='Failed', error_msg=str(e))
                    logger.exception('Frame failed for %s_%s_%s', category_name, instance_key, frame_idx)
                    continue

def main(args):
    db_dir = os.path.dirname(args.db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    db = ResearchSql(os.path.join(db_dir, 'pipeline_info.db'))
    logger = store_logs('pipeline_info', os.path.join(db_dir, 'pipeline_info.log'))
    logger.record('\n\nPipeline started with args: %s', args)

    base_dir = args.base_dir
    embedding_type = args.embedding_type
    print('HF requires proxy, turning on clash.')
    proxy_on()
    text_embedding_func = get_text_embedding_options(embedding_type)
    use_data_link_segs = args.use_data_link_segs if args.use_data_link_segs is not None else False
    top_k = args.top_k if args.top_k is not None else 3
    query_save_dir = os.path.join(base_dir, 'vlm_query_imgs')
    sim_proj_dir = args.sim_proj_dir or os.path.join(base_dir, 'sim_proj')
    os.makedirs(query_save_dir, exist_ok=True)
    os.makedirs(sim_proj_dir, exist_ok=True)

    # Initialize DINO model
    dino_name = 'dinov2_vitl14'
    dino_model = load_pretrained_dino(dino_name, use_registers=True, torch_path=args.torch_path)
    logger.record(f'Loaded dino model, type {dino_name}')

    # Initialize MHACoT VLM model
    vlm_model_name = args.vlm_model_name
    print('OpenAI lib does not accept proxy, turning off clash.')
    proxy_off()
    router_client = init_client()
    logger.record(f'Loaded MHACoT model: {vlm_model_name}.')
    
    # start processing
    for name in os.listdir(base_dir):
        h5_path = os.path.join(base_dir, name)
        if not os.path.isfile(h5_path):
            continue
        if not name.endswith('.h5'):
            continue
        logger.record(f'Processing file: {name}')
        try:
            process_category(
                h5_path,
                db,
                dino_model,
                query_save_dir,
                sim_proj_dir,
                embedding_type,
                text_embedding_func,
                client=router_client,
                model_name=vlm_model_name,
                logger=logger,
                use_existing_cluster=True,
                use_data_link_segs=use_data_link_segs,
                top_k=top_k,
                sim_proj_tolerance=args.sim_proj_tolerance,
                sim_proj_min_pixels=args.sim_proj_min_pixels,
            )
        except Exception as e:
            logger.exception('Category failed for %s', name)
            continue




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='dataset/h5')
    parser.add_argument('--db_path', default='dataset/h5/db/')
    parser.add_argument('--embedding_type', type=str, default='embeddings_oai', choices=['embeddings_oai', 'embeddings_st'])
    parser.add_argument('--use_data_link_segs', action='store_true')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--sim_proj_dir', type=str, default=None, help='Directory to save proposal-mask sim_proj .npy files')
    parser.add_argument('--sim_proj_tolerance', type=float, default=8.0, help='RGB distance tolerance for proposal palette matching')
    parser.add_argument('--sim_proj_min_pixels', type=int, default=1, help='Minimum pixels required per proposal color mask')
    parser.add_argument('--torch_path', type=str, default=None, help='Path to torch model cache directory')
    parser.add_argument('--vlm_model_name', type=str, default='qwen/qwen3.5-flash-02-23',
                        help='API VL model name (vision-capable)')
    args = parser.parse_args()

    main(args)


    
