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

from src.fusion import create_fusion
from src.cluster import cluster
from src.MHACoT import init_client, process_image, detect_colors

# from utils.utils import find_similarity
from utils.img_utils import grid_visualize, load_pretrained_dino
from utils.file_utils import store_or_update_dataset, save_image, save_response
from utils.vlm_utils import get_text_embedding_options

from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPTextModelWithProjection
sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_description_list(descriptions):
    """
    将任意形式的描述字段尽量稳健地解析成 `list[str]`。

    这个函数主要用于处理 VLM 返回的区域描述文本。上游返回格式并不稳定，
    可能是：
    1. Python 的 `list` / `tuple`
    2. 字符串形式的列表，例如 `"['handle', 'seat']"`
    3. 单个普通字符串
    4. 混杂引号、转义字符或格式不完整的文本

    处理策略：
    1. 如果本身就是列表或元组，直接逐项转成字符串并去空白。
    2. 如果是字符串，优先用 `ast.literal_eval` 尝试恢复为 Python 字面量。
    3. 若恢复失败，再用正则提取被引号包裹的片段。
    4. 如果以上都失败，则把整段文本当作一个描述返回。

    参数:
        descriptions:
            待解析的描述内容，类型不固定，通常来自 `region_matching` 的 value。

    返回:
        list[str]:
            清洗后的描述字符串列表；如果输入为空，则返回空列表。
    """
    if isinstance(descriptions, (list, tuple)):
        return [str(item).strip() for item in descriptions if str(item).strip()]

    text = str(descriptions).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, str):
            parsed = parsed.strip()
            return [parsed] if parsed else []
    except (SyntaxError, ValueError):
        pass

    quoted_parts = re.findall(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', text)
    # recovered = []
    # for double_quoted, single_quoted in quoted_parts:
    #     part = double_quoted or single_quoted
    #     part = part.strip()
    #     if part:
    #         recovered.append(part)
    # if recovered:
    #     return recovered

    # return [text]


def _normalize_region_matching(region_matching):
    """
    将 region matching 统一规范为 `{color: [description, ...]}` 结构。

    兼容两种输入：
    1. 新格式：`{"Red": ["...", "..."]}`
    2. 旧格式：`{"['...', '...']": "Red"}`

    这样后续 embedding 流程只需要处理一种稳定的数据结构。
    """
    if not isinstance(region_matching, dict):
        return {}

    normalized = {}
    saw_new_format = False

    # 新格式特征：key 是颜色，value 本身就是描述列表，或者是可解析成列表的字符串。
    for key, value in region_matching.items():
        color = str(key).strip()
        parsed_value = value
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed_value = value

        if isinstance(parsed_value, (list, tuple)):
            descriptions = _parse_description_list(parsed_value)
            if color and descriptions:
                normalized[color] = descriptions
                saw_new_format = True

    if saw_new_format:
        return normalized

    # 兼容旧格式：key 是描述列表字符串，value 是颜色。
    for descriptions, color in region_matching.items():
        color_str = str(color).strip()
        description_list = _parse_description_list(descriptions)
        if color_str and description_list:
            normalized[color_str] = description_list

    return normalized


def proxy_off():
    """
    关闭当前 Python 进程里的代理环境变量。

    作用:
    1. 删除常见的 HTTP/HTTPS/ALL_PROXY 大小写变量。
    2. 让后续网络请求绕过本地代理设置。

    这个函数在本脚本里主要用于切换到不兼容代理环境的组件，
    例如部分 OpenAI 相关客户端初始化阶段。
    """
    for var in ['http_proxy', 'https_proxy', 'all_proxy',
                'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
        os.environ.pop(var, None)
    print("😼 已关闭代理环境")

def proxy_on(proxy_url="http://127.0.0.1:7890"):
    """
    为当前 Python 进程设置代理环境变量。

    作用:
    1. 设置 `http_proxy`、`https_proxy`、`all_proxy`
    2. 让后续依赖这些环境变量的下载或模型加载逻辑走指定代理

    参数:
        proxy_url:
            代理地址，默认指向本机 `7890` 端口。
    """
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    os.environ['all_proxy'] = proxy_url
    print("🚀 代理已开启")

def find_best_camera_angle(h5_file_path, top_k=3):
    """
    为一个类别对应的 HDF5 文件挑选“最能代表该物体”的 Top-K 视角。

    背景:
    同一个实例通常有多张不同相机视角的 RGB 图像，但并不是每个视角都适合
    做后续的聚类可视化和 VLM 区域理解。这里使用 CLIP 计算“图像视角”与
    “类别名文本”之间的相似度，选出最相关的若干帧。

    工作流程:
    1. 从文件名中推断类别名，例如 `chair.h5 -> chair`
    2. 使用 CLIP 文本编码器对类别名生成文本特征
    3. 遍历每个实例的所有 RGB 帧，提取图像特征
    4. 计算每一帧与类别文本之间的余弦相似度
    5. 选出相似度最高的 `top_k` 帧
    6. 将结果写回原始 HDF5 文件

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
        """计算一组图像特征与单个查询特征之间的余弦相似度。"""
        similarity = np.dot(features_flat , query_feature) / (np.linalg.norm(features_flat, axis=1) * np.linalg.norm(query_feature))
        return similarity.reshape(-1, 1)

    print('HF requires proxy, turning on clash by excuting clashon')
    os.system("clashoff")
    model_vision = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # currently assuming the category name is the file name
    category_name = os.path.basename(h5_file_path).split('.')[0]
    in_text = tokenizer([category_name], padding=True, return_tensors="pt")
    text_embeds = model_text(**in_text).text_embeds
    text_embeds = text_embeds.squeeze(0).detach().numpy()
    
    # Dictionary to store similarities and top-k frame indices for each instance
    instance_data = {}

    # First pass: read the file and compute best frames
    with h5py.File(h5_file_path, 'r') as h5_file:
        instance_keys = list(h5_file.keys())
        
        for instance_key in instance_keys:
            instance = h5_file[instance_key]

            # get the rgb images
            rgb_images = instance['rgb'][:]

            all_image_features = []
            for frame_idx in range(rgb_images.shape[0]):
                rgb_image = rgb_images[frame_idx]
                inputs = processor(images=rgb_image, return_tensors="pt")
                image_features = model_vision.get_image_features(**inputs)
                image_features = image_features.squeeze(0).detach().numpy()
                all_image_features.append(image_features)

            all_image_features = np.array(all_image_features)
            clip_similarities = find_similarity(all_image_features, text_embeds)
            print(clip_similarities.shape)
            
            # Get top-k frame indices (ordered by similarity)
            top_k_indices = np.argsort(clip_similarities.flatten())[-top_k:][::-1]  # reverse to get descending order
            print(f"Instance {instance_key} top-k frames: {top_k_indices}")
            print(f"Top-k similarities: {clip_similarities[top_k_indices].flatten()}")
            
            # Store both similarities and top-k indices
            instance_data[instance_key] = {
                'clip_similarities': clip_similarities.flatten(),
                'top_k_indices': top_k_indices
            }

    # Second pass: update the file with the similarities and top-k indices
    with h5py.File(h5_file_path, 'r+') as h5_file:
        for instance_key, data in instance_data.items():
            # Store CLIP similarities
            store_or_update_dataset(h5_file[instance_key], 'clip_similarities', data['clip_similarities'])

            # Store top-k indices
            store_or_update_dataset(h5_file[instance_key], 'top_k_indices', data['top_k_indices'])


def process_instance(
    instance: h5py.Group,
    dinov2,
    query_original_path: str,
    query_proposal_path: str,
    use_data_link_segs: bool = False,
    top_k: int = 3
):
    """
    处理单个实例，生成 3D 融合结果、聚类结果及后续 VLM 所需的中间数据。

    这是整条流水线里的核心步骤之一，负责把“多视角 2D 观测”整理成
    “可用于语义理解的 3D 聚类表达”。执行后，单个实例会多出一批可复用的
    HDF5 字段，同时还会在磁盘上保存两张给 VLM 使用的配对图片。

    主要流程:
    1. 读取 `top_k_indices`，确定最佳视角和查询帧
    2. 调用 `create_fusion` 构建 3D 融合对象，并逐帧融合
    3. 在 3D 空间上执行聚类，得到每个点/区域的簇标签
    4. 聚合每个簇的特征，建立“颜色名 -> 聚类特征”映射
    5. 把每个簇的相似度重新投影回 Top-K 图像平面，生成热力图
    6. 将聚类相关结果写回 HDF5
    7. 导出原图和颜色编码后的 proposal 图，供 VLM 做区域匹配

    输入实例中预期已有的数据:
        - `rgb`: 多视角彩色图
        - `depth`: 深度图
        - `link_segs`: 连通/部件分割信息
        - `intrinsics` / `extrinsics`: 相机参数
        - `features`: 每像素或每点特征
        - `top_k_indices`: 已经选好的最佳视角索引

    本函数新增/更新的数据:
        - `color_label_names`: 每个聚类对应的颜色标签名
        - `color_name_features`: 每个颜色标签对应的聚类特征
        - `similarity_projections`: 每个簇投影到 Top-K 帧上的相似度图
        - `top_k_rgb`: Top-K 视角的原始 RGB 图像

    参数:
        instance:
            当前实例对应的 HDF5 group。
        dinov2:
            DINOv2 特征提取模型，用于构建融合表达。
        query_original_path:
            保存原始查询图的路径。
        query_proposal_path:
            保存聚类 proposal 图的路径。
        use_data_link_segs:
            是否优先使用数据中已有的 `link_segs`。
        top_k:
            使用多少个最佳视角做投影和保存。

    返回:
        无返回值。结果直接写入 HDF5，并在磁盘保存图片。
    """

    # -------------------------------------------------------------------------
    # 1) Retrieve the top-3 frames from HDF5
    # -------------------------------------------------------------------------
    
    top_k_indices = instance["top_k_indices"][:]  # shape (top_k,)
    query_frame_idx = top_k_indices[0]            # the "main" query frame

    # -------------------------------------------------------------------------
    # 2) Create the 3D fusion object from the HDF5 group and fuse frames
    # -------------------------------------------------------------------------
    fusion = create_fusion(instance, dinov2, use_data_link_segs=use_data_link_segs)
    for i in range(fusion.num_frames):
        _ = fusion.fuse_frame(i)

    # -------------------------------------------------------------------------
    # 3) Run clustering on the 3D fusion
    #    (pca_dim, use_loc, etc. are from your pipeline needs)
    # -------------------------------------------------------------------------
    cluster_results, proposal_img, used_colors = cluster(
        fusion,
        pca_dim=3,
        use_loc=0.0,
        frame_idx=query_frame_idx,  # "Query" viewpoint for the color-coded result
        return_color_names=True,
        proj_3d=True,
        min_num_clusters=5,
        enable_per_link_cluster=True
    )

    # -------------------------------------------------------------------------
    # 4) Aggregate cluster features; build color->feature mapping
    # -------------------------------------------------------------------------
    unique_labels = list(np.unique(cluster_results))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    aggr_cluster_feat, similarities = fusion.aggregate_cluster_feature(
        cluster_results,
        return_similarities=True
    )

    # "color_name_feat" was originally a dict: { used_colors[label]: features_vector }
    color_name_feat = {}
    for label in unique_labels:
        color_str = used_colors[label]
        color_name_feat[color_str] = aggr_cluster_feat[label]

    # -------------------------------------------------------------------------
    # 5) Build similarity projections for top-3 frames only
    #    We'll collect them into one big array: shape (num_clusters, 3, H, W)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 6) Save new data to the HDF5 group using store_or_update_dataset
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 7) Save the original (query) and proposal images to disk
    # -------------------------------------------------------------------------
    original_img = fusion.images[query_frame_idx]  # shape (H, W, 3)
    save_image(original_img, query_original_path)
    save_image(proposal_img, query_proposal_path)

    print("Saved query original image to:", query_original_path)
    print("Saved proposal (cluster) image to:", query_proposal_path)
    print("Stored color_label_names, color_name_features, and similarity_projections in HDF5.")


def process_category(category_h5_path, dinov2, query_save_dir, embedding_type, text_embedding_func,
                     client, model_name, logger,
                     use_existing_cluster=True, use_data_link_segs=False, top_k=3):
    """
    处理一个类别级别的 HDF5 文件，完成该类别下所有实例的聚类、区域匹配和文本嵌入。

    这个函数负责把前面的视觉中间结果与后面的语言理解步骤串起来。对一个类别
    文件中的每个实例，它会：

    1. 调用 `process_instance` 生成聚类结果和 VLM 查询图片
    2. 使用 `detect_colors` 从 proposal 图中提取颜色标签
    3. 调用 `process_image` 让 MHACoT/VLM 根据原图和 proposal 图做区域匹配
    4. 将 VLM 返回的 `region_matching` 写回 HDF5
    5. 把每个颜色区域的文本描述编码成向量并保存

    为什么按类别文件处理:
    一个 `.h5` 文件通常对应一个类别，例如 `chair.h5`。文件内包含多个实例，
    比如 `chair1`、`chair2` 等。这样便于按类批量处理和后续检索。

    参数:
        category_h5_path:
            当前类别 HDF5 文件路径。
        dinov2:
            已加载的 DINOv2 模型。
        query_save_dir:
            保存 VLM 查询图像的目录。
        embedding_type:
            文本嵌入存储到实例中的分组名，例如 `embeddings_oai`。
        text_embedding_func:
            文本转向量函数，输入一段描述，输出一个 embedding。
        client:
            VLM 客户端。
        model_name:
            要调用的视觉语言模型名称。
        logger:
            日志记录器。
        use_existing_cluster:
            若已有聚类结果和输出图片，是否尽量复用，避免重复计算。
        use_data_link_segs:
            是否使用数据中现成的 link segmentation。
        top_k:
            使用的最佳视角数。

    返回:
        无返回值。结果直接写入原始 HDF5 文件。
    """
    category_name = os.path.basename(category_h5_path).split('.')[0] # eg. chair
    with h5py.File(category_h5_path, 'r+') as h5_file:
        instance_keys = list(h5_file.keys())  # chair1, chair2...

        for instance_key in tqdm(instance_keys):
            instance = h5_file[instance_key]

            # Emit an early log line so the log file is created even if VLM crashes later.
            logger.info('Target: %s (category=%s)', instance_key, category_name)

            # process the instance
            query_original_path = os.path.join(query_save_dir, f'{category_name}_{instance_key}_original.png')
            query_proposal_path = os.path.join(query_save_dir, f'{category_name}_{instance_key}_proposal.png')

            if not use_existing_cluster or not os.path.exists(query_original_path) or not os.path.exists(query_proposal_path):
                process_instance(instance, dinov2, query_original_path, query_proposal_path, use_data_link_segs=use_data_link_segs, top_k=top_k)
            else:
                print(f'Using existing cluster for {category_name}_{instance_key}')

            # skip only when region_matching exists and letter count is sufficiently large
            if use_existing_cluster and "region_matching" in instance:
                letter_count = 0
                try:
                    region_matching_val = instance["region_matching"][()]
                    if isinstance(region_matching_val, bytes):
                        region_matching_text = region_matching_val.decode("utf-8", errors="ignore")
                    else:
                        region_matching_text = str(region_matching_val)
                    letter_count = len(re.findall(r"[A-Za-z]", region_matching_text))
                except Exception:
                    letter_count = 0

                if letter_count > 1000:
                    print(f'Skipping {category_name}_{instance_key}: region_matching letter_count={letter_count} > 1000')
                    logger.info(
                        'Skipping %s_%s: region_matching letter_count=%s > 1000',
                        category_name,
                        instance_key,
                        letter_count,
                    )
                    continue

            # query MHACoT and save the region matching/description
            colors = detect_colors(query_proposal_path)
            try:
                region_matching, cot_resoponse = process_image(
                    client,
                    model_name,
                    image_pair_path=[query_original_path, query_proposal_path],
                    object_name=category_name,
                    colors=colors,
                )
                save_response(cot_resoponse, logger)
                logger.info('Region matching for %s_%s: %s', category_name, instance_key, region_matching)
            except Exception:
                logger.exception('MHACoT failed for %s_%s', category_name, instance_key)
                for handler in logger.handlers:
                    try:
                        handler.flush()
                    except Exception:
                        pass
                raise
            # for i, resp in enumerate(cot_responses, 1):
            #     print(f"  CoT Step {i}: {resp[:200]}...")
            if region_matching is None:
                print(f'No region matching found for {category_name}_{instance_key}')
                continue
            region_matching_json = json.dumps(region_matching)
            store_or_update_dataset(instance, "region_matching", region_matching_json)

            # process the region matching and get the text embedding
            region_matching = _normalize_region_matching(region_matching)
            if not region_matching:
                logger.warning('Skip %s_%s because normalized region_matching is empty', category_name, instance_key)
                continue

            embedding_dict = {}
            for color, descriptions in region_matching.items():
                description_list = _parse_description_list(descriptions)
                if not description_list:
                    logger.warning(
                        'Skip empty descriptions for %s_%s color=%s',
                        category_name,
                        instance_key,
                        color,
                    )
                    continue
                embedding = [text_embedding_func(description) for description in description_list]
                embedding = np.array(embedding)
                embedding_dict[color] = embedding

            # save the embedding dictionary
            # create a new dataset/group for embeddings
            if embedding_type not in instance.keys():
                embedding_group = instance.create_group(embedding_type)
            else:
                embedding_group = instance[embedding_type]

            # for each color, save the embedding array
            for color, embedding in embedding_dict.items():
                store_or_update_dataset(embedding_group, color, embedding)

            # save the instance
            h5_file.flush()

    h5_file.close()          


def main(args):
    """
    脚本入口函数，负责初始化环境、模型和全局配置，然后逐类别执行处理流程。

    整体职责:
    1. 初始化日志
    2. 读取命令行参数
    3. 打开代理以支持部分模型/依赖下载
    4. 初始化文本嵌入函数
    5. 收集需要处理的类别 HDF5 文件
    6. 加载 DINOv2 模型
    7. 关闭代理并初始化 MHACoT/VLM 客户端
    8. 对每个类别文件调用 `process_category`

    输入:
        args:
            命令行参数对象，包含数据目录、嵌入类型、类别过滤条件、
            top_k、torch 缓存路径以及 VLM 模型名等配置。

    返回:
        无返回值。该函数以“读写 HDF5 + 生成图片 + 记录日志”的方式完成整个 pipeline。
    """
    logging.basicConfig(
        filename=f'{args.base_dir}/vlm_response.log',
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)
    logger.info('Pipeline started. base_dir=%s embedding_type=%s vlm_model_name=%s', args.base_dir, args.embedding_type, args.vlm_model_name)
    # 读取h5文件例如chair.h5
    # 文件通常包含多个实例（比如不同的椅子），每个实例下存储了多视角的数据
    # rgb(N, H, W, 3)，depth(N, H, W)，intrinsics / extrinsics: 相机内参和外参
    base_dir = args.base_dir
    embedding_type = args.embedding_type
    print('HF requires proxy, turning on clash.')
    proxy_on()
    text_embedding_func = get_text_embedding_options(embedding_type)
    category_names = args.category_names
    use_data_link_segs = args.use_data_link_segs if args.use_data_link_segs is not None else False
    top_k = args.top_k if args.top_k is not None else 3

    if category_names:  # user-specified subset
        if isinstance(category_names, str):
            category_names = [category_names]
        all_category_h5_paths = []
        for cat_name in category_names:
            category_h5_path = os.path.join(base_dir, f'{cat_name}.h5')
            if not os.path.exists(category_h5_path):
                raise ValueError(f'Category {cat_name} not found in {base_dir}/h5')
            all_category_h5_paths.append(category_h5_path)
    else:  # no specific category supplied -> process entire directory
        all_category_h5_paths = glob.glob(os.path.join(base_dir, '*.h5'))

    query_save_dir = os.path.join(base_dir, 'vlm_query_imgs')
    os.makedirs(query_save_dir, exist_ok=True)

    # Initialize DINO model
    dino_name = 'dinov2_vitl14'
    dinov2 = load_pretrained_dino(dino_name, use_registers=True, torch_path=args.torch_path)
    print(f'Loaded dinov2 model, type {dino_name}')

    # Initialize MHACoT VLM model
    vlm_model_name = args.vlm_model_name
    print(f'Loading MHACoT model: {vlm_model_name}')
    print('OpenAI lib does not accept proxy, turning off clash.')
    proxy_off()
    router_client = init_client()
    print(f'Loaded MHACoT model.')

    if all_category_h5_paths == []:
        print('No h5 found.')
    else:
        print(all_category_h5_paths)

    for category_h5_path in all_category_h5_paths:
        print(f'Processing {category_h5_path}')

        process_category(category_h5_path, dinov2, query_save_dir, embedding_type, text_embedding_func,
                         client=router_client, model_name=vlm_model_name, logger=logger,
                         use_data_link_segs=use_data_link_segs, top_k=top_k)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='dataset/h5')
    parser.add_argument('--embedding_type', type=str, default='embeddings_oai', choices=['embeddings_oai', 'embeddings_st'])
    parser.add_argument('--use_data_link_segs', action='store_true')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--category_names', '-c',
                        nargs='+', dest='category_names', default=None,
                        help='Optional: one or more category names to process. If omitted, '
                             'all categories present in <base_dir>/h5 are processed.')
    parser.add_argument('--torch_path', type=str, default=None, help='Path to torch model cache directory')
    parser.add_argument('--vlm_model_name', type=str, default='qwen/qwen3.5-flash-02-23',
                        help='API VL model name (vision-capable)')
    args = parser.parse_args()

    main(args)



    
