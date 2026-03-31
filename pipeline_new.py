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

def find_best_camera_angle(dataset_path, top_k=5):
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
    model_text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # the rgb photos are stored under the following path: 
    # {object_name}.h5/{instanec_name}/rgb: shape=(14, 378, 378, 3)
    category_name = os.path.basename(h5_file_path).split('.')[0]
    