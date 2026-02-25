import os
import sys
import glob

import h5py
import numpy as np
from tqdm import tqdm
import json
import ast
import subprocess

from src.fusion import create_fusion
from src.cluster import cluster
from src.MHACoT import init_client, process_image, detect_colors

# from utils.utils import find_similarity
from utils.img_utils import grid_visualize, load_pretrained_dino
from utils.file_utils import store_or_update_dataset, save_image
from utils.vlm_utils import get_text_embedding_options
from pipeline import process_category, process_instance, process_image

def proxy_off():
    """关闭代理 - 纯 Python 实现"""
    for var in ['http_proxy', 'https_proxy', 'all_proxy',
                'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
        os.environ.pop(var, None)
    print("😼 已关闭代理环境")

def proxy_on(proxy_url="http://127.0.0.1:7890"):
    """开启代理 - 纯 Python 实现"""
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    os.environ['all_proxy'] = proxy_url
    print("🚀 代理已开启")

def main(args):
    base_dir = args.base_dir
    category_names = args.category_names
    embedding_type = args.embedding_type
    text_embedding_func = get_text_embedding_options(embedding_type)
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
    print('Closing proxy.')
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
                         client=router_client, model_name=vlm_model_name,
                         use_data_link_segs=use_data_link_segs, top_k=top_k)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./dataset/test')
    parser.add_argument('--embedding_type', type=str, default='embeddings_oai', choices=['embeddings_oai', 'embeddings_st'])
    parser.add_argument('--use_data_link_segs', action='store_true')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--category_names', '-c',
                        nargs='+', dest='category_names', default=None,
                        help='Optional: one or more category names to process. If omitted, '
                                'all categories present in <base_dir>/h5 are processed.')
    parser.add_argument('--torch_path', type=str, default=None, help='Path to torch model cache directory')
    parser.add_argument('--vlm_model_name', type=str, default='opengvlab/internvl3-78b',
                        help='API VL model name (vision-capable)')
    args = parser.parse_args()

    main(args)