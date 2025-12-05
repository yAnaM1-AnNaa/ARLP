import os
import sys
import glob

import h5py
import numpy as np
from tqdm import tqdm
import json
import ast

from fusion import create_fusion
from cluster import cluster
from get_vlm_response import get_end_to_end_matching

# from utils.utils import find_similarity
from utils.img_utils import grid_visualize, load_pretrained_dino
from utils.file_utils import store_or_update_dataset, save_image
from utils.vlm_utils import get_text_embedding_options

from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPTextModelWithProjection
sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_best_camera_angle(h5_file_path, top_k=3):
    """
    For each instance in the h5 file, find the camera angle that has largest CLIP cosine similarity to the instance description (views that are most similar to the instance description).
    Input:
        h5_file_path: path to the h5 file
    No output, but will save the CLIP similarities and top-k frame indices to the h5 file.
    并不是所有拍摄角度都能清楚地展示物体。这个阶段利用 CLIP 模型找出最能代表该物体类别的“最佳视角”（Top-K）。
    对于h5文件中的每个实例，找到与实例描述具有最大CLIP余弦相似性的摄像机角度
    会将CLIP相似度和top-k帧索引保存到h5文件中
    """
    def find_similarity(features_flat, query_feature):
        similarity = np.dot(features_flat , query_feature) / (np.linalg.norm(features_flat, axis=1) * np.linalg.norm(query_feature))
        return similarity.reshape(-1, 1)

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
    Create a fused 3D representation from HDF5, cluster it, save
    color-based features and top-3-frame similarity projections
    back into the HDF5 group. Also save the query original and
    proposal images to disk.
    从HDF5创建一个融合的3D表示，将其聚类，
    将基于颜色的特征和顶部top_k帧的相似性投影保存回HDF5组。
    还要将查询原始映像和建议映像保存到磁盘
    得到一个带有丰富语义特征的 3D 点云。
    Parameters
    ----------
    instance : h5py.Group
        An HDF5 group with datasets:
            - rgb (N, H, W, 3)
            - depth (N, H, W)
            - link_segs (N, H, W)
            - intrinsics (N, 3, 3)
            - extrinsics (N, 4, 4)
            - features (N, H, W, 1024)
            - clip_similarities (N)
            - top_k_indices (top_k,)
        We will add new datasets here, 
            - similarity_projections (num_clusters, top_k, H, W) # similarity projections for top-k frames
            - color_label_names (num_clusters,)
            - color_name_features (num_clusters, D)
    dinov2 : object
        Your DINO model or equivalent feature extractor object.
    query_original_path : str
        Local file path to save the original query image.
    query_proposal_path : str
        Local file path to save the proposal (cluster) image.
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


def process_category(category_h5_path, dinov2, query_save_dir, embedding_type, text_embedding_func,use_existing_cluster=True, use_data_link_segs=False, top_k=3):
    """
    For each instance in the h5 file, will do some processing for fusion and clustering.
    Assumes store the new data in the same h5 group. Write to the same file.
    """
    category_name = os.path.basename(category_h5_path).split('.')[0]
    with h5py.File(category_h5_path, 'r+') as h5_file:
        instance_keys = list(h5_file.keys())

        for instance_key in tqdm(instance_keys):
            instance = h5_file[instance_key]

            # process the instance
            query_original_path = os.path.join(query_save_dir, f'{category_name}_{instance_key}_original.png')
            query_proposal_path = os.path.join(query_save_dir, f'{category_name}_{instance_key}_proposal.png')

            if not use_existing_cluster or not os.path.exists(query_original_path) or not os.path.exists(query_proposal_path):
                process_instance(instance, dinov2, query_original_path, query_proposal_path, use_data_link_segs=use_data_link_segs, top_k=top_k)
            else:
                print(f'Using existing cluster for {category_name}_{instance_key}')

            # query VLM and save the region matching
            region_matching = get_end_to_end_matching(obj=category_name, query_path=query_save_dir, query_prefix=[f"{category_name}_{instance_key}"], use_vlm='gpt')
            print(instance_key, region_matching)
            if region_matching is None:
                print(f'No region matching found for {category_name}_{instance_key}')
                continue
            region_matching_json = json.dumps(region_matching)
            store_or_update_dataset(instance, "region_matching", region_matching_json)

            # process the region matching and get the text embedding
            embedding_dict = {}
            for descriptions, color in region_matching.items():
                description_list = ast.literal_eval(descriptions)
                embedding = [text_embedding_func(description) for description in description_list]
                embedding = np.array(embedding)
                # map the color to the embedding array
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
    Iterates over each h5 file (corresponding to a category) in the base directory, and processes each file.
    """
    # 读取h5文件例如chair.h5
    # 文件通常包含多个实例（比如不同的椅子），每个实例下存储了多视角的数据
    # rgb(N, H, W, 3)，depth(N, H, W)，intrinsics / extrinsics: 相机内参和外参
    base_dir = args.base_dir
    if not os.path.exists(os.path.join(base_dir, 'h5')):
        raise FileNotFoundError(f"h5 folder not found in {base_dir}, please pass in the parent of the h5 folder")
    embedding_type = args.embedding_type
    text_embedding_func = get_text_embedding_options(embedding_type)
    category_names = args.category_names
    use_data_link_segs = args.use_data_link_segs if args.use_data_link_segs is not None else False
    top_k = args.top_k if args.top_k is not None else 3

    if category_names:  # user-specified subset
        if isinstance(category_names, str):
            category_names = [category_names]
        all_category_h5_paths = []
        for cat_name in category_names:
            category_h5_path = os.path.join(base_dir, 'h5', f'{cat_name}.h5')
            if not os.path.exists(category_h5_path):
                raise ValueError(f'Category {cat_name} not found in {base_dir}/h5')
            all_category_h5_paths.append(category_h5_path)
    else:  # no specific category supplied -> process entire directory
        all_category_h5_paths = glob.glob(os.path.join(base_dir, 'h5', '*.h5'))

    query_save_dir = os.path.join(base_dir, 'vlm_query_imgs')
    os.makedirs(query_save_dir, exist_ok=True)

    dinov2 = load_pretrained_dino('dinov2_vits14', use_registers=True, torch_path=args.torch_path)

    for category_h5_path in all_category_h5_paths:
        print(f'Processing {category_h5_path}')

        process_category(category_h5_path, dinov2, query_save_dir, embedding_type, text_embedding_func, use_data_link_segs=use_data_link_segs, top_k=top_k)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--embedding_type', type=str, default='embeddings_oai', choices=['embeddings_oai', 'embeddings_st'])
    parser.add_argument('--use_data_link_segs', action='store_true')
    parser.add_argument('--top_k', type=int, default=3)

    parser.add_argument('--category_names', '-c',
                        nargs='+', dest='category_names', default=None,
                        help='Optional: one or more category names to process. If omitted, '
                             'all categories present in <base_dir>/h5 are processed.')
    parser.add_argument('--torch_path', type=str, default=None, help='Path to torch model cache directory')
    args = parser.parse_args()

    main(args)



    
