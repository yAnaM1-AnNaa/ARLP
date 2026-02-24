'''
Multi-Hop Affordance Chain-of-Thought (MHACoT)

The CoT contains 4 prompts, intended to reveal the following features per colored region:
1. Where to interact - identify each colored region's functional part
2. Why this region can interact based on geometric shapes
3. The affordance of the object - interaction description per region
4. Further possible affordance - additional interactions + structured output

Adapted for clustered images where each color represents a different object part.
Output format: region_matching dict compatible with pipeline.py

Sample output:
{
    '["sit", "rest", "the area to sit on", "where to relax with comfort"]': 'Red',
    '["support", "stabilize", "the base that supports weight", "where to place feet"]': 'Blue'
}
'''
import ast
import json
import math
import re

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Color palette consistent with unsup-affordance/src/utils/vlm_utils.py
PALETTE = [
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


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def detect_colors(image_path, min_pixels=20):
    """Detect major colors in the clustered image using the same palette as pipeline."""
    img = Image.open(image_path).convert('RGB')
    color_counts = {color_name: 0 for _, color_name in PALETTE}

    for pixel in img.getdata():
        for color_rgb, color_name in PALETTE:
            if pixel == tuple(color_rgb):
                color_counts[color_name] += 1

    detected = [name for name, count in color_counts.items() if count >= min_pixels]
    if 'Black' in detected:
        detected.remove('Black')
    return detected


def init_model(model_name='InternVL2-Llama3-76B'):
    """Load model and tokenizer once for reuse across batch processing."""
    path = f"/root/.cache/modelscope/hub/models/OpenGVLab/{model_name}"

    device_map = split_model(model_name)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    return model, tokenizer


def parse_region_matching(response_text):
    """
    Parse the structured output from Q4 into a region_matching dict.
    Compatible with pipeline.py's parse_lm_output(parse_dict=True).
    """
    # Try to find ANSWER: marker
    answer_match = re.search(r'ANSWER:\s*(\{.*\})', response_text, re.DOTALL)
    if answer_match:
        dict_str = answer_match.group(1)
    else:
        # Fallback: find any dict-like structure
        dict_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(1)
        else:
            print("Failed to find dict structure in response")
            return None

    # Try ast.literal_eval first (same as pipeline)
    try:
        result = ast.literal_eval(dict_str)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # Fallback: try json.loads
    try:
        result = json.loads(dict_str)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    print("Failed to parse region matching output:")
    print(response_text)
    return None


def process_image(model, tokenizer, image_path, object_name, colors=None,
                  original_image_path=None, generation_config=None):
    """
    Process one clustered image with 4-step Chain-of-Thought.

    Args:
        model: loaded InternVL2 model
        tokenizer: loaded tokenizer
        image_path: path to the clustered/proposal image
        object_name: name of the object (e.g., 'chair', 'kettle')
        colors: list of color names detected in the image; auto-detected if None
        original_image_path: optional path to the original (non-clustered) image;
                             when provided, both images are sent to the model
        generation_config: optional dict for generation; defaults to max_new_tokens=1024

    Returns:
        region_matching: dict in format {description_list_str: color_name}, or None on failure
        raw_responses: list of 4 raw response strings from the model
    """
    if generation_config is None:
        generation_config = dict(max_new_tokens=1024, do_sample=True)

    if colors is None:
        colors = detect_colors(image_path)

    if not colors:
        print(f"Warning: No colors detected in {image_path}")
        return None, []

    colors_str = ', '.join(colors)

    # Load images: support both single (proposal only) and dual (original + proposal)
    if original_image_path is not None:
        pv_original = load_image(original_image_path, max_num=12).to(torch.bfloat16).cuda()
        pv_proposal = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        pixel_values = torch.cat([pv_original, pv_proposal], dim=0)
        image_prefix = '<image>\n<image>\n'
        image_desc = (
            f'The first image shows the original {object_name}. '
            f'The second image shows the same {object_name} with different parts highlighted '
            f'in distinct colors: {colors_str}.'
        )
    else:
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        image_prefix = ''
        image_desc = (
            f'This image shows a {object_name} with different parts highlighted '
            f'in distinct colors: {colors_str}.'
        )

    responses = []

    # ---- Q1: Where to interact ----
    question1 = (
        f'{image_prefix}'
        f'{image_desc} '
        f'For each colored region, identify what functional part of the {object_name} it represents '
        f'and determine whether it is a part that directly interacts with people.'
    )
    resp1, history = model.chat(
        tokenizer, pixel_values, question1, generation_config,
        history=None, return_history=True
    )
    responses.append(resp1)

    # ---- Q2: Why based on geometric structure ----
    question2 = (
        f'For each colored region you identified, explain from the geometric structure of the {object_name} '
        f'why that part can interact with people. Give a concise explanation for each color.'
    )
    resp2, history = model.chat(
        tokenizer, pixel_values, question2, generation_config,
        history=history, return_history=True
    )
    responses.append(resp2)

    # ---- Q3: Affordance description ----
    question3 = (
        f'For each colored region of the {object_name}, describe the primary interaction between '
        f'that part and a person, including the interaction type, the specific part of the {object_name}, '
        f'and how a person would physically interact with it.'
    )
    resp3, history = model.chat(
        tokenizer, pixel_values, question3, generation_config,
        history=history, return_history=True
    )
    responses.append(resp3)

    # ---- Q4: Further affordances + structured output ----
    question4 = (
        f'Based on all your analysis, for each colored region of the {object_name}, provide a comprehensive '
        f'set of affordance descriptions for human interaction. Also consider additional common interactions '
        f'beyond those already discussed. For each color, include:\n'
        f'- 1-2 simple action verbs (e.g., "grip", "sit")\n'
        f'- 1-2 action phrases (e.g., "holding to stabilize")\n'
        f'- 1-2 natural language descriptions (e.g., "the area to sit on", "where to relax with comfort")\n\n'
        f'Output your answer STRICTLY as a Python dictionary in the following format:\n'
        f'ANSWER: {{"[\\"verb1\\", \\"verb2\\", \\"phrase1\\", \\"description1\\"]": "Color1", '
        f'"[\\"verb1\\", \\"phrase1\\", \\"description1\\", \\"description2\\"]": "Color2"}}\n\n'
        f'Rules:\n'
        f'- Use double quotes for all strings\n'
        f'- Each key must be a JSON-formatted list of 4-5 description strings\n'
        f'- Each value must be exactly one of: {colors_str}\n'
        f'- Every detected color must appear exactly once as a value\n'
        f'- Do NOT use single quotes. The output must be parseable by Python ast.literal_eval().'
    )
    resp4, history = model.chat(
        tokenizer, pixel_values, question4, generation_config,
        history=history, return_history=True
    )
    responses.append(resp4)

    # Parse structured output
    region_matching = parse_region_matching(resp4)

    return region_matching, responses
