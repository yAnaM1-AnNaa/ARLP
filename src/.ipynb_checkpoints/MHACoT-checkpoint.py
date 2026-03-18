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
import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time
import subprocess

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


def init_client():
    """Prepare OpenRouter API config for reuse across batch processing."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    site_url = os.getenv('OPENROUTER_SITE_URL')
    site_name = os.getenv('OPENROUTER_SITE_NAME')
    if not api_key:
        raise EnvironmentError('OPENROUTER_API_KEY is not set')

    client = OpenAI(
        api_key = api_key,
        base_url = base_url
    )

    return client


def encode_image_as_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        mime = 'image/jpeg'
    elif ext == '.png':
        mime = 'image/png'
    elif ext == '.webp':
        mime = 'image/webp'
    else:
        mime = 'image/jpeg'
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f'data:{mime};base64,{data}'


def vlm_response(
    client: OpenAI,
    model_name: str,
    prompt,
    encoded_images: List[str],  # 预编码的图像数据
    history: Optional[List[Dict[str, str]]] = None,
    generation_config: Optional[Dict[str, Any]] = None) -> Tuple[str, List[Dict[str, str]]]:
    # Ensure prompt is always a plain string, even if a tuple/list was passed accidentally.
    if isinstance(prompt, (list, tuple)):
        prompt = '\n'.join(str(p) for p in prompt)
    else:
        prompt = str(prompt)
    content = [{'type': 'text', 'text': prompt}]

    # 使用预编码的图像数据
    for encoded_image in encoded_images:
        content.append({'type': 'image_url', 'image_url': {'url': encoded_image}})

    messages: List[Dict[str, Any]] = []
    if history:
        messages.extend(history)
    messages.append({'role': 'user', 'content': content})

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages)
            raw_content = completion.choices[0].message.content
            if raw_content is None:
                text = None
            elif isinstance(raw_content, str):
                text = raw_content
            elif isinstance(raw_content, list):
                text_parts = []
                for item in raw_content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text' and item.get('text') is not None:
                            text_parts.append(str(item['text']))
                    else:
                        text_parts.append(str(item))
                text = '\n'.join(text_parts).strip() or None
            else:
                text = str(raw_content)
            break
        except Exception as e:
            if attempt < 2:  # 如果不是最后一次尝试
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                raise RuntimeError(f'OpenRouter API error: {str(e)}')

    new_history = list(history) if history else []
    new_history.append({'role': 'user', 'content': prompt})
    new_history.append({'role': 'assistant', 'content': text})
    return text, new_history


def parse_region_matching(response_text):
    """
    Parse the structured output from Q4 into a region_matching dict.
    Compatible with pipeline.py's parse_lm_output(parse_dict=True).
    """
    if response_text is None:
        print("Failed to parse region matching: response_text is None")
        return None
    if not isinstance(response_text, (str, bytes)):
        response_text = str(response_text)
    if isinstance(response_text, bytes):
        response_text = response_text.decode("utf-8", errors="ignore")

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

    # Try standard parsing first (works when format is correct: str keys)
    try:
        result = ast.literal_eval(dict_str)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError, TypeError):
        pass

    try:
        result = json.loads(dict_str)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Handle inverted format: {[list]: "Color", ...} -> {"Color": [list]}
    # The VLM sometimes returns list keys which are unhashable.
    # Extract pairs by finding each [...] : "Color" pattern.
    pair_pattern = re.compile(
        r'(\[.*?\])\s*:\s*"([^"]+)"', re.DOTALL
    )
    pairs = pair_pattern.findall(dict_str)
    if pairs:
        result = {}
        for list_str, color in pairs:
            try:
                desc_list = ast.literal_eval(list_str)
                result[str(desc_list)] = color
            except (ValueError, SyntaxError):
                result[list_str] = color
        if result:
            return result

    print("Failed to parse region matching output:")
    print(response_text)
    return None


def process_image(client, model_name, image_pair_path, object_name, colors=None,
                  generation_config=None):
    """
    Process one clustered image with 4-step Chain-of-Thought.

    Args:
        client: OpenAI client instance
        image_path: path to the clustered/proposal image
        object_name: name of the object (e.g., 'chair', 'kettle')
        colors: list of color names detected in the image; auto-detected if None
        generation_config: optional dict for generation; defaults to max_new_tokens=1024

    Returns:
        region_matching: dict in format {description_list_str: color_name}, or None on failure
        raw_responses: list of 4 raw response strings from the model
    """
    if generation_config is None:
        generation_config = dict(max_new_tokens=1024, do_sample=True)

    # Normalize generation_config for OpenRouter/OpenAI-style APIs
    if 'max_tokens' not in generation_config and 'max_new_tokens' in generation_config:
        generation_config['max_tokens'] = generation_config.pop('max_new_tokens')
    generation_config.pop('do_sample', None)

    if colors is None:
        colors = detect_colors(image_pair_path[1])

    colors_str = ', '.join(colors)

    # 预编码图像数据，避免重复加载
    encoded_images = [encode_image_as_data_url(path) for path in image_pair_path]

    # Prepare images for OpenRouter API
    image_desc = (
        f'The first image shows the original {object_name}. '
        f'The second image shows the same {object_name} with different parts highlighted '
        f'in distinct colors: {colors_str}.'
    )


    # ---- Q1: Where to interact ----
    question1 = (
        f'{image_desc} '
        f'For each colored region, identify what functional part of the {object_name} it represents '
        f'and determine whether it directly interacts with people.'
    )
    resp1, history = vlm_response(
        client, model_name, question1, encoded_images, history=None, generation_config=generation_config
    )


    # ---- Q2: Why based on geometric structure ----
    question2 = (
        f'For each colored region you identified, explain from the geometric structure of the {object_name} '
        f'why this part can interact with people. Give a concise explanation for each color.'
    )
    resp2, history = vlm_response(
        client, model_name, question2, encoded_images, history=history, generation_config=generation_config
    )

    # ---- Q3: Affordance description ----
    question3 = (
        f'For each colored region of the {object_name}, describe the primary(most popular and possible) interaction between '
        f'that part and a person, including the interaction type, the specific part of the {object_name}, '
        f'and how a person would physically interact with it.'
    )
    resp3, history = vlm_response(
        client, model_name, question3, encoded_images, history=history, generation_config=generation_config
    )


    # ---- Q4: Further affordances + structured output ----
    question4 = (
        f'Based on all your analysis, for each colored region of the {object_name}, provide a comprehensive '
        f'set of affordance descriptions for human interaction. Also consider additional common interactions '
        f'beyond those already discussed.' )
    resp4, history = vlm_response(
        client, model_name, question4, encoded_images, history=history, generation_config=generation_config
    )

    # ----  structured output ----
    question5 = (
        f'Now you should structure your output for each color based on your history responses as follows:\n'
        f'- Part1: Where to interact (according to response 1)\n'
        f'- Part2: Why it can interact based on geometric structure (according to response 2)\n'
        f'- Part3: Describe the basic affordance (according to response 3)\n'
        f'- Part4: Further possible affordance (e.g., "the area to sit on", "where to relax with comfort")\n\n'
        f'Output your answer STRICTLY as a Python dictionary in the following format:\n'
        f'ANSWER FORMAT:\n'
        '{"[\\"The region with affordance is The rim.\\", \\"This is the top edge of the beaker.\\", '
        '\\"The affordance is Pouring. This region allows liquid to be poured into or out of the beaker.\\", '
        '\\"Further possible affordance: 1. Observation. 2. Hold.\\"]" : "Red", ...}\n\n'
        f'Rules:\n'
        f'- The content must be parseable by Python ast.literal_eval().\n'
        f'- Use double quotes for all strings; escape inner double quotes with \\.\n'
        f'- Each key must be a list of 4 description strings (one per Part).\n'
        f'- Each value must be exactly one of: {colors_str}.\n'
        f'- Every detected color must appear exactly once as a value.\n'
        f'- Do NOT use single quotes.\n'
        f'- Use longer sentences (not single words); use a definite tone.\n'
        f'- Output ONLY the dictionary, starting with ANSWER: '
    )
    resp5, history = vlm_response(
        client, model_name, question5, encoded_images, history=history, generation_config=generation_config
    )
    print(resp5)
    print(type(resp5))
    region_matching = parse_region_matching(resp5)
    
    log = []
    log.extend([question1, resp1, question2, resp2, question3, resp3, question4, resp4, question5, resp5])
    for i in range(len(log)):
        log[i] = str(log[i])
    print(len(log))
    print('log len showed above')
    return region_matching, log
