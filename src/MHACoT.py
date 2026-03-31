'''
Multi-Hop Affordance Chain-of-Thought (MHACoT)

The CoT contains 4 prompts, intended to reveal the following features per colored region:
1. Where to interact - identify each colored region's functional part
2. Why this region can interact based on geometric shapes
3. The affordance of the object - interaction description per region
4. Further possible affordance - additional interactions + structured output

Adapted for clustered images where each color represents a different object part.
Canonical output format: a JSON object with a `regions` array.

Sample output:
{
    "regions": [
        {
            "color": "Red",
            "descriptions": [
                "Where to interact: seat surface",
                "Why: broad flat area that supports the body",
                "Basic affordance: sit on the chair",
                "Further affordance: lean or rest briefly"
            ]
        },
        {
            "color": "Blue",
            "descriptions": [
                "Where to interact: backrest",
                "Why: upright panel behind the seat",
                "Basic affordance: support the back",
                "Further affordance: lean against comfortably"
            ]
        }
    ]
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


def _normalize_description_list(descriptions):
    """Normalize a candidate descriptions field into a clean list[str]."""
    if isinstance(descriptions, (list, tuple)):
        return [str(item).strip() for item in descriptions if str(item).strip()]
    if descriptions is None:
        return []
    text = str(descriptions).strip()
    return [text] if text else []


def _canonicalize_region_matching(payload):
    """Convert parsed model output into canonical {color: [descriptions]} format."""
    if not isinstance(payload, dict):
        return None

    regions = payload.get('regions')
    if isinstance(regions, list):
        result = {}
        for item in regions:
            if not isinstance(item, dict):
                return None
            color = str(item.get('color', '')).strip()
            descriptions = _normalize_description_list(item.get('descriptions', []))
            if not color or not descriptions:
                return None
            result[color] = descriptions
        return result or None

    # Backward compatibility with the old format: {description_list_str: color}
    result = {}
    for descriptions, color in payload.items():
        color_str = str(color).strip()
        if not color_str:
            continue
        normalized = _normalize_description_list(descriptions)
        if not normalized:
            try:
                parsed = ast.literal_eval(str(descriptions))
            except (ValueError, SyntaxError, TypeError):
                parsed = descriptions
            normalized = _normalize_description_list(parsed)
        if normalized:
            result[color_str] = normalized
    return result or None


def parse_region_matching(response_text):
    """
    Parse the final VLM response into canonical `{color: [descriptions, ...]}` format.

    Preferred input format is strict JSON with a top-level `regions` field.
    A best-effort fallback is kept for older dict-shaped outputs so existing logs
    and cached responses remain readable.
    """
    if response_text is None:
        print("Failed to parse region matching: response_text is None")
        return None
    if not isinstance(response_text, (str, bytes)):
        response_text = str(response_text)
    if isinstance(response_text, bytes):
        response_text = response_text.decode("utf-8", errors="ignore")

    text = response_text.strip()

    answer_match = re.search(r'ANSWER:\s*(\{.*\})', text, re.DOTALL)
    if answer_match:
        candidate = answer_match.group(1)
    else:
        fenced_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fenced_match:
            candidate = fenced_match.group(1)
        else:
            dict_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if not dict_match:
                print("Failed to find JSON/dict structure in response")
                return None
            candidate = dict_match.group(1)

    parsed_payload = None
    try:
        parsed_payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed_payload = ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError):
            parsed_payload = None

    result = _canonicalize_region_matching(parsed_payload)
    if result is not None:
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
        region_matching: canonical dict in format {color_name: [description1, ...]}, or None on failure
        raw_responses: list of raw prompt/response strings from the model
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
        f'Now convert your analysis into STRICT JSON.\n'
        f'For each detected color of the {object_name}, produce exactly one object with these fields:\n'
        f'- `color`: the color name\n'
        f'- `descriptions`: an array of exactly 4 strings in this order:\n'
        f'  1. Where to interact\n'
        f'  2. Why it can interact based on geometric structure\n'
        f'  3. Basic affordance\n'
        f'  4. Further possible affordance\n\n'
        f'Output format must be EXACTLY:\n'
        f'{{\n'
        f'  "regions": [\n'
        f'    {{"color": "Red", "descriptions": ["...", "...", "...", "..."]}}\n'
        f'  ]\n'
        f'}}\n\n'
        f'Rules:\n'
        f'- Output ONLY JSON. No markdown. No explanation. No prefix.\n'
        f'- `regions` must be a JSON array.\n'
        f'- Each `descriptions` value must be a JSON array of exactly 4 non-empty strings.\n'
        f'- Each `color` must be exactly one of: {colors_str}.\n'
        f'- Every detected color must appear exactly once.\n'
        f'- Do not include any colors outside this set.\n'
        f'- Use double quotes for every JSON string.\n'
        f'- If uncertain, still return valid JSON using your best grounded guess from the images.\n'
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
