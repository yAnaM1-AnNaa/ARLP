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

import numpy as np
from PIL import Image
import requests
import urllib.request

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


def init_model(model_name: str = 'google/gemini-2.0-flash-001'):
    """Prepare OpenRouter API config for reuse across batch processing."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise EnvironmentError('OPENROUTER_API_KEY is not set')

    base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1/chat/completions')
    site_url = os.getenv('OPENROUTER_SITE_URL')
    site_name = os.getenv('OPENROUTER_SITE_NAME')

    client = {
        'api_key': api_key,
        'base_url': base_url,
        'model': model_name,
        'site_url': site_url,
        'site_name': site_name,
    }
    return client, None


def _encode_image_as_data_uri(image_path: str) -> str:
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


def _openrouter_chat(
    client: Dict[str, Any],
    prompt: str,
    image_paths: List[str],
    history: Optional[List[Dict[str, str]]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    headers = {
        'Authorization': f"Bearer {client['api_key']}",
        'Content-Type': 'application/json',
    }
    if client.get('site_url'):
        headers['HTTP-Referer'] = client['site_url']
    if client.get('site_name'):
        headers['X-Title'] = client['site_name']

    content = [{'type': 'text', 'text': prompt}]
    for path in image_paths:
        content.append({'type': 'image_url', 'image_url': {'url': _encode_image_as_data_uri(path)}})

    messages: List[Dict[str, Any]] = []
    if history:
        messages.extend(history)
    messages.append({'role': 'user', 'content': content})

    payload: Dict[str, Any] = {
        'model': client['model'],
        'messages': messages,
    }
    if generation_config:
        payload.update(generation_config)

    if requests is not None:
        resp = requests.post(client['base_url'], headers=headers, json=payload, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f'OpenRouter API error {resp.status_code}: {resp.text}')
        data = resp.json()
    else:
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(client['base_url'], data=body, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_body = resp.read().decode('utf-8')
                data = json.loads(resp_body)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode('utf-8') if exc.fp else str(exc)
            raise RuntimeError(f'OpenRouter API error {exc.code}: {err_body}') from exc
    text = data['choices'][0]['message']['content']

    new_history = list(history) if history else []
    new_history.append({'role': 'user', 'content': prompt})
    new_history.append({'role': 'assistant', 'content': text})
    return text, new_history


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
        model: OpenRouter client config
        tokenizer: unused (kept for compatibility)
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

    # Normalize generation_config for OpenRouter/OpenAI-style APIs
    if 'max_tokens' not in generation_config and 'max_new_tokens' in generation_config:
        generation_config['max_tokens'] = generation_config.pop('max_new_tokens')
    generation_config.pop('do_sample', None)

    if colors is None:
        colors = detect_colors(image_path)

    if not colors:
        print(f"Warning: No colors detected in {image_path}")
        return None, []

    colors_str = ', '.join(colors)

    # Prepare images for OpenRouter API
    if original_image_path is not None:
        image_paths = [original_image_path, image_path]
        image_desc = (
            f'The first image shows the original {object_name}. '
            f'The second image shows the same {object_name} with different parts highlighted '
            f'in distinct colors: {colors_str}.'
        )
    else:
        image_paths = [image_path]
        image_desc = (
            f'This image shows a {object_name} with different parts highlighted '
            f'in distinct colors: {colors_str}.'
        )

    responses = []

    # ---- Q1: Where to interact ----
    question1 = (
        f'{image_desc} '
        f'For each colored region, identify what functional part of the {object_name} it represents '
        f'and determine whether it is a part that directly interacts with people.'
    )
    resp1, history = _openrouter_chat(
        model, question1, image_paths, history=None, generation_config=generation_config
    )
    responses.append(resp1)

    # ---- Q2: Why based on geometric structure ----
    question2 = (
        f'For each colored region you identified, explain from the geometric structure of the {object_name} '
        f'why that part can interact with people. Give a concise explanation for each color.'
    )
    resp2, history = _openrouter_chat(
        model, question2, image_paths, history=history, generation_config=generation_config
    )
    responses.append(resp2)

    # ---- Q3: Affordance description ----
    question3 = (
        f'For each colored region of the {object_name}, describe the primary interaction between '
        f'that part and a person, including the interaction type, the specific part of the {object_name}, '
        f'and how a person would physically interact with it.'
    )
    resp3, history = _openrouter_chat(
        model, question3, image_paths, history=history, generation_config=generation_config
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
    resp4, history = _openrouter_chat(
        model, question4, image_paths, history=history, generation_config=generation_config
    )
    responses.append(resp4)

    # Parse structured output
    region_matching = parse_region_matching(resp4)

    return region_matching, responses
