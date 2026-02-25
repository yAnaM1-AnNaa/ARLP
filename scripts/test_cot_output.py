import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, List, Tuple

from src.MHACoT import parse_region_matching


def _load_results(path: str) -> List[Dict[str, Any]]:
    if os.path.isdir(path):
        results_path = os.path.join(path, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # results.json is a dict keyed by stem
            return list(data.values())

        # fallback: load all per-image json files
        items: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith('.json'):
                continue
            if name == 'results.json':
                continue
            with open(os.path.join(path, name), 'r', encoding='utf-8') as f:
                items.append(json.load(f))
        return items

    # single file
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'image' in data:
        return [data]
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError('Unsupported JSON structure')


def _validate_one(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    required_keys = ['image', 'object_name', 'detected_colors', 'region_matching', 'raw_responses']
    for key in required_keys:
        if key not in item:
            errors.append(f'missing key: {key}')

    detected_colors = item.get('detected_colors') or []
    if not isinstance(detected_colors, list):
        errors.append('detected_colors is not a list')
        detected_colors = []

    region_matching = item.get('region_matching')
    if region_matching is None:
        errors.append('region_matching is null')
        return False, errors
    if not isinstance(region_matching, dict):
        errors.append('region_matching is not a dict')
        return False, errors

    # validate keys and values
    used_colors: List[str] = []
    for key, value in region_matching.items():
        if not isinstance(value, str):
            errors.append(f'region_matching value is not str: {value!r}')
            continue
        used_colors.append(value)
        if detected_colors and value not in detected_colors:
            errors.append(f'color not in detected_colors: {value}')

        try:
            desc_list = ast.literal_eval(key)
        except (ValueError, SyntaxError):
            errors.append(f'key is not parseable list: {key!r}')
            continue
        if not isinstance(desc_list, list):
            errors.append(f'key does not parse to list: {key!r}')
            continue
        if not (4 <= len(desc_list) <= 5):
            errors.append(f'list length not 4-5: {key!r}')
        for desc in desc_list:
            if not isinstance(desc, str) or not desc.strip():
                errors.append(f'description not non-empty string: {key!r}')
                break

    # ensure each detected color appears exactly once
    if detected_colors:
        missing = [c for c in detected_colors if used_colors.count(c) == 0]
        duplicates = [c for c in detected_colors if used_colors.count(c) > 1]
        if missing:
            errors.append(f'missing colors in region_matching: {missing}')
        if duplicates:
            errors.append(f'duplicate colors in region_matching: {duplicates}')

    # optional re-parse from raw Q4
    raw = item.get('raw_responses') or {}
    if isinstance(raw, dict):
        q4 = raw.get('Q4_structured_output')
        if isinstance(q4, str) and q4.strip():
            parsed = parse_region_matching(q4)
            if parsed is None:
                errors.append('Q4_structured_output failed to parse')
            elif parsed != region_matching:
                errors.append('Q4_structured_output parse does not match region_matching')

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate CoT output JSON from CoT.py')
    parser.add_argument('--input', '-i', required=True, help='Path to results.json, a per-image json, or an output directory')
    parser.add_argument('--fail-fast', action='store_true', help='Stop at first error')
    args = parser.parse_args()

    try:
        items = _load_results(args.input)
    except Exception as exc:
        print(f'Failed to load input: {exc}')
        return 2

    total = len(items)
    if total == 0:
        print('No items found')
        return 2

    ok_count = 0
    for idx, item in enumerate(items, 1):
        ok, errors = _validate_one(item)
        if ok:
            ok_count += 1
            continue
        name = item.get('image', f'item#{idx}')
        print(f'ERROR: {name}')
        for err in errors:
            print(f'  - {err}')
        if args.fail_fast:
            break

    print(f'Checked {total} items, OK: {ok_count}, FAILED: {total - ok_count}')
    return 0 if ok_count == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
