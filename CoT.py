# Generate CoT results in batch. Processing multiple folders and all the images inside.
# Usage: python ARLP/CoT.py --img_dir <img_dir> --output_dir <output_dir> --object_name <object_name>
#
# img_dir: directory containing clustered .jpg images (proposal images with colored regions)
# output_dir: directory to save .json result files
# object_name: the object category name (e.g., 'chair', 'kettle')
#
# Optional:
#   --original_dir: directory containing corresponding original images (same filename stem)
#   --model_name: OpenRouter model name (vision-capable)
#
# Output: one JSON file per image + a combined results.json with all results

import os
import sys
import glob
import json
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.MHACoT import init_model, process_image, detect_colors


def main():
    parser = argparse.ArgumentParser(description='Batch MHACoT processing for clustered affordance images')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing clustered/proposal .jpg images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save JSON output files')
    parser.add_argument('--object_name', type=str, required=True,
                        help='Object category name (e.g., chair, kettle)')
    parser.add_argument('--original_dir', type=str, default=None,
                        help='Optional: directory containing original (non-clustered) images with matching filenames')
    parser.add_argument('--model_name', type=str, default='qwen/qwen3-vl-30b-a3b-thinking',
                        help='OpenRouter model name (vision-capable)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all jpg images
    image_paths = sorted(
        glob.glob(os.path.join(args.img_dir, '*.jpg')) +
        glob.glob(os.path.join(args.img_dir, '*.jpeg')) +
        glob.glob(os.path.join(args.img_dir, '*.png'))
    )

    if not image_paths:
        print(f"No images found in {args.img_dir}. Try put the image under a folder")
        return

    print(f"Found {len(image_paths)} images to process")
    print(f"Object: {args.object_name}, Model: {args.model_name}")

    # Load model once
    print("Loading model...")
    model, tokenizer = init_model(args.model_name)
    print("Model loaded.")

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    all_results = {}

    for idx, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]

        print(f"\n[{idx + 1}/{len(image_paths)}] Processing: {filename}")

        # Find corresponding original image if original_dir is provided
        original_image_path = None
        if args.original_dir is not None:
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = os.path.join(args.original_dir, stem + ext)
                if os.path.exists(candidate):
                    original_image_path = candidate
                    break
            if original_image_path:
                print(f"  Using original image: {os.path.basename(original_image_path)}")
            else:
                print(f"  Warning: No matching original image found for {filename}")

        # Detect colors
        colors = detect_colors(image_path)
        print(f"  Detected colors: {colors}")

        if not colors:
            print(f"  Skipping: no colors detected")
            continue

        # Run 4-step CoT
        region_matching, raw_responses = process_image(
            model, tokenizer, image_path, args.object_name,
            colors=colors,
            original_image_path=original_image_path,
            generation_config=generation_config
        )

        # Build per-image result
        result = {
            'image': filename,
            'object_name': args.object_name,
            'detected_colors': colors,
            'region_matching': region_matching,
            'raw_responses': {
                'Q1_interaction_parts': raw_responses[0] if len(raw_responses) > 0 else None,
                'Q2_geometric_reasoning': raw_responses[1] if len(raw_responses) > 1 else None,
                'Q3_affordance_description': raw_responses[2] if len(raw_responses) > 2 else None,
                'Q4_structured_output': raw_responses[3] if len(raw_responses) > 3 else None,
            }
        }

        # Save per-image JSON
        per_image_path = os.path.join(args.output_dir, f'{stem}.json')
        with open(per_image_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        all_results[stem] = result

        if region_matching:
            print(f"  Region matching: {json.dumps(region_matching, ensure_ascii=False)}")
        else:
            print(f"  Warning: Failed to parse structured output for {filename}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, 'results.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Processed {len(all_results)}/{len(image_paths)} images.")
    print(f"Combined results saved to: {combined_path}")


if __name__ == '__main__':
    main()
