import argparse
import json
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval_agd import (
    _get_detector_runtime_params,
    _get_task_config,
    _load_prompt_config,
)
from eval_pano import overlay_heatmap
from local_inference import build_detection_local_inference
from pano_inference import build_pano_inference
from utils.eval_utils import cal_kl, cal_nss, cal_sim
from utils.file_utils import load_config
from utils.img_utils import grid_visualize


########################################
##### Config
########################################

def build_earlyfilm_eval_config(base_config_path, checkpoint_path):
    cfg = load_config(base_config_path)
    cfg["local_inferer"] = {
        "name": "earlyfilm",
        "checkpoint_path": checkpoint_path,
        "load_strict": False,
        "batch_size": 1,
        "model": {
            "vision_model_name": "vit_large_patch14_dinov2.lvd142m",
            "text_encoder": "roberta-large",
            "resolution": 756,
            "film_layers": [6, 12, 18, 23],
            "film_hidden_dim": None,
            "feature_aggregation": "mean",
            "modulate_prefix_tokens": False,
            "pretrained_vision": True,
        },
    }

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name, cfg


########################################
##### Eval set
########################################

def build_eval_set(dataset_root, prompt_cfg):
    egocentric_dir = os.path.join(dataset_root, "egocentric")
    gt_dir = os.path.join(dataset_root, "GT")
    eval_items = []

    for action_name in sorted(os.listdir(gt_dir)):
        action_dir = os.path.join(gt_dir, action_name)
        if not os.path.isdir(action_dir):
            continue

        for obj_name in sorted(os.listdir(action_dir)):
            obj_dir = os.path.join(action_dir, obj_name)
            if not os.path.isdir(obj_dir):
                continue

            task_cfg = _get_task_config(prompt_cfg, action_name, obj_name)
            for idx, img_name in enumerate(sorted(os.listdir(obj_dir))):
                if not img_name.endswith(".png"):
                    continue
                eval_items.append({
                    "img_path": os.path.join(
                        egocentric_dir,
                        action_name,
                        obj_name,
                        img_name.replace(".png", ".jpg"),
                    ),
                    "gt_path": os.path.join(gt_dir, action_name, obj_name, img_name),
                    "affordance_prompt": task_cfg["affordance_prompt"],
                    "detector_classes": task_cfg["detector_classes"],
                    "viz_name": f"{action_name}_{obj_name}_{idx}",
                })

    return eval_items


########################################
##### Run
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/eval_agd.yaml")
    parser.add_argument("--agd_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    config_path, cfg = build_earlyfilm_eval_config(args.base_config, args.checkpoint)
    prompt_cfg = _load_prompt_config(cfg)
    det_conf, det_text_conf, det_iou, max_det = _get_detector_runtime_params(cfg)
    eval_items = build_eval_set(args.agd_root, prompt_cfg)
    if args.max_images is not None:
        eval_items = eval_items[:args.max_images]

    detector_classes = sorted({
        cls
        for item in eval_items
        for cls in item["detector_classes"]
    })
    print(f"Dataset root: {args.agd_root}")
    print(f"Eval images: {len(eval_items)}")
    print(f"Detector classes: {len(detector_classes)}")
    print(f"Visualization dir: {viz_dir}")

    pano_inferer = build_pano_inference(
        config_path=config_path,
        detector_classes=detector_classes,
    )
    local_inferer = build_detection_local_inference(config_path)

    result_path = os.path.join(args.out_dir, "per_image_metrics.jsonl")
    KLs, SIMs, NSSs_01, NSSs_05 = [], [], [], []
    no_detection_count = 0
    completed = set()

    if os.path.exists(result_path):
        with open(result_path) as existing_file:
            for line in existing_file:
                if not line.strip():
                    continue
                item = json.loads(line)
                completed.add(item["viz_name"])
                KLs.append(float(item["KL"]))
                SIMs.append(float(item["SIM"]))
                NSSs_01.append(float(item["NSS"]))
                NSSs_05.append(float(item["NSS_05"]))
                if int(item["detections"]) == 0:
                    no_detection_count += 1

    if completed:
        print(f"Resuming from {len(completed)} completed image(s).")

    with open(result_path, "a") as result_file:
        for data in tqdm(eval_items):
            if data["viz_name"] in completed:
                continue

            img = np.array(Image.open(data["img_path"]).convert("RGB"))
            gt_mask = plt.imread(data["gt_path"])

            pano_inferer.detector.set_classes(data["detector_classes"])
            detections = pano_inferer.predict(
                image_rgb=img,
                conf=det_conf,
                text_conf=det_text_conf,
                iou=det_iou,
                max_det=max_det,
            )
            if len(detections) == 0:
                no_detection_count += 1

            out = local_inferer.predict(
                image_rgb=img,
                detections=detections,
                affordance_text=data["affordance_prompt"],
            )
            out = np.clip(out, 1e-3, 1 - 1e-3)

            kld = float(cal_kl(out, gt_mask))
            sim = float(cal_sim(out, gt_mask))
            nss_01 = float(cal_nss(out, gt_mask, threshold=0.1))
            nss_05 = float(cal_nss(out, gt_mask, threshold=0.5))
            KLs.append(kld)
            SIMs.append(sim)
            NSSs_01.append(nss_01)
            NSSs_05.append(nss_05)

            overlay_out = overlay_heatmap(img, out, alpha=0.3)
            overlay_gt = overlay_heatmap(img, gt_mask, alpha=0.3)
            grid_visualize(
                img_list=[overlay_out, overlay_gt],
                name_list=["model pred", "gt"],
                save_path=os.path.join(viz_dir, f"{data['viz_name']}.png"),
                n_rows=1,
                title=data["viz_name"],
            )

            result_file.write(json.dumps({
                "viz_name": data["viz_name"],
                "img_path": data["img_path"],
                "gt_path": data["gt_path"],
                "detections": len(detections),
                "KL": kld,
                "SIM": sim,
                "NSS": nss_01,
                "NSS_05": nss_05,
            }) + "\n")
            result_file.flush()

    summary = {
        "dataset_root": args.agd_root,
        "checkpoint": args.checkpoint,
        "images": len(eval_items),
        "KL": float(np.mean(KLs)),
        "SIM": float(np.mean(SIMs)),
        "NSS": float(np.mean(NSSs_01)),
        "NSS_05": float(np.mean(NSSs_05)),
        "no_detections": no_detection_count,
    }
    summary["no_detection_ratio"] = no_detection_count / max(len(eval_items), 1)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics: {result_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
