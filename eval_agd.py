import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from utils.eval_utils import (
    cal_kl, cal_sim, cal_nss
)
from utils.img_utils import grid_visualize
from utils.file_utils import load_config

import argparse
from tqdm import tqdm
from eval_pano import overlay_heatmap
from local_inference import build_detection_local_inference
from pano_inference import build_pano_inference


########################################
##### Config helpers
########################################

def _get_detector_runtime_params(cfg):
    detector_cfg = cfg.get("global_detector", {})
    params = detector_cfg.get("params", {})
    conf = params.get("conf", params.get("box_threshold", 0.2))
    text_conf = params.get("text_conf", params.get("text_threshold", conf))
    iou = params.get("iou", 0.7)
    max_det = params.get("max_det", 20)
    return conf, text_conf, iou, max_det


def _load_prompt_config(cfg):
    prompt_config_path = cfg.get("eval", {}).get("prompt_config")
    if prompt_config_path is None:
        raise ValueError("Missing eval.prompt_config in eval config.")
    return load_config(prompt_config_path)


def _get_task_config(prompt_cfg, action_name, obj_name):
    task_cfg = prompt_cfg.get(action_name, {}).get(obj_name)
    if task_cfg is None:
        raise ValueError(
            "Missing prompt config for "
            f"action={action_name!r}, object={obj_name!r}."
        )
    if "affordance_prompt" not in task_cfg:
        raise ValueError(
            "Missing affordance_prompt for "
            f"action={action_name!r}, object={obj_name!r}."
        )
    if "detector_classes" not in task_cfg:
        raise ValueError(
            "Missing detector_classes for "
            f"action={action_name!r}, object={obj_name!r}."
        )
    return task_cfg


########################################
##### Eval AGD
########################################

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/eval_agd.yaml', help="Path to config YAML file")
    parser.add_argument("--agd_root", default='dataset/dataset6/Unseen/testset', help="Path to AGD20K root")
    parser.add_argument("--viz_dir", required=False, default=None, help="Path to save visualization")
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    prompt_cfg = _load_prompt_config(cfg)
    det_conf, det_text_conf, det_iou, max_det = _get_detector_runtime_params(cfg)

    ### Load eval data
    if args.viz_dir is not None:
        os.makedirs(args.viz_dir, exist_ok=True)

    agd_egocentric_dir = os.path.join(args.agd_root, "egocentric")
    agd_gt_dir = os.path.join(args.agd_root, "GT")

    ########################################
    ##### Build eval set
    ########################################

    eval_set_info = []
    
    for action_name in os.listdir(agd_gt_dir):
        action_dir = os.path.join(agd_gt_dir, action_name)
        for obj_name in os.listdir(action_dir):
            obj_dir = os.path.join(action_dir, obj_name)
            if not os.path.isdir(obj_dir):
                continue

            task_cfg = _get_task_config(prompt_cfg, action_name, obj_name)
            detector_classes = task_cfg["detector_classes"]
            affordance_prompt = task_cfg["affordance_prompt"]

            for i, img_name in enumerate(sorted(os.listdir(obj_dir))):
                gt_path = os.path.join(agd_gt_dir, action_name, obj_name, img_name)
                img_path = os.path.join(agd_egocentric_dir, action_name, obj_name, img_name.replace(".png", ".jpg"))

                eval_set_info.append({
                    "img_path": img_path,
                    "gt_path": gt_path,
                    "affordance_prompt": affordance_prompt,
                    "obj_name": obj_name,
                    "detector_classes": detector_classes,
                    "viz_name": f"{action_name}_{obj_name}_{i}" # for visualization
                })

    detector_classes = sorted({
        cls
        for item in eval_set_info
        for cls in item["detector_classes"]
    })
    print(f"Using pano-style detector-crop inference with {len(detector_classes)} detector classes.")

    ########################################
    ##### Build inference modules
    ########################################

    pano_inferer = build_pano_inference(
        config_path=args.config,
        detector_classes=detector_classes,
    )
    local_inferer = build_detection_local_inference(args.config)

    KLs = []
    SIMs = []
    NSSs_01 = [] # original NSS 
    NSSs_05 = [] # NSS with threshold 0.5 (see Appendix B for details)
    no_detection_count = 0

    ########################################
    ##### Run evaluation
    ########################################
    
    for data in tqdm(eval_set_info):
        # load eval data
        img = np.array(Image.open(data["img_path"]).convert("RGB"))
        gt_mask = plt.imread(data["gt_path"])
        text = data["affordance_prompt"]

        if len(img.shape) == 2: # handle grayscale images
            img = np.stack([img, img, img], axis=-1)

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
            affordance_text=text,
        )

        ########################################
        ##### Postprocess
        ########################################

        out = np.clip(out, 1e-3, 1-(1e-3))

        ########################################
        ##### Visualization
        ########################################

        if args.viz_dir is not None:
            np.save(f"{args.viz_dir}/{data['viz_name']}.npy", out)
            overlay_out = overlay_heatmap(img, out, alpha=0.3)
            overlay_gt = overlay_heatmap(img, gt_mask, alpha=0.3)
            grid_visualize(
                img_list=[overlay_out, overlay_gt],
                name_list=["model pred", "gt"],
                save_path=f"{args.viz_dir}/{data['viz_name']}.png",
                n_rows=1,
                title=f"{data['viz_name']}"
            )

        ########################################
        ##### Metrics
        ########################################

        kld, sim, nss_01, nss_05 = cal_kl(out, gt_mask), cal_sim(out, gt_mask), cal_nss(out, gt_mask, threshold=0.1), cal_nss(out, gt_mask, threshold=0.5)
        KLs.append(kld)
        SIMs.append(sim)
        NSSs_01.append(nss_01)
        NSSs_05.append(nss_05)

    print(f"KL: {np.mean(KLs)}, SIM: {np.mean(SIMs)}, NSS: {np.mean(NSSs_01)}, NSS_05: {np.mean(NSSs_05)}")
    print(f"No detections: {no_detection_count}/{len(eval_set_info)}")

if __name__ == "__main__":
    eval()
