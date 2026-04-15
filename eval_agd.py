import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from utils.eval_utils import (
    cal_kl, cal_sim, cal_nss
)
from utils.img_utils import grid_visualize

import argparse
from tqdm import tqdm
from eval_pano import overlay_heatmap
from pano_inference import build_pano_inference

### Eval on AGD20K
def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/oai_vitl_cot.yaml', help="Path to config YAML file")
    parser.add_argument("--checkpoint", default='logs/finetune_1/20260414/finetune/ckpts/best.pth', help="Path to model checkpoint")
    parser.add_argument("--agd_root", default='dataset/dataset6/Seen/testset', help="Path to AGD20K root")
    parser.add_argument("--viz_dir", required=False, default='runs/eval_pano_eazy_groundDINO', help="Path to save visualization")
    parser.add_argument("--det-backend", default="grounding_dino", help="Detector backend name")
    parser.add_argument("--det-model", default="/root/autodl-tmp/yolo/models/grounding-dino-base", help="Detector model path or model name")
    parser.add_argument("--det-conf", type=float, default=0.2, help="Detection confidence")
    parser.add_argument("--det-iou", type=float, default=0.7, help="Detection NMS IoU")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum detections per image")
    parser.add_argument("--box-expand-ratio", type=float, default=0.0, help="Optional box expansion ratio before crop inference")
    parser.add_argument("--heatmap-thresh", type=float, default=None, help="Optional threshold after local heatmaps are pasted back")
    
    args = parser.parse_args()

    ### Load eval data
    if args.viz_dir is not None:
        os.makedirs(args.viz_dir, exist_ok=True)

    agd_egocentric_dir = os.path.join(args.agd_root, "egocentric")
    agd_gt_dir = os.path.join(args.agd_root, "GT")

    # build all info
    eval_set_info = []
    # One prompt per {affordance/action}/{object} directory in dataset6v2.
    disambiguation_set = {
        ("bathe", "bathtub"): "inside basin surface of the bathtub where a person sits or lies while bathing",
        ("climb", "stairway"): "horizontal step surfaces of the stairway for placing feet while climbing",
        ("display", "screen"): "front screen surface that displays visual information",
        ("drop", "garbage"): "opening or inner container region where garbage should be dropped",
        ("heating", "microwave"): "interior cavity of the microwave where items are placed for heating",
        ("lean_back", "backrest"): "front support surface of the backrest where the back leans",
        ("lie", "bed"): "top mattress surface of the bed where a person lies down",
        ("light", "lamp"): "light-emitting bulb or shade area of the lamp",
        ("look_through", "window"): "transparent glass pane area of the window to look through",
        ("lying_on", "pillow"): "soft top surface of the pillow where the head rests",
        ("open", "door"): "handle or edge region of the door used to open it",
        ("place", "table"): "flat top surface of the table where objects are placed",
        ("pull", "drawer"): "front handle or grip region of the drawer used for pulling",
        ("reflect_image", "mirror"): "reflective front surface of the mirror",
        ("refrigerate", "refrigerator"): "interior storage area of the refrigerator where items are kept cold",
        ("rest_arm", "armset"): "upper support surface of the armrest where the arm rests",
        ("sit", "seat"): "top sitting surface of the seat where a person sits",
        ("swing_open", "cabinet_door"): "handle or outer edge region of the cabinet door used to swing it open",
        ("wash", "sink"): "basin area of the sink where washing happens",
    }

    dataset_pairs = {
        (action_name, obj_name)
        for action_name in os.listdir(agd_gt_dir)
        for obj_name in os.listdir(os.path.join(agd_gt_dir, action_name))
        if os.path.isdir(os.path.join(agd_gt_dir, action_name, obj_name))
    }
    if set(disambiguation_set) != dataset_pairs:
        missing = sorted(dataset_pairs - set(disambiguation_set))
        extra = sorted(set(disambiguation_set) - dataset_pairs)
        raise ValueError(
            "disambiguation_set must match GT/{aff}/{obj} directories. "
            f"missing={missing}, extra={extra}"
        )
    
    for action_name in os.listdir(agd_gt_dir):
        action_dir = os.path.join(agd_gt_dir, action_name)
        for obj_name in os.listdir(action_dir):
            obj_dir = os.path.join(action_dir, obj_name)
            for i in range(len(os.listdir(obj_dir))):
                img_name = os.listdir(obj_dir)[i]
                gt_path = os.path.join(agd_gt_dir, action_name, obj_name, img_name)
                img_path = os.path.join(agd_egocentric_dir, action_name, obj_name, img_name.replace(".png", ".jpg"))
                
                # text query
                if (action_name, obj_name) in disambiguation_set:
                    text_desc = disambiguation_set[(action_name, obj_name)]
                else:
                    text_desc = f"region to {action_name} the {obj_name}"

                eval_set_info.append({
                    "img_path": img_path,
                    "gt_path": gt_path,
                    "text_desc": text_desc,
                    "obj_name": obj_name,
                    "det_class": obj_name.replace("_", " "),
                    "viz_name": f"{action_name}_{obj_name}_{i}" # for visualization
                })

    detector_classes = sorted({item["det_class"] for item in eval_set_info})
    print(f"Using pano-style detector-crop inference with {len(detector_classes)} detector classes.")
    pipeline = build_pano_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        detector_backend=args.det_backend,
        detector_model=args.det_model,
        detector_classes=detector_classes,
    )

    KLs = []
    SIMs = []
    NSSs_01 = [] # original NSS 
    NSSs_05 = [] # NSS with threshold 0.5 (see Appendix B for details)
    no_detection_count = 0
    
    for data in tqdm(eval_set_info):
        # load eval data
        img = np.array(Image.open(data["img_path"]).convert("RGB"))
        gt_mask = plt.imread(data["gt_path"])
        text = data["text_desc"]

        if len(img.shape) == 2: # handle grayscale images
            img = np.stack([img, img, img], axis=-1)

        pipeline.detector.set_classes([data["det_class"]])
        detections = pipeline.detect(
            image_rgb=img,
            conf=args.det_conf,
            iou=args.det_iou,
            max_det=args.max_det,
        )
        if len(detections) == 0:
            no_detection_count += 1

        result = pipeline.run_on_detections(
            image_rgb=img,
            detections=detections,
            affordance_text=text,
            box_expand_ratio=args.box_expand_ratio,
            heatmap_thresh=args.heatmap_thresh,
        )
        out = result["heatmap"]

        # postprocess
        out = np.clip(out, 1e-3, 1-(1e-3))

        # save visualization of output
        if args.viz_dir is not None:
            overlay_out = overlay_heatmap(img, out, alpha=0.3)
            overlay_gt = overlay_heatmap(img, gt_mask, alpha=0.3)
            grid_visualize(
                img_list=[overlay_out, overlay_gt],
                name_list=["model pred", "gt"],
                save_path=f"{args.viz_dir}/{data['viz_name']}.png",
                n_rows=1,
                title=f"{data['viz_name']}"
            )

        # compute metrics
        kld, sim, nss_01, nss_05 = cal_kl(out, gt_mask), cal_sim(out, gt_mask), cal_nss(out, gt_mask, threshold=0.1), cal_nss(out, gt_mask, threshold=0.5)
        KLs.append(kld)
        SIMs.append(sim)
        NSSs_01.append(nss_01)
        NSSs_05.append(nss_05)

    print(f"KL: {np.mean(KLs)}, SIM: {np.mean(SIMs)}, NSS: {np.mean(NSSs_01)}, NSS_05: {np.mean(NSSs_05)}")
    print(f"No detections: {no_detection_count}/{len(eval_set_info)}")

if __name__ == "__main__":
    eval()
