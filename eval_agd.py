"""
eval_agd.py - main quantitative AGD evaluation for this project.
Role:
1. Builds the project pano pipeline with detector-crop local inference.
2. Evaluates the dataset with KL, SIM, NSS, and NSS_05.
3. This is the primary file for reporting quantitative results.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from utils.eval_utils import (cal_kl, cal_sim, cal_nss)
from utils.img_utils import grid_visualize
from utils.file_utils import load_config
from utils.vlm_utils import get_text_embedding_options

import argparse
from tqdm import tqdm
from eval_windowslide import overlay_heatmap
from src.pano_inference import PanoAffordanceInference


def eval():
    ##### Parse args #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/oai_vitl_cot.yaml', help="Path to config YAML file")
    parser.add_argument("--agd_root", default='dataset/dataset6/Unseen/testset', help="Path to AGD20K root")
    parser.add_argument("--viz_dir", required=False, default=None, help="Path to save visualization")
    
    args = parser.parse_args()

    ##### Load config #####
    cfg = load_config(args.config)
    cfg["text_embedding_func"] = get_text_embedding_options(cfg["text_embedding_func"])
    
    ##### Load pano and local model #####
    pipeline = PanoAffordanceInference(cfg)
    print('Models Loaded Successfully.')

    ##### Set eval paths #####
    if args.viz_dir is not None:
        os.makedirs(args.viz_dir, exist_ok=True)
    agd_egocentric_dir = os.path.join(args.agd_root, "egocentric")
    agd_gt_dir = os.path.join(args.agd_root, "GT")

    ##### Set affordance query #####
    disambiguation_set = {
        ("bathe", "bathtub"): "Inside basin surface of the bathtub where a person sits or lies while bathing. Focus on the smooth inner bottom and side-contact area that supports the body, not the outer wall or floor around the tub.",
        ("climb", "stairway"): "Horizontal step surfaces of the stairway where a foot is placed while climbing. Focus on the flat treads that support the feet, not the vertical risers, railings, or side walls.",
        ("display", "screen"): "Front screen surface that displays visual information to a viewer. Focus on the active flat display panel, not the bezel, stand, buttons, or surrounding frame.",
        ("drop", "garbage"): "Opening or inner container region where garbage should be dropped. Focus on the accessible mouth, bin cavity, or exposed receptacle area, not the outer shell or nearby floor.",
        ("heating", "microwave"): "Interior cavity of the microwave where items are placed for heating. Focus on the inner tray, shelf, and chamber area that holds food, not the door front, buttons, or exterior casing.",
        ("lean_back", "backreat"): "Front support surface of the backrest where a person's back leans. Focus on the broad rear support pad or panel behind the seat, not the seat cushion, legs, or outer frame.",
        ("lie", "bed"): "Top mattress surface of the bed where a person lies down. Focus on the broad horizontal sleeping area that supports the body, not the headboard, bed frame, pillows, or floor.",
        ("light", "lamp"): "Light-emitting bulb, shade opening, or illuminated shade area of the lamp. Focus on the part that produces or diffuses light, not the pole, base, cord, or surrounding furniture.",
        ("look_through", "window"): "Transparent glass pane area of the window that a person looks through. Focus on the see-through pane surface, not the frame, wall, sill, handle, or curtain.",
        ("lying_on", "pillow"): "Soft top surface of the pillow where the head rests. Focus on the cushioned upper contact area, not the bed, blanket, pillow edge seam, or surrounding objects.",
        ("open", "door"): "Handle, knob, grip, or exposed edge region of the door used to open it. Focus on the part a hand contacts to pull or push the door, not the broad decorative door panel or nearby wall.",
        ("place", "table"): "Flat top surface of the table where objects are placed. Focus on the horizontal tabletop support area, not the legs, side panels, floor, or items already sitting on it.",
        ("pull", "drawer"): "Front handle, knob, groove, or grip region of the drawer used for pulling it outward. Focus on the hand-contact pull area, not the drawer interior, cabinet frame, or surrounding furniture.",
        ("reflect_image", "mirror"): "Reflective front surface of the mirror where images appear. Focus on the smooth reflective glass area, not the frame, wall, stand, or nearby objects.",
        ("refrigerate", "refrigerator"): "Interior storage area of the refrigerator where items are kept cold. Focus on the shelves, bins, and inner compartment surfaces that hold food, not the exterior door, handle, or control panel.",
        ("rest_arm", "armset"): "Upper support surface of the armrest where a person's arm rests. Focus on the elongated top contact area beside the seat, not the seat cushion, backrest, legs, or side frame.",
        ("sit", "sofa_seat"): "Top sitting surface of the seat where a person places their body weight. Focus on the horizontal cushion or sitting platform, not the backrest, legs, armrests, or surrounding floor.",
        ("swing_open", "cabinet_door"): "Handle, knob, pull edge, or outer graspable region of the cabinet door used to swing it open. Focus on the hand-contact opening part, not the fixed cabinet body, shelves, or adjacent wall.",
        ("wash", "sink"): "Basin area of the sink where washing takes place. Focus on the inner bowl, drain area, and water-contact surface, not the faucet, countertop, cabinet, or surrounding wall.",
    }
    object_aliases = {
        "armset": ["armrest", "chair armrest", "sofa armrest"],
        "backreat": ["backrest", "chair backrest", "seat backrest"],
        "bathtub": ["bathtub", "tub", "bath tub"],
        "bed": ["bed", "mattress"],
        "cabinet_door": ["cabinet door", "cupboard door"],
        "door": ["door", "door panel"],
        "drawer": ["drawer", "cabinet drawer"],
        "garbage": ["trash can", "garbage can", "bin", "waste bin"],
        "lamp": ["lamp", "light", "lampshade"],
        "microwave": ["microwave", "microwave oven"],
        "mirror": ["mirror", "looking glass"],
        "pillow": ["pillow", "cushion"],
        "refrigerator": ["refrigerator", "fridge"],
        "screen": ["screen", "display", "monitor"],
        "sink": ["sink", "wash basin", "basin"],
        "sofa_seat": ["sofa seat", "sofa", "couch", "seat cushion"],
        "stairway": ["stairway", "stairs", "staircase", "steps"],
        "table": ["table", "desk", "tabletop"],
        "window": ["window", "glass window"],
    }

    ##### Build eval set #####
    eval_set_info = []
    dataset_pairs = {
        (aff, obj)
        for aff in os.listdir(agd_gt_dir)
        for obj in os.listdir(os.path.join(agd_gt_dir, aff))
        if os.path.isdir(os.path.join(agd_gt_dir, aff, obj))
    }
    if not dataset_pairs.issubset(disambiguation_set):
        missing = sorted(dataset_pairs - set(disambiguation_set))
        raise ValueError(
            "disambiguation_set must match GT/{aff}/{obj} directories. "
            f"missing={missing}"
        )
    
    for action_name in os.listdir(agd_gt_dir):
        action_dir = os.path.join(agd_gt_dir, action_name)
        for obj_name in os.listdir(action_dir):
            obj_dir = os.path.join(action_dir, obj_name)
            img_names = sorted(os.listdir(obj_dir))
            aff_query = disambiguation_set[(action_name, obj_name)]
            classes = object_aliases.get(obj_name, [obj_name.replace("_", " ")])

            eval_set_info.append({
                "img_paths": [
                    os.path.join(agd_egocentric_dir, action_name, obj_name, img_name.replace(".png", ".jpg"))
                    for img_name in img_names
                ],
                "gt_paths": [
                    os.path.join(agd_gt_dir, action_name, obj_name, img_name)
                    for img_name in img_names
                ],
                "aff_query": aff_query,
                "classes": classes,
                "viz_names": [
                    f"{action_name}_{obj_name}_{i}"
                    for i in range(len(img_names))
                ]
            })

    ##### Start eval #####
    KLs = []
    SIMs = []
    NSSs_01 = [] # original NSS 
    NSSs_05 = [] # NSS with threshold 0.5 (see Appendix B for details)
    no_detection_count = 0
    
    total_count = 0

    for data in tqdm(eval_set_info):
        pano_heatmaps, _ = pipeline.run(
            image_paths=data["img_paths"],
            classes=data["classes"],
            aff_query=data["aff_query"],
        )

        for img_path, gt_path, viz_name, out in zip(data["img_paths"], data["gt_paths"], data["viz_names"], pano_heatmaps):
            total_count += 1
            img = np.array(Image.open(img_path).convert("RGB"))
            gt_mask = plt.imread(gt_path)

            if out is None:
                no_detection_count += 1
                out = np.zeros(img.shape[:2], dtype=np.float32)

            # postprocess
            out = np.clip(out, 1e-3, 1-(1e-3))

            # save visualization of output
            if args.viz_dir is not None:
                overlay_out = overlay_heatmap(img, out, alpha=0.3)
                overlay_gt = overlay_heatmap(img, gt_mask, alpha=0.3)
                grid_visualize(
                    img_list=[overlay_out, overlay_gt],
                    name_list=["model pred", "gt"],
                    save_path=f"{args.viz_dir}/{viz_name}.png",
                    n_rows=1,
                    title=f"{viz_name}"
                )

            # compute metrics
            kld, sim, nss_01, nss_05 = cal_kl(out, gt_mask), cal_sim(out, gt_mask), cal_nss(out, gt_mask, threshold=0.1), cal_nss(out, gt_mask, threshold=0.5)
            KLs.append(kld)
            SIMs.append(sim)
            NSSs_01.append(nss_01)
            NSSs_05.append(nss_05)

    ##### Print results #####
    print(f"KL: {np.mean(KLs)}, SIM: {np.mean(SIMs)}, NSS: {np.mean(NSSs_01)}, NSS_05: {np.mean(NSSs_05)}")
    print(f"No detections: {no_detection_count}/{total_count}")

if __name__ == "__main__":
    eval()
