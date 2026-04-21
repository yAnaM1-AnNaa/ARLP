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
    parser.add_argument("--checkpoint", default='logs/20260413/oai_vitl_cot/ckpts/best.pth', help="Path to model checkpoint")
    parser.add_argument("--agd_root", default='dataset/dataset6/Unseen/testset', help="Path to AGD20K root")
    parser.add_argument("--viz_dir", required=False, default=None, help="Path to save visualization")
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
    # disambiguation_set = {
    #     ("bathe", "bathtub"): "Inside basin surface of the bathtub where a person sits or lies while bathing. Focus on the smooth inner bottom and side-contact area that supports the body, not the outer wall or floor around the tub.",
    #     ("climb", "stairway"): "Horizontal step surfaces of the stairway where a foot is placed while climbing. Focus on the flat treads that support the feet, not the vertical risers, railings, or side walls.",
    #     ("display", "screen"): "Front screen surface that displays visual information to a viewer. Focus on the active flat display panel, not the bezel, stand, buttons, or surrounding frame.",
    #     ("drop", "garbage"): "Opening or inner container region where garbage should be dropped. Focus on the accessible mouth, bin cavity, or exposed receptacle area, not the outer shell or nearby floor.",
    #     ("heating", "microwave"): "Interior cavity of the microwave where items are placed for heating. Focus on the inner tray, shelf, and chamber area that holds food, not the door front, buttons, or exterior casing.",
    #     ("lean_back", "backreat"): "Front support surface of the backrest where a person's back leans. Focus on the broad rear support pad or panel behind the seat, not the seat cushion, legs, or outer frame.",
    #     ("lie", "bed"): "Top mattress surface of the bed where a person lies down. Focus on the broad horizontal sleeping area that supports the body, not the headboard, bed frame, pillows, or floor.",
    #     ("light", "lamp"): "Light-emitting bulb, shade opening, or illuminated shade area of the lamp. Focus on the part that produces or diffuses light, not the pole, base, cord, or surrounding furniture.",
    #     ("look_through", "window"): "Transparent glass pane area of the window that a person looks through. Focus on the see-through pane surface, not the frame, wall, sill, handle, or curtain.",
    #     ("lying_on", "pillow"): "Soft top surface of the pillow where the head rests. Focus on the cushioned upper contact area, not the bed, blanket, pillow edge seam, or surrounding objects.",
    #     ("open", "door"): "Handle, knob, grip, or exposed edge region of the door used to open it. Focus on the part a hand contacts to pull or push the door, not the broad decorative door panel or nearby wall.",
    #     ("place", "table"): "Flat top surface of the table where objects are placed. Focus on the horizontal tabletop support area, not the legs, side panels, floor, or items already sitting on it.",
    #     ("pull", "drawer"): "Front handle, knob, groove, or grip region of the drawer used for pulling it outward. Focus on the hand-contact pull area, not the drawer interior, cabinet frame, or surrounding furniture.",
    #     ("reflect_image", "mirror"): "Reflective front surface of the mirror where images appear. Focus on the smooth reflective glass area, not the frame, wall, stand, or nearby objects.",
    #     ("refrigerate", "refrigerator"): "Interior storage area of the refrigerator where items are kept cold. Focus on the shelves, bins, and inner compartment surfaces that hold food, not the exterior door, handle, or control panel.",
    #     ("rest_arm", "armset"): "Upper support surface of the armrest where a person's arm rests. Focus on the elongated top contact area beside the seat, not the seat cushion, backrest, legs, or side frame.",
    #     ("sit", "sofa_seat"): "Top sitting surface of the seat where a person places their body weight. Focus on the horizontal cushion or sitting platform, not the backrest, legs, armrests, or surrounding floor.",
    #     ("swing_open", "cabinet_door"): "Handle, knob, pull edge, or outer graspable region of the cabinet door used to swing it open. Focus on the hand-contact opening part, not the fixed cabinet body, shelves, or adjacent wall.",
    #     ("wash", "sink"): "Basin area of the sink where washing takes place. Focus on the inner bowl, drain area, and water-contact surface, not the faucet, countertop, cabinet, or surrounding wall.",
    # }
    disambiguation_set = {
        ("bathe", "bathtub"): "Inside basin surface of the bathtub where a person sits or lies while bathing. Focus on the smooth inner bottom and side-contact area that supports the body, not the outer wall or floor around the tub.",
        ("climb", "stairway"): "Horizontal step surfaces of the stairway where a foot is placed while climbing. Focus on the flat treads that support the feet, not the vertical risers, railings, or side walls.",
        ("drop", "garbage"): "Opening or inner container region where garbage should be dropped. Focus on the accessible mouth, bin cavity, or exposed receptacle area, not the outer shell or nearby floor.",
        ("heating", "microwave"): "Interior cavity of the microwave where items are placed for heating. Focus on the inner tray, shelf, and chamber area that holds food, not the door front, buttons, or exterior casing.",        ("light", "lamp"): "Light-emitting bulb, shade opening, or illuminated shade area of the lamp. Focus on the part that produces or diffuses light, not the pole, base, cord, or surrounding furniture.",
        ("look_through", "window"): "Transparent glass pane area of the window that a person looks through. Focus on the see-through pane surface, not the frame, wall, sill, handle, or curtain.",
        ("lying_on", "pillow"): "Soft top surface of the pillow where the head rests. Focus on the cushioned upper contact area, not the bed, blanket, pillow edge seam, or surrounding objects.",
        ("open", "door"): "Handle, knob, grip, or exposed edge region of the door used to open it. Focus on the part a hand contacts to pull or push the door, not the broad decorative door panel or nearby wall.",
        ("place", "table"): "Flat top surface of the table where objects are placed. Focus on the horizontal tabletop support area, not the legs, side panels, floor, or items already sitting on it.",
        ("pull", "drawer"): "Front handle, knob, groove, or grip region of the drawer used for pulling it outward. Focus on the hand-contact pull area, not the drawer interior, cabinet frame, or surrounding furniture.",
        ("reflect_image", "mirror"): "Reflective front surface of the mirror where images appear. Focus on the smooth reflective glass area, not the frame, wall, stand, or nearby objects.",
        ("wash", "sink"): "Basin area of the sink where washing takes place. Focus on the inner bowl, drain area, and water-contact surface, not the faucet, countertop, cabinet, or surrounding wall.",
        ("light", "lamp"): "Light-emitting bulb, shade opening, or illuminated shade area of the lamp. Focus on the part that produces or diffuses light, not the pole, base, cord, or surrounding furniture.",
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
