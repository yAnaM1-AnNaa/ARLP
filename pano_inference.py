from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from local_inference import AffordanceInference
from model import DetectionResult, OpenVocabDetector, build_open_vocab_detector
from utils.file_utils import load_config, store_logs
from utils.vlm_utils import get_text_embedding_options


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_comma_separated_list(raw_text: str) -> List[str]:
    return [item.strip() for item in raw_text.split(",") if item.strip()]


def list_image_paths(img_dir: str) -> List[str]:
    image_paths = []
    for name in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
    return image_paths


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap=plt.cm.jet):
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = np.array(
            Image.fromarray(heatmap.astype(np.float32), mode="F").resize(
                (img.shape[1], img.shape[0]), Image.BILINEAR
            )
        )

    heatmap_colored = colormap(np.clip(heatmap, 0.0, 1.0))[:, :, :3]
    overlay = (1 - alpha) * img + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def save_heatmap_png(heatmap: np.ndarray, save_path: str):
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(heatmap_uint8).save(save_path)


def save_detection_preview(img_np: np.ndarray, detections: Sequence[dict], save_path: str):
    image = Image.fromarray(img_np)
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        label = f'{det["class_name"]} {det["confidence"]:.3f}'
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), label, fill="red")
    image.save(save_path)


def append_timing_record(json_path: str, record: dict):
    with open(json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_timing_summary(summary_path: str, records: Sequence[dict]):
    if not records:
        summary = {
            "num_images": 0,
            "avg_load_time": 0.0,
            "avg_detector_time": 0.0,
            "avg_affordance_time": 0.0,
            "avg_save_time": 0.0,
            "avg_total_time": 0.0,
            "total_elapsed_time": 0.0,
            "total_detections": 0,
            "avg_detections": 0.0,
        }
    else:
        num_images = len(records)
        summary = {
            "num_images": num_images,
            "avg_load_time": sum(r["load_time"] for r in records) / num_images,
            "avg_detector_time": sum(r["detector_time"] for r in records) / num_images,
            "avg_affordance_time": sum(r["affordance_time"] for r in records) / num_images,
            "avg_save_time": sum(r["save_time"] for r in records) / num_images,
            "avg_total_time": sum(r["total_time"] for r in records) / num_images,
            "total_elapsed_time": sum(r["total_time"] for r in records),
            "total_detections": sum(r["num_detections"] for r in records),
            "avg_detections": sum(r["num_detections"] for r in records) / num_images,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


class PanoAffordanceInference:
    def __init__(self, affordance_inference: AffordanceInference, detector: OpenVocabDetector):
        self.affordance_inference = affordance_inference
        self.detector = detector

    @staticmethod
    def expand_and_clip_box(xyxy: Sequence[int], width: int, height: int, expand_ratio: float):
        x1, y1, x2, y2 = xyxy
        box_w = x2 - x1
        box_h = y2 - y1

        pad_w = box_w * expand_ratio / 2.0
        pad_h = box_h * expand_ratio / 2.0

        x1 = max(0, int(np.floor(x1 - pad_w)))
        y1 = max(0, int(np.floor(y1 - pad_h)))
        x2 = min(width, int(np.ceil(x2 + pad_w)))
        y2 = min(height, int(np.ceil(y2 + pad_h)))
        return x1, y1, x2, y2

    def detect(self, image_rgb: np.ndarray, conf: float = 0.05, iou: float = 0.7, max_det: int = 20):
        return self.detector.detect(image_rgb=image_rgb, conf=conf, iou=iou, max_det=max_det)

    def run_on_detections(
        self,
        image_rgb: np.ndarray,
        detections: Sequence[DetectionResult],
        affordance_text: str,
        box_expand_ratio: float = 0.0,
        heatmap_thresh: Optional[float] = None,
    ):
        height, width = image_rgb.shape[:2]
        heatmap_sum = np.zeros((height, width), dtype=np.float32)
        heatmap_count = np.zeros((height, width), dtype=np.float32)
        enriched_detections = []
        crop_records = []
        crop_images = []

        for det in detections:
            x1, y1, x2, y2 = self.expand_and_clip_box(
                det.box_xyxy, width, height, box_expand_ratio
            )
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_records.append((det, x1, y1, x2, y2, crop.shape[:2]))
            crop_images.append(crop)

        if crop_images:
            local_heatmaps = self.affordance_inference.predict_batch(
                crop_images,
                affordance_text,
                thresh=None,
            )
        else:
            local_heatmaps = []

        for local_heatmap, (det, x1, y1, x2, y2, crop_hw) in zip(local_heatmaps, crop_records):
            crop_h, crop_w = crop_hw
            if local_heatmap.shape != (crop_h, crop_w):
                local_heatmap = np.array(
                    Image.fromarray(local_heatmap.astype(np.float32), mode="F").resize(
                        (crop_w, crop_h), Image.BILINEAR
                    )
                )

            heatmap_sum[y1:y2, x1:x2] += local_heatmap
            heatmap_count[y1:y2, x1:x2] += 1.0

            det_dict = asdict(det)
            det_dict["crop_box_xyxy"] = [x1, y1, x2, y2]
            det_dict["crop_size_hw"] = [int(crop_h), int(crop_w)]
            det_dict["affordance_text"] = affordance_text
            enriched_detections.append(det_dict)

        final_heatmap = np.divide(
            heatmap_sum,
            np.maximum(heatmap_count, 1e-6),
            out=np.zeros_like(heatmap_sum),
            where=heatmap_count > 0,
        )
        if heatmap_thresh is not None:
            final_heatmap = (final_heatmap > heatmap_thresh).astype(np.float32)

        return {
            "heatmap": final_heatmap,
            "coverage": heatmap_count,
            "detections": enriched_detections,
        }

    def run(
        self,
        image_rgb: np.ndarray,
        affordance_text: str,
        det_conf: float = 0.05,
        det_iou: float = 0.7,
        max_det: int = 20,
        box_expand_ratio: float = 0.0,
        heatmap_thresh: Optional[float] = None,
    ) -> Tuple[dict, dict]:
        detector_start = time.perf_counter()
        detections = self.detect(image_rgb=image_rgb, conf=det_conf, iou=det_iou, max_det=max_det)
        detector_time = time.perf_counter() - detector_start

        affordance_start = time.perf_counter()
        result = self.run_on_detections(
            image_rgb=image_rgb,
            detections=detections,
            affordance_text=affordance_text,
            box_expand_ratio=box_expand_ratio,
            heatmap_thresh=heatmap_thresh,
        )
        affordance_time = time.perf_counter() - affordance_start

        timing = {
            "detector_time": detector_time,
            "affordance_time": affordance_time,
            "num_detections": len(result["detections"]),
        }
        return result, timing


def build_pano_inference(
    config_path: str,
    checkpoint_path: str,
    detector_backend: str,
    detector_model: str,
    detector_classes: Sequence[str],
) -> PanoAffordanceInference:
    cfg = load_config(config_path)
    text_embedding_option = cfg.get("text_embedding", "embeddings_oai")
    text_embedding_func = get_text_embedding_options(text_embedding_option)
    affordance_inference = AffordanceInference(config_path, checkpoint_path, text_embedding_func)
    detector = build_open_vocab_detector(
        backend=detector_backend,
        model_path=detector_model,
        classes=detector_classes,
    )
    return PanoAffordanceInference(affordance_inference=affordance_inference, detector=detector)


def save_outputs(
    image_path: str,
    image_rgb: np.ndarray,
    output_dir: str,
    affordance_text: str,
    detector_classes: Sequence[str],
    result: dict,
):
    overlay_dir = os.path.join(output_dir, "overlay")
    others_dir = os.path.join(output_dir, "others")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(others_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    detections_json_path = os.path.join(others_dir, f"{base_name}_detections.json")
    heatmap_npy_path = os.path.join(others_dir, f"{base_name}_heatmap.npy")
    heatmap_png_path = os.path.join(others_dir, f"{base_name}_heatmap.png")
    coverage_path = os.path.join(others_dir, f"{base_name}_coverage.png")
    det_preview_path = os.path.join(others_dir, f"{base_name}_detections.png")
    overlay_path = os.path.join(overlay_dir, f"{base_name}_overlay.png")

    with open(detections_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": image_path,
                "det_classes": list(detector_classes),
                "affordance_text": affordance_text,
                "detections": result["detections"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.save(heatmap_npy_path, result["heatmap"])
    save_heatmap_png(result["heatmap"], heatmap_png_path)

    coverage = result["coverage"]
    coverage_norm = np.clip(coverage / np.maximum(coverage.max(), 1.0), 0, 1)
    save_heatmap_png(coverage_norm, coverage_path)

    overlay = overlay_heatmap(image_rgb, result["heatmap"], alpha=0.45)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)
    save_detection_preview(image_rgb, result["detections"], det_preview_path)

    return {
        "detections_json": detections_json_path,
        "heatmap_npy": heatmap_npy_path,
        "heatmap_png": heatmap_png_path,
        "coverage_png": coverage_path,
        "detections_png": det_preview_path,
        "overlay_png": overlay_path,
    }


def run_single_image(
    image_path: str,
    pipeline: PanoAffordanceInference,
    affordance_text: str,
    detector_classes: Sequence[str],
    det_conf: float,
    det_iou: float,
    max_det: int,
    box_expand_ratio: float,
    heatmap_thresh: Optional[float],
    output_dir: str,
    logger: store_logs,
    timing_json_path: str,
):
    total_start = time.perf_counter()
    logger.record("Start inference for image: %s", image_path)

    load_start = time.perf_counter()
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    load_time = time.perf_counter() - load_start

    result, timing = pipeline.run(
        image_rgb=image_rgb,
        affordance_text=affordance_text,
        det_conf=det_conf,
        det_iou=det_iou,
        max_det=max_det,
        box_expand_ratio=box_expand_ratio,
        heatmap_thresh=heatmap_thresh,
    )

    save_start = time.perf_counter()
    saved = save_outputs(
        image_path=image_path,
        image_rgb=image_rgb,
        output_dir=output_dir,
        affordance_text=affordance_text,
        detector_classes=detector_classes,
        result=result,
    )
    save_time = time.perf_counter() - save_start
    total_time = time.perf_counter() - total_start

    timing_record = {
        "image": image_path,
        "load_time": load_time,
        "detector_time": timing["detector_time"],
        "affordance_time": timing["affordance_time"],
        "save_time": save_time,
        "total_time": total_time,
        "num_detections": timing["num_detections"],
    }
    append_timing_record(timing_json_path, timing_record)

    logger.record(
        "Finished inference for image: %s | detections=%d | load=%.4fs | detector=%.4fs | affordance=%.4fs | save=%.4fs | total=%.4fs",
        image_path,
        timing_record["num_detections"],
        timing_record["load_time"],
        timing_record["detector_time"],
        timing_record["affordance_time"],
        timing_record["save_time"],
        timing_record["total_time"],
    )
    return result, saved, timing_record


def resolve_input_images(args) -> List[str]:
    image_paths: List[str] = []
    if args.image:
        image_paths.append(args.image)
    if args.img_dir:
        image_paths.extend(list_image_paths(args.img_dir))
    if not image_paths:
        raise ValueError("At least one input source is required. Use --image or --img-dir.")
    return image_paths


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "inference.log")
    timing_json_path = os.path.join(args.output_dir, "timing.jsonl")
    timing_summary_path = os.path.join(args.output_dir, "timing_summary.json")

    if os.path.exists(timing_json_path):
        os.remove(timing_json_path)

    logger = store_logs("pano_inference", log_path)

    detector_classes = parse_comma_separated_list(args.det_classes)
    pipeline = build_pano_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        detector_backend=args.det_backend,
        detector_model=args.det_model,
        detector_classes=detector_classes,
    )

    image_paths = resolve_input_images(args)
    timing_records = []

    if args.img_dir:
        print(f"Processing {len(image_paths)} images from: {args.img_dir}")
        logger.record("Processing %d images from directory: %s", len(image_paths), args.img_dir)

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Processing: {image_path}")
        result, saved, timing_record = run_single_image(
            image_path=image_path,
            pipeline=pipeline,
            affordance_text=args.affordance_text,
            detector_classes=detector_classes,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            max_det=args.max_det,
            box_expand_ratio=args.box_expand_ratio,
            heatmap_thresh=args.heatmap_thresh,
            output_dir=args.output_dir,
            logger=logger,
            timing_json_path=timing_json_path,
        )
        timing_records.append(timing_record)

        print(f"Detections: {timing_record['num_detections']}")
        print(
            "Timing: "
            f"load={timing_record['load_time']:.4f}s, "
            f"detector={timing_record['detector_time']:.4f}s, "
            f"affordance={timing_record['affordance_time']:.4f}s, "
            f"save={timing_record['save_time']:.4f}s, "
            f"total={timing_record['total_time']:.4f}s"
        )
        for key, value in saved.items():
            print(f"Saved {key} to: {value}")

    write_timing_summary(timing_summary_path, timing_records)
    logger.record("Saved timing JSONL to: %s", timing_json_path)
    logger.record("Saved timing summary JSON to: %s", timing_summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pano affordance inference with pluggable open-vocabulary detectors")
    parser.add_argument("--image", default=None, help="Single input RGB image path")
    parser.add_argument("--img-dir", default='dataset/tests', help="Directory containing input RGB images")
    parser.add_argument("--config", default='configs/eval_agd.yaml', help="ARLP config YAML")
    parser.add_argument("--checkpoint", default='checkpoints/eval_agd.pth', help="ARLP checkpoint path")# logs/20260413/oai_vitl_cot/ckpts/final.pth
    parser.add_argument("--det-backend", default="yolo_world", help="Detector backend name")
    parser.add_argument("--det-model", default="yolov8l-worldv2.pt", help="Detector model path or model name")
    parser.add_argument("--det-classes", default='sofa, chair', help="Comma-separated open-vocabulary detection classes")
    parser.add_argument("--affordance-text", default='a place to rest and sit on.', help="Text query passed to the ARLP affordance model")
    parser.add_argument("--det-conf", type=float, default=0.3, help="Detection confidence")
    parser.add_argument("--det-iou", type=float, default=0.7, help="Detection NMS IoU")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum detections")
    parser.add_argument("--box-expand-ratio", type=float, default=0.0, help="Optional box expansion ratio before cropping")
    parser.add_argument("--heatmap-thresh", type=float, default=None, help="Optional threshold applied after all local heatmaps are pasted back")
    parser.add_argument("--output-dir", default='runs/pano_inference_baseline', help="Directory to save outputs")
    args = parser.parse_args()
    if not args.image and not args.img_dir:
        parser.error("One of --image or --img-dir is required.")
    main(args)
