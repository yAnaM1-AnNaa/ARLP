from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.file_utils import load_config, store_logs


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


########################################
##### Detection result
########################################

@dataclass
class DetectionResult:
    box_xyxy_float: List[float]
    box_xyxy: List[int]
    confidence: float
    class_id: int
    class_name: str


########################################
##### Detector backends
########################################

class YOLOWorldDetector:
    def __init__(self, model_path: str, classes: Sequence[str]):
        from ultralytics import YOLOWorld

        self.model_path = model_path
        self.model = YOLOWorld(model_path)
        self.classes = []
        self.set_classes(classes)

    def set_classes(self, classes: Sequence[str]):
        new_classes = [item.strip() for item in classes if item.strip()]
        if not new_classes:
            raise ValueError("Detector classes must not be empty.")
        if new_classes == self.classes:
            return
        self.classes = new_classes

        try:
            device = next(self.model.model.parameters()).device
        except StopIteration:
            device = None

        if device is not None and device.type != "cpu":
            self.model.to("cpu")
            self.model.set_classes(self.classes)
            self.model.to(device)
        else:
            self.model.set_classes(self.classes)

    def detect(
        self,
        image_rgb: np.ndarray,
        conf: float = 0.3,
        iou: float = 0.7,
        max_det: int = 20,
        **kwargs,
    ) -> List[DetectionResult]:
        results = self.model.predict(
            source=image_rgb,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        detections = []
        for xyxy, score, class_id in zip(boxes, scores, class_ids):
            detections.append(
                DetectionResult(
                    box_xyxy_float=[float(v) for v in xyxy.tolist()],
                    box_xyxy=[int(round(v)) for v in xyxy.tolist()],
                    confidence=float(score),
                    class_id=int(class_id),
                    class_name=names[class_id],
                )
            )
        return detections


class GroundingDINODetector:
    def __init__(self, model_path: str, classes: Sequence[str]):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device).eval()
        self.classes = []
        self.set_classes(classes)

    def set_classes(self, classes: Sequence[str]):
        new_classes = [item.strip() for item in classes if item.strip()]
        if not new_classes:
            raise ValueError("Detector classes must not be empty.")
        self.classes = new_classes
        self.prompt_classes = [str(c).strip() for c in self.classes if str(c).strip()]

    def _match_class_id(self, label: str) -> int:
        label_norm = label.lower().strip().rstrip(".")
        for idx, cls_name in enumerate(self.prompt_classes):
            cls_norm = cls_name.lower().strip().rstrip(".")
            if label_norm == cls_norm:
                return idx
        for idx, cls_name in enumerate(self.prompt_classes):
            cls_norm = cls_name.lower().strip().rstrip(".")
            if cls_norm in label_norm or label_norm in cls_norm:
                return idx
        return -1

    @staticmethod
    def _nms_keep_indices(boxes: torch.Tensor, scores: torch.Tensor, iou: float, max_det: int) -> List[int]:
        if len(boxes) == 0:
            return []

        order = torch.argsort(scores, descending=True)
        keep = []
        while len(order) > 0 and len(keep) < max_det:
            current = int(order[0].item())
            keep.append(current)
            if len(order) == 1:
                break

            current_box = boxes[current].unsqueeze(0)
            rest = order[1:]
            rest_boxes = boxes[rest]

            x1 = torch.maximum(current_box[:, 0], rest_boxes[:, 0])
            y1 = torch.maximum(current_box[:, 1], rest_boxes[:, 1])
            x2 = torch.minimum(current_box[:, 2], rest_boxes[:, 2])
            y2 = torch.minimum(current_box[:, 3], rest_boxes[:, 3])

            inter_w = torch.clamp(x2 - x1, min=0)
            inter_h = torch.clamp(y2 - y1, min=0)
            inter = inter_w * inter_h

            current_area = torch.clamp(current_box[:, 2] - current_box[:, 0], min=0) * torch.clamp(
                current_box[:, 3] - current_box[:, 1], min=0
            )
            rest_area = torch.clamp(rest_boxes[:, 2] - rest_boxes[:, 0], min=0) * torch.clamp(
                rest_boxes[:, 3] - rest_boxes[:, 1], min=0
            )
            union = current_area + rest_area - inter
            ious = inter / torch.clamp(union, min=1e-6)
            order = rest[ious <= iou]

        return keep

    def detect(
        self,
        image_rgb: np.ndarray,
        conf: float = 0.3,
        iou: float = 0.7,
        max_det: int = 20,
        text_conf: Optional[float] = None,
        **kwargs,
    ) -> List[DetectionResult]:
        image = Image.fromarray(image_rgb).convert("RGB")
        text_labels = [[c.lower() for c in self.prompt_classes]]
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=conf,
            text_threshold=conf if text_conf is None else text_conf,
            target_sizes=[image.size[::-1]],
        )
        result = results[0]

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        if boxes is None or len(boxes) == 0:
            return []

        keep_indices = self._nms_keep_indices(boxes, scores, iou=iou, max_det=max_det)
        detections = []
        for index in keep_indices:
            xyxy = boxes[index]
            score = scores[index]
            label = labels[index]
            class_id = self._match_class_id(label)
            class_name = self.prompt_classes[class_id] if class_id >= 0 else str(label)
            xyxy_list = xyxy.detach().cpu().tolist()

            detections.append(
                DetectionResult(
                    box_xyxy_float=[float(v) for v in xyxy_list],
                    box_xyxy=[int(round(v)) for v in xyxy_list],
                    confidence=float(score.detach().cpu().item()),
                    class_id=int(class_id),
                    class_name=class_name,
                )
            )
        return detections


def build_detector(backend: str, model_path: str, classes: Sequence[str]):
    backend = backend.lower().replace("-", "_")
    if backend == "yolo_world":
        return YOLOWorldDetector(model_path=model_path, classes=classes)
    if backend in {"grounding_dino", "groundingdino"}:
        return GroundingDINODetector(model_path=model_path, classes=classes)
    raise ValueError(f"Unsupported detector backend: {backend}.")


########################################
##### File helpers
########################################

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


########################################
##### Detection visualization
########################################

def save_detection_preview(img_np: np.ndarray, detections: Sequence[dict], save_path: str):
    image = Image.fromarray(img_np)
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        label = f'{det["class_name"]} {det["confidence"]:.3f}'
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), label, fill="red")
    image.save(save_path)


########################################
##### Timing logs
########################################

def append_timing_record(json_path: str, record: dict):
    with open(json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_timing_summary(summary_path: str, records: Sequence[dict]):
    if not records:
        summary = {
            "num_images": 0,
            "avg_load_time": 0.0,
            "avg_detector_time": 0.0,
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
            "avg_save_time": sum(r["save_time"] for r in records) / num_images,
            "avg_total_time": sum(r["total_time"] for r in records) / num_images,
            "total_elapsed_time": sum(r["total_time"] for r in records),
            "total_detections": sum(r["num_detections"] for r in records),
            "avg_detections": sum(r["num_detections"] for r in records) / num_images,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


########################################
##### Pano inference
########################################

class PanoInference:
    def __init__(self, detector):
        self.detector = detector

    def detect(
        self,
        image_rgb: np.ndarray,
        conf: float = 0.05,
        iou: float = 0.7,
        max_det: int = 20,
        text_conf: Optional[float] = None,
    ) -> List[DetectionResult]:
        return self.detector.detect(
            image_rgb=image_rgb,
            conf=conf,
            iou=iou,
            max_det=max_det,
            text_conf=text_conf,
        )

    def predict(
        self,
        image_rgb: np.ndarray,
        conf: float = 0.05,
        iou: float = 0.7,
        max_det: int = 20,
        text_conf: Optional[float] = None,
    ) -> List[DetectionResult]:
        return self.detect(
            image_rgb=image_rgb,
            conf=conf,
            iou=iou,
            max_det=max_det,
            text_conf=text_conf,
        )


def build_pano_inference(
    config_path: str,
    detector_backend: Optional[str] = None,
    detector_model: Optional[str] = None,
    detector_classes: Optional[Sequence[str]] = None,
) -> PanoInference:
    cfg = load_config(config_path)
    if 'global_detector' in cfg:
        detector_cfg = cfg['global_detector']
        detector_backend = detector_backend or detector_cfg['name']
        detector_model = detector_model or detector_cfg['model_path']

    if detector_classes is None:
        raise ValueError("detector_classes must be provided.")
    if detector_backend is None or detector_model is None:
        raise ValueError("Detector backend and model path must be provided.")

    detector = build_detector(
        backend=detector_backend,
        model_path=detector_model,
        classes=detector_classes,
    )
    return PanoInference(detector=detector)


########################################
##### Detection output
########################################

def save_detection_outputs(
    image_path: str,
    image_rgb: np.ndarray,
    output_dir: str,
    detector_classes: Sequence[str],
    detections: Sequence[DetectionResult],
):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    detections_json_path = os.path.join(output_dir, f"{base_name}_detections.json")
    det_preview_path = os.path.join(output_dir, f"{base_name}_detections.png")
    detections_dict = [asdict(det) for det in detections]

    with open(detections_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": image_path,
                "det_classes": list(detector_classes),
                "detections": detections_dict,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    save_detection_preview(image_rgb, detections_dict, det_preview_path)
    return {
        "detections_json": detections_json_path,
        "detections_png": det_preview_path,
    }


########################################
##### CLI runner
########################################

def run_single_image(
    image_path: str,
    pipeline: PanoInference,
    detector_classes: Sequence[str],
    det_conf: float,
    det_iou: float,
    max_det: int,
    output_dir: str,
    logger: store_logs,
    timing_json_path: str,
):
    total_start = time.perf_counter()
    logger.record("Start inference for image: %s", image_path)

    load_start = time.perf_counter()
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    load_time = time.perf_counter() - load_start

    detector_start = time.perf_counter()
    detections = pipeline.predict(
        image_rgb=image_rgb,
        conf=det_conf,
        iou=det_iou,
        max_det=max_det,
    )
    detector_time = time.perf_counter() - detector_start

    save_start = time.perf_counter()
    saved = save_detection_outputs(
        image_path=image_path,
        image_rgb=image_rgb,
        output_dir=output_dir,
        detector_classes=detector_classes,
        detections=detections,
    )
    save_time = time.perf_counter() - save_start
    total_time = time.perf_counter() - total_start

    timing_record = {
        "image": image_path,
        "load_time": load_time,
        "detector_time": detector_time,
        "save_time": save_time,
        "total_time": total_time,
        "num_detections": len(detections),
    }
    append_timing_record(timing_json_path, timing_record)

    logger.record(
        "Finished inference for image: %s | detections=%d | load=%.4fs | detector=%.4fs | save=%.4fs | total=%.4fs",
        image_path,
        timing_record["num_detections"],
        timing_record["load_time"],
        timing_record["detector_time"],
        timing_record["save_time"],
        timing_record["total_time"],
    )
    return detections, saved, timing_record


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
        detections, saved, timing_record = run_single_image(
            image_path=image_path,
            pipeline=pipeline,
            detector_classes=detector_classes,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            max_det=args.max_det,
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
    parser.add_argument("--config", default='configs/oai_vitl_registers_cot.yaml', help="ARLP config YAML")
    parser.add_argument("--det-backend", default="grounding_dino", help="Detector backend name")
    parser.add_argument("--det-model", default="/root/autodl-tmp/yolo/models/grounding-dino-base", help="Detector model path or model name")
    parser.add_argument("--det-classes", default="armrest, handrail, secure hand", help="Comma-separated open-vocabulary detection classes")
    parser.add_argument("--det-conf", type=float, default=0.3, help="Detection confidence")
    parser.add_argument("--det-iou", type=float, default=0.7, help="Detection NMS IoU")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum detections")
    parser.add_argument("--output-dir", default='runs/pano_inference_detections', help="Directory to save detection outputs")
    args = parser.parse_args()
    if not args.image and not args.img_dir:
        parser.error("One of --image or --img-dir is required.")
    main(args)
