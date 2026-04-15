from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Sequence

from PIL import Image
import numpy as np
import torch
from ultralytics import YOLOWorld
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 



@dataclass
class DetectionResult:
    box_xyxy_float: List[float]
    box_xyxy: List[int]
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> dict:
        return asdict(self)


class OpenVocabDetector(ABC):
    def __init__(self, classes: Sequence[str]):
        # classes: ['chair', 'table', 'laptop'...]
        self.classes = []
        self.set_classes(classes)

    def set_classes(self, classes: Sequence[str]):
        new_classes = [item.strip() for item in classes if item.strip()]
        if not new_classes:
            raise ValueError("Detector classes must not be empty.")
        changed = new_classes != self.classes
        self.classes = new_classes
        return changed

    @abstractmethod   # 父类这里只定义接口
    def detect(self, image_rgb: np.ndarray, 
               conf: float = 0.3, iou: float = 0.7, max_det: int = 20) -> List[DetectionResult]:
        raise NotImplementedError  # 这个父类方法没有真正实现，如果有人直接调用它，就报错。


class YOLOWorldDetector(OpenVocabDetector):
    def __init__(self, model_path: str, classes: Sequence[str]):
        self.model_path = model_path
        self.model = YOLOWorld(model_path)
        super().__init__(classes)

    def set_classes(self, classes: Sequence[str]):
        if not super().set_classes(classes):
            return

        # Ultralytics YOLO-World may leave CLIP text weights on CUDA after
        # prediction while newly tokenized class prompts are on CPU. Build text
        # features on CPU, then restore the detector to its previous device.
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

    def detect(self, image_rgb: np.ndarray,
               conf: float = 0.3, iou: float = 0.7, max_det: int = 20) -> List[DetectionResult]:
        # This method is for only 1 image, and outputs the classes and their bounding boxes.
        # In yolo, results is a list of results for each image.
        # results = [img1_result, img2_result...], and each img_result has boxes, scores, class_ids, names.
        results = self.model.predict(
            source=image_rgb,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        # model.set_classes(["table", "chair", "lamp"]) -> class_ids: [0, 1, 2]
        names = result.names

        detections: List[DetectionResult] = []
        for xyxy, score, class_id in zip(boxes, scores, class_ids):
            detections.append(DetectionResult(
                    box_xyxy_float=[float(v) for v in xyxy.tolist()],
                    box_xyxy=[int(round(v)) for v in xyxy.tolist()],
                    confidence=float(score),
                    class_id=int(class_id),
                    class_name=names[class_id],
                )
            )
        return detections


class GroundingDINODetector(OpenVocabDetector):
    def __init__(self, model_path: str, classes: Sequence[str]):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device).eval()
        super().__init__(classes)

    def set_classes(self, classes: Sequence[str]):
        if not super().set_classes(classes):
            return
        # Classes are converted into a text prompt during detect().
        self.prompt_classes = [str(c).strip() for c in self.classes if str(c).strip()]

    def _match_class_id(self, label: str) -> int:
        label_norm = label.lower().strip().rstrip(".")
        for idx, cls_name in enumerate(self.prompt_classes):
            cls_norm = cls_name.lower().strip().rstrip(".")
            if label_norm == cls_norm:
                return idx
        # Fallback for labels like "a cat" when class is "cat".
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
        keep: List[int] = []
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
    
    def detect(self, image_rgb: np.ndarray,
               conf: float = 0.3, iou: float = 0.7, max_det: int = 20) -> List[DetectionResult]:
        # This method is for only 1 image, and outputs the classes and their bounding boxes.
        image = Image.fromarray(image_rgb).convert("RGB")
        text_labels = [[c.lower() for c in self.prompt_classes]]
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=conf,
            text_threshold=conf,
            target_sizes=[image.size[::-1]]
        )
        result = results[0]

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        if boxes is None or len(boxes) == 0:
            return []

        keep_indices = self._nms_keep_indices(boxes, scores, iou=iou, max_det=max_det)
        # model.set_classes(["table", "chair", "lamp"]) -> class_ids: [0, 1, 2]
        detections: List[DetectionResult] = []

        for index in keep_indices:
            xyxy = boxes[index]
            score = scores[index]
            label = labels[index]
            class_id = self._match_class_id(label)

            if class_id >= 0:
                class_name = self.prompt_classes[class_id]
            else:
                class_name = str(label)

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



def build_open_vocab_detector(backend: str, model_path: str, classes: Sequence[str]) -> OpenVocabDetector:
    backend = backend.lower().replace("-", "_")
    if backend == "yolo_world":
        return YOLOWorldDetector(model_path=model_path, classes=classes)
    elif backend in {"grounding_dino", "groundingdino"}:
        return GroundingDINODetector(model_path=model_path, classes=classes)
    raise ValueError(
        f"Unsupported detector backend: {backend}. "
        "Add a new OpenVocabDetector implementation for this foundation model."
    )
