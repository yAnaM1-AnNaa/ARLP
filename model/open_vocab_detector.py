from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Sequence

import numpy as np
from ultralytics import YOLOWorld



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
        self.classes = [item.strip() for item in classes if item.strip()]
        if not self.classes:
            raise ValueError("Detector classes must not be empty.")

    @abstractmethod   # 父类这里只定义接口
    def detect(self, image_rgb: np.ndarray, 
               conf: float = 0.3, iou: float = 0.7, max_det: int = 20) -> List[DetectionResult]:
        raise NotImplementedError  # 这个父类方法没有真正实现，如果有人直接调用它，就报错。


class YOLOWorldDetector(OpenVocabDetector):
    def __init__(self, model_path: str, classes: Sequence[str]):
        super().__init__(classes)
        self.model_path = model_path
        self.model = YOLOWorld(model_path)
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


def build_open_vocab_detector(backend: str, model_path: str, classes: Sequence[str]) -> OpenVocabDetector:
    backend = backend.lower()
    if backend == "yolo_world":
        return YOLOWorldDetector(model_path=model_path, classes=classes)
    raise ValueError(
        f"Unsupported detector backend: {backend}. "
        "Add a new OpenVocabDetector implementation for this foundation model."
    )
