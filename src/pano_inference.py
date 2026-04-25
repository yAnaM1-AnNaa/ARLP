from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch


from model import build_open_vocab_detector
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.file_utils import load_config, store_logs
from utils.vlm_utils import get_text_embedding_options
from local_inference import LateFiLMAffordanceInference, SteerViTAffordanceInference, EarlyFiLMAffordanceInference


class YOLOWorldDetector:
    '''input: cfg; image_rgb
    output: detections: List = [[x1, y1, x2, y2], class_id, class_name, confidence]'''
    def __init__(self, cfg):
        from ultralytics import YOLOWorld
        self.cfg = cfg
        self.model_path = cfg['pano_detector_path']
        self.model = YOLOWorld(self.model_path)
    
    def detect(self, image_rgb: np.ndarray) -> List:
        '''This method is for only 1 image, and outputs the classes and their bounding boxes.
        In yolo, results is a list of results for each image.
        results = [img1_result, img2_result...], and each img_result has boxes, scores, class_ids, names.'''
        conf = self.cfg['pano_detector_conf']
        iou = self.cfg['pano_detector_iou']
        max_det = self.cfg['max_det']
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
        names = result.names

        detections = []
        for xyxy, score, class_id in zip(boxes, scores, class_ids):
            detections.append(box_xyxy_float=[float(v) for v in xyxy.tolist()])
            detections.append(class_id=int(class_id))
            detections.append(class_name=names[class_id])
            detections.append(confidence=float(score))
        return detections


class GroundingDINODetector:
    '''Processes a whole class of objects.
    input: 
        cfg
        images_rgb: [PIL.Image, PIL.Image,...]
        text_labels: [[class1, class2,...],
                  [class1, class2,...],
                  [class1, class2,...],...]
    output:
        batch_detections: [
            [{"bbox": [...], "classname": "chair", "confidence": 0.52},
             {"bbox": [...], "classname": "chair", "confidence": 0.52},
             ...],  # detections for image 1
            [{"bbox": [...], "classname": "table", "confidence": 0.72},
             {"bbox": [...], "classname": "chair", "confidence": 0.62},
             ...],  # detections for image 2
            ...
        ]'''
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_path = cfg['pano_detector_path']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path).to(self.device).eval()

    def detect(self, images_rgb: List, text_labels: List) -> List:
        conf = self.cfg['pano_detector_conf']
        iou = self.cfg['pano_detector_iou']
        max_det = self.cfg['max_det']
        inputs = self.processor(images=images_rgb, 
                                text=text_labels, 
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = [img.size[::-1] for img in images_rgb]
        results = self.processor.post_process_grounded_object_detection(
            outputs, 
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=target_sizes
        )

        batch_detections = []
        for result in results:
            image_detections = []
            for box, score, label in zip(result['boxes'], result['scores'], result["text_labels"]):
                image_detections.append({
                    'bbox': [float(x) for x in box.tolist()],
                    'classname': str(label),
                    'confidence': float(score.item()),
                })
            batch_detections.append(image_detections)
        return batch_detections
    

class PanoAffordanceInference:
    '''FOR 1 SINGLE CATEGORY
    use a pano detector to get the bounding boxes, 
    then use a local inferer to process the content inside the box'''
    def __init__(self, cfg):
        pano_detector_type = cfg['pano_detector']
        local_inferer_type = cfg['local_inferer']
        if 'grounding_dino' in pano_detector_type.lower():
            self.pano_detector = GroundingDINODetector(cfg)
        elif 'yolo_world' in pano_detector_type.lower():
            self.pano_detector = YOLOWorldDetector(cfg)
        else:
            raise ValueError(f"Unsupported pano_detector type: {pano_detector_type}")
        
        if 'latefilm' in local_inferer_type.lower():
            self.local_inferer = LateFiLMAffordanceInference(cfg)
        elif 'steer' in local_inferer_type.lower():
            self.local_inferer = SteerViTAffordanceInference(cfg)
        elif 'earlyfilm' in local_inferer_type.lower():
            self.local_inferer = EarlyFiLMAffordanceInference(cfg)
        else:
            raise ValueError(f"Unsupported local_inferer type: {local_inferer_type}")

    def crop_single_detection(self, single_detection: List, image_rgb: Image):
        '''crop simge detection into smaller patches according to bboxes'''
        # single_detection = [{"bbox": [...], "classname": "chair", "confidence": 0.52},
        #                     {"bbox": [...], "classname": "chair", "confidence": 0.52},...]
        crops = []
        bboxes = [i['bbox'] for i in single_detection] # [[x1, y1, x2, y2], [x1, y1, x2, y2],...]
        for xyxy in bboxes:
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            crop = image_rgb[y1:y2, x1:x2]
            crops.append(crop)
        return crops, bboxes
    
    def predict_single_crop(self, crops: List, bboxes: List, aff_query: str):
        # bboxes = [[x1, y1, x2, y2], [x1, y1, x2, y2],...]
        if crops == []:
            return None
        pano_heatmap = np.zeros(crop.shape[:2], dtype=np.uint8)
        for idx, crop in enumerate(crops):
            x1, y1, x2, y2 = bboxes[idx][:]
            local_heatmap = self.local_inferer.predict(crop, aff_query, thresh=0.1) # same size as crop
            if local_heatmap.shape != crop.shape[:2]:
                local_heatmap = np.array(
                    Image.fromarray(local_heatmap.astype(np.float32), mode="F").resize(
                        (crop.shape[1], crop.shape[0]), Image.BILINEAR
                    )
                )
            pano_heatmap[y1:y2, x1:x2] += local_heatmap
        return pano_heatmap

    def run(self, image_paths: List, classes: List,
            aff_query: str,
            det_conf: float = 0.05,
            det_iou: float = 0.7,
            max_det: int = 20,
            heatmap_thresh: Optional[float] = None):
        '''HEAPMAPS ARE ONLY NUMPY ARRAYS, NOT IMAGES.'''
        # classes serves for only 1 image, so it will be duplicated for all images in the batch.
        ##### Start pano detector #####
        detector_start_time = time.perf_counter()
        images_rgb: List = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        text_labels = [classes for _ in image_paths]
        pano_detections = self.pano_detector.detect(image_paths=image_paths, 
                                                    text_labels=text_labels)
        
        ##### Start local inference on each detection #####
        affordance_start_time = time.perf_counter()
        pano_heatmaps = []
        for i in range(len(image_paths)):
            crops, bboxes = self.crop_detections(single_detection=pano_detections[i],
                                         image_rgb=images_rgb[i])
            pano_heatmap = self.predict_single_crop(crops=crops, bboxes=bboxes, aff_query=aff_query)
            pano_heatmaps.append(pano_heatmap)
        end_time = time.perf_counter()
        
        timing = {
            "detector_time": affordance_start_time - detector_start_time,
            "affordance_time": end_time - affordance_start_time,
            "num_detections": len(pano_heatmaps),
        }
        return pano_heatmaps, timing


