import cv2
import numpy as np
from ultralytics import YOLO


class YoloSegmenter:
    def __init__(self, model_path: str):
        """
        Initialize the segmentation model.
        model_path - path to the model file.
        """
        self.model = YOLO(model_path)

    def segment_frame(self, frame: np.ndarray) -> list:
        """
        Process a single frame and return a list of detections.
        Each detection is a dict with keys:
            'bbox': [x1, y1, x2, y2],
            'mask': np.ndarray binary mask (values 0 or 1),
            'class': int (0 - in-focus bubble, 1 - out-of-focus bubble),
            'confidence': float
        """
        results = self.model(frame, verbose=False)
        result = results[0]

        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
        else:
            boxes, scores, classes = [], [], []

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
        else:
            masks = [None] * len(boxes)

        for i, bbox in enumerate(boxes):
            mask = masks[i] if i < len(masks) else None
            detection = {
                'bbox': bbox.tolist(),
                'mask': (mask > 0.5).astype(np.uint8) if mask is not None else None,
                'class': int(classes[i]),
                'confidence': float(scores[i])
            }
            detections.append(detection)

        return detections
