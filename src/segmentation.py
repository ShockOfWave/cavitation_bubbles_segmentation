import cv2
import numpy as np
from ultralytics import YOLO

class YoloSegmenter:
    def __init__(self, model_path: str):
        """
        Инициализация модели сегментации.
        model_path – путь к файлу модели.
        """
        self.model = YOLO(model_path)  # Загружаем модель через ultralytics

    def segment_frame(self, frame: np.ndarray) -> list:
        """
        Обрабатывает один кадр и возвращает список детекций.
        Каждая детекция – словарь с ключами:
            'bbox': [x1, y1, x2, y2],
            'mask': np.ndarray бинарная маска (значения 0 или 1),
            'class': int (0 – пузырь в фокусе, 1 – пузырь вне фокуса),
            'confidence': float
        """
        # Выполняем инференс для одного кадра
        results = self.model(frame, verbose=False)
        result = results[0]  # При обработке одного кадра возвращается список из одного результата

        detections = []

        # Извлекаем bounding boxes, оценки и классы
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # shape: (n, 4)
            scores = result.boxes.conf.cpu().numpy()   # shape: (n,)
            classes = result.boxes.cls.cpu().numpy()   # shape: (n,)
        else:
            boxes, scores, classes = [], [], []

        # Извлекаем маски (если модель обучена на segmentation)
        if result.masks is not None:
            # result.masks.data имеет размер (n, height, width) – значения от 0 до 1
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
