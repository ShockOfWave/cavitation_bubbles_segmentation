import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bbox1: list, bbox2: list) -> float:
    """
    Вычисляет IoU (Intersection over Union) для двух bounding box.
    Формат bbox: [x1, y1, x2, y2]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def iou_matrix_vectorized(trackers_boxes: np.ndarray, detections_boxes: np.ndarray) -> np.ndarray:
    """Вычисляет IoU-матрицу между двумя наборами bbox через numpy-векторизацию."""
    t = trackers_boxes
    d = detections_boxes

    xx1 = np.maximum(t[:, 0:1], d[:, 0:1].T)
    yy1 = np.maximum(t[:, 1:2], d[:, 1:2].T)
    xx2 = np.minimum(t[:, 2:3], d[:, 2:3].T)
    yy2 = np.minimum(t[:, 3:4], d[:, 3:4].T)

    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

    area_t = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    area_d = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    union = area_t[:, None] + area_d[None, :] - inter

    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class KalmanBoxTracker:
    """
    Отслеживает отдельный объект с использованием полноценного Калмана.
    Состояние: [x, y, s, r, vx, vy, vs]
      x, y - координаты центра,
      s - площадь (масштаб),
      r - соотношение сторон (предполагается относительно стабильным),
      vx, vy, vs - скорости соответствующих параметров.
    """

    def __init__(self, bbox: list, frame_idx: int, timestamp: float, detection: dict | None = None, tracker_id: int = 0):
        """
        Инициализация трека по начальному bbox.
        bbox: [x1, y1, x2, y2]
        tracker_id: уникальный ID, назначаемый ByteTracker.
        """
        self.id = tracker_id

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / (h + 1e-6)

        self.state = np.array([x, y, s, r, 0, 0, 0], dtype=np.float32)
        self.P = np.diag([10, 10, 100, 10, 1000, 1000, 1000]).astype(np.float32)

        dt = 1.0
        self.F = np.array([
            [1, 0, 0, 0, dt,  0,   0],
            [0, 1, 0, 0,  0, dt,   0],
            [0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 1,  0,  0,   0],
            [0, 0, 0, 0,  1,  0,   0],
            [0, 0, 0, 0,  0,  1,   0],
            [0, 0, 0, 0,  0,  0,   1]
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        self.Q = np.diag([0.1, 0.1, 0.1, 0.001, 50, 50, 50]).astype(np.float32)
        self.R = np.diag([0.5, 0.5, 10, 0.01]).astype(np.float32)

        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.time_since_update = 0
        self.history = [bbox]
        self.detection = detection

    def predict(self, dt=None):
        if dt is None:
            dt = 1.0
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt

        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox, frame_idx, timestamp, detection=None):
        dt = timestamp - self.timestamp if self.timestamp is not None else 1.0
        if dt <= 0:
            dt = 1.0

        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / (h + 1e-6)
        z = np.array([x, y, s, r], dtype=np.float32)

        y_meas = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y_meas)
        I = np.eye(self.F.shape[0], dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        self.time_since_update = 0
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.history.append(bbox)
        self.detection = detection

    def get_state(self):
        x, y, s, r = self.state[0:4]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = math.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return [x1, y1, x2, y2]


class ByteTracker:
    def __init__(self, high_thresh=0.6, low_thresh=0.1, max_time_lost=10, iou_threshold=0.2, distance_threshold=50):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = max_time_lost
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.finished_tracks: list[KalmanBoxTracker] = []
        self._next_id = 0

    def update(self, detections, frame_idx, timestamp):
        high_detections = [det for det in detections if det['confidence'] >= self.high_thresh]
        low_detections = [det for det in detections if self.low_thresh <= det['confidence'] < self.high_thresh]

        high_boxes = np.array([det['bbox'] for det in high_detections]) if high_detections else np.empty((0, 4))
        low_boxes = np.array([det['bbox'] for det in low_detections]) if low_detections else np.empty((0, 4))

        for tracker in self.trackers:
            dt = timestamp - tracker.timestamp if tracker.timestamp is not None else 1.0
            if dt <= 0:
                dt = 1.0
            tracker.predict(dt)
        predicted_boxes = np.array([tracker.get_state() for tracker in self.trackers]) if self.trackers else np.empty((0, 4))

        matches, unmatched_trackers, unmatched_detections = self.associate_detections_to_trackers(predicted_boxes, high_boxes)

        for tracker_idx, detection_idx in matches:
            self.trackers[tracker_idx].update(high_boxes[detection_idx], frame_idx, timestamp,
                                              detection=high_detections[detection_idx])

        if len(unmatched_trackers) > 0 and low_boxes.shape[0] > 0:
            unmatched_predicted = predicted_boxes[unmatched_trackers]
            matches_low, unmatched_trackers_final, unmatched_low = self.associate_detections_to_trackers(unmatched_predicted, low_boxes)
            for local_tracker_idx, detection_idx in matches_low:
                global_tracker_idx = unmatched_trackers[local_tracker_idx]
                self.trackers[global_tracker_idx].update(low_boxes[detection_idx], frame_idx, timestamp,
                                                          detection=low_detections[detection_idx])
            unmatched_trackers = [unmatched_trackers[i] for i in unmatched_trackers_final]

        for idx in unmatched_trackers:
            self.trackers[idx].time_since_update += 1

        for detection_idx in unmatched_detections:
            det = high_detections[detection_idx]
            new_tracker = KalmanBoxTracker(det['bbox'], frame_idx, timestamp, detection=det, tracker_id=self._next_id)
            self._next_id += 1
            self.trackers.append(new_tracker)

        active_trackers = []
        for tracker in self.trackers:
            if tracker.time_since_update > self.max_time_lost:
                self.finished_tracks.append(tracker)
            else:
                active_trackers.append(tracker)
        self.trackers = active_trackers

        active = {tracker.id: tracker for tracker in self.trackers if tracker.time_since_update <= 1}
        return active

    def associate_detections_to_trackers(self, trackers_boxes, detections_boxes):
        if trackers_boxes.shape[0] == 0 or detections_boxes.shape[0] == 0:
            return [], list(range(trackers_boxes.shape[0])), list(range(detections_boxes.shape[0]))

        iou_mat = iou_matrix_vectorized(trackers_boxes, detections_boxes)

        row_indices, col_indices = linear_sum_assignment(-iou_mat)
        matches = []
        unmatched_trackers = []
        unmatched_detections = []

        for t, d in zip(row_indices, col_indices):
            tb = trackers_boxes[t]
            db = detections_boxes[d]
            center_tracker = ((tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0)
            center_detection = ((db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0)
            dist = math.hypot(center_tracker[0] - center_detection[0],
                              center_tracker[1] - center_detection[1])
            if iou_mat[t, d] >= self.iou_threshold or dist < self.distance_threshold:
                matches.append((t, d))
            else:
                unmatched_trackers.append(t)
                unmatched_detections.append(d)

        for t in range(trackers_boxes.shape[0]):
            if t not in row_indices:
                unmatched_trackers.append(t)
        for d in range(detections_boxes.shape[0]):
            if d not in col_indices:
                unmatched_detections.append(d)
        return matches, unmatched_trackers, unmatched_detections
