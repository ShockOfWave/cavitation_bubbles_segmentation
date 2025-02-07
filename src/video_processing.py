import cv2
import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from src.segmentation import YoloSegmenter
from src.tracker_bytetrack import ByteTracker
from src.utils import draw_mask

class VideoProcessor:
    def __init__(self, model_path: str):
        self.segmenter = YoloSegmenter(model_path)
        # Настраиваем трекер; параметры можно изменять
        self.tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, max_time_lost=10, iou_threshold=0.2, distance_threshold=50)

    def process_video(self, input_video_path: str, output_video_path: str, csv_path: str, hist_folder: str):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть видеофайл.")
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['tracker_id', 'frame_idx', 'timestamp', 'centroid_x', 'centroid_y', 'area', 'class', 'speed'])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps

            detections = self.segmenter.segment_frame(frame)
            tracked_objects = self.tracker.update(detections, frame_idx, timestamp)
            annotated_frame = frame.copy()

            for tracker in tracked_objects.values():
                bbox = tracker.get_state()  # [x1, y1, x2, y2]
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                detection = tracker.detection
                if detection is None:
                    continue
                mask = detection.get('mask', None)
                detection_class = detection.get('class', None)
                color = (0, 255, 0) if detection_class == 0 else (0, 0, 255)

                cv2.putText(annotated_frame,
                            f"ID: {tracker.id}",
                            (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2)
                if mask is not None:
                    annotated_frame = draw_mask(annotated_frame, mask, color)

                # Вычисляем скорость как sqrt(vx^2 + vy^2) из состояния трека
                vx = tracker.state[4]
                vy = tracker.state[5]
                speed = math.sqrt(vx*vx + vy*vy)
                # Площадь берем из state[2]
                area = tracker.state[2]

                csv_writer.writerow([
                    tracker.id,
                    frame_idx,
                    timestamp,
                    cx,
                    cy,
                    area,
                    detection_class,
                    speed
                ])

            out.write(annotated_frame)
            frame_idx += 1

        cap.release()
        out.release()
        csv_file.close()

        # Генерируем гистограммы для топ-20 долгоживущих треков
        speed_hist_file, area_hist_file = self.generate_histograms(hist_folder)
        return speed_hist_file, area_hist_file

    def generate_histograms(self, hist_folder: str):
        """
        Собирает все треки (активные + завершённые), выбирает топ-20 по длине жизни,
        вычисляет для каждого скорость и площадь, строит гистограммы и сохраняет их как файлы.
        Возвращает пути к файлам гистограмм (speed_hist_file, area_hist_file).
        """
        # Объединяем завершённые и активные треки
        all_tracks = self.tracker.finished_tracks + self.tracker.trackers
        if not all_tracks:
            return None, None

        # Сортируем треки по числу кадров (длина истории)
        sorted_tracks = sorted(all_tracks, key=lambda tr: len(tr.history), reverse=True)
        top20 = sorted_tracks[:20]
        speeds = []
        areas = []
        for tr in top20:
            vx = tr.state[4]
            vy = tr.state[5]
            speed = math.sqrt(vx*vx + vy*vy)
            speeds.append(speed)
            areas.append(tr.state[2])

        # Строим гистограмму скорости
        plt.figure()
        plt.hist(speeds, bins=10, color='blue', alpha=0.7)
        plt.title("Гистограмма скорости (топ-20 долгоживущих)")
        plt.xlabel("Скорость (пикселей/кадр)")
        plt.ylabel("Частота")
        speed_hist_file = os.path.join(hist_folder, "histogram_speed.png")
        plt.savefig(speed_hist_file)
        plt.close()

        # Строим гистограмму площади
        plt.figure()
        plt.hist(areas, bins=10, color='green', alpha=0.7)
        plt.title("Гистограмма площади (топ-20 долгоживущих)")
        plt.xlabel("Площадь (пикселей^2)")
        plt.ylabel("Частота")
        area_hist_file = os.path.join(hist_folder, "histogram_area.png")
        plt.savefig(area_hist_file)
        plt.close()

        return speed_hist_file, area_hist_file
