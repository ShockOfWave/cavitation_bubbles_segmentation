import csv
import logging
import math
import os
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.ml.segmentation import YoloSegmenter
from src.ml.tracking import ByteTracker
from src.utils.visualization import draw_mask

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, model_path: str):
        self.segmenter = YoloSegmenter(model_path)

    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
        csv_path: str,
        hist_folder: str,
    ) -> tuple[str | None, str | None]:
        tracker = ByteTracker(
            high_thresh=0.6,
            low_thresh=0.1,
            max_time_lost=10,
            iou_threshold=0.2,
            distance_threshold=50,
        )

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            with open(csv_path, mode="w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["tracker_id", "frame_idx", "timestamp", "centroid_x", "centroid_y", "area", "class", "speed"])

                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    timestamp = frame_idx / fps

                    detections = self.segmenter.segment_frame(frame)
                    tracked_objects = tracker.update(detections, frame_idx, timestamp)
                    annotated_frame = frame.copy()

                    for trk in tracked_objects.values():
                        bbox = trk.get_state()
                        cx = int((bbox[0] + bbox[2]) / 2)
                        cy = int((bbox[1] + bbox[3]) / 2)

                        detection = trk.detection
                        if detection is None:
                            continue
                        mask = detection.get("mask", None)
                        detection_class = detection.get("class", None)
                        color = (0, 255, 0) if detection_class == 0 else (0, 0, 255)

                        cv2.putText(
                            annotated_frame,
                            f"ID: {trk.id}",
                            (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )
                        if mask is not None:
                            annotated_frame = draw_mask(annotated_frame, mask, color)

                        vx = trk.state[4]
                        vy = trk.state[5]
                        speed = math.sqrt(vx * vx + vy * vy)
                        area = trk.state[2]

                        csv_writer.writerow([trk.id, frame_idx, timestamp, cx, cy, area, detection_class, speed])

                    out.write(annotated_frame)
                    frame_idx += 1

        finally:
            out.release()
            cap.release()

        logger.info("Processed %d frames from %s", frame_idx, input_video_path)

        h264_path = output_video_path + ".h264.mp4"
        subprocess.run(
            ["ffmpeg", "-i", output_video_path, "-c:v", "libx264", "-preset", "fast", "-y", h264_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        os.replace(h264_path, output_video_path)

        return _generate_histograms(tracker, hist_folder)


def _generate_histograms(
    tracker: ByteTracker, hist_folder: str
) -> tuple[str | None, str | None]:
    all_tracks = tracker.finished_tracks + tracker.trackers
    if not all_tracks:
        return None, None

    sorted_tracks = sorted(all_tracks, key=lambda tr: tr.history_len, reverse=True)
    top20 = sorted_tracks[:20]
    speeds = []
    areas = []
    for tr in top20:
        vx = tr.state[4]
        vy = tr.state[5]
        speeds.append(math.sqrt(vx * vx + vy * vy))
        areas.append(tr.state[2])

    plt.figure()
    plt.hist(speeds, bins=10, color="blue", alpha=0.7)
    plt.title("Speed histogram (top-20 longest-lived)")
    plt.xlabel("Speed (pixels/frame)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    speed_hist_file = os.path.join(hist_folder, "histogram_speed.png")
    plt.savefig(speed_hist_file)
    plt.close()

    plt.figure()
    plt.hist(areas, bins=10, color="green", alpha=0.7)
    plt.title("Area histogram (top-20 longest-lived)")
    plt.xlabel("Area (pixels^2)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    area_hist_file = os.path.join(hist_folder, "histogram_area.png")
    plt.savefig(area_hist_file)
    plt.close()

    return speed_hist_file, area_hist_file
