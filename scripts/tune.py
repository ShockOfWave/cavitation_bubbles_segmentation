import os

from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

from src.config import get_settings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ultra_settings.update({"runs_dir": "tune_runs", "tensorboard": False, "wandb": True})

project_settings = get_settings()

rf = Roboflow(api_key=project_settings.roboflow_api_key)

project = rf.workspace().project("cavitation_bubbles_segmentation")

dataset = project.version(9).download("yolov11")

model = YOLO("yolo11m-seg", task="segment")

model.tune(data=os.path.join(dataset.location, "data.yaml"),
           use_ray=True,
           project="cavitation_bubbles_segmentation",
           name="yolov11m-v9-tune",
           seed=42,
           plots=True,
           save=True,
           val=True,
           gpu_per_trial=1,
           workers=1,
           iterations=50,
           device=1)
