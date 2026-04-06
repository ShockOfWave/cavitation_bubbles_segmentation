import os

from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

from src.config import get_settings

ultra_settings.update({"wandb": True})

project_settings = get_settings()

rf = Roboflow(api_key=project_settings.roboflow_api_key)

project = rf.workspace().project("cavitation_bubbles_segmentation")

dataset = project.version(9).download("yolov11")

model = YOLO("yolo11m-seg.pt", task="segment")

# config["epochs"] = 300

model.train(
    # **config,
    data=os.path.join(dataset.location, "data.yaml"),
    epochs=100,
    seed=42,
    plots=True,
    save=True,
    val=True,
    device=1,
    project="cavitation_bubbles_segmentation",
    name="yolov11m_v9",
)

version = project.version(9)

version.deploy("yolov11", "cavitation_bubbles_segmentation/yolov11m_v9")
