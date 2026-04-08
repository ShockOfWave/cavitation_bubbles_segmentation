import logging
import os

import torch
from clearml import Task
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

from src.config import get_settings

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available: %d device(s), using %s", torch.cuda.device_count(), torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS device")
    else:
        device = "cpu"
        logger.info("No GPU found, using CPU")
    return device


def main() -> None:
    project_settings = get_settings()

    device = get_device()

    Task.set_credentials(
        web_host=project_settings.clearml_web_host,
        api_host=project_settings.clearml_api_host,
        files_host=project_settings.clearml_files_host,
        key=project_settings.clearml_api_access_key,
        secret=project_settings.clearml_api_secret_key,
    )

    ultra_settings.update({"runs_dir": "tune_runs", "tensorboard": False, "clearml": False, "wandb": False})

    dataset_version = project_settings.roboflow_dataset_version

    task = Task.init(
        project_name=project_settings.clearml_project,
        task_name=f"yolo26x-v{dataset_version}-tune",
        auto_connect_frameworks={"pytorch": False, "matplotlib": False},
        output_uri=False,
    )
    logger.info("ClearML Task created: %s", task.id)
    dataset_path = os.path.join("data", "data.yaml")

    model_path = os.path.join("models", "base", "yolo26x-seg.pt")
    logger.info("Loading model from %s", model_path)
    model = YOLO(model_path, task="segment")

    logger.info("Starting tuning on device=%s, data=%s", device, dataset_path)
    model.tune(
        data=dataset_path,
        use_ray=True,
        project=project_settings.clearml_project,
        name=f"yolo26x-v{dataset_version}-tune",
        seed=42,
        plots=True,
        save=True,
        val=True,
        gpu_per_trial=1,
        workers=1,
        iterations=50,
        device=device,
    )


if __name__ == "__main__":
    main()
