import logging
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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


def on_train_epoch_end(trainer) -> None:
    task = Task.current_task()
    if not task:
        return
    for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
        task.get_logger().report_scalar(k, "results", v, iteration=trainer.epoch)
    for k, v in trainer.lr.items():
        task.get_logger().report_scalar(f"lr/{k}", "results", v, iteration=trainer.epoch)


def on_fit_epoch_end(trainer) -> None:
    task = Task.current_task()
    if not task:
        return
    for k, v in trainer.metrics.items():
        task.get_logger().report_scalar(k, "results", v, iteration=trainer.epoch)


def on_train_end(trainer) -> None:
    task = Task.current_task()
    if not task:
        return
    for f in [*trainer.plots.keys(), *trainer.validator.plots.keys()]:
        if "batch" not in f.name and f.exists():
            img = mpimg.imread(str(f))
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])
            ax.imshow(img)
            task.get_logger().report_matplotlib_figure(
                title=f.stem, series="", figure=fig, report_interactive=False,
            )
            plt.close(fig)
    for k, v in trainer.validator.metrics.results_dict.items():
        task.get_logger().report_single_value(f"val/{k}", v)


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

    ultra_settings.update({"clearml": False, "wandb": False})

    dataset_version = project_settings.roboflow_dataset_version

    task = Task.init(
        project_name=project_settings.clearml_project,
        task_name=f"yolo26n_v{dataset_version}",
        auto_connect_frameworks={"pytorch": False, "matplotlib": False},
        output_uri=False,
    )
    logger.info("ClearML Task created: %s", task.id)
    dataset_path = os.path.join("data", "data.yaml")

    model_path = os.path.join("models", "base", "yolo26n-seg.pt")
    logger.info("Loading model from %s", model_path)
    model = YOLO(model_path, task="segment")

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    logger.info("Starting training on device=%s, data=%s", device, dataset_path)
    model.train(
        data=dataset_path,
        epochs=100,
        seed=42,
        plots=True,
        save=True,
        val=True,
        device=device,
        project=project_settings.clearml_project,
        name=f"yolo26n_v{dataset_version}",
        verbose=True,
    )

    logger.info("Running test evaluation")
    test_results = model.val(data=dataset_path, split="test", device=device)
    for k, v in test_results.results_dict.items():
        task.get_logger().report_single_value(f"test/{k}", v)


if __name__ == "__main__":
    main()
