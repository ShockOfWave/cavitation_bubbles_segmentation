from roboflow import Roboflow

from src.config import get_settings


def main() -> None:
    settings = get_settings()

    rf = Roboflow(api_key=settings.roboflow_api_key)

    project = rf.workspace(settings.roboflow_workspace).project(settings.roboflow_project)

    dataset = project.version(settings.roboflow_dataset_version).download(
        "yolov11",
        location="data",
    )

    print(f"Dataset downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
