import argparse
import logging

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "ShockOfWave/cavitation_bubbles_segmentation"


LOCAL_DIR = "models/base"


def download_model(repo_id: str = DEFAULT_HF_REPO, version: str = "latest") -> None:
    logger.info("Downloading model %s, version: %s...", repo_id, version)
    snapshot_download(repo_id=repo_id, revision=version, local_dir=LOCAL_DIR)
    logger.info("Model %s (version %s) downloaded to %s!", repo_id, version, LOCAL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument("--repo", default=DEFAULT_HF_REPO, help="HF repo id")
    parser.add_argument("--version", default="latest", help="Model version/revision")
    args = parser.parse_args()
    download_model(repo_id=args.repo, version=args.version)
