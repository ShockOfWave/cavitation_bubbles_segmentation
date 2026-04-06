import base64
import logging
import os
import shutil
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.auth import get_current_user
from src.ml.processing import VideoProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_SIZE_MB = 500
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

MODEL_PATH = "hf_model_repo/model.pt"
video_processor = VideoProcessor(MODEL_PATH)


def _read_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@router.post("/process_video/")
async def process_video_endpoint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    tmp_dir = tempfile.mkdtemp()
    try:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(tmp_dir, unique_filename)

        size = 0
        with open(input_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB} MB",
                    )
                buffer.write(chunk)

        output_video_path = os.path.join(tmp_dir, f"processed_{unique_filename}")
        csv_path = os.path.join(tmp_dir, f"data_{unique_filename.split('.')[0]}.csv")

        speed_hist_file, area_hist_file = video_processor.process_video(
            input_path, output_video_path, csv_path, tmp_dir
        )

        return {
            "output_video": _read_file_b64(output_video_path),
            "output_video_name": os.path.basename(output_video_path),
            "csv_file": _read_file_b64(csv_path),
            "csv_file_name": os.path.basename(csv_path),
            "speed_hist_file": _read_file_b64(speed_hist_file) if speed_hist_file else None,
            "speed_hist_name": os.path.basename(speed_hist_file) if speed_hist_file else None,
            "area_hist_file": _read_file_b64(area_hist_file) if area_hist_file else None,
            "area_hist_name": os.path.basename(area_hist_file) if area_hist_file else None,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
