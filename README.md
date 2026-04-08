# Cavitation Bubbles Segmentation

A tool for detecting, segmenting, and tracking cavitation bubbles in video. Uses a YOLO-based model and the ByteTrack tracker to build trajectories and compute bubble statistics.

## Features

- Frame-by-frame bubble detection and segmentation
- Object tracking with trajectory export
- Memory-efficient tracking: bounded history window and automatic cleanup of finished trackers
- Automatic CSV reports with size and velocity per bubble
- Histograms and summary plots
- Streamlit web interface and FastAPI REST service
- Docker support (CPU and GPU)
- Experiment tracking with [ClearML](https://clear.ml/)

## Requirements

- Linux
- [uv](https://github.com/astral-sh/uv) — for local runs
- [Docker](https://docs.docker.com/engine/install/) + [Docker Compose](https://docs.docker.com/compose/install/) — for containerised runs
- *(optional)* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — for GPU acceleration

## Project Structure

```
├── pyproject.toml                # Project dependencies (uv)
├── Dockerfile                    # Universal image (CPU/GPU)
├── docker-compose.yml            # Run without GPU
├── docker-compose.gpu.yml        # Override for NVIDIA GPU
├── .env.example                  # Environment variables template
├── src/
│   ├── config.py                 # Configuration via environment variables
│   ├── api/
│   │   ├── main.py               # FastAPI app factory
│   │   ├── auth.py               # JWT authentication
│   │   └── video.py              # Video processing endpoint
│   ├── ml/
│   │   ├── segmentation.py       # YOLO segmentation logic
│   │   ├── tracking.py           # ByteTrack tracker
│   │   └── processing.py         # Video processing pipeline
│   └── utils/
│       ├── visualization.py      # Mask drawing
│       └── geometry.py           # Centroid, distance calculations
├── frontend/
│   └── streamlit_app.py          # Web interface
└── scripts/
    ├── download_dataset.py       # Download dataset from Roboflow
    ├── train.py                  # Model training with ClearML logging
    ├── tune.py                   # Hyperparameter search with ClearML logging
    └── download_model.py         # Download weights from Hugging Face
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-user/cavitation_bubbles_segmentation.git
cd cavitation_bubbles_segmentation
```

### 2. Download model weights (Git LFS)

```bash
git lfs install
git lfs pull
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the values:

```env
ROBOFLOW_API_KEY=your_key
ROBOFLOW_WORKSPACE=itmo-ai-in-chemistry
ROBOFLOW_PROJECT=cavitation_bubbles_merged
ROBOFLOW_DATASET_VERSION=1
HUGGINGFACE_TOKEN=your_token
CLEARML_WEB_HOST=https://app.your-clearml-instance.example.com
CLEARML_API_HOST=https://api.your-clearml-instance.example.com
CLEARML_FILES_HOST=https://files.your-clearml-instance.example.com
CLEARML_PROJECT=your_clearml_project
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key
username=admin
password=your_password
fastapi_host=fastapi     # use "fastapi" for Docker, "localhost" for local runs
fastapi_port=8000
streamlit_port=8501
secret_key=random_string
```

---

## Running with Docker

### CPU (any machine)

```bash
docker compose up --build
```

### NVIDIA GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Open [http://localhost:8501](http://localhost:8501) to access the Streamlit interface.

---

## Running Locally (without Docker)

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Set `localhost` in `.env`

```env
fastapi_host=localhost
```

### 4. Start FastAPI (terminal 1)

```bash
uv run uvicorn src.api.main:app --port 8000 --reload
```

### 5. Start Streamlit (terminal 2)

```bash
uv run streamlit run frontend/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/token` | Obtain a JWT token |
| `POST` | `/process_video/` | Process a video file |

Interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Training

### 1. Download the dataset from Roboflow

```bash
uv run python scripts/download_dataset.py
```

### 2. Train the model

```bash
uv run python scripts/train.py
```

Training metrics and artifacts are logged to ClearML automatically.

### 3. Hyperparameter search

```bash
uv run python scripts/tune.py
```

### 4. Download weights from Hugging Face

```bash
uv run python scripts/download_model.py
```

---

## License

MIT License

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Roboflow](https://roboflow.com) — dataset hosting
- [ClearML](https://clear.ml/) — experiment tracking
