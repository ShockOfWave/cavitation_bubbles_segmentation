FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Копируем файлы зависимостей и устанавливаем их
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Копируем остальной код проекта
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${fastapi_port:-8000}
