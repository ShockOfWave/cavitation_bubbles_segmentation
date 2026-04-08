from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    roboflow_api_key: str = ""
    huggingface_token: str = ""
    clearml_web_host: str = ""
    clearml_api_host: str = ""
    clearml_files_host: str = ""
    clearml_api_access_key: str = ""
    clearml_api_secret_key: str = ""
    clearml_project: str = "cavitation_tracker_yolo26"
    roboflow_workspace: str = "itmo-ai-in-chemistry"
    roboflow_project: str = "cavitation_bubbles_merged"
    roboflow_dataset_version: int = 1
    username: str
    password: str

    fastapi_host: str = Field(default="fastapi")
    fastapi_port: int = Field(default=8000)
    streamlit_port: int = Field(default=8501)
    secret_key: str = Field(min_length=16)

    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
