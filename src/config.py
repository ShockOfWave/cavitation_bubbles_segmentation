from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    roboflow_api_key: str = ""
    huggingface_token: str = ""
    wandb_api_key: str = ""
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
