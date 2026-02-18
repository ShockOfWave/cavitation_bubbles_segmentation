from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    roboflow_api_key: str
    huggingface_token: str
    wandb_api_key: str
    username: str
    password: str
    
    fastapi_host: str = Field(default="fastapi", env="fastapi_host")
    fastapi_port: int = Field(default=8000, env="fastapi_port")
    streamlit_port: int = Field(default=8501, env="streamlit_port")
    secret_key: str
    
    model_config = SettingsConfigDict(env_file=".env")


def get_settings() -> Settings:
    return Settings()
