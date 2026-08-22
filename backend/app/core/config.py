"""Application settings: environment variables + YAML configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_YAML = PROJECT_ROOT / "configs" / "app.yaml"


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "PatternPulse AI"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./data/patternpulse.db"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    redis_url: str = ""
    mlflow_tracking_uri: str = ""

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_app_config() -> dict:
    with open(APP_YAML) as f:
        return yaml.safe_load(f)


def scanner_config() -> dict:
    return get_app_config()["scanner"]


def fusion_config() -> dict:
    return get_app_config()["fusion"]


def signal_config() -> dict:
    return get_app_config()["signal"]


def model_config() -> dict:
    return get_app_config()["model"]
