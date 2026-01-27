"""Configuration management for Shad."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Get SHAD_HOME for .env file location
_shad_home = Path(os.environ.get("SHAD_HOME", os.path.expanduser("~/.shad")))
_env_files = [
    str(_shad_home / ".env.local"),
    str(_shad_home / ".env"),
    ".env.local",
    ".env",
]


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=tuple(_env_files),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Redis Settings
    redis_url: str = "redis://localhost:6379/0"

    # Vault Settings
    obsidian_vault_path: str = ""  # Default vault path (for env var fallback)

    # LLM Provider Settings
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Default model settings
    orchestrator_model: str = "claude-sonnet-4-20250514"
    worker_model: str = "claude-sonnet-4-20250514"
    leaf_model: str = "claude-haiku-4-20250514"

    # Path Settings
    history_path: Path = Path("./History")
    skills_path: Path = Path("./Skills")
    core_path: Path = Path("./CORE")

    # Budget Defaults
    default_max_depth: int = 3
    default_max_nodes: int = 50
    default_max_wall_time: int = 300
    default_max_tokens: int = 100000


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
