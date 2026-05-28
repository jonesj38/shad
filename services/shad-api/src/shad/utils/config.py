"""Configuration management for Shad."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
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

    # Collection Settings
    default_collection_path: str = Field(
        default="",
        alias="SHAD_COLLECTION_PATH",
        description="Default collection path (env var fallback)",
    )

    # LLM Provider Settings
    # SHAD_LLM_PROVIDER supports: auto, claude-code, gemini-cli, anthropic,
    # openai, openai-compatible, edwin-gateway. The OpenAI-compatible modes
    # allow Shad to use EdwinPAI/provider-router style endpoints in addition
    # to direct vendor APIs.
    llm_provider: str = Field(
        default="auto",
        validation_alias=AliasChoices("SHAD_LLM_PROVIDER", "LLM_PROVIDER"),
    )
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("SHAD_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )
    openai_organization: str = Field(
        default="",
        validation_alias=AliasChoices("SHAD_OPENAI_ORGANIZATION", "OPENAI_ORGANIZATION"),
    )
    edwin_gateway_base_url: str = Field(
        default="http://127.0.0.1:18789/v1",
        validation_alias=AliasChoices(
            "SHAD_EDWIN_GATEWAY_BASE_URL",
            "EDWIN_GATEWAY_BASE_URL",
        ),
    )
    edwin_gateway_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SHAD_EDWIN_GATEWAY_API_KEY",
            "EDWIN_GATEWAY_API_KEY",
        ),
    )

    # CLI Settings
    use_gemini_cli: bool = False
    gemini_cli_timeout: int = 120
    claude_cli_timeout: int = 300
    gemini_orchestrator_model: str = "gemini-3-pro-preview"
    gemini_worker_model: str = "gemini-3-pro-preview"
    gemini_leaf_model: str = "gemini-3-flash-preview"

    # Default model settings
    orchestrator_model: str = Field(
        default="claude-opus-4-6",
        validation_alias=AliasChoices("SHAD_ORCHESTRATOR_MODEL", "ORCHESTRATOR_MODEL"),
    )
    worker_model: str = Field(
        default="claude-sonnet-4-6",
        validation_alias=AliasChoices("SHAD_WORKER_MODEL", "WORKER_MODEL"),
    )
    leaf_model: str = Field(
        default="claude-sonnet-4-6",
        validation_alias=AliasChoices("SHAD_LEAF_MODEL", "LEAF_MODEL"),
    )

    # Path Settings (default to ~/.shad/)
    history_path: Path = _shad_home / "history"
    skills_path: Path = _shad_home / "skills"
    core_path: Path = _shad_home / "CORE"

    # Budget Defaults
    default_max_depth: int = 3
    default_max_nodes: int = 50
    default_max_wall_time: int = 1200
    default_max_tokens: int = 2000000  # 2M - allows complex tasks spanning many subtasks


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
