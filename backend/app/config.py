"""
Settings module for the Agentic Research Assistant.

Pattern: env-file driven configuration via pydantic-settings. All API keys are
typed as SecretStr to prevent accidental key exposure in logs or tracebacks.
get_settings() is decorated with @lru_cache so the Settings object is constructed
once per process and reused — call get_settings.cache_clear() in tests that need
fresh settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Args:
        groq_api_key: API key for Groq LLM inference.
        google_api_key: API key for Google Gemini.
        tavily_api_key: API key for Tavily web search.
        app_env: Deployment environment.
        log_level: Python logging level name.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    groq_api_key: SecretStr
    google_api_key: SecretStr
    tavily_api_key: SecretStr
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings instance, constructing it on first call.

    Returns:
        The cached Settings object populated from environment / .env file.

    Raises:
        ValidationError: If any required API key is missing from the environment.
    """
    return Settings()
