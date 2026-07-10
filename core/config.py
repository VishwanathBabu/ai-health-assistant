"""
core/config.py
==============
Centralised configuration for the AI Health Assistant.
All settings are loaded from environment variables (.env file).
No hardcoded values anywhere in the codebase — all tunables live here.
"""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # local, free, no API key required


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment / .env file.
    All fields have safe defaults where possible.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="Which LLM backend to use: openai | anthropic | ollama",
    )

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")

    # ── Anthropic ───────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514", description="Anthropic model name"
    )

    # ── Ollama (local) ──────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server",
    )
    ollama_model: str = Field(
        default="llama3",
        description="Ollama model tag to use (e.g. llama3, mistral, phi3)",
    )

    # ── Agent Behaviour ─────────────────────────────────────
    agent_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM temperature for all agents. Low = more deterministic.",
    )
    agent_max_tokens: int = Field(
        default=1024,
        ge=128,
        le=4096,
        description="Max tokens per agent LLM call",
    )
    agent_timeout_seconds: int = Field(
        default=60,  # raised from 30 — local models are slower than cloud APIs
        ge=5,
        le=300,
        description="Hard timeout per agent call in seconds",
    )

    # ── Logging ─────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="DEBUG | INFO | WARNING | ERROR")
    log_format: LogFormat = Field(
        default=LogFormat.CONSOLE, description="json | console"
    )

    # ── API ─────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_reload: bool = Field(default=True)

    # ── Phase 2: Qdrant ─────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="medical_docs")

    # ── Phase 2: PostgreSQL ─────────────────────────────────
    database_url: str = Field(default="")

    # ── Derived helpers ──────────────────────────────────────

    @property
    def active_model(self) -> str:
        """Returns the model name for whichever provider is active."""
        if self.llm_provider == LLMProvider.OPENAI:
            return self.openai_model
        if self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_model
        return self.ollama_model  # LLMProvider.OLLAMA

    @property
    def active_api_key(self) -> str:
        """
        Returns the API key for whichever provider is active.
        Ollama runs locally and requires no key — returns a sentinel
        string so _validate_config() in BaseAgent stays happy without
        special-casing Ollama at the agent level.
        """
        if self.llm_provider == LLMProvider.OPENAI:
            return self.openai_api_key
        if self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        return "ollama-local"  # non-empty sentinel; no real key needed


# Singleton — import this everywhere instead of instantiating again
settings = Settings()
