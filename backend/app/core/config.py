import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root:
# TECHSUPPORTAI/
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # -------------------------
    # API Keys
    # -------------------------
    google_api_key: str
    langchain_api_key: str

    # -------------------------
    # LangSmith
    # -------------------------
    langchain_tracing_v2: bool = True
    langchain_project: str = "TECHSUPPORTAI"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # -------------------------
    # LLM
    # -------------------------
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.0

    # -------------------------
    # Embeddings
    # -------------------------
    embedding_model: str = "all-MiniLM-L6-v2"

    # -------------------------
    # Document Processing
    # -------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # -------------------------
    # Retrieval
    # -------------------------
    dense_top_k: int = 10
    bm25_top_k: int = 10
    reranker_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
os.environ["LANGCHAIN_TRACING_V2"] = str(
    settings.langchain_tracing_v2
).lower()
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
