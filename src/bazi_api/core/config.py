from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "命盘显微镜 API"
    environment: str = "development"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"

    embedding_provider: str = "hash"
    embedding_model: str = "BAAI/bge-m3"
    remote_embedding_model: str = ""
    vector_backend: str = "memory"
    qdrant_path: Path = BACKEND_ROOT / "data/qdrant"
    reranker_provider: str = "lexical"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    lightrag_base_url: str = ""
    lightrag_api_key: str = ""

    database_path: Path = BACKEND_ROOT / "data/app.db"
    knowledge_path: Path = BACKEND_ROOT / "knowledge"
    evals_path: Path = BACKEND_ROOT / "evals"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
