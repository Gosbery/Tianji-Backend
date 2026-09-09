from functools import lru_cache
from ipaddress import ip_network
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
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
    app_access_key: str = Field(default="", repr=False, max_length=512)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    anthropic_auth_token: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_chat_model: str = "claude-sonnet-4-6"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=256, le=16_384)
    llm_timeout_seconds: float = Field(default=90.0, gt=0.0, le=600.0)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    http_connect_retries: int = Field(default=2, ge=0, le=5)
    http_request_retries: int = Field(default=2, ge=0, le=5)
    http_max_connections: int = Field(default=50, ge=1, le=500)
    http_max_keepalive_connections: int = Field(default=20, ge=0, le=500)

    embedding_provider: Literal["sentence_transformer", "remote", "hash"] = "sentence_transformer"
    embedding_model: str = "BAAI/bge-m3"
    local_embedding_device: Literal["cpu", "mps", "cuda"] = "cpu"
    local_embedding_batch_size: int = Field(default=2, ge=1, le=32)
    local_embedding_max_seq_length: int = Field(default=512, ge=64, le=8192)
    local_embedding_window_overlap: int = Field(default=64, ge=0, le=1024)
    local_embedding_cache_batch_size: int = Field(default=8, ge=1, le=128)
    remote_embedding_model: str = ""
    vector_backend: Literal["memory", "qdrant"] = "memory"
    qdrant_path: Path = BACKEND_ROOT / "data/qdrant"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    reranker_provider: Literal["cross_encoder", "lexical"] = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_batch_size: int = Field(default=2, ge=1, le=32)
    reranker_max_length: int = Field(default=512, ge=128, le=2048)
    build_embeddings_on_startup: bool = False
    local_models_only: bool = False
    embedding_cache_path: Path = BACKEND_ROOT / "data/embedding-cache.sqlite3"
    dense_recall_limit: int = Field(default=30, ge=1, le=1000)
    sparse_recall_limit: int = Field(default=30, ge=1, le=1000)
    rerank_limit: int = Field(default=20, ge=1, le=1000)
    result_limit: int = Field(default=6, ge=1, le=100)
    retrieval_rrf_k: int = Field(default=60, gt=0)
    bm25_k1: float = Field(default=1.2, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    bm25_heading_boost: float = Field(default=50.0, ge=0.0)
    bm25_topic_boost: float = Field(default=12.0, ge=0.0)
    exact_title_boost: float = Field(default=0.5, ge=0.0)
    chapter_heading_boost: float = Field(default=0.4, ge=0.0)
    chapter_topic_boost: float = Field(default=0.12, ge=0.0)
    rerank_fused_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    rerank_model_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    graph_boost_per_match: float = Field(default=0.006, ge=0.0, le=1.0)
    graph_boost_max: float = Field(default=0.018, ge=0.0, le=1.0)
    concept_boost_per_match: float = Field(default=0.006, ge=0.0, le=1.0)
    concept_boost_max: float = Field(default=0.02, ge=0.0, le=1.0)
    evidence_chain_score_ratio: float = Field(default=0.98, ge=0.0, le=1.0)
    lightrag_base_url: str = ""
    lightrag_api_key: str = ""

    database_path: Path = BACKEND_ROOT / "data/app.db"
    database_busy_timeout_ms: int = Field(default=5000, ge=100, le=60_000)
    knowledge_path: Path = BACKEND_ROOT / "knowledge"
    evals_path: Path = BACKEND_ROOT / "evals"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    cors_allow_credentials: bool = False
    observability_api_key: str = ""
    feedback_rate_limit_per_minute: int = Field(default=10, ge=1, le=1000)
    feedback_client_rate_limit_per_minute: int = Field(default=30, ge=1, le=10_000)
    internal_proxy_secret: str = ""
    trusted_proxy_cidrs: list[str] = Field(default_factory=list)
    legacy_api_enabled: bool = True
    legacy_api_sunset: str = "Thu, 01 Oct 2026 00:00:00 GMT"

    @field_validator("cors_origins", "trusted_proxy_cidrs", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("app_access_key")
    @classmethod
    def validate_access_key(cls, value: str) -> str:
        if value and len(value.strip()) < 32:
            raise ValueError("APP_ACCESS_KEY 至少需要 32 个字符")
        return value

    @model_validator(mode="after")
    def validate_local_embedding_settings(self) -> Self:
        if self.local_embedding_window_overlap >= self.local_embedding_max_seq_length - 2:
            raise ValueError("Local embedding overlap must be smaller than the token window")
        return self

    @model_validator(mode="after")
    def validate_http_settings(self) -> Self:
        if self.http_max_keepalive_connections > self.http_max_connections:
            raise ValueError("HTTP keepalive 连接数不能超过总连接数")
        if self.cors_allow_credentials and "*" in self.cors_origins:
            raise ValueError("启用跨域凭证时不能使用通配 origin")
        proxy_secret_configured = bool(self.internal_proxy_secret)
        proxy_networks_configured = bool(self.trusted_proxy_cidrs)
        if proxy_secret_configured != proxy_networks_configured:
            raise ValueError("INTERNAL_PROXY_SECRET 与 TRUSTED_PROXY_CIDRS 必须同时配置")
        for network in self.trusted_proxy_cidrs:
            ip_network(network, strict=False)
        return self

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
