from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    cards: int = Field(ge=0)
    passages: int = Field(ge=0)
    knowledge_layers: dict[str, int]
    retrieval_documents: int = Field(ge=0)
    preview_retrieval_documents: int = Field(ge=0)
    llm_configured: bool
    embedding_provider: str
    embedding_model: str
    vector_backend: str
    index_version: str
    embedding_cache: dict[str, int]
    vector_documents: int = Field(default=0, ge=0)
    vector_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding_build_required: bool = False
    default_retrieval_mode: Literal["hybrid_rerank"]
