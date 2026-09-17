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
    index_version: str
    default_retrieval_mode: Literal["bm25"]
