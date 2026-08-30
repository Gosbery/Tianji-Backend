from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalDocument(BaseModel):
    id: str
    kind: Literal["knowledge_card", "source_passage"]
    title: str
    text: str
    source: str
    school: str
    concepts: list[str]
    conditions: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    document: RetrievalDocument
    score: float
    matched_by: list[str]
    component_scores: dict[str, float] = Field(default_factory=dict)
