from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from bazi_api.modules.charts.schemas import ChartFacts


class Evidence(BaseModel):
    id: str
    kind: Literal["chart_fact", "knowledge_card", "source_passage"]
    title: str
    quote: str
    source: str
    score: float = 0.0
    matched_by: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    chart: ChartFacts
    question: str = Field(min_length=2, max_length=1000)
    session_id: str | None = None
    school: str = "基础共识"
    mode: Literal["dense", "hybrid", "hybrid_rerank", "lightrag"] = "hybrid"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    evidence: list[Evidence]
    uncertainties: list[str]
    followups: list[str]
    mode: str
    latency_ms: int
    token_usage: int | None = None
