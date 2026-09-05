from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import UUID4, BaseModel, Field

from bazi_api.modules.charts.schemas import ChartFacts
from bazi_api.modules.knowledge.schemas import ReviewStatus, VerificationLevel


class Evidence(BaseModel):
    id: str
    kind: Literal[
        "chart_fact",
        "knowledge_card",
        "modern_annotation",
        "canonical_passage",
        "source_passage",
    ]
    layer: Literal[1, 2, 3] | None = None
    title: str
    quote: str
    source: str
    score: float = 0.0
    matched_by: list[str] = Field(default_factory=list)
    trace_refs: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = "reviewed"
    verification_level: VerificationLevel = "human_review"
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    warning: str = ""
    unresolved_variants: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    chart: ChartFacts
    question: str = Field(min_length=2, max_length=1000)
    session_id: UUID4 | None = None
    expert_id: str = "comprehensive"
    school: str = "基础共识"
    mode: Literal["dense", "hybrid", "hybrid_rerank", "lightrag"] = "hybrid_rerank"
    evidence_scope: Literal["reviewed_only", "personal_preview"] = "reviewed_only"


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    evidence: list[Evidence]
    uncertainties: list[str]
    followups: list[str]
    mode: str
    latency_ms: int
    token_usage: int | None = None
    policy_decision: str = "allow"
    citations_validated: bool = False
    degradation_reason: str = ""


class ConversationMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    turn_index: int
    created_at: datetime
    response: ChatResponse | None = None


class ConversationHistory(BaseModel):
    session_id: str
    chart_fingerprint: str
    school: str
    evidence_scope: str
    messages: list[ConversationMessage]
