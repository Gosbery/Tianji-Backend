from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from bazi_api.modules.knowledge.schemas import (
    ReviewStatus,
    SchoolPosition,
    SourceRef,
    VerificationLevel,
)

RetrievalKind = Literal[
    "knowledge_card",
    "modern_annotation",
    "canonical_passage",
    "source_passage",
]


class KnowledgeRuleContext(BaseModel):
    content: str
    card_type: str
    rule: str = ""
    premises: list[str] = Field(default_factory=list)
    conclusion: str = ""
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    break_conditions: list[str] = Field(default_factory=list)
    rescue_conditions: list[str] = Field(default_factory=list)
    priority: int = 50
    priority_note: str = ""
    counterexamples: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    prohibited_uses: list[str] = Field(default_factory=list)
    disagreements: list[SchoolPosition] = Field(default_factory=list)
    case_pillars: list[str] = Field(default_factory=list)
    application_steps: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    annotation_refs: list[str] = Field(default_factory=list)
    case_refs: list[str] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)


class RetrievalDocument(BaseModel):
    id: str
    kind: RetrievalKind
    layer: Literal[1, 2, 3]
    title: str
    text: str
    source: str
    school: str
    concepts: list[str]
    conditions: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    trace_refs: list[str] = Field(default_factory=list)
    graph_refs: list[str] = Field(default_factory=list)
    retrieval_terms: list[str] = Field(default_factory=list)
    normalized_text: str = ""
    rerank_text: str = ""
    rule_context: KnowledgeRuleContext | None = None
    review_status: ReviewStatus = "reviewed"
    verification_level: VerificationLevel = "human_review"
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    warning: str = ""
    unresolved_variants: list[str] = Field(default_factory=list)
    content_sha256: str = ""
    version: int = Field(default=1, ge=1)


class RetrievalHit(BaseModel):
    document: RetrievalDocument
    score: float
    matched_by: list[str]
    component_scores: dict[str, float] = Field(default_factory=dict)
