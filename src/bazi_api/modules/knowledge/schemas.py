from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ReviewStatus = Literal["draft", "machine_verified", "reviewed", "retired"]
EvidenceScope = Literal["reviewed_only", "personal_preview"]
VerificationLevel = Literal[
    "unverified",
    "single_source_integrity",
    "multi_source_alignment",
    "human_review",
]


class VerificationMetadata(BaseModel):
    """Shared provenance fields for content that can become retrieval evidence."""

    status: ReviewStatus = "draft"
    content_sha256: str = ""
    source_pages: list[str] = Field(default_factory=list)
    collation_method: str = ""
    source_count: int = Field(default=0, ge=0)
    verification_level: VerificationLevel = "unverified"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    unresolved_variants: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def infer_legacy_verification_metadata(self) -> VerificationMetadata:
        if self.status == "reviewed" and self.verification_level == "unverified":
            self.verification_level = "human_review"
            self.confidence = max(self.confidence, 0.95)
        elif self.status == "machine_verified" and self.verification_level == "unverified":
            self.verification_level = "single_source_integrity"
            self.confidence = min(self.confidence, 0.65)
        return self


class SourceRef(BaseModel):
    """A resolvable citation; legacy source/section references remain supported."""

    source_id: str = ""
    section: str = ""
    locator: str = ""
    passage_id: str = ""
    annotation_id: str = ""

    @model_validator(mode="after")
    def require_target(self) -> SourceRef:
        if not any((self.source_id, self.passage_id, self.annotation_id)):
            raise ValueError("source reference requires source_id, passage_id, or annotation_id")
        return self


class CanonicalWork(VerificationMetadata):
    id: str
    title: str
    attributed_author: str = ""
    dynasty: str = ""
    edition: str
    edition_notes: str = ""
    source_url: str = ""
    rights: str = "public_domain"


class CanonicalPassage(VerificationMetadata):
    """Layer 1: a stable, edition-specific unit of a canonical text."""

    id: str
    work_id: str
    volume: str = ""
    chapter_path: list[str]
    sequence: int = Field(ge=1)
    text: str
    normalized_text: str = ""
    locator: str
    concepts: list[str] = Field(default_factory=list)
    graph_refs: list[str] = Field(default_factory=list)


class CanonicalCorpus(BaseModel):
    works: list[CanonicalWork]
    passages: list[CanonicalPassage]


class ModernAnnotation(VerificationMetadata):
    """Layer 2: translation, commentary, or research tied to canonical passages."""

    id: str
    title: str
    kind: Literal["translation", "commentary", "research_note"]
    content: str
    passage_refs: list[str] = Field(default_factory=list)
    author: str = ""
    publication: str = ""
    published_at: date | None = None
    school: str = "基础共识"
    concepts: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    graph_refs: list[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)


class SchoolPosition(BaseModel):
    school: str
    claim: str
    source_refs: list[SourceRef] = Field(default_factory=list)


class KnowledgeCard(VerificationMetadata):
    """Layer 3: an auditable rule with explicit scope and safety boundaries."""

    id: str
    title: str
    content: str
    card_type: Literal["concept", "rule", "method", "boundary", "dispute"] = "concept"
    rule: str = ""
    school: str = "基础共识"
    concepts: list[str]
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    disagreements: list[SchoolPosition] = Field(default_factory=list)
    prohibited_uses: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    annotation_refs: list[str] = Field(default_factory=list)
    graph_refs: list[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    reviewed_by: str = ""
    reviewed_at: date | None = None


class GraphNode(VerificationMetadata):
    """Layer 4: a named entity. Nodes and edges are retrieval hints, not evidence."""

    id: str
    type: Literal[
        "person",
        "concept",
        "rule",
        "school",
        "work",
        "source",
        "annotation",
        "knowledge_card",
    ]
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    school: str = ""
    source_refs: list[SourceRef] = Field(default_factory=list)


class GraphEdge(VerificationMetadata):
    id: str
    source: str
    target: str
    relation: str
    description: str = ""
    school: str = ""
    source_refs: list[SourceRef] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class KnowledgeOverview(BaseModel):
    layers: dict[str, int]
    reviewed: dict[str, int]
    machine_verified: dict[str, int]
    retrieval_documents: int
    preview_documents: int
