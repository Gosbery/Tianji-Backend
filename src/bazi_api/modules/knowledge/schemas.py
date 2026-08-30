from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    source_id: str
    section: str
    locator: str = ""


class KnowledgeCard(BaseModel):
    id: str
    title: str
    content: str
    school: str = "基础共识"
    concepts: list[str]
    conditions: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    status: Literal["draft", "reviewed", "retired"] = "draft"
    version: int = 1


class SourcePassage(BaseModel):
    id: str
    source_id: str
    title: str
    chapter_path: str
    text: str
    locator: str = ""
    concepts: list[str] = Field(default_factory=list)
