from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bazi_api.modules.charts.schemas import BirthInput, ChartFacts
from bazi_api.modules.conversations.schemas import ConversationMessage
from bazi_api.modules.knowledge.schemas import EvidenceScope

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
RetrievalMode = Literal["dense", "hybrid", "hybrid_rerank", "lightrag"]


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    birth: BirthInput
    expert_id: str = "liang-xiangrun"
    evidence_scope: EvidenceScope = "personal_preview"
    mode: RetrievalMode = "hybrid_rerank"


class TaskRename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=120)


class TaskMessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=2, max_length=1000)


class GenerationJob(BaseModel):
    id: str
    task_id: str
    question: str
    status: JobStatus
    progress: str = ""
    error: str = ""
    assistant_message_id: str = ""
    attempt_count: int = 0
    recovery_count: int = 0
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime


class TaskSummary(BaseModel):
    id: str
    title: str
    expert_id: str
    expert_version: str
    expert_name: str
    birth: BirthInput
    chart: ChartFacts
    school: str
    evidence_scope: EvidenceScope
    mode: RetrievalMode
    archived: bool
    created_at: datetime
    updated_at: datetime
    active_job: GenerationJob | None = None


class TaskDetail(TaskSummary):
    messages: list[ConversationMessage] = Field(default_factory=list)


class TaskList(BaseModel):
    tasks: list[TaskSummary]
