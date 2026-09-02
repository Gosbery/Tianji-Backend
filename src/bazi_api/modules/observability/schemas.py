from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class RetrievalTrace(BaseModel):
    id: str
    session_id: str
    question: str
    mode: str
    evidence_scope: str
    model_version: str
    index_version: str
    hits: list[dict[str, Any]]
    latency_ms: int
    token_usage: int | None = None
    created_at: datetime
