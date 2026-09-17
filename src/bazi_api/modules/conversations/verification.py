from __future__ import annotations

import logging
import re
import time

from bazi_api.core.async_utils import run_sync
from bazi_api.core.errors import SessionNotFoundError, TaskNotFoundError
from bazi_api.modules.charts.schemas import ChartFacts
from bazi_api.modules.knowledge.schemas import KnowledgeTopic
from bazi_api.modules.observability.repository import TraceRepository
from bazi_api.modules.retrieval.service import RetrievalService
from bazi_api.modules.tasks.repository import TaskRepository

from .repository import ConversationRepository
from .schemas import VerificationResponse
from .service import evidence_from_hits, fortune_topic_terms

logger = logging.getLogger(__name__)


class VerificationService:
    """按需查典：对一条已有回答做一次 bm25 核验检索，并把结果落到该消息 payload。"""

    def __init__(
        self,
        retrieval: RetrievalService,
        conversations: ConversationRepository,
        traces: TraceRepository,
        tasks: TaskRepository,
        topics: list[KnowledgeTopic] | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.conversations = conversations
        self.traces = traces
        self.tasks = tasks
        self.topics = topics or []

    async def verify(
        self, session_id: str, message_id: str, focus: str | None = None
    ) -> VerificationResponse:
        started = time.perf_counter()
        exchange = await run_sync(self.conversations.exchange, session_id, message_id)
        if exchange is None or exchange["role"] != "assistant":
            raise SessionNotFoundError()
        question = str(exchange["question"])
        answer = str(exchange["answer"])
        terms = fortune_topic_terms(question, self.topics)
        headings = re.findall(r"^##\s*(.+)$", answer, flags=re.MULTILINE)
        expansion = " ".join(item for item in [*terms, *headings] if item)
        if focus and focus.strip():
            question = f"{focus.strip()}\n{question}"
        query = f"{question}\n检索扩展：{expansion}" if expansion else question
        chart: ChartFacts | None = None
        try:
            task = await run_sync(self.tasks.get, session_id)
            chart = ChartFacts.model_validate(task["chart"])
        except TaskNotFoundError:
            chart = None
        hits = await self.retrieval.search(
            query=query,
            chart=chart,
            school=str(exchange["school"]),
            mode="bm25",
            limit=8,
            evidence_scope=str(exchange["evidence_scope"]),  # type: ignore[arg-type]
        )
        evidence = evidence_from_hits(hits)
        latency_ms = int((time.perf_counter() - started) * 1000)
        await run_sync(
            self.conversations.store_verification,
            session_id,
            message_id,
            {
                "query": query,
                "hits": [item.model_dump() for item in evidence],
                "latency_ms": latency_ms,
            },
        )
        self.traces.add(
            session_id=session_id,
            question=query,
            mode="bm25",
            evidence_scope=str(exchange["evidence_scope"]),
            model_version=self.retrieval.model_version,
            index_version=self.retrieval.index_version,
            hits=[hit.model_dump() for hit in hits],
            latency_ms=latency_ms,
            token_usage=None,
            message_id=message_id,
            generation_model="",
            prompt_version="verification-v1",
            question_policy="evidence_answer",
            policy_decision="allow",
            citations_validated=False,
            degradation_reason="verification",
        )
        logger.info(
            "verification_completed",
            extra={"message_id": message_id, "hits": len(hits), "duration_ms": latency_ms},
        )
        return VerificationResponse(message_id=message_id, query=query, hits=evidence)
