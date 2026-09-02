from __future__ import annotations

import asyncio
import logging
import time

from bazi_api.integrations.llm import AnswerGenerator
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.observability.repository import TraceRepository
from bazi_api.modules.retrieval.schemas import RetrievalHit
from bazi_api.modules.retrieval.service import RetrievalService

from .repository import ConversationRepository
from .schemas import ChatRequest, ChatResponse, Evidence

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: AnswerGenerator,
        conversations: ConversationRepository,
        traces: TraceRepository,
        charts: ChartCalculator,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator
        self.conversations = conversations
        self.traces = traces
        self.charts = charts

    async def answer(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        chart = self.charts.calculate(request.chart.birth)
        requested_session_id = str(request.session_id) if request.session_id else None
        session_id, is_new_session = await asyncio.to_thread(
            self.conversations.resolve_session, requested_session_id
        )
        retrieval_started = time.perf_counter()
        hits = await self.retrieval.search(
            query=request.question,
            chart=chart,
            school=request.school,
            mode=request.mode,
            limit=self.retrieval.settings.result_limit,
            evidence_scope=request.evidence_scope,
        )
        logger.info(
            "retrieval_completed",
            extra={
                "duration_ms": round((time.perf_counter() - retrieval_started) * 1000, 2),
                "hits": len(hits),
                "mode": request.mode,
            },
        )
        generated = await self.generator.generate(request.question, chart, hits)
        evidence = [
            Evidence(
                id=hit.document.id,
                kind=hit.document.kind,
                layer=hit.document.layer,
                title=hit.document.title,
                quote=hit.document.text,
                source=hit.document.source,
                score=round(hit.score, 6),
                matched_by=hit.matched_by,
                trace_refs=hit.document.trace_refs,
                review_status=hit.document.review_status,
                verification_level=hit.document.verification_level,
                confidence=hit.document.confidence,
                warning=hit.document.warning,
                unresolved_variants=hit.document.unresolved_variants,
            )
            for hit in hits
        ]
        pillars = " ".join(
            f"{pillar.label}{pillar.stem}{pillar.branch}" for pillar in chart.pillars
        )
        evidence.append(
            Evidence(
                id="chart-fact:day-master",
                kind="chart_fact",
                title="命盘事实 · 日主与四柱",
                quote=(
                    f"日主为{chart.day_master}（"
                    f"{chart.day_master_yin_yang}{chart.day_master_element}）；"
                    f"{pillars}。"
                ),
                source=chart.calculation_basis,
                score=1.0,
                matched_by=["deterministic-chart"],
            )
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        message_id = await asyncio.to_thread(
            self._persist_answer,
            session_id,
            is_new_session,
            request,
            generated.answer,
            evidence,
            hits,
            latency_ms,
            generated.token_usage,
        )
        logger.info(
            "chat_answer_persisted",
            extra={"duration_ms": latency_ms, "hits": len(hits), "mode": request.mode},
        )
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=generated.answer,
            evidence=evidence,
            uncertainties=[
                *chart.uncertainties,
                *generated.uncertainties,
                *dict.fromkeys(hit.document.warning for hit in hits if hit.document.warning),
            ],
            followups=generated.followups,
            mode=request.mode,
            latency_ms=latency_ms,
            token_usage=generated.token_usage,
        )

    def _persist_answer(
        self,
        session_id: str,
        is_new_session: bool,
        request: ChatRequest,
        answer: str,
        evidence: list[Evidence],
        hits: list[RetrievalHit],
        latency_ms: int,
        token_usage: int | None,
    ) -> str:
        with self.conversations.database.transaction() as connection:
            _, assistant_message_id = self.conversations.save_exchange(
                connection,
                session_id=session_id,
                is_new_session=is_new_session,
                question=request.question,
                answer=answer,
                assistant_payload={"evidence": [item.model_dump() for item in evidence]},
            )
            self.traces.add(
                session_id=session_id,
                question=request.question,
                mode=request.mode,
                evidence_scope=request.evidence_scope,
                model_version=self.retrieval.model_version,
                index_version=self.retrieval.index_version,
                hits=[hit.model_dump() for hit in hits],
                latency_ms=latency_ms,
                token_usage=token_usage,
                connection=connection,
            )
        return assistant_message_id
