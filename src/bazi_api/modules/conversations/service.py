from __future__ import annotations

import time

from bazi_api.integrations.llm import AnswerGenerator
from bazi_api.modules.observability.repository import TraceRepository
from bazi_api.modules.retrieval.service import RetrievalService

from .repository import ConversationRepository
from .schemas import ChatRequest, ChatResponse, Evidence


class ChatService:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: AnswerGenerator,
        conversations: ConversationRepository,
        traces: TraceRepository,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator
        self.conversations = conversations
        self.traces = traces

    async def answer(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        session_id = self.conversations.ensure_session(request.session_id)
        self.conversations.add_message(session_id, "user", request.question)
        hits = await self.retrieval.search(
            query=request.question,
            chart=request.chart,
            school=request.school,
            mode=request.mode,
        )
        generated = await self.generator.generate(request.question, request.chart, hits)
        evidence = [
            Evidence(
                id=hit.document.id,
                kind=hit.document.kind,
                title=hit.document.title,
                quote=hit.document.text,
                source=hit.document.source,
                score=round(hit.score, 6),
                matched_by=hit.matched_by,
            )
            for hit in hits
        ]
        pillars = " ".join(
            f"{pillar.label}{pillar.stem}{pillar.branch}" for pillar in request.chart.pillars
        )
        evidence.append(
            Evidence(
                id="chart-fact:day-master",
                kind="chart_fact",
                title="命盘事实 · 日主与四柱",
                quote=(
                    f"日主为{request.chart.day_master}（"
                    f"{request.chart.day_master_yin_yang}{request.chart.day_master_element}）；"
                    f"{pillars}。"
                ),
                source=request.chart.calculation_basis,
                score=1.0,
                matched_by=["deterministic-chart"],
            )
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        self.conversations.add_message(
            session_id,
            "assistant",
            generated.answer,
            {"evidence": [item.model_dump() for item in evidence]},
        )
        self.traces.add(
            session_id=session_id,
            question=request.question,
            mode=request.mode,
            hits=[hit.model_dump() for hit in hits],
            latency_ms=latency_ms,
            token_usage=generated.token_usage,
        )
        return ChatResponse(
            session_id=session_id,
            answer=generated.answer,
            evidence=evidence,
            uncertainties=[*request.chart.uncertainties, *generated.uncertainties],
            followups=generated.followups,
            mode=request.mode,
            latency_ms=latency_ms,
            token_usage=generated.token_usage,
        )
