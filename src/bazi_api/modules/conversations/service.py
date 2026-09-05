from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress

from bazi_api.core.errors import ExpertNotFoundError
from bazi_api.integrations.llm import AnswerGenerator, GenerationResult
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.experts.repository import ExpertRepository
from bazi_api.modules.knowledge.schemas import KnowledgeTopic
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
        topics: list[KnowledgeTopic] | None = None,
        experts: ExpertRepository | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.generator = generator
        self.conversations = conversations
        self.traces = traces
        self.charts = charts
        self.topics = topics or []
        self.experts = experts

    async def answer(self, request: ChatRequest) -> ChatResponse:
        return await self._answer(request)

    async def answer_for_job(
        self,
        request: ChatRequest,
        on_progress: Callable[[str], Awaitable[None]],
        on_persist: Callable[[sqlite3.Connection, str], None],
    ) -> ChatResponse:
        return await self._answer(
            request,
            on_progress=on_progress,
            on_persist=on_persist,
        )

    async def _answer(
        self,
        request: ChatRequest,
        on_answer_chunk: Callable[[str], Awaitable[None]] | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_persist: Callable[[sqlite3.Connection, str], None] | None = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        if on_progress is not None:
            await on_progress("正在核验命盘信息")
        chart = self.charts.calculate(request.chart.birth)
        chart_fingerprint = self.chart_fingerprint(chart)
        requested_session_id = str(request.session_id) if request.session_id else None
        session_id, is_new_session = await asyncio.to_thread(
            self.conversations.resolve_session,
            requested_session_id,
            chart_fingerprint=chart_fingerprint,
            school=request.school,
            evidence_scope=request.evidence_scope,
        )
        history = (
            []
            if is_new_session
            else await asyncio.to_thread(self.conversations.recent_context, session_id, 4)
        )
        if on_progress is not None:
            await on_progress("命盘信息已核验，正在检索相关资料")
        try:
            expert = self.experts.get(request.expert_id) if self.experts else None
        except KeyError as exc:
            raise ExpertNotFoundError() from exc
        effective_school = (
            expert.preferred_schools[0]
            if expert and expert.preferred_schools
            else request.school
        )
        allowed_schools = expert.allowed_schools if expert else None
        expert_context = self.experts.prompt_context(expert) if self.experts and expert else ""
        retrieval_started = time.perf_counter()
        hits = await self.retrieval.search(
            query=self._retrieval_query(request.question, history, self.topics),
            chart=chart,
            school=effective_school,
            mode=request.mode,
            limit=self.retrieval.settings.result_limit,
            evidence_scope=request.evidence_scope,
            allowed_schools=allowed_schools,
            preferred_schools=expert.preferred_schools if expert else None,
        )
        logger.info(
            "retrieval_completed",
            extra={
                "duration_ms": round((time.perf_counter() - retrieval_started) * 1000, 2),
                "hits": len(hits),
                "mode": request.mode,
            },
        )
        if on_progress is not None:
            await on_progress(f"已检索到 {len(hits)} 条相关依据，正在组织分析")
        if on_answer_chunk is None:
            generated = await self.generator.generate(
                request.question,
                chart,
                hits,
                school=effective_school,
                evidence_scope=request.evidence_scope,
                history=self._generation_history(history),
                expert_context=expert_context,
            )
        else:
            generated = await self.generator.generate_stream(
                request.question,
                chart,
                hits,
                on_answer_chunk,
                school=effective_school,
                evidence_scope=request.evidence_scope,
                history=self._generation_history(history),
                expert_context=expert_context,
            )
        if on_progress is not None:
            await on_progress("回答已生成，正在整理引用与不确定性")
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
                concepts=hit.document.concepts,
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
        relation_labels = "、".join(item.label for item in chart.structural_relations) or "未检出"
        pattern_labels = "、".join(item.name for item in chart.pattern_candidates) or "未提出"
        day_roots = (
            "、".join(
                f"{item.branch}({item.relation})"
                for item in chart.roots
                if item.stem_position == "day"
            )
            or "未检出"
        )
        evidence.append(
            Evidence(
                id="chart-fact:structure",
                kind="chart_fact",
                title="命盘事实 · 月令与结构关系",
                quote=(
                    f"月令{chart.month_command.branch}，本气{chart.month_command.main_hidden_stem}"
                    f"（{chart.month_command.ten_god}）；格局候选：{pattern_labels}；"
                    f"日主根气：{day_roots}；结构关系：{relation_labels}。"
                    "以上只确认结构出现，不代表合化成功、身强身弱或已经成格。"
                ),
                source=chart.calculation_basis,
                score=1.0,
                matched_by=["deterministic-chart"],
            )
        )
        current_luck = chart.luck.current_cycle
        next_luck = chart.luck.next_cycle
        evidence.append(
            Evidence(
                id="chart-fact:luck",
                kind="chart_fact",
                title="命盘事实 · 大运",
                quote=(
                    f"按{chart.luck.direction_label}计算，起运时间为"
                    f"{chart.luck.start_at:%Y-%m-%d %H:%M}；"
                    f"当前大运为{current_luck.ganzhi}（{current_luck.start_year}—"
                    f"{current_luck.end_year}）"
                    if current_luck
                    else f"按{chart.luck.direction_label}计算，尚未进入第一步大运；"
                )
                + (
                    f"；下一大运为{next_luck.ganzhi}（{next_luck.start_year}—"
                    f"{next_luck.end_year}）。"
                    if next_luck
                    else "；当前排运范围内没有下一步大运。"
                ),
                source=chart.calculation_basis,
                score=1.0,
                matched_by=["deterministic-chart", "luck-cycle"],
                concepts=["大运", "起运", chart.luck.direction_label],
            )
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        uncertainties = list(
            dict.fromkeys(
                [
                    *chart.uncertainties,
                    *generated.uncertainties,
                    *(hit.document.warning for hit in hits if hit.document.warning),
                ]
            )
        )
        if on_progress is not None:
            await on_progress("资料整理完成，正在保存本次分析")
        message_id = await asyncio.to_thread(
            self._persist_answer,
            session_id,
            is_new_session,
            request,
            generated.answer,
            evidence,
            uncertainties,
            hits,
            latency_ms,
            generated,
            chart_fingerprint,
            on_persist,
        )
        logger.info(
            "chat_answer_persisted",
            extra={"duration_ms": latency_ms, "hits": len(hits), "mode": request.mode},
        )
        if on_progress is not None:
            await on_progress("分析完成")
        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=generated.answer,
            evidence=evidence,
            uncertainties=uncertainties,
            followups=generated.followups,
            mode=request.mode,
            latency_ms=latency_ms,
            token_usage=generated.token_usage,
            policy_decision=generated.policy_decision,
            citations_validated=generated.citations_validated,
            degradation_reason=generated.degradation_reason,
        )

    async def validate_session_context(self, request: ChatRequest) -> None:
        if request.session_id is None:
            return
        chart = self.charts.calculate(request.chart.birth)
        await asyncio.to_thread(
            self.conversations.resolve_session,
            str(request.session_id),
            chart_fingerprint=self.chart_fingerprint(chart),
            school=request.school,
            evidence_scope=request.evidence_scope,
        )

    @staticmethod
    def chart_fingerprint(chart: object) -> str:
        birth = getattr(chart, "birth")
        payload = birth.model_dump(mode="json", exclude={"name"})
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _generation_history(history: list[dict[str, object]]) -> list[dict[str, str]]:
        return [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]

    @staticmethod
    def _retrieval_query(
        question: str,
        history: list[dict[str, object]],
        topics: list[KnowledgeTopic] | None = None,
    ) -> str:
        followup_markers = (
            "它",
            "这个",
            "上述",
            "前面",
            "刚才",
            "那",
            "其",
            "这些",
            "继续",
            "例外",
        )
        contextual = len(question) <= 18 or any(item in question for item in followup_markers)
        topic_terms = ChatService._fortune_topic_terms(question, topics)
        if not history or not contextual:
            if not topic_terms:
                return question
            return f"{question}\n检索扩展：{' '.join(topic_terms)}"
        previous_question = next(
            (str(item["content"]) for item in reversed(history) if item.get("role") == "user"),
            "",
        )
        assistant = next(
            (item for item in reversed(history) if item.get("role") == "assistant"),
            {},
        )
        payload = assistant.get("payload") if isinstance(assistant, dict) else {}
        evidence = payload.get("evidence", []) if isinstance(payload, dict) else []
        titles: list[str] = []
        concepts: list[str] = []
        for item in evidence if isinstance(evidence, list) else []:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("title"), str):
                titles.append(item["title"])
            raw_concepts = item.get("concepts", [])
            if isinstance(raw_concepts, list):
                concepts.extend(str(value) for value in raw_concepts if value)
        context_terms = list(dict.fromkeys([*titles[:4], *concepts[:8]]))
        expansion = " ".join(
            item for item in [*topic_terms, previous_question, *context_terms] if item
        )
        return f"{question}\n检索扩展：{expansion}" if expansion else question

    @staticmethod
    def _fortune_topic_terms(
        question: str, topics: list[KnowledgeTopic] | None = None
    ) -> list[str]:
        if topics is None:
            from pathlib import Path

            import yaml

            from bazi_api.modules.knowledge.schemas import TopicCatalog

            path = Path(__file__).resolve().parents[4] / "knowledge/catalog/young-user-topics.yml"
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
            topics = TopicCatalog.model_validate(payload).topics if payload else []
        terms: list[str] = []
        for topic in topics:
            if any(marker in question for marker in topic.query_terms):
                terms.extend(topic.retrieval_terms)
        return list(dict.fromkeys(terms))

    async def answer_stream(self, request: ChatRequest) -> AsyncIterator[dict[str, object]]:
        queue: asyncio.Queue[dict[str, object] | BaseException] = asyncio.Queue()

        async def on_chunk(content: str) -> None:
            await queue.put({"type": "chunk", "content": content})

        async def on_progress(message: str) -> None:
            await queue.put({"type": "progress", "message": message})

        async def produce() -> None:
            try:
                response = await self._answer(request, on_chunk, on_progress)
                await queue.put({"type": "done", "response": response.model_dump(mode="json")})
            except BaseException as exc:
                await queue.put(exc)

        producer = asyncio.create_task(produce())
        yield {"type": "start"}
        try:
            while True:
                event = await queue.get()
                if isinstance(event, BaseException):
                    raise event
                yield event
                if event["type"] == "done":
                    break
        finally:
            if not producer.done():
                producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer

    def _persist_answer(
        self,
        session_id: str,
        is_new_session: bool,
        request: ChatRequest,
        answer: str,
        evidence: list[Evidence],
        uncertainties: list[str],
        hits: list[RetrievalHit],
        latency_ms: int,
        generated: GenerationResult,
        chart_fingerprint: str,
        on_persist: Callable[[sqlite3.Connection, str], None] | None = None,
    ) -> str:
        with self.conversations.database.transaction() as connection:
            _, assistant_message_id = self.conversations.save_exchange(
                connection,
                session_id=session_id,
                is_new_session=is_new_session,
                question=request.question,
                answer=answer,
                chart_fingerprint=chart_fingerprint,
                school=request.school,
                evidence_scope=request.evidence_scope,
                assistant_payload={
                    "evidence": [item.model_dump() for item in evidence],
                    "uncertainties": uncertainties,
                    "followups": generated.followups,
                    "mode": request.mode,
                    "latency_ms": latency_ms,
                    "token_usage": generated.token_usage,
                    "policy_decision": generated.policy_decision,
                    "question_policy": generated.question_policy,
                    "citations_validated": generated.citations_validated,
                    "degradation_reason": generated.degradation_reason,
                },
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
                token_usage=generated.token_usage,
                message_id=assistant_message_id,
                generation_model=generated.model_version,
                prompt_version=generated.prompt_version,
                question_policy=generated.question_policy,
                policy_decision=generated.policy_decision,
                citations_validated=generated.citations_validated,
                degradation_reason=generated.degradation_reason,
                connection=connection,
            )
            if on_persist is not None:
                on_persist(connection, assistant_message_id)
        return assistant_message_id
