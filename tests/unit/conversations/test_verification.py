from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bazi_api.core.dependencies import get_container
from bazi_api.core.errors import SessionNotFoundError, TaskNotFoundError
from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.modules.charts.schemas import BirthInput, ChartFacts
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.conversations.router import router
from bazi_api.modules.conversations.schemas import ChatRequest, ConversationHistory
from bazi_api.modules.conversations.service import ChatService
from bazi_api.modules.conversations.verification import VerificationService
from bazi_api.modules.knowledge.schemas import KnowledgeTopic
from bazi_api.modules.observability.repository import TraceRepository
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit


class FakeRetrieval:
    settings = SimpleNamespace(result_limit=6)
    model_version = "test-model"
    index_version = "test-index"

    def __init__(self, hits: list[RetrievalHit] | None = None) -> None:
        self.queries: list[dict[str, object]] = []
        self.hits = hits or []

    async def search(self, **kwargs: object) -> list[RetrievalHit]:
        self.queries.append(kwargs)
        return list(self.hits)


class FakeTasks:
    def __init__(self, chart: ChartFacts | None = None) -> None:
        self.asked: list[str] = []
        self.chart = chart

    def get(self, task_id: str) -> dict[str, object]:
        self.asked.append(task_id)
        if self.chart is None:
            raise TaskNotFoundError()
        return {"chart": self.chart.model_dump(mode="json")}


class FakeGenerator:
    async def generate_direct(self, *_: object, **kwargs: object) -> object:
        from bazi_api.integrations.llm import GenerationResult

        return GenerationResult(answer="## 结论\n\n回答", uncertainties=["不确定"], followups=[])


def _hit(document_id: str = "doc:1", title: str = "财星取用") -> RetrievalHit:
    return RetrievalHit(
        document=RetrievalDocument(
            id=document_id,
            kind="knowledge_card",
            layer=1,
            title=title,
            text="财星宜通门户。",
            source="测试资料",
            school="基础共识",
            concepts=["财星"],
        ),
        score=0.5,
        matched_by=["bm25"],
    )


def _chart() -> ChartFacts:
    return ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
    )


async def _answer(database: SQLiteDatabase, conversations: ConversationRepository):
    service = ChatService(
        retrieval=FakeRetrieval(),  # type: ignore[arg-type]
        generator=FakeGenerator(),  # type: ignore[arg-type]
        conversations=conversations,
        traces=TraceRepository(database),
        charts=ChartCalculator(),
    )
    request = ChatRequest(chart=_chart(), question="看下财运", topic_id="topic:wealth")
    return await service.answer(request)


def _payload(database: SQLiteDatabase, message_id: str) -> dict[str, object]:
    with database.read() as connection:
        row = connection.execute(
            "SELECT payload_json FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
    return json.loads(row["payload_json"])


@pytest.mark.asyncio
async def test_verification_persists_into_message_payload(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        conversations = ConversationRepository(database)
        response = await _answer(database, conversations)

        retrieval = FakeRetrieval()
        verification = VerificationService(
            retrieval=retrieval,  # type: ignore[arg-type]
            conversations=conversations,
            traces=TraceRepository(database),
            tasks=FakeTasks(),  # type: ignore[arg-type]
            topics=[],
        )
        result = await verification.verify(response.session_id, response.message_id)

        assert result.message_id == response.message_id
        assert "看下财运" in result.query
        assert len(retrieval.queries) == 1
        assert retrieval.queries[0]["mode"] == "bm25"
        assert retrieval.queries[0]["chart"] is None
        assert retrieval.queries[0]["school"] == "基础共识"
        assert retrieval.queries[0]["evidence_scope"] == "reviewed_only"

        payload = _payload(database, response.message_id)
        assert payload["verification"]["query"] == result.query
        assert isinstance(payload["verification"]["latency_ms"], int)
        assert payload["verification"]["hits"] == []
        # 回答本身仍然落在同一条 payload 上，核验不覆盖已有内容。
        assert payload["mode"] == "direct"
        assert "evidence" in payload

        # 重跑覆盖：再跑一次不报错，且 payload 被新结果覆盖。
        again = await verification.verify(
            response.session_id, response.message_id, focus="财星强弱"
        )
        assert "财星强弱" in again.query
        assert len(retrieval.queries) == 2
        assert _payload(database, response.message_id)["verification"]["query"] == again.query
    finally:
        database.close()


@pytest.mark.asyncio
async def test_verification_query_construction_and_hits(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        conversations = ConversationRepository(database)
        response = await _answer(database, conversations)

        retrieval = FakeRetrieval([_hit()])
        verification = VerificationService(
            retrieval=retrieval,  # type: ignore[arg-type]
            conversations=conversations,
            traces=TraceRepository(database),
            tasks=FakeTasks(_chart()),  # type: ignore[arg-type]
            topics=[
                KnowledgeTopic(
                    id="topic:wealth",
                    label="财运",
                    priority="high_frequency",
                    description="",
                    query_terms=["财运"],
                    retrieval_terms=["财星", "食伤生财"],
                    related_work_ids=[],
                )
            ],
        )
        result = await verification.verify(response.session_id, response.message_id)

        # 问题 + 话题检索词 + 回答小标题构成检索扩展。
        assert result.query == "看下财运\n检索扩展：财星 食伤生财 结论"
        assert [item.id for item in result.hits] == ["doc:1"]
        assert result.hits[0].matched_by == ["bm25"]
        assert retrieval.queries[0]["limit"] == 8
        assert retrieval.queries[0]["chart"] is not None

        payload = _payload(database, response.message_id)
        assert payload["verification"]["hits"][0]["id"] == "doc:1"
        assert payload["verification"]["hits"][0]["quote"] == "财星宜通门户。"

        # 带 focus 时进入检索问题。
        focused = await verification.verify(
            response.session_id, response.message_id, focus=" 财星 "
        )
        assert focused.query == "财星\n看下财运\n检索扩展：财星 食伤生财 结论"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_history_exposes_verification_hits_as_response_field(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        conversations = ConversationRepository(database)
        response = await _answer(database, conversations)
        verification = VerificationService(
            retrieval=FakeRetrieval([_hit()]),  # type: ignore[arg-type]
            conversations=conversations,
            traces=TraceRepository(database),
            tasks=FakeTasks(),  # type: ignore[arg-type]
        )
        await verification.verify(response.session_id, response.message_id)

        history = ConversationHistory.model_validate(conversations.history(response.session_id))
        assistant = next(item for item in history.messages if item.role == "assistant")
        assert assistant.response is not None
        # payload 里的核验结果是 {query, hits, latency_ms}，对外只暴露 hits 列表。
        assert [item.id for item in assistant.response.verification or []] == ["doc:1"]
        assert assistant.response.mode == "direct"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_verify_rejects_non_assistant_message(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        conversations = ConversationRepository(database)
        response = await _answer(database, conversations)
        with database.read() as connection:
            user_message = connection.execute(
                "SELECT id FROM messages WHERE session_id = ? AND role = 'user'",
                (response.session_id,),
            ).fetchone()

        verification = VerificationService(
            retrieval=FakeRetrieval(),  # type: ignore[arg-type]
            conversations=conversations,
            traces=TraceRepository(database),
            tasks=FakeTasks(),  # type: ignore[arg-type]
        )
        with pytest.raises(SessionNotFoundError):
            await verification.verify(response.session_id, str(user_message["id"]))
        with pytest.raises(SessionNotFoundError):
            await verification.verify("00000000-0000-4000-8000-000000000000", "missing")
    finally:
        database.close()


def test_exchange_returns_none_for_missing_message(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        conversations = ConversationRepository(database)
        import uuid

        assert conversations.exchange(str(uuid.uuid4()), str(uuid.uuid4())) is None
    finally:
        database.close()


def test_verification_endpoint_returns_message_id_query_hits() -> None:
    calls: list[tuple[str, str, str | None]] = []

    class FakeVerification:
        async def verify(self, session_id: str, message_id: str, focus: str | None = None):
            from bazi_api.modules.conversations.schemas import VerificationResponse

            calls.append((session_id, message_id, focus))
            return VerificationResponse(
                message_id=message_id, query="看下财运", hits=[_hit_evidence()]
            )

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_container] = lambda: SimpleNamespace(
        verification=FakeVerification()
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/conversations/8e3b0a5e-1e4e-4b95-9d4a-4f0f9c9f0f11"
            "/messages/msg-1/verification",
            json={"focus": "财星"},
        )
        empty = client.post(
            "/api/v1/conversations/8e3b0a5e-1e4e-4b95-9d4a-4f0f9c9f0f11/messages/msg-1/verification"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["message_id"] == "msg-1"
    assert body["query"] == "看下财运"
    assert body["hits"][0]["id"] == "chart-fact:day-master"
    assert empty.status_code == 200
    assert [call[2] for call in calls] == ["财星", None]


def _hit_evidence():
    from bazi_api.modules.conversations.schemas import Evidence

    return Evidence(
        id="chart-fact:day-master",
        kind="chart_fact",
        title="命盘事实",
        quote="日主为甲。",
        source="测试",
    )
