from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import pytest

from bazi_api.core.errors import SessionContextMismatchError
from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.integrations.llm import GenerationResult
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.conversations.schemas import ChatRequest
from bazi_api.modules.conversations.service import ChatService
from bazi_api.modules.observability.repository import TraceRepository


class FakeRetrieval:
    settings = SimpleNamespace(result_limit=6)
    model_version = "test-model"
    index_version = "test-index"

    async def search(self, **_: object) -> list[object]:
        return []


class FakeGenerator:
    calls: list[dict[str, object]]

    def __init__(self) -> None:
        self.calls = []

    async def generate(self, *_: object, **kwargs: object) -> GenerationResult:
        self.calls.append(kwargs)
        return GenerationResult(
            answer="测试回答",
            uncertainties=["测试不确定性"],
            followups=["测试追问"],
            token_usage=12,
        )

    async def generate_stream(
        self,
        _question: object,
        _chart: object,
        _hits: object,
        on_chunk: Callable[[str], Awaitable[None]],
        **kwargs: object,
    ) -> GenerationResult:
        result = await self.generate(**kwargs)
        await on_chunk(result.answer)
        return result


def chat_request() -> ChatRequest:
    chart = ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
    )
    return ChatRequest(chart=chart, question="如何理解日主？", mode="hybrid")


def build_service(database: SQLiteDatabase) -> tuple[ChatService, TraceRepository]:
    traces = TraceRepository(database)
    service = ChatService(
        retrieval=FakeRetrieval(),  # type: ignore[arg-type]
        generator=FakeGenerator(),  # type: ignore[arg-type]
        conversations=ConversationRepository(database),
        traces=traces,
        charts=ChartCalculator(),
    )
    return service, traces


@pytest.mark.asyncio
async def test_chat_persists_complete_exchange_and_trace_atomically(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        service, _ = build_service(database)

        response = await service.answer(chat_request())

        with database.read() as connection:
            session_count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages = connection.execute(
                "SELECT id, role FROM messages ORDER BY created_at, role DESC"
            ).fetchall()
            trace_count = connection.execute(
                "SELECT COUNT(*) FROM retrieval_traces"
            ).fetchone()[0]
        assert session_count == 1
        assert {row["role"] for row in messages} == {"user", "assistant"}
        assert response.message_id == next(
            row["id"] for row in messages if row["role"] == "assistant"
        )
        assert trace_count == 1
    finally:
        database.close()


@pytest.mark.asyncio
async def test_chat_stream_emits_chunks_and_terminal_response(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        service, _ = build_service(database)

        events = [event async for event in service.answer_stream(chat_request())]

        assert events[0] == {"type": "start"}
        assert events[-1]["type"] == "done"
        assert events[-1]["response"]["answer"] == "测试回答"  # type: ignore[index]
        chunks = [event["content"] for event in events if event["type"] == "chunk"]
        assert "".join(chunks) == "测试回答"
        progress = [event["message"] for event in events if event["type"] == "progress"]
        assert progress[0] == "正在核验命盘信息"
        assert "已检索到 0 条相关依据" in str(progress[2])
        assert progress[-1] == "分析完成"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_chat_recalculates_client_supplied_chart(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        service, _ = build_service(database)
        request = chat_request()
        request.chart.day_master = "甲"
        request.chart.pillars = []
        request.chart.calculation_basis = "客户端伪造"

        response = await service.answer(request)

        chart_fact = next(item for item in response.evidence if item.kind == "chart_fact")
        assert "日主为丙" in chart_fact.quote
        assert "年柱己巳" in chart_fact.quote
        assert chart_fact.source != "客户端伪造"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_trace_failure_rolls_back_session_and_both_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        service, traces = build_service(database)

        def fail_trace(**_: object) -> None:
            raise RuntimeError("trace failed")

        monkeypatch.setattr(traces, "add", fail_trace)

        with pytest.raises(RuntimeError, match="trace failed"):
            await service.answer(chat_request())

        with database.read() as connection:
            counts = [
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("sessions", "messages", "retrieval_traces")
            ]
        assert counts == [0, 0, 0]
    finally:
        database.close()


def test_repository_binds_context_and_restores_ordered_history(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    conversations = ConversationRepository(database)
    try:
        session_id, is_new = conversations.resolve_session(
            chart_fingerprint="chart-a",
            school="子平格局法",
            evidence_scope="personal_preview",
        )
        with database.transaction() as connection:
            conversations.save_exchange(
                connection,
                session_id=session_id,
                is_new_session=is_new,
                question="正官格有哪些条件？",
                answer="第一轮 [1]",
                chart_fingerprint="chart-a",
                school="子平格局法",
                evidence_scope="personal_preview",
                assistant_payload={
                    "evidence": [],
                    "uncertainties": ["仍需核对"],
                    "followups": [],
                    "mode": "hybrid",
                    "latency_ms": 1,
                    "policy_decision": "allow",
                    "citations_validated": True,
                    "degradation_reason": "",
                },
            )
        with database.transaction() as connection:
            conversations.save_exchange(
                connection,
                session_id=session_id,
                is_new_session=False,
                question="它有哪些例外？",
                answer="第二轮 [1]",
                chart_fingerprint="chart-a",
                school="子平格局法",
                evidence_scope="personal_preview",
                assistant_payload={
                    "evidence": [],
                    "uncertainties": ["仍需核对"],
                    "followups": [],
                    "mode": "hybrid",
                    "latency_ms": 1,
                    "policy_decision": "allow",
                    "citations_validated": True,
                    "degradation_reason": "",
                },
            )

        recent = conversations.recent_context(session_id)
        restored = conversations.history(session_id)

        assert [item["content"] for item in recent] == [
            "正官格有哪些条件？",
            "第一轮 [1]",
            "它有哪些例外？",
            "第二轮 [1]",
        ]
        assert [item["turn_index"] for item in restored["messages"]] == [1, 1, 2, 2]
        assert restored["messages"][-1]["response"]["answer"] == "第二轮 [1]"
        with pytest.raises(SessionContextMismatchError):
            conversations.resolve_session(
                session_id,
                chart_fingerprint="chart-a",
                school="基础共识",
                evidence_scope="personal_preview",
            )
    finally:
        database.close()


def test_followup_retrieval_uses_prior_question_and_evidence_not_assistant_claims() -> None:
    history = [
        {"role": "user", "content": "正官格有哪些条件？", "payload": {}},
        {
            "role": "assistant",
            "content": "未经证据约束的自由发挥不应进入检索",
            "payload": {
                "evidence": [
                    {"title": "正官格的成格条件", "concepts": ["正官", "月令"]}
                ]
            },
        },
    ]

    query = ChatService._retrieval_query("它有哪些例外？", history)

    assert "正官格有哪些条件" in query
    assert "正官格的成格条件" in query
    assert "正官" in query and "月令" in query
    assert "自由发挥" not in query


@pytest.mark.parametrize(
    ("question", "expected_terms"),
    [
        ("父母对我的助益如何？", ["六亲", "正印", "偏财"]),
        ("兄弟姐妹和朋友人脉怎么样？", ["比肩", "劫财", "贵人"]),
        ("分析桃花、事业和财运", ["夫妻宫", "正官", "正财", "大运"]),
        ("看看子女和异地发展", ["时柱", "食神", "冲"]),
    ],
)
def test_retrieval_query_expands_common_fortune_topics(
    question: str, expected_terms: list[str]
) -> None:
    query = ChatService._retrieval_query(question, [])

    assert question in query
    assert all(term in query for term in expected_terms)
