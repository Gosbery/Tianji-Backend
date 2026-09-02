from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    async def generate(self, *_: object) -> GenerationResult:
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
    ) -> GenerationResult:
        await on_chunk("测试")
        await on_chunk("回答")
        return await self.generate()


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
