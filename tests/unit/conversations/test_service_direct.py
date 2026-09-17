from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import pytest

from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.integrations.llm import GenerationResult
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.conversations.schemas import ChatRequest
from bazi_api.modules.conversations.service import ChatService
from bazi_api.modules.observability.repository import TraceRepository


class FakeDirectGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def generate_direct(
        self,
        question: str,
        chart: object,
        topic_pack: object,
        **kwargs: object,
    ) -> GenerationResult:
        self.calls.append({"question": question, "topic_pack": topic_pack, **kwargs})
        return GenerationResult(
            answer="## 结论\n\n直接解读回答",
            uncertainties=["测试不确定性"],
            followups=["测试追问"],
            token_usage=21,
        )


def chat_request(question: str = "看下财运", topic_id: str | None = "topic:wealth") -> ChatRequest:
    chart = ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
    )
    return ChatRequest(chart=chart, question=question, topic_id=topic_id)


def build_service(database: SQLiteDatabase) -> tuple[ChatService, FakeDirectGenerator]:
    generator = FakeDirectGenerator()
    service = ChatService(
        generator=generator,  # type: ignore[arg-type]
        conversations=ConversationRepository(database),
        traces=TraceRepository(database),
        charts=ChartCalculator(),
    )
    return service, generator


@pytest.mark.asyncio
async def test_direct_mode_skips_retrieval_and_uses_topic_pack(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        service, generator = build_service(database)

        response = await service.answer(chat_request())

        assert response.mode == "direct"
        assert response.answer.startswith("## 结论")
        assert len(generator.calls) == 1
        call = generator.calls[0]
        assert call["topic_pack"] is not None
        assert call["topic_pack"].topic_id == "topic:wealth"
        # direct 模式 evidence 只含三条命盘事实
        assert len(response.evidence) == 3
        assert all(item.kind == "chart_fact" for item in response.evidence)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_direct_mode_without_topic_passes_none_pack(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        service, generator = build_service(database)

        await service.answer(chat_request(question="自由提问", topic_id=None))

        assert generator.calls[0]["topic_pack"] is None
    finally:
        database.close()


@pytest.mark.asyncio
async def test_direct_persists_topic_id_in_payload(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        service, _ = build_service(database)

        response = await service.answer(chat_request())

        with database.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM messages WHERE id = ?", (response.message_id,)
            ).fetchone()

        payload = json.loads(row["payload_json"])
        assert payload["mode"] == "direct"
        assert payload["topic_id"] == "topic:wealth"
    finally:
        database.close()
