import asyncio
from datetime import date, time
from pathlib import Path
from typing import Any

import pytest

from bazi_api.core.config import Settings
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.knowledge.schemas import GraphNode
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit
from bazi_api.modules.retrieval.service import (
    KnowledgeGraphIndex,
    RetrievalService,
    query_mentions_term,
)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {"database_path": tmp_path / "app.db"}
    values.update(overrides)
    return Settings(**values)


def _documents() -> list[RetrievalDocument]:
    return [
        RetrievalDocument(
            id="fire",
            kind="knowledge_card",
            layer=3,
            title="火的基础象义",
            text="火是一种五行关系语言。",
            source="test",
            school="基础共识",
            concepts=["火", "五行"],
        ),
        RetrievalDocument(
            id="train",
            kind="knowledge_card",
            layer=3,
            title="出行说明",
            text="火车是一种交通工具。",
            source="test",
            school="基础共识",
            concepts=["出行"],
        ),
    ]


def test_single_han_concepts_require_boundaries() -> None:
    assert not query_mentions_term("我今天坐火车出行", "火")
    assert not query_mentions_term("甲方提出了方案", "甲")
    assert not query_mentions_term("不要冲动决定", "冲")
    assert query_mentions_term("请解释「火」", "火")
    assert query_mentions_term("火是不是只代表热情", "火")
    assert query_mentions_term("怎样理解五行", "五行")

    graph = KnowledgeGraphIndex(
        [
            GraphNode(
                id="concept:fire",
                type="concept",
                name="火",
                status="reviewed",
                verification_level="human_review",
                reviewed_by="test-reviewer",
                reviewed_at="2026-09-03",
                review_note="测试夹具人工确认",
            )
        ],
        [],
    )
    assert graph.expand("火车如何运行", "reviewed_only") == ([], set())
    assert graph.expand("解释「火」", "reviewed_only")[1] == {"concept:fire"}


@pytest.mark.asyncio
async def test_bm25_index_search_runs_in_a_worker_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = await RetrievalService.create(_settings(tmp_path), _documents())
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12)))
    real_to_thread = asyncio.to_thread
    owners: list[object | None] = []

    async def tracked_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        owners.append(getattr(function, "__self__", None))
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", tracked_to_thread)

    await service.search("请解释「火」", chart, "基础共识", "bm25", 2)

    assert service.bm25 in owners


@pytest.mark.asyncio
async def test_tuning_changes_ranking_and_index_version(tmp_path: Path) -> None:
    baseline = await RetrievalService.create(
        _settings(tmp_path, exact_title_boost=0.5), _documents()
    )
    tuned = await RetrievalService.create(
        _settings(tmp_path, exact_title_boost=0.8), _documents()
    )
    baseline_hits = [
        RetrievalHit(document=baseline.documents["fire"], score=0.0, matched_by=[]),
        RetrievalHit(document=baseline.documents["train"], score=0.6, matched_by=[]),
    ]
    tuned_hits = [
        RetrievalHit(document=tuned.documents["fire"], score=0.0, matched_by=[]),
        RetrievalHit(document=tuned.documents["train"], score=0.6, matched_by=[]),
    ]

    baseline._apply_title_boost("火的基础象义", baseline_hits)
    tuned._apply_title_boost("火的基础象义", tuned_hits)

    assert [item.document.id for item in baseline_hits] == ["train", "fire"]
    assert [item.document.id for item in tuned_hits] == ["fire", "train"]
    assert baseline.index_version != tuned.index_version
