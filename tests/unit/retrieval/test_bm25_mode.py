from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any

import pytest

from bazi_api.core.config import Settings
from bazi_api.integrations.embeddings import HashEmbeddingProvider
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.retrieval.schemas import RetrievalDocument
from bazi_api.modules.retrieval.service import RetrievalService


def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        embedding_provider="hash",
        vector_backend="memory",
        reranker_provider="lexical",
        database_path=tmp_path / "app.sqlite3",
        embedding_cache_path=tmp_path / "embed.sqlite3",
        qdrant_path=tmp_path / "qdrant",
    )


def document(document_id: str, text: str, **overrides: Any) -> RetrievalDocument:
    values: dict[str, Any] = {
        "id": document_id,
        "kind": "knowledge_card",
        "layer": 3,
        "title": document_id,
        "text": text,
        "source": "test source",
        "school": "test-school",
        "concepts": [],
    }
    values.update(overrides)
    return RetrievalDocument(**values)


@pytest.mark.asyncio
async def test_bm25_mode_ranks_lexical_match_without_chart(tmp_path: Path) -> None:
    docs = [
        document("doc-wealth", "财星宜藏不宜露，比劫夺财须防破财。"),
        document("doc-officer", "正官格喜印绶相生，忌伤官见官。"),
    ]
    service = await RetrievalService.create(
        settings=settings(tmp_path),
        documents=docs,
        embedding_provider=HashEmbeddingProvider(),
    )
    try:
        hits = await service.search(
            query="财运与破财",
            chart=None,
            school="test-school",
            mode="bm25",
            limit=2,
        )

        assert hits
        assert hits[0].document.id == "doc-wealth"
        assert "bm25" in hits[0].matched_by
        # bm25 是单路检索：不得混入 dense / RRF 分量，也不得触发重排。
        assert set(hits[0].component_scores) == {"bm25"}
    finally:
        service.close()


@pytest.mark.asyncio
async def test_bm25_mode_accepts_chart(tmp_path: Path) -> None:
    docs = [document("doc-officer", "正官格喜印绶相生，忌伤官见官。")]
    service = await RetrievalService.create(
        settings=settings(tmp_path),
        documents=docs,
        embedding_provider=HashEmbeddingProvider(),
    )
    try:
        chart = ChartCalculator().calculate(
            BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
        )
        hits = await service.search(
            query="正官格的条件",
            chart=chart,
            school="test-school",
            mode="bm25",
            limit=2,
        )
        assert hits
    finally:
        service.close()


@pytest.mark.asyncio
async def test_bm25_mode_without_chart_still_filters_school_and_review_status(
    tmp_path: Path,
) -> None:
    docs = [
        document("doc-keep", "财星宜藏不宜露，比劫夺财须防破财。"),
        document("doc-other-school", "财星宜藏不宜露。", school="other-school"),
        document("doc-machine-verified", "财星宜藏不宜露。", review_status="machine_verified"),
    ]
    service = await RetrievalService.create(
        settings=settings(tmp_path),
        documents=docs,
        embedding_provider=HashEmbeddingProvider(),
    )
    try:
        hits = await service.search(
            query="财星",
            chart=None,
            school="test-school",
            mode="bm25",
            limit=5,
        )

        assert [hit.document.id for hit in hits] == ["doc-keep"]
    finally:
        service.close()


@pytest.mark.asyncio
async def test_bm25_mode_without_chart_skips_condition_and_exclusion_filters(
    tmp_path: Path,
) -> None:
    chart = ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
    )
    docs = [
        document(
            "doc-excluded",
            "财星宜藏不宜露，比劫夺财须防破财。",
            exclusions=[f"day_master={chart.day_master}"],
        )
    ]
    service = await RetrievalService.create(
        settings=settings(tmp_path),
        documents=docs,
        embedding_provider=HashEmbeddingProvider(),
    )
    try:
        with_chart = await service.search(
            query="财星",
            chart=chart,
            school="test-school",
            mode="bm25",
            limit=2,
        )
        without_chart = await service.search(
            query="财星",
            chart=None,
            school="test-school",
            mode="bm25",
            limit=2,
        )

        assert with_chart == []
        assert [hit.document.id for hit in without_chart] == ["doc-excluded"]
    finally:
        service.close()
