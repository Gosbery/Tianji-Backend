from datetime import date, time
from pathlib import Path

import pytest

from bazi_api.core.config import Settings
from bazi_api.integrations.embeddings import HashEmbeddingProvider
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.retrieval.service import RetrievalService

BACKEND_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.asyncio
async def test_hybrid_retrieval_returns_expected_card(tmp_path: Path) -> None:
    settings = Settings(
        knowledge_path=BACKEND_ROOT / "knowledge",
        vector_backend="memory",
        database_path=tmp_path / "app.db",
        qdrant_path=tmp_path / "qdrant",
    )
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    service = await RetrievalService.create(
        settings, repository.documents(), HashEmbeddingProvider()
    )
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12, 0)))

    hits = await service.search("偏财是不是一定代表意外之财？", chart, "基础共识", "hybrid", 5)

    assert "god-piancai" in {hit.document.id for hit in hits}
    assert all(hit.matched_by for hit in hits)


@pytest.mark.asyncio
async def test_lightrag_mode_requires_explicit_server(tmp_path: Path) -> None:
    settings = Settings(
        knowledge_path=BACKEND_ROOT / "knowledge",
        vector_backend="memory",
        database_path=tmp_path / "app.db",
        qdrant_path=tmp_path / "qdrant",
    )
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    service = await RetrievalService.create(
        settings, repository.documents(), HashEmbeddingProvider()
    )
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12, 0)))

    with pytest.raises(RuntimeError, match="LIGHTRAG_BASE_URL"):
        await service.search("月令是什么？", chart, "基础共识", "lightrag", 5)
