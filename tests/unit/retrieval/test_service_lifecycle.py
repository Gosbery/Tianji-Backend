import asyncio
import threading
from datetime import date, time
from pathlib import Path

import pytest

from bazi_api.core.config import Settings
from bazi_api.integrations.embeddings import HashEmbeddingProvider
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.retrieval.schemas import RetrievalDocument
from bazi_api.modules.retrieval.service import RetrievalService


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["vectors", "reranker"])
async def test_cancelled_search_drains_inference_before_releasing_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="hash",
        reranker_provider="lexical",
        vector_backend="memory",
        embedding_cache_path=tmp_path / "embeddings.db",
    )
    doc = RetrievalDocument(
        id="day-master",
        kind="knowledge_card",
        layer=3,
        title="Day master",
        text="The day stem is the day master.",
        source="test",
        school="test",
        concepts=[],
    )
    service = await RetrievalService.create(settings, [doc], HashEmbeddingProvider(8))
    entered = threading.Event()
    released = threading.Event()
    finished = threading.Event()
    target = service.vectors if stage == "vectors" else service.reranker
    name = "search" if stage == "vectors" else "score"
    operation = getattr(target, name)

    def blocked(*args):
        entered.set()
        assert released.wait(timeout=5)
        try:
            return operation(*args)
        finally:
            finished.set()

    monkeypatch.setattr(target, name, blocked)
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12)))
    task = asyncio.create_task(
        service.search(
            "day master", chart, "test", "hybrid_rerank" if stage == "reranker" else "dense"
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        released.set()
        await asyncio.gather(task, return_exceptions=True)
        service.close()
