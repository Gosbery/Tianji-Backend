import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, time
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock

import pytest

from bazi_api.core.config import Settings
from bazi_api.integrations.embeddings import HashEmbeddingProvider
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.retrieval.schemas import RetrievalDocument
from bazi_api.modules.retrieval.service import (
    CrossEncoderReranker,
    LexicalReranker,
    RetrievalService,
)


def documents() -> list[RetrievalDocument]:
    return [
        RetrievalDocument(
            id=name,
            kind="knowledge_card",
            layer=3,
            title=name,
            text=f"Evidence about {name}.",
            source="test",
            school="test-school",
            concepts=[],
        )
        for name in ["first", "second"]
    ]


def fake_cross_encoder(monkeypatch: pytest.MonkeyPatch, constructor: Any) -> None:
    module = ModuleType("sentence_transformers")
    module.CrossEncoder = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_cross_encoder_defaults_bound_device_sequence_length_and_prediction_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    class FakeModel:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            calls.append(("load", {"model_name": model_name, **kwargs}))

        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
            calls.append(("predict", {"pairs": pairs, **kwargs}))
            return [0.2, 0.8]

    fake_cross_encoder(monkeypatch, FakeModel)
    reranker = CrossEncoderReranker("fake-reranker", LexicalReranker(0.35, 0.65), True)
    assert reranker.score("query", []) == []
    assert calls == []

    assert reranker.score("query", documents()) == [0.2, 0.8]
    assert calls[0] == (
        "load",
        {
            "model_name": "fake-reranker",
            "local_files_only": True,
            "device": "cpu",
            "max_length": 512,
        },
    )
    assert calls[1][1]["batch_size"] == 2
    assert calls[1][1]["show_progress_bar"] is False
    assert [pair[0] for pair in calls[1][1]["pairs"]] == ["query", "query"]
    reranker.score("another query", documents())
    assert sum(name == "load" for name, _ in calls) == 1


def test_failed_model_loading_is_attempted_once_even_for_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    attempts: list[str] = []

    def unavailable(model_name: str, **_: Any) -> None:
        attempts.append(model_name)
        raise RuntimeError("Model is unavailable locally")

    fake_cross_encoder(monkeypatch, unavailable)
    fallback = LexicalReranker(0.35, 0.65)
    reranker = CrossEncoderReranker("missing-model", fallback)
    docs = documents()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: reranker.score("query", docs), range(8)))

    assert attempts == ["missing-model"]
    assert all(scores == fallback.score("query", docs) for scores in results)
    assert sum(record.message == "reranker_fallback" for record in caplog.records) == 1
    assert reranker.model is None


def test_prediction_failure_releases_model_and_does_not_retry_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class OutOfMemoryModel:
        def __init__(self, *_: Any, **__: Any) -> None:
            calls.append("load")

        def predict(self, *_: Any, **__: Any) -> list[float]:
            calls.append("predict")
            raise MemoryError("fake allocation failure")

    fake_cross_encoder(monkeypatch, OutOfMemoryModel)
    fallback = LexicalReranker(0.35, 0.65)
    reranker = CrossEncoderReranker("fake-reranker", fallback)
    docs = documents()

    assert reranker.score("query", docs) == fallback.score("query", docs)
    assert reranker.score("query", docs) == fallback.score("query", docs)
    assert calls == ["load", "predict"]
    assert reranker.model is None


@pytest.mark.asyncio
async def test_service_passes_reranker_controls_and_preserves_model_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeModel:
        def __init__(self, _: str, **kwargs: Any) -> None:
            calls.append(kwargs)

        def predict(self, _: Any, **kwargs: Any) -> list[float]:
            calls.append(kwargs)
            return [0.1, 0.9]

    fake_cross_encoder(monkeypatch, FakeModel)
    config = Settings(
        _env_file=None,
        embedding_provider="hash",
        vector_backend="memory",
        reranker_provider="cross_encoder",
        reranker_model="fake-model",
        reranker_device="cpu",
        reranker_batch_size=1,
        reranker_max_length=256,
        local_models_only=True,
        database_path=tmp_path / "app.sqlite3",
        embedding_cache_path=tmp_path / "embeddings.sqlite3",
    )
    service = await RetrievalService.create(config, documents(), HashEmbeddingProvider(8))
    monkeypatch.setattr(
        service, "_dense_search", AsyncMock(return_value=[("first", 1), ("second", 0.9)])
    )
    monkeypatch.setattr(service.bm25, "search", lambda *_: [("first", 1), ("second", 0.9)])
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12)))
    try:
        assert calls == []
        selected = await service.search("ambiguous query", chart, "test-school", "hybrid_rerank", 2)
        assert [hit.document.id for hit in selected] == ["second", "first"]
        assert calls[0] == {"local_files_only": True, "device": "cpu", "max_length": 256}
        assert calls[1]["batch_size"] == 1
        assert all("reranker" in hit.matched_by for hit in selected)
    finally:
        service.close()
