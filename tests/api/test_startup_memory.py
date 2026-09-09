from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bazi_api.core.config import Settings
from bazi_api.integrations.embeddings import SentenceTransformerEmbeddingProvider
from bazi_api.main import create_app


def test_empty_local_cache_starts_without_loading_embedding_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_calls: list[list[str]] = []

    def unexpected_encode(
        self: SentenceTransformerEmbeddingProvider, texts: list[str]
    ) -> list[list[float]]:
        model_calls.append(texts)
        raise AssertionError("Normal startup must not load the embedding model")

    monkeypatch.setattr(SentenceTransformerEmbeddingProvider, "_encode", unexpected_encode)
    settings = Settings(
        _env_file=None,
        app_access_key="test-startup-access-key-with-32-characters",
        database_path=tmp_path / "app.db",
        embedding_cache_path=tmp_path / "empty-cache.sqlite3",
        qdrant_path=tmp_path / "qdrant",
        embedding_provider="sentence_transformer",
        reranker_provider="lexical",
        vector_backend="memory",
        openai_api_key="",
        anthropic_auth_token="",
    )
    with TestClient(
        create_app(settings), headers={"X-Bazi-Access-Key": settings.app_access_key}
    ) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        data = health.json()
        assert data["vector_documents"] == 0
        assert data["vector_coverage"] == 0.0
        assert data["embedding_build_required"] is True
        assert data["preview_retrieval_documents"] > 1000
        assert data["embedding_model"].startswith("sentence-transformers:")
        chart = client.post("/api/v1/chart", json={"date": "1990-01-01", "time": "12:00"})
        answer = client.post(
            "/api/v1/chat",
            json={
                "chart": chart.json(),
                "question": "如何理解日主？",
                "mode": "hybrid",
            },
        )
        assert answer.status_code == 200
        assert answer.json()["evidence"]
        assert model_calls == []
