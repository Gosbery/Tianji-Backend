import threading

import pytest

from bazi_api.integrations.embeddings import (
    HashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


class FakeVectors:
    def tolist(self) -> list[list[float]]:
        return [[1.0, 0.0]]


@pytest.mark.asyncio
async def test_sentence_transformer_encoding_runs_off_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SentenceTransformerEmbeddingProvider("test-model")
    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def fake_encode(_: list[str]) -> FakeVectors:
        worker_threads.append(threading.get_ident())
        return FakeVectors()

    monkeypatch.setattr(provider, "_encode", fake_encode)

    assert await provider.embed(["文本"]) == [[1.0, 0.0]]
    assert worker_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_hash_embedding_batch_runs_off_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HashEmbeddingProvider()
    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def fake_vector(_: str) -> list[float]:
        worker_threads.append(threading.get_ident())
        return [1.0]

    monkeypatch.setattr(provider, "_vector", fake_vector)

    assert await provider.embed(["文本"]) == [[1.0]]
    assert worker_threads[0] != event_loop_thread
