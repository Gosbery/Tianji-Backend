import asyncio
import sqlite3
from pathlib import Path

import pytest

from bazi_api.modules.retrieval.cache import EmbeddingCache


class CountingEmbeddingProvider:
    dimension = 2

    def __init__(self, model_version: str) -> None:
        self.model_version = model_version
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


@pytest.mark.asyncio
async def test_cache_keys_include_model_version_and_content_hash(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "vectors.sqlite3")
    first = CountingEmbeddingProvider("model-v1")

    _, cold = await cache.vectors(first, ["hash-a"], ["原文甲"])
    _, warm = await cache.vectors(first, ["hash-a"], ["原文甲"])
    _, changed = await cache.vectors(first, ["hash-b"], ["原文乙"])
    second = CountingEmbeddingProvider("model-v2")
    _, new_model = await cache.vectors(second, ["hash-a"], ["原文甲"])

    assert cold == {"hits": 0, "misses": 1}
    assert warm == {"hits": 1, "misses": 0}
    assert changed == {"hits": 0, "misses": 1}
    assert new_model == {"hits": 0, "misses": 1}
    assert first.calls == 2
    assert second.calls == 1
    cache.connection.close()


class InterruptingEmbeddingProvider(CountingEmbeddingProvider):
    def __init__(self, *, fail_second_batch: bool = False) -> None:
        super().__init__("model-v1")
        self.fail_second_batch = fail_second_batch
        self.batches: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        if self.fail_second_batch and len(self.batches) == 2:
            raise RuntimeError("interrupted indexing")
        return await super().embed(texts)


@pytest.mark.asyncio
async def test_cache_commits_each_batch_and_resumes_after_failure(tmp_path: Path) -> None:
    path = tmp_path / "vectors.sqlite3"
    hashes = ["h0", "h1", "h0", "h2", "h3", "h4"]
    texts = ["a", "bb", "a", "ccc", "dddd", "eeeee"]
    cache = EmbeddingCache(path)
    interrupted = InterruptingEmbeddingProvider(fail_second_batch=True)
    try:
        with pytest.raises(RuntimeError, match="interrupted indexing"):
            await cache.vectors(interrupted, hashes, texts, batch_size=2)
        assert set(await cache.cached(interrupted, hashes)) == {"h0", "h1"}
    finally:
        cache.close()

    resumed = EmbeddingCache(path)
    provider = InterruptingEmbeddingProvider()
    try:
        vectors, stats = await resumed.vectors(provider, hashes, texts, batch_size=2)
        assert provider.batches == [["ccc", "dddd"], ["eeeee"]]
        assert stats == {"hits": 3, "misses": 3}
        assert vectors == [[float(len(text)), 1.0] for text in texts]
        assert len(vectors) == len(hashes)
    finally:
        resumed.close()


@pytest.mark.asyncio
async def test_cached_only_returns_matching_vectors_without_calling_model(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "vectors.sqlite3")
    provider = CountingEmbeddingProvider("model-v1")
    try:
        await cache.vectors(provider, ["h0"], ["a"])
        assert await cache.cached(provider, ["h0", "missing"]) == {"h0": [1.0, 1.0]}
        assert provider.calls == 1
        other = CountingEmbeddingProvider("model-v2")
        assert await cache.cached(other, ["h0"]) == {}
        assert other.calls == 0
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_cancelled_indexing_preserves_completed_batches(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "vectors.sqlite3")
    entered = asyncio.Event()

    class SlowProvider(CountingEmbeddingProvider):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            if self.calls == 1:
                entered.set()
                await asyncio.Event().wait()
            return await super().embed(texts)

    provider = SlowProvider("model-v1")
    job = asyncio.create_task(cache.vectors(provider, ["h0", "h1"], ["a", "bb"], batch_size=1))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(job, timeout=2)
        assert await cache.cached(provider, ["h0", "h1"]) == {"h0": [1.0, 1.0]}
        resumed = CountingEmbeddingProvider("model-v1")
        _, stats = await cache.vectors(resumed, ["h0", "h1"], ["a", "bb"], batch_size=1)
        assert stats == {"hits": 1, "misses": 1}
        assert resumed.calls == 1
    finally:
        job.cancel()
        await asyncio.gather(job, return_exceptions=True)
        cache.close()


@pytest.mark.asyncio
async def test_cache_reads_large_collections_in_bounded_sql_batches(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "vectors.sqlite3")
    cache.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 501)
    hashes = [f"hash-{index}" for index in range(1100)]
    provider = CountingEmbeddingProvider("model-v1")
    try:
        await cache.vectors(provider, hashes, hashes, batch_size=128)
        assert len(await cache.cached(provider, hashes)) == len(hashes)
    finally:
        cache.close()
