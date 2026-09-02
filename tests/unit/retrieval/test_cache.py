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
