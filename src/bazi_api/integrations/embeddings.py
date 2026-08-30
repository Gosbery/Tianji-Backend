from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from bazi_api.core.config import Settings


class EmbeddingProvider(Protocol):
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic Chinese character n-gram vectors for offline demos and tests."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        compact = "".join(text.lower().split())
        grams = list(compact)
        grams.extend(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
        grams.extend(compact[index : index + 3] for index in range(max(0, len(compact) - 2)))
        vector = [0.0] * self.dimension
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers 未安装，请安装 requirements-local-ml.txt"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class RemoteEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key or not settings.remote_embedding_model:
            raise RuntimeError("远程 embedding 需要 OPENAI_API_KEY 和 REMOTE_EMBEDDING_MODEL")
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url.rstrip("/")
        self.model = settings.remote_embedding_model
        self.dimension = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
        vectors = [item["embedding"] for item in response.json()["data"]]
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "sentence_transformer":
        return SentenceTransformerEmbeddingProvider(settings.embedding_model)
    if settings.embedding_provider == "remote":
        return RemoteEmbeddingProvider(settings)
    return HashEmbeddingProvider()
