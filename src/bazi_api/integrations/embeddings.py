from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
from typing import Protocol

import httpx

from bazi_api.core.async_utils import run_sync
from bazi_api.core.config import Settings
from bazi_api.core.errors import InvalidUpstreamResponseError, UpstreamServiceError

from .http import post_with_retries

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    dimension: int
    model_version: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic Chinese character n-gram vectors for offline demos and tests."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.model_version = f"hash-blake2b-char-ngram-v1-d{dimension}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(lambda: [self._vector(text) for text in texts])

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
    def __init__(
        self,
        model_name: str,
        local_files_only: bool = False,
        *,
        device: str = "cpu",
        batch_size: int = 2,
        max_seq_length: int = 512,
        window_overlap: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        if max_seq_length < 3 or not 0 <= window_overlap < max_seq_length - 2:
            raise ValueError("Embedding overlap must be smaller than the token window")
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.window_overlap = window_overlap
        self.model_version = (
            f"sentence-transformers:{model_name}:tokens{max_seq_length}"
            f":overlap{window_overlap}:weighted-window-mean-v1"
        )
        self.model = None
        self.dimension = 0
        self._model_lock = threading.Lock()
        self._disabled = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await run_sync(self._encode, texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        with self._model_lock:
            if self._disabled:
                raise RuntimeError("Local embedding is disabled after a failure; restart to retry")
            try:
                if self.model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RuntimeError(
                            "sentence-transformers 未安装，请安装 requirements-local-ml.txt"
                        ) from exc
                    logger.info(
                        "embedding_model_loading",
                        extra={"provider": self.model_version},
                    )
                    self.model = SentenceTransformer(
                        self.model_name,
                        local_files_only=self.local_files_only,
                        device=self.device,
                    )
                    self.model.max_seq_length = self.max_seq_length
                    self.model.eval()
                    get_dimension = getattr(self.model, "get_embedding_dimension", None)
                    if get_dimension is None:
                        get_dimension = self.model.get_sentence_embedding_dimension
                    self.dimension = int(get_dimension())
                return self._encode_windows(texts)
            except Exception:
                self.model = None
                self._disabled = True
                raise

    def _encode_windows(self, texts: list[str]) -> list[list[float]]:
        import torch

        assert self.model is not None
        tokenizer = self.model.tokenizer
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        token_budget = self.max_seq_length - special_tokens
        if token_budget <= self.window_overlap:
            raise ValueError("Embedding token window leaves no space after special tokens")
        sums = [[0.0] * self.dimension for _ in texts]
        batch: list[dict[str, object]] = []
        owners: list[tuple[int, int]] = []

        def encode_batch() -> None:
            features = tokenizer.pad(batch, padding=True, return_tensors="pt")
            features = {name: value.to(self.device) for name, value in features.items()}
            with torch.inference_mode():
                vectors = self.model(features)["sentence_embedding"].detach().cpu().tolist()
            for (owner, weight), vector in zip(owners, vectors, strict=True):
                norm = math.sqrt(sum(value * value for value in vector)) or 1.0
                for index, value in enumerate(vector):
                    sums[owner][index] += value / norm * weight
            batch.clear()
            owners.clear()

        for owner, text in enumerate(texts):
            windows = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_length,
                stride=self.window_overlap,
                return_overflowing_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                padding=False,
                verbose=False,
            )
            for index, token_ids in enumerate(windows["input_ids"]):
                if not isinstance(token_ids, list) or len(token_ids) > self.max_seq_length:
                    raise RuntimeError("Embedding tokenizer must support bounded overflow windows")
                batch.append(
                    {
                        "input_ids": token_ids,
                        "attention_mask": windows["attention_mask"][index],
                    }
                )
                # Overlapping windows contribute according to newly covered tokens.
                overlap = self.window_overlap if index else 0
                owners.append((owner, max(1, len(token_ids) - special_tokens - overlap)))
                if len(batch) == self.batch_size:
                    encode_batch()
        if batch:
            encode_batch()
        result = []
        for vector in sums:
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            result.append([value / norm for value in vector])
        return result


class RemoteEmbeddingProvider:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        if not settings.openai_api_key or not settings.remote_embedding_model:
            raise RuntimeError("远程 embedding 需要 OPENAI_API_KEY 和 REMOTE_EMBEDDING_MODEL")
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url.rstrip("/")
        self.model = settings.remote_embedding_model
        self.model_version = f"remote:{self.base_url}:{self.model}"
        self.dimension = 0
        self.http_client = http_client
        self.timeout = settings.embedding_timeout_seconds
        self.max_retries = settings.http_request_retries

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await post_with_retries(
                self.http_client,
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                payload={"model": self.model, "input": texts},
                timeout=self.timeout,
                max_retries=self.max_retries,
                operation=self.model_version,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "embedding_request_failed",
                extra={"provider": self.model_version},
            )
            raise UpstreamServiceError("远程向量服务暂时不可用") from exc

        try:
            payload = response.json()
            data = payload["data"]
            vectors = [[float(value) for value in item["embedding"]] for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidUpstreamResponseError("远程向量服务返回了无法识别的响应") from exc
        if len(vectors) != len(texts):
            raise InvalidUpstreamResponseError("远程向量服务返回的向量数量不匹配")
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors


def create_embedding_provider(
    settings: Settings,
    http_client: httpx.AsyncClient | None = None,
) -> EmbeddingProvider:
    if settings.embedding_provider == "sentence_transformer":
        return SentenceTransformerEmbeddingProvider(
            settings.embedding_model,
            settings.local_models_only,
            device=settings.local_embedding_device,
            batch_size=settings.local_embedding_batch_size,
            max_seq_length=settings.local_embedding_max_seq_length,
            window_overlap=settings.local_embedding_window_overlap,
        )
    if settings.embedding_provider == "remote":
        if http_client is None:
            raise RuntimeError("远程 embedding 需要由应用容器提供共享 HTTP client")
        return RemoteEmbeddingProvider(settings, http_client)
    return HashEmbeddingProvider()
