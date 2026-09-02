from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from bazi_api.integrations.embeddings import EmbeddingProvider


class EmbeddingCache:
    """A local model-version + content-hash vector cache."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model_version TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    PRIMARY KEY (model_version, content_sha256)
                )
                """
            )

    async def vectors(
        self,
        provider: EmbeddingProvider,
        content_hashes: Sequence[str],
        texts: Sequence[str],
    ) -> tuple[list[list[float]], dict[str, int]]:
        if len(content_hashes) != len(texts):
            raise ValueError("content_hashes and texts must have the same length")
        cached = await asyncio.to_thread(
            self._read, provider.model_version, set(content_hashes)
        )
        missing_hashes = [item for item in content_hashes if item not in cached]
        unique_missing = list(dict.fromkeys(missing_hashes))
        text_by_hash = dict(zip(content_hashes, texts, strict=True))
        if unique_missing:
            embedded = await provider.embed([text_by_hash[item] for item in unique_missing])
            cached.update(dict(zip(unique_missing, embedded, strict=True)))
            await asyncio.to_thread(
                self._write, provider.model_version, cached, set(unique_missing)
            )
        return (
            [cached[item] for item in content_hashes],
            {"hits": len(content_hashes) - len(missing_hashes), "misses": len(unique_missing)},
        )

    def _read(self, model_version: str, hashes: set[str]) -> dict[str, list[float]]:
        if not hashes:
            return {}
        placeholders = ",".join("?" for _ in hashes)
        with self.lock:
            rows = self.connection.execute(
                f"SELECT content_sha256, vector_json FROM embedding_cache "
                f"WHERE model_version = ? AND content_sha256 IN ({placeholders})",
                [model_version, *sorted(hashes)],
            ).fetchall()
        return {str(row[0]): json.loads(row[1]) for row in rows}

    def _write(
        self,
        model_version: str,
        vectors: dict[str, list[float]],
        hashes: set[str],
    ) -> None:
        rows = [
            (
                model_version,
                content_hash,
                json.dumps(vectors[content_hash], separators=(",", ":")),
                len(vectors[content_hash]),
            )
            for content_hash in hashes
        ]
        with self.lock, self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?, ?)", rows
            )

    def close(self) -> None:
        with self.lock:
            self.connection.close()
