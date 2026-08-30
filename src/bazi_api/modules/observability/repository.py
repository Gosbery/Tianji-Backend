import json
import uuid
from datetime import UTC, datetime
from typing import Any

from bazi_api.db.sqlite import SQLiteDatabase


class TraceRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def add(
        self,
        session_id: str,
        question: str,
        mode: str,
        hits: list[dict[str, Any]],
        latency_ms: int,
        token_usage: int | None,
    ) -> None:
        with self.database.lock, self.database.connection:
            self.database.connection.execute(
                "INSERT INTO retrieval_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    session_id,
                    question,
                    mode,
                    json.dumps(hits, ensure_ascii=False),
                    latency_ms,
                    token_usage,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.database.connection.execute(
            "SELECT * FROM retrieval_traces ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{**dict(row), "hits": json.loads(row["hits_json"])} for row in rows]
