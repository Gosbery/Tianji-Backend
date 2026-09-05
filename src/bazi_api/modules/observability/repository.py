import json
import sqlite3
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
        evidence_scope: str,
        model_version: str,
        index_version: str,
        hits: list[dict[str, Any]],
        latency_ms: int,
        token_usage: int | None,
        message_id: str,
        generation_model: str,
        prompt_version: str,
        question_policy: str,
        policy_decision: str,
        citations_validated: bool,
        degradation_reason: str,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        values = (
            str(uuid.uuid4()),
            session_id,
            question,
            mode,
            evidence_scope,
            model_version,
            index_version,
            json.dumps(hits, ensure_ascii=False),
            latency_ms,
            token_usage,
            message_id,
            generation_model,
            prompt_version,
            question_policy,
            policy_decision,
            int(citations_validated),
            degradation_reason,
            datetime.now(UTC).isoformat(),
        )
        if connection is not None:
            self._insert(connection, values)
            return
        with self.database.transaction() as managed_connection:
            self._insert(managed_connection, values)

    @staticmethod
    def _insert(connection: sqlite3.Connection, values: tuple[object, ...]) -> None:
        connection.execute(
            """
            INSERT INTO retrieval_traces (
                id, session_id, question, mode, evidence_scope, model_version,
                index_version, hits_json, latency_ms, token_usage, message_id,
                generation_model, prompt_version, question_policy, policy_decision,
                citations_validated, degradation_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, question, mode, evidence_scope, model_version,
                       index_version, hits_json, latency_ms, token_usage, message_id,
                       generation_model, prompt_version, question_policy, policy_decision,
                       citations_validated, degradation_reason, created_at
                FROM retrieval_traces
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                **{
                    key: bool(row[key]) if key == "citations_validated" else row[key]
                    for key in row.keys()
                    if key != "hits_json"
                },
                "hits": json.loads(row["hits_json"]),
            }
            for row in rows
        ]
