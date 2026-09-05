import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from bazi_api.core.errors import SessionContextMismatchError, SessionNotFoundError
from bazi_api.db.sqlite import SQLiteDatabase


class ConversationRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def resolve_session(
        self,
        session_id: str | None = None,
        *,
        chart_fingerprint: str = "",
        school: str = "基础共识",
        evidence_scope: str = "reviewed_only",
    ) -> tuple[str, bool]:
        if session_id is None:
            return str(uuid.uuid4()), True

        try:
            parsed = uuid.UUID(session_id)
        except ValueError as exc:
            raise SessionNotFoundError() from exc
        if parsed.version != 4 or str(parsed) != session_id.lower():
            raise SessionNotFoundError()

        with self.database.read() as connection:
            session = connection.execute(
                """
                SELECT chart_fingerprint, school, evidence_scope
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        if session is None:
            raise SessionNotFoundError()
        if session["chart_fingerprint"] and (
            session["chart_fingerprint"] != chart_fingerprint
            or session["school"] != school
            or session["evidence_scope"] != evidence_scope
        ):
            raise SessionContextMismatchError()
        return session_id, False

    def recent_context(self, session_id: str, exchanges: int = 4) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, payload_json, turn_index, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY turn_index DESC, created_at DESC,
                         CASE role WHEN 'assistant' THEN 0 ELSE 1 END
                LIMIT ?
                """,
                (session_id, exchanges * 2),
            ).fetchall()
        return [self._message_dict(row) for row in reversed(rows)]

    def history(self, session_id: str) -> dict[str, Any]:
        with self.database.read() as connection:
            session = connection.execute(
                """
                SELECT id, chart_fingerprint, school, evidence_scope
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if session is None:
                raise SessionNotFoundError()
            rows = connection.execute(
                """
                SELECT id, role, content, payload_json, turn_index, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY turn_index, created_at,
                         CASE role WHEN 'user' THEN 0 ELSE 1 END
                """,
                (session_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = self._message_dict(row)
            payload = item.pop("payload")
            response = None
            if row["role"] == "assistant" and {"evidence", "mode", "latency_ms"}.issubset(payload):
                response = {
                    "session_id": session_id,
                    "message_id": row["id"],
                    "answer": row["content"],
                    "uncertainties": [],
                    "followups": [],
                    "token_usage": None,
                    **payload,
                }
            messages.append({**item, "response": response})
        return {
            "session_id": session["id"],
            "chart_fingerprint": session["chart_fingerprint"],
            "school": session["school"],
            "evidence_scope": session["evidence_scope"],
            "messages": messages,
        }

    def save_exchange(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        is_new_session: bool,
        question: str,
        answer: str,
        chart_fingerprint: str = "",
        school: str = "基础共识",
        evidence_scope: str = "reviewed_only",
        assistant_payload: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        now = datetime.now(UTC).isoformat()
        if is_new_session:
            connection.execute(
                """
                INSERT INTO sessions(
                    id, created_at, updated_at, chart_fingerprint, school, evidence_scope
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, now, now, chart_fingerprint, school, evidence_scope),
            )
        else:
            updated = connection.execute(
                """
                UPDATE sessions
                SET updated_at = ?,
                    chart_fingerprint = CASE
                        WHEN chart_fingerprint = '' THEN ? ELSE chart_fingerprint END,
                    school = CASE WHEN chart_fingerprint = '' THEN ? ELSE school END,
                    evidence_scope = CASE
                        WHEN chart_fingerprint = '' THEN ? ELSE evidence_scope END
                WHERE id = ?
                  AND (chart_fingerprint = '' OR (
                    chart_fingerprint = ? AND school = ? AND evidence_scope = ?
                  ))
                """,
                (
                    now,
                    chart_fingerprint,
                    school,
                    evidence_scope,
                    session_id,
                    chart_fingerprint,
                    school,
                    evidence_scope,
                ),
            )
            if updated.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if exists is None:
                    raise SessionNotFoundError()
                raise SessionContextMismatchError()

        turn_index = connection.execute(
            "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]

        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        connection.executemany(
            """
            INSERT INTO messages(
                id, session_id, role, content, payload_json, created_at, turn_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (user_message_id, session_id, "user", question, "{}", now, turn_index),
                (
                    assistant_message_id,
                    session_id,
                    "assistant",
                    answer,
                    json.dumps(assistant_payload or {}, ensure_ascii=False),
                    now,
                    turn_index,
                ),
            ),
        )
        return user_message_id, assistant_message_id

    @staticmethod
    def _message_dict(row: sqlite3.Row) -> dict[str, Any]:
        try:
            payload = json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "payload": payload,
            "turn_index": row["turn_index"],
            "created_at": row["created_at"],
        }
