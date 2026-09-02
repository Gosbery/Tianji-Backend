import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from bazi_api.core.errors import SessionNotFoundError
from bazi_api.db.sqlite import SQLiteDatabase


class ConversationRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def resolve_session(self, session_id: str | None = None) -> tuple[str, bool]:
        if session_id is None:
            return str(uuid.uuid4()), True

        try:
            parsed = uuid.UUID(session_id)
        except ValueError as exc:
            raise SessionNotFoundError() from exc
        if parsed.version != 4 or str(parsed) != session_id.lower():
            raise SessionNotFoundError()

        with self.database.read() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if exists is None:
            raise SessionNotFoundError()
        return session_id, False

    def save_exchange(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        is_new_session: bool,
        question: str,
        answer: str,
        assistant_payload: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        now = datetime.now(UTC).isoformat()
        if is_new_session:
            connection.execute(
                "INSERT INTO sessions(id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
        else:
            updated = connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
            if updated.rowcount != 1:
                raise SessionNotFoundError()

        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        connection.executemany(
            """
            INSERT INTO messages(id, session_id, role, content, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (user_message_id, session_id, "user", question, "{}", now),
                (
                    assistant_message_id,
                    session_id,
                    "assistant",
                    answer,
                    json.dumps(assistant_payload or {}, ensure_ascii=False),
                    now,
                ),
            ),
        )
        return user_message_id, assistant_message_id
