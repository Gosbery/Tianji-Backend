import json
import uuid
from datetime import UTC, datetime
from typing import Any

from bazi_api.db.sqlite import SQLiteDatabase


class ConversationRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def ensure_session(self, session_id: str | None = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self.database.lock, self.database.connection:
            self.database.connection.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
            self.database.connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
        return session_id

    def add_message(
        self, session_id: str, role: str, content: str, payload: dict[str, Any] | None = None
    ) -> str:
        message_id = str(uuid.uuid4())
        with self.database.lock, self.database.connection:
            self.database.connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    json.dumps(payload or {}, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return message_id
