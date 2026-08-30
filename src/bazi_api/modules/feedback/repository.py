import uuid
from datetime import UTC, datetime

from bazi_api.db.sqlite import SQLiteDatabase


class FeedbackRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def add(self, session_id: str, message_id: str | None, rating: int, note: str) -> None:
        with self.database.lock, self.database.connection:
            self.database.connection.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    session_id,
                    message_id,
                    rating,
                    note,
                    datetime.now(UTC).isoformat(),
                ),
            )
