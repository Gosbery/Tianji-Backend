import uuid
from datetime import UTC, datetime, timedelta

from bazi_api.core.errors import (
    FeedbackRateLimitError,
    FeedbackTargetError,
    SessionNotFoundError,
)
from bazi_api.db.sqlite import SQLiteDatabase


class FeedbackRepository:
    def __init__(
        self,
        database: SQLiteDatabase,
        rate_limit_per_minute: int = 10,
        client_rate_limit_per_minute: int = 30,
    ) -> None:
        self.database = database
        self.rate_limit_per_minute = rate_limit_per_minute
        self.client_rate_limit_per_minute = client_rate_limit_per_minute

    def add(
        self,
        session_id: str,
        message_id: str,
        rating: int,
        note: str,
        client_key: str = "unknown",
    ) -> None:
        now = datetime.now(UTC)
        with self.database.transaction() as connection:
            session_exists = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session_exists is None:
                raise SessionNotFoundError()

            message_exists = connection.execute(
                """
                SELECT 1 FROM messages
                WHERE id = ? AND session_id = ? AND role = 'assistant'
                """,
                (message_id, session_id),
            ).fetchone()
            if message_exists is None:
                raise FeedbackTargetError()

            cutoff = (now - timedelta(minutes=1)).isoformat()
            connection.execute(
                "DELETE FROM feedback_rate_events WHERE created_at < ?", (cutoff,)
            )
            recent_count = connection.execute(
                """
                SELECT COUNT(*) FROM feedback_rate_events
                WHERE session_id = ? AND created_at >= ?
                """,
                (session_id, cutoff),
            ).fetchone()[0]
            client_recent_count = connection.execute(
                """
                SELECT COUNT(*) FROM feedback_rate_events
                WHERE client_key = ? AND created_at >= ?
                """,
                (client_key, cutoff),
            ).fetchone()[0]
            if (
                recent_count >= self.rate_limit_per_minute
                or client_recent_count >= self.client_rate_limit_per_minute
            ):
                raise FeedbackRateLimitError()
            connection.execute(
                """
                INSERT INTO feedback_rate_events(id, session_id, client_key, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), session_id, client_key, now.isoformat()),
            )

            existing_feedback = connection.execute(
                """
                SELECT id FROM feedback
                WHERE session_id = ? AND message_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, message_id),
            ).fetchone()
            if existing_feedback is not None:
                connection.execute(
                    """
                    UPDATE feedback
                    SET rating = ?, note = ?, created_at = ?
                    WHERE id = ?
                    """,
                    (rating, note, now.isoformat(), existing_feedback["id"]),
                )
                return

            connection.execute(
                """
                INSERT INTO feedback(id, session_id, message_id, rating, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    session_id,
                    message_id,
                    rating,
                    note,
                    now.isoformat(),
                ),
            )
