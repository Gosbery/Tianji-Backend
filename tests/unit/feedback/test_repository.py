from pathlib import Path

import pytest

from bazi_api.core.errors import (
    FeedbackRateLimitError,
    FeedbackTargetError,
    SessionNotFoundError,
)
from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.feedback.repository import FeedbackRepository


def create_exchange(database: SQLiteDatabase) -> tuple[str, str]:
    conversations = ConversationRepository(database)
    session_id, is_new = conversations.resolve_session()
    with database.transaction() as connection:
        _, message_id = conversations.save_exchange(
            connection,
            session_id=session_id,
            is_new_session=is_new,
            question="问题",
            answer="回答",
        )
    return session_id, message_id


def test_feedback_validates_message_ownership_and_updates_duplicates(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        first_session, first_message = create_exchange(database)
        second_session, _ = create_exchange(database)
        feedback = FeedbackRepository(database)

        feedback.add(first_session, first_message, 1, "初次")
        feedback.add(first_session, first_message, -1, "修正")

        with pytest.raises(FeedbackTargetError):
            feedback.add(second_session, first_message, 1, "跨会话")
        with pytest.raises(SessionNotFoundError):
            feedback.add(
                "00000000-0000-4000-8000-000000000000", first_message, 1, "伪造"
            )

        with database.read() as connection:
            rows = connection.execute(
                "SELECT rating, note FROM feedback WHERE message_id = ?", (first_message,)
            ).fetchall()
        assert [(row["rating"], row["note"]) for row in rows] == [(-1, "修正")]
    finally:
        database.close()


def test_feedback_rate_limit_is_enforced_per_session(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        session_id, message_id = create_exchange(database)
        feedback = FeedbackRepository(database, rate_limit_per_minute=1)

        feedback.add(session_id, message_id, 1, "first")
        with pytest.raises(FeedbackRateLimitError):
            feedback.add(session_id, message_id, -1, "second")
    finally:
        database.close()


def test_feedback_rate_limit_cannot_be_bypassed_with_new_sessions(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        first_session, first_message = create_exchange(database)
        second_session, second_message = create_exchange(database)
        feedback = FeedbackRepository(
            database,
            rate_limit_per_minute=10,
            client_rate_limit_per_minute=1,
        )

        feedback.add(first_session, first_message, 1, "first", "203.0.113.9")
        with pytest.raises(FeedbackRateLimitError):
            feedback.add(second_session, second_message, 1, "second", "203.0.113.9")

        feedback.add(second_session, second_message, 1, "other client", "203.0.113.10")
    finally:
        database.close()
