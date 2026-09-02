import sqlite3
from pathlib import Path

import pytest

from bazi_api.db.sqlite import SQLiteDatabase


def test_sqlite_enables_concurrency_pragmas_and_indexes(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db", busy_timeout_ms=1234)
    try:
        with database.read() as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            indexes = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
        assert "idx_retrieval_traces_created_at" in indexes
        assert "idx_messages_session_created" in indexes
        assert "idx_feedback_session_created" in indexes
        assert "idx_feedback_rate_session_created" in indexes
        assert "idx_feedback_rate_client_created" in indexes
    finally:
        database.close()


def test_transaction_rolls_back_all_writes(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    try:
        with pytest.raises(RuntimeError, match="stop"):
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO sessions(id, created_at, updated_at) VALUES (?, ?, ?)",
                    ("session", "now", "now"),
                )
                raise RuntimeError("stop")

        with database.read() as connection:
            assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        database.close()


def test_legacy_feedback_table_is_migrated_with_constraints(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE feedback (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT,
                rating INTEGER NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO sessions VALUES ('session', 'now', 'now');
            INSERT INTO sessions VALUES ('other-session', 'now', 'now');
            INSERT INTO messages VALUES ('assistant', 'session', 'assistant', 'ok', '{}', 'now');
            INSERT INTO messages VALUES (
                'other-assistant', 'other-session', 'assistant', 'ok', '{}', 'now'
            );
            INSERT INTO feedback VALUES ('valid', 'session', 'assistant', 1, 'keep', 'now');
            INSERT INTO feedback VALUES ('null-target', 'session', NULL, -1, 'drop', 'now');
            INSERT INTO feedback VALUES ('orphan', 'session', 'missing', -1, 'drop', 'now');
            INSERT INTO feedback VALUES (
                'cross-session', 'session', 'other-assistant', -1, 'drop', 'now'
            );
            """
        )

    database = SQLiteDatabase(path)
    try:
        with database.read() as connection:
            columns = {
                row[1]: row for row in connection.execute("PRAGMA table_info(feedback)")
            }
            foreign_keys = {
                (row[3], row[2], row[4])
                for row in connection.execute("PRAGMA foreign_key_list(feedback)")
            }
            rows = connection.execute(
                "SELECT id, message_id FROM feedback ORDER BY id"
            ).fetchall()
            archived = connection.execute(
                "SELECT id, message_id FROM feedback_migration_archive ORDER BY id"
            ).fetchall()

        assert bool(columns["message_id"][3])
        assert foreign_keys == {
            ("session_id", "sessions", "id"),
            ("message_id", "messages", "id"),
            ("session_id", "messages", "session_id"),
        }
        assert [tuple(row) for row in rows] == [("valid", "assistant")]
        assert [tuple(row) for row in archived] == [
            ("cross-session", "other-assistant"),
            ("null-target", None),
            ("orphan", "missing"),
        ]

        with pytest.raises(sqlite3.IntegrityError):
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO feedback VALUES ('bad', 'session', NULL, 1, '', 'now')"
                )
        with pytest.raises(sqlite3.IntegrityError):
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO feedback VALUES ('bad', 'session', 'missing', 1, '', 'now')"
                )
        with pytest.raises(sqlite3.IntegrityError):
            with database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO feedback
                    VALUES ('bad', 'session', 'other-assistant', 1, '', 'now')
                    """
                )
    finally:
        database.close()


def test_interrupted_feedback_migration_is_resumed_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.db"
    database = SQLiteDatabase(path)
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO sessions VALUES ('session', 'now', 'now')"
        )
        connection.execute(
            """
            INSERT INTO messages
            VALUES ('assistant', 'session', 'assistant', 'answer', '{}', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO feedback
            VALUES ('feedback', 'session', 'assistant', 1, 'keep', 'now')
            """
        )
    database.close()

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            ALTER TABLE feedback RENAME TO feedback_migration_archive;
            CREATE TABLE feedback (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(message_id, session_id)
                    REFERENCES messages(id, session_id)
            );
            """
        )

    recovered = SQLiteDatabase(path)
    recovered.close()
    reopened = SQLiteDatabase(path)
    try:
        with reopened.read() as connection:
            live_rows = connection.execute(
                "SELECT id, note FROM feedback"
            ).fetchall()
            archived_rows = connection.execute(
                "SELECT COUNT(*) FROM feedback_migration_archive"
            ).fetchone()[0]
            feedback_index_owner = connection.execute(
                """
                SELECT tbl_name FROM sqlite_master
                WHERE type = 'index' AND name = 'idx_feedback_session_created'
                """
            ).fetchone()[0]

        assert [tuple(row) for row in live_rows] == [("feedback", "keep")]
        assert archived_rows == 0
        assert feedback_index_owner == "feedback"
    finally:
        reopened.close()
