import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from bazi_api.core.errors import BaziApiError

logger = logging.getLogger(__name__)


class SQLiteDatabase:
    def __init__(self, path: Path, busy_timeout_ms: int = 5000) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            path,
            check_same_thread=False,
            timeout=busy_timeout_ms / 1000,
        )
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.busy_timeout_ms = busy_timeout_ms
        self._initialize()

    def _initialize(self) -> None:
        with self.lock:
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            journal_mode = self.connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        with self.lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    chart_fingerprint TEXT NOT NULL DEFAULT '',
                    school TEXT NOT NULL DEFAULT '基础共识',
                    evidence_scope TEXT NOT NULL DEFAULT 'reviewed_only'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    turn_index INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_id_session
                    ON messages(id, session_id);
                CREATE TABLE IF NOT EXISTS retrieval_traces (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    evidence_scope TEXT NOT NULL DEFAULT 'reviewed_only',
                    model_version TEXT NOT NULL DEFAULT '',
                    index_version TEXT NOT NULL DEFAULT '',
                    hits_json TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    token_usage INTEGER,
                    message_id TEXT NOT NULL DEFAULT '',
                    generation_model TEXT NOT NULL DEFAULT '',
                    prompt_version TEXT NOT NULL DEFAULT '',
                    question_policy TEXT NOT NULL DEFAULT 'evidence_answer',
                    policy_decision TEXT NOT NULL DEFAULT 'allow',
                    citations_validated INTEGER NOT NULL DEFAULT 0,
                    degradation_reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS feedback (
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
                CREATE TABLE IF NOT EXISTS feedback_rate_events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    client_key TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    expert_id TEXT NOT NULL,
                    expert_version TEXT NOT NULL,
                    expert_name TEXT NOT NULL,
                    birth_json TEXT NOT NULL,
                    chart_json TEXT NOT NULL,
                    school TEXT NOT NULL,
                    evidence_scope TEXT NOT NULL DEFAULT 'personal_preview',
                    mode TEXT NOT NULL DEFAULT 'hybrid_rerank',
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(id) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    assistant_message_id TEXT NOT NULL DEFAULT '',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    recovery_count INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );
                """
            )
            trace_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(retrieval_traces)")
            }
            for name, definition in {
                "evidence_scope": "TEXT NOT NULL DEFAULT 'reviewed_only'",
                "model_version": "TEXT NOT NULL DEFAULT ''",
                "index_version": "TEXT NOT NULL DEFAULT ''",
                "message_id": "TEXT NOT NULL DEFAULT ''",
                "generation_model": "TEXT NOT NULL DEFAULT ''",
                "prompt_version": "TEXT NOT NULL DEFAULT ''",
                "question_policy": "TEXT NOT NULL DEFAULT 'evidence_answer'",
                "policy_decision": "TEXT NOT NULL DEFAULT 'allow'",
                "citations_validated": "INTEGER NOT NULL DEFAULT 0",
                "degradation_reason": "TEXT NOT NULL DEFAULT ''",
            }.items():
                if name not in trace_columns:
                    self.connection.execute(
                        f"ALTER TABLE retrieval_traces ADD COLUMN {name} {definition}"
                    )
            session_columns = {
                row[1] for row in self.connection.execute("PRAGMA table_info(sessions)")
            }
            for name, definition in {
                "chart_fingerprint": "TEXT NOT NULL DEFAULT ''",
                "school": "TEXT NOT NULL DEFAULT '基础共识'",
                "evidence_scope": "TEXT NOT NULL DEFAULT 'reviewed_only'",
            }.items():
                if name not in session_columns:
                    self.connection.execute(f"ALTER TABLE sessions ADD COLUMN {name} {definition}")
            message_columns = {
                row[1] for row in self.connection.execute("PRAGMA table_info(messages)")
            }
            if "turn_index" not in message_columns:
                self.connection.execute(
                    "ALTER TABLE messages ADD COLUMN turn_index INTEGER NOT NULL DEFAULT 0"
                )
            rate_event_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(feedback_rate_events)")
            }
            if "client_key" not in rate_event_columns:
                self.connection.execute(
                    """
                    ALTER TABLE feedback_rate_events
                    ADD COLUMN client_key TEXT NOT NULL DEFAULT 'unknown'
                    """
                )
            self._migrate_feedback_table()
            self.connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                    ON messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_retrieval_traces_created_at
                    ON retrieval_traces(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_session_created
                    ON feedback(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_rate_session_created
                    ON feedback_rate_events(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_rate_client_created
                    ON feedback_rate_events(client_key, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_updated
                    ON tasks(archived, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_generation_jobs_status_created
                    ON generation_jobs(status, created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_generation_jobs_active_task
                    ON generation_jobs(task_id)
                    WHERE status IN ('queued', 'running');
                """
            )
        logger.info(
            "sqlite_initialized",
            extra={"provider": "sqlite", "journal_mode": journal_mode},
        )

    def _migrate_feedback_table(self) -> None:
        self.connection.execute("SAVEPOINT migrate_feedback_schema")
        try:
            migrated_count, archived_count = self._migrate_feedback_table_locked()
        except BaseException:
            self.connection.execute("ROLLBACK TO SAVEPOINT migrate_feedback_schema")
            self.connection.execute("RELEASE SAVEPOINT migrate_feedback_schema")
            raise
        else:
            self.connection.execute("RELEASE SAVEPOINT migrate_feedback_schema")
        if migrated_count or archived_count:
            logger.warning(
                "sqlite_feedback_schema_migrated",
                extra={
                    "archived_rows": archived_count,
                    "provider": "sqlite",
                    "hits": migrated_count,
                },
            )

    def _migrate_feedback_table_locked(self) -> tuple[int, int]:
        # A named index follows the table during RENAME; recreate it on the live table below.
        self.connection.execute("DROP INDEX IF EXISTS idx_feedback_session_created")
        columns = {
            row[1]: row for row in self.connection.execute("PRAGMA table_info(feedback)")
        }
        foreign_keys = {
            (row[3], row[2], row[4])
            for row in self.connection.execute("PRAGMA foreign_key_list(feedback)")
        }
        expected_foreign_keys = {
            ("session_id", "sessions", "id"),
            ("message_id", "messages", "id"),
            ("session_id", "messages", "session_id"),
        }
        message_id = columns.get("message_id")
        schema_is_current = (
            message_id is not None
            and bool(message_id[3])
            and expected_foreign_keys.issubset(foreign_keys)
        )
        archive_names = [
            name
            for name in (
                "feedback_migration_archive",
                *(f"feedback_migration_archive_v{version}" for version in range(2, 11)),
            )
            if self._table_exists(name)
        ]
        if not schema_is_current:
            archive_name = next(
                name
                for name in (
                    "feedback_migration_archive",
                    *(f"feedback_migration_archive_v{version}" for version in range(2, 11)),
                )
                if name not in archive_names
            )
            self.connection.execute(f"ALTER TABLE feedback RENAME TO {archive_name}")
            self._create_feedback_table()
            archive_names.append(archive_name)

        before_count = self.connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        for archive_name in archive_names:
            self._restore_feedback_archive(archive_name)
        after_count = self.connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        archived_count = sum(
            self.connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in archive_names
        )
        return after_count - before_count, archived_count

    def _restore_feedback_archive(self, archive_name: str) -> None:
        required_columns = {
            "id",
            "session_id",
            "message_id",
            "rating",
            "note",
            "created_at",
        }
        archive_columns = {
            row[1]
            for row in self.connection.execute(f"PRAGMA table_info({archive_name})")
        }
        if not required_columns.issubset(archive_columns):
            return
        self.connection.execute(
            f"""
            INSERT OR IGNORE INTO feedback(
                id, session_id, message_id, rating, note, created_at
            )
            SELECT f.id, f.session_id, f.message_id, f.rating, f.note, f.created_at
            FROM {archive_name} AS f
            JOIN sessions AS s ON s.id = f.session_id
            JOIN messages AS m
              ON m.id = f.message_id
             AND m.session_id = f.session_id
             AND m.role = 'assistant'
            WHERE f.message_id IS NOT NULL
            """
        )
        self.connection.execute(
            f"""
            DELETE FROM {archive_name}
            WHERE EXISTS (
                SELECT 1 FROM feedback AS current
                WHERE current.id = {archive_name}.id
                  AND current.session_id = {archive_name}.session_id
                  AND current.message_id = {archive_name}.message_id
                  AND current.rating = {archive_name}.rating
                  AND current.note = {archive_name}.note
                  AND current.created_at = {archive_name}.created_at
            )
            """
        )

    def _create_feedback_table(self) -> None:
        self.connection.execute(
            """
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
            )
            """
        )

    def _table_exists(self, name: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone() is not None

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            yield self.connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                yield self.connection
            except BaziApiError:
                self.connection.rollback()
                raise
            except BaseException:
                self.connection.rollback()
                logger.exception("sqlite_transaction_rolled_back")
                raise
            else:
                self.connection.commit()

    def close(self) -> None:
        with self.lock:
            self.connection.close()
