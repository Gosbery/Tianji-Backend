from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from bazi_api.core.errors import InvalidJobStateError, TaskBusyError, TaskNotFoundError
from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.modules.charts.schemas import BirthInput, ChartFacts
from bazi_api.modules.experts.schemas import ExpertProfile


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TaskRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def create(
        self,
        *,
        birth: BirthInput,
        chart: ChartFacts,
        expert: ExpertProfile,
        evidence_scope: str,
        mode: str,
    ) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        now = _now()
        school = expert.preferred_schools[0] if expert.preferred_schools else "基础共识"
        title = f"{birth.name or '访客'} · {expert.display_name}"
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sessions(
                    id, created_at, updated_at, chart_fingerprint, school, evidence_scope
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, now, now, "", school, evidence_scope),
            )
            connection.execute(
                """
                INSERT INTO tasks(
                    id, title, expert_id, expert_version, expert_name, birth_json,
                    chart_json, school, evidence_scope, mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    title,
                    expert.id,
                    expert.version,
                    expert.display_name,
                    json.dumps(birth.model_dump(mode="json"), ensure_ascii=False),
                    json.dumps(chart.model_dump(mode="json"), ensure_ascii=False),
                    school,
                    evidence_scope,
                    mode,
                    now,
                    now,
                ),
            )
        return self.get(task_id)

    def list(self) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY archived, updated_at DESC"
            ).fetchall()
            return [self._task_dict(connection, row) for row in rows]

    def get(self, task_id: str) -> dict[str, Any]:
        with self.database.read() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                raise TaskNotFoundError()
            return self._task_dict(connection, row)

    def rename(self, task_id: str, title: str) -> dict[str, Any]:
        now = _now()
        with self.database.transaction() as connection:
            changed = connection.execute(
                "UPDATE tasks SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip(), now, task_id),
            )
            if changed.rowcount != 1:
                raise TaskNotFoundError()
        return self.get(task_id)

    def set_archived(self, task_id: str, archived: bool) -> dict[str, Any]:
        now = _now()
        with self.database.transaction() as connection:
            changed = connection.execute(
                "UPDATE tasks SET archived = ?, updated_at = ? WHERE id = ?",
                (int(archived), now, task_id),
            )
            if changed.rowcount != 1:
                raise TaskNotFoundError()
        return self.get(task_id)

    def enqueue(self, task_id: str, question: str) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        now = _now()
        try:
            with self.database.transaction() as connection:
                task = connection.execute(
                    "SELECT id FROM tasks WHERE id = ? AND archived = 0", (task_id,)
                ).fetchone()
                if task is None:
                    raise TaskNotFoundError()
                connection.execute(
                    """
                    INSERT INTO generation_jobs(
                        id, task_id, question, status, progress, created_at, updated_at
                    ) VALUES (?, ?, ?, 'queued', '等待生成', ?, ?)
                    """,
                    (job_id, task_id, question, now, now),
                )
                connection.execute(
                    "UPDATE tasks SET updated_at = ? WHERE id = ?", (now, task_id)
                )
        except sqlite3.IntegrityError as exc:
            if "idx_generation_jobs_active_task" in str(exc) or "UNIQUE" in str(exc):
                raise TaskBusyError() from exc
            raise
        return self.get_job(job_id)

    def claim_next(self) -> dict[str, Any] | None:
        now = _now()
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT id FROM generation_jobs
                WHERE status = 'queued' ORDER BY created_at LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            changed = connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'running', progress = '正在启动分析', started_at = ?,
                    updated_at = ?, attempt_count = attempt_count + 1
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, row["id"]),
            )
            if changed.rowcount != 1:
                return None
            return self._job_dict(
                connection.execute(
                    "SELECT * FROM generation_jobs WHERE id = ?", (row["id"],)
                ).fetchone()
            )

    def recover_interrupted(self) -> tuple[int, int]:
        now = _now()
        with self.database.transaction() as connection:
            failed = connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'failed', error = '服务重启后已达到最大恢复次数',
                    finished_at = ?, updated_at = ?
                WHERE status = 'running' AND recovery_count >= 1
                """,
                (now, now),
            ).rowcount
            recovered = connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'queued', progress = '服务重启，等待恢复', started_at = NULL,
                    updated_at = ?, recovery_count = recovery_count + 1
                WHERE status = 'running'
                """,
                (now,),
            ).rowcount
        return recovered, failed

    def update_progress(self, job_id: str, progress: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE generation_jobs SET progress = ?, updated_at = ? WHERE id = ?",
                (progress, _now(), job_id),
            )

    def complete_in_transaction(
        self, connection: sqlite3.Connection, job_id: str, assistant_message_id: str
    ) -> None:
        now = _now()
        row = connection.execute(
            "SELECT task_id, cancel_requested FROM generation_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise InvalidJobStateError()
        if row["cancel_requested"]:
            raise InvalidJobStateError("生成已取消")
        changed = connection.execute(
            """
            UPDATE generation_jobs
            SET status = 'succeeded', progress = '分析完成', assistant_message_id = ?,
                finished_at = ?, updated_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (assistant_message_id, now, now, job_id),
        )
        if changed.rowcount != 1:
            raise InvalidJobStateError()
        connection.execute(
            "UPDATE tasks SET updated_at = ? WHERE id = ?", (now, row["task_id"])
        )

    def mark_failed(self, job_id: str, error: str) -> None:
        now = _now()
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'failed', progress = '', error = ?, finished_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (error[:1000], now, now, job_id),
            )

    def cancel(self, job_id: str) -> dict[str, Any]:
        now = _now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise TaskNotFoundError("生成任务不存在")
            if row["status"] not in {"queued", "running"}:
                raise InvalidJobStateError()
            connection.execute(
                """
                UPDATE generation_jobs
                SET status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                    cancel_requested = 1, progress = '正在取消', updated_at = ?,
                    finished_at = CASE WHEN status = 'queued' THEN ? ELSE finished_at END
                WHERE id = ?
                """,
                (now, now, job_id),
            )
        return self.get_job(job_id)

    def mark_cancelled(self, job_id: str) -> None:
        now = _now()
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE generation_jobs SET status = 'cancelled', progress = '', error = '',
                    finished_at = ?, updated_at = ? WHERE id = ?
                """,
                (now, now, job_id),
            )

    def retry(self, job_id: str) -> dict[str, Any]:
        now = _now()
        try:
            with self.database.transaction() as connection:
                changed = connection.execute(
                    """
                    UPDATE generation_jobs
                    SET status = 'queued', progress = '等待重试', error = '', cancel_requested = 0,
                        started_at = NULL, finished_at = NULL, updated_at = ?
                    WHERE id = ? AND status IN ('failed', 'cancelled')
                    """,
                    (now, job_id),
                )
                if changed.rowcount != 1:
                    raise InvalidJobStateError()
        except sqlite3.IntegrityError as exc:
            raise TaskBusyError() from exc
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT * FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise TaskNotFoundError("生成任务不存在")
            return self._job_dict(row)

    def task_for_job(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        task = self.get(job["task_id"])
        return {**task, "job": job}

    @classmethod
    def _task_dict(cls, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        job = connection.execute(
            """
            SELECT * FROM generation_jobs WHERE task_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (row["id"],),
        ).fetchone()
        active = cls._job_dict(job) if job is not None else None
        return {
            "id": row["id"],
            "title": row["title"],
            "expert_id": row["expert_id"],
            "expert_version": row["expert_version"],
            "expert_name": row["expert_name"],
            "birth": json.loads(row["birth_json"]),
            "chart": json.loads(row["chart_json"]),
            "school": row["school"],
            "evidence_scope": row["evidence_scope"],
            "mode": row["mode"],
            "archived": bool(row["archived"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "active_job": active,
        }

    @staticmethod
    def _job_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "id", "task_id", "question", "status", "progress", "error",
                "assistant_message_id", "attempt_count", "recovery_count", "created_at",
                "started_at", "finished_at", "updated_at",
            )
        }
