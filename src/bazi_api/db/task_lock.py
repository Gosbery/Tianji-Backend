from __future__ import annotations

import fcntl
import os
from pathlib import Path

from .sqlite import SQLiteDatabase


class TaskExecutionLock:
    def __init__(self, database: SQLiteDatabase) -> None:
        with database.read() as connection:
            filename = next(
                row["file"]
                for row in connection.execute("PRAGMA database_list")
                if row["name"] == "main"
            )
        if not filename:
            raise RuntimeError("Task execution requires a file-backed SQLite database")
        database_path = Path(filename).resolve()
        self.path = database_path.with_name(f"{database_path.name}.tasks.lock")
        self._descriptor: int | None = None

    def acquire(self) -> None:
        if self._descriptor is not None:
            raise RuntimeError("Task executor is already running")
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RuntimeError(
                "Another task executor is using this database; run only one API worker "
                f"per database ({self.path})"
            ) from exc
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            # Keep the lock file: unlinking allows another process to lock a different inode.
            os.close(descriptor)
