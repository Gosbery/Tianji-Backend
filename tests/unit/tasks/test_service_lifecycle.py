from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import pytest

from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.integrations.llm import GenerationResult
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.conversations.service import ChatService
from bazi_api.modules.experts.schemas import ExpertProfile
from bazi_api.modules.observability.repository import TraceRepository
from bazi_api.modules.tasks.repository import TaskRepository
from bazi_api.modules.tasks.service import TaskService

BACKEND_ROOT = Path(__file__).resolve().parents[3]
CHILD_SERVICE = """
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.tasks.repository import TaskRepository
from bazi_api.modules.tasks.service import TaskService

async def main():
    database = SQLiteDatabase(Path(sys.argv[1]))
    class Chat:
        calls = 0
        async def answer_for_job(self, *args):
            self.calls += 1
            if sys.argv[2] == 'hold':
                print('running', flush=True)
            await asyncio.Event().wait()
    chat = Chat()
    service = TaskService(TaskRepository(database), ConversationRepository(database),
                          ChartCalculator(), SimpleNamespace(), chat, concurrency=1)
    try:
        try:
            await service.start()
        except RuntimeError as error:
            print(json.dumps({'started': False, 'error': str(error), 'calls': chat.calls}))
            return
        if sys.argv[2] == 'hold':
            await asyncio.Event().wait()
        else:
            await asyncio.sleep(0.05)
            print(json.dumps({'started': True, 'calls': chat.calls}))
    finally:
        await service.close()
        database.close()
asyncio.run(main())
"""


def child_environment() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(BACKEND_ROOT / "src")}


def attempt_child_start(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SERVICE, str(path), "attempt"],
        cwd=BACKEND_ROOT,
        env=child_environment(),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return json.loads(result.stdout)


def add_task(repository: TaskRepository) -> dict[str, object]:
    birth = BirthInput(date=date(1990, 1, 1), time=time(12))
    expert = ExpertProfile(
        id="test-expert",
        display_name="Test",
        version="test",
        review_status="reviewed",
        allowed_schools=["基础共识"],
        preferred_schools=["基础共识"],
        methodology=["test"],
        boundaries=["test"],
    )
    return repository.create(
        birth=birth,
        chart=ChartCalculator().calculate(birth),
        expert=expert,
        evidence_scope="reviewed_only",
        mode="hybrid",
    )


class WaitingChat:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    async def answer_for_job(self, *_: object) -> None:
        self.calls += 1
        self.started.set()
        await asyncio.Event().wait()


def build_service(database: SQLiteDatabase, chat: object | None = None) -> TaskService:
    return TaskService(
        repository=TaskRepository(database),
        conversations=ConversationRepository(database),
        charts=ChartCalculator(),
        experts=SimpleNamespace(),  # type: ignore[arg-type]
        chat=chat or WaitingChat(),  # type: ignore[arg-type]
        concurrency=1,
    )


@pytest.mark.asyncio
async def test_second_process_cannot_recover_or_execute_live_job(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    database = SQLiteDatabase(path)
    chat = WaitingChat()
    service = build_service(database, chat)
    task = add_task(service.repository)
    job = service.repository.enqueue(str(task["id"]), "Test question")
    try:
        await service.start()
        await asyncio.wait_for(chat.started.wait(), timeout=3)
        inode = service._execution_lock.path.stat().st_ino

        result = await asyncio.to_thread(attempt_child_start, path)

        assert result["started"] is False
        assert "only one API worker" in str(result["error"])
        assert result["calls"] == 0
        assert chat.calls == 1
        live = service.repository.get_job(str(job["id"]))
        assert live["status"] == "running"
        assert live["attempt_count"] == 1
        assert live["recovery_count"] == 0

        await service.close()
        replacement = await asyncio.to_thread(attempt_child_start, path)
        assert replacement["started"] is True
        assert service._execution_lock.path.stat().st_ino == inode
    finally:
        await service.close()
        database.close()


@pytest.mark.asyncio
async def test_start_failure_releases_execution_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    service = build_service(database)
    replacement = build_service(database)

    def fail_recovery() -> tuple[int, int]:
        raise RuntimeError("recovery failed")

    monkeypatch.setattr(service.repository, "recover_interrupted", fail_recovery)
    try:
        with pytest.raises(RuntimeError, match="recovery failed"):
            await service.start()
        await replacement.start()
        assert replacement._workers
    finally:
        await service.close()
        await replacement.close()
        database.close()


@pytest.mark.asyncio
async def test_cancelled_start_keeps_lock_until_recovery_thread_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    service = build_service(database)
    replacement = build_service(database)
    entered, release = threading.Event(), threading.Event()

    def slow_recovery() -> tuple[int, int]:
        entered.set()
        assert release.wait(timeout=5)
        return 0, 0

    monkeypatch.setattr(service.repository, "recover_interrupted", slow_recovery)
    startup = asyncio.create_task(service.start())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        startup.cancel()
        await asyncio.sleep(0)
        startup.cancel()
        with pytest.raises(RuntimeError, match="only one API worker"):
            await replacement.start()
        assert not startup.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(startup, timeout=3)
        await replacement.start()
    finally:
        release.set()
        await asyncio.gather(startup, return_exceptions=True)
        await service.close()
        await replacement.close()
        database.close()


class EmptyRetrieval:
    settings = SimpleNamespace(result_limit=6)
    model_version = "test"
    index_version = "test"

    async def search(self, **_: object) -> list[object]:
        return []


class FixedGenerator:
    async def generate(self, *_: object, **__: object) -> GenerationResult:
        return GenerationResult(answer="Test answer", uncertainties=[], followups=[])


@pytest.mark.asyncio
async def test_cancelled_close_waits_for_answer_persistence_before_unlocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    chat = ChatService(
        retrieval=EmptyRetrieval(),  # type: ignore[arg-type]
        generator=FixedGenerator(),  # type: ignore[arg-type]
        conversations=ConversationRepository(database),
        traces=TraceRepository(database),
        charts=ChartCalculator(),
    )
    service = build_service(database, chat)
    replacement = build_service(database)
    task = add_task(service.repository)
    job = service.repository.enqueue(str(task["id"]), "Test question")
    entered, release = threading.Event(), threading.Event()
    original_persist = chat._persist_answer

    def slow_persist(*args: object, **kwargs: object) -> str:
        entered.set()
        assert release.wait(timeout=5)
        return original_persist(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(chat, "_persist_answer", slow_persist)
    shutdown = None
    try:
        await service.start()
        assert await asyncio.to_thread(entered.wait, 3)
        shutdown = asyncio.create_task(service.close())
        await asyncio.sleep(0)
        shutdown.cancel()
        await asyncio.sleep(0)
        shutdown.cancel()
        with pytest.raises(RuntimeError, match="only one API worker"):
            await replacement.start()
        assert not shutdown.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(shutdown, timeout=3)
        completed = service.repository.get_job(str(job["id"]))
        assert completed["status"] == "succeeded"
        assert len(service.conversations.history(str(task["id"]))["messages"]) == 2
        await replacement.start()
        assert service.repository.get_job(str(job["id"]))["recovery_count"] == 0
    finally:
        release.set()
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        await service.close()
        await replacement.close()
        database.close()


@pytest.mark.asyncio
async def test_process_crash_releases_lock_and_preserves_job_recovery(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    database = SQLiteDatabase(path)
    chat = WaitingChat()
    service = build_service(database, chat)
    task = add_task(service.repository)
    job = service.repository.enqueue(str(task["id"]), "Test question")
    process = subprocess.Popen(
        [sys.executable, "-c", CHILD_SERVICE, str(path), "hold"],
        cwd=BACKEND_ROOT,
        env=child_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready = await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), timeout=5)
        assert ready.strip() == "running"
        process.kill()
        await asyncio.to_thread(process.wait, timeout=5)

        await service.start()
        await asyncio.wait_for(chat.started.wait(), timeout=3)

        recovered = service.repository.get_job(str(job["id"]))
        assert recovered["status"] == "running"
        assert recovered["attempt_count"] == 2
        assert recovered["recovery_count"] == 1
        assert chat.calls == 1
    finally:
        if process.poll() is None:
            process.kill()
        await asyncio.to_thread(process.communicate, timeout=5)
        await service.close()
        database.close()
