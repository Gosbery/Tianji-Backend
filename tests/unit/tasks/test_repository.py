from __future__ import annotations

import asyncio
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import pytest

from bazi_api.core.errors import TaskBusyError
from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.experts.schemas import ExpertProfile
from bazi_api.modules.tasks.repository import TaskRepository
from bazi_api.modules.tasks.service import TaskService


def expert() -> ExpertProfile:
    return ExpertProfile(
        id="liang-xiangrun",
        display_name="梁湘润方法",
        version="test",
        review_status="machine_verified",
        allowed_schools=["基础共识", "子平格局法"],
        preferred_schools=["子平格局法"],
        methodology=["月令与格局"],
        boundaries=["不冒充人物"],
        is_default=True,
    )


def create_task(repository: TaskRepository) -> dict[str, object]:
    birth = BirthInput(date=date(1990, 1, 1), time=time(12), name="测试")
    return repository.create(
        birth=birth,
        chart=ChartCalculator().calculate(birth),
        expert=expert(),
        evidence_scope="personal_preview",
        mode="hybrid_rerank",
    )


def test_task_context_is_immutable_and_duplicate_generation_is_rejected(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    repository = TaskRepository(database)
    try:
        task = create_task(repository)
        repository.rename(str(task["id"]), "新标题")
        restored = repository.get(str(task["id"]))

        assert restored["title"] == "新标题"
        assert restored["expert_id"] == "liang-xiangrun"
        assert restored["birth"] == task["birth"]
        assert restored["chart"] == task["chart"]

        repository.enqueue(str(task["id"]), "第一个问题")
        with pytest.raises(TaskBusyError):
            repository.enqueue(str(task["id"]), "第二个问题")
    finally:
        database.close()


def test_interrupted_job_recovers_only_once(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    repository = TaskRepository(database)
    try:
        task = create_task(repository)
        job = repository.enqueue(str(task["id"]), "恢复测试")
        assert repository.claim_next()["id"] == job["id"]  # type: ignore[index]
        assert repository.recover_interrupted() == (1, 0)
        assert repository.get_job(str(job["id"]))["recovery_count"] == 1

        repository.claim_next()
        assert repository.recover_interrupted() == (0, 1)
        assert repository.get_job(str(job["id"]))["status"] == "failed"
    finally:
        database.close()


class BlockingChat:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0
        self.three_started = asyncio.Event()
        self.release = asyncio.Event()

    async def answer_for_job(self, *_: object) -> None:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        if self.active == 3:
            self.three_started.set()
        try:
            await self.release.wait()
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_three_jobs_run_and_fourth_remains_queued(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "app.db")
    repository = TaskRepository(database)
    chat = BlockingChat()
    service = TaskService(
        repository=repository,
        conversations=ConversationRepository(database),
        charts=ChartCalculator(),
        experts=SimpleNamespace(),  # type: ignore[arg-type]
        chat=chat,  # type: ignore[arg-type]
        concurrency=3,
    )
    try:
        jobs = []
        for index in range(4):
            task = create_task(repository)
            jobs.append(repository.enqueue(str(task["id"]), f"问题 {index}"))
        await service.start()
        await asyncio.wait_for(chat.three_started.wait(), timeout=2)

        statuses = [repository.get_job(str(job["id"]))["status"] for job in jobs]
        assert statuses.count("running") == 3
        assert statuses.count("queued") == 1
        assert chat.maximum == 3
    finally:
        await service.close()
        database.close()
