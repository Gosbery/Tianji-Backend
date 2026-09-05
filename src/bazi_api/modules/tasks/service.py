from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import suppress
from typing import Any

from bazi_api.core.errors import ExpertNotFoundError
from bazi_api.modules.charts.schemas import BirthInput, ChartFacts
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.conversations.schemas import ChatRequest
from bazi_api.modules.conversations.service import ChatService
from bazi_api.modules.experts.repository import ExpertRepository

from .repository import TaskRepository

logger = logging.getLogger(__name__)


class TaskEventHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, object]]] = set()

    async def publish(self, event: dict[str, object]) -> None:
        for queue in tuple(self._subscribers):
            with suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue[dict[str, object]]:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, object]]) -> None:
        self._subscribers.discard(queue)


class TaskService:
    def __init__(
        self,
        repository: TaskRepository,
        conversations: ConversationRepository,
        charts: ChartCalculator,
        experts: ExpertRepository,
        chat: ChatService,
        concurrency: int = 3,
    ) -> None:
        self.repository = repository
        self.conversations = conversations
        self.charts = charts
        self.experts = experts
        self.chat = chat
        self.concurrency = concurrency
        self.events = TaskEventHub()
        self._workers: list[asyncio.Task[None]] = []
        self._wake = asyncio.Event()
        self._running: dict[str, asyncio.Task[Any]] = {}
        self._closing = False

    async def start(self) -> None:
        recovered, failed = await asyncio.to_thread(self.repository.recover_interrupted)
        if recovered or failed:
            logger.warning(
                "generation_jobs_recovered",
                extra={"hits": recovered, "failed": failed},
            )
        self._closing = False
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"task-worker-{index}")
            for index in range(self.concurrency)
        ]
        self._wake.set()

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        for worker in self._workers:
            worker.cancel()
        for job in tuple(self._running.values()):
            job.cancel()
        for worker in self._workers:
            with suppress(asyncio.CancelledError):
                await worker
        self._workers.clear()

    def create(
        self,
        birth: BirthInput,
        expert_id: str,
        evidence_scope: str,
        mode: str,
    ) -> dict[str, Any]:
        try:
            expert = self.experts.get(expert_id)
        except KeyError as exc:
            raise ExpertNotFoundError() from exc
        visible = {item.id for item in self.experts.list(evidence_scope)}
        if expert.id not in visible:
            raise ExpertNotFoundError()
        chart = self.charts.calculate(birth)
        return self.repository.create(
            birth=birth,
            chart=chart,
            expert=expert,
            evidence_scope=evidence_scope,
            mode=mode,
        )

    def detail(self, task_id: str) -> dict[str, Any]:
        task = self.repository.get(task_id)
        history = self.conversations.history(task_id)
        return {**task, "messages": history["messages"]}

    async def enqueue(self, task_id: str, question: str) -> dict[str, Any]:
        job = await asyncio.to_thread(self.repository.enqueue, task_id, question)
        await self.events.publish(self._event("queued", job))
        self._wake.set()
        return job

    async def cancel(self, task_id: str, job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(self.repository.get_job, job_id)
        if job["task_id"] != task_id:
            from bazi_api.core.errors import TaskNotFoundError

            raise TaskNotFoundError("生成任务不属于该任务")
        job = await asyncio.to_thread(self.repository.cancel, job_id)
        running = self._running.get(job_id)
        if running is not None:
            running.cancel()
        await self.events.publish(self._event("cancelled", job))
        return job

    async def retry(self, task_id: str, job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(self.repository.get_job, job_id)
        if job["task_id"] != task_id:
            from bazi_api.core.errors import TaskNotFoundError

            raise TaskNotFoundError("生成任务不属于该任务")
        job = await asyncio.to_thread(self.repository.retry, job_id)
        await self.events.publish(self._event("queued", job))
        self._wake.set()
        return job

    async def _worker(self, _: int) -> None:
        while not self._closing:
            job = await asyncio.to_thread(self.repository.claim_next)
            if job is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except TimeoutError:
                    pass
                continue
            current = asyncio.current_task()
            assert current is not None
            self._running[job["id"]] = current
            await self.events.publish(self._event("running", job))
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                if self._closing:
                    raise
                await asyncio.to_thread(self.repository.mark_cancelled, job["id"])
                cancelled = await asyncio.to_thread(self.repository.get_job, job["id"])
                await self.events.publish(self._event("cancelled", cancelled))
            except Exception:
                logger.exception("task_generation_failed", extra={"job_id": job["id"]})
                await asyncio.to_thread(
                    self.repository.mark_failed, job["id"], "回答生成失败，请稍后重试"
                )
                failed = await asyncio.to_thread(self.repository.get_job, job["id"])
                await self.events.publish(self._event("failed", failed))
            finally:
                self._running.pop(job["id"], None)

    async def _run_job(self, job: dict[str, Any]) -> None:
        task = await asyncio.to_thread(self.repository.get, job["task_id"])
        request = ChatRequest(
            chart=ChartFacts.model_validate(task["chart"]),
            question=job["question"],
            session_id=task["id"],
            expert_id=task["expert_id"],
            school=task["school"],
            mode=task["mode"],
            evidence_scope=task["evidence_scope"],
        )

        async def progress(message: str) -> None:
            await asyncio.to_thread(self.repository.update_progress, job["id"], message)
            updated = await asyncio.to_thread(self.repository.get_job, job["id"])
            await self.events.publish(self._event("progress", updated))

        def complete(connection: sqlite3.Connection, message_id: str) -> None:
            self.repository.complete_in_transaction(connection, job["id"], message_id)

        response = await self.chat.answer_for_job(request, progress, complete)
        completed = await asyncio.to_thread(self.repository.get_job, job["id"])
        await self.events.publish(
            {**self._event("succeeded", completed), "response": response.model_dump(mode="json")}
        )

    @staticmethod
    def _event(event_type: str, job: dict[str, Any]) -> dict[str, object]:
        return {
            "type": event_type,
            "task_id": job["task_id"],
            "job": job,
        }
