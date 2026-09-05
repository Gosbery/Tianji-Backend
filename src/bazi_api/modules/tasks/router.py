from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import (
    GenerationJob,
    TaskCreate,
    TaskDetail,
    TaskList,
    TaskMessageCreate,
    TaskRename,
    TaskSummary,
)

router = APIRouter(tags=["tasks"])


@router.post("/tasks", response_model=TaskDetail, status_code=201)
async def create_task(
    payload: TaskCreate,
    container: ApplicationContainer = Depends(get_container),
) -> TaskDetail:
    task = await asyncio.to_thread(
        container.tasks.create,
        payload.birth,
        payload.expert_id,
        payload.evidence_scope,
        payload.mode,
    )
    return TaskDetail.model_validate({**task, "messages": []})


@router.get("/tasks", response_model=TaskList)
async def list_tasks(
    container: ApplicationContainer = Depends(get_container),
) -> TaskList:
    tasks = await asyncio.to_thread(container.task_repository.list)
    return TaskList(tasks=[TaskSummary.model_validate(task) for task in tasks])


@router.get("/tasks/events")
async def task_events(
    container: ApplicationContainer = Depends(get_container),
) -> StreamingResponse:
    queue = container.tasks.events.subscribe()

    async def events() -> AsyncIterator[str]:
        try:
            yield "event: ready\ndata: {\"type\":\"ready\"}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"event: {event['type']}\ndata: {payload}\n\n"
                except TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            container.tasks.events.unsubscribe(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(
    task_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> TaskDetail:
    detail = await asyncio.to_thread(container.tasks.detail, task_id)
    return TaskDetail.model_validate(detail)


@router.patch("/tasks/{task_id}", response_model=TaskSummary)
async def rename_task(
    task_id: str,
    payload: TaskRename,
    container: ApplicationContainer = Depends(get_container),
) -> TaskSummary:
    task = await asyncio.to_thread(container.task_repository.rename, task_id, payload.title)
    return TaskSummary.model_validate(task)


@router.post("/tasks/{task_id}/archive", response_model=TaskSummary)
async def archive_task(
    task_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> TaskSummary:
    task = await asyncio.to_thread(container.task_repository.set_archived, task_id, True)
    return TaskSummary.model_validate(task)


@router.post("/tasks/{task_id}/restore", response_model=TaskSummary)
async def restore_task(
    task_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> TaskSummary:
    task = await asyncio.to_thread(container.task_repository.set_archived, task_id, False)
    return TaskSummary.model_validate(task)


@router.post("/tasks/{task_id}/messages", response_model=GenerationJob, status_code=202)
async def send_task_message(
    task_id: str,
    payload: TaskMessageCreate,
    container: ApplicationContainer = Depends(get_container),
) -> GenerationJob:
    return GenerationJob.model_validate(await container.tasks.enqueue(task_id, payload.question))


@router.post("/tasks/{task_id}/jobs/{job_id}/cancel", response_model=GenerationJob)
async def cancel_task_job(
    task_id: str,
    job_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> GenerationJob:
    return GenerationJob.model_validate(await container.tasks.cancel(task_id, job_id))


@router.post("/tasks/{task_id}/jobs/{job_id}/retry", response_model=GenerationJob)
async def retry_task_job(
    task_id: str,
    job_id: str,
    container: ApplicationContainer = Depends(get_container),
) -> GenerationJob:
    return GenerationJob.model_validate(await container.tasks.retry(task_id, job_id))
