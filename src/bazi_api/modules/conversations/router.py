import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import UUID4

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import ChatRequest, ChatResponse, ConversationHistory

router = APIRouter(tags=["conversations"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    container: ApplicationContainer = Depends(get_container),
) -> ChatResponse:
    return await container.chat.answer(payload)


def _sse_event(event: dict[str, object]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    container: ApplicationContainer = Depends(get_container),
) -> StreamingResponse:
    await container.chat.validate_session_context(payload)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in container.chat.answer_stream(payload):
                yield _sse_event(event)
        except Exception:
            # Exceptions cannot be converted to an HTTP error after the stream
            # starts, so expose a safe, client-readable terminal event.
            logger.exception("chat_stream_failed")
            yield _sse_event({"type": "error", "message": "回答生成失败，请稍后重试。"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations/{session_id}", response_model=ConversationHistory)
async def conversation_history(
    session_id: UUID4,
    container: ApplicationContainer = Depends(get_container),
) -> ConversationHistory:
    history = await asyncio.to_thread(container.conversations.history, str(session_id))
    return ConversationHistory.model_validate(history)
