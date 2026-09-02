from fastapi import APIRouter, Depends

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["conversations"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    container: ApplicationContainer = Depends(get_container),
) -> ChatResponse:
    return await container.chat.answer(payload)
