import httpx
from fastapi import APIRouter, Depends, HTTPException

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["conversations"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    container: ApplicationContainer = Depends(get_container),
) -> ChatResponse:
    try:
        return await container.chat.answer(payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"模型接口调用失败：{exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
