from typing import Any

from fastapi import APIRouter, Depends

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(
    container: ApplicationContainer = Depends(get_container),
) -> dict[str, Any]:
    return {
        "status": "ok",
        "cards": len(container.knowledge.cards),
        "passages": len(container.knowledge.passages),
        "llm_configured": bool(container.settings.openai_api_key),
        "embedding_provider": container.settings.embedding_provider,
        "vector_backend": container.settings.vector_backend,
    }
