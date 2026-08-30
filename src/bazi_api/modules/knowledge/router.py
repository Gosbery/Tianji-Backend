from fastapi import APIRouter, Depends

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import KnowledgeCard

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/cards", response_model=list[KnowledgeCard])
async def list_knowledge_cards(
    container: ApplicationContainer = Depends(get_container),
) -> list[KnowledgeCard]:
    return container.knowledge.cards
