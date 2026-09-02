from fastapi import APIRouter, Depends

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import (
    CanonicalCorpus,
    KnowledgeCard,
    KnowledgeGraph,
    KnowledgeOverview,
    ModernAnnotation,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/overview", response_model=KnowledgeOverview)
async def get_knowledge_overview(
    container: ApplicationContainer = Depends(get_container),
) -> KnowledgeOverview:
    return container.knowledge.overview()


@router.get("/originals", response_model=CanonicalCorpus)
async def list_canonical_passages(
    container: ApplicationContainer = Depends(get_container),
) -> CanonicalCorpus:
    return CanonicalCorpus(
        works=container.knowledge.works,
        passages=container.knowledge.original_passages,
    )


@router.get("/annotations", response_model=list[ModernAnnotation])
async def list_modern_annotations(
    container: ApplicationContainer = Depends(get_container),
) -> list[ModernAnnotation]:
    return container.knowledge.annotations


@router.get("/cards", response_model=list[KnowledgeCard])
async def list_knowledge_cards(
    container: ApplicationContainer = Depends(get_container),
) -> list[KnowledgeCard]:
    return container.knowledge.cards


@router.get("/graph", response_model=KnowledgeGraph)
async def get_knowledge_graph(
    container: ApplicationContainer = Depends(get_container),
) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=container.knowledge.graph_nodes,
        edges=container.knowledge.graph_edges,
    )
