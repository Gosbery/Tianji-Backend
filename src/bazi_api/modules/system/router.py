from fastapi import APIRouter, Depends

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(
    container: ApplicationContainer = Depends(get_container),
) -> HealthResponse:
    overview = container.knowledge.overview()
    llm_configured = bool(
        container.settings.anthropic_auth_token
        if container.settings.llm_provider == "anthropic"
        else container.settings.openai_api_key
    )
    return HealthResponse(
        status="ok",
        cards=len(container.knowledge.cards),
        passages=len(container.knowledge.original_passages),
        knowledge_layers=overview.layers,
        retrieval_documents=overview.retrieval_documents,
        preview_retrieval_documents=overview.preview_documents,
        llm_configured=llm_configured,
        embedding_provider=container.settings.embedding_provider,
        embedding_model=container.retrieval.model_version,
        vector_backend=container.settings.vector_backend,
        index_version=container.retrieval.index_version,
        embedding_cache=container.retrieval.cache_stats,
        default_retrieval_mode="hybrid_rerank",
    )
