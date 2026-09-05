from fastapi import APIRouter, Depends, Query

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container
from bazi_api.modules.knowledge.schemas import EvidenceScope

from .schemas import ExpertList

router = APIRouter(tags=["experts"])


@router.get("/experts", response_model=ExpertList)
async def list_experts(
    scope: EvidenceScope = Query(default="reviewed_only"),
    container: ApplicationContainer = Depends(get_container),
) -> ExpertList:
    return ExpertList(
        experts=container.experts.list(scope),
        default_expert_id=container.experts.default_expert_id,
    )
