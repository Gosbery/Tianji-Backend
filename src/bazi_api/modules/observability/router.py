from typing import Any

from fastapi import APIRouter, Depends, Query

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/recent")
async def recent_traces(
    limit: int = Query(default=20, ge=1, le=100),
    container: ApplicationContainer = Depends(get_container),
) -> list[dict[str, Any]]:
    return container.traces.recent(limit)
