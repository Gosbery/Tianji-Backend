import asyncio
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import RetrievalTrace

router = APIRouter(prefix="/observability", tags=["observability"])
bearer = HTTPBearer(auto_error=False)


def require_observability_access(
    container: ApplicationContainer = Depends(get_container),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer),
) -> None:
    expected = container.settings.observability_api_key
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not secrets.compare_digest(credentials.credentials, expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要内部观测访问凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get(
    "/recent",
    response_model=list[RetrievalTrace],
    dependencies=[Depends(require_observability_access)],
)
async def recent_traces(
    limit: int = Query(default=20, ge=1, le=100),
    container: ApplicationContainer = Depends(get_container),
) -> list[RetrievalTrace]:
    traces = await asyncio.to_thread(container.traces.recent, limit)
    return [RetrievalTrace.model_validate(item) for item in traces]
