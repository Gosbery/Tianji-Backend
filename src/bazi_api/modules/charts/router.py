from fastapi import APIRouter, Depends, HTTPException

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import BirthInput, ChartFacts

router = APIRouter(tags=["charts"])


@router.post("/chart", response_model=ChartFacts)
async def calculate_chart(
    payload: BirthInput,
    container: ApplicationContainer = Depends(get_container),
) -> ChartFacts:
    try:
        return container.charts.calculate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"排盘失败：{exc}") from exc
