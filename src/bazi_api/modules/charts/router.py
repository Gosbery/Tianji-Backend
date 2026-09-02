from fastapi import APIRouter, Depends

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import BirthInput, ChartFacts

router = APIRouter(tags=["charts"])


@router.post("/chart", response_model=ChartFacts)
async def calculate_chart(
    payload: BirthInput,
    container: ApplicationContainer = Depends(get_container),
) -> ChartFacts:
    return container.charts.calculate(payload)
